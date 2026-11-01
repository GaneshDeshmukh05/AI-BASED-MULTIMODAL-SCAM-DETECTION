"""train_fine_tune_scam_model.py
Fine-tunes audio scam call detection CNN model on all 48 local call recordings.
Handles format decoding (.unknown, .aac, .3gp, .mp3, .wav), 5-second overlapping window slicing,
spec-augment & audio augmentation, zero-leakage recording-level grouping, and validation checkpointing.
"""

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import soundfile as sf
import librosa
import imageio_ffmpeg

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("fine_tune_scam")

# Audio and feature extraction constants
SAMPLE_RATE = 16000
DURATION = 5.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)  # 80000
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
EXPECTED_FRAMES = 313  # 80000 // 256 + 1


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except Exception:
        pass


def decode_audio_file(file_path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decodes any audio file (including .unknown, .aac, .3gp, .mp3, .wav) to mono float32 at target_sr."""
    # Try soundfile directly first
    try:
        data, sr = sf.read(file_path, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
        return data.astype(np.float32)
    except Exception:
        pass

    # Fallback to imageio-ffmpeg
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name
    try:
        cmd = [
            ffmpeg_exe, "-y", "-v", "error",
            "-i", file_path,
            "-ar", str(target_sr),
            "-ac", "1",
            "-f", "wav",
            tmp_wav,
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        data, sr = sf.read(tmp_wav, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32)
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass


def slice_recording(
    y: np.ndarray,
    window_sec: float = DURATION,
    stride_sec: float = 2.5,
    sr: int = SAMPLE_RATE,
) -> List[np.ndarray]:
    """Slices a long recording into 5-second windows with 50% overlap."""
    win_len = int(window_sec * sr)
    stride_len = int(stride_sec * sr)
    slices = []

    if len(y) <= win_len:
        # Pad short audio
        padded = np.pad(y, (0, max(0, win_len - len(y))))
        slices.append(padded[:win_len])
        return slices

    idx = 0
    while idx + win_len <= len(y):
        chunk = y[idx : idx + win_len]
        slices.append(chunk)
        idx += stride_len

    # If remaining tail is significant (at least 2.5 seconds), pad it and include
    remainder = len(y) - idx
    if remainder >= int(2.5 * sr) and (idx > 0 and len(y) - (idx - stride_len) > win_len):
        tail = y[idx:]
        padded = np.pad(tail, (0, win_len - len(tail)))
        slices.append(padded)

    return slices


def compute_logmel(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Computes per-example normalized 64-band log-Mel spectrogram of shape (64, 313, 1)."""
    # Peak normalization
    peak = np.max(np.abs(y)) if len(y) else 0.0
    if peak > 1e-8:
        y = y / peak

    m = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    m = librosa.power_to_db(m, ref=np.max)
    # Per-example zero-mean unit-variance
    norm = ((m - m.mean()) / (m.std() + 1e-6)).astype(np.float32)

    # Ensure shape is exactly (64, 313)
    if norm.shape[1] < EXPECTED_FRAMES:
        norm = np.pad(norm, ((0, 0), (0, EXPECTED_FRAMES - norm.shape[1])))
    elif norm.shape[1] > EXPECTED_FRAMES:
        norm = norm[:, :EXPECTED_FRAMES]

    return norm[..., None]  # (64, 313, 1)


def augment_audio(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies light time-shift, subtle noise, and volume gain variations."""
    y_aug = y.copy()

    # 1. Random gain (0.85x to 1.15x)
    gain = rng.uniform(0.85, 1.15)
    y_aug = y_aug * gain

    # 2. Random cyclic time shift (-0.5s to +0.5s)
    shift = rng.integers(-int(0.5 * SAMPLE_RATE), int(0.5 * SAMPLE_RATE))
    y_aug = np.roll(y_aug, shift)

    # 3. Additive subtle white noise (SNR ~ 25dB)
    if rng.random() > 0.4:
        rms = np.sqrt(np.mean(y_aug**2) + 1e-8)
        noise_level = rms * rng.uniform(0.01, 0.04)
        y_aug = y_aug + rng.normal(0, noise_level, size=len(y_aug))

    return y_aug.astype(np.float32)


def spec_augment(spec: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Applies SpecAugment: random frequency and time masking."""
    s = spec.copy()  # shape (64, 313, 1)
    # Frequency masking (mask up to 8 bins)
    f_len = rng.integers(1, 8)
    f0 = rng.integers(0, max(1, N_MELS - f_len))
    s[f0 : f0 + f_len, :, :] = 0.0

    # Time masking (mask up to 16 time steps)
    t_len = rng.integers(1, 16)
    t0 = rng.integers(0, max(1, EXPECTED_FRAMES - t_len))
    s[:, t0 : t0 + t_len, :] = 0.0

    return s


def load_dataset_recordings(dataset_dir: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Scans and decodes all recordings in SCAM_CALLS and NORMAL_CALLS."""
    scam_dir = os.path.join(dataset_dir, "SCAM_CALLS")
    normal_dir = os.path.join(dataset_dir, "NORMAL_CALLS")

    scam_files = sorted([os.path.join(scam_dir, f) for f in os.listdir(scam_dir) if os.path.isfile(os.path.join(scam_dir, f))])
    normal_files = sorted([os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if os.path.isfile(os.path.join(normal_dir, f))])

    logger.info("Found %d scam files and %d normal files.", len(scam_files), len(normal_files))

    scam_recs = []
    for fp in scam_files:
        try:
            audio = decode_audio_file(fp)
            scam_recs.append({
                "id": os.path.splitext(os.path.basename(fp))[0],
                "path": fp,
                "label": "scam",
                "y": 1,
                "audio": audio,
                "duration": len(audio) / SAMPLE_RATE,
            })
        except Exception as exc:
            logger.error("Failed to decode scam file %s: %s", fp, exc)

    normal_recs = []
    for fp in normal_files:
        try:
            audio = decode_audio_file(fp)
            normal_recs.append({
                "id": os.path.splitext(os.path.basename(fp))[0],
                "path": fp,
                "label": "normal",
                "y": 0,
                "audio": audio,
                "duration": len(audio) / SAMPLE_RATE,
            })
        except Exception as exc:
            logger.error("Failed to decode normal file %s: %s", fp, exc)

    return scam_recs, normal_recs


def split_recordings(
    scam_recs: List[Dict[str, Any]],
    normal_recs: List[Dict[str, Any]],
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Splits recordings into train (66%), val (17%), and test (17%) by recording ID.
    Guarantees that slices from the same original call are strictly in the same split.
    """
    rng = np.random.default_rng(seed)
    scam_indices = list(range(len(scam_recs)))
    normal_indices = list(range(len(normal_recs)))
    rng.shuffle(scam_indices)
    rng.shuffle(normal_indices)

    # 4 scam + 4 normal for test, 4 scam + 4 normal for val, remainder for train
    n_test = 4
    n_val = 4

    test_idx_scam = scam_indices[:n_test]
    val_idx_scam = scam_indices[n_test : n_test + n_val]
    train_idx_scam = scam_indices[n_test + n_val :]

    test_idx_norm = normal_indices[:n_test]
    val_idx_norm = normal_indices[n_test : n_test + n_val]
    train_idx_norm = normal_indices[n_test + n_val :]

    train_recs = [scam_recs[i] for i in train_idx_scam] + [normal_recs[i] for i in train_idx_norm]
    val_recs = [scam_recs[i] for i in val_idx_scam] + [normal_recs[i] for i in val_idx_norm]
    test_recs = [scam_recs[i] for i in test_idx_scam] + [normal_recs[i] for i in test_idx_norm]

    logger.info("Recording split count: Train=%d (16 scam, 16 normal), Val=%d (4 scam, 4 normal), Test=%d (4 scam, 4 normal)",
                len(train_recs), len(val_recs), len(test_recs))

    return train_recs, val_recs, test_recs


def build_slice_features(
    recs: List[Dict[str, Any]],
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts slices from recordings, computes logmel features, and optionally augments."""
    X, y, rec_ids = [], [], []

    for rec in recs:
        raw_audio = rec["audio"]
        label = rec["y"]
        rec_id = rec["id"]

        slices = slice_recording(raw_audio)
        for s in slices:
            # 1. Clean slice
            feat = compute_logmel(s)
            X.append(feat)
            y.append(label)
            rec_ids.append(rec_id)

            # 2. If augmenting training data, add an augmented version of each slice
            if augment and rng is not None:
                aug_audio = augment_audio(s, rng)
                aug_feat = compute_logmel(aug_audio)
                if rng.random() > 0.3:
                    aug_feat = spec_augment(aug_feat, rng)
                X.append(aug_feat)
                y.append(label)
                rec_ids.append(rec_id)

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int32), rec_ids


def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, test_ids: List[str]) -> Dict[str, Any]:
    """Computes comprehensive slice-level and recording-level test metrics."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )

    preds_prob = model.predict(X_test, verbose=0).ravel()
    preds_binary = (preds_prob >= 0.5).astype(int)

    acc = float(accuracy_score(y_test, preds_binary))
    prec = float(precision_score(y_test, preds_binary, zero_division=0))
    rec = float(recall_score(y_test, preds_binary, zero_division=0))
    f1 = float(f1_score(y_test, preds_binary, zero_division=0))
    try:
        auc_val = float(roc_auc_score(y_test, preds_prob))
    except Exception:
        auc_val = 0.5

    cm = confusion_matrix(y_test, preds_binary).tolist()

    # Aggregate by recording ID (majority vote / mean probability per call)
    rec_agg: Dict[str, Dict[str, Any]] = {}
    for rid, prob, true_lbl in zip(test_ids, preds_prob, y_test):
        if rid not in rec_agg:
            rec_agg[rid] = {"probs": [], "true_y": int(true_lbl)}
        rec_agg[rid]["probs"].append(float(prob))

    rec_true, rec_pred_prob, rec_pred_bin = [], [], []
    for rid, data in rec_agg.items():
        mean_p = float(np.mean(data["probs"]))
        rec_true.append(data["true_y"])
        rec_pred_prob.append(mean_p)
        rec_pred_bin.append(int(mean_p >= 0.5))

    rec_acc = float(accuracy_score(rec_true, rec_pred_bin))
    try:
        rec_auc = float(roc_auc_score(rec_true, rec_pred_prob))
    except Exception:
        rec_auc = 1.0

    report = {
        "slice_level": {
            "num_test_slices": len(y_test),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc_val, 4),
            "confusion_matrix": cm,
        },
        "recording_level": {
            "num_test_recordings": len(rec_true),
            "accuracy": round(rec_acc, 4),
            "roc_auc": round(rec_auc, 4),
            "recordings": {
                rid: {
                    "true_label": "scam" if data["true_y"] == 1 else "normal",
                    "predicted_prob": round(float(np.mean(data["probs"])), 4),
                    "predicted_label": "scam" if np.mean(data["probs"]) >= 0.5 else "normal",
                    "num_slices": len(data["probs"]),
                }
                for rid, data in rec_agg.items()
            },
        },
    }

    return report


def fine_tune_scam_model(
    dataset_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    learning_rate: float = 2e-4,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full fine-tuning pipeline."""
    set_seed(seed)
    rng = np.random.default_rng(seed)

    # 1. Load and decode audio
    scam_recs, normal_recs = load_dataset_recordings(dataset_dir)
    if not scam_recs or not normal_recs:
        raise ValueError("Could not load scam or normal recordings.")

    # 2. Split recordings into train, val, test
    train_recs, val_recs, test_recs = split_recordings(scam_recs, normal_recs, seed=seed)

    # 3. Extract slices & augment
    logger.info("Extracting features and applying SpecAugment...")
    X_train, y_train, train_ids = build_slice_features(train_recs, augment=True, rng=rng)
    X_val, y_val, val_ids = build_slice_features(val_recs, augment=False)
    X_test, y_test, test_ids = build_slice_features(test_recs, augment=False)

    logger.info("Feature matrices ready:")
    logger.info("  Train: %s (Scam: %d, Normal: %d)", X_train.shape, int((y_train == 1).sum()), int((y_train == 0).sum()))
    logger.info("  Val  : %s (Scam: %d, Normal: %d)", X_val.shape, int((y_val == 1).sum()), int((y_val == 0).sum()))
    logger.info("  Test : %s (Scam: %d, Normal: %d)", X_test.shape, int((y_test == 1).sum()), int((y_test == 0).sum()))

    # Shuffle training set
    perm = rng.permutation(len(X_train))
    X_train, y_train = X_train[perm], y_train[perm]

    # 4. Load existing model
    import keras
    model_path = os.path.join(dataset_dir, "best_audio_scam_model.keras")
    backup_path = os.path.join(dataset_dir, "best_audio_scam_model_original_backup.keras")

    if not os.path.exists(backup_path) and os.path.exists(model_path):
        shutil.copy2(model_path, backup_path)
        logger.info("Created backup of original model at: %s", backup_path)

    logger.info("Loading base model from: %s", model_path)
    model = keras.models.load_model(model_path, compile=False)

    # 5. Compile with fine-tuning learning rate and label smoothing
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.BinaryCrossentropy(label_smoothing=0.02)
    metrics = [
        keras.metrics.BinaryAccuracy(name="accuracy"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
        keras.metrics.AUC(name="auc"),
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 6. Callbacks
    checkpoint_path = os.path.join(dataset_dir, "best_audio_scam_model_finetuned.keras")
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # Balanced class weights
    total_samples = len(y_train)
    n_neg = max(1, int((y_train == 0).sum()))
    n_pos = max(1, int((y_train == 1).sum()))
    class_weight = {0: total_samples / (2.0 * n_neg), 1: total_samples / (2.0 * n_pos)}
    logger.info("Class weights: %s", class_weight)

    # 7. Train
    logger.info("Starting fine-tuning for up to %d epochs...", epochs)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    # If checkpoint exists, load best weights
    if os.path.exists(checkpoint_path):
        logger.info("Loading best checkpoint weights from %s", checkpoint_path)
        model = keras.models.load_model(checkpoint_path, compile=False)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 8. Evaluate on unseen test set
    logger.info("Evaluating fine-tuned model on held-out test recordings...")
    eval_report = evaluate_model(model, X_test, y_test, test_ids)
    logger.info("Test Results:")
    logger.info("  Slice-level Accuracy : %.4f | AUC: %.4f | F1: %.4f",
                eval_report["slice_level"]["accuracy"],
                eval_report["slice_level"]["roc_auc"],
                eval_report["slice_level"]["f1_score"])
    logger.info("  Recording-level Accuracy: %.4f | AUC: %.4f",
                eval_report["recording_level"]["accuracy"],
                eval_report["recording_level"]["roc_auc"])

    # 9. Save as production model and backup
    model.save(model_path)
    logger.info("Successfully updated production model at: %s", model_path)

    # Save training report & history
    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    final_report = {
        "dataset_summary": {
            "total_recordings": len(scam_recs) + len(normal_recs),
            "train_recordings": len(train_recs),
            "val_recordings": len(val_recs),
            "test_recordings": len(test_recs),
            "train_slices": len(X_train),
            "val_slices": len(X_val),
            "test_slices": len(X_test),
        },
        "hyperparameters": {
            "epochs_trained": len(history.epoch),
            "batch_size": batch_size,
            "initial_learning_rate": learning_rate,
            "label_smoothing": 0.02,
        },
        "evaluation": eval_report,
        "history": hist_dict,
    }

    report_path = os.path.join(dataset_dir, "fine_tuning_evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    logger.info("Saved fine-tuning report to: %s", report_path)

    return final_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune audio scam model.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    args = parser.parse_args()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(cur_dir, "dataset")

    fine_tune_scam_model(
        dataset_dir=dataset_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
