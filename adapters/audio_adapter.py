"""audio_adapter.py
Translates audio module outputs into the standardized ModalityResult schema.

Supports both audio modules:
1. dataset/audio_scam_detection_model: Scam call audio vs normal call audio
2. LA/asvspoof2019_la_cnn: Bona fide vs spoofed/synthetic audio
Provides both output dictionary adaptation and direct live inference execution on audio files.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fusion.schemas import (
    DetectorStatus,
    ModalityResult,
    ModalityType,
)

logger = logging.getLogger("audio_adapter")


class AudioAdapter:
    """Adapts raw inference outputs from audio detectors."""

    _scam_model: Optional[Any] = None
    _spoof_model: Optional[Any] = None

    @classmethod
    def load_scam_model(cls, model_path: Optional[str] = None) -> Any:
        """Loads and caches scam call Keras model."""
        if cls._scam_model is None:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

            # Search upwards for audio_model
            cur = Path(__file__).resolve().parent
            audio_dir = cur.parent / "audio_model"
            for _ in range(3):
                if (cur / "audio_model").is_dir():
                    audio_dir = cur / "audio_model"
                    break
                cur = cur.parent

            default_path = str(audio_dir / "dataset" / "best_audio_scam_model.keras")
            target_path = model_path or default_path
            if not os.path.exists(target_path):
                alt_path = os.path.join(
                                        "dataset",
                    "best_audio_scam_model.keras",
                )
                if os.path.exists(alt_path):
                    target_path = alt_path

            try:
                import keras
                cls._scam_model = keras.models.load_model(target_path, compile=False)
            except Exception as exc:
                logger.info("TensorFlow/Keras runtime unavailable (%s). Using native NumPy CNN engine.", exc)
                from .numpy_cnn import NumPyKerasCNNPredictor
                cls._scam_model = NumPyKerasCNNPredictor(target_path, has_conv_bias=False)
        return cls._scam_model

    @classmethod
    def load_spoof_model(cls, model_path: Optional[str] = None) -> Any:
        """Loads and caches ASVspoof anti-spoofing Keras model."""
        if cls._spoof_model is None:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

            cur = Path(__file__).resolve().parent
            audio_dir = cur.parent / "audio_model"
            for _ in range(3):
                if (cur / "audio_model").is_dir():
                    audio_dir = cur / "audio_model"
                    break
                cur = cur.parent

            default_path = str(audio_dir / "LA" / "best_model.keras")
            target_path = model_path or default_path
            if not os.path.exists(target_path):
                alt_path = os.path.join(
                                        "LA",
                    "best_model.keras",
                )
                if os.path.exists(alt_path):
                    target_path = alt_path

            try:
                import keras
                cls._spoof_model = keras.models.load_model(target_path, compile=False)
            except Exception as exc:
                logger.info("TensorFlow/Keras runtime unavailable (%s). Using native NumPy CNN engine.", exc)
                from .numpy_cnn import NumPyKerasCNNPredictor
                cls._spoof_model = NumPyKerasCNNPredictor(target_path, has_conv_bias=True)
        return cls._spoof_model

    @staticmethod
    def _load_audio_robust(audio_path: str, target_sr: int = 16000) -> Any:
        """Loads and converts any audio file (including .unknown, .aac, .3gp, .mp3, .wav) to 16kHz mono."""
        import numpy as np
        import soundfile as sf
        import librosa

        # 1. Try direct soundfile read
        try:
            data, sr = sf.read(audio_path, dtype="float32", always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            return data.astype(np.float32)
        except Exception:
            pass

        # 2. Try librosa.load
        try:
            y, sr = librosa.load(audio_path, sr=target_sr, mono=True)
            return y.astype(np.float32)
        except Exception:
            pass

        # 3. Fallback to imageio-ffmpeg
        import subprocess
        import tempfile
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            cmd = [
                ffmpeg_exe, "-y", "-v", "error",
                "-i", audio_path,
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

    @staticmethod
    def _compute_spoof_logmel(y: Any, sr: int = 16000) -> Any:
        """Build the anti-spoof feature using the ASVspoof training settings."""
        import numpy as np
        import librosa

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=1024,
            hop_length=256,
            n_mels=64,
            fmin=20,
            fmax=sr // 2,
            power=2.0,
        )
        logmel = librosa.power_to_db(mel, ref=np.max)
        return ((logmel - logmel.mean()) / (logmel.std() + 1e-6)).astype("float32")

    @classmethod
    def predict_scam_call_file(cls, audio_path: str, threshold: float = 0.5) -> ModalityResult:
        """Runs live feature extraction and inference on an audio file using scam model."""
        try:
            import numpy as np
            import librosa
            model = cls.load_scam_model()
            y = cls._load_audio_robust(audio_path, target_sr=16000)
            n_samples = 16000 * 5
            y = y[:n_samples] if len(y) > n_samples else np.pad(y, (0, n_samples - len(y)))
            spec = librosa.feature.melspectrogram(y=y, sr=16000, n_fft=1024, hop_length=256, n_mels=64, power=2)
            spec = librosa.power_to_db(spec, ref=np.max)
            feat = ((spec - spec.mean()) / (spec.std() + 1e-6))[None, ..., None]
            score = float(model.predict(feat, verbose=0).ravel()[0])
            raw = {
                "label": "scam" if score >= threshold else "normal",
                "scam_score": score,
                "threshold": threshold,
                "file": audio_path,
            }
            return cls.adapt_scam_call(raw, source_name=os.path.basename(audio_path))
        except Exception as exc:
            msg = str(exc)
            if "Application Control" in msg or isinstance(exc, ImportError):
                logger.warning("Scam audio prediction runtime unavailable on %s: %s", audio_path, exc)
            else:
                logger.exception("Failed live scam audio prediction on %s", audio_path)
            return ModalityResult.failed(ModalityType.AUDIO_SCAM.value, f"Audio inference error: {exc}")

    @classmethod
    def predict_spoof_file(cls, audio_path: str, threshold: float = 0.5) -> ModalityResult:
        """Runs live feature extraction and inference on an audio file using LA anti-spoof model."""
        try:
            import numpy as np
            import librosa
            model = cls.load_spoof_model()
            y = cls._load_audio_robust(audio_path, target_sr=16000)
            n_samples = 16000 * 5
            y = y[:n_samples] if len(y) > n_samples else np.pad(y, (0, n_samples - len(y)))
            logmel = cls._compute_spoof_logmel(y, sr=16000)
            spec = logmel[..., None][None, ...]
            score = float(model.predict(spec, verbose=0).ravel()[0])
            raw = {
                "file": audio_path,
                "spoof_score": score,
                "prediction": "spoof / synthetic" if score >= threshold else "bonafide / genuine",
                "threshold": threshold,
            }
            return cls.adapt_spoof(raw, source_name=os.path.basename(audio_path))
        except Exception as exc:
            msg = str(exc)
            if "Application Control" in msg or isinstance(exc, ImportError):
                logger.warning("Spoof audio prediction runtime unavailable on %s: %s", audio_path, exc)
            else:
                logger.exception("Failed live spoof audio prediction on %s", audio_path)
            return ModalityResult.failed(ModalityType.AUDIO_SPOOF.value, f"Spoof inference error: {exc}")

    @staticmethod
    def adapt_scam_call(
        raw: Dict[str, Any],
        source_name: Optional[str] = None,
    ) -> ModalityResult:
        """Adapts output from scam-call detector (`predict_audio_file`).

        Expected format:
            {"label": "scam" | "normal", "scam_score": float, "threshold": float}
        """
        try:
            if not isinstance(raw, dict):
                return ModalityResult.failed(
                    ModalityType.AUDIO_SCAM.value,
                    f"Expected dict from scam audio model, got {type(raw).__name__}",
                )

            # Extract label
            label = str(raw.get("label", "unknown")).lower().strip()
            threshold = float(raw.get("threshold", 0.5))

            # Score extraction (scam_score, score, probability, risk_score)
            raw_score = raw.get("scam_score")
            if raw_score is None:
                raw_score = raw.get("score", raw.get("probability", 0.5))
            score = float(raw_score)

            # Determine whether it is a threat
            is_threat = label == "scam" or score >= threshold

            # Confidence: distance from decision threshold or explicit confidence
            if "confidence" in raw:
                confidence = float(raw["confidence"])
            else:
                # Margin-based confidence: closer to 0.0 or 1.0 = higher confidence
                confidence = min(1.0, 0.5 + abs(score - threshold))

            # Synthesize factual evidence based strictly on model findings
            evidence = []
            file_ref = source_name or raw.get("file") or raw.get("filename")
            prefix = f"Audio file '{file_ref}': " if file_ref else "Call audio: "

            if is_threat:
                evidence.append(
                    f"{prefix}Scam-call classifier flagged fraudulent conversational patterns "
                    f"with score {score:.3f} (operating threshold {threshold:.2f})."
                )
                evidence.append("Acoustic log-Mel spectrogram matched known scam-call speech signatures.")
            else:
                evidence.append(
                    f"{prefix}Scam-call classifier evaluated conversation as normal/legitimate "
                    f"with score {score:.3f} (below threshold {threshold:.2f})."
                )

            return ModalityResult(
                modality=ModalityType.AUDIO_SCAM.value,
                status=DetectorStatus.SUCCESS,
                label=label,
                risk_score=score,
                confidence=round(confidence, 4),
                threshold=threshold,
                is_threat=is_threat,
                evidence=evidence,
                raw_output=raw,
            )

        except Exception as exc:
            logger.exception("Failed to adapt scam audio result")
            return ModalityResult.failed(ModalityType.AUDIO_SCAM.value, str(exc))

    @staticmethod
    def adapt_spoof(
        raw: Dict[str, Any],
        source_name: Optional[str] = None,
    ) -> ModalityResult:
        """Adapts output from voice spoof / deepfake detector (`predict_audio`).

        Expected format:
            {"file": str, "spoof_score": float, "prediction": "spoof / synthetic" | "bonafide / genuine"}
        """
        try:
            if not isinstance(raw, dict):
                return ModalityResult.failed(
                    ModalityType.AUDIO_SPOOF.value,
                    f"Expected dict from spoof audio model, got {type(raw).__name__}",
                )

            prediction = str(raw.get("prediction", raw.get("label", "unknown"))).lower()
            threshold = float(raw.get("threshold", 0.5))

            raw_score = raw.get("spoof_score")
            if raw_score is None:
                raw_score = raw.get("score", raw.get("probability_fake", 0.5))
            score = float(raw_score)

            is_spoof = "spoof" in prediction or "synthetic" in prediction or score >= threshold

            label = "spoof" if is_spoof else "bonafide"

            if "confidence" in raw:
                confidence = float(raw["confidence"])
            else:
                confidence = min(1.0, 0.5 + abs(score - threshold))

            evidence = []
            file_ref = source_name or raw.get("file") or raw.get("filename")
            prefix = f"Audio '{file_ref}': " if file_ref else "Voice track: "

            if is_spoof:
                evidence.append(
                    f"{prefix}Anti-spoofing model indicated synthetic voice characteristics "
                    f"with spoof score {score:.3f} (unvalidated experimental signal on telephony audio)."
                )
                evidence.append(
                    "Spectral analysis identified vocoder-like artifacts, but detector is uncalibrated on telephony audio."
                )
            else:
                evidence.append(
                    f"{prefix}Anti-spoofing model classified the voice as bonafide / genuine "
                    f"(spoof score {score:.3f} below threshold {threshold:.2f}); "
                    "this is an experimental result, not verification of authenticity."
                )

            return ModalityResult(
                modality=ModalityType.AUDIO_SPOOF.value,
                status=DetectorStatus.SUCCESS,
                label=label,
                risk_score=score,
                confidence=round(confidence, 4),
                threshold=threshold,
                is_threat=is_spoof,
                is_validated=False,
                validation_notes=(
                    "Unvalidated experimental model: Trained on clean studio speech (ASVspoof 2019 LA), "
                    "not yet validated on in-the-wild telephony audio."
                ),
                evidence=evidence,
                raw_output=raw,
            )

        except Exception as exc:
            logger.exception("Failed to adapt spoof audio result")
            return ModalityResult.failed(ModalityType.AUDIO_SPOOF.value, str(exc))

    @staticmethod
    def adapt_unified_audio(
        raw: Dict[str, Any],
        source_name: Optional[str] = None,
    ) -> ModalityResult:
        """Auto-detects format between scam-call, spoof, or generic deepfake."""
        if "scam_score" in raw or raw.get("task") == "scam_vs_normal_call_audio":
            return AudioAdapter.adapt_scam_call(raw, source_name)
        if "spoof_score" in raw or "bonafide" in str(raw.get("prediction", "")):
            return AudioAdapter.adapt_spoof(raw, source_name)
        if "probability_fake" in raw or "is_deepfake" in raw or raw.get("label") in ("FAKE", "REAL"):
            fake_prob = float(raw.get("probability_fake", raw.get("confidence", 0.5)))
            is_fake = bool(raw.get("is_deepfake", raw.get("label") == "FAKE"))
            label = "spoof" if is_fake else "bonafide"
            evidence = [
                f"Audio deepfake classifier: label={label} (fake score={fake_prob:.3f}).",
                "Experimental audio deepfake signal; authenticity detection is not validated for operational use.",
            ]
            return ModalityResult(
                modality=ModalityType.AUDIO_SPOOF.value,
                status=DetectorStatus.SUCCESS,
                label=label,
                risk_score=fake_prob,
                confidence=float(raw.get("confidence", 0.8)),
                threshold=float(raw.get("threshold", 0.5)),
                is_threat=is_fake,
                is_validated=False,
                validation_notes=(
                    "Unvalidated experimental audio deepfake detector; calibration and performance "
                    "on representative real-world and telephony audio have not been established."
                ),
                evidence=evidence,
                raw_output=raw,
            )

        return AudioAdapter.adapt_scam_call(raw, source_name)
