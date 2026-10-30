"""Metric, plotting, and latency helpers for URL-model experiments."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


METRIC_COLUMNS = [
    "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "false_positive_rate", "false_negative_rate",
]


def prediction_scores(model, features: pd.DataFrame) -> np.ndarray:
    """Return phishing scores; probability when the estimator provides it."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))[:, 1]
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(features))
    return np.asarray(model.predict(features))


def binary_metrics(y_true: Iterable[int], scores: Iterable[float], threshold: float = 0.5, score_is_probability: bool = True) -> dict[str, float]:
    """Calculate classification and error-rate metrics without hidden defaults."""
    actual = np.asarray(list(y_true), dtype=int)
    values = np.asarray(list(scores), dtype=float)
    cutoff = threshold if score_is_probability else 0.0
    predicted = (values >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    negative = max(1, tn + fp)
    positive = max(1, tp + fn)
    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, values)),
        "pr_auc": float(average_precision_score(actual, values)),
        "false_positive_rate": float(fp / negative),
        "false_negative_rate": float(fn / positive),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def plot_model_comparison(results: pd.DataFrame, output_path: str | Path) -> None:
    """Save a compact, data-driven bar chart of key model-comparison metrics."""
    metrics = ["f1", "roc_auc", "pr_auc"]
    plot = results.set_index("model")[metrics]
    axis = plot.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=20)
    axis.set_ylabel("Score")
    axis.set_title("Baseline comparison on the fixed unseen-domain test set")
    axis.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_xgboost_curves(y_true: Iterable[int], probabilities: Iterable[float], output_dir: str | Path) -> None:
    """Generate ROC, precision-recall, and confusion-matrix plots from real predictions."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    actual = np.asarray(list(y_true), dtype=int)
    scores = np.asarray(list(probabilities), dtype=float)
    fpr, tpr, _ = roc_curve(actual, scores)
    precision, recall, _ = precision_recall_curve(actual, scores)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(actual, scores):.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("XGBoost ROC curve — unseen-domain test")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(destination / "roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"PR-AUC = {average_precision_score(actual, scores):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("XGBoost precision-recall curve — unseen-domain test")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(destination / "precision_recall_curve.png", dpi=180)
    plt.close()


def plot_confusion_matrix(y_true: Iterable[int], probabilities: Iterable[float], threshold: float, output_path: str | Path) -> None:
    """Save a confusion matrix at the threshold selected without using final-test labels."""
    actual = np.asarray(list(y_true), dtype=int)
    predicted = (np.asarray(list(probabilities), dtype=float) >= threshold).astype(int)
    matrix = confusion_matrix(actual, predicted, labels=[0, 1])
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks([0, 1], ["Legitimate", "Phishing"])
    axis.set_yticks([0, 1], ["Legitimate", "Phishing"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title(f"XGBoost confusion matrix (threshold={threshold:.2f})")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def latency_summary(values_ms: list[float], operation: str, samples: int) -> dict[str, float | int | str]:
    """Summarize observed per-request latency in milliseconds."""
    values = np.asarray(values_ms, dtype=float)
    return {
        "operation": operation,
        "samples": samples,
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(np.min(values)),
        "max_ms": float(np.max(values)),
    }


def measure_per_item(operation: str, items: Iterable, function: Callable, maximum_samples: int = 200) -> dict[str, float | int | str]:
    """Measure one operation per item, avoiding aggregate-only timing."""
    observed: list[float] = []
    for item in list(items)[:maximum_samples]:
        start = perf_counter()
        function(item)
        observed.append((perf_counter() - start) * 1000)
    if not observed:
        raise ValueError(f"No samples available for latency operation: {operation}")
    return latency_summary(observed, operation, len(observed))
