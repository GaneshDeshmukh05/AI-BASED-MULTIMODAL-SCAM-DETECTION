"""SHAP explanations for the trained URL-only XGBoost model.

SHAP values describe how the fitted model used its input features. They do not
establish that any feature causes phishing or legitimacy.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from .config import EXPLAINER_PATH, SHAP_RESULTS_DIR, ensure_output_directories


def load_explainer():
    """Load the explainer produced alongside the final model."""
    try:
        return joblib.load(EXPLAINER_PATH)
    except Exception:
        from .config import MODEL_PATH
        model = joblib.load(MODEL_PATH)
        return shap.TreeExplainer(model)


def _values(explanation) -> np.ndarray:
    values = explanation.values if hasattr(explanation, "values") else explanation
    values = np.asarray(values)
    # Some older SHAP releases retain an explicit class axis for classifiers.
    if values.ndim == 3:
        values = values[:, :, 1] if values.shape[-1] == 2 else values[:, 1, :]
    return values


def individual_contributions(explainer, features: pd.DataFrame) -> dict[str, float]:
    """Return signed per-feature SHAP values for one feature row."""
    explanation = explainer(features)
    values = _values(explanation)
    if values.ndim != 2 or len(values) != 1:
        raise ValueError("Expected a single-row SHAP explanation")
    return {name: float(value) for name, value in zip(features.columns, values[0])}


def save_global_shap_plots(explainer, features: pd.DataFrame, maximum_samples: int = 1000) -> pd.DataFrame:
    """Save global summary and mean-absolute-importance SHAP visualizations."""
    ensure_output_directories()
    sample = features.iloc[:maximum_samples].copy()
    explanation = explainer(sample)
    values = _values(explanation)
    mean_abs = np.abs(values).mean(axis=0)
    importance = pd.DataFrame({"feature": sample.columns, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(SHAP_RESULTS_DIR / "global_feature_importance.csv", index=False)

    plt.figure(figsize=(9, 7))
    shap.summary_plot(values, sample, show=False, plot_size=None)
    plt.title("SHAP summary — final XGBoost URL model")
    plt.tight_layout()
    plt.savefig(SHAP_RESULTS_DIR / "summary.png", dpi=180, bbox_inches="tight")
    plt.close()

    plot_data = importance.iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(plot_data["feature"], plot_data["mean_abs_shap"])
    plt.xlabel("Mean absolute SHAP value")
    plt.title("Global SHAP feature importance")
    plt.tight_layout()
    plt.savefig(SHAP_RESULTS_DIR / "feature_importance.png", dpi=180)
    plt.close()
    return importance.reset_index(drop=True)
