"""Training, tuning, threshold selection, and model persistence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from .config import CV_FOLDS, EXPLAINER_PATH, FEATURE_MANIFEST_PATH, MODEL_CONFIG_PATH, MODEL_PATH, RANDOM_SEED, ensure_output_directories
from .evaluate import binary_metrics
from .feature_engineering import FEATURE_NAMES, extract_feature_frame
from .preprocessing import domain_cv_splits


def make_xgboost_classifier(params: dict[str, Any] | None = None) -> XGBClassifier:
    """Create one deterministic CPU XGBoost classifier for binary URL labels."""
    base: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": 0,
    }
    if params:
        base.update(params)
    return XGBClassifier(**base)


def xgboost_search(frame: pd.DataFrame, feature_columns: list[str] | tuple[str, ...] = FEATURE_NAMES, n_iter: int = 8) -> tuple[dict[str, Any], float]:
    """Tune XGBoost on domain-disjoint folds; the final test set is untouched."""
    features = extract_feature_frame(frame["url"])[list(feature_columns)]
    labels = frame["target"].to_numpy()
    distributions: dict[str, list[Any]] = {
        "n_estimators": [150, 250, 350],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.08],
        "subsample": [0.75, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.3],
        "reg_alpha": [0.0, 0.01, 0.1],
        "reg_lambda": [1.0, 2.0, 5.0],
    }
    search = RandomizedSearchCV(
        estimator=make_xgboost_classifier(),
        param_distributions=distributions,
        n_iter=n_iter,
        scoring="average_precision",
        cv=list(domain_cv_splits(frame, n_splits=CV_FOLDS)),
        random_state=RANDOM_SEED,
        n_jobs=1,
        refit=False,
        verbose=1,
    )
    search.fit(features, labels)
    return dict(search.best_params_), float(search.best_score_)


def choose_threshold_from_domain_oof(frame: pd.DataFrame, parameters: dict[str, Any], feature_columns: list[str] | tuple[str, ...] = FEATURE_NAMES) -> tuple[float, pd.DataFrame, np.ndarray]:
    """Select a threshold from domain-disjoint OOF scores, never final-test labels."""
    features = extract_feature_frame(frame["url"])[list(feature_columns)]
    oof = np.full(len(frame), np.nan, dtype=float)
    for train_index, valid_index in domain_cv_splits(frame, n_splits=CV_FOLDS):
        model = make_xgboost_classifier(parameters)
        model.fit(features.iloc[train_index], frame.iloc[train_index]["target"])
        oof[valid_index] = model.predict_proba(features.iloc[valid_index])[:, 1]
    if np.isnan(oof).any():
        raise RuntimeError("Threshold OOF predictions are incomplete")
    rows: list[dict[str, Any]] = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        metric = binary_metrics(frame["target"], oof, threshold=threshold)
        rows.append({"split": "training_domain_oof", "threshold": threshold, **metric})
    results = pd.DataFrame(rows)
    # F1 is the pre-declared operating objective; break ties toward lower FPR.
    selected = results.sort_values(["f1", "false_positive_rate", "threshold"], ascending=[False, True, True]).iloc[0]
    results["selected"] = results["threshold"].eq(float(selected["threshold"]))
    return float(selected["threshold"]), results, oof


def fit_final_xgboost(frame: pd.DataFrame, parameters: dict[str, Any], feature_columns: list[str] | tuple[str, ...] = FEATURE_NAMES) -> tuple[XGBClassifier, pd.DataFrame]:
    """Fit the final model strictly on the fixed training partition."""
    features = pd.DataFrame(extract_feature_frame(frame["url"])[list(feature_columns)])
    model = make_xgboost_classifier(parameters)
    model.fit(features, frame["target"])
    return model, features


def save_model_bundle(model: XGBClassifier, parameters: dict[str, Any], threshold: float, training_report: dict[str, Any], feature_columns: list[str] | tuple[str, ...] = FEATURE_NAMES, cv_score: float | None = None) -> None:
    """Persist model/explainer plus portable JSON metadata."""
    ensure_output_directories()
    explainer = shap.TreeExplainer(model)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(explainer, EXPLAINER_PATH)
    model_config = {
        "model_type": "XGBClassifier",
        "parameters": parameters,
        "random_seed": RANDOM_SEED,
        "cv_scoring": "average_precision",
        "domain_cv_folds": CV_FOLDS,
        "best_domain_cv_pr_auc": cv_score,
        "selected_threshold": threshold,
        "threshold_selection": "maximize F1 on domain-disjoint out-of-fold training predictions; ties minimize FPR then threshold",
        "feature_columns": list(feature_columns),
    }
    MODEL_CONFIG_PATH.write_text(json.dumps(model_config, indent=2), encoding="utf-8")
    manifest = {
        "feature_columns": list(feature_columns),
        "training_report": training_report,
        "model_path": MODEL_PATH.name,
        "explainer_path": EXPLAINER_PATH.name,
    }
    FEATURE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_best_parameters() -> tuple[dict[str, Any], float]:
    """Read trained configuration for experiment/API reuse."""
    config = json.loads(Path(MODEL_CONFIG_PATH).read_text(encoding="utf-8"))
    return dict(config["parameters"]), float(config["selected_threshold"])
