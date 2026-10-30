"""Shared implementations used by individual and full research experiments."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_SEED
from src.evaluate import binary_metrics, measure_per_item, plot_model_comparison, prediction_scores
from src.feature_engineering import FEATURE_NAMES, extract_feature_frame
from src.train import make_xgboost_classifier


def model_comparison(train_frame: pd.DataFrame, test_frame: pd.DataFrame, xgboost_params: dict[str, Any], output_csv, output_png) -> pd.DataFrame:
    """Fit required baselines on identical data and record actual test metrics."""
    train_features = extract_feature_frame(train_frame["url"])
    test_features = extract_feature_frame(test_frame["url"])
    candidates = [
        ("Logistic Regression", Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))]), True),
        ("Decision Tree", DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=20, min_samples_leaf=2), True),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, min_samples_leaf=2), True),
        ("Linear SVM", Pipeline([("scale", StandardScaler()), ("model", LinearSVC(random_state=RANDOM_SEED, dual="auto", max_iter=5000))]), False),
        ("XGBoost", make_xgboost_classifier(xgboost_params), True),
    ]
    rows: list[dict[str, Any]] = []
    for name, model, is_probability in candidates:
        start = perf_counter()
        model.fit(train_features, train_frame["target"])
        fit_seconds = perf_counter() - start
        scores = prediction_scores(model, test_features)
        metrics = binary_metrics(test_frame["target"], scores, threshold=0.5, score_is_probability=is_probability)
        latency = measure_per_item(
            "model_prediction",
            range(len(test_features)),
            lambda index: prediction_scores(model, test_features.iloc[[index]]),
            maximum_samples=200,
        )
        rows.append(
            {
                "model": name,
                "evaluation_split": "fixed_unseen_domain_test",
                "score_type": "probability" if is_probability else "decision_function",
                "training_rows": len(train_frame),
                "test_rows": len(test_frame),
                "fit_time_seconds": fit_seconds,
                "inference_time_ms_mean": latency["mean_ms"],
                **metrics,
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)
    plot_model_comparison(results, output_png)
    return results


def ablation_study(train_frame: pd.DataFrame, test_frame: pd.DataFrame, xgboost_params: dict[str, Any], feature_sets: dict[str, list[str]], output_csv, output_png) -> pd.DataFrame:
    """Evaluate feature groups with the same unseen-domain split and model params."""
    train_features = extract_feature_frame(train_frame["url"])
    test_features = extract_feature_frame(test_frame["url"])
    rows: list[dict[str, Any]] = []
    for group, columns in feature_sets.items():
        model = make_xgboost_classifier(xgboost_params)
        start = perf_counter()
        model.fit(train_features[columns], train_frame["target"])
        fit_seconds = perf_counter() - start
        scores = model.predict_proba(test_features[columns])[:, 1]
        rows.append(
            {
                "feature_group": group,
                "feature_count": len(columns),
                "features": "|".join(columns),
                "evaluation_split": "fixed_unseen_domain_test",
                "threshold": 0.5,
                "fit_time_seconds": fit_seconds,
                **binary_metrics(test_frame["target"], scores, threshold=0.5),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(output_csv, index=False)
    axis = results.set_index("feature_group")[["f1", "roc_auc", "pr_auc"]].plot(kind="bar", figsize=(11, 5), ylim=(0, 1), rot=20)
    axis.set_ylabel("Score")
    axis.set_title("Feature ablation on the fixed unseen-domain test set")
    plt.tight_layout()
    plt.savefig(output_png, dpi=180)
    plt.close()
    return results
