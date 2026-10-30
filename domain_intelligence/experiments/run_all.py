"""Run the complete URL-only research experiment package.

Example:
    python -m experiments.run_all --tuning-iterations 8
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src.config import CROSS_DATASET, PRIMARY_DATASET, RESULTS_DIR, ensure_output_directories
from src.evaluate import binary_metrics, measure_per_item, plot_confusion_matrix, plot_xgboost_curves
from src.explainability import load_explainer, save_global_shap_plots
from src.feature_engineering import ablation_feature_sets, extract_feature_frame
from src.preprocessing import (
    load_and_clean_dataset,
    make_stratified_random_split,
    make_unseen_domain_split,
    remove_overlapping_domains,
)
from src.robustness import evaluate_robustness
from src.train import (
    choose_threshold_from_domain_oof,
    fit_final_xgboost,
    save_model_bundle,
    xgboost_search,
)

from .common import ablation_study, model_comparison


def _split_summary(name: str, train: pd.DataFrame, test: pd.DataFrame) -> dict[str, object]:
    return {
        "name": name,
        "train_rows": len(train),
        "test_rows": len(test),
        "train_domains": int(train["domain"].nunique()),
        "test_domains": int(test["domain"].nunique()),
        "shared_domains": len(set(train["domain"]) & set(test["domain"])),
        "train_class_counts": {str(key): int(value) for key, value in train["target"].value_counts().sort_index().items()},
        "test_class_counts": {str(key): int(value) for key, value in test["target"].value_counts().sort_index().items()},
    }


def run(tuning_iterations: int = 8) -> None:
    ensure_output_directories()
    primary, primary_report = load_and_clean_dataset(PRIMARY_DATASET)
    external, external_report = load_and_clean_dataset(CROSS_DATASET)
    (RESULTS_DIR / "dataset_cleaning_report.json").write_text(
        json.dumps({"primary": primary_report.to_dict(), "cross_dataset": external_report.to_dict()}, indent=2), encoding="utf-8"
    )

    random_train, random_test = make_stratified_random_split(primary)
    domain_train, domain_test = make_unseen_domain_split(primary)
    split_summary = {
        "stratified_random": _split_summary("stratified_random", random_train, random_test),
        "unseen_domain": _split_summary("unseen_domain", domain_train, domain_test),
    }
    (RESULTS_DIR / "split_summary.json").write_text(json.dumps(split_summary, indent=2), encoding="utf-8")

    parameters, cv_score = xgboost_search(domain_train, n_iter=tuning_iterations)
    threshold, threshold_oof, oof_scores = choose_threshold_from_domain_oof(domain_train, parameters)
    final_model, _ = fit_final_xgboost(domain_train, parameters)
    save_model_bundle(
        final_model,
        parameters,
        threshold,
        training_report={"cleaning": primary_report.to_dict(), "split": split_summary["unseen_domain"]},
        cv_score=cv_score,
    )

    domain_test_features = extract_feature_frame(domain_test["url"])
    domain_probabilities = final_model.predict_proba(domain_test_features)[:, 1]
    zero_day_metrics = binary_metrics(domain_test["target"], domain_probabilities, threshold=threshold)
    pd.DataFrame(
        [{"evaluation_type": "unseen_domain_fixed_holdout", "threshold": threshold, "training_rows": len(domain_train), "test_rows": len(domain_test), "test_domains": int(domain_test["domain"].nunique()), **zero_day_metrics}]
    ).to_csv(RESULTS_DIR / "zero_day_results.csv", index=False)
    plot_xgboost_curves(domain_test["target"], domain_probabilities, RESULTS_DIR)
    plot_confusion_matrix(domain_test["target"], domain_probabilities, threshold, RESULTS_DIR / "confusion_matrix.png")

    # Domain-CV aggregate comes from OOF predictions generated only on the training partition.
    pd.DataFrame(
        [{"evaluation_type": "domain_disjoint_oof_training", "threshold": 0.5, "rows": len(domain_train), **binary_metrics(domain_train["target"], oof_scores, threshold=0.5)}]
    ).to_csv(RESULTS_DIR / "domain_cv_results.csv", index=False)

    random_model, _ = fit_final_xgboost(random_train, parameters)
    random_scores = random_model.predict_proba(extract_feature_frame(random_test["url"]))[:, 1]
    pd.DataFrame(
        [{"evaluation_type": "stratified_random_fixed_holdout", "threshold": threshold, "training_rows": len(random_train), "test_rows": len(random_test), **binary_metrics(random_test["target"], random_scores, threshold=threshold)}]
    ).to_csv(RESULTS_DIR / "random_split_results.csv", index=False)

    test_threshold_rows: list[dict[str, object]] = []
    for threshold_value in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
        test_threshold_rows.append(
            {"split": "fixed_unseen_domain_test_not_used_for_selection", "threshold": threshold_value, "selected": False, **binary_metrics(domain_test["target"], domain_probabilities, threshold=threshold_value)}
        )
    threshold_results = pd.concat([threshold_oof, pd.DataFrame(test_threshold_rows)], ignore_index=True, sort=False)
    threshold_results.to_csv(RESULTS_DIR / "threshold_results.csv", index=False)

    model_comparison(domain_train, domain_test, parameters, RESULTS_DIR / "model_comparison.csv", RESULTS_DIR / "model_comparison.png")
    ablation_study(domain_train, domain_test, parameters, ablation_feature_sets(), RESULTS_DIR / "ablation_results.csv", RESULTS_DIR / "ablation_results.png")

    external_unseen = remove_overlapping_domains(external, primary)
    external_features = extract_feature_frame(external_unseen["url"])
    external_scores = final_model.predict_proba(external_features)[:, 1]
    pd.DataFrame(
        [{"evaluation_type": "external_dataset_unseen_domains", "source_dataset": str(CROSS_DATASET), "training_rows": len(domain_train), "test_rows": len(external_unseen), "test_domains": int(external_unseen["domain"].nunique()), "removed_overlapping_domain_rows": len(external) - len(external_unseen), "threshold": threshold, **binary_metrics(external_unseen["target"], external_scores, threshold=threshold)}]
    ).to_csv(RESULTS_DIR / "cross_dataset_results.csv", index=False)

    robustness = evaluate_robustness(final_model, extract_feature_frame, domain_test, threshold)
    robustness.to_csv(RESULTS_DIR / "robustness_results.csv", index=False)

    explainer = load_explainer()
    save_global_shap_plots(explainer, domain_test_features)
    latency_rows = [
        measure_per_item("feature_extraction", domain_test["url"], lambda url: extract_feature_frame([url])),
        measure_per_item("prediction_only", domain_test["url"], lambda url: final_model.predict_proba(extract_feature_frame([url]))),
        measure_per_item("shap_explanation", domain_test["url"], lambda url: explainer(extract_feature_frame([url]))),
        measure_per_item("prediction_plus_shap", domain_test["url"], lambda url: (final_model.predict_proba(extract_feature_frame([url])), explainer(extract_feature_frame([url])))),
    ]
    pd.DataFrame(latency_rows).to_csv(RESULTS_DIR / "latency_results.csv", index=False)
    print(json.dumps({"selected_threshold": threshold, "best_domain_cv_pr_auc": cv_score, "results_directory": str(RESULTS_DIR)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run reproducible phishing URL experiments")
    parser.add_argument("--tuning-iterations", type=int, default=8, help="Number of domain-CV XGBoost parameter samples (default: 8)")
    arguments = parser.parse_args()
    run(tuning_iterations=arguments.tuning_iterations)
