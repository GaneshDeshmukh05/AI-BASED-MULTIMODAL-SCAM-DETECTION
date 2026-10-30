"""Enrich a held-out URL sample with domain-intelligence signals, fit a
logistic-regression scorer to replace the hand-picked weights, and report
results with and without domain intelligence.

Usage
-----
    python -m experiments.evaluate_domain_signals [--sample N] [--output-dir DIR]

Defaults
--------
    --sample    2000   URLs from the model's held-out test set
    --output-dir results/domain_signals/

Output files
------------
    enriched_test_sample.csv        raw + domain-intel features per URL
    enriched_test_sample_report.txt classification report for both conditions
    domain_scorer.joblib            saved to models/ (used by src.domain_scorer)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ensure project root is on the path when run as a module.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import (
    DOMAIN_SCORER_PATH,
    MODEL_CONFIG_PATH,
    MODEL_PATH,
    PRIMARY_DATASET,
    RANDOM_SEED,
    RESULTS_DIR,
)
from src.domain_intelligence import collect_intelligence
from src.domain_scorer import evidence_to_scorer_row, SCORER_FEATURES
from src.feature_engineering import extract_features, InvalidURLException, FEATURE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "domain_signals"


def _load_held_out_urls(sample_size: int) -> tuple[list[str], list[int]]:
    """Load a stratified sample from the primary dataset held-out split."""
    log.info("Loading dataset from %s …", PRIMARY_DATASET)
    df = pd.read_csv(PRIMARY_DATASET, usecols=["url", "label"], low_memory=False)
    df = df.dropna(subset=["url", "label"]).drop_duplicates(subset=["url"])
    df["label"] = df["label"].astype(int)
    # Recreate the same 80/20 split used during training so we evaluate on
    # strictly unseen data.
    _, test = train_test_split(df, test_size=0.20, random_state=RANDOM_SEED, stratify=df["label"])
    if len(test) > sample_size:
        test = test.groupby("label", group_keys=False).apply(
            lambda g: g.sample(min(len(g), sample_size // 2), random_state=RANDOM_SEED)
        )
    urls = test["url"].tolist()
    labels = test["label"].tolist()
    log.info("Held-out sample: %d URLs (%d phishing, %d legitimate)",
             len(urls), sum(labels), sum(1 - l for l in labels))
    return urls, labels


def _ml_probabilities(urls: list[str]) -> np.ndarray:
    """Return XGBoost phishing probabilities for each URL."""
    log.info("Loading XGBoost model from %s …", MODEL_PATH)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run experiments/run_all.py first.")
    model = joblib.load(MODEL_PATH)
    config = json.loads(MODEL_CONFIG_PATH.read_text())
    feature_cols = list(config["feature_columns"])
    rows = []
    for url in urls:
        try:
            rows.append([extract_features(url)[n] for n in feature_cols])
        except InvalidURLException:
            rows.append([0.0] * len(feature_cols))
    X = pd.DataFrame(rows, columns=feature_cols)
    return model.predict_proba(X)[:, 1]


def _enrich_one(url: str, ml_prob: float, url_is_https: bool) -> dict:
    """Collect domain-intelligence signals for one URL (called in thread pool)."""
    try:
        ev = collect_intelligence(url, timeout=5.0)
        row = evidence_to_scorer_row(ml_prob, ev, url_is_https)
        result: dict[str, Any] = dict(zip(SCORER_FEATURES, row))
        result["url"] = url
        result["ml_probability"] = ml_prob
        return result
    except Exception as exc:
        log.debug("Enrichment failed for %s: %s", url, exc)
        # Return a zero-filled row so we don't drop the URL from evaluation.
        return {"url": url, "ml_probability": ml_prob,
                **{f: 0.0 for f in SCORER_FEATURES if f != "ml_probability"}}


def enrich_sample(urls: list[str], ml_probs: np.ndarray, max_workers: int = 8) -> pd.DataFrame:
    """Enrich URLs with domain signals using a thread pool."""
    log.info("Enriching %d URLs with domain intelligence (workers=%d) …", len(urls), max_workers)
    results: list[dict] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_enrich_one, url, float(prob), url.startswith("https")): i
            for i, (url, prob) in enumerate(zip(urls, ml_probs))
        }
        for done, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            if done % 100 == 0:
                elapsed = time.perf_counter() - start
                log.info("  %d / %d enriched (%.1fs elapsed)", done, len(urls), elapsed)
    return pd.DataFrame(results)


def fit_and_report(df: pd.DataFrame, labels: list[int], output_dir: Path) -> None:
    """Fit logistic regression on enrichment features and print a comparison report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    y = np.array(labels)

    # --- ML-only baseline ---
    ml_probs = np.asarray(df["ml_probability"], dtype=float)
    ml_preds = (ml_probs >= 0.5).astype(int)
    ml_roc = roc_auc_score(y, ml_probs)
    ml_prc = average_precision_score(y, ml_probs)
    log.info("ML-only  ROC-AUC=%.4f  PR-AUC=%.4f", ml_roc, ml_prc)

    # --- Logistic regression on enrichment signals ---
    X = df[SCORER_FEATURES].fillna(0.0).values
    # Keep the same 80/20 split *within* the enriched sample for the scorer.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_SEED, stratify=y
    )
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])
    lr_pipeline.fit(X_train, y_train)
    lr_probs = lr_pipeline.predict_proba(X_test)[:, 1]
    lr_preds = (lr_probs >= 0.5).astype(int)
    lr_roc = roc_auc_score(y_test, lr_probs)
    lr_prc = average_precision_score(y_test, lr_probs)
    log.info("LR+Intel ROC-AUC=%.4f  PR-AUC=%.4f", lr_roc, lr_prc)

    # --- Save report ---
    report_lines = [
        "=" * 60,
        "Domain-Intelligence Signal Evaluation",
        "=" * 60,
        "",
        "--- ML-only (XGBoost, no domain intelligence) ---",
        f"ROC-AUC : {ml_roc:.4f}",
        f"PR-AUC  : {ml_prc:.4f}",
        "",
        classification_report(y, ml_preds, target_names=["legitimate", "phishing"]),
        "",
        "--- Logistic Regression with Domain-Intelligence Signals ---",
        f"ROC-AUC : {lr_roc:.4f}",
        f"PR-AUC  : {lr_prc:.4f}",
        "",
        classification_report(y_test, lr_preds, target_names=["legitimate", "phishing"]),
        "",
        f"Δ ROC-AUC = {lr_roc - ml_roc:+.4f}",
        f"Δ PR-AUC  = {lr_prc - ml_prc:+.4f}",
        "",
        "Feature importances (logistic regression coefficients):",
        *[f"  {name:35s}: {coef:+.4f}"
          for name, coef in sorted(
              zip(SCORER_FEATURES, lr_pipeline.named_steps["lr"].coef_[0]),
              key=lambda x: abs(x[1]), reverse=True
          )],
    ]
    report_path = output_dir / "enriched_test_sample_report.txt"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    log.info("Report saved to %s", report_path)
    print("\n".join(report_lines))

    # --- Save enriched data ---
    df["label"] = labels
    csv_path = output_dir / "enriched_test_sample.csv"
    df.to_csv(csv_path, index=False)
    log.info("Enriched data saved to %s", csv_path)

    # --- Save scorer artifact ---
    # Refit on the full enriched set for production use.
    lr_full = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])
    lr_full.fit(X, y)
    DOMAIN_SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr_full, DOMAIN_SCORER_PATH)
    log.info("Domain scorer saved to %s", DOMAIN_SCORER_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=2000, help="Number of held-out URLs to enrich (default: 2000)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for output files")
    parser.add_argument("--workers", type=int, default=8, help="Thread pool size for enrichment (default: 8)")
    args = parser.parse_args()

    urls, labels = _load_held_out_urls(args.sample)
    ml_probs = _ml_probabilities(urls)
    enriched_df = enrich_sample(urls, ml_probs, max_workers=args.workers)
    fit_and_report(enriched_df, labels, args.output_dir)


if __name__ == "__main__":
    main()
