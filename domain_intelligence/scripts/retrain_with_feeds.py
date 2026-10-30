"""Conservative retraining on existing primary data + current OpenPhish URLs.

This is a separate, explicit command: it does not run at API startup and does
not automatically deploy a new model. A held-out evaluation is still required
before using the resulting artifacts in production.
"""
from __future__ import annotations
import json
import pandas as pd
from src.config import PRIMARY_DATASET, MODELS_DIR, RESULTS_DIR, ensure_output_directories
from src.preprocessing import load_and_clean_dataset, make_unseen_domain_split, domain_cv_splits
from src.feature_engineering import extract_feature_frame
from src.train import make_xgboost_classifier, choose_threshold_from_domain_oof, save_model_bundle
from src.threat_feeds import load_feed


def main():
    ensure_output_directories()
    base, report = load_and_clean_dataset(PRIMARY_DATASET)
    feed = load_feed()
    if not feed.get("urls"):
        raise SystemExit("No cached OpenPhish feed. Run: python -m scripts.update_threat_feed")
    fresh = pd.DataFrame({"url": list(feed["urls"].keys()), "target": 1})
    combined = pd.DataFrame(pd.concat([base[["url", "target"]], fresh], ignore_index=True))
    # Re-clean URL keys and remove conflicts: feed-positive wins only when the
    # same URL is not labeled legitimate by the existing source; conflicts drop.
    from src.feature_engineering import canonical_url_key, get_domain, InvalidURLException
    keys=[]; domains=[]
    for u in combined.url:
        try: keys.append(canonical_url_key(u)); domains.append(get_domain(u))
        except Exception: keys.append(None); domains.append(None)
    combined["url_key"], combined["domain"] = keys, domains
    combined = combined.dropna(subset=["url_key","domain"])
    conflicts = combined.groupby("url_key").target.nunique()
    bad = conflicts[conflicts > 1].index.tolist()
    filtered: pd.DataFrame = combined.loc[~combined["url_key"].isin(bad)]
    combined = filtered.drop_duplicates(subset=["url_key"]).reset_index(drop=True)
    if combined.target.nunique()!=2 or combined.target.value_counts().min()<3:
        raise SystemExit("Need at least 3 examples in each class after deduplication")
    train, test = make_unseen_domain_split(combined, n_splits=5)
    # Fit and select threshold using training rows only. Test rows are saved for
    # an explicit evaluation step and never used for threshold selection.
    try:
        old_cfg=json.loads((MODELS_DIR/"best_xgboost_config.json").read_text(encoding="utf-8"))
        params=old_cfg.get("parameters", {})
    except Exception:
        params={"n_estimators":250,"max_depth":4,"learning_rate":0.05,"subsample":0.9,"colsample_bytree":0.85}
    threshold, threshold_rows, _ = choose_threshold_from_domain_oof(train, params)
    model = make_xgboost_classifier(params)
    model.fit(extract_feature_frame(train.url), train.target)
    # Hold out an untouched domain-disjoint test set for review.
    threshold_rows.to_csv(RESULTS_DIR/"feed_retrain_thresholds.csv", index=False)
    test[["url","target","domain"]].to_csv(RESULTS_DIR/"feed_retrain_holdout.csv", index=False)
    report_out={"dataset":"primary + OpenPhish community feed", "base_rows":len(base), "feed_rows":len(fresh), "combined_rows":len(combined), "train_rows":len(train), "holdout_rows":len(test), "feed_updated_at":feed.get("updated_at"), "note":"Evaluate holdout before deployment; feed labels can be stale and source distribution differs."}
    save_model_bundle(model, params, threshold, report_out)
    print(json.dumps(report_out, indent=2))
    print("Model artifacts updated. Restart FastAPI to reload them, and evaluate results/feed_retrain_holdout.csv before deployment.")

if __name__ == "__main__": main()
