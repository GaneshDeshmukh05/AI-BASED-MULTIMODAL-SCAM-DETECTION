# Experimental Protocol

## Data cleaning

Each input is read reproducibly, labels are mapped to `1=phishing` and
`0=legitimate`, missing/invalid/unknown-label rows are removed, conflicting
canonical URL labels are removed, and exact canonical URL duplicates are
deduplicated. Different paths at the same hostname are retained because they
are valid URL observations, but hostname is used as the group for all
unseen-domain protections. Cleaning counts are saved in
`results/dataset_cleaning_report.json`.

## Split protocol and leakage controls

* **Stratified random split:** an 80/20 URL-level split, included only as a
  conventional same-dataset reference (`random_split_results.csv`).
* **Fixed unseen-domain holdout:** a grouped, approximately stratified 5-fold
  partition holds out one fold of hostnames. These domains never enter final
  training, tuning, threshold selection, or feature selection.
* **Domain-aware CV:** three grouped folds inside the training partition tune
  XGBoost using PR-AUC and produce OOF probabilities for threshold selection.
* **Cross-dataset evaluation:** all external URLs whose hostname appears
  anywhere in the primary dataset are removed before scoring. It remains an
  external-distribution result, not proof of universal generalization.
* **Temporal evaluation:** not run because neither supplied file has a
  trustworthy timestamp column.

The supplied primary and external datasets share URLs and domains; the
cross-dataset de-overlap is essential.

## Models and metrics

Logistic Regression, Decision Tree, Random Forest, Linear SVM, and XGBoost are
fit on identical unseen-domain-training rows and evaluated on the same fixed
unseen-domain test rows. XGBoost hyperparameters are searched only using
domain-aware training folds. Reported CSVs contain accuracy, precision, recall,
F1, ROC-AUC, PR-AUC, FPR, FNR, counts, fit time, and/or inference latency where
applicable.

The threshold is chosen by maximizing F1 from domain-disjoint OOF predictions
on the training data. Ties choose lower FPR and then a lower threshold. The
fixed test threshold table is descriptive and is explicitly not used for
selection.

## Ablation and robustness

The ablation compares cumulative lexical, structural/security, entropy, and
keyword feature groups using identical XGBoost settings and the fixed final
holdout. It is not a license to select features after looking at that test
holdout; any new feature-set decision must be rerun through the training-CV
selection protocol.

Robustness variants are generated offline from URL strings only: added
subdomains/path/query segments, hyphenated segments, percent encoding, and
keyword removal when applicable. No original or modified URL is visited. The
result table reports original/modified predictions and score change, not a
claim that samples reflect live attacks.

## SHAP and latency

Global SHAP summary and mean-absolute feature-importance plots are generated
from a capped held-out sample. Per-URL API explanations include signed feature
contributions. SHAP describes the fitted model's behavior and does not prove
causality. Latency is measured per URL for feature extraction, prediction,
SHAP, and prediction-plus-SHAP; performance claims must use the values actually
written to `results/latency_results.csv`.

## Commands

```powershell
python -m experiments.run_all --tuning-iterations 8
pytest -q
uvicorn main:app --host 127.0.0.1 --port 8000
```

The small wrappers in `experiments/baseline_comparison.py`,
`ablation_study.py`, `cross_dataset.py`, and `zero_day.py` intentionally invoke
the complete protocol so outputs always share the same controlled split and
model configuration.
