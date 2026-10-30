# Project Audit — Existing Phishing URL Detection System

Audit completed: 2026-09-04. This document records the repository state before
the research-pipeline upgrade.

## Current architecture

The repository is a small Python project with no package structure,
configuration, tests, dependency manifest, or version-control metadata in the
workspace. It contains:

| Path | Purpose |
| --- | --- |
| `main.py` | FastAPI inference service, URL feature extraction, XGBoost and SHAP artifact loading. |
| `Untitled.ipynb` | Exploratory inspection plus the only documented training workflow. |
| `phishing.csv` | PhiUSIIL-style dataset used by the notebook to train the exported model. |
| `raw_data.csv` | A second two-column URL dataset; it is not used by the notebook or API. |
| `xgboost_model.pkl` | Exported model artifact. |
| `shap_explainer.pkl` | Exported SHAP TreeExplainer artifact. |

At API import time, `main.py` calls `joblib.load` for both artifacts. The
current environment cannot deserialize them because `xgboost` is not installed;
the notebook source nevertheless establishes the intended artifact type and
feature schema.

## Existing URL features

The training notebook extracts 12 URL-only features, in this order:

1. `url_length`
2. `domain_length`
3. `dot_count`
4. `hyphen_count`
5. `digit_count`
6. `subdomain_count`
7. `uses_https`
8. `ip_in_url`
9. `suspicious_keyword_count`
10. `url_entropy`
11. `path_length`
12. `query_param_count`

The suspicious-keyword list is `login`, `verify`, `account`, `banking`,
`secure`, `update`, `signin`, `wp-content`, and `admin`. Entropy is Shannon
entropy over URL characters. The notebook creates the complete feature matrix
from URLs and uses none of the page-content/site-derived columns in
`phishing.csv`.

## Dataset details

### `raw_data.csv`

* Records: 253,098
* Columns: `url`, `label`
* Missing values: none
* Classes: 129,420 legitimate (51.13%); 123,678 phishing (48.87%)
* Normalized duplicate-URL rows: 36; unique normalized URLs: 253,062
* Invalid/no-host URLs under offline parsing: 0
* Unique hostnames: 128,379; duplicate-domain rows: 124,719
* Exact duplicate URLs with conflicting labels: 0
* Domains observed with both labels: 79

### `phishing.csv`

* Records: 235,795
* Columns: 56 (`FILENAME`, `URL`, URL/domain lexical fields, page/HTML fields,
  and `label`)
* Missing values: none in any column
* Stored labels: 134,850 label `1` (57.19%) and 100,945 label `0` (42.81%)
* The notebook maps PhiUSIIL labels as `0 -> phishing (1)` and
  `1 -> legitimate (0)`, producing 100,945 phishing and 134,850 legitimate
  samples before deduplication.
* Normalized duplicate-URL rows: 425; unique normalized URLs: 235,370
* Invalid/no-host URLs under offline parsing: 0
* Unique hostnames: 220,077; duplicate-domain rows: 15,718
* Exact duplicate URLs with conflicting labels: 0
* Domains observed with both labels: 54

The two datasets overlap: 1,805 normalized URLs and 6,305 hostnames occur in
both. They therefore cannot be treated as independently sourced evaluation
data without de-overlapping and label-consistency checks.

## Model and training details

The training cells load `phishing.csv`, lowercase column names, drop records
missing `url`/`label`, remove duplicate URLs, apply the inverted PhiUSIIL label
mapping, extract the 12 features above, then use:

```text
train_test_split(test_size=0.20, random_state=42, stratify=y)
XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
              eval_metric="logloss", random_state=42)
```

The notebook reports ROC-AUC and a classification report interactively but
does not persist the split indices, metric values, plots, parameters as JSON,
or a feature manifest. It serializes the model and a `shap.TreeExplainer` with
Joblib.

## API details

`main.py` exposes only `POST /analyze-url`. It accepts a Pydantic model with a
single unvalidated string, extracts features, returns the binary prediction,
an integer risk score, confidence, three SHAP-ranked feature names, all SHAP
contributions, and all extracted features. It has no health endpoint, response
schema, URL validation, controlled error responses, logging, configurable
paths, or lazy/model-load diagnostics.

## Existing problems and risks

1. **Training/inference feature drift.** Training uses the literal input URL
   for `url_length`, `dot_count`, `hyphen_count`, `digit_count`, HTTPS,
   entropy, and keyword count. The API first adds `http://` to schemeless URLs
   and, for two-label root domains, inserts `www.`. Consequently API
   `url_length`, `domain_length`, `dot_count`, `subdomain_count`, entropy,
   path parsing, and potentially other feature values can differ from the
   training values for the same submitted string.
2. **Domain leakage.** The random URL-level split can place different URLs
   from the same hostname in train and test. The datasets contain many such
   rows. This can overstate generalization to newly registered or unseen
   domains.
3. **Near-duplicate leakage.** Exact URL duplicates are dropped in training,
   but path/query variants and other related URLs are not grouped before the
   split.
4. **No final holdout protocol.** The notebook has a deterministic split but
   does not persist it. Re-running after cleaning, feature changes, or tuning
   means the intended final test boundary is not preserved. It has no
   group/domain split, cross-validation, unseen-domain test, temporal test, or
   external evaluation.
5. **No reproducibility controls.** There are no pinned requirements,
   configuration, dataset provenance/checksums, commands, persisted experiment
   metadata, or complete seed controls.
6. **Insufficient model comparison and diagnostics.** There are no baselines,
   threshold analysis, calibration consideration, PR-AUC/FPR/FNR reporting,
   ablation experiments, robustness tests, latency measurements, or persisted
   figures/results.
7. **Artifact portability risk.** Joblib pickles are version-sensitive. In the
   current environment, importing the API/model fails until compatible XGBoost
   is installed.
8. **Input-safety and operational gaps.** Invalid/blank URLs can become
   malformed feature rows; no request size limits, structured validation,
   logging, health check, or error handling are present.
9. **Unclear dataset role.** `raw_data.csv` is present but unused. Because it
   overlaps with `phishing.csv`, it must be de-overlapped before any
   cross-dataset claim.

## Recommended upgrades

Build a shared, deterministic URL-only feature extractor and a documented
cleaning/splitting pipeline. Preserve the original useful lexical features,
add selected structural/count/ratio features, and write a feature dictionary.
Use a fixed, domain-disjoint final test split; perform model selection and
feature ablations only on training folds. Run stratified CV, domain-aware
evaluation, offline URL perturbation testing, threshold analysis, SHAP plots,
and prediction/SHAP latency measurements. Add reproducible scripts,
configuration, tests, result artifacts, and a backwards-compatible FastAPI
endpoint with validated input and a health check.

No model-performance claim is supported by the repository's currently
persisted artifacts alone. In particular, same-dataset, unseen-domain,
cross-dataset, and robustness results must be generated by the upgraded
pipeline and reported separately.
