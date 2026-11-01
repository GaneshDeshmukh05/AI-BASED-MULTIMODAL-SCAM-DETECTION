# Configuration

The pipeline is deterministic by default. `src/config.py` defines the seed
(`42`), 20% holdout target, three domain-aware CV folds, artifact paths, and
dataset paths.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `PHISHING_PRIMARY_DATASET` | `phishing.csv` | Primary labeled URL dataset used for tuning/training. PhiUSIIL labels are mapped as `0=phishing`, `1=legitimate`. |
| `PHISHING_CROSS_DATASET` | `raw_data.csv` | Compatible two-column external dataset used only for de-overlapped cross-dataset evaluation. |
| `PHISHING_MODEL_PATH` | `models/xgboost_url_model.joblib` | Final trained XGBoost bundle read by the API. |
| `PHISHING_EXPLAINER_PATH` | `models/shap_url_explainer.joblib` | Matching TreeExplainer artifact read by the API. |

| `PHISHING_DOMAIN_INTEL` | `1` | Server default for DNS/WHOIS/TLS enrichment (`0`/`false` disables). A request can override it with `include_domain_intelligence`. |
| `PHISHING_INTEL_TIMEOUT` | `5.0` | Overall wall-clock budget in seconds for the parallel DNS/WHOIS/TLS lookups. |

Scoring constants (new-domain window and penalty, certificate penalty,
established/mature domain thresholds and factors) are defined in `src/config.py`.
They affect only the post-model composite score, not training.

Example:

```powershell
$env:PHISHING_PRIMARY_DATASET = 'D:\datasets\primary_urls.csv'
$env:PHISHING_CROSS_DATASET = 'D:\datasets\external_urls.csv'
python -m experiments.run_all
```

Input datasets must have a URL column (case-insensitive) and a label/type/class/
target/result column. Generic labels accepted are `phishing`, `malicious`,
`bad`, `1`, `legitimate`, `benign`, `good`, and `0`. Unrecognized or invalid
records are documented in `results/dataset_cleaning_report.json` and excluded.

Do not change feature definitions or class mapping after training without
re-running the full experiment package. The API rejects a model whose saved
feature schema differs from the source code's schema.
