# Phishing URL & Domain Intelligence Module

An explainable and robust phishing URL threat detector built around 26 engineered lexical and structural features, XGBoost, SHAP explanations, and passive domain OSINT (DNS, WHOIS age, TLS/SSL).
WHOIS registration age, TLS certificate) that adjusts the final risk score; see
[Domain intelligence](#domain-intelligence) below. No browser, CNN, or LLM
dependencies are used.

## What is preserved and improved

`POST /analyze-url` remains the inference endpoint and `uvicorn main:app`
remains the server command. The former notebook training procedure and 12
useful lexical features are preserved conceptually, but training and serving
now share a single extractor and the trained research model has a documented
26-feature schema. The original `xgboost_model.pkl` and `shap_explainer.pkl`
are left untouched as legacy artifacts; the upgraded model is written under
`models/`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The checked-in data paths default to `phishing.csv` (primary PhiUSIIL-style
source) and `raw_data.csv` (external two-column source). Configure replacements
as described in [CONFIG.md](CONFIG.md).

## Reproduce the full experiment package

```powershell
python -m experiments.run_all --tuning-iterations 8
```

This command cleans data, creates both random and domain-disjoint splits,
tunes XGBoost only on training-domain folds, selects an operating threshold
from training OOF scores, trains the final model, and writes actual outputs to
`models/` and `results/`. It may take several minutes on the supplied datasets.
Use a smaller number such as `--tuning-iterations 2` only for a smoke test; it
is not the recommended research run.

Key outputs are:

* `models/best_xgboost_config.json` — selected parameters, CV score, threshold, and schema.
* `results/model_comparison.csv` / `.png` — baseline results from the same fixed unseen-domain test set.
* `results/ablation_results.csv` / `.png` — cumulative feature-group ablation.
* `results/zero_day_results.csv` — fixed unseen-domain holdout results.
* `results/cross_dataset_results.csv` — external dataset results after removing all primary-domain overlaps.
* `results/robustness_results.csv` — offline perturbation outcomes; no URLs are visited.
* `results/threshold_results.csv`, `latency_results.csv`, curves, confusion matrix, and `results/shap/` plots.

Do not treat a successful run as evidence of causal feature effects or of
superiority to other papers. See [EXPERIMENTS.md](EXPERIMENTS.md) for the
protocol and limitations.

## Run the API

Run the experiments once so that the compatible model bundle exists, then:

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/analyze-url -Method Post -ContentType 'application/json' -Body '{"url":"https://example.com/login"}'
```

The response contains the phishing probability, selected operating threshold,
risk level, extracted features, signed SHAP contributions, and top model risk
and legitimate factors. SHAP is an explanation of the model's score, not a
causal conclusion about the URL.

## Domain intelligence

`POST /analyze-url` first scores the URL with the model, then (unless disabled)
looks up DNS, WHOIS and TLS data for the host and combines them with the model
probability into a composite score. The model itself never sees these signals —
the training data has no such columns — so `phishing_probability` and all SHAP
fields are always the raw model output.

| Signal | Effect on the 0–100 composite score |
| --- | --- |
| Domain registered < 30 days ago | +15 |
| `https://` URL whose certificate **fails verification** (expired, self-signed, wrong host, untrusted CA) | +20 |
| Domain > 10 years old **and** valid certificate | × 0.3 |
| Domain > 5 years old **and** valid certificate | × 0.6 |
| Lookup failed / timed out / not applicable | no change |

The reduction for old domains is **not** applied to shared-hosting or
user-content hosts (e.g. `github.io`, `docs.google.com`, `sharepoint.com`),
since an old parent domain says nothing about a tenant's page. The list lives in
`src/domain_intelligence.py` (`SHARED_HOSTING_SUFFIXES`) and is a heuristic, not
a reputation feed. A phishing page on a *compromised* old site with a valid
certificate can still be discounted; domain age is not proof of safety.

The decision is `composite >= operating_threshold`. With no adjustments it is
identical to the model's own verdict. The response keeps the model-only verdict
as `ml_prediction` / `ml_risk_score`, lists every applied adjustment in
`risk_adjustments`, and returns the raw `domain_intelligence` and `ssl_analysis`
objects. `composite_risk_score` mirrors `risk_score` for clients of the earlier
domain-intelligence service (e.g. the browser extension).

**Network behaviour and privacy.** Enrichment is *not* offline. For each
request the server may send DNS queries, a WHOIS query to the registry, and open
a TLS connection to the submitted host on port 443 (certificate only; no HTTP
request is sent and page content is never fetched). Consequently:

* only public hosts are contacted — loopback/private/link-local addresses, IP
  literals and single-label hostnames are skipped, and the TLS port is always
  443, so the endpoint cannot be used to probe the server's internal network;
* results are cached briefly and bounded by `PHISHING_INTEL_TIMEOUT`;
* send `{"include_domain_intelligence": false}` (or set
  `PHISHING_DOMAIN_INTEL=0`) for a purely offline, model-only result.

## Test

```powershell
pytest -q
```

The tests cover URL validation/parsing, feature extraction, cleaning/split
leakage protection, model-service prediction behavior, and API validation.
