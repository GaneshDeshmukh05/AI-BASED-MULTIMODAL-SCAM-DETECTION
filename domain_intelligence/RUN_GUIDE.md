# Run Guide

The project currently has a trained model bundle in `models/`, so you can run
the API immediately without retraining.

## 1. Open PowerShell in the project folder

```powershell
Set-Location D:\URL
```

## 2. Install the tested dependencies

Optional but recommended: create an isolated virtual environment first.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 3. Start the FastAPI server

```powershell
uvicorn main:app --host 127.0.0.1 --port 8000
```

Keep that terminal open while using the API. Interactive documentation is at:

```text
http://127.0.0.1:8000/docs
```

## 4. Check that the model is healthy

Open a second PowerShell terminal and run:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected fields include `status: ok`, `model_loaded: true`, `feature_count: 26`,
and the selected operating threshold.

## 5. Analyze a URL

```powershell
$body = @{ url = 'https://example.com/login?ref=mail' } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/analyze-url -Method Post -ContentType 'application/json' -Body $body
```

The response includes a prediction, phishing probability, confidence, risk
level, extracted URL features, and signed SHAP model explanations. By default it
also includes `domain_intelligence`, `ssl_analysis` and `risk_adjustments` from
DNS/WHOIS/TLS lookups, which adjust `risk_score`/`prediction` (the raw model
verdict stays in `ml_prediction`). Page content is never fetched. For a purely
offline result:

```powershell
$body = @{ url = 'https://example.com/login?ref=mail'; include_domain_intelligence = $false } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/analyze-url -Method Post -ContentType 'application/json' -Body $body
```

## 6. Run the tests

```powershell
pytest -q
```

## 7. Reproduce all experiments (optional)

This retrains the XGBoost model and regenerates result CSVs/plots. It takes
several minutes with the supplied datasets.

```powershell
python -m experiments.run_all --tuning-iterations 8
```

The outputs are written to `models/` and `results/`. See `EXPERIMENTS.md` for
the leakage controls and interpretation rules.

## Stop the server

Press `Ctrl+C` in the terminal that started Uvicorn. If that terminal is no
longer available, first confirm that port 8000 belongs to this project, then:

```powershell
$server = Get-NetTCPConnection -LocalPort 8000 -State Listen
Stop-Process -Id $server.OwningProcess
```

## Troubleshooting

* **Port 8000 is already in use:** use the existing server, stop the process
  after confirming it is Uvicorn for this project, or run with `--port 8001`.
* **`model bundle is unavailable`:** run the experiment command in step 7.
* **`uvicorn` is not recognized:** use
  `python -m uvicorn main:app --host 127.0.0.1 --port 8000` after installing
  the requirements.
* **`domain_intelligence.lookup_notes` mentions "not installed":** run
  `python -m pip install -r requirements.txt` (adds `python-whois`, `dnspython`).
* **Slow responses:** WHOIS can be slow; lower `PHISHING_INTEL_TIMEOUT` or set
  `PHISHING_DOMAIN_INTEL=0`.
* **Validation error 422:** submit a non-empty HTTP(S) URL or schemeless
  hostname; unsupported schemes, whitespace, and malformed ports are rejected.
