# Setup & Execution Guide

Everything needed to install, configure, run, and verify the system.
Commands are written for **Windows / PowerShell** from the repository root — on macOS/Linux
the only difference is activating the venv (`source .venv/bin/activate`).

---

## 1. Prerequisites

- **Python 3.10–3.13** (development used 3.13). TensorFlow on native Windows is CPU-only —
  that is fine, inference runs comfortably on CPU.
- ~4 GB free disk for dependencies + the audio sample data (the 1.8 GB screenshot dataset
  archive is optional for running the system; it is needed only for Text/Screenshot Module's dataset tests).

## 2. Install dependencies

```powershell
# from the repository root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` covers the fusion layer, adapters, CLI, demo, and tests. It includes
TensorFlow/Keras (audio models), librosa/soundfile (audio features), XGBoost/SHAP/scikit-learn
(URL model), Pillow (screenshots), pandas (dataset tests), and pytest.

> URL Module's module has its own, stricter `domain_intelligence/requirements.txt` (pinned
> versions, plus FastAPI/uvicorn for its standalone service). Install it only if you want to
> run that module's research experiments or its standalone API:
> `python -m pip install -r domain_intelligence/requirements.txt`

## 3. Environment configuration

```powershell
Copy-Item .env.example .env
```

`.env` is optional — the system runs fully offline with zero keys. Only set:

| Variable | Effect |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Enables LLM-generated explanations when `explainer_provider` is `gemini`/`auto` |
| `OPENAI_API_KEY` | Same for `openai`/`auto` (OpenAI-compatible endpoints configurable in `fusion/llm_explainer.py`) |
| `PHISHING_*` | Optional overrides for URL Module's module (dataset/model paths, domain-intel toggle) |

Without keys, every provider choice falls back to the offline deterministic explainer.
**Never commit `.env`** — it is git-ignored by `.gitignore`.

## 4. Verify the installation

```powershell
python scripts/verify_installation.py
```

Checks imports, locates and loads all four model artifacts (two Keras audio models, XGBoost +
SHAP bundle, text classifier), runs a quick offline URL inference and a fusion smoke test.

## 5. Run the system

### 5.1 Interactive demo (recommended first run)

```powershell
python demo.py        # 6 scenarios: adapter-level fusion demos + 1 fully live run
python main.py        # 3 fully live scenarios (audio+URL, conflict case, audio-only)
```

Both must be run **from the repository root** (they reference sample audio by relative path).

### 5.2 Targeted CLI scans

```powershell
# Real audio + real phishing URL (full OSINT):
python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login"

# Fast offline URL mode (no DNS/WHOIS/TLS):
python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login" --no-intel

# Structured JSON output (what the backend receives):
python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login" --json

# Text / screenshot / provider options:
python main.py --text "URGENT your account will be suspended, send the OTP" 
python main.py --screenshot path\to\page.png
python main.py --url "https://example.com" --provider gemini
```

### 5.3 Programmatic API (what Client API's backend calls)

```python
from fusion import analyze_multimodal

result = analyze_multimodal(
    audio_path="audio_model/dataset/SCAM_CALLS/scam_22.wav",
    url="http://paypal.com-account-verify.online/login",
    include_domain_intelligence=False,
)
result.print_summary()
payload = result.to_dict()   # JSON-ready
```

Full details: [API_CONTRACT.md](API_CONTRACT.md).

### 5.4 Teammate standalone entry points

```powershell
# Audio Specialist: live audio detection + fusion
python audio_model/run_live_audio_detection.py audio_model/dataset/SCAM_CALLS/scam_22.wav
python audio_model/test_load_models.py

# URL Specialist: standalone URL scanner / API service (install his requirements first)
python domain_intelligence/main.py https://paypal.com-verify.online
uvicorn app.main:app --app-dir domain_intelligence --port 8000

# Text/Vision Specialist: dataset diagnostic suite
python text_and_screenshot_module/test_dataset.py
```

## 6. Run the tests

```powershell
pytest                 # runs all 62 tests across all 4 suites
pytest tests/test_phase1_regressions.py            # Phase 1 domain lookalike & anti-spoof regressions (7 tests)
pytest tests/test_phase2a_reliability.py           # Phase 2a fusion reliability & conflict resolution (6 tests)
pytest tests/test_phase3_text_screenshot_fixes.py   # Phase 3 text/screenshot heuristic & pipeline fixes (30 tests)
pytest tests/test_text_screenshot_regression.py    # Text & screenshot calibrated regression suite (19 tests)
```

Details: [DEMONSTRATION_RUNBOOK.md](DEMONSTRATION_RUNBOOK.md).

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `ModuleNotFoundError: fusion` / `adapters` | Not running from the repo root, or venv not activated. `cd` to the root, or set `PYTHONPATH=.` (see `.env.example`). |
| `ModuleNotFoundError: src` from the URL adapter | `domain_intelligence/` was moved/renamed — the adapter locates that exact directory name. |
| Keras/TF errors on model load | Model artifacts missing (`audio_model/dataset/best_audio_scam_model.keras`, `LA/best_model.keras`). Run `python audio_model/test_load_models.py`. |
| URL OSINT hangs | Live DNS/WHOIS can be slow on some networks — use `--no-intel` / `include_domain_intelligence=False`. |
| First audio request is slow | TensorFlow + model load cost (~10–20 s), cached in-process afterwards. |
| `librosa`/`soundfile` import errors | `pip install -r requirements.txt` was not run into the active venv. |
