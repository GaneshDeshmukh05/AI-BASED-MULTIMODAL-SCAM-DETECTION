# Multimodal Scam Detection & LLM Defense Layer

An AI-powered system that analyses suspicious communications — **phone calls, links,
messages, and screenshots** — and returns a single, explainable verdict backed by evidence.

Four specialised detection models run in parallel, a decision-fusion engine combines their
outputs with conflict resolution and defense-in-depth escalation, and a grounded explanation
layer tells the user *what was detected and why* — citing only evidence the detectors
actually produced.

---

## 🎯 The problem

Modern scams are multi-channel: an **AI-cloned voice** on a call, a **phishing link** by SMS,
**pressure text** ("account suspended in 24 hours"), and a **fake login page**. Each signal
in isolation can look harmless. This project evaluates all available channels at once and
fuses their verdicts safely — one strong verified threat cannot be averaged down to "safe",
and one benign signal cannot green-light a conversation in which another detector found
fraud.

## 🏛️ Team & module ownership

This is a collaborative final-year engineering project. Each module directory is named after
and owned by its author:

| Member | Module | What it does |
|---|---|---|
| **Audio Specialist** | `audio_model/` | Two Keras models: scam-call classifier (`dataset/best_audio_scam_model.keras`) and voice anti-spoofing / deepfake detection (`LA/best_model.keras`, ASVspoof 2019) |
| **URL Specialist** | `domain_intelligence/` | 26-feature XGBoost phishing URL classifier with SHAP explanations, passive domain OSINT (DNS, WHOIS age, TLS certificates), brand-lookalike scanner, threat-feed matching, plus a standalone FastAPI URL service |
| **Text/Vision Specialist** | `text_and_screenshot_module/` | Labelled datasets (20,000-row scam text corpus; 8,256 screenshot images) and the trained text scam classifier (`models/text_scam_classifier.joblib`). Text & screenshot detection logic lives in `adapters/text_adapter.py` / `adapters/screenshot_adapter.py` |
| **Fusion Engineer** | `adapters/`, `fusion/`, `main.py`, `demo.py`, `tests/`, `docs/` | Model fusion & explanation layer: normalises every detector into one contract, fuses results with conflict resolution, generates evidence-grounded explanations, and exposes the unified interface used by the backend |
| **Client Engineer** | Backend API & frontend *(separate repository)* | Consumes this pipeline via the documented contract — see [docs/API_CONTRACT.md](docs/API_CONTRACT.md) |

Full responsibility matrix: [docs/TEAM_RESPONSIBILITIES.md](docs/TEAM_RESPONSIBILITIES.md)

## ⚙️ How it works

```
User inputs (audio file / URL / text / screenshot — any combination)
        ↓
Modality adapters (adapters/)            → normalise each detector into ModalityResult
        ↓
Detection modules (teammate models)      → Audio Specialist audio · URL Specialist URL/OSINT · Text/Vision Specialist text/screenshot
        ↓
Model Fusion (fusion/fusion.py)          → weighted risk, defense-in-depth escalation,
        ↓                                  conflict resolution, missing-data handling
Explanation engine (fusion/llm_explainer.py) → evidence-only, hallucination-free explanation
        ↓
MultimodalAnalysisResult.to_dict()       → one JSON document for the backend/frontend
```

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) · step-by-step flow:
[docs/DATA_FLOW.md](docs/DATA_FLOW.md)

Key fusion behaviours:

- **Defense-in-depth**: a detector scoring ≥ 0.70 with ≥ 0.75 confidence raises the overall
  risk to its own score — high-severity threats are never averaged away.
- **Conflict resolution**: contradictory content detectors trigger `CONFLICT_WARNING`.
- **Voice-impersonation nuance**: synthetic audio alone yields `SUSPICIOUS` (impersonation
  risk), not `SCAM` — cloned speech does not by itself prove scam intent.
- **Zero-hallucination explanations**: the deterministic engine is template-based; optional
  Gemini/OpenAI output is filtered so citations must match real detector evidence.

## 🧰 Technology stack

Python 3.10–3.13 · TensorFlow/Keras 3 (audio) · XGBoost + SHAP + scikit-learn (URL) ·
joblib (text model) · librosa + soundfile (spectrograms) · dnspython + python-whois +
tldextract (OSINT) · Pillow (images) · Pydantic v2 (contracts) · FastAPI (URL Module's standalone
service) · pytest (62 tests).

## 📂 Repository structure

```
LLM integration/
├── README.md, requirements.txt, .env.example, .gitignore
├── main.py                    # Unified CLI scanner + live demo scenarios
├── demo.py                    # Interactive 6-scenario walkthrough
├── adapters/                  # [Fusion Engineer] detector-output normalisation layer
├── fusion/                    # [Fusion Engineer] schemas, fusion engine, explanations, pipeline
├── tests/                     # [Fusion Engineer] 62 automated regression & integration tests across 4 suites
├── docs/                      # Documentation system (see below)
├── scripts/                   # verify_installation.py health check
├── audio_model/        # [Audio Specialist] audio models, training notebooks, sample calls
├── domain_intelligence/ # [URL Specialist] URL classifier, OSINT, experiments, own tests
└── text_and_screenshot_module/  # [Text/Vision Specialist] datasets + trained text model
```

Every file explained: [docs/FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md)

## 🚀 Quickstart

```powershell
# 1. Install (Windows/PowerShell; see docs for macOS/Linux)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Verify models & imports resolve
python scripts/verify_installation.py

# 3. Run the interactive walkthrough
python demo.py        # or: python main.py  (3 fully live scenarios)

# 4. Scan something specific
python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login"

# 5. Run the test suite
pytest
```

No API keys are required — the explanation engine runs offline by default.
Full instructions: [docs/SETUP_AND_EXECUTION.md](docs/SETUP_AND_EXECUTION.md)

### Programmatic API (for the backend team)

```python
from fusion import analyze_multimodal

result = analyze_multimodal(
    audio_path="call.wav",
    url="http://paypal.com-verify.online/login",
    text="URGENT your account will be suspended",
    screenshot_path="login_page.png",
    include_domain_intelligence=False,   # fast offline URL mode
)
result.print_summary()
payload = result.to_dict()               # JSON-ready dict for your API response
```

**Client Engineer → start here: [docs/API_CONTRACT.md](docs/API_CONTRACT.md)** — exact function
signature, parameters, real response JSON, verdict/risk enums, error semantics, and a
ready-to-use FastAPI route.

## 📚 Documentation

| Document | Contents |
|---|---|
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | Problem, objectives, features, stack, team |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Every component, in data-flow order, with source files |
| [docs/DATA_FLOW.md](docs/DATA_FLOW.md) | How one input travels through the system |
| [docs/FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md) | Every folder & file explained, with ownership |
| [docs/TEAM_RESPONSIBILITIES.md](docs/TEAM_RESPONSIBILITIES.md) | Responsibility matrix & integration status |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | Backend integration contract (canonical) |
| [docs/SETUP_AND_EXECUTION.md](docs/SETUP_AND_EXECUTION.md) | Install, configure, run, troubleshoot |
| [docs/DEMONSTRATION_RUNBOOK.md](docs/DEMONSTRATION_RUNBOOK.md) | Step-by-step evaluator presentation guide & live demo script |
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | Verified vs sample-tested vs heuristic vs planned |

## 🧪 Tests

62 automated regression and validation tests across 4 test suites, all passing:

```powershell
pytest                    # runs all 62 automated tests
pytest tests/test_phase1_regressions.py            # Phase 1 domain lookalike & anti-spoof regressions (7 tests)
pytest tests/test_phase2a_reliability.py           # Phase 2a fusion reliability & conflict resolution (6 tests)
pytest tests/test_phase3_text_screenshot_fixes.py   # Phase 3 text/screenshot heuristic & pipeline fixes (30 tests)
pytest tests/test_text_screenshot_regression.py    # Text & screenshot calibrated regression suite (19 tests)
```

URL Module's module additionally ships its own suite: `pytest domain_intelligence`.
Details: [docs/DEMONSTRATION_RUNBOOK.md](docs/DEMONSTRATION_RUNBOOK.md)

## ⚠️ Current limitations (honest summary)

- **URL model performance is strictly benchmarked on unseen domains**: 93.94% accuracy and 0.9845 ROC-AUC
  on the primary zero-day holdout; 88.48% accuracy on external cross-dataset evaluation. Conventional random
  splits (94.44%) contain domain leakage and are reported only for comparison.
- **Text detection is an unvalidated hybrid system**: Combines TF-IDF, Multinomial Naive Bayes, and heuristic
  rules; informal training-set claims of ~98% accuracy are unverified on held-out test data.
- **Screenshot detection is OCR content analysis**: Windows Native OCR extracts textual indicators (credentials,
  advance-fee prize claims) and analyzes geometry; no trained computer-vision model exists in this repository.
- **Audio detection is experimental and unvalidated for telephony**: Acoustic models show near-identical scores
  on normal and scam calls (~0.87–0.88), and ASVspoof produces false positives on telephony audio. Audio spoofing
  is excluded from overall risk scoring (`is_validated = False`).
- **The default explainer is deterministic rule-based**, not a locally running generative LLM (by design,
  for zero-hallucination guarantees; optional Gemini/OpenAI generation is supported).
- TensorFlow on native Windows is CPU-only; the first audio analysis takes ~10–20 s while
  models load.
- Client API's backend/frontend are separate — no end-user product exists in this repository.

Full details: [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md)
