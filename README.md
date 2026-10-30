# Multimodal Scam Detection System

An AI-powered security system that detects scams across **phone calls, web links, text messages, and screenshots**. 

Instead of checking only one type of input, the system checks all available channels together. It combines findings from multiple machine learning models and gives a single, clear verdict with plain-English explanations.

---

## The Problem

Modern scams use multiple channels at the same time:
* A scammer might call you using a **cloned voice**.
* They follow up with an **SMS phishing link**.
* The message uses **urgent text** ("Your account will be closed in 24 hours").
* The link opens a **fake login page** to steal your password.

If you only check the text message, it might look like a normal security alert. This project evaluates all pieces together. If even one part is a confirmed scam, the system will not let it pass as safe.

---

## Team and Modules

This is a collaborative final-year engineering project. Each module is handled by a team role:

| Role | Directory | Responsibility |
|---|---|---|
| **Audio Specialist** | `audio_model/` | Scam call detection model and voice anti-spoofing / deepfake detection model. |
| **URL Specialist** | `domain_intelligence/` | 26-feature XGBoost phishing link classifier, domain checks (DNS, WHOIS age, SSL certificates), lookalike domain detection, and FastAPI service. |
| **Text/Vision Specialist** | `text_and_screenshot_module/` | Text scam classifier (trained on 20,000 messages) and screenshot dataset. Adapters live in `adapters/`. |
| **Fusion Engineer** | `adapters/`, `fusion/`, `main.py`, `demo.py`, `tests/`, `docs/` | Combines outputs from all models, handles conflicts, creates clear explanations, and manages the test suite. |
| **Client Engineer** | Separate repository | Connects the web frontend and mobile backend to our pipeline using [docs/API_CONTRACT.md](docs/API_CONTRACT.md). |

Full responsibility details: [docs/TEAM_RESPONSIBILITIES.md](docs/TEAM_RESPONSIBILITIES.md)

---

## How It Works

```
User Inputs (Audio / URL / Text / Screenshot)
       ↓
Modality Adapters (adapters/)
  Converts raw model outputs into a common format
       ↓
Detection Models
  Audio CNN · XGBoost URL Classifier · Text Naive Bayes · Screenshot OCR
       ↓
Decision Fusion (fusion/fusion.py)
  Combines risk scores using defense-in-depth rules and flags conflicts
       ↓
Explanation Engine (fusion/llm_explainer.py)
  Generates a clear explanation citing only real evidence
       ↓
Final Result (JSON / CLI output)
```

Key rules used by the fusion engine:
1. **Safety First (Defense-in-Depth)**: If any detector finds a confirmed, high-risk threat (score >= 0.70), the overall result is flagged as high risk. A safe link cannot cancel out a dangerous scam call.
2. **Conflict Warning**: If one detector says "Scam" and another says "Safe", the system alerts the user that findings conflict.
3. **Evidence-Based Explanations**: The explanation only mentions facts that the detectors actually found. It will never invent reasons.

---

## Technology Stack

* **Programming Language**: Python 3.10 – 3.13
* **Audio Models**: TensorFlow / Keras 3, Librosa, SoundFile
* **URL Phishing Model**: XGBoost, Scikit-learn, SHAP (for feature explanations)
* **Domain OSINT**: dnspython, python-whois, tldextract
* **Text & Screenshot**: Scikit-learn (TF-IDF + Naive Bayes), Pillow, Windows Native OCR
* **API & Schemas**: Pydantic v2, FastAPI, Uvicorn
* **Testing**: Pytest (62 automated tests)

---

## Repository Structure

```
LLM integration/
├── README.md, requirements.txt, .env.example, .gitignore
├── main.py                    # Command-line scanner and live demo
├── demo.py                    # 6-scenario walkthrough
├── adapters/                  # Normalizes output from each model
├── fusion/                    # Fusion engine, schemas, and explanations
├── tests/                     # 62 automated regression and reliability tests
├── docs/                      # Complete documentation
├── scripts/                   # System verification script
├── audio_model/               # Audio models, training notebooks, and sample audio
├── domain_intelligence/       # URL classifier, OSINT tools, and experiments
└── text_and_screenshot_module/ # Text model and datasets
```

Every file explained: [docs/FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md)

---

## Quickstart

### 1. Set up the virtual environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Verify installation and model files

```powershell
python scripts/verify_installation.py
```
> Checks all packages, model files, and runs a quick smoke test.

### 3. Run the interactive demo

```powershell
python demo.py
```
> Runs 6 different scam detection scenarios.

### 4. Scan your own inputs

```powershell
# Scan an audio file and a URL together:
python main.py --audio audio_model/dataset/SCAM_CALLS/scam_22.wav --url "http://paypal.com-verify.online/login"

# Fast scan without live DNS lookups:
python main.py --url "http://paypal.com-verify.online/login" --no-intel
```

### 5. Run the tests

```powershell
pytest -v
```

Full setup instructions: [docs/SETUP_AND_EXECUTION.md](docs/SETUP_AND_EXECUTION.md)

---

## Programmatic API (For Developers)

You can call the fusion pipeline directly in Python:

```python
from fusion import analyze_multimodal

result = analyze_multimodal(
    audio_path="call.wav",
    url="http://paypal.com-verify.online/login",
    text="URGENT: Your bank account will be suspended.",
    screenshot_path="login_page.png",
    include_domain_intelligence=False,   # Fast offline mode
)

result.print_summary()
json_data = result.to_dict()  # Ready for REST API responses
```

For full API specifications, see [docs/API_CONTRACT.md](docs/API_CONTRACT.md).

---

## Documentation Index

| Document | Description |
|---|---|
| [docs/PROJECT_OVERVIEW.md](docs/PROJECT_OVERVIEW.md) | High-level project summary, goals, and team roles |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Detailed architecture and component design |
| [docs/DATA_FLOW.md](docs/DATA_FLOW.md) | How an input moves through the entire system |
| [docs/FILE_STRUCTURE.md](docs/FILE_STRUCTURE.md) | Explanation of every file and folder |
| [docs/TEAM_RESPONSIBILITIES.md](docs/TEAM_RESPONSIBILITIES.md) | Responsibilities and integration status |
| [docs/API_CONTRACT.md](docs/API_CONTRACT.md) | API format, data types, and sample responses |
| [docs/SETUP_AND_EXECUTION.md](docs/SETUP_AND_EXECUTION.md) | Step-by-step installation and troubleshooting guide |
| [docs/DEMONSTRATION_RUNBOOK.md](docs/DEMONSTRATION_RUNBOOK.md) | College viva and evaluator presentation guide |
| [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md) | Full breakdown of model evaluation and known limitations |

---

## Automated Tests

We maintain **62 automated tests** across 4 suites. All tests pass:

```powershell
pytest                                             # Runs all 62 tests
pytest tests/test_phase1_regressions.py            # Domain lookalike and audio feature tests (7 tests)
pytest tests/test_phase2a_reliability.py           # Fusion conflict and reliability tests (6 tests)
pytest tests/test_phase3_text_screenshot_fixes.py   # Text and screenshot scam logic tests (30 tests)
pytest tests/test_text_screenshot_regression.py    # Screenshot OCR and text regression tests (19 tests)
```

The URL module also includes its own standalone test suite: `pytest domain_intelligence`.

---

## Project Limitations (Honest Summary)

To maintain scientific integrity for our college evaluation, here is the honest status of each component:

* **URL Model (Reliable)**: Tested on 50,571 test URLs from 25,678 brand-new, unseen domains. Achieved **93.94% accuracy** and **0.9845 ROC-AUC**. On a separate external dataset, it achieved **88.48% accuracy**.
* **Text Model (Hybrid Rules + ML)**: Combines TF-IDF, Naive Bayes, and keyword rules. Claims of ~98% accuracy from training are not supported by formal held-out tests.
* **Screenshot Detection (OCR-Based)**: Uses Windows Native OCR to read text on the screen and searches for fake login forms and lottery/prize scams. It is not an end-to-end computer vision neural network.
* **Audio Detection (Experimental)**: Normal phone calls and scam calls currently receive similar scores (~0.87–0.88). The anti-spoofing model also produces false alarms on phone audio. Because of this, audio deepfake scores are marked as unvalidated (`is_validated = False`) and do not drive high-confidence scam verdicts by themselves.
* **Explanation Engine**: Uses a reliable rule-based template by default so it never hallucinates. Optional Gemini / OpenAI explanations are supported if API keys are provided.
* **Backend Integration**: This repository contains the detection and fusion engine. The user interface and mobile app are developed in a separate repository.

For full details, see [docs/KNOWN_LIMITATIONS.md](docs/KNOWN_LIMITATIONS.md).
