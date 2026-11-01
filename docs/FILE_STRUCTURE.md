# File Structure Guide

What every important folder and file is for, who owns it, what depends on it, and where its
tests live. Dataset contents are described at directory level — not record by record.

```
LLM integration/                        ← repository root
│
├── README.md                           # Main entry point — start here
├── requirements.txt                    # Root Python dependencies (verified against the dev venv)
├── .env.example                        # Template for optional environment variables
├── .gitignore                          # Excludes .venv, caches, .env secrets, archive/, tool junk
├── pytest.ini                          # Root pytest config (testpaths = tests)
├── pyrightconfig.json                  # Type-checking config for the integration layer
├── main.py                             # CLI scanner + live 3-scenario demo          [Fusion Engineer]
├── demo.py                             # Interactive 6-scenario walkthrough           [Fusion Engineer]
│
├── adapters/                           # Adapter / normalisation layer               [Fusion Engineer]
│   ├── __init__.py                     #     exports AudioAdapter, URLAdapter, TextAdapter, ScreenshotAdapter
│   ├── audio_adapter.py                #     wraps Audio Module's 2 Keras models (live spectrogram inference)
│   ├── url_adapter.py                  #     wraps URL Module's XGBoost/SHAP/OSINT service
│   ├── text_adapter.py                 #     Text/Screenshot Module's text classifier + tactic heuristics
│   └── screenshot_adapter.py           #     screenshot heuristic (filename/dimensions — no trained CV model)
│
├── fusion/                             # Decision fusion & explanation layer          [Fusion Engineer]
│   ├── __init__.py                     #     public API: analyze_multimodal, MultimodalPipeline, schemas
│   ├── schemas.py                      #     Pydantic contracts (ModalityResult, FusionResult, …)
│   ├── fusion.py                       #     ModelFusion: weighted fusion, escalation, conflicts
│   ├── llm_explainer.py                #     LLMExplainer: deterministic + optional Gemini/OpenAI
│   └── pipeline.py                     #     MultimodalPipeline orchestrator + MultimodalAnalysisResult
│
├── tests/                              # 62 automated tests                           [Fusion Engineer]
│   ├── unit/                           #     30 fast tests, no live I/O
│   │   ├── test_adapters.py            #         adapter translation & error handling
│   │   ├── test_fusion.py              #         10 fusion scenarios (conflict, escalation, missing, failed)
│   │   └── test_llm_explainer.py       #         grounding, fallback, injection resistance
│   └── integration/                    #     8 slower tests with live models / datasets
│       ├── test_multimodal_integration.py   #  live audio + URL end-to-end, JSON serialization
│       └── test_text_and_screenshot.py      #  Text/Screenshot Module's dataset integrity + fusion of text/screenshot
│
├── docs/                               # Documentation system (this folder)
│   ├── PROJECT_OVERVIEW.md             # What the project is, features, stack, team
│   ├── ARCHITECTURE.md                 # Component-by-component architecture
│   ├── DATA_FLOW.md                    # One input's journey through the system
│   ├── FILE_STRUCTURE.md               # This document
│   ├── TEAM_RESPONSIBILITIES.md        # Ownership matrix
│   ├── API_CONTRACT.md                 # Interface for Client API's backend (canonical)
│   ├── SETUP_AND_EXECUTION.md          # Install, configure, run everything
│   ├── TESTING_GUIDE.md                # How to run and extend the test suite
│   └── KNOWN_LIMITATIONS.md            # Honest limits — verified vs heuristic vs planned
│
├── scripts/
│   └── verify_installation.py          # Post-setup health check (imports + model artifacts)
│
├── archive/                            # Superseded backups & notes (git-ignored, safe to delete)
│
├── 🎧 audio_model/              # ═══ TEAMMATE MODULE — Audio Specialist (audio) ═══
│   ├── dataset/
│   │   ├── best_audio_scam_model.keras     # scam-call classifier (loaded by adapters/audio_adapter.py)
│   │   ├── audio_scam_detection_model.keras# earlier iteration kept for reference
│   │   ├── audio_scam_model_config.json
│   │   ├── Audio_Scam_Detection_Training_(1).ipynb  # training notebook
│   │   ├── SCAM_CALLS/                     # 24 real scam-call sample .wav files (test inputs)
│   │   └── NORMAL_CALLS/                   # 24 real normal-call sample .wav files (test inputs)
│   ├── LA/                                 # "Logical Access" anti-spoofing
│   │   ├── best_model.keras                # ASVspoof anti-spoofing model (loaded by audio_adapter.py)
│   │   ├── asvspoof2019_la_cnn.keras       # earlier iteration kept for reference
│   │   ├── preprocessing_config.json
│   │   ├── training_log.csv
│   │   └── ASVspoof2019_AntiSpoofing_Model2.ipynb   # training notebook
│   ├── run_live_audio_detection.py         # standalone live-audio runner feeding the fusion engine
│   ├── test_load_models.py                 # standalone "do both models load?" check
│   └── README.md
│
├── 🌐 domain_intelligence/       # ═══ TEAMMATE MODULE — URL Specialist (URL/OSINT) ═══
│   ├── models/                             # production artifacts: xgboost_url_model.joblib,
│   │                                       # shap_url_explainer.joblib, feature_manifest.json,
│   │                                       # best_xgboost_config.json
│   ├── src/                                # 17 modules: config, feature_engineering, preprocessing,
│   │                                       # model_service(+_services alias), domain_intelligence,
│   │                                       # domain_scorer, redirect_resolver, lookalike, threat_feeds,
│   │                                       # risk_scoring, evidence, explainability, llm_analyst,
│   │                                       # robustness, train, evaluate
│   ├── app/                                # standalone FastAPI service for URL analysis
│   ├── experiments/                        # research: ablation, baselines, cross-dataset, zero-day
│   ├── scripts/                            # retrain_with_feeds.py, update_threat_feed.py
│   ├── tests/                              # URL Module's own 9-file pytest suite (pytest.ini inside)
│   ├── results/                            # experiment outputs: CSVs, plots, shap/
│   ├── phishing.csv, raw_data.csv          # training datasets (paths pinned by src/config.py → must stay here)
│   ├── *.pkl                               # legacy first-generation artifacts (xgboost_model.pkl,
│   │                                       # shap_explainer.pkl, baseline models) — kept, untouched
│   ├── CONFIG.md, EXPERIMENTS.md, FEATURE_DICTIONARY.md,
│   │   LLM_INTEGRATION.md, PROJECT_AUDIT.md, RUN_GUIDE.md, README.md
│   ├── phishing_model_evaluation.ipynb, phishing_model_comparison_updated.ipynb,
│   │   Untitled.ipynb                      # research notebooks
│   └── requirements.txt, pytest.ini
│
├── 📱 text_and_screenshot_module/   # ═══ TEAMMATE MODULE — Text/Vision Specialist (text/screenshot) ═══
│   ├── models/
│   │   └── text_scam_classifier.joblib     # trained text scam classifier (loaded by text_adapter.py)
│   ├── Screenshot_Dataset-…-001/
│   │   └── Screenshot_Dataset/FINAL_DATASET_1_3.zip    # ~1.8 GB, 8,256 screenshots
│   │                                       #   (6,192 phishing / 2,064 legitimate)
│   ├── Text_Dataset-…-001/
│   │   └── Text_Dataset/FINAL_TEXT_DATASET_1_3.zip     # 20,000-row SMS/scam text corpus
│   │                                       #   (15,000 scam / 5,000 legitimate)
│   ├── test_dataset.py                     # dataset diagnostic & fusion walkthrough script
│   └── README.md
│
└── .venv/                              # Python virtual environment (do not commit; do not move)
```

---

## Module reference

### `adapters/` — integration layer (owner: Fusion Engineer)

- **What it does**: converts each teammate detector's raw output into the shared
  `ModalityResult` contract, and runs live inference on demand.
- **Why it exists**: the fusion engine must not know anything about Keras, XGBoost, or joblib —
  adapters are the firewalls between teammate code and the fusion core.
- **Entry points**: the four adapter classes; used via `from adapters import …`.
- **Depends on**: `fusion/schemas.py`; Audio Module's model files; URL Module's `src.model_service` /
  `src.llm_analyst` (via `sys.path` injection of `domain_intelligence/`); Text/Screenshot Module's
  `models/text_scam_classifier.joblib`.
- **Called by**: `fusion/pipeline.py`, `main.py`, `demo.py`, `audio_model/run_live_audio_detection.py`,
  `text_and_screenshot_module/test_dataset.py`.
- **Tests**: `tests/test_text_screenshot_regression.py` (+ live-URL test); adapter behaviour is also
  exercised indirectly by every other test file.
- **Why it must stay at the root**: `audio_adapter.py` and `url_adapter.py` locate the
  teammate module folders by walking up from the adapter file and matching directory names
  (`audio_model`, `domain_intelligence`); `text_adapter.py` resolves Text/Screenshot Module's model
  path relative to the adapter package. Moving this package or the teammate modules breaks
  those resolutions.

### `fusion/` — decision fusion & explanations (owner: Fusion Engineer)

- **What it does**: standardises results (`schemas.py`), fuses them with conflict resolution
  and defense-in-depth (`fusion.py`), produces grounded explanations (`llm_explainer.py`),
  and orchestrates the whole run (`pipeline.py`).
- **Entry point for other teams**: `from fusion import analyze_multimodal` — see
  [API_CONTRACT.md](API_CONTRACT.md).
- **Depends on**: `adapters/` (imported lazily inside `pipeline.analyze` so fusion itself
  stays testable without model files), Pydantic.
- **Called by**: everything — `main.py`, `demo.py`, both teammate runner scripts, Client API's
  future backend.
- **Tests**: `tests/unit/test_fusion.py`, `tests/unit/test_llm_explainer.py`,
  `tests/test_phase2a_reliability.py`.

### `audio_model/` (owner: Audio Specialist) — *location deliberately preserved*

- Two trained Keras models + training notebooks + 48 labelled sample WAVs + standalone runner
  scripts. The adapter references these exact directory names and file names
  (`dataset/best_audio_scam_model.keras`, `LA/best_model.keras`); the sample WAVs are
  referenced by relative path from `main.py`, `demo.py`, and the integration tests. The
  notebooks record the training procedure and must stay beside their datasets.
- **Tests**: loading is checked by `test_load_models.py`; live inference is exercised by
  `tests/test_phase2a_reliability.py`.

### `domain_intelligence/` (owner: URL Specialist) — *location deliberately preserved*

- Full research-to-production module: `src/` implementation, `models/` artifacts, FastAPI
  service (`main.py` + `app/`), experiments and results, its own test suite, and its own
  docs. `src/config.py` pins `PROJECT_ROOT` to this directory and expects `phishing.csv` /
  `raw_data.csv` there; the adapter imports it as package `src` after putting this directory
  on `sys.path`. Both constraints fix its location.
- **Tests**: URL Module's own `domain_intelligence/tests/` (run with
  `pytest domain_intelligence` or from inside the folder); adapter-level integration is
  covered by the root suite.

### `text_and_screenshot_module/` (owner: Text/Vision Specialist) — *location deliberately preserved*

- The trained text model artifact referenced by name from `adapters/text_adapter.py`
  (including a legacy `text and screenshot/` fallback alias), plus the two labelled dataset
  archives referenced by exact path from `tests/test_text_screenshot_regression.py` and
  `test_dataset.py`. The 1.8 GB screenshot archive is the training/test source for a future
  CV model — it is data, not clutter, and stays.

### Root files

| File | Purpose | Notes |
|---|---|---|
| `main.py` | CLI: `python main.py --audio … --url …` or bare for the 3-scenario live demo | Uses only relative dataset paths, so run from the repo root |
| `demo.py` | 6-scenario walkthrough: 5 mock/fusion scenarios + 1 fully live run | Same: run from repo root |
| `requirements.txt` | Root dependencies (expanded & verified against the dev venv) | URL Module's module keeps its own stricter `requirements.txt` |
| `pytest.ini` | Root test config (`testpaths = tests`) | URL Module's module has its own |
| `pyrightconfig.json` | Type-checks `fusion/`, `adapters/`, `tests/` only | Teammate modules are excluded deliberately |
| `.env` / `.env.example` | Local environment (only `PYTHONPATH=.` currently) | `.env` is git-ignored; never commit keys |
| `scripts/verify_installation.py` | Health check after setup: imports, model artifacts, quick fusion smoke test | See [SETUP_AND_EXECUTION.md](SETUP_AND_EXECUTION.md) |
| `archive/` | Superseded `.bak` files and the reorganisation request note | Git-ignored; safe to delete |

## Deliberate non-changes

The following were **intentionally left at their original paths** during reorganisation, and
why:

- `adapters/`, `fusion/`, `main.py`, `demo.py`, root `tests/` — working imports and the public
  interface documented to Client Engineer depend on these exact locations.
- The three teammate module directories — model-loading path resolution, `sys.path` injection,
  pinned dataset paths in `src/config.py`, and VS Code/pyright configs all reference them.
- `domain_intelligence/phishing.csv`, `raw_data.csv` and the legacy `.pkl` files —
  `src/config.py` and the experiment scripts expect them at the module root.
- `.venv/`, `.vscode/`, `pyrightconfig.json`, `pytest.ini` — environment/config, functioning.

A top-level `datasets/` or `configs/` directory was considered and rejected: datasets are
owned by, and path-pinned to, the teammate modules, and there is no cross-module shared
configuration yet.
