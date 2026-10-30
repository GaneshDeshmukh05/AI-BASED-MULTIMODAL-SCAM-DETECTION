# Project Overview

**Multimodal Scam Detection & LLM Defense Layer**

An AI system that analyses a suspicious communication — a phone call (audio), a link (URL),
a message (text), or a screenshot — and returns a single, explainable verdict: is this a scam,
and *why*?

---

## Problem statement

Scams no longer arrive through a single channel. A modern attack may combine:

- an **AI-cloned voice** on a phone call (deepfake audio),
- a **phishing link** sent by SMS or chat,
- **pressure text** ("your account will be suspended in 24 hours"),
- and a **fake login page** that mimics a real bank or PayPal.

Each individual signal can look harmless in isolation. A voice may sound genuine but be
synthetic; a text may look routine; a URL may hide behind a familiar brand name. Defending
against this requires looking at **all available channels at once** and combining their
verdicts safely.

## Objectives

1. **Detect threats across four modalities** — audio (scam calls + synthetic voice), URLs
   (phishing + domain reputation), text (smishing / social engineering), and screenshots
   (fake login UIs).
2. **Fuse detector outputs intelligently** — weighted risk aggregation, conflict resolution,
   defense-in-depth escalation so one strong verified threat cannot be averaged away to "safe".
3. **Explain every verdict without hallucinating** — the explanation engine may only cite
   evidence actually produced by the detectors, never invented details.
4. **Expose one clean interface** — a single Python function that a backend (Client API's part)
   can call with any combination of inputs and receive JSON back.

## Main features

| Feature | Where | Status |
|---|---|---|
| Scam-call audio classification (Keras CNN, log-mel spectrograms) | `audio_model/` | **Experimental** (poor telephony separability) |
| Voice anti-spoofing / deepfake detection (ASVspoof 2019 CNN) | `audio_model/` | **Experimental** (unvalidated on telephony, `is_validated=False`) |
| 26-feature XGBoost phishing URL classifier + SHAP attribution | `domain_intelligence/` | **Verified**: 93.94% unseen-domain acc, 0.9845 ROC-AUC |
| Passive domain OSINT: DNS, WHOIS age, TLS certificate, brand lookalikes | `domain_intelligence/` | Implemented (offline fallback supported) |
| Threat-feed matching for known-phishing URLs | `domain_intelligence/` | Implemented (needs network) |
| Hybrid Text/SMS scam detection (TF-IDF + Naive Bayes + heuristic rules) | `adapters/text_adapter.py`, model in `text_and_screenshot_module/` | **Hybrid system** (unvalidated on held-out test data) |
| Screenshot analysis (OCR content analysis & heuristic rules) | `adapters/screenshot_adapter.py` | **OCR heuristics — no trained CV model** |
| Weighted multimodal fusion with conflict resolution | `fusion/fusion.py` | Implemented, unit tested |
| Grounded explanation engine (deterministic + optional Gemini/OpenAI) | `fusion/llm_explainer.py` | Implemented, unit tested |
| Unified CLI, demo walkthrough, 62 automated tests | `main.py`, `demo.py`, `tests/` | Implemented, all passing |

## High-level architecture

```
User inputs (audio file / URL / text / screenshot)
        ↓
   Modality adapters  (adapters/)          ← normalise every detector into ModalityResult
        ↓
   Detection modules (teammate models)     ← Audio Specialist audio · URL Specialist URL/OSINT · Text/Vision Specialist text/screenshot
        ↓
   Model Fusion (fusion/fusion.py)         ← weighted risk, conflict checks, escalation
        ↓
   Explanation engine (fusion/llm_explainer.py) ← grounded, evidence-only explanations
        ↓
   MultimodalAnalysisResult.to_dict()      ← one JSON document
        ↓
   Backend API + frontend (Client Engineer — separate repository)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for component detail and [DATA_FLOW.md](DATA_FLOW.md)
for a step-by-step walkthrough of one analysis.

## Technology stack

- **Language**: Python 3.13 (developed and tested on Windows; CPU-only TensorFlow)
- **Deep learning**: TensorFlow / Keras 3 (audio models), XGBoost (URL model)
- **Classical ML / NLP**: scikit-learn, joblib (text classifier), SHAP (URL explanations)
- **Signal processing**: librosa, soundfile (log-mel spectrograms)
- **OSINT**: dnspython, python-whois, tldextract, requests
- **Schemas & API**: Pydantic v2; FastAPI + uvicorn for URL Module's standalone URL service
- **Testing**: pytest (62 unit & regression tests)

## Team

| Member | Module | Summary |
|---|---|---|
| **Audio Specialist** | `audio_model/` | Audio scam-call classifier & voice anti-spoofing models |
| **URL Specialist** | `domain_intelligence/` | Phishing URL classifier, SHAP, domain OSINT |
| **Text/Vision Specialist** | `text_and_screenshot_module/` + text/screenshot adapters | Text scam model & datasets; screenshot datasets & heuristic |
| **Fusion Engineer** | `adapters/`, `fusion/`, `main.py`, `demo.py`, `tests/`, `docs/` | Integration, decision fusion, explanations, unified interface |
| **Client Engineer** | Backend API & frontend (separate repository) | Consumes this pipeline via [API_CONTRACT.md](API_CONTRACT.md) |

Details in [TEAM_RESPONSIBILITIES.md](TEAM_RESPONSIBILITIES.md).

## Start here

- Setup & running: [SETUP_AND_EXECUTION.md](SETUP_AND_EXECUTION.md)
- Full architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- Where every file lives: [FILE_STRUCTURE.md](FILE_STRUCTURE.md)
- Honest limitations: [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md)
