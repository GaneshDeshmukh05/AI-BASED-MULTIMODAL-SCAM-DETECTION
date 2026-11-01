# Team Responsibilities

Who owns what, verified against the actual code and artifacts in this repository.
Each module directory carries an owner's name in its folder name and `__init__.py` /
`README.md`, so ownership is visible directly in the tree.

---

## Responsibility matrix

| Team member | Module owned | Main responsibilities | Key files & directories | Inputs received | Outputs produced | Depends on | Integration status (verified) |
|---|---|---|---|---|---|---|---|
| **Audio Specialist** | `audio_model/` | Audio scam-call classification; voice anti-spoofing / deepfake detection; audio preprocessing & inference artifacts | `dataset/best_audio_scam_model.keras`, `LA/best_model.keras`, `dataset/` & `LA/` training notebooks, `run_live_audio_detection.py`, `test_load_models.py` | `.wav` audio (16 kHz mono preferred) | Scam probability + spoof probability per file (via adapters) | — | Models load and infer; **experimental on telephony audio** (poor separability on normal vs scam; spoof uncalibrated) |
| **URL Specialist** | `domain_intelligence/` | URL classification; 26-feature extraction; SHAP explanations; domain OSINT (DNS/WHOIS/TLS); brand lookalikes; threat feeds; standalone URL API | `models/xgboost_url_model.joblib`, `models/shap_url_explainer.joblib`, `models/feature_manifest.json`, `src/` (17 modules), `main.py`, `app/`, `experiments/`, `tests/` | URL / domain string | Phishing probability, SHAP factor list, composite OSINT score, risk adjustments | — | **Verified empirical performance**: 93.94% unseen-domain acc, 0.9845 ROC-AUC; 88.48% cross-dataset acc |
| **Text/Vision Specialist** | `text_and_screenshot_module/` + text/screenshot adapter logic | Text scam classification (trained model); text & screenshot datasets; screenshot threat detection | `models/text_scam_classifier.joblib`, `FINAL_TEXT_DATASET_1_3.zip` (20,000 rows), `FINAL_DATASET_1_3.zip` (8,256 screenshots), `test_dataset.py`; detection logic in `adapters/text_adapter.py` and `adapters/screenshot_adapter.py` | Text string; screenshot image path | Text scam score + detected tactics; screenshot visual risk score | — | Text model is a **hybrid system** (TF-IDF + Naive Bayes + rules; held-out accuracy unvalidated); screenshot path is **OCR content analysis — no trained CV model exists yet** |
| **Fusion Engineer** | `adapters/`, `fusion/`, `main.py`, `demo.py`, root `tests/`, `docs/` | Integrating the modality adapters; combining detector outputs; resolving conflicts; handling missing modalities & detector failures; computing the final risk score & verdict; generating evidence-grounded explanations; providing the unified interface for the backend | `adapters/*.py`, `fusion/schemas.py`, `fusion/fusion.py`, `fusion/llm_explainer.py`, `fusion/pipeline.py`, `main.py`, `demo.py`, `tests/unit/`, `tests/integration/` | `ModalityResult`s from the three modules (or raw detector dicts) | `MultimodalAnalysisResult` → JSON via `to_dict()`; CLI & demo output; `docs/API_CONTRACT.md` | Audio Module's, URL Module's and Text/Screenshot Module's detectors | 62 automated tests pass; live audio+URL+text+screenshot pipeline verified end-to-end |
| **Client Engineer** | Backend API & frontend (separate repository — **not in this repo**) | Backend API routes; frontend UI; wiring the UI to the backend and to this analysis engine | Not present in this repository | `MultimodalAnalysisResult.to_dict()` JSON from this pipeline | User-facing web application | Fusion Engineer's fusion pipeline (via `analyze_multimodal`) | Integration contract agreed in `docs/API_CONTRACT.md`; implementation lives in Client API's repo |

---

## Fusion Engineer's role in detail — Model Fusion & Explanation Layer

Fusion Engineer owns everything between the raw detectors and the backend:

1. **Adapters (`adapters/`)** — one per modality. Each translates a teammate detector's raw
   output (Keras probabilities, XGBoost + SHAP payloads, joblib classifier output, PIL image
   checks) into the single `ModalityResult` contract. The text and screenshot adapters
   implement Text/Screenshot Module's detection logic and credit him in their docstrings; the package itself is
   part of the integration layer. Adapters also own failure isolation: any detector exception
   becomes a `FAILED` result instead of crashing the pipeline.

2. **Schemas (`fusion/schemas.py`)** — the Pydantic contracts (`ModalityResult`,
   `FusionResult`, `ExplanationResult`, enums) that make heterogeneous detectors comparable.

3. **Fusion (`fusion/fusion.py`)** — confidence-weighted aggregation with per-modality
   weights; a **coverage penalty** so partial scans can't claim full certainty;
   **defense-in-depth escalation** so one strong verified threat (≥ 0.70 score, ≥ 0.75
   confidence) can't be averaged down; **conflict detection** between content detectors; the
   deliberate distinction between *synthetic voice* (`audio_spoof` → at most `SUSPICIOUS`
   alone) and *proven scam content* (→ `SCAM`).

4. **Explanation (`fusion/llm_explainer.py`)** — a deterministic, template-based explanation
   engine (zero-hallucination by construction) with optional Gemini/OpenAI generation wrapped
   in citation sanitisation and verdict-integrity checks. Missing-data disclaimers are always
   app-generated.

5. **Orchestration & interface (`fusion/pipeline.py`, `main.py`, `demo.py`)** — the
   `MultimodalPipeline` / `analyze_multimodal()` entry point, the CLI, and the demo.

6. **Quality (`tests/`, `docs/`)** — the automated regression test suite and this documentation system.

## Ownership nuances worth stating precisely

- `adapters/text_adapter.py` / `adapters/screenshot_adapter.py` live in Fusion Engineer's adapter
  package (integration layer) but implement **Text/Screenshot Module's** detection logic — Text/Vision Specialist is credited in
  their module docstrings, and the text model artifact is Text/Screenshot Module's.
- The screenshot capability is an OCR-based content analysis heuristic over Text/Screenshot Module's *datasets*; **no trained
  computer-vision model exists in this repository**. A trained CV model is future work.
- Client API's backend/frontend code is intentionally *not* in this repository; the only
  commitment here is the interface documented in [API_CONTRACT.md](API_CONTRACT.md).
