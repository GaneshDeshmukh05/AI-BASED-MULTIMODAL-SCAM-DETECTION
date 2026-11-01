# System Architecture

This document explains every component of the system, in the order data flows through it,
with the exact source files that implement each part.

```
User inputs
    ↓ (audio file, URL string, text string, screenshot file — any combination)
Available modality adapters                       adapters/*.py
    ↓
Text / URL / Domain / Audio / Screenshot analysis  teammate modules
    ↓
Normalized detector results (ModalityResult)       fusion/schemas.py
    ↓
Model Fusion                                       fusion/fusion.py
    ↓
Conflict resolution and risk assessment            fusion/fusion.py
    ↓
Evidence-grounded explanation                      fusion/llm_explainer.py
    ↓
Structured JSON result                             fusion/pipeline.py (to_dict)
    ↓
Main backend and frontend integration              Client API's repository (see docs/API_CONTRACT.md)
```

---

## 1. Adapters — the normalisation layer (`adapters/`, Fusion Engineer)

Each adapter wraps one teammate's detector and converts its raw output into a single
standardised Pydantic model, `ModalityResult` (defined in `fusion/schemas.py`).
A `ModalityResult` carries: `modality`, `status` (success / failed / not_provided / skipped),
`label`, `risk_score` (0–1 threat likelihood), `confidence` (0–1 certainty), `threshold`,
`is_threat`, human-readable `evidence` strings, and the untouched `raw_output`.

Two rules are enforced everywhere:

- **risk_score ≠ confidence.** A detector can be 100% sure the input is benign.
- **Missing or failed detectors have `risk_score=None`** so they are never folded into the
  maths as if they had voted "safe".

| Adapter | Wraps | Live inference |
|---|---|---|
| `adapters/audio_adapter.py` | Audio Module's two Keras models (scam-call + anti-spoofing) | Yes — librosa/soundfile feature extraction, then `model.predict` |
| `adapters/url_adapter.py` | URL Module's XGBoost + SHAP + OSINT service | Yes — offline model-only mode, or full DNS/WHOIS/TLS OSINT |
| `adapters/text_adapter.py` | Text/Screenshot Module's trained text classifier (`text_scam_classifier.joblib`) + tactic heuristics | Yes |
| `adapters/screenshot_adapter.py` | **Heuristic only** (filename keywords + image dimensions via PIL). No trained CV model exists. | Yes (heuristic) |

Every adapter also exposes a `failed(...)` path: any exception becomes a `ModalityResult`
with `status=FAILED` and an `error_message` — a broken detector can never crash the pipeline.

## 2. Detection modules (teammate-owned)

### Audio Specialist — `audio_model/`
Two independent Keras CNNs, both consuming 64-band log-mel spectrograms of 16 kHz mono audio
(5-second window, n_fft=1024, hop=256):

1. **Scam-call classifier** — `dataset/best_audio_scam_model.keras` — scam vs normal call.
2. **Voice anti-spoofing** — `LA/best_model.keras` — ASVspoof 2019 logical-access track,
   bonafide vs synthetic/voice-cloned.

Sample audio used by the demo and tests lives in `dataset/SCAM_CALLS/` and `dataset/NORMAL_CALLS/`.

### URL Specialist — `domain_intelligence/`
- **26-feature XGBoost model** (`models/xgboost_url_model.joblib`) over lexical/structural URL
  features (entropy, path length, suspicious keywords, TLD risk, …). Feature schema is pinned in
  `models/feature_manifest.json` — the feature order must not change.
- **Tree SHAP explainer** (`models/shap_url_explainer.joblib`) producing per-feature
  contributions ("'uses_https' contributed +4.635 toward phishing").
- **Passive domain OSINT** (`src/domain_intelligence.py`): DNS resolution, WHOIS registration
  age, TLS certificate validation. Composite-score adjustments: domain < 30 days +15; invalid
  certificate +20; domain > 10 years old with valid cert ×0.3; > 5 years ×0.6 (with
  shared-hosting exclusions).
- **Brand lookalike scanner** (`src/lookalike.py`): typo/Levenshtein matching against known
  brands (PayPal, banks, …).
- **Threat feeds** (`src/threat_feeds.py`): known-phishing list matching.
- Also ships a standalone FastAPI service (`main.py` + `app/`) and its own research
  experiments (`experiments/`, `scripts/`, `results/`).

### Text/Vision Specialist — `text_and_screenshot_module/` + `adapters/text_adapter.py`, `adapters/screenshot_adapter.py`
- **Text**: a trained scikit-learn classifier (`models/text_scam_classifier.joblib`, trained on
  Text/Screenshot Module's 20,000-row labelled corpus) whose probability is combined with four heuristic
  "pressure tactic" detectors (urgency, financial impersonation, credential harvesting,
  payload requests). Flagged as a threat if the model score crosses the threshold **or** two
  or more tactics fire.
- **Screenshot datasets**: 8,256 labelled screenshots (6,192 phishing / 2,064 legitimate) in
  `FINAL_DATASET_1_3.zip`, plus the 20,000-row text corpus in `FINAL_TEXT_DATASET_1_3.zip`.
- **Screenshot detection**: the adapter applies a **filename-keyword + image-dimension
  heuristic** (brand names like "paypal"/"bank" in the filename, login-viewport aspect ratio).
  It is honest to call this a heuristic, not a computer-vision model; a trained CV model is
  future work for the datasets above.

## 3. Model Fusion (`fusion/fusion.py`, Fusion Engineer)

`ModelFusion.fuse()` accepts a mapping of modality → `ModalityResult` (or raw dict, or None)
and returns a `FusionResult`. Algorithm, in order:

1. **Standardise & categorise** every entry into active (successful with a score), missing,
   or failed. Failed/missing entries keep `risk_score=None` and are excluded from arithmetic.
2. **No active detectors** → verdict `INSUFFICIENT_DATA`, risk level `INCONCLUSIVE`.
3. **Confidence-weighted risk aggregation.** Each modality has a configured weight
   (`FusionConfig.modality_weights`: `audio_scam` 1.3, `url` 1.3, `screenshot` 1.1,
   `audio_spoof` 1.1, `text` 1.0, `domain` 0.9); its confidence scales the effective weight.
4. **Coverage penalty on confidence.** `overall_confidence = mean(confidence) × (0.6 + 0.4 ×
   active/5)` — a single-modality scan can never report 100% certainty.
5. **Defense-in-depth escalation.** If any active detector scored ≥ 0.70 *and* its confidence
   ≥ 0.75, the overall risk is raised to that detector's score rather than averaged down.
   A critical voice clone cannot be diluted to "safe" by benign text.
6. **Conflict detection.** Among *content* detectors (everything except `audio_spoof`), if one
   flags ≥ 0.65 and another scores ≤ 0.30, both with confidence ≥ 0.60, the result is flagged
   `has_conflict` with details. Spoof-vs-content disagreement is deliberately *not* a conflict:
   a synthetic voice reciting innocent text is an impersonation risk, not a contradiction.
7. **Risk level**: CRITICAL ≥ 0.85 · HIGH ≥ 0.65 · MEDIUM ≥ 0.35 · LOW below.
8. **Verdict**:
   - conflict found → `CONFLICT_WARNING`
   - only the spoof detector is a threat → `SUSPICIOUS` (impersonation risk, unproven intent)
   - HIGH/CRITICAL risk → `SCAM`
   - MEDIUM → `SUSPICIOUS`
   - otherwise → `SAFE`
9. **Evidence preservation.** All per-detector evidence is deduplicated into
   `evidence_summary`, failures are recorded, and missing modalities generate an explicit
   "[SYSTEM] Unexamined modalities … are not presumed safe" note.

## 4. Explanation engine (`fusion/llm_explainer.py`, Fusion Engineer)

`LLMExplainer.explain(FusionResult) → ExplanationResult` with three providers:

- **`deterministic`** (default; zero network, zero keys): rule-based templates that emit the
  headline, a plain-language "why", evidence citations, and verdict-tailored recommended
  actions. This is **template-based reasoning, not a locally running generative LLM**.
- **`gemini`** / **`openai`** (optional): calls the provider's REST API with a strict
  grounding prompt. Two anti-hallucination guards wrap every LLM response:
  - `_sanitize_citations` — every cited evidence item must match an item in the detector
    evidence summary; anything invented is dropped, and if *all* citations are foreign the
    verified evidence is used verbatim.
  - `_validate_llm_integrity` — a response whose headline contradicts the verdict
    (e.g. "completely safe" on a SCAM verdict, a prompt-injection signature) is rejected and
    the deterministic explanation is used instead.
- The **missing-data disclaimer is always generated by the application**, never the LLM.

Environment variables: `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `OPENAI_API_KEY`. Without
keys, every provider setting falls back to deterministic.

## 5. Pipeline orchestrator (`fusion/pipeline.py`, Fusion Engineer)

`MultimodalPipeline.analyze(...)` is the conductor: it invokes the adapters for whichever
inputs were supplied, merges any caller-provided `raw_modalities`, runs fusion, generates the
explanation, and wraps everything in `MultimodalAnalysisResult` — which offers property
shortcuts (`verdict`, `headline`, `evidence_citations`, …), `print_summary()` for console
output, and `to_dict()` for JSON transport. The one-line functional wrapper is
`analyze_multimodal(...)` — see [API_CONTRACT.md](API_CONTRACT.md).

## 6. Entry points

| Entry point | Purpose |
|---|---|
| `main.py` | Unified CLI: targeted `--audio/--url/--text` scans, or the 3-scenario live demo when run bare |
| `demo.py` | 6-scenario walkthrough: 5 fused mock scenarios + 1 fully live run |
| `audio_model/run_live_audio_detection.py` | Audio Module's standalone live-audio runner (feeds results through fusion) |
| `audio_model/test_load_models.py` | Standalone check that both Keras models load |
| `domain_intelligence/main.py` | URL Module's standalone URL scanner / FastAPI service |
| `text_and_screenshot_module/test_dataset.py` | Text/Screenshot Module's dataset diagnostic suite |
| `scripts/verify_installation.py` | Post-setup health check (imports + artifact loading) |
