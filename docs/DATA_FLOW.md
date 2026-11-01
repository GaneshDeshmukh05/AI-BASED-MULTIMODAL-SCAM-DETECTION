# Data Flow

How one analysis travels through the system, step by step, matching the actual code in
`fusion/pipeline.py`.

```
User submits message / URL / audio recording / screenshot (any combination; all optional)
    ↓
MultimodalPipeline.analyze()                       fusion/pipeline.py
    ↓
Applicable adapters run live inference             adapters/*.py
    ↓
Each adapter produces a ModalityResult             fusion/schemas.py
    ↓
ModelFusion.fuse() combines available results      fusion/fusion.py
    ↓
Risk evaluation: weighted score → escalation → conflict check → verdict
    ↓
LLMExplainer.explain() produces the explanation    fusion/llm_explainer.py
    ↓
MultimodalAnalysisResult (fusion + explanation + per-modality detail)
    ↓
result.to_dict() → JSON → Client API's backend → frontend
```

---

## Step 0 — Inputs

`analyze_multimodal()` / `MultimodalPipeline.analyze()` accept four independent inputs, all
optional: `audio_path`, `url`, `text`, `screenshot_path`. Whatever is supplied is analysed;
whatever is absent is simply not run. There is no required combination — a URL-only scan or an
audio-only scan is a first-class use case.

The `include_domain_intelligence` flag (default `True`) controls whether the URL analysis
performs live DNS/WHOIS/TLS lookups or stays offline with the XGBoost model only.

## Step 1 — Audio (`adapters/audio_adapter.py`)

Triggered when `audio_path` is given. **Both** audio detectors run on the same file:

1. **Scam-call classifier** — `librosa` loads the file at 16 kHz mono, pads/trims to exactly
   5 seconds, computes a 64-band log-mel spectrogram (n_fft 1024, hop 256), normalises it, and
   calls Audio Module's `best_audio_scam_model.keras`.
2. **Anti-spoofing model** — `soundfile`/`librosa` produce the same spectrogram representation
   and call `LA/best_model.keras`.

Each result is wrapped by `AudioAdapter.adapt_scam_call` / `adapt_spoof`, producing
`audio_scam` and `audio_spoof` `ModalityResult`s. The evidence strings quote the actual score
and threshold (e.g. *"spoof score 1.000 (threshold 0.50)"*). Any exception (missing file,
corrupt audio, model failure) becomes a `FAILED` result with an error message — it never
raises into the pipeline.

## Step 2 — URL & domain (`adapters/url_adapter.py`)

Triggered when `url` is given. The adapter puts `domain_intelligence` on `sys.path` and
uses URL Module's service:

- **Offline mode** (`include_domain_intelligence=False`): `ModelService.analyze(url)` runs the
  26-feature extraction and XGBoost prediction, returning `phishing_probability` and the top
  SHAP risk factors.
- **Full OSINT mode** (default): `URLThreatAnalyst.gather_telemetry(url)` additionally performs
  DNS lookup, WHOIS age, TLS certificate validation, brand-lookalike matching, and threat-feed
  checks, producing a composite score plus itemised `risk_adjustments`.

`URLAdapter.adapt()` normalises whichever shape came back into one `ModalityResult` (`url`),
extracting the score from `composite_risk_score` / `phishing_probability` / `risk_score`
(whichever is present, rescaled to 0–1) and translating SHAP factors, brand spoofing, domain
age, SSL problems, and feed matches into evidence strings.

## Step 3 — Text (`adapters/text_adapter.py`)

Triggered when `text` is given:

1. Four heuristic tactic detectors scan the text: artificial urgency, financial impersonation,
   credential harvesting, unsolicited payload.
2. Text/Screenshot Module's trained classifier (`text_scam_classifier.joblib`) produces a scam probability
   (label 1 = scam). If the model can't load or fails, the score falls back to a tactic-count
   heuristic so the pipeline still returns a usable result.
3. The result is a threat if the score ≥ threshold **or** ≥ 2 tactics fired.

## Step 4 — Screenshot (`adapters/screenshot_adapter.py`)

Triggered when `screenshot_path` is given. This is a **heuristic, not a trained model**: PIL
reads the image format/dimensions, the filename is scanned for brand/keyword markers
("paypal", "bank", "login", "verify", "phish", …), and a base risk of 0.15 is raised by brand
hits (+0.40 each, capped at 2) and by a login-viewport aspect ratio combined with
login/verify/phish keywords. Output is a `screenshot` `ModalityResult` whose evidence honestly
reports what was matched.

## Step 5 — Fusion (`fusion/fusion.py`)

`ModelFusion.fuse()` receives the dict of results (only the supplied keys — plus anything the
caller injected via `raw_modalities`, which may include explicit `None` entries):

1. Results are categorised: **active** (success with score), **failed** (error), **missing**
   (explicitly `None` / not provided).
2. If nothing is active → `INSUFFICIENT_DATA` / `INCONCLUSIVE`, zero risk, zero confidence.
3. Otherwise the **confidence-weighted average** of active risk scores is computed, using
   per-modality weights (`audio_scam` 1.3, `url` 1.3, `screenshot` 1.1, `audio_spoof` 1.1,
   `text` 1.0, `domain` 0.9).
4. **Coverage penalty**: overall confidence is scaled by `(0.6 + 0.4 × active/5)`.
5. **Defense-in-depth**: a threat scored ≥ 0.70 with confidence ≥ 0.75 lifts the overall risk
   to its own score instead of being averaged down.
6. **Conflict check** among content detectors (excluding `audio_spoof`): a ≥ 0.65 threat and a
   ≤ 0.30 "safe", both with confidence ≥ 0.60, sets `has_conflict` + details.
7. Risk level and verdict are chosen (see [ARCHITECTURE.md](ARCHITECTURE.md) §3 for the exact
   decision table — e.g. spoof-only threats yield `SUSPICIOUS`, not `SCAM`).
8. Evidence is consolidated; failures and unexamined modalities are recorded explicitly.

## Step 6 — Explanation (`fusion/llm_explainer.py`)

`LLMExplainer.explain(FusionResult)` builds the user-facing `ExplanationResult`:

- **Deterministic provider** (default): template-based headline/why/evidence/actions, chosen
  by verdict. Zero network use. (Not a generative LLM — by design, zero hallucination risk.)
- **Gemini/OpenAI providers** (only if API keys are set): the fusion result is serialised into
  a strict grounding prompt; the response is then filtered — citations must match real
  detector evidence, and any response contradicting the verdict (including prompt-injection
  payloads inside untrusted evidence) causes a silent fallback to deterministic.
- The missing-data disclaimer is always app-generated from `missing_modalities` /
  `failed_modalities`.

## Step 7 — Result & transport (`fusion/pipeline.py`)

Everything is assembled into `MultimodalAnalysisResult`:

- `fusion: FusionResult` — verdict, scores, conflicts, per-modality breakdown, metadata.
- `explanation: ExplanationResult` — headline, why, citations, actions, provider used.
- `modality_results` — the per-detector `ModalityResult`s.

`to_dict()` renders a fully JSON-serialisable dict (verified by
`tests/test_phase2a_reliability.py::test_backward_compatibility_with_existing_api_response`):

```
verdict, risk_level, overall_risk_score, overall_confidence, has_conflict,
conflict_details, active_modalities, missing_modalities, failed_modalities,
explanation { headline, why, evidence_citations, conflict_warning,
              missing_data_disclaimer, recommended_actions, llm_provider },
modalities { <name>: { modality, status, label, risk_score, confidence, threshold,
                       is_threat, evidence, raw_output, error_message, latency_ms } }
```

A representative real response is included in [API_CONTRACT.md](API_CONTRACT.md), which is
also the document Client API's backend should implement against.

## Error handling & missing modalities — summary of guarantees

| Situation | Behaviour |
|---|---|
| Input omitted | Adapter never runs; modality absent from results. Not treated as safe. |
| Adapter raises (bad file, model failure, network error) | `ModalityResult.failed(...)`; listed in `failed_modalities`; excluded from risk maths; failure recorded in evidence. |
| All detectors failed / no inputs | Verdict `INSUFFICIENT_DATA`, risk `INCONCLUSIVE`, score 0.0. |
| Explicit `None` passed in `raw_modalities` | Listed in `missing_modalities` with the "not presumed safe" system note. |
| Unexamined channels | Called out in `explanation.missing_data_disclaimer` (app-generated). |
