# 🔌 Fusion & Explanation API Contract

**From**: Fusion Engineer (Model Fusion & LLM Explanation Layer)
**To**: Client Engineer (Backend API & Frontend UI)
**Status**: Ready for integration

> This document is the **canonical** integration contract. It supersedes the former
> earlier draft API contracts (now consolidated here) and corrects two inaccuracies in them:
> the verdict value is `"SAFE"` (not `"BENIGN"`), and the risk-level ranges below now match
> the code exactly. The JSON example in §5 is a real captured response, not a mock-up.

---

## 1. Quick integration — the one function to call

Import `analyze_multimodal` from the `fusion` package (run your backend with this repository
root on the Python path):

```python
from fusion import analyze_multimodal

# One call for any combination of modalities — every argument is optional:
result = analyze_multimodal(
    audio_path="temp_uploads/user_call.wav",    # Optional: path to a saved .wav file
    url="http://example.com/login",             # Optional: target URL string
    text="Your bank account is locked...",      # Optional: SMS / transcript text
    screenshot_path="temp_uploads/portal.png",  # Optional: path to a saved image
    include_domain_intelligence=True,           # Optional (default True): DNS/WHOIS/TLS OSINT
    explainer_provider="deterministic",         # Optional: 'deterministic' | 'gemini' | 'openai'
)

# Convert directly to a clean JSON-serializable dict for your API response:
response_json = result.to_dict()
```

Any input can be `None` / omitted. Submitting only an audio file is a fully supported call —
the fusion engine handles partial input and records what was *not* examined.

For advanced use, `MultimodalPipeline.analyze(...)` additionally accepts `raw_modalities`
(pre-computed `ModalityResult`s or raw detector dicts to merge in) and `url_provider`
('deterministic' | 'gemini' | 'openai' for URL Module's module's own reasoning). Most backends
only need `analyze_multimodal`.

**Environment**: no API keys are required. With `GEMINI_API_KEY` / `OPENAI_API_KEY` set and
`explainer_provider` set accordingly, explanations are LLM-generated (with automatic fallback
to deterministic on any failure). Default is fully offline.

---

## 2. Input parameters (`analyze_multimodal`)

Actual signature:

```python
def analyze_multimodal(
    audio_path: Optional[str] = None,
    url: Optional[str] = None,
    text: Optional[str] = None,
    screenshot_path: Optional[str] = None,
    include_domain_intelligence: bool = True,
    explainer_provider: str = "deterministic",
) -> MultimodalAnalysisResult: ...
```

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `audio_path` | `str \| None` | Optional | `None` | Filepath to an audio recording (`.wav` recommended). Runs **both** Audio Specialist models: scam-call classifier and anti-spoofing. |
| `url` | `str \| None` | Optional | `None` | URL or domain string for URL Module's XGBoost + SHAP + OSINT evaluation. |
| `text` | `str \| None` | Optional | `None` | SMS / email / transcript text for Text/Screenshot Module's NLP classifier + tactic heuristics. |
| `screenshot_path` | `str \| None` | Optional | `None` | Filepath to an uploaded screenshot (`.png`/`.jpg`). Runs Text/Screenshot Module's heuristic detector. |
| `include_domain_intelligence` | `bool` | Optional | `True` | `False` = fast offline URL inference (skips live DNS/WHOIS/TLS; recommended for latency-sensitive or sandboxed deployments). |
| `explainer_provider` | `str` | Optional | `"deterministic"` | `"deterministic"` (instant, offline, no keys) · `"gemini"` · `"openai"` · `"auto"` (uses keys if present). |

Errors never raise out of the function: a failing detector returns as a `FAILED` modality
result inside the response (see §6).

---

## 3. Return object

`analyze_multimodal` returns a `MultimodalAnalysisResult` with convenient properties
(`verdict`, `risk_level`, `overall_risk_score`, `overall_confidence`, `has_conflict`,
`headline`, `why`, `evidence_citations`, `recommended_actions`) plus:

- `result.fusion` — full `FusionResult` (per-modality breakdown, conflict details, metadata).
- `result.explanation` — full `ExplanationResult`.
- `result.to_dict()` — the JSON-serialisable dict described in §5.
- `result.print_summary()` — formatted console report (useful in logs/debugging).

`to_dict()` is verified JSON-serialisable by
`tests/test_phase2a_reliability.py::test_backward_compatibility_with_existing_api_response`.

---

## 4. Verdicts & risk levels (for UI badges)

### `verdict` — actual enum values (`fusion/schemas.py::OverallVerdict`)

| Value | Meaning | Suggested UI |
|---|---|---|
| `"SCAM"` | Confirmed scam: high/critical aggregated risk or multiple corroborating threats | Red |
| `"SUSPICIOUS"` | Elevated risk — e.g. synthetic voice alone, or medium aggregated risk | Orange/Yellow |
| `"SAFE"` | All analyzed modalities scored within safe limits | Green |
| `"CONFLICT_WARNING"` | Content detectors strongly disagree; caution warranted | Amber/red pulsing |
| `"INSUFFICIENT_DATA"` | No inputs supplied, or every detector failed | Grey |

> ⚠️ The legitimate/verified-safe verdict is `"SAFE"`. (An earlier revision of this contract
> said `"BENIGN"` — that value does not exist in the code.)

### `risk_level` — actual thresholds (`fusion/schemas.py` + `FusionConfig`)

| Value | Range | Badge |
|---|---|---|
| `"LOW"` | 0.00 – 0.35 | Green |
| `"MEDIUM"` | 0.35 – 0.65 | Yellow |
| `"HIGH"` | 0.65 – 0.85 | Orange |
| `"CRITICAL"` | 0.85 – 1.00 | Red |
| `"INCONCLUSIVE"` | n/a (no usable signals) | Grey |

### `llm_provider` values seen in `explanation`

`"deterministic_fallback"`, `"gemini (<model>)"`, `"openai (<model>)"`.

---

## 5. Response JSON schema — real captured example

Captured verbatim from `analyze_multimodal(url="http://paypal.com-verify.online/login",
include_domain_intelligence=False).to_dict()` (only `raw_output` is abridged):

```json
{
  "verdict": "SCAM",
  "risk_level": "CRITICAL",
  "overall_risk_score": 1.0,
  "overall_confidence": 0.68,
  "has_conflict": false,
  "conflict_details": null,
  "active_modalities": ["url"],
  "missing_modalities": [],
  "failed_modalities": [],
  "explanation": {
    "headline": "High Threat Alert: Potential Scam Activity Detected (100% Risk)",
    "why": "Multiple forensic detectors identified strong scam indicators (100% risk score). The observed content matches verified patterns of fraudulent calls, phishing, or spoofing.",
    "evidence_citations": [
      "[URL] URL 'http://paypal.com-verify.online/login': Phishing classifier flagged link as malicious with probability 1.000 (operating threshold 0.50).",
      "[URL] High-risk factor: 'uses_https' contributed +4.809 toward phishing.",
      "[URL] High-risk factor: 'path_length' contributed +4.087 toward phishing.",
      "[URL] High-risk factor: 'suspicious_keyword_count' contributed +0.832 toward phishing."
    ],
    "conflict_warning": null,
    "missing_data_disclaimer": null,
    "recommended_actions": [
      "Do NOT transfer funds, purchase gift cards, or share OTPs / passwords.",
      "Do NOT click links, download attachments, or enter credentials on the referenced webpage.",
      "Report the suspicious call, message, or link to relevant cybersecurity authorities."
    ],
    "llm_provider": "deterministic_fallback"
  },
  "modalities": {
    "url": {
      "modality": "url",
      "status": "success",
      "label": "phishing",
      "risk_score": 0.999964,
      "confidence": 1.0,
      "threshold": 0.5,
      "is_threat": true,
      "evidence": [
        "URL 'http://paypal.com-verify.online/login': Phishing classifier flagged link as malicious with probability 1.000 (operating threshold 0.50).",
        "High-risk factor: 'uses_https' contributed +4.809 toward phishing."
      ],
      "raw_output": "<full URL Specialist model payload: prediction, phishing_probability, confidence, operating_threshold, risk_level, top_risk_factors, shap_contributions, extracted_features, ...>",
      "error_message": null,
      "latency_ms": null
    }
  }
}
```

### Field semantics the frontend should rely on

| Field | Semantics |
|---|---|
| `active_modalities` | Detectors that actually ran and produced a score. Only supplied inputs appear. |
| `missing_modalities` | Inputs explicitly passed as `None` (via `raw_modalities`) or otherwise not provided *within the results mapping*. **Note:** when using `analyze_multimodal`, modalities you simply omitted are absent from this list — derive "not examined" in the UI from `active_modalities` vs the five channels (`audio_scam`, `audio_spoof`, `text`, `url`, `screenshot`). |
| `failed_modalities` | Detectors that ran but errored; their `error_message` is in `modalities.<name>.error_message` and the failure is repeated in `evidence_citations`. |
| `explanation.missing_data_disclaimer` | App-generated; non-null when `missing_modalities` / `failed_modalities` exist. The UI may render it verbatim. |
| `explanation.evidence_citations` | Prefixed with the modality tag (`[URL]`, `[AUDIO_SPOOF]`, …). Guaranteed to come from real detector output — the pipeline filters invented citations. |
| `modalities.<name>.raw_output` | Untouched original detector payload (SHAP vectors, OSINT details). Useful for an "advanced details" view; hide by default. |
| `overall_confidence` | Already penalised for partial coverage — do not recompute. |

---

## 6. Error handling contract

| Situation | What the backend receives |
|---|---|
| Detector file missing / corrupt audio / network failure / bad URL | `modalities.<name>.status == "failed"`, `error_message` set, `risk_score` and `confidence` `null`; name listed in `failed_modalities`; other modalities unaffected. **No exception is raised.** |
| No inputs at all | `verdict: "INSUFFICIENT_DATA"`, `risk_level: "INCONCLUSIVE"`, scores 0.0, empty `active_modalities`. |
| URL classifier fails but audio succeeds | Mixed response: `failed_modalities: ["url"]`, verdict computed from audio alone. |
| LLM provider unreachable / key invalid / malformed output | Silent fallback to `deterministic_fallback` — `explanation.llm_provider` tells you which engine actually produced the text. |

Recommended backend practice: still wrap the call in a try/except (e.g. model files absent on
a fresh deploy), but treat per-modality failures as normal data, not exceptions.

---

## 7. Ready-to-use FastAPI route example

```python
import os
import shutil
from typing import Optional
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from fusion import analyze_multimodal

app = FastAPI(title="Multimodal Scam & Deepfake Detection API")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.post("/api/scan")
async def scan_multimodal_threat(
    url: Optional[str] = Form(None),
    text: Optional[str] = Form(None),
    audio_file: Optional[UploadFile] = File(None),
    screenshot_file: Optional[UploadFile] = File(None),
    offline_mode: bool = Form(False),
):
    audio_path = None
    screenshot_path = None

    try:
        if audio_file:
            audio_path = os.path.join(UPLOAD_DIR, audio_file.filename)
            with open(audio_path, "wb") as f:
                shutil.copyfileobj(audio_file.file, f)

        if screenshot_file:
            screenshot_path = os.path.join(UPLOAD_DIR, screenshot_file.filename)
            with open(screenshot_path, "wb") as f:
                shutil.copyfileobj(screenshot_file.file, f)

        analysis = analyze_multimodal(
            audio_path=audio_path,
            url=url if url and url.strip() else None,
            text=text if text and text.strip() else None,
            screenshot_path=screenshot_path,
            include_domain_intelligence=not offline_mode,
        )

        return JSONResponse(content=analysis.to_dict())

    finally:
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)
        if screenshot_path and os.path.exists(screenshot_path):
            os.remove(screenshot_path)
```

Deployment notes:

- The backend process needs this repository's root on its Python path (e.g. run uvicorn from
  the repo root, or install the repo as a package). Model artifacts must be present at their
  documented locations — run `python scripts/verify_installation.py` after deploying.
- First audio analysis loads TensorFlow + both Keras models (~10–20 s); models stay cached
  in-process afterwards. Warm up at startup if your first request must be fast.
- `include_domain_intelligence=True` makes outbound DNS/WHOIS/TLS queries per request; use
  `offline_mode` where that is undesirable.

---

## 8. Support

Questions about inputs, response formats, or edge cases → **Fusion Engineer** (fusion & explanation
layer). Questions about URL/OSINT internals → **URL Specialist**; audio models → **Audio Specialist**; datasets
and text model → **Text/Vision Specialist**.
