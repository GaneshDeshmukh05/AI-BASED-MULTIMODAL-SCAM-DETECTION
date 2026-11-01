# Known Limitations

An honest account of what is verified, what is only sample-tested, what is heuristic, and
what is planned. Nothing in this document is exaggerated, and nothing is hidden — evaluators
should read this alongside [TESTING_GUIDE.md](TESTING_GUIDE.md).

---

## 1. Verified functionality (backed by the automated suite)

These claims are directly supported by the 62 passing automated tests:

- Adapter translation of every detector payload format into `ModalityResult`, including error
  paths (malformed input → `FAILED` result, not a crash).
- Fusion decision logic: aggregation, conflict detection, defense-in-depth escalation,
  missing/failed modality handling, `INSUFFICIENT_DATA`, spoof-vs-scam distinction.
- Explanation grounding: citations must come from real detector evidence; verdict integrity
  checks; LLM failure → deterministic fallback; missing-data disclaimers are app-generated.
- End-to-end live pipeline: real audio through both Keras models + real URL through XGBoost →
  correct fused verdict, and `to_dict()` JSON serialisation.
- Dataset integrity: Text/Screenshot Module's text corpus (20,000 rows, 15,000/5,000 split) and screenshot
  archive (8,256 images, 3:1 split) exist, unzip, and decode.

## 2. Verified only on a small sample

- **Audio model accuracy is NOT established.** The Keras models load and infer, but the only
  real inputs exercised are the 48 bundled sample WAVs. Notably, on the two samples used in
  the demo/tests the anti-spoofing model scores **both** the scam sample and the "normal"
  sample as synthetic (spoof score 1.000), and the scam-call classifier scores both ≈ 0.479
  (just below the 0.50 threshold). Successful inference is not evidence of accuracy — a
  proper evaluation needs a held-out, labelled audio benchmark (as ASVspoof provides for
  researchers). Do not quote detection numbers from the demo runs.
- **URL Module's URL model** has an independently verified empirical research evaluation under
  [`domain_intelligence/results/`](file:///c:/Users/Abhishek/OneDrive/Desktop/LLM%20integration/domain_intelligence/results/):
  - **Primary Unseen-Domain Holdout** (zero shared domains): **93.94% Accuracy**, **96.28% Precision**,
    **91.09% Recall**, **93.62% F1**, **0.9845 ROC-AUC**, **0.9860 PR-AUC**, **3.35% FPR**, **8.91% FNR**
    (on 50,571 test URLs / 25,678 held-out domains; `zero_day_results.csv`).
  - **External Cross-Dataset Generalization**: **88.48% Accuracy**, **85.18% Precision**, **87.03% Recall**,
    **86.09% F1**, **0.9264 ROC-AUC**, **0.9398 PR-AUC**, **10.51% FPR**, **12.97% FNR**
    (evaluated on external `phishing.csv` after removing all 11,777 overlapping-domain rows; `cross_dataset_results.csv`).
  - **Conventional Random Split Baseline**: **94.44% Accuracy**, **0.9871 ROC-AUC** (contains 8,927 shared domains
    between train and test, which introduces domain leakage and overestimates zero-day generalization; `random_split_results.csv`).
- **Text model is an unvalidated hybrid system on held-out data.** The text scam classifier operates as a hybrid
  pipeline combining TF-IDF vectorization, Multinomial Naive Bayes classification, and heuristic semantic fraud
  rules (account takeover, OTP harvesting, advance-fee prize lures). While the training dataset exists (20,000 rows
  in `FINAL_TEXT_DATASET_1_3.zip`), this repository contains no independent held-out evaluation report, test split,
  or confusion matrix. Any informal claims that the text classifier achieves ~98% accuracy are unverified training-fit
  metrics and must not be presented as validated operational test performance.

## 3. Heuristic detection (not trained classification)

Be precise about these — they are rule-based, and the code is honest about it:

- **Screenshot detection is OCR-based content analysis and heuristic rules, not a computer-vision model.**
  `adapters/screenshot_adapter.py` extracts text via Windows Native OCR and scans for visual login prompts,
  password fields, brand impersonation, and advance-fee prize lures. There is **no trained computer-vision model in this
  repository**, despite the labeled screenshot dataset (8,256 images) being available for future training.
  Any UI copy or presentation must not claim "computer-vision / AI screenshot detection".
- **Text tactic detectors** (urgency, financial impersonation, credential harvesting, payload
  requests) are keyword rules alongside Text/Screenshot Module's trained classifier; they can be bypassed by
  paraphrasing.
- **URL composite OSINT adjustments** (domain age ±, certificate validity, brand lookalike
  edit distance) are transparent rule adjustments layered on the model score — documented by
  URL Specialist, including their failure modes (an old *compromised* domain with a valid certificate
  can still be discounted).
- **The deterministic explainer is a rule-based template engine**, not a locally running
  generative LLM. It is deterministic *by design* for zero-hallucination guarantees. Gemini /
  OpenAI generation is available when API keys are configured, and is then wrapped in
  citation sanitisation and verdict-integrity checks.

## 4. Model accuracy vs successful inference

A recurring theme: every "success" in this system means *the pipeline ran and produced a
verdict* — it does not mean the verdict is correct. Thresholds (0.50 per detector, 0.70/0.75
escalation, 0.65/0.30 conflict) are engineering choices, not calibrated operating points.
`risk_score` is a model probability (or heuristic score), and `confidence` is a margin-based
formula — neither is a calibrated probability of ground truth.

## 5. Structural / environmental limitations

- **Native Windows TensorFlow is CPU-only** (TF ≥ 2.11). Inference works; GPU acceleration
  would require WSL2 or the DirectML plugin.
- **First audio analysis is slow** (~10–20 s: TF + two Keras model loads, cached per process).
- **Domain OSINT is network-dependent** and may be slow or blocked on restricted networks;
  `include_domain_intelligence=False` gives the offline model-only result.
- **Machine-specific fallback path**: `adapters/audio_adapter.py` contains a hardcoded
  fallback to a `Downloads\AI-BASED-MULTIMODAL-SCAM-DETECTION-main` path from the original
  development machine. It is only used if the in-repo model files are missing, but it should
  be removed for a clean deployment.
- **URL Module's module is imported as package `src`** after a `sys.path` insertion of his
  directory — a pragmatic choice that works but is fragile if two modules both named `src`
  end up on the path. It also means static analysers need the extraPaths configured in
  `pyrightconfig.json` / `.vscode/settings.json`.
- **`missing_modalities` nuance**: with `analyze_multimodal`, omitted inputs are simply
  absent from the result mapping, so `missing_modalities` lists only explicitly-`None`
  entries. Frontends should derive "not examined" from `active_modalities` vs the five
  channels. Documented in [API_CONTRACT.md](API_CONTRACT.md) §5.
- The five expected modalities used by the coverage penalty (`FusionConfig.expected_modalities`)
  count `audio_scam` and `audio_spoof` separately — an audio-only scan can therefore never
  exceed 2/5 coverage, which intentionally caps its confidence.

## 6. Not implemented / planned (do not claim these)

- **Trained screenshot CV model** — datasets exist; training is future work (Text/Vision Specialist).
- **Client API's backend & frontend** — separate repository; only the interface contract exists
  here. Any "full product" claims await that integration.
- **Live microphone streaming analysis** — audio input is file-based (`run_live_audio_detection.py`
  analyses files; there is no real-time stream pipeline in this repo).
- **Calibrated thresholds / per-detector evaluation benchmarks** for the audio and text models.
- **Multilingual scam detection** — detectors are English-oriented.
- **User feedback loop / model retraining pipeline in production** — URL Specialist has retraining
  scripts (`scripts/retrain_with_feeds.py`), but there is no automated MLOps.

## 7. Testing scope summary

| Layer | Automated tests | Real-world evaluation |
|---|---|---|
| Schemas & fusion logic | ✅ 10 unit tests | n/a (logic) |
| Adapters | ✅ 8 unit tests | Live URL/audio adapter exercised in integration tests |
| LLM explainer | ✅ 12 unit tests (mocked HTTP) | Live Gemini/OpenAI calls not exercised in CI |
| Audio models | Load + infer (integration) | ❌ No accuracy benchmark in this repo |
| URL model | Live infer (unit + integration) | ✅ Via URL Module's own experiments suite |
| Text model | Load + infer (mock-level in pytest; live in `test_dataset.py`) | ❌ No independent benchmark in this repo |
| Screenshot detection | Mock-level only | ❌ Heuristic; no model |
| Datasets | ✅ Integrity + schema tests | n/a |
| Backend integration | `to_dict()` serialisation test | ⏳ Awaiting Client API's implementation |
