"""demo.py
Demonstration script for Multimodal Models Fusion and LLM Explanation.

Runs real-world detection scenarios:
1. Single Modality: Audio Scam Detector (`dataset/audio_scam_detection_model`)
2. Single Modality: Voice Anti-Spoofing Detector (`LA/asvspoof2019_la_cnn`)
3. Multimodal Convergence: Audio Scam + Spoof + Phishing URL + Fake Screenshot
4. Conflicting Signals: Audio flags High Scam Threat, but URL reports Benign
5. Inconclusive Data: No modalities or all detectors failed
"""

from adapters import (
    AudioAdapter,
    ScreenshotAdapter,
    URLAdapter,
)
from fusion import (
    ExplainerConfig,
    LLMExplainer,
    ModelFusion,
    MultimodalPipeline,
)


def print_separator(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def display_result(fusion_result, explanation):
    print(f"Overall Verdict    : {fusion_result.verdict.value}")
    print(f"Risk Level         : {fusion_result.risk_level.value}")
    print(f"Overall Risk Score : {fusion_result.overall_risk_score:.2f} ({int(fusion_result.overall_risk_score * 100)}%)")
    print(f"Confidence         : {fusion_result.overall_confidence:.2f}")
    print(f"Active Modalities  : {', '.join(fusion_result.active_modalities) or 'None'}")
    print(f"Missing Modalities : {', '.join(fusion_result.missing_modalities) or 'None'}")
    print(f"Conflict Detected  : {'YES' if fusion_result.has_conflict else 'No'}")

    print("\n[EXPLANATION]")
    print(f"Headline : {explanation.headline}")
    print(f"Why      : {explanation.why}")

    print("\n[EVIDENCE CITATIONS]")
    for ev in explanation.evidence_citations:
        print(f"  - {ev}")

    if explanation.conflict_warning:
        print(f"\n[CONFLICT WARNING]\n  {explanation.conflict_warning}")

    if explanation.missing_data_disclaimer:
        print(f"\n[DISCLAIMER]\n  {explanation.missing_data_disclaimer}")

    print("\n[RECOMMENDED ACTIONS]")
    for act in explanation.recommended_actions:
        print(f"  * {act}")


def main():
    fusion = ModelFusion()
    explainer = LLMExplainer(ExplainerConfig(provider="deterministic"))

    # -------------------------------------------------------------
    # Scenario 1: Only Audio Scam Call Module Output
    # -------------------------------------------------------------
    print_separator("SCENARIO 1: Audio Scam Module (Single Modality)")
    # Exact output format from predict_audio_file in dataset notebook
    sample_scam_call_raw = {
        "label": "scam",
        "scam_score": 0.884,
        "threshold": 0.5,
    }
    audio_scam_res = AudioAdapter.adapt_scam_call(sample_scam_call_raw, source_name="urgent_bank_call.wav")

    res1 = fusion.fuse({
        "audio_scam": audio_scam_res,
        "audio_spoof": None,
        "url": None,
        "text": None,
        "screenshot": None,
    })
    exp1 = explainer.explain(res1)
    display_result(res1, exp1)

    # -------------------------------------------------------------
    # Scenario 2: Anti-Spoofing / Voice Clone Module
    # -------------------------------------------------------------
    print_separator("SCENARIO 2: Anti-Spoofing Voice Deepfake (Single Modality)")
    # Exact output format from predict_audio in LA notebook
    sample_spoof_raw = {
        "file": "/content/audio/ceo_voice_memo.wav",
        "spoof_score": 0.942,
        "prediction": "spoof / synthetic",
        "threshold": 0.5,
    }
    audio_spoof_res = AudioAdapter.adapt_spoof(sample_spoof_raw)

    res2 = fusion.fuse({
        "audio_spoof": audio_spoof_res,
        "audio_scam": None,
        "url": None,
    })
    exp2 = explainer.explain(res2)
    display_result(res2, exp2)

    # -------------------------------------------------------------
    # Scenario 3: Full Multimodal Convergence (Audio + Spoof + URL + Screenshot)
    # -------------------------------------------------------------
    print_separator("SCENARIO 3: Full Multimodal Threat Convergence")
    url_mock = URLAdapter.create_mock(is_phishing=True, url="http://192.168.1.10/login.php", probability=0.96)
    shot_mock = ScreenshotAdapter.create_mock(is_threat=True, image_name="fake_portal.png", score=0.89)

    res3 = fusion.fuse({
        "audio_scam": audio_scam_res,
        "audio_spoof": audio_spoof_res,
        "url": url_mock,
        "screenshot": shot_mock,
        "text": None,
    })
    exp3 = explainer.explain(res3)
    display_result(res3, exp3)

    # -------------------------------------------------------------
    # Scenario 4: Conflicting Predictions (Audio flags Scam, URL reports Safe)
    # -------------------------------------------------------------
    print_separator("SCENARIO 4: Conflicting Detector Signals")
    benign_url = URLAdapter.create_mock(is_phishing=False, url="https://official-bank.com", probability=0.06)

    res4 = fusion.fuse({
        "audio_scam": audio_scam_res,
        "url": benign_url,
    })
    exp4 = explainer.explain(res4)
    display_result(res4, exp4)

    # -------------------------------------------------------------
    # Scenario 5: Inconclusive / Missing Data
    # -------------------------------------------------------------
    print_separator("SCENARIO 5: All Inputs Omitted / Inconclusive")
    res5 = fusion.fuse({
        "audio_scam": None,
        "url": None,
        "screenshot": None,
    })
    exp5 = explainer.explain(res5)
    display_result(res5, exp5)

    # -------------------------------------------------------------
    # SCENARIO 6: LIVE Multimodal Execution (Real Audio + Real XGBoost URL)
    # -------------------------------------------------------------
    print_separator("SCENARIO 6: LIVE End-to-End Multimodal Execution (Real Audio + Real URL Models)")
    import os

    audio_file = os.path.join("audio_model", "dataset", "SCAM_CALLS", "scam_22.wav")
    live_url = "http://paypal.com-account-verify.online/login"

    if os.path.exists(audio_file):
        print(f"Running live inference on audio: {audio_file}")
        print(f"Running live XGBoost + SHAP on URL: {live_url}\n")
        pipe = MultimodalPipeline(explainer_config=ExplainerConfig(provider="deterministic"))
        live_res = pipe.analyze(audio_path=audio_file, url=live_url, include_domain_intelligence=False)
        display_result(live_res.fusion, live_res.explanation)
    else:
        print(f"Audio file not found at {audio_file}, skipping live run.")


if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            reconfig_out = getattr(sys.stdout, "reconfigure", None)
            if callable(reconfig_out):
                reconfig_out(encoding="utf-8", errors="replace")
            reconfig_err = getattr(sys.stderr, "reconfigure", None)
            if callable(reconfig_err):
                reconfig_err(encoding="utf-8", errors="replace")
        except Exception:
            pass
    main()
