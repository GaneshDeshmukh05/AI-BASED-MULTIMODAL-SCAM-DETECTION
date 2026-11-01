"""run_live_audio_detection.py
Runs live inference using trained Keras models on real audio files,
feeds the results through the Fusion engine, and generates grounded LLM explanations.
"""

import os
import sys

# Suppress low-level TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from fusion import ModelFusion, LLMExplainer, ExplainerConfig
from adapters import AudioAdapter


def analyze_audio_file(file_path: str):
    print("\n" + "=" * 70)
    print(f"ANALYZING REAL AUDIO: {file_path}")
    print("=" * 70)

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    # 1. Execute live scam call model
    print("[1/3] Running Scam Call Model (dataset/best_audio_scam_model.keras)...")
    scam_result = AudioAdapter.predict_scam_call_file(file_path)
    print(f"  -> Risk Score: {scam_result.risk_score:.4f} | Label: {scam_result.label} | Threat: {scam_result.is_threat}")

    # 2. Execute live anti-spoofing model
    print("[2/3] Running Anti-Spoofing Model (LA/best_model.keras)...")
    spoof_result = AudioAdapter.predict_spoof_file(file_path)
    print(f"  -> Spoof Score: {spoof_result.risk_score:.4f} | Prediction: {spoof_result.label} | Threat: {spoof_result.is_threat}")

    # 3. Fuse both detector results
    print("[3/3] Fusing live detector outputs...")
    fusion_engine = ModelFusion()
    fusion_result = fusion_engine.fuse({
        "audio_scam": scam_result,
        "audio_spoof": spoof_result,
    })

    print(f"\n[FUSION VERDICT]")
    print(f"  Overall Verdict    : {fusion_result.verdict.value}")
    print(f"  Risk Level         : {fusion_result.risk_level.value}")
    print(f"  Aggregated Risk    : {fusion_result.overall_risk_score:.4f} ({int(fusion_result.overall_risk_score * 100)}%)")
    print(f"  Confidence         : {fusion_result.overall_confidence:.4f}")
    print(f"  Conflict Detected  : {'YES' if fusion_result.has_conflict else 'No'}")

    # 4. Generate user-facing explanation
    explainer = LLMExplainer(ExplainerConfig(provider="deterministic"))
    explanation = explainer.explain(fusion_result)

    print(f"\n[EXPLANATION]")
    print(f"  Headline : {explanation.headline}")
    print(f"  Why      : {explanation.why}")

    print("\n[GROUNDED EVIDENCE CITATIONS]")
    for ev in explanation.evidence_citations:
        print(f"  - {ev}")

    if explanation.conflict_warning:
        print(f"\n[CONFLICT WARNING]\n  {explanation.conflict_warning}")

    if explanation.missing_data_disclaimer:
        print(f"\n[DISCLAIMER]\n  {explanation.missing_data_disclaimer}")

    print("\n[ACTIONABLE RECOMMENDATIONS]")
    for action in explanation.recommended_actions:
        print(f"  * {action}")


def main():
    _base_dir = os.path.dirname(os.path.abspath(__file__))
    default_scam = os.path.join(_base_dir, "dataset", "SCAM_CALLS", "scam_22.wav")
    default_normal = os.path.join(_base_dir, "dataset", "NORMAL_CALLS", "normal_17.wav")

    target = sys.argv[1] if len(sys.argv) > 1 else default_scam
    analyze_audio_file(target)

    if len(sys.argv) <= 1:
        analyze_audio_file(default_normal)


if __name__ == "__main__":
    main()
