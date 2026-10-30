"""tests/test_phase2a_reliability.py
Regression tests for Phase 2A: Safe reliability and validation handling for anti-spoofing.

Verifies:
1. Genuine audio with an extremely high spoof score is not treated as a confirmed/critical threat.
2. A validated phishing URL combined with unvalidated audio correctly triggers SCAM verdict.
3. Unvalidated audio as the only available signal returns an inconclusive / insufficient-evidence result.
4. Missing or failed modality results are explicitly handled.
5. A confirmed threat supported by other validated evidence is correctly classified.
6. Backward compatibility with existing API schemas and response structures is fully preserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path when executed directly or via pytest
WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from fusion.schemas import (
    DetectorStatus,
    ModalityResult,
    ModalityType,
    OverallVerdict,
    RiskLevel,
)
from fusion.fusion import ModelFusion, FusionConfig
from fusion.llm_explainer import LLMExplainer, ExplainerConfig
from fusion.pipeline import MultimodalAnalysisResult


def _make_unvalidated_spoof_result(score: float = 0.99986) -> ModalityResult:
    return ModalityResult(
        modality=ModalityType.AUDIO_SPOOF.value,
        status=DetectorStatus.SUCCESS,
        label="spoof",
        risk_score=score,
        confidence=round(min(1.0, 0.5 + abs(score - 0.5)), 4),
        threshold=0.5,
        is_threat=True,
        is_validated=False,
        validation_notes="Unvalidated experimental model: Trained on clean studio speech.",
        evidence=[
            f"Anti-spoofing model indicated synthetic voice characteristics with score {score:.3f} "
            "(unvalidated experimental signal on telephony audio)."
        ],
        raw_output={"spoof_score": score, "prediction": "spoof / synthetic", "file": "normal_17.wav"},
    )


def _make_validated_normal_scam_result(score: float = 0.04) -> ModalityResult:
    return ModalityResult(
        modality=ModalityType.AUDIO_SCAM.value,
        status=DetectorStatus.SUCCESS,
        label="normal",
        risk_score=score,
        confidence=0.92,
        threshold=0.5,
        is_threat=False,
        is_validated=True,
        evidence=[f"Scam-call classifier evaluated conversation as normal with score {score:.3f}."],
        raw_output={"scam_score": score, "label": "normal", "file": "normal_17.wav"},
    )


def _make_validated_scam_call_result(score: float = 0.88) -> ModalityResult:
    return ModalityResult(
        modality=ModalityType.AUDIO_SCAM.value,
        status=DetectorStatus.SUCCESS,
        label="scam",
        risk_score=score,
        confidence=0.88,
        threshold=0.5,
        is_threat=True,
        is_validated=True,
        evidence=[f"Scam-call classifier flagged fraudulent pattern with score {score:.3f}."],
        raw_output={"scam_score": score, "label": "scam"},
    )


def _make_validated_phishing_url_result(score: float = 0.99) -> ModalityResult:
    return ModalityResult(
        modality=ModalityType.URL.value,
        status=DetectorStatus.SUCCESS,
        label="phishing",
        risk_score=score,
        confidence=0.95,
        threshold=0.5,
        is_threat=True,
        is_validated=True,
        evidence=[f"Phishing classifier flagged link as malicious with probability {score:.3f}."],
        raw_output={"phishing_probability": score, "prediction": "phishing"},
    )


def test_genuine_audio_with_extremely_high_spoof_score():
    """Requirement 1: Genuine audio with high spoof score (like normal_17.wav) must not be a confirmed threat."""
    fusion = ModelFusion()
    scam_res = _make_validated_normal_scam_result(score=0.04)
    spoof_res = _make_unvalidated_spoof_result(score=0.99986)

    res = fusion.fuse({
        "audio_scam": scam_res,
        "audio_spoof": spoof_res,
    })

    # Overall risk must not be escalated by the unvalidated spoof model
    assert res.overall_risk_score == 0.04
    assert res.risk_level == RiskLevel.LOW
    assert res.verdict == OverallVerdict.SAFE
    assert res.metadata.get("defense_in_depth_escalated") is False

    # Raw model output must be preserved for research and evaluation
    assert "audio_spoof" in res.modality_breakdown
    assert res.modality_breakdown["audio_spoof"].risk_score == 0.99986
    assert res.modality_breakdown["audio_spoof"].is_validated is False
    assert "audio_spoof" in res.metadata.get("unvalidated_modalities", [])


def test_validated_phishing_url_combined_with_unvalidated_audio():
    """Requirement 2: A validated phishing threat must remain detected and confirmed as SCAM."""
    fusion = ModelFusion()
    url_res = _make_validated_phishing_url_result(score=0.99)
    spoof_res = _make_unvalidated_spoof_result(score=0.99986)

    res = fusion.fuse({
        "url": url_res,
        "audio_spoof": spoof_res,
    })

    # Phishing threat must not be weakened
    assert res.verdict == OverallVerdict.SCAM
    assert res.risk_level == RiskLevel.CRITICAL
    assert res.overall_risk_score >= 0.85
    assert "url" in res.active_modalities


def test_unvalidated_audio_as_only_available_signal():
    """Requirement 3: When only unvalidated audio is supplied, result must be inconclusive."""
    fusion = ModelFusion()
    spoof_res = _make_unvalidated_spoof_result(score=0.99986)

    res = fusion.fuse({
        "audio_spoof": spoof_res,
        "audio_scam": None,
        "url": None,
    })

    # Must return inconclusive rather than asserting safe or scam
    assert res.verdict == OverallVerdict.INSUFFICIENT_DATA
    assert res.risk_level == RiskLevel.INCONCLUSIVE
    assert res.overall_confidence == 0.0
    assert res.metadata.get("reason") == "insufficient_validated_evidence"

    # Raw score preserved
    assert res.metadata.get("unvalidated_scores", {}).get("audio_spoof") == 0.99986

    # Explainer should reflect inconclusive status
    explainer = LLMExplainer(ExplainerConfig(provider="deterministic"))
    exp = explainer.explain(res)
    assert "Inconclusive" in exp.headline


def test_missing_or_failed_modality_results():
    """Requirement 4: Missing and failed modalities are handled explicitly."""
    fusion = ModelFusion()
    failed_res = ModalityResult.failed("url", "Connection timed out")
    missing_res = ModalityResult.not_provided("text")

    res = fusion.fuse({
        "url": failed_res,
        "text": missing_res,
    })

    assert res.verdict == OverallVerdict.INSUFFICIENT_DATA
    assert res.risk_level == RiskLevel.INCONCLUSIVE
    assert "url" in res.failed_modalities
    assert "text" in res.missing_modalities


def test_confirmed_threat_supported_by_validated_evidence():
    """Requirement 5: Validated scam speech pattern correctly triggers a confirmed threat."""
    fusion = ModelFusion()
    scam_res = _make_validated_scam_call_result(score=0.88)
    spoof_res = _make_unvalidated_spoof_result(score=0.99986)

    res = fusion.fuse({
        "audio_scam": scam_res,
        "audio_spoof": spoof_res,
    })

    # Validated conversational scam call triggers SCAM
    assert res.verdict == OverallVerdict.SCAM
    assert res.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
    assert res.overall_risk_score >= 0.85


def test_backward_compatibility_with_existing_api_response():
    """Requirement 6: API contracts, to_dict(), and existing schemas remain backward-compatible."""
    # Instantiating ModalityResult without is_validated defaults to True
    legacy_style_result = ModalityResult(
        modality="text",
        status=DetectorStatus.SUCCESS,
        label="scam",
        risk_score=0.85,
        confidence=0.90,
    )
    assert legacy_style_result.is_validated is True
    assert legacy_style_result.validation_notes is None

    # Serialization test
    fusion = ModelFusion()
    f_res = fusion.fuse({"text": legacy_style_result})
    explainer = LLMExplainer(ExplainerConfig(provider="deterministic"))
    exp_res = explainer.explain(f_res)

    pipeline_res = MultimodalAnalysisResult(
        fusion=f_res,
        explanation=exp_res,
        modality_results={"text": legacy_style_result},
    )

    data = pipeline_res.to_dict()

    # Ensure required JSON contract keys exist
    expected_top_keys = {
        "verdict", "risk_level", "overall_risk_score", "overall_confidence",
        "has_conflict", "conflict_details", "active_modalities", "missing_modalities",
        "failed_modalities", "explanation", "modalities",
    }
    assert expected_top_keys.issubset(data.keys())

    # Ensure JSON serializable without error
    json_str = json.dumps(data)
    assert json_str is not None
    assert data["verdict"] == "SCAM"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
