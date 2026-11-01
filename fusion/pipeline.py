"""pipeline.py
End-to-end multimodal scam and fraud detection pipeline.

Integrates:
1. Audio Module: Live scam call conversational detection & voice anti-spoofing (audio_model)
2. Domain Intelligence: 26-feature XGBoost URL classifier, SHAP attribution, & Domain OSINT (domain_intelligence)
3. Text & Screenshot detectors (mock or live)
4. ModelFusion: Weighted fusion with defense-in-depth escalation and conflict resolution
5. LLMExplainer: Grounded, hallucination-free explanations (Gemini / OpenAI / Deterministic)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from .fusion import FusionConfig, ModelFusion
from .llm_explainer import ExplainerConfig, LLMExplainer
from .schemas import (
    ExplanationResult,
    FusionResult,
    ModalityResult,
    OverallVerdict,
    RiskLevel,
)
logger = logging.getLogger("multimodal_pipeline")


@dataclass
class MultimodalAnalysisResult:
    """Consolidated outcome of live multimodal threat analysis."""

    fusion: FusionResult
    explanation: ExplanationResult
    modality_results: Dict[str, ModalityResult] = field(default_factory=dict)

    @property
    def verdict(self) -> OverallVerdict:
        return self.fusion.verdict

    @property
    def risk_level(self) -> RiskLevel:
        return self.fusion.risk_level

    @property
    def overall_risk_score(self) -> float:
        return self.fusion.overall_risk_score

    @property
    def overall_confidence(self) -> float:
        return self.fusion.overall_confidence

    @property
    def has_conflict(self) -> bool:
        return self.fusion.has_conflict

    @property
    def active_modalities(self) -> List[str]:
        return self.fusion.active_modalities

    @property
    def missing_modalities(self) -> List[str]:
        return self.fusion.missing_modalities

    @property
    def failed_modalities(self) -> List[str]:
        return self.fusion.failed_modalities

    @property
    def headline(self) -> str:
        return self.explanation.headline

    @property
    def why(self) -> str:
        return self.explanation.why

    @property
    def evidence_citations(self) -> List[str]:
        return self.explanation.evidence_citations

    @property
    def recommended_actions(self) -> List[str]:
        return self.explanation.recommended_actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to JSON-serializable dictionary."""
        return {
            "verdict": self.verdict.value,
            "risk_level": self.risk_level.value,
            "overall_risk_score": self.overall_risk_score,
            "overall_confidence": self.overall_confidence,
            "has_conflict": self.has_conflict,
            "conflict_details": self.fusion.conflict_details,
            "active_modalities": self.fusion.active_modalities,
            "missing_modalities": self.fusion.missing_modalities,
            "failed_modalities": self.fusion.failed_modalities,
            "explanation": {
                "headline": self.headline,
                "why": self.why,
                "evidence_citations": self.evidence_citations,
                "conflict_warning": self.explanation.conflict_warning,
                "missing_data_disclaimer": self.explanation.missing_data_disclaimer,
                "recommended_actions": self.recommended_actions,
                "llm_provider": self.explanation.llm_provider,
            },
            "modalities": {
                mod_name: res.model_dump(mode="json")
                for mod_name, res in self.modality_results.items()
            },
        }

    def print_summary(self) -> None:
        """Prints a human-readable console report."""
        print("\n" + "=" * 70)
        print(f"MULTIMODAL THREAT ANALYSIS: {self.verdict.value} (Risk: {int(self.overall_risk_score * 100)}%)")
        print("=" * 70)
        print(f"Verdict            : {self.verdict.value}")
        print(f"Risk Level         : {self.risk_level.value}")
        print(f"Aggregated Risk    : {self.overall_risk_score:.4f} ({int(self.overall_risk_score * 100)}%)")
        print(f"Confidence         : {self.overall_confidence:.4f}")
        print(f"Active Modalities  : {', '.join(self.fusion.active_modalities) or 'None'}")
        print(f"Missing Modalities : {', '.join(self.fusion.missing_modalities) or 'None'}")
        if self.fusion.failed_modalities:
            print(f"Failed Modalities  : {', '.join(self.fusion.failed_modalities)}")
        print(f"Conflict Detected  : {'YES' if self.has_conflict else 'No'}")

        print(f"\n[EXPLANATION - {self.explanation.llm_provider.upper()}]")
        print(f"Headline : {self.headline}")
        print(f"Why      : {self.why}")

        print("\n[GROUNDED EVIDENCE CITATIONS]")
        for ev in self.evidence_citations:
            print(f"  - {ev}")

        if self.explanation.conflict_warning:
            print(f"\n[CONFLICT WARNING]\n  {self.explanation.conflict_warning}")

        if self.explanation.missing_data_disclaimer:
            print(f"\n[DISCLAIMER]\n  {self.explanation.missing_data_disclaimer}")

        print("\n[RECOMMENDED ACTIONS]")
        for action in self.recommended_actions:
            print(f"  * {action}")
        print("=" * 70 + "\n")


class MultimodalPipeline:
    """Orchestrates live inference across both audio and URL modules with fusion."""

    def __init__(
        self,
        fusion_config: Optional[FusionConfig] = None,
        explainer_config: Optional[ExplainerConfig] = None,
    ):
        self.fusion = ModelFusion(fusion_config)
        self.explainer = LLMExplainer(explainer_config)

    def analyze(
        self,
        audio_path: Optional[str] = None,
        url: Optional[str] = None,
        text: Optional[str] = None,
        screenshot_path: Optional[str] = None,
        raw_modalities: Optional[Dict[str, Optional[Union[ModalityResult, Dict[str, Any]]]]] = None,
        include_domain_intelligence: bool = True,
        url_provider: str = "deterministic",
        explainer_provider: Optional[str] = None,
    ) -> MultimodalAnalysisResult:
        """Executes live inference on all supplied modalities, fuses them, and explains.

        Args:
            audio_path: Path to .wav or audio file for live audio models.
            url: URL string for live 26-feature XGBoost & Domain OSINT module.
            text: SMS or transcript text string.
            screenshot_path: Path to webpage or credential harvest screenshot.
            raw_modalities: Optional pre-computed or custom ModalityResults.
            include_domain_intelligence: Whether URL evaluation includes live DNS/WHOIS/TLS.
            url_provider: LLM provider for URL module reasoning ('deterministic', 'gemini').
            explainer_provider: Override LLM provider for the multimodal explanation.
        """
        modalities: Dict[str, Optional[Union[ModalityResult, Dict[str, Any]]]] = {}

        from adapters.audio_adapter import AudioAdapter
        from adapters.screenshot_adapter import ScreenshotAdapter
        from adapters.text_adapter import TextAdapter
        from adapters.url_adapter import URLAdapter

        # 1. Live Audio Detection (Scam + Anti-Spoofing)
        if audio_path is not None:
            logger.info("Executing live audio models on: %s", audio_path)
            scam_res = AudioAdapter.predict_scam_call_file(audio_path)
            spoof_res = AudioAdapter.predict_spoof_file(audio_path)
            modalities["audio_scam"] = scam_res
            modalities["audio_spoof"] = spoof_res

        # 2. Live URL & Domain Intelligence Module
        if url is not None:
            logger.info("Executing live XGBoost URL analysis on: %s", url)
            url_res = URLAdapter.predict_url(
                url=url,
                include_domain_intelligence=include_domain_intelligence,
                provider=url_provider,
            )
            modalities["url"] = url_res

        # 3. Text / SMS Scam Detection
        if text is not None:
            text_res = TextAdapter.predict_text(text)
            modalities["text"] = text_res

        # 4. Screenshot Detection
        if screenshot_path is not None:
            shot_res = ScreenshotAdapter.predict_screenshot(screenshot_path)
            modalities["screenshot"] = shot_res

        # 5. Merge any custom or pre-computed modalities
        if raw_modalities:
            for k, v in raw_modalities.items():
                modalities[k] = v

        # 6. Run Fusion
        fusion_result = self.fusion.fuse(modalities)

        # 7. Generate Grounded Explanation
        if explainer_provider:
            orig_provider = self.explainer.config.provider
            self.explainer.config.provider = explainer_provider
            try:
                explanation = self.explainer.explain(fusion_result)
            finally:
                self.explainer.config.provider = orig_provider
        else:
            explanation = self.explainer.explain(fusion_result)

        # Build clean ModalityResult lookup
        parsed_modalities: Dict[str, ModalityResult] = {}
        for mod_key, res in fusion_result.modality_breakdown.items():
            if isinstance(res, ModalityResult):
                parsed_modalities[mod_key] = res

        return MultimodalAnalysisResult(
            fusion=fusion_result,
            explanation=explanation,
            modality_results=parsed_modalities,
        )


def analyze_multimodal(
    audio_path: Optional[str] = None,
    url: Optional[str] = None,
    text: Optional[str] = None,
    screenshot_path: Optional[str] = None,
    include_domain_intelligence: bool = True,
    explainer_provider: str = "deterministic",
) -> MultimodalAnalysisResult:
    """Convenience functional API for multimodal threat analysis.

    Example:
        from fusion import analyze_multimodal
        result = analyze_multimodal(
            audio_path="audio_model/dataset/SCAM_CALLS/scam_22.wav",
            url="http://paypal.com-verify.online/login",
        )
        result.print_summary()
    """
    pipeline = MultimodalPipeline(
        explainer_config=ExplainerConfig(provider=explainer_provider)
    )
    return pipeline.analyze(
        audio_path=audio_path,
        url=url,
        text=text,
        screenshot_path=screenshot_path,
        include_domain_intelligence=include_domain_intelligence,
    )
