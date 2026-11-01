"""Fusion and Explanation package for Multimodal Scam Detection."""

from .schemas import (
    ModalityType,
    DetectorStatus,
    RiskLevel,
    OverallVerdict,
    ModalityResult,
    FusionResult,
    ExplanationResult,
)
from .fusion import ModelFusion, FusionConfig
from .llm_explainer import LLMExplainer, ExplainerConfig
from .pipeline import MultimodalPipeline, MultimodalAnalysisResult, analyze_multimodal

__all__ = [
    "ModalityType",
    "DetectorStatus",
    "RiskLevel",
    "OverallVerdict",
    "ModalityResult",
    "FusionResult",
    "ExplanationResult",
    "ModelFusion",
    "FusionConfig",
    "LLMExplainer",
    "ExplainerConfig",
    "MultimodalPipeline",
    "MultimodalAnalysisResult",
    "analyze_multimodal",
]
