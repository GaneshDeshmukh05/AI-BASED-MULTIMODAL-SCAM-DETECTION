"""schemas.py
Common schemas and contracts for Multimodal Scam & Fraud Detection.
Provides standardized representations for detector results, fusion verdicts,
and LLM-generated explanations.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ModalityType(str, Enum):
    AUDIO_SCAM = "audio_scam"        # Scam call conversational / acoustic pattern
    AUDIO_SPOOF = "audio_spoof"      # Synthetic voice / voice clone / deepfake
    TEXT = "text"                    # SMS / email / chat transcript
    URL = "url"                      # Phishing link / malicious URL
    SCREENSHOT = "screenshot"        # Phishing webpage UI / fake document screenshot
    DOMAIN = "domain"                # Domain intelligence / WHOIS / DNS


class DetectorStatus(str, Enum):
    SUCCESS = "success"              # Successfully processed and returned prediction
    FAILED = "failed"                # Error during inference or preprocessing
    NOT_PROVIDED = "not_provided"    # Modality input was not supplied by user
    SKIPPED = "skipped"              # Intentionally bypassed or disabled


class RiskLevel(str, Enum):
    LOW = "LOW"                      # 0.00 <= risk < 0.35
    MEDIUM = "MEDIUM"                # 0.35 <= risk < 0.65
    HIGH = "HIGH"                    # 0.65 <= risk < 0.85
    CRITICAL = "CRITICAL"            # 0.85 <= risk <= 1.00
    INCONCLUSIVE = "INCONCLUSIVE"    # No reliable signals available


class OverallVerdict(str, Enum):
    SCAM = "SCAM"                                # Confirmed scam / malicious threat detected
    SUSPICIOUS = "SUSPICIOUS"                    # Elevated risk / synthetic audio / partial threat
    SAFE = "SAFE"                                # Observed indicators appear legitimate
    CONFLICT_WARNING = "CONFLICT_WARNING"        # Detectors strongly disagree
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"      # No valid inputs or all failed


class ModalityResult(BaseModel):
    """Standardized result returned by or adapted from any detector.

    Separates risk_score (how indicative of fraud/threat) from confidence (certainty).
    Missing and failed detectors have risk_score=None and confidence=None so they
    are never mathematically aggregated into risk calculations.
    """
    modality: str = Field(..., description="Modality identifier (e.g. audio_scam, audio_spoof, url)")
    status: DetectorStatus = Field(default=DetectorStatus.SUCCESS, description="Execution status")
    label: str = Field(default="unknown", description="Detector-specific label (e.g. scam, normal, spoof, bonafide)")
    risk_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threat/scam likelihood: 0.0 (safe) to 1.0 (malicious). None if missing or failed.",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Model prediction certainty: 0.0 (uncertain) to 1.0 (certain). None if missing or failed.",
    )
    threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Decision threshold used by detector")
    is_threat: bool = Field(default=False, description="Whether this modality independently flags a threat")
    is_validated: bool = Field(
        default=True,
        description="Whether this detector has been evaluated and validated for operational deployment.",
    )
    validation_notes: Optional[str] = Field(
        default=None,
        description="Operational validation state or disclaimer (e.g. experimental, uncalibrated).",
    )
    evidence: List[str] = Field(default_factory=list, description="Factual evidence statements from the model")
    raw_output: Dict[str, Any] = Field(
        default_factory=dict, description="Original unadapted model response"
    )
    error_message: Optional[str] = Field(default=None, description="Error details if status is FAILED")
    latency_ms: Optional[float] = Field(default=None, description="Inference time in milliseconds")

    @classmethod
    def failed(cls, modality: str, error: str) -> "ModalityResult":
        """Factory for failed detector execution (unknown risk, excluded from math)."""
        return cls(
            modality=modality,
            status=DetectorStatus.FAILED,
            label="error",
            risk_score=None,
            confidence=None,
            threshold=0.5,
            is_threat=False,
            evidence=[f"Detector failed: {error}"],
            error_message=error,
            raw_output={},
        )

    @classmethod
    def not_provided(cls, modality: str) -> "ModalityResult":
        """Factory for unsupplied modalities (unknown risk, excluded from math)."""
        return cls(
            modality=modality,
            status=DetectorStatus.NOT_PROVIDED,
            label="not_provided",
            risk_score=None,
            confidence=None,
            threshold=0.5,
            is_threat=False,
            evidence=[],
            raw_output={},
        )


class FusionResult(BaseModel):
    """Consolidated assessment produced by the Fusion module."""
    verdict: OverallVerdict = Field(..., description="High-level categorical verdict")
    risk_level: RiskLevel = Field(..., description="Risk tier: LOW, MEDIUM, HIGH, CRITICAL, INCONCLUSIVE")
    overall_risk_score: float = Field(..., ge=0.0, le=1.0, description="Aggregated risk score [0, 1]")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Aggregated confidence [0, 1]")
    active_modalities: List[str] = Field(default_factory=list, description="Modalities that provided successful results")
    missing_modalities: List[str] = Field(default_factory=list, description="Modalities not provided")
    failed_modalities: List[str] = Field(default_factory=list, description="Modalities that encountered errors")
    has_conflict: bool = Field(default=False, description="Whether strong contradictory signals were detected")
    conflict_details: Optional[str] = Field(default=None, description="Description of the disagreement if any")
    evidence_summary: List[str] = Field(default_factory=list, description="Consolidated deduplicated evidence points")
    modality_breakdown: Dict[str, ModalityResult] = Field(default_factory=dict, description="Detailed per-modality results")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic and fusion configuration details")


class ExplanationResult(BaseModel):
    """User-facing structured explanation generated by the LLM Explainer."""
    verdict: str = Field(..., description="Overall verdict (e.g. Potential Scam, Safe, Suspicious)")
    risk_level: str = Field(..., description="Risk tier string (e.g. HIGH, CRITICAL, LOW)")
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Overall risk score")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Overall confidence score")
    headline: str = Field(..., description="Punchy, clear summary headline")
    why: str = Field(..., description="Clear non-technical explanation of the reasons")
    evidence_citations: List[str] = Field(default_factory=list, description="Factual evidence items strictly from detectors")
    conflict_warning: Optional[str] = Field(default=None, description="Warning if models disagreed")
    missing_data_disclaimer: Optional[str] = Field(default=None, description="Disclaimer regarding unexamined modalities")
    recommended_actions: List[str] = Field(default_factory=list, description="Concrete next steps for the user")
    llm_provider: str = Field(default="deterministic_fallback", description="Engine used (gemini, openai, fallback)")
    raw_llm_response: Optional[str] = Field(default=None, description="Raw LLM text if generated by API")
