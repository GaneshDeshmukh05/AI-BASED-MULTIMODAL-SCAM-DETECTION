"""Validated public API request and response contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from src.feature_engineering import InvalidURLException, normalise_url


class URLRequest(BaseModel):
    """One URL to classify. The model is offline; domain enrichment is optional."""

    url: str = Field(..., min_length=1, max_length=4096, description="HTTP(S) URL or schemeless hostname/path")
    include_domain_intelligence: bool | None = Field(
        default=None,
        description="Run DNS/RDAP/TLS/lookalike enrichment. Omit to use the server default (PHISHING_DOMAIN_INTEL). "
        "Set false for a purely offline, model-only result.",
    )
    include_redirect_resolution: bool = Field(
        default=False,
        description="Follow redirect chain (opt-in). Runs in an isolated subprocess with SSRF guards.",
    )

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        try:
            normalise_url(value)
        except InvalidURLException as error:
            raise ValueError(str(error)) from error
        return value.strip()


class RiskFactor(BaseModel):
    feature: str
    shap_value: float
    direction: Literal["phishing", "legitimate"]


class RiskAdjustment(BaseModel):
    signal: str
    effect: str
    reason: str


class URLAnalysisResponse(BaseModel):
    url: str
    # Final decision. With domain intelligence off (or all lookups unknown) this
    # equals the model-only verdict; otherwise it reflects the composite score.
    prediction: Literal["phishing", "legitimate"]
    # Raw XGBoost probability, never modified by domain intelligence.
    phishing_probability: float = Field(..., ge=0, le=1)
    risk_score: int = Field(..., ge=0, le=100)
    composite_risk_score: int | None = Field(default=None, ge=0, le=100)
    ml_prediction: Literal["phishing", "legitimate"] | None = None
    ml_risk_score: int | None = Field(default=None, ge=0, le=100)
    risk_adjustments: list[RiskAdjustment] = Field(default_factory=list)
    domain_intelligence_used: bool = False
    # Legacy flat dicts (backward compat)
    domain_intelligence: dict[str, Any] | None = None
    ssl_analysis: dict[str, Any] | None = None
    # Full typed domain evidence (new, richer format)
    domain_evidence: dict[str, Any] | None = None
    confidence: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high", "critical"]
    operating_threshold: float = Field(..., ge=0, le=1)
    extracted_features: dict[str, float]
    top_risk_factors: list[RiskFactor]
    top_legitimate_factors: list[RiskFactor]
    # Legacy aliases retained for clients of the original API.
    top_features: list[str]
    shap_contributions: dict[str, float]
    shap_explanation: dict[str, float]
    # Exact match against the cached feed; a miss is not a clean verdict.
    threat_intelligence: dict[str, Any] | None = None
    # Optional redirect chain (only when include_redirect_resolution=True)
    redirect_chain: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    feature_count: int
    operating_threshold: float
