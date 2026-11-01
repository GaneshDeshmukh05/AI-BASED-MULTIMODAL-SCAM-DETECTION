"""url_adapter.py
Adapter for URL and domain phishing detectors.
Translates offline lexical features, XGBoost predictions, SHAP risk factors,
and live domain OSINT (DNS, WHOIS age, TLS certs, brand lookalikes).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fusion.schemas import (
    DetectorStatus,
    ModalityResult,
    ModalityType,
)

logger = logging.getLogger("url_adapter")


class URLAdapter:
    """Adapts URL analysis responses (including XGBoost & SHAP) into ModalityResult."""

    _model_service: Optional[Any] = None
    _threat_analyst: Optional[Any] = None

    @classmethod
    def _ensure_url_module_path(cls) -> None:
        """Ensures the domain_intelligence module directory is on sys.path."""
        import sys
        from pathlib import Path

        cur = Path(__file__).resolve().parent
        candidate_dirs = [
            cur.parent / "domain_intelligence",
                        cur / "domain_intelligence",
                    ]
        url_dir = None
        for cd in candidate_dirs:
            if cd.is_dir() and (cd / "src").is_dir():
                url_dir = cd
                break

        if url_dir and str(url_dir) not in sys.path:
            sys.path.insert(0, str(url_dir))

    @classmethod
    def load_model_service(cls) -> Any:
        """Loads and caches the 26-feature XGBoost & SHAP ModelService."""
        if cls._model_service is None:
            cls._ensure_url_module_path()
            from src.model_service import get_model_service

            svc = get_model_service()
            svc.load()
            cls._model_service = svc
        return cls._model_service

    @classmethod
    def load_threat_analyst(cls, provider: str = "deterministic") -> Any:
        """Loads and caches the composite URLThreatAnalyst."""
        if cls._threat_analyst is None:
            cls._ensure_url_module_path()
            from src.llm_analyst import URLThreatAnalyst

            cls._threat_analyst = URLThreatAnalyst(provider=provider)
        return cls._threat_analyst

    @classmethod
    def predict_url(
        cls,
        url: str,
        include_domain_intelligence: bool = True,
        provider: str = "deterministic",
    ) -> ModalityResult:
        """Runs live feature extraction and inference on a URL.

        Args:
            url: The target URL (e.g., https://paypal.com-verify.online/login).
            include_domain_intelligence: If True, queries passive DNS, WHOIS age,
                SSL validation, and threat feeds. If False, runs fast offline
                26-feature XGBoost + SHAP inference only.
            provider: LLM analyst provider ('deterministic', 'gemini', 'openai').
        """
        try:
            if not url or not isinstance(url, str):
                return ModalityResult.failed(
                    ModalityType.URL.value, "URL must be a non-empty string"
                )

            if include_domain_intelligence:
                analyst = cls.load_threat_analyst(provider=provider)
                raw = analyst.gather_telemetry(
                    url,
                    include_domain_intelligence=True,
                )
            else:
                service = cls.load_model_service()
                raw = service.analyze(url)

            return cls.adapt(raw, url_string=url)
        except Exception as exc:
            msg = str(exc)
            if "Application Control" in msg or isinstance(exc, ImportError):
                logger.warning("URL threat prediction runtime unavailable on %s: %s", url, exc)
            else:
                logger.exception("Failed live URL threat prediction on %s", url)
            return ModalityResult.failed(ModalityType.URL.value, f"URL inference error: {exc}")

    @staticmethod
    def adapt(
        raw: Dict[str, Any],
        url_string: Optional[str] = None,
    ) -> ModalityResult:
        try:
            if not isinstance(raw, dict):
                return ModalityResult.failed(
                    ModalityType.URL.value,
                    f"Expected dict from URL detector, got {type(raw).__name__}",
                )

            prediction = str(raw.get("prediction", raw.get("label", "unknown"))).lower()
            threshold = float(raw.get("operating_threshold", raw.get("threshold", 0.5)))

            # Extract probability / score
            if "composite_risk_score" in raw and raw["composite_risk_score"] is not None:
                comp_score = float(raw["composite_risk_score"])
                score = comp_score / 100.0 if comp_score > 1.0 else comp_score
            elif "phishing_probability" in raw and raw["phishing_probability"] is not None:
                score = float(raw["phishing_probability"])
            elif "risk_score" in raw and raw["risk_score"] is not None:
                raw_risk = float(raw["risk_score"])
                score = raw_risk / 100.0 if raw_risk > 1.0 else raw_risk
            else:
                score = float(raw.get("score", 0.5))

            score = max(0.0, min(1.0, score))

            threat_intel = raw.get("threat_intelligence") or {}
            is_feed_phish = bool(threat_intel.get("is_known_phishing"))

            is_threat = prediction == "phishing" or score >= threshold or is_feed_phish

            confidence = float(raw.get("confidence", min(1.0, 0.5 + abs(score - threshold))))

            # Compile grounded evidence from SHAP risk factors or extracted features
            evidence: List[str] = []
            target_url = url_string or raw.get("url", "")
            prefix = f"URL '{target_url}': " if target_url else "Link analysis: "

            if is_threat:
                evidence.append(
                    f"{prefix}Phishing classifier flagged link as malicious with probability {score:.3f} "
                    f"(operating threshold {threshold:.2f})."
                )
            else:
                evidence.append(
                    f"{prefix}Phishing classifier evaluated link as legitimate with probability {score:.3f}."
                )

            # Add top SHAP risk factors if available
            top_risks = raw.get("top_risk_factors", [])
            for item in top_risks[:3]:
                if isinstance(item, dict):
                    feat = item.get("feature", "unknown")
                    shap_val = item.get("shap_value", 0.0)
                    evidence.append(f"High-risk factor: '{feat}' contributed +{shap_val:.3f} toward phishing.")
                elif isinstance(item, str):
                    evidence.append(f"High-risk indicator: {item}")

            # Fallback to legacy top_features if present
            if not top_risks and raw.get("top_features"):
                features = ", ".join(raw["top_features"][:3])
                evidence.append(f"Key suspicious lexical features: {features}")

            # Domain Intelligence enrichments (WHOIS age, brand lookalikes, SSL)
            domain_intel = raw.get("domain_intelligence")
            if isinstance(domain_intel, dict):
                lookalike = domain_intel.get("lookalike") or {}
                if lookalike.get("matched_brand"):
                    brand = lookalike["matched_brand"]
                    evidence.append(f"Brand spoofing detected: Domain targets brand '{brand}'.")

                age_days = domain_intel.get("age_days")
                if age_days is not None:
                    if age_days < 30:
                        evidence.append(
                            f"Domain registration is very recent ({age_days} days old), "
                            "a primary indicator of temporary phishing infrastructure."
                        )
                    elif age_days >= 365:
                        evidence.append(f"Domain is established ({age_days} days old).")

            # SSL / TLS certificate analysis
            ssl_info = raw.get("ssl_analysis")
            if isinstance(ssl_info, dict):
                if ssl_info.get("has_ssl") is False:
                    evidence.append("Insecure transport: URL does not use SSL/TLS encryption.")
                elif ssl_info.get("valid") is False:
                    evidence.append("Invalid or untrusted SSL/TLS certificate detected.")

            # Threat feeds
            if is_feed_phish:
                evidence.append("Threat intelligence feed match: URL is confirmed malicious on community threat feeds.")

            # Risk score adjustments from telemetry
            adjustments = raw.get("risk_adjustments", [])
            for adj in adjustments[:2]:
                if isinstance(adj, dict) and adj.get("reason"):
                    evidence.append(f"Risk adjustment: {adj['reason']} (+{adj.get('points', 0)} pts)")

            return ModalityResult(
                modality=ModalityType.URL.value,
                status=DetectorStatus.SUCCESS,
                label=prediction,
                risk_score=score,
                confidence=round(confidence, 4),
                threshold=threshold,
                is_threat=is_threat,
                evidence=evidence,
                raw_output=raw,
            )

        except Exception as exc:
            logger.exception("Failed to adapt URL detector output")
            return ModalityResult.failed(ModalityType.URL.value, str(exc))

    @staticmethod
    def create_mock(
        is_phishing: bool = True,
        url: str = "http://secure-login.bank-update.xyz/verify",
        probability: Optional[float] = None,
    ) -> ModalityResult:
        """Helper to generate realistic mock output for testing."""
        prob = probability if probability is not None else (0.92 if is_phishing else 0.08)
        raw = {
            "url": url,
            "prediction": "phishing" if is_phishing else "legitimate",
            "phishing_probability": prob,
            "confidence": 0.91,
            "operating_threshold": 0.5,
            "risk_level": "critical" if is_phishing else "low",
            "top_risk_factors": (
                [
                    {"feature": "ip_in_url", "shap_value": 0.35, "direction": "phishing"},
                    {"feature": "suspicious_keyword_count", "shap_value": 0.28, "direction": "phishing"},
                    {"feature": "url_entropy", "shap_value": 0.19, "direction": "phishing"},
                ]
                if is_phishing
                else []
            ),
        }
        return URLAdapter.adapt(raw, url_string=url)
