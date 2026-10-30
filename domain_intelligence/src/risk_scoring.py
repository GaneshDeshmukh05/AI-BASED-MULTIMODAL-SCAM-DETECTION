"""Combine the URL-only model probability with domain-intelligence signals.

Pure functions, no I/O. The model probability is never altered; adjustments are
applied to a 0-100 composite score and each one is reported so a client can show
*why* the composite differs from the model.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from .config import (
    ESTABLISHED_DOMAIN_DAYS,
    ESTABLISHED_DOMAIN_FACTOR,
    INVALID_CERT_PENALTY,
    MATURE_DOMAIN_DAYS,
    MATURE_DOMAIN_FACTOR,
    NEW_DOMAIN_PENALTY,
)

if TYPE_CHECKING:
    from .evidence import DomainEvidence

# Risk points added when URLhaus lists the hostname.
URLHAUS_PENALTY = 20
# Risk points added when the domain is a confirmed lookalike/typosquat.
LOOKALIKE_PENALTY = 10


def risk_level(probability: float) -> str:
    """Same bands as the model service, applied to the composite probability."""
    if probability >= 0.80:
        return "critical"
    if probability >= 0.60:
        return "high"
    if probability >= 0.40:
        return "medium"
    return "low"


class EvidenceAdapter:
    """Unified adapter supporting DomainEvidence objects, tuples, or raw dicts."""

    def __init__(self, domain_ev: Any = None, tls_ev: Any = None) -> None:
        self._domain_ev = domain_ev
        self._tls_ev = tls_ev or {}

        # Handle tuple inputs: (domain_dict, ssl_dict)
        if isinstance(domain_ev, tuple) and len(domain_ev) == 2:
            domain_ev, tls_ev = domain_ev[0], domain_ev[1]
            self._domain_ev = domain_ev
            self._tls_ev = tls_ev or {}

        if isinstance(domain_ev, dict):
            self._domain_dict = domain_ev
        elif domain_ev is not None and hasattr(domain_ev, "to_legacy_domain_dict"):
            self._domain_dict = domain_ev.to_legacy_domain_dict() or {}
        else:
            self._domain_dict = {}

        if isinstance(tls_ev, dict):
            self._tls_dict = tls_ev
        elif domain_ev is not None and hasattr(domain_ev, "to_legacy_ssl_dict"):
            self._tls_dict = domain_ev.to_legacy_ssl_dict() or {}
        else:
            self._tls_dict = {}

    @property
    def is_suspicious_new_domain(self) -> bool:
        if self._domain_ev is not None and hasattr(self._domain_ev, "is_suspicious_new_domain"):
            return bool(getattr(self._domain_ev, "is_suspicious_new_domain", False))
        return bool(self._domain_dict.get("is_suspicious_new_domain", False))

    @property
    def domain_age_days(self) -> int | None:
        if self._domain_ev is not None and hasattr(self._domain_ev, "domain_age_days"):
            return getattr(self._domain_ev, "domain_age_days", None)
        return self._domain_dict.get("domain_age_days")

    @property
    def tls_check_status(self) -> str:
        if self._domain_ev is not None and hasattr(self._domain_ev, "tls_check_status"):
            return getattr(self._domain_ev, "tls_check_status", "unknown")
        status = (
            self._tls_dict.get("check_status")
            or self._tls_dict.get("status")
            or self._tls_dict.get("tls_check_status")
            or self._domain_dict.get("tls_check_status")
        )
        return status or "unknown"

    @property
    def has_valid_ssl(self) -> bool:
        if self._domain_ev is not None and hasattr(self._domain_ev, "has_valid_ssl"):
            return bool(getattr(self._domain_ev, "has_valid_ssl", False))
        if "has_valid_ssl" in self._tls_dict:
            return bool(self._tls_dict["has_valid_ssl"])
        return self.tls_check_status == "valid"

    @property
    def is_shared_hosting(self) -> bool:
        if self._domain_ev is not None and hasattr(self._domain_ev, "is_shared_hosting"):
            return bool(getattr(self._domain_ev, "is_shared_hosting", False))
        return bool(self._domain_dict.get("is_shared_hosting", False))

    @property
    def urlhaus_listed(self) -> bool:
        if self._domain_ev is not None and hasattr(self._domain_ev, "urlhaus_listed"):
            return bool(getattr(self._domain_ev, "urlhaus_listed", False))
        return bool(self._domain_dict.get("urlhaus_listed", False))

    @property
    def urlhaus_status_stale(self) -> bool:
        if self._domain_ev is not None and hasattr(self._domain_ev, "urlhaus_status_stale"):
            return bool(getattr(self._domain_ev, "urlhaus_status_stale", False))
        return bool(self._domain_dict.get("urlhaus_status_stale", False))

    @property
    def lookalike_is_lookalike(self) -> bool:
        if self._domain_ev is not None and hasattr(self._domain_ev, "lookalike"):
            lookalike = getattr(self._domain_ev, "lookalike", None)
            return bool(getattr(lookalike, "is_lookalike", False))
        lookalike = self._domain_dict.get("lookalike")
        if isinstance(lookalike, dict):
            return bool(lookalike.get("is_lookalike", False))
        elif hasattr(lookalike, "is_lookalike"):
            return bool(getattr(lookalike, "is_lookalike", False))
        return bool(self._domain_dict.get("is_lookalike", False))

    @property
    def lookalike_matched_brand(self) -> str | None:
        if self._domain_ev is not None and hasattr(self._domain_ev, "lookalike"):
            lookalike = getattr(self._domain_ev, "lookalike", None)
            return getattr(lookalike, "matched_brand", None)
        lookalike = self._domain_dict.get("lookalike")
        if isinstance(lookalike, dict):
            return lookalike.get("matched_brand")
        elif hasattr(lookalike, "matched_brand"):
            return getattr(lookalike, "matched_brand", None)
        return self._domain_dict.get("matched_brand")

    @property
    def lookalike_technique(self) -> str | None:
        if self._domain_ev is not None and hasattr(self._domain_ev, "lookalike"):
            lookalike = getattr(self._domain_ev, "lookalike", None)
            return getattr(lookalike, "technique", None)
        lookalike = self._domain_dict.get("lookalike")
        if isinstance(lookalike, dict):
            return lookalike.get("technique")
        elif hasattr(lookalike, "technique"):
            return getattr(lookalike, "technique", None)
        return self._domain_dict.get("technique")

    @property
    def lookalike_edit_distance(self) -> int | None:
        if self._domain_ev is not None and hasattr(self._domain_ev, "lookalike"):
            lookalike = getattr(self._domain_ev, "lookalike", None)
            return getattr(lookalike, "edit_distance", None)
        lookalike = self._domain_dict.get("lookalike")
        if isinstance(lookalike, dict):
            return lookalike.get("edit_distance")
        elif hasattr(lookalike, "edit_distance"):
            return getattr(lookalike, "edit_distance", None)
        return self._domain_dict.get("edit_distance")

    def to_legacy_ssl_dict(self) -> dict[str, Any]:
        if self._domain_ev is not None and hasattr(self._domain_ev, "to_legacy_ssl_dict"):
            return self._domain_ev.to_legacy_ssl_dict() or {}
        return self._tls_dict

    def to_legacy_domain_dict(self) -> dict[str, Any]:
        if self._domain_ev is not None and hasattr(self._domain_ev, "to_legacy_domain_dict"):
            return self._domain_ev.to_legacy_domain_dict() or {}
        return self._domain_dict


def compose_risk(
    ml_probability: float,
    threshold: float,
    domain_evidence: Any = None,
    arg4: Any = True,
    arg5: Any = None,
) -> dict[str, Any]:
    """Return composite score, decision, and applied adjustments.

    Supports both:
    - 4-arg signature: compose_risk(prob, threshold, domain_evidence, url_is_https)
    - 5-arg signature: compose_risk(prob, threshold, domain_dict, tls_dict, url_is_https)
    """
    tls_evidence = None
    url_is_https = True

    if arg5 is not None:
        # 5 positional args passed: (ml_prob, threshold, domain_dict, tls_dict, url_is_https)
        tls_evidence = arg4
        url_is_https = bool(arg5)
    elif isinstance(domain_evidence, tuple) and len(domain_evidence) == 2:
        # Tuple passed as 3rd arg: (domain_dict, ssl_dict)
        domain_dict, tls_dict = domain_evidence[0], domain_evidence[1]
        domain_evidence = domain_dict
        tls_evidence = tls_dict
        url_is_https = bool(arg4)
    else:
        # 4 positional args passed: (ml_prob, threshold, domain_evidence, url_is_https)
        url_is_https = bool(arg4)

    score = ml_probability * 100.0
    adjustments: list[dict[str, Any]] = []

    if domain_evidence is None and tls_evidence is None:
        return _finish(threshold, score, adjustments, intelligence_used=False)

    ev = EvidenceAdapter(domain_evidence, tls_evidence)

    # --- New domain penalty ---
    if ev.is_suspicious_new_domain:
        score = min(100.0, score + NEW_DOMAIN_PENALTY)
        adjustments.append({
            "signal": "new_domain",
            "effect": f"+{NEW_DOMAIN_PENALTY} points",
            "reason": f"Domain registered {ev.domain_age_days} day(s) ago.",
        })

    # --- Invalid certificate penalty ---
    ssl_status = ev.tls_check_status
    if url_is_https and ssl_status == "invalid":
        score = min(100.0, score + INVALID_CERT_PENALTY)
        tls_dict = ev.to_legacy_ssl_dict()
        detail = "self-signed certificate" if tls_dict.get("is_self_signed") else "certificate failed verification"
        adjustments.append({
            "signal": "invalid_certificate",
            "effect": f"+{INVALID_CERT_PENALTY} points",
            "reason": f"HTTPS URL presents a {detail}. {tls_dict.get('error') or ''}".strip(),
        })

    # --- URLhaus penalty ---
    if ev.urlhaus_listed and not ev.urlhaus_status_stale:
        score = min(100.0, score + URLHAUS_PENALTY)
        adjustments.append({
            "signal": "urlhaus_listed",
            "effect": f"+{URLHAUS_PENALTY} points",
            "reason": "Hostname is listed in URLhaus malware database.",
        })

    # --- Lookalike / typosquat penalty ---
    if ev.lookalike_is_lookalike:
        score = min(100.0, score + LOOKALIKE_PENALTY)
        adjustments.append({
            "signal": "lookalike_domain",
            "effect": f"+{LOOKALIKE_PENALTY} points",
            "reason": (
                f"Domain resembles '{ev.lookalike_matched_brand}' "
                f"(technique: {ev.lookalike_technique}, "
                f"edit_distance: {ev.lookalike_edit_distance})."
            ),
        })

    # --- Established domain discount ---
    age = ev.domain_age_days or 0
    has_valid_cert = ev.has_valid_ssl and ssl_status == "valid"
    if has_valid_cert and url_is_https and not ev.is_shared_hosting:
        factor = None
        if age > ESTABLISHED_DOMAIN_DAYS:
            factor = ESTABLISHED_DOMAIN_FACTOR
        elif age > MATURE_DOMAIN_DAYS:
            factor = MATURE_DOMAIN_FACTOR
        if factor is not None:
            score *= factor
            adjustments.append({
                "signal": "established_domain",
                "effect": f"x{factor}",
                "reason": f"Domain is {age // 365} years old with a valid certificate.",
            })
    elif has_valid_cert and ev.is_shared_hosting and age > MATURE_DOMAIN_DAYS:
        adjustments.append({
            "signal": "shared_hosting_no_discount",
            "effect": "no change",
            "reason": "Host is on a shared-hosting/user-content domain; domain age is not used to lower risk.",
        })

    intel_used = domain_evidence is not None or tls_evidence is not None
    return _finish(threshold, score, adjustments, intelligence_used=intel_used)


def _finish(threshold: float, score: float, adjustments: list[dict[str, Any]], intelligence_used: bool) -> dict[str, Any]:
    score = max(0.0, min(100.0, score))
    composite_probability = score / 100.0
    is_phishing = composite_probability >= threshold
    return {
        "composite_probability": composite_probability,
        "composite_risk_score": round(score),
        "prediction": "phishing" if is_phishing else "legitimate",
        "risk_level": risk_level(composite_probability),
        "adjustments": adjustments,
        "intelligence_used": intelligence_used,
    }


def merge_intelligence(
    model_result: dict[str, Any],
    domain_evidence: Any = None,
    tls_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``model_result`` extended with the composite decision."""
    probability = float(model_result["phishing_probability"])
    threshold = float(model_result["operating_threshold"])
    url_is_https = bool(model_result.get("extracted_features", {}).get("uses_https", True))

    if isinstance(domain_evidence, tuple) and len(domain_evidence) == 2:
        domain_evidence, tls_evidence = domain_evidence[0], domain_evidence[1]

    decision = compose_risk(probability, threshold, domain_evidence, tls_evidence, url_is_https)
    composite_probability = decision["composite_probability"]
    merged = dict(model_result)

    ev = EvidenceAdapter(domain_evidence, tls_evidence)
    domain_intel_dict = ev.to_legacy_domain_dict() if (domain_evidence is not None or tls_evidence is not None) else None
    ssl_intel_dict = ev.to_legacy_ssl_dict() if (domain_evidence is not None or tls_evidence is not None) else None

    domain_ev_dump = None
    if domain_evidence is not None:
        if hasattr(domain_evidence, "model_dump"):
            domain_ev_dump = domain_evidence.model_dump()
        elif isinstance(domain_evidence, dict):
            domain_ev_dump = domain_evidence

    merged.update(
        {
            "ml_prediction": model_result.get("prediction"),
            "ml_risk_score": model_result.get("risk_score"),
            "prediction": decision["prediction"],
            "risk_score": decision["composite_risk_score"],
            "composite_risk_score": decision["composite_risk_score"],
            "risk_level": decision["risk_level"],
            "confidence": round(max(composite_probability, 1 - composite_probability), 6),
            "risk_adjustments": decision["adjustments"],
            "domain_intelligence_used": decision["intelligence_used"],
            "domain_intelligence": domain_intel_dict,
            "ssl_analysis": ssl_intel_dict,
            "domain_evidence": domain_ev_dump,
        }
    )
    return merged