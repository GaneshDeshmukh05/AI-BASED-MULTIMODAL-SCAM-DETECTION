"""Calibrated logistic-regression domain scorer.

Replaces the hand-picked penalty/discount constants in ``src.risk_scoring``
once a ``domain_scorer.joblib`` artifact has been produced by
``experiments/evaluate_domain_signals.py``.

The scorer takes a flat feature vector that combines:
  * The raw XGBoost ``phishing_probability`` (the ML-only signal).
  * Numeric encodings of the domain-intelligence signals
    (age, cert validity, URLhaus, lookalike, shared-hosting …).

It outputs a calibrated composite probability in [0, 1].

When the scorer artifact is unavailable (e.g. during development or CI)
the existing hand-picked ``compose_risk()`` is used unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .config import DOMAIN_SCORER_PATH, NEW_DOMAIN_DAYS

if TYPE_CHECKING:
    from .evidence import DomainEvidence

log = logging.getLogger("phishing_url_detector.domain_scorer")

# Column order must match experiments/evaluate_domain_signals.py.
SCORER_FEATURES = [
    "ml_probability",
    "domain_age_days",          # clipped to [0, 10_000], 0 when unknown
    "domain_age_log",           # log1p of domain_age_days
    "is_suspicious_new_domain", # 1/0
    "has_valid_ssl",            # 1/0
    "is_self_signed",           # 1/0
    "ssl_days_left",            # clipped to [0, 1000], 0 when unknown
    "is_shared_hosting",        # 1/0
    "is_lookalike",             # 1/0
    "urlhaus_listed",           # 1/0
    "has_dns_records",          # 1/0/-1 (unknown → 0)
    "ns_count",                 # number of NS records, 0 when unknown
    "mx_count",                 # number of MX records, 0 when unknown
    "resolved_ip_count",        # number of resolved IPs, 0 when unknown
    "url_is_https",             # 1/0
]


def evidence_to_scorer_row(
    ml_probability: float,
    domain_evidence: "DomainEvidence",
    url_is_https: bool,
) -> list[float]:
    """Convert a ``DomainEvidence`` into the numeric feature vector for the scorer."""
    ev = domain_evidence
    age = ev.domain_age_days or 0
    ssl_days = ev.days_until_expiration or 0
    tls_dict = ev.to_legacy_ssl_dict()
    return [
        float(ml_probability),
        float(min(age, 10_000)),
        float(np.log1p(age)),
        float(ev.is_suspicious_new_domain),
        float(ev.has_valid_ssl),
        float(tls_dict.get("is_self_signed", False)),
        float(min(ssl_days, 1_000)),
        float(ev.is_shared_hosting),
        float(ev.lookalike.is_lookalike),
        float(ev.urlhaus_listed),
        float(1 if ev.has_dns_records is True else (0 if ev.has_dns_records is False else 0)),
        float(len(ev.ns_records)),
        float(len(ev.mx_records)),
        float(len(ev.resolved_ips)),
        float(url_is_https),
    ]


class DomainScorer:
    """Logistic regression wrapper that acts like ``compose_risk()``."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._loaded: bool = False

    def load(self) -> bool:
        """Try to load the scorer artifact. Returns True on success."""
        if self._loaded:
            return self._model is not None
        self._loaded = True
        if not DOMAIN_SCORER_PATH.exists():
            log.debug("Domain scorer artifact not found at %s; using hand-picked weights.", DOMAIN_SCORER_PATH)
            return False
        try:
            import joblib
            self._model = joblib.load(DOMAIN_SCORER_PATH)
            log.info("Domain scorer loaded from %s", DOMAIN_SCORER_PATH)
            return True
        except Exception as exc:
            log.warning("Could not load domain scorer: %s", exc)
            return False

    def is_available(self) -> bool:
        return self._model is not None

    def score(
        self,
        ml_probability: float,
        domain_evidence: "DomainEvidence",
        url_is_https: bool,
    ) -> float | None:
        """Return calibrated composite probability, or None when unavailable."""
        model = self._model
        if model is None:
            return None
        try:
            row = evidence_to_scorer_row(ml_probability, domain_evidence, url_is_https)
            X = np.array([row], dtype=float)
            prob = float(model.predict_proba(X)[0, 1])
            return max(0.0, min(1.0, prob))
        except Exception as exc:
            log.warning("DomainScorer.score() failed: %s", exc)
            return None


# Process-wide singleton — loaded lazily.
_SCORER = DomainScorer()


def get_scorer() -> DomainScorer:
    """Return the process-wide ``DomainScorer`` (loads on first call)."""
    _SCORER.load()
    return _SCORER
