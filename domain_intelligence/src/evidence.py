"""Typed evidence schema for all domain-intelligence sources.

Every online lookup (RDAP, WHOIS, DNS, TLS, URLhaus, VirusTotal …) is wrapped
in a ``SourceEvidence`` envelope that records *when* the lookup ran, *whether*
it succeeded, and *whether* the data is fresh enough to trust.

``DomainEvidence`` is the top-level object returned by
``src.domain_intelligence.collect_intelligence()``.  It exposes the union of
all sources and convenience properties that extract the values used by the
composite risk scorer so that the rest of the code-base can stay readable.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field


SourceStatus = Literal["available", "unavailable", "rate_limited", "stale"]


class SourceEvidence(BaseModel):
    """One source's enrichment result with freshness metadata."""

    source: str = Field(..., description="Logical name: 'rdap', 'whois', 'dns', 'tls', 'openphish', 'urlhaus', …")
    status: SourceStatus = Field(..., description="Outcome of the lookup attempt.")
    checked_at: datetime = Field(..., description="Wall-clock time (UTC) when the lookup ran.")
    data: dict[str, Any] = Field(default_factory=dict, description="Source-specific payload.")
    stale_after_hours: float = Field(
        default=24.0,
        description="The data is considered stale when the cache entry is older than this many hours.",
    )

    @computed_field  # type: ignore[misc]
    @property
    def is_stale(self) -> bool:
        """True when the data is older than ``stale_after_hours``."""
        if self.status == "stale":
            return True
        age_hours = (datetime.now(timezone.utc) - self.checked_at).total_seconds() / 3600
        return age_hours > self.stale_after_hours

    def get(self, key: str, default: Any = None) -> Any:
        """Convenience accessor so callers can treat evidence like a dict."""
        return self.data.get(key, default)

    @classmethod
    def unavailable(cls, source: str, reason: str, stale_after_hours: float = 24.0) -> "SourceEvidence":
        return cls(
            source=source,
            status="unavailable",
            checked_at=datetime.now(timezone.utc),
            data={"error": reason},
            stale_after_hours=stale_after_hours,
        )

    @classmethod
    def ok(cls, source: str, data: dict[str, Any], stale_after_hours: float = 24.0) -> "SourceEvidence":
        return cls(
            source=source,
            status="available",
            checked_at=datetime.now(timezone.utc),
            data=data,
            stale_after_hours=stale_after_hours,
        )

    @classmethod
    def stale_feed(cls, source: str, data: dict[str, Any], stale_after_hours: float = 6.0) -> "SourceEvidence":
        return cls(
            source=source,
            status="stale",
            checked_at=datetime.now(timezone.utc),
            data=data,
            stale_after_hours=stale_after_hours,
        )


class LookalikeScan(BaseModel):
    """Result of the offline lookalike / typosquat detector."""

    is_lookalike: bool = False
    matched_brand: str | None = None
    technique: Literal["homoglyph", "edit_distance", "punycode", "bit_flip", "typo_pattern", "compound_brand"] | None = None
    decoded_unicode: str | None = None
    edit_distance: int | None = None


class DomainEvidence(BaseModel):
    """Aggregated intelligence for one hostname.

    Each lookup is stored as a ``SourceEvidence`` entry in ``sources``.
    Convenience properties extract the values most commonly needed by the
    risk scorer and response serialiser, so callers need not iterate
    ``sources`` manually.
    """

    hostname: str
    registered_domain: str
    is_shared_hosting: bool = False
    sources: list[SourceEvidence] = Field(default_factory=list)
    lookalike: LookalikeScan = Field(default_factory=LookalikeScan)
    lookup_notes: list[str] = Field(default_factory=list)

    # ---------------------------------------------------------------- helpers
    def _source(self, name: str) -> SourceEvidence | None:
        for s in self.sources:
            if s.source == name:
                return s
        return None

    def _first_available(self, *names: str) -> SourceEvidence | None:
        for name in names:
            s = self._source(name)
            if s and s.status == "available":
                return s
        return None

    # --------------------------------------------------------- domain / WHOIS
    @computed_field  # type: ignore[misc]
    @property
    def domain_age_days(self) -> int | None:
        ev = self._first_available("rdap", "whois")
        return ev.data.get("domain_age_days") if ev else None

    @computed_field  # type: ignore[misc]
    @property
    def creation_date(self) -> str | None:
        ev = self._first_available("rdap", "whois")
        return ev.data.get("creation_date") if ev else None

    @computed_field  # type: ignore[misc]
    @property
    def registrar(self) -> str | None:
        ev = self._first_available("rdap", "whois")
        return ev.data.get("registrar") if ev else None

    @computed_field  # type: ignore[misc]
    @property
    def whois_lookup_success(self) -> bool:
        return self.domain_age_days is not None

    @computed_field  # type: ignore[misc]
    @property
    def is_suspicious_new_domain(self) -> bool:
        age = self.domain_age_days
        if age is None:
            return False
        from .config import NEW_DOMAIN_DAYS
        return age < NEW_DOMAIN_DAYS

    # ------------------------------------------------------------------ DNS
    @computed_field  # type: ignore[misc]
    @property
    def has_dns_records(self) -> bool | None:
        ev = self._source("dns")
        if ev is None:
            return None
        return ev.data.get("has_dns_records")

    @computed_field  # type: ignore[misc]
    @property
    def ns_records(self) -> list[str]:
        ev = self._source("dns")
        return (ev.data.get("ns_records") or []) if ev else []

    @computed_field  # type: ignore[misc]
    @property
    def mx_records(self) -> list[str]:
        ev = self._source("dns")
        return (ev.data.get("mx_records") or []) if ev else []

    @computed_field  # type: ignore[misc]
    @property
    def resolved_ips(self) -> list[str]:
        ev = self._source("dns")
        return (ev.data.get("resolved_ips") or []) if ev else []

    # ------------------------------------------------------------------ TLS
    @computed_field  # type: ignore[misc]
    @property
    def tls_check_status(self) -> str:
        ev = self._source("tls")
        return ev.data.get("check_status", "skipped") if ev else "skipped"

    @computed_field  # type: ignore[misc]
    @property
    def has_valid_ssl(self) -> bool:
        ev = self._source("tls")
        return bool(ev.data.get("has_valid_ssl")) if ev else False

    @computed_field  # type: ignore[misc]
    @property
    def cert_issued_at(self) -> str | None:
        ev = self._source("tls")
        return ev.data.get("cert_issued_at") if ev else None

    @computed_field  # type: ignore[misc]
    @property
    def san_hostnames(self) -> list[str]:
        ev = self._source("tls")
        return (ev.data.get("san_hostnames") or []) if ev else []

    @computed_field  # type: ignore[misc]
    @property
    def tls_issuer(self) -> str | None:
        ev = self._source("tls")
        return ev.data.get("issuer") if ev else None

    @computed_field  # type: ignore[misc]
    @property
    def days_until_expiration(self) -> int | None:
        ev = self._source("tls")
        return ev.data.get("days_until_expiration") if ev else None

    # ------------------------------------------------------- Reputation feeds
    @computed_field  # type: ignore[misc]
    @property
    def is_known_phishing(self) -> bool:
        ev = self._source("openphish")
        return bool(ev.data.get("is_known_phishing")) if ev else False

    @computed_field  # type: ignore[misc]
    @property
    def urlhaus_listed(self) -> bool:
        ev = self._source("urlhaus")
        return bool(ev.data.get("is_listed")) if ev else False

    @computed_field  # type: ignore[misc]
    @property
    def urlhaus_status_stale(self) -> bool:
        ev = self._source("urlhaus")
        return ev.is_stale if ev else False

    # ----------------------------------------------- backward-compat helpers
    def to_legacy_domain_dict(self) -> dict[str, Any]:
        """Flatten to the dict format expected by the old risk_scoring API."""
        return {
            "domain": self.registered_domain,
            "domain_age_days": self.domain_age_days,
            "creation_date": self.creation_date,
            "registrar": self.registrar,
            "has_dns_records": self.has_dns_records,
            "whois_lookup_success": self.whois_lookup_success,
            "is_suspicious_new_domain": self.is_suspicious_new_domain,
            "is_shared_hosting": self.is_shared_hosting,
            "lookup_notes": self.lookup_notes,
            # new fields
            "ns_records": self.ns_records,
            "mx_records": self.mx_records,
            "resolved_ips": self.resolved_ips,
            "lookalike": self.lookalike.model_dump(),
            "urlhaus_listed": self.urlhaus_listed,
            "urlhaus_status_stale": self.urlhaus_status_stale,
            "cert_issued_at": self.cert_issued_at,
            "san_hostnames": self.san_hostnames,
        }

    def to_legacy_ssl_dict(self) -> dict[str, Any]:
        """Flatten the TLS evidence to the dict format used by the old API."""
        ev = self._source("tls")
        if ev is None:
            return {
                "check_status": "skipped",
                "has_valid_ssl": False,
                "issuer": None,
                "days_until_expiration": None,
                "is_self_signed": False,
                "error": "No TLS evidence collected",
                "cert_issued_at": None,
                "san_hostnames": [],
            }
        return {
            "check_status": ev.data.get("check_status", "skipped"),
            "has_valid_ssl": ev.data.get("has_valid_ssl", False),
            "issuer": ev.data.get("issuer"),
            "days_until_expiration": ev.data.get("days_until_expiration"),
            "is_self_signed": ev.data.get("is_self_signed", False),
            "error": ev.data.get("error"),
            "cert_issued_at": ev.data.get("cert_issued_at"),
            "san_hostnames": ev.data.get("san_hostnames") or [],
        }
