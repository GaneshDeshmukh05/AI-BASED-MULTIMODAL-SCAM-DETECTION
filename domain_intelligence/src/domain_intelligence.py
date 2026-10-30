"""Optional online domain intelligence: DNS, RDAP/WHOIS registration age, TLS certificate.

Unlike ``src.feature_engineering`` this module DOES touch the network, so it is
kept separate and is never used to build model features. Its output is combined
with the model probability by ``src.risk_scoring``.

Safety properties
-----------------
* Only public hosts are contacted. Loopback/private/link-local/reserved
  addresses (including names that *resolve* to them) are skipped, so the API
  cannot be used to probe the server's own network.
* The TLS probe always uses port 443 (never a caller-chosen port), reads only
  the certificate and sends no HTTP request.
* Every lookup is bounded by a wall-clock budget and failures degrade to
  "unknown" rather than to a risk signal.
* Results are cached briefly because browsers re-check the same host often.
* tldextract (Mozilla Public Suffix List) replaces the old hand-rolled suffix
  table, giving correct eTLD+1 for all current and future TLDs.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import socket
import ssl
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any, Callable
from urllib.parse import urlsplit

import tldextract

from .config import DOMAIN_INTEL_TIMEOUT, NEW_DOMAIN_DAYS
from .evidence import DomainEvidence, LookalikeScan, SourceEvidence
from .feature_engineering import InvalidURLException, normalise_url
from .lookalike import scan_domain

logger = logging.getLogger("phishing_url_detector.intel")

# Hosts whose *sub-domains or paths* are controlled by arbitrary tenants. A
# domain being old says nothing about a page served from these, so they never
# receive the "established domain" risk reduction. Heuristic list, not a feed.
SHARED_HOSTING_SUFFIXES = frozenset(
    {
        "github.io", "githubusercontent.com", "gitlab.io", "pages.dev", "workers.dev",
        "netlify.app", "vercel.app", "herokuapp.com", "onrender.com", "glitch.me", "repl.co",
        "web.app", "firebaseapp.com", "appspot.com", "storage.googleapis.com",
        "sites.google.com", "docs.google.com", "drive.google.com", "forms.gle",
        "blogspot.com", "wordpress.com", "weebly.com", "wixsite.com", "myshopify.com",
        "carrd.co", "notion.site", "typeform.com", "square.site",
        "azurewebsites.net", "blob.core.windows.net", "cloudfront.net", "amazonaws.com",
        "forms.office.com", "sharepoint.com", "onedrive.live.com", "dropbox.com",
        "ngrok.io", "ngrok-free.app", "000webhostapp.com", "infinityfreeapp.com",
    }
)

_CACHE_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str], tuple[float, Any]] = {}
_CACHE_MAX = 1024
_TTL_OK = {"dns": 300.0, "rdap": 6 * 3600.0, "whois": 6 * 3600.0, "ssl": 900.0}
_TTL_FAILED = 60.0

_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="domain-intel")

# Well-known RDAP base URLs per TLD (bootstrap cache so we work without IANA lookup)
_RDAP_BOOTSTRAP: dict[str, str] = {
    "com": "https://rdap.verisign.com/com/v1/",
    "net": "https://rdap.verisign.com/net/v1/",
    "org": "https://rdap.publicinterestregistry.org/rdap/",
    "info": "https://rdap.afilias.net/rdap/info/",
    "biz": "https://rdap.nic.biz/",
    "io": "https://rdap.nic.io/",
    "co": "https://rdap.nic.co/",
    "us": "https://rdap.nic.us/",
    "uk": "https://rdap.nominet.uk/uk/",
    "de": "https://rdap.denic.de/",
    "fr": "https://rdap.nic.fr/",
    "eu": "https://rdap.eu/",
    "nl": "https://rdap.sidn.nl/rdap/",
    "au": "https://rdap.auda.org.au/",
    "ca": "https://rdap.cira.ca/rdap/",
    "jp": "https://rdap.jprs.jp/",
    "cn": "https://rdap.cnnic.cn/rdap/",
    "br": "https://rdap.registro.br/",
    "in": "https://rdap.registry.in/rdap/",
}


# ---------------------------------------------------------------- helpers ----
def _cache_get(kind: str, key: str) -> Any | None:
    with _CACHE_LOCK:
        entry = _CACHE.get((kind, key))
        if entry and entry[0] > time.monotonic():
            return entry[1]
        _CACHE.pop((kind, key), None)
    return None


def _cache_put(kind: str, key: str, value: Any, ok: bool) -> None:
    with _CACHE_LOCK:
        if len(_CACHE) >= _CACHE_MAX:
            _CACHE.clear()
        _CACHE[(kind, key)] = (time.monotonic() + (_TTL_OK[kind] if ok else _TTL_FAILED), value)


def clear_cache() -> None:
    """Drop all cached lookups (used by tests)."""
    with _CACHE_LOCK:
        _CACHE.clear()


def _is_ip(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return True
    except ValueError:
        return False


def registered_domain(hostname: str) -> str:
    """Return the registrable domain (eTLD+1) using the Mozilla Public Suffix List via tldextract."""
    if not hostname:
        return hostname
    extracted = tldextract.extract(hostname.lower().rstrip("."))
    # tldextract returns (subdomain, domain, suffix)
    # registered domain = domain + suffix
    if extracted.domain and extracted.suffix:
        return f"{extracted.domain}.{extracted.suffix}"
    if extracted.domain:
        return extracted.domain
    # Fallback for edge cases (single label, IP literals)
    return hostname.lower().rstrip(".")


def is_shared_hosting(hostname: str) -> bool:
    """True when the host is, or is a sub-domain of, a multi-tenant hosting domain."""
    host = hostname.lower().rstrip(".")
    return any(host == suffix or host.endswith("." + suffix) for suffix in SHARED_HOSTING_SUFFIXES)


def _public_addresses(hostname: str, port: int = 443) -> list[str]:
    """Resolve ``hostname`` and return its addresses, or [] if ANY is non-public."""
    if _is_ip(hostname):
        addresses = [hostname.strip("[]")]
    else:
        infos = socket.getaddrinfo(hostname, port, type=socket.SOCK_STREAM)
        addresses = list(dict.fromkeys(str(info[4][0]) for info in infos))
    if not addresses:
        return []
    for address in addresses:
        if not ipaddress.ip_address(address.split("%")[0]).is_global:
            return []
    return addresses


def _split_url(url: str) -> tuple[str, str]:
    """Return (scheme, lower-cased hostname) for a URL or schemeless host."""
    normalized = normalise_url(url)
    parsed = urlsplit(normalized)
    return parsed.scheme.lower(), (parsed.hostname or "").lower().rstrip(".")


# ------------------------------------------------------------------- DNS ----
def lookup_dns(hostname: str, timeout: float) -> dict[str, Any]:
    """Whether ``hostname`` has A/AAAA records plus NS, MX and resolved IPs."""
    cached = _cache_get("dns", hostname)
    if cached is not None:
        return cached
    result: dict[str, Any] = {
        "has_dns_records": None,
        "ns_records": [],
        "mx_records": [],
        "resolved_ips": [],
        "error": None,
    }
    try:
        import dns.resolver

        resolver = dns.resolver.Resolver()
        resolver.lifetime = timeout
        resolver.timeout = min(timeout, 2.0)
        found = False

        # A / AAAA records (and collect IPs while we're at it)
        ips: list[str] = []
        for record_type in ("A", "AAAA"):
            try:
                answers = resolver.resolve(hostname, record_type)
                if len(answers) > 0:
                    found = True
                    ips.extend(str(rdata) for rdata in answers)
            except dns.resolver.NoAnswer:
                continue
            except dns.resolver.NXDOMAIN:
                break
        result["has_dns_records"] = found
        result["resolved_ips"] = list(dict.fromkeys(ips))

        # NS records
        try:
            ns_answers = resolver.resolve(hostname, "NS")
            result["ns_records"] = sorted(str(rdata).rstrip(".") for rdata in ns_answers)
        except Exception:
            pass

        # MX records
        try:
            mx_answers = resolver.resolve(hostname, "MX")
            result["mx_records"] = sorted(str(rdata.exchange).rstrip(".") for rdata in mx_answers)
        except Exception:
            pass

    except ImportError:
        result["error"] = "dnspython is not installed"
    except Exception as error:
        result["error"] = f"DNS lookup failed: {type(error).__name__}"
    _cache_put("dns", hostname, result, ok=result["error"] is None)
    return result


# ----------------------------------------------------------------- WHOIS ----
def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        text = value.strip().split("T")[0].split(" ")[0]
        for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%Y/%m/%d", "%d/%m/%Y"):
            try:
                moment = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        else:
            return None
    else:
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)


def lookup_whois(domain: str, timeout: float) -> dict[str, Any]:
    """Registration date/registrar for a registrable domain."""
    cached = _cache_get("whois", domain)
    if cached is not None:
        return cached
    result: dict[str, Any] = {"creation_date": None, "domain_age_days": None, "registrar": None, "error": None}
    try:
        import whois

        try:
            record: Any = whois.whois(domain, timeout=int(timeout))
        except TypeError:
            record = whois.whois(domain)
        raw = getattr(record, "creation_date", None)
        candidates = raw if isinstance(raw, (list, tuple)) else [raw]
        moments = [m for m in (_coerce_datetime(item) for item in candidates) if m is not None]
        if moments:
            created = min(moments)
            age = (datetime.now(timezone.utc) - created).days
            result["creation_date"] = created.strftime("%Y-%m-%d")
            result["domain_age_days"] = max(0, age)
        registrar = getattr(record, "registrar", None)
        if registrar:
            result["registrar"] = registrar if isinstance(registrar, str) else str(registrar[0])
        if result["domain_age_days"] is None:
            result["error"] = "WHOIS record has no usable creation date"
    except ImportError:
        result["error"] = "python-whois is not installed"
    except Exception as error:
        result["error"] = f"WHOIS lookup failed: {type(error).__name__}"
    _cache_put("whois", domain, result, ok=result["domain_age_days"] is not None)
    return result


# ------------------------------------------------------------------ RDAP ----
def _rdap_base_url(tld: str) -> str | None:
    """Return a known RDAP base URL for ``tld``, or None."""
    return _RDAP_BOOTSTRAP.get(tld.lower())


def _parse_rdap_response(data: dict[str, Any]) -> dict[str, Any]:
    """Extract creation_date / domain_age_days / registrar from an RDAP response."""
    result: dict[str, Any] = {"creation_date": None, "domain_age_days": None, "registrar": None, "error": None}
    for event in data.get("events", []):
        if event.get("eventAction") == "registration":
            raw_date = event.get("eventDate", "")
            moment = _coerce_datetime(raw_date)
            if moment:
                age = (datetime.now(timezone.utc) - moment).days
                result["creation_date"] = moment.strftime("%Y-%m-%d")
                result["domain_age_days"] = max(0, age)
            break
    # Registrar is in entities with role "registrar"
    for entity in data.get("entities", []):
        roles = entity.get("roles", [])
        if "registrar" in roles:
            vcard = entity.get("vcardArray", [])
            if isinstance(vcard, list) and len(vcard) > 1:
                for field in vcard[1]:
                    if field[0] == "fn":
                        result["registrar"] = field[3]
                        break
            if result["registrar"] is None:
                result["registrar"] = entity.get("handle")
            break
    return result


def lookup_rdap(domain: str, timeout: float) -> dict[str, Any]:
    """Query RDAP for registration info; falls back to WHOIS on any failure."""
    cached = _cache_get("rdap", domain)
    if cached is not None:
        return cached

    tld = domain.split(".")[-1] if "." in domain else domain
    base = _rdap_base_url(tld)
    result: dict[str, Any] = {"creation_date": None, "domain_age_days": None, "registrar": None, "error": None, "source": "none"}

    if base:
        try:
            import requests as _requests  # noqa: PLC0415

            url = f"{base.rstrip('/')}/domain/{domain}"
            resp = _requests.get(
                url,
                timeout=min(timeout, 4.0),
                headers={"Accept": "application/rdap+json, application/json"},
            )
            if resp.status_code == 200:
                data = resp.json()
                parsed = _parse_rdap_response(data)
                if parsed["domain_age_days"] is not None:
                    parsed["source"] = "rdap"
                    _cache_put("rdap", domain, parsed, ok=True)
                    return parsed
            elif resp.status_code == 404:
                result["error"] = "RDAP: domain not found (404)"
            else:
                result["error"] = f"RDAP: HTTP {resp.status_code}"
        except Exception as exc:
            result["error"] = f"RDAP lookup failed: {type(exc).__name__}"

    # Fall back to WHOIS
    whois_result = lookup_whois(domain, timeout)
    if whois_result.get("domain_age_days") is not None:
        whois_result["source"] = "whois"
        _cache_put("rdap", domain, whois_result, ok=True)
        return whois_result

    result.setdefault("error", whois_result.get("error") or "RDAP and WHOIS both failed")
    result["source"] = "none"
    _cache_put("rdap", domain, result, ok=False)
    return result


# ------------------------------------------------------------------- TLS ----
def _flatten_name(name: Any) -> dict[str, str]:
    flat: dict[str, str] = {}
    for rdn in name or ():
        for key, value in rdn:
            flat.setdefault(key, value)
    return flat


def _extract_san(cert: dict[str, Any]) -> list[str]:
    """Return the list of dNSName entries from the SAN extension."""
    san: list[str] = []
    for entry in cert.get("subjectAltName", ()):
        if entry[0].lower() == "dns":
            san.append(entry[1])
    return san


def inspect_tls(hostname: str, timeout: float) -> dict[str, Any]:
    """Verify the host's certificate chain and report issuer/expiry/SAN/issue-date.

    ``check_status`` is one of:
      ``valid``       chain and hostname verified.
      ``invalid``     the server presented a certificate that fails verification
                      (expired, self-signed, wrong host, untrusted CA).
      ``unreachable`` could not complete a TLS handshake (timeout, refused…).
      ``skipped``     deliberately not probed (non-public address).
    """
    cached = _cache_get("ssl", hostname)
    if cached is not None:
        return cached
    info: dict[str, Any] = {
        "check_status": "unreachable",
        "has_valid_ssl": False,
        "issuer": None,
        "cert_issued_at": None,
        "days_until_expiration": None,
        "san_hostnames": [],
        "is_self_signed": False,
        "error": None,
    }
    try:
        addresses = _public_addresses(hostname)
        if not addresses:
            info.update(check_status="skipped", error="Host resolves to a non-public address; not probed")
            _cache_put("ssl", hostname, info, ok=True)
            return info
        context = ssl.create_default_context()
        with socket.create_connection((addresses[0], 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as tls:
                peer_cert = tls.getpeercert()
        if not peer_cert:
            raise ValueError("TLS peer did not present a certificate")
        cert: Any = peer_cert
        issuer = _flatten_name(cert.get("issuer"))
        subject = _flatten_name(cert.get("subject"))
        expires = ssl.cert_time_to_seconds(cert["notAfter"])
        days_left = int((expires - time.time()) // 86400)
        # cert_issued_at: notBefore field
        issued_at_str: str | None = None
        not_before = cert.get("notBefore")
        if not_before:
            try:
                issued_ts = ssl.cert_time_to_seconds(not_before)
                issued_at_str = datetime.fromtimestamp(issued_ts, timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                pass
        san_hostnames = _extract_san(cert)
        info.update(
            check_status="valid",
            has_valid_ssl=True,
            issuer=issuer.get("organizationName") or issuer.get("commonName") or "Unknown",
            cert_issued_at=issued_at_str,
            days_until_expiration=max(0, days_left),
            san_hostnames=san_hostnames,
            is_self_signed=bool(issuer) and issuer == subject,
        )
    except ssl.SSLCertVerificationError as error:
        info.update(
            check_status="invalid",
            is_self_signed=getattr(error, "verify_code", None) in (18, 19),
            error=f"Certificate verification failed: {getattr(error, 'verify_message', str(error))}",
        )
    except Exception as error:
        info["error"] = f"TLS probe failed: {type(error).__name__}"
    _cache_put("ssl", hostname, info, ok=info["check_status"] in {"valid", "invalid", "skipped"})
    return info


# ----------------------------------------------------------- orchestration ----
def collect_intelligence(url: str, timeout: float = DOMAIN_INTEL_TIMEOUT) -> DomainEvidence:
    """Gather enriched domain intelligence for ``url`` and return a ``DomainEvidence``.

    Never raises for network problems: unknown values are ``None`` and the
    reason is recorded in ``lookup_notes``.

    The returned ``DomainEvidence`` supports ``.to_legacy_domain_dict()`` and
    ``.to_legacy_ssl_dict()`` for backward-compatibility with code that still
    expects the old tuple-of-dicts format.
    """
    try:
        scheme, hostname = _split_url(url)
    except InvalidURLException as error:
        ev = DomainEvidence(
            hostname="",
            registered_domain="",
            lookup_notes=[str(error)],
        )
        ev.sources.append(SourceEvidence.unavailable("tls", str(error)))
        return ev

    domain = registered_domain(hostname)
    notes: list[str] = []
    sources: list[SourceEvidence] = []

    shared = is_shared_hosting(hostname)
    lookalike: LookalikeScan = scan_domain(hostname)

    if _is_ip(hostname) or "." not in hostname:
        reason = "IP-literal host" if _is_ip(hostname) else "single-label/internal hostname"
        notes.append(f"Domain lookups skipped: {reason}")
        sources.append(SourceEvidence.unavailable("tls", f"TLS probe skipped: {reason}"))
        return DomainEvidence(
            hostname=hostname,
            registered_domain=domain,
            is_shared_hosting=shared,
            sources=sources,
            lookalike=lookalike,
            lookup_notes=notes,
        )

    # Schedule parallel lookups
    jobs: dict[str, Callable[[], dict[str, Any]]] = {
        "dns": lambda: lookup_dns(hostname, timeout),
        "rdap": lambda: lookup_rdap(domain, timeout),
    }
    if scheme == "https":
        jobs["tls"] = lambda: inspect_tls(hostname, min(timeout, 3.0))
    futures = {name: _EXECUTOR.submit(job) for name, job in jobs.items()}
    wait(futures.values(), timeout=timeout)

    results: dict[str, dict[str, Any]] = {}
    for name, future in futures.items():
        if future.done() and future.exception() is None:
            results[name] = future.result()
        else:
            notes.append(f"{name} lookup timed out or failed")

    # --- DNS ---
    dns_raw = results.get("dns")
    if dns_raw:
        if dns_raw.get("error"):
            notes.append(dns_raw["error"])
        sources.append(
            SourceEvidence.ok("dns", dns_raw, stale_after_hours=1.0)
            if dns_raw.get("error") is None
            else SourceEvidence.unavailable("dns", dns_raw.get("error", "dns failed"))
        )
    else:
        sources.append(SourceEvidence.unavailable("dns", "dns lookup did not complete"))

    # --- RDAP / WHOIS ---
    rdap_raw = results.get("rdap")
    if rdap_raw:
        src_name = rdap_raw.get("source", "rdap")
        if rdap_raw.get("domain_age_days") is not None:
            sources.append(SourceEvidence.ok(src_name, rdap_raw, stale_after_hours=24.0))
        else:
            if rdap_raw.get("error"):
                notes.append(rdap_raw["error"])
            sources.append(SourceEvidence.unavailable(src_name, rdap_raw.get("error", "lookup failed")))
    else:
        sources.append(SourceEvidence.unavailable("rdap", "rdap lookup did not complete"))

    # --- TLS ---
    if scheme != "https":
        sources.append(SourceEvidence.unavailable("tls", "HTTP scheme used (no TLS to inspect)"))
    else:
        tls_raw = results.get("tls")
        if tls_raw:
            ok_status = tls_raw.get("check_status") in {"valid", "invalid", "skipped"}
            sources.append(
                SourceEvidence.ok("tls", tls_raw, stale_after_hours=4.0)
                if ok_status
                else SourceEvidence.unavailable("tls", tls_raw.get("error", "tls probe failed"))
            )
        else:
            sources.append(SourceEvidence.unavailable("tls", "TLS probe timed out"))

    return DomainEvidence(
        hostname=hostname,
        registered_domain=domain,
        is_shared_hosting=shared,
        sources=sources,
        lookalike=lookalike,
        lookup_notes=notes,
    )
