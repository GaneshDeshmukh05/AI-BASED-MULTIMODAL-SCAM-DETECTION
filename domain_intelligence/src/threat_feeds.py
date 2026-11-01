"""Local threat-feed cache for the URL-only phishing detector.

Two feeds are managed here:

OpenPhish (URL-level)
    The community feed of full phishing URLs. Used for exact-URL matching and
    for generating phishing training labels.  Feed URLs are never visited by
    this module.

URLhaus (hostname-level)
    abuse.ch URLhaus API v1. Queried at hostname granularity so we catch
    malware-hosting domains even when the exact path differs.  Not used as
    training labels because its focus is malware distribution, not phishing.

Both sources report their freshness via ``SourceEvidence`` so the API
response can flag stale data.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import requests

from src.config import FEED_MAX_AGE_HOURS, PROJECT_ROOT

log = logging.getLogger(__name__)
DATA_DIR = PROJECT_ROOT / "data"
FEED_DB = DATA_DIR / "threat_feed_urls.json"
URLHAUS_CACHE_DB = DATA_DIR / "urlhaus_cache.json"
FEED_URL = os.getenv("OPENPHISH_FEED_URL", "https://openphish.com/feed.txt")
URLHAUS_API = "https://urlhaus-api.abuse.ch/v1/host/"
USER_AGENT = "URLDomainPhishingDetector/1.0 (defensive research project)"


def normalize_url(url: str) -> str:
    """Canonicalize URL enough for exact feed matching without contacting it."""
    value = (url or "").strip()
    if not value:
        return ""
    if "://" not in value:
        value = "https://" + value
    p = urlsplit(value)
    if p.scheme.lower() not in {"http", "https"} or not p.hostname:
        return ""
    host = p.hostname.lower().rstrip(".")
    try:
        port = p.port
    except ValueError:
        return ""
    if p.username or p.password:
        return ""  # avoid storing credential-bearing URLs
    netloc = host
    if port and not ((p.scheme.lower() == "http" and port == 80) or (p.scheme.lower() == "https" and port == 443)):
        netloc = f"{host}:{port}"
    path = p.path or "/"
    return urlunsplit((p.scheme.lower(), netloc, path, p.query, ""))


def _hostname_from_url(url: str) -> str:
    """Extract the bare hostname from a URL for hostname-level feed matching."""
    value = (url or "").strip()
    if "://" not in value:
        value = "https://" + value
    try:
        p = urlsplit(value)
        return (p.hostname or "").lower().rstrip(".")
    except Exception:
        return ""


def _atomic_json_write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        temp = Path(f.name)
    temp.replace(path)


# ---------------------------------------------------------------- OpenPhish --
def load_feed() -> dict:
    if not FEED_DB.exists():
        return {"updated_at": None, "source": FEED_URL, "urls": {}}
    try:
        data = json.loads(FEED_DB.read_text(encoding="utf-8"))
        if not isinstance(data.get("urls"), dict):
            raise ValueError("bad cache schema")
        return data
    except (OSError, ValueError):
        log.exception("Threat feed cache unreadable")
        return {"updated_at": None, "source": FEED_URL, "urls": {}}


def _feed_is_stale(updated_at: str | None, max_age_hours: float = FEED_MAX_AGE_HOURS) -> bool:
    """Return True when the feed timestamp is older than ``max_age_hours``."""
    if not updated_at:
        return True
    try:
        ts = datetime.fromisoformat(updated_at)
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
        return age_hours > max_age_hours
    except (ValueError, TypeError):
        return True


def update_openphish_feed(timeout: int = 30) -> dict:
    """Fetch feed and atomically replace cache only after validating response."""
    response = requests.get(FEED_URL, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    response.raise_for_status()
    if len(response.content) > 20_000_000:
        raise ValueError("Feed response unexpectedly large; refusing to store it")
    text = response.content.decode("utf-8-sig", errors="strict")
    urls = {}
    for line in text.splitlines():
        normalized = normalize_url(line)
        if normalized:
            urls[normalized] = {"source": "OpenPhish Community Feed", "label": "phishing"}
    if not urls:
        raise ValueError("Feed returned no valid HTTP(S) URLs; existing cache was preserved")
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), "source": FEED_URL, "count": len(urls), "urls": urls}
    _atomic_json_write(FEED_DB, payload)
    return {"status": "success", "records": len(urls), "updated_at": payload["updated_at"], "cache_file": str(FEED_DB)}


def lookup_url(url: str) -> dict:
    key = normalize_url(url)
    cache = load_feed()
    match = cache["urls"].get(key) if key else None
    stale = _feed_is_stale(cache.get("updated_at"))
    status = "match" if match else ("no_match" if cache.get("updated_at") else "unavailable")
    return {
        "status": status,
        "is_known_phishing": bool(match),
        "source": match.get("source") if match else None,
        "feed_updated_at": cache.get("updated_at"),
        "feed_is_stale": stale,
        "message": (
            "Exact URL matched in cached threat feed" if match
            else ("Feed is stale; result may be outdated" if stale and cache.get("updated_at")
                  else ("No exact feed match; this does not mean the URL is safe" if cache.get("updated_at")
                        else "Threat feed has not been downloaded yet"))
        ),
    }


def export_training_rows(output_path: Path | None = None) -> dict:
    """Export feed URLs as labeled rows for later merge/retraining; no auto-deploy."""
    cache = load_feed()
    if not cache.get("urls"):
        raise RuntimeError("No cached feed. Run update_openphish_feed first.")
    output_path = output_path or DATA_DIR / "openphish_training_rows.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["url", "label", "source", "feed_updated_at"])
        writer.writeheader()
        for url, meta in cache["urls"].items():
            writer.writerow({"url": url, "label": "phishing", "source": meta["source"], "feed_updated_at": cache.get("updated_at")})
    return {"status": "success", "records": len(cache["urls"]), "output": str(output_path)}


# ---------------------------------------------------------------- URLhaus ---
def _load_urlhaus_cache() -> dict:
    if not URLHAUS_CACHE_DB.exists():
        return {"entries": {}}
    try:
        data = json.loads(URLHAUS_CACHE_DB.read_text(encoding="utf-8"))
        return data if isinstance(data.get("entries"), dict) else {"entries": {}}
    except (OSError, ValueError):
        return {"entries": {}}


def lookup_urlhaus(url: str, timeout: float = 5.0) -> dict:
    """Query URLhaus at hostname level; uses a per-host file cache.

    Returns a dict with:
      is_listed (bool), query_status, urls_count, cached_at, is_stale.
    """
    hostname = _hostname_from_url(url)
    if not hostname:
        return {"is_listed": False, "query_status": "invalid_host", "urls_count": 0,
                "cached_at": None, "is_stale": True, "error": "Could not extract hostname"}

    cache = _load_urlhaus_cache()
    entry = cache["entries"].get(hostname)
    if entry:
        cached_at = entry.get("cached_at")
        stale = _feed_is_stale(cached_at, max_age_hours=FEED_MAX_AGE_HOURS)
        if not stale:
            return {**entry, "is_stale": False}

    # Live query
    result: dict = {
        "is_listed": False,
        "query_status": "unknown",
        "urls_count": 0,
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "is_stale": False,
        "error": None,
    }
    try:
        resp = requests.post(
            URLHAUS_API,
            data={"host": hostname},
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        query_status = data.get("query_status", "unknown")
        is_listed = query_status == "is_listed"
        urls_count = len(data.get("urls", []))
        result.update(
            is_listed=is_listed,
            query_status=query_status,
            urls_count=urls_count,
        )
        # Persist to cache
        cache["entries"][hostname] = {**result}
        _atomic_json_write(URLHAUS_CACHE_DB, cache)
    except requests.RequestException as exc:
        result["error"] = f"URLhaus request failed: {type(exc).__name__}"
        result["is_stale"] = True
    except Exception as exc:
        result["error"] = f"URLhaus lookup error: {type(exc).__name__}"
        result["is_stale"] = True

    return result
