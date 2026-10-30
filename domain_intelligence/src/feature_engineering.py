"""Deterministic, offline URL-only feature engineering.

No feature in this module resolves, visits, redirects to, or otherwise contacts
a URL.  The same functions are used for training and API inference to prevent
training-serving skew.
"""

from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from typing import Iterable
from urllib.parse import urlsplit

import pandas as pd


SUSPICIOUS_KEYWORDS: tuple[str, ...] = (
    "login",
    "verify",
    "account",
    "banking",
    "secure",
    "update",
    "signin",
    "wp-content",
    "admin",
)

# A deliberately small, documented heuristic list. It is not a reputation
# feed and is never used as a claim that a TLD itself is malicious.
SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {"zip", "mov", "country", "stream", "gq", "tk", "ml", "ga", "cf", "work", "click", "top"}
)

FEATURE_NAMES: tuple[str, ...] = (
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "dot_count",
    "hyphen_count",
    "underscore_count",
    "digit_count",
    "special_char_count",
    "subdomain_count",
    "query_param_count",
    "at_count",
    "equals_count",
    "ampersand_count",
    "percent_count",
    "colon_count",
    "suspicious_keyword_count",
    "digit_ratio",
    "special_char_ratio",
    "url_entropy",
    "uses_https",
    "ip_in_url",
    "suspicious_tld",
    "domain_path_ratio",
    "has_punycode",
    "has_port",
)

_SPECIAL_CHARACTERS = set("!#$%&'()*+,/:;<=>?@[\\]^_`{|}~-")
_MULTIPART_SUFFIXES = frozenset({"co.uk", "org.uk", "ac.uk", "com.au", "net.au", "co.jp", "co.in", "com.br"})


class InvalidURLException(ValueError):
    """Raised when a value cannot safely be interpreted as a web URL."""


def normalise_url(url: str) -> str:
    """Return a canonical parseable URL without changing its path semantics.

    A scheme is added only for parsing. Hostname case is normalized by the URL
    parser when domain grouping is performed; the original path/query text is
    retained for lexical characteristics.
    """
    if not isinstance(url, str):
        raise InvalidURLException("URL must be a string")
    cleaned = url.strip()
    if not cleaned:
        raise InvalidURLException("URL must not be empty")
    if any(character.isspace() for character in cleaned):
        raise InvalidURLException("URL must not contain whitespace")
    candidate = cleaned if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", cleaned) else f"https://{cleaned}"
    parsed = urlsplit(candidate)
    if parsed.scheme not in {"http", "https"}:
        raise InvalidURLException("URL scheme must be http or https")
    if not parsed.hostname:
        raise InvalidURLException("URL must include a hostname")
    try:
        # ``urlsplit`` validates ports lazily; force that validation here so
        # malformed ports produce a request-validation error rather than a
        # later feature-extraction/server error.
        _ = parsed.port
    except ValueError as error:
        raise InvalidURLException("URL port is invalid") from error
    if candidate.endswith("/") and parsed.path == "/" and not parsed.query:
        candidate = candidate[:-1]
    return candidate


def parse_url(url: str):
    """Parse a validated URL once for extraction and cleaning."""
    normalized = normalise_url(url)
    return normalized, urlsplit(normalized)


def get_domain(url: str) -> str:
    """Return the lower-cased hostname for leakage-aware grouping."""
    _, parsed = parse_url(url)
    hostname = parsed.hostname
    if hostname is None:
        raise InvalidURLException("URL must include a hostname")
    return hostname.lower().rstrip(".")


def canonical_url_key(url: str) -> str:
    """Canonicalize scheme/host case and discard fragments for duplicate checks."""
    normalized, parsed = parse_url(url)
    hostname = parsed.hostname
    assert hostname is not None
    port = f":{parsed.port}" if parsed.port else ""
    path = parsed.path or "/"
    return f"{parsed.scheme.lower()}://{hostname.lower().rstrip('.')}{port}{path}?{parsed.query}".rstrip("?")


def _is_ip_address(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname.strip("[]"))
        return True
    except ValueError:
        return False


def _registered_suffix_parts(hostname: str) -> int:
    labels = hostname.split(".")
    last_two = ".".join(labels[-2:]) if len(labels) >= 2 else ""
    return 2 if last_two in _MULTIPART_SUFFIXES else 1


def calculate_entropy(text: str) -> float:
    """Shannon entropy in bits per character; zero for empty text."""
    if not text:
        return 0.0
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in Counter(text).values())


def extract_features(url: str) -> dict[str, float | int]:
    """Extract the documented feature vector for one valid URL.

    The function is deterministic, has no network side effects, and guarantees
    exactly the order and names in ``FEATURE_NAMES``.
    """
    normalized, parsed = parse_url(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")
    path = parsed.path or ""
    query = parsed.query or ""
    host_labels = [label for label in hostname.split(".") if label]
    suffix_parts = _registered_suffix_parts(hostname)
    # A hostname normally contains a registrable label plus its public suffix.
    # The small multipart list avoids treating ``www.example.co.uk`` as having
    # two subdomains. IP literals do not have DNS subdomains.
    subdomain_count = 0 if _is_ip_address(hostname) else max(0, len(host_labels) - suffix_parts - 1)
    tld = host_labels[-1] if host_labels else ""
    special_count = sum(character in _SPECIAL_CHARACTERS for character in normalized)
    digit_count = sum(character.isdigit() for character in normalized)
    denominator = max(1, len(normalized))
    keyword_count = sum(normalized.lower().count(keyword) for keyword in SUSPICIOUS_KEYWORDS)
    path_denominator = max(1, len(path))

    values: dict[str, float | int] = {
        "url_length": len(normalized),
        "domain_length": len(hostname),
        "path_length": len(path),
        "query_length": len(query),
        "dot_count": normalized.count("."),
        "hyphen_count": normalized.count("-"),
        "underscore_count": normalized.count("_"),
        "digit_count": digit_count,
        "special_char_count": special_count,
        "subdomain_count": subdomain_count,
        "query_param_count": len([part for part in query.split("&") if part]) if query else 0,
        "at_count": normalized.count("@"),
        "equals_count": normalized.count("="),
        "ampersand_count": normalized.count("&"),
        "percent_count": normalized.count("%"),
        "colon_count": normalized.count(":"),
        "suspicious_keyword_count": keyword_count,
        "digit_ratio": round(digit_count / denominator, 6),
        "special_char_ratio": round(special_count / denominator, 6),
        "url_entropy": round(calculate_entropy(normalized), 6),
        "uses_https": int(parsed.scheme.lower() == "https"),
        "ip_in_url": int(_is_ip_address(hostname)),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "domain_path_ratio": round(len(hostname) / path_denominator, 6),
        "has_punycode": int("xn--" in hostname),
        "has_port": int(parsed.port is not None),
    }
    return {name: values[name] for name in FEATURE_NAMES}


def extract_feature_frame(urls: Iterable[str]) -> pd.DataFrame:
    """Extract an ordered, numeric feature DataFrame for valid URLs."""
    frame = pd.DataFrame((extract_features(url) for url in urls), columns=FEATURE_NAMES)
    if frame.empty:
        return pd.DataFrame(columns=FEATURE_NAMES, dtype=float)
    return frame.astype(float)


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "basic_lexical": (
        "url_length", "domain_length", "path_length", "query_length", "dot_count",
        "hyphen_count", "underscore_count", "digit_count", "special_char_count",
        "subdomain_count", "query_param_count", "at_count", "equals_count",
        "ampersand_count", "percent_count", "colon_count", "digit_ratio",
        "special_char_ratio",
    ),
    "structural_security": (
        "uses_https", "ip_in_url", "suspicious_tld", "domain_path_ratio",
        "has_punycode", "has_port",
    ),
    "entropy": ("url_entropy",),
    "suspicious_keywords": ("suspicious_keyword_count",),
}


def ablation_feature_sets() -> dict[str, list[str]]:
    """Return cumulative, de-duplicated feature sets for the ablation study."""
    basic = list(FEATURE_GROUPS["basic_lexical"])
    structural = basic + list(FEATURE_GROUPS["structural_security"])
    entropy = structural + list(FEATURE_GROUPS["entropy"])
    keywords = structural + list(FEATURE_GROUPS["suspicious_keywords"])
    return {
        "group_1_basic_lexical": basic,
        "group_2_lexical_structural": structural,
        "group_3_lexical_structural_entropy": entropy,
        "group_4_lexical_structural_keywords": keywords,
        "group_5_all_selected": list(FEATURE_NAMES),
    }
