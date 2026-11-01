"""Offline lookalike and typosquat detector.

Detects suspicious domain names that impersonate well-known brands using:
- Punycode / IDN decoding (reveal hidden Unicode in xn-- labels)
- Homoglyph substitution (confusable Unicode chars mapped to ASCII)
- Levenshtein edit-distance ≤ 2 against a brand list
- Single-bit-flip variants (bit-squatting)
- Common keyboard-adjacency typo patterns (doubled/missing/swapped letters)

This module is **purely offline** — it never opens a network connection.
Add additional brands via the env var ``PHISHING_BRAND_LIST_PATH`` pointing
to a newline-delimited file (one brand per line, comments starting with #
are ignored).
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

from .evidence import LookalikeScan


# ---------------------------------------------------------------------------
# Brand list & Deceptive keywords
# ---------------------------------------------------------------------------
_DECEPTIVE_TERMS: frozenset[str] = frozenset(
    {
        "verify", "verification",
        "secure", "security",
        "login", "signin", "logon", "sign-in", "log-in",
        "account", "banking",
        "update", "confirm", "confirmation",
        "auth", "authenticate", "authentication",
        "portal", "password", "credential",
        "support", "helpdesk", "recovery", "restore",
    }
)
_BUILTIN_BRANDS: frozenset[str] = frozenset(
    {
        # Financial
        "paypal", "chase", "wellsfargo", "bankofamerica", "citibank", "barclays",
        "hsbc", "santander", "usbank", "capitalone", "americanexpress", "amex",
        "visa", "mastercard", "zelle", "venmo", "cashapp",
        # E-commerce / tech
        "amazon", "ebay", "alibaba", "aliexpress", "walmart", "target", "bestbuy",
        "etsy", "shopify", "apple", "google", "microsoft", "facebook", "instagram",
        "twitter", "linkedin", "netflix", "spotify", "dropbox", "github", "gitlab",
        "zoom", "slack", "adobe", "docusign",
        # Email / cloud
        "gmail", "outlook", "yahoo", "icloud", "protonmail", "office365",
        "onedrive", "sharepoint",
        # Logistics / gov
        "fedex", "ups", "dhl", "usps", "irs", "doj", "fbi",
    }
)


@lru_cache(maxsize=1)
def _brand_list() -> frozenset[str]:
    env_path = os.getenv("PHISHING_BRAND_LIST_PATH", "")
    if not env_path:
        return _BUILTIN_BRANDS
    extra: set[str] = set()
    try:
        for line in Path(env_path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip().lower()
            if stripped and not stripped.startswith("#"):
                extra.add(stripped)
    except OSError:
        pass
    return _BUILTIN_BRANDS | frozenset(extra)


# ---------------------------------------------------------------------------
# Homoglyph table
# (Sources: Unicode Confusables data + common phishing substitutions)
# ---------------------------------------------------------------------------
_HOMOGLYPH_MAP: dict[str, str] = {
    # Cyrillic → Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d", "ԝ": "w", "ԛ": "q",
    # Greek
    "α": "a", "β": "b", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ο": "o",
    "ρ": "p", "τ": "t", "υ": "u", "χ": "x", "ω": "w",
    # Common visual substitutions
    "0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "8": "b",
    "|": "l", "!": "i", "@": "a",
    # Full-width Latin
    "ａ": "a", "ｂ": "b", "ｃ": "c", "ｄ": "d", "ｅ": "e", "ｆ": "f",
    "ｇ": "g", "ｈ": "h", "ｉ": "i", "ｊ": "j", "ｋ": "k", "ｌ": "l",
    "ｍ": "m", "ｎ": "n", "ｏ": "o", "ｐ": "p", "ｑ": "q", "ｒ": "r",
    "ｓ": "s", "ｔ": "t", "ｕ": "u", "ｖ": "v", "ｗ": "w", "ｘ": "x",
    "ｙ": "y", "ｚ": "z",
}


def _apply_homoglyphs(text: str) -> str:
    return "".join(_HOMOGLYPH_MAP.get(ch, ch) for ch in text.lower())


# ---------------------------------------------------------------------------
# Levenshtein distance
# ---------------------------------------------------------------------------
def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # Use a rolling two-row DP to keep memory O(min(|a|,|b|)).
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for j, ch_b in enumerate(b, 1):
        curr = [j] + [0] * len(a)
        for i, ch_a in enumerate(a, 1):
            cost = 0 if ch_a == ch_b else 1
            curr[i] = min(curr[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = curr
    return prev[len(a)]


# ---------------------------------------------------------------------------
# Punycode / IDN decoding
# ---------------------------------------------------------------------------
def _decode_punycode(label: str) -> str:
    """Decode a single DNS label; return the ASCII label on failure."""
    if label.lower().startswith("xn--"):
        try:
            return label.encode("ascii").decode("idna")
        except UnicodeError:
            pass
    return label


def _decoded_hostname(hostname: str) -> str:
    return ".".join(_decode_punycode(label) for label in hostname.split("."))


# ---------------------------------------------------------------------------
# Bit-squatting
# ---------------------------------------------------------------------------
def _bit_flip_variants(name: str) -> list[str]:
    """Yield all single-bit-flip ASCII variants that are still printable."""
    variants: list[str] = []
    chars = list(name)
    for i, ch in enumerate(chars):
        code = ord(ch)
        for bit in range(7):  # ASCII is 7-bit
            flipped = code ^ (1 << bit)
            if 32 <= flipped <= 126 and flipped != code:  # printable ASCII
                variant = chars[:i] + [chr(flipped)] + chars[i + 1:]
                variants.append("".join(variant))
    return variants


# ---------------------------------------------------------------------------
# Common typo patterns
# ---------------------------------------------------------------------------
_KEYBOARD_NEIGHBOURS: dict[str, str] = {
    "a": "sqwz", "b": "vghn", "c": "xdfv", "d": "serfcx", "e": "wsdr",
    "f": "drtgvc", "g": "ftyhbv", "h": "gyujnb", "i": "ujko", "j": "huikm",
    "k": "jilom", "l": "kop", "m": "njk", "n": "bhjm", "o": "iklp",
    "p": "lo", "q": "wa", "r": "edft", "s": "awedxz", "t": "rfgy",
    "u": "yhji", "v": "cfgb", "w": "qase", "x": "zsdc", "y": "tghu",
    "z": "asx",
}


def _typo_variants(name: str) -> list[str]:
    """Generate common typo variants: adjacent-key swaps, doubled, missing."""
    variants: list[str] = []
    chars = list(name)
    length = len(chars)
    # Adjacent-key substitution
    for i, ch in enumerate(chars):
        for neighbour in _KEYBOARD_NEIGHBOURS.get(ch, ""):
            variants.append("".join(chars[:i] + [neighbour] + chars[i + 1:]))
    # Missing a character
    for i in range(length):
        variants.append("".join(chars[:i] + chars[i + 1:]))
    # Doubled a character
    for i, ch in enumerate(chars):
        variants.append("".join(chars[:i] + [ch, ch] + chars[i + 1:]))
    # Transposed adjacent characters
    for i in range(length - 1):
        swapped = chars[:]
        swapped[i], swapped[i + 1] = swapped[i + 1], swapped[i]
        variants.append("".join(swapped))
    return variants


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def scan_domain(hostname: str) -> LookalikeScan:
    """Check ``hostname`` for lookalike / typosquat signals.

    Returns a :class:`LookalikeScan` with the first (most specific) match found.
    The order of checks is intentional: punycode and homoglyphs are higher
    confidence than edit-distance, which is higher than bit-flip.
    """
    # Strip trailing dot, lower-case.
    host = hostname.lower().rstrip(".")
    labels = host.split(".")
    candidate = ".".join(labels[:-1]) if len(labels) > 1 else host
    plain = candidate.replace("-", "")

    brands = _brand_list()
    candidate_labels = [label for label in labels[:-1] if label] or [host]
    registrable_label = candidate_labels[-1].replace("-", "")
    exact_brand = registrable_label in brands

    # --- 1. Punycode / IDN decoding ---
    decoded = _decoded_hostname(host)
    decoded_candidate = ".".join(decoded.split(".")[:-1]) if "." in decoded else decoded
    if decoded != host:
        for brand in brands:
            if brand in decoded_candidate:
                return LookalikeScan(
                    is_lookalike=True,
                    matched_brand=brand,
                    technique="punycode",
                    decoded_unicode=decoded,
                    edit_distance=_levenshtein(plain, brand),
                )

   # --- 2. Homoglyph substitution ---
    normalised = _apply_homoglyphs(candidate)
    if normalised != candidate:  # had substitutable chars
        for brand in brands:
            if brand == normalised or brand in normalised:
                return LookalikeScan(
                    is_lookalike=True,
                    matched_brand=brand,
                    technique="homoglyph",
                    decoded_unicode=normalised,
                    edit_distance=_levenshtein(plain, brand),
                )

    # --- 3. Compound brand lookalike / tokenized brand impersonation ---
    # Catches deceptive compound domains combining a known brand with security/action
    # terms (e.g. "paypal.com-verify.online", "paypal-login.com", "google.com.security-check.example").
    # Excludes genuine brand apex domains and legitimate brand subdomains (e.g. "paypal.com", "login.paypal.com").
    try:
        import tldextract
        extracted = tldextract.extract(host)
        apex_label = extracted.domain.lower() if extracted.domain else ""
        tld_suffix = extracted.suffix.lower() if extracted.suffix else ""
    except Exception:
        extracted = None
        apex_label = registrable_label
        tld_suffix = labels[-1] if len(labels) > 1 else ""

    if apex_label not in brands:
        candidate_part = host[:-(len(tld_suffix) + 1)] if tld_suffix and host.endswith("." + tld_suffix) else candidate
        tokens = [t for t in re.split(r"[\.\-_]+", candidate_part) if t]
        for brand in brands:
            if brand in tokens:
                for term in _DECEPTIVE_TERMS:
                    if term in tokens or any(term in tok for tok in tokens if tok != brand):
                        return LookalikeScan(
                            is_lookalike=True,
                            matched_brand=brand,
                            technique="compound_brand",
                            decoded_unicode=None,
                            edit_distance=0,
                        )

    # --- 4. Edit distance ≤ 2 ---
    for brand in brands:
        dist = _levenshtein(plain, brand)
        if not exact_brand and 0 < dist <= 2:
            return LookalikeScan(
                is_lookalike=True,
                matched_brand=brand,
                technique="edit_distance",
                decoded_unicode=None,
                edit_distance=dist,
            )

    # --- 4. Bit-squatting ---
    for variant in _bit_flip_variants(plain):
        if variant != plain and variant in brands:
            return LookalikeScan(
                is_lookalike=True,
                matched_brand=variant,
                technique="bit_flip",
                decoded_unicode=None,
                edit_distance=_levenshtein(plain, variant),
            )

    # --- 5. Keyboard-typo variants ---
    for variant in _typo_variants(plain):
        if variant != plain and variant in brands:
            return LookalikeScan(
                is_lookalike=True,
                matched_brand=variant,
                technique="typo_pattern",
                decoded_unicode=None,
                edit_distance=_levenshtein(plain, variant),
            )

    return LookalikeScan(is_lookalike=False)