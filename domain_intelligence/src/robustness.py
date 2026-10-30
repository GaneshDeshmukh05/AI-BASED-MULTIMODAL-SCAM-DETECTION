"""Offline URL-string perturbations for safe robustness measurement."""

from __future__ import annotations

from urllib.parse import quote, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from .feature_engineering import normalise_url


def _replace(parts, *, hostname: str | None = None, path: str | None = None, query: str | None = None) -> str:
    netloc = hostname if hostname is not None else parts.netloc
    return urlunsplit((parts.scheme, netloc, path if path is not None else parts.path, query if query is not None else parts.query, ""))


def offline_perturbations(url: str) -> list[tuple[str, str]]:
    """Create deterministic URL variants without contacting any network service."""
    normalized = normalise_url(url)
    parts = urlsplit(normalized)
    hostname = parts.hostname or ""
    prefix = f"{parts.username}@" if parts.username else ""
    if parts.port:
        hostname_with_port = f"{hostname}:{parts.port}"
    else:
        hostname_with_port = hostname
    path = parts.path or "/"
    query_join = "&" if parts.query else ""
    variants = [
        ("add_benign_subdomain", _replace(parts, hostname=f"cdn.{hostname_with_port}")),
        ("append_benign_path", _replace(parts, path=path.rstrip("/") + "/assets/index.html")),
        ("append_query_parameter", _replace(parts, query=f"{parts.query}{query_join}utm_source=mail")),
        ("append_hyphenated_segment", _replace(parts, path=path.rstrip("/") + "/account-info")),
        ("percent_encode_path", _replace(parts, path=quote(path, safe="/%"))),
    ]
    lowered_path = path.lower()
    for keyword in ("login", "verify", "account", "secure", "update"):
        if keyword in lowered_path:
            variants.append(("remove_suspicious_keyword", _replace(parts, path=path.replace(keyword, "portal", 1))))
            break
    # Keep only unique, syntactically valid variants. URL strings are never visited.
    unique: list[tuple[str, str]] = []
    seen = {normalized}
    for transformation, modified in variants:
        if modified not in seen:
            normalise_url(modified)
            seen.add(modified)
            unique.append((transformation, modified))
    return unique


def evaluate_robustness(model, feature_extractor, frame: pd.DataFrame, threshold: float, per_class: int = 100, seed: int = 42) -> pd.DataFrame:
    """Measure score/prediction changes on deterministic offline perturbations."""
    selected = pd.concat(
        [group.sample(min(len(group), per_class), random_state=seed) for _, group in frame.groupby("target")],
        ignore_index=True,
    )
    rows: list[dict[str, object]] = []
    for _, row in selected.iterrows():
        original_features = feature_extractor([row["url"]])
        original_probability = float(model.predict_proba(original_features)[0, 1])
        for transformation, modified in offline_perturbations(row["url"]):
            modified_features = feature_extractor([modified])
            modified_probability = float(model.predict_proba(modified_features)[0, 1])
            rows.append(
                {
                    "source_label": int(row["target"]),
                    "domain": row["domain"],
                    "transformation": transformation,
                    "original_url": row["url"],
                    "original_prediction": "phishing" if original_probability >= threshold else "legitimate",
                    "original_probability": original_probability,
                    "modified_url": modified,
                    "modified_prediction": "phishing" if modified_probability >= threshold else "legitimate",
                    "modified_probability": modified_probability,
                    "probability_change": modified_probability - original_probability,
                    "absolute_probability_change": abs(modified_probability - original_probability),
                    "prediction_changed": (original_probability >= threshold) != (modified_probability >= threshold),
                }
            )
    return pd.DataFrame(rows)
