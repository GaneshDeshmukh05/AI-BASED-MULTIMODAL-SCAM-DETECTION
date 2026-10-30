"""Reproducible cleaning and leakage-aware split utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .config import RANDOM_SEED, TEST_SIZE
from .feature_engineering import InvalidURLException, canonical_url_key, get_domain


@dataclass
class CleaningReport:
    source: str
    input_rows: int
    removed_missing: int = 0
    removed_invalid_url: int = 0
    removed_unrecognized_label: int = 0
    removed_conflicting_duplicate_urls: int = 0
    removed_duplicate_urls: int = 0
    retained_rows: int = 0
    retained_domains: int = 0
    class_counts: dict[int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _find_column(columns: list[str], candidates: set[str]) -> str | None:
    return next((column for column in columns if column.strip().lower() in candidates), None)


def _map_labels(values: pd.Series, dataset_kind: str) -> pd.Series:
    """Map known source label semantics to 1=phishing, 0=legitimate."""
    normalized = values.astype(str).str.strip().str.lower()
    if dataset_kind == "phiusiil":
        return normalized.map({"0": 1, "1": 0})
    return normalized.map(
        {
            "phishing": 1,
            "malicious": 1,
            "bad": 1,
            "1": 1,
            "legitimate": 0,
            "benign": 0,
            "good": 0,
            "0": 0,
        }
    )


def infer_dataset_kind(frame: pd.DataFrame) -> str:
    """Identify PhiUSIIL by its characteristic columns, not its filename."""
    lowered = {column.lower() for column in frame.columns}
    return "phiusiil" if {"filename", "urlsimilarityindex", "label"}.issubset(lowered) else "generic"


def load_and_clean_dataset(path: str | Path) -> tuple[pd.DataFrame, CleaningReport]:
    """Load URL/label data, remove unusable rows, and make leakage keys.

    Different valid URLs on one hostname are retained: collapsing them would
    distort URL-level distributions. Splits group on ``domain`` where an
    unseen-domain protocol is needed.
    """
    source = Path(path)
    raw = pd.read_csv(source, low_memory=False)
    report = CleaningReport(source=str(source), input_rows=len(raw))
    url_column = _find_column(list(raw.columns), {"url"})
    label_column = _find_column(list(raw.columns), {"label", "type", "class", "target", "result"})
    if url_column is None or label_column is None:
        raise ValueError(f"{source} must have URL and label columns")

    selected: pd.DataFrame = raw.loc[:, [url_column, label_column]]
    frame = selected.rename(columns={url_column: "url", label_column: "source_label"}).copy()
    missing_mask = frame["url"].isna() | frame["source_label"].isna()
    report.removed_missing = int(missing_mask.sum())
    frame = frame.loc[~missing_mask].copy()
    frame["url"] = frame["url"].astype(str).str.strip()
    frame["target"] = _map_labels(frame["source_label"], infer_dataset_kind(raw))
    unknown_mask = frame["target"].isna()
    report.removed_unrecognized_label = int(unknown_mask.sum())
    frame = frame.loc[~unknown_mask].copy()
    frame["target"] = frame["target"].astype(int)

    url_keys: list[str | None] = []
    domains: list[str | None] = []
    for url in frame["url"]:
        try:
            url_keys.append(canonical_url_key(url))
            domains.append(get_domain(url))
        except (InvalidURLException, ValueError):
            url_keys.append(None)
            domains.append(None)
    frame["url_key"] = url_keys
    frame["domain"] = domains
    invalid_mask = frame["url_key"].isna() | frame["domain"].isna()
    report.removed_invalid_url = int(invalid_mask.sum())
    frame = frame.loc[~invalid_mask].copy()

    conflicting_keys = frame.groupby("url_key")["target"].nunique()
    conflicting_keys = set(conflicting_keys[conflicting_keys > 1].index)
    conflict_mask = frame["url_key"].isin(conflicting_keys)
    report.removed_conflicting_duplicate_urls = int(conflict_mask.sum())
    frame = frame.loc[~conflict_mask].copy()
    duplicate_mask = frame.duplicated("url_key", keep="first")
    report.removed_duplicate_urls = int(duplicate_mask.sum())
    frame = frame.loc[~duplicate_mask].copy()
    frame = frame[["url", "target", "url_key", "domain"]].reset_index(drop=True)
    report.retained_rows = len(frame)
    report.retained_domains = int(frame["domain"].nunique())
    report.class_counts = {int(label): int(count) for label, count in frame["target"].value_counts().sort_index().items()}
    return frame, report


def make_stratified_random_split(frame: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A conventional URL-level split retained for same-dataset comparison."""
    train, test = train_test_split(frame, test_size=test_size, random_state=seed, stratify=frame["target"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_unseen_domain_split(frame: pd.DataFrame, n_splits: int = 5, seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out whole hostnames with approximately stratified class labels."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, test_index = next(splitter.split(frame, frame["target"], groups=frame["domain"]))
    train = frame.iloc[train_index].reset_index(drop=True)
    test = frame.iloc[test_index].reset_index(drop=True)
    if set(train["domain"]) & set(test["domain"]):
        raise RuntimeError("Domain leakage detected while creating unseen-domain split")
    return train, test


def domain_cv_splits(frame: pd.DataFrame, n_splits: int = 3, seed: int = RANDOM_SEED):
    """Yield deterministic train/validation positions without shared domains."""
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return splitter.split(frame, frame["target"], groups=frame["domain"])


def remove_overlapping_domains(candidate: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Remove rows whose hostnames occur in the reference data."""
    reference_domains = [str(d) for d in reference["domain"].dropna().tolist()]
    return candidate.loc[~candidate["domain"].isin(reference_domains)].reset_index(drop=True)
