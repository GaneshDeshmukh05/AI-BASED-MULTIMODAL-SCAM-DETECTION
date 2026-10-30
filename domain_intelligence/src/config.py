"""Central paths and deterministic experiment settings."""

from __future__ import annotations

import os
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SHAP_RESULTS_DIR = RESULTS_DIR / "shap"

# --- Experiment Settings ---
RANDOM_SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 3
DEFAULT_THRESHOLD = 0.50

# --- File Paths ---
PRIMARY_DATASET = Path(os.getenv("PHISHING_PRIMARY_DATASET", DATA_DIR / "raw_data.csv"))
CROSS_DATASET = Path(os.getenv("PHISHING_CROSS_DATASET", DATA_DIR / "phishing.csv"))
MODEL_PATH = Path(os.getenv("PHISHING_MODEL_PATH", MODELS_DIR / "xgboost_url_model.joblib"))
EXPLAINER_PATH = Path(os.getenv("PHISHING_EXPLAINER_PATH", MODELS_DIR / "shap_url_explainer.joblib"))
MODEL_CONFIG_PATH = MODELS_DIR / "best_xgboost_config.json"
FEATURE_MANIFEST_PATH = MODELS_DIR / "feature_manifest.json"
DOMAIN_SCORER_PATH = MODELS_DIR / "domain_scorer.joblib"


# --- Environment Parsing Helpers ---
def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


# --- Domain intelligence (optional online enrichment, applied AFTER the model) ---
# The XGBoost model stays URL-only/offline. These signals (DNS, WHOIS, TLS) are
# combined with its probability by ``src.risk_scoring`` and are never fed to the
# model, because the training data has no such columns.
DOMAIN_INTEL_ENABLED = _env_flag("PHISHING_DOMAIN_INTEL", True)
DOMAIN_INTEL_TIMEOUT = _env_float("PHISHING_INTEL_TIMEOUT", 5.0)  # overall wall-clock budget, seconds

NEW_DOMAIN_DAYS = 30            # registered fewer than this many days ago -> suspicious
NEW_DOMAIN_PENALTY = 15         # risk points added (0-100 scale)
INVALID_CERT_PENALTY = 20       # risk points added when an https:// URL fails certificate verification
ESTABLISHED_DOMAIN_DAYS = 3650  # > 10 years + valid cert
ESTABLISHED_DOMAIN_FACTOR = 0.3
MATURE_DOMAIN_DAYS = 1825       # > 5 years + valid cert
MATURE_DOMAIN_FACTOR = 0.6

# --- Feed staleness ---
# Feeds older than this many hours are flagged as stale in SourceEvidence.
FEED_MAX_AGE_HOURS = _env_float("PHISHING_FEED_MAX_AGE_HOURS", 6.0)

# --- Optional API keys ---
VIRUSTOTAL_API_KEY: str = os.getenv("VIRUSTOTAL_API_KEY", "")

# --- Admin authentication ---
# Set PHISHING_ADMIN_TOKEN to a hard-to-guess secret before public deployment.
# If unset, admin endpoints return 501 with an explanatory message.
ADMIN_TOKEN: str = os.getenv("PHISHING_ADMIN_TOKEN", "")

# --- Redirect resolver ---
REDIRECT_MAX_HOPS: int = _env_int("PHISHING_REDIRECT_MAX_HOPS", 5)
REDIRECT_TIMEOUT: float = _env_float("PHISHING_REDIRECT_TIMEOUT", 3.0)


def ensure_output_directories() -> None:
    """Create the non-source directories used by reproducible experiments."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SHAP_RESULTS_DIR.mkdir(parents=True, exist_ok=True)