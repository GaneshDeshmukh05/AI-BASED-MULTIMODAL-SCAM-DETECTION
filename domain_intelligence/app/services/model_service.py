"""Thread-safe loading and inference for the trained URL model bundle."""

from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path Auto-Resolution: Ensures 'src' is always importable regardless of CWD
# ---------------------------------------------------------------------------
_current_file = Path(__file__).resolve()
_search_dirs = [
    _current_file.parent,
    _current_file.parent.parent,
    _current_file.parent.parent.parent,
]
for _candidate in _search_dirs:
    if (_candidate / "src").is_dir() and (_candidate / "models").is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

import joblib
import pandas as pd

from src.config import EXPLAINER_PATH, MODEL_CONFIG_PATH, MODEL_PATH
from src.explainability import individual_contributions
from src.feature_engineering import FEATURE_NAMES, extract_features

# Suppress XGBoost / pickle backward-compatibility warning during deserialization
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="pickle")


class ModelUnavailableError(RuntimeError):
    """A trained compatible model bundle is not yet available to the API."""


class ModelService:
    """Load one matching model/explainer bundle and analyze validated URLs."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        explainer_path: Path = EXPLAINER_PATH,
        config_path: Path = MODEL_CONFIG_PATH,
    ):
        self.model_path = Path(model_path)
        self.explainer_path = Path(explainer_path)
        self.config_path = Path(config_path)
        self.model: Any | None = None
        self.explainer: Any | None = None
        self.feature_columns: list[str] = list(FEATURE_NAMES)
        self.threshold: float = 0.5

    def load(self) -> None:
        """Load artifacts and enforce their documented feature schema."""
        if self.model is not None and self.explainer is not None:
            return

        missing = [
            path.name
            for path in (self.model_path, self.explainer_path, self.config_path)
            if not path.exists()
        ]
        if missing:
            raise ModelUnavailableError(
                f"Trained model bundle is unavailable ({', '.join(missing)}). Run `python -m experiments.run_all` first."
            )

        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
            feature_columns = list(config["feature_columns"])
            if feature_columns != list(FEATURE_NAMES):
                raise ModelUnavailableError(
                    "Model feature schema does not match this API version"
                )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = joblib.load(self.model_path)
                try:
                    self.explainer = joblib.load(self.explainer_path)
                except Exception:
                    import shap
                    self.explainer = shap.TreeExplainer(self.model)

            self.feature_columns = feature_columns
            self.threshold = float(config.get("selected_threshold", 0.5))
        except ModelUnavailableError:
            raise
        except Exception as error:
            raise ModelUnavailableError(
                f"Could not load the trained model bundle: {error}"
            ) from error

    @staticmethod
    def _risk_level(probability: float) -> str:
        if probability >= 0.80:
            return "critical"
        if probability >= 0.60:
            return "high"
        if probability >= 0.40:
            return "medium"
        return "low"

    def analyze(self, url: str) -> dict[str, Any]:
        """Return a model prediction and signed SHAP explanation for one URL."""
        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string.")

        clean_url = url.strip()
        self.load()
        assert self.model is not None and self.explainer is not None

        feature_values = extract_features(clean_url)
        features = pd.DataFrame(
            [[feature_values[name] for name in self.feature_columns]],
            columns=self.feature_columns,
        )

        probability = float(self.model.predict_proba(features)[0, 1])
        contributions = individual_contributions(self.explainer, features)

        positive = sorted(
            ((name, value) for name, value in contributions.items() if value > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        negative = sorted(
            ((name, value) for name, value in contributions.items() if value < 0),
            key=lambda item: item[1],
        )[:3]

        if not positive:
            positive = sorted(
                contributions.items(), key=lambda item: item[1], reverse=True
            )[:3]
        if not negative:
            negative = sorted(contributions.items(), key=lambda item: item[1])[:3]

        is_phishing = probability >= self.threshold
        rounded_contributions = {
            name: round(value, 6) for name, value in contributions.items()
        }

        return {
            "url": clean_url,
            "prediction": "phishing" if is_phishing else "legitimate",
            "phishing_probability": round(probability, 6),
            "risk_score": round(probability * 100),
            "confidence": round(max(probability, 1 - probability), 6),
            "risk_level": self._risk_level(probability),
            "operating_threshold": self.threshold,
            "extracted_features": {
                name: float(value) for name, value in feature_values.items()
            },
            "top_risk_factors": [
                {
                    "feature": name,
                    "shap_value": round(value, 6),
                    "direction": "phishing",
                }
                for name, value in positive
            ],
            "top_legitimate_factors": [
                {
                    "feature": name,
                    "shap_value": round(value, 6),
                    "direction": "legitimate",
                }
                for name, value in negative
            ],
            "top_features": [name for name, _ in positive],
            "shap_contributions": rounded_contributions,
            "shap_explanation": rounded_contributions,
        }


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Provide a process-wide lazily loaded service for FastAPI handlers."""
    return ModelService()


if __name__ == "__main__":
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    print(f"\n[+] Testing ModelService with: {test_url}")
    try:
        service = get_model_service()
        service.load()
        print(f"[OK] Model Loaded Successfully! ({len(service.feature_columns)} features, threshold={service.threshold})")
        res = service.analyze(test_url)
        print(f"[OK] Prediction: {res['prediction'].upper()} (Prob: {res['phishing_probability']:.4f}, Risk: {res['risk_score']}/100, Level: {res['risk_level'].upper()})")
        print(f"[OK] Top Risk Factors: {', '.join(f['feature'] for f in res['top_risk_factors'])}")
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
