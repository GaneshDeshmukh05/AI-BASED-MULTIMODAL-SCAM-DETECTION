"""Production-facing FastAPI endpoints for phishing URL analysis (offline model + optional domain intelligence)."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.config import ADMIN_TOKEN, DOMAIN_INTEL_ENABLED
from src.domain_intelligence import collect_intelligence
from src.evidence import SourceEvidence
from src.risk_scoring import merge_intelligence
from src.threat_feeds import (
    export_training_rows,
    lookup_url,
    lookup_urlhaus,
    update_openphish_feed,
)

# Relative imports for schemas and services inside app/
try:
    from .schemas import URLRequest
    from .services.model_service import ModelUnavailableError, get_model_service
except ImportError:
    from app.schemas import URLRequest
    from app.services.model_service import ModelUnavailableError, get_model_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("phishing_url_detector")

# ---------------------------------------------------------------------------
# Rate Limiter Setup
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

app = FastAPI(
    title="Explainable Phishing URL Detector",
    version="3.0.0",
    description=(
        "URL-feature analysis with XGBoost and SHAP explanations, optionally combined with "
        "DNS/RDAP/TLS/lookalike domain intelligence and URLhaus reputation."
    ),
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# Enable Cross-Origin Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root_redirect() -> Any:
    """Redirect root path to interactive API documentation."""
    return RedirectResponse(url="/docs")

# ---------------------------------------------------------------------------
# Admin Authentication
# ---------------------------------------------------------------------------
_bearer = HTTPBearer(auto_error=False)


def _verify_admin(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> None:
    """Dependency: verify the Bearer token for admin endpoints."""
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Admin endpoints are disabled. Set PHISHING_ADMIN_TOKEN to enable.",
        )
    if credentials is None or credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", tags=["operational"])
def health() -> Dict[str, Any]:
    """Verify that a compatible trained model bundle can be loaded."""
    try:
        service = get_model_service()
        service.load()
        return {
            "status": "ok",
            "model_loaded": True,
            "feature_count": len(service.feature_columns),
            "operating_threshold": service.threshold,
        }
    except ModelUnavailableError as error:
        logger.error("Health check failed: %s", error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error


@app.post(
    "/admin/update-threat-feed",
    tags=["threat-intelligence"],
    dependencies=[Depends(_verify_admin)],
)
def refresh_threat_feed(request: Request) -> Dict[str, Any]:
    """Manually refresh local OpenPhish cache."""
    try:
        result = update_openphish_feed()
        result["training_export"] = export_training_rows()
        return result
    except Exception as error:
        logger.exception("Threat feed refresh failed")
        raise HTTPException(status_code=502, detail=f"Threat feed refresh failed: {type(error).__name__}") from error


@app.post("/analyze-url", tags=["analysis"])
def analyze_url(payload: URLRequest, request: Request) -> Dict[str, Any]:
    """Classify one validated URL and return model-interpretation information."""
    try:
        result = get_model_service().analyze(payload.url)

        threat_intel = lookup_url(payload.url)

        use_intel = (
            DOMAIN_INTEL_ENABLED
            if payload.include_domain_intelligence is None
            else payload.include_domain_intelligence
        )
        domain_evidence = None
        if use_intel:
            try:
                domain_evidence = collect_intelligence(payload.url)

                try:
                    urlhaus_result = lookup_urlhaus(payload.url, timeout=3.0)
                    stale = urlhaus_result.get("is_stale", False)
                    uh_ev = (
                        SourceEvidence.stale_feed("urlhaus", urlhaus_result)
                        if stale
                        else SourceEvidence.ok("urlhaus", urlhaus_result, stale_after_hours=6.0)
                    )
                    domain_evidence.sources.append(uh_ev)
                except Exception:
                    logger.debug("URLhaus lookup failed; skipping", exc_info=True)

            except Exception:
                logger.exception("Domain intelligence failed; returning model-only result")

        result = merge_intelligence(result, domain_evidence)
        result["threat_intelligence"] = threat_intel

        if threat_intel.get("is_known_phishing"):
            result["prediction"] = "phishing"
            result["composite_risk_score"] = 100
            result["risk_score"] = 100
            result["risk_level"] = "critical"

        if payload.include_redirect_resolution:
            try:
                from src.redirect_resolver import resolve_redirects  # noqa: PLC0415
                chain = resolve_redirects(payload.url)
                result["redirect_chain"] = chain.to_dict()
            except Exception:
                logger.exception("Redirect resolution failed; skipping")
                result["redirect_chain"] = None
        else:
            result["redirect_chain"] = None

        logger.info(
            "URL analyzed: prediction=%s probability=%.4f composite=%s intel=%s",
            result["prediction"],
            result["phishing_probability"],
            result["composite_risk_score"],
            result["domain_intelligence_used"],
        )
        return result
    except ModelUnavailableError as error:
        logger.error("Analysis unavailable: %s", error)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    except HTTPException:
        raise
    except Exception as error:
        logger.exception("Unexpected URL analysis error")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="URL analysis failed") from error


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=True)