"""api.py
Production-ready FastAPI REST service for the Multimodal Scam Detection System.
Implements the canonical integration contract specified in docs/API_CONTRACT.md.

Allows any web frontend (React, Next.js, Vue, mobile apps, curl) to run
multimodal threat analyses over Audio, URLs, SMS/Text, and Screenshots.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure root repository directory is on sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fusion import analyze_multimodal

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("scam_detection_api")

app = FastAPI(
    title="Multimodal Scam & Deepfake Detection API",
    version="1.0.0",
    description="Unified API combining Audio Deepfake & Scam detection, 26-feature XGBoost URL phishing detection, NLP text analysis, and screenshot OCR forensic analysis.",
)

# Enable CORS for web and mobile frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextURLRequest(BaseModel):
    """Payload for text and URL only analysis via JSON."""

    url: Optional[str] = Field(default=None, description="Target URL or domain to evaluate")
    text: Optional[str] = Field(default=None, description="SMS, email, or chat transcript text")
    include_domain_intelligence: bool = Field(
        default=False,
        description="Whether to perform live DNS/WHOIS/TLS OSINT lookups (True is slower)",
    )
    explainer_provider: str = Field(
        default="auto",
        description="Explanation provider: 'auto', 'gemini', 'openai', or 'deterministic'",
    )


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    """Returns server status and available models."""
    return {
        "status": "healthy",
        "service": "Multimodal Scam & Deepfake Detection API",
        "version": "1.0.0",
        "available_modalities": ["audio_scam", "audio_spoof", "url", "text", "screenshot"],
        "docs_url": "/docs",
    }


@app.get("/api/scenarios")
def list_scenarios() -> List[Dict[str, Any]]:
    """Returns preset test scenarios for easy frontend demo integration."""
    return [
        {
            "id": "threat_convergence",
            "name": "Threat Convergence (Audio + URL Phishing)",
            "description": "Simultaneous fraudulent phone call and malicious PayPal credential harvest URL.",
            "url": "http://paypal.com-account-verify.online/login",
            "text": "URGENT: Your Chase bank account is suspended. Verify credentials immediately at http://paypal.com-account-verify.online/login",
            "sample_audio": "audio_model/dataset/SCAM_CALLS/scam_22.wav",
        },
        {
            "id": "benign_safe",
            "name": "Legitimate Communication",
            "description": "Normal conversation and verified secure website.",
            "url": "https://www.google.com",
            "text": "Hi Mom, I will be home by 6 PM for dinner. Let me know if you need any groceries on my way back.",
            "sample_audio": "audio_model/dataset/NORMAL_CALLS/normal_17.wav",
        },
        {
            "id": "delivery_scam",
            "name": "Package Delivery Smishing",
            "description": "Postal failure text asking for fake redelivery fee.",
            "url": "http://usps-update-parcel.com/track",
            "text": "USPS Alert: Your package delivery has failed due to an incomplete address. Pay the 1.99 redelivery fee to reschedule.",
            "sample_audio": None,
        },
    ]


@app.post("/api/analyze")
async def analyze(
    url: Optional[str] = Form(default=None),
    text: Optional[str] = Form(default=None),
    include_domain_intelligence: bool = Form(default=False),
    explainer_provider: str = Form(default="auto"),
    audio: Optional[UploadFile] = File(default=None),
    screenshot: Optional[UploadFile] = File(default=None),
) -> Dict[str, Any]:
    """Multipart endpoint to analyze any combination of Audio, URL, Text, and Screenshot.

    Files are streamed to temporary storage, evaluated through the multimodal pipeline,
    and automatically cleaned up.
    """
    temp_audio_path: Optional[str] = None
    temp_screenshot_path: Optional[str] = None

    try:
        # Save audio file to temp if uploaded
        if audio is not None and audio.filename:
            suffix = Path(audio.filename).suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as af:
                shutil.copyfileobj(audio.file, af)
                temp_audio_path = af.name

        # Save screenshot file to temp if uploaded
        if screenshot is not None and screenshot.filename:
            suffix = Path(screenshot.filename).suffix or ".png"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as sf:
                shutil.copyfileobj(screenshot.file, sf)
                temp_screenshot_path = sf.name

        clean_url = url.strip() if url and url.strip() else None
        clean_text = text.strip() if text and text.strip() else None

        if not any([temp_audio_path, temp_screenshot_path, clean_url, clean_text]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one modality (audio, url, text, screenshot) must be supplied.",
            )

        logger.info(
            "Analyzing multimodal request (audio=%s, url=%s, text=%s, screenshot=%s, osint=%s)",
            bool(temp_audio_path),
            bool(clean_url),
            bool(clean_text),
            bool(temp_screenshot_path),
            include_domain_intelligence,
        )

        result = analyze_multimodal(
            audio_path=temp_audio_path,
            url=clean_url,
            text=clean_text,
            screenshot_path=temp_screenshot_path,
            include_domain_intelligence=include_domain_intelligence,
            explainer_provider=explainer_provider,
        )

        return result.to_dict()

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Analysis failed with unexpected exception: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {exc}",
        )
    finally:
        # Guarantee cleanup of temporary files
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except OSError:
                pass
        if temp_screenshot_path and os.path.exists(temp_screenshot_path):
            try:
                os.remove(temp_screenshot_path)
            except OSError:
                pass


@app.post("/api/analyze/json")
def analyze_json(payload: TextURLRequest) -> Dict[str, Any]:
    """JSON-only endpoint for lightweight clients analyzing URLs and text without file uploads."""
    clean_url = payload.url.strip() if payload.url and payload.url.strip() else None
    clean_text = payload.text.strip() if payload.text and payload.text.strip() else None

    if not clean_url and not clean_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either 'url' or 'text' must be provided.",
        )

    result = analyze_multimodal(
        url=clean_url,
        text=clean_text,
        include_domain_intelligence=payload.include_domain_intelligence,
        explainer_provider=payload.explainer_provider,
    )
    return result.to_dict()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

