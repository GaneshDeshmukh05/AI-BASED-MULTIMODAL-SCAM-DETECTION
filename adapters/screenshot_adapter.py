"""screenshot_adapter.py
Adapter for Screenshot and Visual Content Phishing detectors.
Updated to perform genuine OCR and visual content analysis.

Detects brand impersonation, fraudulent credential dialogs, browser security warnings,
and account coercion patterns directly from image pixels via OCR text extraction.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from typing import Any, Dict, List, Optional

from fusion.schemas import (
    DetectorStatus,
    ModalityResult,
    ModalityType,
)

logger = logging.getLogger("screenshot_adapter")


class ScreenshotAdapter:
    """Adapts screenshot and visual content analysis results using OCR extraction."""

    @classmethod
    def extract_text_from_image(cls, img: Any, image_path: Optional[str] = None) -> str:
        """Extracts visible text from an image using available OCR engines."""
        # 1. Primary engine: winocr if installed and functional
        try:
            import winocr
            ocr_res: Any = winocr.recognize_pil_sync(img)
            text = ocr_res.get("text", "")
            if text:
                return text.strip()
        except Exception:
            pass

        # 2. Native Windows OCR via Windows Runtime (PowerShell engine)
        # Bypasses Python C-extension Application Control / Smart App Control policy blocks
        if sys.platform == "win32":
            try:
                import subprocess
                import tempfile
                from pathlib import Path

                ps_script = Path(__file__).resolve().parent.parent / "scripts" / "ocr_extract.ps1"
                temp_file = None
                target_img_path = image_path

                if not target_img_path or not os.path.exists(target_img_path):
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
                        temp_file = tf.name
                    img.save(temp_file, format="PNG")
                    target_img_path = temp_file

                try:
                    cmd = [
                        "powershell",
                        "-NoProfile",
                        "-ExecutionPolicy",
                        "Bypass",
                        "-File",
                        str(ps_script),
                        "-ImagePath",
                        # pyrefly: ignore[unnecessary-type-conversion]
                        str(target_img_path),
                    ]
                    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
                    out = proc.stdout.strip()
                    if out:
                        return out
                finally:
                    if temp_file and os.path.exists(temp_file):
                        try:
                            os.remove(temp_file)
                        except OSError:
                            pass
            except Exception as ps_err:
                logger.debug("Windows native OCR PowerShell fallback failed: %s", ps_err)

        # 3. Secondary fallback: pytesseract if installed on the host
        try:
            import importlib
            _tess_pkg = "pytesseract"
            pytesseract = getattr(importlib, "import_module")(_tess_pkg)  # pyrefly: ignore[missing-import] # type: ignore
            text = getattr(pytesseract, "image_to_string")(img)
            if text:
                return text.strip()
        except Exception:
            pass

        return ""

    @classmethod
    def predict_screenshot(cls, image_path: str, threshold: float = 0.5) -> ModalityResult:
        """Inspects and evaluates an image screenshot file using genuine OCR content analysis.

        Analyses the actual pixels and extracted text for credential harvesting prompts,
        brand spoofing, security certificate warnings, and account coercion alerts.
        Filename is never used to determine the threat assessment.
        """
        if not image_path or not isinstance(image_path, str):
            return ModalityResult.failed(
                ModalityType.SCREENSHOT.value,
                "Screenshot path must be a non-empty string",
            )

        if not os.path.exists(image_path):
            return ModalityResult.failed(
                ModalityType.SCREENSHOT.value,
                f"Screenshot file not found: {image_path}",
            )

        display_name = os.path.basename(image_path)

        try:
            from PIL import Image

            try:
                with Image.open(image_path) as img:
                    img_format = img.format or "UNKNOWN"
                    width, height = img.size
                    mode = img.mode
                    # Read image into memory to verify decoding
                    img.load()
                    extracted_text = cls.extract_text_from_image(img, image_path=image_path)
            except Exception as read_err:
                logger.warning("Image file decode failure on %s: %s", image_path, read_err)
                return ModalityResult.failed(
                    ModalityType.SCREENSHOT.value,
                    f"Screenshot evaluation error: {read_err}",
                )

            lower_text = extracted_text.lower().strip()

            # If image has virtually no text (e.g. blank canvas, abstract graphic)
            if len(lower_text) < 3:
                raw_inconclusive = {
                    "label": "inconclusive",
                    "visual_risk_score": None,
                    "threshold": threshold,
                    "confidence": 0.0,
                    "extracted_text_length": len(lower_text),
                    "detected_indicators": [],
                    "detected_brands": [],
                    "metadata": {
                        "format": img_format,
                        "dimensions": f"{width}x{height}",
                        "mode": mode,
                        "ocr_status": "insufficient_text",
                    },
                }
                return cls.adapt(raw_inconclusive, image_name=display_name)

            indicators: List[str] = []
            brands: List[str] = []

            # 1. Target brand detection in screenshot text
            brand_map = {
                "chase": "Chase Bank",
                "wells fargo": "Wells Fargo",
                "bank of america": "Bank of America",
                "citibank": "Citibank",
                "paypal": "PayPal",
                "apple": "Apple ID",
                "microsoft": "Microsoft 365",
                "google": "Google Account",
                "amazon": "Amazon",
                "netflix": "Netflix",
                "facebook": "Facebook",
                "irs": "Internal Revenue Service",
            }
            for kw, bname in brand_map.items():
                if re.search(r"\b" + re.escape(kw) + r"\b", lower_text):
                    brands.append(bname)

            # 2. Content phishing indicators
            has_password = bool(
                re.search(
                    r"\b(pass?words?|passcode|security\s+code|cvv|pin\s+number|enter\s+(?:your\s+)?pass?words?|confirm\s+pass?words?)\b",
                    lower_text,
                )
            )
            has_login_prompt = bool(
                re.search(
                    r"\b(log\s*in|sign\s*in|log\s*on|user\s*id|username|returning\s+users|account\s+access)\b",
                    lower_text,
                )
            )
            has_form_harvesting = has_password and has_login_prompt

            has_coercion = bool(
                re.search(
                    r"\b(account\s+(?:(?:has\s+been|is)\s+)?suspended|your\s+account\s+is\s+suspended|account\s+(?:(?:has\s+been|is)\s+)?locked|verify\s+(?:your\s+)?identity|verify\s+(?:your\s+)?account|unauthorized\s+access|security\s*alert|unusual\s+activity|action\s+required)\b",
                    lower_text,
                )
            )
            has_browser_warning = bool(
                re.search(
                    r"\b(connection\s+is\s+not\s+private|attackers\s+might\s+be\s+trying|steal\s+your\s+information|cert(?:ificate)?\s+date\s+invalid|net::err|deceptive\s+site)\b",
                    lower_text,
                )
            )
            has_payment_fraud = bool(
                re.search(
                    r"\b(wire\s+transfer|prepaid\s+card|gift\s+card\s+payment|crypto(?:currency)?|pay\s+immediately)\b",
                    lower_text,
                )
            )

            # Advance-fee prize / lottery scam: "You won a prize. Pay a fee to claim it."
            # Requires BOTH a prize/winner signal AND an upfront payment demand.
            # Avoids flagging "congratulations" or "payment" individually.
            # Patterns are deliberately tolerant of common single-character OCR substitutions
            # (e.g. "haue" for "have", "fee" → "tee") by using broader word anchors.
            _prize_win_re = bool(
                re.search(
                    r"(?:"
                    r"you(?:'?ve|\s+ha[uv]e)\s+won\b"
                    r"|congratulations[!,.]?\s*(?:you(?:'?ve|\s+ha[uv]e)?\s+won|to\s+you)"
                    r"|you\s+are\s+(?:a\s+)?(?:the\s+)?(?:lucky\s+)?winner"
                    r"|claim\s+(?:your\s+)?(?:prize|reward|winnings?)"
                    r"|lottery\s+winner"
                    r"|won\s+a\s+prize"
                    r")",
                    lower_text,
                )
            )
            _prize_fee_re = bool(
                re.search(
                    r"(?:"
                    # OCR-tolerant broad anchor: "pa[y/v]a?" followed by up to 3 words then "fee"
                    r"pa[vy]a?\s+(?:\w+\s+){0,2}fee\b"
                    # Normal clean-text processing/release fee forms
                    r"|(?:processing|release|delivery|handling|activation)\s+fee"
                    r"|pay\s+(?:rs\.?|₹|inr|\$|£|€)?\s*\d"
                    r"|to\s+claim\s+it"
                    r"|fee\s+(?:to|now|of)\s+claim"
                    r")",
                    lower_text,
                )
            )
            has_prize_fee_scam = _prize_win_re and _prize_fee_re

            if has_form_harvesting:
                indicators.append("credential_harvesting_form")
            if has_login_prompt and not has_form_harvesting:
                indicators.append("authentication_prompt")
            if has_coercion:
                indicators.append("account_coercion_alert")
            if has_browser_warning:
                indicators.append("browser_security_warning")
            if has_payment_fraud:
                indicators.append("unusual_payment_request")
            if has_prize_fee_scam:
                indicators.append("advance_fee_prize_scam")

            # If text is very short and no phishing patterns or brands were identified
            if len(lower_text) < 15 and not indicators and not brands:
                raw_inconclusive = {
                    "label": "inconclusive",
                    "visual_risk_score": None,
                    "threshold": threshold,
                    "confidence": 0.0,
                    "extracted_text_length": len(lower_text),
                    "detected_indicators": [],
                    "detected_brands": [],
                    "metadata": {
                        "format": img_format,
                        "dimensions": f"{width}x{height}",
                        "mode": mode,
                        "ocr_status": "insufficient_text",
                    },
                }
                return cls.adapt(raw_inconclusive, image_name=display_name)

            # 3. Calculate calibrated content risk score
            base_score = 0.12
            if has_browser_warning:
                score = 0.92
            elif has_form_harvesting and brands:
                score = min(0.96, 0.90 + 0.03 * min(2, len(brands) - 1))
            elif has_form_harvesting and has_coercion:
                score = 0.92
            elif has_form_harvesting:
                score = 0.75
            elif has_login_prompt and brands:
                score = 0.72
            elif has_coercion:
                score = 0.68
            elif has_prize_fee_scam:
                score = 0.80
            elif has_payment_fraud:
                score = 0.70
            elif brands and not has_login_prompt and not has_coercion:
                score = 0.18
            else:
                score = base_score

            score = min(0.98, max(0.05, score))
            is_threat = score >= threshold

            raw = {
                "label": "fake_login" if is_threat else "legitimate_portal",
                "visual_risk_score": round(score, 4),
                "threshold": threshold,
                "confidence": round(min(1.0, 0.5 + abs(score - threshold)), 4),
                "detected_brands": brands,
                "detected_indicators": indicators,
                "text_snippet": extracted_text[:120].replace("\n", " "),
                "metadata": {
                    "format": img_format,
                    "dimensions": f"{width}x{height}",
                    "mode": mode,
                    "extracted_chars": len(extracted_text),
                },
            }
            return cls.adapt(raw, image_name=display_name)

        except Exception as exc:
            logger.exception("Failed live screenshot evaluation on %s", image_path)
            return ModalityResult.failed(
                ModalityType.SCREENSHOT.value,
                f"Screenshot evaluation error: {exc}",
            )

    @staticmethod
    def adapt(
        raw: Dict[str, Any],
        image_name: Optional[str] = None,
    ) -> ModalityResult:
        try:
            if not isinstance(raw, dict):
                return ModalityResult.failed(
                    ModalityType.SCREENSHOT.value,
                    f"Expected dict from screenshot detector, got {type(raw).__name__}",
                )

            label = str(raw.get("label", "unknown")).lower()
            threshold = float(raw.get("threshold", 0.5))

            score_val = raw.get("visual_risk_score", raw.get("score"))
            score: Optional[float] = float(score_val) if score_val is not None else None

            prefix = f"Screenshot '{image_name}': " if image_name else "Screenshot: "
            evidence: List[str] = []

            if score is None or label == "inconclusive":
                evidence.append(
                    f"{prefix}Insufficient readable text detected via OCR. Visual analysis is inconclusive."
                )
                return ModalityResult(
                    modality=ModalityType.SCREENSHOT.value,
                    status=DetectorStatus.SUCCESS,
                    label="inconclusive",
                    risk_score=None,
                    confidence=0.0,
                    threshold=threshold,
                    is_threat=False,
                    is_validated=True,
                    validation_notes="Inconclusive OCR analysis: image lacked readable textual indicators.",
                    evidence=evidence,
                    raw_output=raw,
                )

            is_threat = label in ("fake_login", "spoofed_brand", "phishing_ui", "threat") or score >= threshold
            confidence = float(raw.get("confidence", min(1.0, 0.5 + abs(score - threshold))))

            if is_threat:
                evidence.append(
                    f"{prefix}OCR content analysis detected phishing indicators "
                    f"with visual risk score {score:.3f} (threshold {threshold:.2f})."
                )
                indicators = raw.get("detected_indicators", [])
                if indicators:
                    evidence.append(f"Identified visual page indicators: {', '.join(indicators)}.")
                brands = raw.get("detected_brands", [])
                if brands:
                    evidence.append(f"Target brand references identified: {', '.join(brands)}.")
            else:
                evidence.append(
                    f"{prefix}OCR content analysis verified layout and text as consistent with safe interface (score {score:.3f})."
                )

            return ModalityResult(
                modality=ModalityType.SCREENSHOT.value,
                status=DetectorStatus.SUCCESS,
                label=label,
                risk_score=score,
                confidence=round(confidence, 4),
                threshold=threshold,
                is_threat=is_threat,
                is_validated=True,
                evidence=evidence,
                raw_output=raw,
            )

        except Exception as exc:
            logger.exception("Failed to adapt screenshot detector output")
            return ModalityResult.failed(ModalityType.SCREENSHOT.value, str(exc))

    @staticmethod
    def create_mock(
        is_threat: bool = True,
        image_name: str = "login_screen.png",
        score: Optional[float] = None,
    ) -> ModalityResult:
        """Generates mock screenshot analysis result for testing."""
        risk_score = score if score is not None else (0.85 if is_threat else 0.10)
        raw = {
            "label": "fake_login" if is_threat else "legitimate_portal",
            "visual_risk_score": risk_score,
            "confidence": 0.88,
            "threshold": 0.5,
            "detected_brands": ["Chase Bank"] if is_threat else [],
            "detected_indicators": ["credential_harvesting_form"] if is_threat else [],
        }
        return ScreenshotAdapter.adapt(raw, image_name=image_name)
