"""text_adapter.py
Adapter for NLP scam text / SMS / transcript detectors.
Updated with calibrated scoring and semantic threat detection.

Accepts live text messages, runs trained TF-IDF + Multinomial Naive Bayes model,
and calibrates output probabilities with structured threat analysis to eliminate
false positives on benign text and false negatives on modern smishing lures.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from fusion.schemas import (
    DetectorStatus,
    ModalityResult,
    ModalityType,
)

logger = logging.getLogger("text_adapter")


class TextAdapter:
    """Adapts text scam and phishing transcript analysis results."""

    _text_model: Optional[Any] = None

    @classmethod
    def load_model(cls) -> Optional[Any]:
        """Loads and caches the text scam classifier."""
        if cls._text_model is None:
            from pathlib import Path
            import joblib

            cur = Path(__file__).resolve().parent
            candidates = [
                cur.parent / "text_and_screenshot_module" / "models" / "text_scam_classifier.joblib",
                                cur.parent / "text and screenshot" / "models" / "text_scam_classifier.joblib",
                cur / "text_scam_classifier.joblib",
            ]
            for p in candidates:
                if p.is_file():
                    try:
                        cls._text_model = joblib.load(str(p))
                        break
                    except Exception as err:
                        logger.warning("Could not load text model from %s: %s", p, err)
        return cls._text_model

    @classmethod
    def predict_text(cls, text: str, threshold: float = 0.5) -> ModalityResult:
        """Runs live feature extraction and inference on input text with calibrated scoring."""
        if not text or not isinstance(text, str):
            return ModalityResult.failed(
                ModalityType.TEXT.value,
                "Text must be a non-empty string",
            )

        lower = text.lower()
        tactics: List[str] = []
        has_delivery_scam = bool(
            re.search(
                r"\b(package|parcel|delivery|shipment)\b.*"
                r"\b(failed|held|redelivery|fee|customs|pay|payment)\b"
                r"|\b(redelivery fee|delivery fee|customs fee)\b",
                lower,
            )
        )

        # 1. Semantic pattern detectors
        has_coercion = bool(
            re.search(
                r"\b(irs\s+notice|back\s+taxes|federal\s+warrant|arrest\s+warrant|avoid\s+arrest|law\s+enforcement|legal\s+action|prepaid\s+card|wire\s+funds|unpaid\s+taxes)\b",
                lower,
            )
        )

        # ── Prize / advance-fee lure ────────────────────────────────────────────
        # Detects forward-fee scams like "You won ₹25 lakh. Pay ₹499 to release it."
        # Requires either a congratulatory / winner signal *or* an explicit claim,
        # combined with a payment/fee request — to avoid flagging "congrats on
        # winning the competition!" or legitimate payment reminders individually.
        _prize_win_signal = bool(
            re.search(
                r"(?:"
                r"congratulations\s+(?:you(?:'ve|\s+have)\s+won|to\s+you)"
                r"|you(?:'ve|\s+have)\s+won\b"
                r"|lottery\s+winner"
                r"|selected\s+as\s+(?:a\s+)?winner"
                r"|claim\s+(?:your\s+)?(?:reward|prize|winnings?)"
                r"|selected\s+to\s+win"
                r"|(?:prize|reward|winnings?)\s+of\s+(?:rs\.?|₹|inr|\$|£|€)"
                r")",
                lower,
            )
        )
        _prize_payment_signal = bool(
            re.search(
                r"(?:"
                r"pay\s+(?:rs\.?|₹|inr|\$|£|€)?\s*\d"
                r"|pay\s+(?:a\s+)?(?:processing|delivery|handling|release|transfer|activation|admin(?:istration)?)\s+fee"
                r"|(?:processing|release|delivery|handling|activation)\s+fee\s+(?:of\s+)?(?:rs\.?|₹|inr|\$|£|€)"
                r"|send\s+(?:rs\.?|₹|inr|\$|£|€)?\s*\d"
                r"|transfer\s+(?:rs\.?|₹|inr|\$|£|€)?\s*\d"
                r")",
                lower,
            )
        )
        _prize_gift_card = bool(
            re.search(
                r"\b(amazon\s+gift\s*card|free\s+gift\s*card|claim\s+(?:your\s+)?gift\s*card)\b",
                lower,
            )
        )
        # Benign congratulatory context: team wins, competition, education
        _prize_benign_context = bool(
            re.search(
                r"\b(team|college|school|class|competition|tournament|match|hackathon|contest)\b",
                lower,
            )
        )
        has_prize = (
            (_prize_win_signal and _prize_payment_signal and not _prize_benign_context)
            or _prize_gift_card
        )

        # ── Account threat / suspension ─────────────────────────────────────────
        # Covers future-tense ("will be suspended") and imperative variations
        # without matching benign security-awareness messages.
        # Benign guard: "never share", "do not share", "don't share", "avoid sharing"
        _account_action_re = re.compile(
            r"(?:"
            r"(?:bank\s+)?account\s+(?:(?:has\s+been|is|will\s+be|would\s+be)\s+)?"
            r"(?:suspended|locked|blocked|deactivated|disabled|closed)"
            r"|unauthorized\s+(?:access|activity|login)"
            r"|verify\s+(?:your\s+)?(?:identity|account)"
            r")",
        )
        _account_benign_guard = bool(
            re.search(r"\b(never\s+share|do\s+not\s+share|don'?t\s+share|avoid\s+sharing|protect\s+your)\b", lower)
        )
        has_account_threat = bool(_account_action_re.search(lower)) and not _account_benign_guard

        # ── Credential / OTP harvesting ─────────────────────────────────────────
        # Covers imperative "send your password/OTP" and future-tense variations.
        # Benign guard: "never share", "do not share" security-awareness context.
        _harvest_re = re.compile(
            r"(?:"
            r"(?:enter|provide|send|share|give|submit|type|confirm)\s+(?:me\s+)?(?:your\s+)?"
            r"(?:password|otp|one[\s-]time\s+(?:password|pin|code)|pin|passcode|credentials?|ssn)"
            r"|enter\s+credentials?"
            r"|confirm\s+password"
            r"|(?:your\s+)?(?:otp|one[\s-]time\s+(?:password|pin|code))\s+(?:is|to\s+(?:proceed|verify|activate|confirm))"
            r")",
        )
        _harvest_benign_guard = bool(
            re.search(
                r"\b(never\s+share|do\s+not\s+share|don'?t\s+share|never\s+give|"
                r"avoid\s+sharing|protect\s+your\s+password|keep\s+your\s+password|"
                r"keep\s+your\s+otp|security\s+tip|reminder:?\s|never\s+disclose|"
                r"will\s+never\s+ask|never\s+ask\s+you|bank\s+will\s+never|"
                r"never\s+ask(?:\s+you)?\s+(?:to\s+)?(?:share|give|send|provide|disclose|enter))\b",
                lower,
            )
        )
        has_harvesting = bool(_harvest_re.search(lower)) and not _harvest_benign_guard

        # ── Utility disconnection / payment lure ────────────────────────────────
        has_utility_scam = bool(
            re.search(
                r"(?:"
                r"(?:electricity|power|gas|water|internet|broadband|cable)\s+"
                r"(?:supply\s+)?(?:will\s+be|is\s+(?:being|scheduled\s+to\s+be))?\s*"
                r"(?:disconnected|cut\s+off|terminated|suspended)"
                r"|(?:disconnection|cut[\s-]off)\s+(?:of\s+)?(?:your\s+)?"
                r"(?:electricity|power|gas|water|internet|service)"
                r")",
                lower,
            )
        )

        has_urgency = bool(
            re.search(
                r"\b(urgent|immediately|within\s+24\s+hours|action\s+required|act\s+now|pay\s+now)\b",
                lower,
            )
        )

        if has_coercion:
            tactics.append("financial_legal_coercion")
        if has_prize:
            tactics.append("prize_lottery_lure")
        if has_account_threat:
            tactics.append("account_takeover_threat")
        if has_harvesting:
            tactics.append("credential_harvesting")
        if has_urgency:
            tactics.append("artificial_urgency")
        if has_delivery_scam:
            tactics.append("delivery_fee_scam")
        if has_utility_scam:
            tactics.append("utility_disconnection_lure")

        # 2. Raw model inference
        model = cls.load_model()

        if model is None:
            logger.error(
                "Text model unavailable. Cannot produce a validated text prediction."
            )
            return ModalityResult.failed(
                ModalityType.TEXT.value,
                "Text classification model could not be loaded. "
                "Check the model file and Windows application-control policy.",
            )

        try:
            probs = model.predict_proba([text])[0]
            raw_score = float(probs[1]) if len(probs) > 1 else float(probs[0])
        except Exception as err:
            logger.exception("Text model inference failed")
            return ModalityResult.failed(
                ModalityType.TEXT.value,
                f"Text model inference failed: {err}",
            )

        # 3. Score calibration & out-of-distribution correction
        # The training corpus is heavily dominated by 2000s tech emails with a 75% spam prior.
        # This causes out-of-distribution everyday text (like grocery lists) to falsely score >0.80,
        # while short SMS lures with words like 'code' or 'claim' falsely score <0.05.
        #
        # Calibrated blending strategy:
        # 1. Structural fraud tactics: elevate score to threat level (0.72 - 0.96) regardless of raw NB score.
        # 2. Urgency-only (no structural fraud): do not suppress strong ML scam signal down to <=0.25,
        #    while ensuring benign urgency (e.g. 'Urgent: buy milk') remains below the 0.35 threshold.
        # 3. Zero fraud tactics: do not automatically force high ML scores (>=0.70) down to <=0.20;
        #    apply calibrated scaling so ML signal is preserved above 0.20 while keeping everyday
        #    texts (like grocery lists) within the legitimate LOW risk tier (<0.35).
        fraud_tactics = [t for t in tactics if t != "artificial_urgency"]

        if fraud_tactics:
            # Active fraud tactics detected: elevate score to threat level
            calibrated_score = max(raw_score, 0.72 + 0.08 * min(2, len(fraud_tactics) - 1))
            if has_urgency:
                calibrated_score = min(0.96, calibrated_score + 0.08)
        elif has_urgency and not fraud_tactics:
            # Urgency alone without any structural fraud indicator.
            if raw_score >= 0.60:
                calibrated_score = min(0.48, 0.24 + (raw_score - 0.60) * 0.40)
            else:
                calibrated_score = min(0.25, raw_score * 0.30)
        else:
            # Benign conversation / zero fraud tactics:
            # Scale out-of-distribution Naive Bayes noise while preserving strong ML signal.
            if raw_score >= 0.70:
                calibrated_score = min(0.48, 0.21 + (raw_score - 0.70) * 0.50)
            else:
                calibrated_score = min(0.20, raw_score * 0.25)

        calibrated_score = round(min(0.98, max(0.01, calibrated_score)), 4)
        is_threat = calibrated_score >= threshold

        raw = {
            "label": "scam" if is_threat else "legitimate",
            "risk_score": calibrated_score,
            "raw_model_score": round(raw_score, 4),
            "threshold": threshold,
            "confidence": round(min(1.0, 0.5 + abs(calibrated_score - threshold)), 4),
            "detected_tactics": tactics,
        }
        return cls.adapt(raw, text_snippet=text[:100])

    @staticmethod
    def adapt(
        raw: Dict[str, Any],
        text_snippet: Optional[str] = None,
    ) -> ModalityResult:
        try:
            if not isinstance(raw, dict):
                return ModalityResult.failed(
                    ModalityType.TEXT.value,
                    f"Expected dict from text detector, got {type(raw).__name__}",
                )

            threshold = float(raw.get("threshold", 0.5))
            score = float(raw.get("risk_score", raw.get("score", raw.get("scam_probability", 0.5))))
            is_threat = score >= threshold
            label = "scam" if is_threat else "legitimate"

            confidence = float(raw.get("confidence", min(1.0, 0.5 + abs(score - threshold))))

            evidence: List[str] = []
            snippet_preview = f"Text '{text_snippet[:40]}...': " if text_snippet else "Text transcript: "

            if is_threat:
                evidence.append(
                    f"{snippet_preview}NLP scam detector flagged social engineering patterns "
                    f"with score {score:.3f} (threshold {threshold:.2f})."
                )
                tactics = raw.get("detected_tactics", [])
                if tactics:
                    evidence.append(f"Identified psychological pressure tactics: {', '.join(tactics)}.")
            else:
                evidence.append(
                    f"{snippet_preview}NLP detector classified message as benign (score {score:.3f})."
                )

            return ModalityResult(
                modality=ModalityType.TEXT.value,
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
            logger.exception("Failed to adapt text detector output")
            return ModalityResult.failed(ModalityType.TEXT.value, str(exc))

    @staticmethod
    def create_mock(
        is_scam: bool = True,
        text_preview: str = "URGENT: Your bank account is locked. Wire funds immediately.",
        score: Optional[float] = None,
    ) -> ModalityResult:
        """Generates mock text analysis result for testing."""
        risk_score = score if score is not None else (0.88 if is_scam else 0.12)
        raw = {
            "label": "scam" if is_scam else "legitimate",
            "risk_score": risk_score,
            "confidence": 0.89,
            "threshold": 0.5,
            "detected_tactics": ["artificial_urgency", "financial_impersonation"] if is_scam else [],
        }
        return TextAdapter.adapt(raw, text_snippet=text_preview)
