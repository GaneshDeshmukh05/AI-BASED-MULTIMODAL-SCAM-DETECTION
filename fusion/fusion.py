"""fusion.py
Multimodal Decision Fusion Engine for Scam & Deepfake Detection.

Combines heterogeneous detector outputs (audio scam, audio spoof, text, URL,
screenshot, etc.) into an aggregated risk assessment with conflict resolution,
evidence preservation, and defense-in-depth security escalation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Union

from .schemas import (
    DetectorStatus,
    FusionResult,
    ModalityResult,
    ModalityType,
    OverallVerdict,
    RiskLevel,
)

logger = logging.getLogger("fusion_engine")


@dataclass
class FusionConfig:
    """Configurable weights and thresholds for multi-modality fusion."""

    # Domain importance weights: higher weight = greater influence in weighted score
    modality_weights: Dict[str, float] = field(
        default_factory=lambda: {
            ModalityType.AUDIO_SCAM.value: 1.3,   # Direct conversational scam patterns
            ModalityType.URL.value: 1.3,          # Phishing links have direct compromise risk
            ModalityType.SCREENSHOT.value: 1.1,   # Visual mimicry / fake login
            ModalityType.AUDIO_SPOOF.value: 1.1,  # Synthetic voice / deepfake indicator
            ModalityType.TEXT.value: 1.0,         # Phishing keyword / NLP text
            ModalityType.DOMAIN.value: 0.9,       # WHOIS / Domain age
        }
    )

    # Thresholds for threat escalation and categorization
    critical_threshold: float = 0.85
    high_threshold: float = 0.65
    medium_threshold: float = 0.35

    # Conflict detection thresholds:
    conflict_high_risk: float = 0.65
    conflict_low_risk: float = 0.30
    conflict_min_confidence: float = 0.60

    # Escalation policy: If any single reliable detector flags high risk,
    # prevent the overall risk from being averaged down below this floor.
    enable_defense_in_depth: bool = True
    escalation_threat_floor: float = 0.70
    escalation_min_confidence: float = 0.75

    # Expected standard evaluation modalities for coverage scoring
    expected_modalities: List[str] = field(
        default_factory=lambda: [
            ModalityType.AUDIO_SCAM.value,
            ModalityType.AUDIO_SPOOF.value,
            ModalityType.TEXT.value,
            ModalityType.URL.value,
            ModalityType.SCREENSHOT.value,
        ]
    )


class ModelFusion:
    """Combines heterogeneous model outputs into an explainable fusion verdict."""

    def __init__(self, config: Optional[FusionConfig] = None):
        self.config = config or FusionConfig()

    def fuse(
        self,
        results: Mapping[str, Optional[Union[ModalityResult, Dict[str, Any]]]],
    ) -> FusionResult:
        """Fuses a dictionary of modality results into a single FusionResult.

        Args:
            results: Mapping from modality key (e.g. 'audio_scam', 'url') to
                     ModalityResult, raw dict, or None (if omitted).

        Returns:
            FusionResult containing consolidated risk score, verdict, conflict
            analysis, and preserved evidence.
        """
        parsed_results: Dict[str, ModalityResult] = {}

        # 1. Standardize and categorize all input items
        for mod_key, item in results.items():
            if item is None:
                parsed_results[mod_key] = ModalityResult.not_provided(mod_key)
            elif isinstance(item, ModalityResult):
                parsed_results[mod_key] = item
            elif isinstance(item, dict):
                data = dict(item)
                data.setdefault("modality", mod_key)
                parsed_results[mod_key] = ModalityResult(**data)
            else:
                logger.warning("Unrecognized result type for %s: %s", mod_key, type(item))
                parsed_results[mod_key] = ModalityResult.failed(
                    mod_key, f"Invalid result type: {type(item).__name__}"
                )

        # Separate into active (valid successful predictions), missing, and failed
        active: List[ModalityResult] = []
        missing: List[str] = []
        failed: List[str] = []

        for mod_key, res in parsed_results.items():
            if res.status == DetectorStatus.SUCCESS and res.risk_score is not None:
                active.append(res)
            elif res.status == DetectorStatus.FAILED:
                failed.append(mod_key)
            elif res.status in (DetectorStatus.NOT_PROVIDED, DetectorStatus.SKIPPED):
                missing.append(mod_key)
            else:
                # If status is SUCCESS but risk_score was omitted, treat as missing/invalid
                missing.append(mod_key)

        validated_active = [r for r in active if getattr(r, "is_validated", True)]
        unvalidated_active = [r for r in active if not getattr(r, "is_validated", True)]

        # 2. Handle scenario where all models failed or are missing
        if not active:
            evidence = []
            if failed:
                evidence.append(f"Detectors encountered errors in: {', '.join(failed)}.")
            if missing:
                evidence.append(f"No results supplied for: {', '.join(missing)}.")
            if not evidence:
                evidence.append("No detector inputs were provided for evaluation.")

            return FusionResult(
                verdict=OverallVerdict.INSUFFICIENT_DATA,
                risk_level=RiskLevel.INCONCLUSIVE,
                overall_risk_score=0.0,
                overall_confidence=0.0,
                active_modalities=[],
                missing_modalities=missing,
                failed_modalities=failed,
                has_conflict=False,
                conflict_details=None,
                evidence_summary=evidence,
                modality_breakdown=parsed_results,
                metadata={"reason": "no_active_modalities"},
            )

        # 2b. Handle scenario where only unvalidated experimental detectors were active
        if not validated_active:
            evidence_summary: List[str] = []
            for res in sorted(unvalidated_active, key=lambda r: r.risk_score or 0.0, reverse=True):
                for ev in res.evidence:
                    formatted_ev = f"[{res.modality.upper()}] {ev}"
                    if formatted_ev not in evidence_summary:
                        evidence_summary.append(formatted_ev)

            for fail_mod in failed:
                err_msg = parsed_results[fail_mod].error_message or "Unknown error"
                evidence_summary.append(f"[{fail_mod.upper()}] Detector execution failed: {err_msg}")

            if missing:
                evidence_summary.append(
                    f"[SYSTEM] Unexamined modalities (not provided): {', '.join(missing)}. "
                    "Missing modalities are not presumed safe."
                )

            evidence_summary.append(
                "[SYSTEM] Assessment inconclusive: Only unvalidated experimental detector outputs were available. "
                "Insufficient validated operational evidence exists to reach a definitive threat verdict."
            )

            unval_scores = {r.modality: r.risk_score for r in unvalidated_active}
            return FusionResult(
                verdict=OverallVerdict.INSUFFICIENT_DATA,
                risk_level=RiskLevel.INCONCLUSIVE,
                overall_risk_score=0.0,
                overall_confidence=0.0,
                active_modalities=[r.modality for r in active],
                missing_modalities=missing,
                failed_modalities=failed,
                has_conflict=False,
                conflict_details=None,
                evidence_summary=evidence_summary,
                modality_breakdown=parsed_results,
                metadata={
                    "reason": "insufficient_validated_evidence",
                    "unvalidated_modalities": [r.modality for r in unvalidated_active],
                    "unvalidated_scores": unval_scores,
                },
            )

        # 3. Compute weighted risk score and confidence (strictly using validated active results)
        total_weight = 0.0
        weighted_risk_sum = 0.0
        confidence_sum = 0.0

        for res in validated_active:
            conf = res.confidence if res.confidence is not None else 0.5
            weight = self.config.modality_weights.get(res.modality, 1.0)
            # Confidence scales the influence: higher certainty gives stronger weight
            effective_weight = weight * max(0.1, conf)
            weighted_risk_sum += res.risk_score * effective_weight  # type: ignore
            total_weight += effective_weight
            confidence_sum += conf

        base_risk_score = weighted_risk_sum / total_weight if total_weight > 0 else 0.0
        avg_confidence = confidence_sum / len(validated_active)

        # Apply coverage penalty: If few modalities were evaluated out of expected set,
        # discount overall confidence so a partial check cannot give false 100% certainty.
        coverage_ratio = len(validated_active) / max(1, len(self.config.expected_modalities))
        overall_confidence = avg_confidence * (0.6 + 0.4 * min(1.0, coverage_ratio))

        # 4. Check for high-threat escalation (Defense-in-Depth strictly on validated modalities)
        escalated = False
        highest_threat_score = 0.0
        highest_threat_modality = ""

        for res in validated_active:
            score = res.risk_score or 0.0
            if res.is_threat or score >= self.config.high_threshold:
                if score > highest_threat_score:
                    highest_threat_score = score
                    highest_threat_modality = res.modality

        overall_risk_score = base_risk_score

        if self.config.enable_defense_in_depth and highest_threat_score >= self.config.escalation_threat_floor:
            threat_res = next(r for r in validated_active if r.modality == highest_threat_modality)
            threat_conf = threat_res.confidence if threat_res.confidence is not None else 0.5
            if threat_conf >= self.config.escalation_min_confidence:
                if overall_risk_score < (threat_res.risk_score or 0.0):
                    logger.info(
                        "Defense-in-depth triggered: escalating risk from %.3f to %.3f due to %s threat",
                        overall_risk_score,
                        threat_res.risk_score,
                        highest_threat_modality,
                    )
                    overall_risk_score = threat_res.risk_score or 0.0
                    escalated = True

        # 5. Detect Conflicting Signals among validated modalities
        has_conflict = False
        conflict_details: Optional[str] = None

        content_threats = [
            r for r in validated_active
            if r.modality != ModalityType.AUDIO_SPOOF.value
            and (r.risk_score or 0.0) >= self.config.conflict_high_risk
            and (r.confidence or 0.0) >= self.config.conflict_min_confidence
        ]
        content_safes = [
            r for r in validated_active
            if r.modality != ModalityType.AUDIO_SPOOF.value
            and (r.risk_score or 0.0) <= self.config.conflict_low_risk
            and (r.confidence or 0.0) >= self.config.conflict_min_confidence
        ]

        if content_threats and content_safes:
            has_conflict = True
            threat_names = [f"{r.modality} (risk: {r.risk_score:.2f})" for r in content_threats]
            safe_names = [f"{r.modality} (risk: {r.risk_score:.2f})" for r in content_safes]
            conflict_details = (
                f"Contradictory signals: High-risk flagged by [{', '.join(threat_names)}], "
                f"while [{', '.join(safe_names)}] indicated safe."
            )

        # 6. Determine Risk Level
        if overall_risk_score >= self.config.critical_threshold:
            risk_level = RiskLevel.CRITICAL
        elif overall_risk_score >= self.config.high_threshold:
            risk_level = RiskLevel.HIGH
        elif overall_risk_score >= self.config.medium_threshold:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # 7. Determine Overall Verdict with AUDIO_SPOOF Nuance
        only_spoof_threat = (
            any(r.modality == ModalityType.AUDIO_SPOOF.value and r.is_threat for r in validated_active)
            and not any(r.modality != ModalityType.AUDIO_SPOOF.value and r.is_threat for r in validated_active)
        )

        if has_conflict:
            verdict = OverallVerdict.CONFLICT_WARNING
        elif only_spoof_threat and not any((r.risk_score or 0.0) >= self.config.high_threshold for r in validated_active if r.modality != ModalityType.AUDIO_SPOOF.value):
            verdict = OverallVerdict.SUSPICIOUS
        elif risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
            verdict = OverallVerdict.SCAM
        elif risk_level == RiskLevel.MEDIUM:
            verdict = OverallVerdict.SUSPICIOUS
        else:
            verdict = OverallVerdict.SAFE

        # 8. Consolidate and preserve all evidence
        evidence_summary: List[str] = []

        # Active validated models: sorted by risk score
        for res in sorted(validated_active, key=lambda r: r.risk_score or 0.0, reverse=True):
            for ev in res.evidence:
                formatted_ev = f"[{res.modality.upper()}] {ev}"
                if formatted_ev not in evidence_summary:
                    evidence_summary.append(formatted_ev)

        # Unvalidated models: evidence preserved with research advisory
        for res in sorted(unvalidated_active, key=lambda r: r.risk_score or 0.0, reverse=True):
            for ev in res.evidence:
                formatted_ev = f"[{res.modality.upper()}] {ev}"
                if formatted_ev not in evidence_summary:
                    evidence_summary.append(formatted_ev)
            evidence_summary.append(
                f"[SYSTEM] Notice: {res.modality.upper()} output (score {res.risk_score:.3f}) was recorded "
                "for research/evaluation but excluded from operational threat scoring (unvalidated model)."
            )

        # Append failure notices if any detectors crashed
        for fail_mod in failed:
            err_msg = parsed_results[fail_mod].error_message or "Unknown error"
            evidence_summary.append(f"[{fail_mod.upper()}] Detector execution failed: {err_msg}")

        # Missing data notice
        if missing:
            evidence_summary.append(
                f"[SYSTEM] Unexamined modalities (not provided): {', '.join(missing)}. "
                "Missing modalities are not presumed safe."
            )

        return FusionResult(
            verdict=verdict,
            risk_level=risk_level,
            overall_risk_score=round(overall_risk_score, 4),
            overall_confidence=round(overall_confidence, 4),
            active_modalities=[r.modality for r in active],
            missing_modalities=missing,
            failed_modalities=failed,
            has_conflict=has_conflict,
            conflict_details=conflict_details,
            evidence_summary=evidence_summary,
            modality_breakdown=parsed_results,
            metadata={
                "base_risk_score": round(base_risk_score, 4),
                "defense_in_depth_escalated": escalated,
                "only_spoof_threat": only_spoof_threat,
                "coverage_ratio": round(coverage_ratio, 2),
                "unvalidated_modalities": [r.modality for r in unvalidated_active],
                "unvalidated_scores": {r.modality: r.risk_score for r in unvalidated_active},
            },
        )
