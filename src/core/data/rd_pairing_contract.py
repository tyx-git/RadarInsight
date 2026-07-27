from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


PAIRING_CONTRACT_VERSION = "expert1_rd_tr_pairing_v1"
FUSION_ELIGIBLE_SYNC_STATUSES = frozenset({"exact_time_sync", "within_tolerance"})
PROHIBITED_SYNC_STATUSES = frozenset({"proportional_fallback", "time_axis_unavailable", "summary_rd", "unavailable"})


@dataclass(frozen=True)
class RDPairingPolicy:
    min_window_coverage_ratio: float = 0.80
    min_continuous_rd_seconds: float = 1.00
    max_frame_gap_seconds: float = 0.10
    require_explicit_target_association: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RDPairingDecision:
    decision: str
    fusion_eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decision": self.decision,
            "fusion_eligible": self.fusion_eligible,
            "reasons": list(self.reasons),
        }


def _text(value: object) -> str:
    return str(value if value is not None else "").strip()


def _finite_float(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and result not in {float("inf"), float("-inf")} else None


def _positive_duration(start: object, end: object, fallback: object) -> float | None:
    start_value = _finite_float(start)
    end_value = _finite_float(end)
    if start_value is not None and end_value is not None and end_value >= start_value:
        return end_value - start_value
    return _finite_float(fallback)


def evaluate_rd_tr_pairing(
    record: Mapping[str, Any],
    policy: RDPairingPolicy | None = None,
) -> RDPairingDecision:
    """Classify one RD/TR pairing without using provenance or labels as model features."""
    active_policy = policy or RDPairingPolicy()
    reasons: list[str] = []
    sync_status = _text(record.get("rd_time_sync_status") or record.get("rd_realtime_sync_status")).lower()
    parent_id = _text(record.get("parent_sample_id"))
    rd_target_id = _text(record.get("rd_target_association_id"))
    tr_target_id = _text(record.get("tr_target_association_id"))
    target_association_verified = bool(record.get("target_association_verified", False))
    if not target_association_verified:
        target_association_verified = bool(parent_id and rd_target_id and tr_target_id and rd_target_id == tr_target_id)

    if not parent_id:
        reasons.append("missing_parent_association")
    if active_policy.require_explicit_target_association and not target_association_verified:
        reasons.append("target_association_unverified")
    if sync_status in PROHIBITED_SYNC_STATUSES:
        reasons.append(f"prohibited_sync_status:{sync_status or 'missing'}")
    elif sync_status not in FUSION_ELIGIBLE_SYNC_STATUSES:
        reasons.append(f"unverified_sync_status:{sync_status or 'missing'}")

    coverage_ratio = _finite_float(record.get("rd_time_coverage_ratio"))
    if coverage_ratio is None:
        coverage_ratio = _finite_float(record.get("rd_coverage_ratio"))
    if coverage_ratio is None or coverage_ratio < active_policy.min_window_coverage_ratio:
        reasons.append("insufficient_window_coverage")

    continuous_seconds = _positive_duration(
        record.get("rd_time_start_seconds"),
        record.get("rd_time_end_seconds"),
        record.get("segment_duration_seconds"),
    )
    frame_gap_seconds = _finite_float(record.get("rd_frame_gap_p95_seconds"))
    if frame_gap_seconds is not None and frame_gap_seconds > active_policy.max_frame_gap_seconds:
        reasons.append("frame_gap_exceeds_limit")

    if reasons:
        return RDPairingDecision("rejected", False, tuple(sorted(set(reasons))))
    if continuous_seconds is None or continuous_seconds < active_policy.min_continuous_rd_seconds:
        return RDPairingDecision("diagnostic_only", False, ("continuous_rd_too_short",))
    return RDPairingDecision("fusion_eligible", True, ())


def pairing_contract_payload(policy: RDPairingPolicy | None = None) -> dict[str, object]:
    active_policy = policy or RDPairingPolicy()
    return {
        "schema_version": PAIRING_CONTRACT_VERSION,
        "purpose": "Strict RD/TR fusion eligibility; proportional time mapping is prohibited.",
        "policy": active_policy.to_dict(),
        "required_metadata": [
            "parent_sample_id",
            "rd_target_association_id",
            "tr_target_association_id",
            "target_association_verified",
            "rd_time_sync_status",
            "rd_time_start_seconds",
            "rd_time_end_seconds",
            "rd_time_coverage_ratio",
            "rd_frame_gap_p95_seconds",
            "rd_quality_flags",
        ],
        "fusion_eligible_sync_statuses": sorted(FUSION_ELIGIBLE_SYNC_STATUSES),
        "prohibited_sync_statuses": sorted(PROHIBITED_SYNC_STATUSES),
        "decisions": {
            "fusion_eligible": "All target, clock, coverage, continuity, and frame-gap checks pass.",
            "diagnostic_only": "Clock and association pass, but RD duration is too short for fusion.",
            "rejected": "Missing or unsafe evidence; RD must not train or influence fusion.",
        },
        "forbidden_model_inputs": [
            "source_name",
            "file_path",
            "sample_id",
            "parent_sample_id",
            "split_name",
            "label_name",
            "timestamp",
        ],
    }


__all__ = [
    "FUSION_ELIGIBLE_SYNC_STATUSES",
    "PAIRING_CONTRACT_VERSION",
    "PROHIBITED_SYNC_STATUSES",
    "RDPairingDecision",
    "RDPairingPolicy",
    "evaluate_rd_tr_pairing",
    "pairing_contract_payload",
]
