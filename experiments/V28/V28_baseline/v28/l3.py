"""Fail-closed V28 L3 action resolver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .protocol import V28_L2_CLASSES


@dataclass(frozen=True)
class V28Decision:
    action: str
    reason_code: str
    details: dict[str, Any]


def resolve_action_verified(*, registry: Mapping[str, Any], l1_candidate_set: Sequence[str], l1_prediction: str | None, l1_knownness: bool | None, l1_quality_eligible: bool, l1_target: bool, l1_clutter: bool, l2_candidate_set: Sequence[str], l2_prediction: str | None, l2_knownness: bool | None, l2_quality_eligible: bool | None, streaming: bool, audit_mode: bool = False) -> V28Decision:
    common = {"l1_candidates": list(l1_candidate_set), "l2_candidates": list(l2_candidate_set), "l1_prediction": l1_prediction, "l2_prediction": l2_prediction}
    if not l1_quality_eligible:
        return V28Decision("PENDING_ABSTAIN", "LOW_QUALITY_OR_IMMATURE", common)
    common = {**common, "registry_status": registry.get("status")}
    registry_ready = registry.get("status") == "fitted" and registry.get("deployment_eligible") is True
    if not registry_ready and not audit_mode:
        return V28Decision("PENDING_ABSTAIN", "CALIBRATION_NOT_FITTED", common)
    if audit_mode:
        audit = registry.get("calibration", {}).get("post_selection_audit", {})
        if audit.get("status") != "registered" or audit.get("threshold_fitting_allowed") is not False:
            return V28Decision("PENDING_ABSTAIN", "AUDIT_POOL_INVALID", common)
        common = {**common, "audit_mode": True, "registry_deployment_eligible": bool(registry.get("deployment_eligible"))}
    if set(l1_candidate_set) - {"KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE"} or set(l2_candidate_set) - set(V28_L2_CLASSES):
        return V28Decision("PENDING_ABSTAIN", "CANDIDATE_SCHEMA_INVALID", common)
    if not l1_candidate_set or len(l1_candidate_set) != 1 or l1_prediction not in l1_candidate_set:
        return V28Decision("PENDING_ABSTAIN", "L1_ROUTE_CONFLICT", common)
    if l1_knownness is not True:
        return V28Decision("UNKNOWN_REJECTED", "OOD_NONCONFORMING", common)
    if l1_target and l1_clutter:
        return V28Decision("PENDING_ABSTAIN", "L1_ROUTE_CONFLICT", common)
    if l1_clutter:
        return V28Decision("Clutter", "L1_CLUTTER_ACCEPTED", common)
    if not l1_target:
        return V28Decision("UNKNOWN_REJECTED", "OOD_NONCONFORMING", common)
    if l2_quality_eligible is not True:
        return V28Decision("PENDING_ABSTAIN", "LOW_QUALITY_OR_IMMATURE", common)
    if l2_knownness is not True:
        return V28Decision("UNKNOWN_REJECTED", "OOD_NONCONFORMING", common)
    if len(l2_candidate_set) == 1 and l2_prediction in l2_candidate_set and l2_prediction in V28_L2_CLASSES:
        return V28Decision(str(l2_prediction), "KNOWN_DBB_CLASS_ACCEPTED", common)
    if len(l2_candidate_set) > 1:
        return V28Decision("PENDING_ABSTAIN" if streaming else "UNKNOWN_REJECTED", "KNOWN_CLASS_AMBIGUOUS", common)
    return V28Decision("UNKNOWN_REJECTED", "OOD_NONCONFORMING", common)




__all__ = ["V28Decision", "resolve_action_verified"]