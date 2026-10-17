"""Canonical strict-open-set action semantics for the current product.

Historical experiment modules are no longer product entry points.  The
current topology is L1 KnownGate, L2 DBB specialist, and L3 semantic decision;
Unknown is an action rather than a learned L1/L2 class.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


ARCHITECTURE_SCHEMA = "RADAR-LN3-V21-strict-open-set-architecture-v1"
KNOWN_GATE_TASK_NAME = "known_gate"
DBB_SPECIALIST_TASK_NAME = "dbb_specialist"


class KnownGateClass(StrEnum):
    KNOWN_TARGET_EVIDENCE = "KNOWN_TARGET_EVIDENCE"
    KNOWN_CLUTTER_EVIDENCE = "KNOWN_CLUTTER_EVIDENCE"


class ProductAction(StrEnum):
    DRONE = "DroneTarget"
    BIRD = "BirdTarget"
    BALLOON = "BalloonTarget"
    CLUTTER = "ClutterTarget"
    UNKNOWN_REJECTED = "UNKNOWN_REJECTED"
    PENDING_ABSTAIN = "PENDING_ABSTAIN"


DBB_ACTIONS = frozenset(
    {ProductAction.DRONE, ProductAction.BIRD, ProductAction.BALLOON}
)


@dataclass(frozen=True)
class SemanticDecision:
    action: ProductAction
    reason: str
    observation_eligible: bool
    known_target_accepted: bool
    known_clutter_accepted: bool
    dbb_subclass_accepted: bool


def resolve_semantic_action(
    *,
    observation_eligible: bool,
    known_target_accepted: bool,
    known_clutter_accepted: bool,
    dbb_subclass: ProductAction | None = None,
    dbb_subclass_accepted: bool = False,
) -> SemanticDecision:
    """Resolve one mutually exclusive product action without an Unknown class.

    Threshold fitting is intentionally outside this function.  Callers pass
    already-calibrated acceptance booleans and receive an auditable action.
    """

    eligible = bool(observation_eligible)
    target = bool(known_target_accepted)
    clutter = bool(known_clutter_accepted)
    subclass_ok = bool(dbb_subclass_accepted)
    if dbb_subclass is not None and dbb_subclass not in DBB_ACTIONS:
        raise ValueError("dbb_subclass must be DroneTarget, BirdTarget or BalloonTarget")
    if subclass_ok and dbb_subclass is None:
        raise ValueError("an accepted DBB subclass requires an explicit DBB action")
    if not eligible:
        return SemanticDecision(
            ProductAction.PENDING_ABSTAIN,
            "insufficient_observation_or_quality",
            eligible,
            target,
            clutter,
            subclass_ok,
        )
    if target and clutter:
        return SemanticDecision(
            ProductAction.PENDING_ABSTAIN,
            "known_gate_acceptance_conflict",
            eligible,
            target,
            clutter,
            subclass_ok,
        )
    if subclass_ok and not target:
        return SemanticDecision(
            ProductAction.PENDING_ABSTAIN,
            "dbb_subclass_accepted_without_known_target_route",
            eligible,
            target,
            clutter,
            subclass_ok,
        )
    if clutter:
        return SemanticDecision(
            ProductAction.CLUTTER,
            "known_clutter_evidence_accepted",
            eligible,
            target,
            clutter,
            subclass_ok,
        )
    if target and subclass_ok:
        assert dbb_subclass is not None
        return SemanticDecision(
            dbb_subclass,
            "known_target_and_dbb_subclass_evidence_accepted",
            eligible,
            target,
            clutter,
            subclass_ok,
        )
    return SemanticDecision(
        ProductAction.UNKNOWN_REJECTED,
        (
            "dbb_subclass_evidence_insufficient"
            if target
            else "known_target_and_known_clutter_not_accepted"
        ),
        eligible,
        target,
        clutter,
        subclass_ok,
    )


__all__ = ["ARCHITECTURE_SCHEMA", "DBB_ACTIONS", "DBB_SPECIALIST_TASK_NAME", "KNOWN_GATE_TASK_NAME", "KnownGateClass", "ProductAction", "SemanticDecision", "resolve_semantic_action"]