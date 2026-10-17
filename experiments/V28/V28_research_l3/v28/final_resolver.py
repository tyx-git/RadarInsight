"""Research-only V28 L3 action resolver.

This resolver mirrors the product action contract while allowing an explicitly
marked exploratory registry to be replayed on its fitting role.  It never
turns an unavailable production registry into a deployment-ready decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from v28.protocol import V28_L2_CLASSES


L1_CLASSES = frozenset(("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE"))


@dataclass(frozen=True)
class ResearchDecision:
    action: str
    reason_code: str
    details: dict[str, Any]


def resolve_research_action(
    *,
    registry: Mapping[str, Any],
    l1_candidate_set: Sequence[str],
    l1_prediction: str | None,
    l1_knownness: bool,
    l1_quality_eligible: bool,
    l1_target: bool,
    l1_clutter: bool,
    l2_candidate_set: Sequence[str],
    l2_prediction: str | None,
    l2_knownness: bool | None,
    l2_quality_eligible: bool | None,
    streaming: bool = False,
) -> ResearchDecision:
    """Apply the strict priority order to one frozen endpoint result."""

    l1_candidates = list(l1_candidate_set)
    l2_candidates = list(l2_candidate_set)
    details: dict[str, Any] = {
        "research_only": True,
        "registry_status": registry.get("status"),
        "registry_deployment_eligible": bool(registry.get("deployment_eligible")),
        "l1_candidates": l1_candidates,
        "l2_candidates": l2_candidates,
        "l1_prediction": l1_prediction,
        "l2_prediction": l2_prediction,
    }
    if not l1_quality_eligible:
        return ResearchDecision("PENDING_ABSTAIN", "LOW_QUALITY_OR_IMMATURE", details)
    if registry.get("status") != "internal_exploratory" or registry.get("research_only") is not True:
        return ResearchDecision("PENDING_ABSTAIN", "REGISTRY_INVALID_OR_UNAVAILABLE", details)
    if any(value not in L1_CLASSES for value in l1_candidates):
        return ResearchDecision("PENDING_ABSTAIN", "CANDIDATE_SCHEMA_INVALID", details)
    if len(l1_candidates) != 1 or l1_prediction not in l1_candidates:
        return ResearchDecision("PENDING_ABSTAIN", "L1_ROUTE_CONFLICT", details)
    predicted_target = l1_prediction == "KNOWN_TARGET_EVIDENCE"
    predicted_clutter = l1_prediction == "KNOWN_CLUTTER_EVIDENCE"
    if (bool(l1_target), bool(l1_clutter)) != (predicted_target, predicted_clutter):
        return ResearchDecision("PENDING_ABSTAIN", "L1_ROUTE_CONFLICT", details)
    if not l1_knownness:
        return ResearchDecision("UNKNOWN_REJECTED", "OOD_NONCONFORMING", details)
    if l1_target and l1_clutter:
        return ResearchDecision("PENDING_ABSTAIN", "L1_ROUTE_CONFLICT", details)
    if l1_clutter:
        return ResearchDecision("Clutter", "L1_CLUTTER_ACCEPTED", details)
    if not l1_target:
        return ResearchDecision("PENDING_ABSTAIN", "L1_ROUTE_CONFLICT", details)
    if l2_quality_eligible is not True:
        return ResearchDecision("PENDING_ABSTAIN", "LOW_QUALITY_OR_IMMATURE", details)
    if l2_knownness is not True:
        return ResearchDecision("UNKNOWN_REJECTED", "OOD_NONCONFORMING", details)
    if len(l2_candidates) == 1 and l2_prediction in l2_candidates and l2_prediction in V28_L2_CLASSES:
        return ResearchDecision(str(l2_prediction), "KNOWN_DBB_CLASS_ACCEPTED", details)
    if len(l2_candidates) > 1:
        return ResearchDecision("PENDING_ABSTAIN" if streaming else "UNKNOWN_REJECTED", "KNOWN_CLASS_AMBIGUOUS", details)
    return ResearchDecision("UNKNOWN_REJECTED", "OOD_NONCONFORMING", details)


__all__ = ["ResearchDecision", "resolve_research_action"]
