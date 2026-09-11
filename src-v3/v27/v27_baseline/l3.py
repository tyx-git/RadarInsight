"""Fail-closed V27 L3 action contract and registry validation.

L3 is deliberately a semantic boundary.  It consumes calibrated verdicts and
prediction sets produced by :mod:`v27.open_set`; it never turns a raw softmax,
distance, or uncalibrated score into a product action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from radar_ln3.runtime import canonical_sha256

from .protocol import V27_L1_TASK_NAME, V27_L2_CLASS_NAMES, V27_L2_TASK_NAME


L3_SCHEMA_VERSION = "RADAR-LN3-V27-l3-calibrated-action-v3"
ACTION_SELECTION_METHOD = (
    "joint_l1_l2_l3_action_replay_independent_conformal_grid_v2"
)
L1_CLASS_NAMES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")


class V27Action(StrEnum):
    DRONE = "DroneTarget"
    BIRD = "BirdTarget"
    BALLOON = "BalloonTarget"
    CLUTTER = "ClutterTarget"
    UNKNOWN_REJECTED = "UNKNOWN_REJECTED"
    PENDING_ABSTAIN = "PENDING_ABSTAIN"


_DBB_TO_ACTION = {
    "DroneTarget": V27Action.DRONE,
    "BirdTarget": V27Action.BIRD,
    "BalloonTarget": V27Action.BALLOON,
}


@dataclass(frozen=True)
class V27L3Decision:
    action: V27Action
    reason_code: str
    calibration_status: str
    l1_target_accepted: bool
    l1_clutter_accepted: bool
    l1_knownness_accepted: bool | None
    l2_knownness_accepted: bool | None
    l1_deterministic_quality_eligible: bool
    l1_learned_quality_eligible: bool | None
    l1_learned_quality_gate_active: bool | None
    l2_deterministic_quality_eligible: bool | None
    l2_learned_quality_eligible: bool | None
    l2_learned_quality_gate_active: bool | None
    l1_quality_eligible: bool
    l2_quality_eligible: bool | None
    learned_quality_eligible: bool | None
    quality_eligible: bool
    dbb_candidates: tuple[str, ...]


def l3_contract() -> dict[str, Any]:
    """Return the immutable product action contract."""

    return {
        "schema_version": L3_SCHEMA_VERSION,
        "raw_score_policy": "forbidden; resolver accepts calibrated booleans and conformal candidate set only",
        "calibration_policy": "dynamic_registry_required; unfitted_registry_fails_closed",
        "action_order": [
            "LOW_QUALITY_OR_IMMATURE",
            "CALIBRATION_NOT_FITTED",
            "L1_ROUTE_CONFLICT",
            "OOD_NONCONFORMING",
            "L1_CLUTTER_ACCEPTED",
            "KNOWN_DBB_CLASS_ACCEPTED",
            "KNOWN_CLASS_AMBIGUOUS",
        ],
        "unknown_is_model_class": False,
        "single_distance_gate_allowed": False,
        "fixed_softmax_threshold_allowed": False,
        "learned_quality_policy": (
            "explicit_gate_status_required_for_fitted_none_verdict; "
            "inactive_gate_is_advisory_only"
        ),
    }


def _verify_digest(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"V27 L3 {field} must be a SHA-256 digest")
    normalized = value.lower()
    if any(character not in "0123456789abcdef" for character in normalized):
        raise ValueError(f"V27 L3 {field} must be a SHA-256 digest")
    return normalized


def _finite(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27 L3 {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"V27 L3 {field} must be finite")
    return number


def _probability(value: Any, *, field: str) -> float:
    number = _finite(value, field=field)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"V27 L3 {field} must be within [0, 1]")
    return number


def make_unfitted_registry(*, system_manifest_sha256: str) -> dict[str, Any]:
    """Create an explicit placeholder that cannot emit an acceptance."""

    payload = {
        "schema_version": L3_SCHEMA_VERSION,
        "status": "unfitted",
        "system_manifest_sha256": _verify_digest(
            system_manifest_sha256, field="system_manifest_sha256"
        ),
        "l1_task": V27_L1_TASK_NAME,
        "l2_task": V27_L2_TASK_NAME,
        "class_names": list(V27_L2_CLASS_NAMES),
        "contract": l3_contract(),
        "fitted_scores": None,
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


def make_fitted_registry(
    *,
    system_manifest_sha256: str,
    fitted_scores: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a fitted registry after validating its score payload."""

    if not isinstance(fitted_scores, Mapping):
        raise TypeError("V27 fitted L3 scores must be an object")
    scores = dict(fitted_scores)
    manifest_sha256 = _verify_digest(
        system_manifest_sha256, field="system_manifest_sha256"
    )
    _verify_fitted_scores(
        scores,
        expected_system_manifest_sha256=manifest_sha256,
    )
    payload = {
        "schema_version": L3_SCHEMA_VERSION,
        "status": "fitted",
        "system_manifest_sha256": manifest_sha256,
        "l1_task": V27_L1_TASK_NAME,
        "l2_task": V27_L2_TASK_NAME,
        "class_names": list(V27_L2_CLASS_NAMES),
        "contract": l3_contract(),
        "fitted_scores": scores,
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


def _verify_task_section(
    section: Mapping[str, Any],
    *,
    task: str,
) -> None:
    classes = list(V27_L2_CLASS_NAMES if task == "l2" else L1_CLASS_NAMES)
    if list(section.get("class_names") or ()) != classes:
        raise ValueError(f"V27 L3 fitted {task} class order differs")
    temperature = _finite(section.get("temperature"), field=f"{task}.temperature")
    if temperature <= 0.0:
        raise ValueError(f"V27 L3 fitted {task} temperature must be positive")
    _probability(
        section.get("conformal_error_rate"),
        field=f"{task}.conformal_error_rate",
    )
    method = section.get("conformal_method")
    if method != "class_conditional_split_conformal_1_minus_calibrated_probability_higher":
        raise ValueError(f"V27 L3 fitted {task} conformal method differs")
    quantiles = section.get("conformal_quantile_by_class")
    floors = section.get("conformal_candidate_probability_floor_by_class")
    supports = section.get("conformal_support_by_class")
    coverages = section.get("conformal_coverage_on_calibration_by_class")
    if not all(
        isinstance(value, Mapping)
        for value in (quantiles, floors, supports, coverages)
    ):
        raise ValueError(f"V27 L3 fitted {task} class-conditional conformal fields are missing")
    expected_names = set(classes)
    if (
        set(quantiles) != expected_names
        or set(floors) != expected_names
        or set(supports) != expected_names
        or set(coverages) != expected_names
    ):
        raise ValueError(f"V27 L3 fitted {task} class-conditional conformal classes differ")
    for class_name in classes:
        quantile = _probability(
            quantiles[class_name],
            field=f"{task}.conformal_quantile_by_class.{class_name}",
        )
        floor = _probability(
            floors[class_name],
            field=(
                f"{task}.conformal_candidate_probability_floor_by_class."
                f"{class_name}"
            ),
        )
        if not math.isclose(floor, 1.0 - quantile, rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"V27 L3 fitted {task} class-conditional conformal floor differs")
        if int(supports[class_name]) <= 0:
            raise ValueError(f"V27 L3 fitted {task} class-conditional conformal support is invalid")
        _probability(
            coverages[class_name],
            field=f"{task}.conformal_coverage_on_calibration_by_class.{class_name}",
        )
    coverage = _probability(
        section.get("conformal_coverage_on_calibration"),
        field=f"{task}.conformal_coverage_on_calibration",
    )
    threshold = _probability(
        section.get("knownness_threshold"),
        field=f"{task}.knownness_threshold",
    )
    if section.get("knownness_score_direction") != "higher_is_known":
        raise ValueError(f"V27 L3 fitted {task} knownness direction differs")
    if section.get("knownness_threshold_space") != "sigmoid_probability":
        raise ValueError(f"V27 L3 fitted {task} knownness space differs")
    if int(section.get("knownness_positive_support", 0)) <= 0 or int(
        section.get("knownness_negative_support", 0)
    ) <= 0:
        raise ValueError(f"V27 L3 fitted {task} knownness support is invalid")
    quality_threshold = _probability(
        section.get("quality_threshold"),
        field=f"{task}.quality_threshold",
    )
    if section.get("quality_score_direction") != "higher_is_eligible":
        raise ValueError(f"V27 L3 fitted {task} quality direction differs")
    if int(section.get("quality_positive_support", 0)) <= 0 or int(
        section.get("quality_negative_support", 0)
    ) <= 0:
        raise ValueError(f"V27 L3 fitted {task} quality support is invalid")
    conformal_support = int(section.get("conformal_support", 0))
    if conformal_support <= 0 or conformal_support != sum(
        int(supports[class_name]) for class_name in classes
    ):
        raise ValueError(f"V27 L3 fitted {task} conformal support is invalid")
    quality_gate = section.get("quality_gate")
    if not isinstance(quality_gate, Mapping):
        raise ValueError(f"V27 L3 fitted {task} quality-gate policy is missing")
    quality_gate_active = quality_gate.get("active")
    required_negative_support = int(
        quality_gate.get("required_negative_support", 0)
    )
    observed_negative_support = int(
        quality_gate.get("observed_negative_support", -1)
    )
    if (
        not isinstance(quality_gate_active, bool)
        or required_negative_support <= 0
        or observed_negative_support != int(section.get("quality_negative_support", -2))
    ):
        raise ValueError(f"V27 L3 fitted {task} quality-gate support policy differs")
    if quality_gate_active:
        if observed_negative_support < required_negative_support or quality_gate.get(
            "status"
        ) != "active":
            raise ValueError(f"V27 L3 fitted {task} active quality gate is unsupported")
    elif (
        observed_negative_support >= required_negative_support
        or quality_gate.get("status")
        != "inactive_insufficient_final_negative_support"
    ):
        raise ValueError(f"V27 L3 fitted {task} inactive quality gate policy differs")
    # Keep these assignments explicit: they make malformed values fail before
    # the registry is accepted, and document that both thresholds are used.
    if (
        not math.isfinite(threshold)
        or not math.isfinite(quality_threshold)
        or not math.isfinite(coverage)
    ):
        raise ValueError(f"V27 L3 fitted {task} thresholds are invalid")


def _verify_fitted_scores(
    scores: Mapping[str, Any],
    *,
    expected_system_manifest_sha256: str,
) -> None:
    """Validate provenance, score directions, thresholds and supports."""

    if scores.get("schema_version") != "RADAR-LN3-V27-dynamic-calibration-v3":
        raise ValueError("V27 L3 fitted score schema differs")
    if scores.get("status") != "fitted":
        raise ValueError("V27 L3 fitted score status differs")
    if scores.get("experiment_id") != "V27":
        raise ValueError("V27 L3 fitted score experiment differs")
    if scores.get("system_manifest_sha256") != expected_system_manifest_sha256:
        raise ValueError("V27 L3 fitted score manifest binding differs")
    _verify_digest(
        scores.get("frozen_base_binding_sha256"),
        field="frozen_base_binding_sha256",
    )
    _verify_digest(
        scores.get("calibration_pool_registry_sha256"),
        field="calibration_pool_registry_sha256",
    )
    _verify_digest(
        scores.get("representation_cache_manifest_sha256"),
        field="representation_cache_manifest_sha256",
    )
    score_contract_digest = _verify_digest(
        scores.get("score_contract_sha256"), field="score_contract_sha256"
    )
    method = scores.get("fitting_method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError("V27 L3 fitting_method must be non-empty")
    contract = scores.get("score_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("V27 L3 fitted score contract is missing")
    if canonical_sha256(dict(contract)) != score_contract_digest:
        raise ValueError("V27 L3 fitted score contract hash differs")
    if contract.get("schema_version") != scores.get("schema_version"):
        raise ValueError("V27 L3 fitted score contract schema differs")
    if contract.get("fixed_softmax_threshold") is not False or contract.get(
        "single_distance_gate"
    ) is not False:
        raise ValueError("V27 L3 fitted score contract enables a forbidden gate")
    if contract.get("quality_label_policy") != "causal_deterministic_facts_v1":
        raise ValueError("V27 L3 fitted quality-label policy differs")
    if (
        contract.get("classifier_probability_set")
        != "dynamic_temperature_class_conditional_conformal_probability_set"
    ):
        raise ValueError("V27 L3 fitted probability-set contract differs")
    action_selection = scores.get("action_selection")
    if not isinstance(action_selection, Mapping):
        raise ValueError("V27 L3 fitted action-selection record is missing")
    if action_selection.get("method") != ACTION_SELECTION_METHOD:
        raise ValueError("V27 L3 fitted action-selection method differs")
    for field in ("known_target_action_floor", "clutter_action_floor"):
        floor = _probability(action_selection.get(field), field=f"action_selection.{field}")
        if floor <= 0.0:
            raise ValueError("V27 L3 fitted action-selection floor is invalid")
    error_rate_grids: dict[str, list[float]] = {}
    for task in ("l1", "l2"):
        field = f"{task}_conformal_error_rate_grid"
        grid = action_selection.get(field)
        if (
            not isinstance(grid, list)
            or not grid
            or any(
                not 0.0 < _probability(value, field=f"action_selection.{field}") < 1.0
                for value in grid
            )
            or grid != sorted(set(grid))
        ):
            raise ValueError("V27 L3 fitted action-selection grid differs")
        error_rate_grids[task] = [float(value) for value in grid]
    if int(action_selection.get("candidates_considered", 0)) <= 0 or int(
        action_selection.get("feasible_candidate_count", 0)
    ) <= 0:
        raise ValueError("V27 L3 fitted action-selection has no feasible evidence")
    if action_selection.get("low_quality_pending_counted_as_unknown_rejection") is not False:
        raise ValueError("V27 L3 fitted action-selection conflates Pending with OOD")
    selected = action_selection.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("V27 L3 fitted action-selection result is missing")
    for task in ("l1", "l2"):
        field = f"{task}_conformal_error_rate"
        selected_error_rate = _probability(
            selected.get(field), field=f"action_selection.selected.{field}"
        )
        section_error_rate = _probability(
            scores[task].get("conformal_error_rate"),
            field=f"{task}.conformal_error_rate",
        )
        if (
            not math.isclose(
                selected_error_rate,
                section_error_rate,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
            or not any(
                math.isclose(
                    selected_error_rate,
                    candidate,
                    rel_tol=0.0,
                    abs_tol=1.0e-12,
                )
                for candidate in error_rate_grids[task]
            )
        ):
            raise ValueError("V27 L3 fitted action-selection conformal binding differs")
    for task, field in (("l1", "l1_knownness_threshold"), ("l2", "l2_knownness_threshold")):
        if not math.isclose(
            _probability(
                selected.get(field), field=f"action_selection.selected.{field}"
            ),
            _probability(
                scores[task].get("knownness_threshold"),
                field=f"{task}.knownness_threshold",
            ),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("V27 L3 fitted action-selection knownness binding differs")
    metrics = selected.get("action_metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("V27 L3 fitted action-selection metrics are missing")
    for field in (
        "all_known_target_parent_count",
        "eligible_known_target_support",
        "all_clutter_parent_count",
        "eligible_clutter_support",
        "unknown_parent_count",
    ):
        if int(metrics.get(field, 0)) <= 0:
            raise ValueError("V27 L3 fitted action-selection support is invalid")
    for all_field, eligible_field, pending_field in (
        (
            "all_known_target_parent_count",
            "eligible_known_target_support",
            "known_target_deterministic_quality_pending_count",
        ),
        (
            "all_clutter_parent_count",
            "eligible_clutter_support",
            "clutter_deterministic_quality_pending_count",
        ),
    ):
        all_count = int(metrics[all_field])
        eligible_count = int(metrics[eligible_field])
        pending_count = int(metrics.get(pending_field, -1))
        if (
            eligible_count > all_count
            or pending_count < 0
            or eligible_count + pending_count != all_count
        ):
            raise ValueError("V27 L3 fitted action-selection quality accounting differs")
    known_target_recall = _probability(
        metrics.get("eligible_known_target_dbb_route_recall"),
        field="action_selection.selected.eligible_known_target_dbb_route_recall",
    )
    clutter_recall = _probability(
        metrics.get("eligible_clutter_recall"),
        field="action_selection.selected.eligible_clutter_recall",
    )
    if (
        known_target_recall + 1.0e-12
        < float(action_selection["known_target_action_floor"])
        or clutter_recall + 1.0e-12
        < float(action_selection["clutter_action_floor"])
    ):
        raise ValueError("V27 L3 fitted action-selection violates a known safety floor")
    for task in ("l1", "l2"):
        section = scores.get(task)
        if not isinstance(section, Mapping):
            raise ValueError(f"V27 L3 fitted {task} section is missing")
        _verify_task_section(section, task=task)
    quality = scores.get("quality")
    if not isinstance(quality, Mapping):
        raise ValueError("V27 L3 fitted quality section is missing")
    if quality.get("threshold_space") != "sigmoid_probability" or quality.get(
        "score_direction"
    ) != "higher_is_eligible":
        raise ValueError("V27 L3 fitted quality score contract differs")
    thresholds = quality.get("task_specific_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("V27 L3 fitted quality task thresholds are missing")
    for task in ("known_gate", "dbb_specialist"):
        _probability(thresholds.get(task), field=f"quality.{task}.threshold")
    active_task_gates = quality.get("task_specific_gate_active")
    if not isinstance(active_task_gates, Mapping) or set(active_task_gates) != {
        "known_gate",
        "dbb_specialist",
    } or any(not isinstance(value, bool) for value in active_task_gates.values()):
        raise ValueError("V27 L3 fitted quality active-gate mapping differs")
    _probability(quality.get("threshold"), field="quality.threshold")
    if int(quality.get("positive_support", 0)) <= 0 or int(
        quality.get("negative_support", 0)
    ) <= 0:
        raise ValueError("V27 L3 fitted quality support is invalid")
    head_checkpoints = scores.get("head_checkpoints")
    if not isinstance(head_checkpoints, Mapping) or set(head_checkpoints) != {
        V27_L1_TASK_NAME,
        V27_L2_TASK_NAME,
    }:
        raise ValueError("V27 L3 fitted head checkpoint provenance differs")
    for task, entry in head_checkpoints.items():
        if not isinstance(entry, Mapping):
            raise ValueError(f"V27 L3 fitted head provenance is invalid: {task}")
        _verify_digest(entry.get("sha256"), field=f"head_checkpoints.{task}.sha256")
        _verify_digest(
            entry.get("trajectory_head_state_sha256"),
            field=f"head_checkpoints.{task}.trajectory_head_state_sha256",
        )
    score_caches = scores.get("calibration_score_cache_sha256")
    if not isinstance(score_caches, Mapping) or set(score_caches) != {
        V27_L1_TASK_NAME,
        V27_L2_TASK_NAME,
    }:
        raise ValueError("V27 L3 fitted score-cache provenance differs")
    for task, digest in score_caches.items():
        _verify_digest(digest, field=f"calibration_score_cache_sha256.{task}")
    if int(scores.get("calibration_pool_parent_count", 0)) <= 0:
        raise ValueError("V27 L3 fitted calibration support is invalid")
    calibration_scope = scores.get("calibration_scope")
    if not isinstance(calibration_scope, Mapping):
        raise ValueError("V27 L3 fitted calibration scope is missing")
    if (
        calibration_scope.get("pool_role") != "calibration"
        or calibration_scope.get("head_optimization_independent") is not True
        or calibration_scope.get("classifier_checkpoint_selection_independent")
        is not False
        or calibration_scope.get("coverage_guarantee")
        != "not_claimed_after_joint_action_selection_or_classifier_checkpoint_selection"
    ):
        raise ValueError("V27 L3 fitted calibration scope differs")
    if scores.get("unknown_used_for_classifier_training") is not False:
        raise ValueError("V27 L3 fitted scores violate the Unknown training boundary")
    if int(scores.get("test_payload_access_count", -1)) != 0:
        raise ValueError("V27 L3 fitted scores violate the test boundary")


def verify_l3_registry(
    registry: Mapping[str, Any],
    *,
    expected_system_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify both the custom registry digest and optional sealed envelope."""

    payload = dict(registry)
    payload_digest = payload.pop("payload_sha256", None)
    if payload_digest is not None:
        if not isinstance(payload_digest, str) or payload_digest != canonical_sha256(
            payload
        ):
            raise ValueError("V27 sealed L3 registry payload hash cannot be recomputed")
    declared = payload.pop("registry_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("V27 L3 registry hash cannot be recomputed")
    expected_manifest = _verify_digest(
        expected_system_manifest_sha256, field="expected_system_manifest_sha256"
    )
    if payload.get("schema_version") != L3_SCHEMA_VERSION or payload.get(
        "system_manifest_sha256"
    ) != expected_manifest:
        raise ValueError("V27 L3 registry schema/manifest binding differs")
    if (
        payload.get("l1_task") != V27_L1_TASK_NAME
        or payload.get("l2_task") != V27_L2_TASK_NAME
        or payload.get("class_names") != list(V27_L2_CLASS_NAMES)
    ):
        raise ValueError("V27 L3 registry task/class binding differs")
    if payload.get("contract") != l3_contract():
        raise ValueError("V27 L3 registry action contract differs")
    status = payload.get("status")
    if status not in {"unfitted", "fitted"}:
        raise ValueError("V27 L3 registry status is invalid")
    if status == "unfitted":
        if payload.get("fitted_scores") is not None:
            raise ValueError("unfitted V27 L3 registry cannot carry score thresholds")
    else:
        scores = payload.get("fitted_scores")
        if not isinstance(scores, Mapping):
            raise ValueError("fitted V27 L3 registry requires registered calibration payload")
        _verify_fitted_scores(
            scores,
            expected_system_manifest_sha256=expected_manifest,
        )
    result = {**payload, "registry_sha256": declared}
    if payload_digest is not None:
        result["payload_sha256"] = payload_digest
    return result


def _resolve_learned_quality_gate(
    *,
    task: str,
    calibration_status: str,
    gate_active: bool | None,
    eligible: bool | None,
    required: bool,
) -> tuple[bool | None, bool | None]:
    """Validate one learned-quality gate without conflating ``None`` states.

    ``None`` is the only valid learned-quality verdict for a calibrated,
    inactive gate: that score is retained for diagnostics but must not affect a
    product action.  On a fitted route that needs the task, a caller must state
    this explicitly with ``gate_active=False``; otherwise an omitted model
    output could silently become an advisory score.

    Existing direct callers that provide a concrete bool retain their legacy
    meaning (an active gate).  This keeps smoke/API compatibility while all
    registry-driven paths carry the explicit status.
    """

    if gate_active is not None and not isinstance(gate_active, bool):
        raise TypeError(f"V27 L3 {task} learned-quality gate status must be bool or None")
    if eligible is not None and not isinstance(eligible, bool):
        raise TypeError(f"V27 L3 {task} learned-quality verdict must be bool or None")
    if gate_active is False:
        if eligible is not None:
            raise ValueError(
                f"V27 L3 inactive {task} learned-quality gate cannot carry a verdict"
            )
        return None, False
    if gate_active is True:
        if eligible is None and required:
            raise ValueError(
                f"fitted V27 {task} route requires a learned-quality verdict "
                "when its gate is active"
            )
        return eligible, True
    if eligible is not None:
        # Compatibility with the former API: a concrete verdict meant an
        # active gate before gate state was made explicit.
        return eligible, True
    if required and calibration_status == "fitted":
        raise ValueError(
            f"fitted V27 {task} route requires explicit learned-quality gate "
            "status when its verdict is None"
        )
    return None, None


def resolve_l3_action(
    *,
    calibration_status: str,
    l1_target_accepted: bool,
    l1_clutter_accepted: bool,
    l1_knownness_accepted: bool | None,
    l2_knownness_accepted: bool | None,
    dbb_conformal_candidates: tuple[str, ...] | list[str] | None,
    streaming: bool,
    l1_deterministic_quality_eligible: bool | None = None,
    l1_learned_quality_eligible: bool | None = None,
    l1_learned_quality_gate_active: bool | None = None,
    l2_deterministic_quality_eligible: bool | None = None,
    l2_learned_quality_eligible: bool | None = None,
    l2_learned_quality_gate_active: bool | None = None,
    # Compatibility aliases for the original smoke/API shape.  New callers
    # must use task-specific quality arguments; aliases are mapped to both
    # tasks only when an explicit task-specific value is absent.
    deterministic_quality_eligible: bool | None = None,
    learned_quality_eligible: bool | None = None,
) -> V27L3Decision:
    """Resolve one action in the task-specific quality-first order.

    L2 quality is intentionally evaluated only after the L1 target route has
    been accepted.  A malformed or immature L2 observation must therefore not
    suppress a valid L1 Clutter action.
    """

    status = str(calibration_status)
    if status not in {"unfitted", "fitted"}:
        raise ValueError("V27 L3 calibration status is invalid")
    if l1_deterministic_quality_eligible is None:
        l1_deterministic_quality_eligible = deterministic_quality_eligible
    if l1_learned_quality_eligible is None:
        l1_learned_quality_eligible = learned_quality_eligible
    if l2_deterministic_quality_eligible is None and deterministic_quality_eligible is not None:
        l2_deterministic_quality_eligible = deterministic_quality_eligible
    if l2_learned_quality_eligible is None and learned_quality_eligible is not None:
        l2_learned_quality_eligible = learned_quality_eligible
    if l1_deterministic_quality_eligible is None:
        raise TypeError("V27 L3 requires an L1 deterministic quality verdict")

    for name, value in (
        ("l1_deterministic_quality_eligible", l1_deterministic_quality_eligible),
        ("l2_deterministic_quality_eligible", l2_deterministic_quality_eligible),
        ("l1_knownness_accepted", l1_knownness_accepted),
        ("l2_knownness_accepted", l2_knownness_accepted),
    ):
        if value is not None and not isinstance(value, bool):
            raise TypeError(f"V27 L3 {name} must be bool or None")
    if not isinstance(l1_target_accepted, bool) or not isinstance(
        l1_clutter_accepted, bool
    ):
        raise TypeError("V27 L3 L1 route verdicts must be bool")
    if not isinstance(streaming, bool):
        raise TypeError("V27 L3 streaming must be bool")
    candidates = tuple(str(value) for value in (dbb_conformal_candidates or ()))
    if len(set(candidates)) != len(candidates) or any(
        value not in _DBB_TO_ACTION for value in candidates
    ):
        raise ValueError("V27 L3 DBB candidate set differs from canonical classes")
    (
        l1_learned_quality_eligible,
        l1_learned_quality_gate_active,
    ) = _resolve_learned_quality_gate(
        task="L1",
        calibration_status=status,
        gate_active=l1_learned_quality_gate_active,
        eligible=l1_learned_quality_eligible,
        required=status == "fitted",
    )
    (
        l2_learned_quality_eligible,
        l2_learned_quality_gate_active,
    ) = _resolve_learned_quality_gate(
        task="L2",
        calibration_status=status,
        gate_active=l2_learned_quality_gate_active,
        eligible=l2_learned_quality_eligible,
        required=False,
    )
    # Deterministic causal maturity remains a hard Pending boundary.  A
    # learned score only participates when its independently calibrated gate
    # is active; an inactive score remains available for diagnostics.
    l1_quality_eligible = bool(l1_deterministic_quality_eligible) and (
        not l1_learned_quality_gate_active or l1_learned_quality_eligible is True
    )
    l2_quality_eligible = (
        None
        if l2_deterministic_quality_eligible is None
        else (
            None
            if l2_learned_quality_gate_active is None
            else bool(l2_deterministic_quality_eligible)
            and (
                not l2_learned_quality_gate_active
                or l2_learned_quality_eligible is True
            )
        )
    )
    common = {
        "calibration_status": status,
        "l1_target_accepted": bool(l1_target_accepted),
        "l1_clutter_accepted": bool(l1_clutter_accepted),
        "l1_knownness_accepted": l1_knownness_accepted,
        "l2_knownness_accepted": l2_knownness_accepted,
        "l1_deterministic_quality_eligible": bool(
            l1_deterministic_quality_eligible
        ),
        "l1_learned_quality_eligible": l1_learned_quality_eligible,
        "l1_learned_quality_gate_active": l1_learned_quality_gate_active,
        "l2_deterministic_quality_eligible": l2_deterministic_quality_eligible,
        "l2_learned_quality_eligible": l2_learned_quality_eligible,
        "l2_learned_quality_gate_active": l2_learned_quality_gate_active,
        "l1_quality_eligible": l1_quality_eligible,
        "l2_quality_eligible": l2_quality_eligible,
        # These two fields remain as readable compatibility summaries.  They
        # describe the L1 gate because L1 is the first semantic boundary.
        "learned_quality_eligible": l1_learned_quality_eligible,
        "quality_eligible": l1_quality_eligible,
        "dbb_candidates": candidates,
    }
    if not bool(l1_deterministic_quality_eligible):
        return V27L3Decision(
            V27Action.PENDING_ABSTAIN, "LOW_QUALITY_OR_IMMATURE", **common
        )
    if status != "fitted":
        return V27L3Decision(
            V27Action.PENDING_ABSTAIN, "CALIBRATION_NOT_FITTED", **common
        )
    if (
        l1_learned_quality_gate_active
        and l1_learned_quality_eligible is False
    ):
        return V27L3Decision(
            V27Action.PENDING_ABSTAIN, "LOW_QUALITY_OR_IMMATURE", **common
        )
    if bool(l1_target_accepted) and bool(l1_clutter_accepted):
        return V27L3Decision(V27Action.PENDING_ABSTAIN, "L1_ROUTE_CONFLICT", **common)
    if l1_knownness_accepted is None:
        raise ValueError("fitted V27 L3 requires an L1 knownness verdict")
    if l1_knownness_accepted is False:
        return V27L3Decision(V27Action.UNKNOWN_REJECTED, "OOD_NONCONFORMING", **common)
    if bool(l1_clutter_accepted):
        return V27L3Decision(V27Action.CLUTTER, "L1_CLUTTER_ACCEPTED", **common)
    if not bool(l1_target_accepted):
        return V27L3Decision(V27Action.UNKNOWN_REJECTED, "OOD_NONCONFORMING", **common)
    if l2_deterministic_quality_eligible is None:
        raise ValueError("fitted V27 target route requires an L2 deterministic quality verdict")
    (
        l2_learned_quality_eligible,
        l2_learned_quality_gate_active,
    ) = _resolve_learned_quality_gate(
        task="L2 target",
        calibration_status=status,
        gate_active=l2_learned_quality_gate_active,
        eligible=l2_learned_quality_eligible,
        required=True,
    )
    if l2_deterministic_quality_eligible is not True:
        return V27L3Decision(
            V27Action.PENDING_ABSTAIN, "LOW_QUALITY_OR_IMMATURE", **common
        )
    if (
        l2_learned_quality_gate_active
        and l2_learned_quality_eligible is False
    ):
        return V27L3Decision(
            V27Action.PENDING_ABSTAIN, "LOW_QUALITY_OR_IMMATURE", **common
        )
    if l2_knownness_accepted is None:
        raise ValueError("fitted V27 target route requires an L2 knownness verdict")
    if l2_knownness_accepted is False or not candidates:
        return V27L3Decision(
            V27Action.UNKNOWN_REJECTED, "OOD_NONCONFORMING", **common
        )
    if len(candidates) == 1 and l2_knownness_accepted is True:
        return V27L3Decision(
            _DBB_TO_ACTION[candidates[0]], "KNOWN_DBB_CLASS_ACCEPTED", **common
        )
    if len(candidates) > 1:
        action = V27Action.PENDING_ABSTAIN if bool(streaming) else V27Action.UNKNOWN_REJECTED
        return V27L3Decision(action, "KNOWN_CLASS_AMBIGUOUS", **common)
    return V27L3Decision(V27Action.UNKNOWN_REJECTED, "OOD_NONCONFORMING", **common)


__all__ = [
    "ACTION_SELECTION_METHOD",
    "L1_CLASS_NAMES",
    "L3_SCHEMA_VERSION",
    "V27Action",
    "V27L3Decision",
    "l3_contract",
    "make_fitted_registry",
    "make_unfitted_registry",
    "resolve_l3_action",
    "verify_l3_registry",
]
