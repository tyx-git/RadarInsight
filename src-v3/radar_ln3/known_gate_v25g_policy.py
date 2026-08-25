"""Pure causal ACCEPT/PENDING policy utilities for V25g.

V25g deliberately changes only the interpretation of frozen KnownGate prefix
probabilities.  It does not alter the liquid encoder, features, dt semantics,
loss, or checkpoint weights.  ``None`` is used internally for PENDING rather
than being overloaded as an Unknown class.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence


V25G_POLICY_SCHEMA = "RADAR-LN3-V25g-causal-accept-pending-policy-v1"
V25G_MINIMUM_EVIDENCE_EVENTS = 3
V25G_MINIMUM_THRESHOLD = 0.5
V25G_TOLERANCE = 1.0e-12
V25G_MAX_THRESHOLD_CANDIDATES = 256
V25G_PENDING_ACTION = "PENDING_ABSTAIN"
V25G_CLASS_NAMES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
V25G_BASE_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")


def _finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V25g {name} must be finite")
    return result


def _validate_source_row(row: Mapping[str, Any], *, minimum_events: int) -> None:
    if int(row.get("true_id", -1)) not in {0, 1}:
        raise ValueError("V25g policy row has an invalid true_id")
    if int(row.get("pred_id", -1)) not in {0, 1}:
        raise ValueError("V25g policy row has an invalid local pred_id")
    event_index = int(row.get("event_index", -1))
    if event_index < 0:
        raise ValueError("V25g policy row has an invalid event_index")
    confidence = _finite_float(row.get("confidence"), name="confidence")
    if confidence < -V25G_TOLERANCE or confidence > 1.0 + V25G_TOLERANCE:
        raise ValueError("V25g policy confidence is outside [0,1]")
    expected_eligible = event_index + 1 >= int(minimum_events)
    if bool(row.get("product_action_eligible")) != expected_eligible:
        raise ValueError("V25g source row disagrees with the frozen eligibility contract")
    parent_id = str(row.get("parent_sample_id", ""))
    if not parent_id:
        raise ValueError("V25g policy row has an empty parent_sample_id")


def apply_accept_pending_policy(
    records: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
    minimum_events: int = V25G_MINIMUM_EVIDENCE_EVENTS,
) -> list[dict[str, Any]]:
    """Attach a causal ACCEPT/PENDING action to every local prefix record.

    The local class prediction and confidence are preserved verbatim.  A row
    is accepted only when it is eligible and its local argmax confidence is at
    least ``threshold``.  Pending rows have ``policy_action_id=None`` and are
    never converted into an Unknown class.
    """

    threshold_value = _finite_float(threshold, name="threshold")
    if threshold_value < V25G_MINIMUM_THRESHOLD - V25G_TOLERANCE or threshold_value > 1.0 + V25G_TOLERANCE:
        raise ValueError("V25g threshold must be in [0.5,1]")
    minimum = int(minimum_events)
    if minimum < 2:
        raise ValueError("V25g minimum evidence events must exclude the first event")

    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for source in records:
        row = dict(source)
        _validate_source_row(row, minimum_events=minimum)
        key = (str(row["parent_sample_id"]), int(row["event_index"]))
        if key in seen:
            raise ValueError("V25g policy records repeat a parent/event key")
        seen.add(key)
        eligible = bool(row["product_action_eligible"])
        accepted = eligible and float(row["confidence"]) + V25G_TOLERANCE >= threshold_value
        action_id = int(row["pred_id"]) if accepted else None
        row.update(
            {
                "policy_schema": V25G_POLICY_SCHEMA,
                "policy_threshold": threshold_value,
                "policy_action_id": action_id,
                "policy_action": (
                    V25G_CLASS_NAMES[action_id]
                    if action_id is not None
                    else V25G_PENDING_ACTION
                ),
                "policy_accepted": bool(accepted),
                "policy_pending": not bool(accepted),
            }
        )
        output.append(row)
    output.sort(key=lambda row: (str(row["parent_sample_id"]), int(row["event_index"])))
    return output


def first_causal_decisions(
    policy_records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return one first-ACCEPT-or-PENDING decision per parent.

    The input is normally ``eligible_prefix`` and therefore excludes the
    parent-final event.  No later event may revise an already accepted action;
    this is intentionally a one-factor action-layer diagnostic without
    patience or revocation.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in policy_records:
        row = dict(source)
        parent_id = str(row.get("parent_sample_id", ""))
        if not parent_id:
            raise ValueError("V25g decision record has an empty parent_sample_id")
        grouped[parent_id].append(row)

    decisions: list[dict[str, Any]] = []
    for parent_id, rows in grouped.items():
        rows.sort(key=lambda row: int(row["event_index"]))
        first = next((row for row in rows if row.get("policy_accepted") is True), None)
        reference = rows[-1]
        if first is None:
            decision = {
                "parent_sample_id": parent_id,
                "track_id": str(reference.get("track_id", "")),
                "source_name": str(reference.get("source_name", "")),
                "label_name": str(reference.get("label_name", "")),
                "true_id": int(reference["true_id"]),
                "policy_action_id": None,
                "policy_action": V25G_PENDING_ACTION,
                "policy_accepted": False,
                "policy_pending": True,
                "decision_event_index": None,
                "decision_window_length": None,
                "decision_confidence": None,
                "decision_local_pred_id": None,
                "policy_threshold": float(reference["policy_threshold"]),
                "eligible_prefix_event_count": len(rows),
            }
        else:
            decision = {
                "parent_sample_id": parent_id,
                "track_id": str(first.get("track_id", "")),
                "source_name": str(first.get("source_name", "")),
                "label_name": str(first.get("label_name", "")),
                "true_id": int(first["true_id"]),
                "policy_action_id": int(first["policy_action_id"]),
                "policy_action": str(first["policy_action"]),
                "policy_accepted": True,
                "policy_pending": False,
                "decision_event_index": int(first["event_index"]),
                "decision_window_length": int(first["window_length"]),
                "decision_confidence": float(first["confidence"]),
                "decision_local_pred_id": int(first["pred_id"]),
                "policy_threshold": float(first["policy_threshold"]),
                "eligible_prefix_event_count": len(rows),
            }
        decisions.append(decision)
    decisions.sort(key=lambda row: str(row["parent_sample_id"]))
    return decisions


def _metric_row(
    *,
    class_id: int,
    true_mass: float,
    predicted_mass: float,
    true_positive_mass: float,
    support: int,
) -> dict[str, Any]:
    precision = true_positive_mass / predicted_mass if predicted_mass > 0.0 else 0.0
    recall = true_positive_mass / true_mass if true_mass > 0.0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0.0 else 0.0
    return {
        "class": V25G_CLASS_NAMES[class_id],
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(support),
        "true_mass": float(true_mass),
        "predicted_mass": float(predicted_mass),
        "true_positive_mass": float(true_positive_mass),
    }


def summarize_policy_records(
    records: Iterable[Mapping[str, Any]],
    *,
    scope: str,
    parent_balanced: bool = False,
) -> dict[str, Any]:
    """Summarize two class actions while keeping Pending outside Unknown."""

    rows = [dict(row) for row in records]
    if not rows:
        raise ValueError("V25g policy metric scope is empty")
    weights: dict[tuple[str, int], float] = {}
    if parent_balanced:
        grouped: dict[str, int] = defaultdict(int)
        for row in rows:
            grouped[str(row["parent_sample_id"])] += 1
        weights = {
            (str(row["parent_sample_id"]), int(row["event_index"])): 1.0
            / float(grouped[str(row["parent_sample_id"])]
            )
            for row in rows
        }
    else:
        weights = {
            (str(row["parent_sample_id"]), int(row.get("event_index", index))): 1.0
            for index, row in enumerate(rows)
        }

    matrix = [[0.0, 0.0], [0.0, 0.0]]
    true_mass = [0.0, 0.0]
    predicted_mass = [0.0, 0.0]
    true_positive_mass = [0.0, 0.0]
    true_parent_sets = [set(), set()]
    correct_mass = 0.0
    accepted_mass = 0.0
    pending_mass = 0.0
    for index, row in enumerate(rows):
        true_id = int(row["true_id"])
        action_id = row.get("policy_action_id")
        action_id = None if action_id is None else int(action_id)
        key = (str(row["parent_sample_id"]), int(row.get("event_index", index)))
        weight = float(weights[key])
        true_mass[true_id] += weight
        true_parent_sets[true_id].add(str(row["parent_sample_id"]))
        if action_id is None:
            pending_mass += weight
            continue
        if action_id not in {0, 1}:
            raise ValueError("V25g policy action id is invalid")
        predicted_mass[action_id] += weight
        matrix[true_id][action_id] += weight
        accepted_mass += weight
        correct = action_id == true_id
        if correct:
            true_positive_mass[true_id] += weight
            correct_mass += weight

    per_class = [
        _metric_row(
            class_id=class_id,
            true_mass=true_mass[class_id],
            predicted_mass=predicted_mass[class_id],
            true_positive_mass=true_positive_mass[class_id],
            support=len(true_parent_sets[class_id]),
        )
        for class_id in range(2)
    ]
    macro_precision = sum(row["precision"] for row in per_class) / 2.0
    macro_recall = sum(row["recall"] for row in per_class) / 2.0
    macro_f1 = sum(row["f1"] for row in per_class) / 2.0
    total_mass = sum(true_mass)
    accepted_accuracy = correct_mass / accepted_mass if accepted_mass > 0.0 else 0.0
    return {
        "scope": scope,
        "aggregation": (
            "one parent total mass with each eligible prefix weighted 1/n_parent"
            if parent_balanced
            else "event rows"
        ),
        "support": int(sum(len(value) for value in true_parent_sets)) if parent_balanced else len(rows),
        "event_support": len(rows),
        "weighted_support": float(total_mass),
        "accepted_count": float(accepted_mass),
        "pending_count": float(pending_mass),
        "coverage": float(accepted_mass / total_mass) if total_mass > 0.0 else 0.0,
        "accepted_accuracy": float(accepted_accuracy),
        "accuracy": float(correct_mass / total_mass) if total_mass > 0.0 else 0.0,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(macro_recall),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": [
            [float(value) for value in row] for row in matrix
        ],
        "pending_by_true_class": {
            V25G_CLASS_NAMES[class_id]: float(true_mass[class_id] - sum(matrix[class_id]))
            for class_id in range(2)
        },
    }


def class_metric(metrics: Mapping[str, Any], class_name: str, field: str) -> float:
    for row in list(metrics.get("per_class") or ()):
        if str(row.get("class")) == class_name:
            return float(row[field])
    raise ValueError(f"V25g metrics lack class {class_name}")


def _decision_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    policy = apply_accept_pending_policy(records, threshold=threshold)
    decisions = first_causal_decisions(policy)
    metrics = summarize_policy_records(decisions, scope="first_causal_decision")
    return {
        "threshold": float(threshold),
        "metrics": metrics,
        "accepted_parent_count": sum(bool(row["policy_accepted"]) for row in decisions),
        "pending_parent_count": sum(bool(row["policy_pending"]) for row in decisions),
        "mean_decision_event_index": (
            sum(
                int(row["decision_event_index"])
                for row in decisions
                if row["decision_event_index"] is not None
            )
            / max(1, sum(row["decision_event_index"] is not None for row in decisions))
        ),
    }


def select_train_only_threshold(
    records: Sequence[Mapping[str, Any]],
    *,
    minimum_clutter_precision: float = 0.90,
) -> dict[str, Any]:
    """Select a threshold on train-only frozen-checkpoint predictions.

    This function intentionally does not call the result OOF.  The caller must
    record the calibration provenance and keep the experiment development-only
    until an exact parent-grouped OOF artifact is supplied.
    """

    if not records:
        raise ValueError("V25g calibration records are empty")
    eligible = [row for row in records if bool(row.get("product_action_eligible"))]
    if not eligible:
        raise ValueError("V25g calibration has no eligible prefix records")
    confidence_values = sorted({
        max(V25G_MINIMUM_THRESHOLD, min(1.0, _finite_float(row["confidence"], name="confidence")))
        for row in eligible
    })
    if len(confidence_values) > V25G_MAX_THRESHOLD_CANDIDATES - 1:
        # A complete unique-score sweep is quadratic because every candidate
        # replays all parent prefixes.  Deterministic quantile representatives
        # preserve the operating frontier while bounding both time and memory.
        budget = V25G_MAX_THRESHOLD_CANDIDATES - 1
        selected_values = {
            confidence_values[
                int(round(index * (len(confidence_values) - 1) / max(1, budget - 1)))
            ]
            for index in range(budget)
        }
        confidence_values = sorted(selected_values)
    candidates = sorted({V25G_MINIMUM_THRESHOLD, *confidence_values})
    if len(candidates) > V25G_MAX_THRESHOLD_CANDIDATES:
        raise AssertionError("V25g threshold candidate bound was not enforced")
    baseline = _decision_summary(eligible, threshold=V25G_MINIMUM_THRESHOLD)
    target_recall_floor = class_metric(
        baseline["metrics"], "KNOWN_TARGET_EVIDENCE", "recall"
    )
    clutter_recall_floor = class_metric(
        baseline["metrics"], "KNOWN_CLUTTER_EVIDENCE", "recall"
    )
    evaluated: list[dict[str, Any]] = []
    for threshold in candidates:
        candidate = _decision_summary(eligible, threshold=threshold)
        metrics = candidate["metrics"]
        clutter_precision = class_metric(metrics, "KNOWN_CLUTTER_EVIDENCE", "precision")
        target_recall = class_metric(metrics, "KNOWN_TARGET_EVIDENCE", "recall")
        clutter_recall = class_metric(metrics, "KNOWN_CLUTTER_EVIDENCE", "recall")
        candidate["feasible"] = bool(
            clutter_precision + V25G_TOLERANCE >= float(minimum_clutter_precision)
            and target_recall + V25G_TOLERANCE >= target_recall_floor
            and clutter_recall + V25G_TOLERANCE >= clutter_recall_floor
        )
        candidate["selection_key"] = [
            float(candidate["metrics"]["coverage"]),
            float(candidate["metrics"]["macro_f1"]),
            float(candidate["metrics"]["macro_precision"]),
            -float(candidate["mean_decision_event_index"]),
            -float(threshold),
        ]
        evaluated.append(candidate)
    feasible = [candidate for candidate in evaluated if candidate["feasible"]]
    feasible.sort(key=lambda item: tuple(item["selection_key"]), reverse=True)
    selected = feasible[0] if feasible else None
    return {
        "protocol": "train_only_frozen_checkpoint_calibration",
        "is_oof": False,
        "candidate_count": len(evaluated),
        "feasible_candidate_count": len(feasible),
        "minimum_clutter_precision": float(minimum_clutter_precision),
        "baseline_threshold": baseline,
        "target_recall_floor": float(target_recall_floor),
        "clutter_recall_floor": float(clutter_recall_floor),
        "selected": selected,
        "top_candidates": sorted(
            evaluated,
            key=lambda item: tuple(item["selection_key"]),
            reverse=True,
        )[:25],
    }


def directional_action_changes(
    baseline_records: Sequence[Mapping[str, Any]],
    policy_records: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    before = {
        (str(row["parent_sample_id"]), int(row["event_index"])): int(row["pred_id"])
        for row in baseline_records
    }
    after = {
        (str(row["parent_sample_id"]), int(row["event_index"])): row.get("policy_action_id")
        for row in policy_records
    }
    if set(before) != set(after):
        raise ValueError("V25g paired action records differ")
    result = {
        "prediction_changed_to_pending": 0,
        "prediction_changed_to_target": 0,
        "prediction_changed_to_clutter": 0,
        "pending_count": 0,
        "accepted_count": 0,
    }
    for key, old_id in before.items():
        new_id = after[key]
        if new_id is None:
            result["pending_count"] += 1
            if old_id is not None:
                result["prediction_changed_to_pending"] += 1
        else:
            result["accepted_count"] += 1
            if int(new_id) == 0:
                result["prediction_changed_to_target"] += int(int(old_id) != 0)
            else:
                result["prediction_changed_to_clutter"] += int(int(old_id) != 1)
    return result


__all__ = [
    "V25G_BASE_LABELS",
    "V25G_CLASS_NAMES",
    "V25G_MINIMUM_EVIDENCE_EVENTS",
    "V25G_MINIMUM_THRESHOLD",
    "V25G_MAX_THRESHOLD_CANDIDATES",
    "V25G_PENDING_ACTION",
    "V25G_POLICY_SCHEMA",
    "apply_accept_pending_policy",
    "class_metric",
    "directional_action_changes",
    "first_causal_decisions",
    "select_train_only_threshold",
    "summarize_policy_records",
]
