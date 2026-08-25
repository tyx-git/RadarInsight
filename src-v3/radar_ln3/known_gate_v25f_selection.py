"""V25f checkpoint utility selection for the frozen self-developed KnownGate run.

V25f does not retrain the model.  It scans the completed V25a epoch states and
separates two questions that were previously conflated:

* strict product-safe feasibility, where KnownTarget parent recall may not fall;
* a diagnostic frontier that permits one KnownTarget parent loss solely to
  reveal whether the causal-prefix objective is hidden by checkpoint choice.

The diagnostic frontier is never an adoption gate.
"""

from __future__ import annotations

import json
import math
from typing import Any, Iterable, Mapping, Sequence

from .known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)


V25F_SELECTOR_SCHEMA = "RADAR-LN3-V25f-causal-prefix-checkpoint-utility-selector-v1"
V25F_MINIMUM_CLUTTER_PARENT_SUPPORT = 9
V25F_MINIMUM_CLUTTER_TRUE_POSITIVES = 8
V25F_MINIMUM_CLUTTER_PRECISION = 0.90
V25F_MINIMUM_CLUTTER_RECALL = (
    V25F_MINIMUM_CLUTTER_TRUE_POSITIVES / V25F_MINIMUM_CLUTTER_PARENT_SUPPORT
)
V25F_DIAGNOSTIC_MAX_KNOWN_TARGET_PARENT_LOSS = 1
V25F_PREFIX_MINIMUM_EVENTS = 3
_TOLERANCE = 1.0e-12


def _copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(dict(value), ensure_ascii=False))


def _finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _class_row(metrics: Mapping[str, Any], class_name: str) -> dict[str, Any]:
    rows = {
        str(row.get("class")): dict(row)
        for row in list(metrics.get("per_class") or ())
    }
    if set(rows) != set(KNOWN_GATE_CLASS_NAMES):
        raise ValueError("V25f metric classes differ from the frozen KnownGate classes")
    return rows[class_name]


def metrics_from_records(
    records: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    parent = [dict(row) for row in records.get("parent", ())]
    eligible = [dict(row) for row in records.get("eligible_prefix", ())]
    window = [dict(row) for row in records.get("window", ())]
    if len(parent) != 232 or len(eligible) != 3756 or len(window) != 4452:
        raise ValueError("V25f frozen validation record counts differ")
    return {
        "parent_metrics": summarize_known_gate_records(parent, scope="parent_final"),
        "eligible_prefix_metrics": summarize_known_gate_records(
            eligible, scope="eligible_causal_prefix_auxiliary_excluding_final"
        ),
        "parent_balanced_eligible_prefix_metrics": (
            summarize_parent_balanced_eligible_prefix_records(
                eligible,
                scope="parent_balanced_eligible_causal_prefix_auxiliary_excluding_final",
            )
        ),
        "window_metrics": summarize_known_gate_records(
            window, scope="causal_prefix_windows"
        ),
    }


def reference_from_records(
    records: Mapping[str, Iterable[Mapping[str, Any]]],
    *,
    epoch: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    if int(epoch) <= 0:
        raise ValueError("V25f legacy reference epoch must be positive")
    metrics = metrics_from_records(records)
    if not _finite(metrics):
        raise ValueError("V25f legacy reference contains non-finite metrics")
    return {
        "schema_version": V25F_SELECTOR_SCHEMA,
        "role": "frozen_v25a_legacy_parent_final_selector_reference",
        "epoch": int(epoch),
        "source": dict(source),
        "metrics": {key: _copy_json(value) for key, value in metrics.items()},
    }


def _parent_counts(metrics: Mapping[str, Any]) -> dict[str, int]:
    matrix = list(metrics.get("confusion_matrix_rows_true_columns_pred") or ())
    if len(matrix) != 2 or any(len(list(row)) != 2 for row in matrix):
        raise ValueError("V25f parent confusion matrix is invalid")
    values = [[int(value) for value in row] for row in matrix]
    if any(value < 0 for row in values for value in row):
        raise ValueError("V25f parent confusion matrix contains negative counts")
    return {
        "known_target_support": values[0][0] + values[0][1],
        "known_target_true_positive_count": values[0][0],
        "clutter_support": values[1][0] + values[1][1],
        "clutter_true_positive_count": values[1][1],
        "clutter_false_positive_count": values[0][1],
        "clutter_false_negative_count": values[1][0],
    }


def _snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "parent_final": _copy_json(metrics["parent_metrics"]),
        "eligible_prefix_event_pooled": _copy_json(
            metrics["eligible_prefix_metrics"]
        ),
        "eligible_prefix_parent_balanced": _copy_json(
            metrics["parent_balanced_eligible_prefix_metrics"]
        ),
        "window_event_pooled": _copy_json(metrics["window_metrics"]),
    }


def _ranking_key(
    *,
    balanced_clutter: Mapping[str, Any],
    parent_clutter: Mapping[str, Any],
    balanced_target: Mapping[str, Any],
    parent_metrics: Mapping[str, Any],
    epoch: int,
) -> list[float]:
    # F1 is primary; min(P,R) prevents a selector from choosing a pure
    # precision or pure recall operating point when both are product evidence.
    return [
        float(balanced_clutter["f1"]),
        min(float(balanced_clutter["precision"]), float(balanced_clutter["recall"])),
        float(balanced_clutter["recall"]),
        float(balanced_clutter["precision"]),
        float(balanced_target["recall"]),
        float(parent_metrics["macro_f1"]),
        -float(epoch),
    ]


def assess_checkpoint(
    metrics: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    epoch: int,
    prefix_action_contract: Mapping[str, Any],
) -> dict[str, Any]:
    if int(epoch) <= 0 or not _finite(metrics):
        raise ValueError("V25f checkpoint metrics are invalid")
    reference_metrics = dict(reference.get("metrics") or {})
    if set(metrics) != set(reference_metrics):
        raise ValueError("V25f checkpoint metric scopes differ from the reference")
    parent = dict(metrics["parent_metrics"])
    balanced = dict(metrics["parent_balanced_eligible_prefix_metrics"])
    reference_parent = dict(reference_metrics["parent_metrics"])
    reference_balanced = dict(reference_metrics["parent_balanced_eligible_prefix_metrics"])
    counts = _parent_counts(parent)
    parent_target = _class_row(parent, "KNOWN_TARGET_EVIDENCE")
    parent_clutter = _class_row(parent, "KNOWN_CLUTTER_EVIDENCE")
    balanced_target = _class_row(balanced, "KNOWN_TARGET_EVIDENCE")
    balanced_clutter = _class_row(balanced, "KNOWN_CLUTTER_EVIDENCE")
    reference_parent_target = _class_row(reference_parent, "KNOWN_TARGET_EVIDENCE")
    reference_balanced_target = _class_row(reference_balanced, "KNOWN_TARGET_EVIDENCE")

    common = {
        "all_selector_metrics_are_finite": True,
        "parent_final_clutter_support_is_9": (
            counts["clutter_support"] == V25F_MINIMUM_CLUTTER_PARENT_SUPPORT
        ),
        "parent_final_clutter_true_positives_at_least_8_of_9": (
            counts["clutter_true_positive_count"]
            >= V25F_MINIMUM_CLUTTER_TRUE_POSITIVES
        ),
        "parent_final_clutter_recall_at_least_88_89_percent": (
            float(parent_clutter["recall"]) + _TOLERANCE
            >= V25F_MINIMUM_CLUTTER_RECALL
        ),
        "parent_final_clutter_precision_at_least_0_90": (
            float(parent_clutter["precision"]) + _TOLERANCE
            >= V25F_MINIMUM_CLUTTER_PRECISION
        ),
        "parent_target_precision_is_not_lower_than_legacy": (
            float(parent_target["precision"]) + _TOLERANCE
            >= float(reference_parent_target["precision"])
        ),
        "parent_balanced_prefix_known_target_recall_is_not_lower": (
            float(balanced_target["recall"]) + _TOLERANCE
            >= float(reference_balanced_target["recall"])
        ),
        "prefix_action_contract_is_frozen": (
            int(prefix_action_contract.get("minimum_evidence_events", -1))
            == V25F_PREFIX_MINIMUM_EVENTS
            and prefix_action_contract.get("events_before_minimum")
            == "PENDING_ABSTAIN"
            and prefix_action_contract.get("local_argmax_is_product_action") is False
        ),
    }
    strict = {
        **common,
        "strict_parent_known_target_recall_is_not_lower": (
            float(parent_target["recall"]) + _TOLERANCE
            >= float(reference_parent_target["recall"])
        ),
    }
    diagnostic = {
        **common,
        "diagnostic_known_target_parent_loss_at_most_one": (
            counts["known_target_true_positive_count"]
            >= counts["known_target_support"]
            - V25F_DIAGNOSTIC_MAX_KNOWN_TARGET_PARENT_LOSS
        ),
    }
    strict_feasible = all(strict.values())
    diagnostic_feasible = all(diagnostic.values())
    return {
        "schema_version": V25F_SELECTOR_SCHEMA,
        "epoch": int(epoch),
        "strict_product_safe_feasible": strict_feasible,
        "diagnostic_utility_feasible": diagnostic_feasible,
        "strict_constraints": strict,
        "diagnostic_constraints": diagnostic,
        "parent_final_counts": counts,
        "ranking_definition": [
            "parent_balanced_prefix_clutter_f1_desc",
            "parent_balanced_prefix_clutter_min_precision_recall_desc",
            "parent_balanced_prefix_clutter_recall_desc",
            "parent_balanced_prefix_clutter_precision_desc",
            "parent_balanced_prefix_known_target_recall_desc",
            "parent_final_macro_f1_desc",
            "earlier_epoch_asc_for_exact_tie",
        ],
        "ranking_key": _ranking_key(
            balanced_clutter=balanced_clutter,
            parent_clutter=parent_clutter,
            balanced_target=balanced_target,
            parent_metrics=parent,
            epoch=epoch,
        ),
        "metrics": _snapshot(metrics),
    }


def select_best(
    candidates: Sequence[Mapping[str, Any]], *, mode: str
) -> dict[str, Any] | None:
    if mode not in {"strict_product_safe", "diagnostic_utility"}:
        raise ValueError("V25f selector mode is invalid")
    flag = (
        "strict_product_safe_feasible"
        if mode == "strict_product_safe"
        else "diagnostic_utility_feasible"
    )
    eligible = [dict(item) for item in candidates if item.get(flag) is True]
    if not eligible:
        return None
    eligible.sort(
        key=lambda item: tuple(float(value) for value in item["ranking_key"]),
        reverse=True,
    )
    return eligible[0]


def metric_delta(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in (
        "parent_final",
        "eligible_prefix_event_pooled",
        "eligible_prefix_parent_balanced",
        "window_event_pooled",
    ):
        before = dict(reference[scope])
        after = dict(candidate[scope])
        before_rows = {
            str(row["class"]): row for row in before["per_class"]
        }
        after_rows = {str(row["class"]): row for row in after["per_class"]}
        result[scope] = {
            "accuracy_delta": float(after["accuracy"] - before["accuracy"]),
            "macro_f1_delta": float(after["macro_f1"] - before["macro_f1"]),
            "known_target_recall_delta": float(
                after_rows["KNOWN_TARGET_EVIDENCE"]["recall"]
                - before_rows["KNOWN_TARGET_EVIDENCE"]["recall"]
            ),
            "known_target_precision_delta": float(
                after_rows["KNOWN_TARGET_EVIDENCE"]["precision"]
                - before_rows["KNOWN_TARGET_EVIDENCE"]["precision"]
            ),
            "clutter_precision_delta": float(
                after_rows["KNOWN_CLUTTER_EVIDENCE"]["precision"]
                - before_rows["KNOWN_CLUTTER_EVIDENCE"]["precision"]
            ),
            "clutter_recall_delta": float(
                after_rows["KNOWN_CLUTTER_EVIDENCE"]["recall"]
                - before_rows["KNOWN_CLUTTER_EVIDENCE"]["recall"]
            ),
            "clutter_f1_delta": float(
                after_rows["KNOWN_CLUTTER_EVIDENCE"]["f1"]
                - before_rows["KNOWN_CLUTTER_EVIDENCE"]["f1"]
            ),
        }
    return result


__all__ = [
    "V25F_DIAGNOSTIC_MAX_KNOWN_TARGET_PARENT_LOSS",
    "V25F_MINIMUM_CLUTTER_PARENT_SUPPORT",
    "V25F_MINIMUM_CLUTTER_PRECISION",
    "V25F_MINIMUM_CLUTTER_RECALL",
    "V25F_MINIMUM_CLUTTER_TRUE_POSITIVES",
    "V25F_PREFIX_MINIMUM_EVENTS",
    "V25F_SELECTOR_SCHEMA",
    "assess_checkpoint",
    "metric_delta",
    "metrics_from_records",
    "reference_from_records",
    "select_best",
]
