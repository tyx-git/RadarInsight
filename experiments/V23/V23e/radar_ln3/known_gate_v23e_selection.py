"""Frozen V23e checkpoint-selection contract for official-CfC KnownGate work.

V23e is deliberately *not* a new model or a new optimization run.  It scans
the twenty already saved V23a states and asks whether a checkpoint-selection
rule alone can reconcile parent-final Clutter evidence with early causal
actions.  V23f reuses this exact selector while changing only the declared
causal-prefix loss coefficient.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    summarize_known_gate_records,
    summarize_parent_balanced_eligible_prefix_records,
)
from .known_gate_v23 import sha256_file
from .known_gate_v23_adapters import V23A_VARIANT, artifact_relative


V23E_SELECTOR_SCHEMA = "RADAR-LN3-V23e-parent-balanced-checkpoint-selector-v1"
V23E_MINIMUM_CLUTTER_PARENT_SUPPORT = 9
V23E_MINIMUM_CLUTTER_TRUE_POSITIVES = 8
V23E_MINIMUM_CLUTTER_PRECISION = 0.90
V23E_MINIMUM_CLUTTER_RECALL = (
    V23E_MINIMUM_CLUTTER_TRUE_POSITIVES / V23E_MINIMUM_CLUTTER_PARENT_SUPPORT
)
_TOLERANCE = 1.0e-12


def _copy_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a JSON metric payload without carrying prediction records."""

    return json.loads(json.dumps(dict(metrics), ensure_ascii=False))


def _class_row(metrics: Mapping[str, Any], class_name: str) -> dict[str, Any]:
    rows = {
        str(row.get("class")): dict(row)
        for row in list(metrics.get("per_class") or ())
    }
    if set(rows) != set(KNOWN_GATE_CLASS_NAMES):
        raise ValueError("V23e selector metric classes differ from KnownGate")
    return rows[class_name]


def _finite_metric_payload(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return all(_finite_metric_payload(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return all(_finite_metric_payload(value) for value in payload)
    # ``major_confusion_to`` is intentionally null when no wrong action exists;
    # it is categorical metadata, not a missing numeric selector value.
    if isinstance(payload, bool) or payload is None or isinstance(payload, str):
        return True
    if isinstance(payload, (int, float)):
        return math.isfinite(float(payload))
    return False


def metrics_from_records(records: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
    """Derive all selector scopes from hard action records.

    The new parent-balanced scope preserves per-prefix actions.  It differs
    from historical event pooling only in the contribution weight of each
    prefix, never by averaging probabilities or collapsing a parent into one
    early prediction.
    """

    parent = [dict(row) for row in records.get("parent", ())]
    eligible = [dict(row) for row in records.get("eligible_prefix", ())]
    window = [dict(row) for row in records.get("window", ())]
    if not parent or not eligible or not window:
        raise ValueError("V23e selector requires parent, eligible-prefix and window records")
    return {
        "parent_metrics": summarize_known_gate_records(parent, scope="parent_final"),
        "eligible_prefix_metrics": summarize_known_gate_records(
            eligible,
            scope="eligible_causal_prefix_auxiliary_excluding_final",
        ),
        "parent_balanced_eligible_prefix_metrics": (
            summarize_parent_balanced_eligible_prefix_records(
                eligible,
                scope="parent_balanced_eligible_causal_prefix_auxiliary_excluding_final",
            )
        ),
        "window_metrics": summarize_known_gate_records(
            window,
            scope="causal_prefix_windows",
        ),
    }


def selector_reference(
    evaluation: Mapping[str, Any],
    *,
    epoch: int,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze the legacy V23a selection state used as V23e/V23f safety bound."""

    if int(epoch) <= 0:
        raise ValueError("V23e selector reference epoch must be positive")
    metrics = {
        name: _copy_metrics(dict(evaluation.get(name) or {}))
        for name in (
            "parent_metrics",
            "eligible_prefix_metrics",
            "parent_balanced_eligible_prefix_metrics",
            "window_metrics",
        )
    }
    if not all(metrics.values()) or not _finite_metric_payload(metrics):
        raise ValueError("V23e selector reference metrics are missing or non-finite")
    parent = _class_row(metrics["parent_metrics"], "KNOWN_CLUTTER_EVIDENCE")
    if int(parent.get("support", -1)) != V23E_MINIMUM_CLUTTER_PARENT_SUPPORT:
        raise ValueError("V23e selector reference Clutter parent support differs")
    return {
        "schema_version": V23E_SELECTOR_SCHEMA,
        "role": "frozen_v23a_legacy_parent_final_selector_reference",
        "epoch": int(epoch),
        "source": dict(source),
        "metrics": metrics,
    }


def _parent_final_counts(metrics: Mapping[str, Any]) -> dict[str, int]:
    matrix = list(metrics.get("confusion_matrix_rows_true_columns_pred") or ())
    if len(matrix) != 2 or any(len(list(row)) != 2 for row in matrix):
        raise ValueError("V23e parent-final confusion matrix is invalid")
    values = [[int(value) for value in row] for row in matrix]
    if any(value < 0 for row in values for value in row):
        raise ValueError("V23e parent-final confusion matrix is negative")
    clutter_support = values[1][0] + values[1][1]
    target_support = values[0][0] + values[0][1]
    return {
        "known_target_support": target_support,
        "known_target_true_positive_count": values[0][0],
        "clutter_support": clutter_support,
        "clutter_true_positive_count": values[1][1],
        "clutter_false_positive_count": values[0][1],
        "clutter_false_negative_count": values[1][0],
    }


def _metric_snapshot(metrics: Mapping[str, Any]) -> dict[str, Any]:
    parent = dict(metrics["parent_metrics"])
    event_prefix = dict(metrics["eligible_prefix_metrics"])
    balanced_prefix = dict(metrics["parent_balanced_eligible_prefix_metrics"])
    window = dict(metrics["window_metrics"])
    return {
        "parent_final": parent,
        "eligible_prefix_event_pooled": event_prefix,
        "eligible_prefix_parent_balanced": balanced_prefix,
        "window_event_pooled": window,
    }


def assess_checkpoint(
    evaluation: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Apply the frozen V23e feasibility and lexicographic selector.

    The 8/9 criterion is intentionally encoded as an integer true-positive
    count.  A looser decimal ``0.88`` would silently mean something different
    if the development support ever changed.
    """

    if reference.get("schema_version") != V23E_SELECTOR_SCHEMA:
        raise ValueError("V23e selector reference schema differs")
    if int(epoch) <= 0:
        raise ValueError("V23e candidate epoch must be positive")
    metrics = {
        name: dict(evaluation.get(name) or {})
        for name in (
            "parent_metrics",
            "eligible_prefix_metrics",
            "parent_balanced_eligible_prefix_metrics",
            "window_metrics",
        )
    }
    reference_metrics = dict(reference.get("metrics") or {})
    if not all(metrics.values()) or not all(
        dict(reference_metrics.get(name) or {}) for name in metrics
    ):
        raise ValueError("V23e selector metrics are incomplete")
    finite = _finite_metric_payload(metrics)
    parent = dict(metrics["parent_metrics"])
    balanced = dict(metrics["parent_balanced_eligible_prefix_metrics"])
    reference_parent = dict(reference_metrics["parent_metrics"])
    reference_balanced = dict(reference_metrics["parent_balanced_eligible_prefix_metrics"])
    counts = _parent_final_counts(parent)
    parent_target = _class_row(parent, "KNOWN_TARGET_EVIDENCE")
    parent_clutter = _class_row(parent, "KNOWN_CLUTTER_EVIDENCE")
    balanced_target = _class_row(balanced, "KNOWN_TARGET_EVIDENCE")
    balanced_clutter = _class_row(balanced, "KNOWN_CLUTTER_EVIDENCE")
    reference_parent_target = _class_row(reference_parent, "KNOWN_TARGET_EVIDENCE")
    reference_balanced_target = _class_row(
        reference_balanced, "KNOWN_TARGET_EVIDENCE"
    )
    reference_balanced_clutter = _class_row(
        reference_balanced, "KNOWN_CLUTTER_EVIDENCE"
    )

    constraints = {
        "all_selector_metrics_are_finite": finite,
        "parent_final_clutter_support_is_9": (
            counts["clutter_support"] == V23E_MINIMUM_CLUTTER_PARENT_SUPPORT
            and int(parent_clutter.get("support", -1))
            == V23E_MINIMUM_CLUTTER_PARENT_SUPPORT
        ),
        "parent_final_clutter_true_positives_at_least_8_of_9": (
            counts["clutter_true_positive_count"]
            >= V23E_MINIMUM_CLUTTER_TRUE_POSITIVES
        ),
        "parent_final_clutter_recall_at_least_88_89_percent": (
            float(parent_clutter["recall"])
            + _TOLERANCE
            >= V23E_MINIMUM_CLUTTER_RECALL
        ),
        "parent_final_clutter_precision_at_least_0_90": (
            float(parent_clutter["precision"])
            + _TOLERANCE
            >= V23E_MINIMUM_CLUTTER_PRECISION
        ),
        "parent_final_known_target_recall_non_decreasing_vs_v23a_legacy": (
            float(parent_target["recall"])
            + _TOLERANCE
            >= float(reference_parent_target["recall"])
        ),
        "parent_balanced_prefix_class_support_matches_v23a_legacy": (
            int(balanced_target.get("support", -1))
            == int(reference_balanced_target.get("support", -2))
            and int(balanced_clutter.get("support", -1))
            == int(reference_balanced_clutter.get("support", -2))
        ),
        "parent_balanced_prefix_known_target_recall_non_decreasing_vs_v23a_legacy": (
            float(balanced_target["recall"])
            + _TOLERANCE
            >= float(reference_balanced_target["recall"])
        ),
        "parent_balanced_prefix_clutter_recall_non_decreasing_vs_v23a_legacy": (
            float(balanced_clutter["recall"])
            + _TOLERANCE
            >= float(reference_balanced_clutter["recall"])
        ),
        "parent_balanced_prefix_clutter_f1_non_decreasing_vs_v23a_legacy": (
            float(balanced_clutter["f1"])
            + _TOLERANCE
            >= float(reference_balanced_clutter["f1"])
        ),
    }
    feasible = all(constraints.values())
    ranking_key = (
        [
            float(balanced_clutter["recall"]),
            float(balanced_clutter["f1"]),
            float(parent_clutter["precision"]),
            float(parent_clutter["recall"]),
            float(parent["macro_f1"]),
            -float(epoch),
        ]
        if feasible
        else None
    )
    return {
        "schema_version": V23E_SELECTOR_SCHEMA,
        "epoch": int(epoch),
        "feasible": feasible,
        "constraints": constraints,
        "parent_final_counts": counts,
        "ranking_definition": [
            "parent_balanced_eligible_prefix_clutter_recall_desc",
            "parent_balanced_eligible_prefix_clutter_f1_desc",
            "parent_final_clutter_precision_desc",
            "parent_final_clutter_recall_desc",
            "parent_final_macro_f1_desc",
            "earlier_epoch_asc_for_exact_tie",
        ],
        "ranking_key": ranking_key,
        "metrics": _metric_snapshot(metrics),
    }


def is_better_candidate(
    candidate: Mapping[str, Any], incumbent: Mapping[str, Any] | None
) -> bool:
    """Return a deterministic selector decision for two feasible epochs."""

    if candidate.get("feasible") is not True:
        return False
    if incumbent is None:
        return True
    if incumbent.get("feasible") is not True:
        return True
    candidate_key = candidate.get("ranking_key")
    incumbent_key = incumbent.get("ranking_key")
    if not isinstance(candidate_key, list) or not isinstance(incumbent_key, list):
        raise ValueError("V23e feasible selector candidate lacks a ranking key")
    return tuple(float(value) for value in candidate_key) > tuple(
        float(value) for value in incumbent_key
    )




def select_best_candidate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Select one feasible epoch, rejecting duplicate or unordered input."""

    epochs = [int(candidate.get("epoch", -1)) for candidate in candidates]
    if not epochs or any(epoch <= 0 for epoch in epochs) or len(set(epochs)) != len(epochs):
        raise ValueError("V23e selector candidates must have unique positive epochs")
    best: dict[str, Any] | None = None
    for candidate in sorted((dict(item) for item in candidates), key=lambda item: int(item["epoch"])):
        if is_better_candidate(candidate, best):
            best = candidate
    return best


def frozen_v23a_legacy_reference(root: Path) -> dict[str, Any]:
    """Load the sealed V23a legacy-selected state from its stored hard actions."""

    root = root.resolve()
    artifact = (root / artifact_relative(V23A_VARIANT)).resolve()
    summary_path = artifact / "training-summary.json"
    detail_path = artifact / "sample_detail_information.json"
    checkpoint_path = artifact / "best_model.pt"
    if not all(path.is_file() for path in (summary_path, detail_path, checkpoint_path)):
        raise FileNotFoundError("V23e requires the completed V23a legacy artifact")
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    detail = dict(json.loads(detail_path.read_text(encoding="utf-8")))
    best_epoch = int(summary.get("best_epoch", -1))
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != V23A_VARIANT
        or best_epoch not in range(1, 21)
        or int(detail.get("epoch", -1)) != best_epoch
        or Path(str(detail.get("checkpoint", ""))).resolve() != checkpoint_path
    ):
        raise ValueError("V23a legacy selection artifact is stale or incomplete")
    records = {
        scope: [dict(row) for row in list(dict(detail.get("records") or {}).get(scope) or ())]
        for scope in ("parent", "eligible_prefix", "window")
    }
    metrics = metrics_from_records(records)
    source_epoch_path = artifact / "epochs" / f"epoch-{best_epoch:03d}.pt"
    if not source_epoch_path.is_file():
        raise FileNotFoundError(source_epoch_path)
    return selector_reference(
        metrics,
        epoch=best_epoch,
        source={
            "variant": V23A_VARIANT,
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "legacy_best_checkpoint_path": str(checkpoint_path),
            "legacy_best_checkpoint_sha256": sha256_file(checkpoint_path),
            "epoch_checkpoint_path": str(source_epoch_path),
            "epoch_checkpoint_sha256": sha256_file(source_epoch_path),
            "sample_detail_path": str(detail_path),
            "sample_detail_sha256": sha256_file(detail_path),
        },
    )


def clutter_prefix_action_trajectories(
    records: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Return auditable Clutter causal-action traces without changing metrics."""

    parent_rows = [
        dict(row)
        for row in records.get("parent", ())
        if str(row.get("label_name")) == "ClutterTarget"
    ]
    eligible_rows = [
        dict(row)
        for row in records.get("eligible_prefix", ())
        if str(row.get("label_name")) == "ClutterTarget"
    ]
    result: dict[str, Any] = {}
    for row in parent_rows:
        parent_id = str(row["parent_sample_id"])
        result[parent_id] = {
            "parent_final": row,
            "eligible_prefix_actions": [],
        }
    for row in eligible_rows:
        parent_id = str(row["parent_sample_id"])
        if parent_id not in result:
            raise ValueError("V23e Clutter prefix has no matching parent-final record")
        result[parent_id]["eligible_prefix_actions"].append(row)
    for payload in result.values():
        payload["eligible_prefix_actions"].sort(key=lambda row: int(row["event_index"]))
    if len(result) != V23E_MINIMUM_CLUTTER_PARENT_SUPPORT:
        raise ValueError("V23e Clutter parent support differs from the frozen development set")
    return result


__all__ = ["V23E_MINIMUM_CLUTTER_PARENT_SUPPORT", "V23E_MINIMUM_CLUTTER_PRECISION", "V23E_MINIMUM_CLUTTER_RECALL", "V23E_MINIMUM_CLUTTER_TRUE_POSITIVES", "V23E_SELECTOR_SCHEMA", "assess_checkpoint", "clutter_prefix_action_trajectories", "frozen_v23a_legacy_reference", "is_better_candidate", "metrics_from_records", "select_best_candidate", "selector_reference"]