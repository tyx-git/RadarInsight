"""Evaluate the V26a frozen post-checkpoint open-set rejection diagnostic.

The only treatment is a threshold on the frozen V26 binary route confidence.
Validation selects a diagnostic operating point under pre-registered known
safety floors.  The test forward, when allowed, happens only after that point
is frozen and is never used to change the threshold.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from build.known_gate_v26 import (
    AUDIT_TEST_CACHE,
    AUDIT_VALIDATION_CACHE,
    EXPECTED_PARENT_COUNTS,
    materialize_test_audit_cache,
    SOURCE_INDEX,
    _selected_index,
)
from build.known_gate_v26a import (
    V26A_BOOTSTRAP_REPLICATES,
    V26A_BOOTSTRAP_SEED,
    V26A_CONTRACT,
    V26A_SELECTION,
    V26A_KNOWN_ACCEPTANCE_FLOOR,
    V26A_KNOWN_CLUTTER_RECALL_FLOOR,
    V26A_KNOWN_TARGET_RECALL_FLOOR,
    V26A_MIN_UNKNOWN_REJECTION_COUNT,
    V26A_SCHEMA_VERSION,
    V26A_THRESHOLD_GRID,
    V26A_VARIANT,
    V26_AUDIT_VALIDATION_CACHE,
    V26_CHECKPOINT,
    V26_CONFIG,
    V26_EXPERIMENT_CONTRACT,
    V26_FEATURE_MANIFEST,
    V26_READINESS,
    V26_REPORT,
    V26_TRAINING_SUMMARY,
)
from build.micro_doppler_v17 import _write_json_new_or_identical
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_LABELS,
    UNKNOWN_LABEL,
    build_known_gate_model_for_config,
    known_gate_device,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import MAX_KNOWN_GATE_FEATURE_CACHE_BYTES
from report.known_gate_v26 import _run_split
from train.known_gate_v21 import _load_checkpoint, _verify_contract, _verify_readiness


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-diagnostic.json"
)
SCAN_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-threshold-scan.csv"
)
PARENT_DETAIL_VALIDATION_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-validation-parent-details.csv"
)
PARENT_DETAIL_TEST_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-test-parent-details.csv"
)
ACTION_NAMES = {
    0: KNOWN_GATE_CLASS_NAMES[0],
    1: KNOWN_GATE_CLASS_NAMES[1],
    -1: "UNKNOWN_REJECTED",
    -2: "PENDING_ABSTAIN",
}
DECISION_COLUMNS = (0, 1, -1, -2)
TRUTH_COLUMNS = (0, 1, -1)
EPS = 1.0e-12
MAX_PERSISTED_REPORT_BYTES = 8 * 1024 * 1024


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > MAX_PERSISTED_REPORT_BYTES:
        raise MemoryError(
            "V26a JSON artifact exceeds its bounded size before write: "
            f"{path}"
        )
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _write_bounded_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if len(text.encode("utf-8")) > MAX_PERSISTED_REPORT_BYTES:
        raise MemoryError(
            "V26a immutable JSON artifact exceeds its bounded size before write: "
            f"{path}"
        )
    _write_json_new_or_identical(path, dict(payload))


def _write_scan_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "threshold",
        "known_parent_support",
        "known_acceptance_rate",
        "known_target_recall",
        "known_clutter_recall",
        "known_clutter_precision",
        "known_clutter_f1",
        "unknown_parent_support",
        "unknown_rejection_count",
        "unknown_rejection_rate",
        "unknown_false_accept_rate",
        "eligible_prefix_known_clutter_recall",
        "safe_known_route",
        "useful_threshold",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    temporary.replace(path)


def _compact_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Drop event-level records before persisting a report.

    The forward pass still keeps records in memory long enough to calculate
    threshold metrics, bootstrap intervals and parent-level changes.  Only
    aggregate metrics and the small changed-parent list belong in the durable
    JSON artifact; persisting every causal window repeatedly caused the first
    V26a report to grow to tens of megabytes.
    """

    result = {
        "threshold": summary.get("threshold"),
        "known_metrics": {
            name: _compact_metric_scope(metrics)
            for name, metrics in dict(summary.get("known_metrics") or {}).items()
        },
        "mixed_action_metrics": {
            name: _compact_mixed_metrics(metrics)
            for name, metrics in dict(summary.get("mixed_action_metrics") or {}).items()
        },
        "open_set_metrics": {
            name: _compact_open_set_metrics(metrics)
            for name, metrics in dict(summary.get("open_set_metrics") or {}).items()
        },
        "decision_counts": dict(summary.get("decision_counts") or {}),
    }
    records = dict(summary.get("records") or {})
    result["record_counts"] = {
        name: len(list(rows or ())) for name, rows in records.items()
    }
    result["changed_record_counts"] = {
        name: sum(bool(row.get("decision_changed")) for row in list(rows or ()))
        for name, rows in records.items()
    }
    result["persisted_event_records"] = False
    result["parent_change_records_persisted"] = True
    result["changed_parent_count"] = len(list(summary.get("changed_parent_records") or ()))
    return result


def _parent_detail_rows(
    split: str,
    summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = list(dict(summary.get("records") or {}).get("parent") or ())
    return [
        {
            "split": split,
            "parent_sample_id": str(row["parent_sample_id"]),
            "label_name": str(row["label_name"]),
            "base_action": str(row["base_action"]),
            "decision_action": str(row["decision_action"]),
            "decision_changed": bool(row["decision_changed"]),
            "confidence": float(row["confidence"]),
            "threshold": float(row["rejection_threshold"]),
            "parent_length": int(row["parent_length"]),
            "rd_frame_count": int(row.get("rd_frame_count", 0)),
            "capture_id": str(row.get("capture_id", "")),
            "track_id": str(row.get("track_id", "")),
            "event_index": int(row.get("event_index", -1)),
        }
        for row in rows
    ]


def _write_parent_detail_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "split",
        "parent_sample_id",
        "label_name",
        "base_action",
        "decision_action",
        "decision_changed",
        "confidence",
        "threshold",
        "parent_length",
        "rd_frame_count",
        "capture_id",
        "track_id",
        "event_index",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows({column: row.get(column) for column in columns} for row in rows)
    temporary.replace(path)


def _assert_report_size(path: Path) -> None:
    size = path.stat().st_size
    if size > MAX_PERSISTED_REPORT_BYTES:
        raise MemoryError(
            "V26a persisted report exceeds its bounded artifact limit: "
            f"{path}={size} > {MAX_PERSISTED_REPORT_BYTES} bytes"
        )


def _class_row(metrics: Mapping[str, Any], class_name: str) -> Mapping[str, Any]:
    for row in list(metrics.get("per_class") or []):
        if str(row.get("class")) == class_name:
            return row
    raise KeyError(f"class is absent from metrics: {class_name}")


def _decision_for_row(row: Mapping[str, Any], threshold: float) -> int:
    if not bool(row.get("product_action_eligible")):
        return -2
    confidence = float(row["confidence"])
    if confidence + EPS < float(threshold):
        return -1
    return int(row["pred_id"])


def _apply_threshold(
    rows: Iterable[Mapping[str, Any]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        base_pred_id = int(row["pred_id"])
        decision_id = _decision_for_row(row, threshold)
        row.update(
            {
                "base_pred_id": base_pred_id,
                "base_action": ACTION_NAMES[base_pred_id],
                "decision_pred_id": decision_id,
                "decision_action": ACTION_NAMES[decision_id],
                "rejection_threshold": float(threshold),
                "rejection_applied": bool(decision_id == -1),
                "pending_applied": bool(decision_id == -2),
                "decision_changed": bool(decision_id != base_pred_id),
            }
        )
        output.append(row)
    return output


def _weighted_rows(rows: Sequence[Mapping[str, Any]]) -> list[tuple[Mapping[str, Any], float]]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        if not bool(row.get("product_action_eligible")):
            raise ValueError("parent-balanced V26a rows must all be eligible")
        parent_id = str(row.get("parent_sample_id", ""))
        if not parent_id:
            raise ValueError("V26a prefix row has no parent_sample_id")
        groups.setdefault(parent_id, []).append(row)
    weighted: list[tuple[Mapping[str, Any], float]] = []
    for parent_id, group in groups.items():
        labels = {str(row.get("label_name", "")) for row in group}
        indices = [int(row.get("event_index", -1)) for row in group]
        if len(labels) != 1 or len(set(indices)) != len(indices) or min(indices) < 0:
            raise ValueError(f"V26a parent prefix identity is inconsistent: {parent_id}")
        weight = 1.0 / len(group)
        weighted.extend((row, weight) for row in group)
    return weighted


def _known_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    parent_balanced: bool = False,
) -> dict[str, Any]:
    selected = [row for row in rows if int(row.get("true_id", -1)) in {0, 1}]
    weighted = _weighted_rows(selected) if parent_balanced else [(row, 1.0) for row in selected]
    matrix = np.zeros((2, 4), dtype=np.float64)
    event_support = np.zeros(2, dtype=np.int64)
    parent_ids: list[set[str]] = [set(), set()]
    for row, weight in weighted:
        true_id = int(row["true_id"])
        decision_id = int(row["decision_pred_id"])
        if decision_id not in DECISION_COLUMNS:
            raise ValueError(f"V26a invalid decision id: {decision_id}")
        column = DECISION_COLUMNS.index(decision_id)
        matrix[true_id, column] += float(weight)
        event_support[true_id] += 1
        parent_ids[true_id].add(str(row["parent_sample_id"]))
    support = np.asarray([len(value) for value in parent_ids], dtype=np.int64)
    total = float(matrix.sum())
    predicted = matrix.sum(axis=0)
    per_class: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(KNOWN_GATE_CLASS_NAMES):
        tp = float(matrix[class_id, class_id])
        precision = tp / float(predicted[class_id]) if predicted[class_id] else 0.0
        true_mass = float(matrix[class_id].sum())
        recall = tp / true_mass if true_mass else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        other = 1 - class_id
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        per_class.append(
            {
                "class": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": int(support[class_id]),
                "event_support": int(event_support[class_id]),
                "weighted_support": true_mass,
                "correct_count_or_mass": tp,
                "false_rejection_count_or_mass": float(
                    matrix[class_id, DECISION_COLUMNS.index(-1)]
                    + matrix[class_id, DECISION_COLUMNS.index(-2)]
                ),
                "major_confusion_to": (
                    KNOWN_GATE_CLASS_NAMES[other]
                    if matrix[class_id, other] > EPS
                    else None
                ),
                "major_confusion_count_or_mass": float(matrix[class_id, other]),
            }
        )
    accepted_mass = float(predicted[0] + predicted[1])
    rejection_mass = float(predicted[DECISION_COLUMNS.index(-1)])
    pending_mass = float(predicted[DECISION_COLUMNS.index(-2)])
    return {
        "scope": scope,
        "aggregation": (
            "parent-balanced eligible prefixes; each parent has total weight one"
            if parent_balanced
            else "unweighted rows"
        ),
        "support": int(support.sum()) if parent_balanced else int(total),
        "event_support": int(len(selected)),
        "weighted_support": total,
        "accuracy": float((matrix[0, 0] + matrix[1, 1]) / total) if total else None,
        "macro_precision": float(np.mean(precision_values)) if per_class else None,
        "macro_recall": float(np.mean(recall_values)) if per_class else None,
        "macro_f1": float(np.mean(f1_values)) if per_class else None,
        "balanced_accuracy": float(np.mean(recall_values)) if per_class else None,
        "known_acceptance_rate": accepted_mass / total if total else None,
        "known_rejection_rate": rejection_mass / total if total else None,
        "pending_rate": pending_mass / total if total else None,
        "decision_columns": [ACTION_NAMES[value] for value in DECISION_COLUMNS],
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_decision": matrix.tolist(),
    }


def _mixed_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    parent_balanced: bool = False,
) -> dict[str, Any]:
    selected = [row for row in rows if int(row.get("true_id", -1)) in {0, 1, -1}]
    weighted = _weighted_rows(selected) if parent_balanced else [(row, 1.0) for row in selected]
    matrix = np.zeros((3, 4), dtype=np.float64)
    parent_ids: list[set[str]] = [set(), set(), set()]
    event_support = np.zeros(3, dtype=np.int64)
    for row, weight in weighted:
        true_id = int(row["true_id"])
        decision_id = int(row["decision_pred_id"])
        if decision_id not in DECISION_COLUMNS:
            raise ValueError(f"V26a invalid mixed decision id: {decision_id}")
        matrix[0 if true_id == 0 else 1 if true_id == 1 else 2, DECISION_COLUMNS.index(decision_id)] += float(weight)
        parent_ids[0 if true_id == 0 else 1 if true_id == 1 else 2].add(str(row["parent_sample_id"]))
        event_support[0 if true_id == 0 else 1 if true_id == 1 else 2] += 1
    total = float(matrix.sum())
    predicted = matrix.sum(axis=0)
    known_class_rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(KNOWN_GATE_CLASS_NAMES):
        tp = float(matrix[class_id, class_id])
        precision = tp / float(predicted[class_id]) if predicted[class_id] else 0.0
        support = float(matrix[class_id].sum())
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        known_class_rows.append(
            {
                "class": class_name,
                "precision_including_unknown_false_accepts": precision,
                "recall_on_known_truth": recall,
                "f1_including_unknown_false_accepts": f1,
                "known_truth_support": int(len(parent_ids[class_id])) if parent_balanced else int(support),
                "unknown_false_accept_mass_into_class": float(matrix[2, class_id]),
                "pending_or_rejected_known_mass": float(matrix[class_id, 2] + matrix[class_id, 3]),
            }
        )
    unknown_tp = float(matrix[2, DECISION_COLUMNS.index(-1)])
    unknown_action_mass = float(predicted[DECISION_COLUMNS.index(-1)])
    unknown_support = float(matrix[2].sum())
    return {
        "scope": scope,
        "aggregation": (
            "parent-balanced eligible prefixes; each parent has total weight one"
            if parent_balanced
            else "unweighted rows"
        ),
        "support": int(sum(len(value) for value in parent_ids)) if parent_balanced else int(total),
        "event_support": int(len(selected)),
        "weighted_support": total,
        "action_columns": [ACTION_NAMES[value] for value in DECISION_COLUMNS],
        "truth_rows": [*KNOWN_GATE_CLASS_NAMES, "UNKNOWN_OPEN_SET_TRUTH"],
        "confusion_matrix_rows_true_columns_action": matrix.tolist(),
        "known_action_metrics": known_class_rows,
        "unknown_rejection_precision": unknown_tp / unknown_action_mass if unknown_action_mass else None,
        "unknown_rejection_recall": unknown_tp / unknown_support if unknown_support else None,
        "mixed_exact_action_rate": float(
            (matrix[0, 0] + matrix[1, 1] + matrix[2, DECISION_COLUMNS.index(-1)]) / total
        ) if total else None,
    }


def _unknown_metrics(rows: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, Any]:
    selected = [row for row in rows if int(row.get("true_id", -1)) == -1]
    eligible = [row for row in selected if bool(row.get("product_action_eligible"))]
    pending = [row for row in selected if int(row["decision_pred_id"]) == -2]
    rejected = [row for row in eligible if int(row["decision_pred_id"]) == -1]
    accepted_target = [row for row in eligible if int(row["decision_pred_id"]) == 0]
    accepted_clutter = [row for row in eligible if int(row["decision_pred_id"]) == 1]
    parent_ids = sorted({str(row["parent_sample_id"]) for row in selected})
    eligible_parent_ids = sorted({str(row["parent_sample_id"]) for row in eligible})
    rejected_parent_ids = sorted({str(row["parent_sample_id"]) for row in rejected})
    all_eligible_by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible:
        all_eligible_by_parent.setdefault(str(row["parent_sample_id"]), []).append(row)
    all_rejected_parent_ids = sorted(
        parent_id
        for parent_id, group in all_eligible_by_parent.items()
        if group and all(int(row["decision_pred_id"]) == -1 for row in group)
    )
    confidence = np.asarray([float(row["confidence"]) for row in selected], dtype=np.float64)
    return {
        "scope": scope,
        "parent_support": len(parent_ids),
        "eligible_parent_support": len(eligible_parent_ids),
        "event_support": len(selected),
        "eligible_event_support": len(eligible),
        "pending_event_count": len(pending),
        "rejected_event_count": len(rejected),
        "accepted_known_target_event_count": len(accepted_target),
        "accepted_known_clutter_event_count": len(accepted_clutter),
        "rejected_parent_count": len(rejected_parent_ids),
        "all_eligible_prefixes_rejected_parent_count": len(all_rejected_parent_ids),
        "unknown_rejection_rate": len(rejected) / len(eligible) if eligible else None,
        "unknown_false_accept_rate": (
            (len(accepted_target) + len(accepted_clutter)) / len(eligible)
            if eligible
            else None
        ),
        "unknown_pending_rate": len(pending) / len(selected) if selected else None,
        "parent_any_rejection_rate": (
            len(rejected_parent_ids) / len(eligible_parent_ids) if eligible_parent_ids else None
        ),
        "parent_all_prefixes_rejected_rate": (
            len(all_rejected_parent_ids) / len(eligible_parent_ids)
            if eligible_parent_ids
            else None
        ),
        "confidence": {
            "support": int(confidence.size),
            "mean": float(confidence.mean()) if confidence.size else None,
            "median": float(np.median(confidence)) if confidence.size else None,
            "minimum": float(confidence.min()) if confidence.size else None,
            "maximum": float(confidence.max()) if confidence.size else None,
            "quantiles": {
                str(q): float(np.quantile(confidence, q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            } if confidence.size else {},
        },
        "parent_ids_rejected": rejected_parent_ids,
        "parent_ids_all_eligible_prefixes_rejected": all_rejected_parent_ids,
    }


def _threshold_summary(
    split_result: Mapping[str, Any],
    *,
    threshold: float,
    include_records: bool = False,
) -> dict[str, Any]:
    raw_records = dict(split_result["records"])
    transformed = {
        name: _apply_threshold(rows, threshold=threshold)
        for name, rows in raw_records.items()
        if name in {"parent", "eligible_prefix", "window"}
    }
    parent = transformed["parent"]
    eligible = transformed["eligible_prefix"]
    window = transformed["window"]
    known_parent = _known_metrics(parent, scope="known_parent_final")
    known_eligible = _known_metrics(eligible, scope="known_eligible_prefix")
    known_parent_balanced = _known_metrics(
        eligible,
        scope="known_parent_balanced_eligible_prefix",
        parent_balanced=True,
    )
    known_window = _known_metrics(window, scope="known_causal_windows")
    unknown_parent = _unknown_metrics(parent, scope="unknown_parent_final")
    unknown_eligible = _unknown_metrics(eligible, scope="unknown_eligible_prefix")
    unknown_window = _unknown_metrics(window, scope="unknown_causal_windows")
    summary: dict[str, Any] = {
        "threshold": float(threshold),
        "known_metrics": {
            "parent_final": known_parent,
            "eligible_prefix": known_eligible,
            "parent_balanced_eligible_prefix": known_parent_balanced,
            "window": known_window,
        },
        "mixed_action_metrics": {
            "parent_final": _mixed_metrics(parent, scope="mixed_parent_final"),
            "parent_balanced_eligible_prefix": _mixed_metrics(
                eligible,
                scope="mixed_parent_balanced_eligible_prefix",
                parent_balanced=True,
            ),
        },
        "open_set_metrics": {
            "parent_final": unknown_parent,
            "eligible_prefix": unknown_eligible,
            "window": unknown_window,
        },
        "decision_counts": {
            action: sum(int(row["decision_action"] == action) for row in parent)
            for action in ACTION_NAMES.values()
        },
    }
    if include_records:
        summary["records"] = {
            "parent": parent,
            "eligible_prefix": eligible,
            "window": window,
        }
        summary["changed_parent_records"] = [
            row for row in parent if bool(row["decision_changed"])
        ]
    return summary


def _compact_scan_row(summary: Mapping[str, Any], *, safety: Mapping[str, Any]) -> dict[str, Any]:
    parent = summary["known_metrics"]["parent_final"]
    target = _class_row(parent, KNOWN_GATE_CLASS_NAMES[0])
    clutter = _class_row(parent, KNOWN_GATE_CLASS_NAMES[1])
    prefix_clutter = _class_row(
        summary["known_metrics"]["parent_balanced_eligible_prefix"],
        KNOWN_GATE_CLASS_NAMES[1],
    )
    unknown = summary["open_set_metrics"]["parent_final"]
    safe = bool(
        float(parent["known_acceptance_rate"]) + EPS >= V26A_KNOWN_ACCEPTANCE_FLOOR
        and float(target["recall"]) + EPS >= V26A_KNOWN_TARGET_RECALL_FLOOR
        and float(clutter["recall"]) + EPS >= V26A_KNOWN_CLUTTER_RECALL_FLOOR
    )
    useful = bool(safe and int(unknown["rejected_parent_count"]) >= V26A_MIN_UNKNOWN_REJECTION_COUNT)
    return {
        "threshold": float(summary["threshold"]),
        "known_parent_support": int(parent["support"]),
        "known_acceptance_rate": float(parent["known_acceptance_rate"]),
        "known_target_recall": float(target["recall"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_f1": float(clutter["f1"]),
        "unknown_parent_support": int(unknown["parent_support"]),
        "unknown_rejection_count": int(unknown["rejected_parent_count"]),
        "unknown_rejection_rate": float(unknown["unknown_rejection_rate"])
        if unknown["unknown_rejection_rate"] is not None
        else None,
        "unknown_false_accept_rate": float(unknown["unknown_false_accept_rate"])
        if unknown["unknown_false_accept_rate"] is not None
        else None,
        "eligible_prefix_known_clutter_recall": float(prefix_clutter["recall"]),
        "safe_known_route": safe,
        "useful_threshold": useful,
        "selection_safety_contract": dict(safety),
    }


def _select_threshold(scan: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    safe = [row for row in scan if bool(row["safe_known_route"])]
    useful = [row for row in scan if bool(row["useful_threshold"])]
    if useful:
        selected = sorted(
            useful,
            key=lambda row: (
                -int(row["unknown_rejection_count"]),
                -float(row["known_acceptance_rate"]),
                float(row["threshold"]),
            ),
        )[0]
        return {
            "status": "useful_threshold_found",
            "selected_threshold": float(selected["threshold"]),
            "selected_scan_row": dict(selected),
            "safe_threshold_count": len(safe),
            "useful_threshold_count": len(useful),
            "selection_used_test": False,
        }
    if safe:
        # A stress operating point is still useful when the diagnostic finds
        # no threshold with meaningful Unknown rejection: choose the highest
        # safe threshold to expose the strongest allowed trade-off, but mark
        # the experiment as failed to find a useful rejection rule.
        selected = sorted(safe, key=lambda row: float(row["threshold"]), reverse=True)[0]
        return {
            "status": "no_useful_threshold",
            "selected_threshold": float(selected["threshold"]),
            "selected_scan_row": dict(selected),
            "safe_threshold_count": len(safe),
            "useful_threshold_count": 0,
            "selection_used_test": False,
        }
    return {
        "status": "no_safe_threshold",
        "selected_threshold": None,
        "selected_scan_row": None,
        "safe_threshold_count": 0,
        "useful_threshold_count": 0,
        "selection_used_test": False,
    }


def _rank_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    negative = labels == 0
    n_positive = int(positive.sum())
    n_negative = int(negative.sum())
    if not n_positive or not n_negative:
        return None
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[positive].sum())
    return (positive_rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)


def _score_separation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [row for row in rows if int(row.get("true_id", -1)) in {0, 1, -1}]
    labels = np.asarray([int(int(row["true_id"]) == -1) for row in selected], dtype=np.int64)
    scores = np.asarray([1.0 - float(row["confidence"]) for row in selected], dtype=np.float64)
    known_conf = np.asarray(
        [float(row["confidence"]) for row in selected if int(row["true_id"]) in {0, 1}],
        dtype=np.float64,
    )
    unknown_conf = np.asarray(
        [float(row["confidence"]) for row in selected if int(row["true_id"]) == -1],
        dtype=np.float64,
    )
    return {
        "score_name": "one_minus_max_known_route_confidence",
        "positive_class": "UnknownTarget",
        "auroc": _rank_auc(labels, scores),
        "known_confidence": _distribution(known_conf),
        "unknown_confidence": _distribution(unknown_conf),
        "unknown_support": int(unknown_conf.size),
        "known_support": int(known_conf.size),
    }


def _distribution(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "support": int(values.size),
        "mean": float(values.mean()) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "minimum": float(values.min()) if values.size else None,
        "maximum": float(values.max()) if values.size else None,
        "quantiles": {
            str(q): float(np.quantile(values, q))
            for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        } if values.size else {},
    }


def _bootstrap_selected(
    summary: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    records = list(summary["records"]["parent"])
    known = {
        label: [row for row in records if str(row["label_name"]) == label]
        for label in KNOWN_LABELS
    }
    unknown = [row for row in records if str(row["label_name"]) == UNKNOWN_LABEL]
    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {
        "known_target_recall": [],
        "known_clutter_recall": [],
        "known_clutter_precision": [],
        "known_clutter_f1": [],
        "known_acceptance_rate": [],
        "unknown_rejection_rate": [],
    }
    for _ in range(V26A_BOOTSTRAP_REPLICATES):
        sampled_known: list[dict[str, Any]] = []
        for label in (*KNOWN_LABELS[:-1], KNOWN_LABELS[-1]):
            group = known[label]
            indices = rng.integers(0, len(group), size=len(group))
            sampled_known.extend(dict(group[int(index)]) for index in indices)
        known_metrics = _known_metrics(sampled_known, scope="bootstrap")
        target = _class_row(known_metrics, KNOWN_GATE_CLASS_NAMES[0])
        clutter = _class_row(known_metrics, KNOWN_GATE_CLASS_NAMES[1])
        unknown_indices = rng.integers(0, len(unknown), size=len(unknown))
        sampled_unknown = [dict(unknown[int(index)]) for index in unknown_indices]
        unknown_metrics = _unknown_metrics(sampled_unknown, scope="bootstrap")
        values["known_target_recall"].append(float(target["recall"]))
        values["known_clutter_recall"].append(float(clutter["recall"]))
        values["known_clutter_precision"].append(float(clutter["precision"]))
        values["known_clutter_f1"].append(float(clutter["f1"]))
        values["known_acceptance_rate"].append(float(known_metrics["known_acceptance_rate"]))
        values["unknown_rejection_rate"].append(float(unknown_metrics["unknown_rejection_rate"]))
    return {
        "method": "stratified parent bootstrap by known label plus Unknown parent bootstrap",
        "replicates": V26A_BOOTSTRAP_REPLICATES,
        "seed": seed,
        "confidence_level": 0.95,
        "known_stratum_support": {label: len(rows) for label, rows in known.items()},
        "unknown_parent_support": len(unknown),
        "metrics": {
            name: {
                "median": float(np.median(data)),
                "lower_95": float(np.quantile(data, 0.025)),
                "upper_95": float(np.quantile(data, 0.975)),
            }
            for name, data in values.items()
        },
    }


def _strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        capture = str(row.get("capture_id") or "<missing_capture>")
        groups.setdefault(capture, []).append(row)
    result: dict[str, Any] = {}
    for capture, group in sorted(groups.items()):
        known = [row for row in group if int(row.get("true_id", -1)) in {0, 1}]
        unknown = [row for row in group if int(row.get("true_id", -1)) == -1]
        result[capture] = {
            "parent_support": len(group),
            "known_parent_support": len(known),
            "unknown_parent_support": len(unknown),
            "known_rejected_parent_count": sum(int(row["decision_pred_id"] == -1) for row in known),
            "known_pending_parent_count": sum(int(row["decision_pred_id"] == -2) for row in known),
            "unknown_rejected_parent_count": sum(int(row["decision_pred_id"] == -1) for row in unknown),
            "unknown_pending_parent_count": sum(int(row["decision_pred_id"] == -2) for row in unknown),
            "sequence_length_min": min(int(row["parent_length"]) for row in group) if group else None,
            "sequence_length_max": max(int(row["parent_length"]) for row in group) if group else None,
            "rd_frame_count_distribution": {
                str(value): sum(int(int(row.get("rd_frame_count", 0)) == value) for row in group)
                for value in sorted({int(row.get("rd_frame_count", 0)) for row in group})
            },
        }
    return result


def _capture_strata_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep only the capture-level evidence needed in the main report.

    Full capture rows are intentionally not embedded in JSON.  Parent-level
    detail remains available in the split-specific CSV artifacts.
    """

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row.get("capture_id") or "<missing_capture>"), []).append(row)
    unknown_captures: list[dict[str, Any]] = []
    known_parent_count = 0
    unknown_parent_count = 0
    known_rejected_parent_count = 0
    unknown_rejected_parent_count = 0
    for capture_id, group in sorted(groups.items()):
        parent_rows = {
            str(row["parent_sample_id"]): row
            for row in group
            if int(row.get("event_index", -1)) == int(row.get("parent_length", 0)) - 1
        }
        # Parent-level inputs are used by the persisted report.  The fallback
        # keeps this helper correct for callers that provide one row per parent
        # without a conventional final event index.
        if not parent_rows:
            parent_rows = {str(row["parent_sample_id"]): row for row in group}
        known = [row for row in parent_rows.values() if int(row.get("true_id", -1)) in {0, 1}]
        unknown = [row for row in parent_rows.values() if int(row.get("true_id", -1)) == -1]
        known_rejected = sum(int(row.get("decision_pred_id", -99) == -1) for row in known)
        unknown_rejected = sum(int(row.get("decision_pred_id", -99) == -1) for row in unknown)
        known_parent_count += len(known)
        unknown_parent_count += len(unknown)
        known_rejected_parent_count += known_rejected
        unknown_rejected_parent_count += unknown_rejected
        if unknown:
            unknown_captures.append(
                {
                    "capture_id": capture_id,
                    "unknown_parent_count": len(unknown),
                    "unknown_rejected_parent_count": unknown_rejected,
                    "unknown_false_accept_parent_count": len(unknown) - unknown_rejected,
                }
            )
    return {
        "capture_count": len(groups),
        "known_parent_count": known_parent_count,
        "unknown_parent_count": unknown_parent_count,
        "captures_with_unknown_count": len(unknown_captures),
        "known_rejected_parent_count": known_rejected_parent_count,
        "unknown_rejected_parent_count": unknown_rejected_parent_count,
        "unknown_captures": unknown_captures,
        "detail_available_in_parent_csv": True,
    }


def _compact_metric_scope(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    result: dict[str, Any] = {
        key: metrics[key]
        for key in (
            "scope",
            "aggregation",
            "support",
            "event_support",
            "weighted_support",
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
            "known_acceptance_rate",
            "known_rejection_rate",
            "pending_rate",
        )
        if key in metrics
    }
    result["per_class"] = [
        {
            key: row[key]
            for key in (
                "class",
                "precision",
                "recall",
                "f1",
                "support",
                "event_support",
                "weighted_support",
            )
            if key in row
        }
        for row in list(metrics.get("per_class") or [])
    ]
    return result


def _compact_mixed_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    return {
        key: metrics[key]
        for key in (
            "scope",
            "aggregation",
            "support",
            "event_support",
            "weighted_support",
            "known_action_metrics",
            "unknown_rejection_precision",
            "unknown_rejection_recall",
            "mixed_exact_action_rate",
        )
        if key in metrics
    }


def _compact_open_set_metrics(metrics: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if metrics is None:
        return None
    confidence = dict(metrics.get("confidence") or {})
    return {
        key: metrics[key]
        for key in (
            "scope",
            "parent_support",
            "eligible_parent_support",
            "event_support",
            "eligible_event_support",
            "pending_event_count",
            "rejected_event_count",
            "rejected_parent_count",
            "all_eligible_prefixes_rejected_parent_count",
            "unknown_rejection_rate",
            "unknown_false_accept_rate",
            "unknown_pending_rate",
            "parent_any_rejection_rate",
            "parent_all_prefixes_rejected_rate",
        )
        if key in metrics
    }
    result["confidence"] = {
        key: confidence[key]
        for key in ("support", "mean", "median", "minimum", "maximum")
        if key in confidence
    }
    return result


def _compact_score_separation(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        key: value[key]
        for key in ("score_name", "positive_class", "auroc", "known_support", "unknown_support")
        if key in value
    }
    for name in ("known_confidence", "unknown_confidence"):
        distribution = dict(value.get(name) or {})
        result[name] = {
            key: distribution[key]
            for key in ("support", "mean", "median", "minimum", "maximum")
            if key in distribution
        }
    return result


def _preflight_test_cache_materialization(root: Path, config: Any) -> dict[str, Any]:
    """Bound the index-derived allocation before V26 builds the test NPZ."""

    selected = _selected_index(root / SOURCE_INDEX, "test", include_unknown=True)
    expected_parent_count = sum(EXPECTED_PARENT_COUNTS["test"].values())
    if len(selected) > expected_parent_count * int(config.data.max_sequence_length):
        raise MemoryError(
            "V26a test cache event count exceeds the bounded sequence allocation: "
            f"{len(selected)} > {expected_parent_count * int(config.data.max_sequence_length)}"
        )
    lengths = selected.groupby("parent_sample_id", sort=False).size()
    maximum_length = int(lengths.max()) if not lengths.empty else 0
    if maximum_length > int(config.data.max_sequence_length):
        raise MemoryError(
            "V26a test parent sequence exceeds max_sequence_length: "
            f"{maximum_length} > {config.data.max_sequence_length}"
        )
    return {
        "event_count": int(len(selected)),
        "parent_count": int(lengths.size),
        "max_parent_length": maximum_length,
        "event_allocation_bound": int(
            expected_parent_count * int(config.data.max_sequence_length)
        ),
    }


def _source_and_model(root: Path, contract: Mapping[str, Any]) -> tuple[Any, Any, torch.device, dict[str, Any]]:
    source = dict(contract["source_v26"])
    config_path = root / V26_CONFIG
    readiness_path = root / V26_READINESS
    checkpoint_path = root / V26_CHECKPOINT
    summary_path = root / V26_TRAINING_SUMMARY
    for key, path in {
        "config": config_path,
        "readiness": readiness_path,
        "checkpoint": checkpoint_path,
        "training_summary": summary_path,
    }.items():
        expected = source[f"{key}_sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"V26a source {key} hash changed: {path}")
    config = load_known_gate_config(config_path)
    contract_evidence = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract_evidence)
    summary = _load_json(summary_path)
    if int(summary.get("best_epoch", -1)) != int(source["checkpoint_epoch"]):
        raise ValueError("V26a source checkpoint epoch changed")
    model, reference, pairing = build_known_gate_model_for_config(root, config)
    del reference, pairing
    device = known_gate_device(config)
    model.to(device)
    _load_checkpoint(
        checkpoint_path,
        model,
        None,
        config=config,
        config_path=config_path,
        readiness_sha256=readiness["sha256"],
        map_location=device,
    )
    return model, config, device, {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": int(source["checkpoint_epoch"]),
        "readiness_sha256": readiness["sha256"],
    }


def _validate_contract(root: Path) -> tuple[dict[str, Any], Path]:
    contract_path = root / V26A_CONTRACT
    if not contract_path.is_file():
        raise FileNotFoundError(
            f"V26a contract is missing; run build.known_gate_v26a: {contract_path}"
        )
    contract = _load_json(contract_path)
    if (
        contract.get("schema_version") != V26A_SCHEMA_VERSION
        or contract.get("internal_variant") != V26A_VARIANT
    ):
        raise ValueError("V26a contract identity is invalid")
    source = contract["source_v26"]
    checks = (
        (V26_REPORT, source["development_report_sha256"]),
        (V26_EXPERIMENT_CONTRACT, source["experiment_contract_sha256"]),
        (V26_FEATURE_MANIFEST, source["feature_manifest_sha256"]),
        (V26_AUDIT_VALIDATION_CACHE, source["validation_audit_cache_sha256"]),
    )
    for relative, expected in checks:
        path = root / relative
        if relative == V26_AUDIT_VALIDATION_CACHE and path.stat().st_size > MAX_KNOWN_GATE_FEATURE_CACHE_BYTES:
            raise MemoryError(
                "V26a validation feature cache exceeds its fail-closed limit: "
                f"{path.stat().st_size} > {MAX_KNOWN_GATE_FEATURE_CACHE_BYTES}"
            )
        if sha256_file(path) != expected:
            raise ValueError(f"V26a source hash changed: {relative}")
    return contract, contract_path


def _validation_phase(root: Path) -> dict[str, Any]:
    contract, contract_path = _validate_contract(root)
    model, config, device, source_identity = _source_and_model(root, contract)
    validation = _run_split(
        model,
        config,
        root,
        split="val",
        cache_path=root / AUDIT_VALIDATION_CACHE,
        device=device,
        minimum_evidence_events=config.train.prefix_auxiliary_min_events,
    )
    control_summary = _threshold_summary(
        validation,
        threshold=0.0,
        include_records=False,
    )
    safety = {
        "known_acceptance_floor": V26A_KNOWN_ACCEPTANCE_FLOOR,
        "known_target_recall_floor": V26A_KNOWN_TARGET_RECALL_FLOOR,
        "known_clutter_recall_floor": V26A_KNOWN_CLUTTER_RECALL_FLOOR,
        "minimum_unknown_rejection_count": V26A_MIN_UNKNOWN_REJECTION_COUNT,
    }
    scan_rows: list[dict[str, Any]] = []
    for threshold in V26A_THRESHOLD_GRID:
        summary = _threshold_summary(
            validation,
            threshold=threshold,
            include_records=False,
        )
        scan_rows.append(_compact_scan_row(summary, safety=safety))
    selection = _select_threshold(scan_rows)
    selected_threshold = selection["selected_threshold"]
    selected_validation = (
        _threshold_summary(
            validation,
            threshold=float(selected_threshold),
            include_records=True,
        )
        if selected_threshold is not None
        else None
    )
    selected_validation_bootstrap = (
        _bootstrap_selected(selected_validation, seed=V26A_BOOTSTRAP_SEED)
        if selected_validation is not None
        else None
    )
    selected_validation_strata = (
        _capture_strata_summary(selected_validation["records"]["parent"])
        if selected_validation is not None
        else {}
    )
    _write_parent_detail_csv(
        root / PARENT_DETAIL_VALIDATION_CSV,
        _parent_detail_rows("validation", selected_validation)
        if selected_validation is not None
        else [],
    )
    compact_control = _compact_summary(control_summary)
    compact_selected = (
        _compact_summary(selected_validation)
        if selected_validation is not None
        else None
    )
    _write_scan_csv(root / SCAN_CSV, scan_rows)
    selection_payload = {
        "status": "completed_validation_threshold_selection",
        "schema_version": V26A_SCHEMA_VERSION,
        "variant": V26A_VARIANT,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "source_v26": source_identity,
        "threshold_scan": {
            "control": compact_control,
            "csv": str(root / SCAN_CSV),
            "csv_sha256": sha256_file(root / SCAN_CSV),
            "threshold_count": len(scan_rows),
            "rows_embedded_in_json": False,
        },
        "selection": selection,
        "selection_used_test": False,
        "validation": {
            "parent_support": validation["parent_count"],
            "label_counts": validation["label_counts"],
            "score_separation": _compact_score_separation(
                _score_separation(validation["records"]["parent"])
            ),
            "control": compact_control,
            "selected": compact_selected,
            "selected_bootstrap": selected_validation_bootstrap,
            "selected_capture_strata": selected_validation_strata,
            "parent_detail_csv": str(root / PARENT_DETAIL_VALIDATION_CSV),
            "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_VALIDATION_CSV),
        },
        "test": {
            "status": "not_executed",
            "access_count": 0,
            "used_for_threshold_selection": False,
        },
        "access_audit": {
            "validation_forward_count": 1,
            "test_forward_count": 0,
            "test_used_for_threshold_selection": False,
            "test_payload_access_count_before_selection": 0,
        },
        "semantics": {
            "unknown_is_model_class": False,
            "score_is_true_knownness_probability": False,
            "pending_is_not_unknown_rejection": True,
            "local_threshold_is_not_formal_l3_calibration": True,
        },
        "cache_safety": {
            "rd_tensor_cache_size": 1,
            "dataloader_num_workers": 0,
            "feature_cache_preflight": "fail_closed_archive_and_shape_validation",
            "rd_archive_preflight": "fail_closed_archive_member_and_shape_validation",
            "persisted_event_records": False,
            "max_persisted_json_bytes": MAX_PERSISTED_REPORT_BYTES,
        },
    }
    _write_bounded_immutable_json(root / V26A_SELECTION, selection_payload)
    # Keep the human-facing report explicitly partial until the separate test
    # phase consumes this immutable selection artifact.
    partial_report = {
        **selection_payload,
        "report_status": "validation_only_threshold_selection; test_not_executed",
        "test": {"status": "not_executed", "access_count": 0, "used_for_threshold_selection": False},
        "OOF": "not_executed",
        "full_cascade": "not_executed",
        "formal_rejection": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, partial_report)
    _assert_report_size(root / OUTPUT)
    return {
        "status": selection_payload["status"],
        "selection": str(root / V26A_SELECTION),
        "selection_sha256": sha256_file(root / V26A_SELECTION),
        "report": str(root / OUTPUT),
        "selected_threshold": selected_threshold,
        "selection_status": selection["status"],
        "test": "not_executed",
    }


def _load_selection(root: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    contract, contract_path = _validate_contract(root)
    selection_path = root / V26A_SELECTION
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"V26a selection artifact is missing; run --phase validation: {selection_path}"
        )
    _assert_report_size(selection_path)
    selection = _load_json(selection_path)
    if (
        selection.get("schema_version") != V26A_SCHEMA_VERSION
        or selection.get("variant") != V26A_VARIANT
        or selection.get("contract_sha256") != sha256_file(contract_path)
        or selection.get("selection_used_test") is not False
        or selection.get("access_audit", {}).get("test_used_for_threshold_selection") is not False
    ):
        raise ValueError("V26a selection artifact is stale or used test data")
    source = selection.get("source_v26") or {}
    if source.get("checkpoint_sha256") != contract["source_v26"]["checkpoint_sha256"]:
        raise ValueError("V26a selection checkpoint identity changed")
    return selection, contract_path, contract


def _test_phase(root: Path) -> dict[str, Any]:
    selection, contract_path, contract = _load_selection(root)
    selected_threshold = selection["selection"].get("selected_threshold")
    if selected_threshold is None:
        final = {
            **selection,
            "status": "completed_frozen_open_set_rejection_diagnostic",
            "report_status": "no_safe_threshold; test_not_executed",
            "test": {"status": "not_executed_no_safe_threshold", "access_count": 0, "used_for_threshold_selection": False},
            "access_audit": {
                **selection["access_audit"],
                "test_forward_count": 0,
                "test_access_after_threshold_selection": False,
            },
            "OOF": "not_executed",
            "full_cascade": "not_executed",
            "formal_rejection": "not_executed",
            "stable": "not_executed",
            "P95": "not_executed",
            "deployment_eligible": False,
        }
        _write_json(root / OUTPUT, final)
        _assert_report_size(root / OUTPUT)
        return {
            "status": final["status"],
            "report": str(root / OUTPUT),
            "selected_threshold": None,
            "test": "not_executed_no_safe_threshold",
        }
    model, config, device, source_identity = _source_and_model(root, contract)
    # The selection artifact is immutable and has already captured the
    # validation decision.  Bound the index-derived allocation before the
    # mixed test cache is materialized.
    test_cache_preflight = _preflight_test_cache_materialization(root, config)
    test_cache_path, test_cache_stats = materialize_test_audit_cache(root)
    if test_cache_path.stat().st_size > MAX_KNOWN_GATE_FEATURE_CACHE_BYTES:
        raise MemoryError(
            "V26a materialized test feature cache exceeds its fail-closed limit: "
            f"{test_cache_path.stat().st_size} > {MAX_KNOWN_GATE_FEATURE_CACHE_BYTES}"
        )
    test = _run_split(
        model,
        config,
        root,
        split="test",
        cache_path=test_cache_path,
        device=device,
        minimum_evidence_events=config.train.prefix_auxiliary_min_events,
    )
    test_control = _threshold_summary(test, threshold=0.0, include_records=False)
    selected_test = _threshold_summary(
        test,
        threshold=float(selected_threshold),
        include_records=True,
    )
    selected_test_bootstrap = _bootstrap_selected(
        selected_test,
        seed=V26A_BOOTSTRAP_SEED + 1,
    )
    selected_test_strata = _capture_strata_summary(selected_test["records"]["parent"])
    _write_parent_detail_csv(
        root / PARENT_DETAIL_TEST_CSV,
        _parent_detail_rows("test", selected_test),
    )
    compact_test_control = _compact_summary(test_control)
    compact_selected_test = _compact_summary(selected_test)
    final = {
        **selection,
        "status": "completed_frozen_open_set_rejection_diagnostic",
        "report_status": "test_audit_completed_after_immutable_validation_selection",
        "source_v26": source_identity,
        "test": {
            "status": "completed_after_threshold_freeze",
            "access_count": 1,
            "cache_statistics": test_cache_stats,
            "cache_materialization_preflight": test_cache_preflight,
            "cache_size_bytes": int(test_cache_path.stat().st_size),
            "parent_support": test["parent_count"],
            "label_counts": test["label_counts"],
            "score_separation": _compact_score_separation(
                _score_separation(test["records"]["parent"])
            ),
            "control": compact_test_control,
            "selected": compact_selected_test,
            "selected_bootstrap": selected_test_bootstrap,
            "selected_capture_strata": selected_test_strata,
            "parent_detail_csv": str(root / PARENT_DETAIL_TEST_CSV),
            "parent_detail_csv_sha256": sha256_file(root / PARENT_DETAIL_TEST_CSV),
            "used_for_threshold_selection": False,
        },
        "access_audit": {
            **selection["access_audit"],
            "test_forward_count": 1,
            "test_access_after_threshold_selection": True,
            "test_used_for_threshold_selection": False,
            "test_payload_access_count_before_selection": 0,
        },
        "OOF": "not_executed",
        "full_cascade": "not_executed",
        "formal_rejection": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, final)
    _assert_report_size(root / OUTPUT)
    return {
        "status": final["status"],
        "report": str(root / OUTPUT),
        "report_sha256": sha256_file(root / OUTPUT),
        "selected_threshold": selected_threshold,
        "test": "completed_after_threshold_freeze",
    }


def run(project_root: Path, *, phase: str = "all") -> dict[str, Any]:
    if phase not in {"validation", "test", "all"}:
        raise ValueError(f"unsupported V26a phase: {phase}")
    root = project_root.resolve()
    if phase == "validation":
        return _validation_phase(root)
    if phase == "test":
        return _test_phase(root)
    _validation_phase(root)
    return _test_phase(root)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V26a frozen KnownGate open-set rejection.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--phase", choices=("validation", "test", "all"), default="all")
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), phase=args.phase), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
