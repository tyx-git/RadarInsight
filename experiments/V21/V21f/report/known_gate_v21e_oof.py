"""Aggregate frozen V21e KnownGate treatment-only or paired OOF predictions."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate import KNOWN_GATE_CLASS_NAMES, KNOWN_LABELS, summarize_known_gate_records
from radar_ln3.known_gate_v21e_oof import FOLD_COUNT, PREDICTION_SCHEMA, fold_record, load_fold_contract, prediction_relative, sequence_sha256, sha256_file


BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 2104301
PRIMARY_DELTA = 0.02
ABSOLUTE_FLOOR = 0.90


def _load_arm(root: Path, arm: str) -> dict[str, Any]:
    contract, assignment = load_fold_contract(root)
    predictions: list[dict[str, Any]] = []
    records_by_scope = {scope: [] for scope in ("parent", "window", "eligible_prefix")}
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(contract, fold_id)
        path = root / prediction_relative(arm, fold_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        evaluation = dict(payload.get("evaluation") or {})
        records = dict(payload.get("records") or {})
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != PREDICTION_SCHEMA
            or payload.get("arm") != arm
            or int(payload.get("fold_id", -1)) != fold_id
            or payload.get("grouping") != "capture_content_provenance_component"
            or payload.get("prediction_protocol")
            != "out_of_fold_fixed_final_epoch"
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
            or int(payload.get("held_out_prediction_pass_count", -1)) != 1
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or set(records) != set(records_by_scope)
        ):
            raise ValueError(f"V21e OOF {arm} fold {fold_id} prediction is invalid")
        parent_ids = sorted(str(row["parent_sample_id"]) for row in records["parent"])
        if parent_ids != fold["held_out_parent_ids"]:
            raise ValueError(f"V21e OOF {arm} fold {fold_id} held-out set changed")
        history = list(payload.get("fold_training_history") or [])
        if len(history) != 20 or [int(row["epoch"]) for row in history] != list(range(1, 21)):
            raise ValueError(f"V21e OOF {arm} fold {fold_id} lacks 20 epochs")
        for scope in records_by_scope:
            records_by_scope[scope].extend(dict(row) for row in records[scope])
        predictions.append(payload)
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
                "train_parent_count": int(payload["train_scope"]["parent_count"]),
                "held_out_parent_count": len(parent_ids),
                "held_out_label_parent_counts": fold["held_out_label_parent_counts"],
                "parent_metrics": evaluation["parent_metrics"],
                "eligible_prefix_metrics": evaluation["eligible_prefix_metrics"],
                "initial_model_state_sha256": payload["initial_model_state_sha256"],
                "paired_baseline_reference_state_sha256": payload.get(
                    "paired_baseline_reference_state_sha256"
                ),
                "epoch_parent_order_hashes": [
                    str(row["parent_order_sha256"]) for row in history
                ],
            }
        )
    parent_ids = sorted(str(row["parent_sample_id"]) for row in records_by_scope["parent"])
    assignment_ids = sorted(
        str(row["parent_sample_id"]) for row in assignment["assignments"]
    )
    if (
        len(parent_ids) != 986
        or len(set(parent_ids)) != 986
        or parent_ids != assignment_ids
        or sequence_sha256(parent_ids) != assignment["known_parent_ids_sha256"]
    ):
        raise ValueError(f"V21e OOF {arm} does not cover every known train parent once")
    metrics_by_scope = {
        scope: summarize_known_gate_records(rows, scope=f"oof_{arm}_{scope}")
        for scope, rows in records_by_scope.items()
    }
    return {
        "arm": arm,
        "predictions": predictions,
        "fold_summaries": fold_summaries,
        "records_by_scope": records_by_scope,
        "metrics_by_scope": metrics_by_scope,
    }




def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    rows = {str(row["class"]): row for row in metrics["per_class"]}
    target = rows[KNOWN_GATE_CLASS_NAMES[0]]
    clutter = rows[KNOWN_GATE_CLASS_NAMES[1]]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "known_target_precision": float(target["precision"]),
        "known_target_recall": float(target["recall"]),
        "known_target_f1": float(target["f1"]),
        "known_clutter_precision": float(clutter["precision"]),
        "known_clutter_recall": float(clutter["recall"]),
        "known_clutter_f1": float(clutter["f1"]),
        "weak_class_precision_recall_f1_floor": min(
            float(row[field])
            for row in metrics["per_class"]
            for field in ("precision", "recall", "f1")
        ),
    }


def _delta(
    baseline: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, float]:
    before = _metric_vector(baseline)
    after = _metric_vector(treatment)
    return {key: after[key] - before[key] for key in before}


def _record_key(scope: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    parent_id = str(row["parent_sample_id"])
    return (parent_id,) if scope == "parent" else (parent_id, int(row["event_index"]))


def _ordered_pair(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    before = {_record_key(scope, row): dict(row) for row in baseline}
    after = {_record_key(scope, row): dict(row) for row in treatment}
    if len(before) != len(baseline) or len(after) != len(treatment) or set(before) != set(after):
        raise ValueError(f"V21e OOF paired {scope} sets differ or repeat keys")
    ordered = sorted(before)
    for key in ordered:
        if (
            str(before[key]["label_name"]) != str(after[key]["label_name"])
            or int(before[key]["true_id"]) != int(after[key]["true_id"])
        ):
            raise ValueError(f"V21e OOF paired {scope} labels differ")
    return [before[key] for key in ordered], [after[key] for key in ordered]


def _base_label_outcomes(
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment, scope=scope)
    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        indices = [index for index, row in enumerate(before) if row["label_name"] == label]
        changed = rescued = harmed = 0
        for index in indices:
            before_correct = int(before[index]["pred_id"]) == int(before[index]["true_id"])
            after_correct = int(after[index]["pred_id"]) == int(after[index]["true_id"])
            changed += int(int(before[index]["pred_id"]) != int(after[index]["pred_id"]))
            rescued += int((not before_correct) and after_correct)
            harmed += int(before_correct and (not after_correct))
        result[label] = {
            "support": len(indices),
            "changed": changed,
            "rescued": rescued,
            "harmed": harmed,
            "net_rescue": rescued - harmed,
        }
    return result


def _ci(values: Sequence[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _component_bootstrap(
    assignment: Mapping[str, Any],
    baseline: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    before, after = _ordered_pair(baseline, treatment, scope="parent")
    assignment_by_parent = {
        str(row["parent_sample_id"]): dict(row)
        for row in assignment["assignments"]
    }
    component_indices: dict[str, list[int]] = defaultdict(list)
    component_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for index, row in enumerate(before):
        source = assignment_by_parent[str(row["parent_sample_id"])]
        component = str(source["provenance_component_id"])
        component_indices[component].append(index)
        component_label_counts[component][str(row["label_name"])] += 1
    strata: dict[str, list[str]] = defaultdict(list)
    for component, counts in component_label_counts.items():
        signature = "+".join(
            f"{label}:{counts[label]}" for label in KNOWN_LABELS if counts[label]
        )
        strata[signature].append(component)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_components: list[str] = []
        for values in strata.values():
            sampled_components.extend(
                rng.choice(values, size=len(values), replace=True).tolist()
            )
        sampled_indices = [
            index
            for component in sampled_components
            for index in component_indices[str(component)]
        ]
        before_metrics = summarize_known_gate_records(
            [before[index] for index in sampled_indices],
            scope="oof_component_bootstrap_control",
        )
        after_metrics = summarize_known_gate_records(
            [after[index] for index in sampled_indices],
            scope="oof_component_bootstrap_treatment",
        )
        vector = _delta(before_metrics, after_metrics)
        for key, value in vector.items():
            deltas[key].append(float(value))
    return {
        "method": "paired capture/content provenance-component bootstrap stratified by component label signature",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "component_count": len(component_indices),
        "stratum_component_counts": {
            key: len(values) for key, values in sorted(strata.items())
        },
        "treatment_minus_control": {
            key: _ci(values) for key, values in deltas.items()
        },
    }









