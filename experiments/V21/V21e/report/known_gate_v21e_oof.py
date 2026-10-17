"""Aggregate frozen V21e KnownGate treatment-only or paired OOF predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_LABELS,
    summarize_known_gate_records,
)
from radar_ln3.known_gate_v21e_oof import FOLD_COUNT, PAIRED_REPORT_RELATIVE, PAIRED_REPORT_SCHEMA, PREDICTION_SCHEMA, TREATMENT_REPORT_RELATIVE, TREATMENT_REPORT_SCHEMA, fold_record, load_fold_contract, prediction_relative, sequence_sha256, sha256_file


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


def _base_label_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        selected = [row for row in records if str(row["label_name"]) == label]
        expected = KNOWN_GATE_LABEL_TO_ID[label]
        correct = sum(int(row["pred_id"]) == expected for row in selected)
        result[label] = {
            "support": len(selected),
            "correct_parent_count": int(correct),
            "acceptance_recall": float(correct / len(selected)) if selected else None,
        }
    return result


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


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def treatment_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    treatment = _load_arm(root, "treatment")
    parent = treatment["metrics_by_scope"]["parent"]
    vector = _metric_vector(parent)
    absolute_guard = bool(
        vector["weak_class_precision_recall_f1_floor"] >= ABSOLUTE_FLOOR
    )
    report = {
        "status": "passed" if absolute_guard else "completed_absolute_guard_failed",
        "schema_version": TREATMENT_REPORT_SCHEMA,
        "experiment": "V21e KnownGate treatment-first OOF futility gate",
        "scope": "known train capture/content-grouped OOF; no validation/test/Unknown",
        "metrics_by_scope": treatment["metrics_by_scope"],
        "base_label_parent_metrics": _base_label_metrics(
            treatment["records_by_scope"]["parent"]
        ),
        "fold_summaries": treatment["fold_summaries"],
        "absolute_guard": {
            "threshold": ABSOLUTE_FLOOR,
            "observed_weak_class_precision_recall_f1_floor": vector[
                "weak_class_precision_recall_f1_floor"
            ],
            "passed": absolute_guard,
        },
        "advance_to_control_training": absolute_guard,
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / TREATMENT_REPORT_RELATIVE
    _write(output, report)
    report["report_path"] = str(output)
    return report


def paired_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract, assignment = load_fold_contract(root)
    control = _load_arm(root, "control")
    treatment = _load_arm(root, "treatment")
    for fold_id in range(FOLD_COUNT):
        before = control["fold_summaries"][fold_id]
        after = treatment["fold_summaries"][fold_id]
        if (
            before["held_out_parent_count"] != after["held_out_parent_count"]
            or before["epoch_parent_order_hashes"] != after["epoch_parent_order_hashes"]
            or before["initial_model_state_sha256"]
            != after["paired_baseline_reference_state_sha256"]
        ):
            raise ValueError(f"V21e OOF fold {fold_id} pair is not initialization/order matched")
    deltas = {
        scope: _delta(
            control["metrics_by_scope"][scope],
            treatment["metrics_by_scope"][scope],
        )
        for scope in ("parent", "window", "eligible_prefix")
    }
    parent_outcomes = _base_label_outcomes(
        control["records_by_scope"]["parent"],
        treatment["records_by_scope"]["parent"],
        scope="parent",
    )
    prefix_outcomes = _base_label_outcomes(
        control["records_by_scope"]["eligible_prefix"],
        treatment["records_by_scope"]["eligible_prefix"],
        scope="eligible_prefix",
    )
    bootstrap = _component_bootstrap(
        assignment,
        control["records_by_scope"]["parent"],
        treatment["records_by_scope"]["parent"],
    )
    treatment_vector = _metric_vector(treatment["metrics_by_scope"]["parent"])
    absolute_guard = bool(
        treatment_vector["weak_class_precision_recall_f1_floor"] >= ABSOLUTE_FLOOR
    )
    direction_safe = bool(
        deltas["parent"]["known_target_recall"] >= 0.0
        and deltas["parent"]["known_clutter_recall"] >= 0.0
        and all(int(parent_outcomes[label]["net_rescue"]) >= 0 for label in KNOWN_LABELS)
    )
    primary = bool(deltas["parent"]["macro_f1"] > PRIMARY_DELTA)
    robust = bool(
        float(bootstrap["treatment_minus_control"]["macro_f1"]["lower_95"])
        > 0.0
    )
    prefix_guard = bool(
        deltas["eligible_prefix"]["macro_f1"] >= 0.0
        and deltas["eligible_prefix"]["known_target_recall"] >= 0.0
        and deltas["eligible_prefix"]["known_clutter_recall"] >= 0.0
    )
    parent_confirmed = absolute_guard and primary and direction_safe and robust
    report = {
        "status": "passed" if parent_confirmed else "completed_but_not_confirmed",
        "schema_version": PAIRED_REPORT_SCHEMA,
        "experiment": "V21e KnownGate paired capture/content-grouped five-fold OOF",
        "scope": "known train OOF only; Unknown/validation/test excluded",
        "fold_contract_path": str(root / "data/processed/RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-contract.json"),
        "fold_contract_sha256": sha256_file(
            root / "data/processed/RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-contract.json"
        ),
        "control_metrics_by_scope": control["metrics_by_scope"],
        "treatment_metrics_by_scope": treatment["metrics_by_scope"],
        "treatment_minus_control": deltas,
        "parent_base_label_outcomes": parent_outcomes,
        "eligible_prefix_base_label_outcomes": prefix_outcomes,
        "paired_component_bootstrap": bootstrap,
        "fold_pairing": [
            {
                "fold_id": fold_id,
                "initial_shared_state_equal": True,
                "all_20_epoch_parent_orders_equal": True,
                "held_out_parent_count": control["fold_summaries"][fold_id][
                    "held_out_parent_count"
                ],
            }
            for fold_id in range(FOLD_COUNT)
        ],
        "decision": {
            "treatment_absolute_guard_passed": absolute_guard,
            "primary_macro_f1_delta_strictly_above_0_02": primary,
            "parent_direction_safety_passed": direction_safe,
            "component_bootstrap_lower_95_above_zero": robust,
            "parent_final_oof_confirmation_passed": parent_confirmed,
            "eligible_prefix_guard_passed": prefix_guard,
            "advance_to_known_gate_oof_threshold_calibration": parent_confirmed,
            "online_known_gate_adoption_allowed": False,
        },
        "interpretation_contract": {
            "oof_prediction_count": 986,
            "provenance_component_count": int(
                assignment["provenance_component_count"]
            ),
            "fold_models_are_not_deployment_checkpoints": True,
            "seed43_full_train_checkpoint_remains_reference_only": True,
            "unknown_is_not_a_training_class": True,
            "thresholds_are_not_fitted_in_this_stage": True,
        },
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / PAIRED_REPORT_RELATIVE
    _write(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate V21e KnownGate OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("treatment", "paired"), required=True)
    args = parser.parse_args()
    report = (
        treatment_report(args.project_root)
        if args.mode == "treatment"
        else paired_report(args.project_root)
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()