"""Aggregate V21h/V21i nested paired OOF against frozen V21e Treatment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import summarize_known_gate_records
from radar_ln3.known_gate_v21e_oof import sequence_sha256, sha256_file
from radar_ln3.known_gate_v21hi_oof import FOLD_COUNT, get_spec, prediction_relative
from report.known_gate_v21e_oof import (
    ABSOLUTE_FLOOR,
    PRIMARY_DELTA,
    _base_label_outcomes,
    _component_bootstrap,
    _delta,
    _load_arm as load_v21e_arm,
    _metric_vector,
)
from train.known_gate_v21hi_oof import _fold_binding


def _load_variant(root: Path, variant: str) -> dict[str, Any]:
    spec = get_spec(variant)
    readiness_path = root / spec.readiness_relative
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    contract_path = root / readiness["oof_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != spec.readiness_schema
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != spec.oof_contract_schema
    ):
        raise ValueError(f"{spec.code} OOF readiness is invalid or stale")
    records_by_scope = {scope: [] for scope in ("parent", "window", "eligible_prefix")}
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        binding = _fold_binding(contract, fold_id)
        path = root / prediction_relative(variant, fold_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = dict(payload.get("records") or {})
        evaluation = dict(payload.get("evaluation") or {})
        history = list(payload.get("fold_training_history") or [])
        parent_ids = sorted(str(row["parent_sample_id"]) for row in records.get("parent") or [])
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != spec.prediction_schema
            or payload.get("experiment_name") != f"{spec.code} OOF"
            or payload.get("arm") != "treatment"
            or int(payload.get("fold_id", -1)) != fold_id
            or payload.get("prediction_protocol")
            != "out_of_fold_fixed_final_epoch"
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
            or int(payload.get("held_out_prediction_pass_count", -1)) != 1
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or set(records) != set(records_by_scope)
            or len(history) != 20
            or [int(row["epoch"]) for row in history] != list(range(1, 21))
            or sequence_sha256(parent_ids) != binding["held_out_parent_ids_sha256"]
        ):
            raise ValueError(f"{spec.code} OOF fold {fold_id} prediction is invalid")
        for scope in records_by_scope:
            records_by_scope[scope].extend(dict(row) for row in records[scope])
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
                "held_out_parent_count": len(parent_ids),
                "parent_metrics": evaluation["parent_metrics"],
                "eligible_prefix_metrics": evaluation["eligible_prefix_metrics"],
                "initial_model_state_sha256": payload["initial_model_state_sha256"],
                "paired_baseline_reference_state_sha256": payload[
                    "paired_baseline_reference_state_sha256"
                ],
                "epoch_parent_order_hashes": [
                    str(row["parent_order_sha256"]) for row in history
                ],
                "pairing": payload["paired_initialization"],
            }
        )
    parent_ids = [str(row["parent_sample_id"]) for row in records_by_scope["parent"]]
    if len(parent_ids) != 986 or len(set(parent_ids)) != 986:
        raise ValueError(f"{spec.code} OOF does not cover each known train parent once")
    return {
        "fold_summaries": fold_summaries,
        "records_by_scope": records_by_scope,
        "metrics_by_scope": {
            scope: summarize_known_gate_records(rows, scope=f"oof_{spec.code}_{scope}")
            for scope, rows in records_by_scope.items()
        },
        "contract": contract,
        "readiness": readiness,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different OOF report: {path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def run(project_root: Path, *, variant: str) -> dict[str, Any]:
    root = project_root.resolve()
    spec = get_spec(variant)
    treatment = _load_variant(root, variant)
    control = load_v21e_arm(root, "treatment")
    for fold_id in range(FOLD_COUNT):
        before = control["fold_summaries"][fold_id]
        after = treatment["fold_summaries"][fold_id]
        pairing = after["pairing"]
        if (
            before["held_out_parent_count"] != after["held_out_parent_count"]
            or before["epoch_parent_order_hashes"]
            != after["epoch_parent_order_hashes"]
            or before["initial_model_state_sha256"]
            != after["paired_baseline_reference_state_sha256"]
            or int(pairing.get("baseline_evidence_dim", -1)) != 1
            or int(pairing.get("treatment_evidence_dim", -1)) != 2
            or int(pairing.get("new_fusion_columns", -1)) != 1
            or pairing.get("new_fusion_columns_exactly_zero") is not True
            or pairing.get("initial_logits_exactly_equal") is not True
        ):
            raise ValueError(f"{spec.code} fold {fold_id} is not strictly paired to V21e")
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
    assignment = json.loads(
        (
            root
            / treatment["contract"]["source_assignment_path"]
        ).read_text(encoding="utf-8")
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
    primary_net = sum(
        int(parent_outcomes[label]["net_rescue"])
        for label in spec.primary_labels
    )
    primary_direction_safe = all(
        int(parent_outcomes[label]["net_rescue"]) >= 0
        for label in spec.primary_labels
    )
    safety_direction_safe = all(
        int(parent_outcomes[label]["net_rescue"]) >= 0
        for label in spec.safety_labels
    )
    primary = bool(deltas["parent"]["macro_f1"] > PRIMARY_DELTA)
    direction_safe = bool(
        primary_net > 0
        and primary_direction_safe
        and safety_direction_safe
        and deltas["parent"]["known_target_recall"] >= 0.0
        and deltas["parent"]["known_clutter_recall"] >= 0.0
    )
    robust = bool(
        float(bootstrap["treatment_minus_control"]["macro_f1"]["lower_95"]) > 0.0
    )
    prefix_guard = bool(
        deltas["eligible_prefix"]["macro_f1"] >= 0.0
        and deltas["eligible_prefix"]["known_target_recall"] >= 0.0
        and deltas["eligible_prefix"]["known_clutter_recall"] >= 0.0
    )
    incremental_signal = primary and direction_safe and robust
    candidate_signal = incremental_signal and absolute_guard
    report = {
        "status": (
            "exploratory_candidate_signal_passed"
            if candidate_signal
            else "completed_exploratory_not_confirmed"
        ),
        "schema_version": spec.paired_report_schema,
        "experiment": f"{spec.code} V21e plus {spec.slug} nested paired OOF",
        "scope": "known train reused OOF only; V21e-based exploratory single-factor experiment",
        "control": "frozen V21e OOF Treatment",
        "treatment": f"{spec.code} {spec.slug} OOF Treatment",
        "primary_hypothesis": {
            "labels": list(spec.primary_labels),
            "combined_net_rescue": primary_net,
        },
        "safety_labels": list(spec.safety_labels),
        "control_metrics_by_scope": control["metrics_by_scope"],
        "treatment_metrics_by_scope": treatment["metrics_by_scope"],
        "treatment_minus_control": deltas,
        "parent_base_label_outcomes": parent_outcomes,
        "eligible_prefix_base_label_outcomes": prefix_outcomes,
        "paired_component_bootstrap": bootstrap,
        "fold_pairing": [
            {
                "fold_id": fold_id,
                "initial_existing_state_equal": True,
                "one_new_column_zero": True,
                "all_20_epoch_parent_orders_equal": True,
            }
            for fold_id in range(FOLD_COUNT)
        ],
        "train_only_prior": treatment["contract"].get("train_only_prior"),
        "decision": {
            "treatment_absolute_guard_passed": absolute_guard,
            "primary_macro_f1_delta_strictly_above_0_02": primary,
            "primary_and_safety_label_direction_safety_passed": direction_safe,
            "component_bootstrap_lower_95_above_zero": robust,
            "incremental_exploratory_signal_passed": incremental_signal,
            "absolute_and_incremental_candidate_signal_passed": candidate_signal,
            "eligible_prefix_guard_passed": prefix_guard,
            "advance_to_new_independent_confirmation": candidate_signal,
            "advance_to_threshold_calibration": False,
            "online_known_gate_adoption_allowed": False,
        },
        "interpretation_contract": {
            "post_hoc_same_oof_reuse": True,
            "independent_confirmation": False,
            "unknown_is_not_a_training_class": True,
            "fold_models_are_not_deployment_checkpoints": True,
            "thresholds_are_not_fitted": True,
        },
        "clutter_support_warning": {
            "pooled_parent_support": 33,
            "development_validation_support_not_used": 9,
            "component_bootstrap_executed": True,
        },
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / spec.paired_report_relative
    _write(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate V21h/V21i nested OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=("h", "i"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), variant=args.variant), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()