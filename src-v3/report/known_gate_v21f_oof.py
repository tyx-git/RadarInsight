"""Aggregate exploratory V21f OOF against frozen V21e OOF Treatment controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.known_gate import KNOWN_LABELS, summarize_known_gate_records
from radar_ln3.known_gate_v21e_oof import sequence_sha256, sha256_file
from radar_ln3.known_gate_v21f_oof import (
    ARM,
    FOLD_COUNT,
    PAIRED_REPORT_RELATIVE,
    PAIRED_REPORT_SCHEMA,
    PREDICTION_SCHEMA,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    prediction_relative,
)
from report.known_gate_v21e_oof import (
    ABSOLUTE_FLOOR,
    PRIMARY_DELTA,
    _base_label_outcomes,
    _component_bootstrap,
    _delta,
    _load_arm as load_v21e_arm,
    _metric_vector,
)
from train.known_gate_v21f_oof import _fold_binding


def _load_v21f_treatment(root: Path) -> dict[str, Any]:
    readiness_path = root / READINESS_RELATIVE
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    contract_path = root / readiness["oof_contract_path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
    ):
        raise ValueError("V21f OOF readiness is invalid or stale")
    records_by_scope = {scope: [] for scope in ("parent", "window", "eligible_prefix")}
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        binding = _fold_binding(contract, fold_id)
        path = root / prediction_relative(ARM, fold_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = dict(payload.get("records") or {})
        evaluation = dict(payload.get("evaluation") or {})
        history = list(payload.get("fold_training_history") or [])
        expected_ids_hash = str(binding["held_out_parent_ids_sha256"])
        parent_ids = sorted(
            str(row["parent_sample_id"]) for row in records.get("parent") or []
        )
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != PREDICTION_SCHEMA
            or payload.get("experiment_name") != "V21f OOF"
            or payload.get("arm") != ARM
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
            or sequence_sha256(parent_ids) != expected_ids_hash
        ):
            raise ValueError(f"V21f OOF fold {fold_id} prediction is invalid")
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
                "initial_model_state_sha256": payload[
                    "initial_model_state_sha256"
                ],
                "paired_baseline_reference_state_sha256": payload[
                    "paired_baseline_reference_state_sha256"
                ],
                "epoch_parent_order_hashes": [
                    str(row["parent_order_sha256"]) for row in history
                ],
                "pairing": payload["paired_initialization"],
            }
        )
    parent_ids = [
        str(row["parent_sample_id"]) for row in records_by_scope["parent"]
    ]
    if len(parent_ids) != 986 or len(set(parent_ids)) != 986:
        raise ValueError("V21f OOF does not cover every known train parent exactly once")
    return {
        "fold_summaries": fold_summaries,
        "records_by_scope": records_by_scope,
        "metrics_by_scope": {
            scope: summarize_known_gate_records(rows, scope=f"oof_v21f_{scope}")
            for scope, rows in records_by_scope.items()
        },
        "contract": contract,
        "readiness": readiness,
    }


def _hard_bird_rescue(
    control: list[Mapping[str, Any]],
    treatment: list[Mapping[str, Any]],
) -> dict[str, Any]:
    before = {str(row["parent_sample_id"]): dict(row) for row in control}
    after = {str(row["parent_sample_id"]): dict(row) for row in treatment}
    if len(before) != len(control) or set(before) != set(after):
        raise ValueError("V21f hard-Bird paired parent sets differ")
    hard_ids = sorted(
        parent
        for parent, row in before.items()
        if row["label_name"] == "BirdTarget"
        and int(row["pred_id"]) != int(row["true_id"])
    )
    if len(hard_ids) != 14:
        raise ValueError("V21f frozen hard-Bird set changed")
    rescued = [
        parent
        for parent in hard_ids
        if int(after[parent]["pred_id"]) == int(after[parent]["true_id"])
    ]
    persistent = sorted(set(hard_ids) - set(rescued))
    new_bird_harms = sorted(
        parent
        for parent, row in before.items()
        if row["label_name"] == "BirdTarget"
        and int(row["pred_id"]) == int(row["true_id"])
        and int(after[parent]["pred_id"]) != int(after[parent]["true_id"])
    )
    return {
        "frozen_v21e_hard_bird_support": len(hard_ids),
        "rescued_count": len(rescued),
        "persistent_error_count": len(persistent),
        "new_bird_harm_count": len(new_bird_harms),
        "net_bird_rescue": len(rescued) - len(new_bird_harms),
        "hard_parent_ids": hard_ids,
        "rescued_parent_ids": rescued,
        "persistent_parent_ids": persistent,
        "new_bird_harm_parent_ids": new_bird_harms,
    }


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise FileExistsError(f"refusing to overwrite a different V21f report: {path}")
    if not path.exists():
        path.write_text(text, encoding="utf-8")


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    treatment = _load_v21f_treatment(root)
    control = load_v21e_arm(root, "treatment")
    assignment = json.loads(
        (root / treatment["contract"]["source_assignment_path"]).read_text(
            encoding="utf-8"
        )
    )
    for fold_id in range(FOLD_COUNT):
        before = control["fold_summaries"][fold_id]
        after = treatment["fold_summaries"][fold_id]
        if (
            before["held_out_parent_count"] != after["held_out_parent_count"]
            or before["epoch_parent_order_hashes"]
            != after["epoch_parent_order_hashes"]
            or before["initial_model_state_sha256"]
            != after["paired_baseline_reference_state_sha256"]
            or int(after["pairing"].get("baseline_evidence_dim", -1)) != 1
            or int(after["pairing"].get("treatment_evidence_dim", -1)) != 7
            or int(after["pairing"].get("new_fusion_columns", -1)) != 6
            or after["pairing"].get("new_fusion_columns_exactly_zero") is not True
        ):
            raise ValueError(f"V21f fold {fold_id} is not strictly paired to V21e")
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
    hard_bird = _hard_bird_rescue(
        control["records_by_scope"]["parent"],
        treatment["records_by_scope"]["parent"],
    )
    treatment_vector = _metric_vector(treatment["metrics_by_scope"]["parent"])
    absolute_guard = bool(
        treatment_vector["weak_class_precision_recall_f1_floor"]
        >= ABSOLUTE_FLOOR
    )
    primary = bool(deltas["parent"]["macro_f1"] > PRIMARY_DELTA)
    direction_safe = bool(
        deltas["parent"]["known_target_recall"] >= 0.0
        and deltas["parent"]["known_clutter_recall"] >= 0.0
        and int(parent_outcomes["BirdTarget"]["net_rescue"]) > 0
        and all(
            int(parent_outcomes[label]["net_rescue"]) >= 0
            for label in ("DroneTarget", "BalloonTarget", "ClutterTarget")
        )
    )
    robust = bool(
        float(bootstrap["treatment_minus_control"]["macro_f1"]["lower_95"])
        > 0.0
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
        "schema_version": PAIRED_REPORT_SCHEMA,
        "experiment": "V21f V21e plus inter-frame spectral paired OOF",
        "scope": "known train reused OOF only; exploratory after V21e error inspection",
        "control": "frozen V21e OOF Treatment",
        "treatment": "V21f course plus inter-frame spectral OOF Treatment",
        "control_metrics_by_scope": control["metrics_by_scope"],
        "treatment_metrics_by_scope": treatment["metrics_by_scope"],
        "treatment_minus_control": deltas,
        "parent_base_label_outcomes": parent_outcomes,
        "eligible_prefix_base_label_outcomes": prefix_outcomes,
        "frozen_hard_bird_outcomes": hard_bird,
        "paired_component_bootstrap": bootstrap,
        "fold_pairing": [
            {
                "fold_id": fold_id,
                "initial_existing_state_equal": True,
                "six_new_columns_zero": True,
                "all_20_epoch_parent_orders_equal": True,
            }
            for fold_id in range(FOLD_COUNT)
        ],
        "decision": {
            "treatment_absolute_guard_passed": absolute_guard,
            "primary_macro_f1_delta_strictly_above_0_02": primary,
            "bird_rescue_and_other_label_direction_safety_passed": direction_safe,
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
            "v21e_absolute_guard_previously_failed": True,
            "unknown_is_not_a_training_class": True,
            "fold_models_are_not_deployment_checkpoints": True,
            "thresholds_are_not_fitted": True,
        },
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / PAIRED_REPORT_RELATIVE
    _write(output, report)
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate exploratory V21f OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
