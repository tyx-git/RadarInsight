"""Aggregate V22b late-readout OOF against frozen V21e OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import summarize_known_gate_records
from radar_ln3.known_gate_oof import load_fold_contract, sequence_sha256, sha256_file
from radar_ln3.known_gate_v22b_oof import (
    FOLD_COUNT,
    V22B_OOF_CONTRACT,
    V22B_OOF_CONTRACT_SCHEMA,
    V22B_OOF_PREDICTION_SCHEMA,
    V22B_OOF_READINESS,
    V22B_OOF_READINESS_SCHEMA,
    V22B_OOF_REPORT,
    V22B_OOF_REPORT_SCHEMA,
    prediction_relative,
)
from report.known_gate_oof import (
    PRIMARY_DELTA,
    _base_label_outcomes,
    _component_bootstrap,
    _delta,
    _load_arm as load_v21e_arm,
)


def _binding(contract: dict[str, Any], fold_id: int) -> dict[str, Any]:
    rows = [dict(row) for row in contract["folds"] if int(row["fold_id"]) == fold_id]
    if len(rows) != 1:
        raise ValueError(f"V22b fold binding unavailable: {fold_id}")
    return rows[0]


def _load_treatment(root: Path) -> dict[str, Any]:
    readiness_path = root / V22B_OOF_READINESS
    contract_path = root / V22B_OOF_CONTRACT
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != V22B_OOF_READINESS_SCHEMA
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != V22B_OOF_CONTRACT_SCHEMA
    ):
        raise ValueError("V22b readiness/contract is stale")
    source_contract, assignment = load_fold_contract(root)
    records_by_scope = {scope: [] for scope in ("parent", "window", "eligible_prefix")}
    folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        binding = _binding(contract, fold_id)
        control_prediction = root / binding["control_prediction_path"]
        if (
            not control_prediction.is_file()
            or sha256_file(control_prediction)
            != binding["control_prediction_sha256"]
        ):
            raise ValueError(f"V22b frozen Control prediction changed: fold {fold_id}")
        path = root / prediction_relative(fold_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = dict(payload.get("records") or {})
        history = list(payload.get("fold_training_history") or [])
        parent_ids = sorted(str(row["parent_sample_id"]) for row in records.get("parent") or [])
        source_fold = next(row for row in source_contract["folds"] if int(row["fold_id"]) == fold_id)
        pairing = dict(payload.get("paired_initialization") or {})
        audit = dict(payload.get("late_residual_audit") or {})
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != V22B_OOF_PREDICTION_SCHEMA
            or payload.get("experiment_name") != "V22b RD temporal late-readout OOF"
            or payload.get("arm") != "treatment"
            or int(payload.get("fold_id", -1)) != fold_id
            or payload.get("prediction_protocol") != "out_of_fold_fixed_final_epoch"
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
            or int(payload.get("held_out_prediction_pass_count", -1)) != 1
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or payload.get("config_sha256") != binding["config_sha256"]
            or set(records) != set(records_by_scope)
            or len(history) != 20
            or [int(row["epoch"]) for row in history] != list(range(1, 21))
            or sequence_sha256(parent_ids) != binding["held_out_parent_ids_sha256"]
            or sequence_sha256(parent_ids) != source_fold["held_out_parent_ids_sha256"]
            or pairing.get("initial_logits_exactly_equal") is not True
            or pairing.get("all_model_parameters_trainable") is not True
            or pairing.get("late_weight_receives_gradient") is not True
            or pairing.get("liquid_state_receives_residual") is not False
            or int(pairing.get("added_trainable_parameter_count", -1)) != 24
            or int(audit.get("final_nonzero_weight_count", 0)) <= 0
            or audit.get("liquid_state_receives_residual") is not False
        ):
            raise ValueError(f"V22b fold prediction is invalid: {fold_id}")
        for scope in records_by_scope:
            records_by_scope[scope].extend(dict(row) for row in records[scope])
        folds.append(
            {
                "fold_id": fold_id,
                "held_out_parent_count": len(parent_ids),
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
                "parent_metrics": payload["evaluation"]["parent_metrics"],
                "eligible_prefix_metrics": payload["evaluation"]["eligible_prefix_metrics"],
                "initial_model_state_sha256": payload["initial_model_state_sha256"],
                "paired_baseline_reference_state_sha256": payload["paired_baseline_reference_state_sha256"],
                "initial_logits_sha256": pairing["initial_logits_sha256"],
                "epoch_parent_order_hashes": [str(row["parent_order_sha256"]) for row in history],
                "pairing": pairing,
            }
        )
    parent_ids = [str(row["parent_sample_id"]) for row in records_by_scope["parent"]]
    if len(parent_ids) != 986 or len(set(parent_ids)) != 986:
        raise ValueError("V22b OOF does not cover 986 known parents exactly once")
    return {
        "assignment": assignment,
        "records_by_scope": records_by_scope,
        "fold_summaries": folds,
        "metrics_by_scope": {
            scope: summarize_known_gate_records(rows, scope=f"oof_v22b_{scope}")
            for scope, rows in records_by_scope.items()
        },
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    treatment = _load_treatment(root)
    control = load_v21e_arm(root, "treatment")
    for fold_id in range(FOLD_COUNT):
        before = control["fold_summaries"][fold_id]
        after = treatment["fold_summaries"][fold_id]
        control_pairing = dict(
            control["predictions"][fold_id].get("paired_initialization") or {}
        )
        if (
            before["held_out_parent_count"] != after["held_out_parent_count"]
            or before["epoch_parent_order_hashes"] != after["epoch_parent_order_hashes"]
            or before["initial_model_state_sha256"] != after["initial_model_state_sha256"]
            or control_pairing.get("initial_logits_sha256")
            != after["initial_logits_sha256"]
            or after["pairing"].get("initial_logits_exactly_equal") is not True
        ):
            raise ValueError(f"V22b fold {fold_id} is not strictly paired to V21e")
    deltas = {
        scope: _delta(control["metrics_by_scope"][scope], treatment["metrics_by_scope"][scope])
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
        treatment["assignment"],
        control["records_by_scope"]["parent"],
        treatment["records_by_scope"]["parent"],
    )
    parent_delta = deltas["parent"]
    target_safe = (
        parent_delta["known_target_recall"] >= 0.0
        and all(int(parent_outcomes[label]["net_rescue"]) >= 0 for label in ("DroneTarget", "BirdTarget", "BalloonTarget"))
    )
    clutter_safe = all(
        parent_delta[f"known_clutter_{field}"] >= 0.0
        for field in ("precision", "recall", "f1")
    )
    prefix_safe = (
        deltas["eligible_prefix"]["macro_f1"] >= 0.0
        and deltas["eligible_prefix"]["known_target_recall"] >= 0.0
        and deltas["eligible_prefix"]["known_clutter_recall"] >= 0.0
    )
    robust = float(bootstrap["treatment_minus_control"]["macro_f1"]["lower_95"]) > 0.0
    primary = parent_delta["macro_f1"] > PRIMARY_DELTA
    adoption = target_safe and clutter_safe and prefix_safe and robust and primary
    mechanism_signal = target_safe and clutter_safe and prefix_safe and parent_delta["macro_f1"] > 0.0
    report = {
        "status": "adoption_candidate_signal" if adoption else "mechanism_signal_below_adoption_threshold" if mechanism_signal else "hypothesis_not_supported",
        "schema_version": V22B_OOF_REPORT_SCHEMA,
        "experiment": "V22b RD temporal late-readout paired OOF",
        "scope": "known train capture/content-grouped OOF only; Unknown/validation/test excluded",
        "control": "frozen V21e OOF Treatment predictions",
        "treatment": "V22b full-parameter late-readout OOF",
        "control_metrics_by_scope": control["metrics_by_scope"],
        "treatment_metrics_by_scope": treatment["metrics_by_scope"],
        "treatment_minus_control": deltas,
        "parent_base_label_outcomes": parent_outcomes,
        "eligible_prefix_base_label_outcomes": prefix_outcomes,
        "paired_component_bootstrap": bootstrap,
        "fold_summaries": treatment["fold_summaries"],
        "decision": {
            "parent_macro_f1_delta_strictly_above_0_02": primary,
            "known_target_and_drone_bird_balloon_recall_safe": target_safe,
            "clutter_precision_recall_f1_all_non_decreasing": clutter_safe,
            "eligible_prefix_guard_passed": prefix_safe,
            "component_bootstrap_macro_f1_lower_95_above_zero": robust,
            "late_readout_mechanism_positive": mechanism_signal,
            "adoption_candidate_signal": adoption,
            "advance_to_l3_calibration": False,
            "online_known_gate_adoption_allowed": False,
        },
        "interpretation_contract": {
            "v22a_recurrent_injection_was_negative": True,
            "v22b_tests_the_same_residual_after_liquid_readout": True,
            "neutral_or_negative_discards_first_last_embedding_residual": True,
            "fold_models_are_not_deployment_checkpoints": True,
            "unknown_is_not_a_training_class": True,
        },
        "test": "未执行",
        "threshold_fitting": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / V22B_OOF_REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate V22b paired OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
