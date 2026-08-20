"""Aggregate V23 official-CfC OOF predictions against frozen V21e/V22 controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import pstdev
from typing import Any

from radar_ln3.known_gate import KNOWN_GATE_CLASS_NAMES, summarize_known_gate_records
from radar_ln3.known_gate_v21e_oof import (
    fold_record,
    load_fold_contract,
    sequence_sha256,
    sha256_file,
)
from radar_ln3.known_gate_v23 import config_identity_sha256, load_v23_config
from radar_ln3.known_gate_v23_oof import (
    ARMS,
    FIXED_EPOCHS,
    FOLD_COUNT,
    artifact_relative,
    config_relative,
    contract_relative,
    contract_schema,
    prediction_relative,
    prediction_schema,
    readiness_relative,
    readiness_schema,
    report_relative,
    report_schema,
    source_arm_name,
)
from report.known_gate_v21e_oof import (
    PRIMARY_DELTA,
    _base_label_outcomes,
    _component_bootstrap,
    _delta,
    _load_arm as _load_v21e_arm,
)
from report.known_gate_v22_oof import _load_treatment as _load_v22_treatment


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _class_metric(metrics: dict[str, Any], class_name: str, field: str) -> float:
    row = next(row for row in metrics["per_class"] if row["class"] == class_name)
    return float(row[field])


def _fold_dispersion(rows: list[dict[str, Any]]) -> dict[str, float]:
    class_name = KNOWN_GATE_CLASS_NAMES[1]
    values = {
        field: [
            _class_metric(dict(row["parent_metrics"]), class_name, field)
            for row in rows
        ]
        for field in ("precision", "recall", "f1")
    }
    return {
        f"known_clutter_{field}_population_std": pstdev(data)
        for field, data in values.items()
    }


def _load_control(root: Path, arm: str) -> dict[str, Any]:
    if arm == "v23a":
        return _load_v21e_arm(root, "treatment")
    if arm == "v23b":
        return _load_v22_treatment(root)
    raise ValueError(f"unsupported V23 arm: {arm}")


def _load_treatment(root: Path, arm: str) -> dict[str, Any]:
    readiness_path = root / readiness_relative(arm)
    contract_path = root / contract_relative(arm)
    readiness = _read_json(readiness_path)
    contract = _read_json(contract_path)
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version") != readiness_schema(arm)
        or readiness.get("arm") != arm
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or contract.get("schema_version") != contract_schema(arm)
        or contract.get("arm") != arm
    ):
        raise ValueError("V23 readiness/contract is invalid or stale")
    source_contract, assignment = load_fold_contract(root)
    records_by_scope = {scope: [] for scope in ("parent", "window", "eligible_prefix")}
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(source_contract, fold_id)
        binding = next(
            dict(row)
            for row in contract["folds"]
            if int(row["fold_id"]) == fold_id
        )
        config_path = root / config_relative(arm, fold_id)
        config, source_config, source_config_path = load_v23_config(root, config_path)
        source = dict(binding["source"])
        treatment = dict(binding["treatment"])
        path = root / prediction_relative(arm, fold_id)
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = _read_json(path)
        records = dict(payload.get("records") or {})
        evaluation = dict(payload.get("evaluation") or {})
        history = list(payload.get("fold_training_history") or [])
        parent_ids = sorted(str(row["parent_sample_id"]) for row in records.get("parent") or [])
        official = dict(payload.get("official_cfc") or {})
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != prediction_schema(arm)
            or payload.get("experiment_name") != f"{arm.upper()} official ncps CfC architecture-only OOF"
            or payload.get("arm") != arm
            or payload.get("architecture") != "official_ncps_cfc"
            or int(payload.get("fold_id", -1)) != fold_id
            or int(payload.get("epochs", -1)) != FIXED_EPOCHS
            or payload.get("grouping") != "capture_content_provenance_component"
            or payload.get("prediction_protocol") != "out_of_fold_fixed_final_epoch"
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
            or int(payload.get("held_out_prediction_pass_count", -1)) != 1
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or payload.get("config_sha256") != sha256_file(config_path)
            or payload.get("config_identity_sha256") != config_identity_sha256(config)
            or payload.get("source_control", {}).get("config_sha256")
            != config.source_config_sha256
            or payload.get("source_control", {}).get("prediction_sha256")
            != config.source_prediction_sha256
            or payload.get("source_control", {}).get("architecture")
            != "project_local_cfc_inspired_encoder"
            or official.get("distribution") != config.official_cfc.distribution
            or official.get("distribution_version")
            != config.official_cfc.distribution_version
            or official.get("git_commit") != config.official_cfc.git_commit
            or official.get("class_path") != config.official_cfc.class_path
            or official.get("class_module") != "ncps.torch.cfc"
            or official.get("mode") != config.official_cfc.mode
            or official.get("activation") != config.official_cfc.activation
            or official.get("hidden_dim") != config.official_cfc.hidden_dim
            or official.get("backbone_units") != config.official_cfc.backbone_units
            or official.get("backbone_layers") != config.official_cfc.backbone_layers
            or official.get("backbone_dropout") != config.official_cfc.backbone_dropout
            or official.get("mixed_memory") != config.official_cfc.mixed_memory
            or official.get("parameter_count") != 14064
            or set(records) != set(records_by_scope)
            or len(history) != FIXED_EPOCHS
            or [int(row.get("epoch", -1)) for row in history]
            != list(range(1, FIXED_EPOCHS + 1))
            or not all(row.get("parent_order_matches_frozen_source") is True for row in history)
            or [str(row["parent_order_sha256"]) for row in history]
            != list(source["history_parent_order_sha256"])
            or parent_ids != fold["held_out_parent_ids"]
            or treatment.get("config_sha256") != sha256_file(config_path)
            or source.get("config_sha256") != config.source_config_sha256
            or source.get("prediction_sha256") != config.source_prediction_sha256
            or source.get("resolved_class_weight_normalization")
            != config.train.class_weight_normalization
            or source.get("config_identity_sha256")
            != config.source_config_identity_sha256
            or source.get("config_path")
            != config.source_config_path.replace("\\", "/")
            or source.get("prediction_path")
            != config.source_prediction_path.replace("\\", "/")
        ):
            raise ValueError(f"V23 {arm} fold {fold_id} prediction is invalid")
        # Verify the frozen source payload still agrees with the binding rather
        # than trusting only a hash recorded at prepare time.
        source_prediction_path = root / config.source_prediction_path
        if sha256_file(source_prediction_path) != config.source_prediction_sha256:
            raise ValueError(f"V23 {arm} fold {fold_id} source prediction changed")
        for scope in records_by_scope:
            records_by_scope[scope].extend(dict(row) for row in records[scope])
        fold_summaries.append(
            {
                "fold_id": fold_id,
                "held_out_parent_count": len(parent_ids),
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
                "parent_metrics": evaluation["parent_metrics"],
                "eligible_prefix_metrics": evaluation["eligible_prefix_metrics"],
                "initial_model_state_sha256": payload["initial_model_state_sha256"],
                "source_initial_model_state_sha256": source["initial_model_state_sha256"],
                "epoch_parent_order_hashes": [
                    str(row["parent_order_sha256"]) for row in history
                ],
                "official_cfc": official,
            }
        )
    parent_ids = sorted(str(row["parent_sample_id"]) for row in records_by_scope["parent"])
    assignment_ids = sorted(str(row["parent_sample_id"]) for row in assignment["assignments"])
    if (
        len(parent_ids) != 986
        or len(set(parent_ids)) != 986
        or parent_ids != assignment_ids
        or sequence_sha256(parent_ids) != assignment["known_parent_ids_sha256"]
    ):
        raise ValueError("V23 does not cover every known train parent exactly once")
    return {
        "arm": arm,
        "contract": contract,
        "assignment": assignment,
        "records_by_scope": records_by_scope,
        "fold_summaries": fold_summaries,
        "metrics_by_scope": {
            scope: summarize_known_gate_records(rows, scope=f"oof_{arm}_{scope}")
            for scope, rows in records_by_scope.items()
        },
    }


def run(project_root: Path, *, arm: str) -> dict[str, Any]:
    root = project_root.resolve()
    if arm not in ARMS:
        raise ValueError(f"V23 arm must be one of {ARMS}")
    control = _load_control(root, arm)
    treatment = _load_treatment(root, arm)
    for fold_id in range(FOLD_COUNT):
        before = control["fold_summaries"][fold_id]
        after = treatment["fold_summaries"][fold_id]
        if (
            before["held_out_parent_count"] != after["held_out_parent_count"]
            or before["epoch_parent_order_hashes"] != after["epoch_parent_order_hashes"]
        ):
            raise ValueError(f"V23 {arm} fold {fold_id} is not order/fold matched to its frozen control")
    deltas = {
        scope: _delta(
            control["metrics_by_scope"][scope], treatment["metrics_by_scope"][scope]
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
        treatment["assignment"],
        control["records_by_scope"]["parent"],
        treatment["records_by_scope"]["parent"],
    )
    control_dispersion = _fold_dispersion(control["fold_summaries"])
    treatment_dispersion = _fold_dispersion(treatment["fold_summaries"])
    dispersion_delta = {
        key: treatment_dispersion[key] - control_dispersion[key]
        for key in control_dispersion
    }
    parent_delta = deltas["parent"]
    base_safe = all(
        int(parent_outcomes[label]["net_rescue"]) >= 0
        for label in ("DroneTarget", "BirdTarget", "BalloonTarget")
    )
    direction_safe = (
        parent_delta["known_target_recall"] >= 0.0
        and parent_delta["known_clutter_recall"] >= 0.0
        and base_safe
    )
    prefix_safe = (
        deltas["eligible_prefix"]["macro_f1"] >= 0.0
        and deltas["eligible_prefix"]["known_target_recall"] >= 0.0
        and deltas["eligible_prefix"]["known_clutter_recall"] >= 0.0
    )
    robust_macro = (
        float(bootstrap["treatment_minus_control"]["macro_f1"]["lower_95"]) > 0.0
    )
    primary = parent_delta["macro_f1"] > PRIMARY_DELTA
    architecture_signal = primary and direction_safe and prefix_safe and robust_macro
    if architecture_signal:
        status = "architecture_adoption_candidate_signal"
    elif direction_safe and robust_macro:
        status = "architecture_signal_below_2pp_adoption_threshold"
    else:
        status = "hypothesis_not_supported"
    clutter_support_by_fold = [
        int(
            next(
                row
                for row in fold["parent_metrics"]["per_class"]
                if row["class"] == KNOWN_GATE_CLASS_NAMES[1]
            )["support"]
        )
        for fold in control["fold_summaries"]
    ]
    report = {
        "status": status,
        "schema_version": report_schema(arm),
        "experiment": f"{arm.upper()} official ncps CfC architecture-only paired OOF",
        "scope": "known train capture/content-grouped OOF only; Unknown/validation/test excluded",
        "control": {
            "name": source_arm_name(arm),
            "encoder": "project_local_cfc_inspired_encoder",
            "loss_class_weight_normalization": treatment["contract"]["loss_class_weight_normalization"],
        },
        "treatment": {
            "name": "official_ncps_cfc",
            "encoder": "ncps.torch.CfC",
            "runtime": treatment["contract"]["official_runtime"],
            "parameter_matching": treatment["contract"]["parameter_matching"],
        },
        "control_metrics_by_scope": control["metrics_by_scope"],
        "treatment_metrics_by_scope": treatment["metrics_by_scope"],
        "treatment_minus_control": deltas,
        "parent_base_label_outcomes": parent_outcomes,
        "eligible_prefix_base_label_outcomes": prefix_outcomes,
        "paired_component_bootstrap": bootstrap,
        "fold_clutter_dispersion": {
            "control": control_dispersion,
            "treatment": treatment_dispersion,
            "treatment_minus_control": dispersion_delta,
            "support_per_fold": clutter_support_by_fold,
            "warning": "five-fold dispersion is descriptive with only 6-7 Clutter parents per fold",
        },
        "decision": {
            "parent_macro_f1_delta_strictly_above_0_02": primary,
            "known_target_clutter_and_base_label_direction_safe": direction_safe,
            "eligible_prefix_guard_passed": prefix_safe,
            "component_bootstrap_macro_f1_lower_95_above_zero": robust_macro,
            "architecture_signal": architecture_signal,
            "advance_to_l3_calibration": False,
            "online_known_gate_adoption_allowed": False,
            "reason_l3_remains_blocked": (
                "KnownGate OOF does not substitute for strict Unknown open-set calibration, "
                "semantic-decision-layer validation, full cascade, stable or test"
            ),
        },
        "interpretation_contract": {
            "official_cfc_is_a_true_architecture_replacement_not_logit_matched_pair": True,
            "same_fold_parent_orders_loss_and_training_budget_are_verified": True,
            "source_control_predictions_are_frozen": True,
            "unknown_is_not_a_training_class": True,
            "fold_models_are_not_deployment_checkpoints": True,
            "pending_abstain_is_not_directly_supervised": True,
        },
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output = root / report_relative(arm)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate V23 official-CfC KnownGate OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve(), arm=args.arm), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
