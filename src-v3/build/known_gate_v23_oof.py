"""Prepare V23a/V23b official-``ncps`` CfC KnownGate OOF experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import (
    KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
    KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_v21e_oof import (
    fold_record,
    load_fold_contract,
    relative_path,
    sha256_file,
    write_new_or_identical,
)
from radar_ln3.known_gate_v23_oof import (
    ARMS,
    FIXED_EPOCHS,
    FOLD_COUNT,
    TRAINING_SEED,
    V21E_ASSIGNMENT_RELATIVE,
    V21E_CONTROL_REPORT_RELATIVE,
    V21E_FOLD_CONTRACT_RELATIVE,
    V22_CONTRACT_RELATIVE,
    V22_CONTROL_REPORT_RELATIVE,
    artifact_relative,
    config_relative,
    contract_relative,
    contract_schema,
    prediction_relative,
    readiness_relative,
    readiness_schema,
    source_arm_name,
    source_config_relative,
    source_prediction_relative,
    variant_slug,
)
from radar_ln3.model import CfCLiquidCausalEncoder
from radar_ln3.official_cfc import OfficialCfCCausalEncoder, official_cfc_runtime_binding


def _source_report_relative(arm: str) -> Path:
    if arm == "v23a":
        return V21E_CONTROL_REPORT_RELATIVE
    if arm == "v23b":
        return V22_CONTROL_REPORT_RELATIVE
    raise ValueError(f"unsupported V23 arm: {arm}")


def _source_report_status_is_usable(arm: str, report: dict[str, Any]) -> bool:
    status = str(report.get("status") or "")
    if arm == "v23a":
        return status in {"passed", "completed_absolute_guard_failed"}
    if arm == "v23b":
        return status in {
            "adoption_candidate_signal",
            "mechanism_supported_below_2pp_adoption_threshold",
            "hypothesis_not_supported",
        }
    return False


def _expected_loss_normalization(arm: str) -> str:
    if arm == "v23a":
        return KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION
    if arm == "v23b":
        return KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION
    raise ValueError(f"unsupported V23 arm: {arm}")


def _parameter_match(source_config: Any) -> dict[str, Any]:
    custom = CfCLiquidCausalEncoder(
        source_config.model.liquid_hidden_dim,
        source_config.model.liquid_hidden_dim,
        tau_min_seconds=source_config.model.liquid_tau_min_seconds,
        tau_max_seconds=source_config.model.liquid_tau_max_seconds,
        zero_dt_event_floor_seconds=source_config.model.liquid_zero_dt_event_floor_seconds,
        dt_max_seconds=source_config.model.liquid_dt_max_seconds,
        invalid_dt_update_mode=source_config.model.liquid_invalid_dt_update_mode,
        invalid_dt_observation_gain=source_config.model.liquid_invalid_dt_observation_gain,
        state_normalization_mode=source_config.model.liquid_state_normalization,
    )
    official = OfficialCfCCausalEncoder(
        source_config.model.liquid_hidden_dim,
        source_config.model.liquid_hidden_dim,
        dt_max_seconds=source_config.model.liquid_dt_max_seconds,
        backbone_units=source_config.model.liquid_hidden_dim,
    )
    custom_count = sum(parameter.numel() for parameter in custom.parameters())
    official_count = sum(parameter.numel() for parameter in official.parameters())
    if (custom_count, official_count) != (14112, 14064):
        raise ValueError(
            "V23 parameter-matched CfC protocol drifted: expected custom/official "
            f"14112/14064, got {custom_count}/{official_count}"
        )
    return {
        "historical_custom_cfc_inspired_encoder_parameter_count": custom_count,
        "official_ncps_cfc_parameter_count": official_count,
        "official_minus_historical_parameter_count": official_count - custom_count,
        "absolute_parameter_difference": abs(official_count - custom_count),
        "parameter_matching_decision": "official 48x1 backbone is closest among the pre-registered candidates",
    }


def _config_payload(
    root: Path,
    *,
    arm: str,
    fold_id: int,
    source_config_path: Path,
    source_config: Any,
    source_prediction_path: Path,
) -> dict[str, Any]:
    artifact = artifact_relative(arm, fold_id)
    runtime = official_cfc_runtime_binding()
    return {
        "schema_version": f"RADAR-LN3-{arm.upper()}",
        "config_name": (
            f"RADAR-LN3-{arm.upper()}-cq08-known-gate-official-cfc-oof-fold-"
            f"{fold_id:02d}-treatment-seed43"
        ),
        "arm": arm,
        "fold_id": fold_id,
        "source": {
            "role": (
                "frozen V21e OOF treatment control" if arm == "v23a" else "frozen V22 OOF treatment control"
            ),
            "config_path": relative_path(root, source_config_path),
            "config_sha256": sha256_file(source_config_path),
            "config_identity_sha256": known_gate_config_identity_sha256(source_config),
            "prediction_path": relative_path(root, source_prediction_path),
            "prediction_sha256": sha256_file(source_prediction_path),
        },
        "official_cfc": {
            **runtime,
            "mode": "default",
            "activation": "lecun_tanh",
            "hidden_dim": int(source_config.model.liquid_hidden_dim),
            "backbone_units": int(source_config.model.liquid_hidden_dim),
            "backbone_layers": 1,
            "backbone_dropout": 0.0,
            "mixed_memory": False,
            "return_sequences": True,
            "batch_first": True,
            "per_parent_forward_for_upstream_batch_timespan_compatibility": True,
            "state_equation_owner": "official ncps.torch.CfC; Radar-LN3 does not reimplement CfCCell",
        },
        "train": {
            "batch_size": int(source_config.train.batch_size),
            "learning_rate": float(source_config.train.learning_rate),
            "seed": TRAINING_SEED,
            "dataloader_seed": TRAINING_SEED,
            "device": str(source_config.train.device),
            "epochs": FIXED_EPOCHS,
            "gradient_clip_norm": float(source_config.train.gradient_clip_norm),
            "class_weights": [float(value) for value in source_config.train.class_weights],
            "loss_mode": str(source_config.train.loss_mode),
            "class_weight_normalization": str(source_config.train.class_weight_normalization),
            "prefix_auxiliary_weight": float(source_config.train.prefix_auxiliary_weight),
            "prefix_auxiliary_min_events": int(source_config.train.prefix_auxiliary_min_events),
            "checkpoint_path": str(artifact / "final_model.pt").replace("\\", "/"),
            "last_checkpoint_path": str(artifact / "last_model.pt").replace("\\", "/"),
            "formal_artifact_dir": str(artifact).replace("\\", "/"),
            "epoch_checkpoint_dir": str(artifact / "epochs").replace("\\", "/"),
            "checkpoint_schema": (
                f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-oof-fold-{fold_id:02d}-"
                "treatment-checkpoint-v1"
            ),
        },
    }


def _validate_source_prediction(
    root: Path,
    *,
    arm: str,
    fold: dict[str, Any],
    source_config_path: Path,
    source_config: Any,
    source_prediction_path: Path,
) -> dict[str, Any]:
    if not source_prediction_path.is_file():
        raise FileNotFoundError(source_prediction_path)
    payload = json.loads(source_prediction_path.read_text(encoding="utf-8"))
    records = dict(payload.get("records") or {})
    parent_ids = sorted(str(row["parent_sample_id"]) for row in records.get("parent") or [])
    history = list(payload.get("fold_training_history") or [])
    prediction_loss = dict(payload.get("loss_aggregation") or {})
    prediction_loss_normalization = prediction_loss.get("class_weight_normalization")
    loss_metadata_matches = (
        prediction_loss_normalization == _expected_loss_normalization(arm)
        if arm == "v23b"
        else prediction_loss_normalization in (None, "", _expected_loss_normalization(arm))
    )
    if (
        payload.get("status") != "passed"
        or int(payload.get("fold_id", -1)) != int(fold["fold_id"])
        or int(payload.get("epochs", -1)) != FIXED_EPOCHS
        or payload.get("config_sha256") != sha256_file(source_config_path)
        or payload.get("config_identity_sha256") != known_gate_config_identity_sha256(source_config)
        or payload.get("held_out_checkpoint_selection") is not False
        or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
        or int(payload.get("held_out_prediction_pass_count", -1)) != 1
        or int(payload.get("unknown_training_exposure", -1)) != 0
        or int(payload.get("validation_access_count", -1)) != 0
        or int(payload.get("test_payload_access_count", -1)) != 0
        or parent_ids != fold["held_out_parent_ids"]
        or len(history) != FIXED_EPOCHS
        or [int(row.get("epoch", -1)) for row in history] != list(range(1, FIXED_EPOCHS + 1))
        or not loss_metadata_matches
    ):
        raise ValueError(f"V23 {arm} frozen source prediction is invalid for fold {fold['fold_id']}")
    return {
        "prediction_schema": str(payload.get("schema_version")),
        "loss_aggregation_provenance": (
            "prediction_payload" if prediction_loss_normalization is not None else "source_config_default"
        ),
        "resolved_class_weight_normalization": _expected_loss_normalization(arm),
        "history_parent_order_sha256": [str(row["parent_order_sha256"]) for row in history],
        "initial_model_state_sha256": str(payload.get("initial_model_state_sha256")),
        "final_model_state_sha256": str(payload.get("final_model_state_sha256")),
    }


def build(project_root: Path, *, arm: str) -> dict[str, Any]:
    root = project_root.resolve()
    if arm not in ARMS:
        raise ValueError(f"V23 arm must be one of {ARMS}")
    source_contract, assignment = load_fold_contract(root)
    required_paths = {
        "v21e_fold_contract": root / V21E_FOLD_CONTRACT_RELATIVE,
        "v21e_assignment": root / V21E_ASSIGNMENT_RELATIVE,
        "source_report": root / _source_report_relative(arm),
    }
    if arm == "v23b":
        required_paths["v22_contract"] = root / V22_CONTRACT_RELATIVE
    missing = [str(path) for path in required_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V23 {arm} sources are missing: {missing}")
    source_report = json.loads(required_paths["source_report"].read_text(encoding="utf-8"))
    if not _source_report_status_is_usable(arm, source_report):
        raise ValueError(f"V23 {arm} source report is not a usable frozen baseline")
    if (
        int(source_contract.get("known_parent_count", -1)) != 986
        or int(assignment.get("known_parent_count", -1)) != 986
        or int(assignment.get("provenance_component_count", -1)) != 975
        or int(source_contract.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V23 requires the frozen V21e known-only provenance-grouped folds")
    if arm == "v23b":
        v22_contract = json.loads((root / V22_CONTRACT_RELATIVE).read_text(encoding="utf-8"))
        if (
            v22_contract.get("control_reused_from") != "V21e OOF Treatment"
            or v22_contract.get("new_training_arm") != "V22 treatment only"
            or int(v22_contract.get("known_parent_count", -1)) != 986
            or v22_contract.get("unknown_training_exposure") != 0
        ):
            raise ValueError("V23b requires the frozen V22 fixed-parent loss baseline")

    parameter_match: dict[str, Any] | None = None
    fold_rows: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(source_contract, fold_id)
        source_config_path = root / source_config_relative(arm, fold_id)
        source_prediction_path = root / source_prediction_relative(arm, fold_id)
        if not source_config_path.is_file():
            raise FileNotFoundError(source_config_path)
        source_config = load_known_gate_config(source_config_path)
        if (
            source_config.train.seed != TRAINING_SEED
            or source_config.train.dataloader_seed != TRAINING_SEED
            or source_config.train.epochs != FIXED_EPOCHS
            or source_config.train.early_stopping_patience != 0
            or source_config.train.class_weight_normalization != _expected_loss_normalization(arm)
            or source_config.model.evidence_dim != 1
            or source_config.model.liquid_hidden_dim != 48
            or source_config.model.rd_temporal_residual_dim != 0
        ):
            raise ValueError(f"V23 {arm} source config violates its frozen one-factor source contract")
        if parameter_match is None:
            parameter_match = _parameter_match(source_config)
        source_prediction_evidence = _validate_source_prediction(
            root,
            arm=arm,
            fold=fold,
            source_config_path=source_config_path,
            source_config=source_config,
            source_prediction_path=source_prediction_path,
        )
        path = root / config_relative(arm, fold_id)
        payload = _config_payload(
            root,
            arm=arm,
            fold_id=fold_id,
            source_config_path=source_config_path,
            source_config=source_config,
            source_prediction_path=source_prediction_path,
        )
        write_new_or_identical(path, payload)
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_parent_ids_sha256": fold["train_parent_ids_sha256"],
                "held_out_parent_ids_sha256": fold["held_out_parent_ids_sha256"],
                "train_label_parent_counts": fold["train_label_parent_counts"],
                "held_out_label_parent_counts": fold["held_out_label_parent_counts"],
                "normalization_stats_path": fold["normalization_stats_path"],
                "normalization_stats_sha256": fold["normalization_stats_sha256"],
                "data_contract_path": fold["data_contract_path"],
                "data_contract_sha256": fold["data_contract_sha256"],
                "source": {
                    "config_path": relative_path(root, source_config_path),
                    "config_sha256": sha256_file(source_config_path),
                    "config_identity_sha256": known_gate_config_identity_sha256(source_config),
                    "prediction_path": relative_path(root, source_prediction_path),
                    "prediction_sha256": sha256_file(source_prediction_path),
                    **source_prediction_evidence,
                },
                "treatment": {
                    "config_path": relative_path(root, path),
                    "config_sha256": sha256_file(path),
                },
            }
        )
    if parameter_match is None:
        raise RuntimeError("V23 did not construct an official CfC parameter-matching probe")
    contract = {
        "schema_version": contract_schema(arm),
        "experiment": f"{arm.upper()} KnownGate official ncps CfC architecture-only OOF",
        "arm": arm,
        "control": {
            "name": source_arm_name(arm),
            "report_path": relative_path(root, required_paths["source_report"]),
            "report_sha256": sha256_file(required_paths["source_report"]),
            "report_status": source_report["status"],
        },
        "treatment": {
            "encoder": "official ncps.torch.CfC",
            "state_equation_owner": "official ncps CfCCell",
            "architectural_variable": "replace project-local CfC-inspired encoder with parameter-matched official CfC",
            "not_a_reimplementation": True,
        },
        "official_runtime": official_cfc_runtime_binding(),
        "parameter_matching": parameter_match,
        "single_factor": {
            "changed_component": "causal sequence encoder only",
            "frozen": [
                "V16 index and V21e capture/content provenance folds",
                "fold-train normalization",
                "TR/RD front ends and point fusion",
                "V21e causal-course evidence",
                "known-only labels and Unknown exclusion",
                "batch size, seed, parent order, optimizer, epochs, gradient clipping",
                "source loss aggregation and loss coefficients",
                "fixed-final-epoch held-out prediction protocol",
            ],
            "official_cfc_dt_difference_is_intrinsic_to_encoder": (
                "raw zero timespans are given to official CfC; the historical floor/"
                "bounded-observation update rule is not retained inside the replacement encoder"
            ),
        },
        "source_fold_contract_path": relative_path(root, required_paths["v21e_fold_contract"]),
        "source_fold_contract_sha256": sha256_file(required_paths["v21e_fold_contract"]),
        "source_assignment_path": relative_path(root, required_paths["v21e_assignment"]),
        "source_assignment_sha256": sha256_file(required_paths["v21e_assignment"]),
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "grouping": "capture_content_provenance_component",
        "prediction_protocol": "out_of_fold_fixed_final_epoch",
        "loss_class_weight_normalization": _expected_loss_normalization(arm),
        "folds": fold_rows,
        "unknown_training_exposure": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "threshold_fitting": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / contract_relative(arm)
    write_new_or_identical(contract_path, contract)
    checks = {
        "official_ncps_runtime_is_exactly_pinned": True,
        "official_cfc_parameter_difference_is_only_48": parameter_match[
            "absolute_parameter_difference"
        ] == 48,
        "same_five_provenance_grouped_folds_are_reused": True,
        "same_fold_train_normalization_is_reused": True,
        "same_course_feature_cache_and_feature_order_are_reused": True,
        "source_loss_protocol_is_preserved": True,
        "only_sequence_encoder_is_replaced": True,
        "unknown_training_exposure_is_zero": True,
        "validation_access_count_is_zero": True,
        "test_payload_access_count_is_zero": True,
        "held_out_checkpoint_selection_is_disabled": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": readiness_schema(arm),
        "arm": arm,
        "checks": checks,
        "oof_contract_path": relative_path(root, contract_path),
        "oof_contract_sha256": sha256_file(contract_path),
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "selected_seed": TRAINING_SEED,
        "new_training_run_count": FOLD_COUNT,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行（等待用户运行五个V23官方CfC Treatment fold）",
        "OOF": "未执行",
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / readiness_relative(arm)
    write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "arm": arm,
        "oof_contract": str(contract_path),
        "readiness": str(readiness_path),
        "new_training_run_count": FOLD_COUNT,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V23 official-CfC KnownGate OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root, arm=args.arm), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
