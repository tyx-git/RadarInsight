"""Prepare V22b late-readout Treatment-only paired OOF support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import load_known_gate_config, sha256_file
from radar_ln3.known_gate_v21e_oof import (
    fold_record,
    load_fold_contract,
    prediction_relative as v21e_prediction_relative,
    relative_path,
    resolve_path,
    write_new_or_identical,
)
from radar_ln3.known_gate_v22b_oof import (
    FOLD_COUNT,
    FIXED_EPOCHS,
    TRAINING_SEED,
    V21E_ASSIGNMENT,
    V21E_CONTROL_REPORT,
    V21E_FOLD_CONTRACT,
    V22B_OOF_CHECKPOINT_SCHEMA,
    V22B_OOF_CONFIG_SCHEMA,
    V22B_OOF_CONTRACT,
    V22B_OOF_CONTRACT_SCHEMA,
    V22B_OOF_PREDICTION_SCHEMA,
    V22B_OOF_READINESS,
    V22B_OOF_READINESS_SCHEMA,
    V22B_OOF_REPORT,
    V22B_OOF_REPORT_SCHEMA,
    V22B_OOF_VARIANT,
    artifact_relative,
    config_relative,
    prediction_relative,
)


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _source_config(root: Path, fold_id: int) -> tuple[Path, dict[str, Any], Any]:
    path = root / (
        "config/RADAR-LN3/"
        f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_id:02d}-"
        "treatment-seed43-config.json"
    )
    payload = _read(path)
    config = load_known_gate_config(path)
    if (
        config.variant != "v21e_course_prefix"
        or config.train.seed != TRAINING_SEED
        or config.train.dataloader_seed != TRAINING_SEED
        or config.train.epochs != FIXED_EPOCHS
        or config.train.save_every_epoch is not False
        or config.train.class_weight_normalization != "batch_weight_sum"
        or config.model.rd_frame_aggregation != "masked_mean"
        or config.model.rd_temporal_residual_dim != 0
    ):
        raise ValueError(f"V22b source fold config differs: {fold_id}")
    return path, payload, config


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_contract, assignment = load_fold_contract(root)
    source_contract_path = root / V21E_FOLD_CONTRACT
    source_assignment_path = root / V21E_ASSIGNMENT
    source_report_path = root / V21E_CONTROL_REPORT
    source_report = _read(source_report_path)
    if (
        source_report.get("schema_version")
        != "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment-report-v1"
        or source_report.get("test") != "未执行"
        or source_report.get("deployment_eligible") is not False
    ):
        raise ValueError("V22b frozen V21e OOF Control report is invalid")

    folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = fold_record(source_contract, fold_id)
        source_path, source_payload, source_config = _source_config(root, fold_id)
        control_prediction = root / v21e_prediction_relative("treatment", fold_id)
        control_payload = _read(control_prediction)
        if (
            control_payload.get("status") != "passed"
            or control_payload.get("arm") != "treatment"
            or int(control_payload.get("fold_id", -1)) != fold_id
            or control_payload.get("held_out_checkpoint_selection") is not False
            or int(control_payload.get("unknown_training_exposure", -1)) != 0
            or int(control_payload.get("validation_access_count", -1)) != 0
            or int(control_payload.get("test_payload_access_count", -1)) != 0
            or len(control_payload.get("fold_training_history") or []) != FIXED_EPOCHS
        ):
            raise ValueError(f"V22b frozen Control prediction is invalid: fold {fold_id}")

        artifact = root / artifact_relative(fold_id)
        payload = json.loads(json.dumps(source_payload, ensure_ascii=False))
        payload["schema_version"] = V22B_OOF_CONFIG_SCHEMA
        payload["config_name"] = (
            "RADAR-LN3-V22b-cq08-known-gate-rd-temporal-late-readout-oof-fold-"
            f"{fold_id:02d}-treatment-seed43"
        )
        payload["variant"] = V22B_OOF_VARIANT
        payload["model"]["rd_temporal_residual_dim"] = 0
        payload["train"].update(
            {
                "checkpoint_path": relative_path(root, artifact / "final_model.pt"),
                "last_checkpoint_path": relative_path(root, artifact / "last_model.pt"),
                "formal_artifact_dir": relative_path(root, artifact),
                "epoch_checkpoint_dir": relative_path(root, artifact / "epochs"),
                "checkpoint_schema": V22B_OOF_CHECKPOINT_SCHEMA,
                "save_every_epoch": False,
                "full_parameter_training": True,
            }
        )
        payload["v22b_oof"] = {
            "fold_id": fold_id,
            "source_v21e_config_path": relative_path(root, source_path),
            "source_v21e_config_sha256": sha256_file(source_path),
            "source_control_prediction_path": relative_path(root, control_prediction),
            "source_control_prediction_sha256": sha256_file(control_prediction),
            "source_data_contract_sha256": source_config.data.expected_contract_sha256,
            "source_normalization_sha256": source_config.data.expected_stats_sha256,
            "residual_definition": (
                "last_active_rd_frame_embedding - first_active_rd_frame_embedding"
            ),
            "injection_point": "per_event_logits_after_liquid_readout",
            "liquid_state_receives_residual": False,
            "added_trainable_parameter_count": 24,
            "unknown_training_exposure": 0,
            "held_out_checkpoint_selection": False,
            "validation_access_count": 0,
            "test_payload_access_count": 0,
        }
        output = root / config_relative(fold_id)
        write_new_or_identical(output, payload)
        folds.append(
            {
                "fold_id": fold_id,
                "train_parent_count": len(fold["train_parent_ids"]),
                "held_out_parent_count": len(fold["held_out_parent_ids"]),
                "train_parent_ids_sha256": fold["train_parent_ids_sha256"],
                "held_out_parent_ids_sha256": fold["held_out_parent_ids_sha256"],
                "config_path": relative_path(root, output),
                "config_sha256": sha256_file(output),
                "source_config_path": relative_path(root, source_path),
                "source_config_sha256": sha256_file(source_path),
                "control_prediction_path": relative_path(root, control_prediction),
                "control_prediction_sha256": sha256_file(control_prediction),
                "treatment_prediction_path": relative_path(
                    root, root / prediction_relative(fold_id)
                ),
            }
        )

    contract = {
        "schema_version": V22B_OOF_CONTRACT_SCHEMA,
        "experiment": "V22b RD temporal late-readout Treatment-only paired OOF",
        "scope": "known train capture/content-grouped OOF only",
        "source_fold_contract_path": relative_path(root, source_contract_path),
        "source_fold_contract_sha256": sha256_file(source_contract_path),
        "source_assignment_path": relative_path(root, source_assignment_path),
        "source_assignment_sha256": sha256_file(source_assignment_path),
        "source_control_report_path": relative_path(root, source_report_path),
        "source_control_report_sha256": sha256_file(source_report_path),
        "fold_count": FOLD_COUNT,
        "epochs": FIXED_EPOCHS,
        "seed": TRAINING_SEED,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "grouping": "capture_content_provenance_component",
        "prediction_protocol": "out_of_fold_fixed_final_epoch",
        "held_out_checkpoint_selection": False,
        "control_retrained": False,
        "single_factor": {
            "control": "V21e course-prefix OOF Treatment",
            "treatment": "same model plus 24-weight late RD temporal residual",
            "residual_definition": (
                "last_active_rd_frame_embedding - first_active_rd_frame_embedding"
            ),
            "liquid_state_receives_residual": False,
            "full_parameter_training": True,
        },
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "folds": folds,
        "report_path": relative_path(root, root / V22B_OOF_REPORT),
        "prediction_schema": V22B_OOF_PREDICTION_SCHEMA,
        "report_schema": V22B_OOF_REPORT_SCHEMA,
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    write_new_or_identical(root / V22B_OOF_CONTRACT, contract)
    contract_sha = sha256_file(root / V22B_OOF_CONTRACT)
    checks = {
        "v21e_fold_assignment_is_reused": True,
        "v21e_fold_train_normalization_is_reused": True,
        "v21e_control_predictions_are_frozen": True,
        "development_validation_is_not_used": True,
        "test_is_not_used": True,
        "unknown_is_not_trained_or_predicted": True,
        "held_out_is_not_used_for_checkpoint_selection": True,
        "all_folds_require_full_20_epoch_training": True,
        "all_shared_and_24_new_parameters_are_trainable": True,
        "residual_does_not_enter_liquid_state": True,
        "single_or_missing_rd_residual_is_zero": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": V22B_OOF_READINESS_SCHEMA,
        "checks": checks,
        "oof_contract_path": relative_path(root, root / V22B_OOF_CONTRACT),
        "oof_contract_sha256": contract_sha,
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "selected_seed": TRAINING_SEED,
        "added_trainable_parameter_count": 24,
        "validation_used_for_readiness_gate": False,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "folds": folds,
        "formal_training": "未执行（OOF折模型不是部署模型）",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    write_new_or_identical(root / V22B_OOF_READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": V22B_OOF_READINESS_SCHEMA,
        "contract": str(root / V22B_OOF_CONTRACT),
        "readiness": str(root / V22B_OOF_READINESS),
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V22b late-readout OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
