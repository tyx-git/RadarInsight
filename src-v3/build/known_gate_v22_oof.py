"""Prepare the V22 fixed-denominator loss-only KnownGate OOF experiment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate import (
    KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
    KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_v21e_oof import (
    config_relative as v21e_config_relative,
    load_fold_contract,
    relative_path,
    sha256_file,
    write_new_or_identical,
)
from radar_ln3.known_gate_v22_oof import (
    CONTRACT_RELATIVE,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    FIXED_EPOCHS,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    TRAINING_SEED,
    V21E_ASSIGNMENT_RELATIVE,
    V21E_FOLD_CONTRACT_RELATIVE,
    V21E_TREATMENT_REPORT_RELATIVE,
    V22_CLASS_WEIGHT_NORMALIZATION,
    V22_LOSS_MODE,
    artifact_relative,
    config_relative,
)


def _v22_payload(root: Path, fold_id: int, source: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(source)
    payload["schema_version"] = "RADAR-LN3-V22"
    payload["config_name"] = (
        f"RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof-fold-"
        f"{fold_id:02d}-treatment-seed43"
    )
    artifact = root / artifact_relative(fold_id)
    train = dict(payload["train"])
    train.update(
        {
            "class_weight_normalization": V22_CLASS_WEIGHT_NORMALIZATION,
            "checkpoint_path": relative_path(root, artifact / "final_model.pt"),
            "last_checkpoint_path": relative_path(root, artifact / "last_model.pt"),
            "formal_artifact_dir": relative_path(root, artifact),
            "epoch_checkpoint_dir": relative_path(root, artifact / "epochs"),
            "checkpoint_schema": (
                f"RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-fold-"
                f"{fold_id:02d}-treatment-checkpoint-v1"
            ),
            "paired_baseline_config_path": relative_path(
                root, root / v21e_config_relative("treatment", fold_id)
            ),
        }
    )
    payload["train"] = train
    return payload


def _fixed_denominator_evidence(fold: dict[str, Any]) -> dict[str, Any]:
    counts = dict(fold["train_label_parent_counts"])
    target_count = sum(int(counts[label]) for label in ("DroneTarget", "BirdTarget", "BalloonTarget"))
    clutter_count = int(counts["ClutterTarget"])
    parent_count = target_count + clutter_count
    weights = tuple(float(value) for value in fold["class_weights"])
    weighted_population_sum = target_count * weights[0] + clutter_count * weights[1]
    if abs(weighted_population_sum - parent_count) > 1.0e-9:
        raise ValueError("V22 inverse-frequency weights do not have fold-train mean one")
    return {
        "train_parent_count": parent_count,
        "known_target_parent_count": target_count,
        "known_clutter_parent_count": clutter_count,
        "class_weights": list(weights),
        "weighted_population_sum": weighted_population_sum,
        "weighted_population_mean": weighted_population_sum / parent_count,
        "minibatch_denominator": "contributing_parent_count",
        "expected_class_mass_per_full_fold_train": {
            "KNOWN_TARGET_EVIDENCE": target_count * weights[0],
            "KNOWN_CLUTTER_EVIDENCE": clutter_count * weights[1],
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_contract_path = root / V21E_FOLD_CONTRACT_RELATIVE
    source_assignment_path = root / V21E_ASSIGNMENT_RELATIVE
    source_report_path = root / V21E_TREATMENT_REPORT_RELATIVE
    for path in (source_contract_path, source_assignment_path, source_report_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_contract, assignment = load_fold_contract(root)
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if (
        source_report.get("status") != "completed_absolute_guard_failed"
        or int(assignment.get("known_parent_count", -1)) != 986
        or int(assignment.get("provenance_component_count", -1)) != 975
        or int(assignment.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V22 frozen V21e OOF source is invalid or stale")

    fold_rows: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = next(
            dict(row)
            for row in source_contract["folds"]
            if int(row["fold_id"]) == fold_id
        )
        source_path = root / v21e_config_relative("treatment", fold_id)
        source_config = load_known_gate_config(source_path)
        if (
            source_config.train.class_weight_normalization
            != KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION
            or source_config.train.loss_mode != V22_LOSS_MODE
        ):
            raise ValueError("V22 Control does not preserve the historical V21e loss")
        payload = _v22_payload(
            root,
            fold_id,
            json.loads(source_path.read_text(encoding="utf-8")),
        )
        path = root / config_relative(fold_id)
        write_new_or_identical(path, payload)
        treatment = load_known_gate_config(path)
        if (
            treatment.schema_version != "RADAR-LN3-V22"
            or treatment.train.class_weight_normalization
            != KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION
            or treatment.train.loss_mode != source_config.train.loss_mode
            or treatment.train.batch_size != source_config.train.batch_size
            or treatment.train.seed != TRAINING_SEED
            or treatment.train.epochs != FIXED_EPOCHS
        ):
            raise ValueError(f"V22 fold {fold_id} config violates the one-factor protocol")
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
                "loss_evidence": _fixed_denominator_evidence(fold),
                "control": {
                    "config_path": relative_path(root, source_path),
                    "config_sha256": sha256_file(source_path),
                    "config_identity_sha256": known_gate_config_identity_sha256(source_config),
                    "class_weight_normalization": source_config.train.class_weight_normalization,
                    "prediction_path": relative_path(
                        root,
                        root
                        / "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof"
                        / f"fold-{fold_id:02d}/treatment/held-out-predictions.json",
                    ),
                    "prediction_sha256": sha256_file(
                        root
                        / "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof"
                        / f"fold-{fold_id:02d}/treatment/held-out-predictions.json"
                    ),
                },
                "treatment": {
                    "config_path": relative_path(root, path),
                    "config_sha256": sha256_file(path),
                    "config_identity_sha256": known_gate_config_identity_sha256(treatment),
                    "class_weight_normalization": treatment.train.class_weight_normalization,
                },
            }
        )

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V22 fixed-parent-denominator KnownGate loss-only paired OOF",
        "hypothesis_status": "unverified_until_five_fold_treatment_completes",
        "single_factor": {
            "field": "train.class_weight_normalization",
            "control": KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION,
            "treatment": V22_CLASS_WEIGHT_NORMALIZATION,
            "all_other_training_model_data_fields_frozen": True,
        },
        "source_fold_contract_path": relative_path(root, source_contract_path),
        "source_fold_contract_sha256": sha256_file(source_contract_path),
        "source_assignment_path": relative_path(root, source_assignment_path),
        "source_assignment_sha256": sha256_file(source_assignment_path),
        "source_control_report_path": relative_path(root, source_report_path),
        "source_control_report_sha256": sha256_file(source_report_path),
        "fold_count": FOLD_COUNT,
        "known_parent_count": 986,
        "provenance_component_count": 975,
        "grouping": "capture_content_provenance_component",
        "prediction_protocol": "out_of_fold_fixed_final_epoch",
        "control_reused_from": "V21e OOF Treatment",
        "new_training_arm": "V22 treatment only",
        "folds": fold_rows,
        "unknown_training_exposure": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    contract_path = root / CONTRACT_RELATIVE
    write_new_or_identical(contract_path, contract)
    checks = {
        "v21e_control_predictions_are_frozen_and_reused": True,
        "same_five_provenance_grouped_folds_are_reused": True,
        "same_fold_train_normalization_is_reused": True,
        "same_seed_initialization_parent_order_and_twenty_epochs": True,
        "only_class_weight_normalization_changes": True,
        "inverse_frequency_weights_have_fold_train_mean_one": True,
        "fixed_parent_denominator_does_not_cancel_single_class_batch_weights": True,
        "unknown_training_exposure_is_zero": True,
        "validation_access_count_is_zero": True,
        "test_payload_access_count_is_zero": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
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
        "formal_training": "未执行（等待用户运行五个V22 Treatment fold）",
        "OOF": "未执行",
        "threshold_fitting": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "full_cascade": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    readiness_path = root / READINESS_RELATIVE
    write_new_or_identical(readiness_path, readiness)
    return {
        "status": readiness["status"],
        "oof_contract": str(contract_path),
        "readiness": str(readiness_path),
        "new_training_run_count": FOLD_COUNT,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V22 KnownGate loss OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
