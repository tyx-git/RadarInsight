"""Prepare the immutable, read-only V26d L1 endpoint diagnostic.

The V26c outer folds are already fully trained.  V26d does not create a new
training run: it binds each fold's sealed epoch-14 control checkpoint and its
same-trajectory epoch-20 ``last_model.pt`` treatment checkpoint.  The later
reporter is permitted to run exactly one held-out L1 inference per treatment
fold, while retaining V26c's sealed L2 outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.known_gate_v26c_oof import (
    CONTRACT as V26C_CONTRACT,
    CONTRACT_SCHEMA as V26C_CONTRACT_SCHEMA,
    FOLD_COUNT as V26C_FOLD_COUNT,
    L1_CHECKPOINT_SCHEMA,
    L1_EPOCH,
    L1_PREDICTION_SCHEMA,
    L2_PREDICTION_SCHEMA,
    L3_REPORT as V26C_REPORT,
    READINESS as V26C_READINESS,
    READINESS_SCHEMA as V26C_READINESS_SCHEMA,
    fold_record,
    l1_checkpoint_path,
    l1_config_path,
    l1_last_checkpoint_path,
    l1_prediction_path,
    l1_training_summary_path,
    l2_prediction_path,
    l2_training_summary_path,
    sequence_sha256,
)
from radar_ln3.known_gate_v26d_endpoint import (
    CONTRACT,
    CONTRACT_SCHEMA,
    CONTROL_EPOCH,
    FOLD_COUNT,
    READINESS,
    READINESS_SCHEMA,
    TREATMENT_EPOCH,
    relative_path,
    sha256_file,
    write_new_or_identical,
)


IMPLEMENTATION_FILES = (
    Path("src-v3/build/known_gate_v26d_endpoint.py"),
    Path("src-v3/report/known_gate_v26d_endpoint.py"),
    Path("src-v3/radar_ln3/known_gate_v26d_endpoint.py"),
    Path("src-v3/smoke/known_gate_v26d_endpoint_contract_test.py"),
    Path("src-v3/scripts/v26d_known_gate_endpoint_diagnostic_pipeline.ps1"),
    Path("src-v3/build/known_gate_v26c_oof.py"),
    Path("src-v3/train/known_gate_v26c_oof.py"),
    Path("src-v3/report/known_gate_v26c_oof.py"),
    Path("src-v3/radar_ln3/known_gate_v26c_oof.py"),
    Path("src-v3/radar_ln3/known_gate.py"),
    Path("src-v3/radar_ln3/open_set_architecture.py"),
)
KNOWN_LABEL_SUPPORT = {
    "BalloonTarget": 238,
    "BirdTarget": 239,
    "ClutterTarget": 33,
    "DroneTarget": 476,
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26d expected a JSON object: {path}")
    return value


def implementation_sha256(root: Path) -> dict[str, str]:
    missing = [relative_path(root, root / path) for path in IMPLEMENTATION_FILES if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("V26d implementation files are missing: " + ", ".join(missing))
    return {
        relative_path(root, root / path): sha256_file(root / path)
        for path in IMPLEMENTATION_FILES
    }


def _require_v26c(root: Path) -> dict[str, Any]:
    paths = {
        "contract": root / V26C_CONTRACT,
        "readiness": root / V26C_READINESS,
        "report": root / V26C_REPORT,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("V26d requires completed V26c artifacts: " + ", ".join(missing))
    contract = _read(paths["contract"])
    readiness = _read(paths["readiness"])
    report = _read(paths["report"])
    assignment_path = root / Path(str(report.get("assignment_path", "")))
    checks = dict(readiness.get("checks") or {})
    if (
        V26C_FOLD_COUNT != FOLD_COUNT
        or L1_EPOCH != CONTROL_EPOCH
        or contract.get("schema_version") != V26C_CONTRACT_SCHEMA
        or readiness.get("schema_version") != V26C_READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(paths["contract"])
        or not checks
        or not all(value is True for value in checks.values())
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or report.get("status") != "completed_cross_fitted_known_only_l3_diagnostic"
        or report.get("contract_sha256") != sha256_file(paths["contract"])
        or not assignment_path.is_file()
        or report.get("assignment_sha256") != sha256_file(assignment_path)
        or int(report.get("known_parent_count", -1)) != 986
        or dict(report.get("known_label_parent_support") or {}) != KNOWN_LABEL_SUPPORT
        or int(report.get("unknown_training_exposure", -1)) != 0
        or int(report.get("unknown_threshold_fit_exposure", -1)) != 0
        or int(report.get("validation_access_count", -1)) != 0
        or int(report.get("test_access_count", -1)) != 0
    ):
        raise ValueError("V26d source V26c contract/readiness/report is invalid")
    return {
        "contract": contract,
        "paths": {
            name: relative_path(root, path) for name, path in paths.items()
        } | {"assignment": relative_path(root, assignment_path)},
        "sha256": {name: sha256_file(path) for name, path in paths.items()} | {"assignment": sha256_file(assignment_path)},
    }


def _checkpoint_payload(path: Path, *, fold_id: int, epoch: int) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("checkpoint_schema") != L1_CHECKPOINT_SCHEMA
        or payload.get("task_name") != "known_gate"
        or int(payload.get("fold_id", -1)) != fold_id
        or int(payload.get("epoch", -1)) != epoch
        or payload.get("held_out_checkpoint_selection") is not False
        or payload.get("unknown_is_model_class") is not False
        or payload.get("full_parameter_training") is not True
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError(
            f"V26d L1 checkpoint has an invalid V26c binding: fold={fold_id}; epoch={epoch}"
        )
    return dict(payload)


def _fold_binding(root: Path, contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    fold = fold_record(contract, fold_id)
    paths = {
        "l1_config": root / l1_config_path(fold_id),
        "control_checkpoint": root / l1_checkpoint_path(fold_id),
        "treatment_checkpoint": root / l1_last_checkpoint_path(fold_id),
        "control_prediction": root / l1_prediction_path(fold_id),
        "l1_summary": root / l1_training_summary_path(fold_id),
        "l2_prediction": root / l2_prediction_path(fold_id),
        "l2_summary": root / l2_training_summary_path(fold_id),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"V26d fold {fold_id} is missing V26c sources: " + ", ".join(missing)
        )
    control = _checkpoint_payload(paths["control_checkpoint"], fold_id=fold_id, epoch=CONTROL_EPOCH)
    treatment = _checkpoint_payload(paths["treatment_checkpoint"], fold_id=fold_id, epoch=TREATMENT_EPOCH)
    shared_fields = (
        "config_identity_sha256",
        "config_file_sha256",
        "fold_contract_sha256",
        "readiness_sha256",
        "source_config_sha256",
        "initial_model_state_sha256",
    )
    if any(control.get(field) != treatment.get(field) for field in shared_fields):
        raise ValueError(f"V26d control/treatment checkpoint lineage differs: fold={fold_id}")
    config_sha = sha256_file(paths["l1_config"])
    if control.get("config_file_sha256") != config_sha:
        raise ValueError(f"V26d control checkpoint config hash differs: fold={fold_id}")
    summary = _read(paths["l1_summary"])
    prediction = _read(paths["control_prediction"])
    l2_prediction = _read(paths["l2_prediction"])
    l2_summary = _read(paths["l2_summary"])
    parent_rows = list(dict(prediction.get("records") or {}).get("parent") or ())
    parent_ids = sorted(str(row.get("parent_sample_id", "")) for row in parent_rows)
    if (
        summary.get("status") != "completed"
        or int(summary.get("fold_id", -1)) != fold_id
        or int(summary.get("fixed_snapshot_epoch", -1)) != CONTROL_EPOCH
        or summary.get("checkpoint_sha256") != sha256_file(paths["control_checkpoint"])
        or prediction.get("schema_version") != L1_PREDICTION_SCHEMA
        or int(prediction.get("fold_id", -1)) != fold_id
        or int(prediction.get("fixed_snapshot_epoch", -1)) != CONTROL_EPOCH
        or prediction.get("checkpoint_sha256") != sha256_file(paths["control_checkpoint"])
        or prediction.get("held_out_checkpoint_selection") is not False
        or int(prediction.get("held_out_forward_count_before_fixed_snapshot", -1)) != 0
        or int(prediction.get("held_out_prediction_pass_count", -1)) != 1
        or int(prediction.get("unknown_training_exposure", -1)) != 0
        or int(prediction.get("validation_access_count", -1)) != 0
        or int(prediction.get("test_payload_access_count", -1)) != 0
        or parent_ids != list(fold["held_out_parent_ids"])
        or len(parent_ids) != len(set(parent_ids))
        or l2_prediction.get("schema_version") != L2_PREDICTION_SCHEMA
        or int(l2_prediction.get("fold_id", -1)) != fold_id
        or l2_summary.get("status") != "completed"
        or int(l2_summary.get("fold_id", -1)) != fold_id
    ):
        raise ValueError(f"V26d V26c fold binding differs: fold={fold_id}")
    return {
        "fold_id": fold_id,
        "held_out_parent_count": len(parent_ids),
        "held_out_parent_ids_sha256": sequence_sha256(parent_ids),
        "control_epoch": CONTROL_EPOCH,
        "treatment_epoch": TREATMENT_EPOCH,
        "paths": {name: relative_path(root, path) for name, path in paths.items()},
        "sha256": {name: sha256_file(path) for name, path in paths.items()},
        "shared_checkpoint_lineage": {field: str(control[field]) for field in shared_fields},
        "control_global_step": int(control.get("global_step", -1)),
        "treatment_global_step": int(treatment.get("global_step", -1)),
    }


def _source_binding(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    source = _require_v26c(root)
    v26c_contract = dict(source["contract"])
    folds = [_fold_binding(root, v26c_contract, fold_id) for fold_id in range(FOLD_COUNT)]
    if sum(int(row["held_out_parent_count"]) for row in folds) != 986:
        raise ValueError("V26d source fold support does not cover exactly 986 Known parents")
    return source, folds


def _contract_payload(root: Path) -> dict[str, Any]:
    source, folds = _source_binding(root)
    return {
        "schema_version": CONTRACT_SCHEMA,
        "status": "prepared_read_only_endpoint_diagnostic",
        "implementation_sha256": implementation_sha256(root),
        "source_v26c": {
            "paths": source["paths"],
            "sha256": source["sha256"],
            "known_parent_count": 986,
            "known_label_parent_support": KNOWN_LABEL_SUPPORT,
        },
        "checkpoint_protocol": {
            "control_epoch": CONTROL_EPOCH,
            "treatment_epoch": TREATMENT_EPOCH,
            "control": "V26c frozen upstream-validation-selected epoch-14 snapshot",
            "treatment": "same V26c L1 trajectory fixed endpoint last_model.pt at epoch 20",
            "outer_held_out_checkpoint_selection": False,
            "per_fold_epoch_selection": False,
            "checkpoint_scan": False,
            "l2_checkpoint": "sealed V26c epoch-15 L2 outputs are reused unchanged",
        },
        "threshold_views": {
            "frozen_control_thresholds": "apply V26c cross-fitted fold thresholds unchanged to epoch-20 L1 scores",
            "same_protocol_recalibration": "fit each fold threshold from the other four epoch-20 parent-final OOF folds only",
            "outer_fold_labels_used_for_threshold_fit": False,
        },
        "scope": {
            "training_executed": False,
            "unknown_training_exposure": 0,
            "unknown_normalization_exposure": 0,
            "unknown_oof_prediction_count": 0,
            "unknown_threshold_fit_exposure": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
            "full_cascade": "not_executed",
            "deployment_eligible": False,
        },
        "folds": folds,
    }


def _readiness_payload(root: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    source, folds = _source_binding(root)
    expected = _contract_payload(root)
    checks = {
        "v26c_source_contract_readiness_and_report_are_valid": True,
        "all_five_epoch14_control_checkpoints_are_bound": True,
        "all_five_epoch20_treatment_last_checkpoints_are_bound": True,
        "control_and_treatment_share_each_fold_training_lineage": True,
        "control_epoch_is_14_and_treatment_epoch_is_20": True,
        "l2_outputs_are_sealed_and_not_retrained": True,
        "outer_held_out_checkpoint_selection_is_disabled": True,
        "unknown_validation_and_test_exposure_are_zero": True,
        "fold_parent_coverage_is_exactly_986_known_parents": True,
        "contract_is_currently_source_bound": contract == expected,
    }
    if not all(checks.values()):
        raise ValueError("V26d contract/source binding is stale")
    return {
        "schema_version": READINESS_SCHEMA,
        "status": "passed",
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "implementation_sha256": implementation_sha256(root),
        "checks": checks,
        "fold_count": len(folds),
        "known_parent_count": 986,
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "unknown_threshold_fit_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "training_executed": False,
        "deployment_eligible": False,
    }


def require_prepared(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify V26d's immutable preparation and all source bindings."""

    root = project_root.resolve()
    contract_path, readiness_path = root / CONTRACT, root / READINESS
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V26d prepare must complete before evaluation")
    contract, readiness = _read(contract_path), _read(readiness_path)
    checks = dict(readiness.get("checks") or {})
    expected = _contract_payload(root)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or contract != expected
        or readiness.get("implementation_sha256") != implementation_sha256(root)
        or not checks
        or not all(value is True for value in checks.values())
        or int(readiness.get("unknown_training_exposure", -1)) != 0
        or int(readiness.get("unknown_normalization_exposure", -1)) != 0
        or int(readiness.get("unknown_oof_prediction_count", -1)) != 0
        or int(readiness.get("unknown_threshold_fit_exposure", -1)) != 0
        or int(readiness.get("validation_access_count", -1)) != 0
        or int(readiness.get("test_access_count", -1)) != 0
        or readiness.get("training_executed") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V26d prepared contract/readiness is invalid or stale")
    return contract, readiness


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract = _contract_payload(root)
    write_new_or_identical(root / CONTRACT, contract)
    readiness = _readiness_payload(root, contract)
    write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "contract": str(root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "readiness": str(root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "fold_count": FOLD_COUNT,
        "control_epoch": CONTROL_EPOCH,
        "treatment_epoch": TREATMENT_EPOCH,
        "training_executed": False,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V26d's read-only endpoint diagnostic.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
