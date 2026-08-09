from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.cascade import EXPECTED_TASKS, load_cascade, verify_cascade_data_contracts
from radar_ln3.rejection import (
    CU_CLUTTER_PROBABILITY_SCORE,
    DBB_KNOWN_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    load_strict_open_set_rejection_calibration,
    validate_hierarchical_calibration_deployment,
)
from radar_ln3.strict_open_set_oof import (
    OOF_PREDICTION_SCHEMA,
    canonical_sha256,
    fit_known_only_oof_boundaries,
    fold_prediction_path,
    fold_record,
    load_oof_fold_contract,
    sequence_sha256,
    sha256_file,
)
from radar_ln3.task_models import (
    STRICT_OPEN_SET_SCHEMA_VERSION,
    load_task_config,
    task_config_identity_sha256,
)


DEFAULT_CONFIGS = {
    "dbb_cu_gate": Path(
        "config/RADAR-LN3/RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-cu-gate-config.json"
    ),
    "dbb_specialist": Path(
        "config/RADAR-LN3/RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-specialist-config.json"
    ),
    "cu_specialist": Path(
        "config/RADAR-LN3/RADAR-LN3-V14-cq08-liquid-strict-open-set-cu-specialist-config.json"
    ),
}
DEFAULT_CALIBRATION = Path(
    "artifacts/RADAR-LN3-V14-cq08-liquid-strict-open-set-cascade-formal/"
    "strict-open-set-rejection-calibration.json"
)


def _canonical_sha256(value: Any) -> str:
    return canonical_sha256(value)


def _deployment_identity(
    configs: dict[str, Any],
    checkpoint_paths: dict[str, Path],
    payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    epochs = {int(payloads[name].get("epoch", -1)) for name in EXPECTED_TASKS}
    global_steps = {int(payloads[name].get("global_step", -1)) for name in EXPECTED_TASKS}
    set_ids = {
        str(dict(payloads[name].get("metrics") or {}).get("checkpoint_set_id", ""))
        for name in EXPECTED_TASKS
    }
    selections = [
        dict(payloads[name].get("metrics") or {}).get("checkpoint_selection")
        for name in EXPECTED_TASKS
    ]
    if (
        len(epochs) != 1
        or len(global_steps) != 1
        or min(epochs) < 1
        or min(global_steps) < 1
        or len(set_ids) != 1
        or len(next(iter(set_ids))) != 32
        or any(not isinstance(value, dict) for value in selections)
        or len({_canonical_sha256(value) for value in selections}) != 1
    ):
        raise ValueError("V14 formal deployment checkpoints are not one atomic product candidate")
    return {
        "task_config_identity_sha256": {
            name: task_config_identity_sha256(configs[name]) for name in EXPECTED_TASKS
        },
        "checkpoint_sha256": {
            name: sha256_file(checkpoint_paths[name]) for name in EXPECTED_TASKS
        },
        "epoch": next(iter(epochs)),
        "global_step": next(iter(global_steps)),
        "product_selection_sha256": _canonical_sha256(selections[0]),
    }


def _load_oof_records(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fold_contract, assignment = load_oof_fold_contract(project_root)
    records: list[dict[str, Any]] = []
    fold_provenance: list[dict[str, Any]] = []
    for fold_id in range(int(fold_contract["fold_count"])):
        fold = fold_record(fold_contract, fold_id)
        path = fold_prediction_path(project_root, fold_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"V14 OOF fold {fold_id:02d} predictions are missing; run train.strict_open_set_oof first: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = list(payload.get("records") or [])
        parent_ids = sorted(str(value["parent_sample_id"]) for value in fold_records)
        if (
            payload.get("schema_version") != OOF_PREDICTION_SCHEMA
            or int(payload.get("fold_id", -1)) != fold_id
            or payload.get("fit_split") != "train"
            or payload.get("fit_population") != "known_only"
            or payload.get("grouping") != "parent"
            or payload.get("prediction_protocol") != "out_of_fold_fixed_final_epoch"
            or int(payload.get("epochs", -1)) != 20
            or payload.get("early_stopping") is not False
            or payload.get("held_out_checkpoint_selection") is not False
            or int(payload.get("held_out_forward_count_before_final_epoch", -1)) != 0
            or int(payload.get("held_out_prediction_pass_count", -1)) != 1
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("unknown_oof_prediction_count", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or payload.get("assignment_manifest_sha256")
            != fold_contract["assignment_manifest_sha256"]
            or payload.get("fold_contract_sha256")
            != sha256_file(project_root / "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json")
            or parent_ids != fold["held_out_parent_ids"]
            or int(payload.get("oof_prediction_parent_count", -1)) != len(parent_ids)
            or payload.get("oof_prediction_parent_ids_sha256") != sequence_sha256(parent_ids)
        ):
            raise ValueError(f"V14 OOF fold {fold_id:02d} prediction provenance is invalid")
        records.extend(fold_records)
        fold_provenance.append({
            "fold_id": fold_id,
            "path": str(path),
            "sha256": sha256_file(path),
            "parent_count": len(parent_ids),
            "parent_ids_sha256": sequence_sha256(parent_ids),
            "normalization_stats_sha256": payload["normalization_stats_sha256"],
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "checkpoint_set_id": payload["checkpoint_set_id"],
            "epoch": payload["epoch"],
            "global_step": payload["global_step"],
        })
    parent_ids = sorted(str(value["parent_sample_id"]) for value in records)
    assigned_ids = sorted(str(value["parent_sample_id"]) for value in assignment["assignments"])
    if (
        len(records) != int(assignment["known_parent_count"])
        or len(set(parent_ids)) != len(records)
        or parent_ids != assigned_ids
        or sequence_sha256(parent_ids) != assignment["known_parent_ids_sha256"]
        or any(int(value["true_id"]) >= 4 for value in records)
        or any(value["label_name"] == "UnknownTarget" for value in records)
    ):
        raise ValueError("V14 OOF aggregation does not cover every known train parent exactly once")
    return records, {
        "fold_count": int(fold_contract["fold_count"]),
        "known_parent_count": len(records),
        "known_parent_ids_sha256": sequence_sha256(parent_ids),
        "assignment_manifest_sha256": fold_contract["assignment_manifest_sha256"],
        "fold_contract_sha256": sha256_file(
            project_root / "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-fold-contract.json"
        ),
        "folds": fold_provenance,
    }


def fit_calibration(
    project_root: Path,
    *,
    config_paths: dict[str, Path],
    checkpoint_paths: dict[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    configs = {name: load_task_config(config_paths[name]) for name in EXPECTED_TASKS}
    if any(config.schema_version != STRICT_OPEN_SET_SCHEMA_VERSION for config in configs.values()):
        raise ValueError("V14 OOF calibration requires three strict-open-set configurations")
    contracts = verify_cascade_data_contracts(root, configs)
    device = torch.device("cpu")
    _, checkpoint_payloads = load_cascade(configs, checkpoint_paths, device=device)
    deployment = _deployment_identity(configs, checkpoint_paths, checkpoint_payloads)
    records, oof = _load_oof_records(root)
    gate = configs["dbb_cu_gate"]
    fitted = fit_known_only_oof_boundaries(
        records,
        gate_target=float(gate.train.gate_min_precision_recall_target),
        dbb_target=float(gate.train.dbb_min_precision_recall_target),
        cu_target=float(gate.train.cu_min_precision_recall_target),
    )
    calibration = {
        "schema_version": STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        "source": TRAIN_PARENT_GROUPED_OOF_SOURCE,
        "gate_score_name": GATE_DBB_PROBABILITY_SCORE,
        "dbb_known_score_name": DBB_KNOWN_PROBABILITY_SCORE,
        "cu_score_name": CU_CLUTTER_PROBABILITY_SCORE,
        "gate_accept_dbb_threshold": fitted["gate"]["probability_threshold"],
        "gate_accept_dbb_margin": fitted["gate"]["margin_threshold"],
        "dbb_accept_known_threshold": fitted["dbb_knownness"]["probability_threshold"],
        "dbb_accept_known_margin": fitted["dbb_knownness"]["margin_threshold"],
        "cu_accept_clutter_threshold": fitted["clutter_acceptance"]["probability_threshold"],
        "cu_accept_clutter_margin": fitted["clutter_acceptance"]["margin_threshold"],
        "min_valid_points": 1,
        "min_track_quality": 0.0,
        "max_gap_fraction": 1.0,
        "provenance": {
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "parent",
            "prediction_protocol": "out_of_fold",
            "unknown_training_exposure": 0,
            "unknown_threshold_fit_count": 0,
            "validation_used_for_threshold_fitting": False,
            "test_used": False,
            "known_parent_count": oof["known_parent_count"],
            "oof_prediction_parent_count": oof["known_parent_count"],
            "oof_unique_parent_count": oof["known_parent_count"],
            "data_contract_hashes": contracts["dbb_cu_gate"]["hashes"],
            "deployment_checkpoint_set": deployment,
            "assignment_manifest_sha256": oof["assignment_manifest_sha256"],
            "fold_contract_sha256": oof["fold_contract_sha256"],
            "oof_prediction_files": oof["folds"],
        },
        "selection_diagnostics": fitted,
        "interpretation_limits": [
            "UnknownTarget was absent from normalization, optimizer, OOF prediction and threshold fitting.",
            "Thresholds control known-parent route/acceptance and known false rejection only.",
            "Unknown rejection recall, false acceptance rate, stable behavior, test and P95 remain unassessed until separate frozen evaluation.",
            "The calibration is valid only for the exact formal three-checkpoint deployment identity recorded above.",
        ],
        "formal_rejection": "未执行",
        "stable": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    text = json.dumps(calibration, ensure_ascii=False, indent=2) + "\n"
    output_path = output_path.resolve()
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V14 calibration: {output_path}")
        write_status = "reused_identical"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        write_status = "created"
    loaded, loaded_payload = load_strict_open_set_rejection_calibration(output_path)
    validate_hierarchical_calibration_deployment(loaded_payload, deployment)
    if loaded.source != TRAIN_PARENT_GROUPED_OOF_SOURCE:
        raise RuntimeError("V14 calibration round-trip changed its source")
    return {
        "status": "passed",
        "schema_version": calibration["schema_version"],
        "calibration_path": str(output_path),
        "calibration_sha256": sha256_file(output_path),
        "write_status": write_status,
        "known_oof_parent_count": oof["known_parent_count"],
        "unknown_threshold_fit_count": 0,
        "validation_used_for_threshold_fitting": False,
        "test_used": False,
        "deployment_checkpoint_set": deployment,
        "selected_boundaries": {
            "gate": fitted["gate"],
            "dbb_knownness": fitted["dbb_knownness"],
            "clutter_acceptance": fitted["clutter_acceptance"],
        },
        "combined_known_action": fitted["combined_known_action"],
        "formal_rejection": "未执行",
        "stable": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit V14 strict-open-set acceptance boundaries from known-only train parent-grouped "
            "OOF predictions and bind them to one formal three-checkpoint deployment set."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_CONFIGS["dbb_cu_gate"])
    parser.add_argument("--dbb-config", type=Path, default=DEFAULT_CONFIGS["dbb_specialist"])
    parser.add_argument("--cu-config", type=Path, default=DEFAULT_CONFIGS["cu_specialist"])
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--dbb-checkpoint", type=Path, required=True)
    parser.add_argument("--cu-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_CALIBRATION)
    args = parser.parse_args()
    root = args.project_root.resolve()

    def absolute(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    result = fit_calibration(
        root,
        config_paths={
            "dbb_cu_gate": absolute(args.gate_config),
            "dbb_specialist": absolute(args.dbb_config),
            "cu_specialist": absolute(args.cu_config),
        },
        checkpoint_paths={
            "dbb_cu_gate": absolute(args.gate_checkpoint),
            "dbb_specialist": absolute(args.dbb_checkpoint),
            "cu_specialist": absolute(args.cu_checkpoint),
        },
        output_path=absolute(args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
