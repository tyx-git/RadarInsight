from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.angular_knownness_oof import (
    NESTED_OOF_PREDICTION_SCHEMA,
    derive_nested_stage_configs,
    load_nested_oof_contract,
    nested_config_identities,
    nested_fold_record,
    nested_prediction_path,
    validate_completed_outer_held_out_forward_claim,
)
from radar_ln3.cascade import EXPECTED_TASKS, load_cascade, verify_cascade_data_contracts
from radar_ln3.rejection import (
    CALIBRATION_QUALITY_AUDIT_ONLY,
    CALIBRATION_QUALITY_DEPLOYMENT_ELIGIBLE,
    CU_CLUTTER_PROBABILITY_SCORE,
    DBB_KNOWN_COSINE_SCORE,
    DBB_KNOWN_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    V15_CONTROL_REJECTION_CALIBRATION_SCHEMA,
    V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA,
    load_strict_open_set_rejection_calibration,
    strict_open_set_calibration_quality,
    validate_hierarchical_calibration_deployment,
    validate_strict_open_set_calibration_model_binding,
)
from radar_ln3.strict_open_set_oof import (
    fit_known_only_oof_boundaries,
    sequence_sha256,
    sha256_file,
)
from radar_ln3.task_models import (
    ANGULAR_KNOWNNESS_SCHEMA_VERSION,
    UNKNOWN_BASE_LABEL,
    load_task_config,
    task_config_identity_sha256,
)


TASK_SLUGS = {
    "dbb_cu_gate": "dbb-cu-gate",
    "dbb_specialist": "dbb-specialist",
    "cu_specialist": "cu-specialist",
}


def _config_path(project_root: Path, arm: str, task_name: str) -> Path:
    if arm not in {"control", "treatment"} or task_name not in TASK_SLUGS:
        raise ValueError("V15 OOF report requires a valid paired arm and task")
    return (
        project_root
        / "config"
        / "RADAR-LN3"
        / (
            "RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-"
            f"{arm}-{TASK_SLUGS[task_name]}-config.json"
        )
    )


def _formal_checkpoint_path(project_root: Path, arm: str, task_name: str) -> Path:
    return (
        project_root
        / "artifacts"
        / f"RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-{arm}-formal"
        / task_name
        / "best_model.pt"
    )


def _default_calibration_path(project_root: Path, arm: str) -> Path:
    return (
        project_root
        / "artifacts"
        / f"RADAR-LN3-V15-cq08-liquid-strict-open-set-angular-knownness-{arm}-formal"
        / "strict-open-set-rejection-calibration.json"
    )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deployment_identity(
    configs: Mapping[str, Any],
    checkpoint_paths: Mapping[str, Path],
    payloads: Mapping[str, Mapping[str, Any]],
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
        raise ValueError("V15 formal deployment checkpoints are not one atomic product candidate")
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


def _load_nested_oof_records(
    project_root: Path,
    *,
    arm: str,
    configs: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    contract = load_nested_oof_contract(project_root)
    expected_score = configs["dbb_specialist"].model.dbb_knownness_score_name
    base_identities = nested_config_identities(configs)
    records: list[dict[str, Any]] = []
    fold_provenance: list[dict[str, Any]] = []
    assigned = sorted(str(row["parent_sample_id"]) for row in contract["_assignments"])
    for outer_fold_id in range(int(contract["fold_count"])):
        fold = nested_fold_record(contract, outer_fold_id)
        inner_configs = derive_nested_stage_configs(
            project_root, configs, fold, stage="inner_selection"
        )
        outer_configs = derive_nested_stage_configs(
            project_root, configs, fold, stage="outer_retrain"
        )
        inner_identities = nested_config_identities(inner_configs)
        outer_identities = nested_config_identities(outer_configs)
        path = nested_prediction_path(project_root, arm, outer_fold_id)
        if not path.is_file():
            raise FileNotFoundError(
                f"V15 {arm} nested OOF fold {outer_fold_id:02d} is missing: {path}"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        fold_records = list(payload.get("records") or [])
        ids = sorted(str(record.get("parent_sample_id")) for record in fold_records)
        if (
            payload.get("status") != "passed"
            or payload.get("schema_version") != NESTED_OOF_PREDICTION_SCHEMA
            or payload.get("model_schema_version") != ANGULAR_KNOWNNESS_SCHEMA_VERSION
            or payload.get("arm") != arm
            or int(payload.get("outer_fold_id", -1)) != outer_fold_id
            or payload.get("fit_split") != "train"
            or payload.get("fit_population") != "known_only"
            or payload.get("grouping") != "parent"
            or payload.get("prediction_protocol") != "nested_out_of_fold"
            or payload.get("epoch_protocol")
            != "inner_parent_selection_then_outer_train_retrain"
            or int(payload.get("unknown_training_exposure", -1)) != 0
            or int(payload.get("unknown_oof_prediction_count", -1)) != 0
            or int(payload.get("validation_access_count", -1)) != 0
            or int(payload.get("test_payload_access_count", -1)) != 0
            or payload.get("outer_held_out_used_for_normalization") is not False
            or payload.get("outer_held_out_used_for_epoch_selection") is not False
            or int(payload.get("outer_held_out_forward_count_before_final", -1)) != 0
            or int(payload.get("outer_held_out_prediction_pass_count", -1)) != 1
            or payload.get("nested_oof_contract_sha256") != contract["_contract_sha256"]
            or payload.get("base_config_identities") != base_identities
            or payload.get("inner_config_identities") != inner_identities
            or payload.get("outer_config_identities") != outer_identities
            or payload.get("dbb_knownness_score_name") != expected_score
            or ids != fold["outer_held_out_parent_ids"]
            or int(payload.get("oof_prediction_parent_count", -1)) != len(ids)
            or payload.get("oof_prediction_parent_ids_sha256") != sequence_sha256(ids)
            or any(
                int(record.get("true_id", -1)) >= 4
                or record.get("label_name") == UNKNOWN_BASE_LABEL
                or record.get("dbb_knownness_score_name") != expected_score
                for record in fold_records
            )
        ):
            raise ValueError(f"V15 {arm} nested OOF fold {outer_fold_id:02d} violates provenance")
        validate_completed_outer_held_out_forward_claim(
            project_root,
            contract=contract,
            arm=arm,
            fold=fold,
            outer_config_identities=outer_identities,
            dbb_knownness_score_name=expected_score,
            prediction_path=path,
        )
        records.extend(fold_records)
        fold_provenance.append(
            {
                "outer_fold_id": outer_fold_id,
                "prediction_path": str(path),
                "prediction_sha256": sha256_file(path),
                "parent_count": len(ids),
                "parent_ids_sha256": sequence_sha256(ids),
                "inner_selected_epoch": int(payload["inner_selected_epoch"]),
                "outer_retrain_epoch": int(payload["outer_retrain_epoch"]),
                "checkpoint_sha256": dict(payload["checkpoint_sha256"]),
                "checkpoint_set_id": str(payload["checkpoint_set_id"]),
            }
        )
    parent_ids = sorted(str(record["parent_sample_id"]) for record in records)
    if (
        len(records) != 1011
        or len(set(parent_ids)) != 1011
        or parent_ids != assigned
        or sequence_sha256(parent_ids) != contract["known_parent_ids_sha256"]
    ):
        raise ValueError("V15 nested OOF does not cover each known train parent exactly once")
    return records, {
        "known_parent_count": len(records),
        "known_parent_ids_sha256": sequence_sha256(parent_ids),
        "nested_oof_contract": str(contract["_contract_path"]),
        "nested_oof_contract_sha256": str(contract["_contract_sha256"]),
        "fold_count": int(contract["fold_count"]),
        "folds": fold_provenance,
    }


def _quality_gate(fitted: Mapping[str, Any]) -> tuple[bool, dict[str, Any]]:
    component_targets_passed = {
        "gate": bool(fitted["gate"]["all_precision_recall_targets_passed"]),
        "dbb_knownness": bool(
            fitted["dbb_knownness"]["all_precision_recall_targets_passed"]
        ),
        "clutter_acceptance": bool(
            fitted["clutter_acceptance"]["all_precision_recall_targets_passed"]
        ),
    }
    deployment_eligible = all(component_targets_passed.values())
    return deployment_eligible, {
        "required_components": list(component_targets_passed),
        "component_targets_passed": component_targets_passed,
        "all_required_targets_passed": deployment_eligible,
        "failure_reasons": [
            f"{name}_precision_recall_target_not_met"
            for name, passed in component_targets_passed.items()
            if not passed
        ],
    }


def fit_calibration(
    project_root: Path,
    *,
    arm: str,
    config_paths: Mapping[str, Path],
    checkpoint_paths: Mapping[str, Path],
    output_path: Path,
) -> dict[str, Any]:
    root = project_root.resolve()
    if arm not in {"control", "treatment"}:
        raise ValueError("V15 calibration arm must be control or treatment")
    missing_checkpoints = [
        str(checkpoint_paths[name])
        for name in EXPECTED_TASKS
        if not checkpoint_paths[name].is_file()
    ]
    if missing_checkpoints:
        raise FileNotFoundError(
            "V15 calibration requires the three completed formal checkpoints; "
            "formal training is not available yet: " + "; ".join(missing_checkpoints)
        )
    configs = {name: load_task_config(config_paths[name]) for name in EXPECTED_TASKS}
    if any(config.schema_version != ANGULAR_KNOWNNESS_SCHEMA_VERSION for config in configs.values()):
        raise ValueError("V15 calibration requires V15 configurations")
    if {config.train.paired_arm for config in configs.values()} != {arm}:
        raise ValueError("V15 calibration arm does not match all three configurations")
    contracts = verify_cascade_data_contracts(root, configs)
    _, checkpoint_payloads = load_cascade(configs, checkpoint_paths, device=torch.device("cpu"))
    deployment = _deployment_identity(configs, checkpoint_paths, checkpoint_payloads)
    records, oof = _load_nested_oof_records(root, arm=arm, configs=configs)
    dbb_score = configs["dbb_specialist"].model.dbb_knownness_score_name
    expected_score = DBB_KNOWN_COSINE_SCORE if arm == "treatment" else DBB_KNOWN_PROBABILITY_SCORE
    if dbb_score != expected_score:
        raise ValueError("V15 calibration score semantics do not match the paired arm")
    fitted = fit_known_only_oof_boundaries(
        records,
        gate_target=float(configs["dbb_cu_gate"].train.gate_min_precision_recall_target),
        dbb_target=float(configs["dbb_specialist"].train.dbb_min_precision_recall_target),
        cu_target=float(configs["cu_specialist"].train.cu_min_precision_recall_target),
        dbb_score_name=dbb_score,
    )
    deployment_eligible, quality_gate = _quality_gate(fitted)
    calibration = {
        "schema_version": (
            V15_TREATMENT_REJECTION_CALIBRATION_SCHEMA
            if arm == "treatment"
            else V15_CONTROL_REJECTION_CALIBRATION_SCHEMA
        ),
        "model_schema_version": ANGULAR_KNOWNNESS_SCHEMA_VERSION,
        "paired_arm": arm,
        "source": TRAIN_PARENT_GROUPED_OOF_SOURCE,
        "gate_score_name": GATE_DBB_PROBABILITY_SCORE,
        "dbb_known_score_name": dbb_score,
        "cu_score_name": CU_CLUTTER_PROBABILITY_SCORE,
        "gate_accept_dbb_threshold": fitted["gate"]["probability_threshold"],
        "gate_accept_dbb_margin": fitted["gate"]["margin_threshold"],
        "dbb_accept_known_threshold": fitted["dbb_knownness"]["probability_threshold"],
        "dbb_accept_known_margin": fitted["dbb_knownness"]["margin_threshold"],
        "cu_accept_clutter_threshold": fitted["clutter_acceptance"]["probability_threshold"],
        "cu_accept_clutter_margin": fitted["clutter_acceptance"]["margin_threshold"],
        "calibration_quality_status": (
            CALIBRATION_QUALITY_DEPLOYMENT_ELIGIBLE
            if deployment_eligible
            else CALIBRATION_QUALITY_AUDIT_ONLY
        ),
        "deployment_eligible": deployment_eligible,
        "quality_gate": quality_gate,
        "min_valid_points": 1,
        "min_track_quality": 0.0,
        "max_gap_fraction": 1.0,
        "provenance": {
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "parent",
            "prediction_protocol": "nested_out_of_fold",
            "epoch_protocol": "inner_parent_selection_then_outer_train_retrain",
            "unknown_training_exposure": 0,
            "unknown_threshold_fit_count": 0,
            "validation_used_for_threshold_fitting": False,
            "test_used": False,
            "outer_held_out_used_for_normalization": False,
            "outer_held_out_used_for_epoch_selection": False,
            "outer_held_out_prediction_pass_count": 1,
            "known_parent_count": oof["known_parent_count"],
            "oof_prediction_parent_count": oof["known_parent_count"],
            "oof_unique_parent_count": oof["known_parent_count"],
            "data_contract_hashes": contracts["dbb_cu_gate"]["hashes"],
            "deployment_checkpoint_set": deployment,
            "nested_oof_contract": oof["nested_oof_contract"],
            "nested_oof_contract_sha256": oof["nested_oof_contract_sha256"],
            "oof_prediction_files": oof["folds"],
        },
        "selection_diagnostics": fitted,
        "interpretation_limits": [
            "UnknownTarget was absent from normalization, optimizer, nested OOF prediction and threshold fitting.",
            "Nested OOF thresholds measure known-parent route/acceptance and known false rejection only.",
            "Development Unknown is reserved for one post-freeze audit and cannot select the V15 arm, margin, scale, epoch or threshold.",
            "The calibration is valid only for the exact formal three-checkpoint deployment identity recorded above.",
        ],
        "formal_rejection": "未执行",
        "stable": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    text = json.dumps(calibration, ensure_ascii=False, indent=2) + "\n"
    target = output_path.resolve()
    if target.exists():
        existing, existing_payload = load_strict_open_set_rejection_calibration(
            target, purpose="audit"
        )
        validate_strict_open_set_calibration_model_binding(
            existing,
            model_schema_version=ANGULAR_KNOWNNESS_SCHEMA_VERSION,
            paired_arm=arm,
            dbb_known_score_name=dbb_score,
        )
        validate_hierarchical_calibration_deployment(existing_payload, deployment)
        comparable = dict(calibration)
        comparable.pop("formal_rejection")
        comparable.pop("stable")
        comparable.pop("test")
        comparable.pop("p95")
        existing_comparable = dict(existing_payload)
        for key in ("formal_rejection", "stable", "test", "p95"):
            existing_comparable.pop(key, None)
        if existing_comparable != comparable:
            raise FileExistsError(f"refusing to overwrite a different V15 calibration: {target}")
        write_status = "reused_identical"
        loaded_payload = existing_payload
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        write_status = "created"
        _, loaded_payload = load_strict_open_set_rejection_calibration(target, purpose="audit")
    quality = strict_open_set_calibration_quality(loaded_payload)
    return {
        "status": quality["calibration_quality_status"],
        "schema_version": loaded_payload["schema_version"],
        "arm": arm,
        "calibration_path": str(target),
        "calibration_sha256": sha256_file(target),
        "write_status": write_status,
        "known_oof_parent_count": oof["known_parent_count"],
        "unknown_threshold_fit_count": 0,
        "validation_used_for_threshold_fitting": False,
        "test_used": False,
        "deployment_checkpoint_set": deployment,
        "calibration_quality_status": quality["calibration_quality_status"],
        "deployment_eligible": quality["deployment_eligible"],
        "quality_gate": quality["quality_gate"],
        "selected_boundaries": {
            "gate": fitted["gate"],
            "dbb_knownness": fitted["dbb_knownness"],
            "clutter_acceptance": fitted["clutter_acceptance"],
        },
        "combined_known_action": fitted["combined_known_action"],
        "development_unknown_audit": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate V15 nested known-only parent-grouped OOF and fit one arm-bound "
            "strict-open-set calibration for an exact formal three-checkpoint set."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--arm", choices=("control", "treatment"), required=True)
    parser.add_argument("--gate-config", type=Path, default=None)
    parser.add_argument("--dbb-config", type=Path, default=None)
    parser.add_argument("--cu-config", type=Path, default=None)
    parser.add_argument("--gate-checkpoint", type=Path, default=None)
    parser.add_argument("--dbb-checkpoint", type=Path, default=None)
    parser.add_argument("--cu-checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()

    def absolute(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    config_paths = {
        "dbb_cu_gate": (
            absolute(args.gate_config)
            if args.gate_config
            else _config_path(root, args.arm, "dbb_cu_gate")
        ),
        "dbb_specialist": (
            absolute(args.dbb_config)
            if args.dbb_config
            else _config_path(root, args.arm, "dbb_specialist")
        ),
        "cu_specialist": (
            absolute(args.cu_config)
            if args.cu_config
            else _config_path(root, args.arm, "cu_specialist")
        ),
    }
    checkpoint_paths = {
        "dbb_cu_gate": (
            absolute(args.gate_checkpoint)
            if args.gate_checkpoint
            else _formal_checkpoint_path(root, args.arm, "dbb_cu_gate")
        ),
        "dbb_specialist": (
            absolute(args.dbb_checkpoint)
            if args.dbb_checkpoint
            else _formal_checkpoint_path(root, args.arm, "dbb_specialist")
        ),
        "cu_specialist": (
            absolute(args.cu_checkpoint)
            if args.cu_checkpoint
            else _formal_checkpoint_path(root, args.arm, "cu_specialist")
        ),
    }
    output = (
        absolute(args.output)
        if args.output is not None
        else _default_calibration_path(root, args.arm)
    )
    print(
        json.dumps(
            fit_calibration(
                root,
                arm=args.arm,
                config_paths=config_paths,
                checkpoint_paths=checkpoint_paths,
                output_path=output,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
