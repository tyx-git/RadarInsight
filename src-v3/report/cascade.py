from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.cascade import (
    EXPECTED_TASKS,
    build_cascade_dataset,
    evaluate_cascade,
    load_cascade,
    verify_cascade_data_contracts,
)
from radar_ln3.task_models import (
    STRICT_OPEN_SET_SCHEMA_VERSION,
    is_strict_open_set_schema,
    load_task_config,
    sha256_file,
    task_config_identity_sha256,
    task_device,
)
from radar_ln3.rejection import (
    CALIBRATION_PURPOSE_AUDIT,
    CALIBRATION_PURPOSE_DEPLOYMENT,
    HierarchicalRejectionPolicy,
    StrictOpenSetRejectionPolicy,
    load_hierarchical_rejection_calibration,
    load_strict_open_set_rejection_calibration,
    validate_hierarchical_calibration_deployment,
    validate_strict_open_set_calibration_model_binding,
)


def _without_records(result: dict[str, Any]) -> dict[str, Any]:
    summary = dict(result)
    summary.pop("records", None)
    return summary


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deployment_checkpoint_identity(
    configs: dict[str, Any],
    checkpoint_paths: dict[str, Path],
    payloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    epochs = {int(payloads[name].get("epoch", -1)) for name in EXPECTED_TASKS}
    global_steps = {int(payloads[name].get("global_step", -1)) for name in EXPECTED_TASKS}
    if len(epochs) != 1 or len(global_steps) != 1:
        raise ValueError("evaluated hierarchical checkpoints do not share epoch/global step")
    selection = dict(payloads["dbb_cu_gate"].get("metrics") or {}).get(
        "checkpoint_selection"
    )
    if not isinstance(selection, dict):
        raise ValueError("evaluated hierarchical checkpoints lack product checkpoint selection")
    return {
        "task_config_identity_sha256": {
            name: task_config_identity_sha256(configs[name]) for name in EXPECTED_TASKS
        },
        "checkpoint_sha256": {
            name: sha256_file(checkpoint_paths[name]) for name in EXPECTED_TASKS
        },
        "epoch": next(iter(epochs)),
        "global_step": next(iter(global_steps)),
        "product_selection_sha256": _canonical_sha256(selection),
    }


def run(
    project_root: Path,
    *,
    config_paths: dict[str, Path],
    checkpoint_paths: dict[str, Path],
    split: str,
    batch_size: int | None,
    output_dir: Path | None,
    calibration_path: Path | None = None,
    calibration_purpose: str = CALIBRATION_PURPOSE_DEPLOYMENT,
) -> dict[str, Any]:
    if calibration_path is None and calibration_purpose != CALIBRATION_PURPOSE_DEPLOYMENT:
        raise ValueError("calibration-purpose audit requires an explicit calibration artifact")
    configs = {name: load_task_config(config_paths[name]) for name in EXPECTED_TASKS}
    contracts = verify_cascade_data_contracts(project_root, configs)
    split_name = {
        "train": configs["dbb_cu_gate"].data.train_split_name,
        "val": configs["dbb_cu_gate"].data.validation_split_name,
        "test": configs["dbb_cu_gate"].data.test_split_name,
    }.get(split, split)
    if split_name == configs["dbb_cu_gate"].data.test_split_name:
        raise PermissionError("cascade test evaluation is disabled until explicitly authorized")
    device = task_device(configs["dbb_cu_gate"])
    cascade, payloads = load_cascade(configs, checkpoint_paths, device=device)
    strict_open_set = is_strict_open_set_schema(configs["dbb_cu_gate"].schema_version)
    if (
        calibration_purpose == CALIBRATION_PURPOSE_AUDIT
        and (not strict_open_set or split != "val")
    ):
        raise PermissionError(
            "audit-only calibration is restricted to strict-open-set frozen development validation"
        )
    evaluation_scope = (
        "known_only_checkpoint_selection"
        if strict_open_set and split_name == configs["dbb_cu_gate"].data.train_split_name
        else "development_open_set_audit"
        if strict_open_set
        else None
    )
    dataset = build_cascade_dataset(
        project_root,
        configs["dbb_cu_gate"],
        split_name=split_name,
        label_scope=(
            "known_only"
            if strict_open_set and evaluation_scope == "known_only_checkpoint_selection"
            else "development_open_set_audit"
            if strict_open_set
            else None
        ),
    )
    hierarchical_policy = None
    calibration_payload = None
    if calibration_path is not None:
        if strict_open_set:
            calibration, calibration_payload = load_strict_open_set_rejection_calibration(
                calibration_path,
                purpose=calibration_purpose,
            )
            validate_strict_open_set_calibration_model_binding(
                calibration,
                model_schema_version=configs["dbb_cu_gate"].schema_version,
                paired_arm=configs["dbb_specialist"].train.paired_arm,
                dbb_known_score_name=configs["dbb_specialist"].model.dbb_knownness_score_name,
            )
        else:
            if calibration_purpose != CALIBRATION_PURPOSE_DEPLOYMENT:
                raise ValueError("calibration-purpose audit is supported only for strict-open-set models")
            calibration, calibration_payload = load_hierarchical_rejection_calibration(calibration_path)
        expected_hashes = contracts["dbb_cu_gate"]["hashes"]
        actual_hashes = dict(calibration_payload["provenance"]["data_contract_hashes"])
        if actual_hashes != expected_hashes:
            raise ValueError(
                "hierarchical calibration data hashes do not match the evaluated cascade: "
                f"expected={expected_hashes}, actual={actual_hashes}"
            )
        deployment_identity = _deployment_checkpoint_identity(
            configs,
            checkpoint_paths,
            payloads,
        )
        validate_hierarchical_calibration_deployment(
            calibration_payload,
            deployment_identity,
        )
        hierarchical_policy = (
            StrictOpenSetRejectionPolicy(
                calibration,
                enabled=True,
                allow_audit_only=calibration_purpose == CALIBRATION_PURPOSE_AUDIT,
            )
            if strict_open_set
            else HierarchicalRejectionPolicy(calibration, enabled=True)
        )
    result = evaluate_cascade(
        cascade,
        dataset,
        device=device,
        batch_size=batch_size or configs["dbb_cu_gate"].train.batch_size,
        split_name=split_name,
        hierarchical_rejection_policy=hierarchical_policy,
        evaluation_scope=evaluation_scope,
    )
    target = output_dir or checkpoint_paths["dbb_cu_gate"].parent / "cascade"
    target.mkdir(parents=True, exist_ok=True)
    sample_path = target / "sample_detail_information.json"
    sample_path.write_text(json.dumps({
        "schema_version": f"{configs['dbb_cu_gate'].schema_version}-liquid-soft-cascade-sample-detail-v1",
        "split": split_name,
        "checkpoints": {name: str(checkpoint_paths[name]) for name in EXPECTED_TASKS},
        "records": result.get("records", {}),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = _without_records(result)
    calibration_quality_status = (
        None
        if calibration_payload is None or not strict_open_set
        else str(calibration_payload["calibration_quality_status"])
    )
    deployment_eligible = (
        None
        if calibration_payload is None or not strict_open_set
        else bool(calibration_payload["deployment_eligible"])
    )
    report.update({
        "status": (
            "audit_only"
            if strict_open_set and calibration_quality_status == "audit_only"
            else "passed"
        ),
        "execution_status": "passed",
        "deployment_eligible": deployment_eligible,
        "calibration_purpose": (
            None if calibration_path is None else calibration_purpose
        ),
        "device": str(device),
        "configs": {name: str(config_paths[name]) for name in EXPECTED_TASKS},
        "checkpoints": {name: str(checkpoint_paths[name]) for name in EXPECTED_TASKS},
        "checkpoint_epochs": {name: int(payloads[name].get("epoch", -1)) for name in EXPECTED_TASKS},
        "data_contracts": contracts,
        "sample_detail_information": str(sample_path),
        "evaluation_scope": (
            f"{split_name}:strict_open_set_three_checkpoint_with_named_evidence"
            if strict_open_set
            else f"{split_name}:three_checkpoint_hierarchical_cascade_with_soft_posterior_audit"
        ),
        "hierarchical_calibration": (
            "未执行"
            if calibration_path is None
            else {
                "path": str(calibration_path),
                "source": calibration_payload["source"],
                "provenance": calibration_payload["provenance"],
                "purpose": calibration_purpose,
                "calibration_quality_status": calibration_quality_status,
                "deployment_eligible": deployment_eligible,
                "quality_gate": calibration_payload.get("quality_gate"),
                "quality_status_source": calibration_payload.get("quality_status_source"),
            }
        ),
        "full_validation": "executed" if split == "val" else "未执行",
        "formal_rejection": (
            "未执行"
            if calibration_path is None
            else "audit_only_evaluated_with_calibration"
            if strict_open_set and not deployment_eligible
            else "executed_with_deployment_eligible_calibration"
            if strict_open_set
            else "executed_with_calibration"
        ),
        "stable": (
            "未执行"
            if calibration_path is None
            else "audit_only_evaluated_with_calibration"
            if strict_open_set and not deployment_eligible
            else "executed_with_deployment_eligible_calibration"
            if strict_open_set
            else "executed_with_calibration"
        ),
        "p95": "未执行",
        "test_evaluation": "未执行",
    })
    report_path = target / f"evaluation-{split_name}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the V13/V14 three-layer liquid hierarchy with schema-safe audit semantics."
    )
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--gate-config", type=Path, required=True)
    parser.add_argument("--dbb-config", type=Path, required=True)
    parser.add_argument("--cu-config", type=Path, required=True)
    parser.add_argument("--gate-checkpoint", type=Path, required=True)
    parser.add_argument("--dbb-checkpoint", type=Path, required=True)
    parser.add_argument("--cu-checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--calibration", type=Path, default=None)
    parser.add_argument(
        "--calibration-purpose",
        choices=(CALIBRATION_PURPOSE_DEPLOYMENT, CALIBRATION_PURPOSE_AUDIT),
        default=CALIBRATION_PURPOSE_DEPLOYMENT,
        help=(
            "deployment rejects audit-only V14 calibration; audit permits read-only "
            "development evaluation and marks the report non-deployable"
        ),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    result = run(
        root,
        config_paths={
            "dbb_cu_gate": resolve(args.gate_config),
            "dbb_specialist": resolve(args.dbb_config),
            "cu_specialist": resolve(args.cu_config),
        },
        checkpoint_paths={
            "dbb_cu_gate": resolve(args.gate_checkpoint),
            "dbb_specialist": resolve(args.dbb_checkpoint),
            "cu_specialist": resolve(args.cu_checkpoint),
        },
        split=args.split,
        batch_size=args.batch_size,
        output_dir=None if args.output_dir is None else resolve(args.output_dir),
        calibration_path=None if args.calibration is None else resolve(args.calibration),
        calibration_purpose=args.calibration_purpose,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
