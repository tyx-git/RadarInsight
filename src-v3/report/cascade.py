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
    load_task_config,
    sha256_file,
    task_config_identity_sha256,
    task_device,
)
from radar_ln3.rejection import (
    HierarchicalRejectionPolicy,
    load_hierarchical_rejection_calibration,
    validate_hierarchical_calibration_deployment,
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
        raise ValueError("evaluated V13 checkpoints do not share epoch/global step")
    selection = dict(payloads["dbb_cu_gate"].get("metrics") or {}).get(
        "checkpoint_selection"
    )
    if not isinstance(selection, dict):
        raise ValueError("evaluated V13 checkpoints lack product checkpoint selection")
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
) -> dict[str, Any]:
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
    dataset = build_cascade_dataset(project_root, configs["dbb_cu_gate"], split_name=split_name)
    hierarchical_policy = None
    calibration_payload = None
    if calibration_path is not None:
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
        hierarchical_policy = HierarchicalRejectionPolicy(calibration, enabled=True)
    result = evaluate_cascade(
        cascade,
        dataset,
        device=device,
        batch_size=batch_size or configs["dbb_cu_gate"].train.batch_size,
        split_name=split_name,
        hierarchical_rejection_policy=hierarchical_policy,
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
    report.update({
        "status": "passed",
        "device": str(device),
        "configs": {name: str(config_paths[name]) for name in EXPECTED_TASKS},
        "checkpoints": {name: str(checkpoint_paths[name]) for name in EXPECTED_TASKS},
        "checkpoint_epochs": {name: int(payloads[name].get("epoch", -1)) for name in EXPECTED_TASKS},
        "data_contracts": contracts,
        "sample_detail_information": str(sample_path),
        "evaluation_scope": f"{split_name}:three_checkpoint_hierarchical_cascade_with_soft_posterior_audit",
        "hierarchical_calibration": (
            "未执行"
            if calibration_path is None
            else {
                "path": str(calibration_path),
                "source": calibration_payload["source"],
                "provenance": calibration_payload["provenance"],
            }
        ),
        "full_validation": "executed" if split == "val" else "未执行",
        "test_evaluation": "未执行",
    })
    report_path = target / f"evaluation-{split_name}.json"
    report["report_path"] = str(report_path)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the three-layer liquid hierarchy; five-class soft posterior is audit-only."
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
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
