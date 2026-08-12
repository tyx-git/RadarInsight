from __future__ import annotations

import argparse
import json
from pathlib import Path

from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
    load_task_config,
    is_product_local_schema,
    summarize_task_performance_guardrails,
    task_device,
    verify_task_data_contract,
)


def run(project_root: Path, config_path: Path, checkpoint_path: Path, *, split: str, batch_size: int | None = None) -> dict[str, object]:
    config = load_task_config(config_path)
    contract = verify_task_data_contract(project_root, config)
    split_name = {
        "train": config.data.train_split_name,
        "val": config.data.validation_split_name,
        "test": config.data.test_split_name,
    }.get(split, split)
    if split_name == config.data.test_split_name:
        raise PermissionError("task test evaluation is disabled until explicitly authorized")
    dataset = build_task_dataset(project_root, config, split_name=split_name)
    device = task_device(config)
    model = LiquidTaskClassifier(config).to(device)
    checkpoint = load_task_checkpoint(checkpoint_path, model, expected_schema=config.train.checkpoint_schema, map_location=device)
    result = evaluate_task(model, dataset, config=config, device=device, batch_size=batch_size)
    artifact_dir = checkpoint_path.parent
    sample_path = artifact_dir / "sample_detail_information.json"
    sample_path.write_text(json.dumps({"schema_version": f"{config.schema_version}-task-sample-detail-v1", "task_name": config.task.name, "checkpoint": str(checkpoint_path), "split": split_name, "records": result.get("records", {})}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = dict(result)
    report.pop("records", None)
    performance = summarize_task_performance_guardrails(
        result,
        target=(
            config.train.task_min_precision_recall_target
            if is_product_local_schema(config.schema_version)
            else None
        ),
    )
    report.update({
        "status": "completed" if is_product_local_schema(config.schema_version) else "passed",
        **performance,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "data_contract": contract,
        "sample_detail_information": str(sample_path),
        "local_task_only": is_product_local_schema(config.schema_version),
        "deployment_eligible": False,
        "full_cascade_evaluation": "未执行",
        "oof_rejection_calibration": "未执行",
        "stable_evaluation": "未执行",
        "p95_latency": "未执行",
        "test_evaluation": "未执行",
    })
    output_path = artifact_dir / f"evaluation-{split_name}.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a repaired RADAR-LN3 liquid task checkpoint on train or validation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    result = run(root, config_path.resolve(), checkpoint_path.resolve(), split=args.split, batch_size=args.batch_size)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
