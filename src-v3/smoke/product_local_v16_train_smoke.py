from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.task_models import (
    LiquidTaskClassifier,
    load_task_checkpoint,
    load_task_config,
)
from train.tasks import train_task


CONFIG_NAMES = {
    "dbb_cu_gate": "RADAR-LN3-V16-cq08-liquid-product-local-dbb-cu-gate-config.json",
    "dbb_specialist": "RADAR-LN3-V16-cq08-liquid-product-local-dbb-specialist-config.json",
    "cu_specialist": "RADAR-LN3-V16-cq08-liquid-product-local-cu-specialist-config.json",
}


def _prepare_config(
    project_root: Path,
    *,
    task_name: str,
    source_name: str,
    device_name: str,
    run_root: Path,
) -> Path:
    source = project_root / "config" / "RADAR-LN3" / source_name
    payload = json.loads(source.read_text(encoding="utf-8"))
    artifact = run_root / task_name
    relative_artifact = str(artifact.relative_to(project_root)).replace("\\", "/")
    payload["config_name"] = f"{payload['config_name']}-bounded-smoke-{device_name}"
    payload["train"]["device"] = device_name
    payload["train"]["formal_artifact_dir"] = relative_artifact
    payload["train"]["checkpoint_path"] = f"{relative_artifact}/best_model.pt"
    payload["train"]["last_checkpoint_path"] = f"{relative_artifact}/last_model.pt"
    payload["train"]["epoch_checkpoint_dir"] = f"{relative_artifact}/epochs"
    config_path = run_root / f"{task_name}-config.json"
    config_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def run(project_root: Path, *, device_name: str, run_id: str) -> dict[str, Any]:
    root = project_root.resolve()
    if device_name not in {"cpu", "cuda"}:
        raise ValueError("V16 train-entry smoke device must be cpu or cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA V16 train-entry smoke requested but CUDA is unavailable")
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in run_id):
        raise ValueError("run_id may contain only letters, digits, hyphen and underscore")
    run_root = (
        root
        / "src-v3"
        / "smoke"
        / f"RADAR-LN3-V16-product-local-train-entry-smoke-{device_name}-{run_id}"
    )
    if run_root.exists() and any(run_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite an existing V16 train-entry smoke: {run_root}")
    run_root.mkdir(parents=True, exist_ok=False)

    results: dict[str, Any] = {}
    for task_name, source_name in CONFIG_NAMES.items():
        config_path = _prepare_config(
            root,
            task_name=task_name,
            source_name=source_name,
            device_name=device_name,
            run_root=run_root,
        )
        summary = train_task(
            root,
            config_path,
            max_steps=1,
            epochs_override=1,
            skip_validation=True,
            allow_bounded_smoke=True,
        )
        if (
            summary.get("status") != "completed"
            or summary.get("performance_status") != "not_evaluated"
            or int(summary.get("global_step", -1)) != 1
            or summary.get("deployment_eligible") is not False
        ):
            raise AssertionError(f"V16 {task_name} train-entry summary semantics are invalid")
        config = load_task_config(config_path)
        model = LiquidTaskClassifier(config)
        checkpoint = load_task_checkpoint(
            Path(summary["last_checkpoint"]),
            model,
            expected_schema=config.train.checkpoint_schema,
            map_location="cpu",
        )
        results[task_name] = {
            "status": summary["status"],
            "performance_status": summary["performance_status"],
            "device": summary["device"],
            "global_step": summary["global_step"],
            "stop_reason": summary["stop_reason"],
            "train_parent_count": summary["train_parent_count"],
            "validation_parent_count": summary["validation_parent_count"],
            "best_checkpoint": summary["best_checkpoint"],
            "last_checkpoint": summary["last_checkpoint"],
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "training_summary": str(Path(config.train.formal_artifact_dir) / "training-summary.json"),
            "loss_mode": config.train.loss_mode,
            "dt_reason_mode": config.model.dt_reason_mode,
        }

    payload = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V16-product-local-train-entry-smoke-v1",
        "device": device_name,
        "run_id": run_id,
        "tasks": results,
        "scope": "one optimizer step per local task; validation/checkpoint ranking/product cascade not executed",
        "formal_training": "未执行",
        "full_cascade_evaluation": "未执行",
        "oof_calibration": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "p95": "未执行",
    }
    output = run_root / "result.json"
    payload["result_path"] = str(output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one bounded V16 train.tasks step for each local task.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--run-id", default="v1")
    args = parser.parse_args()
    run(args.project_root.resolve(), device_name=args.device, run_id=args.run_id)


if __name__ == "__main__":
    main()
