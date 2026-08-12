from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    compose_soft_cascade,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
    seed_everything,
    task_device,
    task_training_objective,
    verify_task_data_contract,
)


def run(
    project_root: Path,
    config_path: Path,
    *,
    parent_limit: int = 6,
    output_suffix: str = "",
    device_override: str | None = None,
) -> dict[str, object]:
    config = load_task_config(config_path)
    if config.task.name != "dbb_cu_gate":
        raise ValueError("task_dbb_cu requires the dbb_cu_gate task")
    seed_everything(config.train.seed)
    contract = verify_task_data_contract(project_root, config)
    dataset = build_task_dataset(project_root, config, split_name=config.data.train_split_name, parent_limit=parent_limit)
    loader = build_task_dataloader(dataset, batch_size=min(config.train.batch_size, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    if device_override not in {None, "cpu", "cuda"}:
        raise ValueError("device_override must be cpu, cuda or omitted")
    if device_override == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke was requested but CUDA is unavailable")
    device = torch.device(device_override) if device_override is not None else task_device(config)
    model = LiquidTaskClassifier(config).to(device)
    tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    output = model(
        tr_features=tensors["tr_features"],
        tr_missing_mask=tensors["tr_missing_mask"],
        tr_dt_seconds=tensors["tr_dt_seconds"],
        tr_dt_valid=tensors["tr_dt_valid"],
        tr_dt_reason=tensors.get("tr_dt_reason"),
        tr_motion_features=tensors.get("tr_motion_features"),
        tr_motion_valid_mask=tensors.get("tr_motion_valid_mask"),
        rd_frames=tensors["rd_frames"],
        rd_frame_mask=tensors["rd_frame_mask"],
        rd_quality=tensors["rd_quality"],
        tr_padding_mask=tensors["tr_padding_mask"],
    )
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
    objective = task_training_objective(output, tensors, config, class_weights=weights)
    loss = objective["loss"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item())
    motion_gradient_norm = None
    if model.motion_encoder is not None:
        motion_gradient_squared = sum(
            float(parameter.grad.detach().pow(2).sum().item())
            for parameter in model.motion_encoder.parameters()
            if parameter.grad is not None
        )
        motion_gradient_norm = motion_gradient_squared ** 0.5
        if motion_gradient_norm <= 0.0:
            raise ValueError("DBB/CU Gate smoke did not propagate a gradient through the causal TR encoder")
    optimizer.step()
    if not torch.isfinite(loss).item() or not torch.isfinite(torch.tensor(grad_norm)).item():
        raise ValueError("DBB/CU smoke produced non-finite loss or gradient")
    with torch.inference_mode():
        updated_output = model(
            tr_features=tensors["tr_features"],
            tr_missing_mask=tensors["tr_missing_mask"],
            tr_dt_seconds=tensors["tr_dt_seconds"],
            tr_dt_valid=tensors["tr_dt_valid"],
            tr_dt_reason=tensors.get("tr_dt_reason"),
            tr_motion_features=tensors.get("tr_motion_features"),
            tr_motion_valid_mask=tensors.get("tr_motion_valid_mask"),
            rd_frames=tensors["rd_frames"],
            rd_frame_mask=tensors["rd_frame_mask"],
            rd_quality=tensors["rd_quality"],
            tr_padding_mask=tensors["tr_padding_mask"],
        )
    gate = output["sequence_logits"]
    dbb = torch.zeros((*gate.shape[:-1], 3), device=device)
    cu = torch.zeros((*gate.shape[:-1], 2), device=device)
    combined = compose_soft_cascade(gate, dbb, cu)
    if output_suffix and any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in output_suffix):
        raise ValueError("smoke output_suffix may contain only letters, digits, hyphen and underscore")
    smoke_dir = project_root / "src-v3" / "smoke" / f"{config.schema_version}-dbb-cu-gate-contract-smoke{output_suffix}"
    checkpoint_path = smoke_dir / "checkpoint.pt"
    save_task_checkpoint(checkpoint_path, model, optimizer, config=config, epoch=1, global_step=1, metrics={"loss": float(loss.item())})
    restored = LiquidTaskClassifier(config).to(device)
    checkpoint = load_task_checkpoint(checkpoint_path, restored, expected_schema=config.train.checkpoint_schema, map_location=device)
    with torch.inference_mode():
        restored_output = restored(
            tr_features=tensors["tr_features"],
            tr_missing_mask=tensors["tr_missing_mask"],
            tr_dt_seconds=tensors["tr_dt_seconds"],
            tr_dt_valid=tensors["tr_dt_valid"],
            tr_dt_reason=tensors.get("tr_dt_reason"),
            tr_motion_features=tensors.get("tr_motion_features"),
            tr_motion_valid_mask=tensors.get("tr_motion_valid_mask"),
            rd_frames=tensors["rd_frames"],
            rd_frame_mask=tensors["rd_frame_mask"],
            rd_quality=tensors["rd_quality"],
            tr_padding_mask=tensors["tr_padding_mask"],
        )
    if not torch.allclose(updated_output["window_logits"], restored_output["window_logits"], atol=1e-6, rtol=1e-5):
        raise AssertionError("DBB/CU checkpoint round-trip changed window logits")
    payload = {
        "status": "passed",
        "task": config.task.name,
        "config": str(config_path),
        "device": str(device),
        "data_contract_hashes": contract["hashes"],
        "parent_limit": parent_limit,
        "label_counts": dataset.label_counts(),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "sequence_logits_shape": list(output["sequence_logits"].shape),
        "window_logits_shape": list(output["window_logits"].shape),
        "soft_cascade_probability_shape": list(combined.shape),
        "loss": float(loss.item()),
        "main_loss": float(objective["main_loss"].item()),
        "prefix_auxiliary_loss": float(objective["prefix_auxiliary_loss"].item()),
        "gradient_norm": grad_norm,
        "motion_gradient_norm": motion_gradient_norm,
        "checkpoint_path": str(checkpoint_path),
        "smoke_artifact_scope": "bounded smoke only; not a formal training artifact",
        "checkpoint_round_trip": "passed",
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "formal_training": "未执行",
        "validation": "未执行",
        "test": "未执行",
    }
    smoke_dir.mkdir(parents=True, exist_ok=True)
    (smoke_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded liquid DBB/CU gate and soft-cascade contract smoke.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--parent-limit", type=int, default=6)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    result = run(
        args.project_root.resolve(),
        config_path.resolve(),
        parent_limit=args.parent_limit,
        output_suffix=args.output_suffix,
        device_override=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
