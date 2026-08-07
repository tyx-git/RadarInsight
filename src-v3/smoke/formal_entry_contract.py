from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from radar_ln3.config import load_config, resolve_project_path
from radar_ln3.contracts import verify_formal_data_contract
from radar_ln3.data import CQ08PointSequenceDataset, collate_point_sequences
from radar_ln3.evaluator import evaluate_dataset
from radar_ln3.factory import build_model
from radar_ln3.losses import causal_prefix_loss, five_class_loss
from radar_ln3.checkpoint import load_formal_checkpoint
from train.formal import train_formal


def run(project_root: Path, config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    contract = verify_formal_data_contract(project_root, config)
    entry_artifact_dir = f"artifacts/{config.config_name.replace('-formal-supervised', '-entry-contract')}"
    entry_config = replace(
        config,
        train=replace(
            config.train,
            checkpoint_path=f"{entry_artifact_dir}/best_model.pt",
            last_checkpoint_path=f"{entry_artifact_dir}/last_model.pt",
            formal_artifact_dir=entry_artifact_dir,
        ),
    )
    result = train_formal(project_root, entry_config, max_steps_override=1, epochs_override=1, skip_validation=True)
    checkpoint_path = resolve_project_path(project_root, entry_config.train.checkpoint_path)
    device = torch.device("cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device).eval()
    checkpoint = load_formal_checkpoint(
        checkpoint_path,
        model,
        map_location=device,
        expected_schema=entry_config.train.checkpoint_schema,
    )
    train_dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name="train",
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=2,
        expected_schema_version=config.data.index_schema_version or "RADAR-LN3-V1",
    )
    batch = collate_point_sequences([train_dataset[0], train_dataset[1]])
    tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    with torch.inference_mode():
        output = model(
            tr_features=tensor_batch["tr_features"],
            tr_missing_mask=tensor_batch["tr_missing_mask"],
            tr_dt_seconds=tensor_batch["tr_dt_seconds"],
            tr_dt_valid=tensor_batch.get("tr_dt_valid"),
            rd_frames=tensor_batch["rd_frames"],
            rd_frame_mask=tensor_batch["rd_frame_mask"],
            rd_quality=tensor_batch["rd_quality"],
            tr_padding_mask=tensor_batch["tr_padding_mask"],
        )
        if entry_config.train.loss_mode == "causal_prefix":
            weighted_loss = causal_prefix_loss(
                output["sequence_logits"],
                tensor_batch["labels"],
                tensor_batch["tr_padding_mask"],
                torch.tensor(entry_config.train.class_weights, device=device),
            )
        else:
            weighted_loss = five_class_loss(
                output["window_logits"],
                tensor_batch["labels"],
                torch.tensor(entry_config.train.class_weights, device=device),
            )
    val_dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name="val",
        # Keep this as a bounded evaluator contract: one parent and one
        # causal point are enough to verify the formal evaluator path without
        # turning the smoke into a full validation run.
        max_sequence_length=1,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=1,
        expected_schema_version=config.data.index_schema_version or "RADAR-LN3-V1",
    )
    val_result = evaluate_dataset(model, val_dataset, config=entry_config, device=device, split_name="val", batch_size=1)
    if not torch.isfinite(output["window_logits"]).all() or not torch.isfinite(weighted_loss):
        raise AssertionError("formal entry contract produced non-finite output")
    if entry_config.train.loss_mode == "causal_prefix":
        invalid_count = int(output["dt_invalid_mask"].sum().item())
        expected_invalid_count = int((~tensor_batch["tr_dt_valid"] & ~tensor_batch["tr_padding_mask"]).sum().item())
        if invalid_count != expected_invalid_count:
            raise AssertionError(f"dt invalid mask mismatch: model={invalid_count}, batch={expected_invalid_count}")
    else:
        invalid_count = 0
    output_result = {
        "status": "passed",
        "config": str(config_path),
        "artifact_scope": "bounded formal entry contract only; not a formal training artifact",
        "device": str(device),
        "data_contract": contract,
        "training_steps": result["global_step"],
        "formal_checkpoint_schema": checkpoint["checkpoint_schema"],
        "checkpoint_path": str(checkpoint_path),
        "train_batch_parent_count": len(train_dataset),
        "train_batch_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "window_logits_shape": list(output["window_logits"].shape),
        "weighted_loss": float(weighted_loss.item()),
        "loss_mode": entry_config.train.loss_mode,
        "tr_dt_valid_shape": list(batch["tr_dt_valid"].shape),
        "dt_invalid_event_count": invalid_count,
        "validation_parent_count": val_result["parent_count"],
        "validation_metrics_are_focused_only": True,
        "full_training": "未执行",
        "full_validation": "未执行",
        "test_evaluation": "未执行",
    }
    print(json.dumps(output_result, ensure_ascii=False, indent=2))
    return output_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded formal trainer/evaluator/checkpoint contract smoke.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    run(project_root, config_path.resolve())


if __name__ == "__main__":
    main()
