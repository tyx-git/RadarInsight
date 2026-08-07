from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_ln3.checkpoint import load_checkpoint, save_checkpoint
from radar_ln3.config import load_config, resolve_project_path
from radar_ln3.data import CQ08PointSequenceDataset, build_point_dataloader, collate_point_sequences
from radar_ln3.losses import five_class_loss
from radar_ln3.losses import causal_prefix_loss
from radar_ln3.model import RadarLiquidClassifier
from radar_ln3.stable import TargetStableStateMachine
from radar_ln3.streaming import OnlineTargetClassifier


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_from_config(config) -> RadarLiquidClassifier:
    return RadarLiquidClassifier(
        tr_value_dim=config.model.tr_value_dim,
        tr_missing_dim=config.model.tr_missing_dim,
        rd_quality_dim=config.model.rd_quality_dim,
        hidden_dim=config.model.hidden_dim,
        rd_embedding_dim=config.model.rd_embedding_dim,
        liquid_hidden_dim=config.model.liquid_hidden_dim,
        num_classes=config.model.num_classes,
        tau_min_seconds=config.model.liquid_tau_min_seconds,
        tau_max_seconds=config.model.liquid_tau_max_seconds,
        zero_dt_event_floor_seconds=config.model.liquid_zero_dt_event_floor_seconds,
        dt_max_seconds=config.model.liquid_dt_max_seconds,
        dt_valid_dim=config.model.dt_valid_dim,
        liquid_invalid_dt_update_mode=config.model.liquid_invalid_dt_update_mode,
        liquid_invalid_dt_observation_gain=config.model.liquid_invalid_dt_observation_gain,
        liquid_state_normalization=config.model.liquid_state_normalization,
        rd_frame_aggregation=config.model.rd_frame_aggregation,
        rd_quality_schema=config.data.rd_quality_schema,
    )


def run(project_root: Path, config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    _seed(config.train.seed)
    dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=config.data.split_name,
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=config.data.exclude_anomalous_parents,
        sample_limit=max(2, config.data.sample_limit or 2),
        expected_schema_version=config.data.index_schema_version or config.schema_version,
    )
    items = [dataset[index] for index in range(min(2, len(dataset)))]
    batch = collate_point_sequences(items)
    loader = build_point_dataloader(dataset, batch_size=2, shuffle=False)
    loader_batch = next(iter(loader))
    if loader_batch["tr_features"].shape[0] != 2 or loader_batch["rd_frames"].shape[-3:] != (2, 21, 384):
        raise AssertionError("v3 DataLoader/collate contract failed")
    assert batch["tr_features"].shape[0] == len(items)
    assert batch["tr_features"].shape[-1] == 14
    assert batch["tr_missing_mask"].shape[-1] == 11
    assert batch["rd_frames"].shape[2:] == (12, 2, 21, 384)
    assert batch["rd_frame_mask"].shape == batch["rd_frames"].shape[:3]
    assert batch["tr_padding_mask"].shape == batch["tr_dt_seconds"].shape
    assert batch["tr_dt_valid"].shape == batch["tr_dt_seconds"].shape
    assert torch.isfinite(batch["tr_features"]).all()
    assert torch.isfinite(batch["rd_frames"]).all()
    if config.schema_version == "RADAR-LN3-V3" and not bool((~batch["tr_dt_valid"] & ~batch["tr_padding_mask"]).any()):
        raise AssertionError("V3 smoke batch did not contain an invalid observed dt")
    device_name = config.train.device
    device = torch.device("cuda" if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = _model_from_config(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    start = time.perf_counter()
    output = model(
        tr_features=batch["tr_features"],
        tr_missing_mask=batch["tr_missing_mask"],
        tr_dt_seconds=batch["tr_dt_seconds"],
        tr_dt_valid=batch["tr_dt_valid"],
        rd_frames=batch["rd_frames"],
        rd_frame_mask=batch["rd_frame_mask"],
        rd_quality=batch["rd_quality"],
        tr_padding_mask=batch["tr_padding_mask"],
    )
    forward_ms = (time.perf_counter() - start) * 1000.0
    loss = five_class_loss(output["window_logits"], batch["labels"])
    prefix_loss = causal_prefix_loss(output["sequence_logits"], batch["labels"], batch["tr_padding_mask"])
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0).item())
    optimizer.step()
    if not torch.isfinite(loss).item() or not np.isfinite(gradient_norm):
        raise ValueError("contract test produced a non-finite loss or gradient")
    if not torch.isfinite(prefix_loss).item():
        raise ValueError("causal prefix loss is non-finite")
    # Prefix invariance checks causal behavior: earlier logits cannot depend on future points.
    model.eval()
    with torch.no_grad():
        full = model(
            tr_features=batch["tr_features"], tr_missing_mask=batch["tr_missing_mask"], tr_dt_seconds=batch["tr_dt_seconds"],
            tr_dt_valid=batch["tr_dt_valid"], rd_frames=batch["rd_frames"], rd_frame_mask=batch["rd_frame_mask"], rd_quality=batch["rd_quality"], tr_padding_mask=batch["tr_padding_mask"],
        )
        prefix_length = min(2, int(batch["lengths"][0].item()))
        prefix = {key: value[:, :prefix_length] for key, value in {
            "tr_features": batch["tr_features"], "tr_missing_mask": batch["tr_missing_mask"], "tr_dt_seconds": batch["tr_dt_seconds"], "tr_dt_valid": batch["tr_dt_valid"],
            "rd_frames": batch["rd_frames"], "rd_frame_mask": batch["rd_frame_mask"], "rd_quality": batch["rd_quality"], "tr_padding_mask": batch["tr_padding_mask"],
        }.items()}
        prefix["tr_padding_mask"] = torch.zeros_like(prefix["tr_padding_mask"])
        prefix_output = model(**prefix)
        if not torch.allclose(full["sequence_logits"][:, prefix_length - 1], prefix_output["window_logits"], atol=1.0e-5, rtol=1.0e-5):
            raise AssertionError("causal prefix invariance failed")
        negative_dt = batch["tr_dt_seconds"].clone()
        negative_dt[:, 0] = -1.0
        negative_output = model(
            tr_features=batch["tr_features"], tr_missing_mask=batch["tr_missing_mask"], tr_dt_seconds=negative_dt, tr_dt_valid=batch["tr_dt_valid"],
            rd_frames=batch["rd_frames"], rd_frame_mask=batch["rd_frame_mask"], rd_quality=batch["rd_quality"], tr_padding_mask=batch["tr_padding_mask"],
        )
        if not bool(negative_output["dt_clamped_mask"][:, 0].all()):
            raise AssertionError("negative dt was not exposed as clamped")
    checkpoint_path = project_root / f"artifacts/{config.config_name.replace('-formal-supervised', '-contract-smoke')}/checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, config_name=config.config_name, step=1)
    restored = _model_from_config(config).to(device)
    restored.eval()
    load_checkpoint(checkpoint_path, restored, map_location=device)
    with torch.no_grad():
        restored_output = restored(
            tr_features=batch["tr_features"], tr_missing_mask=batch["tr_missing_mask"], tr_dt_seconds=batch["tr_dt_seconds"], tr_dt_valid=batch["tr_dt_valid"],
            rd_frames=batch["rd_frames"], rd_frame_mask=batch["rd_frame_mask"], rd_quality=batch["rd_quality"], tr_padding_mask=batch["tr_padding_mask"],
        )
    if not torch.allclose(full["window_logits"], restored_output["window_logits"], atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError("checkpoint round-trip changed logits")
    state = TargetStableStateMachine(confirm_count=2)
    decision_a = state.update(full["window_logits"][0].cpu())
    decision_b = state.update(full["window_logits"][0].cpu())
    if decision_b.update_count != 2:
        raise AssertionError("stable state machine did not update causally")
    online = OnlineTargetClassifier(restored, confirm_count=2)
    online_item = items[0]
    online_batch = collate_point_sequences([online_item])
    online_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in online_batch.items()}
    online_result = online.step(
        tr_features=online_batch["tr_features"][:, 0],
        tr_missing_mask=online_batch["tr_missing_mask"][:, 0],
        tr_dt_seconds=online_batch["tr_dt_seconds"][:, 0],
        tr_dt_valid=online_batch["tr_dt_valid"][:, 0],
        rd_frames=online_batch["rd_frames"][:, 0],
        rd_frame_mask=online_batch["rd_frame_mask"][:, 0],
        rd_quality=online_batch["rd_quality"][:, 0],
    )
    if online_result.window_logits.shape != (1, 5) or not torch.isfinite(online_result.window_logits).all():
        raise AssertionError("online point inference contract failed")
    result = {
        "status": "passed",
        "device": str(device),
        "dataset_parent_count_selected": len(dataset),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items() if isinstance(value, torch.Tensor)},
        "forward_ms": forward_ms,
        "loss": float(loss.item()),
        "causal_prefix_loss": float(prefix_loss.item()),
        "gradient_norm_before_clip": gradient_norm,
        "checkpoint_path": str(checkpoint_path.relative_to(project_root)).replace("\\", "/"),
        "causal_prefix_invariance": True,
        "negative_dt_clamp": True,
        "stable_decisions": [decision_a.stable_label_name, decision_b.stable_label_name],
        "dataloader_contract": True,
        "online_point_inference": {
            "window_logits_shape": list(online_result.window_logits.shape),
            "window_label_name": online_result.window_label_name,
            "stable_label_name": online_result.stable_label_name,
        },
        "formal_training": "未执行",
        "full_evaluation": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RADAR-LN3 shape, mask, forward, loss, gradient and checkpoint tests.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    run(args.project_root.resolve(), config_path.resolve())


if __name__ == "__main__":
    main()
