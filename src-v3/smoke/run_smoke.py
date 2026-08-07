from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from radar_ln3.checkpoint import load_checkpoint, save_checkpoint
from radar_ln3.config import load_config, resolve_project_path
from radar_ln3.data import CQ08PointSequenceDataset, collate_point_sequences
from radar_ln3.losses import five_class_loss
from radar_ln3.model import RadarLiquidClassifier


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_model(config):
    return RadarLiquidClassifier(
        tr_value_dim=config.model.tr_value_dim, tr_missing_dim=config.model.tr_missing_dim,
        rd_quality_dim=config.model.rd_quality_dim, hidden_dim=config.model.hidden_dim,
        rd_embedding_dim=config.model.rd_embedding_dim, liquid_hidden_dim=config.model.liquid_hidden_dim,
        num_classes=config.model.num_classes, tau_min_seconds=config.model.liquid_tau_min_seconds,
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
    if not config.train.allow_parameter_updates:
        raise PermissionError(
            "this RADAR-LN3 configuration forbids parameter updates; audit views are forward-only"
        )
    _seed(config.train.seed)
    dataset = CQ08PointSequenceDataset(
        project_root, index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=config.data.split_name, max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center, rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=config.data.exclude_anomalous_parents, sample_limit=config.data.sample_limit,
        expected_schema_version=config.data.index_schema_version or config.schema_version,
    )
    device = torch.device("cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = _make_model(config).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    step_records: list[dict[str, object]] = []
    loader_indices = list(range(len(dataset)))
    for step in range(config.train.smoke_steps):
        selected = [loader_indices[(step * config.train.batch_size + offset) % len(loader_indices)] for offset in range(config.train.batch_size)]
        batch = collate_point_sequences([dataset[index] for index in selected])
        tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        start = time.perf_counter()
        output = model(
            tr_features=tensor_batch["tr_features"], tr_missing_mask=tensor_batch["tr_missing_mask"],
            tr_dt_seconds=tensor_batch["tr_dt_seconds"], tr_dt_valid=tensor_batch["tr_dt_valid"], rd_frames=tensor_batch["rd_frames"],
            rd_frame_mask=tensor_batch["rd_frame_mask"], rd_quality=tensor_batch["rd_quality"],
            tr_padding_mask=tensor_batch["tr_padding_mask"],
        )
        loss = five_class_loss(output["window_logits"], tensor_batch["labels"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=100.0).item())
        optimizer.step()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not torch.isfinite(loss).item() or not np.isfinite(grad_norm):
            raise ValueError("smoke produced non-finite loss or gradient")
        step_records.append({
            "step": step + 1, "loss": float(loss.item()), "gradient_norm_before_clip": grad_norm,
            "forward_backward_update_ms": elapsed_ms, "window_logits_shape": list(output["window_logits"].shape),
            "dt_clamped_count": int(output["dt_clamped_mask"].sum().item()),
        })
    checkpoint_path = project_root / f"artifacts/{config.schema_version}-cq08-tr-rd-prefix-dt-smoke/checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, config_name=config.config_name, step=config.train.smoke_steps)
    restored = _make_model(config).to(device)
    load_checkpoint(checkpoint_path, restored, map_location=device)
    result = {
        "status": "passed",
        "schema_version": config.schema_version,
        "device_requested": config.train.device,
        "device_used": str(device),
        "cuda_available": bool(torch.cuda.is_available()),
        "parent_count_used": len(dataset),
        "steps": config.train.smoke_steps,
        "step_records": step_records,
        "checkpoint_path": str(checkpoint_path.relative_to(project_root)).replace("\\", "/"),
        "formal_pretraining": "未执行",
        "formal_training": "未执行",
        "full_evaluation": "未执行",
        "production_p95_2s": "未验收；本结果仅为短时smoke测量",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a finite RADAR-LN3 CQ08 smoke training loop.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    run(args.project_root.resolve(), config_path.resolve())


if __name__ == "__main__":
    main()
