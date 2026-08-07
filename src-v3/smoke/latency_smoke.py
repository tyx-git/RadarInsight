from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from radar_ln3.config import load_config, resolve_project_path
from radar_ln3.data import CQ08PointSequenceDataset, collate_point_sequences
from radar_ln3.model import RadarLiquidClassifier


def _model(config) -> RadarLiquidClassifier:
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


def run(project_root: Path, config_path: Path, iterations: int = 30) -> dict[str, object]:
    config = load_config(config_path)
    dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name="train",
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=True,
        sample_limit=2,
        expected_schema_version=config.data.index_schema_version or config.schema_version,
    )
    batch = collate_point_sequences([dataset[0], dataset[1]])
    device = torch.device("cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu")
    tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    model = _model(config).to(device).eval()
    with torch.inference_mode():
        for _ in range(5):
            model(
                tr_features=tensor_batch["tr_features"], tr_missing_mask=tensor_batch["tr_missing_mask"],
                tr_dt_seconds=tensor_batch["tr_dt_seconds"], tr_dt_valid=tensor_batch["tr_dt_valid"], rd_frames=tensor_batch["rd_frames"],
                rd_frame_mask=tensor_batch["rd_frame_mask"], rd_quality=tensor_batch["rd_quality"],
                tr_padding_mask=tensor_batch["tr_padding_mask"],
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings_ms: list[float] = []
        for _ in range(max(5, int(iterations))):
            start = time.perf_counter()
            output = model(
                tr_features=tensor_batch["tr_features"], tr_missing_mask=tensor_batch["tr_missing_mask"],
                tr_dt_seconds=tensor_batch["tr_dt_seconds"], tr_dt_valid=tensor_batch["tr_dt_valid"], rd_frames=tensor_batch["rd_frames"],
                rd_frame_mask=tensor_batch["rd_frame_mask"], rd_quality=tensor_batch["rd_quality"],
                tr_padding_mask=tensor_batch["tr_padding_mask"],
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings_ms.append((time.perf_counter() - start) * 1000.0)
    if not torch.isfinite(output["window_logits"]).all():
        raise ValueError("latency smoke produced non-finite logits")
    values = np.asarray(timings_ms, dtype=np.float64)
    result = {
        "status": "passed",
        "device": str(device),
        "iterations": len(timings_ms),
        "batch_size": 2,
        "sequence_lengths": [int(value) for value in batch["lengths"].tolist()],
        "model_only_forward_ms": {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        },
        "scope": "focused model-only forward; excludes disk/RD decode/event aggregation/service scheduling",
        "end_to_end_p95_2s": "not_accepted",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure focused GPU model-only P95 latency for RADAR-LN3.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    run(args.project_root.resolve(), config_path.resolve(), args.iterations)


if __name__ == "__main__":
    main()
