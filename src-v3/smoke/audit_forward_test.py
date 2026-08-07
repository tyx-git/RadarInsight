from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.config import load_config, resolve_project_path
from radar_ln3.data import CQ08PointSequenceDataset, build_point_dataloader
from radar_ln3.model import RadarLiquidClassifier


def _make_model(config) -> RadarLiquidClassifier:
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
    if config.train.allow_parameter_updates:
        raise AssertionError("audit forward test requires allow_parameter_updates=false")
    dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name="audit",
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=None,
        expected_schema_version=config.data.index_schema_version or config.schema_version,
    )
    loader = build_point_dataloader(dataset, batch_size=1, shuffle=False)
    model = _make_model(config).eval()
    device = torch.device("cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    parent_ids: list[str] = []
    output_shapes: list[list[int]] = []
    finite = True
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(
                tr_features=tensor_batch["tr_features"],
                tr_missing_mask=tensor_batch["tr_missing_mask"],
                tr_dt_seconds=tensor_batch["tr_dt_seconds"],
                tr_dt_valid=tensor_batch["tr_dt_valid"],
                rd_frames=tensor_batch["rd_frames"],
                rd_frame_mask=tensor_batch["rd_frame_mask"],
                rd_quality=tensor_batch["rd_quality"],
                tr_padding_mask=tensor_batch["tr_padding_mask"],
            )
            parent_ids.extend(batch["parent_sample_ids"])
            output_shapes.append(list(output["window_logits"].shape))
            finite = finite and bool(torch.isfinite(output["window_logits"]).all())
    if parent_ids != sorted(parent_ids):
        raise AssertionError("audit parent order is not deterministic")
    if set(parent_ids) != {"cq08|track|634", "cq08|track|785", "cq08|track|3338"}:
        raise AssertionError(f"unexpected audit parents: {parent_ids}")
    if not finite:
        raise AssertionError("audit forward produced non-finite logits")
    result = {
        "status": "passed",
        "config": str(config_path.relative_to(project_root)).replace("\\", "/"),
        "device": str(device),
        "parameter_updates_allowed": config.train.allow_parameter_updates,
        "parent_ids": parent_ids,
        "parent_count": len(parent_ids),
        "window_logits_shapes": output_shapes,
        "forward_only": True,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run forward-only inference on the RADAR-LN3-V1 anomaly audit view.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    run(args.project_root.resolve(), config_path.resolve())


if __name__ == "__main__":
    main()
