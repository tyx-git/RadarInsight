from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def run(project_root: Path, config_path: Path) -> dict[str, object]:
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
        sample_limit=None,
        expected_schema_version=config.data.index_schema_version or config.schema_version,
    )
    selected_parent = None
    selected_item = None
    selected_event = None
    for index, parent_id in enumerate(dataset.parent_ids):
        item = dataset[index]
        missing_events = (~item.rd_frame_mask.any(dim=1)).nonzero(as_tuple=False).flatten()
        if missing_events.numel():
            selected_parent = parent_id
            selected_item = item
            selected_event = int(missing_events[0].item())
            break
    if selected_item is None or selected_parent is None or selected_event is None:
        raise AssertionError("no real CQ08 missing-RD event was selected")
    batch = collate_point_sequences([selected_item])
    if bool(batch["rd_frame_mask"][0, selected_event].any()):
        raise AssertionError("missing RD event has a nonzero frame mask")
    if float(batch["rd_quality"][0, selected_event, 0].item()) != 0.0:
        raise AssertionError("missing RD event quality present flag is not zero")
    if float(batch["rd_quality"][0, selected_event, 3].item()) != 1.0:
        raise AssertionError("missing RD event quality missing flag is not one")
    model = _model(config).eval()
    with torch.no_grad():
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
    if not torch.isfinite(output["window_logits"]).all():
        raise AssertionError("missing RD forward produced non-finite logits")
    result = {
        "status": "passed",
        "parent_sample_id": selected_parent,
        "event_index": selected_event,
        "sequence_length": int(selected_item.event_count),
        "missing_rd_event_count": int((~selected_item.rd_frame_mask.any(dim=1)).sum().item()),
        "missing_event_rd_mask_sum": int(batch["rd_frame_mask"][0, selected_event].sum().item()),
        "missing_event_quality": batch["rd_quality"][0, selected_event].tolist(),
        "window_logits_shape": list(output["window_logits"].shape),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the real CQ08 missing-RD mask branch.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else args.project_root / args.config
    run(args.project_root.resolve(), config_path.resolve())


if __name__ == "__main__":
    main()
