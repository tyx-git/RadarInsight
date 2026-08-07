from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from . import LABELS
from .config import V3Config, resolve_project_path
from .data import CQ08PointSequenceDataset, build_point_dataloader
from .metrics import confidence_summary, grouped_summaries, summarize_records
from .model import RadarLiquidClassifier
from .stable import TargetStableStateMachine


SAMPLE_DETAIL_SCHEMA = "RADAR-LN3-V2-sample-detail-v1"
LOW_CONFIDENCE_THRESHOLD = 0.60


def strip_detailed_records(result: dict[str, Any]) -> dict[str, Any]:
    """Return the metric/report view without per-sample records."""

    summary = dict(result)
    summary.pop("records", None)
    return summary


def write_sample_detail_information(
    output_dir: str | Path,
    result: dict[str, Any],
    *,
    checkpoint_path: str | Path | None = None,
    epoch: int | None = None,
    selection_metric: str | None = None,
) -> Path:
    """Persist all evaluation records separately from metric summaries."""

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    records = result.get("records", {})
    if not isinstance(records, dict):
        raise ValueError("evaluation records must be a mapping")
    normalized_records = {
        "window": list(records.get("window", [])),
        "parent": list(records.get("parent", [])),
        "stable": list(records.get("stable", [])),
        "stable_unconfirmed": list(records.get("stable_unconfirmed", [])),
    }
    payload = {
        "schema_version": SAMPLE_DETAIL_SCHEMA,
        "split": result.get("split"),
        "confidence_threshold": LOW_CONFIDENCE_THRESHOLD,
        "selection": {
            "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
            "epoch": epoch,
            "metric": selection_metric,
        },
        "counts": {name: len(rows) for name, rows in normalized_records.items()},
        "records": normalized_records,
    }
    path = target_dir / "sample_detail_information.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _device(config: V3Config) -> torch.device:
    return torch.device("cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu")


def evaluate_dataset(
    model: RadarLiquidClassifier,
    dataset: CQ08PointSequenceDataset,
    *,
    config: V3Config,
    device: torch.device,
    batch_size: int | None = None,
    split_name: str,
) -> dict[str, Any]:
    model.eval()
    loader = build_point_dataloader(dataset, batch_size=batch_size or config.train.batch_size, shuffle=False)
    window_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    stable_records: list[dict[str, Any]] = []
    stable_unconfirmed_records: list[dict[str, Any]] = []
    dt_clamped = 0
    dt_zero = 0
    dt_invalid = 0
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
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
            probabilities = torch.softmax(output["sequence_logits"], dim=-1)
            for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][batch_index].item())
                true_id = int(batch["labels"][batch_index].item())
                stable = TargetStableStateMachine(
                    confirm_count=config.evaluation.stable_confirm_count,
                    ema_decay=config.evaluation.stable_ema_decay,
                )
                final_window: dict[str, Any] | None = None
                decision = None
                for event_index in range(length):
                    logits = output["sequence_logits"][batch_index, event_index]
                    event_probabilities = probabilities[batch_index, event_index]
                    pred_id = int(event_probabilities.argmax().item())
                    confidence = float(event_probabilities[pred_id].item())
                    window_row = {
                        "true_id": true_id,
                        "pred_id": pred_id,
                        "confidence": confidence,
                        "parent_sample_id": parent_id,
                        "track_id": batch["track_ids"][batch_index],
                        "source_name": batch["source_names"][batch_index],
                        "label_name": batch["label_names"][batch_index],
                        "split_name": split_name,
                        "event_index": event_index,
                        "window_length": event_index + 1,
                    }
                    window_records.append(window_row)
                    final_window = window_row
                    decision = stable.update(logits.cpu())
                if final_window is None:
                    raise ValueError(f"empty parent sequence reached evaluator: {parent_id}")
                final_window = dict(final_window)
                parent_records.append(final_window)
                if decision is None:
                    raise ValueError(f"stable state was not updated: {parent_id}")
                stable_base = {
                    "true_id": true_id,
                    "parent_sample_id": parent_id,
                    "track_id": batch["track_ids"][batch_index],
                    "source_name": batch["source_names"][batch_index],
                    "label_name": batch["label_names"][batch_index],
                    "split_name": split_name,
                    "stable_confirmed": decision.stable_confirmed,
                    "event_count": length,
                }
                if decision.stable_confirmed:
                    stable_records.append({
                        **stable_base,
                        "pred_id": int(decision.stable_label_id),
                        "confidence": float(decision.stable_confidence),
                    })
                else:
                    stable_unconfirmed_records.append({
                        **stable_base,
                        "pred_id": None,
                        "confidence": None,
                        "candidate_label_id": decision.candidate_label_id,
                        "candidate_count": decision.candidate_count,
                    })
            dt_clamped += int(output["dt_clamped_mask"].sum().item())
            dt_zero += int(output["dt_zero_mask"].sum().item())
            dt_invalid += int(output["dt_invalid_mask"].sum().item())
    stable_confirmed_records = stable_records
    bucketed = []
    for row in window_records:
        length = int(row["window_length"])
        bucket = "le_10" if length <= 10 else "11_20" if length <= 20 else "gt_20"
        item = dict(row)
        item["window_length_bucket"] = bucket
        bucketed.append(item)
    return {
        "split": split_name,
        "dataset_parent_count": len(dataset),
        "window_count": len(window_records),
        "parent_count": len(parent_records),
        "stable_confirmed_parent_count": len(stable_confirmed_records),
        "stable_unconfirmed_parent_count": len(stable_unconfirmed_records),
        "stable_coverage": len(stable_confirmed_records) / len(parent_records) if parent_records else 0.0,
        "window_metrics": summarize_records(window_records, scope=f"{split_name}:causal_prefix_windows"),
        "parent_metrics": summarize_records(parent_records, scope=f"{split_name}:parent_final_prefix"),
        "stable_metrics": summarize_records(stable_confirmed_records, scope=f"{split_name}:target_stable_confirmed") if stable_confirmed_records else None,
        "stable_confirmed_metrics": summarize_records(stable_confirmed_records, scope=f"{split_name}:target_stable_confirmed_only") if stable_confirmed_records else None,
        "window_length_bucket_metrics": grouped_summaries(bucketed, field="window_length_bucket", scope_prefix=f"{split_name}:window_length_bucket"),
        "source_metrics": grouped_summaries(window_records, field="source_name", scope_prefix=f"{split_name}:source"),
        "parent_source_metrics": grouped_summaries(parent_records, field="source_name", scope_prefix=f"{split_name}:parent_source"),
        "session_metrics": {
            "status": "not_available",
            "reason": "V1 point index and Dataset contract contain no session/capture field; session metrics were not estimated",
        },
        "confidence": {
            "window": confidence_summary(window_records),
            "parent": confidence_summary(parent_records),
            "stable": confidence_summary(stable_records),
        },
        "dt_quality": {
            "dt_clamped_event_count": dt_clamped,
            "dt_zero_event_count": dt_zero,
            "dt_invalid_event_count": dt_invalid,
        },
        "class_names": list(LABELS),
        "records": {
            "window": window_records,
            "parent": parent_records,
            "stable": stable_records,
            "stable_unconfirmed": stable_unconfirmed_records,
        },
    }


def evaluate_checkpoint(
    project_root: Path,
    config: V3Config,
    checkpoint_path: Path,
    *,
    split_name: str,
    batch_size: int | None = None,
) -> dict[str, Any]:
    from .checkpoint import load_formal_checkpoint
    from .contracts import verify_formal_data_contract
    from .factory import build_model

    contract = verify_formal_data_contract(project_root, config)
    if split_name == config.data.test_split_name and not config.train.evaluate_test:
        raise PermissionError("test evaluation is disabled by the formal configuration")
    dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=split_name,
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=None,
        expected_schema_version=config.data.index_schema_version or "RADAR-LN3-V1",
    )
    device = _device(config)
    model = build_model(config).to(device)
    checkpoint = load_formal_checkpoint(
        checkpoint_path,
        model,
        map_location=device,
        expected_schema=config.train.checkpoint_schema,
        expected_model_contract={
            "tr_value_dim": config.model.tr_value_dim,
            "tr_missing_dim": config.model.tr_missing_dim,
            "rd_quality_dim": config.model.rd_quality_dim,
            "rd_shape": [config.model.rd_height, config.model.rd_width],
            "num_classes": config.model.num_classes,
            "dt_valid_dim": config.model.dt_valid_dim,
            "rd_quality_schema": config.data.rd_quality_schema,
            "liquid_tau_min_seconds": config.model.liquid_tau_min_seconds,
            "liquid_tau_max_seconds": config.model.liquid_tau_max_seconds,
            "liquid_zero_dt_event_floor_seconds": config.model.liquid_zero_dt_event_floor_seconds,
            "liquid_dt_max_seconds": config.model.liquid_dt_max_seconds,
            "liquid_invalid_dt_update_mode": config.model.liquid_invalid_dt_update_mode,
            "liquid_invalid_dt_observation_gain": config.model.liquid_invalid_dt_observation_gain,
            "liquid_state_normalization": config.model.liquid_state_normalization,
            "rd_frame_aggregation": config.model.rd_frame_aggregation,
            "loss_mode": config.train.loss_mode,
        },
    )
    result = evaluate_dataset(model, dataset, config=config, device=device, batch_size=batch_size, split_name=split_name)
    result["status"] = "passed"
    result["device"] = str(device)
    result["checkpoint_path"] = str(checkpoint_path)
    result["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
    result["checkpoint_global_step"] = int(checkpoint.get("global_step", -1))
    result["data_contract"] = contract
    return result


def write_report(project_root: Path, config: V3Config, result: dict[str, Any], split_name: str) -> Path:
    output_dir = resolve_project_path(project_root, config.train.formal_artifact_dir or "artifacts/RADAR-LN3-V2-formal")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_detail_path = write_sample_detail_information(
        output_dir,
        result,
        checkpoint_path=result.get("checkpoint_path"),
        epoch=result.get("checkpoint_epoch"),
        selection_metric=config.train.checkpoint_selection_metric,
    )
    result["sample_detail_information"] = str(sample_detail_path)
    path = output_dir / f"evaluation-{split_name}.json"
    path.write_text(json.dumps(strip_detailed_records(result), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "LOW_CONFIDENCE_THRESHOLD",
    "SAMPLE_DETAIL_SCHEMA",
    "evaluate_checkpoint",
    "evaluate_dataset",
    "strip_detailed_records",
    "write_report",
    "write_sample_detail_information",
]
