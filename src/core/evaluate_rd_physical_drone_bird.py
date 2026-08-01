from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.rd_physical_window_dataset import QUALITY_FEATURE_NAMES, PhysicalRDWindowDataset, collate_physical_rd_windows
from .rd_physical_artifacts import load_physical_rd_checkpoint
from .rd_physical_drone_bird_trainer import RD_ONLY_LABELS, _metrics, _parent_balance_metadata, compute_parent_evidence_weights


def _confusion_matrix(logits: torch.Tensor, labels: torch.Tensor) -> list[list[int]]:
    predictions = logits.argmax(dim=-1)
    matrix = [[0, 0], [0, 0]]
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(truth)][int(prediction)] += 1
    return matrix


def _label_index(label_name: object) -> int:
    try:
        return RD_ONLY_LABELS.index(str(label_name))
    except ValueError as error:
        raise ValueError(f"physical RD evaluator received an invalid label: {label_name!r}") from error


def _grouped_metrics(
    predictions: list[dict[str, object]],
    *,
    field_name: str | None,
    unavailable_reason: str,
) -> dict[str, object]:
    if field_name is None:
        return {"available": False, "reason": unavailable_reason, "groups": {}}
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        value = str(row.get(field_name, "")).strip()
        if not value or value.lower() == "nan":
            return {
                "available": False,
                "field_name": field_name,
                "reason": f"evaluation metadata field {field_name!r} is incomplete",
                "groups": {},
            }
        groups[value].append(row)
    payload: dict[str, object] = {}
    for value, rows in sorted(groups.items()):
        logits = torch.tensor([[row["logit_drone"], row["logit_bird"]] for row in rows], dtype=torch.float32)
        labels = torch.tensor([_label_index(row["label_name_true"]) for row in rows], dtype=torch.long)
        weights = torch.tensor([float(row["aggregation_weight"]) for row in rows], dtype=torch.float32)
        payload[value] = _metrics(
            logits,
            labels,
            [str(row["parent_sample_id"]) for row in rows],
            weights,
        )
    return {
        "available": True,
        "field_name": field_name,
        "group_count": len(payload),
        "groups": payload,
    }


def _balanced_smoke_frame(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows < 1:
        raise ValueError("balanced smoke max_rows must be positive")
    by_label = {
        label: frame.index[frame["label_name"].astype(str).eq(label)].tolist()
        for label in RD_ONLY_LABELS
    }
    if any(not indices for indices in by_label.values()):
        raise ValueError("balanced physical RD smoke requires both DroneTarget and BirdTarget")
    positions = {label: 0 for label in RD_ONLY_LABELS}
    selected: list[int] = []
    while len(selected) < max_rows:
        progressed = False
        for label in RD_ONLY_LABELS:
            position = positions[label]
            indices = by_label[label]
            if position < len(indices) and len(selected) < max_rows:
                selected.append(int(indices[position]))
                positions[label] = position + 1
                progressed = True
        if not progressed:
            break
    return frame.loc[selected].reset_index(drop=True)


def evaluate_physical_rd_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    split_name: str = "val",
    device_name: str = "cuda",
    batch_size: int | None = None,
    num_workers: int = 0,
    max_batches: int = 0,
) -> dict[str, object]:
    if split_name != "val":
        raise ValueError("physical RD evaluator only permits split_name='val'; test evaluation requires separate approval")
    if not torch.cuda.is_available() or not str(device_name).startswith("cuda"):
        raise RuntimeError("physical RD evaluator requires CUDA")
    if max_batches < 0:
        raise ValueError("max_batches must be non-negative")
    project_root = Path.cwd()
    device = torch.device(device_name)
    loaded = load_physical_rd_checkpoint(checkpoint_path, project_root, device)
    config = loaded.config
    data_root = project_root / "data"
    dataset = PhysicalRDWindowDataset(
        data_root,
        index_name=config.index_name,
        split_name=split_name,
        normalization_stats_name=config.normalization_stats_name,
        include_missing_rd=config.include_missing_rd,
    )
    full_dataset_window_count = int(len(dataset))
    effective_batch_size = int(batch_size or config.batch_size)
    if max_batches > 0:
        dataset.frame = _balanced_smoke_frame(dataset.frame, max_batches * effective_batch_size)
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_physical_rd_windows,
    )
    logits_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    predictions: list[dict[str, object]] = []
    aggregation_weight_rows: list[torch.Tensor] = []
    present_rows: list[torch.Tensor] = []
    balance = _parent_balance_metadata(dataset, config)
    metadata_frame = dataset.frame.copy()
    metadata_frame["_sample_id"] = metadata_frame["sample_id"].astype(str)
    source_field = "source_name" if "source_name" in metadata_frame.columns else None
    session_fields = [name for name in ("session_id", "session_name") if name in metadata_frame.columns]
    if len(session_fields) > 1:
        raise ValueError(f"physical RD index has ambiguous session fields: {session_fields}")
    session_field = session_fields[0] if session_fields else None
    evaluation_metadata = metadata_frame.set_index("_sample_id").to_dict(orient="index")
    evaluated_batch_count = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            if max_batches > 0 and batch_index > max_batches:
                break
            evaluated_batch_count = batch_index
            output = loaded.model(
                batch.features.to(device, non_blocking=True),
                batch.time_seconds.to(device, non_blocking=True),
                batch.padding_mask.to(device, non_blocking=True),
                batch.quality_features.to(device, non_blocking=True),
                batch.range_offset_m.to(device, non_blocking=True),
                batch.velocity_mps.to(device, non_blocking=True),
                batch.raw_amplitude_db.to(device, non_blocking=True),
            )
            logits = output.logits.detach().cpu()
            probabilities = torch.softmax(logits, dim=-1)
            predicted = logits.argmax(dim=-1)
            present = batch.quality_features[:, QUALITY_FEATURE_NAMES.index("rd_present")].bool()
            batch_aggregation_weights = compute_parent_evidence_weights(
                batch,
                balance["parent_window_counts"],
                prefix_max_loss_weight=config.prefix_max_loss_weight,
                evidence_target_frames=config.evidence_target_frames,
                parent_evidence_sums=balance["parent_evidence_sums"],
                parent_balance_weights=balance["parent_balance_weights"],
            ).detach().cpu()
            for index, sample_id in enumerate(batch.sample_ids):
                sample_metadata = evaluation_metadata[str(sample_id)]
                length = int(batch.lengths[index].item())
                times = batch.time_seconds[index, :length]
                prediction: dict[str, object] = {
                        "sample_id": str(sample_id),
                        "parent_sample_id": str(batch.parent_sample_ids[index]),
                        "split_name": str(batch.split_names[index]),
                        "label_name_true": str(batch.label_names[index]),
                        "label_name_pred": RD_ONLY_LABELS[int(predicted[index].item())],
                        "logit_drone": float(logits[index, 0]),
                        "logit_bird": float(logits[index, 1]),
                        "probability_drone": float(probabilities[index, 0]),
                        "probability_bird": float(probabilities[index, 1]),
                        "is_correct": bool(predicted[index].item() == batch.labels[index].item()),
                        "rd_present": bool(present[index].item()),
                        "aggregation_weight": float(batch_aggregation_weights[index].item()),
                        "window_mode": str(batch.window_modes[index]),
                        "is_prefix_window": bool(batch.is_prefix_windows[index].item()),
                        "rd_window_frame_count": int(batch.window_frame_counts[index].item()),
                        "rd_time_start_seconds": float(times[0].item()) if length else None,
                        "rd_time_end_seconds": float(times[-1].item()) if length else None,
                        "rd_window_duration_seconds": float((times[-1] - times[0]).item()) if length else 0.0,
                        "quality_metadata": [float(value) for value in batch.quality_metadata[index].tolist()],
                    }
                if source_field is not None:
                    prediction[source_field] = str(sample_metadata[source_field])
                if session_field is not None:
                    prediction[session_field] = str(sample_metadata[session_field])
                predictions.append(prediction)
            logits_rows.append(logits)
            labels_rows.append(batch.labels.detach().cpu())
            parent_ids.extend(batch.parent_sample_ids)
            present_rows.append(present.detach().cpu())
            aggregation_weight_rows.append(batch_aggregation_weights)
    if not logits_rows:
        raise RuntimeError("physical RD evaluator produced no batches")
    logits_tensor = torch.cat(logits_rows)
    labels_tensor = torch.cat(labels_rows)
    present_tensor = torch.cat(present_rows)
    present_predictions = [row for row in predictions if bool(row["rd_present"])]
    full_split_evaluated = max_batches == 0 and len(predictions) == full_dataset_window_count
    evaluation_mode = "full_validation" if full_split_evaluated else "restricted_smoke_validation"
    payload = {
        "scope": (
            "physical RD-only natural validation; closed-set Drone/Bird diagnostic, no Unknown/OOD rejection, "
            "no fusion or test evaluation"
            if full_split_evaluated
            else "physical RD-only restricted smoke validation; incomplete split, no performance claim, no fusion or test evaluation"
        ),
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_schema_version": loaded.checkpoint["checkpoint_schema_version"],
        "checkpoint_epoch": int(loaded.checkpoint["epoch"]),
        "artifact_manifest": loaded.artifacts.manifest,
        "split_name": split_name,
        "metrics": _metrics(logits_tensor, labels_tensor, parent_ids, torch.cat(aggregation_weight_rows), present_tensor),
        "confusion_matrix": {"labels": list(RD_ONLY_LABELS), "matrix": _confusion_matrix(logits_tensor[present_tensor], labels_tensor[present_tensor])},
        "evaluation_levels": {
            "source": _grouped_metrics(
                present_predictions,
                field_name=source_field,
                unavailable_reason="active window index has no source_name field",
            ),
            "session": _grouped_metrics(
                present_predictions,
                field_name=session_field,
                unavailable_reason="active window index has no session_id or session_name field; parent metrics are not session metrics",
            ),
            "online_stable_decision": {
                "available": False,
                "reason": "active Physical RD-only classifier is not connected to a production streaming stable-decision state machine",
            },
        },
        "evaluation_completion": {
            "mode": evaluation_mode,
            "full_split_evaluated": full_split_evaluated,
            "dataset_window_count": full_dataset_window_count,
            "loaded_smoke_subset_window_count": int(len(dataset)),
            "evaluated_window_count": int(len(predictions)),
            "excluded_by_batch_limit_count": int(full_dataset_window_count - len(predictions)),
            "loader_batch_count": int(len(loader)),
            "evaluated_batch_count": int(evaluated_batch_count),
            "max_batches": int(max_batches),
        },
        "predictions": predictions,
        "missing_rd_window_count": int((~present_tensor).sum()),
        "rejection_capability": {
            "supports_rejection": False,
            "unknown_target_output": False,
            "abstain_output": False,
            "threshold": None,
            "status": "当前支路不具备已验证的拒识能力",
        },
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_csv_path = output_path.with_suffix(".predictions.csv")
    pd.DataFrame(predictions).to_csv(prediction_csv_path, index=False)
    payload["prediction_csv"] = prediction_csv_path.as_posix()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a physical RD-only checkpoint on natural validation only.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="0 evaluates the complete validation split; positive values are restricted smoke-only and cannot support performance claims.",
    )
    args = parser.parse_args()
    result = evaluate_physical_rd_checkpoint(
        Path(args.checkpoint),
        Path(args.output),
        split_name=args.split,
        device_name=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
    )
    summary = {key: value for key, value in result.items() if key != "predictions"}
    summary["output"] = Path(args.output).as_posix()
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
