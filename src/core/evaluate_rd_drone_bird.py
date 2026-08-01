from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .device import resolve_cuda_device
from .experts import LowAltitudeSuspectExpert
from .metrics import compute_classification_metrics
from .rd_drone_bird_config import RDDroneBirdDiagnosticConfig
from .rd_drone_bird_trainer import _make_loader, _select_rd_only_ids
from .system_metrics import compute_per_class_metrics


RD_ONLY_LABELS = ["DroneTarget", "BirdTarget"]


def _causal_parent_payload(predictions: list[dict[str, object]]) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for prediction in predictions:
        grouped[str(prediction["parent_sample_id"])].append(prediction)
    final_logits: list[torch.Tensor] = []
    targets: list[int] = []
    parent_records: list[dict[str, object]] = []
    flip_counts: list[int] = []
    first_correct_seconds: list[float] = []
    stable_correct_seconds: list[float] = []
    for parent_id, rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda row: (
                float(row["window_end_seconds"]),
                float(row["window_start_seconds"]),
                str(row["sample_id"]),
            ),
        )
        truth_values = {str(row["label_name_true"]) for row in ordered}
        if len(truth_values) != 1:
            raise ValueError(f"parent has inconsistent validation labels: {parent_id}")
        truth = next(iter(truth_values))
        logits = torch.tensor(
            [[float(row["bird_logit_drone"]), float(row["bird_logit_bird"])] for row in ordered],
            dtype=torch.float32,
        )
        cumulative_logits = logits.cumsum(dim=0) / torch.arange(1, len(ordered) + 1, dtype=torch.float32).unsqueeze(1)
        prefix_predictions = cumulative_logits.argmax(dim=-1).tolist()
        truth_index = RD_ONLY_LABELS.index(truth)
        correctness = [int(prediction) == truth_index for prediction in prefix_predictions]
        suffix_correct = [False] * len(correctness)
        running = True
        for index in range(len(correctness) - 1, -1, -1):
            running = running and correctness[index]
            suffix_correct[index] = running
        transition_count = sum(
            previous != current
            for previous, current in zip(prefix_predictions, prefix_predictions[1:])
        )
        first_correct_index = next((index for index, correct in enumerate(correctness) if correct), None)
        stable_correct_index = next((index for index, correct in enumerate(suffix_correct) if correct), None)
        if first_correct_index is not None:
            first_correct_seconds.append(float(ordered[first_correct_index]["window_end_seconds"]))
        if stable_correct_index is not None:
            stable_correct_seconds.append(float(ordered[stable_correct_index]["window_end_seconds"]))
        flip_counts.append(transition_count)
        final_logits.append(cumulative_logits[-1])
        targets.append(truth_index)
        parent_records.append(
            {
                "parent_sample_id": parent_id,
                "label_name_true": truth,
                "window_count": len(ordered),
                "prefix_prediction_flip_count": transition_count,
                "prefix_predictions": [RD_ONLY_LABELS[index] for index in prefix_predictions],
                "final_causal_pred": RD_ONLY_LABELS[int(prefix_predictions[-1])],
                "final_causal_correct": bool(correctness[-1]),
                "first_correct_window_end_seconds": (
                    float(ordered[first_correct_index]["window_end_seconds"])
                    if first_correct_index is not None else None
                ),
                "stable_correct_window_end_seconds": (
                    float(ordered[stable_correct_index]["window_end_seconds"])
                    if stable_correct_index is not None else None
                ),
            }
        )
    parent_logits = torch.stack(final_logits)
    parent_targets = torch.tensor(targets, dtype=torch.long)
    metrics = compute_classification_metrics(parent_logits, parent_targets, num_classes=2)
    final_correct_count = sum(record["final_causal_correct"] for record in parent_records)
    return {
        "metrics": asdict(metrics),
        "parent_count": int(len(parent_records)),
        "final_correct_count": int(final_correct_count),
        "prefix_flip_count_total": int(sum(flip_counts)),
        "prefix_flip_count_mean": float(sum(flip_counts) / len(flip_counts)) if flip_counts else 0.0,
        "first_correct_parent_count": int(len(first_correct_seconds)),
        "first_correct_window_end_seconds_mean": (
            float(sum(first_correct_seconds) / len(first_correct_seconds)) if first_correct_seconds else None
        ),
        "stable_correct_parent_count": int(len(stable_correct_seconds)),
        "stable_correct_window_end_seconds_mean": (
            float(sum(stable_correct_seconds) / len(stable_correct_seconds)) if stable_correct_seconds else None
        ),
        "records": parent_records,
    }


def _metric_payload(logits: torch.Tensor, labels: torch.Tensor, parents: list[str]) -> dict[str, object]:
    window = compute_classification_metrics(logits, labels, num_classes=2)
    per_class = [asdict(item) for item in compute_per_class_metrics(logits, labels, RD_ONLY_LABELS)]
    grouped_logits: dict[str, list[torch.Tensor]] = defaultdict(list)
    grouped_labels: dict[str, list[int]] = defaultdict(list)
    for parent, logit, label in zip(parents, logits, labels.tolist()):
        grouped_logits[str(parent)].append(logit)
        grouped_labels[str(parent)].append(int(label))
    parent_logits = torch.stack([torch.stack(values).mean(dim=0) for values in grouped_logits.values()])
    parent_targets = torch.tensor([values[0] for values in grouped_labels.values()], dtype=torch.long)
    parent = compute_classification_metrics(parent_logits, parent_targets, num_classes=2)
    return {
        "window": asdict(window),
        "per_class": per_class,
        "parent_mean_logit": asdict(parent),
        "parent_count": int(len(grouped_logits)),
        "window_count": int(logits.shape[0]),
    }


def evaluate_rd_drone_bird_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    *,
    device_name: str = "cuda",
    num_workers: int | None = None,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    values = dict(checkpoint.get("config", {}))
    if num_workers is not None:
        values["num_workers"] = int(num_workers)
    values["device"] = device_name
    values["run_test_evaluation"] = False
    config = RDDroneBirdDiagnosticConfig(**values)
    device, device_info = resolve_cuda_device(config.device)
    project_root = Path.cwd()
    index = pd.read_csv(project_root / "data" / "processed" / config.index_name, low_memory=False)
    anchor_map = pd.read_csv(project_root / "data" / "processed" / config.window_segment_map_name, low_memory=False)
    val_ids = _select_rd_only_ids(index, anchor_map, "val")
    loader = _make_loader(str(project_root / "data"), val_ids, config, shuffle=False)
    model = LowAltitudeSuspectExpert(
        input_dim=160,
        rd_input_dim=4,
        rd_patch_input_dim=3,
        rd_context_dim=0,
        use_rd_patch_cnn=config.use_rd_patch_cnn,
        use_rd_continuous_time_encoding=config.use_rd_continuous_time_encoding,
        rd_time_encoding_mode=config.rd_time_encoding_mode,
        rd_local_delta_time_clip_seconds=config.rd_local_delta_time_clip_seconds,
        rd_local_delta_time_scale=config.rd_local_delta_time_scale,
        dropout=config.dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logits_rows: list[torch.Tensor] = []
    label_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            rd_patch_features = batch.rd_patch_features.to(device, non_blocking=True)
            rd_patch_padding = batch.rd_patch_padding_mask.to(device, non_blocking=True)
            output = model.rd_drone_bird_diagnostic(
                batch.rd_features.to(device, non_blocking=True),
                rd_patch_features=rd_patch_features if config.use_rd_patch_cnn else None,
                rd_padding_mask=batch.rd_padding_mask.to(device, non_blocking=True),
                rd_patch_padding_mask=rd_patch_padding if config.use_rd_patch_cnn else None,
                rd_time_seconds=batch.rd_time_seconds.to(device, non_blocking=True),
            )
            logits = output.bird_logits.detach().cpu()
            labels = torch.tensor(
                [1 if value == "BirdTarget" else 0 for value in batch.label_names],
                dtype=torch.long,
            )
            predicted = logits.argmax(dim=-1)
            for index_in_batch, (sample_id, parent_id, source, truth, logit, pred) in enumerate(
                zip(batch.sample_ids, batch.parent_sample_ids, batch.source_names, batch.label_names, logits, predicted.tolist())
            ):
                predictions.append(
                    {
                        "sample_id": str(sample_id),
                        "parent_sample_id": str(parent_id),
                        "source_name": str(source),
                        "label_name_true": str(truth),
                        "label_name_rd_only_pred": RD_ONLY_LABELS[int(pred)],
                        "bird_logit_drone": float(logit[0]),
                        "bird_logit_bird": float(logit[1]),
                        "is_correct": bool(RD_ONLY_LABELS[int(pred)] == str(truth)),
                        "rd_time_sync_status": str(batch.rd_time_sync_status[index_in_batch]),
                        "rd_valid_length": int(batch.rd_lengths[index_in_batch].item()),
                        "window_start_seconds": float(batch.window_start_seconds[index_in_batch].item()),
                        "window_end_seconds": float(batch.window_end_seconds[index_in_batch].item()),
                    }
                )
            logits_rows.append(logits)
            label_rows.append(labels)
            parent_ids.extend(batch.parent_sample_ids)
    logits_tensor = torch.cat(logits_rows)
    labels_tensor = torch.cat(label_rows)
    payload = {
        "evaluation_scope": "validation_only_rd_drone_bird_diagnostic",
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "device": device_info,
        "config": asdict(config),
        "metrics": _metric_payload(logits_tensor, labels_tensor, parent_ids),
        "causal_parent_cumulative_logit": _causal_parent_payload(predictions),
        "predictions": predictions,
        "test_evaluation_executed": False,
        "fusion_enabled": False,
        "timing_status": "track_point_anchor_diagnostic_only",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only evaluator for diagnostic RD-only Drone/Bird checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_rd_drone_bird_checkpoint(
        Path(args.checkpoint),
        Path(args.output),
        device_name=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps({"output": args.output, "metrics": result["metrics"], "test_evaluation_executed": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
