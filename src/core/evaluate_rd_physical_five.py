from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES, PhysicalRDFiveWindowDataset, collate_physical_rd_five_windows
from .rd_physical_five_artifacts import load_physical_rd_five_checkpoint
from .rd_physical_parent_aggregation import mean_parent_probabilities


def _metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, object]:
    predictions = logits.argmax(-1)
    matrix = [[0 for _ in RD_FIVE_LABELS] for _ in RD_FIVE_LABELS]
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(truth)][int(prediction)] += 1
    values: dict[str, dict[str, float | int]] = {}
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    for index, label in enumerate(RD_FIVE_LABELS):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(RD_FIVE_LABELS)) if row != index)
        fn = sum(matrix[index][col] for col in range(len(RD_FIVE_LABELS)) if col != index)
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        score = 2.0 * p * r / max(1.0e-12, p + r)
        support = sum(matrix[index])
        values[label] = {
            "precision": p,
            "recall": r,
            "f1": score,
            "support": support,
            "support_stability": "no_support" if support == 0 else ("low" if support < 30 else "adequate"),
        }
        precision.append(p); recall.append(r); f1.append(score)
    return {
        "accuracy": float((predictions == labels).float().mean()) if labels.numel() else 0.0,
        "macro_precision": float(sum(precision) / len(precision)),
        "macro_recall": float(sum(recall) / len(recall)),
        "macro_f1": float(sum(f1) / len(f1)),
        "balanced_accuracy": float(sum(recall) / len(recall)),
        "sample_count": int(labels.numel()),
        "per_class": values,
        "confusion_matrix": {"labels": list(RD_FIVE_LABELS), "matrix": matrix},
        "directional_errors": {
            "DroneTarget_to_BirdTarget": matrix[0][1],
            "BirdTarget_to_DroneTarget": matrix[1][0],
            "BirdTarget_to_BalloonTarget": matrix[1][2],
            "BirdTarget_to_UnknownTarget": matrix[1][4],
            "ClutterTarget_to_BirdTarget": matrix[3][1],
            "ClutterTarget_to_UnknownTarget": matrix[3][4],
            "UnknownTarget_to_BirdTarget": matrix[4][1],
            "UnknownTarget_to_ClutterTarget": matrix[4][3],
        },
    }


def _parent_metrics(logits: torch.Tensor, labels: torch.Tensor, parent_ids: list[str]) -> dict[str, object]:
    parent_probabilities, parent_labels, parent_count = mean_parent_probabilities(logits, labels, parent_ids)
    return {"parent_count": parent_count, "aggregation": "arithmetic mean of per-window probabilities, then argmax", "metrics": _metrics(parent_probabilities, parent_labels)}


def evaluate_physical_rd_five_checkpoint(checkpoint_path: Path, output_path: Path, *, split_name: str = "val", device_name: str = "cuda", max_batches: int = 0) -> dict[str, object]:
    if split_name != "val":
        raise ValueError("five-class RD evaluator only permits split_name='val'; test evaluation is disabled")
    if not torch.cuda.is_available() or not str(device_name).startswith("cuda"):
        raise RuntimeError("five-class RD evaluator requires CUDA")
    project_root = Path.cwd()
    device = torch.device(device_name)
    loaded = load_physical_rd_five_checkpoint(checkpoint_path, project_root, device)
    config = loaded.config
    dataset = PhysicalRDFiveWindowDataset(project_root / "data", index_name=config.index_name, split_name=split_name, normalization_stats_name=config.normalization_stats_name)
    full_count = len(dataset)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_physical_rd_five_windows)
    flat_rows: list[torch.Tensor] = []
    cascade_rows: list[torch.Tensor] = []
    labels_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    predictions: list[dict[str, object]] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, 1):
            if max_batches > 0 and batch_index > max_batches:
                break
            output = loaded.model(batch.features.to(device), batch.time_seconds.to(device), batch.padding_mask.to(device), batch.quality_features.to(device), batch.range_offset_m.to(device), batch.velocity_mps.to(device), batch.raw_amplitude_db.to(device))
            flat = output.flat_logits.detach().cpu()
            cascade = output.cascade_logits.detach().cpu()
            labels = batch.labels.detach().cpu()
            flat_rows.append(flat); cascade_rows.append(cascade); labels_rows.append(labels)
            parent_ids.extend(batch.parent_sample_ids)
            flat_pred = flat.argmax(-1); cascade_pred = cascade.argmax(-1)
            for index, sample_id in enumerate(batch.sample_ids):
                predictions.append({
                    "sample_id": str(sample_id), "parent_sample_id": str(batch.parent_sample_ids[index]), "label_name_true": str(batch.label_names[index]),
                    "label_name_flat_pred": RD_FIVE_LABELS[int(flat_pred[index])], "label_name_cascade_pred": RD_FIVE_LABELS[int(cascade_pred[index])],
                    "flat_probabilities": [float(x) for x in torch.softmax(flat[index], -1)], "cascade_probabilities": [float(x) for x in torch.softmax(cascade[index], -1)],
                })
    flat_logits = torch.cat(flat_rows); cascade_logits = torch.cat(cascade_rows); labels = torch.cat(labels_rows)
    full = max_batches == 0 and len(predictions) == full_count
    payload = {
        "scope": "physical RD-only five-class validation; no TR input, fusion, reject tuning, or test evaluation" if full else "restricted five-class RD smoke validation; incomplete split, no performance claim",
        "checkpoint": checkpoint_path.as_posix(), "checkpoint_epoch": int(loaded.checkpoint["epoch"]), "artifact_manifest": loaded.artifacts.manifest,
        "split_name": split_name, "labels": list(RD_FIVE_LABELS), "deployment_head": config.deployment_head,
        "selected_metrics": _metrics(flat_logits if config.deployment_head == "flat" else cascade_logits, labels),
        "flat_metrics": _metrics(flat_logits, labels), "cascade_metrics": _metrics(cascade_logits, labels),
        "flat_parent_metrics": _parent_metrics(flat_logits, labels, parent_ids), "cascade_parent_metrics": _parent_metrics(cascade_logits, labels, parent_ids),
        "evaluation_completion": {"full_split_evaluated": full, "dataset_window_count": full_count, "evaluated_window_count": len(predictions), "excluded_by_batch_limit_count": full_count - len(predictions), "max_batches": max_batches},
        "source": {"available": True, "value": "cq08_track"},
        "source_metrics": {"available": True, "values": {"cq08_track": _metrics(flat_logits if config.deployment_head == "flat" else cascade_logits, labels)}},
        "session": {"available": False, "reason": "physical index has no session field"},
        "online_stable_decision": {"available": False, "reason": "no production stable-decision state machine connected"},
        "rejection_capability": {"supports_rejection": False, "unknown_target_is_class": True, "abstain_output": False, "threshold": None},
        "test_evaluation_executed": False, "fusion_enabled": False, "predictions": predictions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_csv(output_path.with_suffix(".predictions.csv"), index=False)
    payload["prediction_csv"] = output_path.with_suffix(".predictions.csv").as_posix()
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate five-class physical RD-only checkpoint on validation only.")
    parser.add_argument("--checkpoint", required=True); parser.add_argument("--output", required=True); parser.add_argument("--split", default="val"); parser.add_argument("--device", default="cuda"); parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_physical_rd_five_checkpoint(Path(args.checkpoint), Path(args.output), split_name=args.split, device_name=args.device, max_batches=args.max_batches)
    print(json.dumps({key: value for key, value in result.items() if key != "predictions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
