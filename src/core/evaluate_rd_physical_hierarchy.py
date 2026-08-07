from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.rd_physical_five_dataset import (
    RD_FIVE_LABELS,
    PhysicalRDFiveWindowDataset,
    collate_physical_rd_five_windows,
)
from .rd_physical_five_artifacts import load_physical_rd_five_checkpoint
from .rd_physical_hierarchical_composer import compose_rd_specialists, compose_with_oracle_parent
from .rd_physical_specialist_artifacts import load_physical_rd_specialist_checkpoint


def _metrics(probabilities: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    predictions = probabilities.argmax(-1)
    class_count = len(RD_FIVE_LABELS)
    matrix = [[0 for _ in range(class_count)] for _ in range(class_count)]
    for truth, prediction in zip(labels.tolist(), predictions.tolist()):
        matrix[int(truth)][int(prediction)] += 1
    precision: list[float] = []
    recall: list[float] = []
    f1: list[float] = []
    per_class: dict[str, Any] = {}
    for index, name in enumerate(RD_FIVE_LABELS):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(class_count) if row != index)
        fn = sum(matrix[index][col] for col in range(class_count) if col != index)
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2.0 * p * r / max(1.0e-12, p + r)
        support = sum(matrix[index])
        per_class[name] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "support": support,
            "support_stability": "no_support" if support == 0 else ("low" if support < 30 else "adequate"),
        }
        precision.append(p)
        recall.append(r)
        f1.append(f)
    return {
        "accuracy": float((predictions == labels).float().mean()) if labels.numel() else 0.0,
        "macro_precision": sum(precision) / len(precision),
        "macro_recall": sum(recall) / len(recall),
        "macro_f1": sum(f1) / len(f1),
        "balanced_accuracy": sum(recall) / len(recall),
        "sample_count": int(labels.numel()),
        "per_class": per_class,
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


def _parent_metrics(probabilities: torch.Tensor, labels: torch.Tensor, parent_ids: list[str]) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    grouped_labels: dict[str, int] = {}
    for index, (label, parent_id) in enumerate(zip(labels.tolist(), parent_ids)):
        key = str(parent_id)
        grouped[key].append(index)
        previous = grouped_labels.setdefault(key, int(label))
        if previous != int(label):
            raise ValueError(f"five-class validation parent has inconsistent labels: {key}")
    parent_probabilities = torch.stack([probabilities[indices].mean(0) for indices in grouped.values()])
    parent_labels = torch.tensor(list(grouped_labels.values()), dtype=torch.long)
    return {
        "parent_count": len(grouped),
        "aggregation": "arithmetic mean of per-window probabilities, then argmax",
        "metrics": _metrics(parent_probabilities, parent_labels),
    }


def _calibration(probabilities: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> dict[str, Any]:
    confidence, predictions = probabilities.max(-1)
    correct = predictions.eq(labels).float()
    ece = 0.0
    calibration_bins: list[dict[str, Any]] = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        selected = (confidence >= low) & ((confidence < high) if index + 1 < bins else (confidence <= high))
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean())
            mean_confidence = float(confidence[selected].mean())
            ece += count / max(1, labels.numel()) * abs(accuracy - mean_confidence)
        else:
            accuracy = None
            mean_confidence = None
        calibration_bins.append({"lower": low, "upper": high, "count": count, "accuracy": accuracy, "confidence": mean_confidence})
    brier = {
        name: float((probabilities[:, index] - labels.eq(index).float()).pow(2).mean())
        for index, name in enumerate(RD_FIVE_LABELS)
    }
    quantiles = torch.quantile(confidence, torch.tensor([0.1, 0.5, 0.9])) if confidence.numel() else torch.zeros(3)
    return {
        "ece": ece,
        "brier": brier,
        "confidence_mean": float(confidence.mean()) if confidence.numel() else 0.0,
        "confidence_p10": float(quantiles[0]),
        "confidence_median": float(quantiles[1]),
        "confidence_p90": float(quantiles[2]),
        "bins": calibration_bins,
        "posthoc_calibration_applied": False,
        "threshold_selection_performed": False,
        "scope": "validation diagnostic only; uncalibrated checkpoint probabilities",
    }


def _forward(model: torch.nn.Module, batch: Any, device: torch.device) -> Any:
    return model(
        batch.features.to(device),
        batch.time_seconds.to(device),
        batch.padding_mask.to(device),
        batch.quality_features.to(device),
        batch.range_offset_m.to(device),
        batch.velocity_mps.to(device),
        batch.raw_amplitude_db.to(device),
    )


def evaluate_rd_physical_hierarchy(
    *,
    flat_checkpoint: Path,
    parent_checkpoint: Path,
    dbb_checkpoint: Path,
    cu_checkpoint: Path,
    output: Path,
    device_name: str = "cuda",
    max_batches: int = 0,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or not str(device_name).startswith("cuda"):
        raise RuntimeError("RD hierarchy evaluator requires CUDA")
    device = torch.device(device_name)
    project_root = Path.cwd()
    flat_loaded = load_physical_rd_five_checkpoint(flat_checkpoint, project_root, device)
    parent_model, parent_config, parent_artifacts, parent_payload = load_physical_rd_specialist_checkpoint(parent_checkpoint, project_root, device)
    dbb_model, dbb_config, dbb_artifacts, dbb_payload = load_physical_rd_specialist_checkpoint(dbb_checkpoint, project_root, device)
    cu_model, cu_config, cu_artifacts, cu_payload = load_physical_rd_specialist_checkpoint(cu_checkpoint, project_root, device)
    if flat_loaded.config.index_name != parent_config.index_name or flat_loaded.config.index_name != dbb_config.index_name or flat_loaded.config.index_name != cu_config.index_name:
        raise ValueError("hierarchy checkpoints do not share one validation index")
    if {parent_config.kind, dbb_config.kind, cu_config.kind} != {"parent_gate", "dbb", "cu"}:
        raise ValueError("hierarchy requires exactly parent_gate, dbb and cu checkpoints")
    dataset = PhysicalRDFiveWindowDataset(
        project_root / "data",
        index_name=flat_loaded.config.index_name,
        split_name="val",
        trainable_only=True,
        normalization_stats_name=flat_loaded.config.normalization_stats_name,
        include_missing_rd=False,
        present_only=False,
    )
    loader = DataLoader(dataset, batch_size=flat_loaded.config.batch_size, shuffle=False, num_workers=0, collate_fn=collate_physical_rd_five_windows)
    mode_probabilities: dict[str, list[torch.Tensor]] = defaultdict(list)
    labels_rows: list[torch.Tensor] = []
    parent_ids: list[str] = []
    sample_ids: list[str] = []
    true_names: list[str] = []
    for batch_index, batch in enumerate(loader, 1):
        if max_batches > 0 and batch_index > max_batches:
            break
        with torch.no_grad():
            flat_output = _forward(flat_loaded.model, batch, device)
            parent_output = _forward(parent_model, batch, device)
            dbb_output = _forward(dbb_model, batch, device)
            cu_output = _forward(cu_model, batch, device)
            predicted_soft = compose_rd_specialists(parent_output.logits, dbb_output.logits, cu_output.logits, flat_logits=flat_output.flat_logits, mode="predicted_soft")
            hard = compose_rd_specialists(parent_output.logits, dbb_output.logits, cu_output.logits, mode="hard_routing")
            parent_labels = (batch.labels.to(device) >= 3).long()
            oracle = compose_with_oracle_parent(parent_labels, dbb_output.logits, cu_output.logits)
            mode_probabilities["flat"].append(torch.softmax(flat_output.flat_logits, -1).cpu())
            mode_probabilities["cascade"].append(flat_output.cascade_probabilities.cpu())
            mode_probabilities["flat_fallback"].append(torch.softmax(flat_output.flat_logits, -1).cpu())
            mode_probabilities["predicted_soft"].append(predicted_soft.probabilities.cpu())
            mode_probabilities["hard_routing"].append(hard.probabilities.cpu())
            mode_probabilities["oracle_parent"].append(oracle.probabilities.cpu())
        labels_rows.append(batch.labels.cpu())
        parent_ids.extend(str(value) for value in batch.parent_sample_ids)
        sample_ids.extend(str(value) for value in batch.sample_ids)
        true_names.extend(str(value) for value in batch.label_names)
    if not labels_rows:
        raise ValueError("hierarchy evaluator selected no validation windows")
    labels = torch.cat(labels_rows)
    full_count = len(dataset)
    evaluated_count = len(sample_ids)
    full = max_batches == 0 and evaluated_count == full_count
    payload: dict[str, Any] = {
        "scope": "same-window RD-only hierarchy validation; no TR input, fusion, reject tuning, or test evaluation" if full else "restricted same-window hierarchy validation; incomplete split, no performance claim",
        "split_name": "val",
        "labels": list(RD_FIVE_LABELS),
        "checkpoints": {
            "flat": flat_checkpoint.as_posix(),
            "parent_gate": parent_checkpoint.as_posix(),
            "dbb": dbb_checkpoint.as_posix(),
            "cu": cu_checkpoint.as_posix(),
        },
        "checkpoint_epochs": {
            "flat": int(flat_loaded.checkpoint["epoch"]),
            "parent_gate": int(parent_payload["epoch"]),
            "dbb": int(dbb_payload["epoch"]),
            "cu": int(cu_payload["epoch"]),
        },
        "evaluation_completion": {
            "full_split_evaluated": full,
            "dataset_window_count": full_count,
            "evaluated_window_count": evaluated_count,
            "excluded_by_batch_limit_count": full_count - evaluated_count,
            "max_batches": max_batches,
        },
        "modes": {},
        "source": {"available": True, "value": "cq08_track"},
        "session": {"available": False, "reason": "physical index has no session field"},
        "online_stable_decision": {"available": False, "reason": "no production stable-decision state machine connected"},
        "rejection_capability": {"supports_rejection": False, "unknown_target_is_class": True, "abstain_output": False, "threshold": None},
        "test_evaluation_executed": False,
        "tr_inputs_loaded": False,
    }
    prediction_rows: list[dict[str, Any]] = []
    for index, (sample_id, parent_id, true_name, label) in enumerate(zip(sample_ids, parent_ids, true_names, labels.tolist())):
        row: dict[str, Any] = {"sample_id": sample_id, "parent_sample_id": parent_id, "label_name_true": true_name}
        for mode, values in mode_probabilities.items():
            probability = torch.cat(values, 0)[index]
            row[f"label_name_{mode}_pred"] = RD_FIVE_LABELS[int(probability.argmax())]
            row[f"{mode}_confidence"] = float(probability.max())
            row[f"{mode}_probabilities"] = [float(value) for value in probability]
        prediction_rows.append(row)
    for mode, values in mode_probabilities.items():
        probabilities = torch.cat(values, 0)
        mode_payload: dict[str, Any] = {
            "window": _metrics(probabilities, labels),
            "parent": _parent_metrics(probabilities, labels, parent_ids),
            "source": {"cq08_track": _metrics(probabilities, labels)},
            "calibration": _calibration(probabilities, labels),
            "diagnostic_only": mode in {"oracle_parent", "hard_routing"},
        }
        payload["modes"][mode] = mode_payload
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        csv_row = {key: value for key, value in row.items() if not key.endswith("_probabilities")}
        for mode in mode_probabilities:
            for class_name, probability in zip(RD_FIVE_LABELS, row[f"{mode}_probabilities"]):
                csv_row[f"{mode}_prob_{class_name}"] = probability
        csv_rows.append(csv_row)
    csv_path = output.with_suffix(".predictions.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    payload["prediction_csv"] = csv_path.as_posix()
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate flat and RD specialist hierarchy on one validation window set.")
    parser.add_argument("--flat-checkpoint", required=True)
    parser.add_argument("--parent-checkpoint", required=True)
    parser.add_argument("--dbb-checkpoint", required=True)
    parser.add_argument("--cu-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()
    result = evaluate_rd_physical_hierarchy(
        flat_checkpoint=Path(args.flat_checkpoint),
        parent_checkpoint=Path(args.parent_checkpoint),
        dbb_checkpoint=Path(args.dbb_checkpoint),
        cu_checkpoint=Path(args.cu_checkpoint),
        output=Path(args.output),
        device_name=args.device,
        max_batches=args.max_batches,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "modes"}, ensure_ascii=False, indent=2))
    for mode, values in result["modes"].items():
        print(json.dumps({"mode": mode, "window": values["window"], "parent": values["parent"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
