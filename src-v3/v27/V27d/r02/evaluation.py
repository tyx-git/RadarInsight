"""V27d-r02 parent-level与 source_suffix 分层评估。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from v27.V27d.r20.data import R20Dataset, build_r20_loader, r20_model_input_kwargs
from v27.V27d.r20.training import load_checkpoint as load_r20_checkpoint
from v27.v27_baseline.metrics import summarize_parent_groups, summarize_records

from .calibration import ClassConditionalCalibration, calibration_metrics
from .model import R02BirdDroneModel
from .protocol import (
    R02_CLASSES,
    R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS,
    R02Config,
    load_json,
    project_path,
    sha256_file,
    write_json,
)
from .training import load_r02_checkpoint


def _device(requested: str | None) -> torch.device:
    value = str(requested or "cuda").strip().lower()
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" or value.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("r02 requested CUDA but CUDA is unavailable")
        return torch.device(value if value.startswith("cuda:") else "cuda:0")
    if value == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported r02 device: {value}")


def _load_calibration(root: Path, config: R02Config) -> ClassConditionalCalibration:
    payload = load_json(project_path(root, config.artifacts["calibration_path"], must_exist=True))
    names = tuple(str(value) for value in payload["class_names"])
    return ClassConditionalCalibration(
        log_temperature=tuple(float(value) for value in payload["log_temperature"]),
        bias=tuple(float(value) for value in payload["bias"]),
        class_names=names,
        fit_scope=str(payload.get("fit_scope", "train_only")),
        optimizer=str(payload.get("optimizer", "LBFGS")),
        steps=int(payload.get("steps", 0)),
    )


def _records(
    model: torch.nn.Module,
    dataset: R20Dataset,
    *,
    device: torch.device,
    batch_size: int,
    calibration: ClassConditionalCalibration | None = None,
    include_auxiliary: bool = False,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for batch in build_r20_loader(dataset, batch_size=batch_size, shuffle=False):
            output = model(**r20_model_input_kwargs(batch, device))
            logits = output["window_logits"]
            if calibration is not None:
                calibrated_logits = calibration.transform_logits(logits)
                probabilities = torch.softmax(calibrated_logits, dim=-1)
                prediction_logits = calibrated_logits
            else:
                calibrated_logits = None
                probabilities = torch.softmax(logits, dim=-1)
                prediction_logits = logits
            for index, parent_id in enumerate(batch["parent_sample_ids"]):
                probability = probabilities[index].detach().cpu()
                raw = logits[index].detach().cpu()
                row: dict[str, Any] = {
                    "parent_sample_id": str(parent_id),
                    "source_suffix": str(parent_id).split("|")[2],
                    "label_name": str(batch["label_names"][index]),
                    "true_id": int(batch["l2_labels"][index]),
                    "pred_id": int(prediction_logits[index].argmax().item()),
                    "probabilities": [float(value) for value in probability.tolist()],
                    "raw_logits": [float(value) for value in raw.tolist()],
                    "event_count": int(batch["lengths"][index]),
                    "raw_event_count": int(batch["raw_event_counts"][index]),
                    "model_truncation_applied": bool(batch["raw_event_counts"][index] > batch["lengths"][index]),
                }
                if calibrated_logits is not None:
                    row["calibrated_logits"] = [float(value) for value in calibrated_logits[index].detach().cpu().tolist()]
                if include_auxiliary and "bird_reliability" in output:
                    length = int(batch["lengths"][index])
                    row["bird_reliability_final"] = float(output["bird_reliability"][index, length - 1].item())
                    row["maturity_final"] = float(output["maturity"][index, length - 1].item())
                rows.append(row)
    if [row["parent_sample_id"] for row in rows] != dataset.parent_ids:
        raise ValueError("r02 evaluation changed parent order")
    return rows


def _source_metrics(records: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[str(row["source_suffix"])].append(row)
    return {
        source: {
            "support": len(rows),
            "metrics": summarize_records(rows, class_names=R02_CLASSES, scope=f"{scope}:source={source}"),
        }
        for source, rows in sorted(groups.items())
    }


def _error_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "bird_to_drone": sum(row["label_name"] == "BirdTarget" and row["pred_id"] == 0 for row in records),
        "drone_to_bird": sum(row["label_name"] == "DroneTarget" and row["pred_id"] == 1 for row in records),
        "bird_support": sum(row["label_name"] == "BirdTarget" for row in records),
        "drone_support": sum(row["label_name"] == "DroneTarget" for row in records),
    }


def _paired(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    left = {row["parent_sample_id"]: row for row in before}
    right = {row["parent_sample_id"]: row for row in after}
    if set(left) != set(right):
        raise ValueError("r02 paired records have different parent populations")
    transitions: dict[str, int] = defaultdict(int)
    changed: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        before_row, after_row = left[parent_id], right[parent_id]
        if before_row["true_id"] != after_row["true_id"]:
            raise ValueError("r02 paired records have different labels")
        before_correct = before_row["true_id"] == before_row["pred_id"]
        after_correct = after_row["true_id"] == after_row["pred_id"]
        key = f"{'correct' if before_correct else 'wrong'}_to_{'correct' if after_correct else 'wrong'}"
        transitions[key] += 1
        if before_row["pred_id"] != after_row["pred_id"]:
            changed.append(
                {
                    "parent_sample_id": parent_id,
                    "source_suffix": before_row["source_suffix"],
                    "label_name": before_row["label_name"],
                    "baseline_prediction": before_row["pred_id"],
                    "after_prediction": after_row["pred_id"],
                    "baseline_correct": before_correct,
                    "after_correct": after_correct,
                }
            )
    return {"correctness_transitions": dict(sorted(transitions.items())), "changed_predictions": changed}


def evaluate(project_root: str | Path, config: R02Config, *, device: str | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runtime_device = _device(device)
    model, checkpoint = load_r02_checkpoint(root, config, device=str(runtime_device))
    r20_config = config.r20
    dataset = R20Dataset(root, r20_config, split_name="validation", arm="zero_control")
    baseline_model, baseline_checkpoint = load_r20_checkpoint(
        root,
        r20_config,
        arm="zero_control",
        seed=42,
        checkpoint_path=project_path(root, config.source["r20_checkpoint_path"], must_exist=True),
    )
    calibration = _load_calibration(root, config)
    baseline_records = _records(
        baseline_model.to(runtime_device), dataset, device=runtime_device, batch_size=int(config.training["batch_size"])
    )
    raw_records = _records(
        model, dataset, device=runtime_device, batch_size=int(config.training["batch_size"]), include_auxiliary=True
    )
    calibrated_records = _records(
        model,
        dataset,
        device=runtime_device,
        batch_size=int(config.training["batch_size"]),
        calibration=calibration,
        include_auxiliary=True,
    )
    raw_logits = torch.tensor([row["raw_logits"] for row in raw_records], dtype=torch.float32)
    labels = torch.tensor([row["true_id"] for row in raw_records], dtype=torch.long)
    report = {
        "schema_version": "RADAR-LN3-V27d-r02-evaluation-v1",
        "experiment_id": "V27d-r02",
        "status": "completed_validation_audit",
        "device": str(runtime_device),
        "scope": {
            "dataset": config.source["dataset"],
            "split": "r20 validation parent_final",
            "support": len(dataset),
            "validation_used_for_training": False,
            "validation_used_for_calibration": False,
            "complete_parent_sequence": True,
            "event_random_split": False,
            "base_checkpoint_epoch": int(baseline_checkpoint["epoch"]),
            "r02_checkpoint_epoch": int(checkpoint["best_epoch"]),
        },
        "baseline_r20": {
            "metrics": summarize_records(baseline_records, class_names=R02_CLASSES, scope="validation:r20:baseline"),
            "source_metrics": _source_metrics(baseline_records, "validation:r20:baseline"),
            "error_counts": _error_counts(baseline_records),
        },
        "r02_raw": {
            "metrics": summarize_records(raw_records, class_names=R02_CLASSES, scope="validation:r02:raw"),
            "source_metrics": _source_metrics(raw_records, "validation:r02:raw"),
            "error_counts": _error_counts(raw_records),
        },
        "r02_calibrated": {
            "metrics": summarize_records(calibrated_records, class_names=R02_CLASSES, scope="validation:r02:calibrated"),
            "source_metrics": _source_metrics(calibrated_records, "validation:r02:calibrated"),
            "error_counts": _error_counts(calibrated_records),
        },
        "paired": {
            "baseline_to_raw": _paired(baseline_records, raw_records),
            "baseline_to_calibrated": _paired(baseline_records, calibrated_records),
        },
        "calibration_audit": {
            **calibration.to_json(),
            "validation_metrics": calibration_metrics(raw_logits, labels, calibration),
        },
        "audit_hard_negative_results": [
            {
                "parent_sample_id": parent_id,
                "baseline": next(row for row in baseline_records if row["parent_sample_id"] == parent_id),
                "r02_raw": next(row for row in raw_records if row["parent_sample_id"] == parent_id),
                "r02_calibrated": next(row for row in calibrated_records if row["parent_sample_id"] == parent_id),
            }
            for parent_id in R02_VALIDATION_AUDIT_HARD_NEGATIVE_IDS
        ],
        "limitations": [
            "四个指定 validation hard-negative 仅用于最终审计，未参与训练或校准。",
            "校准参数由 train parent logits 拟合，未使用严格 OOF；结果属于 train-only exploratory calibration。",
            "source_suffix 分层来自同一 r20 split 的源内分层，不代表 unseen-source 泛化。",
            "validation 仅 84 个 parent，Bird→Drone 计数的统计不确定性较高。",
        ],
    }
    path = project_path(root, config.artifacts["report_path"])
    write_json(path, report)
    return {"report_path": path.relative_to(root).as_posix(), **report}


__all__ = ["evaluate"]
