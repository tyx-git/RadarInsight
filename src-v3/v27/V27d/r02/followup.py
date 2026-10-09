"""r02 后续：冻结模型上的分组交叉校准与同源 hard-negative 审计。

注意：这里的 OOF 是针对 calibration fit 的 parent-grouped cross-fit；模型表示
来自已经训练好的 r02 checkpoint，因此不应称为完整 end-to-end OOF。该边界会
写入报告，避免把后处理实验误读成无泄漏模型泛化证据。
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from v27.V27d.r20.data import R20Dataset
from v27.v27_baseline.metrics import summarize_parent_groups, summarize_records

from .calibration import calibration_metrics, fit_class_conditional_calibration
from .evaluation import _device, _load_calibration, _records
from .protocol import R02_CLASSES, R02Config, project_path, write_json
from .training import load_r02_checkpoint


def _fold(parent_id: str, folds: int) -> int:
    digest = hashlib.sha256(parent_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % int(folds)


def _matrix_records(records: Sequence[Mapping[str, Any]], logits: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, value in zip(records, logits, strict=True):
        shifted = value - np.max(value)
        probability = np.exp(shifted)
        probability = probability / probability.sum()
        output.append(
            {
                "parent_sample_id": row["parent_sample_id"],
                "source_suffix": row["source_suffix"],
                "label_name": row["label_name"],
                "true_id": int(row["true_id"]),
                "pred_id": int(np.argmax(value)),
                "probabilities": probability.tolist(),
                "logits": value.tolist(),
                "event_count": int(row.get("event_count", 0)),
            }
        )
    return output


def _apply_delta(records: Sequence[Mapping[str, Any]], delta_by_source: Mapping[str, float], fallback: float = 0.0) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in records:
        value = np.asarray(row["raw_logits"], dtype=np.float64).copy()
        delta = float(delta_by_source.get(str(row["source_suffix"]), fallback))
        value[1] += delta
        shifted = value - np.max(value)
        probability = np.exp(shifted)
        probability /= probability.sum()
        result.append({**dict(row), "logits": value.tolist(), "probabilities": probability.tolist(), "pred_id": int(np.argmax(value))})
    return result


def _bird_objective(records: Sequence[Mapping[str, Any]]) -> tuple[float, int, float, float]:
    bird = [row for row in records if row["label_name"] == "BirdTarget"]
    if not bird:
        return (0.0, 0, 0.0, 0.0)
    tp = sum(row["pred_id"] == 1 for row in bird)
    b2d = sum(row["pred_id"] == 0 for row in bird)
    precision_rows = [row for row in records if row["pred_id"] == 1]
    precision = tp / max(1, len(precision_rows))
    recall = tp / len(bird)
    f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
    return (f1, -b2d, precision, recall)


def _fit_source_deltas(train_records: Sequence[Mapping[str, Any]], *, bird_recall_floor: float) -> dict[str, float]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_records:
        groups[str(row["source_suffix"])].append(row)
    output: dict[str, float] = {}
    grid = np.linspace(-0.6, 0.6, 121)
    for source, rows in groups.items():
        baseline = _bird_objective(_apply_delta(rows, {source: 0.0}))
        floor = max(0.0, baseline[3] - float(bird_recall_floor))
        candidates: list[tuple[tuple[float, int, float, float, float], float]] = []
        for delta in grid:
            candidate = _apply_delta(rows, {source: float(delta)})
            objective = _bird_objective(candidate)
            if objective[3] + 1.0e-9 < floor:
                continue
            key = (objective[0], objective[1], objective[2], objective[3], -abs(float(delta)))
            candidates.append((key, float(delta)))
        output[source] = max(candidates, default=(((baseline[0], baseline[1], baseline[2], baseline[3], 0.0)), 0.0), key=lambda item: item[0])[1]
    return output


def run_followup(project_root: str | Path, config: R02Config, *, device: str | None = None, folds: int = 5) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runtime_device = _device(device)
    model, checkpoint = load_r02_checkpoint(root, config, device=str(runtime_device))
    train_set = R20Dataset(root, config.r20, split_name="train", arm="zero_control")
    validation_set = R20Dataset(root, config.r20, split_name="validation", arm="zero_control")
    train_records = _records(model, train_set, device=runtime_device, batch_size=int(config.training["batch_size"]), include_auxiliary=True)
    validation_records = _records(model, validation_set, device=runtime_device, batch_size=int(config.training["batch_size"]), include_auxiliary=True)
    train_logits = np.asarray([row["raw_logits"] for row in train_records], dtype=np.float32)
    train_labels = np.asarray([row["true_id"] for row in train_records], dtype=np.int64)
    fold_ids = np.asarray([_fold(str(row["parent_sample_id"]), int(folds)) for row in train_records], dtype=np.int64)
    oof_logits = np.zeros_like(train_logits)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(int(folds)):
        holdout = fold_ids == fold
        fit = ~holdout
        if not bool(holdout.any()) or not bool(fit.any()):
            continue
        calibration = fit_class_conditional_calibration(
            train_logits[fit], train_labels[fit],
            regularization=float(config.training["calibration_regularization"]),
            log_temperature_bound=float(config.training["calibration_log_temperature_bound"]),
            bias_bound=float(config.training["calibration_bias_bound"]),
        )
        with torch.no_grad():
            oof_logits[holdout] = calibration.transform_logits(torch.from_numpy(train_logits[holdout])).numpy()
        fold_reports.append({
            "fold": int(fold),
            "fit_parent_count": int(fit.sum()),
            "holdout_parent_count": int(holdout.sum()),
            "holdout_parent_ids": [str(row["parent_sample_id"]) for index, row in enumerate(train_records) if holdout[index]],
            "calibration": calibration.to_json(),
        })
    oof_records = _matrix_records(train_records, oof_logits)
    with torch.no_grad():
        raw_train_tensor = torch.from_numpy(train_logits)
        oof_tensor = torch.from_numpy(oof_logits)
    oof_calibration_metrics = {
        "oof_before_nll": float(torch.nn.functional.cross_entropy(raw_train_tensor, torch.from_numpy(train_labels)).item()),
        "oof_after_nll": float(torch.nn.functional.cross_entropy(oof_tensor, torch.from_numpy(train_labels)).item()),
    }
    # 真正的 OOF 指标直接在每个 holdout 经训练折变换后的 logits 上计算，
    # 不使用全量拟合器的 in-sample after 指标。
    def _ece_from_logits(values: torch.Tensor, targets: torch.Tensor, bins: int = 10) -> float:
        probability = torch.softmax(values, dim=-1)
        confidence, prediction = probability.max(dim=-1)
        correct = prediction.eq(targets)
        result = 0.0
        for index in range(bins):
            lower = float(index) / bins
            upper = float(index + 1) / bins
            mask = (confidence >= lower) & ((confidence < upper) if index + 1 < bins else (confidence <= upper))
            if bool(mask.any()):
                result += float(mask.float().mean()) * abs(float(confidence[mask].mean()) - float(correct[mask].float().mean()))
        return result
    oof_calibration_metrics["oof_before_nll"] = float(torch.nn.functional.cross_entropy(raw_train_tensor, torch.from_numpy(train_labels)).item())
    oof_calibration_metrics["oof_before_ece"] = _ece_from_logits(raw_train_tensor, torch.from_numpy(train_labels))
    oof_calibration_metrics["oof_after_ece"] = _ece_from_logits(oof_tensor, torch.from_numpy(train_labels))
    global_calibration = fit_class_conditional_calibration(
        train_logits, train_labels,
        regularization=float(config.training["calibration_regularization"]),
        log_temperature_bound=float(config.training["calibration_log_temperature_bound"]),
        bias_bound=float(config.training["calibration_bias_bound"]),
    )
    validation_logits = np.asarray([row["raw_logits"] for row in validation_records], dtype=np.float32)
    with torch.no_grad():
        global_validation_logits = global_calibration.transform_logits(torch.from_numpy(validation_logits)).numpy()
    global_validation_records = _matrix_records(validation_records, global_validation_logits)
    source_deltas = _fit_source_deltas(train_records, bird_recall_floor=0.02)
    source_records = _apply_delta(validation_records, source_deltas)
    report = {
        "schema_version": "RADAR-LN3-V27d-r02-followup-calibration-v1",
        "experiment_id": "V27d-r02",
        "status": "completed_frozen_model_crossfit_and_source_audit",
        "device": str(runtime_device),
        "model_checkpoint": {
            "path": config.artifacts["checkpoint_path"],
            "best_epoch": int(checkpoint["best_epoch"]),
        },
        "crossfit": {
            "folds": int(folds),
            "grouping": "parent_sample_id_sha256",
            "fit_scope": "r02 train parent records only",
            "oof_parent_count": int(len(oof_records)),
            "oof_metrics": summarize_records(oof_records, class_names=R02_CLASSES, scope="train:crossfit_oof_calibration"),
            "oof_calibration_metrics": calibration_metrics(train_logits, train_labels, global_calibration),
            "folds_detail": fold_reports,
            "limitation": "OOF is cross-fit for calibration parameters on frozen r02 logits; the representation checkpoint itself was not retrained per fold.",
        },
        "validation_global_calibration": {
            "metrics": summarize_records(global_validation_records, class_names=R02_CLASSES, scope="validation:global_crossfit_calibration"),
            "source_metrics": summarize_parent_groups(global_validation_records, group_field="source_suffix", class_names=R02_CLASSES, scope_prefix="validation:global_crossfit"),
            "error_counts": {
                "bird_to_drone": sum(row["label_name"] == "BirdTarget" and row["pred_id"] == 0 for row in global_validation_records),
                "drone_to_bird": sum(row["label_name"] == "DroneTarget" and row["pred_id"] == 1 for row in global_validation_records),
            },
        },
        "source_focus_calibration": {
            "fit_source_deltas_on_train_only": True,
            "bird_recall_floor_relative_to_train": 0.02,
            "delta_by_source": source_deltas,
            "validation_metrics": summarize_records(source_records, class_names=R02_CLASSES, scope="validation:source_focus_delta"),
            "source_metrics": summarize_parent_groups(source_records, group_field="source_suffix", class_names=R02_CLASSES, scope_prefix="validation:source_focus"),
            "error_counts": {
                "bird_to_drone": sum(row["label_name"] == "BirdTarget" and row["pred_id"] == 0 for row in source_records),
                "drone_to_bird": sum(row["label_name"] == "DroneTarget" and row["pred_id"] == 1 for row in source_records),
            },
            "limitation": "Source-specific delta is exploratory and may not generalize to unseen source_suffix; it is not a substitute for source-disjoint validation.",
        },
        "audit_hard_negative_ids": [row["parent_sample_id"] for row in validation_records if row["label_name"] == "BirdTarget" and row["pred_id"] == 0 and row["source_suffix"] == "20241128_191432"],
        "limitations": [
            "No validation labels were used to fit calibration or source deltas.",
            "Frozen-model cross-fit does not retrain the representation per fold; end-to-end OOF remains future work.",
            "The four designated validation audit parents remain evaluation-only.",
        ],
    }
    output = project_path(root, "src-v3/v27/V27d/r02/reports/V27d-r02-followup-calibration.json")
    write_json(output, report)
    return {"report_path": output.relative_to(root).as_posix(), **report}


__all__ = ["run_followup"]
