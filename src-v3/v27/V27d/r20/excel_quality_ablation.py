"""Excel TR-only 数据质量门控离线对照。

目的：不是把规则包装成已验证的降噪模型，而是回答一个可复核问题：
训练集拟合的结构质量分数，是否能在 r20 validation 中减少 Bird->Drone，
同时保持 Drone/Bird 的 Recall。所有规则只产生软权重，不覆盖原始数据。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from v27.V27d.r20.data import R20Dataset, build_r20_loader, r20_model_input_kwargs
from v27.V27d.r20.protocol import R20_ARMS, load_config
from v27.V27d.r20.training import load_checkpoint

CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
QUALITY_FIELDS = (
    "range_m",
    "radial_speed_mps",
    "speed_mps",
    "height_m",
    "elevation_deg",
    "azimuth_deg",
    "course_deg",
    "amplitude_db",
)


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _xyz(row: Mapping[str, Any]) -> np.ndarray | None:
    range_m = _finite(row.get("range_m"))
    azimuth = _finite(row.get("azimuth_deg"))
    elevation = _finite(row.get("elevation_deg"))
    if range_m is None or azimuth is None or elevation is None or range_m < 0.0:
        return None
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    horizontal = range_m * np.cos(elevation_rad)
    return np.asarray(
        [
            horizontal * np.cos(azimuth_rad),
            horizontal * np.sin(azimuth_rad),
            range_m * np.sin(elevation_rad),
        ],
        dtype=np.float64,
    )


def _fit_stats(dataset: R20Dataset) -> dict[str, Any]:
    values: dict[str, list[float]] = {name: [] for name in QUALITY_FIELDS}
    for item in dataset:
        for row in item.sequence.event_rows:
            for name in QUALITY_FIELDS:
                number = _finite(row.get(name))
                if number is not None:
                    values[name].append(number)
    result: dict[str, Any] = {}
    for name, raw in values.items():
        array = np.asarray(raw, dtype=np.float64)
        result[name] = {
            "count": int(len(array)),
            "q001": float(np.quantile(array, 0.001)) if len(array) else None,
            "q999": float(np.quantile(array, 0.999)) if len(array) else None,
            "median": float(np.median(array)) if len(array) else None,
            "mad_scale": float(max(1.0e-6, 1.4826 * np.median(np.abs(array - np.median(array))))) if len(array) else 1.0,
        }
    return result


def _sequence_quality(sequence: Any, stats: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(sequence.event_rows)
    if not rows:
        return {"scores": np.zeros(0), "flags": {}, "missing_count": 0}
    scores = np.zeros(len(rows), dtype=np.float64)
    missing = np.zeros(len(rows), dtype=np.float64)
    range_flags = np.zeros(len(rows), dtype=np.float64)
    jump_flags = np.zeros(len(rows), dtype=np.float64)
    field_flags = {name: np.zeros(len(rows), dtype=np.float64) for name in QUALITY_FIELDS}
    coordinates = [_xyz(row) for row in rows]
    for index, row in enumerate(rows):
        for name in QUALITY_FIELDS:
            number = _finite(row.get(name))
            if number is None:
                missing[index] += 1.0
                continue
            bounds = stats[name]
            low, high = bounds["q001"], bounds["q999"]
            scale = max(float(bounds["mad_scale"]), 1.0e-6)
            if low is not None and number < low:
                field_flags[name][index] = min(3.0, (low - number) / scale)
            elif high is not None and number > high:
                field_flags[name][index] = min(3.0, (number - high) / scale)
        if index > 0 and coordinates[index] is not None and coordinates[index - 1] is not None:
            dt = max(0.0, _finite(row.get("dt_seconds")) or 0.0)
            if dt > 0.0:
                speed = max(0.0, _finite(row.get("speed_mps")) or 0.0)
                gap = float(np.linalg.norm(coordinates[index] - coordinates[index - 1]))
                expected = max(8.0, 2.0 * speed * dt)
                ratio = gap / expected
                if ratio > 3.0:
                    jump_flags[index] = min(3.0, ratio - 3.0)
    sum_field = sum(field_flags.values(), start=np.zeros(len(rows), dtype=np.float64))
    # 规则只表示结构异常强度，不表示类别概率。
    scores = np.clip(
        0.20 * np.clip(missing / 2.0, 0.0, 1.0)
        + 0.35 * np.clip(sum_field / 3.0, 0.0, 1.0)
        + 0.45 * np.clip(jump_flags / 3.0, 0.0, 1.0),
        0.0,
        1.0,
    )
    return {
        "scores": scores,
        "flags": {
            "missing": missing,
            "field": field_flags,
            "jump": jump_flags,
        },
        "missing_count": int(np.sum(missing > 0.0)),
    }


def _metrics(true_ids: list[int], pred_ids: list[int]) -> dict[str, Any]:
    true = np.asarray(true_ids, dtype=np.int64)
    pred = np.asarray(pred_ids, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, selected in zip(true, pred, strict=True):
        confusion[int(actual), int(selected)] += 1
    for index, name in enumerate(CLASS_NAMES):
        actual = true == index
        selected = pred == index
        tp = int(np.sum(actual & selected))
        fp = int(np.sum(~actual & selected))
        fn = int(np.sum(actual & ~selected))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1.0e-12, precision + recall)
        rows.append({"class": name, "support": int(np.sum(actual)), "tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1})
    return {
        "support": int(true.size),
        "accuracy": float(np.mean(true == pred)) if true.size else 0.0,
        "macro_precision": float(np.mean([row["precision"] for row in rows])),
        "macro_recall": float(np.mean([row["recall"] for row in rows])),
        "macro_f1": float(np.mean([row["f1"] for row in rows])),
        "per_class": rows,
        "confusion_matrix_rows_true_columns_pred": confusion.tolist(),
        "bird_to_drone": int(confusion[1, 0]),
        "drone_to_bird": int(confusion[0, 1]),
    }


def _apply_quality_weights(inputs: dict[str, torch.Tensor], weights: torch.Tensor) -> dict[str, torch.Tensor]:
    result = dict(inputs)
    base = inputs["tr_features"]
    gate = weights.to(device=base.device, dtype=base.dtype)
    for key in ("tr_features", "rd_frames", "rd_quality", "rd_physical_features", "micro_doppler_features", "radial_sidecar_values"):
        value = result.get(key)
        if not isinstance(value, torch.Tensor) or value.ndim < 2 or value.shape[1] != gate.shape[1]:
            continue
        expanded = gate
        while expanded.ndim < value.ndim:
            expanded = expanded.unsqueeze(-1)
        result[key] = value * expanded
    return result


def evaluate_arm(root: Path, config: Any, arm: str, stats: Mapping[str, Any], strengths: tuple[float, ...]) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(root, config, arm=arm, seed=42)
    model.eval()
    dataset = R20Dataset(root, config, split_name="validation", arm=arm)
    quality = {item.sequence.parent_sample_id: _sequence_quality(item.sequence, stats) for item in dataset}
    outputs: dict[float, tuple[list[int], list[int]]] = {strength: ([], []) for strength in strengths}
    with torch.inference_mode():
        for batch in build_r20_loader(dataset, batch_size=int(config.training["batch_size"]), shuffle=False):
            base_inputs = r20_model_input_kwargs(batch, torch.device("cpu"))
            width = int(batch["tr_features"].shape[1])
            score_matrix = torch.zeros((len(batch["parent_sample_ids"]), width), dtype=torch.float32)
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                score = quality[str(parent_id)]["scores"]
                length = min(len(score), width)
                score_matrix[row, :length] = torch.from_numpy(score[:length].astype(np.float32))
            true = batch["l2_labels"].tolist()
            for strength in strengths:
                weights = torch.ones_like(score_matrix)
                active = score_matrix > 0.0
                weights[active] = torch.exp(-float(strength) * score_matrix[active])
                output = model(**_apply_quality_weights(base_inputs, weights))
                preds = output["window_logits"].argmax(dim=-1).tolist()
                outputs[strength][0].extend(int(value) for value in true)
                outputs[strength][1].extend(int(value) for value in preds)
    return {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "quality_score": {
            "event_count": int(sum(len(value["scores"]) for value in quality.values())),
            "median": float(np.percentile(np.concatenate([value["scores"] for value in quality.values()]), 50)),
            "mean": float(np.mean(np.concatenate([value["scores"] for value in quality.values()]))),
            "p95": float(np.percentile(np.concatenate([value["scores"] for value in quality.values()]), 95)),
            "p99": float(np.percentile(np.concatenate([value["scores"] for value in quality.values()]), 99)),
            "max": float(np.max(np.concatenate([value["scores"] for value in quality.values()]))),
            "nonzero_fraction": float(np.mean(np.concatenate([value["scores"] for value in quality.values()]) > 0.0)),
        },
        "policies": {str(strength): _metrics(*outputs[strength]) for strength in strengths},
    }


def run(project_root: str | Path, *, strengths: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0)) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(root)
    train = R20Dataset(root, config, split_name="train", arm="zero_control")
    stats = _fit_stats(train)
    return {
        "schema_version": "RADAR-LN3-V27d-r20-excel-quality-ablation-v1",
        "experiment_id": "V27d-r20",
        "data_scope": {
            "quality_stats_fit_split": "train",
            "evaluation_split": "validation",
            "source_grouping": "parent/file; no event random split",
            "raw_source_modified": False,
            "noise_field_used": False,
        },
        "quality_stats": stats,
        "arms": {arm: evaluate_arm(root, config, arm, stats, strengths) for arm in R20_ARMS},
        "interpretation": [
            "结构质量分数不是 clutter probability；没有标签时只能作为 exploratory soft gate。",
            "若所有 policy 的 Bird->Drone 不变，说明当前 Excel 质量规则没有命中主要错误机制。",
            "若强 gate 改善 Bird->Drone 但损失 Recall，必须拒绝接入默认管线并改用监督式 reliability head。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Excel TR-only 质量门控离线对照")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("tmp/x5a_excel_quality_ablation.json"))
    args = parser.parse_args()
    result = run(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": result["arms"], "data_scope": result["data_scope"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
