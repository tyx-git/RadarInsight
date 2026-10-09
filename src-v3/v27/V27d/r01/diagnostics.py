"""r01 只读诊断：确认 X5A 降噪不会自动改变 Excel 分类。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .denoising import DenoisingConfig, fit_denoiser, score_sequence
from .protocol import artifact_path, load_config, validate_r20_reference
from .x5a import X5ALayout, diagnose_layout, summarize_x5a_file


def _weights_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p01": 0.0, "p50": 0.0, "p99": 0.0, "removed_fraction": 0.0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "p01": float(np.quantile(array, 0.01)),
        "p50": float(np.quantile(array, 0.50)),
        "p99": float(np.quantile(array, 0.99)),
        "removed_fraction": float(np.mean(array <= 0.0)),
    }


def diagnose_x5a(project_root: str | Path, *, limit_files: int | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config = load_config(root)
    directory = root / "OriData" / "梁子湖X5A"
    paths = sorted(directory.glob("*.dat"))
    if limit_files is not None:
        paths = paths[: int(limit_files)]
    if not paths:
        raise FileNotFoundError(directory)
    default_layout = X5ALayout()
    summaries = [summarize_x5a_file(path, layout=default_layout) for path in paths]
    return {
        "schema_version": "RADAR-LN3-V27d-r01-x5a-diagnostic-v1",
        "experiment_id": config["experiment_id"],
        "input": {
            "directory": directory.as_posix(),
            "file_count": len(paths),
            "layout_used": {
                "payload_offset": default_layout.payload_offset,
                "count_mode": default_layout.count_mode,
                "tail_bytes": default_layout.tail_bytes,
            },
            "layout_candidates_first_file": diagnose_layout(paths[0]),
        },
        "files": summaries,
        "limitations": [
            "X5A 字段语义未知，u16 列不解释为物理坐标。",
            "X5A 没有逐点标签，不能计算真实 Drone/Bird Precision 或 Recall。",
            "X5A 不进入 r20 的 Excel parent split；本报告只做外部质量与格式诊断。",
        ],
    }


def diagnose_excel_identity(
    project_root: str | Path,
    sequences: list[Any],
    *,
    target_retention_floor: float = 0.985,
) -> dict[str, Any]:
    """在不加载 checkpoint 的情况下，核实降噪权重是否可能为恒等门。"""

    root = Path(project_root).resolve()
    config = load_config(root)
    if not sequences:
        raise ValueError("Excel sequence list is empty")
    calibration = fit_denoiser(
        sequences,
        config=DenoisingConfig(target_retention_floor=float(target_retention_floor)),
    )
    by_label: dict[str, list[float]] = {}
    score_by_label: dict[str, list[float]] = {}
    removed_by_label: dict[str, int] = {}
    event_count_by_label: dict[str, int] = {}
    for sequence in sequences:
        label = str(getattr(sequence, "label_name", ""))
        result = score_sequence(calibration, sequence)
        by_label.setdefault(label, []).extend(result.keep_weight.astype(float).tolist())
        score_by_label.setdefault(label, []).extend(result.remove_score.astype(float).tolist())
        removed_by_label[label] = removed_by_label.get(label, 0) + int(result.removed_count)
        event_count_by_label[label] = event_count_by_label.get(label, 0) + int(len(result.keep_weight))
    all_weights = [value for values in by_label.values() for value in values]
    all_scores = [value for values in score_by_label.values() for value in values]
    return {
        "schema_version": "RADAR-LN3-V27d-r01-excel-identity-diagnostic-v1",
        "experiment_id": config["experiment_id"],
        "scope": {
            "sequence_count": len(sequences),
            "fit_split": "caller-provided train-only sequences",
            "target_retention_floor": float(target_retention_floor),
            "r20_reference": validate_r20_reference(root, config),
        },
        "calibration": {
            "threshold": float(calibration.threshold),
            "threshold_selection": dict(calibration.threshold_selection),
        },
        "keep_weight": {
            "all": _weights_summary(all_weights),
            "by_label": {key: _weights_summary(value) for key, value in sorted(by_label.items())},
            "removed_by_label": removed_by_label,
            "event_count_by_label": event_count_by_label,
        },
        "remove_score": {
            "min": float(min(all_scores)) if all_scores else 0.0,
            "p50": float(np.quantile(all_scores, 0.50)) if all_scores else 0.0,
            "p95": float(np.quantile(all_scores, 0.95)) if all_scores else 0.0,
            "max": float(max(all_scores)) if all_scores else 0.0,
        },
        "interpretation": {
            "identity_gate_possible": bool(all_weights and np.allclose(all_weights, 1.0)),
            "classification_effect_not_measured": True,
            "reason": "r01 逐点评分与 r20 parent-final 分类是不同层级；只有将权重实际送入模型并观察 logits/argmax 才能判断分类影响。",
        },
    }


def diagnose_excel_from_project(project_root: str | Path, *, split_name: str = "train") -> dict[str, Any]:
    """只读加载已有 r17 processed Excel 数据，复核逐点分数分布。"""

    root = Path(project_root).resolve()
    from v27.V27d.r17.data import R17Dataset
    from v27.V27d.r17.protocol import load_config as load_r17_config

    r17_config = load_r17_config(root)
    dataset = R17Dataset(root, r17_config, task="l1", split_name=split_name)
    report = diagnose_excel_identity(root, list(dataset))
    report["scope"]["split_name"] = split_name
    report["scope"]["dataset_count"] = len(dataset)
    return report


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


__all__ = ["diagnose_excel_from_project", "diagnose_excel_identity", "diagnose_x5a", "write_json"]
