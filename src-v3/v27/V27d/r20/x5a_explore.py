"""梁子湖 X5A 原始点集探查器。

X5A ``.dat`` 不是 CQ08 raw-DAT（后者以 0xFA55FA55 帧头开头），当前解析器
只依赖 X5A 文件内部的 BB/AA 标记、长度字段和重复点记录结构，语义未知字段
统一命名为 ``u16_00`` ... ``u16_11``，避免把候选坐标误写成物理含义。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import openpyxl

MAGIC = b"\xbb\xbb\xbb\xbb"
AA_MARK = b"\xaa\xaa\xaa\xaa"
POINT_WIDTH = 12
POINT_BYTES = POINT_WIDTH * 2
POINT_PAYLOAD_OFFSET = 50
LABEL_MAP = {
    "10": "Other",
    "20": "DroneTarget",
    "30": "BirdTarget",
    "40": "BalloonTarget",
    "99": "ClutterTarget",
}


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    result = np.percentile(values, [1, 5, 50, 95, 99])
    return {
        "p01": float(result[0]),
        "p05": float(result[1]),
        "p50": float(result[2]),
        "p95": float(result[3]),
        "p99": float(result[4]),
    }


def _new_field_stats() -> dict[str, Any]:
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": None,
        "max": None,
        "zero_count": 0,
        "saturated_65535_count": 0,
        "quantile_sample": [],
    }


def _update_field_stats(stats: dict[str, Any], values: np.ndarray, *, rng: np.random.Generator) -> None:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return
    stats["count"] += int(values.size)
    stats["sum"] += float(values.sum(dtype=np.float64))
    stats["sum_sq"] += float(np.square(values).sum(dtype=np.float64))
    current_min = float(values.min())
    current_max = float(values.max())
    stats["min"] = current_min if stats["min"] is None else min(stats["min"], current_min)
    stats["max"] = current_max if stats["max"] is None else max(stats["max"], current_max)
    stats["zero_count"] += int(np.sum(values == 0.0))
    stats["saturated_65535_count"] += int(np.sum(values >= 65535.0))
    sample = stats["quantile_sample"]
    limit = 50000
    if len(sample) < limit:
        take = min(limit - len(sample), values.size)
        indices = rng.choice(values.size, size=take, replace=False)
        sample.extend(float(value) for value in values[indices])
    else:
        # Reservoir sampling，保持大文件与小文件的采样机会近似一致。
        for value in values:
            position = int(rng.integers(0, max(1, stats["count"])))
            if position < limit:
                sample[position] = float(value)


def _finalize_field_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = int(stats.pop("count"))
    total = float(stats.pop("sum"))
    total_sq = float(stats.pop("sum_sq"))
    sample = np.asarray(stats.pop("quantile_sample"), dtype=np.float64)
    mean = total / max(1, count)
    variance = max(0.0, total_sq / max(1, count) - mean * mean)
    result = {
        "dtype": "uint16",
        "count": count,
        "min": stats.pop("min"),
        "max": stats.pop("max"),
        "mean": mean,
        "variance_population": variance,
        "std_population": float(np.sqrt(variance)),
        "zero_fraction": stats.pop("zero_count") / max(1, count),
        "saturated_65535_fraction": stats.pop("saturated_65535_count") / max(1, count),
    }
    result.update(_percentiles(sample))
    return result


def _parse_file(path: Path, *, sample_limit: int = 20000, seed: int = 20260916) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    payload = path.read_bytes()
    offset = 0
    records = 0
    point_count = 0
    lengths: list[int] = []
    points_per_record: list[int] = []
    marker_counts: Counter[str] = Counter()
    frame_values: list[int] = []
    field_stats = [_new_field_stats() for _ in range(POINT_WIDTH)]
    sample: list[np.ndarray] = []
    rng = np.random.default_rng(seed)
    while offset < len(payload):
        if payload[offset : offset + 4] != MAGIC:
            raise ValueError(f"X5A record magic mismatch: {path}:{offset}")
        if offset + 6 > len(payload):
            raise ValueError(f"X5A record header truncated: {path}:{offset}")
        record_bytes = int.from_bytes(payload[offset + 4 : offset + 6], "little") + 18
        end = offset + record_bytes
        if end > len(payload):
            raise ValueError(f"X5A record exceeds file: {path}:{offset}")
        record = payload[offset:end]
        if record[18:22] != AA_MARK:
            raise ValueError(f"X5A AA marker mismatch: {path}:{offset}")
        # 该字段为“点数减 1”，在所有八个文件上与 record_bytes 完全一致。
        points_in_record = int.from_bytes(record[22:26], "little") + 1
        expected_bytes = POINT_PAYLOAD_OFFSET + POINT_BYTES * points_in_record
        if expected_bytes != record_bytes:
            raise ValueError(
                f"X5A record length/point count mismatch: {path}:{offset}: "
                f"{record_bytes} != {expected_bytes}"
            )
        marker_counts[record[30:34].hex()] += 1
        frame_values.append(int.from_bytes(record[26:30], "little"))
        values = np.frombuffer(
            record,
            dtype="<u2",
            count=points_in_record * POINT_WIDTH,
            offset=POINT_PAYLOAD_OFFSET,
        ).reshape(points_in_record, POINT_WIDTH)
        for index in range(POINT_WIDTH):
            _update_field_stats(field_stats[index], values[:, index], rng=rng)
        if len(sample) < sample_limit:
            take = min(sample_limit - len(sample), points_in_record)
            chosen = rng.choice(points_in_record, size=take, replace=False)
            sample.extend(values[chosen].astype(np.float64, copy=True))
        else:
            for row in values:
                position = int(rng.integers(0, max(1, point_count + 1)))
                if position < sample_limit:
                    sample[position] = row.astype(np.float64, copy=True)
        records += 1
        point_count += points_in_record
        lengths.append(record_bytes)
        points_per_record.append(points_in_record)
        offset = end
    sample_array = np.asarray(sample, dtype=np.float64).reshape(-1, POINT_WIDTH)
    fields = {
        f"u16_{index:02d}": _finalize_field_stats(field_stats[index])
        for index in range(POINT_WIDTH)
    }
    summary = {
        "path": path.as_posix(),
        "file_name": path.name,
        "file_size_bytes": len(payload),
        "record_count": records,
        "point_count": point_count,
        "record_bytes": {
            "min": int(min(lengths)),
            "mean": float(np.mean(lengths)),
            "max": int(max(lengths)),
        },
        "points_per_record": {
            "min": int(min(points_per_record)),
            "mean": float(np.mean(points_per_record)),
            "max": int(max(points_per_record)),
        },
        "frame_or_tick_u32": {
            "min": int(min(frame_values)),
            "max": int(max(frame_values)),
            "monotonic_non_decreasing": bool(np.all(np.diff(frame_values) >= 0)),
        },
        "record_marker_variants": dict(marker_counts),
        "weak_file_name_label": "BirdCaptureCandidate" if "鸟" in path.stem else None,
        "fields": fields,
    }
    return summary, sample_array, np.asarray(frame_values, dtype=np.float64)


def _density_proxy(samples: Iterable[np.ndarray], *, bins: int = 32) -> dict[str, Any]:
    arrays = [value for value in samples if value.size]
    if not arrays:
        return {"status": "no_samples"}
    points = np.concatenate(arrays, axis=0)
    axes = points[:, :3]
    lower = np.percentile(axes, 1, axis=0)
    upper = np.percentile(axes, 99, axis=0)
    scale = np.maximum(upper - lower, 1.0)
    normalized = np.clip((axes - lower) / scale, 0.0, 1.0)
    coordinates = np.minimum((normalized * bins).astype(np.int64), bins - 1)
    keys = coordinates[:, 0] * bins * bins + coordinates[:, 1] * bins + coordinates[:, 2]
    counts = np.bincount(keys, minlength=bins**3)
    nonzero = counts[counts > 0]
    return {
        "status": "proxy_only_unknown_semantics",
        "axes": ["u16_00", "u16_01", "u16_02"],
        "grid_bins_per_axis": bins,
        "sample_count": int(len(points)),
        "robust_lower_p01": lower.tolist(),
        "robust_upper_p99": upper.tolist(),
        "occupied_cell_count": int(len(nonzero)),
        "occupied_fraction": float(len(nonzero) / (bins**3)),
        "cell_count_min": int(nonzero.min()),
        "cell_count_median": float(np.median(nonzero)),
        "cell_count_p95": float(np.percentile(nonzero, 95)),
        "cell_count_max": int(nonzero.max()),
        "sparse_cell_fraction_count_le_2": float(np.mean(nonzero <= 2)),
    }


def _companion_excel_summary(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"dataset": "OriData/excel", "status": "separate_labeled_proxy"}
    files = sorted((root / "OriData" / "excel").glob("tck_*.xlsx"))
    counts: Counter[str] = Counter()
    file_rows: dict[str, int] = {}
    columns: list[str] = []
    for path in files:
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheet = book.active
        iterator = sheet.iter_rows(values_only=True)
        header = next(iterator, ())
        columns = [str(value) for value in header]
        rows = 0
        for row in iterator:
            rows += 1
            raw = str(row[14]).strip() if len(row) > 14 and row[14] is not None else ""
            counts[LABEL_MAP.get(raw, raw)] += 1
        file_rows[path.name] = rows
        book.close()
    result.update({"file_count": len(files), "columns": columns, "rows_by_file": file_rows, "label_counts": dict(counts), "total_rows": int(sum(counts.values()))})
    total = max(1, sum(counts.values()))
    result["label_ratios"] = {key: value / total for key, value in counts.items()}
    return result


def explore(root: str | Path) -> dict[str, Any]:
    project_root = Path(root).resolve()
    directory = project_root / "OriData" / "梁子湖X5A"
    paths = sorted(directory.glob("*.dat"))
    if not paths:
        raise FileNotFoundError(directory)
    summaries: list[dict[str, Any]] = []
    samples: list[np.ndarray] = []
    for path in paths:
        summary, sample, _ = _parse_file(path)
        summaries.append(summary)
        samples.append(sample)
    total_points = sum(item["point_count"] for item in summaries)
    aggregate: dict[str, Any] = {
        "file_count": len(paths),
        "total_bytes": int(sum(item["file_size_bytes"] for item in summaries)),
        "total_records": int(sum(item["record_count"] for item in summaries)),
        "total_points": int(total_points),
        "field_semantics": "unknown; u16 offset schema only",
        "fields": {},
    }
    for index in range(POINT_WIDTH):
        field = f"u16_{index:02d}"
        values = []
        for item in summaries:
            values.append(item["fields"][field])
        count = sum(int(value["count"]) for value in values)
        total = sum(float(value["mean"]) * int(value["count"]) for value in values)
        variance = sum(
            int(value["count"])
            * (float(value["variance_population"]) + (float(value["mean"]) - total / max(1, count)) ** 2)
            for value in values
        ) / max(1, count)
        aggregate["fields"][field] = {
            "dtype": "uint16",
            "count": count,
            "min": min(value["min"] for value in values),
            "max": max(value["max"] for value in values),
            "mean": total / max(1, count),
            "variance_population": variance,
            "zero_fraction": sum(float(value["zero_fraction"]) * int(value["count"]) for value in values) / max(1, count),
            "saturated_65535_fraction": sum(float(value["saturated_65535_fraction"]) * int(value["count"]) for value in values) / max(1, count),
        }
    return {
        "schema_version": "RADAR-LN3-X5A-point-set-exploration-v1",
        "project_root": project_root.as_posix(),
        "raw_directory": directory.as_posix(),
        "binary_layout": {
            "record_magic": "bbbbbbbb",
            "record_length_field": "uint16_le_at_offset_4; record_bytes=field+18",
            "aa_marker": "aaaaaaaa_at_offset_18",
            "point_count": "uint32_le_at_offset_22_plus_1",
            "frame_or_tick_candidate": "uint32_le_at_offset_26",
            "variant_marker": "4_bytes_at_offset_30",
            "fixed_header_candidate": "8 x uint16_le_at_offset_34",
            "point_payload": "point_count x 12 x uint16_le_at_offset_50",
            "semantic_status": "candidate_offsets_only; no X5A schema/label sidecar found",
        },
        "aggregate": aggregate,
        "files": summaries,
        "density_proxy": _density_proxy(samples),
        "companion_labeled_proxy": _companion_excel_summary(project_root),
        "limitations": [
            "X5A .dat 内没有可识别的逐点类别标签；文件名中的“鸟”只能作为弱文件级提示。",
            "u16_00..u16_11 的物理含义尚未由官方协议或 sidecar 证实，不能直接等同于距离/方位/速度。",
            "因此 X5A 原始点集本身不能合法计算 Drone/Bird 的 Precision、Recall、F1；指标必须来自有标签且与模型输入一致的评估集。",
            "OriData/excel 的 24,758 行标签统计是独立的 TR 代理数据，不是 X5A .dat 的逐点标签映射。",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="探查梁子湖 X5A 原始点集")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("tmp/x5a_exploration.json"))
    args = parser.parse_args()
    result = explore(args.project_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
    print(json.dumps(result["companion_labeled_proxy"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
