from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.rd_physical_schema import PHYSICAL_RD_NORMALIZATION_SCHEMA_VERSION


SCHEMA_VERSION = PHYSICAL_RD_NORMALIZATION_SCHEMA_VERSION
DEFAULT_LABELS = ("DroneTarget", "BirdTarget")
BALANCED_GROUP_SAMPLE_COUNT = 4096


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve_project_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def _parse_indices(value: object, *, sample_id: str) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError(f"normalization index frame references are not JSON text: {sample_id}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid normalization frame references: {sample_id}") from error
    indices = np.asarray(parsed, dtype=np.int64)
    if indices.ndim != 1 or not indices.size or (indices < 0).any() or (np.diff(indices) < 0).any():
        raise ValueError(f"invalid normalization frame references: {sample_id}")
    return indices


def _balanced_group_sample(samples_by_group: dict[str, list[np.ndarray]]) -> np.ndarray:
    balanced: list[np.ndarray] = []
    for group_id in sorted(samples_by_group):
        parent_values = [item.reshape(-1) for item in samples_by_group[group_id] if item.size]
        if not parent_values:
            continue
        per_parent_count = max(1, BALANCED_GROUP_SAMPLE_COUNT // len(parent_values))
        parent_samples: list[np.ndarray] = []
        for values in parent_values:
            if values.size > per_parent_count:
                positions = np.linspace(0, values.size - 1, per_parent_count, dtype=np.int64)
                values = values[positions]
            parent_samples.append(values)
        group_values = np.concatenate(parent_samples)
        if group_values.size > BALANCED_GROUP_SAMPLE_COUNT:
            positions = np.linspace(0, group_values.size - 1, BALANCED_GROUP_SAMPLE_COUNT, dtype=np.int64)
            group_values = group_values[positions]
        balanced.append(group_values)
    if not balanced:
        raise ValueError("normalization produced no balanced group samples")
    return np.concatenate(balanced)


def build_rd_physical_normalization_stats(
    project_root: Path,
    *,
    index_name: str = "expert1_rd_only_causal_window_index_v1_physical_v3_1.csv",
    output_name: str = "expert1_rd_physical_train_db_normalization_v4.json",
    labels: tuple[str, ...] = DEFAULT_LABELS,
    sample_stride: int = 257,
) -> dict[str, object]:
    if sample_stride < 2:
        raise ValueError("sample_stride must be at least two")
    processed_root = project_root / "data" / "processed"
    index_path = processed_root / index_name
    index = pd.read_csv(index_path, low_memory=False)
    required_columns = {
        "sample_id",
        "parent_sample_id",
        "parent_split_group_id",
        "label_name",
        "split_name",
        "is_trainable",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
        "rd_present",
        "rd_is_multimodal",
        "rd_phase_missing",
        "rd_coverage_ratio",
        "rd_time_sync_status",
    }
    missing = sorted(required_columns - set(index.columns))
    if missing:
        raise ValueError(f"normalization index is missing columns: {missing}")
    selected = index[
        index["split_name"].astype(str).eq("train")
        & index["is_trainable"].map(_as_bool)
        & index["label_name"].astype(str).isin(set(labels))
        & index["rd_present"].map(_as_bool)
        & index["rd_is_multimodal"].map(_as_bool)
        & ~index["rd_phase_missing"].map(_as_bool)
        & index["rd_coverage_ratio"].astype(float).ge(1.0 - 1.0e-6)
        & index["rd_time_sync_status"].astype(str).eq("canonical_parent_association")
    ].copy()
    if selected.empty:
        raise ValueError("normalization selected no trainable train windows")
    group_label_counts = selected.assign(
        _group=selected["parent_split_group_id"].astype(str).str.strip()
    ).groupby("_group")["label_name"].nunique()
    if int(group_label_counts.max()) > 1:
        raise ValueError("normalization content group has inconsistent labels")
    selected_by_parent: dict[str, dict[str, object]] = {}
    for row in selected.itertuples(index=False):
        parent_id = str(row.parent_sample_id)
        item = selected_by_parent.setdefault(
            parent_id,
            {
                "label_name": str(row.label_name),
                "group_id": str(row.parent_split_group_id).strip(),
                "rd_tensor_path": str(row.rd_tensor_path),
                "indices": set(),
            },
        )
        if not item["group_id"] or item["group_id"].lower() == "nan":
            raise ValueError(f"normalization parent has no content group: {parent_id}")
        if (
            item["label_name"] != str(row.label_name)
            or item["group_id"] != str(row.parent_split_group_id).strip()
            or item["rd_tensor_path"] != str(row.rd_tensor_path)
        ):
            raise ValueError(f"normalization parent metadata conflict: {parent_id}")
        item["indices"].update(int(value) for value in _parse_indices(row.rd_tensor_frame_indices_json, sample_id=str(row.sample_id)))
    parent_records: list[dict[str, object]] = []
    parent_count_by_label: defaultdict[str, int] = defaultdict(int)
    frame_count_by_label: defaultdict[str, int] = defaultdict(int)
    for parent_id in sorted(selected_by_parent):
        item = selected_by_parent[parent_id]
        path = _resolve_project_path(project_root, str(item["rd_tensor_path"]))
        if not path.exists():
            raise FileNotFoundError(f"normalization physical tensor not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            amplitude = np.asarray(payload["amplitude"], dtype=np.float32)
        indices = np.asarray(sorted(item["indices"]), dtype=np.int64)
        if int(indices[-1]) >= int(amplitude.shape[0]):
            raise ValueError(f"normalization frame reference exceeds tensor bounds: {parent_id}")
        values = amplitude[indices]
        finite = values[np.isfinite(values)]
        if not finite.size:
            raise ValueError(f"normalization parent has no finite amplitude: {parent_id}")
        values64 = finite.astype(np.float64, copy=False)
        offset = int(hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:8], 16) % sample_stride
        sampled_values = finite.reshape(-1)[offset::sample_stride]
        if not sampled_values.size:
            sampled_values = finite.reshape(-1)[:1]
        parent_records.append(
            {
                "parent_id": parent_id,
                "group_id": str(item["group_id"]),
                "path": path,
                "indices": indices,
                "mean": float(values64.mean()),
                "second_moment": float(np.square(values64).mean()),
                "sampled_values": sampled_values,
            }
        )
        label_name = str(item["label_name"])
        parent_count_by_label[label_name] += 1
        frame_count_by_label[label_name] += int(indices.size)
    records_by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
    sample_values_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in parent_records:
        records_by_group[str(record["group_id"])].append(record)
        sample_values_by_group[str(record["group_id"])].append(record["sampled_values"])
    group_means = np.asarray(
        [np.mean([float(record["mean"]) for record in records]) for records in records_by_group.values()],
        dtype=np.float64,
    )
    group_second_moments = np.asarray(
        [
            np.mean([float(record["second_moment"]) for record in records])
            for records in records_by_group.values()
        ],
        dtype=np.float64,
    )
    mean = float(group_means.mean())
    variance = max(0.0, float(group_second_moments.mean()) - mean * mean)
    std = float(np.sqrt(variance))
    if std <= 1.0e-6:
        raise ValueError("normalization amplitude standard deviation is not positive")
    sample = _balanced_group_sample(sample_values_by_group)
    clip_low = float(np.quantile(sample, 0.01))
    clip_high = float(np.quantile(sample, 0.99))
    if clip_high <= clip_low:
        raise ValueError("normalization amplitude clipping bounds are invalid")
    valid_intervals: list[np.ndarray] = []
    for record in parent_records:
        path = record["path"]
        indices = record["indices"]
        with np.load(path, allow_pickle=False) as payload:
            time_seconds = np.asarray(payload["time_seconds"], dtype=np.float64)
        selected_time = time_seconds[indices]
        differences = np.diff(selected_time)
        consecutive = np.diff(indices) == 1
        finite_positive = differences[(consecutive) & np.isfinite(differences) & (differences > 0.0)]
        if finite_positive.size:
            valid_intervals.append(finite_positive)
    if not valid_intervals:
        raise ValueError("normalization found no valid consecutive RD frame intervals")
    interval_values = np.concatenate(valid_intervals)
    time_floor = max(1.0e-4, float(np.quantile(interval_values, 0.01) * 0.5))
    time_max = max(time_floor, float(np.quantile(interval_values, 0.99) * 2.0))
    delta_samples_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in parent_records:
        parent_id = str(record["parent_id"])
        path = record["path"]
        indices = record["indices"]
        with np.load(path, allow_pickle=False) as payload:
            amplitude = np.asarray(payload["amplitude"], dtype=np.float32)
            time_seconds = np.asarray(payload["time_seconds"], dtype=np.float64)
        selected_amplitude = np.nan_to_num(amplitude[indices], nan=0.0, posinf=0.0, neginf=0.0)
        selected_time = time_seconds[indices]
        differences = np.diff(selected_time)
        consecutive = np.diff(indices) == 1
        valid = consecutive & np.isfinite(differences) & (differences >= time_floor) & (differences <= time_max)
        if not valid.any():
            continue
        residual = (np.clip(selected_amplitude, clip_low, clip_high) - mean) / std
        delta = (residual[1:] - residual[:-1]) / np.maximum(differences, time_floor)[:, None, None]
        values = delta[valid].reshape(-1)
        offset = int(hashlib.sha256(parent_id.encode("utf-8")).hexdigest()[:8], 16) % sample_stride
        sampled_delta = values[offset::sample_stride]
        if sampled_delta.size:
            delta_samples_by_group[str(record["group_id"])].append(sampled_delta)
    if not delta_samples_by_group:
        raise ValueError("normalization found no valid delta-rate samples")
    delta_sample = _balanced_group_sample(delta_samples_by_group)
    delta_center = float(np.median(delta_sample))
    delta_scale = max(float((np.quantile(delta_sample, 0.75) - np.quantile(delta_sample, 0.25)) / 1.349), 1.0e-3)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "train-only parent/content-group balanced complete-present physical RD normalization; CNN consumes normalized residual while raw dB remains statistics-only",
        "index_name": index_name,
        "index_sha256": _sha256_file(index_path),
        "split_name": "train",
        "labels": list(labels),
        "generator_code_sha256": _sha256_file(Path(__file__).resolve()),
        "selection": {
            "trainable_window_count": int(len(selected)),
            "parent_count": int(len(selected_by_parent)),
            "content_group_count": int(len(records_by_group)),
            "parent_count_by_label": dict(sorted(parent_count_by_label.items())),
            "unique_frame_count_by_label": dict(sorted(frame_count_by_label.items())),
            "sample_stride": int(sample_stride),
            "balanced_group_sample_count": int(BALANCED_GROUP_SAMPLE_COUNT),
            "sample_value_count": int(sample.size),
        },
        "aggregation": "parent_moments -> content_group_mean -> equal-content-group moments; balanced group samples for quantiles",
        "amplitude_db": {
            "mean": float(mean),
            "std": std,
            "clip_low": clip_low,
            "clip_high": clip_high,
            "normalized_residual_formula": "(clip(raw_db, clip_low, clip_high) - mean) / std",
        },
        "delta_rate": {
            "time_floor_seconds": time_floor,
            "time_max_seconds": time_max,
            "center": delta_center,
            "scale": delta_scale,
            "formula": "tanh((delta(normalized_residual) / max(delta_t, time_floor) - center) / scale); invalid intervals map to 0",
        },
        "test_evaluation_executed": False,
    }
    output_path = processed_root / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**payload, "output_path": output_path.relative_to(project_root).as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only physical RD dB normalization statistics.")
    parser.add_argument("--index-name", default="expert1_rd_only_causal_window_index_v1_physical_v3_1.csv")
    parser.add_argument("--output-name", default="expert1_rd_physical_train_db_normalization_v4.json")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--sample-stride", type=int, default=257)
    args = parser.parse_args()
    labels = tuple(value.strip() for value in str(args.labels).split(",") if value.strip())
    result = build_rd_physical_normalization_stats(
        Path.cwd(),
        index_name=args.index_name,
        output_name=args.output_name,
        labels=labels,
        sample_stride=args.sample_stride,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
