from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.core.data.rd import (
    EncodedRDSequence,
    MULTIMODAL_RD_PATCH_DOPPLER_BINS,
    MULTIMODAL_RD_PATCH_RANGE_BINS,
    encode_multimodal_rd,
    encode_rd_summary,
    load_rd_summary_frame,
    load_rd_tensor,
)


EXPERT1_LABELS = {
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
}


MANIFEST_COLUMNS = [
    "segment_id",
    "parent_sample_id",
    "track_id",
    "source_name",
    "label_name",
    "segment_start_seconds",
    "segment_end_seconds",
    "segment_duration_seconds",
    "segment_frame_count",
    "rd_feature_path",
    "rd_patch_path",
    "rd_deep_patch_path",
    "shard_key",
    "segment_index",
    "rd_feature_key",
    "rd_patch_key",
    "rd_deep_patch_key",
    "rd_coverage_ratio",
    "rd_frame_start_index",
    "rd_frame_end_index",
    "rd_time_start_seconds",
    "rd_time_end_seconds",
    "rd_time_coverage_ratio",
    "rd_quality_flags",
    "rd_time_sync_status",
    "rd_preprocess_version",
    "rd_is_multimodal",
]


PREPROCESS_VERSION = "expert1_rd_segment_store_v3"


@dataclass(frozen=True)
class ParentRDData:
    encoded: EncodedRDSequence | None
    rd_is_multimodal: bool
    time_sync_status: str
    rd_tensor: dict[str, np.ndarray] | None = None
    rd_time_seconds: np.ndarray | None = None
    rd_time_axis_is_exact: bool = False


def _safe_name(value: object) -> str:
    safe = str(value).replace("|", "__").replace("/", "_").replace("\\", "_").replace(":", "_")
    return safe.replace(" ", "_")


def _is_present(value: object) -> bool:
    text = str(value if value is not None else "").strip().lower()
    return text not in {"", "nan", "none"}


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _parent_id_from_row(row: pd.Series | dict[str, object]) -> str:
    parent_id = str(row.get("parent_sample_id", "") or "").strip()
    if parent_id:
        return parent_id
    sample_id = str(row.get("sample_id", "") or "").strip()
    if "|win" in sample_id:
        return sample_id.split("|win", 1)[0]
    return sample_id


def _expand_summary_rd(encoded: EncodedRDSequence) -> EncodedRDSequence:
    summary_values = encoded.features[:, :3]
    width = 512
    expanded = torch.zeros((summary_values.shape[0], width, 4), dtype=torch.float32)
    expanded[:, :, 0] = summary_values[:, :1].repeat(1, width)
    expanded[:, :, 1] = summary_values[:, 1:2].repeat(1, width)
    expanded[:, :, 2] = summary_values[:, 2:3].repeat(1, width)
    patch = torch.zeros(
        (
            int(expanded.shape[0]),
            MULTIMODAL_RD_PATCH_RANGE_BINS,
            MULTIMODAL_RD_PATCH_DOPPLER_BINS,
            3,
        ),
        dtype=torch.float32,
    )
    return EncodedRDSequence(
        features=expanded,
        valid_length=int(expanded.shape[0]),
        patch_features=patch,
        patch_valid_length=int(patch.shape[0]),
        deep_patch_features=patch,
        deep_patch_valid_length=int(patch.shape[0]),
        time_seconds=encoded.time_seconds,
        patch_time_seconds=encoded.patch_time_seconds,
        deep_patch_time_seconds=encoded.deep_patch_time_seconds,
    )


def _segment_ranges(start: float, end: float, segment_seconds: float) -> list[tuple[float, float]]:
    if end < start:
        start, end = end, start
    if segment_seconds <= 0.0:
        raise ValueError("segment_seconds must be positive")
    if end <= start:
        return [(float(start), float(end))]
    ranges: list[tuple[float, float]] = []
    current = float(start)
    epsilon = 1.0e-6
    while current < end - epsilon:
        segment_end = min(float(end), current + float(segment_seconds))
        ranges.append((float(current), float(segment_end)))
        if segment_end >= end - epsilon:
            break
        current = segment_end
    return ranges or [(float(start), float(end))]


def _slice_time_tensor(
    tensor: torch.Tensor,
    valid_length: int,
    start_fraction: float,
    end_fraction: float,
    min_frames: int,
) -> tuple[torch.Tensor, int]:
    length = min(max(1, int(valid_length)), int(tensor.shape[0]))
    start_index = int(math.floor(start_fraction * length))
    end_index = int(math.ceil(end_fraction * length))
    start_index = min(max(0, start_index), length - 1)
    end_index = min(max(start_index + 1, end_index), length)
    min_frames = min(max(1, int(min_frames)), length)
    if end_index - start_index < min_frames:
        center = 0.5 * (start_index + end_index)
        start_index = int(round(center - 0.5 * min_frames))
        start_index = min(max(0, start_index), length - min_frames)
        end_index = start_index + min_frames
    sliced = tensor[:length][start_index:end_index].contiguous()
    return sliced, int(sliced.shape[0])


def _normalize_raw_time_seconds(time_raw: object, *, milliseconds: bool = False) -> np.ndarray | None:
    values = np.asarray(time_raw, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    finite = np.isfinite(values)
    if not bool(finite.all()):
        return None
    if values.size > 1:
        diffs = np.diff(values)
        finite_diffs = diffs[np.isfinite(diffs)]
        if finite_diffs.size == 0 or float(np.nanmedian(np.abs(finite_diffs))) <= 0.0:
            return None
    base = float(values[0])
    span = float(values[-1] - base)
    if values.size > 1 and span <= 0.0:
        return None
    if milliseconds or abs(float(values[0])) > 1000.0:
        normalized = (values - base) / 1000.0
    else:
        normalized = values - base
    return normalized.astype(np.float64)


def _slice_rd_tensor_frames(rd_tensor: dict[str, np.ndarray], start_index: int, end_index: int) -> dict[str, np.ndarray]:
    sliced: dict[str, np.ndarray] = {}
    for key, value in rd_tensor.items():
        array = np.asarray(value)
        if key == "time_raw":
            sliced[key] = array[start_index:end_index].copy()
            continue
        if array.ndim == 2:
            sliced[key] = array[:, start_index:end_index].copy()
        elif array.ndim >= 3:
            sliced[key] = array[start_index:end_index].copy()
        else:
            sliced[key] = array.copy()
    return sliced


def _frame_range_from_time_axis(
    rd_time_seconds: np.ndarray,
    segment_start: float,
    segment_end: float,
    min_frames: int,
) -> tuple[int, int, float, str]:
    values = np.asarray(rd_time_seconds, dtype=np.float64).reshape(-1)
    frame_count = int(values.size)
    if frame_count <= 0:
        return 0, 0, 0.0, "time_axis_unavailable"

    if segment_end < segment_start:
        segment_start, segment_end = segment_end, segment_start
    start = float(segment_start)
    end = float(segment_end)
    epsilon = 1.0e-9
    selected = np.where((values >= start - epsilon) & (values <= end + epsilon))[0]
    if selected.size == 0:
        nearest = int(np.argmin(np.abs(values - (0.5 * (start + end)))))
        start_index = nearest
        end_index = nearest + 1
    else:
        start_index = int(selected[0])
        end_index = int(selected[-1]) + 1

    min_frames = min(max(1, int(min_frames)), frame_count)
    if end_index - start_index < min_frames:
        center = 0.5 * (start_index + end_index - 1)
        start_index = int(round(center - 0.5 * (min_frames - 1)))
        start_index = min(max(0, start_index), frame_count - min_frames)
        end_index = start_index + min_frames

    clipped_times = values[start_index:end_index]
    if clipped_times.size == 0:
        return 0, 0, 0.0, "time_axis_unavailable"
    if end <= start:
        coverage_ratio = 1.0
    else:
        covered_start = max(start, float(clipped_times[0]))
        covered_end = min(end, float(clipped_times[-1]))
        coverage_ratio = max(0.0, min(1.0, (covered_end - covered_start) / max(1.0e-6, end - start)))
        if selected.size > 0 and clipped_times.size == 1:
            coverage_ratio = max(coverage_ratio, min(1.0, 1.0 / max(1, frame_count)))
    status = "exact_time_sync"
    if coverage_ratio < 0.25:
        status = "low_coverage"
    return int(start_index), int(end_index), float(coverage_ratio), status


def _frame_range_from_parent_fraction(
    frame_count: int,
    parent_start: float,
    parent_end: float,
    segment_start: float,
    segment_end: float,
    min_frames: int,
) -> tuple[int, int]:
    frame_count = int(max(0, frame_count))
    if frame_count <= 0:
        return 0, 0
    parent_span = max(1.0e-6, float(parent_end) - float(parent_start))
    start_fraction = (float(segment_start) - float(parent_start)) / parent_span
    end_fraction = (float(segment_end) - float(parent_start)) / parent_span
    start_fraction = min(max(start_fraction, 0.0), 1.0)
    end_fraction = min(max(end_fraction, 0.0), 1.0)
    if end_fraction <= start_fraction:
        end_fraction = min(1.0, start_fraction + 1.0 / max(1, frame_count))

    start_index = int(math.floor(start_fraction * frame_count))
    end_index = int(math.ceil(end_fraction * frame_count))
    start_index = min(max(0, start_index), frame_count - 1)
    end_index = min(max(start_index + 1, end_index), frame_count)
    min_frames = min(max(1, int(min_frames)), frame_count)
    if end_index - start_index < min_frames:
        center = 0.5 * (start_index + end_index)
        start_index = int(round(center - 0.5 * min_frames))
        start_index = min(max(0, start_index), frame_count - min_frames)
        end_index = start_index + min_frames
    return int(start_index), int(end_index)


def _slice_encoded_rd(
    encoded: EncodedRDSequence,
    parent_start: float,
    parent_end: float,
    segment_start: float,
    segment_end: float,
    min_frames: int,
    deep_patch_mode: str,
) -> EncodedRDSequence:
    parent_span = max(1.0e-6, float(parent_end) - float(parent_start))
    start_fraction = (float(segment_start) - float(parent_start)) / parent_span
    end_fraction = (float(segment_end) - float(parent_start)) / parent_span
    start_fraction = min(max(start_fraction, 0.0), 1.0)
    end_fraction = min(max(end_fraction, 0.0), 1.0)
    if end_fraction <= start_fraction:
        end_fraction = min(1.0, start_fraction + 1.0 / max(1, int(encoded.valid_length)))

    features, valid_length = _slice_time_tensor(
        encoded.features,
        encoded.valid_length,
        start_fraction,
        end_fraction,
        min_frames,
    )

    patch_features = None
    patch_valid_length = 0
    if encoded.patch_features is not None:
        patch_features, patch_valid_length = _slice_time_tensor(
            encoded.patch_features,
            encoded.patch_valid_length or int(encoded.patch_features.shape[0]),
            start_fraction,
            end_fraction,
            min_frames,
        )

    deep_patch_features = None
    deep_patch_valid_length = 0
    if deep_patch_mode == "full" and encoded.deep_patch_features is not None:
        deep_patch_features, deep_patch_valid_length = _slice_time_tensor(
            encoded.deep_patch_features,
            encoded.deep_patch_valid_length or int(encoded.deep_patch_features.shape[0]),
            start_fraction,
            end_fraction,
            min_frames,
        )
    elif deep_patch_mode == "patch" and patch_features is not None:
        deep_patch_features = patch_features
        deep_patch_valid_length = patch_valid_length

    return EncodedRDSequence(
        features=features,
        valid_length=valid_length,
        patch_features=patch_features,
        patch_valid_length=patch_valid_length,
        deep_patch_features=deep_patch_features,
        deep_patch_valid_length=deep_patch_valid_length,
        time_seconds=(
            _slice_time_tensor(encoded.time_seconds, encoded.valid_length, start_fraction, end_fraction, min_frames)[0]
            if encoded.time_seconds is not None
            else None
        ),
        patch_time_seconds=(
            _slice_time_tensor(
                encoded.patch_time_seconds,
                encoded.patch_valid_length or int(encoded.patch_time_seconds.shape[0]),
                start_fraction,
                end_fraction,
                min_frames,
            )[0]
            if encoded.patch_time_seconds is not None
            else None
        ),
        deep_patch_time_seconds=(
            _slice_time_tensor(
                encoded.deep_patch_time_seconds,
                encoded.deep_patch_valid_length or int(encoded.deep_patch_time_seconds.shape[0]),
                start_fraction,
                end_fraction,
                min_frames,
            )[0]
            if encoded.deep_patch_time_seconds is not None and deep_patch_features is not None
            else None
        ),
    )


def _slice_parent_rd_segment(
    parent_rd: ParentRDData,
    parent_start: float,
    parent_end: float,
    segment_start: float,
    segment_end: float,
    min_frames: int,
    deep_patch_mode: str,
) -> tuple[EncodedRDSequence, str, int, int, float | None, float | None, float]:
    if parent_rd.rd_tensor is not None and parent_rd.rd_time_seconds is not None and parent_rd.rd_time_axis_is_exact:
        start_index, end_index, time_coverage, status = _frame_range_from_time_axis(
            parent_rd.rd_time_seconds,
            segment_start,
            segment_end,
            min_frames=min_frames,
        )
        if end_index > start_index:
            clipped_tensor = _slice_rd_tensor_frames(parent_rd.rd_tensor, start_index, end_index)
            clipped_encoded = encode_multimodal_rd(clipped_tensor, legacy=True)
            if deep_patch_mode == "patch" and clipped_encoded.patch_features is not None:
                clipped_encoded = EncodedRDSequence(
                    features=clipped_encoded.features,
                    valid_length=clipped_encoded.valid_length,
                    patch_features=clipped_encoded.patch_features,
                    patch_valid_length=clipped_encoded.patch_valid_length,
                    deep_patch_features=clipped_encoded.patch_features,
                    deep_patch_valid_length=clipped_encoded.patch_valid_length,
                    time_seconds=clipped_encoded.time_seconds,
                    patch_time_seconds=clipped_encoded.patch_time_seconds,
                    deep_patch_time_seconds=clipped_encoded.patch_time_seconds,
                )
            elif deep_patch_mode == "none":
                clipped_encoded = EncodedRDSequence(
                    features=clipped_encoded.features,
                    valid_length=clipped_encoded.valid_length,
                    patch_features=clipped_encoded.patch_features,
                    patch_valid_length=clipped_encoded.patch_valid_length,
                    deep_patch_features=None,
                    deep_patch_valid_length=0,
                    time_seconds=clipped_encoded.time_seconds,
                    patch_time_seconds=clipped_encoded.patch_time_seconds,
                    deep_patch_time_seconds=None,
                )
            segment_times = parent_rd.rd_time_seconds[start_index:end_index]
            return (
                clipped_encoded,
                status,
                start_index,
                end_index - 1,
                float(segment_times[0]),
                float(segment_times[-1]),
                time_coverage,
            )

    if parent_rd.rd_tensor is not None and parent_rd.time_sync_status == "proportional_fallback":
        amplitude = np.asarray(parent_rd.rd_tensor.get("amplitude"))
        frame_count = int(amplitude.shape[0]) if amplitude.ndim >= 3 else int(amplitude.shape[-1]) if amplitude.ndim >= 1 else 0
        start_index, end_index = _frame_range_from_parent_fraction(
            frame_count,
            parent_start,
            parent_end,
            segment_start,
            segment_end,
            min_frames=min_frames,
        )
        if end_index > start_index:
            clipped_tensor = _slice_rd_tensor_frames(parent_rd.rd_tensor, start_index, end_index)
            clipped_encoded = encode_multimodal_rd(clipped_tensor, legacy=True)
            if deep_patch_mode == "patch" and clipped_encoded.patch_features is not None:
                clipped_encoded = EncodedRDSequence(
                    features=clipped_encoded.features,
                    valid_length=clipped_encoded.valid_length,
                    patch_features=clipped_encoded.patch_features,
                    patch_valid_length=clipped_encoded.patch_valid_length,
                    deep_patch_features=clipped_encoded.patch_features,
                    deep_patch_valid_length=clipped_encoded.patch_valid_length,
                    time_seconds=clipped_encoded.time_seconds,
                    patch_time_seconds=clipped_encoded.patch_time_seconds,
                    deep_patch_time_seconds=clipped_encoded.patch_time_seconds,
                )
            elif deep_patch_mode == "none":
                clipped_encoded = EncodedRDSequence(
                    features=clipped_encoded.features,
                    valid_length=clipped_encoded.valid_length,
                    patch_features=clipped_encoded.patch_features,
                    patch_valid_length=clipped_encoded.patch_valid_length,
                    deep_patch_features=None,
                    deep_patch_valid_length=0,
                    time_seconds=clipped_encoded.time_seconds,
                    patch_time_seconds=clipped_encoded.patch_time_seconds,
                    deep_patch_time_seconds=None,
                )
            return (
                clipped_encoded,
                parent_rd.time_sync_status,
                start_index,
                end_index - 1,
                None,
                None,
                0.0,
            )

    if parent_rd.encoded is None:
        raise ValueError("missing parent RD sequence")
    segment = _slice_encoded_rd(
        parent_rd.encoded,
        parent_start,
        parent_end,
        segment_start,
        segment_end,
        min_frames=min_frames,
        deep_patch_mode=deep_patch_mode,
    )
    return segment, parent_rd.time_sync_status, -1, -1, None, None, 0.0


def _load_parent_rd(data_root: Path, row: pd.Series) -> ParentRDData:
    rd_tensor_path = str(row.get("rd_tensor_path", "") or "").strip()
    if _is_present(rd_tensor_path):
        rd_tensor = load_rd_tensor(data_root, rd_tensor_path)
        source_name = str(row.get("source_name", ""))
        rd_time_seconds = _normalize_raw_time_seconds(
            rd_tensor.get("time_raw"),
            milliseconds=source_name == "raw_tr_rd",
        )
        if rd_time_seconds is not None and source_name == "raw_tr_rd":
            return ParentRDData(
                encoded=None,
                rd_is_multimodal=True,
                time_sync_status="exact_time_sync",
                rd_tensor=rd_tensor,
                rd_time_seconds=rd_time_seconds,
                rd_time_axis_is_exact=True,
            )
        return ParentRDData(
            encoded=encode_multimodal_rd(rd_tensor, legacy=True),
            rd_is_multimodal=True,
            time_sync_status="time_axis_unavailable" if rd_time_seconds is None else "proportional_fallback",
            rd_tensor=rd_tensor if rd_time_seconds is not None else None,
            rd_time_seconds=rd_time_seconds,
            rd_time_axis_is_exact=False,
        )
    rd_summary_path = str(row.get("rd_summary_csv_path", "") or "").strip()
    if _is_present(rd_summary_path):
        encoded = encode_rd_summary(load_rd_summary_frame(data_root, rd_summary_path))
        return ParentRDData(
            encoded=_expand_summary_rd(encoded),
            rd_is_multimodal=False,
            time_sync_status="summary_rd",
        )
    return ParentRDData(encoded=None, rd_is_multimodal=False, time_sync_status="unavailable")


def _prepare_parent_rows(index_df: pd.DataFrame) -> pd.DataFrame:
    frame = index_df.copy()
    frame = frame[frame["label_name"].astype(str).isin(EXPERT1_LABELS)].copy()
    if "is_trainable" in frame.columns:
        trainable_mask = frame["is_trainable"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"})
        frame = frame[trainable_mask].copy()
    if frame.empty:
        return frame
    frame["parent_sample_id"] = frame.apply(_parent_id_from_row, axis=1)
    frame["_has_rd"] = frame.apply(
        lambda row: _is_present(row.get("rd_tensor_path", "")) or _is_present(row.get("rd_summary_csv_path", "")),
        axis=1,
    )
    frame = frame[frame["_has_rd"]].copy()
    if frame.empty:
        return frame.drop(columns=["_has_rd"], errors="ignore")
    rows: list[pd.Series] = []
    for _, group in frame.groupby(["source_name", "parent_sample_id"], sort=True):
        base = group.iloc[0].copy()
        starts: list[float] = []
        ends: list[float] = []
        for _, row in group.iterrows():
            if "window_start_seconds" in group.columns:
                value = _finite_float(row.get("window_start_seconds"))
                if value is not None:
                    starts.append(value)
            value = _finite_float(row.get("time_seconds_min"))
            if value is not None:
                starts.append(value)
            if "window_end_seconds" in group.columns:
                value = _finite_float(row.get("window_end_seconds"))
                if value is not None:
                    ends.append(value)
            value = _finite_float(row.get("time_seconds_max"))
            if value is not None:
                ends.append(value)
        if starts:
            base["time_seconds_min"] = min(starts)
        if ends:
            base["time_seconds_max"] = max(ends)
        base["record_count"] = int(pd.to_numeric(group.get("record_count", pd.Series(dtype=float)), errors="coerce").fillna(0).max())
        rows.append(base)
    return pd.DataFrame(rows).drop(columns=["_has_rd"], errors="ignore").reset_index(drop=True)


def _quality_flags(
    segment: EncodedRDSequence,
    coverage_ratio: float,
    time_sync_status: str,
    rd_is_multimodal: bool,
    rd_time_coverage_ratio: float = 0.0,
) -> str:
    flags: list[str] = []
    if coverage_ratio < 0.99:
        flags.append("short_segment")
    if time_sync_status in {"exact_time_sync", "low_coverage"} and rd_time_coverage_ratio < 0.50:
        flags.append("low_coverage")
    if int(segment.valid_length) <= 1:
        flags.append("low_frame_count")
    if not rd_is_multimodal:
        flags.append("summary_rd")
    if time_sync_status != "exact_time_sync":
        flags.append(time_sync_status)
    return ";".join(sorted(set(flags))) if flags else "ok"


def build_expert1_rd_segment_store(
    project_root: Path,
    index_name: str = "expert1_track_index.csv",
    output_name: str = "expert1_rd_segment_store",
    segment_seconds: float = 2.0,
    min_frames_per_segment: int = 1,
    deep_patch_mode: str = "full",
    max_parents: int = 0,
    source_names: Iterable[str] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    if deep_patch_mode not in {"full", "patch", "none"}:
        raise ValueError("deep_patch_mode must be one of: full, patch, none")
    data_root = project_root / "data"
    processed_root = data_root / "processed"
    index_path = processed_root / index_name
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    output_root = processed_root / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale_path in output_root.glob("*_segments.pt"):
            stale_path.unlink()
    if output_name == "expert1_rd_segment_store":
        manifest_path = processed_root / "expert1_rd_segment_manifest.csv"
        summary_path = processed_root / "expert1_rd_segment_summary.json"
    else:
        manifest_path = processed_root / f"{output_name}_manifest.csv"
        summary_path = processed_root / f"{output_name}_summary.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite to rebuild")

    index_df = pd.read_csv(index_path)
    parent_rows = _prepare_parent_rows(index_df)
    if source_names:
        allowed_sources = {str(item) for item in source_names}
        parent_rows = parent_rows[parent_rows["source_name"].astype(str).isin(allowed_sources)].copy()
    if max_parents > 0:
        parent_rows = parent_rows.head(max_parents).copy()

    manifest_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    parent_count = 0
    source_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    quality_counts: Counter[str] = Counter()

    for _, row in parent_rows.iterrows():
        parent_sample_id = str(row.get("parent_sample_id", "") or row.get("sample_id", ""))
        source_name = str(row.get("source_name", ""))
        label_name = str(row.get("label_name", ""))
        parent_start = _finite_float(row.get("time_seconds_min"))
        parent_end = _finite_float(row.get("time_seconds_max"))
        if parent_start is None or parent_end is None:
            errors.append({"parent_sample_id": parent_sample_id, "error": "missing_parent_time_bounds"})
            continue
        try:
            parent_rd = _load_parent_rd(data_root, row)
            if parent_rd.encoded is None and parent_rd.rd_tensor is None:
                errors.append({"parent_sample_id": parent_sample_id, "error": "missing_rd"})
                continue
            parent_path = output_root / f"{_safe_name(source_name)}__{_safe_name(parent_sample_id)}__segments.pt"
            segment_payloads: list[dict[str, object]] = []
            ranges = _segment_ranges(parent_start, parent_end, segment_seconds)
            for segment_index, (segment_start, segment_end) in enumerate(ranges):
                (
                    segment,
                    segment_time_sync_status,
                    frame_start_index,
                    frame_end_index,
                    rd_time_start_seconds,
                    rd_time_end_seconds,
                    rd_time_coverage_ratio,
                ) = _slice_parent_rd_segment(
                    parent_rd,
                    parent_start,
                    parent_end,
                    segment_start,
                    segment_end,
                    min_frames=min_frames_per_segment,
                    deep_patch_mode=deep_patch_mode,
                )
                duration = max(0.0, float(segment_end) - float(segment_start))
                coverage_ratio = min(1.0, duration / float(segment_seconds)) if segment_seconds > 0 else 0.0
                quality_flags = _quality_flags(
                    segment,
                    coverage_ratio,
                    segment_time_sync_status,
                    parent_rd.rd_is_multimodal,
                    rd_time_coverage_ratio,
                )
                segment_id = f"{parent_sample_id}|seg{segment_index:04d}"
                payload = {
                    "segment_id": segment_id,
                    "segment_index": int(segment_index),
                    "segment_start_seconds": float(segment_start),
                    "segment_end_seconds": float(segment_end),
                    "segment_duration_seconds": float(duration),
                    "rd_frame_start_index": int(frame_start_index),
                    "rd_frame_end_index": int(frame_end_index),
                    "rd_time_start_seconds": rd_time_start_seconds,
                    "rd_time_end_seconds": rd_time_end_seconds,
                    "rd_features": segment.features.to(torch.float32).cpu(),
                    "rd_time_seconds": (
                        segment.time_seconds.to(torch.float32).cpu()
                        if segment.time_seconds is not None
                        else torch.arange(segment.valid_length, dtype=torch.float32)
                    ),
                    "rd_valid_length": int(segment.valid_length),
                    "rd_patch_features": (
                        segment.patch_features.to(torch.float32).cpu()
                        if segment.patch_features is not None
                        else torch.zeros((1, MULTIMODAL_RD_PATCH_RANGE_BINS, MULTIMODAL_RD_PATCH_DOPPLER_BINS, 3), dtype=torch.float32)
                    ),
                    "rd_patch_valid_length": int(segment.patch_valid_length or 1),
                    "rd_patch_time_seconds": (
                        segment.patch_time_seconds.to(torch.float32).cpu()
                        if segment.patch_time_seconds is not None
                        else torch.arange(segment.patch_valid_length or 1, dtype=torch.float32)
                    ),
                    "rd_deep_patch_features": (
                        segment.deep_patch_features.to(torch.float32).cpu()
                        if segment.deep_patch_features is not None
                        else torch.zeros((1, MULTIMODAL_RD_PATCH_RANGE_BINS, MULTIMODAL_RD_PATCH_DOPPLER_BINS, 3), dtype=torch.float32)
                    ),
                    "rd_deep_patch_valid_length": int(segment.deep_patch_valid_length or 1),
                    "rd_deep_patch_time_seconds": (
                        segment.deep_patch_time_seconds.to(torch.float32).cpu()
                        if segment.deep_patch_time_seconds is not None
                        else torch.arange(segment.deep_patch_valid_length or 1, dtype=torch.float32)
                    ),
                    "rd_coverage_ratio": float(coverage_ratio),
                    "rd_time_coverage_ratio": float(rd_time_coverage_ratio),
                    "rd_quality_flags": quality_flags,
                    "rd_time_sync_status": segment_time_sync_status,
                }
                segment_payloads.append(payload)
                relative_path = parent_path.relative_to(project_root).as_posix()
                manifest_rows.append(
                    {
                        "segment_id": segment_id,
                        "parent_sample_id": parent_sample_id,
                        "track_id": parent_sample_id,
                        "source_name": source_name,
                        "label_name": label_name,
                        "segment_start_seconds": float(segment_start),
                        "segment_end_seconds": float(segment_end),
                        "segment_duration_seconds": float(duration),
                        "segment_frame_count": int(segment.valid_length),
                        "rd_feature_path": relative_path,
                        "rd_patch_path": relative_path,
                        "rd_deep_patch_path": relative_path,
                        "shard_key": relative_path,
                        "segment_index": int(segment_index),
                        "rd_feature_key": f"segments.{segment_index}.rd_features",
                        "rd_patch_key": f"segments.{segment_index}.rd_patch_features",
                        "rd_deep_patch_key": f"segments.{segment_index}.rd_deep_patch_features",
                        "rd_coverage_ratio": float(coverage_ratio),
                        "rd_frame_start_index": int(frame_start_index),
                        "rd_frame_end_index": int(frame_end_index),
                        "rd_time_start_seconds": rd_time_start_seconds,
                        "rd_time_end_seconds": rd_time_end_seconds,
                        "rd_time_coverage_ratio": float(rd_time_coverage_ratio),
                        "rd_quality_flags": quality_flags,
                        "rd_time_sync_status": segment_time_sync_status,
                        "rd_preprocess_version": PREPROCESS_VERSION,
                        "rd_is_multimodal": bool(parent_rd.rd_is_multimodal),
                    }
                )
                status_counts[segment_time_sync_status] += 1
                for flag in quality_flags.split(";"):
                    quality_counts[flag] += 1
            torch.save(
                {
                    "schema_version": 2,
                    "rd_preprocess_version": PREPROCESS_VERSION,
                    "segment_seconds": float(segment_seconds),
                    "deep_patch_mode": deep_patch_mode,
                    "parent_sample_id": parent_sample_id,
                    "track_id": parent_sample_id,
                    "source_name": source_name,
                    "label_name": label_name,
                    "parent_start_seconds": float(parent_start),
                    "parent_end_seconds": float(parent_end),
                    "rd_is_multimodal": bool(parent_rd.rd_is_multimodal),
                    "rd_time_sync_status": parent_rd.time_sync_status,
                    "segments": segment_payloads,
                },
                parent_path,
            )
            parent_count += 1
            source_counts[source_name] += 1
            label_counts[label_name] += 1
        except Exception as exc:  # pragma: no cover - defensive for batch conversion.
            errors.append({"parent_sample_id": parent_sample_id, "error": repr(exc)})

    manifest_df = pd.DataFrame(manifest_rows, columns=MANIFEST_COLUMNS)
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
    summary: dict[str, object] = {
        "index_name": index_name,
        "output_name": output_name,
        "segment_seconds": float(segment_seconds),
        "min_frames_per_segment": int(min_frames_per_segment),
        "deep_patch_mode": deep_patch_mode,
        "rd_preprocess_version": PREPROCESS_VERSION,
        "parent_count": int(parent_count),
        "segment_count": int(len(manifest_rows)),
        "candidate_parent_count": int(len(parent_rows)),
        "failed_parent_count": int(len(errors)),
        "source_parent_counts": dict(source_counts),
        "label_parent_counts": dict(label_counts),
        "rd_time_sync_status_counts": dict(status_counts),
        "rd_quality_flag_counts": dict(quality_counts),
        "manifest_path": manifest_path.relative_to(project_root).as_posix(),
        "store_dir": output_root.relative_to(project_root).as_posix(),
        "errors": errors[:50],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build reusable 2s RD segment store for Expert1.")
    parser.add_argument("--index-name", default="expert1_track_index.csv")
    parser.add_argument("--output-name", default="expert1_rd_segment_store")
    parser.add_argument("--segment-seconds", type=float, default=2.0)
    parser.add_argument("--min-frames-per-segment", type=int, default=1)
    parser.add_argument("--deep-patch-mode", choices=["full", "patch", "none"], default="full")
    parser.add_argument("--max-parents", type=int, default=0)
    parser.add_argument("--source-name", action="append", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    project_root = Path(__file__).resolve().parents[2]
    summary = build_expert1_rd_segment_store(
        project_root,
        index_name=args.index_name,
        output_name=args.output_name,
        segment_seconds=args.segment_seconds,
        min_frames_per_segment=args.min_frames_per_segment,
        deep_patch_mode=args.deep_patch_mode,
        max_parents=args.max_parents,
        source_names=args.source_name,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
