from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..rd_physical_schema import PHYSICAL_RD_NORMALIZATION_SCHEMA_VERSION
from .rd_physical_window_dataset import (
    PHYSICAL_RD_DOPPLER_BINS,
    PHYSICAL_RD_FEATURE_NAMES,
    PHYSICAL_RD_RANGE_BINS,
    PhysicalRDWindowBatch,
    PhysicalRDWindowItem,
    _as_bool,
    _physical_rd_features,
    _resolve_data_path,
    _sha256_file,
    _parse_frame_indices,
)


RD_FIVE_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
RD_FIVE_QUALITY_NAMES = (
    "rd_present",
    "phase_complete",
    "multimodal",
    "coverage_ratio",
    "log_frame_count",
    "log_duration",
    "dynamic_range_proxy",
)
RD_FIVE_NORMALIZATION_SCHEMA_VERSION = "rd_physical_train_normalization_v4_five"


def encode_rd_five_label(label_name: str) -> int:
    try:
        return RD_FIVE_LABELS.index(str(label_name))
    except ValueError as error:
        raise ValueError(f"five-class RD dataset requires one of {RD_FIVE_LABELS}, got {label_name!r}") from error


def _load_normalization(path: Path, *, index_name: str, index_sha256: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RD_FIVE_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError(f"unexpected five-class RD normalization schema: {path}")
    if payload.get("index_name") != index_name or payload.get("index_sha256") != index_sha256:
        raise ValueError("five-class RD normalization provenance does not match the active window index")
    amplitude = payload.get("amplitude_db")
    delta = payload.get("delta_rate")
    if not isinstance(amplitude, dict) or not isinstance(delta, dict):
        raise ValueError("five-class RD normalization is missing amplitude_db or delta_rate")
    values = {name: float(amplitude[name]) for name in ("mean", "std", "clip_low", "clip_high")}
    values.update({f"delta_{name}": float(delta[name]) for name in ("time_floor_seconds", "time_max_seconds", "center", "scale")})
    if values["std"] <= 1.0e-6 or values["clip_high"] <= values["clip_low"] or values["delta_scale"] <= 1.0e-6:
        raise ValueError("five-class RD normalization values are invalid")
    return values


class PhysicalRDFiveWindowDataset(Dataset[PhysicalRDWindowItem]):
    """Independent five-class dataset; it never calls the binary label encoder."""

    def __init__(
        self,
        data_root: str | Path,
        *,
        index_name: str,
        split_name: str | None = None,
        trainable_only: bool = True,
        tensor_cache_size: int = 8,
        normalization_stats_name: str,
        include_missing_rd: bool = False,
        present_only: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.processed_root = self.data_root / "processed"
        self.index_path = self.processed_root / index_name
        if not self.index_path.exists():
            raise FileNotFoundError(f"five-class RD window index not found: {self.index_path}")
        frame = pd.read_csv(self.index_path, low_memory=False)
        required = {
            "sample_id", "parent_sample_id", "parent_split_group_id", "label_name", "split_name",
            "rd_tensor_path", "rd_tensor_frame_indices_json", "rd_time_start_seconds", "rd_time_end_seconds",
            "is_trainable", "rd_time_semantics", "rd_fusion_eligible", "rd_present", "rd_is_multimodal",
            "rd_phase_missing", "rd_coverage_ratio", "rd_time_sync_status", "rd_window_mode",
            "is_prefix_window", "rd_window_frame_count",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"five-class RD window index is missing columns: {missing}")
        if split_name is not None:
            frame = frame[frame["split_name"].astype(str).eq(str(split_name))]
        frame = frame[frame["label_name"].astype(str).isin(set(RD_FIVE_LABELS))]
        if trainable_only:
            frame = frame[frame["is_trainable"].map(_as_bool)]
        if present_only:
            frame = frame[frame["rd_present"].map(_as_bool)]
        if frame.empty:
            raise ValueError("five-class RD dataset selected no rows")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("five-class RD index contains duplicate sample IDs")
        groups = frame["parent_split_group_id"].astype(str).str.strip()
        if groups.eq("").any() or groups.str.lower().eq("nan").any():
            raise ValueError("five-class RD index has missing content groups")
        if int(frame.assign(_group=groups).groupby("_group")["label_name"].nunique().max()) > 1:
            raise ValueError("five-class RD content group has inconsistent labels")
        if frame["rd_fusion_eligible"].map(_as_bool).any():
            raise ValueError("five-class RD dataset refuses fusion-eligible rows")
        present = frame["rd_present"].map(_as_bool)
        complete = present & frame["rd_is_multimodal"].map(_as_bool) & ~frame["rd_phase_missing"].map(_as_bool) & frame["rd_coverage_ratio"].astype(float).ge(1.0 - 1.0e-6) & frame["rd_time_sync_status"].astype(str).eq("canonical_parent_association")
        if (present & ~complete).any():
            raise ValueError("five-class RD dataset requires complete canonical RD rows")
        if (~present).any() and not include_missing_rd:
            raise ValueError("five-class RD dataset contains missing RD rows")
        if (frame.loc[present, "rd_time_semantics"].astype(str) != "canonical_sensor_relative_elapsed").any():
            raise ValueError("five-class present RD rows have invalid time semantics")
        self.frame = frame.sort_values(["parent_sample_id", "rd_time_end_seconds", "sample_id"], kind="stable").reset_index(drop=True)
        self.normalization_stats = _load_normalization(self.processed_root / normalization_stats_name, index_name=index_name, index_sha256=_sha256_file(self.index_path))
        self.tensor_cache_size = max(1, int(tensor_cache_size))
        self.tensor_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return int(len(self.frame))

    def _load_tensor(self, raw_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cached = self.tensor_cache.get(raw_path)
        if cached is not None:
            self.tensor_cache.move_to_end(raw_path)
            return cached
        path = _resolve_data_path(self.data_root, raw_path)
        if not path.exists():
            raise FileNotFoundError(f"five-class RD tensor not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            amplitude = np.asarray(payload["amplitude"], dtype=np.float32)
            phase = np.asarray(payload["phase"], dtype=np.float32)
            times = np.asarray(payload["time_seconds"], dtype=np.float32)
            ranges = np.asarray(payload["range_offset_m"], dtype=np.float32)
            velocities = np.asarray(payload["velocity_mps"], dtype=np.float32)
        if amplitude.ndim != 3 or phase.shape != amplitude.shape or tuple(amplitude.shape[1:]) != (PHYSICAL_RD_RANGE_BINS, PHYSICAL_RD_DOPPLER_BINS):
            raise ValueError(f"invalid five-class RD tensor shape: {path}")
        if not np.isfinite(amplitude).all() or not np.isfinite(phase).all() or not np.isfinite(times).all() or (np.diff(times) <= 1.0e-9).any():
            raise ValueError(f"invalid five-class RD tensor values/time: {path}")
        if ranges.shape != (PHYSICAL_RD_RANGE_BINS,) or velocities.shape != (PHYSICAL_RD_DOPPLER_BINS,) or not np.isfinite(ranges).all() or not np.isfinite(velocities).all():
            raise ValueError(f"invalid five-class RD tensor axes: {path}")
        value = (amplitude, phase, times, ranges, velocities)
        self.tensor_cache[raw_path] = value
        while len(self.tensor_cache) > self.tensor_cache_size:
            self.tensor_cache.popitem(last=False)
        return value

    def __getitem__(self, item_index: int) -> PhysicalRDWindowItem:
        row = self.frame.iloc[int(item_index)]
        sample_id = str(row["sample_id"])
        is_present = _as_bool(row["rd_present"])
        indices = _parse_frame_indices(row["rd_tensor_frame_indices_json"], sample_id=sample_id, allow_empty=not is_present)
        if int(row["rd_window_frame_count"]) != int(indices.size):
            raise ValueError(f"five-class RD frame count mismatch: {sample_id}")
        if is_present:
            amplitude, phase, times, ranges, velocities = self._load_tensor(str(row["rd_tensor_path"]))
        else:
            amplitude = np.zeros((0, PHYSICAL_RD_RANGE_BINS, PHYSICAL_RD_DOPPLER_BINS), dtype=np.float32)
            phase = np.zeros_like(amplitude)
            times = np.zeros((0,), dtype=np.float32)
            ranges = np.linspace(-75.0, 75.0, PHYSICAL_RD_RANGE_BINS, dtype=np.float32)
            velocities = np.linspace(-85.0, 85.0, PHYSICAL_RD_DOPPLER_BINS, dtype=np.float32)
        if indices.size and int(indices[-1]) >= int(amplitude.shape[0]):
            raise ValueError(f"five-class RD frame index exceeds tensor bounds: {sample_id}")
        selected_times = times[indices]
        if is_present:
            if max(abs(float(selected_times[0]) - float(row["rd_time_start_seconds"])), abs(float(selected_times[-1]) - float(row["rd_time_end_seconds"]))) > 5.0e-7:
                raise ValueError(f"five-class RD index/tensor time mismatch: {sample_id}")
        features = _physical_rd_features(amplitude[indices], phase[indices], selected_times, ranges, velocities, self.normalization_stats)
        finite = amplitude[indices][np.isfinite(amplitude[indices])]
        dynamic_range = float(finite.max() - finite.min()) if finite.size else 0.0
        duration = float(selected_times[-1] - selected_times[0]) if selected_times.size else 0.0
        quality = torch.tensor([
            float(is_present), float(not _as_bool(row["rd_phase_missing"])), float(_as_bool(row["rd_is_multimodal"])),
            float(np.clip(float(row["rd_coverage_ratio"]), 0.0, 1.0)),
            float(np.log1p(max(0, int(amplitude[indices].shape[0]))) / np.log1p(128.0)),
            float(np.log1p(max(0.0, duration)) / np.log1p(0.5)), float(np.clip(dynamic_range / 40.0, 0.0, 1.0)),
        ], dtype=torch.float32)
        return PhysicalRDWindowItem(
            sample_id=sample_id, parent_sample_id=str(row["parent_sample_id"]), label_name=str(row["label_name"]), split_name=str(row["split_name"]),
            window_mode=str(row["rd_window_mode"]), is_prefix_window=_as_bool(row["is_prefix_window"]), window_frame_count=int(row["rd_window_frame_count"]),
            features=features, raw_amplitude_db=torch.from_numpy(np.ascontiguousarray(np.nan_to_num(amplitude[indices], nan=0.0, posinf=0.0, neginf=0.0))).float(),
            time_seconds=torch.from_numpy(np.ascontiguousarray(selected_times)).float(), quality_features=quality, quality_metadata=quality,
            range_offset_m=torch.from_numpy(np.ascontiguousarray(ranges)).float(), velocity_mps=torch.from_numpy(np.ascontiguousarray(velocities)).float(), label=encode_rd_five_label(str(row["label_name"])),
        )


def collate_physical_rd_five_windows(items: list[PhysicalRDWindowItem]) -> PhysicalRDWindowBatch:
    if not items:
        raise ValueError("cannot collate an empty five-class RD batch")
    feature_shape = tuple(items[0].features.shape[1:])
    max_length = max(1, max(int(item.features.shape[0]) for item in items))
    batch_size = len(items)
    features = torch.zeros((batch_size, max_length, *feature_shape), dtype=torch.float32)
    raw_db = torch.zeros((batch_size, max_length, feature_shape[1], feature_shape[2]), dtype=torch.float32)
    times = torch.zeros((batch_size, max_length), dtype=torch.float32)
    padding = torch.ones((batch_size, max_length), dtype=torch.bool)
    lengths = torch.empty((batch_size,), dtype=torch.long)
    quality = torch.empty((batch_size, len(RD_FIVE_QUALITY_NAMES)), dtype=torch.float32)
    labels = torch.empty((batch_size,), dtype=torch.long)
    prefixes = torch.empty((batch_size,), dtype=torch.bool)
    counts = torch.empty((batch_size,), dtype=torch.long)
    ranges = items[0].range_offset_m.clone()
    velocities = items[0].velocity_mps.clone()
    for index, item in enumerate(items):
        length = int(item.features.shape[0])
        if tuple(item.features.shape[1:]) != feature_shape or item.quality_features.shape != (len(RD_FIVE_QUALITY_NAMES),) or not torch.equal(item.range_offset_m, ranges) or not torch.equal(item.velocity_mps, velocities):
            raise ValueError("five-class RD batch contains incompatible item shapes")
        features[index, :length] = item.features
        raw_db[index, :length] = item.raw_amplitude_db
        times[index, :length] = item.time_seconds
        padding[index, :length] = False
        lengths[index] = length
        quality[index] = item.quality_features
        labels[index] = item.label
        prefixes[index] = item.is_prefix_window
        counts[index] = item.window_frame_count
    return PhysicalRDWindowBatch(
        sample_ids=[item.sample_id for item in items], parent_sample_ids=[item.parent_sample_id for item in items], label_names=[item.label_name for item in items], split_names=[item.split_name for item in items],
        window_modes=[item.window_mode for item in items], is_prefix_windows=prefixes, window_frame_counts=counts, features=features, raw_amplitude_db=raw_db, time_seconds=times, padding_mask=padding, lengths=lengths,
        quality_features=quality, range_offset_m=ranges, velocity_mps=velocities, labels=labels, quality_metadata=quality,
    )


__all__ = ["RD_FIVE_LABELS", "RD_FIVE_QUALITY_NAMES", "PhysicalRDFiveWindowDataset", "collate_physical_rd_five_windows", "encode_rd_five_label"]
