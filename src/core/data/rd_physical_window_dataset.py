from __future__ import annotations

import json
import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..rd_physical_schema import PHYSICAL_RD_NORMALIZATION_SCHEMA_VERSION


RD_ONLY_LABELS = ("DroneTarget", "BirdTarget")
PHYSICAL_RD_FEATURE_NAMES = (
    "normalized_db_residual",
    "phase_sin",
    "phase_cos",
    "bounded_temporal_delta",
    "range_coordinate",
    "velocity_coordinate",
)
QUALITY_METADATA_NAMES = (
    "rd_present",
    "phase_complete",
    "multimodal",
    "coverage_ratio",
    "log_frame_count",
    "log_duration",
    "dynamic_range_proxy",
    "canonical_time",
)
NORMALIZATION_SCHEMA_VERSION = PHYSICAL_RD_NORMALIZATION_SCHEMA_VERSION
QUALITY_FEATURE_NAMES: tuple[str, ...] = QUALITY_METADATA_NAMES
QUALITY_FEATURE_DIM = len(QUALITY_FEATURE_NAMES)
QUALITY_METADATA_DIM = len(QUALITY_METADATA_NAMES)
RD_ONLY_QUALITY_NAMES: tuple[str, ...] = QUALITY_METADATA_NAMES[:-1]
RD_ONLY_QUALITY_SCHEMA = "rd_only_v5"
LEGACY_QUALITY_SCHEMA = "legacy_v4"
PHYSICAL_RD_INPUT_CHANNELS = len(PHYSICAL_RD_FEATURE_NAMES)
PHYSICAL_RD_RANGE_BINS = 21
PHYSICAL_RD_DOPPLER_BINS = 384
REQUIRED_WINDOW_COLUMNS = {
    "sample_id",
    "parent_sample_id",
    "parent_split_group_id",
    "label_name",
    "split_name",
    "rd_tensor_path",
    "rd_tensor_frame_indices_json",
    "rd_time_start_seconds",
    "rd_time_end_seconds",
    "is_trainable",
    "rd_time_semantics",
    "rd_fusion_eligible",
    "rd_present",
    "rd_is_multimodal",
    "rd_phase_missing",
    "rd_coverage_ratio",
    "rd_time_sync_status",
    "rd_window_mode",
    "is_prefix_window",
    "rd_window_frame_count",
}


def encode_rd_drone_bird_label(label_name: str) -> int:
    try:
        return RD_ONLY_LABELS.index(str(label_name))
    except ValueError as error:
        raise ValueError(f"RD-only physical classifier requires one of {RD_ONLY_LABELS}, got {label_name!r}") from error


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def _resolve_data_path(data_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else data_root.parent / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_normalization_stats(path: Path, *, index_name: str, index_sha256: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != NORMALIZATION_SCHEMA_VERSION:
        raise ValueError(f"unexpected physical RD normalization schema: {path}")
    if payload.get("index_name") != index_name or payload.get("index_sha256") != index_sha256:
        raise ValueError("physical RD normalization provenance does not match the active window index")
    amplitude = payload.get("amplitude_db")
    if not isinstance(amplitude, dict):
        raise ValueError("physical RD normalization is missing amplitude_db")
    values = {name: float(amplitude[name]) for name in ("mean", "std", "clip_low", "clip_high")}
    if values["std"] <= 1.0e-6 or values["clip_high"] <= values["clip_low"]:
        raise ValueError("physical RD normalization amplitude parameters are invalid")
    delta = payload.get("delta_rate")
    if not isinstance(delta, dict):
        raise ValueError("physical RD normalization is missing delta_rate")
    for name in ("time_floor_seconds", "time_max_seconds", "center", "scale"):
        values[f"delta_{name}"] = float(delta[name])
    if values["delta_time_floor_seconds"] <= 0.0 or values["delta_time_max_seconds"] < values["delta_time_floor_seconds"]:
        raise ValueError("physical RD delta time bounds are invalid")
    if values["delta_scale"] <= 1.0e-6:
        raise ValueError("physical RD delta rate scale is invalid")
    return values


def _parse_frame_indices(value: object, *, sample_id: str, allow_empty: bool = False) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError(f"RD-only frame indices are not JSON text: {sample_id}")
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid RD-only frame indices JSON: {sample_id}") from error
    if not isinstance(parsed, list) or (not parsed and not allow_empty):
        raise ValueError(f"RD-only frame indices must be a non-empty list: {sample_id}")
    indices = np.asarray(parsed, dtype=np.int64)
    if (indices < 0).any() or np.unique(indices).size != indices.size or (np.diff(indices) < 0).any():
        raise ValueError(f"RD-only frame indices are invalid: {sample_id}")
    return indices


def _physical_rd_features(
    amplitude: np.ndarray,
    phase: np.ndarray,
    time_seconds: np.ndarray,
    range_offset_m: np.ndarray,
    velocity_mps: np.ndarray,
    normalization_stats: dict[str, float],
) -> torch.Tensor:
    if amplitude.ndim != 3 or phase.shape != amplitude.shape:
        raise ValueError("physical RD requires matching [time, range, doppler] amplitude and phase")
    if time_seconds.ndim != 1 or int(time_seconds.size) != int(amplitude.shape[0]):
        raise ValueError("physical RD time axis must match frame count")
    if not np.isfinite(time_seconds).all() or (np.diff(time_seconds) < -1.0e-9).any():
        raise ValueError("physical RD time axis must be finite and monotonic")
    if range_offset_m.ndim != 1 or int(range_offset_m.size) != int(amplitude.shape[1]):
        raise ValueError("physical RD range axis must match range dimension")
    if velocity_mps.ndim != 1 or int(velocity_mps.size) != int(amplitude.shape[2]):
        raise ValueError("physical RD velocity axis must match doppler dimension")
    if not np.isfinite(range_offset_m).all() or not np.isfinite(velocity_mps).all():
        raise ValueError("physical RD axes must be finite")
    amplitude_t = torch.from_numpy(np.ascontiguousarray(np.nan_to_num(amplitude, nan=0.0, posinf=0.0, neginf=0.0))).to(torch.float32)
    phase_t = torch.from_numpy(np.ascontiguousarray(np.nan_to_num(phase, nan=0.0, posinf=0.0, neginf=0.0))).to(torch.float32)
    time_t = torch.from_numpy(np.ascontiguousarray(time_seconds)).to(torch.float32)
    normalized_residual = (
        amplitude_t.clamp(
            min=float(normalization_stats["clip_low"]),
            max=float(normalization_stats["clip_high"]),
        )
        - float(normalization_stats["mean"])
    ) / float(normalization_stats["std"])
    delta_time_floor = float(normalization_stats["delta_time_floor_seconds"])
    delta_time_max = float(normalization_stats["delta_time_max_seconds"])
    delta_center = float(normalization_stats["delta_center"])
    delta_scale = float(normalization_stats["delta_scale"])
    delta = torch.zeros_like(normalized_residual)
    if amplitude_t.shape[0] > 1:
        intervals = time_t[1:] - time_t[:-1]
        valid_intervals = (intervals >= delta_time_floor) & (intervals <= delta_time_max)
        safe_intervals = intervals.clamp_min(delta_time_floor)
        raw_delta = (normalized_residual[1:] - normalized_residual[:-1]) / safe_intervals.view(-1, 1, 1)
        bounded_delta = torch.tanh((raw_delta - delta_center) / delta_scale)
        delta[1:] = torch.where(valid_intervals.view(-1, 1, 1), bounded_delta, torch.zeros_like(bounded_delta))
    range_axis = torch.from_numpy(np.ascontiguousarray(range_offset_m)).to(torch.float32)
    velocity_axis = torch.from_numpy(np.ascontiguousarray(velocity_mps)).to(torch.float32)
    range_coordinate = range_axis / range_axis.abs().max().clamp_min(1.0)
    velocity_coordinate = velocity_axis / velocity_axis.abs().max().clamp_min(1.0)
    range_plane = range_coordinate.view(1, -1, 1).expand(amplitude_t.shape[0], -1, amplitude_t.shape[2])
    velocity_plane = velocity_coordinate.view(1, 1, -1).expand(amplitude_t.shape[0], amplitude_t.shape[1], -1)
    channels = [normalized_residual, torch.sin(phase_t), torch.cos(phase_t), delta, range_plane, velocity_plane]
    if len(channels) != PHYSICAL_RD_INPUT_CHANNELS:
        raise AssertionError("physical RD feature schema channel count drifted")
    return torch.stack(channels, dim=1)


def _quality_metadata(
    row: pd.Series,
    amplitude: np.ndarray,
    time_seconds: np.ndarray,
    *,
    quality_schema: str = LEGACY_QUALITY_SCHEMA,
) -> torch.Tensor:
    finite = amplitude[np.isfinite(amplitude)]
    dynamic_range = float(finite.max() - finite.min()) if finite.size else 0.0
    duration = float(time_seconds[-1] - time_seconds[0]) if time_seconds.size else 0.0
    values = [
        float(_as_bool(row["rd_present"])),
        float(not _as_bool(row["rd_phase_missing"])),
        float(_as_bool(row["rd_is_multimodal"])),
        float(np.clip(float(row["rd_coverage_ratio"]), 0.0, 1.0)),
        float(np.log1p(max(0, int(amplitude.shape[0]))) / np.log1p(128.0)),
        float(np.log1p(max(0.0, duration)) / np.log1p(0.5)),
        float(np.clip(dynamic_range / 40.0, 0.0, 1.0)),
    ]
    if quality_schema == LEGACY_QUALITY_SCHEMA:
        # Explicitly retained for read-only loading of historical v4 artifacts.
        values.append(float(str(row["rd_time_sync_status"]) == "canonical_parent_association"))
    elif quality_schema != RD_ONLY_QUALITY_SCHEMA:
        raise ValueError(f"unknown physical RD quality schema: {quality_schema!r}")
    return torch.tensor(values, dtype=torch.float32)


@dataclass(frozen=True)
class PhysicalRDWindowItem:
    sample_id: str
    parent_sample_id: str
    label_name: str
    split_name: str
    window_mode: str
    is_prefix_window: bool
    window_frame_count: int
    features: torch.Tensor
    raw_amplitude_db: torch.Tensor
    time_seconds: torch.Tensor
    quality_features: torch.Tensor
    quality_metadata: torch.Tensor
    range_offset_m: torch.Tensor
    velocity_mps: torch.Tensor
    label: int


@dataclass(frozen=True)
class PhysicalRDWindowBatch:
    sample_ids: list[str]
    parent_sample_ids: list[str]
    label_names: list[str]
    split_names: list[str]
    window_modes: list[str]
    is_prefix_windows: torch.Tensor
    window_frame_counts: torch.Tensor
    features: torch.Tensor
    time_seconds: torch.Tensor
    padding_mask: torch.Tensor
    lengths: torch.Tensor
    quality_features: torch.Tensor
    range_offset_m: torch.Tensor
    velocity_mps: torch.Tensor
    labels: torch.Tensor
    raw_amplitude_db: torch.Tensor = field(default_factory=lambda: torch.empty((0, 0, 0, 0), dtype=torch.float32))
    quality_metadata: torch.Tensor = field(default_factory=lambda: torch.empty((0, QUALITY_METADATA_DIM), dtype=torch.float32))


class PhysicalRDWindowDataset(Dataset[PhysicalRDWindowItem]):
    def __init__(
        self,
        data_root: str | Path,
        *,
        index_name: str,
        split_name: str | None = None,
        labels: tuple[str, ...] = RD_ONLY_LABELS,
        trainable_only: bool = True,
        tensor_cache_size: int = 8,
        normalization_stats_name: str | None = None,
        include_missing_rd: bool = False,
        present_only: bool = False,
        quality_schema: str = LEGACY_QUALITY_SCHEMA,
    ) -> None:
        self.data_root = Path(data_root)
        if quality_schema not in {LEGACY_QUALITY_SCHEMA, RD_ONLY_QUALITY_SCHEMA}:
            raise ValueError(f"unknown physical RD quality schema: {quality_schema!r}")
        self.quality_schema = quality_schema
        self.quality_names = QUALITY_FEATURE_NAMES if quality_schema == LEGACY_QUALITY_SCHEMA else RD_ONLY_QUALITY_NAMES
        self.processed_root = self.data_root / "processed"
        self.index_path = self.processed_root / index_name
        if not self.index_path.exists():
            raise FileNotFoundError(f"RD-only physical window index not found: {self.index_path}")
        frame = pd.read_csv(self.index_path, low_memory=False)
        missing = sorted(REQUIRED_WINDOW_COLUMNS - set(frame.columns))
        if missing:
            raise ValueError(f"RD-only physical window index is missing columns: {missing}")
        if split_name is not None:
            frame = frame[frame["split_name"].astype(str).eq(str(split_name))]
        if labels:
            frame = frame[frame["label_name"].astype(str).isin(set(labels))]
        if trainable_only:
            frame = frame[frame["is_trainable"].map(_as_bool)]
        if present_only:
            frame = frame[frame["rd_present"].map(_as_bool)]
        if frame.empty:
            raise ValueError("RD-only physical window dataset selected no rows")
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("RD-only physical window index contains duplicate sample IDs")
        group_values = frame["parent_split_group_id"].astype(str).str.strip()
        if group_values.eq("").any() or group_values.str.lower().eq("nan").any():
            raise ValueError("RD-only physical window index has missing content groups")
        group_label_counts = frame.assign(_group=group_values).groupby("_group")["label_name"].nunique()
        if int(group_label_counts.max()) > 1:
            raise ValueError("RD-only physical content group has inconsistent labels")
        if frame["rd_fusion_eligible"].map(_as_bool).any():
            raise ValueError("RD-only physical window dataset refuses fusion-eligible rows")
        complete_rd = (
            frame["rd_present"].map(_as_bool)
            & frame["rd_is_multimodal"].map(_as_bool)
            & ~frame["rd_phase_missing"].map(_as_bool)
            & frame["rd_coverage_ratio"].astype(float).ge(1.0 - 1.0e-6)
            & frame["rd_time_sync_status"].astype(str).eq("canonical_parent_association")
        )
        present = frame["rd_present"].map(_as_bool)
        if (present & ~complete_rd).any():
            raise ValueError(
                "active physical RD dataset requires complete canonical RD rows; "
                "partial present RD must use a separately designed preprocessing path"
            )
        missing = ~present
        if missing.any() and not include_missing_rd:
            raise ValueError(
                "active physical RD dataset contains missing RD rows; "
                "set include_missing_rd=True to use the explicit all-padding missing-modality path"
            )
        if missing.any():
            missing_frame_counts = pd.to_numeric(frame.loc[missing, "rd_window_frame_count"], errors="coerce")
            if missing_frame_counts.isna().any() or (missing_frame_counts != 0).any():
                raise ValueError("missing RD rows must declare rd_window_frame_count=0")
            missing_coverage = pd.to_numeric(frame.loc[missing, "rd_coverage_ratio"], errors="coerce")
            if missing_coverage.isna().any() or (missing_coverage != 0.0).any():
                raise ValueError("missing RD rows must declare rd_coverage_ratio=0")
            if frame.loc[missing, "rd_is_multimodal"].map(_as_bool).any():
                raise ValueError("missing RD rows cannot be marked multimodal")
            indices_values = frame.loc[missing, "rd_tensor_frame_indices_json"]
            for sample_id, value in zip(frame.loc[missing, "sample_id"].astype(str), indices_values):
                _parse_frame_indices(value, sample_id=sample_id, allow_empty=True)
        present_semantics = frame.loc[frame["rd_present"].map(_as_bool), "rd_time_semantics"].astype(str)
        if (present_semantics != "canonical_sensor_relative_elapsed").any():
            raise ValueError("present RD rows must use canonical_sensor_relative_elapsed semantics")
        missing_semantics = frame.loc[~frame["rd_present"].map(_as_bool), "rd_time_semantics"].astype(str)
        if (missing_semantics != "unavailable").any():
            raise ValueError("missing RD rows must use unavailable time semantics")
        for label_name in frame["label_name"].astype(str).unique().tolist():
            encode_rd_drone_bird_label(label_name)
        self.frame = frame.sort_values(["parent_sample_id", "rd_time_end_seconds", "sample_id"], kind="stable").reset_index(drop=True)
        if not normalization_stats_name:
            raise ValueError("active physical RD windows require a train-only normalization artifact")
        self.normalization_stats = _load_normalization_stats(
            self.processed_root / normalization_stats_name,
            index_name=index_name,
            index_sha256=_sha256_file(self.index_path),
        )
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
            raise FileNotFoundError(f"physical RD tensor not found: {path}")
        with np.load(path, allow_pickle=False) as payload:
            amplitude = np.asarray(payload["amplitude"], dtype=np.float32)
            phase = np.asarray(payload["phase"], dtype=np.float32)
            time_seconds = np.asarray(payload["time_seconds"], dtype=np.float32)
            range_offset_m = np.asarray(payload["range_offset_m"], dtype=np.float32)
            velocity_mps = np.asarray(payload["velocity_mps"], dtype=np.float32)
        if amplitude.ndim != 3 or phase.shape != amplitude.shape:
            raise ValueError(f"invalid physical RD tensor shape: {path}")
        if tuple(amplitude.shape[1:]) != (PHYSICAL_RD_RANGE_BINS, PHYSICAL_RD_DOPPLER_BINS):
            raise ValueError(
                f"physical RD tensor must preserve canonical shape "
                f"[{PHYSICAL_RD_RANGE_BINS},{PHYSICAL_RD_DOPPLER_BINS}]: {path}"
            )
        if time_seconds.ndim != 1 or int(time_seconds.size) != int(amplitude.shape[0]):
            raise ValueError(f"invalid physical RD tensor time axis: {path}")
        if not np.isfinite(amplitude).all() or not np.isfinite(phase).all():
            raise ValueError(f"physical RD tensor contains non-finite amplitude or phase: {path}")
        if not np.isfinite(time_seconds).all() or (np.diff(time_seconds) <= 1.0e-9).any():
            raise ValueError(f"physical RD tensor time axis is not strictly increasing: {path}")
        if range_offset_m.ndim != 1 or int(range_offset_m.size) != int(amplitude.shape[1]):
            raise ValueError(f"invalid physical RD tensor range axis: {path}")
        if velocity_mps.ndim != 1 or int(velocity_mps.size) != int(amplitude.shape[2]):
            raise ValueError(f"invalid physical RD tensor velocity axis: {path}")
        if not np.isfinite(range_offset_m).all() or not np.isfinite(velocity_mps).all():
            raise ValueError(f"physical RD tensor axes are not finite: {path}")
        self.tensor_cache[raw_path] = (amplitude, phase, time_seconds, range_offset_m, velocity_mps)
        while len(self.tensor_cache) > self.tensor_cache_size:
            self.tensor_cache.popitem(last=False)
        return amplitude, phase, time_seconds, range_offset_m, velocity_mps

    def __getitem__(self, item_index: int) -> PhysicalRDWindowItem:
        row = self.frame.iloc[int(item_index)]
        sample_id = str(row["sample_id"])
        is_present = _as_bool(row["rd_present"])
        indices = _parse_frame_indices(
            row["rd_tensor_frame_indices_json"],
            sample_id=sample_id,
            allow_empty=not is_present,
        )
        declared_frame_count = int(row["rd_window_frame_count"])
        if declared_frame_count != int(indices.size):
            raise ValueError(
                f"RD-only index frame count mismatch: {sample_id}; "
                f"declared={declared_frame_count}, actual={int(indices.size)}"
            )
        if not is_present:
            if indices.size:
                raise ValueError(f"missing RD row contains frame indices: {sample_id}")
            amplitude = np.zeros((0, PHYSICAL_RD_RANGE_BINS, PHYSICAL_RD_DOPPLER_BINS), dtype=np.float32)
            phase = np.zeros_like(amplitude)
            time_seconds = np.zeros((0,), dtype=np.float32)
            range_offset_m = np.linspace(-75.0, 75.0, PHYSICAL_RD_RANGE_BINS, dtype=np.float32)
            velocity_mps = np.linspace(-85.0, 85.0, PHYSICAL_RD_DOPPLER_BINS, dtype=np.float32)
        else:
            amplitude, phase, time_seconds, range_offset_m, velocity_mps = self._load_tensor(str(row["rd_tensor_path"]))
        if indices.size and int(indices[-1]) >= int(amplitude.shape[0]):
            raise ValueError(f"RD-only frame index exceeds tensor bounds: {sample_id}")
        selected_time_seconds = time_seconds[indices]
        if is_present:
            start_error = abs(float(selected_time_seconds[0]) - float(row["rd_time_start_seconds"]))
            end_error = abs(float(selected_time_seconds[-1]) - float(row["rd_time_end_seconds"]))
            if max(start_error, end_error) > 5.0e-7:
                raise ValueError(f"RD-only index/tensor time mismatch: {sample_id}")
        features = _physical_rd_features(
            amplitude[indices],
            phase[indices],
            selected_time_seconds,
            range_offset_m,
            velocity_mps,
            self.normalization_stats,
        )
        quality_features = _quality_metadata(
            row,
            amplitude[indices],
            selected_time_seconds,
            quality_schema=self.quality_schema,
        )
        return PhysicalRDWindowItem(
            sample_id=sample_id,
            parent_sample_id=str(row["parent_sample_id"]),
            label_name=str(row["label_name"]),
            split_name=str(row["split_name"]),
            window_mode=str(row["rd_window_mode"]),
            is_prefix_window=_as_bool(row["is_prefix_window"]),
            window_frame_count=int(row["rd_window_frame_count"]),
            features=features,
            raw_amplitude_db=torch.from_numpy(
                np.ascontiguousarray(np.nan_to_num(amplitude[indices], nan=0.0, posinf=0.0, neginf=0.0))
            ).to(torch.float32),
            time_seconds=torch.from_numpy(np.ascontiguousarray(selected_time_seconds)).to(torch.float32),
            quality_features=quality_features,
            quality_metadata=quality_features,
            range_offset_m=torch.from_numpy(np.ascontiguousarray(range_offset_m)).to(torch.float32),
            velocity_mps=torch.from_numpy(np.ascontiguousarray(velocity_mps)).to(torch.float32),
            label=encode_rd_drone_bird_label(str(row["label_name"])),
        )


def collate_physical_rd_windows(items: list[PhysicalRDWindowItem]) -> PhysicalRDWindowBatch:
    if not items:
        raise ValueError("cannot collate an empty physical RD window batch")
    feature_shape = tuple(items[0].features.shape[1:])
    if len(feature_shape) != 3:
        raise ValueError("physical RD item features must be [time, channel, range, doppler]")
    max_length = max(1, max(int(item.features.shape[0]) for item in items))
    batch_size = len(items)
    features = torch.zeros((batch_size, max_length, *feature_shape), dtype=torch.float32)
    raw_amplitude_db = torch.zeros((batch_size, max_length, feature_shape[1], feature_shape[2]), dtype=torch.float32)
    time_seconds = torch.zeros((batch_size, max_length), dtype=torch.float32)
    padding_mask = torch.ones((batch_size, max_length), dtype=torch.bool)
    lengths = torch.empty((batch_size,), dtype=torch.long)
    quality_dim = int(items[0].quality_features.numel())
    metadata_dim = int(items[0].quality_metadata.numel())
    quality_features = torch.empty((batch_size, quality_dim), dtype=torch.float32)
    quality_metadata = torch.empty((batch_size, metadata_dim), dtype=torch.float32)
    labels = torch.empty((batch_size,), dtype=torch.long)
    is_prefix_windows = torch.empty((batch_size,), dtype=torch.bool)
    window_frame_counts = torch.empty((batch_size,), dtype=torch.long)
    range_offset_m = items[0].range_offset_m.clone()
    velocity_mps = items[0].velocity_mps.clone()
    for batch_index, item in enumerate(items):
        length = int(item.features.shape[0])
        if (
            tuple(item.features.shape[1:]) != feature_shape
            or tuple(item.raw_amplitude_db.shape) != (length, feature_shape[1], feature_shape[2])
            or int(item.time_seconds.numel()) != length
            or tuple(item.quality_features.shape) != (quality_dim,)
            or tuple(item.quality_metadata.shape) != (metadata_dim,)
            or not torch.equal(item.range_offset_m, range_offset_m)
            or not torch.equal(item.velocity_mps, velocity_mps)
        ):
            raise ValueError("physical RD batch contains incompatible item shapes")
        features[batch_index, :length] = item.features
        raw_amplitude_db[batch_index, :length] = item.raw_amplitude_db
        time_seconds[batch_index, :length] = item.time_seconds
        padding_mask[batch_index, :length] = False
        lengths[batch_index] = length
        quality_features[batch_index] = item.quality_features
        quality_metadata[batch_index] = item.quality_metadata
        labels[batch_index] = item.label
        is_prefix_windows[batch_index] = item.is_prefix_window
        window_frame_counts[batch_index] = item.window_frame_count
    if metadata_dim == quality_dim and all(torch.equal(item.quality_metadata, item.quality_features) for item in items):
        quality_metadata = quality_features
    return PhysicalRDWindowBatch(
        sample_ids=[item.sample_id for item in items],
        parent_sample_ids=[item.parent_sample_id for item in items],
        label_names=[item.label_name for item in items],
        split_names=[item.split_name for item in items],
        window_modes=[item.window_mode for item in items],
        is_prefix_windows=is_prefix_windows,
        window_frame_counts=window_frame_counts,
        features=features,
        raw_amplitude_db=raw_amplitude_db,
        time_seconds=time_seconds,
        padding_mask=padding_mask,
        lengths=lengths,
        quality_features=quality_features,
        range_offset_m=range_offset_m,
        velocity_mps=velocity_mps,
        labels=labels,
        quality_metadata=quality_metadata,
    )


__all__ = [
    "PhysicalRDWindowBatch",
    "PhysicalRDWindowDataset",
    "PhysicalRDWindowItem",
    "RD_ONLY_LABELS",
    "QUALITY_FEATURE_DIM",
    "QUALITY_FEATURE_NAMES",
    "QUALITY_METADATA_DIM",
    "QUALITY_METADATA_NAMES",
    "RD_ONLY_QUALITY_NAMES",
    "RD_ONLY_QUALITY_SCHEMA",
    "LEGACY_QUALITY_SCHEMA",
    "PHYSICAL_RD_FEATURE_NAMES",
    "PHYSICAL_RD_INPUT_CHANNELS",
    "PHYSICAL_RD_DOPPLER_BINS",
    "PHYSICAL_RD_RANGE_BINS",
    "collate_physical_rd_windows",
    "encode_rd_drone_bird_label",
]
