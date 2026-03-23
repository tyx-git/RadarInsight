import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import h5py
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


@dataclass(slots=True)
class ClipRecord:
    session_id: str
    sample_id: str
    track_id: int
    label_name: str
    t0: float
    t1: float
    rd_key: str
    rd_path: Path | None
    has_rd: bool


@dataclass(slots=True)
class SessionContext:
    track_frame: pd.DataFrame
    rd_data_path: Path | None = None


class RadarFusionDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        session_ids: Sequence[str] | None = None,
        label_order: Sequence[str] | None = None,
        sequence_length: int = 16,
        rd_frames: int = 16,
        rd_height: int = 32,
        rd_width: int = 32,
        rd_input_channels: int = 1,
        feature_columns: Sequence[str] = (
            "range_m",
            "azimuth_deg",
            "radial_speed_mps",
            "snr_db",
        ),
        include_rd: bool = True,
    ):
        self.data_root = Path(data_root)
        self.sequence_length = sequence_length
        self.rd_frames = rd_frames
        self.rd_height = rd_height
        self.rd_width = rd_width
        self.rd_input_channels = rd_input_channels
        self.feature_columns = tuple(feature_columns)
        self.include_rd = include_rd
        self.sessions: dict[str, SessionContext] = {}
        self.samples: list[ClipRecord] = []
        discovered_label_names: set[str] = set()

        for session_dir in self._resolve_session_dirs(session_ids):
            context, records = self._load_session(session_dir)
            if context is None:
                continue

            self.sessions[session_dir.name] = context
            self.samples.extend(records)
            discovered_label_names.update(record.label_name for record in records)

        if not self.samples:
            raise ValueError("No labeled samples found for training. Run process and generate matched labels first.")

        resolved_label_names = tuple(label_order) if label_order is not None else tuple(sorted(discovered_label_names))
        self.index_to_label = {index: label for index, label in enumerate(resolved_label_names)}
        self.label_to_index = {label: index for index, label in self.index_to_label.items()}
        unknown_labels = sorted({record.label_name for record in self.samples if record.label_name not in self.label_to_index})
        if unknown_labels:
            raise ValueError(f"Found labels not present in the provided label mapping: {unknown_labels}")

        self.track_feature_center, self.track_feature_scale = self._build_track_feature_stats()

    def build_session_index(self) -> dict[str, list[int]]:
        session_to_indices: dict[str, list[int]] = {}
        for index, record in enumerate(self.samples):
            session_to_indices.setdefault(record.session_id, []).append(index)
        return session_to_indices

    @staticmethod
    def _safe_time_delta(time_values: np.ndarray) -> np.ndarray:
        if len(time_values) == 0:
            return np.ones(1, dtype=np.float32)

        deltas = np.diff(time_values, prepend=time_values[0]).astype(np.float32)
        positive_deltas = deltas[deltas > 0]
        fallback_delta = float(np.median(positive_deltas)) if len(positive_deltas) else 1.0
        deltas[deltas <= 0] = fallback_delta
        return deltas

    @classmethod
    def _augment_track_features(cls, window: pd.DataFrame) -> pd.DataFrame:
        if window.empty:
            return window

        for column in ("t", "range_m", "azimuth_deg", "radial_speed_mps", "snr_db"):
            if column not in window.columns:
                window[column] = 0.0

        time_values = window["t"].to_numpy(dtype=np.float32)
        deltas = cls._safe_time_delta(time_values)

        window["delta_range_m"] = window["range_m"].diff().fillna(0.0)
        window["delta_azimuth_deg"] = window["azimuth_deg"].diff().fillna(0.0)
        window["speed_delta_mps"] = window["radial_speed_mps"].diff().fillna(0.0)
        window["acceleration_mps2"] = window["speed_delta_mps"].to_numpy(dtype=np.float32) / deltas
        window["turn_rate_degps"] = window["delta_azimuth_deg"].to_numpy(dtype=np.float32) / deltas
        window["snr_delta_db"] = window["snr_db"].diff().fillna(0.0)

        smoothing_window = max(2, min(5, len(window)))
        rolling_range = window["range_m"].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        window["range_jitter_m"] = window["range_m"] - rolling_range
        return window

    def _resolve_session_dirs(self, session_ids: Sequence[str] | None) -> list[Path]:
        if session_ids:
            return [self.data_root / session_id for session_id in session_ids if (self.data_root / session_id).exists()]

        return sorted(path for path in self.data_root.iterdir() if path.is_dir())

    def _load_session(self, session_dir: Path) -> tuple[SessionContext | None, list[ClipRecord]]:
        track_csv = session_dir / "radar/track/track_data.csv"
        track_clip_file = session_dir / "labels/labels_trackclip.jsonl"
        matched_file = session_dir / "labels/labels_matched.jsonl"
        rd_clip_file = session_dir / "radar/rd/rd_clips.h5"

        if not track_csv.exists() or not track_clip_file.exists() or not matched_file.exists():
            return None, []

        track_frame = pd.read_csv(track_csv).sort_values("t")
        label_by_track = {
            int(item["track_id"]): str(item["label_main"])
            for item in self._load_jsonl(matched_file)
            if "track_id" in item and "label_main" in item
        }
        if not label_by_track:
            return None, []

        available_rd_keys: set[str] = set()
        if self.include_rd and rd_clip_file.exists():
            with h5py.File(rd_clip_file, "r") as handle:
                available_rd_keys = set(handle.keys())

        records: list[ClipRecord] = []
        for clip in self._load_jsonl(track_clip_file):
            track_id = int(clip["track_id"])
            label_name = label_by_track.get(track_id)
            if label_name is None:
                continue

            rd_key = f"rd_clip_{clip['sample_id']}"
            has_rd = self.include_rd and rd_key in available_rd_keys
            records.append(
                ClipRecord(
                    session_id=session_dir.name,
                    sample_id=str(clip["sample_id"]),
                    track_id=track_id,
                    label_name=label_name,
                    t0=float(clip["t0"]),
                    t1=float(clip["t1"]),
                    rd_key=rd_key,
                    rd_path=rd_clip_file if rd_clip_file.exists() else None,
                    has_rd=has_rd,
                )
            )

        if not records:
            return None, []

        rd_data_file = session_dir / "radar/rd/rd_data.h5"
        return SessionContext(track_frame=track_frame, rd_data_path=rd_data_file if rd_data_file.exists() else None), records

    @staticmethod
    def _load_jsonl(file_path: Path) -> list[dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as file:
            return [json.loads(line) for line in file if line.strip()]

    @staticmethod
    def _resample_sequence(array: np.ndarray, target_length: int) -> np.ndarray:
        if len(array) == 0:
            feature_dim = array.shape[1] if array.ndim > 1 else 1
            return np.zeros((target_length, feature_dim), dtype=np.float32)

        if len(array) == 1:
            return np.repeat(array, target_length, axis=0).astype(np.float32)

        source_positions = np.linspace(0, len(array) - 1, num=len(array), dtype=np.float32)
        target_positions = np.linspace(0, len(array) - 1, num=target_length, dtype=np.float32)
        resampled = np.empty((target_length, array.shape[1]), dtype=np.float32)
        for column_index in range(array.shape[1]):
            resampled[:, column_index] = np.interp(target_positions, source_positions, array[:, column_index])
        return resampled

    @staticmethod
    def _standardize(array: np.ndarray) -> np.ndarray:
        mean = array.mean(axis=0, keepdims=True)
        std = array.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return ((array - mean) / std).astype(np.float32)

    def _build_track_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        feature_blocks: list[np.ndarray] = []
        for context in self.sessions.values():
            augmented = self._augment_track_features(context.track_frame.copy())
            for column in self.feature_columns:
                if column not in augmented.columns:
                    augmented[column] = 0.0
            feature_blocks.append(augmented.loc[:, self.feature_columns].to_numpy(dtype=np.float32))

        stacked = np.concatenate(feature_blocks, axis=0) if feature_blocks else np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        center = np.median(stacked, axis=0, keepdims=True)
        q25 = np.percentile(stacked, 25, axis=0, keepdims=True)
        q75 = np.percentile(stacked, 75, axis=0, keepdims=True)
        scale = q75 - q25
        fallback = stacked.std(axis=0, keepdims=True)
        scale[scale < 1e-6] = fallback[scale < 1e-6]
        scale[scale < 1e-6] = 1.0
        return center.astype(np.float32), scale.astype(np.float32)

    def _normalize_track_sequence(self, array: np.ndarray) -> np.ndarray:
        normalized = (array - self.track_feature_center) / self.track_feature_scale
        normalized = np.clip(normalized, -8.0, 8.0)
        return normalized.astype(np.float32)

    @staticmethod
    def _standardize_channel_first(array: np.ndarray) -> np.ndarray:
        scale = np.percentile(np.abs(array), 95, axis=(1, 2, 3), keepdims=True)
        scale[scale < 1e-6] = 1.0
        scaled = array / scale
        scaled = np.clip(scaled, -8.0, 8.0)
        return scaled.astype(np.float32)

    @staticmethod
    def _resample_frames(array: np.ndarray, target_frames: int) -> np.ndarray:
        if array.shape[0] == 0:
            return np.zeros((target_frames, array.shape[1], array.shape[2]), dtype=array.dtype)

        if array.shape[0] == target_frames:
            return np.asarray(array, dtype=array.dtype)

        if array.shape[0] == 1:
            return np.repeat(array, target_frames, axis=0).astype(array.dtype, copy=False)

        indices = np.linspace(0, array.shape[0] - 1, num=target_frames).round().astype(int)
        return np.asarray(array[indices], dtype=array.dtype)

    @staticmethod
    def _center_crop_or_pad(array: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        output = np.zeros((array.shape[0], target_height, target_width), dtype=array.dtype)
        copy_height = min(array.shape[1], target_height)
        copy_width = min(array.shape[2], target_width)

        source_top = max((array.shape[1] - copy_height) // 2, 0)
        source_left = max((array.shape[2] - copy_width) // 2, 0)
        target_top = max((target_height - copy_height) // 2, 0)
        target_left = max((target_width - copy_width) // 2, 0)

        output[
            :,
            target_top : target_top + copy_height,
            target_left : target_left + copy_width,
        ] = array[
            :,
            source_top : source_top + copy_height,
            source_left : source_left + copy_width,
        ]
        return output

    @staticmethod
    def _resize_full_context(array: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        if array.shape[1] == target_height and array.shape[2] == target_width:
            return np.asarray(array, dtype=array.dtype)

        row_indices = np.linspace(0, array.shape[1] - 1, num=target_height).round().astype(int)
        col_indices = np.linspace(0, array.shape[2] - 1, num=target_width).round().astype(int)
        return np.asarray(array[:, row_indices][:, :, col_indices], dtype=array.dtype)

    @staticmethod
    def _spatial_mean_3x3(array: np.ndarray) -> np.ndarray:
        padded = np.pad(array, ((0, 0), (1, 1), (1, 1)), mode="edge")
        accumulator = np.zeros_like(array, dtype=np.float32)
        for row_offset in range(3):
            for col_offset in range(3):
                accumulator += padded[:, row_offset : row_offset + array.shape[1], col_offset : col_offset + array.shape[2]]
        return accumulator / 9.0

    def _build_rd_feature_channels(self, local_rd_cube: np.ndarray, global_rd_cube: np.ndarray | None = None) -> np.ndarray:
        if local_rd_cube.ndim == 4:
            scale_cubes = [local_rd_cube[scale_index] for scale_index in range(local_rd_cube.shape[0])]
        else:
            scale_cubes = [local_rd_cube]

        magnitude_cubes = [np.abs(scale_cube).astype(np.float32) for scale_cube in scale_cubes]
        log_amplitude_cubes = [np.log1p(magnitude_cube).astype(np.float32) for magnitude_cube in magnitude_cubes]
        middle_index = len(scale_cubes) // 2
        mid_cube = scale_cubes[middle_index]
        mid_log_amplitude = log_amplitude_cubes[middle_index]

        if np.iscomplexobj(mid_cube):
            real_component = mid_cube.real.astype(np.float32)
            imag_component = mid_cube.imag.astype(np.float32)
            previous_cube = np.concatenate([mid_cube[:1], mid_cube[:-1]], axis=0)
            phase_difference = np.angle(mid_cube * np.conjugate(previous_cube)).astype(np.float32)
            coherence = np.abs(mid_cube * np.conjugate(previous_cube)) / (
                (np.abs(mid_cube) * np.abs(previous_cube)) + 1e-6
            )
            coherence = coherence.astype(np.float32)
        else:
            real_component = np.asarray(mid_cube, dtype=np.float32)
            imag_component = np.zeros_like(real_component, dtype=np.float32)
            phase_difference = np.zeros_like(real_component, dtype=np.float32)
            coherence = np.zeros_like(real_component, dtype=np.float32)

        temporal_difference = np.diff(
            mid_log_amplitude,
            axis=0,
            prepend=mid_log_amplitude[:1],
        )

        background = np.median(mid_log_amplitude, axis=0, keepdims=True)
        background_suppressed = np.maximum(mid_log_amplitude - background, 0.0)

        local_mean = self._spatial_mean_3x3(mid_log_amplitude)
        local_contrast = mid_log_amplitude - local_mean

        if global_rd_cube is None:
            global_log_amplitude = np.zeros_like(mid_log_amplitude, dtype=np.float32)
            global_background_suppressed = np.zeros_like(mid_log_amplitude, dtype=np.float32)
            global_temporal_difference = np.zeros_like(mid_log_amplitude, dtype=np.float32)
            global_contrast = np.zeros_like(mid_log_amplitude, dtype=np.float32)
        else:
            global_magnitude = np.abs(global_rd_cube).astype(np.float32)
            global_log_amplitude = np.log1p(global_magnitude).astype(np.float32)
            global_temporal_difference = np.diff(
                global_log_amplitude,
                axis=0,
                prepend=global_log_amplitude[:1],
            )
            global_background = np.median(global_log_amplitude, axis=0, keepdims=True)
            global_background_suppressed = np.maximum(global_log_amplitude - global_background, 0.0)
            global_mean = self._spatial_mean_3x3(global_log_amplitude)
            global_contrast = global_log_amplitude - global_mean

        if len(scale_cubes) == 1:
            channels = [
                real_component,
                imag_component,
                mid_log_amplitude,
                temporal_difference,
                background_suppressed,
                local_contrast,
                global_log_amplitude,
                global_background_suppressed,
                global_temporal_difference,
                global_contrast,
                phase_difference,
                coherence,
            ]
        else:
            short_log_amplitude = log_amplitude_cubes[0]
            long_log_amplitude = log_amplitude_cubes[-1]
            channels = [
                real_component,
                imag_component,
                short_log_amplitude,
                mid_log_amplitude,
                long_log_amplitude,
                temporal_difference,
                background_suppressed,
                local_contrast,
                global_log_amplitude,
                global_background_suppressed,
                global_temporal_difference,
                global_contrast,
                phase_difference,
                coherence,
            ]
        selected = channels[: self.rd_input_channels]
        while len(selected) < self.rd_input_channels:
            selected.append(np.zeros_like(mid_log_amplitude, dtype=np.float32))
        return self._standardize_channel_first(np.stack(selected, axis=0))

    def _load_global_rd_cube(self, record: ClipRecord) -> np.ndarray | None:
        context = self.sessions[record.session_id]
        if context.rd_data_path is None or not context.rd_data_path.exists():
            return None

        track_frame = context.track_frame
        window = track_frame[
            (track_frame["track_id"] == record.track_id)
            & (track_frame["t"] >= record.t0)
            & (track_frame["t"] < record.t1)
        ]
        frame_indices = window["frame_id"].to_numpy(dtype=np.int64)

        with h5py.File(context.rd_data_path, "r") as handle:
            rd_frames_dataset = handle.get("rd_frames")
            if not isinstance(rd_frames_dataset, h5py.Dataset):
                return None

            max_frames = int(rd_frames_dataset.shape[0])
            if frame_indices.size == 0:
                clip_center_time = 0.5 * (record.t0 + record.t1)
                nearest_index = int(np.argmin(np.abs(track_frame["t"].to_numpy(dtype=np.float32) - clip_center_time)))
                frame_indices = np.asarray([nearest_index], dtype=np.int64)
            frame_indices = frame_indices[frame_indices < max_frames]
            if frame_indices.size == 0:
                return None

            global_rd_cube = np.asarray(rd_frames_dataset[frame_indices])

        if global_rd_cube.ndim == 2:
            global_rd_cube = global_rd_cube[None, :, :]
        global_rd_cube = self._resample_frames(global_rd_cube, self.rd_frames)
        global_rd_cube = self._resize_full_context(global_rd_cube, self.rd_height, self.rd_width)
        return global_rd_cube

    def _load_track_sequence(self, record: ClipRecord) -> np.ndarray:
        track_frame = self.sessions[record.session_id].track_frame
        window = track_frame[
            (track_frame["track_id"] == record.track_id)
            & (track_frame["t"] >= record.t0)
            & (track_frame["t"] < record.t1)
        ].copy()

        if window.empty:
            window = track_frame[track_frame["track_id"] == record.track_id].copy()

        window = self._augment_track_features(window)

        for column in self.feature_columns:
            if column not in window.columns:
                window[column] = 0.0

        sequence = window.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        sequence = self._resample_sequence(sequence, self.sequence_length)
        return self._normalize_track_sequence(sequence)

    def _load_rd_cube(self, record: ClipRecord) -> np.ndarray:
        if not self.include_rd or not record.has_rd or record.rd_path is None:
            return np.zeros((self.rd_input_channels, self.rd_frames, self.rd_height, self.rd_width), dtype=np.float32)

        with h5py.File(record.rd_path, "r") as handle:
            rd_cube = np.asarray(handle[record.rd_key][:])

        if rd_cube.ndim == 2:
            rd_cube = rd_cube[None, :, :]

        if rd_cube.ndim == 4:
            processed_scales: list[np.ndarray] = []
            for scale_cube in rd_cube:
                scale_cube = self._resample_frames(scale_cube, self.rd_frames)
                scale_cube = self._center_crop_or_pad(scale_cube, self.rd_height, self.rd_width)
                processed_scales.append(scale_cube)
            rd_cube = np.stack(processed_scales, axis=0)
        else:
            rd_cube = self._resample_frames(rd_cube, self.rd_frames)
            rd_cube = self._center_crop_or_pad(rd_cube, self.rd_height, self.rd_width)
        global_rd_cube = self._load_global_rd_cube(record)
        return self._build_rd_feature_channels(rd_cube, global_rd_cube=global_rd_cube)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        return {
            "sample_id": record.sample_id,
            "track_features": self._load_track_sequence(record),
            "rd_features": self._load_rd_cube(record),
            "label": np.int64(self.label_to_index[record.label_name]),
            "label_name": record.label_name,
            "has_rd": np.int64(record.has_rd),
        }
