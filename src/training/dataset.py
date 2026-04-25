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
    source_name: str
    t0: float
    t1: float
    rd_key: str
    rd_path: Path | None
    has_rd: bool


@dataclass(slots=True)
class SessionContext:
    track_csv_path: Path
    rd_data_path: Path | None = None
    source_name: str = 'Unknown'
    track_frame: pd.DataFrame | None = None


class RadarFusionDataset(Dataset):
    DATASET_INDEX_CACHE_FILENAME = 'dataset_clip_index_cache.json'
    DATASET_STATS_CACHE_FILENAME = 'dataset_track_stats_cache.json'

    @staticmethod
    def _sanitize_rd_array(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if np.iscomplexobj(array):
            real = np.nan_to_num(np.asarray(array.real, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            imag = np.nan_to_num(np.asarray(array.imag, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            return (real + 1j * imag).astype(np.complex64)
        return np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    def __init__(
        self,
        data_root: str,
        session_ids: Sequence[str] | None = None,
        label_order: Sequence[str] | None = None,
        label_granularity: str = 'coarse',
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
        global_feature_columns: Sequence[str] = (),
        include_rd: bool = True,
        conditional_ship_drone_session_ids: Sequence[str] = (),
    ):
        self.data_root = Path(data_root)
        self.label_granularity = str(label_granularity).lower()
        self.sequence_length = sequence_length
        self.rd_frames = rd_frames
        self.rd_height = rd_height
        self.rd_width = rd_width
        self.rd_input_channels = rd_input_channels
        self.feature_columns = tuple(feature_columns)
        self.global_feature_columns = tuple(global_feature_columns)
        self.include_rd = include_rd
        self.conditional_ship_drone_session_ids = {str(item) for item in conditional_ship_drone_session_ids}
        self.track_short_length = max(4, sequence_length // 2)
        self.track_long_length = max(sequence_length, sequence_length + max(4, sequence_length // 2))
        self.window_mix_probability = 0.35
        self.window_mix_enabled_indices: set[int] | None = None
        self.track_global_feature_cache: dict[str, np.ndarray] = {}
        self.sessions: dict[str, SessionContext] = {}
        self.samples: list[ClipRecord] = []
        discovered_label_names: set[str] = set()
        self._dataset_index_cache_path = self.data_root / self.DATASET_INDEX_CACHE_FILENAME
        self._dataset_index_cache = self._load_dataset_index_cache()
        self._dataset_index_cache_dirty = False
        self._dataset_stats_cache_path = self.data_root / self.DATASET_STATS_CACHE_FILENAME

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

        if self._dataset_index_cache_dirty:
            self._save_dataset_index_cache()

        self.track_feature_center = np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        self.track_feature_scale = np.ones((1, len(self.feature_columns)), dtype=np.float32)
        self.track_global_center = np.zeros((1, len(self.global_feature_columns)), dtype=np.float32)
        self.track_global_scale = np.ones((1, len(self.global_feature_columns)), dtype=np.float32)
        self.track_feature_center_by_source: dict[str, np.ndarray] = {}
        self.track_feature_scale_by_source: dict[str, np.ndarray] = {}
        self.track_global_center_by_source: dict[str, np.ndarray] = {}
        self.track_global_scale_by_source: dict[str, np.ndarray] = {}

    def set_track_normalization(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        by_source: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.track_feature_center = np.asarray(center, dtype=np.float32)
        self.track_feature_scale = np.asarray(scale, dtype=np.float32)
        self.track_feature_center_by_source = {}
        self.track_feature_scale_by_source = {}
        if isinstance(by_source, dict):
            for source_name, payload in by_source.items():
                if not isinstance(payload, dict):
                    continue
                source_center = np.asarray(payload.get('center', []), dtype=np.float32)
                source_scale = np.asarray(payload.get('scale', []), dtype=np.float32)
                if source_center.size == len(self.feature_columns) and source_scale.size == len(self.feature_columns):
                    self.track_feature_center_by_source[str(source_name)] = source_center.reshape(1, -1)
                    self.track_feature_scale_by_source[str(source_name)] = source_scale.reshape(1, -1)

    def get_track_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        return self.track_feature_center.copy(), self.track_feature_scale.copy()

    def get_track_normalization_by_source(self) -> dict[str, dict[str, list[float]]]:
        payload: dict[str, dict[str, list[float]]] = {}
        for source_name, center in self.track_feature_center_by_source.items():
            scale = self.track_feature_scale_by_source.get(source_name)
            if scale is None:
                continue
            payload[str(source_name)] = {
                'center': center.reshape(-1).tolist(),
                'scale': scale.reshape(-1).tolist(),
            }
        return payload

    def set_track_global_normalization(
        self,
        center: np.ndarray,
        scale: np.ndarray,
        by_source: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self.track_global_center = np.asarray(center, dtype=np.float32)
        self.track_global_scale = np.asarray(scale, dtype=np.float32)
        self.track_global_center_by_source = {}
        self.track_global_scale_by_source = {}
        if isinstance(by_source, dict):
            for source_name, payload in by_source.items():
                if not isinstance(payload, dict):
                    continue
                source_center = np.asarray(payload.get('center', []), dtype=np.float32)
                source_scale = np.asarray(payload.get('scale', []), dtype=np.float32)
                if source_center.size == len(self.global_feature_columns) and source_scale.size == len(self.global_feature_columns):
                    self.track_global_center_by_source[str(source_name)] = source_center.reshape(1, -1)
                    self.track_global_scale_by_source[str(source_name)] = source_scale.reshape(1, -1)

    def get_track_global_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        return self.track_global_center.copy(), self.track_global_scale.copy()

    def get_track_global_normalization_by_source(self) -> dict[str, dict[str, list[float]]]:
        payload: dict[str, dict[str, list[float]]] = {}
        for source_name, center in self.track_global_center_by_source.items():
            scale = self.track_global_scale_by_source.get(source_name)
            if scale is None:
                continue
            payload[str(source_name)] = {
                'center': center.reshape(-1).tolist(),
                'scale': scale.reshape(-1).tolist(),
            }
        return payload

    def set_window_mix_enabled_indices(self, indices: Sequence[int] | None) -> None:
        if indices is None:
            self.window_mix_enabled_indices = None
            return
        self.window_mix_enabled_indices = {int(index) for index in indices}

    def _load_dataset_index_cache(self) -> dict[str, Any]:
        if not self._dataset_index_cache_path.exists():
            return {'version': 1, 'sessions': {}}
        try:
            payload = json.loads(self._dataset_index_cache_path.read_text(encoding='utf-8'))
        except Exception:
            return {'version': 1, 'sessions': {}}
        sessions_payload = payload.get('sessions')
        if not isinstance(sessions_payload, dict):
            return {'version': 1, 'sessions': {}}
        return {'version': 1, 'sessions': sessions_payload}

    def _save_dataset_index_cache(self) -> None:
        payload = {'version': 1, 'sessions': self._dataset_index_cache.get('sessions', {})}
        self._dataset_index_cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    def _load_dataset_stats_cache(self) -> dict[str, Any]:
        if not self._dataset_stats_cache_path.exists():
            return {'entries': {}}
        try:
            payload = json.loads(self._dataset_stats_cache_path.read_text(encoding='utf-8'))
        except Exception:
            return {'entries': {}}
        if not isinstance(payload, dict):
            return {'entries': {}}
        entries = payload.get('entries')
        if not isinstance(entries, dict):
            return {'entries': {}}
        return {'entries': entries}

    def _save_dataset_stats_cache(self, payload: dict[str, Any]) -> None:
        self._dataset_stats_cache_path.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

    @staticmethod
    def _build_session_fingerprint(track_csv: Path, track_clip_file: Path, matched_file: Path, rd_clip_file: Path) -> str:
        parts: list[str] = []
        for path in (track_csv, track_clip_file, matched_file, rd_clip_file):
            if path.exists():
                stat = path.stat()
                parts.append(f'{path.name}:{stat.st_size}:{stat.st_mtime_ns}')
            else:
                parts.append(f'{path.name}:missing')
        return '|'.join(parts)

    def _get_track_frame(self, session_id: str) -> pd.DataFrame:
        context = self.sessions[session_id]
        if context.track_frame is None:
            context.track_frame = pd.read_csv(context.track_csv_path).sort_values('t')
        return context.track_frame

    def fit_track_normalization_for_sessions(self, session_ids: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        cache_key = {
            'kind': 'track_feature',
            'sessions': list(session_ids) if session_ids is not None else '__all__',
            'columns': list(self.feature_columns),
            'normalization_mode': 'by_source_v1',
        }
        cache_payload = self._load_dataset_stats_cache()
        cache_entry = cache_payload.get('entries', {}).get('track_feature', {})
        if cache_entry.get('cache_key') == cache_key:
            center = np.asarray(cache_entry.get('center', []), dtype=np.float32)
            scale = np.asarray(cache_entry.get('scale', []), dtype=np.float32)
            if center.size == len(self.feature_columns) and scale.size == len(self.feature_columns):
                self.set_track_normalization(center.reshape(1, -1), scale.reshape(1, -1), by_source=cache_entry.get('by_source'))
                return self.get_track_normalization()

        feature_blocks: list[np.ndarray] = []
        feature_blocks_by_source: dict[str, list[np.ndarray]] = {}
        selected_session_ids = set(session_ids) if session_ids is not None else None
        for session_id, context in self.sessions.items():
            if selected_session_ids is not None and session_id not in selected_session_ids:
                continue
            augmented = self._augment_track_features(self._get_track_frame(session_id).copy())
            for column in self.feature_columns:
                if column not in augmented.columns:
                    augmented[column] = 0.0
            values = augmented.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
            feature_blocks.append(values)
            feature_blocks_by_source.setdefault(str(context.source_name), []).append(values)

        stacked = np.concatenate(feature_blocks, axis=0) if feature_blocks else np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        center = np.median(stacked, axis=0, keepdims=True)
        q25 = np.percentile(stacked, 25, axis=0, keepdims=True)
        q75 = np.percentile(stacked, 75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-3] = 1.0
        by_source_payload: dict[str, dict[str, list[float]]] = {}
        for source_name, blocks in feature_blocks_by_source.items():
            source_stacked = np.concatenate(blocks, axis=0) if blocks else np.zeros((1, len(self.feature_columns)), dtype=np.float32)
            source_center = np.median(source_stacked, axis=0, keepdims=True)
            source_q25 = np.percentile(source_stacked, 25, axis=0, keepdims=True)
            source_q75 = np.percentile(source_stacked, 75, axis=0, keepdims=True)
            source_scale = source_q75 - source_q25
            source_scale[source_scale < 1e-3] = 1.0
            by_source_payload[str(source_name)] = {
                'center': source_center.reshape(-1).astype(np.float32).tolist(),
                'scale': source_scale.reshape(-1).astype(np.float32).tolist(),
            }
        self.set_track_normalization(center.astype(np.float32), scale.astype(np.float32), by_source=by_source_payload)
        cache_payload.setdefault('entries', {})['track_feature'] = {
            'cache_key': cache_key,
            'center': self.track_feature_center.reshape(-1).tolist(),
            'scale': self.track_feature_scale.reshape(-1).tolist(),
            'by_source': by_source_payload,
        }
        self._save_dataset_stats_cache(cache_payload)
        return self.get_track_normalization()

    def fit_track_global_normalization_for_sessions(self, session_ids: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        cache_key = {
            'kind': 'track_global_feature',
            'sessions': list(session_ids) if session_ids is not None else '__all__',
            'columns': list(self.global_feature_columns),
            'normalization_mode': 'by_source_v1',
        }
        cache_payload = self._load_dataset_stats_cache()
        cache_entry = cache_payload.get('entries', {}).get('track_global_feature', {})
        if cache_entry.get('cache_key') == cache_key:
            center = np.asarray(cache_entry.get('center', []), dtype=np.float32)
            scale = np.asarray(cache_entry.get('scale', []), dtype=np.float32)
            if center.size == len(self.global_feature_columns) and scale.size == len(self.global_feature_columns):
                self.set_track_global_normalization(center.reshape(1, -1), scale.reshape(1, -1), by_source=cache_entry.get('by_source'))
                return self.get_track_global_normalization()

        feature_blocks: list[np.ndarray] = []
        feature_blocks_by_source: dict[str, list[np.ndarray]] = {}
        selected_session_ids = set(session_ids) if session_ids is not None else None
        for record in self.samples:
            if selected_session_ids is not None and record.session_id not in selected_session_ids:
                continue
            values = self._extract_track_global_features(record)
            feature_blocks.append(values)
            feature_blocks_by_source.setdefault(str(record.source_name), []).append(values)

        stacked = np.stack(feature_blocks, axis=0) if feature_blocks else np.zeros((1, len(self.global_feature_columns)), dtype=np.float32)
        center = np.median(stacked, axis=0, keepdims=True)
        q25 = np.percentile(stacked, 25, axis=0, keepdims=True)
        q75 = np.percentile(stacked, 75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-3] = 1.0
        by_source_payload: dict[str, dict[str, list[float]]] = {}
        for source_name, blocks in feature_blocks_by_source.items():
            source_stacked = np.stack(blocks, axis=0) if blocks else np.zeros((1, len(self.global_feature_columns)), dtype=np.float32)
            source_center = np.median(source_stacked, axis=0, keepdims=True)
            source_q25 = np.percentile(source_stacked, 25, axis=0, keepdims=True)
            source_q75 = np.percentile(source_stacked, 75, axis=0, keepdims=True)
            source_scale = source_q75 - source_q25
            source_scale[source_scale < 1e-3] = 1.0
            by_source_payload[str(source_name)] = {
                'center': source_center.reshape(-1).astype(np.float32).tolist(),
                'scale': source_scale.reshape(-1).astype(np.float32).tolist(),
            }
        self.set_track_global_normalization(center.astype(np.float32), scale.astype(np.float32), by_source=by_source_payload)
        cache_payload.setdefault('entries', {})['track_global_feature'] = {
            'cache_key': cache_key,
            'center': self.track_global_center.reshape(-1).tolist(),
            'scale': self.track_global_scale.reshape(-1).tolist(),
            'by_source': by_source_payload,
        }
        self._save_dataset_stats_cache(cache_payload)
        return self.get_track_global_normalization()

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
        window["abs_radial_speed_mps"] = np.abs(window["radial_speed_mps"].to_numpy(dtype=np.float32))
        window["abs_speed_delta_mps"] = np.abs(window["speed_delta_mps"].to_numpy(dtype=np.float32))
        window["acceleration_mps2"] = window["speed_delta_mps"].to_numpy(dtype=np.float32) / deltas
        window["turn_rate_degps"] = window["delta_azimuth_deg"].to_numpy(dtype=np.float32) / deltas
        window["jerk_mps3"] = window["acceleration_mps2"].diff().fillna(0.0).to_numpy(dtype=np.float32) / deltas
        window["turn_acceleration_degps2"] = window["turn_rate_degps"].diff().fillna(0.0).to_numpy(dtype=np.float32) / deltas
        window["snr_delta_db"] = window["snr_db"].diff().fillna(0.0)

        smoothing_window = max(2, min(5, len(window)))
        rolling_range = window["range_m"].rolling(window=smoothing_window, min_periods=1, center=True).mean()
        window["range_jitter_m"] = window["range_m"] - rolling_range

        clip_range_values = window["range_m"].to_numpy(dtype=np.float32)
        clip_speed_values = window["radial_speed_mps"].to_numpy(dtype=np.float32)
        clip_azimuth_values = window["azimuth_deg"].to_numpy(dtype=np.float32)
        clip_snr_values = window["snr_db"].to_numpy(dtype=np.float32)

        clip_range_mean = float(np.mean(clip_range_values)) if len(clip_range_values) else 0.0
        clip_range_span = float(np.max(clip_range_values) - np.min(clip_range_values)) if len(clip_range_values) else 0.0
        clip_speed_mean = float(np.mean(clip_speed_values)) if len(clip_speed_values) else 0.0
        clip_speed_std = float(np.std(clip_speed_values)) if len(clip_speed_values) else 0.0
        clip_speed_span = float(np.max(clip_speed_values) - np.min(clip_speed_values)) if len(clip_speed_values) else 0.0
        clip_abs_speed_values = np.abs(clip_speed_values)
        clip_abs_speed_mean = float(np.mean(clip_abs_speed_values)) if len(clip_abs_speed_values) else 0.0
        clip_abs_speed_std = float(np.std(clip_abs_speed_values)) if len(clip_abs_speed_values) else 0.0
        clip_abs_speed_max = float(np.max(clip_abs_speed_values)) if len(clip_abs_speed_values) else 0.0
        clip_speed_direction_consistency = abs(clip_speed_mean) / max(clip_abs_speed_mean, 1e-6)
        clip_azimuth_std = float(np.std(clip_azimuth_values)) if len(clip_azimuth_values) else 0.0
        clip_snr_mean = float(np.mean(clip_snr_values)) if len(clip_snr_values) else 0.0
        clip_abs_acceleration_mean = float(np.mean(np.abs(window["acceleration_mps2"].to_numpy(dtype=np.float32)))) if len(window) else 0.0
        clip_abs_turn_rate_mean = float(np.mean(np.abs(window["turn_rate_degps"].to_numpy(dtype=np.float32)))) if len(window) else 0.0
        clip_abs_jerk_mean = float(np.mean(np.abs(window["jerk_mps3"].to_numpy(dtype=np.float32)))) if len(window) else 0.0
        clip_abs_turn_acceleration_mean = float(np.mean(np.abs(window["turn_acceleration_degps2"].to_numpy(dtype=np.float32)))) if len(window) else 0.0
        clip_range_jitter_std = float(np.std(window["range_jitter_m"].to_numpy(dtype=np.float32))) if len(window) else 0.0
        if len(clip_speed_values) > 1:
            speed_signs = np.sign(clip_speed_values)
            clip_speed_zero_crossing_rate = float(np.mean(speed_signs[1:] * speed_signs[:-1] < 0))
        else:
            clip_speed_zero_crossing_rate = 0.0
        delta_range_values = window["delta_range_m"].to_numpy(dtype=np.float32)
        total_range_travel = float(np.sum(np.abs(delta_range_values))) if len(delta_range_values) else 0.0
        net_range_change = float(abs(window["range_m"].iloc[-1] - window["range_m"].iloc[0])) if len(window) else 0.0
        clip_range_monotonicity = net_range_change / max(total_range_travel, 1e-6)

        window["clip_range_mean_m"] = clip_range_mean
        window["clip_range_span_m"] = clip_range_span
        window["clip_speed_mean_mps"] = clip_speed_mean
        window["clip_speed_std_mps"] = clip_speed_std
        window["clip_speed_span_mps"] = clip_speed_span
        window["clip_abs_speed_mean_mps"] = clip_abs_speed_mean
        window["clip_abs_speed_std_mps"] = clip_abs_speed_std
        window["clip_abs_speed_max_mps"] = clip_abs_speed_max
        window["clip_speed_direction_consistency"] = clip_speed_direction_consistency
        window["clip_range_monotonicity"] = clip_range_monotonicity
        window["clip_abs_acceleration_mean_mps2"] = clip_abs_acceleration_mean
        window["clip_abs_turn_rate_mean_degps"] = clip_abs_turn_rate_mean
        window["clip_abs_jerk_mean_mps3"] = clip_abs_jerk_mean
        window["clip_abs_turn_acceleration_mean_degps2"] = clip_abs_turn_acceleration_mean
        window["clip_range_jitter_std_m"] = clip_range_jitter_std
        window["clip_speed_zero_crossing_rate"] = clip_speed_zero_crossing_rate
        window["clip_azimuth_std_deg"] = clip_azimuth_std
        window["clip_snr_mean_db"] = clip_snr_mean
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
        meta_file = session_dir / "meta_session.json"

        if not track_csv.exists() or not track_clip_file.exists() or not matched_file.exists():
            return None, []

        fingerprint = self._build_session_fingerprint(track_csv, track_clip_file, matched_file, rd_clip_file)
        source_name = 'Unknown'
        session_label_name = None
        if meta_file.exists():
            try:
                meta_payload = json.loads(meta_file.read_text(encoding='utf-8'))
                source_name = str(meta_payload.get('source_name') or meta_payload.get('source') or 'Unknown')
                if self.label_granularity == 'detail':
                    session_label_name = str(meta_payload.get('label_detail') or meta_payload.get('label_name') or meta_payload.get('label_main') or '')
                else:
                    session_label_name = str(meta_payload.get('label_coarse') or meta_payload.get('label_name') or meta_payload.get('label_main') or '')
            except Exception:
                source_name = 'Unknown'
                session_label_name = None
        cached_session = self._dataset_index_cache.get('sessions', {}).get(session_dir.name)
        records: list[ClipRecord] = []
        if isinstance(cached_session, dict) and cached_session.get('fingerprint') == fingerprint:
            for item in cached_session.get('records', []):
                cached_label_name = str(item.get('label_name', ''))
                if session_label_name:
                    coarse_like = str(cached_label_name).split('__', 1)[0]
                    desired_coarse = str(session_label_name).split('__', 1)[0]
                    if coarse_like == desired_coarse:
                        cached_label_name = str(session_label_name)
                records.append(
                    ClipRecord(
                        session_id=session_dir.name,
                        sample_id=str(item['sample_id']),
                        track_id=int(item['track_id']),
                        label_name=cached_label_name,
                        source_name=str(item.get('source_name', source_name)),
                        t0=float(item['t0']),
                        t1=float(item['t1']),
                        rd_key=f"rd_clip_{item['sample_id']}",
                        rd_path=rd_clip_file if rd_clip_file.exists() else None,
                        has_rd=bool(item['has_rd']),
                    )
                )
        else:
            label_by_track = {
                int(item["track_id"]): str(session_label_name or item["label_main"])
                for item in self._load_jsonl(matched_file)
                if "track_id" in item and "label_main" in item
            }
            if not label_by_track:
                return None, []

            available_rd_keys: set[str] = set()
            if self.include_rd and rd_clip_file.exists():
                with h5py.File(rd_clip_file, "r") as handle:
                    available_rd_keys = set(handle.keys())

            cache_records: list[dict[str, Any]] = []
            for clip in self._load_jsonl(track_clip_file):
                track_id = int(clip["track_id"])
                label_name = label_by_track.get(track_id)
                if label_name is None:
                    continue

                sample_id = str(clip["sample_id"])
                rd_key = f"rd_clip_{sample_id}"
                has_rd = self.include_rd and rd_key in available_rd_keys
                records.append(
                    ClipRecord(
                        session_id=session_dir.name,
                        sample_id=sample_id,
                        track_id=track_id,
                        label_name=label_name,
                        source_name=source_name,
                        t0=float(clip["t0"]),
                        t1=float(clip["t1"]),
                        rd_key=rd_key,
                        rd_path=rd_clip_file if rd_clip_file.exists() else None,
                        has_rd=has_rd,
                    )
                )
                cache_records.append(
                    {
                        'sample_id': sample_id,
                        'track_id': int(track_id),
                        'label_name': label_name,
                        'source_name': source_name,
                        't0': float(clip['t0']),
                        't1': float(clip['t1']),
                        'has_rd': bool(has_rd),
                    }
                )
            self._dataset_index_cache.setdefault('sessions', {})[session_dir.name] = {
                'fingerprint': fingerprint,
                'records': cache_records,
            }
            self._dataset_index_cache_dirty = True

        if not records:
            return None, []

        rd_data_file = session_dir / "radar/rd/rd_data.h5"
        return SessionContext(track_csv_path=track_csv, rd_data_path=rd_data_file if rd_data_file.exists() else None, source_name=source_name), records

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

    def _normalize_track_sequence(self, array: np.ndarray, source_name: str | None = None) -> np.ndarray:
        center = self.track_feature_center
        scale = self.track_feature_scale
        if source_name is not None and source_name in self.track_feature_center_by_source and source_name in self.track_feature_scale_by_source:
            center = self.track_feature_center_by_source[source_name]
            scale = self.track_feature_scale_by_source[source_name]
        normalized = (array - center) / scale
        normalized = np.clip(normalized, -8.0, 8.0)
        return normalized.astype(np.float32)

    def _build_track_global_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        feature_blocks: list[np.ndarray] = [self._extract_track_global_features(record) for record in self.samples]
        stacked = np.stack(feature_blocks, axis=0) if feature_blocks else np.zeros((1, len(self.global_feature_columns)), dtype=np.float32)
        center = np.median(stacked, axis=0, keepdims=True)
        q25 = np.percentile(stacked, 25, axis=0, keepdims=True)
        q75 = np.percentile(stacked, 75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-3] = 1.0
        return center.astype(np.float32), scale.astype(np.float32)

    def _normalize_track_global_features(self, feature_vector: np.ndarray, source_name: str | None = None) -> np.ndarray:
        if feature_vector.size == 0:
            return feature_vector.astype(np.float32)
        center = self.track_global_center.reshape(-1)
        scale = self.track_global_scale.reshape(-1)
        if source_name is not None and source_name in self.track_global_center_by_source and source_name in self.track_global_scale_by_source:
            center = self.track_global_center_by_source[source_name].reshape(-1)
            scale = self.track_global_scale_by_source[source_name].reshape(-1)
        normalized = (feature_vector - center) / scale
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
        local_doppler_symmetric = 0.5 * (mid_log_amplitude + mid_log_amplitude[:, :, ::-1])

        if global_rd_cube is None:
            global_log_amplitude = np.zeros_like(mid_log_amplitude, dtype=np.float32)
            global_background_suppressed = np.zeros_like(mid_log_amplitude, dtype=np.float32)
            global_temporal_difference = np.zeros_like(mid_log_amplitude, dtype=np.float32)
            global_contrast = np.zeros_like(mid_log_amplitude, dtype=np.float32)
            global_doppler_symmetric = np.zeros_like(mid_log_amplitude, dtype=np.float32)
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
            global_doppler_symmetric = 0.5 * (global_log_amplitude + global_log_amplitude[:, :, ::-1])

        if len(scale_cubes) == 1:
            channels = [
                real_component,
                imag_component,
                mid_log_amplitude,
                temporal_difference,
                background_suppressed,
                local_contrast,
                local_doppler_symmetric,
                phase_difference,
                coherence,
                global_log_amplitude,
                global_background_suppressed,
                global_doppler_symmetric,
                global_temporal_difference,
                global_contrast,
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
                local_doppler_symmetric,
                phase_difference,
                coherence,
                global_log_amplitude,
                global_background_suppressed,
                global_doppler_symmetric,
                global_temporal_difference,
                global_contrast,
            ]
        selected = channels[: self.rd_input_channels]
        while len(selected) < self.rd_input_channels:
            selected.append(np.zeros_like(mid_log_amplitude, dtype=np.float32))
        return self._standardize_channel_first(np.stack(selected, axis=0))

    def _load_global_rd_cube(self, record: ClipRecord) -> np.ndarray | None:
        context = self.sessions[record.session_id]
        if context.rd_data_path is None or not context.rd_data_path.exists():
            return None

        track_frame = self._get_track_frame(record.session_id)
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

            sorted_order = np.argsort(frame_indices, kind="stable")
            sorted_indices = frame_indices[sorted_order]
            unique_indices, inverse_indices = np.unique(sorted_indices, return_inverse=True)
            unique_rd_frames = self._sanitize_rd_array(np.asarray(rd_frames_dataset[unique_indices]))
            sorted_rd_cube = unique_rd_frames[inverse_indices]
            restore_order = np.argsort(sorted_order, kind="stable")
            global_rd_cube = sorted_rd_cube[restore_order]

        if global_rd_cube.ndim == 2:
            global_rd_cube = global_rd_cube[None, :, :]
        global_rd_cube = self._resample_frames(global_rd_cube, self.rd_frames)
        global_rd_cube = self._resize_full_context(global_rd_cube, self.rd_height, self.rd_width)
        return global_rd_cube

    def _build_track_window(self, record: ClipRecord) -> pd.DataFrame:
        track_frame = self._get_track_frame(record.session_id)
        window = track_frame[
            (track_frame["track_id"] == record.track_id)
            & (track_frame["t"] >= record.t0)
            & (track_frame["t"] < record.t1)
        ].copy()

        if window.empty:
            window = track_frame[track_frame["track_id"] == record.track_id].copy()

        return self._augment_track_features(window)

    def _extract_track_global_features(self, record: ClipRecord) -> np.ndarray:
        if not self.global_feature_columns:
            return np.zeros((0,), dtype=np.float32)
        cached = self.track_global_feature_cache.get(record.sample_id)
        if cached is not None:
            return cached.copy()
        window = self._build_track_window(record)
        if window.empty:
            feature_vector = np.zeros((len(self.global_feature_columns),), dtype=np.float32)
            self.track_global_feature_cache[record.sample_id] = feature_vector
            return feature_vector.copy()
        for column in self.global_feature_columns:
            if column not in window.columns:
                window[column] = 0.0
        first_row = window.iloc[0]
        values: list[float] = []
        for column in self.global_feature_columns:
            value = first_row.get(column, 0.0)
            if hasattr(value, "iloc"):
                value = value.iloc[0]
            values.append(float(value))
        feature_vector = np.asarray(values, dtype=np.float32)
        self.track_global_feature_cache[record.sample_id] = feature_vector
        return feature_vector.copy()

    def _load_track_sequence(self, record: ClipRecord, target_length: int | None = None) -> np.ndarray:
        window = self._build_track_window(record)

        for column in self.feature_columns:
            if column not in window.columns:
                window[column] = 0.0

        sequence = window.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        sequence = self._resample_sequence(sequence, target_length or self.sequence_length)
        return self._normalize_track_sequence(sequence, source_name=record.source_name)

    def _load_track_global_features(self, record: ClipRecord) -> np.ndarray:
        return self._normalize_track_global_features(self._extract_track_global_features(record))

    def _load_rd_cube(self, record: ClipRecord) -> np.ndarray:
        if not self.include_rd or not record.has_rd or record.rd_path is None:
            return np.zeros((self.rd_input_channels, self.rd_frames, self.rd_height, self.rd_width), dtype=np.float32)

        with h5py.File(record.rd_path, "r") as handle:
            rd_cube = self._sanitize_rd_array(np.asarray(handle[record.rd_key][:]))

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
        track_mid = self._load_track_sequence(record, self.sequence_length)
        track_short = self._load_track_sequence(record, self.track_short_length)
        track_long = self._load_track_sequence(record, self.track_long_length)
        mix_enabled = self.window_mix_enabled_indices is None or index in self.window_mix_enabled_indices
        if self.include_rd and mix_enabled and np.random.rand() < self.window_mix_probability:
            mix_weight = float(np.random.uniform(0.35, 0.65))
            mid_projection = self._resample_sequence(track_mid, self.track_long_length)
            track_long = (mix_weight * track_long + (1.0 - mix_weight) * mid_projection).astype(np.float32)
        bird_surface_target = np.float32(
            1.0 if record.label_name == "BirdTarget" else 0.0 if record.label_name == "ShipTarget" else -1.0
        )
        bird_rd_target = np.float32(
            1.0 if record.label_name == "BirdTarget" else 0.0 if record.label_name in {"AircraftTarget", "DroneTarget", "ShipTarget"} else -1.0
        )
        ship_drone_target = np.float32(
            1.0 if record.label_name == "ShipTarget" else 0.0 if record.label_name == "DroneTarget" else -1.0
        )
        ship_bird_target = np.float32(
            1.0 if record.label_name == "ShipTarget" else 0.0 if record.label_name == "BirdTarget" else -1.0
        )
        ship_air_target = np.float32(
            1.0 if record.label_name == "ShipTarget" else 0.0 if record.label_name in {"AircraftTarget", "BirdTarget", "DroneTarget"} else -1.0
        )
        track_router_target = np.int64(
            0 if record.label_name == "ShipTarget" else 1 if record.label_name == "AircraftTarget" else 2 if record.label_name in {"BirdTarget", "DroneTarget"} else -1
        )
        raw_global_features = self._extract_track_global_features(record)
        global_features = self._normalize_track_global_features(raw_global_features, source_name=record.source_name)
        global_feature_lookup = {name: idx for idx, name in enumerate(self.global_feature_columns)}
        clip_speed_mean = float(raw_global_features[global_feature_lookup["clip_speed_mean_mps"]]) if "clip_speed_mean_mps" in global_feature_lookup else 0.0
        clip_speed_direction_consistency = float(raw_global_features[global_feature_lookup["clip_speed_direction_consistency"]]) if "clip_speed_direction_consistency" in global_feature_lookup else 0.0
        clip_range_monotonicity = float(raw_global_features[global_feature_lookup["clip_range_monotonicity"]]) if "clip_range_monotonicity" in global_feature_lookup else 0.0
        clip_snr_mean = float(raw_global_features[global_feature_lookup["clip_snr_mean_db"]]) if "clip_snr_mean_db" in global_feature_lookup else 0.0
        suspicious_ship_hard_pattern = np.float32(
            1.0
            if (
                record.label_name == "ShipTarget"
                and (
                    record.session_id in self.conditional_ship_drone_session_ids
                    or (
                        abs(clip_speed_mean) >= 3.0
                        and clip_speed_direction_consistency >= 0.95
                        and clip_range_monotonicity <= 0.9
                        and clip_snr_mean >= 45.0
                    )
                )
            )
            else 0.0
        )
        return {
            "sample_id": record.sample_id,
            "track_features": track_mid,
            "track_short_features": track_short,
            "track_long_features": track_long,
            "track_global_features": global_features,
            "rd_features": self._load_rd_cube(record),
            "label": np.int64(self.label_to_index[record.label_name]),
            "label_name": record.label_name,
            "source_name": record.source_name,
            "has_rd": np.int64(record.has_rd),
            "bird_rd_target": bird_rd_target,
            "bird_surface_target": bird_surface_target,
            "ship_drone_target": ship_drone_target,
            "ship_bird_target": ship_bird_target,
            "ship_air_target": ship_air_target,
            "track_router_target": track_router_target,
            "suspicious_ship_hard_pattern": suspicious_ship_hard_pattern,
        }
