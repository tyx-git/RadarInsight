from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from src.training.config import TrainingConfig


@dataclass(slots=True)
class TsvClipRecord:
    sample_id: str
    session_id: str
    track_id: str
    object_key: str
    label_name: str
    t0_ms: int
    t1_ms: int
    notes_name: str


@dataclass(slots=True)
class TsvSessionContext:
    track_frame: pd.DataFrame
    labels_frame: pd.DataFrame
    notes_frame: pd.DataFrame


class TrackOnlyTsvDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        config: TrainingConfig,
        session_ids: Sequence[str] | None = None,
        label_order: Sequence[str] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.config = config
        self.sequence_length = config.sequence_length
        self.track_short_length = config.track_short_length
        self.track_mid_length = config.track_mid_length
        self.track_long_length = config.track_long_length
        self.track_window_points = config.track_window_points
        self.track_window_stride_points = config.track_window_stride_points
        self.track_min_window_points = config.track_min_window_points
        self.feature_columns = tuple(config.track_feature_columns)
        self.global_feature_columns = tuple(config.track_global_feature_columns)
        self.label_granularity = str(getattr(config, "label_granularity", "coarse"))

        self.sessions: dict[str, TsvSessionContext] = {}
        self.label_order = tuple(label_order) if label_order is not None else None
        self.samples: list[TsvClipRecord] = []
        self.label_to_index: dict[str, int] = {}
        self.index_to_label: dict[int, str] = {}
        self.note_to_index: dict[str, int] = {}
        self.index_to_note: dict[int, str] = {}

        self._load_sessions(session_ids=session_ids)
        self._build_label_index()
        self._build_note_index()
        self.track_feature_center, self.track_feature_scale = self._build_track_feature_stats()
        self.track_global_center, self.track_global_scale = self._build_track_global_feature_stats()

    @property
    def global_feature_dim(self) -> int:
        return len(self.global_feature_columns)

    def get_track_feature_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        return self.track_feature_center.copy(), self.track_feature_scale.copy()

    def set_track_feature_normalization(self, center: Sequence[float], scale: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        self.track_feature_center = np.asarray(center, dtype=np.float32)
        self.track_feature_scale = np.asarray(scale, dtype=np.float32)
        return self.get_track_feature_normalization()

    def get_track_global_normalization(self) -> tuple[np.ndarray, np.ndarray]:
        return self.track_global_center.copy(), self.track_global_scale.copy()

    def set_track_global_normalization(self, center: Sequence[float], scale: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        self.track_global_center = np.asarray(center, dtype=np.float32)
        self.track_global_scale = np.asarray(scale, dtype=np.float32)
        return self.get_track_global_normalization()


    def set_track_normalization(self, center: Sequence[float], scale: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        self.track_feature_center = np.asarray(center, dtype=np.float32)
        self.track_feature_scale = np.asarray(scale, dtype=np.float32)
        return self.get_track_feature_normalization()

    def fit_track_normalization_for_sessions(self, session_ids: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        feature_blocks: list[np.ndarray] = []
        selected_session_ids = set(session_ids) if session_ids is not None else None
        for record in self.samples:
            if selected_session_ids is not None and record.session_id not in selected_session_ids:
                continue
            window = self._build_track_window(record)
            for column in self.feature_columns:
                if column not in window.columns:
                    window[column] = 0.0
            feature_blocks.append(window.loc[:, self.feature_columns].to_numpy(dtype=np.float32))
        stacked = np.concatenate(feature_blocks, axis=0) if feature_blocks else np.zeros((1, len(self.feature_columns)), dtype=np.float32)
        center = np.median(stacked, axis=0, keepdims=True)
        q25 = np.percentile(stacked, 25, axis=0, keepdims=True)
        q75 = np.percentile(stacked, 75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-3] = 1.0
        self.track_feature_center = center.astype(np.float32)
        self.track_feature_scale = scale.astype(np.float32)
        return self.get_track_feature_normalization()

    def fit_track_global_normalization_for_sessions(self, session_ids: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        feature_blocks: list[np.ndarray] = []
        selected_session_ids = set(session_ids) if session_ids is not None else None
        for record in self.samples:
            if selected_session_ids is not None and record.session_id not in selected_session_ids:
                continue
            feature_blocks.append(self._extract_track_global_features(record))
        stacked = np.stack(feature_blocks, axis=0) if feature_blocks else np.zeros((1, len(self.global_feature_columns)), dtype=np.float32)
        center = np.median(stacked, axis=0, keepdims=True)
        q25 = np.percentile(stacked, 25, axis=0, keepdims=True)
        q75 = np.percentile(stacked, 75, axis=0, keepdims=True)
        scale = q75 - q25
        scale[scale < 1e-3] = 1.0
        self.track_global_center = center.astype(np.float32)
        self.track_global_scale = scale.astype(np.float32)
        return self.get_track_global_normalization()

    def build_session_index(self) -> dict[str, list[int]]:
        session_to_indices: dict[str, list[int]] = {}
        for index, record in enumerate(self.samples):
            session_to_indices.setdefault(record.session_id, []).append(index)
        return session_to_indices

    def _resolve_session_dirs(self, session_ids: Sequence[str] | None) -> list[Path]:
        if session_ids:
            return [self.data_root / session_id for session_id in session_ids if (self.data_root / session_id).exists()]
        return sorted(path for path in self.data_root.iterdir() if path.is_dir())

    def _load_sessions(self, session_ids: Sequence[str] | None) -> None:
        for session_dir in self._resolve_session_dirs(session_ids):
            track_points_file = session_dir / 'track_points.tsv'
            labels_file = session_dir / 'track_labels.tsv'
            notes_file = session_dir / 'track_notes.tsv'
            if not track_points_file.exists() or not labels_file.exists() or not notes_file.exists():
                continue

            track_frame = pd.read_csv(track_points_file, sep='	')
            labels_frame = pd.read_csv(labels_file, sep='	')
            notes_frame = pd.read_csv(notes_file, sep='	')
            if track_frame.empty or labels_frame.empty:
                continue

            track_frame['track_id'] = track_frame['track_id'].astype(str)
            track_frame['object_key'] = track_frame['object_key'].astype(str)
            track_frame['timestamp_ms'] = track_frame['timestamp_ms'].astype(np.int64)
            track_frame = track_frame.sort_values(['track_id', 'timestamp_ms'], kind='stable').reset_index(drop=True)

            labels_frame['track_id'] = labels_frame['track_id'].astype(str)
            labels_frame['object_key'] = labels_frame['object_key'].astype(str)
            notes_frame['track_id'] = notes_frame['track_id'].astype(str)
            notes_frame['object_key'] = notes_frame['object_key'].astype(str)

            session_id = session_dir.name
            self.sessions[session_id] = TsvSessionContext(
                track_frame=track_frame,
                labels_frame=labels_frame,
                notes_frame=notes_frame,
            )
            self._build_session_samples(session_id)

    def _build_session_samples(self, session_id: str) -> None:
        context = self.sessions[session_id]
        notes_lookup = {
            (str(row['track_id']), str(row['object_key'])): str(row.get('notes_name', ''))
            for _, row in context.notes_frame.iterrows()
        }
        for _, row in context.labels_frame.iterrows():
            if int(row.get('is_valid', 1)) != 1:
                continue
            track_id = str(row['track_id'])
            object_key = str(row['object_key'])
            fine_label_name = str(row['label_name'])
            label_name = str(self.config.coarse_label_map.get(fine_label_name, fine_label_name)) if self.label_granularity == 'coarse' else fine_label_name
            track_points = context.track_frame[
                (context.track_frame['track_id'] == track_id)
                & (context.track_frame['object_key'] == object_key)
            ].copy()
            if track_points.empty:
                continue
            times = track_points['timestamp_ms'].to_numpy(dtype=np.int64)
            if times.size == 0:
                continue

            num_points = len(track_points)
            if num_points < self.track_min_window_points:
                continue

            if num_points <= self.track_window_points:
                windows = [(0, num_points)]
            else:
                windows = []
                start_index = 0
                while start_index + self.track_window_points <= num_points:
                    windows.append((start_index, start_index + self.track_window_points))
                    start_index += self.track_window_stride_points
                if windows and windows[-1][1] < num_points:
                    windows.append((num_points - self.track_window_points, num_points))

            for window_index, (start_index, end_index) in enumerate(windows):
                clip_frame = track_points.iloc[start_index:end_index].copy()
                clip_times = clip_frame['timestamp_ms'].to_numpy(dtype=np.int64)
                self.samples.append(
                    TsvClipRecord(
                        sample_id=f'{session_id}:{object_key}:{track_id}:{window_index}',
                        session_id=session_id,
                        track_id=track_id,
                        object_key=object_key,
                        label_name=label_name,
                        t0_ms=int(clip_times.min()),
                        t1_ms=int(clip_times.max()),
                        notes_name=notes_lookup.get((track_id, object_key), ''),
                    )
                )



    def _build_note_index(self) -> None:
        note_values = sorted({note for record in self.samples for note in record.notes_name.split(';') if note})
        self.note_to_index = {note: idx for idx, note in enumerate(note_values)}
        self.index_to_note = {idx: note for note, idx in self.note_to_index.items()}

    @property
    def note_dim(self) -> int:
        return len(self.note_to_index)

    def _build_note_target(self, record: TsvClipRecord) -> np.ndarray:
        target = np.zeros((len(self.note_to_index),), dtype=np.float32)
        for note in [part for part in record.notes_name.split(';') if part]:
            index = self.note_to_index.get(note)
            if index is not None:
                target[index] = 1.0
        return target

    def _build_label_index(self) -> None:
        labels = tuple(self.label_order) if self.label_order is not None else tuple(sorted({record.label_name for record in self.samples}))
        self.label_to_index = {label: idx for idx, label in enumerate(labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def _augment_track_features(self, window: pd.DataFrame) -> pd.DataFrame:
        window = window.copy()
        window['range_m'] = window['range_m'].astype(np.float32)
        window['azimuth_deg'] = window['azimuth_deg'].astype(np.float32)
        window['radial_speed_mps'] = window['doppler_speed_mps'].astype(np.float32)
        window['abs_radial_speed_mps'] = np.abs(window['radial_speed_mps'])
        window['speed_mps'] = window['speed_mps'].astype(np.float32)
        window['height_m'] = window['height_m'].astype(np.float32)
        window['elevation_deg'] = window['elevation_deg'].astype(np.float32)
        window['course_deg'] = window['course_deg'].astype(np.float32)
        window['amplitude_db'] = window['amplitude_db'].astype(np.float32)
        window['noise_db'] = window['noise_db'].astype(np.float32)
        window['snr_db'] = window['snr_db'].astype(np.float32)
        time_s = window['timestamp_ms'].astype(np.float64).to_numpy() / 1000.0
        dt = np.diff(time_s, prepend=time_s[0])
        dt[dt <= 0.0] = np.median(dt[dt > 0.0]) if np.any(dt > 0.0) else 0.5

        range_values = window['range_m'].to_numpy(dtype=np.float32)
        azimuth_values = window['azimuth_deg'].to_numpy(dtype=np.float32)
        radial_speed_values = window['radial_speed_mps'].to_numpy(dtype=np.float32)
        speed_values = window['speed_mps'].to_numpy(dtype=np.float32)
        height_values = window['height_m'].to_numpy(dtype=np.float32)
        elevation_values = window['elevation_deg'].to_numpy(dtype=np.float32)
        course_values = window['course_deg'].to_numpy(dtype=np.float32)
        amplitude_values = window['amplitude_db'].to_numpy(dtype=np.float32)
        noise_values = window['noise_db'].to_numpy(dtype=np.float32)
        snr_values = window['snr_db'].to_numpy(dtype=np.float32)

        delta_range = np.diff(range_values, prepend=range_values[0])
        delta_azimuth = np.diff(azimuth_values, prepend=azimuth_values[0])
        speed_delta = np.diff(radial_speed_values, prepend=radial_speed_values[0])
        abs_speed_delta = np.abs(speed_delta)
        acceleration = speed_delta / dt
        jerk = np.diff(acceleration, prepend=acceleration[0]) / dt
        turn_rate = delta_azimuth / dt
        turn_acceleration = np.diff(turn_rate, prepend=turn_rate[0]) / dt
        snr_delta = np.diff(snr_values, prepend=snr_values[0])
        height_delta = np.diff(height_values, prepend=height_values[0])
        elevation_delta = np.diff(elevation_values, prepend=elevation_values[0])
        course_delta = np.diff(course_values, prepend=course_values[0])
        amplitude_delta = np.diff(amplitude_values, prepend=amplitude_values[0])
        noise_delta = np.diff(noise_values, prepend=noise_values[0])

        smoothing_window = max(2, min(5, len(window)))
        rolling_range = pd.Series(range_values).rolling(window=smoothing_window, min_periods=1, center=True).mean().to_numpy(dtype=np.float32)
        range_jitter = range_values - rolling_range

        clip_range_mean = float(np.mean(range_values)) if len(range_values) else 0.0
        clip_range_span = float(np.max(range_values) - np.min(range_values)) if len(range_values) else 0.0
        clip_speed_values = radial_speed_values
        clip_speed_mean = float(np.mean(clip_speed_values)) if len(clip_speed_values) else 0.0
        clip_speed_std = float(np.std(clip_speed_values)) if len(clip_speed_values) else 0.0
        clip_speed_span = float(np.max(clip_speed_values) - np.min(clip_speed_values)) if len(clip_speed_values) else 0.0
        clip_abs_speed_values = np.abs(clip_speed_values)
        clip_abs_speed_mean = float(np.mean(clip_abs_speed_values)) if len(clip_abs_speed_values) else 0.0
        clip_abs_speed_std = float(np.std(clip_abs_speed_values)) if len(clip_abs_speed_values) else 0.0
        clip_abs_speed_max = float(np.max(clip_abs_speed_values)) if len(clip_abs_speed_values) else 0.0
        clip_speed_direction_consistency = abs(clip_speed_mean) / max(clip_abs_speed_mean, 1e-6)
        clip_azimuth_std = float(np.std(azimuth_values)) if len(azimuth_values) else 0.0
        clip_snr_mean = float(np.mean(snr_values)) if len(snr_values) else 0.0
        clip_height_mean = float(np.mean(height_values)) if len(height_values) else 0.0
        clip_height_span = float(np.max(height_values) - np.min(height_values)) if len(height_values) else 0.0
        clip_elevation_mean = float(np.mean(elevation_values)) if len(elevation_values) else 0.0
        clip_elevation_std = float(np.std(elevation_values)) if len(elevation_values) else 0.0
        clip_course_std = float(np.std(course_values)) if len(course_values) else 0.0
        clip_amplitude_mean = float(np.mean(amplitude_values)) if len(amplitude_values) else 0.0
        clip_noise_mean = float(np.mean(noise_values)) if len(noise_values) else 0.0
        clip_abs_acceleration_mean = float(np.mean(np.abs(acceleration))) if len(acceleration) else 0.0
        clip_abs_turn_rate_mean = float(np.mean(np.abs(turn_rate))) if len(turn_rate) else 0.0
        clip_abs_jerk_mean = float(np.mean(np.abs(jerk))) if len(jerk) else 0.0
        clip_abs_turn_acceleration_mean = float(np.mean(np.abs(turn_acceleration))) if len(turn_acceleration) else 0.0
        clip_range_jitter_std = float(np.std(range_jitter)) if len(range_jitter) else 0.0
        speed_signs = np.sign(clip_speed_values)
        clip_speed_zero_crossing_rate = float(np.mean(speed_signs[1:] * speed_signs[:-1] < 0)) if len(speed_signs) > 1 else 0.0
        total_range_travel = float(np.sum(np.abs(delta_range))) if len(delta_range) else 0.0
        net_range_change = float(abs(range_values[-1] - range_values[0])) if len(range_values) else 0.0
        clip_range_monotonicity = net_range_change / max(total_range_travel, 1e-6)

        def _spectral_summary(values: np.ndarray, dt_values: np.ndarray) -> tuple[float, float, float, float]:
            if len(values) < 4:
                return 0.0, 0.0, 0.0, 0.0
            centered = values - np.mean(values)
            fft_values = np.fft.rfft(centered.astype(np.float32))
            power = np.abs(fft_values) ** 2
            if power.size <= 1:
                return 0.0, 0.0, 0.0, 0.0
            dt_median = float(np.median(dt_values[dt_values > 0])) if np.any(dt_values > 0) else 0.5
            freqs = np.fft.rfftfreq(len(centered), d=dt_median)
            power = power[1:]
            freqs = freqs[1:]
            if power.size == 0:
                return 0.0, 0.0, 0.0, 0.0
            total = float(np.sum(power)) if np.sum(power) > 0 else 1.0
            low = float(np.sum(power[freqs < 0.5]) / total)
            mid = float(np.sum(power[(freqs >= 0.5) & (freqs < 2.0)]) / total)
            high = float(np.sum(power[freqs >= 2.0]) / total)
            dominant = float(freqs[int(np.argmax(power))]) if power.size else 0.0
            return low, mid, high, dominant

        speed_low, speed_mid, speed_high, speed_dom = _spectral_summary(clip_speed_values, dt)
        range_low, range_mid, range_high, range_dom = _spectral_summary(range_values, dt)

        window['delta_range_m'] = delta_range
        window['delta_azimuth_deg'] = delta_azimuth
        window['speed_delta_mps'] = speed_delta
        window['abs_speed_delta_mps'] = abs_speed_delta
        window['acceleration_mps2'] = acceleration
        window['turn_rate_degps'] = turn_rate
        window['jerk_mps3'] = jerk
        window['turn_acceleration_degps2'] = turn_acceleration
        window['range_jitter_m'] = range_jitter
        window['snr_delta_db'] = snr_delta
        window['height_delta_m'] = height_delta
        window['elevation_delta_deg'] = elevation_delta
        window['course_delta_deg'] = course_delta
        window['amplitude_delta_db'] = amplitude_delta
        window['noise_delta_db'] = noise_delta
        window['clip_range_mean_m'] = clip_range_mean
        window['clip_range_span_m'] = clip_range_span
        window['clip_speed_mean_mps'] = clip_speed_mean
        window['clip_speed_std_mps'] = clip_speed_std
        window['clip_speed_span_mps'] = clip_speed_span
        window['clip_abs_speed_mean_mps'] = clip_abs_speed_mean
        window['clip_abs_speed_std_mps'] = clip_abs_speed_std
        window['clip_abs_speed_max_mps'] = clip_abs_speed_max
        window['clip_speed_direction_consistency'] = clip_speed_direction_consistency
        window['clip_range_monotonicity'] = clip_range_monotonicity
        window['clip_abs_acceleration_mean_mps2'] = clip_abs_acceleration_mean
        window['clip_abs_turn_rate_mean_degps'] = clip_abs_turn_rate_mean
        window['clip_abs_jerk_mean_mps3'] = clip_abs_jerk_mean
        window['clip_abs_turn_acceleration_mean_degps2'] = clip_abs_turn_acceleration_mean
        window['clip_range_jitter_std_m'] = clip_range_jitter_std
        window['clip_speed_zero_crossing_rate'] = clip_speed_zero_crossing_rate
        window['clip_azimuth_std_deg'] = clip_azimuth_std
        window['clip_snr_mean_db'] = clip_snr_mean
        window['clip_height_mean_m'] = clip_height_mean
        window['clip_height_span_m'] = clip_height_span
        window['clip_elevation_mean_deg'] = clip_elevation_mean
        window['clip_elevation_std_deg'] = clip_elevation_std
        window['clip_course_std_deg'] = clip_course_std
        window['clip_amplitude_mean_db'] = clip_amplitude_mean
        window['clip_noise_mean_db'] = clip_noise_mean
        window['clip_speed_energy_low'] = speed_low
        window['clip_speed_energy_mid'] = speed_mid
        window['clip_speed_energy_high'] = speed_high
        window['clip_range_energy_low'] = range_low
        window['clip_range_energy_mid'] = range_mid
        window['clip_range_energy_high'] = range_high
        window['clip_speed_dominant_frequency_hz'] = speed_dom
        window['clip_range_dominant_frequency_hz'] = range_dom
        return window

    def _build_track_window(self, record: TsvClipRecord) -> pd.DataFrame:
        track_frame = self.sessions[record.session_id].track_frame
        window = track_frame[
            (track_frame['track_id'] == record.track_id)
            & (track_frame['object_key'] == record.object_key)
            & (track_frame['timestamp_ms'] >= record.t0_ms)
            & (track_frame['timestamp_ms'] <= record.t1_ms)
        ].copy()
        if window.empty:
            window = track_frame[
                (track_frame['track_id'] == record.track_id)
                & (track_frame['object_key'] == record.object_key)
            ].copy()
        return self._augment_track_features(window)

    def _extract_track_global_features(self, record: TsvClipRecord) -> np.ndarray:
        window = self._build_track_window(record)
        for column in self.global_feature_columns:
            if column not in window.columns:
                window[column] = 0.0
        return window.iloc[0].loc[list(self.global_feature_columns)].to_numpy(dtype=np.float32)

    def _build_track_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        sequences = [self._build_track_window(record).loc[:, self.feature_columns].to_numpy(dtype=np.float32) for record in self.samples]
        if not sequences:
            return np.zeros((len(self.feature_columns),), dtype=np.float32), np.ones((len(self.feature_columns),), dtype=np.float32)
        stacked = np.concatenate(sequences, axis=0)
        center = np.median(stacked, axis=0).astype(np.float32)
        scale = (np.percentile(stacked, 75, axis=0) - np.percentile(stacked, 25, axis=0)).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return center, scale

    def _build_track_global_feature_stats(self) -> tuple[np.ndarray, np.ndarray]:
        feature_blocks = [self._extract_track_global_features(record) for record in self.samples]
        if not feature_blocks:
            return np.zeros((len(self.global_feature_columns),), dtype=np.float32), np.ones((len(self.global_feature_columns),), dtype=np.float32)
        stacked = np.stack(feature_blocks, axis=0)
        center = np.median(stacked, axis=0).astype(np.float32)
        scale = (np.percentile(stacked, 75, axis=0) - np.percentile(stacked, 25, axis=0)).astype(np.float32)
        scale[scale < 1e-6] = 1.0
        return center, scale

    def _normalize_track_sequence(self, sequence: np.ndarray) -> np.ndarray:
        return (sequence - self.track_feature_center.reshape(1, -1)) / self.track_feature_scale.reshape(1, -1)

    def _normalize_track_global_features(self, vector: np.ndarray) -> np.ndarray:
        return (vector - self.track_global_center.reshape(-1)) / self.track_global_scale.reshape(-1)

    def _resample_sequence(self, sequence: np.ndarray, target_length: int) -> np.ndarray:
        if sequence.shape[0] == 0:
            return np.zeros((target_length, sequence.shape[1]), dtype=np.float32)
        if sequence.shape[0] == target_length:
            return sequence
        source_positions = np.linspace(0.0, 1.0, num=sequence.shape[0], dtype=np.float32)
        target_positions = np.linspace(0.0, 1.0, num=target_length, dtype=np.float32)
        resampled = np.empty((target_length, sequence.shape[1]), dtype=np.float32)
        for column_index in range(sequence.shape[1]):
            resampled[:, column_index] = np.interp(target_positions, source_positions, sequence[:, column_index])
        return resampled

    def _load_track_sequence(self, record: TsvClipRecord, target_length: int | None = None) -> np.ndarray:
        window = self._build_track_window(record)
        for column in self.feature_columns:
            if column not in window.columns:
                window[column] = 0.0
        sequence = window.loc[:, self.feature_columns].to_numpy(dtype=np.float32)
        sequence = self._resample_sequence(sequence, target_length or self.sequence_length)
        return self._normalize_track_sequence(sequence)

    def _load_track_global_features(self, record: TsvClipRecord) -> np.ndarray:
        return self._normalize_track_global_features(self._extract_track_global_features(record))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.samples[index]
        return {
            'sample_id': record.sample_id,
            'track_features': self._load_track_sequence(record, self.track_mid_length),
            'track_short_features': self._load_track_sequence(record, self.track_short_length),
            'track_long_features': self._load_track_sequence(record, self.track_long_length),
            'track_global_features': self._load_track_global_features(record),
            'rd_features': np.zeros((1, 1, 1, 1), dtype=np.float32),
            'label': np.int64(self.label_to_index[record.label_name]),
            'label_name': record.label_name,
            'has_rd': np.int64(0),
            'notes_target': self._build_note_target(record),
            'bird_drone_target': np.float32(1.0 if record.label_name == 'BirdTarget' else 0.0 if record.label_name == 'DroneTarget' else -1.0),
            'bird_surface_target': np.float32(1.0 if record.label_name == 'BirdTarget' else 0.0 if record.label_name == 'ShipTarget' else -1.0),
            'ship_drone_target': np.float32(1.0 if record.label_name == 'ShipTarget' else 0.0 if record.label_name == 'DroneTarget' else -1.0),
            'ship_bird_target': np.float32(1.0 if record.label_name == 'ShipTarget' else 0.0 if record.label_name == 'BirdTarget' else -1.0),
            'track_router_target': np.int64(0 if record.label_name == 'ShipTarget' else 1 if record.label_name == 'AircraftTarget' else 2 if record.label_name in {'BirdTarget', 'DroneTarget'} else -1),
        }
