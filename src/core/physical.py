from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .encoding import track_feature_missing_mask
from .features import TRACK_FEATURE_COLUMNS


PHYSICAL_FEATURE_COLUMNS = [
    "height_mean",
    "height_std",
    "height_slope",
    "ground_speed_mean",
    "ground_speed_std",
    "ground_speed_max",
    "course_change_rate_mean",
    "trajectory_curvature_mean",
    "acceleration_std",
    "hover_ratio",
    "radial_speed_std",
    "radial_speed_flatness",
    "amplitude_mean",
    "amplitude_std",
    "snr_mean",
    "height_missing_flag",
    "phase_missing_flag",
]

PHYSICAL_FEATURE_COLUMNS_V2 = [
    column for column in PHYSICAL_FEATURE_COLUMNS if column != "phase_missing_flag"
]

# V3 trajectory sequences carry explicit per-field missingness channels.  The
# aggregate physical branch keeps its established V2 feature vector for now;
# its existing height/phase flags remain explicit binary physical features.
PHYSICAL_FEATURE_COLUMNS_V3 = list(PHYSICAL_FEATURE_COLUMNS_V2)
PHYSICAL_FEATURE_COLUMNS_V3.extend(
    [
        f"{column}_missing_flag"
        for column in (
            "time_seconds",
            "azimuth_deg",
            "range_m",
            "radial_speed_mps",
            "elevation_deg",
            "amplitude_db",
            "snr_db",
            "course_deg",
            "speed_mps",
        )
    ]
)


def physical_feature_columns(feature_schema: str = "v1") -> list[str]:
    normalized = str(feature_schema).strip().lower()
    if normalized == "v1":
        return list(PHYSICAL_FEATURE_COLUMNS)
    if normalized == "v2":
        return list(PHYSICAL_FEATURE_COLUMNS_V2)
    if normalized == "v3":
        return list(PHYSICAL_FEATURE_COLUMNS_V3)
    raise ValueError(f"unsupported physical feature schema: {feature_schema}")


def physical_feature_dim(feature_schema: str = "v1") -> int:
    return len(physical_feature_columns(feature_schema))


def physical_binary_feature_indices(feature_schema: str = "v1") -> list[int]:
    """Return missingness flags, which are categorical rather than continuous."""
    return [
        index
        for index, column in enumerate(physical_feature_columns(feature_schema))
        if column.endswith("_missing_flag")
    ]


def physical_continuous_feature_indices(feature_schema: str = "v1") -> list[int]:
    binary_indices = set(physical_binary_feature_indices(feature_schema))
    return [index for index in range(physical_feature_dim(feature_schema)) if index not in binary_indices]


def signed_log1p_physical_continuous(values: np.ndarray, continuous_indices: list[int]) -> np.ndarray:
    """Compress source-unit magnitude while leaving binary missingness untouched."""
    transformed = np.asarray(values, dtype=np.float32).copy()
    if transformed.size == 0 or not continuous_indices:
        return transformed
    indices = np.asarray(continuous_indices, dtype=np.int64)
    selected = transformed[..., indices]
    transformed[..., indices] = np.sign(selected) * np.log1p(np.abs(selected))
    return transformed


@dataclass(frozen=True)
class PhysicalFeatureVector:
    values: np.ndarray
    feature_schema: str = "v1"

    def as_dict(self) -> dict[str, float]:
        columns = physical_feature_columns(self.feature_schema)
        return {name: float(self.values[idx]) for idx, name in enumerate(columns)}


def _safe_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def _raw_time_seconds(frame: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(frame["time_seconds"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    return values


def _linear_slope(values: np.ndarray, time_seconds: np.ndarray | None = None) -> float:
    if values.size < 2:
        return 0.0
    if time_seconds is None:
        x = np.arange(values.size, dtype=np.float64)
        y = values.astype(np.float64, copy=False)
    else:
        x = time_seconds.astype(np.float64, copy=False)
        y = values.astype(np.float64, copy=False)
        valid = np.isfinite(x) & np.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.size < 2:
            return 0.0
    x_centered = x - x.mean()
    denom = float(np.square(x_centered).sum())
    if denom <= 1.0e-12:
        return 0.0
    return float((x_centered * (y - y.mean())).sum() / denom)


def _spectral_flatness(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    magnitude = np.abs(values.astype(np.float64)) + 1e-6
    geo = float(np.exp(np.log(magnitude).mean()))
    ari = float(magnitude.mean())
    return geo / ari if ari > 0.0 else 0.0


def _circular_delta_degrees(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.zeros((0,), dtype=np.float64)
    delta = np.diff(values.astype(np.float64, copy=False), prepend=values[:1])
    return (delta + 180.0) % 360.0 - 180.0


def _valid_time_derivative(values: np.ndarray, time_seconds: np.ndarray) -> np.ndarray:
    if values.size < 2 or time_seconds.size != values.size:
        return np.zeros((0,), dtype=np.float64)
    delta_value = np.diff(values.astype(np.float64, copy=False))
    delta_time = np.diff(time_seconds.astype(np.float64, copy=False))
    valid = np.isfinite(delta_time) & (delta_time > 1.0e-3)
    if not valid.any():
        return np.zeros((0,), dtype=np.float64)
    return delta_value[valid] / delta_time[valid]


def _v2_curvature_mean(
    track_range: np.ndarray,
    azimuth_deg: np.ndarray,
    elevation_deg: np.ndarray,
) -> float:
    if track_range.size < 3:
        return 0.0
    azimuth = np.deg2rad(azimuth_deg.astype(np.float64, copy=False))
    elevation = np.deg2rad(elevation_deg.astype(np.float64, copy=False))
    radius = track_range.astype(np.float64, copy=False)
    points = np.column_stack(
        [
            radius * np.cos(elevation) * np.cos(azimuth),
            radius * np.cos(elevation) * np.sin(azimuth),
            radius * np.sin(elevation),
        ]
    )
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    valid = lengths > 1.0e-6
    if int(valid.sum()) < 2:
        return 0.0
    headings = segments[valid] / lengths[valid, None]
    dots = np.sum(headings[1:] * headings[:-1], axis=1).clip(-1.0, 1.0)
    turns = np.arccos(dots)
    scale = 0.5 * (lengths[valid][1:] + lengths[valid][:-1])
    curvature = turns / np.maximum(scale, 1.0e-6)
    return float(curvature.mean()) if curvature.size else 0.0


def extract_physical_features(
    frame: pd.DataFrame,
    *,
    height_missing: bool = False,
    phase_missing: bool = False,
    feature_schema: str = "v1",
) -> PhysicalFeatureVector:
    schema = str(feature_schema).strip().lower()
    columns = physical_feature_columns(schema)
    if frame.empty:
        return PhysicalFeatureVector(values=np.zeros(len(columns), dtype=np.float32), feature_schema=schema)

    height = _safe_numeric(frame["height_m"])
    speed = _safe_numeric(frame["speed_mps"])
    course = _safe_numeric(frame["course_deg"])
    radial_speed = _safe_numeric(frame["radial_speed_mps"])
    amplitude = _safe_numeric(frame["amplitude_db"])
    snr = _safe_numeric(frame["snr_db"])
    azimuth = _safe_numeric(frame["azimuth_deg"])
    elevation = _safe_numeric(frame["elevation_deg"])
    track_range = _safe_numeric(frame["range_m"])

    if schema == "v1":
        speed_diff = np.diff(speed, prepend=speed[:1]) if speed.size else np.zeros(0, dtype=np.float32)
        acceleration_std = float(speed_diff.std()) if speed_diff.size else 0.0
        course_diff = np.diff(course, prepend=course[:1]) if course.size else np.zeros(0, dtype=np.float32)
        course_change_rate_mean = float(np.abs(course_diff).mean()) if course_diff.size else 0.0
        az_diff = np.diff(azimuth, prepend=azimuth[:1]) if azimuth.size else np.zeros(0, dtype=np.float32)
        range_diff = np.diff(track_range, prepend=track_range[:1]) if track_range.size else np.zeros(0, dtype=np.float32)
        curvature_mean = float(np.sqrt(az_diff ** 2 + range_diff ** 2).mean()) if az_diff.size else 0.0
        height_slope = _linear_slope(height)
    elif schema in {"v2", "v3"}:
        time_seconds = _raw_time_seconds(frame)
        acceleration = _valid_time_derivative(speed, time_seconds)
        acceleration_std = float(acceleration.std()) if acceleration.size else 0.0
        course_delta = _circular_delta_degrees(course)[1:]
        delta_time = np.diff(time_seconds)
        valid_course = np.isfinite(delta_time) & (delta_time > 1.0e-3)
        course_rates = np.abs(course_delta[valid_course] / delta_time[valid_course]) if valid_course.any() else np.zeros((0,), dtype=np.float64)
        course_change_rate_mean = float(course_rates.mean()) if course_rates.size else 0.0
        curvature_mean = _v2_curvature_mean(track_range, azimuth, elevation)
        height_slope = _linear_slope(height, time_seconds)
    else:
        raise ValueError(f"unsupported physical feature schema: {feature_schema}")

    hover_ratio = float((np.abs(speed) <= 1.0).mean()) if speed.size else 0.0
    values = [
        float(height.mean()) if height.size else 0.0,
        float(height.std()) if height.size else 0.0,
        float(height_slope),
        float(speed.mean()) if speed.size else 0.0,
        float(speed.std()) if speed.size else 0.0,
        float(speed.max()) if speed.size else 0.0,
        float(course_change_rate_mean),
        float(curvature_mean),
        float(acceleration_std),
        float(hover_ratio),
        float(radial_speed.std()) if radial_speed.size else 0.0,
        float(_spectral_flatness(radial_speed)),
        float(amplitude.mean()) if amplitude.size else 0.0,
        float(amplitude.std()) if amplitude.size else 0.0,
        float(snr.mean()) if snr.size else 0.0,
        float(height_missing),
    ]
    if schema == "v1":
        values.append(float(phase_missing))
    elif schema == "v3":
        raw_missing = track_feature_missing_mask(frame).any(axis=0)
        raw_columns = [
            "time_seconds",
            "azimuth_deg",
            "range_m",
            "radial_speed_mps",
            "elevation_deg",
            "amplitude_db",
            "snr_db",
            "course_deg",
            "speed_mps",
        ]
        values.extend(
            float(raw_missing[list(TRACK_FEATURE_COLUMNS).index(column)])
            for column in raw_columns
        )
    return PhysicalFeatureVector(values=np.asarray(values, dtype=np.float32), feature_schema=schema)
