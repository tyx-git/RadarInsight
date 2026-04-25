from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


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
]


@dataclass(frozen=True)
class PhysicalFeatureVector:
    values: np.ndarray

    def as_dict(self) -> dict[str, float]:
        return {
            name: float(self.values[idx])
            for idx, name in enumerate(PHYSICAL_FEATURE_COLUMNS)
        }


def _safe_numeric(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float32, copy=True)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return values


def _linear_slope(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=np.float32)
    x_mean = float(x.mean())
    y_mean = float(values.mean())
    denom = float(((x - x_mean) ** 2).sum())
    if denom <= 0.0:
        return 0.0
    numer = float(((x - x_mean) * (values - y_mean)).sum())
    return numer / denom


def _spectral_flatness(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    magnitude = np.abs(values.astype(np.float64)) + 1e-6
    geo = float(np.exp(np.log(magnitude).mean()))
    ari = float(magnitude.mean())
    if ari <= 0.0:
        return 0.0
    return geo / ari


def extract_physical_features(frame: pd.DataFrame) -> PhysicalFeatureVector:
    if frame.empty:
        return PhysicalFeatureVector(values=np.zeros(len(PHYSICAL_FEATURE_COLUMNS), dtype=np.float32))

    height = _safe_numeric(frame["height_m"])
    speed = _safe_numeric(frame["speed_mps"])
    course = _safe_numeric(frame["course_deg"])
    radial_speed = _safe_numeric(frame["radial_speed_mps"])
    amplitude = _safe_numeric(frame["amplitude_db"])
    snr = _safe_numeric(frame["snr_db"])
    azimuth = _safe_numeric(frame["azimuth_deg"])
    track_range = _safe_numeric(frame["range_m"])

    speed_diff = np.diff(speed, prepend=speed[:1]) if speed.size else np.zeros(0, dtype=np.float32)
    acceleration_std = float(speed_diff.std()) if speed_diff.size else 0.0

    course_diff = np.diff(course, prepend=course[:1]) if course.size else np.zeros(0, dtype=np.float32)
    course_change_rate_mean = float(np.abs(course_diff).mean()) if course_diff.size else 0.0

    az_diff = np.diff(azimuth, prepend=azimuth[:1]) if azimuth.size else np.zeros(0, dtype=np.float32)
    range_diff = np.diff(track_range, prepend=track_range[:1]) if track_range.size else np.zeros(0, dtype=np.float32)
    curvature = np.sqrt(az_diff ** 2 + range_diff ** 2)

    hover_ratio = float((np.abs(speed) <= 1.0).mean()) if speed.size else 0.0

    vector = np.array(
        [
            float(height.mean()) if height.size else 0.0,
            float(height.std()) if height.size else 0.0,
            float(_linear_slope(height)),
            float(speed.mean()) if speed.size else 0.0,
            float(speed.std()) if speed.size else 0.0,
            float(speed.max()) if speed.size else 0.0,
            float(course_change_rate_mean),
            float(curvature.mean()) if curvature.size else 0.0,
            float(acceleration_std),
            float(hover_ratio),
            float(radial_speed.std()) if radial_speed.size else 0.0,
            float(_spectral_flatness(radial_speed)),
            float(amplitude.mean()) if amplitude.size else 0.0,
            float(amplitude.std()) if amplitude.size else 0.0,
            float(snr.mean()) if snr.size else 0.0,
        ],
        dtype=np.float32,
    )
    return PhysicalFeatureVector(values=vector)
