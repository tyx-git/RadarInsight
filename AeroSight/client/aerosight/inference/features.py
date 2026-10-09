# -*- coding: utf-8 -*-
"""Feature adapters for the AeroSight v3 ONNX classifier.

The model receives:
* latest 29 track points: [B,29,14]
* current complete RD map: [B,2,H,W]

Channel 0 is amplitude and channel 1 is phase. The complete RD map is not
cropped, resized, or converted into compact descriptors.
"""
from __future__ import annotations

import numpy as np

TRACK_FEATURE_SIZE = 14
TRACK_HISTORY_LENGTH = 29
DEFAULT_SEQUENCE_LENGTH = TRACK_HISTORY_LENGTH

TRACK_FEATURE_NAMES = (
    "time_delta_s",
    "range_m",
    "azimuth_deg",
    "elevation_deg",
    "ground_range_m",
    "altitude_m",
    "radial_speed_mps",
    "speed_mps",
    "course_deg",
    "snr_db",
    "amplitude_db",
    "x_m",
    "y_m",
    "z_m",
)


def track_features(observation, time_delta_s: float) -> np.ndarray:
    """Return the 14-D record for one target at one time step."""
    values = np.asarray(
        [
            max(0.0, float(time_delta_s)),
            observation.range_m,
            observation.azimuth_deg,
            observation.elevation_deg,
            observation.ground_range_m,
            observation.altitude_m,
            observation.radial_speed_mps,
            observation.speed_mps,
            observation.course_deg,
            observation.snr_db,
            observation.amplitude_db,
            observation.x_m,
            observation.y_m,
            observation.z_m,
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def clean_rd_map(amplitude, phase=None) -> tuple[np.ndarray, np.ndarray]:
    """Validate and clean a complete 2-D RD map without resizing it."""
    amp = np.asarray(amplitude, dtype=np.float32)
    if amp.ndim != 2 or amp.shape[0] < 1 or amp.shape[1] < 1:
        raise ValueError(
            f"RD amplitude must be a non-empty 2D map, got shape {getattr(amp, 'shape', None)}"
        )

    ph = np.asarray(phase, dtype=np.float32)
    if ph.shape != amp.shape:
        ph = np.zeros_like(amp)

    amp = np.nan_to_num(amp, nan=0.0, posinf=0.0, neginf=0.0)
    ph = np.nan_to_num(ph, nan=0.0, posinf=0.0, neginf=0.0)
    return amp, ph


def rd_map_channels(amplitude, phase) -> np.ndarray:
    """Return complete RD map as [2,H,W]: channel 0 amplitude, channel 1 phase."""
    amp, ph = clean_rd_map(amplitude, phase)
    return np.stack((amp, ph), axis=0).astype(np.float32)
