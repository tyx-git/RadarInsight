from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import (
    TRACK_FEATURE_COLUMNS,
    track_sequence_continuous_feature_indices,
    track_sequence_feature_columns,
)


@dataclass(frozen=True)
class EncodedTrackSequence:
    features: np.ndarray
    valid_length: int


def sanitize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None:
        return None
    cleaned = frame[TRACK_FEATURE_COLUMNS].copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    for column in TRACK_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.fillna(0.0)


def track_feature_missing_mask(frame: pd.DataFrame) -> np.ndarray:
    """Return a per-row mask for raw fields before zero-value imputation."""
    if frame is None:
        return np.zeros((0, len(TRACK_FEATURE_COLUMNS)), dtype=np.float32)
    raw = frame[TRACK_FEATURE_COLUMNS].copy()
    for column in TRACK_FEATURE_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    values = raw.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    return (~np.isfinite(values)).astype(np.float32)


def track_sequence_feature_missing_mask(
    frame: pd.DataFrame,
    *,
    feature_schema: str,
) -> np.ndarray:
    """Map raw-field missingness onto the encoded sequence feature positions.

    The returned mask is element-level: ``True`` means that the corresponding
    encoded value is unavailable and must be excluded from reconstruction
    loss.  V3's explicit missingness channels are never marked unavailable;
    they are observed metadata about the raw fields.
    """
    schema = str(feature_schema).strip().lower()
    raw_mask = track_feature_missing_mask(frame)
    feature_dim = len(track_sequence_feature_columns(schema))
    if schema == "v1":
        return raw_mask.astype(bool, copy=False)
    if schema not in {"v2", "v3"}:
        raise ValueError(f"unsupported trajectory feature schema: {feature_schema}")
    output = np.zeros((raw_mask.shape[0], feature_dim), dtype=bool)
    raw_to_encoded = [
        (0,),       # time_seconds -> relative_time_seconds
        (1, 2),     # azimuth_deg -> sin/cos
        (3,),       # range_m
        (4,),       # radial_speed_mps
        (5,),       # elevation_deg
        (6,),       # height_m
        (7,),       # amplitude_db
        (8,),       # snr_db
        (9, 10),    # course_deg -> sin/cos
        (11,),      # speed_mps
    ]
    for raw_index, encoded_indices in enumerate(raw_to_encoded):
        for encoded_index in encoded_indices:
            output[:, encoded_index] = raw_mask[:, raw_index].astype(bool, copy=False)
    all_raw_missing = raw_mask.all(axis=1)
    if bool(all_raw_missing.any()):
        output[all_raw_missing, :] = True
    return output


def prepare_track_sequence_features(
    features: np.ndarray,
    *,
    source_name: str,
    feature_schema: str,
) -> np.ndarray:
    """Apply source-aware representation transforms without touching masks.

    CQ08 `和幅度` is a linear magnitude stored in the legacy amplitude column.
    V3 compresses only that source's present amplitude values with signed log1p;
    V1/V2 remain byte-for-byte compatible with their historical paths.
    """
    values = np.asarray(features, dtype=np.float32).copy()
    schema = str(feature_schema).strip().lower()
    if schema != "v3" or str(source_name).strip() != "cq08_track":
        return values
    columns = track_sequence_feature_columns(schema)
    amplitude_index = columns.index("amplitude_db")
    missing_index = columns.index("missing_amplitude_db")
    present = values[..., missing_index] < 0.5
    amplitude = values[..., amplitude_index]
    compressed = np.sign(amplitude) * np.log1p(np.abs(amplitude))
    values[..., amplitude_index] = np.where(present, compressed, amplitude)
    return values


def normalize_track_sequence_features(
    features: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    *,
    feature_schema: str,
) -> np.ndarray:
    """Normalize continuous sequence fields while preserving V3 masks."""
    values = np.asarray(features, dtype=np.float32).copy()
    center_array = np.asarray(center, dtype=np.float32).reshape(-1)
    scale_array = np.asarray(scale, dtype=np.float32).reshape(-1)
    if center_array.size != values.shape[-1] or scale_array.size != values.shape[-1]:
        raise ValueError(
            "sequence source statistics dimension does not match encoded features: "
            f"features={values.shape[-1]}, center={center_array.size}, scale={scale_array.size}"
        )
    indices = track_sequence_continuous_feature_indices(feature_schema)
    if indices:
        selected = np.asarray(indices, dtype=np.int64)
        values[..., selected] = (
            values[..., selected] - center_array[selected]
        ) / np.maximum(scale_array[selected], 1.0e-3)
    return values


def _relative_time_seconds(frame: pd.DataFrame, cleaned: pd.DataFrame) -> np.ndarray:
    """Encode time relative to the first finite observation in ``frame``."""
    raw = pd.to_numeric(frame["time_seconds"], errors="coerce").to_numpy(dtype=np.float64, copy=True)
    finite = np.isfinite(raw)
    if not finite.any():
        return np.zeros((len(cleaned),), dtype=np.float32)
    first = float(raw[finite][0])
    relative = np.where(finite, raw - first, 0.0)
    return np.nan_to_num(relative, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def encode_track_sequence(frame: pd.DataFrame, *, feature_schema: str = "v1") -> EncodedTrackSequence:
    """Encode raw track rows into a versioned sequence feature schema."""
    cleaned = sanitize_feature_frame(frame)
    schema = str(feature_schema).strip().lower()
    if schema == "v1":
        features = cleaned.to_numpy(dtype=np.float32)
    elif schema in {"v2", "v3"}:
        azimuth = np.deg2rad(cleaned["azimuth_deg"].to_numpy(dtype=np.float32, copy=False))
        course = np.deg2rad(cleaned["course_deg"].to_numpy(dtype=np.float32, copy=False))
        base_features = np.column_stack(
            [
                _relative_time_seconds(frame, cleaned),
                np.sin(azimuth),
                np.cos(azimuth),
                cleaned["range_m"].to_numpy(dtype=np.float32, copy=False),
                cleaned["radial_speed_mps"].to_numpy(dtype=np.float32, copy=False),
                cleaned["elevation_deg"].to_numpy(dtype=np.float32, copy=False),
                cleaned["height_m"].to_numpy(dtype=np.float32, copy=False),
                cleaned["amplitude_db"].to_numpy(dtype=np.float32, copy=False),
                cleaned["snr_db"].to_numpy(dtype=np.float32, copy=False),
                np.sin(course),
                np.cos(course),
                cleaned["speed_mps"].to_numpy(dtype=np.float32, copy=False),
            ]
        ).astype(np.float32, copy=False)
        if schema == "v3":
            features = np.column_stack(
                [base_features, track_feature_missing_mask(frame)]
            ).astype(np.float32, copy=False)
        else:
            features = base_features
    else:
        raise ValueError(f"unsupported trajectory feature schema: {feature_schema}")
    expected_dim = len(track_sequence_feature_columns(schema))
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise RuntimeError("encoded track feature dimension does not match schema")
    return EncodedTrackSequence(features=features, valid_length=int(features.shape[0]))
