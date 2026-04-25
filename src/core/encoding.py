from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import TRACK_FEATURE_COLUMNS


@dataclass(frozen=True)
class EncodedTrackSequence:
    features: np.ndarray
    valid_length: int


def sanitize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame[TRACK_FEATURE_COLUMNS].copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    for column in TRACK_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.fillna(0.0)
    return cleaned


def encode_track_sequence(frame: pd.DataFrame) -> EncodedTrackSequence:
    cleaned = sanitize_feature_frame(frame)
    features = cleaned.to_numpy(dtype=np.float32)
    return EncodedTrackSequence(features=features, valid_length=int(features.shape[0]))
