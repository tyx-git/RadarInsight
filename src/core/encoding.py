from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import TRACK_FEATURE_COLUMNS


@dataclass(frozen=True)
class EncodedTrackSequence:
    """
    编码后的轨迹序列类，包含特征表示和有效长度。
    """
    features: np.ndarray
    valid_length: int


def sanitize_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """
    清理特征数据，处理异常值和缺失值。
    """
    if frame is None:
        return None
    cleaned = frame[TRACK_FEATURE_COLUMNS].copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    for column in TRACK_FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.fillna(0.0)
    return cleaned


def encode_track_sequence(frame: pd.DataFrame) -> EncodedTrackSequence:
    """
    编码轨迹序列，将 DataFrame 转换为 EncodedTrackSequence 实例。
    """
    cleaned = sanitize_feature_frame(frame)
    features = cleaned.to_numpy(dtype=np.float32)
    return EncodedTrackSequence(features=features, valid_length=int(features.shape[0])) # 有效长度为特征数，获取时间帧数(shape[0])
