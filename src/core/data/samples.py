from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import TRACK_FEATURE_COLUMNS, TRACK_META_COLUMNS


@dataclass(frozen=True)
class TrackSample:
    sample_id: str
    source_name: str
    label_name_raw: str
    label_name: str
    csv_path: Path
    rd_path: str
    rd_summary_csv_path: str
    rd_tensor_path: str
    rd_phase_missing: bool
    height_missing: bool
    record_count: int
    time_seconds_min: float
    time_seconds_max: float

    def load_frame(self) -> pd.DataFrame:
        return pd.read_csv(self.csv_path)

    def load_feature_frame(self) -> pd.DataFrame:
        frame = self.load_frame()
        return frame[TRACK_FEATURE_COLUMNS].copy()

    def load_meta_frame(self) -> pd.DataFrame:
        frame = self.load_frame()
        return frame[TRACK_META_COLUMNS].copy()


__all__ = [
    "TrackSample",
]
