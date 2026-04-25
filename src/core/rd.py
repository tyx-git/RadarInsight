from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class EncodedRDSequence:
    features: torch.Tensor
    valid_length: int


RD_SUMMARY_COLUMNS = [
    "frame_energy_mean",
    "frame_energy_max",
    "frame_energy_min",
]


def load_rd_summary_frame(data_root: Path | str, summary_csv_path: str) -> pd.DataFrame:
    data_root = Path(data_root)
    csv_path = Path(summary_csv_path)
    if not csv_path.is_absolute():
        csv_path = data_root.parent / csv_path
    return pd.read_csv(csv_path)


def encode_rd_summary(frame: pd.DataFrame) -> EncodedRDSequence:
    features = frame[RD_SUMMARY_COLUMNS].copy()
    features = features.replace([float("inf"), float("-inf")], 0.0).fillna(0.0)
    values = features.to_numpy(dtype="float32")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.clip(values, a_min=0.0, a_max=None)
    values = np.log1p(values)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    values = (values - mean) / std
    rd_presence = np.ones((values.shape[0], 1), dtype=np.float32)
    values = np.concatenate([values.astype(np.float32), rd_presence], axis=1)
    tensor = torch.tensor(values, dtype=torch.float32)
    return EncodedRDSequence(features=tensor, valid_length=int(tensor.shape[0]))
