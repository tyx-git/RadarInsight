"""r18 views over the immutable r17 processed TR-only events."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.features import encode_causal_tr_observation_increment_sequence, encode_tr_feature_values
from v27.V27d.r17.data import R17Dataset, R17Sequence, collate_r17, r17_l2_inputs
from v27.V27d.r17.data import _event_values, _l2_robust_values

from .protocol import R18Config, build_source_stratified_split


def _radial_speed_ablation(base: R17Sequence, config: R18Config) -> R17Sequence:
    encoded_values: list[np.ndarray] = []
    missing_values: list[np.ndarray] = []
    raw_values: list[dict[str, Any]] = []
    for row in base.event_rows:
        values = _event_values(row)
        values, _, _ = _l2_robust_values(values, config.r17)
        values["radial_speed_mps"] = None
        encoded, missing = encode_tr_feature_values(values)
        encoded_values.append(encoded)
        missing_values.append(missing)
        raw_values.append(values)
    observation, valid = encode_causal_tr_observation_increment_sequence(raw_values)
    observation = np.where(valid, observation, 0.0).astype(np.float32, copy=False)
    return replace(
        base,
        tr_features=torch.from_numpy(np.stack(encoded_values).astype(np.float32)),
        tr_missing_mask=torch.from_numpy(np.stack(missing_values).astype(np.float32)),
        observation_increment_values=torch.from_numpy(observation),
    )


class R18Dataset(Dataset[R17Sequence]):
    def __init__(self, project_root: str | Path, config: R18Config, *, split_name: str, radial_speed_ablation: bool) -> None:
        if split_name not in {"train", "validation"}:
            raise ValueError("r18 split must be train or validation")
        root = Path(project_root).resolve()
        split = build_source_stratified_split(root, config)
        requested = [pid for pid, value in split["assignments"].items() if value == split_name]
        base = R17Dataset(root, config.r17, task="l2", parent_ids=requested)
        self.project_root = root
        self.config = config
        self.split_name = split_name
        self.radial_speed_ablation = bool(radial_speed_ablation)
        self.parent_ids = list(base.parent_ids)
        self._items = []
        for item in (base[index] for index in range(len(base))):
            self._items.append(_radial_speed_ablation(item, config) if self.radial_speed_ablation else item)
        expected = set(requested) - set(self.parent_ids)
        if expected:
            raise ValueError(f"r18 L2 dataset lacks requested parents: {sorted(expected)[:3]}")

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> R17Sequence:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("r18 dataset index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError("r18 dataset index is out of range")
        return self._items[index]


def build_r18_loader(dataset: R18Dataset, *, batch_size: int, shuffle: bool, generator: torch.Generator | None = None) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("r18 batch size must be positive")
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=bool(shuffle), num_workers=0, pin_memory=torch.cuda.is_available(), collate_fn=collate_r17, generator=generator)


__all__ = ["R18Dataset", "build_r18_loader", "r17_l2_inputs"]