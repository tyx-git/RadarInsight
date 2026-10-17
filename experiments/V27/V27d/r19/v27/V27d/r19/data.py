"""r19 views over r18's sealed source-stratified TR-only population."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from v27.V27d.r18.data import R18Dataset, r17_l2_inputs

from .protocol import R19Config, R19_ARMS, R19_SNR_MISSING_INDEX, R19_SNR_VALUE_INDEX


class R19Dataset(Dataset):
    def __init__(self, project_root: str | Path, config: R19Config, *, split_name: str, arm: str) -> None:
        if split_name not in {"train", "validation"}:
            raise ValueError("r19 split must be train or validation")
        if arm not in R19_ARMS:
            raise ValueError("r19 arm is invalid")
        root = Path(project_root).resolve()
        base = R18Dataset(
            root,
            config.r18,
            split_name=split_name,
            radial_speed_ablation=False,
        )
        self.project_root = root
        self.config = config
        self.split_name = split_name
        self.arm = arm
        self.parent_ids = list(base.parent_ids)
        self._items = []
        for item in (base[index] for index in range(len(base))):
            if arm == "snr_missing_only":
                transformed = item
            else:
                values = item.tr_features.detach().clone()
                missing = item.tr_missing_mask.detach().clone()
                if values.shape[-1] <= R19_SNR_VALUE_INDEX or missing.shape[-1] <= R19_SNR_MISSING_INDEX:
                    raise ValueError("r19 SNR channel indices exceed TR input width")
                if bool(values[..., R19_SNR_VALUE_INDEX].ne(0).any()):
                    raise ValueError("r19 SNR ablation found a nonzero SNR value")
                if not bool(missing[..., R19_SNR_MISSING_INDEX].eq(1).all()):
                    raise ValueError("r19 control SNR missingness is not uniformly one")
                values[..., R19_SNR_VALUE_INDEX] = 0.0
                missing[..., R19_SNR_MISSING_INDEX] = 0.0
                transformed = replace(item, tr_features=values, tr_missing_mask=missing)
            if transformed.tr_features[..., 1].ne(item.tr_features[..., 1]).any():
                raise ValueError("r19 SNR arm changed the radial-speed value channel")
            self._items.append(transformed)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("r19 dataset index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError("r19 dataset index is out of range")
        return self._items[index]


def build_r19_loader(
    dataset: R19Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("r19 batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=__import__("v27.V27d.r17.data", fromlist=["collate_r17"]).collate_r17,
        generator=generator,
    )


__all__ = ["R19Dataset", "build_r19_loader", "r17_l2_inputs"]