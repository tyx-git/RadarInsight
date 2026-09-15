"""r21 views that reuse r20 data and sidecar construction exactly."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader, Dataset

from v27.V27d.r20.data import R20Dataset, R20Sequence, collate_r20

from .protocol import R21_ARMS, R21Config


class R21Dataset(Dataset[R20Sequence]):
    """Exact r20 data population with r21 arm labels only."""

    def __init__(
        self,
        project_root: str | Path,
        config: R21Config,
        *,
        split_name: str,
        arm: str,
    ) -> None:
        if split_name not in {"train", "validation"}:
            raise ValueError("r21 split must be train or validation")
        if arm not in R21_ARMS:
            raise ValueError("r21 arm is invalid")
        root = Path(project_root).resolve()
        source_arm = "zero_control" if arm == "zero_control" else "radial_decomposed"
        base = R20Dataset(
            root,
            config.r20,
            split_name=split_name,
            arm=source_arm,
        )
        self.project_root = root
        self.config = config
        self.split_name = split_name
        self.arm = arm
        self.parent_ids = list(base.parent_ids)
        self._items = [base[index] for index in range(len(base))]

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> R20Sequence:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("r21 dataset index must be an integer")
        if not 0 <= index < len(self):
            raise IndexError("r21 dataset index is out of range")
        return self._items[index]


def build_r21_loader(
    dataset: R21Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if isinstance(batch_size, bool) or int(batch_size) <= 0:
        raise ValueError("r21 batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r20,
        generator=generator,
    )


def r21_model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    keys = (
        "tr_features",
        "tr_missing_mask",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_dt_reason",
        "tr_padding_mask",
        "rd_frames",
        "rd_frame_mask",
        "rd_quality",
        "rd_physical_features",
        "rd_physical_valid",
        "micro_doppler_features",
        "micro_doppler_valid",
        "radial_sidecar_values",
        "radial_sidecar_valid",
    )
    result: dict[str, torch.Tensor] = {}
    for key in keys:
        value = batch.get(key)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"r21 batch is missing {key}")
        result[key] = value.to(device=device, non_blocking=True)
    return result


__all__ = ["R21Dataset", "build_r21_loader", "r21_model_input_kwargs"]
