"""r22 data view: exact r20 population with only the registered scale changed."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

from v27.V27d.r20.data import R20Dataset, build_r20_loader, r20_model_input_kwargs

from .protocol import R22Config


class R22Dataset(R20Dataset):
    """Exact r20 parent population and sidecar logic under the r22 config."""

    def __init__(
        self,
        project_root: str | Path,
        config: R22Config,
        *,
        split_name: str,
        arm: str,
    ) -> None:
        super().__init__(project_root, config, split_name=split_name, arm=arm)


def build_r22_loader(
    dataset: R22Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return build_r20_loader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


def r22_model_input_kwargs(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return r20_model_input_kwargs(batch, device)


__all__ = ["R22Dataset", "build_r22_loader", "r22_model_input_kwargs"]