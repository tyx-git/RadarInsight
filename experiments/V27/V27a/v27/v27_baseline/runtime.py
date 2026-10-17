"""Shared execution helpers for the sealed V27 baseline.

This module is intentionally small and policy-free beyond the V27 protocol:
it is the single place where an entry point resolves the verified context,
chooses the configured device, seeds the three RNG families, and writes a
mutable JSON artifact.  It never opens test or Unknown payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


from .model import V27LiquidV2TaskModel
from .protocol import V27Config


@dataclass(frozen=True)
class V27Context:
    """All provenance objects an entry point may consume."""

    project_root: Path
    config: V27Config
    data_binding: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str




def configured_device(config: V27Config) -> torch.device:
    """Resolve the exact configured device and fail closed if unavailable."""

    requested = str(config.training.device).strip().lower()
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("V27 configuration requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported V27 device: {requested}")




def build_v27_model(
    context: V27Context,
    *,
    task: str,
    device: torch.device | None = None,
) -> V27LiquidV2TaskModel:
    """Construct one task-local V27 model and move it to the verified device."""

    target = configured_device(context.config) if device is None else device
    model = V27LiquidV2TaskModel(context.config, task=task)
    model.to(target)
    return model


















__all__ = ["V27Context", "build_v27_model", "configured_device"]

