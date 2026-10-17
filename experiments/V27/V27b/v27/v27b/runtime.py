"""Runtime helpers for the isolated V27b experiment."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.runtime import atomic_write_text, json_text

from .protocol import V27BContext, v27b_artifact_path




def configured_device(context: V27BContext) -> torch.device:
    requested = context.config.training.device.strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("V27b configuration requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    raise ValueError(f"unsupported V27b device: {requested}")


def seed_v27b(seed: int) -> None:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27b seed must be a non-negative integer")
    value = int(seed)
    random.seed(value)
    np.random.seed(value % (2**32))
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def environment_snapshot() -> dict[str, Any]:
    return {
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
    }


def feature_cache_path(context: V27BContext, split: str) -> Path:
    canonical = str(split).strip().lower()
    if canonical == "train":
        template = context.config.artifacts.train_feature_cache_path
    elif canonical in {"val", "validation"}:
        template = context.config.artifacts.validation_feature_cache_path
    else:
        raise ValueError(f"unsupported V27b feature cache split: {split}")
    return v27b_artifact_path(context.project_root, template)


def checkpoint_path(context: V27BContext, *, seed: int, epoch: int) -> Path:
    return v27b_artifact_path(
        context.project_root,
        context.config.artifacts.epoch_checkpoint_template,
        seed=seed,
        epoch=epoch,
    )


def training_log_path(context: V27BContext, *, seed: int) -> Path:
    return v27b_artifact_path(
        context.project_root,
        context.config.artifacts.training_log_template,
        seed=seed,
    )


def training_summary_path(context: V27BContext, *, seed: int) -> Path:
    return v27b_artifact_path(
        context.project_root,
        context.config.artifacts.training_summary_template,
        seed=seed,
    )


def write_json_if_absent_or_identical(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    text = json_text(payload, indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27b JSON artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


__all__ = ["checkpoint_path", "configured_device", "environment_snapshot", "feature_cache_path", "seed_v27b", "training_log_path", "training_summary_path", "write_json_if_absent_or_identical"]