"""Runtime helpers shared by the isolated V27a experiment stages."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.runtime import (
    atomic_write_text,
    json_text,
    resolve_project_path,
    sha256_file,
)

from .cache import cache_payload_sha256, load_cache_payload
from .protocol import (
    V27AContext,
    load_v27a_context as _load_v27a_context,
    read_sealed_json,
    write_sealed_json,
)


def load_v27a_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> V27AContext:
    """Load and fail-closed verify the current V27a manifest."""

    return _load_v27a_context(project_root, config_path)


def configured_device(context: V27AContext) -> torch.device:
    requested = str(context.config.training.device).strip().lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError(
                "V27a configuration requires CUDA, but CUDA is unavailable"
            )
        return torch.device("cuda:0")
    raise ValueError(f"unsupported V27a device: {requested}")


def seed_v27a(seed: int) -> None:
    value = int(seed)
    if value < 0:
        raise ValueError("V27a seed must be non-negative")
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


def cache_path(context: V27AContext, split_name: str) -> Path:
    split = str(split_name).strip().lower()
    if split == "train":
        value = context.config.artifacts.train_cache_path
    elif split in {"val", "validation"}:
        value = context.config.artifacts.validation_cache_path
    else:
        raise ValueError(f"unsupported V27a cache split: {split_name}")
    return resolve_project_path(context.project_root, value, must_exist=True)


def load_verified_cache(context: V27AContext, split_name: str) -> dict[str, Any]:
    """Load a materialized cache and verify it belongs to this exact run."""

    split = "val" if str(split_name).strip().lower() in {"val", "validation"} else str(split_name).strip().lower()
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported V27a cache split: {split_name}")
    path = cache_path(context, split)
    payload = load_cache_payload(path)
    expected = {
        "split_name": split,
        "base_checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "base_classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "v27a_manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "stable8_indices": list(context.config.data.stable8_indices),
        "stable8_names": list(context.config.data.stable8_names),
    }
    for key, value in expected.items():
        actual = payload.get(key)
        if actual != value:
            raise ValueError(
                f"V27a cache {path.name} provenance differs at {key}: "
                f"expected={value!r}, actual={actual!r}"
            )
    return payload


def cache_payload_digest(payload: Mapping[str, Any]) -> str:
    """Return the verified full-content digest declared by a V27a cache."""

    declared = payload.get("cache_payload_sha256")
    actual = cache_payload_sha256(payload)
    if not isinstance(declared, str) or declared != actual:
        raise ValueError("V27a cache payload digest differs")
    return actual


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a mutable JSON progress artifact atomically."""

    target = Path(path)
    atomic_write_text(target, json_text(dict(payload), indent=2))
    return target


def write_json_if_absent_or_identical(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Create a progress JSON artifact without replacing a different run."""

    target = Path(path)
    text = json_text(dict(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27a JSON artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


__all__ = [
    "cache_path",
    "cache_payload_digest",
    "configured_device",
    "environment_snapshot",
    "load_verified_cache",
    "load_v27a_context",
    "read_sealed_json",
    "resolve_project_path",
    "seed_v27a",
    "sha256_file",
    "write_json",
    "write_json_if_absent_or_identical",
    "write_sealed_json",
]
