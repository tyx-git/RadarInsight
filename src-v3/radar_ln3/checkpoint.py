from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .runtime import atomic_torch_save


CHECKPOINT_SCHEMA = "RADAR-LN3-V0-checkpoint-v1"
FORMAL_CHECKPOINT_SCHEMA = "RADAR-LN3-V2-formal-checkpoint-v1"
FORMAL_CHECKPOINT_SCHEMA_V3 = "RADAR-LN3-V3-formal-checkpoint-v1"


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None, *, config_name: str, step: int) -> None:
    target = Path(path)
    payload: dict[str, Any] = {
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "config_name": str(config_name),
        "step": int(step),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    atomic_torch_save(target, payload)


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None = None, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema") != CHECKPOINT_SCHEMA:
        raise ValueError("invalid RADAR-LN3 checkpoint schema")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


def save_formal_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    *,
    config_name: str,
    epoch: int,
    global_step: int,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    checkpoint_schema: str = FORMAL_CHECKPOINT_SCHEMA,
) -> None:
    target = Path(path)
    payload: dict[str, Any] = {
        "checkpoint_schema": str(checkpoint_schema),
        "config_name": str(config_name),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "metadata": dict(metadata),
        "model_state": model.state_dict(),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    atomic_torch_save(target, payload)


def load_formal_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    map_location: str | torch.device = "cpu",
    expected_schema: str = FORMAL_CHECKPOINT_SCHEMA,
    expected_model_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema") != str(expected_schema):
        raise ValueError(f"invalid RADAR-LN3 formal checkpoint schema: expected={expected_schema}")
    if expected_model_contract is not None:
        actual_contract = payload.get("metadata", {}).get("model_contract")
        if actual_contract != expected_model_contract:
            raise ValueError("formal checkpoint model/input contract does not match the configured repaired model")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "FORMAL_CHECKPOINT_SCHEMA",
    "FORMAL_CHECKPOINT_SCHEMA_V3",
    "load_checkpoint",
    "load_formal_checkpoint",
    "save_checkpoint",
    "save_formal_checkpoint",
]
