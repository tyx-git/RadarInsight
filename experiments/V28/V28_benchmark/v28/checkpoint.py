"""Provenance-bound V28 task checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, resolve_project_path

from .model import V28TaskModel
from .protocol import V28Context, checkpoint_manifest_compatibility


def _tensor_hash(state: Mapping[str, Any]) -> str:
    values: list[Any] = []
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"V28 state entry is not a tensor: {name}")
        tensor = tensor.detach().cpu().contiguous()
        values.append((str(name), str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes().hex()))
    return canonical_sha256(values)








def load_checkpoint(context: V28Context, model: V28TaskModel, path: str | Path, *, task: str, seed: int, optimizer: torch.optim.Optimizer | None = None, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    target = resolve_project_path(context.project_root, path, must_exist=True)
    payload = torch.load(target, map_location=map_location, weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "RADAR-LN3-V28-task-checkpoint-v2":
        raise ValueError("V28 checkpoint schema differs")
    for key, expected in (("task", task), ("seed", int(seed))):
        if payload.get(key) != expected:
            raise ValueError(f"V28 checkpoint binding differs: {key}")
    # The formal Unknown-support gate is an L3 policy and does not alter the
    # frozen classifier parameters or their input contract.
    config_matches = payload.get("config_sha256") == context.config.sha256
    compatibility = checkpoint_manifest_compatibility(
        context.project_root,
        {**context.manifest, "_manifest_path": context.manifest_path.relative_to(context.project_root).as_posix()},
        str(payload.get("manifest_sha256")),
        task=task,
    )
    if not config_matches and compatibility.get("status") != "compatible_archived_manifest":
        raise ValueError("V28 checkpoint config binding differs")
    contract = model.model_contract()
    if payload.get("model_contract") != contract or payload.get("model_contract_sha256") != canonical_sha256(contract):
        raise ValueError("V28 checkpoint model contract differs")
    state = payload.get("model_state")
    if not isinstance(state, Mapping) or payload.get("model_state_sha256") != _tensor_hash(state):
        raise ValueError("V28 checkpoint state hash differs")
    model.load_state_dict(state, strict=True)
    if optimizer is not None:
        saved = payload.get("optimizer_state")
        if not isinstance(saved, Mapping):
            raise ValueError("V28 checkpoint has no optimizer state")
        optimizer.load_state_dict(saved)
    return {**payload, "manifest_compatibility": compatibility}


__all__ = ["load_checkpoint"]