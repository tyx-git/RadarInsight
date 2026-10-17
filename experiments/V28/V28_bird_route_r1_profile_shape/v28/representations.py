"""Materialize and verify frozen V28 auxiliary-head representations."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256

from .protocol import V28Context, V28_REPRESENTATION_SCHEMA, artifact_path


def _tensor_hash(values: Mapping[str, torch.Tensor]) -> str:
    records: list[Any] = []
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        records.append((name, str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes().hex()))
    return canonical_sha256(records)


def representation_path(context: V28Context, *, task: str, role: str) -> Path:
    return artifact_path(context.project_root, str(context.config.artifacts["representation_template"]), task=task, role=role, seed=0)








def load_representation(context: V28Context, *, task: str, role: str) -> dict[str, Any]:
    path = representation_path(context, task=task, role=role)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("schema_version") != V28_REPRESENTATION_SCHEMA or payload.get("system_manifest_sha256") != context.manifest_sha256:
        raise ValueError(f"V28 representation identity differs: {task}/{role}")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V28 representation tensors are missing")
    digest = canonical_sha256({"metadata": {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}, "tensor_sha256": _tensor_hash(tensors)})
    if payload.get("cache_sha256") != digest:
        raise ValueError(f"V28 representation hash differs: {task}/{role}")
    return dict(payload)


__all__ = ["load_representation", "representation_path"]