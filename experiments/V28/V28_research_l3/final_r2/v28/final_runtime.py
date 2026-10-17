"""Hash-bound loaders for the final V28 research endpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256









def tensor_hash(values: Mapping[str, torch.Tensor]) -> str:
    records = []
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        records.append((name, str(tensor.dtype), tuple(tensor.shape), tensor.numpy().tobytes().hex()))
    return canonical_sha256(records)


def load_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("tensors"), Mapping):
        raise ValueError(f"V28 final tensor payload is invalid: {path}")
    value = dict(payload)
    digest = canonical_sha256({"metadata": {key: item for key, item in value.items() if key not in {"tensors", "cache_sha256"}}, "tensor_sha256": tensor_hash(value["tensors"])})
    if value.get("cache_sha256") != digest:
        raise ValueError(f"V28 final tensor payload hash differs: {path}")
    return value














__all__ = ["load_payload", "tensor_hash"]