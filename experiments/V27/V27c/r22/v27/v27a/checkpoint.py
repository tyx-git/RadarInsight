"""Fail-closed checkpoint serialization for V27a residual arms."""

from __future__ import annotations

from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256







def tensor_state_hash(state: Mapping[str, Any]) -> str:
    """Hash state tensors deterministically, independent of torch archives."""

    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27a state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend(
            (
                str(name),
                str(tensor.dtype),
                repr(tuple(tensor.shape)),
                tensor.numpy().tobytes().hex(),
            )
        )
    if not parts:
        raise ValueError("V27a checkpoint state cannot be empty")
    return canonical_sha256(parts)








__all__ = ["tensor_state_hash"]