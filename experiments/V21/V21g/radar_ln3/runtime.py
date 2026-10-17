"""Shared runtime primitives for the isolated ``src-v3`` implementation.

This module intentionally contains no model, dataset, or experiment policy.
It centralizes the small pieces of infrastructure that used to be copied
through the V21--V26 modules: strict boolean parsing, project-relative path
resolution, content hashing, and bounded atomic artifact writes.

Historical callers may still import compatibility aliases from their original
modules.  New product code should import these functions from here.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
from pathlib import Path
from typing import Any

import torch




























def _torch_payload_equal(left: Any, right: Any) -> bool:
    """Compare the subset of values used by local Torch artifacts exactly.

    ``torch.save`` archives contain implementation-specific storage identifiers,
    so serializing the same payload twice is not a reliable byte comparison.
    Artifact reuse therefore compares decoded values instead: tensors retain
    exact dtype, shape, and values; containers retain their structure.
    """

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    if isinstance(left, MappingABC) or isinstance(right, MappingABC):
        if not isinstance(left, MappingABC) or not isinstance(right, MappingABC):
            return False
        if set(left) != set(right):
            return False
        return all(_torch_payload_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if (
            not isinstance(left, (list, tuple))
            or not isinstance(right, (list, tuple))
            or type(left) is not type(right)
            or len(left) != len(right)
        ):
            return False
        return all(_torch_payload_equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, Path) or isinstance(right, Path):
        return isinstance(left, Path) and isinstance(right, Path) and left == right
    if type(left) is not type(right):
        return False
    try:
        return bool(left == right)
    except (TypeError, ValueError):
        return False







