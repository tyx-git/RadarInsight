"""Immutable bindings for the V27 classification base checkpoints.

The classification checkpoints are the experimental boundary for the V27
baseline.  Later knownness, quality and calibration stages are allowed to
change only ``trajectory_heads``; this module records the exact base files
and the classifier-only state hash so that such changes can be checked
without rewriting the historical classification checkpoint.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256



_HEAD_PREFIX = "trajectory_heads."
# The classifier checkpoint is immutable.  These are the *only* keys that a
# later, independently fitted open-set stage may replace.  Do not use a broad
# ``startswith`` exemption here: a future accidental module under
# ``trajectory_heads`` must not silently become mutable classifier state.
_MUTABLE_TRAJECTORY_HEAD_SUFFIXES = frozenset(
    {
        "trajectory_encoder.weight_ih_l0",
        "trajectory_encoder.weight_hh_l0",
        "trajectory_encoder.bias_ih_l0",
        "trajectory_encoder.bias_hh_l0",
        "knownness_projection.0.weight",
        "knownness_projection.0.bias",
        "knownness_projection.1.weight",
        "knownness_projection.1.bias",
        "knownness_head.weight",
        "knownness_head.bias",
        "quality_head.0.weight",
        "quality_head.0.bias",
        "quality_head.2.weight",
        "quality_head.2.bias",
    }
)


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    """Hash state tensors deterministically, independent of torch serialization."""

    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27 base state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend(
            (
                str(name),
                str(tensor.dtype),
                repr(tuple(tensor.shape)),
                tensor.numpy().tobytes().hex(),
            )
        )
    return canonical_sha256(parts)


def mutable_trajectory_head_keys(state: Mapping[str, Any]) -> frozenset[str]:
    """Return the exact permitted mutable state keys, failing on drift."""

    declared = {
        name for name in state if isinstance(name, str) and name.startswith(_HEAD_PREFIX)
    }
    expected = {
        f"{_HEAD_PREFIX}{suffix}" for suffix in _MUTABLE_TRAJECTORY_HEAD_SUFFIXES
    }
    if declared != expected:
        missing = sorted(expected - declared)
        unexpected = sorted(declared - expected)
        raise ValueError(
            "V27 trajectory-head state boundary differs: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return frozenset(expected)


def classifier_state_hash(state: Mapping[str, Any]) -> str:
    """Hash every parameter/buffer outside the explicitly mutable heads."""

    mutable = mutable_trajectory_head_keys(state)
    classifier_state = {name: value for name, value in state.items() if name not in mutable}
    if not classifier_state:
        raise ValueError("V27 checkpoint has no classifier/base state")
    return tensor_state_hash(classifier_state)




















__all__ = ["classifier_state_hash", "mutable_trajectory_head_keys", "tensor_state_hash"]