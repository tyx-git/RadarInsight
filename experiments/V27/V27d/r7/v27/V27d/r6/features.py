"""Route-matched Bird verifier population caches for V27d-r6."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256


from .protocol import R6_CACHE_SCHEMA


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name].detach().cpu().contiguous()
        parts.extend((name, str(value.dtype), repr(tuple(value.shape)), value.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r6 cache lacks tensors")
    metadata = {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}
    return canonical_sha256({"metadata": metadata, "tensor_sha256": _tensor_digest(tensors)})








def load_population_cache(path: str | Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or value.get("schema_version") != R6_CACHE_SCHEMA or value.get("experiment_id") != "V27d-r6-bird-liquid-verifier":
        raise ValueError("V27d-r6 cache identity differs")
    value = dict(value)
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r6 cache hash differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r6 cache tensors are missing")
    count = len(value["parent_ids"])
    width = 29
    expected = {
        "summary_features": (count, width, 672), "event_trajectory": (count, width, 355), "task_facts_sequence": (count, width, 7), "event_facts_sequence": (count, width, 19),
        "padding_mask": (count, width), "lengths": (count,), "quality_target_sequence": (count, width), "quality_target_final": (count,),
        "candidate_population": (count,), "bird_positive": (count,), "route_false_positive_negative": (count,), "knownness_target": (count,),
        "prefix_route_mask": (count, width), "endpoint_final_logits": (count, 3), "l1_endpoint_sequence_logits": (count, width, 2), "l2_endpoint_sequence_logits": (count, width, 3),
        "stable8_causal_features": (count, width, 38),
        "event_frame_values": (count, width, 12, 8), "event_frame_mask": (count, width, 12),
    }
    for name, shape in expected.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r6 tensor shape differs: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"V27d-r6 tensor is non-finite: {name}")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if padding.dtype != torch.bool or lengths.dtype != torch.long:
        raise ValueError("V27d-r6 padding/length dtype differs")
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r6 padding differs from lengths")
    if not torch.equal(tensors["candidate_population"], tensors["bird_positive"] | tensors["route_false_positive_negative"]):
        raise ValueError("V27d-r6 population masks differ")
    for name in ("event_trajectory", "task_facts_sequence", "event_facts_sequence", "prefix_route_mask", "l1_endpoint_sequence_logits", "l2_endpoint_sequence_logits", "stable8_causal_features", "event_frame_values", "event_frame_mask"):
        tensor = tensors[name]
        mask = padding.reshape(*padding.shape, *([1] * (tensor.ndim - 2))).expand_as(tensor)
        if bool(tensor.masked_select(mask).ne(0).any()):
            raise ValueError(f"V27d-r6 padded tensor is non-zero: {name}")
    return {**value, "tensors": {str(key): tensor for key, tensor in tensors.items()}}




__all__ = ["cache_digest", "load_population_cache"]