"""V27d-r5-v2 population caches with audit-only endpoint evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256


from .protocol import V2_CACHE_SCHEMA


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r5-v2 tensor is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r5-v2 cache lacks tensors")
    metadata = {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}
    return canonical_sha256({"metadata": metadata, "tensor_sha256": _tensor_digest(dict(tensors))})










def load_population_cache(path: str | Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping) or value.get("schema_version") != V2_CACHE_SCHEMA or value.get("experiment_id") != "V27d-r5-bird-route-v2":
        raise ValueError("V27d-r5-v2 cache identity differs")
    value = dict(value)
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r5-v2 cache hash differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r5-v2 cache tensors are missing")
    count = len(value.get("parent_ids", ()))
    width = 29
    expected = {
        "summary_features": (count, width, 672),
        "event_trajectory": (count, width, 355),
        "task_facts_sequence": (count, width, 7),
        "stable8_causal_features": (count, width, 38),
        "padding_mask": (count, width),
        "lengths": (count,),
        "quality_target_sequence": (count, width),
        "quality_target_final": (count,),
        "bird_route_positive": (count,),
        "bird_route_negative": (count,),
        "bird_route_population": (count,),
        "bird_knownness_target": (count,),
        "l1_target_route": (count,),
        "l2_predicted_bird": (count,),
        "endpoint_final_logits": (count, 3),
        "classification_prediction": (count,),
        "event_facts_sequence": (count, width, 19),
        "endpoint_sequence_logits": (count, width, 3),
        "stable8_summary": (count, width, 8),
        "stable8_reliable": (count, width),
        "stable8_reliable_count": (count, width),
        "stable8_reliable_fraction": (count, width),
        "rd_present": (count, width),
        "event_frame_values": (count, width, 12, 8),
        "event_frame_mask": (count, width, 12),
    }
    for name, shape in expected.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r5-v2 tensor shape differs: {name}")
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"V27d-r5-v2 tensor is non-finite: {name}")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if padding.dtype != torch.bool or lengths.dtype != torch.long:
        raise ValueError("V27d-r5-v2 padding/length dtype differs")
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r5-v2 padding differs from lengths")
    if not torch.equal(tensors["bird_route_population"], tensors["bird_route_positive"] | tensors["bird_route_negative"]):
        raise ValueError("V27d-r5-v2 population mask differs")
    if bool((tensors["bird_route_positive"] & tensors["bird_route_negative"]).any()):
        raise ValueError("V27d-r5-v2 positive and negative populations overlap")
    for name in ("event_facts_sequence", "endpoint_sequence_logits", "stable8_summary", "stable8_reliable_count", "stable8_reliable_fraction", "rd_present", "event_frame_values", "event_frame_mask"):
        value_tensor = tensors[name]
        mask = padding.reshape(*padding.shape, *([1] * (value_tensor.ndim - 2))).expand_as(value_tensor)
        if bool(value_tensor.masked_select(mask).ne(0).any()):
            raise ValueError(f"V27d-r5-v2 padded audit tensor is non-zero: {name}")
    return {**value, "tensors": {str(key): tensor for key, tensor in tensors.items()}}




__all__ = ["cache_digest", "load_population_cache"]