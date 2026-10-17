"""Bird-route population caches derived from immutable V27d-r1 caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256




CACHE_SCHEMA = "RADAR-LN3-V27d-r5-bird-route-population-cache-v1"


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        tensor = values[name].detach().cpu().contiguous()
        parts.extend((name, str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r5 cache lacks tensors")
    metadata = {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}
    return canonical_sha256({"metadata": metadata, "tensor_sha256": _tensor_digest(tensors)})










def load_population_cache(path: str | Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise ValueError("V27d-r5 population cache must be an object")
    value = dict(value)
    if value.get("schema_version") != CACHE_SCHEMA or value.get("experiment_id") != "V27d-r5-bird-route":
        raise ValueError("V27d-r5 population cache identity differs")
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r5 population cache hash differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r5 population tensors are missing")
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
    }
    for name, shape in expected.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r5 population tensor shape differs: {name}")
        if not torch.isfinite(tensor).all() and tensor.is_floating_point():
            raise ValueError(f"V27d-r5 population tensor is non-finite: {name}")
    if tensors["padding_mask"].dtype != torch.bool or tensors["lengths"].dtype != torch.long:
        raise ValueError("V27d-r5 padding or length dtype differs")
    lengths = tensors["lengths"]
    expected_padding = torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(tensors["padding_mask"], expected_padding):
        raise ValueError("V27d-r5 padding differs from lengths")
    if not torch.equal(tensors["bird_route_population"], tensors["bird_route_positive"] | tensors["bird_route_negative"]):
        raise ValueError("V27d-r5 Bird population mask differs")
    if bool((tensors["bird_route_positive"] & tensors["bird_route_negative"]).any()):
        raise ValueError("V27d-r5 positive and negative populations overlap")
    if bool((tensors["stable8_causal_features"].masked_select(tensors["padding_mask"].unsqueeze(-1))).ne(0).any()):
        raise ValueError("V27d-r5 stable8 padding is non-zero")
    return {**value, "tensors": {str(key): tensor for key, tensor in tensors.items()}}




__all__ = ["CACHE_SCHEMA", "cache_digest", "load_population_cache"]