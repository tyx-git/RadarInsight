"""Causal stable8 dispersion caches for V27d-r7."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical
from v27.V27d.r6.features import load_population_cache as load_r6_population_cache

from .protocol import R7_CACHE_SCHEMA, R7_ROLES, artifact_path, load_context, source_path


DISPERSION_NAMES = (
    "current_frame_std",
    "causal_event_std",
    "causal_adjacent_delta",
)


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r7 tensor is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r7 cache lacks tensors")
    metadata = {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}
    return canonical_sha256({"metadata": metadata, "tensor_sha256": _tensor_digest(tensors)})


def _masked_event_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    count = valid.sum(dim=-1)
    total = values.masked_fill(~valid, 0.0).sum(dim=-1)
    return total / count.clamp_min(1).to(dtype=values.dtype), count


def causal_dispersion_features(
    event_frame_values: torch.Tensor,
    event_frame_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    channel_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build prefix-safe dispersion statistics from one stable8 channel.

    The returned feature tensor contains current-frame standard deviation,
    causal event-level standard deviation, and causal adjacent-event delta.
    An event with no reliable frame contributes no new evidence and receives
    zero dispersion features.  Valid fraction is returned separately for
    quality diagnostics and is never used by the knownness model.
    """

    if event_frame_values.ndim != 4 or event_frame_mask.ndim != 3 or padding_mask.ndim != 2:
        raise ValueError("V27d-r7 stable8 input dimensions differ")
    if event_frame_values.shape[:3] != event_frame_mask.shape or event_frame_values.shape[:2] != padding_mask.shape:
        raise ValueError("V27d-r7 stable8 input shapes differ")
    if event_frame_mask.dtype != torch.bool or padding_mask.dtype != torch.bool:
        raise ValueError("V27d-r7 stable8 masks must be bool")
    if not 0 <= int(channel_index) < event_frame_values.shape[-1]:
        raise ValueError("V27d-r7 stable8 channel index is out of bounds")
    if not event_frame_values.is_floating_point() or not torch.isfinite(event_frame_values).all():
        raise ValueError("V27d-r7 stable8 values are invalid")

    values = event_frame_values[..., int(channel_index)]
    valid = event_frame_mask & ~padding_mask.unsqueeze(-1)
    count = valid.sum(dim=-1)
    event_valid = count.gt(0) & ~padding_mask
    valid_fraction = (count.to(dtype=values.dtype) / float(event_frame_values.shape[2])).masked_fill(padding_mask, 0.0)

    mean, _ = _masked_event_mean(values, valid)
    centered = values - mean.unsqueeze(-1)
    current_var = (centered.square().masked_fill(~valid, 0.0).sum(dim=-1) / count.clamp_min(1).to(dtype=values.dtype))
    current_std = current_var.clamp_min(0.0).sqrt().masked_fill(~event_valid, 0.0)
    event_values = mean.masked_fill(~event_valid, 0.0)

    batch, length = padding_mask.shape
    causal_std = torch.zeros((batch, length), dtype=values.dtype, device=values.device)
    adjacent_delta = torch.zeros_like(causal_std)
    running_count = torch.zeros((batch,), dtype=values.dtype, device=values.device)
    running_sum = torch.zeros_like(running_count)
    running_square_sum = torch.zeros_like(running_count)
    previous_value = torch.zeros_like(running_count)
    previous_valid = torch.zeros((batch,), dtype=torch.bool, device=values.device)
    for event in range(length):
        current_valid = event_valid[:, event]
        current_value = event_values[:, event]
        adjacent_delta[:, event] = torch.where(
            current_valid & previous_valid,
            (current_value - previous_value).abs(),
            torch.zeros_like(current_value),
        )
        next_count = running_count + current_valid.to(dtype=values.dtype)
        next_sum = running_sum + torch.where(current_valid, current_value, torch.zeros_like(current_value))
        next_square_sum = running_square_sum + torch.where(current_valid, current_value.square(), torch.zeros_like(current_value))
        mean_prefix = next_sum / next_count.clamp_min(1.0)
        var_prefix = (next_square_sum / next_count.clamp_min(1.0) - mean_prefix.square()).clamp_min(0.0)
        causal_std[:, event] = torch.where(current_valid, var_prefix.sqrt(), torch.zeros_like(mean_prefix))
        running_count, running_sum, running_square_sum = next_count, next_sum, next_square_sum
        previous_value = torch.where(current_valid, current_value, previous_value)
        previous_valid = previous_valid | current_valid

    features = torch.stack((current_std, causal_std, adjacent_delta), dim=-1)
    features = features.masked_fill(padding_mask.unsqueeze(-1) | ~event_valid.unsqueeze(-1), 0.0)
    if not torch.isfinite(features).all() or not torch.isfinite(valid_fraction).all():
        raise FloatingPointError("V27d-r7 causal dispersion is non-finite")
    return features, valid_fraction, event_valid, event_values.masked_fill(~event_valid, 0.0)


def _build_payload(root: Path, context: tuple[dict[str, Any], dict[str, Any]], role: str) -> dict[str, Any]:
    config, manifest = context
    source = config["source"]
    r6_path = source_path(root, source["r6_cache_template"].format(role=role), must_exist=True)
    r6 = load_r6_population_cache(r6_path)
    tensors = {str(name): value.detach().cpu().clone() for name, value in r6["tensors"].items()}
    dispersion, valid_fraction, event_valid, event_values = causal_dispersion_features(
        tensors["event_frame_values"],
        tensors["event_frame_mask"],
        tensors["padding_mask"],
        channel_index=int(config["model"]["dispersion_channel_index"]),
    )
    tensors["dispersion_features"] = dispersion
    tensors["dispersion_valid_fraction"] = valid_fraction
    tensors["dispersion_event_valid"] = event_valid
    tensors["dispersion_event_value"] = event_values
    payload: dict[str, Any] = {
        "schema_version": R7_CACHE_SCHEMA,
        "experiment_id": "V27d-r7-causal-stable8-dispersion",
        "role": role,
        "r7_manifest_sha256": manifest["manifest_sha256"],
        "source_r6_cache_path": relative_project_path(root, r6_path),
        "source_r6_cache_sha256": sha256_file(r6_path),
        "source_r6_payload_sha256": r6["cache_sha256"],
        "parent_ids": list(r6["parent_ids"]),
        "parent_ids_sha256": r6["parent_ids_sha256"],
        "label_names": list(r6["label_names"]),
        "dispersion_channel_index": int(config["model"]["dispersion_channel_index"]),
        "dispersion_names": list(DISPERSION_NAMES),
        "dispersion_valid_fraction_is_model_input": False,
        "unknown_is_classifier_class": False,
        "classification_logits_input_to_head": False,
        "softmax_input_to_head": False,
        "label_identity_input_to_head": False,
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
        "tensors": tensors,
    }
    payload["cache_sha256"] = cache_digest(payload)
    return payload


def _validate_cache_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    if payload.get("schema_version") != R7_CACHE_SCHEMA or payload.get("experiment_id") != "V27d-r7-causal-stable8-dispersion":
        raise ValueError("V27d-r7 cache identity differs")
    if payload.get("cache_sha256") != cache_digest(payload):
        raise ValueError("V27d-r7 cache hash differs")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r7 cache tensors are missing")
    count = len(payload["parent_ids"])
    width = 29
    required = {
        "summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence", "padding_mask", "lengths",
        "quality_target_sequence", "quality_target_final", "candidate_population", "bird_positive", "route_false_positive_negative", "knownness_target", "prefix_route_mask",
        "endpoint_final_logits", "l1_endpoint_sequence_logits", "l2_endpoint_sequence_logits", "stable8_causal_features", "event_frame_values", "event_frame_mask",
        "dispersion_features", "dispersion_valid_fraction", "dispersion_event_valid", "dispersion_event_value",
    }
    if set(tensors) != required:
        raise ValueError("V27d-r7 cache tensor fields differ")
    expected_shapes = {
        "summary_features": (count, width, 672), "event_trajectory": (count, width, 355), "task_facts_sequence": (count, width, 7), "event_facts_sequence": (count, width, 19),
        "padding_mask": (count, width), "lengths": (count,), "quality_target_sequence": (count, width), "quality_target_final": (count,), "candidate_population": (count,), "bird_positive": (count,), "route_false_positive_negative": (count,), "knownness_target": (count,), "prefix_route_mask": (count, width),
        "endpoint_final_logits": (count, 3), "l1_endpoint_sequence_logits": (count, width, 2), "l2_endpoint_sequence_logits": (count, width, 3), "stable8_causal_features": (count, width, 38), "event_frame_values": (count, width, 12, 8), "event_frame_mask": (count, width, 12),
        "dispersion_features": (count, width, 3), "dispersion_valid_fraction": (count, width), "dispersion_event_valid": (count, width), "dispersion_event_value": (count, width),
    }
    bool_fields = {"padding_mask", "quality_target_sequence", "quality_target_final", "candidate_population", "bird_positive", "route_false_positive_negative", "prefix_route_mask", "event_frame_mask", "dispersion_event_valid"}
    integer_fields = {"lengths"}
    for name, shape in expected_shapes.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r7 tensor shape differs: {name}")
        if name in bool_fields and tensor.dtype != torch.bool:
            raise ValueError(f"V27d-r7 bool dtype differs: {name}")
        if name in integer_fields and tensor.dtype != torch.long:
            raise ValueError(f"V27d-r7 integer dtype differs: {name}")
        if name not in bool_fields and name not in integer_fields and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d-r7 tensor values are invalid: {name}")
    if payload.get("dispersion_names") != list(DISPERSION_NAMES) or payload.get("dispersion_valid_fraction_is_model_input") is not False:
        raise ValueError("V27d-r7 dispersion contract differs")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r7 padding differs")
    for name in ("dispersion_features", "dispersion_valid_fraction", "dispersion_event_valid", "dispersion_event_value"):
        expanded = padding.unsqueeze(-1) if tensors[name].ndim == 3 else padding
        if bool(tensors[name].masked_select(expanded).ne(0).any()):
            raise ValueError(f"V27d-r7 padded dispersion is non-zero: {name}")
    expected_features, expected_fraction, expected_valid, expected_values = causal_dispersion_features(
        tensors["event_frame_values"], tensors["event_frame_mask"], padding, channel_index=int(payload["dispersion_channel_index"])
    )
    if not torch.equal(tensors["dispersion_features"], expected_features) or not torch.equal(tensors["dispersion_event_valid"], expected_valid) or not torch.equal(tensors["dispersion_event_value"], expected_values) or not torch.allclose(tensors["dispersion_valid_fraction"], expected_fraction, rtol=0.0, atol=1.0e-7):
        raise ValueError("V27d-r7 causal dispersion recomputation differs")
    return {**payload, "tensors": {str(name): tensor for name, tensor in tensors.items()}}


def load_population_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r7 cache is not an object")
    return _validate_cache_payload(payload)


def materialize_population_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    context = load_context(root, config_path)
    config, manifest = context
    results: dict[str, Any] = {}
    for role in R7_ROLES:
        payload = _build_payload(root, context, role)
        path = artifact_path(root, config["artifacts"]["cache_template"].format(role=role))
        write_torch_if_absent_or_identical(path, payload)
        checked = load_population_cache(path)
        results[role] = {"path": relative_project_path(root, path), "sha256": sha256_file(path), "cache_sha256": checked["cache_sha256"], "parent_count": len(checked["parent_ids"]), "candidate_count": int(checked["tensors"]["candidate_population"].sum())}
    return {"status": "materialized", "manifest_sha256": manifest["manifest_sha256"], "roles": results, "formal_l3_enabled": False}


__all__ = ["DISPERSION_NAMES", "cache_digest", "causal_dispersion_features", "load_population_cache", "materialize_population_caches"]
