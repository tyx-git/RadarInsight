"""Causal full-stable8 vector-shape features for V27d-r11."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import (
    canonical_sha256,
    relative_project_path,
    sha256_file,
    verify_file_hash,
    write_torch_if_absent_or_identical,
)
from v27.V27d.r6.features import load_population_cache as load_r6_population_cache

from .protocol import R11_CACHE_SCHEMA_VERSION, R11_EXPERIMENT_ID, R11_ROLES, R11_SHAPE_FEATURE_DIM, R11_STABLE8_DIM, artifact_path, load_context, source_path


_R6_TENSOR_NAMES = {
    "summary_features",
    "event_trajectory",
    "task_facts_sequence",
    "event_facts_sequence",
    "padding_mask",
    "lengths",
    "quality_target_sequence",
    "quality_target_final",
    "candidate_population",
    "bird_positive",
    "route_false_positive_negative",
    "knownness_target",
    "prefix_route_mask",
    "endpoint_final_logits",
    "l1_endpoint_sequence_logits",
    "l2_endpoint_sequence_logits",
    "stable8_causal_features",
    "event_frame_values",
    "event_frame_mask",
}


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r11 cache tensor is not a tensor: {name}")
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


def cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r11 cache lacks tensors")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256"}
    }
    return canonical_sha256(
        {"metadata": metadata, "tensor_sha256": _tensor_digest(tensors)}
    )


def causal_stable8_shape_features(
    event_frame_values: torch.Tensor,
    event_frame_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build current and prefix-only stable8 vector-shape features.

    Each reliable frame vector is L2-normalized before the event mean is
    formed.  Event vectors then enter causal running mean/std and adjacent
    absolute-delta statistics.  The validity mask is used only to decide
    whether an observation updates the state; it is not concatenated as a
    numeric feature.
    """

    if event_frame_values.ndim != 4 or event_frame_mask.ndim != 3 or padding_mask.ndim != 2:
        raise ValueError("V27d-r11 stable8 frame tensors have invalid rank")
    if event_frame_values.shape[:3] != event_frame_mask.shape or event_frame_values.shape[:2] != padding_mask.shape:
        raise ValueError("V27d-r11 stable8 frame tensor shapes differ")
    if event_frame_values.shape[-1] != R11_STABLE8_DIM or event_frame_mask.dtype != torch.bool or padding_mask.dtype != torch.bool:
        raise ValueError("V27d-r11 stable8 frame contract differs")
    if not event_frame_values.is_floating_point() or not torch.isfinite(event_frame_values).all():
        raise ValueError("V27d-r11 stable8 frame values must be finite floats")
    if bool(event_frame_values.masked_select(padding_mask.unsqueeze(-1).unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d-r11 padded frame values must be zero")
    if bool(event_frame_mask.masked_select(padding_mask.unsqueeze(-1)).any()):
        raise ValueError("V27d-r11 padded frame masks must be inactive")
    inactive = (~event_frame_mask).unsqueeze(-1)
    if bool(event_frame_values.masked_select(inactive).ne(0).any()):
        raise ValueError("V27d-r11 inactive frame values must be zero")

    batch, length, frame_count, width = event_frame_values.shape
    selected = event_frame_values.reshape(batch * length, frame_count, width)
    selected_mask = event_frame_mask.reshape(batch * length, frame_count)
    norms = selected.norm(dim=-1, keepdim=True)
    if bool(norms.masked_select(selected_mask.unsqueeze(-1)).le(1.0e-12).any()):
        raise ValueError("V27d-r11 reliable stable8 vectors must have positive norm")
    normalized = selected / norms.clamp_min(1.0e-12)
    normalized = normalized.masked_fill(~selected_mask.unsqueeze(-1), 0.0)
    counts = selected_mask.sum(dim=-1).reshape(batch, length)
    event_valid = counts.gt(0) & ~padding_mask
    event_vectors = (
        normalized.sum(dim=1).reshape(batch, length, width)
        / counts.reshape(batch, length, 1).clamp_min(1).to(event_frame_values.dtype)
    )
    event_norms = event_vectors.norm(dim=-1, keepdim=True)
    event_vectors = event_vectors / event_norms.clamp_min(1.0e-12)
    event_vectors = event_vectors.masked_fill(~event_valid.unsqueeze(-1), 0.0)

    running_count = torch.zeros((batch, 1), device=event_vectors.device, dtype=event_vectors.dtype)
    running_sum = torch.zeros((batch, width), device=event_vectors.device, dtype=event_vectors.dtype)
    running_square_sum = torch.zeros_like(running_sum)
    previous = torch.zeros_like(running_sum)
    previous_valid = torch.zeros((batch, 1), device=event_vectors.device, dtype=torch.bool)
    rows: list[torch.Tensor] = []
    for event in range(length):
        valid = event_valid[:, event]
        indicator = valid.unsqueeze(-1).to(event_vectors.dtype)
        current = event_vectors[:, event]
        next_count = running_count + indicator
        next_sum = running_sum + indicator * current
        next_square_sum = running_square_sum + indicator * current.square()
        mean = next_sum / next_count.clamp_min(1.0)
        std = (next_square_sum / next_count.clamp_min(1.0) - mean.square()).clamp_min(0.0).sqrt()
        delta = torch.where(previous_valid, (current - previous).abs(), torch.zeros_like(current))
        rows.append(torch.cat((current, mean, std, delta), dim=-1).masked_fill(~valid.unsqueeze(-1), 0.0))
        running_count, running_sum, running_square_sum = next_count, next_sum, next_square_sum
        previous = torch.where(indicator.bool(), current, previous)
        previous_valid = previous_valid | valid.unsqueeze(-1)
    features = torch.stack(rows, dim=1)
    if features.shape[-1] != R11_SHAPE_FEATURE_DIM or not torch.isfinite(features).all():
        raise FloatingPointError("V27d-r11 causal stable8 shape features are invalid")
    return features, event_valid, counts.to(dtype=torch.long)


def _build_payload(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    role: str,
    r6_payload: Mapping[str, Any],
    r6_path: Path,
) -> dict[str, Any]:
    if role not in R11_ROLES:
        raise ValueError("V27d-r11 role is invalid")
    tensors = {
        str(name): value.detach().cpu().clone()
        for name, value in r6_payload["tensors"].items()
    }
    if set(tensors) != _R6_TENSOR_NAMES:
        raise ValueError("V27d-r11 source r6 tensor fields differ")
    features, valid, counts = causal_stable8_shape_features(
        tensors["event_frame_values"],
        tensors["event_frame_mask"],
        tensors["padding_mask"],
    )
    tensors["shape_features"] = features.detach().cpu().clone()
    tensors["shape_event_valid"] = valid.detach().cpu().clone()
    tensors["shape_event_frame_count"] = counts.detach().cpu().clone()
    source = config["source"]
    payload: dict[str, Any] = {
        "schema_version": R11_CACHE_SCHEMA_VERSION,
        "experiment_id": R11_EXPERIMENT_ID,
        "role": role,
        "r11_manifest_sha256": manifest["manifest_sha256"],
        "source_r6_cache_path": relative_project_path(root, r6_path),
        "source_r6_cache_sha256": sha256_file(r6_path),
        "source_r6_payload_sha256": r6_payload["cache_sha256"],
        "source_r6_manifest_sha256": source["expected_r6_manifest_sha256"],
        "parent_ids": [str(value) for value in r6_payload["parent_ids"]],
        "parent_ids_sha256": r6_payload["parent_ids_sha256"],
        "label_names": [str(value) for value in r6_payload["label_names"]],
        "shape_feature_names": [
            "current_event_shape",
            "causal_running_mean_shape",
            "causal_running_std_shape",
            "previous_valid_event_absolute_delta",
        ],
        "shape_feature_dim": R11_SHAPE_FEATURE_DIM,
        "stable8_dim": R11_STABLE8_DIM,
        "shape_normalization": "per-reliable-frame-l2-then-event-mean-l2",
        "shape_valid_is_numeric_input": False,
        "quality_facts_are_numeric_input": False,
        "classification_logits_input_to_branch": False,
        "softmax_input_to_branch": False,
        "label_identity_input_to_branch": False,
        "unknown_is_classifier_class": False,
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
        "tensors": tensors,
    }
    payload["cache_sha256"] = cache_digest(payload)
    return payload


def _validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if value.get("schema_version") != R11_CACHE_SCHEMA_VERSION or value.get("experiment_id") != R11_EXPERIMENT_ID:
        raise ValueError("V27d-r11 cache identity differs")
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r11 cache hash differs")
    if value.get("shape_feature_dim") != R11_SHAPE_FEATURE_DIM or value.get("stable8_dim") != R11_STABLE8_DIM:
        raise ValueError("V27d-r11 shape dimensions differ")
    if value.get("shape_feature_names") != [
        "current_event_shape",
        "causal_running_mean_shape",
        "causal_running_std_shape",
        "previous_valid_event_absolute_delta",
    ] or value.get("shape_normalization") != "per-reliable-frame-l2-then-event-mean-l2":
        raise ValueError("V27d-r11 shape normalization contract differs")
    if value.get("shape_valid_is_numeric_input") is not False or value.get("quality_facts_are_numeric_input") is not False:
        raise ValueError("V27d-r11 shape validity contract differs")
    if value.get("formal_l3_enabled") is not False or value.get("test_payload_access_count") != 0:
        raise ValueError("V27d-r11 cache scope differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != _R6_TENSOR_NAMES | {"shape_features", "shape_event_valid", "shape_event_frame_count"}:
        raise ValueError("V27d-r11 cache tensor fields differ")
    tensors = {str(name): tensor for name, tensor in tensors.items()}
    parent_ids = [str(item) for item in value.get("parent_ids", ())]
    labels = [str(item) for item in value.get("label_names", ())]
    if not parent_ids or len(parent_ids) != len(set(parent_ids)) or len(labels) != len(parent_ids):
        raise ValueError("V27d-r11 cache parent metadata differs")
    if any(item not in {"DroneTarget", "BirdTarget", "BalloonTarget", "UnknownTarget", "ClutterTarget"} for item in labels):
        raise ValueError("V27d-r11 cache label metadata differs")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool or padding.ndim != 2 or padding.shape[0] != len(parent_ids):
        raise ValueError("V27d-r11 cache padding differs")
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long or lengths.shape != (len(parent_ids),) or bool(lengths.le(0).any()):
        raise ValueError("V27d-r11 cache lengths differ")
    width = int(padding.shape[1])
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r11 cache padding must be a suffix")
    count = len(parent_ids)
    expected_shapes = {
        "summary_features": (count, width, 672),
        "event_trajectory": (count, width, 355),
        "task_facts_sequence": (count, width, 7),
        "event_facts_sequence": (count, width, 19),
        "padding_mask": (count, width),
        "lengths": (count,),
        "quality_target_sequence": (count, width),
        "quality_target_final": (count,),
        "candidate_population": (count,),
        "bird_positive": (count,),
        "route_false_positive_negative": (count,),
        "knownness_target": (count,),
        "prefix_route_mask": (count, width),
        "endpoint_final_logits": (count, 3),
        "l1_endpoint_sequence_logits": (count, width, 2),
        "l2_endpoint_sequence_logits": (count, width, 3),
        "stable8_causal_features": (count, width, 38),
        "event_frame_values": (count, width, 12, 8),
        "event_frame_mask": (count, width, 12),
        "shape_features": (count, width, R11_SHAPE_FEATURE_DIM),
        "shape_event_valid": (count, width),
        "shape_event_frame_count": (count, width),
    }
    bool_fields = {
        "padding_mask",
        "quality_target_sequence",
        "quality_target_final",
        "candidate_population",
        "bird_positive",
        "route_false_positive_negative",
        "prefix_route_mask",
        "event_frame_mask",
        "shape_event_valid",
    }
    integer_fields = {"lengths", "shape_event_frame_count"}
    for name, shape in expected_shapes.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r11 cache tensor shape differs: {name}")
        if name in bool_fields and tensor.dtype != torch.bool:
            raise ValueError(f"V27d-r11 boolean dtype differs: {name}")
        if name in integer_fields and tensor.dtype != torch.long:
            raise ValueError(f"V27d-r11 integer dtype differs: {name}")
        if name not in bool_fields and name not in integer_fields and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d-r11 floating tensor is invalid: {name}")
    for name in ("shape_features",):
        if bool(tensors[name].masked_select(padding.unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27d-r11 padded shape tensor is non-zero: {name}")
    if bool(tensors["shape_features"].masked_select(~tensors["shape_event_valid"].unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d-r11 invalid shape features must be zero")
    if bool(tensors["shape_event_frame_count"].lt(0).any()) or bool(tensors["shape_event_frame_count"].gt(12).any()):
        raise ValueError("V27d-r11 shape frame counts are out of bounds")
    if not torch.equal(tensors["shape_event_valid"], tensors["shape_event_frame_count"].gt(0) & ~padding):
        raise ValueError("V27d-r11 shape validity/count semantics differ")
    expected_features, expected_valid, expected_counts = causal_stable8_shape_features(
        tensors["event_frame_values"], tensors["event_frame_mask"], padding
    )
    if not torch.equal(tensors["shape_event_valid"], expected_valid) or not torch.equal(tensors["shape_event_frame_count"], expected_counts) or not torch.allclose(tensors["shape_features"], expected_features, rtol=0.0, atol=1.0e-6):
        raise ValueError("V27d-r11 causal shape feature recomputation differs")
    return {**value, "tensors": tensors}


def load_population_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r11 cache is not a mapping")
    return _validate_payload(payload)


def validate_cache_binding(payload: Mapping[str, Any], *, root: Path, config: Mapping[str, Any], role: str) -> None:
    if role not in R11_ROLES:
        raise ValueError("V27d-r11 role is invalid")
    source = config["source"]
    source_cache = source_path(root, source["r6_cache_template"].format(role=role), must_exist=True)
    expected_file = source["expected_cache_file_sha256s"][role]
    expected_payload = source["expected_cache_payload_sha256s"][role]
    if payload.get("source_r6_cache_sha256") != expected_file or payload.get("source_r6_payload_sha256") != expected_payload:
        raise ValueError(f"V27d-r11 {role} source cache provenance differs")
    if sha256_file(source_cache) != expected_file:
        raise ValueError(f"V27d-r11 {role} source cache changed")


def materialize_population_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    source = config["source"]
    results: dict[str, Any] = {}
    for role in R11_ROLES:
        r6_path = source_path(root, source["r6_cache_template"].format(role=role), must_exist=True)
        verify_file_hash(r6_path, source["expected_cache_file_sha256s"][role], field=f"V27d-r11 r6 {role} cache")
        r6_payload = load_r6_population_cache(r6_path)
        if r6_payload.get("cache_sha256") != source["expected_cache_payload_sha256s"][role]:
            raise ValueError(f"V27d-r11 r6 {role} payload hash differs")
        payload = _build_payload(
            root=root,
            config=config,
            manifest=manifest,
            role=role,
            r6_payload=r6_payload,
            r6_path=r6_path,
        )
        target = artifact_path(root, config["artifacts"]["cache_template"].format(role=role))
        write_torch_if_absent_or_identical(target, payload)
        checked = load_population_cache(target)
        results[role] = {
            "path": relative_project_path(root, target),
            "sha256": sha256_file(target),
            "cache_sha256": checked["cache_sha256"],
            "parent_count": len(checked["parent_ids"]),
            "candidate_count": int(checked["tensors"]["candidate_population"].sum()),
            "shape_valid_event_count": int(checked["tensors"]["shape_event_valid"].sum()),
        }
    return {
        "status": "materialized_v27d_r11_population_caches",
        "experiment_id": R11_EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "roles": results,
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
    }


__all__ = ["causal_stable8_shape_features", "cache_digest", "load_population_cache", "materialize_population_caches", "validate_cache_binding"]