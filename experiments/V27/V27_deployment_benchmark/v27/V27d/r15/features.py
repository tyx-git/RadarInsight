"""Materialize and validate the V10 observation-increment sidecar for r15."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from radar_ln3.features import TR_OBSERVATION_INCREMENT_CLIP, TR_OBSERVATION_INCREMENT_MASK_MODE, TR_OBSERVATION_INCREMENT_NAMES, TR_OBSERVATION_INCREMENT_SCHEMA, TR_OBSERVATION_INCREMENT_VALID_NAMES
from radar_ln3.runtime import canonical_sha256

from .protocol import R15_CACHE_SCHEMA, R15_CACHE_VIEWS, R15Context, artifact_path, project_path


def _tensor_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    parts: list[Any] = []
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r15 cache tensor is invalid: {name}")
        item = value.detach().cpu().contiguous()
        parts.extend((name, str(item.dtype), repr(tuple(item.shape)), item.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def observation_cache_digest(payload: Mapping[str, Any]) -> str:
    views = payload.get("views")
    if not isinstance(views, Mapping):
        raise ValueError("V27d-r15 cache lacks views")
    metadata = {key: value for key, value in payload.items() if key not in {"views", "cache_sha256"}}
    view_digests: dict[str, Any] = {}
    for name, view in views.items():
        if not isinstance(view, Mapping):
            raise ValueError(f"V27d-r15 cache view is invalid: {name}")
        tensors = {
            key: view[key]
            for key in ("lengths", "padding_mask", "values", "valid")
            if key in view
        }
        if len(tensors) != 4:
            raise ValueError(f"V27d-r15 cache view tensors are incomplete: {name}")
        view_metadata = {
            key: value
            for key, value in view.items()
            if key not in {"lengths", "padding_mask", "values", "valid"}
        }
        view_digests[str(name)] = {
            "metadata": view_metadata,
            "tensor_sha256": _tensor_digest(tensors),
        }
    return canonical_sha256({"metadata": metadata, "view_digests": view_digests})
















def _validate_view(view: Mapping[str, Any], *, max_length: int) -> None:
    parent_ids = [str(value) for value in view.get("parent_ids") or ()]
    labels = [str(value) for value in view.get("label_names") or ()]
    if not parent_ids or len(parent_ids) != len(labels) or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V27d-r15 cache parent metadata is invalid")
    lengths = view.get("lengths")
    values = view.get("values")
    valid = view.get("valid")
    padding = view.get("padding_mask")
    if not all(isinstance(item, torch.Tensor) for item in (lengths, values, valid, padding)):
        raise ValueError("V27d-r15 cache view tensors are missing")
    expected = (len(parent_ids), max_length)
    if tuple(lengths.shape) != (len(parent_ids),) or tuple(values.shape) != (*expected, 9) or tuple(valid.shape) != (*expected, 9) or tuple(padding.shape) != expected:
        raise ValueError("V27d-r15 cache view tensor shape differs")
    if lengths.dtype != torch.long or values.dtype != torch.float32 or valid.dtype != torch.bool or padding.dtype != torch.bool:
        raise ValueError("V27d-r15 cache view tensor dtype differs")
    if not torch.isfinite(values).all() or bool(values.abs().gt(TR_OBSERVATION_INCREMENT_CLIP + 1.0e-6).any()):
        raise ValueError("V27d-r15 observation-increment values are invalid")
    if bool(values.masked_select(~valid).ne(0).any()) or bool(values.masked_select(padding.unsqueeze(-1)).ne(0).any()) or bool(valid.masked_select(padding.unsqueeze(-1)).any()):
        raise ValueError("V27d-r15 cache validity/padding contract differs")
    expected_padding = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(padding, expected_padding):
        raise ValueError("V27d-r15 cache padding differs from lengths")
    if view.get("parent_ids_sha256") != canonical_sha256(parent_ids):
        raise ValueError("V27d-r15 cache parent hash differs")


def load_observation_cache(context: R15Context, path: str | Path | None = None) -> dict[str, Any]:
    target = artifact_path(context.project_root, context.config.artifacts["observation_cache_path"]) if path is None else project_path(context.project_root, path, must_exist=True)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r15 observation cache is not an object")
    value = dict(payload)
    if value.get("cache_sha256") != observation_cache_digest(value):
        raise ValueError("V27d-r15 observation cache hash differs")
    if value.get("schema_version") != R15_CACHE_SCHEMA or value.get("experiment_id") != "V27d-r15":
        raise ValueError("V27d-r15 observation cache identity differs")
    if value.get("observation_increment_schema") != TR_OBSERVATION_INCREMENT_SCHEMA or value.get("observation_increment_names") != list(TR_OBSERVATION_INCREMENT_NAMES) or value.get("observation_increment_valid_names") != list(TR_OBSERVATION_INCREMENT_VALID_NAMES):
        raise ValueError("V27d-r15 observation-increment contract differs")
    if value.get("observation_increment_mask_mode") != TR_OBSERVATION_INCREMENT_MASK_MODE or value.get("unknown_training_exposure") != 0 or value.get("test_payload_access_count") != 0:
        raise ValueError("V27d-r15 cache scope contract differs")
    views = value.get("views")
    if not isinstance(views, Mapping) or set(views) != set(R15_CACHE_VIEWS):
        raise ValueError("V27d-r15 cache view set differs")
    for name in R15_CACHE_VIEWS:
        _validate_view(views[name], max_length=int(context.config.data["max_sequence_length"]))
    return value


__all__ = ["load_observation_cache", "observation_cache_digest"]