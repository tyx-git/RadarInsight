"""Replay the frozen L1/L2 endpoints into V27d-owned caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import torch

from radar_ln3.runtime import canonical_sha256, sha256_file


from .pools import load_pools
from .protocol import V27DContext, V27D_EXPERIMENT_ID, V27D_ROLES, artifact_path, project_path, read_sealed_json


ENDPOINT_CACHE_SCHEMA = "RADAR-LN3-V27d-frozen-endpoint-cache-v1"
ENDPOINT_MANIFEST_SCHEMA = "RADAR-LN3-V27d-endpoint-cache-manifest-v1"


def _expected_parent_labels(
    pools: Mapping[str, Any],
    *,
    role: str,
    parent_ids: list[str],
) -> list[str]:
    """Return canonical labels in the exact pool parent order."""

    frame = pools.get("index_frame")
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("V27d pool index frame is missing")
    metadata = (
        frame[["parent_sample_id", "label_name"]]
        .drop_duplicates("parent_sample_id")
        .set_index("parent_sample_id")["label_name"]
        .astype(str)
        .to_dict()
    )
    labels = [metadata.get(parent_id) for parent_id in parent_ids]
    if any(label is None for label in labels):
        raise ValueError(f"V27d {role} parent label coverage differs")
    return [str(label) for label in labels]


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d endpoint tensor is not a tensor: {name}")
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


def _padded_values_are_zero(value: torch.Tensor, padding: torch.Tensor) -> bool:
    """Return whether all sequence-padding positions are represented by zero."""

    if value.ndim < padding.ndim or tuple(value.shape[: padding.ndim]) != tuple(padding.shape):
        raise ValueError("V27d endpoint padding cannot be applied to tensor")
    mask = padding.reshape(*padding.shape, *([1] * (value.ndim - padding.ndim))).expand_as(value)
    return not bool(value.masked_select(mask).ne(0).any())


def endpoint_cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d endpoint cache lacks tensors")
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256"}
    }
    return canonical_sha256(
        {"metadata": metadata, "tensor_sha256": _tensor_digest(dict(tensors))}
    )
















def _validate_sidecar(sidecar: Mapping[str, torch.Tensor], *, parent_count: int, width: int, frame_count: int) -> None:
    expected = {
        "stable8_summary": (parent_count, width, 8),
        "stable8_reliable": (parent_count, width),
        "stable8_reliable_count": (parent_count, width),
        "stable8_reliable_fraction": (parent_count, width),
        "rd_present": (parent_count, width),
        "event_frame_values": (parent_count, width, frame_count, 8),
        "event_frame_mask": (parent_count, width, frame_count),
    }
    if set(sidecar) != set(expected):
        raise ValueError("V27d endpoint sidecar fields differ")
    for name, shape in expected.items():
        value = sidecar[name]
        if tuple(value.shape) != shape:
            raise ValueError(f"V27d endpoint sidecar shape differs: {name}")
        if name in {"stable8_reliable", "rd_present", "event_frame_mask"}:
            if value.dtype != torch.bool:
                raise ValueError(f"V27d sidecar boolean field must be bool: {name}")
            continue
        if name == "stable8_reliable_count":
            if value.dtype != torch.long:
                raise ValueError("V27d sidecar reliable count must be int64")
            continue
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"V27d sidecar value is invalid: {name}")
    if bool((sidecar["stable8_reliable_count"] < 0).any()) or bool((sidecar["stable8_reliable_count"] > frame_count).any()):
        raise ValueError("V27d sidecar reliable count is out of bounds")
    if bool((sidecar["stable8_reliable_fraction"] < 0).any()) or bool((sidecar["stable8_reliable_fraction"] > 1).any()):
        raise ValueError("V27d sidecar reliable fraction is out of bounds")
    if not torch.equal(sidecar["stable8_reliable_count"], sidecar["event_frame_mask"].sum(dim=-1)):
        raise ValueError("V27d sidecar reliable count does not match frame mask")
    if not torch.equal(sidecar["stable8_reliable"], sidecar["stable8_reliable_count"].gt(0)):
        raise ValueError("V27d sidecar reliable flag does not match frame count")
    if bool(sidecar["event_frame_values"].masked_select(~sidecar["event_frame_mask"].unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d inactive frame values are non-zero")
    if bool(sidecar["stable8_summary"].masked_select(~sidecar["stable8_reliable"].unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d unreliable stable8 summaries are non-zero")








def load_endpoint_manifest(context: V27DContext) -> dict[str, Any]:
    path = artifact_path(context.project_root, "src-v3/v27/V27d/cache/V27d-endpoint-manifest.json")
    payload = read_sealed_json(path, expected_schema=ENDPOINT_MANIFEST_SCHEMA)
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    declared = body.pop("endpoint_manifest_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError("V27d endpoint manifest hash differs")
    if body.get("v27d_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27d endpoint manifest context differs")
    tasks = body.get("tasks")
    expected_tasks = {f"{task}:{role}" for task in ("known_gate", "dbb_specialist") for role in V27D_ROLES}
    if not isinstance(tasks, Mapping) or set(tasks) != expected_tasks:
        raise ValueError("V27d endpoint manifest task set differs")
    for key, entry in tasks.items():
        task, role = key.split(":", 1)
        if not isinstance(entry, Mapping):
            raise ValueError(f"V27d endpoint manifest entry is invalid: {key}")
        expected_path = artifact_path(
            context.project_root,
            context.config.artifacts.endpoint_cache_template,
            task=task,
            role=role,
        )
        declared_path = project_path(context.project_root, str(entry.get("path", "")))
        if declared_path != expected_path:
            raise ValueError(f"V27d endpoint manifest path differs: {key}")
        for field in ("sha256", "cache_sha256"):
            value = str(entry.get(field, "")).lower()
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"V27d endpoint manifest {field} differs: {key}")
    return payload


def _endpoint_manifest_entry(
    context: V27DContext,
    *,
    task: str,
    role: str,
    target: Path,
) -> Mapping[str, Any]:
    manifest = load_endpoint_manifest(context)
    entry = manifest["tasks"].get(f"{task}:{role}")
    expected = artifact_path(
        context.project_root,
        context.config.artifacts.endpoint_cache_template,
        task=task,
        role=role,
    )
    if not isinstance(entry, Mapping) or target != expected:
        raise ValueError(f"V27d endpoint cache path is not canonical: {task}/{role}")
    if sha256_file(target) != entry.get("sha256"):
        raise ValueError(f"V27d endpoint cache file hash differs: {task}/{role}")
    return entry


def load_endpoint_cache(
    context: V27DContext,
    *,
    path: str | Path,
    pools: Mapping[str, Any] | None = None,
    task: str,
    role: str,
    verify_manifest: bool = True,
) -> dict[str, Any]:
    if pools is None:
        pools = load_pools(context)
    target = project_path(context.project_root, path, must_exist=True)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d endpoint cache is not an object")
    value = dict(payload)
    declared = value.get("cache_sha256")
    if not isinstance(declared, str) or declared != endpoint_cache_digest(value):
        raise ValueError(f"V27d endpoint cache hash differs: {task}/{role}")
    if value.get("schema_version") != ENDPOINT_CACHE_SCHEMA or value.get("experiment_id") != V27D_EXPERIMENT_ID or value.get("task") != task or value.get("role") != role:
        raise ValueError("V27d endpoint cache identity differs")
    if value.get("unknown_is_classifier_class") is not False or value.get("unknown_classifier_training_exposure") != 0:
        raise ValueError("V27d endpoint Unknown classifier contract differs")
    if not isinstance(value.get("classification_endpoint"), str) or not value["classification_endpoint"].strip():
        raise ValueError("V27d endpoint classification provenance is missing")
    parent_ids = [str(item) for item in value.get("parent_ids") or ()]
    if not parent_ids:
        raise ValueError("V27d endpoint cache cannot be empty")
    expected_ids = [str(item) for item in pools["roles"][role]["parent_ids"]]
    if parent_ids != expected_ids or value.get("parent_ids_sha256") != canonical_sha256(parent_ids):
        raise ValueError("V27d endpoint cache parent identity differs")
    labels = [str(item) for item in value.get("label_names") or ()]
    if labels != _expected_parent_labels(pools, role=role, parent_ids=expected_ids):
        raise ValueError(f"V27d endpoint parent labels differ: {task}/{role}")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d endpoint cache tensors are missing")
    tensors = {str(name): item for name, item in tensors.items()}
    expected_names = {
        "summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence",
        "padding_mask", "lengths", "baseline_sequence_logits", "endpoint_sequence_logits",
        "stable8_summary", "stable8_reliable", "stable8_reliable_count", "stable8_reliable_fraction",
        "rd_present", "event_frame_values", "event_frame_mask",
    }
    if set(tensors) != expected_names:
        raise ValueError("V27d endpoint cache tensor fields differ")
    count = len(parent_ids)
    width = context.config.data.max_sequence_length
    hidden = context.config.data.liquid_hidden_dim
    task_fact_dim = context.config.data.l1_task_fact_dim if task == "known_gate" else context.config.data.l2_task_fact_dim
    classes = context.config.data.l1_class_count if task == "known_gate" else context.config.data.l2_class_count
    if value.get("max_sequence_length") != width or value.get("max_frames_per_event") != context.config.data.max_frames_per_event or value.get("liquid_hidden_dim") != context.config.data.liquid_hidden_dim or value.get("event_fact_dim") != context.config.data.event_fact_dim:
        raise ValueError("V27d endpoint cache configuration metadata differs")
    shapes = {
        "summary_features": (count, width, hidden * 14),
        "event_trajectory": (count, width, hidden * 7 + context.config.data.event_fact_dim),
        "task_facts_sequence": (count, width, task_fact_dim),
        "event_facts_sequence": (count, width, context.config.data.event_fact_dim),
        "padding_mask": (count, width), "lengths": (count,),
        "baseline_sequence_logits": (count, width, classes), "endpoint_sequence_logits": (count, width, classes),
        "stable8_summary": (count, width, 8), "stable8_reliable": (count, width),
        "stable8_reliable_count": (count, width), "stable8_reliable_fraction": (count, width),
        "rd_present": (count, width), "event_frame_values": (count, width, context.config.data.max_frames_per_event, 8),
        "event_frame_mask": (count, width, context.config.data.max_frames_per_event),
    }
    for name, shape in shapes.items():
        tensor_value = tensors[name]
        if not isinstance(tensor_value, torch.Tensor) or tuple(tensor_value.shape) != shape:
            raise ValueError(f"V27d endpoint tensor shape differs: {name}")
        if name in {"padding_mask", "stable8_reliable", "rd_present", "event_frame_mask"}:
            if tensor_value.dtype != torch.bool:
                raise ValueError(f"V27d endpoint bool dtype differs: {name}")
        elif name in {"lengths", "stable8_reliable_count"}:
            if tensor_value.dtype != torch.long:
                raise ValueError(f"V27d endpoint integer dtype differs: {name}")
        elif not tensor_value.is_floating_point() or not torch.isfinite(tensor_value).all():
            raise ValueError(f"V27d endpoint tensor is non-finite: {name}")
    lengths = tensors["lengths"]
    padding = tensors["padding_mask"]
    if bool((lengths < 1).any()) or bool((lengths > width).any()) or not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d endpoint lengths/padding differ")
    for name in ("summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence", "baseline_sequence_logits", "endpoint_sequence_logits", "stable8_summary", "stable8_reliable_count", "stable8_reliable_fraction", "rd_present", "event_frame_values", "event_frame_mask"):
        tensor_value = tensors[name]
        if not _padded_values_are_zero(tensor_value, padding):
            raise ValueError(f"V27d endpoint padded value is non-zero: {name}")
    _validate_sidecar(
        {name: tensors[name] for name in ("stable8_summary", "stable8_reliable", "stable8_reliable_count", "stable8_reliable_fraction", "rd_present", "event_frame_values", "event_frame_mask")},
        parent_count=count,
        width=width,
        frame_count=context.config.data.max_frames_per_event,
    )
    if verify_manifest:
        entry = _endpoint_manifest_entry(
            context,
            task=task,
            role=role,
            target=target,
        )
        if declared != entry.get("cache_sha256"):
            raise ValueError(f"V27d endpoint cache manifest binding differs: {task}/{role}")
    return {**payload, "tensors": tensors, "cache_sha256": declared}


__all__ = ["ENDPOINT_CACHE_SCHEMA", "ENDPOINT_MANIFEST_SCHEMA", "endpoint_cache_digest", "load_endpoint_cache", "load_endpoint_manifest"]