"""Replay the frozen L1/L2 endpoints into V27d-owned caches."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from radar_ln3.runtime import (
    canonical_sha256,
    relative_project_path,
    sha256_file,
    write_torch_if_absent_or_identical,
)

from v27.v27a.cache import cache_payload_sha256, load_cache_payload
from v27.v27a.checkpoint import tensor_state_hash
from v27.v27a.evidence import load_normalization
from v27.v27a.model import build_v27a_model
from v27.v27a.protocol import load_v27a_context
from build.micro_doppler_v17 import _CanonicalFrameProvenance
from build.micro_doppler_v18 import _build_split

from .pools import load_frozen_binding, load_pools, load_source_manifests
from .protocol import (
    V27DContext,
    V27D_EXPERIMENT_ID,
    V27D_KNOWN_LABELS,
    V27D_ROLES,
    V27D_STABLE8_INDICES,
    artifact_path,
    project_path,
    read_sealed_json,
    write_sealed_json,
)


ENDPOINT_CACHE_SCHEMA = "RADAR-LN3-V27d-frozen-endpoint-cache-v1"
ENDPOINT_MANIFEST_SCHEMA = "RADAR-LN3-V27d-endpoint-cache-manifest-v1"
_V27D_NON_KNOWN_LABELS = frozenset({"ClutterTarget", "UnknownTarget"})
_V27A_CACHE_PAYLOAD_HASHES = {
    "train": "38b5b9067c9dfffd7dbec752d1a30e6f1785ce00bf542af777bcc41dbd0b8326",
    "val": "3fd22cd76049a697bd8cad454c6138998b8552b742131a647e2ae31628c39ebe",
}


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


def _load_representation(
    context: V27DContext,
    *,
    task: str,
    role: str,
    pools: Mapping[str, Any],
    manifests: Mapping[str, Any],
) -> dict[str, Any]:
    entry = manifests["representation"]["tasks"][f"{task}:{role}"]
    path = project_path(context.project_root, str(entry["path"]), must_exist=True)
    if sha256_file(path) != entry.get("sha256"):
        raise ValueError(f"V27d representation cache file hash differs: {task}/{role}")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("V27d representation cache is not an object")
    value = dict(raw)
    declared = value.get("cache_sha256")
    if not isinstance(declared, str) or declared != entry.get("cache_sha256") or declared != _legacy_representation_digest(value):
        raise ValueError(f"V27d representation cache digest differs: {task}/{role}")
    if value.get("task") != task or value.get("role") != role:
        raise ValueError("V27d representation cache task/role differs")
    expected_ids = [str(item) for item in pools["roles"][role]["parent_ids"]]
    parent_ids = [str(item) for item in value.get("parent_ids") or ()]
    if parent_ids != expected_ids:
        raise ValueError(f"V27d representation parent order differs: {task}/{role}")
    labels = [str(item) for item in value.get("label_names") or ()]
    if labels != _expected_parent_labels(pools, role=role, parent_ids=expected_ids):
        raise ValueError(f"V27d representation parent labels differ: {task}/{role}")
    if len(labels) != len(parent_ids) or any(item not in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"} for item in labels):
        raise ValueError("V27d representation labels differ")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d representation cache tensors are missing")
    tensors = {str(name): item for name, item in tensors.items()}
    expected_names = {
        "summary_features", "event_trajectory", "task_facts_sequence",
        "event_facts_sequence", "padding_mask", "lengths", "sequence_logits",
    }
    if set(tensors) != expected_names:
        raise ValueError("V27d representation tensor fields differ")
    count = len(parent_ids)
    width = int(context.config.data.max_sequence_length)
    hidden = int(context.config.data.liquid_hidden_dim)
    task_facts = int(context.config.data.l1_task_fact_dim if task == "known_gate" else context.config.data.l2_task_fact_dim)
    classes = int(context.config.data.l1_class_count if task == "known_gate" else context.config.data.l2_class_count)
    expected_shapes = {
        "summary_features": (count, width, hidden * 14),
        "event_trajectory": (count, width, hidden * 7 + context.config.data.event_fact_dim),
        "task_facts_sequence": (count, width, task_facts),
        "event_facts_sequence": (count, width, context.config.data.event_fact_dim),
        "padding_mask": (count, width),
        "lengths": (count,),
        "sequence_logits": (count, width, classes),
    }
    for name, shape in expected_shapes.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d representation tensor shape differs: {task}/{role}/{name}")
        if name == "padding_mask" and tensor.dtype != torch.bool:
            raise ValueError("V27d representation padding must be bool")
        elif name == "lengths" and tensor.dtype != torch.long:
            raise ValueError("V27d representation lengths must be int64")
        elif name not in {"padding_mask", "lengths"} and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d representation tensor is non-finite: {name}")
    lengths = tensors["lengths"]
    if bool((lengths < 1).any()) or bool((lengths > width).any()):
        raise ValueError("V27d representation length is out of bounds")
    expected_padding = torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(tensors["padding_mask"], expected_padding):
        raise ValueError("V27d representation padding differs from lengths")
    padding = tensors["padding_mask"]
    for name in ("summary_features", "event_trajectory", "task_facts_sequence", "event_facts_sequence", "sequence_logits"):
        if bool(tensors[name].masked_select(padding.unsqueeze(-1)).ne(0).any()):
            raise ValueError(f"V27d padded representation is non-zero: {name}")
    return {
        "path": path,
        "parent_ids": parent_ids,
        "label_names": labels,
        "tensors": tensors,
        "cache_sha256": declared,
    }


def _legacy_representation_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("legacy representation cache lacks tensors")
    metadata = {
        str(key): value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256"}
    }
    return canonical_sha256(
        {"metadata": metadata, "tensor_sha256": _tensor_digest(dict(tensors))}
    )


def _load_v27a_inputs(
    context: V27DContext,
) -> tuple[torch.nn.Module, dict[str, Mapping[str, Any]]]:
    root = context.project_root
    a_context = load_v27a_context(root, context.config.source.v27a_config_path)
    model = build_v27a_model(a_context.config, arm="main", seed=42)
    checkpoint_path = project_path(root, context.config.source.v27a_main_checkpoint_path, must_exist=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V27d V27a Main checkpoint is not an object")
    if (
        checkpoint.get("checkpoint_schema") != "RADAR-LN3-V27a-late-residual-checkpoint-v2"
        or checkpoint.get("experiment_id") != "V27a"
        or checkpoint.get("arm") != "main"
        or int(checkpoint.get("seed", -1)) != 42
        or int(checkpoint.get("epoch", -1)) != 20
        or checkpoint.get("config_sha256") != "ca71e5bda960f58012f56e158c2c9c9199b72defc3141866c0344fd0fc0b2723"
        or checkpoint.get("config_identity_sha256") != "07bdd372cebe74f95f9031355f0db641ac3f5ec7b2bb567f8c2396585a9cc0fc"
        or checkpoint.get("manifest_sha256") != "be939ffbe1a5f8ab0edfe7404eb5aee1aa278e1c12d6ae66326fd083d02afe80"
        or checkpoint.get("base_checkpoint_sha256") != context.config.source.expected_hashes["l2_checkpoint"]
    ):
        raise ValueError("V27d V27a Main checkpoint provenance differs")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or checkpoint.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27d V27a Main model state hash differs")
    if checkpoint.get("model_contract_sha256") != "7bfd3488b064e8479c5b0e71192a2e2ac44dc88e08f3fa34912e6fdcec6909db" or checkpoint.get("model_contract") != model.model_contract():
        raise ValueError("V27d V27a Main model contract differs")
    model.load_state_dict(state, strict=True)
    if checkpoint.get("model_state_sha256") != "2429d0fa3d68d7d7405192c4f9f2edfb8b27faf5e64921e4a542ff6780fb7be0":
        raise ValueError("V27d V27a Main checkpoint state hash differs")
    cache_paths = {
        "train": project_path(root, context.config.source.v27a_train_cache_path, must_exist=True),
        "val": project_path(root, context.config.source.v27a_validation_cache_path, must_exist=True),
    }
    loaded: dict[str, Any] = {}
    for split, path in cache_paths.items():
        payload = load_cache_payload(path)
        if sha256_file(path) != context.config.source.expected_hashes[f"v27a_{'train' if split == 'train' else 'validation'}_cache"]:
            raise ValueError(f"V27d V27a {split} cache file hash differs")
        if payload.get("cache_payload_sha256") != _V27A_CACHE_PAYLOAD_HASHES[split]:
            raise ValueError(f"V27d V27a {split} cache payload hash differs")
        loaded[split] = payload
    return model, loaded


def _empty_sidecar(parent_count: int, width: int, frame_count: int) -> dict[str, torch.Tensor]:
    return {
        "stable8_summary": torch.zeros((parent_count, width, 8), dtype=torch.float32),
        "stable8_reliable": torch.zeros((parent_count, width), dtype=torch.bool),
        "stable8_reliable_count": torch.zeros((parent_count, width), dtype=torch.long),
        "stable8_reliable_fraction": torch.zeros((parent_count, width), dtype=torch.float32),
        "rd_present": torch.zeros((parent_count, width), dtype=torch.bool),
        "event_frame_values": torch.zeros((parent_count, width, frame_count, 8), dtype=torch.float32),
        "event_frame_mask": torch.zeros((parent_count, width, frame_count), dtype=torch.bool),
    }


def _known_sidecar_row(
    payload: Mapping[str, Any],
    row: int,
    sidecar: dict[str, torch.Tensor],
    target_row: int,
    *,
    parent_id: str,
    label: str,
) -> None:
    length = int(payload["lengths"][row])
    if length < 1 or length > sidecar["stable8_summary"].shape[1]:
        raise ValueError(f"V27d known sidecar sequence length is out of bounds: {length}")
    parent_ids = [str(value) for value in payload.get("parent_sample_ids", ())]
    parent_labels = [str(value) for value in payload.get("parent_labels", ())]
    event_parent_ids = [str(value) for value in payload.get("event_parent_ids", ())]
    event_labels = [str(value) for value in payload.get("event_labels", ())]
    if row >= len(parent_ids) or parent_ids[row] != parent_id or row >= len(parent_labels) or parent_labels[row] != label:
        raise ValueError("V27d known sidecar parent identity or label differs")
    if len(event_parent_ids) != len(event_labels):
        raise ValueError("V27d known sidecar event metadata lengths differ")
    start = sum(int(value) for value in payload["lengths"][:row])
    stop = start + length
    if len(event_parent_ids) < stop or event_parent_ids[start:stop] != [parent_id] * length or event_labels[start:stop] != [label] * length:
        raise ValueError("V27d known sidecar event order or labels differ")
    for name in (
        "event_summary_tokens",
        "event_summary_valid",
        "event_reliable_frame_count",
        "event_reliable_frame_fraction",
        "event_rd_present",
        "event_frame_values",
        "event_frame_mask",
    ):
        if name not in payload:
            raise ValueError(f"V27d known sidecar payload lacks {name}")
        if len(payload[name][row]) < length:
            raise ValueError(f"V27d known sidecar payload is shorter than declared length: {name}")
    sidecar["stable8_summary"][target_row, :length] = torch.from_numpy(np.asarray(payload["event_summary_tokens"][row, :length], dtype=np.float32))
    sidecar["stable8_reliable"][target_row, :length] = torch.from_numpy(np.asarray(payload["event_summary_valid"][row, :length], dtype=np.bool_))
    sidecar["stable8_reliable_count"][target_row, :length] = torch.from_numpy(np.asarray(payload["event_reliable_frame_count"][row, :length], dtype=np.int64))
    sidecar["stable8_reliable_fraction"][target_row, :length] = torch.from_numpy(np.asarray(payload["event_reliable_frame_fraction"][row, :length], dtype=np.float32))
    sidecar["rd_present"][target_row, :length] = torch.from_numpy(np.asarray(payload["event_rd_present"][row, :length], dtype=np.bool_))
    sidecar["event_frame_values"][target_row, :length] = torch.from_numpy(np.asarray(payload["event_frame_values"][row, :length], dtype=np.float32))
    sidecar["event_frame_mask"][target_row, :length] = torch.from_numpy(np.asarray(payload["event_frame_mask"][row, :length], dtype=np.bool_))


def _extract_non_known_sidecar(
    context: V27DContext,
    *,
    role_record: Mapping[str, Any],
    index: pd.DataFrame,
    sidecar: dict[str, torch.Tensor],
    target_positions: Mapping[str, tuple[int, int]],
) -> dict[str, Any]:
    requested = [str(value) for value in role_record["sample_ids"]]
    selected_ids = {
        sample_id
        for sample_id in requested
        if str(index.loc[index["sample_id"].eq(sample_id), "label_name"].iloc[0]) in _V27D_NON_KNOWN_LABELS
    }
    if not selected_ids:
        return {"status": "not_required", "event_count": 0}
    selected = index[index["sample_id"].isin(selected_ids)].copy()
    selected = selected.sort_values(["parent_sample_id", "causal_event_index"], kind="stable").reset_index(drop=True)
    labels = set(selected["label_name"].astype(str))
    if not labels.issubset(_V27D_NON_KNOWN_LABELS):
        raise ValueError("V27d non-known extraction scope contains a known label")
    provenance_payload = json.loads(project_path(context.project_root, context.config.source.v16_provenance_path, must_exist=True).read_text(encoding="utf-8"))
    provenance = {str(item["parent_sample_id"]): dict(item) for item in provenance_payload.get("records", ())}
    canonical = _CanonicalFrameProvenance(
        context.project_root,
        project_path(context.project_root, context.config.source.canonical_frame_index_path, must_exist=True),
    )
    arrays, statistics = _build_split(
        context.project_root,
        str(role_record["split_name"]),
        selected,
        provenance,
        canonical,
        allowed_labels=tuple(sorted(labels)),
        include_evidence=True,
    )
    extracted_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    expected_ids = [value for value in requested if value in selected_ids]
    if extracted_ids != expected_ids:
        raise ValueError("V27d non-known evidence event order differs")
    normalization_path = project_path(context.project_root, context.config.source.v27a_normalization_path, must_exist=True)
    median, scale, _ = load_normalization(normalization_path)
    offsets = np.asarray(arrays["event_offsets"], dtype=np.int64)
    full_values = np.asarray(arrays["frame_values"], dtype=np.float32)
    reliable = np.asarray(arrays["frame_feature_reliable"], dtype=np.bool_)
    stable_values = full_values[:, V27D_STABLE8_INDICES]
    for event_index, sample_id in enumerate(extracted_ids):
        parent_row, event_row = target_positions[sample_id]
        start, stop = int(offsets[event_index]), int(offsets[event_index + 1])
        event_reliable = reliable[start:stop]
        selected_values = stable_values[start:stop].copy()
        normalized = np.zeros_like(selected_values, dtype=np.float32)
        if bool(event_reliable.any()):
            normalized[event_reliable] = ((selected_values[event_reliable] - median) / scale).astype(np.float32)
        count = int(event_reliable.sum())
        if count > sidecar["event_frame_values"].shape[2]:
            raise ValueError("V27d reliable frame count exceeds configured frame cap")
        if count:
            compact = torch.from_numpy(normalized[event_reliable])
            sidecar["event_frame_values"][parent_row, event_row, :count] = compact
            sidecar["event_frame_mask"][parent_row, event_row, :count] = True
            sidecar["stable8_summary"][parent_row, event_row] = compact.median(dim=0).values
        sidecar["stable8_reliable"][parent_row, event_row] = count > 0
        sidecar["stable8_reliable_count"][parent_row, event_row] = count
        sidecar["stable8_reliable_fraction"][parent_row, event_row] = count / float(stop - start) if stop > start else 0.0
        sidecar["rd_present"][parent_row, event_row] = bool(np.asarray(arrays["valid"])[event_index])
    return {
        "status": "label_free_v18_current_event_extraction",
        "event_count": len(extracted_ids),
        "statistics": json.loads(json.dumps(dict(statistics))),
        "raw_feature_values_depend_on_label": False,
    }


def _build_sidecars(
    context: V27DContext,
    pools: Mapping[str, Any],
    *,
    known_cache: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    sidecars: dict[str, dict[str, torch.Tensor]] = {}
    provenance: dict[str, Any] = {}
    index = pools["index_frame"]
    for role in V27D_ROLES:
        parent_ids = [str(value) for value in pools["roles"][role]["parent_ids"]]
        sidecar = _empty_sidecar(len(parent_ids), context.config.data.max_sequence_length, context.config.data.max_frames_per_event)
        target_positions: dict[str, tuple[int, int]] = {}
        for parent_row, parent_id in enumerate(parent_ids):
            rows = index[index["parent_sample_id"].eq(parent_id)].sort_values("causal_event_index", kind="stable")
            sample_ids = rows["sample_id"].astype(str).tolist()
            if not sample_ids or len(sample_ids) > context.config.data.max_sequence_length:
                raise ValueError(f"V27d {role} parent event count is out of bounds: {parent_id}")
            for event_row, sample_id in enumerate(sample_ids):
                if sample_id in target_positions:
                    raise ValueError(f"V27d {role} repeats an event id: {sample_id}")
                target_positions[sample_id] = (parent_row, event_row)
        split = "val" if role == "calibration" else "train"
        cache = known_cache[split]
        cache_positions = {str(parent): row for row, parent in enumerate(cache["parent_sample_ids"])}
        known_provenance: dict[str, Any] = {"status": "reused_v27a_frozen_known_cache", "parent_count": 0}
        for parent_row, parent_id in enumerate(parent_ids):
            label = str(index.loc[index["parent_sample_id"].eq(parent_id), "label_name"].iloc[0])
            if label in V27D_KNOWN_LABELS:
                cache_row = cache_positions.get(parent_id)
                if cache_row is None:
                    raise ValueError(f"V27d known parent absent from V27a cache: {parent_id}")
                _known_sidecar_row(
                    cache,
                    cache_row,
                    sidecar,
                    parent_row,
                    parent_id=parent_id,
                    label=label,
                )
                known_provenance["parent_count"] += 1
        non_known_provenance = _extract_non_known_sidecar(
            context,
            role_record=pools["roles"][role],
            index=index,
            sidecar=sidecar,
            target_positions=target_positions,
        )
        for name, value in sidecar.items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"V27d sidecar is non-finite: {role}/{name}")
        _validate_sidecar(
            sidecar,
            parent_count=len(parent_ids),
            width=context.config.data.max_sequence_length,
            frame_count=context.config.data.max_frames_per_event,
        )
        sidecars[role] = sidecar
        provenance[role] = {"known": known_provenance, "non_known": non_known_provenance}
    return sidecars, provenance


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


def _build_endpoint_payload(
    *,
    context: V27DContext,
    task: str,
    role: str,
    representation: Mapping[str, Any],
    endpoint_logits: torch.Tensor,
    sidecar: Mapping[str, torch.Tensor],
    provenance: Mapping[str, Any],
    baseline_logit_replay: Mapping[str, Any],
) -> dict[str, Any]:
    tensors = representation["tensors"]
    parent_ids = list(representation["parent_ids"])
    labels = list(representation["label_names"])
    lengths = tensors["lengths"].clone()
    padding = tensors["padding_mask"].clone()
    classes = context.config.data.l1_class_count if task == "known_gate" else context.config.data.l2_class_count
    if tuple(endpoint_logits.shape) != (len(parent_ids), context.config.data.max_sequence_length, classes):
        raise ValueError("V27d endpoint logits shape differs")
    endpoint_logits = endpoint_logits.detach().cpu().to(dtype=torch.float32)
    endpoint_logits = endpoint_logits.masked_fill(padding.unsqueeze(-1), 0.0)
    tensor_payload = {
        "summary_features": tensors["summary_features"].clone(),
        "event_trajectory": tensors["event_trajectory"].clone(),
        "task_facts_sequence": tensors["task_facts_sequence"].clone(),
        "event_facts_sequence": tensors["event_facts_sequence"].clone(),
        "padding_mask": padding,
        "lengths": lengths,
        "baseline_sequence_logits": tensors["sequence_logits"].clone(),
        "endpoint_sequence_logits": endpoint_logits,
        **{name: value.clone() for name, value in sidecar.items()},
    }
    payload: dict[str, Any] = {
        "schema_version": ENDPOINT_CACHE_SCHEMA,
        "experiment_id": V27D_EXPERIMENT_ID,
        "task": task,
        "role": role,
        "parent_ids": parent_ids,
        "parent_ids_sha256": canonical_sha256(parent_ids),
        "label_names": labels,
        "max_sequence_length": context.config.data.max_sequence_length,
        "max_frames_per_event": context.config.data.max_frames_per_event,
        "liquid_hidden_dim": context.config.data.liquid_hidden_dim,
        "event_fact_dim": context.config.data.event_fact_dim,
        "endpoint_policy": "l1_frozen_v27_epoch14_or_l2_v27a_main_epoch20",
        "classification_endpoint": "v27_baseline_l1_epoch14" if task == "known_gate" else "v27a_main_l2_epoch20",
        "unknown_is_classifier_class": False,
        "unknown_classifier_training_exposure": 0,
        "test_payload_access_count": 0,
        "sidecar_provenance": dict(provenance),
        "baseline_logit_replay": dict(baseline_logit_replay),
        "tensors": tensor_payload,
    }
    payload["cache_sha256"] = endpoint_cache_digest(payload)
    return payload


def _run_v27a_endpoint(
    model: torch.nn.Module,
    *,
    baseline_logits: torch.Tensor,
    sidecar: Mapping[str, torch.Tensor],
    padding: torch.Tensor,
) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        output = model(
            baseline_logits,
            sidecar["stable8_summary"],
            sidecar["stable8_reliable"],
            sidecar["event_frame_values"],
            sidecar["event_frame_mask"],
            padding,
        )
    result = output["sequence_logits"]
    if not torch.isfinite(result).all():
        raise FloatingPointError("V27d V27a endpoint produced non-finite logits")
    return result.detach().cpu()


def materialize_endpoint_caches(context: V27DContext) -> dict[str, Any]:
    pools = load_pools(context)
    manifests = load_source_manifests(context, pools)
    binding = load_frozen_binding(context)
    v27a_model, v27a_caches = _load_v27a_inputs(context)
    sidecars, sidecar_provenance = _build_sidecars(
        context,
        pools,
        known_cache=v27a_caches,
    )
    results: dict[str, Any] = {}
    for task in ("known_gate", "dbb_specialist"):
        for role in V27D_ROLES:
            representation = _load_representation(context, task=task, role=role, pools=pools, manifests=manifests)
            sidecar = sidecars[role]
            _validate_sidecar(sidecar, parent_count=len(representation["parent_ids"]), width=context.config.data.max_sequence_length, frame_count=context.config.data.max_frames_per_event)
            baseline = representation["tensors"]["sequence_logits"].clone()
            if task == "known_gate":
                endpoint = baseline.clone()
                replay = {"status": "not_applicable_l1", "max_abs_error": 0.0, "known_parent_count": 0}
            else:
                split = "val" if role == "calibration" else "train"
                cache = v27a_caches[split]
                cache_positions = {str(parent): row for row, parent in enumerate(cache["parent_sample_ids"])}
                base_for_endpoint = baseline.clone()
                max_error = 0.0
                known_count = 0
                for row, parent_id in enumerate(representation["parent_ids"]):
                    if str(representation["label_names"][row]) not in V27D_KNOWN_LABELS:
                        continue
                    cache_row = cache_positions.get(str(parent_id))
                    if cache_row is None:
                        raise ValueError(f"V27d V27a cache lacks known parent: {parent_id}")
                    length = int(representation["tensors"]["lengths"][row])
                    reference = torch.from_numpy(np.asarray(cache["event_baseline_logits"][cache_row, :length], dtype=np.float32))
                    observed = base_for_endpoint[row, :length]
                    max_error = max(max_error, float(torch.max(torch.abs(reference - observed)).item()))
                    base_for_endpoint[row, :length] = reference
                    known_count += 1
                if max_error > 3.0e-6:
                    raise ValueError(f"V27d baseline/V27a known-logit replay exceeds tolerance: {max_error}")
                endpoint = _run_v27a_endpoint(
                    v27a_model,
                    baseline_logits=base_for_endpoint,
                    sidecar=sidecar,
                    padding=representation["tensors"]["padding_mask"],
                )
                replay = {
                    "status": "known_parent_allclose_then_v27a_main_replay",
                    "max_abs_error": max_error,
                    "tolerance": 3.0e-6,
                    "known_parent_count": known_count,
                    "non_known_baseline_source": "frozen_v27_baseline_representation_sequence_logits",
                }
            payload = _build_endpoint_payload(
                context=context,
                task=task,
                role=role,
                representation=representation,
                endpoint_logits=endpoint,
                sidecar=sidecar,
                provenance=sidecar_provenance[role],
                baseline_logit_replay=replay,
            )
            path = artifact_path(context.project_root, context.config.artifacts.endpoint_cache_template, role=role, task=task)
            write_torch_if_absent_or_identical(path, payload)
            checked = load_endpoint_cache(
                context,
                path=path,
                pools=pools,
                task=task,
                role=role,
                verify_manifest=False,
            )
            results[f"{task}:{role}"] = {
                "path": relative_project_path(context.project_root, path),
                "sha256": sha256_file(path),
                "cache_sha256": checked["cache_sha256"],
                "parent_count": len(representation["parent_ids"]),
                "label_counts": dict(Counter(representation["label_names"])),
                "classification_endpoint": payload["classification_endpoint"],
                "baseline_replay": replay,
                "binding_seed": binding["seed"],
            }
    manifest: dict[str, Any] = {
        "schema_version": ENDPOINT_MANIFEST_SCHEMA,
        "experiment_id": V27D_EXPERIMENT_ID,
        "status": "prepared",
        "v27d_manifest_sha256": context.manifest_sha256,
        "source_hashes": dict(context.manifest["source_hashes"]),
        "frozen_binding_sha256": binding["binding_sha256"],
        "pool_registry_sha256": pools["registry_sha256"],
        "representation_manifest_file_sha256": sha256_file(project_path(context.project_root, context.config.source.baseline_representation_manifest_path, must_exist=True)),
        "tasks": results,
        "test_payload_access_count": 0,
    }
    manifest["endpoint_manifest_sha256"] = canonical_sha256(manifest)
    manifest_path = artifact_path(context.project_root, "src-v3/v27/V27d/cache/V27d-endpoint-manifest.json")
    write_sealed_json(manifest_path, manifest)
    for task in ("known_gate", "dbb_specialist"):
        for role in V27D_ROLES:
            entry = manifest["tasks"][f"{task}:{role}"]
            load_endpoint_cache(
                context,
                path=entry["path"],
                pools=pools,
                task=task,
                role=role,
            )
    return {**manifest, "manifest_path": str(manifest_path), "manifest_file_sha256": sha256_file(manifest_path)}


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


__all__ = [
    "ENDPOINT_CACHE_SCHEMA",
    "ENDPOINT_MANIFEST_SCHEMA",
    "endpoint_cache_digest",
    "load_endpoint_cache",
    "load_endpoint_manifest",
    "materialize_endpoint_caches",
]
