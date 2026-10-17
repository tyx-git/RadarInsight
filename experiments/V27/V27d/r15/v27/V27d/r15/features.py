"""Materialize and validate the V10 observation-increment sidecar for r15."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from radar_ln3.features import (
    TR_OBSERVATION_INCREMENT_CLIP,
    TR_OBSERVATION_INCREMENT_MASK_MODE,
    TR_OBSERVATION_INCREMENT_NAMES,
    TR_OBSERVATION_INCREMENT_SCHEMA,
    TR_OBSERVATION_INCREMENT_VALID_NAMES,
    encode_causal_tr_observation_increment_sequence,
)
from radar_ln3.runtime import canonical_sha256, sequence_sha256, sha256_file, write_torch_if_absent_or_identical
from v27.v27_baseline.data import V27TaskDataset
from v27.v27_baseline.protocol import V27_L1_TASK_NAME

from .protocol import R15_CACHE_MANIFEST_SCHEMA, R15_CACHE_SCHEMA, R15_CACHE_VIEWS, R15_ROLES, R15Context, artifact_path, project_path, read_json, write_sealed_json_if_identical


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


def _load_index(context: R15Context) -> pd.DataFrame:
    frame = pd.read_csv(project_path(context.project_root, context.config.source["index_path"], must_exist=True), low_memory=False)
    required = {"sample_id", "parent_sample_id", "causal_event_index", "label_name", "tr_feature_values_json"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V27d-r15 index lacks columns: {missing}")
    frame["sample_id"] = frame["sample_id"].astype(str)
    frame["parent_sample_id"] = frame["parent_sample_id"].astype(str)
    frame["causal_event_index"] = frame["causal_event_index"].astype(int)
    if frame["sample_id"].duplicated().any():
        raise ValueError("V27d-r15 index contains duplicate sample ids")
    return frame


def _endpoint_view(context: R15Context, role: str) -> dict[str, Any]:
    manifest = read_json(project_path(context.project_root, context.config.source["v27d_endpoint_manifest_path"], must_exist=True))
    entry = manifest.get("tasks", {}).get(f"known_gate:{role}")
    if not isinstance(entry, Mapping):
        raise ValueError(f"V27d-r15 endpoint view is missing: {role}")
    payload = torch.load(
        project_path(context.project_root, str(entry["path"]), must_exist=True),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"V27d-r15 endpoint cache is invalid: {role}")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError(f"V27d-r15 endpoint tensors are missing: {role}")
    return {
        "parent_ids": [str(value) for value in payload["parent_ids"]],
        "label_names": [str(value) for value in payload["label_names"]],
        "lengths": [int(value) for value in tensors["lengths"].tolist()],
        "source": f"endpoint:{role}",
    }


def _known_view(context: R15Context, split_name: str) -> dict[str, Any]:
    dataset = V27TaskDataset(
        context.project_root,
        _load_v27_config(context),
        task=V27_L1_TASK_NAME,
        split_name=split_name,
    )
    labels = [str(dataset.base._groups[parent_id].iloc[0]["label_name"]) for parent_id in dataset.parent_ids]
    lengths = [min(len(dataset.base._groups[parent_id]), int(context.config.data["max_sequence_length"])) for parent_id in dataset.parent_ids]
    return {
        "parent_ids": [str(value) for value in dataset.parent_ids],
        "label_names": labels,
        "lengths": lengths,
        "source": f"v27_known_gate:{split_name}",
    }


def _load_v27_config(context: R15Context) -> Any:
    # 延迟导入可避免 r15 的 sidecar 模块在 smoke 阶段提前初始化 V27 模型。
    from v27.v27_baseline.protocol import load_v27_config

    return load_v27_config(
        context.project_root,
        project_path(context.project_root, context.config.source["v27_config_path"], must_exist=True),
    )


def _views(context: R15Context) -> dict[str, dict[str, Any]]:
    result = {
        "train": _known_view(context, str(context.config.data["train_split_name"])),
        "validation": _known_view(context, str(context.config.data["validation_split_name"])),
    }
    for role in R15_ROLES:
        result[role] = _endpoint_view(context, role)
    return result


def _build_view_payload(
    frame: pd.DataFrame,
    *,
    parent_ids: Sequence[str],
    label_names: Sequence[str],
    lengths: Sequence[int],
    max_length: int,
) -> dict[str, Any]:
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V27d-r15 cache parent ids must be unique")
    if len(parent_ids) != len(label_names) or len(parent_ids) != len(lengths):
        raise ValueError("V27d-r15 cache parent metadata lengths differ")
    groups = {
        str(parent_id): group.sort_values("causal_event_index", kind="stable").reset_index(drop=True)
        for parent_id, group in frame.groupby("parent_sample_id", sort=False)
    }
    values = torch.zeros((len(parent_ids), max_length, len(TR_OBSERVATION_INCREMENT_NAMES)), dtype=torch.float32)
    valid = torch.zeros_like(values, dtype=torch.bool)
    padding = torch.ones((len(parent_ids), max_length), dtype=torch.bool)
    lengths_tensor = torch.zeros((len(parent_ids),), dtype=torch.long)
    event_ids: list[str] = []
    event_parent_ids: list[str] = []
    for row, (parent_id, label_name, expected_length) in enumerate(zip(parent_ids, label_names, lengths, strict=True)):
        group = groups.get(str(parent_id))
        if group is None:
            raise ValueError(f"V27d-r15 index lacks parent: {parent_id}")
        length = int(expected_length)
        if length <= 0 or length > max_length or len(group) < length:
            raise ValueError(f"V27d-r15 parent length differs: {parent_id}")
        selected = group.iloc[:length]
        if str(selected.iloc[0]["label_name"]) != str(label_name):
            raise ValueError(f"V27d-r15 parent label differs: {parent_id}")
        causal_indices = selected["causal_event_index"].astype(int).tolist()
        if causal_indices != list(range(length)):
            raise ValueError(f"V27d-r15 event indices are not contiguous: {parent_id}")
        raw_values = [json.loads(str(value)) for value in selected["tr_feature_values_json"].tolist()]
        encoded, encoded_valid = encode_causal_tr_observation_increment_sequence(raw_values)
        values[row, :length] = torch.from_numpy(encoded)
        valid[row, :length] = torch.from_numpy(encoded_valid)
        padding[row, :length] = False
        lengths_tensor[row] = length
        selected_ids = [str(value) for value in selected["sample_id"].tolist()]
        event_ids.extend(selected_ids)
        event_parent_ids.extend([str(parent_id)] * length)
    if bool(values.masked_select(~valid).ne(0).any()):
        raise ValueError("V27d-r15 invalid observation-increment values must be zero")
    if bool(values.masked_select(padding.unsqueeze(-1)).ne(0).any()) or bool(valid.masked_select(padding.unsqueeze(-1)).any()):
        raise ValueError("V27d-r15 padded observation-increment values must be zero/invalid")
    return {
        "parent_ids": [str(value) for value in parent_ids],
        "label_names": [str(value) for value in label_names],
        "lengths": lengths_tensor,
        "padding_mask": padding,
        "values": values,
        "valid": valid,
        "event_ids": event_ids,
        "event_parent_ids": event_parent_ids,
        "event_ids_order_sha256": sequence_sha256(event_ids),
        "event_parent_ids_order_sha256": sequence_sha256(event_parent_ids),
        "parent_ids_sha256": canonical_sha256([str(value) for value in parent_ids]),
    }


def materialize_observation_cache(context: R15Context) -> dict[str, Any]:
    frame = _load_index(context)
    views = _views(context)
    max_length = int(context.config.data["max_sequence_length"])
    view_payloads = {
        name: _build_view_payload(
            frame,
            parent_ids=view["parent_ids"],
            label_names=view["label_names"],
            lengths=view["lengths"],
            max_length=max_length,
        )
        for name, view in views.items()
    }
    payload: dict[str, Any] = {
        "schema_version": R15_CACHE_SCHEMA,
        "experiment_id": "V27d-r15",
        "observation_increment_schema": TR_OBSERVATION_INCREMENT_SCHEMA,
        "observation_increment_names": list(TR_OBSERVATION_INCREMENT_NAMES),
        "observation_increment_valid_names": list(TR_OBSERVATION_INCREMENT_VALID_NAMES),
        "observation_increment_clip_absolute": float(TR_OBSERVATION_INCREMENT_CLIP),
        "observation_increment_mask_mode": TR_OBSERVATION_INCREMENT_MASK_MODE,
        "v10_semantics": "per_report_change_not_per_second_rate; dt and dt_valid are not read",
        "source_index_sha256": sha256_file(project_path(context.project_root, context.config.source["index_path"], must_exist=True)),
        "v27d_manifest_sha256": context.manifest_sha256,
        "views": view_payloads,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
    }
    payload["cache_sha256"] = observation_cache_digest(payload)
    cache_path = artifact_path(context.project_root, context.config.artifacts["observation_cache_path"])
    write_torch_if_absent_or_identical(cache_path, payload)
    checked = load_observation_cache(context, cache_path)
    manifest_payload = {
        "schema_version": R15_CACHE_MANIFEST_SCHEMA,
        "experiment_id": "V27d-r15",
        "v27d_r15_manifest_sha256": context.manifest_sha256,
        "cache_path": str(cache_path.relative_to(context.project_root).as_posix()),
        "cache_file_sha256": sha256_file(cache_path),
        "cache_sha256": checked["cache_sha256"],
        "views": {
            name: {
                "parent_count": len(value["parent_ids"]),
                "event_count": len(value["event_ids"]),
                "source": views[name]["source"],
                "parent_ids_sha256": value["parent_ids_sha256"],
                "event_ids_order_sha256": value["event_ids_order_sha256"],
            }
            for name, value in checked["views"].items()
        },
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
    }
    manifest_path = artifact_path(context.project_root, context.config.artifacts["observation_cache_manifest_path"])
    write_sealed_json_if_identical(manifest_path, manifest_payload)
    return {
        "status": "materialized",
        "cache_path": str(cache_path),
        "cache_file_sha256": sha256_file(cache_path),
        "cache_sha256": checked["cache_sha256"],
        "manifest_path": str(manifest_path),
        "views": manifest_payload["views"],
    }


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


__all__ = ["materialize_observation_cache", "load_observation_cache", "observation_cache_digest"]