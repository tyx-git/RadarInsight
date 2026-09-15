"""V27d-r5-v2 population caches with audit-only endpoint evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical

from v27.V27d.endpoint import load_endpoint_cache
from v27.V27d.pools import load_pools
from v27.V27d.protocol import load_context as load_v27d_context
from v27.V27d.r5.features import load_population_cache as load_r5_population_cache

from .protocol import V2_CACHE_SCHEMA, V2_ROLES, artifact_path, load_context, source_path


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


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(role=role))


def _r5_cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return source_path(root, config["source"]["r5_population_cache_template"].format(role=role), must_exist=True)


def _endpoint_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return source_path(root, config["source"]["v27d_endpoint_cache_template"].format(role=role), must_exist=True)


def _build_payload(
    *,
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    role: str,
    r5_payload: Mapping[str, Any],
    endpoint_payload: Mapping[str, Any],
) -> dict[str, Any]:
    if role not in V2_ROLES:
        raise ValueError("V27d-r5-v2 role is invalid")
    if r5_payload.get("arm") != "control" or r5_payload.get("role") != role:
        raise ValueError("V27d-r5-v2 source is not the r5 control role cache")
    if r5_payload["parent_ids"] != endpoint_payload["parent_ids"]:
        raise ValueError("V27d-r5-v2 source parent order differs")
    if r5_payload["label_names"] != endpoint_payload["label_names"]:
        raise ValueError("V27d-r5-v2 source labels differ")
    r5_tensors = r5_payload["tensors"]
    endpoint_tensors = endpoint_payload["tensors"]
    required_r5 = {
        "summary_features", "event_trajectory", "task_facts_sequence", "stable8_causal_features",
        "padding_mask", "lengths", "quality_target_sequence", "quality_target_final",
        "bird_route_positive", "bird_route_negative", "bird_route_population", "bird_knownness_target",
        "l1_target_route", "l2_predicted_bird", "endpoint_final_logits", "classification_prediction",
    }
    if set(r5_tensors) != required_r5:
        raise ValueError("V27d-r5-v2 r5 source tensor fields differ")
    required_endpoint = {
        "event_facts_sequence", "endpoint_sequence_logits", "stable8_summary", "stable8_reliable",
        "stable8_reliable_count", "stable8_reliable_fraction", "rd_present", "event_frame_values", "event_frame_mask",
    }
    if not required_endpoint.issubset(endpoint_tensors):
        raise ValueError("V27d-r5-v2 endpoint audit fields are incomplete")
    if not torch.equal(r5_tensors["endpoint_final_logits"], endpoint_tensors["endpoint_sequence_logits"][torch.arange(len(r5_payload["parent_ids"])), r5_tensors["lengths"] - 1]):
        raise ValueError("V27d-r5-v2 endpoint final logits differ from source sequence logits")
    tensors = {name: value.clone() for name, value in r5_tensors.items()}
    for name in required_endpoint:
        tensors[name] = endpoint_tensors[name].clone()
    payload: dict[str, Any] = {
        "schema_version": V2_CACHE_SCHEMA,
        "experiment_id": "V27d-r5-bird-route-v2",
        "role": role,
        "direct_baseline": "V27d-r5-control",
        "source_r5_cache_path": relative_project_path(root, _r5_cache_path(root, config, role)),
        "source_r5_cache_sha256": sha256_file(_r5_cache_path(root, config, role)),
        "source_r5_cache_payload_sha256": r5_payload["cache_sha256"],
        "source_endpoint_cache_path": relative_project_path(root, _endpoint_path(root, config, role)),
        "source_endpoint_cache_sha256": sha256_file(_endpoint_path(root, config, role)),
        "source_endpoint_cache_payload_sha256": endpoint_payload["cache_sha256"],
        "v2_manifest_sha256": manifest["manifest_sha256"],
        "parent_ids": list(r5_payload["parent_ids"]),
        "parent_ids_sha256": r5_payload["parent_ids_sha256"],
        "label_names": list(r5_payload["label_names"]),
        "classification_logits_input_to_head": False,
        "softmax_input_to_head": False,
        "label_identity_input_to_head": False,
        "unknown_is_classifier_class": False,
        "formal_l3_enabled": False,
        "test_payload_access_count": 0,
        "tensors": tensors,
    }
    payload["cache_sha256"] = cache_digest(payload)
    return payload


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


def materialize_population_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    v27d_context = load_v27d_context(root, config["source"]["v27d_config_path"])
    pools = load_pools(v27d_context)
    tasks: dict[str, Any] = {}
    for role in V2_ROLES:
        r5_path = _r5_cache_path(root, config, role)
        r5_payload = load_r5_population_cache(r5_path)
        endpoint_payload = load_endpoint_cache(
            v27d_context,
            path=_endpoint_path(root, config, role),
            pools=pools,
            task="dbb_specialist",
            role=role,
        )
        payload = _build_payload(
            root=root,
            config=config,
            manifest=manifest,
            role=role,
            r5_payload=r5_payload,
            endpoint_payload=endpoint_payload,
        )
        path = _cache_path(root, config, role)
        write_torch_if_absent_or_identical(path, payload)
        checked = load_population_cache(path)
        tasks[role] = {
            "path": relative_project_path(root, path),
            "sha256": sha256_file(path),
            "cache_sha256": checked["cache_sha256"],
            "parent_count": len(checked["parent_ids"]),
            "bird_positive": int(checked["tensors"]["bird_route_positive"].sum()),
            "bird_route_unknown_negative": int(checked["tensors"]["bird_route_negative"].sum()),
            "quality_eligible": int(checked["tensors"]["quality_target_final"].sum()),
        }
    return {"status": "materialized", "experiment_id": "V27d-r5-bird-route-v2", "manifest_sha256": manifest["manifest_sha256"], "tasks": tasks}


__all__ = ["cache_digest", "load_population_cache", "materialize_population_caches"]
