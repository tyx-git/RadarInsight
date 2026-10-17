"""Route-matched Bird verifier population caches for V27d-r6."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical

from v27.V27d.endpoint import load_endpoint_cache
from v27.V27d.pools import load_pools
from v27.V27d.protocol import load_context as load_v27d_context
from v27.V27d.r5_v2.features import load_population_cache as load_r5_v2_cache

from .protocol import R6_CACHE_SCHEMA, R6_ROLES, artifact_path, load_context, source_path


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


def _r5_v2_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return source_path(root, config["source"]["r5_v2_cache_template"].format(role=role), must_exist=True)


def _l1_endpoint_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return source_path(root, config["source"]["v27d_l1_endpoint_cache_template"].format(role=role), must_exist=True)


def _build_payload(*, root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], role: str, r5: Mapping[str, Any], l1_endpoint: Mapping[str, Any]) -> dict[str, Any]:
    if role not in R6_ROLES:
        raise ValueError("V27d-r6 role is invalid")
    if r5["parent_ids"] != l1_endpoint["parent_ids"] or r5["label_names"] != l1_endpoint["label_names"]:
        raise ValueError("V27d-r6 parent/label order differs between source caches")
    r5_tensors = r5["tensors"]
    l1_tensors = l1_endpoint["tensors"]
    lengths = r5_tensors["lengths"]
    final = lengths - 1
    rows = torch.arange(len(r5["parent_ids"]))
    l1_final_prediction = l1_tensors["endpoint_sequence_logits"][rows, final].argmax(dim=-1)
    l1_target_route = l1_final_prediction.eq(0)
    quality_final = r5_tensors["quality_target_final"].bool()
    l2_bird = r5_tensors["l2_predicted_bird"].bool()
    candidate = quality_final & l1_target_route & l2_bird
    positive = candidate & torch.tensor([label == "BirdTarget" for label in r5["label_names"]], dtype=torch.bool)
    negative = candidate & ~positive
    if not torch.equal(candidate, positive | negative) or bool((positive & negative).any()):
        raise ValueError("V27d-r6 population masks are invalid")
    # Prefix route facts are copied from the frozen L1 endpoint only as
    # discrete masks. They are never exposed as continuous verifier inputs.
    l1_prefix_prediction = l1_tensors["endpoint_sequence_logits"].argmax(dim=-1)
    l2_prefix_prediction = r5_tensors["endpoint_sequence_logits"].argmax(dim=-1)
    event_number = torch.arange(1, r5_tensors["padding_mask"].shape[1] + 1).unsqueeze(0)
    prefix_mask = (
        ~r5_tensors["padding_mask"]
        & r5_tensors["quality_target_sequence"].bool()
        & event_number.ge(int(config["training"]["minimum_quality_events"]))
        & l1_prefix_prediction.eq(0)
        & l2_prefix_prediction.eq(1)
    )
    labels = list(r5["label_names"])
    tensors = {
        "summary_features": r5_tensors["summary_features"].clone(),
        "event_trajectory": r5_tensors["event_trajectory"].clone(),
        "task_facts_sequence": r5_tensors["task_facts_sequence"].clone(),
        "event_facts_sequence": r5_tensors["event_facts_sequence"].clone(),
        "padding_mask": r5_tensors["padding_mask"].clone(),
        "lengths": lengths.clone(),
        "quality_target_sequence": r5_tensors["quality_target_sequence"].clone(),
        "quality_target_final": quality_final.clone(),
        "candidate_population": candidate.clone(),
        "bird_positive": positive.clone(),
        "route_false_positive_negative": negative.clone(),
        "knownness_target": positive.to(dtype=torch.float32),
        "prefix_route_mask": prefix_mask,
        "endpoint_final_logits": r5_tensors["endpoint_final_logits"].clone(),
        "l1_endpoint_sequence_logits": l1_tensors["endpoint_sequence_logits"].clone(),
        "l2_endpoint_sequence_logits": r5_tensors["endpoint_sequence_logits"].clone(),
        "stable8_causal_features": r5_tensors["stable8_causal_features"].clone(),
        "event_frame_values": r5_tensors["event_frame_values"].clone(),
        "event_frame_mask": r5_tensors["event_frame_mask"].clone(),
    }
    payload: dict[str, Any] = {
        "schema_version": R6_CACHE_SCHEMA,
        "experiment_id": "V27d-r6-bird-liquid-verifier",
        "role": role,
        "direct_baseline": "V27d-r5-control",
        "source_r5_v2_cache_path": relative_project_path(root, _r5_v2_path(root, config, role)),
        "source_r5_v2_cache_sha256": sha256_file(_r5_v2_path(root, config, role)),
        "source_r5_v2_payload_sha256": r5["cache_sha256"],
        "source_l1_endpoint_path": relative_project_path(root, _l1_endpoint_path(root, config, role)),
        "source_l1_endpoint_sha256": sha256_file(_l1_endpoint_path(root, config, role)),
        "source_l1_endpoint_payload_sha256": l1_endpoint["cache_sha256"],
        "r6_manifest_sha256": manifest["manifest_sha256"],
        "parent_ids": list(r5["parent_ids"]),
        "parent_ids_sha256": r5["parent_ids_sha256"],
        "label_names": labels,
        "candidate_population_policy": "quality_eligible_L1_target_L2_bird",
        "positive_population_policy": "candidate_BirdTarget",
        "negative_population_policy": "candidate_all_non_BirdTarget",
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


def materialize_population_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    v27d_context = load_v27d_context(root, config["source"]["v27d_config_path"])
    pools = load_pools(v27d_context)
    tasks: dict[str, Any] = {}
    for role in R6_ROLES:
        r5_path = _r5_v2_path(root, config, role)
        l1_path = _l1_endpoint_path(root, config, role)
        r5 = load_r5_v2_cache(r5_path)
        l1_endpoint = load_endpoint_cache(v27d_context, path=l1_path, pools=pools, task="known_gate", role=role)
        payload = _build_payload(root=root, config=config, manifest=manifest, role=role, r5=r5, l1_endpoint=l1_endpoint)
        path = artifact_path(root, config["artifacts"]["cache_template"].format(role=role))
        write_torch_if_absent_or_identical(path, payload)
        checked = load_population_cache(path)
        tasks[role] = {
            "path": relative_project_path(root, path), "sha256": sha256_file(path), "cache_sha256": checked["cache_sha256"],
            "parent_count": len(checked["parent_ids"]), "positive_support": int(checked["tensors"]["bird_positive"].sum()),
            "negative_support": int(checked["tensors"]["route_false_positive_negative"].sum()), "candidate_support": int(checked["tensors"]["candidate_population"].sum()),
        }
    return {"status": "materialized", "experiment_id": "V27d-r6-bird-liquid-verifier", "manifest_sha256": manifest["manifest_sha256"], "tasks": tasks}


__all__ = ["cache_digest", "load_population_cache", "materialize_population_caches"]
