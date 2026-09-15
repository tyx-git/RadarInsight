"""Bird-route population caches derived from immutable V27d-r1 caches."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical

from v27.V27d.endpoint import load_endpoint_cache
from v27.V27d.features import load_feature_cache
from v27.V27d.features import stable8_causal_features
from v27.V27d.pools import load_pools
from v27.V27d.protocol import artifact_path as v27d_artifact_path

from .protocol import R5_ARMS, R5_ROLES, artifact_path, load_context


CACHE_SCHEMA = "RADAR-LN3-V27d-r5-bird-route-population-cache-v1"
MANIFEST_SCHEMA = "RADAR-LN3-V27d-r5-population-manifest-v1"


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


def _source_cache_path(context: Any, *, role: str) -> Path:
    return artifact_path(
        context.project_root,
        context.config.artifacts.feature_cache_template,
        variant="r1",
        role=role,
        task="dbb_specialist",
    )


def _source_endpoint_path(context: Any, *, role: str) -> Path:
    return artifact_path(
        context.project_root,
        context.config.artifacts.endpoint_cache_template,
        role=role,
        task="dbb_specialist",
    )


def _joint_quality(l1: Mapping[str, Any], l2: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    l1_tensors = l1["tensors"]
    l2_tensors = l2["tensors"]
    sequence = l1_tensors["quality_target_sequence"].bool() & l2_tensors["quality_target_sequence"].bool()
    final = l1_tensors["quality_target_final"].bool() & l2_tensors["quality_target_final"].bool()
    return sequence, final


def _build_payload(
    *,
    context: Any,
    role: str,
    arm: str,
    l1: Mapping[str, Any],
    l2: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    pool: Mapping[str, Any],
    r5_manifest_sha256: str,
) -> dict[str, Any]:
    if arm not in R5_ARMS or role not in R5_ROLES:
        raise ValueError("V27d-r5 arm or role is invalid")
    if l1["parent_ids"] != l2["parent_ids"] or l1["parent_ids"] != endpoint["parent_ids"]:
        raise ValueError("V27d-r5 L1/L2/endpoint parent order differs")
    labels = [str(value) for value in l2["label_names"]]
    l1_prediction = l1["tensors"]["classification_prediction"].long()
    l2_prediction = l2["tensors"]["classification_prediction"].long()
    quality_sequence, quality_final = _joint_quality(l1, l2)
    l1_target_route = l1_prediction.eq(0)
    l2_predicted_bird = l2_prediction.eq(1)
    positive = torch.tensor([label == "BirdTarget" for label in labels], dtype=torch.bool) & quality_final
    negative = (
        torch.tensor([label == "UnknownTarget" for label in labels], dtype=torch.bool)
        & l1_target_route
        & l2_predicted_bird
        & quality_final
    )
    population = positive | negative
    target = positive.to(dtype=torch.float32)
    source_tensors = l2["tensors"]
    endpoint_tensors = endpoint["tensors"]
    stable = stable8_causal_features(
        endpoint_tensors["stable8_summary"],
        endpoint_tensors["stable8_reliable"],
        endpoint_tensors["stable8_reliable_count"],
        endpoint_tensors["stable8_reliable_fraction"],
        endpoint_tensors["padding_mask"],
        max_frames_per_event=context.config.data.max_frames_per_event,
    )
    if arm == "control":
        stable = torch.zeros_like(stable)
    if stable.shape[-1] != 38:
        raise ValueError("V27d-r5 stable8 feature width differs")
    tensors = {
        "summary_features": source_tensors["summary_features"].clone(),
        "event_trajectory": source_tensors["event_trajectory"].clone(),
        "task_facts_sequence": source_tensors["task_facts_sequence"].clone(),
        "stable8_causal_features": stable,
        "padding_mask": source_tensors["padding_mask"].clone(),
        "lengths": source_tensors["lengths"].clone(),
        "quality_target_sequence": quality_sequence,
        "quality_target_final": quality_final,
        "bird_route_positive": positive,
        "bird_route_negative": negative,
        "bird_route_population": population,
        "bird_knownness_target": target,
        "l1_target_route": l1_target_route,
        "l2_predicted_bird": l2_predicted_bird,
        "endpoint_final_logits": source_tensors["endpoint_final_logits"].clone(),
        "classification_prediction": l2_prediction,
    }
    padding_mask = tensors["padding_mask"]
    for name, value in tensors.items():
        if name in {"padding_mask", "lengths"}:
            continue
        if value.ndim >= 2 and value.shape[:2] == padding_mask.shape:
            mask = padding_mask.reshape(*padding_mask.shape, *([1] * (value.ndim - 2))).expand_as(value)
            if bool(value.masked_select(mask).ne(0).any()):
                raise ValueError(f"V27d-r5 padded tensor is non-zero: {name}")
    payload: dict[str, Any] = {
        "schema_version": CACHE_SCHEMA,
        "experiment_id": "V27d-r5-bird-route",
        "arm": arm,
        "role": role,
        "source_variant": "r1",
        "r5_manifest_sha256": str(r5_manifest_sha256),
        "source_feature_cache_sha256": l2["cache_sha256"],
        "source_endpoint_cache_sha256": endpoint["cache_sha256"],
        "pool_registry_parent_ids_sha256": canonical_sha256(list(pool["parent_ids"])),
        "parent_ids": list(l2["parent_ids"]),
        "parent_ids_sha256": canonical_sha256(list(l2["parent_ids"])),
        "label_names": labels,
        "label_counts": dict(Counter(labels)),
        "positive_population_policy": "quality_eligible_BirdTarget_all_predictions",
        "negative_population_policy": "quality_eligible_UnknownTarget_L1_Target_L2_predicted_Bird",
        "unknown_is_classifier_class": False,
        "classification_logits_input_to_head": False,
        "softmax_input_to_head": False,
        "test_payload_access_count": 0,
        "tensors": tensors,
    }
    payload["cache_sha256"] = cache_digest(payload)
    return payload


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


def materialize_population_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, r1, manifest = load_context(root, config_path)
    pools = load_pools(r1)
    results: dict[str, Any] = {}
    for arm in R5_ARMS:
        for role in R5_ROLES:
            from v27.V27d.features import load_feature_cache as load_r1_feature_cache

            l1_target = v27d_artifact_path(root, r1.config.artifacts.feature_cache_template, variant="r1", role=role, task="known_gate")
            l2_target = v27d_artifact_path(root, r1.config.artifacts.feature_cache_template, variant="r1", role=role, task="dbb_specialist")
            l1 = load_r1_feature_cache(r1, l1_target)
            l2 = load_r1_feature_cache(r1, l2_target)
            endpoint_target = v27d_artifact_path(root, r1.config.artifacts.endpoint_cache_template, role=role, task="dbb_specialist")
            endpoint = load_endpoint_cache(r1, path=endpoint_target, pools=pools, task="dbb_specialist", role=role)
            payload = _build_payload(
                context=r1,
                role=role,
                arm=arm,
                l1=l1,
                l2=l2,
                endpoint=endpoint,
                pool=pools["roles"][role],
                r5_manifest_sha256=manifest["manifest_sha256"],
            )
            path = artifact_path(root, config["artifacts"]["cache_template"].format(arm=arm, role=role))
            write_torch_if_absent_or_identical(path, payload)
            checked = load_population_cache(path)
            results[f"{arm}:{role}"] = {
                "path": relative_project_path(root, path),
                "sha256": sha256_file(path),
                "cache_sha256": checked["cache_sha256"],
                "parent_count": len(checked["parent_ids"]),
                "bird_positive": int(checked["tensors"]["bird_route_positive"].sum()),
                "bird_route_unknown_negative": int(checked["tensors"]["bird_route_negative"].sum()),
                "quality_eligible": int(checked["tensors"]["quality_target_final"].sum()),
            }
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": "V27d-r5-bird-route",
        "r5_manifest_sha256": manifest["manifest_sha256"],
        "tasks": results,
        "test_payload_access_count": 0,
    }
    manifest_path = artifact_path(root, config["artifacts"]["manifest_path"].replace("manifests/V27d-r5-manifest.json", "cache/V27d-r5-population-manifest.json"))
    from .protocol import write_sealed_json

    write_sealed_json(manifest_path, payload)
    return {**payload, "manifest_path": relative_project_path(root, manifest_path), "manifest_file_sha256": sha256_file(manifest_path)}


__all__ = ["CACHE_SCHEMA", "MANIFEST_SCHEMA", "cache_digest", "load_population_cache", "materialize_population_caches"]
