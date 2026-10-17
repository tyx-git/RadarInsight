"""Profile-shape feature materialization for the independent Bird-route r1."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.runtime import atomic_torch_save, canonical_sha256, relative_project_path, sha256_file

from .evidence import _load_cache, _source_path
from .protocol import FEATURE_SCHEMA, R1Context
from v28.representations import load_representation


def _downsample(values: np.ndarray, bins: int) -> np.ndarray:
    chunks = np.array_split(np.asarray(values, dtype=np.float32), int(bins))
    return np.asarray([float(chunk.mean()) if chunk.size else 0.0 for chunk in chunks], dtype=np.float32)


def _entropy(values: np.ndarray) -> float:
    power = np.asarray(values, dtype=np.float64)
    power = np.clip(power, 0.0, None)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    probability = power / total
    probability = probability[probability > 0.0]
    if probability.size <= 1:
        return 0.0
    return float(-np.sum(probability * np.log(probability)) / np.log(probability.size))


def _shape_vector(residual: np.ndarray, envelope: np.ndarray, reliable: np.ndarray) -> np.ndarray:
    residual = np.asarray(residual, dtype=np.float32)
    envelope = np.asarray(envelope, dtype=np.float32)
    reliable = np.asarray(reliable, dtype=bool)
    if residual.ndim != 2 or residual.shape[1] != 181 or envelope.ndim != 2 or envelope.shape[1] != 64 or reliable.shape != (residual.shape[0],):
        raise ValueError("V28 Bird-route r1 profile evidence shapes differ")
    if bool(reliable.any()):
        residual_mean = residual[reliable].mean(axis=0)
        envelope_mean = envelope[reliable].mean(axis=0)
    else:
        residual_mean = np.zeros((181,), dtype=np.float32)
        envelope_mean = np.zeros((64,), dtype=np.float32)
    residual_profile = _downsample(residual_mean, 16)
    envelope_profile = _downsample(envelope_mean, 8)
    residual_axis = np.linspace(-1.0, 1.0, 181, dtype=np.float32)
    envelope_axis = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    residual_mass = float(np.clip(residual_mean, 0.0, None).sum())
    envelope_mass = float(np.clip(envelope_mean, 0.0, None).sum())
    residual_power = np.clip(residual_mean, 0.0, None)
    envelope_power = np.clip(envelope_mean, 0.0, None)
    residual_center = float((residual_power * residual_axis).sum() / residual_mass) if residual_mass > 0 else 0.0
    envelope_center = float((envelope_power * envelope_axis).sum() / envelope_mass) if envelope_mass > 0 else 0.0
    residual_symmetry = float(abs(residual_power[residual_axis >= 0].sum() - residual_power[residual_axis < 0].sum()) / max(residual_mass, 1.0e-6))
    residual_concentration = float(np.sort(residual_power / max(residual_mass, 1.0e-6))[-5:].sum())
    envelope_concentration = float(np.sort(envelope_power / max(envelope_mass, 1.0e-6))[-3:].sum())
    stats = np.asarray([
        _entropy(residual_mean),
        _entropy(envelope_mean),
        residual_center,
        envelope_center,
        residual_symmetry,
        residual_concentration,
        envelope_concentration,
        float(reliable.mean()) if reliable.size else 0.0,
    ], dtype=np.float32)
    result = np.concatenate((residual_profile, envelope_profile, stats), axis=0).astype(np.float32)
    if result.shape != (32,) or not np.isfinite(result).all():
        raise ValueError("V28 Bird-route r1 profile feature is invalid")
    return result


def _role_entry(context: R1Context, role: str) -> Mapping[str, Any]:
    import json
    pool_path = context.project_root / context.v28_registry["pool_registry"]["path"]
    payload = json.loads(pool_path.read_text(encoding="utf-8"))
    return payload["roles"]["l2_dbb_specialist"][role]


def materialize_role(context: R1Context, *, role: str, split: str) -> dict[str, Any]:
    cache = _load_cache(_source_path(context, split))
    positions = {str(value): index for index, value in enumerate(cache["sample_ids"].astype(str).tolist())}
    entry = _role_entry(context, role)
    parent_ids = [str(value) for value in entry["parent_ids"]]
    sample_ids = [str(value) for value in entry["sample_ids"]]
    index_by_parent: dict[str, list[int]] = {}
    for sample_id in sample_ids:
        if sample_id not in positions:
            raise ValueError(f"Bird-route r1 evidence lacks role event: {role}/{sample_id}")
        parent = sample_id.split("#", 1)[0]
        index_by_parent.setdefault(parent, []).append(positions[sample_id])
    max_length = int(context.config["data"]["max_sequence_length"])
    features = np.zeros((len(parent_ids), max_length, 32), dtype=np.float32)
    padding = np.ones((len(parent_ids), max_length), dtype=bool)
    lengths = np.zeros((len(parent_ids),), dtype=np.int64)
    labels: list[str] = []
    ordered_parent_ids: list[str] = []
    for row, parent_id in enumerate(parent_ids):
        indices = index_by_parent.get(parent_id, [])
        if not indices or len(indices) > max_length:
            raise ValueError(f"Bird-route r1 parent event count is invalid: {role}/{parent_id}")
        for time, event_index in enumerate(indices):
            start, stop = int(cache["event_offsets"][event_index]), int(cache["event_offsets"][event_index + 1])
            features[row, time] = _shape_vector(
                cache["residual_doppler_profiles"][start:stop].astype(np.float32),
                cache["envelope_modulation_profiles"][start:stop].astype(np.float32),
                cache["frame_feature_reliable"][start:stop],
            )
        lengths[row] = len(indices)
        padding[row, : len(indices)] = False
        ordered_parent_ids.append(parent_id)
        labels.append("")
    # Labels are sourced from the V28 role entry's parent order through the
    # index, never inferred from feature values.
    import pandas as pd
    index = pd.read_csv(context.project_root / context.config["data"]["index_path"], low_memory=False)
    first = index.drop_duplicates("parent_sample_id").set_index("parent_sample_id")
    labels = [str(first.loc[parent_id, "label_name"]) for parent_id in ordered_parent_ids]
    v28_role = load_representation(context.v28_context, task="l2_dbb_specialist", role=role)
    if [str(value) for value in v28_role["parent_ids"]] != ordered_parent_ids:
        raise ValueError(f"Bird-route r1 profile/V28 representation parent order differs: {role}")
    temperature = float(context.v28_registry["calibration"]["tasks"]["l2_dbb_specialist"]["temperature"]["temperature"])
    lengths_v28 = v28_role["tensors"]["lengths"].long()
    rows_v28 = torch.arange(lengths_v28.shape[0])
    logits = v28_role["tensors"]["sequence_logits"][rows_v28, lengths_v28 - 1].numpy() / temperature
    route_names = np.asarray([("DroneTarget", "BirdTarget", "BalloonTarget")[int(index)] for index in logits.argmax(axis=1)], dtype=np.str_)
    bird_route = route_names == "BirdTarget"
    payload: dict[str, Any] = {
        "schema_version": FEATURE_SCHEMA,
        "experiment_id": "V28-bird-route-r1-profile-shape",
        "role": role,
        "split": split,
        "system_manifest_sha256": context.v28_context.manifest_sha256,
        "v28_registry_sha256": context.v28_registry["registry_sha256"],
        "parent_ids": ordered_parent_ids,
        "label_names": labels,
        "frozen_l2_route_names": route_names.tolist(),
        "frozen_bird_route_mask": bird_route.tolist(),
        "tensors": {
            "profile_features": torch.from_numpy(features),
            "padding_mask": torch.from_numpy(padding),
            "lengths": torch.from_numpy(lengths),
        },
        "test_access_count": 0,
    }
    digest_body = {key: value for key, value in payload.items() if key != "tensors"}
    payload["cache_sha256"] = canonical_sha256({"metadata": digest_body, "tensor_sha256": {key: value.numpy().tobytes().hex() for key, value in payload["tensors"].items()}})
    path = context.project_root / str(context.config["data"]["role_profile_template"].format(role=role))
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(path, payload)
    return {"role": role, "path": relative_project_path(context.project_root, path), "sha256": sha256_file(path), "cache_sha256": payload["cache_sha256"], "parent_count": len(parent_ids), "label_counts": dict(Counter(labels))}


def load_role(context: R1Context, role: str) -> dict[str, Any]:
    path = context.project_root / str(context.config["data"]["role_profile_template"].format(role=role))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != FEATURE_SCHEMA or payload.get("system_manifest_sha256") != context.v28_context.manifest_sha256:
        raise ValueError(f"Bird-route r1 feature identity differs: {role}")
    tensors = payload["tensors"]
    digest = canonical_sha256({"metadata": {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}, "tensor_sha256": {key: value.numpy().tobytes().hex() for key, value in tensors.items()}})
    if digest != payload.get("cache_sha256"):
        raise ValueError(f"Bird-route r1 feature hash differs: {role}")
    return payload




__all__ = ["load_role", "materialize_role"]