"""Causal V21e course and stable8 caches for V27d-r8."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch

from radar_ln3.runtime import canonical_sha256, relative_project_path, sha256_file, write_torch_if_absent_or_identical
from v27.V27d.r6.features import load_population_cache as load_r6_population_cache

from .protocol import R8_CACHE_SCHEMA, R8_EXPERIMENT_ID, R8_ROLES, artifact_path, load_context, source_path


DISPERSION_NAMES = ("current_frame_std", "causal_event_std", "causal_adjacent_delta")
INDEX_COLUMNS = ("parent_sample_id", "sample_id", "causal_event_index")


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r8 tensor is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    return canonical_sha256(parts)


def cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r8 cache lacks tensors")
    metadata = {key: value for key, value in payload.items() if key not in {"tensors", "cache_sha256"}}
    return canonical_sha256({"metadata": metadata, "tensor_sha256": _tensor_digest(tensors)})


def causal_dispersion_features(
    event_frame_values: torch.Tensor,
    event_frame_mask: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    channel_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return current-frame std, causal event std and adjacent delta.

    Frame-valid fraction is returned only as a diagnostic mask. It is not a
    model input, so missing RD frames cannot become a proxy for knownness.
    """

    if event_frame_values.ndim != 4 or event_frame_mask.ndim != 3 or padding_mask.ndim != 2:
        raise ValueError("V27d-r8 stable8 dimensions differ")
    if event_frame_values.shape[:3] != event_frame_mask.shape or event_frame_values.shape[:2] != padding_mask.shape:
        raise ValueError("V27d-r8 stable8 shapes differ")
    if event_frame_mask.dtype != torch.bool or padding_mask.dtype != torch.bool:
        raise ValueError("V27d-r8 stable8 masks differ")
    if not 0 <= int(channel_index) < event_frame_values.shape[-1]:
        raise ValueError("V27d-r8 stable8 channel index is out of bounds")
    if not event_frame_values.is_floating_point() or not torch.isfinite(event_frame_values).all():
        raise ValueError("V27d-r8 stable8 values are invalid")
    values = event_frame_values[..., int(channel_index)]
    valid = event_frame_mask & ~padding_mask.unsqueeze(-1)
    frame_count = valid.sum(dim=-1)
    event_valid = frame_count.gt(0) & ~padding_mask
    valid_fraction = (frame_count.to(values.dtype) / float(event_frame_values.shape[2])).masked_fill(padding_mask, 0.0)
    count = frame_count.clamp_min(1).to(values.dtype)
    mean = values.masked_fill(~valid, 0.0).sum(dim=-1) / count
    current_var = (values - mean.unsqueeze(-1)).square().masked_fill(~valid, 0.0).sum(dim=-1) / count
    current_std = current_var.clamp_min(0.0).sqrt().masked_fill(~event_valid, 0.0)
    event_value = mean.masked_fill(~event_valid, 0.0)

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
        current_value = event_value[:, event]
        adjacent_delta[:, event] = torch.where(
            current_valid & previous_valid,
            (current_value - previous_value).abs(),
            torch.zeros_like(current_value),
        )
        next_count = running_count + current_valid.to(values.dtype)
        next_sum = running_sum + torch.where(current_valid, current_value, torch.zeros_like(current_value))
        next_square_sum = running_square_sum + torch.where(current_valid, current_value.square(), torch.zeros_like(current_value))
        prefix_mean = next_sum / next_count.clamp_min(1.0)
        prefix_var = (next_square_sum / next_count.clamp_min(1.0) - prefix_mean.square()).clamp_min(0.0)
        causal_std[:, event] = torch.where(current_valid, prefix_var.sqrt(), torch.zeros_like(prefix_mean))
        running_count, running_sum, running_square_sum = next_count, next_sum, next_square_sum
        previous_value = torch.where(current_valid, current_value, previous_value)
        previous_valid = previous_valid | current_valid
    features = torch.stack((current_std, causal_std, adjacent_delta), dim=-1)
    features = features.masked_fill(padding_mask.unsqueeze(-1) | ~event_valid.unsqueeze(-1), 0.0)
    if not torch.isfinite(features).all() or not torch.isfinite(valid_fraction).all():
        raise FloatingPointError("V27d-r8 stable8 features are non-finite")
    return features, valid_fraction, event_valid


def _load_course(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V27d-r8 course fields differ: {path}")
        sample_ids = np.asarray(archive["sample_ids"])
        values = np.asarray(archive["values"])
        valid = np.asarray(archive["valid"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("V27d-r8 course sample_ids differ")
    if values.shape != (len(sample_ids), 1) or valid.shape != values.shape or valid.dtype != np.bool_:
        raise ValueError("V27d-r8 course shapes differ")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("V27d-r8 course values must be finite and in [0,1]")
    if np.any(values[~valid] != 0.0):
        raise ValueError("V27d-r8 invalid course values must be zero")
    return [str(value) for value in sample_ids.tolist()], values.astype(np.float32, copy=False), valid


def _load_index(root: Path, path_value: str) -> pd.DataFrame:
    index = pd.read_csv(source_path(root, path_value, must_exist=True), low_memory=False)
    if any(column not in index.columns for column in INDEX_COLUMNS):
        raise ValueError("V27d-r8 index columns differ")
    index = index.loc[:, list(INDEX_COLUMNS)].copy()
    index["parent_sample_id"] = index["parent_sample_id"].astype(str)
    index["sample_id"] = index["sample_id"].astype(str)
    index["causal_event_index"] = index["causal_event_index"].astype(int)
    return index


def _event_ids(index: pd.DataFrame, parent_id: str) -> list[str]:
    rows = index[index["parent_sample_id"].eq(str(parent_id))].sort_values("causal_event_index", kind="stable")
    indices = rows["causal_event_index"].tolist()
    if indices != list(range(len(indices))):
        raise ValueError(f"V27d-r8 causal event order is not contiguous: {parent_id}")
    return rows["sample_id"].tolist()


def _course_sequence(
    root: Path,
    config: Mapping[str, Any],
    role: str,
    parent_ids: list[str],
    lengths: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    index = _load_index(root, config["source"]["index_path"])
    path = source_path(root, config["source"]["course_cache_template"].format(role=role), must_exist=True)
    sample_ids, values, valid = _load_course(path)
    position = {sample_id: row for row, sample_id in enumerate(sample_ids)}
    result = torch.zeros((len(parent_ids), width, 1), dtype=torch.float32)
    result_valid = torch.zeros((len(parent_ids), width), dtype=torch.bool)
    expected_ids: list[str] = []
    for row, parent_id in enumerate(parent_ids):
        event_ids = _event_ids(index, parent_id)
        length = int(lengths[row].item())
        if len(event_ids) != length:
            raise ValueError(f"V27d-r8 course/r6 length differs: {parent_id}")
        expected_ids.extend(event_ids)
        for event, event_id in enumerate(event_ids):
            if event_id not in position:
                raise ValueError(f"V27d-r8 course is missing event: {event_id}")
            course_row = position[event_id]
            result[row, event, 0] = float(values[course_row, 0])
            result_valid[row, event] = bool(valid[course_row, 0])
    if sample_ids != expected_ids:
        raise ValueError(f"V27d-r8 course event order differs for {role}")
    return result, result_valid, {
        "path": relative_project_path(root, path),
        "sha256": sha256_file(path),
        "event_count": len(sample_ids),
        "valid_event_count": int(valid.sum()),
    }


def _build_payload(root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], role: str) -> dict[str, Any]:
    r6_path = source_path(root, config["source"]["r6_cache_template"].format(role=role), must_exist=True)
    r6 = load_r6_population_cache(r6_path)
    tensors = {str(name): value.detach().cpu().clone() for name, value in r6["tensors"].items()}
    dispersion, fraction, event_valid = causal_dispersion_features(
        tensors["event_frame_values"], tensors["event_frame_mask"], tensors["padding_mask"], channel_index=int(config["model"]["dispersion_channel_index"])
    )
    course, course_valid, course_provenance = _course_sequence(root, config, role, [str(value) for value in r6["parent_ids"]], tensors["lengths"], int(tensors["padding_mask"].shape[1]))
    tensors["dispersion_features"] = dispersion
    tensors["dispersion_valid_fraction"] = fraction
    tensors["dispersion_event_valid"] = event_valid
    tensors["course_features"] = course
    tensors["course_valid"] = course_valid
    payload: dict[str, Any] = {
        "schema_version": R8_CACHE_SCHEMA,
        "experiment_id": R8_EXPERIMENT_ID,
        "role": role,
        "r8_manifest_sha256": manifest["manifest_sha256"],
        "source_r6_cache_path": relative_project_path(root, r6_path),
        "source_r6_cache_sha256": sha256_file(r6_path),
        "source_r6_payload_sha256": r6["cache_sha256"],
        "course_provenance": course_provenance,
        "parent_ids": list(r6["parent_ids"]),
        "parent_ids_sha256": r6["parent_ids_sha256"],
        "label_names": list(r6["label_names"]),
        "dispersion_channel_index": int(config["model"]["dispersion_channel_index"]),
        "dispersion_names": list(DISPERSION_NAMES),
        "dispersion_valid_fraction_is_model_input": False,
        "course_valid_is_model_input": False,
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


def _validate(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if value.get("schema_version") != R8_CACHE_SCHEMA or value.get("experiment_id") != R8_EXPERIMENT_ID:
        raise ValueError("V27d-r8 cache identity differs")
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r8 cache hash differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r8 cache tensors are missing")
    count = len(value["parent_ids"])
    width = 29
    expected = {
        "summary_features": (count, width, 672), "event_trajectory": (count, width, 355), "task_facts_sequence": (count, width, 7), "event_facts_sequence": (count, width, 19),
        "padding_mask": (count, width), "lengths": (count,), "quality_target_sequence": (count, width), "quality_target_final": (count,), "candidate_population": (count,), "bird_positive": (count,), "route_false_positive_negative": (count,), "knownness_target": (count,), "prefix_route_mask": (count, width),
        "endpoint_final_logits": (count, 3), "l1_endpoint_sequence_logits": (count, width, 2), "l2_endpoint_sequence_logits": (count, width, 3), "stable8_causal_features": (count, width, 38), "event_frame_values": (count, width, 12, 8), "event_frame_mask": (count, width, 12),
        "dispersion_features": (count, width, 3), "dispersion_valid_fraction": (count, width), "dispersion_event_valid": (count, width), "course_features": (count, width, 1), "course_valid": (count, width),
    }
    bool_fields = {"padding_mask", "quality_target_sequence", "quality_target_final", "candidate_population", "bird_positive", "route_false_positive_negative", "prefix_route_mask", "event_frame_mask", "dispersion_event_valid", "course_valid"}
    for name, shape in expected.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r8 tensor shape differs: {name}")
        if name in bool_fields and tensor.dtype != torch.bool:
            raise ValueError(f"V27d-r8 bool dtype differs: {name}")
        if name not in bool_fields and name != "lengths" and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d-r8 tensor values are invalid: {name}")
    if tensors["lengths"].dtype != torch.long:
        raise ValueError("V27d-r8 lengths dtype differs")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r8 padding differs")
    for name in ("event_trajectory", "task_facts_sequence", "event_facts_sequence", "prefix_route_mask", "dispersion_features", "dispersion_valid_fraction", "dispersion_event_valid", "course_features", "course_valid"):
        mask = padding.unsqueeze(-1) if tensors[name].ndim == 3 else padding
        if bool(tensors[name].masked_select(mask).ne(0).any()):
            raise ValueError(f"V27d-r8 padded tensor is non-zero: {name}")
    if value.get("dispersion_names") != list(DISPERSION_NAMES) or value.get("dispersion_valid_fraction_is_model_input") is not False or value.get("course_valid_is_model_input") is not False:
        raise ValueError("V27d-r8 feature input contract differs")
    if not isinstance(value.get("dispersion_channel_index"), int):
        raise ValueError("V27d-r8 dispersion channel binding is missing")
    expected_dispersion, expected_fraction, expected_valid = causal_dispersion_features(tensors["event_frame_values"], tensors["event_frame_mask"], padding, channel_index=int(value["dispersion_channel_index"]))
    if not torch.equal(tensors["dispersion_features"], expected_dispersion) or not torch.equal(tensors["dispersion_event_valid"], expected_valid) or not torch.allclose(tensors["dispersion_valid_fraction"], expected_fraction, rtol=0.0, atol=1.0e-7):
        raise ValueError("V27d-r8 dispersion recomputation differs")
    return {**value, "tensors": {str(name): tensor for name, tensor in tensors.items()}}


def load_population_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r8 cache is not an object")
    return _validate(payload)


def materialize_population_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    result: dict[str, Any] = {}
    for role in R8_ROLES:
        payload = _build_payload(root, config, manifest, role)
        path = artifact_path(root, config["artifacts"]["cache_template"].format(role=role))
        write_torch_if_absent_or_identical(path, payload)
        checked = load_population_cache(path)
        result[role] = {"path": relative_project_path(root, path), "sha256": sha256_file(path), "cache_sha256": checked["cache_sha256"], "parent_count": len(checked["parent_ids"]), "candidate_count": int(checked["tensors"]["candidate_population"].sum())}
    return {"status": "materialized", "experiment_id": R8_EXPERIMENT_ID, "manifest_sha256": manifest["manifest_sha256"], "roles": result, "formal_l3_enabled": False}


__all__ = ["DISPERSION_NAMES", "cache_digest", "causal_dispersion_features", "load_population_cache", "materialize_population_caches"]
