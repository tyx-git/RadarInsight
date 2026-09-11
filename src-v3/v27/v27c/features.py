"""Materialize V27c's numeric-only causal motion sidecars.

The frozen V27a cache supplies the classification endpoint and parent layout.
This module reconstructs only the V9 causal motion values from the canonical
index.  It never copies a validity mask into a trainable input: the mask is
used exactly once to force invalid numeric values to zero.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.data import CQ08PointSequenceDataset
from radar_ln3.features import TR_MOTION_SCHEMA, resolve_causal_tr_feature_contract
from radar_ln3.runtime import canonical_sha256, sha256_file, write_torch_if_absent_or_identical

from .protocol import (
    V27C_FEATURE_CACHE_SCHEMA_VERSION,
    V27C_FEATURE_MANIFEST_SCHEMA_VERSION,
    V27C_FROZEN_MAIN_CONFUSION,
    V27C_L2_CLASSES,
    V27C_SPLITS,
    V27CContext,
    v27c_artifact_path,
    write_sealed_json_if_identical,
)


_TENSOR_FIELDS = ("frozen_endpoint_logits", "motion_values", "padding_mask", "lengths")


def _sequence_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _tensor_digest(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27c sidecar tensor is not a tensor: {name}")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def feature_cache_digest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "cache_sha256"}
    tensors = body.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27c motion cache lacks tensors")
    metadata = {key: value for key, value in body.items() if key != "tensors"}
    return canonical_sha256({"metadata": metadata, "tensor_sha256": _tensor_digest(dict(tensors))})


def _reference_path(context: V27CContext, split: str) -> Path:
    if split == "train":
        return context.project_root / context.config.base.v27a_train_reference_cache_path
    if split == "val":
        return context.project_root / context.config.base.v27a_validation_reference_cache_path
    raise ValueError("V27c motion sidecar needs train or val")


def _reference_payload(context: V27CContext, split: str) -> dict[str, Any]:
    from v27.v27a.cache import load_cache_payload

    return load_cache_payload(_reference_path(context, split))


def _dataset(context: V27CContext, split: str) -> CQ08PointSequenceDataset:
    split_name = (
        context.config.data.train_split_name if split == "train"
        else context.config.data.validation_split_name if split == "val"
        else None
    )
    if split_name is None:
        raise ValueError("V27c dataset split is invalid")
    return CQ08PointSequenceDataset(
        context.project_root,
        index_path=context.config.data.index_path,
        split_name=split_name,
        max_sequence_length=context.config.data.max_sequence_length,
        max_rd_frames_per_point=context.config.data.max_rd_frames_per_point,
        rd_amplitude_center=context.config.data.rd_amplitude_center,
        rd_amplitude_scale=context.config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=None,
        tensor_cache_size=1,
        expected_schema_version=context.config.data.index_schema_version,
        tr_motion_schema=TR_MOTION_SCHEMA,
        allowed_labels=V27C_L2_CLASSES,
        forbidden_labels={"ClutterTarget", "UnknownTarget"},
    )


def _cache_path(context: V27CContext, split: str) -> Path:
    return v27c_artifact_path(
        context.project_root, context.config.artifacts.feature_cache_template, split=split
    )


def _expected_reference_payload_hash(context: V27CContext, split: str) -> str:
    if split == "train":
        return context.config.base.expected_v27a_train_reference_cache_payload_sha256
    if split == "val":
        return context.config.base.expected_v27a_validation_reference_cache_payload_sha256
    raise ValueError("V27c reference-cache payload hash needs train or val")


def _expected_reference_file_sha256(context: V27CContext, split: str) -> str:
    if split == "train":
        return context.config.base.expected_v27a_train_reference_cache_sha256
    if split == "val":
        return context.config.base.expected_v27a_validation_reference_cache_sha256
    raise ValueError("V27c reference-cache file hash needs train or val")


def _expected_event_ids_order_sha256(context: V27CContext, split: str) -> str:
    if split == "train":
        return context.config.base.expected_v27a_train_event_ids_order_sha256
    if split == "val":
        return context.config.base.expected_v27a_validation_event_ids_order_sha256
    raise ValueError("V27c event-order hash needs train or val")


def _frozen_main_logits(context: V27CContext, reference: Mapping[str, Any]) -> torch.Tensor:
    """Replay V27a Main once; its logits are an immutable output anchor only."""

    from v27.v27a.model import V27AEvidenceResidual
    from v27.v27a.protocol import load_v27a_config

    config = load_v27a_config(
        context.project_root, context.project_root / context.config.base.v27a_runtime_config_path
    )
    checkpoint = torch.load(
        context.project_root / context.config.base.v27a_main_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V27c V27a Main checkpoint is not an object")
    model = V27AEvidenceResidual(config, arm="main")
    model.load_state_dict(dict(checkpoint["model_state"]), strict=True)
    model.eval()
    baseline = torch.from_numpy(np.asarray(reference["event_baseline_logits"], dtype=np.float32))
    result = torch.zeros_like(baseline)
    batch_size = 64
    with torch.inference_mode():
        for start in range(0, baseline.shape[0], batch_size):
            stop = min(baseline.shape[0], start + batch_size)
            inputs = {
                "baseline_logits": baseline[start:stop],
                "summary_tokens": torch.from_numpy(reference["event_summary_tokens"][start:stop]),
                "summary_valid": torch.from_numpy(reference["event_summary_valid"][start:stop]),
                "frame_values": torch.from_numpy(reference["event_frame_values"][start:stop]),
                "frame_mask": torch.from_numpy(reference["event_frame_mask"][start:stop]),
                "padding_mask": torch.from_numpy(reference["padding_mask"][start:stop]),
            }
            output = model(**inputs)
            if not torch.equal(output["baseline_logits"], inputs["baseline_logits"]):
                raise AssertionError("V27c V27a Main replay altered its frozen base logits")
            result[start:stop] = output["sequence_logits"]
    if not torch.isfinite(result).all():
        raise FloatingPointError("V27c frozen endpoint replay produced non-finite logits")
    return result


def replay_frozen_main_logits(context: V27CContext, *, split: str) -> torch.Tensor:
    """Return the exact V27a Main logit anchor for one canonical split.

    This is intentionally separate from sidecar loading: a smoke check needs
    to prove the stored anchor still equals a fresh frozen-endpoint replay,
    while normal training should not repeat that comparatively expensive work.
    """

    if split not in V27C_SPLITS:
        raise ValueError("V27c frozen endpoint replay needs train or val")
    return _frozen_main_logits(context, _reference_payload(context, split))


def _validate_feature_cache(payload: Mapping[str, Any], context: V27CContext, *, split: str) -> dict[str, Any]:
    if split not in V27C_SPLITS:
        raise ValueError("V27c feature cache split is invalid")
    contract = resolve_causal_tr_feature_contract(context.config.data.motion_schema)
    expected_reference_path = str(
        _reference_path(context, split).relative_to(context.project_root)
    ).replace("\\", "/")
    motion_clip = payload.get("motion_clip_absolute")
    if (
        payload.get("schema_version") != V27C_FEATURE_CACHE_SCHEMA_VERSION
        or payload.get("experiment_id") != context.config.experiment_id
        or payload.get("split") != split
        or payload.get("manifest_sha256") != context.manifest_sha256
        or payload.get("motion_schema") != context.config.data.motion_schema
        or payload.get("motion_names") != list(contract.value_names)
        or isinstance(motion_clip, bool)
        or not isinstance(motion_clip, (int, float))
        or not math.isclose(
            float(motion_clip), contract.clip_absolute, rel_tol=0.0, abs_tol=1.0e-9
        )
        or payload.get("motion_mask_mode") != "numerical_gating_only_not_model_input"
        or payload.get("reference_cache_path") != expected_reference_path
        or payload.get("reference_cache_sha256")
        != _expected_reference_file_sha256(context, split)
        or payload.get("reference_cache_payload_sha256") != _expected_reference_payload_hash(context, split)
        or payload.get("frozen_v27a_main_checkpoint_sha256")
        != context.config.base.expected_v27a_main_checkpoint_sha256
        or payload.get("frozen_v27a_main_state_sha256")
        != context.config.base.expected_v27a_main_state_sha256
        or payload.get("event_ids_order_sha256")
        != _expected_event_ids_order_sha256(context, split)
    ):
        raise ValueError("V27c motion cache identity differs")
    parent_ids = [str(value) for value in payload.get("parent_ids") or ()]
    labels = [str(value) for value in payload.get("label_names") or ()]
    if (
        not parent_ids
        or len(parent_ids) != len(labels)
        or len(set(parent_ids)) != len(parent_ids)
        or any(label not in V27C_L2_CLASSES for label in labels)
        or payload.get("parent_ids_sha256") != canonical_sha256(parent_ids)
    ):
        raise ValueError("V27c motion cache parent metadata differs")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != set(_TENSOR_FIELDS):
        raise ValueError("V27c motion cache tensor fields differ")
    count, width, motion_dim = (
        len(parent_ids), context.config.data.max_sequence_length, context.config.data.motion_dim
    )
    frozen = tensors["frozen_endpoint_logits"]
    values = tensors["motion_values"]
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if (
        not isinstance(frozen, torch.Tensor)
        or frozen.shape != (count, width, len(V27C_L2_CLASSES))
        or frozen.dtype != torch.float32
        or not torch.isfinite(frozen).all()
        or not isinstance(values, torch.Tensor)
        or values.shape != (count, width, motion_dim)
        or values.dtype != torch.float32
        or not torch.isfinite(values).all()
        or not isinstance(padding, torch.Tensor)
        or padding.shape != (count, width)
        or padding.dtype != torch.bool
        or not isinstance(lengths, torch.Tensor)
        or lengths.shape != (count,)
        or lengths.dtype != torch.long
    ):
        raise ValueError("V27c motion cache tensor shapes or dtypes differ")
    expected_padding = torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(padding.cpu(), expected_padding):
        raise ValueError("V27c motion cache padding differs from lengths")
    if bool((lengths <= 0).any()) or bool((lengths > width).any()):
        raise ValueError("V27c motion cache length is out of bounds")
    if bool(frozen.masked_select(padding.unsqueeze(-1).expand_as(frozen)).ne(0).any()):
        raise ValueError("V27c padded frozen endpoint logits must be zero")
    if bool(values.masked_select(padding.unsqueeze(-1).expand_as(values)).ne(0).any()):
        raise ValueError("V27c padded motion values must be zero")
    if bool(values[:, 0].ne(0).any()):
        raise ValueError("V27c first event motion must be zero after every parent reset")
    if bool((values.abs() > context.config.data.motion_clip_absolute + 1.0e-6).any()):
        raise ValueError("V27c motion values exceed the frozen clip")
    reference = _reference_payload(context, split)
    if (
        parent_ids != [str(value) for value in reference["parent_sample_ids"]]
        or labels != [str(value) for value in reference["parent_labels"]]
        or not torch.equal(lengths.cpu(), torch.from_numpy(reference["lengths"]).long())
        or not torch.equal(padding.cpu(), torch.from_numpy(reference["padding_mask"]).bool())
    ):
        raise ValueError("V27c motion cache does not align with frozen V27a cache")
    if split == "val":
        rows = torch.arange(count)
        final_logits = frozen[rows, lengths - 1]
        prediction = final_logits.argmax(dim=-1)
        truth = torch.tensor(
            [V27C_L2_CLASSES.index(label) for label in labels], dtype=torch.long
        )
        matrix = torch.zeros((len(V27C_L2_CLASSES), len(V27C_L2_CLASSES)), dtype=torch.long)
        matrix.index_put_((truth, prediction), torch.ones_like(truth), accumulate=True)
        if matrix.tolist() != [list(row) for row in V27C_FROZEN_MAIN_CONFUSION]:
            raise ValueError("V27c sidecar frozen endpoint does not replay V27a Main")
    if payload.get("cache_sha256") != feature_cache_digest(payload):
        raise ValueError("V27c motion cache hash differs")
    return dict(payload)


def _event_ids_by_parent(dataset: CQ08PointSequenceDataset) -> dict[str, list[str]]:
    """Return public-index event IDs in the Dataset's causal parent layout."""

    groups: dict[str, list[str]] = {}
    for parent_id, group in dataset.frame.groupby("parent_sample_id", sort=False):
        event_indices = group["causal_event_index"].tolist()
        if event_indices != list(range(len(group))):
            raise ValueError("V27c motion Dataset event order differs")
        groups[str(parent_id)] = group["sample_id"].astype(str).tolist()
    if set(groups) != set(dataset.parent_ids) or any(not values for values in groups.values()):
        raise ValueError("V27c motion Dataset event-id support differs")
    return groups


def materialize_feature_caches(context: V27CContext) -> dict[str, Any]:
    """Build independent, cache-aligned numerical motion sidecars."""

    contract = resolve_causal_tr_feature_contract(context.config.data.motion_schema)
    result: dict[str, Any] = {}
    for split in V27C_SPLITS:
        reference = _reference_payload(context, split)
        dataset = _dataset(context, split)
        expected_parents = [str(value) for value in reference["parent_sample_ids"]]
        positions = {parent: index for index, parent in enumerate(dataset.parent_ids)}
        if set(positions) != set(expected_parents):
            raise ValueError("V27c motion Dataset parent population differs from frozen V27a cache")
        event_ids_by_parent = _event_ids_by_parent(dataset)
        width = context.config.data.max_sequence_length
        frozen_logits = _frozen_main_logits(context, reference)
        motion = torch.zeros((len(expected_parents), width, contract.value_dim), dtype=torch.float32)
        padded = torch.from_numpy(reference["padding_mask"]).bool()
        lengths = torch.from_numpy(reference["lengths"]).long()
        labels: list[str] = []
        event_ids: list[str] = []
        final_nonzero = 0
        active_nonzero = 0
        active_count = 0
        for row, parent_id in enumerate(expected_parents):
            item = dataset[positions[parent_id]]
            length = int(item.event_count)
            if (
                length != int(lengths[row])
                or item.label_name != str(reference["parent_labels"][row])
                or item.tr_motion_features.shape != (length, contract.value_dim)
                or item.tr_motion_valid_mask.shape != item.tr_motion_features.shape
            ):
                raise ValueError("V27c motion Dataset event alignment differs")
            raw = item.tr_motion_features.float()
            valid = item.tr_motion_valid_mask.bool()
            if not torch.isfinite(raw).all() or bool((raw.abs() > contract.clip_absolute + 1.0e-6).any()):
                raise ValueError("V27c raw motion values violate their contract")
            if bool(raw.masked_select(~valid).ne(0).any()):
                raise ValueError("V27c invalid raw motion values must be zero")
            gated = raw * valid.to(dtype=raw.dtype)
            motion[row, :length] = gated
            parent_event_ids = event_ids_by_parent[parent_id][:length]
            if len(parent_event_ids) != length:
                raise ValueError("V27c motion Dataset event id layout differs")
            event_ids.extend(parent_event_ids)
            labels.append(item.label_name)
            nonzero = gated.abs().amax(dim=-1).gt(0.0)
            final_nonzero += int(nonzero[-1].item())
            active_nonzero += int(nonzero.sum().item())
            active_count += length
        payload: dict[str, Any] = {
            "schema_version": V27C_FEATURE_CACHE_SCHEMA_VERSION,
            "experiment_id": context.config.experiment_id,
            "split": split,
            "manifest_sha256": context.manifest_sha256,
            "reference_cache_path": str(_reference_path(context, split).relative_to(context.project_root)).replace("\\", "/"),
            "reference_cache_sha256": sha256_file(_reference_path(context, split)),
            "reference_cache_payload_sha256": reference["cache_payload_sha256"],
            "frozen_v27a_main_checkpoint_sha256": context.config.base.expected_v27a_main_checkpoint_sha256,
            "frozen_v27a_main_state_sha256": context.config.base.expected_v27a_main_state_sha256,
            "motion_schema": contract.schema,
            "motion_names": list(contract.value_names),
            "motion_clip_absolute": contract.clip_absolute,
            "motion_mask_mode": "numerical_gating_only_not_model_input",
            "event_ids_order_sha256": _sequence_digest(event_ids),
            "parent_ids": expected_parents,
            "parent_ids_sha256": canonical_sha256(expected_parents),
            "label_names": labels,
            "tensors": {
                "frozen_endpoint_logits": frozen_logits,
                "motion_values": motion,
                "padding_mask": padded,
                "lengths": lengths,
            },
        }
        payload["cache_sha256"] = feature_cache_digest(payload)
        checked = _validate_feature_cache(payload, context, split=split)
        target = _cache_path(context, split)
        write_torch_if_absent_or_identical(target, checked)
        result[split] = {
            "path": str(target.relative_to(context.project_root)).replace("\\", "/"),
            "cache_sha256": checked["cache_sha256"],
            "parent_count": len(expected_parents),
            "active_motion_nonzero_fraction": float(active_nonzero / active_count),
            "final_motion_nonzero_fraction": float(final_nonzero / len(expected_parents)),
        }
    manifest = {
        "schema_version": V27C_FEATURE_MANIFEST_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "manifest_sha256": context.manifest_sha256,
        "motion_schema": contract.as_dict(),
        "caches": result,
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
    }
    path = v27c_artifact_path(context.project_root, context.config.artifacts.feature_manifest_path)
    write_sealed_json_if_identical(path, manifest)
    return {"feature_manifest_path": str(path), "feature_manifest_sha256": sha256_file(path), "caches": result}


def load_feature_cache(context: V27CContext, *, split: str) -> dict[str, Any]:
    target = _cache_path(context, split)
    payload = torch.load(target, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27c motion cache is not an object")
    return _validate_feature_cache(payload, context, split=split)


def load_feature_manifest(context: V27CContext) -> dict[str, Any]:
    path = v27c_artifact_path(context.project_root, context.config.artifacts.feature_manifest_path)
    from .protocol import read_sealed_json

    payload = read_sealed_json(path, schema=V27C_FEATURE_MANIFEST_SCHEMA_VERSION)
    if payload.get("manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27c feature manifest experiment binding differs")
    for split in V27C_SPLITS:
        cache = load_feature_cache(context, split=split)
        declared = payload.get("caches", {}).get(split, {})
        if declared.get("cache_sha256") != cache["cache_sha256"]:
            raise ValueError("V27c feature manifest cache binding differs")
    return payload


__all__ = [
    "feature_cache_digest", "load_feature_cache", "load_feature_manifest",
    "materialize_feature_caches", "replay_frozen_main_logits",
]
