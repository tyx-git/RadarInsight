"""Causal residual Doppler profile-shape cache for V27d-r9."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.runtime import (
    canonical_sha256,
    preflight_npz_archive,
    relative_project_path,
    sha256_file,
    verify_file_hash,
    write_torch_if_absent_or_identical,
)
from v27.v27b.cache import cache_payload_sha256 as v27b_cache_payload_sha256

from .protocol import (
    R9_CACHE_SCHEMA_VERSION,
    R9_CLASS_NAMES,
    R9_EXPERIMENT_ID,
    R9_PROFILE_BINS,
    artifact_path,
    load_context,
    source_path,
)


RAW_FIELDS = {
    "sample_ids",
    "event_offsets",
    "frame_feature_reliable",
    "residual_doppler_profiles",
}


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r9 cache tensor is not a tensor: {name}")
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


def cache_digest(payload: Mapping[str, Any]) -> str:
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r9 cache lacks tensors")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256"}
    }
    return canonical_sha256(
        {
            "metadata": metadata,
            "tensor_sha256": _tensor_digest(tensors),
        }
    )


def _array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError(f"V27d-r9 field is an object array: {name}")
    return result


def _load_v27b_cache(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    expected_split: str,
) -> dict[str, Any]:
    verify_file_hash(path, expected_file_sha256, field=f"V27d-r9 {expected_split} V27b cache")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r9 V27b source cache is not a mapping")
    result = dict(payload)
    if result.get("experiment_id") != "V27b" or result.get("split_name") != expected_split:
        raise ValueError("V27d-r9 V27b source cache identity differs")
    declared = result.get("cache_payload_sha256")
    if declared != expected_payload_sha256 or declared != v27b_cache_payload_sha256(result):
        raise ValueError("V27d-r9 V27b source cache payload hash differs")
    required = {
        "parent_sample_ids",
        "parent_labels",
        "labels",
        "lengths",
        "padding_mask",
        "frozen_l2_logits",
        "source_checkpoint_file_sha256",
        "source_cache_payload_sha256",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"V27d-r9 V27b source cache lacks fields: {missing}")
    if result["source_checkpoint_file_sha256"] != "5c416261810bdaf6f8e6e751c51612c98d1f77c8abc23b6060d7ac2898078305":
        raise ValueError("V27d-r9 V27b source is not the registered V27a Main checkpoint")
    return result


def _load_raw_evidence(
    path: Path,
    *,
    expected_sha256: str,
) -> dict[str, np.ndarray]:
    verify_file_hash(path, expected_sha256, field=f"V27d-r9 raw evidence {path.name}")
    preflight_npz_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        if not RAW_FIELDS.issubset(set(archive.files)):
            raise ValueError(f"V27d-r9 raw evidence lacks fields: {sorted(RAW_FIELDS - set(archive.files))}")
        result = {name: np.asarray(archive[name]) for name in RAW_FIELDS}
    sample_ids = result["sample_ids"]
    offsets = result["event_offsets"]
    reliable = result["frame_feature_reliable"]
    profiles = result["residual_doppler_profiles"]
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("V27d-r9 raw sample_ids are invalid")
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != len(sample_ids) + 1:
        raise ValueError("V27d-r9 raw event_offsets are invalid")
    offsets = offsets.astype(np.int64, copy=False)
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(reliable) or bool((np.diff(offsets) < 0).any()):
        raise ValueError("V27d-r9 raw event_offsets are not monotonic/bounded")
    if profiles.shape != (len(reliable), R9_PROFILE_BINS) or profiles.dtype not in (np.float16, np.float32):
        raise ValueError("V27d-r9 raw residual Doppler profile shape/dtype differs")
    if reliable.shape != (len(profiles),) or reliable.dtype != np.bool_:
        raise ValueError("V27d-r9 raw frame reliability differs")
    profiles = profiles.astype(np.float32, copy=False)
    if not np.isfinite(profiles).all() or np.any(profiles < 0.0) or np.any(profiles > 1.0):
        raise ValueError("V27d-r9 raw residual Doppler profiles must be finite in [0,1]")
    return {
        "sample_ids": sample_ids.astype(str),
        "event_offsets": offsets,
        "frame_feature_reliable": reliable,
        "residual_doppler_profiles": profiles,
    }


def _event_profiles(raw: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = raw["event_offsets"]
    profiles = raw["residual_doppler_profiles"]
    reliable = raw["frame_feature_reliable"]
    event_values = np.zeros((len(offsets) - 1, R9_PROFILE_BINS), dtype=np.float32)
    event_valid = np.zeros((len(offsets) - 1,), dtype=np.bool_)
    event_frame_count = np.zeros((len(offsets) - 1,), dtype=np.int64)
    for event, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        selected = reliable[int(start) : int(stop)]
        count = int(selected.sum())
        event_frame_count[event] = count
        if count <= 0:
            continue
        value = profiles[int(start) : int(stop)][selected].mean(axis=0, dtype=np.float64)
        total = float(value.sum(dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("V27d-r9 reliable event profile has no positive shape mass")
        event_values[event] = (value / total).astype(np.float32)
        event_valid[event] = True
    return event_values, event_valid, event_frame_count


def _pad_events(values: np.ndarray, source: Mapping[str, Any], *, dtype: np.dtype) -> np.ndarray:
    lengths = _array(source["lengths"], name="lengths").astype(np.int64)
    padding = _array(source["padding_mask"], name="padding_mask")
    result = np.zeros((len(lengths), padding.shape[1], values.shape[-1]), dtype=dtype)
    cursor = 0
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        result[row, :length] = values[cursor : cursor + length]
        cursor += length
    if cursor != len(values):
        raise ValueError("V27d-r9 source event traversal is incomplete")
    return result


def _pad_event_scalar(values: np.ndarray, source: Mapping[str, Any], *, dtype: np.dtype) -> np.ndarray:
    lengths = _array(source["lengths"], name="lengths").astype(np.int64)
    padding = _array(source["padding_mask"], name="padding_mask")
    result = np.zeros((len(lengths), padding.shape[1]), dtype=dtype)
    cursor = 0
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        result[row, :length] = values[cursor : cursor + length]
        cursor += length
    if cursor != len(values):
        raise ValueError("V27d-r9 scalar event traversal is incomplete")
    return result


def _build_cache(
    *,
    split: str,
    source: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
    source_file_sha256: str,
    raw_file_sha256: str,
) -> dict[str, Any]:
    expected_event_ids = [
        str(parent)
        for parent, length in zip(source["parent_sample_ids"], source["lengths"], strict=True)
        for _ in range(int(length))
    ]
    raw_event_ids = [str(value).split("#point", 1)[0] for value in raw["sample_ids"].tolist()]
    if raw_event_ids != expected_event_ids:
        raise ValueError(f"V27d-r9 {split} raw profile event order differs")

    event_values, event_valid, event_frame_count = _event_profiles(raw)
    padding = _array(source["padding_mask"], name="padding_mask").astype(np.bool_)
    lengths = _array(source["lengths"], name="lengths").astype(np.int64)
    logits = _array(source["frozen_l2_logits"], name="frozen_l2_logits").astype(np.float32)
    if logits.ndim != 3 or logits.shape[-1] != len(R9_CLASS_NAMES) or logits.shape[:2] != padding.shape:
        raise ValueError(f"V27d-r9 {split} frozen logits shape differs")
    if np.any(logits[padding] != 0.0):
        raise ValueError(f"V27d-r9 {split} padded frozen logits are non-zero")
    tensors = {
        "frozen_l2_logits": torch.from_numpy(logits.copy()),
        "profile_values": torch.from_numpy(_pad_events(event_values, source, dtype=np.float32)),
        "profile_valid": torch.from_numpy(_pad_event_scalar(event_valid, source, dtype=np.bool_)),
        "profile_frame_count": torch.from_numpy(_pad_event_scalar(event_frame_count, source, dtype=np.int64)),
        "padding_mask": torch.from_numpy(padding.copy()),
        "lengths": torch.from_numpy(lengths.copy()),
        "labels": torch.from_numpy(_array(source["labels"], name="labels").astype(np.int64).copy()),
    }
    payload: dict[str, Any] = {
        "schema_version": R9_CACHE_SCHEMA_VERSION,
        "experiment_id": R9_EXPERIMENT_ID,
        "split_name": split,
        "parent_sample_ids": [str(value) for value in source["parent_sample_ids"]],
        "parent_labels": [str(value) for value in source["parent_labels"]],
        "source_v27b_cache_file_sha256": source_file_sha256,
        "source_v27b_cache_payload_sha256": source["cache_payload_sha256"],
        "source_raw_evidence_file_sha256": raw_file_sha256,
        "profile_bins": R9_PROFILE_BINS,
        "profile_axis_mps": np.linspace(-90.0, 90.0, R9_PROFILE_BINS, dtype=np.float64).tolist(),
        "profile_normalization": "per-frame-profile-mean-then-event-l1-shape",
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "tensors": tensors,
    }
    payload["cache_sha256"] = cache_digest(payload)
    return payload


def _validate_cache(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if value.get("schema_version") != R9_CACHE_SCHEMA_VERSION or value.get("experiment_id") != R9_EXPERIMENT_ID:
        raise ValueError("V27d-r9 cache identity differs")
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r9 cache hash differs")
    if value.get("profile_bins") != R9_PROFILE_BINS or value.get("unknown_payload_access_count") != 0 or value.get("test_payload_access_count") != 0:
        raise ValueError("V27d-r9 cache scope/profile contract differs")
    expected_axis = np.linspace(-90.0, 90.0, R9_PROFILE_BINS, dtype=np.float64).tolist()
    if value.get("profile_axis_mps") != expected_axis:
        raise ValueError("V27d-r9 profile axis differs")
    if value.get("profile_normalization") != "per-frame-profile-mean-then-event-l1-shape":
        raise ValueError("V27d-r9 profile normalization contract differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r9 cache tensors are missing")
    tensors = {str(name): tensor for name, tensor in tensors.items()}
    parent_ids = [str(item) for item in value.get("parent_sample_ids", ())]
    parent_labels = [str(item) for item in value.get("parent_labels", ())]
    if not parent_ids or len(parent_ids) != len(set(parent_ids)) or len(parent_labels) != len(parent_ids):
        raise ValueError("V27d-r9 cache parent metadata differs")
    if any(label not in R9_CLASS_NAMES for label in parent_labels):
        raise ValueError("V27d-r9 cache contains a non-DBB label")
    padding = tensors.get("padding_mask")
    lengths = tensors.get("lengths")
    if not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool or padding.ndim != 2 or padding.shape[0] != len(parent_ids):
        raise ValueError("V27d-r9 cache padding differs")
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long or lengths.shape != (len(parent_ids),):
        raise ValueError("V27d-r9 cache lengths differ")
    width = int(padding.shape[1])
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r9 cache padding is not a suffix")
    expected_shapes = {
        "frozen_l2_logits": (len(parent_ids), width, 3),
        "profile_values": (len(parent_ids), width, R9_PROFILE_BINS),
        "profile_valid": (len(parent_ids), width),
        "profile_frame_count": (len(parent_ids), width),
        "labels": (len(parent_ids),),
    }
    for name, shape in expected_shapes.items():
        tensor = tensors.get(name)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r9 cache tensor shape differs: {name}")
        if name == "profile_valid" and tensor.dtype != torch.bool:
            raise ValueError("V27d-r9 profile_valid dtype differs")
        if name in {"profile_frame_count", "labels"} and tensor.dtype != torch.long:
            raise ValueError(f"V27d-r9 integer dtype differs: {name}")
        if name not in {"profile_valid", "profile_frame_count", "labels"} and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d-r9 floating tensor is invalid: {name}")
    if torch.any(tensors["labels"] != torch.tensor([R9_CLASS_NAMES.index(label) for label in parent_labels], dtype=torch.long)):
        raise ValueError("V27d-r9 cache labels differ from parent labels")
    if torch.any(tensors["profile_values"].lt(0.0)) or torch.any(tensors["profile_values"].gt(1.0)):
        raise ValueError("V27d-r9 profile values are outside [0,1]")
    if torch.any(tensors["profile_frame_count"].lt(0)) or torch.any(tensors["profile_frame_count"].gt(12)):
        raise ValueError("V27d-r9 profile frame counts are out of bounds")
    for name in ("frozen_l2_logits", "profile_values", "profile_valid", "profile_frame_count"):
        mask = padding.unsqueeze(-1) if tensors[name].ndim == 3 else padding
        if bool(tensors[name].masked_select(mask).ne(0).any()):
            raise ValueError(f"V27d-r9 padded tensor is non-zero: {name}")
    if bool(tensors["profile_values"].masked_select(~tensors["profile_valid"].unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d-r9 invalid profile values must be zero")
    if not torch.equal(tensors["profile_valid"], tensors["profile_frame_count"].gt(0) & ~padding):
        raise ValueError("V27d-r9 profile validity/count semantics differ")
    valid_profiles = tensors["profile_values"][tensors["profile_valid"]]
    if valid_profiles.numel() and not torch.allclose(
        valid_profiles.sum(dim=-1),
        torch.ones(valid_profiles.shape[0], dtype=valid_profiles.dtype),
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise ValueError("V27d-r9 event profiles must be L1-normalized")
    for name in (
        "source_v27b_cache_file_sha256",
        "source_v27b_cache_payload_sha256",
        "source_raw_evidence_file_sha256",
    ):
        digest = value.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"V27d-r9 cache provenance digest is malformed: {name}")
    return {**value, "tensors": tensors}


def load_profile_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r9 cache is not a mapping")
    return _validate_cache(payload)


def validate_profile_cache_binding(
    payload: Mapping[str, Any],
    *,
    root: Path,
    config: Mapping[str, Any],
    split: str,
) -> None:
    """Verify a derived cache against the current registered source files."""

    role = "train" if str(split).strip().lower() == "train" else "validation"
    if role not in {"train", "validation"}:
        raise ValueError(f"V27d-r9 cache split is unsupported: {split}")
    source = config["source"]
    expected = {
        "source_v27b_cache_file_sha256": source[f"expected_v27b_{role}_cache_file_sha256"],
        "source_v27b_cache_payload_sha256": source[f"expected_v27b_{role}_cache_payload_sha256"],
        "source_raw_evidence_file_sha256": source[f"expected_raw_{'train' if role == 'train' else 'validation'}_evidence_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V27d-r9 cache provenance differs at {key}")
    cache_key = role
    raw_key = "train" if role == "train" else "validation"
    source_cache = source_path(root, source[f"v27b_{cache_key}_cache_path"], must_exist=True)
    raw_evidence = source_path(root, source[f"raw_{raw_key}_evidence_path"], must_exist=True)
    if sha256_file(source_cache) != expected["source_v27b_cache_file_sha256"]:
        raise ValueError("V27d-r9 V27b source cache changed after materialization")
    if sha256_file(raw_evidence) != expected["source_raw_evidence_file_sha256"]:
        raise ValueError("V27d-r9 raw evidence changed after materialization")


def materialize_profile_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    results: dict[str, Any] = {}
    source = config["source"]
    for split in ("train", "val"):
        cache_key = "train" if split == "train" else "validation"
        raw_key = "train" if split == "train" else "validation"
        source_cache_path = source_path(root, source[f"v27b_{cache_key}_cache_path"], must_exist=True)
        raw_path = source_path(root, source[f"raw_{raw_key}_evidence_path"], must_exist=True)
        source_file_sha256 = sha256_file(source_cache_path)
        raw_file_sha256 = sha256_file(raw_path)
        source_payload = _load_v27b_cache(
            source_cache_path,
            expected_file_sha256=source[f"expected_v27b_{cache_key}_cache_file_sha256"],
            expected_payload_sha256=source[f"expected_v27b_{cache_key}_cache_payload_sha256"],
            expected_split=split,
        )
        raw = _load_raw_evidence(raw_path, expected_sha256=source[f"expected_raw_{raw_key}_evidence_sha256"])
        payload = _build_cache(
            split=split,
            source=source_payload,
            raw=raw,
            source_file_sha256=source_file_sha256,
            raw_file_sha256=raw_file_sha256,
        )
        target = artifact_path(root, config["artifacts"]["cache_template"].format(split=split))
        write_torch_if_absent_or_identical(target, payload)
        checked = load_profile_cache(target)
        results[split] = {
            "path": relative_project_path(root, target),
            "sha256": sha256_file(target),
            "cache_sha256": checked["cache_sha256"],
            "parent_count": len(checked["parent_sample_ids"]),
            "valid_event_count": int(checked["tensors"]["profile_valid"].sum()),
        }
    return {
        "status": "materialized_v27d_r9_profile_shape_caches",
        "experiment_id": R9_EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": results,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }


__all__ = ["cache_digest", "load_profile_cache", "materialize_profile_caches", "validate_profile_cache_binding"]