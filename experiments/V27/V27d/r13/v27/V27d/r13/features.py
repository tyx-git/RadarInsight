"""Causal raw envelope-modulation profile caches for V27d-r13."""

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
    R13_CACHE_SCHEMA_VERSION,
    R13_CLASS_NAMES,
    R13_EXPERIMENT_ID,
    R13_PROFILE_AXIS_HZ,
    R13_PROFILE_BINS,
    artifact_path,
    load_context,
    source_path,
)


RAW_FIELDS = {
    "sample_ids",
    "event_offsets",
    "frame_feature_reliable",
    "envelope_modulation_profiles",
}


def _array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError(f"V27d-r13 field is an object array: {name}")
    return result


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r13 cache tensor is not a tensor: {name}")
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
        raise ValueError("V27d-r13 cache lacks tensors")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"tensors", "cache_sha256"}
    }
    return canonical_sha256(
        {"metadata": metadata, "tensor_sha256": _tensor_digest(tensors)}
    )


def _load_v27b_cache(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_payload_sha256: str,
    expected_split: str,
) -> dict[str, Any]:
    verify_file_hash(path, expected_file_sha256, field=f"V27d-r13 {expected_split} V27b cache")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r13 V27b source cache is not a mapping")
    result = dict(payload)
    if result.get("experiment_id") != "V27b" or result.get("split_name") != expected_split:
        raise ValueError("V27d-r13 V27b source cache identity differs")
    if (
        result.get("cache_payload_sha256") != expected_payload_sha256
        or result.get("cache_payload_sha256") != v27b_cache_payload_sha256(result)
    ):
        raise ValueError("V27d-r13 V27b source cache payload hash differs")
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
        raise ValueError(f"V27d-r13 V27b source cache lacks fields: {missing}")
    if result["source_checkpoint_file_sha256"] != "5c416261810bdaf6f8e6e751c51612c98d1f77c8abc23b6060d7ac2898078305":
        raise ValueError("V27d-r13 V27b source is not the registered V27a Main checkpoint")

    parent_ids = [str(value) for value in result["parent_sample_ids"]]
    parent_labels = [str(value) for value in result["parent_labels"]]
    lengths = _array(result["lengths"], name="lengths").astype(np.int64)
    padding = _array(result["padding_mask"], name="padding_mask")
    labels = _array(result["labels"], name="labels").astype(np.int64)
    logits = _array(result["frozen_l2_logits"], name="frozen_l2_logits").astype(np.float32)
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V27d-r13 V27b parent ids must be unique")
    if len(parent_labels) != len(parent_ids) or any(label not in R13_CLASS_NAMES for label in parent_labels):
        raise ValueError("V27d-r13 V27b parent labels differ")
    if lengths.shape != (len(parent_ids),) or bool((lengths <= 0).any()):
        raise ValueError("V27d-r13 V27b parent lengths differ")
    if padding.ndim != 2 or padding.shape[0] != len(parent_ids) or padding.dtype != np.bool_:
        raise ValueError("V27d-r13 V27b padding differs")
    if np.any(lengths > padding.shape[1]) or not np.array_equal(
        padding, np.arange(padding.shape[1])[None, :] >= lengths[:, None]
    ):
        raise ValueError("V27d-r13 V27b padding is not a suffix")
    if labels.shape != (len(parent_ids),) or not np.array_equal(
        labels, np.asarray([R13_CLASS_NAMES.index(label) for label in parent_labels], dtype=np.int64)
    ):
        raise ValueError("V27d-r13 V27b labels differ")
    if logits.shape != (*padding.shape, len(R13_CLASS_NAMES)) or not np.isfinite(logits).all():
        raise ValueError("V27d-r13 V27b frozen logits differ")
    if np.any(logits[padding] != 0.0):
        raise ValueError("V27d-r13 V27b padded logits are non-zero")
    result.update(
        parent_sample_ids=parent_ids,
        parent_labels=parent_labels,
        lengths=lengths,
        padding_mask=padding,
        labels=labels,
        frozen_l2_logits=logits,
    )
    return result


def _load_raw_evidence(path: Path, *, expected_sha256: str) -> dict[str, np.ndarray]:
    verify_file_hash(path, expected_sha256, field=f"V27d-r13 raw evidence {path.name}")
    preflight_npz_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = RAW_FIELDS - set(archive.files)
        if missing:
            raise ValueError(f"V27d-r13 raw evidence lacks fields: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in RAW_FIELDS}

    sample_ids = result["sample_ids"]
    offsets = result["event_offsets"]
    reliable = result["frame_feature_reliable"]
    profiles = result["envelope_modulation_profiles"]
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("V27d-r13 raw sample_ids are invalid")
    if len(set(sample_ids.astype(str).tolist())) != len(sample_ids):
        raise ValueError("V27d-r13 raw sample_ids contain duplicates")
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != len(sample_ids) + 1:
        raise ValueError("V27d-r13 raw event_offsets are invalid")
    offsets = offsets.astype(np.int64, copy=False)
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(profiles) or bool((np.diff(offsets) < 0).any()):
        raise ValueError("V27d-r13 raw event_offsets are not monotonic/bounded")
    if reliable.shape != (len(profiles),) or reliable.dtype != np.bool_:
        raise ValueError("V27d-r13 raw frame reliability differs")
    if profiles.shape != (len(reliable), R13_PROFILE_BINS) or profiles.dtype not in (np.float16, np.float32):
        raise ValueError("V27d-r13 raw envelope profile shape/dtype differs")
    profiles = profiles.astype(np.float64, copy=False)
    if not np.isfinite(profiles).all() or np.any(profiles < 0.0) or np.any(profiles > 1.0):
        raise ValueError("V27d-r13 raw envelope profiles must be finite in [0,1]")
    return {
        "sample_ids": sample_ids.astype(str),
        "event_offsets": offsets,
        "frame_feature_reliable": reliable,
        "envelope_modulation_profiles": profiles,
    }


def _event_profiles(
    raw: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    offsets = raw["event_offsets"]
    profiles = raw["envelope_modulation_profiles"]
    reliable = raw["frame_feature_reliable"]
    event_values = np.zeros((len(offsets) - 1, R13_PROFILE_BINS), dtype=np.float32)
    event_valid = np.zeros((len(offsets) - 1,), dtype=np.bool_)
    event_frame_count = np.zeros((len(offsets) - 1,), dtype=np.int64)
    zero_mass_reliable_frame_count = 0
    for event, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        frame_values = profiles[int(start) : int(stop)]
        frame_reliable = reliable[int(start) : int(stop)]
        positive_mass = frame_values.sum(axis=1, dtype=np.float64) > 0.0
        selected = frame_reliable & positive_mass
        zero_mass_reliable_frame_count += int((frame_reliable & ~positive_mass).sum())
        event_frame_count[event] = int(selected.sum())
        if not bool(selected.any()):
            continue
        value = frame_values[selected].mean(axis=0, dtype=np.float64)
        total = float(value.sum(dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("V27d-r13 reliable envelope event has no positive shape mass")
        event_values[event] = (value / total).astype(np.float32)
        event_valid[event] = True
    return event_values, event_valid, event_frame_count, zero_mass_reliable_frame_count


def _pad_events(values: np.ndarray, source: Mapping[str, Any]) -> np.ndarray:
    lengths = _array(source["lengths"], name="lengths").astype(np.int64)
    padding = _array(source["padding_mask"], name="padding_mask")
    result = np.zeros((len(lengths), padding.shape[1], values.shape[-1]), dtype=np.float32)
    cursor = 0
    for row, length in enumerate(lengths.tolist()):
        result[row, : int(length)] = values[cursor : cursor + int(length)]
        cursor += int(length)
    if cursor != len(values):
        raise ValueError("V27d-r13 event profile traversal is incomplete")
    return result


def _pad_scalar(values: np.ndarray, source: Mapping[str, Any], *, dtype: np.dtype) -> np.ndarray:
    lengths = _array(source["lengths"], name="lengths").astype(np.int64)
    padding = _array(source["padding_mask"], name="padding_mask")
    result = np.zeros((len(lengths), padding.shape[1]), dtype=dtype)
    cursor = 0
    for row, length in enumerate(lengths.tolist()):
        result[row, : int(length)] = values[cursor : cursor + int(length)]
        cursor += int(length)
    if cursor != len(values):
        raise ValueError("V27d-r13 event scalar traversal is incomplete")
    return result


def _build_cache(
    *,
    split: str,
    source: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
    source_file_sha256: str,
    raw_file_sha256: str,
) -> dict[str, Any]:
    lengths = _array(source["lengths"], name="lengths").astype(np.int64)
    padding = _array(source["padding_mask"], name="padding_mask").astype(np.bool_)
    expected_event_ids = [
        str(parent)
        for parent, length in zip(source["parent_sample_ids"], lengths.tolist(), strict=True)
        for _ in range(int(length))
    ]
    raw_event_ids = [str(value).split("#point", 1)[0] for value in raw["sample_ids"].tolist()]
    if raw_event_ids != expected_event_ids:
        raise ValueError(f"V27d-r13 {split} raw envelope event order differs")
    if len(raw_event_ids) != int(lengths.sum()):
        raise ValueError(f"V27d-r13 {split} source/profile event counts differ")

    event_values, event_valid, event_frame_count, zero_mass_count = _event_profiles(raw)
    logits = _array(source["frozen_l2_logits"], name="frozen_l2_logits").astype(np.float32)
    if logits.shape != (*padding.shape, len(R13_CLASS_NAMES)) or np.any(logits[padding] != 0.0):
        raise ValueError(f"V27d-r13 {split} frozen logits shape/padding differs")
    tensors = {
        "frozen_l2_logits": torch.from_numpy(logits.copy()),
        "profile_values": torch.from_numpy(_pad_events(event_values, source)),
        "profile_valid": torch.from_numpy(_pad_scalar(event_valid, source, dtype=np.bool_)),
        "profile_frame_count": torch.from_numpy(_pad_scalar(event_frame_count, source, dtype=np.int64)),
        "padding_mask": torch.from_numpy(padding.copy()),
        "lengths": torch.from_numpy(lengths.copy()),
        "labels": torch.from_numpy(_array(source["labels"], name="labels").astype(np.int64).copy()),
    }
    payload: dict[str, Any] = {
        "schema_version": R13_CACHE_SCHEMA_VERSION,
        "experiment_id": R13_EXPERIMENT_ID,
        "split_name": str(split),
        "parent_sample_ids": [str(value) for value in source["parent_sample_ids"]],
        "parent_labels": [str(value) for value in source["parent_labels"]],
        "source_v27b_cache_file_sha256": source_file_sha256,
        "source_v27b_cache_payload_sha256": str(source["cache_payload_sha256"]),
        "source_raw_evidence_file_sha256": raw_file_sha256,
        "profile_bins": R13_PROFILE_BINS,
        "profile_axis_hz": list(R13_PROFILE_AXIS_HZ),
        "profile_frequency_support_hz": [300.0, 2000.0],
        "profile_normalization": "per-frame-profile-mean-then-event-l1-shape",
        "raw_profile_source": "envelope_modulation_profiles",
        "zero_mass_reliable_frame_count": int(zero_mass_count),
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "tensors": tensors,
    }
    payload["cache_sha256"] = cache_digest(payload)
    return payload


def _validate_cache(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if value.get("schema_version") != R13_CACHE_SCHEMA_VERSION or value.get("experiment_id") != R13_EXPERIMENT_ID:
        raise ValueError("V27d-r13 cache identity differs")
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r13 cache hash differs")
    if (
        value.get("profile_bins") != R13_PROFILE_BINS
        or value.get("profile_axis_hz") != list(R13_PROFILE_AXIS_HZ)
        or value.get("profile_frequency_support_hz") != [300.0, 2000.0]
        or value.get("raw_profile_source") != "envelope_modulation_profiles"
        or value.get("profile_normalization") != "per-frame-profile-mean-then-event-l1-shape"
    ):
        raise ValueError("V27d-r13 profile contract differs")
    if value.get("unknown_payload_access_count") != 0 or value.get("test_payload_access_count") != 0:
        raise ValueError("V27d-r13 cache scope differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r13 cache tensors are missing")
    tensors = {str(name): tensor for name, tensor in tensors.items()}
    expected_tensor_names = {
        "frozen_l2_logits",
        "profile_values",
        "profile_valid",
        "profile_frame_count",
        "padding_mask",
        "lengths",
        "labels",
    }
    if set(tensors) != expected_tensor_names:
        raise ValueError("V27d-r13 cache tensor fields differ")
    parent_ids = [str(item) for item in value.get("parent_sample_ids", ())]
    parent_labels = [str(item) for item in value.get("parent_labels", ())]
    if not parent_ids or len(parent_ids) != len(set(parent_ids)) or len(parent_labels) != len(parent_ids):
        raise ValueError("V27d-r13 cache parent metadata differs")
    if any(label not in R13_CLASS_NAMES for label in parent_labels):
        raise ValueError("V27d-r13 cache contains a non-DBB label")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool or padding.ndim != 2 or padding.shape[0] != len(parent_ids):
        raise ValueError("V27d-r13 cache padding differs")
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long or lengths.shape != (len(parent_ids),) or bool(lengths.le(0).any()):
        raise ValueError("V27d-r13 cache lengths differ")
    width = int(padding.shape[1])
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r13 cache padding is not a suffix")
    expected_shapes = {
        "frozen_l2_logits": (len(parent_ids), width, 3),
        "profile_values": (len(parent_ids), width, R13_PROFILE_BINS),
        "profile_valid": (len(parent_ids), width),
        "profile_frame_count": (len(parent_ids), width),
        "labels": (len(parent_ids),),
    }
    for name, shape in expected_shapes.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r13 cache tensor shape differs: {name}")
        if name == "profile_valid" and tensor.dtype != torch.bool:
            raise ValueError("V27d-r13 profile_valid dtype differs")
        if name in {"profile_frame_count", "labels"} and tensor.dtype != torch.long:
            raise ValueError(f"V27d-r13 integer dtype differs: {name}")
        if name not in {"profile_valid", "profile_frame_count", "labels"} and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d-r13 floating tensor is invalid: {name}")
    expected_labels = torch.tensor([R13_CLASS_NAMES.index(label) for label in parent_labels], dtype=torch.long)
    if not torch.equal(tensors["labels"], expected_labels):
        raise ValueError("V27d-r13 cache labels differ from parent labels")
    if bool(tensors["profile_values"].lt(0.0).any()) or bool(tensors["profile_values"].gt(1.0).any()):
        raise ValueError("V27d-r13 profile values are outside [0,1]")
    if bool(tensors["profile_frame_count"].lt(0).any()) or bool(tensors["profile_frame_count"].gt(12).any()):
        raise ValueError("V27d-r13 profile frame counts are out of bounds")
    for name in ("frozen_l2_logits", "profile_values", "profile_valid", "profile_frame_count"):
        mask = padding.unsqueeze(-1) if tensors[name].ndim == 3 else padding
        if bool(tensors[name].masked_select(mask).ne(0).any()):
            raise ValueError(f"V27d-r13 padded tensor is non-zero: {name}")
    if bool(tensors["profile_values"].masked_select(~tensors["profile_valid"].unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d-r13 invalid profiles must be zero")
    if not torch.equal(tensors["profile_valid"], tensors["profile_frame_count"].gt(0) & ~padding):
        raise ValueError("V27d-r13 profile validity/count semantics differ")
    valid_profiles = tensors["profile_values"][tensors["profile_valid"]]
    if valid_profiles.numel() and not torch.allclose(
        valid_profiles.sum(dim=-1),
        torch.ones(valid_profiles.shape[0], dtype=valid_profiles.dtype),
        rtol=0.0,
        atol=2.0e-6,
    ):
        raise ValueError("V27d-r13 event profiles must be L1-normalized")
    for name in (
        "source_v27b_cache_file_sha256",
        "source_v27b_cache_payload_sha256",
        "source_raw_evidence_file_sha256",
    ):
        digest = value.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"V27d-r13 cache provenance digest is malformed: {name}")
    return {**value, "tensors": tensors}


def load_profile_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r13 cache is not a mapping")
    return _validate_cache(payload)


def validate_profile_cache_binding(
    payload: Mapping[str, Any],
    *,
    root: Path,
    config: Mapping[str, Any],
    split: str,
) -> None:
    if split not in {"train", "val"}:
        raise ValueError(f"V27d-r13 cache split is unsupported: {split}")
    source = config["source"]
    role = "train" if split == "train" else "validation"
    expected = {
        "source_v27b_cache_file_sha256": source[f"expected_v27b_{role}_cache_file_sha256"],
        "source_v27b_cache_payload_sha256": source[f"expected_v27b_{role}_cache_payload_sha256"],
        "source_raw_evidence_file_sha256": source[f"expected_raw_{role}_evidence_sha256"],
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise ValueError(f"V27d-r13 cache provenance differs at {key}")
    source_cache = source_path(root, source[f"v27b_{role}_cache_path"], must_exist=True)
    raw_evidence = source_path(root, source[f"raw_{role}_evidence_path"], must_exist=True)
    if sha256_file(source_cache) != expected["source_v27b_cache_file_sha256"]:
        raise ValueError("V27d-r13 V27b source cache changed after materialization")
    if sha256_file(raw_evidence) != expected["source_raw_evidence_file_sha256"]:
        raise ValueError("V27d-r13 raw evidence changed after materialization")


def materialize_profile_caches(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    source = config["source"]
    results: dict[str, Any] = {}
    for split in ("train", "val"):
        role = "train" if split == "train" else "validation"
        source_cache_path = source_path(root, source[f"v27b_{role}_cache_path"], must_exist=True)
        raw_path = source_path(root, source[f"raw_{role}_evidence_path"], must_exist=True)
        source_file_sha256 = sha256_file(source_cache_path)
        raw_file_sha256 = sha256_file(raw_path)
        source_payload = _load_v27b_cache(
            source_cache_path,
            expected_file_sha256=source[f"expected_v27b_{role}_cache_file_sha256"],
            expected_payload_sha256=source[f"expected_v27b_{role}_cache_payload_sha256"],
            expected_split=split,
        )
        raw = _load_raw_evidence(
            raw_path,
            expected_sha256=source[f"expected_raw_{role}_evidence_sha256"],
        )
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
            "valid_frame_count": int(checked["tensors"]["profile_frame_count"].sum()),
            "zero_mass_reliable_frame_count": checked["zero_mass_reliable_frame_count"],
        }
    return {
        "status": "materialized_v27d_r13_envelope_profile_caches",
        "experiment_id": R13_EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": results,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }


__all__ = ["cache_digest", "load_profile_cache", "materialize_profile_caches", "validate_profile_cache_binding"]