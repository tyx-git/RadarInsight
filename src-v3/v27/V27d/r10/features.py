"""Compact causal Doppler profile-shape descriptors for V27d-r10."""

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
    R10_CACHE_SCHEMA_VERSION,
    R10_CLASS_NAMES,
    R10_DESCRIPTOR_DIM,
    R10_DESCRIPTOR_NAMES,
    R10_EXPERIMENT_ID,
    R10_ROOT,
    R10_SEEDS,
    artifact_path,
    load_context,
    source_path,
)


PROFILE_BINS = 181
PROFILE_AXIS_MPS = np.linspace(-90.0, 90.0, PROFILE_BINS, dtype=np.float64)
RAW_FIELDS = {
    "sample_ids",
    "event_offsets",
    "frame_feature_reliable",
    "residual_doppler_profiles",
}
_V27A_MAIN_CHECKPOINT_SHA256 = (
    "5c416261810bdaf6f8e6e751c51612c98d1f77c8abc23b6060d7ac2898078305"
)


def _array(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError(f"V27d-r10 field is an object array: {name}")
    return result


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    parts: list[str] = []
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27d-r10 cache tensor is not a tensor: {name}")
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
        raise ValueError("V27d-r10 cache lacks tensors")
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
    verify_file_hash(path, expected_file_sha256, field=f"V27d-r10 {expected_split} V27b cache")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r10 V27b source cache is not a mapping")
    result = dict(payload)
    if result.get("experiment_id") != "V27b" or result.get("split_name") != expected_split:
        raise ValueError("V27d-r10 V27b source cache identity differs")
    declared = result.get("cache_payload_sha256")
    if declared != expected_payload_sha256 or declared != v27b_cache_payload_sha256(result):
        raise ValueError("V27d-r10 V27b source cache payload hash differs")
    required = {
        "parent_sample_ids",
        "parent_labels",
        "labels",
        "lengths",
        "padding_mask",
        "frozen_l2_logits",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"V27d-r10 V27b source cache lacks fields: {missing}")
    if result.get("source_checkpoint_file_sha256") != _V27A_MAIN_CHECKPOINT_SHA256:
        raise ValueError("V27d-r10 V27b source is not the registered V27a Main checkpoint")
    parent_ids = [str(value) for value in result["parent_sample_ids"]]
    parent_labels = [str(value) for value in result["parent_labels"]]
    lengths = _array(result["lengths"], name="lengths").astype(np.int64)
    padding = _array(result["padding_mask"], name="padding_mask")
    labels = _array(result["labels"], name="labels").astype(np.int64)
    logits = _array(result["frozen_l2_logits"], name="frozen_l2_logits").astype(np.float32)
    if not parent_ids or len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V27d-r10 V27b parent ids must be unique")
    if len(parent_labels) != len(parent_ids) or any(label not in R10_CLASS_NAMES for label in parent_labels):
        raise ValueError("V27d-r10 V27b parent labels differ")
    if lengths.shape != (len(parent_ids),) or np.any(lengths <= 0):
        raise ValueError("V27d-r10 V27b parent lengths differ")
    if padding.ndim != 2 or padding.shape[0] != len(parent_ids) or padding.dtype != np.bool_:
        raise ValueError("V27d-r10 V27b padding differs")
    if np.any(lengths > padding.shape[1]) or not np.array_equal(
        padding, np.arange(padding.shape[1])[None, :] >= lengths[:, None]
    ):
        raise ValueError("V27d-r10 V27b padding is not a suffix")
    if labels.shape != (len(parent_ids),) or not np.array_equal(
        labels, np.asarray([R10_CLASS_NAMES.index(label) for label in parent_labels], dtype=np.int64)
    ):
        raise ValueError("V27d-r10 V27b labels differ")
    if logits.shape != (*padding.shape, len(R10_CLASS_NAMES)) or not np.isfinite(logits).all():
        raise ValueError("V27d-r10 V27b frozen logits differ")
    if np.any(logits[padding] != 0.0):
        raise ValueError("V27d-r10 V27b padded logits are non-zero")
    return {
        **result,
        "parent_sample_ids": parent_ids,
        "parent_labels": parent_labels,
        "lengths": lengths,
        "padding_mask": padding,
        "labels": labels,
        "frozen_l2_logits": logits,
    }


def _load_raw_evidence(path: Path, *, expected_sha256: str) -> dict[str, np.ndarray]:
    verify_file_hash(path, expected_sha256, field=f"V27d-r10 raw evidence {path.name}")
    preflight_npz_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = RAW_FIELDS - set(archive.files)
        if missing:
            raise ValueError(f"V27d-r10 raw evidence lacks fields: {sorted(missing)}")
        result = {name: np.asarray(archive[name]) for name in RAW_FIELDS}
    sample_ids = result["sample_ids"]
    offsets = result["event_offsets"]
    reliable = result["frame_feature_reliable"]
    profiles = result["residual_doppler_profiles"]
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in {"U", "S"}:
        raise ValueError("V27d-r10 raw sample_ids are invalid")
    if len(set(sample_ids.astype(str).tolist())) != len(sample_ids):
        raise ValueError("V27d-r10 raw sample_ids contain duplicates")
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != len(sample_ids) + 1:
        raise ValueError("V27d-r10 raw event_offsets are invalid")
    offsets = offsets.astype(np.int64, copy=False)
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(reliable) or bool((np.diff(offsets) < 0).any()):
        raise ValueError("V27d-r10 raw event_offsets are not monotonic/bounded")
    if reliable.shape != (len(profiles),) or reliable.dtype != np.bool_:
        raise ValueError("V27d-r10 raw frame reliability differs")
    if profiles.shape != (len(reliable), PROFILE_BINS) or profiles.dtype not in (np.float16, np.float32):
        raise ValueError("V27d-r10 raw Doppler profile shape/dtype differs")
    profiles = profiles.astype(np.float64, copy=False)
    if not np.isfinite(profiles).all() or np.any(profiles < 0.0) or np.any(profiles > 1.0):
        raise ValueError("V27d-r10 raw Doppler profiles must be finite in [0,1]")
    if bool((reliable & (profiles.sum(axis=1) <= 0.0)).any()):
        raise ValueError("V27d-r10 reliable raw profiles must have positive mass")
    return {
        "sample_ids": sample_ids.astype(str),
        "event_offsets": offsets,
        "frame_feature_reliable": reliable,
        "residual_doppler_profiles": profiles,
    }


def _event_descriptors(raw: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = raw["event_offsets"]
    profiles = raw["residual_doppler_profiles"]
    reliable = raw["frame_feature_reliable"]
    values = np.zeros((len(offsets) - 1, R10_DESCRIPTOR_DIM), dtype=np.float32)
    valid = np.zeros((len(offsets) - 1,), dtype=np.bool_)
    frame_counts = np.zeros((len(offsets) - 1,), dtype=np.int64)
    for event, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        mask = reliable[int(start) : int(stop)]
        selected = profiles[int(start) : int(stop)][mask]
        frame_counts[event] = int(selected.shape[0])
        if selected.shape[0] == 0:
            continue
        per_frame = selected / selected.sum(axis=1, keepdims=True)
        shape = per_frame.mean(axis=0)
        shape /= shape.sum()
        centroid = float(np.sum(shape * PROFILE_AXIS_MPS, dtype=np.float64))
        variance = float(np.sum(shape * np.square(PROFILE_AXIS_MPS - centroid), dtype=np.float64))
        spread = float(np.sqrt(max(0.0, variance)))
        positive = float(shape[PROFILE_AXIS_MPS > 0.0].sum(dtype=np.float64))
        negative = float(shape[PROFILE_AXIS_MPS < 0.0].sum(dtype=np.float64))
        balance = (positive - negative) / max(positive + negative, np.finfo(np.float64).tiny)
        nonzero = shape[shape > 0.0]
        entropy = float(-np.sum(nonzero * np.log(nonzero), dtype=np.float64) / np.log(PROFILE_BINS))
        top4 = float(np.partition(shape, -4)[-4:].sum(dtype=np.float64))
        scaled_rms = float(np.sqrt(PROFILE_BINS) * np.sqrt(np.mean(np.square(shape), dtype=np.float64)))
        if per_frame.shape[0] > 1:
            norms = np.linalg.norm(per_frame, axis=1)
            cosine = np.sum(per_frame[1:] * per_frame[:-1], axis=1) / (
                norms[1:] * norms[:-1] + np.finfo(np.float64).tiny
            )
            cosine_mean = float(np.clip(np.mean(cosine), 0.0, 1.0))
            peak_drift = float(np.mean(np.abs(np.diff(np.argmax(per_frame, axis=1))))) / 180.0
        else:
            cosine_mean = 1.0
            peak_drift = 0.0
        values[event] = np.asarray(
            [
                centroid / 90.0,
                spread / 90.0,
                entropy,
                balance,
                top4,
                scaled_rms,
                cosine_mean,
                peak_drift,
            ],
            dtype=np.float32,
        )
        valid[event] = True
    if not np.isfinite(values).all():
        raise FloatingPointError("V27d-r10 Doppler descriptors are non-finite")
    return values, valid, frame_counts


def _pad_events(values: np.ndarray, source: Mapping[str, Any], *, dtype: np.dtype) -> np.ndarray:
    lengths = np.asarray(source["lengths"], dtype=np.int64)
    padding = np.asarray(source["padding_mask"], dtype=np.bool_)
    result = np.zeros((len(lengths), padding.shape[1], values.shape[-1]), dtype=dtype)
    cursor = 0
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        result[row, :length] = values[cursor : cursor + length]
        cursor += length
    if cursor != len(values):
        raise ValueError("V27d-r10 event descriptor traversal is incomplete")
    return result


def _pad_scalar(values: np.ndarray, source: Mapping[str, Any], *, dtype: np.dtype) -> np.ndarray:
    lengths = np.asarray(source["lengths"], dtype=np.int64)
    padding = np.asarray(source["padding_mask"], dtype=np.bool_)
    result = np.zeros((len(lengths), padding.shape[1]), dtype=dtype)
    cursor = 0
    for row, length in enumerate(lengths.tolist()):
        length = int(length)
        result[row, :length] = values[cursor : cursor + length]
        cursor += length
    if cursor != len(values):
        raise ValueError("V27d-r10 scalar descriptor traversal is incomplete")
    return result


def _build_cache(
    *,
    split: str,
    source: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
    source_file_sha256: str,
    raw_file_sha256: str,
) -> dict[str, Any]:
    lengths = np.asarray(source["lengths"], dtype=np.int64)
    padding = np.asarray(source["padding_mask"], dtype=np.bool_)
    expected_event_ids = [
        str(parent)
        for parent, length in zip(source["parent_sample_ids"], lengths.tolist(), strict=True)
        for _ in range(int(length))
    ]
    raw_event_ids = [str(value).split("#point", 1)[0] for value in raw["sample_ids"].tolist()]
    if raw_event_ids != expected_event_ids:
        raise ValueError(f"V27d-r10 {split} raw/profile parent order differs")
    if len(raw["sample_ids"]) != int(lengths.sum()):
        raise ValueError(f"V27d-r10 {split} source/profile event counts differ")
    descriptors, valid, frame_counts = _event_descriptors(raw)
    logits = np.asarray(source["frozen_l2_logits"], dtype=np.float32)
    tensors = {
        "frozen_l2_logits": torch.from_numpy(logits.copy()),
        "descriptor_values": torch.from_numpy(_pad_events(descriptors, source, dtype=np.float32)),
        "descriptor_valid": torch.from_numpy(_pad_scalar(valid, source, dtype=np.bool_)),
        "descriptor_frame_count": torch.from_numpy(_pad_scalar(frame_counts, source, dtype=np.int64)),
        "padding_mask": torch.from_numpy(padding.copy()),
        "lengths": torch.from_numpy(lengths.copy()),
        "labels": torch.from_numpy(np.asarray(source["labels"], dtype=np.int64).copy()),
    }
    payload: dict[str, Any] = {
        "schema_version": R10_CACHE_SCHEMA_VERSION,
        "experiment_id": R10_EXPERIMENT_ID,
        "split_name": str(split),
        "parent_sample_ids": [str(value) for value in source["parent_sample_ids"]],
        "parent_labels": [str(value) for value in source["parent_labels"]],
        "source_v27b_cache_file_sha256": source_file_sha256,
        "source_v27b_cache_payload_sha256": str(source["cache_payload_sha256"]),
        "source_raw_evidence_file_sha256": raw_file_sha256,
        "descriptor_names": list(R10_DESCRIPTOR_NAMES),
        "descriptor_dim": R10_DESCRIPTOR_DIM,
        "profile_bins": PROFILE_BINS,
        "profile_axis_mps": PROFILE_AXIS_MPS.tolist(),
        "descriptor_normalization": "per-frame-l1-profile-then-event-mean-shape-statistics",
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "tensors": tensors,
    }
    payload["cache_sha256"] = cache_digest(payload)
    return payload


def _validate_cache(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(payload)
    if value.get("schema_version") != R10_CACHE_SCHEMA_VERSION or value.get("experiment_id") != R10_EXPERIMENT_ID:
        raise ValueError("V27d-r10 cache identity differs")
    if value.get("cache_sha256") != cache_digest(value):
        raise ValueError("V27d-r10 cache hash differs")
    if value.get("descriptor_names") != list(R10_DESCRIPTOR_NAMES) or value.get("descriptor_dim") != R10_DESCRIPTOR_DIM or value.get("profile_bins") != PROFILE_BINS:
        raise ValueError("V27d-r10 descriptor contract differs")
    if value.get("profile_axis_mps") != PROFILE_AXIS_MPS.tolist() or value.get("descriptor_normalization") != "per-frame-l1-profile-then-event-mean-shape-statistics":
        raise ValueError("V27d-r10 profile descriptor normalization differs")
    if value.get("unknown_payload_access_count") != 0 or value.get("test_payload_access_count") != 0:
        raise ValueError("V27d-r10 cache scope differs")
    tensors = value.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27d-r10 cache tensors are missing")
    tensors = {str(name): tensor for name, tensor in tensors.items()}
    expected_tensor_names = {
        "frozen_l2_logits",
        "descriptor_values",
        "descriptor_valid",
        "descriptor_frame_count",
        "padding_mask",
        "lengths",
        "labels",
    }
    if set(tensors) != expected_tensor_names:
        raise ValueError("V27d-r10 cache tensor fields differ")
    parent_ids = [str(value) for value in value.get("parent_sample_ids", ())]
    parent_labels = [str(value) for value in value.get("parent_labels", ())]
    if not parent_ids or len(parent_ids) != len(set(parent_ids)) or len(parent_labels) != len(parent_ids):
        raise ValueError("V27d-r10 cache parent metadata differs")
    if any(label not in R10_CLASS_NAMES for label in parent_labels):
        raise ValueError("V27d-r10 cache contains a non-DBB label")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    if not isinstance(padding, torch.Tensor) or padding.dtype != torch.bool or padding.ndim != 2 or padding.shape[0] != len(parent_ids):
        raise ValueError("V27d-r10 cache padding differs")
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.long or lengths.shape != (len(parent_ids),) or bool(lengths.le(0).any()):
        raise ValueError("V27d-r10 cache lengths differ")
    width = int(padding.shape[1])
    if not torch.equal(padding, torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)):
        raise ValueError("V27d-r10 cache padding must be a suffix")
    expected_shapes = {
        "frozen_l2_logits": (len(parent_ids), width, 3),
        "descriptor_values": (len(parent_ids), width, R10_DESCRIPTOR_DIM),
        "descriptor_valid": (len(parent_ids), width),
        "descriptor_frame_count": (len(parent_ids), width),
        "labels": (len(parent_ids),),
    }
    for name, shape in expected_shapes.items():
        tensor = tensors[name]
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape:
            raise ValueError(f"V27d-r10 cache tensor shape differs: {name}")
        if name == "descriptor_valid" and tensor.dtype != torch.bool:
            raise ValueError("V27d-r10 descriptor_valid dtype differs")
        if name in {"descriptor_frame_count", "labels"} and tensor.dtype != torch.long:
            raise ValueError(f"V27d-r10 integer dtype differs: {name}")
        if name not in {"descriptor_valid", "descriptor_frame_count", "labels"} and (not tensor.is_floating_point() or not torch.isfinite(tensor).all()):
            raise ValueError(f"V27d-r10 floating tensor is invalid: {name}")
    expected_labels = torch.tensor([R10_CLASS_NAMES.index(label) for label in parent_labels], dtype=torch.long)
    if not torch.equal(tensors["labels"], expected_labels):
        raise ValueError("V27d-r10 cache labels differ from parent labels")
    descriptors = tensors["descriptor_values"]
    if bool(descriptors[..., (0, 3)].abs().gt(1.0 + 1.0e-6).any()):
        raise ValueError("V27d-r10 signed descriptors exceed [-1,1]")
    if bool(descriptors[..., (1, 2, 4, 5, 6, 7)].lt(-1.0e-6).any()) or bool(descriptors[..., (1, 2, 4, 5, 6, 7)].gt(1.0 + 1.0e-6).any()):
        raise ValueError("V27d-r10 bounded descriptors exceed [0,1]")
    if bool(tensors["descriptor_frame_count"].lt(0).any()) or bool(tensors["descriptor_frame_count"].gt(12).any()):
        raise ValueError("V27d-r10 descriptor frame counts are out of bounds")
    for name in ("frozen_l2_logits", "descriptor_values", "descriptor_valid", "descriptor_frame_count"):
        mask = padding.unsqueeze(-1) if tensors[name].ndim == 3 else padding
        if bool(tensors[name].masked_select(mask).ne(0).any()):
            raise ValueError(f"V27d-r10 padded tensor is non-zero: {name}")
    if bool(descriptors.masked_select(~tensors["descriptor_valid"].unsqueeze(-1)).ne(0).any()):
        raise ValueError("V27d-r10 invalid descriptors must be zero")
    if not torch.equal(tensors["descriptor_valid"], tensors["descriptor_frame_count"].gt(0) & ~padding):
        raise ValueError("V27d-r10 descriptor validity/count semantics differ")
    for name in ("source_v27b_cache_file_sha256", "source_v27b_cache_payload_sha256", "source_raw_evidence_file_sha256"):
        digest = value.get(name)
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            raise ValueError(f"V27d-r10 cache provenance digest is malformed: {name}")
    return {**value, "tensors": tensors}


def load_descriptor_cache(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27d-r10 cache is not a mapping")
    return _validate_cache(payload)


def validate_cache_binding(payload: Mapping[str, Any], *, root: Path, config: Mapping[str, Any], split: str) -> None:
    normalized = "train" if str(split).strip().lower() == "train" else "validation"
    if normalized not in {"train", "validation"}:
        raise ValueError(f"V27d-r10 cache split is unsupported: {split}")
    source = config["source"]
    expected = {
        "source_v27b_cache_file_sha256": source[f"expected_v27b_{normalized}_cache_file_sha256"],
        "source_v27b_cache_payload_sha256": source[f"expected_v27b_{normalized}_cache_payload_sha256"],
        "source_raw_evidence_file_sha256": source[f"expected_raw_{'train' if normalized == 'train' else 'validation'}_evidence_sha256"],
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"V27d-r10 cache provenance differs at {key}")
    source_cache = source_path(root, source[f"v27b_{normalized}_cache_path"], must_exist=True)
    raw = source_path(root, source[f"raw_{'train' if normalized == 'train' else 'validation'}_evidence_path"], must_exist=True)
    if sha256_file(source_cache) != expected["source_v27b_cache_file_sha256"] or sha256_file(raw) != expected["source_raw_evidence_file_sha256"]:
        raise ValueError("V27d-r10 registered source changed after cache materialization")


def materialize_descriptor_caches(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
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
        raw_payload = _load_raw_evidence(raw_path, expected_sha256=source[f"expected_raw_{role}_evidence_sha256"])
        payload = _build_cache(
            split=split,
            source=source_payload,
            raw=raw_payload,
            source_file_sha256=source_file_sha256,
            raw_file_sha256=raw_file_sha256,
        )
        target = artifact_path(root, config["artifacts"]["cache_template"].format(split=split))
        write_torch_if_absent_or_identical(target, payload)
        checked = load_descriptor_cache(target)
        results[split] = {
            "path": relative_project_path(root, target),
            "sha256": sha256_file(target),
            "cache_sha256": checked["cache_sha256"],
            "parent_count": len(checked["parent_sample_ids"]),
            "valid_event_count": int(checked["tensors"]["descriptor_valid"].sum()),
        }
    return {
        "status": "materialized_v27d_r10_descriptor_caches",
        "experiment_id": R10_EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "splits": results,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }


__all__ = [
    "PROFILE_AXIS_MPS",
    "PROFILE_BINS",
    "R10_DESCRIPTOR_NAMES",
    "cache_digest",
    "load_descriptor_cache",
    "materialize_descriptor_caches",
    "validate_cache_binding",
]
