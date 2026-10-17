"""Materialized V27a frozen-logit evidence cache and loader."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from radar_ln3.runtime import resolve_project_path, write_torch_if_absent_or_identical

from .evidence import EvidenceSplit, load_normalization, normalize_stable8
from .protocol import (
    V27A_CACHE_SCHEMA_VERSION,
    V27A_L2_CLASS_NAMES,
    V27AConfig,
    V27AContext,
)


_SHA256_HEX = frozenset("0123456789abcdef")


def _digest_cache_chunk(digest: Any, value: bytes | memoryview) -> None:
    """Feed one length-delimited cache-hash component."""

    digest.update(len(value).to_bytes(8, byteorder="big", signed=False))
    digest.update(value)


def cache_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash every semantic cache field independently of Torch serialization.

    A Torch archive's byte layout is not stable across equivalent saves.  This
    digest instead preserves mapping keys, scalar types, array dtype/shape and
    exact C-order bytes, so cache loading fails closed on partial corruption or
    a changed payload under the same provenance metadata.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("V27a cache payload must be a mapping to hash")
    body = {key: value for key, value in payload.items() if key != "cache_payload_sha256"}
    digest = hashlib.sha256()

    def visit(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            visit(value.detach().cpu().numpy())
            return
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            if array.dtype.hasobject:
                raise TypeError("V27a cache hash does not permit object arrays")
            _digest_cache_chunk(digest, b"ndarray")
            _digest_cache_chunk(digest, array.dtype.str.encode("ascii"))
            _digest_cache_chunk(digest, repr(tuple(array.shape)).encode("ascii"))
            _digest_cache_chunk(digest, memoryview(array).cast("B"))
            return
        if isinstance(value, np.generic):
            visit(value.item())
            return
        if value is None:
            _digest_cache_chunk(digest, b"none")
            return
        if isinstance(value, bool):
            _digest_cache_chunk(digest, b"bool")
            _digest_cache_chunk(digest, b"1" if value else b"0")
            return
        if isinstance(value, int):
            _digest_cache_chunk(digest, b"int")
            _digest_cache_chunk(digest, str(value).encode("ascii"))
            return
        if isinstance(value, float):
            if not np.isfinite(value):
                raise ValueError("V27a cache hash does not permit non-finite floats")
            _digest_cache_chunk(digest, b"float64")
            _digest_cache_chunk(digest, np.float64(value).tobytes())
            return
        if isinstance(value, str):
            _digest_cache_chunk(digest, b"str")
            _digest_cache_chunk(digest, value.encode("utf-8"))
            return
        if isinstance(value, (list, tuple)):
            _digest_cache_chunk(digest, b"list" if isinstance(value, list) else b"tuple")
            _digest_cache_chunk(
                digest, len(value).to_bytes(8, byteorder="big", signed=False)
            )
            for item in value:
                visit(item)
            return
        if isinstance(value, Mapping):
            if any(not isinstance(key, str) for key in value):
                raise TypeError("V27a cache hash requires string mapping keys")
            _digest_cache_chunk(digest, b"mapping")
            _digest_cache_chunk(
                digest, len(value).to_bytes(8, byteorder="big", signed=False)
            )
            for key in sorted(value):
                visit(key)
                visit(value[key])
            return
        raise TypeError(
            f"V27a cache hash does not support {type(value).__name__} values"
        )

    visit(body)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _SHA256_HEX for character in value.lower())
    )


def _parent_order(split: EvidenceSplit) -> list[str]:
    result: list[str] = []
    for raw_value in split.parent_ids.tolist():
        value = str(raw_value)
        if not value:
            raise ValueError("V27a evidence parent id must be non-empty")
        if not result or result[-1] != value:
            result.append(value)
    if len(result) != len(set(result)):
        raise ValueError("V27a evidence events for a parent are not contiguous")
    return result


def _parent_slices(split: EvidenceSplit) -> list[tuple[str, int, int]]:
    parents = _parent_order(split)
    result: list[tuple[str, int, int]] = []
    for parent in parents:
        indices = np.flatnonzero(split.parent_ids == parent)
        if not len(indices):
            raise ValueError(f"V27a parent has no events: {parent}")
        start, stop = int(indices[0]), int(indices[-1]) + 1
        if split.parent_ids[start:stop].tolist() != [parent] * (stop - start):
            raise ValueError(f"V27a parent events are not contiguous: {parent}")
        result.append((parent, start, stop))
    return result


def _event_summary_tokens(
    split: EvidenceSplit,
    normalized_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tokens = np.zeros((split.event_count, normalized_frames.shape[-1]), dtype=np.float32)
    valid = split.recomputed_summary_valid.copy()
    if valid.shape != (split.event_count,):
        raise ValueError("V27a event summary validity has an invalid shape")
    for event_index, (start, stop) in enumerate(
        zip(split.event_offsets[:-1], split.event_offsets[1:])
    ):
        reliable = split.frame_feature_reliable[int(start) : int(stop)]
        if bool(reliable.any()):
            # The mechanism-control token is the same event median used by
            # the frozen V18 stable8 summary contract.  A mean here would
            # silently make the event-summary arm a second feature variant.
            tokens[event_index] = np.median(
                normalized_frames[int(start) : int(stop)][reliable], axis=0
            )
        elif valid[event_index]:
            raise ValueError("V27a summary validity is inconsistent with reliable frames")
    tokens[~valid] = 0.0
    return tokens, valid


def _pad_event_frames(
    split: EvidenceSplit,
    starts: list[tuple[str, int, int]],
    *,
    normalized_frames: np.ndarray,
    sequence_width: int,
    frame_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack reliable frame rows into a parent/event tensor without sentinels.

    The flat evidence cache is retained for auditability.  The model-facing
    representation is compacted per event so invalid rows (which are known to
    contain historical non-zero values in the source cache) can never become
    a learned input.  Original order is preserved among reliable rows.
    """

    if normalized_frames.shape != split.stable8_values.shape:
        raise ValueError("V27a normalized frame shape differs while packing events")
    if normalized_frames.dtype != np.float32 or not np.isfinite(normalized_frames).all():
        raise ValueError("V27a normalized frame values must be finite float32")
    values = np.zeros(
        (len(starts), sequence_width, frame_width, split.stable8_values.shape[-1]),
        dtype=np.float32,
    )
    mask = np.zeros((len(starts), sequence_width, frame_width), dtype=np.bool_)
    for parent_row, (_, event_start, event_stop) in enumerate(starts):
        event_count = event_stop - event_start
        for local_event in range(event_count):
            event_index = event_start + local_event
            start = int(split.event_offsets[event_index])
            stop = int(split.event_offsets[event_index + 1])
            reliable = split.frame_feature_reliable[start:stop]
            selected = normalized_frames[start:stop][reliable]
            if selected.shape[0] > frame_width:
                raise ValueError("V27a reliable frame count exceeds the configured frame width")
            if selected.shape[0]:
                values[parent_row, local_event, : selected.shape[0]] = selected
                mask[parent_row, local_event, : selected.shape[0]] = True
    return values, mask


def _pad_parent_array(
    values: np.ndarray,
    starts: list[tuple[str, int, int]],
    *,
    width: int,
    dtype: np.dtype,
) -> np.ndarray:
    result = np.zeros((len(starts), width, values.shape[-1]), dtype=dtype)
    for row, (_, start, stop) in enumerate(starts):
        length = stop - start
        if length > width:
            raise ValueError("V27a parent exceeds configured sequence length")
        result[row, :length] = values[start:stop]
    return result


def _pad_parent_mask(
    values: np.ndarray,
    starts: list[tuple[str, int, int]],
    *,
    width: int,
) -> np.ndarray:
    result = np.ones((len(starts), width), dtype=np.bool_)
    for row, (_, start, stop) in enumerate(starts):
        length = stop - start
        if length > width:
            raise ValueError("V27a parent exceeds configured sequence length")
        result[row, :length] = values[start:stop]
    return result


def _pad_parent_scalar(
    values: np.ndarray,
    starts: list[tuple[str, int, int]],
    *,
    width: int,
    dtype: np.dtype,
) -> np.ndarray:
    result = np.zeros((len(starts), width), dtype=dtype)
    for row, (_, start, stop) in enumerate(starts):
        length = stop - start
        if length > width:
            raise ValueError("V27a parent exceeds configured sequence length")
        result[row, :length] = values[start:stop]
    return result


def build_cache_payload(
    split: EvidenceSplit,
    *,
    normalized_frames: np.ndarray,
    baseline_event_logits: np.ndarray,
    config: V27AConfig,
    split_name: str,
    base_checkpoint_sha256: str,
    base_classifier_state_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    if baseline_event_logits.shape != (split.event_count, len(V27A_L2_CLASS_NAMES)):
        raise ValueError("V27a baseline event logits shape differs")
    if not np.isfinite(baseline_event_logits).all():
        raise ValueError("V27a baseline logits must be finite")
    if normalized_frames.shape != split.stable8_values.shape:
        raise ValueError("V27a normalized frame shape differs")
    if normalized_frames.dtype != np.float32 or not np.isfinite(normalized_frames).all():
        raise ValueError("V27a normalized frame values must be finite float32")
    summary_tokens, summary_valid = _event_summary_tokens(split, normalized_frames)
    starts = _parent_slices(split)
    width = config.data.max_sequence_length
    # Event-level arrays are duplicated only in the compact frozen cache.  The
    # raw frame table remains represented by offsets and frame arrays below.
    parent_ids = [item[0] for item in starts]
    parent_labels = []
    for _, start, stop in starts:
        labels = {str(value) for value in split.labels[start:stop].tolist()}
        if len(labels) != 1 or next(iter(labels)) not in V27A_L2_CLASS_NAMES:
            raise ValueError("V27a parent has inconsistent or unsupported labels")
        parent_labels.append(next(iter(labels)))
    label_ids = np.asarray(
        [V27A_L2_CLASS_NAMES.index(value) for value in parent_labels], dtype=np.int64
    )
    parent_event_lengths = np.asarray(
        [stop - start for _, start, stop in starts], dtype=np.int64
    )
    sanitized_frames = normalized_frames.astype(np.float32, copy=True)
    sanitized_frames[~split.frame_feature_reliable] = 0.0
    frame_values_padded, frame_mask_padded = _pad_event_frames(
        split,
        starts,
        normalized_frames=sanitized_frames,
        sequence_width=width,
        frame_width=config.data.max_frames_per_event,
    )
    frame_values_padded[~frame_mask_padded] = 0.0
    event_frame_counts = frame_mask_padded.sum(axis=-1).astype(np.int64)
    event_rd_present = _pad_parent_scalar(
        split.rd_present.astype(np.bool_),
        starts,
        width=width,
        dtype=np.dtype(np.bool_),
    )
    event_reliable_fraction = np.zeros((len(starts), width), dtype=np.float32)
    for row, (_, start, stop) in enumerate(starts):
        for event_index in range(stop - start):
            count = int(event_frame_counts[row, event_index])
            total = int(split.event_offsets[start + event_index + 1]) - int(
                split.event_offsets[start + event_index]
            )
            event_reliable_fraction[row, event_index] = (
                float(count) / float(total) if total > 0 else 0.0
            )
    payload = {
        "schema_version": V27A_CACHE_SCHEMA_VERSION,
        "experiment_id": "V27a",
        "split_name": str(split_name),
        "parent_sample_ids": parent_ids,
        "parent_labels": parent_labels,
        "labels": label_ids,
        "lengths": parent_event_lengths,
        "padding_mask": _pad_parent_mask(
            np.zeros((split.event_count,), dtype=np.bool_), starts, width=width
        ),
        "event_baseline_logits": _pad_parent_array(
            baseline_event_logits,
            starts,
            width=width,
            dtype=np.dtype(np.float32),
        ),
        "event_summary_tokens": _pad_parent_array(
            summary_tokens,
            starts,
            width=width,
            dtype=np.dtype(np.float32),
        ),
        "event_summary_valid": _pad_parent_scalar(
            summary_valid,
            starts,
            width=width,
            dtype=np.dtype(np.bool_),
        ),
        "event_reliable_frame_count": _pad_parent_scalar(
            np.asarray(
                [
                    int(split.frame_feature_reliable[int(a) : int(b)].sum())
                    for a, b in zip(split.event_offsets[:-1], split.event_offsets[1:])
                ],
                dtype=np.int64,
            ),
            starts,
            width=width,
            dtype=np.dtype(np.int64),
        ),
        "event_rd_present": event_rd_present,
        "event_reliable_frame_fraction": event_reliable_fraction,
        "event_frame_values": frame_values_padded,
        "event_frame_mask": frame_mask_padded,
        "event_frame_count": event_frame_counts,
        "frame_values": sanitized_frames,
        "frame_feature_reliable": split.frame_feature_reliable.astype(np.bool_),
        "event_offsets": split.event_offsets.astype(np.int64),
        "event_parent_ids": split.parent_ids.tolist(),
        "event_labels": split.labels.tolist(),
        "base_checkpoint_sha256": str(base_checkpoint_sha256),
        "base_classifier_state_sha256": str(base_classifier_state_sha256),
        "v27a_manifest_sha256": str(manifest_sha256),
        "config_identity_sha256": config.config_identity_sha256,
        "stable8_indices": list(config.data.stable8_indices),
        "stable8_names": list(config.data.stable8_names),
        "normalized_frame_values": True,
    }
    payload["cache_payload_sha256"] = cache_payload_sha256(payload)
    return payload


def save_cache_payload(path: Path, payload: Mapping[str, Any]) -> Path:
    declared = payload.get("cache_payload_sha256")
    if not _valid_sha256(declared) or declared != cache_payload_sha256(payload):
        raise ValueError("V27a cache payload hash differs before serialization")
    write_torch_if_absent_or_identical(path, dict(payload))
    return path


def load_cache_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27a cache payload must be a mapping")
    result = dict(payload)
    if result.get("schema_version") != V27A_CACHE_SCHEMA_VERSION:
        raise ValueError("V27a cache schema differs")
    if result.get("experiment_id") != "V27a":
        raise ValueError("V27a cache experiment id differs")
    required = {
        "cache_payload_sha256",
        "parent_labels",
        "parent_sample_ids",
        "labels",
        "lengths",
        "padding_mask",
        "event_baseline_logits",
        "event_summary_tokens",
        "event_summary_valid",
        "event_reliable_frame_count",
        "event_rd_present",
        "event_reliable_frame_fraction",
        "event_frame_values",
        "event_frame_mask",
        "event_frame_count",
        "frame_values",
        "frame_feature_reliable",
        "event_offsets",
        "event_parent_ids",
        "event_labels",
        "base_checkpoint_sha256",
        "base_classifier_state_sha256",
        "v27a_manifest_sha256",
        "config_identity_sha256",
        "stable8_indices",
        "stable8_names",
        "normalized_frame_values",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"V27a cache lacks fields: {missing}")
    declared_payload_sha256 = result["cache_payload_sha256"]
    if (
        not _valid_sha256(declared_payload_sha256)
        or declared_payload_sha256 != cache_payload_sha256(result)
    ):
        raise ValueError("V27a cache payload content hash differs")
    if result.get("split_name") not in {"train", "val"}:
        raise ValueError("V27a cache split name differs")
    parents = [str(value) for value in result["parent_sample_ids"]]
    parent_labels = [str(value) for value in result["parent_labels"]]
    def array(value: Any) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    array_fields = {
        "labels",
        "lengths",
        "padding_mask",
        "event_baseline_logits",
        "event_summary_tokens",
        "event_summary_valid",
        "event_reliable_frame_count",
        "event_rd_present",
        "event_reliable_frame_fraction",
        "event_frame_values",
        "event_frame_mask",
        "event_frame_count",
        "frame_values",
        "frame_feature_reliable",
        "event_offsets",
    }
    for name in array_fields:
        result[name] = array(result[name])

    labels = array(result["labels"])
    lengths = array(result["lengths"])
    padding = array(result["padding_mask"])
    logits = array(result["event_baseline_logits"])
    summary = array(result["event_summary_tokens"])
    summary_valid = array(result["event_summary_valid"])
    counts = array(result["event_reliable_frame_count"])
    rd_present = array(result["event_rd_present"])
    reliable_fraction = array(result["event_reliable_frame_fraction"])
    frame_values = array(result["event_frame_values"])
    frame_mask = array(result["event_frame_mask"])
    frame_count = array(result["event_frame_count"])
    frames = array(result["frame_values"])
    reliable = array(result["frame_feature_reliable"])
    offsets = array(result["event_offsets"])
    event_parent_ids = [str(value) for value in result["event_parent_ids"]]
    event_labels = [str(value) for value in result["event_labels"]]
    if len(parents) != len(set(parents)) or not parents:
        raise ValueError("V27a cache parent ids must be unique and non-empty")
    if len(parent_labels) != len(parents) or any(
        value not in V27A_L2_CLASS_NAMES for value in parent_labels
    ):
        raise ValueError("V27a cache parent labels differ")
    if not all(
        isinstance(result[name], str) and len(result[name]) == 64
        and all(character in "0123456789abcdefABCDEF" for character in result[name])
        for name in (
            "base_checkpoint_sha256",
            "base_classifier_state_sha256",
            "v27a_manifest_sha256",
            "config_identity_sha256",
        )
    ):
        raise ValueError("V27a cache provenance digests are malformed")
    if labels.shape != (len(parents),) or labels.dtype.kind not in "iu":
        raise ValueError("V27a cache labels differ")
    expected_labels = np.asarray(
        [V27A_L2_CLASS_NAMES.index(value) for value in parent_labels],
        dtype=np.int64,
    )
    if not np.array_equal(labels.astype(np.int64), expected_labels):
        raise ValueError("V27a cache numeric labels differ from parent labels")
    if lengths.shape != (len(parents),) or lengths.dtype.kind not in "iu" or np.any(lengths <= 0):
        raise ValueError("V27a cache lengths differ")
    if padding.ndim != 2 or padding.shape[0] != len(parents) or padding.dtype != np.bool_:
        raise ValueError("V27a cache padding differs")
    width = padding.shape[1]
    if np.any(lengths > width) or not np.array_equal((~padding).sum(axis=1), lengths):
        raise ValueError("V27a cache lengths/padding differ")
    expected_padding = np.ones_like(padding, dtype=np.bool_)
    for row, length in enumerate(lengths.tolist()):
        expected_padding[row, : int(length)] = False
    if not np.array_equal(padding, expected_padding):
        raise ValueError("V27a cache padding must be a suffix")
    for value, name, shape in (
        (logits, "event_baseline_logits", (len(parents), width, 3)),
        (summary, "event_summary_tokens", (len(parents), width, 8)),
        (summary_valid, "event_summary_valid", (len(parents), width)),
        (counts, "event_reliable_frame_count", (len(parents), width)),
        (rd_present, "event_rd_present", (len(parents), width)),
        (reliable_fraction, "event_reliable_frame_fraction", (len(parents), width)),
    ):
        if value.shape != shape:
            raise ValueError(f"V27a cache {name} shape differs")
    if (
        summary_valid.dtype != np.bool_
        or counts.dtype.kind not in "iu"
        or rd_present.dtype != np.bool_
        or reliable_fraction.dtype != np.float32
    ):
        raise ValueError("V27a cache summary validity/count dtype differs")
    if not np.isfinite(logits).all() or not np.isfinite(summary).all():
        raise ValueError("V27a cache tensors contain non-finite values")
    if frames.ndim != 2 or frames.shape[1] != 8 or frames.dtype != np.float32 or not np.isfinite(frames).all():
        raise ValueError("V27a cache frame values differ")
    if reliable.shape != (len(frames),) or reliable.dtype != np.bool_:
        raise ValueError("V27a cache reliable mask differs")
    if frame_values.ndim != 4 or frame_values.shape[:2] != (len(parents), width) or frame_values.shape[-1] != 8 or frame_values.dtype != np.float32 or not np.isfinite(frame_values).all():
        raise ValueError("V27a cache event frame values differ")
    if frame_mask.shape != frame_values.shape[:3] or frame_mask.dtype != np.bool_:
        raise ValueError("V27a cache event frame mask differs")
    for row in range(frame_mask.shape[0]):
        for event in range(frame_mask.shape[1]):
            mask = frame_mask[row, event]
            inactive = np.flatnonzero(~mask)
            if inactive.size and bool(mask[int(inactive[0]) :].any()):
                raise ValueError("V27a cache frame masks must be suffix-padded")
    if frame_count.shape != (len(parents), width) or frame_count.dtype.kind not in "iu":
        raise ValueError("V27a cache event frame counts differ")
    if not np.array_equal(frame_count, frame_mask.sum(axis=-1)):
        raise ValueError("V27a cache event frame counts are inconsistent")
    if offsets.shape != (len(event_parent_ids) + 1,) or offsets.dtype.kind not in "iu" or int(offsets[-1]) != len(frames) or int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("V27a cache frame offsets differ")
    if len(event_parent_ids) != len(event_labels) or len(event_parent_ids) != len(offsets) - 1:
        raise ValueError("V27a cache event metadata differs")
    if any(not value for value in event_parent_ids) or any(
        value not in V27A_L2_CLASS_NAMES for value in event_labels
    ):
        raise ValueError("V27a cache event metadata contains empty/unsupported values")
    expected_event_parents = [
        parent
        for parent, length in zip(parents, lengths.tolist(), strict=True)
        for _ in range(int(length))
    ]
    if event_parent_ids != expected_event_parents:
        raise ValueError("V27a cache event parent order differs from parent padding")
    expected_event_labels = [
        label
        for label, length in zip(event_labels_for_parents(result, parents), lengths.tolist(), strict=True)
        for _ in range(int(length))
    ]
    if event_labels != expected_event_labels:
        raise ValueError("V27a cache event label order differs from parent padding")
    expected_parent_from_offsets: list[str] = []
    expected_label_from_offsets: list[str] = []
    for parent, label, length in zip(
        parents, parent_labels, lengths.tolist(), strict=True
    ):
        expected_parent_from_offsets.extend([parent] * int(length))
        expected_label_from_offsets.extend([label] * int(length))
    if event_parent_ids != expected_parent_from_offsets or event_labels != expected_label_from_offsets:
        raise ValueError("V27a cache event metadata is not aligned with parent layout")
    if np.any(frame_values[~frame_mask] != 0.0):
        raise ValueError("V27a invalid event frame values must be zero")
    if np.any(frames[~reliable] != 0.0):
        raise ValueError("V27a invalid flat frame values must be zero")
    if (
        np.any(reliable_fraction < 0.0)
        or np.any(reliable_fraction > 1.0)
        or np.any(reliable_fraction[padding] != 0.0)
        or np.any(rd_present[padding])
    ):
        raise ValueError("V27a event quality facts violate the padding contract")
    if np.any(summary[~summary_valid] != 0.0):
        raise ValueError("V27a invalid summary values must be zero")
    if np.any(logits[padding] != 0.0) or np.any(summary[padding] != 0.0):
        raise ValueError("V27a padded logits/summary values must be zero")
    if np.any(summary_valid[padding]) or np.any(counts[padding] != 0):
        raise ValueError("V27a cache padded/invalid values violate zero contract")
    if np.any(frame_mask[padding]):
        raise ValueError("V27a padded events cannot contain frame tokens")
    if np.any(counts < 0) or np.any(counts > frame_values.shape[2]):
        raise ValueError("V27a cache reliable frame counts are out of bounds")
    if not np.array_equal(counts, frame_count):
        raise ValueError("V27a cache reliable frame count fields differ")
    if not np.array_equal(summary_valid, counts > 0):
        raise ValueError("V27a cache summary validity/count fields differ")
    flat_event_index = 0
    for row, length in enumerate(lengths.tolist()):
        for local_event in range(int(length)):
            start = int(offsets[flat_event_index])
            stop = int(offsets[flat_event_index + 1])
            mask = reliable[start:stop]
            expected_count = int(mask.sum())
            if int(counts[row, local_event]) != expected_count:
                raise ValueError("V27a cache flat/compact reliable counts differ")
            total = stop - start
            expected_fraction = expected_count / total if total else 0.0
            if not np.isclose(
                float(reliable_fraction[row, local_event]),
                expected_fraction,
                rtol=1.0e-6,
                atol=1.0e-6,
            ):
                raise ValueError("V27a cache reliable fraction differs from offsets")
            expected_values = frames[start:stop][mask]
            if not np.array_equal(
                frame_values[row, local_event, :expected_count], expected_values
            ):
                raise ValueError("V27a cache compact/flat frame values differ")
            if expected_count and not np.allclose(
                summary[row, local_event],
                np.median(expected_values, axis=0),
                rtol=0.0,
                atol=1.0e-6,
            ):
                raise ValueError("V27a cache summary differs from flat frame values")
            flat_event_index += 1
    if flat_event_index != len(event_parent_ids):
        raise ValueError("V27a cache event traversal is incomplete")
    try:
        stable8_indices = tuple(int(value) for value in result["stable8_indices"])
    except (TypeError, ValueError) as exc:
        raise ValueError("V27a cache stable8 indices are malformed") from exc
    if stable8_indices != tuple(
        (1, 2, 3, 7, 11, 13, 14, 22)
    ):
        raise ValueError("V27a cache stable8 indices differ")
    if [str(value) for value in result["stable8_names"]] != [
        "residual_positive_8_30mps_energy_fraction",
        "residual_negative_30_60mps_energy_fraction",
        "residual_positive_30_60mps_energy_fraction",
        "residual_rms_offset_mps_over_90",
        "residual_top4_resolution_cell_concentration",
        "residual_sideband_snr_db_over_30",
        "pulse_envelope_rms_clipped_0_2_over_2",
        "pulse_envelope_modulation_snr_db_over_30",
    ]:
        raise ValueError("V27a cache stable8 names differ")
    if result["normalized_frame_values"] is not True:
        raise ValueError("V27a cache must declare normalized frame values")
    return result


def event_labels_for_parents(payload: Mapping[str, Any], parents: list[str]) -> list[str]:
    raw = payload.get("parent_labels")
    if not isinstance(raw, (list, tuple)) or len(raw) != len(parents):
        raise ValueError("V27a cache parent labels are missing or misaligned")
    labels = [str(value) for value in raw]
    if any(value not in V27A_L2_CLASS_NAMES for value in labels):
        raise ValueError("V27a cache contains an unsupported parent label")
    return labels


class V27AEvidenceCacheDataset(Dataset[dict[str, Any]]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.payload = load_cache_payload(self.path)
        self.parent_ids = [str(value) for value in self.payload["parent_sample_ids"]]

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= int(index) < len(self):
            raise IndexError("V27a cache index is out of range")
        row = int(index)
        return {
            "parent_sample_id": self.parent_ids[row],
            "label": torch.as_tensor(self.payload["labels"][row], dtype=torch.long),
            "length": torch.as_tensor(self.payload["lengths"][row], dtype=torch.long),
            "padding_mask": torch.from_numpy(self.payload["padding_mask"][row].copy()),
            "baseline_logits": torch.from_numpy(self.payload["event_baseline_logits"][row].copy()),
            "summary_tokens": torch.from_numpy(self.payload["event_summary_tokens"][row].copy()),
            "summary_valid": torch.from_numpy(self.payload["event_summary_valid"][row].copy()),
            "reliable_frame_count": torch.from_numpy(self.payload["event_reliable_frame_count"][row].copy()),
            "rd_present": torch.from_numpy(self.payload["event_rd_present"][row].copy()),
            "reliable_frame_fraction": torch.from_numpy(
                self.payload["event_reliable_frame_fraction"][row].copy()
            ),
            "frame_values": torch.from_numpy(self.payload["event_frame_values"][row].copy()),
            "frame_mask": torch.from_numpy(self.payload["event_frame_mask"][row].copy()),
            "frame_count": torch.from_numpy(self.payload["event_frame_count"][row].copy()),
        }


def materialize_frozen_logit_caches(
    context: V27AContext,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Run the frozen V27 L2 classifier once and persist train/val caches."""

    import types

    from v27.v27_baseline.base_binding import load_frozen_base_model
    from v27.v27_baseline.data import (
        V27TaskDataset,
        build_v27_dataloader,
        model_input_kwargs,
    )

    from .evidence import _load_split, _validate_index
    from .runtime import load_verified_cache

    root = context.project_root
    index = _validate_index(root, context.config)
    normalization_path = resolve_project_path(
        root, context.config.artifacts.normalization_path, must_exist=True
    )
    median, scale, normalization = load_normalization(normalization_path)
    frozen_context = types.SimpleNamespace(
        project_root=root,
        config=context.v27_config,
    )
    base_model, checkpoint, binding = load_frozen_base_model(
        frozen_context,
        task="dbb_specialist",
        device=device,
        binding_path=context.config.base.frozen_binding_path,
    )
    base_model.eval()
    split_results: dict[str, Any] = {}
    for split_name in ("train", "val"):
        target = (
            resolve_project_path(root, context.config.artifacts.train_cache_path)
            if split_name == "train"
            else resolve_project_path(root, context.config.artifacts.validation_cache_path)
        )
        if target.exists():
            # Torch archive bytes are not guaranteed to be stable across saves.
            # Reuse an existing cache after verifying its semantic payload and
            # provenance instead of comparing serialized bytes.
            stored = load_verified_cache(context, split_name)
            split_results[split_name] = {
                "path": str(target),
                "parent_count": len(stored["parent_sample_ids"]),
                "event_count": len(stored["event_parent_ids"]),
                "frame_count": int(stored["frame_values"].shape[0]),
                "normalization_parent_count_used": normalization["parent_count_used"],
                "cache_payload_sha256": cache_payload_sha256(stored),
                "reuse": "verified_existing_semantic_payload",
            }
            continue
        evidence = _load_split(root, context.config, split_name=split_name, index=index)
        dataset = V27TaskDataset(
            root,
            context.v27_config,
            task="dbb_specialist",
            split_name=split_name,
        )
        dataset_event_ids = dataset.base.frame["sample_id"].astype(str).tolist()
        if dataset_event_ids != evidence.sample_ids.tolist():
            raise ValueError(
                f"V27a V27 dataset/evidence event order differs for {split_name}"
            )
        loader = build_v27_dataloader(
            dataset,
            batch_size=context.config.training.batch_size,
            shuffle=False,
        )
        logits: list[np.ndarray] = []
        parent_ids: list[str] = []
        with torch.inference_mode():
            for batch in loader:
                output = base_model(
                    **model_input_kwargs(batch, device), return_trace=False
                )
                sequence = output["sequence_logits"].detach().cpu().numpy()
                for row, parent_id in enumerate(batch["parent_sample_ids"]):
                    length = int(batch["lengths"][row])
                    logits.append(sequence[row, :length].astype(np.float32, copy=True))
                    parent_ids.append(str(parent_id))
        baseline_event_logits = np.concatenate(logits, axis=0)
        expected_parents = _parent_order(evidence)
        if parent_ids != expected_parents:
            raise ValueError(f"V27a frozen base parent order differs for {split_name}")
        normalized_frames = normalize_stable8(
            evidence.stable8_values,
            evidence.frame_feature_reliable,
            median=median,
            scale=scale,
        )
        payload = build_cache_payload(
            evidence,
            normalized_frames=normalized_frames,
            baseline_event_logits=baseline_event_logits,
            config=context.config,
            split_name=split_name,
            base_checkpoint_sha256=str(
                context.config.base.expected_l2_checkpoint_sha256
            ),
            base_classifier_state_sha256=str(
                context.config.base.expected_l2_classifier_state_sha256
            ),
            manifest_sha256=context.manifest_sha256,
        )
        save_cache_payload(target, payload)
        stored = load_cache_payload(target)
        if stored["cache_payload_sha256"] != payload["cache_payload_sha256"]:
            raise ValueError("V27a serialized cache content hash differs")
        split_results[split_name] = {
            "path": str(target),
            "parent_count": len(parent_ids),
            "event_count": int(evidence.event_count),
            "frame_count": int(evidence.frame_count),
            "normalization_parent_count_used": normalization["parent_count_used"],
            "cache_payload_sha256": stored["cache_payload_sha256"],
        }
    return {
        "status": "materialized_frozen_logit_evidence_caches",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "binding_sha256": binding["binding_sha256"],
        "splits": split_results,
    }


def build_cache_dataloader(
    dataset: V27AEvidenceCacheDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    if int(batch_size) <= 0:
        raise ValueError("V27a cache batch size must be positive")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


__all__ = ["V27AEvidenceCacheDataset", "build_cache_dataloader", "build_cache_payload", "cache_payload_sha256", "load_cache_payload", "materialize_frozen_logit_caches", "save_cache_payload"]