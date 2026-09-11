"""Mixed-pool feature materialization for V27b-r6.

The implementation deliberately keeps the frozen V27 representation cache
immutable.  R6 creates a new sidecar cache carrying only the additional
reliable-stable8 facts required by the treatment arm.  This prevents the
historical V26 `valid == RD present` bit from being silently mistaken for
frame-level micro-Doppler reliability.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from build.micro_doppler_v17 import _CanonicalFrameProvenance
from build.micro_doppler_v18 import _build_split
from radar_ln3.runtime import (
    canonical_sha256,
    relative_project_path,
    resolve_project_path,
    sha256_file,
    write_torch_if_absent_or_identical,
)

from .protocol import (
    R6_DIAGNOSTIC_LABEL,
    R6_FEATURE_CACHE_SCHEMA_VERSION,
    R6_FEATURE_MANIFEST_SCHEMA_VERSION,
    R6_KNOWN_LABELS,
    R6_NEGATIVE_LABEL,
    R6_ROLES,
    R6_STABLE8_INDICES,
    R6_STABLE8_NAMES,
    R6Context,
    r6_artifact_path,
    write_sealed_json_if_identical,
)


_CACHE_TENSOR_FIELDS = (
    "event_trajectory",
    "summary_features",
    "task_facts_sequence",
    "padding_mask",
    "lengths",
    "stable8_summary",
    "stable8_reliable",
    "stable8_reliable_count",
    "stable8_reliable_fraction",
    "rd_present",
    "quality_eligible",
    "fit_eligible",
)


def _tensor_digest(values: Mapping[str, torch.Tensor]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27b-r6 cache tensor is not a tensor: {name}")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def feature_cache_digest(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "cache_sha256"}
    tensors = body.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27b-r6 feature cache lacks tensors")
    metadata = {key: value for key, value in body.items() if key != "tensors"}
    return canonical_sha256(
        {"metadata": metadata, "tensor_sha256": _tensor_digest(dict(tensors))}
    )


def _project_root_frame(context: R6Context) -> pd.DataFrame:
    from v27.v27_baseline.runtime import load_v27_context

    v27_context = load_v27_context(
        context.project_root, context.config.base.v27_system_config_path
    )
    source = resolve_project_path(
        context.project_root, v27_context.config.data.index_path, must_exist=True
    )
    frame = pd.read_csv(source, low_memory=False)
    required = {
        "sample_id",
        "parent_sample_id",
        "split_name",
        "label_name",
        "causal_event_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V27b-r6 index lacks required columns: {missing}")
    return frame


def _r6_reference_cache_path(context: R6Context, *, split_name: str) -> Path:
    """Resolve the archived V27a cache that defines reliable stable8 facts."""

    if split_name == "train":
        value = context.config.base.v27a_train_reference_cache_path
        expected = context.config.base.expected_v27a_train_reference_cache_sha256
    elif split_name == "val":
        value = context.config.base.v27a_validation_reference_cache_path
        expected = context.config.base.expected_v27a_validation_reference_cache_sha256
    else:
        raise ValueError(f"V27b-r6 has no V27a reference cache for split: {split_name}")
    path = resolve_project_path(context.project_root, value, must_exist=True)
    if sha256_file(path) != expected:
        raise ValueError("V27b-r6 archived V27a reference-cache file hash differs")
    return path


def _archived_v27a_known_sidecar(
    context: R6Context,
    *,
    sample_ids: Sequence[str],
    labels: Sequence[str],
    split_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reuse V27a's frozen, reliable-frame event summaries for known DBB rows.

    This prevents r6 from silently reimplementing a historical normalisation or
    treating the older V27 mixed-cache ``valid`` flag as frame reliability.
    """

    from v27.v27a.cache import load_cache_payload

    path = _r6_reference_cache_path(context, split_name=split_name)
    payload = load_cache_payload(path)
    expected_digest = (
        context.config.base.expected_v27a_train_reference_cache_payload_sha256
        if split_name == "train"
        else context.config.base.expected_v27a_validation_reference_cache_payload_sha256
    )
    if payload.get("cache_payload_sha256") != expected_digest:
        raise ValueError("V27b-r6 archived V27a reference-cache payload hash differs")
    archived_parent_positions = {
        str(parent_id): index
        for index, parent_id in enumerate(payload["parent_sample_ids"])
    }
    labels_by_parent = {
        str(parent_id): str(label)
        for parent_id, label in zip(
            payload["parent_sample_ids"], payload["parent_labels"], strict=True
        )
    }
    requested = [str(value) for value in sample_ids]
    requested_labels = [str(value) for value in labels]
    if len(requested) != len(requested_labels):
        raise ValueError("V27b-r6 known stable8 request metadata differs")
    event_index_by_sample = _project_root_frame(context).set_index("sample_id")[
        ["parent_sample_id", "causal_event_index", "label_name"]
    ]
    result = {
        "stable8_summary": np.zeros(
            (len(requested), len(R6_STABLE8_INDICES)), dtype=np.float32
        ),
        "stable8_reliable": np.zeros((len(requested),), dtype=np.bool_),
        "stable8_reliable_count": np.zeros((len(requested),), dtype=np.int64),
        "stable8_reliable_fraction": np.zeros((len(requested),), dtype=np.float32),
        "rd_present": np.zeros((len(requested),), dtype=np.bool_),
    }
    for index, (sample_id, label) in enumerate(
        zip(requested, requested_labels, strict=True)
    ):
        if label not in R6_KNOWN_LABELS:
            continue
        if sample_id not in event_index_by_sample.index:
            raise ValueError("V27b-r6 known event is absent from the canonical index")
        event_record = event_index_by_sample.loc[sample_id]
        parent_id = str(event_record["parent_sample_id"])
        if str(event_record["label_name"]) != label or labels_by_parent.get(parent_id) != label:
            raise ValueError("V27b-r6 archived V27a known parent binding differs")
        parent_row = archived_parent_positions.get(parent_id)
        if parent_row is None:
            raise ValueError("V27b-r6 archived V27a known parent is absent")
        event_index_value = event_record["causal_event_index"]
        if not float(event_index_value).is_integer():
            raise ValueError("V27b-r6 canonical causal event index is non-integral")
        event_index = int(event_index_value)
        length = int(payload["lengths"][parent_row])
        if not 0 <= event_index < length:
            raise ValueError("V27b-r6 known event index exceeds archived V27a parent")
        result["stable8_summary"][index] = payload["event_summary_tokens"][
            parent_row, event_index
        ]
        result["stable8_reliable"][index] = bool(
            payload["event_summary_valid"][parent_row, event_index]
        )
        result["stable8_reliable_count"][index] = int(
            payload["event_reliable_frame_count"][parent_row, event_index]
        )
        result["stable8_reliable_fraction"][index] = float(
            payload["event_reliable_frame_fraction"][parent_row, event_index]
        )
        result["rd_present"][index] = bool(
            payload["event_rd_present"][parent_row, event_index]
        )
    return result, {
        "path": relative_project_path(context.project_root, path),
        "sha256": sha256_file(path),
        "cache_payload_sha256": expected_digest,
        "reliability_field": "event_summary_valid_from_frame_feature_reliable",
        "normalized_frame_source": "frozen_v27a_train_known_median_iqr",
    }


def _normalization(context: R6Context) -> tuple[np.ndarray, np.ndarray]:
    """Load the frozen V27a train-only stable8 normalization contract."""

    from v27.v27a.evidence import load_normalization

    path = resolve_project_path(
        context.project_root, context.config.base.v27a_normalization_path, must_exist=True
    )
    if sha256_file(path) != context.config.base.expected_v27a_normalization_sha256:
        raise ValueError("V27b-r6 V27a stable8 normalization hash differs")
    median, scale, _ = load_normalization(path)
    return median, scale


def _evidence_arrays(
    context: R6Context,
    *,
    role: str,
    sample_ids: Sequence[str],
    labels: Sequence[str],
    split_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Return reliable stable8 summaries aligned to a mixed role event order.

    Known DBB rows reuse the archived V27a reliable-frame sidecar. Train
    Unknown rows are materialized with the same label-free extractor and
    normalised by V27a's frozen train-only contract. Clutter remains
    diagnostic-only, but receives the same extractor so its score cannot be
    mistaken for a synthetic missing-modality pattern.
    """

    root = context.project_root
    known_set = set(R6_KNOWN_LABELS)
    requested = [str(value) for value in sample_ids]
    requested_labels = [str(value) for value in labels]
    if len(requested) != len(requested_labels) or len(requested) != len(set(requested)):
        raise ValueError("V27b-r6 requested event identities differ")
    output = {
        "stable8_summary": np.zeros((len(requested), len(R6_STABLE8_INDICES)), dtype=np.float32),
        "stable8_reliable": np.zeros((len(requested),), dtype=np.bool_),
        "stable8_reliable_count": np.zeros((len(requested),), dtype=np.int64),
        "stable8_reliable_fraction": np.zeros((len(requested),), dtype=np.float32),
        "rd_present": np.zeros((len(requested),), dtype=np.bool_),
    }
    positions = {sample_id: index for index, sample_id in enumerate(requested)}
    observed_ids: set[str] = set()
    archived_known, archived_known_provenance = _archived_v27a_known_sidecar(
        context,
        sample_ids=requested,
        labels=requested_labels,
        split_name=split_name,
    )
    known_positions = [
        index
        for index, label in enumerate(requested_labels)
        if label in known_set
    ]
    for field, value in archived_known.items():
        output[field][known_positions] = value[known_positions]
    observed_ids.update(requested[index] for index in known_positions)
    median, scale = _normalization(context)


    def merge_into(
        destination_output: dict[str, np.ndarray],
        *,
        source_ids: Sequence[str],
        valid: np.ndarray,
        offsets: np.ndarray,
        frames: np.ndarray,
        reliable: np.ndarray,
    ) -> None:
        if len(source_ids) + 1 != len(offsets):
            raise ValueError("V27b-r6 evidence offsets differ from source events")
        for source_index, raw_id in enumerate(source_ids):
            sample_id = str(raw_id)
            destination = positions.get(sample_id)
            if destination is None:
                continue
            if sample_id in observed_ids:
                raise ValueError("V27b-r6 sidecar source repeats a requested event")
            observed_ids.add(sample_id)
            start, stop = int(offsets[source_index]), int(offsets[source_index + 1])
            event_reliable = reliable[start:stop]
            event_values = frames[start:stop, :]
            count = int(event_reliable.sum())
            destination_output["rd_present"][destination] = bool(valid[source_index])
            destination_output["stable8_reliable_count"][destination] = count
            destination_output["stable8_reliable_fraction"][destination] = (
                float(count) / float(stop - start) if stop > start else 0.0
            )
            if count:
                summary = np.median(
                    event_values[event_reliable][:, R6_STABLE8_INDICES], axis=0
                ).astype(np.float32)
                destination_output["stable8_summary"][destination] = summary
                destination_output["stable8_reliable"][destination] = True
            if count and not bool(valid[source_index]):
                raise ValueError("V27b-r6 reliable stable8 frame belongs to RD-missing event")

    non_known_labels = {R6_NEGATIVE_LABEL, R6_DIAGNOSTIC_LABEL}
    if any(label not in known_set | non_known_labels for label in requested_labels):
        raise ValueError("V27b-r6 sidecar contains an unsupported label")
    non_known_requested = {
        sample_id
        for sample_id, label in zip(requested, requested_labels, strict=True)
        if label in non_known_labels
    }
    extracted_provenance: dict[str, Any] = {"status": "not_required"}
    if non_known_requested:
        frame = _project_root_frame(context)
        selected = frame[
            frame["sample_id"].astype(str).isin(non_known_requested)
        ].copy()
        if set(selected["sample_id"].astype(str)) != non_known_requested:
            raise ValueError("V27b-r6 non-known event identities differ from index")
        allowed = tuple(sorted(set(selected["label_name"].astype(str))))
        if not set(allowed).issubset(non_known_labels):
            raise ValueError("V27b-r6 non-known extraction label scope differs")
        provenance_path = resolve_project_path(root, context.config.base.v16_provenance_path, must_exist=True)
        canonical_path = resolve_project_path(root, context.config.base.canonical_frame_index_path, must_exist=True)
        if (
            sha256_file(provenance_path) != context.config.base.expected_v16_provenance_sha256
            or sha256_file(canonical_path) != context.config.base.expected_canonical_frame_index_sha256
        ):
            raise ValueError("V27b-r6 non-known source provenance hash differs")
        source_manifest = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance = {
            str(row["parent_sample_id"]): dict(row)
            for row in list(source_manifest.get("records") or ())
        }
        canonical = _CanonicalFrameProvenance(root, canonical_path)
        arrays, statistics = _build_split(
            root,
            split_name,
            selected,
            provenance,
            canonical,
            allowed_labels=allowed,
            include_evidence=True,
        )
        raw_sidecar = {
            "stable8_summary": np.zeros_like(output["stable8_summary"]),
            "stable8_reliable": np.zeros_like(output["stable8_reliable"]),
            "stable8_reliable_count": np.zeros_like(output["stable8_reliable_count"]),
            "stable8_reliable_fraction": np.zeros_like(output["stable8_reliable_fraction"]),
            "rd_present": np.zeros_like(output["rd_present"]),
        }
        merge_into(
            raw_sidecar,
            source_ids=[str(value) for value in arrays["sample_ids"].tolist()],
            valid=np.asarray(arrays["valid"]),
            offsets=np.asarray(arrays["event_offsets"]),
            frames=np.asarray(arrays["frame_values"]),
            reliable=np.asarray(arrays["frame_feature_reliable"]),
        )
        reliable_rows = raw_sidecar["stable8_reliable"]
        non_known_mask = np.asarray(
            [label in non_known_labels for label in requested_labels], dtype=np.bool_
        )
        normalized_rows = reliable_rows & non_known_mask
        output["stable8_summary"][normalized_rows] = (
            (raw_sidecar["stable8_summary"][normalized_rows] - median) / scale
        ).astype(np.float32)
        for field in (
            "stable8_reliable",
            "stable8_reliable_count",
            "stable8_reliable_fraction",
            "rd_present",
        ):
            output[field][non_known_mask] = raw_sidecar[field][non_known_mask]
        extracted_provenance = {
            "status": "label_free_current_event_evidence_materialized",
            "labels_select_scope_only": True,
            "allowed_labels": list(allowed),
            "statistics": json.loads(json.dumps(dict(statistics))),
            "raw_feature_values_depend_on_label": False,
        }
    if observed_ids != set(requested):
        missing = sorted(set(requested) - observed_ids)
        raise ValueError(f"V27b-r6 sidecar source lacks requested events: {missing[:3]}")
    if not np.isfinite(output["stable8_summary"]).all():
        raise FloatingPointError("V27b-r6 stable8 summary is non-finite")
    if np.any(output["stable8_summary"][~output["stable8_reliable"]] != 0.0):
        raise ValueError("V27b-r6 unreliable stable8 summary must be zero")
    if np.any(output["stable8_reliable"] & ~output["rd_present"]):
        raise ValueError("V27b-r6 reliable stable8 requires RD presence")
    return output, {
        "known_source": archived_known_provenance,
        "non_known_source": extracted_provenance,
        "stable8_names": list(R6_STABLE8_NAMES),
        "reliability_is_not_rd_present": True,
    }


def _quality_eligible(
    task_facts: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    """Reproduce the V27 deterministic final-quality fact target locally."""

    if task_facts.ndim != 3 or task_facts.shape[-1] != 7 or padding.shape != task_facts.shape[:2]:
        raise ValueError("V27b-r6 quality fact shapes differ")
    lengths = (~padding).sum(dim=1).to(dtype=torch.long)
    rows = torch.arange(lengths.shape[0])
    final = lengths - 1
    selected = task_facts[rows, final]
    return (
        (lengths >= 3)
        & selected[:, 1].ge(0.5)
        & selected[:, 3].ge(0.5)
        & torch.minimum(selected[:, 5], selected[:, 6]).ge(0.5)
    )


def _summary_stability(
    summary: torch.Tensor,
    reliable: torch.Tensor,
    reliable_count: torch.Tensor,
    reliable_fraction: torch.Tensor,
    padding: torch.Tensor,
) -> torch.Tensor:
    """Create causal stable8 summary features without a learnable encoder.

    At each prefix the branch records reliable coverage, last reliable token,
    reliable-token mean/std, and deviation of the last token from the causal
    mean.  This is the smallest faithful transfer of V27a's successful
    *event-summary* carrier; it intentionally does not revive r5's failed
    frame-GRU classification residual.
    """

    if summary.ndim != 3 or summary.shape[-1] != len(R6_STABLE8_INDICES):
        raise ValueError("V27b-r6 stable8 summary shape differs")
    if (
        reliable.shape != summary.shape[:2]
        or reliable_count.shape != reliable.shape
        or reliable_fraction.shape != reliable.shape
        or padding.shape != reliable.shape
    ):
        raise ValueError("V27b-r6 stable8 reliability shape differs")
    if reliable_count.dtype != torch.long or not reliable_fraction.is_floating_point():
        raise ValueError("V27b-r6 stable8 reliability dtypes differ")
    if not torch.isfinite(reliable_fraction).all() or bool((reliable_count < 0).any()):
        raise ValueError("V27b-r6 stable8 reliability values differ")
    active = ~padding
    valid = reliable & active
    weights = valid.to(dtype=summary.dtype).unsqueeze(-1)
    counts = weights.cumsum(dim=1)
    totals = (summary * weights).cumsum(dim=1)
    squared = (summary.square() * weights).cumsum(dim=1)
    denominator = counts.clamp_min(1.0)
    mean = totals / denominator
    variance = (squared / denominator - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    last = summary * weights
    for index in range(1, summary.shape[1]):
        last[:, index] = torch.where(
            valid[:, index].unsqueeze(-1),
            summary[:, index],
            last[:, index - 1],
        )
    deviation = (last - mean).abs()
    prefix_events = active.to(dtype=summary.dtype).cumsum(dim=1).clamp_min(1.0)
    coverage = counts.squeeze(-1) / prefix_events
    normalized_count = reliable_count.to(dtype=summary.dtype) / 12.0
    observed_fraction = reliable_fraction * valid.to(dtype=summary.dtype)
    frame_fraction_mean = observed_fraction.cumsum(dim=1) / counts.squeeze(-1).clamp_min(1.0)
    frame_count_mean = (
        normalized_count * valid.to(dtype=summary.dtype)
    ).cumsum(dim=1) / counts.squeeze(-1).clamp_min(1.0)
    result = torch.cat(
        (
            last,
            mean,
            std,
            deviation,
            coverage.unsqueeze(-1),
            counts.clamp(max=float(summary.shape[1])) / float(summary.shape[1]),
            observed_fraction.unsqueeze(-1),
            frame_fraction_mean.unsqueeze(-1),
            (normalized_count * valid.to(dtype=summary.dtype)).unsqueeze(-1),
            frame_count_mean.unsqueeze(-1),
        ),
        dim=-1,
    )
    return result.masked_fill(padding.unsqueeze(-1), 0.0)


def _validate_feature_cache(payload: Mapping[str, Any], context: R6Context, *, role: str) -> dict[str, Any]:
    if payload.get("schema_version") != R6_FEATURE_CACHE_SCHEMA_VERSION:
        raise ValueError("V27b-r6 feature cache schema differs")
    if payload.get("experiment_id") != context.config.experiment_id or payload.get("role") != role:
        raise ValueError("V27b-r6 feature cache identity differs")
    if payload.get("system_manifest_sha256") != context.config.base.expected_v27_system_manifest_sha256:
        raise ValueError("V27b-r6 feature cache V27 manifest differs")
    if payload.get("r6_manifest_sha256") != context.manifest_sha256:
        raise ValueError("V27b-r6 feature cache manifest binding differs")
    if payload.get("representation_cache_manifest_sha256") != context.config.base.expected_v27_representation_cache_manifest_sha256:
        raise ValueError("V27b-r6 feature cache representation binding differs")
    if payload.get("pool_registry_sha256") != context.config.base.expected_v27_open_set_pool_registry_sha256:
        raise ValueError("V27b-r6 feature cache pool binding differs")
    if payload.get("stable8_indices") != list(R6_STABLE8_INDICES) or payload.get("stable8_names") != list(R6_STABLE8_NAMES):
        raise ValueError("V27b-r6 feature cache stable8 schema differs")
    tensors = payload.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != set(_CACHE_TENSOR_FIELDS):
        raise ValueError("V27b-r6 feature cache tensor fields differ")
    parents = [str(value) for value in payload.get("parent_ids") or ()]
    labels = [str(value) for value in payload.get("label_names") or ()]
    if not parents or len(parents) != len(labels) or len(parents) != len(set(parents)):
        raise ValueError("V27b-r6 feature cache parents/labels differ")
    if payload.get("parent_ids_sha256") != canonical_sha256(parents):
        raise ValueError("V27b-r6 feature cache parent hash differs")
    if any(
        label not in {*R6_KNOWN_LABELS, R6_NEGATIVE_LABEL, R6_DIAGNOSTIC_LABEL}
        for label in labels
    ):
        raise ValueError("V27b-r6 feature cache has an unsupported label")
    count = len(parents)
    width = context.config.data.max_sequence_length
    expected = {
        "event_trajectory": (count, width, context.config.data.event_trajectory_dim),
        "summary_features": (count, width, context.config.data.summary_feature_dim),
        "task_facts_sequence": (count, width, context.config.data.task_fact_dim),
        "padding_mask": (count, width),
        "lengths": (count,),
        "stable8_summary": (count, width, context.config.data.stable8_dim),
        "stable8_reliable": (count, width),
        "stable8_reliable_count": (count, width),
        "stable8_reliable_fraction": (count, width),
        "rd_present": (count, width),
        "quality_eligible": (count,),
        "fit_eligible": (count,),
    }
    for name, shape in expected.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            raise ValueError(f"V27b-r6 feature cache tensor shape differs: {name}")
        if name in {"padding_mask", "stable8_reliable", "rd_present", "quality_eligible", "fit_eligible"}:
            if value.dtype != torch.bool:
                raise ValueError(f"V27b-r6 feature cache bool tensor differs: {name}")
        elif name in {"lengths", "stable8_reliable_count"}:
            if value.dtype != torch.long:
                raise ValueError(f"V27b-r6 feature cache integer tensor differs: {name}")
        elif not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"V27b-r6 feature cache float tensor differs: {name}")
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    expected_padding = torch.arange(width).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(padding.cpu(), expected_padding):
        raise ValueError("V27b-r6 feature cache padding differs from lengths")
    if bool((lengths < 1).any()) or bool((lengths > width).any()):
        raise ValueError("V27b-r6 feature cache lengths are out of bounds")
    for name in (
        "event_trajectory",
        "summary_features",
        "task_facts_sequence",
        "stable8_summary",
    ):
        if bool(
            tensors[name]
            .masked_select(padding.unsqueeze(-1).expand_as(tensors[name]))
            .ne(0)
            .any()
        ):
            raise ValueError(f"V27b-r6 padded value is non-zero: {name}")
    if bool(tensors["stable8_reliable_fraction"].masked_select(padding).ne(0).any()):
        raise ValueError("V27b-r6 padded value is non-zero: stable8_reliable_fraction")
    for name in ("stable8_reliable", "rd_present"):
        if bool(tensors[name].masked_select(padding).any()):
            raise ValueError(f"V27b-r6 padded fact is true: {name}")
    if bool(tensors["stable8_reliable"].logical_and(~tensors["rd_present"]).any()):
        raise ValueError("V27b-r6 reliable stable8 exists without RD")
    if bool(
        tensors["stable8_summary"]
        .masked_select(
            (~tensors["stable8_reliable"])
            .unsqueeze(-1)
            .expand_as(tensors["stable8_summary"])
        )
        .ne(0)
        .any()
    ):
        raise ValueError("V27b-r6 unreliable stable8 summary is non-zero")
    reliable = tensors["stable8_reliable"]
    reliable_count = tensors["stable8_reliable_count"]
    reliable_fraction = tensors["stable8_reliable_fraction"]
    if bool((reliable_count < 0).any()) or bool(
        reliable_count.masked_select(~reliable).ne(0).any()
    ):
        raise ValueError("V27b-r6 reliable stable8 frame count differs")
    if bool((reliable_count > context.config.data.max_frames_per_event).any()):
        raise ValueError("V27b-r6 reliable stable8 frame count exceeds the event bound")
    if bool((reliable_fraction < 0).any()) or bool((reliable_fraction > 1).any()):
        raise ValueError("V27b-r6 reliable stable8 frame fraction is out of bounds")
    if bool(reliable_fraction.masked_select(~reliable).ne(0).any()):
        raise ValueError("V27b-r6 unreliable stable8 frame fraction differs")
    quality = tensors["quality_eligible"]
    expected_quality = _quality_eligible(tensors["task_facts_sequence"], padding)
    if not torch.equal(quality.cpu(), expected_quality.cpu()):
        raise ValueError("V27b-r6 deterministic quality eligibility differs")
    final_reliable_count = reliable.sum(dim=1)
    final_coverage = final_reliable_count.to(dtype=torch.float32) / lengths.to(dtype=torch.float32)
    known_or_unknown = torch.tensor(
        [label in {*R6_KNOWN_LABELS, R6_NEGATIVE_LABEL} for label in labels],
        dtype=torch.bool,
    )
    expected_fit = (
        quality
        & final_reliable_count.ge(context.config.quality.minimum_reliable_event_count)
        & final_coverage.ge(context.config.quality.minimum_reliable_event_fraction)
        & known_or_unknown
    )
    if not torch.equal(tensors["fit_eligible"].cpu(), expected_fit.cpu()):
        raise ValueError("V27b-r6 knownness fit eligibility differs")
    if payload.get("cache_sha256") != feature_cache_digest(payload):
        raise ValueError("V27b-r6 feature cache hash differs")
    return dict(payload)


def _cache_path(context: R6Context, *, role: str) -> Path:
    return r6_artifact_path(context.project_root, context.config.artifacts.feature_cache_template, role=role)


def materialize_feature_caches(context: R6Context) -> dict[str, Any]:
    """Materialize r6 sidecars for all roles without altering V27 caches."""

    from v27.v27_baseline.open_set import load_representation_cache
    from v27.v27_baseline.open_set_data import load_open_set_pool_artifacts

    v27_context = __import__("v27.v27_baseline.runtime", fromlist=["load_v27_context"]).load_v27_context(
        context.project_root, context.config.base.v27_system_config_path
    )
    pools, _ = load_open_set_pool_artifacts(v27_context)
    results: dict[str, Any] = {}
    for role in R6_ROLES:
        role_record = pools["roles"][role]
        representation = load_representation_cache(
            v27_context, task=context.config.data.task, role=role
        )
        parents = [str(value) for value in representation["parent_ids"]]
        labels = [str(value) for value in representation["label_names"]]
        if parents != [str(value) for value in role_record["parent_ids"]]:
            raise ValueError("V27b-r6 representation parent order differs")
        index = _project_root_frame(context).set_index("sample_id")
        role_sample_ids = [str(value) for value in role_record["sample_ids"]]
        if index.index.duplicated().any():
            raise ValueError("V27b-r6 canonical index has duplicate sample ids")
        role_labels = [
            str(value)
            for value in index.loc[role_sample_ids, "label_name"].tolist()
        ]
        sidecar, sidecar_provenance = _evidence_arrays(
            context,
            role=role,
            sample_ids=role_sample_ids,
            labels=role_labels,
            split_name=str(role_record["split_name"]),
        )
        lengths = representation["tensors"]["lengths"].clone()
        padding = representation["tensors"]["padding_mask"].clone()
        event_count = int(lengths.sum().item())
        if event_count != len(role_record["sample_ids"]):
            raise ValueError("V27b-r6 representation event count differs from role")
        shape = (len(parents), context.config.data.max_sequence_length)
        stable_summary = torch.zeros(
            (*shape, context.config.data.stable8_dim), dtype=torch.float32
        )
        reliable = torch.zeros(shape, dtype=torch.bool)
        reliable_count = torch.zeros(shape, dtype=torch.long)
        reliable_fraction = torch.zeros(shape, dtype=torch.float32)
        rd_present = torch.zeros(shape, dtype=torch.bool)
        cursor = 0
        for row, length in enumerate(lengths.tolist()):
            stop = cursor + int(length)
            stable_summary[row, :length] = torch.from_numpy(sidecar["stable8_summary"][cursor:stop])
            reliable[row, :length] = torch.from_numpy(sidecar["stable8_reliable"][cursor:stop])
            reliable_count[row, :length] = torch.from_numpy(sidecar["stable8_reliable_count"][cursor:stop])
            reliable_fraction[row, :length] = torch.from_numpy(sidecar["stable8_reliable_fraction"][cursor:stop])
            rd_present[row, :length] = torch.from_numpy(sidecar["rd_present"][cursor:stop])
            cursor = stop
        if cursor != event_count:
            raise ValueError("V27b-r6 sidecar event traversal is incomplete")
        quality = _quality_eligible(representation["tensors"]["task_facts_sequence"], padding)
        reliable_final = reliable[torch.arange(len(parents)), lengths - 1]
        reliable_coverage = reliable.sum(dim=1).to(dtype=torch.float32) / lengths.to(dtype=torch.float32)
        labels_tensor = np.asarray(labels, dtype=str)
        known = torch.from_numpy(np.isin(labels_tensor, R6_KNOWN_LABELS))
        unknown = torch.from_numpy(labels_tensor == R6_NEGATIVE_LABEL)
        fit_eligible = (
            quality
            & (reliable.sum(dim=1) >= context.config.quality.minimum_reliable_event_count)
            & reliable_coverage.ge(context.config.quality.minimum_reliable_event_fraction)
            & (known | unknown)
        )
        if bool(fit_eligible.logical_and(torch.from_numpy(labels_tensor == R6_DIAGNOSTIC_LABEL)).any()):
            raise AssertionError("V27b-r6 Clutter cannot become a fit example")
        tensors = {
            "event_trajectory": representation["tensors"]["event_trajectory"].clone(),
            "summary_features": representation["tensors"]["summary_features"].clone(),
            "task_facts_sequence": representation["tensors"]["task_facts_sequence"].clone(),
            "padding_mask": padding,
            "lengths": lengths,
            "stable8_summary": stable_summary,
            "stable8_reliable": reliable,
            "stable8_reliable_count": reliable_count,
            "stable8_reliable_fraction": reliable_fraction,
            "rd_present": rd_present,
            "quality_eligible": quality,
            "fit_eligible": fit_eligible,
        }
        payload: dict[str, Any] = {
            "schema_version": R6_FEATURE_CACHE_SCHEMA_VERSION,
            "experiment_id": context.config.experiment_id,
            "role": role,
            "r6_manifest_sha256": context.manifest_sha256,
            "system_manifest_sha256": context.config.base.expected_v27_system_manifest_sha256,
            "pool_registry_sha256": pools["registry_sha256"],
            "representation_cache_manifest_sha256": context.config.base.expected_v27_representation_cache_manifest_sha256,
            "representation_cache_sha256": representation["cache_sha256"],
            "parent_ids": parents,
            "parent_ids_sha256": canonical_sha256(parents),
            "label_names": labels,
            "stable8_indices": list(R6_STABLE8_INDICES),
            "stable8_names": list(R6_STABLE8_NAMES),
            "stable8_normalization": "v27a_train_known_parent_equal_median_iqr",
            "stable8_reliability": "frame_feature_reliable_not_rd_present",
            "sidecar_provenance": sidecar_provenance,
            "tensors": tensors,
            "classification_logits_source": "frozen_v27_l2_representation_cache",
            "unknown_is_classifier_class": False,
            "test_payload_access_count": 0,
        }
        payload["cache_sha256"] = feature_cache_digest(payload)
        checked = _validate_feature_cache(payload, context, role=role)
        target = _cache_path(context, role=role)
        if target.exists():
            existing = torch.load(target, map_location="cpu", weights_only=False)
            if not isinstance(existing, Mapping) or existing.get("cache_sha256") != payload["cache_sha256"]:
                raise FileExistsError(
                    f"refusing to replace a different V27b-r6 feature cache: {target}"
                )
        else:
            write_torch_if_absent_or_identical(target, payload)
        results[role] = {
            "path": relative_project_path(context.project_root, target),
            "sha256": sha256_file(target),
            "cache_sha256": checked["cache_sha256"],
            "parent_count": len(parents),
            "label_counts": dict(Counter(labels)),
            "quality_eligible_count": int(quality.sum().item()),
            "fit_eligible_count": int(fit_eligible.sum().item()),
            "reliable_final_count": int(reliable_final.sum().item()),
            "rd_present_without_reliable_event_count": int(
                rd_present.logical_and(~reliable).sum().item()
            ),
        }
    manifest = {
        "schema_version": R6_FEATURE_MANIFEST_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "r6_manifest_sha256": context.manifest_sha256,
        "system_manifest_sha256": context.config.base.expected_v27_system_manifest_sha256,
        "pool_registry_sha256": context.config.base.expected_v27_open_set_pool_registry_sha256,
        "representation_cache_manifest_sha256": context.config.base.expected_v27_representation_cache_manifest_sha256,
        "roles": results,
        "stable8_reliability": "frame_feature_reliable_not_rd_present",
        "test_payload_access_count": 0,
    }
    manifest["feature_manifest_sha256"] = canonical_sha256(manifest)
    path = r6_artifact_path(context.project_root, context.config.artifacts.feature_manifest_path)
    write_sealed_json_if_identical(path, manifest)
    return {**manifest, "manifest_path": str(path)}


def load_feature_cache(context: R6Context, *, role: str) -> dict[str, Any]:
    if role not in R6_ROLES:
        raise ValueError("V27b-r6 role is invalid")
    payload = torch.load(_cache_path(context, role=role), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27b-r6 feature cache is not an object")
    return _validate_feature_cache(payload, context, role=role)


def load_feature_manifest(context: R6Context) -> dict[str, Any]:
    from .protocol import read_sealed_json

    path = r6_artifact_path(context.project_root, context.config.artifacts.feature_manifest_path)
    payload = read_sealed_json(path, schema=R6_FEATURE_MANIFEST_SCHEMA_VERSION)
    body = dict(payload)
    sealed = body.pop("payload_sha256")
    declared = body.pop("feature_manifest_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError("V27b-r6 feature manifest hash differs")
    if (
        body.get("r6_manifest_sha256") != context.manifest_sha256
        or body.get("system_manifest_sha256") != context.config.base.expected_v27_system_manifest_sha256
        or body.get("pool_registry_sha256") != context.config.base.expected_v27_open_set_pool_registry_sha256
        or set(body.get("roles") or {}) != set(R6_ROLES)
    ):
        raise ValueError("V27b-r6 feature manifest binding differs")
    for role, record in dict(body["roles"]).items():
        cache = load_feature_cache(context, role=role)
        path_value = resolve_project_path(context.project_root, str(record.get("path", "")), must_exist=True)
        if (
            sha256_file(path_value) != record.get("sha256")
            or cache["cache_sha256"] != record.get("cache_sha256")
        ):
            raise ValueError("V27b-r6 feature manifest cache binding differs")
    return {**body, "feature_manifest_sha256": declared, "payload_sha256": sealed}


def stable8_stability_features(payload: Mapping[str, Any]) -> torch.Tensor:
    """Expose the deterministic treatment input for model and smoke reuse."""

    tensors = payload["tensors"]
    return _summary_stability(
        tensors["stable8_summary"],
        tensors["stable8_reliable"],
        tensors["stable8_reliable_count"],
        tensors["stable8_reliable_fraction"],
        tensors["padding_mask"],
    )


__all__ = [
    "feature_cache_digest",
    "load_feature_cache",
    "load_feature_manifest",
    "materialize_feature_caches",
    "stable8_stability_features",
]
