"""Read-only V18 evidence-cache audit and V27a normalization fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar_ln3.runtime import preflight_npz_archive, resolve_project_path, verify_file_hash

from .protocol import V27A_AUDIT_SCHEMA_VERSION, V27A_NORMALIZATION_SCHEMA_VERSION, V27A_STABLE8_INDICES, V27A_STABLE8_NAMES, V27AConfig


EVIDENCE_FIELDS = (
    "sample_ids",
    "valid",
    "event_offsets",
    "raw_dat_frame_indices_zero_based",
    "legacy_mat_list_ordinals_zero_based",
    "canonical_tensor_frame_indices_zero_based",
    "mat_acquisition_frame_numbers_one_based",
    "frame_values",
    "frame_feature_reliable",
    "residual_doppler_profiles",
    "envelope_modulation_profiles",
    "target_range_indices",
    "target_doppler_indices",
    "target_doppler_mps",
    "refined_body_doppler_mps",
    "velocity_nyquist_mps",
    "rayleigh_velocity_mps",
    "residual_sideband_snr_db",
    "envelope_modulation_snr_db",
    "frame_prt_num",
    "frame_prt_seconds",
    "frame_frequency_hz",
    "frame_cpi_count",
)
SUMMARY_FIELDS = ("sample_ids", "values", "valid")
_FRAME_LENGTH_FIELDS = {
    name
    for name in EVIDENCE_FIELDS
    if name
    not in {
        "sample_ids",
        "valid",
        "event_offsets",
    }
}
NORMALIZATION_SCALE_FLOOR = 1.0e-6


@dataclass(frozen=True)
class EvidenceSplit:
    split_name: str
    sample_ids: np.ndarray
    rd_present: np.ndarray
    event_offsets: np.ndarray
    stable8_values: np.ndarray
    frame_feature_reliable: np.ndarray
    recomputed_summary_values: np.ndarray
    recomputed_summary_valid: np.ndarray
    source_summary_values: np.ndarray
    source_summary_valid: np.ndarray
    parent_ids: np.ndarray
    labels: np.ndarray
    event_indices: np.ndarray
    audit: dict[str, Any]

    @property
    def event_count(self) -> int:
        return int(self.sample_ids.shape[0])

    @property
    def frame_count(self) -> int:
        return int(self.stable8_values.shape[0])


def _load_npz(path: Path, *, expected_sha256: str, fields: tuple[str, ...]) -> dict[str, np.ndarray]:
    verify_file_hash(path, expected_sha256, field=f"V27a {path.name}")
    preflight_npz_archive(path)
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != fields:
            raise ValueError(
                f"V27a cache fields differ for {path.name}: "
                f"expected={fields}, actual={tuple(archive.files)}"
            )
        return {name: np.asarray(archive[name]) for name in fields}


def _as_string_ids(values: np.ndarray, *, name: str) -> np.ndarray:
    if values.ndim != 1 or values.dtype.kind not in {"U", "S"}:
        raise ValueError(f"V27a {name} must be a one-dimensional string array")
    result = values.astype(str)
    if any(not value for value in result.tolist()):
        raise ValueError(f"V27a {name} contains an empty identifier")
    if len(set(result.tolist())) != len(result):
        raise ValueError(f"V27a {name} contains duplicate identifiers")
    return result


def _validate_index(root: Path, config: V27AConfig) -> pd.DataFrame:
    path = resolve_project_path(root, config.data.index_path, must_exist=True)
    verify_file_hash(path, config.data.expected_index_sha256, field="V27a index")
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "sample_id",
        "parent_sample_id",
        "track_id",
        "source_name",
        "split_name",
        "label_name",
        "causal_event_index",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V27a index lacks required columns: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("V27a index contains duplicate sample_id values")
    if frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("V27a evidence audit received a non-CQ08 row")
    return frame


def _event_layout(
    index: pd.DataFrame,
    *,
    split_name: str,
    sample_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align index metadata to the immutable evidence-cache event order.

    The processed index is a relational table and its physical row order is
    not part of the cache contract.  ``sample_ids`` is the authoritative event
    order for this audit; the index contributes only metadata after an explicit
    identifier join.  This distinction matters for Track 1010, whose rows are
    intentionally interleaved in the CSV while the evidence cache is grouped
    in causal parent order.
    """

    selected = index[
        index["split_name"].astype(str).eq(split_name)
        & index["label_name"].astype(str).isin(
            {"DroneTarget", "BirdTarget", "BalloonTarget"}
        )
    ].copy()
    selected["sample_id"] = selected["sample_id"].astype(str)
    selected["parent_sample_id"] = selected["parent_sample_id"].astype(str)
    numeric_indices = pd.to_numeric(
        selected["causal_event_index"], errors="raise"
    )
    if not np.isfinite(numeric_indices.to_numpy(dtype=np.float64)).all() or bool(
        (numeric_indices != numeric_indices.astype(np.int64)).any()
    ):
        raise ValueError(
            f"V27a index causal_event_index must contain integral finite values for {split_name}"
        )
    selected["causal_event_index"] = numeric_indices.astype(np.int64)
    if selected["sample_id"].duplicated().any():
        raise ValueError(f"V27a index contains duplicate known event ids for {split_name}")
    requested = sample_ids.astype(str).tolist()
    available = selected["sample_id"].tolist()
    missing = sorted(set(requested) - set(available))
    extra = sorted(set(available) - set(requested))
    if missing or extra:
        raise ValueError(
            f"V27a evidence/index event order differs for {split_name}: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    aligned = selected.set_index("sample_id", drop=False).loc[requested]
    if aligned.index.tolist() != requested:
        raise ValueError(f"V27a failed to align index metadata for {split_name}")
    parent_ids = aligned["parent_sample_id"].astype(str).to_numpy(dtype=str)
    labels = aligned["label_name"].astype(str).to_numpy(dtype=str)
    event_indices = aligned["causal_event_index"].to_numpy(dtype=np.int64)

    seen_parents: set[str] = set()
    previous_parent: str | None = None
    for raw_parent in parent_ids.tolist():
        parent_id = str(raw_parent)
        if not parent_id:
            raise ValueError(f"V27a {split_name} contains an empty parent id")
        if previous_parent is not None and parent_id != previous_parent:
            seen_parents.add(previous_parent)
        if parent_id in seen_parents:
            raise ValueError(
                f"V27a evidence events are not contiguous for parent {parent_id}"
            )
        previous_parent = parent_id
    for parent_id, group in aligned.groupby("parent_sample_id", sort=False):
        values = group["causal_event_index"].tolist()
        expected_indices = list(range(len(values)))
        if values != expected_indices:
            raise ValueError(
                f"V27a event indices are not contiguous for parent {parent_id}: "
                f"expected={expected_indices[:3]}..., actual={values[:3]}..."
            )
        if group["label_name"].astype(str).nunique() != 1:
            raise ValueError(f"V27a parent crosses labels: {parent_id}")
    return parent_ids, labels, event_indices


def _recompute_summaries(
    values: np.ndarray,
    reliable: np.ndarray,
    offsets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    event_count = int(len(offsets) - 1)
    summaries = np.zeros((event_count, values.shape[-1]), dtype=np.float32)
    valid = np.zeros((event_count,), dtype=np.bool_)
    for event_index, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        mask = reliable[int(start) : int(stop)]
        if bool(mask.any()):
            summaries[event_index] = np.median(
                values[int(start) : int(stop)][mask], axis=0
            ).astype(np.float32)
            valid[event_index] = True
    return summaries, valid


def _load_split(
    root: Path,
    config: V27AConfig,
    *,
    split_name: str,
    index: pd.DataFrame,
) -> EvidenceSplit:
    if split_name == "train":
        evidence_path = resolve_project_path(
            root, config.data.train_evidence_path, must_exist=True
        )
        summary_path = resolve_project_path(
            root, config.data.train_summary_path, must_exist=True
        )
        evidence_hash = config.data.expected_train_evidence_sha256
        summary_hash = config.data.expected_train_summary_sha256
    elif split_name == "val":
        evidence_path = resolve_project_path(
            root, config.data.validation_evidence_path, must_exist=True
        )
        summary_path = resolve_project_path(
            root, config.data.validation_summary_path, must_exist=True
        )
        evidence_hash = config.data.expected_validation_evidence_sha256
        summary_hash = config.data.expected_validation_summary_sha256
    else:
        raise ValueError(f"V27a unsupported evidence split: {split_name}")

    raw = _load_npz(evidence_path, expected_sha256=evidence_hash, fields=EVIDENCE_FIELDS)
    summary = _load_npz(summary_path, expected_sha256=summary_hash, fields=SUMMARY_FIELDS)
    sample_ids = _as_string_ids(raw["sample_ids"], name=f"{split_name} evidence sample_ids")
    summary_ids = _as_string_ids(
        summary["sample_ids"], name=f"{split_name} summary sample_ids"
    )
    if summary_ids.tolist() != sample_ids.tolist():
        raise ValueError(f"V27a summary/evidence sample order differs for {split_name}")
    event_count = len(sample_ids)
    offsets = np.asarray(raw["event_offsets"])
    if offsets.ndim != 1 or offsets.dtype.kind not in "iu" or len(offsets) != event_count + 1:
        raise ValueError(f"V27a {split_name} event_offsets shape/dtype differs")
    offsets = offsets.astype(np.int64, copy=False)
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != len(raw["frame_values"])
        or bool((np.diff(offsets) < 0).any())
    ):
        raise ValueError(f"V27a {split_name} event_offsets are not monotonic/bounded")
    frame_count = int(offsets[-1])
    if bool((np.diff(offsets) > config.data.max_frames_per_event).any()):
        raise ValueError(f"V27a {split_name} event exceeds max_frames_per_event")

    frame_values = np.asarray(raw["frame_values"])
    reliable = np.asarray(raw["frame_feature_reliable"])
    if frame_values.ndim != 2 or frame_values.shape[1] <= max(V27A_STABLE8_INDICES):
        raise ValueError(f"V27a {split_name} frame_values shape differs")
    if frame_values.dtype != np.float32 or not np.isfinite(frame_values).all():
        raise ValueError(f"V27a {split_name} frame_values must be finite float32")
    if reliable.shape != (frame_count,) or reliable.dtype != np.bool_:
        raise ValueError(f"V27a {split_name} frame_feature_reliable differs")
    for name in _FRAME_LENGTH_FIELDS:
        value = raw[name]
        if value.shape[0] != frame_count:
            raise ValueError(f"V27a {split_name} frame field length differs: {name}")
    stable8 = frame_values[:, V27A_STABLE8_INDICES].astype(np.float32, copy=True)
    if not np.isfinite(stable8).all():
        raise ValueError(f"V27a {split_name} stable8 values are non-finite")
    if np.any(stable8 < 0.0) or np.any(stable8 > 1.0):
        raise ValueError(f"V27a {split_name} stable8 values are outside [0,1]")
    rd_present = np.asarray(raw["valid"])
    if rd_present.shape != (event_count,) or rd_present.dtype != np.bool_:
        raise ValueError(f"V27a {split_name} event RD-valid field differs")
    if bool((reliable & ~np.repeat(rd_present, np.diff(offsets))).any()):
        raise ValueError(f"V27a {split_name} has reliable frames in an RD-missing event")
    source_values = np.asarray(summary["values"])
    source_valid = np.asarray(summary["valid"])
    if (
        source_values.shape != (event_count, len(V27A_STABLE8_INDICES))
        or source_values.dtype != np.float32
        or source_valid.shape != (event_count,)
        or source_valid.dtype != np.bool_
        or not np.isfinite(source_values).all()
        or np.any(source_values < 0.0)
        or np.any(source_values > 1.0)
    ):
        raise ValueError(f"V27a {split_name} stable8 summary shape/type/bounds differ")
    if np.any(source_values[~source_valid] != 0.0):
        raise ValueError(f"V27a {split_name} invalid summary values are not zero")
    recomputed_values, recomputed_valid = _recompute_summaries(
        stable8, reliable, offsets
    )
    reliable_rows = recomputed_valid
    max_difference = (
        float(np.max(np.abs(recomputed_values[reliable_rows] - source_values[reliable_rows])))
        if bool(reliable_rows.any())
        else 0.0
    )
    if max_difference > 1.0e-6:
        raise ValueError(
            f"V27a {split_name} recomputed stable8 summary differs: {max_difference}"
        )
    parent_ids, labels, event_indices = _event_layout(
        index, split_name=split_name, sample_ids=sample_ids
    )
    reliable_counts = np.asarray(
        [
            int(reliable[int(start) : int(stop)].sum())
            for start, stop in zip(offsets[:-1], offsets[1:])
        ],
        dtype=np.int64,
    )
    audit = {
        "split_name": split_name,
        "event_count": event_count,
        "frame_count": frame_count,
        "rd_present_event_count": int(rd_present.sum()),
        "rd_missing_event_count": int((~rd_present).sum()),
        "reliable_frame_count": int(reliable.sum()),
        "reliable_event_count": int(recomputed_valid.sum()),
        "no_reliable_event_count": int((~recomputed_valid).sum()),
        "rd_present_without_reliable_event_count": int(
            (rd_present & ~recomputed_valid).sum()
        ),
        "summary_source_valid_count": int(source_valid.sum()),
        "summary_valid_semantics_mismatch_count": int(
            np.count_nonzero(source_valid != recomputed_valid)
        ),
        "summary_recomputed_max_abs_difference": max_difference,
        "invalid_frame_stable8_nonzero_value_count": int(
            np.count_nonzero(stable8[~reliable])
        ),
        "invalid_frame_stable8_abs_max": float(
            np.abs(stable8[~reliable]).max() if bool((~reliable).any()) else 0.0
        ),
        "reliable_frame_count_distribution": {
            "min": int(reliable_counts.min()) if event_count else 0,
            "median": float(np.median(reliable_counts)) if event_count else 0.0,
            "p90": float(np.quantile(reliable_counts, 0.90)) if event_count else 0.0,
            "max": int(reliable_counts.max()) if event_count else 0,
        },
        "event_ids_order_sha256": _sequence_digest(sample_ids.tolist()),
    }
    return EvidenceSplit(
        split_name=split_name,
        sample_ids=sample_ids,
        rd_present=rd_present,
        event_offsets=offsets,
        stable8_values=stable8,
        frame_feature_reliable=reliable,
        recomputed_summary_values=recomputed_values,
        recomputed_summary_valid=recomputed_valid,
        source_summary_values=source_values,
        source_summary_valid=source_valid,
        parent_ids=parent_ids,
        labels=labels,
        event_indices=event_indices,
        audit=audit,
    )


def _sequence_digest(values: list[str]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _fit_normalization(train: EvidenceSplit, config: V27AConfig) -> dict[str, Any]:
    parent_medians: list[np.ndarray] = []
    parent_ids: list[str] = []
    for parent_id in sorted(set(train.parent_ids.tolist())):
        mask = train.parent_ids == parent_id
        rows: list[np.ndarray] = []
        for event_index in np.flatnonzero(mask):
            start = int(train.event_offsets[event_index])
            stop = int(train.event_offsets[event_index + 1])
            reliable = train.frame_feature_reliable[start:stop]
            if bool(reliable.any()):
                rows.append(train.stable8_values[start:stop][reliable])
        if rows:
            parent_medians.append(np.median(np.concatenate(rows, axis=0), axis=0))
            parent_ids.append(str(parent_id))
    if not parent_medians:
        raise ValueError("V27a normalization has no reliable train parent")
    matrix = np.asarray(parent_medians, dtype=np.float64)
    median = np.median(matrix, axis=0)
    q25 = np.quantile(matrix, 0.25, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    iqr = q75 - q25
    scale = np.maximum(iqr, NORMALIZATION_SCALE_FLOOR)
    return {
        "schema_version": V27A_NORMALIZATION_SCHEMA_VERSION,
        "experiment_id": "V27a",
        "fit_split": "train",
        "fit_scope": "known_dbB_train_parents_with_reliable_frames",
        "feature_names": list(V27A_STABLE8_NAMES),
        "feature_indices_in_evidence_cache": list(V27A_STABLE8_INDICES),
        "parent_equal_weighting": True,
        "parent_count_total": int(len(set(train.parent_ids.tolist()))),
        "parent_count_used": int(len(parent_medians)),
        "parent_ids_sha256": _sequence_digest(parent_ids),
        "reliable_frame_count": int(train.frame_feature_reliable.sum()),
        "median": median.astype(np.float64).tolist(),
        "q25": q25.astype(np.float64).tolist(),
        "q75": q75.astype(np.float64).tolist(),
        "iqr": iqr.astype(np.float64).tolist(),
        "scale_floor": NORMALIZATION_SCALE_FLOOR,
        "scale": scale.astype(np.float64).tolist(),
        "source_evidence_sha256": config.data.expected_train_evidence_sha256,
        "source_index_sha256": config.data.expected_index_sha256,
        "validation_not_used": True,
    }


def prepare_evidence_inputs(
    root: Path,
    *,
    config: V27AConfig,
    v27_config: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Audit both known DBB evidence caches and fit train-only normalization."""

    del v27_config  # kept in the interface to make the base boundary explicit
    index = _validate_index(root, config)
    train = _load_split(root, config, split_name="train", index=index)
    validation = _load_split(root, config, split_name="val", index=index)
    train_events = set(train.sample_ids.tolist())
    validation_events = set(validation.sample_ids.tolist())
    overlap = sorted(train_events & validation_events)
    if overlap:
        raise ValueError(
            "V27a train and validation evidence event sets overlap: "
            f"{overlap[:3]}"
        )
    train_parents = set(train.parent_ids.tolist())
    validation_parents = set(validation.parent_ids.tolist())
    parent_overlap = sorted(train_parents & validation_parents)
    if parent_overlap:
        raise ValueError(
            "V27a train and validation parent sets overlap: "
            f"{parent_overlap[:3]}"
        )
    normalization = _fit_normalization(train, config)
    audit = {
        "schema_version": V27A_AUDIT_SCHEMA_VERSION,
        "experiment_id": "V27a",
        "status": "audited",
        "source": {
            "index_path": config.data.index_path,
            "index_sha256": config.data.expected_index_sha256,
            "train_evidence_path": config.data.train_evidence_path,
            "train_evidence_sha256": config.data.expected_train_evidence_sha256,
            "validation_evidence_path": config.data.validation_evidence_path,
            "validation_evidence_sha256": config.data.expected_validation_evidence_sha256,
            "train_summary_path": config.data.train_summary_path,
            "train_summary_sha256": config.data.expected_train_summary_sha256,
            "validation_summary_path": config.data.validation_summary_path,
            "validation_summary_sha256": config.data.expected_validation_summary_sha256,
        },
        "stable8_contract": {
            "indices": list(V27A_STABLE8_INDICES),
            "names": list(V27A_STABLE8_NAMES),
            "invalid_value_policy": "zero_for_model_input_but_original_nonzero_invalid_cells_are_reported",
            "reliable_field": "frame_feature_reliable",
            "rd_present_field": "valid",
        },
        "splits": {"train": train.audit, "val": validation.audit},
        "parent_counts": {
            "train": int(len(set(train.parent_ids.tolist()))),
            "val": int(len(set(validation.parent_ids.tolist()))),
        },
        "normalization_fit": {
            "schema_version": normalization["schema_version"],
            "parent_count_used": normalization["parent_count_used"],
            "validation_not_used": True,
        },
        "provenance_fields_excluded_from_model": [
            "raw_dat_frame_indices_zero_based",
            "legacy_mat_list_ordinals_zero_based",
            "canonical_tensor_frame_indices_zero_based",
            "mat_acquisition_frame_numbers_one_based",
            "sample_ids",
            "parent_sample_id",
            "track_id",
            "source_name",
            "sampling_mode",
        ],
        "future_information_policy": "only_current_and_prior_frames/events_are_consumed",
    }
    return audit, normalization


def load_normalization(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    from .protocol import read_sealed_json

    payload = read_sealed_json(path, schema=V27A_NORMALIZATION_SCHEMA_VERSION)
    median = np.asarray(payload["median"], dtype=np.float32)
    scale = np.asarray(payload["scale"], dtype=np.float32)
    if median.shape != (8,) or scale.shape != (8,):
        raise ValueError("V27a normalization vectors must have shape [8]")
    if not np.isfinite(median).all() or not np.isfinite(scale).all() or np.any(scale <= 0.0):
        raise ValueError("V27a normalization vectors must be finite and positive")
    return median, scale, payload


def normalize_stable8(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    if values.shape[-1] != 8 or median.shape != (8,) or scale.shape != (8,):
        raise ValueError("V27a stable8 normalization shapes differ")
    if valid.shape != values.shape[:-1] or valid.dtype != np.bool_:
        raise ValueError("V27a stable8 normalization mask differs")
    result = (values.astype(np.float32) - median) / scale
    result = np.where(valid[..., None], result, 0.0).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("V27a normalized stable8 contains non-finite values")
    return result


__all__ = ["EVIDENCE_FIELDS", "EvidenceSplit", "NORMALIZATION_SCALE_FLOOR", "SUMMARY_FIELDS", "load_normalization", "normalize_stable8", "prepare_evidence_inputs"]