"""Prepare V26c's strict Known-only L1/L2 provenance-grouped OOF protocol.

The only reused V21e artifact is the already-sealed provenance assignment.
Every V26c fold receives fresh L1/L2 initialization, fold-train-only RD
normalization, fresh label-free source-derived feature extraction, and a fixed
epoch snapshot. Unknown parents never enter model training, normalization,
cache selection, sampling, or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from build.known_gate_v21e import _build_split as _build_course_feature_split
from build.micro_doppler_v17 import (
    _CanonicalFrameProvenance,
    _write_npz_new_or_identical,
)
from build.micro_doppler_v18 import _build_split as _build_v18_feature_split
from build.radar_ln3_normalization import build as build_normalization
from radar_ln3.known_gate import KNOWN_LABELS, load_known_gate_config
from radar_ln3.known_gate_v26c_oof import (
    CONTRACT,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    L1_CHECKPOINT_SCHEMA,
    L1_EPOCH,
    L2_CHECKPOINT_SCHEMA,
    L2_EPOCH,
    READINESS,
    READINESS_SCHEMA,
    SOURCE_FEATURE_MANIFEST,
    TRAINING_SEED_L1,
    TRAINING_SEED_L2,
    V21E_ASSIGNMENT,
    V21E_FOLD_CONTRACT,
    fold_record,
    l1_checkpoint_path,
    l1_config_path,
    l1_data_contract_path,
    l1_held_out_feature_cache_path,
    l1_last_checkpoint_path,
    l1_normalization_path,
    l2_checkpoint_path,
    l2_config_path,
    l2_data_contract_path,
    l2_last_checkpoint_path,
    l2_held_out_micro_cache_path,
    l2_held_out_mixed_micro_cache_path,
    l2_held_out_mixed_physical_cache_path,
    l2_normalization_path,
    l2_held_out_physical_cache_path,
    relative_path,
    sequence_sha256,
    sha256_file,
    l1_train_feature_cache_path,
    l1_source_feature_cache_path,
    l2_train_micro_cache_path,
    l2_train_physical_cache_path,
    l2_source_micro_cache_path,
    l2_source_physical_cache_path,
    l2_source_mixed_micro_cache_path,
    l2_source_mixed_physical_cache_path,
    write_new_or_identical,
)
from radar_ln3.task_models import load_task_config


INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
L1_BASE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43-config.json"
)
L1_BASE_EXPERIMENT_CONTRACT = Path(
    "data/processed/RADAR-LN3-V26-cq08-known-gate-known-only-open-set-baseline-contract.json"
)
L2_BASE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V18r-cq08-liquid-dbb-specialist-rd-physical4-micro-doppler8-"
    "stable-density-treatment-seed42-config.json"
)
L2_BASE_CONTRACT = Path(
    "data/processed/RADAR-LN3-V18-cq08-dbb-broadband-multiscale-micro-doppler8-"
    "stable-data-contract.json"
)
SOURCE_PROVENANCE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
SOURCE_CANONICAL_FRAME_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)
DOCUMENT = Path("src-v3/docs/radar_ln3_v26c_l1_l2_oof_l3_calibration.md")

L1_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
L2_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
EXPECTED_TRAIN_COUNTS = {
    "DroneTarget": 476,
    "BirdTarget": 239,
    "BalloonTarget": 238,
    "ClutterTarget": 33,
}
L1_CACHE_MAX_BYTES = 64 * 1024 * 1024
L2_CACHE_MAX_BYTES = 64 * 1024 * 1024


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _validate_sources(root: Path) -> dict[str, Any]:
    paths = {
        "index": root / INDEX,
        "v21e_assignment": root / V21E_ASSIGNMENT,
        "v21e_fold_contract": root / V21E_FOLD_CONTRACT,
        "l1_base_config": root / L1_BASE_CONFIG,
        "l1_base_experiment": root / L1_BASE_EXPERIMENT_CONTRACT,
        "l2_base_config": root / L2_BASE_CONFIG,
        "l2_base_contract": root / L2_BASE_CONTRACT,
        "source_provenance": root / SOURCE_PROVENANCE,
        "source_canonical_frame_index": root / SOURCE_CANONICAL_FRAME_INDEX,
        "document": root / DOCUMENT,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("V26c source files are missing: " + ", ".join(missing))
    l1 = load_known_gate_config(paths["l1_base_config"])
    l2 = load_task_config(paths["l2_base_config"])
    assignment = _read_json(paths["v21e_assignment"])
    fold_contract = _read_json(paths["v21e_fold_contract"])
    provenance_payload = _read_json(paths["source_provenance"])
    provenance = {
        str(row["parent_sample_id"]): dict(row)
        for row in list(provenance_payload.get("records") or ())
    }
    if (
        l1.variant != "v26_known_only_open_set_baseline"
        or l1.train.seed != TRAINING_SEED_L1
        or l2.schema_version != "RADAR-LN3-V18r"
        or l2.task.name != "dbb_specialist"
        or l2.model.rd_physical_dim != 4
        or l2.model.micro_doppler_dim != 8
        or l2.model.liquid_invalid_dt_update_mode != "density_weighted"
        or l2.train.seed != TRAINING_SEED_L2
        or assignment.get("schema_version")
        != "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-assignment-v1"
        or fold_contract.get("schema_version")
        != "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-contract-v1"
        or fold_contract.get("assignment_sha256") != sha256_file(paths["v21e_assignment"])
        or int(fold_contract.get("fold_count", -1)) != FOLD_COUNT
        or int(fold_contract.get("known_parent_count", -1)) != 986
        or fold_contract.get("grouping") != "capture_content_provenance_component"
        or int(fold_contract.get("unknown_training_exposure", -1)) != 0
        or provenance_payload.get("schema_version") != "RADAR-LN3-V16"
        or len(provenance) != int(provenance_payload.get("parent_count", -1))
        or int(provenance_payload.get("cross_label_content_conflict_count", -1)) != 0
    ):
        raise ValueError("V26c frozen source protocol is invalid")
    return {
        "paths": paths,
        "hashes": {name: sha256_file(path) for name, path in paths.items()},
        "l1": l1,
        "l2": l2,
        "assignment": assignment,
        "fold_contract": fold_contract,
        "provenance": provenance,
        "canonical_frame_provenance": _CanonicalFrameProvenance(
            root, paths["source_canonical_frame_index"]
        ),
    }


def _index_parent_labels(index_path: Path) -> dict[str, str]:
    frame = pd.read_csv(
        index_path,
        usecols=["schema_version", "parent_sample_id", "split_name", "label_name"],
        low_memory=False,
    )
    if set(frame["schema_version"].astype(str)) != {"RADAR-LN3-V16"}:
        raise ValueError("V26c index schema changed")
    train = frame[
        frame["split_name"].astype(str).eq("train")
        & frame["label_name"].astype(str).isin(L1_LABELS)
    ].copy()
    grouped = train.groupby("parent_sample_id", sort=False)["label_name"].nunique()
    if bool(grouped.ne(1).any()):
        raise ValueError("V26c index parent labels are not unique")
    labels = (
        train[["parent_sample_id", "label_name"]]
        .drop_duplicates("parent_sample_id")
        .assign(
            parent_sample_id=lambda value: value["parent_sample_id"].astype(str),
            label_name=lambda value: value["label_name"].astype(str),
        )
    )
    output = dict(zip(labels["parent_sample_id"], labels["label_name"], strict=True))
    counts = Counter(output.values())
    if len(output) != 986 or dict(counts) != EXPECTED_TRAIN_COUNTS:
        raise ValueError("V26c known train parent support changed")
    return output


def _selected_fold_events(
    index_path: Path,
    parent_ids: Sequence[str],
    labels: Sequence[str],
    *,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return one ordered train-fold event population with no label-derived value."""

    base_columns = {
        "parent_sample_id",
        "split_name",
        "label_name",
        "sample_id",
        "causal_event_index",
    }
    usecols = None if required_columns is None else sorted(
        base_columns | {str(value) for value in required_columns}
    )
    frame = pd.read_csv(
        index_path,
        usecols=usecols,
        low_memory=False,
    )
    requested = set(str(value) for value in parent_ids)
    selected = frame[
        frame["split_name"].astype(str).eq("train")
        & frame["parent_sample_id"].astype(str).isin(requested)
        & frame["label_name"].astype(str).isin(tuple(labels))
    ].copy()
    selected = selected.sort_values(["parent_sample_id", "causal_event_index"], kind="stable")
    actual_parents = sorted(selected["parent_sample_id"].astype(str).unique().tolist())
    if actual_parents != sorted(requested):
        raise ValueError("V26c selected event rows do not cover every requested parent")
    values = selected["sample_id"].astype(str).tolist()
    if not values or len(values) != len(set(values)):
        raise ValueError("V26c selected event rows are empty or duplicate")
    return selected.reset_index(drop=True)


def _cache_metadata(
    root: Path,
    output_path: Path,
    *,
    sample_ids: Sequence[str],
    values: np.ndarray,
    valid: np.ndarray,
    source_index: Path,
    source_description: str,
    size_limit: int,
) -> dict[str, Any]:
    values = np.asarray(values)
    valid = np.asarray(valid)
    identifiers = [str(value) for value in sample_ids]
    if (
        len(identifiers) != len(set(identifiers))
        or values.dtype != np.float32
        or valid.dtype != np.bool_
        or values.ndim != 2
        or values.shape[0] != len(identifiers)
        or valid.shape[0] != len(identifiers)
        or not np.isfinite(values).all()
    ):
        raise ValueError("V26c recomputed feature cache arrays are invalid")
    if valid.ndim == 2:
        if valid.shape != values.shape or np.any(values[~valid] != 0.0):
            raise ValueError("V26c vector cache has invalid value/mask semantics")
        valid_event_count = int(valid.all(axis=1).sum())
    elif valid.ndim == 1:
        if valid.shape != (values.shape[0],) or np.any(values[~valid] != 0.0):
            raise ValueError("V26c event cache has invalid value/mask semantics")
        valid_event_count = int(valid.sum())
    else:
        raise ValueError("V26c feature cache valid rank is invalid")
    _write_npz_new_or_identical(
        output_path,
        (
            ("sample_ids", np.asarray(identifiers, dtype=np.str_)),
            ("values", values),
            ("valid", valid),
        ),
    )
    if output_path.stat().st_size > size_limit:
        raise MemoryError(f"V26c fold cache exceeds bounded size: {output_path}")
    return {
        "path": relative_path(root, output_path),
        "sha256": sha256_file(output_path),
        "event_count": len(identifiers),
        "feature_width": int(values.shape[1]),
        "valid_event_count": valid_event_count,
        "invalid_event_count": int(len(identifiers) - valid_event_count),
        "selected_sample_ids_sha256": sequence_sha256(identifiers),
        "source_index": relative_path(root, source_index),
        "source_index_sha256": sha256_file(source_index),
        "cache_value_provenance": source_description,
        "reused_prior_feature_cache_values": False,
        "label_free_feature_value_extraction": True,
    }


def _read_recomputed_cache(
    root: Path,
    path: Path,
    *,
    width: int,
    valid_rank: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V26c source rebuild cache fields differ: {path}")
        sample_ids = [str(value) for value in np.asarray(archive["sample_ids"]).tolist()]
        values = np.asarray(archive["values"])
        valid = np.asarray(archive["valid"])
    if (
        len(sample_ids) != len(set(sample_ids))
        or values.dtype != np.float32
        or values.shape != (len(sample_ids), int(width))
        or valid.dtype != np.bool_
        or valid.ndim != int(valid_rank)
        or valid.shape[0] != len(sample_ids)
    ):
        raise ValueError(f"V26c source rebuild cache layout differs: {path}")
    if valid_rank == 2 and valid.shape != values.shape:
        raise ValueError("V26c course source rebuild validity shape differs")
    if valid_rank == 1 and valid.shape != (len(sample_ids),):
        raise ValueError("V26c L2 source rebuild validity shape differs")
    if not np.isfinite(values).all() or np.any(values[~valid] != 0.0):
        raise ValueError("V26c source rebuild cache values are invalid")
    return sample_ids, values, valid


def _slice_recomputed_cache(
    root: Path,
    *,
    source_path: Path,
    output_path: Path,
    selected: pd.DataFrame,
    width: int,
    valid_rank: int,
    source_description: str,
    size_limit: int,
) -> dict[str, Any]:
    source_ids, source_values, source_valid = _read_recomputed_cache(
        root, source_path, width=width, valid_rank=valid_rank
    )
    requested = selected["sample_id"].astype(str).tolist()
    index = {sample_id: position for position, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in requested if sample_id not in index]
    if missing:
        raise ValueError(f"V26c source rebuild cache lacks fold event: {missing[:3]}")
    positions = np.asarray([index[sample_id] for sample_id in requested], dtype=np.int64)
    result = _cache_metadata(
        root,
        output_path,
        sample_ids=requested,
        values=source_values[positions].copy(),
        valid=source_valid[positions].copy(),
        source_index=Path(root / INDEX),
        source_description=source_description,
        size_limit=size_limit,
    )
    result.update(
        {
            "source_rebuild_cache": relative_path(root, source_path),
            "source_rebuild_cache_sha256": sha256_file(source_path),
            "source_rebuild_cache_value_reuse": True,
            "source_rebuild_cache_is_label_free_over_all_known_train_parents": True,
        }
    )
    return result


def _rebuild_course_cache(
    root: Path,
    *,
    index_path: Path,
    parent_ids: Sequence[str],
    output_path: Path,
) -> dict[str, Any]:
    selected = _selected_fold_events(
        index_path,
        parent_ids,
        L1_LABELS,
        required_columns=("track_id", "tr_feature_values_json"),
    )
    arrays, statistics, parents = _build_course_feature_split(selected, split="train")
    sample_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    expected = selected["sample_id"].astype(str).tolist()
    if (
        sample_ids != expected
        or arrays["values"].shape != (len(expected), 1)
        or arrays["valid"].shape != arrays["values"].shape
        or int(statistics.get("parent_count", -1)) != len(parent_ids)
        or len(parents) != len(parent_ids)
    ):
        raise ValueError("V26c recomputed course cache differs from the fold population")
    result = _cache_metadata(
        root,
        output_path,
        sample_ids=sample_ids,
        values=np.asarray(arrays["values"], dtype=np.float32),
        valid=np.asarray(arrays["valid"], dtype=np.bool_),
        source_index=index_path,
        source_description=(
            "recomputed causally from V16 TR course payloads; no V21e feature-cache values reused"
        ),
        size_limit=L1_CACHE_MAX_BYTES,
    )
    result["recomputation_statistics"] = {
        key: value for key, value in statistics.items() if key != "test_payload_access_count"
    }
    return result


def _rebuild_l2_caches(
    root: Path,
    *,
    source: Mapping[str, Any],
    parent_ids: Sequence[str],
    physical_output_path: Path,
    micro_output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_path = Path(source["paths"]["index"])
    selected = _selected_fold_events(index_path, parent_ids, L2_LABELS)
    arrays, statistics = _build_v18_feature_split(
        root,
        "train",
        selected,
        source["provenance"],
        source["canonical_frame_provenance"],
        allowed_labels=L2_LABELS,
        include_evidence=False,
    )
    sample_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    expected = selected["sample_id"].astype(str).tolist()
    physical_values = np.asarray(arrays["physical_values"], dtype=np.float32)
    micro_values = np.asarray(arrays["summary_values"], dtype=np.float32)
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    if (
        sample_ids != expected
        or physical_values.shape != (len(expected), 4)
        or micro_values.shape != (len(expected), 8)
        or valid.shape != (len(expected),)
    ):
        raise ValueError("V26c recomputed stable8 cache differs from the fold population")
    access = dict(statistics.get("stable8_prepare_access_counts") or {})
    shared_statistics = {
        "raw_file_count": int(access.get("raw_file_count", -1)),
        "rd_tensor_count": int(access.get("rd_tensor_count", -1)),
        "complete_evidence_profiles_materialized": False,
    }
    if min(shared_statistics["raw_file_count"], shared_statistics["rd_tensor_count"]) < 0:
        raise ValueError("V26c stable8 recomputation statistics are incomplete")
    physical = _cache_metadata(
        root,
        physical_output_path,
        sample_ids=sample_ids,
        values=physical_values,
        valid=valid,
        source_index=index_path,
        source_description=(
            "recomputed V18 current-event RD physical4 values from V16 source files; no V18 cache values reused"
        ),
        size_limit=L2_CACHE_MAX_BYTES,
    )
    micro = _cache_metadata(
        root,
        micro_output_path,
        sample_ids=sample_ids,
        values=micro_values,
        valid=valid,
        source_index=index_path,
        source_description=(
            "recomputed V18 current-event stable8 micro-Doppler values from V16 source files; no V18 cache values reused"
        ),
        size_limit=L2_CACHE_MAX_BYTES,
    )
    physical["recomputation_statistics"] = dict(shared_statistics)
    micro["recomputation_statistics"] = dict(shared_statistics)
    return physical, micro


def _rebuild_l2_mixed_held_out_caches(
    root: Path,
    *,
    source: Mapping[str, Any],
    parent_ids: Sequence[str],
    physical_output_path: Path,
    micro_output_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare L2 inference-only values for the full L1 held-out population.

    The extraction is label-free after the L1 held-out parent set is chosen.
    Clutter values are never passed to the L2 optimizer, its normalization or
    L3 threshold fitting; this cache merely makes the real L1->L2 call path
    auditable when L1's target acceptance is evaluated on a Clutter parent.
    """

    index_path = Path(source["paths"]["index"])
    selected = _selected_fold_events(index_path, parent_ids, L1_LABELS)
    arrays, statistics = _build_v18_feature_split(
        root,
        "train",
        selected,
        source["provenance"],
        source["canonical_frame_provenance"],
        allowed_labels=L1_LABELS,
        include_evidence=False,
    )
    sample_ids = [str(value) for value in arrays["sample_ids"].tolist()]
    expected = selected["sample_id"].astype(str).tolist()
    physical_values = np.asarray(arrays["physical_values"], dtype=np.float32)
    micro_values = np.asarray(arrays["summary_values"], dtype=np.float32)
    valid = np.asarray(arrays["valid"], dtype=np.bool_)
    if (
        sample_ids != expected
        or physical_values.shape != (len(expected), 4)
        or micro_values.shape != (len(expected), 8)
        or valid.shape != (len(expected),)
    ):
        raise ValueError("V26c inference-only mixed L2 cache differs from the held-out population")
    access = dict(statistics.get("stable8_prepare_access_counts") or {})
    physical = _cache_metadata(
        root,
        physical_output_path,
        sample_ids=sample_ids,
        values=physical_values,
        valid=valid,
        source_index=index_path,
        source_description=(
            "recomputed label-free V18 RD physical4 inference cache for all L1 held-out Known parents; never used by L2 training or normalization"
        ),
        size_limit=L2_CACHE_MAX_BYTES,
    )
    micro = _cache_metadata(
        root,
        micro_output_path,
        sample_ids=sample_ids,
        values=micro_values,
        valid=valid,
        source_index=index_path,
        source_description=(
            "recomputed label-free V18 stable8 inference cache for all L1 held-out Known parents; never used by L2 training or normalization"
        ),
        size_limit=L2_CACHE_MAX_BYTES,
    )
    for cache in (physical, micro):
        cache.update(
            {
                "inference_only_mixed_l1_held_out_population": True,
                "l2_training_exposure": 0,
                "l2_normalization_exposure": 0,
                "l3_threshold_fit_exposure": 0,
                "raw_file_count": int(access.get("raw_file_count", -1)),
                "rd_tensor_count": int(access.get("rd_tensor_count", -1)),
            }
        )
    return physical, micro


def _prepare_label_free_source_rebuilds(
    root: Path,
    *,
    source: Mapping[str, Any],
    labels_by_parent: Mapping[str, str],
) -> dict[str, Any]:
    """Build each deterministic V26c source feature value once.

    Course, physical4 and stable8 extraction uses no fitted statistic and no
    parent label value after the immutable Known/DBB scopes are selected. The
    expensive raw-IQ extraction can consequently run once over the complete
    Known training universe; each fold later receives an exact disjoint slice.
    This is materially different from reusing V21e/V18 cache values.
    """

    all_known_ids = sorted(labels_by_parent)
    all_dbb_ids = sorted(
        parent_id for parent_id, label in labels_by_parent.items() if label in L2_LABELS
    )
    course_path = root / l1_source_feature_cache_path()
    physical_path = root / l2_source_physical_cache_path()
    micro_path = root / l2_source_micro_cache_path()
    mixed_physical_path = root / l2_source_mixed_physical_cache_path()
    mixed_micro_path = root / l2_source_mixed_micro_cache_path()
    manifest_path = root / SOURCE_FEATURE_MANIFEST
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        expected = {
            "schema_version": "RADAR-LN3-V26c-label-free-source-feature-rebuild-manifest-v1",
            "source_index": relative_path(root, source["paths"]["index"]),
            "source_index_sha256": sha256_file(source["paths"]["index"]),
            "known_train_parent_count": len(all_known_ids),
            "dbb_train_parent_count": len(all_dbb_ids),
            "known_train_parent_ids_sha256": sequence_sha256(all_known_ids),
            "dbb_train_parent_ids_sha256": sequence_sha256(all_dbb_ids),
            "unknown_training_exposure": 0,
            "validation_access_count": 0,
            "test_access_count": 0,
            "feature_value_label_free_after_scope_selection": True,
            "previous_v21e_or_v18_cache_values_reused": False,
        }
        if any(existing.get(key) != value for key, value in expected.items()):
            raise ValueError("V26c existing source feature rebuild manifest differs from its sealed protocol")
        cache_specs = (
            ("course", course_path, 1, 2),
            ("rd_physical4", physical_path, 4, 1),
            ("micro_doppler8", micro_path, 8, 1),
            ("mixed_l1_known_inference_only_rd_physical4", mixed_physical_path, 4, 1),
            ("mixed_l1_known_inference_only_micro_doppler8", mixed_micro_path, 8, 1),
        )
        for name, path, width, valid_rank in cache_specs:
            metadata = dict(existing.get(name) or {})
            identifiers, _, _ = _read_recomputed_cache(
                root, path, width=width, valid_rank=valid_rank
            )
            expected_ids = (
                _selected_fold_events(source["paths"]["index"], all_known_ids, L1_LABELS)["sample_id"].astype(str).tolist()
                if name.startswith("course") or name.startswith("mixed_")
                else _selected_fold_events(source["paths"]["index"], all_dbb_ids, L2_LABELS)["sample_id"].astype(str).tolist()
            )
            if (
                identifiers != expected_ids
                or metadata.get("path") != relative_path(root, path)
                or metadata.get("sha256") != sha256_file(path)
                or int(metadata.get("event_count", -1)) != len(expected_ids)
                or metadata.get("selected_sample_ids_sha256") != sequence_sha256(expected_ids)
                or metadata.get("reused_prior_feature_cache_values") is not False
            ):
                raise ValueError(f"V26c existing {name} source rebuild cache differs from its manifest")
        return existing
    course = _rebuild_course_cache(
        root,
        index_path=source["paths"]["index"],
        parent_ids=all_known_ids,
        output_path=course_path,
    )
    physical, micro = _rebuild_l2_caches(
        root,
        source=source,
        parent_ids=all_dbb_ids,
        physical_output_path=physical_path,
        micro_output_path=micro_path,
    )
    mixed_physical, mixed_micro = _rebuild_l2_mixed_held_out_caches(
        root,
        source=source,
        parent_ids=all_known_ids,
        physical_output_path=mixed_physical_path,
        micro_output_path=mixed_micro_path,
    )
    manifest = {
        "schema_version": "RADAR-LN3-V26c-label-free-source-feature-rebuild-manifest-v1",
        "purpose": "one-time deterministic source rebuild before V26c disjoint fold cache slicing",
        "source_index": relative_path(root, source["paths"]["index"]),
        "source_index_sha256": sha256_file(source["paths"]["index"]),
        "known_train_parent_count": len(all_known_ids),
        "dbb_train_parent_count": len(all_dbb_ids),
        "known_train_parent_ids_sha256": sequence_sha256(all_known_ids),
        "dbb_train_parent_ids_sha256": sequence_sha256(all_dbb_ids),
        "unknown_training_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "feature_value_label_free_after_scope_selection": True,
        "previous_v21e_or_v18_cache_values_reused": False,
        "course": course,
        "rd_physical4": physical,
        "micro_doppler8": micro,
        "mixed_l1_known_inference_only_rd_physical4": mixed_physical,
        "mixed_l1_known_inference_only_micro_doppler8": mixed_micro,
    }
    write_new_or_identical(root / SOURCE_FEATURE_MANIFEST, manifest)
    return manifest


def _fold_contract(
    *,
    task: str,
    fold_id: int,
    train_ids: Sequence[str],
    held_out_ids: Sequence[str],
    train_counts: Mapping[str, int],
    held_out_counts: Mapping[str, int],
    normalization: Mapping[str, Any],
    train_caches: Mapping[str, Any],
    held_out_caches: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": f"RADAR-LN3-V26c-{task}-fold-data-contract-v1",
        "task_name": task,
        "fold_id": int(fold_id),
        "fit_split": "train",
        "fit_population": "known_fold_train_only",
        "grouping": "capture_content_provenance_component",
        "train_parent_count": len(train_ids),
        "held_out_parent_count": len(held_out_ids),
        "train_parent_ids_sha256": sequence_sha256(sorted(str(value) for value in train_ids)),
        "held_out_parent_ids_sha256": sequence_sha256(sorted(str(value) for value in held_out_ids)),
        "train_label_parent_counts": dict(train_counts),
        "held_out_label_parent_counts": dict(held_out_counts),
        "normalization": dict(normalization),
        "cache_reuse": {
            "train": dict(train_caches),
            "held_out": dict(held_out_caches),
            "isolation": "each cache contains exactly one disjoint fold population",
        },
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_cache_selection_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "checkpoint_selection": "fixed epoch snapshot; held-out parent labels are never used",
        "l1_l2_fusion": "not_performed_in_fold_training",
    }


def _l1_config(
    source: Mapping[str, Any],
    *,
    fold_id: int,
    contract_path: Path,
    normalization_path: Path,
    stats: Mapping[str, Any],
    train_course_cache: Mapping[str, Any],
    held_out_course_cache: Mapping[str, Any],
    train_counts: Mapping[str, int],
) -> dict[str, Any]:
    payload = json.loads((source["paths"]["l1_base_config"]).read_text(encoding="utf-8"))
    payload["schema_version"] = "RADAR-LN3-V26c-OOF-L1"
    payload["config_name"] = f"RADAR-LN3-V26c-cq08-known-gate-oof-fold-{fold_id:02d}-seed43"
    data = dict(payload["data"])
    data.update(
        {
            "contract_path": relative_path(source["root"], contract_path),
            "expected_contract_sha256": sha256_file(contract_path),
            "normalization_stats_path": relative_path(source["root"], normalization_path),
            "expected_stats_sha256": sha256_file(normalization_path),
            "rd_amplitude_center": float(dict(stats["amplitude"])["center"]),
            "rd_amplitude_scale": float(dict(stats["amplitude"])["scale"]),
            "train_feature_cache_path": str(train_course_cache["path"]),
            "validation_feature_cache_path": str(held_out_course_cache["path"]),
            "expected_train_feature_cache_sha256": str(train_course_cache["sha256"]),
            "expected_validation_feature_cache_sha256": str(held_out_course_cache["sha256"]),
        }
    )
    # Fold-specific config is intentionally not routed through the frozen V26
    # loader; the OOF trainer validates this local contract directly.
    payload["data"] = data
    artifact = source["root"] / l1_checkpoint_path(fold_id).parent
    train = dict(payload["train"])
    target_count = sum(int(train_counts[name]) for name in L2_LABELS)
    clutter_count = int(train_counts["ClutterTarget"])
    train.update(
        {
            "seed": TRAINING_SEED_L1,
            "dataloader_seed": TRAINING_SEED_L1,
            "epochs": 20,
            "early_stopping_patience": 0,
            "class_weights": [
                (target_count + clutter_count) / (2.0 * target_count),
                (target_count + clutter_count) / (2.0 * clutter_count),
            ],
            "checkpoint_path": relative_path(source["root"], source["root"] / l1_checkpoint_path(fold_id)),
            "last_checkpoint_path": relative_path(source["root"], source["root"] / l1_last_checkpoint_path(fold_id)),
            "formal_artifact_dir": relative_path(source["root"], artifact),
            "epoch_checkpoint_dir": relative_path(source["root"], artifact / "epochs"),
            "checkpoint_schema": L1_CHECKPOINT_SCHEMA,
            "save_every_epoch": False,
            "paired_baseline_config_path": str(source["l1"].train.paired_baseline_config_path),
        }
    )
    payload["train"] = train
    payload["v26c_oof"] = {
        "fold_id": fold_id,
        "role": "fresh L1 OOF support only; not a deployment checkpoint",
        "fixed_snapshot_epoch": L1_EPOCH,
        "unknown_training_exposure": 0,
        "held_out_checkpoint_selection": False,
    }
    return payload


def _l2_config(
    source: Mapping[str, Any],
    *,
    fold_id: int,
    contract_path: Path,
    normalization_path: Path,
    stats: Mapping[str, Any],
    train_physical_cache: Mapping[str, Any],
    held_out_physical_cache: Mapping[str, Any],
    train_micro_cache: Mapping[str, Any],
    held_out_micro_cache: Mapping[str, Any],
    train_counts: Mapping[str, int],
) -> dict[str, Any]:
    payload = json.loads((source["paths"]["l2_base_config"]).read_text(encoding="utf-8"))
    payload["schema_version"] = "RADAR-LN3-V26c-OOF-L2"
    payload["config_name"] = f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{fold_id:02d}-seed42"
    data = dict(payload["data"])
    data.update(
        {
            "contract_path": relative_path(source["root"], contract_path),
            "expected_contract_sha256": sha256_file(contract_path),
            "normalization_stats_path": relative_path(source["root"], normalization_path),
            "expected_stats_sha256": sha256_file(normalization_path),
            "rd_amplitude_center": float(dict(stats["amplitude"])["center"]),
            "rd_amplitude_scale": float(dict(stats["amplitude"])["scale"]),
            "train_rd_physical_cache_path": str(train_physical_cache["path"]),
            "validation_rd_physical_cache_path": str(held_out_physical_cache["path"]),
            "expected_train_rd_physical_cache_sha256": str(train_physical_cache["sha256"]),
            "expected_validation_rd_physical_cache_sha256": str(held_out_physical_cache["sha256"]),
            "train_micro_doppler_cache_path": str(train_micro_cache["path"]),
            "validation_micro_doppler_cache_path": str(held_out_micro_cache["path"]),
            "expected_train_micro_doppler_cache_sha256": str(train_micro_cache["sha256"]),
            "expected_validation_micro_doppler_cache_sha256": str(held_out_micro_cache["sha256"]),
        }
    )
    payload["data"] = data
    artifact = source["root"] / l2_checkpoint_path(fold_id).parent
    total = sum(int(train_counts[name]) for name in L2_LABELS)
    train = dict(payload["train"])
    train.update(
        {
            "seed": TRAINING_SEED_L2,
            "dataloader_seed": TRAINING_SEED_L2,
            "epochs": 20,
            "early_stopping_patience": 0,
            "class_weights": [total / (3.0 * int(train_counts[name])) for name in L2_LABELS],
            "checkpoint_path": relative_path(source["root"], source["root"] / l2_checkpoint_path(fold_id)),
            "last_checkpoint_path": relative_path(source["root"], source["root"] / l2_last_checkpoint_path(fold_id)),
            "formal_artifact_dir": relative_path(source["root"], artifact),
            "epoch_checkpoint_dir": relative_path(source["root"], artifact / "epochs"),
            "checkpoint_schema": L2_CHECKPOINT_SCHEMA,
            "save_every_epoch": False,
            "paired_protocol": str(source["l2"].train.paired_protocol),
            "paired_arm": str(source["l2"].train.paired_arm),
            "paired_seed": int(source["l2"].train.paired_seed),
            "paired_counterpart_config_path": str(source["l2"].train.paired_counterpart_config_path),
            "allow_parameter_updates": bool(source["l2"].train.allow_parameter_updates),
        }
    )
    payload["train"] = train
    payload["v26c_oof"] = {
        "fold_id": fold_id,
        "role": "fresh L2 OOF support only; not a deployment checkpoint",
        "fixed_snapshot_epoch": L2_EPOCH,
        "unknown_training_exposure": 0,
        "held_out_checkpoint_selection": False,
        "frozen_feature_protocol": "V18r stable8 feature schema selected before V26c; no fold-label feature selection",
    }
    return payload


def build(project_root: Path, *, preview: bool = False) -> dict[str, Any]:
    root = project_root.resolve()
    source = _validate_sources(root)
    source["root"] = root
    labels_by_parent = _index_parent_labels(source["paths"]["index"])
    v21_contract = source["fold_contract"]
    assignment = source["assignment"]
    assignments = list(assignment.get("assignments") or ())
    assignment_by_parent = {str(row["parent_sample_id"]): dict(row) for row in assignments}
    if len(assignment_by_parent) != 986:
        raise ValueError("V26c reused assignment repeats or omits known parents")
    fold_specs: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        frozen_fold = fold_record(v21_contract, fold_id)
        train_ids = list(frozen_fold["train_parent_ids"])
        held_out_ids = list(frozen_fold["held_out_parent_ids"])
        if set(train_ids) & set(held_out_ids):
            raise ValueError("V26c fold has parent overlap")
        train_components = {assignment_by_parent[parent]["provenance_component_id"] for parent in train_ids}
        held_components = {assignment_by_parent[parent]["provenance_component_id"] for parent in held_out_ids}
        if train_components & held_components:
            raise ValueError("V26c fold split a frozen provenance component")
        train_counts = Counter(labels_by_parent[parent] for parent in train_ids)
        held_counts = Counter(labels_by_parent[parent] for parent in held_out_ids)
        if set(train_counts) != set(L1_LABELS) or set(held_counts) != set(L1_LABELS):
            raise ValueError("V26c fold lacks a KnownGate label")
        l2_train_ids = [parent for parent in train_ids if labels_by_parent[parent] in L2_LABELS]
        l2_held_ids = [parent for parent in held_out_ids if labels_by_parent[parent] in L2_LABELS]
        l2_train_counts = Counter(labels_by_parent[parent] for parent in l2_train_ids)
        l2_held_counts = Counter(labels_by_parent[parent] for parent in l2_held_ids)
        if set(l2_train_counts) != set(L2_LABELS) or set(l2_held_counts) != set(L2_LABELS):
            raise ValueError("V26c fold lacks a DBB label")
        fold_specs.append(
            {
                "fold_id": fold_id,
                "train_ids": train_ids,
                "held_out_ids": held_out_ids,
                "train_components": train_components,
                "held_components": held_components,
                "train_counts": train_counts,
                "held_counts": held_counts,
                "l2_train_ids": l2_train_ids,
                "l2_held_ids": l2_held_ids,
                "l2_train_counts": l2_train_counts,
                "l2_held_counts": l2_held_counts,
            }
        )
    if preview:
        return {
            "status": "preview_passed",
            "read_only": True,
            "fold_count": FOLD_COUNT,
            "l1_fixed_snapshot_epoch": L1_EPOCH,
            "l2_fixed_snapshot_epoch": L2_EPOCH,
            "folds": [
                {
                    "fold_id": int(row["fold_id"]),
                    "l1_train": len(row["train_ids"]),
                    "l1_held_out": len(row["held_out_ids"]),
                    "l2_train": len(row["l2_train_ids"]),
                    "l2_held_out": len(row["l2_held_ids"]),
                }
                for row in fold_specs
            ],
        }
    source_rebuild = _prepare_label_free_source_rebuilds(
        root, source=source, labels_by_parent=labels_by_parent
    )
    fold_records: list[dict[str, Any]] = []
    for spec in fold_specs:
        fold_id = int(spec["fold_id"])
        train_ids = list(spec["train_ids"])
        held_out_ids = list(spec["held_out_ids"])
        train_components = set(spec["train_components"])
        held_components = set(spec["held_components"])
        train_counts = Counter(spec["train_counts"])
        held_counts = Counter(spec["held_counts"])
        l2_train_ids = list(spec["l2_train_ids"])
        l2_held_ids = list(spec["l2_held_ids"])
        l2_train_counts = Counter(spec["l2_train_counts"])
        l2_held_counts = Counter(spec["l2_held_counts"])
        l1_stats_path = root / l1_normalization_path(fold_id)
        l1_stats = build_normalization(
            root,
            source["paths"]["index"],
            l1_stats_path,
            "train",
            allowed_labels=L1_LABELS,
            forbidden_labels=("UnknownTarget",),
            allowed_parent_ids=tuple(train_ids),
            schema_version="RADAR-LN3-V26c",
            stats_name=f"cq08_v26c_known_gate_oof_fold_{fold_id:02d}_train_v1",
            scope="V26c L1 known fold-train parents only; held-out, Unknown, validation and test excluded",
        )
        l2_stats_path = root / l2_normalization_path(fold_id)
        l2_stats = build_normalization(
            root,
            source["paths"]["index"],
            l2_stats_path,
            "train",
            allowed_labels=L2_LABELS,
            forbidden_labels=("UnknownTarget", "ClutterTarget"),
            allowed_parent_ids=tuple(l2_train_ids),
            schema_version="RADAR-LN3-V26c",
            stats_name=f"cq08_v26c_dbb_specialist_oof_fold_{fold_id:02d}_train_v1",
            scope="V26c L2 DBB fold-train parents only; Clutter, held-out, Unknown, validation and test excluded",
        )
        for stats, expected_ids in ((l1_stats, train_ids), (l2_stats, l2_train_ids)):
            if (
                int(stats.get("parent_count", -1)) != len(expected_ids)
                or int(stats.get("allowed_parent_ids_count", -1)) != len(expected_ids)
                or stats.get("allowed_parent_ids_sha256") != sequence_sha256(sorted(expected_ids))
                or int(stats.get("selected_excluded_rd_tensor_path_overlap_count", -1)) != 0
            ):
                raise ValueError("V26c fold normalization is not strict fold-train-only")
        l1_train_events = _selected_fold_events(source["paths"]["index"], train_ids, L1_LABELS)
        l1_held_out_events = _selected_fold_events(source["paths"]["index"], held_out_ids, L1_LABELS)
        l2_train_events = _selected_fold_events(source["paths"]["index"], l2_train_ids, L2_LABELS)
        l2_held_out_events = _selected_fold_events(source["paths"]["index"], l2_held_ids, L2_LABELS)
        train_course_cache = _slice_recomputed_cache(
            root, source_path=root / l1_source_feature_cache_path(), output_path=root / l1_train_feature_cache_path(fold_id),
            selected=l1_train_events, width=1, valid_rank=2, size_limit=L1_CACHE_MAX_BYTES,
            source_description="disjoint slice of V26c's one-time label-free source-recomputed causal course cache",
        )
        held_out_course_cache = _slice_recomputed_cache(
            root, source_path=root / l1_source_feature_cache_path(), output_path=root / l1_held_out_feature_cache_path(fold_id),
            selected=l1_held_out_events, width=1, valid_rank=2, size_limit=L1_CACHE_MAX_BYTES,
            source_description="disjoint slice of V26c's one-time label-free source-recomputed causal course cache",
        )
        train_physical_cache = _slice_recomputed_cache(
            root, source_path=root / l2_source_physical_cache_path(), output_path=root / l2_train_physical_cache_path(fold_id),
            selected=l2_train_events, width=4, valid_rank=1, size_limit=L2_CACHE_MAX_BYTES,
            source_description="disjoint slice of V26c's one-time label-free source-recomputed RD physical4 cache",
        )
        held_out_physical_cache = _slice_recomputed_cache(
            root, source_path=root / l2_source_physical_cache_path(), output_path=root / l2_held_out_physical_cache_path(fold_id),
            selected=l2_held_out_events, width=4, valid_rank=1, size_limit=L2_CACHE_MAX_BYTES,
            source_description="disjoint slice of V26c's one-time label-free source-recomputed RD physical4 cache",
        )
        train_micro_cache = _slice_recomputed_cache(
            root, source_path=root / l2_source_micro_cache_path(), output_path=root / l2_train_micro_cache_path(fold_id),
            selected=l2_train_events, width=8, valid_rank=1, size_limit=L2_CACHE_MAX_BYTES,
            source_description="disjoint slice of V26c's one-time label-free source-recomputed stable8 micro-Doppler cache",
        )
        held_out_micro_cache = _slice_recomputed_cache(
            root, source_path=root / l2_source_micro_cache_path(), output_path=root / l2_held_out_micro_cache_path(fold_id),
            selected=l2_held_out_events, width=8, valid_rank=1, size_limit=L2_CACHE_MAX_BYTES,
            source_description="disjoint slice of V26c's one-time label-free source-recomputed stable8 micro-Doppler cache",
        )
        l1_held_out_events_for_l2 = _selected_fold_events(
            source["paths"]["index"], held_out_ids, L1_LABELS
        )
        held_out_mixed_physical_cache = _slice_recomputed_cache(
            root,
            source_path=root / l2_source_mixed_physical_cache_path(),
            output_path=root / l2_held_out_mixed_physical_cache_path(fold_id),
            selected=l1_held_out_events_for_l2,
            width=4,
            valid_rank=1,
            size_limit=L2_CACHE_MAX_BYTES,
            source_description="inference-only disjoint slice of V26c's all-Known label-free source-recomputed RD physical4 cache",
        )
        held_out_mixed_micro_cache = _slice_recomputed_cache(
            root,
            source_path=root / l2_source_mixed_micro_cache_path(),
            output_path=root / l2_held_out_mixed_micro_cache_path(fold_id),
            selected=l1_held_out_events_for_l2,
            width=8,
            valid_rank=1,
            size_limit=L2_CACHE_MAX_BYTES,
            source_description="inference-only disjoint slice of V26c's all-Known label-free source-recomputed stable8 micro-Doppler cache",
        )
        for cache in (held_out_mixed_physical_cache, held_out_mixed_micro_cache):
            cache.update(
                {
                    "inference_only_mixed_l1_held_out_population": True,
                    "l2_training_exposure": 0,
                    "l2_normalization_exposure": 0,
                    "l3_threshold_fit_exposure": 0,
                }
            )
        l1_contract_path = root / l1_data_contract_path(fold_id)
        l1_contract = _fold_contract(
            task="known_gate",
            fold_id=fold_id,
            train_ids=train_ids,
            held_out_ids=held_out_ids,
            train_counts={label: int(train_counts[label]) for label in L1_LABELS},
            held_out_counts={label: int(held_counts[label]) for label in L1_LABELS},
            normalization={
                "path": relative_path(root, l1_stats_path),
                "sha256": sha256_file(l1_stats_path),
                "parent_count": len(train_ids),
            },
            train_caches={"course": train_course_cache},
            held_out_caches={"course": held_out_course_cache},
        )
        write_new_or_identical(l1_contract_path, l1_contract)
        l2_contract_path = root / l2_data_contract_path(fold_id)
        l2_contract = _fold_contract(
            task="dbb_specialist",
            fold_id=fold_id,
            train_ids=l2_train_ids,
            held_out_ids=l2_held_ids,
            train_counts={label: int(l2_train_counts[label]) for label in L2_LABELS},
            held_out_counts={label: int(l2_held_counts[label]) for label in L2_LABELS},
            normalization={
                "path": relative_path(root, l2_stats_path),
                "sha256": sha256_file(l2_stats_path),
                "parent_count": len(l2_train_ids),
            },
            train_caches={
                "rd_physical4": train_physical_cache,
                "micro_doppler8": train_micro_cache,
            },
            held_out_caches={
                "rd_physical4": held_out_physical_cache,
                "micro_doppler8": held_out_micro_cache,
                "mixed_l1_held_out_inference_only_rd_physical4": held_out_mixed_physical_cache,
                "mixed_l1_held_out_inference_only_micro_doppler8": held_out_mixed_micro_cache,
            },
        )
        write_new_or_identical(l2_contract_path, l2_contract)
        l1_path = root / l1_config_path(fold_id)
        write_new_or_identical(
            l1_path,
            _l1_config(
                source,
                fold_id=fold_id,
                contract_path=l1_contract_path,
                normalization_path=l1_stats_path,
                stats=l1_stats,
                train_course_cache=train_course_cache,
                held_out_course_cache=held_out_course_cache,
                train_counts=train_counts,
            ),
        )
        l2_path = root / l2_config_path(fold_id)
        write_new_or_identical(
            l2_path,
            _l2_config(
                source,
                fold_id=fold_id,
                contract_path=l2_contract_path,
                normalization_path=l2_stats_path,
                stats=l2_stats,
                train_physical_cache=train_physical_cache,
                held_out_physical_cache=held_out_physical_cache,
                train_micro_cache=train_micro_cache,
                held_out_micro_cache=held_out_micro_cache,
                train_counts=l2_train_counts,
            ),
        )
        fold_records.append(
            {
                "fold_id": fold_id,
                "source_v21e_fold_contract_match": True,
                "train_parent_ids": train_ids,
                "held_out_parent_ids": held_out_ids,
                "train_parent_ids_sha256": sequence_sha256(train_ids),
                "held_out_parent_ids_sha256": sequence_sha256(held_out_ids),
                "train_component_count": len(train_components),
                "held_out_component_count": len(held_components),
                "component_overlap_count": 0,
                "l1": {
                    "train_parent_count": len(train_ids),
                    "held_out_parent_count": len(held_out_ids),
                    "train_label_parent_counts": {label: int(train_counts[label]) for label in L1_LABELS},
                    "held_out_label_parent_counts": {label: int(held_counts[label]) for label in L1_LABELS},
                    "config_path": relative_path(root, l1_path),
                    "config_sha256": sha256_file(l1_path),
                    "data_contract_path": relative_path(root, l1_contract_path),
                    "data_contract_sha256": sha256_file(l1_contract_path),
                    "normalization_path": relative_path(root, l1_stats_path),
                    "normalization_sha256": sha256_file(l1_stats_path),
                    "train_course_cache": train_course_cache,
                    "held_out_course_cache": held_out_course_cache,
                    "fixed_snapshot_epoch": L1_EPOCH,
                },
                "l2": {
                    "train_parent_count": len(l2_train_ids),
                    "held_out_parent_count": len(l2_held_ids),
                    "train_parent_ids": sorted(l2_train_ids),
                    "held_out_parent_ids": sorted(l2_held_ids),
                    "train_label_parent_counts": {label: int(l2_train_counts[label]) for label in L2_LABELS},
                    "held_out_label_parent_counts": {label: int(l2_held_counts[label]) for label in L2_LABELS},
                    "config_path": relative_path(root, l2_path),
                    "config_sha256": sha256_file(l2_path),
                    "data_contract_path": relative_path(root, l2_contract_path),
                    "data_contract_sha256": sha256_file(l2_contract_path),
                    "normalization_path": relative_path(root, l2_stats_path),
                    "normalization_sha256": sha256_file(l2_stats_path),
                    "train_rd_physical_cache": train_physical_cache,
                    "held_out_rd_physical_cache": held_out_physical_cache,
                    "train_micro_doppler_cache": train_micro_cache,
                    "held_out_micro_doppler_cache": held_out_micro_cache,
                    "held_out_mixed_l1_inference_only_rd_physical_cache": held_out_mixed_physical_cache,
                    "held_out_mixed_l1_inference_only_micro_doppler_cache": held_out_mixed_micro_cache,
                    "fixed_snapshot_epoch": L2_EPOCH,
                },
            }
        )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26c strict Known-only L1/L2 provenance-grouped OOF for L3 calibration",
        "scope": "fresh fold-local L1/L2 model training; Known OOF threshold fitting only; no Unknown threshold fitting or test access",
        "assignment_reuse": {
            "source": relative_path(root, source["paths"]["v21e_assignment"]),
            "sha256": source["hashes"]["v21e_assignment"],
            "fold_contract": relative_path(root, source["paths"]["v21e_fold_contract"]),
            "fold_contract_sha256": source["hashes"]["v21e_fold_contract"],
            "reuse_scope": "provenance-grouped parent assignment only; no model, normalization, cache values, checkpoint or prediction reuse",
        },
        "feature_value_rebuild": {
            "manifest_path": relative_path(root, root / SOURCE_FEATURE_MANIFEST),
            "manifest_sha256": sha256_file(root / SOURCE_FEATURE_MANIFEST),
            "strategy": "one label-free source extraction over fixed Known/DBB training universes, followed by exact disjoint fold cache slices",
            "previous_v21e_v18_cache_values_reused": False,
        },
        "source_bindings": {
            name: {"path": relative_path(root, path), "sha256": source["hashes"][name]}
            for name, path in source["paths"].items()
        },
        "fold_count": FOLD_COUNT,
        "known_train_parent_count": 986,
        "grouping": "capture_content_provenance_component",
        "l1": {
            "task": "known_gate",
            "training_seed": TRAINING_SEED_L1,
            "fixed_snapshot_epoch": L1_EPOCH,
            "unknown_training_exposure": 0,
        },
        "l2": {
            "task": "dbb_specialist",
            "training_seed": TRAINING_SEED_L2,
            "fixed_snapshot_epoch": L2_EPOCH,
            "frozen_feature_protocol": "V18r RD physical4 plus stable8 selected before V26c; no fold-label feature selection",
            "unknown_training_exposure": 0,
        },
        "l3_calibration_protocol": {
            "fit_population": "Known parent-final OOF scores only",
            "outer_cross_fit": "thresholds for fold f are fitted with other four folds only",
            "unknown_threshold_fitting": False,
            "pending_is_unknown_rejection_success": False,
            "unknown_mixed_audit": "permitted only after fitted L3 registry is frozen",
        },
        "folds": fold_records,
        "validation_access_count": 0,
        "test_access_count": 0,
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "deployment_eligible": False,
    }
    write_new_or_identical(root / CONTRACT, contract)
    checks = {
        "v21e_provenance_assignment_is_reused_byte_for_byte": True,
        "each_known_train_parent_is_held_out_once_for_l1": True,
        "each_dbb_train_parent_is_held_out_once_for_l2": True,
        "l1_l2_held_out_known_parent_sets_align_on_dbb_subset": True,
        "no_provenance_component_crosses_a_fold": True,
        "l1_normalization_is_known_fold_train_only": True,
        "l2_normalization_is_dbb_fold_train_only": True,
        "unknown_is_excluded_from_l1_l2_training_normalization_and_cache_selection": True,
        "fold_feature_caches_are_disjoint_slices_of_one_recomputed_label_free_v26c_source": True,
        "prior_v21e_v18_feature_cache_values_are_not_reused": True,
        "l1_snapshot_epoch_is_frozen_to_14": True,
        "l2_snapshot_epoch_is_frozen_to_15": True,
        "validation_and_test_are_unread": True,
        "l3_thresholds_are_not_fitted_during_prepare": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "checks": checks,
        "fold_count": FOLD_COUNT,
        "l1_fixed_snapshot_epoch": L1_EPOCH,
        "l2_fixed_snapshot_epoch": L2_EPOCH,
        "known_train_parent_count": 986,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "L3_calibration": "not_executed",
        "mixed_unknown_audit": "not_executed",
        "deployment_eligible": False,
    }
    write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "contract": str(root / CONTRACT),
        "readiness": str(root / READINESS),
        "fold_count": FOLD_COUNT,
        "l1_runs": FOLD_COUNT,
        "l2_runs": FOLD_COUNT,
        "L3_calibration": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V26c L1/L2 Known-only OOF calibration.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve(), preview=args.preview), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
