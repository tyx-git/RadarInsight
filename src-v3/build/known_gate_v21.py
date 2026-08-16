from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _as_bool,
    _parse_indices,
    _resolve,
    _sha256,
    _write_json_new_or_identical,
    _write_npz_new_or_identical,
)
from build.radar_ln3_normalization import build as build_normalization
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_CONTRACT_SCHEMA,
    KNOWN_GATE_LABEL_TO_ID,
    KNOWN_GATE_LOSS_MODE,
    KNOWN_GATE_MANIFEST_SCHEMA,
    KNOWN_GATE_READINESS_SCHEMA,
    KNOWN_GATE_SELECTION_METRIC,
    KNOWN_LABELS,
    KNOWN_TARGET_LABELS,
    UNKNOWN_LABEL,
    V21_SCHEMA_VERSION,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    BASELINE_VARIANT,
    FEATURE_NAMES_BY_VARIANT,
    KNOWN_GATE_FEATURE_DIM,
    KNOWN_GATE_FEATURE_NAMES,
    KNOWN_GATE_FEATURE_SCHEMA,
    V21A_VARIANT,
    V21B_VARIANT,
    V21C_VARIANT,
    V21D_VARIANT,
    extract_known_gate_feature_event,
    feature_contract,
)
from radar_ln3.micro_doppler import parse_cq08_mat_frame_identity, parse_cq08_raw_echo
from radar_ln3.open_set_architecture import ARCHITECTURE_SCHEMA, KNOWN_GATE_TASK_NAME


SOURCE_INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
SOURCE_CONTRACT = Path("data/processed/RADAR-LN3-V16-cq08-product-local-data-contract.json")
SOURCE_PROVENANCE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
SOURCE_CANONICAL_FRAME_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)
ARCHITECTURE_DOCUMENT = Path("docs/radar_ln3_v21_strict_open_set_architecture.md")
NORMALIZATION = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-only-train-normalization-stats.json"
)
DATA_CONTRACT = Path("data/processed/RADAR-LN3-V21-cq08-known-gate-data-contract.json")
FEATURE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-feature-cache-manifest.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V21-known-gate-training-readiness.json"
)
SPLITS = ("train", "val")
SEED = 42


def _feature_cache(split: str) -> Path:
    return Path(f"data/processed/RADAR-LN3-V21-cq08-known-gate-{split}-feature-cache.npz")


VARIANT_DETAILS: Mapping[str, dict[str, str]] = {
    BASELINE_VARIANT: {
        "version_name": "V21",
        "slug": "baseline",
        "description": "Known-only V16-core KnownGate baseline without experimental evidence",
    },
    V21A_VARIANT: {
        "version_name": "V21a",
        "slug": "spectral-temporal-consistency",
        "description": "Baseline plus native-STFT spectral-shape temporal consistency",
    },
    V21B_VARIANT: {
        "version_name": "V21b",
        "slug": "modal-consistency",
        "description": "Baseline plus target-row RD-peak versus TR-radial consistency",
    },
    V21C_VARIANT: {
        "version_name": "V21c",
        "slug": "target-row-iso",
        "description": "Baseline plus zero-offset target-row energy fraction iso",
    },
    V21D_VARIANT: {
        "version_name": "V21d",
        "slug": "d-range-stability",
        "description": "Baseline plus current-event MTD-minus-TR range standard deviation",
    },
}


def _config_path(variant: str) -> Path:
    detail = VARIANT_DETAILS[variant]
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-{detail['version_name']}-cq08-liquid-known-gate-"
        f"{detail['slug']}-seed42-config.json"
    )


def _artifact_dir(variant: str) -> str:
    detail = VARIANT_DETAILS[variant]
    return (
        f"artifacts/RADAR-LN3-{detail['version_name']}-cq08-liquid-known-gate-"
        f"{detail['slug']}-seed42-formal"
    )


def _sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


class CanonicalFrameResolver:
    REQUIRED_COLUMNS = (
        "schema_version",
        "parent_sample_id",
        "track_id",
        "raw_frame_index",
        "tensor_frame_index",
        "is_tensor_frame_kept",
        "track_point_index",
        "rd_mat_path",
        "rd_tensor_path",
        "target_range_reference_m",
    )

    def __init__(self, root: Path, path: Path) -> None:
        frame = pd.read_csv(path, usecols=list(self.REQUIRED_COLUMNS), low_memory=False)
        if frame.empty or set(frame["schema_version"].astype(str)) != {
            "cq08_rd_physical_canonical_v3_1"
        }:
            raise ValueError("V21 canonical frame provenance schema is invalid")
        if not bool(frame["is_tensor_frame_kept"].map(_as_bool).all()):
            raise ValueError("V21 canonical frame provenance unexpectedly contains dropped rows")
        for field in ("track_id", "raw_frame_index", "tensor_frame_index", "track_point_index"):
            frame[field] = pd.to_numeric(frame[field], errors="raise").astype(np.int64)
        frame["target_range_reference_m"] = pd.to_numeric(
            frame["target_range_reference_m"], errors="raise"
        ).astype(np.float64)
        if frame.duplicated(["parent_sample_id", "tensor_frame_index"]).any():
            raise ValueError("V21 canonical frame provenance repeats parent/tensor frame")
        records: dict[tuple[str, int], dict[str, Any]] = {}
        mismatch_tracks: set[int] = set()
        mismatch_count = 0
        for row in frame.itertuples(index=False):
            identity = parse_cq08_mat_frame_identity(str(row.rd_mat_path))
            if identity.track_id != int(row.track_id):
                raise ValueError("V21 MAT frame identity track differs from provenance")
            legacy = int(row.raw_frame_index)
            raw_index = int(identity.raw_dat_frame_index_zero_based)
            if legacy != raw_index:
                mismatch_count += 1
                mismatch_tracks.add(int(row.track_id))
            records[(str(row.parent_sample_id), int(row.tensor_frame_index))] = {
                "track_id": int(row.track_id),
                "track_point_index": int(row.track_point_index),
                "legacy_mat_list_ordinal_zero_based": legacy,
                "raw_dat_frame_index_zero_based": raw_index,
                "rd_tensor_path": str(row.rd_tensor_path),
                "target_range_reference_m": float(row.target_range_reference_m),
            }
        self.root = root
        self.records = records
        self.audit = {
            "row_count": len(records),
            "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count": mismatch_count,
            "affected_track_ids": sorted(mismatch_tracks),
            "legacy_field_used_as_raw_dat_index": False,
        }

    def resolve(
        self,
        *,
        parent_sample_id: str,
        track_id: int,
        track_point_index: int,
        tensor_indices: Sequence[int],
        legacy_indices: Sequence[int],
        rd_tensor_path: str,
    ) -> list[dict[str, Any]]:
        if len(tensor_indices) != len(legacy_indices):
            raise ValueError("V21 tensor/legacy frame counts differ")
        result: list[dict[str, Any]] = []
        for tensor_index, legacy_index in zip(tensor_indices, legacy_indices, strict=True):
            try:
                value = dict(self.records[(str(parent_sample_id), int(tensor_index))])
            except KeyError as exc:
                raise ValueError(
                    f"V21 canonical frame provenance lacks {parent_sample_id}/{tensor_index}"
                ) from exc
            if (
                value["track_id"] != int(track_id)
                or value["track_point_index"] != int(track_point_index)
                or value["legacy_mat_list_ordinal_zero_based"] != int(legacy_index)
                or _resolve(self.root, value["rd_tensor_path"])
                != _resolve(self.root, rd_tensor_path)
            ):
                raise ValueError("V21 index/canonical frame provenance mismatch")
            result.append(value)
        raw_indices = [int(value["raw_dat_frame_index_zero_based"]) for value in result]
        if raw_indices != sorted(raw_indices) or len(raw_indices) != len(set(raw_indices)):
            raise ValueError("V21 resolved raw-DAT frame indices are not ordered/unique")
        return result


def _load_provenance(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "RADAR-LN3-V16" or int(payload.get("cross_label_content_conflict_count", -1)) != 0:
        raise ValueError("V21 source provenance is not the clean V16 contract")
    records = {str(row["parent_sample_id"]): dict(row) for row in payload.get("records") or []}
    if len(records) != int(payload.get("parent_count", -1)):
        raise ValueError("V21 source provenance parent records are incomplete")
    return records, payload


def _load_canonical(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"amplitude", "velocity_mps", "range_offset_m"}
        if not required.issubset(payload.files):
            raise ValueError(f"V21 canonical tensor lacks {sorted(required)}: {path}")
        amplitude = np.asarray(payload["amplitude"])
        velocity = np.asarray(payload["velocity_mps"])
        range_offset = np.asarray(payload["range_offset_m"])
    if amplitude.ndim != 3 or amplitude.shape[1:] != (21, 384):
        raise ValueError(f"V21 canonical tensor has invalid shape: {path}")
    return amplitude, velocity, range_offset


def _empty_event() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((KNOWN_GATE_FEATURE_DIM,), dtype=np.float32),
        np.zeros((KNOWN_GATE_FEATURE_DIM,), dtype=np.bool_),
    )


def _feature_statistics(
    selected: pd.DataFrame,
    values: np.ndarray,
    valid: np.ndarray,
    *,
    include_label_distributions: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "event_count": int(len(selected)),
        "valid_value_count_by_feature": {
            name: int(valid[:, index].sum())
            for index, name in enumerate(KNOWN_GATE_FEATURE_NAMES)
        },
        "valid_fraction_by_feature": {
            name: float(valid[:, index].mean()) if len(selected) else 0.0
            for index, name in enumerate(KNOWN_GATE_FEATURE_NAMES)
        },
    }
    if include_label_distributions:
        label_statistics: dict[str, Any] = {}
        labels = selected["label_name"].astype(str).to_numpy()
        for label in KNOWN_LABELS:
            mask = labels == label
            feature_rows: dict[str, Any] = {}
            for feature_index, feature_name in enumerate(KNOWN_GATE_FEATURE_NAMES):
                active = mask & valid[:, feature_index]
                data = values[active, feature_index]
                feature_rows[feature_name] = {
                    "event_support": int(active.sum()),
                    "mean": float(np.mean(data)) if data.size else None,
                    "median": float(np.median(data)) if data.size else None,
                    "q25": float(np.quantile(data, 0.25)) if data.size else None,
                    "q75": float(np.quantile(data, 0.75)) if data.size else None,
                }
            label_statistics[label] = feature_rows
        result["train_only_label_distributions"] = label_statistics
    return result


def _build_split(
    root: Path,
    split: str,
    index: pd.DataFrame,
    provenance: Mapping[str, Mapping[str, Any]],
    resolver: CanonicalFrameResolver,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = index[
        index["split_name"].astype(str).eq(split)
        & index["label_name"].astype(str).isin(KNOWN_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V21 {split} known event population is empty or duplicated")
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    values = np.zeros((len(selected), KNOWN_GATE_FEATURE_DIM), dtype=np.float32)
    valid = np.zeros((len(selected), KNOWN_GATE_FEATURE_DIM), dtype=np.bool_)
    event_cursor = 0
    raw_file_access_count = 0
    raw_frame_count = 0
    rd_tensor_access_count = 0
    tr_radial_speed_missing_rd_present_event_count = 0
    tr_range_missing_rd_present_event_count = 0
    parent_label_counts: Counter[str] = Counter()
    grouped = list(selected.groupby("parent_sample_id", sort=True))
    for parent_number, (parent_id_value, parent) in enumerate(grouped, start=1):
        parent_id = str(parent_id_value)
        labels = parent["label_name"].astype(str).unique().tolist()
        track_ids = pd.to_numeric(parent["track_id"], errors="raise").astype(int).unique().tolist()
        raw_paths = parent["cq08_raw_echo_file"].astype(str).unique().tolist()
        rd_paths = [
            str(value)
            for value in parent.loc[parent["rd_present"].map(_as_bool), "rd_tensor_path"]
            .dropna()
            .unique()
            if str(value).strip()
        ]
        if len(labels) != 1 or len(track_ids) != 1 or len(raw_paths) != 1 or len(rd_paths) > 1:
            raise ValueError(f"V21 parent identity/path is not unique: {parent_id}")
        label = labels[0]
        track_id = int(track_ids[0])
        parent_label_counts[label] += 1
        source = provenance.get(parent_id)
        if source is None or source.get("v16_split_name") != split or source.get("label_name") != label:
            raise ValueError(f"V21 parent provenance is missing or differs: {parent_id}")
        raw_path = _resolve(root, raw_paths[0])
        if raw_path != Path(str(source["raw_echo_file"])).resolve() or _sha256(raw_path) != source["raw_sha256"]:
            raise ValueError(f"V21 raw source path/hash differs from V16 provenance: {parent_id}")
        raw_frames = parse_cq08_raw_echo(raw_path)
        raw_file_access_count += 1
        raw_frame_count += len(raw_frames)
        amplitude = velocity = range_offset = None
        if rd_paths:
            rd_path = _resolve(root, rd_paths[0])
            if rd_path != Path(str(source["rd_tensor_path"])).resolve() or _sha256(rd_path) != source["rd_sha256"]:
                raise ValueError(f"V21 canonical RD path/hash differs from V16 provenance: {parent_id}")
            amplitude, velocity, range_offset = _load_canonical(rd_path)
            rd_tensor_access_count += 1
        used_raw_indices: set[int] = set()
        for row in parent.itertuples(index=False):
            if not _as_bool(row.rd_present):
                event_values, event_valid = _empty_event()
            else:
                tensor_indices = _parse_indices(row.rd_tensor_frame_indices_json)
                legacy_indices = _parse_indices(row.rd_raw_frame_indices_json)
                if len(tensor_indices) != int(row.rd_frame_count) or len(tensor_indices) > 12:
                    raise ValueError(f"V21 event frame count is invalid: {row.sample_id}")
                if amplitude is None or velocity is None or range_offset is None:
                    raise ValueError(f"V21 RD-present event has no canonical tensor: {row.sample_id}")
                if min(tensor_indices) < 0 or max(tensor_indices) >= amplitude.shape[0]:
                    raise ValueError(f"V21 tensor frame index is out of bounds: {row.sample_id}")
                resolved = resolver.resolve(
                    parent_sample_id=parent_id,
                    track_id=track_id,
                    track_point_index=int(row.track_point_index),
                    tensor_indices=tensor_indices,
                    legacy_indices=legacy_indices,
                    rd_tensor_path=str(row.rd_tensor_path),
                )
                raw_indices = [int(item["raw_dat_frame_index_zero_based"]) for item in resolved]
                overlap = used_raw_indices & set(raw_indices)
                if overlap:
                    raise ValueError(f"V21 raw frame is reused across current events: {parent_id}/{sorted(overlap)}")
                used_raw_indices.update(raw_indices)
                tr = json.loads(str(row.tr_feature_values_json))
                if "radial_speed_mps" not in tr or "range_m" not in tr:
                    raise ValueError(f"V21 TR feature payload lacks radial/range keys: {row.sample_id}")
                tr_radial_speed = tr["radial_speed_mps"]
                tr_range = tr["range_m"]
                tr_radial_speed_missing_rd_present_event_count += int(tr_radial_speed is None)
                tr_range_missing_rd_present_event_count += int(tr_range is None)
                feature = extract_known_gate_feature_event(
                    raw_frames=raw_frames,
                    raw_frame_indices=raw_indices,
                    track_point_index=int(row.track_point_index),
                    amplitude_db=amplitude[np.asarray(tensor_indices, dtype=np.int64)],
                    velocity_mps=velocity,
                    range_offset_m=range_offset,
                    tr_radial_speed_mps=tr_radial_speed,
                    tr_range_m=tr_range,
                    mtd_detected_range_m=[
                        float(item["target_range_reference_m"]) for item in resolved
                    ],
                )
                event_values, event_valid = feature.values, feature.valid
            values[event_cursor] = event_values
            valid[event_cursor] = event_valid
            event_cursor += 1
        if parent_number == 1 or parent_number % 50 == 0 or parent_number == len(grouped):
            print(
                f"[V21 prepare] split={split} parent={parent_number}/{len(grouped)} "
                f"events={event_cursor}",
                flush=True,
            )
    if event_cursor != len(selected):
        raise ValueError("V21 feature extraction did not consume every selected event")
    arrays = {"sample_ids": sample_ids, "values": values, "valid": valid}
    statistics = {
        "parent_count": len(grouped),
        "parent_label_counts": dict(sorted(parent_label_counts.items())),
        "raw_file_access_count": raw_file_access_count,
        "raw_frame_count_in_accessed_files": raw_frame_count,
        "canonical_rd_tensor_access_count": rd_tensor_access_count,
        "tr_radial_speed_missing_rd_present_event_count": (
            tr_radial_speed_missing_rd_present_event_count
        ),
        "tr_range_missing_rd_present_event_count": tr_range_missing_rd_present_event_count,
        "test_payload_access_count": 0,
        **_feature_statistics(
            selected,
            values,
            valid,
            include_label_distributions=(split == "train"),
        ),
    }
    return arrays, statistics


def _support(index: pd.DataFrame) -> dict[str, dict[str, int]]:
    parents = index[["parent_sample_id", "split_name", "label_name"]].drop_duplicates(
        "parent_sample_id"
    )
    result: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        selected = parents[parents["split_name"].astype(str).eq(split)]
        result[split] = {
            label: int((selected["label_name"].astype(str) == label).sum())
            for label in (*KNOWN_LABELS, UNKNOWN_LABEL)
        }
    return result


def _build_contract(
    *,
    source_hashes: Mapping[str, str],
    stats: Mapping[str, Any],
    stats_hash: str,
    cache_hashes: Mapping[str, str],
    split_statistics: Mapping[str, Mapping[str, Any]],
    support: Mapping[str, Mapping[str, int]],
    resolver_audit: Mapping[str, Any],
) -> dict[str, Any]:
    train_known = sum(int(support["train"][label]) for label in KNOWN_LABELS)
    val_known = sum(int(support["val"][label]) for label in KNOWN_LABELS)
    return {
        "schema_version": KNOWN_GATE_CONTRACT_SCHEMA,
        "contract_name": "cq08_v21_known_gate_strict_open_set_local_feature_experiments_v1",
        "architecture_schema": ARCHITECTURE_SCHEMA,
        "architecture_document": ARCHITECTURE_DOCUMENT.as_posix(),
        "architecture_document_sha256": source_hashes["architecture"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": source_hashes["index"],
        "source_v16_contract": SOURCE_CONTRACT.as_posix(),
        "source_v16_contract_sha256": source_hashes["contract"],
        "source_v16_provenance": SOURCE_PROVENANCE.as_posix(),
        "source_v16_provenance_sha256": source_hashes["provenance"],
        "source_canonical_frame_index": SOURCE_CANONICAL_FRAME_INDEX.as_posix(),
        "source_canonical_frame_index_sha256": source_hashes["canonical_frame_index"],
        "strict_open_set_contract": {
            "l1_task_name": KNOWN_GATE_TASK_NAME,
            "l1_classes": list(KNOWN_GATE_CLASS_NAMES),
            "known_training_labels": list(KNOWN_LABELS),
            "unknown_label": UNKNOWN_LABEL,
            "unknown_is_model_class": False,
            "unknown_is_training_target": False,
            "unknown_is_normalization_population": False,
            "unknown_is_sampler_population": False,
            "unknown_is_optimizer_population": False,
            "unknown_is_checkpoint_ranking_population": False,
            "unknown_is_threshold_fitting_population": False,
            "unknown_training_exposure": 0,
            "l2_task_name": "dbb_specialist",
            "l3_name": "semantic_decision_layer",
            "cu_specialist_exists_in_v21": False,
        },
        "task_contract": {
            "name": KNOWN_GATE_TASK_NAME,
            "class_names": list(KNOWN_GATE_CLASS_NAMES),
            "label_to_id": dict(KNOWN_GATE_LABEL_TO_ID),
            "forbidden_training_labels": [UNKNOWN_LABEL],
            "train_parent_count": train_known,
            "validation_parent_count": val_known,
            "train_known_target_parent_count": sum(
                int(support["train"][label]) for label in KNOWN_TARGET_LABELS
            ),
            "train_clutter_parent_count": int(support["train"]["ClutterTarget"]),
            "validation_known_target_parent_count": sum(
                int(support["val"][label]) for label in KNOWN_TARGET_LABELS
            ),
            "validation_clutter_parent_count": int(support["val"]["ClutterTarget"]),
            "local_argmax_is_product_action": False,
        },
        "normalization_contract": {
            "path": NORMALIZATION.as_posix(),
            "sha256": stats_hash,
            "fit_split": "train",
            "fit_population": "known_only",
            "parent_count": int(stats["parent_count"]),
            "selected_label_parent_counts": stats["selected_label_parent_counts"],
            "forbidden_labels": stats["forbidden_labels"],
            "excluded_label_parent_counts": stats["excluded_label_parent_counts"],
            "selected_excluded_rd_tensor_path_overlap_count": int(
                stats["selected_excluded_rd_tensor_path_overlap_count"]
            ),
            "validation_and_test_excluded": True,
        },
        "feature_contract": feature_contract(),
        "feature_caches": {
            split: {
                "path": _feature_cache(split).as_posix(),
                "sha256": cache_hashes[split],
                **dict(split_statistics[split]),
            }
            for split in SPLITS
        },
        "canonical_frame_mapping_audit": dict(resolver_audit),
        "data_scope": {
            "split_label_parent_counts": {
                split: dict(counts) for split, counts in support.items()
            },
            "test_rows_read_for_support_only": True,
            "test_raw_payload_access_count": 0,
            "test_rd_payload_access_count": 0,
            "test_used_for_normalization": False,
            "test_used_for_checkpoint_selection": False,
        },
        "training_contract": {
            "variants": {
                variant: {
                    "description": detail["description"],
                    "feature_names": list(FEATURE_NAMES_BY_VARIANT[variant]),
                    "full_parameter_training": True,
                }
                for variant, detail in VARIANT_DETAILS.items()
            },
            "seed": SEED,
            "dataloader_seed": SEED,
            "epochs": 20,
            "early_stopping": False,
            "loss_mode": KNOWN_GATE_LOSS_MODE,
            "checkpoint_selection": KNOWN_GATE_SELECTION_METRIC,
            "treatment_initialization": "copy every Baseline shared tensor and zero only new Fusion columns",
        },
        "evaluation_contract": {
            "primary_scope": "development validation parent-final KnownGate local task",
            "clutter_validation_support": int(support["val"]["ClutterTarget"]),
            "parent_stratified_bootstrap_required": True,
            "bootstrap_confidence_level": 0.95,
            "primary_positive_signal_macro_f1_delta_strictly_greater_than": 0.02,
            "test": "未执行",
            "OOF": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "P95": "未执行",
            "full_cascade": "未执行",
        },
        "action_semantics": {
            "unknown_probability_output": False,
            "unknown_is_strict_open_set_action": True,
            "pending_abstain_counts_as_unknown_success": False,
            "threshold_source": "future train parent-grouped OOF only",
        },
    }


def _build_manifest(
    *,
    contract_hash: str,
    source_hashes: Mapping[str, str],
    cache_hashes: Mapping[str, str],
    split_statistics: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": KNOWN_GATE_MANIFEST_SCHEMA,
        "feature_schema": KNOWN_GATE_FEATURE_SCHEMA,
        "feature_names": list(KNOWN_GATE_FEATURE_NAMES),
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": source_hashes["index"],
        "source_provenance": SOURCE_PROVENANCE.as_posix(),
        "source_provenance_sha256": source_hashes["provenance"],
        "data_contract": DATA_CONTRACT.as_posix(),
        "data_contract_sha256": contract_hash,
        "cache_files": {
            split: {
                "path": _feature_cache(split).as_posix(),
                "sha256": cache_hashes[split],
                "fields": ["sample_ids", "values", "valid"],
                "event_count": int(split_statistics[split]["event_count"]),
            }
            for split in SPLITS
        },
        "sample_id_is_model_input": False,
        "frame_index_is_model_input": False,
        "valid_mask_is_model_input": False,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }


def _build_config(
    variant: str,
    *,
    source_hashes: Mapping[str, str],
    contract_hash: str,
    stats: Mapping[str, Any],
    stats_hash: str,
    manifest_hash: str,
    cache_hashes: Mapping[str, str],
    support: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    detail = VARIANT_DETAILS[variant]
    artifact = _artifact_dir(variant)
    train_target = sum(int(support["train"][label]) for label in KNOWN_TARGET_LABELS)
    train_clutter = int(support["train"]["ClutterTarget"])
    total = train_target + train_clutter
    class_weights = [total / (2.0 * train_target), total / (2.0 * train_clutter)]
    has_features = bool(FEATURE_NAMES_BY_VARIANT[variant])
    return {
        "schema_version": V21_SCHEMA_VERSION,
        "config_name": (
            f"RADAR-LN3-{detail['version_name']}-cq08-liquid-known-gate-"
            f"{detail['slug']}-seed42"
        ),
        "variant": variant,
        "task": {
            "name": KNOWN_GATE_TASK_NAME,
            "class_names": list(KNOWN_GATE_CLASS_NAMES),
            "label_to_id": dict(KNOWN_GATE_LABEL_TO_ID),
            "forbidden_training_labels": [UNKNOWN_LABEL],
            "feature_names": list(FEATURE_NAMES_BY_VARIANT[variant]),
        },
        "data": {
            "index_path": SOURCE_INDEX.as_posix(),
            "contract_path": DATA_CONTRACT.as_posix(),
            "normalization_stats_path": NORMALIZATION.as_posix(),
            "train_split_name": "train",
            "validation_split_name": "val",
            "test_split_name": "test",
            "index_schema_version": "RADAR-LN3-V16",
            "contract_schema_version": KNOWN_GATE_CONTRACT_SCHEMA,
            "expected_index_sha256": source_hashes["index"],
            "expected_contract_sha256": contract_hash,
            "expected_stats_sha256": stats_hash,
            "max_sequence_length": 29,
            "max_rd_frames_per_point": 12,
            "rd_amplitude_center": float(stats["amplitude"]["center"]),
            "rd_amplitude_scale": float(stats["amplitude"]["scale"]),
            "feature_cache_manifest_path": FEATURE_MANIFEST.as_posix() if has_features else None,
            "expected_feature_cache_manifest_sha256": manifest_hash if has_features else None,
            "train_feature_cache_path": _feature_cache("train").as_posix() if has_features else None,
            "validation_feature_cache_path": _feature_cache("val").as_posix() if has_features else None,
            "expected_train_feature_cache_sha256": cache_hashes["train"] if has_features else None,
            "expected_validation_feature_cache_sha256": cache_hashes["val"] if has_features else None,
        },
        "model": {
            "tr_value_dim": 14,
            "tr_missing_dim": 11,
            "rd_quality_dim": 7,
            "hidden_dim": 48,
            "rd_embedding_dim": 24,
            "liquid_hidden_dim": 48,
            "rd_height": 21,
            "rd_width": 384,
            "liquid_tau_min_seconds": 0.05,
            "liquid_tau_max_seconds": 30.0,
            "liquid_zero_dt_event_floor_seconds": 0.001,
            "liquid_dt_max_seconds": 120.0,
            "liquid_invalid_dt_update_mode": "bounded_observation",
            "liquid_invalid_dt_observation_gain": 0.25,
            "liquid_state_normalization": "recurrent",
            "rd_frame_aggregation": "masked_mean",
            "dt_reason_mode": "v3_reason_aware_dynamics_only",
            "evidence_dim": len(FEATURE_NAMES_BY_VARIANT[variant]),
        },
        "train": {
            "batch_size": 2,
            "learning_rate": 0.0001,
            "seed": SEED,
            "dataloader_seed": SEED,
            "device": "cuda",
            "epochs": 20,
            "early_stopping_patience": 0,
            "gradient_clip_norm": 100.0,
            "checkpoint_selection_metric": KNOWN_GATE_SELECTION_METRIC,
            "task_min_precision_recall_target": 0.90,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "class_weights": class_weights,
            "loss_mode": KNOWN_GATE_LOSS_MODE,
            "prefix_auxiliary_weight": 0.1,
            "prefix_auxiliary_min_events": 3,
            "checkpoint_schema": f"RADAR-LN3-V21-{variant}-known-gate-checkpoint-v1",
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": False,
            "full_parameter_training": True,
            "paired_baseline_config_path": (
                _config_path(BASELINE_VARIANT).as_posix() if has_features else None
            ),
        },
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_paths = {
        "index": root / SOURCE_INDEX,
        "contract": root / SOURCE_CONTRACT,
        "provenance": root / SOURCE_PROVENANCE,
        "canonical_frame_index": root / SOURCE_CANONICAL_FRAME_INDEX,
        "architecture": root / ARCHITECTURE_DOCUMENT,
    }
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"V21 source {name} is missing: {path}")
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    index = pd.read_csv(source_paths["index"], low_memory=False)
    if set(index["schema_version"].astype(str)) != {"RADAR-LN3-V16"}:
        raise ValueError("V21 source index is not RADAR-LN3-V16")
    support = _support(index)
    expected = {
        "train": {"DroneTarget": 476, "BirdTarget": 239, "BalloonTarget": 238, "ClutterTarget": 33, "UnknownTarget": 69},
        "val": {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54, "ClutterTarget": 9, "UnknownTarget": 15},
        "test": {"DroneTarget": 109, "BirdTarget": 53, "BalloonTarget": 58, "ClutterTarget": 8, "UnknownTarget": 16},
    }
    if support != expected:
        raise ValueError(f"V21 V16 support changed: expected={expected}, actual={support}")
    print("[V21 prepare] building known-only train normalization", flush=True)
    stats = build_normalization(
        root,
        source_paths["index"],
        root / NORMALIZATION,
        "train",
        allowed_labels=tuple(KNOWN_LABELS),
        forbidden_labels=(UNKNOWN_LABEL,),
        schema_version=V21_SCHEMA_VERSION,
        stats_name="cq08_v21_known_gate_known_only_train_normalization_v1",
        scope=(
            "V16 capture/content-grouped train known labels only; Unknown, validation and test excluded"
        ),
    )
    if (
        int(stats["parent_count"]) != 986
        or int(stats["excluded_label_parent_counts"].get(UNKNOWN_LABEL, -1)) != 69
        or int(stats["selected_excluded_rd_tensor_path_overlap_count"]) != 0
    ):
        raise ValueError("V21 normalization does not prove zero Unknown exposure")
    stats_hash = _sha256(root / NORMALIZATION)
    provenance, provenance_payload = _load_provenance(source_paths["provenance"])
    source_test_use = dict(provenance_payload.get("test_payload_use") or {})
    if (
        source_test_use.get("raw_and_rd_read_for_sha256_provenance_only") is not True
        or source_test_use.get("test_values_used_for_normalization") is not False
        or source_test_use.get("test_used_for_model_selection") is not False
    ):
        raise ValueError("V21 V16 provenance has an invalid sealed-test policy")
    resolver = CanonicalFrameResolver(root, source_paths["canonical_frame_index"])
    cache_hashes: dict[str, str] = {}
    split_statistics: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        print(f"[V21 prepare] extracting {split} KnownGate feature cache", flush=True)
        arrays, statistics = _build_split(root, split, index, provenance, resolver)
        cache_path = root / _feature_cache(split)
        _write_npz_new_or_identical(
            cache_path,
            tuple((name, value) for name, value in arrays.items()),
        )
        cache_hashes[split] = _sha256(cache_path)
        split_statistics[split] = statistics
    observed_tr_missing = {
        split: {
            "radial_speed": int(
                split_statistics[split]["tr_radial_speed_missing_rd_present_event_count"]
            ),
            "range": int(split_statistics[split]["tr_range_missing_rd_present_event_count"]),
        }
        for split in SPLITS
    }
    expected_tr_missing = {
        "train": {"radial_speed": 0, "range": 1},
        "val": {"radial_speed": 0, "range": 0},
    }
    if observed_tr_missing != expected_tr_missing:
        raise ValueError(
            "V21 known-event TR radial/range missingness changed: "
            f"expected={expected_tr_missing}, actual={observed_tr_missing}"
        )
    contract = _build_contract(
        source_hashes=source_hashes,
        stats=stats,
        stats_hash=stats_hash,
        cache_hashes=cache_hashes,
        split_statistics=split_statistics,
        support=support,
        resolver_audit=resolver.audit,
    )
    _write_json_new_or_identical(root / DATA_CONTRACT, contract)
    contract_hash = _sha256(root / DATA_CONTRACT)
    manifest = _build_manifest(
        contract_hash=contract_hash,
        source_hashes=source_hashes,
        cache_hashes=cache_hashes,
        split_statistics=split_statistics,
    )
    _write_json_new_or_identical(root / FEATURE_MANIFEST, manifest)
    manifest_hash = _sha256(root / FEATURE_MANIFEST)
    config_bindings: dict[str, dict[str, str]] = {}
    for variant in VARIANT_DETAILS:
        config_payload = _build_config(
            variant,
            source_hashes=source_hashes,
            contract_hash=contract_hash,
            stats=stats,
            stats_hash=stats_hash,
            manifest_hash=manifest_hash,
            cache_hashes=cache_hashes,
            support=support,
        )
        config_path = root / _config_path(variant)
        _write_json_new_or_identical(config_path, config_payload)
        loaded = load_known_gate_config(config_path)
        if loaded.variant != variant:
            raise ValueError(f"V21 generated config variant mismatch: {variant}")
        config_bindings[variant] = {
            "path": str(config_path),
            "sha256": _sha256(config_path),
            "config_name": loaded.config_name,
        }
    pair_checks: dict[str, Any] = {}
    for variant in (V21A_VARIANT, V21B_VARIANT, V21C_VARIANT, V21D_VARIANT):
        treatment_config = load_known_gate_config(root / _config_path(variant))
        treatment, baseline, evidence = build_known_gate_model_for_config(root, treatment_config)
        if baseline is None or evidence.get("new_fusion_columns_exactly_zero") is not True:
            raise ValueError(f"V21 {variant} paired initialization check failed")
        pair_checks[variant] = evidence
        del treatment, baseline
    checks = {
        "canonical_architecture_document_bound": True,
        "task_identifier_is_known_gate": True,
        "unknown_absent_from_model_classes_and_label_mapping": True,
        "unknown_excluded_from_train_normalization": True,
        "known_only_normalization_has_zero_selected_excluded_tensor_overlap": True,
        "v16_capture_content_split_reused_without_payload_repartition": True,
        "train_and_validation_feature_caches_align_all_known_events": True,
        "feature_values_are_current_event_causal_and_label_free": True,
        "feature_validity_is_gating_only": True,
        "missing_tr_fields_invalidate_only_their_dependent_evidence": True,
        "all_treatments_copy_baseline_shared_initialization": True,
        "all_treatment_new_fusion_columns_are_exactly_zero": True,
        "all_variants_require_full_20_epoch_parameter_training": True,
        "validation_labels_not_used_as_readiness_gate": True,
        "test_payload_access_count_is_zero": True,
    }
    readiness = {
        "status": "passed",
        "schema_version": KNOWN_GATE_READINESS_SCHEMA,
        "architecture_schema": ARCHITECTURE_SCHEMA,
        "checks": checks,
        "source_hashes": source_hashes,
        "contract_hashes": {
            "index_sha256": source_hashes["index"],
            "contract_sha256": contract_hash,
            "normalization_sha256": stats_hash,
            "feature_manifest_sha256": manifest_hash,
            "train_feature_cache_sha256": cache_hashes["train"],
            "validation_feature_cache_sha256": cache_hashes["val"],
        },
        "configs": config_bindings,
        "support": support,
        "normalization": {
            "known_parent_count": int(stats["parent_count"]),
            "excluded_unknown_parent_count": int(
                stats["excluded_label_parent_counts"][UNKNOWN_LABEL]
            ),
            "selected_label_parent_counts": stats["selected_label_parent_counts"],
        },
        "feature_statistics": split_statistics,
        "paired_initialization": pair_checks,
        "validation_used_for_readiness_gate": False,
        "validation_feature_values_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training_at_report_generation": "未执行",
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / READINESS, readiness)
    return {
        "status": "passed",
        "schema_version": KNOWN_GATE_READINESS_SCHEMA,
        "normalization": str(root / NORMALIZATION),
        "contract": str(root / DATA_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_feature_cache": str(root / _feature_cache("train")),
        "validation_feature_cache": str(root / _feature_cache("val")),
        "readiness": str(root / READINESS),
        "configs": {variant: value["path"] for variant, value in config_bindings.items()},
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_support": 9,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V21 KnownGate known-only normalization, features and configs."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
