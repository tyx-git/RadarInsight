"""Prepare the V26e fixed-epoch-32 L1 Known-only OOF experiment.

V26e deliberately reuses V26c's validated fold assignment, fold-train
normalization, feature cache values, and L2 epoch-15 outputs.  Only the L1
training horizon and its checkpoint namespace are new.  No validation, test,
or Unknown payload is opened during preparation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate_v26c_oof import (
    L2_PREDICTION_SCHEMA as V26C_L2_PREDICTION_SCHEMA,
    L2_EPOCH as V26C_L2_EPOCH,
    CONTRACT as V26C_CONTRACT,
    READINESS as V26C_READINESS,
    V21E_ASSIGNMENT,
    V21E_FOLD_CONTRACT,
    l1_checkpoint_path as v26c_l1_checkpoint_path,
    l1_prediction_path as v26c_l1_prediction_path,
    l1_training_summary_path as v26c_l1_training_summary_path,
    fold_record as v26c_fold_record,
)
from radar_ln3.known_gate_v26e_oof import CONTRACT, CONTRACT_SCHEMA, FOLD_COUNT, L1_CHECKPOINT_SCHEMA, L1_EPOCH, READINESS, READINESS_SCHEMA, TRAINING_SEED_L1, TRAINING_SEED_L2, l1_checkpoint_path, l1_config_path, l1_data_contract_path, l1_last_checkpoint_path, l2_checkpoint_path, l2_prediction_path, l2_training_summary_path, relative_path, read_feature_archive, sequence_sha256, sha256_file, selected_event_sample_ids, validate_v26c_source_payload, write_new_or_identical


INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
L1_CACHE_MAX_BYTES = 64 * 1024 * 1024
L2_CACHE_MAX_BYTES = 64 * 1024 * 1024
EXPECTED_KNOWN_PARENT_COUNT = 986
EXPECTED_UNKNOWN_TRAIN_PARENT_COUNT = 69
L1_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
L2_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
V26C_L1_CONFIG_SCHEMA = "RADAR-LN3-V26c-OOF-L1"
V26E_L1_CONFIG_SCHEMA = "RADAR-LN3-V26e-OOF-L1"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"V26e expected a JSON object: {path}")
    return payload


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"V26e {description} is missing: {path}")


def _check_bounded_file(
    path: Path, description: str, *, max_bytes: int = L1_CACHE_MAX_BYTES
) -> None:
    _require_file(path, description)
    if path.stat().st_size > int(max_bytes):
        raise MemoryError(f"V26e {description} exceeds the bounded size: {path}")


def _bound_path(root: Path, value: str | Path, description: str) -> Path:
    """Resolve a contract path while refusing paths outside the project."""

    raw = Path(str(value).replace("\\", "/"))
    resolved = raw.resolve() if raw.is_absolute() else (root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"V26e {description} escapes the project root: {value}") from exc
    return resolved


def _selected_event_sample_ids(root: Path, parent_ids: Sequence[str]) -> list[str]:
    return selected_event_sample_ids(root, parent_ids, labels=L1_LABELS)


def _validate_normalization(
    root: Path,
    source_fold: Mapping[str, Any],
    normalization_path: Path,
) -> None:
    payload = _read_json(normalization_path)
    train_ids = sorted(str(value) for value in source_fold["train_parent_ids"])
    train_counts = dict(dict(source_fold["l1"]).get("train_label_parent_counts") or {})
    held_counts = dict(dict(source_fold["l1"]).get("held_out_label_parent_counts") or {})
    expected_excluded = {label: int(held_counts[label]) for label in L1_LABELS}
    expected_excluded["UnknownTarget"] = EXPECTED_UNKNOWN_TRAIN_PARENT_COUNT
    if (
        payload.get("schema_version") != "RADAR-LN3-V26c"
        or payload.get("source_index") != relative_path(root, root / INDEX)
        or payload.get("source_index_sha256") != sha256_file(root / INDEX)
        or payload.get("source_split") != "train"
        or int(payload.get("parent_count", -1)) != len(train_ids)
        or payload.get("selected_parent_ids_sha256") != sequence_sha256(train_ids)
        or payload.get("selected_label_parent_counts") != {
            label: int(train_counts[label]) for label in sorted(L1_LABELS)
        }
        or payload.get("allowed_labels") != sorted(L1_LABELS)
        or payload.get("forbidden_labels") != ["UnknownTarget"]
        or int(payload.get("allowed_parent_ids_count", -1)) != len(train_ids)
        or payload.get("allowed_parent_ids_sha256") != sequence_sha256(train_ids)
        or int(payload.get("excluded_parent_count", -1))
        != EXPECTED_KNOWN_PARENT_COUNT
        + EXPECTED_UNKNOWN_TRAIN_PARENT_COUNT
        - len(train_ids)
        or payload.get("excluded_label_parent_counts") != expected_excluded
        or payload.get("outputs", {}).get("normalization_stats_path")
        != relative_path(root, normalization_path)
    ):
        raise ValueError("V26e fold-train normalization metadata is stale or inconsistent")


def _validate_feature_cache(
    root: Path,
    record: Mapping[str, Any],
    *,
    expected_parent_ids: Sequence[str],
    all_known_parent_ids: Sequence[str],
    description: str,
) -> dict[str, Any]:
    path = _bound_path(root, str(record.get("path", "")), f"{description} path")
    _check_bounded_file(path, description)
    expected_ids = _selected_event_sample_ids(root, expected_parent_ids)
    expected_width = int(record.get("feature_width", -1))
    if expected_width != 1:
        raise ValueError(f"V26e {description} feature width differs")
    sample_ids, values, valid = read_feature_archive(
        path,
        expected_sample_ids=expected_ids,
        expected_width=expected_width,
        valid_rank=2,
        description=description,
        max_bytes=L1_CACHE_MAX_BYTES,
    )
    valid_count = int(valid.all(axis=1).sum())
    if (
        int(record.get("event_count", -1)) != len(expected_ids)
        or int(record.get("valid_event_count", -1)) != valid_count
        or int(record.get("invalid_event_count", -1))
        != len(expected_ids) - valid_count
        or record.get("selected_sample_ids_sha256") != sequence_sha256(expected_ids)
        or record.get("source_index") != relative_path(root, root / INDEX)
        or record.get("source_index_sha256") != sha256_file(root / INDEX)
    ):
        raise ValueError(f"V26e {description} metadata differs from the cache contents")
    for field in ("source_rebuild_cache", "source_rebuild_cache_sha256"):
        if not record.get(field):
            raise ValueError(f"V26e {description} lacks source rebuild provenance: {field}")
    source_rebuild = _bound_path(
        root, str(record["source_rebuild_cache"]), f"{description} source rebuild cache"
    )
    _check_bounded_file(source_rebuild, f"{description} source rebuild cache")
    if sha256_file(source_rebuild) != str(record["source_rebuild_cache_sha256"]):
        raise ValueError(f"V26e {description} source rebuild cache hash differs")
    if (
        record.get("source_rebuild_cache_value_reuse") is not True
        or record.get("source_rebuild_cache_is_label_free_over_all_known_train_parents") is not True
    ):
        raise ValueError(f"V26e {description} source rebuild provenance is not label-free")
    all_known_event_ids = _selected_event_sample_ids(root, all_known_parent_ids)
    source_ids, source_values, source_valid = read_feature_archive(
        source_rebuild,
        expected_sample_ids=all_known_event_ids,
        expected_width=expected_width,
        valid_rank=2,
        description=f"{description} source rebuild cache",
        max_bytes=L1_CACHE_MAX_BYTES,
    )
    source_index = {sample_id: index for index, sample_id in enumerate(source_ids)}
    selected_indices = [source_index[sample_id] for sample_id in expected_ids]
    if not np.array_equal(values, source_values[selected_indices]) or not np.array_equal(
        valid, source_valid[selected_indices]
    ):
        raise ValueError(f"V26e {description} is not an exact source-cache slice")
    return dict(record)


def _validate_frozen_assignment(
    root: Path, source_contract: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    """Validate the exact V21e assignment that V26c claims to reuse.

    V26e must not silently inherit a changed assignment merely because the
    parent lists in an old V26c contract still look plausible.  Bind both the
    path and the byte hash, then verify the fold membership independently.
    """

    binding = dict(source_contract.get("assignment_reuse") or {})
    assignment_path = root / V21E_ASSIGNMENT
    fold_contract_path = root / V21E_FOLD_CONTRACT
    _require_file(assignment_path, "V21e assignment")
    _require_file(fold_contract_path, "V21e fold contract")
    if (
        binding.get("source") != relative_path(root, assignment_path)
        or binding.get("sha256") != sha256_file(assignment_path)
        or binding.get("fold_contract") != relative_path(root, fold_contract_path)
        or binding.get("fold_contract_sha256") != sha256_file(fold_contract_path)
    ):
        raise ValueError("V26c source is not bound to the frozen V21e assignment bytes")
    fold_contract = _read_json(fold_contract_path)
    if (
        fold_contract.get("schema_version")
        != "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-contract-v1"
        or fold_contract.get("assignment_sha256") != sha256_file(assignment_path)
        or int(fold_contract.get("fold_count", -1)) != FOLD_COUNT
        or int(fold_contract.get("known_parent_count", -1)) != EXPECTED_KNOWN_PARENT_COUNT
        or fold_contract.get("grouping") != "capture_content_provenance_component"
        or int(fold_contract.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V26e frozen V21e fold contract is malformed")
    assignment = _read_json(assignment_path)
    rows = list(assignment.get("assignments") or ())
    result: dict[str, dict[str, Any]] = {}
    duplicate_parent_ids: set[str] = set()
    for raw_row in rows:
        row = dict(raw_row)
        parent_id = str(row.get("parent_sample_id") or "")
        if parent_id in result:
            duplicate_parent_ids.add(parent_id)
        result[parent_id] = row
    if (
        assignment.get("schema_version")
        != "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-assignment-v1"
        or assignment.get("grouping") != "capture_content_provenance_component"
        or int(assignment.get("known_parent_count", -1)) != EXPECTED_KNOWN_PARENT_COUNT
        or int(assignment.get("unknown_training_exposure", -1)) != 0
        or len(rows) != EXPECTED_KNOWN_PARENT_COUNT
        or len(result) != EXPECTED_KNOWN_PARENT_COUNT
        or duplicate_parent_ids
        or any(
            not parent_id
            or str(row.get("label_name")) not in L1_LABELS
            or not str(row.get("provenance_component_id"))
            or int(row.get("fold_id", -1)) not in range(FOLD_COUNT)
            for parent_id, row in result.items()
        )
    ):
        raise ValueError("V26e frozen V21e assignment is malformed")
    return result


def _require_v26c_source(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path = root / V26C_CONTRACT
    readiness_path = root / V26C_READINESS
    _require_file(contract_path, "V26c source contract")
    _require_file(readiness_path, "V26c source readiness")
    contract = _read_json(contract_path)
    readiness = _read_json(readiness_path)
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version")
        != "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-contract-v1"
        or readiness.get("schema_version")
        != "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness-v1"
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or int(contract.get("known_train_parent_count", -1))
        != EXPECTED_KNOWN_PARENT_COUNT
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
    ):
        raise ValueError("V26c source contract/readiness is invalid or stale")
    validated = validate_v26c_source_payload(
        root,
        contract_path,
        readiness_path,
        contract=contract,
        readiness=readiness,
    )
    return validated["contract"], validated["readiness"]


def _source_l1_config(root: Path, source_fold: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    binding = dict(source_fold.get("l1") or {})
    path = _bound_path(root, str(binding.get("config_path", "")), "V26c L1 config")
    _require_file(path, "V26c L1 source config")
    payload = _read_json(path)
    if (
        payload.get("schema_version") != V26C_L1_CONFIG_SCHEMA
        or payload.get("variant") != "v26_known_only_open_set_baseline"
        or int(dict(payload.get("v26c_oof") or {}).get("fold_id", -1))
        != int(source_fold["fold_id"])
        or int(dict(payload.get("v26c_oof") or {}).get("fixed_snapshot_epoch", -1)) != 14
        or dict(payload.get("v26c_oof") or {}).get("held_out_checkpoint_selection") is not False
    ):
        raise ValueError(f"V26c L1 source config is not the sealed epoch-14 protocol: {path}")
    if str(binding.get("config_sha256")) != sha256_file(path):
        raise ValueError(f"V26c L1 source config hash differs: {path}")
    return path, payload


def _validate_frozen_l1_data(
    root: Path,
    source_fold: Mapping[str, Any],
    *,
    all_known_parent_ids: Sequence[str],
) -> dict[str, Any]:
    binding = dict(source_fold.get("l1") or {})
    normalization_path = _bound_path(
        root, str(binding.get("normalization_path", "")), "V26c L1 normalization"
    )
    normalization_hash = str(binding.get("normalization_sha256", ""))
    _check_bounded_file(normalization_path, "V26c L1 normalization")
    if sha256_file(normalization_path) != normalization_hash:
        raise ValueError(f"V26c L1 normalization hash differs: {normalization_path}")
    _validate_normalization(root, source_fold, normalization_path)
    cache_records: dict[str, dict[str, Any]] = {}
    for name, parent_key in (
        ("train_course_cache", "train_parent_ids"),
        ("held_out_course_cache", "held_out_parent_ids"),
    ):
        record = dict(binding.get(name) or {})
        cache_records[name] = _validate_feature_cache(
            root,
            record,
            expected_parent_ids=source_fold[parent_key],
            all_known_parent_ids=all_known_parent_ids,
            description=f"V26c L1 {name}",
        )
    return {
        "normalization_path": str(binding["normalization_path"]),
        "normalization_sha256": normalization_hash,
        **cache_records,
    }


def _validate_l2_feature_cache(
    root: Path,
    record: Mapping[str, Any],
    *,
    expected_parent_ids: Sequence[str],
    source_parent_ids: Sequence[str],
    labels: Sequence[str],
    expected_width: int,
    description: str,
) -> dict[str, Any]:
    """Prove that one V26c L2 cache is an exact source-cache slice."""

    path = _bound_path(root, str(record.get("path", "")), f"{description} path")
    _check_bounded_file(path, description, max_bytes=L2_CACHE_MAX_BYTES)
    expected_ids = selected_event_sample_ids(root, expected_parent_ids, labels=labels)
    source_ids_expected = selected_event_sample_ids(
        root, source_parent_ids, labels=labels
    )
    sample_ids, values, valid = read_feature_archive(
        path,
        expected_sample_ids=expected_ids,
        expected_width=expected_width,
        valid_rank=1,
        description=description,
        max_bytes=L2_CACHE_MAX_BYTES,
    )
    valid_count = int(valid.sum())
    if (
        int(record.get("event_count", -1)) != len(expected_ids)
        or int(record.get("valid_event_count", -1)) != valid_count
        or int(record.get("invalid_event_count", -1))
        != len(expected_ids) - valid_count
        or record.get("selected_sample_ids_sha256") != sequence_sha256(expected_ids)
        or record.get("source_index") != relative_path(root, root / INDEX)
        or record.get("source_index_sha256") != sha256_file(root / INDEX)
        or record.get("feature_width") != expected_width
    ):
        raise ValueError(f"V26e {description} metadata differs from the cache contents")
    if (
        record.get("label_free_feature_value_extraction") is not True
        or record.get("source_rebuild_cache_value_reuse") is not True
        or record.get("source_rebuild_cache_is_label_free_over_all_known_train_parents")
        is not True
    ):
        raise ValueError(f"V26e {description} is not marked as label-free source reuse")
    source_rebuild = _bound_path(
        root,
        str(record.get("source_rebuild_cache", "")),
        f"{description} source rebuild cache",
    )
    _check_bounded_file(
        source_rebuild,
        f"{description} source rebuild cache",
        max_bytes=L2_CACHE_MAX_BYTES,
    )
    if sha256_file(source_rebuild) != str(record.get("source_rebuild_cache_sha256", "")):
        raise ValueError(f"V26e {description} source rebuild cache hash differs")
    source_ids, source_values, source_valid = read_feature_archive(
        source_rebuild,
        expected_sample_ids=source_ids_expected,
        expected_width=expected_width,
        valid_rank=1,
        description=f"{description} source rebuild cache",
        max_bytes=L2_CACHE_MAX_BYTES,
    )
    source_index = {sample_id: index for index, sample_id in enumerate(source_ids)}
    selected_indices = [source_index[sample_id] for sample_id in expected_ids]
    if not np.array_equal(values, source_values[selected_indices]) or not np.array_equal(
        valid, source_valid[selected_indices]
    ):
        raise ValueError(f"V26e {description} is not an exact source-cache slice")
    return dict(record)


def _validate_frozen_l2(
    root: Path,
    source_fold: Mapping[str, Any],
    *,
    assignment: Mapping[str, Mapping[str, Any]],
    all_known_parent_ids: Sequence[str],
    all_dbb_parent_ids: Sequence[str],
    source_contract: Mapping[str, Any],
    source_contract_path: Path,
    source_readiness_path: Path,
) -> dict[str, Any]:
    fold_id = int(source_fold["fold_id"])
    binding = dict(source_fold.get("l2") or {})
    train_parent_ids = sorted(str(value) for value in source_fold["l2"]["train_parent_ids"])
    held_out_parent_ids = sorted(
        str(value) for value in source_fold["l2"]["held_out_parent_ids"]
    )
    all_dbb_parent_ids = sorted(str(value) for value in all_dbb_parent_ids)
    all_known_parent_ids = sorted(str(value) for value in all_known_parent_ids)
    if (
        not train_parent_ids
        or not held_out_parent_ids
        or len(train_parent_ids) != len(set(train_parent_ids))
        or len(held_out_parent_ids) != len(set(held_out_parent_ids))
        or set(train_parent_ids) & set(held_out_parent_ids)
        or set(train_parent_ids) | set(held_out_parent_ids) != set(all_dbb_parent_ids)
        or len(all_dbb_parent_ids) != 953
    ):
        raise ValueError(f"V26c L2 fold {fold_id} parent partition is invalid")
    if set(all_dbb_parent_ids) - set(all_known_parent_ids):
        raise ValueError(f"V26c L2 fold {fold_id} is not a Known-parent subset")
    expected_train_parent_ids = sorted(
        parent_id
        for parent_id, row in assignment.items()
        if int(row["fold_id"]) != fold_id and str(row["label_name"]) in L2_LABELS
    )
    expected_held_out_parent_ids = sorted(
        parent_id
        for parent_id, row in assignment.items()
        if int(row["fold_id"]) == fold_id and str(row["label_name"]) in L2_LABELS
    )
    if (
        train_parent_ids != expected_train_parent_ids
        or held_out_parent_ids != expected_held_out_parent_ids
    ):
        raise ValueError(f"V26c L2 fold {fold_id} is not bound to the frozen assignment")
    source_config = _bound_path(
        root, str(binding.get("config_path", "")), f"V26c L2 fold {fold_id} config"
    )
    source_data_contract = _bound_path(
        root,
        str(binding.get("data_contract_path", "")),
        f"V26c L2 fold {fold_id} data contract",
    )
    source_normalization = _bound_path(
        root,
        str(binding.get("normalization_path", "")),
        f"V26c L2 fold {fold_id} normalization",
    )
    for path, expected_hash, description in (
        (source_config, binding.get("config_sha256"), "V26c L2 config"),
        (source_data_contract, binding.get("data_contract_sha256"), "V26c L2 data contract"),
        (source_normalization, binding.get("normalization_sha256"), "V26c L2 normalization"),
    ):
        _check_bounded_file(
            path, f"{description} fold {fold_id}", max_bytes=L2_CACHE_MAX_BYTES
        )
        if sha256_file(path) != str(expected_hash):
            raise ValueError(f"{description} hash differs for fold {fold_id}")
    source_config_payload = _read_json(source_config)
    source_declaration = dict(source_config_payload.get("v26c_oof") or {})
    source_data = dict(source_config_payload.get("data") or {})
    config_cache_pairs = (
        (
            "train_rd_physical_cache_path",
            "expected_train_rd_physical_cache_sha256",
            "train_rd_physical_cache",
        ),
        (
            "validation_rd_physical_cache_path",
            "expected_validation_rd_physical_cache_sha256",
            "held_out_rd_physical_cache",
        ),
        (
            "train_micro_doppler_cache_path",
            "expected_train_micro_doppler_cache_sha256",
            "train_micro_doppler_cache",
        ),
        (
            "validation_micro_doppler_cache_path",
            "expected_validation_micro_doppler_cache_sha256",
            "held_out_micro_doppler_cache",
        ),
    )
    if (
        source_config_payload.get("schema_version") != "RADAR-LN3-V26c-OOF-L2"
        or source_config_payload.get("task", {}).get("name") != "dbb_specialist"
        or int(source_declaration.get("fold_id", -1)) != fold_id
        or int(source_declaration.get("fixed_snapshot_epoch", -1)) != V26C_L2_EPOCH
        or source_declaration.get("held_out_checkpoint_selection") is not False
        or source_declaration.get("unknown_training_exposure") != 0
        or source_data.get("contract_path") != relative_path(root, source_data_contract)
        or source_data.get("expected_contract_sha256") != sha256_file(source_data_contract)
        or source_data.get("normalization_stats_path") != relative_path(root, source_normalization)
        or source_data.get("expected_stats_sha256") != sha256_file(source_normalization)
        or any(
            source_data.get(config_key)
            != relative_path(root, _bound_path(root, binding[record_key]["path"], record_key))
            or source_data.get(expected_hash_key) != binding[record_key].get("sha256")
            for config_key, expected_hash_key, record_key in config_cache_pairs
        )
    ):
        raise ValueError(f"V26c L2 fold config/cache binding differs for fold {fold_id}")

    source_data_payload = _read_json(source_data_contract)
    if (
        source_data_payload.get("schema_version")
        != "RADAR-LN3-V26c-dbb_specialist-fold-data-contract-v1"
        or source_data_payload.get("task_name") != "dbb_specialist"
        or int(source_data_payload.get("fold_id", -1)) != fold_id
        or source_data_payload.get("fit_split") != "train"
        or source_data_payload.get("fit_population") != "known_fold_train_only"
        or source_data_payload.get("grouping")
        != "capture_content_provenance_component"
        or int(source_data_payload.get("train_parent_count", -1))
        != len(train_parent_ids)
        or int(source_data_payload.get("held_out_parent_count", -1))
        != len(held_out_parent_ids)
        or source_data_payload.get("train_parent_ids_sha256")
        != sequence_sha256(train_parent_ids)
        or source_data_payload.get("held_out_parent_ids_sha256")
        != sequence_sha256(held_out_parent_ids)
        or source_data_payload.get("normalization", {}).get("path")
        != relative_path(root, source_normalization)
        or source_data_payload.get("normalization", {}).get("sha256")
        != sha256_file(source_normalization)
        or source_data_payload.get("train_label_parent_counts")
        != dict(source_fold["l2"]["train_label_parent_counts"])
        or source_data_payload.get("held_out_label_parent_counts")
        != dict(source_fold["l2"]["held_out_label_parent_counts"])
        or int(source_data_payload.get("unknown_training_exposure", -1)) != 0
        or int(source_data_payload.get("unknown_normalization_exposure", -1)) != 0
        or int(source_data_payload.get("validation_access_count", -1)) != 0
        or int(source_data_payload.get("test_access_count", -1)) != 0
    ):
        raise ValueError(f"V26c L2 data contract is invalid for fold {fold_id}")

    cache_specs = (
        (
            "train_rd_physical_cache",
            "train",
            "rd_physical4",
            train_parent_ids,
            all_dbb_parent_ids,
            L2_LABELS,
            4,
            "V26c L2 train RD physical4",
        ),
        (
            "held_out_rd_physical_cache",
            "held_out",
            "rd_physical4",
            held_out_parent_ids,
            all_dbb_parent_ids,
            L2_LABELS,
            4,
            "V26c L2 held-out RD physical4",
        ),
        (
            "train_micro_doppler_cache",
            "train",
            "micro_doppler8",
            train_parent_ids,
            all_dbb_parent_ids,
            L2_LABELS,
            8,
            "V26c L2 train micro-Doppler8",
        ),
        (
            "held_out_micro_doppler_cache",
            "held_out",
            "micro_doppler8",
            held_out_parent_ids,
            all_dbb_parent_ids,
            L2_LABELS,
            8,
            "V26c L2 held-out micro-Doppler8",
        ),
        (
            "held_out_mixed_l1_inference_only_rd_physical_cache",
            "held_out",
            "mixed_l1_held_out_inference_only_rd_physical4",
            source_fold["held_out_parent_ids"],
            all_known_parent_ids,
            L1_LABELS,
            4,
            "V26c L2 mixed held-out RD physical4",
        ),
        (
            "held_out_mixed_l1_inference_only_micro_doppler_cache",
            "held_out",
            "mixed_l1_held_out_inference_only_micro_doppler8",
            source_fold["held_out_parent_ids"],
            all_known_parent_ids,
            L1_LABELS,
            8,
            "V26c L2 mixed held-out micro-Doppler8",
        ),
    )
    data_cache_reuse = dict(source_data_payload.get("cache_reuse") or {})
    for (
        binding_name,
        data_side,
        data_name,
        expected_parent_ids,
        source_parent_ids,
        labels,
        width,
        description,
    ) in cache_specs:
        record = dict(binding.get(binding_name) or {})
        _validate_l2_feature_cache(
            root,
            record,
            expected_parent_ids=expected_parent_ids,
            source_parent_ids=source_parent_ids,
            labels=labels,
            expected_width=width,
            description=f"{description} fold {fold_id}",
        )
        # Mixed caches are inference-only and intentionally absent from the
        # L2 train/held-out cache_reuse block.
        if not binding_name.startswith("held_out_mixed"):
            data_record = dict(
                dict(data_cache_reuse.get(data_side) or {}).get(data_name) or {}
            )
            if (
                data_record.get("path") != record.get("path")
                or data_record.get("sha256") != record.get("sha256")
                or int(data_record.get("event_count", -1))
                != int(record.get("event_count", -1))
                or int(data_record.get("feature_width", -1)) != width
                or data_record.get("valid_event_count")
                != record.get("valid_event_count")
                or data_record.get("invalid_event_count")
                != record.get("invalid_event_count")
                or data_record.get("selected_sample_ids_sha256")
                != record.get("selected_sample_ids_sha256")
            ):
                raise ValueError(
                    "V26c L2 data-contract cache binding differs: "
                    f"{binding_name} fold {fold_id}"
                )
        if binding_name.startswith("held_out_mixed") and (
            record.get("inference_only_mixed_l1_held_out_population") is not True
            or int(record.get("l2_training_exposure", -1)) != 0
            or int(record.get("l2_normalization_exposure", -1)) != 0
            or int(record.get("l3_threshold_fit_exposure", -1)) != 0
        ):
            raise ValueError(f"V26c mixed L2 cache is not inference-only: fold {fold_id}")

    checkpoint = _bound_path(
        root, l2_checkpoint_path(fold_id), f"V26c L2 fold {fold_id} checkpoint"
    )
    prediction = _bound_path(
        root, l2_prediction_path(fold_id), f"V26c L2 fold {fold_id} prediction"
    )
    summary = _bound_path(
        root, l2_training_summary_path(fold_id), f"V26c L2 fold {fold_id} summary"
    )
    for path, description in (
        (checkpoint, "V26c L2 epoch-15 checkpoint"),
        (prediction, "V26c L2 prediction"),
        (summary, "V26c L2 training summary"),
    ):
        _require_file(path, description)
    prediction_payload = _read_json(prediction)
    summary_payload = _read_json(summary)
    source_bindings = dict(source_contract.get("source_bindings") or {})
    expected_source_config_sha256 = dict(source_bindings.get("l2_base_config") or {}).get(
        "sha256"
    )
    if (
        prediction_payload.get("prediction_protocol") != "out_of_fold_fixed_epoch_15"
        or prediction_payload.get("fit_population") != "dbb_known_only"
        or prediction_payload.get("grouping") != "capture_content_provenance_component"
        or prediction_payload.get("config_path") != relative_path(root, source_config)
        or prediction_payload.get("config_sha256") != sha256_file(source_config)
        or prediction_payload.get("fold_contract_sha256") != sha256_file(source_contract_path)
        or prediction_payload.get("readiness_sha256") != sha256_file(source_readiness_path)
        or prediction_payload.get("source_config_sha256") != expected_source_config_sha256
        or prediction_payload.get("checkpoint_path") != relative_path(root, checkpoint)
        or prediction_payload.get("checkpoint_sha256") != sha256_file(checkpoint)
    ):
        raise ValueError(f"V26c L2 prediction provenance differs for fold {fold_id}")
    if (
        prediction_payload.get("schema_version") != V26C_L2_PREDICTION_SCHEMA
        or int(prediction_payload.get("fold_id", -1)) != fold_id
        or int(prediction_payload.get("fixed_snapshot_epoch", -1)) != V26C_L2_EPOCH
        or prediction_payload.get("held_out_checkpoint_selection") is not False
        or int(prediction_payload.get("unknown_training_exposure", -1)) != 0
        or int(prediction_payload.get("validation_access_count", -1)) != 0
        or int(prediction_payload.get("test_payload_access_count", -1)) != 0
        or summary_payload.get("status") != "completed"
        or int(summary_payload.get("fixed_snapshot_epoch", -1)) != V26C_L2_EPOCH
        or summary_payload.get("prediction_sha256") != sha256_file(prediction)
        or summary_payload.get("prediction_path") != relative_path(root, prediction)
        or summary_payload.get("checkpoint_path") != relative_path(root, checkpoint)
        or summary_payload.get("checkpoint_sha256") != sha256_file(checkpoint)
    ):
        raise ValueError(f"V26c L2 reuse artifact is invalid for fold {fold_id}")
    joint_l1_binding = dict(prediction_payload.get("joint_l1_binding") or {})
    l1_config = _bound_path(
        root,
        str(source_fold["l1"].get("config_path", "")),
        f"V26c L1 fold {fold_id} config from L2 binding",
    )
    l1_checkpoint = _bound_path(
        root, v26c_l1_checkpoint_path(fold_id), f"V26c L1 fold {fold_id} checkpoint"
    )
    l1_prediction = _bound_path(
        root, v26c_l1_prediction_path(fold_id), f"V26c L1 fold {fold_id} prediction"
    )
    l1_summary = _bound_path(
        root,
        v26c_l1_training_summary_path(fold_id),
        f"V26c L1 fold {fold_id} summary",
    )
    for path, description in (
        (l1_config, "V26c L1 config"),
        (l1_checkpoint, "V26c L1 checkpoint"),
        (l1_prediction, "V26c L1 prediction"),
        (l1_summary, "V26c L1 summary"),
    ):
        _require_file(path, f"{description} used by V26c L2 fold {fold_id}")
    if (
        joint_l1_binding.get("config_path") != relative_path(root, l1_config)
        or joint_l1_binding.get("config_sha256") != sha256_file(l1_config)
        or joint_l1_binding.get("checkpoint_path") != relative_path(root, l1_checkpoint)
        or joint_l1_binding.get("checkpoint_sha256") != sha256_file(l1_checkpoint)
        or joint_l1_binding.get("prediction_path") != relative_path(root, l1_prediction)
        or joint_l1_binding.get("prediction_sha256") != sha256_file(l1_prediction)
        or joint_l1_binding.get("training_summary_path") != relative_path(root, l1_summary)
        or joint_l1_binding.get("training_summary_sha256") != sha256_file(l1_summary)
        or int(joint_l1_binding.get("l1_forward_count", -1)) != 0
    ):
        raise ValueError(f"V26c L2 joint L1 binding differs for fold {fold_id}")
    return {
        "checkpoint_path": relative_path(root, checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "prediction_path": relative_path(root, prediction),
        "prediction_sha256": sha256_file(prediction),
        "summary_path": relative_path(root, summary),
        "summary_sha256": sha256_file(summary),
        "source_contract_fold": binding,
    }


def _l1_data_contract(
    root: Path,
    *,
    source_contract_path: Path,
    source_fold: Mapping[str, Any],
    frozen_data: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    fold_id = int(source_fold["fold_id"])
    train_ids = [str(value) for value in source_fold["train_parent_ids"]]
    held_out_ids = [str(value) for value in source_fold["held_out_parent_ids"]]
    counts = dict(dict(source_fold["l1"]).get("train_label_parent_counts") or {})
    held_counts = dict(dict(source_fold["l1"]).get("held_out_label_parent_counts") or {})
    if set(counts) != set(L1_LABELS) or set(held_counts) != set(L1_LABELS):
        raise ValueError(f"V26c L1 fold {fold_id} label support is incomplete")
    return {
        "schema_version": "RADAR-LN3-V26e-known-gate-fold-data-contract-v1",
        "task_name": "known_gate",
        "fold_id": fold_id,
        "fit_split": "train",
        "fit_population": "known_fold_train_only",
        "grouping": "capture_content_provenance_component",
        "train_parent_count": len(train_ids),
        "held_out_parent_count": len(held_out_ids),
        "train_parent_ids_sha256": sequence_sha256(sorted(train_ids)),
        "held_out_parent_ids_sha256": sequence_sha256(sorted(held_out_ids)),
        "train_label_parent_counts": {label: int(counts[label]) for label in L1_LABELS},
        "held_out_label_parent_counts": {label: int(held_counts[label]) for label in L1_LABELS},
        "normalization": {
            "path": str(frozen_data["normalization_path"]),
            "sha256": str(frozen_data["normalization_sha256"]),
            "reused_from_v26c": True,
        },
        "cache_reuse": {
            "train": dict(frozen_data["train_course_cache"]),
            "held_out": dict(frozen_data["held_out_course_cache"]),
            "reused_from_v26c": True,
            "value_rebuild": False,
            "isolation": "each cache contains exactly one disjoint frozen V26c fold population",
        },
        "source_v26c_contract": relative_path(root, source_contract_path),
        "source_v26c_contract_sha256": sha256_file(source_contract_path),
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_cache_selection_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "checkpoint_selection": "fixed epoch-32 endpoint; held-out parent labels are never used",
        "l1_l2_fusion": "not_performed_in_fold_training",
    }


def _l1_config(
    root: Path,
    *,
    source_payload: Mapping[str, Any],
    source_config_path: Path,
    fold_id: int,
    l1_contract_path: Path,
    frozen_data: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(source_payload)))
    payload.pop("v26c_oof", None)
    payload["schema_version"] = V26E_L1_CONFIG_SCHEMA
    payload["config_name"] = (
        f"RADAR-LN3-V26e-cq08-known-gate-oof-fold-{fold_id:02d}-seed43-epoch32"
    )
    data = dict(payload["data"])
    data.update(
        {
            "contract_path": relative_path(root, l1_contract_path),
            "expected_contract_sha256": sha256_file(l1_contract_path),
            "normalization_stats_path": str(frozen_data["normalization_path"]),
            "expected_stats_sha256": str(frozen_data["normalization_sha256"]),
            "train_feature_cache_path": str(frozen_data["train_course_cache"]["path"]),
            "validation_feature_cache_path": str(frozen_data["held_out_course_cache"]["path"]),
            "expected_train_feature_cache_sha256": str(
                frozen_data["train_course_cache"]["sha256"]
            ),
            "expected_validation_feature_cache_sha256": str(
                frozen_data["held_out_course_cache"]["sha256"]
            ),
            # V26e binds the concrete fold caches directly.  The old V26
            # manifest/experiment-contract fields would describe a different
            # namespace and must not remain as active configuration claims.
            "feature_cache_manifest_path": None,
            "expected_feature_cache_manifest_sha256": None,
            "feature_experiment_contract_path": None,
            "expected_feature_experiment_contract_sha256": None,
        }
    )
    payload["data"] = data
    artifact = root / l1_checkpoint_path(fold_id).parent
    train = dict(payload["train"])
    train.update(
        {
            "epochs": L1_EPOCH,
            "early_stopping_patience": 0,
            "checkpoint_path": relative_path(root, root / l1_checkpoint_path(fold_id)),
            "last_checkpoint_path": relative_path(
                root, root / l1_last_checkpoint_path(fold_id)
            ),
            "formal_artifact_dir": relative_path(root, artifact),
            "epoch_checkpoint_dir": relative_path(root, artifact / "epochs"),
            "checkpoint_schema": L1_CHECKPOINT_SCHEMA,
            "save_every_epoch": False,
            # The custom V26e trainer constructs the sealed source model
            # explicitly.  This generated config is not a paired treatment
            # config, so it must not claim the historical V21 pair.
            "paired_baseline_config_path": None,
        }
    )
    payload["train"] = train
    payload["v26e_oof"] = {
        "fold_id": fold_id,
        "role": "fresh L1 OOF support only; not a deployment checkpoint",
        "train_epoch_limit": L1_EPOCH,
        "fixed_snapshot_epoch": L1_EPOCH,
        "checkpoint_selection": "fixed endpoint; no held-out selection",
        "source_v26c_l1_config_path": relative_path(root, source_config_path),
        "source_v26c_l1_config_sha256": sha256_file(source_config_path),
        "source_v26c_l1_training_epochs": 20,
        "unknown_training_exposure": 0,
        "held_out_checkpoint_selection": False,
    }
    return payload


def build(project_root: Path, *, preview: bool = False) -> dict[str, Any]:
    root = project_root.resolve()
    source_contract, source_readiness = _require_v26c_source(root)
    source_contract_path = root / V26C_CONTRACT
    assignment = _validate_frozen_assignment(root, source_contract)
    all_known_parent_ids = sorted(assignment)
    all_dbb_parent_ids = sorted(
        parent_id
        for parent_id, row in assignment.items()
        if str(row["label_name"]) in L2_LABELS
    )
    if len(all_known_parent_ids) != EXPECTED_KNOWN_PARENT_COUNT or len(all_dbb_parent_ids) != 953:
        raise ValueError("V26e frozen Known/DBB parent universes are incomplete")
    fold_records: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        source_fold = v26c_fold_record(source_contract, fold_id)
        expected_fold_ids = {
            parent_id
            for parent_id, row in assignment.items()
            if int(row["fold_id"]) == fold_id
        }
        if (
            set(source_fold["train_parent_ids"]) | set(source_fold["held_out_parent_ids"])
            != set(assignment)
            or set(source_fold["held_out_parent_ids"]) != expected_fold_ids
        ):
            raise ValueError(f"V26c fold {fold_id} differs from the frozen V21e assignment")
        source_config_path, source_config = _source_l1_config(root, source_fold)
        frozen_data = _validate_frozen_l1_data(
            root, source_fold, all_known_parent_ids=all_known_parent_ids
        )
        l2_reuse = _validate_frozen_l2(
            root,
            source_fold,
            assignment=assignment,
            all_known_parent_ids=all_known_parent_ids,
            all_dbb_parent_ids=all_dbb_parent_ids,
            source_contract=source_contract,
            source_contract_path=source_contract_path,
            source_readiness_path=root / V26C_READINESS,
        )
        if preview:
            fold_records.append(
                {
                    "fold_id": fold_id,
                    "l1_train_parent_count": len(source_fold["train_parent_ids"]),
                    "l1_held_out_parent_count": len(source_fold["held_out_parent_ids"]),
                    "l2_reused_epoch": V26C_L2_EPOCH,
                }
            )
            continue
        l1_contract_path = root / l1_data_contract_path(fold_id)
        l1_contract = _l1_data_contract(
            root,
            source_contract_path=source_contract_path,
            source_fold=source_fold,
            frozen_data=frozen_data,
            contract_path=l1_contract_path,
        )
        write_new_or_identical(l1_contract_path, l1_contract)
        l1_config_path_value = root / l1_config_path(fold_id)
        l1_config = _l1_config(
            root,
            source_payload=source_config,
            source_config_path=source_config_path,
            fold_id=fold_id,
            l1_contract_path=l1_contract_path,
            frozen_data=frozen_data,
        )
        write_new_or_identical(l1_config_path_value, l1_config)
        fold_records.append(
            {
                "fold_id": fold_id,
                "source_v26c_fold_contract_match": True,
                "train_parent_ids": list(source_fold["train_parent_ids"]),
                "held_out_parent_ids": list(source_fold["held_out_parent_ids"]),
                "train_parent_ids_sha256": sequence_sha256(
                    sorted(str(value) for value in source_fold["train_parent_ids"])
                ),
                "held_out_parent_ids_sha256": sequence_sha256(
                    sorted(str(value) for value in source_fold["held_out_parent_ids"])
                ),
                "train_component_count": int(source_fold["train_component_count"]),
                "held_out_component_count": int(source_fold["held_out_component_count"]),
                "component_overlap_count": 0,
                "l1": {
                    "train_parent_count": len(source_fold["train_parent_ids"]),
                    "held_out_parent_count": len(source_fold["held_out_parent_ids"]),
                    "train_label_parent_counts": dict(
                        source_fold["l1"]["train_label_parent_counts"]
                    ),
                    "held_out_label_parent_counts": dict(
                        source_fold["l1"]["held_out_label_parent_counts"]
                    ),
                    "config_path": relative_path(root, l1_config_path_value),
                    "config_sha256": sha256_file(l1_config_path_value),
                    "data_contract_path": relative_path(root, l1_contract_path),
                    "data_contract_sha256": sha256_file(l1_contract_path),
                    "normalization_path": str(frozen_data["normalization_path"]),
                    "normalization_sha256": str(frozen_data["normalization_sha256"]),
                    "train_course_cache": dict(frozen_data["train_course_cache"]),
                    "held_out_course_cache": dict(frozen_data["held_out_course_cache"]),
                    "train_epoch_limit": L1_EPOCH,
                    "fixed_snapshot_epoch": L1_EPOCH,
                    "held_out_checkpoint_selection": False,
                },
                "l2": {
                    **dict(source_fold["l2"]),
                    **l2_reuse,
                    "reused_from_v26c": True,
                    "fixed_snapshot_epoch": V26C_L2_EPOCH,
                    "held_out_checkpoint_selection": False,
                    "training_executed_in_v26e": False,
                },
            }
        )
    if preview:
        return {
            "status": "preview_passed",
            "read_only": True,
            "fold_count": FOLD_COUNT,
            "l1_train_epoch_limit": L1_EPOCH,
            "l1_fixed_snapshot_epoch": L1_EPOCH,
            "l2_reused_fixed_snapshot_epoch": V26C_L2_EPOCH,
            "folds": fold_records,
        }

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26e fixed epoch-32 L1 long-horizon Known-only provenance-grouped OOF",
        "scope": (
            "fresh fold-local L1 training to epoch 32; V26c L2 epoch-15 outputs "
            "reused read-only; no Unknown/validation/test threshold fitting"
        ),
        "source_v26c": {
            "contract_path": relative_path(root, source_contract_path),
            "contract_sha256": sha256_file(source_contract_path),
            "readiness_path": relative_path(root, root / V26C_READINESS),
            "readiness_sha256": sha256_file(root / V26C_READINESS),
            "l1_epoch14_checkpoint_not_reused": True,
            "l1_epoch20_checkpoint_not_reused": True,
            "l2_epoch15_outputs_reused_read_only": True,
        },
        "assignment_reuse": {
            "source": relative_path(root, root / V21E_ASSIGNMENT),
            "sha256": sha256_file(root / V21E_ASSIGNMENT),
            "source_contract": relative_path(root, root / V21E_FOLD_CONTRACT),
            "source_contract_sha256": sha256_file(root / V21E_FOLD_CONTRACT),
            "scope": "V26c frozen provenance-grouped assignment only",
        },
        "feature_and_normalization_reuse": {
            "source_v26c_values_reused_read_only": True,
            "feature_rebuild": False,
            "normalization_rebuild": False,
            "l1_cache_and_normalization_paths_are_v26c": True,
        },
        "fold_count": FOLD_COUNT,
        "known_train_parent_count": EXPECTED_KNOWN_PARENT_COUNT,
        "grouping": "capture_content_provenance_component",
        "l1": {
            "task": "known_gate",
            "training_seed": TRAINING_SEED_L1,
            "train_epoch_limit": L1_EPOCH,
            "fixed_snapshot_epoch": L1_EPOCH,
            "checkpoint_selection": "fixed endpoint; no per-fold held-out selection",
            "unknown_training_exposure": 0,
        },
        "l2": {
            "task": "dbb_specialist",
            "source": "V26c sealed fold L2 outputs",
            "training_seed": TRAINING_SEED_L2,
            "fixed_snapshot_epoch": V26C_L2_EPOCH,
            "training_executed_in_v26e": False,
            "unknown_training_exposure": 0,
        },
        "l3_calibration_protocol": {
            "fit_population": "Known parent-final OOF scores only",
            "outer_cross_fit": "thresholds for fold f are fitted with other four folds only",
            "unknown_threshold_fitting": False,
            "pending_is_unknown_rejection_success": False,
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
        "v26c_source_contract_and_readiness_are_frozen": True,
        "v26c_provenance_assignment_is_reused_byte_for_byte": True,
        "each_known_train_parent_is_held_out_once_for_l1": True,
        "no_provenance_component_crosses_a_fold": True,
        "l1_normalization_is_reused_from_v26c_fold_train_only": True,
        "l1_feature_caches_are_reused_from_v26c_without_value_rebuild": True,
        "unknown_is_excluded_from_training_normalization_cache_and_calibration": True,
        "l1_train_epoch_limit_is_32": True,
        "l1_fixed_endpoint_is_epoch_32": True,
        "l1_checkpoint_selection_is_not_data_driven": True,
        "v26c_l2_epoch15_outputs_are_reused_read_only": True,
        "validation_and_test_are_unread_during_prepare": True,
        "l3_thresholds_are_not_fitted_during_prepare": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": READINESS_SCHEMA,
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "checks": checks,
        "fold_count": FOLD_COUNT,
        "l1_train_epoch_limit": L1_EPOCH,
        "l1_fixed_snapshot_epoch": L1_EPOCH,
        "l2_reused_fixed_snapshot_epoch": V26C_L2_EPOCH,
        "known_train_parent_count": EXPECTED_KNOWN_PARENT_COUNT,
        "unknown_training_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "l3_calibration": "not_executed",
        "mixed_unknown_audit": "not_executed",
        "deployment_eligible": False,
    }
    write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "contract": str(root / CONTRACT),
        "readiness": str(root / READINESS),
        "fold_count": FOLD_COUNT,
        "l1_train_epoch_limit": L1_EPOCH,
        "l1_fixed_snapshot_epoch": L1_EPOCH,
        "l2_reused_fixed_snapshot_epoch": V26C_L2_EPOCH,
        "l1_runs": FOLD_COUNT,
        "l2_runs": 0,
        "l3_calibration": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V26e fixed-epoch-32 L1 Known-only OOF."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve(), preview=args.preview), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()