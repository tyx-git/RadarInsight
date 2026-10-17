"""Frozen mixed Known/Unknown L1->L2->L3 audit contract for V26j.

This module is intentionally inference-only.  It binds V26j's causal-GRU L1
fold checkpoints, V26c's frozen L2 checkpoints and V26j's cross-fitted L3
thresholds.  Unknown is never a model class or a fitting population.
"""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .known_gate_v26c_oof import (
    CONTRACT as V26C_CONTRACT,
    CONTRACT_SCHEMA as V26C_CONTRACT_SCHEMA,
    FOLD_COUNT as V26C_FOLD_COUNT,
    L2_EPOCH as V26C_L2_EPOCH,
    READINESS as V26C_READINESS,
    READINESS_SCHEMA as V26C_READINESS_SCHEMA,
    l2_checkpoint_path as v26c_l2_checkpoint_path,
    l2_config_path as v26c_l2_config_path,
    l2_data_contract_path as v26c_l2_data_contract_path,
    l2_held_out_micro_cache_path as v26c_l2_held_out_micro_cache_path,
    l2_held_out_physical_cache_path as v26c_l2_held_out_physical_cache_path,
    l2_normalization_path as v26c_l2_normalization_path,
    l2_prediction_path as v26c_l2_prediction_path,
    l2_train_micro_cache_path as v26c_l2_train_micro_cache_path,
    l2_train_physical_cache_path as v26c_l2_train_physical_cache_path,
    l2_training_summary_path as v26c_l2_training_summary_path,
)
from .known_gate_v26j_oof import (
    CONTRACT as V26J_CONTRACT,
    CONTRACT_SCHEMA as V26J_CONTRACT_SCHEMA,
    L1_EPOCH as V26J_L1_EPOCH,
    L3_REPORT_SCHEMA as V26J_L3_REPORT_SCHEMA,
    READINESS as V26J_READINESS,
    READINESS_SCHEMA as V26J_READINESS_SCHEMA,
    l1_checkpoint_path as v26j_l1_checkpoint_path,
    l1_config_path as v26j_l1_config_path,
    l1_data_contract_path as v26j_l1_data_contract_path,
    l1_held_out_feature_cache_path as v26j_l1_held_out_feature_cache_path,
    l1_normalization_path as v26j_l1_normalization_path,
    l1_prediction_path as v26j_l1_prediction_path,
    l1_train_feature_cache_path as v26j_l1_train_feature_cache_path,
    l1_training_summary_path as v26j_l1_training_summary_path,
)
from .known_gate_v26e_oof import (
    CONTRACT as V26E_CONTRACT,
    READINESS as V26E_READINESS,
)


FOLD_COUNT = int(V26C_FOLD_COUNT)
INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
V26B_CACHE_MANIFEST = Path("data/processed/RADAR-LN3-V26b-cq08-l2-mixed-open-set-feature-cache-manifest.json")
V26J_REPORT = Path("src-v3/processed/report/RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-cross-fitted-l3-calibration.json")
L1_VALIDATION_CACHE = Path("data/processed/RADAR-LN3-V26-cq08-known-gate-val-mixed-open-set-course-cache.npz")
L1_TEST_CACHE = Path("data/processed/RADAR-LN3-V26-cq08-known-gate-test-mixed-open-set-course-cache.npz")
L2_VALIDATION_PHYSICAL_CACHE = Path("data/processed/RADAR-LN3-V26b-cq08-l2-val-mixed-open-set-rd-physical4-cache.npz")
L2_VALIDATION_MICRO_CACHE = Path("data/processed/RADAR-LN3-V26b-cq08-l2-val-mixed-open-set-micro-doppler8-cache.npz")
L2_TEST_PHYSICAL_CACHE = Path("data/processed/RADAR-LN3-V26b-cq08-l2-test-mixed-open-set-rd-physical4-cache.npz")
L2_TEST_MICRO_CACHE = Path("data/processed/RADAR-LN3-V26b-cq08-l2-test-mixed-open-set-micro-doppler8-cache.npz")
CONTRACT = Path("data/processed/RADAR-LN3-V26j-cq08-frozen-mixed-open-set-audit-contract.json")
READINESS = Path("src-v3/processed/report/RADAR-LN3-V26j-frozen-mixed-open-set-audit-readiness.json")
REPORT = Path("src-v3/processed/report/RADAR-LN3-V26j-frozen-mixed-open-set-audit.json")
PARENT_DETAIL_CSV = Path("src-v3/processed/report/RADAR-LN3-V26j-frozen-mixed-open-set-audit-parent-details.csv")
DOCUMENT = Path("src-v3/docs/radar_ln3_v26j_rd_causal_gru_oof.md")
CONTRACT_SCHEMA = "RADAR-LN3-V26j-frozen-mixed-open-set-audit-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26j-frozen-mixed-open-set-audit-readiness-v1"
REPORT_SCHEMA = "RADAR-LN3-V26j-frozen-mixed-open-set-audit-v1"
L1_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
DBB_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
MIXED_LABELS = (*L1_LABELS, "UnknownTarget")
ACTION_NAMES = (*DBB_LABELS, "ClutterTarget", "UNKNOWN_REJECTED", "PENDING_ABSTAIN")
L1_EPOCH = int(V26J_L1_EPOCH)
L2_EPOCH = int(V26C_L2_EPOCH)
EXPECTED_PARENT_COUNTS = {"val": {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54, "ClutterTarget": 9, "UnknownTarget": 15}, "test": {"DroneTarget": 109, "BirdTarget": 53, "BalloonTarget": 58, "ClutterTarget": 8, "UnknownTarget": 16}}
BOUND_FILE_MAX_BYTES = 64 * 1024 * 1024
FEATURE_CACHE_MAX_BYTES = 64 * 1024 * 1024
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_DETAIL_BYTES = 4 * 1024 * 1024
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 260710
IMPLEMENTATION_FILES = (Path("experiments/V26/V26j/radar_ln3/known_gate_v26j_audit.py"), Path("experiments/V26/V26j/report/known_gate_v26j_audit.py"), Path("experiments/V26/V26j/build/known_gate_v26j_audit.py"), Path("experiments/V26/V26j/scripts/v26j_known_gate_rd_causal_gru_oof_pipeline.ps1"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False)); digest.update(encoded)
    return digest.hexdigest()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def relative_path(root: Path, path: str | Path) -> str:
    return resolve_path(root, path).relative_to(root.resolve()).as_posix()


def bound_path(root: Path, value: str | Path, description: str) -> Path:
    path = resolve_path(root, value)
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"V26j {description} escapes project root: {value}") from exc
    return path


def require_bound_file(root: Path, value: str | Path, description: str, *, expected_sha256: str | None = None, max_bytes: int = BOUND_FILE_MAX_BYTES) -> Path:
    path = bound_path(root, value, description)
    if not path.is_file(): raise FileNotFoundError(f"V26j {description} missing: {path}")
    if path.stat().st_size > int(max_bytes): raise MemoryError(f"V26j {description} exceeds bounded size: {path}")
    if expected_sha256 is not None and sha256_file(path) != str(expected_sha256): raise ValueError(f"V26j {description} hash differs: {path}")
    return path


def read_json(path: str | Path, *, description: str = "JSON artifact") -> dict[str, Any]:
    checked = Path(path)
    if not checked.is_file() or checked.stat().st_size > BOUND_FILE_MAX_BYTES: raise ValueError(f"V26j {description} is unavailable or oversized: {checked}")
    value = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(value, dict): raise ValueError(f"V26j {description} must be a JSON object")
    return value


def write_immutable_text(path: Path, text: str, *, max_bytes: int, description: str) -> str:
    if len(text.encode("utf-8")) > int(max_bytes): raise MemoryError(f"V26j {description} exceeds bounded size")
    if path.exists():
        if path.read_text(encoding="utf-8") != text: raise FileExistsError(f"refusing to overwrite different V26j {description}: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists() and temporary.read_text(encoding="utf-8") != text: raise FileExistsError(f"refusing to replace different V26j temporary {description}")
    if not temporary.exists(): temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path); return "created"


def write_immutable_json(path: Path, payload: Mapping[str, Any], *, max_bytes: int, description: str) -> str:
    return write_immutable_text(path, json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n", max_bytes=max_bytes, description=description)


def implementation_sha256(root: Path) -> dict[str, str]:
    return {relative_path(root, require_bound_file(root, path, f"implementation {path}")): sha256_file(root / path) for path in IMPLEMENTATION_FILES}


def _preflight_npz(path: Path, *, max_bytes: int) -> None:
    if not path.is_file() or path.suffix.lower() != ".npz":
        raise ValueError(f"V26j cache is missing or not an npz archive: {path}")
    if path.stat().st_size > int(max_bytes):
        raise MemoryError(f"V26j cache exceeds the bounded size: {path}")
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            if len(members) > 16:
                raise ValueError(f"V26j cache has too many archive members: {path}")
            declared_size = 0
            for member in members:
                if member.is_dir() or member.filename.replace("\\", "/").endswith("/"):
                    raise ValueError(f"V26j cache contains a directory member: {path}")
                declared_size += int(member.file_size)
            if declared_size > int(max_bytes):
                raise MemoryError(f"V26j cache declared size exceeds the bound: {path}")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"V26j cache is not npz: {path}") from exc


def validate_feature_cache(path: Path, *, expected_sample_ids: Sequence[str], width: int, valid_rank: int, description: str) -> dict[str, Any]:
    _preflight_npz(path, max_bytes=FEATURE_CACHE_MAX_BYTES)
    expected = [str(value) for value in expected_sample_ids]
    if not expected or len(expected) != len(set(expected)):
        raise ValueError(f"V26j {description} expected sample ids are empty or repeated")
    if int(valid_rank) not in {1, 2}:
        raise ValueError(f"V26j {description} validity rank is unsupported")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V26j {description} fields differ")
        raw_ids = np.asarray(archive["sample_ids"])
        values = np.asarray(archive["values"])
        valid = np.asarray(archive["valid"])
    identifiers = [str(value) for value in raw_ids.tolist()]
    expected_valid_shape = (
        (len(expected), int(width)) if int(valid_rank) == 2 else (len(expected),)
    )
    if (
        raw_ids.ndim != 1
        or raw_ids.dtype.kind not in {"U", "S"}
        or identifiers != expected
        or len(identifiers) != len(set(identifiers))
        or values.dtype != np.float32
        or values.shape != (len(expected), int(width))
        or valid.dtype != np.bool_
        or valid.shape != expected_valid_shape
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(f"V26j {description} shape/order/value differs")
    invalid_zero = (
        np.all(values[~valid] == 0.0)
        if int(valid_rank) == 2
        else np.all(values[~valid, :] == 0.0)
    )
    if not bool(invalid_zero):
        raise ValueError(f"V26j {description} invalid values are nonzero")
    valid_events = int(valid.all(axis=1).sum()) if int(valid_rank) == 2 else int(valid.sum())
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "event_count": len(expected),
        "feature_width": int(width),
        "valid_event_count": valid_events,
        "invalid_event_count": len(expected) - valid_events,
        "selected_sample_ids_sha256": sequence_sha256(expected),
    }


def selected_event_sample_ids(root: Path, *, split: str) -> list[str]:
    import csv
    if split not in EXPECTED_PARENT_COUNTS:
        raise ValueError(f"V26j unsupported mixed split: {split}")
    index_path = require_bound_file(root, INDEX, "source index")
    rows = []
    with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split_name")) == split and str(row.get("label_name")) in MIXED_LABELS:
                rows.append((str(row["parent_sample_id"]), int(row["causal_event_index"]), str(row["sample_id"]), str(row["label_name"])))
    rows.sort()
    ids = [row[2] for row in rows]
    if (
        not ids
        or len(ids) != len(set(ids))
        or len({(row[0], row[1]) for row in rows}) != len(rows)
    ):
        raise ValueError(f"V26j {split} event identities differ")
    labels = {label: {parent for parent, _, _, raw_label in rows if raw_label == label} for label in MIXED_LABELS}
    if {label: len(parents) for label, parents in labels.items()} != EXPECTED_PARENT_COUNTS[split]:
        raise ValueError(f"V26j {split} parent support differs")
    return ids


def expected_parent_counts(root: Path, *, split: str) -> dict[str, int]:
    selected_event_sample_ids(root, split=split); return dict(EXPECTED_PARENT_COUNTS[split])


def fold_paths(fold_id: int) -> dict[str, Path]:
    if not 0 <= int(fold_id) < FOLD_COUNT: raise ValueError(f"invalid V26j fold: {fold_id}")
    return {"l1_config": v26j_l1_config_path(fold_id), "l1_data_contract": v26j_l1_data_contract_path(fold_id), "l1_normalization": v26j_l1_normalization_path(fold_id), "l1_train_cache": v26j_l1_train_feature_cache_path(fold_id), "l1_held_out_cache": v26j_l1_held_out_feature_cache_path(fold_id), "l1_checkpoint": v26j_l1_checkpoint_path(fold_id), "l1_prediction": v26j_l1_prediction_path(fold_id), "l1_summary": v26j_l1_training_summary_path(fold_id), "l2_config": v26c_l2_config_path(fold_id), "l2_data_contract": v26c_l2_data_contract_path(fold_id), "l2_normalization": v26c_l2_normalization_path(fold_id), "l2_train_physical_cache": v26c_l2_train_physical_cache_path(fold_id), "l2_held_out_physical_cache": v26c_l2_held_out_physical_cache_path(fold_id), "l2_train_micro_cache": v26c_l2_train_micro_cache_path(fold_id), "l2_held_out_micro_cache": v26c_l2_held_out_micro_cache_path(fold_id), "l2_checkpoint": v26c_l2_checkpoint_path(fold_id), "l2_prediction": v26c_l2_prediction_path(fold_id), "l2_summary": v26c_l2_training_summary_path(fold_id)}


def source_binding(root: Path, path: str | Path, *, description: str) -> dict[str, Any]:
    checked = require_bound_file(root, path, description); return {"path": relative_path(root, checked), "sha256": sha256_file(checked), "size_bytes": int(checked.stat().st_size)}


def validate_prepared_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate every frozen input before mixed V26j inference.

    The audit must not trust only its top-level readiness bit.  It verifies the
    hashes and sizes of all source artifacts, mixed cache order/content, and
    every per-fold V26j L1/V26c L2 binding.  This makes a post-training audit
    fail closed if an OOF artifact or cache is replaced.
    """

    contract_path, readiness_path = root / CONTRACT, root / READINESS
    contract = read_json(contract_path, description="contract")
    readiness = read_json(readiness_path, description="readiness")
    checks = dict(readiness.get("checks") or {})
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
        or not checks
        or not all(value is True for value in checks.values())
        or contract.get("deployment_eligible") is not False
        or readiness.get("deployment_eligible") is not False
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_normalization_exposure", -1)) != 0
        or int(contract.get("unknown_checkpoint_selection_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or int(contract.get("unknown_oof_prediction_count", -1)) != 0
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
    ):
        raise ValueError("V26j mixed audit contract/readiness is invalid")
    if dict(contract.get("implementation_sha256") or {}) != implementation_sha256(root):
        raise ValueError("V26j audit implementation binding differs")

    expected_source_names = {
        "index",
        "v26b_cache_manifest",
        "v26j_contract",
        "v26j_readiness",
        "v26j_report",
        "v26c_contract",
        "v26c_readiness",
        "document",
    }
    source_bindings = {
        str(dict(row).get("name", "")): dict(row)
        for row in list(contract.get("source_bindings") or ())
    }
    if set(source_bindings) != expected_source_names:
        raise ValueError("V26j mixed audit source bindings differ")
    for name, binding in source_bindings.items():
        path = require_bound_file(
            root,
            str(binding.get("path", "")),
            f"V26j audit source binding {name}",
            expected_sha256=str(binding.get("sha256", "")),
        )
        if (
            relative_path(root, path) != str(binding.get("path", ""))
            or int(binding.get("size_bytes", -1)) != int(path.stat().st_size)
        ):
            raise ValueError(f"V26j audit source binding size/path differs: {name}")

    v26j_contract = read_json(root / V26J_CONTRACT, description="V26j L1 contract")
    v26j_readiness = read_json(root / V26J_READINESS, description="V26j L1 readiness")
    v26j_report = read_json(root / V26J_REPORT, description="V26j L3 report")
    v26c_contract = read_json(root / V26C_CONTRACT, description="V26c contract")
    v26c_readiness = read_json(root / V26C_READINESS, description="V26c readiness")
    v26j_folds = {
        int(dict(row).get("fold_id", -1)): dict(row)
        for row in list(v26j_contract.get("folds") or ())
    }
    if set(v26j_folds) != set(range(FOLD_COUNT)):
        raise ValueError("V26j source L1 fold contract coverage differs")
    direct_v26e = dict(v26j_contract.get("direct_v26e_protocol_reference") or {})
    if (
        v26j_contract.get("schema_version") != V26J_CONTRACT_SCHEMA
        or v26j_readiness.get("schema_version") != V26J_READINESS_SCHEMA
        or v26j_readiness.get("status") != "passed"
        or v26j_readiness.get("contract_sha256") != sha256_file(root / V26J_CONTRACT)
        or v26j_contract.get("single_variable", {}).get("field")
        != "model.rd_frame_aggregation"
        or v26j_contract.get("single_variable", {}).get("control") != "masked_mean"
        or v26j_contract.get("single_variable", {}).get("treatment") != "causal_gru"
        or int(v26j_contract.get("l1", {}).get("fixed_snapshot_epoch", -1))
        != L1_EPOCH
        or int(v26j_contract.get("unknown_training_exposure", -1)) != 0
        or int(v26j_contract.get("unknown_normalization_exposure", -1)) != 0
        or direct_v26e.get("contract_path") != relative_path(root, root / V26E_CONTRACT)
        or direct_v26e.get("contract_sha256") != sha256_file(root / V26E_CONTRACT)
        or direct_v26e.get("readiness_path") != relative_path(root, root / V26E_READINESS)
        or direct_v26e.get("readiness_sha256") != sha256_file(root / V26E_READINESS)
        or v26j_report.get("schema_version") != V26J_L3_REPORT_SCHEMA
        or v26j_report.get("status")
        != "completed_v26j_rd_causal_gru_epoch32_known_only_l3_diagnostic"
        or v26j_report.get("contract_sha256") != sha256_file(root / V26J_CONTRACT)
        or v26j_report.get("readiness_sha256") != sha256_file(root / V26J_READINESS)
        or v26j_report.get("mixed_unknown_audit") != "not_executed"
        or int(v26j_report.get("unknown_threshold_fit_exposure", -1)) != 0
        or v26c_contract.get("schema_version") != V26C_CONTRACT_SCHEMA
        or v26c_readiness.get("schema_version") != V26C_READINESS_SCHEMA
        or v26c_readiness.get("status") != "passed"
        or v26c_readiness.get("contract_sha256") != sha256_file(root / V26C_CONTRACT)
    ):
        raise ValueError("V26j mixed audit source semantics differ")

    data = dict(contract.get("data") or {})
    if (
        data.get("source_index") != relative_path(root, root / INDEX)
        or data.get("source_index_sha256") != sha256_file(root / INDEX)
        or tuple(data.get("mixed_labels") or ()) != MIXED_LABELS
        or dict(data.get("split_parent_counts") or {}) != EXPECTED_PARENT_COUNTS
    ):
        raise ValueError("V26j mixed audit data binding differs")
    cache_by_split = {
        str(dict(row).get("split", "")): dict(row)
        for row in list(data.get("cache_bindings") or ())
    }
    if set(cache_by_split) != {"val", "test"}:
        raise ValueError("V26j mixed audit cache bindings lack val/test")
    cache_specs = (
        ("l1_course", 1, 2),
        ("l2_rd_physical4", 4, 1),
        ("l2_micro_doppler8", 8, 1),
    )
    for split, cache_record in cache_by_split.items():
        if dict(cache_record.get("parent_counts") or {}) != EXPECTED_PARENT_COUNTS[split]:
            raise ValueError(f"V26j {split} mixed parent-count binding differs")
        expected_ids = selected_event_sample_ids(root, split=split)
        for name, width, valid_rank in cache_specs:
            binding = dict(cache_record.get(name) or {})
            path = require_bound_file(
                root,
                str(binding.get("path", "")),
                f"V26j {split} {name} cache",
                expected_sha256=str(binding.get("sha256", "")),
                max_bytes=FEATURE_CACHE_MAX_BYTES,
            )
            actual = validate_feature_cache(
                path,
                expected_sample_ids=expected_ids,
                width=width,
                valid_rank=valid_rank,
                description=f"{split} {name} cache",
            )
            if (
                relative_path(root, path) != str(binding.get("path", ""))
                or int(binding.get("size_bytes", -1)) != int(path.stat().st_size)
                or actual["sha256"] != binding.get("sha256")
                or int(actual["event_count"]) != int(binding.get("event_count", -1))
                or int(actual["feature_width"]) != int(binding.get("feature_width", -1))
                or actual["selected_sample_ids_sha256"]
                != binding.get("selected_sample_ids_sha256")
            ):
                raise ValueError(f"V26j {split} {name} cache binding differs")

    folds = list(contract.get("folds") or ())
    if {int(dict(row).get("fold_id", -1)) for row in folds} != set(range(FOLD_COUNT)):
        raise ValueError("V26j mixed audit fold bindings differ")
    expected_file_names = set(fold_paths(0))
    for raw in folds:
        fold = dict(raw)
        fold_id = int(fold["fold_id"])
        files = {
            str(dict(item).get("name", "")): dict(item)
            for item in list(fold.get("source_files") or ())
        }
        if set(files) != expected_file_names:
            raise ValueError(f"V26j fold {fold_id} source-file bindings differ")
        for name, binding in files.items():
            path = require_bound_file(
                root,
                str(binding.get("path", "")),
                f"V26j fold {fold_id} {name}",
                expected_sha256=str(binding.get("sha256", "")),
            )
            if (
                relative_path(root, path) != str(binding.get("path", ""))
                or int(binding.get("size_bytes", -1)) != int(path.stat().st_size)
            ):
                raise ValueError(f"V26j fold {fold_id} source-file size/path differs: {name}")
        source_fold = dict(fold.get("source_v26j_fold_contract") or {})
        expected_source_fold = v26j_folds[fold_id]
        runtime = dict(fold.get("rd_runtime") or {})
        if (
            source_fold.get("train_parent_ids_sha256")
            != expected_source_fold.get("train_parent_ids_sha256")
            or source_fold.get("held_out_parent_ids_sha256")
            != expected_source_fold.get("held_out_parent_ids_sha256")
            or int(source_fold.get("component_overlap_count", -1)) != 0
            or runtime.get("rd_frame_aggregation_config") != "causal_gru"
            or runtime.get("rd_frame_aggregation_runtime") != "causal_gru"
            or runtime.get("rd_temporal_module") != "GRUCell"
            or int(runtime.get("rd_temporal_parameter_count", 0)) <= 0
            or not list(runtime.get("rd_temporal_state_keys") or ())
        ):
            raise ValueError(f"V26j fold {fold_id} causal-GRU binding differs")
    return contract, readiness


def fold_contract_entry(contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    rows = [dict(row) for row in list(contract.get("folds") or ()) if int(dict(row).get("fold_id", -1)) == int(fold_id)]
    if len(rows) != 1: raise ValueError(f"V26j mixed audit fold missing/repeated: {fold_id}")
    return rows[0]

