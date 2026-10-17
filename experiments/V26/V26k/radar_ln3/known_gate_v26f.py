"""Frozen mixed Known/Unknown audit contract for V26f.

V26f is an inference-only audit.  It does not train a model, fit a threshold,
or create a new feature/cache.  It binds the already sealed V26e L1
epoch-32 checkpoints, the V26c L2 epoch-15 checkpoints, and the V26e
cross-fitted thresholds, then permits a separate report process to run the
mixed CQ08 validation/test population through the real L1 -> L2 -> semantic
decision path.
"""

from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .known_gate_v26c_oof import FOLD_COUNT as V26C_FOLD_COUNT, l2_checkpoint_path as v26c_l2_checkpoint_path, l2_config_path as v26c_l2_config_path, l2_data_contract_path as v26c_l2_data_contract_path, l2_held_out_micro_cache_path as v26c_l2_held_out_micro_cache_path, l2_held_out_physical_cache_path as v26c_l2_held_out_physical_cache_path, l2_normalization_path as v26c_l2_normalization_path, l2_prediction_path as v26c_l2_prediction_path, l2_train_micro_cache_path as v26c_l2_train_micro_cache_path, l2_train_physical_cache_path as v26c_l2_train_physical_cache_path, l2_training_summary_path as v26c_l2_training_summary_path
from .known_gate_v26e_oof import l1_checkpoint_path as v26e_l1_checkpoint_path, l1_config_path as v26e_l1_config_path, l1_data_contract_path as v26e_l1_data_contract_path, l1_held_out_feature_cache_path as v26e_l1_held_out_feature_cache_path, l1_normalization_path as v26e_l1_normalization_path, l1_prediction_path as v26e_l1_prediction_path, l1_train_feature_cache_path as v26e_l1_train_feature_cache_path, l1_training_summary_path as v26e_l1_training_summary_path


FOLD_COUNT = int(V26C_FOLD_COUNT)
INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")

CONTRACT = Path(
    "data/processed/RADAR-LN3-V26f-cq08-frozen-mixed-open-set-audit-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V26f-frozen-mixed-open-set-audit-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V26f-frozen-mixed-open-set-audit.json"
)
PARENT_DETAIL_CSV = Path(
    "src-v3/processed/report/RADAR-LN3-V26f-frozen-mixed-open-set-audit-parent-details.csv"
)

CONTRACT_SCHEMA = "RADAR-LN3-V26f-frozen-mixed-open-set-audit-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26f-frozen-mixed-open-set-audit-readiness-v1"
REPORT_SCHEMA = "RADAR-LN3-V26f-frozen-mixed-open-set-audit-v1"

L1_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
DBB_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
MIXED_LABELS = (*L1_LABELS, "UnknownTarget")

EXPECTED_PARENT_COUNTS = {
    "val": {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
        "UnknownTarget": 15,
    },
    "test": {
        "DroneTarget": 109,
        "BirdTarget": 53,
        "BalloonTarget": 58,
        "ClutterTarget": 8,
        "UnknownTarget": 16,
    },
}

BOUND_FILE_MAX_BYTES = 64 * 1024 * 1024
FEATURE_CACHE_MAX_BYTES = 64 * 1024 * 1024

IMPLEMENTATION_FILES = (
    Path("experiments/V26/V26k/radar_ln3/known_gate_v26f.py"),
    Path(),
    Path(),
    Path(),
)


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
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def resolve_path(root: Path, value: str | Path) -> Path:
    raw_text = str(value).strip()
    if not raw_text:
        return root.resolve()
    raw = Path(raw_text.replace("\\", "/"))
    return (raw if raw.is_absolute() else root / raw).resolve()


def relative_path(root: Path, path: str | Path) -> str:
    root_path = Path(root).resolve()
    target = resolve_path(root_path, path)
    try:
        return target.relative_to(root_path).as_posix()
    except ValueError as exc:
        raise ValueError(f"V26f path is outside the project root: {path}") from exc


def bound_path(root: Path, value: str | Path, description: str) -> Path:
    path = resolve_path(root, value)
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ValueError(f"V26f {description} escapes the project root: {value}") from exc
    return path


def require_bound_file(
    root: Path,
    value: str | Path,
    description: str,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = BOUND_FILE_MAX_BYTES,
) -> Path:
    path = bound_path(root, value, description)
    if not path.is_file():
        raise FileNotFoundError(f"V26f {description} is missing: {path}")
    size = int(path.stat().st_size)
    if size > int(max_bytes):
        raise MemoryError(
            f"V26f {description} exceeds the bounded size: {size} > {int(max_bytes)} bytes"
        )
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or sha256_file(path) != expected:
            raise ValueError(f"V26f {description} hash differs: {path}")
    return path


def read_json(path: str | Path, *, description: str = "JSON artifact") -> dict[str, Any]:
    checked = Path(path)
    if not checked.is_file():
        raise FileNotFoundError(f"V26f {description} is missing: {checked}")
    if checked.stat().st_size > BOUND_FILE_MAX_BYTES:
        raise MemoryError(f"V26f {description} exceeds the bounded size: {checked}")
    value = json.loads(checked.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26f {description} must be a JSON object: {checked}")
    return value






def implementation_sha256(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in IMPLEMENTATION_FILES:
        path = require_bound_file(root, relative, f"V26f implementation {relative}")
        result[relative_path(root, path)] = sha256_file(path)
    return result


def fold_paths(fold_id: int) -> dict[str, Path]:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError(f"invalid V26f fold id: {fold_id}")
    return {
        "l1_config": v26e_l1_config_path(fold_id),
        "l1_data_contract": v26e_l1_data_contract_path(fold_id),
        "l1_normalization": v26e_l1_normalization_path(fold_id),
        "l1_train_cache": v26e_l1_train_feature_cache_path(fold_id),
        "l1_held_out_cache": v26e_l1_held_out_feature_cache_path(fold_id),
        "l1_checkpoint": v26e_l1_checkpoint_path(fold_id),
        "l1_prediction": v26e_l1_prediction_path(fold_id),
        "l1_summary": v26e_l1_training_summary_path(fold_id),
        "l2_config": v26c_l2_config_path(fold_id),
        "l2_data_contract": v26c_l2_data_contract_path(fold_id),
        "l2_normalization": v26c_l2_normalization_path(fold_id),
        "l2_train_physical_cache": v26c_l2_train_physical_cache_path(fold_id),
        "l2_held_out_physical_cache": v26c_l2_held_out_physical_cache_path(fold_id),
        "l2_train_micro_cache": v26c_l2_train_micro_cache_path(fold_id),
        "l2_held_out_micro_cache": v26c_l2_held_out_micro_cache_path(fold_id),
        "l2_checkpoint": v26c_l2_checkpoint_path(fold_id),
        "l2_prediction": v26c_l2_prediction_path(fold_id),
        "l2_summary": v26c_l2_training_summary_path(fold_id),
    }




def _preflight_npz(path: Path, *, max_bytes: int) -> None:
    if path.suffix.lower() != ".npz":
        raise ValueError(f"V26f cache must be an .npz archive: {path}")
    if path.stat().st_size > int(max_bytes):
        raise MemoryError(f"V26f cache exceeds the bounded size: {path}")
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            if len(members) > 16:
                raise ValueError(f"V26f cache has too many archive members: {path}")
            declared = 0
            for member in members:
                if member.is_dir() or member.filename.replace("\\", "/").endswith("/"):
                    raise ValueError(f"V26f cache contains a directory member: {path}")
                declared += int(member.file_size)
            if declared > int(max_bytes):
                raise MemoryError(f"V26f cache declared size exceeds the bound: {path}")
    except zipfile.BadZipFile as exc:
        raise ValueError(f"V26f cache is not a valid npz archive: {path}") from exc


def validate_feature_cache(
    path: Path,
    *,
    expected_sample_ids: Sequence[str],
    width: int,
    valid_rank: int,
    description: str,
) -> dict[str, Any]:
    _preflight_npz(path, max_bytes=FEATURE_CACHE_MAX_BYTES)
    expected = [str(value) for value in expected_sample_ids]
    if len(expected) != len(set(expected)):
        raise ValueError(f"V26f {description} expected sample ids are duplicated")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V26f {description} fields differ")
        raw_ids = np.asarray(archive["sample_ids"])
        values = np.asarray(archive["values"])
        valid = np.asarray(archive["valid"])
    identifiers = [str(value) for value in raw_ids.tolist()]
    if int(valid_rank) == 2:
        expected_valid_shape = (len(expected), int(width))
    elif int(valid_rank) == 1:
        expected_valid_shape = (len(expected),)
    else:
        raise ValueError(f"V26f {description} validity rank is unsupported")
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
        raise ValueError(f"V26f {description} shape/order/value contract differs")
    # Only index by the validity mask after all rank/dtype checks.  This keeps
    # malformed archives on the bounded ValueError path instead of allowing a
    # shape-dependent IndexError (or integer bitwise inversion) to escape.
    if int(valid_rank) == 2:
        invalid_values_are_zero = bool(np.all(values[~valid] == 0.0))
    else:
        invalid_values_are_zero = bool(np.all(values[~valid, :] == 0.0))
    if not invalid_values_are_zero:
        raise ValueError(f"V26f {description} invalid values are not zero")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
        "event_count": len(expected),
        "feature_width": int(width),
        "valid_event_count": int(valid.all(axis=1).sum()) if int(valid_rank) == 2 else int(valid.sum()),
        "invalid_event_count": int((~valid.all(axis=1)).sum()) if int(valid_rank) == 2 else int((~valid).sum()),
        "selected_sample_ids_sha256": sequence_sha256(expected),
    }


def selected_event_sample_ids(root: Path, *, split: str) -> list[str]:
    if split not in EXPECTED_PARENT_COUNTS:
        raise ValueError(f"V26f unsupported mixed split: {split}")
    rows: list[tuple[str, int, str, str, str, str, str]] = []
    path = bound_path(root, INDEX, "source index")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("split_name")) != split or str(row.get("label_name")) not in MIXED_LABELS:
                continue
            if str(row.get("schema_version")) != "RADAR-LN3-V16":
                raise ValueError(f"V26f {split} source index schema differs")
            if str(row.get("source_name")) != "CQ08":
                raise ValueError(f"V26f {split} source index contains a non-CQ08 source")
            parent_id = str(row.get("parent_sample_id") or "")
            sample_id = str(row.get("sample_id") or "")
            label = str(row.get("label_name") or "")
            track_id = str(row.get("track_id") or "")
            capture_id = str(row.get("capture_id") or "")
            provenance_id = str(row.get("provenance_component_id") or "")
            if not parent_id or not sample_id or not track_id or not capture_id or not provenance_id:
                raise ValueError("V26f source index contains an empty identity field")
            try:
                event_index = int(row["causal_event_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("V26f source index has an invalid causal event index") from exc
            if event_index < 0:
                raise ValueError(f"V26f {split} source index has a negative causal event index")
            rows.append((parent_id, event_index, sample_id, label, track_id, capture_id, provenance_id))
    rows.sort(key=lambda value: (value[0], value[1]))
    identifiers = [sample_id for _, _, sample_id, *_ in rows]
    if (
        not identifiers
        or len(identifiers) != len(set(identifiers))
        or len({(parent, event) for parent, event, *_ in rows}) != len(rows)
    ):
        raise ValueError(f"V26f {split} source index event identity is invalid")
    parent_labels: dict[str, str] = {}
    parent_tracks: dict[str, str] = {}
    parent_captures: dict[str, str] = {}
    parent_provenance: dict[str, str] = {}
    parent_events: dict[str, list[int]] = {}
    for parent_id, event_index, _, label, track_id, capture_id, provenance_id in rows:
        if parent_id in parent_labels and parent_labels[parent_id] != label:
            raise ValueError(f"V26f {split} parent has multiple labels: {parent_id}")
        if parent_id in parent_tracks and parent_tracks[parent_id] != track_id:
            raise ValueError(f"V26f {split} parent has multiple tracks: {parent_id}")
        if parent_id in parent_captures and parent_captures[parent_id] != capture_id:
            raise ValueError(f"V26f {split} parent has multiple captures: {parent_id}")
        if parent_id in parent_provenance and parent_provenance[parent_id] != provenance_id:
            raise ValueError(f"V26f {split} parent has multiple provenance components: {parent_id}")
        parent_labels[parent_id] = label
        parent_tracks[parent_id] = track_id
        parent_captures[parent_id] = capture_id
        parent_provenance[parent_id] = provenance_id
        parent_events.setdefault(parent_id, []).append(event_index)
    for parent_id, events in parent_events.items():
        if events != list(range(len(events))):
            raise ValueError(f"V26f {split} causal event order is not contiguous: {parent_id}")
    if len(parent_labels) != sum(EXPECTED_PARENT_COUNTS[split].values()):
        raise ValueError(f"V26f {split} source index parent count differs")
    actual = {label: sum(value == label for value in parent_labels.values()) for label in MIXED_LABELS}
    if actual != EXPECTED_PARENT_COUNTS[split]:
        raise ValueError(f"V26f {split} source index label support differs: {actual}")
    return identifiers




def validate_prepared_contract(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract_path, readiness_path = root / CONTRACT, root / READINESS
    contract = read_json(contract_path, description="V26f contract")
    readiness = read_json(readiness_path, description="V26f readiness")
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
        raise ValueError("V26f prepared contract/readiness is invalid or stale")
    if dict(contract.get("implementation_sha256") or {}) != implementation_sha256(root):
        raise ValueError("V26f implementation binding differs from the prepared contract")

    bindings = list(contract.get("source_bindings") or ())
    if not bindings:
        raise ValueError("V26f contract has no source bindings")
    for raw in bindings:
        binding = dict(raw)
        name = str(binding.get("name", ""))
        path = require_bound_file(
            root,
            str(binding.get("path", "")),
            f"V26f source binding {name}",
            expected_sha256=str(binding.get("sha256", "")),
        )
        if int(binding.get("size_bytes", -1)) != int(path.stat().st_size):
            raise ValueError(f"V26f source binding size differs: {name}")

    data = dict(contract.get("data") or {})
    cache_rows = {str(dict(row).get("split", "")): dict(row) for row in list(data.get("cache_bindings") or ())}
    if set(cache_rows) != {"val", "test"}:
        raise ValueError("V26f mixed cache bindings must contain val and test")
    for split, row in cache_rows.items():
        if dict(row.get("parent_counts") or {}) != EXPECTED_PARENT_COUNTS[split]:
            raise ValueError(f"V26f {split} parent-count binding differs")
        expected_ids = selected_event_sample_ids(root, split=split)
        for name, width, valid_rank in (
            ("l1_course", 1, 2),
            ("l2_rd_physical4", 4, 1),
            ("l2_micro_doppler8", 8, 1),
        ):
            binding = dict(row.get(name) or {})
            path = require_bound_file(
                root,
                str(binding.get("path", "")),
                f"V26f {split} {name} cache",
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
                or actual["selected_sample_ids_sha256"] != binding.get("selected_sample_ids_sha256")
            ):
                raise ValueError(f"V26f {split} {name} cache binding differs")

    folds = list(contract.get("folds") or ())
    if {int(dict(row).get("fold_id", -1)) for row in folds} != set(range(FOLD_COUNT)):
        raise ValueError("V26f prepared fold bindings differ")
    expected_names = set(fold_paths(0))
    for raw in folds:
        row = dict(raw)
        fold_id = int(row["fold_id"])
        files = {str(dict(item).get("name", "")): dict(item) for item in list(row.get("source_files") or ())}
        if set(files) != expected_names:
            raise ValueError(f"V26f fold {fold_id} source-file bindings differ")
        for name, binding in files.items():
            path = require_bound_file(
                root,
                str(binding.get("path", "")),
                f"V26f fold {fold_id} {name}",
                expected_sha256=str(binding.get("sha256", "")),
            )
            if int(binding.get("size_bytes", -1)) != int(path.stat().st_size):
                raise ValueError(f"V26f fold {fold_id} source-file size differs: {name}")
    return contract, readiness


def fold_contract_entry(contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    rows = [
        dict(value)
        for value in list(contract.get("folds") or ())
        if int(dict(value).get("fold_id", -1)) == int(fold_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"V26f fold binding is missing or repeated: {fold_id}")
    return rows[0]




__all__ = ["BOUND_FILE_MAX_BYTES", "CONTRACT", "CONTRACT_SCHEMA", "DBB_LABELS", "EXPECTED_PARENT_COUNTS", "FEATURE_CACHE_MAX_BYTES", "FOLD_COUNT", "IMPLEMENTATION_FILES", "INDEX", "L1_LABELS", "MIXED_LABELS", "PARENT_DETAIL_CSV", "READINESS", "READINESS_SCHEMA", "REPORT", "REPORT_SCHEMA", "bound_path", "fold_contract_entry", "fold_paths", "implementation_sha256", "read_json", "relative_path", "require_bound_file", "selected_event_sample_ids", "sequence_sha256", "sha256_file", "validate_feature_cache", "validate_prepared_contract"]