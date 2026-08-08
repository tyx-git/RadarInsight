from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from radar_ln3.rd_physical import (
    RD_PHYSICAL_MASK_MODE,
    RD_PHYSICAL_NAMES,
    RD_PHYSICAL_SCHEMA,
    RD_PHYSICAL_VALUE_DIM,
    RD_VELOCITY_AXIS_SHA256_FLOAT64,
    extract_current_event_rd_physical,
    rd_physical_contract,
)


VERSION = "RADAR-LN3-V11"
PAIRED_PROTOCOL = "v10-control-v11-current-event-rd-physical-v1"
SOURCE_INDEX = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
SOURCE_STATS = Path("data/processed/RADAR-LN3-V2-cq08-tr-rd-train-normalization-stats.json")
SOURCE_CONFIG = Path("config/RADAR-LN3/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-config.json")
SOURCE_CONTRACT = Path("data/processed/RADAR-LN3-V10-cq08-tr-rd-observation-increment-gate-data-contract.json")
SOURCE_INDEX_SHA256 = "a5d054f044709047938c88ef5601caf7d4b482ef980a5a61f19db87b6e649f87"
SOURCE_STATS_SHA256 = "584eddcaeb8d78677f0f4902d622c2aa313d9374b767a47982c12dc058f826b6"
TRAIN_CACHE = Path("data/processed/RADAR-LN3-V11-cq08-train-rd-physical-event-cache.npz")
VAL_CACHE = Path("data/processed/RADAR-LN3-V11-cq08-val-rd-physical-event-cache.npz")
DATA_CONTRACT = Path("data/processed/RADAR-LN3-V11-cq08-tr-rd-physical-gate-data-contract.json")
CACHE_MANIFEST = Path("data/processed/RADAR-LN3-V11-cq08-rd-physical-cache-manifest.json")
EXPECTED_SPLIT_EVENT_COUNTS = {"train": 20684, "val": 4474}
SEALED_TEST_EVENT_COUNT = 4505
SEEDS = (42, 43, 44)
_VERSION_PATTERN = re.compile(r"RADAR-LN3-V(?P<version>\d+)")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _write_json_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    content = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite a different versioned file: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "created"


def _existing_versions(root: Path) -> list[int]:
    versions: set[int] = set()
    for relative in (Path("data/processed"), Path("config/RADAR-LN3")):
        for path in (root / relative).glob("RADAR-LN3-V*"):
            match = _VERSION_PATTERN.search(path.name)
            if match is not None:
                versions.add(int(match.group("version")))
    return sorted(versions)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_indices(value: Any, maximum: int) -> list[int]:
    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("RD frame indices must decode to a list")
    values = [int(item) for item in payload]
    if len(values) != len(set(values)):
        raise ValueError("RD frame indices contain duplicates")
    return values[: max(1, int(maximum))]


class _CanonicalRDCache:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self.current_path: Path | None = None
        self.current_amplitude: np.ndarray | None = None
        self.current_velocity: np.ndarray | None = None
        self.accessed_paths: set[str] = set()

    def load(self, relative_path: str) -> tuple[np.ndarray, np.ndarray]:
        path = Path(str(relative_path).replace("\\", "/"))
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if path == self.current_path and self.current_amplitude is not None and self.current_velocity is not None:
            return self.current_amplitude, self.current_velocity
        with np.load(path, allow_pickle=False) as payload:
            if not {"amplitude", "velocity_mps"}.issubset(payload.files):
                raise ValueError(f"canonical RD tensor lacks amplitude/velocity_mps: {path}")
            amplitude = np.asarray(payload["amplitude"])
            velocity = np.asarray(payload["velocity_mps"])
        if amplitude.ndim != 3 or amplitude.shape[1:] != (21, 384):
            raise ValueError(f"canonical RD amplitude has an invalid shape: {path}: {amplitude.shape}")
        if not np.isfinite(amplitude).all():
            raise ValueError(f"canonical RD amplitude contains a non-finite value: {path}")
        # The shared extractor performs the frozen velocity-axis validation.
        extract_current_event_rd_physical(amplitude[:1], velocity)
        self.current_path = path
        self.current_amplitude = amplitude
        self.current_velocity = velocity
        self.accessed_paths.add(str(path))
        return amplitude, velocity


def _read_train_val_rows(index_path: Path) -> tuple[dict[str, list[dict[str, str]]], dict[str, Any]]:
    required = {
        "schema_version",
        "sample_id",
        "source_name",
        "split_name",
        "rd_present",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
    }
    selected = {"train": [], "val": []}
    skipped_split_counts: Counter[str] = Counter()
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"V3 index lacks fields required for V11 cache: {missing}")
        for row in reader:
            split = str(row["split_name"])
            if split not in selected:
                # Shared V3 index rows are inspected only far enough to exclude
                # their split; no test RD path, frame index, payload or feature
                # is parsed or accessed.
                skipped_split_counts[split] += 1
                continue
            if str(row["schema_version"]) != "RADAR-LN3-V3" or str(row["source_name"]) != "CQ08":
                raise ValueError("V11 cache may consume only CQ08 RADAR-LN3-V3 rows")
            selected[split].append({name: str(row[name]) for name in required})
    for split, expected in EXPECTED_SPLIT_EVENT_COUNTS.items():
        rows = selected[split]
        identifiers = [row["sample_id"] for row in rows]
        if len(rows) != expected or len(identifiers) != len(set(identifiers)):
            raise ValueError(
                f"V11 {split} event set is invalid: expected={expected}, actual={len(rows)}, "
                f"unique={len(set(identifiers))}"
            )
    if dict(skipped_split_counts) != {"test": SEALED_TEST_EVENT_COUNT}:
        raise ValueError(f"unexpected excluded split counts: {dict(skipped_split_counts)}")
    return selected, {
        "selected_split_event_counts": {key: len(value) for key, value in selected.items()},
        "sealed_split_row_counts": dict(skipped_split_counts),
        "test_rd_payload_access_count": 0,
    }


def _compute_split(
    rows: list[dict[str, str]],
    project_root: Path,
    rd_cache: _CanonicalRDCache,
    *,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    sample_ids = np.asarray([row["sample_id"] for row in rows], dtype=np.str_)
    values = np.zeros((len(rows), RD_PHYSICAL_VALUE_DIM), dtype=np.float32)
    valid = np.zeros((len(rows),), dtype=np.bool_)
    present_rows = 0
    selected_frame_count = 0
    truncated_event_count = 0
    for row_index, row in enumerate(rows):
        present = _as_bool(row["rd_present"])
        if not present:
            continue
        present_rows += 1
        raw_indices = json.loads(row["rd_tensor_frame_indices_json"])
        if not isinstance(raw_indices, list):
            raise ValueError(f"RD frame indices are not a list: {row['sample_id']}")
        indices = _parse_indices(row["rd_tensor_frame_indices_json"], max_frames)
        if not indices:
            raise ValueError(f"RD-present event has no frame index: {row['sample_id']}")
        if len(raw_indices) > len(indices):
            truncated_event_count += 1
        amplitude, velocity = rd_cache.load(row["rd_tensor_path"])
        if min(indices) < 0 or max(indices) >= amplitude.shape[0]:
            raise ValueError(f"RD frame index is outside canonical tensor: {row['sample_id']}")
        event = extract_current_event_rd_physical(amplitude[np.asarray(indices, dtype=np.int64)], velocity)
        if not event.valid:
            raise ValueError(f"RD-present event produced invalid physical values: {row['sample_id']}")
        values[row_index] = event.values
        valid[row_index] = True
        selected_frame_count += len(indices)
    if int(valid.sum()) != present_rows or np.any(values[~valid] != 0.0):
        raise ValueError("V11 cache present/valid or missing-zero contract failed")
    return sample_ids, values, valid, {
        "event_count": len(rows),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_fraction": float(valid.mean()),
        "selected_rd_frame_count": selected_frame_count,
        "truncated_event_count": truncated_event_count,
        "feature_minimum": values[valid].min(axis=0).astype(float).tolist() if valid.any() else [None] * 4,
        "feature_maximum": values[valid].max(axis=0).astype(float).tolist() if valid.any() else [None] * 4,
        "feature_mean": values[valid].mean(axis=0, dtype=np.float64).astype(float).tolist() if valid.any() else [None] * 4,
    }


def _same_npz(path: Path, sample_ids: np.ndarray, values: np.ndarray, valid: np.ndarray) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            return (
                set(payload.files) == {"sample_ids", "values", "valid"}
                and np.array_equal(np.asarray(payload["sample_ids"]), sample_ids)
                and np.array_equal(np.asarray(payload["values"]), values)
                and np.array_equal(np.asarray(payload["valid"]), valid)
            )
    except (OSError, ValueError):
        return False


def _normalize_windows_cache_acl(path: Path) -> None:
    """Adopt the cache directory DACL when creator and training accounts differ."""
    if os.name != "nt":
        return
    completed = subprocess.run(
        ["icacls", str(path), "/reset"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PermissionError(f"failed to adopt the parent ACL for V11 cache {path}: {detail}")


def _write_npz_new_or_identical(
    path: Path,
    sample_ids: np.ndarray,
    values: np.ndarray,
    valid: np.ndarray,
) -> str:
    if path.exists():
        if not _same_npz(path, sample_ids, values, valid):
            raise FileExistsError(f"refusing to overwrite a different versioned cache: {path}")
        _normalize_windows_cache_acl(path)
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    cache_stream = io.BytesIO()
    arrays = (("sample_ids", sample_ids), ("values", values), ("valid", valid))
    with zipfile.ZipFile(cache_stream, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, array in arrays:
            array_stream = io.BytesIO()
            np.lib.format.write_array(array_stream, np.asarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_stream.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

    # Create the final file directly in the destination directory. On Windows,
    # moving a file out of a protected TemporaryDirectory preserves the source
    # DACL and can make the cache unreadable to the interactive training user.
    # "xb" retains the versioned no-overwrite contract; the explicit Windows
    # ACL normalization below then adopts data/processed permissions.
    created = False
    try:
        with path.open("xb") as stream:
            created = True
            stream.write(cache_stream.getbuffer())
            stream.flush()
            os.fsync(stream.fileno())
        _normalize_windows_cache_acl(path)
    except FileExistsError:
        if _same_npz(path, sample_ids, values, valid):
            _normalize_windows_cache_acl(path)
            return "reused_identical"
        raise FileExistsError(f"refusing to overwrite a different versioned cache: {path}") from None
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise

    if not _same_npz(path, sample_ids, values, valid):
        path.unlink(missing_ok=True)
        raise ValueError(f"new V11 cache failed its immediate round-trip check: {path}")
    return "created"


def _relative(path: Path) -> str:
    return path.as_posix()


def _build_data_contract(
    source_contract: dict[str, Any],
    cache_hashes: dict[str, str],
    split_statistics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    contract = dict(source_contract)
    feature_contract = dict(source_contract["feature_contract"])
    feature_contract["rd_physical"] = rd_physical_contract()
    model_io = dict(source_contract["model_io"])
    model_io.update(
        {
            "rd_physical_features": "[B,L,4] float32; current-event-rd-physical-spectrum-v1",
            "rd_physical_valid": "[B,L] bool; gating_only and not a classifier feature",
            "task_sequence_logits": "[B,L,2] for the V11 DBB/CU paired Gate only",
            "task_window_logits": "[B,2] for the V11 DBB/CU paired Gate only",
        }
    )
    contract.update(
        {
            "schema_version": VERSION,
            "contract_name": "cq08_current_event_rd_physical_paired_gate_v1",
            "contract_status": "prepared_for_bounded_validation_not_formally_trained",
            "source_contract": _relative(SOURCE_CONTRACT),
            "source_contract_sha256": sha256_file(Path(source_contract["_absolute_path"])),
            "source_index": _relative(SOURCE_INDEX),
            "source_index_sha256": SOURCE_INDEX_SHA256,
            "source_normalization_stats": _relative(SOURCE_STATS),
            "source_normalization_stats_sha256": SOURCE_STATS_SHA256,
            "feature_contract": feature_contract,
            "model_io": model_io,
            "rd_physical_cache": {
                "cache_scope": "CQ08 V3 train and val only",
                "train": {"path": _relative(TRAIN_CACHE), "sha256": cache_hashes["train"], **split_statistics["train"]},
                "val": {"path": _relative(VAL_CACHE), "sha256": cache_hashes["val"], **split_statistics["val"]},
                "npz_fields": {"sample_ids": "[N] unicode; alignment only", "values": "[N,4] float32", "valid": "[N] bool"},
                "strict_alignment": "cache and configured split sample_id sets must be identical with no duplicate, missing or extra ID",
                "test_excluded": True,
                "test_event_count_from_frozen_V3_contract": SEALED_TEST_EVENT_COUNT,
                "test_rd_payload_access_count": 0,
            },
            "training_contract": {
                "task": "dbb_cu_gate",
                "paired_protocol": PAIRED_PROTOCOL,
                "control": "V10 architecture retrained from the paired seed; rd_physical_dim=0 and cache is not loaded",
                "treatment": "only append four current-event RD physical values to Fusion; mathematical input width 127->131",
                "treatment_initialization": "all shared parameters exactly copied from paired Control; four new Fusion columns exactly zero",
                "seeds": list(SEEDS),
                "epochs": 20,
                "early_stopping": False,
                "loader_pairing": "independent generators initialized with the same seed; record each epoch parent-order SHA-256",
                "cuda_deterministic_algorithms": False,
                "determinism_claim": "fixed shared initialization and sampling order; not bitwise CUDA determinism",
                "checkpoint_selection": "each arm uses its own validation parent_min_precision_recall best; same-epoch trajectories are attribution guards",
                "epoch_checkpoints": "save every epoch plus best_model.pt and last_model.pt",
                "pretraining": "not_required_and_not_used",
                "formal_training_status": "not_executed",
                "test_status": "not_executed",
            },
            "stage_one_gate": {
                "seed_rule": "at least 2 of 3 independent Treatment seeds and the 3-seed median must pass",
                "DBB_minimum_precision_recall_f1": 0.95,
                "CU_minimum_precision_recall_f1": 0.90,
                "paired_discrete_pareto": [
                    "DBB_to_CU count no higher than paired Control",
                    "CU_to_DBB count no higher than paired Control",
                    "at least one direction improves by one or more validation parents",
                    "CU min(precision,recall,f1) strictly improves",
                ],
            },
            "stage_two": {
                "status": "not_implemented_not_executed",
                "entry_condition": "stage one passes",
                "planned_scope": "train+val five-fold parent-grouped OOF paired verification; test remains sealed",
                "pooled_DBB_minimum_precision_recall_f1": 0.99,
                "pooled_CU_minimum_precision_recall_f1": 0.95,
            },
            "leakage_policy": list(source_contract["leakage_policy"])
            + [
                "V11 physical values use only current-event RD frames selected by the existing causal V3 mapping",
                "sample_id is cache alignment metadata and never enters a model tensor",
                "RD physical valid gates numerical values and is not concatenated into classifier inputs",
                "no test RD tensor is opened and no test physical cache is generated",
                "Track 333 points 1-15 remain explicit RD-missing events with zero physical values and valid=false",
            ],
            "reuse": {
                "index_reuse": True,
                "index_path": _relative(SOURCE_INDEX),
                "index_sha256": SOURCE_INDEX_SHA256,
                "normalization_stats_reuse": True,
                "normalization_stats_path": _relative(SOURCE_STATS),
                "normalization_stats_sha256": SOURCE_STATS_SHA256,
                "canonical_rd_tensors_reused_read_only": True,
                "V10_checkpoint_compatibility": "forbidden for V11 loading; V10 is a paired architecture reference, not a warm start",
            },
            "version_reason": (
                "V11 is the next unused global version and adds a versioned current-event four-dimensional RD physical "
                "cache/input/checkpoint/training contract. V3 rows/splits, V2 normalization, labels and canonical RD tensors are unchanged."
            ),
            "validation_status": {
                "cache_generation": "completed_by_this_builder",
                "focused_contract_tests": "not_executed_at_generation_time",
                "cpu_smoke": "not_executed_at_generation_time",
                "cuda_smoke": "not_executed_at_generation_time",
                "formal_training": "not_executed",
                "test": "not_executed",
                "OOF": "not_executed",
            },
        }
    )
    contract.pop("paired_baseline_config", None)
    contract.pop("paired_baseline_config_sha256", None)
    contract.pop("observation_increment_causality", None)
    contract.pop("_absolute_path", None)
    return contract


def _build_cache_manifest(
    contract_hash: str,
    cache_hashes: dict[str, str],
    split_statistics: dict[str, dict[str, Any]],
    read_summary: dict[str, Any],
    source_config_hash: str,
    accessed_tensor_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": VERSION,
        "manifest_type": "cq08_current_event_rd_physical_cache",
        "paired_protocol": PAIRED_PROTOCOL,
        "source_name": "CQ08",
        "source_index": _relative(SOURCE_INDEX),
        "source_index_sha256": SOURCE_INDEX_SHA256,
        "source_V10_config": _relative(SOURCE_CONFIG),
        "source_V10_config_sha256": source_config_hash,
        "data_contract": _relative(DATA_CONTRACT),
        "data_contract_sha256": contract_hash,
        "rd_physical_contract": rd_physical_contract(),
        "cache_files": {
            "train": {"path": _relative(TRAIN_CACHE), "sha256": cache_hashes["train"], **split_statistics["train"]},
            "val": {"path": _relative(VAL_CACHE), "sha256": cache_hashes["val"], **split_statistics["val"]},
        },
        "cache_fields": ["sample_ids", "values", "valid"],
        "sample_id_model_input": False,
        "canonical_rd_tensor_accessed_count": accessed_tensor_count,
        "velocity_axis_sha256_float64_c_order": RD_VELOCITY_AXIS_SHA256_FLOAT64,
        "test_excluded": True,
        "test_event_count_from_frozen_V3_contract": SEALED_TEST_EVENT_COUNT,
        "test_rd_payload_access_count": int(read_summary["test_rd_payload_access_count"]),
        "test_cache_generated": False,
        "split_row_evidence": read_summary,
        "formal_training": "not_executed",
        "test_evaluation": "not_executed",
        "OOF": "not_executed",
    }


def _config_path(arm: str, seed: int) -> Path:
    return Path(
        f"config/RADAR-LN3/RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-{arm}-seed{seed}-config.json"
    )


def _counterpart_config_path(arm: str, seed: int) -> str:
    counterpart = "treatment" if arm == "control" else "control"
    return _relative(_config_path(counterpart, seed))


def _artifact_dir(arm: str, seed: int) -> str:
    return f"artifacts/RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-{arm}-seed{seed}-formal"


def _build_config(
    source: dict[str, Any],
    *,
    arm: str,
    seed: int,
    contract_hash: str,
    manifest_hash: str,
    cache_hashes: dict[str, str],
) -> dict[str, Any]:
    if arm not in {"control", "treatment"} or seed not in SEEDS:
        raise ValueError("invalid V11 paired arm or seed")
    payload = json.loads(json.dumps(source))
    artifact = _artifact_dir(arm, seed)
    payload["schema_version"] = VERSION
    payload["config_name"] = f"RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-{arm}-seed{seed}"
    payload["data"].update(
        {
            "contract_path": _relative(DATA_CONTRACT),
            "contract_schema_version": VERSION,
            "expected_contract_sha256": contract_hash,
            "rd_physical_schema": RD_PHYSICAL_SCHEMA if arm == "treatment" else None,
            "rd_physical_mask_mode": RD_PHYSICAL_MASK_MODE if arm == "treatment" else None,
            "rd_physical_cache_manifest_path": _relative(CACHE_MANIFEST) if arm == "treatment" else None,
            "expected_rd_physical_cache_manifest_sha256": manifest_hash if arm == "treatment" else None,
            "train_rd_physical_cache_path": _relative(TRAIN_CACHE) if arm == "treatment" else None,
            "validation_rd_physical_cache_path": _relative(VAL_CACHE) if arm == "treatment" else None,
            "expected_train_rd_physical_cache_sha256": cache_hashes["train"] if arm == "treatment" else None,
            "expected_validation_rd_physical_cache_sha256": cache_hashes["val"] if arm == "treatment" else None,
        }
    )
    payload["model"]["rd_physical_dim"] = RD_PHYSICAL_VALUE_DIM if arm == "treatment" else 0
    payload["train"].update(
        {
            "seed": seed,
            "epochs": 20,
            "early_stopping_patience": 0,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "formal_artifact_dir": artifact,
            "checkpoint_schema": "RADAR-LN3-V11-dbb-cu-gate-rd-physical-paired-checkpoint-v1",
            "paired_protocol": PAIRED_PROTOCOL,
            "paired_arm": arm,
            "paired_seed": seed,
            "dataloader_seed": seed,
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": False,
            "paired_counterpart_config_path": _counterpart_config_path(arm, seed),
        }
    )
    return payload


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    versions = _existing_versions(root)
    expected_v11_outputs = [
        TRAIN_CACHE,
        VAL_CACHE,
        DATA_CONTRACT,
        CACHE_MANIFEST,
        *[_config_path(arm, seed) for seed in SEEDS for arm in ("control", "treatment")],
    ]
    if versions and max(versions) > 11 and not all((root / path).exists() for path in expected_v11_outputs):
        raise ValueError(
            "cannot create a missing V11 output after a higher RADAR-LN3 version exists; "
            f"max={max(versions)}. Existing complete V11 outputs may still be verified/reused identically."
        )
    for relative, expected in ((SOURCE_INDEX, SOURCE_INDEX_SHA256), (SOURCE_STATS, SOURCE_STATS_SHA256)):
        actual = sha256_file(root / relative)
        if actual != expected:
            raise ValueError(f"frozen source hash changed: {relative}: expected={expected}, actual={actual}")
    source_config_path = root / SOURCE_CONFIG
    source_config_hash = sha256_file(source_config_path)
    source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    if Path(source_config["data"]["contract_path"]) != SOURCE_CONTRACT:
        raise ValueError("the frozen V10 config no longer references the expected V10 data contract")
    source_contract_path = root / SOURCE_CONTRACT
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_contract["_absolute_path"] = str(source_contract_path)

    split_rows, read_summary = _read_train_val_rows(root / SOURCE_INDEX)
    cache = _CanonicalRDCache(root)
    split_arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    split_statistics: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        sample_ids, values, valid, statistics = _compute_split(
            split_rows[split], root, cache, max_frames=int(source_config["data"]["max_rd_frames_per_point"])
        )
        split_arrays[split] = (sample_ids, values, valid)
        split_statistics[split] = statistics

    cache_actions = {
        "train": _write_npz_new_or_identical(root / TRAIN_CACHE, *split_arrays["train"]),
        "val": _write_npz_new_or_identical(root / VAL_CACHE, *split_arrays["val"]),
    }
    cache_hashes = {
        "train": sha256_file(root / TRAIN_CACHE),
        "val": sha256_file(root / VAL_CACHE),
    }
    data_contract = _build_data_contract(source_contract, cache_hashes, split_statistics)
    contract_action = _write_json_new_or_identical(root / DATA_CONTRACT, data_contract)
    contract_hash = sha256_file(root / DATA_CONTRACT)
    manifest = _build_cache_manifest(
        contract_hash,
        cache_hashes,
        split_statistics,
        read_summary,
        source_config_hash,
        len(cache.accessed_paths),
    )
    manifest_action = _write_json_new_or_identical(root / CACHE_MANIFEST, manifest)
    manifest_hash = sha256_file(root / CACHE_MANIFEST)

    config_actions: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    for seed in SEEDS:
        for arm in ("control", "treatment"):
            path = _config_path(arm, seed)
            payload = _build_config(
                source_config,
                arm=arm,
                seed=seed,
                contract_hash=contract_hash,
                manifest_hash=manifest_hash,
                cache_hashes=cache_hashes,
            )
            config_actions[str(path)] = _write_json_new_or_identical(root / path, payload)
            config_hashes[str(path)] = sha256_file(root / path)

    return {
        "status": "passed",
        "schema_version": VERSION,
        "paired_protocol": PAIRED_PROTOCOL,
        "version_scan": versions,
        "source_hashes": {
            str(SOURCE_INDEX): SOURCE_INDEX_SHA256,
            str(SOURCE_STATS): SOURCE_STATS_SHA256,
            str(SOURCE_CONFIG): source_config_hash,
            str(source_contract_path.relative_to(root)): sha256_file(source_contract_path),
        },
        "cache_actions": cache_actions,
        "cache_hashes": cache_hashes,
        "cache_statistics": split_statistics,
        "canonical_rd_tensor_accessed_count": len(cache.accessed_paths),
        "data_contract": {"path": str(DATA_CONTRACT), "action": contract_action, "sha256": contract_hash},
        "cache_manifest": {"path": str(CACHE_MANIFEST), "action": manifest_action, "sha256": manifest_hash},
        "config_actions": config_actions,
        "config_hashes": config_hashes,
        "read_scope": read_summary,
        "test_excluded": True,
        "test_rd_payload_access_count": 0,
        "formal_training": "未执行",
        "test_evaluation": "未执行",
        "OOF": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the versioned V11 current-event RD physical paired contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
