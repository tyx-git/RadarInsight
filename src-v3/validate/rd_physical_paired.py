from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from build.rd_physical_paired import (
    CACHE_MANIFEST,
    DATA_CONTRACT,
    EXPECTED_SPLIT_EVENT_COUNTS,
    PAIRED_PROTOCOL,
    SEALED_TEST_EVENT_COUNT,
    SEEDS,
    SOURCE_INDEX,
    SOURCE_INDEX_SHA256,
    TRAIN_CACHE,
    VAL_CACHE,
    VERSION,
    _config_path,
    sha256_file,
)
from radar_ln3.rd_physical import (
    RD_PHYSICAL_VALUE_DIM,
    extract_current_event_rd_physical,
    load_rd_physical_cache,
    rd_physical_contract,
)
from radar_ln3.task_models import load_task_config, verify_task_data_contract


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


class _RDLoader:
    def __init__(self, project_root: Path) -> None:
        self.root = project_root
        self.path: Path | None = None
        self.amplitude: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.accessed_paths: set[str] = set()

    def load(self, relative_path: str) -> tuple[np.ndarray, np.ndarray]:
        path = Path(str(relative_path).replace("\\", "/"))
        path = path if path.is_absolute() else self.root / path
        path = path.resolve()
        if path == self.path and self.amplitude is not None and self.velocity is not None:
            return self.amplitude, self.velocity
        with np.load(path, allow_pickle=False) as payload:
            amplitude = np.asarray(payload["amplitude"])
            velocity = np.asarray(payload["velocity_mps"])
        self.path = path
        self.amplitude = amplitude
        self.velocity = velocity
        self.accessed_paths.add(str(path))
        return amplitude, velocity


def _cache_arrays(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return (
            np.asarray(payload["sample_ids"]),
            np.asarray(payload["values"]),
            np.asarray(payload["valid"]),
        )


def _validate_split(
    root: Path,
    *,
    split: str,
    rows: list[dict[str, str]],
    cache_path: Path,
    loader: _RDLoader,
    max_frames: int,
) -> dict[str, Any]:
    expected_ids = [row["sample_id"] for row in rows]
    load_rd_physical_cache(root / cache_path, expected_sample_ids=expected_ids)
    sample_ids, cached_values, cached_valid = _cache_arrays(root / cache_path)
    if [str(value) for value in sample_ids.tolist()] != expected_ids:
        raise ValueError(f"V11 {split} cache ordering differs from the V3 index order")
    recomputed = np.zeros((len(rows), RD_PHYSICAL_VALUE_DIM), dtype=np.float32)
    recomputed_valid = np.zeros((len(rows),), dtype=np.bool_)
    rd_present = np.zeros((len(rows),), dtype=np.bool_)
    mismatch_ids: list[str] = []
    for index, row in enumerate(rows):
        present = _as_bool(row["rd_present"])
        rd_present[index] = present
        if not present:
            continue
        indices_payload = json.loads(row["rd_tensor_frame_indices_json"])
        if not isinstance(indices_payload, list):
            raise ValueError(f"V3 RD indices are not a list: {row['sample_id']}")
        indices = [int(value) for value in indices_payload[:max_frames]]
        amplitude, velocity = loader.load(row["rd_tensor_path"])
        event = extract_current_event_rd_physical(amplitude[np.asarray(indices, dtype=np.int64)], velocity)
        recomputed[index] = event.values
        recomputed_valid[index] = event.valid
        if not np.array_equal(cached_values[index], event.values) or bool(cached_valid[index]) != event.valid:
            mismatch_ids.append(row["sample_id"])
    if mismatch_ids:
        raise ValueError(f"V11 {split} cache differs from online-source recomputation: {mismatch_ids[:10]}")
    if not np.array_equal(cached_values, recomputed) or not np.array_equal(cached_valid, recomputed_valid):
        raise ValueError(f"V11 {split} full cache arrays differ from recomputation")
    if not np.array_equal(cached_valid, rd_present):
        raise ValueError(f"V11 {split} RD-present and physical-valid masks differ")
    return {
        "event_count": len(rows),
        "cache_sha256": sha256_file(root / cache_path),
        "sample_set_exact": True,
        "sample_order_exact": True,
        "duplicate_count": len(expected_ids) - len(set(expected_ids)),
        "valid_event_count": int(cached_valid.sum()),
        "invalid_event_count": int((~cached_valid).sum()),
        "rd_present_equals_valid": True,
        "full_online_source_recomputation_exact": True,
        "maximum_absolute_error": float(np.max(np.abs(cached_values - recomputed))) if len(rows) else 0.0,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    index_path = root / SOURCE_INDEX
    if sha256_file(index_path) != SOURCE_INDEX_SHA256:
        raise ValueError("V3 source index hash changed before V11 validation")
    required = {
        "sample_id",
        "parent_sample_id",
        "causal_event_index",
        "source_name",
        "split_name",
        "rd_present",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
    }
    rows_by_split = {"train": [], "val": []}
    excluded: Counter[str] = Counter()
    with index_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"V3 index is missing V11 validation fields: {missing}")
        for source_row in reader:
            split = str(source_row["split_name"])
            if split not in rows_by_split:
                excluded[split] += 1
                continue
            row = {key: str(source_row[key]) for key in required}
            if row["source_name"] != "CQ08":
                raise ValueError("V11 validation selected a non-CQ08 row")
            rows_by_split[split].append(row)
    if dict(excluded) != {"test": SEALED_TEST_EVENT_COUNT}:
        raise ValueError(f"V11 validation encountered unexpected excluded splits: {dict(excluded)}")
    for split, expected in EXPECTED_SPLIT_EVENT_COUNTS.items():
        if len(rows_by_split[split]) != expected:
            raise ValueError(f"V11 validation {split} count mismatch")

    manifest_path = root / CACHE_MANIFEST
    contract_path = root / DATA_CONTRACT
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("paired_protocol") != PAIRED_PROTOCOL
        or manifest.get("rd_physical_contract") != rd_physical_contract()
        or manifest.get("test_excluded") is not True
        or int(manifest.get("test_rd_payload_access_count", -1)) != 0
        or manifest.get("test_cache_generated") is not False
    ):
        raise ValueError("V11 cache manifest schema, extractor or test sealing is invalid")
    if (
        contract.get("schema_version") != VERSION
        or contract.get("feature_contract", {}).get("rd_physical") != rd_physical_contract()
        or contract.get("rd_physical_cache", {}).get("test_excluded") is not True
    ):
        raise ValueError("V11 data/input contract is invalid")
    unexpected_test_cache = list((root / "data/processed").glob("RADAR-LN3-V11*test*rd*physical*.npz"))
    if unexpected_test_cache:
        raise ValueError(f"V11 test RD physical cache must not exist: {unexpected_test_cache}")

    source_config = load_task_config(root / _config_path("control", 42))
    loader = _RDLoader(root)
    split_results = {
        "train": _validate_split(
            root,
            split="train",
            rows=rows_by_split["train"],
            cache_path=TRAIN_CACHE,
            loader=loader,
            max_frames=source_config.data.max_rd_frames_per_point,
        ),
        "val": _validate_split(
            root,
            split="val",
            rows=rows_by_split["val"],
            cache_path=VAL_CACHE,
            loader=loader,
            max_frames=source_config.data.max_rd_frames_per_point,
        ),
    }

    track333 = [row for row in rows_by_split["val"] if row["parent_sample_id"] == "cq08|track|333"]
    if len(track333) != 22:
        raise ValueError("V11 validation cannot reproduce Track 333's 22 events")
    _, track333_values, track333_valid = _cache_arrays(root / VAL_CACHE)
    val_positions = {row["sample_id"]: index for index, row in enumerate(rows_by_split["val"])}
    first15 = [val_positions[row["sample_id"]] for row in track333 if int(row["causal_event_index"]) < 15]
    last7 = [val_positions[row["sample_id"]] for row in track333 if int(row["causal_event_index"]) >= 15]
    if (
        len(first15) != 15
        or len(last7) != 7
        or bool(track333_valid[first15].any())
        or bool((track333_values[first15] != 0.0).any())
        or not bool(track333_valid[last7].all())
    ):
        raise ValueError("V11 Track 333 missing/present physical feature semantics changed")

    config_results: list[dict[str, Any]] = []
    artifact_dirs: set[str] = set()
    for seed in SEEDS:
        for arm in ("control", "treatment"):
            path = root / _config_path(arm, seed)
            config = load_task_config(path)
            evidence = verify_task_data_contract(root, config)
            if config.train.formal_artifact_dir in artifact_dirs:
                raise ValueError("V11 paired configs share an artifact directory")
            artifact_dirs.add(config.train.formal_artifact_dir)
            config_results.append(
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "arm": arm,
                    "seed": seed,
                    "rd_physical_dim": config.model.rd_physical_dim,
                    "cache_loaded_by_contract": evidence["rd_physical"] is not None,
                    "artifact_dir": config.train.formal_artifact_dir,
                }
            )

    return {
        "status": "passed",
        "schema_version": VERSION,
        "paired_protocol": PAIRED_PROTOCOL,
        "source_index_sha256": SOURCE_INDEX_SHA256,
        "manifest_sha256": sha256_file(manifest_path),
        "contract_sha256": sha256_file(contract_path),
        "split_results": split_results,
        "canonical_rd_tensor_accessed_count": len(loader.accessed_paths),
        "track_333": {
            "event_count": 22,
            "points_1_to_15_valid_count": int(track333_valid[first15].sum()),
            "points_1_to_15_all_zero": True,
            "points_16_to_22_valid_count": int(track333_valid[last7].sum()),
        },
        "config_results": config_results,
        "artifact_directories_unique": True,
        "test_row_count_sealed": SEALED_TEST_EVENT_COUNT,
        "test_rd_payload_access_count": 0,
        "test_cache_generated": False,
        "formal_training": "未执行",
        "test_evaluation": "未执行",
        "OOF": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate all V11 RD physical train/val cache and paired contracts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
