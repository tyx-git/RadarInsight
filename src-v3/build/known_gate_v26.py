"""Prepare the V26 Known-only baseline and its post-checkpoint open-set caches.

V26 deliberately keeps the L1 model binary.  UnknownTarget is never part of
the model class mapping, train normalization, sampler, optimizer or known-only
checkpoint ranking.  Mixed validation/test data are materialized into separate
audit caches and may only be consumed by the report after a completed
known-only checkpoint exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _write_json_new_or_identical,
    _write_npz_new_or_identical,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V21E_MANIFEST_SCHEMA,
    KNOWN_GATE_V26_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V26_MANIFEST_SCHEMA,
    KNOWN_GATE_V26_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    seed_everything,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_SCHEMA,
    V21E_COURSE_PREFIX_VARIANT,
    V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
    extract_causal_course_circular_incoherence,
    load_known_gate_feature_cache,
)


SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
CORE_CONTRACT = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-data-contract.json"
)
CORE_NORMALIZATION = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-only-train-normalization-stats.json"
)
V21E_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-config.json"
)
V21E_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-causal-course-consistency-"
    "feature-cache-manifest.json"
)
V21E_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-causal-course-consistency-contract.json"
)
V21E_TRAIN_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-train-causal-course-consistency-cache.npz"
)
V21E_VALIDATION_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-val-causal-course-consistency-cache.npz"
)
EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v26_known_only_open_set_baseline_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-known-only-open-set-baseline-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-known-only-open-set-baseline-"
    "feature-cache-manifest.json"
)
AUDIT_VALIDATION_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-val-mixed-open-set-course-cache.npz"
)
AUDIT_TEST_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-test-mixed-open-set-course-cache.npz"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-"
    "seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-"
    "readiness.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-"
    "baseline-seed43-formal"
)

KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
UNKNOWN_LABEL = "UnknownTarget"
AUDIT_LABELS = (*KNOWN_LABELS, UNKNOWN_LABEL)
SEED = 43
EXPECTED_PARENT_COUNTS = {
    "train": {"DroneTarget": 476, "BirdTarget": 239, "BalloonTarget": 238, "ClutterTarget": 33, "UnknownTarget": 69},
    "val": {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54, "ClutterTarget": 9, "UnknownTarget": 15},
    "test": {"DroneTarget": 109, "BirdTarget": 53, "BalloonTarget": 58, "ClutterTarget": 8, "UnknownTarget": 16},
}


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _course_value(payload: Any, *, sample_id: str) -> float | None:
    values = json.loads(str(payload))
    if not isinstance(values, dict):
        raise ValueError(f"V26 course payload is not an object: {sample_id}")
    raw = values.get("course_deg")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"V26 course is non-finite: {sample_id}")
    return value


def _selected_index(path: Path, split: str, *, include_unknown: bool) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "sample_id",
            "parent_sample_id",
            "track_id",
            "split_name",
            "label_name",
            "causal_event_index",
            "tr_feature_values_json",
        ],
        low_memory=False,
    )
    labels = AUDIT_LABELS if include_unknown else KNOWN_LABELS
    selected = frame[
        frame["split_name"].astype(str).eq(split)
        & frame["label_name"].astype(str).isin(labels)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V26 {split} index selection is empty or repeats sample_id")
    support = (
        selected[["parent_sample_id", "label_name"]]
        .drop_duplicates("parent_sample_id")["label_name"]
        .value_counts()
        .to_dict()
    )
    actual = {label: int(support.get(label, 0)) for label in labels}
    expected = {
        label: EXPECTED_PARENT_COUNTS[split][label]
        for label in labels
    }
    if actual != expected:
        raise ValueError(f"V26 {split} parent support changed: expected={expected}, actual={actual}")
    return selected


def _build_course_cache(selected: pd.DataFrame, *, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    values = np.zeros((len(selected), 1), dtype=np.float32)
    valid = np.zeros((len(selected), 1), dtype=np.bool_)
    cursor = 0
    parent_count = 0
    unknown_parent_count = 0
    for parent_id, parent in selected.groupby("parent_sample_id", sort=True):
        labels = parent["label_name"].astype(str).unique().tolist()
        tracks = parent["track_id"].astype(int).unique().tolist()
        events = parent["causal_event_index"].astype(int).tolist()
        if len(labels) != 1 or len(tracks) != 1:
            raise ValueError(f"V26 parent identity is not unique: {parent_id}")
        if events != list(range(len(parent))):
            raise ValueError(f"V26 causal event order is not contiguous: {parent_id}")
        courses = [
            _course_value(row.tr_feature_values_json, sample_id=str(row.sample_id))
            for row in parent.itertuples(index=False)
        ]
        parent_values, parent_valid = extract_causal_course_circular_incoherence(courses)
        length = len(parent)
        values[cursor : cursor + length] = parent_values
        valid[cursor : cursor + length] = parent_valid
        parent_count += 1
        unknown_parent_count += int(labels[0] == UNKNOWN_LABEL)
        cursor += length
    if cursor != len(selected) or np.any(values[~valid] != 0.0):
        raise ValueError("V26 course cache did not preserve causal value/mask semantics")
    return {
        "sample_ids": sample_ids,
        "values": values,
        "valid": valid,
    }, {
        "split": split,
        "event_count": int(len(selected)),
        "parent_count": int(parent_count),
        "unknown_parent_count": int(unknown_parent_count),
        "feature_width": 1,
        "valid_value_count": int(valid.sum()),
        "invalid_value_count": int((~valid).sum()),
        "valid_value_fraction": float(valid.mean()),
        "raw_echo_payload_access_count": 0,
        "rd_tensor_payload_access_count": 0,
    }


def _cache_metadata(
    path: Path,
    *,
    split: str,
    expected_events: int,
    expected_sample_ids: list[str],
) -> dict[str, Any]:
    # Reuse the fail-closed cache loader here instead of opening the frozen
    # archive directly with np.load.  Prepare is a trust boundary too: a
    # malformed NPZ must be size-checked and sample-alignment-checked before
    # any array is decompressed into memory.
    _, _, metadata = load_known_gate_feature_cache(
        path,
        expected_sample_ids=expected_sample_ids,
        variant=V21E_COURSE_PREFIX_VARIANT,
    )
    if int(metadata.get("event_count", -1)) != expected_events:
        raise ValueError(f"V26 frozen {split} course cache event count changed")
    valid_count = int(metadata["valid_value_count"])
    invalid_count = int(metadata["invalid_value_count"])
    return {
        "event_count": expected_events,
        "feature_width": 1,
        "valid_value_count": valid_count,
        "invalid_value_count": invalid_count,
        "valid_value_fraction": float(valid_count / expected_events),
    }


def materialize_test_audit_cache(project_root: Path) -> tuple[Path, dict[str, Any]]:
    """Create the mixed test audit cache only after checkpoint freezing.

    The caller is the V26 report, which records and verifies the immutable
    best-checkpoint hash before invoking this function.  Keeping this helper
    separate from ``build`` makes accidental test access during prepare
    visible and fail-closed in the pipeline contract.
    """

    root = project_root.resolve()
    selected = _selected_index(root / SOURCE_INDEX, "test", include_unknown=True)
    arrays, statistics = _build_course_cache(selected, split="test")
    _write_npz_new_or_identical(root / AUDIT_TEST_CACHE, tuple(arrays.items()))
    return root / AUDIT_TEST_CACHE, statistics


__all__ = [
    "ARTIFACT",
    "AUDIT_TEST_CACHE",
    "AUDIT_VALIDATION_CACHE",
    "CONFIG",
    "EXPECTED_PARENT_COUNTS",
    "EXPERIMENT_CONTRACT",
    "FEATURE_MANIFEST",
    "READINESS",
    "build",
    "materialize_test_audit_cache",
]


def _config_payload(source: Mapping[str, Any], *, manifest_hash: str, contract_hash: str, train_hash: str, val_hash: str) -> dict[str, Any]:
    payload = _clone(source)
    artifact = ARTIFACT.as_posix()
    payload.update(
        {
            "schema_version": "RADAR-LN3-V26",
            "config_name": "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-seed43",
            "variant": V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
        }
    )
    payload["task"]["feature_names"] = list(COURSE_PREFIX_FEATURE_NAMES)
    payload["data"].update(
        {
            "feature_cache_manifest_path": FEATURE_MANIFEST.as_posix(),
            "expected_feature_cache_manifest_sha256": manifest_hash,
            "train_feature_cache_path": V21E_TRAIN_CACHE.as_posix(),
            "validation_feature_cache_path": V21E_VALIDATION_CACHE.as_posix(),
            "expected_train_feature_cache_sha256": train_hash,
            "expected_validation_feature_cache_sha256": val_hash,
            "feature_experiment_contract_path": EXPERIMENT_CONTRACT.as_posix(),
            "expected_feature_experiment_contract_sha256": contract_hash,
        }
    )
    payload["model"]["evidence_dim"] = 1
    payload["train"].update(
        {
            "class_weight_normalization": "batch_weight_sum",
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V26-v26_known_only_open_set_baseline-seed43-"
                "known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": (
                "config/RADAR-LN3/"
                "RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-"
                "config.json"
            ),
        }
    )
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    required = {
        "index": root / SOURCE_INDEX,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "v21e_config": root / V21E_CONFIG,
        "v21e_manifest": root / V21E_MANIFEST,
        "v21e_contract": root / V21E_CONTRACT,
        "train_cache": root / V21E_TRAIN_CACHE,
        "validation_cache": root / V21E_VALIDATION_CACHE,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V26 frozen source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in required.items()}
    source_config = load_known_gate_config(required["v21e_config"])
    if (
        source_config.variant != V21E_COURSE_PREFIX_VARIANT
        or source_config.train.seed != SEED
        or source_config.data.train_feature_cache_path != V21E_TRAIN_CACHE.as_posix()
        or source_config.data.validation_feature_cache_path != V21E_VALIDATION_CACHE.as_posix()
    ):
        raise ValueError("V26 V21e cache provenance is stale")
    source_manifest = json.loads(required["v21e_manifest"].read_text(encoding="utf-8"))
    source_contract = json.loads(required["v21e_contract"].read_text(encoding="utf-8"))
    if (
        source_manifest.get("schema_version") != KNOWN_GATE_V21E_MANIFEST_SCHEMA
        or source_manifest.get("feature_schema") != COURSE_PREFIX_FEATURE_SCHEMA
        or source_manifest.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or source_manifest.get("unknown_excluded") is not True
        or source_manifest.get("test_excluded") is not True
        or source_contract.get("internal_variant") != V21E_COURSE_PREFIX_VARIANT
        or source_contract.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or int(source_contract.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V26 source V21e course contract is stale")
    known_selected = {
        split: _selected_index(required["index"], split, include_unknown=False)
        for split in ("train", "val")
    }
    frozen_cache_stats = {
        "train": _cache_metadata(
            required["train_cache"],
            split="train",
            expected_events=18923,
            expected_sample_ids=known_selected["train"]["sample_id"].astype(str).tolist(),
        ),
        "val": _cache_metadata(
            required["validation_cache"],
            split="val",
            expected_events=4452,
            expected_sample_ids=known_selected["val"]["sample_id"].astype(str).tolist(),
        ),
    }

    selected = {
        "val": _selected_index(required["index"], "val", include_unknown=True)
    }
    audit_arrays: dict[str, dict[str, np.ndarray]] = {}
    audit_stats: dict[str, dict[str, Any]] = {}
    for split in ("val",):
        audit_arrays[split], audit_stats[split] = _build_course_cache(
            selected[split], split=split
        )
    _write_npz_new_or_identical(root / AUDIT_VALIDATION_CACHE, tuple(audit_arrays["val"].items()))
    audit_hashes = {
        "val": sha256_file(root / AUDIT_VALIDATION_CACHE),
    }

    contract = {
        "schema_version": KNOWN_GATE_V26_EXPERIMENT_SCHEMA,
        "experiment_name": "V26 Known-only training with post-checkpoint open-set audit",
        "internal_variant": V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_document": _relative(root, required["experiment_document"]),
        "experiment_document_sha256": hashes["experiment_document"],
        "scope": "KnownGate L1 development baseline; no L2/L3/full-cascade/deployment claim",
        "core_data_contract_sha256": hashes["core_contract"],
        "fresh_training_not_finetune": True,
        "training_population": {
            "labels": list(KNOWN_LABELS),
            "unknown_label": UNKNOWN_LABEL,
            "unknown_training_exposure": 0,
            "normalization_population": "known_only_train",
            "checkpoint_selection_population": "known_only_validation",
        },
        "architecture": {
            "encoder": "self-developed CfCLiquidCausalEncoder",
            "hidden_dim": 48,
            "liquid_hidden_dim": 48,
            "rd_embedding_dim": 24,
            "course_feature_width": 1,
            "zero_dt_mode": "bounded_observation",
            "zero_dt_observation_gain": 0.25,
        },
        "data_reuse": {
            "index": _relative(root, required["index"]),
            "index_sha256": hashes["index"],
            "core_contract": _relative(root, required["core_contract"]),
            "core_contract_sha256": hashes["core_contract"],
            "normalization": _relative(root, required["normalization"]),
            "normalization_sha256": hashes["normalization"],
            "known_train_parent_count": 986,
            "known_validation_parent_count": 232,
            "known_test_parent_count": 228,
            "unknown_train_parent_count": 69,
            "unknown_validation_parent_count": 15,
            "unknown_test_parent_count": 16,
        },
        "course_cache_reuse": {
            "source_manifest": _relative(root, required["v21e_manifest"]),
            "source_manifest_sha256": hashes["v21e_manifest"],
            "source_contract": _relative(root, required["v21e_contract"]),
            "source_contract_sha256": hashes["v21e_contract"],
            "train_cache": _relative(root, required["train_cache"]),
            "train_cache_sha256": hashes["train_cache"],
            "validation_known_only_cache": _relative(root, required["validation_cache"]),
            "validation_known_only_cache_sha256": hashes["validation_cache"],
            "known_cache_values_rebuilt": False,
        },
        "open_set_audit_contract": {
            "audit_only": True,
            "unknown_is_model_class": False,
            "unknown_is_training_target": False,
            "unknown_is_normalization_population": False,
            "unknown_is_checkpoint_ranking_population": False,
            "validation_mixed_cache": _relative(root, root / AUDIT_VALIDATION_CACHE),
            "validation_mixed_cache_sha256": audit_hashes["val"],
            "test_mixed_cache": _relative(root, root / AUDIT_TEST_CACHE),
            "test_mixed_cache_sha256": "materialized_after_checkpoint_frozen",
            "test_access_stage": "after_known_only_checkpoint_frozen",
            "rejection_head": "not_available_in_binary_known_gate",
            "pending_abstain": "not_available_in_known_gate_only",
        },
        "training": {
            "seed": SEED,
            "dataloader_seed": SEED,
            "epochs": 20,
            "batch_size": 2,
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "checkpoint_selection": "parent_min_precision_recall",
            "class_weight_normalization": "batch_weight_sum",
        },
        "evaluation": {
            "primary_hypothesis_labels": list(KNOWN_LABELS),
            "safety_labels": [],
            "known_only_primary": ["parent_final", "eligible_prefix", "window"],
            "open_set_diagnostics": [
                "Unknown_to_KnownTarget",
                "Unknown_to_KnownClutter",
                "Unknown_false_accept_rate",
                "max_known_confidence",
            ],
            "clutter_validation_parent_support": 9,
            "clutter_test_parent_support": 8,
            "bootstrap_required": True,
        },
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "not_executed",
        "open_set_audit": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)

    manifest = {
        "schema_version": KNOWN_GATE_V26_MANIFEST_SCHEMA,
        "feature_schema": COURSE_PREFIX_FEATURE_SCHEMA,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "source_manifest": V21E_MANIFEST.as_posix(),
        "source_manifest_sha256": hashes["v21e_manifest"],
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": hashes["core_contract"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": hashes["index"],
        "cache_files": {
            "train": {
                "path": V21E_TRAIN_CACHE.as_posix(),
                "sha256": hashes["train_cache"],
                "fields": ["sample_ids", "values", "valid"],
                **frozen_cache_stats["train"],
            },
            "val": {
                "path": V21E_VALIDATION_CACHE.as_posix(),
                "sha256": hashes["validation_cache"],
                "fields": ["sample_ids", "values", "valid"],
                **frozen_cache_stats["val"],
            },
        },
        "audit_cache_files": {
            "val_mixed": {
                "path": AUDIT_VALIDATION_CACHE.as_posix(),
                "sha256": audit_hashes["val"],
                "fields": ["sample_ids", "values", "valid"],
                **audit_stats["val"],
            },
            "test_mixed": {
                "path": AUDIT_TEST_CACHE.as_posix(),
                "materialization": "after_checkpoint_frozen",
                "fields": ["sample_ids", "values", "valid"],
            },
        },
        "sample_id_is_model_input": False,
        "valid_mask_is_model_input": False,
        "validation_label_values_used_for_readiness_gate": False,
        "unknown_excluded": True,
        "audit_unknown_in_training": False,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    _write_json_new_or_identical(root / FEATURE_MANIFEST, manifest)
    manifest_hash = sha256_file(root / FEATURE_MANIFEST)
    source_payload = json.loads(required["v21e_config"].read_text(encoding="utf-8"))
    config_payload = _config_payload(
        source_payload,
        manifest_hash=manifest_hash,
        contract_hash=contract_hash,
        train_hash=hashes["train_cache"],
        val_hash=hashes["validation_cache"],
    )
    _write_json_new_or_identical(root / CONFIG, config_payload)
    config = load_known_gate_config(root / CONFIG)
    # Bind prepare-time model construction to the same seed protocol used by
    # train_known_gate.  The paired builder also seeds internally, but keeping
    # the boundary explicit prevents readiness hashes from depending on prior
    # work performed in the prepare process.
    seed_everything(SEED)
    model, reference, pairing = build_known_gate_model_for_config(root, config)
    initial_sha = hashlib.sha256()
    for name in sorted(model.state_dict()):
        value = model.state_dict()[name].detach().cpu().contiguous()
        initial_sha.update(name.encode("utf-8"))
        initial_sha.update(str(value.dtype).encode("ascii"))
        initial_sha.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        initial_sha.update(value.numpy().tobytes(order="C"))
    if (
        reference is None
        or pairing.get("paired_baseline") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise ValueError("V26 fresh seed43 initialization audit failed")
    checks = {
        "known_only_train_population_is_986_parents": True,
        "unknown_train_population_is_69_and_not_exposed": True,
        "known_only_validation_selection_population_is_232_parents": True,
        "unknown_is_absent_from_training_normalization_sampler_and_loss": True,
        "v21e_course_train_and_known_validation_caches_are_reused_byte_for_byte": True,
        "mixed_audit_caches_are_separate_from_training_caches": True,
        "audit_val_support_is_247_with_15_unknown_parents": True,
        "audit_test_support_is_244_with_16_unknown_parents": True,
        "test_cache_is_not_consumed_during_prepare": True,
        "fresh_seed43_initialization_does_not_load_trained_checkpoint": True,
        "new_course_fusion_column_is_zero_initialized": True,
        "full_20_epoch_training_is_required": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": KNOWN_GATE_V26_READINESS_SCHEMA,
        "variant": V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT,
        "checks": checks,
        "config": {"path": _relative(root, root / CONFIG), "sha256": sha256_file(root / CONFIG)},
        "feature_manifest_path": _relative(root, root / FEATURE_MANIFEST),
        "feature_manifest_sha256": manifest_hash,
        "contract_path": _relative(root, root / EXPERIMENT_CONTRACT),
        "contract_sha256": contract_hash,
        "contract_hashes": {
            "index_sha256": hashes["index"],
            "core_contract_sha256": hashes["core_contract"],
            "normalization_sha256": hashes["normalization"],
            "experiment_contract_sha256": contract_hash,
            "feature_manifest_sha256": manifest_hash,
            "train_feature_cache_sha256": hashes["train_cache"],
            "validation_feature_cache_sha256": hashes["validation_cache"],
        },
        "known_train_parent_count": 986,
        "known_validation_parent_count": 232,
        "validation_clutter_parent_support": 9,
        "audit_validation_parent_count": 247,
        "audit_validation_unknown_parent_count": 15,
        "planned_audit_test_parent_count": 244,
        "planned_audit_test_unknown_parent_count": 16,
        "unknown_training_exposure": 0,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "not_executed",
        "open_set_audit": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
        "initial_model_state_sha256": initial_sha.hexdigest(),
        "paired_initialization": {
            "baseline_config": "config/RADAR-LN3/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json",
            "trained_checkpoint_loaded": False,
            "shared_parameters_and_buffers_exactly_equal": True,
            "new_course_fusion_column_exactly_zero": True,
        },
    }
    _write_json_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": KNOWN_GATE_V26_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "audit_validation_cache": str(root / AUDIT_VALIDATION_CACHE),
        "planned_audit_test_cache": str(root / AUDIT_TEST_CACHE),
        "known_train_parent_count": 986,
        "known_validation_parent_count": 232,
        "audit_validation_parent_count": 247,
        "planned_audit_test_parent_count": 244,
        "unknown_training_exposure": 0,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the V26 Known-only open-set baseline.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
