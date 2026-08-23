"""Prepare V24's independently trained course-reason-aware KnownGate baseline.

V24 consumes the frozen V21e causal-course cache but deliberately does not
load, fine-tune, or otherwise derive weights from a V21e trained checkpoint.
Its fresh random state is constructed through the V21 seed43 baseline config,
then the one course evidence fusion column is zero initialized.  This makes it
the clean paired control for V24a--V24c.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.known_gate import (
    KNOWN_GATE_V21E_MANIFEST_SCHEMA,
    KNOWN_GATE_V24_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V24_MANIFEST_SCHEMA,
    KNOWN_GATE_V24_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_SCHEMA,
    V21E_COURSE_PREFIX_VARIANT,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
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
V21_BASELINE_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json"
)
V21_BASELINE_SUMMARY = Path(
    "artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-formal/"
    "training-summary.json"
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
V21E_TREATMENT_SUMMARY = Path(
    "artifacts/RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal/training-summary.json"
)
EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v24_course_reason_aware_baseline_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V24-cq08-known-gate-course-reason-aware-baseline-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V24-cq08-known-gate-course-reason-aware-baseline-"
    "feature-cache-manifest.json"
)
CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-baseline-"
    "seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-"
    "baseline-seed43-formal"
)

SEED = 43
TRAIN_PARENT_COUNT = 986
VALIDATION_PARENT_COUNT = 232
VALIDATION_CLUTTER_PARENT_SUPPORT = 9
EXPECTED_CACHE_EVENT_COUNTS = {"train": 18923, "val": 4452}


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite a different V24 preparation artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return "created"


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _cache_metadata(path: Path, *, split: str) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V24 {split} course cache fields differ")
        sample_ids = np.asarray(cache["sample_ids"])
        values = np.asarray(cache["values"])
        valid = np.asarray(cache["valid"])
    expected_events = EXPECTED_CACHE_EVENT_COUNTS[split]
    if (
        sample_ids.shape != (expected_events,)
        or sample_ids.dtype.kind not in {"U", "S"}
        or len({str(value) for value in sample_ids.tolist()}) != expected_events
        or values.shape != (expected_events, 1)
        or values.dtype != np.float32
        or valid.shape != values.shape
        or valid.dtype != np.bool_
        or not np.isfinite(values).all()
        or bool((values < 0.0).any())
        or bool((values > 1.0).any())
        or bool((values[~valid] != 0.0).any())
    ):
        raise ValueError(f"V24 {split} frozen course cache is malformed")
    return {
        "event_count": expected_events,
        "feature_width": 1,
        "valid_value_count": int(valid.sum()),
        "invalid_value_count": int((~valid).sum()),
        "valid_value_fraction": float(valid.mean()),
    }


def _source_payload(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "index": root / SOURCE_INDEX,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "baseline_config": root / V21_BASELINE_CONFIG,
        "baseline_summary": root / V21_BASELINE_SUMMARY,
        "v21e_manifest": root / V21E_MANIFEST,
        "v21e_contract": root / V21E_CONTRACT,
        "v21e_treatment_summary": root / V21E_TREATMENT_SUMMARY,
        "train_cache": root / V21E_TRAIN_CACHE,
        "validation_cache": root / V21E_VALIDATION_CACHE,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V24 frozen source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    baseline_payload = json.loads(paths["baseline_config"].read_text(encoding="utf-8"))
    baseline_summary = json.loads(paths["baseline_summary"].read_text(encoding="utf-8"))
    source_manifest = json.loads(paths["v21e_manifest"].read_text(encoding="utf-8"))
    source_contract = json.loads(paths["v21e_contract"].read_text(encoding="utf-8"))
    v21e_summary = json.loads(
        paths["v21e_treatment_summary"].read_text(encoding="utf-8")
    )
    if (
        baseline_payload.get("schema_version") != "RADAR-LN3-V21"
        or baseline_payload.get("variant") != "baseline"
        or int(dict(baseline_payload.get("train") or {}).get("seed", -1)) != SEED
        or baseline_summary.get("status") != "completed"
        or baseline_summary.get("variant") != "baseline"
        or int(baseline_summary.get("seed", -1)) != SEED
        or len(list(baseline_summary.get("history") or ())) != 20
        or baseline_summary.get("test_evaluation") != "not_executed"
        or baseline_summary.get("deployment_eligible") is not False
        or not str(baseline_summary.get("initial_model_state_sha256", ""))
    ):
        raise ValueError("V24 seed43 fresh-initialization source is stale")
    cache_files = dict(source_manifest.get("cache_files") or {})
    manifest_train = dict(cache_files.get("train") or {})
    manifest_val = dict(cache_files.get("val") or {})
    if (
        source_manifest.get("schema_version") != KNOWN_GATE_V21E_MANIFEST_SCHEMA
        or source_manifest.get("feature_schema") != COURSE_PREFIX_FEATURE_SCHEMA
        or source_manifest.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or source_manifest.get("core_data_contract") != CORE_CONTRACT.as_posix()
        or source_manifest.get("core_data_contract_sha256") != hashes["core_contract"]
        or source_manifest.get("source_index") != SOURCE_INDEX.as_posix()
        or source_manifest.get("source_index_sha256") != hashes["index"]
        or source_manifest.get("experiment_contract") != V21E_CONTRACT.as_posix()
        or source_manifest.get("experiment_contract_sha256") != hashes["v21e_contract"]
        or manifest_train.get("path") != V21E_TRAIN_CACHE.as_posix()
        or manifest_train.get("sha256") != hashes["train_cache"]
        or manifest_val.get("path") != V21E_VALIDATION_CACHE.as_posix()
        or manifest_val.get("sha256") != hashes["validation_cache"]
        or source_manifest.get("unknown_excluded") is not True
        or source_manifest.get("test_excluded") is not True
        or int(source_manifest.get("test_payload_access_count", -1)) != 0
        or source_contract.get("internal_variant") != V21E_COURSE_PREFIX_VARIANT
        or source_contract.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or source_contract.get("core_data_contract_sha256") != hashes["core_contract"]
        or int(source_contract.get("raw_echo_payload_access_count", -1)) != 0
        or int(source_contract.get("mat_payload_access_count", -1)) != 0
        or int(source_contract.get("rd_tensor_payload_access_count", -1)) != 0
        or int(source_contract.get("test_payload_access_count", -1)) != 0
        or v21e_summary.get("status") != "completed"
        or v21e_summary.get("variant") != V21E_COURSE_PREFIX_VARIANT
        or int(v21e_summary.get("seed", -1)) != SEED
        or len(list(v21e_summary.get("history") or ())) != 20
        or not str(v21e_summary.get("initial_model_state_sha256", ""))
    ):
        raise ValueError("V24 frozen V21e course-cache provenance is stale")
    return paths, hashes, baseline_payload


def _initialization_audit(root: Path, baseline_config_path: Path, config_path: Path) -> dict[str, Any]:
    baseline_config = load_known_gate_config(baseline_config_path)
    baseline, baseline_reference, baseline_pair = build_known_gate_model_for_config(
        root, baseline_config
    )
    if baseline_reference is not None or baseline_pair.get("paired_baseline") is not False:
        raise AssertionError("V24 source baseline did not reconstruct from a fresh seed")
    baseline_state_sha = _state_dict_sha256(baseline.state_dict())
    config = load_known_gate_config(config_path)
    treatment, reference, pairing = build_known_gate_model_for_config(root, config)
    if reference is None:
        raise AssertionError("V24 baseline lacks its fresh paired initialization reference")
    reference_state_sha = _state_dict_sha256(reference.state_dict())
    treatment_state_sha = _state_dict_sha256(treatment.state_dict())
    if (
        config.variant != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or config.model.evidence_dim != 1
        or config.feature_names != COURSE_PREFIX_FEATURE_NAMES
        or baseline_state_sha != reference_state_sha
        or pairing.get("paired_baseline") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise AssertionError("V24 fresh paired initialization audit failed")
    del baseline, reference, treatment
    return {
        "fresh_v21_seed43_baseline_initial_model_state_sha256": baseline_state_sha,
        "v24_initial_model_state_sha256": treatment_state_sha,
        "fresh_random_initialization_reconstructed": True,
        "shared_parameters_and_buffers_exactly_equal": True,
        "course_fusion_column_exactly_zero": True,
        "course_fusion_column_count": 1,
        "trained_checkpoint_loaded": False,
        "trained_checkpoint_accessed": False,
    }


def _config_payload(
    source: Mapping[str, Any],
    *,
    manifest_hash: str,
    contract_hash: str,
    train_cache_hash: str,
    validation_cache_hash: str,
) -> dict[str, Any]:
    payload = _clone(source)
    artifact = ARTIFACT.as_posix()
    payload["schema_version"] = "RADAR-LN3-V24"
    payload["config_name"] = (
        "RADAR-LN3-V24-cq08-liquid-known-gate-course-reason-aware-baseline-seed43"
    )
    payload["variant"] = V24_COURSE_REASON_AWARE_BASELINE_VARIANT
    payload["task"]["feature_names"] = list(COURSE_PREFIX_FEATURE_NAMES)
    payload["data"].update(
        {
            "feature_cache_manifest_path": FEATURE_MANIFEST.as_posix(),
            "expected_feature_cache_manifest_sha256": manifest_hash,
            "train_feature_cache_path": V21E_TRAIN_CACHE.as_posix(),
            "validation_feature_cache_path": V21E_VALIDATION_CACHE.as_posix(),
            "expected_train_feature_cache_sha256": train_cache_hash,
            "expected_validation_feature_cache_sha256": validation_cache_hash,
            "feature_experiment_contract_path": EXPERIMENT_CONTRACT.as_posix(),
            "expected_feature_experiment_contract_sha256": contract_hash,
        }
    )
    payload["model"]["evidence_dim"] = 1
    payload["train"].update(
        {
            # V24 deliberately retains the historical V21e loss denominator.
            # Write it explicitly instead of relying on the parser default so
            # the paired V24a--V24c controls remain self-describing.
            "class_weight_normalization": "batch_weight_sum",
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V24-v24_course_reason_aware_baseline-seed43-"
                "known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": V21_BASELINE_CONFIG.as_posix(),
        }
    )
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paths, hashes, baseline_payload = _source_payload(root)
    source_baseline_config_path = paths["baseline_config"]
    source_baseline_summary = json.loads(
        paths["baseline_summary"].read_text(encoding="utf-8")
    )
    v21e_summary = json.loads(
        paths["v21e_treatment_summary"].read_text(encoding="utf-8")
    )
    baseline_config = load_known_gate_config(source_baseline_config_path)
    reconstructed, reference, pairing = build_known_gate_model_for_config(
        root, baseline_config
    )
    reconstructed_sha = _state_dict_sha256(reconstructed.state_dict())
    if (
        reference is not None
        or pairing.get("paired_baseline") is not False
        or reconstructed_sha
        != str(source_baseline_summary.get("initial_model_state_sha256", ""))
    ):
        raise ValueError("V24 current code cannot reconstruct V21 seed43 initial state")
    del reconstructed

    cache_stats = {
        "train": _cache_metadata(paths["train_cache"], split="train"),
        "val": _cache_metadata(paths["validation_cache"], split="val"),
    }
    contract = {
        "schema_version": KNOWN_GATE_V24_EXPERIMENT_SCHEMA,
        "experiment_name": "V24 course-reason-aware fresh KnownGate baseline",
        "internal_variant": V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_document": _relative(root, paths["experiment_document"]),
        "experiment_document_sha256": hashes["experiment_document"],
        "scope": "development-only KnownGate L1 baseline; no product action claim",
        "fresh_training_not_finetune": True,
        "architecture": {
            "encoder": "self-developed CfCLiquidCausalEncoder",
            "liquid_hidden_dim": 48,
            "state_normalization": "recurrent",
            "learnable_per_dimension_tau": True,
            "rd_frame_aggregation": "masked_mean",
            "dt_reason_mode": "v3_reason_aware_dynamics_only",
            "zero_dt_mode": "bounded_observation",
            "zero_dt_observation_gain": 0.25,
        },
        "data_reuse": {
            "index": _relative(root, paths["index"]),
            "index_sha256": hashes["index"],
            "core_contract": _relative(root, paths["core_contract"]),
            "core_data_contract_sha256": hashes["core_contract"],
            "normalization": _relative(root, paths["normalization"]),
            "normalization_sha256": hashes["normalization"],
            "train_parent_count": TRAIN_PARENT_COUNT,
            "validation_parent_count": VALIDATION_PARENT_COUNT,
            "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
            "unknown_training_exposure": 0,
        },
        "core_data_contract_sha256": hashes["core_contract"],
        "course_cache_reuse": {
            "source_v21e_manifest": _relative(root, paths["v21e_manifest"]),
            "source_v21e_manifest_sha256": hashes["v21e_manifest"],
            "source_v21e_contract": _relative(root, paths["v21e_contract"]),
            "source_v21e_contract_sha256": hashes["v21e_contract"],
            "train_cache": _relative(root, paths["train_cache"]),
            "train_cache_sha256": hashes["train_cache"],
            "validation_cache": _relative(root, paths["validation_cache"]),
            "validation_cache_sha256": hashes["validation_cache"],
            "cache_values_rebuilt": False,
            "raw_echo_payload_access_count": 0,
            "mat_payload_access_count": 0,
            "rd_tensor_payload_access_count": 0,
            "test_payload_access_count": 0,
        },
        "initialization_protocol": {
            "source_config": _relative(root, source_baseline_config_path),
            "source_config_sha256": hashes["baseline_config"],
            "source_summary": _relative(root, paths["baseline_summary"]),
            "source_summary_sha256": hashes["baseline_summary"],
            "source_initial_model_state_sha256": reconstructed_sha,
            "v21e_treatment_summary": _relative(
                root, paths["v21e_treatment_summary"]
            ),
            "v21e_treatment_summary_sha256": hashes["v21e_treatment_summary"],
            "v21e_treatment_initial_model_state_sha256": str(
                v21e_summary["initial_model_state_sha256"]
            ),
            "trained_v21e_checkpoint_loaded": False,
            "trained_v21_baseline_checkpoint_loaded": False,
            "fresh_seed43_baseline_reconstructed": True,
            "shared_parameters_and_buffers_copied": True,
            "new_course_fusion_column_exactly_zero": True,
            "full_parameter_training": True,
        },
        "training": {
            "seed": SEED,
            "dataloader_seed": SEED,
            "epochs": 20,
            "batch_size": 2,
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": 0.1,
            "checkpoint_selection": "parent_min_precision_recall",
            "class_weight_normalization": "batch_weight_sum",
        },
        "evaluation": {
            "primary": "establish clean V24 baseline metrics for V24a-c paired controls",
            "primary_hypothesis_labels": [
                "DroneTarget",
                "BirdTarget",
                "BalloonTarget",
                "ClutterTarget",
            ],
            "safety_labels": [],
            "required_reports": [
                "parent_final",
                "eligible_prefix",
                "parent_balanced_eligible_prefix",
                "window",
                "clutter_wilson_intervals",
                "parent_stratified_bootstrap",
                "KnownTarget_to_Clutter_and_Clutter_to_Known_parent_audit",
            ],
            "clutter_validation_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
            "development_metrics_are_not_release_evidence": True,
        },
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    manifest = {
        "schema_version": KNOWN_GATE_V24_MANIFEST_SCHEMA,
        "feature_schema": COURSE_PREFIX_FEATURE_SCHEMA,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "source_manifest": V21E_MANIFEST.as_posix(),
        "source_manifest_sha256": hashes["v21e_manifest"],
        "source_experiment_contract": V21E_CONTRACT.as_posix(),
        "source_experiment_contract_sha256": hashes["v21e_contract"],
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": hashes["core_contract"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": hashes["index"],
        "cache_files": {
            "train": {
                "path": V21E_TRAIN_CACHE.as_posix(),
                "sha256": hashes["train_cache"],
                "fields": ["sample_ids", "values", "valid"],
                **cache_stats["train"],
            },
            "val": {
                "path": V21E_VALIDATION_CACHE.as_posix(),
                "sha256": hashes["validation_cache"],
                "fields": ["sample_ids", "values", "valid"],
                **cache_stats["val"],
            },
        },
        "sample_id_is_model_input": False,
        "valid_mask_is_model_input": False,
        "validation_label_values_used_for_readiness_gate": False,
        "unknown_excluded": True,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    _write_new_or_identical(root / FEATURE_MANIFEST, manifest)
    manifest_hash = sha256_file(root / FEATURE_MANIFEST)
    config_payload = _config_payload(
        baseline_payload,
        manifest_hash=manifest_hash,
        contract_hash=contract_hash,
        train_cache_hash=hashes["train_cache"],
        validation_cache_hash=hashes["validation_cache"],
    )
    _write_new_or_identical(root / CONFIG, config_payload)
    initialization = _initialization_audit(root, source_baseline_config_path, root / CONFIG)
    if (
        initialization["v24_initial_model_state_sha256"]
        != str(v21e_summary["initial_model_state_sha256"])
    ):
        raise AssertionError(
            "V24 fresh paired initialization does not exactly reproduce V21e initial state"
        )
    checks = {
        "v21_seed43_baseline_initial_state_reconstructs_exactly": True,
        "v24_is_fresh_training_not_checkpoint_finetune": True,
        "v24_variant_is_distinct_from_v21e": True,
        "course_cache_is_reused_byte_for_byte": True,
        "course_cache_contains_one_causal_course_feature": True,
        "shared_initialization_is_exact_and_new_column_is_zero": True,
        "v24_initial_state_exactly_reproduces_v21e_treatment_initial_state": True,
        "v16_index_v21_normalization_and_v21_contract_are_sha_bound": True,
        "raw_mat_rd_and_test_payload_access_are_zero": True,
        "validation_values_are_not_a_readiness_gate": True,
        "unknown_is_excluded_from_training": True,
        "full_20_epoch_training_is_required": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": KNOWN_GATE_V24_READINESS_SCHEMA,
        "variant": V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
        "checks": checks,
        "config": {
            "path": _relative(root, root / CONFIG),
            "sha256": sha256_file(root / CONFIG),
        },
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
        "initialization_source": {
            "config_path": _relative(root, source_baseline_config_path),
            "config_sha256": hashes["baseline_config"],
            "summary_path": _relative(root, paths["baseline_summary"]),
            "summary_sha256": hashes["baseline_summary"],
            "initial_model_state_sha256": initialization[
                "fresh_v21_seed43_baseline_initial_model_state_sha256"
            ],
            "fresh_random_initialization_reconstructed": True,
            "trained_checkpoint_loaded": False,
            "trained_checkpoint_accessed": False,
        },
        "initial_model_state_sha256": initialization[
            "v24_initial_model_state_sha256"
        ],
        "v21e_treatment_initial_model_state_sha256": str(
            v21e_summary["initial_model_state_sha256"]
        ),
        "paired_initialization": initialization,
        "cache_statistics": cache_stats,
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "unknown_training_exposure": 0,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": KNOWN_GATE_V24_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_cache": str(paths["train_cache"]),
        "validation_cache": str(paths["validation_cache"]),
        "initial_model_state_sha256": initialization["v24_initial_model_state_sha256"],
        "fresh_training_not_finetune": True,
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V24 fresh course-reason-aware KnownGate baseline."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
