"""Prepare V25a's gradient-telemetry-only replication of the V24 baseline.

V25a deliberately changes no model, data, loss, optimiser or checkpoint
criterion.  It creates a new artifact namespace solely because its trainer
records pre-clip gradient and batch-composition observations on every update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from build.known_gate_v24 import (
    ARTIFACT as V24_ARTIFACT,
    CONFIG as V24_CONFIG,
    CORE_CONTRACT,
    CORE_NORMALIZATION,
    EXPECTED_CACHE_EVENT_COUNTS,
    FEATURE_MANIFEST as V24_FEATURE_MANIFEST,
    READINESS as V24_READINESS,
    SOURCE_INDEX,
    TRAIN_PARENT_COUNT,
    V21_BASELINE_CONFIG,
    V21_BASELINE_SUMMARY,
    V21E_TRAIN_CACHE,
    V21E_VALIDATION_CACHE,
    VALIDATION_CLUTTER_PARENT_SUPPORT,
    VALIDATION_PARENT_COUNT,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V25A_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V25A_MANIFEST_SCHEMA,
    KNOWN_GATE_V25A_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_SCHEMA,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
    V25A_GRADIENT_TELEMETRY_VARIANT,
)


V24_SUMMARY = V24_ARTIFACT / "training-summary.json"
EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v25a_preclip_gradient_telemetry_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/RADAR-LN3-V25a-cq08-known-gate-course-preclip-gradient-"
    "telemetry-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V25a-cq08-known-gate-course-preclip-gradient-"
    "telemetry-feature-cache-manifest.json"
)
CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-"
    "gradient-telemetry-seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V25a-known-gate-course-preclip-gradient-"
    "telemetry-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-"
    "telemetry-seed43-formal"
)

SEED = 43


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"refusing to overwrite a different V25a preparation artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)
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
    expected_events = EXPECTED_CACHE_EVENT_COUNTS[split]
    with np.load(path, allow_pickle=False) as cache:
        if set(cache.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V25a {split} course cache fields differ")
        sample_ids = np.asarray(cache["sample_ids"])
        values = np.asarray(cache["values"])
        valid = np.asarray(cache["valid"])
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
        raise ValueError(f"V25a {split} frozen course cache is malformed")
    return {
        "event_count": expected_events,
        "feature_width": 1,
        "valid_value_count": int(valid.sum()),
        "invalid_value_count": int((~valid).sum()),
        "valid_value_fraction": float(valid.mean()),
    }


def _load_sources(root: Path) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths = {
        "index": root / SOURCE_INDEX,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "v24_config": root / V24_CONFIG,
        "v24_readiness": root / V24_READINESS,
        "v24_manifest": root / V24_FEATURE_MANIFEST,
        "v24_summary": root / V24_SUMMARY,
        "v21_baseline_config": root / V21_BASELINE_CONFIG,
        "v21_baseline_summary": root / V21_BASELINE_SUMMARY,
        "train_cache": root / V21E_TRAIN_CACHE,
        "validation_cache": root / V21E_VALIDATION_CACHE,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V25a frozen source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    v24_config_payload = json.loads(paths["v24_config"].read_text(encoding="utf-8"))
    v24_readiness = json.loads(paths["v24_readiness"].read_text(encoding="utf-8"))
    v24_manifest = json.loads(paths["v24_manifest"].read_text(encoding="utf-8"))
    v24_summary = json.loads(paths["v24_summary"].read_text(encoding="utf-8"))
    v21_baseline_config = json.loads(
        paths["v21_baseline_config"].read_text(encoding="utf-8")
    )
    v21_baseline_summary = json.loads(
        paths["v21_baseline_summary"].read_text(encoding="utf-8")
    )
    v24_initialization_source = dict(v24_readiness.get("initialization_source") or {})
    history = list(v24_summary.get("history") or ())
    orders = list(dict(v24_summary.get("dataloader_contract") or {}).get("epoch_parent_order_hashes") or ())
    if (
        v24_config_payload.get("schema_version") != "RADAR-LN3-V24"
        or v24_config_payload.get("variant") != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or v24_readiness.get("status") != "passed"
        or v24_readiness.get("variant") != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or dict(v24_readiness.get("config") or {}).get("path")
        != V24_CONFIG.as_posix()
        or dict(v24_readiness.get("config") or {}).get("sha256")
        != hashes["v24_config"]
        or v24_manifest.get("feature_schema") != COURSE_PREFIX_FEATURE_SCHEMA
        or v24_manifest.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or v24_summary.get("status") != "completed"
        or v24_summary.get("variant") != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or int(v24_summary.get("seed", -1)) != SEED
        or len(history) != 20
        or len(orders) != 20
        or [int(row.get("epoch", -1)) for row in history] != list(range(1, 21))
        or [int(row.get("epoch", -1)) for row in orders] != list(range(1, 21))
        or any(int(row.get("count", -1)) != TRAIN_PARENT_COUNT for row in orders)
        or any(not str(row.get("sha256", "")) for row in orders)
        or v24_summary.get("test_evaluation") != "not_executed"
        or v24_summary.get("deployment_eligible") is not False
        or not str(v24_summary.get("initial_model_state_sha256", ""))
        or v21_baseline_config.get("schema_version") != "RADAR-LN3-V21"
        or v21_baseline_config.get("variant") != "baseline"
        or int(dict(v21_baseline_config.get("train") or {}).get("seed", -1))
        != SEED
        or v21_baseline_summary.get("status") != "completed"
        or v21_baseline_summary.get("variant") != "baseline"
        or int(v21_baseline_summary.get("seed", -1)) != SEED
        or len(list(v21_baseline_summary.get("history") or ())) != 20
        or not str(v21_baseline_summary.get("initial_model_state_sha256", ""))
        or v24_initialization_source.get("config_path")
        != V21_BASELINE_CONFIG.as_posix()
        or v24_initialization_source.get("config_sha256")
        != hashes["v21_baseline_config"]
        or v24_initialization_source.get("summary_path")
        != V21_BASELINE_SUMMARY.as_posix()
        or v24_initialization_source.get("summary_sha256")
        != hashes["v21_baseline_summary"]
        or v24_initialization_source.get("initial_model_state_sha256")
        != v21_baseline_summary.get("initial_model_state_sha256")
    ):
        raise ValueError("V25a frozen V24 source/control contract is stale")
    return paths, hashes, {
        "config": v24_config_payload,
        "readiness": v24_readiness,
        "manifest": v24_manifest,
        "summary": v24_summary,
        "v21_baseline_config": v21_baseline_config,
        "v21_baseline_summary": v21_baseline_summary,
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
    payload["schema_version"] = "RADAR-LN3-V25a"
    payload["config_name"] = (
        "RADAR-LN3-V25a-cq08-liquid-known-gate-course-preclip-gradient-telemetry-seed43"
    )
    payload["variant"] = V25A_GRADIENT_TELEMETRY_VARIANT
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
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V25a-v25a_course_preclip_gradient_telemetry-seed43-"
                "known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": V21_BASELINE_CONFIG.as_posix(),
        }
    )
    return payload


def _initialization_audit(root: Path, config_path: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    v24_config = load_known_gate_config(root / V24_CONFIG)
    v25_config = load_known_gate_config(config_path)
    v24_model, v24_reference, v24_pairing = build_known_gate_model_for_config(root, v24_config)
    v25_model, v25_reference, v25_pairing = build_known_gate_model_for_config(root, v25_config)
    if v24_reference is None or v25_reference is None:
        raise AssertionError("V25a requires V24-style paired fresh initialization")
    v24_sha = _state_dict_sha256(v24_model.state_dict())
    v25_sha = _state_dict_sha256(v25_model.state_dict())
    v24_reference_sha = _state_dict_sha256(v24_reference.state_dict())
    v25_reference_sha = _state_dict_sha256(v25_reference.state_dict())
    expected_v24 = str(source["summary"]["initial_model_state_sha256"])
    expected_v21_reference = str(
        source["v21_baseline_summary"]["initial_model_state_sha256"]
    )
    if (
        v24_config.variant != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or v25_config.variant != V25A_GRADIENT_TELEMETRY_VARIANT
        or v25_config.feature_names != COURSE_PREFIX_FEATURE_NAMES
        or v25_config.model.evidence_dim != 1
        or v24_sha != expected_v24
        or v25_sha != expected_v24
        or v24_sha != v25_sha
        or v24_reference_sha != expected_v21_reference
        or v25_reference_sha != expected_v21_reference
        or v24_reference_sha != v25_reference_sha
        or v24_pairing.get("new_fusion_columns_exactly_zero") is not True
        or v25_pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(v24_pairing.get("new_fusion_columns", -1)) != 1
        or int(v25_pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise AssertionError("V25a initial model state differs from frozen V24")
    del v24_model, v24_reference, v25_model, v25_reference
    return {
        "v24_initial_model_state_sha256": v24_sha,
        "v25a_initial_model_state_sha256": v25_sha,
        "paired_v21_seed43_baseline_initial_model_state_sha256": v24_reference_sha,
        "initial_model_state_exactly_matches_v24": True,
        "paired_baseline_state_exactly_matches_v21_seed43_baseline": True,
        "course_fusion_column_exactly_zero": True,
        "fresh_training_not_finetune": True,
        "trained_v24_checkpoint_loaded": False,
        "trained_v24_checkpoint_accessed": False,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paths, hashes, source = _load_sources(root)
    cache_stats = {
        "train": _cache_metadata(paths["train_cache"], split="train"),
        "val": _cache_metadata(paths["validation_cache"], split="val"),
    }
    parent_orders = {
        str(row["epoch"]): str(row["sha256"])
        for row in source["summary"]["dataloader_contract"]["epoch_parent_order_hashes"]
    }
    contract = {
        "schema_version": KNOWN_GATE_V25A_EXPERIMENT_SCHEMA,
        "experiment_name": "V25a V24-replication pre-clip gradient and batch-composition audit",
        "internal_variant": V25A_GRADIENT_TELEMETRY_VARIANT,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_document": _relative(root, paths["experiment_document"]),
        "experiment_document_sha256": hashes["experiment_document"],
        "scope": "development-only KnownGate L1 telemetry audit; no product action claim",
        "fresh_training_not_finetune": True,
        "training_semantics": {
            "model": "exact V24 self-developed CfCLiquidCausalEncoder",
            "data": "exact V24 V21e course cache and V16/V21 contracts",
            "optimizer": "exact V24 torch.optim.AdamW(model.parameters(), lr=1e-4) defaults",
            "loss": "exact V24 parent_final_task_ce_with_prefix_auxiliary",
            "dt": "exact V24 bounded_observation reason-aware dynamics",
            "checkpoint_selection": "exact V24 parent_min_precision_recall",
            "class_weight_normalization": "batch_weight_sum",
            "gradient_clip_norm": 100.0,
            "optimizer_weight_decay": 0.01,
            "epochs": 20,
            "batch_size": 2,
            "telemetry_only_change": True,
            "telemetry_capture_timing": "after_loss_backward_before_existing_global_norm_clip",
            "telemetry_changes_gradients": False,
            "telemetry_changes_optimizer": False,
            "telemetry_changes_loss": False,
            "telemetry_changes_data_order": False,
        },
        "source_v24_control": {
            "config": _relative(root, paths["v24_config"]),
            "config_sha256": hashes["v24_config"],
            "readiness": _relative(root, paths["v24_readiness"]),
            "readiness_sha256": hashes["v24_readiness"],
            "feature_manifest": _relative(root, paths["v24_manifest"]),
            "feature_manifest_sha256": hashes["v24_manifest"],
            "summary": _relative(root, paths["v24_summary"]),
            "summary_sha256": hashes["v24_summary"],
            "trained_checkpoint_loaded": False,
            "trained_checkpoint_accessed": False,
            "initial_model_state_sha256": str(source["summary"]["initial_model_state_sha256"]),
            "epoch_parent_order_hashes": parent_orders,
            "fresh_initialization_reference": {
                "config": _relative(root, paths["v21_baseline_config"]),
                "config_sha256": hashes["v21_baseline_config"],
                "summary": _relative(root, paths["v21_baseline_summary"]),
                "summary_sha256": hashes["v21_baseline_summary"],
                "initial_model_state_sha256": str(
                    source["v21_baseline_summary"]["initial_model_state_sha256"]
                ),
            },
        },
        "data_reuse": {
            "index": _relative(root, paths["index"]),
            "index_sha256": hashes["index"],
            "core_contract": _relative(root, paths["core_contract"]),
            "core_data_contract_sha256": hashes["core_contract"],
            "normalization": _relative(root, paths["normalization"]),
            "normalization_sha256": hashes["normalization"],
            "train_cache": _relative(root, paths["train_cache"]),
            "train_cache_sha256": hashes["train_cache"],
            "validation_cache": _relative(root, paths["validation_cache"]),
            "validation_cache_sha256": hashes["validation_cache"],
            "known_train_parent_count": TRAIN_PARENT_COUNT,
            "known_validation_parent_count": VALIDATION_PARENT_COUNT,
            "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
            "unknown_training_exposure": 0,
            "raw_echo_payload_access_count": 0,
            "mat_payload_access_count": 0,
            "rd_tensor_payload_access_count": 0,
            "test_payload_access_count": 0,
        },
        "core_data_contract_sha256": hashes["core_contract"],
        "v24_equivalence_binding": {
            "model_config_sha256": hashes["v24_config"],
            "expected_initial_model_state_sha256": str(
                source["summary"]["initial_model_state_sha256"]
            ),
            "expected_epoch_parent_order_hashes": parent_orders,
            "trained_v24_checkpoint_loaded": False,
            "trained_v24_checkpoint_accessed": False,
        },
        "evaluation": {
            "primary": "diagnose whether clipping/gradient spikes are early, class-composition or module-concentrated",
            "primary_hypothesis_labels": [
                "DroneTarget",
                "BirdTarget",
                "BalloonTarget",
                "ClutterTarget",
            ],
            "safety_labels": [],
            "required_reports": [
                "per_step_preclip_gradient_norm",
                "clip_frequency_and_scale",
                "task_and_base_label_batch_composition",
                "TR_RD_fusion_liquid_classifier_gradient_attribution",
                "epoch_and_early_late_summary",
                "no_raw_parent_identifier_telemetry",
            ],
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
        "schema_version": KNOWN_GATE_V25A_MANIFEST_SCHEMA,
        "feature_schema": COURSE_PREFIX_FEATURE_SCHEMA,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "source_v24_manifest": V24_FEATURE_MANIFEST.as_posix(),
        "source_v24_manifest_sha256": hashes["v24_manifest"],
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
        source["config"],
        manifest_hash=manifest_hash,
        contract_hash=contract_hash,
        train_cache_hash=hashes["train_cache"],
        validation_cache_hash=hashes["validation_cache"],
    )
    _write_new_or_identical(root / CONFIG, config_payload)
    initialization = _initialization_audit(root, root / CONFIG, source)
    checks = {
        "v24_initial_model_state_reconstructs_exactly": True,
        "v25a_initial_model_state_exactly_matches_v24": True,
        "v25a_is_fresh_training_not_checkpoint_finetune": True,
        "v24_course_cache_is_reused_byte_for_byte": True,
        "model_loss_optimizer_dt_and_checkpoint_rule_are_frozen_to_v24": True,
        "adamw_default_weight_decay_and_global_clip_are_frozen_to_v24": True,
        "all_twenty_v24_parent_order_hashes_are_bound": True,
        "telemetry_is_preclip_read_only_and_uses_no_raw_parent_ids": True,
        "raw_mat_rd_and_test_payload_access_are_zero": True,
        "validation_values_are_not_a_readiness_gate": True,
        "unknown_is_excluded_from_training": True,
        "full_20_epoch_training_is_required": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": KNOWN_GATE_V25A_READINESS_SCHEMA,
        "variant": V25A_GRADIENT_TELEMETRY_VARIANT,
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
            "feature_manifest_sha256": manifest_hash,
            "train_feature_cache_sha256": hashes["train_cache"],
            "validation_feature_cache_sha256": hashes["validation_cache"],
            "experiment_contract_sha256": contract_hash,
        },
        "source_v24": contract["source_v24_control"],
        "initialization": initialization,
        "source_v24_initial_model_state_sha256": initialization[
            "v24_initial_model_state_sha256"
        ],
        "paired_v21_seed43_baseline_initial_model_state_sha256": initialization[
            "paired_v21_seed43_baseline_initial_model_state_sha256"
        ],
        "telemetry_schema": {
            "schema_version": (
                "RADAR-LN3-V25a-known-gate-preclip-gradient-batch-telemetry-v1"
            ),
            "capture_timing": "after_loss_backward_before_existing_global_norm_clip",
            "module_partition": [
                "tr_encoder",
                "rd_encoder",
                "fusion",
                "liquid",
                "classifier",
            ],
            "raw_parent_sample_ids_stored": False,
            "changes_gradients": False,
            "changes_optimizer": False,
            "changes_loss": False,
            "changes_data_order": False,
            "epoch_boundary_resume_only": True,
        },
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
        "schema_version": KNOWN_GATE_V25A_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_cache": str(paths["train_cache"]),
        "validation_cache": str(paths["validation_cache"]),
        "initial_model_state_sha256": initialization["v25a_initial_model_state_sha256"],
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V25a pre-clip gradient telemetry KnownGate replication."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
