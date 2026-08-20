"""Prepare the V22a RD current-event temporal residual experiment.

V22a does not rebuild an index or RD cache.  The only new model input is an
internal residual branch over the already-canonical current-event RD frames;
the V21e course cache is reused unchanged as the paired auxiliary input.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from build.micro_doppler_v17 import _write_json_new_or_identical
from radar_ln3.known_gate import (
    KNOWN_GATE_V22A_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V22A_MANIFEST_SCHEMA,
    KNOWN_GATE_V22A_READINESS_SCHEMA,
    V22A_SCHEMA_VERSION,
    V22A_RD_TEMPORAL_RESIDUAL_VARIANT,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)


V21E_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-seed43-config.json"
)
V21E_SUMMARY = Path(
    "artifacts/RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal/training-summary.json"
)
V21E_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-"
    "seed43-formal/best_model.pt"
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
V21E_VAL_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-val-causal-course-consistency-cache.npz"
)
V22A_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V22a-cq08-known-gate-rd-temporal-residual-contract.json"
)
V22A_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V22a-cq08-known-gate-rd-temporal-residual-feature-cache-manifest.json"
)
V22A_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V22a-cq08-liquid-known-gate-rd-temporal-residual-seed43-config.json"
)
V22A_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22a-known-gate-rd-temporal-residual-training-readiness.json"
)
V22A_ARTIFACT = Path(
    "artifacts/RADAR-LN3-V22a-cq08-liquid-known-gate-rd-temporal-residual-"
    "seed43-formal"
)
SEED = 43


def _clone(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _source_payload(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = root / V21E_CONFIG
    summary_path = root / V21E_SUMMARY
    checkpoint_path = root / V21E_CHECKPOINT
    manifest_path = root / V21E_MANIFEST
    contract_path = root / V21E_CONTRACT
    for path in (config_path, summary_path, checkpoint_path, manifest_path, contract_path,
                 root / V21E_TRAIN_CACHE, root / V21E_VAL_CACHE):
        if not path.is_file():
            raise FileNotFoundError(f"V22a frozen V21e source is missing: {path}")
    source = json.loads(config_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        source.get("schema_version") != "RADAR-LN3-V21"
        or source.get("variant") != "v21e_course_prefix"
        or int(source.get("train", {}).get("seed", -1)) != SEED
        or summary.get("status") != "completed"
        or summary.get("variant") != "v21e_course_prefix"
        or int(summary.get("seed", -1)) != SEED
        or len(summary.get("history") or []) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V22a V21e Control source is not a completed sealed seed43 run")
    return source, summary


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source, source_summary = _source_payload(root)
    source_config_path = root / V21E_CONFIG
    source_manifest_path = root / V21E_MANIFEST
    source_contract_path = root / V21E_CONTRACT
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    source_config = load_known_gate_config(source_config_path)
    source_config_sha = sha256_file(source_config_path)
    source_manifest_sha = sha256_file(source_manifest_path)
    source_contract_sha = sha256_file(source_contract_path)
    source_data = source_config.data
    source_paths = {
        "index": _resolve(root, source_data.index_path),
        "contract": _resolve(root, source_data.contract_path),
        "stats": _resolve(root, source_data.normalization_stats_path),
        "manifest": source_manifest_path,
        "train_cache": _resolve(root, source_data.train_feature_cache_path or ""),
        "validation_cache": _resolve(
            root, source_data.validation_feature_cache_path or ""
        ),
        "experiment": _resolve(
            root, source_data.feature_experiment_contract_path or ""
        ),
        "checkpoint": root / V21E_CHECKPOINT,
        "summary": root / V21E_SUMMARY,
    }
    source_expected_hashes = {
        "index": source_data.expected_index_sha256,
        "contract": source_data.expected_contract_sha256,
        "stats": source_data.expected_stats_sha256,
        "manifest": source_data.expected_feature_cache_manifest_sha256,
        "train_cache": source_data.expected_train_feature_cache_sha256,
        "validation_cache": source_data.expected_validation_feature_cache_sha256,
        "experiment": source_data.expected_feature_experiment_contract_sha256,
    }
    for name, path in source_paths.items():
        if name in source_expected_hashes:
            if not path.is_file() or sha256_file(path) != source_expected_hashes[name]:
                raise ValueError(
                    f"V22a frozen V21e source {name} path/hash is not bound"
                )
    source_cache_files = dict(source_manifest.get("cache_files") or {})
    source_train_meta = dict(source_cache_files.get("train") or {})
    source_val_meta = dict(source_cache_files.get("val") or {})
    if (
        _resolve(root, str(source_manifest.get("experiment_contract", "")))
        != source_paths["experiment"]
        or source_manifest.get("experiment_contract_sha256") != source_contract_sha
        or _resolve(root, str(source_manifest.get("core_data_contract", "")))
        != source_paths["contract"]
        or source_manifest.get("core_data_contract_sha256")
        != source_data.expected_contract_sha256
        or _resolve(root, str(source_train_meta.get("path", "")))
        != source_paths["train_cache"]
        or source_train_meta.get("sha256")
        != source_data.expected_train_feature_cache_sha256
        or _resolve(root, str(source_val_meta.get("path", "")))
        != source_paths["validation_cache"]
        or source_val_meta.get("sha256")
        != source_data.expected_validation_feature_cache_sha256
        or _resolve(root, str(source_config.data.feature_experiment_contract_path or ""))
        != source_paths["experiment"]
    ):
        raise ValueError("V22a frozen V21e source cache/config binding is inconsistent")
    if (
        _resolve(root, str(source_summary.get("best_checkpoint", "")))
        != source_paths["checkpoint"]
        or source_summary.get("full_parameter_training") is not True
        or source_summary.get("all_model_parameters_trainable") is not True
        or source_summary.get("unknown_is_model_class") is not False
        or dict(source_summary.get("training_readiness") or {}).get("status")
        != "passed"
    ):
        raise ValueError("V22a frozen V21e source summary/checkpoint binding is inconsistent")
    if (
        source_manifest.get("schema_version")
        != "RADAR-LN3-V21e-known-gate-causal-course-consistency-feature-cache-manifest-v1"
        or source_manifest.get("test_excluded") is not True
        or int(source_manifest.get("test_payload_access_count", -1)) != 0
        or source_contract.get("test") != "未执行"
        or int(source_contract.get("test_payload_access_count", -1)) != 0
    ):
        raise ValueError("V22a V21e cache/contract source violates the sealed contract")

    contract = {
        "schema_version": KNOWN_GATE_V22A_EXPERIMENT_SCHEMA,
        "experiment_name": "V22a RD current-event temporal residual",
        "internal_variant": V22A_RD_TEMPORAL_RESIDUAL_VARIANT,
        "scope": "KnownGate local L1 only; no Unknown model class",
        "control": {
            "config": _relative(root, source_config_path),
            "config_sha256": source_config_sha,
            "model_rd_frame_aggregation": "masked_mean",
            "trained_reference_checkpoint": _relative(root, root / V21E_CHECKPOINT),
            "trained_reference_checkpoint_sha256": sha256_file(root / V21E_CHECKPOINT),
            "course_cache_reused": True,
        },
        "treatment": {
            "new_branch": "last_active_rd_frame_embedding - first_active_rd_frame_embedding",
            "rd_temporal_residual_dim": int(source_config.model.rd_embedding_dim),
            "fusion_columns_zero_initialized": True,
            "single_factor": "current_event RD frame order information only",
            "missing_or_single_frame_residual": "exact_zero",
            "padding_policy": "rd_frame_mask excludes padding from first/last selection and gradient",
            "causal_scope": "current TR event RD frames only; no future TR event",
        },
        "data_contract": {
            "index_reused": _relative(root, root / source_config.data.index_path),
            "core_contract_reused": _relative(root, root / source_config.data.contract_path),
            "normalization_reused": _relative(root, root / source_config.data.normalization_stats_path),
            "rd_tensor_rebuilt": False,
            "new_input_fields": 0,
            "test_payload_access_count": 0,
        },
        "feature_cache_reuse": {
            "manifest": _relative(root, source_manifest_path),
            "manifest_sha256": source_manifest_sha,
            "contract": _relative(root, source_contract_path),
            "contract_sha256": source_contract_sha,
            "train_cache": _relative(root, root / V21E_TRAIN_CACHE),
            "train_cache_sha256": sha256_file(root / V21E_TRAIN_CACHE),
            "validation_cache": _relative(root, root / V21E_VAL_CACHE),
            "validation_cache_sha256": sha256_file(root / V21E_VAL_CACHE),
        },
        "training": {
            "epochs": 20,
            "seed": SEED,
            "dataloader_seed": SEED,
            "loss_mode": source_config.train.loss_mode,
            "class_weight_normalization": source_config.train.class_weight_normalization,
            "checkpoint_selection": source_config.train.checkpoint_selection_metric,
            "full_parameter_training": True,
            "control_retrained_in_this_pipeline": False,
        },
        "evaluation": {
            "primary": "development validation parent-final KnownGate macro_f1 delta",
            "adoption_delta_strictly_greater_than": 0.02,
            "required_safety": [
                "Clutter precision/recall/f1 non-decrease",
                "KnownTarget recall non-decrease",
                "Drone/Bird/Balloon recall non-decrease",
                "eligible-prefix metrics non-decrease",
            ],
            "required_stratification": [
                "RD frame count K=1",
                "RD frame count K=2-4",
                "RD frame count K>=6",
                "RD quality",
                "source",
                "parent length",
            ],
            "clutter_validation_support": 9,
            "parent_bootstrap_required": True,
            "test": "未执行",
            "OOF": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "P95": "未执行",
        },
        "unknown_training_exposure": 0,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
        "test": "未执行",
    }
    _write_json_new_or_identical(root / V22A_CONTRACT, contract)
    contract_sha = sha256_file(root / V22A_CONTRACT)

    manifest = {
        "schema_version": KNOWN_GATE_V22A_MANIFEST_SCHEMA,
        "feature_schema": "cq08-causal-prefix-course-circular-incoherence-v1",
        "feature_names": list(source_config.feature_names),
        "source_manifest": _relative(root, source_manifest_path),
        "source_manifest_sha256": source_manifest_sha,
        "experiment_contract": _relative(root, root / V22A_CONTRACT),
        "experiment_contract_sha256": contract_sha,
        "cache_files": {
            "train": {
                "path": _relative(root, root / V21E_TRAIN_CACHE),
                "sha256": sha256_file(root / V21E_TRAIN_CACHE),
                "fields": ["sample_ids", "values", "valid"],
            },
            "val": {
                "path": _relative(root, root / V21E_VAL_CACHE),
                "sha256": sha256_file(root / V21E_VAL_CACHE),
                "fields": ["sample_ids", "values", "valid"],
            },
        },
        "internal_rd_residual_is_not_cache_field": True,
        "sample_id_is_model_input": False,
        "valid_mask_is_model_input": False,
        "unknown_excluded": True,
        "test_excluded": True,
        "test_payload_access_count": 0,
    }
    _write_json_new_or_identical(root / V22A_MANIFEST, manifest)
    manifest_sha = sha256_file(root / V22A_MANIFEST)

    payload = _clone(source)
    payload["schema_version"] = V22A_SCHEMA_VERSION
    payload["config_name"] = (
        "RADAR-LN3-V22a-cq08-liquid-known-gate-rd-temporal-residual-seed43"
    )
    payload["variant"] = V22A_RD_TEMPORAL_RESIDUAL_VARIANT
    payload["data"]["feature_cache_manifest_path"] = _relative(root, root / V22A_MANIFEST)
    payload["data"]["expected_feature_cache_manifest_sha256"] = manifest_sha
    payload["data"]["train_feature_cache_path"] = _relative(root, root / V21E_TRAIN_CACHE)
    payload["data"]["validation_feature_cache_path"] = _relative(root, root / V21E_VAL_CACHE)
    payload["data"]["expected_train_feature_cache_sha256"] = sha256_file(root / V21E_TRAIN_CACHE)
    payload["data"]["expected_validation_feature_cache_sha256"] = sha256_file(root / V21E_VAL_CACHE)
    payload["data"]["feature_experiment_contract_path"] = _relative(root, root / V22A_CONTRACT)
    payload["data"]["expected_feature_experiment_contract_sha256"] = contract_sha
    payload["model"]["evidence_dim"] = len(source_config.feature_names)
    payload["model"]["rd_frame_aggregation"] = "masked_mean"
    payload["model"]["rd_temporal_residual_dim"] = int(source_config.model.rd_embedding_dim)
    artifact = root / V22A_ARTIFACT
    payload["train"].update(
        {
            "checkpoint_path": _relative(root, artifact / "best_model.pt"),
            "last_checkpoint_path": _relative(root, artifact / "last_model.pt"),
            "formal_artifact_dir": _relative(root, artifact),
            "epoch_checkpoint_dir": _relative(root, artifact / "epochs"),
            "checkpoint_schema": (
                "RADAR-LN3-V22a-v22a_rd_temporal_residual-seed43-known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": _relative(root, source_config_path),
            "save_every_epoch": True,
        }
    )
    _write_json_new_or_identical(root / V22A_CONFIG, payload)
    config = load_known_gate_config(root / V22A_CONFIG)
    treatment, control, pairing = build_known_gate_model_for_config(root, config)
    if (
        control is None
        or pairing.get("rd_temporal_residual_pair") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1))
        != int(source_config.model.rd_embedding_dim)
    ):
        raise ValueError("V22a paired initialization contract failed")
    del treatment, control

    checks = {
        "v21e_seed43_control_is_completed_and_sealed": True,
        "v21e_course_cache_reused_without_rebuild": True,
        "rd_frame_aggregation_remains_masked_mean": True,
        "rd_temporal_residual_is_internal_not_external_field": True,
        "residual_reads_current_event_only": True,
        "missing_or_single_frame_residual_is_zero": True,
        "padding_is_excluded_from_residual": True,
        "unknown_is_excluded_from_training": True,
        "test_payload_access_count_is_zero": True,
        "treatment_copies_v21e_shared_state_and_zeroes_residual_columns": True,
        "full_parameter_20_epoch_training_is_required": True,
        "validation_values_are_not_a_readiness_gate": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": KNOWN_GATE_V22A_READINESS_SCHEMA,
        "experiment": "V22a RD current-event temporal residual",
        "variant": V22A_RD_TEMPORAL_RESIDUAL_VARIANT,
        "checks": checks,
        "config": {"path": str(root / V22A_CONFIG), "sha256": sha256_file(root / V22A_CONFIG)},
        "control": {
            "config_path": str(root / V21E_CONFIG),
            "config_sha256": source_config_sha,
            "summary_path": str(root / V21E_SUMMARY),
            "summary_sha256": sha256_file(root / V21E_SUMMARY),
            "checkpoint_path": str(root / V21E_CHECKPOINT),
            "checkpoint_sha256": sha256_file(root / V21E_CHECKPOINT),
        },
        "contract_path": str(root / V22A_CONTRACT),
        "contract_sha256": contract_sha,
        "feature_manifest_path": str(root / V22A_MANIFEST),
        "feature_manifest_sha256": manifest_sha,
        "reused_cache_hashes": {
            "train": sha256_file(root / V21E_TRAIN_CACHE),
            "validation": sha256_file(root / V21E_VAL_CACHE),
        },
        "rd_temporal_residual_dim": int(source_config.model.rd_embedding_dim),
        "known_train_parent_count": 986,
        "known_validation_parent_count": 232,
        "validation_clutter_support": 9,
        "unknown_training_exposure": 0,
        "validation_used_for_readiness_gate": False,
        "test_payload_access_count": 0,
        "formal_training": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    _write_json_new_or_identical(root / V22A_READINESS, readiness)
    return {
        "status": readiness["status"],
        "schema_version": KNOWN_GATE_V22A_READINESS_SCHEMA,
        "config": str(root / V22A_CONFIG),
        "contract": str(root / V22A_CONTRACT),
        "manifest": str(root / V22A_MANIFEST),
        "readiness": str(root / V22A_READINESS),
        "control": str(root / V21E_CONFIG),
        "train_parent_count": 986,
        "validation_parent_count": 232,
        "validation_clutter_support": 9,
        "test": "未执行",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare V22a KnownGate RD temporal residual.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
