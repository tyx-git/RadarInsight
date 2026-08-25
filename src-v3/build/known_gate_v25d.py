"""Prepare V25d's present-class-mean mixed-batch loss experiment.

V25d is deliberately a loss-only treatment of the completed V25a run.  It
reuses the V21e causal-course cache byte-for-byte and starts from the exact
same freshly reconstructed model state.  The only training-time difference is
the reduction of a batch that contains more than one KnownGate class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from build.known_gate_v25a import (
    ARTIFACT as V25A_ARTIFACT,
    CONFIG as V25A_CONFIG,
    CORE_CONTRACT,
    CORE_NORMALIZATION,
    EXPECTED_CACHE_EVENT_COUNTS,
    FEATURE_MANIFEST as V25A_FEATURE_MANIFEST,
    READINESS as V25A_READINESS,
    SOURCE_INDEX,
    TRAIN_PARENT_COUNT,
    V21E_TRAIN_CACHE,
    V21E_VALIDATION_CACHE,
    VALIDATION_CLUTTER_PARENT_SUPPORT,
    VALIDATION_PARENT_COUNT,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
    KNOWN_GATE_V25D_EXPERIMENT_SCHEMA,
    KNOWN_GATE_V25D_MANIFEST_SCHEMA,
    KNOWN_GATE_V25D_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    COURSE_PREFIX_FEATURE_SCHEMA,
    V25A_GRADIENT_TELEMETRY_VARIANT,
    V25D_PRESENT_CLASS_MEAN_VARIANT,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary


V25A_SUMMARY = V25A_ARTIFACT / "training-summary.json"
V25A_AUDIT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25a-known-gate-course-preclip-gradient-telemetry-audit.json"
)
V25C_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-"
    "localization.json"
)
EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v25d_present_class_mean_mixed_batch_experiment.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/RADAR-LN3-V25d-cq08-known-gate-course-present-class-"
    "mean-contract.json"
)
FEATURE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V25d-cq08-known-gate-course-present-class-"
    "mean-feature-cache-manifest.json"
)
CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V25d-cq08-liquid-known-gate-course-present-"
    "class-mean-seed43-config.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V25d-known-gate-course-present-class-"
    "mean-training-readiness.json"
)
ARTIFACT = Path(
    "artifacts/RADAR-LN3-V25d-cq08-liquid-known-gate-course-present-class-"
    "mean-seed43-formal"
)

SEED = 43
EXPECTED_EPOCHS = tuple(range(1, 21))
EXPECTED_STEPS_PER_EPOCH = 493


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                "refusing to overwrite a different V25d preparation artifact: "
                f"{path}"
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
            raise ValueError(f"V25d {split} course cache fields differ")
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
        raise ValueError(f"V25d {split} frozen course cache is malformed")
    return {
        "event_count": expected_events,
        "feature_width": 1,
        "valid_value_count": int(valid.sum()),
        "invalid_value_count": int((~valid).sum()),
        "valid_value_fraction": float(valid.mean()),
    }


def _source_orders(summary: Mapping[str, Any]) -> dict[str, str]:
    rows = list(dict(summary.get("dataloader_contract") or {}).get("epoch_parent_order_hashes") or ())
    if len(rows) != len(EXPECTED_EPOCHS):
        raise ValueError("V25d V25a source lacks all twenty parent-order hashes")
    orders = {str(index): str(row.get("sha256", "")) for index, row in enumerate(rows, start=1)}
    if any(not value for value in orders.values()):
        raise ValueError("V25d V25a source contains an empty parent-order hash")
    return orders


def _load_sources(root: Path) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    paths = {
        "index": root / SOURCE_INDEX,
        "core_contract": root / CORE_CONTRACT,
        "normalization": root / CORE_NORMALIZATION,
        "v25a_config": root / V25A_CONFIG,
        "v25a_readiness": root / V25A_READINESS,
        "v25a_manifest": root / V25A_FEATURE_MANIFEST,
        "v25a_summary": root / V25A_SUMMARY,
        "v25a_audit": root / V25A_AUDIT,
        "v25c_report": root / V25C_REPORT,
        "train_cache": root / V21E_TRAIN_CACHE,
        "validation_cache": root / V21E_VALIDATION_CACHE,
        "experiment_document": root / EXPERIMENT_DOCUMENT,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V25d frozen source files are missing: {missing}")
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    v25a_config = load_known_gate_config(paths["v25a_config"])
    readiness = json.loads(paths["v25a_readiness"].read_text(encoding="utf-8"))
    manifest = json.loads(paths["v25a_manifest"].read_text(encoding="utf-8"))
    summary = json.loads(paths["v25a_summary"].read_text(encoding="utf-8"))
    audit = json.loads(paths["v25a_audit"].read_text(encoding="utf-8"))
    v25c = json.loads(paths["v25c_report"].read_text(encoding="utf-8"))
    history = list(summary.get("history") or ())
    orders = _source_orders(summary)
    if (
        v25a_config.variant != V25A_GRADIENT_TELEMETRY_VARIANT
        or v25a_config.schema_version != "RADAR-LN3-V25a"
        or v25a_config.feature_names != COURSE_PREFIX_FEATURE_NAMES
        or v25a_config.model.evidence_dim != 1
        or v25a_config.train.class_weight_normalization != "batch_weight_sum"
        or readiness.get("status") != "passed"
        or readiness.get("variant") != V25A_GRADIENT_TELEMETRY_VARIANT
        or dict(readiness.get("config") or {}).get("path") != V25A_CONFIG.as_posix()
        or dict(readiness.get("config") or {}).get("sha256") != hashes["v25a_config"]
        or manifest.get("feature_schema") != COURSE_PREFIX_FEATURE_SCHEMA
        or manifest.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        or summary.get("status") != "completed"
        or summary.get("variant") != V25A_GRADIENT_TELEMETRY_VARIANT
        or int(summary.get("seed", -1)) != SEED
        or int(summary.get("global_step", -1)) != len(EXPECTED_EPOCHS) * EXPECTED_STEPS_PER_EPOCH
        or len(history) != len(EXPECTED_EPOCHS)
        or [int(row.get("epoch", -1)) for row in history] != list(EXPECTED_EPOCHS)
        or not str(summary.get("initial_model_state_sha256", ""))
        or audit.get("status") != "completed_development_gradient_telemetry_audit_only"
        or int(dict(audit.get("integrity") or {}).get("record_count", -1)) != len(EXPECTED_EPOCHS) * EXPECTED_STEPS_PER_EPOCH
        or dict(audit.get("integrity") or {}).get("all_source_v24_parent_orders_reproduced") is not True
        or v25c.get("status") != "completed_development_exact_replay_module_gradient_conflict_diagnosis_only"
        or int(dict(v25c.get("integrity") or {}).get("replayed_step_count", -1)) != len(EXPECTED_EPOCHS) * EXPECTED_STEPS_PER_EPOCH
        or dict(v25c.get("integrity") or {}).get("every_parent_batch_hash_matched_exactly") is not True
    ):
        raise ValueError("V25d frozen V25a/V25c source contract is stale")
    return paths, hashes, {
        "config": json.loads(paths["v25a_config"].read_text(encoding="utf-8")),
        "readiness": readiness,
        "manifest": manifest,
        "summary": summary,
        "parent_orders": orders,
    }


def _loss_semantics_audit() -> dict[str, Any]:
    """Prove V25d's only new reduction semantics on deterministic tensors."""

    torch.manual_seed(25043)
    logits = torch.randn(2, 5, 2, dtype=torch.float32)
    padding = torch.zeros(2, 5, dtype=torch.bool)
    weights = torch.tensor([0.5173137460650578, 14.93939393939394], dtype=torch.float32)
    mixed = parent_final_task_ce_with_prefix_auxiliary(
        logits,
        torch.tensor([0, 1], dtype=torch.long),
        padding,
        class_weights=weights,
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
        class_weight_normalization=KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
    )
    pure_checks: dict[str, bool] = {}
    for class_id in (0, 1):
        targets = torch.full((2,), class_id, dtype=torch.long)
        historical = parent_final_task_ce_with_prefix_auxiliary(
            logits,
            targets,
            padding,
            class_weights=weights,
            prefix_auxiliary_weight=0.1,
            prefix_auxiliary_min_events=3,
            class_weight_normalization="batch_weight_sum",
        )
        treatment = parent_final_task_ce_with_prefix_auxiliary(
            logits,
            targets,
            padding,
            class_weights=weights,
            prefix_auxiliary_weight=0.1,
            prefix_auxiliary_min_events=3,
            class_weight_normalization=KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
        )
        pure_checks[str(class_id)] = all(
            torch.allclose(historical[key], treatment[key], atol=2.0e-6, rtol=2.0e-6)
            for key in ("loss", "final_state_loss", "prefix_auxiliary_loss")
        )
    final_mass = mixed["final_effective_class_mass"].detach().cpu().tolist()
    prefix_mass = mixed["prefix_effective_class_mass"].detach().cpu().tolist()
    if (
        not np.allclose(final_mass, [0.5, 0.5], atol=0.0, rtol=0.0)
        or not np.allclose(prefix_mass, [0.5, 0.5], atol=0.0, rtol=0.0)
        or not all(pure_checks.values())
    ):
        raise AssertionError("V25d present-class-mean loss semantics audit failed")
    return {
        "mixed_final_effective_class_mass": final_mass,
        "mixed_prefix_effective_class_mass": prefix_mass,
        "pure_class_loss_equivalent_to_batch_weight_sum": pure_checks,
        "legacy_class_weights_are_not_multipliers_in_present_class_mean": True,
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
    payload["schema_version"] = "RADAR-LN3-V25d"
    payload["config_name"] = (
        "RADAR-LN3-V25d-cq08-liquid-known-gate-course-present-class-mean-seed43"
    )
    payload["variant"] = V25D_PRESENT_CLASS_MEAN_VARIANT
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
                "RADAR-LN3-V25d-v25d_course_present_class_mean-seed43-"
                "known-gate-checkpoint-v1"
            ),
            "paired_baseline_config_path": V25A_CONFIG.as_posix(),
            "class_weight_normalization": KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
        }
    )
    return payload


def _initialization_audit(root: Path, config_path: Path, source: Mapping[str, Any]) -> dict[str, Any]:
    control_config = load_known_gate_config(root / V25A_CONFIG)
    treatment_config = load_known_gate_config(config_path)
    control, control_reference, control_pairing = build_known_gate_model_for_config(root, control_config)
    treatment, treatment_reference, treatment_pairing = build_known_gate_model_for_config(root, treatment_config)
    if control_reference is None or treatment_reference is None:
        raise AssertionError("V25d requires paired fresh initialization references")
    control_sha = _state_dict_sha256(control.state_dict())
    treatment_sha = _state_dict_sha256(treatment.state_dict())
    treatment_reference_sha = _state_dict_sha256(treatment_reference.state_dict())
    expected = str(source["summary"]["initial_model_state_sha256"])
    if (
        control_sha != expected
        or treatment_sha != expected
        or treatment_reference_sha != expected
        or control_sha != treatment_sha
        or control_sha != treatment_reference_sha
        or control_pairing.get("new_fusion_columns_exactly_zero") is not True
        or treatment_pairing.get("loss_only_pair") is not True
        or treatment_pairing.get("present_class_mean_loss_only_pair") is not True
        or int(treatment_pairing.get("new_fusion_columns", -1)) != 0
    ):
        raise AssertionError("V25d initial model state differs from frozen V25a control")
    del control, control_reference, treatment, treatment_reference
    return {
        "v25a_initial_model_state_sha256": control_sha,
        "v25d_initial_model_state_sha256": treatment_sha,
        "paired_v25a_state_exactly_matches_source": True,
        "treatment_initial_state_exactly_matches_control": True,
        "loss_only_pair": True,
        "fresh_training_not_finetune": True,
        "trained_v25a_checkpoint_loaded": False,
        "trained_v25a_checkpoint_accessed": False,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paths, hashes, source = _load_sources(root)
    cache_stats = {
        "train": _cache_metadata(paths["train_cache"], split="train"),
        "val": _cache_metadata(paths["validation_cache"], split="val"),
    }
    loss_semantics = _loss_semantics_audit()
    contract = {
        "schema_version": KNOWN_GATE_V25D_EXPERIMENT_SCHEMA,
        "experiment_name": "V25d present-class-mean mixed-batch loss-only KnownGate treatment",
        "internal_variant": V25D_PRESENT_CLASS_MEAN_VARIANT,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_document": _relative(root, paths["experiment_document"]),
        "experiment_document_sha256": hashes["experiment_document"],
        "scope": "development-only KnownGate L1 loss-reduction experiment; no product action claim",
        "fresh_training_not_finetune": True,
        "sole_variable": {
            "field": "train.class_weight_normalization",
            "control": "batch_weight_sum",
            "treatment": KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
            "final_reduction": "mean CE inside each present class, then mean across present classes",
            "prefix_reduction": "mean legal prefixes inside each parent, then mean parents inside each present class, then mean classes",
            "mixed_known_clutter_effective_mass": [0.5, 0.5],
            "pure_class_equivalence_to_historical_weighted_mean": True,
            "new_hyperparameter": False,
        },
        "frozen_training_semantics": {
            "model": "exact V25a/V24 self-developed CfCLiquidCausalEncoder",
            "data": "exact V25a V21e causal-course cache and V16/V21 contracts",
            "optimizer": "exact V25a torch.optim.AdamW(model.parameters(), lr=1e-4) defaults",
            "dt": "exact V25a bounded_observation reason-aware dynamics",
            "checkpoint_selection": "exact V25a parent_min_precision_recall",
            "gradient_clip_norm": 100.0,
            "optimizer_weight_decay": 0.01,
            "epochs": 20,
            "batch_size": 2,
            "feature_change": False,
            "model_change": False,
            "sampler_change": False,
            "data_order_change": False,
            "initialization_change": False,
        },
        "source_v25a_control": {
            "config": _relative(root, paths["v25a_config"]),
            "config_sha256": hashes["v25a_config"],
            "readiness": _relative(root, paths["v25a_readiness"]),
            "readiness_sha256": hashes["v25a_readiness"],
            "feature_manifest": _relative(root, paths["v25a_manifest"]),
            "feature_manifest_sha256": hashes["v25a_manifest"],
            "summary": _relative(root, paths["v25a_summary"]),
            "summary_sha256": hashes["v25a_summary"],
            "telemetry_audit": _relative(root, paths["v25a_audit"]),
            "telemetry_audit_sha256": hashes["v25a_audit"],
            "gradient_conflict_diagnosis": _relative(root, paths["v25c_report"]),
            "gradient_conflict_diagnosis_sha256": hashes["v25c_report"],
            "initial_model_state_sha256": str(source["summary"]["initial_model_state_sha256"]),
            "epoch_parent_order_hashes": source["parent_orders"],
            "trained_checkpoint_loaded": False,
            "trained_checkpoint_accessed": False,
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
        "pre_registered_decision": {
            "parent_final_clutter_recall_required": "9/9",
            "parent_final_known_target_recall_required": "223/223",
            "parent_final_clutter_precision_floor": 0.9,
            "primary_effect": "parent-balanced eligible-prefix Clutter F1 improves by at least 0.02 without reducing its precision/recall or KnownTarget prefix recall/macro-F1",
            "mechanism_evidence": "mixed final and eligible-prefix effective class mass must each be 0.5/0.5; pure-class batches remain historical-equivalent",
            "post_pass_next_step": "multi-seed confirmation before OOF",
        },
        "evaluation": {
            "primary": "test whether present-class mean changes the causal-prefix Clutter boundary while retaining parent-final safety",
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
                "paired_parent_bootstrap",
                "effective_class_mass_telemetry",
                "directional_prediction_changes",
            ],
            "development_metrics_are_not_release_evidence": True,
        },
        "loss_semantics_audit": loss_semantics,
        "validation_used_for_readiness_gate": False,
        "OOF": "not_executed",
        "test": "not_executed",
        "test_payload_access_count": 0,
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    manifest = {
        "schema_version": KNOWN_GATE_V25D_MANIFEST_SCHEMA,
        "feature_schema": COURSE_PREFIX_FEATURE_SCHEMA,
        "feature_names": list(COURSE_PREFIX_FEATURE_NAMES),
        "experiment_contract": EXPERIMENT_CONTRACT.as_posix(),
        "experiment_contract_sha256": contract_hash,
        "source_v25a_manifest": V25A_FEATURE_MANIFEST.as_posix(),
        "source_v25a_manifest_sha256": hashes["v25a_manifest"],
        "core_data_contract": CORE_CONTRACT.as_posix(),
        "core_data_contract_sha256": hashes["core_contract"],
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": hashes["index"],
        "cache_files": {
            "train": {"path": V21E_TRAIN_CACHE.as_posix(), "sha256": hashes["train_cache"], "fields": ["sample_ids", "values", "valid"], **cache_stats["train"]},
            "val": {"path": V21E_VALIDATION_CACHE.as_posix(), "sha256": hashes["validation_cache"], "fields": ["sample_ids", "values", "valid"], **cache_stats["val"]},
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
        "v25a_completed_control_and_v25c_mechanism_diagnosis_are_bound": True,
        "v25d_is_fresh_training_not_checkpoint_finetune": True,
        "v25a_course_cache_is_reused_byte_for_byte": True,
        "model_dt_optimizer_clip_and_checkpoint_rule_are_frozen_to_v25a": True,
        "all_twenty_v25a_parent_order_hashes_are_bound": True,
        "mixed_present_class_mean_is_exactly_equal_class_mass": True,
        "pure_class_loss_is_historical_equivalent": True,
        "raw_mat_rd_and_test_payload_access_are_zero": True,
        "validation_values_are_not_a_readiness_gate": True,
        "unknown_is_excluded_from_training": True,
        "full_20_epoch_training_is_required": True,
    }
    readiness = {
        "status": "passed" if all(checks.values()) else "failed",
        "schema_version": KNOWN_GATE_V25D_READINESS_SCHEMA,
        "variant": V25D_PRESENT_CLASS_MEAN_VARIANT,
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
        "source_v25a": contract["source_v25a_control"],
        "source_v25a_initial_model_state_sha256": initialization["v25a_initial_model_state_sha256"],
        "source_v25a_epoch_parent_order_hashes": source["parent_orders"],
        "initialization": initialization,
        "loss_semantics_audit": loss_semantics,
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
        "schema_version": KNOWN_GATE_V25D_READINESS_SCHEMA,
        "config": str(root / CONFIG),
        "readiness": str(root / READINESS),
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "feature_manifest": str(root / FEATURE_MANIFEST),
        "train_cache": str(paths["train_cache"]),
        "validation_cache": str(paths["validation_cache"]),
        "initial_model_state_sha256": initialization["v25d_initial_model_state_sha256"],
        "known_train_parent_count": TRAIN_PARENT_COUNT,
        "known_validation_parent_count": VALIDATION_PARENT_COUNT,
        "validation_clutter_parent_support": VALIDATION_CLUTTER_PARENT_SUPPORT,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the V25d present-class-mean KnownGate experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
