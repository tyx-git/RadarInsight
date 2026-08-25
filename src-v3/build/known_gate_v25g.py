"""Prepare the V25g frozen causal decision-policy diagnostic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from build.known_gate_v25f import (
    EXPECTED_EPOCHS,
    MAX_V25F_CHECKPOINT_BYTES,
    MAX_V25F_CHECKPOINT_MEMBERS,
    MAX_V25F_CHECKPOINT_UNCOMPRESSED_BYTES,
    MAX_V25F_JSON_BYTES,
    _source,
    validate_v25f_checkpoint_file,
)
from radar_ln3.known_gate import (
    build_known_gate_model_for_config,
    known_gate_config_identity_sha256,
    sha256_file,
)
from radar_ln3.known_gate_v25g_policy import (
    V25G_MAX_THRESHOLD_CANDIDATES,
    V25G_MINIMUM_EVIDENCE_EVENTS,
    V25G_MINIMUM_THRESHOLD,
    V25G_POLICY_SCHEMA,
)


EXPERIMENT_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v25g_causal_decision_policy.md"
)
EXPERIMENT_CONTRACT = Path(
    "data/processed/RADAR-LN3-V25g-cq08-known-gate-causal-decision-policy-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V25g-known-gate-causal-decision-policy-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/RADAR-LN3-V25g-known-gate-causal-decision-policy.json"
)
V25G_SCHEMA = "RADAR-LN3-V25g-known-gate-causal-decision-policy-v1"
CONTRACT_SCHEMA = (
    "RADAR-LN3-V25g-known-gate-causal-decision-policy-experiment-contract-v1"
)
EXPECTED_V25A_VARIANT = "v25a_course_preclip_gradient_telemetry"
EXPECTED_V25A_BEST_EPOCH = 14
EXPECTED_TRAIN_PARENTS = 986
EXPECTED_VALIDATION_PARENTS = 232
EXPECTED_VALIDATION_CLUTTER = 9


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.stat().st_size > MAX_V25G_JSON_BYTES:
            raise ValueError(f"existing V25g artifact exceeds size limit: {path}")
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite a different V25g artifact: {path}")
        return
    if len(content.encode("utf-8")) > MAX_V25G_JSON_BYTES:
        raise ValueError(f"new V25g artifact exceeds size limit: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_source(root: Path) -> dict[str, Any]:
    source = _source(root)
    summary = source["summary"]
    config = source["config"]
    checkpoint = source["epoch_paths"][EXPECTED_V25A_BEST_EPOCH]
    validate_v25f_checkpoint_file(checkpoint)
    model, _, pairing = build_known_gate_model_for_config(root, config)
    del model
    if (
        config.variant != EXPECTED_V25A_VARIANT
        or int(summary.get("best_epoch", -1)) != EXPECTED_V25A_BEST_EPOCH
        or int(summary.get("train_parent_count", -1)) != EXPECTED_TRAIN_PARENTS
        or int(summary.get("validation_parent_count", -1)) != EXPECTED_VALIDATION_PARENTS
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("unknown_is_model_class") is not False
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise ValueError("V25g V25a frozen source is stale or outside the sealed contract")
    return source


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source = _validate_source(root)
    paths = source["paths"]
    hashes = source["hashes"]
    checkpoint = source["epoch_paths"][EXPECTED_V25A_BEST_EPOCH]

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "policy_schema": V25G_POLICY_SCHEMA,
        "experiment_name": "V25g frozen causal ACCEPT/PENDING decision policy",
        "scope": "KnownGate development-only action-layer diagnostic",
        "single_factor": {
            "changed_component": "prefix local argmax action interpretation only",
            "training_step_count": 0,
            "optimizer_step_count": 0,
            "frozen": [
                "V25a epoch14 model weights",
                "V25a features and normalization",
                "V25a parent order, loss and prefix auxiliary semantics",
                "V25a liquid dt and bounded_observation dynamics",
            ],
        },
        "policy": {
            "minimum_evidence_events": V25G_MINIMUM_EVIDENCE_EVENTS,
            "events_before_minimum": "PENDING_ABSTAIN",
            "confidence_rule": "accept local argmax only when max softmax probability >= threshold",
            "pending_is_unknown": False,
            "patience": 1,
            "revocation": False,
            "class_specific_thresholds": False,
            "threshold_candidates": "0.5 plus deterministic quantile representatives of train-only eligible-prefix confidence values, capped at 256",
            "selection": "maximize parent coverage subject to train-only clutter precision and class-recall floors",
        },
        "calibration": {
            "protocol": "train_only_frozen_checkpoint_calibration",
            "is_oof": False,
            "oof_status": "not_executed",
            "reason": "the repository has no exact V25a parent-grouped OOF prediction artifact",
            "validation_or_test_used_for_threshold_fit": False,
            "formal_adoption_requires_exact_oof": True,
        },
        "source_v25a": {
            "config": _relative(root, paths["config"]),
            "config_sha256": hashes["config"],
            "config_identity_sha256": known_gate_config_identity_sha256(source["config"]),
            "readiness": _relative(root, paths["readiness"]),
            "readiness_sha256": hashes["readiness"],
            "training_summary": _relative(root, paths["summary"]),
            "training_summary_sha256": hashes["summary"],
            "checkpoint": _relative(root, checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": EXPECTED_V25A_BEST_EPOCH,
            "feature_manifest": _relative(root, paths["feature_manifest"]),
            "feature_manifest_sha256": hashes["feature_manifest"],
            "feature_contract": _relative(root, paths["feature_contract"]),
            "feature_contract_sha256": hashes["feature_contract"],
        },
        "evaluation": {
            "calibration_split": "train",
            "evaluation_split": "val",
            "validation_parent_count": EXPECTED_VALIDATION_PARENTS,
            "validation_clutter_parent_support": EXPECTED_VALIDATION_CLUTTER,
            "required": [
                "parent-level first causal ACCEPT/PENDING action metrics",
                "parent-balanced eligible-prefix action metrics",
                "Pending coverage and first acceptance event",
                "Clutter parent bootstrap and per-parent trajectories",
                "source/capture and observation-strata metrics",
            ],
        },
        "resource_safety": {
            "json_max_bytes": MAX_V25G_JSON_BYTES,
            "checkpoint_max_bytes": MAX_V25F_CHECKPOINT_BYTES,
            "checkpoint_max_declared_uncompressed_bytes": MAX_V25F_CHECKPOINT_UNCOMPRESSED_BYTES,
            "checkpoint_max_zip_members": MAX_V25F_CHECKPOINT_MEMBERS,
            "checkpoint_validation_before_torch_load": True,
            "new_feature_cache_created": False,
            "checkpoint_mutation": False,
            "max_threshold_candidates": V25G_MAX_THRESHOLD_CANDIDATES,
            "sequential_split_loading": True,
            "max_train_calibration_records": 200000,
            "max_validation_records": 50000,
        },
        "training": "not_executed_by_contract",
        "Unknown": "not_used",
        "OOF": "not_executed",
        "test": "not_executed",
        "full_cascade": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / EXPERIMENT_CONTRACT, contract)
    contract_hash = sha256_file(root / EXPERIMENT_CONTRACT)
    readiness = {
        "status": "passed",
        "schema_version": V25G_SCHEMA,
        "policy_schema": V25G_POLICY_SCHEMA,
        "experiment_contract": {
            "path": _relative(root, root / EXPERIMENT_CONTRACT),
            "sha256": contract_hash,
        },
        "source_v25a": {
            "config_path": _relative(root, paths["config"]),
            "config_sha256": hashes["config"],
            "summary_path": _relative(root, paths["summary"]),
            "summary_sha256": hashes["summary"],
            "checkpoint_path": _relative(root, checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "checkpoint_epoch": EXPECTED_V25A_BEST_EPOCH,
        },
        "checks": {
            "v25a_source_is_completed_seed43_epoch14": True,
            "v25a_has_sealed_parent_orders_and_checkpoint": True,
            "v25a_test_not_executed": True,
            "v25a_unknown_training_exposure_is_zero": True,
            "v25g_has_zero_training_and_optimizer_steps": True,
            "v25g_does_not_create_or_mutate_feature_cache": True,
            "pending_is_separate_from_unknown": True,
            "prefix_minimum_events_is_frozen_at_three": True,
            "validation_is_not_used_for_threshold_fit": True,
            "exact_oof_is_not_claimed": True,
        },
        "training": "not_executed",
        "calibration": "train_only_frozen_checkpoint_in_sample",
        "validation_used_for_threshold_fit": False,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "OOF": "not_executed",
        "test": "not_executed",
        "full_cascade": "not_executed",
        "deployment_eligible": False,
    }
    _write_new_or_identical(root / READINESS, readiness)
    return {
        "status": "passed",
        "schema_version": V25G_SCHEMA,
        "experiment_contract": str(root / EXPERIMENT_CONTRACT),
        "readiness": str(root / READINESS),
        "source_checkpoint": str(checkpoint),
        "checkpoint_epoch": EXPECTED_V25A_BEST_EPOCH,
        "training": "not_executed",
        "calibration": "train_only_frozen_checkpoint_in_sample",
        "OOF": "not_executed",
        "test": "not_executed",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prepare V25g causal decision policy.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


MAX_V25G_JSON_BYTES = 64 * 1024 * 1024


if __name__ == "__main__":
    main()
