"""Prepare the immutable V26k L2 train-only knownness diagnostic contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate_v26k import (
    CONTRACT,
    CONTRACT_SCHEMA,
    DOCUMENT,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    L2_EPOCH,
    MAX_CONTRACT_BYTES,
    MAX_READINESS_BYTES,
    OOD_SCORE_POLICY,
    READINESS,
    READINESS_SCHEMA,
    V26C_CONTRACT,
    V26C_READINESS,
    V26F_CONTRACT,
    V26F_PARENT_DETAIL_CSV,
    V26F_READINESS,
    V26F_REPORT,
    implementation_sha256,
    relative_path,
    require_bound_file,
    sha256_file,
    source_binding,
    validate_v26c_source,
    validate_v26f_source,
    write_immutable_json,
)


def _read_existing(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26k existing artifact must be a JSON object: {path}")
    return value


def _existing_contract_guard(
    root: Path,
    *,
    source_v26c: dict[str, Any],
    source_v26f: dict[str, Any],
) -> dict[str, Any] | None:
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    if not contract_path.exists() and not readiness_path.exists():
        return None
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileExistsError("V26k has an incomplete immutable preparation artifact")
    existing = _read_existing(contract_path)
    readiness = _read_existing(readiness_path)
    if (
        existing.get("schema_version") != CONTRACT_SCHEMA
        or dict(existing.get("source_v26c") or {}) != source_v26c
        or dict(existing.get("source_v26f") or {}) != source_v26f
        or dict(existing.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
        or dict(existing.get("implementation_sha256") or {}) != implementation_sha256(root)
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("contract_sha256") != sha256_file(contract_path)
    ):
        raise FileExistsError(
            "V26k existing contract/readiness is stale; refusing to overwrite an immutable experiment"
        )
    return {
        "status": "passed",
        "schema_version": CONTRACT_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_write_status": "reused_existing_source_equivalent",
        "readiness": str(readiness_path),
        "readiness_write_status": "reused_existing_source_equivalent",
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_v26c = validate_v26c_source(root)
    source_v26f = validate_v26f_source(root)
    document = require_bound_file(root, DOCUMENT, "V26k experiment document")

    reused = _existing_contract_guard(
        root,
        source_v26c=source_v26c,
        source_v26f=source_v26f,
    )
    if reused is not None:
        return reused

    source_paths = (
        ("v26c_contract", root / V26C_CONTRACT),
        ("v26c_readiness", root / V26C_READINESS),
        ("v26f_contract", root / V26F_CONTRACT),
        ("v26f_readiness", root / V26F_READINESS),
        ("v26f_report", root / V26F_REPORT),
        ("v26f_parent_detail", root / V26F_PARENT_DETAIL_CSV),
        ("document", document),
    )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26k L2 train-only strict-open-set knownness diagnostic",
        "scope": (
            "inference-only per-fold L2 DBB embedding reference fitting and frozen mixed audit; "
            "no model, cache, L3 registry or semantic action is changed"
        ),
        "direct_source_experiment": (
            "V26c for fold-local L2 Known-only checkpoints and train parents; "
            "V26f for frozen mixed CQ08 validation/test routes"
        ),
        "source_v26c": source_v26c,
        "source_v26f": source_v26f,
        "source_bindings": [
            {"name": name, **source_binding(root, path, description=f"V26k {name}")}
            for name, path in source_paths
        ],
        "implementation_sha256": implementation_sha256(root),
        "fold_count": int(FOLD_COUNT),
        "fixed_l2_epoch": int(L2_EPOCH),
        "data": {
            "source": "CQ08",
            "index": "V26c/V26f hash-bound RADAR-LN3-V16 CQ08 index",
            "reference_population": "per-fold V26c train-only Drone/Bird/Balloon parents",
            "primary_audit_population": (
                "V26f frozen parent-final rows whose observed L1 route is KNOWN_TARGET_EVIDENCE"
            ),
            "secondary_audit_population": "V26f frozen KNOWN_CLUTTER_EVIDENCE rows, reported separately",
            "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
            "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        },
        "representation": {
            "name": "classifier_layer_norm_liquid_final_state",
            "dimension": 48,
            "verified_path": "L2 liquid_final_state -> classifier[0] LayerNorm -> classifier[1] Linear",
            "classifier_input_is_used": True,
            "raw_liquid_final_state_used_directly": False,
            "angular_prototype_embedding": False,
            "logit_reconstruction_required": True,
        },
        "ood_score_policy": OOD_SCORE_POLICY,
        "ranking_protocol": {
            "unknown_truth_used_only_after_score_and_train_threshold_freeze": True,
            "primary_known_labels": ["DroneTarget", "BirdTarget", "BalloonTarget"],
            "primary_unknown_label": "UnknownTarget",
            "primary_route": "KNOWN_TARGET_EVIDENCE",
            "clutter_is_not_in_primary_l2_reference_or_primary_known_set": True,
            "train_threshold_is_diagnostic_only": True,
            "train_threshold_target_retention": 0.95,
            "threshold_fitted_from": "per-fold train-only DBB parent-final distances",
            "cross_fold_state_or_score_pooling": False,
            "msp_is_recomputed_comparator_only": True,
        },
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_reference_fit_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "unknown_checkpoint_selection_exposure": 0,
        "new_training": False,
        "new_threshold_fit": False,
        "product_threshold_mutation": False,
        "semantic_action_mutation": False,
        "deployment_eligible": False,
        "not_executed": {
            "training": True,
            "oof_retraining": True,
            "normalization_rebuild": True,
            "checkpoint_selection": True,
            "product_threshold_calibration": True,
            "l3_registry_mutation": True,
            "semantic_action_replay": True,
            "deployment": True,
        },
    }
    contract_status = write_immutable_json(
        root / CONTRACT,
        contract,
        max_bytes=MAX_CONTRACT_BYTES,
        description="V26k contract",
    )
    readiness = {
        "status": "passed",
        "schema_version": READINESS_SCHEMA,
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "checks": {
            "v26c_l2_known_only_fold_checkpoints_and_train_partitions_are_hash_bound": True,
            "v26c_l2_fixed_snapshot_is_epoch15": True,
            "v26f_mixed_val_test_population_and_observed_l1_routes_are_hash_bound": True,
            "embedding_is_the_actual_l2_classifier_input": True,
            "per_fold_class_conditional_reference_is_train_dbb_only": True,
            "unknown_is_excluded_from_model_normalization_reference_and_threshold_fit": True,
            "primary_unknown_audit_is_l1_target_route_only": True,
            "msp_is_comparator_only_and_no_product_threshold_is_reused": True,
            "no_cross_fold_pooling_or_semantic_action_mutation": True,
            "no_training_or_product_calibration_is_present": True,
        },
        "fold_count": int(FOLD_COUNT),
        "fixed_l2_epoch": int(L2_EPOCH),
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "unknown_reference_fit_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "training": "not_executed",
        "threshold_fitting": "train-only diagnostic operating point only; not product calibration",
        "deployment_eligible": False,
    }
    readiness_status = write_immutable_json(
        root / READINESS,
        readiness,
        max_bytes=MAX_READINESS_BYTES,
        description="V26k readiness",
    )
    return {
        "status": "passed",
        "schema_version": CONTRACT_SCHEMA,
        "contract": str(root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "contract_write_status": contract_status,
        "readiness": str(root / READINESS),
        "readiness_write_status": readiness_status,
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V26k L2 train-only knownness diagnostic.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
