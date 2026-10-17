"""Prepare the immutable V26h train-only Mahalanobis OOD diagnostic contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate_v26h import (
    CONTRACT,
    CONTRACT_SCHEMA,
    DOCUMENT,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    MAX_CONTRACT_BYTES,
    MAX_READINESS_BYTES,
    OOD_SCORE_POLICY,
    READINESS,
    READINESS_SCHEMA,
    V26F_CONTRACT,
    V26F_PARENT_DETAIL_CSV,
    V26F_READINESS,
    V26F_REPORT,
    V26E_CONTRACT,
    V26E_READINESS,
    implementation_sha256,
    relative_path,
    require_bound_file,
    sha256_file,
    source_binding,
    validate_v26e_source,
    validate_v26f_source,
    write_immutable_json,
)


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    existing_contract_path = root / CONTRACT
    existing_readiness_path = root / READINESS
    if existing_contract_path.exists() or existing_readiness_path.exists():
        # V26h's source protocol is immutable.  Execution-only memory
        # refinements are recorded in the eventual report fingerprint and may
        # reuse this source-equivalent contract; a source or score-policy
        # difference still fails closed and requires a new experiment ID.
        if not (existing_contract_path.is_file() and existing_readiness_path.is_file()):
            raise FileExistsError("V26h has an incomplete existing immutable preparation artifact")
        existing = json.loads(existing_contract_path.read_text(encoding="utf-8"))
        source_v26e = validate_v26e_source(root)
        source_v26f = validate_v26f_source(root)
        if (
            existing.get("schema_version") != CONTRACT_SCHEMA
            or dict(existing.get("source_v26e") or {}) != source_v26e
            or dict(existing.get("source_v26f") or {}) != source_v26f
            or dict(existing.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
            or dict(existing.get("implementation_sha256") or {}) != implementation_sha256(root)
        ):
            raise FileExistsError("V26h existing contract is stale; do not overwrite an immutable experiment identity")
        readiness = json.loads(existing_readiness_path.read_text(encoding="utf-8"))
        if (
            readiness.get("schema_version") != READINESS_SCHEMA
            or readiness.get("status") != "passed"
            or readiness.get("contract_sha256") != sha256_file(existing_contract_path)
        ):
            raise FileExistsError("V26h existing readiness is stale; do not overwrite it")
        return {
            "status": "passed",
            "schema_version": CONTRACT_SCHEMA,
            "contract": str(existing_contract_path),
            "contract_sha256": sha256_file(existing_contract_path),
            "contract_write_status": "reused_existing_source_equivalent",
            "readiness": str(existing_readiness_path),
            "readiness_write_status": "reused_existing_source_equivalent",
            "training": "not_executed",
            "threshold_fitting": "not_executed",
            "deployment_eligible": False,
        }
    source_v26e = validate_v26e_source(root)
    source_v26f = validate_v26f_source(root)
    document = require_bound_file(root, DOCUMENT, "V26h experiment document")
    source_paths = (
        ("v26e_contract", root / V26E_CONTRACT),
        ("v26e_readiness", root / V26E_READINESS),
        ("v26f_contract", root / V26F_CONTRACT),
        ("v26f_readiness", root / V26F_READINESS),
        ("v26f_report", root / V26F_REPORT),
        ("v26f_parent_detail", root / V26F_PARENT_DETAIL_CSV),
        ("document", document),
    )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26h frozen L1 train-only Mahalanobis OOD ranking diagnostic",
        "scope": (
            "read-only per-fold representation scoring after V26e L1 epoch-32 freeze; "
            "train Known parents fit references, mixed CQ08 val/test only evaluates frozen rankings"
        ),
        "direct_source_experiment": "V26e for L1 checkpoints; V26f for frozen mixed evaluation population",
        "source_v26e": source_v26e,
        "source_v26f": source_v26f,
        "source_bindings": [
            {"name": name, **source_binding(root, path, description=f"V26h {name}")}
            for name, path in source_paths
        ],
        "implementation_sha256": implementation_sha256(root),
        "fold_count": int(FOLD_COUNT),
        "data": {
            "source": "CQ08",
            "index": "V16 capture-content-provenance grouped index via V26e/V26f bindings",
            "known_reference_population": "V26e per-fold train KnownTarget/Clutter parents only",
            "mixed_evaluation_population": "V26f frozen mixed validation/test parents only after checkpoint freeze",
            "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
            "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        },
        "ood_score_policy": OOD_SCORE_POLICY,
        "ranking_protocol": {
            "fit_one_reference_per_fold": True,
            "readout_state_is_layer_norm_applied_to_liquid_final_state": True,
            "train_reference_uses_parent_final_causal_state_only": True,
            "mixed_evaluation_uses_parent_final_causal_state_only": True,
            "known_vs_unknown_metrics_are_threshold_free": True,
            "unknown_truth_used_only_for_final_ranking_report": True,
            "msp_comparator_source": "V26a frozen max-known-confidence diagnostic; no threshold reused",
            "cross_fold_state_or_score_pooling": False,
            "semantic_action_changed": False,
        },
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_reference_fit_exposure": 0,
        "unknown_checkpoint_selection_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "new_training": False,
        "new_threshold_fit": False,
        "semantic_action_mutation": False,
        "deployment_eligible": False,
        "not_executed": {
            "training": True,
            "oof_retraining": True,
            "normalization": True,
            "checkpoint_selection": True,
            "threshold_fitting": True,
            "l3_registry_mutation": True,
            "feature_cache_rebuild": True,
            "semantic_action_replay": True,
            "deployment_registry": True,
        },
    }
    contract_status = write_immutable_json(
        root / CONTRACT,
        contract,
        max_bytes=MAX_CONTRACT_BYTES,
        description="contract",
    )
    readiness = {
        "status": "passed",
        "schema_version": READINESS_SCHEMA,
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "checks": {
            "v26e_five_fixed_epoch32_l1_checkpoints_and_train_caches_are_hash_bound": True,
            "v26e_train_and_held_out_parent_sets_are_disjoint_per_fold": True,
            "v26f_frozen_mixed_val_test_caches_and_report_are_hash_bound": True,
            "representation_is_the_actual_l1_classifier_layer_norm_input_space": True,
            "reference_fit_is_train_known_only_per_fold": True,
            "unknown_is_excluded_from_model_normalization_reference_and_threshold_fits": True,
            "unknown_truth_is_report_only_after_score_freeze": True,
            "msp_threshold_oracle_is_not_reused": True,
            "no_cross_fold_state_pooling_or_action_mutation_is_present": True,
            "no_training_or_threshold_fitting_is_present": True,
        },
        "fold_count": int(FOLD_COUNT),
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "unknown_reference_fit_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }
    readiness_status = write_immutable_json(
        root / READINESS,
        readiness,
        max_bytes=MAX_READINESS_BYTES,
        description="readiness",
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
    parser = argparse.ArgumentParser(description="Prepare V26h train-only Mahalanobis OOD diagnostic.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
