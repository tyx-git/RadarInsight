"""Prepare the immutable V26i fixed-bimodal-Clutter OOD diagnostic contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate_v26i import (
    CLUTTER_COMPONENT_COUNT,
    CONTRACT,
    CONTRACT_SCHEMA,
    DOCUMENT,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    KNOWN_TARGET_RECALL_MINIMUM,
    MAX_CONTRACT_BYTES,
    MAX_READINESS_BYTES,
    MIN_CLUTTER_COMPONENT_PARENT_SUPPORT,
    OOD_SCORE_POLICY,
    READINESS,
    READINESS_SCHEMA,
    V26H_CONTRACT,
    V26H_DETAIL_CSV,
    V26H_READINESS,
    V26H_REPORT,
    implementation_sha256,
    relative_path,
    require_bound_file,
    sha256_file,
    source_binding,
    validate_v26h_source,
    write_immutable_json,
)


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract_path = root / CONTRACT
    readiness_path = root / READINESS
    source_v26h = validate_v26h_source(root)
    if contract_path.exists() or readiness_path.exists():
        if not (contract_path.is_file() and readiness_path.is_file()):
            raise FileExistsError("V26i has an incomplete existing immutable preparation artifact")
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
        if (
            existing.get("schema_version") != CONTRACT_SCHEMA
            or dict(existing.get("source_v26h") or {}) != source_v26h
            or dict(existing.get("ood_score_policy") or {}) != OOD_SCORE_POLICY
            or dict(existing.get("implementation_sha256") or {}) != implementation_sha256(root)
            or float(dict(existing.get("future_action_gate") or {}).get("known_target_recall_minimum", -1.0))
            != KNOWN_TARGET_RECALL_MINIMUM
            or readiness.get("schema_version") != READINESS_SCHEMA
            or readiness.get("status") != "passed"
            or readiness.get("contract_sha256") != sha256_file(contract_path)
        ):
            raise FileExistsError("V26i existing contract is stale; do not overwrite immutable experiment identity")
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

    document = require_bound_file(root, DOCUMENT, "V26i experiment document")
    source_paths = (
        ("v26h_contract", root / V26H_CONTRACT),
        ("v26h_readiness", root / V26H_READINESS),
        ("v26h_report", root / V26H_REPORT),
        ("v26h_parent_detail", root / V26H_DETAIL_CSV),
        ("document", document),
    )
    contract: dict[str, Any] = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26i frozen L1 train-only fixed-bimodal-Clutter Mahalanobis geometry diagnostic",
        "scope": (
            "read-only V26h-derived per-fold scoring after V26e L1 epoch-32 freeze; "
            "only train-Known Clutter reference geometry changes from one to fixed two components"
        ),
        "direct_source_experiment": "V26h frozen L1 train-only Mahalanobis OOD ranking diagnostic",
        "source_v26h": source_v26h,
        "source_bindings": [
            {"name": name, **source_binding(root, path, description=f"V26i {name}")}
            for name, path in source_paths
        ],
        "implementation_sha256": implementation_sha256(root),
        "fold_count": int(FOLD_COUNT),
        "data": {
            "source": "CQ08",
            "index": "V16 capture-content-provenance grouped index through V26h bindings",
            "known_reference_population": "V26e per-fold train KnownTarget/Clutter parents only",
            "mixed_evaluation_population": "V26f frozen mixed validation/test parents only after checkpoint freeze",
            "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
            "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        },
        "ood_score_policy": OOD_SCORE_POLICY,
        "causal_variable": {
            "control": "V26h one-centre KnownClutter diagonal reference",
            "treatment": "V26i fixed K=2 deterministic train-only KnownClutter centres with one pooled diagonal variance",
            "unchanged": [
                "V26e L1 epoch-32 checkpoints",
                "LayerNorm classifier readout representation",
                "KnownTarget single centre",
                "CQ08 data/index/splits/parent ordering",
                "normalization/features/model/loss/dt",
                "V26f mixed evaluation parents",
                "no threshold/no semantic action/no L3 mutation",
            ],
        },
        "ranking_protocol": {
            "fit_one_reference_per_fold": True,
            "readout_state_is_layer_norm_applied_to_liquid_final_state": True,
            "train_reference_uses_parent_final_causal_state_only": True,
            "mixed_evaluation_uses_parent_final_causal_state_only": True,
            "known_vs_unknown_metrics_are_threshold_free": True,
            "unknown_truth_used_only_for_final_ranking_report": True,
            "fixed_component_count": int(CLUTTER_COMPONENT_COUNT),
            "minimum_component_parent_support": int(MIN_CLUTTER_COMPONENT_PARENT_SUPPORT),
            "component_count_search": "not_executed",
            "cross_fold_state_or_score_pooling": False,
            "semantic_action_changed": False,
        },
        "future_action_gate": {
            "known_target_recall_minimum": KNOWN_TARGET_RECALL_MINIMUM,
            "definition": (
                "future parent-final mixed semantic KnownTarget Recall after a separately "
                "pre-registered threshold/action experiment; not evaluable in V26i because V26i emits no action"
            ),
            "status_in_v26i": "not_evaluable_no_threshold_or_semantic_action_replay",
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
        contract_path, contract, max_bytes=MAX_CONTRACT_BYTES, description="contract"
    )
    readiness = {
        "status": "passed",
        "schema_version": READINESS_SCHEMA,
        "contract_path": relative_path(root, contract_path),
        "contract_sha256": sha256_file(contract_path),
        "checks": {
            "v26h_completed_source_chain_and_five_fixed_l1_checkpoints_are_hash_bound": True,
            "v26h_train_and_mixed_parent_population_bindings_are_reused_without_change": True,
            "representation_is_the_actual_l1_classifier_layer_norm_input_space": True,
            "only_clutter_reference_geometry_changes_from_one_to_fixed_two_centres": True,
            "component_count_is_fixed_before_scoring_and_not_selected_from_unknown_or_held_out_labels": True,
            "each_component_has_a_pre_registered_minimum_train_parent_support_or_fold_is_infeasible": True,
            "unknown_is_excluded_from_model_normalization_reference_and_threshold_fits": True,
            "unknown_truth_is_report_only_after_score_freeze": True,
            "known_target_recall_minimum_is_recorded_for_future_action_experiment_not_claimed_here": True,
            "no_cross_fold_state_pooling_action_mutation_training_or_threshold_fitting_is_present": True,
        },
        "fold_count": int(FOLD_COUNT),
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "fixed_clutter_component_count": int(CLUTTER_COMPONENT_COUNT),
        "minimum_component_parent_support": int(MIN_CLUTTER_COMPONENT_PARENT_SUPPORT),
        "known_target_recall_minimum_for_future_action_gate": KNOWN_TARGET_RECALL_MINIMUM,
        "known_target_recall_status_in_v26i": "not_evaluable_no_action_replay",
        "unknown_reference_fit_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }
    readiness_status = write_immutable_json(
        readiness_path, readiness, max_bytes=MAX_READINESS_BYTES, description="readiness"
    )
    return {
        "status": "passed",
        "schema_version": CONTRACT_SCHEMA,
        "contract": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "contract_write_status": contract_status,
        "readiness": str(readiness_path),
        "readiness_write_status": readiness_status,
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare V26i fixed-bimodal-Clutter Mahalanobis diagnostic."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
