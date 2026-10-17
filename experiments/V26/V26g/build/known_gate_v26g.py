"""Prepare the immutable V26g quality-aware Pending/Abstain audit contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate_v26g import (
    CONTRACT,
    CONTRACT_SCHEMA,
    DOCUMENT,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    MAX_CONTRACT_BYTES,
    MAX_READINESS_BYTES,
    QUALITY_POLICY,
    READINESS,
    READINESS_SCHEMA,
    V26F_PARENT_DETAIL_CSV,
    V26F_CONTRACT,
    V26F_READINESS,
    V26F_REPORT,
    implementation_sha256,
    relative_path,
    require_bound_file,
    sha256_file,
    source_binding,
    validate_v26f_source_report,
    validate_quality_cache_relationship,
    write_immutable_json,
)


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source = validate_v26f_source_report(root)
    quality_cache_relationship = validate_quality_cache_relationship(root)
    document = require_bound_file(root, DOCUMENT, "V26g experiment document")
    source_paths = (
        ("v26f_contract", root / V26F_CONTRACT),
        ("v26f_report", root / V26F_REPORT),
        ("v26f_readiness", root / V26F_READINESS),
        ("v26f_parent_detail", root / V26F_PARENT_DETAIL_CSV),
        ("document", document),
    )
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26g frozen quality-aware Pending/Abstain audit",
        "scope": (
            "read-only quality-availability action replay over sealed V26f mixed "
            "CQ08 inference; no model, score, cache, normalization or threshold mutation"
        ),
        "direct_source_experiment": "V26f",
        "source_v26f": source,
        "quality_cache_relationship": quality_cache_relationship,
        "source_bindings": [
            {"name": name, **source_binding(root, path, description=f"V26g {name}")}
            for name, path in source_paths
        ],
        "implementation_sha256": implementation_sha256(root),
        "fold_count": int(FOLD_COUNT),
        "data": {
            "source": "CQ08",
            "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
            "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
            "source_replay": "V26f frozen L1/L2/L3 mixed validation/test replay",
        },
        "quality_policy": QUALITY_POLICY,
        "replay_requirements": {
            "v26f_parent_final_action_must_match_source_detail": True,
            "v26f_frozen_checkpoint_and_threshold_bindings_must_match_source": True,
            "l1_course_validity_read_from_frozen_cache": True,
            "l2_validity_read_from_frozen_cache": True,
            "l2_rd_and_micro_validity_masks_must_match": True,
            "truth_labels_never_control_action": True,
            "pending_is_not_unknown_rejection_success": True,
            "folds_are_not_pooled_into_deployment_model": True,
        },
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_checkpoint_selection_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "new_training": False,
        "new_threshold_fit": False,
        "deployment_eligible": False,
        "not_executed": {
            "training": True,
            "oof_retraining": True,
            "normalization": True,
            "threshold_fitting": True,
            "feature_cache_rebuild": True,
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
            "v26f_source_contract_readiness_report_and_detail_are_hash_bound": True,
            "v26f_transitive_l1_l2_checkpoint_cache_and_threshold_contract_is_valid": True,
            "quality_policy_is_availability_only_and_label_free": True,
            "l1_course_and_l2_rd_validity_masks_are_existing_frozen_artifacts": True,
            "rd_physical_and_micro_doppler_validity_masks_are_required_to_match": True,
            "dt_valid_rd_present_and_continuous_scores_are_not_new_quality_gates": True,
            "pending_is_not_counted_as_unknown_rejection": True,
            "five_folds_are_not_pooled_for_deployment": True,
            "no_training_or_threshold_fitting_is_present": True,
        },
        "fold_count": int(FOLD_COUNT),
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "unknown_training_exposure": 0,
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
    parser = argparse.ArgumentParser(description="Prepare the V26g quality-aware Pending audit.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
