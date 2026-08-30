"""Prepare V26j's frozen mixed Known/Unknown L1->L2->L3 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.known_gate_v26j_audit import (
    CONTRACT, CONTRACT_SCHEMA, DOCUMENT, EXPECTED_PARENT_COUNTS, FOLD_COUNT,
    INDEX, L1_TEST_CACHE, L1_VALIDATION_CACHE, L2_TEST_MICRO_CACHE,
    L2_TEST_PHYSICAL_CACHE, L2_VALIDATION_MICRO_CACHE, L2_VALIDATION_PHYSICAL_CACHE,
    READINESS, READINESS_SCHEMA, V26B_CACHE_MANIFEST, V26C_CONTRACT,
    V26C_READINESS, V26J_CONTRACT, V26J_READINESS, V26J_REPORT,
    expected_parent_counts, fold_paths, implementation_sha256, read_json,
    relative_path, require_bound_file, selected_event_sample_ids, sha256_file,
    source_binding, validate_feature_cache, write_immutable_json,
)
from radar_ln3.known_gate_v26e_oof import validate_frozen_v26c_sources


def _source_json(root: Path, path: Path, description: str) -> dict[str, Any]:
    return read_json(require_bound_file(root, path, description), description=description)


def _validate_source_pair(root: Path) -> dict[str, Any]:
    j_contract = _source_json(root, root / V26J_CONTRACT, "V26j L1 contract")
    j_readiness = _source_json(root, root / V26J_READINESS, "V26j L1 readiness")
    j_report = _source_json(root, root / V26J_REPORT, "V26j L3 report")
    c_contract = _source_json(root, root / V26C_CONTRACT, "V26c contract")
    c_readiness = _source_json(root, root / V26C_READINESS, "V26c readiness")
    if (
        j_contract.get("schema_version") != "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-contract-v1"
        or j_readiness.get("schema_version") != "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-calibration-readiness-v1"
        or j_readiness.get("status") != "passed"
        or j_readiness.get("contract_sha256") != sha256_file(root / V26J_CONTRACT)
        or j_contract.get("single_variable", {}).get("treatment") != "causal_gru"
        or j_report.get("status") != "completed_v26j_rd_causal_gru_epoch32_known_only_l3_diagnostic"
        or j_report.get("mixed_unknown_audit") != "not_executed"
        or int(j_contract.get("unknown_training_exposure", -1)) != 0
        or int(j_contract.get("unknown_normalization_exposure", -1)) != 0
        or c_contract.get("schema_version") != "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-contract-v1"
        or c_readiness.get("status") != "passed"
    ):
        raise ValueError("V26j frozen L1/L2 sources are not sealed diagnostics")
    validate_frozen_v26c_sources(root, j_contract)
    return {"v26j_contract": j_contract, "v26j_readiness": j_readiness, "v26j_report": j_report, "v26c_contract": c_contract, "v26c_readiness": c_readiness}


def _validate_fold_sources(root: Path, source: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    paths = fold_paths(fold_id)
    j_fold = next(row for row in source["v26j_contract"]["folds"] if int(row["fold_id"]) == fold_id)
    c_fold = next(row for row in source["v26c_contract"]["folds"] if int(row["fold_id"]) == fold_id)
    if (
        j_fold.get("l1", {}).get("rd_frame_aggregation") != "causal_gru"
        or j_fold.get("l2", {}).get("reused_from_v26c") is not True
        or sorted(str(value) for value in j_fold.get("train_parent_ids", ())) != sorted(str(value) for value in c_fold.get("train_parent_ids", ()))
        or sorted(str(value) for value in j_fold.get("held_out_parent_ids", ())) != sorted(str(value) for value in c_fold.get("held_out_parent_ids", ()))
    ):
        raise ValueError(f"V26j fold {fold_id} source isolation differs")
    l1 = dict(j_fold.get("l1") or {})
    if l1.get("config_path") != relative_path(root, root / paths["l1_config"]):
        raise ValueError(f"V26j fold {fold_id} L1 config path differs")
    for name, path in paths.items():
        require_bound_file(root, root / path, f"V26j fold {fold_id} {name}")
    prediction = _source_json(root, root / paths["l1_prediction"], f"V26j fold {fold_id} L1 prediction")
    summary = _source_json(root, root / paths["l1_summary"], f"V26j fold {fold_id} L1 summary")
    runtime = dict(prediction.get("rd_runtime") or {})
    if (
        prediction.get("schema_version") != "RADAR-LN3-V26j-l1-rd-causal-gru-epoch32-known-only-oof-prediction-v1"
        or int(prediction.get("fold_id", -1)) != fold_id
        or int(prediction.get("fixed_snapshot_epoch", -1)) != 32
        or prediction.get("checkpoint_sha256") != sha256_file(root / paths["l1_checkpoint"])
        or runtime.get("rd_frame_aggregation_config") != "causal_gru"
        or runtime.get("rd_frame_aggregation_runtime") != "causal_gru"
        or runtime.get("rd_temporal_module") != "GRUCell"
        or int(runtime.get("rd_temporal_parameter_count", 0)) <= 0
        or not list(runtime.get("rd_temporal_state_keys") or ())
        or summary.get("status") != "completed"
        or dict(summary.get("rd_runtime") or {}) != runtime
    ):
        raise ValueError(f"V26j fold {fold_id} causal-GRU L1 binding differs")
    return {"fold_id": fold_id, "source_v26j_fold_contract": {"train_parent_ids_sha256": j_fold.get("train_parent_ids_sha256"), "held_out_parent_ids_sha256": j_fold.get("held_out_parent_ids_sha256"), "component_overlap_count": j_fold.get("component_overlap_count")}, "rd_runtime": runtime, "source_files": [{"name": name, **source_binding(root, root / path, description=f"V26j fold {fold_id} {name}")} for name, path in paths.items()]}


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve(); source = _validate_source_pair(root)
    index = require_bound_file(root, root / INDEX, "CQ08 index")
    manifest_path = require_bound_file(root, root / V26B_CACHE_MANIFEST, "V26b mixed cache manifest")
    manifest = read_json(manifest_path, description="V26b mixed cache manifest")
    if manifest.get("schema_version") != "RADAR-LN3-V26b-l2-mixed-open-set-feature-cache-v1": raise ValueError("V26j V26b mixed cache manifest differs")
    document = require_bound_file(root, root / DOCUMENT, "V26j experiment document")
    cache_bindings = []
    for split, l1_path, physical_path, micro_path in (("val", L1_VALIDATION_CACHE, L2_VALIDATION_PHYSICAL_CACHE, L2_VALIDATION_MICRO_CACHE), ("test", L1_TEST_CACHE, L2_TEST_PHYSICAL_CACHE, L2_TEST_MICRO_CACHE)):
        ids = selected_event_sample_ids(root, split=split)
        l1 = validate_feature_cache(root / l1_path, expected_sample_ids=ids, width=1, valid_rank=2, description=f"{split} L1 cache")
        physical = validate_feature_cache(root / physical_path, expected_sample_ids=ids, width=4, valid_rank=1, description=f"{split} L2 physical cache")
        micro = validate_feature_cache(root / micro_path, expected_sample_ids=ids, width=8, valid_rank=1, description=f"{split} L2 micro cache")
        for value, path in ((l1, l1_path), (physical, physical_path), (micro, micro_path)): value["path"] = relative_path(root, root / path)
        cache_bindings.append({"split": split, "parent_counts": expected_parent_counts(root, split=split), "l1_course": l1, "l2_rd_physical4": physical, "l2_micro_doppler8": micro})
    folds = [_validate_fold_sources(root, source, fold_id) for fold_id in range(FOLD_COUNT)]
    contract = {"schema_version": CONTRACT_SCHEMA, "experiment": "V26j frozen mixed Known/Unknown causal-GRU L1 -> V26c L2 -> semantic audit", "scope": "inference-only mixed validation/test audit after V26j L1 and V26c L2 freeze", "source_bindings": [{"name": name, **source_binding(root, path, description=f"V26j {name}")} for name, path in (("index", index), ("v26b_cache_manifest", manifest_path), ("v26j_contract", root / V26J_CONTRACT), ("v26j_readiness", root / V26J_READINESS), ("v26j_report", root / V26J_REPORT), ("v26c_contract", root / V26C_CONTRACT), ("v26c_readiness", root / V26C_READINESS), ("document", document))], "implementation_sha256": implementation_sha256(root), "fold_count": FOLD_COUNT, "unknown_training_exposure": 0, "unknown_normalization_exposure": 0, "unknown_checkpoint_selection_exposure": 0, "unknown_threshold_fit_exposure": 0, "unknown_oof_prediction_count": 0, "validation_access_count": 0, "test_access_count": 0, "data": {"source_index": relative_path(root, index), "source_index_sha256": sha256_file(index), "mixed_labels": ["DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"], "split_parent_counts": EXPECTED_PARENT_COUNTS, "cache_bindings": cache_bindings}, "folds": folds, "frozen_model_bindings": {"l1": "V26j causal-GRU epoch-32 fold checkpoint/config/normalization", "l2": "V26c epoch-15 fold checkpoint/config/normalization read-only", "thresholds": "V26j Known-only outer-cross-fitted L3 thresholds read-only"}, "audit_protocol": {"validation_and_test_are_mixed_known_unknown": True, "unknown_training_exposure": 0, "unknown_normalization_exposure": 0, "unknown_checkpoint_selection_exposure": 0, "unknown_threshold_fit_exposure": 0, "pending_is_not_unknown_rejection_success": True, "true_label_never_controls_l1_or_l2_route": True, "minimum_evidence_events": 3}, "not_executed": {"training": True, "threshold_fitting": True, "oof_retraining": True, "deployment_registry": True}, "deployment_eligible": False}
    write_immutable_json(root / CONTRACT, contract, max_bytes=2*1024*1024, description="contract")
    readiness = {"status": "passed", "schema_version": READINESS_SCHEMA, "contract_path": relative_path(root, root / CONTRACT), "contract_sha256": sha256_file(root / CONTRACT), "checks": {"frozen_v26j_causal_gru_source_is_sealed": True, "frozen_v26c_l2_source_is_sealed": True, "all_mixed_caches_are_hash_and_order_valid": True, "all_five_causal_gru_fold_sources_are_present": True, "unknown_excluded_from_training_normalization_checkpoint_and_threshold_fit": True, "true_labels_do_not_control_routing": True, "pending_is_not_counted_as_unknown_rejection": True, "folds_not_pooled_for_deployment": True}, "fold_count": FOLD_COUNT, "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"], "test_parent_counts": EXPECTED_PARENT_COUNTS["test"], "unknown_training_exposure": 0, "unknown_normalization_exposure": 0, "unknown_checkpoint_selection_exposure": 0, "unknown_threshold_fit_exposure": 0, "unknown_oof_prediction_count": 0, "validation_access_count": 0, "test_access_count": 0, "threshold_fitting": "not_executed", "deployment_eligible": False}
    write_immutable_json(root / READINESS, readiness, max_bytes=512*1024, description="readiness")
    return {"status": "passed", "contract": str(root / CONTRACT), "readiness": str(root / READINESS), "training": "not_executed", "deployment_eligible": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V26j frozen mixed audit."); parser.add_argument("--project-root", type=Path, required=True); args = parser.parse_args(); print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
