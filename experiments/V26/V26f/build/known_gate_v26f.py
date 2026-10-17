"""Prepare the immutable V26f mixed Known/Unknown audit contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.known_gate_v26f import CONTRACT, CONTRACT_SCHEMA, DOCUMENT, EXPECTED_PARENT_COUNTS, FOLD_COUNT, INDEX, L1_TEST_CACHE, L1_VALIDATION_CACHE, L2_TEST_MICRO_CACHE, L2_TEST_PHYSICAL_CACHE, L2_VALIDATION_MICRO_CACHE, L2_VALIDATION_PHYSICAL_CACHE, READINESS, READINESS_SCHEMA, V26B_CACHE_MANIFEST, V26C_CONTRACT, V26C_READINESS, V26E_CONTRACT, V26E_READINESS, V26E_REPORT, expected_parent_counts, fold_paths, implementation_sha256, read_json, relative_path, require_bound_file, selected_event_sample_ids, sha256_file, source_binding, source_constants, validate_feature_cache, write_immutable_json
from radar_ln3.known_gate_v26e_oof import validate_frozen_v26c_sources


def _source_json(root: Path, path: Path, description: str) -> dict[str, Any]:
    checked = require_bound_file(root, path, description)
    return read_json(checked, description=description)


def _validate_source_pair(root: Path) -> dict[str, Any]:
    v26e_contract = _source_json(root, root / V26E_CONTRACT, "V26e contract")
    v26e_readiness = _source_json(root, root / V26E_READINESS, "V26e readiness")
    v26e_report = _source_json(root, root / V26E_REPORT, "V26e report")
    v26c_contract = _source_json(root, root / V26C_CONTRACT, "V26c contract")
    v26c_readiness = _source_json(root, root / V26C_READINESS, "V26c readiness")
    if (
        v26e_contract.get("schema_version") != "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-contract-v1"
        or v26e_readiness.get("schema_version") != "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-readiness-v1"
        or v26e_readiness.get("status") != "passed"
        or v26e_contract.get("deployment_eligible") is not False
        or int(v26e_contract.get("fold_count", -1)) != FOLD_COUNT
        or int(v26e_contract.get("known_train_parent_count", -1)) != 986
        or int(v26e_contract.get("unknown_training_exposure", -1)) != 0
        or int(v26e_contract.get("unknown_normalization_exposure", -1)) != 0
        or v26e_report.get("status") != "completed_v26e_fixed_epoch32_known_only_l3_diagnostic"
        or v26e_report.get("mixed_unknown_audit") != "not_executed"
        or v26e_report.get("deployment_eligible") is not False
        or v26c_contract.get("schema_version") != "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-contract-v1"
        or v26c_readiness.get("schema_version") != "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness-v1"
        or v26c_readiness.get("status") != "passed"
        or v26c_contract.get("deployment_eligible") is not False
        or int(v26c_contract.get("unknown_training_exposure", -1)) != 0
        or int(v26c_contract.get("unknown_normalization_exposure", -1)) != 0
    ):
        raise ValueError("V26f frozen V26e/V26c sources are not sealed successful diagnostics")
    # V26e itself carries a transitive, hash-bound V26c source contract.  Run
    # that validator here as well so V26f cannot bind a superficially valid
    # top-level report while an underlying fold/cache/checkpoint was replaced.
    validate_frozen_v26c_sources(root, v26e_contract)
    if v26e_readiness.get("contract_sha256") != sha256_file(root / V26E_CONTRACT):
        raise ValueError("V26f V26e readiness hash is stale")
    if v26c_readiness.get("contract_sha256") != sha256_file(root / V26C_CONTRACT):
        raise ValueError("V26f V26c readiness hash is stale")
    return {
        "v26e_contract": v26e_contract,
        "v26e_readiness": v26e_readiness,
        "v26e_report": v26e_report,
        "v26c_contract": v26c_contract,
        "v26c_readiness": v26c_readiness,
    }


def _validate_fold_sources(root: Path, source: dict[str, Any], fold_id: int) -> dict[str, Any]:
    paths = fold_paths(fold_id)
    e_fold = next(row for row in source["v26e_contract"]["folds"] if int(row["fold_id"]) == fold_id)
    c_fold = next(row for row in source["v26c_contract"]["folds"] if int(row["fold_id"]) == fold_id)
    if (
        e_fold.get("l2", {}).get("reused_from_v26c") is not True
        or e_fold.get("source_v26c_fold_contract_match", True) is not True
        or c_fold.get("source_v21e_fold_contract_match", True) is not True
        or sorted(str(value) for value in e_fold.get("train_parent_ids", ()))
        != sorted(str(value) for value in c_fold.get("train_parent_ids", ()))
        or sorted(str(value) for value in e_fold.get("held_out_parent_ids", ()))
        != sorted(str(value) for value in c_fold.get("held_out_parent_ids", ()))
    ):
        raise ValueError(f"V26f fold {fold_id} V26e L2 reuse marker is missing")

    def _check_recorded_path(
        record: Mapping[str, Any], key: str, path: Path, description: str
    ) -> None:
        if record.get(key) != relative_path(root, root / path):
            raise ValueError(f"V26f {description} path differs for fold {fold_id}")
        expected_hash = record.get(f"{key[:-5] if key.endswith('_path') else key}_sha256")
        # Most fold records use the shorter ``config_sha256`` style key.  The
        # explicit map below handles prediction/checkpoint/summary bindings.
        if key == "config_path":
            expected_hash = record.get("config_sha256")
        elif key == "checkpoint_path":
            expected_hash = record.get("checkpoint_sha256")
        elif key == "prediction_path":
            expected_hash = record.get("prediction_sha256")
        elif key == "summary_path":
            expected_hash = record.get("summary_sha256")
        if expected_hash is None or len(str(expected_hash)) != 64:
            raise ValueError(f"V26f {description} has no complete SHA256 binding")
        require_bound_file(root, root / path, description, expected_sha256=str(expected_hash))

    # L1's fold contract stores the data/config bindings; its checkpoint and
    # prediction paths are canonical V26e artifact paths and are additionally
    # checked against their JSON payloads below.
    l1_record = dict(e_fold.get("l1") or {})
    if l1_record.get("config_path") != relative_path(root, root / paths["l1_config"]):
        raise ValueError(f"V26f fold {fold_id} L1 config path differs")
    require_bound_file(
        root,
        root / paths["l1_config"],
        f"V26f fold {fold_id} L1 config",
        expected_sha256=str(l1_record.get("config_sha256", "")),
    )
    for name in ("l1_data_contract", "l1_normalization", "l1_train_cache", "l1_held_out_cache", "l1_checkpoint", "l1_prediction", "l1_summary"):
        require_bound_file(root, root / paths[name], f"V26f fold {fold_id} {name}")

    l2_record = dict(e_fold.get("l2") or {})
    c_l2_record = dict(c_fold.get("l2") or {})
    for key, path in (
        ("config_path", paths["l2_config"]),
        ("checkpoint_path", paths["l2_checkpoint"]),
        ("prediction_path", paths["l2_prediction"]),
        ("summary_path", paths["l2_summary"]),
    ):
        # V26e records the complete reused L2 artifact binding; V26c records
        # the immutable config/cache binding.  Both must agree byte-for-byte.
        _check_recorded_path(l2_record, key, path, f"fold {fold_id} L2 {key}")
        if key == "config_path":
            if c_l2_record.get("config_path") != relative_path(root, root / path) or c_l2_record.get("config_sha256") != sha256_file(root / path):
                raise ValueError(f"V26f fold {fold_id} V26c L2 config binding differs")

    l1_prediction = _source_json(root, root / paths["l1_prediction"], f"V26f fold {fold_id} L1 prediction")
    l1_summary = _source_json(root, root / paths["l1_summary"], f"V26f fold {fold_id} L1 summary")
    l2_prediction = _source_json(root, root / paths["l2_prediction"], f"V26f fold {fold_id} L2 prediction")
    l2_summary = _source_json(root, root / paths["l2_summary"], f"V26f fold {fold_id} L2 summary")
    if (
        l1_prediction.get("fold_id") != fold_id
        or l1_prediction.get("fixed_snapshot_epoch") != 32
        or l1_prediction.get("prediction_protocol") != "out_of_fold_fixed_epoch_32"
        or l1_prediction.get("unknown_training_exposure") != 0
        or l1_prediction.get("unknown_oof_prediction_count") != 0
        or l1_prediction.get("validation_access_count") != 0
        or l1_prediction.get("test_payload_access_count") != 0
        or l1_prediction.get("checkpoint_path") != relative_path(root, root / paths["l1_checkpoint"])
        or l1_prediction.get("checkpoint_sha256") != sha256_file(root / paths["l1_checkpoint"])
        or l1_summary.get("status") != "completed"
        or l1_summary.get("fixed_snapshot_epoch") != 32
        or l1_summary.get("prediction_sha256") != sha256_file(root / paths["l1_prediction"])
        or l2_prediction.get("fold_id") != fold_id
        or l2_prediction.get("fixed_snapshot_epoch") != 15
        or l2_prediction.get("prediction_protocol") != "out_of_fold_fixed_epoch_15"
        or l2_prediction.get("unknown_training_exposure") != 0
        or l2_prediction.get("unknown_oof_prediction_count") != 0
        or l2_prediction.get("validation_access_count") != 0
        or l2_prediction.get("test_payload_access_count") != 0
        or l2_prediction.get("checkpoint_path") != relative_path(root, root / paths["l2_checkpoint"])
        or l2_prediction.get("checkpoint_sha256") != sha256_file(root / paths["l2_checkpoint"])
        or l2_summary.get("status") != "completed"
        or l2_summary.get("fixed_snapshot_epoch") != 15
        or l2_summary.get("prediction_sha256") != sha256_file(root / paths["l2_prediction"])
    ):
        raise ValueError(f"V26f fold {fold_id} frozen prediction/summary binding differs")
    bindings: list[dict[str, Any]] = []
    for name, path in paths.items():
        checked = require_bound_file(root, root / path, f"V26f fold {fold_id} {name}")
        bindings.append({"name": name, **source_binding(root, checked, description=f"V26f fold {fold_id} {name}")})
    return {
        "fold_id": fold_id,
        "source_v26e_fold_contract": {
            "fold_id": fold_id,
            "train_parent_count": len(list(e_fold.get("train_parent_ids") or ())),
            "held_out_parent_count": len(list(e_fold.get("held_out_parent_ids") or ())),
            "train_parent_ids_sha256": e_fold.get("train_parent_ids_sha256"),
            "held_out_parent_ids_sha256": e_fold.get("held_out_parent_ids_sha256"),
            "train_component_count": e_fold.get("train_component_count"),
            "held_out_component_count": e_fold.get("held_out_component_count"),
            "component_overlap_count": e_fold.get("component_overlap_count"),
        },
        "source_v26c_fold_contract": {
            "fold_id": fold_id,
            "train_parent_count": len(list(c_fold.get("train_parent_ids") or ())),
            "held_out_parent_count": len(list(c_fold.get("held_out_parent_ids") or ())),
            "train_parent_ids_sha256": c_fold.get("train_parent_ids_sha256"),
            "held_out_parent_ids_sha256": c_fold.get("held_out_parent_ids_sha256"),
            "train_component_count": c_fold.get("train_component_count"),
            "held_out_component_count": c_fold.get("held_out_component_count"),
            "component_overlap_count": c_fold.get("component_overlap_count"),
        },
        "source_files": bindings,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source = _validate_source_pair(root)
    index_path = require_bound_file(root, root / INDEX, "CQ08 source index")
    manifest_path = require_bound_file(root, root / V26B_CACHE_MANIFEST, "V26b mixed cache manifest")
    manifest = read_json(manifest_path, description="V26b mixed cache manifest")
    if manifest.get("schema_version") != "RADAR-LN3-V26b-l2-mixed-open-set-feature-cache-v1":
        raise ValueError("V26f V26b mixed cache manifest schema differs")
    for split in ("validation", "test"):
        entry = dict(manifest.get(split) or {})
        if split == "validation":
            for field, cache_path in (
                ("physical_cache_sha256", L2_VALIDATION_PHYSICAL_CACHE),
                ("micro_doppler_cache_sha256", L2_VALIDATION_MICRO_CACHE),
            ):
                if entry.get(field) != sha256_file(root / cache_path):
                    raise ValueError(f"V26f V26b validation {field} binding differs")
        elif entry.get("test_payload_access_count_during_prepare") not in (0, None):
            raise ValueError("V26f V26b test cache was accessed during preparation")
    document_path = require_bound_file(root, root / DOCUMENT, "V26f experiment document")

    cache_bindings: list[dict[str, Any]] = []
    for split, l1_path, physical_path, micro_path in (
        ("val", L1_VALIDATION_CACHE, L2_VALIDATION_PHYSICAL_CACHE, L2_VALIDATION_MICRO_CACHE),
        ("test", L1_TEST_CACHE, L2_TEST_PHYSICAL_CACHE, L2_TEST_MICRO_CACHE),
    ):
        expected_ids = selected_event_sample_ids(root, split=split)
        l1 = validate_feature_cache(root / l1_path, expected_sample_ids=expected_ids, width=1, valid_rank=2, description=f"{split} L1 course cache")
        physical = validate_feature_cache(root / physical_path, expected_sample_ids=expected_ids, width=4, valid_rank=1, description=f"{split} L2 RD physical4 cache")
        micro = validate_feature_cache(root / micro_path, expected_sample_ids=expected_ids, width=8, valid_rank=1, description=f"{split} L2 micro-Doppler8 cache")
        l1["path"] = relative_path(root, root / l1_path)
        physical["path"] = relative_path(root, root / physical_path)
        micro["path"] = relative_path(root, root / micro_path)
        cache_bindings.append({"split": split, "parent_counts": expected_parent_counts(root, split=split), "l1_course": l1, "l2_rd_physical4": physical, "l2_micro_doppler8": micro})

    folds = [_validate_fold_sources(root, source, fold_id) for fold_id in range(FOLD_COUNT)]
    top_sources = [
        ("index", index_path),
        ("v26b_cache_manifest", manifest_path),
        ("v26e_contract", root / V26E_CONTRACT),
        ("v26e_readiness", root / V26E_READINESS),
        ("v26e_report", root / V26E_REPORT),
        ("v26c_contract", root / V26C_CONTRACT),
        ("v26c_readiness", root / V26C_READINESS),
        ("document", document_path),
    ]
    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26f frozen mixed Known/Unknown open-set audit",
        "scope": "inference-only validation/test audit after V26e L1 epoch-32 and V26c L2 epoch-15 are frozen",
        "source_constants": source_constants(),
        "source_bindings": [
            {"name": name, **source_binding(root, path, description=f"V26f {name}")}
            for name, path in top_sources
        ],
        "implementation_sha256": implementation_sha256(root),
        "fold_count": FOLD_COUNT,
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_checkpoint_selection_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "data": {
            "source_index": relative_path(root, index_path),
            "source_index_sha256": sha256_file(index_path),
            "mixed_labels": list(("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget")),
            "split_parent_counts": EXPECTED_PARENT_COUNTS,
            "cache_bindings": cache_bindings,
        },
        "folds": folds,
        "frozen_model_bindings": {
            "l1": "V26e fold-specific epoch-32 checkpoint/config/normalization; no forward before audit",
            "l2": "V26c fold-specific epoch-15 checkpoint/config/normalization; V26e reuses it read-only",
            "thresholds": "V26e cross-fitted fold-specific L3 thresholds; no refit in V26f",
        },
        "audit_protocol": {
            "validation_and_test_are_mixed_known_unknown": True,
            "unknown_training_exposure": 0,
            "unknown_normalization_exposure": 0,
            "unknown_checkpoint_selection_exposure": 0,
            "unknown_threshold_fit_exposure": 0,
            "threshold_source": "V26e known-only outer-cross-fitted thresholds, read-only",
            "folds_are_not_pooled_into_deployment_model": True,
            "pending_is_not_unknown_rejection_success": True,
            "l2_is_run_from_model_route_only": True,
            "true_label_never_controls_l1_or_l2_route": True,
            "minimum_evidence_events": 3,
        },
        "not_executed": {
            "training": True,
            "threshold_fitting": True,
            "oof_retraining": True,
            "deployment_registry": True,
        },
        "deployment_eligible": False,
    }
    write_status = write_immutable_json(root / CONTRACT, contract, max_bytes=2 * 1024 * 1024, description="contract")
    readiness = {
        "status": "passed",
        "schema_version": READINESS_SCHEMA,
        "contract_path": relative_path(root, root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "checks": {
            "frozen_v26e_source_is_sealed": True,
            "frozen_v26c_source_is_sealed": True,
            "all_mixed_caches_are_hash_and_order_valid": True,
            "all_five_fold_sources_are_present": True,
            "unknown_excluded_from_training_normalization_checkpoint_and_threshold_fit": True,
            "true_labels_do_not_control_routing": True,
            "pending_is_not_counted_as_unknown_rejection": True,
            "folds_not_pooled_for_deployment": True,
        },
        "fold_count": FOLD_COUNT,
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_checkpoint_selection_exposure": 0,
        "unknown_threshold_fit_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "threshold_fitting": "not_executed",
        "deployment_eligible": False,
    }
    readiness_status = write_immutable_json(root / READINESS, readiness, max_bytes=512 * 1024, description="readiness")
    return {
        "status": "passed",
        "schema_version": CONTRACT_SCHEMA,
        "contract": str(root / CONTRACT),
        "contract_sha256": sha256_file(root / CONTRACT),
        "contract_write_status": write_status,
        "readiness": str(root / READINESS),
        "readiness_write_status": readiness_status,
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "training": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the V26f frozen mixed open-set audit contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()