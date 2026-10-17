"""Prepare V26j: V26e's epoch-32 L1 OOF with causal RD GRU aggregation.

No feature or normalization values are rebuilt.  Preparation reads only the
frozen Known-only V26c/V26e protocol artifacts and creates new immutable V26j
contracts/configs whose single changed model field is
``rd_frame_aggregation='causal_gru'``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from build.known_gate_v26e_oof import EXPECTED_KNOWN_PARENT_COUNT, _require_v26c_source, _source_l1_config, _validate_frozen_assignment, _validate_frozen_l1_data, _validate_frozen_l2
from radar_ln3.known_gate import load_known_gate_config
from radar_ln3.known_gate_v26c_oof import fold_record as v26c_fold_record
from radar_ln3.known_gate_v26e_oof import l1_config_path as v26e_l1_config_path
from radar_ln3.known_gate_v26j_oof import CONTRACT, CONTRACT_SCHEMA, FOLD_COUNT, L1_CHECKPOINT_SCHEMA, L1_CONFIG_SCHEMA, L1_DATA_CONTRACT_SCHEMA, L1_EPOCH, L1_LABELS, L1_TRAIN_EPOCHS, L2_EPOCH, L2_LABELS, READINESS, READINESS_SCHEMA, TRAINING_SEED_L1, TRAINING_SEED_L2, V21E_ASSIGNMENT, V21E_FOLD_CONTRACT, V26C_CONTRACT, V26C_READINESS, V26E_CONTRACT, V26E_READINESS, l1_checkpoint_path, l1_config_path, l1_data_contract_path, l1_last_checkpoint_path, relative_path, sequence_sha256, sha256_file, write_new_or_identical


def _model_payload_for_v26j(source_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Create and validate the one-field V26j model declaration."""

    model = dict(source_payload.get("model") or {})
    if not model:
        raise ValueError("V26j V26 source config has no model declaration")
    original = str(model.get("rd_frame_aggregation", ""))
    if original != "masked_mean":
        raise ValueError("V26j source baseline must be the frozen masked_mean model")
    model["rd_frame_aggregation"] = "causal_gru"
    if (
        model.get("liquid_state_normalization") != "recurrent"
        or model.get("rd_frame_aggregation") != "causal_gru"
        or int(model.get("rd_temporal_residual_dim", 0)) != 0
    ):
        raise ValueError("V26j model declaration violates the one-variable causal-GRU design")
    return model


def _assert_direct_v26e_equivalence(
    root: Path,
    *,
    fold_id: int,
    v26j_payload: Mapping[str, Any],
    v26e_config_path: Path,
) -> None:
    """Prove all V26e-vs-V26j differences are namespace or RD aggregation.

    A raw JSON comparison is used here because V26e's private runtime adapter
    intentionally bypasses the current V26j config parser.  The permitted
    differences are only the fold-local data-contract hash/path, artifact
    namespace/checkpoint schema, config/schema/variant names, and the one
    causal variable ``model.rd_frame_aggregation``.
    """

    baseline = json.loads(v26e_config_path.read_text(encoding="utf-8"))
    treatment = json.loads(json.dumps(dict(v26j_payload)))
    baseline_model = dict(baseline.get("model") or {})
    treatment_model = dict(treatment.get("model") or {})
    if (
        baseline_model.get("rd_frame_aggregation") != "masked_mean"
        or treatment_model.get("rd_frame_aggregation") != "causal_gru"
    ):
        raise ValueError(f"V26j fold {fold_id} direct baseline aggregation differs")
    baseline_model.pop("rd_frame_aggregation", None)
    treatment_model.pop("rd_frame_aggregation", None)
    if baseline_model != treatment_model:
        raise ValueError(f"V26j fold {fold_id} model differs from V26e beyond RD aggregation")
    if dict(baseline.get("task") or {}) != dict(treatment.get("task") or {}):
        raise ValueError(f"V26j fold {fold_id} task declaration differs from V26e")
    baseline_data = dict(baseline.get("data") or {})
    treatment_data = dict(treatment.get("data") or {})
    allowed_data = {"contract_path", "expected_contract_sha256"}
    for field in sorted(set(baseline_data) | set(treatment_data)):
        if field not in allowed_data and baseline_data.get(field) != treatment_data.get(field):
            raise ValueError(f"V26j fold {fold_id} data field differs from V26e: {field}")
    baseline_train = dict(baseline.get("train") or {})
    treatment_train = dict(treatment.get("train") or {})
    allowed_train = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "checkpoint_schema",
    }
    for field in sorted(set(baseline_train) | set(treatment_train)):
        if field not in allowed_train and baseline_train.get(field) != treatment_train.get(field):
            raise ValueError(f"V26j fold {fold_id} train field differs from V26e: {field}")
    if (
        baseline.get("schema_version") != "RADAR-LN3-V26e-OOF-L1"
        or baseline.get("variant") != "v26_known_only_open_set_baseline"
        or treatment.get("schema_version") != L1_CONFIG_SCHEMA
        or treatment.get("variant") != "v26j_rd_causal_gru"
    ):
        raise ValueError(f"V26j fold {fold_id} direct baseline namespace differs")


def _l1_data_contract(
    root: Path,
    *,
    source_contract_path: Path,
    source_fold: Mapping[str, Any],
    frozen_data: Mapping[str, Any],
) -> dict[str, Any]:
    fold_id = int(source_fold["fold_id"])
    train_ids = [str(value) for value in source_fold["train_parent_ids"]]
    held_out_ids = [str(value) for value in source_fold["held_out_parent_ids"]]
    counts = dict(dict(source_fold["l1"]).get("train_label_parent_counts") or {})
    held_counts = dict(dict(source_fold["l1"]).get("held_out_label_parent_counts") or {})
    if set(counts) != set(L1_LABELS) or set(held_counts) != set(L1_LABELS):
        raise ValueError(f"V26j fold {fold_id} has incomplete L1 label support")
    return {
        "schema_version": L1_DATA_CONTRACT_SCHEMA,
        "experiment": "V26j RD causal-GRU aggregation only",
        "task_name": "known_gate",
        "fold_id": fold_id,
        "fit_split": "train",
        "fit_population": "known_fold_train_only",
        "grouping": "capture_content_provenance_component",
        "train_parent_count": len(train_ids),
        "held_out_parent_count": len(held_out_ids),
        "train_parent_ids_sha256": sequence_sha256(sorted(train_ids)),
        "held_out_parent_ids_sha256": sequence_sha256(sorted(held_out_ids)),
        "train_label_parent_counts": {label: int(counts[label]) for label in L1_LABELS},
        "held_out_label_parent_counts": {label: int(held_counts[label]) for label in L1_LABELS},
        "normalization": {
            "path": str(frozen_data["normalization_path"]),
            "sha256": str(frozen_data["normalization_sha256"]),
            "reused_from_v26c": True,
        },
        "cache_reuse": {
            "train": dict(frozen_data["train_course_cache"]),
            "held_out": dict(frozen_data["held_out_course_cache"]),
            "reused_from_v26c": True,
            "value_rebuild": False,
            "isolation": "exact disjoint frozen V26c Known-only fold populations",
        },
        "rd_aggregation": {
            "baseline": "masked_mean",
            "treatment": "causal_gru",
            "scope": "within a single RD event only; no cross-event RD state",
        },
        "source_v26c_contract": relative_path(root, source_contract_path),
        "source_v26c_contract_sha256": sha256_file(source_contract_path),
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_cache_selection_exposure": 0,
        "validation_access_count": 0,
        "test_access_count": 0,
        "checkpoint_selection": "fixed epoch-32 endpoint; no held-out labels",
        "l1_l2_fusion": "not_performed_in_fold_training",
    }


def _l1_config(
    root: Path,
    *,
    source_payload: Mapping[str, Any],
    source_config_path: Path,
    direct_v26e_config_path: Path,
    fold_id: int,
    data_contract_path: Path,
    frozen_data: Mapping[str, Any],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(dict(source_payload)))
    payload.pop("v26c_oof", None)
    payload["schema_version"] = L1_CONFIG_SCHEMA
    payload["config_name"] = (
        f"RADAR-LN3-V26j-cq08-known-gate-oof-fold-{fold_id:02d}-seed43-epoch32-rd-causal-gru"
    )
    payload["variant"] = "v26j_rd_causal_gru"
    payload["model"] = _model_payload_for_v26j(source_payload)
    data = dict(payload["data"])
    data.update(
        {
            "contract_path": relative_path(root, data_contract_path),
            "expected_contract_sha256": sha256_file(data_contract_path),
            "normalization_stats_path": str(frozen_data["normalization_path"]),
            "expected_stats_sha256": str(frozen_data["normalization_sha256"]),
            "train_feature_cache_path": str(frozen_data["train_course_cache"]["path"]),
            "validation_feature_cache_path": str(frozen_data["held_out_course_cache"]["path"]),
            "expected_train_feature_cache_sha256": str(frozen_data["train_course_cache"]["sha256"]),
            "expected_validation_feature_cache_sha256": str(frozen_data["held_out_course_cache"]["sha256"]),
            "feature_cache_manifest_path": None,
            "expected_feature_cache_manifest_sha256": None,
            "feature_experiment_contract_path": None,
            "expected_feature_experiment_contract_sha256": None,
        }
    )
    payload["data"] = data
    artifact = root / l1_checkpoint_path(fold_id).parent
    train = dict(payload["train"])
    train.update(
        {
            "epochs": L1_EPOCH,
            "early_stopping_patience": 0,
            "checkpoint_path": relative_path(root, root / l1_checkpoint_path(fold_id)),
            "last_checkpoint_path": relative_path(root, root / l1_last_checkpoint_path(fold_id)),
            "formal_artifact_dir": relative_path(root, artifact),
            "epoch_checkpoint_dir": relative_path(root, artifact / "epochs"),
            "checkpoint_schema": L1_CHECKPOINT_SCHEMA,
            "save_every_epoch": False,
            "paired_baseline_config_path": None,
        }
    )
    payload["train"] = train
    payload["v26j_oof"] = {
        "fold_id": fold_id,
        "role": "fresh L1 causal-GRU RD OOF treatment only; not a deployment checkpoint",
        "direct_baseline": "V26e masked_mean epoch-32 L1 OOF",
        "single_variable": "rd_frame_aggregation: masked_mean -> causal_gru",
        "train_epoch_limit": L1_TRAIN_EPOCHS,
        "fixed_snapshot_epoch": L1_EPOCH,
        "checkpoint_selection": "fixed endpoint; no held-out selection",
        "source_v26c_l1_config_path": relative_path(root, source_config_path),
        "source_v26c_l1_config_sha256": sha256_file(source_config_path),
        "source_v26c_l1_training_epochs": 20,
        "unknown_training_exposure": 0,
        "held_out_checkpoint_selection": False,
    }
    _assert_direct_v26e_equivalence(
        root,
        fold_id=fold_id,
        v26j_payload=payload,
        v26e_config_path=direct_v26e_config_path,
    )
    config_path = root / l1_config_path(fold_id)
    # Validate the generated structure before it ever becomes a contract
    # artifact.  This detects the classic config-only/no-runtime-GRU mistake.
    temporary = config_path.with_suffix(".validation.json")
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        config = load_known_gate_config(temporary)
        if config.model.rd_frame_aggregation != "causal_gru":
            raise ValueError("V26j generated config did not retain causal_gru")
    finally:
        if temporary.exists():
            temporary.unlink()
    return payload


def build(project_root: Path, *, preview: bool = False) -> dict[str, Any]:
    root = project_root.resolve()
    source_contract, _source_readiness = _require_v26c_source(root)
    source_contract_path = root / V26C_CONTRACT
    assignment = _validate_frozen_assignment(root, source_contract)
    all_known_parent_ids = sorted(assignment)
    all_dbb_parent_ids = sorted(
        parent_id
        for parent_id, row in assignment.items()
        if str(row["label_name"]) in L2_LABELS
    )
    if len(all_known_parent_ids) != EXPECTED_KNOWN_PARENT_COUNT or len(all_dbb_parent_ids) != 953:
        raise ValueError("V26j frozen Known/DBB parent universes are incomplete")
    folds: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        source_fold = v26c_fold_record(source_contract, fold_id)
        expected_held = {
            parent_id for parent_id, row in assignment.items() if int(row["fold_id"]) == fold_id
        }
        if set(source_fold["held_out_parent_ids"]) != expected_held:
            raise ValueError(f"V26j fold {fold_id} differs from the frozen assignment")
        source_config_path, source_config = _source_l1_config(root, source_fold)
        direct_v26e_config_path = root / v26e_l1_config_path(fold_id)
        if not direct_v26e_config_path.is_file():
            raise FileNotFoundError(
                f"V26j fold {fold_id} direct V26e config is missing: {direct_v26e_config_path}"
            )
        frozen_data = _validate_frozen_l1_data(root, source_fold, all_known_parent_ids=all_known_parent_ids)
        l2_reuse = _validate_frozen_l2(
            root,
            source_fold,
            assignment=assignment,
            all_known_parent_ids=all_known_parent_ids,
            all_dbb_parent_ids=all_dbb_parent_ids,
            source_contract=source_contract,
            source_contract_path=source_contract_path,
            source_readiness_path=root / V26C_READINESS,
        )
        if preview:
            folds.append({"fold_id": fold_id, "train_parent_count": len(source_fold["train_parent_ids"]), "held_out_parent_count": len(source_fold["held_out_parent_ids"]), "rd_frame_aggregation": "causal_gru"})
            continue
        data_contract_path = root / l1_data_contract_path(fold_id)
        data_contract = _l1_data_contract(
            root,
            source_contract_path=source_contract_path,
            source_fold=source_fold,
            frozen_data=frozen_data,
        )
        write_new_or_identical(data_contract_path, data_contract)
        config_path = root / l1_config_path(fold_id)
        config_payload = _l1_config(
            root,
            source_payload=source_config,
            source_config_path=source_config_path,
            direct_v26e_config_path=direct_v26e_config_path,
            fold_id=fold_id,
            data_contract_path=data_contract_path,
            frozen_data=frozen_data,
        )
        write_new_or_identical(config_path, config_payload)
        config = load_known_gate_config(config_path)
        if config.model.rd_frame_aggregation != "causal_gru":
            raise ValueError("V26j persisted config is not causal_gru")
        folds.append(
            {
                "fold_id": fold_id,
                "source_v26c_fold_contract_match": True,
                "train_parent_ids": list(source_fold["train_parent_ids"]),
                "held_out_parent_ids": list(source_fold["held_out_parent_ids"]),
                "train_parent_ids_sha256": sequence_sha256(sorted(str(value) for value in source_fold["train_parent_ids"])),
                "held_out_parent_ids_sha256": sequence_sha256(sorted(str(value) for value in source_fold["held_out_parent_ids"])),
                "train_component_count": int(source_fold["train_component_count"]),
                "held_out_component_count": int(source_fold["held_out_component_count"]),
                "component_overlap_count": 0,
                "l1": {
                    "train_parent_count": len(source_fold["train_parent_ids"]),
                    "held_out_parent_count": len(source_fold["held_out_parent_ids"]),
                    "train_label_parent_counts": dict(source_fold["l1"]["train_label_parent_counts"]),
                    "held_out_label_parent_counts": dict(source_fold["l1"]["held_out_label_parent_counts"]),
                    "config_path": relative_path(root, config_path),
                    "config_sha256": sha256_file(config_path),
                    "data_contract_path": relative_path(root, data_contract_path),
                    "data_contract_sha256": sha256_file(data_contract_path),
                    "normalization_path": str(frozen_data["normalization_path"]),
                    "normalization_sha256": str(frozen_data["normalization_sha256"]),
                    "train_course_cache": dict(frozen_data["train_course_cache"]),
                    "held_out_course_cache": dict(frozen_data["held_out_course_cache"]),
                    "rd_frame_aggregation": "causal_gru",
                    "rd_aggregation_scope": "within_current_event_only",
                    "train_epoch_limit": L1_EPOCH,
                    "fixed_snapshot_epoch": L1_EPOCH,
                    "held_out_checkpoint_selection": False,
                },
                "l2": {
                    **dict(source_fold["l2"]),
                    **l2_reuse,
                    "reused_from_v26c": True,
                    "fixed_snapshot_epoch": L2_EPOCH,
                    "held_out_checkpoint_selection": False,
                    "training_executed_in_v26j": False,
                },
            }
        )
    if preview:
        return {"status": "preview_passed", "read_only": True, "fold_count": FOLD_COUNT, "l1_train_epoch_limit": L1_EPOCH, "l2_reused_fixed_snapshot_epoch": L2_EPOCH, "rd_frame_aggregation": "causal_gru", "folds": folds}

    contract = {
        "schema_version": CONTRACT_SCHEMA,
        "experiment": "V26j causal-GRU event-internal RD aggregation L1 epoch-32 Known-only provenance-grouped OOF",
        "direct_baseline": "V26e L1 epoch-32 masked_mean OOF",
        "scope": "fresh fold-local L1 training; only RD frame aggregation differs; V26c L2 epoch-15 outputs are reused read-only",
        "single_variable": {"field": "model.rd_frame_aggregation", "control": "masked_mean", "treatment": "causal_gru", "causal_scope": "within current RD event only"},
        "source_v26c": {
            "contract_path": relative_path(root, source_contract_path),
            "contract_sha256": sha256_file(source_contract_path),
            "readiness_path": relative_path(root, root / V26C_READINESS),
            "readiness_sha256": sha256_file(root / V26C_READINESS),
            "l2_epoch15_outputs_reused_read_only": True,
        },
        "direct_v26e_protocol_reference": {
            "contract_path": relative_path(root, root / V26E_CONTRACT),
            "contract_sha256": sha256_file(root / V26E_CONTRACT),
            "readiness_path": relative_path(root, root / V26E_READINESS),
            "readiness_sha256": sha256_file(root / V26E_READINESS),
            "variable_difference_only": "model.rd_frame_aggregation",
        },
        "assignment_reuse": {"source": relative_path(root, root / V21E_ASSIGNMENT), "sha256": sha256_file(root / V21E_ASSIGNMENT), "source_contract": relative_path(root, root / V21E_FOLD_CONTRACT), "source_contract_sha256": sha256_file(root / V21E_FOLD_CONTRACT), "scope": "V26c frozen provenance-grouped assignment only"},
        "feature_and_normalization_reuse": {"source_v26c_values_reused_read_only": True, "feature_rebuild": False, "normalization_rebuild": False, "l1_cache_and_normalization_paths_are_v26c": True},
        "fold_count": FOLD_COUNT,
        "known_train_parent_count": EXPECTED_KNOWN_PARENT_COUNT,
        "grouping": "capture_content_provenance_component",
        "l1": {"task": "known_gate", "training_seed": TRAINING_SEED_L1, "train_epoch_limit": L1_EPOCH, "fixed_snapshot_epoch": L1_EPOCH, "checkpoint_selection": "fixed endpoint; no per-fold held-out selection", "rd_frame_aggregation": "causal_gru", "unknown_training_exposure": 0},
        "l2": {"task": "dbb_specialist", "source": "V26c sealed fold L2 outputs", "training_seed": TRAINING_SEED_L2, "fixed_snapshot_epoch": L2_EPOCH, "training_executed_in_v26j": False, "unknown_training_exposure": 0},
        "l3_calibration_protocol": {"fit_population": "Known parent-final OOF scores only", "outer_cross_fit": "thresholds for fold f are fitted with other four folds only", "unknown_threshold_fitting": False, "pending_is_unknown_rejection_success": False},
        "folds": folds,
        "validation_access_count": 0,
        "test_access_count": 0,
        "unknown_training_exposure": 0,
        "unknown_normalization_exposure": 0,
        "unknown_oof_prediction_count": 0,
        "deployment_eligible": False,
    }
    write_new_or_identical(root / CONTRACT, contract)
    checks = {
        "v26c_source_contract_and_readiness_are_frozen": True,
        "v26e_direct_baseline_protocol_is_hash_bound": True,
        "v26c_provenance_assignment_is_reused_byte_for_byte": True,
        "each_known_train_parent_is_held_out_once_for_l1": True,
        "no_provenance_component_crosses_a_fold": True,
        "l1_normalization_and_feature_values_are_reused_from_v26c": True,
        "only_rd_frame_aggregation_changes_from_v26e": True,
        "rd_frame_aggregation_is_causal_gru_in_all_v26j_configs": True,
        "unknown_is_excluded_from_training_normalization_cache_and_calibration": True,
        "l1_train_epoch_limit_is_32": True,
        "l1_fixed_endpoint_is_epoch_32": True,
        "v26c_l2_epoch15_outputs_are_reused_read_only": True,
        "validation_and_test_are_unread_during_prepare": True,
    }
    readiness = {"status": "passed", "schema_version": READINESS_SCHEMA, "contract_path": relative_path(root, root / CONTRACT), "contract_sha256": sha256_file(root / CONTRACT), "checks": checks, "fold_count": FOLD_COUNT, "l1_train_epoch_limit": L1_EPOCH, "l1_fixed_snapshot_epoch": L1_EPOCH, "l2_reused_fixed_snapshot_epoch": L2_EPOCH, "rd_frame_aggregation": "causal_gru", "known_train_parent_count": EXPECTED_KNOWN_PARENT_COUNT, "unknown_training_exposure": 0, "unknown_oof_prediction_count": 0, "validation_access_count": 0, "test_access_count": 0, "l3_calibration": "not_executed", "mixed_unknown_audit": "not_executed", "deployment_eligible": False}
    write_new_or_identical(root / READINESS, readiness)
    return {"status": readiness["status"], "contract": str(root / CONTRACT), "readiness": str(root / READINESS), "fold_count": FOLD_COUNT, "rd_frame_aggregation": "causal_gru", "l1_runs": FOLD_COUNT, "l2_runs": 0, "l3_calibration": "not_executed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare V26j RD causal-GRU L1 OOF.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve(), preview=args.preview), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()