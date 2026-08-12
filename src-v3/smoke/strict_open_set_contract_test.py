from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from radar_ln3.cascade import (
    EXPECTED_TASKS,
    OnlineLiquidSoftCascade,
    STRICT_OPEN_SET_EVIDENCE_NAMES,
    build_cascade_dataset,
    summarize_strict_open_set_known_records,
    validate_cascade_configs,
)
from radar_ln3.losses import strict_open_set_final_state_nll
from radar_ln3.rejection import (
    CALIBRATION_PURPOSE_AUDIT,
    CU_CLUTTER_PROBABILITY_SCORE,
    DBB_KNOWN_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
    STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA_V1,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    StrictOpenSetRejectionCalibration,
    StrictOpenSetRejectionPolicy,
    load_strict_open_set_rejection_calibration,
)
from radar_ln3.strict_open_set_oof import (
    derive_oof_fold_configs,
    fit_known_only_oof_boundaries,
    fold_record,
    load_oof_fold_contract,
    sequence_sha256,
    verify_oof_dataset,
)
from radar_ln3.task_models import (
    KNOWN_BASE_LABELS,
    STRICT_OPEN_SET_SCHEMA_VERSION,
    UNKNOWN_BASE_LABEL,
    LiquidTaskClassifier,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
    task_config_identity_sha256,
    verify_task_data_contract,
)
from train.cascade import build_product_checkpoint_selection


CONFIG_NAMES = {
    "dbb_cu_gate": "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-cu-gate-config.json",
    "dbb_specialist": "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-specialist-config.json",
    "cu_specialist": "RADAR-LN3-V14-cq08-liquid-strict-open-set-cu-specialist-config.json",
}


def _expect_failure(call: Callable[[], Any], contains: str) -> str:
    try:
        call()
    except (ValueError, RuntimeError, PermissionError) as error:
        message = str(error)
        if contains.lower() not in message.lower():
            raise AssertionError(f"failure did not mention {contains!r}: {message}") from error
        return message
    raise AssertionError(f"expected failure containing {contains!r}")


def _loss_contract() -> dict[str, Any]:
    torch.manual_seed(20260809)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    padding = torch.tensor(
        [
            [False, False, False, False],
            [False, False, False, True],
            [False, False, True, True],
            [False, False, False, False],
        ],
        dtype=torch.bool,
    )
    gate = torch.randn(4, 4, 2, requires_grad=True)
    dbb = torch.randn(4, 4, 3, requires_grad=True)
    cu = torch.randn(4, 4, 2, requires_grad=True)
    objective = strict_open_set_final_state_nll(
        gate,
        dbb,
        cu,
        labels,
        padding,
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
        gate_class_weights=torch.tensor([0.5179303278688525, 14.442857142857143]),
        dbb_class_weights=torch.tensor([0.6666666666666666, 1.3388203017832647, 1.327891156462585]),
        cu_class_weights=torch.tensor([14.442857142857143, 0.5179303278688525]),
    )
    objective["loss"].backward()
    gradients = {
        "gate": float(gate.grad.abs().sum().item()),
        "dbb": float(dbb.grad.abs().sum().item()),
        "cu": float(cu.grad.abs().sum().item()),
    }
    if any(value <= 0.0 for value in gradients.values()):
        raise AssertionError(f"V14 known-only loss omitted a task gradient: {gradients}")
    if int(objective["unknown_parent_count"].item()) != 0:
        raise AssertionError("V14 objective reported Unknown exposure")
    if float(dbb.grad[3].abs().max().item()) != 0.0:
        raise AssertionError("Clutter parent produced a DBB specialist gradient")
    unknown_error = _expect_failure(
        lambda: strict_open_set_final_state_nll(
            torch.randn(1, 2, 2),
            torch.randn(1, 2, 3),
            torch.randn(1, 2, 2),
            torch.tensor([4]),
            torch.zeros(1, 2, dtype=torch.bool),
        ),
        "UnknownTarget",
    )
    return {
        "loss": float(objective["loss"].item()),
        "gradients": gradients,
        "clutter_to_dbb_gradient_max": float(dbb.grad[3].abs().max().item()),
        "unknown_loss_rejected": True,
        "unknown_error": unknown_error,
    }


def _calibration() -> StrictOpenSetRejectionCalibration:
    return StrictOpenSetRejectionCalibration(
        schema_version=STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
        gate_score_name=GATE_DBB_PROBABILITY_SCORE,
        dbb_known_score_name=DBB_KNOWN_PROBABILITY_SCORE,
        cu_score_name=CU_CLUTTER_PROBABILITY_SCORE,
        gate_accept_dbb_threshold=0.70,
        gate_accept_dbb_margin=0.20,
        dbb_accept_known_threshold=0.70,
        dbb_accept_known_margin=0.20,
        cu_accept_clutter_threshold=0.70,
        cu_accept_clutter_margin=0.20,
        calibration_quality_status="deployment_eligible",
        deployment_eligible=True,
        min_valid_points=2,
        min_track_quality=0.0,
        max_gap_fraction=1.0,
    )


def _action_contract() -> dict[str, Any]:
    policy = StrictOpenSetRejectionPolicy(_calibration(), enabled=True)

    def decide(gate: list[float], dbb: list[float], cu: list[float]):
        return policy.decide(
            torch.tensor(gate),
            torch.tensor(dbb),
            torch.tensor(cu),
            valid_point_count=3,
            track_quality=1.0,
            gap_fraction=0.0,
        )

    dbb_accepted = decide([0.90, 0.10], [0.05, 0.90, 0.05], [0.90, 0.10])
    dbb_fallback_clutter = decide([0.90, 0.10], [0.40, 0.35, 0.25], [0.90, 0.10])
    dbb_fallback_unknown = decide([0.90, 0.10], [0.40, 0.35, 0.25], [0.55, 0.45])
    clutter_accepted = decide([0.20, 0.80], [0.90, 0.05, 0.05], [0.90, 0.10])
    cu_rejected = decide([0.20, 0.80], [0.90, 0.05, 0.05], [0.55, 0.45])
    if (
        dbb_accepted.final_action != "DBB_ACCEPTED"
        or dbb_accepted.accepted_label_id != 1
        or dbb_accepted.rejected_to_unknown
    ):
        raise AssertionError("V14 failed the Gate+DBB-knownness accept path")
    if (
        dbb_fallback_clutter.final_action != "CLUTTER_ACCEPTED"
        or dbb_fallback_clutter.accepted_label_id != 3
        or dbb_fallback_clutter.gate_action != "DBB_ROUTE_ACCEPTED_THEN_FALLBACK_TO_CU"
    ):
        raise AssertionError("V14 DBB knownness failure did not fall back to CU Clutter acceptance")
    for decision, source in ((dbb_fallback_unknown, "DBB-to-CU"), (cu_rejected, "CU")):
        if (
            decision.final_action != "UNKNOWN_REJECTED"
            or decision.accepted_label_id is not None
            or decision.rejected_to_unknown is not True
        ):
            raise AssertionError(f"V14 {source} rejection did not converge on UNKNOWN_REJECTED")
    if clutter_accepted.final_action != "CLUTTER_ACCEPTED" or clutter_accepted.accepted_label_id != 3:
        raise AssertionError("V14 failed the Clutter acceptance path")
    uncalibrated = StrictOpenSetRejectionPolicy().decide(
        torch.tensor([0.9, 0.1]),
        torch.tensor([0.05, 0.9, 0.05]),
        torch.tensor([0.9, 0.1]),
        valid_point_count=3,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    if uncalibrated.final_action != "NOT_CALIBRATED" or uncalibrated.accepted_label_id is not None:
        raise AssertionError("uncalibrated V14 policy emitted a product action")
    return {
        "dbb_accepted": dbb_accepted.final_action,
        "dbb_knownness_failed_then_cu_clutter": dbb_fallback_clutter.final_action,
        "dbb_knownness_failed_then_cu_unknown": dbb_fallback_unknown.final_action,
        "clutter_accepted": clutter_accepted.final_action,
        "clutter_not_accepted": cu_rejected.final_action,
        "uncalibrated": uncalibrated.final_action,
        "both_rejection_paths_have_no_accepted_label": True,
    }


def _selection_contract(config) -> dict[str, Any]:
    known_metrics = summarize_strict_open_set_known_records(
        [
            {"true_id": 0, "pred_id": 0},
            {"true_id": 1, "pred_id": 1},
            {"true_id": 2, "pred_id": 2},
            {"true_id": 3, "pred_id": 3},
        ],
        scope="synthetic_known_only",
    )
    binary = {
        "per_class": [
            {"class": "a", "precision": 1.0, "recall": 1.0, "support": 2},
            {"class": "b", "precision": 1.0, "recall": 1.0, "support": 2},
        ]
    }
    dbb = {
        "per_class": [
            {"class": name, "precision": 1.0, "recall": 1.0, "support": 1}
            for name in ("DroneTarget", "BirdTarget", "BalloonTarget")
        ]
    }
    validation = {
        "known_closed_set": {"parent_metrics": known_metrics},
        "checkpoint_selection_inputs": {
            "status": "available_known_only",
            "unknown_parent_count": 0,
            "gate_min_precision_recall": 1.0,
            "dbb_min_precision_recall": 1.0,
            "cu_min_precision_recall": 1.0,
            "known_parent_macro_f1": 1.0,
            "known_parent_drone_bird_min_precision_recall": 1.0,
            "unknown_used_for_selection": False,
        },
        "component_metrics": {
            "gate_known_parent": binary,
            "dbb_known_parent_conditional": dbb,
            "clutter_acceptance_known_parent": binary,
        },
    }
    selection = build_product_checkpoint_selection(validation, config)
    if selection["unknown_parent_count"] != 0 or selection["unknown_used_for_selection"] is not False:
        raise AssertionError("V14 selection did not preserve zero Unknown exposure")
    contaminated = copy.deepcopy(validation)
    contaminated["checkpoint_selection_inputs"]["unknown_parent_count"] = 1
    error = _expect_failure(
        lambda: build_product_checkpoint_selection(contaminated, config),
        "Unknown",
    )
    return {
        "guardrail_pass_count": selection["primary_value"],
        "unknown_used_for_selection": selection["unknown_used_for_selection"],
        "contaminated_selection_rejected": True,
        "contaminated_error": error,
    }


def _calibration_loader_contract(project_root: Path, config) -> dict[str, Any]:
    smoke_dir = project_root / "src-v3" / "smoke"
    valid_path = smoke_dir / ".RADAR-LN3-V14-calibration-valid.tmp.json"
    invalid_path = smoke_dir / ".RADAR-LN3-V14-calibration-invalid.tmp.json"
    audit_path = smoke_dir / ".RADAR-LN3-V14-calibration-audit.tmp.json"
    legacy_path = smoke_dir / ".RADAR-LN3-V14-calibration-legacy.tmp.json"
    inconsistent_path = smoke_dir / ".RADAR-LN3-V14-calibration-inconsistent.tmp.json"
    zeros = {name: "0" * 64 for name in EXPECTED_TASKS}
    identity = {
        "task_config_identity_sha256": zeros,
        "checkpoint_sha256": zeros,
        "epoch": 1,
        "global_step": 1,
        "product_selection_sha256": "0" * 64,
    }
    component_targets_passed = {
        "gate": True,
        "dbb_knownness": True,
        "clutter_acceptance": True,
    }
    payload = {
        "schema_version": STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        "source": TRAIN_PARENT_GROUPED_OOF_SOURCE,
        "gate_score_name": GATE_DBB_PROBABILITY_SCORE,
        "dbb_known_score_name": DBB_KNOWN_PROBABILITY_SCORE,
        "cu_score_name": CU_CLUTTER_PROBABILITY_SCORE,
        "gate_accept_dbb_threshold": 0.7,
        "gate_accept_dbb_margin": 0.2,
        "dbb_accept_known_threshold": 0.7,
        "dbb_accept_known_margin": 0.2,
        "cu_accept_clutter_threshold": 0.7,
        "cu_accept_clutter_margin": 0.2,
        "calibration_quality_status": "deployment_eligible",
        "deployment_eligible": True,
        "quality_gate": {
            "required_components": list(component_targets_passed),
            "component_targets_passed": component_targets_passed,
            "all_required_targets_passed": True,
            "failure_reasons": [],
        },
        "min_valid_points": 2,
        "min_track_quality": 0.0,
        "max_gap_fraction": 1.0,
        "provenance": {
            "fit_split": "train",
            "fit_population": "known_only",
            "grouping": "parent",
            "prediction_protocol": "out_of_fold",
            "unknown_training_exposure": 0,
            "unknown_threshold_fit_count": 0,
            "validation_used_for_threshold_fitting": False,
            "test_used": False,
            "known_parent_count": 1011,
            "oof_prediction_parent_count": 1011,
            "oof_unique_parent_count": 1011,
            "data_contract_hashes": {
                "index_sha256": config.data.expected_index_sha256,
                "contract_sha256": config.data.expected_contract_sha256,
                "stats_sha256": config.data.expected_stats_sha256,
            },
            "deployment_checkpoint_set": identity,
        },
        "selection_diagnostics": {
            name: {"all_precision_recall_targets_passed": passed}
            for name, passed in component_targets_passed.items()
        },
    }
    try:
        valid_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        calibration, _ = load_strict_open_set_rejection_calibration(valid_path)
        invalid = copy.deepcopy(payload)
        invalid["provenance"]["unknown_threshold_fit_count"] = 1
        invalid_path.write_text(json.dumps(invalid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        error = _expect_failure(
            lambda: load_strict_open_set_rejection_calibration(invalid_path),
            "known-only",
        )
        audit_payload = copy.deepcopy(payload)
        audit_payload["selection_diagnostics"]["gate"][
            "all_precision_recall_targets_passed"
        ] = False
        audit_payload["calibration_quality_status"] = "audit_only"
        audit_payload["deployment_eligible"] = False
        audit_payload["quality_gate"] = {
            "required_components": list(component_targets_passed),
            "component_targets_passed": {
                "gate": False,
                "dbb_knownness": True,
                "clutter_acceptance": True,
            },
            "all_required_targets_passed": False,
            "failure_reasons": ["gate_precision_recall_target_not_met"],
        }
        audit_path.write_text(
            json.dumps(audit_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        deployment_quality_error = _expect_failure(
            lambda: load_strict_open_set_rejection_calibration(audit_path),
            "audit_only",
        )
        audit_calibration, normalized_audit = load_strict_open_set_rejection_calibration(
            audit_path,
            purpose=CALIBRATION_PURPOSE_AUDIT,
        )
        policy_quality_error = _expect_failure(
            lambda: StrictOpenSetRejectionPolicy(audit_calibration, enabled=True),
            "audit-only",
        )
        audit_policy = StrictOpenSetRejectionPolicy(
            audit_calibration,
            enabled=True,
            allow_audit_only=True,
        )
        online_audit_error = _expect_failure(
            lambda: OnlineLiquidSoftCascade(
                SimpleNamespace(
                    gate=SimpleNamespace(
                        config=SimpleNamespace(schema_version=STRICT_OPEN_SET_SCHEMA_VERSION)
                    )
                ),
                hierarchical_rejection_policy=audit_policy,
            ),
            "online",
        )
        legacy_payload = copy.deepcopy(audit_payload)
        legacy_payload["schema_version"] = STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA_V1
        for name in ("calibration_quality_status", "deployment_eligible", "quality_gate"):
            legacy_payload.pop(name)
        legacy_path.write_text(
            json.dumps(legacy_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        legacy_deployment_error = _expect_failure(
            lambda: load_strict_open_set_rejection_calibration(legacy_path),
            "audit_only",
        )
        legacy_calibration, normalized_legacy = load_strict_open_set_rejection_calibration(
            legacy_path,
            purpose=CALIBRATION_PURPOSE_AUDIT,
        )
        inconsistent_payload = copy.deepcopy(audit_payload)
        inconsistent_payload["deployment_eligible"] = True
        inconsistent_path.write_text(
            json.dumps(inconsistent_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        inconsistent_quality_error = _expect_failure(
            lambda: load_strict_open_set_rejection_calibration(
                inconsistent_path,
                purpose=CALIBRATION_PURPOSE_AUDIT,
            ),
            "disagrees",
        )
    finally:
        valid_path.unlink(missing_ok=True)
        invalid_path.unlink(missing_ok=True)
        audit_path.unlink(missing_ok=True)
        legacy_path.unlink(missing_ok=True)
        inconsistent_path.unlink(missing_ok=True)
    return {
        "valid_schema": calibration.schema_version,
        "valid_deployment_eligible": calibration.deployment_eligible,
        "unknown_threshold_fit_count_zero_required": True,
        "contaminated_calibration_rejected": True,
        "contaminated_error": error,
        "audit_only_default_deployment_rejected": True,
        "audit_only_deployment_error": deployment_quality_error,
        "audit_only_explicit_audit_loads": normalized_audit["calibration_quality_status"]
        == "audit_only",
        "audit_only_policy_default_rejected": True,
        "audit_only_policy_error": policy_quality_error,
        "audit_only_online_injection_rejected": True,
        "audit_only_online_error": online_audit_error,
        "legacy_v1_quality_derived": (
            legacy_calibration.deployment_eligible is False
            and normalized_legacy["quality_status_source"]
            == "derived_from_legacy_v1_selection_diagnostics"
        ),
        "legacy_v1_default_deployment_rejected": True,
        "legacy_v1_deployment_error": legacy_deployment_error,
        "inconsistent_v2_quality_rejected": True,
        "inconsistent_v2_quality_error": inconsistent_quality_error,
    }


def _checkpoint_contract(project_root: Path, v14_config) -> dict[str, Any]:
    smoke_dir = project_root / "src-v3" / "smoke"
    path = smoke_dir / ".RADAR-LN3-V14-checkpoint-roundtrip.tmp.pt"
    model = LiquidTaskClassifier(v14_config)
    selection = {
        "metric": v14_config.train.checkpoint_selection_metric,
        "ranking_key": [0.0] * 8,
        "unknown_parent_count": 0,
        "unknown_used_for_selection": False,
    }
    save_task_checkpoint(
        path,
        model,
        None,
        config=v14_config,
        epoch=1,
        global_step=1,
        metrics={"checkpoint_selection": selection, "checkpoint_set_id": "0" * 32},
    )
    try:
        restored = LiquidTaskClassifier(v14_config)
        payload = load_task_checkpoint(
            path,
            restored,
            expected_schema=v14_config.train.checkpoint_schema,
            map_location="cpu",
        )
        model_contract = dict(payload["model_contract"])
        if (
            model_contract.get("unknown_training_exposure") != 0
            or model_contract.get("unknown_is_model_class") is not False
            or model_contract.get("unknown_is_rejection_action_only") is not True
        ):
            raise AssertionError("V14 checkpoint omitted strict-open-set fields")
        v13_path = (
            project_root
            / "config/RADAR-LN3/RADAR-LN3-V13-cq08-liquid-hierarchical-dbb-cu-gate-config.json"
        )
        v13_config = load_task_config(v13_path)
        v13_model = LiquidTaskClassifier(v13_config)
        mismatch = _expect_failure(
            lambda: load_task_checkpoint(
                path,
                v13_model,
                expected_schema=v13_config.train.checkpoint_schema,
                map_location="cpu",
            ),
            "schema",
        )
    finally:
        path.unlink(missing_ok=True)
    return {
        "round_trip": True,
        "config_identity_sha256": task_config_identity_sha256(v14_config),
        "unknown_training_exposure": model_contract["unknown_training_exposure"],
        "v13_v14_mismatch_rejected": True,
        "mismatch_error": mismatch,
    }


def _oof_fold_contract(project_root: Path, configs) -> dict[str, Any]:
    fold_contract, assignment = load_oof_fold_contract(project_root)
    seen: list[str] = []
    folds: list[dict[str, Any]] = []
    for fold_id in range(int(fold_contract["fold_count"])):
        fold = fold_record(fold_contract, fold_id)
        derived = derive_oof_fold_configs(project_root, configs, fold)
        contracts = {
            name: verify_task_data_contract(project_root, config)
            for name, config in derived.items()
        }
        train_dataset = build_cascade_dataset(
            project_root,
            derived["dbb_cu_gate"],
            split_name="train",
            label_scope="known_only",
            parent_ids=fold["train_parent_ids"],
        )
        held_out_dataset = build_cascade_dataset(
            project_root,
            derived["dbb_cu_gate"],
            split_name="train",
            label_scope="known_only",
            parent_ids=fold["held_out_parent_ids"],
        )
        train_scope = verify_oof_dataset(
            train_dataset,
            expected_parent_ids=fold["train_parent_ids"],
            scope=f"focused_fold_{fold_id:02d}_train",
        )
        held_out_scope = verify_oof_dataset(
            held_out_dataset,
            expected_parent_ids=fold["held_out_parent_ids"],
            scope=f"focused_fold_{fold_id:02d}_held_out",
        )
        if set(train_dataset.parent_ids) & set(held_out_dataset.parent_ids):
            raise AssertionError(f"V14 OOF fold {fold_id} Dataset scopes overlap")
        seen.extend(held_out_dataset.parent_ids)
        folds.append({
            "fold_id": fold_id,
            "train": train_scope,
            "held_out": held_out_scope,
            "normalization_stats_sha256": derived["dbb_cu_gate"].data.expected_stats_sha256,
            "data_contract_sha256": derived["dbb_cu_gate"].data.expected_contract_sha256,
            "data_contract_verified_for_all_tasks": len(contracts) == len(EXPECTED_TASKS),
            "epochs": derived["dbb_cu_gate"].train.epochs,
            "early_stopping_patience": derived["dbb_cu_gate"].train.early_stopping_patience,
        })
    assigned = sorted(str(row["parent_sample_id"]) for row in assignment["assignments"])
    if sorted(seen) != assigned or len(set(seen)) != 1011:
        raise AssertionError("V14 OOF held-out folds do not cover every known parent exactly once")
    unavailable = _expect_failure(
        lambda: build_cascade_dataset(
            project_root,
            derive_oof_fold_configs(
                project_root,
                configs,
                fold_record(fold_contract, 0),
            )["dbb_cu_gate"],
            split_name="train",
            label_scope="known_only",
            parent_ids=["cq08|track|does-not-exist"],
        ),
        "unavailable",
    )
    return {
        "schema_version": fold_contract["schema_version"],
        "fold_count": fold_contract["fold_count"],
        "known_parent_count": len(seen),
        "known_parent_ids_sha256": sequence_sha256(sorted(seen)),
        "unknown_training_exposure": 0,
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "every_known_parent_exactly_one_held_out_fold": True,
        "parent_whitelist_rejects_unavailable_parent": True,
        "unavailable_parent_error": unavailable,
        "folds": folds,
        "fold_training": "未执行",
        "threshold_fitting": "未执行",
    }


def _synthetic_oof_calibration_contract() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    parent_index = 0
    for true_id in range(4):
        for repeat in range(4):
            dbb_pred = true_id if true_id < 3 else repeat % 3
            records.append({
                "parent_sample_id": f"synthetic-known-{parent_index:04d}",
                "fold_id": parent_index % 5,
                "label_name": KNOWN_BASE_LABELS[true_id],
                "true_id": true_id,
                "gate_dbb_probability": 0.92 if true_id < 3 else 0.08,
                "gate_margin": 0.84 if true_id < 3 else -0.84,
                "dbb_pred_id": dbb_pred,
                "dbb_top_probability": 0.90 if true_id < 3 else 0.40,
                "dbb_top_margin": 0.75 if true_id < 3 else 0.05,
                "cu_clutter_probability": 0.91 if true_id == 3 else 0.09,
                "cu_clutter_margin": 0.82 if true_id == 3 else -0.82,
            })
            parent_index += 1
    # The production fitter requires 1011 unique known parents. Repeat the
    # separable pattern without introducing an Unknown label.
    base = list(records)
    while len(records) < 1011:
        template = dict(base[len(records) % len(base)])
        template["parent_sample_id"] = f"synthetic-known-{len(records):04d}"
        records.append(template)
    fitted = fit_known_only_oof_boundaries(
        records,
        gate_target=0.95,
        dbb_target=0.95,
        cu_target=0.90,
    )
    if (
        fitted["unknown_threshold_fit_count"] != 0
        or fitted["validation_used_for_threshold_fitting"] is not False
        or fitted["test_used"] is not False
        or fitted["combined_known_action"]["known_parent_count"] != 1011
    ):
        raise AssertionError("V14 synthetic OOF calibration violated known-only provenance")
    contaminated = [dict(value) for value in records]
    contaminated[0]["true_id"] = 4
    contaminated[0]["label_name"] = UNKNOWN_BASE_LABEL
    error = _expect_failure(
        lambda: fit_known_only_oof_boundaries(
            contaminated,
            gate_target=0.95,
            dbb_target=0.95,
            cu_target=0.90,
        ),
        "known train OOF",
    )
    return {
        "known_parent_count": fitted["known_parent_count"],
        "unknown_threshold_fit_count": fitted["unknown_threshold_fit_count"],
        "validation_used_for_threshold_fitting": False,
        "test_used": False,
        "gate_all_targets_passed": fitted["gate"]["all_precision_recall_targets_passed"],
        "dbb_all_targets_passed": fitted["dbb_knownness"]["all_precision_recall_targets_passed"],
        "clutter_all_targets_passed": fitted["clutter_acceptance"]["all_precision_recall_targets_passed"],
        "contaminated_unknown_rejected": True,
        "contaminated_error": error,
    }


def _run_legacy_contract(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_root = root / "config" / "RADAR-LN3"
    configs = {
        name: load_task_config(config_root / filename)
        for name, filename in CONFIG_NAMES.items()
    }
    validate_cascade_configs(configs)
    if any(config.schema_version != STRICT_OPEN_SET_SCHEMA_VERSION for config in configs.values()):
        raise AssertionError("focused test loaded a non-V14 config")
    contracts = {
        name: verify_task_data_contract(root, config)
        for name, config in configs.items()
    }
    gate = configs["dbb_cu_gate"]
    train_dataset = build_cascade_dataset(
        root,
        gate,
        split_name=gate.data.train_split_name,
        label_scope="known_only",
    )
    val_known = build_cascade_dataset(
        root,
        gate,
        split_name=gate.data.validation_split_name,
        label_scope="known_only",
    )
    val_audit = build_cascade_dataset(
        root,
        gate,
        split_name=gate.data.validation_split_name,
        label_scope="development_open_set_audit",
    )
    train_labels = set(train_dataset.frame["label_name"].astype(str))
    val_known_labels = set(val_known.frame["label_name"].astype(str))
    val_audit_labels = set(val_audit.frame["label_name"].astype(str))
    if (
        len(train_dataset) != 1011
        or len(val_known) != 217
        or len(val_audit) != 232
        or train_labels != set(KNOWN_BASE_LABELS)
        or val_known_labels != set(KNOWN_BASE_LABELS)
        or val_audit_labels != set((*KNOWN_BASE_LABELS, UNKNOWN_BASE_LABEL))
    ):
        raise AssertionError("V14 Dataset scopes do not match frozen populations")
    train_audit_error = _expect_failure(
        lambda: build_cascade_dataset(
            root,
            gate,
            split_name=gate.data.train_split_name,
            label_scope="development_open_set_audit",
        ),
        "train",
    )
    oof_path = root / "data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json"
    oof = json.loads(oof_path.read_text(encoding="utf-8"))
    assignments = list(oof["assignments"])
    if (
        int(oof["known_parent_count"]) != 1011
        or int(oof["unknown_oof_parent_count"]) != 0
        or len(assignments) != 1011
        or len({row["parent_sample_id"] for row in assignments}) != 1011
        or any(row["label_name"] == UNKNOWN_BASE_LABEL for row in assignments)
        or oof["execution_status"]["fold_training"] != "未执行"
    ):
        raise AssertionError("V14 OOF manifest violates known-only parent grouping")
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V14-strict-open-set-focused-contract-v1",
        "data_contract_hashes": contracts["dbb_cu_gate"]["hashes"],
        "dataset": {
            "train_known_parent_count": len(train_dataset),
            "validation_known_parent_count": len(val_known),
            "validation_development_audit_parent_count": len(val_audit),
            "train_unknown_parent_count": 0,
            "validation_known_unknown_parent_count": 0,
            "validation_development_unknown_parent_count": 15,
            "train_development_audit_scope_rejected": True,
            "train_development_audit_error": train_audit_error,
        },
        "loss": _loss_contract(),
        "action": _action_contract(),
        "selection": _selection_contract(gate),
        "calibration": _calibration_loader_contract(root, gate),
        "checkpoint": _checkpoint_contract(root, gate),
        "oof_fold_contract": _oof_fold_contract(root, configs),
        "synthetic_oof_calibration": _synthetic_oof_calibration_contract(),
        "oof_manifest": {
            "path": str(oof_path),
            "known_parent_count": oof["known_parent_count"],
            "unknown_source_parent_count": oof["unknown_source_parent_count"],
            "unknown_oof_parent_count": oof["unknown_oof_parent_count"],
            "fold_count": oof["fold_count"],
            "every_known_parent_exactly_one_held_out_fold": oof[
                "every_known_parent_exactly_one_held_out_fold"
            ],
            "fold_training": "未执行",
            "threshold_fitting": "未执行",
        },
        "evidence_names": list(STRICT_OPEN_SET_EVIDENCE_NAMES),
        "unknown_is_probability_output": False,
        "formal_training": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    output = root / "src-v3/smoke/RADAR-LN3-V14-strict-open-set-focused-contract-result.json"
    result["result_path"] = str(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run(project_root: Path) -> dict[str, Any]:
    """Verify that stale V14 product topology is sealed after route repair."""

    root = project_root.resolve()
    config_root = root / "config" / "RADAR-LN3"
    configs = {
        name: load_task_config(config_root / filename)
        for name, filename in CONFIG_NAMES.items()
    }
    validate_cascade_configs(configs)
    legacy_contract_errors = {
        name: _expect_failure(
            lambda config=config: verify_task_data_contract(root, config),
            "action contract",
        )
        for name, config in configs.items()
    }
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V14-strict-open-set-route-repair-contract-v1",
        "legacy_contract_status": "incompatible_and_fail_closed",
        "legacy_contract_errors": legacy_contract_errors,
        "action": _action_contract(),
        "historical_v14_artifacts_modified": False,
        "historical_v14_oof_reexecuted": False,
        "historical_v14_calibration_reexecuted": False,
        "formal_training": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    output = root / "src-v3/smoke/RADAR-LN3-V14-strict-open-set-route-repair-contract-result.json"
    result["result_path"] = str(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the V14 strict-open-set fail-closed contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
