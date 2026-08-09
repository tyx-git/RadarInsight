from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)
from radar_ln3.losses import hierarchical_final_state_nll
from radar_ln3.model import CfCLiquidCausalEncoder
from radar_ln3.rejection import (
    CU_CLUTTER_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    HierarchicalRejectionCalibration,
    HierarchicalRejectionPolicy,
    load_hierarchical_rejection_calibration,
    validate_hierarchical_calibration_deployment,
)
from radar_ln3.stable import HierarchicalStableStateMachine
from radar_ln3.cascade import summarize_calibrated_actions


def _loss_contract() -> dict[str, object]:
    torch.manual_seed(20260809)
    padding = torch.tensor([[False, False, False], [False, False, True]])
    labels = torch.tensor([0, 4])
    gate = torch.randn(2, 3, 2, requires_grad=True)
    dbb = torch.randn(2, 3, 3, requires_grad=True)
    cu = torch.randn(2, 3, 2, requires_grad=True)
    objective = hierarchical_final_state_nll(
        gate,
        dbb,
        cu,
        labels,
        padding,
        prefix_auxiliary_weight=0.0,
    )
    objective["loss"].backward()
    if gate.grad is None or dbb.grad is None or cu.grad is None:
        raise AssertionError("hierarchical final objective did not produce gradients")
    if float(gate.grad[:, :1].abs().max().item()) != 0.0:
        raise AssertionError("final-state objective supervised an early Gate prefix")
    if float(dbb.grad[1].abs().max().item()) != 0.0:
        raise AssertionError("Unknown parent produced a DBB specialist gradient")
    if float(cu.grad[0].abs().max().item()) != 0.0:
        raise AssertionError("DBB parent produced a CU specialist gradient")
    if float(dbb.grad[0, 2].abs().sum().item()) <= 0.0:
        raise AssertionError("DBB final state did not update the DBB specialist")
    if float(cu.grad[1, 1].abs().sum().item()) <= 0.0:
        raise AssertionError("Unknown final state did not update the CU reject branch")

    gate_prefix = torch.randn(1, 4, 2, requires_grad=True)
    dbb_prefix = torch.randn(1, 4, 3, requires_grad=True)
    cu_prefix = torch.randn(1, 4, 2, requires_grad=True)
    prefix = hierarchical_final_state_nll(
        gate_prefix,
        dbb_prefix,
        cu_prefix,
        torch.tensor([1]),
        torch.zeros(1, 4, dtype=torch.bool),
        prefix_auxiliary_weight=0.10,
        prefix_auxiliary_min_events=3,
    )
    prefix["loss"].backward()
    if float(gate_prefix.grad[0, :2].abs().max().item()) != 0.0:
        raise AssertionError("prefix auxiliary supervised events below minimum evidence")
    if float(gate_prefix.grad[0, 2].abs().sum().item()) <= 0.0:
        raise AssertionError("eligible prefix did not receive auxiliary supervision")
    return {
        "final_loss": float(objective["final_hierarchical_nll"].item()),
        "early_prefix_gradient_max": float(gate.grad[:, :1].abs().max().item()),
        "unknown_to_dbb_gradient_max": float(dbb.grad[1].abs().max().item()),
        "dbb_to_cu_gradient_max": float(cu.grad[0].abs().max().item()),
        "prefix_min_events": 3,
        "events_before_min_gradient_max": float(gate_prefix.grad[0, :2].abs().max().item()),
    }


def _dt_contract() -> dict[str, object]:
    torch.manual_seed(20260809)
    encoder = CfCLiquidCausalEncoder(
        5,
        7,
        invalid_dt_update_mode="bounded_observation",
        invalid_dt_observation_gain=0.25,
    ).eval()
    events = torch.randn(1, 3, 5)
    dt = torch.tensor([[0.0, 0.0, 1.0]])
    valid = torch.tensor([[False, False, True]])
    reasons = torch.tensor(
        [[DT_REASON_FIRST_EVENT, DT_REASON_ZERO_DELTA, DT_REASON_POSITIVE_DELTA]],
        dtype=torch.long,
    )
    padding = torch.zeros(1, 3, dtype=torch.bool)
    with torch.inference_mode():
        batch = encoder(events, dt, padding, valid, dt_reason=reasons)
        state = None
        online_rows = []
        for index in range(3):
            state, _, _ = encoder.step(
                events[:, index],
                dt[:, index],
                state,
                valid[:, index],
                reasons[:, index],
            )
            online_rows.append(encoder.readout(state))
    online = torch.stack(online_rows, dim=1)
    difference = float((batch.sequence - online).abs().max().item())
    if difference > 1.0e-6:
        raise AssertionError(f"reason-aware liquid batch/online mismatch: {difference}")
    same_event = torch.randn(1, 5)
    initial = torch.zeros(1, 7)
    with torch.inference_mode():
        first_state, _, _ = encoder.step(
            same_event,
            torch.zeros(1),
            None,
            torch.zeros(1, dtype=torch.bool),
            torch.full((1,), DT_REASON_FIRST_EVENT, dtype=torch.long),
        )
        zero_state, _, _ = encoder.step(
            same_event,
            torch.zeros(1),
            initial,
            torch.zeros(1, dtype=torch.bool),
            torch.full((1,), DT_REASON_ZERO_DELTA, dtype=torch.long),
        )
    semantic_difference = float((first_state - zero_state).abs().max().item())
    if semantic_difference <= 0.0:
        raise AssertionError("first event and same-second observation still use the same dynamics")
    try:
        encoder(
            events,
            dt,
            padding,
            valid,
            dt_reason=torch.tensor(
                [[DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA, DT_REASON_POSITIVE_DELTA]]
            ),
        )
    except ValueError:
        invalid_pair_rejected = True
    else:
        raise AssertionError("liquid accepted an inconsistent dt reason/validity pair")
    try:
        encoder(
            events,
            dt,
            padding,
            valid,
            dt_reason=torch.tensor(
                [[DT_REASON_FIRST_EVENT, DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA]]
            ),
        )
    except ValueError:
        repeated_first_reason_rejected = True
    else:
        raise AssertionError("liquid accepted a repeated first-event reason inside one sequence")
    try:
        encoder.step(
            events[:, 1],
            dt[:, 1],
            None,
            valid[:, 1],
            reasons[:, 1],
        )
    except ValueError:
        fresh_online_nonfirst_rejected = True
    else:
        raise AssertionError("fresh online liquid state accepted a non-first reason")
    try:
        encoder.step(
            events[:, 0],
            dt[:, 0],
            torch.zeros(1, 7),
            valid[:, 0],
            reasons[:, 0],
        )
    except ValueError:
        continued_online_first_rejected = True
    else:
        raise AssertionError("continued online liquid state accepted another first-event reason")
    padded = torch.tensor([[False, False, True]])
    padded_reason = torch.tensor(
        [[DT_REASON_FIRST_EVENT, DT_REASON_ZERO_DELTA, DT_REASON_PADDING]], dtype=torch.long
    )
    encoder(events, dt, padded, torch.tensor([[False, False, False]]), dt_reason=padded_reason)
    return {
        "batch_online_max_difference": difference,
        "first_vs_zero_delta_state_max_difference": semantic_difference,
        "invalid_reason_valid_pair_rejected": invalid_pair_rejected,
        "repeated_first_reason_rejected": repeated_first_reason_rejected,
        "fresh_online_nonfirst_reason_rejected": fresh_online_nonfirst_rejected,
        "continued_online_first_reason_rejected": continued_online_first_rejected,
        "padding_reason_is_separate": True,
    }


def _calibration() -> HierarchicalRejectionCalibration:
    return HierarchicalRejectionCalibration(
        schema_version=HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA,
        source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
        gate_score_name=GATE_DBB_PROBABILITY_SCORE,
        cu_score_name=CU_CLUTTER_PROBABILITY_SCORE,
        gate_accept_dbb_threshold=0.70,
        gate_accept_dbb_margin=0.20,
        cu_accept_clutter_threshold=0.70,
        cu_accept_clutter_margin=0.20,
        min_valid_points=2,
        min_track_quality=0.20,
        max_gap_fraction=0.80,
    )


def _action_contract() -> dict[str, object]:
    gate_dbb = torch.tensor([0.90, 0.10])
    gate_cu = torch.tensor([0.20, 0.80])
    dbb = torch.tensor([0.05, 0.90, 0.05])
    clutter = torch.tensor([0.90, 0.10])
    unknown = torch.tensor([0.20, 0.80])
    uncalibrated = HierarchicalRejectionPolicy().decide(
        gate_dbb,
        dbb,
        clutter,
        valid_point_count=2,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    if uncalibrated.accepted_label_id is not None or uncalibrated.rejected_to_unknown is not None:
        raise AssertionError("uncalibrated policy emitted a product action")
    try:
        HierarchicalRejectionCalibration(
            schema_version=HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA,
            source="validation",
            gate_score_name=GATE_DBB_PROBABILITY_SCORE,
            cu_score_name=CU_CLUTTER_PROBABILITY_SCORE,
            gate_accept_dbb_threshold=0.7,
            gate_accept_dbb_margin=0.2,
            cu_accept_clutter_threshold=0.7,
            cu_accept_clutter_margin=0.2,
        )
    except ValueError:
        validation_calibration_rejected = True
    else:
        raise AssertionError("hierarchical policy accepted validation-derived thresholds")
    policy = HierarchicalRejectionPolicy(_calibration(), enabled=True)
    pending = policy.decide(
        gate_dbb,
        dbb,
        clutter,
        valid_point_count=1,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    if pending.final_action != "DBB_PENDING" or pending.rejected_to_unknown is not None:
        raise AssertionError("minimum-evidence event did not remain pending")
    accepted_dbb = policy.decide(
        gate_dbb,
        dbb,
        clutter,
        valid_point_count=2,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    accepted_clutter = policy.decide(
        gate_cu,
        dbb,
        clutter,
        valid_point_count=2,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    rejected_unknown = policy.decide(
        gate_cu,
        dbb,
        unknown,
        valid_point_count=2,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    if accepted_dbb.accepted_label_name != "BirdTarget" or accepted_dbb.final_action != "DBB_ACCEPTED":
        raise AssertionError("DBB hierarchical action is incorrect")
    if accepted_clutter.accepted_label_name != "ClutterTarget" or accepted_clutter.final_action != "CLUTTER_ACCEPTED":
        raise AssertionError("Clutter hierarchical action is incorrect")
    if rejected_unknown.accepted_label_id is not None or rejected_unknown.final_action != "UNKNOWN_REJECTED":
        raise AssertionError("Unknown rejection action is incorrect")
    stable = HierarchicalStableStateMachine(confirm_count=2)
    pending_stable = stable.update(pending)
    if pending_stable.stable_confirmed or pending_stable.state_name != "DBB_PENDING":
        raise AssertionError("pending action incorrectly advanced stable confirmation")
    first = stable.update(accepted_dbb)
    second = stable.update(accepted_dbb)
    conflict = stable.update(rejected_unknown)
    if first.stable_confirmed or not second.stable_confirmed or second.stable_label_name != "BirdTarget":
        raise AssertionError("hierarchical stable confirmation is incorrect")
    if conflict.stable_confirmed or conflict.stable_label_id is not None or conflict.state_name != "CU_PENDING":
        raise AssertionError("rejected transition coexists with a concrete stable class")
    rejected_stable = stable.update(rejected_unknown)
    if not rejected_stable.stable_confirmed or not rejected_stable.stable_rejected_to_unknown:
        raise AssertionError("Unknown rejection did not become a stable action")
    if rejected_stable.stable_label_id is not None or rejected_stable.stable_label_name is not None:
        raise AssertionError("stable Unknown rejection was encoded as an accepted class")
    rows = [
        {"true_id": 0, "pred_id": 0, "accepted_label_id": 0, "final_action": "DBB_ACCEPTED", "decision_state": "accepted"},
        {"true_id": 1, "pred_id": 4, "accepted_label_id": None, "final_action": "UNKNOWN_REJECTED", "decision_state": "rejected"},
        {"true_id": 4, "pred_id": 4, "accepted_label_id": None, "final_action": "UNKNOWN_REJECTED", "decision_state": "rejected"},
        {"true_id": 4, "pred_id": 1, "accepted_label_id": 1, "final_action": "DBB_ACCEPTED", "decision_state": "accepted"},
    ]
    accepted = [row for row in rows if row["accepted_label_id"] is not None]
    metrics = summarize_calibrated_actions(
        rows,
        accepted,
        split_name="synthetic",
        calibration_source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
    )
    expected = {
        "coverage": 0.5,
        "false_acceptance_rate": 0.5,
        "false_rejection_rate": 0.5,
        "unknown_rejection_recall": 0.5,
    }
    for name, value in expected.items():
        if abs(float(metrics[name]) - value) > 1.0e-12:
            raise AssertionError(f"hierarchical metric denominator mismatch: {name}={metrics[name]}")
    if metrics["accepted_metrics"]["macro_denominator_classes"] != [
        "DroneTarget",
        "BirdTarget",
        "BalloonTarget",
        "ClutterTarget",
    ]:
        raise AssertionError("accepted Macro metrics still include Unknown as an accepted class")
    if metrics["accepted_metrics"]["accepted_unknown_false_accept_count"] != 1:
        raise AssertionError("accepted metrics dropped the dangerous Unknown false accept")
    bird_precision_rows = rows + [{
        "true_id": 1,
        "pred_id": 1,
        "accepted_label_id": 1,
        "final_action": "DBB_ACCEPTED",
        "decision_state": "accepted",
    }]
    bird_precision_metrics = summarize_calibrated_actions(
        bird_precision_rows,
        [row for row in bird_precision_rows if row["accepted_label_id"] is not None],
        split_name="synthetic_bird_precision",
        calibration_source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
    )
    bird_row = next(
        row
        for row in bird_precision_metrics["drone_bird_action_metrics"]["per_class"]
        if row["class"] == "BirdTarget"
    )
    if abs(float(bird_row["precision"]) - 0.5) > 1.0e-12:
        raise AssertionError("Drone/Bird action precision dropped an Unknown-to-Bird false accept")
    if abs(float(bird_row["recall"]) - 0.5) > 1.0e-12:
        raise AssertionError("Drone/Bird action recall dropped a rejected true Bird parent")
    perfect_known_rows = [
        {
            "true_id": class_id,
            "pred_id": class_id,
            "accepted_label_id": class_id,
            "final_action": "DBB_ACCEPTED" if class_id < 3 else "CLUTTER_ACCEPTED",
            "decision_state": "accepted",
        }
        for class_id in range(4)
    ]
    perfect_known_metrics = summarize_calibrated_actions(
        perfect_known_rows,
        perfect_known_rows,
        split_name="synthetic_perfect_known",
        calibration_source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
    )
    if abs(float(perfect_known_metrics["accepted_macro_f1"]) - 1.0) > 1.0e-12:
        raise AssertionError("accepted Macro-F1 still has an artificial 0.80 ceiling")
    return {
        "uncalibrated_state": uncalibrated.decision_state,
        "minimum_evidence_state": pending.final_action,
        "validation_calibration_rejected": validation_calibration_rejected,
        "dbb_action": accepted_dbb.final_action,
        "clutter_action": accepted_clutter.final_action,
        "unknown_action": rejected_unknown.final_action,
        "conflict_state": conflict.state_name,
        "rejected_stable_has_no_class_label": rejected_stable.stable_label_id is None,
        "metric_denominator_probe": {name: metrics[name] for name in expected},
        "accepted_macro_classes": metrics["accepted_metrics"]["macro_denominator_classes"],
        "accepted_unknown_false_accept_preserved": True,
        "drone_bird_precision_recall_use_all_parent_denominators": True,
        "perfect_known_accepted_macro_f1": perfect_known_metrics["accepted_macro_f1"],
    }


def _calibration_artifact_contract(project_root: Path) -> dict[str, object]:
    smoke_dir = project_root / "src-v3" / "smoke"
    valid_path = smoke_dir / "RADAR-LN3-V13-synthetic-oof-calibration.json"
    payload = {
        "schema_version": HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA,
        "source": TRAIN_PARENT_GROUPED_OOF_SOURCE,
        "gate_score_name": GATE_DBB_PROBABILITY_SCORE,
        "cu_score_name": CU_CLUTTER_PROBABILITY_SCORE,
        "gate_accept_dbb_threshold": 0.7,
        "gate_accept_dbb_margin": 0.2,
        "cu_accept_clutter_threshold": 0.7,
        "cu_accept_clutter_margin": 0.2,
        "min_valid_points": 2,
        "min_track_quality": 0.2,
        "max_gap_fraction": 0.8,
        "provenance": {
            "fit_split": "train",
            "grouping": "parent",
            "prediction_protocol": "out_of_fold",
            "validation_used_for_threshold_fitting": False,
            "test_used": False,
            "data_contract_hashes": {
                "index_sha256": "a" * 64,
                "contract_sha256": "b" * 64,
                "stats_sha256": "c" * 64,
            },
            "deployment_checkpoint_set": {
                "task_config_identity_sha256": {
                    "dbb_cu_gate": "d" * 64,
                    "dbb_specialist": "e" * 64,
                    "cu_specialist": "f" * 64,
                },
                "checkpoint_sha256": {
                    "dbb_cu_gate": "1" * 64,
                    "dbb_specialist": "2" * 64,
                    "cu_specialist": "3" * 64,
                },
                "epoch": 7,
                "global_step": 700,
                "product_selection_sha256": "4" * 64,
            },
        },
    }
    valid_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    calibration, loaded = load_hierarchical_rejection_calibration(valid_path)
    if calibration.source != TRAIN_PARENT_GROUPED_OOF_SOURCE or loaded["provenance"]["test_used"] is not False:
        raise AssertionError("valid train-parent OOF calibration did not round trip")
    deployment = payload["provenance"]["deployment_checkpoint_set"]
    validate_hierarchical_calibration_deployment(loaded, deployment)
    mismatched_deployment = json.loads(json.dumps(deployment))
    mismatched_deployment["checkpoint_sha256"]["cu_specialist"] = "5" * 64
    try:
        validate_hierarchical_calibration_deployment(loaded, mismatched_deployment)
    except ValueError:
        checkpoint_mismatch_rejected = True
    else:
        raise AssertionError("calibration accepted a different deployment checkpoint set")
    invalid_path = smoke_dir / "RADAR-LN3-V13-synthetic-invalid-calibration.json"
    invalid = json.loads(json.dumps(payload))
    invalid["provenance"]["validation_used_for_threshold_fitting"] = True
    invalid_path.write_text(json.dumps(invalid, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        load_hierarchical_rejection_calibration(invalid_path)
    except ValueError:
        leaked_calibration_rejected = True
    else:
        raise AssertionError("calibration loader accepted validation threshold leakage")
    valid_path.unlink()
    invalid_path.unlink()
    return {
        "train_parent_grouped_oof_loaded": True,
        "validation_threshold_leakage_rejected": leaked_calibration_rejected,
        "deployment_checkpoint_binding_required": checkpoint_mismatch_rejected,
        "synthetic_artifacts_removed": not valid_path.exists() and not invalid_path.exists(),
    }


def run(project_root: Path) -> dict[str, object]:
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V13-hierarchical-focused-contract-v1",
        "loss": _loss_contract(),
        "dt_reason": _dt_contract(),
        "hierarchical_action": _action_contract(),
        "calibration_artifact": _calibration_artifact_contract(project_root),
        "formal_training": "未执行",
        "oof_calibration": "未执行；focused thresholds are synthetic interface constants only",
        "validation": "未执行",
        "test": "未执行",
    }
    path = project_root / "src-v3" / "smoke" / "RADAR-LN3-V13-hierarchical-focused-contract-result.json"
    result["result_path"] = str(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check V13 hierarchical loss, dt reason and action contracts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
