from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from radar_ln3.cascade import (
    EXPECTED_TASKS,
    OnlineLiquidSoftCascade,
    build_cascade,
    build_cascade_dataset,
    evaluate_hierarchical_cascade,
    load_cascade,
    verify_cascade_data_contracts,
)
from radar_ln3.data import PointSequence, collate_point_sequences
from radar_ln3.task_models import (
    LiquidTaskClassifier,
    load_task_checkpoint,
    save_task_checkpoint,
    seed_everything,
)
from radar_ln3.rejection import (
    CU_CLUTTER_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    HierarchicalRejectionCalibration,
    HierarchicalRejectionPolicy,
)
from radar_ln3.stable import HierarchicalStableStateMachine
from train.cascade import (
    _joint_payload,
    cascade_forward_batch,
    cascade_training_objective,
    load_hierarchical_configs,
    load_joint_checkpoint,
    save_joint_checkpoint,
)
from train.tasks import train_task


CONFIG_NAMES = {
    "dbb_cu_gate": "RADAR-LN3-V13-cq08-liquid-hierarchical-dbb-cu-gate-config.json",
    "dbb_specialist": "RADAR-LN3-V13-cq08-liquid-hierarchical-dbb-specialist-config.json",
    "cu_specialist": "RADAR-LN3-V13-cq08-liquid-hierarchical-cu-specialist-config.json",
}


class _PointList(Dataset[PointSequence]):
    def __init__(self, items: list[PointSequence]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> PointSequence:
        return self.items[int(index)]


def _prefix(item: PointSequence, length: int = 3) -> PointSequence:
    selected = min(int(length), int(item.event_count))
    if selected < 1:
        raise ValueError("bounded V13 Smoke received an empty parent")
    sequence_fields = {
        "tr_features",
        "tr_missing_mask",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_dt_reason",
        "tr_motion_features",
        "tr_motion_valid_mask",
        "rd_frames",
        "rd_frame_mask",
        "rd_quality",
        "rd_physical_features",
        "rd_physical_valid",
    }
    updates: dict[str, Any] = {"event_count": selected}
    for field in fields(item):
        if field.name in sequence_fields:
            updates[field.name] = getattr(item, field.name)[:selected].clone()
    return replace(item, **updates)


def _shortest_by_label(dataset) -> dict[str, PointSequence]:
    selected: dict[str, tuple[int, int]] = {}
    for index, parent_id in enumerate(dataset.parent_ids):
        group = dataset._groups[str(parent_id)]
        label = str(group.iloc[0]["label_name"])
        length = min(len(group), dataset.max_sequence_length)
        if label not in selected or length < selected[label][0]:
            selected[label] = (length, index)
    expected = {
        "DroneTarget",
        "BirdTarget",
        "BalloonTarget",
        "ClutterTarget",
        "UnknownTarget",
    }
    if set(selected) != expected:
        raise ValueError(f"V13 Smoke did not find all five train labels: {sorted(selected)}")
    return {label: _prefix(dataset[index]) for label, (_, index) in selected.items()}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _model_change(before: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    return max(
        float((value.detach().cpu() - before[name]).abs().max().item())
        for name, value in model.state_dict().items()
        if torch.is_floating_point(value)
    )


def _output_difference(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    keys = (
        "sequence_probabilities",
        "gate_sequence_logits",
        "dbb_sequence_logits",
        "cu_sequence_logits",
    )
    return max(float((left[key] - right[key]).abs().max().item()) for key in keys)


def run(project_root: Path, *, device_name: str) -> dict[str, Any]:
    project_root = project_root.resolve()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Smoke requested but CUDA is unavailable")
    device = torch.device(device_name)
    config_dir = project_root / "config" / "RADAR-LN3"
    config_paths = {name: config_dir / filename for name, filename in CONFIG_NAMES.items()}
    configs = load_hierarchical_configs(config_paths)
    contracts = verify_cascade_data_contracts(project_root, configs)
    gate_config = configs["dbb_cu_gate"]
    formal_artifact = project_root / str(gate_config.train.joint_artifact_dir)
    if formal_artifact.exists():
        raise FileExistsError(
            "bounded V13 Smoke must run before formal training and must not mix with a formal artifact: "
            f"{formal_artifact}"
        )
    dataset = build_cascade_dataset(project_root, gate_config, split_name=gate_config.data.train_split_name)
    examples = _shortest_by_label(dataset)
    train_items = [examples["BirdTarget"], examples["UnknownTarget"]]
    train_batch_cpu = collate_point_sequences(train_items)
    train_batch = _to_device(train_batch_cpu, device)

    seed_everything(gate_config.train.seed)
    model = build_cascade(configs, device=device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=gate_config.train.learning_rate)
    before = {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in {
            "dbb_cu_gate": model.gate,
            "dbb_specialist": model.dbb_specialist,
            "cu_specialist": model.cu_specialist,
        }.items()
    }
    output = cascade_forward_batch(model, train_batch)
    objective = cascade_training_objective(output, train_batch, configs)
    optimizer.zero_grad(set_to_none=True)
    objective["loss"].backward()
    gradients = {
        "dbb_cu_gate": float(torch.nn.utils.clip_grad_norm_(model.gate.parameters(), 100.0).item()),
        "dbb_specialist": float(torch.nn.utils.clip_grad_norm_(model.dbb_specialist.parameters(), 100.0).item()),
        "cu_specialist": float(torch.nn.utils.clip_grad_norm_(model.cu_specialist.parameters(), 100.0).item()),
    }
    if any(not torch.isfinite(torch.tensor(value)).item() or value <= 0.0 for value in gradients.values()):
        raise AssertionError(f"V13 Smoke has a missing/non-finite task gradient: {gradients}")
    optimizer.step()
    changes = {
        "dbb_cu_gate": _model_change(before["dbb_cu_gate"], model.gate),
        "dbb_specialist": _model_change(before["dbb_specialist"], model.dbb_specialist),
        "cu_specialist": _model_change(before["cu_specialist"], model.cu_specialist),
    }
    if any(value <= 0.0 for value in changes.values()):
        raise AssertionError(f"V13 Smoke did not update all three task models: {changes}")
    model.eval()
    with torch.inference_mode():
        updated_output = cascade_forward_batch(model, train_batch)

    smoke_dir = project_root / "src-v3" / "smoke" / f"RADAR-LN3-V13-hierarchical-cascade-smoke-{device_name}"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "metric": gate_config.train.checkpoint_selection_metric,
        "ranking_key": [0.0] * 9,
        "soft_posterior_used_for_selection": False,
        "smoke_only": True,
    }
    checkpoint_set_id = "0" * 32 if device_name == "cpu" else "1" * 32
    checkpoint_metrics = {
        "checkpoint_selection": selection,
        "checkpoint_set_id": checkpoint_set_id,
        "smoke_only": True,
    }
    task_paths: dict[str, Path] = {}
    modules = {
        "dbb_cu_gate": model.gate,
        "dbb_specialist": model.dbb_specialist,
        "cu_specialist": model.cu_specialist,
    }
    for name in EXPECTED_TASKS:
        path = smoke_dir / f"{name}.pt"
        save_task_checkpoint(
            path,
            modules[name],
            None,
            config=configs[name],
            epoch=1,
            global_step=1,
            metrics=checkpoint_metrics,
        )
        task_paths[name] = path
    restored, payloads = load_cascade(configs, task_paths, device=device)
    with torch.inference_mode():
        restored_output = cascade_forward_batch(restored, train_batch)
    task_checkpoint_difference = _output_difference(updated_output, restored_output)
    if task_checkpoint_difference != 0.0:
        raise AssertionError(f"V13 task checkpoint round trip changed output: {task_checkpoint_difference}")
    changed_gate_config = replace(
        gate_config,
        train=replace(gate_config.train, learning_rate=gate_config.train.learning_rate * 2.0),
    )
    changed_gate_model = LiquidTaskClassifier(changed_gate_config).to(device)
    try:
        load_task_checkpoint(
            task_paths["dbb_cu_gate"],
            changed_gate_model,
            expected_schema=changed_gate_config.train.checkpoint_schema,
            map_location=device,
        )
    except ValueError as error:
        config_identity_mismatch_rejected = "configuration identity" in str(error)
    else:
        config_identity_mismatch_rejected = False
    if not config_identity_mismatch_rejected:
        raise AssertionError("V13 task checkpoint accepted an edited same-name configuration")
    invalid_selection_path = smoke_dir / "invalid_missing_product_selection.pt"
    invalid_selection_payload = dict(payloads["dbb_cu_gate"])
    invalid_selection_payload["metrics"] = {"smoke_only": True}
    torch.save(invalid_selection_payload, invalid_selection_path)
    invalid_paths = dict(task_paths)
    invalid_paths["dbb_cu_gate"] = invalid_selection_path
    try:
        load_cascade(configs, invalid_paths, device=device)
    except ValueError as error:
        missing_product_selection_rejected = "jointly selected" in str(error)
    else:
        missing_product_selection_rejected = False
    invalid_selection_path.unlink(missing_ok=True)
    if not missing_product_selection_rejected:
        raise AssertionError("V13 cascade accepted a checkpoint without product selection metadata")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(gate_config.train.dataloader_seed))
    joint_path = smoke_dir / "joint_checkpoint.pt"
    payload = _joint_payload(
        model,
        optimizer,
        configs,
        generator,
        epoch=1,
        global_step=1,
        best_epoch=1,
        best_selection=selection,
        history=[{"epoch": 1, "smoke_only": True}],
        checkpoint_set_id=checkpoint_set_id,
    )
    save_joint_checkpoint(joint_path, payload)
    seed_everything(gate_config.train.seed + 100)
    joint_restored = build_cascade(configs, device=device)
    joint_optimizer = torch.optim.AdamW(joint_restored.parameters(), lr=gate_config.train.learning_rate)
    joint_generator = torch.Generator(device="cpu")
    joint_state = load_joint_checkpoint(
        joint_path,
        joint_restored,
        joint_optimizer,
        configs,
        joint_generator,
    )
    joint_restored.eval()
    with torch.inference_mode():
        joint_output = cascade_forward_batch(joint_restored, train_batch)
    joint_checkpoint_difference = _output_difference(updated_output, joint_output)
    if (
        joint_checkpoint_difference != 0.0
        or int(joint_state["next_epoch"]) != 2
        or joint_state["checkpoint_set_id"] != checkpoint_set_id
    ):
        raise AssertionError("V13 joint checkpoint did not restore model/output/run state exactly")

    evaluation_items = [
        examples["DroneTarget"],
        examples["BirdTarget"],
        examples["BalloonTarget"],
        examples["ClutterTarget"],
        examples["UnknownTarget"],
    ]
    evaluation = evaluate_hierarchical_cascade(
        joint_restored,
        _PointList(evaluation_items),
        device=device,
        batch_size=2,
        split_name="train_bounded_smoke",
    )
    if evaluation["calibrated_hierarchical_action"]["status"] != "not_evaluated":
        raise AssertionError("V13 bounded Smoke falsely emitted calibrated rejection metrics")
    if evaluation["stable_action"]["status"] != "not_evaluated_without_calibration":
        raise AssertionError("V13 bounded Smoke falsely emitted stable action metrics")

    one = collate_point_sequences([examples["BirdTarget"]])
    one_device = _to_device(one, device)
    with torch.inference_mode():
        offline = cascade_forward_batch(joint_restored, one_device)["sequence_probabilities"]
    online = OnlineLiquidSoftCascade(joint_restored, confirm_count=2)
    online_rows: list[torch.Tensor] = []
    for event_index in range(int(one["lengths"][0].item())):
        step = online.step(
            tr_features=one_device["tr_features"][:, event_index],
            tr_missing_mask=one_device["tr_missing_mask"][:, event_index],
            tr_dt_seconds=one_device["tr_dt_seconds"][:, event_index],
            tr_dt_valid=one_device["tr_dt_valid"][:, event_index],
            tr_dt_reason=one_device["tr_dt_reason"][:, event_index],
            rd_frames=one_device["rd_frames"][:, event_index],
            rd_frame_mask=one_device["rd_frame_mask"][:, event_index],
            rd_quality=one_device["rd_quality"][:, event_index],
        )
        if step.final_action != "NOT_CALIBRATED" or step.stable_confirmed:
            raise AssertionError("uncalibrated V13 online path emitted a product/stable action")
        if (
            step.window_probabilities is not None
            or step.window_label_id is not None
            or step.window_label_name is not None
            or step.window_confidence is not None
        ):
            raise AssertionError("V13 exposed its audit posterior through a legacy product-label field")
        if step.soft_posterior_audit_probabilities is None:
            raise AssertionError("V13 online output omitted the explicitly named soft-posterior audit")
        if any(
            value is None
            for value in (
                step.gate_dbb_probability,
                step.gate_margin,
                step.specialist_label_id,
                step.specialist_confidence,
                step.specialist_margin,
                step.valid_point_count,
                step.track_quality,
                step.gap_fraction,
            )
        ):
            raise AssertionError("V13 online action omitted probability, margin or quality evidence")
        online_rows.append(step.soft_posterior_audit_probabilities)
    online_probabilities = torch.stack(online_rows, dim=1)
    online_difference = float((offline - online_probabilities).abs().max().item())
    if not torch.allclose(offline, online_probabilities, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(f"V13 offline/online probability mismatch: {online_difference}")

    synthetic_calibration = HierarchicalRejectionCalibration(
        schema_version=HIERARCHICAL_REJECTION_CALIBRATION_SCHEMA,
        source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
        gate_score_name=GATE_DBB_PROBABILITY_SCORE,
        cu_score_name=CU_CLUTTER_PROBABILITY_SCORE,
        gate_accept_dbb_threshold=0.0 + 1.0e-6,
        gate_accept_dbb_margin=0.0,
        cu_accept_clutter_threshold=0.0 + 1.0e-6,
        cu_accept_clutter_margin=0.0,
        min_valid_points=2,
        min_track_quality=0.0,
        max_gap_fraction=1.0,
    )
    synthetic_policy = HierarchicalRejectionPolicy(synthetic_calibration, enabled=True)
    calibrated_evaluation = evaluate_hierarchical_cascade(
        joint_restored,
        _PointList(evaluation_items),
        device=device,
        batch_size=2,
        split_name="train_bounded_smoke_synthetic_calibration",
        rejection_policy=synthetic_policy,
        stable_confirm_count=2,
    )
    calibrated_summary = calibrated_evaluation["calibrated_hierarchical_action"]
    if calibrated_summary["status"] != "evaluated_with_train_parent_grouped_oof_calibration":
        raise AssertionError("V13 calibrated evaluator did not execute its action path")
    if (
        calibrated_summary["accepted_parent_count"]
        + calibrated_summary["rejected_parent_count"]
        + calibrated_summary["pending_parent_count"]
        != calibrated_summary["parent_count"]
    ):
        raise AssertionError("V13 calibrated evaluator action counts do not partition parents")
    required_action_evidence = {
        "gate_dbb_probability",
        "gate_margin",
        "specialist_label_id",
        "specialist_confidence",
        "specialist_margin",
        "valid_point_count",
        "track_quality",
        "gap_fraction",
        "trigger_reasons",
    }
    if any(
        not required_action_evidence.issubset(record)
        for record in calibrated_evaluation["records"]["calibrated_window_action"]
    ):
        raise AssertionError("V13 calibrated action records omit probability, margin or quality evidence")
    if any(
        record["confidence"] is not None
        for record in calibrated_evaluation["records"]["calibrated_window_action"]
        if record["decision_state"] == "pending"
    ):
        raise AssertionError("V13 pending action exposed a misleading class/rejection confidence")
    for key in ("coverage", "action_coverage", "false_acceptance_rate", "false_rejection_rate", "selective_risk"):
        value = calibrated_summary[key]
        if value is not None and not 0.0 <= float(value) <= 1.0:
            raise AssertionError(f"V13 calibrated evaluator emitted invalid {key}: {value}")
    calibrated_online = OnlineLiquidSoftCascade(
        joint_restored,
        confirm_count=2,
        hierarchical_rejection_policy=synthetic_policy,
    )
    offline_stable = HierarchicalStableStateMachine(confirm_count=2)
    offline_rd_count = 0
    offline_zero_count = 0
    action_state_matches = True
    action_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        component = cascade_forward_batch(joint_restored, one_device)
    gate_probs = torch.softmax(component["gate_sequence_logits"], dim=-1)
    dbb_probs = torch.softmax(component["dbb_sequence_logits"], dim=-1)
    cu_probs = torch.softmax(component["cu_sequence_logits"], dim=-1)
    for event_index in range(int(one["lengths"][0].item())):
        valid_points = event_index + 1
        offline_rd_count += int(float(one_device["rd_quality"][0, event_index, 0].item()) >= 0.5)
        offline_zero_count += int(int(one_device["tr_dt_reason"][0, event_index].item()) == 2)
        offline_action = synthetic_policy.decide(
            gate_probs[0, event_index].cpu(),
            dbb_probs[0, event_index].cpu(),
            cu_probs[0, event_index].cpu(),
            valid_point_count=valid_points,
            track_quality=offline_rd_count / valid_points,
            gap_fraction=min(1.0, offline_zero_count / max(1, valid_points - 1)),
        )
        offline_stable_result = offline_stable.update(offline_action)
        online_action = calibrated_online.step(
            tr_features=one_device["tr_features"][:, event_index],
            tr_missing_mask=one_device["tr_missing_mask"][:, event_index],
            tr_dt_seconds=one_device["tr_dt_seconds"][:, event_index],
            tr_dt_valid=one_device["tr_dt_valid"][:, event_index],
            tr_dt_reason=one_device["tr_dt_reason"][:, event_index],
            rd_frames=one_device["rd_frames"][:, event_index],
            rd_frame_mask=one_device["rd_frame_mask"][:, event_index],
            rd_quality=one_device["rd_quality"][:, event_index],
        )
        row_matches = (
            online_action.final_action == offline_action.final_action
            and online_action.accepted_label_id == offline_action.accepted_label_id
            and online_action.rejected_to_unknown == offline_action.rejected_to_unknown
            and online_action.stable_state_name == offline_stable_result.state_name
            and online_action.stable_confirmed == offline_stable_result.stable_confirmed
            and online_action.stable_label_id == offline_stable_result.stable_label_id
            and online_action.stable_rejected_to_unknown == offline_stable_result.stable_rejected_to_unknown
        )
        action_state_matches = action_state_matches and row_matches
        action_rows.append({
            "event_index": event_index,
            "offline_final_action": offline_action.final_action,
            "online_final_action": online_action.final_action,
            "offline_stable_state": offline_stable_result.state_name,
            "online_stable_state": online_action.stable_state_name,
            "matches": row_matches,
        })
    if not action_state_matches or action_rows[0]["offline_final_action"] not in {"DBB_PENDING", "CU_PENDING"}:
        raise AssertionError("V13 calibrated offline/online action state machines diverged or skipped pending")

    try:
        train_task(project_root, config_paths["dbb_cu_gate"], max_steps=1)
    except RuntimeError as error:
        legacy_task_entry_rejected = "joint three-checkpoint" in str(error)
    else:
        legacy_task_entry_rejected = False
    if not legacy_task_entry_rejected:
        raise AssertionError("legacy single-task trainer accepted a V13 config")

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V13-hierarchical-cascade-smoke-v1",
        "device": str(device),
        "data_contract_hashes": contracts["dbb_cu_gate"]["hashes"],
        "full_train_parent_count": len(dataset),
        "bounded_train_parent_ids": train_batch_cpu["parent_sample_ids"],
        "bounded_train_labels": train_batch_cpu["label_names"],
        "bounded_evaluation_labels": [item.label_name for item in evaluation_items],
        "bounded_event_limit_per_parent": 3,
        "batch_shapes": {
            key: list(value.shape)
            for key, value in train_batch_cpu.items()
            if isinstance(value, torch.Tensor)
        },
        "objective": {
            key: float(value.item())
            for key, value in objective.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        },
        "gradient_norms": gradients,
        "max_parameter_changes": changes,
        "task_checkpoint_paths": {name: str(path) for name, path in task_paths.items()},
        "task_checkpoint_epochs": {name: int(payloads[name]["epoch"]) for name in EXPECTED_TASKS},
        "task_checkpoint_round_trip_max_difference": task_checkpoint_difference,
        "edited_same_name_config_checkpoint_rejected": config_identity_mismatch_rejected,
        "missing_product_selection_checkpoint_rejected": missing_product_selection_rejected,
        "joint_checkpoint_path": str(joint_path),
        "joint_checkpoint_round_trip_max_difference": joint_checkpoint_difference,
        "joint_checkpoint_next_epoch": int(joint_state["next_epoch"]),
        "offline_online_probability_max_difference": online_difference,
        "calibrated_offline_online_action_state_matches": action_state_matches,
        "calibrated_action_state_trace": action_rows,
        "synthetic_calibration_only": True,
        "calibrated_evaluator_status": calibrated_summary["status"],
        "calibrated_evaluator_action_partition": {
            "accepted": calibrated_summary["accepted_parent_count"],
            "rejected": calibrated_summary["rejected_parent_count"],
            "pending": calibrated_summary["pending_parent_count"],
            "total": calibrated_summary["parent_count"],
        },
        "calibrated_evaluator_rates_are_synthetic_only": {
            key: calibrated_summary[key]
            for key in ("coverage", "action_coverage", "false_acceptance_rate", "false_rejection_rate", "selective_risk")
        },
        "diagnostic_evaluation_parent_count": evaluation["parent_count"],
        "diagnostic_only_not_rejection": evaluation["diagnostic_hierarchical_argmax"]["is_validated_rejection"] is False,
        "legacy_window_classification_fields_disabled": True,
        "soft_posterior_is_explicit_audit_only": True,
        "hierarchical_action_probability_margin_quality_fields_present": True,
        "legacy_single_task_entry_rejected": legacy_task_entry_rejected,
        "formal_artifact_path": str(formal_artifact),
        "formal_artifact_created": formal_artifact.exists(),
        "formal_training": "未执行",
        "oof_calibration": "未执行",
        "calibrated_rejection": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    result_path = smoke_dir / "result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded real-CQ08 V13 hierarchical cascade Smoke.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.project_root.resolve(), device_name=args.device)


if __name__ == "__main__":
    main()
