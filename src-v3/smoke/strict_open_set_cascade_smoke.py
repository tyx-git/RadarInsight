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
    evaluate_strict_open_set_cascade,
    load_cascade,
    verify_cascade_data_contracts,
)
from radar_ln3.data import PointSequence, collate_point_sequences
from radar_ln3.rejection import (
    CU_CLUTTER_PROBABILITY_SCORE,
    DBB_KNOWN_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    StrictOpenSetRejectionCalibration,
    StrictOpenSetRejectionPolicy,
)
from radar_ln3.task_models import save_task_checkpoint, seed_everything
from train.cascade import cascade_forward_batch, cascade_training_objective, load_hierarchical_configs


CONFIG_NAMES = {
    "dbb_cu_gate": "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-cu-gate-config.json",
    "dbb_specialist": "RADAR-LN3-V14-cq08-liquid-strict-open-set-dbb-specialist-config.json",
    "cu_specialist": "RADAR-LN3-V14-cq08-liquid-strict-open-set-cu-specialist-config.json",
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
        raise ValueError("bounded V14 Smoke received an empty parent")
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


def _shortest_by_label(dataset, labels: set[str]) -> dict[str, PointSequence]:
    selected: dict[str, tuple[int, int]] = {}
    for index, parent_id in enumerate(dataset.parent_ids):
        group = dataset._groups[str(parent_id)]
        label = str(group.iloc[0]["label_name"])
        if label not in labels:
            continue
        length = min(len(group), dataset.max_sequence_length)
        if label not in selected or length < selected[label][0]:
            selected[label] = (length, index)
    if set(selected) != labels:
        raise ValueError(f"V14 Smoke did not find labels: expected={sorted(labels)}, actual={sorted(selected)}")
    return {label: _prefix(dataset[index]) for label, (_, index) in selected.items()}


def _device_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def _model_change(before: dict[str, torch.Tensor], model: torch.nn.Module) -> float:
    return max(
        float((value.detach().cpu() - before[name]).abs().max().item())
        for name, value in model.state_dict().items()
        if torch.is_floating_point(value)
    )


def _calibration() -> StrictOpenSetRejectionCalibration:
    return StrictOpenSetRejectionCalibration(
        schema_version=STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
        gate_score_name=GATE_DBB_PROBABILITY_SCORE,
        dbb_known_score_name=DBB_KNOWN_PROBABILITY_SCORE,
        cu_score_name=CU_CLUTTER_PROBABILITY_SCORE,
        gate_accept_dbb_threshold=1.0e-6,
        gate_accept_dbb_margin=0.0,
        dbb_accept_known_threshold=1.0e-6,
        dbb_accept_known_margin=0.0,
        cu_accept_clutter_threshold=1.0e-6,
        cu_accept_clutter_margin=0.0,
        min_valid_points=2,
        min_track_quality=0.0,
        max_gap_fraction=1.0,
    )


def run(project_root: Path, *, device_name: str) -> dict[str, Any]:
    root = project_root.resolve()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA Smoke requested but CUDA is unavailable")
    device = torch.device(device_name)
    config_root = root / "config" / "RADAR-LN3"
    config_paths = {name: config_root / filename for name, filename in CONFIG_NAMES.items()}
    configs = load_hierarchical_configs(config_paths)
    contracts = verify_cascade_data_contracts(root, configs)
    gate_config = configs["dbb_cu_gate"]
    formal_artifact = root / str(gate_config.train.joint_artifact_dir)
    if formal_artifact.exists():
        raise FileExistsError(f"bounded V14 Smoke must not mix with a formal artifact: {formal_artifact}")
    train_dataset = build_cascade_dataset(
        root,
        gate_config,
        split_name=gate_config.data.train_split_name,
        label_scope="known_only",
    )
    examples = _shortest_by_label(
        train_dataset,
        {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"},
    )
    train_items = [examples["BirdTarget"], examples["ClutterTarget"]]
    train_batch_cpu = collate_point_sequences(train_items)
    if "UnknownTarget" in train_batch_cpu["label_names"] or bool((train_batch_cpu["labels"] == 4).any()):
        raise AssertionError("bounded V14 optimizer batch contains UnknownTarget")
    train_batch = _device_batch(train_batch_cpu, device)
    seed_everything(gate_config.train.seed)
    model = build_cascade(configs, device=device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=gate_config.train.learning_rate)
    modules = {
        "dbb_cu_gate": model.gate,
        "dbb_specialist": model.dbb_specialist,
        "cu_specialist": model.cu_specialist,
    }
    before = {
        name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for name, module in modules.items()
    }
    output = cascade_forward_batch(model, train_batch)
    if "sequence_probabilities" in output or "sequence_open_set_evidence" not in output:
        raise AssertionError("V14 forward exposed a learned five-class posterior")
    evidence = output["sequence_open_set_evidence"]
    if evidence.shape != (2, train_batch["tr_padding_mask"].shape[1], 5):
        raise AssertionError(f"V14 evidence shape mismatch: {evidence.shape}")
    if not torch.allclose(evidence.sum(dim=-1), torch.ones_like(evidence[..., 0]), atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError("V14 evidence decomposition is not normalized")
    objective = cascade_training_objective(output, train_batch, configs)
    optimizer.zero_grad(set_to_none=True)
    objective["loss"].backward()
    gradients = {
        name: float(torch.nn.utils.clip_grad_norm_(module.parameters(), 100.0).item())
        for name, module in modules.items()
    }
    if any(not torch.isfinite(torch.tensor(value)).item() or value <= 0.0 for value in gradients.values()):
        raise AssertionError(f"V14 Smoke has a missing/non-finite task gradient: {gradients}")
    optimizer.step()
    changes = {name: _model_change(before[name], module) for name, module in modules.items()}
    if any(value <= 0.0 for value in changes.values()):
        raise AssertionError(f"V14 Smoke did not update all three task models: {changes}")
    model.eval()
    with torch.inference_mode():
        updated = cascade_forward_batch(model, train_batch)
    smoke_dir = root / "src-v3" / "smoke" / f"RADAR-LN3-V14-strict-open-set-cascade-smoke-{device_name}"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    selection = {
        "metric": gate_config.train.checkpoint_selection_metric,
        "ranking_key": [0.0] * 8,
        "unknown_parent_count": 0,
        "unknown_used_for_selection": False,
        "smoke_only": True,
    }
    checkpoint_set_id = "0" * 32 if device_name == "cpu" else "1" * 32
    task_paths: dict[str, Path] = {}
    for name in EXPECTED_TASKS:
        path = smoke_dir / f"{name}.pt"
        save_task_checkpoint(
            path,
            modules[name],
            None,
            config=configs[name],
            epoch=1,
            global_step=1,
            metrics={
                "checkpoint_selection": selection,
                "checkpoint_set_id": checkpoint_set_id,
                "smoke_only": True,
            },
        )
        task_paths[name] = path
    restored, payloads = load_cascade(configs, task_paths, device=device)
    with torch.inference_mode():
        restored_output = cascade_forward_batch(restored, train_batch)
    keys = (
        "sequence_open_set_evidence",
        "gate_sequence_logits",
        "dbb_sequence_logits",
        "cu_sequence_logits",
    )
    checkpoint_difference = max(
        float((updated[key] - restored_output[key]).abs().max().item()) for key in keys
    )
    if checkpoint_difference != 0.0:
        raise AssertionError(f"V14 checkpoint round trip changed output: {checkpoint_difference}")

    evaluation_items = [
        examples["DroneTarget"],
        examples["BirdTarget"],
        examples["BalloonTarget"],
        examples["ClutterTarget"],
    ]
    uncalibrated = evaluate_strict_open_set_cascade(
        restored,
        _PointList(evaluation_items),
        device=device,
        batch_size=2,
        split_name="train_bounded_smoke",
        evaluation_scope="known_only_checkpoint_selection",
    )
    if (
        uncalibrated["calibrated_hierarchical_action"]["status"] != "not_evaluated"
        or uncalibrated["stable_action"]["status"] != "not_evaluated_without_calibration"
        or uncalibrated["unknown_audit_parent_count"] != 0
    ):
        raise AssertionError("uncalibrated V14 Smoke emitted rejection/stable/Unknown selection results")

    one = collate_point_sequences([examples["BirdTarget"]])
    one_device = _device_batch(one, device)
    with torch.inference_mode():
        offline = cascade_forward_batch(restored, one_device)["sequence_open_set_evidence"]
    online = OnlineLiquidSoftCascade(restored, confirm_count=2)
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
        if (
            step.final_action != "NOT_CALIBRATED"
            or step.stable_confirmed
            or step.soft_posterior_audit_probabilities is not None
            or step.open_set_audit_evidence is None
            or step.unresolved_open_set_evidence is None
        ):
            raise AssertionError("V14 uncalibrated online output violated evidence/action semantics")
        online_rows.append(step.open_set_audit_evidence)
    online_evidence = torch.stack(online_rows, dim=1)
    online_difference = float((offline - online_evidence).abs().max().item())
    if not torch.allclose(offline, online_evidence, atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError(f"V14 offline/online evidence mismatch: {online_difference}")

    synthetic_policy = StrictOpenSetRejectionPolicy(_calibration(), enabled=True)
    calibrated_online = OnlineLiquidSoftCascade(
        restored,
        confirm_count=2,
        hierarchical_rejection_policy=synthetic_policy,
    )
    action_rows: list[dict[str, Any]] = []
    for event_index in range(int(one["lengths"][0].item())):
        step = calibrated_online.step(
            tr_features=one_device["tr_features"][:, event_index],
            tr_missing_mask=one_device["tr_missing_mask"][:, event_index],
            tr_dt_seconds=one_device["tr_dt_seconds"][:, event_index],
            tr_dt_valid=one_device["tr_dt_valid"][:, event_index],
            tr_dt_reason=one_device["tr_dt_reason"][:, event_index],
            rd_frames=one_device["rd_frames"][:, event_index],
            rd_frame_mask=one_device["rd_frame_mask"][:, event_index],
            rd_quality=one_device["rd_quality"][:, event_index],
        )
        if step.final_action == "DBB_ACCEPTED" and (
            step.gate_action != "DBB_ROUTE_ACCEPTED"
            or step.dbb_known_confidence is None
            or step.dbb_known_margin is None
        ):
            raise AssertionError("V14 accepted DBB without explicit Gate and DBB-knownness evidence")
        action_rows.append({
            "event_index": event_index,
            "final_action": step.final_action,
            "gate_action": step.gate_action,
            "stable_state_name": step.stable_state_name,
            "stable_confirmed": step.stable_confirmed,
        })
    if action_rows[0]["final_action"] not in {"DBB_PENDING", "CU_PENDING"}:
        raise AssertionError("V14 calibrated online path skipped minimum-evidence pending")

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V14-strict-open-set-cascade-smoke-v1",
        "device": str(device),
        "data_contract_hashes": contracts["dbb_cu_gate"]["hashes"],
        "full_train_known_parent_count": len(train_dataset),
        "bounded_train_parent_ids": train_batch_cpu["parent_sample_ids"],
        "bounded_train_labels": train_batch_cpu["label_names"],
        "bounded_event_limit_per_parent": 3,
        "batch_shapes": {
            key: list(value.shape)
            for key, value in train_batch_cpu.items()
            if isinstance(value, torch.Tensor)
        },
        "open_set_evidence_shape": list(evidence.shape),
        "unknown_is_probability_output": False,
        "objective": {
            key: float(value.item())
            for key, value in objective.items()
            if isinstance(value, torch.Tensor) and value.ndim == 0
        },
        "gradient_norms": gradients,
        "max_parameter_changes": changes,
        "task_checkpoint_paths": {name: str(path) for name, path in task_paths.items()},
        "task_checkpoint_epochs": {name: int(payloads[name]["epoch"]) for name in EXPECTED_TASKS},
        "task_checkpoint_round_trip_max_difference": checkpoint_difference,
        "offline_online_evidence_max_difference": online_difference,
        "uncalibrated_rejection_status": uncalibrated["calibrated_hierarchical_action"]["status"],
        "uncalibrated_stable_status": uncalibrated["stable_action"]["status"],
        "synthetic_calibration_only": True,
        "synthetic_action_trace": action_rows,
        "formal_training": "未执行",
        "oof_fold_training": "未执行",
        "formal_rejection": "未执行",
        "stable_performance": "未执行",
        "test": "未执行",
        "p95": "未执行",
    }
    result_path = smoke_dir / "result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a bounded real-CQ08 V14 strict-open-set Smoke.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args.project_root.resolve(), device_name=args.device)


if __name__ == "__main__":
    main()
