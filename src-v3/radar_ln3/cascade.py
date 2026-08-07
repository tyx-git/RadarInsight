from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from . import LABELS
from .data import CQ08PointSequenceDataset, build_point_dataloader, require_causal_rd_quality_schema
from .metrics import confidence_summary, grouped_summaries, summarize_records
from .rejection import RejectionPolicy
from .stable import StableDecision, TargetStableStateMachine
from .task_models import (
    LiquidTaskClassifier,
    LiquidTaskConfig,
    compose_soft_cascade,
    load_task_checkpoint,
    resolve_project_path,
    verify_task_data_contract,
)


EXPECTED_TASKS = ("dbb_cu_gate", "dbb_specialist", "cu_specialist")


def _shared_data_contract(config: LiquidTaskConfig) -> tuple[Any, ...]:
    data = config.data
    return (
        data.index_path,
        data.contract_path,
        data.normalization_stats_path,
        data.train_split_name,
        data.validation_split_name,
        data.test_split_name,
        data.index_schema_version,
        data.expected_index_sha256,
        data.expected_contract_sha256,
        data.expected_stats_sha256,
        data.max_sequence_length,
        data.max_rd_frames_per_point,
        data.rd_amplitude_center,
        data.rd_amplitude_scale,
        data.exclude_anomalous_parents,
        data.rd_quality_schema,
        data.contract_schema_version,
        data.tr_motion_schema,
        data.tr_motion_mask_mode,
    )


def _shared_model_contract(config: LiquidTaskConfig) -> tuple[Any, ...]:
    model = config.model
    return (
        model.tr_value_dim,
        model.tr_missing_dim,
        model.rd_quality_dim,
        model.hidden_dim,
        model.rd_embedding_dim,
        model.liquid_hidden_dim,
        model.rd_height,
        model.rd_width,
        model.liquid_tau_min_seconds,
        model.liquid_tau_max_seconds,
        model.liquid_zero_dt_event_floor_seconds,
        model.liquid_dt_max_seconds,
        model.dt_valid_dim,
        model.dt_valid_feature_mode,
        model.liquid_invalid_dt_update_mode,
        model.liquid_invalid_dt_observation_gain,
        model.liquid_state_normalization,
        model.rd_frame_aggregation,
        model.tr_motion_dim,
    )


def validate_cascade_configs(configs: Mapping[str, LiquidTaskConfig]) -> None:
    if set(configs) != set(EXPECTED_TASKS):
        raise ValueError(f"soft cascade requires exactly these tasks: {EXPECTED_TASKS}")
    ordered = [configs[name] for name in EXPECTED_TASKS]
    for name, config in zip(EXPECTED_TASKS, ordered):
        if config.task.name != name:
            raise ValueError(f"cascade config task mismatch: key={name}, task={config.task.name}")
        require_causal_rd_quality_schema(config.data.rd_quality_schema)
    if len({config.schema_version for config in ordered}) != 1:
        raise ValueError("soft cascade task configurations must use one run/configuration version")
    if len({_shared_data_contract(config) for config in ordered}) != 1:
        raise ValueError("soft cascade task configurations do not share one data contract")
    if len({_shared_model_contract(config) for config in ordered}) != 1:
        raise ValueError("soft cascade task configurations do not share one encoder/dynamics contract")


class LiquidSoftCascade(nn.Module):
    """Three independent liquid models composed into one five-class posterior."""

    def __init__(
        self,
        gate: LiquidTaskClassifier,
        dbb_specialist: LiquidTaskClassifier,
        cu_specialist: LiquidTaskClassifier,
    ) -> None:
        super().__init__()
        if (gate.task_name, dbb_specialist.task_name, cu_specialist.task_name) != EXPECTED_TASKS:
            raise ValueError("soft cascade received models in an invalid task order")
        configs = {model.task_name: model.config for model in (gate, dbb_specialist, cu_specialist)}
        validate_cascade_configs(configs)
        self.gate = gate
        self.dbb_specialist = dbb_specialist
        self.cu_specialist = cu_specialist

    def forward(self, **batch: torch.Tensor) -> dict[str, torch.Tensor]:
        gate = self.gate(**batch)
        dbb = self.dbb_specialist(**batch)
        cu = self.cu_specialist(**batch)
        sequence_probabilities = compose_soft_cascade(
            gate["sequence_logits"], dbb["sequence_logits"], cu["sequence_logits"]
        )
        window_probabilities = compose_soft_cascade(
            gate["window_logits"], dbb["window_logits"], cu["window_logits"]
        )
        return {
            "sequence_probabilities": sequence_probabilities,
            "window_probabilities": window_probabilities,
            "gate_sequence_logits": gate["sequence_logits"],
            "dbb_sequence_logits": dbb["sequence_logits"],
            "cu_sequence_logits": cu["sequence_logits"],
            "gate_window_logits": gate["window_logits"],
            "dbb_window_logits": dbb["window_logits"],
            "cu_window_logits": cu["window_logits"],
            "gate_final_state": gate["liquid_final_state"],
            "dbb_final_state": dbb["liquid_final_state"],
            "cu_final_state": cu["liquid_final_state"],
            "dt_invalid_mask": gate["dt_invalid_mask"],
        }


@dataclass
class CascadeOnlineStepResult:
    window_probabilities: torch.Tensor
    window_label_id: int
    window_label_name: str
    window_confidence: float
    decision_state: str
    rejected: bool | None
    rejection_score: float | None
    rejection_calibration_source: str | None
    stable_label_id: int | None
    stable_label_name: str | None
    stable_confidence: float | None
    stable_confirmed: bool
    gate_probabilities: torch.Tensor
    dbb_probabilities: torch.Tensor
    cu_probabilities: torch.Tensor
    dt_clamped: bool
    dt_zero: bool
    dt_invalid: bool


class OnlineLiquidSoftCascade:
    """Per-target causal state for the gate, both specialists and stable output."""

    def __init__(
        self,
        model: LiquidSoftCascade,
        *,
        confirm_count: int = 2,
        ema_decay: float = 0.65,
        rejection_policy: RejectionPolicy | None = None,
    ) -> None:
        self.model = model
        self.confirm_count = max(1, int(confirm_count))
        self.ema_decay = float(ema_decay)
        self.rejection_policy = rejection_policy or RejectionPolicy()
        self.reset()

    def reset(self) -> None:
        self.gate_state: torch.Tensor | None = None
        self.dbb_state: torch.Tensor | None = None
        self.cu_state: torch.Tensor | None = None
        self.stable = TargetStableStateMachine(confirm_count=self.confirm_count, ema_decay=self.ema_decay)

    @torch.inference_mode()
    def step(self, **point: torch.Tensor) -> CascadeOnlineStepResult:
        if point.get("tr_features") is None or point["tr_features"].shape[0] != 1:
            raise ValueError("OnlineLiquidSoftCascade requires batch size 1 per target")
        gate = self.model.gate.forward_online_point(**point, liquid_state=self.gate_state)
        dbb = self.model.dbb_specialist.forward_online_point(**point, liquid_state=self.dbb_state)
        cu = self.model.cu_specialist.forward_online_point(**point, liquid_state=self.cu_state)
        self.gate_state = gate["liquid_state"]
        self.dbb_state = dbb["liquid_state"]
        self.cu_state = cu["liquid_state"]
        probabilities = compose_soft_cascade(
            gate["window_logits"], dbb["window_logits"], cu["window_logits"]
        )
        decision: StableDecision = self.stable.update_probabilities(probabilities[0].cpu())
        rejection = self.rejection_policy.decide(probabilities[0].cpu())
        window_id = int(probabilities[0].argmax().item())
        if rejection.class_label_id != window_id:
            raise RuntimeError("rejection policy changed the five-class classification label")
        return CascadeOnlineStepResult(
            window_probabilities=probabilities,
            window_label_id=window_id,
            window_label_name=LABELS[window_id],
            window_confidence=float(probabilities[0, window_id].item()),
            decision_state=rejection.decision_state,
            rejected=rejection.rejected,
            rejection_score=rejection.rejection_score,
            rejection_calibration_source=rejection.calibration_source,
            stable_label_id=decision.stable_label_id,
            stable_label_name=decision.stable_label_name,
            stable_confidence=decision.stable_confidence,
            stable_confirmed=decision.stable_confirmed,
            gate_probabilities=torch.softmax(gate["window_logits"], dim=-1),
            dbb_probabilities=torch.softmax(dbb["window_logits"], dim=-1),
            cu_probabilities=torch.softmax(cu["window_logits"], dim=-1),
            dt_clamped=bool(gate["dt_clamped_mask"][0].item()),
            dt_zero=bool(gate["dt_zero_mask"][0].item()),
            dt_invalid=bool(gate["dt_invalid_mask"][0].item()),
        )


def load_cascade(
    configs: Mapping[str, LiquidTaskConfig],
    checkpoints: Mapping[str, str | Path],
    *,
    device: torch.device,
) -> tuple[LiquidSoftCascade, dict[str, dict[str, Any]]]:
    validate_cascade_configs(configs)
    if set(checkpoints) != set(EXPECTED_TASKS):
        raise ValueError(f"soft cascade requires exactly three checkpoints: {EXPECTED_TASKS}")
    models: dict[str, LiquidTaskClassifier] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name in EXPECTED_TASKS:
        model = LiquidTaskClassifier(configs[name]).to(device)
        payload = load_task_checkpoint(
            checkpoints[name], model, expected_schema=configs[name].train.checkpoint_schema, map_location=device
        )
        models[name] = model.eval()
        payloads[name] = payload
    cascade = LiquidSoftCascade(
        models["dbb_cu_gate"], models["dbb_specialist"], models["cu_specialist"]
    ).to(device).eval()
    return cascade, payloads


def build_cascade_dataset(project_root: Path, config: LiquidTaskConfig, *, split_name: str) -> CQ08PointSequenceDataset:
    return CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=split_name,
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=config.data.exclude_anomalous_parents,
        sample_limit=None,
        expected_schema_version=config.data.index_schema_version,
        tr_motion_schema=config.data.tr_motion_schema if config.model.tr_motion_dim > 0 else None,
    )


def evaluate_cascade(
    model: LiquidSoftCascade,
    dataset: CQ08PointSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    split_name: str,
    stable_confirm_count: int = 2,
    stable_ema_decay: float = 0.65,
) -> dict[str, Any]:
    loader = build_point_dataloader(dataset, batch_size=batch_size, shuffle=False)
    window_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    stable_records: list[dict[str, Any]] = []
    stable_unconfirmed: list[dict[str, Any]] = []
    gate_records: list[dict[str, Any]] = []
    dbb_records: list[dict[str, Any]] = []
    cu_records: list[dict[str, Any]] = []
    dt_invalid = 0
    with torch.inference_mode():
        for batch in loader:
            tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(
                tr_features=tensors["tr_features"],
                tr_missing_mask=tensors["tr_missing_mask"],
                tr_dt_seconds=tensors["tr_dt_seconds"],
                tr_dt_valid=tensors["tr_dt_valid"],
                rd_frames=tensors["rd_frames"],
                rd_frame_mask=tensors["rd_frame_mask"],
                rd_quality=tensors["rd_quality"],
                tr_padding_mask=tensors["tr_padding_mask"],
            )
            dt_invalid += int(output["dt_invalid_mask"].sum().item())
            for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][batch_index].item())
                true_id = int(batch["labels"][batch_index].item())
                stable = TargetStableStateMachine(confirm_count=stable_confirm_count, ema_decay=stable_ema_decay)
                final: dict[str, Any] | None = None
                decision: StableDecision | None = None
                for event_index in range(length):
                    probabilities = output["sequence_probabilities"][batch_index, event_index]
                    pred_id = int(probabilities.argmax().item())
                    base = {
                        "true_id": true_id,
                        "pred_id": pred_id,
                        "confidence": float(probabilities[pred_id].item()),
                        "parent_sample_id": parent_id,
                        "track_id": batch["track_ids"][batch_index],
                        "source_name": batch["source_names"][batch_index],
                        "label_name": batch["label_names"][batch_index],
                        "split_name": split_name,
                        "event_index": event_index,
                        "window_length": event_index + 1,
                    }
                    window_records.append(base)
                    final = dict(base)
                    decision = stable.update_probabilities(probabilities.cpu())
                if final is None or decision is None:
                    raise ValueError(f"empty sequence reached cascade evaluator: {parent_id}")
                parent_records.append(final)
                stable_base = {key: final[key] for key in (
                    "true_id", "parent_sample_id", "track_id", "source_name", "label_name", "split_name"
                )}
                stable_base["event_count"] = length
                stable_base["stable_confirmed"] = decision.stable_confirmed
                if decision.stable_confirmed:
                    stable_records.append({
                        **stable_base,
                        "pred_id": int(decision.stable_label_id),
                        "confidence": float(decision.stable_confidence),
                    })
                else:
                    stable_unconfirmed.append({
                        **stable_base,
                        "pred_id": None,
                        "confidence": None,
                        "candidate_label_id": decision.candidate_label_id,
                        "candidate_count": decision.candidate_count,
                    })
                gate_true = 0 if true_id < 3 else 1
                gate_pred = int(output["gate_window_logits"][batch_index].argmax().item())
                gate_records.append({"true_id": gate_true, "pred_id": gate_pred})
                if true_id < 3:
                    dbb_records.append({
                        "true_id": true_id,
                        "pred_id": int(output["dbb_window_logits"][batch_index].argmax().item()),
                    })
                else:
                    cu_records.append({
                        "true_id": true_id - 3,
                        "pred_id": int(output["cu_window_logits"][batch_index].argmax().item()),
                    })
    bucketed: list[dict[str, Any]] = []
    for row in window_records:
        length = int(row["window_length"])
        bucketed.append({**row, "window_length_bucket": "le_10" if length <= 10 else "11_20" if length <= 20 else "gt_20"})
    return {
        "split": split_name,
        "class_names": list(LABELS),
        "dataset_parent_count": len(dataset),
        "window_count": len(window_records),
        "parent_count": len(parent_records),
        "stable_confirmed_parent_count": len(stable_records),
        "stable_unconfirmed_parent_count": len(stable_unconfirmed),
        "stable_coverage": len(stable_records) / len(parent_records) if parent_records else 0.0,
        "window_metrics": summarize_records(window_records, scope=f"{split_name}:cascade_windows"),
        "parent_metrics": summarize_records(parent_records, scope=f"{split_name}:cascade_parent_final"),
        "stable_metrics": summarize_records(stable_records, scope=f"{split_name}:cascade_stable_confirmed") if stable_records else None,
        "stable_confirmed_metrics": summarize_records(stable_records, scope=f"{split_name}:cascade_stable_confirmed_only") if stable_records else None,
        "source_metrics": grouped_summaries(window_records, field="source_name", scope_prefix=f"{split_name}:cascade_source"),
        "parent_source_metrics": grouped_summaries(parent_records, field="source_name", scope_prefix=f"{split_name}:cascade_parent_source"),
        "window_length_bucket_metrics": grouped_summaries(bucketed, field="window_length_bucket", scope_prefix=f"{split_name}:cascade_length"),
        "session_metrics": {
            "status": "not_available",
            "reason": "the current point index and Dataset contract contain no verified session/capture field; session metrics were not estimated",
        },
        "confidence": {
            "window": confidence_summary(window_records),
            "parent": confidence_summary(parent_records),
            "stable": confidence_summary(stable_records),
        },
        "dt_quality": {"dt_invalid_event_count": dt_invalid},
        "probability_calibration": {
            "status": "not_executed",
            "reason": "the gate-specialist probability product has not undergone temperature calibration or ECE validation",
        },
        "rejection": {
            "interface_available": True,
            "status": "not_calibrated",
            "enabled": False,
            "supports_validated_rejection": False,
            "required_calibration_source": "train_parent_grouped_oof",
            "unknown_target_is_ordinary_class": True,
            "coverage": "not_evaluated",
            "accepted_macro_f1": "not_evaluated",
            "false_acceptance_rate": "not_evaluated",
            "false_rejection_rate": "not_evaluated",
            "selective_risk": "not_evaluated",
        },
        "component_metrics": {
            "gate_parent": summarize_records_with_names(gate_records, ("DBB", "CU"), "gate_parent"),
            "dbb_parent_conditional": summarize_records_with_names(dbb_records, LABELS[:3], "dbb_parent_conditional"),
            "cu_parent_conditional": summarize_records_with_names(cu_records, LABELS[3:], "cu_parent_conditional"),
        },
        "records": {
            "window": window_records,
            "parent": parent_records,
            "stable": stable_records,
            "stable_unconfirmed": stable_unconfirmed,
        },
    }


def summarize_records_with_names(records: list[dict[str, int]], names: tuple[str, ...], scope: str) -> dict[str, Any]:
    from .task_models import summarize_task_records

    return summarize_task_records(records, class_names=names, scope=scope)


def verify_cascade_data_contracts(project_root: Path, configs: Mapping[str, LiquidTaskConfig]) -> dict[str, Any]:
    validate_cascade_configs(configs)
    results = {name: verify_task_data_contract(project_root, configs[name]) for name in EXPECTED_TASKS}
    hashes = {tuple(sorted(result["hashes"].items())) for result in results.values()}
    if len(hashes) != 1:
        raise ValueError("cascade tasks resolved different data hashes")
    return results


__all__ = [
    "CascadeOnlineStepResult",
    "EXPECTED_TASKS",
    "LiquidSoftCascade",
    "OnlineLiquidSoftCascade",
    "build_cascade_dataset",
    "evaluate_cascade",
    "load_cascade",
    "validate_cascade_configs",
    "verify_cascade_data_contracts",
]
