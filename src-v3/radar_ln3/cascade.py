from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from . import LABELS
from .data import CQ08PointSequenceDataset, build_point_dataloader, require_causal_rd_quality_schema
from .metrics import confidence_summary, grouped_summaries, summarize_records
from .rejection import (
    HierarchicalActionDecision,
    HierarchicalRejectionPolicy,
    RejectionPolicy,
    StrictOpenSetRejectionPolicy,
)
from .stable import (
    HierarchicalStableDecision,
    HierarchicalStableStateMachine,
    StableDecision,
    TargetStableStateMachine,
)
from .task_models import (
    HIERARCHICAL_SCHEMA_VERSIONS,
    HIERARCHICAL_CASCADE_SCHEMA_VERSION,
    KNOWN_BASE_LABELS,
    STRICT_OPEN_SET_SCHEMA_VERSION,
    UNKNOWN_BASE_LABEL,
    LiquidTaskClassifier,
    LiquidTaskConfig,
    compose_soft_cascade,
    compose_strict_open_set_evidence,
    load_task_checkpoint,
    resolve_project_path,
    verify_task_data_contract,
)


EXPECTED_TASKS = ("dbb_cu_gate", "dbb_specialist", "cu_specialist")
STRICT_OPEN_SET_EVIDENCE_NAMES = (*LABELS[:4], "UNRESOLVED_OPEN_SET_EVIDENCE")


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
        model.dt_reason_mode,
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
    if ordered[0].schema_version in HIERARCHICAL_SCHEMA_VERSIONS:
        shared_train = {
            (
                config.train.batch_size,
                config.train.learning_rate,
                config.train.seed,
                config.train.device,
                config.train.epochs,
                config.train.gradient_clip_norm,
                config.train.loss_mode,
                config.train.checkpoint_selection_metric,
                config.train.prefix_auxiliary_weight,
                config.train.prefix_auxiliary_min_events,
                config.train.dataloader_seed,
                config.train.record_parent_order_hash,
                config.train.save_every_epoch,
                config.train.joint_artifact_dir,
                config.train.joint_last_checkpoint_path,
                config.train.joint_best_checkpoint_path,
                config.train.joint_checkpoint_schema,
                config.train.gate_min_precision_recall_target,
                config.train.dbb_min_precision_recall_target,
                config.train.cu_min_precision_recall_target,
                config.train.drone_bird_min_precision_recall_target,
            )
            for config in ordered
        }
        if len(shared_train) != 1:
            raise ValueError("hierarchical tasks do not share one joint training/product checkpoint contract")
        if len({config.train.formal_artifact_dir for config in ordered}) != len(EXPECTED_TASKS):
            raise ValueError("hierarchical task checkpoints require distinct task artifact directories")


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
        strict_open_set = self.gate.config.schema_version == STRICT_OPEN_SET_SCHEMA_VERSION
        compose = compose_strict_open_set_evidence if strict_open_set else compose_soft_cascade
        sequence_audit = compose(
            gate["sequence_logits"], dbb["sequence_logits"], cu["sequence_logits"]
        )
        window_audit = compose(
            gate["window_logits"], dbb["window_logits"], cu["window_logits"]
        )
        result = {
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
        if strict_open_set:
            result.update({
                "sequence_open_set_evidence": sequence_audit,
                "window_open_set_evidence": window_audit,
            })
        else:
            result.update({
                "sequence_probabilities": sequence_audit,
                "window_probabilities": window_audit,
            })
        return result


@dataclass
class CascadeOnlineStepResult:
    window_probabilities: torch.Tensor | None
    window_label_id: int | None
    window_label_name: str | None
    window_confidence: float | None
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
    gate_action: str | None = None
    conditional_action: str | None = None
    final_action: str | None = None
    diagnostic_gate_route: str | None = None
    diagnostic_label_id: int | None = None
    diagnostic_label_name: str | None = None
    accepted_label_id: int | None = None
    accepted_label_name: str | None = None
    rejected_to_unknown: bool | None = None
    stable_state_name: str | None = None
    stable_rejected_to_unknown: bool = False
    action_trigger_reasons: tuple[str, ...] = ()
    soft_posterior_audit_probabilities: torch.Tensor | None = None
    soft_posterior_audit_label_id: int | None = None
    soft_posterior_audit_label_name: str | None = None
    soft_posterior_audit_confidence: float | None = None
    gate_dbb_probability: float | None = None
    gate_margin: float | None = None
    specialist_label_id: int | None = None
    specialist_confidence: float | None = None
    specialist_margin: float | None = None
    valid_point_count: int | None = None
    track_quality: float | None = None
    gap_fraction: float | None = None
    dbb_known_confidence: float | None = None
    dbb_known_margin: float | None = None
    clutter_accept_confidence: float | None = None
    clutter_accept_margin: float | None = None
    open_set_audit_evidence: torch.Tensor | None = None
    open_set_audit_evidence_names: tuple[str, ...] = ()
    unresolved_open_set_evidence: float | None = None


class OnlineLiquidSoftCascade:
    """Per-target causal state for the gate, both specialists and stable output."""

    def __init__(
        self,
        model: LiquidSoftCascade,
        *,
        confirm_count: int = 2,
        ema_decay: float = 0.65,
        rejection_policy: RejectionPolicy | None = None,
        hierarchical_rejection_policy: HierarchicalRejectionPolicy | StrictOpenSetRejectionPolicy | None = None,
    ) -> None:
        self.model = model
        self.confirm_count = max(1, int(confirm_count))
        self.ema_decay = float(ema_decay)
        self.rejection_policy = rejection_policy or RejectionPolicy()
        self.hierarchical_mode = model.gate.config.schema_version in HIERARCHICAL_SCHEMA_VERSIONS
        self.strict_open_set_mode = model.gate.config.schema_version == STRICT_OPEN_SET_SCHEMA_VERSION
        if hierarchical_rejection_policy is None:
            self.hierarchical_rejection_policy = (
                StrictOpenSetRejectionPolicy()
                if self.strict_open_set_mode
                else HierarchicalRejectionPolicy()
            )
        else:
            self.hierarchical_rejection_policy = hierarchical_rejection_policy
        if self.strict_open_set_mode and not isinstance(self.hierarchical_rejection_policy, StrictOpenSetRejectionPolicy):
            raise ValueError("V14 online cascade requires the strict-open-set rejection policy")
        if (
            self.strict_open_set_mode
            and self.hierarchical_rejection_policy.enabled
            and self.hierarchical_rejection_policy.calibration is not None
            and not self.hierarchical_rejection_policy.calibration.deployment_eligible
        ):
            raise PermissionError("V14 audit-only calibration cannot enter the online cascade")
        if self.hierarchical_mode and not self.strict_open_set_mode and not isinstance(
            self.hierarchical_rejection_policy, HierarchicalRejectionPolicy
        ):
            raise ValueError("V13 online cascade requires the V13 hierarchical rejection policy")
        if self.hierarchical_mode and rejection_policy is not None:
            raise ValueError(f"{model.gate.config.schema_version} online cascade must not use the legacy five-class rejection policy")
        if not self.hierarchical_mode and hierarchical_rejection_policy is not None:
            raise ValueError("historical online cascade must not use a hierarchical rejection policy")
        self.reset()

    def reset(self) -> None:
        self.gate_state: torch.Tensor | None = None
        self.dbb_state: torch.Tensor | None = None
        self.cu_state: torch.Tensor | None = None
        self.stable = TargetStableStateMachine(confirm_count=self.confirm_count, ema_decay=self.ema_decay)
        self.hierarchical_stable = HierarchicalStableStateMachine(confirm_count=self.confirm_count)
        self.valid_point_count = 0
        self.rd_observed_count = 0
        self.zero_delta_count = 0

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
        probabilities = (
            compose_strict_open_set_evidence(
                gate["window_logits"], dbb["window_logits"], cu["window_logits"]
            )
            if self.strict_open_set_mode
            else compose_soft_cascade(
                gate["window_logits"], dbb["window_logits"], cu["window_logits"]
            )
        )
        gate_probabilities = torch.softmax(gate["window_logits"], dim=-1)
        dbb_probabilities = torch.softmax(dbb["window_logits"], dim=-1)
        cu_probabilities = torch.softmax(cu["window_logits"], dim=-1)
        window_id = int(probabilities[0].argmax().item())
        action: HierarchicalActionDecision | None = None
        hierarchical_stable: HierarchicalStableDecision | None = None
        if self.hierarchical_mode:
            self.valid_point_count += 1
            self.rd_observed_count += int(float(point["rd_quality"][0, 0].item()) >= 0.5)
            reason = point.get("tr_dt_reason")
            if reason is None:
                raise ValueError(f"{self.model.gate.config.schema_version} online cascade requires tr_dt_reason")
            self.zero_delta_count += int(int(reason.reshape(-1)[0].item()) == 2)
            track_quality = self.rd_observed_count / self.valid_point_count
            gap_fraction = self.zero_delta_count / max(1, self.valid_point_count - 1)
            action = self.hierarchical_rejection_policy.decide(
                gate_probabilities[0].cpu(),
                dbb_probabilities[0].cpu(),
                cu_probabilities[0].cpu(),
                valid_point_count=self.valid_point_count,
                track_quality=track_quality,
                gap_fraction=min(1.0, gap_fraction),
            )
            hierarchical_stable = self.hierarchical_stable.update(action)
            decision_state = action.decision_state
            rejected = action.rejected_to_unknown
            rejection_score = None
            rejection_source = action.calibration_source
            stable_label_id = hierarchical_stable.stable_label_id
            stable_label_name = hierarchical_stable.stable_label_name
            stable_confidence = None
            stable_confirmed = hierarchical_stable.stable_confirmed
        else:
            decision = self.stable.update_probabilities(probabilities[0].cpu())
            rejection = self.rejection_policy.decide(probabilities[0].cpu())
            if rejection.class_label_id != window_id:
                raise RuntimeError("rejection policy changed the five-class classification label")
            decision_state = rejection.decision_state
            rejected = rejection.rejected
            rejection_score = rejection.rejection_score
            rejection_source = rejection.calibration_source
            stable_label_id = decision.stable_label_id
            stable_label_name = decision.stable_label_name
            stable_confidence = decision.stable_confidence
            stable_confirmed = decision.stable_confirmed
        audit_confidence = float(probabilities[0, window_id].item())
        return CascadeOnlineStepResult(
            window_probabilities=None if self.hierarchical_mode else probabilities,
            window_label_id=None if self.hierarchical_mode else window_id,
            window_label_name=None if self.hierarchical_mode else LABELS[window_id],
            window_confidence=None if self.hierarchical_mode else audit_confidence,
            decision_state=decision_state,
            rejected=rejected,
            rejection_score=rejection_score,
            rejection_calibration_source=rejection_source,
            stable_label_id=stable_label_id,
            stable_label_name=stable_label_name,
            stable_confidence=stable_confidence,
            stable_confirmed=stable_confirmed,
            gate_probabilities=gate_probabilities,
            dbb_probabilities=dbb_probabilities,
            cu_probabilities=cu_probabilities,
            dt_clamped=bool(gate["dt_clamped_mask"][0].item()),
            dt_zero=bool(gate["dt_zero_mask"][0].item()),
            dt_invalid=bool(gate["dt_invalid_mask"][0].item()),
            gate_action=None if action is None else action.gate_action,
            conditional_action=None if action is None else action.conditional_action,
            final_action=None if action is None else action.final_action,
            diagnostic_gate_route=None if action is None else action.diagnostic_gate_route,
            diagnostic_label_id=None if action is None else action.diagnostic_label_id,
            diagnostic_label_name=None if action is None else action.diagnostic_label_name,
            accepted_label_id=None if action is None else action.accepted_label_id,
            accepted_label_name=None if action is None else action.accepted_label_name,
            rejected_to_unknown=None if action is None else action.rejected_to_unknown,
            stable_state_name=None if hierarchical_stable is None else hierarchical_stable.state_name,
            stable_rejected_to_unknown=False if hierarchical_stable is None else hierarchical_stable.stable_rejected_to_unknown,
            action_trigger_reasons=() if action is None else action.trigger_reasons,
            soft_posterior_audit_probabilities=(
                probabilities if self.hierarchical_mode and not self.strict_open_set_mode else None
            ),
            soft_posterior_audit_label_id=(
                window_id if self.hierarchical_mode and not self.strict_open_set_mode else None
            ),
            soft_posterior_audit_label_name=(
                LABELS[window_id] if self.hierarchical_mode and not self.strict_open_set_mode else None
            ),
            soft_posterior_audit_confidence=(
                audit_confidence if self.hierarchical_mode and not self.strict_open_set_mode else None
            ),
            gate_dbb_probability=None if action is None else action.gate_dbb_probability,
            gate_margin=None if action is None else action.gate_margin,
            specialist_label_id=None if action is None else action.specialist_label_id,
            specialist_confidence=None if action is None else action.specialist_confidence,
            specialist_margin=None if action is None else action.specialist_margin,
            valid_point_count=None if action is None else action.valid_point_count,
            track_quality=None if action is None else action.track_quality,
            gap_fraction=None if action is None else action.gap_fraction,
            dbb_known_confidence=None if action is None else action.dbb_known_confidence,
            dbb_known_margin=None if action is None else action.dbb_known_margin,
            clutter_accept_confidence=None if action is None else action.clutter_accept_confidence,
            clutter_accept_margin=None if action is None else action.clutter_accept_margin,
            open_set_audit_evidence=probabilities if self.strict_open_set_mode else None,
            open_set_audit_evidence_names=(
                STRICT_OPEN_SET_EVIDENCE_NAMES if self.strict_open_set_mode else ()
            ),
            unresolved_open_set_evidence=(
                float(probabilities[0, 4].item()) if self.strict_open_set_mode else None
            ),
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
    if configs["dbb_cu_gate"].schema_version in HIERARCHICAL_SCHEMA_VERSIONS:
        expected_ranking_key_length = (
            8
            if configs["dbb_cu_gate"].schema_version == STRICT_OPEN_SET_SCHEMA_VERSION
            else 9
        )
        config_identity_matches = all(
            payloads[name].get("config_name") == configs[name].config_name
            for name in EXPECTED_TASKS
        )
        epochs = {int(payloads[name].get("epoch", -1)) for name in EXPECTED_TASKS}
        steps = {int(payloads[name].get("global_step", -1)) for name in EXPECTED_TASKS}
        selection_payloads = [
            dict(payloads[name].get("metrics") or {}).get("checkpoint_selection")
            for name in EXPECTED_TASKS
        ]
        checkpoint_set_ids = {
            str(dict(payloads[name].get("metrics") or {}).get("checkpoint_set_id", ""))
            for name in EXPECTED_TASKS
        }
        valid_selections = all(
            isinstance(selection, dict)
            and selection.get("metric") == configs[name].train.checkpoint_selection_metric
            and isinstance(selection.get("ranking_key"), list)
            and len(selection["ranking_key"]) == expected_ranking_key_length
            for name, selection in zip(EXPECTED_TASKS, selection_payloads)
        )
        selections = {
            json.dumps(selection, sort_keys=True, ensure_ascii=False)
            for selection in selection_payloads
            if isinstance(selection, dict)
        }
        if (
            not config_identity_matches
            or epochs == {-1}
            or steps == {-1}
            or min(epochs) < 1
            or min(steps) < 1
            or len(epochs) != 1
            or len(steps) != 1
            or not valid_selections
            or len(selections) != 1
            or len(checkpoint_set_ids) != 1
            or len(next(iter(checkpoint_set_ids))) != 32
            or any(
                character not in "0123456789abcdef"
                for character in next(iter(checkpoint_set_ids))
            )
        ):
            raise ValueError(
                "hierarchical task checkpoints do not match their configs or are not one jointly selected epoch/global-step/product candidate"
            )
    cascade = LiquidSoftCascade(
        models["dbb_cu_gate"], models["dbb_specialist"], models["cu_specialist"]
    ).to(device).eval()
    return cascade, payloads


def build_cascade(configs: Mapping[str, LiquidTaskConfig], *, device: torch.device) -> LiquidSoftCascade:
    """Build one initialized three-task cascade without loading checkpoints."""

    validate_cascade_configs(configs)
    models = {name: LiquidTaskClassifier(configs[name]).to(device) for name in EXPECTED_TASKS}
    return LiquidSoftCascade(
        models["dbb_cu_gate"],
        models["dbb_specialist"],
        models["cu_specialist"],
    ).to(device)


def build_cascade_dataset(
    project_root: Path,
    config: LiquidTaskConfig,
    *,
    split_name: str,
    label_scope: str | None = None,
    parent_ids: tuple[str, ...] | list[str] | set[str] | frozenset[str] | None = None,
) -> CQ08PointSequenceDataset:
    strict_open_set = config.schema_version == STRICT_OPEN_SET_SCHEMA_VERSION
    if strict_open_set:
        if label_scope not in {"known_only", "development_open_set_audit"}:
            raise ValueError("V14 Dataset requires an explicit known_only or development_open_set_audit scope")
        if split_name == config.data.train_split_name and label_scope != "known_only":
            raise PermissionError("V14 train Dataset is permanently restricted to known_only labels")
        if label_scope == "development_open_set_audit" and split_name != config.data.validation_split_name:
            raise PermissionError("V14 open-set audit scope is restricted to the frozen validation split")
    elif label_scope not in {None, "all"}:
        raise ValueError("historical cascade Dataset does not support V14 label scopes")
    strict_known_view = strict_open_set and label_scope == "known_only"
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
        allowed_labels=KNOWN_BASE_LABELS if strict_known_view else None,
        forbidden_labels={UNKNOWN_BASE_LABEL} if strict_known_view else None,
        allowed_parent_ids=parent_ids,
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
    hierarchical_rejection_policy: HierarchicalRejectionPolicy | StrictOpenSetRejectionPolicy | None = None,
    evaluation_scope: str | None = None,
) -> dict[str, Any]:
    if model.gate.config.schema_version == STRICT_OPEN_SET_SCHEMA_VERSION:
        if hierarchical_rejection_policy is not None and not isinstance(
            hierarchical_rejection_policy, StrictOpenSetRejectionPolicy
        ):
            raise ValueError("V14 evaluation requires the strict-open-set rejection policy")
        return evaluate_strict_open_set_cascade(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            split_name=split_name,
            evaluation_scope=evaluation_scope or "development_open_set_audit",
            rejection_policy=hierarchical_rejection_policy,
            stable_confirm_count=stable_confirm_count,
        )
    if model.gate.config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        if hierarchical_rejection_policy is not None and not isinstance(
            hierarchical_rejection_policy, HierarchicalRejectionPolicy
        ):
            raise ValueError("V13 evaluation requires the V13 hierarchical rejection policy")
        return evaluate_hierarchical_cascade(
            model,
            dataset,
            device=device,
            batch_size=batch_size,
            split_name=split_name,
            rejection_policy=hierarchical_rejection_policy,
            stable_confirm_count=stable_confirm_count,
        )
    if hierarchical_rejection_policy is not None:
        raise ValueError("historical cascade evaluation cannot use the V13 hierarchical rejection policy")
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
                tr_dt_reason=tensors.get("tr_dt_reason"),
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
                    gate_event = torch.softmax(
                        output["gate_sequence_logits"][batch_index, event_index], dim=-1
                    )
                    dbb_event = torch.softmax(
                        output["dbb_sequence_logits"][batch_index, event_index], dim=-1
                    )
                    cu_event = torch.softmax(
                        output["cu_sequence_logits"][batch_index, event_index], dim=-1
                    )
                    diagnostic_gate_pred = int(gate_event.argmax().item())
                    diagnostic_action_pred = (
                        int(dbb_event.argmax().item())
                        if diagnostic_gate_pred == 0
                        else 3 + int(cu_event.argmax().item())
                    )
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
                        "soft_posterior_pred_id": pred_id,
                        "diagnostic_gate_pred_id": diagnostic_gate_pred,
                        "diagnostic_hierarchical_pred_id": diagnostic_action_pred,
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
            "unknown_target_is_ordinary_class": False if model.gate.config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION else True,
            "unknown_training_label_semantics": (
                "supervises CU Clutter-not-accepted branch; runtime UNKNOWN_REJECTED remains a distinct action"
                if model.gate.config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION
                else "historical closed-set CU class"
            ),
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
        "hierarchical_action": {
            "product_contract_available": model.gate.config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION,
            "diagnostic_argmax_only": True,
            "diagnostic_parent_metrics": summarize_records(
                [
                    {**row, "pred_id": int(row["diagnostic_hierarchical_pred_id"])}
                    for row in parent_records
                ],
                scope=f"{split_name}:diagnostic_hierarchical_argmax_parent_final",
            ),
            "calibrated_actions": "not_evaluated",
            "stable_actions": "not_evaluated_without_train_parent_grouped_oof_calibration",
            "coverage": "not_evaluated",
            "false_acceptance_rate": "not_evaluated",
            "false_rejection_rate": "not_evaluated",
            "selective_risk": "not_evaluated",
        },
        "records": {
            "window": window_records,
            "parent": parent_records,
            "stable": stable_records,
            "stable_unconfirmed": stable_unconfirmed,
        },
    }


def summarize_strict_open_set_known_records(
    records: list[dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    """Summarize four known classes plus one unresolved-action column."""

    matrix = np.zeros((4, 5), dtype=np.int64)
    for row in records:
        true_id = int(row["true_id"])
        pred_id = int(row["pred_id"])
        if not 0 <= true_id < 4 or not 0 <= pred_id < 5:
            raise ValueError("V14 known metric record is outside its 4x5 contract")
        matrix[true_id, pred_id] += 1
    support = matrix.sum(axis=1)
    predicted_known = matrix[:, :4].sum(axis=0)
    per_class: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_id, name in enumerate(LABELS[:4]):
        tp = int(matrix[class_id, class_id])
        precision = tp / int(predicted_known[class_id]) if predicted_known[class_id] else 0.0
        recall = tp / int(support[class_id]) if support[class_id] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        confusion = matrix[class_id].copy()
        confusion[class_id] = 0
        destination = int(confusion.argmax()) if int(confusion.sum()) else None
        per_class.append({
            "class": name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[class_id]),
            "unresolved_count": int(matrix[class_id, 4]),
            "major_confusion_to": (
                STRICT_OPEN_SET_EVIDENCE_NAMES[destination]
                if destination is not None and confusion[destination] > 0
                else None
            ),
            "major_confusion_count": int(confusion[destination]) if destination is not None else 0,
        })
    total = int(matrix.sum())
    accepted_count = int(matrix[:, :4].sum())
    return {
        "scope": scope,
        "support": total,
        "accepted_known_count": accepted_count,
        "unresolved_known_count": int(matrix[:, 4].sum()),
        "known_action_coverage": accepted_count / total if total else None,
        "accuracy_including_unresolved_as_error": float(np.trace(matrix[:, :4]) / total) if total else None,
        "accepted_accuracy": (
            float(np.trace(matrix[:, :4]) / accepted_count) if accepted_count else None
        ),
        "macro_precision": float(np.mean(precision_values)) if precision_values else None,
        "macro_recall": float(np.mean(recall_values)) if recall_values else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "per_class": per_class,
        "confusion_matrix_rows_known_true_columns_known_plus_unresolved": matrix.tolist(),
        "prediction_columns": list(STRICT_OPEN_SET_EVIDENCE_NAMES),
        "unresolved_is_unknown_class": False,
        "unresolved_semantics": "diagnostic known evidence not accepted as any known class; calibrated rejection not implied",
    }


def _strict_open_set_unknown_audit(
    records: list[dict[str, Any]],
    *,
    split_name: str,
) -> dict[str, Any]:
    if not records:
        return {
            "status": "not_in_scope",
            "parent_count": 0,
            "threshold_fitting_allowed": False,
        }

    def summary(field: str) -> dict[str, Any]:
        values = np.asarray([float(row[field]) for row in records], dtype=np.float64)
        return {
            "minimum": float(values.min()),
            "median": float(np.median(values)),
            "maximum": float(values.max()),
        }

    return {
        "status": "frozen_development_open_set_audit_only",
        "split": split_name,
        "parent_count": len(records),
        "gate_diagnostic_route_counts": {
            "DBB": sum(row["diagnostic_gate_route"] == "DBB" for row in records),
            "CU": sum(row["diagnostic_gate_route"] == "CU" for row in records),
        },
        "gate_dbb_probability": summary("gate_dbb_probability"),
        "dbb_top_probability": summary("dbb_known_confidence"),
        "dbb_top_margin": summary("dbb_known_margin"),
        "clutter_accept_probability": summary("clutter_accept_confidence"),
        "clutter_accept_margin": summary("clutter_accept_margin"),
        "threshold_fitting_allowed": False,
        "checkpoint_selection_allowed": False,
        "unknown_is_learned_class": False,
    }


def evaluate_strict_open_set_cascade(
    model: LiquidSoftCascade,
    dataset: CQ08PointSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    split_name: str,
    evaluation_scope: str,
    rejection_policy: StrictOpenSetRejectionPolicy | None = None,
    stable_confirm_count: int = 2,
) -> dict[str, Any]:
    """Evaluate V14 without allowing Unknown into any learned metric."""

    if model.gate.config.schema_version != STRICT_OPEN_SET_SCHEMA_VERSION:
        raise ValueError("strict open-set evaluator requires three V14 task models")
    if evaluation_scope not in {"known_only_checkpoint_selection", "development_open_set_audit"}:
        raise ValueError("unsupported V14 evaluation scope")
    if hasattr(dataset, "frame"):
        labels_in_dataset = set(dataset.frame["label_name"].astype(str))
    else:
        labels_in_dataset = {
            str(dataset[index].label_name)
            for index in range(len(dataset))
        }
    if evaluation_scope == "known_only_checkpoint_selection" and UNKNOWN_BASE_LABEL in labels_in_dataset:
        raise ValueError("UnknownTarget must not enter V14 checkpoint selection evaluation")
    if split_name == model.gate.config.data.test_split_name:
        raise PermissionError("V14 test evaluation is sealed")
    policy = rejection_policy or StrictOpenSetRejectionPolicy()
    if (
        policy.enabled
        and policy.calibration is not None
        and not policy.calibration.deployment_eligible
        and evaluation_scope != "development_open_set_audit"
    ):
        raise PermissionError(
            "V14 audit-only calibration is restricted to development_open_set_audit"
        )
    loader = build_point_dataloader(dataset, batch_size=batch_size, shuffle=False)
    known_windows: list[dict[str, Any]] = []
    known_parents: list[dict[str, Any]] = []
    unknown_windows: list[dict[str, Any]] = []
    unknown_parents: list[dict[str, Any]] = []
    gate_parents: list[dict[str, int]] = []
    dbb_parents: list[dict[str, int]] = []
    cu_acceptance_parents: list[dict[str, int]] = []
    calibrated_window_actions: list[dict[str, Any]] = []
    calibrated_parent_actions: list[dict[str, Any]] = []
    calibrated_accepted_parents: list[dict[str, Any]] = []
    stable_action_records: list[dict[str, Any]] = []
    stable_unconfirmed_records: list[dict[str, Any]] = []
    dt_counts = {"first_event": 0, "zero_delta": 0, "positive_delta": 0}
    collate_padding_positions = 0
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(
                tr_features=tensors["tr_features"],
                tr_missing_mask=tensors["tr_missing_mask"],
                tr_dt_seconds=tensors["tr_dt_seconds"],
                tr_dt_valid=tensors["tr_dt_valid"],
                tr_dt_reason=tensors["tr_dt_reason"],
                rd_frames=tensors["rd_frames"],
                rd_frame_mask=tensors["rd_frame_mask"],
                rd_quality=tensors["rd_quality"],
                tr_padding_mask=tensors["tr_padding_mask"],
            )
            reasons = tensors["tr_dt_reason"]
            collate_padding_positions += int(tensors["tr_padding_mask"].sum().item())
            dt_counts["first_event"] += int((reasons == 1).sum().item())
            dt_counts["zero_delta"] += int((reasons == 2).sum().item())
            dt_counts["positive_delta"] += int((reasons == 3).sum().item())
            gate_probs = torch.softmax(output["gate_sequence_logits"], dim=-1)
            dbb_probs = torch.softmax(output["dbb_sequence_logits"], dim=-1)
            cu_probs = torch.softmax(output["cu_sequence_logits"], dim=-1)
            evidence = output["sequence_open_set_evidence"]
            for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][batch_index].item())
                true_id = int(batch["labels"][batch_index].item())
                if not 0 <= true_id <= 4:
                    raise ValueError("V14 evaluator received an invalid frozen base label")
                known_final: dict[str, Any] | None = None
                unknown_final: dict[str, Any] | None = None
                action_final: dict[str, Any] | None = None
                stable_final: HierarchicalStableDecision | None = None
                stable_machine = HierarchicalStableStateMachine(confirm_count=stable_confirm_count)
                rd_observed_count = 0
                zero_delta_count = 0
                for event_index in range(length):
                    gate_event = gate_probs[batch_index, event_index]
                    dbb_event = dbb_probs[batch_index, event_index]
                    cu_event = cu_probs[batch_index, event_index]
                    audit_event = evidence[batch_index, event_index]
                    gate_pred = int(gate_event.argmax().item())
                    dbb_pred = int(dbb_event.argmax().item())
                    cu_pred = int(cu_event.argmax().item())
                    pred_id = dbb_pred if gate_pred == 0 else (3 if cu_pred == 0 else 4)
                    dbb_top = torch.topk(dbb_event, k=2).values
                    common = {
                        "true_id": true_id,
                        "parent_sample_id": parent_id,
                        "track_id": batch["track_ids"][batch_index],
                        "source_name": batch["source_names"][batch_index],
                        "label_name": batch["label_names"][batch_index],
                        "split_name": split_name,
                        "event_index": event_index,
                        "window_length": event_index + 1,
                        "diagnostic_gate_route": "DBB" if gate_pred == 0 else "CU",
                        "gate_dbb_probability": float(gate_event[0].item()),
                        "gate_margin": float((gate_event[0] - gate_event[1]).item()),
                        "dbb_known_confidence": float(dbb_event[dbb_pred].item()),
                        "dbb_known_margin": float((dbb_top[0] - dbb_top[1]).item()),
                        "clutter_accept_confidence": float(cu_event[0].item()),
                        "clutter_accept_margin": float((cu_event[0] - cu_event[1]).item()),
                        "audit_evidence": {
                            name: float(audit_event[index].item())
                            for index, name in enumerate(STRICT_OPEN_SET_EVIDENCE_NAMES)
                        },
                    }
                    if true_id < 4:
                        known_row = {
                            **common,
                            "pred_id": pred_id,
                            "pred_name": STRICT_OPEN_SET_EVIDENCE_NAMES[pred_id],
                            "unresolved": pred_id == 4,
                        }
                        known_windows.append(known_row)
                        known_final = known_row
                    else:
                        unknown_row = {
                            **common,
                            "pred_id": None,
                            "pred_name": None,
                            "unknown_is_evaluation_truth_only": True,
                        }
                        unknown_windows.append(unknown_row)
                        unknown_final = unknown_row
                    if policy.enabled:
                        rd_observed_count += int(float(tensors["rd_quality"][batch_index, event_index, 0].item()) >= 0.5)
                        zero_delta_count += int(int(tensors["tr_dt_reason"][batch_index, event_index].item()) == 2)
                        valid_points = event_index + 1
                        action = policy.decide(
                            gate_event.cpu(),
                            dbb_event.cpu(),
                            cu_event.cpu(),
                            valid_point_count=valid_points,
                            track_quality=rd_observed_count / valid_points,
                            gap_fraction=min(1.0, zero_delta_count / max(1, valid_points - 1)),
                        )
                        stable_final = stable_machine.update(action)
                        action_row = {
                            **common,
                            "pred_id": (
                                None
                                if action.decision_state == "pending"
                                else 4
                                if action.final_action == "UNKNOWN_REJECTED"
                                else int(action.accepted_label_id)
                            ),
                            "decision_state": action.decision_state,
                            "gate_action": action.gate_action,
                            "conditional_action": action.conditional_action,
                            "final_action": action.final_action,
                            "accepted_label_id": action.accepted_label_id,
                            "accepted_label_name": action.accepted_label_name,
                            "rejected_to_unknown": action.rejected_to_unknown,
                            "specialist_label_id": action.specialist_label_id,
                            "specialist_confidence": action.specialist_confidence,
                            "specialist_margin": action.specialist_margin,
                            "valid_point_count": action.valid_point_count,
                            "track_quality": action.track_quality,
                            "gap_fraction": action.gap_fraction,
                            "trigger_reasons": list(action.trigger_reasons),
                            "calibration_source": action.calibration_source,
                        }
                        calibrated_window_actions.append(action_row)
                        action_final = action_row
                if true_id < 4:
                    if known_final is None:
                        raise ValueError(f"empty V14 known parent sequence: {parent_id}")
                    known_parents.append(known_final)
                    final_index = length - 1
                    final_gate_pred = int(gate_probs[batch_index, final_index].argmax().item())
                    final_dbb_pred = int(dbb_probs[batch_index, final_index].argmax().item())
                    final_cu_pred = int(cu_probs[batch_index, final_index].argmax().item())
                    gate_parents.append({"true_id": 0 if true_id < 3 else 1, "pred_id": final_gate_pred})
                    cu_acceptance_parents.append({"true_id": 1 if true_id < 3 else 0, "pred_id": final_cu_pred})
                    if true_id < 3:
                        dbb_parents.append({"true_id": true_id, "pred_id": final_dbb_pred})
                else:
                    if unknown_final is None:
                        raise ValueError(f"empty V14 Unknown audit sequence: {parent_id}")
                    unknown_parents.append(unknown_final)
                if policy.enabled:
                    if action_final is None or stable_final is None:
                        raise RuntimeError("calibrated V14 evaluation lost its final action state")
                    calibrated_parent_actions.append(action_final)
                    if action_final["accepted_label_id"] is not None:
                        calibrated_accepted_parents.append(action_final)
                    stable_base = {
                        "true_id": true_id,
                        "parent_sample_id": parent_id,
                        "track_id": batch["track_ids"][batch_index],
                        "source_name": batch["source_names"][batch_index],
                        "label_name": batch["label_names"][batch_index],
                        "split_name": split_name,
                        "event_count": length,
                        "stable_state_name": stable_final.state_name,
                        "stable_confirmed": stable_final.stable_confirmed,
                        "stable_rejected_to_unknown": stable_final.stable_rejected_to_unknown,
                    }
                    if stable_final.stable_confirmed:
                        stable_action_records.append({
                            **stable_base,
                            "pred_id": 4 if stable_final.stable_rejected_to_unknown else int(stable_final.stable_label_id),
                            "accepted_label_id": stable_final.stable_label_id,
                        })
                    else:
                        stable_unconfirmed_records.append({
                            **stable_base,
                            "pred_id": None,
                            "accepted_label_id": None,
                            "candidate_action": stable_final.candidate_action,
                            "candidate_count": stable_final.candidate_count,
                        })

    known_metrics = summarize_strict_open_set_known_records(
        known_parents,
        scope=f"{split_name}:V14_known_parent_final",
    )
    gate_metrics = summarize_records_with_names(
        gate_parents,
        ("DBB_ACCEPT_EVIDENCE", "ROUTE_TO_CU_EVIDENCE"),
        "V14_gate_known_parent_final",
    )
    dbb_metrics = summarize_records_with_names(
        dbb_parents,
        LABELS[:3],
        "V14_dbb_known_parent_conditional",
    )
    cu_metrics = summarize_records_with_names(
        cu_acceptance_parents,
        ("CLUTTER_ACCEPT_EVIDENCE", "NON_CLUTTER_KNOWN_EVIDENCE"),
        "V14_clutter_acceptance_known_parent",
    )
    calibrated_summary = (
        summarize_calibrated_actions(
            calibrated_parent_actions,
            calibrated_accepted_parents,
            split_name=split_name,
            calibration_source=None if policy.calibration is None else policy.calibration.source,
        )
        if policy.enabled
        else {
            "status": "not_evaluated",
            "reason": "known-only train parent-grouped OOF calibration artifact was not supplied",
            "coverage": "not_evaluated",
            "false_acceptance_rate": "not_evaluated",
            "false_rejection_rate": "not_evaluated",
            "selective_risk": "not_evaluated",
        }
    )
    stable_summary = (
        summarize_stable_actions(
            stable_action_records,
            stable_unconfirmed_records,
            total_parent_count=len(known_parents) + len(unknown_parents),
            split_name=split_name,
        )
        if policy.enabled
        else {"status": "not_evaluated_without_calibration", "stable_coverage": "not_evaluated"}
    )
    checkpoint_inputs: dict[str, Any]
    if evaluation_scope == "known_only_checkpoint_selection":
        if unknown_parents:
            raise ValueError("V14 checkpoint inputs unexpectedly contain Unknown parents")
        checkpoint_inputs = {
            "status": "available_known_only",
            "unknown_parent_count": 0,
            "gate_min_precision_recall": minimum_precision_recall(gate_metrics),
            "dbb_min_precision_recall": minimum_precision_recall(dbb_metrics),
            "cu_min_precision_recall": minimum_precision_recall(cu_metrics),
            "known_parent_macro_f1": known_metrics["macro_f1"],
            "known_parent_drone_bird_min_precision_recall": minimum_precision_recall_for_names(
                known_metrics,
                ("DroneTarget", "BirdTarget"),
            ),
            "unknown_used_for_selection": False,
        }
    else:
        checkpoint_inputs = {
            "status": "forbidden_for_development_open_set_audit",
            "unknown_parent_count": len(unknown_parents),
            "unknown_used_for_selection": False,
        }
    return {
        "schema_version": STRICT_OPEN_SET_SCHEMA_VERSION,
        "split": split_name,
        "evaluation_scope": evaluation_scope,
        "dataset_parent_count": len(dataset),
        "known_parent_count": len(known_parents),
        "unknown_audit_parent_count": len(unknown_parents),
        "window_count": len(known_windows) + len(unknown_windows),
        "product_semantics": {
            "architecture": "strict_open_set_three_checkpoint_liquid_soft_cascade",
            "unknown_is_model_class": False,
            "unknown_training_exposure": 0,
            "gate": "accept DBB evidence or route to CU evidence",
            "dbb_specialist": "Drone/Bird/Balloon plus calibrated knownness acceptance",
            "cu_specialist": "Clutter acceptance versus non-Clutter-known evidence",
            "unresolved_open_set_evidence_is_unknown_probability": False,
            "unknown_output": "UNKNOWN_REJECTED action only after known-only OOF calibration",
        },
        "known_closed_set": {
            "window_metrics": summarize_strict_open_set_known_records(
                known_windows,
                scope=f"{split_name}:V14_known_causal_windows",
            ),
            "parent_metrics": known_metrics,
        },
        "component_metrics": {
            "gate_known_parent": gate_metrics,
            "dbb_known_parent_conditional": dbb_metrics,
            "clutter_acceptance_known_parent": cu_metrics,
        },
        "development_unknown_audit": _strict_open_set_unknown_audit(
            unknown_parents,
            split_name=split_name,
        ),
        "checkpoint_selection_inputs": checkpoint_inputs,
        "calibrated_hierarchical_action": calibrated_summary,
        "stable_action": stable_summary,
        "dt_reason_counts_data_events": dt_counts,
        "collate_padding_positions": collate_padding_positions,
        "evaluation_batch_size": int(batch_size),
        "session_metrics": {
            "status": "not_available",
            "reason": "current verified index has no session/capture field",
        },
        "records": {
            "known_window": known_windows,
            "known_parent": known_parents,
            "development_unknown_window": unknown_windows,
            "development_unknown_parent": unknown_parents,
            "calibrated_window_action": calibrated_window_actions,
            "calibrated_parent_action": calibrated_parent_actions,
            "calibrated_accepted_parent": calibrated_accepted_parents,
            "stable_action": stable_action_records,
            "stable_unconfirmed": stable_unconfirmed_records,
        },
    }


def evaluate_hierarchical_cascade(
    model: LiquidSoftCascade,
    dataset: CQ08PointSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    split_name: str,
    rejection_policy: HierarchicalRejectionPolicy | None = None,
    stable_confirm_count: int = 2,
) -> dict[str, Any]:
    """Evaluate V13 topology without pretending uncalibrated actions are reject."""

    if model.gate.config.schema_version != HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        raise ValueError("hierarchical evaluator requires three V13 task models")
    loader = build_point_dataloader(dataset, batch_size=batch_size, shuffle=False)
    soft_windows: list[dict[str, Any]] = []
    diagnostic_windows: list[dict[str, Any]] = []
    soft_parents: list[dict[str, Any]] = []
    diagnostic_parents: list[dict[str, Any]] = []
    gate_parents: list[dict[str, int]] = []
    dbb_parents: list[dict[str, int]] = []
    cu_parents: list[dict[str, int]] = []
    calibrated_window_actions: list[dict[str, Any]] = []
    calibrated_parent_actions: list[dict[str, Any]] = []
    calibrated_accepted_parents: list[dict[str, Any]] = []
    stable_action_records: list[dict[str, Any]] = []
    stable_unconfirmed_records: list[dict[str, Any]] = []
    dt_counts = {"first_event": 0, "zero_delta": 0, "positive_delta": 0}
    collate_padding_positions = 0
    policy = rejection_policy or HierarchicalRejectionPolicy()
    with torch.inference_mode():
        for batch in loader:
            tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(
                tr_features=tensors["tr_features"],
                tr_missing_mask=tensors["tr_missing_mask"],
                tr_dt_seconds=tensors["tr_dt_seconds"],
                tr_dt_valid=tensors["tr_dt_valid"],
                tr_dt_reason=tensors["tr_dt_reason"],
                rd_frames=tensors["rd_frames"],
                rd_frame_mask=tensors["rd_frame_mask"],
                rd_quality=tensors["rd_quality"],
                tr_padding_mask=tensors["tr_padding_mask"],
            )
            reasons = tensors["tr_dt_reason"]
            collate_padding_positions += int(tensors["tr_padding_mask"].sum().item())
            dt_counts["first_event"] += int((reasons == 1).sum().item())
            dt_counts["zero_delta"] += int((reasons == 2).sum().item())
            dt_counts["positive_delta"] += int((reasons == 3).sum().item())
            gate_probs = torch.softmax(output["gate_sequence_logits"], dim=-1)
            dbb_probs = torch.softmax(output["dbb_sequence_logits"], dim=-1)
            cu_probs = torch.softmax(output["cu_sequence_logits"], dim=-1)
            for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][batch_index].item())
                true_id = int(batch["labels"][batch_index].item())
                soft_final: dict[str, Any] | None = None
                diagnostic_final: dict[str, Any] | None = None
                action_final: dict[str, Any] | None = None
                rd_observed_count = 0
                zero_delta_count = 0
                stable_machine = HierarchicalStableStateMachine(confirm_count=stable_confirm_count)
                stable_final: HierarchicalStableDecision | None = None
                for event_index in range(length):
                    soft = output["sequence_probabilities"][batch_index, event_index]
                    soft_pred = int(soft.argmax().item())
                    gate_pred = int(gate_probs[batch_index, event_index].argmax().item())
                    if gate_pred == 0:
                        diagnostic_pred = int(dbb_probs[batch_index, event_index].argmax().item())
                        route_action = "DBB_ACCEPTED_DIAGNOSTIC"
                    else:
                        diagnostic_pred = 3 + int(cu_probs[batch_index, event_index].argmax().item())
                        route_action = (
                            "CLUTTER_ACCEPTED_DIAGNOSTIC"
                            if diagnostic_pred == 3
                            else "UNKNOWN_LABEL_DIAGNOSTIC_NOT_REJECTION"
                        )
                    common = {
                        "true_id": true_id,
                        "parent_sample_id": parent_id,
                        "track_id": batch["track_ids"][batch_index],
                        "source_name": batch["source_names"][batch_index],
                        "label_name": batch["label_names"][batch_index],
                        "split_name": split_name,
                        "event_index": event_index,
                        "window_length": event_index + 1,
                        "gate_pred_id": gate_pred,
                        "route_action": route_action,
                    }
                    soft_row = {
                        **common,
                        "pred_id": soft_pred,
                        "confidence": float(soft[soft_pred].item()),
                    }
                    diagnostic_probability = (
                        gate_probs[batch_index, event_index, 0] * dbb_probs[batch_index, event_index, diagnostic_pred]
                        if diagnostic_pred < 3
                        else gate_probs[batch_index, event_index, 1] * cu_probs[batch_index, event_index, diagnostic_pred - 3]
                    )
                    diagnostic_row = {
                        **common,
                        "pred_id": diagnostic_pred,
                        "confidence": float(diagnostic_probability.item()),
                    }
                    soft_windows.append(soft_row)
                    diagnostic_windows.append(diagnostic_row)
                    soft_final = soft_row
                    diagnostic_final = diagnostic_row
                    if policy.enabled:
                        rd_observed_count += int(float(tensors["rd_quality"][batch_index, event_index, 0].item()) >= 0.5)
                        zero_delta_count += int(int(tensors["tr_dt_reason"][batch_index, event_index].item()) == 2)
                        valid_point_count = event_index + 1
                        action = policy.decide(
                            gate_probs[batch_index, event_index].cpu(),
                            dbb_probs[batch_index, event_index].cpu(),
                            cu_probs[batch_index, event_index].cpu(),
                            valid_point_count=valid_point_count,
                            track_quality=rd_observed_count / valid_point_count,
                            gap_fraction=min(1.0, zero_delta_count / max(1, valid_point_count - 1)),
                        )
                        stable_final = stable_machine.update(action)
                        action_pred_id = (
                            None
                            if action.decision_state == "pending"
                            else 4
                            if action.final_action == "UNKNOWN_REJECTED"
                            else int(action.accepted_label_id)
                        )
                        action_row = {
                            **common,
                            "pred_id": action_pred_id,
                            "confidence": (
                                None
                                if action.decision_state == "pending"
                                else action.specialist_confidence
                            ),
                            "gate_action": action.gate_action,
                            "decision_state": action.decision_state,
                            "conditional_action": action.conditional_action,
                            "final_action": action.final_action,
                            "accepted_label_id": action.accepted_label_id,
                            "accepted_label_name": action.accepted_label_name,
                            "rejected_to_unknown": action.rejected_to_unknown,
                            "diagnostic_gate_route": action.diagnostic_gate_route,
                            "diagnostic_label_id": action.diagnostic_label_id,
                            "diagnostic_label_name": action.diagnostic_label_name,
                            "gate_dbb_probability": action.gate_dbb_probability,
                            "gate_margin": action.gate_margin,
                            "specialist_label_id": action.specialist_label_id,
                            "specialist_confidence": action.specialist_confidence,
                            "specialist_margin": action.specialist_margin,
                            "trigger_reasons": list(action.trigger_reasons),
                            "valid_point_count": action.valid_point_count,
                            "track_quality": action.track_quality,
                            "gap_fraction": action.gap_fraction,
                            "calibration_source": action.calibration_source,
                        }
                        calibrated_window_actions.append(action_row)
                        action_final = action_row
                if soft_final is None or diagnostic_final is None:
                    raise ValueError(f"empty V13 parent sequence: {parent_id}")
                soft_parents.append(soft_final)
                diagnostic_parents.append(diagnostic_final)
                if policy.enabled:
                    if action_final is None or stable_final is None:
                        raise RuntimeError("calibrated V13 evaluation lost its final action state")
                    calibrated_parent_actions.append(action_final)
                    if action_final["accepted_label_id"] is not None:
                        calibrated_accepted_parents.append(action_final)
                    stable_base = {
                        "true_id": true_id,
                        "parent_sample_id": parent_id,
                        "track_id": batch["track_ids"][batch_index],
                        "source_name": batch["source_names"][batch_index],
                        "label_name": batch["label_names"][batch_index],
                        "split_name": split_name,
                        "event_count": length,
                        "stable_state_name": stable_final.state_name,
                        "stable_confirmed": stable_final.stable_confirmed,
                        "stable_rejected_to_unknown": stable_final.stable_rejected_to_unknown,
                    }
                    if stable_final.stable_confirmed:
                        stable_action_records.append({
                            **stable_base,
                            "pred_id": 4 if stable_final.stable_rejected_to_unknown else int(stable_final.stable_label_id),
                            "accepted_label_id": stable_final.stable_label_id,
                        })
                    else:
                        stable_unconfirmed_records.append({
                            **stable_base,
                            "pred_id": None,
                            "accepted_label_id": None,
                            "candidate_action": stable_final.candidate_action,
                            "candidate_count": stable_final.candidate_count,
                        })
                gate_true = 0 if true_id < 3 else 1
                gate_parents.append({"true_id": gate_true, "pred_id": int(diagnostic_final["gate_pred_id"])})
                if true_id < 3:
                    dbb_parents.append({
                        "true_id": true_id,
                        "pred_id": int(dbb_probs[batch_index, length - 1].argmax().item()),
                    })
                else:
                    cu_parents.append({
                        "true_id": true_id - 3,
                        "pred_id": int(cu_probs[batch_index, length - 1].argmax().item()),
                    })
    gate_metrics = summarize_records_with_names(gate_parents, ("DBB", "CU"), "gate_parent_final")
    dbb_metrics = summarize_records_with_names(dbb_parents, LABELS[:3], "dbb_parent_conditional")
    cu_metrics = summarize_records_with_names(cu_parents, LABELS[3:], "cu_parent_conditional")
    diagnostic_metrics = summarize_records(
        diagnostic_parents,
        scope=f"{split_name}:diagnostic_hierarchical_argmax_parent_final",
    )
    soft_metrics = summarize_records(soft_parents, scope=f"{split_name}:soft_posterior_parent_final")
    calibrated_summary = summarize_calibrated_actions(
        calibrated_parent_actions,
        calibrated_accepted_parents,
        split_name=split_name,
        calibration_source=(
            None if policy.calibration is None else policy.calibration.source
        ),
    ) if policy.enabled else {
        "status": "not_evaluated",
        "reason": "train parent-grouped OOF calibration artifact was not supplied",
        "coverage": "not_evaluated",
        "accepted_accuracy": "not_evaluated",
        "accepted_macro_f1": "not_evaluated",
        "false_acceptance_rate": "not_evaluated",
        "false_rejection_rate": "not_evaluated",
        "selective_risk": "not_evaluated",
        "unknown_rejection_recall": "not_evaluated",
    }
    stable_summary = summarize_stable_actions(
        stable_action_records,
        stable_unconfirmed_records,
        total_parent_count=len(diagnostic_parents),
        split_name=split_name,
    ) if policy.enabled else {
        "status": "not_evaluated_without_calibration",
        "stable_coverage": "not_evaluated",
    }
    return {
        "schema_version": HIERARCHICAL_CASCADE_SCHEMA_VERSION,
        "split": split_name,
        "class_names": list(LABELS),
        "dataset_parent_count": len(dataset),
        "window_count": len(diagnostic_windows),
        "parent_count": len(diagnostic_parents),
        "product_semantics": {
            "architecture": "three_layer_soft_cascade_with_separate_hierarchical_action",
            "gate": "accept DBB or route to CU",
            "dbb_specialist": "Drone/Bird/Balloon conditional classification",
            "cu_specialist": "accept Clutter; UnknownTarget training label is not a calibrated reject action",
            "cu_output_semantics": ["CLUTTER_ACCEPT_EVIDENCE", "CLUTTER_REJECT_EVIDENCE"],
            "soft_posterior": "audit probability only",
            "soft_posterior_used_for_checkpoint_selection": False,
        },
        "diagnostic_hierarchical_argmax": {
            "status": "executed_without_thresholds",
            "is_validated_rejection": False,
            "window_metrics": summarize_records(
                diagnostic_windows,
                scope=f"{split_name}:diagnostic_hierarchical_argmax_windows",
            ),
            "parent_metrics": diagnostic_metrics,
            "source_metrics": grouped_summaries(
                diagnostic_parents,
                field="source_name",
                scope_prefix=f"{split_name}:diagnostic_hierarchical_parent_source",
            ),
        },
        "soft_posterior": {
            "window_metrics": summarize_records(soft_windows, scope=f"{split_name}:soft_posterior_windows"),
            "parent_metrics": soft_metrics,
        },
        "component_metrics": {
            "gate_parent": gate_metrics,
            "dbb_parent_conditional": dbb_metrics,
            "cu_parent_conditional": cu_metrics,
        },
        "checkpoint_selection_inputs": {
            "gate_min_precision_recall": minimum_precision_recall(gate_metrics),
            "dbb_min_precision_recall": minimum_precision_recall(dbb_metrics),
            "cu_min_precision_recall": minimum_precision_recall(cu_metrics),
            "diagnostic_parent_macro_f1": diagnostic_metrics["macro_f1"],
            "soft_parent_macro_f1": soft_metrics["macro_f1"],
        },
        "calibrated_hierarchical_action": calibrated_summary,
        "stable_action": stable_summary,
        "dt_reason_counts_data_events": dt_counts,
        "collate_padding_positions": collate_padding_positions,
        "evaluation_batch_size": int(batch_size),
        "session_metrics": {
            "status": "not_available",
            "reason": "current verified index has no session/capture field",
        },
        "records": {
            "window": diagnostic_windows,
            "parent": diagnostic_parents,
            "soft_window": soft_windows,
            "soft_parent": soft_parents,
            "calibrated_window_action": calibrated_window_actions,
            "calibrated_parent_action": calibrated_parent_actions,
            "calibrated_accepted_parent": calibrated_accepted_parents,
            "stable_action": stable_action_records,
            "stable_unconfirmed": stable_unconfirmed_records,
        },
    }


def summarize_calibrated_actions(
    parent_actions: list[dict[str, Any]],
    accepted_actions: list[dict[str, Any]],
    *,
    split_name: str,
    calibration_source: str | None,
) -> dict[str, Any]:
    total = len(parent_actions)
    if total == 0:
        raise ValueError("calibrated action summary received no parents")
    if any("decision_state" not in row for row in parent_actions):
        raise ValueError("calibrated action records lack decision_state")
    pending_rows = [row for row in parent_actions if row["decision_state"] == "pending"]
    decided_rows = [row for row in parent_actions if row["decision_state"] != "pending"]
    unknown_rows = [row for row in parent_actions if int(row["true_id"]) == 4]
    known_rows = [row for row in parent_actions if int(row["true_id"]) < 4]
    unknown_rejected = sum(
        int(row["final_action"] == "UNKNOWN_REJECTED") for row in unknown_rows
    )
    unknown_false_accepted = sum(
        int(row["accepted_label_id"] is not None) for row in unknown_rows
    )
    known_false_rejected = sum(
        int(row["accepted_label_id"] is None) for row in known_rows
    )
    rejected_rows = [row for row in parent_actions if row["final_action"] == "UNKNOWN_REJECTED"]
    if len(accepted_actions) + len(rejected_rows) + len(pending_rows) != total:
        raise ValueError("accepted/rejected/pending action counts do not partition all parents")
    accepted_metrics = (
        _summarize_accepted_known_actions(
            accepted_actions,
            scope=f"{split_name}:accepted_parent_actions_known_classes",
        )
        if accepted_actions
        else None
    )
    accepted_accuracy = None if accepted_metrics is None else accepted_metrics["accuracy"]
    action_metrics = summarize_records(
        decided_rows,
        scope=f"{split_name}:hierarchical_parent_actions_reject_mapped_to_id4",
    ) if decided_rows else None
    drone_bird_metrics = _summarize_drone_bird_safety_metrics(parent_actions)
    return {
        "status": "evaluated_with_train_parent_grouped_oof_calibration",
        "calibration_source": calibration_source,
        "parent_count": total,
        "decided_parent_count": len(decided_rows),
        "pending_parent_count": len(pending_rows),
        "action_coverage": len(decided_rows) / total,
        "accepted_parent_count": len(accepted_actions),
        "rejected_parent_count": len(rejected_rows),
        "coverage": len(accepted_actions) / total,
        "accepted_accuracy": accepted_accuracy,
        "accepted_macro_precision": None if accepted_metrics is None else accepted_metrics["macro_precision"],
        "accepted_macro_recall": None if accepted_metrics is None else accepted_metrics["macro_recall"],
        "accepted_macro_f1": None if accepted_metrics is None else accepted_metrics["macro_f1"],
        "accepted_metrics": accepted_metrics,
        "action_metrics_reject_mapped_to_unknown_id": action_metrics,
        "action_mapping_warning": "pred_id=4 denotes UNKNOWN_REJECTED action, not ordinary closed-set Unknown prediction",
        "false_acceptance_rate": (
            unknown_false_accepted / len(unknown_rows) if unknown_rows else None
        ),
        "false_rejection_rate": (
            known_false_rejected / len(known_rows) if known_rows else None
        ),
        "selective_risk": (
            None if accepted_accuracy is None else 1.0 - float(accepted_accuracy)
        ),
        "unknown_rejection_recall": (
            unknown_rejected / len(unknown_rows) if unknown_rows else None
        ),
        "unknown_support": len(unknown_rows),
        "known_support": len(known_rows),
        "rate_denominator_policy": "FAR uses all true Unknown parents; FRR uses all true known parents and conservatively treats final pending/non-acceptance as false rejection; pending is also reported separately",
        "unknown_false_accept_count": unknown_false_accepted,
        "known_false_reject_count": known_false_rejected,
        "drone_bird_action_metrics": drone_bird_metrics,
    }


def _summarize_drone_bird_safety_metrics(
    parent_actions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Score Drone/Bird acceptance against every parent and every outcome."""

    rows: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_id, class_name in ((0, "DroneTarget"), (1, "BirdTarget")):
        support = sum(int(int(row["true_id"]) == class_id) for row in parent_actions)
        predicted = sum(
            int(row.get("accepted_label_id") == class_id) for row in parent_actions
        )
        true_positive = sum(
            int(
                int(row["true_id"]) == class_id
                and row.get("accepted_label_id") == class_id
            )
            for row in parent_actions
        )
        pending_or_rejected = sum(
            int(
                int(row["true_id"]) == class_id
                and row.get("accepted_label_id") is None
            )
            for row in parent_actions
        )
        accepted_as_other = support - true_positive - pending_or_rejected
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        rows.append({
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted_count": predicted,
            "true_positive_count": true_positive,
            "pending_or_rejected_count": pending_or_rejected,
            "accepted_as_other_class_count": accepted_as_other,
        })
    return {
        "scope": "hierarchical_drone_bird_safety_all_parent_actions",
        "per_class": rows,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "precision_denominator": "all parents accepted as Drone/Bird, including false accepts from every other class",
        "recall_denominator": "all true Drone/Bird parents; pending, rejected and accepted-as-other are false negatives",
    }


def _summarize_accepted_known_actions(
    records: list[dict[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    """Score accepted outputs over the four classes that can be accepted.

    True Unknown parents accepted as a known class remain false positives in
    that known class's precision and remain errors in accepted accuracy.  They
    do not create a fifth accepted class with zero support, because the product
    has no ``UNKNOWN_ACCEPTED`` action.
    """

    if not records:
        raise ValueError("accepted known-action summary received no records")
    known_count = 4
    matrix = np.zeros((known_count, known_count), dtype=np.int64)
    unknown_false_accepts = np.zeros((known_count,), dtype=np.int64)
    correct = 0
    for row in records:
        true_id = int(row["true_id"])
        pred_id = int(row["pred_id"])
        if not 0 <= pred_id < known_count:
            raise ValueError("accepted product action predicted a non-accepted class")
        if 0 <= true_id < known_count:
            matrix[true_id, pred_id] += 1
            correct += int(true_id == pred_id)
        elif true_id == 4:
            unknown_false_accepts[pred_id] += 1
        else:
            raise ValueError("accepted product action contains an invalid true class")
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0) + unknown_false_accepts
    per_class: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(LABELS[:known_count]):
        tp = int(matrix[class_id, class_id])
        precision = tp / int(predicted[class_id]) if predicted[class_id] else 0.0
        recall = tp / int(support[class_id]) if support[class_id] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        confusion = matrix[class_id].copy()
        confusion[class_id] = 0
        destination = int(confusion.argmax()) if int(confusion.sum()) else None
        per_class.append({
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[class_id]),
            "predicted_count_including_unknown_false_accepts": int(predicted[class_id]),
            "unknown_false_accept_count": int(unknown_false_accepts[class_id]),
            "major_confusion_to": (
                LABELS[destination]
                if destination is not None and int(confusion[destination]) > 0
                else None
            ),
            "major_confusion_count": int(confusion[destination]) if destination is not None else 0,
        })
    return {
        "scope": scope,
        "accepted_parent_count": len(records),
        "accepted_known_true_parent_count": int(matrix.sum()),
        "accepted_unknown_false_accept_count": int(unknown_false_accepts.sum()),
        "accuracy": correct / len(records),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_denominator_classes": list(LABELS[:known_count]),
        "unknown_is_not_an_accepted_class": True,
        "per_class": per_class,
        "confusion_matrix_known_true_rows_known_pred_columns": matrix.tolist(),
        "unknown_false_accepts_by_predicted_known_class": unknown_false_accepts.tolist(),
    }


def summarize_stable_actions(
    stable_actions: list[dict[str, Any]],
    unconfirmed: list[dict[str, Any]],
    *,
    total_parent_count: int,
    split_name: str,
) -> dict[str, Any]:
    if total_parent_count <= 0:
        raise ValueError("stable action summary requires a positive parent count")
    if len(stable_actions) + len(unconfirmed) != total_parent_count:
        raise ValueError("stable action confirmed/unconfirmed counts do not cover all parents")
    accepted = [row for row in stable_actions if row["accepted_label_id"] is not None]
    rejected = [row for row in stable_actions if row["stable_rejected_to_unknown"]]
    return {
        "status": "evaluated_with_calibrated_hierarchical_state_machine",
        "parent_count": total_parent_count,
        "stable_confirmed_parent_count": len(stable_actions),
        "stable_unconfirmed_parent_count": len(unconfirmed),
        "stable_action_coverage": len(stable_actions) / total_parent_count,
        "stable_accepted_coverage": len(accepted) / total_parent_count,
        "stable_rejected_count": len(rejected),
        "stable_action_metrics_reject_mapped_to_unknown_id": (
            summarize_records(
                stable_actions,
                scope=f"{split_name}:stable_hierarchical_actions_reject_mapped_to_id4",
            )
            if stable_actions
            else None
        ),
        "action_mapping_warning": "pred_id=4 denotes stable UNKNOWN_REJECTED action",
    }


def minimum_precision_recall(metrics: Mapping[str, Any]) -> float:
    rows = list(metrics.get("per_class") or [])
    if not rows or any(int(row.get("support", 0)) <= 0 for row in rows):
        raise ValueError("product checkpoint floor requires nonzero support for every class")
    values = [
        float(value)
        for row in rows
        for value in (row.get("precision"), row.get("recall"))
    ]
    if not values or any(not torch.isfinite(torch.tensor(value)).item() for value in values):
        raise ValueError("product checkpoint floor contains a non-finite metric")
    return min(values)


def minimum_precision_recall_for_names(
    metrics: Mapping[str, Any],
    names: tuple[str, ...],
) -> float:
    rows = {str(row.get("class")): row for row in metrics.get("per_class") or []}
    values: list[float] = []
    for name in names:
        row = rows.get(name)
        if row is None or int(row.get("support", 0)) <= 0:
            raise ValueError(f"checkpoint floor lacks supported class {name}")
        values.extend((float(row["precision"]), float(row["recall"])))
    if not values or any(not np.isfinite(value) for value in values):
        raise ValueError("checkpoint floor contains a non-finite precision/recall")
    return min(values)


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
    "build_cascade",
    "evaluate_cascade",
    "evaluate_hierarchical_cascade",
    "evaluate_strict_open_set_cascade",
    "load_cascade",
    "minimum_precision_recall",
    "minimum_precision_recall_for_names",
    "summarize_strict_open_set_known_records",
    "summarize_calibrated_actions",
    "summarize_stable_actions",
    "validate_cascade_configs",
    "verify_cascade_data_contracts",
]
