"""Full three-layer radar_ln3 soft-cascade model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from radar_ln3.checkpoint import Checkpoint, load_model_state, sha256_file
from radar_ln3.decision.l3 import load_l3_calibration
from radar_ln3.model import build_l1_model, build_l2_model, load_model_config
from radar_ln3.model.io import (
    FULL_CASCADE_INPUTS,
    L1_MODEL_INPUTS,
    L2_MODEL_INPUTS,
    select_inputs,
)
from radar_ln3.model.l3.knownness import (
    ProfileResidualKnownnessHead,
    TemporalKnownnessHead,
    build_profile_knownness_features,
)
from radar_ln3.training.l3_heads import load_l3_knownness_head


FULL_CASCADE_INPUT_NAMES = FULL_CASCADE_INPUTS

CASCADE_ACTION_NAMES = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "Clutter",
    "UNKNOWN_REJECTED",
    "PENDING_ABSTAIN",
)
CASCADE_REASON_NAMES = (
    "KNOWN_DBB_CLASS_ACCEPTED",
    "L1_CLUTTER_ACCEPTED",
    "UNKNOWN_REJECTED",
    "PENDING_ABSTAIN",
    "L1_KNOWNNESS_BELOW_THRESHOLD",
    "L2_KNOWNNESS_BELOW_THRESHOLD",
    "L1_CONFORMAL_SET_EMPTY",
    "L2_CLASS_AMBIGUOUS",
    "L1_ROUTE_AMBIGUOUS",
    "L2_EVIDENCE_MISSING",
    "L1_CLUTTER_BIRD_RECOVERY",
    "L1_TARGET_BIRD_RECOVERY",
    "L2_CONFORMAL_SET_EMPTY",
    "L2_ROUTE_CONFLICT",
)


def _softmax(logits: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits / temperature, dim=-1)


def _gather_last(logits: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    lengths = (~padding).to(dtype=torch.long).sum(dim=1).clamp_min(1)
    index = (lengths - 1).view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
    return logits.gather(1, index).squeeze(1)


def _rd_embedding_export(base: Any, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor) -> torch.Tensor:
    encoder = base.rd_encoder
    flat = rd_frames.reshape(-1, 2, encoder.rd_height, encoder.rd_width)
    embedded = encoder.projection(encoder.network(flat)).reshape(
        rd_frames.shape[0], rd_frames.shape[1], rd_frames.shape[2], -1
    )
    mask = rd_frame_mask.to(dtype=rd_frames.dtype).unsqueeze(-1)
    return (embedded * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)


def _liquid_sequence_export(
    base: Any,
    events: torch.Tensor,
    inputs: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cell = base.liquid.cell
    padding = inputs["tr_padding_mask"]
    dt_seconds = inputs["tr_dt_seconds"]
    dt_valid = inputs["tr_dt_valid"]
    dt_reason = inputs["tr_dt_reason"]
    state = events.new_zeros((events.shape[0], cell.hidden_dim))
    zero_ordinal = torch.ones_like(dt_reason, dtype=torch.long)
    states: list[torch.Tensor] = []
    trace_values: dict[str, list[torch.Tensor]] = {
        name: [] for name in (
            "candidate_innovation", "mixed_memory_innovation", "actual_update",
            "gate", "alpha", "tau", "zero_ordinal",
        )
    }
    for event_index in range(int(events.shape[1])):
        active = ~padding[:, event_index]
        active_long = active.to(dtype=torch.long)
        active_float = active.to(dtype=dt_seconds.dtype)
        reason = active_long * dt_reason[:, event_index] + (1 - active_long)
        dt = active_float * dt_seconds[:, event_index]
        valid = active & dt_valid[:, event_index]
        previous = zero_ordinal[:, event_index - 1] if event_index else None
        previous_value = previous + 1 if previous is not None else torch.ones_like(reason)
        zero_flag = (reason == 2).to(dtype=torch.long)
        current = zero_flag * previous_value + (1 - zero_flag)
        step_ordinal = active_long * current + (1 - active_long)
        zero_ordinal[:, event_index] = active_long * current
        normalized = cell.compute_norm(state)
        combined = torch.cat((events[:, event_index], normalized), dim=-1)
        candidate = torch.tanh(
            cell.candidate_projection(combined)
            + cell.input_projection(events[:, event_index])
            + cell.state_projection(normalized)
        )
        gate = torch.sigmoid(cell.gate_projection(combined))
        mixed = gate * candidate + (1.0 - gate) * state
        tau = cell.tau(dtype=dt.dtype, device=dt.device).unsqueeze(0).expand_as(state)
        safe_dt = dt.clamp(min=0.0, max=cell.dt_max_seconds)
        floor = (safe_dt <= 1.0e-9) | (~valid)
        safe_dt = torch.where(
            floor,
            torch.full_like(safe_dt, cell.zero_dt_event_floor_seconds),
            safe_dt,
        )
        alpha = 1.0 - torch.exp(-safe_dt.unsqueeze(-1) / tau)
        invalid_gain = torch.full_like(alpha, cell.invalid_dt_observation_gain)
        alpha = valid.to(dtype=alpha.dtype).unsqueeze(-1) * alpha + (~valid).to(dtype=alpha.dtype).unsqueeze(-1) * invalid_gain
        first = (reason == 1).to(dtype=alpha.dtype).unsqueeze(-1)
        zero = (reason == 2).to(dtype=alpha.dtype).unsqueeze(-1)
        alpha = first * torch.ones_like(alpha) + (1.0 - first) * alpha
        alpha = (zero * invalid_gain + (1.0 - zero) * alpha).clamp(0.0, 1.0)
        next_state = (1.0 - alpha) * state + alpha * mixed
        candidate_innovation = candidate - state
        mixed_innovation = mixed - state
        actual_update = next_state - state
        trace_active = active_float.unsqueeze(-1)
        trace_values["candidate_innovation"].append(candidate_innovation * trace_active)
        trace_values["mixed_memory_innovation"].append(mixed_innovation * trace_active)
        trace_values["actual_update"].append(actual_update * trace_active)
        trace_values["gate"].append(gate * trace_active)
        trace_values["alpha"].append(alpha * trace_active)
        trace_values["tau"].append(tau * trace_active)
        trace_values["zero_ordinal"].append(step_ordinal)
        state = active_float.unsqueeze(-1) * next_state + (1.0 - active_float.unsqueeze(-1)) * state
        states.append(state)
    return torch.stack(states, dim=1), {
        name: torch.stack(values, dim=1) for name, values in trace_values.items()
    }


def _classify_sequence(base: Any, sequence: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    logits = base.classifier(base.classifier_norm(sequence)).masked_fill(padding.unsqueeze(-1), 0.0)
    return logits, _gather_last(logits, padding)


def _base_l1_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return select_inputs(inputs, L1_MODEL_INPUTS)


def _base_l2_inputs(inputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return select_inputs(inputs, L2_MODEL_INPUTS)


class FullCascadeModel(nn.Module):
    """L1 + L2 + the current independent-head L3 policy as one graph."""

    def __init__(
        self,
        l1: nn.Module,
        l2: nn.Module,
        l1_knownness: nn.Module,
        l2_knownness: nn.Module,
        calibration: Mapping[str, Any],
    ) -> None:
        super().__init__()
        self.l1 = l1
        self.l2 = l2
        self.l1_knownness = l1_knownness
        self.l2_knownness = l2_knownness
        self.l1_knownness.eval()
        self.l2_knownness.eval()
        self.register_buffer(
            "l1_temperature",
            torch.tensor(float(calibration["temperature"]["l1"]["temperature"])),
        )
        self.register_buffer(
            "l2_temperature",
            torch.tensor(float(calibration["temperature"]["l2"]["temperature"])),
        )
        l1_thresholds = calibration["knownness_thresholds"]["l1"]
        l1_target_override = calibration.get("l1_target_knownness_override")
        l1_target_threshold = float(l1_thresholds["target"]["selected"]["threshold"])
        if l1_target_override is not None:
            l1_target_threshold = float(l1_target_override["threshold"])
        self.register_buffer(
            "l1_knownness_thresholds",
            torch.tensor(
                [
                    l1_target_threshold,
                    float(l1_thresholds["clutter"]["selected"]["threshold"]),
                ]
            ),
        )
        l2_thresholds = calibration["knownness_thresholds"]["l2"]
        self.register_buffer(
            "l2_knownness_thresholds",
            torch.tensor(
                [
                    float(l2_thresholds["DroneTarget"]["selected"]["threshold"]),
                    float(l2_thresholds["BirdTarget"]["selected"]["threshold"]),
                    float(l2_thresholds["BalloonTarget"]["selected"]["threshold"]),
                ]
            ),
        )
        l1_floors = dict(
            calibration["conformal"]["l1"]["candidate_probability_floor_by_class"]
        )
        override = calibration.get("l1_target_conformal_floor_override")
        if override is not None:
            l1_floors["KNOWN_TARGET_EVIDENCE"] = min(
                float(l1_floors["KNOWN_TARGET_EVIDENCE"]),
                float(override["floor"]),
            )
        l2_floors = calibration["conformal"]["l2"]["candidate_probability_floor_by_class"]
        self.register_buffer(
            "l1_candidate_floors",
            torch.tensor(
                [
                    float(l1_floors["KNOWN_TARGET_EVIDENCE"]),
                    float(l1_floors["KNOWN_CLUTTER_EVIDENCE"]),
                ]
            ),
        )
        self.register_buffer(
            "l2_candidate_floors",
            torch.tensor(
                [
                    float(l2_floors["DroneTarget"]),
                    float(l2_floors["BirdTarget"]),
                    float(l2_floors["BalloonTarget"]),
                ]
            ),
        )
        margins = calibration.get("l2_top2_margin_thresholds") or {}
        self.register_buffer(
            "l2_margin_thresholds",
            torch.tensor(
                [
                    float(margins.get("DroneTarget", 1.0)),
                    float(margins.get("BirdTarget", 1.0)),
                    float(margins.get("BalloonTarget", 1.0)),
                ]
            ),
        )
        bird_gate = calibration.get("bird_recovery_gate") or {}
        self.register_buffer(
            "bird_gate_values",
            torch.tensor(
                [
                    float(bird_gate.get("knownness_threshold", 1.0)),
                    float(bird_gate.get("probability_floor", 1.0)),
                    float(bird_gate.get("margin_floor", 1.0)),
                    1.0 if bird_gate.get("enabled", False) else 0.0,
                ]
            ),
        )
        clutter_gate = calibration.get("l1_clutter_bird_recovery_gate") or {}
        self.register_buffer(
            "clutter_bird_gate_values",
            torch.tensor(
                [
                    float(clutter_gate.get("l1_target_score_min", 0.0)),
                    float(clutter_gate.get("l1_target_score_max", 0.0)),
                    float(clutter_gate.get("l2_bird_probability_floor", 1.0)),
                    float(clutter_gate.get("l2_bird_knownness_threshold", 1.0)),
                    1.0 if clutter_gate.get("enabled", False) else 0.0,
                ]
            ),
        )
        target_gate = calibration.get("l1_target_bird_recovery_gate") or {}
        self.register_buffer(
            "target_bird_gate_values",
            torch.tensor(
                [
                    float(target_gate.get("l1_target_score_min", 0.0)),
                    float(target_gate.get("l1_target_score_max", 0.0)),
                    float(target_gate.get("l2_bird_probability_floor", 1.0)),
                    float(target_gate.get("l2_bird_knownness_threshold", 1.0)),
                    float(target_gate.get("l2_bird_margin_floor", 1.0)),
                    1.0 if target_gate.get("enabled", False) else 0.0,
                ]
            ),
        )

    def _knownness_scores(
        self,
        *,
        l1_sequence: torch.Tensor,
        l1_trace: Mapping[str, torch.Tensor],
        l1_inputs: Mapping[str, torch.Tensor],
        l2_sequence: torch.Tensor,
        l2_trace: Mapping[str, torch.Tensor],
        l2_inputs: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """运行迁移后的 L1/L2 knownness heads，且只使用因果表示。"""

        def score(
            task: str,
            base: Any,
            head: nn.Module,
            sequence: torch.Tensor,
            trace: Mapping[str, torch.Tensor],
            values: Mapping[str, torch.Tensor],
        ) -> torch.Tensor:
            padding = values["tr_padding_mask"]
            physical_valid = values.get("rd_physical_valid")
            micro_valid = values.get("micro_doppler_valid")
            course_valid = values.get("course_valid")
            task_facts = base._task_facts_sequence(
                padding=padding,
                rd_frame_mask=values["rd_frame_mask"],
                rd_quality=values["rd_quality"],
                dt_valid=values["tr_dt_valid"],
                dt_reason=values["tr_dt_reason"],
                physical_valid=physical_valid,
                micro_valid=micro_valid,
                course_valid=course_valid,
            )
            event_facts = base._trajectory_event_facts(
                padding=padding,
                rd_frame_mask=values["rd_frame_mask"],
                rd_quality=values["rd_quality"],
                dt_seconds=values["tr_dt_seconds"],
                dt_valid=values["tr_dt_valid"],
                dt_reason=values["tr_dt_reason"],
                physical_valid=physical_valid,
                micro_valid=micro_valid,
                course_features=values.get("course_features"),
                course_valid=course_valid,
                course_support=values.get("course_support"),
                zero_ordinal=trace["zero_ordinal"],
            )
            features = base.trajectory_heads.build_representation_features(
                raw_sequence=sequence,
                trace=trace,
                padding_mask=padding,
                task_facts_sequence=task_facts,
                event_facts=event_facts,
            )
            if task == "l1":
                extra = torch.zeros(
                    (*sequence.shape[:2], 32), dtype=sequence.dtype, device=sequence.device
                )
            else:
                extra = torch.cat(
                    (
                        values["rd_physical_features"],
                        values["rd_physical_valid"].unsqueeze(-1).to(dtype=sequence.dtype),
                        values["micro_doppler_features"],
                        values["micro_doppler_valid"].unsqueeze(-1).to(dtype=sequence.dtype),
                        values["stable_summary_tokens"],
                        values["stable_summary_valid"].unsqueeze(-1).to(dtype=sequence.dtype),
                        values["motion_values"],
                    ),
                    dim=-1,
                )
            representation = torch.cat(
                (
                    features["summary_features"],
                    features["event_trajectory"],
                    features["task_facts_sequence"],
                    extra,
                ),
                dim=-1,
            ).masked_fill(padding.unsqueeze(-1), 0.0)
            if isinstance(head, ProfileResidualKnownnessHead):
                profile = build_profile_knownness_features(
                    values["rd_physical_features"],
                    values["rd_physical_valid"],
                    values["micro_doppler_features"],
                    values["micro_doppler_valid"],
                    padding,
                )
                output = head(representation, profile, padding)
            elif isinstance(head, TemporalKnownnessHead):
                output = head(representation, padding)
            else:
                raise TypeError("L3 knownness head 类型不受当前 product graph 支持")
            lengths = (~padding).long().sum(dim=1)
            rows = output["route_probabilities"][
                torch.arange(sequence.shape[0], device=sequence.device), lengths - 1
            ]
            if task == "l2" and rows.shape[-1] == 1:
                rows = rows.expand(-1, 3)
            return rows

        l1_scores = score(
            "l1", self.l1.base, self.l1_knownness, l1_sequence, l1_trace, l1_inputs
        )
        l2_scores = score(
            "l2",
            self.l2.trainable_base_model,
            self.l2_knownness,
            l2_sequence,
            l2_trace,
            l2_inputs,
        )
        return l1_scores, l2_scores

    def _stable_residual_export(
        self,
        baseline_logits: torch.Tensor,
        summary_tokens: torch.Tensor,
        summary_valid: torch.Tensor,
        frame_values: torch.Tensor,
        frame_mask: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        """ONNX-friendly equivalent of EvidenceResidual.forward.

        The training implementation uses selected-row and packed-token paths.
        Those paths are correct in PyTorch but trace poorly when batch size or
        reliable-frame counts change. The export path keeps the same GRU and
        projection weights and gathers the last reliable token with tensors.
        """

        stable = self.l2.stable_model
        batch, length, frame_count, evidence_dim = frame_values.shape
        state = baseline_logits.new_zeros((batch, stable.hidden_dim))
        previous = baseline_logits.new_zeros((batch, stable.class_count))
        rows: list[torch.Tensor] = []
        for event_index in range(int(length)):
            active = ~padding[:, event_index]
            if stable.arm == "mechanism":
                tokens = summary_tokens[:, event_index].unsqueeze(1)
                valid = summary_valid[:, event_index]
            else:
                tokens = frame_values[:, event_index]
                valid = frame_mask[:, event_index].any(dim=-1)
            projected = stable.token_projection(tokens)
            if stable.config.model.token_activation == "silu":
                projected = torch.nn.functional.silu(projected)
            encoded, _ = stable.frame_encoder(projected)
            counts = (
                frame_mask[:, event_index].sum(dim=-1)
                if stable.arm != "mechanism"
                else torch.ones((batch,), dtype=torch.long, device=frame_values.device)
            ).clamp_min(1).clamp_max(tokens.shape[1])
            gather_index = (counts - 1).view(-1, 1, 1).expand(-1, 1, encoded.shape[-1])
            event_representation = encoded.gather(1, gather_index).squeeze(1)
            event_representation = event_representation * valid.to(dtype=event_representation.dtype).unsqueeze(-1)
            event_representation = event_representation * active.to(dtype=event_representation.dtype).unsqueeze(-1)
            _, next_hidden = stable.parent_encoder(
                event_representation.unsqueeze(1), state.unsqueeze(0)
            )
            candidate_state = next_hidden[-1]
            selected = active & valid
            state = torch.where(selected.unsqueeze(-1), candidate_state, state)
            raw = stable.residual_projection(state)
            centered = raw - raw.mean(dim=-1, keepdim=True)
            maximum = centered.abs().amax(dim=-1, keepdim=True)
            multiplier = torch.clamp(
                float(stable.residual_bound)
                / maximum.clamp_min(torch.finfo(centered.dtype).tiny),
                max=1.0,
            )
            bounded = centered * multiplier
            previous = torch.where(selected.unsqueeze(-1), bounded, previous)
            rows.append(previous)
        return torch.stack(rows, dim=1).masked_fill(padding.unsqueeze(-1), 0.0)

    def _radar_ln3_decision(
        self,
        l1_probabilities: torch.Tensor,
        l2_probabilities: torch.Tensor,
        l1_knownness: torch.Tensor,
        l2_knownness: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """运行最终 L3 语义决策。"""
        l1_pred = l1_probabilities.argmax(dim=-1)
        l1_candidate = l1_probabilities >= self.l1_candidate_floors.view(1, -1)
        l1_candidate_count = l1_candidate.to(dtype=l1_probabilities.dtype).sum(dim=-1)
        l1_selected_score = l1_knownness.gather(1, l1_pred.unsqueeze(-1)).squeeze(-1)
        l1_threshold = self.l1_knownness_thresholds.gather(0, l1_pred)
        l1_known_ok = l1_selected_score >= l1_threshold
        l1_route_exact = (l1_candidate_count == 1) & l1_candidate.gather(
            1, l1_pred.unsqueeze(-1)
        ).squeeze(-1)
        l1_target = (l1_pred == 0) & l1_known_ok & l1_route_exact
        l1_clutter = (l1_pred == 1) & l1_known_ok & l1_route_exact
        l1_unknown = (~l1_clutter) & (~l1_target) & (
            (l1_candidate_count == 0) | (~l1_known_ok)
        )
        l1_pending = ~(l1_target | l1_clutter | l1_unknown)

        l2_pred = l2_probabilities.argmax(dim=-1)
        l2_selected_probability = l2_probabilities.gather(1, l2_pred.unsqueeze(-1)).squeeze(-1)
        l2_selected_score = l2_knownness.gather(1, l2_pred.unsqueeze(-1)).squeeze(-1)
        l2_threshold = self.l2_knownness_thresholds.gather(0, l2_pred)
        l2_candidate = l2_probabilities >= self.l2_candidate_floors.view(1, -1)
        l2_candidate_count = l2_candidate.to(dtype=l2_probabilities.dtype).sum(dim=-1)
        l2_known_ok = l2_selected_score >= l2_threshold
        top2 = torch.topk(l2_probabilities, k=2, dim=-1).values
        l2_margin = top2[:, 0] - top2[:, 1]
        l2_bird = l2_pred == 1
        bird_values = self.bird_gate_values
        bird_gate_ok = (
            (bird_values[3] > 0.5)
            & (~l2_bird | (
                (l2_selected_score >= bird_values[0])
                & (l2_selected_probability >= bird_values[1])
                & (l2_margin >= bird_values[2])
            ))
        )
        l2_single = l2_candidate_count == 1
        l2_margin_accept = (
            (l2_candidate_count > 1)
            & l2_candidate.gather(1, l2_pred.unsqueeze(-1)).squeeze(-1)
            & (l2_margin >= self.l2_margin_thresholds.gather(0, l2_pred))
        )
        l2_accepted = l1_target & l2_known_ok & bird_gate_ok & (l2_single | l2_margin_accept)
        l2_unknown = l1_target & (~l2_accepted) & (
            (l2_candidate_count == 0) | (~l2_known_ok)
        )
        l2_pending = l1_target & (~l2_accepted) & (~l2_unknown)

        # 受限 Bird recovery：只放宽 L1 路由，仍要求 L2 Bird 概率、knownness、margin 和候选集通过。
        target_values = self.target_bird_gate_values
        target_recovery = (
            (~l1_target)
            & (l1_pred == 0)
            & (target_values[5] > 0.5)
            & (l1_selected_score >= target_values[0])
            & (l1_selected_score <= target_values[1])
            & l2_bird
            & (l2_probabilities[:, 1] >= target_values[2])
            & (l2_knownness[:, 1] >= target_values[3])
            & (l2_margin >= target_values[4])
            & (l2_candidate_count == 1)
            & l2_candidate[:, 1]
        )
        clutter_values = self.clutter_bird_gate_values
        clutter_recovery = (
            l1_clutter
            & (clutter_values[4] > 0.5)
            & (l1_knownness[:, 0] >= clutter_values[0])
            & (l1_knownness[:, 0] <= clutter_values[1])
            & l2_bird
            & (l2_probabilities[:, 1] >= clutter_values[2])
            & (l2_knownness[:, 1] >= clutter_values[3])
        )

        action = torch.full_like(l1_pred, 5, dtype=torch.long)
        action = torch.where(l1_clutter, torch.full_like(action, 3), action)
        action = torch.where(l1_unknown | l2_unknown, torch.full_like(action, 4), action)
        action = torch.where(l2_accepted, l2_pred, action)
        action = torch.where(target_recovery | clutter_recovery, torch.ones_like(action), action)
        action = torch.where(l1_pending | l2_pending, torch.full_like(action, 5), action)
        action = torch.where(target_recovery | clutter_recovery, torch.ones_like(action), action)

        reason = torch.full_like(action, 3)
        reason = torch.where(l1_clutter, torch.full_like(reason, 1), reason)
        reason = torch.where(l1_unknown, torch.full_like(reason, 2), reason)
        reason = torch.where(l2_unknown, torch.full_like(reason, 5), reason)
        reason = torch.where(l2_accepted, torch.full_like(reason, 0), reason)
        reason = torch.where(clutter_recovery, torch.full_like(reason, 10), reason)
        reason = torch.where(target_recovery, torch.full_like(reason, 11), reason)

        target_gate = l1_probabilities[:, 0] * torch.sigmoid(
            (l1_knownness[:, 0] - self.l1_knownness_thresholds[0]) / 0.05
        )
        clutter_gate = l1_probabilities[:, 1] * torch.sigmoid(
            (l1_knownness[:, 1] - self.l1_knownness_thresholds[1]) / 0.05
        )
        l2_gate = target_gate.unsqueeze(-1) * l2_probabilities
        l2_gate = l2_gate * torch.sigmoid(
            (l2_knownness - self.l2_knownness_thresholds.view(1, -1)) / 0.05
        )
        known_mass = l2_gate.sum(dim=-1) + clutter_gate
        unknown_mass = (target_gate * (1.0 - l2_gate.sum(dim=-1))).clamp_min(0.0)
        pending_mass = (1.0 - known_mass - unknown_mass).clamp_min(0.0)
        soft = torch.cat(
            (
                l2_gate,
                clutter_gate.unsqueeze(-1),
                unknown_mass.unsqueeze(-1),
                pending_mass.unsqueeze(-1),
            ),
            dim=-1,
        )
        soft = soft / soft.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        return soft, action, reason
    def forward(self, *args: torch.Tensor) -> tuple[torch.Tensor, ...]:
        inputs = dict(zip(FULL_CASCADE_INPUT_NAMES, args, strict=True))
        l1_base = self.l1.base
        l1_inputs = _base_l1_inputs(inputs)
        l1_tr = l1_inputs["tr_features"]
        l1_rd = _rd_embedding_export(l1_base, l1_inputs["rd_frames"], l1_inputs["rd_frame_mask"])
        l1_course = l1_inputs["course_features"][..., 0:1] * l1_inputs["course_valid"].to(dtype=l1_tr.dtype)
        l1_course_valid = l1_inputs["course_valid"].to(dtype=l1_tr.dtype)
        l1_fused = l1_base.fusion(
            torch.cat(
                (
                    l1_base.tr_encoder(torch.cat((l1_tr, l1_inputs["tr_missing_mask"]), dim=-1)),
                    l1_rd,
                    l1_inputs["rd_quality"],
                    l1_course,
                    l1_course_valid,
                ),
                dim=-1,
            )
        )
        l1_encoded = self.l1.observation_adapter(
            inputs["observation_increment_values"], inputs["tr_padding_mask"]
        )
        l1_events = (l1_fused + l1_encoded).masked_fill(
            inputs["tr_padding_mask"].unsqueeze(-1), 0.0
        )
        l1_sequence, l1_trace = _liquid_sequence_export(l1_base, l1_events, l1_inputs)
        _, l1_logits = _classify_sequence(l1_base, l1_sequence, inputs["tr_padding_mask"])
        l2_inputs = _base_l2_inputs(inputs)
        reference_inputs = dict(l2_inputs)
        reference_inputs["tr_features"] = inputs["reference_tr_features"]
        reference_inputs["tr_missing_mask"] = inputs["reference_tr_missing_mask"]
        def native_events(base: Any, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
            tr = values["tr_features"]
            physical = values["rd_physical_features"] * values["rd_physical_valid"].to(dtype=tr.dtype).unsqueeze(-1)
            micro = values["micro_doppler_features"] * values["micro_doppler_valid"].to(dtype=tr.dtype).unsqueeze(-1)
            return base.fusion(
                torch.cat(
                    (
                        base.tr_encoder(torch.cat((tr, values["tr_missing_mask"]), dim=-1)),
                        _rd_embedding_export(base, values["rd_frames"], values["rd_frame_mask"]),
                        values["rd_quality"],
                        physical,
                        micro,
                    ),
                    dim=-1,
                )
            )

        l2_train_base = self.l2.trainable_base_model
        l2_motion = self.l2.motion_adapter(inputs["motion_values"], inputs["tr_padding_mask"])
        l2_events = (native_events(l2_train_base, l2_inputs) + l2_motion).masked_fill(
            inputs["tr_padding_mask"].unsqueeze(-1), 0.0
        )
        l2_sequence, l2_trace = _liquid_sequence_export(l2_train_base, l2_events, l2_inputs)
        l2_sequence_logits, _ = _classify_sequence(
            l2_train_base, l2_sequence, inputs["tr_padding_mask"]
        )
        l2_reference_base = self.l2.frozen_reference_base_model
        l2_reference_events = native_events(l2_reference_base, reference_inputs).masked_fill(
            inputs["tr_padding_mask"].unsqueeze(-1), 0.0
        )
        l2_reference_sequence, _ = _liquid_sequence_export(
            l2_reference_base, l2_reference_events, reference_inputs
        )
        l2_reference_logits, _ = _classify_sequence(
            l2_reference_base, l2_reference_sequence, inputs["tr_padding_mask"]
        )
        baseline = l2_reference_logits.detach()
        stable_residual = self._stable_residual_export(
            baseline,
            inputs["stable_summary_tokens"],
            inputs["stable_summary_valid"],
            inputs["stable_frame_values"],
            inputs["stable_frame_mask"],
            inputs["tr_padding_mask"],
        )
        l2_sequence_logits = (
            l2_sequence_logits + (baseline - l2_reference_logits) + stable_residual
        ).masked_fill(inputs["tr_padding_mask"].unsqueeze(-1), 0.0)
        l2_logits = _gather_last(l2_sequence_logits, inputs["tr_padding_mask"])
        l1_prob = _softmax(l1_logits, self.l1_temperature)
        l2_prob = _softmax(l2_logits, self.l2_temperature)
        l2_knownness_inputs = dict(l2_inputs)
        for name in (
            "stable_summary_tokens",
            "stable_summary_valid",
            "motion_values",
        ):
            l2_knownness_inputs[name] = inputs[name]
        l1_knownness, l2_knownness = self._knownness_scores(
            l1_sequence=l1_sequence,
            l1_trace=l1_trace,
            l1_inputs=l1_inputs,
            l2_sequence=l2_sequence,
            l2_trace=l2_trace,
            l2_inputs=l2_knownness_inputs,
        )
        soft, action, reason = self._radar_ln3_decision(
            l1_prob, l2_prob, l1_knownness, l2_knownness
        )
        # Keep every declared sidecar visible in the exported graph. Some
        # sidecars can be all-zero/false for a dummy export and
        # otherwise disappear under constant folding.
        anchor = l1_prob.new_zeros(())
        for value in inputs.values():
            anchor = anchor + value.to(dtype=l1_prob.dtype).reshape(-1).sum() * 0.0
        l1_prob = l1_prob + anchor
        l2_prob = l2_prob + anchor
        soft = soft + anchor
        action = action + anchor.to(dtype=action.dtype)
        reason = reason + anchor.to(dtype=reason.dtype)
        return (
            l1_prob,
            l2_prob,
            soft,
            action,
            reason,
        )


def build_full_cascade_model(
    project_root: str | Path,
    *,
    l1_checkpoint_path: str | Path,
    l2_checkpoint_path: str | Path,
    l1_knownness_head_path: str | Path,
    l2_knownness_head_path: str | Path,
    calibration_path: str | Path,
    model_config_path: str | Path = "src/config/model.json",
    device: str | torch.device = "cpu",
) -> tuple[FullCascadeModel, dict[str, Any]]:
    """Load the current radar_ln3 model bundle."""

    root = Path(project_root).resolve()
    config = load_model_config(root, model_config_path)
    l1_checkpoint = Checkpoint.load(root, l1_checkpoint_path, expected_task="l1")
    l2_checkpoint = Checkpoint.load(root, l2_checkpoint_path, expected_task="l2")
    base = Checkpoint.load(root, config.artifacts.l2_base_weight_path)
    stable = Checkpoint.load(root, config.artifacts.l2_stable_weight_path)
    l1_contract = l1_checkpoint.payload["model_contract"]
    l2_contract = l2_checkpoint.payload["model_contract"]
    l1 = build_l1_model(
        config,
        arm=str(l1_contract["arm"]),
        seed=int(l1_contract.get("seed", config.training.seed)),
        device=torch.device(str(device)),
    )
    load_model_state(l1, l1_checkpoint.state)
    l2 = build_l2_model(
        config,
        arm=str(l2_contract["arm"]),
        mode=str(l2_contract.get("mode", "residual")),
        base_state=base.state,
        stable_state=stable.state,
        seed=config.training.seed,
    ).to(device=torch.device(str(device)))
    load_model_state(l2, l2_checkpoint.state)
    calibration = load_l3_calibration(root, calibration_path)
    endpoint_hashes = {
        "l1": l1_checkpoint.file_sha256,
        "l2": l2_checkpoint.file_sha256,
    }
    if calibration.get("endpoint_sha256") != endpoint_hashes:
        raise ValueError("L3 calibration 与 endpoint checkpoint 绑定不一致")
    l1_head, l1_head_checkpoint = load_l3_knownness_head(
        root,
        l1_knownness_head_path,
        task="l1",
        expected_endpoint_sha256=endpoint_hashes["l1"],
    )
    l2_head, l2_head_checkpoint = load_l3_knownness_head(
        root,
        l2_knownness_head_path,
        task="l2",
        expected_endpoint_sha256=endpoint_hashes["l2"],
    )
    expected_heads = calibration.get("head_checkpoint_sha256") or {}
    if expected_heads.get("l1") != l1_head_checkpoint.file_sha256 or expected_heads.get("l2") != l2_head_checkpoint.file_sha256:
        raise ValueError("L3 calibration 与 knownness head checkpoint 绑定不一致")
    model = FullCascadeModel(
        l1.eval(), l2.eval(), l1_head.eval(), l2_head.eval(), calibration
    ).to(device=torch.device(str(device))).eval()
    binding = {
        "l1_checkpoint": l1_checkpoint.provenance(kind="radar_ln3_l1_endpoint", task="l1"),
        "l2_checkpoint": l2_checkpoint.provenance(kind="radar_ln3_l2_endpoint", task="l2"),
        "l1_knownness_head": l1_head_checkpoint.provenance(kind="l3_l1_knownness", task="l3_knownness_l1"),
        "l2_knownness_head": l2_head_checkpoint.provenance(kind="l3_l2_knownness", task="l3_knownness_l2"),
        "calibration_path": str(calibration_path),
        "calibration_sha256": sha256_file(root / calibration_path),
        "calibration_status": calibration.get("status"),
        "status": "formal_available",
        "release_status": "formal_available",
        "formal_available": True,
        "contract": "RADAR-LN3-ONNX-FULL-CASCADE-V1",
    }
    return model, binding



__all__ = [
    "CASCADE_ACTION_NAMES",
    "CASCADE_REASON_NAMES",
    "FULL_CASCADE_INPUT_NAMES",
    "FullCascadeModel",
    "build_full_cascade_model",
]
