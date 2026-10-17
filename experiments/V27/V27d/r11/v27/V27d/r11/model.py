"""Liquid V2 Bird-route verifier with a causal full-stable8 shape branch."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper

from .features import causal_stable8_shape_features
from .protocol import R11_EXPERIMENT_ID, R11_SHAPE_FEATURE_DIM, R11_STABLE8_DIM, R11_SCHEMA_VERSION


class BirdVerifierR11(nn.Module):
    """Frozen r6 Liquid verifier plus a trainable zero-initialized shape readout."""

    def __init__(self, *, arm: str, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        if arm not in {"zero_control", "shape_treatment"}:
            raise ValueError("V27d-r11 verifier arm is invalid")
        self.arm = str(arm)
        self.event_trajectory_dim = int(model_config["event_trajectory_dim"])
        self.task_fact_dim = int(model_config["task_fact_dim"])
        self.input_dim = int(model_config["input_dim"])
        self.hidden_dim = int(model_config["hidden_dim"])
        self.summary_dim = int(model_config["summary_dim"])
        self.projection_dim = int(model_config["projection_dim"])
        self.stable8_dim = int(model_config["stable8_dim"])
        self.shape_feature_dim = int(model_config["shape_feature_dim"])
        if self.input_dim != self.event_trajectory_dim + self.task_fact_dim:
            raise ValueError("V27d-r11 recurrent input width differs")
        if self.stable8_dim != R11_STABLE8_DIM or self.shape_feature_dim != R11_SHAPE_FEATURE_DIM:
            raise ValueError("V27d-r11 shape dimensions differ")
        self.core = LiquidV2SequenceWrapper(
            LiquidV2Cell(
                self.input_dim,
                self.hidden_dim,
                tau_min_seconds=float(model_config["liquid_tau_min_seconds"]),
                tau_max_seconds=float(model_config["liquid_tau_max_seconds"]),
                zero_dt_event_floor_seconds=float(model_config["liquid_zero_dt_event_floor_seconds"]),
                dt_max_seconds=float(model_config["liquid_dt_max_seconds"]),
                invalid_dt_update_mode=str(model_config["liquid_invalid_dt_update_mode"]),
                invalid_dt_observation_gain=float(model_config["liquid_invalid_dt_observation_gain"]),
            )
        )
        self.base_readout = nn.Linear(
            self.summary_dim + self.hidden_dim + self.task_fact_dim,
            self.projection_dim,
        )
        self.shape_projection = nn.Linear(
            self.shape_feature_dim,
            self.projection_dim,
            bias=False,
        )
        nn.init.zeros_(self.shape_projection.weight)
        self.readout_activation = nn.SiLU()
        self.verifier_head = nn.Linear(self.projection_dim, 1)

    @property
    def recurrent_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.core.parameters())

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": f"{R11_SCHEMA_VERSION}-model",
            "experiment_id": R11_EXPERIMENT_ID,
            "arm": self.arm,
            "core": "liquid_v2",
            "event_trajectory_dim": self.event_trajectory_dim,
            "task_fact_dim": self.task_fact_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "summary_dim": self.summary_dim,
            "projection_dim": self.projection_dim,
            "stable8_dim": self.stable8_dim,
            "shape_feature_dim": self.shape_feature_dim,
            "shape_features": [
                "current_event_shape",
                "causal_running_mean_shape",
                "causal_running_std_shape",
                "previous_valid_event_absolute_delta",
            ],
            "shape_normalization": "per-reliable-frame-l2-then-event-mean-l2",
            "shape_projection": "zero_initialized_bias_free_linear",
            "trainable_component": "shape_projection_only",
            "classification_logits_input_to_branch": False,
            "softmax_input_to_branch": False,
            "label_identity_input_to_branch": False,
            "shape_valid_is_numeric_input": False,
            "padding_policy": "suffix_masked_no_liquid_or_shape_update",
            "liquid_equation": "RADAR-LN3-V27-liquid-v2-raw-memory-pre-norm-v1",
            "r6_initialization": True,
        }

    @staticmethod
    def _validate_common(
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int]:
        values = (summary_features, event_trajectory, task_facts_sequence)
        if any(value.ndim != 3 for value in values):
            raise ValueError("V27d-r11 verifier tensors must be three-dimensional")
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("V27d-r11 padding mask is invalid")
        if any(value.shape[:2] != padding_mask.shape for value in values):
            raise ValueError("V27d-r11 verifier sequence shapes differ")
        if not all(value.is_floating_point() and torch.isfinite(value).all() for value in values):
            raise ValueError("V27d-r11 verifier inputs must be finite floats")
        for value in values:
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError("V27d-r11 padded verifier inputs must be zero")
        lengths = (~padding_mask).long().sum(dim=1)
        if bool(lengths.le(0).any()):
            raise ValueError("V27d-r11 verifier received an empty sequence")
        return int(padding_mask.shape[0]), int(padding_mask.shape[1])

    @staticmethod
    def _time_contract(
        event_facts: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if event_facts.shape[-1] != 19:
            raise ValueError("V27d-r11 event facts must have width 19")
        dt_seconds = (event_facts[..., 7].clamp_min(0.0) * 120.0).masked_fill(padding_mask, 0.0)
        dt_valid = event_facts[..., 8].ge(0.5) & ~padding_mask
        first = event_facts[..., 9].ge(0.5)
        zero = event_facts[..., 10].ge(0.5)
        positive = event_facts[..., 11].ge(0.5)
        reason = torch.zeros_like(event_facts[..., 8], dtype=torch.long)
        reason = reason.masked_fill(first, 1).masked_fill(zero, 2).masked_fill(positive, 3).masked_fill(padding_mask, 0)
        if bool((reason.eq(3) != dt_valid).masked_select(~padding_mask).any()):
            raise ValueError("V27d-r11 event time reason and validity differ")
        return dt_seconds, dt_valid, reason

    def _validate_frame_inputs(
        self,
        event_frame_values: torch.Tensor,
        event_frame_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> None:
        if event_frame_values.ndim != 4 or event_frame_values.shape[-1] != self.stable8_dim:
            raise ValueError("V27d-r11 event frame values shape differs")
        if event_frame_mask.shape != event_frame_values.shape[:3] or event_frame_mask.dtype != torch.bool:
            raise ValueError("V27d-r11 event frame mask shape/dtype differs")
        if event_frame_values.shape[:2] != padding_mask.shape:
            raise ValueError("V27d-r11 event frame sequence shape differs")
        if not event_frame_values.is_floating_point():
            raise TypeError("V27d-r11 event frame values must be floating point")
        if not torch.isfinite(event_frame_values).all():
            raise ValueError("V27d-r11 event frame values must be finite")
        if event_frame_values.device != padding_mask.device or event_frame_mask.device != padding_mask.device:
            raise ValueError("V27d-r11 event frame tensors must share device")
        if bool(event_frame_mask.masked_select(padding_mask.unsqueeze(-1).expand_as(event_frame_mask)).any()):
            raise ValueError("V27d-r11 padded event frame masks must be inactive")
        expanded_padding = padding_mask.unsqueeze(-1).unsqueeze(-1).expand_as(event_frame_values)
        if bool(event_frame_values.masked_select(expanded_padding).ne(0).any()):
            raise ValueError("V27d-r11 padded event frame values must be zero")
        inactive = (~event_frame_mask).unsqueeze(-1).expand_as(event_frame_values)
        if bool(event_frame_values.masked_select(inactive).ne(0).any()):
            raise ValueError("V27d-r11 inactive event frame values must be zero")
        for row in range(event_frame_mask.shape[0]):
            for event in range(event_frame_mask.shape[1]):
                mask = event_frame_mask[row, event]
                first_inactive = torch.nonzero(~mask, as_tuple=False).flatten()
                if first_inactive.numel() and bool(mask[int(first_inactive[0]) :].any()):
                    raise ValueError("V27d-r11 event frame masks must be suffix-padded")

    def forward(
        self,
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        event_facts_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
        event_frame_values: torch.Tensor,
        event_frame_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch, length = self._validate_common(
            summary_features=summary_features,
            event_trajectory=event_trajectory,
            task_facts_sequence=task_facts_sequence,
            padding_mask=padding_mask,
        )
        if event_trajectory.shape[-1] != self.event_trajectory_dim or task_facts_sequence.shape[-1] != self.task_fact_dim:
            raise ValueError("V27d-r11 verifier input widths differ")
        if event_facts_sequence.shape != (*padding_mask.shape, 19) or not event_facts_sequence.is_floating_point() or not torch.isfinite(event_facts_sequence).all():
            raise ValueError("V27d-r11 event facts shape/value differs")
        if bool(event_facts_sequence.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r11 padded event facts must be zero")
        self._validate_frame_inputs(event_frame_values, event_frame_mask, padding_mask)
        if event_frame_values.dtype != summary_features.dtype or event_frame_values.device != summary_features.device:
            raise ValueError("V27d-r11 event frame values must match model dtype/device")
        if event_frame_mask.device != summary_features.device:
            raise ValueError("V27d-r11 event frame mask device differs")
        dt_seconds, dt_valid, dt_reason = self._time_contract(event_facts_sequence, padding_mask)
        event_input = torch.cat((event_trajectory, task_facts_sequence), dim=-1)
        recurrent = self.core(
            event_input,
            dt_seconds,
            padding_mask,
            dt_valid=dt_valid,
            dt_reason=dt_reason,
            return_trace=True,
        )
        recurrent_sequence = recurrent.sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        shape_features, shape_valid, shape_counts = causal_stable8_shape_features(
            event_frame_values,
            event_frame_mask,
            padding_mask,
        )
        representation = torch.cat((summary_features, recurrent_sequence, task_facts_sequence), dim=-1)
        base_projection = self.base_readout(representation)
        shape_projection = self.shape_projection(shape_features)
        projection_sequence = self.readout_activation(base_projection + shape_projection)
        projection_sequence = projection_sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        logits = self.verifier_head(projection_sequence).squeeze(-1).masked_fill(padding_mask, 0.0)
        rows = torch.arange(batch, device=padding_mask.device)
        final = (~padding_mask).long().sum(dim=1) - 1
        values = (recurrent_sequence, shape_features, projection_sequence, logits)
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("V27d-r11 verifier produced non-finite outputs")
        return {
            "recurrent_sequence": recurrent_sequence,
            "shape_features": shape_features,
            "shape_valid_sequence": shape_valid,
            "shape_frame_count": shape_counts,
            "shape_projection_sequence": shape_projection.masked_fill(padding_mask.unsqueeze(-1), 0.0),
            "projection_sequence": projection_sequence,
            "verifier_logit_sequence": logits,
            "verifier_score_sequence": torch.sigmoid(logits),
            "recurrent_final": recurrent_sequence[rows, final],
            "shape_features_final": shape_features[rows, final],
            "projection_final": projection_sequence[rows, final],
            "verifier_logit_final": logits[rows, final],
            "verifier_score_final": torch.sigmoid(logits[rows, final]),
            "liquid_trace": recurrent.trace,
        }


def build_model(config: Mapping[str, Any], *, arm: str, seed: int) -> BirdVerifierR11:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27d-r11 initialization seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return BirdVerifierR11(arm=arm, model_config=config["model"])


def initialize_from_r6(model: BirdVerifierR11, r6_state: Mapping[str, torch.Tensor]) -> None:
    expected = {
        key
        for key in r6_state
        if key.startswith("core.")
    } | {
        "readout.0.weight",
        "readout.0.bias",
        "verifier_head.weight",
        "verifier_head.bias",
    }
    if set(r6_state) != expected:
        raise ValueError("V27d-r11 r6 checkpoint state keys differ")
    state = model.state_dict()
    for key, value in r6_state.items():
        target = (
            key.replace("readout.0.", "base_readout.", 1)
            if key.startswith("readout.0.")
            else key
        )
        if target not in state or tuple(state[target].shape) != tuple(value.shape):
            raise ValueError(f"V27d-r11 r6 initialization shape differs: {key}")
        state[target].copy_(value)
    state["shape_projection.weight"].zero_()
    model.load_state_dict(state, strict=True)


__all__ = ["BirdVerifierR11", "build_model", "initialize_from_r6"]