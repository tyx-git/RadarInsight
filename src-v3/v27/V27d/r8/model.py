"""Liquid V2 Bird verifier with paired causal course/stable8 readout branches."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper


R8_ARM_FLAGS = {
    "zero_control": (False, False),
    "stable8_treatment": (False, True),
    "course_treatment": (True, False),
    "joint_treatment": (True, True),
}


class BirdVerifierR8(nn.Module):
    """Liquid V2 verifier with independently switchable causal readout inputs."""

    def __init__(self, *, arm: str, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        if arm not in R8_ARM_FLAGS:
            raise ValueError("V27d-r8 arm is invalid")
        self.arm = str(arm)
        self.use_course, self.use_stable8 = R8_ARM_FLAGS[self.arm]
        self.event_trajectory_dim = int(model_config["event_trajectory_dim"])
        self.task_fact_dim = int(model_config["task_fact_dim"])
        self.hidden_dim = int(model_config["hidden_dim"])
        self.summary_dim = int(model_config["summary_dim"])
        self.projection_dim = int(model_config["projection_dim"])
        self.dispersion_dim = int(model_config["dispersion_dim"])
        self.course_dim = int(model_config["course_dim"])
        self.input_dim = self.event_trajectory_dim + self.task_fact_dim
        if self.input_dim != int(model_config["input_dim"]):
            raise ValueError("V27d-r8 input width differs")
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
        self.base_readout = nn.Linear(self.summary_dim + self.hidden_dim + self.task_fact_dim, self.projection_dim)
        self.dispersion_projection = nn.Linear(self.dispersion_dim, self.projection_dim, bias=False)
        self.course_projection = nn.Linear(self.course_dim, self.projection_dim, bias=False)
        self.readout_activation = nn.SiLU()
        self.verifier_head = nn.Linear(self.projection_dim, 1)

    @property
    def recurrent_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.core.parameters())

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27d-r8-verifier-contract-v1",
            "experiment_id": "V27d-r8-joint-course-stable8",
            "arm": self.arm,
            "core": "liquid_v2",
            "event_trajectory_dim": self.event_trajectory_dim,
            "task_fact_dim": self.task_fact_dim,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "summary_dim": self.summary_dim,
            "projection_dim": self.projection_dim,
            "dispersion_dim": self.dispersion_dim,
            "course_dim": self.course_dim,
            "use_course": self.use_course,
            "use_stable8": self.use_stable8,
            "dispersion_statistics": ["current_frame_std", "causal_event_std", "causal_adjacent_delta"],
            "course_feature": "causal_prefix_course_circular_incoherence",
            "course_valid_is_knownness_input": False,
            "dispersion_valid_fraction_is_knownness_input": False,
            "readout_contract": "r6_base_linear_plus_zero_initialized_course_and_stable8_residual_then_silu",
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
            "padding_policy": "suffix_masked_no_recursive_update",
            "liquid_equation": "RADAR-LN3-V27-liquid-v2-raw-memory-pre-norm-v1",
            "r6_initialization": True,
        }

    @staticmethod
    def _time_contract(event_facts: torch.Tensor, padding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if event_facts.shape[-1] != 19:
            raise ValueError("V27d-r8 event facts width differs")
        dt_seconds = (event_facts[..., 7].clamp_min(0.0) * 120.0).masked_fill(padding, 0.0)
        dt_valid = event_facts[..., 8].ge(0.5) & ~padding
        first = event_facts[..., 9].ge(0.5)
        zero = event_facts[..., 10].ge(0.5)
        positive = event_facts[..., 11].ge(0.5)
        reason = torch.zeros_like(event_facts[..., 8], dtype=torch.long)
        reason = reason.masked_fill(first, 1).masked_fill(zero, 2).masked_fill(positive, 3).masked_fill(padding, 0)
        if bool((reason.eq(3) != dt_valid).masked_select(~padding).any()):
            raise ValueError("V27d-r8 dt reason and validity differ")
        return dt_seconds, dt_valid, reason

    def forward(
        self,
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        event_facts_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
        dispersion_features: torch.Tensor,
        course_features: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        values = (summary_features, event_trajectory, task_facts_sequence, dispersion_features, course_features)
        if any(value.ndim != 3 for value in values) or padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("V27d-r8 input dimensions differ")
        if any(value.shape[:2] != padding_mask.shape for value in values):
            raise ValueError("V27d-r8 sequence shapes differ")
        if event_trajectory.shape[-1] != self.event_trajectory_dim or task_facts_sequence.shape[-1] != self.task_fact_dim or dispersion_features.shape[-1] != self.dispersion_dim or course_features.shape[-1] != self.course_dim or event_facts_sequence.shape != (*padding_mask.shape, 19):
            raise ValueError("V27d-r8 input widths differ")
        for name, value in (("summary", summary_features), ("trajectory", event_trajectory), ("task", task_facts_sequence), ("event_facts", event_facts_sequence), ("dispersion", dispersion_features), ("course", course_features)):
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"V27d-r8 {name} values are invalid")
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError(f"V27d-r8 padded {name} values are non-zero")
        dt_seconds, dt_valid, dt_reason = self._time_contract(event_facts_sequence, padding_mask)
        event_input = torch.cat((event_trajectory, task_facts_sequence), dim=-1)
        output = self.core(event_input, dt_seconds, padding_mask, dt_valid=dt_valid, dt_reason=dt_reason, return_trace=True)
        recurrent = output.sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        base_representation = torch.cat((summary_features, recurrent, task_facts_sequence), dim=-1)
        base_projection = self.base_readout(base_representation)
        dispersion = dispersion_features if self.use_stable8 else torch.zeros_like(dispersion_features)
        course = course_features if self.use_course else torch.zeros_like(course_features)
        residual = self.dispersion_projection(dispersion) + self.course_projection(course)
        projection = self.readout_activation(base_projection + residual).masked_fill(padding_mask.unsqueeze(-1), 0.0)
        logits = self.verifier_head(projection).squeeze(-1).masked_fill(padding_mask, 0.0)
        rows = torch.arange(padding_mask.shape[0], device=padding_mask.device)
        final = (~padding_mask).long().sum(dim=1) - 1
        if not all(torch.isfinite(value).all() for value in (recurrent, projection, logits)):
            raise FloatingPointError("V27d-r8 output is non-finite")
        return {
            "recurrent_sequence": recurrent,
            "projection_sequence": projection,
            "verifier_logit_sequence": logits,
            "verifier_score_sequence": torch.sigmoid(logits),
            "recurrent_final": recurrent[rows, final],
            "projection_final": projection[rows, final],
            "verifier_logit_final": logits[rows, final],
            "verifier_score_final": torch.sigmoid(logits[rows, final]),
            "liquid_trace": output.trace,
        }


def build_verifier(config: Mapping[str, Any], *, arm: str, seed: int) -> BirdVerifierR8:
    if int(seed) < 0:
        raise ValueError("V27d-r8 seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return BirdVerifierR8(arm=arm, model_config=config["model"])


def initialize_from_r6(model: BirdVerifierR8, r6_state: Mapping[str, torch.Tensor]) -> None:
    allowed = {"core." + key[len("core."): ] for key in r6_state if key.startswith("core.")} | {"verifier_head.bias", "verifier_head.weight", "readout.0.bias", "readout.0.weight"}
    if set(r6_state) != allowed:
        raise ValueError("V27d-r6 checkpoint state keys differ")
    state = model.state_dict()
    for key, value in r6_state.items():
        if key == "readout.0.weight":
            state["base_readout.weight"].copy_(value)
        elif key == "readout.0.bias":
            state["base_readout.bias"].copy_(value)
        else:
            if key not in state or tuple(state[key].shape) != tuple(value.shape):
                raise ValueError(f"V27d-r6 initialization shape differs: {key}")
            state[key].copy_(value)
    state["dispersion_projection.weight"].zero_()
    state["course_projection.weight"].zero_()
    model.load_state_dict(state, strict=True)


__all__ = ["BirdVerifierR8", "R8_ARM_FLAGS", "build_verifier", "initialize_from_r6"]
