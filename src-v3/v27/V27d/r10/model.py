"""Low-dimensional causal Doppler-shape residual head for V27d-r10."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .protocol import (
    R10_CLASS_NAMES,
    R10_DESCRIPTOR_DIM,
    R10_DESCRIPTOR_NAMES,
    R10_EXPERIMENT_ID,
    R10_SCHEMA_VERSION,
)


class CompactCausalDopplerResidual(nn.Module):
    """Predict a bounded Drone-vs-Bird correction from causal descriptors.

    The head never receives classifier logits.  Its only recurrent state is
    the causal descriptor summary used to form running mean, running standard
    deviation and adjacent-event shape-change features.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.descriptor_dim = int(config["descriptor_dim"])
        self.projection_dim = int(config["projection_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.pairwise_delta_bound = float(config["pairwise_delta_bound"])
        if self.descriptor_dim != R10_DESCRIPTOR_DIM:
            raise ValueError("V27d-r10 descriptor width differs")
        if self.projection_dim <= 0 or self.hidden_dim <= 0 or self.pairwise_delta_bound <= 0.0:
            raise ValueError("V27d-r10 model dimensions/bound must be positive")
        self.descriptor_projection = nn.Linear(self.descriptor_dim, self.projection_dim)
        self.correction_mlp = nn.Sequential(
            nn.Linear(self.projection_dim * 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.correction_mlp[-1].weight)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": f"{R10_SCHEMA_VERSION}-model",
            "experiment_id": R10_EXPERIMENT_ID,
            "class_names": list(R10_CLASS_NAMES),
            "descriptor_names": list(R10_DESCRIPTOR_NAMES),
            "descriptor_dim": self.descriptor_dim,
            "projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim,
            "pairwise_delta_bound": self.pairwise_delta_bound,
            "activation": "SiLU",
            "branch_inputs": [
                "current_event_shape_descriptors",
                "causal_running_mean_descriptors",
                "causal_running_std_descriptors",
                "delta_from_previous_valid_event_descriptors",
            ],
            "correction_layout": [
                "DroneTarget:+delta/2",
                "BirdTarget:-delta/2",
                "BalloonTarget:0",
            ],
            "zero_initialized_output": True,
            "padding_policy": "suffix-masked-no-state-update",
            "invalid_profile_policy": "zero-residual-and-preserve-descriptor-state",
            "classification_logits_input_to_branch": False,
            "softmax_input_to_branch": False,
            "label_identity_input_to_branch": False,
            "gru_used": False,
        }

    @staticmethod
    def _validate_padding(padding_mask: torch.Tensor) -> None:
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("V27d-r10 padding mask must be bool [B,T]")
        for row in range(padding_mask.shape[0]):
            padded = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded.numel() and bool((~padding_mask[row, int(padded[0]) :]).any()):
                raise ValueError("V27d-r10 padding must be a suffix")

    def _validate_inputs(
        self,
        baseline_logits: torch.Tensor,
        descriptor_values: torch.Tensor,
        descriptor_valid: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if baseline_logits.ndim != 3 or baseline_logits.shape[-1] != 3:
            raise ValueError("V27d-r10 baseline logits must have shape [B,T,3]")
        if descriptor_values.ndim != 3 or descriptor_values.shape[-1] != self.descriptor_dim:
            raise ValueError("V27d-r10 descriptors must have shape [B,T,8]")
        if descriptor_values.shape[:2] != baseline_logits.shape[:2]:
            raise ValueError("V27d-r10 baseline/descriptor sequence shapes differ")
        if descriptor_valid.shape != descriptor_values.shape[:2] or descriptor_valid.dtype != torch.bool:
            raise ValueError("V27d-r10 descriptor validity shape/dtype differs")
        self._validate_padding(padding_mask)
        if padding_mask.shape != baseline_logits.shape[:2]:
            raise ValueError("V27d-r10 padding shape differs")
        if any(not value.is_floating_point() for value in (baseline_logits, descriptor_values)):
            raise TypeError("V27d-r10 model values must be floating point")
        if baseline_logits.dtype != descriptor_values.dtype or baseline_logits.device != descriptor_values.device:
            raise ValueError("V27d-r10 model values must share dtype/device")
        if descriptor_valid.device != baseline_logits.device or padding_mask.device != baseline_logits.device:
            raise ValueError("V27d-r10 masks must share device")
        if not torch.isfinite(baseline_logits).all() or not torch.isfinite(descriptor_values).all():
            raise ValueError("V27d-r10 model values must be finite")
        if bool(baseline_logits.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r10 padded baseline logits must be zero")
        if bool(descriptor_values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r10 padded descriptors must be zero")
        if bool(descriptor_valid.masked_select(padding_mask).any()):
            raise ValueError("V27d-r10 padded descriptors cannot be valid")
        if bool(descriptor_values.masked_select((~descriptor_valid).unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r10 invalid descriptors must be zero")
        signed = descriptor_values[..., (0, 3)]
        bounded = descriptor_values[..., (1, 2, 4, 5, 6, 7)]
        if bool(signed.abs().gt(1.0 + 1.0e-6).any()) or bool(bounded.lt(-1.0e-6).any()) or bool(bounded.gt(1.0 + 1.0e-6).any()):
            raise ValueError("V27d-r10 descriptors are outside their registered bounds")
        return int(baseline_logits.shape[0]), int(baseline_logits.shape[1])

    def _project_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.descriptor_dim * 4:
            raise ValueError("V27d-r10 causal descriptor feature width differs")
        grouped = features.reshape(*features.shape[:-1], 4, self.descriptor_dim)
        projected = self.descriptor_projection(grouped)
        return projected.reshape(*features.shape[:-1], self.projection_dim * 4)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("V27d-r10 initial state batch size must be positive")
        return {
            "running_count": torch.zeros((int(batch_size), 1), device=device, dtype=dtype),
            "running_sum": torch.zeros((int(batch_size), self.descriptor_dim), device=device, dtype=dtype),
            "running_square_sum": torch.zeros((int(batch_size), self.descriptor_dim), device=device, dtype=dtype),
            "previous_descriptor": torch.zeros((int(batch_size), self.descriptor_dim), device=device, dtype=dtype),
            "previous_valid": torch.zeros((int(batch_size), 1), device=device, dtype=torch.bool),
            "previous_delta": torch.zeros((int(batch_size),), device=device, dtype=dtype),
        }

    def _validate_state(self, state: Mapping[str, torch.Tensor], *, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        expected = {
            "running_count": (batch_size, 1),
            "running_sum": (batch_size, self.descriptor_dim),
            "running_square_sum": (batch_size, self.descriptor_dim),
            "previous_descriptor": (batch_size, self.descriptor_dim),
            "previous_valid": (batch_size, 1),
            "previous_delta": (batch_size,),
        }
        for name, shape in expected.items():
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"V27d-r10 continuation state differs: {name}")
            if value.device != device:
                raise ValueError("V27d-r10 continuation state device differs")
            if name == "previous_valid":
                if value.dtype != torch.bool:
                    raise ValueError("V27d-r10 continuation previous_valid must be bool")
            elif value.dtype != dtype or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"V27d-r10 continuation state value differs: {name}")
        if bool(state["previous_delta"].abs().gt(self.pairwise_delta_bound + 1.0e-6).any()):
            raise ValueError("V27d-r10 continuation previous_delta exceeds its bound")

    @staticmethod
    def _correction(delta: torch.Tensor) -> torch.Tensor:
        return torch.stack((delta * 0.5, -delta * 0.5, torch.zeros_like(delta)), dim=-1)

    def step_event(
        self,
        *,
        baseline_logits: torch.Tensor,
        descriptor_values: torch.Tensor,
        descriptor_valid: torch.Tensor,
        state: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if baseline_logits.ndim != 2 or baseline_logits.shape[-1] != 3:
            raise ValueError("V27d-r10 step baseline logits shape differs")
        if descriptor_values.ndim != 2 or descriptor_values.shape[-1] != self.descriptor_dim:
            raise ValueError("V27d-r10 step descriptor shape differs")
        if descriptor_valid.shape != (baseline_logits.shape[0],) or descriptor_valid.dtype != torch.bool:
            raise ValueError("V27d-r10 step descriptor validity differs")
        self._validate_state(state, batch_size=baseline_logits.shape[0], device=baseline_logits.device, dtype=baseline_logits.dtype)
        if descriptor_values.device != baseline_logits.device or descriptor_values.dtype != baseline_logits.dtype:
            raise ValueError("V27d-r10 step values must share dtype/device")
        if bool(descriptor_values.masked_select((~descriptor_valid).unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r10 step invalid descriptors must be zero")
        current = descriptor_values
        indicator = descriptor_valid.unsqueeze(-1).to(dtype=baseline_logits.dtype)
        running_count = state["running_count"]
        running_sum = state["running_sum"]
        running_square_sum = state["running_square_sum"]
        previous = state["previous_descriptor"]
        previous_valid = state["previous_valid"]
        next_count = running_count + indicator
        next_sum = running_sum + indicator * current
        next_square_sum = running_square_sum + indicator * current.square()
        mean = next_sum / next_count.clamp_min(1.0)
        std = (next_square_sum / next_count.clamp_min(1.0) - mean.square()).clamp_min(0.0).sqrt()
        delta_shape = torch.where(previous_valid, (current - previous).abs(), torch.zeros_like(current))
        features = torch.cat((current, mean, std, delta_shape), dim=-1)
        features = features.masked_fill(~descriptor_valid.unsqueeze(-1), 0.0)
        raw = self.correction_mlp(self._project_features(features)).squeeze(-1)
        raw = raw.masked_fill(~descriptor_valid, 0.0)
        proposed = self.pairwise_delta_bound * torch.tanh(raw)
        applied = torch.where(descriptor_valid, proposed, torch.zeros_like(proposed))
        next_previous_delta = torch.where(descriptor_valid, proposed, state["previous_delta"])
        return {
            "sequence_logits": baseline_logits + self._correction(applied),
            "correction_logits": self._correction(applied),
            "pairwise_delta": applied,
            "pairwise_delta_increment": torch.where(descriptor_valid, proposed - state["previous_delta"], torch.zeros_like(proposed)),
            "raw_pairwise_delta": raw,
            "causal_descriptor_features": features,
            "next_running_count": next_count,
            "next_running_sum": next_sum,
            "next_running_square_sum": next_square_sum,
            "next_previous_descriptor": torch.where(indicator.bool(), current, previous),
            "next_previous_valid": previous_valid | descriptor_valid.unsqueeze(-1),
            "next_previous_delta": next_previous_delta,
        }

    def forward(
        self,
        *,
        baseline_logits: torch.Tensor,
        descriptor_values: torch.Tensor,
        descriptor_valid: torch.Tensor,
        padding_mask: torch.Tensor,
        initial_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch, length = self._validate_inputs(baseline_logits, descriptor_values, descriptor_valid, padding_mask)
        state = self.initial_state(batch, device=baseline_logits.device, dtype=baseline_logits.dtype) if initial_state is None else dict(initial_state)
        self._validate_state(state, batch_size=batch, device=baseline_logits.device, dtype=baseline_logits.dtype)
        running_count = state["running_count"]
        running_sum = state["running_sum"]
        running_square_sum = state["running_square_sum"]
        previous = state["previous_descriptor"]
        previous_valid = state["previous_valid"]
        previous_delta = state["previous_delta"]
        logits_rows: list[torch.Tensor] = []
        correction_rows: list[torch.Tensor] = []
        delta_rows: list[torch.Tensor] = []
        increment_rows: list[torch.Tensor] = []
        raw_rows: list[torch.Tensor] = []
        feature_rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        for event in range(length):
            active = ~padding_mask[:, event]
            valid = descriptor_valid[:, event] & active
            step = self.step_event(
                baseline_logits=baseline_logits[:, event],
                descriptor_values=descriptor_values[:, event],
                descriptor_valid=valid,
                state={
                    "running_count": running_count,
                    "running_sum": running_sum,
                    "running_square_sum": running_square_sum,
                    "previous_descriptor": previous,
                    "previous_valid": previous_valid,
                    "previous_delta": previous_delta,
                },
            )
            running_count = torch.where(active.unsqueeze(-1), step["next_running_count"], running_count)
            running_sum = torch.where(active.unsqueeze(-1), step["next_running_sum"], running_sum)
            running_square_sum = torch.where(active.unsqueeze(-1), step["next_running_square_sum"], running_square_sum)
            previous = torch.where(active.unsqueeze(-1), step["next_previous_descriptor"], previous)
            previous_valid = torch.where(active.unsqueeze(-1), step["next_previous_valid"], previous_valid)
            previous_delta = torch.where(active, step["next_previous_delta"], previous_delta)
            pad = padding_mask[:, event]
            logits_rows.append(step["sequence_logits"].masked_fill(pad.unsqueeze(-1), 0.0))
            correction_rows.append(step["correction_logits"].masked_fill(pad.unsqueeze(-1), 0.0))
            delta_rows.append(step["pairwise_delta"].masked_fill(pad, 0.0))
            increment_rows.append(step["pairwise_delta_increment"].masked_fill(pad, 0.0))
            raw_rows.append(step["raw_pairwise_delta"].masked_fill(pad, 0.0))
            feature_rows.append(step["causal_descriptor_features"].masked_fill(pad.unsqueeze(-1), 0.0))
            valid_rows.append(valid)
        sequence_logits = torch.stack(logits_rows, dim=1)
        correction = torch.stack(correction_rows, dim=1)
        deltas = torch.stack(delta_rows, dim=1)
        increments = torch.stack(increment_rows, dim=1)
        raw = torch.stack(raw_rows, dim=1)
        features = torch.stack(feature_rows, dim=1)
        valid_sequence = torch.stack(valid_rows, dim=1)
        if not torch.isfinite(sequence_logits).all() or not torch.isfinite(correction).all():
            raise FloatingPointError("V27d-r10 output is non-finite")
        if bool(correction[..., 2].ne(0.0).any()) or bool(correction.sum(dim=-1).abs().gt(1.0e-6).any()):
            raise AssertionError("V27d-r10 correction layout contract failed")
        if bool(correction.abs().gt(self.pairwise_delta_bound / 2.0 + 1.0e-5).any()):
            raise AssertionError("V27d-r10 correction exceeds pairwise bound")
        return {
            "sequence_logits": sequence_logits,
            "baseline_logits": baseline_logits,
            "correction_logits": correction,
            "pairwise_delta_sequence": deltas,
            "pairwise_delta_increment": increments,
            "raw_pairwise_delta_sequence": raw,
            "causal_descriptor_features": features,
            "descriptor_valid_sequence": valid_sequence,
            "final_state": {
                "running_count": running_count,
                "running_sum": running_sum,
                "running_square_sum": running_square_sum,
                "previous_descriptor": previous,
                "previous_valid": previous_valid,
                "previous_delta": previous_delta,
            },
        }


def build_model(config: Mapping[str, Any], *, seed: int) -> CompactCausalDopplerResidual:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27d-r10 seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return CompactCausalDopplerResidual(config)


__all__ = ["CompactCausalDopplerResidual", "build_model"]
