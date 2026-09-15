"""Causal raw envelope-profile residual head for V27d-r13."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .protocol import (
    R13_CLASS_NAMES,
    R13_EXPERIMENT_ID,
    R13_PROFILE_BINS,
    R13_SCHEMA_VERSION,
)


class CausalEnvelopeProfileResidual(nn.Module):
    """Predict a bounded Drone-vs-Bird correction from raw envelope shape.

    The profile statistics are causal and stateful, but the intervention is a
    late residual over frozen V27a Main logits.  This isolates the new raw
    profile variable from the Liquid V2 and L1/L2 endpoint parameters.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.profile_bins = int(config["profile_bins"])
        self.projection_dim = int(config["profile_projection_dim"])
        self.hidden_dim = int(config["hidden_dim"])
        self.pairwise_delta_bound = float(config["pairwise_delta_bound"])
        if self.profile_bins != R13_PROFILE_BINS:
            raise ValueError("V27d-r13 model profile width must be 64")
        if self.projection_dim <= 0 or self.hidden_dim <= 0 or self.pairwise_delta_bound <= 0.0:
            raise ValueError("V27d-r13 model dimensions/bound must be positive")

        self.profile_projection = nn.Linear(self.profile_bins, self.projection_dim)
        self.shape_mlp = nn.Sequential(
            nn.Linear(self.projection_dim * 4, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1, bias=False),
        )
        nn.init.zeros_(self.shape_mlp[-1].weight)

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": f"{R13_SCHEMA_VERSION}-model",
            "experiment_id": R13_EXPERIMENT_ID,
            "class_names": list(R13_CLASS_NAMES),
            "profile_source": "V18 envelope_modulation_profiles",
            "profile_bins": self.profile_bins,
            "profile_projection_dim": self.projection_dim,
            "hidden_dim": self.hidden_dim,
            "pairwise_delta_bound": self.pairwise_delta_bound,
            "activation": "SiLU",
            "branch_inputs": [
                "current_valid_event_shape",
                "causal_running_mean_shape",
                "causal_running_std_shape",
                "delta_from_previous_valid_event_shape",
            ],
            "profile_frequency_support_hz": [300.0, 2000.0],
            "profile_normalization": "per-frame-profile-mean-then-event-l1-shape",
            "correction_layout": [
                "DroneTarget:+delta/2",
                "BirdTarget:-delta/2",
                "BalloonTarget:0",
            ],
            "zero_initialized_output": True,
            "padding_policy": "suffix-masked-no-state-update",
            "invalid_profile_policy": "zero-residual-and-preserve-envelope-state",
            "classification_logits_input_to_branch": False,
            "softmax_input_to_branch": False,
            "label_identity_input_to_branch": False,
            "residual_doppler_profile_used": False,
            "gru_used": False,
        }

    @staticmethod
    def _validate_padding(padding_mask: torch.Tensor) -> None:
        if padding_mask.ndim != 2 or padding_mask.dtype != torch.bool:
            raise ValueError("V27d-r13 padding mask must be bool [B,T]")
        for row in range(padding_mask.shape[0]):
            padded = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded.numel() and bool((~padding_mask[row, int(padded[0]) :]).any()):
                raise ValueError("V27d-r13 padding must be a suffix")

    def _validate_inputs(
        self,
        baseline_logits: torch.Tensor,
        profile_values: torch.Tensor,
        profile_valid: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[int, int]:
        if baseline_logits.ndim != 3 or baseline_logits.shape[-1] != 3:
            raise ValueError("V27d-r13 baseline logits must have shape [B,T,3]")
        if profile_values.ndim != 3 or profile_values.shape[-1] != self.profile_bins:
            raise ValueError("V27d-r13 profile values must have shape [B,T,64]")
        if profile_valid.shape != profile_values.shape[:2] or profile_valid.dtype != torch.bool:
            raise ValueError("V27d-r13 profile validity shape/dtype differs")
        if profile_values.shape[:2] != baseline_logits.shape[:2]:
            raise ValueError("V27d-r13 baseline/profile sequence shapes differ")
        self._validate_padding(padding_mask)
        if padding_mask.shape != baseline_logits.shape[:2]:
            raise ValueError("V27d-r13 padding shape differs")
        if any(not value.is_floating_point() for value in (baseline_logits, profile_values)):
            raise TypeError("V27d-r13 model values must be floating point")
        if baseline_logits.dtype != profile_values.dtype or baseline_logits.device != profile_values.device:
            raise ValueError("V27d-r13 model values must share dtype/device")
        if profile_valid.device != baseline_logits.device or padding_mask.device != baseline_logits.device:
            raise ValueError("V27d-r13 masks must share device")
        if not torch.isfinite(baseline_logits).all() or not torch.isfinite(profile_values).all():
            raise ValueError("V27d-r13 model values must be finite")
        if bool(baseline_logits.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r13 padded baseline logits must be zero")
        if bool(profile_values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r13 padded profile values must be zero")
        if bool(profile_values.lt(0.0).any()) or bool(profile_values.gt(1.0).any()):
            raise ValueError("V27d-r13 profile values must be in [0,1]")
        if bool(profile_valid.masked_select(padding_mask).any()):
            raise ValueError("V27d-r13 padded profiles cannot be valid")
        if bool(profile_values.masked_select((~profile_valid).unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r13 invalid profile values must be zero")
        valid_profiles = profile_values[profile_valid]
        if valid_profiles.numel() and bool(valid_profiles.sum(dim=-1).le(0.0).any()):
            raise ValueError("V27d-r13 valid profiles must have positive mass")
        return int(baseline_logits.shape[0]), int(baseline_logits.shape[1])

    def _project_shape_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim < 2 or features.shape[-1] != self.profile_bins * 4:
            raise ValueError("V27d-r13 causal shape feature width differs")
        grouped = features.reshape(*features.shape[:-1], 4, self.profile_bins)
        projected = self.profile_projection(grouped)
        return projected.reshape(*features.shape[:-1], self.projection_dim * 4)

    @staticmethod
    def _correction(delta: torch.Tensor) -> torch.Tensor:
        return torch.stack((delta * 0.5, -delta * 0.5, torch.zeros_like(delta)), dim=-1)

    def initial_state(
        self,
        batch_size: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        if isinstance(batch_size, bool) or int(batch_size) <= 0:
            raise ValueError("V27d-r13 initial state batch size must be positive")
        return {
            "running_count": torch.zeros((int(batch_size), 1), device=device, dtype=dtype),
            "running_sum": torch.zeros((int(batch_size), self.profile_bins), device=device, dtype=dtype),
            "running_square_sum": torch.zeros((int(batch_size), self.profile_bins), device=device, dtype=dtype),
            "previous_profile": torch.zeros((int(batch_size), self.profile_bins), device=device, dtype=dtype),
            "previous_valid": torch.zeros((int(batch_size), 1), device=device, dtype=torch.bool),
            "previous_delta": torch.zeros((int(batch_size),), device=device, dtype=dtype),
        }

    def _validate_state(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        expected = {
            "running_count": (batch_size, 1),
            "running_sum": (batch_size, self.profile_bins),
            "running_square_sum": (batch_size, self.profile_bins),
            "previous_profile": (batch_size, self.profile_bins),
            "previous_valid": (batch_size, 1),
            "previous_delta": (batch_size,),
        }
        for name, shape in expected.items():
            value = state.get(name)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"V27d-r13 continuation state differs: {name}")
            if value.device != device:
                raise ValueError("V27d-r13 continuation state device differs")
            if name == "previous_valid":
                if value.dtype != torch.bool:
                    raise ValueError("V27d-r13 continuation previous_valid must be bool")
            elif value.dtype != dtype or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"V27d-r13 continuation state value differs: {name}")
        if bool(state["previous_delta"].abs().gt(self.pairwise_delta_bound + 1.0e-6).any()):
            raise ValueError("V27d-r13 continuation previous_delta exceeds its bound")

    @staticmethod
    def _normalize_profile(profile: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        total = profile.sum(dim=-1, keepdim=True)
        normalized = profile / total.clamp_min(torch.finfo(profile.dtype).tiny)
        normalized = torch.where(valid.unsqueeze(-1), normalized, torch.zeros_like(normalized))
        if bool(valid.any()) and not torch.isfinite(normalized).all():
            raise FloatingPointError("V27d-r13 normalized profile is non-finite")
        return normalized

    def step_event(
        self,
        *,
        baseline_logits: torch.Tensor,
        profile_values: torch.Tensor,
        profile_valid: torch.Tensor,
        state: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if baseline_logits.ndim != 2 or baseline_logits.shape[-1] != 3:
            raise ValueError("V27d-r13 step baseline logits shape differs")
        if profile_values.ndim != 2 or profile_values.shape[-1] != self.profile_bins:
            raise ValueError("V27d-r13 step profile shape differs")
        if profile_valid.shape != (baseline_logits.shape[0],) or profile_valid.dtype != torch.bool:
            raise ValueError("V27d-r13 step profile validity differs")
        self._validate_state(
            state,
            batch_size=baseline_logits.shape[0],
            device=baseline_logits.device,
            dtype=baseline_logits.dtype,
        )
        if profile_values.dtype != baseline_logits.dtype or profile_values.device != baseline_logits.device:
            raise ValueError("V27d-r13 step values must share dtype/device")
        if bool(profile_values.masked_select((~profile_valid).unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27d-r13 step invalid profile must be zero")
        valid_profiles = profile_values[profile_valid]
        if valid_profiles.numel() and bool(valid_profiles.sum(dim=-1).le(0.0).any()):
            raise ValueError("V27d-r13 step valid profiles must have positive mass")

        current = self._normalize_profile(profile_values, profile_valid)
        count = state["running_count"]
        total = state["running_sum"]
        square = state["running_square_sum"]
        previous = state["previous_profile"]
        previous_valid = state["previous_valid"]
        indicator = profile_valid.unsqueeze(-1).to(baseline_logits.dtype)
        next_count = count + indicator
        next_sum = total + indicator * current
        next_square = square + indicator * current.square()
        mean = next_sum / next_count.clamp_min(1.0)
        std = (next_square / next_count.clamp_min(1.0) - mean.square()).clamp_min(0.0).sqrt()
        delta_shape = torch.where(previous_valid, (current - previous).abs(), torch.zeros_like(current))
        features = torch.cat((current, mean, std, delta_shape), dim=-1)
        features = features.masked_fill(~profile_valid.unsqueeze(-1), 0.0)
        raw = self.shape_mlp(self._project_shape_features(features)).squeeze(-1)
        raw = raw.masked_fill(~profile_valid, 0.0)
        proposed = self.pairwise_delta_bound * torch.tanh(raw)
        applied = torch.where(profile_valid, proposed, torch.zeros_like(proposed))
        next_delta = torch.where(profile_valid, proposed, state["previous_delta"])
        return {
            "sequence_logits": baseline_logits + self._correction(applied),
            "correction_logits": self._correction(applied),
            "pairwise_delta": applied,
            "pairwise_delta_increment": torch.where(
                profile_valid,
                proposed - state["previous_delta"],
                torch.zeros_like(proposed),
            ),
            "raw_pairwise_delta": raw,
            "profile_features": features,
            "next_running_count": next_count,
            "next_running_sum": next_sum,
            "next_running_square_sum": next_square,
            "next_previous_profile": torch.where(indicator.bool(), current, previous),
            "next_previous_valid": previous_valid | profile_valid.unsqueeze(-1),
            "next_previous_delta": next_delta,
        }

    def forward(
        self,
        *,
        baseline_logits: torch.Tensor,
        profile_values: torch.Tensor,
        profile_valid: torch.Tensor,
        padding_mask: torch.Tensor,
        initial_state: Mapping[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        batch, length = self._validate_inputs(
            baseline_logits,
            profile_values,
            profile_valid,
            padding_mask,
        )
        state = (
            self.initial_state(batch, device=baseline_logits.device, dtype=baseline_logits.dtype)
            if initial_state is None
            else dict(initial_state)
        )
        self._validate_state(
            state,
            batch_size=batch,
            device=baseline_logits.device,
            dtype=baseline_logits.dtype,
        )
        current_count = state["running_count"]
        current_sum = state["running_sum"]
        current_square = state["running_square_sum"]
        current_previous = state["previous_profile"]
        current_previous_valid = state["previous_valid"]
        current_delta = state["previous_delta"]
        logits_rows: list[torch.Tensor] = []
        correction_rows: list[torch.Tensor] = []
        delta_rows: list[torch.Tensor] = []
        increment_rows: list[torch.Tensor] = []
        raw_rows: list[torch.Tensor] = []
        feature_rows: list[torch.Tensor] = []
        valid_rows: list[torch.Tensor] = []
        for event in range(length):
            active = ~padding_mask[:, event]
            valid = profile_valid[:, event] & active
            step = self.step_event(
                baseline_logits=baseline_logits[:, event],
                profile_values=profile_values[:, event],
                profile_valid=valid,
                state={
                    "running_count": current_count,
                    "running_sum": current_sum,
                    "running_square_sum": current_square,
                    "previous_profile": current_previous,
                    "previous_valid": current_previous_valid,
                    "previous_delta": current_delta,
                },
            )
            if bool(padding_mask[:, event].any()):
                active_column = active.unsqueeze(-1)
                current_count = torch.where(active_column, step["next_running_count"], current_count)
                current_sum = torch.where(active_column, step["next_running_sum"], current_sum)
                current_square = torch.where(active_column, step["next_running_square_sum"], current_square)
                current_previous = torch.where(active_column, step["next_previous_profile"], current_previous)
                current_previous_valid = torch.where(active_column, step["next_previous_valid"], current_previous_valid)
                current_delta = torch.where(active, step["next_previous_delta"], current_delta)
            else:
                current_count = step["next_running_count"]
                current_sum = step["next_running_sum"]
                current_square = step["next_running_square_sum"]
                current_previous = step["next_previous_profile"]
                current_previous_valid = step["next_previous_valid"]
                current_delta = step["next_previous_delta"]
            pad = padding_mask[:, event]
            logits_rows.append(step["sequence_logits"].masked_fill(pad.unsqueeze(-1), 0.0))
            correction_rows.append(step["correction_logits"].masked_fill(pad.unsqueeze(-1), 0.0))
            delta_rows.append(step["pairwise_delta"].masked_fill(pad, 0.0))
            increment_rows.append(step["pairwise_delta_increment"].masked_fill(pad, 0.0))
            raw_rows.append(step["raw_pairwise_delta"].masked_fill(pad, 0.0))
            feature_rows.append(step["profile_features"].masked_fill(pad.unsqueeze(-1), 0.0))
            valid_rows.append(valid)

        sequence_logits = torch.stack(logits_rows, dim=1)
        correction = torch.stack(correction_rows, dim=1)
        delta = torch.stack(delta_rows, dim=1)
        increment = torch.stack(increment_rows, dim=1)
        raw = torch.stack(raw_rows, dim=1)
        features = torch.stack(feature_rows, dim=1)
        valid_sequence = torch.stack(valid_rows, dim=1)
        if not torch.isfinite(sequence_logits).all() or not torch.isfinite(correction).all():
            raise FloatingPointError("V27d-r13 residual output is non-finite")
        if bool(correction[..., 2].ne(0.0).any()) or bool(correction.sum(dim=-1).abs().gt(1.0e-6).any()):
            raise AssertionError("V27d-r13 correction layout contract failed")
        if bool(correction.abs().gt(self.pairwise_delta_bound / 2.0 + 1.0e-5).any()):
            raise AssertionError("V27d-r13 correction exceeds pairwise bound")
        return {
            "sequence_logits": sequence_logits,
            "baseline_logits": baseline_logits,
            "correction_logits": correction,
            "pairwise_delta_sequence": delta,
            "pairwise_delta_increment": increment,
            "raw_pairwise_delta_sequence": raw,
            "profile_features": features,
            "profile_valid_sequence": valid_sequence,
            "final_state": {
                "running_count": current_count,
                "running_sum": current_sum,
                "running_square_sum": current_square,
                "previous_profile": current_previous,
                "previous_valid": current_previous_valid,
                "previous_delta": current_delta,
            },
        }


def build_model(config: Mapping[str, Any], *, seed: int) -> CausalEnvelopeProfileResidual:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27d-r13 seed must be non-negative")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return CausalEnvelopeProfileResidual(config)


__all__ = ["CausalEnvelopeProfileResidual", "build_model"]
