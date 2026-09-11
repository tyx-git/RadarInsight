"""Bias-free causal TR-motion late residual for V27c."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .protocol import V27CConfig


class V27CMotionResidual(nn.Module):
    """Add a bounded, zero-sum causal correction to frozen V27a Main logits.

    The model deliberately receives no validity mask.  Invalid motion values
    were already converted to exact numeric zero in the sidecar.  Both linear
    layers are bias-free, therefore an all-zero motion sequence cannot create
    a correction merely from its missingness pattern or its length.
    """

    def __init__(self, config: V27CConfig) -> None:
        super().__init__()
        model = config.model
        self.config = config
        self.motion_projection = nn.Linear(
            config.data.motion_dim, model.motion_projection_dim, bias=False
        )
        self.motion_encoder = nn.GRUCell(
            model.motion_projection_dim, model.motion_hidden_dim, bias=model.gru_bias
        )
        self.residual_projection = nn.Linear(
            model.motion_hidden_dim, model.class_count, bias=False
        )
        nn.init.zeros_(self.residual_projection.weight)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def model_contract(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V27c-motion-residual-model-contract-v1",
            "experiment_id": self.config.experiment_id,
            "motion_schema": self.config.data.motion_schema,
            "motion_dim": self.config.data.motion_dim,
            "motion_mask_input": False,
            "motion_gating": "numeric_zero_only",
            "motion_projection_dim": self.config.model.motion_projection_dim,
            "motion_hidden_dim": self.config.model.motion_hidden_dim,
            "token_activation": self.config.model.token_activation,
            "class_count": self.config.model.class_count,
            "token_projection_bias": False,
            "gru_bias": self.config.model.gru_bias,
            "residual_projection_bias": False,
            "residual_zero_initialization": True,
            "residual_zero_sum": True,
            "residual_bound": self.config.model.residual_bound,
            "frozen_endpoint_logits_input_to_motion_encoder": False,
            "padding_updates_motion_state": False,
        }

    def _validate(
        self,
        frozen_endpoint_logits: torch.Tensor,
        motion_values: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        allow_completed_rows: bool,
    ) -> tuple[int, int]:
        if frozen_endpoint_logits.ndim != 3:
            raise ValueError("V27c frozen endpoint logits must have shape [B,T,C]")
        batch, length, classes = frozen_endpoint_logits.shape
        if batch <= 0 or length <= 0 or classes != self.config.model.class_count:
            raise ValueError("V27c frozen endpoint logits shape differs")
        if motion_values.shape != (batch, length, self.config.data.motion_dim):
            raise ValueError("V27c motion values shape differs")
        if padding_mask.shape != (batch, length) or padding_mask.dtype != torch.bool:
            raise ValueError("V27c padding shape/dtype differs")
        if bool((padding_mask[:, :-1] & ~padding_mask[:, 1:]).any()):
            raise ValueError("V27c padding must be a suffix")
        if not all(
            value.is_floating_point() and torch.isfinite(value).all()
            for value in (frozen_endpoint_logits, motion_values)
        ):
            raise ValueError("V27c values must be finite floats")
        if frozen_endpoint_logits.device != motion_values.device or padding_mask.device != motion_values.device:
            raise ValueError("V27c values must share a device")
        if frozen_endpoint_logits.dtype != motion_values.dtype:
            raise ValueError("V27c floating values must share a dtype")
        if bool(frozen_endpoint_logits.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27c padded frozen endpoint logits must be zero")
        if bool(motion_values.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27c padded motion values must be zero")
        if bool((motion_values.abs() > self.config.data.motion_clip_absolute + 1.0e-6).any()):
            raise ValueError("V27c motion values exceed the frozen clip")
        if not allow_completed_rows and bool(padding_mask.all(dim=1).any()):
            raise ValueError("V27c received an empty parent")
        return batch, length

    def _bounded_zero_sum(self, raw: torch.Tensor) -> torch.Tensor:
        centered = raw - raw.mean(dim=-1, keepdim=True)
        maximum = centered.abs().amax(dim=-1, keepdim=True)
        scale = torch.clamp(
            raw.new_tensor(self.config.model.residual_bound)
            / maximum.clamp_min(torch.finfo(raw.dtype).tiny),
            max=1.0,
        )
        return centered * scale

    def forward(
        self,
        *,
        frozen_endpoint_logits: torch.Tensor,
        motion_values: torch.Tensor,
        padding_mask: torch.Tensor,
        initial_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch, length = self._validate(
            frozen_endpoint_logits,
            motion_values,
            padding_mask,
            allow_completed_rows=initial_state is not None,
        )
        if initial_state is None:
            state = motion_values.new_zeros((batch, self.config.model.motion_hidden_dim))
        else:
            if (
                initial_state.shape != (batch, self.config.model.motion_hidden_dim)
                or initial_state.device != motion_values.device
                or initial_state.dtype != motion_values.dtype
                or not torch.isfinite(initial_state).all()
            ):
                raise ValueError("V27c initial state differs")
            state = initial_state
        state_rows: list[torch.Tensor] = []
        residual_rows: list[torch.Tensor] = []
        active_rows: list[torch.Tensor] = []
        for event in range(length):
            active = ~padding_mask[:, event]
            token = motion_values[:, event]
            evidence = token.abs().amax(dim=-1).gt(0.0) & active
            next_state = state
            selected = torch.nonzero(evidence, as_tuple=False).flatten()
            if selected.numel():
                projected = torch.nn.functional.silu(self.motion_projection(token[selected]))
                updated = self.motion_encoder(projected, state[selected])
                next_state = state.index_copy(0, selected, updated)
            state = next_state
            raw_residual = self.residual_projection(state)
            residual = self._bounded_zero_sum(raw_residual).masked_fill(
                (~active).unsqueeze(-1), 0.0
            )
            state_rows.append(state.masked_fill((~active).unsqueeze(-1), 0.0))
            residual_rows.append(residual)
            active_rows.append(evidence)
        state_sequence = torch.stack(state_rows, dim=1)
        residual_sequence = torch.stack(residual_rows, dim=1)
        evidence_mask = torch.stack(active_rows, dim=1)
        logits = frozen_endpoint_logits + residual_sequence
        if not all(torch.isfinite(value).all() for value in (state_sequence, residual_sequence, logits)):
            raise FloatingPointError("V27c motion residual produced non-finite values")
        if bool(residual_sequence.sum(dim=-1).abs().gt(1.0e-6).any()):
            raise AssertionError("V27c residual must be zero-sum across DBB classes")
        if bool((residual_sequence.abs() > self.config.model.residual_bound + 1.0e-6).any()):
            raise AssertionError("V27c residual exceeds its registered bound")
        return {
            "sequence_logits": logits,
            "baseline_logits": frozen_endpoint_logits,
            "residual_logits": residual_sequence,
            "motion_state_sequence": state_sequence,
            "motion_evidence_mask": evidence_mask,
            "final_state": state,
        }


def build_v27c_model(config: V27CConfig, *, seed: int) -> V27CMotionResidual:
    if isinstance(seed, bool) or int(seed) < 0:
        raise ValueError("V27c model seed is invalid")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        return V27CMotionResidual(config)


__all__ = ["V27CMotionResidual", "build_v27c_model"]
