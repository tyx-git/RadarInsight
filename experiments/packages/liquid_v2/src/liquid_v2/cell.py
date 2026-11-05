"""第二版液态算法单元（raw-memory / pre-norm，V27）。

来源：src-v3 @ fd34e46610165441d86469802e9ff461d49bd2aa，src-v3/radar_ln3/liquid_v2.py 逐字拆分。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .time_policy import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)

EQUATION_VERSION = "RADAR-LN3-V27-liquid-v2-raw-memory-pre-norm-v1"

TRACE_FIELD_NAMES = (
    "raw_state_before_update",
    "raw_state",
    "normalized_compute_state",
    "candidate",
    "gate",
    "mixed_memory",
    "candidate_innovation",
    "mixed_memory_innovation",
    "actual_update",
    "alpha",
    "tau",
    "dt_reason",
    "zero_ordinal",
    "valid_mask",
)


@dataclass
class LiquidV2StepOutput:
    next_state: torch.Tensor
    normalized_compute_state: torch.Tensor
    candidate: torch.Tensor
    gate: torch.Tensor
    mixed_memory: torch.Tensor
    candidate_innovation: torch.Tensor
    mixed_memory_innovation: torch.Tensor
    actual_update: torch.Tensor
    alpha: torch.Tensor
    tau: torch.Tensor
    dt_clamped_mask: torch.Tensor
    dt_zero_mask: torch.Tensor
    dt_invalid_mask: torch.Tensor
    dt_reason: torch.Tensor


class LiquidV2Cell(nn.Module):
    """One raw-memory Liquid V2 cell."""

    SUPPORTED_UPDATE_MODES = frozenset(
        {"floor", "bounded_observation", "skip", "strict_skip", "density_weighted"}
    )

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        tau_min_seconds: float,
        tau_max_seconds: float,
        zero_dt_event_floor_seconds: float,
        dt_max_seconds: float,
        invalid_dt_update_mode: str,
        invalid_dt_observation_gain: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("Liquid V2 dimensions must be positive")
        if not 0.0 < float(tau_min_seconds) <= float(tau_max_seconds):
            raise ValueError("Liquid V2 tau bounds are invalid")
        if float(zero_dt_event_floor_seconds) <= 0.0:
            raise ValueError("Liquid V2 zero-dt floor must be positive")
        if float(dt_max_seconds) < float(zero_dt_event_floor_seconds):
            raise ValueError("Liquid V2 dt_max is below the zero-dt floor")
        mode = str(invalid_dt_update_mode)
        if mode not in self.SUPPORTED_UPDATE_MODES:
            raise ValueError(f"unsupported Liquid V2 update mode: {mode}")
        gain = float(invalid_dt_observation_gain)
        if not 0.0 < gain <= 1.0:
            raise ValueError("Liquid V2 invalid dt observation gain must be in (0,1]")

        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = float(tau_max_seconds)
        self.zero_dt_event_floor_seconds = float(zero_dt_event_floor_seconds)
        self.dt_max_seconds = float(dt_max_seconds)
        self.invalid_dt_update_mode = mode
        self.invalid_dt_observation_gain = gain

        # ``compute_norm`` is never used as recurrent memory.  The classifier
        # owns another independent LayerNorm in the V27 model.
        self.compute_norm = nn.LayerNorm(self.hidden_dim)
        feature_dim = self.input_dim + self.hidden_dim
        # Keep the V1 candidate-function capacity unchanged.  The state
        # equation is the V27 intervention; removing these two projections
        # would silently turn it into a second architecture intervention.
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        self.state_projection = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        self.candidate_projection = nn.Linear(feature_dim, self.hidden_dim)
        self.gate_projection = nn.Linear(feature_dim, self.hidden_dim)
        self.tau_unconstrained = nn.Parameter(torch.zeros(self.hidden_dim))

    def tau(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        span = self.tau_max_seconds - self.tau_min_seconds
        value = self.tau_min_seconds + span * torch.sigmoid(self.tau_unconstrained)
        return value.to(dtype=dtype, device=device)

    def _validate_inputs(
        self,
        event: torch.Tensor,
        state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        zero_ordinal: torch.Tensor,
    ) -> None:
        if event.ndim != 2 or event.shape[-1] != self.input_dim:
            raise ValueError(
                f"Liquid V2 event must have shape [N,{self.input_dim}], got {tuple(event.shape)}"
            )
        batch = int(event.shape[0])
        if state.shape != (batch, self.hidden_dim):
            raise ValueError("Liquid V2 state shape does not match the cell")
        if not event.is_floating_point() or not state.is_floating_point() or not dt_seconds.is_floating_point():
            raise TypeError("Liquid V2 event, state and dt must use floating-point tensors")
        if state.device != event.device or dt_seconds.device != event.device:
            raise ValueError("Liquid V2 event, state and dt must share a device")
        if state.dtype != event.dtype or dt_seconds.dtype != event.dtype:
            raise ValueError("Liquid V2 event, state and dt must share a dtype")
        for name, value in (
            ("dt_seconds", dt_seconds),
            ("dt_valid", dt_valid),
            ("dt_reason", dt_reason),
            ("zero_ordinal", zero_ordinal),
        ):
            if value.ndim != 1 or value.shape[0] != batch:
                raise ValueError(f"Liquid V2 {name} must have shape [N]")
            if value.device != event.device:
                raise ValueError(f"Liquid V2 {name} must share the event device")
        if dt_valid.dtype != torch.bool:
            raise ValueError("Liquid V2 dt_valid must be boolean")
        if dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 dt_reason must be int64")
        if zero_ordinal.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 zero_ordinal must be int64")
        if bool((zero_ordinal < 1).any()):
            raise ValueError("Liquid V2 zero_ordinal must be positive")
        if not all(torch.isfinite(value).all() for value in (event, state, dt_seconds)):
            raise ValueError("Liquid V2 event, state and dt must be finite")
        allowed = (
            (dt_reason == DT_REASON_FIRST_EVENT)
            | (dt_reason == DT_REASON_ZERO_DELTA)
            | (dt_reason == DT_REASON_POSITIVE_DELTA)
        )
        if not bool(allowed.all()):
            raise ValueError("Liquid V2 active dt_reason contains padding")
        expected_valid = dt_reason == DT_REASON_POSITIVE_DELTA
        if not torch.equal(expected_valid, dt_valid):
            raise ValueError("Liquid V2 dt_reason and dt_valid are inconsistent")
        zero_reason = (dt_reason == DT_REASON_FIRST_EVENT) | (
            dt_reason == DT_REASON_ZERO_DELTA
        )
        if bool(dt_seconds.masked_select(zero_reason).abs().gt(1.0e-9).any()):
            raise ValueError("Liquid V2 first/zero-delta reasons require dt=0")
        if bool(dt_seconds.masked_select(expected_valid).le(0.0).any()):
            raise ValueError("Liquid V2 positive-delta reason requires dt>0")

    def _time_parameters(
        self,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        zero_ordinal: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_dt = dt_seconds
        clamped = (raw_dt < 0.0) | (raw_dt > self.dt_max_seconds)
        safe_dt = raw_dt.clamp(min=0.0, max=self.dt_max_seconds)
        floor = (safe_dt <= 1.0e-9) | (~dt_valid)
        safe_dt = torch.where(
            floor,
            torch.full_like(safe_dt, self.zero_dt_event_floor_seconds),
            safe_dt,
        )
        tau = self.tau(dtype=raw_dt.dtype, device=raw_dt.device)
        tau_batch = tau.unsqueeze(0).expand(raw_dt.shape[0], -1)
        alpha = 1.0 - torch.exp(-safe_dt.unsqueeze(-1) / tau_batch)

        if self.invalid_dt_update_mode in {
            "bounded_observation",
            "skip",
            "strict_skip",
            "density_weighted",
        }:
            observation_gain = torch.full_like(alpha, self.invalid_dt_observation_gain)
            alpha = torch.where((~dt_valid).unsqueeze(-1), observation_gain, alpha)

        first = dt_reason == DT_REASON_FIRST_EVENT
        zero = dt_reason == DT_REASON_ZERO_DELTA
        if self.invalid_dt_update_mode == "skip":
            zero_gain = torch.zeros_like(alpha)
        elif self.invalid_dt_update_mode == "density_weighted":
            zero_gain = torch.clamp(
                self.invalid_dt_observation_gain * zero_ordinal.to(alpha.dtype),
                max=3.0 * self.invalid_dt_observation_gain,
            ).unsqueeze(-1).expand_as(alpha)
        else:
            zero_gain = torch.full_like(alpha, self.invalid_dt_observation_gain)
        alpha = torch.where(first.unsqueeze(-1), torch.ones_like(alpha), alpha)
        alpha = torch.where(zero.unsqueeze(-1), zero_gain, alpha).clamp(0.0, 1.0)
        return alpha, tau_batch, clamped, raw_dt.abs() <= 1.0e-9, ~dt_valid

    def _compute_step(
        self,
        event: torch.Tensor,
        state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        zero_ordinal: torch.Tensor,
    ) -> LiquidV2StepOutput:
        normalized = self.compute_norm(state)
        combined = torch.cat((event, normalized), dim=-1)
        candidate = torch.tanh(
            self.candidate_projection(combined)
            + self.input_projection(event)
            + self.state_projection(normalized)
        )
        gate = torch.sigmoid(self.gate_projection(combined))
        mixed = gate * candidate + (1.0 - gate) * state
        alpha, tau, clamped, zero_dt, invalid_dt = self._time_parameters(
            dt_seconds, dt_valid, dt_reason, zero_ordinal
        )
        next_state = (1.0 - alpha) * state + alpha * mixed
        candidate_innovation = candidate - state
        mixed_innovation = mixed - state
        actual_update = next_state - state
        values = (
            normalized,
            candidate,
            gate,
            mixed,
            next_state,
            candidate_innovation,
            mixed_innovation,
            actual_update,
            alpha,
            tau,
        )
        if not all(torch.isfinite(value).all() for value in values):
            raise FloatingPointError("Liquid V2 produced a non-finite value")
        return LiquidV2StepOutput(
            next_state=next_state,
            normalized_compute_state=normalized,
            candidate=candidate,
            gate=gate,
            mixed_memory=mixed,
            candidate_innovation=candidate_innovation,
            mixed_memory_innovation=mixed_innovation,
            actual_update=actual_update,
            alpha=alpha,
            tau=tau,
            dt_clamped_mask=clamped,
            dt_zero_mask=zero_dt,
            dt_invalid_mask=invalid_dt,
            dt_reason=dt_reason,
        )

    def skipped_step(
        self,
        state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        zero_ordinal: torch.Tensor,
    ) -> LiquidV2StepOutput:
        """Return an exact identity for strict-skip rows without calling F."""

        if state.ndim != 2 or state.shape[-1] != self.hidden_dim:
            raise ValueError("Liquid V2 skipped state has an invalid shape")
        self._validate_inputs(
            torch.zeros(
                (state.shape[0], self.input_dim), dtype=state.dtype, device=state.device
            ),
            state,
            dt_seconds,
            dt_valid,
            dt_reason,
            zero_ordinal,
        )
        if bool((dt_reason != DT_REASON_ZERO_DELTA).any()):
            raise ValueError("Liquid V2 skipped_step accepts only zero-delta rows")
        zeros = state.new_zeros((state.shape[0], self.hidden_dim))
        tau = self.tau(dtype=state.dtype, device=state.device).unsqueeze(0).expand_as(zeros)
        return LiquidV2StepOutput(
            next_state=state,
            normalized_compute_state=zeros.clone(),
            candidate=zeros.clone(),
            gate=zeros.clone(),
            mixed_memory=zeros.clone(),
            candidate_innovation=zeros.clone(),
            mixed_memory_innovation=zeros.clone(),
            actual_update=zeros.clone(),
            alpha=zeros.clone(),
            tau=tau,
            dt_clamped_mask=torch.zeros(
                (state.shape[0],), dtype=torch.bool, device=state.device
            ),
            dt_zero_mask=torch.ones(
                (state.shape[0],), dtype=torch.bool, device=state.device
            ),
            dt_invalid_mask=torch.ones(
                (state.shape[0],), dtype=torch.bool, device=state.device
            ),
            dt_reason=dt_reason,
        )

    def step(
        self,
        event: torch.Tensor,
        state: torch.Tensor,
        dt_seconds: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        zero_ordinal: torch.Tensor,
    ) -> LiquidV2StepOutput:
        """Compute active rows, explicitly bypassing strict-skip rows."""

        self._validate_inputs(
            event, state, dt_seconds, dt_valid, dt_reason, zero_ordinal
        )
        if self.invalid_dt_update_mode != "strict_skip":
            return self._compute_step(
                event, state, dt_seconds, dt_valid, dt_reason, zero_ordinal
            )
        skip_mask = dt_reason == DT_REASON_ZERO_DELTA
        if not bool(skip_mask.any()):
            return self._compute_step(
                event, state, dt_seconds, dt_valid, dt_reason, zero_ordinal
            )

        skip_indices = torch.nonzero(skip_mask, as_tuple=False).flatten()
        compute_indices = torch.nonzero(~skip_mask, as_tuple=False).flatten()
        skipped = self.skipped_step(
            state.index_select(0, skip_indices),
            dt_seconds.index_select(0, skip_indices),
            dt_valid.index_select(0, skip_indices),
            dt_reason.index_select(0, skip_indices),
            zero_ordinal.index_select(0, skip_indices),
        )
        computed = (
            self._compute_step(
                event.index_select(0, compute_indices),
                state.index_select(0, compute_indices),
                dt_seconds.index_select(0, compute_indices),
                dt_valid.index_select(0, compute_indices),
                dt_reason.index_select(0, compute_indices),
                zero_ordinal.index_select(0, compute_indices),
            )
            if compute_indices.numel()
            else None
        )

        def merge_state(name: str) -> torch.Tensor:
            result = state.new_zeros(state.shape)
            if computed is not None:
                result = result.index_copy(0, compute_indices, getattr(computed, name))
            result = result.index_copy(0, skip_indices, getattr(skipped, name))
            return result

        def merge_bool(name: str) -> torch.Tensor:
            result = torch.zeros(
                (state.shape[0],), dtype=torch.bool, device=state.device
            )
            if computed is not None:
                result = result.index_copy(0, compute_indices, getattr(computed, name))
            return result.index_copy(0, skip_indices, getattr(skipped, name))

        reason = torch.zeros(
            (state.shape[0],), dtype=torch.long, device=state.device
        )
        if computed is not None:
            reason = reason.index_copy(0, compute_indices, computed.dt_reason)
        reason = reason.index_copy(0, skip_indices, skipped.dt_reason)
        return LiquidV2StepOutput(
            next_state=merge_state("next_state"),
            normalized_compute_state=merge_state("normalized_compute_state"),
            candidate=merge_state("candidate"),
            gate=merge_state("gate"),
            mixed_memory=merge_state("mixed_memory"),
            candidate_innovation=merge_state("candidate_innovation"),
            mixed_memory_innovation=merge_state("mixed_memory_innovation"),
            actual_update=merge_state("actual_update"),
            alpha=merge_state("alpha"),
            tau=merge_state("tau"),
            dt_clamped_mask=merge_bool("dt_clamped_mask"),
            dt_zero_mask=merge_bool("dt_zero_mask"),
            dt_invalid_mask=merge_bool("dt_invalid_mask"),
            dt_reason=reason,
        )


__all__ = ["EQUATION_VERSION", "TRACE_FIELD_NAMES", "LiquidV2StepOutput", "LiquidV2Cell"]
