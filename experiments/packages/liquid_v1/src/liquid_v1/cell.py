"""Radar-LN3 第一版液态算法：CfC-style 因果递归编码器（冻结参考实现）。

来源（逐字提取，未改写行为）：src-v3 @ fd34e46610165441d86469802e9ff461d49bd2aa
- CfCLiquidCausalEncoder / LiquidForwardState：src-v3/radar_ln3/model.py
- build_cfc_liquid_encoder：src-v3/radar_ln3/liquid.py（导入路径适配为包内）
- DT_REASON_* 常量：src-v3/radar_ln3/data.py（见 time_policy.py）

该实现把 cell 与序列循环融合在单一 nn.Module 内；离线 forward 与在线 step
共享同一条 _step 更新路径（含 recurrent 归一化与 zero-delta 策略）。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .time_policy import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)


@dataclass
class LiquidForwardState:
    sequence: torch.Tensor
    final_state: torch.Tensor
    dt_clamped_mask: torch.Tensor
    dt_zero_mask: torch.Tensor
    dt_invalid_mask: torch.Tensor
    dt_first_event_mask: torch.Tensor | None = None
    dt_zero_delta_mask: torch.Tensor | None = None


class CfCLiquidCausalEncoder(nn.Module):
    """A stable closed-form liquid-style causal recurrent encoder.

    Each event uses the caller-provided TR wall-clock dt. The update is an
    exponential interpolation toward a learned nonlinear candidate, so no
    future event or sequence-wide attention is used. Invalid or zero dt uses
    the configured floor mode, bounded observation gain, legacy skip
    diagnostic, strict same-second skip, or density-weighted gain; the
    explicit validity mask remains available to the model and callers in all
    modes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        *,
        tau_min_seconds: float = 0.05,
        tau_max_seconds: float = 30.0,
        zero_dt_event_floor_seconds: float = 1.0e-3,
        dt_max_seconds: float = 120.0,
        invalid_dt_update_mode: str = "floor",
        invalid_dt_observation_gain: float = 0.25,
        state_normalization_mode: str = "recurrent",
    ) -> None:
        super().__init__()
        if not 0.0 < tau_min_seconds <= tau_max_seconds:
            raise ValueError("invalid liquid tau bounds")
        self.hidden_dim = int(hidden_dim)
        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = float(tau_max_seconds)
        self.zero_dt_event_floor_seconds = float(max(1.0e-8, zero_dt_event_floor_seconds))
        self.dt_max_seconds = float(max(self.zero_dt_event_floor_seconds, dt_max_seconds))
        self.invalid_dt_update_mode = str(invalid_dt_update_mode)
        if self.invalid_dt_update_mode not in {
            "floor",
            "bounded_observation",
            "skip",
            "strict_skip",
            "density_weighted",
        }:
            raise ValueError(
                "invalid_dt_update_mode must be floor, bounded_observation, "
                "skip, strict_skip, or density_weighted"
            )
        self.invalid_dt_observation_gain = float(invalid_dt_observation_gain)
        if not 0.0 < self.invalid_dt_observation_gain <= 1.0:
            raise ValueError("invalid_dt_observation_gain must be in (0, 1]")
        self.state_normalization_mode = str(state_normalization_mode)
        if self.state_normalization_mode not in {"recurrent", "readout_only"}:
            raise ValueError("state_normalization_mode must be recurrent or readout_only")
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.state_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate_projection = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.candidate_projection = nn.Linear(input_dim + hidden_dim, hidden_dim)
        self.tau_unconstrained = nn.Parameter(torch.zeros(hidden_dim))
        self.output_norm = nn.LayerNorm(hidden_dim)

    def _tau(self) -> torch.Tensor:
        span = self.tau_max_seconds - self.tau_min_seconds
        return self.tau_min_seconds + span * torch.sigmoid(self.tau_unconstrained)

    def _step(
        self,
        event: torch.Tensor,
        dt_seconds: torch.Tensor,
        state: torch.Tensor,
        dt_valid: torch.Tensor | None = None,
        dt_reason: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if event.ndim != 2 or state.ndim != 2 or event.shape[0] != state.shape[0]:
            raise ValueError("liquid event/state shapes are incompatible")
        if dt_seconds.ndim != 1 or dt_seconds.shape[0] != event.shape[0]:
            raise ValueError("liquid dt must have shape [batch]")
        if dt_valid is None:
            dt_valid = torch.ones_like(dt_seconds, dtype=torch.bool)
        if dt_valid.ndim != 1 or dt_valid.shape[0] != event.shape[0]:
            raise ValueError("liquid dt_valid must have shape [batch]")
        if zero_ordinal is not None:
            if zero_ordinal.shape != dt_seconds.shape or zero_ordinal.dtype not in (
                torch.int64,
                torch.long,
            ):
                raise ValueError("liquid zero_ordinal must be long with shape [batch]")
            if bool((zero_ordinal < 1).any()):
                raise ValueError("liquid zero_ordinal must be positive")
        if (
            self.invalid_dt_update_mode == "density_weighted"
            and dt_reason is not None
            and zero_ordinal is None
        ):
            raise ValueError(
                "density_weighted online liquid step requires an explicit causal zero_ordinal"
            )
        if self.invalid_dt_update_mode == "strict_skip" and dt_reason is None:
            raise ValueError(
                "strict_skip liquid updates require an explicit reason-aware dt sequence"
            )
        raw_dt = dt_seconds
        if dt_reason is not None:
            if dt_reason.shape != dt_seconds.shape or dt_reason.dtype not in (torch.int64, torch.long):
                raise ValueError("liquid dt_reason must be long with shape [batch]")
            allowed = (
                (dt_reason == DT_REASON_FIRST_EVENT)
                | (dt_reason == DT_REASON_ZERO_DELTA)
                | (dt_reason == DT_REASON_POSITIVE_DELTA)
            )
            if not bool(allowed.all()):
                raise ValueError("online liquid dt_reason contains padding or an unsupported value")
            expected_valid = dt_reason == DT_REASON_POSITIVE_DELTA
            if not torch.equal(expected_valid, dt_valid):
                raise ValueError("liquid dt_reason and dt_valid are inconsistent")
            zero_reason = (dt_reason == DT_REASON_FIRST_EVENT) | (dt_reason == DT_REASON_ZERO_DELTA)
            if bool(raw_dt.masked_select(zero_reason).abs().gt(1.0e-9).any()):
                raise ValueError("first-event and same-second dt reasons require raw dt=0")
            if bool(raw_dt.masked_select(expected_valid).le(0.0).any()):
                raise ValueError("positive-delta dt reason requires raw dt>0")
        if not torch.isfinite(event).all() or not torch.isfinite(state).all() or not torch.isfinite(dt_seconds).all():
            raise ValueError("liquid inputs must be finite")
        negative_mask = raw_dt < 0.0
        zero_mask = raw_dt.abs() <= 1.0e-9
        safe_dt = raw_dt.clamp(min=0.0, max=self.dt_max_seconds)
        clamped_mask = negative_mask | (raw_dt > self.dt_max_seconds)
        floor_mask = (safe_dt <= 1.0e-9) | (~dt_valid)
        safe_dt = torch.where(floor_mask, torch.full_like(safe_dt, self.zero_dt_event_floor_seconds), safe_dt)
        combined = torch.cat([event, state], dim=-1)
        candidate = torch.tanh(self.candidate_projection(combined) + self.state_projection(state) + self.input_projection(event))
        gate = torch.sigmoid(self.gate_projection(combined))
        candidate = gate * candidate + (1.0 - gate) * state
        tau = self._tau().to(dtype=event.dtype, device=event.device).unsqueeze(0)
        alpha = 1.0 - torch.exp(-safe_dt.unsqueeze(-1) / tau)
        if self.invalid_dt_update_mode in {
            "bounded_observation",
            "skip",
            "density_weighted",
        }:
            observation_gain = torch.full_like(alpha, self.invalid_dt_observation_gain)
            alpha = torch.where((~dt_valid).unsqueeze(-1), observation_gain, alpha)
        if dt_reason is not None:
            # The first event initializes evidence without pretending elapsed
            # time; positive intervals preserve the continuous-time update.
            # The legacy V20 ``skip`` sets the same-second interpolation gain
            # to zero, but recurrent LayerNorm may still transform the state.
            # V24c ``strict_skip`` is intentionally different: it restores the
            # exact incoming state after normalization below.  V20
            # ``density_weighted`` uses the causal same-second ordinal supplied
            # by forward/online.
            first_gain = torch.ones_like(alpha)
            if self.invalid_dt_update_mode == "skip":
                zero_gain = torch.zeros_like(alpha)
            elif self.invalid_dt_update_mode == "density_weighted":
                ordinal = (
                    torch.ones_like(dt_seconds, dtype=torch.long)
                    if zero_ordinal is None
                    else zero_ordinal
                )
                zero_gain = torch.clamp(
                    self.invalid_dt_observation_gain
                    * ordinal.to(dtype=alpha.dtype),
                    max=3.0 * self.invalid_dt_observation_gain,
                ).unsqueeze(-1).expand_as(alpha)
            else:
                zero_gain = torch.full_like(alpha, self.invalid_dt_observation_gain)
            alpha = torch.where((dt_reason == DT_REASON_FIRST_EVENT).unsqueeze(-1), first_gain, alpha)
            alpha = torch.where((dt_reason == DT_REASON_ZERO_DELTA).unsqueeze(-1), zero_gain, alpha)
        next_state = (1.0 - alpha) * state + alpha * candidate
        if self.state_normalization_mode == "recurrent":
            # In legacy V20 skip mode alpha=0 removes event information, while
            # this deterministic affine normalization may still transform
            # state.  V24c's strict_skip override below deliberately bypasses
            # that transformation for each same-second element.
            next_state = self.output_norm(next_state)
        if self.invalid_dt_update_mode == "strict_skip":
            # This is a deliberately narrow, reason-aware diagnostic: a
            # zero_delta event is neither visible in that event's recurrent
            # readout nor carried into any later state.  It is applied after
            # recurrent normalization so the state is bitwise the state that
            # entered this step.  First-event initialization and positive-dt
            # continuous updates remain unchanged.
            strict_skip_mask = (dt_reason == DT_REASON_ZERO_DELTA).unsqueeze(-1)
            next_state = torch.where(strict_skip_mask, state, next_state)
        return next_state, clamped_mask, zero_mask

    def readout(self, state: torch.Tensor) -> torch.Tensor:
        if self.state_normalization_mode == "readout_only":
            return self.output_norm(state)
        return state

    def forward(
        self,
        events: torch.Tensor,
        dt_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
        dt_valid: torch.Tensor | None = None,
        dt_reason: torch.Tensor | None = None,
        initial_state: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> LiquidForwardState:
        if events.ndim != 3:
            raise ValueError("liquid events must have shape [batch, sequence, features]")
        if dt_seconds.shape != events.shape[:2] or padding_mask.shape != events.shape[:2]:
            raise ValueError("liquid dt and padding mask must have shape [batch, sequence]")
        if dt_valid is None:
            dt_valid = torch.ones_like(dt_seconds, dtype=torch.bool)
        if dt_valid.shape != events.shape[:2]:
            raise ValueError("liquid dt_valid must have shape [batch, sequence]")
        if dt_reason is not None:
            if dt_reason.shape != events.shape[:2] or dt_reason.dtype not in (torch.int64, torch.long):
                raise ValueError("liquid dt_reason must be long with shape [batch, sequence]")
            padded_reasons = dt_reason.masked_select(padding_mask)
            active_reasons = dt_reason.masked_select(~padding_mask)
            if bool((padded_reasons != DT_REASON_PADDING).any()):
                raise ValueError("padded liquid events must use the padding dt reason")
            if bool((active_reasons == DT_REASON_PADDING).any()):
                raise ValueError("active liquid events must not use the padding dt reason")
            active = ~padding_mask
            active_counts = active.sum(dim=1)
            if bool((active_counts <= 0).any()):
                raise ValueError("reason-aware liquid sequence cannot be entirely padded")
            first_reason = (dt_reason == DT_REASON_FIRST_EVENT) & active
            first_counts = first_reason.sum(dim=1)
            if initial_state is None:
                first_active_indices = active.to(dtype=torch.int64).argmax(dim=1)
                batch_indices = torch.arange(events.shape[0], device=events.device)
                if bool((first_counts != 1).any()) or bool(
                    (dt_reason[batch_indices, first_active_indices] != DT_REASON_FIRST_EVENT).any()
                ):
                    raise ValueError("fresh liquid sequences require exactly one first-event reason at their first observation")
            elif bool((first_counts != 0).any()):
                raise ValueError("continued liquid sequences must not contain a first-event reason")
        if not torch.isfinite(events).all() or not torch.isfinite(dt_seconds).all():
            raise ValueError("liquid events and dt must be finite")
        batch_size, sequence_length, _ = events.shape
        if initial_state is None:
            state = events.new_zeros((batch_size, self.hidden_dim))
        else:
            if initial_state.shape != (batch_size, self.hidden_dim):
                raise ValueError("invalid liquid initial_state shape")
            state = initial_state
        if zero_ordinal is not None:
            if zero_ordinal.shape != events.shape[:2] or zero_ordinal.dtype not in (
                torch.int64,
                torch.long,
            ):
                raise ValueError(
                    "offline liquid zero_ordinal must be long with shape [batch, sequence]"
                )
            if bool((zero_ordinal.masked_select(padding_mask) != 0).any()):
                raise ValueError("padded offline zero_ordinal values must be zero")
        derived_zero_ordinal = torch.zeros(
            (batch_size,), dtype=torch.long, device=events.device
        )
        outputs: list[torch.Tensor] = []
        clamped: list[torch.Tensor] = []
        zero_dt: list[torch.Tensor] = []
        invalid_dt: list[torch.Tensor] = []
        first_event: list[torch.Tensor] = []
        zero_delta: list[torch.Tensor] = []
        for event_index in range(sequence_length):
            active = ~padding_mask[:, event_index]
            step_reason = None
            step_dt = torch.where(active, dt_seconds[:, event_index], torch.zeros_like(dt_seconds[:, event_index]))
            step_valid = dt_valid[:, event_index] & active
            if dt_reason is not None:
                # `_step` is also the online API and correctly rejects the
                # collate-only padding code. Batch padding is mapped to an
                # invalid first-observation placeholder before the call; the
                # candidate state is discarded by `active` immediately after.
                step_reason = torch.where(
                    active,
                    dt_reason[:, event_index],
                    torch.full_like(dt_reason[:, event_index], DT_REASON_FIRST_EVENT),
                )
            step_ordinal = None
            if dt_reason is not None:
                reason_at_event = step_reason
                reset_ordinal = (reason_at_event == DT_REASON_FIRST_EVENT) | (
                    reason_at_event == DT_REASON_POSITIVE_DELTA
                )
                same_second = reason_at_event == DT_REASON_ZERO_DELTA
                derived_zero_ordinal = torch.where(
                    active & reset_ordinal,
                    torch.ones_like(derived_zero_ordinal),
                    derived_zero_ordinal,
                )
                derived_zero_ordinal = torch.where(
                    active & same_second,
                    derived_zero_ordinal + 1,
                    derived_zero_ordinal,
                )
                step_ordinal = derived_zero_ordinal
                if zero_ordinal is not None:
                    supplied = zero_ordinal[:, event_index]
                    if bool(
                        (
                            supplied.masked_select(active)
                            != step_ordinal.masked_select(active)
                        ).any()
                    ):
                        raise ValueError(
                            "offline zero_ordinal does not match the causal dt_reason sequence"
                        )
            candidate_state, dt_clamped, dt_zero = self._step(
                events[:, event_index],
                step_dt,
                state,
                step_valid,
                step_reason,
                step_ordinal,
            )
            state = torch.where(active.unsqueeze(-1), candidate_state, state)
            outputs.append(self.readout(state))
            clamped.append(dt_clamped & active)
            zero_dt.append(dt_zero & active)
            invalid_dt.append((~dt_valid[:, event_index]) & active)
            first_event.append(
                torch.zeros_like(active)
                if dt_reason is None
                else (dt_reason[:, event_index] == DT_REASON_FIRST_EVENT) & active
            )
            zero_delta.append(
                torch.zeros_like(active)
                if dt_reason is None
                else (dt_reason[:, event_index] == DT_REASON_ZERO_DELTA) & active
            )
        sequence = torch.stack(outputs, dim=1) if outputs else events.new_zeros((batch_size, 0, self.hidden_dim))
        return LiquidForwardState(
            sequence=sequence,
            final_state=state,
            dt_clamped_mask=torch.stack(clamped, dim=1) if clamped else padding_mask.clone(),
            dt_zero_mask=torch.stack(zero_dt, dim=1) if zero_dt else padding_mask.clone(),
            dt_invalid_mask=torch.stack(invalid_dt, dim=1) if invalid_dt else padding_mask.clone(),
            dt_first_event_mask=torch.stack(first_event, dim=1) if first_event else padding_mask.clone(),
            dt_zero_delta_mask=torch.stack(zero_delta, dim=1) if zero_delta else padding_mask.clone(),
        )

    def step(
        self,
        event: torch.Tensor,
        dt_seconds: torch.Tensor,
        state: torch.Tensor | None = None,
        dt_valid: torch.Tensor | None = None,
        dt_reason: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if event.ndim != 2:
            raise ValueError("online liquid event must have shape [batch, features]")
        fresh_state = state is None
        if dt_reason is not None:
            first_reason = dt_reason == DT_REASON_FIRST_EVENT
            if fresh_state and not bool(first_reason.all()):
                raise ValueError("a fresh online liquid state requires the first-event reason")
            if not fresh_state and bool(first_reason.any()):
                raise ValueError("an existing online liquid state cannot consume another first-event reason")
        if state is None:
            state = event.new_zeros((event.shape[0], self.hidden_dim))
        return self._step(event, dt_seconds, state, dt_valid, dt_reason, zero_ordinal)


def build_cfc_liquid_encoder(
    input_dim: int,
    hidden_dim: int,
    *,
    tau_min_seconds: float,
    tau_max_seconds: float,
    zero_dt_event_floor_seconds: float,
    dt_max_seconds: float,
    invalid_dt_update_mode: str,
    invalid_dt_observation_gain: float,
    state_normalization_mode: str,
) -> CfCLiquidCausalEncoder:
    """构建自研因果液态编码器；全部动力学参数必须显式绑定，不提供默认值。"""

    return CfCLiquidCausalEncoder(
        input_dim,
        hidden_dim,
        tau_min_seconds=tau_min_seconds,
        tau_max_seconds=tau_max_seconds,
        zero_dt_event_floor_seconds=zero_dt_event_floor_seconds,
        dt_max_seconds=dt_max_seconds,
        invalid_dt_update_mode=invalid_dt_update_mode,
        invalid_dt_observation_gain=invalid_dt_observation_gain,
        state_normalization_mode=state_normalization_mode,
    )


__all__ = ["CfCLiquidCausalEncoder", "LiquidForwardState", "build_cfc_liquid_encoder"]
