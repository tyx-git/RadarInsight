"""Liquid V2 递归运行时。

产品状态是 raw memory ``r``。只有在计算 candidate 和 gate 时，才从输入状态
生成 normalized view：

    n = LayerNorm(r_previous)
    (c, g) = F(x, n)
    m = g*c + (1-g)*r_previous
    r_next = (1-alpha)*r_previous + alpha*m

更新后不再执行 normalization。本模块独立于历史 V1/task-model 模块，且不
包含类别或 action 语义。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from radar_ln3.data.temporal import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)


EQUATION_VERSION = "RADAR-LN3-liquid-v2-raw-memory-pre-norm-v1"
TIME_POLICY_VERSION = "RADAR-LN3-reason-aware-time-policy-v2"
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


@dataclass
class LiquidV2SequenceOutput:
    """raw state 序列以及可选的诊断轨迹。"""

    sequence: torch.Tensor
    final_state: torch.Tensor
    dt_clamped_mask: torch.Tensor
    dt_zero_mask: torch.Tensor
    dt_invalid_mask: torch.Tensor
    dt_first_event_mask: torch.Tensor
    dt_zero_delta_mask: torch.Tensor
    trace: dict[str, torch.Tensor] | None = None


def time_policy_contract(
    *,
    tau_min_seconds: float,
    tau_max_seconds: float,
    zero_dt_event_floor_seconds: float,
    dt_max_seconds: float,
    invalid_dt_update_mode: str,
    invalid_dt_observation_gain: float,
) -> dict[str, Any]:
    """序列化 Liquid V2 cell 使用的全部时间动力学选择。"""

    mode = str(invalid_dt_update_mode)
    mode_rules = {
        "floor": {
            "zero_delta": "使用 zero_dt_event_floor_seconds 通过指数公式计算 alpha",
            "invalid_dt": "使用 zero_dt_event_floor_seconds 通过指数公式计算 alpha",
        },
        "bounded_observation": {
            "zero_delta": "使用 invalid_dt_observation_gain 进行观测更新",
            "invalid_dt": "使用 invalid_dt_observation_gain 进行观测更新",
        },
        "skip": {
            "zero_delta": "恒等更新；跳过 candidate/gate function",
            "invalid_dt": "使用 invalid_dt_observation_gain 进行观测更新",
        },
        "strict_skip": {
            "zero_delta": "恒等更新；跳过 candidate/gate function",
            "invalid_dt": "使用 invalid_dt_observation_gain 进行观测更新",
        },
        "density_weighted": {
            "zero_delta": "使用 clip(invalid_dt_observation_gain * zero_ordinal, 0, 3 * invalid_dt_observation_gain)",
            "invalid_dt": "使用 invalid_dt_observation_gain 进行观测更新",
        },
    }
    if mode not in mode_rules:
        raise ValueError(f"不支持的 Liquid V2 time-policy mode：{mode}")
    return {
        "version": TIME_POLICY_VERSION,
        "tau_min_seconds": float(tau_min_seconds),
        "tau_max_seconds": float(tau_max_seconds),
        "zero_dt_event_floor_seconds": float(zero_dt_event_floor_seconds),
        "dt_max_seconds": float(dt_max_seconds),
        "invalid_dt_update_mode": mode,
        "active_mode": mode,
        "invalid_dt_observation_gain": float(invalid_dt_observation_gain),
        "dt_reason_ids": {
            "padding": DT_REASON_PADDING,
            "first_event_no_predecessor": DT_REASON_FIRST_EVENT,
            "zero_delta_at_second_precision": DT_REASON_ZERO_DELTA,
            "positive_delta_at_second_precision": DT_REASON_POSITIVE_DELTA,
        },
        "first_event_rule": "alpha=1; 当前观测初始化 evidence",
        "zero_delta_rule": mode_rules[mode]["zero_delta"],
        "positive_delta_rule": "alpha=1-exp(-positive_dt/tau)",
        "padding_rule": "padding 不调用 cell，且保持 raw memory 不变",
        "active_zero_delta_rule": mode_rules[mode]["zero_delta"],
        "active_invalid_dt_rule": mode_rules[mode]["invalid_dt"],
        "supported_modes": sorted(mode_rules),
        "mode_rules": mode_rules,
        "strict_skip_rule": "仅支持该 mode：zero-delta 行执行恒等更新，不进入 F",
    }


class LiquidV2Cell(nn.Module):
    """一个使用 raw memory 的 Liquid V2 cell。"""

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
            raise ValueError("Liquid V2 维度必须为正")
        if not 0.0 < float(tau_min_seconds) <= float(tau_max_seconds):
            raise ValueError("Liquid V2 tau 边界无效")
        if float(zero_dt_event_floor_seconds) <= 0.0:
            raise ValueError("Liquid V2 zero-dt floor 必须为正")
        if float(dt_max_seconds) < float(zero_dt_event_floor_seconds):
            raise ValueError("Liquid V2 dt_max 不能小于 zero-dt floor")
        mode = str(invalid_dt_update_mode)
        if mode not in self.SUPPORTED_UPDATE_MODES:
            raise ValueError(f"不支持的 Liquid V2 update mode：{mode}")
        gain = float(invalid_dt_observation_gain)
        if not 0.0 < gain <= 1.0:
            raise ValueError("Liquid V2 invalid dt observation gain 必须位于 (0,1]")

        self.tau_min_seconds = float(tau_min_seconds)
        self.tau_max_seconds = float(tau_max_seconds)
        self.zero_dt_event_floor_seconds = float(zero_dt_event_floor_seconds)
        self.dt_max_seconds = float(dt_max_seconds)
        self.invalid_dt_update_mode = mode
        self.invalid_dt_observation_gain = gain

        # compute_norm 不作为递归记忆；task model 的 classifier 另有独立的
        # LayerNorm。
        self.compute_norm = nn.LayerNorm(self.hidden_dim)
        feature_dim = self.input_dim + self.hidden_dim
        # 保持 candidate function 的容量不变。登记的改动只涉及 state equation；
        # 删除这两个 projection 会无意中引入第二项 architecture 改动。
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
                f"Liquid V2 event shape 必须为 [N,{self.input_dim}]，实际为 {tuple(event.shape)}"
            )
        batch = int(event.shape[0])
        if state.shape != (batch, self.hidden_dim):
            raise ValueError("Liquid V2 state shape 与 cell 不一致")
        if not event.is_floating_point() or not state.is_floating_point() or not dt_seconds.is_floating_point():
            raise TypeError("Liquid V2 event、state 和 dt 必须使用浮点 tensor")
        if state.device != event.device or dt_seconds.device != event.device:
            raise ValueError("Liquid V2 event、state 和 dt 必须使用同一 device")
        if state.dtype != event.dtype or dt_seconds.dtype != event.dtype:
            raise ValueError("Liquid V2 event、state 和 dt 必须使用同一 dtype")
        for name, value in (
            ("dt_seconds", dt_seconds),
            ("dt_valid", dt_valid),
            ("dt_reason", dt_reason),
            ("zero_ordinal", zero_ordinal),
        ):
            if value.ndim != 1 or value.shape[0] != batch:
                raise ValueError(f"Liquid V2 {name} 的 shape 必须为 [N]")
            if value.device != event.device:
                raise ValueError(f"Liquid V2 {name} 必须与 event 使用同一 device")
        if dt_valid.dtype != torch.bool:
            raise ValueError("Liquid V2 dt_valid 必须是布尔类型")
        if dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 dt_reason 必须是 int64")
        if zero_ordinal.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 zero_ordinal 必须是 int64")
        if bool((zero_ordinal < 1).any()):
            raise ValueError("Liquid V2 zero_ordinal 必须为正整数")
        if not all(torch.isfinite(value).all() for value in (event, state, dt_seconds)):
            raise ValueError("Liquid V2 event、state 和 dt 必须是有限值")
        allowed = (
            (dt_reason == DT_REASON_FIRST_EVENT)
            | (dt_reason == DT_REASON_ZERO_DELTA)
            | (dt_reason == DT_REASON_POSITIVE_DELTA)
        )
        if not bool(allowed.all()):
            raise ValueError("Liquid V2 active dt_reason 不能包含 padding")
        expected_valid = dt_reason == DT_REASON_POSITIVE_DELTA
        if not torch.equal(expected_valid, dt_valid):
            raise ValueError("Liquid V2 dt_reason 与 dt_valid 不一致")
        zero_reason = (dt_reason == DT_REASON_FIRST_EVENT) | (
            dt_reason == DT_REASON_ZERO_DELTA
        )
        if bool(dt_seconds.masked_select(zero_reason).abs().gt(1.0e-9).any()):
            raise ValueError("Liquid V2 first/zero-delta reason 要求 dt=0")
        if bool(dt_seconds.masked_select(expected_valid).le(0.0).any()):
            raise ValueError("Liquid V2 positive-delta reason 要求 dt>0")

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
            raise FloatingPointError("Liquid V2 产生了非有限值")
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
        """对 strict-skip 行返回不调用 F 的精确恒等更新。"""

        if state.ndim != 2 or state.shape[-1] != self.hidden_dim:
            raise ValueError("Liquid V2 跳过状态的 shape 无效")
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
            raise ValueError("Liquid V2 skipped_step 只接受 zero-delta 行")
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
        """计算 active 行，并显式跳过 strict-skip 行。"""

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


class LiquidV2SequenceWrapper(nn.Module):
    """变长离线 wrapper 与在线 continuation 边界。"""

    def __init__(self, cell: LiquidV2Cell) -> None:
        super().__init__()
        self.cell = cell

    @property
    def hidden_dim(self) -> int:
        return self.cell.hidden_dim

    def _validate_sequence(
        self,
        events: torch.Tensor,
        dt_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
    ) -> None:
        if events.ndim != 3 or events.shape[-1] != self.cell.input_dim:
            raise ValueError("Liquid V2 events 的 shape 必须为 [B,T,input_dim]")
        if events.shape[0] <= 0 or events.shape[1] <= 0:
            raise ValueError("Liquid V2 sequence 的 batch 和 time 维度必须为正")
        expected = events.shape[:2]
        for name, value in (
            ("dt_seconds", dt_seconds),
            ("padding_mask", padding_mask),
            ("dt_valid", dt_valid),
            ("dt_reason", dt_reason),
        ):
            if value.shape != expected:
                raise ValueError(f"Liquid V2 {name} 的 shape 必须为 [B,T]")
        if padding_mask.dtype != torch.bool or dt_valid.dtype != torch.bool:
            raise ValueError("Liquid V2 padding_mask 和 dt_valid 必须是布尔类型")
        if dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 dt_reason 必须是 int64")
        if not events.is_floating_point() or not dt_seconds.is_floating_point():
            raise TypeError("Liquid V2 events 和 dt 必须使用浮点 tensor")
        if dt_seconds.device != events.device or padding_mask.device != events.device:
            raise ValueError("Liquid V2 sequence tensor 必须与 event 使用同一 device")
        if dt_valid.device != events.device or dt_reason.device != events.device:
            raise ValueError("Liquid V2 sequence time mask 必须与 event 使用同一 device")
        if not all(torch.isfinite(value).all() for value in (events, dt_seconds)):
            raise ValueError("Liquid V2 events 和 dt 必须是有限值")
        padded = padding_mask
        if bool((dt_reason.masked_select(padded) != DT_REASON_PADDING).any()):
            raise ValueError("padded Liquid V2 event 必须使用 reason=padding")
        if bool(dt_valid.masked_select(padded).any()):
            raise ValueError("padded Liquid V2 event 的 dt_valid 必须为 false")
        if bool(dt_seconds.masked_select(padded).abs().gt(1.0e-9).any()):
            raise ValueError("padded Liquid V2 event 的 dt 必须为 0")
        # collate contract 规定前缀 active、后缀 padding。padding 后再次出现
        # active event 会让同一个 tensor 表示两条不同的因果时间线。
        for row in range(padding_mask.shape[0]):
            padded_indices = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded_indices.numel():
                first_padding = int(padded_indices[0].item())
                if bool((~padding_mask[row, first_padding:]).any()):
                    raise ValueError("Liquid V2 padding 必须是每条 sequence 的后缀")
        active = ~padding_mask
        active_reasons = dt_reason.masked_select(active)
        if bool((active_reasons == DT_REASON_PADDING).any()):
            raise ValueError("active Liquid V2 event 不能使用 reason=padding")
        if bool(
            ((dt_reason == DT_REASON_POSITIVE_DELTA) != dt_valid)
            .masked_select(active)
            .any()
        ):
            raise ValueError("active Liquid V2 dt_reason 与 dt_valid 不一致")

    @staticmethod
    def derive_zero_ordinals(
        dt_reason: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        supplied: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dt_reason.shape != padding_mask.shape:
            raise ValueError("Liquid V2 ordinal 输入的 shape 不一致")
        result = torch.zeros_like(dt_reason, dtype=torch.long)
        if supplied is not None:
            if supplied.shape != dt_reason.shape or supplied.dtype not in (
                torch.int64,
                torch.long,
            ):
                raise ValueError("Liquid V2 zero_ordinal 必须是 shape=[B,T] 的 int64")
            if bool((supplied < 0).any()):
                raise ValueError("Liquid V2 zero_ordinal 不能为负数")
            if bool((supplied.masked_select(padding_mask) != 0).any()):
                raise ValueError("padded Liquid V2 zero_ordinal 必须为 0")
            if supplied.device != dt_reason.device:
                raise ValueError("Liquid V2 zero_ordinal 必须与 sequence 使用同一 device")

        for row in range(dt_reason.shape[0]):
            active_indices = torch.nonzero(~padding_mask[row], as_tuple=False).flatten()
            if not active_indices.numel():
                continue
            previous_reason: int | None = None
            previous_ordinal: int | None = None
            for event_index_tensor in active_indices:
                event_index = int(event_index_tensor.item())
                reason = int(dt_reason[row, event_index].item())
                if supplied is not None:
                    ordinal = int(supplied[row, event_index].item())
                    if reason == DT_REASON_POSITIVE_DELTA:
                        if ordinal != 1:
                            raise ValueError(
                                "positive-delta Liquid V2 event 必须将 zero_ordinal 重置为 1"
                            )
                    elif reason == DT_REASON_ZERO_DELTA:
                        if ordinal < 1:
                            raise ValueError("zero-delta Liquid V2 ordinal 必须为正数")
                        if previous_reason == DT_REASON_ZERO_DELTA and ordinal != int(previous_ordinal) + 1:
                            raise ValueError(
                                "连续 zero-delta Liquid V2 ordinal 必须递增 1"
                            )
                    elif reason == DT_REASON_FIRST_EVENT and ordinal != 1:
                        raise ValueError("first-event Liquid V2 ordinal 必须为 1")
                    result[row, event_index] = ordinal
                    previous_ordinal = ordinal
                else:
                    if initial_state is not None and reason == DT_REASON_ZERO_DELTA:
                        raise ValueError(
                            "连续 density-aware Liquid V2 chunk 必须显式提供 zero_ordinal"
                        )
                    if reason in (DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA):
                        previous_ordinal = 1
                    elif reason == DT_REASON_ZERO_DELTA:
                        previous_ordinal = int(previous_ordinal or 0) + 1
                    else:
                        raise ValueError("不支持的 active Liquid V2 dt reason")
                    result[row, event_index] = int(previous_ordinal)
                previous_reason = reason
        return result

    def _validate_first_event_contract(
        self,
        dt_reason: torch.Tensor,
        padding_mask: torch.Tensor,
        initial_state: torch.Tensor | None,
    ) -> None:
        active = ~padding_mask
        # batched causal sequence 使用后缀 padding。padding 后的 active event
        # 不能被静默视为后续事件，否则 ordinal/time contract 会失真。
        for row in range(padding_mask.shape[0]):
            padded_indices = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded_indices.numel():
                first_padding = int(padded_indices[0].item())
                if bool((~padding_mask[row, first_padding:]).any()):
                    raise ValueError("Liquid V2 padding 必须是每条 sequence 的后缀")
        for row in range(dt_reason.shape[0]):
            indices = torch.nonzero(active[row], as_tuple=False).flatten()
            if not indices.numel():
                continue
            reasons = dt_reason[row, indices]
            first_count = int((reasons == DT_REASON_FIRST_EVENT).sum().item())
            if initial_state is None:
                if first_count != 1 or int(reasons[0].item()) != DT_REASON_FIRST_EVENT:
                    raise ValueError(
                        "新的 Liquid V2 sequence 必须恰好包含一个 first-event reason"
                    )
            elif first_count:
                raise ValueError("连续 Liquid V2 sequence 不能包含 first-event reason")

    def forward(
        self,
        events: torch.Tensor,
        dt_seconds: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        initial_state: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
        return_trace: bool = False,
    ) -> LiquidV2SequenceOutput:
        self._validate_sequence(events, dt_seconds, padding_mask, dt_valid, dt_reason)
        self._validate_first_event_contract(dt_reason, padding_mask, initial_state)
        batch, length, _ = events.shape
        if initial_state is None:
            state = events.new_zeros((batch, self.hidden_dim))
        else:
            if initial_state.shape != (batch, self.hidden_dim):
                raise ValueError("Liquid V2 initial_state 的 shape 无效")
            if not initial_state.is_floating_point():
                raise TypeError("Liquid V2 initial_state 必须使用浮点 tensor")
            if initial_state.device != events.device or initial_state.dtype != events.dtype:
                raise ValueError("Liquid V2 initial_state 必须匹配 event 的 device 和 dtype")
            if not torch.isfinite(initial_state).all():
                raise ValueError("Liquid V2 initial_state 必须是有限值")
            state = initial_state
        ordinals = self.derive_zero_ordinals(
            dt_reason,
            padding_mask,
            initial_state=initial_state,
            supplied=zero_ordinal,
        )

        states: list[torch.Tensor] = []
        clamped: list[torch.Tensor] = []
        zero_dt: list[torch.Tensor] = []
        invalid: list[torch.Tensor] = []
        first: list[torch.Tensor] = []
        zero_delta: list[torch.Tensor] = []
        traces: dict[str, list[torch.Tensor]] | None = (
            {name: [] for name in TRACE_FIELD_NAMES} if return_trace else None
        )

        for event_index in range(length):
            active = ~padding_mask[:, event_index]
            indices = torch.nonzero(active, as_tuple=False).flatten()
            state_before_event = state
            full_clamped = torch.zeros((batch,), dtype=torch.bool, device=events.device)
            full_zero = torch.zeros((batch,), dtype=torch.bool, device=events.device)
            full_invalid = torch.zeros((batch,), dtype=torch.bool, device=events.device)
            full_first = torch.zeros((batch,), dtype=torch.bool, device=events.device)
            full_zero_delta = torch.zeros((batch,), dtype=torch.bool, device=events.device)
            full_trace: dict[str, torch.Tensor] | None = None
            if traces is not None:
                full_trace = {
                    "raw_state_before_update": state_before_event,
                    "raw_state": state_before_event,
                    "normalized_compute_state": state.new_zeros(state.shape),
                    "candidate": state.new_zeros(state.shape),
                    "gate": state.new_zeros(state.shape),
                    "mixed_memory": state.new_zeros(state.shape),
                    "candidate_innovation": state.new_zeros(state.shape),
                    "mixed_memory_innovation": state.new_zeros(state.shape),
                    "actual_update": state.new_zeros(state.shape),
                    "alpha": state.new_zeros(state.shape),
                    "tau": state.new_zeros(state.shape),
                    "dt_reason": torch.zeros(
                        (batch,), dtype=torch.long, device=events.device
                    ),
                    "zero_ordinal": torch.zeros(
                        (batch,), dtype=torch.long, device=events.device
                    ),
                    "valid_mask": active.clone(),
                }
            if indices.numel() == batch:
                # 常见的无 padding 时间步直接处理完整 batch，避免每步
                # nonzero/index_select/index_copy 带来的 Python 和同步开销。
                step = self.cell.step(
                    events[:, event_index],
                    state,
                    dt_seconds[:, event_index],
                    dt_valid[:, event_index],
                    dt_reason[:, event_index],
                    ordinals[:, event_index],
                )
                state = step.next_state
                full_clamped = step.dt_clamped_mask
                full_zero = step.dt_zero_mask
                full_invalid = step.dt_invalid_mask
                reasons = dt_reason[:, event_index]
                full_first = reasons == DT_REASON_FIRST_EVENT
                full_zero_delta = reasons == DT_REASON_ZERO_DELTA
                if full_trace is not None:
                    for name in (
                        "normalized_compute_state",
                        "candidate",
                        "gate",
                        "mixed_memory",
                        "candidate_innovation",
                        "mixed_memory_innovation",
                        "actual_update",
                        "alpha",
                        "tau",
                    ):
                        full_trace[name] = getattr(step, name)
                    full_trace["dt_reason"] = step.dt_reason
                    full_trace["zero_ordinal"] = ordinals[:, event_index]
                    full_trace["raw_state"] = state
            elif indices.numel():
                step = self.cell.step(
                    events.index_select(0, indices)[:, event_index],
                    state.index_select(0, indices),
                    dt_seconds.index_select(0, indices)[:, event_index],
                    dt_valid.index_select(0, indices)[:, event_index],
                    dt_reason.index_select(0, indices)[:, event_index],
                    ordinals.index_select(0, indices)[:, event_index],
                )
                state = state.index_copy(0, indices, step.next_state)
                full_clamped = full_clamped.index_copy(0, indices, step.dt_clamped_mask)
                full_zero = full_zero.index_copy(0, indices, step.dt_zero_mask)
                full_invalid = full_invalid.index_copy(0, indices, step.dt_invalid_mask)
                reasons = dt_reason.index_select(0, indices)[:, event_index]
                full_first = full_first.index_copy(
                    0, indices, reasons == DT_REASON_FIRST_EVENT
                )
                full_zero_delta = full_zero_delta.index_copy(
                    0, indices, reasons == DT_REASON_ZERO_DELTA
                )
                if full_trace is not None:
                    full_trace["raw_state_before_update"] = full_trace[
                        "raw_state_before_update"
                    ].index_copy(
                        0,
                        indices,
                        state_before_event.index_select(0, indices),
                    )
                    for name in (
                        "normalized_compute_state",
                        "candidate",
                        "gate",
                        "mixed_memory",
                        "candidate_innovation",
                        "mixed_memory_innovation",
                        "actual_update",
                        "alpha",
                        "tau",
                    ):
                        full_trace[name] = full_trace[name].index_copy(
                            0, indices, getattr(step, name)
                        )
                    full_trace["dt_reason"] = full_trace["dt_reason"].index_copy(
                        0, indices, step.dt_reason
                    )
                    full_trace["zero_ordinal"] = full_trace["zero_ordinal"].index_copy(
                        0,
                        indices,
                        ordinals.index_select(0, indices)[:, event_index],
                    )
                    full_trace["raw_state"] = state
            states.append(state)
            clamped.append(full_clamped)
            zero_dt.append(full_zero)
            invalid.append(full_invalid)
            first.append(full_first)
            zero_delta.append(full_zero_delta)
            if full_trace is not None:
                for name, value in full_trace.items():
                    traces[name].append(value)

        sequence = (
            torch.stack(states, dim=1)
            if states
            else events.new_zeros((batch, 0, self.hidden_dim))
        )
        trace: dict[str, torch.Tensor] | None = None
        if traces is not None:
            trace = {}
            for name, values in traces.items():
                trace[name] = torch.stack(values, dim=1)
                if name in {"dt_reason", "zero_ordinal"}:
                    trace[name] = trace[name].to(dtype=torch.long)
                elif name == "valid_mask":
                    trace[name] = trace[name].to(dtype=torch.bool)
        return LiquidV2SequenceOutput(
            sequence=sequence,
            final_state=state,
            dt_clamped_mask=torch.stack(clamped, dim=1)
            if clamped
            else padding_mask.clone(),
            dt_zero_mask=torch.stack(zero_dt, dim=1)
            if zero_dt
            else padding_mask.clone(),
            dt_invalid_mask=torch.stack(invalid, dim=1)
            if invalid
            else padding_mask.clone(),
            dt_first_event_mask=torch.stack(first, dim=1)
            if first
            else padding_mask.clone(),
            dt_zero_delta_mask=torch.stack(zero_delta, dim=1)
            if zero_delta
            else padding_mask.clone(),
            trace=trace,
        )

    def step(
        self,
        event: torch.Tensor,
        dt_seconds: torch.Tensor,
        *,
        dt_valid: torch.Tensor,
        dt_reason: torch.Tensor,
        state: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> LiquidV2StepOutput:
        """使用与离线 forward 相同的 cell 执行 online continuation。"""

        if event.ndim != 2 or event.shape[-1] != self.cell.input_dim:
            raise ValueError("Liquid V2 online event 的 shape 无效")
        if not event.is_floating_point() or not dt_seconds.is_floating_point():
            raise TypeError("Liquid V2 online event 和 dt 必须使用浮点 tensor")
        batch = int(event.shape[0])
        if dt_seconds.shape != (batch,) or dt_valid.shape != (batch,) or dt_reason.shape != (batch,):
            raise ValueError("Liquid V2 online time tensor 的 shape 必须为 [B]")
        if dt_valid.dtype != torch.bool or dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 online validity/reason tensor 的 dtype 无效")
        if any(value.device != event.device for value in (dt_seconds, dt_valid, dt_reason)):
            raise ValueError("Liquid V2 online tensor 必须与 event 使用同一 device")
        fresh = state is None
        if state is None:
            state = event.new_zeros((batch, self.hidden_dim))
        elif state.device != event.device or state.dtype != event.dtype:
            raise ValueError("Liquid V2 online state 必须匹配 event 的 device 和 dtype")
        if fresh and bool((dt_reason != DT_REASON_FIRST_EVENT).any()):
            raise ValueError("新的 online Liquid V2 state 必须使用 first-event reason")
        if not fresh and bool((dt_reason == DT_REASON_FIRST_EVENT).any()):
            raise ValueError("连续 online Liquid V2 state 不能重新开始")
        if zero_ordinal is None:
            if bool((dt_reason == DT_REASON_ZERO_DELTA).any()):
                # ordinal 属于 continuation state contract。仅凭 state 无法
                # 推断它，静默重置会导致 online/offline 的 density/trajectory
                # facts 不一致。
                raise ValueError(
                    "online zero-delta step 必须显式提供 zero_ordinal"
                )
            zero_ordinal = torch.ones(
                (batch,), dtype=torch.long, device=event.device
            )
        elif (
            zero_ordinal.shape != (batch,)
            or zero_ordinal.dtype not in (torch.int64, torch.long)
            or zero_ordinal.device != event.device
            or bool((zero_ordinal < 1).any())
        ):
            raise ValueError(
                "Liquid V2 online zero_ordinal 必须是 event device 上的正 int64"
            )
        return self.cell.step(
            event, state, dt_seconds, dt_valid, dt_reason, zero_ordinal
        )


def liquid_v2_contract(cell: LiquidV2Cell) -> dict[str, Any]:
    """返回写入每个 checkpoint 的结构 contract。"""

    return {
        "equation_version": EQUATION_VERSION,
        "cell_class": "radar_ln3.liquid_v2.LiquidV2Cell",
        "sequence_wrapper_class": "radar_ln3.liquid_v2.LiquidV2SequenceWrapper",
        "input_dim": cell.input_dim,
        "hidden_dim": cell.hidden_dim,
        "memory_state": "raw_r_t",
        "compute_state": "n_t=LayerNorm(r_t)，仅用于临时计算",
        "post_update_normalization": False,
        "candidate_function": {
            "inputs": ["x_t", "n_{t-1}"],
            "candidate": "tanh(candidate_projection([x_t,n_{t-1}]) + input_projection(x_t) + state_projection(n_{t-1}))",
            "gate": "sigmoid(gate_projection([x_t,n_{t-1}]))",
            "v1_capacity_preserved": True,
        },
        "classifier_readout_must_be_independent": True,
        "innovation_names": {
            "candidate": "candidate_innovation=c_t-r_{t-1}",
            "mixed_memory": "mixed_memory_innovation=m_t-r_{t-1}",
            "actual_update": "actual_update=r_t-r_{t-1}",
        },
        "trace_semantics": {
            "raw_state_before_update": "r_{t-1}，当前 active event 更新前；padded 行保持前一状态",
            "raw_state": "r_t，当前 active event 更新后；padded 行保持前一状态",
        },
        "time_policy": time_policy_contract(
            tau_min_seconds=cell.tau_min_seconds,
            tau_max_seconds=cell.tau_max_seconds,
            zero_dt_event_floor_seconds=cell.zero_dt_event_floor_seconds,
            dt_max_seconds=cell.dt_max_seconds,
            invalid_dt_update_mode=cell.invalid_dt_update_mode,
            invalid_dt_observation_gain=cell.invalid_dt_observation_gain,
        ),
        "padding_calls_cell": False,
    }


__all__ = [
    "EQUATION_VERSION",
    "LiquidV2Cell",
    "LiquidV2SequenceOutput",
    "LiquidV2SequenceWrapper",
    "LiquidV2StepOutput",
    "TIME_POLICY_VERSION",
    "TRACE_FIELD_NAMES",
    "liquid_v2_contract",
    "time_policy_contract",
]
