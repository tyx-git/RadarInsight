"""第二版液态算法的序列包装（padding 不调用 cell；离线/在线同构）。

来源：src-v3 @ fd34e46610165441d86469802e9ff461d49bd2aa，src-v3/radar_ln3/liquid_v2.py 逐字拆分。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .cell import (
    LiquidV2Cell,
    LiquidV2StepOutput,
    TRACE_FIELD_NAMES,
)

from .time_policy import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)

@dataclass
class LiquidV2SequenceOutput:
    """A sequence of raw states and optional diagnostic trajectories."""

    sequence: torch.Tensor
    final_state: torch.Tensor
    dt_clamped_mask: torch.Tensor
    dt_zero_mask: torch.Tensor
    dt_invalid_mask: torch.Tensor
    dt_first_event_mask: torch.Tensor
    dt_zero_delta_mask: torch.Tensor
    trace: dict[str, torch.Tensor] | None = None


class LiquidV2SequenceWrapper(nn.Module):
    """Variable-length offline wrapper and online continuation boundary."""

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
            raise ValueError("Liquid V2 events must have shape [B,T,input_dim]")
        if events.shape[0] <= 0 or events.shape[1] <= 0:
            raise ValueError("Liquid V2 sequences must have positive batch and time dimensions")
        expected = events.shape[:2]
        for name, value in (
            ("dt_seconds", dt_seconds),
            ("padding_mask", padding_mask),
            ("dt_valid", dt_valid),
            ("dt_reason", dt_reason),
        ):
            if value.shape != expected:
                raise ValueError(f"Liquid V2 {name} must have shape [B,T]")
        if padding_mask.dtype != torch.bool or dt_valid.dtype != torch.bool:
            raise ValueError("Liquid V2 padding_mask and dt_valid must be boolean")
        if dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 dt_reason must be int64")
        if not events.is_floating_point() or not dt_seconds.is_floating_point():
            raise TypeError("Liquid V2 events and dt must use floating-point tensors")
        if dt_seconds.device != events.device or padding_mask.device != events.device:
            raise ValueError("Liquid V2 sequence tensors must share the event device")
        if dt_valid.device != events.device or dt_reason.device != events.device:
            raise ValueError("Liquid V2 sequence time masks must share the event device")
        if not all(torch.isfinite(value).all() for value in (events, dt_seconds)):
            raise ValueError("Liquid V2 events and dt must be finite")
        padded = padding_mask
        if bool((dt_reason.masked_select(padded) != DT_REASON_PADDING).any()):
            raise ValueError("padded Liquid V2 events must use reason=padding")
        if bool(dt_valid.masked_select(padded).any()):
            raise ValueError("padded Liquid V2 events must have dt_valid=false")
        if bool(dt_seconds.masked_select(padded).abs().gt(1.0e-9).any()):
            raise ValueError("padded Liquid V2 events must have dt=0")
        # The collate contract is prefix-active/suffix-padded.  Allowing a
        # live event after a padded slot would make the same tensor represent
        # two different causal timelines and would break continuation
        # equivalence.
        for row in range(padding_mask.shape[0]):
            padded_indices = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded_indices.numel():
                first_padding = int(padded_indices[0].item())
                if bool((~padding_mask[row, first_padding:]).any()):
                    raise ValueError("Liquid V2 padding must be a suffix of each sequence")
        active = ~padding_mask
        active_reasons = dt_reason.masked_select(active)
        if bool((active_reasons == DT_REASON_PADDING).any()):
            raise ValueError("active Liquid V2 events must not use reason=padding")
        if bool(
            ((dt_reason == DT_REASON_POSITIVE_DELTA) != dt_valid)
            .masked_select(active)
            .any()
        ):
            raise ValueError("active Liquid V2 dt_reason and dt_valid are inconsistent")

    @staticmethod
    def derive_zero_ordinals(
        dt_reason: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        initial_state: torch.Tensor | None = None,
        supplied: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if dt_reason.shape != padding_mask.shape:
            raise ValueError("Liquid V2 ordinal inputs have incompatible shapes")
        result = torch.zeros_like(dt_reason, dtype=torch.long)
        if supplied is not None:
            if supplied.shape != dt_reason.shape or supplied.dtype not in (
                torch.int64,
                torch.long,
            ):
                raise ValueError("Liquid V2 zero_ordinal must be int64 with shape [B,T]")
            if bool((supplied < 0).any()):
                raise ValueError("Liquid V2 zero_ordinal cannot be negative")
            if bool((supplied.masked_select(padding_mask) != 0).any()):
                raise ValueError("padded Liquid V2 zero_ordinal must be zero")
            if supplied.device != dt_reason.device:
                raise ValueError("Liquid V2 zero_ordinal must share the sequence device")

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
                                "positive-delta Liquid V2 events must reset zero_ordinal to one"
                            )
                    elif reason == DT_REASON_ZERO_DELTA:
                        if ordinal < 1:
                            raise ValueError("zero-delta Liquid V2 ordinal must be positive")
                        if previous_reason == DT_REASON_ZERO_DELTA and ordinal != int(previous_ordinal) + 1:
                            raise ValueError(
                                "consecutive zero-delta Liquid V2 ordinals must increase by one"
                            )
                    elif reason == DT_REASON_FIRST_EVENT and ordinal != 1:
                        raise ValueError("first-event Liquid V2 ordinal must be one")
                    result[row, event_index] = ordinal
                    previous_ordinal = ordinal
                else:
                    if initial_state is not None and reason == DT_REASON_ZERO_DELTA:
                        raise ValueError(
                            "continued density-aware Liquid V2 chunks require explicit zero_ordinal"
                        )
                    if reason in (DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA):
                        previous_ordinal = 1
                    elif reason == DT_REASON_ZERO_DELTA:
                        previous_ordinal = int(previous_ordinal or 0) + 1
                    else:
                        raise ValueError("unsupported active Liquid V2 dt reason")
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
        # Batched causal sequences use suffix padding.  An active event after
        # padding would otherwise be silently treated as a later event while
        # the ordinal/time contract still describes a contiguous sequence.
        for row in range(padding_mask.shape[0]):
            padded_indices = torch.nonzero(padding_mask[row], as_tuple=False).flatten()
            if padded_indices.numel():
                first_padding = int(padded_indices[0].item())
                if bool((~padding_mask[row, first_padding:]).any()):
                    raise ValueError("Liquid V2 padding must be a suffix of each sequence")
        for row in range(dt_reason.shape[0]):
            indices = torch.nonzero(active[row], as_tuple=False).flatten()
            if not indices.numel():
                continue
            reasons = dt_reason[row, indices]
            first_count = int((reasons == DT_REASON_FIRST_EVENT).sum().item())
            if initial_state is None:
                if first_count != 1 or int(reasons[0].item()) != DT_REASON_FIRST_EVENT:
                    raise ValueError(
                        "fresh Liquid V2 sequence requires exactly one first-event reason"
                    )
            elif first_count:
                raise ValueError("continued Liquid V2 sequence cannot contain first-event reason")

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
                raise ValueError("Liquid V2 initial_state has an invalid shape")
            if not initial_state.is_floating_point():
                raise TypeError("Liquid V2 initial_state must use a floating-point tensor")
            if initial_state.device != events.device or initial_state.dtype != events.dtype:
                raise ValueError("Liquid V2 initial_state must match event device and dtype")
            if not torch.isfinite(initial_state).all():
                raise ValueError("Liquid V2 initial_state must be finite")
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
            if indices.numel():
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
        """Online continuation through the same cell as offline forward."""

        if event.ndim != 2 or event.shape[-1] != self.cell.input_dim:
            raise ValueError("Liquid V2 online event has an invalid shape")
        if not event.is_floating_point() or not dt_seconds.is_floating_point():
            raise TypeError("Liquid V2 online event and dt must use floating-point tensors")
        batch = int(event.shape[0])
        if dt_seconds.shape != (batch,) or dt_valid.shape != (batch,) or dt_reason.shape != (batch,):
            raise ValueError("Liquid V2 online time tensors must have shape [B]")
        if dt_valid.dtype != torch.bool or dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("Liquid V2 online validity/reason tensors have invalid dtypes")
        if any(value.device != event.device for value in (dt_seconds, dt_valid, dt_reason)):
            raise ValueError("Liquid V2 online tensors must share the event device")
        fresh = state is None
        if state is None:
            state = event.new_zeros((batch, self.hidden_dim))
        elif state.device != event.device or state.dtype != event.dtype:
            raise ValueError("Liquid V2 online state must match event device and dtype")
        if fresh and bool((dt_reason != DT_REASON_FIRST_EVENT).any()):
            raise ValueError("fresh online Liquid V2 state requires first-event reason")
        if not fresh and bool((dt_reason == DT_REASON_FIRST_EVENT).any()):
            raise ValueError("continued online Liquid V2 state cannot restart")
        if zero_ordinal is None:
            if bool((dt_reason == DT_REASON_ZERO_DELTA).any()):
                # The ordinal is part of the continuation state contract.  A
                # wrapper cannot infer it from ``state`` without also carrying
                # the previous time metadata, and silently resetting it would
                # make density/trajectory facts differ between online and
                # offline execution.
                raise ValueError(
                    "online zero-delta steps require explicit zero_ordinal"
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
                "Liquid V2 online zero_ordinal must be positive int64 on the event device"
            )
        return self.cell.step(
            event, state, dt_seconds, dt_valid, dt_reason, zero_ordinal
        )


__all__ = ["LiquidV2SequenceOutput", "LiquidV2SequenceWrapper"]
