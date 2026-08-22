from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
    require_causal_rd_quality_schema,
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


class RDSpatialEncoder(nn.Module):
    """Small spatial front-end for canonical [2, 21, 384] RD frames."""

    def __init__(self, embedding_dim: int, *, aggregation_mode: str = "masked_mean") -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.aggregation_mode = str(aggregation_mode)
        if self.aggregation_mode not in {"masked_mean", "causal_gru"}:
            raise ValueError("RD aggregation_mode must be masked_mean or causal_gru")
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=(1, 4), padding=1),
            nn.GroupNorm(4, 8),
            nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=(2, 4), padding=1),
            nn.GroupNorm(4, 16),
            nn.SiLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 24),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(24, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(),
        )
        self.temporal = nn.GRUCell(self.embedding_dim, self.embedding_dim) if self.aggregation_mode == "causal_gru" else None

    def _encode_frames(
        self, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor
    ) -> torch.Tensor:
        if rd_frames.ndim != 6 or rd_frames.shape[3:] != (2, 21, 384):
            raise ValueError(f"rd_frames must have shape [B,L,K,2,21,384], got {tuple(rd_frames.shape)}")
        if rd_frame_mask.shape != rd_frames.shape[:3]:
            raise ValueError("rd_frame_mask must have shape [B,L,K]")
        if not torch.isfinite(rd_frames).all():
            raise ValueError("rd_frames must be finite")
        batch_size, sequence_length, frame_count = rd_frames.shape[:3]
        flattened = rd_frames.reshape(batch_size * sequence_length * frame_count, 2, 21, 384)
        return self.projection(self.network(flattened)).reshape(
            batch_size, sequence_length, frame_count, -1
        )

    @staticmethod
    def _masked_mean(
        frame_embeddings: torch.Tensor, rd_frame_mask: torch.Tensor
    ) -> torch.Tensor:
        mask = rd_frame_mask.to(dtype=frame_embeddings.dtype).unsqueeze(-1)
        frame_embeddings = frame_embeddings * mask
        denominator = mask.sum(dim=2).clamp_min(1.0)
        return frame_embeddings.sum(dim=2) / denominator

    def forward_with_temporal_residual(
        self, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked-mean evidence plus a causal current-event frame residual.

        The residual is the last active frame embedding minus the first active
        frame embedding within the same TR event.  It never reads a later TR
        event, and it is exactly zero for missing or single-frame events.  The
        frame mask is applied after selection so padded frames cannot provide
        either evidence or gradient.
        """

        if self.aggregation_mode != "masked_mean":
            raise ValueError(
                "V22a temporal residual requires the frozen masked_mean RD path"
            )
        frame_embeddings = self._encode_frames(rd_frames, rd_frame_mask)
        mean_embedding = self._masked_mean(frame_embeddings, rd_frame_mask)
        frame_count = int(frame_embeddings.shape[2])
        active_mask = rd_frame_mask.to(dtype=torch.bool)
        active_count = active_mask.to(dtype=torch.long).sum(dim=2)
        first_index = active_mask.to(dtype=torch.long).argmax(dim=2)
        reverse_index = active_mask.flip(dims=(2,)).to(dtype=torch.long).argmax(dim=2)
        last_index = (frame_count - 1 - reverse_index).to(dtype=torch.long)
        dimension = frame_embeddings.shape[-1]
        first = torch.gather(
            frame_embeddings,
            2,
            first_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, dimension),
        ).squeeze(2)
        last = torch.gather(
            frame_embeddings,
            2,
            last_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, dimension),
        ).squeeze(2)
        has_temporal_pair = (active_count >= 2).unsqueeze(-1)
        residual = (last - first) * has_temporal_pair.to(dtype=frame_embeddings.dtype)
        return mean_embedding, residual

    def forward(self, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor) -> torch.Tensor:
        frame_embeddings = self._encode_frames(rd_frames, rd_frame_mask)
        batch_size, sequence_length, frame_count = frame_embeddings.shape[:3]
        if self.aggregation_mode == "masked_mean":
            return self._masked_mean(frame_embeddings, rd_frame_mask)
        state = frame_embeddings.new_zeros((batch_size, sequence_length, self.embedding_dim))
        if self.temporal is None:
            raise RuntimeError("causal RD aggregation is not initialized")
        for frame_index in range(frame_count):
            candidate = self.temporal(
                frame_embeddings[:, :, frame_index].reshape(batch_size * sequence_length, self.embedding_dim),
                state.reshape(batch_size * sequence_length, self.embedding_dim),
            ).reshape(batch_size, sequence_length, self.embedding_dim)
            active = rd_frame_mask[:, :, frame_index].unsqueeze(-1)
            state = torch.where(active, candidate, state)
        return state


class CfCLiquidCausalEncoder(nn.Module):
    """A stable closed-form liquid-style causal recurrent encoder.

    Each event uses the caller-provided TR wall-clock dt. The update is an
    exponential interpolation toward a learned nonlinear candidate, so no
    future event or sequence-wide attention is used. Invalid or zero dt uses
    the configured floor mode, bounded observation gain, skip diagnostic, or
    density-weighted gain; the explicit validity mask remains available to
    the model and callers in all modes.
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
            "density_weighted",
        }:
            raise ValueError(
                "invalid_dt_update_mode must be floor, bounded_observation, "
                "skip, or density_weighted"
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
            # V20 skip sets same-second gain to zero. V20 density_weighted
            # uses the causal same-second ordinal supplied by forward/online.
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
            # In V20 skip mode alpha=0 removes event information, while this
            # deterministic affine normalization may still transform state.
            next_state = self.output_norm(next_state)
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


class RadarLiquidClassifier(nn.Module):
    def __init__(
        self,
        *,
        tr_value_dim: int = 14,
        tr_missing_dim: int = 11,
        rd_quality_dim: int = 7,
        hidden_dim: int = 64,
        rd_embedding_dim: int = 32,
        liquid_hidden_dim: int = 64,
        num_classes: int = 5,
        tau_min_seconds: float = 0.05,
        tau_max_seconds: float = 30.0,
        zero_dt_event_floor_seconds: float = 1.0e-3,
        dt_max_seconds: float = 120.0,
        dt_valid_dim: int = 0,
        liquid_invalid_dt_update_mode: str = "floor",
        liquid_invalid_dt_observation_gain: float = 0.25,
        liquid_state_normalization: str = "recurrent",
        rd_frame_aggregation: str = "masked_mean",
        rd_quality_schema: str | None = None,
    ) -> None:
        super().__init__()
        self.tr_value_dim = int(tr_value_dim)
        self.tr_missing_dim = int(tr_missing_dim)
        self.rd_quality_dim = int(rd_quality_dim)
        self.num_classes = int(num_classes)
        self.dt_valid_dim = int(dt_valid_dim)
        require_causal_rd_quality_schema(rd_quality_schema)
        self.rd_quality_schema = rd_quality_schema
        if self.dt_valid_dim not in {0, 1}:
            raise ValueError("dt_valid_dim must be 0 or 1")
        self.rd_frame_aggregation = str(rd_frame_aggregation)
        self.liquid_state_normalization = str(liquid_state_normalization)
        if self.rd_frame_aggregation != "masked_mean" and self.liquid_state_normalization != "recurrent":
            raise ValueError("only one liquid/RD temporal ablation may be enabled at a time")
        self.rd_encoder = RDSpatialEncoder(rd_embedding_dim, aggregation_mode=self.rd_frame_aggregation)
        self.tr_encoder = nn.Sequential(
            nn.Linear(tr_value_dim + tr_missing_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        fusion_dim = hidden_dim + rd_embedding_dim + rd_quality_dim + self.dt_valid_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, liquid_hidden_dim),
            nn.LayerNorm(liquid_hidden_dim),
            nn.SiLU(),
        )
        self.liquid = CfCLiquidCausalEncoder(
            liquid_hidden_dim,
            liquid_hidden_dim,
            tau_min_seconds=tau_min_seconds,
            tau_max_seconds=tau_max_seconds,
            zero_dt_event_floor_seconds=zero_dt_event_floor_seconds,
            dt_max_seconds=dt_max_seconds,
            invalid_dt_update_mode=liquid_invalid_dt_update_mode,
            invalid_dt_observation_gain=liquid_invalid_dt_observation_gain,
            state_normalization_mode=self.liquid_state_normalization,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(liquid_hidden_dim),
            nn.Linear(liquid_hidden_dim, num_classes),
        )

    def forward(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        tr_padding_mask: torch.Tensor,
        tr_dt_valid: torch.Tensor | None = None,
        tr_dt_reason: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 3 or tr_missing_mask.shape[:2] != tr_features.shape[:2]:
            raise ValueError("TR tensors must have shape [batch, sequence, features]")
        if tr_features.shape[-1] != self.tr_value_dim or tr_missing_mask.shape[-1] != self.tr_missing_dim:
            raise ValueError("TR feature dimensions do not match model")
        if rd_quality.shape != (*tr_features.shape[:2], self.rd_quality_dim):
            raise ValueError("RD quality shape does not match model")
        if tr_padding_mask.shape != tr_features.shape[:2] or tr_dt_seconds.shape != tr_features.shape[:2]:
            raise ValueError("TR padding/dt shape does not match model")
        if tr_dt_valid is None:
            tr_dt_valid = torch.ones_like(tr_dt_seconds, dtype=torch.bool)
        if tr_dt_valid.shape != tr_features.shape[:2]:
            raise ValueError("TR dt_valid shape does not match model")
        if tr_dt_reason is not None and tr_dt_reason.shape != tr_features.shape[:2]:
            raise ValueError("TR dt_reason shape does not match model")
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        rd_embedding = self.rd_encoder(rd_frames, rd_frame_mask)
        fusion_inputs = [tr_embedding, rd_embedding, rd_quality]
        if self.dt_valid_dim:
            fusion_inputs.append(tr_dt_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
        point_events = self.fusion(torch.cat(fusion_inputs, dim=-1))
        liquid_state = self.liquid(
            point_events,
            tr_dt_seconds,
            tr_padding_mask,
            dt_valid=tr_dt_valid,
            dt_reason=tr_dt_reason,
            zero_ordinal=zero_ordinal,
        )
        logits_sequence = self.classifier(liquid_state.sequence)
        valid = ~tr_padding_mask
        last_indices = valid.long().sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(tr_features.shape[0], device=tr_features.device)
        window_logits = logits_sequence[batch_indices, last_indices]
        return {
            "window_logits": window_logits,
            "sequence_logits": logits_sequence,
            "liquid_sequence": liquid_state.sequence,
            "liquid_final_state": liquid_state.final_state,
            "dt_clamped_mask": liquid_state.dt_clamped_mask,
            "dt_zero_mask": liquid_state.dt_zero_mask,
            "dt_invalid_mask": liquid_state.dt_invalid_mask,
        }

    def forward_online_point(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        liquid_state: torch.Tensor | None = None,
        tr_dt_valid: torch.Tensor | None = None,
        tr_dt_reason: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Consume exactly one causal TR point and return the next state."""

        if tr_features.ndim != 2 or tr_missing_mask.ndim != 2 or rd_quality.ndim != 2:
            raise ValueError("online TR tensors must have shape [batch, features]")
        if tr_dt_seconds.ndim != 1 or tr_dt_seconds.shape[0] != tr_features.shape[0]:
            raise ValueError("online dt must have shape [batch]")
        if tr_dt_valid is None:
            tr_dt_valid = torch.ones_like(tr_dt_seconds, dtype=torch.bool)
        if tr_dt_valid.ndim != 1 or tr_dt_valid.shape[0] != tr_features.shape[0]:
            raise ValueError("online dt_valid must have shape [batch]")
        if tr_dt_reason is not None and tr_dt_reason.shape != tr_dt_seconds.shape:
            raise ValueError("online tr_dt_reason shape does not match model")
        if rd_frames.ndim != 5 or rd_frames.shape[2:] != (2, 21, 384):
            raise ValueError("online RD frames must have shape [batch, K, 2, 21, 384]")
        if rd_frame_mask.shape != rd_frames.shape[:2]:
            raise ValueError("online RD frame mask must have shape [batch, K]")
        if tr_features.shape[0] != rd_frames.shape[0] or tr_features.shape[0] != rd_quality.shape[0]:
            raise ValueError("online modality batch sizes do not match")
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        rd_embedding = self.rd_encoder(rd_frames.unsqueeze(1), rd_frame_mask.unsqueeze(1)).squeeze(1)
        fusion_inputs = [tr_embedding, rd_embedding, rd_quality]
        if self.dt_valid_dim:
            fusion_inputs.append(tr_dt_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
        point_event = self.fusion(torch.cat(fusion_inputs, dim=-1))
        state, dt_clamped, dt_zero = self.liquid.step(
            point_event,
            tr_dt_seconds,
            liquid_state,
            tr_dt_valid,
            tr_dt_reason,
            zero_ordinal,
        )
        logits = self.classifier(self.liquid.readout(state))
        return {
            "window_logits": logits,
            "liquid_state": state,
            "dt_clamped_mask": dt_clamped,
            "dt_zero_mask": dt_zero,
            "dt_invalid_mask": ~tr_dt_valid,
        }


__all__ = ["CfCLiquidCausalEncoder", "LiquidForwardState", "RadarLiquidClassifier", "RDSpatialEncoder"]
