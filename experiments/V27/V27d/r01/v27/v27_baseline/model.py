"""Independent Liquid V2 task models for the complete V27 baseline."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from radar_ln3.data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
    RD_QUALITY_NAMES,
)
from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper, liquid_v2_contract

from .protocol import V27Config, V27_L1_TASK_NAME, V27_L2_TASK_NAME, task_class_names


class V27RDSpatialEncoder(nn.Module):
    """The task-owned masked-mean spatial encoder used by both independent tasks."""

    def __init__(self, embedding_dim: int, *, rd_height: int, rd_width: int) -> None:
        super().__init__()
        self.embedding_dim, self.rd_height, self.rd_width = int(embedding_dim), int(rd_height), int(rd_width)
        self.network = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=(1, 4), padding=1), nn.GroupNorm(4, 8), nn.SiLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=(2, 4), padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1), nn.GroupNorm(4, 24), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.projection = nn.Sequential(nn.Flatten(), nn.Linear(24, self.embedding_dim), nn.LayerNorm(self.embedding_dim), nn.SiLU())

    def forward(self, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor) -> torch.Tensor:
        if rd_frames.ndim != 6 or tuple(rd_frames.shape[3:]) != (2, self.rd_height, self.rd_width) or rd_frame_mask.shape != rd_frames.shape[:3] or rd_frame_mask.dtype != torch.bool:
            raise ValueError("V27 RD frame/mask shape or dtype differs")
        if not rd_frames.is_floating_point() or not torch.isfinite(rd_frames).all():
            raise ValueError("V27 RD frames must be finite floating point")
        batch, length, count = rd_frames.shape[:3]
        output = self.projection(self.network(rd_frames.reshape(batch * length * count, 2, self.rd_height, self.rd_width))).reshape(batch, length, count, self.embedding_dim)
        mask = rd_frame_mask.to(dtype=output.dtype).unsqueeze(-1)
        return (output * mask).sum(dim=2) / mask.sum(dim=2).clamp_min(1.0)


class V27TrajectoryHeads(nn.Module):
    """Separate raw-trajectory knownness and quality heads.

    These heads are deliberately output-only. Their learned outputs never feed
    the recurrent cell or classifier. Both heads expose a score at every
    causal prefix so the quality task can learn maturity from deterministic
    prefix facts instead of manufacturing labels from classification errors.
    """

    def __init__(self, hidden_dim: int, projection_dim: int, task_fact_dim: int, event_fact_dim: int) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.projection_dim = int(projection_dim)
        self.task_fact_dim = int(task_fact_dim)
        self.event_fact_dim = int(event_fact_dim)
        if min(self.hidden_dim, self.projection_dim, self.task_fact_dim, self.event_fact_dim) <= 0:
            raise ValueError("V27 trajectory-head dimensions must be positive")
        # The GRU sees the complete causal event trajectory, not merely a
        # parent-final point: r_t; three distinct innovations; gate, alpha and
        # tau; plus time/RD/course quality facts for the same event.
        self.trajectory_input_dim = self.hidden_dim * 7 + self.event_fact_dim
        self.trajectory_encoder = nn.GRU(self.trajectory_input_dim, self.hidden_dim, batch_first=True)
        # Four H-wide mean/std summaries (raw state + three innovations),
        # three H-wide mean/std summaries (gate/alpha/tau), and the causal
        # GRU state at the same prefix: 15H total.
        trajectory_dim = self.hidden_dim * 15 + self.task_fact_dim
        self.knownness_projection = nn.Sequential(nn.Linear(trajectory_dim, self.projection_dim), nn.LayerNorm(self.projection_dim), nn.SiLU())
        self.knownness_head = nn.Linear(self.projection_dim, 1)
        # Quality must see both accumulated maturity facts and current-event
        # availability. A running mean alone cannot tell whether the current
        # RD/micro-Doppler observation is missing after a long good prefix.
        quality_input_dim = self.task_fact_dim + self.event_fact_dim
        self.quality_head = nn.Sequential(nn.Linear(quality_input_dim, max(8, self.hidden_dim // 2)), nn.SiLU(), nn.Linear(max(8, self.hidden_dim // 2), 1))

    @staticmethod
    def _causal_mean_std(value: torch.Tensor, active: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return prefix-local mean and standard deviation without future use."""

        if value.shape[:2] != active.shape:
            raise ValueError("V27 trajectory value/mask does not align")
        weights = active.to(dtype=value.dtype).unsqueeze(-1)
        count = weights.cumsum(dim=1).clamp_min(1.0)
        total = (value * weights).cumsum(dim=1)
        squared_total = (value.square() * weights).cumsum(dim=1)
        mean = total / count
        variance = (squared_total / count - mean.square()).clamp_min(0.0)
        return mean, variance.sqrt()

    @property
    def summary_feature_dim(self) -> int:
        """Width of the fixed causal summary passed to the trainable head."""

        return self.hidden_dim * 14

    @property
    def event_trajectory_dim(self) -> int:
        """Width of one raw causal event row consumed by the trajectory GRU."""

        return self.trajectory_input_dim

    def build_representation_features(
        self,
        *,
        raw_sequence: torch.Tensor,
        trace: Mapping[str, torch.Tensor],
        padding_mask: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        event_facts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the causal, frozen-backbone inputs for later head fitting.

        The returned tensors contain no trainable trajectory-head operation.
        They can therefore be materialized once from a frozen L1/L2 base and
        replayed while fitting knownness and quality heads.  Every summary is
        prefix-local; padded suffixes are explicitly zeroed.
        """

        active = ~padding_mask
        if raw_sequence.ndim != 3 or raw_sequence.shape[-1] != self.hidden_dim:
            raise ValueError("V27 raw trajectory state has an invalid shape")
        if padding_mask.shape != raw_sequence.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("V27 trajectory padding mask has an invalid shape/dtype")
        if task_facts_sequence.shape != (*raw_sequence.shape[:2], self.task_fact_dim):
            raise ValueError("V27 task-fact sequence has an invalid shape")
        if event_facts.shape != (*raw_sequence.shape[:2], self.event_fact_dim) or not torch.isfinite(event_facts).all():
            raise ValueError("V27 trajectory event facts have an invalid shape/value")
        for name, value in (("event", event_facts), ("task", task_facts_sequence)):
            if not torch.isfinite(value).all() or bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError(f"V27 padded trajectory {name} facts must be finite zero")
        names = ("candidate_innovation", "mixed_memory_innovation", "actual_update", "gate", "alpha", "tau")
        if any(name not in trace for name in names):
            raise ValueError("V27 raw trajectory trace is incomplete")
        for name in names:
            value = trace[name]
            if value.shape != raw_sequence.shape or not torch.isfinite(value).all():
                raise ValueError(f"V27 raw trajectory trace field is invalid: {name}")
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError(f"V27 padded raw trajectory field must be zero: {name}")
        summaries: list[torch.Tensor] = []
        for value in (raw_sequence, trace["candidate_innovation"], trace["mixed_memory_innovation"], trace["actual_update"]):
            mean, std = self._causal_mean_std(value, active)
            summaries.extend((mean, std))
        # Gate/alpha/tau are summaries consumed in addition to, rather than as
        # a replacement for, the causal sequence encoder below.
        gate_mean, gate_std = self._causal_mean_std(trace["gate"], active)
        alpha_mean, alpha_std = self._causal_mean_std(trace["alpha"], active)
        tau_mean, tau_std = self._causal_mean_std(trace["tau"], active)
        event_trajectory = torch.cat(
            (
                raw_sequence,
                trace["candidate_innovation"],
                trace["mixed_memory_innovation"],
                trace["actual_update"],
                trace["gate"],
                trace["alpha"],
                trace["tau"],
                event_facts,
            ),
            dim=-1,
        )
        summary_features = torch.cat(
            (
                *summaries,
                gate_mean,
                gate_std,
                alpha_mean,
                alpha_std,
                tau_mean,
                tau_std,
            ),
            dim=-1,
        )
        if summary_features.shape[-1] != self.summary_feature_dim:
            raise RuntimeError("V27 trajectory summary width differs")
        summary_features = summary_features.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        event_trajectory = event_trajectory.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        return {
            "summary_features": summary_features,
            "event_trajectory": event_trajectory,
            "task_facts_sequence": task_facts_sequence,
            "event_facts_sequence": event_facts,
        }

    def forward_from_cached_features(
        self,
        *,
        summary_features: torch.Tensor,
        event_trajectory: torch.Tensor,
        task_facts_sequence: torch.Tensor,
        event_facts_sequence: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run only trainable trajectory heads over cached causal features."""

        if summary_features.ndim != 3 or summary_features.shape[-1] != self.summary_feature_dim:
            raise ValueError("V27 cached trajectory summary has an invalid shape")
        if event_trajectory.ndim != 3 or event_trajectory.shape[-1] != self.event_trajectory_dim:
            raise ValueError("V27 cached event trajectory has an invalid shape")
        if summary_features.shape[:2] != event_trajectory.shape[:2] or task_facts_sequence.shape != (*event_trajectory.shape[:2], self.task_fact_dim) or event_facts_sequence.shape != (*event_trajectory.shape[:2], self.event_fact_dim) or padding_mask.shape != event_trajectory.shape[:2] or padding_mask.dtype != torch.bool:
            raise ValueError("V27 cached trajectory feature shapes differ")
        if not all(
            torch.isfinite(value).all()
            for value in (
                summary_features,
                event_trajectory,
                task_facts_sequence,
                event_facts_sequence,
            )
        ):
            raise ValueError("V27 cached trajectory features must be finite")
        active = ~padding_mask
        for name, value in (
            ("summary", summary_features),
            ("event", event_trajectory),
            ("task", task_facts_sequence),
            ("event_facts", event_facts_sequence),
        ):
            if bool(value.masked_select(padding_mask.unsqueeze(-1)).ne(0).any()):
                raise ValueError(f"V27 padded cached trajectory {name} must be zero")
        lengths = active.long().sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("V27 trajectory encoder received an empty sequence")
        packed = pack_padded_sequence(event_trajectory, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.trajectory_encoder(packed)
        trajectory_encoder_sequence, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=event_trajectory.shape[1],
        )
        representation = torch.cat(
            (
                summary_features,
                trajectory_encoder_sequence,
                task_facts_sequence,
            ),
            dim=-1,
        )
        projection_sequence = self.knownness_projection(representation)
        knownness_logit_sequence = self.knownness_head(projection_sequence).squeeze(-1)
        quality_logit_sequence = self.quality_head(
            torch.cat((task_facts_sequence, event_facts_sequence), dim=-1)
        ).squeeze(-1)
        projection_sequence = projection_sequence.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        knownness_logit_sequence = knownness_logit_sequence.masked_fill(padding_mask, 0.0)
        quality_logit_sequence = quality_logit_sequence.masked_fill(padding_mask, 0.0)
        trajectory_encoder_sequence = trajectory_encoder_sequence.masked_fill(
            padding_mask.unsqueeze(-1), 0.0
        )
        return {
            "knownness_projection_sequence": projection_sequence,
            "knownness_logit_sequence": knownness_logit_sequence,
            "quality_logit_sequence": quality_logit_sequence,
            "task_facts_sequence": task_facts_sequence,
            "event_facts_sequence": event_facts_sequence,
            "trajectory_encoder_sequence": trajectory_encoder_sequence,
        }

    def forward(self, *, raw_sequence: torch.Tensor, trace: Mapping[str, torch.Tensor], padding_mask: torch.Tensor, task_facts_sequence: torch.Tensor, event_facts: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.build_representation_features(
            raw_sequence=raw_sequence,
            trace=trace,
            padding_mask=padding_mask,
            task_facts_sequence=task_facts_sequence,
            event_facts=event_facts,
        )
        result = self.forward_from_cached_features(
            summary_features=features["summary_features"],
            event_trajectory=features["event_trajectory"],
            task_facts_sequence=features["task_facts_sequence"],
            event_facts_sequence=features["event_facts_sequence"],
            padding_mask=padding_mask,
        )
        result["summary_features"] = features["summary_features"]
        result["event_trajectory"] = features["event_trajectory"]
        result["event_facts_sequence"] = features["event_facts_sequence"]
        return result


class V27LiquidV2TaskModel(nn.Module):
    """One task-local Liquid V2 model; never shared between L1 and L2."""

    def __init__(self, config: V27Config, *, task: str) -> None:
        super().__init__()
        if task not in {V27_L1_TASK_NAME, V27_L2_TASK_NAME}:
            raise ValueError("V27 task model must be L1 or L2")
        self.config, self.task, self.class_names = config, task, task_class_names(task)
        model = config.model
        self.tr_value_dim, self.tr_missing_dim, self.rd_quality_dim = model.tr_value_dim, model.tr_missing_dim, model.rd_quality_dim
        self.rd_height, self.rd_width = model.rd_height, model.rd_width
        self.course_dim = 1 if task == V27_L1_TASK_NAME else 0
        self.physical_dim = 0 if task == V27_L1_TASK_NAME else config.data.l2.rd_physical_dim
        self.micro_dim = 0 if task == V27_L1_TASK_NAME else config.data.l2.micro_doppler_dim
        self.tr_encoder = nn.Sequential(nn.Linear(self.tr_value_dim + self.tr_missing_dim, model.hidden_dim), nn.LayerNorm(model.hidden_dim), nn.SiLU(), nn.Linear(model.hidden_dim, model.hidden_dim), nn.SiLU())
        self.rd_encoder = V27RDSpatialEncoder(model.rd_embedding_dim, rd_height=model.rd_height, rd_width=model.rd_width)
        fusion_dim = model.hidden_dim + model.rd_embedding_dim + model.rd_quality_dim + self.course_dim * 2 + self.physical_dim + self.micro_dim
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, model.liquid_input_dim), nn.LayerNorm(model.liquid_input_dim), nn.SiLU())
        cell = LiquidV2Cell(model.liquid_input_dim, model.liquid_hidden_dim, tau_min_seconds=model.tau_min_seconds, tau_max_seconds=model.tau_max_seconds, zero_dt_event_floor_seconds=model.zero_dt_event_floor_seconds, dt_max_seconds=model.dt_max_seconds, invalid_dt_update_mode=model.invalid_dt_update_mode, invalid_dt_observation_gain=model.invalid_dt_observation_gain)
        self.liquid = LiquidV2SequenceWrapper(cell)
        self.classifier_norm = nn.LayerNorm(model.liquid_hidden_dim)
        self.classifier = nn.Linear(model.liquid_hidden_dim, len(self.class_names))
        # Facts: event count, RD presence, RD frame density, usable dt
        # semantics, zero-delta density, micro validity, physical validity,
        # plus L1 course maturity.
        self.task_fact_dim = 8 if task == V27_L1_TASK_NAME else 7
        # The additional ordinal is an explicit causal density fact.  It is
        # derived by the same Liquid V2 wrapper that performs the update and
        # is never inferred from a future aggregate.
        self.trajectory_event_fact_dim = 19
        self.trajectory_heads = V27TrajectoryHeads(model.liquid_hidden_dim, model.knownness_projection_dim, self.task_fact_dim, self.trajectory_event_fact_dim)
        self._rd_quality_names = tuple(RD_QUALITY_NAMES)

    @staticmethod
    def _require_shape(value: torch.Tensor, shape: tuple[int, ...], name: str) -> None:
        if tuple(value.shape) != shape:
            raise ValueError(f"V27 {name} shape mismatch: expected={shape}, actual={tuple(value.shape)}")

    def model_contract(self) -> dict[str, Any]:
        return {
            "class_path": "src-v3.v27.model.V27LiquidV2TaskModel", "task": self.task,
            "class_names": list(self.class_names), "tr_value_dim": self.tr_value_dim, "tr_missing_dim": self.tr_missing_dim,
            "rd_quality_dim": self.rd_quality_dim, "rd_quality_names": list(self._rd_quality_names), "course_dim": self.course_dim,
            "rd_physical_dim": self.physical_dim, "micro_doppler_dim": self.micro_dim,
            "classifier_readout": "independent_layer_norm_on_raw_state", "knownness_input": "causal_gru_over_raw_trajectory_innovations_gate_alpha_tau_time_quality_facts",
            "knownness_projection_dim": self.trajectory_heads.projection_dim, "trajectory_event_fact_dim": self.trajectory_event_fact_dim,
            "quality_facts": "causal_event_count_rd_dt_and_task_specific_validity_fractions", "quality_output_feedback": "forbidden",
            "liquid_v2_contract": liquid_v2_contract(self.liquid.cell),
        }

    def _validate_common(self, *, tr_features: torch.Tensor, tr_missing_mask: torch.Tensor, tr_dt_seconds: torch.Tensor, tr_dt_valid: torch.Tensor, tr_dt_reason: torch.Tensor, tr_padding_mask: torch.Tensor, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor, rd_quality: torch.Tensor) -> tuple[int, int]:
        if tr_features.ndim != 3:
            raise ValueError("V27 TR features must have [B,T,D]")
        batch, length = tr_features.shape[:2]
        if batch <= 0 or length <= 0:
            raise ValueError("V27 requires non-empty batch and sequence dimensions")
        for value, shape, name in ((tr_features, (batch, length, self.tr_value_dim), "tr_features"), (tr_missing_mask, (batch, length, self.tr_missing_dim), "tr_missing_mask"), (tr_dt_seconds, (batch, length), "tr_dt_seconds"), (tr_dt_valid, (batch, length), "tr_dt_valid"), (tr_dt_reason, (batch, length), "tr_dt_reason"), (tr_padding_mask, (batch, length), "tr_padding_mask"), (rd_quality, (batch, length, self.rd_quality_dim), "rd_quality")):
            self._require_shape(value, shape, name)
        if rd_frames.ndim != 6 or rd_frames.shape[:2] != (batch, length) or tuple(rd_frames.shape[3:]) != (2, self.rd_height, self.rd_width) or rd_frame_mask.shape != rd_frames.shape[:3]:
            raise ValueError("V27 RD input shape differs")
        if tr_dt_valid.dtype != torch.bool or tr_padding_mask.dtype != torch.bool or rd_frame_mask.dtype != torch.bool or tr_dt_reason.dtype not in (torch.int64, torch.long):
            raise ValueError("V27 time/RD mask dtypes differ")
        tensors = (tr_features, tr_missing_mask, tr_dt_seconds, rd_frames, rd_quality)
        if not all(value.device == tr_features.device and value.dtype == tr_features.dtype and value.is_floating_point() and torch.isfinite(value).all() for value in tensors):
            raise ValueError("V27 common inputs must be finite floating tensors on one device")
        if bool(tr_padding_mask.all(dim=1).any()):
            raise ValueError("V27 model received an empty parent sequence")
        if bool(tr_features.masked_select(tr_padding_mask.unsqueeze(-1)).ne(0).any()) or bool(tr_missing_mask.masked_select(tr_padding_mask.unsqueeze(-1)).ne(0).any()) or bool(tr_dt_seconds.masked_select(tr_padding_mask).ne(0).any()) or bool(rd_quality.masked_select(tr_padding_mask.unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27 padded common values must be zero")
        if bool((tr_missing_mask < 0).any()) or bool((tr_missing_mask > 1).any()) or bool((rd_quality < 0).any()) or bool((rd_quality > 1).any()):
            raise ValueError("V27 masks/quality must be bounded")
        if bool(rd_frame_mask.masked_select(tr_padding_mask.unsqueeze(-1)).any()):
            raise ValueError("V27 padded events cannot have RD frames")
        if bool(rd_frames.masked_select((~rd_frame_mask).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).ne(0).any()):
            raise ValueError("V27 invalid RD frames must be zero")
        return batch, length

    def _task_facts_sequence(self, *, padding: torch.Tensor, rd_frame_mask: torch.Tensor, rd_quality: torch.Tensor, dt_valid: torch.Tensor, dt_reason: torch.Tensor, physical_valid: torch.Tensor | None, micro_valid: torch.Tensor | None, course_valid: torch.Tensor | None) -> torch.Tensor:
        """Build cumulative, prefix-local quality facts for every event.

        This is deliberately not a final-track summary repeated at every
        event.  Each row at ``t`` may use only observations through ``t``;
        otherwise a quality/knownness prefix score could leak future evidence
        while appearing to be causal in an offline batch.
        """

        active = ~padding
        dtype = rd_quality.dtype
        weights = active.to(dtype=dtype)
        count = weights.cumsum(dim=1).clamp_min(1.0)

        def causal_fraction(value: torch.Tensor) -> torch.Tensor:
            if value.shape != active.shape:
                raise ValueError("V27 task fact does not align with padding")
            return (value.to(dtype=dtype) * weights).cumsum(dim=1) / count

        # Normalise event maturity against the registered maximum track
        # length, never against a collated batch's longest parent.
        event_fraction = (weights.cumsum(dim=1) / float(self.config.data.max_sequence_length)).clamp(max=1.0)
        facts = [
            event_fraction,
            causal_fraction(rd_frame_mask.any(dim=-1)),
            causal_fraction(rd_quality[..., 1]),
            # A zero-delta timestamp is an observed, usable timing relation
            # under V27's bounded-observation policy.  ``dt_valid`` instead
            # means only "strictly positive delta" in the data schema, and
            # must not turn dense same-second tracks into low-quality samples.
            causal_fraction(
                dt_reason.eq(DT_REASON_FIRST_EVENT)
                | dt_reason.eq(DT_REASON_ZERO_DELTA)
                | dt_reason.eq(DT_REASON_POSITIVE_DELTA)
            ),
            causal_fraction(dt_reason.eq(DT_REASON_ZERO_DELTA)),
            causal_fraction(
                micro_valid if micro_valid is not None else torch.zeros_like(active)
            ),
            causal_fraction(
                physical_valid if physical_valid is not None else torch.zeros_like(active)
            ),
        ]
        if self.task == V27_L1_TASK_NAME:
            if course_valid is None or course_valid.shape != (*active.shape, 1):
                raise ValueError("V27 L1 requires aligned course validity")
            # Use the causal fraction of usable circular-dispersion evidence,
            # rather than normalising a support count against an unrelated
            # maximum track length.  The raw support count remains available
            # to knownness in ``event_facts``.
            facts.append(causal_fraction(course_valid[..., 0]))
        result = torch.stack(facts, dim=-1)
        if result.shape[-1] != self.task_fact_dim:
            raise RuntimeError("V27 task-fact width differs")
        return result.masked_fill(padding.unsqueeze(-1), 0.0)

    def _trajectory_event_facts(self, *, padding: torch.Tensor, rd_frame_mask: torch.Tensor, rd_quality: torch.Tensor, dt_seconds: torch.Tensor, dt_valid: torch.Tensor, dt_reason: torch.Tensor, physical_valid: torch.Tensor | None, micro_valid: torch.Tensor | None, course_features: torch.Tensor | None, course_valid: torch.Tensor | None, course_support: torch.Tensor | None, zero_ordinal: torch.Tensor) -> torch.Tensor:
        """Build per-event causal facts for the read-only knownness branch."""

        dtype = rd_quality.dtype
        zeros = torch.zeros_like(dt_seconds, dtype=dtype)
        physical = zeros if physical_valid is None else physical_valid.to(dtype=dtype)
        micro = zeros if micro_valid is None else micro_valid.to(dtype=dtype)
        if course_features is None:
            course_value = course_is_valid = course_count = zeros
        else:
            assert course_valid is not None and course_support is not None
            course_value = course_features[..., 0] * course_valid[..., 0].to(dtype=dtype)
            course_is_valid = course_valid[..., 0].to(dtype=dtype)
            course_count = (course_support.to(dtype=dtype) / float(max(1, self.config.data.max_sequence_length - 1))).clamp(max=1.0)
        if zero_ordinal.shape != dt_seconds.shape or zero_ordinal.dtype not in (torch.int64, torch.long):
            raise ValueError("V27 zero ordinal does not align with trajectory facts")
        # Log scaling preserves the distinction between the first few
        # same-second observations without allowing an unbounded ordinal to
        # dominate the learned head input.
        ordinal_scale = float(max(1, self.config.data.max_sequence_length))
        ordinal = (
            torch.log1p(zero_ordinal.to(dtype=dtype).clamp_min(0.0))
            / math.log1p(ordinal_scale)
        ).clamp(0.0, 1.0)
        result = torch.cat(
            (
                rd_quality,
                (dt_seconds.clamp(min=0.0, max=self.config.model.dt_max_seconds) / float(self.config.model.dt_max_seconds)).unsqueeze(-1),
                dt_valid.to(dtype=dtype).unsqueeze(-1),
                torch.stack((dt_reason.eq(DT_REASON_FIRST_EVENT).to(dtype=dtype), dt_reason.eq(DT_REASON_ZERO_DELTA).to(dtype=dtype), dt_reason.eq(DT_REASON_POSITIVE_DELTA).to(dtype=dtype)), dim=-1),
                rd_frame_mask.any(dim=-1).to(dtype=dtype).unsqueeze(-1),
                physical.unsqueeze(-1),
                micro.unsqueeze(-1),
                course_value.unsqueeze(-1),
                course_is_valid.unsqueeze(-1),
                course_count.unsqueeze(-1),
                ordinal.unsqueeze(-1),
            ),
            dim=-1,
        )
        if result.shape[-1] != self.trajectory_event_fact_dim:
            raise RuntimeError("V27 trajectory event fact width differs")
        return result.masked_fill(padding.unsqueeze(-1), 0.0)

    def forward(self, *, tr_features: torch.Tensor, tr_missing_mask: torch.Tensor, tr_dt_seconds: torch.Tensor, tr_dt_valid: torch.Tensor, tr_dt_reason: torch.Tensor, tr_padding_mask: torch.Tensor, rd_frames: torch.Tensor, rd_frame_mask: torch.Tensor, rd_quality: torch.Tensor, course_features: torch.Tensor | None = None, course_valid: torch.Tensor | None = None, course_support: torch.Tensor | None = None, rd_physical_features: torch.Tensor | None = None, rd_physical_valid: torch.Tensor | None = None, micro_doppler_features: torch.Tensor | None = None, micro_doppler_valid: torch.Tensor | None = None, zero_ordinal: torch.Tensor | None = None, initial_state: torch.Tensor | None = None, return_trace: bool = False) -> dict[str, Any]:
        batch, length = self._validate_common(tr_features=tr_features, tr_missing_mask=tr_missing_mask, tr_dt_seconds=tr_dt_seconds, tr_dt_valid=tr_dt_valid, tr_dt_reason=tr_dt_reason, tr_padding_mask=tr_padding_mask, rd_frames=rd_frames, rd_frame_mask=rd_frame_mask, rd_quality=rd_quality)
        if self.task == V27_L1_TASK_NAME:
            if course_features is None or course_valid is None or course_support is None or any(value is not None for value in (rd_physical_features, rd_physical_valid, micro_doppler_features, micro_doppler_valid)):
                raise ValueError("V27 L1 input contract differs")
            self._require_shape(course_features, (batch, length, 1), "course_features"); self._require_shape(course_valid, (batch, length, 1), "course_valid"); self._require_shape(course_support, (batch, length), "course_support")
            expected_course_valid = (
                course_support.ge(
                    self.config.data.l1.course_minimum_valid_nonfirst_points
                ).unsqueeze(-1)
                & ~tr_padding_mask.unsqueeze(-1)
            )
            if course_valid.dtype != torch.bool or course_support.dtype not in (torch.int64, torch.long) or not torch.isfinite(course_features).all() or bool((course_features < 0).any()) or bool((course_features > 1).any()) or bool((course_support < 0).any()) or bool(course_features.masked_select(~course_valid).ne(0).any()) or bool(course_valid.masked_select(tr_padding_mask.unsqueeze(-1)).any()) or bool(course_support.masked_select(tr_padding_mask).ne(0).any()) or not torch.equal(course_valid, expected_course_valid):
                raise ValueError("V27 L1 course value/valid/support contract differs")
            optional = [course_features * course_valid.to(dtype=tr_features.dtype), course_valid.to(dtype=tr_features.dtype)]
            physical_valid = micro_valid = None
        else:
            if any(value is not None for value in (course_features, course_valid, course_support)) or any(value is None for value in (rd_physical_features, rd_physical_valid, micro_doppler_features, micro_doppler_valid)):
                raise ValueError("V27 L2 input contract differs")
            self._require_shape(rd_physical_features, (batch, length, self.physical_dim), "rd_physical_features"); self._require_shape(rd_physical_valid, (batch, length), "rd_physical_valid"); self._require_shape(micro_doppler_features, (batch, length, self.micro_dim), "micro_doppler_features"); self._require_shape(micro_doppler_valid, (batch, length), "micro_doppler_valid")
            if rd_physical_valid.dtype != torch.bool or micro_doppler_valid.dtype != torch.bool or not torch.isfinite(rd_physical_features).all() or not torch.isfinite(micro_doppler_features).all() or bool(rd_physical_features.masked_select(~rd_physical_valid.unsqueeze(-1)).ne(0).any()) or bool(micro_doppler_features.masked_select(~micro_doppler_valid.unsqueeze(-1)).ne(0).any()) or bool(rd_physical_valid.masked_select(tr_padding_mask).any()) or bool(micro_doppler_valid.masked_select(tr_padding_mask).any()):
                raise ValueError("V27 L2 physical/micro-Doppler contract differs")
            optional = [rd_physical_features * rd_physical_valid.to(dtype=tr_features.dtype).unsqueeze(-1), micro_doppler_features * micro_doppler_valid.to(dtype=tr_features.dtype).unsqueeze(-1)]
            physical_valid, micro_valid = rd_physical_valid, micro_doppler_valid
        if zero_ordinal is not None and (zero_ordinal.shape != (batch, length) or zero_ordinal.dtype not in (torch.int64, torch.long) or zero_ordinal.device != tr_features.device):
            raise ValueError("V27 zero_ordinal contract differs")
        events = self.fusion(torch.cat([self.tr_encoder(torch.cat((tr_features, tr_missing_mask), dim=-1)), self.rd_encoder(rd_frames, rd_frame_mask), rd_quality, *optional], dim=-1))
        output = self.liquid(events, tr_dt_seconds, tr_padding_mask, dt_valid=tr_dt_valid, dt_reason=tr_dt_reason, initial_state=initial_state, zero_ordinal=zero_ordinal, return_trace=True)
        if output.trace is None:
            raise RuntimeError("V27 task model requires a raw trace")
        normalized = self.classifier_norm(output.sequence)
        sequence_logits = self.classifier(normalized)
        lengths = (~tr_padding_mask).long().sum(dim=1)
        indices = torch.arange(batch, device=tr_features.device)
        task_facts_sequence = self._task_facts_sequence(padding=tr_padding_mask, rd_frame_mask=rd_frame_mask, rd_quality=rd_quality, dt_valid=tr_dt_valid, dt_reason=tr_dt_reason, physical_valid=physical_valid, micro_valid=micro_valid, course_valid=course_valid)
        event_facts = self._trajectory_event_facts(padding=tr_padding_mask, rd_frame_mask=rd_frame_mask, rd_quality=rd_quality, dt_seconds=tr_dt_seconds, dt_valid=tr_dt_valid, dt_reason=tr_dt_reason, physical_valid=physical_valid, micro_valid=micro_valid, course_features=course_features, course_valid=course_valid, course_support=course_support, zero_ordinal=output.trace["zero_ordinal"])
        heads = self.trajectory_heads(raw_sequence=output.sequence, trace=output.trace, padding_mask=tr_padding_mask, task_facts_sequence=task_facts_sequence, event_facts=event_facts)
        final_heads = {
            "knownness_projection": heads["knownness_projection_sequence"][indices, lengths - 1],
            "knownness_logit": heads["knownness_logit_sequence"][indices, lengths - 1],
            "quality_logit": heads["quality_logit_sequence"][indices, lengths - 1],
            "task_facts": heads["task_facts_sequence"][indices, lengths - 1],
            "trajectory_encoder": heads["trajectory_encoder_sequence"][indices, lengths - 1],
        }
        result: dict[str, Any] = {"task": self.task, "window_logits": sequence_logits[indices, lengths - 1], "sequence_logits": sequence_logits, "liquid_sequence": output.sequence, "liquid_final_state": output.final_state, "dt_clamped_mask": output.dt_clamped_mask, "dt_zero_mask": output.dt_zero_mask, "dt_invalid_mask": output.dt_invalid_mask, "dt_first_event_mask": output.dt_first_event_mask, "dt_zero_delta_mask": output.dt_zero_delta_mask, **heads, **final_heads}
        if return_trace:
            result["trace"] = output.trace
        return result






__all__ = ["V27LiquidV2TaskModel", "V27RDSpatialEncoder", "V27TrajectoryHeads"]