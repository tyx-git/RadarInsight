"""V23 KnownGate classifier using the official ``ncps.torch.CfC`` backbone."""

from __future__ import annotations

import hashlib
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KnownGateConfig,
    KnownGateFusionLinear,
    build_known_gate_model_for_config,
    seed_everything,
)
from .model import RDSpatialEncoder
from .official_cfc import OfficialCfCCausalEncoder, OfficialCfCTimePolicy
from .official_cfc_state_adapters import (
    ADAPTER_NONE,
    ExternalOfficialCfCStateAdapter,
    ExternalStateAdapterSpec,
)


@dataclass(frozen=True)
class CausalObservationIntensitySpec:
    """Immutable V23k causal TR/RD observation-intensity encoding.

    The state is deliberately *not* a recurrent-cell state and never changes
    official CfC timespans, hidden-state updates, or external state adapters.
    It is an ordinary two-dimensional causal input to the existing fusion
    layer.  ``kappa`` is fixed from V23a's already frozen 15.025-second
    retention scale, rather than tuned on development performance.
    """

    kappa_per_second: float
    feature_dim: int = 2

    def __post_init__(self) -> None:
        if self.feature_dim != 2:
            raise ValueError("causal observation intensity requires exactly TR/RD features")
        if not math.isfinite(self.kappa_per_second) or self.kappa_per_second <= 0.0:
            raise ValueError("causal observation-intensity kappa must be finite and positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "RADAR-LN3-V23k-causal-observation-intensity-v1",
            "feature_names": [
                "causal_log1p_decayed_tr_observation_intensity",
                "causal_log1p_decayed_rd_observation_intensity",
            ],
            "feature_dim": self.feature_dim,
            "kappa_per_second": self.kappa_per_second,
            "kappa_origin": "1 / V23a fixed_tau_seconds=15.025; fixed before development evaluation",
            "raw_state_formula": {
                "c_TR": "c_TR[t]=exp(-kappa*raw_nonnegative_dt_seconds[t])*c_TR[t-1]+1",
                "c_RD": "c_RD[t]=exp(-kappa*raw_nonnegative_dt_seconds[t])*c_RD[t-1]+sum(current_rd_frame_mask[t])",
            },
            "encoding": "feature[t]=log1p([c_TR[t],c_RD[t]]) without fitted normalization or clipping",
            "causal_scope": "current and earlier dt reasons plus current rd_quality.rd_present only; no labels, future events, parent-final aggregate, identifiers or model outputs",
            "first_and_zero_delta": "raw dt=0, so the previous intensity is retained before the current increment",
            "positive_delta": "uses the same raw positive second-precision duration supplied to the frozen V23a official-CfC path",
            "padding": "does not update intensity state and emits zero feature values",
            "state_effect": "input evidence only; does not alter official CfC timespan, official CfC hidden state equation, or V23a external positive-dt retention adapter",
            "rd_increment": "number of current RD frames visible to the frozen model after its existing max-frame cap; rd_quality.rd_present is checked for exact availability consistency but existing frame-count/duration quality inputs remain unchanged",
        }


def _validate_intensity_reason_inputs(
    *,
    dt_seconds: torch.Tensor,
    dt_valid: torch.Tensor,
    dt_reason: torch.Tensor,
    active: torch.Tensor,
) -> None:
    """Validate the same V3 timing semantics used by the official encoder."""

    from .data import (
        DT_REASON_FIRST_EVENT,
        DT_REASON_PADDING,
        DT_REASON_POSITIVE_DELTA,
        DT_REASON_ZERO_DELTA,
    )

    if dt_seconds.shape != active.shape or dt_valid.shape != active.shape or dt_reason.shape != active.shape:
        raise ValueError("V23k observation-intensity dt tensors must align")
    if (
        active.dtype != torch.bool
        or dt_valid.dtype != torch.bool
        or dt_reason.dtype not in (torch.int64, torch.long)
    ):
        raise ValueError("V23k observation-intensity dt validity/reason dtypes are invalid")
    if not torch.isfinite(dt_seconds).all():
        raise ValueError("V23k observation-intensity received non-finite dt")
    if bool((dt_seconds.masked_select(active) < 0.0).any()):
        raise ValueError("V23k observation-intensity does not accept negative active dt")
    if bool((dt_reason.masked_select(~active) != DT_REASON_PADDING).any()):
        raise ValueError("V23k padded events must use the padding dt reason")
    if bool((dt_reason.masked_select(active) == DT_REASON_PADDING).any()):
        raise ValueError("V23k active events must not use the padding dt reason")
    first = (dt_reason == DT_REASON_FIRST_EVENT) & active
    zero = (dt_reason == DT_REASON_ZERO_DELTA) & active
    positive = (dt_reason == DT_REASON_POSITIVE_DELTA) & active
    if bool((active & ~(first | zero | positive)).any()):
        raise ValueError("V23k observation-intensity received an unsupported active dt reason")
    if bool(dt_seconds.masked_select(first | zero).abs().gt(1.0e-9).any()):
        raise ValueError("V23k first/same-second observation intensity requires raw dt=0")
    if bool(dt_seconds.masked_select(positive).le(0.0).any()):
        raise ValueError("V23k positive observation intensity requires raw dt>0")
    if not torch.equal(positive, dt_valid & active):
        raise ValueError("V23k observation-intensity dt_reason and dt_valid are inconsistent")


def causal_observation_intensity_sequence(
    *,
    spec: CausalObservationIntensitySpec,
    dt_seconds: torch.Tensor,
    dt_valid: torch.Tensor,
    dt_reason: torch.Tensor,
    rd_quality: torch.Tensor,
    rd_frame_mask: torch.Tensor,
    padding_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return causal log1p intensity inputs and the final raw count state.

    ``c_TR`` counts every valid TR observation. ``c_RD`` counts only current
    events with an available RD observation. Both use the same *fixed* wall
    clock decay and are emitted as normal fusion inputs, never as a gain for
    official CfC or an external state adapter.
    """

    if dt_seconds.ndim != 2 or padding_mask.shape != dt_seconds.shape:
        raise ValueError("V23k observation-intensity sequence tensors must have shape [batch,sequence]")
    if rd_quality.shape != (*dt_seconds.shape, 7):
        raise ValueError("V23k observation-intensity requires [batch,sequence,7] RD quality")
    if (
        rd_frame_mask.ndim != 3
        or rd_frame_mask.shape[:2] != dt_seconds.shape
        or rd_frame_mask.dtype != torch.bool
    ):
        raise ValueError("V23k observation-intensity requires boolean RD frame masks [batch,sequence,frames]")
    active = ~padding_mask
    _validate_intensity_reason_inputs(
        dt_seconds=dt_seconds,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        active=active,
    )
    rd_present = rd_quality[..., 0]
    rd_frame_count = rd_frame_mask.to(dtype=dt_seconds.dtype).sum(dim=-1)
    active_present = rd_present.masked_select(active)
    if not torch.isfinite(rd_present).all() or bool(
        (~((active_present == 0.0) | (active_present == 1.0))).any()
    ):
        raise ValueError("V23k rd_quality.rd_present must be finite binary evidence")
    if not torch.equal(
        (rd_frame_count > 0.0).masked_select(active),
        (rd_present > 0.0).masked_select(active),
    ):
        raise ValueError("V23k RD frame availability differs from frozen rd_quality.rd_present")
    state = torch.zeros(
        (*dt_seconds.shape[:1], spec.feature_dim),
        dtype=dt_seconds.dtype,
        device=dt_seconds.device,
    )
    outputs: list[torch.Tensor] = []
    kappa = torch.as_tensor(spec.kappa_per_second, dtype=dt_seconds.dtype, device=dt_seconds.device)
    for event_index in range(dt_seconds.shape[1]):
        current_active = active[:, event_index]
        decay = torch.exp(-kappa * dt_seconds[:, event_index].clamp_min(0.0))
        increment = torch.stack(
            [torch.ones_like(decay), rd_frame_count[:, event_index]], dim=-1
        )
        updated = state * decay.unsqueeze(-1) + increment
        state = torch.where(current_active.unsqueeze(-1), updated, state)
        feature = torch.where(
            current_active.unsqueeze(-1),
            torch.log1p(state),
            torch.zeros_like(state),
        )
        outputs.append(feature)
    return torch.stack(outputs, dim=1), state


def causal_observation_intensity_step(
    *,
    spec: CausalObservationIntensitySpec,
    dt_seconds: torch.Tensor,
    dt_valid: torch.Tensor,
    dt_reason: torch.Tensor,
    rd_quality: torch.Tensor,
    rd_frame_mask: torch.Tensor,
    previous_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one online V23k intensity state without touching CfC state."""

    if dt_seconds.ndim != 1 or dt_valid.shape != dt_seconds.shape or dt_reason.shape != dt_seconds.shape:
        raise ValueError("V23k online observation-intensity dt tensors must have shape [batch]")
    if rd_quality.shape != (dt_seconds.shape[0], 7):
        raise ValueError("V23k online observation-intensity RD quality must have shape [batch,7]")
    if (
        rd_frame_mask.ndim != 2
        or rd_frame_mask.shape[0] != dt_seconds.shape[0]
        or rd_frame_mask.dtype != torch.bool
    ):
        raise ValueError("V23k online observation-intensity RD frame mask must have shape [batch,frames]")
    active = torch.ones_like(dt_seconds, dtype=torch.bool)
    _validate_intensity_reason_inputs(
        dt_seconds=dt_seconds,
        dt_valid=dt_valid,
        dt_reason=dt_reason,
        active=active,
    )
    from .data import DT_REASON_FIRST_EVENT

    if previous_state is None:
        if not bool((dt_reason == DT_REASON_FIRST_EVENT).all()):
            raise ValueError(
                "V23k online observation intensity requires first-event reason without prior state"
            )
        state = torch.zeros(
            (dt_seconds.shape[0], spec.feature_dim),
            dtype=dt_seconds.dtype,
            device=dt_seconds.device,
        )
    else:
        if previous_state.shape != (dt_seconds.shape[0], spec.feature_dim) or not torch.isfinite(previous_state).all() or bool((previous_state < 0.0).any()):
            raise ValueError("V23k online observation-intensity state is invalid")
        state = previous_state
    rd_present = rd_quality[:, 0]
    if not torch.isfinite(rd_present).all() or bool((~((rd_present == 0.0) | (rd_present == 1.0))).any()):
        raise ValueError("V23k online rd_quality.rd_present must be finite binary evidence")
    rd_frame_count = rd_frame_mask.to(dtype=dt_seconds.dtype).sum(dim=-1)
    if not torch.equal(rd_frame_count > 0.0, rd_present > 0.0):
        raise ValueError("V23k online RD frame availability differs from frozen rd_quality.rd_present")
    kappa = torch.as_tensor(spec.kappa_per_second, dtype=dt_seconds.dtype, device=dt_seconds.device)
    decay = torch.exp(-kappa * dt_seconds.clamp_min(0.0))
    increment = torch.stack([torch.ones_like(decay), rd_frame_count], dim=-1)
    state = state * decay.unsqueeze(-1) + increment
    return torch.log1p(state), state


class OfficialCfCKnownGateClassifier(nn.Module):
    """V23's pure official-CfC KnownGate baseline.

    V23 deliberately reuses V21's ordinary TR/RD front end and output heads,
    but replaces the complete project-local recurrent core with dense official
    ``ncps.torch.CfC``.  A registered experiment may retain the frozen
    KnownGate evidence-cache columns and/or append V23k's two causal
    observation-intensity columns.  The input regions are ordered as
    ``base, observation_intensity, legacy_evidence`` so both V23k (base plus
    two new zero-initialized columns) and V23l (base plus the already-frozen
    V21e course column) preserve their respective source fusion layouts.
    """

    def __init__(
        self,
        config: KnownGateConfig,
        *,
        state_adapter_spec: ExternalStateAdapterSpec | None = None,
        time_policy: OfficialCfCTimePolicy | None = None,
        observation_intensity_spec: CausalObservationIntensitySpec | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        model = config.model
        if model.rd_temporal_residual_dim != 0:
            raise ValueError("V23 official CfC excludes the rejected V22a residual branch")
        if model.evidence_dim < 0:
            raise ValueError("V23 official CfC legacy evidence dimension is invalid")
        self.task_name = "known_gate"
        self.class_names = KNOWN_GATE_CLASS_NAMES
        self.evidence_dim = int(model.evidence_dim)
        self.rd_temporal_residual_dim = 0
        self.observation_intensity_spec = observation_intensity_spec
        self.observation_intensity_dim = (
            0 if observation_intensity_spec is None else observation_intensity_spec.feature_dim
        )
        self.tr_encoder = nn.Sequential(
            nn.Linear(model.tr_value_dim + model.tr_missing_dim, model.hidden_dim),
            nn.LayerNorm(model.hidden_dim),
            nn.SiLU(),
            nn.Linear(model.hidden_dim, model.hidden_dim),
            nn.SiLU(),
        )
        self.rd_encoder = RDSpatialEncoder(
            model.rd_embedding_dim, aggregation_mode=model.rd_frame_aggregation
        )
        self.base_fusion_input_dim = model.hidden_dim + model.rd_embedding_dim + model.rd_quality_dim
        self.fusion_input_dim = (
            self.base_fusion_input_dim
            + self.observation_intensity_dim
            + self.evidence_dim
        )
        self.fusion = nn.Sequential(
            KnownGateFusionLinear(
                self.fusion_input_dim,
                model.liquid_hidden_dim,
                base_input_dim=self.base_fusion_input_dim,
                zero_input_dim=self.observation_intensity_dim,
            ),
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.SiLU(),
        )
        self.liquid = OfficialCfCCausalEncoder(
            model.liquid_hidden_dim,
            model.liquid_hidden_dim,
            backbone_units=model.liquid_hidden_dim,
            backbone_layers=1,
            backbone_dropout=0.0,
            mixed_memory=False,
            time_policy=time_policy,
        )
        self.state_adapter_spec = state_adapter_spec or ExternalStateAdapterSpec(
            kind=ADAPTER_NONE,
            hidden_dim=model.liquid_hidden_dim,
        )
        if self.state_adapter_spec.hidden_dim != model.liquid_hidden_dim:
            raise ValueError("V23 official-CfC state adapter hidden dimension differs from model")
        self.state_adapter = ExternalOfficialCfCStateAdapter(self.state_adapter_spec)
        self.classifier = nn.Sequential(
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.Linear(model.liquid_hidden_dim, len(KNOWN_GATE_CLASS_NAMES)),
        )

    def backbone_specification(self) -> dict[str, Any]:
        return self.liquid.specification()

    def external_state_adapter_specification(self) -> dict[str, Any]:
        return self.state_adapter.specification()

    def observation_intensity_specification(self) -> dict[str, Any] | None:
        return (
            None
            if self.observation_intensity_spec is None
            else self.observation_intensity_spec.as_dict()
        )

    def _legacy_evidence_input(
        self,
        features: torch.Tensor | None,
        valid: torch.Tensor | None,
        *,
        leading_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.evidence_dim == 0:
            if features is not None or valid is not None:
                raise ValueError("V23 baseline-shaped input must not include legacy evidence")
            return None
        expected = (*leading_shape, self.evidence_dim)
        if features is None or valid is None or features.shape != expected or valid.shape != expected:
            raise ValueError("V23 official CfC legacy evidence value/valid shape is invalid")
        if valid.dtype != torch.bool or not torch.isfinite(features).all():
            raise ValueError("V23 official CfC legacy evidence must be finite with boolean validity")
        if bool((features < 0.0).any()) or bool((features > 1.0).any()):
            raise ValueError("V23 official CfC legacy evidence must be bounded to [0,1]")
        if bool((features.masked_select(~valid) != 0).any()):
            raise ValueError("V23 invalid legacy evidence must be exactly zero")
        return features * valid.to(dtype=dtype)

    def _observation_intensity_input(
        self,
        features: torch.Tensor | None,
        *,
        leading_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.observation_intensity_dim == 0:
            if features is not None:
                raise ValueError("V23 baseline-shaped input must not include observation intensity")
            return None
        expected = (*leading_shape, self.observation_intensity_dim)
        if features is None or features.shape != expected:
            raise ValueError("V23 official CfC observation-intensity shape is invalid")
        if not torch.isfinite(features).all() or bool((features < 0.0).any()):
            raise ValueError("V23 official CfC observation intensity must be finite and non-negative")
        return features.to(dtype=dtype)

    def _point_events(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        known_gate_features: torch.Tensor | None,
        known_gate_feature_valid: torch.Tensor | None,
        observation_intensity_features: torch.Tensor | None = None,
        leading_shape: tuple[int, ...],
    ) -> torch.Tensor:
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        if tr_features.ndim == 3:
            rd_embedding = self.rd_encoder(rd_frames, rd_frame_mask)
        elif tr_features.ndim == 2:
            rd_embedding = self.rd_encoder(
                rd_frames.unsqueeze(1), rd_frame_mask.unsqueeze(1)
            ).squeeze(1)
        else:
            raise ValueError("V23 official CfC TR features must be rank 2 or 3")
        inputs = [tr_embedding, rd_embedding, rd_quality]
        observation_intensity = self._observation_intensity_input(
            observation_intensity_features,
            leading_shape=leading_shape,
            dtype=tr_features.dtype,
        )
        if observation_intensity is not None:
            inputs.append(observation_intensity)
        evidence = self._legacy_evidence_input(
            known_gate_features,
            known_gate_feature_valid,
            leading_shape=leading_shape,
            dtype=tr_features.dtype,
        )
        if evidence is not None:
            inputs.append(evidence)
        return self.fusion(torch.cat(inputs, dim=-1))

    def forward(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        tr_padding_mask: torch.Tensor,
        known_gate_features: torch.Tensor | None = None,
        known_gate_feature_valid: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model = self.config.model
        if tr_features.ndim != 3 or tr_features.shape[-1] != model.tr_value_dim:
            raise ValueError("V23 TR features must have shape [batch,sequence,14]")
        if tr_missing_mask.shape != (*tr_features.shape[:2], model.tr_missing_dim):
            raise ValueError("V23 TR missing-mask shape is invalid")
        if (
            tr_dt_seconds.shape != tr_features.shape[:2]
            or tr_dt_valid.shape != tr_features.shape[:2]
            or tr_dt_reason.shape != tr_features.shape[:2]
            or tr_padding_mask.shape != tr_features.shape[:2]
        ):
            raise ValueError("V23 dt/padding tensors must align with the sequence")
        if rd_quality.shape != (*tr_features.shape[:2], model.rd_quality_dim):
            raise ValueError("V23 RD-quality shape is invalid")
        if not all(
            torch.isfinite(value).all()
            for value in (tr_features, tr_missing_mask, tr_dt_seconds, rd_frames, rd_quality)
        ):
            raise ValueError("V23 official CfC model received a non-finite base input")
        observation_intensity_features: torch.Tensor | None = None
        if known_gate_feature_valid is not None and bool(
            known_gate_feature_valid.masked_select(tr_padding_mask.unsqueeze(-1)).any()
        ):
            raise ValueError("V23 padded events contain valid legacy evidence")
        if self.observation_intensity_spec is None:
            observation_intensity_features = None
        else:
            expected_features, _final_intensity = causal_observation_intensity_sequence(
                spec=self.observation_intensity_spec,
                dt_seconds=tr_dt_seconds,
                dt_valid=tr_dt_valid,
                dt_reason=tr_dt_reason,
                rd_quality=rd_quality,
                rd_frame_mask=rd_frame_mask,
                padding_mask=tr_padding_mask,
            )
            observation_intensity_features = expected_features
        point_events = self._point_events(
            tr_features=tr_features,
            tr_missing_mask=tr_missing_mask,
            rd_frames=rd_frames,
            rd_frame_mask=rd_frame_mask,
            rd_quality=rd_quality,
            known_gate_features=known_gate_features,
            known_gate_feature_valid=known_gate_feature_valid,
            observation_intensity_features=observation_intensity_features,
            leading_shape=tuple(tr_features.shape[:2]),
        )
        state = self.liquid(
            point_events,
            tr_dt_seconds,
            tr_padding_mask,
            dt_valid=tr_dt_valid,
            dt_reason=tr_dt_reason,
            # Preserve frozen V23 exactly when adaptation is disabled: the
            # original parent-wise official-CfC path remains the baseline.
            # V23a/V23b/V23c alone opt into the explicit per-event external
            # adapter path below.
            state_adapter=(
                None
                if self.state_adapter_spec.kind == ADAPTER_NONE
                else self.state_adapter
            ),
            zero_ordinal=zero_ordinal,
        )
        logits = self.classifier(state.sequence)
        # Padding never reaches the official recurrence.  Keep the readout's
        # padded bias values untouched so V23 has the same observable tensor
        # contract as the frozen V21e/V22 controls; loss and evaluation use
        # only true causal lengths and therefore never supervise/read them.
        lengths = (~tr_padding_mask).long().sum(dim=1)
        indices = lengths - 1
        batch_indices = torch.arange(tr_features.shape[0], device=tr_features.device)
        result = {
            "sequence_logits": logits,
            "window_logits": logits[batch_indices, indices],
            "liquid_sequence": state.sequence,
            "liquid_final_state": state.final_state,
            "dt_clamped_mask": state.dt_clamped_mask,
            "dt_zero_mask": state.dt_zero_mask,
            "dt_invalid_mask": state.dt_invalid_mask,
            "dt_first_event_mask": state.dt_first_event_mask,
            "dt_zero_delta_mask": state.dt_zero_delta_mask,
            "rd_temporal_residual": logits.new_zeros((*logits.shape[:-1], 0)),
        }
        if self.observation_intensity_spec is not None:
            result["observation_intensity_final_state"] = _final_intensity
        return result

    def forward_online_point(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        liquid_state: torch.Tensor | None = None,
        known_gate_features: torch.Tensor | None = None,
        known_gate_feature_valid: torch.Tensor | None = None,
        zero_ordinal: torch.Tensor | None = None,
        observation_intensity_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 2 or tr_missing_mask.ndim != 2 or rd_quality.ndim != 2:
            raise ValueError("V23 online TR/RD-quality tensors must be rank 2")
        observation_intensity_features: torch.Tensor | None = None
        if self.observation_intensity_spec is None:
            if observation_intensity_state is not None:
                raise ValueError("V23 without intensity spec must not receive observation intensity state")
            next_observation_intensity_state = None
        else:
            from .data import DT_REASON_FIRST_EVENT

            if tr_dt_valid is None or tr_dt_reason is None:
                raise ValueError(
                    "V23k online observation intensity requires explicit dt_valid and dt_reason"
                )
            if (liquid_state is None) != (observation_intensity_state is None):
                raise ValueError(
                    "V23k online liquid and observation-intensity state must reset together"
                )
            if observation_intensity_state is None and not bool(
                (tr_dt_reason == DT_REASON_FIRST_EVENT).all()
            ):
                raise ValueError(
                    "V23k online observation intensity requires prior state after the first event"
                )
            expected_features, next_observation_intensity_state = causal_observation_intensity_step(
                spec=self.observation_intensity_spec,
                dt_seconds=tr_dt_seconds,
                dt_valid=tr_dt_valid,
                dt_reason=tr_dt_reason,
                rd_quality=rd_quality,
                rd_frame_mask=rd_frame_mask,
                previous_state=observation_intensity_state,
            )
            observation_intensity_features = expected_features
        point_event = self._point_events(
            tr_features=tr_features,
            tr_missing_mask=tr_missing_mask,
            rd_frames=rd_frames,
            rd_frame_mask=rd_frame_mask,
            rd_quality=rd_quality,
            known_gate_features=known_gate_features,
            known_gate_feature_valid=known_gate_feature_valid,
            observation_intensity_features=observation_intensity_features,
            leading_shape=(tr_features.shape[0],),
        )
        state, dt_clamped, dt_zero = self.liquid.step(
            point_event,
            tr_dt_seconds,
            liquid_state,
            tr_dt_valid,
            tr_dt_reason,
            (
                None
                if self.state_adapter_spec.kind == ADAPTER_NONE
                else self.state_adapter
            ),
            zero_ordinal,
        )
        logits = self.classifier(self.liquid.readout(state))
        result = {
            "window_logits": logits,
            "liquid_state": state,
            "dt_clamped_mask": dt_clamped,
            "dt_zero_mask": dt_zero,
            "dt_invalid_mask": ~tr_dt_valid,
            "rd_temporal_residual": logits.new_zeros((*logits.shape[:-1], 0)),
        }
        if self.observation_intensity_spec is not None:
            result["observation_intensity_state"] = next_observation_intensity_state
        return result


def _state_subset_sha256(
    state: dict[str, torch.Tensor], *, names: set[str]
) -> str:
    """Hash a named state subset without importing a training entry point."""

    digest = hashlib.sha256()
    for name in sorted(names):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash complete initialization state with the same stable tensor ordering."""

    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_official_cfc_model_from_frozen_source(
    project_root: Path,
    source_config: KnownGateConfig,
    *,
    state_adapter_spec: ExternalStateAdapterSpec | None = None,
    time_policy: OfficialCfCTimePolicy | None = None,
    observation_intensity_spec: CausalObservationIntensitySpec | None = None,
) -> tuple[OfficialCfCKnownGateClassifier, dict[str, Any]]:
    """Construct V23 with every non-recurrent tensor copied from its Control.

    The recurrent state equations cannot share weights across the historical
    project-local encoder and the official CfC.  All compatible TR/RD front
    end, fusion and classifier parameters/buffers *can* and must be copied
    exactly.  This prevents differences in random-number consumption during
    CfC construction from becoming an unregistered second experiment factor.
    """

    source_model, source_reference, source_pairing = build_known_gate_model_for_config(
        project_root, source_config
    )
    del source_reference
    source_state = dict(source_model.state_dict())
    source_initial_hash = _state_dict_sha256(source_state)
    source_parameter_names = {name for name, _value in source_model.named_parameters()}
    source_buffer_names = {name for name, _value in source_model.named_buffers()}

    # Deliberately give the newly introduced official recurrent module a
    # reproducible seed of its own, then overwrite every compatible tensor.
    # The training entry point resets all RNGs again before optimization.
    seed_everything(source_config.train.seed)
    if observation_intensity_spec is None:
        official_model = OfficialCfCKnownGateClassifier(
            source_config,
            state_adapter_spec=state_adapter_spec,
            time_policy=time_policy,
        )
    else:
        # Construct the un-widened V23a shape first under the same post-seed
        # RNG state as V23 itself.  Fusion widening otherwise shifts random
        # consumption before official CfC construction and would silently add
        # a second initialization factor.  The wide model's non-recurrent
        # tensors are copied below and its liquid state is replaced here.
        reference_official = OfficialCfCKnownGateClassifier(
            source_config,
            state_adapter_spec=state_adapter_spec,
            time_policy=time_policy,
        )
        official_model = OfficialCfCKnownGateClassifier(
            source_config,
            state_adapter_spec=state_adapter_spec,
            time_policy=time_policy,
            observation_intensity_spec=observation_intensity_spec,
        )
        with torch.no_grad():
            official_model.liquid.load_state_dict(
                reference_official.liquid.state_dict(), strict=True
            )
        del reference_official
        # Restore the post-construction RNG state to the exact V23a path.
        # The widened Fusion constructor consumed two extra columns' random
        # draws, but those values are discarded and must not perturb a later
        # paired build in the same process.
        seed_everything(source_config.train.seed)
        _rng_alignment = OfficialCfCKnownGateClassifier(
            source_config,
            state_adapter_spec=state_adapter_spec,
            time_policy=time_policy,
        )
        del _rng_alignment
    official_state = official_model.state_dict()
    source_names = {name for name in source_state if not name.startswith("liquid.")}
    official_names = {
        name
        for name in official_state
        if not name.startswith("liquid.") and not name.startswith("state_adapter.")
    }
    if observation_intensity_spec is not None:
        # V23k adds precisely two zero-initialized Fusion input columns.  Every
        # historical non-recurrent tensor is copied bit-for-bit; the new columns
        # are structurally visible but contribute zero at initialization.
        source_names = {name for name in source_names if name != "fusion.0.weight"}
        official_names = {name for name in official_names if name != "fusion.0.weight"}
    if source_names != official_names:
        missing = sorted(source_names - official_names)
        unexpected = sorted(official_names - source_names)
        raise ValueError(
            "V23 non-recurrent source/official state namespaces differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    with torch.no_grad():
        for name in sorted(source_names):
            source_value = source_state[name]
            official_value = official_state[name]
            if source_value.shape != official_value.shape or source_value.dtype != official_value.dtype:
                raise ValueError(f"V23 non-recurrent source/official state differs at {name}")
            official_value.copy_(source_value)
        if observation_intensity_spec is not None:
            source_value = source_state["fusion.0.weight"]
            official_value = official_state["fusion.0.weight"]
            if (
                official_value.shape[0] != source_value.shape[0]
                or official_value.shape[1]
                != source_value.shape[1] + observation_intensity_spec.feature_dim
            ):
                raise ValueError("V23k Fusion width does not match the registered two input columns")
            official_value[:, : source_value.shape[1]].copy_(source_value)
            official_value[:, source_value.shape[1] :].zero_()
    copied_state = official_model.state_dict()
    if any(not torch.equal(source_state[name], copied_state[name]) for name in source_names):
        raise AssertionError("V23 non-recurrent initialization copy is not tensor-exact")
    if observation_intensity_spec is not None and not torch.equal(
        copied_state["fusion.0.weight"][:, : source_state["fusion.0.weight"].shape[1]],
        source_state["fusion.0.weight"],
    ):
        raise AssertionError("V23k shared Fusion columns are not tensor-exact")
    parameter_names = {name for name, _value in official_model.named_parameters()}
    buffer_names = {name for name, _value in official_model.named_buffers()}
    audit = {
        "strategy": "copy_frozen_source_non_recurrent_state_then_seed_official_cfc",
        "source_initial_model_state_sha256": source_initial_hash,
        "source_initialization_pairing": dict(source_pairing),
        "shared_non_recurrent_state_sha256": _state_subset_sha256(
            copied_state,
            names=source_names | ({"fusion.0.weight"} if observation_intensity_spec is None else set()),
        ),
        "shared_non_recurrent_tensor_count": len(source_names) + int(observation_intensity_spec is None),
        "shared_non_recurrent_parameter_count": len(source_names & parameter_names) + int(
            observation_intensity_spec is None and "fusion.0.weight" in parameter_names
        ),
        "shared_non_recurrent_buffer_count": len(source_names & buffer_names),
        "source_non_recurrent_parameter_count": len(source_names & source_parameter_names),
        "source_non_recurrent_buffer_count": len(source_names & source_buffer_names),
        "shared_non_recurrent_state_exactly_equal": True,
        "official_recurrent_state_is_distinct_by_architecture": True,
        "external_state_adapter": official_model.external_state_adapter_specification(),
        "observation_intensity": official_model.observation_intensity_specification(),
    }
    # Keep the complete source model out of the training graph and CUDA memory.
    del source_model
    return official_model, audit


__all__ = [
    "CausalObservationIntensitySpec",
    "OfficialCfCKnownGateClassifier",
    "build_official_cfc_model_from_frozen_source",
    "causal_observation_intensity_sequence",
    "causal_observation_intensity_step",
]
