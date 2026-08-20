"""V23 KnownGate classifier using the official ``ncps.torch.CfC`` backbone."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .known_gate import KNOWN_GATE_CLASS_NAMES, KnownGateConfig, KnownGateFusionLinear
from .model import RDSpatialEncoder
from .official_cfc import OfficialCfCCausalEncoder


class OfficialCfCKnownGateClassifier(nn.Module):
    """Keep V21e/V22 feature interfaces while replacing only recurrence in V23."""

    def __init__(self, config: KnownGateConfig) -> None:
        super().__init__()
        self.config = config
        model = config.model
        if model.rd_temporal_residual_dim != 0:
            raise ValueError("V23 official CfC excludes the rejected V22a residual branch")
        self.task_name = "known_gate"
        self.class_names = KNOWN_GATE_CLASS_NAMES
        self.evidence_dim = model.evidence_dim
        self.rd_temporal_residual_dim = 0
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
        self.fusion_input_dim = self.base_fusion_input_dim + self.evidence_dim
        self.fusion = nn.Sequential(
            KnownGateFusionLinear(
                self.fusion_input_dim,
                model.liquid_hidden_dim,
                base_input_dim=self.base_fusion_input_dim,
                zero_input_dim=0,
            ),
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.SiLU(),
        )
        self.liquid = OfficialCfCCausalEncoder(
            model.liquid_hidden_dim,
            model.liquid_hidden_dim,
            dt_max_seconds=model.liquid_dt_max_seconds,
            backbone_units=model.liquid_hidden_dim,
            backbone_layers=1,
            backbone_dropout=0.0,
            mixed_memory=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.Linear(model.liquid_hidden_dim, len(KNOWN_GATE_CLASS_NAMES)),
        )

    def backbone_specification(self) -> dict[str, Any]:
        return self.liquid.specification()

    def _evidence_input(
        self,
        features: torch.Tensor | None,
        valid: torch.Tensor | None,
        *,
        leading_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.evidence_dim == 0:
            if features is not None or valid is not None:
                raise ValueError("V23 baseline-shaped input must not include evidence")
            return None
        expected = (*leading_shape, self.evidence_dim)
        if features is None or valid is None or features.shape != expected or valid.shape != expected:
            raise ValueError("V23 official CfC evidence value/valid shape is invalid")
        if valid.dtype != torch.bool or not torch.isfinite(features).all():
            raise ValueError("V23 official CfC evidence must be finite with boolean validity")
        if bool((features < 0.0).any()) or bool((features > 1.0).any()):
            raise ValueError("V23 official CfC evidence must stay in [0,1]")
        if bool((features.masked_select(~valid) != 0).any()):
            raise ValueError("V23 invalid evidence must be exactly zero")
        return features * valid.to(dtype=dtype)

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
        evidence = self._evidence_input(
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
        if known_gate_feature_valid is not None and bool(
            known_gate_feature_valid.masked_select(tr_padding_mask.unsqueeze(-1)).any()
        ):
            raise ValueError("V23 padded events contain valid course evidence")
        point_events = self._point_events(
            tr_features=tr_features,
            tr_missing_mask=tr_missing_mask,
            rd_frames=rd_frames,
            rd_frame_mask=rd_frame_mask,
            rd_quality=rd_quality,
            known_gate_features=known_gate_features,
            known_gate_feature_valid=known_gate_feature_valid,
            leading_shape=tuple(tr_features.shape[:2]),
        )
        state = self.liquid(
            point_events,
            tr_dt_seconds,
            tr_padding_mask,
            dt_valid=tr_dt_valid,
            dt_reason=tr_dt_reason,
        )
        logits = self.classifier(state.sequence)
        # The official encoder never receives padding, but the classifier has
        # a bias.  Mask its padded outputs explicitly so padding cannot become
        # audit evidence or receive a gradient through a downstream loss.
        logits = logits.masked_fill(tr_padding_mask.unsqueeze(-1), 0.0)
        lengths = (~tr_padding_mask).long().sum(dim=1)
        indices = lengths - 1
        batch_indices = torch.arange(tr_features.shape[0], device=tr_features.device)
        return {
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
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 2 or tr_missing_mask.ndim != 2 or rd_quality.ndim != 2:
            raise ValueError("V23 online TR/RD-quality tensors must be rank 2")
        point_event = self._point_events(
            tr_features=tr_features,
            tr_missing_mask=tr_missing_mask,
            rd_frames=rd_frames,
            rd_frame_mask=rd_frame_mask,
            rd_quality=rd_quality,
            known_gate_features=known_gate_features,
            known_gate_feature_valid=known_gate_feature_valid,
            leading_shape=(tr_features.shape[0],),
        )
        state, dt_clamped, dt_zero = self.liquid.step(
            point_event,
            tr_dt_seconds,
            liquid_state,
            tr_dt_valid,
            tr_dt_reason,
        )
        logits = self.classifier(self.liquid.readout(state))
        return {
            "window_logits": logits,
            "liquid_state": state,
            "dt_clamped_mask": dt_clamped,
            "dt_zero_mask": dt_zero,
            "dt_invalid_mask": ~tr_dt_valid,
            "rd_temporal_residual": logits.new_zeros((*logits.shape[:-1], 0)),
        }


__all__ = ["OfficialCfCKnownGateClassifier"]
