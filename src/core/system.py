from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .experts import (
    ConventionalAirExpert,
    ExpertOutput,
    GroundTargetExpert,
    LowAltitudeFusionOutput,
    LowAltitudeSuspectExpert,
    RuleBasedHighAltitudeExpert,
)
from .router import DeepGatingRouter, RouteDecision
from .trajectory import TrackRepresentation, TrackRepresentationBuilder


@dataclass(frozen=True)
class RadarSystemOutput:
    representation: TrackRepresentation
    routing: RouteDecision
    rd_features: torch.Tensor
    low_altitude_detail: LowAltitudeFusionOutput
    conventional_air_detail: ExpertOutput
    high_altitude_detail: ExpertOutput
    ground_detail: ExpertOutput
    low_altitude_logits: torch.Tensor
    conventional_air_logits: torch.Tensor
    high_altitude_logits: torch.Tensor
    ground_logits: torch.Tensor


class RadarClassificationSystem(nn.Module):
    def __init__(
        self,
        sequence_input_dim: int,
        physical_input_dim: int,
        model_dim: int = 128,
        dropout: float = 0.1,
        low_altitude_cosface_scale: float = 16.0,
        low_altitude_cosface_margin: float = 0.2,
        low_altitude_reject_threshold: float = 0.35,
        low_altitude_rd_activation_threshold: float = 0.8,
        low_altitude_use_rd_patch_cnn: bool = True,
        low_altitude_rd_patch_input_dim: int = 3,
        low_altitude_initial_rd_gate_bias: float = 0.0,
    ) -> None:
        super().__init__()
        rd_dim = 16
        fused_dim = model_dim + 32
        self.representation_builder = TrackRepresentationBuilder(
            sequence_input_dim=sequence_input_dim,
            physical_input_dim=physical_input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.rd_proj = nn.Sequential(
            nn.Linear(6, rd_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rd_dim, rd_dim),
            nn.LayerNorm(rd_dim),
        )
        self.router = DeepGatingRouter(input_dim=physical_input_dim)
        self.low_altitude_expert = LowAltitudeSuspectExpert(
            input_dim=fused_dim,
            rd_input_dim=4,
            rd_patch_input_dim=low_altitude_rd_patch_input_dim,
            rd_context_dim=rd_dim,
            dropout=dropout,
            cosface_scale=low_altitude_cosface_scale,
            cosface_margin=low_altitude_cosface_margin,
            reject_threshold=low_altitude_reject_threshold,
            rd_activation_threshold=low_altitude_rd_activation_threshold,
            use_rd_patch_cnn=low_altitude_use_rd_patch_cnn,
            initial_rd_gate_bias=low_altitude_initial_rd_gate_bias,
        )
        self.conventional_air_expert = ConventionalAirExpert(sequence_dim=model_dim, physical_dim=physical_input_dim, dropout=dropout)
        self.high_altitude_expert = RuleBasedHighAltitudeExpert()
        self.ground_expert = GroundTargetExpert(deep_dim=model_dim, raw_physical_dim=10, dropout=dropout)

    def forward(
        self,
        sequence_features: torch.Tensor,
        physical_features: torch.Tensor,
        rd_features: torch.Tensor | None = None,
        rd_patch_features: torch.Tensor | None = None,
        low_altitude_targets: torch.Tensor | None = None,
        rd_is_multimodal: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        rd_padding_mask: torch.Tensor | None = None,
        rd_patch_padding_mask: torch.Tensor | None = None,
        enable_low_altitude_reject: bool = True,
    ) -> RadarSystemOutput:
        representation = self.representation_builder(
            sequence_features=sequence_features,
            physical_features=physical_features,
            padding_mask=padding_mask,
        )
        if rd_features is not None and rd_features.ndim == 4:
            amp_range = rd_features[..., 0].mean(dim=1)
            amp_dopp = rd_features[..., 2].mean(dim=1)
            rd_summary = torch.stack(
                [
                    amp_range.mean(dim=-1),
                    amp_range.max(dim=-1).values,
                    amp_range.min(dim=-1).values,
                    amp_dopp.mean(dim=-1),
                    amp_dopp.std(dim=-1),
                    rd_is_multimodal.float() if rd_is_multimodal is not None else torch.ones((sequence_features.shape[0],), dtype=sequence_features.dtype, device=sequence_features.device),
                ],
                dim=-1,
            )
        else:
            rd_summary = rd_features.mean(dim=1) if rd_features is not None else torch.zeros(
            (sequence_features.shape[0], 6),
            dtype=sequence_features.dtype,
            device=sequence_features.device,
            )
        rd_embedding = self.rd_proj(rd_summary)
        routing = self.router(
            representation.raw_physical_features,
            rule_features=representation.raw_physical_features,
        )
        low_altitude = self.low_altitude_expert(
            representation.fused_features,
            rd_features=rd_features,
            rd_patch_features=rd_patch_features,
            rd_context=rd_embedding,
            low_altitude_targets=low_altitude_targets,
            rd_is_multimodal=rd_is_multimodal,
            rd_padding_mask=rd_padding_mask,
            rd_patch_padding_mask=rd_patch_padding_mask,
            enable_reject=enable_low_altitude_reject,
            force_rd_activation=False,
        )
        conventional_air = self.conventional_air_expert(
            representation.sequence_features,
            representation.raw_physical_features,
            padding_mask=padding_mask,
        )
        high_altitude = self.high_altitude_expert(
            height_mean=physical_features[:, 0],
            speed_mean=physical_features[:, 3],
        )
        ground = self.ground_expert(representation.deep_features, representation.raw_physical_features)
        return RadarSystemOutput(
            representation=representation,
            routing=routing,
            rd_features=rd_summary,
            low_altitude_detail=low_altitude,
            conventional_air_detail=conventional_air,
            high_altitude_detail=high_altitude,
            ground_detail=ground,
            low_altitude_logits=low_altitude.logits,
            conventional_air_logits=conventional_air.logits,
            high_altitude_logits=high_altitude.logits,
            ground_logits=ground.logits,
        )
