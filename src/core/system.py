from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .experts import ConventionalAirExpert, GroundTargetExpert, LowAltitudeSuspectExpert, RuleBasedHighAltitudeExpert
from .router import DeepGatingRouter, RouteDecision
from .trajectory import TrackRepresentation, TrackRepresentationBuilder


@dataclass(frozen=True)
class RadarSystemOutput:
    representation: TrackRepresentation
    routing: RouteDecision
    rd_features: torch.Tensor
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
    ) -> None:
        super().__init__()
        rd_dim = 16
        fused_dim = model_dim + 32 + rd_dim
        self.representation_builder = TrackRepresentationBuilder(
            sequence_input_dim=sequence_input_dim,
            physical_input_dim=physical_input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.rd_proj = nn.Sequential(
            nn.Linear(4, rd_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rd_dim, rd_dim),
            nn.LayerNorm(rd_dim),
        )
        self.router = DeepGatingRouter(input_dim=32)
        self.low_altitude_expert = LowAltitudeSuspectExpert(input_dim=fused_dim, dropout=dropout)
        self.conventional_air_expert = ConventionalAirExpert(input_dim=fused_dim, dropout=dropout)
        self.high_altitude_expert = RuleBasedHighAltitudeExpert()
        self.ground_expert = GroundTargetExpert(input_dim=fused_dim, dropout=dropout)

    def forward(
        self,
        sequence_features: torch.Tensor,
        physical_features: torch.Tensor,
        rd_features: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> RadarSystemOutput:
        representation = self.representation_builder(
            sequence_features=sequence_features,
            physical_features=physical_features,
            padding_mask=padding_mask,
        )
        rd_summary = rd_features.mean(dim=1) if rd_features is not None else torch.zeros(
            (sequence_features.shape[0], 4),
            dtype=sequence_features.dtype,
            device=sequence_features.device,
        )
        rd_embedding = self.rd_proj(rd_summary)
        fused_with_rd = torch.cat([representation.fused_features, rd_embedding], dim=-1)
        routing = self.router(representation.physical_features)
        low_altitude = self.low_altitude_expert(fused_with_rd)
        conventional_air = self.conventional_air_expert(fused_with_rd)
        high_altitude = self.high_altitude_expert(
            height_mean=physical_features[:, 0],
            speed_mean=physical_features[:, 3],
        )
        ground = self.ground_expert(fused_with_rd)
        return RadarSystemOutput(
            representation=representation,
            routing=routing,
            rd_features=rd_summary,
            low_altitude_logits=low_altitude.logits,
            conventional_air_logits=conventional_air.logits,
            high_altitude_logits=high_altitude.logits,
            ground_logits=ground.logits,
        )
