from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrackRepresentation:
    sequence_features: torch.Tensor
    deep_features: torch.Tensor
    physical_features: torch.Tensor
    fused_features: torch.Tensor


class TransformerTrackEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        model_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(model_dim)

    def forward(self, features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.input_proj(features)
        hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
        return self.norm(hidden)


class TrackRepresentationBuilder(nn.Module):
    def __init__(
        self,
        sequence_input_dim: int,
        physical_input_dim: int,
        model_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.sequence_encoder = TransformerTrackEncoder(
            input_dim=sequence_input_dim,
            model_dim=model_dim,
            dropout=dropout,
        )
        self.physical_proj = nn.Sequential(
            nn.Linear(physical_input_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),
        )
        self.fusion = nn.Sequential(
            nn.Linear(model_dim + 32, model_dim + 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(model_dim + 32),
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        physical_features: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> TrackRepresentation:
        encoded = self.sequence_encoder(sequence_features, padding_mask=padding_mask)
        if padding_mask is None:
            deep_features = encoded.mean(dim=1)
        else:
            valid = (~padding_mask).float().unsqueeze(-1)
            denom = valid.sum(dim=1).clamp_min(1.0)
            deep_features = (encoded * valid).sum(dim=1) / denom
        physical_proj = self.physical_proj(physical_features)
        fused = self.fusion(torch.cat([deep_features, physical_proj], dim=-1))
        return TrackRepresentation(
            sequence_features=encoded,
            deep_features=deep_features,
            physical_features=physical_proj,
            fused_features=fused,
        )
