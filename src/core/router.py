from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


ROUTE_NAMES = [
    "low_altitude_suspect",
    "conventional_air",
    "high_altitude_filter",
    "ground_target",
]


@dataclass(frozen=True)
class RouteDecision:
    route_logits: torch.Tensor
    route_weights: torch.Tensor


class DeepGatingRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, len(ROUTE_NAMES)),
        )

    def forward(self, physical_features: torch.Tensor) -> RouteDecision:
        logits = self.network(physical_features)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        return RouteDecision(route_logits=logits, route_weights=weights)
