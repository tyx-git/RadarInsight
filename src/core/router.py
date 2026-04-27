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
    network_logits: torch.Tensor
    network_weights: torch.Tensor
    coarse_logits: torch.Tensor
    route_indices: torch.Tensor
    override_mask: torch.Tensor


class DeepGatingRouter(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        temperature: float = 1.0,
        coarse_num_classes: int = 9,
        use_gumbel_softmax: bool = True,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.use_gumbel_softmax = use_gumbel_softmax
        self.input_norm = nn.LayerNorm(input_dim)
        self.high_altitude_threshold_m = 5000.0
        self.high_speed_threshold_mps = 100.0
        self.ground_height_threshold_m = 2.0
        self.ground_speed_threshold_mps = 5.0
        self.ground_height_std_threshold_m = 1.0
        self.ground_height_slope_threshold = 0.5
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )
        self.route_head = nn.Linear(hidden_dim // 2, len(ROUTE_NAMES))
        self.coarse_head = nn.Linear(hidden_dim // 2, coarse_num_classes)

    def _apply_rule_override(self, logits: torch.Tensor, physical_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        corrected = logits.clone()
        override_mask = torch.zeros((logits.shape[0],), dtype=torch.bool, device=logits.device)

        height_mean = physical_features[:, 0]
        height_std = physical_features[:, 1]
        height_slope = physical_features[:, 2].abs()
        speed_mean = physical_features[:, 3]
        height_available = torch.ones_like(height_mean, dtype=torch.bool)
        if physical_features.shape[1] > 15:
            height_available = physical_features[:, 15] < 0.5

        high_alt_mask = height_available & (height_mean > self.high_altitude_threshold_m) & (speed_mean > self.high_speed_threshold_mps)
        if high_alt_mask.any():
            corrected[high_alt_mask] = -8.0
            corrected[high_alt_mask, ROUTE_NAMES.index("high_altitude_filter")] = 8.0
            override_mask |= high_alt_mask

        ground_mask = (
            height_available
            & (height_mean < self.ground_height_threshold_m)
            & (speed_mean < self.ground_speed_threshold_mps)
            & (height_std < self.ground_height_std_threshold_m)
            & (height_slope < self.ground_height_slope_threshold)
        )
        if ground_mask.any():
            corrected[ground_mask] = -8.0
            corrected[ground_mask, ROUTE_NAMES.index("ground_target")] = 8.0
            override_mask |= ground_mask

        return corrected, override_mask

    def forward(self, physical_features: torch.Tensor, rule_features: torch.Tensor | None = None) -> RouteDecision:
        hidden = self.backbone(self.input_norm(physical_features))
        network_logits = self.route_head(hidden)
        coarse_logits = self.coarse_head(hidden)
        if self.training and self.use_gumbel_softmax:
            network_weights = torch.nn.functional.gumbel_softmax(
                network_logits,
                tau=self.temperature,
                hard=False,
                dim=-1,
            )
        else:
            network_weights = torch.softmax(network_logits / self.temperature, dim=-1)
        corrected_logits, override_mask = self._apply_rule_override(
            network_logits,
            physical_features if rule_features is None else rule_features,
        )
        if self.training and self.use_gumbel_softmax:
            corrected_weights = torch.nn.functional.gumbel_softmax(
                corrected_logits,
                tau=self.temperature,
                hard=False,
                dim=-1,
            )
        else:
            corrected_weights = torch.softmax(corrected_logits / self.temperature, dim=-1)
        route_indices = corrected_weights.argmax(dim=-1)
        return RouteDecision(
            route_logits=corrected_logits,
            route_weights=corrected_weights,
            network_logits=network_logits,
            network_weights=network_weights,
            coarse_logits=coarse_logits,
            route_indices=route_indices,
            override_mask=override_mask,
        )
