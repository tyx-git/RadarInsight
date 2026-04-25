from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


LOW_ALTITUDE_CLASSES = ["DroneTarget", "BirdTarget", "BalloonTarget", "UnknownTarget"]
CONVENTIONAL_AIR_CLASSES = ["AircraftTarget"]
GROUND_TARGET_CLASSES = ["PeopleTarget", "CarTarget", "ShipTarget"]


@dataclass(frozen=True)
class ExpertOutput:
    logits: torch.Tensor


class MLPExpert(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, fused_features: torch.Tensor) -> ExpertOutput:
        return ExpertOutput(logits=self.network(fused_features))


class LowAltitudeSuspectExpert(MLPExpert):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__(input_dim=input_dim, num_classes=len(LOW_ALTITUDE_CLASSES), hidden_dim=hidden_dim, dropout=dropout)


class ConventionalAirExpert(MLPExpert):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__(input_dim=input_dim, num_classes=max(2, len(CONVENTIONAL_AIR_CLASSES)), hidden_dim=hidden_dim, dropout=dropout)


class GroundTargetExpert(MLPExpert):
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__(input_dim=input_dim, num_classes=len(GROUND_TARGET_CLASSES), hidden_dim=hidden_dim, dropout=dropout)


class RuleBasedHighAltitudeExpert(nn.Module):
    def __init__(self, altitude_threshold_m: float = 5000.0, speed_threshold_mps: float = 100.0) -> None:
        super().__init__()
        self.altitude_threshold_m = altitude_threshold_m
        self.speed_threshold_mps = speed_threshold_mps

    def forward(self, height_mean: torch.Tensor, speed_mean: torch.Tensor) -> ExpertOutput:
        positive = (height_mean > self.altitude_threshold_m) & (speed_mean > self.speed_threshold_mps)
        logits = torch.zeros((height_mean.shape[0], 2), dtype=height_mean.dtype, device=height_mean.device)
        logits[:, 0] = torch.where(positive, torch.tensor(-1.0, device=height_mean.device), torch.tensor(1.0, device=height_mean.device))
        logits[:, 1] = torch.where(positive, torch.tensor(1.0, device=height_mean.device), torch.tensor(-1.0, device=height_mean.device))
        return ExpertOutput(logits=logits)
