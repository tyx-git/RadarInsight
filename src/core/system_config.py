from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemTrainConfig:
    data_root: str = "data"
    output_dir: str = "artifacts/system_v1"
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 12
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    hidden_dim: int = 128
    dropout: float = 0.1
    route_loss_weight: float = 0.3
    skip_non_finite_batch: bool = True
    seed: int = 42
    device: str = "cuda"
