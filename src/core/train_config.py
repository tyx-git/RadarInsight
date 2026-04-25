from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainConfig:
    data_root: str = "data"
    output_dir: str = "artifacts/baseline_gru"
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.1
    seed: int = 42
    device: str = "cuda"
