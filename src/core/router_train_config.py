from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouterTrainConfig:
    data_root: str = "data"
    output_dir: str = "artifacts/router_v1"
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    balanced_sampling: bool = True
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda"
