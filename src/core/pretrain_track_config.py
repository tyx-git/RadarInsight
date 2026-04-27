from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrackPretrainConfig:
    data_root: str = "data"
    output_dir: str = "artifacts/pretrain_track"
    batch_size: int = 32
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    model_dim: int = 128
    dropout: float = 0.1
    mask_ratio: float = 0.15
    contrastive_temperature: float = 0.1
    mtm_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.2
    seed: int = 42
    device: str = "cuda"
