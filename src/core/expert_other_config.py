from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertOtherTrainConfig:
    data_root: str = "data"
    output_dir: str = "artifacts/expert_other"
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    expert_name: str = "expert2_air"
    pretrained_track_checkpoint: str = ""
    shared_representation_checkpoint: str = ""
    train_representation_builder: bool = False
    source_normalization: bool = True
    seed: int = 42
    device: str = "cuda"
