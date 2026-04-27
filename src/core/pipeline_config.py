from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MainPipelineConfig:
    data_root: str = "data"
    output_root: str = "artifacts/main_pipeline"
    device: str = "cuda"
    pretrain_epochs: int = 5
    router_epochs: int = 5
    expert1_epochs: int = 5
    expert2_epochs: int = 5
    expert4_epochs: int = 5
    system_epochs: int = 10
    pretrain_batch_size: int = 16
    router_batch_size: int = 32
    expert1_batch_size: int = 8
    expert2_batch_size: int = 16
    expert4_batch_size: int = 16
    system_batch_size: int = 16
    learning_rate: float = 3e-4
    freeze_track_encoder_epochs: int = 1
    stage_warmup_epochs: int = 1
    stage_expert_warmup_epochs: int = 1
    stage_joint_epochs: int = 3
