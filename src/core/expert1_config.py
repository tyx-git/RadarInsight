from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Expert1TrainConfig:
    data_root: str = "data"
    output_dir: str = "artifacts/expert1_v1"
    batch_size: int = 8
    num_workers: int = 0
    epochs: int = 20
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    dropout: float = 0.1
    low_altitude_aux_weight: float = 0.5
    low_altitude_mutual_weight: float = 1.0
    low_altitude_contrastive_weight: float = 0.2
    low_altitude_temperature: float = 0.1
    cosface_scale: float = 16.0
    cosface_margin: float = 0.2
    reject_threshold: float = 0.35
    rd_activation_threshold: float = 0.8
    rd_sample_boost: float = 1.2
    raw_source_boost: float = 2.0
    excel_source_boost: float = 2.0
    cq08_source_boost: float = 1.0
    unknown_sample_boost: float = 1.0
    source_normalization: bool = True
    unknown_loss_weight: float = 1.0
    enable_reject_in_train: bool = False
    enable_reject_in_eval: bool = False
    stage1_traj_only_epochs: int = 12
    stage2_ramp_epochs: int = 6
    prefer_stage2_checkpoint: bool = True
    sequence_encoder_lr_multiplier: float = 0.1
    pretrained_track_checkpoint: str = ""
    freeze_track_encoder_epochs: int = 0
    skip_non_finite_batch: bool = True
    seed: int = 42
    device: str = "cuda"
