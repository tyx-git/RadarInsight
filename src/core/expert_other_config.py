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
    balanced_sampling: bool = True
    balanced_positive_multiplier: int = 8
    focal_loss_gamma: float = 1.5
    positive_class_weight: float = 12.0
    negative_class_weight: float = 1.0
    aircraft_positive_noise_std: float = 0.015
    open_set_aircraft: bool = True
    open_set_logit_scale: float = 10.0
    open_set_initial_threshold: float = 0.35
    threshold_min: float = -0.2
    threshold_max: float = 0.8
    threshold_steps: int = 101
    selection_metric: str = "aircraft_macro_f1"
    include_ground_hard_negatives: bool = True
    groundness_loss_weight: float = 0.7
    groundness_positive_weight: float = 1.0
    groundness_negative_weight: float = 1.5
    groundness_initial_threshold: float = 0.5
    groundness_threshold_min: float = 0.2
    groundness_threshold_max: float = 0.9
    groundness_threshold_steps: int = 71
    seed: int = 42
    device: str = "cuda"
