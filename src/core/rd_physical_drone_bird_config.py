from __future__ import annotations

from dataclasses import dataclass

from .data.rd_physical_window_dataset import (
    LEGACY_QUALITY_SCHEMA,
    PHYSICAL_RD_INPUT_CHANNELS,
    QUALITY_FEATURE_DIM,
    RD_ONLY_QUALITY_NAMES,
    RD_ONLY_QUALITY_SCHEMA,
    QUALITY_FEATURE_NAMES,
)


@dataclass(frozen=True)
class PhysicalRDDroneBirdConfig:
    index_name: str = "expert1_rd_only_causal_window_index_v1_physical_v3_1.csv"
    normalization_stats_name: str = "expert1_rd_physical_train_db_normalization_v4.json"
    feature_stats_name: str = "expert1_rd_physical_feature_stats_v4.json"
    output_dir: str = "artifacts/rd_physical_drone_bird"
    epochs: int = 18
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.1
    input_channels: int = PHYSICAL_RD_INPUT_CHANNELS
    token_dim: int = 256
    quality_dim: int = QUALITY_FEATURE_DIM
    quality_schema: str = LEGACY_QUALITY_SCHEMA
    include_missing_rd: bool = False
    prefix_max_loss_weight: float = 0.5
    prefix_aux_weight: float = 0.2
    micro_aux_weight: float = 0.1
    bird_loss_weight: float = 1.0
    focal_gamma: float = 0.0
    evidence_target_frames: int = 24
    max_train_batches_per_epoch: int = 0
    max_val_batches: int = 0
    log_interval_batches: int = 25
    num_workers: int = 0
    seed: int = 42
    device: str = "cuda"
    run_test_evaluation: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least one")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.token_dim < 32:
            raise ValueError("token_dim must be at least 32")
        if self.input_channels != PHYSICAL_RD_INPUT_CHANNELS:
            raise ValueError(
                f"physical RD input_channels is schema-fixed at {PHYSICAL_RD_INPUT_CHANNELS}; "
                f"got {self.input_channels}"
            )
        expected_quality_dim = len(QUALITY_FEATURE_NAMES) if self.quality_schema == LEGACY_QUALITY_SCHEMA else len(RD_ONLY_QUALITY_NAMES) if self.quality_schema == RD_ONLY_QUALITY_SCHEMA else -1
        if expected_quality_dim < 0:
            raise ValueError(f"unknown physical RD quality schema: {self.quality_schema!r}")
        if self.quality_dim != expected_quality_dim:
            raise ValueError(
                f"physical RD quality_dim is schema-fixed at {expected_quality_dim} for {self.quality_schema}; got {self.quality_dim}"
            )
        if not 0.0 < self.prefix_max_loss_weight <= 1.0:
            raise ValueError("prefix_max_loss_weight must be in (0, 1]")
        if self.prefix_aux_weight < 0.0 or self.micro_aux_weight < 0.0:
            raise ValueError("auxiliary loss weights must be non-negative")
        if self.bird_loss_weight <= 0.0:
            raise ValueError("bird_loss_weight must be positive")
        if self.focal_gamma < 0.0:
            raise ValueError("focal_gamma must be non-negative")
        if self.evidence_target_frames < 1:
            raise ValueError("evidence_target_frames must be positive")
        if self.max_train_batches_per_epoch < 0 or self.max_val_batches < 0:
            raise ValueError("batch limits must be non-negative")
        if self.log_interval_batches < 1:
            raise ValueError("log_interval_batches must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.run_test_evaluation:
            raise ValueError("physical RD training does not support test evaluation")
