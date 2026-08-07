from __future__ import annotations

from dataclasses import dataclass

from .data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES
from .data.rd_physical_window_dataset import PHYSICAL_RD_INPUT_CHANNELS


@dataclass(frozen=True)
class PhysicalRDFiveConfig:
    index_name: str = "expert1_rd_only_causal_window_index_v1_physical_v3_1.csv"
    normalization_stats_name: str = "expert1_rd_physical_train_db_normalization_five_v1.json"
    feature_stats_name: str = "expert1_rd_physical_feature_stats_five_v1.json"
    output_dir: str = "artifacts/rd-only-five"
    epochs: int = 18
    batch_size: int = 4
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    dropout: float = 0.1
    input_channels: int = PHYSICAL_RD_INPUT_CHANNELS
    token_dim: int = 256
    quality_dim: int = len(RD_FIVE_QUALITY_NAMES)
    include_missing_rd: bool = False
    prefix_max_loss_weight: float = 0.5
    cascade_loss_weight: float = 0.25
    parent_loss_weight: float = 0.15
    prefix_aux_weight: float = 0.10
    micro_aux_weight: float = 0.10
    focal_gamma: float = 0.0
    evidence_target_frames: int = 24
    class_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0, 1.0)
    parent_partition: tuple[tuple[int, ...], ...] = ((0, 1, 2), (3, 4))
    tr_entry_preserved: bool = True
    allow_tr_inputs: bool = False
    deployment_head: str = "flat"
    max_train_batches_per_epoch: int = 0
    max_val_batches: int = 0
    log_interval_batches: int = 25
    num_workers: int = 0
    drone_reference_precision: float = 0.9858156028368794
    drone_reference_recall: float = 0.9858156028368794
    drone_reference_f1: float = 0.9858156028368794
    bird_reference_precision: float = 0.9655831739961759
    bird_reference_recall: float = 0.9655831739961759
    bird_reference_f1: float = 0.9655831739961759
    non_degradation_tolerance: float = 0.0
    enforce_drone_bird_non_degradation: bool = True
    reference_source: str = "Stage011-v9-binary-validation"
    seed: int = 42
    device: str = "cuda"
    run_test_evaluation: bool = False

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.input_channels != PHYSICAL_RD_INPUT_CHANNELS:
            raise ValueError(f"five-class physical RD input_channels is schema-fixed at {PHYSICAL_RD_INPUT_CHANNELS}")
        if self.quality_dim != len(RD_FIVE_QUALITY_NAMES):
            raise ValueError(f"five-class physical RD quality_dim is schema-fixed at {len(RD_FIVE_QUALITY_NAMES)}")
        if self.token_dim < 32:
            raise ValueError("token_dim must be at least 32")
        if not 0.0 < self.prefix_max_loss_weight <= 1.0:
            raise ValueError("prefix_max_loss_weight must be in (0, 1]")
        for name, value in (
            ("cascade_loss_weight", self.cascade_loss_weight),
            ("parent_loss_weight", self.parent_loss_weight),
            ("prefix_aux_weight", self.prefix_aux_weight),
            ("micro_aux_weight", self.micro_aux_weight),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.focal_gamma < 0.0 or self.evidence_target_frames < 1:
            raise ValueError("focal_gamma must be non-negative and evidence_target_frames positive")
        if self.max_train_batches_per_epoch < 0 or self.max_val_batches < 0 or self.num_workers < 0:
            raise ValueError("batch limits and num_workers must be non-negative")
        reference_values = (
            self.drone_reference_precision,
            self.drone_reference_recall,
            self.drone_reference_f1,
            self.bird_reference_precision,
            self.bird_reference_recall,
            self.bird_reference_f1,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in reference_values):
            raise ValueError("Drone/Bird reference metrics must be in [0, 1]")
        if not 0.0 <= self.non_degradation_tolerance <= 1.0:
            raise ValueError("non_degradation_tolerance must be in [0, 1]")
        if not str(self.reference_source).strip():
            raise ValueError("reference_source must be non-empty")
        if self.include_missing_rd:
            raise ValueError("five-class training requires complete RD windows; missing RD is unavailable, not a class")
        if self.log_interval_batches < 1:
            raise ValueError("log_interval_batches must be positive")
        if self.deployment_head not in {"flat", "cascade"}:
            raise ValueError("deployment_head must be 'flat' or 'cascade'")
        if not self.tr_entry_preserved or self.allow_tr_inputs:
            raise ValueError("RD-only five-class training must preserve the TR entry but disallow TR inputs")
        if self.run_test_evaluation:
            raise ValueError("five-class RD training does not support test evaluation")
        weights = tuple(float(value) for value in self.class_weights)
        if len(weights) != len(RD_FIVE_LABELS) or any(value <= 0.0 for value in weights):
            raise ValueError("class_weights must contain five positive values in LOW_ALTITUDE_CLASSES order")
        object.__setattr__(self, "class_weights", weights)
        partition = tuple(tuple(int(index) for index in group) for group in self.parent_partition)
        flattened = [index for group in partition for index in group]
        if sorted(flattened) != list(range(len(RD_FIVE_LABELS))):
            raise ValueError("parent_partition must cover five classes exactly once")
        object.__setattr__(self, "parent_partition", partition)


__all__ = ["PhysicalRDFiveConfig"]
