"""Configuration and provenance contracts for the isolated V27b experiment.

V27b consumes a frozen V27a main checkpoint and its known-class feature
caches.  It trains one small, zero-initialised correction head and never
opens Unknown or test payloads.  The old paths embedded in the archived V27a
artifacts are part of the input provenance, not paths that this package
resolves at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping




V27B_EXPERIMENT_ID = "V27b"
V27B_CONFIG_SCHEMA_VERSION = "RADAR-LN3-V27b-config-v2"
V27B_SCHEMA_VERSION = V27B_CONFIG_SCHEMA_VERSION

V27B_L2_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
V27B_TREATMENT_ARM = "conditional"



@dataclass(frozen=True)
class V27BBaseConfig:
    v27a_checkpoint_path: str
    v27a_train_cache_path: str
    v27a_validation_cache_path: str
    v27a_runtime_config_path: str
    v27a_runtime_manifest_path: str
    expected_v27a_checkpoint_file_sha256: str
    expected_v27a_checkpoint_state_sha256: str
    expected_v27a_model_contract_sha256: str
    expected_v27a_checkpoint_training_config_identity_sha256: str
    expected_v27a_checkpoint_training_manifest_sha256: str
    expected_v27a_train_cache_file_sha256: str
    expected_v27a_validation_cache_file_sha256: str
    expected_v27a_train_cache_payload_sha256: str
    expected_v27a_validation_cache_payload_sha256: str
    expected_v27a_runtime_config_sha256: str
    expected_v27a_runtime_manifest_sha256: str
    expected_v27a_runtime_config_identity_sha256: str
    expected_v27a_runtime_manifest_payload_sha256: str
    expected_l2_class_weights: tuple[float, ...]
    expected_v27a_epoch: int
    expected_v27a_arm: str
    expected_v27a_seed: int


@dataclass(frozen=True)
class V27BDataConfig:
    class_names: tuple[str, ...]
    stable8_dim: int
    max_sequence_length: int
    max_frames_per_event: int
    feature_cache_schema: str
    expected_v27a_control_validation_confusion_matrix: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class V27BModelConfig:
    parent_state_dim: int
    event_representation_dim: int
    decision_state_dim: int
    input_dim: int
    hidden_dim: int
    class_count: int
    activation: str
    residual_bound: float
    pairwise_delta_bound: float
    residual_projection_bias: bool
    residual_zero_initialization: bool
    new_correction_pair: str
    evidence_carrier: str
    frame_encoder_hidden_dim: int
    frame_encoder_layers: int


@dataclass(frozen=True)
class V27BTrainingConfig:
    seeds: tuple[int, ...]
    dataloader_seed: int
    batch_size: int
    epochs: int
    primary_epoch: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    device: str
    eligible_prefix_min_events: int
    bootstrap_replicates: int
    loss_name: str
    loss_reduction: str


@dataclass(frozen=True)
class V27BArtifactConfig:
    manifest_path: str
    train_feature_cache_path: str
    validation_feature_cache_path: str
    smoke_report_path: str
    training_log_template: str
    training_summary_template: str
    epoch_checkpoint_template: str
    evaluation_report_template: str
    paired_evaluation_report_path: str


@dataclass(frozen=True)
class V27BConfig:
    schema_version: str
    experiment_id: str
    variant: str
    objective: str
    base: V27BBaseConfig
    data: V27BDataConfig
    model: V27BModelConfig
    training: V27BTrainingConfig
    artifacts: V27BArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


































































__all__ = ["V27B_CONFIG_SCHEMA_VERSION", "V27BConfig", "V27BDataConfig", "V27B_EXPERIMENT_ID", "V27B_L2_CLASS_NAMES", "V27B_SCHEMA_VERSION", "V27B_TREATMENT_ARM"]