"""Immutable configuration and provenance protocol for the complete V27 baseline.

The V27 system has two independently parameterised learned layers (L1 and
L2), plus an L3 action resolver.  This module owns the one configuration
parser and the one data/implementation registry.  It deliberately does not
open test or Unknown payloads while preparing classifier training.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import numpy as np

from radar_ln3.data import REQUIRED_INDEX_COLUMNS, load_micro_doppler_cache, load_rd_physical_cache
from radar_ln3.known_gate_features import V21E_COURSE_PREFIX_VARIANT, load_known_gate_feature_cache
from radar_ln3.runtime import canonical_sha256, parse_bool, relative_project_path, resolve_project_path, sequence_sha256, sha256_file, verify_file_hash


V27_SCHEMA_VERSION = "RADAR-LN3-V27"
V27_EXPERIMENT_ID = "V27"
V27_SYSTEM_TASK_NAME = "complete_l1_l2_l3_baseline"
V27_L1_TASK_NAME = "known_gate"
V27_L2_TASK_NAME = "dbb_specialist"
V27_L1_CLASS_NAMES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
V27_L2_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
V27_L1_LABEL_TO_ID = {
    "DroneTarget": 0,
    "BirdTarget": 0,
    "BalloonTarget": 0,
    "ClutterTarget": 1,
}
V27_L2_LABEL_TO_ID = {name: index for index, name in enumerate(V27_L2_CLASS_NAMES)}
V27_L1_ALLOWED_LABELS = frozenset(V27_L1_LABEL_TO_ID)
V27_L2_ALLOWED_LABELS = frozenset(V27_L2_CLASS_NAMES)
V27_ALLOWED_TASKS = frozenset((V27_L1_TASK_NAME, V27_L2_TASK_NAME))
V27_EQUATION_VERSION = "RADAR-LN3-V27-liquid-v2-raw-memory-pre-norm-v1"
V27_LOSS_NAME = "parent_final_fixed_parent_count_ce"
V27_LOSS_REDUCTION = "fixed_parent_count"
V27_EXPERIMENT_ROOT = Path("src-v3") / "v27" / "v27_baseline"
V27_DEFAULT_CONFIG = V27_EXPERIMENT_ROOT / "config" / "V27-liquid-v2-dbb-baseline.json"

EXPECTED_L1_PARENT_COUNTS = {
    "train": {"KNOWN_TARGET_EVIDENCE": 953, "KNOWN_CLUTTER_EVIDENCE": 33},
    "val": {"KNOWN_TARGET_EVIDENCE": 223, "KNOWN_CLUTTER_EVIDENCE": 9},
}
EXPECTED_L2_PARENT_COUNTS = {
    "train": {"DroneTarget": 476, "BirdTarget": 239, "BalloonTarget": 238},
    "val": {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54},
}


@dataclass(frozen=True)
class V27L1DataConfig:
    contract_path: str
    normalization_stats_path: str
    feature_cache_manifest_path: str
    train_course_cache_path: str
    validation_course_cache_path: str
    expected_contract_sha256: str
    expected_normalization_stats_sha256: str
    expected_feature_cache_manifest_sha256: str
    expected_train_course_cache_sha256: str
    expected_validation_course_cache_sha256: str
    rd_amplitude_center: float
    rd_amplitude_scale: float
    course_feature_name: str
    course_feature_variant: str
    course_feature_dim: int
    course_minimum_valid_nonfirst_points: int


@dataclass(frozen=True)
class V27L2DataConfig:
    contract_path: str
    normalization_stats_path: str
    feature_source_normalization_stats_path: str
    cache_manifest_path: str
    train_rd_physical_cache_path: str
    validation_rd_physical_cache_path: str
    train_micro_doppler_cache_path: str
    validation_micro_doppler_cache_path: str
    expected_contract_sha256: str
    expected_normalization_stats_sha256: str
    expected_feature_source_normalization_stats_sha256: str
    expected_cache_manifest_sha256: str
    expected_train_rd_physical_cache_sha256: str
    expected_validation_rd_physical_cache_sha256: str
    expected_train_micro_doppler_cache_sha256: str
    expected_validation_micro_doppler_cache_sha256: str
    rd_amplitude_center: float
    rd_amplitude_scale: float
    rd_physical_schema: str
    rd_physical_dim: int
    micro_doppler_schema: str
    micro_doppler_dim: int


@dataclass(frozen=True)
class V27DataConfig:
    index_path: str
    expected_index_sha256: str
    index_schema_version: str
    source_name: str
    train_split_name: str
    validation_split_name: str
    test_split_name: str
    max_sequence_length: int
    max_rd_frames_per_point: int
    rd_quality_schema: str
    l1: V27L1DataConfig
    l2: V27L2DataConfig


@dataclass(frozen=True)
class V27ModelConfig:
    tr_value_dim: int
    tr_missing_dim: int
    rd_quality_dim: int
    rd_height: int
    rd_width: int
    rd_embedding_dim: int
    hidden_dim: int
    liquid_input_dim: int
    liquid_hidden_dim: int
    rd_frame_aggregation: str
    equation_version: str
    memory_state: str
    post_update_normalization: bool
    classifier_readout_normalization: str
    classifier_readout_shares_liquid_norm: bool
    knownness_projection_dim: int
    knownness_summary: str
    knownness_score_status: str
    quality_head_status: str
    tau_min_seconds: float
    tau_max_seconds: float
    zero_dt_event_floor_seconds: float
    dt_max_seconds: float
    invalid_dt_update_mode: str
    invalid_dt_observation_gain: float
    dt_reason_mode: str
    dt_valid_feature_mode: str


@dataclass(frozen=True)
class V27TrainingConfig:
    seeds: tuple[int, ...]
    dataloader_seed: int
    batch_size: int
    epochs: int
    early_stopping_patience: int
    learning_rate: float
    optimizer: str
    weight_decay: float
    gradient_clip_norm: float
    loss_name: str
    loss_reduction: str
    l1_class_weights: tuple[float, ...]
    l2_class_weights: tuple[float, ...]
    prefix_auxiliary_weight: float
    eligible_prefix_min_events: int
    checkpoint_selection_metrics: Mapping[str, str]
    checkpoint_selection_uses_test: bool
    checkpoint_selection_uses_unknown: bool
    save_every_epoch: bool
    device: str
    cuda_deterministic_algorithms: bool
    overfit_parent_count: int
    overfit_steps: int


@dataclass(frozen=True)
class V27OpenSetConfig:
    """Configuration for the post-classification open-set stages.

    These values deliberately live outside ``V27TrainingConfig``.  The
    classifier stage must not accidentally consume a head/calibration
    setting, and the resulting hashes make that separation visible in every
    derived artifact.
    """

    representation_batch_size: int
    head_batch_size: int
    head_epochs: int
    head_learning_rate: float
    head_weight_decay: float
    head_gradient_clip_norm: float
    knownness_min_events: int
    quality_min_events: int
    quality_min_rd_fraction: float
    quality_min_dt_fraction: float
    quality_min_aux_fraction: float
    knownness_recall_floor: float
    quality_recall_floor: float
    quality_min_final_negative_support: int
    conformal_error_rate: float
    action_known_target_recall_floor: float
    action_clutter_recall_floor: float
    action_conformal_error_rate_grid_size: int
    action_knownness_threshold_grid_size: int
    temperature_min: float
    temperature_max: float


@dataclass(frozen=True)
class V27L3Config:
    schema_version: str
    calibration_status: str
    calibration_registry_path: str
    dynamic_calibration_required: bool
    fixed_softmax_threshold_allowed: bool
    single_distance_gate_allowed: bool
    quality_supervision_status: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class V27ArtifactConfig:
    manifest_path: str
    training_log_template: str
    training_summary_template: str
    best_checkpoint_template: str
    last_checkpoint_template: str
    epoch_checkpoint_dir_template: str
    evaluation_report_template: str
    evaluation_summary_path: str
    overfit_report_path: str
    contract_smoke_report_path: str
    head_training_log_template: str
    head_training_summary_template: str
    head_checkpoint_template: str
    knownness_head_training_summary_template: str
    knownness_head_checkpoint_template: str
    quality_head_training_summary_template: str
    quality_head_checkpoint_template: str
    representation_cache_manifest_path: str
    representation_cache_template: str
    head_score_cache_template: str
    open_set_evaluation_report_path: str
    open_set_audit_report_path: str
    open_set_predictions_path: str


@dataclass(frozen=True)
class V27Config:
    schema_version: str
    experiment_id: str
    direct_base_experiment: str
    variant: str
    objective: str
    data: V27DataConfig
    model: V27ModelConfig
    training: V27TrainingConfig
    open_set: V27OpenSetConfig
    l3: V27L3Config
    artifacts: V27ArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27 {context} is missing {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27 {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27 {context}.{key} must be a non-empty string")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"V27 {context}.{key} must be a positive integer")
    return int(value)


def _finite_float(mapping: Mapping[str, Any], key: str, context: str, *, positive: bool = False) -> float:
    raw = _required(mapping, key, context)
    if isinstance(raw, bool):
        raise ValueError(f"V27 {context}.{key} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or (positive and value <= 0.0):
        raise ValueError(f"V27 {context}.{key} must be {'positive and ' if positive else ''}finite")
    return value


def _sha(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(mapping, key, context).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"V27 {context}.{key} must be a SHA-256 digest")
    return value


def _positive_weights(mapping: Mapping[str, Any], key: str, expected: int) -> tuple[float, ...]:
    raw = _required(mapping, key, "training")
    if not isinstance(raw, (list, tuple)) or len(raw) != expected:
        raise ValueError(f"V27 training.{key} must have {expected} entries")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"V27 training.{key} must be positive finite")
    return values


def _task_classes(payload: Mapping[str, Any], task: str) -> tuple[str, ...]:
    task_payload = _mapping(_mapping(payload, "task", "root"), task, "task")
    values = tuple(str(value) for value in _required(task_payload, "class_names", f"task.{task}"))
    expected = V27_L1_CLASS_NAMES if task == "l1" else V27_L2_CLASS_NAMES
    if values != expected:
        raise ValueError(f"V27 task.{task} class order differs from the canonical contract")
    return values


def load_v27_config(project_root: str | Path, config_path: str | Path | None = None) -> V27Config:
    root = Path(project_root).resolve()
    candidate = root / V27_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("V27 configuration must remain inside project root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema_version") != V27_SCHEMA_VERSION:
        raise ValueError("V27 configuration schema differs")
    if payload.get("experiment_id") != V27_EXPERIMENT_ID:
        raise ValueError("V27 experiment_id must be exactly V27")
    task = _mapping(payload, "task", "root")
    if task.get("name") != V27_SYSTEM_TASK_NAME:
        raise ValueError("V27 must declare the complete L1-L2-L3 system task")
    if parse_bool(_required(task, "unknown_is_model_class", "task"), field="task.unknown_is_model_class"):
        raise ValueError("Unknown cannot be a V27 model class")
    if int(_required(task, "unknown_training_exposure", "task")) != 0:
        raise ValueError("V27 classifier training must have zero Unknown exposure")
    for key in ("open_set_capability", "knownness_head", "quality_head", "semantic_decision_layer"):
        if not parse_bool(_required(task, key, "task"), field=f"task.{key}"):
            raise ValueError(f"V27 must enable task.{key}")
    _task_classes(payload, "l1")
    _task_classes(payload, "l2")
    l1_task = _mapping(task, "l1", "task")
    l2_task = _mapping(task, "l2", "task")
    if {str(key): int(value) for key, value in _mapping(l1_task, "label_to_id", "task.l1").items()} != V27_L1_LABEL_TO_ID:
        raise ValueError("V27 L1 label mapping differs")
    if {str(key): int(value) for key, value in _mapping(l2_task, "label_to_id", "task.l2").items()} != V27_L2_LABEL_TO_ID:
        raise ValueError("V27 L2 label mapping differs")
    if set(l1_task.get("forbidden_training_labels") or ()) != {"UnknownTarget"}:
        raise ValueError("V27 L1 must forbid only UnknownTarget from known-class training")
    if set(l2_task.get("forbidden_training_labels") or ()) != {"ClutterTarget", "UnknownTarget"}:
        raise ValueError("V27 L2 forbidden labels differ")

    data_raw = _mapping(payload, "data", "root")
    l1_raw = _mapping(data_raw, "l1", "data")
    l2_raw = _mapping(data_raw, "l2", "data")
    l1 = V27L1DataConfig(
        contract_path=_text(l1_raw, "contract_path", "data.l1"),
        normalization_stats_path=_text(l1_raw, "normalization_stats_path", "data.l1"),
        feature_cache_manifest_path=_text(l1_raw, "feature_cache_manifest_path", "data.l1"),
        train_course_cache_path=_text(l1_raw, "train_course_cache_path", "data.l1"),
        validation_course_cache_path=_text(l1_raw, "validation_course_cache_path", "data.l1"),
        expected_contract_sha256=_sha(l1_raw, "expected_contract_sha256", "data.l1"),
        expected_normalization_stats_sha256=_sha(l1_raw, "expected_normalization_stats_sha256", "data.l1"),
        expected_feature_cache_manifest_sha256=_sha(l1_raw, "expected_feature_cache_manifest_sha256", "data.l1"),
        expected_train_course_cache_sha256=_sha(l1_raw, "expected_train_course_cache_sha256", "data.l1"),
        expected_validation_course_cache_sha256=_sha(l1_raw, "expected_validation_course_cache_sha256", "data.l1"),
        rd_amplitude_center=_finite_float(l1_raw, "rd_amplitude_center", "data.l1"),
        rd_amplitude_scale=_finite_float(l1_raw, "rd_amplitude_scale", "data.l1", positive=True),
        course_feature_name=_text(l1_raw, "course_feature_name", "data.l1"),
        course_feature_variant=_text(l1_raw, "course_feature_variant", "data.l1"),
        course_feature_dim=_positive_int(l1_raw, "course_feature_dim", "data.l1"),
        course_minimum_valid_nonfirst_points=_positive_int(l1_raw, "course_minimum_valid_nonfirst_points", "data.l1"),
    )
    l2 = V27L2DataConfig(
        contract_path=_text(l2_raw, "contract_path", "data.l2"),
        normalization_stats_path=_text(l2_raw, "normalization_stats_path", "data.l2"),
        feature_source_normalization_stats_path=_text(l2_raw, "feature_source_normalization_stats_path", "data.l2"),
        cache_manifest_path=_text(l2_raw, "cache_manifest_path", "data.l2"),
        train_rd_physical_cache_path=_text(l2_raw, "train_rd_physical_cache_path", "data.l2"),
        validation_rd_physical_cache_path=_text(l2_raw, "validation_rd_physical_cache_path", "data.l2"),
        train_micro_doppler_cache_path=_text(l2_raw, "train_micro_doppler_cache_path", "data.l2"),
        validation_micro_doppler_cache_path=_text(l2_raw, "validation_micro_doppler_cache_path", "data.l2"),
        expected_contract_sha256=_sha(l2_raw, "expected_contract_sha256", "data.l2"),
        expected_normalization_stats_sha256=_sha(l2_raw, "expected_normalization_stats_sha256", "data.l2"),
        expected_feature_source_normalization_stats_sha256=_sha(l2_raw, "expected_feature_source_normalization_stats_sha256", "data.l2"),
        expected_cache_manifest_sha256=_sha(l2_raw, "expected_cache_manifest_sha256", "data.l2"),
        expected_train_rd_physical_cache_sha256=_sha(l2_raw, "expected_train_rd_physical_cache_sha256", "data.l2"),
        expected_validation_rd_physical_cache_sha256=_sha(l2_raw, "expected_validation_rd_physical_cache_sha256", "data.l2"),
        expected_train_micro_doppler_cache_sha256=_sha(l2_raw, "expected_train_micro_doppler_cache_sha256", "data.l2"),
        expected_validation_micro_doppler_cache_sha256=_sha(l2_raw, "expected_validation_micro_doppler_cache_sha256", "data.l2"),
        rd_amplitude_center=_finite_float(l2_raw, "rd_amplitude_center", "data.l2"),
        rd_amplitude_scale=_finite_float(l2_raw, "rd_amplitude_scale", "data.l2", positive=True),
        rd_physical_schema=_text(l2_raw, "rd_physical_schema", "data.l2"),
        rd_physical_dim=_positive_int(l2_raw, "rd_physical_dim", "data.l2"),
        micro_doppler_schema=_text(l2_raw, "micro_doppler_schema", "data.l2"),
        micro_doppler_dim=_positive_int(l2_raw, "micro_doppler_dim", "data.l2"),
    )
    data = V27DataConfig(
        index_path=_text(data_raw, "index_path", "data"),
        expected_index_sha256=_sha(data_raw, "expected_index_sha256", "data"),
        index_schema_version=_text(data_raw, "index_schema_version", "data"),
        source_name=_text(data_raw, "source_name", "data"),
        train_split_name=_text(data_raw, "train_split_name", "data"),
        validation_split_name=_text(data_raw, "validation_split_name", "data"),
        test_split_name=_text(data_raw, "test_split_name", "data"),
        max_sequence_length=_positive_int(data_raw, "max_sequence_length", "data"),
        max_rd_frames_per_point=_positive_int(data_raw, "max_rd_frames_per_point", "data"),
        rd_quality_schema=_text(data_raw, "rd_quality_schema", "data"),
        l1=l1,
        l2=l2,
    )
    if data.source_name != "CQ08" or (data.train_split_name, data.validation_split_name, data.test_split_name) != ("train", "val", "test"):
        raise ValueError("V27 data source/split contract differs")
    if l1.course_feature_name != "causal_prefix_course_circular_incoherence" or l1.course_feature_variant != V21E_COURSE_PREFIX_VARIANT or l1.course_feature_dim != 1 or l1.course_minimum_valid_nonfirst_points != 3:
        raise ValueError("V27 L1 course feature contract differs from V21e")
    if l2.rd_physical_dim != 4 or l2.micro_doppler_dim != 8:
        raise ValueError("V27 L2 must use physical4 and stable8")

    model_raw = _mapping(payload, "model", "root")
    model = V27ModelConfig(
        tr_value_dim=_positive_int(model_raw, "tr_value_dim", "model"), tr_missing_dim=_positive_int(model_raw, "tr_missing_dim", "model"), rd_quality_dim=_positive_int(model_raw, "rd_quality_dim", "model"), rd_height=_positive_int(model_raw, "rd_height", "model"), rd_width=_positive_int(model_raw, "rd_width", "model"), rd_embedding_dim=_positive_int(model_raw, "rd_embedding_dim", "model"), hidden_dim=_positive_int(model_raw, "hidden_dim", "model"), liquid_input_dim=_positive_int(model_raw, "liquid_input_dim", "model"), liquid_hidden_dim=_positive_int(model_raw, "liquid_hidden_dim", "model"), rd_frame_aggregation=_text(model_raw, "rd_frame_aggregation", "model"), equation_version=_text(model_raw, "equation_version", "model"), memory_state=_text(model_raw, "memory_state", "model"), post_update_normalization=parse_bool(_required(model_raw, "post_update_normalization", "model"), field="model.post_update_normalization"), classifier_readout_normalization=_text(model_raw, "classifier_readout_normalization", "model"), classifier_readout_shares_liquid_norm=parse_bool(_required(model_raw, "classifier_readout_shares_liquid_norm", "model"), field="model.classifier_readout_shares_liquid_norm"), knownness_projection_dim=_positive_int(model_raw, "knownness_projection_dim", "model"), knownness_summary=_text(model_raw, "knownness_summary", "model"), knownness_score_status=_text(model_raw, "knownness_score_status", "model"), quality_head_status=_text(model_raw, "quality_head_status", "model"), tau_min_seconds=_finite_float(model_raw, "tau_min_seconds", "model", positive=True), tau_max_seconds=_finite_float(model_raw, "tau_max_seconds", "model", positive=True), zero_dt_event_floor_seconds=_finite_float(model_raw, "zero_dt_event_floor_seconds", "model", positive=True), dt_max_seconds=_finite_float(model_raw, "dt_max_seconds", "model", positive=True), invalid_dt_update_mode=_text(model_raw, "invalid_dt_update_mode", "model"), invalid_dt_observation_gain=_finite_float(model_raw, "invalid_dt_observation_gain", "model", positive=True), dt_reason_mode=_text(model_raw, "dt_reason_mode", "model"), dt_valid_feature_mode=_text(model_raw, "dt_valid_feature_mode", "model"),
    )
    if (model.tr_value_dim, model.tr_missing_dim, model.rd_quality_dim, model.rd_height, model.rd_width, model.rd_embedding_dim, model.hidden_dim, model.liquid_input_dim, model.liquid_hidden_dim) != (14, 11, 7, 21, 384, 24, 48, 48, 48):
        raise ValueError("V27 model dimensions differ from the frozen capacity")
    if model.equation_version != V27_EQUATION_VERSION or model.memory_state != "raw_r_t" or model.post_update_normalization:
        raise ValueError("V27 must use raw-memory Liquid V2 without post-update LayerNorm")
    # V27RDSpatialEncoder has one deliberate implementation: masked mean over
    # valid RD frames.  Reject other declarations instead of silently running
    # the same implementation under a different configuration label.
    if model.rd_frame_aggregation != "masked_mean":
        raise ValueError("V27 baseline requires masked_mean RD frame aggregation")
    if model.classifier_readout_normalization != "independent_layer_norm_on_raw_state" or model.classifier_readout_shares_liquid_norm:
        raise ValueError("V27 classifier norm contract differs")
    if model.invalid_dt_update_mode != "bounded_observation" or model.dt_valid_feature_mode != "deterministic_facts_allowed_no_learned_quality_feedback":
        raise ValueError("V27 active time/quality-feedback policy differs")
    if not (model.tau_min_seconds <= model.tau_max_seconds and model.dt_max_seconds >= model.zero_dt_event_floor_seconds and 0.0 < model.invalid_dt_observation_gain <= 1.0):
        raise ValueError("V27 time policy bounds differ")

    train_raw = _mapping(payload, "training", "root")
    raw_seeds = _required(train_raw, "seeds", "training")
    if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds or any(isinstance(value, bool) or int(value) < 0 for value in raw_seeds):
        raise ValueError("V27 training.seeds must be unique non-negative integers")
    seeds = tuple(int(value) for value in raw_seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("V27 training.seeds cannot repeat")
    training = V27TrainingConfig(
        seeds=seeds, dataloader_seed=_positive_int(train_raw, "dataloader_seed", "training"), batch_size=_positive_int(train_raw, "batch_size", "training"), epochs=_positive_int(train_raw, "epochs", "training"), early_stopping_patience=int(_required(train_raw, "early_stopping_patience", "training")), learning_rate=_finite_float(train_raw, "learning_rate", "training", positive=True), optimizer=_text(train_raw, "optimizer", "training"), weight_decay=_finite_float(train_raw, "weight_decay", "training"), gradient_clip_norm=_finite_float(train_raw, "gradient_clip_norm", "training", positive=True), loss_name=_text(train_raw, "loss_name", "training"), loss_reduction=_text(train_raw, "loss_reduction", "training"), l1_class_weights=_positive_weights(train_raw, "l1_class_weights", 2), l2_class_weights=_positive_weights(train_raw, "l2_class_weights", 3), prefix_auxiliary_weight=_finite_float(train_raw, "prefix_auxiliary_weight", "training"), eligible_prefix_min_events=_positive_int(train_raw, "eligible_prefix_min_events", "training"), checkpoint_selection_metrics={str(key): str(value) for key, value in _mapping(train_raw, "checkpoint_selection_metrics", "training").items()}, checkpoint_selection_uses_test=parse_bool(_required(train_raw, "checkpoint_selection_uses_test", "training"), field="training.checkpoint_selection_uses_test"), checkpoint_selection_uses_unknown=parse_bool(_required(train_raw, "checkpoint_selection_uses_unknown", "training"), field="training.checkpoint_selection_uses_unknown"), save_every_epoch=parse_bool(_required(train_raw, "save_every_epoch", "training"), field="training.save_every_epoch"), device=_text(train_raw, "device", "training"), cuda_deterministic_algorithms=parse_bool(_required(train_raw, "cuda_deterministic_algorithms", "training"), field="training.cuda_deterministic_algorithms"), overfit_parent_count=_positive_int(train_raw, "overfit_parent_count", "training"), overfit_steps=_positive_int(train_raw, "overfit_steps", "training"),
    )
    if training.optimizer != "AdamW" or training.loss_name != V27_LOSS_NAME or training.loss_reduction != V27_LOSS_REDUCTION or training.prefix_auxiliary_weight != 0.0:
        raise ValueError("V27 optimizer/loss contract differs")
    expected_selection = {
        V27_L1_TASK_NAME: "parent_min_precision_recall_f1_then_macro_f1",
        V27_L2_TASK_NAME: "parent_min_precision_recall_f1_then_min_drone_bird_errors_then_macro_f1",
    }
    if dict(training.checkpoint_selection_metrics) != expected_selection:
        raise ValueError("V27 task-specific checkpoint selection policy differs")
    if training.early_stopping_patience < 0 or training.weight_decay < 0.0 or training.device not in {"cpu", "cuda"} or training.checkpoint_selection_uses_test or training.checkpoint_selection_uses_unknown:
        raise ValueError("V27 training safety contract differs")

    open_set_raw = _mapping(payload, "open_set", "root")
    open_set = V27OpenSetConfig(
        representation_batch_size=_positive_int(
            open_set_raw, "representation_batch_size", "open_set"
        ),
        head_batch_size=_positive_int(open_set_raw, "head_batch_size", "open_set"),
        head_epochs=_positive_int(open_set_raw, "head_epochs", "open_set"),
        head_learning_rate=_finite_float(
            open_set_raw, "head_learning_rate", "open_set", positive=True
        ),
        head_weight_decay=_finite_float(
            open_set_raw, "head_weight_decay", "open_set"
        ),
        head_gradient_clip_norm=_finite_float(
            open_set_raw, "head_gradient_clip_norm", "open_set", positive=True
        ),
        knownness_min_events=_positive_int(
            open_set_raw, "knownness_min_events", "open_set"
        ),
        quality_min_events=_positive_int(
            open_set_raw, "quality_min_events", "open_set"
        ),
        quality_min_rd_fraction=_finite_float(
            open_set_raw, "quality_min_rd_fraction", "open_set"
        ),
        quality_min_dt_fraction=_finite_float(
            open_set_raw, "quality_min_dt_fraction", "open_set"
        ),
        quality_min_aux_fraction=_finite_float(
            open_set_raw, "quality_min_aux_fraction", "open_set"
        ),
        knownness_recall_floor=_finite_float(
            open_set_raw, "knownness_recall_floor", "open_set"
        ),
        quality_recall_floor=_finite_float(
            open_set_raw, "quality_recall_floor", "open_set"
        ),
        quality_min_final_negative_support=_positive_int(
            open_set_raw,
            "quality_min_final_negative_support",
            "open_set",
        ),
        conformal_error_rate=_finite_float(
            open_set_raw, "conformal_error_rate", "open_set"
        ),
        action_known_target_recall_floor=_finite_float(
            open_set_raw, "action_known_target_recall_floor", "open_set"
        ),
        action_clutter_recall_floor=_finite_float(
            open_set_raw, "action_clutter_recall_floor", "open_set"
        ),
        action_conformal_error_rate_grid_size=_positive_int(
            open_set_raw,
            "action_conformal_error_rate_grid_size",
            "open_set",
        ),
        action_knownness_threshold_grid_size=_positive_int(
            open_set_raw,
            "action_knownness_threshold_grid_size",
            "open_set",
        ),
        temperature_min=_finite_float(
            open_set_raw, "temperature_min", "open_set", positive=True
        ),
        temperature_max=_finite_float(
            open_set_raw, "temperature_max", "open_set", positive=True
        ),
    )
    fractions = (
        open_set.quality_min_rd_fraction,
        open_set.quality_min_dt_fraction,
        open_set.quality_min_aux_fraction,
        open_set.knownness_recall_floor,
        open_set.quality_recall_floor,
        open_set.conformal_error_rate,
        open_set.action_known_target_recall_floor,
        open_set.action_clutter_recall_floor,
    )
    if any(not 0.0 <= value <= 1.0 for value in fractions):
        raise ValueError("V27 open_set fractions must be within [0, 1]")
    if (
        open_set.knownness_recall_floor <= 0.0
        or open_set.quality_recall_floor <= 0.0
        or open_set.conformal_error_rate <= 0.0
        or open_set.conformal_error_rate >= 1.0
        or open_set.action_known_target_recall_floor <= 0.0
        or open_set.action_clutter_recall_floor <= 0.0
    ):
        raise ValueError("V27 open_set recall floors and conformal error must be strict")
    if (
        open_set.knownness_min_events > data.max_sequence_length
        or open_set.quality_min_events > data.max_sequence_length
        or open_set.quality_min_final_negative_support > 1_000_000
        or open_set.action_conformal_error_rate_grid_size > 1_000
        or open_set.action_knownness_threshold_grid_size > 1_000
        or open_set.head_weight_decay < 0.0
        or open_set.temperature_min > open_set.temperature_max
    ):
        raise ValueError("V27 open_set bounds differ")

    l3_raw = _mapping(payload, "l3", "root")
    l3 = V27L3Config(
        schema_version=_text(l3_raw, "schema_version", "l3"), calibration_status=_text(l3_raw, "calibration_status", "l3"), calibration_registry_path=_text(l3_raw, "calibration_registry_path", "l3"), dynamic_calibration_required=parse_bool(_required(l3_raw, "dynamic_calibration_required", "l3"), field="l3.dynamic_calibration_required"), fixed_softmax_threshold_allowed=parse_bool(_required(l3_raw, "fixed_softmax_threshold_allowed", "l3"), field="l3.fixed_softmax_threshold_allowed"), single_distance_gate_allowed=parse_bool(_required(l3_raw, "single_distance_gate_allowed", "l3"), field="l3.single_distance_gate_allowed"), quality_supervision_status=_text(l3_raw, "quality_supervision_status", "l3"), reason_codes=tuple(str(value) for value in _required(l3_raw, "reason_codes", "l3")),
    )
    # ``l3.py`` owns the resolver implementation.  Import lazily here because
    # the resolver imports the canonical task constants from this module.
    # Comparing the declared version at configuration load time prevents a
    # manifest from recording an action schema that the runtime cannot execute.
    from .l3 import L3_SCHEMA_VERSION

    if l3.schema_version != L3_SCHEMA_VERSION:
        raise ValueError("V27 L3 configuration schema differs from the resolver")
    quality_target_status = "deterministic_causal_facts_maturity_target_unfitted"
    if (
        model.quality_head_status != quality_target_status
        or l3.calibration_status != "unfitted"
        or not l3.dynamic_calibration_required
        or l3.fixed_softmax_threshold_allowed
        or l3.single_distance_gate_allowed
        or l3.quality_supervision_status != quality_target_status
    ):
        raise ValueError("V27 L3 must remain dynamically calibrated and fail-closed before fitting")

    artifact_raw = _mapping(payload, "artifacts", "root")
    artifacts = V27ArtifactConfig(
        **{
            key: _text(artifact_raw, key, "artifacts")
            for key in V27ArtifactConfig.__dataclass_fields__
        }
    )
    if "{task}" not in artifacts.representation_cache_template or "{role}" not in artifacts.representation_cache_template:
        raise ValueError("V27 representation cache template must be task/role scoped")
    if "{task}" not in artifacts.head_score_cache_template or "{role}" not in artifacts.head_score_cache_template:
        raise ValueError("V27 head score cache template must be task/role scoped")
    protocol = _mapping(payload, "protocol", "root")
    expected_protocol = {
        "runtime_data_boundary": "CQ08 processed index and versioned processed feature caches only; no independent raw dataset is opened at runtime",
        "l1_l2_parameters_and_recurrent_state": "strictly_independent",
        "quality_output_feedback": "forbidden",
        "open_set_action_owner": "L3_after_independent_dynamic_calibration",
        "oof": "not_executed",
        "cross_source": "not_executed",
        "deployment": "not_executed",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27 protocol.{key} differs")
    return V27Config(
        schema_version=V27_SCHEMA_VERSION,
        experiment_id=V27_EXPERIMENT_ID,
        direct_base_experiment=_text(payload, "direct_base_experiment", "root"),
        variant=_text(payload, "variant", "root"),
        objective=_text(payload, "objective", "root"),
        data=data,
        model=model,
        training=training,
        open_set=open_set,
        l3=l3,
        artifacts=artifacts,
        protocol=protocol,
        config_path=path,
        config_sha256=sha256_file(path),
        config_identity_sha256=canonical_sha256(payload),
    )


def task_class_names(task: str) -> tuple[str, ...]:
    if task == V27_L1_TASK_NAME:
        return V27_L1_CLASS_NAMES
    if task == V27_L2_TASK_NAME:
        return V27_L2_CLASS_NAMES
    raise ValueError(f"unsupported V27 task: {task}")


def task_allowed_labels(task: str) -> frozenset[str]:
    if task == V27_L1_TASK_NAME:
        return V27_L1_ALLOWED_LABELS
    if task == V27_L2_TASK_NAME:
        return V27_L2_ALLOWED_LABELS
    raise ValueError(f"unsupported V27 task: {task}")


def task_label_to_id(task: str) -> Mapping[str, int]:
    if task == V27_L1_TASK_NAME:
        return V27_L1_LABEL_TO_ID
    if task == V27_L2_TASK_NAME:
        return V27_L2_LABEL_TO_ID
    raise ValueError(f"unsupported V27 task: {task}")


def task_class_weights(config: V27Config, task: str) -> tuple[float, ...]:
    if task == V27_L1_TASK_NAME:
        return config.training.l1_class_weights
    if task == V27_L2_TASK_NAME:
        return config.training.l2_class_weights
    raise ValueError(f"unsupported V27 task: {task}")


def _relative(root: Path, value: str, *, must_exist: bool = True) -> Path:
    return resolve_project_path(root, value, must_exist=must_exist)


def _parent_details(index: pd.DataFrame, *, task: str, split_name: str) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    labels = task_allowed_labels(task)
    rows = index[index["split_name"].astype(str).eq(split_name) & index["label_name"].astype(str).isin(labels)].copy()
    mapping = task_label_to_id(task)
    if rows.empty:
        raise ValueError(f"V27 {task} has no {split_name} rows")
    parent_labels = rows.groupby("parent_sample_id", sort=True)["label_name"].nunique()
    if not bool(parent_labels.eq(1).all()):
        raise ValueError(f"V27 {task} parent crosses labels")
    task_labels = rows["label_name"].map(mapping)
    if task_labels.isna().any():
        raise ValueError(f"V27 {task} mapping has an unexpected label")
    counts = {name: int((task_labels == class_id).groupby(rows["parent_sample_id"]).first().sum()) for class_id, name in enumerate(task_class_names(task))}
    sample_ids = rows["sample_id"].astype(str).tolist()
    details = {
        "parent_count": int(rows["parent_sample_id"].nunique()),
        "event_count": int(len(rows)),
        "parent_label_counts": counts,
        "parent_ids_sha256": sequence_sha256(sorted(rows["parent_sample_id"].astype(str).unique())),
        "sample_ids_sha256": sequence_sha256(sample_ids),
    }
    return rows, details, sample_ids


def _verify_weights(config: V27Config, task: str, details: Mapping[str, Any]) -> None:
    counts = dict(details["parent_label_counts"])
    expected = tuple(sum(counts.values()) / (len(counts) * counts[name]) for name in task_class_names(task))
    actual = task_class_weights(config, task)
    if any(abs(left - right) > 1.0e-9 for left, right in zip(actual, expected, strict=True)):
        raise ValueError(f"V27 {task} class weights differ from train-only inverse parent frequencies")


def verify_v27_data_contract(project_root: str | Path, config: V27Config) -> dict[str, Any]:
    """Verify files and load only train/validation known-class feature payloads."""

    root = Path(project_root).resolve()
    paths = {
        "index": _relative(root, config.data.index_path),
        "l1_contract": _relative(root, config.data.l1.contract_path),
        "l1_normalization": _relative(root, config.data.l1.normalization_stats_path),
        "l1_feature_manifest": _relative(root, config.data.l1.feature_cache_manifest_path),
        "l1_train_course": _relative(root, config.data.l1.train_course_cache_path),
        "l1_validation_course": _relative(root, config.data.l1.validation_course_cache_path),
        "l2_contract": _relative(root, config.data.l2.contract_path),
        "l2_normalization": _relative(root, config.data.l2.normalization_stats_path),
        "l2_feature_source_normalization": _relative(root, config.data.l2.feature_source_normalization_stats_path),
        "l2_cache_manifest": _relative(root, config.data.l2.cache_manifest_path),
        "l2_train_physical": _relative(root, config.data.l2.train_rd_physical_cache_path),
        "l2_validation_physical": _relative(root, config.data.l2.validation_rd_physical_cache_path),
        "l2_train_micro": _relative(root, config.data.l2.train_micro_doppler_cache_path),
        "l2_validation_micro": _relative(root, config.data.l2.validation_micro_doppler_cache_path),
    }
    expected = {
        "index": config.data.expected_index_sha256,
        "l1_contract": config.data.l1.expected_contract_sha256,
        "l1_normalization": config.data.l1.expected_normalization_stats_sha256,
        "l1_feature_manifest": config.data.l1.expected_feature_cache_manifest_sha256,
        "l1_train_course": config.data.l1.expected_train_course_cache_sha256,
        "l1_validation_course": config.data.l1.expected_validation_course_cache_sha256,
        "l2_contract": config.data.l2.expected_contract_sha256,
        "l2_normalization": config.data.l2.expected_normalization_stats_sha256,
        "l2_feature_source_normalization": config.data.l2.expected_feature_source_normalization_stats_sha256,
        "l2_cache_manifest": config.data.l2.expected_cache_manifest_sha256,
        "l2_train_physical": config.data.l2.expected_train_rd_physical_cache_sha256,
        "l2_validation_physical": config.data.l2.expected_validation_rd_physical_cache_sha256,
        "l2_train_micro": config.data.l2.expected_train_micro_doppler_cache_sha256,
        "l2_validation_micro": config.data.l2.expected_validation_micro_doppler_cache_sha256,
    }
    hashes = {name: verify_file_hash(path, expected[name], field=f"V27 {name}") for name, path in paths.items()}
    index = pd.read_csv(paths["index"], low_memory=False)
    missing = sorted(REQUIRED_INDEX_COLUMNS - set(index.columns))
    if missing or set(index["schema_version"].astype(str)) != {config.data.index_schema_version} or set(index["source_name"].astype(str)) != {config.data.source_name} or index["sample_id"].astype(str).duplicated().any():
        raise ValueError("V27 index boundary/schema differs")
    parent_meta = index.groupby("parent_sample_id", sort=True).agg({"split_name": "nunique", "label_name": "nunique"})
    if not bool(parent_meta["split_name"].eq(1).all()) or not bool(parent_meta["label_name"].eq(1).all()):
        raise ValueError("V27 parent crosses split or label boundary")
    task_details: dict[str, dict[str, Any]] = {}
    task_rows: dict[tuple[str, str], pd.DataFrame] = {}
    task_samples: dict[tuple[str, str], list[str]] = {}
    expected_counts = {V27_L1_TASK_NAME: EXPECTED_L1_PARENT_COUNTS, V27_L2_TASK_NAME: EXPECTED_L2_PARENT_COUNTS}
    for task in V27_ALLOWED_TASKS:
        task_details[task] = {}
        for split_name in (config.data.train_split_name, config.data.validation_split_name):
            rows, details, sample_ids = _parent_details(index, task=task, split_name=split_name)
            if details["parent_label_counts"] != expected_counts[task][split_name]:
                raise ValueError(f"V27 {task}/{split_name} parent support differs")
            task_details[task][split_name] = details
            task_rows[(task, split_name)] = rows
            task_samples[(task, split_name)] = sample_ids
        _verify_weights(config, task, task_details[task][config.data.train_split_name])

    l1_normalization = json.loads(paths["l1_normalization"].read_text(encoding="utf-8"))
    if l1_normalization.get("source_index_sha256", "").lower() != hashes["index"] or set(l1_normalization.get("allowed_labels") or ()) != set(V27_L1_ALLOWED_LABELS) or set(l1_normalization.get("forbidden_labels") or ()) != {"UnknownTarget"}:
        raise ValueError("V27 L1 known-only normalization boundary differs")
    l2_normalization = json.loads(paths["l2_normalization"].read_text(encoding="utf-8"))
    if l2_normalization.get("schema_version") != V27_SCHEMA_VERSION or set(l2_normalization.get("allowed_labels") or ()) != set(V27_L2_ALLOWED_LABELS) or set(l2_normalization.get("forbidden_labels") or ()) != {"ClutterTarget", "UnknownTarget"} or l2_normalization.get("source_index_sha256", "").lower() != hashes["index"]:
        raise ValueError("V27 L2 DBB-only normalization boundary differs")
    if float(l1_normalization["amplitude"]["center"]) != config.data.l1.rd_amplitude_center or float(l1_normalization["amplitude"]["scale"]) != config.data.l1.rd_amplitude_scale or float(l2_normalization["amplitude"]["center"]) != config.data.l2.rd_amplitude_center or float(l2_normalization["amplitude"]["scale"]) != config.data.l2.rd_amplitude_scale:
        raise ValueError("V27 task normalization values differ")

    l1_contract = json.loads(paths["l1_contract"].read_text(encoding="utf-8"))
    l1_manifest = json.loads(paths["l1_feature_manifest"].read_text(encoding="utf-8"))
    if l1_contract.get("source_index_sha256", "").lower() != hashes["index"] or l1_manifest.get("source_index_sha256", "").lower() != hashes["index"] or l1_manifest.get("feature_names") != [config.data.l1.course_feature_name]:
        raise ValueError("V27 L1 feature cache provenance differs")
    for split_name, cache_key in ((config.data.train_split_name, "l1_train_course"), (config.data.validation_split_name, "l1_validation_course")):
        # The V21e selected cache is ordered by the historical parent ordering
        # (lexicographic string order), whereas the V16 CSV is not.  Loading
        # the full known-only cache and validating set equality avoids a false
        # order assumption; the dataset later aligns every value by sample_id.
        with np.load(paths[cache_key], allow_pickle=False) as archive:
            cache_ids = [str(value) for value in archive["sample_ids"].tolist()]
        values, valid, metadata = load_known_gate_feature_cache(paths[cache_key], expected_sample_ids=cache_ids, variant=V21E_COURSE_PREFIX_VARIANT, allow_superset=False)
        required_ids = set(task_samples[(V27_L1_TASK_NAME, split_name)])
        if not required_ids.issubset(values) or not required_ids.issubset(valid) or metadata["feature_names"] != [config.data.l1.course_feature_name]:
            raise ValueError("V27 L1 course cache boundary/alignment differs")

    l2_contract = json.loads(paths["l2_contract"].read_text(encoding="utf-8"))
    l2_manifest = json.loads(paths["l2_cache_manifest"].read_text(encoding="utf-8"))
    if l2_contract.get("source_index_sha256", "").lower() != hashes["index"] or l2_manifest.get("source_index_sha256", "").lower() != hashes["index"]:
        raise ValueError("V27 L2 feature provenance differs")
    for split_name, physical_key, micro_key in ((config.data.train_split_name, "l2_train_physical", "l2_train_micro"), (config.data.validation_split_name, "l2_validation_physical", "l2_validation_micro")):
        sample_ids = task_samples[(V27_L2_TASK_NAME, split_name)]
        load_rd_physical_cache(paths[physical_key], expected_sample_ids=sample_ids)
        load_micro_doppler_cache(paths[micro_key], expected_sample_ids=sample_ids, expected_value_dim=8)
    raw_l2_norm = json.loads(paths["l2_feature_source_normalization"].read_text(encoding="utf-8"))
    if raw_l2_norm.get("source_index_sha256", "").lower() != hashes["index"]:
        raise ValueError("V27 L2 feature-generation provenance differs")

    parent_split = sequence_sha256(
        f"{row.split_name}|{row.parent_sample_id}|{row.label_name}"
        for row in index[["split_name", "parent_sample_id", "label_name"]].drop_duplicates().sort_values(["split_name", "parent_sample_id"]).itertuples(index=False)
        if row.split_name in {config.data.train_split_name, config.data.validation_split_name}
    )
    registry = {
        "paths": {name: relative_project_path(root, path) for name, path in paths.items()},
        "hashes": hashes,
        "task_split_details": task_details,
        "parent_split_sha256": parent_split,
        "feature_config_sha256": canonical_sha256({"l1": {"course_feature": config.data.l1.course_feature_name, "cache_manifest": hashes["l1_feature_manifest"]}, "l2": {"contract": hashes["l2_contract"], "cache_manifest": hashes["l2_cache_manifest"]}}),
        "normalization_roles": {"l1": "known-only train (DBB plus Clutter; Unknown excluded)", "l2": "DBB-only train (Clutter and Unknown excluded)", "l2_feature_source": "historical feature-generation provenance only"},
        "unknown_payload_access": 0,
        "test_payload_access": 0,
        "independent_raw_dataset_runtime_access": 0,
    }
    registry["data_registry_sha256"] = canonical_sha256(registry)
    return registry

















__all__ = ["V27_ALLOWED_TASKS", "V27Config", "V27_EQUATION_VERSION", "V27_EXPERIMENT_ID", "V27_L1_CLASS_NAMES", "V27_L1_LABEL_TO_ID", "V27_L1_TASK_NAME", "V27_L2_CLASS_NAMES", "V27_L2_LABEL_TO_ID", "V27_L2_TASK_NAME", "V27_LOSS_NAME", "V27_LOSS_REDUCTION", "load_v27_config", "task_allowed_labels", "task_class_names", "task_class_weights", "task_label_to_id", "verify_v27_data_contract"]
