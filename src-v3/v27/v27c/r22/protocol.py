"""Immutable provenance and scope contracts for V27c-r22.

R22 is deliberately a *known-only L2 retraining* experiment.  It binds the
approved V27 L1 checkpoint by file and registry hash but never deserializes it
into a model.  The trainable path starts from V27 L2 epoch 11 and the V27a
Main stable8 branch, then compares a zero-motion control against one
pre-Liquid causal-motion treatment.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from radar_ln3.features import TR_MOTION_SCHEMA, resolve_causal_tr_feature_contract
from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)
from v27.v27_baseline.base_binding import classifier_state_hash, tensor_state_hash
from v27.v27_baseline.protocol import (
    V27_L2_CLASS_NAMES,
    load_v27_config,
    verify_v27_data_contract,
)
from v27.v27a.cache import load_cache_payload
from v27.v27a.protocol import load_v27a_config, read_sealed_json as read_v27a_sealed_json


R22_EXPERIMENT_ID = "V27c-r22"
R22_SCHEMA_VERSION = "RADAR-LN3-V27c-r22-integrated-l2-motion-v1"
R22_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27c-r22-manifest-v1"
R22_CHECKPOINT_SCHEMA_VERSION = "RADAR-LN3-V27c-r22-integrated-l2-checkpoint-v1"
R22_REPORT_SCHEMA_VERSION = "RADAR-LN3-V27c-r22-paired-validation-report-v1"
R22_SUMMARY_SCHEMA_VERSION = "RADAR-LN3-V27c-r22-three-seed-summary-v1"
R22_SMOKE_SCHEMA_VERSION = "RADAR-LN3-V27c-r22-contract-smoke-v1"

R22_ARMS = ("control", "motion_treatment")
R22_SPLITS = ("train", "val")
R22_L2_CLASSES = V27_L2_CLASS_NAMES
R22_FROZEN_ERROR_PARENT_IDS = (
    "cq08|track|1535",
    "cq08|track|1908",
    "cq08|track|2083",
    "cq08|track|2103",
)
R22_V27A_MAIN_CONFUSION = ((109, 4, 0), (1, 54, 1), (1, 1, 52))

_SHA256_HEX = frozenset("0123456789abcdef")
_R22_ROOT = Path("src-v3") / "v27" / "v27c" / "r22"
_R21_MOTION_TENSOR_NAMES = frozenset(
    {"frozen_endpoint_logits", "motion_values", "padding_mask", "lengths"}
)


@dataclass(frozen=True)
class R22BaseConfig:
    v27_config_path: str
    expected_v27_config_sha256: str
    frozen_binding_path: str
    expected_frozen_binding_file_sha256: str
    expected_frozen_binding_payload_sha256: str
    l1_checkpoint_path: str
    expected_l1_checkpoint_sha256: str
    expected_l1_state_sha256: str
    l2_checkpoint_path: str
    expected_l2_checkpoint_sha256: str
    expected_l2_state_sha256: str
    expected_l2_classifier_state_sha256: str
    v27a_config_path: str
    expected_v27a_config_sha256: str
    v27a_manifest_path: str
    expected_v27a_manifest_sha256: str
    expected_v27a_manifest_payload_sha256: str
    v27a_evidence_audit_path: str
    expected_v27a_evidence_audit_sha256: str
    v27a_main_checkpoint_path: str
    expected_v27a_main_checkpoint_sha256: str
    expected_v27a_main_state_sha256: str
    expected_v27a_main_model_contract_sha256: str
    v27a_train_cache_path: str
    expected_v27a_train_cache_sha256: str
    expected_v27a_train_cache_payload_sha256: str
    v27a_validation_cache_path: str
    expected_v27a_validation_cache_sha256: str
    expected_v27a_validation_cache_payload_sha256: str
    r21_motion_feature_manifest_path: str
    expected_r21_motion_feature_manifest_sha256: str
    expected_r21_motion_feature_manifest_payload_sha256: str
    r21_train_motion_cache_path: str
    expected_r21_train_motion_cache_sha256: str
    expected_r21_train_motion_cache_payload_sha256: str
    r21_validation_motion_cache_path: str
    expected_r21_validation_motion_cache_sha256: str
    expected_r21_validation_motion_cache_payload_sha256: str
    expected_l2_class_weights: tuple[float, ...]


@dataclass(frozen=True)
class R22DataConfig:
    index_path: str
    expected_index_sha256: str
    index_schema_version: str
    train_split_name: str
    validation_split_name: str
    max_sequence_length: int
    max_rd_frames_per_point: int
    rd_amplitude_center: float
    rd_amplitude_scale: float
    motion_schema: str
    motion_dim: int
    motion_clip_absolute: float
    motion_mask_mode: str
    stable8_dim: int
    train_parent_count: int
    validation_parent_count: int


@dataclass(frozen=True)
class R22ModelConfig:
    motion_projection_dim: int
    motion_adapter_dim: int
    class_count: int
    motion_projection_bias: bool
    motion_output_bias: bool
    motion_output_zero_initialization: bool
    stable_evidence_arm: str
    trajectory_heads: str


@dataclass(frozen=True)
class R22TrainingConfig:
    seeds: tuple[int, ...]
    dataloader_seed: int
    batch_size: int
    epochs: int
    primary_epoch: int
    base_learning_rate: float
    stable_learning_rate: float
    motion_learning_rate: float
    optimizer: str
    weight_decay: float
    gradient_clip_norm: float
    loss_name: str
    loss_reduction: str
    device: str
    cuda_deterministic_algorithms: bool
    eligible_prefix_min_events: int
    bootstrap_replicates: int


@dataclass(frozen=True)
class R22ArtifactConfig:
    manifest_path: str
    smoke_report_path: str
    training_log_template: str
    training_summary_template: str
    checkpoint_template: str
    evaluation_report_template: str
    summary_report_path: str


@dataclass(frozen=True)
class R22Config:
    schema_version: str
    experiment_id: str
    variant: str
    objective: str
    base: R22BaseConfig
    data: R22DataConfig
    model: R22ModelConfig
    training: R22TrainingConfig
    artifacts: R22ArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


@dataclass(frozen=True)
class R22Context:
    project_root: Path
    config: R22Config
    v27_config: Any
    v27a_config: Any
    input_binding: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27c-r22 {context} is missing {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27c-r22 {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27c-r22 {context}.{key} must be non-empty text")
    return value.strip()


def _sha(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(mapping, key, context).lower()
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"V27c-r22 {context}.{key} must be a SHA-256 digest")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27c-r22 {context}.{key} must be a positive integer")
    return int(value)


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
    *,
    positive: bool = False,
) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ValueError(f"V27c-r22 {context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27c-r22 {context}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27c-r22 {context}.{key} is outside its numeric bounds")
    return result


def _int_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[int, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27c-r22 {context}.{key} must be a non-empty integer list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError(f"V27c-r22 {context}.{key} must contain integers")
    values = tuple(int(value) for value in raw)
    if any(value < 0 for value in values) or len(values) != len(set(values)):
        raise ValueError(f"V27c-r22 {context}.{key} must be unique non-negative integers")
    return values


def _float_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[float, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27c-r22 {context}.{key} must be a non-empty numeric list")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"V27c-r22 {context}.{key} must contain finite positive values")
    return values


def _bool(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    value = _required(mapping, key, context)
    if not isinstance(value, bool):
        raise ValueError(f"V27c-r22 {context}.{key} must be boolean")
    return value


def _path(root: Path, value: str, *, must_exist: bool = True) -> Path:
    return resolve_project_path(root, value, must_exist=must_exist)


def _artifact_path(
    root: Path,
    template: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
) -> Path:
    if "{arm}" in template and arm not in R22_ARMS:
        raise ValueError("V27c-r22 artifact requires a registered arm")
    if "{seed}" in template and (
        seed is None or isinstance(seed, bool) or int(seed) < 0
    ):
        raise ValueError("V27c-r22 artifact requires a non-negative seed")
    if "{epoch" in template and (
        epoch is None or isinstance(epoch, bool) or int(epoch) <= 0
    ):
        raise ValueError("V27c-r22 artifact requires a positive epoch")
    try:
        rendered = template.format(arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V27c-r22 artifact template is invalid") from exc
    target = resolve_project_path(root, rendered)
    scope = (root / _R22_ROOT).resolve()
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27c-r22 artifacts must remain inside its experiment root") from exc
    return target


def r22_artifact_path(
    project_root: str | Path,
    template: str,
    *,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
) -> Path:
    return _artifact_path(
        Path(project_root).resolve(), template, arm=arm, seed=seed, epoch=epoch
    )


def load_r22_config(
    project_root: str | Path, config_path: str | Path | None = None
) -> R22Config:
    root = Path(project_root).resolve()
    candidate = (
        root / _R22_ROOT / "config" / "V27c-r22-integrated-l2-motion.json"
        if config_path is None
        else Path(config_path)
    )
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("V27c-r22 configuration must remain inside the project root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r22 configuration root must be an object")
    if payload.get("schema_version") != R22_SCHEMA_VERSION:
        raise ValueError("V27c-r22 configuration schema differs")
    if payload.get("experiment_id") != R22_EXPERIMENT_ID:
        raise ValueError("V27c-r22 experiment id differs")

    base_raw = _mapping(payload, "base", "root")
    base = R22BaseConfig(
        v27_config_path=_text(base_raw, "v27_config_path", "base"),
        expected_v27_config_sha256=_sha(base_raw, "expected_v27_config_sha256", "base"),
        frozen_binding_path=_text(base_raw, "frozen_binding_path", "base"),
        expected_frozen_binding_file_sha256=_sha(base_raw, "expected_frozen_binding_file_sha256", "base"),
        expected_frozen_binding_payload_sha256=_sha(base_raw, "expected_frozen_binding_payload_sha256", "base"),
        l1_checkpoint_path=_text(base_raw, "l1_checkpoint_path", "base"),
        expected_l1_checkpoint_sha256=_sha(base_raw, "expected_l1_checkpoint_sha256", "base"),
        expected_l1_state_sha256=_sha(base_raw, "expected_l1_state_sha256", "base"),
        l2_checkpoint_path=_text(base_raw, "l2_checkpoint_path", "base"),
        expected_l2_checkpoint_sha256=_sha(base_raw, "expected_l2_checkpoint_sha256", "base"),
        expected_l2_state_sha256=_sha(base_raw, "expected_l2_state_sha256", "base"),
        expected_l2_classifier_state_sha256=_sha(base_raw, "expected_l2_classifier_state_sha256", "base"),
        v27a_config_path=_text(base_raw, "v27a_config_path", "base"),
        expected_v27a_config_sha256=_sha(base_raw, "expected_v27a_config_sha256", "base"),
        v27a_manifest_path=_text(base_raw, "v27a_manifest_path", "base"),
        expected_v27a_manifest_sha256=_sha(base_raw, "expected_v27a_manifest_sha256", "base"),
        expected_v27a_manifest_payload_sha256=_sha(base_raw, "expected_v27a_manifest_payload_sha256", "base"),
        v27a_evidence_audit_path=_text(base_raw, "v27a_evidence_audit_path", "base"),
        expected_v27a_evidence_audit_sha256=_sha(base_raw, "expected_v27a_evidence_audit_sha256", "base"),
        v27a_main_checkpoint_path=_text(base_raw, "v27a_main_checkpoint_path", "base"),
        expected_v27a_main_checkpoint_sha256=_sha(base_raw, "expected_v27a_main_checkpoint_sha256", "base"),
        expected_v27a_main_state_sha256=_sha(base_raw, "expected_v27a_main_state_sha256", "base"),
        expected_v27a_main_model_contract_sha256=_sha(base_raw, "expected_v27a_main_model_contract_sha256", "base"),
        v27a_train_cache_path=_text(base_raw, "v27a_train_cache_path", "base"),
        expected_v27a_train_cache_sha256=_sha(base_raw, "expected_v27a_train_cache_sha256", "base"),
        expected_v27a_train_cache_payload_sha256=_sha(base_raw, "expected_v27a_train_cache_payload_sha256", "base"),
        v27a_validation_cache_path=_text(base_raw, "v27a_validation_cache_path", "base"),
        expected_v27a_validation_cache_sha256=_sha(base_raw, "expected_v27a_validation_cache_sha256", "base"),
        expected_v27a_validation_cache_payload_sha256=_sha(base_raw, "expected_v27a_validation_cache_payload_sha256", "base"),
        r21_motion_feature_manifest_path=_text(base_raw, "r21_motion_feature_manifest_path", "base"),
        expected_r21_motion_feature_manifest_sha256=_sha(base_raw, "expected_r21_motion_feature_manifest_sha256", "base"),
        expected_r21_motion_feature_manifest_payload_sha256=_sha(base_raw, "expected_r21_motion_feature_manifest_payload_sha256", "base"),
        r21_train_motion_cache_path=_text(base_raw, "r21_train_motion_cache_path", "base"),
        expected_r21_train_motion_cache_sha256=_sha(base_raw, "expected_r21_train_motion_cache_sha256", "base"),
        expected_r21_train_motion_cache_payload_sha256=_sha(base_raw, "expected_r21_train_motion_cache_payload_sha256", "base"),
        r21_validation_motion_cache_path=_text(base_raw, "r21_validation_motion_cache_path", "base"),
        expected_r21_validation_motion_cache_sha256=_sha(base_raw, "expected_r21_validation_motion_cache_sha256", "base"),
        expected_r21_validation_motion_cache_payload_sha256=_sha(base_raw, "expected_r21_validation_motion_cache_payload_sha256", "base"),
        expected_l2_class_weights=_float_tuple(base_raw, "expected_l2_class_weights", "base"),
    )
    if len(base.expected_l2_class_weights) != len(R22_L2_CLASSES):
        raise ValueError("V27c-r22 L2 class-weight width differs")

    data_raw = _mapping(payload, "data", "root")
    motion_contract = resolve_causal_tr_feature_contract(
        _text(data_raw, "motion_schema", "data")
    )
    data = R22DataConfig(
        index_path=_text(data_raw, "index_path", "data"),
        expected_index_sha256=_sha(data_raw, "expected_index_sha256", "data"),
        index_schema_version=_text(data_raw, "index_schema_version", "data"),
        train_split_name=_text(data_raw, "train_split_name", "data"),
        validation_split_name=_text(data_raw, "validation_split_name", "data"),
        max_sequence_length=_positive_int(data_raw, "max_sequence_length", "data"),
        max_rd_frames_per_point=_positive_int(data_raw, "max_rd_frames_per_point", "data"),
        rd_amplitude_center=_finite_float(data_raw, "rd_amplitude_center", "data"),
        rd_amplitude_scale=_finite_float(data_raw, "rd_amplitude_scale", "data", positive=True),
        motion_schema=motion_contract.schema,
        motion_dim=_positive_int(data_raw, "motion_dim", "data"),
        motion_clip_absolute=_finite_float(data_raw, "motion_clip_absolute", "data", positive=True),
        motion_mask_mode=_text(data_raw, "motion_mask_mode", "data"),
        stable8_dim=_positive_int(data_raw, "stable8_dim", "data"),
        train_parent_count=_positive_int(data_raw, "train_parent_count", "data"),
        validation_parent_count=_positive_int(data_raw, "validation_parent_count", "data"),
    )
    if (
        data.motion_schema != TR_MOTION_SCHEMA
        or data.motion_dim != motion_contract.value_dim
        or not math.isclose(
            data.motion_clip_absolute,
            motion_contract.clip_absolute,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or data.motion_mask_mode != "numerical_gating_only_not_model_input"
        or data.stable8_dim != 8
        or (data.train_split_name, data.validation_split_name) != ("train", "val")
        or (data.max_sequence_length, data.max_rd_frames_per_point) != (29, 12)
        or (data.train_parent_count, data.validation_parent_count) != (953, 223)
    ):
        raise ValueError("V27c-r22 data contract differs from the approved design")

    model_raw = _mapping(payload, "model", "root")
    model = R22ModelConfig(
        motion_projection_dim=_positive_int(model_raw, "motion_projection_dim", "model"),
        motion_adapter_dim=_positive_int(model_raw, "motion_adapter_dim", "model"),
        class_count=_positive_int(model_raw, "class_count", "model"),
        motion_projection_bias=_bool(model_raw, "motion_projection_bias", "model"),
        motion_output_bias=_bool(model_raw, "motion_output_bias", "model"),
        motion_output_zero_initialization=_bool(model_raw, "motion_output_zero_initialization", "model"),
        stable_evidence_arm=_text(model_raw, "stable_evidence_arm", "model"),
        trajectory_heads=_text(model_raw, "trajectory_heads", "model"),
    )
    if (
        (model.motion_projection_dim, model.motion_adapter_dim, model.class_count)
        != (16, 16, len(R22_L2_CLASSES))
        or model.motion_projection_bias
        or model.motion_output_bias
        or not model.motion_output_zero_initialization
        or model.stable_evidence_arm != "main"
        or model.trajectory_heads != "frozen_unused_by_classification"
    ):
        raise ValueError("V27c-r22 model contract differs from the approved design")

    training_raw = _mapping(payload, "training", "root")
    training = R22TrainingConfig(
        seeds=_int_tuple(training_raw, "seeds", "training"),
        dataloader_seed=_positive_int(training_raw, "dataloader_seed", "training"),
        batch_size=_positive_int(training_raw, "batch_size", "training"),
        epochs=_positive_int(training_raw, "epochs", "training"),
        primary_epoch=_positive_int(training_raw, "primary_epoch", "training"),
        base_learning_rate=_finite_float(training_raw, "base_learning_rate", "training", positive=True),
        stable_learning_rate=_finite_float(training_raw, "stable_learning_rate", "training", positive=True),
        motion_learning_rate=_finite_float(training_raw, "motion_learning_rate", "training", positive=True),
        optimizer=_text(training_raw, "optimizer", "training"),
        weight_decay=_finite_float(training_raw, "weight_decay", "training"),
        gradient_clip_norm=_finite_float(training_raw, "gradient_clip_norm", "training", positive=True),
        loss_name=_text(training_raw, "loss_name", "training"),
        loss_reduction=_text(training_raw, "loss_reduction", "training"),
        device=_text(training_raw, "device", "training").lower(),
        cuda_deterministic_algorithms=_bool(training_raw, "cuda_deterministic_algorithms", "training"),
        eligible_prefix_min_events=_positive_int(training_raw, "eligible_prefix_min_events", "training"),
        bootstrap_replicates=_positive_int(training_raw, "bootstrap_replicates", "training"),
    )
    if (
        training.seeds != (42, 43, 44)
        or training.batch_size != 2
        or training.epochs != 20
        or training.primary_epoch != 20
        or training.optimizer != "AdamW"
        or training.loss_name != "parent_final_fixed_parent_count_ce"
        or training.loss_reduction != "fixed_parent_count"
        or training.device != "cuda"
        or not training.cuda_deterministic_algorithms
        or training.weight_decay < 0.0
        or training.bootstrap_replicates < 100
    ):
        raise ValueError("V27c-r22 training contract differs from the approved design")

    artifacts_raw = _mapping(payload, "artifacts", "root")
    artifacts = R22ArtifactConfig(
        **{
            name: _text(artifacts_raw, name, "artifacts")
            for name in R22ArtifactConfig.__dataclass_fields__
        }
    )
    for template in artifacts.__dict__.values():
        _artifact_path(root, template, arm="control", seed=42, epoch=1)

    protocol = _mapping(payload, "protocol", "root")
    expected_protocol = {
        "l1": "frozen_checkpoint_bound_not_loaded_or_optimized",
        "l2": "retrain_base_classifier_and_v27a_main_stable8_residual",
        "arms": list(R22_ARMS),
        "only_trainable_arm_difference": "motion_treatment_receives_gated_causal_tr_motion_numeric_values_before_liquid_recurrence",
        "control_motion_input": "all_zero_after_the_same_numeric_gating_boundary",
        "motion_mask": "tr_motion_valid_mask_used_once_for_numeric_gating_then_not_serialized_as_model_input",
        "stable8": "reuse_v27a_main_full_reliable_frame_path_and_retrain_it_with_l2",
        "unknown_training_exposure": 0,
        "clutter_training_exposure": 0,
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "calibration": "not_in_scope",
        "l3": "not_in_scope",
        "test": "sealed_and_not_used",
        "checkpoint_selection": "fixed_primary_epoch_without_validation_best_selection",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27c-r22 protocol.{key} differs")

    return R22Config(
        schema_version=R22_SCHEMA_VERSION,
        experiment_id=R22_EXPERIMENT_ID,
        variant=_text(payload, "variant", "root"),
        objective=_text(payload, "objective", "root"),
        base=base,
        data=data,
        model=model,
        training=training,
        artifacts=artifacts,
        protocol=protocol,
        config_path=path,
        config_sha256=sha256_file(path),
        config_identity_sha256=canonical_sha256(payload),
    )


def _read_json(path: Path, *, description: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"V27c-r22 {description} must be a JSON object")
    return dict(payload)


def _verify_sealed_payload(
    payload: Mapping[str, Any], *, expected: str, description: str
) -> None:
    declared = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if declared != canonical_sha256(body) or declared != expected:
        raise ValueError(f"V27c-r22 {description} payload hash differs")


def _verify_frozen_binding(root: Path, config: R22Config) -> dict[str, Any]:
    """Bind L1 by bytes and registry metadata without deserializing it.

    The L1 file is deliberately never passed to ``torch.load`` in R22.  The
    frozen binding already records its state digest; R22 checks that record
    and the file digest, then only deserializes the L2 checkpoint it retrains.
    """

    path = _path(root, config.base.frozen_binding_path)
    if sha256_file(path) != config.base.expected_frozen_binding_file_sha256:
        raise ValueError("V27c-r22 frozen-base binding file hash differs")
    payload = _read_json(path, description="frozen-base binding")
    declared = payload.get("binding_sha256")
    body = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if declared != canonical_sha256(body) or declared != config.base.expected_frozen_binding_payload_sha256:
        raise ValueError("V27c-r22 frozen-base binding payload hash differs")
    if payload.get("status") != "frozen" or payload.get("experiment_id") != "V27":
        raise ValueError("V27c-r22 frozen-base binding schema/status differs")

    l1 = payload.get("l1")
    l2 = payload.get("l2")
    if not isinstance(l1, Mapping) or not isinstance(l2, Mapping):
        raise ValueError("V27c-r22 frozen-base binding lacks L1 or L2")
    expected_l1_path = relative_project_path(root, _path(root, config.base.l1_checkpoint_path))
    expected_l2_path = relative_project_path(root, _path(root, config.base.l2_checkpoint_path))
    if (
        l1.get("task") != "known_gate"
        or int(l1.get("epoch", -1)) != 14
        or l1.get("checkpoint_path") != expected_l1_path
        or l1.get("checkpoint_sha256") != config.base.expected_l1_checkpoint_sha256
        or l1.get("model_state_sha256") != config.base.expected_l1_state_sha256
    ):
        raise ValueError("V27c-r22 frozen L1 binding differs")
    if (
        l2.get("task") != "dbb_specialist"
        or int(l2.get("epoch", -1)) != 11
        or l2.get("checkpoint_path") != expected_l2_path
        or l2.get("checkpoint_sha256") != config.base.expected_l2_checkpoint_sha256
        or l2.get("model_state_sha256") != config.base.expected_l2_state_sha256
        or l2.get("classifier_state_sha256")
        != config.base.expected_l2_classifier_state_sha256
        or list(l2.get("class_weights") or ())
        != list(config.base.expected_l2_class_weights)
    ):
        raise ValueError("V27c-r22 frozen L2 binding differs")
    # Hashing the L1 artifact verifies the exact approved file while keeping
    # it outside all R22 model construction and checkpoint deserialization.
    if sha256_file(_path(root, config.base.l1_checkpoint_path)) != config.base.expected_l1_checkpoint_sha256:
        raise ValueError("V27c-r22 frozen L1 checkpoint file hash differs")
    return dict(payload)


def _load_l2_checkpoint(root: Path, config: R22Config) -> dict[str, Any]:
    path = _path(root, config.base.l2_checkpoint_path)
    if sha256_file(path) != config.base.expected_l2_checkpoint_sha256:
        raise ValueError("V27c-r22 frozen L2 checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r22 frozen L2 checkpoint is not an object")
    checked = dict(payload)
    state = checked.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27c-r22 frozen L2 checkpoint lacks model state")
    if (
        checked.get("checkpoint_schema") != "RADAR-LN3-V27-liquid-v2-task-checkpoint-v2"
        or checked.get("experiment_id") != "V27"
        or checked.get("task") != "dbb_specialist"
        or checked.get("checkpoint_kind") != "epoch"
        or int(checked.get("epoch", -1)) != 11
        or int(checked.get("seed", -1)) != 42
        or checked.get("model_state_sha256") != config.base.expected_l2_state_sha256
        or tensor_state_hash(state) != config.base.expected_l2_state_sha256
        or classifier_state_hash(state) != config.base.expected_l2_classifier_state_sha256
        or checked.get("class_names") != list(R22_L2_CLASSES)
        or list(checked.get("class_weights") or ())
        != list(config.base.expected_l2_class_weights)
        or checked.get("unknown_is_model_class") is not False
        or int(checked.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V27c-r22 frozen L2 checkpoint contract differs")
    # Existing V27 trajectory-head facts are allowed to evolve independently
    # after the frozen classifier was created.  The r22 model will therefore
    # partial-load only the classifier state and requires the exact head key
    # boundary in its builder; this check keeps that fact visible here.
    return checked


def _r21_tensor_digest(tensors: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27c-r22 r21 sidecar contains a non-tensor: {name}")
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(name).encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(repr(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def r21_motion_cache_payload_sha256(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "cache_sha256"}
    tensors = body.get("tensors")
    if not isinstance(tensors, Mapping):
        raise ValueError("V27c-r22 r21 sidecar lacks tensors")
    metadata = {key: value for key, value in body.items() if key != "tensors"}
    return canonical_sha256(
        {"metadata": metadata, "tensor_sha256": _r21_tensor_digest(tensors)}
    )


def _motion_cache_path(config: R22Config, split: str) -> str:
    if split == "train":
        return config.base.r21_train_motion_cache_path
    if split == "val":
        return config.base.r21_validation_motion_cache_path
    raise ValueError("V27c-r22 motion cache split is invalid")


def _motion_file_hash(config: R22Config, split: str) -> str:
    if split == "train":
        return config.base.expected_r21_train_motion_cache_sha256
    if split == "val":
        return config.base.expected_r21_validation_motion_cache_sha256
    raise ValueError("V27c-r22 motion cache split is invalid")


def _motion_payload_hash(config: R22Config, split: str) -> str:
    if split == "train":
        return config.base.expected_r21_train_motion_cache_payload_sha256
    if split == "val":
        return config.base.expected_r21_validation_motion_cache_payload_sha256
    raise ValueError("V27c-r22 motion cache split is invalid")


def _v27a_cache_path(config: R22Config, split: str) -> str:
    if split == "train":
        return config.base.v27a_train_cache_path
    if split == "val":
        return config.base.v27a_validation_cache_path
    raise ValueError("V27c-r22 V27a cache split is invalid")


def _v27a_cache_file_hash(config: R22Config, split: str) -> str:
    if split == "train":
        return config.base.expected_v27a_train_cache_sha256
    if split == "val":
        return config.base.expected_v27a_validation_cache_sha256
    raise ValueError("V27c-r22 V27a cache split is invalid")


def _v27a_cache_payload_hash(config: R22Config, split: str) -> str:
    if split == "train":
        return config.base.expected_v27a_train_cache_payload_sha256
    if split == "val":
        return config.base.expected_v27a_validation_cache_payload_sha256
    raise ValueError("V27c-r22 V27a cache split is invalid")


def load_verified_v27a_cache(context: R22Context, *, split: str) -> dict[str, Any]:
    if split not in R22_SPLITS:
        raise ValueError("V27c-r22 V27a cache split is invalid")
    path = _path(context.project_root, _v27a_cache_path(context.config, split))
    if sha256_file(path) != _v27a_cache_file_hash(context.config, split):
        raise ValueError("V27c-r22 V27a cache file hash differs")
    payload = load_cache_payload(path)
    if payload.get("cache_payload_sha256") != _v27a_cache_payload_hash(
        context.config, split
    ):
        raise ValueError("V27c-r22 V27a cache payload hash differs")
    parents = [str(value) for value in payload["parent_sample_ids"]]
    labels = [str(value) for value in payload["parent_labels"]]
    expected_count = (
        context.config.data.train_parent_count
        if split == "train"
        else context.config.data.validation_parent_count
    )
    if (
        len(parents) != expected_count
        or len(set(parents)) != len(parents)
        or len(labels) != expected_count
        or any(label not in R22_L2_CLASSES for label in labels)
        or payload["event_baseline_logits"].shape
        != (expected_count, context.config.data.max_sequence_length, len(R22_L2_CLASSES))
        or payload["event_summary_tokens"].shape
        != (expected_count, context.config.data.max_sequence_length, context.config.data.stable8_dim)
        or payload["event_frame_values"].shape[:2]
        != (expected_count, context.config.data.max_sequence_length)
        or payload["event_frame_values"].shape[-1] != context.config.data.stable8_dim
    ):
        raise ValueError("V27c-r22 V27a cache population or stable8 shape differs")
    return payload


def load_verified_r21_motion_cache(context: R22Context, *, split: str) -> dict[str, Any]:
    if split not in R22_SPLITS:
        raise ValueError("V27c-r22 r21 motion cache split is invalid")
    path = _path(context.project_root, _motion_cache_path(context.config, split))
    if sha256_file(path) != _motion_file_hash(context.config, split):
        raise ValueError("V27c-r22 r21 motion cache file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r22 r21 motion cache is not an object")
    checked = dict(payload)
    if checked.get("cache_sha256") != _motion_payload_hash(context.config, split):
        raise ValueError("V27c-r22 r21 motion cache declared payload hash differs")
    if r21_motion_cache_payload_sha256(checked) != checked["cache_sha256"]:
        raise ValueError("V27c-r22 r21 motion cache content hash differs")
    tensors = checked.get("tensors")
    if not isinstance(tensors, Mapping) or set(tensors) != _R21_MOTION_TENSOR_NAMES:
        raise ValueError("V27c-r22 r21 motion cache tensor fields differ")
    if "tr_motion_valid_mask" in checked or "tr_motion_valid_mask" in tensors:
        raise ValueError("V27c-r22 r21 validity mask must not enter the r22 model")
    expected_count = (
        context.config.data.train_parent_count
        if split == "train"
        else context.config.data.validation_parent_count
    )
    values = tensors["motion_values"]
    padding = tensors["padding_mask"]
    lengths = tensors["lengths"]
    frozen_endpoint = tensors["frozen_endpoint_logits"]
    if (
        checked.get("schema_version") != "RADAR-LN3-V27c-motion-feature-cache-v1"
        or checked.get("experiment_id") != "V27c"
        or checked.get("split") != split
        or checked.get("motion_schema") != context.config.data.motion_schema
        or checked.get("motion_names")
        != list(
            resolve_causal_tr_feature_contract(
                context.config.data.motion_schema
            ).value_names
        )
        or not math.isclose(
            float(checked.get("motion_clip_absolute", float("nan"))),
            context.config.data.motion_clip_absolute,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or checked.get("motion_mask_mode") != "numerical_gating_only_not_model_input"
        or not isinstance(values, torch.Tensor)
        or values.shape
        != (
            expected_count,
            context.config.data.max_sequence_length,
            context.config.data.motion_dim,
        )
        or values.dtype != torch.float32
        or not isinstance(padding, torch.Tensor)
        or padding.shape != (expected_count, context.config.data.max_sequence_length)
        or padding.dtype != torch.bool
        or not isinstance(lengths, torch.Tensor)
        or lengths.shape != (expected_count,)
        or lengths.dtype != torch.long
        or not isinstance(frozen_endpoint, torch.Tensor)
        or frozen_endpoint.shape
        != (
            expected_count,
            context.config.data.max_sequence_length,
            len(R22_L2_CLASSES),
        )
        or frozen_endpoint.dtype != torch.float32
        or not torch.isfinite(values).all()
        or not torch.isfinite(frozen_endpoint).all()
        or bool((values.abs() > context.config.data.motion_clip_absolute + 1.0e-6).any())
        or bool(values.masked_select(padding.unsqueeze(-1)).ne(0).any())
        or bool(values[:, 0].ne(0).any())
        or bool(frozen_endpoint.masked_select(padding.unsqueeze(-1)).ne(0).any())
        or bool((lengths <= 0).any())
        or bool((lengths > context.config.data.max_sequence_length).any())
    ):
        raise ValueError("V27c-r22 r21 motion cache shape/value contract differs")
    expected_padding = torch.arange(
        context.config.data.max_sequence_length
    ).unsqueeze(0) >= lengths.unsqueeze(1)
    if not torch.equal(padding.cpu(), expected_padding):
        raise ValueError("V27c-r22 r21 motion cache padding differs from lengths")
    if split == "val":
        rows = torch.arange(expected_count)
        prediction = frozen_endpoint[rows, lengths - 1].argmax(dim=-1)
        truth = torch.tensor(
            [R22_L2_CLASSES.index(str(label)) for label in checked["label_names"]],
            dtype=torch.long,
        )
        matrix = torch.zeros(
            (len(R22_L2_CLASSES), len(R22_L2_CLASSES)), dtype=torch.long
        )
        matrix.index_put_((truth, prediction), torch.ones_like(truth), accumulate=True)
        if matrix.tolist() != [list(row) for row in R22_V27A_MAIN_CONFUSION]:
            raise ValueError("V27c-r22 r21 V27a Main audit endpoint differs")
    return checked


def _sequence_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _verify_r21_alignment(
    *,
    config: R22Config,
    split: str,
    v27a_cache: Mapping[str, Any],
    motion_cache: Mapping[str, Any],
) -> dict[str, Any]:
    """Check r21's numeric sidecar against the stable8 cache exactly."""

    parents = [str(value) for value in v27a_cache["parent_sample_ids"]]
    labels = [str(value) for value in v27a_cache["parent_labels"]]
    motion_parents = [str(value) for value in motion_cache.get("parent_ids") or ()]
    motion_labels = [str(value) for value in motion_cache.get("label_names") or ()]
    tensors = motion_cache["tensors"]
    if (
        parents != motion_parents
        or labels != motion_labels
        or not np.array_equal(v27a_cache["lengths"], tensors["lengths"].numpy())
        or not np.array_equal(v27a_cache["padding_mask"], tensors["padding_mask"].numpy())
    ):
        raise ValueError("V27c-r22 r21 motion/V27a stable8 parent alignment differs")
    # The r21 sidecar was materialized from the public processed index, so its
    # own event-order digest must remain valid.  The direct dataset crosscheck
    # is performed again in r22.data before any training batch is built.
    expected_event_parent_order = [
        parent
        for parent, length in zip(parents, v27a_cache["lengths"], strict=True)
        for _ in range(int(length))
    ]
    if expected_event_parent_order != [str(value) for value in v27a_cache["event_parent_ids"]]:
        raise ValueError("V27c-r22 V27a event parent ordering differs")
    declared = motion_cache.get("event_ids_order_sha256")
    if not isinstance(declared, str) or len(declared) != 64:
        raise ValueError("V27c-r22 r21 motion event-order digest is malformed")
    return {
        "parent_count": len(parents),
        "parent_ids_sha256": canonical_sha256(parents),
        "label_names_sha256": canonical_sha256(labels),
        "lengths_sha256": canonical_sha256(v27a_cache["lengths"].astype(np.int64).tolist()),
        "padding_sha256": canonical_sha256(v27a_cache["padding_mask"].astype(np.uint8).tolist()),
        "r21_event_ids_order_sha256": declared,
        "v27a_event_parent_order_sha256": _sequence_digest(expected_event_parent_order),
    }


def _verify_v27a_main_checkpoint(
    root: Path, config: R22Config, v27a_config: Any
) -> dict[str, Any]:
    path = _path(root, config.base.v27a_main_checkpoint_path)
    if sha256_file(path) != config.base.expected_v27a_main_checkpoint_sha256:
        raise ValueError("V27c-r22 V27a Main checkpoint file hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r22 V27a Main checkpoint is not an object")
    checked = dict(payload)
    state = checked.get("model_state")
    if not isinstance(state, Mapping):
        raise ValueError("V27c-r22 V27a Main checkpoint lacks model state")
    from v27.v27a.checkpoint import tensor_state_hash as v27a_state_hash
    from v27.v27a.model import V27AEvidenceResidual

    model = V27AEvidenceResidual(v27a_config, arm="main")
    contract = model.model_contract()
    if (
        checked.get("checkpoint_schema") != "RADAR-LN3-V27a-late-residual-checkpoint-v2"
        or checked.get("experiment_id") != "V27a"
        or checked.get("checkpoint_kind") != "epoch"
        or checked.get("arm") != "main"
        or int(checked.get("seed", -1)) != 42
        or int(checked.get("epoch", -1)) != 20
        or checked.get("model_state_sha256") != config.base.expected_v27a_main_state_sha256
        or v27a_state_hash(state) != config.base.expected_v27a_main_state_sha256
        or checked.get("model_contract_sha256")
        != config.base.expected_v27a_main_model_contract_sha256
        or canonical_sha256(contract) != config.base.expected_v27a_main_model_contract_sha256
        or checked.get("model_contract") != contract
        or checked.get("base_checkpoint_sha256") != config.base.expected_l2_checkpoint_sha256
        or checked.get("base_classifier_state_sha256")
        != config.base.expected_l2_classifier_state_sha256
        or checked.get("class_names") != list(R22_L2_CLASSES)
    ):
        raise ValueError("V27c-r22 V27a Main checkpoint contract differs")
    return checked


def _source_paths() -> tuple[Path, ...]:
    return (
        Path("src-v3/v27/v27c/r22/__init__.py"),
        Path("src-v3/v27/v27c/r22/protocol.py"),
        Path("src-v3/v27/v27c/r22/data.py"),
        Path("src-v3/v27/v27c/r22/model.py"),
        Path("src-v3/v27/v27c/r22/training.py"),
        Path("src-v3/v27/v27c/r22/evaluation.py"),
        Path("src-v3/v27/v27c/r22/smoke.py"),
        Path("src-v3/v27/v27c/r22/cli.py"),
        Path("src-v3/v27/v27_baseline/base_binding.py"),
        Path("src-v3/v27/v27_baseline/data.py"),
        Path("src-v3/v27/v27_baseline/logging_utils.py"),
        Path("src-v3/v27/v27_baseline/metrics.py"),
        Path("src-v3/v27/v27_baseline/model.py"),
        Path("src-v3/v27/v27_baseline/protocol.py"),
        Path("src-v3/v27/v27_baseline/runtime.py"),
        Path("src-v3/v27/v27a/cache.py"),
        Path("src-v3/v27/v27a/checkpoint.py"),
        Path("src-v3/v27/v27a/model.py"),
        Path("src-v3/v27/v27a/protocol.py"),
        Path("src-v3/radar_ln3/data.py"),
        Path("src-v3/radar_ln3/features.py"),
        Path("src-v3/radar_ln3/liquid_v2.py"),
        Path("src-v3/radar_ln3/losses.py"),
        Path("src-v3/radar_ln3/runtime.py"),
    )


def verify_r22_inputs(
    project_root: str | Path, config: R22Config
) -> tuple[Any, Any, dict[str, Any]]:
    """Verify all frozen inputs without opening L1 as a model checkpoint."""

    root = Path(project_root).resolve()
    v27_path = _path(root, config.base.v27_config_path)
    if sha256_file(v27_path) != config.base.expected_v27_config_sha256:
        raise ValueError("V27c-r22 V27 config file hash differs")
    v27_config = load_v27_config(root, v27_path)
    if v27_config.config_sha256 != config.base.expected_v27_config_sha256:
        raise ValueError("V27c-r22 V27 config identity differs")
    if (
        v27_config.data.index_path != config.data.index_path
        or v27_config.data.index_schema_version != config.data.index_schema_version
        or v27_config.data.max_sequence_length != config.data.max_sequence_length
        or v27_config.data.max_rd_frames_per_point != config.data.max_rd_frames_per_point
        or not math.isclose(
            v27_config.data.l2.rd_amplitude_center,
            config.data.rd_amplitude_center,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or not math.isclose(
            v27_config.data.l2.rd_amplitude_scale,
            config.data.rd_amplitude_scale,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("V27c-r22 V27 L2 data shape/normalization contract differs")
    data_binding = verify_v27_data_contract(root, v27_config)
    if data_binding["hashes"]["index"] != config.data.expected_index_sha256:
        raise ValueError("V27c-r22 processed index hash differs")
    frozen_binding = _verify_frozen_binding(root, config)
    l2_checkpoint = _load_l2_checkpoint(root, config)

    v27a_path = _path(root, config.base.v27a_config_path)
    if sha256_file(v27a_path) != config.base.expected_v27a_config_sha256:
        raise ValueError("V27c-r22 V27a config file hash differs")
    v27a_config = load_v27a_config(root, v27a_path)
    if v27a_config.config_sha256 != config.base.expected_v27a_config_sha256:
        raise ValueError("V27c-r22 V27a config identity differs")
    manifest_path = _path(root, config.base.v27a_manifest_path)
    if sha256_file(manifest_path) != config.base.expected_v27a_manifest_sha256:
        raise ValueError("V27c-r22 V27a manifest file hash differs")
    v27a_manifest = read_v27a_sealed_json(manifest_path)
    _verify_sealed_payload(
        v27a_manifest,
        expected=config.base.expected_v27a_manifest_payload_sha256,
        description="V27a manifest",
    )
    audit_path = _path(root, config.base.v27a_evidence_audit_path)
    if sha256_file(audit_path) != config.base.expected_v27a_evidence_audit_sha256:
        raise ValueError("V27c-r22 V27a evidence audit file hash differs")
    v27a_main = _verify_v27a_main_checkpoint(root, config, v27a_config)

    r21_manifest_path = _path(root, config.base.r21_motion_feature_manifest_path)
    if sha256_file(r21_manifest_path) != config.base.expected_r21_motion_feature_manifest_sha256:
        raise ValueError("V27c-r22 r21 motion manifest file hash differs")
    r21_manifest = _read_json(r21_manifest_path, description="r21 motion manifest")
    _verify_sealed_payload(
        r21_manifest,
        expected=config.base.expected_r21_motion_feature_manifest_payload_sha256,
        description="r21 motion manifest",
    )
    if (
        r21_manifest.get("experiment_id") != "V27c"
        or r21_manifest.get("unknown_training_exposure") != 0
        or r21_manifest.get("test_payload_access_count") != 0
    ):
        raise ValueError("V27c-r22 r21 motion manifest scope differs")

    # Build a provisional context solely for the public cache loaders.  The
    # loaders use only frozen path/hash fields, not the r22 manifest.
    provisional = R22Context(
        project_root=root,
        config=config,
        v27_config=v27_config,
        v27a_config=v27a_config,
        input_binding={},
        manifest={},
        manifest_path=Path(),
        manifest_sha256="",
    )
    cache_bindings: dict[str, Any] = {}
    for split in R22_SPLITS:
        stable_cache = load_verified_v27a_cache(provisional, split=split)
        motion_cache = load_verified_r21_motion_cache(provisional, split=split)
        alignment = _verify_r21_alignment(
            config=config,
            split=split,
            v27a_cache=stable_cache,
            motion_cache=motion_cache,
        )
        declared = r21_manifest.get("caches", {}).get(split, {})
        if (
            not isinstance(declared, Mapping)
            or declared.get("cache_sha256") != motion_cache["cache_sha256"]
            or int(declared.get("parent_count", -1)) != alignment["parent_count"]
        ):
            raise ValueError("V27c-r22 r21 manifest/cache binding differs")
        cache_bindings[split] = {
            "v27a_cache_path": relative_project_path(
                root, _path(root, _v27a_cache_path(config, split))
            ),
            "v27a_cache_file_sha256": _v27a_cache_file_hash(config, split),
            "v27a_cache_payload_sha256": stable_cache["cache_payload_sha256"],
            "r21_motion_cache_path": relative_project_path(
                root, _path(root, _motion_cache_path(config, split))
            ),
            "r21_motion_cache_file_sha256": _motion_file_hash(config, split),
            "r21_motion_cache_payload_sha256": motion_cache["cache_sha256"],
            **alignment,
        }

    binding = {
        "v27": {
            "config_path": relative_project_path(root, v27_path),
            "config_sha256": v27_config.config_sha256,
            "data_registry_sha256": data_binding["data_registry_sha256"],
            "l2_classifier_state_sha256": config.base.expected_l2_classifier_state_sha256,
        },
        "l1_binding_only": {
            "checkpoint_path": relative_project_path(
                root, _path(root, config.base.l1_checkpoint_path)
            ),
            "checkpoint_sha256": config.base.expected_l1_checkpoint_sha256,
            "state_sha256_from_frozen_binding": config.base.expected_l1_state_sha256,
            "checkpoint_deserialized": False,
            "model_loaded": False,
            "forward_calls": 0,
            "optimizer_parameter_count": 0,
        },
        "l2_initialization": {
            "checkpoint_path": relative_project_path(
                root, _path(root, config.base.l2_checkpoint_path)
            ),
            "checkpoint_sha256": config.base.expected_l2_checkpoint_sha256,
            "model_state_sha256": l2_checkpoint["model_state_sha256"],
            "classifier_state_sha256": config.base.expected_l2_classifier_state_sha256,
            "epoch": 11,
            "class_weights": list(config.base.expected_l2_class_weights),
        },
        "v27a_main": {
            "config_path": relative_project_path(root, v27a_path),
            "config_sha256": v27a_config.config_sha256,
            "manifest_path": relative_project_path(root, manifest_path),
            "manifest_file_sha256": config.base.expected_v27a_manifest_sha256,
            "manifest_payload_sha256": config.base.expected_v27a_manifest_payload_sha256,
            "evidence_audit_sha256": config.base.expected_v27a_evidence_audit_sha256,
            "checkpoint_path": relative_project_path(
                root, _path(root, config.base.v27a_main_checkpoint_path)
            ),
            "checkpoint_sha256": config.base.expected_v27a_main_checkpoint_sha256,
            "state_sha256": v27a_main["model_state_sha256"],
            "model_contract_sha256": v27a_main["model_contract_sha256"],
            "arm": "main",
            "epoch": 20,
        },
        "r21_motion": {
            "feature_manifest_path": relative_project_path(root, r21_manifest_path),
            "feature_manifest_file_sha256": config.base.expected_r21_motion_feature_manifest_sha256,
            "feature_manifest_payload_sha256": config.base.expected_r21_motion_feature_manifest_payload_sha256,
            "numeric_gating_only": True,
            "validity_mask_model_input": False,
            "caches": cache_bindings,
        },
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "test_payload_access_count": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
        },
        "frozen_binding_payload_sha256": frozen_binding["binding_sha256"],
    }
    return v27_config, v27a_config, binding


def build_r22_manifest(
    project_root: str | Path,
    config: R22Config,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    hashes: dict[str, str] = {}
    for path in _source_paths():
        target = root / path
        if not target.is_file():
            raise FileNotFoundError(f"V27c-r22 source file is missing: {target}")
        hashes[path.as_posix()] = sha256_file(target)
    return {
        "schema_version": R22_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R22_EXPERIMENT_ID,
        "variant": config.variant,
        "objective": config.objective,
        "config": {
            "path": relative_project_path(root, config.config_path),
            "sha256": config.config_sha256,
            "identity_sha256": config.config_identity_sha256,
        },
        "input_binding": dict(input_binding),
        "code_hashes": hashes,
        "model": {
            "base": "V27 L2 epoch-11 classifier/recurrent state only",
            "stable8": "V27a Main epoch-20 full reliable-frame residual",
            "motion": "bias-free 9-to-16-to-48 adapter after fusion before Liquid V2",
            "initial_logit_equation": "cached_v27_base + (current_l2 - frozen_l2_reference) + v27a_main_stable8_residual",
            "trajectory_heads": "frozen_and_unused",
        },
        "training": {
            "arms": list(R22_ARMS),
            "seeds": list(config.training.seeds),
            "fixed_primary_epoch": config.training.primary_epoch,
            "checkpoint_selection": "fixed_primary_epoch_without_validation_best_selection",
            "class_weights": list(config.base.expected_l2_class_weights),
        },
        "protocol": dict(config.protocol),
        "scope": dict(input_binding["scope"]),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    if "payload_sha256" in body:
        raise ValueError("V27c-r22 payload is already sealed")
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_seal(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V27c-r22 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically replace a mutable diagnostic artifact with a sealed body.

    R22's manifest, checkpoints, reports, and summaries are immutable.  The
    pre-training smoke diagnostic is intentionally mutable until the first
    formal checkpoint exists, because it is specifically used while finishing
    the executable contract.  Keeping that one exception here avoids treating
    a preliminary smoke as experimental evidence.
    """

    target = Path(path)
    atomic_write_text(target, json_text(_seal(payload), indent=2))
    return target


def read_sealed_json(path: str | Path, *, schema: str | None = None) -> dict[str, Any]:
    payload = _read_json(Path(path), description="sealed artifact")
    declared = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if declared != canonical_sha256(body):
        raise ValueError("V27c-r22 sealed artifact payload hash differs")
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError("V27c-r22 sealed artifact schema differs")
    return payload


def prepare_r22_manifest(
    project_root: str | Path, config_path: str | Path | None = None
) -> tuple[R22Config, dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_r22_config(root, config_path)
    _, _, input_binding = verify_r22_inputs(root, config)
    payload = build_r22_manifest(root, config, input_binding)
    path = r22_artifact_path(root, config.artifacts.manifest_path)
    # A manifest is immutable once a smoke/train/evaluation artifact exists.
    # Before that boundary, source completion may legitimately change its code
    # digest; replacing only the manifest is safe and avoids a stale-prepared
    # file preventing the first smoke from inspecting the completed runtime.
    if path.exists():
        existing = read_sealed_json(path, schema=R22_MANIFEST_SCHEMA_VERSION)
        existing_body = {key: value for key, value in existing.items() if key != "payload_sha256"}
        if existing_body != payload:
            protected = [
                *[
                    r22_artifact_path(
                        root,
                        config.artifacts.training_summary_template,
                        arm=arm,
                        seed=seed,
                    )
                    for arm in R22_ARMS
                    for seed in config.training.seeds
                ],
                *[
                    r22_artifact_path(
                        root,
                        config.artifacts.evaluation_report_template,
                        seed=seed,
                    )
                    for seed in config.training.seeds
                ],
            ]
            if any(candidate.exists() for candidate in protected):
                raise ValueError(
                    "V27c-r22 manifest is stale after a derived artifact exists; "
                    "create a new experiment revision instead"
                )
            atomic_write_text(path, json_text(_seal(payload), indent=2))
        return config, input_binding, path, sha256_file(path)
    write_sealed_json_if_identical(path, payload)
    return config, input_binding, path, sha256_file(path)


def load_r22_context(
    project_root: str | Path, config_path: str | Path | None = None
) -> R22Context:
    root = Path(project_root).resolve()
    config = load_r22_config(root, config_path)
    v27_config, v27a_config, input_binding = verify_r22_inputs(root, config)
    path = r22_artifact_path(root, config.artifacts.manifest_path)
    manifest = read_sealed_json(path, schema=R22_MANIFEST_SCHEMA_VERSION)
    expected = build_r22_manifest(root, config, input_binding)
    actual_body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual_body != expected:
        raise ValueError("V27c-r22 manifest differs from the current code, config, or inputs")
    return R22Context(
        project_root=root,
        config=config,
        v27_config=v27_config,
        v27a_config=v27a_config,
        input_binding=input_binding,
        manifest=manifest,
        manifest_path=path,
        manifest_sha256=sha256_file(path),
    )


__all__ = [
    "R22_ARMS",
    "R22_CHECKPOINT_SCHEMA_VERSION",
    "R22_EXPERIMENT_ID",
    "R22_L2_CLASSES",
    "R22_MANIFEST_SCHEMA_VERSION",
    "R22_REPORT_SCHEMA_VERSION",
    "R22_SCHEMA_VERSION",
    "R22_SMOKE_SCHEMA_VERSION",
    "R22_SUMMARY_SCHEMA_VERSION",
    "R22Context",
    "R22Config",
    "load_r22_config",
    "load_r22_context",
    "load_verified_r21_motion_cache",
    "load_verified_v27a_cache",
    "prepare_r22_manifest",
    "r21_motion_cache_payload_sha256",
    "r22_artifact_path",
    "read_sealed_json",
    "verify_r22_inputs",
    "write_sealed_json_if_identical",
    "write_sealed_json",
]
