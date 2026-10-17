"""Immutable contracts for the V27c-r23 TR-clipping factorial experiment."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from v27.v27a.protocol import (
    load_v27a_config,
    read_sealed_json as read_v27a_sealed_json,
)

from .preprocessing import clipping_audit, fit_tr_clipping


R23_EXPERIMENT_ID = "V27c-r23"
R23_SCHEMA_VERSION = "RADAR-LN3-V27c-r23-tr-clipping-causal-motion-v1"
R23_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27c-r23-manifest-v1"
R23_CHECKPOINT_SCHEMA_VERSION = "RADAR-LN3-V27c-r23-factorial-l2-checkpoint-v1"
R23_REPORT_SCHEMA_VERSION = "RADAR-LN3-V27c-r23-factorial-validation-report-v1"
R23_SUMMARY_SCHEMA_VERSION = "RADAR-LN3-V27c-r23-two-seed-summary-v1"
R23_SMOKE_SCHEMA_VERSION = "RADAR-LN3-V27c-r23-contract-smoke-v1"

R23_ARMS = (
    "legacy_control",
    "legacy_motion",
    "robust_control",
    "robust_motion",
)
R23_SPLITS = ("train", "val")
R23_L2_CLASSES = V27_L2_CLASS_NAMES
R23_ARM_SPECS = {
    "legacy_control": {"input_mode": "legacy", "motion_mode": "zero"},
    "legacy_motion": {"input_mode": "legacy", "motion_mode": "causal"},
    "robust_control": {"input_mode": "robust", "motion_mode": "zero"},
    "robust_motion": {"input_mode": "robust", "motion_mode": "causal"},
}
R23_V27A_MAIN_CONFUSION = ((109, 4, 0), (1, 54, 1), (1, 1, 52))

_SHA256_HEX = frozenset("0123456789abcdef")
_R23_ROOT = Path("src-v3") / "v27" / "v27c" / "r23"


@dataclass(frozen=True)
class R23BaseConfig:
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
    expected_l2_class_weights: tuple[float, ...]


@dataclass(frozen=True)
class R23DataConfig:
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
    tr_clipping_quantile: float
    tr_range_lower_m: float
    tr_speed_lower_mps: float
    tr_clipping_fit_split: str
    stable8_dim: int
    train_parent_count: int
    validation_parent_count: int


@dataclass(frozen=True)
class R23ModelConfig:
    motion_projection_dim: int
    motion_adapter_dim: int
    class_count: int
    motion_projection_bias: bool
    motion_output_bias: bool
    motion_output_zero_initialization: bool
    stable_evidence_arm: str
    trajectory_heads: str


@dataclass(frozen=True)
class R23TrainingConfig:
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
class R23ArtifactConfig:
    manifest_path: str
    smoke_report_path: str
    training_log_template: str
    training_summary_template: str
    checkpoint_template: str
    evaluation_report_template: str
    summary_report_path: str


@dataclass(frozen=True)
class R23Config:
    schema_version: str
    experiment_id: str
    variant: str
    objective: str
    base: R23BaseConfig
    data: R23DataConfig
    model: R23ModelConfig
    training: R23TrainingConfig
    artifacts: R23ArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


@dataclass(frozen=True)
class R23Context:
    project_root: Path
    config: R23Config
    v27_config: Any
    v27a_config: Any
    input_binding: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27c-r23 {context} is missing {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27c-r23 {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27c-r23 {context}.{key} must be non-empty text")
    return value.strip()


def _sha(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(mapping, key, context).lower()
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"V27c-r23 {context}.{key} must be a SHA-256 digest")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27c-r23 {context}.{key} must be a positive integer")
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
        raise ValueError(f"V27c-r23 {context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27c-r23 {context}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27c-r23 {context}.{key} is outside its numeric bounds")
    return result


def _int_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[int, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27c-r23 {context}.{key} must be a non-empty integer list")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise ValueError(f"V27c-r23 {context}.{key} must contain integers")
    values = tuple(int(value) for value in raw)
    if any(value < 0 for value in values) or len(values) != len(set(values)):
        raise ValueError(f"V27c-r23 {context}.{key} must be unique non-negative integers")
    return values


def _float_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[float, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27c-r23 {context}.{key} must be a non-empty numeric list")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"V27c-r23 {context}.{key} must contain finite positive values")
    return values


def _bool(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    value = _required(mapping, key, context)
    if not isinstance(value, bool):
        raise ValueError(f"V27c-r23 {context}.{key} must be boolean")
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
    if "{arm}" in template and arm not in R23_ARMS:
        raise ValueError("V27c-r23 artifact requires a registered arm")
    if "{seed}" in template and (
        seed is None or isinstance(seed, bool) or int(seed) < 0
    ):
        raise ValueError("V27c-r23 artifact requires a non-negative seed")
    if "{epoch" in template and (
        epoch is None or isinstance(epoch, bool) or int(epoch) <= 0
    ):
        raise ValueError("V27c-r23 artifact requires a positive epoch")
    try:
        rendered = template.format(arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V27c-r23 artifact template is invalid") from exc
    target = resolve_project_path(root, rendered)
    scope = (root / _R23_ROOT).resolve()
    try:
        target.relative_to(scope)
    except ValueError as exc:
        raise ValueError("V27c-r23 artifacts must remain inside its experiment root") from exc
    return target


def r23_artifact_path(
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


def load_r23_config(
    project_root: str | Path, config_path: str | Path | None = None
) -> R23Config:
    root = Path(project_root).resolve()
    candidate = (
        root / _R23_ROOT / "config" / "V27c-r23-tr-clipping-causal-motion.json"
        if config_path is None
        else Path(config_path)
    )
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("V27c-r23 configuration must remain inside the project root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27c-r23 configuration root must be an object")
    if payload.get("schema_version") != R23_SCHEMA_VERSION:
        raise ValueError("V27c-r23 configuration schema differs")
    if payload.get("experiment_id") != R23_EXPERIMENT_ID:
        raise ValueError("V27c-r23 experiment id differs")

    base_raw = _mapping(payload, "base", "root")
    base_fields = {
        name: _sha(base_raw, name, "base")
        if "sha256" in name or name.endswith("_sha256")
        else _text(base_raw, name, "base")
        for name in R23BaseConfig.__dataclass_fields__
        if name != "expected_l2_class_weights"
    }
    base = R23BaseConfig(
        **base_fields,
        expected_l2_class_weights=_float_tuple(
            base_raw, "expected_l2_class_weights", "base"
        ),
    )
    if len(base.expected_l2_class_weights) != len(R23_L2_CLASSES):
        raise ValueError("V27c-r23 L2 class-weight width differs")

    data_raw = _mapping(payload, "data", "root")
    motion_contract = resolve_causal_tr_feature_contract(
        _text(data_raw, "motion_schema", "data")
    )
    data = R23DataConfig(
        index_path=_text(data_raw, "index_path", "data"),
        expected_index_sha256=_sha(data_raw, "expected_index_sha256", "data"),
        index_schema_version=_text(data_raw, "index_schema_version", "data"),
        train_split_name=_text(data_raw, "train_split_name", "data"),
        validation_split_name=_text(data_raw, "validation_split_name", "data"),
        max_sequence_length=_positive_int(data_raw, "max_sequence_length", "data"),
        max_rd_frames_per_point=_positive_int(data_raw, "max_rd_frames_per_point", "data"),
        rd_amplitude_center=_finite_float(data_raw, "rd_amplitude_center", "data"),
        rd_amplitude_scale=_finite_float(
            data_raw, "rd_amplitude_scale", "data", positive=True
        ),
        motion_schema=motion_contract.schema,
        motion_dim=_positive_int(data_raw, "motion_dim", "data"),
        motion_clip_absolute=_finite_float(
            data_raw, "motion_clip_absolute", "data", positive=True
        ),
        motion_mask_mode=_text(data_raw, "motion_mask_mode", "data"),
        tr_clipping_quantile=_finite_float(
            data_raw, "tr_clipping_quantile", "data"
        ),
        tr_range_lower_m=_finite_float(data_raw, "tr_range_lower_m", "data"),
        tr_speed_lower_mps=_finite_float(data_raw, "tr_speed_lower_mps", "data"),
        tr_clipping_fit_split=_text(data_raw, "tr_clipping_fit_split", "data"),
        stable8_dim=_positive_int(data_raw, "stable8_dim", "data"),
        train_parent_count=_positive_int(data_raw, "train_parent_count", "data"),
        validation_parent_count=_positive_int(
            data_raw, "validation_parent_count", "data"
        ),
    )
    if (
        data.motion_schema != TR_MOTION_SCHEMA
        or data.motion_dim != motion_contract.value_dim
        or not math.isclose(data.motion_clip_absolute, motion_contract.clip_absolute)
        or data.motion_mask_mode != "numerical_gating_only_not_model_input"
        or not 0.5 < data.tr_clipping_quantile < 1.0
        or data.tr_range_lower_m < 0.0
        or data.tr_speed_lower_mps < 0.0
        or data.tr_clipping_fit_split != "train"
        or data.stable8_dim != 8
        or (data.train_split_name, data.validation_split_name) != ("train", "val")
        or (data.max_sequence_length, data.max_rd_frames_per_point) != (29, 12)
        or (data.train_parent_count, data.validation_parent_count) != (953, 223)
    ):
        raise ValueError("V27c-r23 data contract differs from the approved design")

    model_raw = _mapping(payload, "model", "root")
    model = R23ModelConfig(
        motion_projection_dim=_positive_int(model_raw, "motion_projection_dim", "model"),
        motion_adapter_dim=_positive_int(model_raw, "motion_adapter_dim", "model"),
        class_count=_positive_int(model_raw, "class_count", "model"),
        motion_projection_bias=_bool(model_raw, "motion_projection_bias", "model"),
        motion_output_bias=_bool(model_raw, "motion_output_bias", "model"),
        motion_output_zero_initialization=_bool(
            model_raw, "motion_output_zero_initialization", "model"
        ),
        stable_evidence_arm=_text(model_raw, "stable_evidence_arm", "model"),
        trajectory_heads=_text(model_raw, "trajectory_heads", "model"),
    )
    if (
        model.class_count != len(R23_L2_CLASSES)
        or model.motion_projection_bias
        or model.motion_output_bias
        or not model.motion_output_zero_initialization
        or model.stable_evidence_arm != "main"
        or model.trajectory_heads != "frozen_unused_by_classification"
    ):
        raise ValueError("V27c-r23 model contract differs from the approved design")

    training_raw = _mapping(payload, "training", "root")
    training = R23TrainingConfig(
        seeds=_int_tuple(training_raw, "seeds", "training"),
        dataloader_seed=_positive_int(training_raw, "dataloader_seed", "training"),
        batch_size=_positive_int(training_raw, "batch_size", "training"),
        epochs=_positive_int(training_raw, "epochs", "training"),
        primary_epoch=_positive_int(training_raw, "primary_epoch", "training"),
        base_learning_rate=_finite_float(
            training_raw, "base_learning_rate", "training", positive=True
        ),
        stable_learning_rate=_finite_float(
            training_raw, "stable_learning_rate", "training", positive=True
        ),
        motion_learning_rate=_finite_float(
            training_raw, "motion_learning_rate", "training", positive=True
        ),
        optimizer=_text(training_raw, "optimizer", "training"),
        weight_decay=_finite_float(training_raw, "weight_decay", "training"),
        gradient_clip_norm=_finite_float(
            training_raw, "gradient_clip_norm", "training", positive=True
        ),
        loss_name=_text(training_raw, "loss_name", "training"),
        loss_reduction=_text(training_raw, "loss_reduction", "training"),
        device=_text(training_raw, "device", "training").lower(),
        cuda_deterministic_algorithms=_bool(
            training_raw, "cuda_deterministic_algorithms", "training"
        ),
        eligible_prefix_min_events=_positive_int(
            training_raw, "eligible_prefix_min_events", "training"
        ),
        bootstrap_replicates=_positive_int(
            training_raw, "bootstrap_replicates", "training"
        ),
    )
    if (
        training.seeds != (42, 43)
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
        raise ValueError("V27c-r23 training contract differs from the approved design")

    artifacts_raw = _mapping(payload, "artifacts", "root")
    artifacts = R23ArtifactConfig(
        **{
            name: _text(artifacts_raw, name, "artifacts")
            for name in R23ArtifactConfig.__dataclass_fields__
        }
    )
    for template in artifacts.__dict__.values():
        _artifact_path(root, template, arm=R23_ARMS[0], seed=42, epoch=1)

    protocol = _mapping(payload, "protocol", "root")
    checkpoint_selection = protocol.get(
        "checkpoint_selection", "fixed_primary_epoch_without_validation_best_selection"
    )
    if checkpoint_selection not in {
        "fixed_primary_epoch_without_validation_best_selection",
        "global_validation_per_class_f1",
    }:
        raise ValueError("V27c-r23 checkpoint selection policy is unsupported")
    expected_protocol = {
        "arms": list(R23_ARMS),
        "factorial_design": "2x2_input_clipping_by_causal_motion",
        "legacy_input": "unclipped_tr_features_and_motion_from_unclipped_raw_values",
        "robust_input": "train_only_fitted_range_speed_clipping_then_reencoded_tr_and_motion",
        "reference_input": "legacy_unclipped_v27_l2_state_for_all_arms",
        "motion_mask": "causal_validity_used_for_numeric_zeroing_only_not_model_input",
        "stable8": "reuse_v27a_main_full_reliable_frame_path_and_retrain_it_with_l2",
        "unknown_training_exposure": 0,
        "clutter_training_exposure": 0,
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "calibration": "not_in_scope",
        "l3": "not_in_scope",
        "test": "sealed_and_not_used",
        "checkpoint_selection": checkpoint_selection,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27c-r23 protocol.{key} differs")

    return R23Config(
        schema_version=R23_SCHEMA_VERSION,
        experiment_id=R23_EXPERIMENT_ID,
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
        raise ValueError(f"V27c-r23 {description} must be an object")
    return dict(payload)


def _verify_sealed_payload(
    payload: Mapping[str, Any], *, expected: str, description: str
) -> None:
    declared = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if declared != canonical_sha256(body) or declared != expected:
        raise ValueError(f"V27c-r23 {description} payload hash differs")


def _verify_frozen_binding(root: Path, config: R23Config) -> dict[str, Any]:
    path = _path(root, config.base.frozen_binding_path)
    if sha256_file(path) != config.base.expected_frozen_binding_file_sha256:
        raise ValueError("V27c-r23 frozen binding file hash differs")
    payload = _read_json(path, description="frozen binding")
    body = {key: value for key, value in payload.items() if key != "binding_sha256"}
    if (
        payload.get("binding_sha256") != canonical_sha256(body)
        or payload.get("binding_sha256") != config.base.expected_frozen_binding_payload_sha256
        or payload.get("status") != "frozen"
        or payload.get("experiment_id") != "V27"
    ):
        raise ValueError("V27c-r23 frozen binding payload differs")
    l1, l2 = payload.get("l1"), payload.get("l2")
    if not isinstance(l1, Mapping) or not isinstance(l2, Mapping):
        raise ValueError("V27c-r23 frozen binding lacks L1 or L2")
    expected_l1 = relative_project_path(root, _path(root, config.base.l1_checkpoint_path))
    expected_l2 = relative_project_path(root, _path(root, config.base.l2_checkpoint_path))
    if (
        l1.get("task") != "known_gate"
        or int(l1.get("epoch", -1)) != 14
        or l1.get("checkpoint_path") != expected_l1
        or l1.get("checkpoint_sha256") != config.base.expected_l1_checkpoint_sha256
        or l1.get("model_state_sha256") != config.base.expected_l1_state_sha256
        or l2.get("task") != "dbb_specialist"
        or int(l2.get("epoch", -1)) != 11
        or l2.get("checkpoint_path") != expected_l2
        or l2.get("checkpoint_sha256") != config.base.expected_l2_checkpoint_sha256
        or l2.get("model_state_sha256") != config.base.expected_l2_state_sha256
        or l2.get("classifier_state_sha256")
        != config.base.expected_l2_classifier_state_sha256
        or list(l2.get("class_weights") or ())
        != list(config.base.expected_l2_class_weights)
    ):
        raise ValueError("V27c-r23 frozen classification binding differs")
    if sha256_file(_path(root, config.base.l1_checkpoint_path)) != config.base.expected_l1_checkpoint_sha256:
        raise ValueError("V27c-r23 frozen L1 checkpoint hash differs")
    return payload


def _load_l2_checkpoint(root: Path, config: R23Config) -> dict[str, Any]:
    path = _path(root, config.base.l2_checkpoint_path)
    if sha256_file(path) != config.base.expected_l2_checkpoint_sha256:
        raise ValueError("V27c-r23 initial L2 checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("V27c-r23 initial L2 checkpoint is invalid")
    state = payload["model_state"]
    if (
        payload.get("checkpoint_schema") != "RADAR-LN3-V27-liquid-v2-task-checkpoint-v2"
        or payload.get("experiment_id") != "V27"
        or payload.get("task") != "dbb_specialist"
        or int(payload.get("epoch", -1)) != 11
        or int(payload.get("seed", -1)) != 42
        or payload.get("model_state_sha256") != config.base.expected_l2_state_sha256
        or tensor_state_hash(state) != config.base.expected_l2_state_sha256
        or classifier_state_hash(state) != config.base.expected_l2_classifier_state_sha256
        or payload.get("class_names") != list(R23_L2_CLASSES)
        or list(payload.get("class_weights") or ()) != list(config.base.expected_l2_class_weights)
        or payload.get("unknown_is_model_class") is not False
        or int(payload.get("unknown_training_exposure", -1)) != 0
    ):
        raise ValueError("V27c-r23 initial L2 checkpoint contract differs")
    return dict(payload)


def _cache_path(config: R23Config, split: str) -> str:
    if split == "train":
        return config.base.v27a_train_cache_path
    if split == "val":
        return config.base.v27a_validation_cache_path
    raise ValueError("V27c-r23 cache split is invalid")


def _cache_file_hash(config: R23Config, split: str) -> str:
    if split == "train":
        return config.base.expected_v27a_train_cache_sha256
    if split == "val":
        return config.base.expected_v27a_validation_cache_sha256
    raise ValueError("V27c-r23 cache split is invalid")


def _cache_payload_hash(config: R23Config, split: str) -> str:
    if split == "train":
        return config.base.expected_v27a_train_cache_payload_sha256
    if split == "val":
        return config.base.expected_v27a_validation_cache_payload_sha256
    raise ValueError("V27c-r23 cache split is invalid")


def load_verified_v27a_cache(context: R23Context, *, split: str) -> dict[str, Any]:
    if split not in R23_SPLITS:
        raise ValueError("V27c-r23 V27a cache split is invalid")
    path = _path(context.project_root, _cache_path(context.config, split))
    if sha256_file(path) != _cache_file_hash(context.config, split):
        raise ValueError("V27c-r23 V27a cache file hash differs")
    payload = load_cache_payload(path)
    if payload.get("cache_payload_sha256") != _cache_payload_hash(context.config, split):
        raise ValueError("V27c-r23 V27a cache payload hash differs")
    expected_count = (
        context.config.data.train_parent_count
        if split == "train"
        else context.config.data.validation_parent_count
    )
    parents = [str(value) for value in payload.get("parent_sample_ids", ())]
    labels = [str(value) for value in payload.get("parent_labels", ())]
    if (
        len(parents) != expected_count
        or len(set(parents)) != expected_count
        or len(labels) != expected_count
        or any(label not in R23_L2_CLASSES for label in labels)
        or payload["event_baseline_logits"].shape
        != (expected_count, context.config.data.max_sequence_length, len(R23_L2_CLASSES))
        or payload["event_summary_tokens"].shape
        != (expected_count, context.config.data.max_sequence_length, context.config.data.stable8_dim)
        or payload["event_frame_values"].shape[:2]
        != (expected_count, context.config.data.max_sequence_length)
        or payload["event_frame_values"].shape[-1] != context.config.data.stable8_dim
    ):
        raise ValueError("V27c-r23 V27a cache population or shape differs")
    return payload


def _verify_v27a_main_checkpoint(
    root: Path, config: R23Config, v27a_config: Any
) -> dict[str, Any]:
    path = _path(root, config.base.v27a_main_checkpoint_path)
    if sha256_file(path) != config.base.expected_v27a_main_checkpoint_sha256:
        raise ValueError("V27c-r23 V27a Main checkpoint hash differs")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model_state"), Mapping):
        raise ValueError("V27c-r23 V27a Main checkpoint is invalid")
    from v27.v27a.checkpoint import tensor_state_hash as v27a_state_hash
    from v27.v27a.model import V27AEvidenceResidual

    model = V27AEvidenceResidual(v27a_config, arm="main")
    contract = model.model_contract()
    if (
        payload.get("checkpoint_schema") != "RADAR-LN3-V27a-late-residual-checkpoint-v2"
        or payload.get("experiment_id") != "V27a"
        or payload.get("arm") != "main"
        or int(payload.get("seed", -1)) != 42
        or int(payload.get("epoch", -1)) != 20
        or payload.get("model_state_sha256") != config.base.expected_v27a_main_state_sha256
        or v27a_state_hash(payload["model_state"]) != config.base.expected_v27a_main_state_sha256
        or payload.get("model_contract") != contract
        or payload.get("model_contract_sha256") != config.base.expected_v27a_main_model_contract_sha256
        or canonical_sha256(contract) != config.base.expected_v27a_main_model_contract_sha256
        or payload.get("base_checkpoint_sha256") != config.base.expected_l2_checkpoint_sha256
        or payload.get("base_classifier_state_sha256")
        != config.base.expected_l2_classifier_state_sha256
        or payload.get("class_names") != list(R23_L2_CLASSES)
    ):
        raise ValueError("V27c-r23 V27a Main checkpoint contract differs")
    return dict(payload)


def _fit_clipping_binding(root: Path, config: R23Config) -> dict[str, Any]:
    fitted = fit_tr_clipping(
        root,
        index_path=config.data.index_path,
        split_name=config.data.tr_clipping_fit_split,
        schema_version=config.data.index_schema_version,
        quantile=config.data.tr_clipping_quantile,
        range_lower_m=config.data.tr_range_lower_m,
        speed_lower_mps=config.data.tr_speed_lower_mps,
    )
    fitted["index_path"] = relative_project_path(
        root, _path(root, config.data.index_path)
    )
    return fitted


def verify_r23_inputs(
    project_root: str | Path, config: R23Config
) -> tuple[Any, Any, dict[str, Any]]:
    """Verify frozen inputs and derive the train-only clipping provenance."""

    root = Path(project_root).resolve()
    v27_path = _path(root, config.base.v27_config_path)
    if sha256_file(v27_path) != config.base.expected_v27_config_sha256:
        raise ValueError("V27c-r23 V27 config hash differs")
    v27_config = load_v27_config(root, v27_path)
    if v27_config.config_sha256 != config.base.expected_v27_config_sha256:
        raise ValueError("V27c-r23 V27 config identity differs")
    if (
        v27_config.data.index_path != config.data.index_path
        or v27_config.data.index_schema_version != config.data.index_schema_version
        or v27_config.data.max_sequence_length != config.data.max_sequence_length
        or v27_config.data.max_rd_frames_per_point != config.data.max_rd_frames_per_point
        or not math.isclose(v27_config.data.l2.rd_amplitude_center, config.data.rd_amplitude_center)
        or not math.isclose(v27_config.data.l2.rd_amplitude_scale, config.data.rd_amplitude_scale)
    ):
        raise ValueError("V27c-r23 V27 L2 data contract differs")
    data_binding = verify_v27_data_contract(root, v27_config)
    if data_binding["hashes"]["index"] != config.data.expected_index_sha256:
        raise ValueError("V27c-r23 index hash differs")
    frozen_binding = _verify_frozen_binding(root, config)
    l2_checkpoint = _load_l2_checkpoint(root, config)

    v27a_path = _path(root, config.base.v27a_config_path)
    if sha256_file(v27a_path) != config.base.expected_v27a_config_sha256:
        raise ValueError("V27c-r23 V27a config hash differs")
    v27a_config = load_v27a_config(root, v27a_path)
    if v27a_config.config_sha256 != config.base.expected_v27a_config_sha256:
        raise ValueError("V27c-r23 V27a config identity differs")
    manifest_path = _path(root, config.base.v27a_manifest_path)
    if sha256_file(manifest_path) != config.base.expected_v27a_manifest_sha256:
        raise ValueError("V27c-r23 V27a manifest hash differs")
    v27a_manifest = read_v27a_sealed_json(manifest_path)
    _verify_sealed_payload(
        v27a_manifest,
        expected=config.base.expected_v27a_manifest_payload_sha256,
        description="V27a manifest",
    )
    audit_path = _path(root, config.base.v27a_evidence_audit_path)
    if sha256_file(audit_path) != config.base.expected_v27a_evidence_audit_sha256:
        raise ValueError("V27c-r23 V27a evidence audit hash differs")
    v27a_main = _verify_v27a_main_checkpoint(root, config, v27a_config)

    provisional = R23Context(
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
    for split in R23_SPLITS:
        cache = load_verified_v27a_cache(provisional, split=split)
        cache_bindings[split] = {
            "path": relative_project_path(root, _path(root, _cache_path(config, split))),
            "file_sha256": _cache_file_hash(config, split),
            "payload_sha256": cache["cache_payload_sha256"],
            "parent_count": len(cache["parent_sample_ids"]),
        }

    clipping = _fit_clipping_binding(root, config)
    clipping_audits = {
        split: clipping_audit(
            root,
            index_path=config.data.index_path,
            split_name=split,
            schema_version=config.data.index_schema_version,
            clipping=clipping,
        )
        for split in R23_SPLITS
    }
    binding = {
        "v27": {
            "config_path": relative_project_path(root, v27_path),
            "config_sha256": v27_config.config_sha256,
            "data_registry_sha256": data_binding["data_registry_sha256"],
            "l2_classifier_state_sha256": config.base.expected_l2_classifier_state_sha256,
        },
        "l1_binding_only": {
            "checkpoint_path": relative_project_path(root, _path(root, config.base.l1_checkpoint_path)),
            "checkpoint_sha256": config.base.expected_l1_checkpoint_sha256,
            "state_sha256_from_frozen_binding": config.base.expected_l1_state_sha256,
            "checkpoint_deserialized": False,
            "model_loaded": False,
            "forward_calls": 0,
            "optimizer_parameter_count": 0,
        },
        "l2_initialization": {
            "checkpoint_path": relative_project_path(root, _path(root, config.base.l2_checkpoint_path)),
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
            "checkpoint_path": relative_project_path(root, _path(root, config.base.v27a_main_checkpoint_path)),
            "checkpoint_sha256": config.base.expected_v27a_main_checkpoint_sha256,
            "state_sha256": v27a_main["model_state_sha256"],
            "model_contract_sha256": v27a_main["model_contract_sha256"],
            "arm": "main",
            "epoch": 20,
        },
        "v27a_caches": cache_bindings,
        "arm_specs": {
            arm: dict(spec) for arm, spec in R23_ARM_SPECS.items()
        },
        "tr_clipping": clipping,
        "tr_clipping_audits": clipping_audits,
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


def _source_paths() -> tuple[Path, ...]:
    return (
        Path("src-v3/v27/v27c/r23/__init__.py"),
        Path("src-v3/v27/v27c/r23/protocol.py"),
        Path("src-v3/v27/v27c/r23/preprocessing.py"),
        Path("src-v3/v27/v27c/r23/data.py"),
        Path("src-v3/v27/v27c/r23/model.py"),
        Path("src-v3/v27/v27c/r23/training.py"),
        Path("src-v3/v27/v27c/r23/evaluation.py"),
        Path("src-v3/v27/v27c/r23/cli.py"),
        Path("src-v3/v27/v27_baseline/base_binding.py"),
        Path("src-v3/v27/v27_baseline/data.py"),
        Path("src-v3/v27/v27_baseline/logging_utils.py"),
        Path("src-v3/v27/v27_baseline/metrics.py"),
        Path("src-v3/v27/v27_baseline/model.py"),
        Path("src-v3/v27/v27_baseline/protocol.py"),
        Path("src-v3/radar_ln3/data.py"),
        Path("src-v3/radar_ln3/features.py"),
        Path("src-v3/radar_ln3/liquid_v2.py"),
        Path("src-v3/radar_ln3/losses.py"),
        Path("src-v3/radar_ln3/runtime.py"),
        Path("src-v3/v27/v27a/cache.py"),
        Path("src-v3/v27/v27a/checkpoint.py"),
        Path("src-v3/v27/v27a/model.py"),
        Path("src-v3/v27/v27a/protocol.py"),
    )


def build_r23_manifest(
    project_root: str | Path,
    config: R23Config,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    hashes: dict[str, str] = {}
    for path in _source_paths():
        target = root / path
        if not target.is_file():
            raise FileNotFoundError(f"V27c-r23 source file is missing: {target}")
        hashes[path.as_posix()] = sha256_file(target)
    return {
        "schema_version": R23_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R23_EXPERIMENT_ID,
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
            "base": "V27 L2 epoch-11 state initialized in every arm",
            "reference": "legacy V27 L2 input and frozen reference state",
            "stable8": "V27a Main epoch-20 full reliable-frame residual",
            "motion": "bias-free causal motion adapter before Liquid V2 recurrence",
            "initial_logit_equation": "cached_v27a_logits + current_l2(reference_input_delta) + stable8_residual",
            "trajectory_heads": "frozen_and_unused_by_classification",
        },
        "training": {
            "arms": list(R23_ARMS),
            "arm_specs": R23_ARM_SPECS,
            "seeds": list(config.training.seeds),
            "primary_epoch_budget": config.training.primary_epoch,
            "checkpoint_selection": config.protocol.get(
                "checkpoint_selection",
                "fixed_primary_epoch_without_validation_best_selection",
            ),
            "class_weights": list(config.base.expected_l2_class_weights),
        },
        "protocol": dict(config.protocol),
        "scope": dict(input_binding["scope"]),
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    if "payload_sha256" in body:
        raise ValueError("V27c-r23 payload is already sealed")
    return {**body, "payload_sha256": canonical_sha256(body)}


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_seal(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V27c-r23 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    atomic_write_text(target, json_text(_seal(payload), indent=2))
    return target


def read_sealed_json(path: str | Path, *, schema: str | None = None) -> dict[str, Any]:
    payload = _read_json(Path(path), description="sealed artifact")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("V27c-r23 sealed artifact payload hash differs")
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError("V27c-r23 sealed artifact schema differs")
    return payload


def prepare_r23_manifest(
    project_root: str | Path, config_path: str | Path | None = None
) -> tuple[R23Config, dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_r23_config(root, config_path)
    _, _, input_binding = verify_r23_inputs(root, config)
    payload = build_r23_manifest(root, config, input_binding)
    path = r23_artifact_path(root, config.artifacts.manifest_path)
    if path.exists():
        existing = read_sealed_json(path, schema=R23_MANIFEST_SCHEMA_VERSION)
        existing_body = {key: value for key, value in existing.items() if key != "payload_sha256"}
        if existing_body != payload:
            protected = [
                r23_artifact_path(
                    root,
                    config.artifacts.training_summary_template,
                    arm=arm,
                    seed=seed,
                )
                for arm in R23_ARMS
                for seed in config.training.seeds
            ] + [
                r23_artifact_path(
                    root, config.artifacts.evaluation_report_template, seed=seed
                )
                for seed in config.training.seeds
            ]
            if any(candidate.exists() for candidate in protected):
                raise ValueError("V27c-r23 manifest is stale after a derived artifact exists")
            atomic_write_text(path, json_text(_seal(payload), indent=2))
        return config, input_binding, path, sha256_file(path)
    write_sealed_json_if_identical(path, payload)
    return config, input_binding, path, sha256_file(path)


def load_r23_context(
    project_root: str | Path, config_path: str | Path | None = None
) -> R23Context:
    root = Path(project_root).resolve()
    config = load_r23_config(root, config_path)
    v27_config, v27a_config, input_binding = verify_r23_inputs(root, config)
    path = r23_artifact_path(root, config.artifacts.manifest_path)
    manifest = read_sealed_json(path, schema=R23_MANIFEST_SCHEMA_VERSION)
    expected = build_r23_manifest(root, config, input_binding)
    actual_body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual_body != expected:
        raise ValueError("V27c-r23 manifest differs from current code, config, or inputs")
    return R23Context(
        project_root=root,
        config=config,
        v27_config=v27_config,
        v27a_config=v27a_config,
        input_binding=input_binding,
        manifest=manifest,
        manifest_path=path,
        manifest_sha256=sha256_file(path),
    )


__all__ = ["R23_ARM_SPECS", "R23_ARMS", "R23_CHECKPOINT_SCHEMA_VERSION", "R23_EXPERIMENT_ID", "R23_L2_CLASSES", "R23_MANIFEST_SCHEMA_VERSION", "R23_REPORT_SCHEMA_VERSION", "R23_SCHEMA_VERSION", "R23_SMOKE_SCHEMA_VERSION", "R23_SUMMARY_SCHEMA_VERSION", "R23Context", "R23Config", "build_r23_manifest", "load_r23_config", "load_r23_context", "load_verified_v27a_cache", "prepare_r23_manifest", "r23_artifact_path", "read_sealed_json", "verify_r23_inputs", "write_sealed_json", "write_sealed_json_if_identical"]