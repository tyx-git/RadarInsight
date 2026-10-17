"""Configuration and provenance boundaries for V27b-r6.

R6 is deliberately a representation experiment, not another L2 classifier.
It consumes frozen V27 L2 trajectory representations and a separately
materialized reliable-stable8 sidecar.  The sidecar must retain
``frame_feature_reliable`` semantics; ``RD present`` is insufficient.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)


R6_EXPERIMENT_ID = "V27b-r6"
R6_SCHEMA_VERSION = "RADAR-LN3-V27b-r6-config-v1"
R6_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27b-r6-manifest-v1"
R6_FEATURE_CACHE_SCHEMA_VERSION = "RADAR-LN3-V27b-r6-mixed-knownness-feature-cache-v1"
R6_FEATURE_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27b-r6-feature-cache-manifest-v1"
R6_HEAD_CHECKPOINT_SCHEMA_VERSION = "RADAR-LN3-V27b-r6-knownness-head-checkpoint-v1"
R6_REPORT_SCHEMA_VERSION = "RADAR-LN3-V27b-r6-knownness-report-v1"
R6_SMOKE_SCHEMA_VERSION = "RADAR-LN3-V27b-r6-contract-smoke-v1"

R6_ARMS = ("control", "stable8_treatment")
R6_ROLES = ("representation_fit", "representation_development", "calibration")
R6_KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
R6_NEGATIVE_LABEL = "UnknownTarget"
R6_DIAGNOSTIC_LABEL = "ClutterTarget"
R6_STABLE8_INDICES = (1, 2, 3, 7, 11, 13, 14, 22)
R6_STABLE8_NAMES = (
    "residual_positive_8_30mps_energy_fraction",
    "residual_negative_30_60mps_energy_fraction",
    "residual_positive_30_60mps_energy_fraction",
    "residual_rms_offset_mps_over_90",
    "residual_top4_resolution_cell_concentration",
    "residual_sideband_snr_db_over_30",
    "pulse_envelope_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_snr_db_over_30",
)

_SHA256_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class R6BaseConfig:
    v27_system_config_path: str
    expected_v27_system_config_sha256: str
    expected_v27_system_manifest_sha256: str
    expected_v27_open_set_pool_registry_sha256: str
    expected_v27_representation_cache_manifest_sha256: str
    v27a_config_path: str
    expected_v27a_config_sha256: str
    v27a_manifest_path: str
    expected_v27a_manifest_sha256: str
    v27a_main_checkpoint_path: str
    expected_v27a_main_checkpoint_sha256: str
    expected_v27a_main_state_sha256: str
    expected_v27a_main_model_contract_sha256: str
    expected_v27a_main_training_config_sha256: str
    expected_v27a_main_training_config_identity_sha256: str
    expected_v27a_main_training_manifest_sha256: str
    expected_v27a_main_base_l2_checkpoint_sha256: str
    expected_v27a_main_base_l2_state_sha256: str
    v27a_normalization_path: str
    expected_v27a_normalization_sha256: str
    v27a_train_reference_cache_path: str
    expected_v27a_train_reference_cache_sha256: str
    expected_v27a_train_reference_cache_payload_sha256: str
    v27a_validation_reference_cache_path: str
    expected_v27a_validation_reference_cache_sha256: str
    expected_v27a_validation_reference_cache_payload_sha256: str
    v27a_paired_report_path: str
    expected_v27a_paired_report_sha256: str
    v16_provenance_path: str
    expected_v16_provenance_sha256: str
    canonical_frame_index_path: str
    expected_canonical_frame_index_sha256: str


@dataclass(frozen=True)
class R6DataConfig:
    task: str
    known_labels: tuple[str, ...]
    negative_label: str
    diagnostic_label: str
    stable8_indices: tuple[int, ...]
    stable8_dim: int
    max_sequence_length: int
    max_frames_per_event: int
    event_trajectory_dim: int
    summary_feature_dim: int
    task_fact_dim: int


@dataclass(frozen=True)
class R6QualityConfig:
    base_target: str
    minimum_reliable_event_count: int
    minimum_reliable_event_fraction: float
    low_quality_policy: str


@dataclass(frozen=True)
class R6ModelConfig:
    trajectory_projection_dim: int
    trajectory_hidden_dim: int
    summary_projection_dim: int
    stable_projection_dim: int
    fusion_hidden_dim: int


@dataclass(frozen=True)
class R6TrainingConfig:
    seeds: tuple[int, ...]
    dataloader_seed: int
    batch_size: int
    epochs: int
    primary_epoch: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    device: str


@dataclass(frozen=True)
class R6EvaluationConfig:
    known_recall_floors: tuple[float, ...]
    paired_bootstrap_replicates: int
    primary_role: str
    calibration: str
    l3: str


@dataclass(frozen=True)
class R6ArtifactConfig:
    manifest_path: str
    feature_cache_template: str
    feature_manifest_path: str
    smoke_report_path: str
    training_log_template: str
    training_summary_template: str
    checkpoint_template: str
    evaluation_report_template: str
    comparison_report_template: str
    summary_report_template: str


@dataclass(frozen=True)
class R6Config:
    schema_version: str
    experiment_id: str
    variant: str
    objective: str
    base: R6BaseConfig
    data: R6DataConfig
    quality: R6QualityConfig
    model: R6ModelConfig
    training: R6TrainingConfig
    evaluation: R6EvaluationConfig
    artifacts: R6ArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


@dataclass(frozen=True)
class R6Context:
    project_root: Path
    config: R6Config
    input_binding: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27b-r6 {context} lacks {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27b-r6 {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27b-r6 {context}.{key} must be a non-empty string")
    return value.strip()


def _sha(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(mapping, key, context).lower()
    if len(value) != 64 or any(character not in _SHA256_HEX for character in value):
        raise ValueError(f"V27b-r6 {context}.{key} must be a SHA-256 digest")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27b-r6 {context}.{key} must be a positive integer")
    return int(value)


def _nonnegative_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V27b-r6 {context}.{key} must be a non-negative integer")
    return int(value)


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ValueError(f"V27b-r6 {context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27b-r6 {context}.{key} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"V27b-r6 {context}.{key} must be finite")
    if lower is not None and result < lower:
        raise ValueError(f"V27b-r6 {context}.{key} is below its lower bound")
    if upper is not None and result > upper:
        raise ValueError(f"V27b-r6 {context}.{key} is above its upper bound")
    return result


def _text_tuple(
    mapping: Mapping[str, Any], key: str, context: str
) -> tuple[str, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27b-r6 {context}.{key} must be a non-empty list")
    values = tuple(str(value).strip() for value in raw)
    if any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"V27b-r6 {context}.{key} has invalid values")
    return values


def _int_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[int, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27b-r6 {context}.{key} must be a non-empty list")
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"V27b-r6 {context}.{key} must contain integers")
        values.append(int(value))
    if len(set(values)) != len(values):
        raise ValueError(f"V27b-r6 {context}.{key} repeats a value")
    return tuple(values)


def _float_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[float, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27b-r6 {context}.{key} must be a non-empty list")
    values = tuple(float(value) for value in raw)
    if any(not math.isfinite(value) or not 0.0 < value <= 1.0 for value in values):
        raise ValueError(f"V27b-r6 {context}.{key} must contain recall floors in (0, 1]")
    if tuple(sorted(values, reverse=True)) != values:
        raise ValueError(f"V27b-r6 {context}.{key} must be descending")
    return values


def _expected_protocol() -> dict[str, Any]:
    return {
        "classification_endpoint": "frozen_v27a_main_epoch_20",
        "classifier_logits_input_to_knownness": False,
        "classification_logits_must_remain_bitwise_unchanged": True,
        "unknown_is_l2_classifier_class": False,
        "unknown_used_only_as_knownness_negative": True,
        "clutter_used_for_knownness_optimization": False,
        "low_quality_used_for_knownness_optimization": False,
        "stable8_reliability": "frame_feature_reliable_not_rd_present",
        "selection": "fixed_primary_epoch; representation_development_diagnostic_only",
        "calibration": "not_fitted",
        "l3": "not_fitted",
        "test": "sealed_and_not_used",
    }


def load_r6_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> R6Config:
    root = Path(project_root).resolve()
    candidate = (
        root / "src-v3" / "v27" / "v27b" / "r6" / "config" / "V27b-r6-knownness-stable8-summary.json"
        if config_path is None
        else Path(config_path)
    )
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("V27b-r6 config must be an object")
    payload = dict(raw)
    if payload.get("schema_version") != R6_SCHEMA_VERSION or payload.get("experiment_id") != R6_EXPERIMENT_ID:
        raise ValueError("V27b-r6 config identity differs")

    base_raw = _mapping(payload, "base", "root")
    base = R6BaseConfig(
        **{
            key: (
                _sha(base_raw, key, "base")
                if key.startswith("expected_")
                else _text(base_raw, key, "base")
            )
            for key in R6BaseConfig.__dataclass_fields__
        }
    )
    data_raw = _mapping(payload, "data", "root")
    data = R6DataConfig(
        task=_text(data_raw, "task", "data"),
        known_labels=_text_tuple(data_raw, "known_labels", "data"),
        negative_label=_text(data_raw, "negative_label", "data"),
        diagnostic_label=_text(data_raw, "diagnostic_label", "data"),
        stable8_indices=_int_tuple(data_raw, "stable8_indices", "data"),
        stable8_dim=_positive_int(data_raw, "stable8_dim", "data"),
        max_sequence_length=_positive_int(data_raw, "max_sequence_length", "data"),
        max_frames_per_event=_positive_int(data_raw, "max_frames_per_event", "data"),
        event_trajectory_dim=_positive_int(data_raw, "event_trajectory_dim", "data"),
        summary_feature_dim=_positive_int(data_raw, "summary_feature_dim", "data"),
        task_fact_dim=_positive_int(data_raw, "task_fact_dim", "data"),
    )
    if (
        data.task != "dbb_specialist"
        or data.known_labels != R6_KNOWN_LABELS
        or data.negative_label != R6_NEGATIVE_LABEL
        or data.diagnostic_label != R6_DIAGNOSTIC_LABEL
        or data.stable8_indices != R6_STABLE8_INDICES
        or data.stable8_dim != len(R6_STABLE8_INDICES)
        or data.max_sequence_length != 29
        or data.max_frames_per_event != 12
    ):
        raise ValueError("V27b-r6 data contract differs")
    if (
        data.event_trajectory_dim != 355
        or data.summary_feature_dim != 672
        or data.task_fact_dim != 7
    ):
        raise ValueError("V27b-r6 frozen L2 representation dimensions differ")

    quality_raw = _mapping(payload, "quality", "root")
    quality = R6QualityConfig(
        base_target=_text(quality_raw, "base_target", "quality"),
        minimum_reliable_event_count=_positive_int(
            quality_raw, "minimum_reliable_event_count", "quality"
        ),
        minimum_reliable_event_fraction=_finite_float(
            quality_raw, "minimum_reliable_event_fraction", "quality", lower=0.0, upper=1.0
        ),
        low_quality_policy=_text(quality_raw, "low_quality_policy", "quality"),
    )
    if (
        quality.base_target != "v27_deterministic_causal_quality_facts_v1"
        or quality.low_quality_policy
        != "exclude_from_knownness_fit_and_report_as_pending_diagnostic"
    ):
        raise ValueError("V27b-r6 quality contract differs")
    if quality.minimum_reliable_event_count > data.max_sequence_length:
        raise ValueError("V27b-r6 reliable-event minimum exceeds sequence length")

    model_raw = _mapping(payload, "model", "root")
    model = R6ModelConfig(
        trajectory_projection_dim=_positive_int(model_raw, "trajectory_projection_dim", "model"),
        trajectory_hidden_dim=_positive_int(model_raw, "trajectory_hidden_dim", "model"),
        summary_projection_dim=_positive_int(model_raw, "summary_projection_dim", "model"),
        stable_projection_dim=_positive_int(model_raw, "stable_projection_dim", "model"),
        fusion_hidden_dim=_positive_int(model_raw, "fusion_hidden_dim", "model"),
    )

    training_raw = _mapping(payload, "training", "root")
    seeds = _int_tuple(training_raw, "seeds", "training")
    training = R6TrainingConfig(
        seeds=seeds,
        dataloader_seed=_nonnegative_int(training_raw, "dataloader_seed", "training"),
        batch_size=_positive_int(training_raw, "batch_size", "training"),
        epochs=_positive_int(training_raw, "epochs", "training"),
        primary_epoch=_positive_int(training_raw, "primary_epoch", "training"),
        learning_rate=_finite_float(training_raw, "learning_rate", "training", lower=1.0e-12),
        weight_decay=_finite_float(training_raw, "weight_decay", "training", lower=0.0),
        gradient_clip_norm=_finite_float(training_raw, "gradient_clip_norm", "training", lower=1.0e-12),
        device=_text(training_raw, "device", "training").lower(),
    )
    if training.primary_epoch > training.epochs or training.device not in {"cpu", "cuda"}:
        raise ValueError("V27b-r6 training contract differs")

    evaluation_raw = _mapping(payload, "evaluation", "root")
    evaluation = R6EvaluationConfig(
        known_recall_floors=_float_tuple(evaluation_raw, "known_recall_floors", "evaluation"),
        paired_bootstrap_replicates=_positive_int(
            evaluation_raw, "paired_bootstrap_replicates", "evaluation"
        ),
        primary_role=_text(evaluation_raw, "primary_role", "evaluation"),
        calibration=_text(evaluation_raw, "calibration", "evaluation"),
        l3=_text(evaluation_raw, "l3", "evaluation"),
    )
    if (
        evaluation.paired_bootstrap_replicates < 100
        or evaluation.paired_bootstrap_replicates > 100_000
        or
        evaluation.primary_role != "representation_development"
        or evaluation.calibration != "not_in_scope"
        or evaluation.l3 != "not_in_scope"
    ):
        raise ValueError("V27b-r6 evaluation contract differs")

    artifact_raw = _mapping(payload, "artifacts", "root")
    artifacts = R6ArtifactConfig(
        **{
            key: _text(artifact_raw, key, "artifacts")
            for key in R6ArtifactConfig.__dataclass_fields__
        }
    )
    artifact_root = (root / "src-v3" / "v27" / "v27b" / "r6").resolve()
    for field, template in artifacts.__dict__.items():
        try:
            rendered = template.format(role=R6_ROLES[0], arm=R6_ARMS[0], seed=0, epoch=1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27b-r6 artifact template is invalid: {field}") from exc
        resolved = resolve_project_path(root, rendered)
        try:
            resolved.relative_to(artifact_root)
        except ValueError as exc:
            raise ValueError(f"V27b-r6 artifact escapes its package: {field}") from exc
    for template in (
        artifacts.training_log_template,
        artifacts.training_summary_template,
        artifacts.checkpoint_template,
        artifacts.evaluation_report_template,
        artifacts.comparison_report_template,
    ):
        if "{seed}" not in template:
            raise ValueError("V27b-r6 seed-scoped artifact template lacks {seed}")

    protocol = _mapping(payload, "protocol", "root")
    if protocol != _expected_protocol():
        raise ValueError("V27b-r6 protocol differs from the registered design")
    return R6Config(
        schema_version=R6_SCHEMA_VERSION,
        experiment_id=R6_EXPERIMENT_ID,
        variant=_text(payload, "variant", "root"),
        objective=_text(payload, "objective", "root"),
        base=base,
        data=data,
        quality=quality,
        model=model,
        training=training,
        evaluation=evaluation,
        artifacts=artifacts,
        protocol=protocol,
        config_path=path,
        config_sha256=sha256_file(path),
        config_identity_sha256=canonical_sha256(payload),
    )


R6_SOURCE_PATHS = (
    Path("src-v3/v27/__init__.py"),
    Path("src-v3/v27/v27b/r6/__init__.py"),
    Path("src-v3/v27/v27b/r6/protocol.py"),
    Path("src-v3/v27/v27b/r6/features.py"),
    Path("src-v3/v27/v27b/r6/model.py"),
    Path("src-v3/v27/v27b/r6/training.py"),
    Path("src-v3/v27/v27b/r6/evaluation.py"),
    Path("src-v3/v27/v27b/r6/cli.py"),
    Path("src-v3/v27/v27_baseline/open_set.py"),
    Path("src-v3/v27/v27_baseline/open_set_data.py"),
    Path("src-v3/v27/v27_baseline/model.py"),
    Path("src-v3/v27/v27_baseline/runtime.py"),
    Path("src-v3/v27/v27_baseline/logging_utils.py"),
    Path("src-v3/v27/v27a/evidence.py"),
    Path("src-v3/v27/v27a/cache.py"),
    Path("src-v3/v27/v27a/model.py"),
    Path("src-v3/v27/v27a/protocol.py"),
    Path("src-v3/build/micro_doppler_v17.py"),
    Path("src-v3/build/micro_doppler_v18.py"),
    Path("src-v3/radar_ln3/runtime.py"),
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V27b-r6 JSON artifact is not an object: {path}")
    return dict(value)


def _file_binding(root: Path, path_value: str, expected_hash: str, *, field: str) -> dict[str, str]:
    path = resolve_project_path(root, path_value, must_exist=True)
    actual = sha256_file(path)
    if actual != expected_hash:
        raise ValueError(f"V27b-r6 {field} file hash differs")
    return {"path": relative_project_path(root, path), "sha256": actual}


def verify_r6_inputs(project_root: str | Path, config: R6Config) -> dict[str, Any]:
    """Bind all frozen endpoints and source evidence before r6 runs."""

    root = Path(project_root).resolve()
    files = {
        "v27_system_config": _file_binding(
            root, config.base.v27_system_config_path, config.base.expected_v27_system_config_sha256, field="V27 config"
        ),
        "v27a_config": _file_binding(
            root, config.base.v27a_config_path, config.base.expected_v27a_config_sha256, field="V27a config"
        ),
        "v27a_manifest": _file_binding(
            root, config.base.v27a_manifest_path, config.base.expected_v27a_manifest_sha256, field="V27a manifest"
        ),
        "v27a_main_checkpoint": _file_binding(
            root, config.base.v27a_main_checkpoint_path, config.base.expected_v27a_main_checkpoint_sha256, field="V27a Main checkpoint"
        ),
        "v27a_normalization": _file_binding(
            root, config.base.v27a_normalization_path, config.base.expected_v27a_normalization_sha256, field="V27a normalization"
        ),
        "v27a_train_reference_cache": _file_binding(
            root, config.base.v27a_train_reference_cache_path, config.base.expected_v27a_train_reference_cache_sha256, field="V27a train reference cache"
        ),
        "v27a_validation_reference_cache": _file_binding(
            root, config.base.v27a_validation_reference_cache_path, config.base.expected_v27a_validation_reference_cache_sha256, field="V27a reference cache"
        ),
        "v27a_paired_report": _file_binding(
            root, config.base.v27a_paired_report_path, config.base.expected_v27a_paired_report_sha256, field="V27a paired report"
        ),
        "v16_provenance": _file_binding(
            root, config.base.v16_provenance_path, config.base.expected_v16_provenance_sha256, field="V16 provenance"
        ),
        "canonical_frame_index": _file_binding(
            root, config.base.canonical_frame_index_path, config.base.expected_canonical_frame_index_sha256, field="canonical frame index"
        ),
    }

    from v27.v27_baseline.open_set import load_representation_cache_manifest
    from v27.v27_baseline.open_set_data import load_open_set_pool_artifacts
    from v27.v27_baseline.runtime import load_v27_context

    v27_context = load_v27_context(root, config.base.v27_system_config_path)
    if v27_context.manifest_sha256 != config.base.expected_v27_system_manifest_sha256:
        raise ValueError("V27b-r6 active V27 manifest differs")
    pools, _ = load_open_set_pool_artifacts(v27_context)
    if pools["registry_sha256"] != config.base.expected_v27_open_set_pool_registry_sha256:
        raise ValueError("V27b-r6 active V27 pool registry differs")
    representation_manifest = load_representation_cache_manifest(v27_context)
    if (
        representation_manifest["representation_cache_manifest_sha256"]
        != config.base.expected_v27_representation_cache_manifest_sha256
    ):
        raise ValueError("V27b-r6 active V27 representation manifest differs")

    import torch

    checkpoint_path = resolve_project_path(root, config.base.v27a_main_checkpoint_path, must_exist=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("experiment_id") != "V27a"
        or checkpoint.get("arm") != "main"
        or int(checkpoint.get("epoch", -1)) != 20
        or checkpoint.get("model_state_sha256") != config.base.expected_v27a_main_state_sha256
        or checkpoint.get("model_contract_sha256")
        != config.base.expected_v27a_main_model_contract_sha256
        or checkpoint.get("config_sha256")
        != config.base.expected_v27a_main_training_config_sha256
        or checkpoint.get("config_identity_sha256")
        != config.base.expected_v27a_main_training_config_identity_sha256
        or checkpoint.get("manifest_sha256")
        != config.base.expected_v27a_main_training_manifest_sha256
        or checkpoint.get("base_checkpoint_sha256")
        != config.base.expected_v27a_main_base_l2_checkpoint_sha256
        or checkpoint.get("base_classifier_state_sha256")
        != config.base.expected_v27a_main_base_l2_state_sha256
    ):
        raise ValueError("V27b-r6 V27a Main checkpoint identity differs")
    cache_payload_sha256s = checkpoint.get("cache_payload_sha256s")
    if not isinstance(cache_payload_sha256s, Mapping) or {
        str(key): str(value) for key, value in cache_payload_sha256s.items()
    } != {
        "train": config.base.expected_v27a_train_reference_cache_payload_sha256,
        "val": config.base.expected_v27a_validation_reference_cache_payload_sha256,
    }:
        raise ValueError("V27b-r6 V27a Main checkpoint cache binding differs")

    from v27.v27a.cache import load_cache_payload

    reference_cache_payloads = {
        "train": load_cache_payload(
            resolve_project_path(root, config.base.v27a_train_reference_cache_path, must_exist=True)
        ),
        "val": load_cache_payload(
            resolve_project_path(root, config.base.v27a_validation_reference_cache_path, must_exist=True)
        ),
    }
    for split, expected_payload_sha256 in (
        ("train", config.base.expected_v27a_train_reference_cache_payload_sha256),
        ("val", config.base.expected_v27a_validation_reference_cache_payload_sha256),
    ):
        reference = reference_cache_payloads[split]
        if (
            reference.get("cache_payload_sha256") != expected_payload_sha256
            or reference.get("base_checkpoint_sha256")
            != config.base.expected_v27a_main_base_l2_checkpoint_sha256
            or reference.get("base_classifier_state_sha256")
            != config.base.expected_v27a_main_base_l2_state_sha256
            or reference.get("v27a_manifest_sha256")
            != config.base.expected_v27a_main_training_manifest_sha256
            or reference.get("config_identity_sha256")
            != config.base.expected_v27a_main_training_config_identity_sha256
        ):
            raise ValueError(f"V27b-r6 V27a {split} reference cache identity differs")
    report = _read_json(resolve_project_path(root, config.base.v27a_paired_report_path, must_exist=True))
    main = dict(dict(report.get("arms") or {}).get("main") or {})
    matrix = dict(main.get("parent_metrics") or {}).get("confusion_matrix_rows_true_columns_pred")
    if matrix != [[109, 4, 0], [1, 54, 1], [1, 1, 52]]:
        raise ValueError("V27b-r6 V27a Main reference metrics differ")

    return {
        "files": files,
        "v27": {
            "system_manifest_sha256": v27_context.manifest_sha256,
            "pool_registry_sha256": pools["registry_sha256"],
            "representation_cache_manifest_sha256": representation_manifest[
                "representation_cache_manifest_sha256"
            ],
        },
        "v27a_main": {
            "checkpoint_state_sha256": checkpoint["model_state_sha256"],
            "model_contract_sha256": checkpoint["model_contract_sha256"],
            "training_manifest_sha256": checkpoint["manifest_sha256"],
            "cache_payload_sha256s": dict(cache_payload_sha256s),
            "classification_metrics_reference": matrix,
        },
        "unknown_is_classifier_class": False,
        "test_payload_access_count": 0,
    }


def _git_head(root: Path) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else "unavailable"


def build_r6_manifest(
    project_root: str | Path,
    config: R6Config,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    code_hashes = {path.as_posix(): sha256_file(root / path) for path in R6_SOURCE_PATHS}
    dependencies: dict[str, str] = {}
    for package in ("torch", "numpy", "pandas"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "schema_version": R6_MANIFEST_SCHEMA_VERSION,
        "experiment_id": R6_EXPERIMENT_ID,
        "variant": config.variant,
        "objective": config.objective,
        "config": {
            "path": relative_project_path(root, config.config_path),
            "sha256": config.config_sha256,
            "identity_sha256": config.config_identity_sha256,
        },
        "code_hashes": code_hashes,
        "git_head": _git_head(root),
        "input_binding": dict(input_binding),
        "data": {
            "task": config.data.task,
            "known_labels": list(config.data.known_labels),
            "negative_label": config.data.negative_label,
            "diagnostic_label": config.data.diagnostic_label,
            "stable8_indices": list(config.data.stable8_indices),
            "stable8_names": list(R6_STABLE8_NAMES),
            "stable8_reliability": "frame_feature_reliable_not_rd_present",
        },
        "quality": dict(config.quality.__dict__),
        "training": {
            **dict(config.training.__dict__),
            "seeds": list(config.training.seeds),
        },
        "evaluation": {
            **dict(config.evaluation.__dict__),
            "known_recall_floors": list(config.evaluation.known_recall_floors),
        },
        "protocol": dict(config.protocol),
        "dependencies": dependencies,
        "scope": {
            "classification_logits": "frozen_v27a_main_bitwise_unchanged",
            "unknown": "knownness_negative_only_after_classifier_freeze",
            "clutter": "diagnostic_only",
            "calibration": "not_fitted",
            "l3": "not_fitted",
            "test": "sealed",
        },
    }


def _sealed_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27b-r6 payload is already sealed")
    body = dict(payload)
    return {**body, "payload_sha256": canonical_sha256(body)}


def verify_sealed_payload(payload: Mapping[str, Any], *, schema: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("V27b-r6 sealed payload must be an object")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("V27b-r6 sealed payload hash differs")
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError("V27b-r6 sealed payload schema differs")
    return dict(payload)


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_sealed_payload(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace different V27b-r6 artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, schema: str | None = None) -> dict[str, Any]:
    return verify_sealed_payload(
        _read_json(Path(path)), schema=schema
    )


def r6_artifact_path(
    project_root: str | Path,
    template: str,
    *,
    role: str | None = None,
    arm: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
) -> Path:
    if "{role}" in template and role not in R6_ROLES:
        raise ValueError("V27b-r6 artifact needs a canonical role")
    if "{arm}" in template and arm not in R6_ARMS:
        raise ValueError("V27b-r6 artifact needs a canonical arm")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("V27b-r6 artifact needs a non-negative seed")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("V27b-r6 artifact needs a positive epoch")
    try:
        rendered = template.format(role=role, arm=arm, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V27b-r6 artifact template is invalid") from exc
    return resolve_project_path(project_root, rendered)


def prepare_r6_manifest(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> tuple[R6Config, dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_r6_config(root, config_path)
    binding = verify_r6_inputs(root, config)
    manifest = build_r6_manifest(root, config, binding)
    path = r6_artifact_path(root, config.artifacts.manifest_path)
    write_sealed_json_if_identical(path, manifest)
    return config, binding, path, sha256_file(path)


def load_r6_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> R6Context:
    root = Path(project_root).resolve()
    config = load_r6_config(root, config_path)
    binding = verify_r6_inputs(root, config)
    path = r6_artifact_path(root, config.artifacts.manifest_path)
    manifest = read_sealed_json(path, schema=R6_MANIFEST_SCHEMA_VERSION)
    expected = build_r6_manifest(root, config, binding)
    if {key: value for key, value in manifest.items() if key != "payload_sha256"} != expected:
        raise ValueError("V27b-r6 manifest differs from current inputs or code")
    return R6Context(
        project_root=root,
        config=config,
        input_binding=dict(binding),
        manifest=manifest,
        manifest_path=path,
        manifest_sha256=sha256_file(path),
    )


__all__ = [
    "R6_ARMS",
    "R6_DIAGNOSTIC_LABEL",
    "R6_EXPERIMENT_ID",
    "R6_FEATURE_CACHE_SCHEMA_VERSION",
    "R6_FEATURE_MANIFEST_SCHEMA_VERSION",
    "R6_HEAD_CHECKPOINT_SCHEMA_VERSION",
    "R6_KNOWN_LABELS",
    "R6_MANIFEST_SCHEMA_VERSION",
    "R6_NEGATIVE_LABEL",
    "R6_REPORT_SCHEMA_VERSION",
    "R6_ROLES",
    "R6_SCHEMA_VERSION",
    "R6_SMOKE_SCHEMA_VERSION",
    "R6_STABLE8_INDICES",
    "R6_STABLE8_NAMES",
    "R6Config",
    "R6Context",
    "build_r6_manifest",
    "load_r6_config",
    "load_r6_context",
    "prepare_r6_manifest",
    "r6_artifact_path",
    "read_sealed_json",
    "verify_r6_inputs",
    "write_sealed_json_if_identical",
]
