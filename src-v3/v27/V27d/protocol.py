"""V27d configuration, path and provenance contracts.

V27d is an independent experiment package.  It consumes immutable artifacts
from the earlier V27 stages, but writes every derived artifact below this
directory.  The package deliberately does not load the active V27 baseline
manifest because that manifest is a historical classifier snapshot and may
not match the current working tree.
"""

from __future__ import annotations

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


V27D_EXPERIMENT_ID = "V27d"
V27D_SCHEMA_VERSION = "RADAR-LN3-V27d-frozen-endpoint-knownness-v1"
V27D_MANIFEST_SCHEMA = "RADAR-LN3-V27d-manifest-v1"
V27D_ARTIFACT_ROOT = Path("src-v3/v27/V27d")
V27D_DEFAULT_CONFIG = V27D_ARTIFACT_ROOT / "config" / "V27d-r1-frozen-endpoint-control.json"
V27D_TASKS = ("known_gate", "dbb_specialist")
V27D_ROLES = ("representation_fit", "representation_development", "calibration")
V27D_VARIANTS = ("r1", "r2", "r3", "r4")
V27D_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
V27D_KNOWN_LABELS = V27D_LABELS[:3]
V27D_STABLE8_INDICES = (1, 2, 3, 7, 11, 13, 14, 22)
V27D_STABLE8_NAMES = (
    "residual_positive_8_30mps_energy_fraction",
    "residual_negative_30_60mps_energy_fraction",
    "residual_positive_30_60mps_energy_fraction",
    "residual_rms_offset_mps_over_90",
    "residual_top4_resolution_cell_concentration",
    "residual_sideband_snr_db_over_30",
    "pulse_envelope_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_snr_db_over_30",
)


@dataclass(frozen=True)
class SourceConfig:
    baseline_config_path: str
    baseline_manifest_path: str
    baseline_pool_registry_path: str
    baseline_feature_manifest_path: str
    baseline_representation_manifest_path: str
    frozen_binding_path: str
    baseline_l1_checkpoint_path: str
    baseline_l2_checkpoint_path: str
    v27a_config_path: str
    v27a_main_checkpoint_path: str
    v27a_train_cache_path: str
    v27a_validation_cache_path: str
    v27a_normalization_path: str
    index_path: str
    v16_provenance_path: str
    canonical_frame_index_path: str
    expected_hashes: Mapping[str, str]


@dataclass(frozen=True)
class DataConfig:
    max_sequence_length: int
    max_frames_per_event: int
    liquid_hidden_dim: int
    l1_task_fact_dim: int
    l2_task_fact_dim: int
    event_fact_dim: int
    l1_class_count: int
    l2_class_count: int


@dataclass(frozen=True)
class ModelConfig:
    knownness_projection_dim: int
    trajectory_hidden_dim: int
    raw_scale_feature_dim: int
    stable8_feature_dim: int


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...]
    dataloader_seed: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    device: str


@dataclass(frozen=True)
class CalibrationConfig:
    minimum_quality_events: int
    minimum_quality_rd_fraction: float
    minimum_quality_dt_fraction: float
    minimum_quality_aux_fraction: float
    known_target_recall_floor: float
    clutter_recall_floor: float
    max_correct_known_new_rejections: int
    max_per_class_correct_known_loss: int
    conformal_error_rate: float
    minimum_calibration_unknown_support_for_formal: int


@dataclass(frozen=True)
class ArtifactConfig:
    manifest_path: str
    endpoint_cache_template: str
    feature_cache_template: str
    head_checkpoint_template: str
    head_summary_template: str
    training_log_template: str
    calibration_registry_path: str
    evaluation_report_template: str
    smoke_report_path: str


@dataclass(frozen=True)
class V27DConfig:
    schema_version: str
    experiment_id: str
    variant: str
    objective: str
    source: SourceConfig
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    artifacts: ArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


@dataclass(frozen=True)
class V27DContext:
    project_root: Path
    config: V27DConfig
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27d {context} is missing {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27d {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27d {context}.{key} must be a non-empty string")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27d {context}.{key} must be a positive integer")
    return int(value)


def _nonnegative_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V27d {context}.{key} must be a non-negative integer")
    return int(value)


def _finite_float(mapping: Mapping[str, Any], key: str, context: str, *, lower: float | None = None, upper: float | None = None) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ValueError(f"V27d {context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27d {context}.{key} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"V27d {context}.{key} must be finite")
    if lower is not None and result < lower:
        raise ValueError(f"V27d {context}.{key} is below its lower bound")
    if upper is not None and result > upper:
        raise ValueError(f"V27d {context}.{key} is above its upper bound")
    return result


def _sha_mapping(mapping: Mapping[str, Any], context: str) -> dict[str, str]:
    raw = _mapping(mapping, "expected_hashes", context)
    result: dict[str, str] = {}
    for key, value in raw.items():
        digest = str(value).strip().lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"V27d {context}.expected_hashes.{key} is not SHA-256")
        result[str(key)] = digest
    if not result:
        raise ValueError(f"V27d {context}.expected_hashes is empty")
    return result


def _int_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[int, ...]:
    raw = _required(mapping, key, context)
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"V27d {context}.{key} must be a non-empty list")
    values: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"V27d {context}.{key} must contain non-negative integers")
        values.append(int(value))
    if len(set(values)) != len(values):
        raise ValueError(f"V27d {context}.{key} contains duplicates")
    return tuple(values)


def load_config(project_root: str | Path, config_path: str | Path | None = None) -> V27DConfig:
    root = Path(project_root).resolve()
    candidate = root / V27D_DEFAULT_CONFIG if config_path is None else Path(config_path)
    path = resolve_project_path(root, candidate, must_exist=True)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("V27d configuration must be an object")
    payload = dict(raw)
    if payload.get("schema_version") != V27D_SCHEMA_VERSION or payload.get("experiment_id") != V27D_EXPERIMENT_ID:
        raise ValueError("V27d configuration identity differs")
    source_raw = _mapping(payload, "source", "root")
    source = SourceConfig(
        baseline_config_path=_text(source_raw, "baseline_config_path", "source"),
        baseline_manifest_path=_text(source_raw, "baseline_manifest_path", "source"),
        baseline_pool_registry_path=_text(source_raw, "baseline_pool_registry_path", "source"),
        baseline_feature_manifest_path=_text(source_raw, "baseline_feature_manifest_path", "source"),
        baseline_representation_manifest_path=_text(source_raw, "baseline_representation_manifest_path", "source"),
        frozen_binding_path=_text(source_raw, "frozen_binding_path", "source"),
        baseline_l1_checkpoint_path=_text(source_raw, "baseline_l1_checkpoint_path", "source"),
        baseline_l2_checkpoint_path=_text(source_raw, "baseline_l2_checkpoint_path", "source"),
        v27a_config_path=_text(source_raw, "v27a_config_path", "source"),
        v27a_main_checkpoint_path=_text(source_raw, "v27a_main_checkpoint_path", "source"),
        v27a_train_cache_path=_text(source_raw, "v27a_train_cache_path", "source"),
        v27a_validation_cache_path=_text(source_raw, "v27a_validation_cache_path", "source"),
        v27a_normalization_path=_text(source_raw, "v27a_normalization_path", "source"),
        index_path=_text(source_raw, "index_path", "source"),
        v16_provenance_path=_text(source_raw, "v16_provenance_path", "source"),
        canonical_frame_index_path=_text(source_raw, "canonical_frame_index_path", "source"),
        expected_hashes=_sha_mapping(source_raw, "source"),
    )
    data_raw = _mapping(payload, "data", "root")
    data = DataConfig(
        max_sequence_length=_positive_int(data_raw, "max_sequence_length", "data"),
        max_frames_per_event=_positive_int(data_raw, "max_frames_per_event", "data"),
        liquid_hidden_dim=_positive_int(data_raw, "liquid_hidden_dim", "data"),
        l1_task_fact_dim=_positive_int(data_raw, "l1_task_fact_dim", "data"),
        l2_task_fact_dim=_positive_int(data_raw, "l2_task_fact_dim", "data"),
        event_fact_dim=_positive_int(data_raw, "event_fact_dim", "data"),
        l1_class_count=_positive_int(data_raw, "l1_class_count", "data"),
        l2_class_count=_positive_int(data_raw, "l2_class_count", "data"),
    )
    model_raw = _mapping(payload, "model", "root")
    model = ModelConfig(
        knownness_projection_dim=_positive_int(model_raw, "knownness_projection_dim", "model"),
        trajectory_hidden_dim=_positive_int(model_raw, "trajectory_hidden_dim", "model"),
        raw_scale_feature_dim=_positive_int(model_raw, "raw_scale_feature_dim", "model"),
        stable8_feature_dim=_positive_int(model_raw, "stable8_feature_dim", "model"),
    )
    training_raw = _mapping(payload, "training", "root")
    seeds = _int_tuple(training_raw, "seeds", "training")
    if any(seed < 0 for seed in seeds):
        raise ValueError("V27d training seeds must be non-negative")
    training = TrainingConfig(
        seeds=seeds,
        dataloader_seed=_nonnegative_int(training_raw, "dataloader_seed", "training"),
        batch_size=_positive_int(training_raw, "batch_size", "training"),
        epochs=_positive_int(training_raw, "epochs", "training"),
        learning_rate=_finite_float(training_raw, "learning_rate", "training", lower=0.0),
        weight_decay=_finite_float(training_raw, "weight_decay", "training", lower=0.0),
        gradient_clip_norm=_finite_float(training_raw, "gradient_clip_norm", "training", lower=0.0),
        device=_text(training_raw, "device", "training").lower(),
    )
    if training.device not in {"cpu", "cuda"}:
        raise ValueError("V27d training.device must be cpu or cuda")
    calibration_raw = _mapping(payload, "calibration", "root")
    calibration = CalibrationConfig(
        minimum_quality_events=_positive_int(calibration_raw, "minimum_quality_events", "calibration"),
        minimum_quality_rd_fraction=_finite_float(calibration_raw, "minimum_quality_rd_fraction", "calibration", lower=0.0, upper=1.0),
        minimum_quality_dt_fraction=_finite_float(calibration_raw, "minimum_quality_dt_fraction", "calibration", lower=0.0, upper=1.0),
        minimum_quality_aux_fraction=_finite_float(calibration_raw, "minimum_quality_aux_fraction", "calibration", lower=0.0, upper=1.0),
        known_target_recall_floor=_finite_float(calibration_raw, "known_target_recall_floor", "calibration", lower=0.0, upper=1.0),
        clutter_recall_floor=_finite_float(calibration_raw, "clutter_recall_floor", "calibration", lower=0.0, upper=1.0),
        max_correct_known_new_rejections=_nonnegative_int(calibration_raw, "max_correct_known_new_rejections", "calibration"),
        max_per_class_correct_known_loss=_nonnegative_int(calibration_raw, "max_per_class_correct_known_loss", "calibration"),
        conformal_error_rate=_finite_float(calibration_raw, "conformal_error_rate", "calibration", lower=0.0, upper=1.0),
        minimum_calibration_unknown_support_for_formal=_positive_int(calibration_raw, "minimum_calibration_unknown_support_for_formal", "calibration"),
    )
    artifacts_raw = _mapping(payload, "artifacts", "root")
    artifacts = ArtifactConfig(**{key: _text(artifacts_raw, key, "artifacts") for key in ArtifactConfig.__dataclass_fields__})
    protocol = _mapping(payload, "protocol", "root")
    expected_protocol = {
        "unknown_is_l2_classifier_class": False,
        "clutter_used_for_knownness_optimization": False,
        "low_quality_pending_before_unknown": True,
        "fixed_softmax_threshold_allowed": False,
        "single_global_distance_gate_allowed": False,
        "l3_fail_closed_before_calibration": True,
        "test_payload_access_count": 0,
        "oof": "not_executed",
        "cross_source": "not_executed",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27d protocol.{key} differs")
    return V27DConfig(
        schema_version=V27D_SCHEMA_VERSION,
        experiment_id=V27D_EXPERIMENT_ID,
        variant=_text(payload, "variant", "root"),
        objective=_text(payload, "objective", "root"),
        source=source,
        data=data,
        model=model,
        training=training,
        calibration=calibration,
        artifacts=artifacts,
        protocol=protocol,
        config_path=path,
        config_sha256=sha256_file(path),
        config_identity_sha256=canonical_sha256(payload),
    )


def project_path(root: Path, value: str | Path, *, must_exist: bool = False) -> Path:
    return resolve_project_path(root, value, must_exist=must_exist)


def artifact_path(root: Path, template: str, **values: Any) -> Path:
    try:
        rendered = template.format(**values)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid V27d artifact template: {template}") from exc
    path = resolve_project_path(root, rendered)
    artifact_root = resolve_project_path(root, V27D_ARTIFACT_ROOT)
    try:
        path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValueError("V27d artifact escaped its experiment directory") from exc
    return path


def verify_config_sources(root: Path, config: V27DConfig) -> dict[str, str]:
    paths = {
        "baseline_config": config.source.baseline_config_path,
        "baseline_manifest": config.source.baseline_manifest_path,
        "pool_registry": config.source.baseline_pool_registry_path,
        "feature_manifest": config.source.baseline_feature_manifest_path,
        "representation_manifest": config.source.baseline_representation_manifest_path,
        "frozen_binding": config.source.frozen_binding_path,
        "l1_checkpoint": config.source.baseline_l1_checkpoint_path,
        "l2_checkpoint": config.source.baseline_l2_checkpoint_path,
        "v27a_config": config.source.v27a_config_path,
        "v27a_checkpoint": config.source.v27a_main_checkpoint_path,
        "v27a_train_cache": config.source.v27a_train_cache_path,
        "v27a_validation_cache": config.source.v27a_validation_cache_path,
        "v27a_normalization": config.source.v27a_normalization_path,
        "index": config.source.index_path,
        "v16_provenance": config.source.v16_provenance_path,
        "canonical_frame_index": config.source.canonical_frame_index_path,
    }
    observed: dict[str, str] = {}
    for key, value in paths.items():
        path = project_path(root, value, must_exist=True)
        observed[key] = sha256_file(path)
        expected = config.source.expected_hashes.get(key)
        if expected is not None and observed[key] != expected:
            raise ValueError(f"V27d source hash differs for {key}: expected={expected}, actual={observed[key]}")
    if set(config.source.expected_hashes) != set(paths):
        raise ValueError("V27d source expected-hash keys differ from declared source files")
    return observed


def build_manifest(root: Path, config: V27DConfig, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    # The root contract covers only the shared V27d pipeline.  Each nested
    # experiment (r5, r5_v2, r6, r7, ...) owns and hashes its package in its
    # own manifest.  Including nested packages here would make adding an
    # independent experiment invalidate every previously materialized root
    # endpoint/cache artifact.
    artifact_root = root / V27D_ARTIFACT_ROOT
    source_files = sorted(
        path
        for path in artifact_root.glob("*.py")
        if "__pycache__" not in path.parts
    )
    code_hashes = {relative_project_path(root, path): sha256_file(path) for path in source_files}
    return {
        "schema_version": V27D_MANIFEST_SCHEMA,
        "experiment_id": V27D_EXPERIMENT_ID,
        "variant": config.variant,
        "objective": config.objective,
        "config": {
            "path": relative_project_path(root, config.config_path),
            "sha256": config.config_sha256,
            "identity_sha256": config.config_identity_sha256,
        },
        "source_hashes": dict(source_hashes),
        "code_hashes": code_hashes,
        "protocol": dict(config.protocol),
        "training": {
            "seeds": list(config.training.seeds),
            "epochs": config.training.epochs,
            "device": config.training.device,
        },
        "calibration": {
            "unknown_support": "internal_exploratory_until_minimum_support",
            "risk_constrained": True,
        },
    }


def seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27d payload already has payload_sha256")
    result = dict(payload)
    result["payload_sha256"] = canonical_sha256(payload)
    return result


def read_sealed_json(path: str | Path, *, expected_schema: str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27d sealed JSON must be an object")
    declared = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError(f"V27d sealed JSON hash differs: {path}")
    if expected_schema is not None and payload.get("schema_version") != expected_schema:
        raise ValueError("V27d sealed JSON schema differs")
    return dict(payload)


def write_sealed_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(target, json_text(seal(payload), indent=2))
    return target


def prepare_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[V27DConfig, Path, str]:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    sources = verify_config_sources(root, config)
    manifest = build_manifest(root, config, sources)
    target = artifact_path(root, config.artifacts.manifest_path)
    write_sealed_json(target, manifest)
    return config, target, sha256_file(target)


def load_context(project_root: str | Path, config_path: str | Path | None = None) -> V27DContext:
    root = Path(project_root).resolve()
    config = load_config(root, config_path)
    source_hashes = verify_config_sources(root, config)
    path = artifact_path(root, config.artifacts.manifest_path)
    manifest = read_sealed_json(path, expected_schema=V27D_MANIFEST_SCHEMA)
    expected = build_manifest(root, config, source_hashes)
    actual = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual != expected:
        raise ValueError("V27d manifest differs from configuration, sources or implementation")
    return V27DContext(root, config, manifest, path, sha256_file(path))


__all__ = [
    "V27D_ARTIFACT_ROOT",
    "V27D_DEFAULT_CONFIG",
    "V27D_EXPERIMENT_ID",
    "V27D_KNOWN_LABELS",
    "V27D_LABELS",
    "V27D_MANIFEST_SCHEMA",
    "V27D_ROLES",
    "V27D_SCHEMA_VERSION",
    "V27D_STABLE8_INDICES",
    "V27D_STABLE8_NAMES",
    "V27D_TASKS",
    "V27D_VARIANTS",
    "V27DConfig",
    "V27DContext",
    "artifact_path",
    "build_manifest",
    "load_config",
    "load_context",
    "prepare_manifest",
    "project_path",
    "read_sealed_json",
    "verify_config_sources",
    "write_sealed_json",
]
