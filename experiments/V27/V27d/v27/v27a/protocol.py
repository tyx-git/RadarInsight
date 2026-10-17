"""Configuration, artifact, and provenance contracts for V27a.

V27a is intentionally a separate package.  It consumes the immutable V27 L2
classification base and never writes to the V27 manifest or L3 registry.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.runtime import canonical_sha256, relative_project_path, resolve_project_path, sha256_file

from v27.v27_baseline.base_binding import load_frozen_base_binding
from v27.v27_baseline.protocol import load_v27_config


V27A_EXPERIMENT_ID = "V27a"
V27A_SCHEMA_VERSION = "RADAR-LN3-V27a-micro-doppler-evidence-decoupling-v1"
V27A_CONFIG_SCHEMA_VERSION = "RADAR-LN3-V27a-config-v4"
V27A_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27a-manifest-v4"
V27A_NORMALIZATION_SCHEMA_VERSION = (
    "RADAR-LN3-V27a-stable8-parent-balanced-normalization-v1"
)
V27A_CACHE_SCHEMA_VERSION = "RADAR-LN3-V27a-frozen-logit-evidence-cache-v2"

V27A_L2_TASK_NAME = "dbb_specialist"
V27A_L2_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
V27A_ARMS = ("mechanism", "main")
V27A_STABLE8_INDICES = (1, 2, 3, 7, 11, 13, 14, 22)
V27A_STABLE8_NAMES = (
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
class V27ABaseConfig:
    v27_config_path: str
    frozen_binding_path: str
    expected_current_v27_config_sha256: str
    expected_checkpoint_v27_config_sha256: str
    expected_binding_payload_sha256: str
    expected_l2_checkpoint_sha256: str
    expected_l2_classifier_state_sha256: str
    expected_l2_epoch: int


@dataclass(frozen=True)
class V27ADataConfig:
    index_path: str
    train_evidence_path: str
    validation_evidence_path: str
    train_summary_path: str
    validation_summary_path: str
    expected_index_sha256: str
    expected_train_evidence_sha256: str
    expected_validation_evidence_sha256: str
    expected_train_summary_sha256: str
    expected_validation_summary_sha256: str
    max_sequence_length: int
    max_frames_per_event: int
    stable8_indices: tuple[int, ...]
    stable8_names: tuple[str, ...]


@dataclass(frozen=True)
class V27AModelConfig:
    evidence_dim: int
    hidden_dim: int
    class_count: int
    residual_bound: float
    token_activation: str
    frame_gru_layers: int
    parent_gru_layers: int


@dataclass(frozen=True)
class V27ATrainingConfig:
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


@dataclass(frozen=True)
class V27AArtifactConfig:
    manifest_path: str
    audit_report_path: str
    normalization_path: str
    train_cache_path: str
    validation_cache_path: str
    smoke_report_path: str
    training_log_template: str
    training_summary_template: str
    epoch_checkpoint_template: str
    evaluation_report_template: str
    paired_evaluation_report_path: str


@dataclass(frozen=True)
class V27AConfig:
    schema_version: str
    experiment_id: str
    variant: str
    objective: str
    base: V27ABaseConfig
    data: V27ADataConfig
    model: V27AModelConfig
    training: V27ATrainingConfig
    artifacts: V27AArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


@dataclass(frozen=True)
class V27AContext:
    project_root: Path
    config: V27AConfig
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    frozen_binding: dict[str, Any]
    v27_config: Any


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27a {context} is missing {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27a {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27a {context}.{key} must be a non-empty string")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"V27a {context}.{key} must be a positive integer")
    if value <= 0:
        raise ValueError(f"V27a {context}.{key} must be a positive integer")
    return value


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
    *,
    positive: bool = False,
) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ValueError(f"V27a {context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27a {context}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27a {context}.{key} has an invalid numeric value")
    return result


def _sha(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(mapping, key, context).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"V27a {context}.{key} must be a SHA-256 digest")
    return value


def _int_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[int, ...]:
    value = _required(mapping, key, context)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"V27a {context}.{key} must be a non-empty list")
    result_values: list[int] = []
    for item in value:
        if isinstance(item, bool):
            raise ValueError(
                f"V27a {context}.{key} must contain integers, not booleans"
            )
        if not isinstance(item, int):
            raise ValueError(f"V27a {context}.{key} must contain integers")
        result_values.append(item)
    result = tuple(result_values)
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"V27a {context}.{key} must contain unique non-negative integers")
    return result


def _text_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = _required(mapping, key, context)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"V27a {context}.{key} must be a non-empty list")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError(f"V27a {context}.{key} must contain unique non-empty strings")
    return result


def load_v27a_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> V27AConfig:
    root = Path(project_root).resolve()
    candidate = (
        root
        / "src-v3"
        / "v27"
        / "v27a"
        / "config"
        / "V27a-micro-doppler-evidence-decoupling.json"
        if config_path is None
        else Path(config_path)
    )
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("V27a configuration must remain inside project root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27a configuration must be a JSON object")
    if payload.get("schema_version") != V27A_CONFIG_SCHEMA_VERSION:
        raise ValueError("V27a configuration schema differs")
    if payload.get("experiment_id") != V27A_EXPERIMENT_ID:
        raise ValueError("V27a experiment_id must be exactly V27a")

    base_raw = _mapping(payload, "base", "root")
    base = V27ABaseConfig(
        v27_config_path=_text(base_raw, "v27_config_path", "base"),
        frozen_binding_path=_text(base_raw, "frozen_binding_path", "base"),
        expected_current_v27_config_sha256=_sha(
            base_raw, "expected_current_v27_config_sha256", "base"
        ),
        expected_checkpoint_v27_config_sha256=_sha(
            base_raw, "expected_checkpoint_v27_config_sha256", "base"
        ),
        expected_binding_payload_sha256=_sha(
            base_raw, "expected_binding_payload_sha256", "base"
        ),
        expected_l2_checkpoint_sha256=_sha(
            base_raw, "expected_l2_checkpoint_sha256", "base"
        ),
        expected_l2_classifier_state_sha256=_sha(
            base_raw, "expected_l2_classifier_state_sha256", "base"
        ),
        expected_l2_epoch=_positive_int(base_raw, "expected_l2_epoch", "base"),
    )

    data_raw = _mapping(payload, "data", "root")
    data = V27ADataConfig(
        index_path=_text(data_raw, "index_path", "data"),
        train_evidence_path=_text(data_raw, "train_evidence_path", "data"),
        validation_evidence_path=_text(
            data_raw, "validation_evidence_path", "data"
        ),
        train_summary_path=_text(data_raw, "train_summary_path", "data"),
        validation_summary_path=_text(
            data_raw, "validation_summary_path", "data"
        ),
        expected_index_sha256=_sha(data_raw, "expected_index_sha256", "data"),
        expected_train_evidence_sha256=_sha(
            data_raw, "expected_train_evidence_sha256", "data"
        ),
        expected_validation_evidence_sha256=_sha(
            data_raw, "expected_validation_evidence_sha256", "data"
        ),
        expected_train_summary_sha256=_sha(
            data_raw, "expected_train_summary_sha256", "data"
        ),
        expected_validation_summary_sha256=_sha(
            data_raw, "expected_validation_summary_sha256", "data"
        ),
        max_sequence_length=_positive_int(
            data_raw, "max_sequence_length", "data"
        ),
        max_frames_per_event=_positive_int(
            data_raw, "max_frames_per_event", "data"
        ),
        stable8_indices=_int_tuple(data_raw, "stable8_indices", "data"),
        stable8_names=_text_tuple(data_raw, "stable8_names", "data"),
    )
    if data.stable8_indices != V27A_STABLE8_INDICES:
        raise ValueError("V27a stable8 indices differ from the frozen V18 contract")
    if data.stable8_names != V27A_STABLE8_NAMES:
        raise ValueError("V27a stable8 names differ from the frozen V18 contract")
    if len(data.stable8_indices) != len(data.stable8_names):
        raise ValueError("V27a stable8 index/name widths differ")

    model_raw = _mapping(payload, "model", "root")
    model = V27AModelConfig(
        evidence_dim=_positive_int(model_raw, "evidence_dim", "model"),
        hidden_dim=_positive_int(model_raw, "hidden_dim", "model"),
        class_count=_positive_int(model_raw, "class_count", "model"),
        residual_bound=_finite_float(
            model_raw, "residual_bound", "model", positive=True
        ),
        token_activation=_text(model_raw, "token_activation", "model"),
        frame_gru_layers=_positive_int(model_raw, "frame_gru_layers", "model"),
        parent_gru_layers=_positive_int(model_raw, "parent_gru_layers", "model"),
    )
    if (
        model.evidence_dim != 8
        or model.class_count != len(V27A_L2_CLASS_NAMES)
        or model.token_activation != "silu"
        or model.frame_gru_layers != 1
        or model.parent_gru_layers != 1
    ):
        raise ValueError("V27a model contract differs from the registered design")

    training_raw = _mapping(payload, "training", "root")
    raw_seeds = _required(training_raw, "seeds", "training")
    if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds:
        raise ValueError("V27a training.seeds must be a non-empty list")
    seeds = tuple(int(item) for item in raw_seeds)
    if any(item < 0 for item in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("V27a training.seeds must be unique non-negative integers")
    training = V27ATrainingConfig(
        seeds=seeds,
        dataloader_seed=_positive_int(
            training_raw, "dataloader_seed", "training"
        ),
        batch_size=_positive_int(training_raw, "batch_size", "training"),
        epochs=_positive_int(training_raw, "epochs", "training"),
        primary_epoch=_positive_int(
            training_raw, "primary_epoch", "training"
        ),
        learning_rate=_finite_float(
            training_raw, "learning_rate", "training", positive=True
        ),
        weight_decay=_finite_float(training_raw, "weight_decay", "training"),
        gradient_clip_norm=_finite_float(
            training_raw, "gradient_clip_norm", "training", positive=True
        ),
        device=_text(training_raw, "device", "training").lower(),
        eligible_prefix_min_events=_positive_int(
            training_raw, "eligible_prefix_min_events", "training"
        ),
        bootstrap_replicates=_positive_int(
            training_raw, "bootstrap_replicates", "training"
        ),
    )
    if training.primary_epoch > training.epochs:
        raise ValueError("V27a primary_epoch cannot exceed epochs")
    if training.device not in {"cpu", "cuda"}:
        raise ValueError("V27a training.device must be cpu or cuda")
    if training.weight_decay < 0.0:
        raise ValueError("V27a weight_decay cannot be negative")
    if training.eligible_prefix_min_events < 1:
        raise ValueError("V27a eligible prefix minimum must be positive")
    if training.bootstrap_replicates < 100:
        raise ValueError("V27a bootstrap_replicates must be at least 100")

    artifact_raw = _mapping(payload, "artifacts", "root")
    artifact_fields = {
        key: _text(artifact_raw, key, "artifacts")
        for key in V27AArtifactConfig.__dataclass_fields__
    }
    artifacts = V27AArtifactConfig(**artifact_fields)
    artifact_scope = (root / "src-v3" / "v27" / "v27a").resolve()
    for field_name, template in artifact_fields.items():
        try:
            probe = template.format(arm="main", seed=0, split="val", epoch=1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"V27a artifacts.{field_name} contains an unsupported placeholder"
            ) from exc
        resolved_probe = resolve_project_path(root, probe)
        try:
            resolved_probe.relative_to(artifact_scope)
        except ValueError as exc:
            raise ValueError(
                f"V27a artifacts.{field_name} must remain inside src-v3/v27/v27a"
            ) from exc
    for template in (
        artifacts.training_log_template,
        artifacts.training_summary_template,
        artifacts.epoch_checkpoint_template,
    ):
        if "{arm}" not in template or "{seed}" not in template:
            raise ValueError("V27a run artifact templates must contain arm and seed")
    if (
        "{seed}" not in artifacts.evaluation_report_template
        or "{arm}" not in artifacts.evaluation_report_template
    ):
        raise ValueError("V27a evaluation report template must contain arm and seed")
    if "{" in artifacts.paired_evaluation_report_path or "}" in artifacts.paired_evaluation_report_path:
        raise ValueError("V27a paired evaluation report path must be a concrete path")

    protocol = _mapping(payload, "protocol", "root")
    expected_protocol = {
        "frozen_base": "V27_L2_epoch_11",
        "unknown_training_exposure": 0,
        "l3": "not_in_scope",
        "shared_fusion_reentry": "forbidden",
        "backbone_unfreeze": "forbidden",
        "input_variable": "reliable_stable8_token_count_only",
        "summary_and_frame_encoder_capacity": "identical",
        "online_chunk_boundary": "event_boundary_only",
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27a protocol.{key} differs")
    if int(protocol.get("unknown_training_exposure", -1)) != 0:
        raise ValueError("V27a cannot expose Unknown to residual classification loss")

    return V27AConfig(
        schema_version=V27A_SCHEMA_VERSION,
        experiment_id=V27A_EXPERIMENT_ID,
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






def verify_sealed_payload(
    payload: Mapping[str, Any],
    *,
    schema: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("V27a sealed artifact must be an object")
    declared = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError("V27a sealed artifact payload hash differs")
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError("V27a sealed artifact schema differs")
    return dict(payload)






def read_sealed_json(path: str | Path, *, schema: str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return verify_sealed_payload(payload, schema=schema)


def _source_paths() -> tuple[Path, ...]:
    return (
        Path("src-v3/v27/v27a/__init__.py"),
        Path("src-v3/v27/v27a/protocol.py"),
        Path("src-v3/v27/v27a/evidence.py"),
        Path("src-v3/v27/v27a/cache.py"),
        Path("src-v3/v27/v27a/model.py"),
        Path("src-v3/v27/v27a/checkpoint.py"),
        Path("src-v3/v27/v27a/runtime.py"),
        Path("src-v3/v27/v27a/logging_utils.py"),
        Path("src-v3/v27/v27a/evaluation.py"),
        Path("src-v3/v27/v27a/training.py"),
        Path("src-v3/v27/v27a/cli.py"),
    )


def _binding_file_path(root: Path, config: V27AConfig) -> Path:
    return resolve_project_path(root, config.base.frozen_binding_path, must_exist=True)


def build_v27a_manifest(
    project_root: str | Path,
    config: V27AConfig,
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    binding_path = _binding_file_path(root, config)
    l2 = binding.get("l2")
    if not isinstance(l2, Mapping):
        raise ValueError("V27a frozen binding lacks l2")
    audit_path = resolve_project_path(root, config.artifacts.audit_report_path, must_exist=True)
    normalization_path = resolve_project_path(
        root, config.artifacts.normalization_path, must_exist=True
    )
    evidence_paths = {
        "train": resolve_project_path(root, config.data.train_evidence_path, must_exist=True),
        "val": resolve_project_path(root, config.data.validation_evidence_path, must_exist=True),
        "train_summary": resolve_project_path(root, config.data.train_summary_path, must_exist=True),
        "val_summary": resolve_project_path(root, config.data.validation_summary_path, must_exist=True),
    }
    code_hashes = {
        path.as_posix(): sha256_file(root / path) for path in _source_paths()
    }
    manifest = {
        "schema_version": V27A_MANIFEST_SCHEMA_VERSION,
        "experiment_id": V27A_EXPERIMENT_ID,
        "variant": config.variant,
        "objective": config.objective,
        "config": {
            "path": relative_project_path(root, config.config_path),
            "sha256": config.config_sha256,
            "identity_sha256": config.config_identity_sha256,
        },
        "frozen_base": {
            "binding_path": relative_project_path(root, binding_path),
            "binding_file_sha256": sha256_file(binding_path),
            "binding_payload_sha256": str(binding.get("binding_sha256", "")),
            "l2_checkpoint_path": str(l2.get("checkpoint_path", "")),
            "l2_checkpoint_sha256": str(l2.get("checkpoint_sha256", "")),
            "l2_classifier_state_sha256": str(l2.get("classifier_state_sha256", "")),
            "l2_epoch": int(l2.get("epoch", -1)),
            "v27_config_sha256": str(l2.get("config_sha256", "")),
            "v27_manifest_sha256": str(l2.get("manifest_sha256", "")),
        },
        "code_hashes": code_hashes,
        "data": {
            "index_path": config.data.index_path,
            "index_sha256": config.data.expected_index_sha256,
            "train_evidence_path": config.data.train_evidence_path,
            "train_evidence_sha256": config.data.expected_train_evidence_sha256,
            "validation_evidence_path": config.data.validation_evidence_path,
            "validation_evidence_sha256": config.data.expected_validation_evidence_sha256,
            "train_summary_path": config.data.train_summary_path,
            "train_summary_sha256": config.data.expected_train_summary_sha256,
            "validation_summary_path": config.data.validation_summary_path,
            "validation_summary_sha256": config.data.expected_validation_summary_sha256,
            "stable8_indices": list(config.data.stable8_indices),
            "stable8_names": list(config.data.stable8_names),
            "max_sequence_length": config.data.max_sequence_length,
            "max_frames_per_event": config.data.max_frames_per_event,
            "observed_file_sha256": {
                name: sha256_file(path) for name, path in evidence_paths.items()
            },
        },
        "model": {
            "evidence_dim": config.model.evidence_dim,
            "hidden_dim": config.model.hidden_dim,
            "class_count": config.model.class_count,
            "residual_bound": config.model.residual_bound,
            "token_activation": config.model.token_activation,
            "frame_gru_layers": config.model.frame_gru_layers,
            "parent_gru_layers": config.model.parent_gru_layers,
            "summary_and_main_parameter_shapes_identical": True,
            "residual_projection_bias": False,
            "residual_zero_initialization": True,
            "residual_zero_sum": True,
        },
        "training": {
            "seeds": list(config.training.seeds),
            "dataloader_seed": config.training.dataloader_seed,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "primary_epoch": config.training.primary_epoch,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "gradient_clip_norm": config.training.gradient_clip_norm,
            "loss": "parent_final_fixed_parent_count_ce",
            "class_weight_source": "frozen V27 L2 train-only weights",
        },
        "audit_report_sha256": sha256_file(audit_path),
        "normalization_sha256": sha256_file(normalization_path),
        "protocol": dict(config.protocol),
        "scope": {
            "unknown": "sealed_and_not_used",
            "test": "sealed_and_not_used",
            "l1": "not_in_scope",
            "l3": "not_in_scope",
            "cross_source": "not_in_scope",
            "matched_retraining": "not_proven_by_this_experiment",
        },
    }
    return manifest


def verify_v27a_binding(
    root: Path,
    config: V27AConfig,
) -> tuple[dict[str, Any], Any]:
    """Verify the immutable V27 binding and the referenced V27 configuration."""

    binding_path = _binding_file_path(root, config)
    binding = load_frozen_base_binding(root, binding_path)
    if binding.get("binding_sha256") != config.base.expected_binding_payload_sha256:
        raise ValueError("V27a frozen binding payload hash differs")
    l2 = binding.get("l2")
    if not isinstance(l2, Mapping):
        raise ValueError("V27a frozen binding lacks L2 record")
    if (
        str(l2.get("checkpoint_sha256")) != config.base.expected_l2_checkpoint_sha256
        or str(l2.get("classifier_state_sha256"))
        != config.base.expected_l2_classifier_state_sha256
        or int(l2.get("epoch", -1)) != config.base.expected_l2_epoch
        or int(l2.get("epoch", -1)) != 11
        or str(l2.get("task")) != V27A_L2_TASK_NAME
    ):
        raise ValueError("V27a frozen L2 binding differs from epoch-11 contract")
    v27_config_path = resolve_project_path(root, config.base.v27_config_path, must_exist=True)
    v27_config = load_v27_config(root, v27_config_path)
    if v27_config.config_sha256 != config.base.expected_current_v27_config_sha256:
        raise ValueError("V27a referenced V27 configuration hash differs")
    if str(l2.get("config_sha256")) != config.base.expected_checkpoint_v27_config_sha256:
        raise ValueError("V27a frozen L2 checkpoint config hash differs")
    return binding, v27_config




def load_v27a_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> V27AContext:
    root = Path(project_root).resolve()
    config = load_v27a_config(root, config_path)
    binding, v27_config = verify_v27a_binding(root, config)
    manifest_path = resolve_project_path(root, config.artifacts.manifest_path, must_exist=True)
    manifest = read_sealed_json(manifest_path, schema=V27A_MANIFEST_SCHEMA_VERSION)
    expected = build_v27a_manifest(root, config, binding)
    actual_body = {
        key: value for key, value in manifest.items() if key != "payload_sha256"
    }
    if actual_body != expected:
        raise ValueError(
            "V27a manifest differs from the current configuration, frozen base, "
            "data inputs, or implementation"
        )
    return V27AContext(
        project_root=root,
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
        frozen_binding=binding,
        v27_config=v27_config,
    )




__all__ = ["V27A_ARMS", "V27A_CACHE_SCHEMA_VERSION", "V27A_L2_CLASS_NAMES", "V27A_CONFIG_SCHEMA_VERSION", "V27AContext", "V27AConfig", "V27A_EXPERIMENT_ID", "V27A_MANIFEST_SCHEMA_VERSION", "V27A_NORMALIZATION_SCHEMA_VERSION", "V27A_SCHEMA_VERSION", "V27A_STABLE8_INDICES", "V27A_STABLE8_NAMES", "build_v27a_manifest", "load_v27a_config", "load_v27a_context", "read_sealed_json", "verify_sealed_payload", "verify_v27a_binding"]