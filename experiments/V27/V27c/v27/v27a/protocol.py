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

from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file



V27A_EXPERIMENT_ID = "V27a"
V27A_SCHEMA_VERSION = "RADAR-LN3-V27a-micro-doppler-evidence-decoupling-v1"
V27A_CONFIG_SCHEMA_VERSION = "RADAR-LN3-V27a-config-v4"
V27A_CACHE_SCHEMA_VERSION = "RADAR-LN3-V27a-frozen-logit-evidence-cache-v2"

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




























__all__ = ["V27A_ARMS", "V27A_CACHE_SCHEMA_VERSION", "V27A_L2_CLASS_NAMES", "V27A_CONFIG_SCHEMA_VERSION", "V27AConfig", "V27A_EXPERIMENT_ID", "V27A_SCHEMA_VERSION", "V27A_STABLE8_INDICES", "V27A_STABLE8_NAMES", "load_v27a_config"]