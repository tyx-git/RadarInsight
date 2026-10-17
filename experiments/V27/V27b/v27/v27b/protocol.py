"""Configuration and provenance contracts for the isolated V27b experiment.

V27b consumes a frozen V27a main checkpoint and its known-class feature
caches.  It trains one small, zero-initialised correction head and never
opens Unknown or test payloads.  The old paths embedded in the archived V27a
artifacts are part of the input provenance, not paths that this package
resolves at runtime.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    parse_bool,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)


V27B_EXPERIMENT_ID = "V27b"
V27B_CONFIG_SCHEMA_VERSION = "RADAR-LN3-V27b-config-v2"
V27B_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27b-manifest-v2"
V27B_CACHE_SCHEMA_VERSION = "RADAR-LN3-V27b-frozen-v27a-feature-cache-v2"
V27B_DIRECT_FRAME_CACHE_SCHEMA_VERSION = "RADAR-LN3-V27b-direct-frame-feature-cache-v3"
V27B_CHECKPOINT_SCHEMA_VERSION = "RADAR-LN3-V27b-conditioned-pair-residual-checkpoint-v3"
V27B_REPORT_SCHEMA_VERSION = "RADAR-LN3-V27b-paired-validation-report-v3"
V27B_SMOKE_SCHEMA_VERSION = "RADAR-LN3-V27b-contract-smoke-v3"
V27B_SCHEMA_VERSION = V27B_CONFIG_SCHEMA_VERSION

V27B_L2_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
V27B_TREATMENT_ARM = "conditional"
V27B_ARMS = (V27B_TREATMENT_ARM,)
V27B_ALL_ARMS = ("control", *V27B_ARMS)
V27B_EVIDENCE_CARRIERS = (
    "frozen_v27a_event_representation",
    "direct_frame_gru",
)
V27B_STABLE8_INDICES = (1, 2, 3, 7, 11, 13, 14, 22)
V27B_STABLE8_NAMES = (
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
_EXPECTED_PARENT_LABEL_COUNTS = {
    "train": {"DroneTarget": 476, "BirdTarget": 239, "BalloonTarget": 238},
    "val": {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54},
}


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


@dataclass(frozen=True)
class V27BContext:
    project_root: Path
    config: V27BConfig
    input_binding: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27b {context} is missing {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27b {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27b {context}.{key} must be a non-empty string")
    return value.strip()


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27b {context}.{key} must be a positive integer")
    return int(value)


def _nonnegative_int_or_default(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
    *,
    default: int,
) -> int:
    if key not in mapping:
        return int(default)
    return _nonnegative_int(mapping, key, context)


def _text_or_default(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
    *,
    default: str,
) -> str:
    if key not in mapping:
        return str(default)
    return _text(mapping, key, context)


def _nonnegative_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"V27b {context}.{key} must be a non-negative integer")
    return int(value)


def _expected_protocol(evidence_carrier: str) -> dict[str, Any]:
    """Return the exact protocol registered for one V27b evidence carrier.

    The compressed r1--r4 arm and the direct-frame r5 arm intentionally share
    the same experiment namespace, but they are different model contracts.
    Keeping the protocol as a carrier-specific closed dictionary prevents a
    configuration from silently mixing the two representations.
    """

    common = {
        "frozen_control": "V27a_main_epoch_20",
        "frozen_v27a_encoder": True,
        "new_correction_pair": "DroneTarget__vs__BirdTarget",
        "unknown_training_exposure": 0,
        "unknown": "sealed_and_not_used",
        "test": "sealed_and_not_used",
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "l3": "not_in_scope",
        "liquid_v2_backbone": "frozen_indirectly_through_v27a_checkpoint",
        "v27a_control_output_must_remain_unchanged": True,
        "no_reliable_event_new_correction": (
            "exact_zero_correction_and_increment_preserve_state"
        ),
        "loss": "parent_final_fixed_parent_count_ce",
        "checkpoint_selection": "fixed_primary_epoch; own_best_diagnostic_only",
    }
    if evidence_carrier == "frozen_v27a_event_representation":
        return {
            **common,
            "trainable_component": (
                "baseline_conditioned_drone_bird_pair_residual_head_only"
            ),
            "condition_inputs": [
                "causal_v27a_parent_state",
                "causal_v27a_event_representation",
                "frozen_l2_logits",
                "frozen_l2_pairwise_margins",
                "frozen_v27a_residual",
            ],
        }
    if evidence_carrier == "direct_frame_gru":
        return {
            **common,
            "trainable_component": (
                "baseline_conditioned_direct_frame_gru_and_drone_bird_pair_residual_head"
            ),
            "condition_inputs": [
                "causal_v27a_parent_state",
                "causal_v27a_event_representation",
                "frozen_l2_logits",
                "frozen_l2_pairwise_margins",
                "frozen_v27a_residual",
                "event_frame_values",
                "event_frame_mask",
            ],
            "evidence_carrier": "direct_frame_gru",
            "frame_encoder": "mask_aware_event_local_gru",
            "frame_encoder_input": "normalized_stable8",
            "frame_encoder_layers": 1,
            "frame_sequence_scope": "within_event_only",
        }
    raise ValueError(f"unsupported V27b evidence carrier: {evidence_carrier}")


def _finite_float(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
    *,
    positive: bool = False,
) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ValueError(f"V27b {context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27b {context}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27b {context}.{key} has an invalid numeric value")
    return result


def _sha(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(mapping, key, context).lower()
    if len(value) != 64 or any(char not in _SHA256_HEX for char in value):
        raise ValueError(f"V27b {context}.{key} must be a SHA-256 digest")
    return value




def _class_weights(mapping: Mapping[str, Any], key: str, context: str) -> tuple[float, ...]:
    value = _required(mapping, key, context)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"V27b {context}.{key} must be a non-empty list")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"V27b {context}.{key} must contain positive finite values")
    return result


def _matrix(mapping: Mapping[str, Any], key: str, context: str) -> tuple[tuple[int, ...], ...]:
    value = _required(mapping, key, context)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"V27b {context}.{key} must be a non-empty matrix")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not row:
            raise ValueError(f"V27b {context}.{key} has an invalid row")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in row):
            raise ValueError(f"V27b {context}.{key} has an invalid count")
        parsed = tuple(int(item) for item in row)
        rows.append(parsed)
    width = len(rows[0])
    if len(rows) != width or any(len(row) != width for row in rows):
        raise ValueError(f"V27b {context}.{key} must be square")
    return tuple(rows)


def load_v27b_config(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> V27BConfig:
    root = Path(project_root).resolve()
    candidate = (
        root / "src-v3" / "v27" / "v27b" / "config" / "V27b-baseline-conditioned-stable8.json"
        if config_path is None
        else Path(config_path)
    )
    path = resolve_project_path(root, candidate)
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27b configuration must be a JSON object")
    if payload.get("schema_version") != V27B_CONFIG_SCHEMA_VERSION:
        raise ValueError("V27b configuration schema differs")
    if payload.get("experiment_id") != V27B_EXPERIMENT_ID:
        raise ValueError("V27b experiment_id must be exactly V27b")

    base_raw = _mapping(payload, "base", "root")
    base = V27BBaseConfig(
        v27a_checkpoint_path=_text(base_raw, "v27a_checkpoint_path", "base"),
        v27a_train_cache_path=_text(base_raw, "v27a_train_cache_path", "base"),
        v27a_validation_cache_path=_text(base_raw, "v27a_validation_cache_path", "base"),
        v27a_runtime_config_path=_text(base_raw, "v27a_runtime_config_path", "base"),
        v27a_runtime_manifest_path=_text(base_raw, "v27a_runtime_manifest_path", "base"),
        expected_v27a_checkpoint_file_sha256=_sha(base_raw, "expected_v27a_checkpoint_file_sha256", "base"),
        expected_v27a_checkpoint_state_sha256=_sha(base_raw, "expected_v27a_checkpoint_state_sha256", "base"),
        expected_v27a_model_contract_sha256=_sha(base_raw, "expected_v27a_model_contract_sha256", "base"),
        expected_v27a_checkpoint_training_config_identity_sha256=_sha(base_raw, "expected_v27a_checkpoint_training_config_identity_sha256", "base"),
        expected_v27a_checkpoint_training_manifest_sha256=_sha(base_raw, "expected_v27a_checkpoint_training_manifest_sha256", "base"),
        expected_v27a_train_cache_file_sha256=_sha(base_raw, "expected_v27a_train_cache_file_sha256", "base"),
        expected_v27a_validation_cache_file_sha256=_sha(base_raw, "expected_v27a_validation_cache_file_sha256", "base"),
        expected_v27a_train_cache_payload_sha256=_sha(base_raw, "expected_v27a_train_cache_payload_sha256", "base"),
        expected_v27a_validation_cache_payload_sha256=_sha(base_raw, "expected_v27a_validation_cache_payload_sha256", "base"),
        expected_v27a_runtime_config_sha256=_sha(base_raw, "expected_v27a_runtime_config_sha256", "base"),
        expected_v27a_runtime_manifest_sha256=_sha(base_raw, "expected_v27a_runtime_manifest_sha256", "base"),
        expected_v27a_runtime_config_identity_sha256=_sha(base_raw, "expected_v27a_runtime_config_identity_sha256", "base"),
        expected_v27a_runtime_manifest_payload_sha256=_sha(base_raw, "expected_v27a_runtime_manifest_payload_sha256", "base"),
        expected_l2_class_weights=_class_weights(base_raw, "expected_l2_class_weights", "base"),
        expected_v27a_epoch=_positive_int(base_raw, "expected_v27a_epoch", "base"),
        expected_v27a_arm=_text(base_raw, "expected_v27a_arm", "base"),
        expected_v27a_seed=_nonnegative_int(base_raw, "expected_v27a_seed", "base"),
    )

    data_raw = _mapping(payload, "data", "root")
    raw_class_names = _required(data_raw, "class_names", "data")
    if not isinstance(raw_class_names, (list, tuple)):
        raise ValueError("V27b data.class_names must be a list")
    class_names = tuple(str(value).strip() for value in raw_class_names)
    if any(not value for value in class_names):
        raise ValueError("V27b data.class_names must contain non-empty names")
    data = V27BDataConfig(
        class_names=class_names,
        stable8_dim=_positive_int(data_raw, "stable8_dim", "data"),
        max_sequence_length=_positive_int(data_raw, "max_sequence_length", "data"),
        max_frames_per_event=_positive_int(data_raw, "max_frames_per_event", "data"),
        feature_cache_schema=_text(data_raw, "feature_cache_schema", "data"),
        expected_v27a_control_validation_confusion_matrix=_matrix(
            data_raw,
            "expected_v27a_control_validation_confusion_matrix",
            "data",
        ),
    )
    if data.class_names != V27B_L2_CLASS_NAMES:
        raise ValueError("V27b class names differ from the frozen DBB contract")
    if data.stable8_dim != 8 or data.feature_cache_schema not in {
        V27B_CACHE_SCHEMA_VERSION,
        V27B_DIRECT_FRAME_CACHE_SCHEMA_VERSION,
    }:
        raise ValueError("V27b stable8/cache schema contract differs")
    if data.expected_v27a_control_validation_confusion_matrix != (
        (109, 4, 0),
        (1, 54, 1),
        (1, 1, 52),
    ):
        raise ValueError("V27b frozen V27a control confusion matrix differs")

    model_raw = _mapping(payload, "model", "root")
    model = V27BModelConfig(
        parent_state_dim=_positive_int(model_raw, "parent_state_dim", "model"),
        event_representation_dim=_positive_int(model_raw, "event_representation_dim", "model"),
        decision_state_dim=_positive_int(model_raw, "decision_state_dim", "model"),
        input_dim=_positive_int(model_raw, "input_dim", "model"),
        hidden_dim=_positive_int(model_raw, "hidden_dim", "model"),
        class_count=_positive_int(model_raw, "class_count", "model"),
        activation=_text(model_raw, "activation", "model"),
        residual_bound=_finite_float(model_raw, "residual_bound", "model", positive=True),
        pairwise_delta_bound=_finite_float(model_raw, "pairwise_delta_bound", "model", positive=True),
        residual_projection_bias=parse_bool(model_raw.get("residual_projection_bias"), field="V27b model.residual_projection_bias"),
        residual_zero_initialization=parse_bool(model_raw.get("residual_zero_initialization"), field="V27b model.residual_zero_initialization"),
        new_correction_pair=_text(model_raw, "new_correction_pair", "model"),
        evidence_carrier=_text_or_default(
            model_raw,
            "evidence_carrier",
            "model",
            default="frozen_v27a_event_representation",
        ),
        frame_encoder_hidden_dim=_nonnegative_int_or_default(
            model_raw,
            "frame_encoder_hidden_dim",
            "model",
            default=0,
        ),
        frame_encoder_layers=_nonnegative_int_or_default(
            model_raw,
            "frame_encoder_layers",
            "model",
            default=0,
        ),
    )
    if model.class_count != len(V27B_L2_CLASS_NAMES) or model.activation != "silu":
        raise ValueError("V27b model class/activation contract differs")
    expected_input_dim = (
        model.parent_state_dim
        + model.event_representation_dim
        + model.decision_state_dim
        + (
            model.frame_encoder_hidden_dim
            if model.evidence_carrier == "direct_frame_gru"
            else 0
        )
    )
    if model.input_dim != expected_input_dim:
        raise ValueError("V27b model input_dim does not match its feature layout")
    if not math.isclose(model.pairwise_delta_bound, 2.0 * model.residual_bound):
        raise ValueError("V27b pairwise delta bound must equal twice residual bound")
    if model.new_correction_pair != "DroneTarget__vs__BirdTarget":
        raise ValueError("V27b correction pair differs")
    if model.residual_projection_bias or not model.residual_zero_initialization:
        raise ValueError("V27b requires a bias-free zero-initialised correction projection")
    if model.evidence_carrier not in V27B_EVIDENCE_CARRIERS:
        raise ValueError("V27b evidence carrier differs")
    if model.evidence_carrier == "frozen_v27a_event_representation":
        if (
            data.feature_cache_schema != V27B_CACHE_SCHEMA_VERSION
            or model.frame_encoder_hidden_dim != 0
            or model.frame_encoder_layers != 0
        ):
            raise ValueError("V27b compressed evidence carrier contract differs")
    elif (
        data.feature_cache_schema != V27B_DIRECT_FRAME_CACHE_SCHEMA_VERSION
        or model.frame_encoder_hidden_dim <= 0
        or model.frame_encoder_layers != 1
    ):
        raise ValueError("V27b direct frame evidence carrier contract differs")

    training_raw = _mapping(payload, "training", "root")
    raw_seeds = _required(training_raw, "seeds", "training")
    if not isinstance(raw_seeds, (list, tuple)) or not raw_seeds:
        raise ValueError("V27b training.seeds must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw_seeds):
        raise ValueError("V27b training.seeds must contain integers")
    seeds = tuple(int(item) for item in raw_seeds)
    if any(item < 0 for item in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("V27b training.seeds must be unique non-negative integers")
    training = V27BTrainingConfig(
        seeds=seeds,
        dataloader_seed=_nonnegative_int(training_raw, "dataloader_seed", "training"),
        batch_size=_positive_int(training_raw, "batch_size", "training"),
        epochs=_positive_int(training_raw, "epochs", "training"),
        primary_epoch=_positive_int(training_raw, "primary_epoch", "training"),
        learning_rate=_finite_float(training_raw, "learning_rate", "training", positive=True),
        weight_decay=_finite_float(training_raw, "weight_decay", "training"),
        gradient_clip_norm=_finite_float(training_raw, "gradient_clip_norm", "training", positive=True),
        device=_text(training_raw, "device", "training").lower(),
        eligible_prefix_min_events=_positive_int(training_raw, "eligible_prefix_min_events", "training"),
        bootstrap_replicates=_positive_int(training_raw, "bootstrap_replicates", "training"),
        loss_name=_text(training_raw, "loss_name", "training"),
        loss_reduction=_text(training_raw, "loss_reduction", "training"),
    )
    if training.primary_epoch > training.epochs or training.device not in {"cpu", "cuda"}:
        raise ValueError("V27b training epoch/device contract differs")
    if training.weight_decay < 0.0 or training.bootstrap_replicates < 100:
        raise ValueError("V27b training numeric contract differs")
    if training.loss_name != "parent_final_fixed_parent_count_ce" or training.loss_reduction != "fixed_parent_count":
        raise ValueError("V27b loss contract differs")

    artifact_raw = _mapping(payload, "artifacts", "root")
    artifact_values = {
        key: _text(artifact_raw, key, "artifacts")
        for key in V27BArtifactConfig.__dataclass_fields__
    }
    artifacts = V27BArtifactConfig(**artifact_values)
    artifact_scope = (root / "src-v3" / "v27" / "v27b").resolve()
    for field_name, template in artifact_values.items():
        try:
            probe = template.format(arm=V27B_TREATMENT_ARM, seed=0, epoch=1)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"V27b artifacts.{field_name} contains an unsupported placeholder") from exc
        resolved = resolve_project_path(root, probe)
        try:
            resolved.relative_to(artifact_scope)
        except ValueError as exc:
            raise ValueError(f"V27b artifacts.{field_name} must remain inside src-v3/v27/v27b") from exc
    for template in (artifacts.training_log_template, artifacts.training_summary_template, artifacts.epoch_checkpoint_template):
        if "{seed}" not in template:
            raise ValueError("V27b run artifact templates must contain seed")
    if "{seed}" not in artifacts.paired_evaluation_report_path:
        raise ValueError("V27b paired report template must contain seed")

    protocol = _mapping(payload, "protocol", "root")
    expected_protocol = _expected_protocol(model.evidence_carrier)
    if protocol != expected_protocol:
        raise ValueError("V27b protocol differs from the registered design")

    return V27BConfig(
        schema_version=V27B_CONFIG_SCHEMA_VERSION,
        experiment_id=V27B_EXPERIMENT_ID,
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


def _tensor_state_hash(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"V27b state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    if not parts:
        raise ValueError("V27b state cannot be empty")
    return canonical_sha256(parts)


def tensor_state_hash(state: Mapping[str, Any]) -> str:
    """Public deterministic state hash used by checkpoint and input audits."""

    return _tensor_state_hash(state)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V27b JSON artifact is not an object: {path}")
    return dict(value)


def _validate_current_v27a_source_files(root: Path, config: V27BConfig) -> dict[str, Any]:
    """Bind the current source copies separately from archived input metadata."""

    source_config = resolve_project_path(root, config.base.v27a_runtime_config_path, must_exist=True)
    source_manifest = resolve_project_path(root, config.base.v27a_runtime_manifest_path, must_exist=True)
    if sha256_file(source_config) != config.base.expected_v27a_runtime_config_sha256:
        raise ValueError("V27b current V27a runtime config file hash differs")
    if sha256_file(source_manifest) != config.base.expected_v27a_runtime_manifest_sha256:
        raise ValueError("V27b current V27a runtime manifest file hash differs")
    config_payload = _read_json(source_config)
    if canonical_sha256(config_payload) != config.base.expected_v27a_runtime_config_identity_sha256:
        raise ValueError("V27b current V27a runtime config identity differs")
    manifest_payload = _read_json(source_manifest)
    declared_payload = manifest_payload.get("payload_sha256")
    body = {key: value for key, value in manifest_payload.items() if key != "payload_sha256"}
    if declared_payload != config.base.expected_v27a_runtime_manifest_payload_sha256 or declared_payload != canonical_sha256(body):
        raise ValueError("V27b current V27a runtime manifest payload hash differs")
    if manifest_payload.get("experiment_id") != "V27a":
        raise ValueError("V27b current V27a runtime manifest experiment differs")
    return {
        "runtime_config_path": relative_project_path(root, source_config),
        "runtime_config_sha256": sha256_file(source_config),
        "runtime_config_identity_sha256": config.base.expected_v27a_runtime_config_identity_sha256,
        "runtime_manifest_path": relative_project_path(root, source_manifest),
        "runtime_manifest_sha256": sha256_file(source_manifest),
        "runtime_manifest_payload_sha256": declared_payload,
    }


def verify_v27b_inputs(project_root: str | Path, config: V27BConfig) -> dict[str, Any]:
    """Verify copied V27a inputs and return a stable input binding.

    The copied cache/checkpoint metadata intentionally refers to the archived
    V27a manifest.  This function checks that historical identity and also
    checks the current V27a source files named by this experiment.  Mixing the
    two identities is rejected explicitly.
    """

    root = Path(project_root).resolve()
    source_binding = _validate_current_v27a_source_files(root, config)
    checkpoint_path = resolve_project_path(root, config.base.v27a_checkpoint_path, must_exist=True)
    train_path = resolve_project_path(root, config.base.v27a_train_cache_path, must_exist=True)
    val_path = resolve_project_path(root, config.base.v27a_validation_cache_path, must_exist=True)
    observed_files = {
        "checkpoint": sha256_file(checkpoint_path),
        "train_cache": sha256_file(train_path),
        "validation_cache": sha256_file(val_path),
    }
    expected_files = {
        "checkpoint": config.base.expected_v27a_checkpoint_file_sha256,
        "train_cache": config.base.expected_v27a_train_cache_file_sha256,
        "validation_cache": config.base.expected_v27a_validation_cache_file_sha256,
    }
    if observed_files != expected_files:
        raise ValueError(f"V27b copied V27a input file hashes differ: expected={expected_files}, actual={observed_files}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V27b V27a checkpoint must be a mapping")
    checkpoint = dict(checkpoint)
    if (
        checkpoint.get("checkpoint_schema") != "RADAR-LN3-V27a-late-residual-checkpoint-v2"
        or checkpoint.get("experiment_id") != "V27a"
        or checkpoint.get("arm") != config.base.expected_v27a_arm
        or int(checkpoint.get("seed", -1)) != config.base.expected_v27a_seed
        or int(checkpoint.get("epoch", -1)) != config.base.expected_v27a_epoch
    ):
        raise ValueError("V27b V27a checkpoint identity differs")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping) or checkpoint.get("model_state_sha256") != tensor_state_hash(state):
        raise ValueError("V27b V27a checkpoint state hash differs")
    if checkpoint.get("model_state_sha256") != config.base.expected_v27a_checkpoint_state_sha256:
        raise ValueError("V27b V27a checkpoint state is not the registered epoch-20 state")
    if checkpoint.get("model_contract_sha256") != config.base.expected_v27a_model_contract_sha256:
        raise ValueError("V27b V27a checkpoint model contract hash differs")
    if checkpoint.get("config_identity_sha256") != config.base.expected_v27a_checkpoint_training_config_identity_sha256:
        raise ValueError("V27b archived checkpoint config identity differs")
    if checkpoint.get("manifest_sha256") != config.base.expected_v27a_checkpoint_training_manifest_sha256:
        raise ValueError("V27b archived checkpoint manifest identity differs")
    if checkpoint.get("class_names") != list(V27B_L2_CLASS_NAMES):
        raise ValueError("V27b V27a checkpoint class names differ")

    from v27.v27a.cache import load_cache_payload, cache_payload_sha256

    payloads: dict[str, dict[str, Any]] = {}
    cache_paths = {"train": train_path, "val": val_path}
    expected_payloads = {
        "train": config.base.expected_v27a_train_cache_payload_sha256,
        "val": config.base.expected_v27a_validation_cache_payload_sha256,
    }
    for split, path in cache_paths.items():
        payload = load_cache_payload(path)
        if payload.get("split_name") != split:
            raise ValueError(f"V27b copied V27a {split} cache split differs")
        if payload.get("cache_payload_sha256") != expected_payloads[split] or payload.get("cache_payload_sha256") != cache_payload_sha256(payload):
            raise ValueError(f"V27b copied V27a {split} cache payload hash differs")
        if payload.get("v27a_manifest_sha256") != config.base.expected_v27a_checkpoint_training_manifest_sha256 or payload.get("config_identity_sha256") != config.base.expected_v27a_checkpoint_training_config_identity_sha256:
            raise ValueError(f"V27b copied V27a {split} cache archived provenance differs")
        labels = [str(value) for value in payload["parent_labels"]]
        counts = {name: labels.count(name) for name in V27B_L2_CLASS_NAMES}
        if counts != _EXPECTED_PARENT_LABEL_COUNTS[split]:
            raise ValueError(f"V27b copied V27a {split} parent support differs: {counts}")
        payloads[split] = payload
    if set(payloads["train"]["parent_sample_ids"]) & set(payloads["val"]["parent_sample_ids"]):
        raise ValueError("V27b V27a train and validation parents overlap")

    # Verify the V27a model contract against the current implementation.  The
    # checkpoint metadata path may be old, but the architecture must not be.
    from v27.v27a.model import build_v27a_model
    from v27.v27a.protocol import load_v27a_config

    source_config = load_v27a_config(root, resolve_project_path(root, config.base.v27a_runtime_config_path))
    source_model = build_v27a_model(source_config, arm=config.base.expected_v27a_arm, seed=config.base.expected_v27a_seed)
    if source_model.model_contract() != checkpoint.get("model_contract"):
        raise ValueError("V27b V27a checkpoint model contract differs from current V27a model")

    return {
        "source_files": {
            "checkpoint": {"path": relative_project_path(root, checkpoint_path), "sha256": observed_files["checkpoint"]},
            "train_cache": {"path": relative_project_path(root, train_path), "sha256": observed_files["train_cache"]},
            "validation_cache": {"path": relative_project_path(root, val_path), "sha256": observed_files["validation_cache"]},
            **source_binding,
        },
        "checkpoint": {
            "epoch": int(checkpoint["epoch"]),
            "arm": str(checkpoint["arm"]),
            "seed": int(checkpoint["seed"]),
            "state_sha256": str(checkpoint["model_state_sha256"]),
            "model_contract_sha256": str(checkpoint["model_contract_sha256"]),
            "archived_config_identity_sha256": str(checkpoint["config_identity_sha256"]),
            "archived_manifest_sha256": str(checkpoint["manifest_sha256"]),
        },
        "cache_payload_sha256s": {
            "train": expected_payloads["train"],
            "val": expected_payloads["val"],
        },
        "parent_label_counts": _EXPECTED_PARENT_LABEL_COUNTS,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }


V27B_SOURCE_PATHS = (
    Path("src-v3/v27/__init__.py"),
    Path("src-v3/v27/v27b/__init__.py"),
    Path("src-v3/v27/v27b/protocol.py"),
    Path("src-v3/v27/v27b/model.py"),
    Path("src-v3/v27/v27b/cache.py"),
    Path("src-v3/v27/v27b/checkpoint.py"),
    Path("src-v3/v27/v27b/runtime.py"),
    Path("src-v3/v27/v27b/training.py"),
    Path("src-v3/v27/v27b/evaluation.py"),
    Path("src-v3/v27/v27b/cli.py"),
    Path("src-v3/v27/v27b/logging_utils.py"),
    Path("src-v3/v27/v27a/model.py"),
    Path("src-v3/v27/v27a/cache.py"),
    Path("src-v3/v27/v27a/protocol.py"),
    Path("src-v3/v27/v27_baseline/metrics.py"),
    Path("src-v3/v27/v27_baseline/logging_utils.py"),
    Path("src-v3/radar_ln3/losses.py"),
    Path("src-v3/radar_ln3/runtime.py"),
)


def _git_snapshot(root: Path) -> dict[str, Any]:
    """Return a stable repository reference for the manifest.

    A live ``git status`` is intentionally excluded: creating a manifest,
    cache, or report changes status and would make the manifest impossible to
    reload byte-for-byte.  Source and configuration content are already bound
    by their explicit hashes below.
    """

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
    head = result.stdout.strip()
    if result.returncode != 0 or len(head) != 40:
        head = "unavailable"
    return {"head": head}


def build_v27b_manifest(
    project_root: str | Path,
    config: V27BConfig,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    code_hashes = {
        path.as_posix(): sha256_file(root / path)
        for path in V27B_SOURCE_PATHS
    }
    dependencies: dict[str, str] = {}
    for package in ("torch", "numpy"):
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    model_manifest = {
        "parent_state_dim": config.model.parent_state_dim,
        "event_representation_dim": config.model.event_representation_dim,
        "decision_state_dim": config.model.decision_state_dim,
        "input_dim": config.model.input_dim,
        "hidden_dim": config.model.hidden_dim,
        "class_count": config.model.class_count,
        "activation": config.model.activation,
        "residual_bound": config.model.residual_bound,
        "pairwise_delta_bound": config.model.pairwise_delta_bound,
        "residual_projection_bias": config.model.residual_projection_bias,
        "residual_zero_initialization": config.model.residual_zero_initialization,
        "correction_layout": [
            "DroneTarget:+delta/2",
            "BirdTarget:-delta/2",
            "BalloonTarget:0",
        ],
        "no_reliable_event": (
            "zero_output_correction_and_increment_preserve_continuation_state"
        ),
    }
    if config.model.evidence_carrier == "direct_frame_gru":
        model_manifest.update(
            {
                "evidence_carrier": config.model.evidence_carrier,
                "frame_encoder_hidden_dim": config.model.frame_encoder_hidden_dim,
                "frame_encoder_layers": config.model.frame_encoder_layers,
            }
        )
    return {
        "schema_version": V27B_MANIFEST_SCHEMA_VERSION,
        "experiment_id": V27B_EXPERIMENT_ID,
        "variant": config.variant,
        "objective": config.objective,
        "config": {
            "path": relative_project_path(root, config.config_path),
            "sha256": config.config_sha256,
            "identity_sha256": config.config_identity_sha256,
        },
        "code_hashes": code_hashes,
        "git_snapshot": _git_snapshot(root),
        "input_binding": dict(input_binding),
        "model": model_manifest,
        "training": {
            "seeds": list(config.training.seeds),
            "dataloader_seed": config.training.dataloader_seed,
            "batch_size": config.training.batch_size,
            "epochs": config.training.epochs,
            "primary_epoch": config.training.primary_epoch,
            "learning_rate": config.training.learning_rate,
            "weight_decay": config.training.weight_decay,
            "gradient_clip_norm": config.training.gradient_clip_norm,
            "loss_name": config.training.loss_name,
            "loss_reduction": config.training.loss_reduction,
        },
        "evaluation": {
            "class_names": list(config.data.class_names),
            "expected_v27a_control_validation_confusion_matrix": [list(row) for row in config.data.expected_v27a_control_validation_confusion_matrix],
            "eligible_prefix_min_events": config.training.eligible_prefix_min_events,
            "bootstrap_replicates": config.training.bootstrap_replicates,
        },
        "protocol": dict(config.protocol),
        "dependencies": dependencies,
        "scope": {
            "unknown": "sealed_and_not_used",
            "test": "sealed_and_not_used",
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
            "cross_source": "not_in_scope",
        },
    }


def _sealed_body(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27b payload already contains payload_sha256")
    result = dict(payload)
    result["payload_sha256"] = canonical_sha256(payload)
    return result


def verify_sealed_payload(payload: Mapping[str, Any], *, schema: str | None = None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("V27b sealed payload must be an object")
    declared = payload.get("payload_sha256")
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if not isinstance(declared, str) or declared != canonical_sha256(body):
        raise ValueError("V27b sealed payload hash differs")
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError("V27b sealed payload schema differs")
    return dict(payload)




def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_sealed_body(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27b sealed artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def read_sealed_json(path: str | Path, *, schema: str | None = None) -> dict[str, Any]:
    return verify_sealed_payload(json.loads(Path(path).read_text(encoding="utf-8")), schema=schema)


def prepare_v27b_manifest(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> tuple[V27BConfig, dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_v27b_config(root, config_path)
    input_binding = verify_v27b_inputs(root, config)
    manifest = build_v27b_manifest(root, config, input_binding)
    path = resolve_project_path(root, config.artifacts.manifest_path)
    write_sealed_json_if_identical(path, manifest)
    return config, input_binding, path, sha256_file(path)


def load_v27b_context(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> V27BContext:
    root = Path(project_root).resolve()
    config = load_v27b_config(root, config_path)
    input_binding = verify_v27b_inputs(root, config)
    manifest_path = resolve_project_path(root, config.artifacts.manifest_path, must_exist=True)
    manifest = read_sealed_json(manifest_path, schema=V27B_MANIFEST_SCHEMA_VERSION)
    expected = build_v27b_manifest(root, config, input_binding)
    actual_body = {key: value for key, value in manifest.items() if key != "payload_sha256"}
    if actual_body != expected:
        raise ValueError("V27b manifest differs from the current config, inputs, or implementation")
    return V27BContext(
        project_root=root,
        config=config,
        input_binding=dict(input_binding),
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=sha256_file(manifest_path),
    )


def v27b_artifact_path(
    project_root: str | Path,
    template: str,
    *,
    seed: int | None = None,
    arm: str | None = None,
    epoch: int | None = None,
) -> Path:
    if "{seed}" in template and seed is None:
        raise ValueError("V27b artifact template requires seed")
    if "{arm}" in template and arm not in V27B_ALL_ARMS:
        raise ValueError("V27b artifact template requires a canonical arm")
    if "{epoch" in template and epoch is None:
        raise ValueError("V27b artifact template requires epoch")
    try:
        value = template.format(
            seed=int(seed) if seed is not None else None,
            arm=arm,
            epoch=int(epoch) if epoch is not None else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V27b artifact template is invalid") from exc
    return resolve_project_path(project_root, value)


def expected_parent_label_counts(split: str) -> dict[str, int]:
    key = str(split).strip().lower()
    if key not in _EXPECTED_PARENT_LABEL_COUNTS:
        raise ValueError(f"unsupported V27b split: {split}")
    return dict(_EXPECTED_PARENT_LABEL_COUNTS[key])


__all__ = ["V27B_ALL_ARMS", "V27B_ARMS", "V27B_CACHE_SCHEMA_VERSION", "V27B_DIRECT_FRAME_CACHE_SCHEMA_VERSION", "V27B_CHECKPOINT_SCHEMA_VERSION", "V27B_CONFIG_SCHEMA_VERSION", "V27BContext", "V27BConfig", "V27BDataConfig", "V27B_EXPERIMENT_ID", "V27B_L2_CLASS_NAMES", "V27B_STABLE8_INDICES", "V27B_STABLE8_NAMES", "V27B_MANIFEST_SCHEMA_VERSION", "V27B_REPORT_SCHEMA_VERSION", "V27B_SCHEMA_VERSION", "V27B_SMOKE_SCHEMA_VERSION", "V27B_TREATMENT_ARM", "V27B_SOURCE_PATHS", "build_v27b_manifest", "expected_parent_label_counts", "load_v27b_config", "load_v27b_context", "prepare_v27b_manifest", "read_sealed_json", "tensor_state_hash", "verify_sealed_payload", "verify_v27b_inputs", "v27b_artifact_path", "write_sealed_json_if_identical"]