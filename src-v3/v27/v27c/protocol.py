"""Configuration and immutable provenance contracts for V27c.

V27c is intentionally a narrow, known-only experiment.  It starts from the
frozen V27a Main classifier endpoint and tests one additional signal only:
the V9 ``causal-tr-motion-v1`` numeric values, after their validity mask has
been used for numerical gating.  The mask itself is never model input.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.features import (
    TR_MOTION_SCHEMA,
    resolve_causal_tr_feature_contract,
)
from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)


V27C_EXPERIMENT_ID = "V27c"
V27C_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-late-residual-v1"
V27C_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-manifest-v1"
V27C_FEATURE_CACHE_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-feature-cache-v1"
V27C_FEATURE_MANIFEST_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-feature-manifest-v1"
V27C_CHECKPOINT_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-residual-checkpoint-v1"
V27C_REPORT_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-paired-report-v1"
V27C_SMOKE_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-contract-smoke-v1"

V27C_L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
V27C_SPLITS = ("train", "val")
V27C_FROZEN_ERROR_PARENT_IDS = (
    "cq08|track|1535",
    "cq08|track|1908",
    "cq08|track|2083",
    "cq08|track|2103",
)
V27C_FROZEN_MAIN_CONFUSION = ((109, 4, 0), (1, 54, 1), (1, 1, 52))


@dataclass(frozen=True)
class V27CBaseConfig:
    v27a_runtime_config_path: str
    expected_v27a_runtime_config_sha256: str
    expected_v27a_runtime_config_identity_sha256: str
    v27a_runtime_manifest_path: str
    expected_v27a_runtime_manifest_sha256: str
    expected_v27a_runtime_manifest_payload_sha256: str
    v27a_evidence_audit_path: str
    expected_v27a_evidence_audit_sha256: str
    expected_v27a_train_event_ids_order_sha256: str
    expected_v27a_validation_event_ids_order_sha256: str
    v27a_main_checkpoint_path: str
    expected_v27a_main_checkpoint_sha256: str
    expected_v27a_main_state_sha256: str
    expected_v27a_main_model_contract_sha256: str
    expected_v27a_main_training_config_identity_sha256: str
    expected_v27a_main_training_manifest_sha256: str
    expected_v27a_main_base_l2_checkpoint_sha256: str
    expected_v27a_main_base_l2_classifier_state_sha256: str
    v27a_train_reference_cache_path: str
    expected_v27a_train_reference_cache_sha256: str
    expected_v27a_train_reference_cache_payload_sha256: str
    v27a_validation_reference_cache_path: str
    expected_v27a_validation_reference_cache_sha256: str
    expected_v27a_validation_reference_cache_payload_sha256: str
    v27a_paired_report_path: str
    expected_v27a_paired_report_sha256: str
    expected_l2_class_weights: tuple[float, ...]


@dataclass(frozen=True)
class V27CDataConfig:
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


@dataclass(frozen=True)
class V27CModelConfig:
    motion_projection_dim: int
    motion_hidden_dim: int
    class_count: int
    residual_bound: float
    token_activation: str
    token_projection_bias: bool
    gru_bias: bool
    residual_projection_bias: bool


@dataclass(frozen=True)
class V27CTrainingConfig:
    seeds: tuple[int, ...]
    dataloader_seed: int
    batch_size: int
    epochs: int
    primary_epoch: int
    learning_rate: float
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
class V27CArtifactConfig:
    manifest_path: str
    feature_cache_template: str
    feature_manifest_path: str
    smoke_report_path: str
    training_log_template: str
    training_summary_template: str
    checkpoint_template: str
    evaluation_report_template: str
    summary_report_path: str


@dataclass(frozen=True)
class V27CConfig:
    schema_version: str
    experiment_id: str
    variant: str
    objective: str
    base: V27CBaseConfig
    data: V27CDataConfig
    model: V27CModelConfig
    training: V27CTrainingConfig
    artifacts: V27CArtifactConfig
    protocol: Mapping[str, Any]
    config_path: Path
    config_sha256: str
    config_identity_sha256: str


@dataclass(frozen=True)
class V27CContext:
    project_root: Path
    config: V27CConfig
    input_binding: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str


def _required(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V27c {context} is missing {key}")
    return mapping[key]


def _mapping(mapping: Mapping[str, Any], key: str, context: str) -> dict[str, Any]:
    value = _required(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ValueError(f"V27c {context}.{key} must be an object")
    return dict(value)


def _text(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _required(mapping, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"V27c {context}.{key} must be non-empty text")
    return value.strip()


def _sha(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = _text(mapping, key, context).lower()
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"V27c {context}.{key} must be a SHA-256 digest")
    return value


def _positive_int(mapping: Mapping[str, Any], key: str, context: str) -> int:
    value = _required(mapping, key, context)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"V27c {context}.{key} must be a positive integer")
    return int(value)


def _finite_float(
    mapping: Mapping[str, Any], key: str, context: str, *, positive: bool = False
) -> float:
    value = _required(mapping, key, context)
    if isinstance(value, bool):
        raise ValueError(f"V27c {context}.{key} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"V27c {context}.{key} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"V27c {context}.{key} has an invalid numeric value")
    return result


def _int_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[int, ...]:
    value = _required(mapping, key, context)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"V27c {context}.{key} must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
        raise ValueError(f"V27c {context}.{key} must contain integers")
    result = tuple(int(item) for item in value)
    if any(item < 0 for item in result) or len(set(result)) != len(result):
        raise ValueError(f"V27c {context}.{key} must contain unique non-negative integers")
    return result


def _float_tuple(mapping: Mapping[str, Any], key: str, context: str) -> tuple[float, ...]:
    value = _required(mapping, key, context)
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"V27c {context}.{key} must be a non-empty list")
    result = tuple(float(item) for item in value)
    if any(not math.isfinite(item) or item <= 0.0 for item in result):
        raise ValueError(f"V27c {context}.{key} must contain finite positive values")
    return result


def _bool(mapping: Mapping[str, Any], key: str, context: str) -> bool:
    value = _required(mapping, key, context)
    if not isinstance(value, bool):
        raise ValueError(f"V27c {context}.{key} must be boolean")
    return value


def _artifact_path(root: Path, template: str, *, split: str | None = None, seed: int | None = None, epoch: int | None = None) -> Path:
    if "{split}" in template and split not in V27C_SPLITS:
        raise ValueError("V27c artifact needs a canonical split")
    if "{seed}" in template and (seed is None or isinstance(seed, bool) or int(seed) < 0):
        raise ValueError("V27c artifact needs a non-negative seed")
    if "{epoch" in template and (epoch is None or isinstance(epoch, bool) or int(epoch) <= 0):
        raise ValueError("V27c artifact needs a positive epoch")
    try:
        rendered = template.format(split=split, seed=seed, epoch=epoch)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("V27c artifact template is invalid") from exc
    return resolve_project_path(root, rendered)


def v27c_artifact_path(project_root: str | Path, template: str, *, split: str | None = None, seed: int | None = None, epoch: int | None = None) -> Path:
    return _artifact_path(Path(project_root).resolve(), template, split=split, seed=seed, epoch=epoch)


def load_v27c_config(project_root: str | Path, config_path: str | Path | None = None) -> V27CConfig:
    root = Path(project_root).resolve()
    candidate = (
        root / "src-v3" / "v27" / "v27c" / "config" / "V27c-causal-tr-motion-late-residual.json"
        if config_path is None
        else Path(config_path)
    )
    path = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("V27c configuration must remain inside project root") from exc
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("V27c configuration root must be an object")
    if payload.get("schema_version") != V27C_SCHEMA_VERSION:
        raise ValueError("V27c configuration schema differs")
    if payload.get("experiment_id") != V27C_EXPERIMENT_ID:
        raise ValueError("V27c configuration experiment id differs")

    base_raw = _mapping(payload, "base", "root")
    base = V27CBaseConfig(
        v27a_runtime_config_path=_text(base_raw, "v27a_runtime_config_path", "base"),
        expected_v27a_runtime_config_sha256=_sha(base_raw, "expected_v27a_runtime_config_sha256", "base"),
        expected_v27a_runtime_config_identity_sha256=_sha(base_raw, "expected_v27a_runtime_config_identity_sha256", "base"),
        v27a_runtime_manifest_path=_text(base_raw, "v27a_runtime_manifest_path", "base"),
        expected_v27a_runtime_manifest_sha256=_sha(base_raw, "expected_v27a_runtime_manifest_sha256", "base"),
        expected_v27a_runtime_manifest_payload_sha256=_sha(base_raw, "expected_v27a_runtime_manifest_payload_sha256", "base"),
        v27a_evidence_audit_path=_text(base_raw, "v27a_evidence_audit_path", "base"),
        expected_v27a_evidence_audit_sha256=_sha(base_raw, "expected_v27a_evidence_audit_sha256", "base"),
        expected_v27a_train_event_ids_order_sha256=_sha(base_raw, "expected_v27a_train_event_ids_order_sha256", "base"),
        expected_v27a_validation_event_ids_order_sha256=_sha(base_raw, "expected_v27a_validation_event_ids_order_sha256", "base"),
        v27a_main_checkpoint_path=_text(base_raw, "v27a_main_checkpoint_path", "base"),
        expected_v27a_main_checkpoint_sha256=_sha(base_raw, "expected_v27a_main_checkpoint_sha256", "base"),
        expected_v27a_main_state_sha256=_sha(base_raw, "expected_v27a_main_state_sha256", "base"),
        expected_v27a_main_model_contract_sha256=_sha(base_raw, "expected_v27a_main_model_contract_sha256", "base"),
        expected_v27a_main_training_config_identity_sha256=_sha(base_raw, "expected_v27a_main_training_config_identity_sha256", "base"),
        expected_v27a_main_training_manifest_sha256=_sha(base_raw, "expected_v27a_main_training_manifest_sha256", "base"),
        expected_v27a_main_base_l2_checkpoint_sha256=_sha(base_raw, "expected_v27a_main_base_l2_checkpoint_sha256", "base"),
        expected_v27a_main_base_l2_classifier_state_sha256=_sha(base_raw, "expected_v27a_main_base_l2_classifier_state_sha256", "base"),
        v27a_train_reference_cache_path=_text(base_raw, "v27a_train_reference_cache_path", "base"),
        expected_v27a_train_reference_cache_sha256=_sha(base_raw, "expected_v27a_train_reference_cache_sha256", "base"),
        expected_v27a_train_reference_cache_payload_sha256=_sha(base_raw, "expected_v27a_train_reference_cache_payload_sha256", "base"),
        v27a_validation_reference_cache_path=_text(base_raw, "v27a_validation_reference_cache_path", "base"),
        expected_v27a_validation_reference_cache_sha256=_sha(base_raw, "expected_v27a_validation_reference_cache_sha256", "base"),
        expected_v27a_validation_reference_cache_payload_sha256=_sha(base_raw, "expected_v27a_validation_reference_cache_payload_sha256", "base"),
        v27a_paired_report_path=_text(base_raw, "v27a_paired_report_path", "base"),
        expected_v27a_paired_report_sha256=_sha(base_raw, "expected_v27a_paired_report_sha256", "base"),
        expected_l2_class_weights=_float_tuple(base_raw, "expected_l2_class_weights", "base"),
    )
    if len(base.expected_l2_class_weights) != len(V27C_L2_CLASSES):
        raise ValueError("V27c L2 class weight width differs")

    data_raw = _mapping(payload, "data", "root")
    contract = resolve_causal_tr_feature_contract(_text(data_raw, "motion_schema", "data"))
    data = V27CDataConfig(
        index_path=_text(data_raw, "index_path", "data"),
        expected_index_sha256=_sha(data_raw, "expected_index_sha256", "data"),
        index_schema_version=_text(data_raw, "index_schema_version", "data"),
        train_split_name=_text(data_raw, "train_split_name", "data"),
        validation_split_name=_text(data_raw, "validation_split_name", "data"),
        max_sequence_length=_positive_int(data_raw, "max_sequence_length", "data"),
        max_rd_frames_per_point=_positive_int(data_raw, "max_rd_frames_per_point", "data"),
        rd_amplitude_center=_finite_float(data_raw, "rd_amplitude_center", "data"),
        rd_amplitude_scale=_finite_float(data_raw, "rd_amplitude_scale", "data", positive=True),
        motion_schema=contract.schema,
        motion_dim=_positive_int(data_raw, "motion_dim", "data"),
        motion_clip_absolute=_finite_float(data_raw, "motion_clip_absolute", "data", positive=True),
        motion_mask_mode=_text(data_raw, "motion_mask_mode", "data"),
    )
    if (
        data.motion_schema != TR_MOTION_SCHEMA
        or data.motion_dim != contract.value_dim
        or not math.isclose(data.motion_clip_absolute, contract.clip_absolute, rel_tol=0.0, abs_tol=1.0e-9)
        or data.motion_mask_mode != contract.mask_mode
    ):
        raise ValueError("V27c motion contract differs from causal-tr-motion-v1")
    if data.train_split_name != "train" or data.validation_split_name != "val":
        raise ValueError("V27c split names must match archived V27a caches")

    model_raw = _mapping(payload, "model", "root")
    model = V27CModelConfig(
        motion_projection_dim=_positive_int(model_raw, "motion_projection_dim", "model"),
        motion_hidden_dim=_positive_int(model_raw, "motion_hidden_dim", "model"),
        class_count=_positive_int(model_raw, "class_count", "model"),
        residual_bound=_finite_float(model_raw, "residual_bound", "model", positive=True),
        token_activation=_text(model_raw, "token_activation", "model").lower(),
        token_projection_bias=_bool(model_raw, "token_projection_bias", "model"),
        gru_bias=_bool(model_raw, "gru_bias", "model"),
        residual_projection_bias=_bool(model_raw, "residual_projection_bias", "model"),
    )
    if model.class_count != len(V27C_L2_CLASSES):
        raise ValueError("V27c classifier width differs")
    if (
        model.token_activation != "silu"
        or model.token_projection_bias
        or model.gru_bias
        or model.residual_projection_bias
    ):
        raise ValueError("V27c motion branch must be bias-free to exclude mask-only shortcuts")

    training_raw = _mapping(payload, "training", "root")
    training = V27CTrainingConfig(
        seeds=_int_tuple(training_raw, "seeds", "training"),
        dataloader_seed=_positive_int(training_raw, "dataloader_seed", "training"),
        batch_size=_positive_int(training_raw, "batch_size", "training"),
        epochs=_positive_int(training_raw, "epochs", "training"),
        primary_epoch=_positive_int(training_raw, "primary_epoch", "training"),
        learning_rate=_finite_float(training_raw, "learning_rate", "training", positive=True),
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
    if training.seeds != (42, 43, 44):
        raise ValueError("V27c uses the preregistered three head seeds")
    if training.primary_epoch != training.epochs or training.epochs != 20:
        raise ValueError("V27c uses a fixed 20-epoch primary endpoint")
    if (
        training.device not in {"cpu", "cuda"}
        or training.optimizer != "AdamW"
        or training.loss_name != "parent_final_fixed_parent_count_ce"
        or training.loss_reduction != "fixed_parent_count"
        or training.bootstrap_replicates < 100
        or training.weight_decay < 0.0
        or not training.cuda_deterministic_algorithms
    ):
        raise ValueError("V27c training runtime or bootstrap count differs")

    artifacts_raw = _mapping(payload, "artifacts", "root")
    artifacts = V27CArtifactConfig(**{
        field: _text(artifacts_raw, field, "artifacts")
        for field in V27CArtifactConfig.__dataclass_fields__
    })
    artifact_scope = (root / "src-v3" / "v27" / "v27c").resolve()
    for field, template in artifacts.__dict__.items():
        probe = _artifact_path(root, template, split="train", seed=42, epoch=1)
        try:
            probe.relative_to(artifact_scope)
        except ValueError as exc:
            raise ValueError(f"V27c artifacts.{field} must remain in src-v3/v27/v27c") from exc

    protocol = _mapping(payload, "protocol", "root")
    required_protocol = {
        "frozen_endpoint": "V27a_main_epoch_20",
        "trainable_component": "causal_tr_motion_late_residual_only",
        "motion_mask": "numerical_gating_only_not_model_input",
        "frozen_endpoint_logits_as_additive_output_anchor": True,
        "frozen_endpoint_logits_input_to_motion_encoder": False,
        "unknown_training_exposure": 0,
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "calibration": "not_in_scope",
        "l3": "not_in_scope",
        "test": "sealed_and_not_used",
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"V27c protocol.{key} differs")
    return V27CConfig(
        schema_version=V27C_SCHEMA_VERSION,
        experiment_id=V27C_EXPERIMENT_ID,
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


def _file_binding(root: Path, path: str, expected: str, *, field: str) -> dict[str, str]:
    target = resolve_project_path(root, path, must_exist=True)
    observed = sha256_file(target)
    if observed != expected:
        raise ValueError(f"V27c {field} file hash differs")
    return {"path": relative_project_path(root, target), "sha256": observed}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"V27c JSON artifact is not an object: {path}")
    return dict(value)


def _state_hash(state: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("V27c frozen V27a state contains a non-tensor")
        tensor = value.detach().cpu().contiguous()
        parts.extend((str(name), str(tensor.dtype), repr(tuple(tensor.shape)), tensor.numpy().tobytes().hex()))
    if not parts:
        raise ValueError("V27c frozen V27a state is empty")
    return canonical_sha256(parts)


def _load_reference_cache(root: Path, path: str, *, split: str, expected_file: str, expected_payload: str, base: V27CBaseConfig) -> dict[str, Any]:
    target = resolve_project_path(root, path, must_exist=True)
    if sha256_file(target) != expected_file:
        raise ValueError(f"V27c {split} reference cache file hash differs")
    from v27.v27a.cache import cache_payload_sha256, load_cache_payload

    payload = load_cache_payload(target)
    if (
        payload.get("split_name") != split
        or payload.get("cache_payload_sha256") != expected_payload
        or cache_payload_sha256(payload) != expected_payload
        or payload.get("config_identity_sha256") != base.expected_v27a_main_training_config_identity_sha256
        or payload.get("v27a_manifest_sha256") != base.expected_v27a_main_training_manifest_sha256
    ):
        raise ValueError(f"V27c {split} reference cache identity differs")
    return payload


def verify_v27c_inputs(project_root: str | Path, config: V27CConfig) -> dict[str, Any]:
    """Verify the archived endpoint and current compatible source code.

    V27a checkpoint metadata carries paths from before the V27 directory move.
    This deliberately checks its archived identity separately from the current
    compatible V27a source implementation instead of rewriting history.
    """

    root = Path(project_root).resolve()
    base = config.base
    files = {
        "v27a_runtime_config": _file_binding(root, base.v27a_runtime_config_path, base.expected_v27a_runtime_config_sha256, field="V27a runtime config"),
        "v27a_runtime_manifest": _file_binding(root, base.v27a_runtime_manifest_path, base.expected_v27a_runtime_manifest_sha256, field="V27a runtime manifest"),
        "v27a_evidence_audit": _file_binding(root, base.v27a_evidence_audit_path, base.expected_v27a_evidence_audit_sha256, field="V27a evidence audit"),
        "v27a_main_checkpoint": _file_binding(root, base.v27a_main_checkpoint_path, base.expected_v27a_main_checkpoint_sha256, field="V27a Main checkpoint"),
        "v27a_train_cache": _file_binding(root, base.v27a_train_reference_cache_path, base.expected_v27a_train_reference_cache_sha256, field="V27a train reference cache"),
        "v27a_validation_cache": _file_binding(root, base.v27a_validation_reference_cache_path, base.expected_v27a_validation_reference_cache_sha256, field="V27a validation reference cache"),
        "v27a_paired_report": _file_binding(root, base.v27a_paired_report_path, base.expected_v27a_paired_report_sha256, field="V27a paired report"),
        "index": _file_binding(root, config.data.index_path, config.data.expected_index_sha256, field="canonical index"),
    }
    runtime_config = _read_json(resolve_project_path(root, base.v27a_runtime_config_path, must_exist=True))
    runtime_manifest = _read_json(resolve_project_path(root, base.v27a_runtime_manifest_path, must_exist=True))
    evidence_audit = _read_json(resolve_project_path(root, base.v27a_evidence_audit_path, must_exist=True))
    runtime_body = {key: value for key, value in runtime_manifest.items() if key != "payload_sha256"}
    if (
        canonical_sha256(runtime_config) != base.expected_v27a_runtime_config_identity_sha256
        or runtime_manifest.get("payload_sha256") != base.expected_v27a_runtime_manifest_payload_sha256
        or canonical_sha256(runtime_body) != base.expected_v27a_runtime_manifest_payload_sha256
    ):
        raise ValueError("V27c current V27a runtime identity differs")
    endpoint_source_paths = (
        "src-v3/v27/v27a/model.py",
        "src-v3/v27/v27a/cache.py",
        "src-v3/v27/v27a/protocol.py",
    )
    archived_source_hashes = runtime_manifest.get("code_hashes")
    if not isinstance(archived_source_hashes, Mapping) or any(
        archived_source_hashes.get(path) != sha256_file(root / path)
        for path in endpoint_source_paths
    ):
        raise ValueError("V27c V27a endpoint source hash differs")
    audit_body = {key: value for key, value in evidence_audit.items() if key != "payload_sha256"}
    if (
        runtime_manifest.get("audit_report_sha256") != base.expected_v27a_evidence_audit_sha256
        or evidence_audit.get("schema_version") != "RADAR-LN3-V27a-evidence-cache-audit-v1"
        or evidence_audit.get("experiment_id") != "V27a"
        or evidence_audit.get("payload_sha256") != canonical_sha256(audit_body)
        or evidence_audit.get("splits", {}).get("train", {}).get("event_ids_order_sha256")
        != base.expected_v27a_train_event_ids_order_sha256
        or evidence_audit.get("splits", {}).get("val", {}).get("event_ids_order_sha256")
        != base.expected_v27a_validation_event_ids_order_sha256
    ):
        raise ValueError("V27c V27a evidence-order audit differs")

    checkpoint_path = resolve_project_path(root, base.v27a_main_checkpoint_path, must_exist=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("V27c V27a Main checkpoint is not an object")
    checkpoint = dict(checkpoint)
    state = checkpoint.get("model_state")
    if (
        checkpoint.get("checkpoint_schema") != "RADAR-LN3-V27a-late-residual-checkpoint-v2"
        or checkpoint.get("experiment_id") != "V27a"
        or checkpoint.get("arm") != "main"
        or int(checkpoint.get("seed", -1)) != 42
        or int(checkpoint.get("epoch", -1)) != 20
        or not isinstance(state, Mapping)
        or checkpoint.get("model_state_sha256") != _state_hash(state)
        or checkpoint.get("model_state_sha256") != base.expected_v27a_main_state_sha256
        or checkpoint.get("model_contract_sha256") != base.expected_v27a_main_model_contract_sha256
        or checkpoint.get("config_identity_sha256") != base.expected_v27a_main_training_config_identity_sha256
        or checkpoint.get("manifest_sha256") != base.expected_v27a_main_training_manifest_sha256
        or checkpoint.get("base_checkpoint_sha256") != base.expected_v27a_main_base_l2_checkpoint_sha256
        or checkpoint.get("base_classifier_state_sha256") != base.expected_v27a_main_base_l2_classifier_state_sha256
        or checkpoint.get("class_names") != list(V27C_L2_CLASSES)
    ):
        raise ValueError("V27c V27a Main checkpoint identity differs")

    from v27.v27a.model import V27AEvidenceResidual
    from v27.v27a.protocol import load_v27a_config

    source_config = load_v27a_config(root, resolve_project_path(root, base.v27a_runtime_config_path))
    source_model = V27AEvidenceResidual(source_config, arm="main")
    if source_model.model_contract() != checkpoint.get("model_contract"):
        raise ValueError("V27c V27a Main model contract differs from current source")
    train = _load_reference_cache(root, base.v27a_train_reference_cache_path, split="train", expected_file=base.expected_v27a_train_reference_cache_sha256, expected_payload=base.expected_v27a_train_reference_cache_payload_sha256, base=base)
    val = _load_reference_cache(root, base.v27a_validation_reference_cache_path, split="val", expected_file=base.expected_v27a_validation_reference_cache_sha256, expected_payload=base.expected_v27a_validation_reference_cache_payload_sha256, base=base)
    expected_cache_hashes = {
        "train": base.expected_v27a_train_reference_cache_payload_sha256,
        "val": base.expected_v27a_validation_reference_cache_payload_sha256,
    }
    if checkpoint.get("cache_payload_sha256s") != expected_cache_hashes:
        raise ValueError("V27c V27a Main checkpoint cache binding differs")
    v27a_base = runtime_config.get("base")
    if not isinstance(v27a_base, Mapping):
        raise ValueError("V27c V27a runtime config lacks its frozen base binding")
    frozen_binding_path = v27a_base.get("frozen_binding_path")
    frozen_binding_hash = v27a_base.get("expected_binding_payload_sha256")
    if not isinstance(frozen_binding_path, str) or not isinstance(frozen_binding_hash, str):
        raise ValueError("V27c V27a frozen base binding declaration differs")
    from v27.v27_baseline.base_binding import load_frozen_base_binding

    frozen_binding_target = resolve_project_path(
        root, frozen_binding_path, must_exist=True
    )
    frozen_binding = load_frozen_base_binding(root, frozen_binding_target)
    if frozen_binding.get("binding_sha256") != frozen_binding_hash:
        raise ValueError("V27c V27a frozen base binding payload differs")
    bound_l2_weights = tuple(
        float(value) for value in frozen_binding.get("l2", {}).get("class_weights", ())
    )
    if bound_l2_weights != base.expected_l2_class_weights:
        raise ValueError("V27c L2 class weights differ from frozen V27 binding")
    for split, reference in (("train", train), ("val", val)):
        if (
            reference.get("base_checkpoint_sha256")
            != base.expected_v27a_main_base_l2_checkpoint_sha256
            or reference.get("base_classifier_state_sha256")
            != base.expected_v27a_main_base_l2_classifier_state_sha256
        ):
            raise ValueError(f"V27c V27a {split} reference cache base binding differs")
    if set(train["parent_sample_ids"]) & set(val["parent_sample_ids"]):
        raise ValueError("V27c V27a reference train/validation parents overlap")
    report = _read_json(resolve_project_path(root, base.v27a_paired_report_path, must_exist=True))
    matrix = report.get("arms", {}).get("main", {}).get("parent_metrics", {}).get("confusion_matrix_rows_true_columns_pred")
    if matrix != [list(row) for row in V27C_FROZEN_MAIN_CONFUSION]:
        raise ValueError("V27c V27a Main reference confusion matrix differs")
    return {
        "files": files,
        "archived_v27a_main": {
            "checkpoint_epoch": 20,
            "checkpoint_state_sha256": checkpoint["model_state_sha256"],
            "model_contract_sha256": checkpoint["model_contract_sha256"],
            "training_config_identity_sha256": checkpoint["config_identity_sha256"],
            "training_manifest_sha256": checkpoint["manifest_sha256"],
            "base_l2_checkpoint_sha256": checkpoint["base_checkpoint_sha256"],
            "base_l2_classifier_state_sha256": checkpoint["base_classifier_state_sha256"],
        },
        "reference_caches": {
            "train_payload_sha256": train["cache_payload_sha256"],
            "val_payload_sha256": val["cache_payload_sha256"],
            "train_parent_count": len(train["parent_sample_ids"]),
            "val_parent_count": len(val["parent_sample_ids"]),
        },
        "v27a_evidence_order": {
            "train_event_ids_order_sha256": base.expected_v27a_train_event_ids_order_sha256,
            "val_event_ids_order_sha256": base.expected_v27a_validation_event_ids_order_sha256,
        },
        "frozen_v27_l2_binding": {
            "path": relative_project_path(root, frozen_binding_target),
            "file_sha256": sha256_file(frozen_binding_target),
            "payload_sha256": frozen_binding["binding_sha256"],
            "class_weights": list(bound_l2_weights),
        },
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
    }


V27C_SOURCE_PATHS = (
    Path("src-v3/radar_ln3/data.py"),
    Path("src-v3/radar_ln3/features.py"),
    Path("src-v3/radar_ln3/losses.py"),
    Path("src-v3/radar_ln3/runtime.py"),
    Path("src-v3/v27/__init__.py"),
    Path("src-v3/v27/v27_baseline/logging_utils.py"),
    Path("src-v3/v27/v27_baseline/metrics.py"),
    Path("src-v3/v27/v27_baseline/runtime.py"),
    Path("src-v3/v27/v27_baseline/base_binding.py"),
    Path("src-v3/v27/v27a/cache.py"),
    Path("src-v3/v27/v27a/model.py"),
    Path("src-v3/v27/v27a/protocol.py"),
    Path("src-v3/v27/v27c/__init__.py"),
    Path("src-v3/v27/v27c/protocol.py"),
    Path("src-v3/v27/v27c/features.py"),
    Path("src-v3/v27/v27c/model.py"),
    Path("src-v3/v27/v27c/training.py"),
    Path("src-v3/v27/v27c/evaluation.py"),
    Path("src-v3/v27/v27c/cli.py"),
)


def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        encoding="utf-8", errors="replace", stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def build_v27c_manifest(project_root: str | Path, config: V27CConfig, binding: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(project_root).resolve()
    return {
        "schema_version": V27C_MANIFEST_SCHEMA_VERSION,
        "experiment_id": config.experiment_id,
        "variant": config.variant,
        "objective": config.objective,
        "config": {
            "path": relative_project_path(root, config.config_path),
            "sha256": config.config_sha256,
            "identity_sha256": config.config_identity_sha256,
        },
        "input_binding": dict(binding),
        "motion_contract": resolve_causal_tr_feature_contract(config.data.motion_schema).as_dict(),
        "model": dict(config.model.__dict__),
        "training": {
            **dict(config.training.__dict__),
            "seeds": list(config.training.seeds),
        },
        "protocol": dict(config.protocol),
        "source_hashes": {path.as_posix(): sha256_file(root / path) for path in V27C_SOURCE_PATHS},
        "git_head": _git_head(root),
        "scope": {
            "known_only_classification": True,
            "unknown": "sealed_and_not_used",
            "clutter": "sealed_and_not_used",
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test": "sealed_and_not_used",
        },
    }


def _seal(payload: Mapping[str, Any]) -> dict[str, Any]:
    if "payload_sha256" in payload:
        raise ValueError("V27c payload is already sealed")
    result = dict(payload)
    result["payload_sha256"] = canonical_sha256(payload)
    return result


def read_sealed_json(path: str | Path, *, schema: str | None = None) -> dict[str, Any]:
    payload = _read_json(Path(path))
    body = {key: value for key, value in payload.items() if key != "payload_sha256"}
    if payload.get("payload_sha256") != canonical_sha256(body):
        raise ValueError("V27c sealed JSON payload hash differs")
    if schema is not None and payload.get("schema_version") != schema:
        raise ValueError("V27c sealed JSON schema differs")
    return payload


def write_sealed_json_if_identical(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    text = json_text(_seal(payload), indent=2)
    if target.exists():
        if target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to replace a different V27c artifact: {target}")
        return target
    atomic_write_text(target, text)
    return target


def prepare_v27c_manifest(project_root: str | Path, config_path: str | Path | None = None) -> tuple[V27CConfig, dict[str, Any], Path, str]:
    root = Path(project_root).resolve()
    config = load_v27c_config(root, config_path)
    binding = verify_v27c_inputs(root, config)
    manifest = build_v27c_manifest(root, config, binding)
    path = v27c_artifact_path(root, config.artifacts.manifest_path)
    write_sealed_json_if_identical(path, manifest)
    return config, binding, path, sha256_file(path)


def load_v27c_context(project_root: str | Path, config_path: str | Path | None = None) -> V27CContext:
    root = Path(project_root).resolve()
    config = load_v27c_config(root, config_path)
    binding = verify_v27c_inputs(root, config)
    path = v27c_artifact_path(root, config.artifacts.manifest_path)
    manifest = read_sealed_json(path, schema=V27C_MANIFEST_SCHEMA_VERSION)
    expected = build_v27c_manifest(root, config, binding)
    if {key: value for key, value in manifest.items() if key != "payload_sha256"} != expected:
        raise ValueError("V27c manifest differs from current inputs or code")
    return V27CContext(root, config, dict(binding), manifest, path, sha256_file(path))


__all__ = [
    "V27C_CHECKPOINT_SCHEMA_VERSION", "V27C_EXPERIMENT_ID", "V27C_FEATURE_CACHE_SCHEMA_VERSION",
    "V27C_FEATURE_MANIFEST_SCHEMA_VERSION", "V27C_FROZEN_ERROR_PARENT_IDS", "V27C_FROZEN_MAIN_CONFUSION",
    "V27C_L2_CLASSES", "V27C_MANIFEST_SCHEMA_VERSION", "V27C_REPORT_SCHEMA_VERSION",
    "V27C_SCHEMA_VERSION", "V27C_SMOKE_SCHEMA_VERSION", "V27C_SPLITS", "V27CConfig", "V27CContext",
    "build_v27c_manifest", "load_v27c_config", "load_v27c_context", "prepare_v27c_manifest",
    "read_sealed_json", "v27c_artifact_path", "verify_v27c_inputs", "write_sealed_json_if_identical",
]
