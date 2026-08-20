"""Configuration contract for V23 official-CfC KnownGate OOF treatments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .known_gate import KnownGateConfig, known_gate_config_identity_sha256, load_known_gate_config
from .known_gate_v23_oof import (
    ARMS,
    FIXED_EPOCHS,
    FOLD_COUNT,
    TRAINING_SEED,
    artifact_relative,
    config_relative,
    source_config_relative,
)
from .official_cfc import (
    OFFICIAL_CFC_ACTIVATION,
    OFFICIAL_CFC_CLASS_PATH,
    OFFICIAL_CFC_DISTRIBUTION,
    OFFICIAL_CFC_DISTRIBUTION_VERSION,
    OFFICIAL_CFC_GIT_COMMIT,
    OFFICIAL_CFC_MODE,
    official_cfc_runtime_binding,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V23 config is missing {key}")
    return mapping[key]


@dataclass(frozen=True)
class OfficialCfCConfig:
    distribution: str
    distribution_version: str
    git_commit: str
    class_path: str
    mode: str
    activation: str
    hidden_dim: int
    backbone_units: int
    backbone_layers: int
    backbone_dropout: float
    mixed_memory: bool
    return_sequences: bool
    batch_first: bool
    per_parent_forward_for_upstream_batch_timespan_compatibility: bool


@dataclass(frozen=True)
class V23TrainConfig:
    batch_size: int
    learning_rate: float
    seed: int
    dataloader_seed: int
    device: str
    epochs: int
    gradient_clip_norm: float
    class_weights: tuple[float, float]
    loss_mode: str
    class_weight_normalization: str
    prefix_auxiliary_weight: float
    prefix_auxiliary_min_events: int
    checkpoint_path: str
    last_checkpoint_path: str
    formal_artifact_dir: str
    epoch_checkpoint_dir: str
    checkpoint_schema: str


@dataclass(frozen=True)
class V23Config:
    schema_version: str
    config_name: str
    arm: str
    fold_id: int
    source_config_path: str
    source_config_sha256: str
    source_config_identity_sha256: str
    source_prediction_path: str
    source_prediction_sha256: str
    official_cfc: OfficialCfCConfig
    train: V23TrainConfig


def config_identity_sha256(config: V23Config) -> str:
    payload = {
        "schema_version": config.schema_version,
        "config_name": config.config_name,
        "arm": config.arm,
        "fold_id": config.fold_id,
        "source_config_path": config.source_config_path,
        "source_config_sha256": config.source_config_sha256,
        "source_config_identity_sha256": config.source_config_identity_sha256,
        "source_prediction_path": config.source_prediction_path,
        "source_prediction_sha256": config.source_prediction_sha256,
        "official_cfc": config.official_cfc.__dict__,
        "train": config.train.__dict__,
    }
    return canonical_sha256(payload)


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_v23_config(project_root: Path, path: Path) -> tuple[V23Config, KnownGateConfig, Path]:
    """Load a V23 sidecar config and its immutable V21e/V22 source config."""

    root = project_root.resolve()
    config_path = path.resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    arm = str(_require(payload, "arm"))
    if arm not in ARMS:
        raise ValueError("V23 arm must be v23a or v23b")
    schema_version = str(_require(payload, "schema_version"))
    expected_schema = f"RADAR-LN3-{arm.upper()}"
    if schema_version != expected_schema:
        raise ValueError("V23 schema version does not match its arm")
    fold_id = int(_require(payload, "fold_id"))
    if not 0 <= fold_id < FOLD_COUNT:
        raise ValueError("V23 fold id is invalid")
    config_name = str(_require(payload, "config_name"))
    if config_name != config_path.stem.removesuffix("-config"):
        raise ValueError("V23 config filename/name mismatch")
    if config_path != (root / config_relative(arm, fold_id)).resolve():
        raise ValueError("V23 config path is outside its frozen namespace")
    source = dict(_require(payload, "source"))
    source_config_path = str(_require(source, "config_path"))
    source_config_sha256 = str(_require(source, "config_sha256")).lower()
    source_config_identity = str(_require(source, "config_identity_sha256")).lower()
    source_prediction_path = str(_require(source, "prediction_path"))
    source_prediction_sha256 = str(_require(source, "prediction_sha256")).lower()
    expected_source_config = source_config_relative(arm, fold_id)
    if source_config_path.replace("\\", "/") != str(expected_source_config).replace("\\", "/"):
        raise ValueError("V23 source config path does not match its arm/fold")
    actual_source_path = _resolve(root, source_config_path)
    if not actual_source_path.is_file() or sha256_file(actual_source_path) != source_config_sha256:
        raise ValueError("V23 source config hash is stale")
    source_config = load_known_gate_config(actual_source_path)
    if known_gate_config_identity_sha256(source_config) != source_config_identity:
        raise ValueError("V23 source KnownGate config identity is stale")
    if source_config.train.seed != TRAINING_SEED or source_config.train.dataloader_seed != TRAINING_SEED:
        raise ValueError("V23 source config does not use frozen seed43")
    if source_config.train.epochs != FIXED_EPOCHS or source_config.train.early_stopping_patience != 0:
        raise ValueError("V23 source config does not use the fixed 20-epoch OOF protocol")
    if source_config.model.liquid_hidden_dim != 48:
        raise ValueError("V23 parameter-matched protocol requires liquid_hidden_dim=48")
    if source_config.model.rd_temporal_residual_dim != 0:
        raise ValueError("V23 excludes rejected RD temporal residual branches")
    if source_config.model.evidence_dim != 1:
        raise ValueError("V23 requires frozen V21e causal-course evidence only")
    actual_prediction_path = _resolve(root, source_prediction_path)
    if not actual_prediction_path.is_file() or sha256_file(actual_prediction_path) != source_prediction_sha256:
        raise ValueError("V23 source prediction hash is stale")

    official_payload = dict(_require(payload, "official_cfc"))
    official = OfficialCfCConfig(
        distribution=str(_require(official_payload, "distribution")),
        distribution_version=str(_require(official_payload, "distribution_version")),
        git_commit=str(_require(official_payload, "git_commit")),
        class_path=str(_require(official_payload, "class_path")),
        mode=str(_require(official_payload, "mode")),
        activation=str(_require(official_payload, "activation")),
        hidden_dim=int(_require(official_payload, "hidden_dim")),
        backbone_units=int(_require(official_payload, "backbone_units")),
        backbone_layers=int(_require(official_payload, "backbone_layers")),
        backbone_dropout=float(_require(official_payload, "backbone_dropout")),
        mixed_memory=bool(_require(official_payload, "mixed_memory")),
        return_sequences=bool(_require(official_payload, "return_sequences")),
        batch_first=bool(_require(official_payload, "batch_first")),
        per_parent_forward_for_upstream_batch_timespan_compatibility=bool(
            _require(official_payload, "per_parent_forward_for_upstream_batch_timespan_compatibility")
        ),
    )
    if (
        official.distribution != OFFICIAL_CFC_DISTRIBUTION
        or official.distribution_version != OFFICIAL_CFC_DISTRIBUTION_VERSION
        or official.git_commit != OFFICIAL_CFC_GIT_COMMIT
        or official.class_path != OFFICIAL_CFC_CLASS_PATH
        or official.mode != OFFICIAL_CFC_MODE
        or official.activation != OFFICIAL_CFC_ACTIVATION
        or official.hidden_dim != source_config.model.liquid_hidden_dim
        or official.backbone_units != source_config.model.liquid_hidden_dim
        or official.backbone_layers != 1
        or official.backbone_dropout != 0.0
        or official.mixed_memory
        or not official.return_sequences
        or not official.batch_first
        or not official.per_parent_forward_for_upstream_batch_timespan_compatibility
    ):
        raise ValueError("V23 official CfC specification differs from the frozen parameter-matched protocol")
    runtime = official_cfc_runtime_binding()
    if (
        runtime["distribution"] != official.distribution
        or runtime["distribution_version"] != official.distribution_version
        or runtime["git_commit"] != official.git_commit
        or runtime["class_path"] != official.class_path
    ):
        raise RuntimeError("installed official ncps runtime differs from the V23 config")

    train_payload = dict(_require(payload, "train"))
    class_weights = tuple(float(value) for value in _require(train_payload, "class_weights"))
    train = V23TrainConfig(
        batch_size=int(_require(train_payload, "batch_size")),
        learning_rate=float(_require(train_payload, "learning_rate")),
        seed=int(_require(train_payload, "seed")),
        dataloader_seed=int(_require(train_payload, "dataloader_seed")),
        device=str(_require(train_payload, "device")),
        epochs=int(_require(train_payload, "epochs")),
        gradient_clip_norm=float(_require(train_payload, "gradient_clip_norm")),
        class_weights=class_weights,
        loss_mode=str(_require(train_payload, "loss_mode")),
        class_weight_normalization=str(_require(train_payload, "class_weight_normalization")),
        prefix_auxiliary_weight=float(_require(train_payload, "prefix_auxiliary_weight")),
        prefix_auxiliary_min_events=int(_require(train_payload, "prefix_auxiliary_min_events")),
        checkpoint_path=str(_require(train_payload, "checkpoint_path")),
        last_checkpoint_path=str(_require(train_payload, "last_checkpoint_path")),
        formal_artifact_dir=str(_require(train_payload, "formal_artifact_dir")),
        epoch_checkpoint_dir=str(_require(train_payload, "epoch_checkpoint_dir")),
        checkpoint_schema=str(_require(train_payload, "checkpoint_schema")),
    )
    artifact = artifact_relative(arm, fold_id)
    expected_checkpoint_schema = (
        f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-oof-fold-{fold_id:02d}-"
        "treatment-checkpoint-v1"
    )
    if (
        len(train.class_weights) != 2
        or train.batch_size != source_config.train.batch_size
        or train.learning_rate != source_config.train.learning_rate
        or train.seed != TRAINING_SEED
        or train.dataloader_seed != TRAINING_SEED
        or train.device != source_config.train.device
        or train.epochs != FIXED_EPOCHS
        or train.gradient_clip_norm != source_config.train.gradient_clip_norm
        or train.class_weights != source_config.train.class_weights
        or train.loss_mode != source_config.train.loss_mode
        or train.class_weight_normalization != source_config.train.class_weight_normalization
        or train.prefix_auxiliary_weight != source_config.train.prefix_auxiliary_weight
        or train.prefix_auxiliary_min_events != source_config.train.prefix_auxiliary_min_events
        or train.checkpoint_path != str(artifact / "final_model.pt").replace("\\", "/")
        or train.last_checkpoint_path != str(artifact / "last_model.pt").replace("\\", "/")
        or train.formal_artifact_dir != str(artifact).replace("\\", "/")
        or train.epoch_checkpoint_dir != str(artifact / "epochs").replace("\\", "/")
        or train.checkpoint_schema != expected_checkpoint_schema
    ):
        raise ValueError("V23 train contract differs beyond the official CfC encoder")

    config = V23Config(
        schema_version=schema_version,
        config_name=config_name,
        arm=arm,
        fold_id=fold_id,
        source_config_path=source_config_path,
        source_config_sha256=source_config_sha256,
        source_config_identity_sha256=source_config_identity,
        source_prediction_path=source_prediction_path,
        source_prediction_sha256=source_prediction_sha256,
        official_cfc=official,
        train=train,
    )
    return config, source_config, actual_source_path


__all__ = [
    "OfficialCfCConfig",
    "V23Config",
    "V23TrainConfig",
    "config_identity_sha256",
    "load_v23_config",
    "sha256_file",
]
