"""Frozen contract for the V23 pure official-CfC KnownGate baseline.

V23 is deliberately a single full-training baseline, not an OOF experiment.
It replaces the project-local recurrent encoder in the frozen V21 seed43
baseline with dense ``ncps.torch.CfC`` and leaves five-fold OOF for a later,
separately approved confirmation phase.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .known_gate import KnownGateConfig, known_gate_config_identity_sha256, load_known_gate_config
from .official_cfc import (
    OFFICIAL_CFC_ACTIVATION,
    OFFICIAL_CFC_CLASS_MODULE,
    OFFICIAL_CFC_CLASS_PATH,
    OFFICIAL_CFC_DIRECT_URL,
    OFFICIAL_CFC_DISTRIBUTION,
    OFFICIAL_CFC_DISTRIBUTION_VERSION,
    OFFICIAL_CFC_GIT_COMMIT,
    OFFICIAL_CFC_MODE,
    OFFICIAL_CFC_STATE_EQUATION_OWNER,
    official_cfc_runtime_binding,
)


FIXED_EPOCHS = 20
TRAINING_SEED = 43
SOURCE_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-config.json"
)
SOURCE_ARTIFACT_RELATIVE = Path(
    "artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-formal"
)
SOURCE_SUMMARY_RELATIVE = SOURCE_ARTIFACT_RELATIVE / "training-summary.json"
CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-seed43-config.json"
)
ARTIFACT_RELATIVE = Path(
    "artifacts/RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-seed43-formal"
)
CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-contract.json"
)
READINESS_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V23-known-gate-official-cfc-baseline-training-readiness.json"
)
RUNTIME_COMPATIBILITY_REPORT_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V23-official-cfc-runtime-compatibility.json"
)
REPORT_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V23-known-gate-official-cfc-baseline-validation.json"
)

CONFIG_SCHEMA = "RADAR-LN3-V23-known-gate-official-cfc-baseline-config-v1"
CONTRACT_SCHEMA = "RADAR-LN3-V23-known-gate-official-cfc-baseline-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V23-known-gate-official-cfc-baseline-readiness-v1"
RUNTIME_COMPATIBILITY_REPORT_SCHEMA = "RADAR-LN3-V23-official-cfc-runtime-compatibility-v1"
RUNTIME_COMPATIBILITY_REQUIRED_CHECKS = (
    "exact_ncps_distribution_version",
    "exact_official_runtime_binding",
    "official_cfc_forward_has_no_padding_mask",
    "official_cfc_batch_two_heterogeneous_timespans_are_not_supported",
    "official_cfc_zero_dt_is_finite_but_not_state_identity",
    "official_ltc_zero_dt_is_not_safe_for_v23_direct_swap",
    "official_cfc_baseline_is_dense_not_wired",
)
CHECKPOINT_SCHEMA = "RADAR-LN3-V23-known-gate-official-cfc-baseline-seed43-checkpoint-v1"
REPORT_SCHEMA = "RADAR-LN3-V23-known-gate-official-cfc-baseline-validation-report-v1"


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


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V23 config is missing {key}")
    return mapping[key]


@dataclass(frozen=True)
class OfficialCfCConfig:
    distribution: str
    distribution_version: str
    direct_url: str
    git_commit: str
    class_path: str
    class_module: str
    mode: str
    activation: str
    input_dim: int
    hidden_dim: int
    backbone_units: int
    backbone_layers: int
    backbone_dropout: float
    mixed_memory: bool
    proj_size: None
    wiring: None
    return_sequences: bool
    batch_first: bool
    per_parent_forward_for_upstream_batch_timespan_compatibility: bool
    state_equation_owner: str
    dt_semantics: str


@dataclass(frozen=True)
class V23TrainConfig:
    batch_size: int
    learning_rate: float
    seed: int
    dataloader_seed: int
    device: str
    epochs: int
    early_stopping_patience: int
    gradient_clip_norm: float
    checkpoint_selection_metric: str
    task_min_precision_recall_target: float
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
    record_parent_order_hash: bool
    save_every_epoch: bool
    cuda_deterministic_algorithms: bool
    full_parameter_training: bool


@dataclass(frozen=True)
class V23Config:
    schema_version: str
    config_name: str
    variant: str
    source_config_path: str
    source_config_sha256: str
    source_config_identity_sha256: str
    official_cfc: OfficialCfCConfig
    train: V23TrainConfig


def config_identity_sha256(config: V23Config) -> str:
    return canonical_sha256(
        {
            "schema_version": config.schema_version,
            "config_name": config.config_name,
            "variant": config.variant,
            "source_config_path": config.source_config_path,
            "source_config_sha256": config.source_config_sha256,
            "source_config_identity_sha256": config.source_config_identity_sha256,
            "official_cfc": config.official_cfc.__dict__,
            "train": config.train.__dict__,
        }
    )


def load_v23_config(
    project_root: Path, path: Path
) -> tuple[V23Config, KnownGateConfig, Path]:
    """Load V23 and validate its immutable V21 baseline source binding."""

    root = project_root.resolve()
    config_path = path.resolve()
    if config_path != (root / CONFIG_RELATIVE).resolve():
        raise ValueError("V23 config path is outside the frozen baseline namespace")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if set(payload) != {
        "schema_version",
        "config_name",
        "variant",
        "source",
        "official_cfc",
        "train",
    }:
        raise ValueError("V23 config has unexpected or missing top-level fields")
    if str(_require(payload, "schema_version")) != CONFIG_SCHEMA:
        raise ValueError("V23 config schema is invalid")
    if str(_require(payload, "config_name")) != config_path.stem.removesuffix("-config"):
        raise ValueError("V23 config filename/name mismatch")
    if str(_require(payload, "variant")) != "official_cfc_baseline":
        raise ValueError("V23 must be the pure official-CfC baseline variant")

    source = dict(_require(payload, "source"))
    if set(source) != {
        "role",
        "config_path",
        "config_sha256",
        "config_identity_sha256",
    }:
        raise ValueError("V23 source binding fields differ from the frozen contract")
    if str(_require(source, "role")) != "frozen V21 seed43 formal baseline":
        raise ValueError("V23 source role differs from the formal baseline contract")
    source_config_path = str(_require(source, "config_path"))
    if source_config_path.replace("\\", "/") != str(SOURCE_CONFIG_RELATIVE).replace("\\", "/"):
        raise ValueError("V23 source config path is invalid")
    source_path = _resolve(root, source_config_path)
    source_sha256 = str(_require(source, "config_sha256")).lower()
    source_identity = str(_require(source, "config_identity_sha256")).lower()
    if not source_path.is_file() or sha256_file(source_path) != source_sha256:
        raise ValueError("V23 source config hash is stale")
    source_config = load_known_gate_config(source_path)
    if known_gate_config_identity_sha256(source_config) != source_identity:
        raise ValueError("V23 source config identity is stale")
    if (
        source_config.variant != "baseline"
        or source_config.model.evidence_dim != 0
        or source_config.model.rd_temporal_residual_dim != 0
        or source_config.train.seed != TRAINING_SEED
        or source_config.train.dataloader_seed != TRAINING_SEED
        or source_config.train.epochs != FIXED_EPOCHS
        or source_config.train.early_stopping_patience != 0
        or source_config.train.class_weight_normalization != "batch_weight_sum"
    ):
        raise ValueError("V23 source is not the frozen V21 seed43 no-evidence baseline")

    official_payload = dict(_require(payload, "official_cfc"))
    expected_official_fields = {
        "distribution",
        "distribution_version",
        "direct_url",
        "git_commit",
        "class_path",
        "class_module",
        "mode",
        "activation",
        "input_dim",
        "hidden_dim",
        "backbone_units",
        "backbone_layers",
        "backbone_dropout",
        "mixed_memory",
        "proj_size",
        "wiring",
        "return_sequences",
        "batch_first",
        "per_parent_forward_for_upstream_batch_timespan_compatibility",
        "state_equation_owner",
        "dt_semantics",
    }
    if set(official_payload) != expected_official_fields:
        raise ValueError("V23 official CfC fields differ from the frozen contract")
    official = OfficialCfCConfig(
        distribution=str(_require(official_payload, "distribution")),
        distribution_version=str(_require(official_payload, "distribution_version")),
        direct_url=str(_require(official_payload, "direct_url")),
        git_commit=str(_require(official_payload, "git_commit")),
        class_path=str(_require(official_payload, "class_path")),
        class_module=str(_require(official_payload, "class_module")),
        mode=str(_require(official_payload, "mode")),
        activation=str(_require(official_payload, "activation")),
        input_dim=int(_require(official_payload, "input_dim")),
        hidden_dim=int(_require(official_payload, "hidden_dim")),
        backbone_units=int(_require(official_payload, "backbone_units")),
        backbone_layers=int(_require(official_payload, "backbone_layers")),
        backbone_dropout=float(_require(official_payload, "backbone_dropout")),
        mixed_memory=bool(_require(official_payload, "mixed_memory")),
        proj_size=_require(official_payload, "proj_size"),
        wiring=_require(official_payload, "wiring"),
        return_sequences=bool(_require(official_payload, "return_sequences")),
        batch_first=bool(_require(official_payload, "batch_first")),
        per_parent_forward_for_upstream_batch_timespan_compatibility=bool(
            _require(official_payload, "per_parent_forward_for_upstream_batch_timespan_compatibility")
        ),
        state_equation_owner=str(_require(official_payload, "state_equation_owner")),
        dt_semantics=str(_require(official_payload, "dt_semantics")),
    )
    if (
        official.distribution != OFFICIAL_CFC_DISTRIBUTION
        or official.distribution_version != OFFICIAL_CFC_DISTRIBUTION_VERSION
        or official.direct_url != OFFICIAL_CFC_DIRECT_URL
        or official.git_commit != OFFICIAL_CFC_GIT_COMMIT
        or official.class_path != OFFICIAL_CFC_CLASS_PATH
        or official.class_module != OFFICIAL_CFC_CLASS_MODULE
        or official.mode != OFFICIAL_CFC_MODE
        or official.activation != OFFICIAL_CFC_ACTIVATION
        or official.input_dim != 48
        or official.hidden_dim != 48
        or official.backbone_units != 48
        or official.backbone_layers != 1
        or official.backbone_dropout != 0.0
        or official.mixed_memory
        or official.proj_size is not None
        or official.wiring is not None
        or not official.return_sequences
        or not official.batch_first
        or not official.per_parent_forward_for_upstream_batch_timespan_compatibility
        or official.state_equation_owner != OFFICIAL_CFC_STATE_EQUATION_OWNER
        or "without_cap" not in official.dt_semantics
    ):
        raise ValueError("V23 is not the frozen pure official dense-CfC specification")
    runtime = official_cfc_runtime_binding()
    if any(runtime[key] != getattr(official, key) for key in runtime):
        raise RuntimeError("installed official ncps runtime differs from V23 config")

    train_payload = dict(_require(payload, "train"))
    expected_train_fields = {
        "batch_size",
        "learning_rate",
        "seed",
        "dataloader_seed",
        "device",
        "epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
        "checkpoint_selection_metric",
        "task_min_precision_recall_target",
        "class_weights",
        "loss_mode",
        "class_weight_normalization",
        "prefix_auxiliary_weight",
        "prefix_auxiliary_min_events",
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "checkpoint_schema",
        "record_parent_order_hash",
        "save_every_epoch",
        "cuda_deterministic_algorithms",
        "full_parameter_training",
    }
    if set(train_payload) != expected_train_fields:
        raise ValueError("V23 training fields differ from the frozen contract")
    train = V23TrainConfig(
        batch_size=int(_require(train_payload, "batch_size")),
        learning_rate=float(_require(train_payload, "learning_rate")),
        seed=int(_require(train_payload, "seed")),
        dataloader_seed=int(_require(train_payload, "dataloader_seed")),
        device=str(_require(train_payload, "device")),
        epochs=int(_require(train_payload, "epochs")),
        early_stopping_patience=int(_require(train_payload, "early_stopping_patience")),
        gradient_clip_norm=float(_require(train_payload, "gradient_clip_norm")),
        checkpoint_selection_metric=str(_require(train_payload, "checkpoint_selection_metric")),
        task_min_precision_recall_target=float(_require(train_payload, "task_min_precision_recall_target")),
        class_weights=tuple(float(value) for value in _require(train_payload, "class_weights")),
        loss_mode=str(_require(train_payload, "loss_mode")),
        class_weight_normalization=str(_require(train_payload, "class_weight_normalization")),
        prefix_auxiliary_weight=float(_require(train_payload, "prefix_auxiliary_weight")),
        prefix_auxiliary_min_events=int(_require(train_payload, "prefix_auxiliary_min_events")),
        checkpoint_path=str(_require(train_payload, "checkpoint_path")),
        last_checkpoint_path=str(_require(train_payload, "last_checkpoint_path")),
        formal_artifact_dir=str(_require(train_payload, "formal_artifact_dir")),
        epoch_checkpoint_dir=str(_require(train_payload, "epoch_checkpoint_dir")),
        checkpoint_schema=str(_require(train_payload, "checkpoint_schema")),
        record_parent_order_hash=bool(_require(train_payload, "record_parent_order_hash")),
        save_every_epoch=bool(_require(train_payload, "save_every_epoch")),
        cuda_deterministic_algorithms=bool(_require(train_payload, "cuda_deterministic_algorithms")),
        full_parameter_training=bool(_require(train_payload, "full_parameter_training")),
    )
    expected_artifact = str(ARTIFACT_RELATIVE).replace("\\", "/")
    if (
        len(train.class_weights) != 2
        or train.batch_size != source_config.train.batch_size
        or train.learning_rate != source_config.train.learning_rate
        or train.seed != TRAINING_SEED
        or train.dataloader_seed != TRAINING_SEED
        or train.device != source_config.train.device
        or train.epochs != FIXED_EPOCHS
        or train.early_stopping_patience != 0
        or train.gradient_clip_norm != source_config.train.gradient_clip_norm
        or train.checkpoint_selection_metric != source_config.train.checkpoint_selection_metric
        or train.task_min_precision_recall_target != source_config.train.task_min_precision_recall_target
        or train.class_weights != source_config.train.class_weights
        or train.loss_mode != source_config.train.loss_mode
        or train.class_weight_normalization != source_config.train.class_weight_normalization
        or train.prefix_auxiliary_weight != source_config.train.prefix_auxiliary_weight
        or train.prefix_auxiliary_min_events != source_config.train.prefix_auxiliary_min_events
        or train.checkpoint_path != f"{expected_artifact}/best_model.pt"
        or train.last_checkpoint_path != f"{expected_artifact}/last_model.pt"
        or train.formal_artifact_dir != expected_artifact
        or train.epoch_checkpoint_dir != f"{expected_artifact}/epochs"
        or train.checkpoint_schema != CHECKPOINT_SCHEMA
        or not train.record_parent_order_hash
        or not train.save_every_epoch
        or train.cuda_deterministic_algorithms != source_config.train.cuda_deterministic_algorithms
        or not train.full_parameter_training
    ):
        raise ValueError("V23 train contract differs beyond the official CfC encoder")

    return (
        V23Config(
            schema_version=CONFIG_SCHEMA,
            config_name=str(_require(payload, "config_name")),
            variant="official_cfc_baseline",
            source_config_path=source_config_path,
            source_config_sha256=source_sha256,
            source_config_identity_sha256=source_identity,
            official_cfc=official,
            train=train,
        ),
        source_config,
        source_path,
    )


__all__ = ["ARTIFACT_RELATIVE", "CHECKPOINT_SCHEMA", "CONFIG_RELATIVE", "CONFIG_SCHEMA", "CONTRACT_RELATIVE", "CONTRACT_SCHEMA", "FIXED_EPOCHS", "READINESS_RELATIVE", "READINESS_SCHEMA", "REPORT_RELATIVE", "REPORT_SCHEMA", "RUNTIME_COMPATIBILITY_REPORT_RELATIVE", "RUNTIME_COMPATIBILITY_REQUIRED_CHECKS", "RUNTIME_COMPATIBILITY_REPORT_SCHEMA", "SOURCE_CONFIG_RELATIVE", "SOURCE_ARTIFACT_RELATIVE", "SOURCE_SUMMARY_RELATIVE", "TRAINING_SEED", "OfficialCfCConfig", "V23Config", "V23TrainConfig", "canonical_sha256", "config_identity_sha256", "load_v23_config", "sha256_file"]