"""Contracts for V23f's prefix-weight-only official-CfC experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .known_gate import KnownGateConfig, known_gate_config_identity_sha256, load_known_gate_config
from .known_gate_v23 import FIXED_EPOCHS, SOURCE_CONFIG_RELATIVE, TRAINING_SEED, sha256_file
from .known_gate_v23_adapters import (
    V23A_VARIANT,
    artifact_relative as v23a_artifact_relative,
    config_identity_sha256 as v23a_config_identity_sha256,
    config_relative as v23a_config_relative,
    load_adapter_config,
    readiness_relative as v23a_readiness_relative,
    source_runtime_report,
    source_v23_training,
)
from .official_cfc_state_adapters import ExternalStateAdapterSpec


V23F_VARIANT = "official_cfc_positive_dt_fixed_retention_prefix_weight_1_0"
V23F_PREFIX_AUXILIARY_WEIGHT = 1.0
V23F_CONFIG_SCHEMA = "RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-config-v1"
V23F_CONTRACT_SCHEMA = "RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-contract-v1"
V23F_READINESS_SCHEMA = "RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-readiness-v1"
V23F_CHECKPOINT_SCHEMA = "RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-seed43-checkpoint-v1"
V23F_REPORT_SCHEMA = "RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-report-v1"


def config_relative() -> Path:
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V23f-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "prefix-weight-1-0-seed43-config.json"
    )


def artifact_relative() -> Path:
    return Path(
        "artifacts/"
        "RADAR-LN3-V23f-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "prefix-weight-1-0-seed43-formal"
    )


def contract_relative() -> Path:
    return Path(
        "data/processed/"
        "RADAR-LN3-V23f-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "prefix-weight-1-0-contract.json"
    )


def readiness_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-training-readiness.json"
    )


def report_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23f-known-gate-official-cfc-prefix-weight-paired-validation.json"
    )


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class V23FTrainConfig:
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
class V23FConfig:
    schema_version: str
    config_name: str
    variant: str
    source_config_path: str
    source_config_sha256: str
    source_config_identity_sha256: str
    source_v23a: dict[str, Any]
    adapter: ExternalStateAdapterSpec
    selector: dict[str, Any]
    train: V23FTrainConfig


def config_identity_sha256(config: V23FConfig) -> str:
    return canonical_sha256(
        {
            "schema_version": config.schema_version,
            "config_name": config.config_name,
            "variant": config.variant,
            "source_config_path": config.source_config_path,
            "source_config_sha256": config.source_config_sha256,
            "source_config_identity_sha256": config.source_config_identity_sha256,
            "source_v23a": config.source_v23a,
            "adapter": config.adapter.__dict__,
            "selector": config.selector,
            "train": config.train.__dict__,
        }
    )


def source_v23a_control(root: Path) -> dict[str, Any]:
    """Bind V23f to the completed V23a run and frozen V23e selector rule.

    V23f may be prepared before the read-only V23e scan has finished: the
    selection *rule* is already pre-registered in source, while V23e's result
    is a diagnostic rather than a training prerequisite.  The completed V23e
    report is bound later by V23f evaluation, before any comparative claim.
    """

    root = root.resolve()
    config_path = (root / v23a_config_relative(V23A_VARIANT)).resolve()
    readiness_path = (root / v23a_readiness_relative(V23A_VARIANT)).resolve()
    artifact = (root / v23a_artifact_relative(V23A_VARIANT)).resolve()
    summary_path = artifact / "training-summary.json"
    checkpoint_path = artifact / "best_model.pt"
    required = (
        config_path,
        readiness_path,
        summary_path,
        checkpoint_path,
    )
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("V23f requires completed V23a/V23e controls: " + ", ".join(missing))
    config, _source, _source_path = load_adapter_config(root, V23A_VARIANT)
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != V23A_VARIANT
        or int(summary.get("seed", -1)) != TRAINING_SEED
        or len(list(summary.get("history") or ())) != FIXED_EPOCHS
    ):
        raise ValueError("V23f V23a control is incomplete")
    return {
        "variant": V23A_VARIANT,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": v23a_config_identity_sha256(config),
        "readiness_path": _relative(root, readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "summary_path": _relative(root, summary_path),
        "summary_sha256": sha256_file(summary_path),
        "legacy_best_checkpoint_path": _relative(root, checkpoint_path),
        "legacy_best_checkpoint_sha256": sha256_file(checkpoint_path),
        "legacy_best_epoch": int(summary["best_epoch"]),
        "v23e_selector_rule_schema": "RADAR-LN3-V23e-parent-balanced-checkpoint-selector-v1",
    }


def load_config(root: Path) -> tuple[V23FConfig, KnownGateConfig, Path]:
    root = root.resolve()
    path = (root / config_relative()).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    if set(payload) != {
        "schema_version",
        "config_name",
        "variant",
        "source",
        "adapter",
        "selector",
        "train",
    }:
        raise ValueError("V23f config top-level fields differ")
    if (
        payload.get("schema_version") != V23F_CONFIG_SCHEMA
        or payload.get("variant") != V23F_VARIANT
        or payload.get("config_name") != path.stem.removesuffix("-config")
    ):
        raise ValueError("V23f config identity differs")
    source = dict(payload["source"])
    if set(source) != {
        "config_path",
        "config_sha256",
        "config_identity_sha256",
        "v23a_control",
    }:
        raise ValueError("V23f source fields differ")
    source_path_value = str(source["config_path"])
    if source_path_value.replace("\\", "/") != str(SOURCE_CONFIG_RELATIVE).replace("\\", "/"):
        raise ValueError("V23f source must be frozen V21 seed43 baseline")
    source_path = _resolve(root, source_path_value)
    source_config = load_known_gate_config(source_path)
    if (
        not source_path.is_file()
        or sha256_file(source_path) != str(source["config_sha256"]).lower()
        or known_gate_config_identity_sha256(source_config)
        != str(source["config_identity_sha256"]).lower()
        or source_config.variant != "baseline"
        or source_config.model.evidence_dim != 0
        or source_config.model.rd_temporal_residual_dim != 0
        or source_config.train.seed != TRAINING_SEED
        or source_config.train.dataloader_seed != TRAINING_SEED
        or source_config.train.epochs != FIXED_EPOCHS
        or source_config.train.class_weight_normalization != "batch_weight_sum"
    ):
        raise ValueError("V23f frozen V21 source differs")
    direct_control = dict(source["v23a_control"])
    if direct_control != source_v23a_control(root):
        raise ValueError("V23f V23a/V23e control binding is stale")
    adapter_payload = dict(payload["adapter"])
    expected_adapter = load_adapter_config(root, V23A_VARIANT)[0].adapter
    if adapter_payload != expected_adapter.__dict__:
        raise ValueError("V23f adapter must exactly reuse V23a fixed retention")
    selector = dict(payload["selector"])
    expected_selector = {
        "schema_version": "RADAR-LN3-V23e-parent-balanced-checkpoint-selector-v1",
        "name": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
        "source_v23a_legacy_reference_epoch": int(direct_control["legacy_best_epoch"]),
    }
    if selector != expected_selector:
        raise ValueError("V23f selector must exactly use the frozen V23e rule")
    train_payload = dict(payload["train"])
    expected_fields = set(V23FTrainConfig.__dataclass_fields__)
    if set(train_payload) != expected_fields:
        raise ValueError("V23f train fields differ")
    train = V23FTrainConfig(
        batch_size=int(train_payload["batch_size"]),
        learning_rate=float(train_payload["learning_rate"]),
        seed=int(train_payload["seed"]),
        dataloader_seed=int(train_payload["dataloader_seed"]),
        device=str(train_payload["device"]),
        epochs=int(train_payload["epochs"]),
        early_stopping_patience=int(train_payload["early_stopping_patience"]),
        gradient_clip_norm=float(train_payload["gradient_clip_norm"]),
        checkpoint_selection_metric=str(train_payload["checkpoint_selection_metric"]),
        task_min_precision_recall_target=float(train_payload["task_min_precision_recall_target"]),
        class_weights=tuple(float(value) for value in train_payload["class_weights"]),
        loss_mode=str(train_payload["loss_mode"]),
        class_weight_normalization=str(train_payload["class_weight_normalization"]),
        prefix_auxiliary_weight=float(train_payload["prefix_auxiliary_weight"]),
        prefix_auxiliary_min_events=int(train_payload["prefix_auxiliary_min_events"]),
        checkpoint_path=str(train_payload["checkpoint_path"]),
        last_checkpoint_path=str(train_payload["last_checkpoint_path"]),
        formal_artifact_dir=str(train_payload["formal_artifact_dir"]),
        epoch_checkpoint_dir=str(train_payload["epoch_checkpoint_dir"]),
        checkpoint_schema=str(train_payload["checkpoint_schema"]),
        record_parent_order_hash=bool(train_payload["record_parent_order_hash"]),
        save_every_epoch=bool(train_payload["save_every_epoch"]),
        cuda_deterministic_algorithms=bool(train_payload["cuda_deterministic_algorithms"]),
        full_parameter_training=bool(train_payload["full_parameter_training"]),
    )
    artifact = str(artifact_relative()).replace("\\", "/")
    source_train = source_config.train
    if (
        train.batch_size != source_train.batch_size
        or train.learning_rate != source_train.learning_rate
        or train.seed != TRAINING_SEED
        or train.dataloader_seed != TRAINING_SEED
        or train.device != source_train.device
        or train.epochs != FIXED_EPOCHS
        or train.early_stopping_patience != 0
        or train.gradient_clip_norm != source_train.gradient_clip_norm
        or train.checkpoint_selection_metric
        != "V23e_parent_final_and_parent_balanced_eligible_prefix_selector"
        or train.task_min_precision_recall_target != source_train.task_min_precision_recall_target
        or train.class_weights != source_train.class_weights
        or train.loss_mode != source_train.loss_mode
        or train.class_weight_normalization != source_train.class_weight_normalization
        or train.prefix_auxiliary_weight != V23F_PREFIX_AUXILIARY_WEIGHT
        or train.prefix_auxiliary_min_events != source_train.prefix_auxiliary_min_events
        or train.checkpoint_path != f"{artifact}/best_model.pt"
        or train.last_checkpoint_path != f"{artifact}/last_model.pt"
        or train.formal_artifact_dir != artifact
        or train.epoch_checkpoint_dir != f"{artifact}/epochs"
        or train.checkpoint_schema != V23F_CHECKPOINT_SCHEMA
        or not train.record_parent_order_hash
        or not train.save_every_epoch
        or train.cuda_deterministic_algorithms != source_train.cuda_deterministic_algorithms
        or not train.full_parameter_training
    ):
        raise ValueError("V23f protocol differs beyond its pre-registered prefix weight")
    source_v23_training(root)
    source_runtime_report(root)
    return (
        V23FConfig(
            schema_version=V23F_CONFIG_SCHEMA,
            config_name=str(payload["config_name"]),
            variant=V23F_VARIANT,
            source_config_path=source_path_value,
            source_config_sha256=str(source["config_sha256"]).lower(),
            source_config_identity_sha256=str(source["config_identity_sha256"]).lower(),
            source_v23a=direct_control,
            adapter=expected_adapter,
            selector=selector,
            train=train,
        ),
        source_config,
        source_path,
    )


__all__ = ["V23F_CHECKPOINT_SCHEMA", "V23F_CONFIG_SCHEMA", "V23F_CONTRACT_SCHEMA", "V23F_PREFIX_AUXILIARY_WEIGHT", "V23F_READINESS_SCHEMA", "V23F_REPORT_SCHEMA", "V23F_VARIANT", "V23FConfig", "V23FTrainConfig", "artifact_relative", "config_identity_sha256", "config_relative", "contract_relative", "load_config", "readiness_relative", "report_relative", "source_v23a_control"]