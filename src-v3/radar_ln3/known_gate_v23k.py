"""Frozen contract for V23k causal TR/RD observation-intensity inputs.

V23k is a low-risk alternative to the rejected V23gr state-gain mechanism.
It retains V23a's official-CfC and positive-dt fixed-retention state paths
unchanged, then appends two *causal input* features to fusion:

``log1p(c_TR)``, ``log1p(c_RD)``.

The two decayed counts use a fixed rate ``kappa=1/15.025 s^-1``.  They are
not learned, are not normalized from validation, and do not change CfC
timespans, hidden-state equations or adapter gains.
"""

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
    source_v23_parent_order_hashes,
    source_v23_training,
)
from .known_gate_v23e_selection import V23E_SELECTOR_SCHEMA
from .known_gate_official_cfc import CausalObservationIntensitySpec
from .official_cfc_state_adapters import ExternalStateAdapterSpec


V23K_VARIANT = "official_cfc_positive_dt_fixed_retention_causal_tr_rd_observation_intensity"
V23K_FIXED_TAU_SECONDS = 15.025
V23K_KAPPA_PER_SECOND = 1.0 / V23K_FIXED_TAU_SECONDS
V23K_FEATURE_NAMES = (
    "causal_log1p_decayed_tr_observation_intensity",
    "causal_log1p_decayed_rd_observation_intensity",
)
V23K_CONFIG_SCHEMA = "RADAR-LN3-V23k-known-gate-official-cfc-observation-intensity-config-v1"
V23K_CONTRACT_SCHEMA = "RADAR-LN3-V23k-known-gate-official-cfc-observation-intensity-contract-v1"
V23K_READINESS_SCHEMA = "RADAR-LN3-V23k-known-gate-official-cfc-observation-intensity-readiness-v1"
V23K_CHECKPOINT_SCHEMA = "RADAR-LN3-V23k-known-gate-official-cfc-observation-intensity-seed43-checkpoint-v1"
V23K_REPORT_SCHEMA = "RADAR-LN3-V23k-known-gate-official-cfc-observation-intensity-report-v1"


def config_relative() -> Path:
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V23k-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "causal-tr-rd-observation-intensity-seed43-config.json"
    )


def artifact_relative() -> Path:
    return Path(
        "artifacts/"
        "RADAR-LN3-V23k-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "causal-tr-rd-observation-intensity-seed43-formal"
    )


def contract_relative() -> Path:
    return Path(
        "data/processed/"
        "RADAR-LN3-V23k-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "causal-tr-rd-observation-intensity-contract.json"
    )


def readiness_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23k-known-gate-official-cfc-observation-intensity-training-readiness.json"
    )


def report_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23k-known-gate-official-cfc-observation-intensity-paired-validation.json"
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


def observation_intensity_spec() -> CausalObservationIntensitySpec:
    return CausalObservationIntensitySpec(kappa_per_second=V23K_KAPPA_PER_SECOND)


@dataclass(frozen=True)
class V23KTrainConfig:
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
class V23KConfig:
    schema_version: str
    config_name: str
    variant: str
    source_config_path: str
    source_config_sha256: str
    source_config_identity_sha256: str
    source_v23a: dict[str, Any]
    adapter: ExternalStateAdapterSpec
    observation_intensity: CausalObservationIntensitySpec
    selector: dict[str, Any]
    train: V23KTrainConfig


def config_identity_sha256(config: V23KConfig) -> str:
    return canonical_sha256(
        {
            "schema_version": config.schema_version,
            "config_name": config.config_name,
            "variant": config.variant,
            "source_config_path": config.source_config_path,
            "source_config_sha256": config.source_config_sha256,
            "source_config_identity_sha256": config.source_config_identity_sha256,
            "source_v23a": config.source_v23a,
            "adapter": config.adapter.as_dict(),
            "observation_intensity": config.observation_intensity.as_dict(),
            "selector": config.selector,
            "train": config.train.__dict__,
        }
    )


def source_v23a_control(root: Path) -> dict[str, Any]:
    """Bind V23k to the completed V23a state-path direct control."""

    root = root.resolve()
    config_path = (root / v23a_config_relative(V23A_VARIANT)).resolve()
    readiness_path = (root / v23a_readiness_relative(V23A_VARIANT)).resolve()
    artifact = (root / v23a_artifact_relative(V23A_VARIANT)).resolve()
    summary_path = artifact / "training-summary.json"
    checkpoint_path = artifact / "best_model.pt"
    required = (config_path, readiness_path, summary_path, checkpoint_path)
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("V23k requires completed V23a files: " + ", ".join(missing))
    config, _source, _source_path = load_adapter_config(root, V23A_VARIANT)
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    history = list(summary.get("history") or ())
    _v23_summary_path, v23_summary, _v23_checkpoint_path, _v23_checkpoint_sha = source_v23_training(
        root
    )
    source_parent_orders = source_v23_parent_order_hashes(v23_summary)
    parent_orders_match = len(history) == FIXED_EPOCHS and all(
        int(dict(row).get("epoch", -1)) == epoch
        and dict(row).get("parent_order_sha256") == source_parent_orders[epoch - 1]
        and dict(row).get("v23_control_parent_order_sha256")
        == source_parent_orders[epoch - 1]
        and dict(row).get("parent_order_exactly_matches_v23_control") is True
        and int(dict(row).get("parent_order_count", -1)) == 986
        and int(dict(row).get("parent_order_unique_count", -1)) == 986
        for epoch, row in enumerate(history, start=1)
    )
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != V23A_VARIANT
        or summary.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or int(summary.get("seed", -1)) != TRAINING_SEED
        or len(history) != FIXED_EPOCHS
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or not parent_orders_match
    ):
        raise ValueError("V23k V23a control is incomplete")
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
        "v23e_selector_rule_schema": V23E_SELECTOR_SCHEMA,
        "state_path": "official CfC raw dt plus V23a positive-dt fixed retention; unchanged in V23k",
    }


def load_config(root: Path) -> tuple[V23KConfig, KnownGateConfig, Path]:
    root = root.resolve()
    path = (root / config_relative()).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    expected_top_level = {
        "schema_version",
        "config_name",
        "variant",
        "source",
        "adapter",
        "observation_intensity",
        "selector",
        "train",
    }
    if set(payload) != expected_top_level:
        raise ValueError("V23k config top-level fields differ")
    if (
        payload.get("schema_version") != V23K_CONFIG_SCHEMA
        or payload.get("variant") != V23K_VARIANT
        or payload.get("config_name") != path.stem.removesuffix("-config")
    ):
        raise ValueError("V23k config identity differs")
    source = dict(payload["source"])
    if set(source) != {
        "config_path",
        "config_sha256",
        "config_identity_sha256",
        "v23a_control",
    }:
        raise ValueError("V23k source fields differ")
    source_path_value = str(source["config_path"])
    if source_path_value.replace("\\", "/") != str(SOURCE_CONFIG_RELATIVE).replace("\\", "/"):
        raise ValueError("V23k source must be frozen V21 seed43 baseline")
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
    ):
        raise ValueError("V23k frozen V21 source differs")
    direct_control = dict(source["v23a_control"])
    if direct_control != source_v23a_control(root):
        raise ValueError("V23k V23a control binding is stale")
    adapter = dict(payload["adapter"])
    expected_adapter = load_adapter_config(root, V23A_VARIANT)[0].adapter
    if adapter != expected_adapter.as_dict():
        raise ValueError("V23k adapter must exactly reuse V23a fixed retention")
    intensity_payload = dict(payload["observation_intensity"])
    intensity = observation_intensity_spec()
    if intensity_payload != intensity.as_dict():
        raise ValueError("V23k causal TR/RD observation-intensity contract differs")
    selector = dict(payload["selector"])
    expected_selector = {
        "schema_version": V23E_SELECTOR_SCHEMA,
        "name": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
        "source_v23a_legacy_reference_epoch": int(direct_control["legacy_best_epoch"]),
    }
    if selector != expected_selector:
        raise ValueError("V23k selector must exactly use the frozen V23e rule")
    train_payload = dict(payload["train"])
    if set(train_payload) != set(V23KTrainConfig.__dataclass_fields__):
        raise ValueError("V23k train fields differ")
    train = V23KTrainConfig(
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
        or train.prefix_auxiliary_weight != source_train.prefix_auxiliary_weight
        or train.prefix_auxiliary_min_events != source_train.prefix_auxiliary_min_events
        or train.checkpoint_path != f"{artifact}/best_model.pt"
        or train.last_checkpoint_path != f"{artifact}/last_model.pt"
        or train.formal_artifact_dir != artifact
        or train.epoch_checkpoint_dir != f"{artifact}/epochs"
        or train.checkpoint_schema != V23K_CHECKPOINT_SCHEMA
        or not train.record_parent_order_hash
        or not train.save_every_epoch
        or train.cuda_deterministic_algorithms != source_train.cuda_deterministic_algorithms
        or not train.full_parameter_training
    ):
        raise ValueError("V23k protocol differs beyond its causal input features")
    source_v23_training(root)
    source_runtime_report(root)
    return (
        V23KConfig(
            schema_version=V23K_CONFIG_SCHEMA,
            config_name=str(payload["config_name"]),
            variant=V23K_VARIANT,
            source_config_path=source_path_value,
            source_config_sha256=str(source["config_sha256"]).lower(),
            source_config_identity_sha256=str(source["config_identity_sha256"]).lower(),
            source_v23a=direct_control,
            adapter=expected_adapter,
            observation_intensity=intensity,
            selector=selector,
            train=train,
        ),
        source_config,
        source_path,
    )


__all__ = [
    "V23K_CHECKPOINT_SCHEMA",
    "V23K_CONFIG_SCHEMA",
    "V23K_CONTRACT_SCHEMA",
    "V23K_FEATURE_NAMES",
    "V23K_FIXED_TAU_SECONDS",
    "V23K_KAPPA_PER_SECOND",
    "V23K_READINESS_SCHEMA",
    "V23K_REPORT_SCHEMA",
    "V23K_VARIANT",
    "V23KConfig",
    "V23KTrainConfig",
    "artifact_relative",
    "config_identity_sha256",
    "config_relative",
    "contract_relative",
    "load_config",
    "observation_intensity_spec",
    "readiness_relative",
    "report_relative",
    "source_v23a_control",
]
