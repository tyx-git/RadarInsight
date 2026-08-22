"""Frozen contract for V23j's prefix-ramp official-CfC KnownGate experiment.

V23j is deliberately a training-objective-only continuation of V23a.  It
keeps V23a's official CfC backbone and fixed-retention state adapter, data,
initialization, optimizer and parent order.  The sole treatment is that the
causal-prefix auxiliary coefficient starts at V23a's 0.1 in epoch one and
linearly reaches V23f's already-tested 1.0 in epoch twenty.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .known_gate import (
    KnownGateConfig,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from .known_gate_v23 import (
    FIXED_EPOCHS,
    SOURCE_CONFIG_RELATIVE,
    TRAINING_SEED,
    sha256_file,
)
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
from .known_gate_v23e_selection import V23E_SELECTOR_SCHEMA
from .official_cfc_state_adapters import ExternalStateAdapterSpec


V23J_VARIANT = "official_cfc_positive_dt_fixed_retention_prefix_linear_ramp_0_1_to_1_0"
V23J_PREFIX_RAMP_KIND = "linear_epoch_inclusive"
V23J_PREFIX_RAMP_START_WEIGHT = 0.1
V23J_PREFIX_RAMP_END_WEIGHT = 1.0
V23J_PREFIX_RAMP_START_EPOCH = 1
V23J_PREFIX_RAMP_END_EPOCH = FIXED_EPOCHS

V23J_CONFIG_SCHEMA = "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-config-v1"
V23J_CONTRACT_SCHEMA = "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-contract-v1"
V23J_READINESS_SCHEMA = "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-readiness-v1"
V23J_CHECKPOINT_SCHEMA = "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-seed43-checkpoint-v1"
V23J_REPORT_SCHEMA = "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-report-v1"


def config_relative() -> Path:
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V23j-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "prefix-ramp-0-1-to-1-0-seed43-config.json"
    )


def artifact_relative() -> Path:
    return Path(
        "artifacts/"
        "RADAR-LN3-V23j-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "prefix-ramp-0-1-to-1-0-seed43-formal"
    )


def contract_relative() -> Path:
    return Path(
        "data/processed/"
        "RADAR-LN3-V23j-cq08-known-gate-official-cfc-positive-dt-fixed-retention-"
        "prefix-ramp-0-1-to-1-0-contract.json"
    )


def readiness_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-training-readiness.json"
    )


def report_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23j-known-gate-official-cfc-prefix-ramp-paired-validation.json"
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
class V23JPrefixRamp:
    kind: str
    start_weight: float
    end_weight: float
    start_epoch: int
    end_epoch: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "start_weight": self.start_weight,
            "end_weight": self.end_weight,
            "start_epoch": self.start_epoch,
            "end_epoch": self.end_epoch,
        }


@dataclass(frozen=True)
class V23JTrainConfig:
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
class V23JConfig:
    schema_version: str
    config_name: str
    variant: str
    source_config_path: str
    source_config_sha256: str
    source_config_identity_sha256: str
    source_v23a: dict[str, Any]
    adapter: ExternalStateAdapterSpec
    selector: dict[str, Any]
    prefix_ramp: V23JPrefixRamp
    train: V23JTrainConfig


def prefix_auxiliary_weight_for_epoch(epoch: int, ramp: V23JPrefixRamp) -> float:
    """Return the pre-registered epoch-level prefix-loss coefficient.

    The endpoints and duration are not tuneable fields: V23j bridges the
    already-run V23a (0.1) and V23f (1.0) objectives across the same frozen
    20-epoch training horizon.
    """

    if not ramp_matches_protocol(ramp):
        raise ValueError("V23j prefix ramp differs from the frozen protocol")
    position = int(epoch)
    if not ramp.start_epoch <= position <= ramp.end_epoch:
        raise ValueError("V23j prefix-ramp epoch is outside the fixed training horizon")
    span = ramp.end_epoch - ramp.start_epoch
    if span <= 0:
        raise ValueError("V23j prefix-ramp epoch span must be positive")
    fraction = (position - ramp.start_epoch) / span
    return ramp.start_weight + (ramp.end_weight - ramp.start_weight) * fraction


def ramp_matches_protocol(ramp: V23JPrefixRamp) -> bool:
    return (
        ramp.kind == V23J_PREFIX_RAMP_KIND
        and math.isclose(ramp.start_weight, V23J_PREFIX_RAMP_START_WEIGHT, rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(ramp.end_weight, V23J_PREFIX_RAMP_END_WEIGHT, rel_tol=0.0, abs_tol=1.0e-15)
        and ramp.start_epoch == V23J_PREFIX_RAMP_START_EPOCH
        and ramp.end_epoch == V23J_PREFIX_RAMP_END_EPOCH
    )


def checkpoint_prefix_ramp_metadata(
    *, epoch: int, ramp: V23JPrefixRamp
) -> dict[str, Any]:
    return {
        "prefix_ramp": ramp.as_dict(),
        "prefix_auxiliary_weight_for_epoch": prefix_auxiliary_weight_for_epoch(
            epoch, ramp
        ),
    }


def validate_checkpoint_prefix_ramp_metadata(
    payload: Mapping[str, Any], *, epoch: int, ramp: V23JPrefixRamp
) -> None:
    expected = checkpoint_prefix_ramp_metadata(epoch=epoch, ramp=ramp)
    if dict(payload.get("prefix_ramp") or {}) != expected["prefix_ramp"]:
        raise ValueError("V23j checkpoint prefix-ramp contract differs")
    actual_weight = float(payload.get("prefix_auxiliary_weight_for_epoch", -1.0))
    if not math.isclose(
        actual_weight,
        float(expected["prefix_auxiliary_weight_for_epoch"]),
        rel_tol=0.0,
        abs_tol=1.0e-15,
    ):
        raise ValueError("V23j checkpoint epoch prefix weight differs")


def config_identity_sha256(config: V23JConfig) -> str:
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
            "prefix_ramp": config.prefix_ramp.as_dict(),
            "train": config.train.__dict__,
        }
    )


def source_v23a_control(root: Path) -> dict[str, Any]:
    """Bind V23j to the completed V23a fixed-retention control."""

    config_path, config, summary_path, summary, checkpoint_path = source_v23a_training(
        root
    )
    root = root.resolve()
    readiness_path = (root / v23a_readiness_relative(V23A_VARIANT)).resolve()
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
    }


def source_v23a_training(
    root: Path,
) -> tuple[Path, Any, Path, dict[str, Any], Path]:
    """Load and validate the completed V23a pairing source.

    V23j's epoch-one objective is V23a's objective, not the pure V23 one.
    This helper keeps that distinction explicit while preserving the existing
    V23a metadata shape embedded in the generated V23j config.
    """

    root = root.resolve()
    config_path = (root / v23a_config_relative(V23A_VARIANT)).resolve()
    readiness_path = (root / v23a_readiness_relative(V23A_VARIANT)).resolve()
    artifact = (root / v23a_artifact_relative(V23A_VARIANT)).resolve()
    summary_path = artifact / "training-summary.json"
    checkpoint_path = artifact / "best_model.pt"
    detail_path = artifact / "sample_detail_information.json"
    last_checkpoint_path = artifact / "last_model.pt"
    required = (
        config_path,
        readiness_path,
        summary_path,
        checkpoint_path,
        detail_path,
        last_checkpoint_path,
    )
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError(
            "V23j requires completed V23a fixed-retention control: "
            + ", ".join(missing)
        )
    config, _source, _source_path = load_adapter_config(root, V23A_VARIANT)
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    readiness = dict(json.loads(readiness_path.read_text(encoding="utf-8")))
    source_summary_path, source_summary, _source_checkpoint_path, _source_checkpoint_sha = (
        source_v23_training(root)
    )
    source_order_hashes = tuple(
        str(row["parent_order_sha256"])
        for row in list(source_summary.get("history") or ())
    )
    history = list(summary.get("history") or ())
    history_is_paired = (
        len(history) == FIXED_EPOCHS
        and all(
            int(dict(row).get("epoch", -1)) == epoch
            and str(dict(row).get("parent_order_sha256", ""))
            == source_order_hashes[epoch - 1]
            and str(dict(row).get("v23_control_parent_order_sha256", ""))
            == source_order_hashes[epoch - 1]
            and dict(row).get("parent_order_exactly_matches_v23_control") is True
            and int(dict(row).get("parent_order_count", -1)) == 986
            and int(dict(row).get("parent_order_unique_count", -1)) == 986
            for epoch, row in enumerate(history, start=1)
        )
    )
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != V23A_VARIANT
        or summary.get("architecture") != "official_ncps_cfc_external_state_adapter"
        or int(summary.get("seed", -1)) != TRAINING_SEED
        or int(summary.get("dataloader_seed", -1)) != TRAINING_SEED
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or summary.get("config_sha256") != sha256_file(config_path)
        or summary.get("config_identity_sha256") != v23a_config_identity_sha256(config)
        or str(summary.get("initial_model_state_sha256", ""))
        != str(source_summary.get("initial_model_state_sha256", ""))
        or Path(str(summary.get("best_checkpoint", ""))).resolve()
        != checkpoint_path.resolve()
        or Path(str(summary.get("sample_detail_information", ""))).resolve()
        != detail_path.resolve()
        or Path(str(summary.get("last_checkpoint", ""))).resolve()
        != last_checkpoint_path.resolve()
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or summary.get("external_state_adapter") != config.adapter.as_dict()
        or summary.get("full_parameter_training") is not True
        or summary.get("all_model_parameters_trainable") is not True
        or summary.get("unknown_is_model_class") is not False
        or readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != "RADAR-LN3-V23a-known-gate-official-cfc-adapter-readiness-v1"
        or readiness.get("variant") != V23A_VARIANT
        or readiness.get("config_sha256") != sha256_file(config_path)
        or readiness.get("config_identity_sha256")
        != v23a_config_identity_sha256(config)
        or int(readiness.get("selected_seed", -1)) != TRAINING_SEED
        or int(readiness.get("train_parent_count", -1)) != 986
        or int(readiness.get("validation_parent_count", -1)) != 232
        or int(readiness.get("validation_clutter_parent_support", -1)) != 9
        or readiness.get("validation_used_for_readiness_gate") is not False
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False
        or not isinstance(readiness.get("checks"), Mapping)
        or not readiness["checks"]
        or any(value is not True for value in readiness["checks"].values())
        or not history_is_paired
    ):
        raise ValueError("V23j V23a fixed-retention control is incomplete")
    if source_summary_path.resolve() == summary_path.resolve():
        raise AssertionError("V23j V23a source unexpectedly aliases pure V23")
    return config_path, config, summary_path, summary, checkpoint_path


def _expected_selector(control: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": V23E_SELECTOR_SCHEMA,
        "name": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
        "source_v23a_legacy_reference_epoch": int(control["legacy_best_epoch"]),
    }


def load_config(root: Path) -> tuple[V23JConfig, KnownGateConfig, Path]:
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
        "prefix_ramp",
        "train",
    }:
        raise ValueError("V23j config top-level fields differ")
    if (
        payload.get("schema_version") != V23J_CONFIG_SCHEMA
        or payload.get("variant") != V23J_VARIANT
        or payload.get("config_name") != path.stem.removesuffix("-config")
    ):
        raise ValueError("V23j config identity differs")

    source = dict(payload["source"])
    if set(source) != {
        "config_path",
        "config_sha256",
        "config_identity_sha256",
        "v23a_control",
    }:
        raise ValueError("V23j source fields differ")
    source_path_value = str(source["config_path"])
    if source_path_value.replace("\\", "/") != str(SOURCE_CONFIG_RELATIVE).replace("\\", "/"):
        raise ValueError("V23j source must be frozen V21 seed43 baseline")
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
        or source_config.train.prefix_auxiliary_weight != V23J_PREFIX_RAMP_START_WEIGHT
        or source_config.train.class_weight_normalization != "batch_weight_sum"
    ):
        raise ValueError("V23j frozen V21 source differs")

    direct_control = dict(source["v23a_control"])
    if direct_control != source_v23a_control(root):
        raise ValueError("V23j V23a control binding is stale")

    adapter_payload = dict(payload["adapter"])
    expected_adapter = load_adapter_config(root, V23A_VARIANT)[0].adapter
    if adapter_payload != expected_adapter.__dict__:
        raise ValueError("V23j adapter must exactly reuse V23a fixed retention")

    selector = dict(payload["selector"])
    if selector != _expected_selector(direct_control):
        raise ValueError("V23j selector must exactly use the frozen V23e rule")

    ramp_payload = dict(payload["prefix_ramp"])
    if set(ramp_payload) != {
        "kind",
        "start_weight",
        "end_weight",
        "start_epoch",
        "end_epoch",
    }:
        raise ValueError("V23j prefix-ramp fields differ")
    ramp = V23JPrefixRamp(
        kind=str(ramp_payload["kind"]),
        start_weight=float(ramp_payload["start_weight"]),
        end_weight=float(ramp_payload["end_weight"]),
        start_epoch=int(ramp_payload["start_epoch"]),
        end_epoch=int(ramp_payload["end_epoch"]),
    )
    if not ramp_matches_protocol(ramp):
        raise ValueError("V23j prefix-ramp differs from the pre-registered schedule")

    train_payload = dict(payload["train"])
    expected_fields = set(V23JTrainConfig.__dataclass_fields__)
    if set(train_payload) != expected_fields:
        raise ValueError("V23j train fields differ")
    train = V23JTrainConfig(
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
        or train.prefix_auxiliary_min_events != source_train.prefix_auxiliary_min_events
        or train.checkpoint_path != f"{artifact}/best_model.pt"
        or train.last_checkpoint_path != f"{artifact}/last_model.pt"
        or train.formal_artifact_dir != artifact
        or train.epoch_checkpoint_dir != f"{artifact}/epochs"
        or train.checkpoint_schema != V23J_CHECKPOINT_SCHEMA
        or not train.record_parent_order_hash
        or not train.save_every_epoch
        or train.cuda_deterministic_algorithms != source_train.cuda_deterministic_algorithms
        or not train.full_parameter_training
    ):
        raise ValueError("V23j protocol differs beyond its pre-registered prefix ramp")
    source_v23_training(root)
    source_runtime_report(root)
    return (
        V23JConfig(
            schema_version=V23J_CONFIG_SCHEMA,
            config_name=str(payload["config_name"]),
            variant=V23J_VARIANT,
            source_config_path=source_path_value,
            source_config_sha256=str(source["config_sha256"]).lower(),
            source_config_identity_sha256=str(source["config_identity_sha256"]).lower(),
            source_v23a=direct_control,
            adapter=expected_adapter,
            selector=selector,
            prefix_ramp=ramp,
            train=train,
        ),
        source_config,
        source_path,
    )


__all__ = [
    "V23J_CHECKPOINT_SCHEMA",
    "V23J_CONFIG_SCHEMA",
    "V23J_CONTRACT_SCHEMA",
    "V23J_PREFIX_RAMP_END_EPOCH",
    "V23J_PREFIX_RAMP_END_WEIGHT",
    "V23J_PREFIX_RAMP_KIND",
    "V23J_PREFIX_RAMP_START_EPOCH",
    "V23J_PREFIX_RAMP_START_WEIGHT",
    "V23J_READINESS_SCHEMA",
    "V23J_REPORT_SCHEMA",
    "V23J_VARIANT",
    "V23JConfig",
    "V23JPrefixRamp",
    "V23JTrainConfig",
    "artifact_relative",
    "checkpoint_prefix_ramp_metadata",
    "config_identity_sha256",
    "config_relative",
    "contract_relative",
    "load_config",
    "prefix_auxiliary_weight_for_epoch",
    "ramp_matches_protocol",
    "readiness_relative",
    "report_relative",
    "source_v23a_control",
    "source_v23a_training",
    "validate_checkpoint_prefix_ramp_metadata",
]
