"""Frozen contract for V23l's two-backbone temporal-mechanism comparison.

V23l is intentionally an *upper-bound architecture comparison*, not a claim
that each component is a new independent single-factor improvement.  The two
arms consume the identical V21e causal-course feature cache and KnownGate
training protocol:

* self-developed arm: ``CfCLiquidCausalEncoder`` with density-weighted same-
  second gain ``min(0.25*k, 0.75)`` (the completed V23g state path);
* official arm: unmodified ``ncps.torch.CfC`` followed by a reason-disjoint
  external adapter: fixed 15.025-second retention on positive dt and the same
  density schedule on zero dt.

The official cell itself is never reimplemented.  Optimizer, learning rate,
class weights, loss, checkpoint selector, data split, feature cache,
normalization, seed, parent order and epoch count remain frozen.  This makes
the resulting difference attributable to the recurrent backbone plus the
necessary representation of the same temporal semantics, not to optimisation
or feature changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .known_gate import KnownGateConfig, known_gate_config_identity_sha256, load_known_gate_config
from .known_gate_v23 import FIXED_EPOCHS, TRAINING_SEED, sha256_file
from .known_gate_v23e_selection import V23E_SELECTOR_SCHEMA
from .known_gate_v23g import (
    V23G_VARIANT,
    artifact_relative as v23g_artifact_relative,
    config_identity_sha256 as v23g_config_identity_sha256,
    config_relative as v23g_config_relative,
    load_config as load_v23g_config,
    readiness_relative as v23g_readiness_relative,
    source_control as source_v23g_control,
)
from .official_cfc_state_adapters import (
    ADAPTER_POSITIVE_DT_FIXED_RETENTION_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY,
    ExternalStateAdapterSpec,
)


V23L_VARIANT = "official_cfc_fixed_retention_density_weighted_course_prefix"
V23L_FIXED_TAU_SECONDS = 15.025
V23L_ZERO_DELTA_GAIN = 0.25
V23L_CONFIG_SCHEMA = "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-config-v1"
V23L_CONTRACT_SCHEMA = "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-contract-v1"
V23L_READINESS_SCHEMA = "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-readiness-v1"
V23L_CHECKPOINT_SCHEMA = "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-seed43-checkpoint-v1"
V23L_REPORT_SCHEMA = "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-report-v1"
V23L_SOURCE_CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V21e-cq08-liquid-known-gate-causal-course-consistency-seed43-config.json"
)


def config_relative() -> Path:
    return Path(
        "config/RADAR-LN3/"
        "RADAR-LN3-V23l-cq08-known-gate-official-cfc-fixed-retention-"
        "density-weighted-course-prefix-seed43-config.json"
    )


def artifact_relative() -> Path:
    return Path(
        "artifacts/"
        "RADAR-LN3-V23l-cq08-known-gate-official-cfc-fixed-retention-"
        "density-weighted-course-prefix-seed43-formal"
    )


def contract_relative() -> Path:
    return Path(
        "data/processed/"
        "RADAR-LN3-V23l-cq08-known-gate-official-cfc-fixed-retention-"
        "density-weighted-course-prefix-contract.json"
    )


def readiness_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-"
        "training-readiness.json"
    )


def report_relative() -> Path:
    return Path(
        "src-v3/processed/report/"
        "RADAR-LN3-V23l-known-gate-official-cfc-full-temporal-adapter-"
        "paired-validation.json"
    )


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def full_temporal_adapter_spec() -> ExternalStateAdapterSpec:
    return ExternalStateAdapterSpec(
        kind=(
            ADAPTER_POSITIVE_DT_FIXED_RETENTION_ZERO_DELTA_DENSITY_WEIGHTED_MULTIPLICITY
        ),
        hidden_dim=48,
        fixed_tau_seconds=V23L_FIXED_TAU_SECONDS,
        zero_delta_gain=V23L_ZERO_DELTA_GAIN,
    )


@dataclass(frozen=True)
class V23LTrainConfig:
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
class V23LConfig:
    schema_version: str
    config_name: str
    variant: str
    source_config_path: str
    source_config_sha256: str
    source_config_identity_sha256: str
    source_v23g: dict[str, Any]
    adapter: ExternalStateAdapterSpec
    selector: dict[str, Any]
    train: V23LTrainConfig


def config_identity_sha256(config: V23LConfig) -> str:
    return canonical_sha256(
        {
            "schema_version": config.schema_version,
            "config_name": config.config_name,
            "variant": config.variant,
            "source_config_path": config.source_config_path,
            "source_config_sha256": config.source_config_sha256,
            "source_config_identity_sha256": config.source_config_identity_sha256,
            "source_v23g": config.source_v23g,
            "adapter": config.adapter.as_dict(),
            "selector": config.selector,
            "train": config.train.__dict__,
        }
    )


def source_v23g_density(root: Path) -> dict[str, Any]:
    """Bind the official arm to the completed self-developed V23g arm."""

    root = root.resolve()
    control = source_v23g_control(root)
    config_path = (root / v23g_config_relative()).resolve()
    artifact = (root / v23g_artifact_relative()).resolve()
    readiness_path = (root / v23g_readiness_relative()).resolve()
    summary_path = artifact / "training-summary.json"
    epoch20_checkpoint_path = artifact / "epochs" / "epoch-020.pt"
    # V23g's legacy selector produced a diagnostic best checkpoint that fails
    # the later V23e feasibility rule.  V23l compares the complete 20-epoch
    # trajectory and its fixed epoch-20 state, so bind the comparator to the
    # invariant last checkpoint rather than treating that legacy best as an
    # admissible selected model.
    checkpoint_path = artifact / "last_model.pt"
    detail_path = artifact / "sample_detail_information.json"
    required = (
        config_path,
        readiness_path,
        summary_path,
        checkpoint_path,
        epoch20_checkpoint_path,
        detail_path,
    )
    if not all(path.is_file() for path in required):
        missing = [str(path) for path in required if not path.is_file()]
        raise FileNotFoundError("V23l requires completed V23g density arm: " + ", ".join(missing))
    summary = dict(json.loads(summary_path.read_text(encoding="utf-8")))
    readiness = dict(json.loads(readiness_path.read_text(encoding="utf-8")))
    detail = dict(json.loads(detail_path.read_text(encoding="utf-8")))
    history = list(summary.get("history") or ())
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != V23G_VARIANT
        or summary.get("architecture")
        != "self_developed_cfc_liquid_causal_encoder_density_weighted"
        or int(summary.get("seed", -1)) != TRAINING_SEED
        or len(history) != FIXED_EPOCHS
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or int(summary.get("best_epoch", -1)) not in range(1, FIXED_EPOCHS + 1)
        or detail.get("variant") != V23G_VARIANT
        or int(detail.get("epoch", -1)) != int(summary.get("best_epoch", -2))
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
        or readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != "RADAR-LN3-V23g-known-gate-density-weighted-readiness-v1"
        or readiness.get("variant") != V23G_VARIANT
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V23l V23g density arm is incomplete or outside frozen scope")
    parent_orders: list[str] = []
    for epoch, row in enumerate(history, start=1):
        record = dict(row)
        order = str(record.get("parent_order_sha256", ""))
        if (
            int(record.get("epoch", -1)) != epoch
            or len(order) != 64
            or int(record.get("parent_order_count", -1)) != 986
            or int(record.get("parent_order_unique_count", -1)) != 986
        ):
            raise ValueError("V23l V23g density parent-order contract is invalid")
        parent_orders.append(order)
    return {
        "variant": V23G_VARIANT,
        "config_path": _relative(root, config_path),
        "config_sha256": sha256_file(config_path),
        "config_identity_sha256": v23g_config_identity_sha256(
            load_v23g_config(root)[0]
        ),
        "readiness_path": _relative(root, readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "summary_path": _relative(root, summary_path),
        "summary_sha256": sha256_file(summary_path),
        "last_checkpoint_path": _relative(root, checkpoint_path),
        "last_checkpoint_sha256": sha256_file(checkpoint_path),
        "epoch20_checkpoint_path": _relative(root, epoch20_checkpoint_path),
        "epoch20_checkpoint_sha256": sha256_file(epoch20_checkpoint_path),
        "legacy_best_detail_path": _relative(root, detail_path),
        "legacy_best_detail_sha256": sha256_file(detail_path),
        "legacy_best_epoch": int(summary["best_epoch"]),
        "legacy_best_checkpoint_is_not_v23e_feasible": True,
        "parent_order_sha256_by_epoch": parent_orders,
        "dynamics": {
            "positive_delta": "self-developed learned tau/candidate/gate continuous-time update",
            "zero_delta": "alpha_k=min(0.25*k,0.75)",
            "causal_same_second_ordinal": "first/positive set k=1; zero_delta increments k",
        },
        "course_feature_cache": "V21e frozen causal-course consistency evidence_dim=1",
        "source_v21e_control": control,
    }


def _assert_source_course_protocol(source: KnownGateConfig) -> None:
    if (
        source.variant != "v21e_course_prefix"
        or source.model.evidence_dim != 1
        or tuple(source.feature_names) != ("causal_prefix_course_circular_incoherence",)
        or source.model.rd_temporal_residual_dim != 0
        or source.model.liquid_invalid_dt_update_mode != "bounded_observation"
        or abs(source.model.liquid_invalid_dt_observation_gain - V23L_ZERO_DELTA_GAIN) > 1.0e-12
        or source.train.seed != TRAINING_SEED
        or source.train.dataloader_seed != TRAINING_SEED
        or source.train.epochs != FIXED_EPOCHS
        or source.train.class_weight_normalization != "batch_weight_sum"
    ):
        raise ValueError("V23l source must be the frozen V21e course-prefix protocol")


def load_config(root: Path) -> tuple[V23LConfig, KnownGateConfig, Path]:
    root = root.resolve()
    path = (root / config_relative()).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = dict(json.loads(path.read_text(encoding="utf-8")))
    expected_top_level = {
        "schema_version", "config_name", "variant", "source", "adapter", "selector", "train"
    }
    if set(payload) != expected_top_level:
        raise ValueError("V23l config top-level fields differ")
    if (
        payload.get("schema_version") != V23L_CONFIG_SCHEMA
        or payload.get("config_name") != path.stem.removesuffix("-config")
        or payload.get("variant") != V23L_VARIANT
    ):
        raise ValueError("V23l config identity differs")
    source = dict(payload["source"])
    if set(source) != {
        "config_path", "config_sha256", "config_identity_sha256", "v23g_density"
    }:
        raise ValueError("V23l source fields differ")
    source_path_value = str(source["config_path"])
    if source_path_value.replace("\\", "/") != str(V23L_SOURCE_CONFIG_RELATIVE).replace("\\", "/"):
        raise ValueError("V23l source must be frozen V21e course-prefix config")
    source_path = _resolve(root, source_path_value)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_config = load_known_gate_config(source_path)
    _assert_source_course_protocol(source_config)
    if (
        sha256_file(source_path) != str(source["config_sha256"]).lower()
        or known_gate_config_identity_sha256(source_config)
        != str(source["config_identity_sha256"]).lower()
    ):
        raise ValueError("V23l source course config binding is stale")
    density = dict(source["v23g_density"])
    if density != source_v23g_density(root):
        raise ValueError("V23l V23g density control binding is stale")
    adapter_payload = dict(payload["adapter"])
    expected_adapter = full_temporal_adapter_spec()
    if adapter_payload != expected_adapter.as_dict():
        raise ValueError("V23l adapter differs from fixed-retention plus density composition")
    selector = dict(payload["selector"])
    expected_selector = {
        "schema_version": V23E_SELECTOR_SCHEMA,
        "name": "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
        "source_v23a_legacy_reference_epoch": 14,
    }
    if selector != expected_selector:
        raise ValueError("V23l selector must exactly use the frozen V23e rule")
    train_payload = dict(payload["train"])
    if set(train_payload) != set(V23LTrainConfig.__dataclass_fields__):
        raise ValueError("V23l train fields differ")
    train = V23LTrainConfig(
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
        or train.checkpoint_schema != V23L_CHECKPOINT_SCHEMA
        or not train.record_parent_order_hash
        or not train.save_every_epoch
        or train.cuda_deterministic_algorithms != source_train.cuda_deterministic_algorithms
        or not train.full_parameter_training
    ):
        raise ValueError("V23l protocol differs beyond its official temporal state path")
    return (
        V23LConfig(
            schema_version=V23L_CONFIG_SCHEMA,
            config_name=str(payload["config_name"]),
            variant=V23L_VARIANT,
            source_config_path=source_path_value,
            source_config_sha256=str(source["config_sha256"]).lower(),
            source_config_identity_sha256=str(source["config_identity_sha256"]).lower(),
            source_v23g=density,
            adapter=expected_adapter,
            selector=selector,
            train=train,
        ),
        source_config,
        source_path,
    )


__all__ = ["V23L_CHECKPOINT_SCHEMA", "V23L_CONFIG_SCHEMA", "V23L_CONTRACT_SCHEMA", "V23L_FIXED_TAU_SECONDS", "V23L_READINESS_SCHEMA", "V23L_REPORT_SCHEMA", "V23L_SOURCE_CONFIG_RELATIVE", "V23L_VARIANT", "V23L_ZERO_DELTA_GAIN", "V23LConfig", "V23LTrainConfig", "artifact_relative", "config_identity_sha256", "config_relative", "contract_relative", "full_temporal_adapter_spec", "load_config", "readiness_relative", "report_relative", "source_v23g_density"]