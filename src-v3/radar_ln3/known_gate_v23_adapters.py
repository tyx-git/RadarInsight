"""Frozen contracts shared by V23a/V23b/V23c official-CfC adapter experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .known_gate import KnownGateConfig, known_gate_config_identity_sha256, load_known_gate_config
from .known_gate_v23 import (
    FIXED_EPOCHS,
    RUNTIME_COMPATIBILITY_REPORT_RELATIVE,
    RUNTIME_COMPATIBILITY_REPORT_SCHEMA,
    RUNTIME_COMPATIBILITY_REQUIRED_CHECKS,
    SOURCE_CONFIG_RELATIVE,
    TRAINING_SEED,
    sha256_file,
)
from .official_cfc import official_cfc_runtime_binding
from .official_cfc_state_adapters import (
    ADAPTER_POSITIVE_DT_FIXED_RETENTION,
    ADAPTER_RECURRENT_LAYER_NORM,
    ADAPTER_ZERO_DELTA_BOUNDED_OBSERVATION,
    ExternalStateAdapterSpec,
)


V23_ADAPTER_VARIANTS = (
    "official_cfc_positive_dt_fixed_retention",
    "official_cfc_recurrent_layer_norm",
    "official_cfc_zero_delta_bounded_observation",
)

V23_ARTIFACT_RELATIVE = Path(
    "artifacts/RADAR-LN3-V23-cq08-known-gate-official-cfc-baseline-seed43-formal"
)
V23_REPORT_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V23-known-gate-official-cfc-baseline-validation.json"
)

VARIANT_TO_SUFFIX = {
    "official_cfc_positive_dt_fixed_retention": "V23a",
    "official_cfc_recurrent_layer_norm": "V23b",
    "official_cfc_zero_delta_bounded_observation": "V23c",
}

VARIANT_TO_ADAPTER = {
    "official_cfc_positive_dt_fixed_retention": ExternalStateAdapterSpec(
        kind=ADAPTER_POSITIVE_DT_FIXED_RETENTION,
        hidden_dim=48,
        fixed_tau_seconds=15.025,
    ),
    "official_cfc_recurrent_layer_norm": ExternalStateAdapterSpec(
        kind=ADAPTER_RECURRENT_LAYER_NORM,
        hidden_dim=48,
        layer_norm_eps=1.0e-5,
    ),
    "official_cfc_zero_delta_bounded_observation": ExternalStateAdapterSpec(
        kind=ADAPTER_ZERO_DELTA_BOUNDED_OBSERVATION,
        hidden_dim=48,
        zero_delta_gain=0.25,
    ),
}


def _relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V23 adapter config is missing {key}")
    return mapping[key]


def canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def state_dict_sha256(state: Mapping[str, Any], *, exclude_prefixes: tuple[str, ...] = ()) -> str:
    """Match V23's frozen training-state hash byte for byte.

    The V23 summary was written with ``train.known_gate_v21._state_dict_sha256``.
    Keep the same dtype/shape/byte encoding locally so V23a/V23c can prove full
    initial-state equality and V23b can prove equality after excluding only its
    newly introduced ``state_adapter.`` parameters.
    """

    digest = hashlib.sha256()
    for name in sorted(str(key) for key in state):
        if any(name.startswith(prefix) for prefix in exclude_prefixes):
            continue
        value = state[name]
        if not hasattr(value, "detach"):
            raise TypeError(f"state value is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def config_relative(variant: str) -> Path:
    suffix = VARIANT_TO_SUFFIX[variant]
    return Path(f"config/RADAR-LN3/RADAR-LN3-{suffix}-cq08-known-gate-official-cfc-{variant.removeprefix('official_cfc_').replace('_', '-')}-seed43-config.json")


def artifact_relative(variant: str) -> Path:
    suffix = VARIANT_TO_SUFFIX[variant]
    return Path(f"artifacts/RADAR-LN3-{suffix}-cq08-known-gate-official-cfc-{variant.removeprefix('official_cfc_').replace('_', '-')}-seed43-formal")


def contract_relative(variant: str) -> Path:
    suffix = VARIANT_TO_SUFFIX[variant]
    return Path(f"data/processed/RADAR-LN3-{suffix}-cq08-known-gate-official-cfc-{variant.removeprefix('official_cfc_').replace('_', '-')}-contract.json")


def readiness_relative(variant: str) -> Path:
    suffix = VARIANT_TO_SUFFIX[variant]
    return Path(f"src-v3/processed/report/RADAR-LN3-{suffix}-known-gate-official-cfc-training-readiness.json")


def report_relative(variant: str) -> Path:
    suffix = VARIANT_TO_SUFFIX[variant]
    return Path(f"src-v3/processed/report/RADAR-LN3-{suffix}-known-gate-official-cfc-paired-validation.json")


def config_schema(variant: str) -> str:
    return f"RADAR-LN3-{VARIANT_TO_SUFFIX[variant]}-known-gate-official-cfc-adapter-config-v1"


def contract_schema(variant: str) -> str:
    return f"RADAR-LN3-{VARIANT_TO_SUFFIX[variant]}-known-gate-official-cfc-adapter-contract-v1"


def readiness_schema(variant: str) -> str:
    return f"RADAR-LN3-{VARIANT_TO_SUFFIX[variant]}-known-gate-official-cfc-adapter-readiness-v1"


def checkpoint_schema(variant: str) -> str:
    return f"RADAR-LN3-{VARIANT_TO_SUFFIX[variant]}-known-gate-official-cfc-adapter-seed43-checkpoint-v1"


def report_schema(variant: str) -> str:
    return f"RADAR-LN3-{VARIANT_TO_SUFFIX[variant]}-known-gate-official-cfc-adapter-paired-validation-v1"


@dataclass(frozen=True)
class AdapterTrainConfig:
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
class AdapterConfig:
    schema_version: str
    config_name: str
    variant: str
    source_config_path: str
    source_config_sha256: str
    source_config_identity_sha256: str
    source_v23_report_path: str
    source_v23_report_sha256: str
    adapter: ExternalStateAdapterSpec
    train: AdapterTrainConfig


def config_identity_sha256(config: AdapterConfig) -> str:
    return canonical_sha256({
        "schema_version": config.schema_version,
        "config_name": config.config_name,
        "variant": config.variant,
        "source_config_path": config.source_config_path,
        "source_config_sha256": config.source_config_sha256,
        "source_config_identity_sha256": config.source_config_identity_sha256,
        "source_v23_report_path": config.source_v23_report_path,
        "source_v23_report_sha256": config.source_v23_report_sha256,
        "adapter": config.adapter.__dict__,
        "train": config.train.__dict__,
    })


def _source_v23_report(root: Path) -> tuple[str, str]:
    path = root / V23_REPORT_RELATIVE
    if not path.is_file():
        raise FileNotFoundError(f"V23 frozen development report is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    clutter = list(dict(payload.get("development_metrics_core") or {}).get("parent_final", {}).get("per_class") or [])
    if (
        payload.get("status") != "completed_development_baseline_only"
        or dict(payload.get("performance") or {}).get("status") != "failed"
        or int(dict(payload.get("development_metrics_core") or {}).get("parent_final", {}).get("support", -1)) != 232
        or len(clutter) != 2
        or int(clutter[1].get("support", -1)) != 9
    ):
        raise ValueError("V23 frozen development report is not the expected negative baseline")
    return _relative(root, path), sha256_file(path)


def source_v23_training(root: Path) -> tuple[Path, dict[str, Any], Path, str]:
    """Load the sealed V23 source summary and checkpoint binding."""

    artifact = root / V23_ARTIFACT_RELATIVE
    summary_path = artifact / "training-summary.json"
    checkpoint_path = artifact / "best_model.pt"
    if not summary_path.is_file() or not checkpoint_path.is_file():
        raise FileNotFoundError("V23 frozen source summary or checkpoint is missing")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    history = list(summary.get("history") or ())
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != "official_cfc_baseline"
        or summary.get("architecture") != "official_ncps_cfc"
        or int(summary.get("seed", -1)) != TRAINING_SEED
        or len(history) != FIXED_EPOCHS
        or int(summary.get("train_parent_count", -1)) != 986
        or int(summary.get("validation_parent_count", -1)) != 232
        or int(summary.get("validation_clutter_support", -1)) != 9
        or summary.get("model_state_changed_from_initialization") is not True
        or not _is_sha256(summary.get("initial_model_state_sha256"))
        or not _is_sha256(summary.get("best_model_state_sha256"))
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V23 frozen source summary is invalid")
    source_v23_parent_order_hashes(summary)
    return summary_path, summary, checkpoint_path, sha256_file(checkpoint_path)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def source_v23_parent_order_hashes(summary: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the sealed 20-epoch V23 parent order sequence.

    The adapter variants reuse V23's independent DataLoader generator and must
    consume the same parent permutation in every epoch.  This is deliberately
    checked before training starts and after each completed treatment epoch.
    """

    history = list(summary.get("history") or ())
    if len(history) != FIXED_EPOCHS:
        raise ValueError("V23 source history does not contain the fixed 20 epochs")
    values: list[str] = []
    for expected_epoch, row in enumerate(history, start=1):
        item = dict(row)
        value = item.get("parent_order_sha256")
        if (
            int(item.get("epoch", -1)) != expected_epoch
            or not _is_sha256(value)
            or int(item.get("parent_order_count", -1)) != 986
            or int(item.get("parent_order_unique_count", -1)) != 986
        ):
            raise ValueError("V23 source history has an invalid parent-order contract")
        values.append(str(value))
    return tuple(values)


def _adapter_from_payload(variant: str, payload: Mapping[str, Any]) -> ExternalStateAdapterSpec:
    expected = VARIANT_TO_ADAPTER[variant]
    if set(payload) != {"kind", "hidden_dim", "fixed_tau_seconds", "zero_delta_gain", "layer_norm_eps"}:
        raise ValueError("V23 adapter config adapter fields differ")
    actual = ExternalStateAdapterSpec(
        kind=str(_require(payload, "kind")),
        hidden_dim=int(_require(payload, "hidden_dim")),
        fixed_tau_seconds=(None if payload["fixed_tau_seconds"] is None else float(payload["fixed_tau_seconds"])),
        zero_delta_gain=(None if payload["zero_delta_gain"] is None else float(payload["zero_delta_gain"])),
        layer_norm_eps=(None if payload["layer_norm_eps"] is None else float(payload["layer_norm_eps"])),
    )
    if actual != expected:
        raise ValueError("V23 adapter config differs from its pre-registered single factor")
    return actual


def load_adapter_config(root: Path, variant: str) -> tuple[AdapterConfig, KnownGateConfig, Path]:
    if variant not in V23_ADAPTER_VARIANTS:
        raise ValueError(f"unknown V23 adapter variant: {variant}")
    root = root.resolve()
    path = (root / config_relative(variant)).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "config_name", "variant", "source", "adapter", "train"}:
        raise ValueError("V23 adapter config top-level fields differ")
    if payload.get("schema_version") != config_schema(variant) or payload.get("variant") != variant:
        raise ValueError("V23 adapter config identity is invalid")
    if payload.get("config_name") != path.stem.removesuffix("-config"):
        raise ValueError("V23 adapter config name differs from filename")
    source = dict(_require(payload, "source"))
    if set(source) != {"config_path", "config_sha256", "config_identity_sha256", "v23_report_path", "v23_report_sha256"}:
        raise ValueError("V23 adapter source binding differs")
    source_path_value = str(_require(source, "config_path"))
    if source_path_value.replace("\\", "/") != str(SOURCE_CONFIG_RELATIVE).replace("\\", "/"):
        raise ValueError("V23 adapter source must be frozen V21 seed43 baseline")
    source_path = _resolve(root, source_path_value)
    source_sha = str(_require(source, "config_sha256")).lower()
    source_identity = str(_require(source, "config_identity_sha256")).lower()
    if not source_path.is_file() or sha256_file(source_path) != source_sha:
        raise ValueError("V23 adapter source config hash is stale")
    source_config = load_known_gate_config(source_path)
    if known_gate_config_identity_sha256(source_config) != source_identity:
        raise ValueError("V23 adapter source config identity is stale")
    if (
        source_config.variant != "baseline"
        or source_config.model.evidence_dim != 0
        or source_config.model.rd_temporal_residual_dim != 0
        or source_config.train.seed != TRAINING_SEED
        or source_config.train.dataloader_seed != TRAINING_SEED
        or source_config.train.epochs != FIXED_EPOCHS
        or source_config.train.class_weight_normalization != "batch_weight_sum"
    ):
        raise ValueError("V23 adapter source protocol is not frozen V21 seed43")
    report_path, report_sha = _source_v23_report(root)
    if str(_require(source, "v23_report_path")).replace("\\", "/") != report_path or str(_require(source, "v23_report_sha256")).lower() != report_sha:
        raise ValueError("V23 adapter source V23 report binding is stale")
    adapter = _adapter_from_payload(variant, dict(_require(payload, "adapter")))
    train_payload = dict(_require(payload, "train"))
    expected_train_fields = set(AdapterTrainConfig.__dataclass_fields__)
    if set(train_payload) != expected_train_fields:
        raise ValueError("V23 adapter train fields differ")
    train = AdapterTrainConfig(
        batch_size=int(train_payload["batch_size"]), learning_rate=float(train_payload["learning_rate"]),
        seed=int(train_payload["seed"]), dataloader_seed=int(train_payload["dataloader_seed"]),
        device=str(train_payload["device"]), epochs=int(train_payload["epochs"]),
        early_stopping_patience=int(train_payload["early_stopping_patience"]),
        gradient_clip_norm=float(train_payload["gradient_clip_norm"]),
        checkpoint_selection_metric=str(train_payload["checkpoint_selection_metric"]),
        task_min_precision_recall_target=float(train_payload["task_min_precision_recall_target"]),
        class_weights=tuple(float(value) for value in train_payload["class_weights"]),
        loss_mode=str(train_payload["loss_mode"]),
        class_weight_normalization=str(train_payload["class_weight_normalization"]),
        prefix_auxiliary_weight=float(train_payload["prefix_auxiliary_weight"]),
        prefix_auxiliary_min_events=int(train_payload["prefix_auxiliary_min_events"]),
        checkpoint_path=str(train_payload["checkpoint_path"]), last_checkpoint_path=str(train_payload["last_checkpoint_path"]),
        formal_artifact_dir=str(train_payload["formal_artifact_dir"]), epoch_checkpoint_dir=str(train_payload["epoch_checkpoint_dir"]),
        checkpoint_schema=str(train_payload["checkpoint_schema"]),
        record_parent_order_hash=bool(train_payload["record_parent_order_hash"]), save_every_epoch=bool(train_payload["save_every_epoch"]),
        cuda_deterministic_algorithms=bool(train_payload["cuda_deterministic_algorithms"]),
        full_parameter_training=bool(train_payload["full_parameter_training"]),
    )
    expected_artifact = str(artifact_relative(variant)).replace("\\", "/")
    source_train = source_config.train
    if (
        len(train.class_weights) != 2 or train.batch_size != source_train.batch_size
        or train.learning_rate != source_train.learning_rate or train.seed != TRAINING_SEED
        or train.dataloader_seed != TRAINING_SEED or train.device != source_train.device
        or train.epochs != FIXED_EPOCHS or train.early_stopping_patience != 0
        or train.gradient_clip_norm != source_train.gradient_clip_norm
        or train.checkpoint_selection_metric != source_train.checkpoint_selection_metric
        or train.task_min_precision_recall_target != source_train.task_min_precision_recall_target
        or train.class_weights != source_train.class_weights or train.loss_mode != source_train.loss_mode
        or train.class_weight_normalization != source_train.class_weight_normalization
        or train.prefix_auxiliary_weight != source_train.prefix_auxiliary_weight
        or train.prefix_auxiliary_min_events != source_train.prefix_auxiliary_min_events
        or train.checkpoint_path != f"{expected_artifact}/best_model.pt"
        or train.last_checkpoint_path != f"{expected_artifact}/last_model.pt"
        or train.formal_artifact_dir != expected_artifact
        or train.epoch_checkpoint_dir != f"{expected_artifact}/epochs"
        or train.checkpoint_schema != checkpoint_schema(variant)
        or not train.record_parent_order_hash or not train.save_every_epoch
        or train.cuda_deterministic_algorithms != source_train.cuda_deterministic_algorithms
        or not train.full_parameter_training
    ):
        raise ValueError("V23 adapter training protocol differs from frozen V23 source")
    runtime = official_cfc_runtime_binding()
    if not runtime:
        raise RuntimeError("official CfC runtime binding failed")
    return AdapterConfig(
        schema_version=config_schema(variant), config_name=str(payload["config_name"]), variant=variant,
        source_config_path=source_path_value, source_config_sha256=source_sha,
        source_config_identity_sha256=source_identity, source_v23_report_path=report_path,
        source_v23_report_sha256=report_sha, adapter=adapter, train=train,
    ), source_config, source_path


def source_runtime_report(root: Path) -> tuple[Path, dict[str, Any], str]:
    path = (root / RUNTIME_COMPATIBILITY_REPORT_RELATIVE).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    checks = dict(payload.get("checks") or {})
    if (
        payload.get("status") != "passed" or payload.get("schema_version") != RUNTIME_COMPATIBILITY_REPORT_SCHEMA
        or set(checks) != set(RUNTIME_COMPATIBILITY_REQUIRED_CHECKS) or any(value is not True for value in checks.values())
        or any(int(payload.get(field, -1)) != 0 for field in ("dataset_access_count", "checkpoint_access_count", "validation_access_count", "test_access_count"))
    ):
        raise ValueError("V23 official runtime compatibility report is invalid")
    return path, payload, sha256_file(path)


__all__ = [
    "AdapterConfig", "AdapterTrainConfig", "V23_ADAPTER_VARIANTS", "VARIANT_TO_ADAPTER", "VARIANT_TO_SUFFIX",
    "V23_ARTIFACT_RELATIVE", "V23_REPORT_RELATIVE", "artifact_relative", "checkpoint_schema", "config_identity_sha256", "config_relative", "config_schema",
    "contract_relative", "contract_schema", "load_adapter_config", "readiness_relative", "readiness_schema",
    "report_relative", "report_schema", "source_runtime_report", "source_v23_training", "state_dict_sha256",
    "source_v23_parent_order_hashes",
]
