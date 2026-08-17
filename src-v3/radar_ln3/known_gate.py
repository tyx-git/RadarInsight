"""Independent V21 KnownGate model, data and evaluation contracts.

This is the canonical L1 path.  It deliberately does not import or rename the
legacy ``dbb_cu_gate``/``cu_specialist`` task specifications because those
names remain bound to historical checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import (
    CQ08PointSequenceDataset,
    DT_REASON_NAMES,
    PointSequence,
    collate_point_sequences,
)
from .known_gate_features import (
    BASELINE_VARIANT,
    COURSE_PREFIX_FEATURE_SCHEMA,
    COURSE_INTERFRAME_FEATURE_NAMES,
    D_RANGE_PREFIX_FEATURE_SCHEMA,
    FEATURE_NAMES_BY_VARIANT,
    INTERFRAME_SPECTRAL_FEATURE_SCHEMA,
    KNOWN_GATE_FEATURE_SCHEMA,
    V21A_INTERFRAME_VARIANT,
    V21A_VARIANT,
    V21D_PREFIX_VARIANT,
    V21E_COURSE_PREFIX_VARIANT,
    V21F_COURSE_INTERFRAME_VARIANT,
    load_known_gate_feature_cache,
)
from .model import CfCLiquidCausalEncoder, RDSpatialEncoder
from .open_set_architecture import KNOWN_GATE_TASK_NAME, KnownGateClass


V21_SCHEMA_VERSION = "RADAR-LN3-V21"
KNOWN_TARGET_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
KNOWN_CLUTTER_LABEL = "ClutterTarget"
UNKNOWN_LABEL = "UnknownTarget"
KNOWN_LABELS = (*KNOWN_TARGET_LABELS, KNOWN_CLUTTER_LABEL)
KNOWN_GATE_CLASS_NAMES = (
    KnownGateClass.KNOWN_TARGET_EVIDENCE.value,
    KnownGateClass.KNOWN_CLUTTER_EVIDENCE.value,
)
KNOWN_GATE_LABEL_TO_ID = {
    **{name: 0 for name in KNOWN_TARGET_LABELS},
    KNOWN_CLUTTER_LABEL: 1,
}
KNOWN_GATE_LOSS_MODE = "parent_final_task_ce_with_prefix_auxiliary"
KNOWN_GATE_SELECTION_METRIC = "parent_min_precision_recall"
KNOWN_GATE_CONTRACT_SCHEMA = "RADAR-LN3-V21-known-gate-data-contract-v1"
KNOWN_GATE_MANIFEST_SCHEMA = "RADAR-LN3-V21-known-gate-feature-cache-manifest-v1"
KNOWN_GATE_V21A_MANIFEST_SCHEMA = (
    "RADAR-LN3-V21a-known-gate-interframe-spectral-feature-cache-manifest-v1"
)
KNOWN_GATE_READINESS_SCHEMA = "RADAR-LN3-V21-known-gate-training-readiness-v1"
KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA = (
    "RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness-v1"
)
KNOWN_GATE_V21A_READINESS_SCHEMA = (
    "RADAR-LN3-V21a-known-gate-interframe-spectral-training-readiness-v1"
)
KNOWN_GATE_V21B_READINESS_SCHEMA = (
    "RADAR-LN3-V21b-known-gate-modal-consistency-clutter-drone-training-readiness-v2"
)
KNOWN_GATE_V21C_READINESS_SCHEMA = (
    "RADAR-LN3-V21c-known-gate-target-row-iso-clutter-drone-training-readiness-v1"
)
KNOWN_GATE_V21D_READINESS_SCHEMA = (
    "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-clutter-bird-training-readiness-v1"
)
KNOWN_GATE_V21D_MANIFEST_SCHEMA = (
    "RADAR-LN3-V21d-known-gate-causal-prefix-d-range-feature-cache-manifest-v1"
)
KNOWN_GATE_V21E_READINESS_SCHEMA = (
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-training-readiness-v1"
)
KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA = (
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-"
    "multiseed-training-readiness-v1"
)
KNOWN_GATE_V21E_MANIFEST_SCHEMA = (
    "RADAR-LN3-V21e-known-gate-causal-course-consistency-feature-cache-manifest-v1"
)
KNOWN_GATE_BASELINE_MULTISEED_SEEDS = (42, 43, 44)
KNOWN_GATE_V21E_MULTISEED_SEEDS = KNOWN_GATE_BASELINE_MULTISEED_SEEDS


def _require(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"V21 KnownGate configuration is missing {key}")
    return mapping[key]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class KnownGateDataConfig:
    index_path: str
    contract_path: str
    normalization_stats_path: str
    train_split_name: str
    validation_split_name: str
    test_split_name: str
    index_schema_version: str
    contract_schema_version: str
    expected_index_sha256: str
    expected_contract_sha256: str
    expected_stats_sha256: str
    max_sequence_length: int
    max_rd_frames_per_point: int
    rd_amplitude_center: float
    rd_amplitude_scale: float
    feature_cache_manifest_path: str | None
    expected_feature_cache_manifest_sha256: str | None
    train_feature_cache_path: str | None
    validation_feature_cache_path: str | None
    expected_train_feature_cache_sha256: str | None
    expected_validation_feature_cache_sha256: str | None
    feature_experiment_contract_path: str | None
    expected_feature_experiment_contract_sha256: str | None


@dataclass(frozen=True)
class KnownGateModelConfig:
    tr_value_dim: int
    tr_missing_dim: int
    rd_quality_dim: int
    hidden_dim: int
    rd_embedding_dim: int
    liquid_hidden_dim: int
    rd_height: int
    rd_width: int
    liquid_tau_min_seconds: float
    liquid_tau_max_seconds: float
    liquid_zero_dt_event_floor_seconds: float
    liquid_dt_max_seconds: float
    liquid_invalid_dt_update_mode: str
    liquid_invalid_dt_observation_gain: float
    liquid_state_normalization: str
    rd_frame_aggregation: str
    dt_reason_mode: str
    evidence_dim: int


@dataclass(frozen=True)
class KnownGateTrainConfig:
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
    checkpoint_path: str
    last_checkpoint_path: str
    formal_artifact_dir: str
    epoch_checkpoint_dir: str
    class_weights: tuple[float, float]
    loss_mode: str
    prefix_auxiliary_weight: float
    prefix_auxiliary_min_events: int
    checkpoint_schema: str
    record_parent_order_hash: bool
    save_every_epoch: bool
    cuda_deterministic_algorithms: bool
    full_parameter_training: bool
    paired_baseline_config_path: str | None


@dataclass(frozen=True)
class KnownGateConfig:
    schema_version: str
    config_name: str
    variant: str
    task_name: str
    class_names: tuple[str, str]
    label_to_id: Mapping[str, int]
    forbidden_training_labels: tuple[str, ...]
    feature_names: tuple[str, ...]
    data: KnownGateDataConfig
    model: KnownGateModelConfig
    train: KnownGateTrainConfig


_V21E_OOF_CONFIG_PATTERN = re.compile(
    r"RADAR-LN3-V21e-cq08-known-gate-oof-fold-(0[0-4])-"
    r"(control|treatment)-seed43"
)
_V21F_OOF_CONFIG_PATTERN = re.compile(
    r"RADAR-LN3-V21f-cq08-known-gate-oof-fold-(0[0-4])-"
    r"treatment-seed43"
)


def _validate_v21e_oof_support_config(
    config_path: Path,
    *,
    config_name: str,
    variant: str,
    data: KnownGateDataConfig,
    train: KnownGateTrainConfig,
) -> bool:
    """Recognize only the frozen V21e fold-support configuration namespace."""

    match = _V21E_OOF_CONFIG_PATTERN.fullmatch(config_name)
    if match is None:
        return False
    fold_token, arm = match.groups()
    expected_variant = (
        BASELINE_VARIANT if arm == "control" else V21E_COURSE_PREFIX_VARIANT
    )
    artifact = (
        "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof/"
        f"fold-{fold_token}/{arm}"
    )
    expected_pair = (
        None
        if arm == "control"
        else (
            "config/RADAR-LN3/"
            f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_token}-"
            "control-seed43-config.json"
        )
    )
    if (
        config_path.name != f"{config_name}-config.json"
        or config_path.parent.name != "RADAR-LN3"
        or config_path.parent.parent.name != "config"
        or variant != expected_variant
        or data.contract_path
        != (
            "data/processed/"
            f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_token}-"
            "data-contract.json"
        )
        or data.normalization_stats_path
        != (
            "data/processed/"
            f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_token}-"
            "train-normalization-stats.json"
        )
        or train.seed != 43
        or train.dataloader_seed != 43
        or train.checkpoint_path != f"{artifact}/final_model.pt"
        or train.last_checkpoint_path != f"{artifact}/last_model.pt"
        or train.formal_artifact_dir != artifact
        or train.epoch_checkpoint_dir != f"{artifact}/epochs"
        or train.checkpoint_schema
        != (
            f"RADAR-LN3-V21e-known-gate-oof-fold-{fold_token}-{arm}-"
            "checkpoint-v1"
        )
        or train.paired_baseline_config_path != expected_pair
        or train.save_every_epoch
    ):
        raise ValueError(
            "V21e OOF support config differs from its frozen fold/arm contract"
        )
    return True


def _validate_v21f_oof_support_config(
    config_path: Path,
    *,
    config_name: str,
    variant: str,
    data: KnownGateDataConfig,
    train: KnownGateTrainConfig,
) -> bool:
    """Recognize only V21f's frozen V21e-plus-spectral OOF Treatment configs."""

    match = _V21F_OOF_CONFIG_PATTERN.fullmatch(config_name)
    if match is None:
        return False
    fold_token = match.group(1)
    artifact = (
        "artifacts/RADAR-LN3-V21f-cq08-known-gate-course-interframe-paired-oof/"
        f"fold-{fold_token}/treatment"
    )
    expected_pair = (
        "config/RADAR-LN3/"
        f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_token}-"
        "treatment-seed43-config.json"
    )
    if (
        config_path.name != f"{config_name}-config.json"
        or config_path.parent.name != "RADAR-LN3"
        or config_path.parent.parent.name != "config"
        or variant != V21F_COURSE_INTERFRAME_VARIANT
        or tuple(COURSE_INTERFRAME_FEATURE_NAMES)
        != FEATURE_NAMES_BY_VARIANT[V21F_COURSE_INTERFRAME_VARIANT]
        or data.contract_path
        != (
            "data/processed/"
            f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_token}-"
            "data-contract.json"
        )
        or data.normalization_stats_path
        != (
            "data/processed/"
            f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{fold_token}-"
            "train-normalization-stats.json"
        )
        or train.seed != 43
        or train.dataloader_seed != 43
        or train.checkpoint_path != f"{artifact}/final_model.pt"
        or train.last_checkpoint_path != f"{artifact}/last_model.pt"
        or train.formal_artifact_dir != artifact
        or train.epoch_checkpoint_dir != f"{artifact}/epochs"
        or train.checkpoint_schema
        != (
            f"RADAR-LN3-V21f-known-gate-oof-fold-{fold_token}-treatment-"
            "checkpoint-v1"
        )
        or train.paired_baseline_config_path != expected_pair
        or train.save_every_epoch
    ):
        raise ValueError(
            "V21f OOF support config differs from its frozen fold Treatment contract"
        )
    return True


def load_known_gate_config(path: str | Path) -> KnownGateConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != V21_SCHEMA_VERSION:
        raise ValueError("KnownGate config must use RADAR-LN3-V21")
    config_name = str(_require(payload, "config_name"))
    variant = str(_require(payload, "variant"))
    if variant not in FEATURE_NAMES_BY_VARIANT:
        raise ValueError(f"unsupported V21 KnownGate variant: {variant!r}")
    task = dict(_require(payload, "task"))
    task_name = str(_require(task, "name"))
    class_names = tuple(str(value) for value in _require(task, "class_names"))
    label_to_id = {str(key): int(value) for key, value in dict(_require(task, "label_to_id")).items()}
    forbidden = tuple(str(value) for value in _require(task, "forbidden_training_labels"))
    feature_names = tuple(str(value) for value in _require(task, "feature_names"))
    if task_name != KNOWN_GATE_TASK_NAME:
        raise ValueError("V21 L1 task name must be known_gate")
    if class_names != KNOWN_GATE_CLASS_NAMES or label_to_id != KNOWN_GATE_LABEL_TO_ID:
        raise ValueError("V21 KnownGate class/label mapping differs from the strict-open-set contract")
    if forbidden != (UNKNOWN_LABEL,) or UNKNOWN_LABEL in label_to_id:
        raise ValueError("UnknownTarget must be explicitly forbidden and absent from V21 label_to_id")
    if feature_names != FEATURE_NAMES_BY_VARIANT[variant]:
        raise ValueError("V21 feature_names do not match the frozen variant")

    data_payload = dict(_require(payload, "data"))
    data = KnownGateDataConfig(
        index_path=str(_require(data_payload, "index_path")),
        contract_path=str(_require(data_payload, "contract_path")),
        normalization_stats_path=str(_require(data_payload, "normalization_stats_path")),
        train_split_name=str(_require(data_payload, "train_split_name")),
        validation_split_name=str(_require(data_payload, "validation_split_name")),
        test_split_name=str(_require(data_payload, "test_split_name")),
        index_schema_version=str(_require(data_payload, "index_schema_version")),
        contract_schema_version=str(_require(data_payload, "contract_schema_version")),
        expected_index_sha256=str(_require(data_payload, "expected_index_sha256")),
        expected_contract_sha256=str(_require(data_payload, "expected_contract_sha256")),
        expected_stats_sha256=str(_require(data_payload, "expected_stats_sha256")),
        max_sequence_length=max(1, int(_require(data_payload, "max_sequence_length"))),
        max_rd_frames_per_point=max(1, int(_require(data_payload, "max_rd_frames_per_point"))),
        rd_amplitude_center=float(_require(data_payload, "rd_amplitude_center")),
        rd_amplitude_scale=float(_require(data_payload, "rd_amplitude_scale")),
        feature_cache_manifest_path=(
            None if data_payload.get("feature_cache_manifest_path") is None else str(data_payload["feature_cache_manifest_path"])
        ),
        expected_feature_cache_manifest_sha256=(
            None
            if data_payload.get("expected_feature_cache_manifest_sha256") is None
            else str(data_payload["expected_feature_cache_manifest_sha256"])
        ),
        train_feature_cache_path=(
            None if data_payload.get("train_feature_cache_path") is None else str(data_payload["train_feature_cache_path"])
        ),
        validation_feature_cache_path=(
            None
            if data_payload.get("validation_feature_cache_path") is None
            else str(data_payload["validation_feature_cache_path"])
        ),
        expected_train_feature_cache_sha256=(
            None
            if data_payload.get("expected_train_feature_cache_sha256") is None
            else str(data_payload["expected_train_feature_cache_sha256"])
        ),
        expected_validation_feature_cache_sha256=(
            None
            if data_payload.get("expected_validation_feature_cache_sha256") is None
            else str(data_payload["expected_validation_feature_cache_sha256"])
        ),
        feature_experiment_contract_path=(
            None
            if data_payload.get("feature_experiment_contract_path") is None
            else str(data_payload["feature_experiment_contract_path"])
        ),
        expected_feature_experiment_contract_sha256=(
            None
            if data_payload.get("expected_feature_experiment_contract_sha256") is None
            else str(data_payload["expected_feature_experiment_contract_sha256"])
        ),
    )
    if (
        data.train_split_name,
        data.validation_split_name,
        data.test_split_name,
        data.index_schema_version,
        data.contract_schema_version,
    ) != ("train", "val", "test", "RADAR-LN3-V16", KNOWN_GATE_CONTRACT_SCHEMA):
        raise ValueError("V21 KnownGate split/index/contract versions are invalid")
    if not np.isfinite(data.rd_amplitude_center) or not np.isfinite(data.rd_amplitude_scale) or data.rd_amplitude_scale <= 0.0:
        raise ValueError("V21 RD normalization center/scale are invalid")

    model_payload = dict(_require(payload, "model"))
    model = KnownGateModelConfig(
        tr_value_dim=int(_require(model_payload, "tr_value_dim")),
        tr_missing_dim=int(_require(model_payload, "tr_missing_dim")),
        rd_quality_dim=int(_require(model_payload, "rd_quality_dim")),
        hidden_dim=int(_require(model_payload, "hidden_dim")),
        rd_embedding_dim=int(_require(model_payload, "rd_embedding_dim")),
        liquid_hidden_dim=int(_require(model_payload, "liquid_hidden_dim")),
        rd_height=int(_require(model_payload, "rd_height")),
        rd_width=int(_require(model_payload, "rd_width")),
        liquid_tau_min_seconds=float(_require(model_payload, "liquid_tau_min_seconds")),
        liquid_tau_max_seconds=float(_require(model_payload, "liquid_tau_max_seconds")),
        liquid_zero_dt_event_floor_seconds=float(_require(model_payload, "liquid_zero_dt_event_floor_seconds")),
        liquid_dt_max_seconds=float(_require(model_payload, "liquid_dt_max_seconds")),
        liquid_invalid_dt_update_mode=str(_require(model_payload, "liquid_invalid_dt_update_mode")),
        liquid_invalid_dt_observation_gain=float(_require(model_payload, "liquid_invalid_dt_observation_gain")),
        liquid_state_normalization=str(_require(model_payload, "liquid_state_normalization")),
        rd_frame_aggregation=str(_require(model_payload, "rd_frame_aggregation")),
        dt_reason_mode=str(_require(model_payload, "dt_reason_mode")),
        evidence_dim=int(_require(model_payload, "evidence_dim")),
    )
    if (model.tr_value_dim, model.tr_missing_dim, model.rd_quality_dim) != (14, 11, 7):
        raise ValueError("V21 KnownGate requires TR/missing/RD-quality dimensions 14/11/7")
    if (model.rd_height, model.rd_width) != (21, 384):
        raise ValueError("V21 KnownGate requires canonical RD shape 21x384")
    if model.evidence_dim != len(feature_names):
        raise ValueError("V21 evidence_dim does not match variant feature_names")
    if model.liquid_invalid_dt_update_mode != "bounded_observation":
        raise ValueError("V21 KnownGate baseline freezes bounded_observation dynamics")
    if model.dt_reason_mode != "v3_reason_aware_dynamics_only":
        raise ValueError("V21 KnownGate must use reason-aware dynamics-only dt semantics")
    if model.liquid_state_normalization != "recurrent" or model.rd_frame_aggregation != "masked_mean":
        raise ValueError("V21 KnownGate freezes recurrent normalization and masked_mean RD aggregation")

    train_payload = dict(_require(payload, "train"))
    class_weights = tuple(float(value) for value in _require(train_payload, "class_weights"))
    if len(class_weights) != 2 or any(not np.isfinite(value) or value <= 0.0 for value in class_weights):
        raise ValueError("V21 KnownGate class_weights must contain two finite positive values")
    train = KnownGateTrainConfig(
        batch_size=max(1, int(_require(train_payload, "batch_size"))),
        learning_rate=float(_require(train_payload, "learning_rate")),
        seed=int(_require(train_payload, "seed")),
        dataloader_seed=int(_require(train_payload, "dataloader_seed")),
        device=str(_require(train_payload, "device")),
        epochs=max(1, int(_require(train_payload, "epochs"))),
        early_stopping_patience=max(0, int(_require(train_payload, "early_stopping_patience"))),
        gradient_clip_norm=float(_require(train_payload, "gradient_clip_norm")),
        checkpoint_selection_metric=str(_require(train_payload, "checkpoint_selection_metric")),
        task_min_precision_recall_target=float(_require(train_payload, "task_min_precision_recall_target")),
        checkpoint_path=str(_require(train_payload, "checkpoint_path")),
        last_checkpoint_path=str(_require(train_payload, "last_checkpoint_path")),
        formal_artifact_dir=str(_require(train_payload, "formal_artifact_dir")),
        epoch_checkpoint_dir=str(_require(train_payload, "epoch_checkpoint_dir")),
        class_weights=(float(class_weights[0]), float(class_weights[1])),
        loss_mode=str(_require(train_payload, "loss_mode")),
        prefix_auxiliary_weight=float(_require(train_payload, "prefix_auxiliary_weight")),
        prefix_auxiliary_min_events=max(1, int(_require(train_payload, "prefix_auxiliary_min_events"))),
        checkpoint_schema=str(_require(train_payload, "checkpoint_schema")),
        record_parent_order_hash=_as_bool(_require(train_payload, "record_parent_order_hash")),
        save_every_epoch=_as_bool(_require(train_payload, "save_every_epoch")),
        cuda_deterministic_algorithms=_as_bool(_require(train_payload, "cuda_deterministic_algorithms")),
        full_parameter_training=_as_bool(_require(train_payload, "full_parameter_training")),
        paired_baseline_config_path=(
            None
            if train_payload.get("paired_baseline_config_path") is None
            else str(train_payload["paired_baseline_config_path"])
        ),
    )
    is_v21e_oof_support = _validate_v21e_oof_support_config(
        config_path,
        config_name=config_name,
        variant=variant,
        data=data,
        train=train,
    )
    is_v21f_oof_support = _validate_v21f_oof_support_config(
        config_path,
        config_name=config_name,
        variant=variant,
        data=data,
        train=train,
    )
    is_oof_support = is_v21e_oof_support or is_v21f_oof_support
    if (
        train.seed != train.dataloader_seed
        or train.seed not in KNOWN_GATE_BASELINE_MULTISEED_SEEDS
        or train.epochs != 20
        or train.early_stopping_patience != 0
        or train.loss_mode != KNOWN_GATE_LOSS_MODE
        or train.checkpoint_selection_metric != KNOWN_GATE_SELECTION_METRIC
        or train.prefix_auxiliary_weight != 0.1
        or train.prefix_auxiliary_min_events != 3
        or not train.record_parent_order_hash
        or (not is_oof_support and not train.save_every_epoch)
        or train.cuda_deterministic_algorithms
        or not train.full_parameter_training
    ):
        raise ValueError("V21 KnownGate training protocol differs from the frozen full-training contract")
    if not 0.0 < train.learning_rate or not 0.0 < train.gradient_clip_norm:
        raise ValueError("V21 learning rate and gradient clip must be positive")
    if abs(train.task_min_precision_recall_target - 0.90) > 1.0e-12:
        raise ValueError("V21 KnownGate local performance target must be 0.90")
    if (
        not is_oof_support
        and not train.checkpoint_schema.startswith(f"{V21_SCHEMA_VERSION}-{variant}-")
    ):
        raise ValueError("V21 checkpoint schema must bind the experiment variant")
    seed_token = f"seed{train.seed}"
    if seed_token not in config_name:
        raise ValueError("V21 config_name must bind the training seed")
    for field_name, field_value in (
        ("checkpoint_path", train.checkpoint_path),
        ("last_checkpoint_path", train.last_checkpoint_path),
        ("formal_artifact_dir", train.formal_artifact_dir),
        ("epoch_checkpoint_dir", train.epoch_checkpoint_dir),
    ):
        if not is_oof_support and seed_token not in field_value:
            raise ValueError(f"V21 {field_name} must bind the training seed")
    if (
        not is_oof_support
        and train.seed != 42
        and seed_token not in train.checkpoint_schema
    ):
        raise ValueError("V21 additional-seed checkpoint schema must bind the seed")

    cache_fields = (
        data.feature_cache_manifest_path,
        data.expected_feature_cache_manifest_sha256,
        data.train_feature_cache_path,
        data.validation_feature_cache_path,
        data.expected_train_feature_cache_sha256,
        data.expected_validation_feature_cache_sha256,
    )
    experiment_contract_fields = (
        data.feature_experiment_contract_path,
        data.expected_feature_experiment_contract_sha256,
    )
    if variant == BASELINE_VARIANT:
        if (
            any(value is not None for value in cache_fields)
            or any(value is not None for value in experiment_contract_fields)
            or train.paired_baseline_config_path is not None
        ):
            raise ValueError("V21 Baseline must not load feature caches or reference itself as a pair")
    else:
        required_cache_fields = (
            data.feature_cache_manifest_path,
            data.expected_feature_cache_manifest_sha256,
            data.train_feature_cache_path,
            data.expected_train_feature_cache_sha256,
        )
        validation_cache_fields = (
            data.validation_feature_cache_path,
            data.expected_validation_feature_cache_sha256,
        )
        if (
            any(value is None for value in required_cache_fields)
            or train.paired_baseline_config_path is None
        ):
            raise ValueError("V21 Treatment requires feature caches and an explicit Baseline config")
        if is_oof_support:
            if (validation_cache_fields[0] is None) != (
                validation_cache_fields[1] is None
            ):
                raise ValueError(
                    "V21 OOF validation cache path/hash must be both present or both absent"
                )
        elif any(value is None for value in validation_cache_fields):
            raise ValueError("V21 formal Treatment requires a validation feature cache")
        if any(value is None for value in experiment_contract_fields) and any(
            value is not None for value in experiment_contract_fields
        ):
            raise ValueError(
                "V21 Treatment feature experiment contract path/hash must be both present or both absent"
            )

    return KnownGateConfig(
        schema_version=V21_SCHEMA_VERSION,
        config_name=config_name,
        variant=variant,
        task_name=task_name,
        class_names=(class_names[0], class_names[1]),
        label_to_id=label_to_id,
        forbidden_training_labels=forbidden,
        feature_names=feature_names,
        data=data,
        model=model,
        train=train,
    )


def known_gate_config_identity_sha256(config: KnownGateConfig) -> str:
    data_payload = asdict(config.data)
    if (
        config.data.feature_experiment_contract_path is None
        and config.data.expected_feature_experiment_contract_sha256 is None
    ):
        data_payload.pop("feature_experiment_contract_path")
        data_payload.pop("expected_feature_experiment_contract_sha256")
    payload = {
        "schema_version": config.schema_version,
        "config_name": config.config_name,
        "variant": config.variant,
        "task_name": config.task_name,
        "class_names": list(config.class_names),
        "label_to_id": dict(config.label_to_id),
        "forbidden_training_labels": list(config.forbidden_training_labels),
        "feature_names": list(config.feature_names),
        "data": data_payload,
        "model": asdict(config.model),
        "train": asdict(config.train),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass
class KnownGateSequence:
    base: PointSequence
    evidence_features: torch.Tensor
    evidence_valid: torch.Tensor


class KnownGateDataset(Dataset[KnownGateSequence]):
    def __init__(
        self,
        base: CQ08PointSequenceDataset,
        config: KnownGateConfig,
        *,
        allow_feature_cache_superset: bool = False,
    ) -> None:
        self.base = base
        self.config = config
        self.parent_ids = list(base.parent_ids)
        self._base_indices = {str(parent_id): index for index, parent_id in enumerate(base.parent_ids)}
        self._evidence_values: dict[str, np.ndarray] | None = None
        self._evidence_valid: dict[str, np.ndarray] | None = None
        self.feature_cache_metadata: dict[str, Any] | None = None
        if config.model.evidence_dim:
            if base.frame["split_name"].astype(str).nunique() != 1:
                raise ValueError("KnownGate base Dataset must contain one split")
            split = str(base.frame["split_name"].iloc[0])
            if split == config.data.train_split_name:
                cache_path = config.data.train_feature_cache_path
            elif split == config.data.validation_split_name:
                cache_path = config.data.validation_feature_cache_path
            else:
                raise PermissionError("V21 feature cache is restricted to train/validation")
            if cache_path is None:
                raise ValueError("V21 Treatment Dataset has no feature cache path")
            values, valid, metadata = load_known_gate_feature_cache(
                _resolve(base.project_root, cache_path),
                expected_sample_ids=base.frame["sample_id"].astype(str).tolist(),
                variant=config.variant,
                allow_superset=allow_feature_cache_superset,
            )
            self._evidence_values = values
            self._evidence_valid = valid
            self.feature_cache_metadata = metadata

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int) -> KnownGateSequence:
        parent_id = self.parent_ids[int(index)]
        base_item = self.base[self._base_indices[parent_id]]
        length = int(base_item.event_count)
        if self.config.model.evidence_dim == 0:
            values = torch.zeros((length, 0), dtype=torch.float32)
            valid = torch.zeros((length, 0), dtype=torch.bool)
        else:
            if self._evidence_values is None or self._evidence_valid is None:
                raise RuntimeError("V21 evidence cache was not initialized")
            sample_ids = self.base._groups[parent_id]["sample_id"].astype(str).tolist()
            values = torch.from_numpy(
                np.stack([self._evidence_values[sample_id] for sample_id in sample_ids], axis=0).astype(np.float32)
            )
            valid = torch.from_numpy(
                np.stack([self._evidence_valid[sample_id] for sample_id in sample_ids], axis=0).astype(np.bool_)
            )
            if values.shape != (length, self.config.model.evidence_dim) or valid.shape != values.shape:
                raise ValueError("V21 parent feature cache shape differs from its causal sequence")
        return KnownGateSequence(base=base_item, evidence_features=values, evidence_valid=valid)

    def label_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.config.class_names}
        for parent_id in self.parent_ids:
            label = str(self.base._groups[parent_id].iloc[0]["label_name"])
            counts[self.config.class_names[KNOWN_GATE_LABEL_TO_ID[label]]] += 1
        return counts


def build_known_gate_dataset(
    project_root: Path,
    config: KnownGateConfig,
    *,
    split_name: str,
    parent_limit: int | None = None,
    parent_ids: Sequence[str] | None = None,
    allow_feature_cache_superset: bool = False,
) -> KnownGateDataset:
    if split_name not in {config.data.train_split_name, config.data.validation_split_name}:
        raise PermissionError("V21 local KnownGate Dataset keeps test sealed")
    base = CQ08PointSequenceDataset(
        project_root,
        index_path=_resolve(project_root, config.data.index_path),
        split_name=split_name,
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=False,
        sample_limit=None,
        expected_schema_version=config.data.index_schema_version,
        allowed_labels=KNOWN_LABELS,
        forbidden_labels={UNKNOWN_LABEL},
    )
    if parent_limit is not None and parent_ids is not None:
        raise ValueError("KnownGate Dataset cannot combine parent_limit with parent_ids")
    if parent_ids is not None:
        requested = {str(value) for value in parent_ids}
        if not requested or len(requested) != len(parent_ids):
            raise ValueError("KnownGate parent_ids must be non-empty and unique")
        available = set(base.parent_ids)
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"KnownGate requested parent_ids are unavailable: {missing[:3]}")
        base.parent_ids = [value for value in base.parent_ids if value in requested]
        base.frame = base.frame[
            base.frame["parent_sample_id"].astype(str).isin(requested)
        ].copy()
        base._groups = {
            key: value for key, value in base._groups.items() if key in requested
        }
    elif parent_limit is not None:
        if parent_limit <= 0:
            raise ValueError("V21 parent_limit must be positive")
        base.parent_ids = base.parent_ids[: int(parent_limit)]
        selected = set(base.parent_ids)
        base.frame = base.frame[base.frame["parent_sample_id"].astype(str).isin(selected)].copy()
        base._groups = {key: value for key, value in base._groups.items() if key in selected}
    return KnownGateDataset(
        base,
        config,
        allow_feature_cache_superset=allow_feature_cache_superset,
    )


def collate_known_gate_sequences(items: list[KnownGateSequence]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty V21 KnownGate batch")
    batch = collate_point_sequences([item.base for item in items])
    max_length = int(batch["tr_features"].shape[1])
    evidence_dim = int(items[0].evidence_features.shape[-1])
    evidence = torch.zeros((len(items), max_length, evidence_dim), dtype=torch.float32)
    valid = torch.zeros((len(items), max_length, evidence_dim), dtype=torch.bool)
    labels: list[int] = []
    for batch_index, item in enumerate(items):
        length = int(item.base.event_count)
        if item.evidence_features.shape != (length, evidence_dim) or item.evidence_valid.shape != (length, evidence_dim):
            raise ValueError("V21 KnownGate batch contains incompatible evidence shapes")
        evidence[batch_index, :length] = item.evidence_features
        valid[batch_index, :length] = item.evidence_valid
        try:
            labels.append(KNOWN_GATE_LABEL_TO_ID[item.base.label_name])
        except KeyError as exc:
            raise ValueError(f"V21 KnownGate batch contains forbidden label {item.base.label_name!r}") from exc
    batch["labels"] = torch.tensor(labels, dtype=torch.long)
    batch["task_label_names"] = [KNOWN_GATE_CLASS_NAMES[value] for value in labels]
    batch["base_label_names"] = [item.base.label_name for item in items]
    if evidence_dim:
        if bool(valid.masked_select(batch["tr_padding_mask"].unsqueeze(-1)).any()):
            raise ValueError("padded V21 events must not contain valid evidence")
        batch["known_gate_features"] = evidence
        batch["known_gate_feature_valid"] = valid
    return batch


def build_known_gate_dataloader(
    dataset: KnownGateDataset,
    *,
    batch_size: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=collate_known_gate_sequences,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


class KnownGateFusionLinear(nn.Linear):
    """Explicit shared base columns plus zero-initialized experiment columns."""

    def __init__(self, in_features: int, out_features: int, *, base_input_dim: int) -> None:
        super().__init__(in_features, out_features)
        self.base_input_dim = int(base_input_dim)
        if not 0 < self.base_input_dim <= self.in_features:
            raise ValueError("V21 Fusion base_input_dim is invalid")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError("V21 Fusion received an invalid final dimension")
        base = nn.functional.linear(
            value[..., : self.base_input_dim],
            self.weight[:, : self.base_input_dim],
            self.bias,
        )
        if self.base_input_dim == self.in_features:
            return base
        return base + nn.functional.linear(
            value[..., self.base_input_dim :],
            self.weight[:, self.base_input_dim :],
            None,
        )


class KnownGateClassifier(nn.Module):
    def __init__(self, config: KnownGateConfig) -> None:
        super().__init__()
        self.config = config
        model = config.model
        self.task_name = KNOWN_GATE_TASK_NAME
        self.class_names = KNOWN_GATE_CLASS_NAMES
        self.evidence_dim = model.evidence_dim
        self.tr_encoder = nn.Sequential(
            nn.Linear(model.tr_value_dim + model.tr_missing_dim, model.hidden_dim),
            nn.LayerNorm(model.hidden_dim),
            nn.SiLU(),
            nn.Linear(model.hidden_dim, model.hidden_dim),
            nn.SiLU(),
        )
        self.rd_encoder = RDSpatialEncoder(
            model.rd_embedding_dim, aggregation_mode=model.rd_frame_aggregation
        )
        self.base_fusion_input_dim = model.hidden_dim + model.rd_embedding_dim + model.rd_quality_dim
        self.fusion_input_dim = self.base_fusion_input_dim + self.evidence_dim
        self.fusion = nn.Sequential(
            KnownGateFusionLinear(
                self.fusion_input_dim,
                model.liquid_hidden_dim,
                base_input_dim=self.base_fusion_input_dim,
            ),
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.SiLU(),
        )
        self.liquid = CfCLiquidCausalEncoder(
            model.liquid_hidden_dim,
            model.liquid_hidden_dim,
            tau_min_seconds=model.liquid_tau_min_seconds,
            tau_max_seconds=model.liquid_tau_max_seconds,
            zero_dt_event_floor_seconds=model.liquid_zero_dt_event_floor_seconds,
            dt_max_seconds=model.liquid_dt_max_seconds,
            invalid_dt_update_mode=model.liquid_invalid_dt_update_mode,
            invalid_dt_observation_gain=model.liquid_invalid_dt_observation_gain,
            state_normalization_mode=model.liquid_state_normalization,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.Linear(model.liquid_hidden_dim, len(KNOWN_GATE_CLASS_NAMES)),
        )

    def _evidence_input(
        self,
        features: torch.Tensor | None,
        valid: torch.Tensor | None,
        *,
        leading_shape: tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.evidence_dim == 0:
            if features is not None or valid is not None:
                raise ValueError("V21 Baseline must not receive experimental evidence")
            return None
        expected = (*leading_shape, self.evidence_dim)
        if features is None or valid is None or features.shape != expected or valid.shape != expected:
            raise ValueError("V21 Treatment evidence value/valid shape is invalid")
        if valid.dtype != torch.bool or not torch.isfinite(features).all():
            raise ValueError("V21 Treatment evidence must be finite with a boolean validity mask")
        if bool((features < 0.0).any()) or bool((features > 1.0).any()):
            raise ValueError("V21 Treatment evidence must be bounded to [0,1]")
        if bool((features.masked_select(~valid) != 0).any()):
            raise ValueError("invalid V21 Treatment evidence must be exactly zero")
        return features * valid.to(dtype=dtype)

    def forward(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        tr_padding_mask: torch.Tensor,
        known_gate_features: torch.Tensor | None = None,
        known_gate_feature_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        model = self.config.model
        if tr_features.ndim != 3 or tr_features.shape[-1] != model.tr_value_dim:
            raise ValueError("V21 TR features must have shape [batch,sequence,14]")
        if tr_missing_mask.shape != (*tr_features.shape[:2], model.tr_missing_dim):
            raise ValueError("V21 TR missing mask shape is invalid")
        if tr_dt_seconds.shape != tr_features.shape[:2] or tr_dt_valid.shape != tr_features.shape[:2]:
            raise ValueError("V21 dt tensors must align with the sequence")
        if tr_dt_reason.shape != tr_features.shape[:2] or tr_padding_mask.shape != tr_features.shape[:2]:
            raise ValueError("V21 dt reason/padding mask shape is invalid")
        if rd_quality.shape != (*tr_features.shape[:2], model.rd_quality_dim):
            raise ValueError("V21 RD quality shape is invalid")
        if not all(torch.isfinite(value).all() for value in (tr_features, tr_missing_mask, tr_dt_seconds, rd_frames, rd_quality)):
            raise ValueError("V21 model received a non-finite base input")
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        rd_embedding = self.rd_encoder(rd_frames, rd_frame_mask)
        fusion_inputs = [tr_embedding, rd_embedding, rd_quality]
        evidence = self._evidence_input(
            known_gate_features,
            known_gate_feature_valid,
            leading_shape=tuple(tr_features.shape[:2]),
            dtype=tr_features.dtype,
        )
        if evidence is not None:
            if bool(known_gate_feature_valid.masked_select(tr_padding_mask.unsqueeze(-1)).any()):
                raise ValueError("padded V21 events contain valid experiment evidence")
            fusion_inputs.append(evidence)
        point_events = self.fusion(torch.cat(fusion_inputs, dim=-1))
        state = self.liquid(
            point_events,
            tr_dt_seconds,
            tr_padding_mask,
            dt_valid=tr_dt_valid,
            dt_reason=tr_dt_reason,
        )
        sequence_logits = self.classifier(state.sequence)
        lengths = (~tr_padding_mask).long().sum(dim=1)
        if bool((lengths <= 0).any()):
            raise ValueError("V21 model received an empty sequence")
        indices = lengths - 1
        batch_indices = torch.arange(tr_features.shape[0], device=tr_features.device)
        return {
            "sequence_logits": sequence_logits,
            "window_logits": sequence_logits[batch_indices, indices],
            "liquid_sequence": state.sequence,
            "liquid_final_state": state.final_state,
            "dt_clamped_mask": state.dt_clamped_mask,
            "dt_zero_mask": state.dt_zero_mask,
            "dt_invalid_mask": state.dt_invalid_mask,
            "dt_first_event_mask": state.dt_first_event_mask,
            "dt_zero_delta_mask": state.dt_zero_delta_mask,
        }

    def forward_online_point(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        liquid_state: torch.Tensor | None = None,
        known_gate_features: torch.Tensor | None = None,
        known_gate_feature_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 2 or tr_missing_mask.ndim != 2 or rd_quality.ndim != 2:
            raise ValueError("V21 online TR/RD quality tensors must be two-dimensional")
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        rd_embedding = self.rd_encoder(rd_frames.unsqueeze(1), rd_frame_mask.unsqueeze(1)).squeeze(1)
        fusion_inputs = [tr_embedding, rd_embedding, rd_quality]
        evidence = self._evidence_input(
            known_gate_features,
            known_gate_feature_valid,
            leading_shape=(tr_features.shape[0],),
            dtype=tr_features.dtype,
        )
        if evidence is not None:
            fusion_inputs.append(evidence)
        point_event = self.fusion(torch.cat(fusion_inputs, dim=-1))
        state, dt_clamped, dt_zero = self.liquid.step(
            point_event,
            tr_dt_seconds,
            liquid_state,
            tr_dt_valid,
            tr_dt_reason,
        )
        logits = self.classifier(self.liquid.readout(state))
        return {
            "window_logits": logits,
            "liquid_state": state,
            "dt_clamped_mask": dt_clamped,
            "dt_zero_mask": dt_zero,
            "dt_invalid_mask": ~tr_dt_valid,
        }


def seed_everything(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def build_known_gate_model_for_config(
    project_root: Path,
    config: KnownGateConfig,
) -> tuple[KnownGateClassifier, KnownGateClassifier | None, dict[str, Any]]:
    if config.variant == BASELINE_VARIANT:
        seed_everything(config.train.seed)
        return KnownGateClassifier(config), None, {
            "paired_baseline": False,
            "initialization": "seeded_full_random_initialization",
            "seed": config.train.seed,
        }
    if config.train.paired_baseline_config_path is None:
        raise ValueError("V21 Treatment lacks paired_baseline_config_path")
    baseline_path = _resolve(project_root, config.train.paired_baseline_config_path)
    baseline_config = load_known_gate_config(baseline_path)
    if (
        baseline_config.variant != BASELINE_VARIANT
        and tuple(config.feature_names[: len(baseline_config.feature_names)])
        != tuple(baseline_config.feature_names)
    ):
        raise ValueError(
            "V21 Treatment paired reference features are not an exact prefix"
        )
    if config.model.evidence_dim <= baseline_config.model.evidence_dim:
        raise ValueError("V21 Treatment must append at least one evidence column")
    if baseline_config.train.seed != config.train.seed or baseline_config.train.dataloader_seed != config.train.dataloader_seed:
        raise ValueError("V21 Baseline/Treatment seeds differ")
    comparable_data = {
        key: value
        for key, value in asdict(config.data).items()
        if not key.startswith("feature_")
        and "feature_cache" not in key
        and "feature_experiment_contract" not in key
    }
    baseline_data = {
        key: value
        for key, value in asdict(baseline_config.data).items()
        if not key.startswith("feature_")
        and "feature_cache" not in key
        and "feature_experiment_contract" not in key
    }
    if comparable_data != baseline_data:
        raise ValueError("V21 Treatment base data contract differs from Baseline")
    base_model_fields = asdict(config.model)
    base_model_fields["evidence_dim"] = baseline_config.model.evidence_dim
    if base_model_fields != asdict(baseline_config.model):
        raise ValueError("V21 Treatment model differs beyond appended evidence_dim")
    excluded_train = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "checkpoint_schema",
        "paired_baseline_config_path",
    }
    treatment_train = {key: value for key, value in asdict(config.train).items() if key not in excluded_train}
    baseline_train = {key: value for key, value in asdict(baseline_config.train).items() if key not in excluded_train}
    if treatment_train != baseline_train:
        raise ValueError("V21 Treatment training core differs from Baseline")

    baseline, nested_reference, nested_pairing = build_known_gate_model_for_config(
        project_root, baseline_config
    )
    nested_reference_present = nested_reference is not None
    del nested_reference
    seed_everything(config.train.seed)
    treatment = KnownGateClassifier(config)
    baseline_parameters = dict(baseline.named_parameters())
    treatment_parameters = dict(treatment.named_parameters())
    baseline_buffers = dict(baseline.named_buffers())
    treatment_buffers = dict(treatment.named_buffers())
    with torch.no_grad():
        for name, baseline_value in baseline_parameters.items():
            treatment_value = treatment_parameters.get(name)
            if treatment_value is None:
                raise ValueError(f"V21 Treatment lacks shared parameter {name}")
            if name == "fusion.0.weight":
                if treatment_value.shape != (
                    baseline_value.shape[0],
                    baseline_value.shape[1]
                    + config.model.evidence_dim
                    - baseline_config.model.evidence_dim,
                ):
                    raise ValueError("V21 Treatment Fusion width is not Baseline plus evidence_dim")
                treatment_value[:, : baseline_value.shape[1]].copy_(baseline_value)
                treatment_value[:, baseline_value.shape[1] :].zero_()
            elif treatment_value.shape != baseline_value.shape:
                raise ValueError(f"V21 shared parameter shape differs at {name}")
            else:
                treatment_value.copy_(baseline_value)
        for name, baseline_value in baseline_buffers.items():
            treatment_value = treatment_buffers.get(name)
            if treatment_value is None or treatment_value.shape != baseline_value.shape:
                raise ValueError(f"V21 Treatment lacks compatible shared buffer {name}")
            treatment_value.copy_(baseline_value)
    for name, baseline_value in baseline.state_dict().items():
        treatment_value = treatment.state_dict()[name]
        comparable = treatment_value[:, : baseline_value.shape[1]] if name == "fusion.0.weight" else treatment_value
        if not torch.equal(baseline_value, comparable):
            raise AssertionError(f"V21 shared initialization differs at {name}")
    new_columns = treatment.state_dict()["fusion.0.weight"][:, baseline.fusion_input_dim :]
    if not torch.equal(new_columns, torch.zeros_like(new_columns)):
        raise AssertionError("V21 new experiment Fusion columns are not exactly zero")
    return treatment, baseline, {
        "paired_baseline": True,
        "baseline_config_path": str(baseline_path),
        "baseline_config_name": baseline_config.config_name,
        "baseline_initialization": nested_pairing,
        "nested_reference_present": nested_reference_present,
        "seed": config.train.seed,
        "shared_parameters_and_buffers_exactly_equal": True,
        "baseline_fusion_input_dim": baseline.fusion_input_dim,
        "treatment_fusion_input_dim": treatment.fusion_input_dim,
        "baseline_evidence_dim": baseline_config.model.evidence_dim,
        "treatment_evidence_dim": config.model.evidence_dim,
        "new_fusion_columns": (
            config.model.evidence_dim - baseline_config.model.evidence_dim
        ),
        "new_fusion_columns_exactly_zero": True,
        "full_parameter_training": True,
    }


def summarize_known_gate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    scope: str,
) -> dict[str, Any]:
    rows = list(records)
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in rows:
        true_id, pred_id = int(row["true_id"]), int(row["pred_id"])
        if true_id not in {0, 1} or pred_id not in {0, 1}:
            raise ValueError("V21 metric record has an invalid class id")
        matrix[true_id, pred_id] += 1
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    per_class: list[dict[str, Any]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(KNOWN_GATE_CLASS_NAMES):
        tp = int(matrix[class_id, class_id])
        precision = tp / int(predicted[class_id]) if predicted[class_id] else 0.0
        recall = tp / int(support[class_id]) if support[class_id] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(float(precision))
        recall_values.append(float(recall))
        f1_values.append(float(f1))
        other = 1 - class_id
        per_class.append(
            {
                "class": class_name,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "support": int(support[class_id]),
                "major_confusion_to": KNOWN_GATE_CLASS_NAMES[other] if matrix[class_id, other] else None,
                "major_confusion_count": int(matrix[class_id, other]),
            }
        )
    total = int(matrix.sum())
    return {
        "scope": scope,
        "support": total,
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
    }


def _base_label_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(records)
    result: dict[str, Any] = {}
    for label in KNOWN_LABELS:
        selected = [row for row in rows if str(row["label_name"]) == label]
        expected_id = KNOWN_GATE_LABEL_TO_ID[label]
        correct = sum(int(row["pred_id"]) == expected_id for row in selected)
        result[label] = {
            "support": len(selected),
            "correct_parent_count": int(correct),
            "expected_gate_class": KNOWN_GATE_CLASS_NAMES[expected_id],
            "acceptance_recall": float(correct / len(selected)) if selected else None,
        }
    return result


def evaluate_known_gate(
    model: KnownGateClassifier,
    dataset: KnownGateDataset,
    *,
    config: KnownGateConfig,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = build_known_gate_dataloader(
        dataset,
        batch_size=batch_size or config.train.batch_size,
        shuffle=False,
    )
    window_records: list[dict[str, Any]] = []
    eligible_prefix_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    dt_reason_counts = {name: 0 for name in DT_REASON_NAMES}
    with torch.inference_mode():
        for batch in loader:
            tensors = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(
                tr_features=tensors["tr_features"],
                tr_missing_mask=tensors["tr_missing_mask"],
                tr_dt_seconds=tensors["tr_dt_seconds"],
                tr_dt_valid=tensors["tr_dt_valid"],
                tr_dt_reason=tensors["tr_dt_reason"],
                rd_frames=tensors["rd_frames"],
                rd_frame_mask=tensors["rd_frame_mask"],
                rd_quality=tensors["rd_quality"],
                tr_padding_mask=tensors["tr_padding_mask"],
                known_gate_features=tensors.get("known_gate_features"),
                known_gate_feature_valid=tensors.get("known_gate_feature_valid"),
            )
            probabilities = torch.softmax(output["sequence_logits"], dim=-1)
            active_reasons = tensors["tr_dt_reason"].masked_select(~tensors["tr_padding_mask"])
            for reason_id, reason_name in enumerate(DT_REASON_NAMES):
                dt_reason_counts[reason_name] += int((active_reasons == reason_id).sum().item())
            for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][batch_index].item())
                true_id = int(batch["labels"][batch_index].item())
                final: dict[str, Any] | None = None
                for event_index in range(length):
                    event_prob = probabilities[batch_index, event_index]
                    pred_id = int(event_prob.argmax().item())
                    final = {
                        "true_id": true_id,
                        "pred_id": pred_id,
                        "confidence": float(event_prob[pred_id].item()),
                        "known_target_probability": float(event_prob[0].item()),
                        "known_clutter_probability": float(event_prob[1].item()),
                        "parent_sample_id": str(parent_id),
                        "track_id": str(batch["track_ids"][batch_index]),
                        "source_name": str(batch["source_names"][batch_index]),
                        "label_name": str(batch["base_label_names"][batch_index]),
                        "event_index": event_index,
                        "window_length": event_index + 1,
                        "product_action_eligible": event_index + 1 >= config.train.prefix_auxiliary_min_events,
                    }
                    window_records.append(dict(final))
                    if event_index + 1 >= config.train.prefix_auxiliary_min_events and event_index + 1 < length:
                        eligible_prefix_records.append(dict(final))
                if final is None:
                    raise ValueError(f"V21 evaluation found an empty parent {parent_id}")
                parent_records.append(final)
    active_event_count = sum(value for key, value in dt_reason_counts.items() if key != "padding")
    result = {
        "schema_version": "RADAR-LN3-V21-known-gate-evaluation-v1",
        "task_name": KNOWN_GATE_TASK_NAME,
        "variant": config.variant,
        "class_names": list(KNOWN_GATE_CLASS_NAMES),
        "parent_count": len(parent_records),
        "window_count": len(window_records),
        "window_metrics": summarize_known_gate_records(window_records, scope="causal_prefix_windows"),
        "eligible_prefix_metrics": summarize_known_gate_records(
            eligible_prefix_records, scope="eligible_causal_prefix_auxiliary_excluding_final"
        ),
        "parent_metrics": summarize_known_gate_records(parent_records, scope="parent_final"),
        "base_label_parent_metrics": _base_label_metrics(parent_records),
        "prefix_action_contract": {
            "minimum_evidence_events": config.train.prefix_auxiliary_min_events,
            "events_before_minimum": "PENDING_ABSTAIN",
            "local_argmax_is_product_action": False,
        },
        "dt_quality": {
            "active_event_count": active_event_count,
            "dt_reason_counts": dt_reason_counts,
            "zero_delta_is_gap": False,
        },
        "records": {
            "window": window_records,
            "eligible_prefix": eligible_prefix_records,
            "parent": parent_records,
        },
        "unknown_probability_output": False,
        "formal_rejection": "未执行",
        "test": "未执行",
        "OOF": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }
    return result


def known_gate_device(config: KnownGateConfig) -> torch.device:
    return torch.device(
        "cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu"
    )


__all__ = [
    "KNOWN_CLUTTER_LABEL",
    "KNOWN_GATE_CLASS_NAMES",
    "KNOWN_GATE_CONTRACT_SCHEMA",
    "KNOWN_GATE_BASELINE_MULTISEED_READINESS_SCHEMA",
    "KNOWN_GATE_BASELINE_MULTISEED_SEEDS",
    "KNOWN_GATE_V21A_READINESS_SCHEMA",
    "KNOWN_GATE_V21A_MANIFEST_SCHEMA",
    "KNOWN_GATE_V21B_READINESS_SCHEMA",
    "KNOWN_GATE_V21C_READINESS_SCHEMA",
    "KNOWN_GATE_V21D_MANIFEST_SCHEMA",
    "KNOWN_GATE_V21D_READINESS_SCHEMA",
    "KNOWN_GATE_V21E_MANIFEST_SCHEMA",
    "KNOWN_GATE_V21E_READINESS_SCHEMA",
    "KNOWN_GATE_LABEL_TO_ID",
    "KNOWN_GATE_LOSS_MODE",
    "KNOWN_GATE_MANIFEST_SCHEMA",
    "KNOWN_GATE_READINESS_SCHEMA",
    "KNOWN_GATE_SELECTION_METRIC",
    "KNOWN_LABELS",
    "KNOWN_TARGET_LABELS",
    "KnownGateClassifier",
    "KnownGateConfig",
    "KnownGateDataset",
    "UNKNOWN_LABEL",
    "V21_SCHEMA_VERSION",
    "build_known_gate_dataloader",
    "build_known_gate_dataset",
    "build_known_gate_model_for_config",
    "evaluate_known_gate",
    "known_gate_config_identity_sha256",
    "known_gate_device",
    "load_known_gate_config",
    "seed_everything",
    "sha256_file",
    "summarize_known_gate_records",
]
