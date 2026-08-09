"""Independent liquid-model task contracts for the DBB/CU experiments.

This module deliberately reuses only the isolated ``src-v3`` implementation.
It does not load the five-class checkpoint, change the CQ08 index, or expose
paths and identifiers to the model.  The three supported tasks are:

* ``dbb_cu_gate``: DBB versus CU;
* ``dbb_specialist``: Drone versus Bird versus Balloon;
* historical ``cu_specialist``: Clutter versus Unknown;
* V13 ``cu_specialist``: Clutter-accept versus Clutter-reject evidence.

V5-V8 use the shared causal TR+RD liquid contract.  V9 is a deliberately
isolated DBB/CU Gate experiment that adds causal TR motion-consistency values;
the specialists remain on their frozen earlier contracts.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import (
    CQ08PointSequenceDataset,
    DT_REASON_NAMES,
    PointSequence,
    RD_QUALITY_NAMES,
    RD_QUALITY_SCHEMA,
    collate_point_sequences,
    require_causal_rd_quality_schema,
)
from .features import (
    TR_MOTION_SCHEMA,
    TR_OBSERVATION_INCREMENT_SCHEMA,
    resolve_causal_tr_feature_contract,
)
from .model import CfCLiquidCausalEncoder, RDSpatialEncoder
from .rd_physical import (
    RD_PHYSICAL_MASK_MODE,
    RD_PHYSICAL_NAMES,
    RD_PHYSICAL_SCHEMA,
    RD_PHYSICAL_VALUE_DIM,
    extract_current_event_rd_physical,
    rd_physical_contract,
)


BASE_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
TASK_SCHEMA_VERSION = "RADAR-LN3-V5"
REPAIRED_TASK_SCHEMA_VERSION = "RADAR-LN3-V6"
DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION = "RADAR-LN3-V7"
WEAK_CLASS_SELECTION_SCHEMA_VERSION = "RADAR-LN3-V8"
TR_MOTION_GATE_SCHEMA_VERSION = "RADAR-LN3-V9"
TR_OBSERVATION_INCREMENT_GATE_SCHEMA_VERSION = "RADAR-LN3-V10"
RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION = "RADAR-LN3-V11"
RD_PHYSICAL_PAIRED_PROTOCOL = "v10-control-v11-current-event-rd-physical-v1"
DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION = "RADAR-LN3-V12"
DBB_AUXILIARY_PAIRED_PROTOCOL = "v11-control-v12-dbb-subclass-auxiliary-v1"
DBB_AUXILIARY_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")
DBB_AUXILIARY_LABEL_TO_ID = {
    class_name: class_id for class_id, class_name in enumerate(DBB_AUXILIARY_CLASS_NAMES)
}
DBB_AUXILIARY_TREATMENT_WEIGHT = 0.20
DBB_AUXILIARY_CLASS_WEIGHTS = (0.6666666666666666, 1.3388203017832648, 1.327891156462585)
DBB_AUXILIARY_SUPERVISION_SCOPE = "train_dbb_parents_final_causal_state_only"
HIERARCHICAL_CASCADE_SCHEMA_VERSION = "RADAR-LN3-V13"
HIERARCHICAL_LOSS_MODE = "parent_final_hierarchical_nll_with_prefix_auxiliary"
HIERARCHICAL_CHECKPOINT_SELECTION = "hierarchical_parent_route_guarded"
HIERARCHICAL_CU_OUTPUT_NAMES = (
    "CLUTTER_ACCEPT_EVIDENCE",
    "CLUTTER_REJECT_EVIDENCE",
)
DT_REASON_DYNAMICS_MODE = "v3_reason_aware_dynamics_only"
FORMAL_PAIRED_SCHEMA_VERSIONS = frozenset(
    {RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION, DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION}
)
DT_VALID_FEATURE_MODES = frozenset({"fusion_and_dynamics", "dynamics_only"})
_TASK_SCHEMA_PATTERN = re.compile(r"^RADAR-LN3-V(?P<version>\d+)$")


def _task_schema_number(schema_version: str) -> int:
    match = _TASK_SCHEMA_PATTERN.fullmatch(str(schema_version))
    if match is None:
        raise ValueError(f"invalid RADAR-LN3 task schema version: {schema_version!r}")
    return int(match.group("version"))


def _is_repaired_task_schema(schema_version: str) -> bool:
    return _task_schema_number(schema_version) >= _task_schema_number(REPAIRED_TASK_SCHEMA_VERSION)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    class_names: tuple[str, ...]
    label_to_id: Mapping[str, int]
    allowed_base_labels: frozenset[str]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def encode(self, base_label: str) -> int:
        try:
            return int(self.label_to_id[str(base_label)])
        except KeyError as exc:
            raise ValueError(f"label {base_label!r} is not valid for task {self.name}") from exc


TASK_SPECS: dict[str, TaskSpec] = {
    "dbb_cu_gate": TaskSpec(
        name="dbb_cu_gate",
        class_names=("DBB", "CU"),
        label_to_id={
            "DroneTarget": 0,
            "BirdTarget": 0,
            "BalloonTarget": 0,
            "ClutterTarget": 1,
            "UnknownTarget": 1,
        },
        allowed_base_labels=frozenset(BASE_LABELS),
    ),
    "dbb_specialist": TaskSpec(
        name="dbb_specialist",
        class_names=("DroneTarget", "BirdTarget", "BalloonTarget"),
        label_to_id={"DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2},
        allowed_base_labels=frozenset({"DroneTarget", "BirdTarget", "BalloonTarget"}),
    ),
    "cu_specialist": TaskSpec(
        name="cu_specialist",
        class_names=("ClutterTarget", "UnknownTarget"),
        label_to_id={"ClutterTarget": 0, "UnknownTarget": 1},
        allowed_base_labels=frozenset({"ClutterTarget", "UnknownTarget"}),
    ),
}


@dataclass(frozen=True)
class TaskDataConfig:
    index_path: str
    contract_path: str
    normalization_stats_path: str
    train_split_name: str
    validation_split_name: str
    test_split_name: str
    index_schema_version: str
    expected_index_sha256: str
    expected_contract_sha256: str
    expected_stats_sha256: str
    max_sequence_length: int
    max_rd_frames_per_point: int
    rd_amplitude_center: float
    rd_amplitude_scale: float
    exclude_anomalous_parents: bool
    rd_quality_schema: str | None
    contract_schema_version: str | None
    tr_motion_schema: str | None
    tr_motion_mask_mode: str | None
    rd_physical_schema: str | None
    rd_physical_mask_mode: str | None
    rd_physical_cache_manifest_path: str | None
    expected_rd_physical_cache_manifest_sha256: str | None
    train_rd_physical_cache_path: str | None
    validation_rd_physical_cache_path: str | None
    expected_train_rd_physical_cache_sha256: str | None
    expected_validation_rd_physical_cache_sha256: str | None


@dataclass(frozen=True)
class TaskModelConfig:
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
    dt_valid_dim: int
    dt_valid_feature_mode: str
    liquid_invalid_dt_update_mode: str
    liquid_invalid_dt_observation_gain: float
    num_classes: int
    liquid_state_normalization: str
    rd_frame_aggregation: str
    tr_motion_dim: int
    rd_physical_dim: int
    dbb_auxiliary_num_classes: int
    dt_reason_mode: str


@dataclass(frozen=True)
class TaskTrainConfig:
    batch_size: int
    learning_rate: float
    seed: int
    device: str
    epochs: int
    early_stopping_patience: int
    gradient_clip_norm: float
    checkpoint_selection_metric: str
    checkpoint_path: str
    last_checkpoint_path: str
    formal_artifact_dir: str
    class_weights: tuple[float, ...]
    loss_mode: str
    checkpoint_schema: str
    allow_parameter_updates: bool
    epoch_checkpoint_dir: str | None
    paired_protocol: str | None
    paired_arm: str | None
    paired_seed: int | None
    dataloader_seed: int | None
    record_parent_order_hash: bool
    save_every_epoch: bool
    cuda_deterministic_algorithms: bool
    paired_counterpart_config_path: str | None
    dbb_auxiliary_loss_weight: float
    dbb_auxiliary_class_weights: tuple[float, ...]
    dbb_auxiliary_supervision_scope: str | None
    prefix_auxiliary_weight: float
    prefix_auxiliary_min_events: int
    joint_artifact_dir: str | None
    joint_last_checkpoint_path: str | None
    joint_best_checkpoint_path: str | None
    joint_checkpoint_schema: str | None
    gate_min_precision_recall_target: float | None
    dbb_min_precision_recall_target: float | None
    cu_min_precision_recall_target: float | None
    drone_bird_min_precision_recall_target: float | None


@dataclass(frozen=True)
class LiquidTaskConfig:
    schema_version: str
    config_name: str
    task: TaskSpec
    data: TaskDataConfig
    model: TaskModelConfig
    train: TaskTrainConfig


def task_config_identity_sha256(config: LiquidTaskConfig) -> str:
    """Return a semantic identity for one task configuration.

    The identity is independent of JSON whitespace and key order.  V13 stores
    it in both task and joint checkpoints so a same-named config edited in
    place cannot be used to resume or assemble a different product run.
    """

    payload = {
        "schema_version": config.schema_version,
        "config_name": config.config_name,
        "task": {
            "name": config.task.name,
            "class_names": list(config.task.class_names),
            "label_to_id": dict(config.task.label_to_id),
            "allowed_base_labels": sorted(config.task.allowed_base_labels),
        },
        "data": asdict(config.data),
        "model": asdict(config.model),
        "train": asdict(config.train),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def task_spec(name: str) -> TaskSpec:
    try:
        return TASK_SPECS[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported RADAR-LN3 task: {name}") from exc


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"task configuration is missing {key}")
    return mapping[key]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_task_config(path: str | Path) -> LiquidTaskConfig:
    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    schema_version = str(payload.get("schema_version"))
    schema_number = _task_schema_number(schema_version)
    if schema_version != TASK_SCHEMA_VERSION and schema_number < _task_schema_number(REPAIRED_TASK_SCHEMA_VERSION):
        raise ValueError(
            f"task configuration must be historical {TASK_SCHEMA_VERSION} or a repaired version >= {REPAIRED_TASK_SCHEMA_VERSION}"
        )
    task_payload = dict(_required(payload, "task"))
    spec = task_spec(str(_required(task_payload, "name")))
    if schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION and spec.name == "cu_specialist":
        spec = replace(spec, class_names=HIERARCHICAL_CU_OUTPUT_NAMES)
    configured_classes = tuple(str(value) for value in _required(task_payload, "class_names"))
    if configured_classes != spec.class_names:
        raise ValueError(f"task class_names do not match frozen mapping for {spec.name}")
    configured_mapping = {str(key): int(value) for key, value in dict(_required(task_payload, "label_to_id")).items()}
    if configured_mapping != dict(spec.label_to_id):
        raise ValueError(f"task label_to_id does not match frozen mapping for {spec.name}")
    data_payload = dict(_required(payload, "data"))
    data = TaskDataConfig(
        index_path=str(_required(data_payload, "index_path")),
        contract_path=str(_required(data_payload, "contract_path")),
        normalization_stats_path=str(_required(data_payload, "normalization_stats_path")),
        train_split_name=str(_required(data_payload, "train_split_name")),
        validation_split_name=str(_required(data_payload, "validation_split_name")),
        test_split_name=str(_required(data_payload, "test_split_name")),
        index_schema_version=str(_required(data_payload, "index_schema_version")),
        expected_index_sha256=str(_required(data_payload, "expected_index_sha256")),
        expected_contract_sha256=str(_required(data_payload, "expected_contract_sha256")),
        expected_stats_sha256=str(_required(data_payload, "expected_stats_sha256")),
        max_sequence_length=max(1, int(_required(data_payload, "max_sequence_length"))),
        max_rd_frames_per_point=max(1, int(_required(data_payload, "max_rd_frames_per_point"))),
        rd_amplitude_center=float(_required(data_payload, "rd_amplitude_center")),
        rd_amplitude_scale=float(_required(data_payload, "rd_amplitude_scale")),
        exclude_anomalous_parents=_as_bool(_required(data_payload, "exclude_anomalous_parents")),
        rd_quality_schema=(None if data_payload.get("rd_quality_schema") is None else str(data_payload["rd_quality_schema"])),
        contract_schema_version=(
            None if data_payload.get("contract_schema_version") is None else str(data_payload["contract_schema_version"])
        ),
        tr_motion_schema=(
            None if data_payload.get("tr_motion_schema") is None else str(data_payload["tr_motion_schema"])
        ),
        tr_motion_mask_mode=(
            None if data_payload.get("tr_motion_mask_mode") is None else str(data_payload["tr_motion_mask_mode"])
        ),
        rd_physical_schema=(
            None if data_payload.get("rd_physical_schema") is None else str(data_payload["rd_physical_schema"])
        ),
        rd_physical_mask_mode=(
            None if data_payload.get("rd_physical_mask_mode") is None else str(data_payload["rd_physical_mask_mode"])
        ),
        rd_physical_cache_manifest_path=(
            None
            if data_payload.get("rd_physical_cache_manifest_path") is None
            else str(data_payload["rd_physical_cache_manifest_path"])
        ),
        expected_rd_physical_cache_manifest_sha256=(
            None
            if data_payload.get("expected_rd_physical_cache_manifest_sha256") is None
            else str(data_payload["expected_rd_physical_cache_manifest_sha256"])
        ),
        train_rd_physical_cache_path=(
            None
            if data_payload.get("train_rd_physical_cache_path") is None
            else str(data_payload["train_rd_physical_cache_path"])
        ),
        validation_rd_physical_cache_path=(
            None
            if data_payload.get("validation_rd_physical_cache_path") is None
            else str(data_payload["validation_rd_physical_cache_path"])
        ),
        expected_train_rd_physical_cache_sha256=(
            None
            if data_payload.get("expected_train_rd_physical_cache_sha256") is None
            else str(data_payload["expected_train_rd_physical_cache_sha256"])
        ),
        expected_validation_rd_physical_cache_sha256=(
            None
            if data_payload.get("expected_validation_rd_physical_cache_sha256") is None
            else str(data_payload["expected_validation_rd_physical_cache_sha256"])
        ),
    )
    if data.rd_amplitude_scale <= 0.0:
        raise ValueError("task data rd_amplitude_scale must be positive")
    model_payload = dict(_required(payload, "model"))
    if schema_number >= _task_schema_number(DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION):
        dt_valid_feature_mode = str(_required(model_payload, "dt_valid_feature_mode"))
    else:
        dt_valid_feature_mode = str(model_payload.get("dt_valid_feature_mode", "fusion_and_dynamics"))
    model = TaskModelConfig(
        tr_value_dim=int(_required(model_payload, "tr_value_dim")),
        tr_missing_dim=int(_required(model_payload, "tr_missing_dim")),
        rd_quality_dim=int(_required(model_payload, "rd_quality_dim")),
        hidden_dim=int(_required(model_payload, "hidden_dim")),
        rd_embedding_dim=int(_required(model_payload, "rd_embedding_dim")),
        liquid_hidden_dim=int(_required(model_payload, "liquid_hidden_dim")),
        rd_height=int(_required(model_payload, "rd_height")),
        rd_width=int(_required(model_payload, "rd_width")),
        liquid_tau_min_seconds=float(_required(model_payload, "liquid_tau_min_seconds")),
        liquid_tau_max_seconds=float(_required(model_payload, "liquid_tau_max_seconds")),
        liquid_zero_dt_event_floor_seconds=float(_required(model_payload, "liquid_zero_dt_event_floor_seconds")),
        liquid_dt_max_seconds=float(_required(model_payload, "liquid_dt_max_seconds")),
        dt_valid_dim=int(_required(model_payload, "dt_valid_dim")),
        dt_valid_feature_mode=dt_valid_feature_mode,
        liquid_invalid_dt_update_mode=str(_required(model_payload, "liquid_invalid_dt_update_mode")),
        liquid_invalid_dt_observation_gain=float(_required(model_payload, "liquid_invalid_dt_observation_gain")),
        num_classes=int(_required(model_payload, "num_classes")),
        liquid_state_normalization=str(model_payload.get("liquid_state_normalization", "recurrent")),
        rd_frame_aggregation=str(model_payload.get("rd_frame_aggregation", "masked_mean")),
        tr_motion_dim=int(model_payload.get("tr_motion_dim", 0)),
        rd_physical_dim=int(model_payload.get("rd_physical_dim", 0)),
        dbb_auxiliary_num_classes=int(model_payload.get("dbb_auxiliary_num_classes", 0)),
        dt_reason_mode=str(model_payload.get("dt_reason_mode", "legacy_dt_valid_only")),
    )
    if model.num_classes != spec.num_classes:
        raise ValueError(f"model.num_classes={model.num_classes} does not match task={spec.num_classes}")
    if (model.tr_value_dim, model.tr_missing_dim, model.rd_quality_dim) != (14, 11, 7):
        raise ValueError("task model must use the current TR/RD input dimensions 14/11/7")
    if (model.rd_height, model.rd_width) != (21, 384):
        raise ValueError("task model must use the current RD shape 21x384")
    if model.dt_valid_dim != 1 or model.liquid_invalid_dt_update_mode not in {"floor", "bounded_observation"}:
        raise ValueError("task model must explicitly consume dt_valid with a supported liquid update mode")
    if model.dt_valid_feature_mode not in DT_VALID_FEATURE_MODES:
        raise ValueError(
            "task model dt_valid_feature_mode must be fusion_and_dynamics or dynamics_only"
        )
    if schema_number < _task_schema_number(DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION):
        if model.dt_valid_feature_mode != "fusion_and_dynamics":
            raise ValueError("task schemas before RADAR-LN3-V7 must preserve fusion_and_dynamics")
    elif model.dt_valid_feature_mode != "dynamics_only":
        raise ValueError("task schemas from RADAR-LN3-V7 onward must use dynamics_only")
    if model.liquid_state_normalization not in {"recurrent", "readout_only"}:
        raise ValueError("task liquid_state_normalization must be recurrent or readout_only")
    if model.rd_frame_aggregation not in {"masked_mean", "causal_gru"}:
        raise ValueError("task rd_frame_aggregation must be masked_mean or causal_gru")
    if model.liquid_state_normalization != "recurrent" and model.rd_frame_aggregation != "masked_mean":
        raise ValueError("task liquid and RD temporal ablations must be run separately")
    motion_fields_present = (
        model.tr_motion_dim > 0
        or data.tr_motion_schema is not None
        or data.tr_motion_mask_mode is not None
    )
    if motion_fields_present and schema_number < _task_schema_number(TR_MOTION_GATE_SCHEMA_VERSION):
        raise ValueError("task schemas before RADAR-LN3-V9 must not enable causal TR motion inputs")
    if schema_number in {
        _task_schema_number(TR_MOTION_GATE_SCHEMA_VERSION),
        _task_schema_number(TR_OBSERVATION_INCREMENT_GATE_SCHEMA_VERSION),
    } and not motion_fields_present:
        raise ValueError(f"{schema_version} is reserved for a causal TR-feature DBB/CU Gate experiment")
    motion_schema_enabled = motion_fields_present
    if motion_schema_enabled:
        if spec.name != "dbb_cu_gate":
            raise ValueError("causal TR derived features are restricted to the DBB/CU Gate experiment")
        expected_schema = (
            TR_MOTION_SCHEMA
            if schema_number == _task_schema_number(TR_MOTION_GATE_SCHEMA_VERSION)
            else TR_OBSERVATION_INCREMENT_SCHEMA
        )
        if data.tr_motion_schema != expected_schema:
            raise ValueError(
                f"{schema_version} Gate requires causal TR schema {expected_schema!r}; "
                f"actual={data.tr_motion_schema!r}"
            )
        motion_contract = resolve_causal_tr_feature_contract(data.tr_motion_schema)
        if model.tr_motion_dim != motion_contract.value_dim:
            raise ValueError(
                f"Gate tr_motion_dim must be {motion_contract.value_dim}; actual={model.tr_motion_dim}"
            )
        if data.tr_motion_mask_mode != motion_contract.mask_mode:
            raise ValueError(
                "Gate TR-feature validity must use gating_only and must not be exposed as a classification feature"
            )
    physical_fields = (
        data.rd_physical_schema,
        data.rd_physical_mask_mode,
        data.rd_physical_cache_manifest_path,
        data.expected_rd_physical_cache_manifest_sha256,
        data.train_rd_physical_cache_path,
        data.validation_rd_physical_cache_path,
        data.expected_train_rd_physical_cache_sha256,
        data.expected_validation_rd_physical_cache_sha256,
    )
    if schema_number < _task_schema_number(RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION):
        if model.rd_physical_dim != 0 or any(value is not None for value in physical_fields):
            raise ValueError("task schemas before RADAR-LN3-V11 must not enable RD physical paired inputs")
    elif schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        if spec.name != "dbb_cu_gate" or data.tr_motion_schema != TR_OBSERVATION_INCREMENT_SCHEMA:
            raise ValueError("RADAR-LN3-V11 is restricted to the V10 observation-increment DBB/CU Gate")
        if model.rd_physical_dim not in {0, RD_PHYSICAL_VALUE_DIM}:
            raise ValueError("RADAR-LN3-V11 rd_physical_dim must be 0 for Control or 4 for Treatment")
        if model.rd_physical_dim:
            expected_physical = (
                RD_PHYSICAL_SCHEMA,
                RD_PHYSICAL_MASK_MODE,
            )
            if (data.rd_physical_schema, data.rd_physical_mask_mode) != expected_physical:
                raise ValueError("V11 Treatment RD physical schema/mask does not match the frozen contract")
            if any(value is None for value in physical_fields[2:]):
                raise ValueError("V11 Treatment requires manifest, train/val cache paths and hashes")
        elif any(value is not None for value in physical_fields):
            raise ValueError("V11 Control must not configure or load an RD physical cache")
    elif model.rd_physical_dim != 0 or any(value is not None for value in physical_fields):
        raise ValueError("task schemas after RADAR-LN3-V11 must not silently retain the V11 RD physical treatment")
    if schema_number < _task_schema_number(DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION):
        if model.dbb_auxiliary_num_classes != 0:
            raise ValueError("task schemas before RADAR-LN3-V12 must not enable the DBB auxiliary head")
    elif schema_version == DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION:
        if spec.name != "dbb_cu_gate" or data.tr_motion_schema != TR_OBSERVATION_INCREMENT_SCHEMA:
            raise ValueError("RADAR-LN3-V12 is restricted to the V10 observation-increment DBB/CU Gate")
        if model.dbb_auxiliary_num_classes != len(DBB_AUXILIARY_CLASS_NAMES):
            raise ValueError("RADAR-LN3-V12 requires a three-class Drone/Bird/Balloon auxiliary head")
    train_payload = dict(_required(payload, "train"))
    class_weights = tuple(float(value) for value in _required(train_payload, "class_weights"))
    if len(class_weights) != spec.num_classes or any(value <= 0.0 or not np.isfinite(value) for value in class_weights):
        raise ValueError("task class_weights must be finite positive values matching task classes")
    train = TaskTrainConfig(
        batch_size=max(1, int(_required(train_payload, "batch_size"))),
        learning_rate=float(_required(train_payload, "learning_rate")),
        seed=int(_required(train_payload, "seed")),
        device=str(_required(train_payload, "device")),
        epochs=max(1, int(_required(train_payload, "epochs"))),
        early_stopping_patience=max(0, int(_required(train_payload, "early_stopping_patience"))),
        gradient_clip_norm=float(_required(train_payload, "gradient_clip_norm")),
        checkpoint_selection_metric=str(_required(train_payload, "checkpoint_selection_metric")),
        checkpoint_path=str(_required(train_payload, "checkpoint_path")),
        last_checkpoint_path=str(_required(train_payload, "last_checkpoint_path")),
        formal_artifact_dir=str(_required(train_payload, "formal_artifact_dir")),
        class_weights=class_weights,
        loss_mode=str(_required(train_payload, "loss_mode")),
        checkpoint_schema=str(_required(train_payload, "checkpoint_schema")),
        allow_parameter_updates=_as_bool(_required(train_payload, "allow_parameter_updates")),
        epoch_checkpoint_dir=(
            None if train_payload.get("epoch_checkpoint_dir") is None else str(train_payload["epoch_checkpoint_dir"])
        ),
        paired_protocol=(None if train_payload.get("paired_protocol") is None else str(train_payload["paired_protocol"])),
        paired_arm=(None if train_payload.get("paired_arm") is None else str(train_payload["paired_arm"])),
        paired_seed=(None if train_payload.get("paired_seed") is None else int(train_payload["paired_seed"])),
        dataloader_seed=(None if train_payload.get("dataloader_seed") is None else int(train_payload["dataloader_seed"])),
        record_parent_order_hash=_as_bool(train_payload.get("record_parent_order_hash", False)),
        save_every_epoch=_as_bool(train_payload.get("save_every_epoch", False)),
        cuda_deterministic_algorithms=_as_bool(train_payload.get("cuda_deterministic_algorithms", False)),
        paired_counterpart_config_path=(
            None
            if train_payload.get("paired_counterpart_config_path") is None
            else str(train_payload["paired_counterpart_config_path"])
        ),
        dbb_auxiliary_loss_weight=float(train_payload.get("dbb_auxiliary_loss_weight", 0.0)),
        dbb_auxiliary_class_weights=tuple(
            float(value) for value in train_payload.get("dbb_auxiliary_class_weights", [])
        ),
        dbb_auxiliary_supervision_scope=(
            None
            if train_payload.get("dbb_auxiliary_supervision_scope") is None
            else str(train_payload["dbb_auxiliary_supervision_scope"])
        ),
        prefix_auxiliary_weight=float(train_payload.get("prefix_auxiliary_weight", 0.0)),
        prefix_auxiliary_min_events=max(1, int(train_payload.get("prefix_auxiliary_min_events", 1))),
        joint_artifact_dir=(
            None if train_payload.get("joint_artifact_dir") is None else str(train_payload["joint_artifact_dir"])
        ),
        joint_last_checkpoint_path=(
            None
            if train_payload.get("joint_last_checkpoint_path") is None
            else str(train_payload["joint_last_checkpoint_path"])
        ),
        joint_best_checkpoint_path=(
            None
            if train_payload.get("joint_best_checkpoint_path") is None
            else str(train_payload["joint_best_checkpoint_path"])
        ),
        joint_checkpoint_schema=(
            None
            if train_payload.get("joint_checkpoint_schema") is None
            else str(train_payload["joint_checkpoint_schema"])
        ),
        gate_min_precision_recall_target=(
            None
            if train_payload.get("gate_min_precision_recall_target") is None
            else float(train_payload["gate_min_precision_recall_target"])
        ),
        dbb_min_precision_recall_target=(
            None
            if train_payload.get("dbb_min_precision_recall_target") is None
            else float(train_payload["dbb_min_precision_recall_target"])
        ),
        cu_min_precision_recall_target=(
            None
            if train_payload.get("cu_min_precision_recall_target") is None
            else float(train_payload["cu_min_precision_recall_target"])
        ),
        drone_bird_min_precision_recall_target=(
            None
            if train_payload.get("drone_bird_min_precision_recall_target") is None
            else float(train_payload["drone_bird_min_precision_recall_target"])
        ),
    )
    if train.learning_rate <= 0.0 or train.gradient_clip_norm <= 0.0:
        raise ValueError("task learning_rate and gradient_clip_norm must be positive")
    if schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        if train.loss_mode != HIERARCHICAL_LOSS_MODE:
            raise ValueError(f"V13 task training must use {HIERARCHICAL_LOSS_MODE}")
        if train.checkpoint_selection_metric != HIERARCHICAL_CHECKPOINT_SELECTION:
            raise ValueError(f"V13 checkpoint selection must use {HIERARCHICAL_CHECKPOINT_SELECTION}")
        if not 0.0 <= train.prefix_auxiliary_weight <= 1.0:
            raise ValueError("V13 prefix_auxiliary_weight must be in [0,1]")
        if train.prefix_auxiliary_weight > 0.0 and train.prefix_auxiliary_min_events < 2:
            raise ValueError("V13 prefix auxiliary supervision must exclude the first evidence event")
        if model.dt_reason_mode != DT_REASON_DYNAMICS_MODE:
            raise ValueError(f"V13 model must use dt_reason_mode={DT_REASON_DYNAMICS_MODE}")
        if data.contract_schema_version != HIERARCHICAL_CASCADE_SCHEMA_VERSION:
            raise ValueError("V13 tasks must consume the V13 reason-aware data contract")
        if model.dbb_auxiliary_num_classes != 0 or model.rd_physical_dim != 0:
            raise ValueError("V13 must not silently retain V11/V12 experimental branches")
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError("V13 requires 20 fixed epochs with early stopping disabled")
        if train.epoch_checkpoint_dir is None or not train.record_parent_order_hash or not train.save_every_epoch:
            raise ValueError("V13 requires epoch checkpoints and parent-order hashes")
        if train.dataloader_seed != train.seed:
            raise ValueError("V13 model seed and DataLoader seed must match")
        if any(value is not None for value in (train.paired_protocol, train.paired_arm, train.paired_seed, train.paired_counterpart_config_path)):
            raise ValueError("V13 is one joint product run, not a paired-arm experiment")
        if any(
            value is None
            for value in (
                train.joint_artifact_dir,
                train.joint_last_checkpoint_path,
                train.joint_best_checkpoint_path,
                train.joint_checkpoint_schema,
            )
        ):
            raise ValueError("V13 requires explicit joint artifact and checkpoint fields")
        if not str(train.joint_checkpoint_schema).startswith(HIERARCHICAL_CASCADE_SCHEMA_VERSION):
            raise ValueError("V13 joint checkpoint schema must use the V13 prefix")
        targets = (
            train.gate_min_precision_recall_target,
            train.dbb_min_precision_recall_target,
            train.cu_min_precision_recall_target,
            train.drone_bird_min_precision_recall_target,
        )
        if any(value is None or not 0.0 < float(value) <= 1.0 for value in targets):
            raise ValueError("V13 product selection targets must all be in (0,1]")
    elif train.loss_mode != "causal_prefix":
        raise ValueError("historical task training must use causal_prefix loss")
    if schema_number == _task_schema_number(DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION):
        if train.checkpoint_selection_metric != "parent_macro_f1":
            raise ValueError("RADAR-LN3-V7 must preserve parent_macro_f1 checkpoint selection")
    if schema_number >= _task_schema_number(WEAK_CLASS_SELECTION_SCHEMA_VERSION) and schema_version != HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        if train.checkpoint_selection_metric != "parent_min_precision_recall":
            raise ValueError(
                "task schemas from RADAR-LN3-V8 onward must use parent_min_precision_recall checkpoint selection"
            )
    if not train.checkpoint_schema.startswith(schema_version):
        raise ValueError(f"task checkpoint schema must use the {schema_version} prefix")
    if schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        expected_arm = "treatment" if model.rd_physical_dim else "control"
        if train.paired_protocol != RD_PHYSICAL_PAIRED_PROTOCOL or train.paired_arm != expected_arm:
            raise ValueError("V11 paired protocol/arm does not match the configured model input")
        if train.paired_seed != train.seed or train.dataloader_seed != train.seed:
            raise ValueError("V11 paired seed, model seed and DataLoader seed must be identical")
        if train.seed not in {42, 43, 44}:
            raise ValueError("V11 paired seed must be one of 42, 43 or 44")
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError("V11 paired training requires exactly 20 epochs with early stopping disabled")
        if train.epoch_checkpoint_dir is None or not train.record_parent_order_hash or not train.save_every_epoch:
            raise ValueError("V11 paired training requires epoch checkpoints and parent-order hashes")
        if train.paired_counterpart_config_path is None:
            raise ValueError("V11 paired training requires an explicit counterpart config path")
        if train.cuda_deterministic_algorithms:
            raise ValueError("V11 protocol fixes initialization/sampling only and must not claim strict CUDA determinism")
    elif schema_version == DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION:
        expected_arm = "control" if train.dbb_auxiliary_loss_weight == 0.0 else "treatment"
        if train.paired_protocol != DBB_AUXILIARY_PAIRED_PROTOCOL or train.paired_arm != expected_arm:
            raise ValueError("V12 paired protocol/arm does not match the configured auxiliary loss weight")
        if expected_arm == "treatment" and train.dbb_auxiliary_loss_weight != DBB_AUXILIARY_TREATMENT_WEIGHT:
            raise ValueError(
                f"V12 Treatment auxiliary loss weight must be frozen at {DBB_AUXILIARY_TREATMENT_WEIGHT}"
            )
        if train.dbb_auxiliary_class_weights != DBB_AUXILIARY_CLASS_WEIGHTS:
            raise ValueError("V12 DBB auxiliary class weights do not match the frozen train-only contract")
        if train.dbb_auxiliary_supervision_scope != DBB_AUXILIARY_SUPERVISION_SCOPE:
            raise ValueError("V12 DBB auxiliary supervision scope must be final causal state only")
        if train.paired_seed != train.seed or train.dataloader_seed != train.seed:
            raise ValueError("V12 paired seed, model seed and DataLoader seed must be identical")
        if train.seed not in {42, 43, 44}:
            raise ValueError("V12 paired seed must be one of 42, 43 or 44")
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError("V12 paired training requires exactly 20 epochs with early stopping disabled")
        if train.epoch_checkpoint_dir is None or not train.record_parent_order_hash or not train.save_every_epoch:
            raise ValueError("V12 paired training requires epoch checkpoints and parent-order hashes")
        if train.paired_counterpart_config_path is None:
            raise ValueError("V12 paired training requires an explicit counterpart config path")
        if train.cuda_deterministic_algorithms:
            raise ValueError("V12 protocol fixes initialization/sampling only and must not claim strict CUDA determinism")
    elif (
        train.dbb_auxiliary_loss_weight != 0.0
        or train.dbb_auxiliary_class_weights
        or train.dbb_auxiliary_supervision_scope is not None
    ):
        raise ValueError("DBB auxiliary training fields are reserved for the RADAR-LN3-V12 paired experiment")
    if schema_version != HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        if (
            model.dt_reason_mode != "legacy_dt_valid_only"
            or train.prefix_auxiliary_weight != 0.0
            or train.prefix_auxiliary_min_events != 1
            or train.joint_artifact_dir is not None
            or train.joint_last_checkpoint_path is not None
            or train.joint_best_checkpoint_path is not None
            or train.joint_checkpoint_schema is not None
            or train.gate_min_precision_recall_target is not None
            or train.dbb_min_precision_recall_target is not None
            or train.cu_min_precision_recall_target is not None
            or train.drone_bird_min_precision_recall_target is not None
        ):
            raise ValueError("dt reason dynamics and prefix auxiliary weight are reserved for V13")
    if _is_repaired_task_schema(schema_version):
        require_causal_rd_quality_schema(data.rd_quality_schema)
        if data.contract_schema_version is None or not _is_repaired_task_schema(data.contract_schema_version):
            raise ValueError("repaired task configuration must declare a repaired data.contract_schema_version")
    return LiquidTaskConfig(
        schema_version=schema_version,
        config_name=str(payload.get("config_name", config_path.name)),
        task=spec,
        data=data,
        model=model,
        train=train,
    )


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate if candidate.is_absolute() else project_root / candidate


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_task_data_contract(project_root: Path, config: LiquidTaskConfig) -> dict[str, Any]:
    require_causal_rd_quality_schema(config.data.rd_quality_schema)
    index_path = resolve_project_path(project_root, config.data.index_path)
    contract_path = resolve_project_path(project_root, config.data.contract_path)
    stats_path = resolve_project_path(project_root, config.data.normalization_stats_path)
    actual_hashes = {
        "index_sha256": sha256_file(index_path),
        "contract_sha256": sha256_file(contract_path),
        "stats_sha256": sha256_file(stats_path),
    }
    expected_hashes = {
        "index_sha256": config.data.expected_index_sha256,
        "contract_sha256": config.data.expected_contract_sha256,
        "stats_sha256": config.data.expected_stats_sha256,
    }
    if actual_hashes != expected_hashes:
        raise ValueError(f"task data contract hash mismatch: expected={expected_hashes}, actual={actual_hashes}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    expected_contract_schema = config.data.contract_schema_version or "RADAR-LN3-V3"
    if contract.get("schema_version") != expected_contract_schema:
        raise ValueError(
            "task data contract schema does not match the task input semantics: "
            f"expected={expected_contract_schema}, actual={contract.get('schema_version')}"
        )
    purity = contract.get("source_purity", {})
    if purity.get("allowed_sources") != ["CQ08"] or set(purity.get("excluded_sources", [])) != {"raw", "Excel", "RD-only"}:
        raise ValueError("task data contract purity is not CQ08-only")
    feature_contract = contract.get("feature_contract", {})
    if feature_contract.get("rd_quality_schema") != RD_QUALITY_SCHEMA:
        raise ValueError("task data contract does not declare the repaired causal RD quality schema")
    if feature_contract.get("rd_quality_names") != list(RD_QUALITY_NAMES):
        raise ValueError("task data contract RD quality names do not match the repaired model input")
    if config.model.tr_motion_dim > 0:
        expected_motion_contract = resolve_causal_tr_feature_contract(
            config.data.tr_motion_schema
        ).as_dict()
        if feature_contract.get("tr_motion") != expected_motion_contract:
            raise ValueError("task data contract causal TR fields do not match the configured Gate input")
    physical_evidence: dict[str, Any] | None = None
    if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        if feature_contract.get("rd_physical") != rd_physical_contract():
            raise ValueError("V11 data contract RD physical fields do not match the frozen extractor")
        cache_contract = dict(contract.get("rd_physical_cache") or {})
        if cache_contract.get("test_excluded") is not True or int(cache_contract.get("test_rd_payload_access_count", -1)) != 0:
            raise ValueError("V11 data contract does not prove that test RD payloads are sealed")
        if config.model.rd_physical_dim:
            manifest_path = resolve_project_path(project_root, config.data.rd_physical_cache_manifest_path or "")
            train_cache_path = resolve_project_path(project_root, config.data.train_rd_physical_cache_path or "")
            val_cache_path = resolve_project_path(project_root, config.data.validation_rd_physical_cache_path or "")
            expected_physical_hashes = {
                "manifest_sha256": config.data.expected_rd_physical_cache_manifest_sha256,
                "train_cache_sha256": config.data.expected_train_rd_physical_cache_sha256,
                "validation_cache_sha256": config.data.expected_validation_rd_physical_cache_sha256,
            }
            actual_physical_hashes = {
                "manifest_sha256": sha256_file(manifest_path),
                "train_cache_sha256": sha256_file(train_cache_path),
                "validation_cache_sha256": sha256_file(val_cache_path),
            }
            if expected_physical_hashes != actual_physical_hashes:
                raise ValueError(
                    "V11 RD physical manifest/cache hash mismatch: "
                    f"expected={expected_physical_hashes}, actual={actual_physical_hashes}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("schema_version") != RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION
                or manifest.get("test_excluded") is not True
                or int(manifest.get("test_rd_payload_access_count", -1)) != 0
                or manifest.get("rd_physical_contract") != rd_physical_contract()
            ):
                raise ValueError("V11 RD physical cache manifest violates schema, extractor or test-sealing contract")
            physical_evidence = {
                "paths": {
                    "manifest": str(manifest_path),
                    "train_cache": str(train_cache_path),
                    "validation_cache": str(val_cache_path),
                },
                "hashes": actual_physical_hashes,
                "test_excluded": True,
            }
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    amplitude = dict(stats.get("amplitude", {}))
    center = float(amplitude.get("center"))
    scale = float(amplitude.get("scale"))
    if abs(center - config.data.rd_amplitude_center) > 1.0e-6 or abs(scale - config.data.rd_amplitude_scale) > 1.0e-6:
        raise ValueError("task RD normalization values do not match the recorded train statistics")
    if stats.get("source_split") != "train" or "no validation or test" not in str(stats.get("scope", "")):
        raise ValueError("task normalization statistics are not explicitly train-only")
    return {
        "paths": {"index": str(index_path), "contract": str(contract_path), "stats": str(stats_path)},
        "hashes": actual_hashes,
        "contract": contract,
        "rd_physical": physical_evidence,
    }


class TaskSequenceDataset(Dataset[PointSequence]):
    """A label-filtered view over the existing causal parent Dataset.

    The underlying Dataset still performs all TR/RD decoding and masking.  This
    view only selects parents whose original label belongs to the task and
    leaves every input tensor unchanged.
    """

    def __init__(self, base: CQ08PointSequenceDataset, spec: TaskSpec, parent_ids: Iterable[str] | None = None) -> None:
        self.base = base
        self.spec = spec
        labels_by_parent = {
            str(parent_id): str(group.iloc[0]["label_name"])
            for parent_id, group in base._groups.items()
        }
        available = [
            str(parent_id)
            for parent_id in base.parent_ids
            if labels_by_parent[str(parent_id)] in spec.allowed_base_labels
        ]
        if parent_ids is not None:
            requested = {str(value) for value in parent_ids}
            available = [parent_id for parent_id in available if parent_id in requested]
        if not available:
            raise ValueError(f"task {spec.name} has no parents in split={base.frame['split_name'].iloc[0]}")
        base_indices = {str(parent_id): index for index, parent_id in enumerate(base.parent_ids)}
        self.parent_ids = available
        self._base_indices = [base_indices[parent_id] for parent_id in available]

    def __len__(self) -> int:
        return len(self._base_indices)

    def __getitem__(self, index: int) -> PointSequence:
        return self.base[self._base_indices[int(index)]]

    @property
    def split_name(self) -> str:
        return str(self.base.frame["split_name"].iloc[0])

    def label_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.spec.class_names}
        for parent_id in self.parent_ids:
            base_label = str(self.base._groups[parent_id].iloc[0]["label_name"])
            counts[self.spec.class_names[self.spec.encode(base_label)]] += 1
        return counts


def _balanced_parent_ids(dataset: TaskSequenceDataset, limit: int | None) -> list[str] | None:
    if limit is None:
        return None
    if limit <= 0:
        raise ValueError("parent limit must be positive")
    groups: dict[int, list[str]] = {index: [] for index in range(dataset.spec.num_classes)}
    for parent_id in dataset.parent_ids:
        base_label = str(dataset.base._groups[parent_id].iloc[0]["label_name"])
        groups[dataset.spec.encode(base_label)].append(parent_id)
    selected: list[str] = []
    cursor = 0
    while len(selected) < min(limit, len(dataset.parent_ids)):
        progressed = False
        for class_id in range(dataset.spec.num_classes):
            values = groups[class_id]
            if cursor < len(values) and len(selected) < limit:
                selected.append(values[cursor])
                progressed = True
        if not progressed:
            break
        cursor += 1
    return selected


def build_task_dataset(
    project_root: Path,
    config: LiquidTaskConfig,
    *,
    split_name: str,
    parent_limit: int | None = None,
) -> TaskSequenceDataset:
    rd_physical_cache_path: str | None = None
    if config.model.rd_physical_dim:
        if split_name == config.data.train_split_name:
            rd_physical_cache_path = config.data.train_rd_physical_cache_path
        elif split_name == config.data.validation_split_name:
            rd_physical_cache_path = config.data.validation_rd_physical_cache_path
        else:
            raise PermissionError(
                "V11 RD physical Dataset is restricted to versioned train/val caches; test remains sealed"
            )
    base = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=split_name,
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=config.data.exclude_anomalous_parents,
        sample_limit=None,
        expected_schema_version=config.data.index_schema_version,
        tr_motion_schema=config.data.tr_motion_schema if config.model.tr_motion_dim > 0 else None,
        rd_physical_cache_path=rd_physical_cache_path,
    )
    view = TaskSequenceDataset(base, config.task)
    selected = _balanced_parent_ids(view, parent_limit)
    return TaskSequenceDataset(base, config.task, selected)


def collate_task_sequences(spec: TaskSpec):
    def collate(items: list[PointSequence]) -> dict[str, Any]:
        batch = collate_point_sequences(items)
        task_labels = [spec.encode(item.label_name) for item in items]
        batch["labels"] = torch.tensor(task_labels, dtype=torch.long)
        batch["task_label_names"] = [spec.class_names[value] for value in task_labels]
        if spec.name == "dbb_cu_gate":
            auxiliary_labels = [
                DBB_AUXILIARY_LABEL_TO_ID.get(item.label_name, -1)
                for item in items
            ]
            batch["dbb_auxiliary_labels"] = torch.tensor(auxiliary_labels, dtype=torch.long)
        return batch

    return collate


def build_task_dataloader(
    dataset: TaskSequenceDataset,
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
        collate_fn=collate_task_sequences(dataset.spec),
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


class PairedFusionLinear(nn.Linear):
    """A Linear layer with an explicit shared-base plus intervention path."""

    def __init__(self, in_features: int, out_features: int, *, base_input_dim: int) -> None:
        super().__init__(in_features, out_features)
        self.base_input_dim = int(base_input_dim)
        if not 0 < self.base_input_dim <= self.in_features:
            raise ValueError("paired Fusion base_input_dim is outside the layer input")

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[-1] != self.in_features:
            raise ValueError("paired Fusion received an invalid final dimension")
        base = nn.functional.linear(
            value[..., : self.base_input_dim],
            self.weight[:, : self.base_input_dim],
            self.bias,
        )
        if self.base_input_dim == self.in_features:
            return base
        intervention = nn.functional.linear(
            value[..., self.base_input_dim :],
            self.weight[:, self.base_input_dim :],
            None,
        )
        return base + intervention


class LiquidTaskClassifier(nn.Module):
    """Current RD spatial encoder + CfC liquid encoder + task-specific head."""

    def __init__(self, config: LiquidTaskConfig) -> None:
        super().__init__()
        if not _is_repaired_task_schema(config.schema_version):
            raise ValueError(
                f"task model execution requires a schema >= {REPAIRED_TASK_SCHEMA_VERSION}; "
                f"legacy configuration {config.schema_version} is audit-only"
            )
        require_causal_rd_quality_schema(config.data.rd_quality_schema)
        self.config = config
        model = config.model
        schema_number = _task_schema_number(config.schema_version)
        if schema_number < _task_schema_number(DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION):
            if model.dt_valid_feature_mode != "fusion_and_dynamics":
                raise ValueError("task schemas before RADAR-LN3-V7 must preserve fusion_and_dynamics")
        elif model.dt_valid_feature_mode != "dynamics_only":
            raise ValueError("task schemas from RADAR-LN3-V7 onward must use dynamics_only")
        self.task_name = config.task.name
        self.class_names = config.task.class_names
        self.num_classes = config.task.num_classes
        self.tr_value_dim = model.tr_value_dim
        self.tr_missing_dim = model.tr_missing_dim
        self.rd_quality_dim = model.rd_quality_dim
        self.dt_valid_dim = model.dt_valid_dim
        self.dt_valid_feature_mode = model.dt_valid_feature_mode
        self.tr_motion_dim = model.tr_motion_dim
        self.tr_motion_schema = config.data.tr_motion_schema
        self.tr_motion_mask_mode = config.data.tr_motion_mask_mode
        self.rd_physical_dim = model.rd_physical_dim
        self.dbb_auxiliary_num_classes = model.dbb_auxiliary_num_classes
        self.dt_reason_mode = model.dt_reason_mode
        self.rd_physical_schema = config.data.rd_physical_schema
        self.rd_physical_mask_mode = config.data.rd_physical_mask_mode
        self.tr_motion_contract = (
            resolve_causal_tr_feature_contract(self.tr_motion_schema)
            if self.tr_motion_dim
            else None
        )
        self.rd_quality_schema = config.data.rd_quality_schema
        self.rd_frame_aggregation = model.rd_frame_aggregation
        self.liquid_state_normalization = model.liquid_state_normalization
        if self.rd_frame_aggregation != "masked_mean" and self.liquid_state_normalization != "recurrent":
            raise ValueError("only one liquid/RD temporal ablation may be enabled at a time")
        self.rd_encoder = RDSpatialEncoder(model.rd_embedding_dim, aggregation_mode=model.rd_frame_aggregation)
        self.tr_encoder = nn.Sequential(
            nn.Linear(model.tr_value_dim + model.tr_missing_dim, model.hidden_dim),
            nn.LayerNorm(model.hidden_dim),
            nn.SiLU(),
            nn.Linear(model.hidden_dim, model.hidden_dim),
            nn.SiLU(),
        )
        self.motion_encoder: nn.Module | None = None
        if self.tr_motion_dim:
            if self.tr_motion_contract is None or self.tr_motion_dim != self.tr_motion_contract.value_dim:
                raise ValueError("task causal TR feature contract does not match model dimension")
            if self.tr_motion_mask_mode != self.tr_motion_contract.mask_mode:
                raise ValueError("task motion mask must use gating_only")
            self.motion_encoder = nn.Sequential(
                nn.Linear(self.tr_motion_dim, model.hidden_dim, bias=False),
                nn.LayerNorm(model.hidden_dim, elementwise_affine=False),
                nn.SiLU(),
                nn.Linear(model.hidden_dim, model.hidden_dim, bias=False),
                nn.SiLU(),
            )
        fusion_dt_valid_dim = model.dt_valid_dim if model.dt_valid_feature_mode == "fusion_and_dynamics" else 0
        fusion_motion_dim = model.hidden_dim if self.motion_encoder is not None else 0
        fusion_dim = (
            model.hidden_dim
            + fusion_motion_dim
            + model.rd_embedding_dim
            + model.rd_quality_dim
            + fusion_dt_valid_dim
            + self.rd_physical_dim
        )
        self.fusion_input_dim = fusion_dim
        fusion_linear: nn.Module
        if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
            fusion_linear = PairedFusionLinear(
                fusion_dim,
                model.liquid_hidden_dim,
                base_input_dim=fusion_dim - self.rd_physical_dim,
            )
        else:
            fusion_linear = nn.Linear(fusion_dim, model.liquid_hidden_dim)
        self.fusion = nn.Sequential(
            fusion_linear,
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
            nn.Linear(model.liquid_hidden_dim, self.num_classes),
        )
        self.dbb_auxiliary_classifier: nn.Module | None = None
        if self.dbb_auxiliary_num_classes:
            if self.task_name != "dbb_cu_gate" or self.dbb_auxiliary_num_classes != len(DBB_AUXILIARY_CLASS_NAMES):
                raise ValueError("DBB auxiliary head is restricted to the three-subclass DBB/CU Gate contract")
            self.dbb_auxiliary_classifier = nn.Sequential(
                nn.LayerNorm(model.liquid_hidden_dim),
                nn.Linear(model.liquid_hidden_dim, self.dbb_auxiliary_num_classes),
            )

    def prepare_online_rd_physical(
        self,
        amplitude_db: np.ndarray,
        velocity_mps: np.ndarray,
        *,
        frame_mask: np.ndarray | Iterable[bool] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute the Treatment input from one raw canonical RD event.

        This is the online counterpart of the versioned cache builder and calls
        the same extractor. The caller must still provide the existing
        normalized amplitude/phase ``rd_frames`` to the spatial RD encoder.
        """

        if self.rd_physical_dim != RD_PHYSICAL_VALUE_DIM:
            raise ValueError("online RD physical extraction is available only for the V11 Treatment")
        amplitude = np.asarray(amplitude_db)
        if amplitude.ndim != 3:
            raise ValueError("online RD physical amplitude must have shape [frames,21,384]")
        if amplitude.shape[0] > self.config.data.max_rd_frames_per_point:
            raise ValueError(
                "online RD physical input exceeds the frozen per-event frame cap; "
                f"maximum={self.config.data.max_rd_frames_per_point}, actual={amplitude.shape[0]}"
            )
        if frame_mask is not None:
            active = np.asarray(
                list(frame_mask) if not isinstance(frame_mask, np.ndarray) else frame_mask,
                dtype=bool,
            )
            if int(active.sum()) > self.config.data.max_rd_frames_per_point:
                raise ValueError("online RD physical valid-frame count exceeds the frozen per-event frame cap")
        event = extract_current_event_rd_physical(amplitude_db, velocity_mps, frame_mask=frame_mask)
        parameter = self.fusion[0].weight
        return {
            "rd_physical_features": torch.as_tensor(
                event.values.copy(),
                dtype=parameter.dtype,
                device=parameter.device,
            ).unsqueeze(0),
            "rd_physical_valid": torch.tensor([event.valid], dtype=torch.bool, device=parameter.device),
        }

    def forward(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        tr_padding_mask: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor | None = None,
        tr_motion_features: torch.Tensor | None = None,
        tr_motion_valid_mask: torch.Tensor | None = None,
        rd_physical_features: torch.Tensor | None = None,
        rd_physical_valid: torch.Tensor | None = None,
        return_dbb_auxiliary_logits: bool = False,
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 3 or tr_missing_mask.shape[:2] != tr_features.shape[:2]:
            raise ValueError("task TR tensors must have shape [batch, sequence, features]")
        if tr_features.shape[-1] != self.tr_value_dim or tr_missing_mask.shape[-1] != self.tr_missing_dim:
            raise ValueError("task TR feature dimensions do not match model")
        if tr_dt_seconds.shape != tr_features.shape[:2] or tr_dt_valid.shape != tr_features.shape[:2]:
            raise ValueError("task dt and dt_valid shapes do not match TR sequence")
        if self.dt_reason_mode == DT_REASON_DYNAMICS_MODE:
            if tr_dt_reason is None or tr_dt_reason.shape != tr_features.shape[:2]:
                raise ValueError("V13 task requires tr_dt_reason with shape [batch,sequence]")
        elif tr_dt_reason is not None:
            tr_dt_reason = None
        if rd_quality.shape != (*tr_features.shape[:2], self.rd_quality_dim):
            raise ValueError("task RD quality shape does not match model")
        if tr_padding_mask.shape != tr_features.shape[:2]:
            raise ValueError("task padding mask shape does not match TR sequence")
        if not all(torch.isfinite(value).all() for value in (tr_features, tr_missing_mask, tr_dt_seconds, rd_frames, rd_quality)):
            raise ValueError("task model received a non-finite input")
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        rd_embedding = self.rd_encoder(rd_frames, rd_frame_mask)
        fusion_inputs = [tr_embedding]
        if self.motion_encoder is not None:
            if tr_motion_features is None or tr_motion_valid_mask is None:
                raise ValueError("Gate requires causal TR motion/derived values and valid mask")
            expected_motion_shape = (*tr_features.shape[:2], self.tr_motion_dim)
            if tr_motion_features.shape != expected_motion_shape or tr_motion_valid_mask.shape != expected_motion_shape:
                raise ValueError("Gate causal TR value/mask shape does not match model")
            if tr_motion_valid_mask.dtype != torch.bool:
                raise ValueError("Gate causal TR valid mask must be boolean")
            if not torch.isfinite(tr_motion_features).all():
                raise ValueError("Gate received a non-finite causal TR value")
            if self.tr_motion_contract is None:
                raise RuntimeError("Gate causal TR feature contract was not initialized")
            if bool((tr_motion_features.abs() > self.tr_motion_contract.clip_absolute + 1.0e-6).any()):
                raise ValueError("Gate received a causal TR value outside the frozen clip")
            if bool((tr_motion_features.masked_select(~tr_motion_valid_mask) != 0).any()):
                raise ValueError("invalid Gate causal TR values must be exactly zero")
            if bool(tr_motion_valid_mask.masked_select(tr_padding_mask.unsqueeze(-1)).any()):
                raise ValueError("padded Gate events must not contain valid causal TR values")
            gated_motion = tr_motion_features * tr_motion_valid_mask.to(dtype=tr_motion_features.dtype)
            fusion_inputs.append(self.motion_encoder(gated_motion))
        elif tr_motion_features is not None or tr_motion_valid_mask is not None:
            if tr_motion_features is None or tr_motion_valid_mask is None:
                raise ValueError("legacy task received an incomplete optional TR motion pair")
        fusion_inputs.extend([rd_embedding, rd_quality])
        if self.rd_physical_dim:
            expected_physical_shape = (*tr_features.shape[:2], self.rd_physical_dim)
            if rd_physical_features is None or rd_physical_valid is None:
                raise ValueError("V11 Treatment requires RD physical values and valid mask")
            if rd_physical_features.shape != expected_physical_shape or rd_physical_valid.shape != tr_features.shape[:2]:
                raise ValueError("V11 RD physical value/valid shape does not match the sequence")
            if rd_physical_valid.dtype != torch.bool or not torch.isfinite(rd_physical_features).all():
                raise ValueError("V11 RD physical values must be finite and valid must be boolean")
            if bool((rd_physical_features < 0.0).any()) or bool((rd_physical_features > 1.0).any()):
                raise ValueError("V11 RD physical values must be bounded to [0,1]")
            if bool((rd_physical_features.masked_select(~rd_physical_valid.unsqueeze(-1)) != 0).any()):
                raise ValueError("invalid V11 RD physical values must be exactly zero")
            if bool(rd_physical_valid.masked_select(tr_padding_mask).any()):
                raise ValueError("padded V11 events must not contain valid RD physical values")
            gated_physical = rd_physical_features * rd_physical_valid.to(dtype=tr_features.dtype).unsqueeze(-1)
            fusion_inputs.append(gated_physical)
        elif rd_physical_features is not None or rd_physical_valid is not None:
            raise ValueError("V11 Control and historical tasks must not receive RD physical tensors")
        if self.dt_valid_feature_mode == "fusion_and_dynamics":
            fusion_inputs.append(tr_dt_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
        point_events = self.fusion(torch.cat(fusion_inputs, dim=-1))
        liquid_state = self.liquid(
            point_events,
            tr_dt_seconds,
            tr_padding_mask,
            dt_valid=tr_dt_valid,
            dt_reason=tr_dt_reason,
        )
        sequence_logits = self.classifier(liquid_state.sequence)
        valid = ~tr_padding_mask
        last_indices = valid.long().sum(dim=1).clamp_min(1) - 1
        batch_indices = torch.arange(tr_features.shape[0], device=tr_features.device)
        window_logits = sequence_logits[batch_indices, last_indices]
        result = {
            "window_logits": window_logits,
            "sequence_logits": sequence_logits,
            "liquid_sequence": liquid_state.sequence,
            "liquid_final_state": liquid_state.final_state,
            "dt_clamped_mask": liquid_state.dt_clamped_mask,
            "dt_zero_mask": liquid_state.dt_zero_mask,
            "dt_invalid_mask": liquid_state.dt_invalid_mask,
            "dt_first_event_mask": liquid_state.dt_first_event_mask,
            "dt_zero_delta_mask": liquid_state.dt_zero_delta_mask,
        }
        if return_dbb_auxiliary_logits and self.dbb_auxiliary_classifier is None:
            raise ValueError("DBB auxiliary logits were requested from a model without the V12 auxiliary head")
        if return_dbb_auxiliary_logits and self.dbb_auxiliary_classifier is not None:
            result["dbb_auxiliary_sequence_logits"] = self.dbb_auxiliary_classifier(liquid_state.sequence)
        return result

    def forward_online_point(
        self,
        *,
        tr_features: torch.Tensor,
        tr_missing_mask: torch.Tensor,
        tr_dt_seconds: torch.Tensor,
        tr_dt_valid: torch.Tensor,
        tr_dt_reason: torch.Tensor | None = None,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        liquid_state: torch.Tensor | None = None,
        tr_motion_features: torch.Tensor | None = None,
        tr_motion_valid_mask: torch.Tensor | None = None,
        rd_physical_features: torch.Tensor | None = None,
        rd_physical_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 2 or tr_missing_mask.ndim != 2 or rd_quality.ndim != 2:
            raise ValueError("online task TR tensors must have shape [batch, features]")
        if tr_features.shape[-1] != self.tr_value_dim or tr_missing_mask.shape[-1] != self.tr_missing_dim:
            raise ValueError("online task TR feature dimensions do not match model")
        if tr_dt_seconds.shape != (tr_features.shape[0],) or tr_dt_valid.shape != (tr_features.shape[0],):
            raise ValueError("online task dt tensors must have shape [batch]")
        if self.dt_reason_mode == DT_REASON_DYNAMICS_MODE:
            if tr_dt_reason is None or tr_dt_reason.shape != tr_dt_seconds.shape:
                raise ValueError("online V13 task requires tr_dt_reason with shape [batch]")
        elif tr_dt_reason is not None:
            tr_dt_reason = None
        if rd_frames.ndim != 5 or rd_frames.shape[2:] != (2, 21, 384):
            raise ValueError("online task RD frames must have shape [batch, K, 2, 21, 384]")
        if rd_frame_mask.shape != rd_frames.shape[:2]:
            raise ValueError("online task RD frame mask must have shape [batch, K]")
        if rd_quality.shape != (tr_features.shape[0], self.rd_quality_dim):
            raise ValueError("online task RD quality shape does not match model")
        if not all(torch.isfinite(value).all() for value in (tr_features, tr_missing_mask, tr_dt_seconds, rd_frames, rd_quality)):
            raise ValueError("online task model received a non-finite input")
        tr_embedding = self.tr_encoder(torch.cat([tr_features, tr_missing_mask], dim=-1))
        rd_embedding = self.rd_encoder(rd_frames.unsqueeze(1), rd_frame_mask.unsqueeze(1)).squeeze(1)
        fusion_inputs = [tr_embedding]
        if self.motion_encoder is not None:
            expected_motion_shape = (tr_features.shape[0], self.tr_motion_dim)
            if tr_motion_features is None or tr_motion_valid_mask is None:
                raise ValueError("online Gate requires causal TR motion/derived values and valid mask")
            if tr_motion_features.shape != expected_motion_shape or tr_motion_valid_mask.shape != expected_motion_shape:
                raise ValueError("online Gate causal TR value/mask shape does not match model")
            if tr_motion_valid_mask.dtype != torch.bool:
                raise ValueError("online Gate causal TR valid mask must be boolean")
            if not torch.isfinite(tr_motion_features).all():
                raise ValueError("online Gate received a non-finite causal TR value")
            if self.tr_motion_contract is None:
                raise RuntimeError("online Gate causal TR feature contract was not initialized")
            if bool((tr_motion_features.abs() > self.tr_motion_contract.clip_absolute + 1.0e-6).any()):
                raise ValueError("online Gate received a causal TR value outside the frozen clip")
            if bool((tr_motion_features.masked_select(~tr_motion_valid_mask) != 0).any()):
                raise ValueError("invalid online Gate causal TR values must be exactly zero")
            gated_motion = tr_motion_features * tr_motion_valid_mask.to(dtype=tr_motion_features.dtype)
            fusion_inputs.append(self.motion_encoder(gated_motion))
        elif tr_motion_features is not None or tr_motion_valid_mask is not None:
            if tr_motion_features is None or tr_motion_valid_mask is None:
                raise ValueError("legacy online task received an incomplete optional TR motion pair")
        fusion_inputs.extend([rd_embedding, rd_quality])
        if self.rd_physical_dim:
            if rd_physical_features is None or rd_physical_valid is None:
                raise ValueError("online V11 Treatment requires RD physical values and valid mask")
            if rd_physical_features.shape != (tr_features.shape[0], self.rd_physical_dim):
                raise ValueError("online V11 RD physical values have an invalid shape")
            if rd_physical_valid.shape != (tr_features.shape[0],) or rd_physical_valid.dtype != torch.bool:
                raise ValueError("online V11 RD physical valid must be boolean with shape [batch]")
            if not torch.isfinite(rd_physical_features).all():
                raise ValueError("online V11 RD physical values must be finite")
            if bool((rd_physical_features < 0.0).any()) or bool((rd_physical_features > 1.0).any()):
                raise ValueError("online V11 RD physical values must be bounded to [0,1]")
            if bool((rd_physical_features.masked_select(~rd_physical_valid.unsqueeze(-1)) != 0).any()):
                raise ValueError("invalid online V11 RD physical values must be exactly zero")
            fusion_inputs.append(rd_physical_features * rd_physical_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
        elif rd_physical_features is not None or rd_physical_valid is not None:
            raise ValueError("online V11 Control and historical tasks must not receive RD physical tensors")
        if self.dt_valid_feature_mode == "fusion_and_dynamics":
            fusion_inputs.append(tr_dt_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
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


def task_prefix_loss(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if sequence_logits.ndim != 3:
        raise ValueError("task sequence_logits must have shape [batch, sequence, classes]")
    if labels.shape != (sequence_logits.shape[0],) or padding_mask.shape != sequence_logits.shape[:2]:
        raise ValueError("task labels or padding mask shape mismatch")
    class_count = sequence_logits.shape[-1]
    if class_weights is not None:
        if class_weights.shape != (class_count,) or not torch.isfinite(class_weights).all() or bool((class_weights <= 0).any()):
            raise ValueError("task class_weights shape or values are invalid")
        class_weights = class_weights.to(device=sequence_logits.device, dtype=sequence_logits.dtype)
    expanded = labels.to(device=sequence_logits.device, dtype=torch.long).unsqueeze(1).expand(-1, sequence_logits.shape[1])
    losses = nn.functional.cross_entropy(
        sequence_logits.reshape(-1, class_count),
        expanded.reshape(-1),
        weight=class_weights,
        reduction="none",
    ).reshape(sequence_logits.shape[:2])
    valid = (~padding_mask).to(dtype=losses.dtype)
    counts = valid.sum(dim=1)
    if bool((counts <= 0).any()):
        raise ValueError("task prefix loss received an empty sequence")
    value = ((losses * valid).sum(dim=1) / counts).mean()
    if not torch.isfinite(value).item():
        raise ValueError("task prefix loss is non-finite")
    return value


def dbb_auxiliary_final_state_loss(
    sequence_logits: torch.Tensor,
    auxiliary_labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    class_weights: Iterable[float] = DBB_AUXILIARY_CLASS_WEIGHTS,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Apply DBB subclass CE to each selected parent's final causal state.

    ``auxiliary_labels`` uses ``0..2`` for Drone/Bird/Balloon and ``-1`` for
    Clutter/Unknown.  Early prefixes are intentionally excluded because their
    evidence need not support the complete-parent subclass label.
    """

    if sequence_logits.ndim != 3 or sequence_logits.shape[-1] != len(DBB_AUXILIARY_CLASS_NAMES):
        raise ValueError("DBB auxiliary logits must have shape [batch, sequence, 3]")
    if auxiliary_labels.shape != (sequence_logits.shape[0],) or padding_mask.shape != sequence_logits.shape[:2]:
        raise ValueError("DBB auxiliary labels or padding mask shape mismatch")
    labels = auxiliary_labels.to(device=sequence_logits.device, dtype=torch.long)
    if bool(((labels < -1) | (labels >= len(DBB_AUXILIARY_CLASS_NAMES))).any()):
        raise ValueError("DBB auxiliary labels must use -1 for CU or 0..2 for DBB subclasses")
    selected = labels >= 0
    selected_parent_count = int(selected.sum().item())
    if selected_parent_count == 0:
        zero = sequence_logits.new_zeros(())
        return zero, {
            "selected_dbb_parent_count": 0,
            "excluded_cu_parent_count": int(labels.numel()),
            "selected_final_state_count": 0,
        }
    selected_logits = sequence_logits[selected]
    selected_padding = padding_mask.to(device=sequence_logits.device)[selected]
    selected_labels = labels[selected]
    lengths = (~selected_padding).long().sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("DBB auxiliary loss received an empty selected DBB sequence")
    final_indices = lengths - 1
    batch_indices = torch.arange(selected_logits.shape[0], device=selected_logits.device)
    final_logits = selected_logits[batch_indices, final_indices]
    weight_tensor = sequence_logits.new_tensor(tuple(float(value) for value in class_weights))
    if (
        weight_tensor.shape != (len(DBB_AUXILIARY_CLASS_NAMES),)
        or not torch.isfinite(weight_tensor).all()
        or bool((weight_tensor <= 0.0).any())
    ):
        raise ValueError("DBB auxiliary class weights must contain three finite positive values")
    value = nn.functional.cross_entropy(final_logits, selected_labels, weight=weight_tensor)
    if not torch.isfinite(value).item():
        raise ValueError("DBB auxiliary prefix loss is non-finite")
    return value, {
        "selected_dbb_parent_count": selected_parent_count,
        "excluded_cu_parent_count": int((~selected).sum().item()),
        "selected_final_state_count": selected_parent_count,
    }


def task_training_objective(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    config: LiquidTaskConfig,
    *,
    class_weights: torch.Tensor,
) -> dict[str, Any]:
    """Compose the frozen main Gate and optional V12 auxiliary objectives."""

    main_loss = task_prefix_loss(
        output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=class_weights,
    )
    auxiliary_loss = main_loss.new_zeros(())
    auxiliary_counts = {
        "selected_dbb_parent_count": 0,
        "excluded_cu_parent_count": 0,
        "selected_final_state_count": 0,
    }
    if config.schema_version == DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION:
        if "dbb_auxiliary_sequence_logits" not in output or "dbb_auxiliary_labels" not in batch:
            raise ValueError("V12 training objective requires auxiliary logits and labels")
        if config.train.dbb_auxiliary_loss_weight > 0.0:
            auxiliary_loss, auxiliary_counts = dbb_auxiliary_final_state_loss(
                output["dbb_auxiliary_sequence_logits"],
                batch["dbb_auxiliary_labels"],
                batch["tr_padding_mask"],
                class_weights=config.train.dbb_auxiliary_class_weights,
            )
        else:
            with torch.no_grad():
                auxiliary_loss, auxiliary_counts = dbb_auxiliary_final_state_loss(
                    output["dbb_auxiliary_sequence_logits"],
                    batch["dbb_auxiliary_labels"],
                    batch["tr_padding_mask"],
                    class_weights=config.train.dbb_auxiliary_class_weights,
                )
    total_loss = (
        main_loss + config.train.dbb_auxiliary_loss_weight * auxiliary_loss
        if config.train.dbb_auxiliary_loss_weight > 0.0
        else main_loss
    )
    if not all(torch.isfinite(value).item() for value in (total_loss, main_loss, auxiliary_loss)):
        raise ValueError("task training objective produced a non-finite loss")
    return {
        "loss": total_loss,
        "main_loss": main_loss,
        "dbb_auxiliary_loss": auxiliary_loss,
        "dbb_auxiliary_counts": auxiliary_counts,
    }


def build_v11_paired_models(
    control_config: LiquidTaskConfig,
    treatment_config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    """Build a V11 pair with exactly shared initialization and zero new columns."""

    if (
        control_config.schema_version != RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION
        or treatment_config.schema_version != RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION
        or control_config.train.paired_arm != "control"
        or treatment_config.train.paired_arm != "treatment"
        or control_config.train.paired_protocol != RD_PHYSICAL_PAIRED_PROTOCOL
        or treatment_config.train.paired_protocol != RD_PHYSICAL_PAIRED_PROTOCOL
    ):
        raise ValueError("paired model builder requires V11 Control and Treatment configs")
    if control_config.train.seed != treatment_config.train.seed:
        raise ValueError("V11 paired configs must use the same seed")
    if control_config.model.rd_physical_dim != 0 or treatment_config.model.rd_physical_dim != RD_PHYSICAL_VALUE_DIM:
        raise ValueError("V11 paired configs must use Control dim 0 and Treatment dim 4")
    if replace(control_config.model, rd_physical_dim=RD_PHYSICAL_VALUE_DIM) != treatment_config.model:
        raise ValueError("V11 paired model configs differ beyond rd_physical_dim")
    excluded_data = {
        "rd_physical_schema",
        "rd_physical_mask_mode",
        "rd_physical_cache_manifest_path",
        "expected_rd_physical_cache_manifest_sha256",
        "train_rd_physical_cache_path",
        "validation_rd_physical_cache_path",
        "expected_train_rd_physical_cache_sha256",
        "expected_validation_rd_physical_cache_sha256",
    }
    control_data = {key: value for key, value in control_config.data.__dict__.items() if key not in excluded_data}
    treatment_data = {key: value for key, value in treatment_config.data.__dict__.items() if key not in excluded_data}
    if control_data != treatment_data:
        raise ValueError("V11 paired data configs differ beyond the RD physical cache fields")
    excluded_train = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "paired_arm",
        "paired_counterpart_config_path",
    }
    control_train = {key: value for key, value in control_config.train.__dict__.items() if key not in excluded_train}
    treatment_train = {key: value for key, value in treatment_config.train.__dict__.items() if key not in excluded_train}
    if control_train != treatment_train:
        raise ValueError("V11 paired training configs differ beyond arm-specific output/counterpart paths")
    seed_everything(control_config.train.seed)
    control = LiquidTaskClassifier(control_config)
    seed_everything(control_config.train.seed)
    treatment = LiquidTaskClassifier(treatment_config)
    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    shared_names: list[str] = []
    for name, control_value in control_state.items():
        if name == "fusion.0.weight":
            treatment_value = treatment_state[name]
            if treatment_value.shape[0] != control_value.shape[0] or treatment_value.shape[1] != control_value.shape[1] + RD_PHYSICAL_VALUE_DIM:
                raise ValueError("V11 Treatment Fusion width is not Control width plus four")
            treatment_value[:, : control_value.shape[1]].copy_(control_value)
            treatment_value[:, control_value.shape[1] :].zero_()
            shared_names.append(name)
            continue
        treatment_value = treatment_state.get(name)
        if treatment_value is None or treatment_value.shape != control_value.shape:
            raise ValueError(f"V11 paired model has a non-shared parameter/buffer outside Fusion columns: {name}")
        treatment_value.copy_(control_value)
        shared_names.append(name)
    treatment.load_state_dict(treatment_state, strict=True)
    shared_equal = True
    for name, control_value in control.state_dict().items():
        treatment_value = treatment.state_dict()[name]
        comparable = treatment_value[:, : control_value.shape[1]] if name == "fusion.0.weight" else treatment_value
        if not torch.equal(control_value, comparable):
            shared_equal = False
            break
    new_columns = treatment.state_dict()["fusion.0.weight"][:, control.fusion_input_dim :]
    if not shared_equal or not torch.equal(new_columns, torch.zeros_like(new_columns)):
        raise AssertionError("V11 shared initialization or zero-column contract failed")
    return control, treatment, {
        "paired_protocol": RD_PHYSICAL_PAIRED_PROTOCOL,
        "seed": control_config.train.seed,
        "shared_state_entry_count": len(shared_names),
        "shared_parameters_and_buffers_exactly_equal": True,
        "control_fusion_input_dim": control.fusion_input_dim,
        "treatment_fusion_input_dim": treatment.fusion_input_dim,
        "treatment_new_fusion_columns": RD_PHYSICAL_VALUE_DIM,
        "treatment_new_fusion_columns_exactly_zero": True,
    }


def build_v11_paired_model_for_config(
    project_root: Path,
    config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier | None, dict[str, Any]]:
    if config.schema_version != RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        return LiquidTaskClassifier(config), None, {
            "paired_protocol": None,
            "paired_arm": None,
            "initialization": "single_config_seeded_initialization",
        }
    counterpart_path = resolve_project_path(project_root, config.train.paired_counterpart_config_path or "")
    counterpart = load_task_config(counterpart_path)
    if counterpart.train.paired_counterpart_config_path is None:
        raise ValueError("V11 counterpart does not reference this config")
    reverse_path = resolve_project_path(project_root, counterpart.train.paired_counterpart_config_path)
    expected_current_path = resolve_project_path(
        project_root,
        f"config/RADAR-LN3/{config.config_name}-config.json",
    )
    if reverse_path.resolve() != expected_current_path.resolve():
        raise ValueError(
            "V11 counterpart reference is not reciprocal: "
            f"expected={expected_current_path}, actual={reverse_path}"
        )
    configs = {config.train.paired_arm: config, counterpart.train.paired_arm: counterpart}
    if set(configs) != {"control", "treatment"}:
        raise ValueError("V11 config/counterpart pair does not contain exactly Control and Treatment")
    control, treatment, evidence = build_v11_paired_models(configs["control"], configs["treatment"])
    evidence.update(
        {
            "paired_arm": config.train.paired_arm,
            "counterpart_config_path": str(counterpart_path),
            "counterpart_config_name": counterpart.config_name,
        }
    )
    if config.train.paired_arm == "control":
        return control, treatment, evidence
    return treatment, control, evidence


def build_v12_paired_models(
    control_config: LiquidTaskConfig,
    treatment_config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    """Build structurally identical V12 models; only the loss weight differs."""

    if (
        control_config.schema_version != DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION
        or treatment_config.schema_version != DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION
        or control_config.train.paired_arm != "control"
        or treatment_config.train.paired_arm != "treatment"
        or control_config.train.paired_protocol != DBB_AUXILIARY_PAIRED_PROTOCOL
        or treatment_config.train.paired_protocol != DBB_AUXILIARY_PAIRED_PROTOCOL
    ):
        raise ValueError("paired model builder requires V12 Control and Treatment configs")
    if control_config.train.seed != treatment_config.train.seed:
        raise ValueError("V12 paired configs must use the same seed")
    if control_config.model != treatment_config.model or control_config.data != treatment_config.data:
        raise ValueError("V12 paired model/data configs must be identical")
    excluded_train = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "paired_arm",
        "paired_counterpart_config_path",
        "dbb_auxiliary_loss_weight",
    }
    control_train = {key: value for key, value in control_config.train.__dict__.items() if key not in excluded_train}
    treatment_train = {key: value for key, value in treatment_config.train.__dict__.items() if key not in excluded_train}
    if control_train != treatment_train:
        raise ValueError("V12 paired training configs differ beyond arm paths and auxiliary loss weight")
    if control_config.train.dbb_auxiliary_loss_weight != 0.0:
        raise ValueError("V12 Control auxiliary loss weight must be zero")
    if treatment_config.train.dbb_auxiliary_loss_weight != DBB_AUXILIARY_TREATMENT_WEIGHT:
        raise ValueError("V12 Treatment auxiliary loss weight does not match the frozen intervention")
    seed_everything(control_config.train.seed)
    control = LiquidTaskClassifier(control_config)
    seed_everything(treatment_config.train.seed)
    treatment = LiquidTaskClassifier(treatment_config)
    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    if set(control_state) != set(treatment_state):
        raise ValueError("V12 paired models do not have identical state entries")
    unequal = [name for name in control_state if not torch.equal(control_state[name], treatment_state[name])]
    if unequal:
        raise AssertionError(f"V12 paired initialization differs: {unequal[:3]}")
    return control, treatment, {
        "paired_protocol": DBB_AUXILIARY_PAIRED_PROTOCOL,
        "seed": control_config.train.seed,
        "shared_state_entry_count": len(control_state),
        "shared_parameters_and_buffers_exactly_equal": True,
        "model_structures_identical": True,
        "control_auxiliary_loss_weight": 0.0,
        "treatment_auxiliary_loss_weight": DBB_AUXILIARY_TREATMENT_WEIGHT,
        "auxiliary_class_names": list(DBB_AUXILIARY_CLASS_NAMES),
    }


def build_paired_model_for_config(
    project_root: Path,
    config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier | None, dict[str, Any]]:
    if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        return build_v11_paired_model_for_config(project_root, config)
    if config.schema_version != DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION:
        return LiquidTaskClassifier(config), None, {
            "paired_protocol": None,
            "paired_arm": None,
            "initialization": "single_config_seeded_initialization",
        }
    counterpart_path = resolve_project_path(project_root, config.train.paired_counterpart_config_path or "")
    counterpart = load_task_config(counterpart_path)
    if counterpart.train.paired_counterpart_config_path is None:
        raise ValueError("V12 counterpart does not reference this config")
    reverse_path = resolve_project_path(project_root, counterpart.train.paired_counterpart_config_path)
    expected_current_path = resolve_project_path(
        project_root,
        f"config/RADAR-LN3/{config.config_name}-config.json",
    )
    if reverse_path.resolve() != expected_current_path.resolve():
        raise ValueError(
            "V12 counterpart reference is not reciprocal: "
            f"expected={expected_current_path}, actual={reverse_path}"
        )
    configs = {config.train.paired_arm: config, counterpart.train.paired_arm: counterpart}
    if set(configs) != {"control", "treatment"}:
        raise ValueError("V12 config/counterpart pair does not contain exactly Control and Treatment")
    control, treatment, evidence = build_v12_paired_models(configs["control"], configs["treatment"])
    evidence.update(
        {
            "paired_arm": config.train.paired_arm,
            "counterpart_config_path": str(counterpart_path),
            "counterpart_config_name": counterpart.config_name,
        }
    )
    if config.train.paired_arm == "control":
        return control, treatment, evidence
    return treatment, control, evidence


def _probabilities(logits: torch.Tensor) -> torch.Tensor:
    if not torch.isfinite(logits).all():
        raise ValueError("task logits must be finite")
    return torch.softmax(logits, dim=-1)


def compose_soft_cascade(
    gate_logits: torch.Tensor,
    dbb_logits: torch.Tensor,
    cu_logits: torch.Tensor,
) -> torch.Tensor:
    """Compose P(DBB/CU) with conditional specialists into five probabilities.

    Class order is Drone, Bird, Balloon, Clutter, Unknown.  No hard route is
    applied, so uncertainty at the gate remains recoverable by the specialist
    probabilities and can later be calibrated or rejected.
    """

    if gate_logits.shape[:-1] != dbb_logits.shape[:-1] or gate_logits.shape[:-1] != cu_logits.shape[:-1]:
        raise ValueError("soft cascade batch/sequence shapes do not match")
    if gate_logits.shape[-1] != 2 or dbb_logits.shape[-1] != 3 or cu_logits.shape[-1] != 2:
        raise ValueError("soft cascade expects gate=2, dbb=3 and cu=2 logits")
    gate = _probabilities(gate_logits)
    dbb = _probabilities(dbb_logits)
    cu = _probabilities(cu_logits)
    return torch.cat([gate[..., 0:1] * dbb, gate[..., 1:2] * cu], dim=-1)


def save_task_checkpoint(
    path: str | Path,
    model: LiquidTaskClassifier,
    optimizer: torch.optim.Optimizer | None,
    *,
    config: LiquidTaskConfig,
    epoch: int,
    global_step: int,
    metrics: Mapping[str, Any],
    training_state: Mapping[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model_contract: dict[str, Any] = {
        "class_names": list(config.task.class_names),
        "num_classes": config.task.num_classes,
        "tr_value_dim": config.model.tr_value_dim,
        "tr_missing_dim": config.model.tr_missing_dim,
        "rd_quality_dim": config.model.rd_quality_dim,
        "dt_valid_dim": config.model.dt_valid_dim,
        "rd_quality_schema": config.data.rd_quality_schema,
        "rd_quality_names": list(RD_QUALITY_NAMES),
        "hidden_dim": config.model.hidden_dim,
        "rd_embedding_dim": config.model.rd_embedding_dim,
        "liquid_hidden_dim": config.model.liquid_hidden_dim,
        "rd_shape": [config.model.rd_height, config.model.rd_width],
        "liquid_tau_min_seconds": config.model.liquid_tau_min_seconds,
        "liquid_tau_max_seconds": config.model.liquid_tau_max_seconds,
        "liquid_zero_dt_event_floor_seconds": config.model.liquid_zero_dt_event_floor_seconds,
        "liquid_dt_max_seconds": config.model.liquid_dt_max_seconds,
        "liquid_invalid_dt_update_mode": config.model.liquid_invalid_dt_update_mode,
        "liquid_invalid_dt_observation_gain": config.model.liquid_invalid_dt_observation_gain,
        "liquid_state_normalization": config.model.liquid_state_normalization,
        "rd_frame_aggregation": config.model.rd_frame_aggregation,
    }
    if _task_schema_number(config.schema_version) >= _task_schema_number(DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION):
        model_contract["dt_valid_feature_mode"] = config.model.dt_valid_feature_mode
    if config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        model_contract.update({
            "dt_reason_mode": config.model.dt_reason_mode,
            "dt_reason_names": list(DT_REASON_NAMES),
            "loss_mode": config.train.loss_mode,
            "prefix_auxiliary_weight": config.train.prefix_auxiliary_weight,
            "prefix_auxiliary_min_events": config.train.prefix_auxiliary_min_events,
            "checkpoint_selection_metric": config.train.checkpoint_selection_metric,
            "cu_output_semantics": ["CLUTTER_ACCEPT_EVIDENCE", "CLUTTER_REJECT_EVIDENCE"],
            "product_selection_targets": {
                "gate_min_precision_recall": config.train.gate_min_precision_recall_target,
                "dbb_min_precision_recall": config.train.dbb_min_precision_recall_target,
                "cu_min_precision_recall": config.train.cu_min_precision_recall_target,
                "drone_bird_min_precision_recall": config.train.drone_bird_min_precision_recall_target,
            },
        })
    if config.model.tr_motion_dim > 0:
        motion_contract = resolve_causal_tr_feature_contract(config.data.tr_motion_schema)
        model_contract.update({
            "tr_motion_dim": config.model.tr_motion_dim,
            "tr_motion_schema": config.data.tr_motion_schema,
            "tr_motion_names": list(motion_contract.value_names),
            "tr_motion_valid_names": list(motion_contract.valid_names),
            "tr_motion_mask_mode": config.data.tr_motion_mask_mode,
            "tr_motion_scales": dict(motion_contract.scales),
            "tr_motion_clip_absolute": motion_contract.clip_absolute,
        })
    if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        model_contract.update(
            {
                "rd_physical_dim": config.model.rd_physical_dim,
                "rd_physical_schema": config.data.rd_physical_schema,
                "rd_physical_names": list(RD_PHYSICAL_NAMES) if config.model.rd_physical_dim else [],
                "rd_physical_mask_mode": config.data.rd_physical_mask_mode,
                "paired_protocol": config.train.paired_protocol,
                "paired_arm": config.train.paired_arm,
                "paired_seed": config.train.paired_seed,
            }
        )
    if config.schema_version == DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION:
        model_contract.update(
            {
                "dbb_auxiliary_num_classes": config.model.dbb_auxiliary_num_classes,
                "dbb_auxiliary_class_names": list(DBB_AUXILIARY_CLASS_NAMES),
                "dbb_auxiliary_class_weights": list(config.train.dbb_auxiliary_class_weights),
                "dbb_auxiliary_supervision_scope": config.train.dbb_auxiliary_supervision_scope,
                "dbb_auxiliary_loss_weight": config.train.dbb_auxiliary_loss_weight,
                "paired_protocol": config.train.paired_protocol,
                "paired_arm": config.train.paired_arm,
                "paired_seed": config.train.paired_seed,
            }
        )
    payload: dict[str, Any] = {
        "checkpoint_schema": config.train.checkpoint_schema,
        "config_name": config.config_name,
        "task_name": config.task.name,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "metrics": dict(metrics),
        "data_contract": {
            "index_sha256": config.data.expected_index_sha256,
            "contract_sha256": config.data.expected_contract_sha256,
            "stats_sha256": config.data.expected_stats_sha256,
        },
        "model_contract": model_contract,
        "model_state": model.state_dict(),
    }
    if config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        payload["config_identity_sha256"] = task_config_identity_sha256(config)
    if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        payload["data_contract"].update(
            {
                "rd_physical_manifest_sha256": config.data.expected_rd_physical_cache_manifest_sha256,
                "train_rd_physical_cache_sha256": config.data.expected_train_rd_physical_cache_sha256,
                "validation_rd_physical_cache_sha256": config.data.expected_validation_rd_physical_cache_sha256,
            }
        )
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = dict(training_state)
    if config.schema_version in FORMAL_PAIRED_SCHEMA_VERSIONS:
        required_resume_fields = {
            "python_random_state",
            "numpy_random_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
            "dataloader_generator_state",
            "run_state",
        }
        if optimizer is None or training_state is None or set(training_state) != required_resume_fields:
            raise ValueError("formal paired checkpoints require optimizer and complete epoch-boundary recovery state")
    torch.save(payload, target)


def load_task_checkpoint(
    path: str | Path,
    model: LiquidTaskClassifier,
    *,
    expected_schema: str,
    map_location: str | torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_schema") != expected_schema:
        raise ValueError(f"invalid task checkpoint schema; expected={expected_schema}")
    if payload.get("task_name") != model.task_name:
        raise ValueError("task checkpoint belongs to a different task")
    if model.config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        if payload.get("config_name") != model.config.config_name:
            raise ValueError("V13 task checkpoint belongs to a different named configuration")
        expected_config_identity = task_config_identity_sha256(model.config)
        if payload.get("config_identity_sha256") != expected_config_identity:
            raise ValueError("V13 task checkpoint configuration identity mismatch")
    expected_data_contract = {
        "index_sha256": model.config.data.expected_index_sha256,
        "contract_sha256": model.config.data.expected_contract_sha256,
        "stats_sha256": model.config.data.expected_stats_sha256,
    }
    if model.config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        expected_data_contract.update(
            {
                "rd_physical_manifest_sha256": model.config.data.expected_rd_physical_cache_manifest_sha256,
                "train_rd_physical_cache_sha256": model.config.data.expected_train_rd_physical_cache_sha256,
                "validation_rd_physical_cache_sha256": model.config.data.expected_validation_rd_physical_cache_sha256,
            }
        )
    if payload.get("data_contract") != expected_data_contract:
        raise ValueError("task checkpoint data hashes do not match the configured data contract")
    expected_contract: dict[str, Any] = {
        "class_names": list(model.class_names),
        "num_classes": model.num_classes,
        "tr_value_dim": model.tr_value_dim,
        "tr_missing_dim": model.tr_missing_dim,
        "rd_quality_dim": model.rd_quality_dim,
        "dt_valid_dim": model.dt_valid_dim,
        "rd_quality_schema": model.rd_quality_schema,
        "rd_quality_names": list(RD_QUALITY_NAMES),
        "hidden_dim": model.config.model.hidden_dim,
        "rd_embedding_dim": model.config.model.rd_embedding_dim,
        "liquid_hidden_dim": model.config.model.liquid_hidden_dim,
        "rd_shape": [model.config.model.rd_height, model.config.model.rd_width],
        "liquid_tau_min_seconds": model.config.model.liquid_tau_min_seconds,
        "liquid_tau_max_seconds": model.config.model.liquid_tau_max_seconds,
        "liquid_zero_dt_event_floor_seconds": model.config.model.liquid_zero_dt_event_floor_seconds,
        "liquid_dt_max_seconds": model.config.model.liquid_dt_max_seconds,
        "liquid_invalid_dt_update_mode": model.config.model.liquid_invalid_dt_update_mode,
        "liquid_invalid_dt_observation_gain": model.config.model.liquid_invalid_dt_observation_gain,
        "liquid_state_normalization": model.config.model.liquid_state_normalization,
        "rd_frame_aggregation": model.config.model.rd_frame_aggregation,
    }
    if _task_schema_number(model.config.schema_version) >= _task_schema_number(DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION):
        expected_contract["dt_valid_feature_mode"] = model.dt_valid_feature_mode
    if model.config.schema_version == HIERARCHICAL_CASCADE_SCHEMA_VERSION:
        expected_contract.update({
            "dt_reason_mode": model.config.model.dt_reason_mode,
            "dt_reason_names": list(DT_REASON_NAMES),
            "loss_mode": model.config.train.loss_mode,
            "prefix_auxiliary_weight": model.config.train.prefix_auxiliary_weight,
            "prefix_auxiliary_min_events": model.config.train.prefix_auxiliary_min_events,
            "checkpoint_selection_metric": model.config.train.checkpoint_selection_metric,
            "cu_output_semantics": ["CLUTTER_ACCEPT_EVIDENCE", "CLUTTER_REJECT_EVIDENCE"],
            "product_selection_targets": {
                "gate_min_precision_recall": model.config.train.gate_min_precision_recall_target,
                "dbb_min_precision_recall": model.config.train.dbb_min_precision_recall_target,
                "cu_min_precision_recall": model.config.train.cu_min_precision_recall_target,
                "drone_bird_min_precision_recall": model.config.train.drone_bird_min_precision_recall_target,
            },
        })
    if model.tr_motion_dim > 0:
        motion_contract = resolve_causal_tr_feature_contract(model.tr_motion_schema)
        expected_contract.update({
            "tr_motion_dim": model.tr_motion_dim,
            "tr_motion_schema": model.tr_motion_schema,
            "tr_motion_names": list(motion_contract.value_names),
            "tr_motion_valid_names": list(motion_contract.valid_names),
            "tr_motion_mask_mode": model.tr_motion_mask_mode,
            "tr_motion_scales": dict(motion_contract.scales),
            "tr_motion_clip_absolute": motion_contract.clip_absolute,
        })
    if model.config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        expected_contract.update(
            {
                "rd_physical_dim": model.rd_physical_dim,
                "rd_physical_schema": model.rd_physical_schema,
                "rd_physical_names": list(RD_PHYSICAL_NAMES) if model.rd_physical_dim else [],
                "rd_physical_mask_mode": model.rd_physical_mask_mode,
                "paired_protocol": model.config.train.paired_protocol,
                "paired_arm": model.config.train.paired_arm,
                "paired_seed": model.config.train.paired_seed,
            }
        )
    if model.config.schema_version == DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION:
        expected_contract.update(
            {
                "dbb_auxiliary_num_classes": model.dbb_auxiliary_num_classes,
                "dbb_auxiliary_class_names": list(DBB_AUXILIARY_CLASS_NAMES),
                "dbb_auxiliary_class_weights": list(model.config.train.dbb_auxiliary_class_weights),
                "dbb_auxiliary_supervision_scope": model.config.train.dbb_auxiliary_supervision_scope,
                "dbb_auxiliary_loss_weight": model.config.train.dbb_auxiliary_loss_weight,
                "paired_protocol": model.config.train.paired_protocol,
                "paired_arm": model.config.train.paired_arm,
                "paired_seed": model.config.train.paired_seed,
            }
        )
    if payload.get("model_contract") != expected_contract:
        raise ValueError("task checkpoint model/input contract does not match the configured repaired model")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        if "optimizer_state" not in payload:
            raise ValueError("task checkpoint does not contain optimizer state")
        optimizer.load_state_dict(payload["optimizer_state"])
    if model.config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        required_resume_fields = {
            "python_random_state",
            "numpy_random_state",
            "torch_cpu_rng_state",
            "torch_cuda_rng_states",
            "dataloader_generator_state",
            "run_state",
        }
        training_state = payload.get("training_state")
        if not isinstance(training_state, dict) or set(training_state) != required_resume_fields:
            raise ValueError("formal paired checkpoint lacks complete epoch-boundary recovery state")
    return payload


def capture_training_state(
    dataloader_generator: torch.Generator,
    *,
    run_state: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "dataloader_generator_state": dataloader_generator.get_state(),
        "run_state": dict(run_state),
    }


def restore_training_state(
    payload: Mapping[str, Any],
    dataloader_generator: torch.Generator,
) -> dict[str, Any]:
    training_state = payload.get("training_state")
    if not isinstance(training_state, Mapping):
        raise ValueError("checkpoint has no restorable training_state")
    random.setstate(training_state["python_random_state"])
    np.random.set_state(training_state["numpy_random_state"])
    torch.set_rng_state(torch.as_tensor(training_state["torch_cpu_rng_state"], device="cpu"))
    cuda_states = list(training_state["torch_cuda_rng_states"])
    if cuda_states:
        if not torch.cuda.is_available() or len(cuda_states) != torch.cuda.device_count():
            raise ValueError("checkpoint CUDA RNG state count does not match the current CUDA environment")
        torch.cuda.set_rng_state_all([torch.as_tensor(value, device="cpu") for value in cuda_states])
    dataloader_generator.set_state(
        torch.as_tensor(training_state["dataloader_generator_state"], device="cpu")
    )
    run_state = training_state.get("run_state")
    if not isinstance(run_state, Mapping):
        raise ValueError("checkpoint training_state lacks a valid run_state")
    return dict(run_state)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def task_device(config: LiquidTaskConfig) -> torch.device:
    return torch.device("cuda" if config.train.device == "cuda" and torch.cuda.is_available() else "cpu")


def summarize_task_records(records: Iterable[Mapping[str, Any]], *, class_names: tuple[str, ...], scope: str) -> dict[str, Any]:
    rows = list(records)
    class_count = len(class_names)
    matrix = np.zeros((class_count, class_count), dtype=np.int64)
    for row in rows:
        true_id = int(row["true_id"])
        pred_id = int(row["pred_id"])
        if not 0 <= true_id < class_count or not 0 <= pred_id < class_count:
            raise ValueError("task metric record contains invalid class id")
        matrix[true_id, pred_id] += 1
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    per_class: list[dict[str, Any]] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1_values: list[float] = []
    for class_id, class_name in enumerate(class_names):
        tp = int(matrix[class_id, class_id])
        precision = tp / int(predicted[class_id]) if predicted[class_id] else 0.0
        recall = tp / int(support[class_id]) if support[class_id] else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_values.append(float(f1))
        confusion = matrix[class_id].copy()
        confusion[class_id] = 0
        destination = int(confusion.argmax()) if confusion.sum() else None
        per_class.append({
            "class": class_name,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(support[class_id]),
            "major_confusion_to": class_names[destination] if destination is not None and confusion[destination] else None,
            "major_confusion_count": int(confusion[destination]) if destination is not None else 0,
        })
    total = int(matrix.sum())
    return {
        "scope": scope,
        "support": total,
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_precision": float(np.mean(precisions)) if precisions else None,
        "macro_recall": float(np.mean(recalls)) if recalls else None,
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "balanced_accuracy": float(np.mean(recalls)) if recalls else None,
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
    }


def evaluate_task(
    model: LiquidTaskClassifier,
    dataset: TaskSequenceDataset,
    *,
    config: LiquidTaskConfig,
    device: torch.device,
    batch_size: int | None = None,
) -> dict[str, Any]:
    model.eval()
    loader = build_task_dataloader(dataset, batch_size=batch_size or config.train.batch_size, shuffle=False)
    window_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    dt_invalid = 0
    with torch.inference_mode():
        for batch in loader:
            tensor_batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            output = model(
                tr_features=tensor_batch["tr_features"],
                tr_missing_mask=tensor_batch["tr_missing_mask"],
                tr_dt_seconds=tensor_batch["tr_dt_seconds"],
                tr_dt_valid=tensor_batch["tr_dt_valid"],
                tr_dt_reason=tensor_batch.get("tr_dt_reason"),
                tr_motion_features=tensor_batch.get("tr_motion_features"),
                tr_motion_valid_mask=tensor_batch.get("tr_motion_valid_mask"),
                rd_physical_features=tensor_batch.get("rd_physical_features"),
                rd_physical_valid=tensor_batch.get("rd_physical_valid"),
                rd_frames=tensor_batch["rd_frames"],
                rd_frame_mask=tensor_batch["rd_frame_mask"],
                rd_quality=tensor_batch["rd_quality"],
                tr_padding_mask=tensor_batch["tr_padding_mask"],
            )
            probabilities = _probabilities(output["sequence_logits"])
            dt_invalid += int(output["dt_invalid_mask"].sum().item())
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
                        "parent_sample_id": parent_id,
                        "track_id": batch["track_ids"][batch_index],
                        "source_name": batch["source_names"][batch_index],
                        "label_name": batch["label_names"][batch_index],
                        "event_index": event_index,
                        "window_length": event_index + 1,
                    }
                    window_records.append(dict(final))
                if final is None:
                    raise ValueError(f"empty task sequence: {parent_id}")
                parent_records.append(final)
    return {
        "task_name": config.task.name,
        "class_names": list(config.task.class_names),
        "parent_count": len(parent_records),
        "window_count": len(window_records),
        "window_metrics": summarize_task_records(window_records, class_names=config.task.class_names, scope="causal_prefix_windows"),
        "parent_metrics": summarize_task_records(parent_records, class_names=config.task.class_names, scope="parent_final_prefix"),
        "dt_quality": {"dt_invalid_event_count": dt_invalid},
        "records": {"window": window_records, "parent": parent_records},
    }


__all__ = [
    "BASE_LABELS",
    "DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION",
    "DT_VALID_FEATURE_MODES",
    "LiquidTaskClassifier",
    "LiquidTaskConfig",
    "REPAIRED_TASK_SCHEMA_VERSION",
    "TASK_SCHEMA_VERSION",
    "TASK_SPECS",
    "TR_MOTION_GATE_SCHEMA_VERSION",
    "TR_OBSERVATION_INCREMENT_GATE_SCHEMA_VERSION",
    "RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION",
    "RD_PHYSICAL_PAIRED_PROTOCOL",
    "DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION",
    "DBB_AUXILIARY_PAIRED_PROTOCOL",
    "DBB_AUXILIARY_CLASS_NAMES",
    "DBB_AUXILIARY_TREATMENT_WEIGHT",
    "DBB_AUXILIARY_CLASS_WEIGHTS",
    "DBB_AUXILIARY_SUPERVISION_SCOPE",
    "FORMAL_PAIRED_SCHEMA_VERSIONS",
    "HIERARCHICAL_CASCADE_SCHEMA_VERSION",
    "HIERARCHICAL_CHECKPOINT_SELECTION",
    "HIERARCHICAL_CU_OUTPUT_NAMES",
    "HIERARCHICAL_LOSS_MODE",
    "DT_REASON_DYNAMICS_MODE",
    "WEAK_CLASS_SELECTION_SCHEMA_VERSION",
    "TaskSequenceDataset",
    "TaskSpec",
    "build_task_dataloader",
    "build_task_dataset",
    "build_v11_paired_model_for_config",
    "build_v11_paired_models",
    "build_v12_paired_models",
    "build_paired_model_for_config",
    "capture_training_state",
    "compose_soft_cascade",
    "evaluate_task",
    "load_task_checkpoint",
    "load_task_config",
    "restore_training_state",
    "save_task_checkpoint",
    "seed_everything",
    "summarize_task_records",
    "task_device",
    "task_prefix_loss",
    "dbb_auxiliary_final_state_loss",
    "task_training_objective",
    "task_spec",
    "task_config_identity_sha256",
    "verify_task_data_contract",
]
