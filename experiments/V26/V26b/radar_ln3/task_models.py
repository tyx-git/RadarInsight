"""Versioned task-model contracts used by historical experiments.

The canonical V21 product architecture lives in ``open_set_architecture.py``
and its L1 implementation lives in ``known_gate.py``.  The identifiers
``dbb_cu_gate`` and ``cu_specialist`` below remain frozen only so historical
configs/checkpoints can be reproduced; they must not be interpreted as V21
layer names or current Unknown semantics.

This module deliberately reuses only the isolated ``src-v3`` implementation.
It does not load the five-class checkpoint, change the CQ08 index, or expose
paths and identifiers to the model.  The three supported tasks are:

* ``dbb_cu_gate``: DBB versus CU;
* ``dbb_specialist``: Drone versus Bird versus Balloon;
* historical ``cu_specialist``: Clutter versus Unknown;
* V13 ``cu_specialist``: Clutter-accept versus learned Unknown evidence;
* V14 strict-open-set ``cu_specialist``: Clutter-accept versus known
  non-Clutter evidence, with Unknown excluded from every learned target.

V5-V8 use the shared causal TR+RD liquid contract.  V9 is a deliberately
isolated DBB/CU Gate experiment that adds causal TR motion-consistency values;
the specialists remain on their frozen earlier contracts.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .data import DT_REASON_NAMES, RD_QUALITY_NAMES, require_causal_rd_quality_schema
from .features import (
    TR_MOTION_SCHEMA,
    TR_OBSERVATION_INCREMENT_SCHEMA,
    resolve_causal_tr_feature_contract,
)
from .liquid import build_cfc_liquid_encoder
from .model import RDSpatialEncoder
from .micro_doppler import MICRO_DOPPLER_MASK_MODE, MICRO_DOPPLER_NAMES, MICRO_DOPPLER_SCHEMA, MICRO_DOPPLER_VALUE_DIM, extract_current_event_micro_doppler
from .micro_doppler_broadband import BROADBAND_MICRO_DOPPLER_MASK_MODE, BROADBAND_MICRO_DOPPLER_NAMES, BROADBAND_MICRO_DOPPLER_SCHEMA, BROADBAND_MICRO_DOPPLER_VALUE_DIM, extract_current_event_broadband_micro_doppler
from .rd_physical import RD_PHYSICAL_MASK_MODE, RD_PHYSICAL_NAMES, RD_PHYSICAL_SCHEMA, RD_PHYSICAL_VALUE_DIM, extract_current_event_rd_physical
from .runtime import canonical_sha256, parse_bool, resolve_project_path


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
STRICT_OPEN_SET_SCHEMA_VERSION = "RADAR-LN3-V14"
ANGULAR_KNOWNNESS_SCHEMA_VERSION = "RADAR-LN3-V15"
PRODUCT_LOCAL_SCHEMA_VERSION = "RADAR-LN3-V16"
MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION = "RADAR-LN3-V17"
MICRO_DOPPLER_PAIRED_PROTOCOL = (
    "v17-dbb-rd-physical4-control-versus-plus-target-anchored-micro-doppler3-treatment-v3"
)
BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION = "RADAR-LN3-V18"
BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL = (
    "v18-dbb-rd-physical4-control-versus-plus-broadband-multiscale-"
    "micro-doppler8-stable-treatment-v3"
)
V18R_STABLE8_DENSITY_SCHEMA_VERSION = "RADAR-LN3-V18r"
V18R_STABLE8_DENSITY_PAIRED_PROTOCOL = (
    "v18r-dbb-stable8-bounded-observation-control-versus-"
    "density-weighted-treatment-v1"
)
V18R_SOURCE_CONTROL_CONFIG_PATH = (
    "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-"
    "broadband-multiscale-micro-doppler8-stable-control-seed42-config.json"
)
V18R_SOURCE_TREATMENT_CONFIG_PATH = (
    "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-"
    "broadband-multiscale-micro-doppler8-stable-treatment-seed42-config.json"
)
LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION = "RADAR-LN3-V20"
V20_SKIP_PAIRED_PROTOCOL = (
    "v20-dbb-bounded-observation-control-versus-zero-delta-skip-treatment-v1"
)
V20_DENSITY_PAIRED_PROTOCOL = (
    "v20r-dbb-bounded-observation-control-versus-density-weighted-treatment-v1"
)
V20_PAIRED_PROTOCOLS = frozenset(
    {V20_SKIP_PAIRED_PROTOCOL, V20_DENSITY_PAIRED_PROTOCOL}
)
PRODUCT_LOCAL_LOSS_MODE = "parent_final_task_ce_with_prefix_auxiliary"
PRODUCT_LOCAL_CHECKPOINT_SELECTION = "parent_min_precision_recall"
BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION = (
    "parent_min_precision_recall_f1_then_min_drone_bird_errors_then_macro_f1"
)
STRICT_OPEN_SET_LOSS_MODE = "parent_final_known_only_open_set_nll_with_prefix_auxiliary"
STRICT_OPEN_SET_CHECKPOINT_SELECTION = "known_only_parent_route_guarded"
STRICT_OPEN_SET_GATE_OUTPUT_NAMES = (
    "DBB_ACCEPT_EVIDENCE",
    "ROUTE_TO_CU_EVIDENCE",
)
STRICT_OPEN_SET_CU_OUTPUT_NAMES = (
    "CLUTTER_ACCEPT_EVIDENCE",
    "NON_CLUTTER_KNOWN_EVIDENCE",
)
KNOWN_BASE_LABELS = BASE_LABELS[:4]
UNKNOWN_BASE_LABEL = "UnknownTarget"
STRICT_OPEN_SET_SCHEMA_VERSIONS = frozenset(
    {STRICT_OPEN_SET_SCHEMA_VERSION, ANGULAR_KNOWNNESS_SCHEMA_VERSION}
)
HIERARCHICAL_SCHEMA_VERSIONS = frozenset(
    {HIERARCHICAL_CASCADE_SCHEMA_VERSION, *STRICT_OPEN_SET_SCHEMA_VERSIONS}
)
DBB_READOUT_LINEAR = "linear"
DBB_READOUT_ANGULAR_PROTOTYPE = "angular_prototype"
DBB_KNOWNNESS_PROBABILITY = "dbb_top_probability_and_margin"
DBB_KNOWNNESS_COSINE = "dbb_top_cosine_and_cosine_gap"
DBB_ANGULAR_EMBEDDING_DIM = 48
DBB_ANGULAR_SCALE = 16.0
DBB_ANGULAR_MARGIN = 0.20
DBB_ANGULAR_PAIRED_PROTOCOL = "v15-linear-control-v15-angular-prototype-treatment-v1"
DT_REASON_DYNAMICS_MODE = "v3_reason_aware_dynamics_only"
FORMAL_PAIRED_SCHEMA_VERSIONS = frozenset(
    {
        RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION,
        DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION,
        MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        V18R_STABLE8_DENSITY_SCHEMA_VERSION,
        LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
    }
)
DT_VALID_FEATURE_MODES = frozenset({"fusion_and_dynamics", "dynamics_only"})
_TASK_SCHEMA_PATTERN = re.compile(
    r"^RADAR-LN3-V(?P<version>\d+)(?P<revision>r)?$"
)


def _task_schema_number(schema_version: str) -> int:
    match = _TASK_SCHEMA_PATTERN.fullmatch(str(schema_version))
    if match is None:
        raise ValueError(f"invalid RADAR-LN3 task schema version: {schema_version!r}")
    if (
        match.group("revision") is not None
        and str(schema_version) != V18R_STABLE8_DENSITY_SCHEMA_VERSION
    ):
        raise ValueError(
            f"unsupported RADAR-LN3 task schema revision: {schema_version!r}"
        )
    return int(match.group("version"))


def _is_repaired_task_schema(schema_version: str) -> bool:
    return _task_schema_number(schema_version) >= _task_schema_number(REPAIRED_TASK_SCHEMA_VERSION)


def is_strict_open_set_schema(schema_version: str) -> bool:
    return str(schema_version) in STRICT_OPEN_SET_SCHEMA_VERSIONS


def is_product_local_schema(schema_version: str) -> bool:
    return str(schema_version) in {
        PRODUCT_LOCAL_SCHEMA_VERSION,
        MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        V18R_STABLE8_DENSITY_SCHEMA_VERSION,
        LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION,
    }


def is_liquid_dynamics_paired_schema(schema_version: str) -> bool:
    return str(schema_version) == LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION


def is_v18r_stable8_density_schema(schema_version: str) -> bool:
    return str(schema_version) == V18R_STABLE8_DENSITY_SCHEMA_VERSION


def is_micro_doppler_paired_schema(schema_version: str) -> bool:
    return str(schema_version) in {
        MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        V18R_STABLE8_DENSITY_SCHEMA_VERSION,
    }


def is_broadband_micro_doppler_schema(schema_version: str) -> bool:
    return str(schema_version) in {
        BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        V18R_STABLE8_DENSITY_SCHEMA_VERSION,
    }




def micro_doppler_names_for_config(config: "LiquidTaskConfig") -> tuple[str, ...]:
    if not config.model.micro_doppler_dim:
        return ()
    if is_broadband_micro_doppler_schema(config.schema_version):
        return BROADBAND_MICRO_DOPPLER_NAMES
    return MICRO_DOPPLER_NAMES


def requires_task_config_identity(schema_version: str) -> bool:
    return str(schema_version) in HIERARCHICAL_SCHEMA_VERSIONS or is_product_local_schema(
        schema_version
    )


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


STRICT_OPEN_SET_TASK_SPECS: dict[str, TaskSpec] = {
    "dbb_cu_gate": TaskSpec(
        name="dbb_cu_gate",
        class_names=STRICT_OPEN_SET_GATE_OUTPUT_NAMES,
        label_to_id={
            "DroneTarget": 0,
            "BirdTarget": 0,
            "BalloonTarget": 0,
            "ClutterTarget": 1,
        },
        allowed_base_labels=frozenset(KNOWN_BASE_LABELS),
    ),
    "dbb_specialist": TaskSpec(
        name="dbb_specialist",
        class_names=("DroneTarget", "BirdTarget", "BalloonTarget"),
        label_to_id={"DroneTarget": 0, "BirdTarget": 1, "BalloonTarget": 2},
        allowed_base_labels=frozenset({"DroneTarget", "BirdTarget", "BalloonTarget"}),
    ),
    "cu_specialist": TaskSpec(
        name="cu_specialist",
        class_names=STRICT_OPEN_SET_CU_OUTPUT_NAMES,
        label_to_id={
            "ClutterTarget": 0,
            "DroneTarget": 1,
            "BirdTarget": 1,
            "BalloonTarget": 1,
        },
        allowed_base_labels=frozenset(KNOWN_BASE_LABELS),
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
    micro_doppler_schema: str | None
    micro_doppler_mask_mode: str | None
    micro_doppler_cache_manifest_path: str | None
    expected_micro_doppler_cache_manifest_sha256: str | None
    train_micro_doppler_cache_path: str | None
    validation_micro_doppler_cache_path: str | None
    expected_train_micro_doppler_cache_sha256: str | None
    expected_validation_micro_doppler_cache_sha256: str | None
    train_micro_doppler_stft_cache_path: str | None
    validation_micro_doppler_stft_cache_path: str | None
    expected_train_micro_doppler_stft_cache_sha256: str | None
    expected_validation_micro_doppler_stft_cache_sha256: str | None
    micro_doppler_frame_provenance_path: str | None
    expected_micro_doppler_frame_provenance_sha256: str | None


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
    micro_doppler_dim: int
    dbb_auxiliary_num_classes: int
    dt_reason_mode: str
    dbb_readout_mode: str
    dbb_embedding_dim: int
    dbb_cosine_scale: float
    dbb_angular_margin: float
    dbb_knownness_score_name: str


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
    task_min_precision_recall_target: float | None


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

    model_identity = asdict(config.model)
    if config.schema_version != ANGULAR_KNOWNNESS_SCHEMA_VERSION:
        for field_name in (
            "dbb_readout_mode",
            "dbb_embedding_dim",
            "dbb_cosine_scale",
            "dbb_angular_margin",
            "dbb_knownness_score_name",
        ):
            model_identity.pop(field_name)
    data_identity = asdict(config.data)
    if not is_micro_doppler_paired_schema(config.schema_version):
        model_identity.pop("micro_doppler_dim")
        for field_name in (
            "micro_doppler_schema",
            "micro_doppler_mask_mode",
            "micro_doppler_cache_manifest_path",
            "expected_micro_doppler_cache_manifest_sha256",
            "train_micro_doppler_cache_path",
            "validation_micro_doppler_cache_path",
            "expected_train_micro_doppler_cache_sha256",
            "expected_validation_micro_doppler_cache_sha256",
            "train_micro_doppler_stft_cache_path",
            "validation_micro_doppler_stft_cache_path",
            "expected_train_micro_doppler_stft_cache_sha256",
            "expected_validation_micro_doppler_stft_cache_sha256",
            "micro_doppler_frame_provenance_path",
            "expected_micro_doppler_frame_provenance_sha256",
        ):
            data_identity.pop(field_name)
    train_identity = asdict(config.train)
    if not is_product_local_schema(config.schema_version):
        train_identity.pop("task_min_precision_recall_target")
    payload = {
        "schema_version": config.schema_version,
        "config_name": config.config_name,
        "task": {
            "name": config.task.name,
            "class_names": list(config.task.class_names),
            "label_to_id": dict(config.task.label_to_id),
            "allowed_base_labels": sorted(config.task.allowed_base_labels),
        },
        "data": data_identity,
        "model": model_identity,
        "train": train_identity,
    }
    return canonical_sha256(payload)


def task_spec(name: str, *, schema_version: str | None = None) -> TaskSpec:
    specs = STRICT_OPEN_SET_TASK_SPECS if is_strict_open_set_schema(str(schema_version)) else TASK_SPECS
    try:
        return specs[str(name)]
    except KeyError as exc:
        raise ValueError(f"unsupported RADAR-LN3 task: {name}") from exc


def _required(mapping: Mapping[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"task configuration is missing {key}")
    return mapping[key]


# Compatibility name retained for historical task callers.  New loaders use
# the shared strict parser directly through this alias.
_as_bool = parse_bool


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
    spec = task_spec(str(_required(task_payload, "name")), schema_version=schema_version)
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
        micro_doppler_schema=(
            None
            if data_payload.get("micro_doppler_schema") is None
            else str(data_payload["micro_doppler_schema"])
        ),
        micro_doppler_mask_mode=(
            None
            if data_payload.get("micro_doppler_mask_mode") is None
            else str(data_payload["micro_doppler_mask_mode"])
        ),
        micro_doppler_cache_manifest_path=(
            None
            if data_payload.get("micro_doppler_cache_manifest_path") is None
            else str(data_payload["micro_doppler_cache_manifest_path"])
        ),
        expected_micro_doppler_cache_manifest_sha256=(
            None
            if data_payload.get("expected_micro_doppler_cache_manifest_sha256") is None
            else str(data_payload["expected_micro_doppler_cache_manifest_sha256"])
        ),
        train_micro_doppler_cache_path=(
            None
            if data_payload.get("train_micro_doppler_cache_path") is None
            else str(data_payload["train_micro_doppler_cache_path"])
        ),
        validation_micro_doppler_cache_path=(
            None
            if data_payload.get("validation_micro_doppler_cache_path") is None
            else str(data_payload["validation_micro_doppler_cache_path"])
        ),
        expected_train_micro_doppler_cache_sha256=(
            None
            if data_payload.get("expected_train_micro_doppler_cache_sha256") is None
            else str(data_payload["expected_train_micro_doppler_cache_sha256"])
        ),
        expected_validation_micro_doppler_cache_sha256=(
            None
            if data_payload.get("expected_validation_micro_doppler_cache_sha256") is None
            else str(data_payload["expected_validation_micro_doppler_cache_sha256"])
        ),
        train_micro_doppler_stft_cache_path=(
            None
            if data_payload.get("train_micro_doppler_stft_cache_path") is None
            else str(data_payload["train_micro_doppler_stft_cache_path"])
        ),
        validation_micro_doppler_stft_cache_path=(
            None
            if data_payload.get("validation_micro_doppler_stft_cache_path") is None
            else str(data_payload["validation_micro_doppler_stft_cache_path"])
        ),
        expected_train_micro_doppler_stft_cache_sha256=(
            None
            if data_payload.get("expected_train_micro_doppler_stft_cache_sha256") is None
            else str(data_payload["expected_train_micro_doppler_stft_cache_sha256"])
        ),
        expected_validation_micro_doppler_stft_cache_sha256=(
            None
            if data_payload.get("expected_validation_micro_doppler_stft_cache_sha256") is None
            else str(data_payload["expected_validation_micro_doppler_stft_cache_sha256"])
        ),
        micro_doppler_frame_provenance_path=(
            None
            if data_payload.get("micro_doppler_frame_provenance_path") is None
            else str(data_payload["micro_doppler_frame_provenance_path"])
        ),
        expected_micro_doppler_frame_provenance_sha256=(
            None
            if data_payload.get("expected_micro_doppler_frame_provenance_sha256")
            is None
            else str(
                data_payload["expected_micro_doppler_frame_provenance_sha256"]
            )
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
        micro_doppler_dim=int(model_payload.get("micro_doppler_dim", 0)),
        dbb_auxiliary_num_classes=int(model_payload.get("dbb_auxiliary_num_classes", 0)),
        dt_reason_mode=str(model_payload.get("dt_reason_mode", "legacy_dt_valid_only")),
        dbb_readout_mode=str(model_payload.get("dbb_readout_mode", DBB_READOUT_LINEAR)),
        dbb_embedding_dim=int(model_payload.get("dbb_embedding_dim", 0)),
        dbb_cosine_scale=float(model_payload.get("dbb_cosine_scale", 1.0)),
        dbb_angular_margin=float(model_payload.get("dbb_angular_margin", 0.0)),
        dbb_knownness_score_name=str(
            model_payload.get("dbb_knownness_score_name", DBB_KNOWNNESS_PROBABILITY)
        ),
    )
    if model.num_classes != spec.num_classes:
        raise ValueError(f"model.num_classes={model.num_classes} does not match task={spec.num_classes}")
    if (model.tr_value_dim, model.tr_missing_dim, model.rd_quality_dim) != (14, 11, 7):
        raise ValueError("task model must use the current TR/RD input dimensions 14/11/7")
    if (model.rd_height, model.rd_width) != (21, 384):
        raise ValueError("task model must use the current RD shape 21x384")
    if model.dt_valid_dim != 1 or model.liquid_invalid_dt_update_mode not in {
        "floor",
        "bounded_observation",
        "skip",
        "strict_skip",
        "density_weighted",
    }:
        raise ValueError(
            "task model must explicitly consume dt_valid with a supported liquid update mode"
        )
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
    if model.dbb_readout_mode not in {DBB_READOUT_LINEAR, DBB_READOUT_ANGULAR_PROTOTYPE}:
        raise ValueError("task DBB readout mode must be linear or angular_prototype")
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
    micro_doppler_fields = (
        data.micro_doppler_schema,
        data.micro_doppler_mask_mode,
        data.micro_doppler_cache_manifest_path,
        data.expected_micro_doppler_cache_manifest_sha256,
        data.train_micro_doppler_cache_path,
        data.validation_micro_doppler_cache_path,
        data.expected_train_micro_doppler_cache_sha256,
        data.expected_validation_micro_doppler_cache_sha256,
        data.train_micro_doppler_stft_cache_path,
        data.validation_micro_doppler_stft_cache_path,
        data.expected_train_micro_doppler_stft_cache_sha256,
        data.expected_validation_micro_doppler_stft_cache_sha256,
        data.micro_doppler_frame_provenance_path,
        data.expected_micro_doppler_frame_provenance_sha256,
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
    elif is_micro_doppler_paired_schema(schema_version):
        if spec.name != "dbb_specialist" or motion_fields_present:
            raise ValueError(
                f"{schema_version} is restricted to the DBB specialist without TR derived inputs"
            )
        if model.rd_physical_dim != RD_PHYSICAL_VALUE_DIM:
            raise ValueError("RADAR-LN3-V17 Control and Treatment both require the frozen four RD physical values")
        if (data.rd_physical_schema, data.rd_physical_mask_mode) != (
            RD_PHYSICAL_SCHEMA,
            RD_PHYSICAL_MASK_MODE,
        ) or any(value is None for value in physical_fields[2:]):
            raise ValueError(
                f"{schema_version} requires the frozen RD physical cache contract in both arms"
            )
        expected_micro_dim = (
            BROADBAND_MICRO_DOPPLER_VALUE_DIM
            if is_broadband_micro_doppler_schema(schema_version)
            else MICRO_DOPPLER_VALUE_DIM
        )
        expected_micro_contract = (
            BROADBAND_MICRO_DOPPLER_SCHEMA,
            BROADBAND_MICRO_DOPPLER_MASK_MODE,
        ) if is_broadband_micro_doppler_schema(schema_version) else (
            MICRO_DOPPLER_SCHEMA,
            MICRO_DOPPLER_MASK_MODE,
        )
        allowed_micro_dims = (
            {expected_micro_dim}
            if is_v18r_stable8_density_schema(schema_version)
            else {0, expected_micro_dim}
        )
        if model.micro_doppler_dim not in allowed_micro_dims:
            raise ValueError(
                f"{schema_version} micro_doppler_dim must be one of "
                f"{sorted(allowed_micro_dims)}"
            )
        if (data.micro_doppler_schema, data.micro_doppler_mask_mode) != expected_micro_contract or any(
            value is None for value in micro_doppler_fields[2:]
        ):
            raise ValueError(
                f"{schema_version} requires versioned summary/evidence cache paths and hashes"
            )
    elif model.rd_physical_dim != 0 or any(value is not None for value in physical_fields):
        raise ValueError("task schemas after RADAR-LN3-V11 must not silently retain the V11 RD physical treatment")
    if (
        not is_micro_doppler_paired_schema(schema_version)
        and (model.micro_doppler_dim != 0 or any(value is not None for value in micro_doppler_fields))
    ):
        raise ValueError("micro-Doppler inputs and cache fields are reserved for RADAR-LN3-V17")
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
        task_min_precision_recall_target=(
            None
            if train_payload.get("task_min_precision_recall_target") is None
            else float(train_payload["task_min_precision_recall_target"])
        ),
    )
    if train.learning_rate <= 0.0 or train.gradient_clip_norm <= 0.0:
        raise ValueError("task learning_rate and gradient_clip_norm must be positive")
    if schema_version in HIERARCHICAL_SCHEMA_VERSIONS:
        expected_loss_mode = (
            STRICT_OPEN_SET_LOSS_MODE
            if is_strict_open_set_schema(schema_version)
            else HIERARCHICAL_LOSS_MODE
        )
        expected_selection = (
            STRICT_OPEN_SET_CHECKPOINT_SELECTION
            if is_strict_open_set_schema(schema_version)
            else HIERARCHICAL_CHECKPOINT_SELECTION
        )
        if train.loss_mode != expected_loss_mode:
            raise ValueError(f"{schema_version} task training must use {expected_loss_mode}")
        if train.checkpoint_selection_metric != expected_selection:
            raise ValueError(f"{schema_version} checkpoint selection must use {expected_selection}")
        if not 0.0 <= train.prefix_auxiliary_weight <= 1.0:
            raise ValueError(f"{schema_version} prefix_auxiliary_weight must be in [0,1]")
        if train.prefix_auxiliary_weight > 0.0 and train.prefix_auxiliary_min_events < 2:
            raise ValueError(f"{schema_version} prefix auxiliary supervision must exclude the first evidence event")
        if model.dt_reason_mode != DT_REASON_DYNAMICS_MODE:
            raise ValueError(f"{schema_version} model must use dt_reason_mode={DT_REASON_DYNAMICS_MODE}")
        allowed_contract_schemas = (
            {STRICT_OPEN_SET_SCHEMA_VERSION, ANGULAR_KNOWNNESS_SCHEMA_VERSION}
            if schema_version == ANGULAR_KNOWNNESS_SCHEMA_VERSION
            else {schema_version}
        )
        if data.contract_schema_version not in allowed_contract_schemas:
            raise ValueError(
                f"{schema_version} tasks must consume an approved reason-aware data contract; "
                f"actual={data.contract_schema_version!r}"
            )
        if model.dbb_auxiliary_num_classes != 0 or model.rd_physical_dim != 0:
            raise ValueError(f"{schema_version} must not silently retain V11/V12 experimental branches")
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError(f"{schema_version} requires 20 fixed epochs with early stopping disabled")
        if train.epoch_checkpoint_dir is None or not train.record_parent_order_hash or not train.save_every_epoch:
            raise ValueError(f"{schema_version} requires epoch checkpoints and parent-order hashes")
        if train.dataloader_seed != train.seed:
            raise ValueError(f"{schema_version} model seed and DataLoader seed must match")
        if schema_version != ANGULAR_KNOWNNESS_SCHEMA_VERSION and any(
            value is not None
            for value in (
                train.paired_protocol,
                train.paired_arm,
                train.paired_seed,
                train.paired_counterpart_config_path,
            )
        ):
            raise ValueError(f"{schema_version} is one joint product run, not a paired-arm experiment")
        if any(
            value is None
            for value in (
                train.joint_artifact_dir,
                train.joint_last_checkpoint_path,
                train.joint_best_checkpoint_path,
                train.joint_checkpoint_schema,
            )
        ):
            raise ValueError(f"{schema_version} requires explicit joint artifact and checkpoint fields")
        if not str(train.joint_checkpoint_schema).startswith(schema_version):
            raise ValueError(f"{schema_version} joint checkpoint schema must use its version prefix")
        targets = (
            train.gate_min_precision_recall_target,
            train.dbb_min_precision_recall_target,
            train.cu_min_precision_recall_target,
            train.drone_bird_min_precision_recall_target,
        )
        if any(value is None or not 0.0 < float(value) <= 1.0 for value in targets):
            raise ValueError(f"{schema_version} product selection targets must all be in (0,1]")
        if is_strict_open_set_schema(schema_version):
            if UNKNOWN_BASE_LABEL in spec.allowed_base_labels or UNKNOWN_BASE_LABEL in spec.label_to_id:
                raise ValueError("strict-open-set task mapping must not expose UnknownTarget")
            if spec.name == "dbb_cu_gate" and spec.class_names != STRICT_OPEN_SET_GATE_OUTPUT_NAMES:
                raise ValueError("strict-open-set Gate output semantics are invalid")
            if spec.name == "cu_specialist" and spec.class_names != STRICT_OPEN_SET_CU_OUTPUT_NAMES:
                raise ValueError("strict-open-set CU output semantics are invalid")
        if schema_version == ANGULAR_KNOWNNESS_SCHEMA_VERSION:
            if (
                train.paired_protocol != DBB_ANGULAR_PAIRED_PROTOCOL
                or train.paired_arm not in {"control", "treatment"}
                or train.paired_seed != train.seed
                or train.dataloader_seed != train.seed
                or train.paired_counterpart_config_path is None
            ):
                raise ValueError("V15 requires the frozen paired DBB angular-knownness protocol")
            if train.cuda_deterministic_algorithms:
                raise ValueError("V15 fixes initialization and parent order but does not claim strict CUDA determinism")
            if spec.name != "dbb_specialist":
                if (
                    model.dbb_readout_mode != DBB_READOUT_LINEAR
                    or model.dbb_embedding_dim != 0
                    or model.dbb_cosine_scale != 1.0
                    or model.dbb_angular_margin != 0.0
                    or model.dbb_knownness_score_name != DBB_KNOWNNESS_PROBABILITY
                ):
                    raise ValueError("V15 Gate/CU must keep the frozen linear readout contract")
            elif train.paired_arm == "control":
                if (
                    model.dbb_readout_mode != DBB_READOUT_LINEAR
                    or model.dbb_embedding_dim != 0
                    or model.dbb_cosine_scale != 1.0
                    or model.dbb_angular_margin != 0.0
                    or model.dbb_knownness_score_name != DBB_KNOWNNESS_PROBABILITY
                ):
                    raise ValueError("V15 Control DBB must retain the linear probability knownness head")
            elif (
                model.dbb_readout_mode != DBB_READOUT_ANGULAR_PROTOTYPE
                or model.dbb_embedding_dim != DBB_ANGULAR_EMBEDDING_DIM
                or model.dbb_cosine_scale != DBB_ANGULAR_SCALE
                or model.dbb_angular_margin != DBB_ANGULAR_MARGIN
                or model.dbb_knownness_score_name != DBB_KNOWNNESS_COSINE
            ):
                raise ValueError("V15 Treatment DBB angular head does not match the frozen contract")
        elif (
            model.dbb_readout_mode != DBB_READOUT_LINEAR
            or model.dbb_embedding_dim != 0
            or model.dbb_cosine_scale != 1.0
            or model.dbb_angular_margin != 0.0
            or model.dbb_knownness_score_name != DBB_KNOWNNESS_PROBABILITY
        ):
            raise ValueError("DBB angular readout fields are reserved for RADAR-LN3-V15")
    elif is_product_local_schema(schema_version):
        is_v17_micro = is_micro_doppler_paired_schema(schema_version)
        is_v20_dynamics = is_liquid_dynamics_paired_schema(schema_version)
        if train.loss_mode != PRODUCT_LOCAL_LOSS_MODE:
            raise ValueError(f"{schema_version} task training must use {PRODUCT_LOCAL_LOSS_MODE}")
        expected_selection = (
            BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION
            if is_broadband_micro_doppler_schema(schema_version)
            else PRODUCT_LOCAL_CHECKPOINT_SELECTION
        )
        if train.checkpoint_selection_metric != expected_selection:
            raise ValueError(
                f"{schema_version} checkpoint selection must use {expected_selection}"
            )
        expected_target = 0.90 if spec.name == "cu_specialist" else 0.95
        if (
            train.task_min_precision_recall_target is None
            or abs(float(train.task_min_precision_recall_target) - expected_target) > 1.0e-12
        ):
            raise ValueError(
                f"{schema_version} {spec.name} minimum precision/recall target must be {expected_target:.2f}"
            )
        if not 0.0 <= train.prefix_auxiliary_weight <= 1.0:
            raise ValueError(f"{schema_version} prefix_auxiliary_weight must be in [0,1]")
        if train.prefix_auxiliary_weight > 0.0 and train.prefix_auxiliary_min_events < 3:
            raise ValueError(f"{schema_version} prefix auxiliary supervision must start at event 3 or later")
        if model.dt_reason_mode != DT_REASON_DYNAMICS_MODE:
            raise ValueError(f"{schema_version} model must use dt_reason_mode={DT_REASON_DYNAMICS_MODE}")
        expected_index_schema = (
            PRODUCT_LOCAL_SCHEMA_VERSION
            if (is_v17_micro or is_v20_dynamics)
            else schema_version
        )
        expected_contract_schema = (
            BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
            if is_v18r_stable8_density_schema(schema_version)
            else (PRODUCT_LOCAL_SCHEMA_VERSION if is_v20_dynamics else schema_version)
        )
        if (
            data.contract_schema_version != expected_contract_schema
            or data.index_schema_version != expected_index_schema
        ):
            raise ValueError(
                f"{schema_version} must consume its approved index/data contract versions"
            )
        if model.dbb_auxiliary_num_classes != 0 or motion_fields_present:
            raise ValueError(f"{schema_version} local task must not retain auxiliary or TR-motion branches")
        if not is_v17_micro and (model.rd_physical_dim != 0 or model.micro_doppler_dim != 0):
            raise ValueError("RADAR-LN3-V16 local baseline must not retain experimental physical inputs")
        if (is_v17_micro or is_v20_dynamics) and spec.name != "dbb_specialist":
            raise ValueError(
                f"{schema_version} is a DBB-specialist-only paired experiment"
            )
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError(f"{schema_version} requires 20 fixed epochs with early stopping disabled")
        if train.epoch_checkpoint_dir is None or not train.record_parent_order_hash or not train.save_every_epoch:
            raise ValueError(f"{schema_version} requires epoch checkpoints and parent-order hashes")
        if train.dataloader_seed != train.seed:
            raise ValueError(f"{schema_version} model seed and DataLoader seed must match")
        forbidden_product_fields = (
            train.joint_artifact_dir,
            train.joint_last_checkpoint_path,
            train.joint_best_checkpoint_path,
            train.joint_checkpoint_schema,
            train.gate_min_precision_recall_target,
            train.dbb_min_precision_recall_target,
            train.cu_min_precision_recall_target,
            train.drone_bird_min_precision_recall_target,
        )
        if any(value is not None for value in forbidden_product_fields):
            raise ValueError(f"{schema_version} local task is not joint product training")
        if not (is_v17_micro or is_v20_dynamics) and any(
            value is not None
            for value in (
                train.paired_protocol,
                train.paired_arm,
                train.paired_seed,
                train.paired_counterpart_config_path,
            )
        ):
            raise ValueError("RADAR-LN3-V16 is three independent local runs, not paired training")
        if (
            model.dbb_readout_mode != DBB_READOUT_LINEAR
            or model.dbb_embedding_dim != 0
            or model.dbb_cosine_scale != 1.0
            or model.dbb_angular_margin != 0.0
            or model.dbb_knownness_score_name != DBB_KNOWNNESS_PROBABILITY
        ):
            raise ValueError(f"{schema_version} must use the frozen linear local-task readout")
    elif train.loss_mode != "causal_prefix":
        raise ValueError("historical task training must use causal_prefix loss")
    if schema_number == _task_schema_number(DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION):
        if train.checkpoint_selection_metric != "parent_macro_f1":
            raise ValueError("RADAR-LN3-V7 must preserve parent_macro_f1 checkpoint selection")
    if (
        schema_number >= _task_schema_number(WEAK_CLASS_SELECTION_SCHEMA_VERSION)
        and schema_version not in HIERARCHICAL_SCHEMA_VERSIONS
        and not is_broadband_micro_doppler_schema(schema_version)
    ):
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
    elif is_v18r_stable8_density_schema(schema_version):
        if (
            train.paired_protocol != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
            or train.paired_arm not in {"control", "treatment"}
        ):
            raise ValueError("V18r requires the frozen stable8 density paired protocol")
        expected_mode = (
            "bounded_observation"
            if train.paired_arm == "control"
            else "density_weighted"
        )
        if model.liquid_invalid_dt_update_mode != expected_mode:
            raise ValueError(
                f"V18r {train.paired_arm} requires "
                f"liquid_invalid_dt_update_mode={expected_mode}"
            )
        if model.micro_doppler_dim != BROADBAND_MICRO_DOPPLER_VALUE_DIM:
            raise ValueError("V18r both arms require the frozen stable8 input")
        if train.paired_seed != train.seed or train.dataloader_seed != train.seed:
            raise ValueError(
                "V18r paired seed, model seed and DataLoader seed must be identical"
            )
        if train.seed != 42:
            raise ValueError("V18r fast paired screening is frozen to seed 42")
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError(
                "V18r paired training requires exactly 20 epochs with early stopping disabled"
            )
        if (
            train.epoch_checkpoint_dir is None
            or not train.record_parent_order_hash
            or not train.save_every_epoch
        ):
            raise ValueError(
                "V18r paired training requires epoch checkpoints and parent-order hashes"
            )
        if train.paired_counterpart_config_path is None:
            raise ValueError("V18r paired training requires an explicit counterpart config")
        if train.cuda_deterministic_algorithms:
            raise ValueError(
                "V18r fixes initialization/sampling only and does not claim strict CUDA determinism"
            )
    elif is_micro_doppler_paired_schema(schema_version):
        expected_arm = "treatment" if model.micro_doppler_dim else "control"
        expected_protocol = (
            BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
            if is_broadband_micro_doppler_schema(schema_version)
            else MICRO_DOPPLER_PAIRED_PROTOCOL
        )
        if (
            train.paired_protocol != expected_protocol
            or train.paired_arm != expected_arm
        ):
            raise ValueError(
                f"{schema_version} paired protocol/arm does not match the configured micro-Doppler input"
            )
        if train.paired_seed != train.seed or train.dataloader_seed != train.seed:
            raise ValueError(
                f"{schema_version} paired seed, model seed and DataLoader seed must be identical"
            )
        if train.seed != 42:
            raise ValueError(f"{schema_version} fast paired screening is frozen to seed 42")
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError(
                f"{schema_version} paired training requires exactly 20 epochs with early stopping disabled"
            )
        if train.epoch_checkpoint_dir is None or not train.record_parent_order_hash or not train.save_every_epoch:
            raise ValueError(
                f"{schema_version} paired training requires epoch checkpoints and parent-order hashes"
            )
        if train.paired_counterpart_config_path is None:
            raise ValueError(
                f"{schema_version} paired training requires an explicit counterpart config path"
            )
        if train.cuda_deterministic_algorithms:
            raise ValueError(
                f"{schema_version} fixes initialization/sampling only and must not claim strict CUDA determinism"
            )
    elif is_liquid_dynamics_paired_schema(schema_version):
        if train.paired_protocol not in V20_PAIRED_PROTOCOLS:
            raise ValueError(
                "V20 paired protocol must be V20 skip or V20r density-weighted"
            )
        if train.paired_arm not in {"control", "treatment"}:
            raise ValueError("V20 paired arm must be control or treatment")
        expected_mode = (
            "bounded_observation"
            if train.paired_arm == "control"
            else (
                "skip"
                if train.paired_protocol == V20_SKIP_PAIRED_PROTOCOL
                else "density_weighted"
            )
        )
        if model.liquid_invalid_dt_update_mode != expected_mode:
            raise ValueError(
                f"V20 {train.paired_arm} requires liquid_invalid_dt_update_mode="
                f"{expected_mode}"
            )
        if train.paired_seed != train.seed or train.dataloader_seed != train.seed:
            raise ValueError("V20 paired seed, model seed and DataLoader seed must be identical")
        allowed_v20_seeds = (
            {42}
            if train.paired_protocol == V20_SKIP_PAIRED_PROTOCOL
            else {42, 43, 44}
        )
        if train.seed not in allowed_v20_seeds:
            raise ValueError(
                "V20 skip screening is frozen to seed 42; V20r density "
                "confirmation is frozen to seeds 42/43/44"
            )
        if train.epochs != 20 or train.early_stopping_patience != 0:
            raise ValueError("V20 paired training requires exactly 20 epochs with early stopping disabled")
        if train.epoch_checkpoint_dir is None or not train.record_parent_order_hash or not train.save_every_epoch:
            raise ValueError("V20 paired training requires epoch checkpoints and parent-order hashes")
        if train.paired_counterpart_config_path is None:
            raise ValueError("V20 paired training requires an explicit counterpart config path")
        if train.cuda_deterministic_algorithms:
            raise ValueError("V20 fixes initialization/sampling only and does not claim strict CUDA determinism")
    elif (
        train.dbb_auxiliary_loss_weight != 0.0
        or train.dbb_auxiliary_class_weights
        or train.dbb_auxiliary_supervision_scope is not None
    ):
        raise ValueError("DBB auxiliary training fields are reserved for the RADAR-LN3-V12 paired experiment")
    if schema_version not in HIERARCHICAL_SCHEMA_VERSIONS and not is_product_local_schema(schema_version):
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
            or train.task_min_precision_recall_target is not None
        ):
            raise ValueError("dt reason dynamics and prefix auxiliary weight are reserved for hierarchical schemas")
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


class AngularPrototypeReadout(nn.Module):
    """Label-free cosine inference head with a separately applied train margin."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        class_count: int,
        *,
        scale: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.embedding_dim = int(embedding_dim)
        self.class_count = int(class_count)
        self.scale = float(scale)
        if self.input_dim <= 0 or self.embedding_dim <= 0 or self.class_count < 2:
            raise ValueError("angular prototype dimensions must be positive")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("angular prototype scale must be finite and positive")
        self.normalization = nn.LayerNorm(self.input_dim)
        self.embedding = nn.Linear(self.input_dim, self.embedding_dim, bias=False)
        self.prototypes = nn.Parameter(torch.empty(self.class_count, self.embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward_with_scores(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if value.shape[-1] != self.input_dim:
            raise ValueError("angular prototype readout received an invalid input dimension")
        embedding = nn.functional.normalize(
            self.embedding(self.normalization(value)),
            p=2.0,
            dim=-1,
            eps=1.0e-6,
        )
        prototypes = nn.functional.normalize(self.prototypes, p=2.0, dim=-1, eps=1.0e-6)
        cosine = nn.functional.linear(embedding, prototypes).clamp(-1.0, 1.0)
        logits = cosine * self.scale
        if not torch.isfinite(logits).all():
            raise ValueError("angular prototype readout produced non-finite values")
        return logits, cosine, embedding

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.forward_with_scores(value)
        return logits


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
        self.micro_doppler_dim = model.micro_doppler_dim
        self.dbb_auxiliary_num_classes = model.dbb_auxiliary_num_classes
        self.dt_reason_mode = model.dt_reason_mode
        self.dbb_readout_mode = model.dbb_readout_mode
        self.dbb_embedding_dim = model.dbb_embedding_dim
        self.dbb_cosine_scale = model.dbb_cosine_scale
        self.dbb_angular_margin = model.dbb_angular_margin
        self.dbb_knownness_score_name = model.dbb_knownness_score_name
        self.rd_physical_schema = config.data.rd_physical_schema
        self.rd_physical_mask_mode = config.data.rd_physical_mask_mode
        self.micro_doppler_schema = config.data.micro_doppler_schema
        self.micro_doppler_mask_mode = config.data.micro_doppler_mask_mode
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
            + self.micro_doppler_dim
        )
        self.fusion_input_dim = fusion_dim
        fusion_linear: nn.Module
        if config.schema_version in {
            RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION,
            MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
            BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
            V18R_STABLE8_DENSITY_SCHEMA_VERSION,
        }:
            fusion_linear = PairedFusionLinear(
                fusion_dim,
                model.liquid_hidden_dim,
                base_input_dim=(
                    fusion_dim - self.rd_physical_dim
                    if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION
                    else fusion_dim - self.micro_doppler_dim
                ),
            )
        else:
            fusion_linear = nn.Linear(fusion_dim, model.liquid_hidden_dim)
        self.fusion = nn.Sequential(
            fusion_linear,
            nn.LayerNorm(model.liquid_hidden_dim),
            nn.SiLU(),
        )
        self.liquid = build_cfc_liquid_encoder(
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
        if self.dbb_readout_mode == DBB_READOUT_ANGULAR_PROTOTYPE:
            if self.task_name != "dbb_specialist":
                raise ValueError("angular prototype readout is restricted to the DBB specialist")
            self.classifier = AngularPrototypeReadout(
                model.liquid_hidden_dim,
                self.dbb_embedding_dim,
                self.num_classes,
                scale=self.dbb_cosine_scale,
            )
        else:
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

    def _classification_readout(
        self,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if isinstance(self.classifier, AngularPrototypeReadout):
            return self.classifier.forward_with_scores(value)
        logits = self.classifier(value)
        knownness_scores = (
            torch.softmax(logits, dim=-1) if self.task_name == "dbb_specialist" else None
        )
        return logits, knownness_scores, None

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
            raise ValueError(
                "online RD physical extraction requires a configured four-value RD physical input"
            )
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

    def prepare_online_micro_doppler(
        self,
        frames: Sequence[Any],
        raw_frame_indices: Iterable[int],
        *,
        track_point_index: int,
    ) -> dict[str, torch.Tensor]:
        """Run the frozen V17 current-event raw-I/Q extractor online."""

        if is_broadband_micro_doppler_schema(self.config.schema_version):
            if self.micro_doppler_dim != BROADBAND_MICRO_DOPPLER_VALUE_DIM:
                raise ValueError("online broadband micro-Doppler extraction requires V18 Treatment")
            event = extract_current_event_broadband_micro_doppler(
                frames, raw_frame_indices, track_point_index=track_point_index
            )
        else:
            if self.micro_doppler_dim != MICRO_DOPPLER_VALUE_DIM:
                raise ValueError("online micro-Doppler extraction is available only for V17 Treatment")
            event = extract_current_event_micro_doppler(
                frames,
                raw_frame_indices,
                track_point_index=track_point_index,
            )
        parameter = self.fusion[0].weight
        return {
            "micro_doppler_features": torch.as_tensor(
                event.values.copy(), dtype=parameter.dtype, device=parameter.device
            ).unsqueeze(0),
            "micro_doppler_valid": torch.tensor(
                [event.valid], dtype=torch.bool, device=parameter.device
            ),
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
        zero_ordinal: torch.Tensor | None = None,
        tr_motion_features: torch.Tensor | None = None,
        tr_motion_valid_mask: torch.Tensor | None = None,
        rd_physical_features: torch.Tensor | None = None,
        rd_physical_valid: torch.Tensor | None = None,
        micro_doppler_features: torch.Tensor | None = None,
        micro_doppler_valid: torch.Tensor | None = None,
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
                raise ValueError("reason-aware task requires tr_dt_reason with shape [batch,sequence]")
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
                raise ValueError("configured task requires RD physical values and valid mask")
            if rd_physical_features.shape != expected_physical_shape or rd_physical_valid.shape != tr_features.shape[:2]:
                raise ValueError("RD physical value/valid shape does not match the sequence")
            if rd_physical_valid.dtype != torch.bool or not torch.isfinite(rd_physical_features).all():
                raise ValueError("RD physical values must be finite and valid must be boolean")
            if bool((rd_physical_features < 0.0).any()) or bool((rd_physical_features > 1.0).any()):
                raise ValueError("RD physical values must be bounded to [0,1]")
            if bool((rd_physical_features.masked_select(~rd_physical_valid.unsqueeze(-1)) != 0).any()):
                raise ValueError("invalid RD physical values must be exactly zero")
            if bool(rd_physical_valid.masked_select(tr_padding_mask).any()):
                raise ValueError("padded events must not contain valid RD physical values")
            gated_physical = rd_physical_features * rd_physical_valid.to(dtype=tr_features.dtype).unsqueeze(-1)
            fusion_inputs.append(gated_physical)
        elif rd_physical_features is not None or rd_physical_valid is not None:
            raise ValueError("tasks without RD physical inputs must not receive RD physical tensors")
        if self.micro_doppler_dim:
            expected_micro_shape = (*tr_features.shape[:2], self.micro_doppler_dim)
            if micro_doppler_features is None or micro_doppler_valid is None:
                raise ValueError("V17 Treatment requires micro-Doppler values and valid mask")
            if (
                micro_doppler_features.shape != expected_micro_shape
                or micro_doppler_valid.shape != tr_features.shape[:2]
            ):
                raise ValueError("V17 micro-Doppler value/valid shape does not match the sequence")
            if micro_doppler_valid.dtype != torch.bool or not torch.isfinite(
                micro_doppler_features
            ).all():
                raise ValueError("V17 micro-Doppler values must be finite and valid must be boolean")
            if bool((micro_doppler_features < 0.0).any()) or bool(
                (micro_doppler_features > 1.0).any()
            ):
                raise ValueError("V17 micro-Doppler values must be bounded to [0,1]")
            if bool(
                (
                    micro_doppler_features.masked_select(
                        ~micro_doppler_valid.unsqueeze(-1)
                    )
                    != 0
                ).any()
            ):
                raise ValueError("invalid V17 micro-Doppler values must be exactly zero")
            if bool(micro_doppler_valid.masked_select(tr_padding_mask).any()):
                raise ValueError("padded V17 events must not contain valid micro-Doppler values")
            if rd_physical_valid is None or not torch.equal(
                micro_doppler_valid, rd_physical_valid
            ):
                raise ValueError(
                    "V17 RD physical and micro-Doppler valid masks must be exactly identical"
                )
            fusion_inputs.append(
                micro_doppler_features
                * micro_doppler_valid.to(dtype=tr_features.dtype).unsqueeze(-1)
            )
        elif micro_doppler_features is not None or micro_doppler_valid is not None:
            raise ValueError("V17 Control and historical tasks must not receive micro-Doppler tensors")
        if self.dt_valid_feature_mode == "fusion_and_dynamics":
            fusion_inputs.append(tr_dt_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
        point_events = self.fusion(torch.cat(fusion_inputs, dim=-1))
        liquid_state = self.liquid(
            point_events,
            tr_dt_seconds,
            tr_padding_mask,
            dt_valid=tr_dt_valid,
            dt_reason=tr_dt_reason,
            zero_ordinal=zero_ordinal,
        )
        sequence_logits, dbb_knownness_scores, dbb_embeddings = self._classification_readout(
            liquid_state.sequence
        )
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
        if self.task_name == "dbb_specialist":
            if dbb_knownness_scores is None:
                raise RuntimeError("DBB specialist did not produce knownness scores")
            result["dbb_knownness_sequence_scores"] = dbb_knownness_scores
            result["dbb_knownness_window_scores"] = dbb_knownness_scores[
                batch_indices, last_indices
            ]
            result["dbb_knownness_score_name"] = self.dbb_knownness_score_name
            if dbb_embeddings is not None:
                result["dbb_embedding_sequence"] = dbb_embeddings
                result["dbb_embedding_final"] = dbb_embeddings[batch_indices, last_indices]
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
        zero_ordinal: torch.Tensor | None = None,
        rd_frames: torch.Tensor,
        rd_frame_mask: torch.Tensor,
        rd_quality: torch.Tensor,
        liquid_state: torch.Tensor | None = None,
        tr_motion_features: torch.Tensor | None = None,
        tr_motion_valid_mask: torch.Tensor | None = None,
        rd_physical_features: torch.Tensor | None = None,
        rd_physical_valid: torch.Tensor | None = None,
        micro_doppler_features: torch.Tensor | None = None,
        micro_doppler_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if tr_features.ndim != 2 or tr_missing_mask.ndim != 2 or rd_quality.ndim != 2:
            raise ValueError("online task TR tensors must have shape [batch, features]")
        if tr_features.shape[-1] != self.tr_value_dim or tr_missing_mask.shape[-1] != self.tr_missing_dim:
            raise ValueError("online task TR feature dimensions do not match model")
        if tr_dt_seconds.shape != (tr_features.shape[0],) or tr_dt_valid.shape != (tr_features.shape[0],):
            raise ValueError("online task dt tensors must have shape [batch]")
        if self.dt_reason_mode == DT_REASON_DYNAMICS_MODE:
            if tr_dt_reason is None or tr_dt_reason.shape != tr_dt_seconds.shape:
                raise ValueError("online reason-aware task requires tr_dt_reason with shape [batch]")
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
                raise ValueError("online configured task requires RD physical values and valid mask")
            if rd_physical_features.shape != (tr_features.shape[0], self.rd_physical_dim):
                raise ValueError("online RD physical values have an invalid shape")
            if rd_physical_valid.shape != (tr_features.shape[0],) or rd_physical_valid.dtype != torch.bool:
                raise ValueError("online RD physical valid must be boolean with shape [batch]")
            if not torch.isfinite(rd_physical_features).all():
                raise ValueError("online RD physical values must be finite")
            if bool((rd_physical_features < 0.0).any()) or bool((rd_physical_features > 1.0).any()):
                raise ValueError("online RD physical values must be bounded to [0,1]")
            if bool((rd_physical_features.masked_select(~rd_physical_valid.unsqueeze(-1)) != 0).any()):
                raise ValueError("invalid online RD physical values must be exactly zero")
            fusion_inputs.append(rd_physical_features * rd_physical_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
        elif rd_physical_features is not None or rd_physical_valid is not None:
            raise ValueError("online tasks without RD physical inputs must not receive RD physical tensors")
        if self.micro_doppler_dim:
            if micro_doppler_features is None or micro_doppler_valid is None:
                raise ValueError("online V17 Treatment requires micro-Doppler values and valid mask")
            if micro_doppler_features.shape != (tr_features.shape[0], self.micro_doppler_dim):
                raise ValueError("online V17 micro-Doppler values have an invalid shape")
            if (
                micro_doppler_valid.shape != (tr_features.shape[0],)
                or micro_doppler_valid.dtype != torch.bool
            ):
                raise ValueError("online V17 micro-Doppler valid must be boolean with shape [batch]")
            if not torch.isfinite(micro_doppler_features).all():
                raise ValueError("online V17 micro-Doppler values must be finite")
            if bool((micro_doppler_features < 0.0).any()) or bool(
                (micro_doppler_features > 1.0).any()
            ):
                raise ValueError("online V17 micro-Doppler values must be bounded to [0,1]")
            if bool(
                (
                    micro_doppler_features.masked_select(
                        ~micro_doppler_valid.unsqueeze(-1)
                    )
                    != 0
                ).any()
            ):
                raise ValueError("invalid online V17 micro-Doppler values must be exactly zero")
            if rd_physical_valid is None or not torch.equal(
                micro_doppler_valid, rd_physical_valid
            ):
                raise ValueError(
                    "online V17 RD physical and micro-Doppler valid masks must be exactly identical"
                )
            fusion_inputs.append(
                micro_doppler_features
                * micro_doppler_valid.to(dtype=tr_features.dtype).unsqueeze(-1)
            )
        elif micro_doppler_features is not None or micro_doppler_valid is not None:
            raise ValueError("online V17 Control and historical tasks must not receive micro-Doppler tensors")
        if self.dt_valid_feature_mode == "fusion_and_dynamics":
            fusion_inputs.append(tr_dt_valid.to(dtype=tr_features.dtype).unsqueeze(-1))
        point_event = self.fusion(torch.cat(fusion_inputs, dim=-1))
        state, dt_clamped, dt_zero = self.liquid.step(
            point_event,
            tr_dt_seconds,
            liquid_state,
            tr_dt_valid,
            tr_dt_reason,
            zero_ordinal,
        )
        logits, dbb_knownness_scores, dbb_embeddings = self._classification_readout(
            self.liquid.readout(state)
        )
        result = {
            "window_logits": logits,
            "liquid_state": state,
            "dt_clamped_mask": dt_clamped,
            "dt_zero_mask": dt_zero,
            "dt_invalid_mask": ~tr_dt_valid,
        }
        if self.task_name == "dbb_specialist":
            if dbb_knownness_scores is None:
                raise RuntimeError("online DBB specialist did not produce knownness scores")
            result["dbb_knownness_window_scores"] = dbb_knownness_scores
            result["dbb_knownness_score_name"] = self.dbb_knownness_score_name
            if dbb_embeddings is not None:
                result["dbb_embedding"] = dbb_embeddings
        return result








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


def build_v17_paired_models(
    control_config: LiquidTaskConfig,
    treatment_config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    """Build one isolated DBB physical4 versus physical4+micro pair."""

    schema_version = control_config.schema_version
    expected_protocol = (
        BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
        if schema_version == BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        else MICRO_DOPPLER_PAIRED_PROTOCOL
    )
    intervention_dim = (
        BROADBAND_MICRO_DOPPLER_VALUE_DIM
        if schema_version == BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
        else MICRO_DOPPLER_VALUE_DIM
    )

    if (
        schema_version
        not in {
            MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
            BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        }
        or treatment_config.schema_version != schema_version
        or control_config.task.name != "dbb_specialist"
        or treatment_config.task.name != "dbb_specialist"
        or control_config.train.paired_arm != "control"
        or treatment_config.train.paired_arm != "treatment"
        or control_config.train.paired_protocol != expected_protocol
        or treatment_config.train.paired_protocol != expected_protocol
    ):
        raise ValueError("paired model builder requires V17 DBB Control and Treatment configs")
    if control_config.train.seed != treatment_config.train.seed:
        raise ValueError(f"{schema_version} paired configs must use the same seed")
    if (
        control_config.model.rd_physical_dim != RD_PHYSICAL_VALUE_DIM
        or treatment_config.model.rd_physical_dim != RD_PHYSICAL_VALUE_DIM
        or control_config.model.micro_doppler_dim != 0
        or treatment_config.model.micro_doppler_dim != intervention_dim
    ):
        raise ValueError(
            f"{schema_version} pair must compare physical4 against physical4 plus "
            f"micro-Doppler{intervention_dim}"
        )
    if (
        replace(control_config.model, micro_doppler_dim=intervention_dim)
        != treatment_config.model
    ):
        raise ValueError(
            f"{schema_version} paired model configs differ beyond micro_doppler_dim"
        )
    if control_config.data != treatment_config.data:
        raise ValueError(
            f"{schema_version} paired data configs, including shared evidence caches, must be identical"
        )
    excluded_train = {
        "checkpoint_path",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "paired_arm",
        "paired_counterpart_config_path",
    }
    control_train = {
        key: value
        for key, value in control_config.train.__dict__.items()
        if key not in excluded_train
    }
    treatment_train = {
        key: value
        for key, value in treatment_config.train.__dict__.items()
        if key not in excluded_train
    }
    if control_train != treatment_train:
        raise ValueError(
            f"{schema_version} paired training configs differ beyond arm-specific paths"
        )
    seed_everything(control_config.train.seed)
    control = LiquidTaskClassifier(control_config)
    seed_everything(treatment_config.train.seed)
    treatment = LiquidTaskClassifier(treatment_config)
    control_parameters = dict(control.named_parameters())
    treatment_parameters = dict(treatment.named_parameters())
    control_buffers = dict(control.named_buffers())
    treatment_buffers = dict(treatment.named_buffers())
    shared_names: list[str] = []
    with torch.no_grad():
        for name, control_value in control_parameters.items():
            treatment_value = treatment_parameters.get(name)
            if treatment_value is None:
                raise ValueError(f"V17 Treatment lacks shared parameter: {name}")
            if name == "fusion.0.weight":
                if (
                    treatment_value.shape[0] != control_value.shape[0]
                    or treatment_value.shape[1]
                    != control_value.shape[1] + intervention_dim
                ):
                    raise ValueError("V17 Treatment Fusion width is not Control width plus three")
                treatment_value[:, : control_value.shape[1]].copy_(control_value)
                treatment_value[:, control_value.shape[1] :].zero_()
            elif treatment_value.shape != control_value.shape:
                raise ValueError(
                    f"V17 paired model has a non-shared parameter shape outside Fusion columns: {name}"
                )
            else:
                treatment_value.copy_(control_value)
            shared_names.append(name)
        for name, control_value in control_buffers.items():
            treatment_value = treatment_buffers.get(name)
            if treatment_value is None or treatment_value.shape != control_value.shape:
                raise ValueError(f"V17 Treatment lacks compatible shared buffer: {name}")
            treatment_value.copy_(control_value)
            shared_names.append(name)
    for name, control_value in control.state_dict().items():
        treatment_value = treatment.state_dict()[name]
        comparable = (
            treatment_value[:, : control_value.shape[1]]
            if name == "fusion.0.weight"
            else treatment_value
        )
        if not torch.equal(control_value, comparable):
            raise AssertionError(f"V17 shared initialization differs at {name}")
    new_columns = treatment.state_dict()["fusion.0.weight"][:, control.fusion_input_dim :]
    if not torch.equal(new_columns, torch.zeros_like(new_columns)):
        raise AssertionError("V17 new micro-Doppler Fusion columns are not exactly zero")
    return control, treatment, {
        "paired_protocol": expected_protocol,
        "seed": control_config.train.seed,
        "shared_state_entry_count": len(shared_names),
        "shared_parameters_and_buffers_exactly_equal": True,
        "control_fusion_input_dim": control.fusion_input_dim,
        "treatment_fusion_input_dim": treatment.fusion_input_dim,
        "shared_rd_physical_columns": RD_PHYSICAL_VALUE_DIM,
        "treatment_new_fusion_columns": intervention_dim,
        "treatment_new_fusion_columns_exactly_zero": True,
    }


def build_v17_paired_model_for_config(
    project_root: Path,
    config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    counterpart_path = resolve_project_path(
        project_root, config.train.paired_counterpart_config_path or ""
    )
    counterpart = load_task_config(counterpart_path)
    if counterpart.train.paired_counterpart_config_path is None:
        raise ValueError("V17 counterpart does not reference this config")
    reverse_path = resolve_project_path(
        project_root, counterpart.train.paired_counterpart_config_path
    )
    expected_current_path = resolve_project_path(
        project_root,
        f"config/RADAR-LN3/{config.config_name}-config.json",
    )
    if reverse_path.resolve() != expected_current_path.resolve():
        raise ValueError("V17 counterpart reference is not reciprocal")
    configs = {config.train.paired_arm: config, counterpart.train.paired_arm: counterpart}
    if set(configs) != {"control", "treatment"}:
        raise ValueError("V17 config pair does not contain exactly Control and Treatment")
    control, treatment, evidence = build_v17_paired_models(
        configs["control"], configs["treatment"]
    )
    evidence.update(
        {
            "paired_arm": config.train.paired_arm,
            "counterpart_config_path": str(counterpart_path),
            "counterpart_config_name": counterpart.config_name,
        }
    )
    return (
        (control, treatment, evidence)
        if config.train.paired_arm == "control"
        else (treatment, control, evidence)
    )


def _state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_v18r_stable8_density_models(
    project_root: Path,
    control_config: LiquidTaskConfig,
    treatment_config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    """Reconstruct the frozen V18 stable8 initialization for a dynamics-only pair."""

    if (
        control_config.schema_version != V18R_STABLE8_DENSITY_SCHEMA_VERSION
        or treatment_config.schema_version != V18R_STABLE8_DENSITY_SCHEMA_VERSION
        or control_config.task.name != "dbb_specialist"
        or treatment_config.task.name != "dbb_specialist"
        or control_config.train.paired_arm != "control"
        or treatment_config.train.paired_arm != "treatment"
        or control_config.train.paired_protocol
        != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or treatment_config.train.paired_protocol
        != V18R_STABLE8_DENSITY_PAIRED_PROTOCOL
        or control_config.model.liquid_invalid_dt_update_mode
        != "bounded_observation"
        or treatment_config.model.liquid_invalid_dt_update_mode
        != "density_weighted"
    ):
        raise ValueError("V18r builder requires the stable8 bounded/density pair")
    if control_config.train.seed != treatment_config.train.seed:
        raise ValueError("V18r paired configs must use the same seed")
    if control_config.data != treatment_config.data:
        raise ValueError("V18r paired configs must reuse one frozen V18 data contract")
    if (
        control_config.model.rd_physical_dim != RD_PHYSICAL_VALUE_DIM
        or control_config.model.micro_doppler_dim
        != BROADBAND_MICRO_DOPPLER_VALUE_DIM
        or replace(
            control_config.model,
            liquid_invalid_dt_update_mode="density_weighted",
        )
        != treatment_config.model
    ):
        raise ValueError(
            "V18r model configs must be identical stable8 models except for density dynamics"
        )
    excluded_train = {
        "checkpoint_path",
        "checkpoint_schema",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "paired_protocol",
        "paired_arm",
        "paired_counterpart_config_path",
        "allow_parameter_updates",
    }
    control_train = {
        key: value
        for key, value in control_config.train.__dict__.items()
        if key not in excluded_train
    }
    treatment_train = {
        key: value
        for key, value in treatment_config.train.__dict__.items()
        if key not in excluded_train
    }
    if control_train != treatment_train:
        raise ValueError("V18r paired training configs differ beyond arm identity")
    if control_config.train.allow_parameter_updates:
        raise ValueError("V18r Control is a read-only reference and must not be trainable")
    if not treatment_config.train.allow_parameter_updates:
        raise ValueError("V18r Treatment must permit parameter updates")

    source_control_path = resolve_project_path(
        project_root, V18R_SOURCE_CONTROL_CONFIG_PATH
    )
    source_treatment_path = resolve_project_path(
        project_root, V18R_SOURCE_TREATMENT_CONFIG_PATH
    )
    source_control_config = load_task_config(source_control_path)
    source_treatment_config = load_task_config(source_treatment_path)
    source_control, source_treatment, source_evidence = build_v17_paired_models(
        source_control_config, source_treatment_config
    )
    if control_config.data != source_treatment_config.data:
        raise ValueError("V18r data config differs from the frozen V18 stable8 Treatment")
    if control_config.model != source_treatment_config.model:
        raise ValueError("V18r Control model differs from the frozen V18 stable8 Treatment")
    source_train = {
        key: value
        for key, value in source_treatment_config.train.__dict__.items()
        if key not in excluded_train
    }
    if source_train != control_train:
        raise ValueError("V18r training core differs from the frozen V18 stable8 Treatment")

    seed_everything(control_config.train.seed)
    control = LiquidTaskClassifier(control_config)
    seed_everything(treatment_config.train.seed)
    treatment = LiquidTaskClassifier(treatment_config)
    source_state = source_treatment.state_dict()
    control.load_state_dict(source_state, strict=True)
    treatment.load_state_dict(source_state, strict=True)
    control_state = control.state_dict()
    treatment_state = treatment.state_dict()
    unequal = [
        name
        for name, value in control_state.items()
        if not torch.equal(value, treatment_state[name])
    ]
    if unequal:
        raise AssertionError(f"V18r initialization differs at {unequal[:3]}")
    source_base_dim = source_control.fusion_input_dim
    stable8_columns = source_state["fusion.0.weight"][:, source_base_dim:]
    if stable8_columns.shape[1] != BROADBAND_MICRO_DOPPLER_VALUE_DIM:
        raise AssertionError("V18r source stable8 Fusion width is not eight")
    if not torch.equal(stable8_columns, torch.zeros_like(stable8_columns)):
        raise AssertionError("V18r source stable8 Fusion columns are not exactly zero")
    source_state_sha256 = _state_dict_sha256(source_state)
    if (
        _state_dict_sha256(control_state) != source_state_sha256
        or _state_dict_sha256(treatment_state) != source_state_sha256
    ):
        raise AssertionError("V18r state hash does not reproduce the V18 stable8 source")
    return control, treatment, {
        "paired_protocol": V18R_STABLE8_DENSITY_PAIRED_PROTOCOL,
        "seed": control_config.train.seed,
        "shared_state_entry_count": len(source_state),
        "shared_parameters_and_buffers_exactly_equal": True,
        "source_v18_control_config_path": str(source_control_path),
        "source_v18_treatment_config_path": str(source_treatment_path),
        "source_v18_paired_protocol": source_evidence["paired_protocol"],
        "source_v18_initial_treatment_state_sha256": source_state_sha256,
        "source_v18_control_fusion_input_dim": source_base_dim,
        "stable8_fusion_columns": BROADBAND_MICRO_DOPPLER_VALUE_DIM,
        "stable8_fusion_columns_exactly_zero": True,
        "treatment_new_fusion_columns_exactly_zero": True,
        "dynamics_control_mode": "bounded_observation",
        "dynamics_treatment_mode": "density_weighted",
    }


def build_v18r_stable8_density_model_for_config(
    project_root: Path,
    config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    counterpart_path = resolve_project_path(
        project_root, config.train.paired_counterpart_config_path or ""
    )
    counterpart = load_task_config(counterpart_path)
    if counterpart.train.paired_counterpart_config_path is None:
        raise ValueError("V18r counterpart does not reference this config")
    reverse_path = resolve_project_path(
        project_root, counterpart.train.paired_counterpart_config_path
    )
    expected_current_path = resolve_project_path(
        project_root, f"config/RADAR-LN3/{config.config_name}-config.json"
    )
    if reverse_path.resolve() != expected_current_path.resolve():
        raise ValueError("V18r counterpart reference is not reciprocal")
    configs = {config.train.paired_arm: config, counterpart.train.paired_arm: counterpart}
    if set(configs) != {"control", "treatment"}:
        raise ValueError("V18r config pair must contain Control and Treatment")
    control, treatment, evidence = build_v18r_stable8_density_models(
        project_root, configs["control"], configs["treatment"]
    )
    evidence.update(
        {
            "paired_arm": config.train.paired_arm,
            "counterpart_config_path": str(counterpart_path),
            "counterpart_config_name": counterpart.config_name,
        }
    )
    return (
        (control, treatment, evidence)
        if config.train.paired_arm == "control"
        else (treatment, control, evidence)
    )


def build_v20_paired_models(
    control_config: LiquidTaskConfig,
    treatment_config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    """Build a same-architecture pair differing only in zero-delta dynamics."""

    if (
        control_config.schema_version != LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION
        or treatment_config.schema_version != LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION
        or control_config.task.name != "dbb_specialist"
        or treatment_config.task.name != "dbb_specialist"
        or control_config.train.paired_arm != "control"
        or treatment_config.train.paired_arm != "treatment"
        or control_config.train.paired_protocol != treatment_config.train.paired_protocol
        or control_config.train.paired_protocol not in V20_PAIRED_PROTOCOLS
        or control_config.model.liquid_invalid_dt_update_mode != "bounded_observation"
        or treatment_config.model.liquid_invalid_dt_update_mode
        != (
            "skip"
            if control_config.train.paired_protocol == V20_SKIP_PAIRED_PROTOCOL
            else "density_weighted"
        )
    ):
        raise ValueError("V20 paired model builder requires a valid Control/Treatment dynamics pair")
    if control_config.train.seed != treatment_config.train.seed:
        raise ValueError("V20 paired configs must use the same seed")
    if control_config.data != treatment_config.data:
        raise ValueError("V20 paired configs must reuse one frozen data contract")
    if replace(
        control_config.model,
        liquid_invalid_dt_update_mode=treatment_config.model.liquid_invalid_dt_update_mode,
    ) != treatment_config.model:
        raise ValueError("V20 paired model configs differ beyond invalid_dt_update_mode")
    excluded_train = {
        "checkpoint_path",
        "checkpoint_schema",
        "last_checkpoint_path",
        "formal_artifact_dir",
        "epoch_checkpoint_dir",
        "paired_arm",
        "paired_counterpart_config_path",
    }
    control_train = {
        key: value
        for key, value in control_config.train.__dict__.items()
        if key not in excluded_train
    }
    treatment_train = {
        key: value
        for key, value in treatment_config.train.__dict__.items()
        if key not in excluded_train
    }
    if control_train != treatment_train:
        raise ValueError("V20 paired training configs differ beyond arm-specific paths")
    seed_everything(control_config.train.seed)
    control = LiquidTaskClassifier(control_config)
    seed_everything(treatment_config.train.seed)
    treatment = LiquidTaskClassifier(treatment_config)
    treatment.load_state_dict(control.state_dict(), strict=True)
    if not all(
        torch.equal(control_value, treatment.state_dict()[name])
        for name, control_value in control.state_dict().items()
    ):
        raise AssertionError("V20 Control/Treatment initialization is not exactly equal")
    return control, treatment, {
        "paired_protocol": control_config.train.paired_protocol,
        "seed": control_config.train.seed,
        "shared_state_entry_count": len(control.state_dict()),
        "shared_parameters_and_buffers_exactly_equal": True,
        "treatment_new_fusion_columns_exactly_zero": True,
        "dynamics_control_mode": control_config.model.liquid_invalid_dt_update_mode,
        "dynamics_treatment_mode": treatment_config.model.liquid_invalid_dt_update_mode,
    }


def build_v20_paired_model_for_config(
    project_root: Path,
    config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier, dict[str, Any]]:
    counterpart_path = resolve_project_path(
        project_root, config.train.paired_counterpart_config_path or ""
    )
    counterpart = load_task_config(counterpart_path)
    if counterpart.train.paired_counterpart_config_path is None:
        raise ValueError("V20 counterpart does not reference this config")
    reverse_path = resolve_project_path(
        project_root, counterpart.train.paired_counterpart_config_path
    )
    expected_current_path = resolve_project_path(
        project_root, f"config/RADAR-LN3/{config.config_name}-config.json"
    )
    if reverse_path.resolve() != expected_current_path.resolve():
        raise ValueError("V20 counterpart reference is not reciprocal")
    configs = {config.train.paired_arm: config, counterpart.train.paired_arm: counterpart}
    if set(configs) != {"control", "treatment"}:
        raise ValueError("V20 config pair does not contain exactly Control and Treatment")
    control, treatment, evidence = build_v20_paired_models(
        configs["control"], configs["treatment"]
    )
    evidence.update(
        {
            "paired_arm": config.train.paired_arm,
            "counterpart_config_path": str(counterpart_path),
            "counterpart_config_name": counterpart.config_name,
        }
    )
    return (
        (control, treatment, evidence)
        if config.train.paired_arm == "control"
        else (treatment, control, evidence)
    )


def build_paired_model_for_config(
    project_root: Path,
    config: LiquidTaskConfig,
) -> tuple[LiquidTaskClassifier, LiquidTaskClassifier | None, dict[str, Any]]:
    if config.schema_version == RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION:
        return build_v11_paired_model_for_config(project_root, config)
    if is_v18r_stable8_density_schema(config.schema_version):
        return build_v18r_stable8_density_model_for_config(project_root, config)
    if is_micro_doppler_paired_schema(config.schema_version):
        return build_v17_paired_model_for_config(project_root, config)
    if is_liquid_dynamics_paired_schema(config.schema_version):
        return build_v20_paired_model_for_config(project_root, config)
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
    if requires_task_config_identity(model.config.schema_version):
        if payload.get("config_name") != model.config.config_name:
            raise ValueError("versioned task checkpoint belongs to a different named configuration")
        expected_config_identity = task_config_identity_sha256(model.config)
        if payload.get("config_identity_sha256") != expected_config_identity:
            raise ValueError("versioned task checkpoint configuration identity mismatch")
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
    if is_micro_doppler_paired_schema(model.config.schema_version):
        expected_data_contract.update(
            {
                "evidence_manifest_sha256": model.config.data.expected_micro_doppler_cache_manifest_sha256,
                "train_rd_physical_cache_sha256": model.config.data.expected_train_rd_physical_cache_sha256,
                "validation_rd_physical_cache_sha256": model.config.data.expected_validation_rd_physical_cache_sha256,
                "train_micro_doppler_cache_sha256": model.config.data.expected_train_micro_doppler_cache_sha256,
                "validation_micro_doppler_cache_sha256": model.config.data.expected_validation_micro_doppler_cache_sha256,
                "train_micro_doppler_stft_cache_sha256": model.config.data.expected_train_micro_doppler_stft_cache_sha256,
                "validation_micro_doppler_stft_cache_sha256": model.config.data.expected_validation_micro_doppler_stft_cache_sha256,
                "micro_doppler_frame_provenance_sha256": model.config.data.expected_micro_doppler_frame_provenance_sha256,
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
    if model.config.schema_version in HIERARCHICAL_SCHEMA_VERSIONS:
        cu_output_semantics = (
            list(STRICT_OPEN_SET_CU_OUTPUT_NAMES)
            if is_strict_open_set_schema(model.config.schema_version)
            else list(HIERARCHICAL_CU_OUTPUT_NAMES)
        )
        expected_contract.update({
            "dt_reason_mode": model.config.model.dt_reason_mode,
            "dt_reason_names": list(DT_REASON_NAMES),
            "loss_mode": model.config.train.loss_mode,
            "prefix_auxiliary_weight": model.config.train.prefix_auxiliary_weight,
            "prefix_auxiliary_min_events": model.config.train.prefix_auxiliary_min_events,
            "checkpoint_selection_metric": model.config.train.checkpoint_selection_metric,
            "cu_output_semantics": cu_output_semantics,
            "product_selection_targets": {
                "gate_min_precision_recall": model.config.train.gate_min_precision_recall_target,
                "dbb_min_precision_recall": model.config.train.dbb_min_precision_recall_target,
                "cu_min_precision_recall": model.config.train.cu_min_precision_recall_target,
                "drone_bird_min_precision_recall": model.config.train.drone_bird_min_precision_recall_target,
            },
        })
        if is_strict_open_set_schema(model.config.schema_version):
            expected_contract.update({
                "gate_output_semantics": list(STRICT_OPEN_SET_GATE_OUTPUT_NAMES),
                "known_training_labels": list(KNOWN_BASE_LABELS),
                "unknown_training_exposure": 0,
                "unknown_is_model_class": False,
                "unknown_is_rejection_action_only": True,
            })
        if model.config.schema_version == ANGULAR_KNOWNNESS_SCHEMA_VERSION:
            expected_contract.update({
                "dbb_readout_mode": model.config.model.dbb_readout_mode,
                "dbb_embedding_dim": model.config.model.dbb_embedding_dim,
                "dbb_cosine_scale": model.config.model.dbb_cosine_scale,
                "dbb_angular_margin": model.config.model.dbb_angular_margin,
                "dbb_knownness_score_name": model.config.model.dbb_knownness_score_name,
                "paired_protocol": model.config.train.paired_protocol,
                "paired_arm": model.config.train.paired_arm,
                "paired_seed": model.config.train.paired_seed,
            })
    elif is_product_local_schema(model.config.schema_version):
        expected_contract.update(
            {
                "dt_reason_mode": model.config.model.dt_reason_mode,
                "dt_reason_names": list(DT_REASON_NAMES),
                "loss_mode": model.config.train.loss_mode,
                "prefix_auxiliary_weight": model.config.train.prefix_auxiliary_weight,
                "prefix_auxiliary_min_events": model.config.train.prefix_auxiliary_min_events,
                "checkpoint_selection_metric": model.config.train.checkpoint_selection_metric,
                "task_min_precision_recall_target": model.config.train.task_min_precision_recall_target,
                "local_task_only": True,
                "full_cascade_evaluation": "not_executed",
            }
        )
        if (
            is_liquid_dynamics_paired_schema(model.config.schema_version)
            or is_v18r_stable8_density_schema(model.config.schema_version)
        ):
            update_mode = model.config.model.liquid_invalid_dt_update_mode
            expected_contract.update(
                {
                    "paired_protocol": model.config.train.paired_protocol,
                    "paired_arm": model.config.train.paired_arm,
                    "paired_seed": model.config.train.paired_seed,
                    "invalid_dt_update_mode": update_mode,
                    "zero_delta_state_semantics": (
                        "alpha_zero_event_information_removed_but_recurrent_output_norm_may_transform_state"
                        if update_mode == "skip"
                        else (
                            "causal_same_second_ordinal_gain_min_0_25k_0_75"
                            if update_mode == "density_weighted"
                            else "bounded_observation_gain_with_reason_aware_zero_delta"
                        )
                    ),
                }
            )
            if is_v18r_stable8_density_schema(model.config.schema_version):
                expected_contract.update(
                    {
                        "source_feature_schema": (
                            BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
                        ),
                        "source_feature_protocol": (
                            BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
                        ),
                        "stable8_input_shared_by_both_arms": True,
                    }
                )
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
    if is_micro_doppler_paired_schema(model.config.schema_version):
        complete_evidence_input_field = (
            "complete_evidence_profiles_are_model_input"
            if is_broadband_micro_doppler_schema(model.config.schema_version)
            else "complete_stft_tensor_is_model_input"
        )
        expected_contract.update(
            {
                "rd_physical_dim": model.rd_physical_dim,
                "rd_physical_schema": model.rd_physical_schema,
                "rd_physical_names": list(RD_PHYSICAL_NAMES),
                "rd_physical_mask_mode": model.rd_physical_mask_mode,
                "micro_doppler_dim": model.micro_doppler_dim,
                "micro_doppler_schema": model.micro_doppler_schema,
                "micro_doppler_names": (
                    list(micro_doppler_names_for_config(model.config))
                ),
                "micro_doppler_mask_mode": model.micro_doppler_mask_mode,
                complete_evidence_input_field: False,
                "legacy_mat_list_ordinal_is_raw_dat_frame_index": False,
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
    if model.config.schema_version in FORMAL_PAIRED_SCHEMA_VERSIONS:
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






def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))










__all__ = ["BASE_LABELS", "DT_VALID_DYNAMICS_ONLY_SCHEMA_VERSION", "DT_VALID_FEATURE_MODES", "LiquidTaskClassifier", "LiquidTaskConfig", "REPAIRED_TASK_SCHEMA_VERSION", "TASK_SCHEMA_VERSION", "TASK_SPECS", "TR_MOTION_GATE_SCHEMA_VERSION", "TR_OBSERVATION_INCREMENT_GATE_SCHEMA_VERSION", "RD_PHYSICAL_PAIRED_GATE_SCHEMA_VERSION", "RD_PHYSICAL_PAIRED_PROTOCOL", "DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION", "DBB_AUXILIARY_PAIRED_PROTOCOL", "DBB_AUXILIARY_CLASS_NAMES", "DBB_AUXILIARY_TREATMENT_WEIGHT", "DBB_AUXILIARY_CLASS_WEIGHTS", "DBB_AUXILIARY_SUPERVISION_SCOPE", "BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION", "BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL", "BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION", "V18R_STABLE8_DENSITY_SCHEMA_VERSION", "V18R_STABLE8_DENSITY_PAIRED_PROTOCOL", "V18R_SOURCE_CONTROL_CONFIG_PATH", "V18R_SOURCE_TREATMENT_CONFIG_PATH", "LIQUID_DYNAMICS_PAIRED_DBB_SCHEMA_VERSION", "V20_SKIP_PAIRED_PROTOCOL", "V20_DENSITY_PAIRED_PROTOCOL", "V20_PAIRED_PROTOCOLS", "FORMAL_PAIRED_SCHEMA_VERSIONS", "HIERARCHICAL_CASCADE_SCHEMA_VERSION", "HIERARCHICAL_SCHEMA_VERSIONS", "HIERARCHICAL_CHECKPOINT_SELECTION", "HIERARCHICAL_CU_OUTPUT_NAMES", "HIERARCHICAL_LOSS_MODE", "KNOWN_BASE_LABELS", "STRICT_OPEN_SET_CHECKPOINT_SELECTION", "STRICT_OPEN_SET_CU_OUTPUT_NAMES", "STRICT_OPEN_SET_GATE_OUTPUT_NAMES", "STRICT_OPEN_SET_LOSS_MODE", "STRICT_OPEN_SET_SCHEMA_VERSION", "STRICT_OPEN_SET_TASK_SPECS", "PRODUCT_LOCAL_CHECKPOINT_SELECTION", "PRODUCT_LOCAL_LOSS_MODE", "PRODUCT_LOCAL_SCHEMA_VERSION", "UNKNOWN_BASE_LABEL", "DT_REASON_DYNAMICS_MODE", "WEAK_CLASS_SELECTION_SCHEMA_VERSION", "TaskSpec", "build_v11_paired_model_for_config", "build_v11_paired_models", "build_v12_paired_models", "build_v18r_stable8_density_models", "build_v18r_stable8_density_model_for_config", "build_v20_paired_models", "build_v20_paired_model_for_config", "build_paired_model_for_config", "is_product_local_schema", "is_broadband_micro_doppler_schema", "is_liquid_dynamics_paired_schema", "is_micro_doppler_paired_schema", "is_v18r_stable8_density_schema", "micro_doppler_names_for_config", "is_strict_open_set_schema", "load_task_checkpoint", "load_task_config", "seed_everything", "task_spec", "task_config_identity_sha256"]