from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from src.core.expert1_config import Expert1TrainConfig
from src.core.features import track_sequence_feature_dim
from src.core.physical import physical_feature_dim
from src.core.expert_other_config import ExpertOtherTrainConfig
from src.core.pipeline_config import MainPipelineConfig
from src.core.pretrain_track_config import TrackPretrainConfig
from src.core.router_train_config import RouterTrainConfig
from src.core.system_config import SystemTrainConfig
from src.core.trajectory import TEMPORAL_ENCODING_MODES, TEMPORAL_POOLING_MODES


T = TypeVar("T")


def load_config_file(path: str | Path = "config/config.json") -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def save_config_file(payload: dict[str, Any], path: str | Path = "config/config.json") -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def update_section(path: str | Path, section_name: str, values: dict[str, Any]) -> dict[str, Any]:
    payload = load_config_file(path)
    section = dict(payload.get(section_name, {}))
    section.update(values)
    payload[section_name] = section
    save_config_file(payload, path)
    return payload


def _project_defaults(payload: dict[str, Any]) -> dict[str, Any]:
    project = payload.get("project", {})
    return {
        "data_root": project.get("data_root"),
        "device": project.get("device"),
        "num_workers": project.get("num_workers"),
        "seed": project.get("seed"),
    }


def _clean_section(payload: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = dict(payload.get(section_name, {}))
    for key, value in _project_defaults(payload).items():
        if value is not None:
            section.setdefault(key, value)
    section.pop("candidate_overrides", None)
    section.pop("checkpoint_policy", None)
    section.pop("negative_classes", None)
    section.pop("shared_encoder_priority", None)
    section.pop("do_not_override_shared_encoder_from", None)
    return section


def _make_dataclass(
    config_type: type[T],
    values: dict[str, Any],
    *,
    reject_unknown: bool = False,
) -> T:
    if not is_dataclass(config_type):
        raise TypeError(f"config_type must be a dataclass: {config_type}")
    allowed = {item.name for item in fields(config_type)}
    unknown = sorted(set(values) - allowed)
    if reject_unknown and unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"unknown {config_type.__name__} configuration field(s): {joined}")
    return config_type(**{key: value for key, value in values.items() if key in allowed})


def load_expert1_config(path: str | Path = "config/config.json") -> Expert1TrainConfig:
    payload = load_config_file(path)
    values = _clean_section(payload, "expert1")
    legacy_key_map = {
        "min_rd_macro_f1_for_best": "min_rd_evidence_macro_f1_for_best",
    }
    for old_key, new_key in legacy_key_map.items():
        if old_key in values and new_key not in values:
            values[new_key] = values[old_key]
        values.pop(old_key, None)
    values.pop("min_rd_net_gain_rate_for_best", None)
    values.pop("max_rd_harm_rate_for_best", None)
    config = _make_dataclass(Expert1TrainConfig, values, reject_unknown=True)
    requested_sources = {str(value).strip() for value in config.required_tr_only_sources}
    if not requested_sources or not requested_sources.issubset({"cq08_track", "raw_tr_rd"}):
        raise ValueError(
            "required_tr_only_sources must be a non-empty subset of ['cq08_track', 'raw_tr_rd']"
        )
    if config.is_pure_tr_only and config.prefer_stage2_checkpoint:
        raise ValueError(
            "pure TR-only config requires prefer_stage2_checkpoint=false; "
            "otherwise no Stage-1 checkpoint can satisfy the Stage-2 preference"
        )
    if config.pretrained_track_checkpoint and config.trajectory_warm_start_checkpoint:
        raise ValueError(
            "pretrained_track_checkpoint and trajectory_warm_start_checkpoint are mutually exclusive; "
            "use one initialization contract per Expert1 run"
        )
    if config.tr_only_parent_macro_f1_selection_weight < 0.0:
        raise ValueError("tr_only_parent_macro_f1_selection_weight must be non-negative")
    if not 0.0 <= config.parent_window_balance_power <= 1.0:
        raise ValueError("parent_window_balance_power must be within [0, 1]")
    supported_physical_contracts = {"legacy_v2", "source_robust_v3", "source_robust_v4"}
    if config.physical_representation_contract_version not in supported_physical_contracts:
        raise ValueError(
            "unsupported physical_representation_contract_version: "
            f"{config.physical_representation_contract_version}"
        )
    if (
        config.physical_representation_contract_version in {"source_robust_v3", "source_robust_v4"}
        and not config.source_normalization
    ):
        raise ValueError(
            "source_robust_v3 physical representation requires source_normalization=true"
        )
    if str(config.physical_statistics_weighting).strip().lower() not in {
        "window_v1",
        "parent_equal_v1",
    }:
        raise ValueError(
            "unsupported physical_statistics_weighting: "
            f"{config.physical_statistics_weighting!r}"
        )
    track_sequence_feature_dim(config.trajectory_feature_schema)
    physical_feature_dim(config.trajectory_feature_schema)
    if str(config.temporal_encoding_mode).strip().lower() not in TEMPORAL_ENCODING_MODES:
        raise ValueError(
            "unsupported temporal encoding mode: "
            f"{config.temporal_encoding_mode!r}; expected one of {sorted(TEMPORAL_ENCODING_MODES)}"
        )
    if str(config.temporal_pooling_mode).strip().lower() not in TEMPORAL_POOLING_MODES:
        raise ValueError(
            "unsupported temporal pooling mode: "
            f"{config.temporal_pooling_mode!r}; expected one of {sorted(TEMPORAL_POOLING_MODES)}"
        )
    return config


def load_expert2_config(path: str | Path = "config/config.json") -> ExpertOtherTrainConfig:
    payload = load_config_file(path)
    values = _clean_section(payload, "expert2")
    values["expert_name"] = "expert2_air"
    values.setdefault("output_dir", "artifacts/expert2_air")
    return _make_dataclass(ExpertOtherTrainConfig, values)


def load_expert4_config(path: str | Path = "config/config.json") -> ExpertOtherTrainConfig:
    payload = load_config_file(path)
    values = _clean_section(payload, "expert4")
    values["expert_name"] = "expert4_ground"
    values.setdefault("output_dir", "artifacts/expert4_ground")
    return _make_dataclass(ExpertOtherTrainConfig, values)


def load_system_config(path: str | Path = "config/config.json") -> SystemTrainConfig:
    payload = load_config_file(path)
    config = _make_dataclass(SystemTrainConfig, _clean_section(payload, "system"))
    track_sequence_feature_dim(config.trajectory_feature_schema)
    physical_feature_dim(config.trajectory_feature_schema)
    supported_physical_contracts = {"legacy_v2", "source_robust_v3", "source_robust_v4"}
    if config.physical_representation_contract_version not in supported_physical_contracts:
        raise ValueError(
            "unsupported system physical_representation_contract_version: "
            f"{config.physical_representation_contract_version}"
        )
    if (
        config.physical_representation_contract_version in {"source_robust_v3", "source_robust_v4"}
        and not config.source_normalization
    ):
        raise ValueError(
            "system source_robust_v3 physical representation requires source_normalization=true"
        )
    if str(config.temporal_encoding_mode).strip().lower() not in TEMPORAL_ENCODING_MODES:
        raise ValueError(
            "unsupported system temporal encoding mode: "
            f"{config.temporal_encoding_mode!r}; expected one of {sorted(TEMPORAL_ENCODING_MODES)}"
        )
    if str(config.temporal_pooling_mode).strip().lower() not in TEMPORAL_POOLING_MODES:
        raise ValueError(
            "unsupported system temporal pooling mode: "
            f"{config.temporal_pooling_mode!r}; expected one of {sorted(TEMPORAL_POOLING_MODES)}"
        )
    return config


def load_router_config(path: str | Path = "config/config.json") -> RouterTrainConfig:
    payload = load_config_file(path)
    return _make_dataclass(RouterTrainConfig, _clean_section(payload, "router"))


def load_pretrain_track_config(path: str | Path = "config/config.json") -> TrackPretrainConfig:
    payload = load_config_file(path)
    config = _make_dataclass(
        TrackPretrainConfig,
        _clean_section(payload, "pretrain_track"),
        reject_unknown=True,
    )
    requested_sources = {str(value).strip() for value in config.allowed_source_names}
    if not requested_sources or not requested_sources.issubset({"cq08_track", "raw_tr_rd"}):
        raise ValueError(
            "allowed_source_names must be a non-empty subset of ['cq08_track', 'raw_tr_rd']"
        )
    if str(config.sampling_mode).strip().lower() not in {"full_track", "causal_window"}:
        raise ValueError(
            "unsupported pretrain sampling_mode: "
            f"{config.sampling_mode!r}; expected 'full_track' or 'causal_window'"
        )
    supported_source_statistics_weighting = {"point_weighted_v1", "parent_equal_v1"}
    if str(config.source_statistics_weighting).strip().lower() not in supported_source_statistics_weighting:
        raise ValueError(
            "unsupported pretrain source_statistics_weighting: "
            f"{config.source_statistics_weighting!r}; expected one of "
            f"{sorted(supported_source_statistics_weighting)}"
        )
    if str(config.contrastive_pairing_mode).strip().lower() not in {
        "diagonal_v1",
        "parent_multi_positive_v1",
    }:
        raise ValueError(
            "unsupported pretrain contrastive_pairing_mode: "
            f"{config.contrastive_pairing_mode!r}; expected 'diagonal_v1' "
            "or 'parent_multi_positive_v1'"
        )
    if str(config.parent_sampling_mode).strip().lower() not in {
        "window_shuffle_v1",
        "parent_equal_v1",
    }:
        raise ValueError(
            "unsupported pretrain parent_sampling_mode: "
            f"{config.parent_sampling_mode!r}; expected 'window_shuffle_v1' "
            "or 'parent_equal_v1'"
        )
    parent_validation_requested = bool(str(config.parent_split_manifest_path).strip()) or bool(
        config.require_parent_validation
    )
    if config.require_parent_validation and not str(config.parent_split_manifest_path).strip():
        raise ValueError(
            "pretrain require_parent_validation=true requires parent_split_manifest_path"
        )
    if parent_validation_requested and not config.require_train_parent_manifest:
        raise ValueError(
            "pretrain parent validation requires require_train_parent_manifest=true"
        )
    if parent_validation_requested and str(config.sampling_mode).strip().lower() != "causal_window":
        raise ValueError("pretrain parent validation currently requires sampling_mode='causal_window'")
    if parent_validation_requested and not config.require_manifest_provenance:
        raise ValueError(
            "pretrain parent validation requires require_manifest_provenance=true"
        )
    track_sequence_feature_dim(config.trajectory_feature_schema)
    if str(config.temporal_encoding_mode).strip().lower() not in TEMPORAL_ENCODING_MODES:
        raise ValueError(
            "unsupported pretrain temporal encoding mode: "
            f"{config.temporal_encoding_mode!r}; expected one of {sorted(TEMPORAL_ENCODING_MODES)}"
        )
    if str(config.temporal_pooling_mode).strip().lower() not in TEMPORAL_POOLING_MODES:
        raise ValueError(
            "unsupported pretrain temporal pooling mode: "
            f"{config.temporal_pooling_mode!r}; expected one of {sorted(TEMPORAL_POOLING_MODES)}"
        )
    return config


def load_pipeline_config(path: str | Path = "config/config.json") -> MainPipelineConfig:
    payload = load_config_file(path)
    return _make_dataclass(MainPipelineConfig, _clean_section(payload, "pipeline"))
