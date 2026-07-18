from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

from src.core.expert1_config import Expert1TrainConfig
from src.core.expert_other_config import ExpertOtherTrainConfig
from src.core.pipeline_config import MainPipelineConfig
from src.core.pretrain_track_config import TrackPretrainConfig
from src.core.router_train_config import RouterTrainConfig
from src.core.system_config import SystemTrainConfig


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


def _make_dataclass(config_type: type[T], values: dict[str, Any]) -> T:
    if not is_dataclass(config_type):
        raise TypeError(f"config_type must be a dataclass: {config_type}")
    allowed = {item.name for item in fields(config_type)}
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
    return _make_dataclass(Expert1TrainConfig, values)


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
    return _make_dataclass(SystemTrainConfig, _clean_section(payload, "system"))


def load_router_config(path: str | Path = "config/config.json") -> RouterTrainConfig:
    payload = load_config_file(path)
    return _make_dataclass(RouterTrainConfig, _clean_section(payload, "router"))


def load_pretrain_track_config(path: str | Path = "config/config.json") -> TrackPretrainConfig:
    payload = load_config_file(path)
    return _make_dataclass(TrackPretrainConfig, _clean_section(payload, "pretrain_track"))


def load_pipeline_config(path: str | Path = "config/config.json") -> MainPipelineConfig:
    payload = load_config_file(path)
    return _make_dataclass(MainPipelineConfig, _clean_section(payload, "pipeline"))
