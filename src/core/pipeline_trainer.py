from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from config.configmanager import (
    load_expert1_config,
    load_expert2_config,
    load_expert4_config,
    load_pretrain_track_config,
    load_router_config,
    load_system_config,
)

from .expert1_trainer import train_expert1
from .expert_other_trainer import train_other_expert
from .pipeline_config import MainPipelineConfig
from .pretrain_track_trainer import pretrain_track_encoder
from .router_trainer import train_router_v1
from .system_trainer import train_system_v1


def _load_existing_summary(output_dir: str) -> dict[str, object] | None:
    summary_path = Path(output_dir) / "training_summary.json"
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _summary_from_checkpoint(checkpoint_path: str) -> dict[str, object] | None:
    if not checkpoint_path:
        return None
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        return None
    return {
        "best_checkpoint": checkpoint.as_posix(),
        "recovered_from_checkpoint_only": True,
    }


def run_main_pipeline(config: MainPipelineConfig, config_path: str = "config/config.json") -> dict[str, object]:
    output_root = Path(config.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {}

    pretrain_summary = None
    if config.pretrain_enabled:
        pretrain_config = replace(
            load_pretrain_track_config(config_path),
            output_dir=config.pretrain_output_dir,
        )
        pretrain_summary = _load_existing_summary(pretrain_config.output_dir) or _summary_from_checkpoint(config.pretrain_checkpoint)
        if pretrain_summary is None:
            pretrain_summary = pretrain_track_encoder(pretrain_config)
        summary["pretrain"] = pretrain_summary

    router_summary = None
    if config.router_enabled:
        router_config = replace(
            load_router_config(config_path),
            output_dir=config.router_output_dir,
        )
        router_summary = _load_existing_summary(router_config.output_dir) or _summary_from_checkpoint(config.router_checkpoint)
        if router_summary is None:
            router_summary = train_router_v1(router_config)
        summary["router"] = router_summary

    expert1_summary = None
    if config.expert1_enabled:
        expert1_config = load_expert1_config(config_path)
        if pretrain_summary is not None:
            expert1_config = replace(
                expert1_config,
                output_dir=config.expert1_output_dir,
                pretrained_track_checkpoint=pretrain_summary["best_checkpoint"],
            )
        else:
            expert1_config = replace(expert1_config, output_dir=config.expert1_output_dir)
        expert1_summary = _load_existing_summary(expert1_config.output_dir) or _summary_from_checkpoint(config.expert1_checkpoint)
        if expert1_summary is None:
            expert1_summary = train_expert1(expert1_config)
        summary["expert1"] = expert1_summary

    expert2_summary = None
    if config.expert2_enabled:
        expert2_config = load_expert2_config(config_path)
        shared_representation_checkpoint = (
            expert1_summary["best_checkpoint"]
            if expert1_summary is not None
            else expert2_config.shared_representation_checkpoint
        )
        expert2_config = replace(
            expert2_config,
            output_dir=config.expert2_output_dir,
            shared_representation_checkpoint=shared_representation_checkpoint,
        )
        expert2_summary = _load_existing_summary(expert2_config.output_dir) or _summary_from_checkpoint(config.expert2_checkpoint)
        if expert2_summary is None:
            expert2_summary = train_other_expert(expert2_config)
        summary["expert2"] = expert2_summary

    expert4_summary = None
    if config.expert4_enabled:
        expert4_config = load_expert4_config(config_path)
        shared_representation_checkpoint = (
            expert1_summary["best_checkpoint"]
            if expert1_summary is not None
            else expert4_config.shared_representation_checkpoint
        )
        expert4_config = replace(
            expert4_config,
            output_dir=config.expert4_output_dir,
            shared_representation_checkpoint=shared_representation_checkpoint,
        )
        expert4_summary = _load_existing_summary(expert4_config.output_dir) or _summary_from_checkpoint(config.expert4_checkpoint)
        if expert4_summary is None:
            expert4_summary = train_other_expert(expert4_config)
        summary["expert4"] = expert4_summary

    system_summary = None
    if config.system_enabled:
        system_config = load_system_config(config_path)
        if pretrain_summary is not None:
            system_config = replace(
                system_config,
                pretrained_track_checkpoint=pretrain_summary["best_checkpoint"],
            )
        if router_summary is not None:
            system_config = replace(
                system_config,
                pretrained_router_checkpoint=router_summary["best_checkpoint"],
            )
        if expert1_summary is not None:
            system_config = replace(
                system_config,
                expert1_checkpoint=expert1_summary["best_checkpoint"],
            )
        if expert2_summary is not None:
            system_config = replace(
                system_config,
                expert2_checkpoint=expert2_summary["best_checkpoint"],
            )
        if expert4_summary is not None:
            system_config = replace(
                system_config,
                expert4_checkpoint=expert4_summary["best_checkpoint"],
            )
        system_config = replace(system_config, output_dir=config.system_output_dir)
        system_summary = _load_existing_summary(system_config.output_dir) or _summary_from_checkpoint(config.system_checkpoint)
        if system_summary is None:
            system_summary = train_system_v1(system_config)
        summary["system"] = system_summary

    summary_path = output_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
