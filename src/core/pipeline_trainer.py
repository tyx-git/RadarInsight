from __future__ import annotations

import json
from pathlib import Path

from .pipeline_config import MainPipelineConfig
from .expert1_config import Expert1TrainConfig
from .expert1_trainer import train_expert1_v1
from .expert_other_config import ExpertOtherTrainConfig
from .expert_other_trainer import train_other_expert
from .pretrain_track_config import TrackPretrainConfig
from .pretrain_track_trainer import pretrain_track_encoder
from .router_train_config import RouterTrainConfig
from .router_trainer import train_router_v1
from .system_config import SystemTrainConfig
from .system_trainer import train_system_v1


def run_main_pipeline(config: MainPipelineConfig) -> dict[str, object]:
    output_root = Path(config.output_root)
    pretrain_summary = pretrain_track_encoder(
        TrackPretrainConfig(
            data_root=config.data_root,
            output_dir=str(output_root / "pretrain_track"),
            device=config.device,
            epochs=config.pretrain_epochs,
            batch_size=config.pretrain_batch_size,
            learning_rate=config.learning_rate,
        )
    )
    router_summary = train_router_v1(
        RouterTrainConfig(
            data_root=config.data_root,
            output_dir=str(output_root / "router_v1"),
            device=config.device,
            epochs=config.router_epochs,
            batch_size=config.router_batch_size,
            learning_rate=config.learning_rate,
        )
    )
    expert1_summary = train_expert1_v1(
        Expert1TrainConfig(
            data_root=config.data_root,
            output_dir=str(output_root / "expert1_v1"),
            device=config.device,
            epochs=config.expert1_epochs,
            batch_size=config.expert1_batch_size,
            learning_rate=config.learning_rate,
            sequence_encoder_lr_multiplier=0.1,
            pretrained_track_checkpoint=pretrain_summary["best_checkpoint"],
            freeze_track_encoder_epochs=1,
        )
    )
    expert2_summary = train_other_expert(
        ExpertOtherTrainConfig(
            data_root=config.data_root,
            output_dir=str(output_root / "expert2_air"),
            device=config.device,
            epochs=config.expert2_epochs,
            batch_size=config.expert2_batch_size,
            learning_rate=config.learning_rate,
            expert_name="expert2_air",
            shared_representation_checkpoint=expert1_summary["best_checkpoint"],
            train_representation_builder=False,
            source_normalization=True,
        )
    )
    expert4_summary = train_other_expert(
        ExpertOtherTrainConfig(
            data_root=config.data_root,
            output_dir=str(output_root / "expert4_ground"),
            device=config.device,
            epochs=config.expert4_epochs,
            batch_size=config.expert4_batch_size,
            learning_rate=config.learning_rate,
            expert_name="expert4_ground",
            shared_representation_checkpoint=expert1_summary["best_checkpoint"],
            train_representation_builder=False,
            source_normalization=True,
        )
    )
    system_summary = train_system_v1(
        SystemTrainConfig(
            data_root=config.data_root,
            output_dir=str(output_root / "system_v1"),
            device=config.device,
            epochs=config.system_epochs,
            batch_size=config.system_batch_size,
            learning_rate=config.learning_rate,
            source_normalization=True,
            pretrained_track_checkpoint=pretrain_summary["best_checkpoint"],
            pretrained_router_checkpoint=router_summary["best_checkpoint"],
            expert1_checkpoint=expert1_summary["best_checkpoint"],
            expert2_checkpoint=expert2_summary["best_checkpoint"],
            expert4_checkpoint=expert4_summary["best_checkpoint"],
            freeze_track_encoder_epochs=config.freeze_track_encoder_epochs,
            stage_warmup_epochs=config.stage_warmup_epochs,
            stage_expert_warmup_epochs=config.stage_expert_warmup_epochs,
            stage_joint_epochs=config.stage_joint_epochs,
            joint_loss_ramp_epochs=2,
            freeze_non_router_in_warmup=True,
        )
    )
    summary = {
        "pretrain": pretrain_summary,
        "router": router_summary,
        "expert1": expert1_summary,
        "expert2": expert2_summary,
        "expert4": expert4_summary,
        "system": system_summary,
    }
    pipeline_root = Path(system_summary["best_checkpoint"]).parent.parent
    summary_path = pipeline_root / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
