from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MainPipelineConfig:
    data_root: str = "data"
    output_root: str = "artifacts/main_pipeline"
    device: str = "cuda"
    pretrain_enabled: bool = True
    router_enabled: bool = True
    expert1_enabled: bool = True
    expert2_enabled: bool = True
    expert4_enabled: bool = True
    system_enabled: bool = True
    pretrain_output_dir: str = "artifacts/main_pipeline/pretrain_track"
    router_output_dir: str = "artifacts/main_pipeline/router_v1"
    expert1_output_dir: str = "artifacts/main_pipeline/expert1_v1"
    expert2_output_dir: str = "artifacts/main_pipeline/expert2_air"
    expert4_output_dir: str = "artifacts/main_pipeline/expert4_ground"
    system_output_dir: str = "artifacts/main_pipeline/system_v1"
    pretrain_checkpoint: str = ""
    router_checkpoint: str = ""
    expert1_checkpoint: str = ""
    expert2_checkpoint: str = ""
    expert4_checkpoint: str = ""
    system_checkpoint: str = ""
