from __future__ import annotations

from dataclasses import dataclass, field

from .router import ROUTE_NAMES


@dataclass(frozen=True)
class RouteSpec:
    route_name: str
    description: str
    class_names: tuple[str, ...]


@dataclass(frozen=True)
class SystemArchitectureSpec:
    sequence_length_hint: int = 50
    resample_step_seconds: float = 0.1
    route_specs: tuple[RouteSpec, ...] = field(
        default_factory=lambda: (
            RouteSpec(
                route_name=ROUTE_NAMES[0],
                description="低空可疑目标专家，采用轨迹常开分支 + 幅度/相位 RD 多模态 Transformer 分支 + 统计特征增强，负责无人机/鸟/气球/低空杂类目标与低置信拒识目标。",
                class_names=("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"),
            ),
            RouteSpec(
                route_name=ROUTE_NAMES[1],
                description="常规空中目标专家，采用轻量1D-CNN处理时序轨迹特征。",
                class_names=("AircraftTarget",),
            ),
            RouteSpec(
                route_name=ROUTE_NAMES[2],
                description="高空高速过滤专家，优先由规则兜底完成过滤。",
                class_names=(),
            ),
            RouteSpec(
                route_name=ROUTE_NAMES[3],
                description="地面目标专家，负责人/车/船。",
                class_names=("PeopleTarget", "CarTarget", "ShipTarget"),
            ),
        )
    )


DEFAULT_ARCHITECTURE_SPEC = SystemArchitectureSpec()
