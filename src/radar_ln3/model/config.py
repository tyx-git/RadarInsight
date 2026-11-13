"""正式模型配置解析。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from radar_ln3.data.paths import resolve_project_path


@dataclass(frozen=True)
class ModelSettings:
    tr_value_dim: int
    tr_missing_dim: int
    rd_quality_dim: int
    rd_height: int
    rd_width: int
    rd_embedding_dim: int
    hidden_dim: int
    liquid_input_dim: int
    liquid_hidden_dim: int
    knownness_projection_dim: int
    tau_min_seconds: float
    tau_max_seconds: float
    zero_dt_event_floor_seconds: float
    dt_max_seconds: float
    invalid_dt_update_mode: str
    invalid_dt_observation_gain: float
    evidence_dim: int
    rd_frame_aggregation: str = "masked_mean"
    residual_bound: float = 3.0
    token_activation: str = "silu"
    frame_gru_layers: int = 1
    parent_gru_layers: int = 1
    motion_projection_dim: int = 16
    motion_adapter_dim: int = 16
    motion_projection_bias: bool = False
    motion_output_bias: bool = False
    motion_output_zero_initialization: bool = True
    stable_evidence_arm: str = "main"
    trajectory_heads: str = "frozen_unused_by_classification"
    stable8_hidden_dim: int = 32
    observation_adapter_hidden_dim: int = 48


@dataclass(frozen=True)
class L1DataSettings:
    course_minimum_valid_nonfirst_points: int = 3
    course_feature_dim: int = 1


@dataclass(frozen=True)
class L2DataSettings:
    rd_physical_dim: int = 4
    stable8_indices: tuple[int, ...] = (1, 2, 3, 7, 11, 13, 14, 22)
    stable8_median: tuple[float, ...] = ()
    stable8_scale: tuple[float, ...] = ()
    tr_clipping_bounds: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    direct_stable8_input_mode: str = "raw"

    @property
    def micro_doppler_dim(self) -> int:
        return len(self.stable8_indices)


@dataclass(frozen=True)
class DataSettings:
    max_sequence_length: int
    max_rd_frames_per_point: int
    l1: L1DataSettings
    l2: L2DataSettings
    stable8_dim: int
    motion_dim: int = 9
    motion_clip_absolute: float = 5.0


@dataclass(frozen=True)
class ArtifactSettings:
    """模型基础权重和正式输出的项目相对路径。"""

    l1_base_weight_path: str
    l1_output_path: str
    l2_base_weight_path: str
    l2_stable_weight_path: str
    l2_baseline_anchor_path: str
    l2_output_path: str
    l2_direct_output_path: str
    l2_direct_grouped_output_path: str


@dataclass(frozen=True)
class TrainingSettings:
    """把 L1/L2 独立训练配置解析为统一访问对象。

    ``l2`` 字段保留旧的扁平访问属性，供模型构造和 L3 代码读取；L1
    训练入口使用 ``l1_*`` 属性，避免把两个任务的优化参数混在一起。
    """

    batch_size: int
    epochs: int
    seed: int
    dataloader_seed: int
    learning_rate: float
    base_learning_rate: float
    stable_learning_rate: float
    motion_learning_rate: float
    weight_decay: float
    l2_weight_decay: float
    gradient_clip_norm: float
    class_weights: Mapping[str, tuple[float, ...]]
    checkpoint_selection: str
    l2_checkpoint_selection: str
    l2_prefix_loss_weight: float = 0.15
    l2_prefix_loss_events: int = 4
    l1_batch_size: int = 4
    l1_epochs: int = 20
    l1_seed: int = 42
    l1_dataloader_seed: int = 42
    l1_learning_rate: float = 0.0001
    l1_weight_decay: float = 0.01
    l1_gradient_clip_norm: float = 100.0
    l1_class_weights: tuple[float, ...] = ()
    l1_checkpoint_selection: str = "minimum_class_f1_macro_f1"


@dataclass(frozen=True)
class ModelConfig:
    """兼容 task model 所需的最小、强类型配置对象。"""

    model: ModelSettings
    data: DataSettings
    class_names: Mapping[str, tuple[str, ...]]
    artifacts: ArtifactSettings
    training: TrainingSettings
    path: Path
    identity: Mapping[str, Any]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"模型配置 {name} 必须是正整数")
    return int(value)


def _finite_float(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"模型配置 {name} 必须是数值") from exc
    import math

    if not math.isfinite(result) or (positive and result <= 0.0):
        raise ValueError(f"模型配置 {name} 数值无效")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"模型配置 {name} 必须是非负整数")
    return int(value)


def _relative_text(value: Any, name: str) -> str:
    text = str(value or "").replace("\\", "/").strip()
    if not text or text.startswith("/") or ":" in text.split("/", 1)[0]:
        raise ValueError(f"模型配置 {name} 必须是项目相对路径")
    if any(part == ".." for part in text.split("/")):
        raise ValueError(f"模型配置 {name} 不得越过项目根目录")
    return text


def _positive_float(value: Any, name: str) -> float:
    return _finite_float(value, name, positive=True)


def _non_negative_index(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"模型配置 {name} 必须是非负整数")
    return int(value)


def _float_tuple(value: Any, name: str, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"模型配置 {name} 必须包含 {length} 个值")
    return tuple(_finite_float(item, name) for item in value)


def load_model_config(
    project_root: str | Path,
    path: str | Path = "src/config/model.json",
    *,
    artifact_root: str | Path | None = None,
) -> ModelConfig:
    """从集中配置加载模型结构和任务输入契约。"""

    root = Path(project_root).resolve()
    target = resolve_project_path(root, path, must_exist=True)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("模型配置根节点必须是对象")
    if payload.get("schema_version") != "RADAR-LN3-MODEL-CONFIG-V2":
        raise ValueError("模型配置 schema_version 不匹配")
    # 正式配置只有 common、L1、L2 三个语义分区。schema_version 是元数据，
    # 不参与任务配置分区。
    common = payload.get("common")
    raw_l1_section = payload.get("L1")
    raw_l2_section = payload.get("L2")
    if not isinstance(common, Mapping) or not isinstance(raw_l1_section, Mapping) or not isinstance(raw_l2_section, Mapping):
        raise ValueError("模型配置必须包含 common、L1、L2 三个对象")
    raw_input = common.get("input")
    raw_liquid = common.get("liquid")
    raw_names = common.get("class_names")
    if not isinstance(raw_input, Mapping) or not isinstance(raw_liquid, Mapping):
        raise ValueError("模型配置缺少 input/liquid")
    if not isinstance(raw_names, Mapping):
        raise ValueError("模型配置缺少 class_names")
    names: dict[str, tuple[str, ...]] = {}
    for task in ("l1", "l2"):
        value = raw_names.get(task)
        if not isinstance(value, list) or not value or any(not str(item) for item in value):
            raise ValueError(f"模型配置 class_names.{task} 无效")
        names[task] = tuple(str(item) for item in value)
    raw_l1 = raw_l1_section.get("model")
    raw_l2 = raw_l2_section.get("model")
    raw_l1_training = raw_l1_section.get("training")
    raw_l2_training = raw_l2_section.get("training")
    if not isinstance(raw_l1, Mapping) or not isinstance(raw_l2, Mapping):
        raise ValueError("模型配置 L1/L2 缺少 model 对象")
    if not isinstance(raw_l1_training, Mapping) or not isinstance(raw_l2_training, Mapping):
        raise ValueError("模型配置 L1/L2 缺少 training 对象")
    model = ModelSettings(
        tr_value_dim=_positive_int(raw_input.get("tr_value_dim"), "input.tr_value_dim"),
        tr_missing_dim=_positive_int(raw_input.get("tr_missing_dim"), "input.tr_missing_dim"),
        rd_quality_dim=_positive_int(raw_input.get("rd_quality_dim"), "input.rd_quality_dim"),
        rd_height=_positive_int(raw_input.get("rd_height"), "input.rd_height"),
        rd_width=_positive_int(raw_input.get("rd_width"), "input.rd_width"),
        rd_embedding_dim=_positive_int(raw_input.get("rd_embedding_dim"), "input.rd_embedding_dim"),
        hidden_dim=_positive_int(raw_input.get("hidden_dim"), "input.hidden_dim"),
        liquid_input_dim=_positive_int(raw_input.get("liquid_input_dim"), "input.liquid_input_dim"),
        liquid_hidden_dim=_positive_int(raw_input.get("liquid_hidden_dim"), "input.liquid_hidden_dim"),
        knownness_projection_dim=32,
        tau_min_seconds=_finite_float(raw_liquid.get("tau_min_seconds"), "liquid.tau_min_seconds", positive=True),
        tau_max_seconds=_finite_float(raw_liquid.get("tau_max_seconds"), "liquid.tau_max_seconds", positive=True),
        zero_dt_event_floor_seconds=_finite_float(raw_liquid.get("zero_dt_event_floor_seconds"), "liquid.zero_dt_event_floor_seconds", positive=True),
        dt_max_seconds=_finite_float(raw_liquid.get("dt_max_seconds"), "liquid.dt_max_seconds", positive=True),
        invalid_dt_update_mode=str(raw_liquid.get("invalid_dt_update_mode", "")),
        invalid_dt_observation_gain=_finite_float(raw_liquid.get("invalid_dt_observation_gain"), "liquid.invalid_dt_observation_gain", positive=True),
        evidence_dim=_positive_int(raw_input.get("stable8_dim"), "input.stable8_dim"),
        residual_bound=_finite_float(
            raw_l2.get("stable8_residual_bound", 3.0),
            "l2.stable8_residual_bound",
            positive=True,
        ),
        token_activation="silu",
        frame_gru_layers=1,
        parent_gru_layers=1,
        motion_projection_dim=_positive_int(raw_l2.get("motion_projection_dim"), "l2.motion_projection_dim"),
        motion_adapter_dim=_positive_int(raw_l2.get("motion_hidden_dim"), "l2.motion_hidden_dim"),
        motion_projection_bias=False,
        motion_output_bias=bool(raw_l2.get("motion_output_bias", False)),
        motion_output_zero_initialization=bool(raw_l2.get("motion_zero_initialize", True)),
        stable_evidence_arm="main",
        trajectory_heads="frozen_unused_by_classification",
        stable8_hidden_dim=_positive_int(raw_l2.get("stable8_hidden_dim"), "l2.stable8_hidden_dim"),
        observation_adapter_hidden_dim=_positive_int(
            raw_l1.get("adapter_hidden_dim"), "l1.adapter_hidden_dim"
        ),
    )
    if model.tau_min_seconds > model.tau_max_seconds:
        raise ValueError("Liquid V2 tau 范围无效")
    if model.invalid_dt_update_mode not in {"floor", "bounded_observation", "skip", "strict_skip", "density_weighted"}:
        raise ValueError("Liquid V2 invalid_dt_update_mode 无效")
    stable_median = _float_tuple(
        raw_l2.get("stable8_median"), "l2.stable8_median", length=model.evidence_dim
    )
    stable_scale = tuple(
        _positive_float(item, "l2.stable8_scale")
        for item in _float_tuple(
            raw_l2.get("stable8_scale"), "l2.stable8_scale", length=model.evidence_dim
        )
    )
    raw_indices = raw_l2.get("stable8_indices")
    if (
        not isinstance(raw_indices, list)
        or len(raw_indices) != model.evidence_dim
        or len(set(raw_indices)) != len(raw_indices)
    ):
        raise ValueError("模型配置 l2.stable8_indices 必须是唯一的定长列表")
    stable_indices = tuple(
        _non_negative_index(item, "l2.stable8_indices") for item in raw_indices
    )
    if any(item >= int(raw_input.get("stable8_evidence_dim", 0)) for item in stable_indices):
        raise ValueError("模型配置 l2.stable8_indices 超出 evidence 范围")
    raw_clipping = raw_l2.get("tr_clipping_bounds")
    if not isinstance(raw_clipping, Mapping):
        raise ValueError("模型配置 l2.tr_clipping_bounds 必须是对象")
    clipping_bounds: dict[str, tuple[float, float]] = {}
    for name in ("range_m", "speed_mps"):
        bound = raw_clipping.get(name)
        if not isinstance(bound, Mapping):
            raise ValueError(f"模型配置 l2.tr_clipping_bounds 缺少 {name}")
        lower = _finite_float(bound.get("lower"), f"l2.tr_clipping_bounds.{name}.lower")
        upper = _finite_float(bound.get("upper"), f"l2.tr_clipping_bounds.{name}.upper")
        if upper < lower:
            raise ValueError(f"模型配置 l2.tr_clipping_bounds.{name} 范围无效")
        clipping_bounds[name] = (lower, upper)
    data = DataSettings(
        max_sequence_length=_positive_int(raw_input.get("max_sequence_length"), "input.max_sequence_length"),
        max_rd_frames_per_point=_positive_int(raw_input.get("rd_max_frames"), "input.rd_max_frames"),
        l1=L1DataSettings(),
        l2=L2DataSettings(
            stable8_indices=stable_indices,
            stable8_median=stable_median,
            stable8_scale=stable_scale,
            tr_clipping_bounds=clipping_bounds,
            direct_stable8_input_mode=str(raw_l2.get("direct_stable8_input_mode", "raw")),
        ),
        stable8_dim=_positive_int(raw_input.get("stable8_dim"), "input.stable8_dim"),
        motion_dim=9,
        motion_clip_absolute=5.0,
    )
    if int(model.evidence_dim) != int(data.stable8_dim):
        raise ValueError("模型配置 input.stable8_dim 与 stable8 证据宽度不一致")
    if len(data.l2.stable8_indices) != int(data.stable8_dim):
        raise ValueError("模型配置 l2.stable8_indices 数量与 input.stable8_dim 不一致")
    def _class_weights(raw_training: Mapping[str, Any], task: str) -> tuple[float, ...]:
        values = raw_training.get("class_weights")
        if not isinstance(values, list) or len(values) != len(names[task]):
            raise ValueError(f"模型配置 {task}.training.class_weights 宽度无效")
        return tuple(_positive_float(value, f"{task}.training.class_weights") for value in values)

    l1_class_weights = _class_weights(raw_l1_training, "l1")
    l2_class_weights = _class_weights(raw_l2_training, "l2")
    artifacts = ArtifactSettings(
        l1_base_weight_path=_relative_text(raw_l1.get("base_weight_path"), "l1.base_weight_path"),
        l1_output_path=_relative_text(raw_l1.get("output_path"), "l1.output_path"),
        l2_base_weight_path=_relative_text(raw_l2.get("base_weight_path"), "l2.base_weight_path"),
        l2_stable_weight_path=_relative_text(raw_l2.get("stable8_weight_path"), "l2.stable8_weight_path"),
        l2_baseline_anchor_path=_relative_text(
            raw_l2.get("baseline_anchor_path"), "l2.baseline_anchor_path"
        ),
        l2_output_path=_relative_text(raw_l2.get("output_path"), "l2.output_path"),
        l2_direct_output_path=_relative_text(
            raw_l2.get(
                "direct_output_path",
                "artifacts/radar_ln3/l2-direct/l2-direct.pt",
            ),
            "l2.direct_output_path",
        ),
        l2_direct_grouped_output_path=_relative_text(
            raw_l2.get(
                "direct_grouped_output_path",
                "artifacts/radar_ln3/l2-direct-grouped/l2-direct-grouped.pt",
            ),
            "l2.direct_grouped_output_path",
        ),
    )
    if artifact_root is not None:
        from dataclasses import replace

        output_root = Path(artifact_root)
        if output_root.is_absolute():
            try:
                output_root = output_root.resolve().relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError("artifact_root must be inside project_root") from exc
        else:
            output_root = output_root.as_posix()
        artifacts = replace(
            artifacts,
            l1_output_path=f"{output_root}/l1/l1.pt",
            l2_output_path=f"{output_root}/l2/l2.pt",
            l2_direct_output_path=f"{output_root}/l2-direct/l2-direct.pt",
            l2_direct_grouped_output_path=(
                f"{output_root}/l2-direct-grouped/l2-direct-grouped.pt"
            ),
        )
    if data.l2.direct_stable8_input_mode not in {"raw", "normalized"}:
        raise ValueError("模型配置 l2.direct_stable8_input_mode 无效")
    training = TrainingSettings(
        batch_size=_positive_int(raw_l2_training.get("batch_size"), "L2.training.batch_size"),
        epochs=_positive_int(raw_l2_training.get("epochs"), "L2.training.epochs"),
        seed=_non_negative_int(raw_l2_training.get("seed"), "L2.training.seed"),
        dataloader_seed=_non_negative_int(raw_l2_training.get("dataloader_seed"), "L2.training.dataloader_seed"),
        learning_rate=_positive_float(raw_l2_training.get("learning_rate"), "L2.training.learning_rate"),
        base_learning_rate=_positive_float(
            raw_l2_training.get("base_learning_rate", raw_l2_training.get("learning_rate")),
            "L2.training.base_learning_rate",
        ),
        stable_learning_rate=_positive_float(
            raw_l2_training.get("stable_learning_rate", raw_l2_training.get("learning_rate")),
            "L2.training.stable_learning_rate",
        ),
        motion_learning_rate=_positive_float(
            raw_l2_training.get("motion_learning_rate", raw_l2_training.get("learning_rate")),
            "L2.training.motion_learning_rate",
        ),
        weight_decay=_finite_float(raw_l2_training.get("weight_decay"), "L2.training.weight_decay"),
        l2_weight_decay=_finite_float(
            raw_l2_training.get("l2_weight_decay", raw_l2_training.get("weight_decay")),
            "L2.training.l2_weight_decay",
        ),
        gradient_clip_norm=_positive_float(raw_l2_training.get("gradient_clip_norm"), "L2.training.gradient_clip_norm"),
        class_weights={"l1": l1_class_weights, "l2": l2_class_weights},
        checkpoint_selection=str(raw_l1_training.get("checkpoint_selection", "")),
        l2_checkpoint_selection=str(
            raw_l2_training.get(
                "l2_checkpoint_selection",
                "minimum_class_f1_drone_bird_confusion_macro_f1",
            )
        ),
        l2_prefix_loss_weight=_finite_float(
            raw_l2_training.get("l2_prefix_loss_weight", 0.15),
            "L2.training.l2_prefix_loss_weight",
        ),
        l2_prefix_loss_events=_positive_int(
            raw_l2_training.get("l2_prefix_loss_events", 4),
            "L2.training.l2_prefix_loss_events",
        ),
        l1_batch_size=_positive_int(raw_l1_training.get("batch_size"), "L1.training.batch_size"),
        l1_epochs=_positive_int(raw_l1_training.get("epochs"), "L1.training.epochs"),
        l1_seed=_non_negative_int(raw_l1_training.get("seed"), "L1.training.seed"),
        l1_dataloader_seed=_non_negative_int(raw_l1_training.get("dataloader_seed"), "L1.training.dataloader_seed"),
        l1_learning_rate=_positive_float(raw_l1_training.get("learning_rate"), "L1.training.learning_rate"),
        l1_weight_decay=_finite_float(raw_l1_training.get("weight_decay"), "L1.training.weight_decay"),
        l1_gradient_clip_norm=_positive_float(raw_l1_training.get("gradient_clip_norm"), "L1.training.gradient_clip_norm"),
        l1_class_weights=l1_class_weights,
        l1_checkpoint_selection=str(raw_l1_training.get("checkpoint_selection", "")),
    )
    if training.l2_prefix_loss_weight < 0.0:
        raise ValueError("training.l2_prefix_loss_weight 必须是非负有限值")
    if training.l1_checkpoint_selection != "minimum_class_f1_macro_f1":
        raise ValueError("L1 当前正式训练只支持 minimum_class_f1_macro_f1")
    if training.l2_checkpoint_selection != "minimum_class_f1_drone_bird_confusion_macro_f1":
        raise ValueError(
            "L2 当前正式训练只支持 "
            "minimum_class_f1_drone_bird_confusion_macro_f1"
        )
    return ModelConfig(
        model=model,
        data=data,
        class_names=names,
        artifacts=artifacts,
        training=training,
        path=target,
        identity=dict(payload),
    )


def task_class_names(config: ModelConfig, task: str) -> tuple[str, ...]:
    try:
        return tuple(config.class_names[str(task)])
    except KeyError as exc:
        raise ValueError(f"不支持的模型 task：{task}") from exc


__all__ = [
    "DataSettings",
    "ArtifactSettings",
    "L1DataSettings",
    "L2DataSettings",
    "ModelConfig",
    "ModelSettings",
    "TrainingSettings",
    "load_model_config",
    "task_class_names",
]
