"""特征层的通用契约。

本模块只描述特征的形状、有效性和来源，不解释任何 CQ08 文件名、MAT
字段或目录。数据集适配器应先把原始数据转换为
``radar_ln3.data.contracts.EventData``，特征层再消费这些通用对象。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Sequence
from typing import Any, Mapping

import numpy as np

from radar_ln3.data.contracts import RawEchoFrame, RawEchoProvider, RawTargetRecord


class FeatureError(ValueError):
    """特征处理错误的基类。"""


class FeatureInputError(FeatureError):
    """输入字段、数值或来源不满足特征契约。"""


class FeatureShapeError(FeatureInputError):
    """特征数组的 shape 或 dtype 不满足契约。"""


class FeatureNumericalError(FeatureInputError):
    """特征计算产生非有限值或越界值。"""


class FeatureSchemaError(FeatureError):
    """schema 定义无效或版本不匹配。"""


class FeatureCacheError(FeatureError):
    """特征缓存错误的基类。"""


class FeatureCachePathError(FeatureCacheError):
    """缓存路径不存在、类型错误或不允许覆盖。"""


class FeatureCacheFormatError(FeatureCacheError):
    """缓存文件结构无效。"""


class FeatureCacheIntegrityError(FeatureCacheError):
    """缓存身份、对齐关系或 hash 绑定不一致。"""


def event_values(item: Any) -> Mapping[str, Any]:
    """取事件的 TR 映射：直接给 mapping，或从 ``tr_values`` 取。"""

    if isinstance(item, Mapping):
        return item
    values = getattr(item, "tr_values", None)
    if not isinstance(values, Mapping):
        raise FeatureInputError("TR event 缺少 mapping 类型的 tr_values")
    return values


def _as_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if not result or any(not value for value in result):
        raise FeatureSchemaError(f"{name} 不能为空")
    if len(result) != len(set(result)):
        raise FeatureSchemaError(f"{name} 不能包含重复名称")
    return result


@dataclass(frozen=True)
class FeatureSchema:
    """一个特征数组及其有效性掩码的稳定描述。"""

    name: str
    schema_version: str
    value_names: tuple[str, ...]
    valid_names: tuple[str, ...] = ()
    mask_mode: str = "gating_only"
    dtype: str = "float32"
    value_bounds: tuple[float | None, float | None] | None = None

    def __post_init__(self) -> None:
        if not str(self.name) or not str(self.schema_version):
            raise FeatureSchemaError("特征 schema 的 name/version 不能为空")
        values = _as_tuple(self.value_names, name="value_names")
        valid = tuple(str(value) for value in self.valid_names)
        if valid and len(valid) != len(set(valid)):
            raise FeatureSchemaError("valid_names 不能包含重复名称")
        if self.mask_mode not in {"gating_only", "explicit_input", "none"}:
            raise FeatureSchemaError(f"不支持的 mask_mode：{self.mask_mode}")
        if self.dtype != "float32":
            raise FeatureSchemaError("当前产品特征只允许 float32 输出")
        if self.value_bounds is not None:
            lower, upper = self.value_bounds
            if lower is not None and upper is not None and float(lower) > float(upper):
                raise FeatureSchemaError("value_bounds 下界不能大于上界")
        object.__setattr__(self, "value_names", values)
        object.__setattr__(self, "valid_names", valid)

    @property
    def value_dim(self) -> int:
        return len(self.value_names)

    @property
    def valid_dim(self) -> int:
        return len(self.valid_names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "schema_version": self.schema_version,
            "value_dim": self.value_dim,
            "value_names": list(self.value_names),
            "valid_dim": self.valid_dim,
            "valid_names": list(self.valid_names),
            "mask_mode": self.mask_mode,
            "dtype": self.dtype,
            "value_bounds": list(self.value_bounds)
            if self.value_bounds is not None
            else None,
        }


@dataclass(frozen=True)
class FeatureEvent:
    """一个事件的特征值和有效性信息。"""

    values: np.ndarray
    valid: np.ndarray | bool
    schema: FeatureSchema | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        valid = np.asarray(self.valid)
        if values.dtype != np.float32 or not np.isfinite(values).all():
            raise FeatureShapeError("FeatureEvent values 必须是有限 float32")
        if valid.dtype != np.bool_:
            raise FeatureShapeError("FeatureEvent valid 必须是 bool")
        if valid.ndim and valid.shape != values.shape:
            raise FeatureShapeError("FeatureEvent valid 必须与 values 同 shape")
        if not valid.ndim and np.any(values != 0.0) and not bool(valid):
            raise FeatureInputError("标量 valid=false 时 values 必须全为零")
        if valid.ndim and np.any(values[~valid] != 0.0):
            raise FeatureInputError("无效特征值必须严格为零")
        if self.schema is not None and values.ndim != 1:
            raise FeatureShapeError("带 schema 的 FeatureEvent values 必须是一维向量")
        if self.schema is not None and values.shape != (self.schema.value_dim,):
            raise FeatureShapeError("FeatureEvent values 与 schema 宽度不一致")


@dataclass(frozen=True)
class FeatureSequence:
    """同一 parent 的因果有序特征序列。"""

    parent_sample_id: str
    sample_ids: tuple[str, ...]
    causal_event_indices: np.ndarray
    values: np.ndarray
    valid: np.ndarray
    schema: FeatureSchema | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_ids = tuple(str(value) for value in self.sample_ids)
        indices = np.asarray(self.causal_event_indices)
        values = np.asarray(self.values)
        valid = np.asarray(self.valid)
        if not sample_ids or len(sample_ids) != len(set(sample_ids)):
            raise FeatureInputError("FeatureSequence sample_id 必须非空且唯一")
        if indices.dtype.kind not in "iu" or indices.shape != (len(sample_ids),):
            raise FeatureShapeError("FeatureSequence causal_event_indices shape/dtype 无效")
        if not np.array_equal(indices, np.arange(len(sample_ids), dtype=indices.dtype)):
            raise FeatureInputError("FeatureSequence event index 必须从 0 连续排列")
        if values.dtype != np.float32 or values.ndim != 2:
            raise FeatureShapeError("FeatureSequence values 必须是 [event, feature] float32")
        if values.shape[0] != len(sample_ids) or not np.isfinite(values).all():
            raise FeatureShapeError("FeatureSequence values 与 sample_id 不对齐或包含非有限值")
        if valid.dtype != np.bool_ or valid.shape not in {
            values.shape,
            values.shape[:1],
        }:
            raise FeatureShapeError("FeatureSequence valid shape/dtype 无效")
        if valid.shape == values.shape and np.any(values[~valid] != 0.0):
            raise FeatureInputError("无效序列特征值必须严格为零")
        if self.schema is not None and values.shape[1] != self.schema.value_dim:
            raise FeatureShapeError("FeatureSequence values 与 schema 宽度不一致")
        object.__setattr__(self, "sample_ids", sample_ids)


@dataclass(frozen=True)
class FeatureCacheDescriptor:
    """特征缓存与其输入版本绑定的最小 provenance。"""

    schema_version: str
    feature_name: str
    source_manifest_hash: str
    feature_config_hash: str
    view_name: str
    view_hash: str
    normalization_hash: str | None = None

    def __post_init__(self) -> None:
        fields = {
            "schema_version": self.schema_version,
            "feature_name": self.feature_name,
            "source_manifest_hash": self.source_manifest_hash,
            "feature_config_hash": self.feature_config_hash,
            "view_name": self.view_name,
            "view_hash": self.view_hash,
        }
        if any(not str(value) for value in fields.values()):
            raise FeatureSchemaError("FeatureCacheDescriptor 关键字段不能为空")
        if self.normalization_hash is not None and not str(self.normalization_hash):
            raise FeatureSchemaError("normalization_hash 不能为空字符串")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "feature_name": self.feature_name,
            "source_manifest_hash": self.source_manifest_hash,
            "feature_config_hash": self.feature_config_hash,
            "view_name": self.view_name,
            "view_hash": self.view_hash,
            "normalization_hash": self.normalization_hash,
        }


@dataclass(frozen=True)
class FeatureNormalization:
    """只允许由训练视图拟合的稳健归一化统计量。"""

    feature_names: tuple[str, ...]
    median: np.ndarray
    scale: np.ndarray
    fit_split: str
    parent_count_used: int
    scale_floor: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = _as_tuple(self.feature_names, name="feature_names")
        median = np.asarray(self.median)
        scale = np.asarray(self.scale)
        if median.dtype != np.float32 or scale.dtype != np.float32:
            raise FeatureShapeError("归一化 median/scale 必须是 float32")
        if median.shape != scale.shape or median.shape != (len(names),):
            raise FeatureShapeError("归一化向量与 feature_names 宽度不一致")
        if (
            not np.isfinite(median).all()
            or not np.isfinite(scale).all()
            or np.any(scale <= 0.0)
        ):
            raise FeatureNumericalError("归一化统计量必须有限且 scale>0")
        if self.fit_split != "train" or int(self.parent_count_used) <= 0:
            raise FeatureInputError("归一化统计量必须来自非空 train parent 集合")
        object.__setattr__(self, "feature_names", names)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "fit_split": self.fit_split,
            "parent_count_used": int(self.parent_count_used),
            "scale_floor": float(self.scale_floor),
            "metadata": dict(self.metadata),
        }


__all__ = [
    "FeatureCacheError",
    "FeatureCacheFormatError",
    "FeatureCacheIntegrityError",
    "FeatureCachePathError",
    "FeatureError",
    "FeatureEvent",
    "FeatureInputError",
    "FeatureNormalization",
    "FeatureNumericalError",
    "FeatureSchema",
    "FeatureSchemaError",
    "FeatureSequence",
    "FeatureShapeError",
    "FeatureCacheDescriptor",
    "RawEchoFrame",
    "RawEchoProvider",
    "RawTargetRecord",
    "event_values",
]
