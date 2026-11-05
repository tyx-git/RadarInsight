"""当前事件 RD velocity spectrum 的物理摘要。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .contracts import FeatureInputError, FeatureNumericalError, FeatureShapeError


RD_PHYSICAL_SCHEMA = "current-event-rd-physical-spectrum-v1"
RD_PHYSICAL_MASK_MODE = "gating_only"
RD_PHYSICAL_NAMES = (
    "velocity_bandwidth_mps_over_85",
    "zero_velocity_energy_fraction_abs_le_2_mps",
    "velocity_spectral_entropy_over_log_384",
    "top10_velocity_energy_fraction",
)
RD_PHYSICAL_VALUE_DIM = len(RD_PHYSICAL_NAMES)
RD_VELOCITY_BIN_COUNT = 384
RD_VELOCITY_MIN_MPS = -85.0
RD_VELOCITY_MAX_MPS = 85.0
RD_ZERO_VELOCITY_HALF_WIDTH_MPS = 2.0
RD_TOP_VELOCITY_BIN_COUNT = 10


@dataclass(frozen=True)
class RDPhysicalEvent:
    values: np.ndarray
    valid: bool

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.shape != (RD_PHYSICAL_VALUE_DIM,) or values.dtype != np.float32:
            raise FeatureShapeError("RD physical values 必须是 float32 [4]")
        if not np.isfinite(values).all():
            raise FeatureNumericalError("RD physical values 不能包含非有限值")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise FeatureInputError("RD physical values 必须位于 [0,1]")
        if not self.valid and np.any(values != 0.0):
            raise FeatureInputError("invalid RD physical values 必须严格为零")


def velocity_axis_sha256(velocity_mps: np.ndarray) -> str:
    axis = np.ascontiguousarray(np.asarray(velocity_mps, dtype=np.float64))
    return hashlib.sha256(axis.tobytes(order="C")).hexdigest()


def validate_velocity_axis(
    velocity_mps: np.ndarray,
    *,
    minimum_mps: float | None = None,
    maximum_mps: float | None = None,
) -> np.ndarray:
    axis = np.asarray(velocity_mps, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2 or not np.isfinite(axis).all():
        raise FeatureShapeError("RD velocity axis 必须是一维有限数组且至少有两个 bin")
    if not np.all(np.diff(axis) > 0.0):
        raise FeatureInputError("RD velocity axis 必须严格递增")
    if minimum_mps is not None and not math.isclose(
        float(axis[0]), float(minimum_mps), abs_tol=1.0e-6
    ):
        raise FeatureInputError("RD velocity axis 最小值与声明不一致")
    if maximum_mps is not None and not math.isclose(
        float(axis[-1]), float(maximum_mps), abs_tol=1.0e-6
    ):
        raise FeatureInputError("RD velocity axis 最大值与声明不一致")
    return axis


def extract_current_event_rd_physical(
    amplitude_db: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    frame_mask: np.ndarray | Iterable[bool] | None = None,
    zero_velocity_half_width_mps: float = RD_ZERO_VELOCITY_HALF_WIDTH_MPS,
    top_velocity_bin_count: int = RD_TOP_VELOCITY_BIN_COUNT,
    velocity_scale_mps: float | None = None,
) -> RDPhysicalEvent:
    """从当前事件的原始 amplitude 提取 4 维物理摘要。

    amplitude 必须是尚未进行模型输入归一化的 ``[frame, range, velocity]``。
    每帧先按本帧最大值做 log-power 相对化，然后聚合当前事件的有效帧和
    range bin；不消费历史或未来事件。
    """

    axis = validate_velocity_axis(velocity_mps)
    amplitude = np.asarray(amplitude_db)
    if amplitude.ndim != 3 or amplitude.shape[0] <= 0 or amplitude.shape[1] <= 0:
        raise FeatureShapeError("RD physical amplitude 必须是非空三维数组")
    if amplitude.shape[2] != axis.size or not np.issubdtype(amplitude.dtype, np.number):
        raise FeatureShapeError("RD physical amplitude 与 velocity axis 不对齐")
    if not np.isfinite(amplitude).all():
        raise FeatureNumericalError("RD physical amplitude 不能包含非有限值")
    if frame_mask is None:
        active = np.ones((amplitude.shape[0],), dtype=np.bool_)
    else:
        active = np.asarray(frame_mask, dtype=np.bool_)
        if active.shape != (amplitude.shape[0],):
            raise FeatureShapeError("RD physical frame_mask shape 无效")
    selected = np.asarray(amplitude[active], dtype=np.float64)
    if selected.shape[0] == 0:
        return RDPhysicalEvent(
            np.zeros((RD_PHYSICAL_VALUE_DIM,), dtype=np.float32), False
        )
    frame_maximum = selected.max(axis=(1, 2), keepdims=True)
    # 每帧最大值已经消除绝对幅度；下限避免极端但有限输入在幂运算中
    # 产生不必要的 underflow/overflow 传播。
    relative_power = np.power(
        10.0, np.clip((selected - frame_maximum) / 10.0, -80.0, 0.0)
    )
    velocity_profile = relative_power.sum(axis=(0, 1), dtype=np.float64)
    total = float(velocity_profile.sum(dtype=np.float64))
    if not math.isfinite(total) or total <= 0.0:
        raise FeatureNumericalError("RD physical total power 无效")
    probability = velocity_profile / total
    mean_velocity = float(np.sum(probability * axis, dtype=np.float64))
    variance = float(
        np.sum(probability * np.square(axis - mean_velocity), dtype=np.float64)
    )
    scale = (
        float(velocity_scale_mps)
        if velocity_scale_mps is not None
        else float(max(abs(axis[0]), abs(axis[-1])))
    )
    if not math.isfinite(scale) or scale <= 0.0:
        raise FeatureInputError("RD physical velocity scale 必须为正有限值")
    zero_fraction = float(
        probability[np.abs(axis) <= float(zero_velocity_half_width_mps)].sum(
            dtype=np.float64
        )
    )
    nonzero = probability[probability > 0.0]
    entropy = float(
        -np.sum(nonzero * np.log(nonzero), dtype=np.float64)
        / math.log(float(axis.size))
    )
    top_count = min(max(1, int(top_velocity_bin_count)), probability.size)
    top_fraction = float(
        np.partition(probability, -top_count)[-top_count:].sum(dtype=np.float64)
    )
    values = np.asarray(
        [math.sqrt(max(0.0, variance)) / scale, zero_fraction, entropy, top_fraction],
        dtype=np.float32,
    )
    if not np.isfinite(values).all():
        raise FeatureNumericalError("RD physical values 产生非有限值")
    values = np.clip(values, 0.0, 1.0).astype(np.float32, copy=False)
    return RDPhysicalEvent(values, True)


def extract_rd_physical(*args: Any, **kwargs: Any) -> RDPhysicalEvent:
    """``extract_current_event_rd_physical`` 的通用短名称。"""

    return extract_current_event_rd_physical(*args, **kwargs)


def rd_physical_schema() -> dict[str, Any]:
    return {
        "schema": RD_PHYSICAL_SCHEMA,
        "value_dim": RD_PHYSICAL_VALUE_DIM,
        "value_names": list(RD_PHYSICAL_NAMES),
        "value_dtype": "float32",
        "value_bounds": [0.0, 1.0],
        "valid_dtype": "bool",
        "valid_mask_mode": RD_PHYSICAL_MASK_MODE,
        "invalid_value": [0.0] * RD_PHYSICAL_VALUE_DIM,
        "input": "current event raw RD amplitude before model normalization; [frame,range,velocity]",
        "power_formula": "10 ** ((amplitude_db - per_frame_max_db) / 10)",
        "profile_reduction": "sum across current-event valid frames and range bins",
        "features": {
            RD_PHYSICAL_NAMES[0]: "power-weighted velocity standard deviation divided by velocity_scale_mps",
            RD_PHYSICAL_NAMES[1]: "velocity-profile power fraction where abs(velocity)<=2 m/s",
            RD_PHYSICAL_NAMES[2]: "velocity-profile Shannon entropy divided by log(bin_count)",
            RD_PHYSICAL_NAMES[3]: "power fraction in the ten largest velocity bins",
        },
        "velocity_axis": "caller supplied, strictly increasing, hashable",
        "causal_scope": "current RD event only",
        "missing_policy": "no valid frame gives exact-zero values and valid=false",
    }


def rd_physical_contract() -> dict[str, Any]:
    return rd_physical_schema()


__all__ = [
    "RD_PHYSICAL_MASK_MODE",
    "RD_PHYSICAL_NAMES",
    "RD_PHYSICAL_SCHEMA",
    "RD_PHYSICAL_VALUE_DIM",
    "RD_VELOCITY_BIN_COUNT",
    "RD_VELOCITY_MAX_MPS",
    "RD_VELOCITY_MIN_MPS",
    "RD_TOP_VELOCITY_BIN_COUNT",
    "RD_ZERO_VELOCITY_HALF_WIDTH_MPS",
    "RDPhysicalEvent",
    "extract_current_event_rd_physical",
    "extract_rd_physical",
    "rd_physical_contract",
    "rd_physical_schema",
    "validate_velocity_axis",
    "velocity_axis_sha256",
]
