"""独立的 raw RD -> RadarInsight canonical RD 转换器。

该模块不依赖 ``src-v3``。输入是带有物理 range/velocity 轴的二维
amplitude/phase frame，输出是当前模型使用的 ``[2, 21, 384]`` canonical
frame。物理轴映射发生在数据层，模型和前端协议不再各自实现 resize。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import DataFormatError, DataIntegrityError


CANONICAL_RANGE_BINS = 21
CANONICAL_VELOCITY_BINS = 384
CANONICAL_VELOCITY_MIN_MPS = -85.0
CANONICAL_VELOCITY_MAX_MPS = 85.0
CANONICAL_RANGE_OFFSETS_M = np.linspace(
    -75.0, 75.0, CANONICAL_RANGE_BINS, dtype=np.float32
)
# The float32 grid is converted to float64 before hashing, matching the
# historical canonical tensor contract.
CANONICAL_VELOCITY_MPS = np.linspace(
    CANONICAL_VELOCITY_MIN_MPS,
    CANONICAL_VELOCITY_MAX_MPS,
    CANONICAL_VELOCITY_BINS,
    dtype=np.float32,
)
CANONICAL_VELOCITY_AXIS_SHA256 = hashlib.sha256(
    np.ascontiguousarray(CANONICAL_VELOCITY_MPS.astype(np.float64)).tobytes(
        order="C"
    )
).hexdigest()
CANONICALIZATION_VERSION = "RADAR-LN3-CANONICAL-RD-INTERP-V1"


@dataclass(frozen=True)
class CanonicalRDFrame:
    """一个 canonical RD frame。"""

    amplitude: np.ndarray
    phase: np.ndarray
    velocity_mps: np.ndarray
    range_offset_m: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        amplitude = np.asarray(self.amplitude)
        phase = np.asarray(self.phase)
        velocity = np.asarray(self.velocity_mps)
        offsets = np.asarray(self.range_offset_m)
        if amplitude.shape != (CANONICAL_RANGE_BINS, CANONICAL_VELOCITY_BINS):
            raise DataFormatError("canonical amplitude 必须是 [21,384]")
        if phase.shape != amplitude.shape:
            raise DataFormatError("canonical phase 与 amplitude shape 不一致")
        if velocity.shape != (CANONICAL_VELOCITY_BINS,):
            raise DataFormatError("canonical velocity axis 必须是 [384]")
        if offsets.shape != (CANONICAL_RANGE_BINS,):
            raise DataFormatError("canonical range offsets 必须是 [21]")
        for name, value in (
            ("amplitude", amplitude),
            ("phase", phase),
            ("velocity_mps", velocity),
            ("range_offset_m", offsets),
        ):
            if not np.isfinite(value).all():
                raise DataIntegrityError(f"canonical {name} 包含非有限值")
        if not np.all(np.diff(velocity.astype(np.float64)) > 0.0):
            raise DataFormatError("canonical velocity axis 必须严格递增")


def canonical_axis_descriptor() -> dict[str, Any]:
    """返回前端/服务端可以登记的 canonical 轴描述。"""

    return {
        "version": CANONICALIZATION_VERSION,
        "range_bins": CANONICAL_RANGE_BINS,
        "velocity_bins": CANONICAL_VELOCITY_BINS,
        "range_offsets_m": CANONICAL_RANGE_OFFSETS_M.astype(float).tolist(),
        "velocity_min_mps": CANONICAL_VELOCITY_MIN_MPS,
        "velocity_max_mps": CANONICAL_VELOCITY_MAX_MPS,
        "velocity_axis_sha256": CANONICAL_VELOCITY_AXIS_SHA256,
        "velocity_axis_mps": CANONICAL_VELOCITY_MPS.astype(float).tolist(),
        "interpolation": "separable_linear_physical_axis; phase_complex_plane",
    }


def _numeric_axis(value: Any, *, name: str) -> np.ndarray:
    axis = np.asarray(value, dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.isfinite(axis).all():
        raise DataFormatError(f"{name} 必须是至少两个有限数值")
    diff = np.diff(axis)
    if np.all(diff < 0.0):
        axis = axis[::-1].copy()
    elif not np.all(diff > 0.0):
        raise DataFormatError(f"{name} 必须严格递增或严格递减")
    return axis


def _prepare_grid(
    values: np.ndarray,
    *,
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    target_range_axis_m: np.ndarray,
    target_velocity_axis_mps: np.ndarray,
) -> np.ndarray:
    """沿物理 velocity、range 轴做两次线性插值。"""

    source = np.asarray(values, dtype=np.float64)
    if source.shape != (range_axis_m.size, velocity_axis_mps.size):
        raise DataFormatError("RD 数组 shape 与物理轴长度不一致")
    if not (
        target_velocity_axis_mps[0] >= velocity_axis_mps[0] - 1.0e-6
        and target_velocity_axis_mps[-1] <= velocity_axis_mps[-1] + 1.0e-6
    ):
        raise DataFormatError("source velocity axis 覆盖不足以生成 canonical axis")
    if not (
        target_range_axis_m[0] >= range_axis_m[0] - 1.0e-6
        and target_range_axis_m[-1] <= range_axis_m[-1] + 1.0e-6
    ):
        raise DataFormatError("source range axis 覆盖不足以生成 canonical axis")

    velocity_resampled = np.stack(
        [
            np.interp(
                target_velocity_axis_mps,
                velocity_axis_mps,
                row,
            )
            for row in source
        ],
        axis=0,
    )
    return np.stack(
        [
            np.interp(target_range_axis_m, range_axis_m, velocity_resampled[:, index])
            for index in range(target_velocity_axis_mps.size)
        ],
        axis=1,
    )


def _amplitude_to_db(amplitude: np.ndarray, mode: str) -> np.ndarray:
    value = np.asarray(amplitude, dtype=np.float64)
    normalized_mode = str(mode).strip().lower()
    if normalized_mode in {"db", "dbfs", "amplitude_db"}:
        result = value
    elif normalized_mode in {"power", "power_db", "linear_power"}:
        if np.any(value <= 0.0):
            raise DataIntegrityError("power RD amplitude 必须为正才能转换为 dB")
        result = 10.0 * np.log10(value)
    elif normalized_mode in {"magnitude", "magnitude_db", "linear_magnitude"}:
        if np.any(value <= 0.0):
            raise DataIntegrityError("magnitude RD amplitude 必须为正才能转换为 dB")
        result = 20.0 * np.log10(value)
    elif normalized_mode in {"raw", "none"}:
        result = value
    else:
        raise DataFormatError(f"不支持的 RD amplitude_mode：{mode!r}")
    if not np.isfinite(result).all():
        raise DataIntegrityError("RD amplitude 转换后包含非有限值")
    return result


def canonicalize_rd_frame(
    amplitude: Any,
    *,
    phase: Any | None,
    range_axis_m: Any,
    velocity_axis_mps: Any,
    target_range_m: float,
    amplitude_mode: str = "magnitude_db",
) -> CanonicalRDFrame:
    """把一个 raw frame 转成 `[21,384]` canonical RD。

    ``amplitude_mode`` 必须由数据源协议声明。当前前端样本的数值分布与
    训练 canonical amplitude 一致地支持 ``magnitude_db``（20*log10）；
    若前端最终声明为 power，则必须显式传入 ``power_db``。
    """

    raw_amplitude = np.asarray(amplitude, dtype=np.float64)
    if raw_amplitude.ndim != 2 or min(raw_amplitude.shape) < 2:
        raise DataFormatError("RD amplitude 必须是非空二维数组")
    if not np.isfinite(raw_amplitude).all():
        raise DataIntegrityError("RD amplitude 包含非有限值")
    original_range = np.asarray(range_axis_m, dtype=np.float64).reshape(-1)
    original_velocity = np.asarray(velocity_axis_mps, dtype=np.float64).reshape(-1)
    source_range = _numeric_axis(original_range, name="range_axis")
    source_velocity = _numeric_axis(original_velocity, name="velocity_axis")
    if raw_amplitude.shape != (source_range.size, source_velocity.size):
        raise DataFormatError("RD amplitude shape 与 range/velocity axis 不一致")
    # Axes can be descending; normalize the corresponding matrix order.
    amplitude_grid = raw_amplitude
    if original_range[0] > original_range[-1]:
        amplitude_grid = amplitude_grid[::-1, :]
    if original_velocity[0] > original_velocity[-1]:
        amplitude_grid = amplitude_grid[:, ::-1]

    source_phase = (
        np.zeros_like(amplitude_grid, dtype=np.float64)
        if phase is None
        else np.asarray(phase, dtype=np.float64)
    )
    if source_phase.shape != amplitude_grid.shape:
        raise DataFormatError("RD phase shape 与 amplitude 不一致")
    if not np.isfinite(source_phase).all():
        raise DataIntegrityError("RD phase 包含非有限值")
    original_range = np.asarray(range_axis_m, dtype=np.float64).reshape(-1)
    original_velocity = np.asarray(velocity_axis_mps, dtype=np.float64).reshape(-1)
    if original_range[0] > original_range[-1]:
        source_phase = source_phase[::-1, :]
    if original_velocity[0] > original_velocity[-1]:
        source_phase = source_phase[:, ::-1]

    target_range = float(target_range_m) + CANONICAL_RANGE_OFFSETS_M.astype(np.float64)
    if not math.isfinite(float(target_range_m)):
        raise DataFormatError("target_range_m 必须是有限数值")
    amplitude_db = _amplitude_to_db(amplitude_grid, amplitude_mode)
    amplitude_out = _prepare_grid(
        amplitude_db,
        range_axis_m=source_range,
        velocity_axis_mps=source_velocity,
        target_range_axis_m=target_range,
        target_velocity_axis_mps=CANONICAL_VELOCITY_MPS.astype(np.float64),
    )

    # Interpolating in the complex plane avoids a false large jump at +/- pi.
    phase_complex = np.exp(1j * source_phase)
    phase_real = _prepare_grid(
        phase_complex.real,
        range_axis_m=source_range,
        velocity_axis_mps=source_velocity,
        target_range_axis_m=target_range,
        target_velocity_axis_mps=CANONICAL_VELOCITY_MPS.astype(np.float64),
    )
    phase_imag = _prepare_grid(
        phase_complex.imag,
        range_axis_m=source_range,
        velocity_axis_mps=source_velocity,
        target_range_axis_m=target_range,
        target_velocity_axis_mps=CANONICAL_VELOCITY_MPS.astype(np.float64),
    )
    phase_out = np.arctan2(phase_imag, phase_real)
    return CanonicalRDFrame(
        amplitude=np.asarray(amplitude_out, dtype=np.float32),
        phase=np.asarray(phase_out, dtype=np.float32),
        velocity_mps=CANONICAL_VELOCITY_MPS.copy(),
        range_offset_m=CANONICAL_RANGE_OFFSETS_M.copy(),
        metadata={
            "canonicalization_version": CANONICALIZATION_VERSION,
            "amplitude_mode": str(amplitude_mode),
            "source_shape": [int(value) for value in raw_amplitude.shape],
            "target_shape": [CANONICAL_RANGE_BINS, CANONICAL_VELOCITY_BINS],
            "velocity_axis_sha256": CANONICAL_VELOCITY_AXIS_SHA256,
        },
    )


__all__ = [
    "CANONICALIZATION_VERSION",
    "CANONICAL_RANGE_BINS",
    "CANONICAL_RANGE_OFFSETS_M",
    "CANONICAL_VELOCITY_AXIS_SHA256",
    "CANONICAL_VELOCITY_BINS",
    "CANONICAL_VELOCITY_MPS",
    "CanonicalRDFrame",
    "canonical_axis_descriptor",
    "canonicalize_rd_frame",
]
