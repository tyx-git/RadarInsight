"""RD 帧规范化、掩码和当前事件质量事实。

RD 内容和 RD quality 是两条不同的输入：内容保留每个事件的帧数组，quality
只描述当前事件可见的帧数量、局部时长和缺失事实。这里不读取 CQ08 字段，也
不计算 RD physical 或 micro-Doppler。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .contracts import FeatureInputError, FeatureNumericalError, FeatureShapeError


RD_QUALITY_SCHEMA = "causal-current-event-v1"
RD_QUALITY_NAMES = (
    "rd_present",
    "rd_frame_count_scaled",
    "rd_local_duration_scaled",
    "rd_missing",
    "rd_local_time_non_monotonic",
    "rd_multiple_frames",
    "rd_frames_truncated",
)
RD_QUALITY_DIM = len(RD_QUALITY_NAMES)
RD_PHASE_SCALE = np.pi
RD_DEFAULT_DURATION_SCALE_SECONDS = 0.5
RD_CANONICAL_VELOCITY_AXIS_HASH = (
    "5912f04e9119f420c668074b3f73dbfeabe1d466e4ad701f9824a1e0598b03fe"
)


@dataclass(frozen=True)
class RDFrameData:
    """一个事件选中的 RD 帧及其局部元数据。"""

    amplitude: np.ndarray
    phase: np.ndarray
    time_seconds: np.ndarray
    frame_mask: np.ndarray
    original_count: int
    truncated: bool
    local_time_non_monotonic: bool

    def __post_init__(self) -> None:
        amplitude = np.asarray(self.amplitude)
        phase = np.asarray(self.phase)
        times = np.asarray(self.time_seconds)
        mask = np.asarray(self.frame_mask)
        if amplitude.ndim != 3 or amplitude.shape[0] <= 0:
            raise FeatureShapeError("RD amplitude 必须是 [frame, range, velocity] 且非空")
        if phase.shape != amplitude.shape:
            raise FeatureShapeError("RD phase 与 amplitude shape 不一致")
        if times.shape != (amplitude.shape[0],) or mask.shape != times.shape:
            raise FeatureShapeError("RD time/frame_mask 与 frame 数量不一致")
        if mask.dtype != np.bool_:
            raise FeatureShapeError("RD frame_mask 必须是 bool")
        if not np.isfinite(amplitude).all() or not np.isfinite(phase).all():
            raise FeatureNumericalError("RD amplitude/phase 不能包含非有限值")
        if not np.isfinite(times).all():
            raise FeatureNumericalError("RD frame time 不能包含非有限值")
        if int(self.original_count) < int(mask.sum()):
            raise FeatureInputError("RD original_count 不能小于当前有效帧数")
        if int(self.original_count) < 0:
            raise FeatureInputError("RD original_count 不能为负数")


def _empty_rd_frame_data() -> RDFrameData | None:
    return None


def _from_frame_records(frames: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not frames:
        raise FeatureInputError("RD frames 不能为空")
    amplitudes: list[np.ndarray] = []
    phases: list[np.ndarray] = []
    times: list[float] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, Mapping):
            raise FeatureInputError(f"RD frame {index} 必须是 mapping")
        amplitude = np.asarray(frame.get("amplitude"))
        if amplitude.ndim != 2:
            raise FeatureShapeError(f"RD frame {index} amplitude 必须是二维数组")
        phase = np.asarray(frame.get("phase", np.zeros_like(amplitude)))
        if phase.shape != amplitude.shape:
            raise FeatureShapeError(f"RD frame {index} phase shape 不一致")
        try:
            time = float(frame.get("time_seconds", 0.0))
        except (TypeError, ValueError) as exc:
            raise FeatureInputError(f"RD frame {index} time 无法解析") from exc
        amplitudes.append(amplitude)
        phases.append(phase)
        times.append(time)
    try:
        amplitude_array = np.stack(amplitudes, axis=0)
        phase_array = np.stack(phases, axis=0)
    except ValueError as exc:
        raise FeatureShapeError("RD frames 的 range/velocity shape 不一致") from exc
    return amplitude_array, phase_array, np.asarray(times, dtype=np.float64)


def prepare_rd_event(
    values: Mapping[str, Any] | None,
    *,
    max_frames: int | None = None,
) -> RDFrameData | None:
    """从通用 ``rd_values`` 提取当前事件的帧，不执行数值归一化。"""

    if values is None:
        return _empty_rd_frame_data()
    if not isinstance(values, Mapping):
        raise FeatureInputError("rd_values 必须是 mapping 或 None")
    frames = values.get("frames")
    if isinstance(frames, Sequence) and not isinstance(frames, (str, bytes)):
        amplitude, phase, times = _from_frame_records(frames)
    else:
        if "amplitude" not in values:
            raise FeatureInputError("RD values 缺少 amplitude")
        amplitude = np.asarray(values["amplitude"])
        if amplitude.ndim == 2:
            amplitude = amplitude[None, ...]
        if amplitude.ndim != 3:
            raise FeatureShapeError("RD amplitude 必须是 [frame, range, velocity]")
        phase = np.asarray(values.get("phase", np.zeros_like(amplitude)))
        if phase.ndim == 2:
            phase = phase[None, ...]
        if phase.shape != amplitude.shape:
            raise FeatureShapeError("RD phase 与 amplitude shape 不一致")
        times = np.asarray(
            values.get("time_seconds", np.zeros((amplitude.shape[0],), dtype=np.float64)),
            dtype=np.float64,
        )
        if times.ndim == 0:
            times = np.repeat(times[None], amplitude.shape[0])
    if times.shape != (amplitude.shape[0],):
        raise FeatureShapeError("RD time_seconds 与 frame 数量不一致")
    if not np.issubdtype(amplitude.dtype, np.number) or not np.issubdtype(
        phase.dtype, np.number
    ):
        raise FeatureShapeError("RD amplitude/phase 必须是数值数组")
    if not np.isfinite(amplitude).all() or not np.isfinite(phase).all():
        raise FeatureNumericalError("RD amplitude/phase 不能包含非有限值")
    if not np.isfinite(times).all():
        raise FeatureNumericalError("RD time_seconds 不能包含非有限值")

    raw_mask = values.get("frame_mask")
    if raw_mask is None:
        active = np.ones((amplitude.shape[0],), dtype=np.bool_)
    else:
        active = np.asarray(raw_mask, dtype=np.bool_)
        if active.shape != (amplitude.shape[0],):
            raise FeatureShapeError("RD frame_mask shape 不一致")
    selected = np.flatnonzero(active)
    original_count = int(values.get("original_count", values.get("frame_count", selected.size)))
    if original_count < int(selected.size):
        raise FeatureInputError("RD original_count 小于 active frame 数量")
    limit = None if max_frames is None else max(1, int(max_frames))
    truncated = bool(values.get("truncated", False))
    if limit is not None and selected.size > limit:
        selected = selected[:limit]
        truncated = True
    if selected.size == 0:
        return _empty_rd_frame_data()
    selected_amplitude = np.asarray(amplitude[selected], dtype=np.float32)
    selected_phase = np.asarray(phase[selected], dtype=np.float32)
    selected_times = np.asarray(times[selected], dtype=np.float32)
    local_non_monotonic = bool(
        selected_times.size > 1
        and (np.diff(selected_times.astype(np.float64)) < -1.0e-8).any()
    )
    return RDFrameData(
        amplitude=selected_amplitude,
        phase=selected_phase,
        time_seconds=selected_times,
        frame_mask=np.ones((selected.size,), dtype=np.bool_),
        original_count=original_count,
        truncated=truncated,
        local_time_non_monotonic=local_non_monotonic,
    )


def resolve_rd_velocity_axis(values: Mapping[str, Any] | None) -> np.ndarray:
    """Resolve the physical velocity axis carried by a canonical RD event.

    Legacy materialized events do not carry an axis and retain the historical
    uniform fallback.  Canonical events carry the exact float32 axis from the
    canonical tensor; its hash is checked before physical features consume it.
    """

    if isinstance(values, Mapping) and values.get("velocity_mps") is not None:
        axis = np.asarray(values["velocity_mps"], dtype=np.float64)
        if axis.shape != (384,) or not np.isfinite(axis).all() or not np.all(np.diff(axis) > 0.0):
            raise FeatureShapeError("canonical RD velocity_mps 轴无效")
        digest = hashlib.sha256(
            np.ascontiguousarray(axis).tobytes(order="C")
        ).hexdigest()
        if digest != RD_CANONICAL_VELOCITY_AXIS_HASH:
            raise FeatureInputError("canonical RD velocity_mps 轴 hash 不一致")
        return axis
    return np.linspace(-85.0, 85.0, 384, dtype=np.float64)


def normalize_rd_amplitude(
    amplitude: np.ndarray,
    *,
    center: float,
    scale: float,
    clip_absolute: float = 6.0,
) -> np.ndarray:
    """按训练视图冻结的统计量规范化 RD amplitude。"""

    array = np.asarray(amplitude, dtype=np.float32)
    if not np.isfinite(array).all():
        raise FeatureNumericalError("RD amplitude 归一化输入包含非有限值")
    if not np.isfinite(float(center)) or not np.isfinite(float(scale)):
        raise FeatureInputError("RD normalization center/scale 必须有限")
    if float(scale) <= 0.0 or float(clip_absolute) <= 0.0:
        raise FeatureInputError("RD normalization scale/clip 必须为正")
    result = np.clip(
        (array - float(center)) / float(scale),
        -float(clip_absolute),
        float(clip_absolute),
    ).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise FeatureNumericalError("RD amplitude 归一化产生非有限值")
    return result


def normalize_rd_phase(phase: np.ndarray, *, scale: float = RD_PHASE_SCALE) -> np.ndarray:
    array = np.asarray(phase, dtype=np.float32)
    if not np.isfinite(array).all() or not np.isfinite(float(scale)) or float(scale) <= 0.0:
        raise FeatureInputError("RD phase 归一化输入无效")
    return np.clip(array / float(scale), -1.0, 1.0).astype(np.float32, copy=False)


def build_rd_model_frames(
    event: RDFrameData | None,
    *,
    amplitude_center: float,
    amplitude_scale: float,
    max_frames: int,
    spatial_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """生成模型使用的 ``[frame,2,range,velocity]`` 和帧 mask。"""

    limit = max(1, int(max_frames))
    if event is None:
        if spatial_shape is None or len(spatial_shape) != 2 or any(
            int(size) <= 0 for size in spatial_shape
        ):
            raise FeatureShapeError(
                "RD 缺失事件生成模型帧时必须显式提供 spatial_shape"
            )
        return np.zeros(
            (limit, 2, int(spatial_shape[0]), int(spatial_shape[1])),
            dtype=np.float32,
        ), np.zeros((limit,), dtype=np.bool_)
    count = min(event.amplitude.shape[0], limit)
    normalized_amplitude = normalize_rd_amplitude(
        event.amplitude[:count], center=amplitude_center, scale=amplitude_scale
    )
    normalized_phase = normalize_rd_phase(event.phase[:count])
    if spatial_shape is not None and tuple(int(size) for size in spatial_shape) != tuple(
        event.amplitude.shape[1:]
    ):
        raise FeatureShapeError("RD event shape 与声明的 spatial_shape 不一致")
    frames = np.zeros(
        (limit, 2, event.amplitude.shape[1], event.amplitude.shape[2]),
        dtype=np.float32,
    )
    frames[:count, 0] = normalized_amplitude
    frames[:count, 1] = normalized_phase
    mask = np.zeros((limit,), dtype=np.bool_)
    mask[:count] = event.frame_mask[:count]
    return frames, mask


def encode_causal_rd_quality(
    *,
    present: bool,
    original_count: int,
    duration_seconds: float,
    local_time_non_monotonic: bool,
    truncated: bool,
    max_rd_frames_per_event: int | None = None,
    max_rd_frames_per_point: int | None = None,
    duration_scale_seconds: float = RD_DEFAULT_DURATION_SCALE_SECONDS,
) -> np.ndarray:
    """只使用当前事件信息编码 7 维 RD quality。"""

    if max_rd_frames_per_event is None:
        max_rd_frames_per_event = max_rd_frames_per_point
    elif (
        max_rd_frames_per_point is not None
        and int(max_rd_frames_per_event) != int(max_rd_frames_per_point)
    ):
        raise FeatureInputError("RD quality 的 max frame 参数不一致")
    if max_rd_frames_per_event is None:
        raise FeatureInputError("RD quality 必须提供 max frame 参数")
    count = max(0, int(original_count))
    max_frames = max(1, int(max_rd_frames_per_event))
    duration = max(0.0, float(duration_seconds))
    if not np.isfinite(duration) or duration_scale_seconds <= 0.0:
        raise FeatureInputError("RD quality duration 参数无效")
    values = np.asarray(
        [
            float(bool(present)),
            float(np.clip(np.log1p(count) / np.log1p(max_frames), 0.0, 1.0)),
            float(np.clip(duration / float(duration_scale_seconds), 0.0, 1.0)),
            float(not bool(present)),
            float(bool(local_time_non_monotonic)),
            float(count > 1),
            float(bool(truncated)),
        ],
        dtype=np.float32,
    )
    if values.shape != (RD_QUALITY_DIM,) or not np.isfinite(values).all():
        raise FeatureNumericalError("RD quality 编码产生无效向量")
    return values


def encode_rd_quality_from_event(
    event: RDFrameData | None,
    *,
    max_rd_frames_per_event: int,
) -> np.ndarray:
    if event is None:
        return encode_causal_rd_quality(
            present=False,
            original_count=0,
            duration_seconds=0.0,
            local_time_non_monotonic=False,
            truncated=False,
            max_rd_frames_per_event=max_rd_frames_per_event,
        )
    duration = (
        float(max(0.0, event.time_seconds[-1] - event.time_seconds[0]))
        if event.time_seconds.size > 1
        else 0.0
    )
    return encode_causal_rd_quality(
        present=True,
        original_count=event.original_count,
        duration_seconds=duration,
        local_time_non_monotonic=event.local_time_non_monotonic,
        truncated=event.truncated,
        max_rd_frames_per_event=max_rd_frames_per_event,
    )


def rd_quality_contract() -> dict[str, Any]:
    """返回当前事件 RD quality 的版本化说明。"""

    return {
        "schema": RD_QUALITY_SCHEMA,
        "value_dim": RD_QUALITY_DIM,
        "value_names": list(RD_QUALITY_NAMES),
        "value_dtype": "float32",
        "value_bounds": [0.0, 1.0],
        "causal_scope": "current RD event only",
        "missing_policy": "rd_present=0 and rd_missing=1 when the event has no RD frame",
        "frame_count_policy": "log1p(count)/log1p(max_frames), clipped to [0,1]",
        "duration_policy": "local selected-frame duration divided by 0.5 seconds, clipped to [0,1]",
        "parent_summary_excluded": True,
    }


__all__ = [
    "RD_CANONICAL_VELOCITY_AXIS_HASH",
    "RD_DEFAULT_DURATION_SCALE_SECONDS",
    "RD_PHASE_SCALE",
    "RD_QUALITY_DIM",
    "RD_QUALITY_NAMES",
    "RD_QUALITY_SCHEMA",
    "RDFrameData",
    "build_rd_model_frames",
    "encode_causal_rd_quality",
    "encode_rd_quality_from_event",
    "normalize_rd_amplitude",
    "normalize_rd_phase",
    "prepare_rd_event",
    "resolve_rd_velocity_axis",
    "rd_quality_contract",
]
