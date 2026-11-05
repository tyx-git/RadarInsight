"""将通用 EventSequence 组装为模型输入 tensor。

本模块只做已经注册特征的编码和 batch padding，不定位数据源，不读取
CQ08 文件，也不实现 L1/L2 模型逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from radar_ln3.data.contracts import EventSequence, RawEchoProvider
from radar_ln3.data.temporal import encode_dt_reason

from .course import encode_causal_course_sequence
from .cache import FeatureCache
from .micro_doppler import STABLE8_INDICES, normalize_stable8
from .motion import (
    encode_causal_tr_motion_sequence,
    encode_causal_tr_observation_increment_sequence,
)
from .physical import extract_current_event_rd_physical
from .rd import (
    encode_rd_quality_from_event,
    build_rd_model_frames,
    prepare_rd_event,
    resolve_rd_velocity_axis,
)
from .tr import encode_tr_sequence


_TASK_LABEL_TO_ID = {
    "l1": {
        "DroneTarget": 0,
        "BirdTarget": 0,
        "BalloonTarget": 0,
        "ClutterTarget": 1,
    },
    "l2": {
        "DroneTarget": 0,
        "BirdTarget": 1,
        "BalloonTarget": 2,
    },
}


def _velocity_axis() -> np.ndarray:
    return np.linspace(-85.0, 85.0, 384, dtype=np.float64)


def _event_values(sequence: EventSequence) -> list[Mapping[str, Any]]:
    values: list[Mapping[str, Any]] = []
    for event in sequence.events:
        if not isinstance(event.tr_values, Mapping):
            raise ValueError(f"事件缺少 TR values：{event.sample_id}")
        values.append(event.tr_values)
    return values


def _stable_sidecar(
    event: Any,
    *,
    max_frames: int,
    stable_dim: int,
) -> tuple[np.ndarray, bool, np.ndarray, np.ndarray]:
    """读取上游显式注入的 stable8 侧车；缺失时返回无效零值。"""

    metadata = event.metadata if isinstance(event.metadata, Mapping) else {}
    values = np.asarray(
        metadata.get("stable8_values", np.zeros((stable_dim,), dtype=np.float32)),
        dtype=np.float32,
    )
    valid = bool(metadata.get("stable8_valid", False))
    frame_values = np.asarray(
        metadata.get(
            "stable8_frame_values",
            np.zeros((max_frames, stable_dim), dtype=np.float32),
        ),
        dtype=np.float32,
    )
    frame_mask = np.asarray(
        metadata.get("stable8_frame_mask", np.zeros((max_frames,), dtype=np.bool_)),
        dtype=np.bool_,
    )
    if values.shape != (stable_dim,) or frame_values.shape != (max_frames, stable_dim):
        raise ValueError(f"stable8 侧车 shape 无效：{event.sample_id}")
    if frame_mask.shape != (max_frames,) or not np.isfinite(values).all() or not np.isfinite(frame_values).all():
        raise ValueError(f"stable8 侧车数值无效：{event.sample_id}")
    if np.any(frame_values[~frame_mask] != 0.0) or (not valid and np.any(values != 0.0)):
        raise ValueError(f"stable8 侧车无效值必须为零：{event.sample_id}")
    if valid != bool(frame_mask.any()):
        raise ValueError(f"stable8 侧车 validity 与 frame mask 不一致：{event.sample_id}")
    return values, valid, frame_values, frame_mask


def _cache_positions(
    feature_caches: Mapping[str, FeatureCache] | None,
    *,
    expected_view_hash: str | None,
) -> dict[str, dict[str, int]]:
    if feature_caches is None:
        return {}
    result: dict[str, dict[str, int]] = {}
    view_hashes: set[str] = set()
    for name, cache in feature_caches.items():
        if not isinstance(cache, FeatureCache):
            raise ValueError(f"feature cache 类型无效：{name}")
        view_hashes.add(cache.descriptor.view_hash)
        positions = dict(cache.sample_positions)
        if len(positions) != len(cache.sample_ids):
            raise ValueError(f"feature cache sample_id 不唯一：{name}")
        result[str(name)] = positions
    if len(view_hashes) > 1:
        raise ValueError("模型 batch 不能混用不同 view 的 feature cache")
    if expected_view_hash is not None and view_hashes != {str(expected_view_hash)}:
        raise ValueError("feature cache 与当前 DatasetView 的 view_hash 不一致")
    return result


def _cache_row(
    feature_caches: Mapping[str, FeatureCache] | None,
    positions: Mapping[str, Mapping[str, int]],
    name: str,
    sample_id: str,
) -> tuple[FeatureCache, int] | None:
    if feature_caches is None or name not in feature_caches:
        return None
    index = positions[name].get(str(sample_id))
    if index is None:
        raise ValueError(f"feature cache 缺少事件：{name}:{sample_id}")
    return feature_caches[name], index


def _tr_missing_from_valid(valid: np.ndarray) -> np.ndarray:
    mask = np.asarray(valid, dtype=np.bool_)
    if mask.shape != (14,):
        raise ValueError("tr_observation cache validity shape 无效")
    missing = np.zeros((11,), dtype=np.float32)
    missing[0] = not mask[0]
    missing[1] = not mask[1]
    missing[2] = not mask[2] or not mask[3]
    missing[3] = not mask[4] or not mask[5]
    missing[4] = not mask[6] or not mask[7]
    missing[5] = not mask[8]
    missing[6] = not mask[9]
    missing[7] = not mask[10]
    missing[8] = not mask[11]
    missing[9] = not mask[12]
    missing[10] = not mask[13]
    return missing


def build_model_batch(
    sequences: Sequence[EventSequence],
    *,
    task: str,
    max_rd_frames: int = 12,
    max_sequence_length: int = 29,
    rd_amplitude_center: float = 55.90625,
    rd_amplitude_scale: float = 6.942990104412662,
    rd_shape: tuple[int, int] = (21, 384),
    stable_dim: int = 8,
    stable_indices: Sequence[int] = STABLE8_INDICES,
    clipping: Mapping[str, Any] | None = None,
    raw_echo_provider: RawEchoProvider | None = None,
    stable_median: np.ndarray | None = None,
    stable_scale: np.ndarray | None = None,
    feature_caches: Mapping[str, FeatureCache] | None = None,
    expected_view_hash: str | None = None,
    baseline_anchor: Any | None = None,
    stable_input_mode: str = "normalized",
    allow_unmapped_labels: bool = False,
    pad_to_max_sequence_length: bool = False,
) -> dict[str, Any]:
    """把 parent 序列编码成 L1/L2 共用的 tensor 输入。

    超过 ``max_sequence_length`` 的 parent 保留因果顺序并截取前缀。原始
    manifest、EventSequence 和 feature cache 仍保留完整事件，截断只发生在
    进入模型的 batch 边界。
    """

    if not sequences:
        raise ValueError("模型 batch 不能为空")
    if task not in {"l1", "l2"}:
        raise ValueError(f"不支持的模型 task：{task}")
    if isinstance(max_sequence_length, bool) or int(max_sequence_length) <= 0:
        raise ValueError("max_sequence_length 必须是正整数")
    if stable_input_mode not in {"normalized", "raw"}:
        raise ValueError("stable_input_mode 必须是 normalized 或 raw")
    if baseline_anchor is not None and task != "l2":
        raise ValueError("baseline anchor 只允许用于 L2 batch")
    stable_indices = tuple(int(value) for value in stable_indices)
    if len(stable_indices) != stable_dim or len(set(stable_indices)) != len(stable_indices):
        raise ValueError("stable_indices 必须是唯一且与 stable_dim 等长的序列")
    if any(value < 0 or value >= 23 for value in stable_indices):
        raise ValueError("stable_indices 超出 23 维 evidence 范围")
    if raw_echo_provider is not None and (
        stable_median is None or stable_scale is None
    ):
        raise ValueError("注入 raw_echo_provider 时必须同时提供 stable8 normalization")
    if stable_median is not None or stable_scale is not None:
        if stable_median is None or stable_scale is None:
            raise ValueError("stable8 median 和 scale 必须同时提供")
        stable_median = np.asarray(stable_median, dtype=np.float32)
        stable_scale = np.asarray(stable_scale, dtype=np.float32)
        if stable_median.shape != (stable_dim,) or stable_scale.shape != (stable_dim,):
            raise ValueError("stable8 normalization shape 无效")
    cache_positions = _cache_positions(
        feature_caches,
        expected_view_hash=expected_view_hash,
    )
    use_cached_tr_features = clipping is None
    original_lengths = [len(item.events) for item in sequences]
    truncated_flags = [
        length > int(max_sequence_length) for length in original_lengths
    ]
    sequences = tuple(
        replace(sequence, events=list(sequence.events[: int(max_sequence_length)]))
        if truncated
        else sequence
        for sequence, truncated in zip(sequences, truncated_flags, strict=True)
    )
    lengths = [len(item.events) for item in sequences]
    if any(length <= 0 for length in lengths):
        raise ValueError("parent 不能没有事件")
    width = int(max_sequence_length) if pad_to_max_sequence_length else max(lengths)
    batch_size = len(sequences)
    tr_values = np.zeros((batch_size, width, 14), dtype=np.float32)
    tr_missing = np.zeros((batch_size, width, 11), dtype=np.float32)
    reference_tr_values = np.zeros((batch_size, width, 14), dtype=np.float32)
    reference_tr_missing = np.zeros((batch_size, width, 11), dtype=np.float32)
    dt_seconds = np.zeros((batch_size, width), dtype=np.float32)
    dt_valid = np.zeros((batch_size, width), dtype=np.bool_)
    dt_reason = np.zeros((batch_size, width), dtype=np.int64)
    padding = np.ones((batch_size, width), dtype=np.bool_)
    rd_frames = np.zeros(
        (batch_size, width, max(1, int(max_rd_frames)), 2, *rd_shape),
        dtype=np.float32,
    )
    rd_mask = np.zeros((batch_size, width, max(1, int(max_rd_frames))), dtype=np.bool_)
    rd_quality = np.zeros((batch_size, width, 7), dtype=np.float32)
    increment = np.zeros((batch_size, width, 9), dtype=np.float32)
    motion = np.zeros((batch_size, width, 9), dtype=np.float32)
    course = np.zeros((batch_size, width, 1), dtype=np.float32)
    course_valid = np.zeros((batch_size, width, 1), dtype=np.bool_)
    course_support = np.zeros((batch_size, width), dtype=np.int64)
    physical = np.zeros((batch_size, width, 4), dtype=np.float32)
    physical_valid = np.zeros((batch_size, width), dtype=np.bool_)
    micro = np.zeros((batch_size, width, 8), dtype=np.float32)
    micro_valid = np.zeros((batch_size, width), dtype=np.bool_)
    stable_summary = np.zeros((batch_size, width, stable_dim), dtype=np.float32)
    stable_summary_valid = np.zeros((batch_size, width), dtype=np.bool_)
    stable_frame_values = np.zeros(
        (batch_size, width, max(1, int(max_rd_frames)), stable_dim),
        dtype=np.float32,
    )
    stable_frame_mask = np.zeros(
        (batch_size, width, max(1, int(max_rd_frames))), dtype=np.bool_
    )
    baseline_logits: np.ndarray | None = None
    if baseline_anchor is not None:
        class_count = int(getattr(baseline_anchor, "class_count", 0))
        if class_count <= 0:
            raise ValueError("baseline anchor 类别数量无效")
        baseline_logits = np.zeros(
            (batch_size, width, class_count), dtype=np.float32
        )
    labels = np.zeros((batch_size,), dtype=np.int64)
    label_names: list[str] = []
    parent_ids: list[str] = []

    for row, sequence in enumerate(sequences):
        reference_values = [dict(item) for item in _event_values(sequence)]
        values = [dict(item) for item in reference_values]
        if clipping is not None:
            bounds = clipping.get("bounds")
            if not isinstance(bounds, Mapping):
                raise ValueError("TR clipping bounds 无效")
            for item in values:
                for name in ("range_m", "speed_mps"):
                    bound = bounds.get(name)
                    if not isinstance(bound, Mapping):
                        raise ValueError(f"TR clipping 缺少 {name} bounds")
                    raw = item.get(name)
                    try:
                        number = float(raw)
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(number):
                        item[name] = float(
                            np.clip(number, float(bound["lower"]), float(bound["upper"]))
                        )
        encoded, missing = encode_tr_sequence(values)
        reference_encoded, reference_missing = encode_tr_sequence(reference_values)
        increments, _ = encode_causal_tr_observation_increment_sequence(values)
        if task == "l2":
            motion_dt = [event.tr_dt_seconds for event in sequence.events]
            motion_dt_valid = [event.tr_dt_valid for event in sequence.events]
            if any(value is None for value in motion_dt) or any(
                value is None for value in motion_dt_valid
            ):
                raise ValueError(
                    f"L2 motion 特征缺少统一 TR timing：{sequence.parent_sample_id}"
                )
            motion_values, _ = encode_causal_tr_motion_sequence(
                values,
                dt_seconds=[float(value) for value in motion_dt],
                dt_valid=[bool(value) for value in motion_dt_valid],
            )
            motion[row, : len(values)] = motion_values
        course_values, course_mask, support = encode_causal_course_sequence(values)
        tr_values[row, : len(values)] = encoded
        tr_missing[row, : len(values)] = missing
        reference_tr_values[row, : len(reference_values)] = reference_encoded
        reference_tr_missing[row, : len(reference_values)] = reference_missing
        increment[row, : len(values)] = increments
        course[row, : len(values)] = course_values
        course_valid[row, : len(values)] = course_mask
        course_support[row, : len(values)] = support
        padding[row, : len(values)] = False
        if baseline_anchor is not None:
            events = sequence.events
            values_for_events = baseline_anchor.values_for_events(
                [str(event.sample_id) for event in events],
                [str(event.parent_sample_id) for event in events],
                [int(event.causal_event_index) for event in events],
                [str(event.split_name) for event in events],
            )
            expected_shape = (len(events), baseline_logits.shape[-1])
            if values_for_events.shape != expected_shape:
                raise ValueError(
                    "baseline anchor 事件 logits shape 无效："
                    f"expected={expected_shape}, actual={values_for_events.shape}"
                )
            if not np.isfinite(values_for_events).all():
                raise ValueError("baseline anchor 事件 logits 包含非有限值")
            baseline_logits[row, : len(events)] = values_for_events
        for event_index, event in enumerate(sequence.events):
            tr_cached = (
                _cache_row(
                    feature_caches,
                    cache_positions,
                    "tr_observation",
                    event.sample_id,
                )
                if use_cached_tr_features
                else None
            )
            if tr_cached is not None:
                cache, cache_index = tr_cached
                cached_values = np.asarray(cache.values[cache_index], dtype=np.float32)
                cached_valid = np.asarray(cache.valid[cache_index], dtype=np.bool_)
                if cached_values.shape != (14,) or cached_valid.shape != (14,):
                    raise ValueError("tr_observation cache row shape 无效")
                if not np.isfinite(cached_values).all():
                    raise ValueError("tr_observation cache row 包含非有限值")
                tr_values[row, event_index] = cached_values
                tr_missing[row, event_index] = _tr_missing_from_valid(cached_valid)
            increment_cached = (
                _cache_row(
                    feature_caches,
                    cache_positions,
                    "tr_observation_increment",
                    event.sample_id,
                )
                if use_cached_tr_features
                else None
            )
            if increment_cached is not None:
                cache, cache_index = increment_cached
                cached_values = np.asarray(cache.values[cache_index], dtype=np.float32)
                if cached_values.shape != (9,):
                    raise ValueError("tr_observation_increment cache row shape 无效")
                increment[row, event_index] = cached_values
            course_cached = _cache_row(
                feature_caches,
                cache_positions,
                "course_circular_incoherence",
                event.sample_id,
            )
            if course_cached is not None:
                cache, cache_index = course_cached
                cached_values = np.asarray(cache.values[cache_index], dtype=np.float32)
                cached_valid = np.asarray(cache.valid[cache_index], dtype=np.bool_)
                if cached_values.shape != (1,) or cached_valid.shape != (1,):
                    raise ValueError("course cache row shape 无效")
                course[row, event_index] = cached_values
                course_valid[row, event_index] = cached_valid
                supports = cache.metadata.get("course_support")
                if not isinstance(supports, list) or cache_index >= len(supports):
                    raise ValueError("course cache 缺少 support metadata")
                course_support[row, event_index] = int(supports[cache_index])
            quality_cached = _cache_row(
                feature_caches,
                cache_positions,
                "rd_quality",
                event.sample_id,
            )
            if event.tr_dt_seconds is None or event.tr_dt_valid is None or event.tr_dt_reason is None:
                raise ValueError(f"事件缺少统一 TR timing：{event.sample_id}")
            dt_seconds[row, event_index] = float(event.tr_dt_seconds)
            dt_valid[row, event_index] = bool(event.tr_dt_valid)
            dt_reason[row, event_index] = encode_dt_reason(
                event.tr_dt_reason, dt_valid=bool(event.tr_dt_valid)
            )
            prepared = prepare_rd_event(
                event.rd_values,
                max_frames=max_rd_frames,
            )
            frames, frame_mask = build_rd_model_frames(
                prepared,
                amplitude_center=rd_amplitude_center,
                amplitude_scale=rd_amplitude_scale,
                max_frames=max_rd_frames,
                spatial_shape=rd_shape,
            )
            if frames.shape[1:] != (2, *rd_shape):
                raise ValueError(
                    f"RD model frame shape differs: expected={(2, *rd_shape)}, actual={frames.shape[1:]}"
                )
            rd_frames[row, event_index, : frames.shape[0]] = frames
            rd_mask[row, event_index, : frame_mask.shape[0]] = frame_mask
            rd_quality[row, event_index] = encode_rd_quality_from_event(
                prepared,
                max_rd_frames_per_event=max_rd_frames,
            )
            if quality_cached is not None:
                cache, cache_index = quality_cached
                cached_values = np.asarray(cache.values[cache_index], dtype=np.float32)
                if cached_values.shape != (7,):
                    raise ValueError("rd_quality cache row shape 无效")
                rd_quality[row, event_index] = cached_values
            if prepared is not None:
                try:
                    physical_result = extract_current_event_rd_physical(
                        prepared.amplitude,
                        resolve_rd_velocity_axis(event.rd_values),
                        frame_mask=prepared.frame_mask,
                    )
                    physical[row, event_index] = physical_result.values
                    physical_valid[row, event_index] = bool(physical_result.valid)
                except (ValueError, FloatingPointError):
                    physical[row, event_index] = 0.0
                    physical_valid[row, event_index] = False
            physical_cached = _cache_row(
                feature_caches,
                cache_positions,
                "rd_physical4",
                event.sample_id,
            )
            if physical_cached is not None:
                cache, cache_index = physical_cached
                cached_values = np.asarray(cache.values[cache_index], dtype=np.float32)
                cached_valid = np.asarray(cache.valid[cache_index], dtype=np.bool_)
                if cached_values.shape != (4,) or cached_valid.ndim != 0:
                    raise ValueError("rd_physical4 cache row shape 无效")
                physical[row, event_index] = cached_values
                physical_valid[row, event_index] = bool(cached_valid)
            stable_cached = _cache_row(
                feature_caches,
                cache_positions,
                "micro_doppler_stable8",
                event.sample_id,
            )
            stable_frame_cached = _cache_row(
                feature_caches,
                cache_positions,
                "micro_doppler_stable8_frame",
                event.sample_id,
            )
            if (stable_cached is None) != (stable_frame_cached is None):
                raise ValueError(
                    "stable8 summary 和 frame cache 必须成对存在："
                    f"{event.sample_id}"
                )
            summary, summary_ok, frame_values, frame_ok = _stable_sidecar(
                event,
                max_frames=max_rd_frames,
                stable_dim=stable_dim,
            )
            sidecar_metadata = (
                event.metadata if isinstance(event.metadata, Mapping) else {}
            )
            sidecar_mode = str(
                sidecar_metadata.get("stable8_sidecar_mode", "normalized")
            )
            if sidecar_mode not in {"normalized", "raw"}:
                raise ValueError(
                    f"不支持的 stable8 sidecar mode：{sidecar_mode}:{event.sample_id}"
                )
            if sidecar_mode == "raw" and task == "l2":
                if stable_median is None or stable_scale is None:
                    raise ValueError(
                        "raw stable8 ASD sidecar 必须提供 normalization 统计量"
                    )
                summary = normalize_stable8(
                    summary,
                    summary_ok,
                    median=stable_median,
                    scale=stable_scale,
                )
                frame_values = normalize_stable8(
                    frame_values,
                    frame_ok,
                    median=stable_median,
                    scale=stable_scale,
                )
            if raw_echo_provider is not None and event.rd_available and (
                stable_cached is None or stable_frame_cached is None
            ):
                from .micro_doppler import extract_current_event_broadband_micro_doppler

                raw_frames = raw_echo_provider.read_event_frames(event)
                micro_event = extract_current_event_broadband_micro_doppler(
                    raw_frames,
                    event_index=int(event.causal_event_index),
                )
                selected_frame_values = micro_event.frame_values[:, stable_indices]
                frame_values = normalize_stable8(
                    selected_frame_values,
                    micro_event.frame_feature_reliable,
                    median=stable_median,
                    scale=stable_scale,
                )
                summary = normalize_stable8(
                    micro_event.values,
                    micro_event.valid,
                    median=stable_median,
                    scale=stable_scale,
                )
                frame_ok = np.asarray(micro_event.frame_feature_reliable, dtype=np.bool_)
                summary_ok = bool(micro_event.valid)
                if frame_values.shape[0] > max_rd_frames:
                    raise ValueError(f"stable8 frame 数超过当前 max_frames：{event.sample_id}")
                if frame_values.shape[0] < max_rd_frames:
                    padded = np.zeros((max_rd_frames, stable_dim), dtype=np.float32)
                    padded[: frame_values.shape[0]] = frame_values
                    frame_values = padded
                    padded_mask = np.zeros((max_rd_frames,), dtype=np.bool_)
                    padded_mask[: frame_ok.shape[0]] = frame_ok
                    frame_ok = padded_mask
            if stable_cached is not None:
                cache, cache_index = stable_cached
                cached_values = np.asarray(cache.values[cache_index], dtype=np.float32)
                cached_valid = np.asarray(cache.valid[cache_index], dtype=np.bool_)
                if cached_values.shape != (stable_dim,) or cached_valid.ndim != 0:
                    raise ValueError("stable8 summary cache row shape 无效")
                summary = cached_values
                summary_ok = bool(cached_valid)
            if stable_frame_cached is not None:
                cache, cache_index = stable_frame_cached
                cached_values = np.asarray(cache.values[cache_index], dtype=np.float32)
                cached_valid = np.asarray(cache.valid[cache_index], dtype=np.bool_)
                if cached_values.shape != (max_rd_frames, stable_dim) or cached_valid.shape != (max_rd_frames,):
                    raise ValueError("stable8 frame cache row shape 无效")
                frame_values = cached_values
                frame_ok = cached_valid
            if stable_input_mode == "raw" and summary_ok:
                if stable_median is None or stable_scale is None:
                    raise ValueError("raw stable8 输入必须提供 normalization 统计量")
                center = np.asarray(stable_median, dtype=np.float32)
                scale = np.asarray(stable_scale, dtype=np.float32)
                summary = np.clip(summary * scale + center, 0.0, 1.0).astype(
                    np.float32, copy=False
                )
                if bool(frame_ok.any()):
                    frame_values = np.clip(
                        frame_values * scale[None, :] + center[None, :],
                        0.0,
                        1.0,
                    ).astype(np.float32, copy=False)
            # L2 的 native micro 分支与 stable8 侧车使用同一份规范化摘要。
            # 这样从缓存读取 sidecar 时也不会静默退化为全零输入。
            micro[row, event_index] = summary
            micro_valid[row, event_index] = summary_ok
            stable_summary[row, event_index] = summary
            stable_summary_valid[row, event_index] = summary_ok
            stable_frame_values[row, event_index] = frame_values
            stable_frame_mask[row, event_index] = frame_ok
        label_name = str(sequence.label_name)
        task_mapping = _TASK_LABEL_TO_ID[task]
        if label_name not in task_mapping:
            if not allow_unmapped_labels:
                raise ValueError(f"{task} 训练序列包含不允许的 label：{label_name}")
            if sequence.label_id is None:
                raise ValueError(
                    f"推理序列缺少 label_id：{sequence.parent_sample_id}"
                )
            labels[row] = -1
            label_names.append(label_name)
            parent_ids.append(str(sequence.parent_sample_id))
            continue
        if sequence.label_id is None:
            raise ValueError(f"训练序列缺少 label_id：{sequence.parent_sample_id}")
        labels[row] = int(task_mapping[label_name])
        label_names.append(label_name)
        parent_ids.append(str(sequence.parent_sample_id))

    result: dict[str, Any] = {
        "task": task,
        "tr_features": torch.from_numpy(tr_values),
        "tr_missing_mask": torch.from_numpy(tr_missing),
        "reference_tr_features": torch.from_numpy(reference_tr_values),
        "reference_tr_missing_mask": torch.from_numpy(reference_tr_missing),
        "tr_dt_seconds": torch.from_numpy(dt_seconds),
        "tr_dt_valid": torch.from_numpy(dt_valid),
        "tr_dt_reason": torch.from_numpy(dt_reason),
        "tr_padding_mask": torch.from_numpy(padding),
        "rd_frames": torch.from_numpy(rd_frames),
        "rd_frame_mask": torch.from_numpy(rd_mask),
        "rd_quality": torch.from_numpy(rd_quality),
        "observation_increment_values": torch.from_numpy(increment),
        "course_features": torch.from_numpy(course),
        "course_valid": torch.from_numpy(course_valid),
        "course_support": torch.from_numpy(course_support),
        "rd_physical_features": torch.from_numpy(physical),
        "rd_physical_valid": torch.from_numpy(physical_valid),
        "micro_doppler_features": torch.from_numpy(micro),
        "micro_doppler_valid": torch.from_numpy(micro_valid),
        "stable_summary_tokens": torch.from_numpy(stable_summary),
        "stable_summary_valid": torch.from_numpy(stable_summary_valid),
        "stable_frame_values": torch.from_numpy(stable_frame_values),
        "stable_frame_mask": torch.from_numpy(stable_frame_mask),
        "motion_values": torch.from_numpy(motion),
        "labels": torch.from_numpy(labels),
        "label_names": label_names,
        "parent_sample_ids": parent_ids,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "original_lengths": torch.tensor(original_lengths, dtype=torch.long),
        "sequence_truncated": torch.tensor(truncated_flags, dtype=torch.bool),
    }
    if baseline_logits is not None:
        result["baseline_logits"] = torch.from_numpy(baseline_logits)
    if task == "l1":
        result.pop("rd_physical_features")
        result.pop("rd_physical_valid")
        result.pop("micro_doppler_features")
        result.pop("micro_doppler_valid")
    return result


__all__ = ["build_model_batch"]
