"""变长 parent 序列的纯结构批处理。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contracts import DataIntegrityError, EventSequence
from .temporal import TR_DT_REASON_NEGATIVE


def _event_frame_count(values: dict[str, Any]) -> int:
    frames = values.get("frames")
    if isinstance(frames, list):
        return len(frames)
    if values.get("frame_count") is not None:
        return int(values["frame_count"])
    return int(np.asarray(values.get("amplitude", [])).shape[0])


def _event_frame_times(values: dict[str, Any]) -> np.ndarray:
    frames = values.get("frames")
    if isinstance(frames, list):
        return np.asarray(
            [float(frame.get("time_seconds", 0.0)) for frame in frames],
            dtype=np.float32,
        )
    raw_times = values.get("time_seconds")
    if raw_times is None:
        return np.zeros((_event_frame_count(values),), dtype=np.float32)
    return np.asarray(raw_times, dtype=np.float32)


def _event_frame_source_paths(values: dict[str, Any]) -> list[str]:
    frames = values.get("frames")
    if isinstance(frames, list):
        return [str(frame.get("source_path", "")) for frame in frames]
    return [str(value) for value in values.get("frame_source_paths", [])]


def _check_sequence(item: EventSequence) -> None:
    if not item.events:
        raise DataIntegrityError(
            f"parent sequence 没有 event：{item.parent_sample_id}"
        )
    indices = [int(event.causal_event_index) for event in item.events]
    if indices != list(range(len(item.events))):
        raise DataIntegrityError(
            f"parent sequence 的 event index 不连续：{item.parent_sample_id}"
        )
    labels = {event.label_name for event in item.events}
    if len(labels) != 1:
        raise DataIntegrityError(
            f"parent sequence 包含多个 label：{item.parent_sample_id}"
        )
    label_ids = {event.label_id for event in item.events}
    if len(label_ids) != 1 or next(iter(label_ids)) != item.label_id:
        raise DataIntegrityError(
            f"parent sequence 的 label ID 不一致：{item.parent_sample_id}"
        )
    splits = {event.split_name for event in item.events}
    if len(splits) != 1 or next(iter(splits)) != item.split_name:
        raise DataIntegrityError(
            f"parent sequence 的 split 不一致：{item.parent_sample_id}"
        )
    for event in item.events:
        if event.parent_sample_id != item.parent_sample_id:
            raise DataIntegrityError(
                f"event 的 parent 不一致：{event.sample_id}"
            )
        if event.track_id != item.track_id:
            raise DataIntegrityError(
                f"event 的 track 不一致：{event.sample_id}"
            )
        if event.tr_available != (event.tr_values is not None):
            raise DataIntegrityError(
                f"TR 可用性与 values 不一致：{event.sample_id}"
            )
        if event.rd_available != (event.rd_values is not None):
            raise DataIntegrityError(
                f"RD 可用性与 values 不一致：{event.sample_id}"
            )
        if event.tr_dt_valid is not None and not isinstance(event.tr_dt_valid, bool):
            raise DataIntegrityError(f"TR dt_valid 必须是布尔值：{event.sample_id}")
        if event.tr_dt_seconds is not None and (
            not np.isfinite(float(event.tr_dt_seconds))
            or (
                float(event.tr_dt_seconds) < 0.0
                and event.tr_dt_reason != TR_DT_REASON_NEGATIVE
            )
        ):
            raise DataIntegrityError(f"TR dt_seconds 无效：{event.sample_id}")
        if event.rd_values is not None:
            frames = event.rd_values.get("frames")
            if isinstance(frames, list):
                if not frames:
                    raise DataIntegrityError(f"RD frames 不能为空：{event.sample_id}")
                for frame in frames:
                    if not isinstance(frame, dict):
                        raise DataIntegrityError(f"RD frame 记录无效：{event.sample_id}")
                    frame_amplitude = frame.get("amplitude")
                    frame_phase = frame.get("phase")
                    if not isinstance(frame_amplitude, np.ndarray) or frame_amplitude.ndim != 2:
                        raise DataIntegrityError(f"RD frame amplitude shape 无效：{event.sample_id}")
                    if not isinstance(frame_phase, np.ndarray) or frame_phase.shape != frame_amplitude.shape:
                        raise DataIntegrityError(f"RD frame phase shape 无效：{event.sample_id}")
            else:
                amplitude = event.rd_values.get("amplitude")
                phase = event.rd_values.get("phase")
                times = event.rd_values.get("time_seconds")
                if not isinstance(amplitude, np.ndarray) or amplitude.ndim < 3:
                    raise DataIntegrityError(f"RD frame shape 不一致：{event.sample_id}")
                if phase is not None:
                    if not isinstance(phase, np.ndarray) or phase.shape != amplitude.shape:
                        raise DataIntegrityError(f"RD frame phase shape 不一致：{event.sample_id}")
                if times is not None and (
                    not isinstance(times, np.ndarray) or times.shape != (amplitude.shape[0],)
                ):
                    raise DataIntegrityError(f"RD frame time shape 不一致：{event.sample_id}")
        for name, values in (
            ("tr_values", event.tr_values),
            ("rd_values", event.rd_values),
        ):
            if isinstance(values, dict):
                _check_finite(values, name=f"{event.sample_id}.{name}")


def _check_finite(value: Any, *, name: str) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fiu" and not np.isfinite(value).all():
            raise DataIntegrityError(f"{name} 包含非有限值")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _check_finite(item, name=f"{name}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_finite(item, name=f"{name}[{index}]")
        return
    if isinstance(value, float) and not np.isfinite(value):
        raise DataIntegrityError(f"{name} 是非有限值")


def collate_parent_sequences(items: list[EventSequence]) -> dict[str, Any]:
    """批处理原始 parent 序列，不执行特征变换。"""

    if not items:
        raise ValueError("不能批处理空的 parent batch")
    for item in items:
        _check_sequence(item)
    batch_size = len(items)
    max_length = max(len(item.events) for item in items)
    event_mask = np.zeros((batch_size, max_length), dtype=np.bool_)
    tr_available = np.zeros((batch_size, max_length), dtype=np.bool_)
    rd_available = np.zeros((batch_size, max_length), dtype=np.bool_)
    label_ids = np.full((batch_size,), -1, dtype=np.int64)
    events: list[list[Any | None]] = []
    parent_ids: list[str] = []
    track_ids: list[str] = []
    label_names: list[str] = []
    split_names: list[str] = []
    tr_time_seconds = np.zeros((batch_size, max_length), dtype=np.float32)
    tr_dt_seconds = np.zeros((batch_size, max_length), dtype=np.float32)
    tr_dt_valid = np.zeros((batch_size, max_length), dtype=np.bool_)
    tr_dt_is_negative = np.zeros((batch_size, max_length), dtype=np.bool_)
    tr_dt_reason: list[list[str]] = []
    tr_time_order_status: list[list[str]] = []
    max_rd_frames = max(
        max(
            int(
                (event.rd_values or {}).get(
                    "frame_count",
                    _event_frame_count(event.rd_values or {}),
                )
            )
            if event.rd_values is not None
            else 0
            for event in item.events
        )
        for item in items
    )
    rd_frame_mask = np.zeros((batch_size, max_length, max_rd_frames), dtype=np.bool_)
    rd_frame_times = np.zeros((batch_size, max_length, max_rd_frames), dtype=np.float32)
    rd_frame_source_paths: list[list[list[str]]] = []
    for row, item in enumerate(items):
        length = len(item.events)
        event_mask[row, :length] = True
        tr_available[row, :length] = [
            event.tr_available for event in item.events
        ]
        rd_available[row, :length] = [
            event.rd_available for event in item.events
        ]
        if item.label_id is not None:
            label_ids[row] = int(item.label_id)
        events.append([*item.events, *([None] * (max_length - length))])
        parent_ids.append(item.parent_sample_id)
        track_ids.append(item.track_id)
        label_names.append(item.label_name)
        split_names.append(item.split_name)
        reasons: list[str] = []
        source_paths_by_event: list[list[str]] = []
        for event_index, event in enumerate(item.events):
            tr_time_seconds[row, event_index] = float(event.tr_time_seconds or 0.0)
            tr_dt_seconds[row, event_index] = float(event.tr_dt_seconds or 0.0)
            tr_dt_valid[row, event_index] = bool(event.tr_dt_valid)
            tr_dt_is_negative[row, event_index] = bool(event.tr_dt_is_negative)
            reasons.append(event.tr_dt_reason or "unavailable")
            rd_values = event.rd_values
            if rd_values is None:
                source_paths_by_event.append([])
                continue
            times = _event_frame_times(rd_values)
            frame_count = min(int(times.shape[0]), max_rd_frames)
            rd_frame_mask[row, event_index, :frame_count] = True
            rd_frame_times[row, event_index, :frame_count] = times[:frame_count]
            source_paths_by_event.append(
                _event_frame_source_paths(rd_values)[:frame_count]
            )
        reasons.extend(["padding"] * (max_length - length))
        tr_dt_reason.append(reasons)
        tr_time_order_status.append(
            [event.tr_time_order_status or "unknown" for event in item.events]
            + ["padding"] * (max_length - length)
        )
        source_paths_by_event.extend([[] for _ in range(max_length - length)])
        rd_frame_source_paths.append(source_paths_by_event)
    return {
        "events": events,
        "event_mask": event_mask,
        "padding_mask": ~event_mask,
        "modality_available": {
            "TR": tr_available,
            "RD": rd_available,
        },
        "tr_available": tr_available,
        "rd_available": rd_available,
        "label_ids": label_ids,
        "parent_sample_ids": parent_ids,
        "track_ids": track_ids,
        "label_names": label_names,
        "split_names": split_names,
        "lengths": event_mask.sum(axis=1, dtype=np.int64),
        "tr_time_seconds": tr_time_seconds,
        "tr_dt_seconds": tr_dt_seconds,
        "tr_dt_valid": tr_dt_valid,
        "tr_dt_is_negative": tr_dt_is_negative,
        "tr_dt_reason": tr_dt_reason,
        "tr_time_order_status": tr_time_order_status,
        "rd_frame_mask": rd_frame_mask,
        "rd_frame_times": rd_frame_times,
        "rd_frame_source_paths": rd_frame_source_paths,
    }


__all__ = ["collate_parent_sequences"]
