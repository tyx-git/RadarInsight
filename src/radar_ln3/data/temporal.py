"""通用因果事件时间语义。

本模块只描述观测时间，不执行任何模型状态更新。TR 时间和 RD 局部帧时间
是两条独立时间轴：TR 的 zero-dt 表示来源时间精度不足，不能用 RD CPI
时长替换；RD 帧时间只描述一个事件内部的采集顺序。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import DataFormatError, DataIntegrityError, EventData, EventRecord


TR_TIME_KEY = "time_seconds"
TR_DT_KEY = "dt_seconds"
TR_DT_VALID_KEY = "dt_valid"
TR_DT_REASON_KEY = "dt_reason"
TR_DT_REASON_FIRST = "first_event_no_predecessor"
TR_DT_REASON_ZERO = "zero_delta_at_second_precision"
TR_DT_REASON_POSITIVE = "positive_delta_at_second_precision"
TR_DT_REASON_NEGATIVE = "negative_delta_in_source"
TR_DT_REASON_PADDING = "padding"
TR_DT_REASON_UNAVAILABLE = "unavailable"
TR_DT_EPSILON_SECONDS = 1.0e-9
TR_TIME_SCHEMA_VERSION = "RADAR-LN3-TR-TIME-V1"
RD_FRAME_TIME_SCHEMA_VERSION = "RADAR-LN3-RD-FRAME-TIME-V1"

# Liquid V2 的模型输入编码只接受 active reason；padding 只在 collate 后出现。
DT_REASON_PADDING = 0
DT_REASON_FIRST_EVENT = 1
DT_REASON_ZERO_DELTA = 2
DT_REASON_POSITIVE_DELTA = 3
DT_REASON_TO_ID = {
    TR_DT_REASON_PADDING: DT_REASON_PADDING,
    TR_DT_REASON_FIRST: DT_REASON_FIRST_EVENT,
    TR_DT_REASON_ZERO: DT_REASON_ZERO_DELTA,
    TR_DT_REASON_POSITIVE: DT_REASON_POSITIVE_DELTA,
}


def encode_dt_reason(value: Any, *, dt_valid: bool) -> int:
    """将 manifest 的时间原因编码为模型使用的整数。"""

    reason = str(value)
    if reason not in DT_REASON_TO_ID or reason == TR_DT_REASON_PADDING:
        raise DataFormatError(f"不支持的 active TR dt reason：{reason!r}")
    encoded = DT_REASON_TO_ID[reason]
    if bool(dt_valid) != (encoded == DT_REASON_POSITIVE_DELTA):
        raise DataIntegrityError(
            f"TR dt_reason/dt_valid 不一致：reason={reason!r}, dt_valid={dt_valid}"
        )
    return encoded


def _time_values(values: Sequence[Mapping[str, Any]]) -> list[float] | None:
    present = [TR_TIME_KEY in value and value[TR_TIME_KEY] is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise DataFormatError("同一 parent 的 TR 时间字段不能部分缺失")
    result: list[float] = []
    for index, value in enumerate(values):
        try:
            current = float(value[TR_TIME_KEY])
        except (TypeError, ValueError) as exc:
            raise DataFormatError(f"TR time_seconds 无法解析：index={index}") from exc
        if not np.isfinite(current):
            raise DataIntegrityError(f"TR time_seconds 包含非有限值：index={index}")
        result.append(current)
    return result


def build_temporal_metadata(time_seconds: Sequence[float]) -> list[dict[str, Any]]:
    """按已有因果顺序生成 TR 时间字段，不按时间戳重新排序。"""

    if not time_seconds:
        raise DataFormatError("TR 时间序列不能为空")
    result: list[dict[str, Any]] = []
    previous: float | None = None
    for index, raw_current in enumerate(time_seconds):
        current = float(raw_current)
        if not np.isfinite(current):
            raise DataIntegrityError(f"TR time_seconds 包含非有限值：index={index}")
        if previous is None:
            delta = 0.0
            valid = False
            reason = TR_DT_REASON_FIRST
        else:
            delta = current - previous
            if delta < -TR_DT_EPSILON_SECONDS:
                valid = False
                reason = TR_DT_REASON_NEGATIVE
            elif abs(delta) <= TR_DT_EPSILON_SECONDS:
                delta = 0.0
                valid = False
                reason = TR_DT_REASON_ZERO
            else:
                valid = True
                reason = TR_DT_REASON_POSITIVE
        result.append(
            {
                "schema_version": TR_TIME_SCHEMA_VERSION,
                TR_TIME_KEY: current,
                TR_DT_KEY: delta,
                TR_DT_VALID_KEY: valid,
                TR_DT_REASON_KEY: reason,
                "dt_is_negative": reason == TR_DT_REASON_NEGATIVE,
                "time_order_status": (
                    "non_monotonic"
                    if reason == TR_DT_REASON_NEGATIVE
                    else "monotonic"
                ),
            }
        )
        previous = current
    return result


def validate_temporal_records(records: Sequence[Mapping[str, Any]]) -> None:
    """校验 manifest 中已登记的 TR timing；旧 sidecar 无该字段时跳过。"""

    by_parent: dict[str, list[Mapping[str, Any]]] = {}
    timing_by_parent: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        parent_id = str(record["parent_sample_id"])
        by_parent.setdefault(parent_id, []).append(record)
        metadata = record.get("metadata")
        if not isinstance(metadata, Mapping) or "tr_timing" not in metadata:
            continue
        timing = metadata.get("tr_timing")
        if not isinstance(timing, Mapping):
            raise DataFormatError(f"TR tr_timing 必须是对象：{record.get('sample_id')}")
        required = {
            "schema_version",
            TR_TIME_KEY,
            TR_DT_KEY,
            TR_DT_VALID_KEY,
            TR_DT_REASON_KEY,
            "dt_is_negative",
            "time_order_status",
        }
        if not required.issubset(timing):
            raise DataFormatError(f"TR tr_timing 缺少字段：{record.get('sample_id')}")
        if timing["schema_version"] != TR_TIME_SCHEMA_VERSION:
            raise DataFormatError(f"TR tr_timing schema_version 不匹配：{record.get('sample_id')}")
        timing_by_parent.setdefault(parent_id, []).append(record)
    for parent_id, values in timing_by_parent.items():
        if len(values) != len(by_parent[parent_id]):
            raise DataFormatError(f"TR tr_timing 在 parent 内不完整：{parent_id}")
        ordered = sorted(values, key=lambda item: int(item["causal_event_index"]))
        times = [float(item["metadata"]["tr_timing"][TR_TIME_KEY]) for item in ordered]
        expected = build_temporal_metadata(times)
        for record, actual in zip(ordered, expected, strict=True):
            timing = record["metadata"]["tr_timing"]
            for key in (TR_TIME_KEY, TR_DT_KEY):
                try:
                    observed = float(timing[key])
                except (TypeError, ValueError) as exc:
                    raise DataFormatError(
                        f"TR tr_timing 数值无效：{record.get('sample_id')}:{key}"
                    ) from exc
                if not np.isclose(observed, float(actual[key]), atol=TR_DT_EPSILON_SECONDS):
                    raise DataIntegrityError(
                        f"TR tr_timing 与因果顺序不一致：{parent_id}:{record.get('sample_id')}:{key}"
                    )
            if bool(timing[TR_DT_VALID_KEY]) != bool(actual[TR_DT_VALID_KEY]):
                raise DataIntegrityError(
                    f"TR tr_dt_valid 与时间原因不一致：{record.get('sample_id')}"
                )
            if str(timing[TR_DT_REASON_KEY]) != str(actual[TR_DT_REASON_KEY]):
                raise DataIntegrityError(
                    f"TR tr_dt_reason 与时间顺序不一致：{record.get('sample_id')}"
                )
            if bool(timing["dt_is_negative"]) != bool(actual["dt_is_negative"]):
                raise DataIntegrityError(
                    f"TR dt_is_negative 与时间顺序不一致：{record.get('sample_id')}"
                )
            if str(timing["time_order_status"]) != str(actual["time_order_status"]):
                raise DataIntegrityError(
                    f"TR time_order_status 与时间顺序不一致：{record.get('sample_id')}"
                )


def annotate_event_records(events: Sequence[EventRecord]) -> list[EventRecord]:
    """为带有 ``tr_values.time_seconds`` 的通用事件补充 TR 时间语义。"""

    if not events:
        return []
    tr_values: list[Mapping[str, Any]] = []
    for event in events:
        values = event.metadata.get("tr_values")
        if not isinstance(values, Mapping):
            return list(events)
        tr_values.append(values)
    times = _time_values(tr_values)
    if times is None:
        return list(events)
    timing = build_temporal_metadata(times)
    parent_anomalous = any(
        item[TR_DT_REASON_KEY] == TR_DT_REASON_NEGATIVE for item in timing
    )
    parent_anomaly_reasons = ["negative_tr_dt_in_source"] if parent_anomalous else []
    result: list[EventRecord] = []
    for event, item in zip(events, timing, strict=True):
        metadata = dict(event.metadata)
        values = dict(tr_values[len(result)])
        values.update(item)
        metadata["tr_values"] = values
        metadata["tr_timing"] = dict(item)
        metadata["tr_dt_is_negative"] = bool(item["dt_is_negative"])
        metadata["tr_time_order_status"] = (
            "non_monotonic" if parent_anomalous else "monotonic"
        )
        metadata["parent_anomalous"] = parent_anomalous
        metadata["parent_anomaly_reasons"] = list(parent_anomaly_reasons)
        metadata["training_excluded"] = parent_anomalous
        result.append(replace(event, metadata=metadata))
    return result


def annotate_event_data(events: Sequence[EventData]) -> list[EventData]:
    """为 Dataset 事件补充显式 TR 时间字段，兼容旧 TR 产物。"""

    if not events:
        return []
    values = [event.tr_values for event in events]
    if any(value is None for value in values):
        return list(events)
    mappings = [value for value in values if isinstance(value, Mapping)]
    if len(mappings) != len(values):
        raise DataFormatError("TR values 必须是对象或整体缺失")
    times = _time_values(mappings)
    if times is None:
        return list(events)
    timing = build_temporal_metadata(times)
    result: list[EventData] = []
    for event, item in zip(events, timing, strict=True):
        metadata = dict(event.metadata)
        metadata["tr_timing"] = dict(item)
        result.append(
            replace(
                event,
                metadata=metadata,
                tr_time_seconds=float(item[TR_TIME_KEY]),
                tr_dt_seconds=float(item[TR_DT_KEY]),
                tr_dt_valid=bool(item[TR_DT_VALID_KEY]),
                tr_dt_reason=str(item[TR_DT_REASON_KEY]),
                tr_dt_is_negative=bool(item["dt_is_negative"]),
                tr_time_order_status=str(item["time_order_status"]),
            )
        )
    return result


__all__ = [
    "TR_TIME_KEY",
    "TR_DT_KEY",
    "TR_DT_VALID_KEY",
    "TR_DT_REASON_KEY",
    "TR_DT_REASON_FIRST",
    "TR_DT_REASON_ZERO",
    "TR_DT_REASON_POSITIVE",
    "TR_DT_REASON_NEGATIVE",
    "TR_DT_REASON_PADDING",
    "TR_DT_REASON_UNAVAILABLE",
    "TR_DT_EPSILON_SECONDS",
    "TR_TIME_SCHEMA_VERSION",
    "RD_FRAME_TIME_SCHEMA_VERSION",
    "DT_REASON_PADDING",
    "DT_REASON_FIRST_EVENT",
    "DT_REASON_ZERO_DELTA",
    "DT_REASON_POSITIVE_DELTA",
    "encode_dt_reason",
    "build_temporal_metadata",
    "validate_temporal_records",
    "annotate_event_records",
    "annotate_event_data",
]
