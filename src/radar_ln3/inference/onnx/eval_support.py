"""ONNX 评测脚本共用的数据回放辅助。

只做「adapter 记录 → EventSequence」的组装，不含模型或 ONNX 逻辑，
供 CompareASDAndCQ08ONNX / EvaluateCQ08ONNX 共用。
"""

from __future__ import annotations

from typing import Any


def sequence_from_adapter(adapter: Any, parent: Any) -> Any:
    """把数据适配器的一个 parent 回放成 ``EventSequence``。"""

    from radar_ln3.data.contracts import EventSequence

    records = adapter.discover_events(parent)
    events = [adapter.read_event(record) for record in records]
    return EventSequence(
        parent_sample_id=parent.parent_sample_id,
        track_id=parent.track_id,
        label_id=parent.label_id,
        label_name=parent.label_name,
        split_name=parent.split_name,
        events=events,
        source_name="cq08",
        metadata={"adapter": "CQ08Adapter", "source_path": parent.source_path},
    )


__all__ = ["sequence_from_adapter"]
