"""通用索引构建器使用的数据集适配器协议。"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .contracts import EventData, EventRecord, ParentData, ParentRecord


@runtime_checkable
class DataSourceAdapter(Protocol):
    dataset_name: str

    def discover_parents(self) -> list[ParentRecord]:
        ...
    def discover_events(self, parent: ParentRecord) -> list[EventRecord]:
        ...
    def read_parent(self, parent: ParentRecord) -> ParentData:
        ...
    def read_event(self, event: EventRecord) -> EventData:
        ...
    def normalize_label(self, raw_label: str) -> tuple[int, str]:
        ...
    def build_sample_id(self, parent: ParentRecord, event_index: int) -> str:
        ...
    def assign_split(self, parent_ids: list[str]) -> dict[str, str]:
        ...



class ExcludedParentDiscoverer(Protocol):
    """可选的排除 parent 审计能力。"""

    def discover_excluded_parents(self) -> list[ParentRecord]:
        ...

class DataMaterializer(Protocol):
    def materialize_parent(
        self,
        parent: ParentRecord,
        events: list[EventRecord],
        *,
        split_name: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        ...


__all__ = ["DataSourceAdapter", "DataMaterializer", "ExcludedParentDiscoverer"]
