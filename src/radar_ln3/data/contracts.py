"""产品数据层与数据集无关的契约。

本模块只描述观测数据和来源信息。数据集专用解析由适配器负责，模型
特征编码留给后续层处理。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


class DataError(Exception):
    """数据契约错误的基类。"""


class DataPathError(DataError):
    """数据路径格式错误或不可用。"""


class PathBoundaryError(DataPathError):
    """路径越过允许的项目根目录或规范化数据根目录。"""


class MissingDataFileError(DataPathError):
    """必需的数据文件不存在。"""


class DataReadError(DataError):
    """原始文件或规范化产物无法安全读取。"""


class DataFormatError(DataError):
    """文件或 manifest 的结构无效。"""


class DataPairingError(DataError):
    """TR/RD 或 parent/event 配对存在歧义或不一致。"""


class DataIntegrityError(DataError):
    """哈希、唯一性、顺序或有限值契约校验失败。"""


class DataSplitError(DataError):
    """parent 级 split 缺失、不完整或跨越边界。"""


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class DatasetRecord:
    """各模态记录共用的来源字段。"""

    dataset_name: str
    source_name: str
    sample_id: str
    parent_sample_id: str
    track_id: str
    causal_event_index: int
    label_id: int | None
    label_name: str
    split_name: str
    available: bool
    source_path: str
    data_path: str
    file_size: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


@dataclass(frozen=True)
class ParentRecord:
    """parent、session 或 track 级分组单元。"""

    dataset_name: str
    source_name: str
    parent_sample_id: str
    track_id: str
    label_id: int | None
    label_name: str
    source_path: str
    data_path: str
    split_name: str = ""
    available: bool = True
    event_count: int | None = None
    file_size: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


@dataclass(frozen=True)
class EventRecord(DatasetRecord):
    """模型特征编码之前的单个因果事件。"""

    modality: str = "TR"


@dataclass(frozen=True)
class ModalityRecord:
    """单个事件模态的可用性和来源信息。"""

    modality: str
    sample_id: str
    parent_sample_id: str
    available: bool
    source_path: str = ""
    data_path: str = ""
    file_size: int | None = None
    sha256: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


@dataclass(frozen=True)
class SplitAssignment:
    parent_sample_id: str
    split_name: str


@dataclass
class ParentData:
    """适配器为一个 parent 输出的、保留原始观测的数据。"""

    parent: ParentRecord
    events: list[EventRecord]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventData:
    """与模态无关的事件值；此处不执行模型特征编码。"""

    sample_id: str
    parent_sample_id: str
    track_id: str
    causal_event_index: int
    label_id: int | None
    label_name: str
    split_name: str
    tr_values: dict[str, Any] | None
    rd_values: dict[str, Any] | None
    tr_available: bool
    rd_available: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    tr_time_seconds: float | None = None
    tr_dt_seconds: float | None = None
    tr_dt_valid: bool | None = None
    tr_dt_reason: str | None = None
    tr_dt_is_negative: bool | None = None
    tr_time_order_status: str | None = None


@dataclass
class EventSequence:
    """属于同一个 parent 的因果有序事件序列。"""

    parent_sample_id: str
    track_id: str
    label_id: int | None
    label_name: str
    split_name: str
    events: list[EventData]
    source_name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RawTargetRecord:
    """raw-IQ 帧中已经转换为 canonical 坐标的目标记录。"""

    target_identifier: int
    event_index: int
    range_bin: int
    doppler_bin: int

    @property
    def raw_target_identifier(self) -> int:
        return int(self.target_identifier)

    @property
    def track_point_index(self) -> int:
        return int(self.event_index)

    @property
    def range_bin_raw(self) -> int:
        return int(self.range_bin)

    @property
    def doppler_bin_raw(self) -> int:
        return int(self.doppler_bin)


@dataclass(frozen=True)
class RawEchoFrame:
    """一个 raw-IQ frame 及其采样和目标定位元数据。"""

    raw_frame_index: int
    frequency_hz: float
    cpi_count: int
    prt_num: int
    prt_seconds: float
    target_records: tuple[RawTargetRecord, ...]
    iq: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        iq = np.asarray(self.iq)
        if iq.ndim != 2 or not np.iscomplexobj(iq):
            raise DataFormatError("RawEchoFrame iq 必须是二维复数数组")
        if not np.isfinite(iq.real).all() or not np.isfinite(iq.imag).all():
            raise DataIntegrityError("RawEchoFrame iq 不能包含非有限值")
        # CQ08 原始文件中 CPIcount=0 是已审计的合法来源值，表示该帧
        # 不能提供正的 CPI 观测时长。时长计算由数据适配器回退到
        # prt_num * prt_seconds；这里不能把它误判为格式错误。
        if int(self.raw_frame_index) < 0 or int(self.cpi_count) < 0:
            raise DataFormatError("RawEchoFrame frame index/cpi_count 无效")
        for name, value in (
            ("frequency_hz", self.frequency_hz),
            ("prt_seconds", self.prt_seconds),
        ):
            if not np.isfinite(float(value)) or float(value) <= 0.0:
                raise DataFormatError(f"RawEchoFrame {name} 必须为正有限值")
        if int(self.prt_num) <= 0:
            raise DataFormatError("RawEchoFrame prt_num 必须为正整数")
        if any(not isinstance(record, RawTargetRecord) for record in self.target_records):
            raise DataFormatError("RawEchoFrame target_records 类型无效")


class RawEchoProvider(Protocol):
    """按事件读取当前事件关联 raw-IQ frame 的数据源协议。"""

    def read_event_frames(self, event: Any) -> Sequence[RawEchoFrame]:
        ...


@dataclass(frozen=True)
class DatasetView:
    """统一的 split/task 数据视图。

    视图由事件主记录的 split、label 和异常标记决定；TR、RD 以及后续
    feature cache 只能消费该视图，不能各自重新解释 training view。
    """

    view_name: str
    split_name: str | None
    include_unknown: bool
    training_view: bool
    allowed_labels: tuple[str, ...] | None
    sample_ids: tuple[str, ...]
    parent_ids: tuple[str, ...]
    event_ids_by_parent: dict[str, tuple[str, ...]]
    task_name: str = ""
    data_binding: str | None = None

    @property
    def view_hash(self) -> str:
        """返回包含顺序和筛选语义的稳定 view hash。"""

        payload = {
            "view_name": self.view_name,
            "task_name": self.task_name,
            "split_name": self.split_name,
            "include_unknown": self.include_unknown,
            "training_view": self.training_view,
            "allowed_labels": list(self.allowed_labels)
            if self.allowed_labels is not None
            else None,
            "parent_ids": list(self.parent_ids),
            "sample_ids": list(self.sample_ids),
            "event_ids_by_parent": {
                key: list(value)
                for key, value in self.event_ids_by_parent.items()
            },
        }
        if self.data_binding is not None:
            payload["data_binding"] = self.data_binding
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class DatasetManifest:
    """单一产品 manifest 经过校验后的内存表示。"""

    schema_version: str
    dataset: str
    data_root: str
    source_policy: dict[str, Any]
    split: dict[str, Any]
    tr_records: list[dict[str, Any]]
    rd_records: list[dict[str, Any]]
    tr_index: dict[str, Any] = field(default_factory=dict)
    rd_index: dict[str, Any] = field(default_factory=dict)
    pairing: dict[str, Any] = field(default_factory=dict)
    audit: dict[str, Any] = field(default_factory=dict)
    registries: dict[str, Any] = field(default_factory=dict)
    manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "data_root": self.data_root,
            "source_policy": json_value(self.source_policy),
            "split": json_value(self.split),
            "TR": json_value(self.tr_index),
            "RD": json_value(self.rd_index),
            "pairing": json_value(self.pairing),
            "audit": json_value(self.audit),
            **(json_value(self.registries) if self.registries else {}),
            **({"manifest_sha256": self.manifest_sha256} if self.manifest_sha256 else {}),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


def record_to_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if isinstance(record, Mapping):
        return json_value(dict(record))
    raise TypeError(f"不支持的 record 类型：{type(record).__name__}")


__all__ = [
    "DataError",
    "DataPathError",
    "PathBoundaryError",
    "MissingDataFileError",
    "DataReadError",
    "DataFormatError",
    "DataPairingError",
    "DataIntegrityError",
    "DataSplitError",
    "DatasetRecord",
    "ParentRecord",
    "EventRecord",
    "ModalityRecord",
    "SplitAssignment",
    "ParentData",
    "EventData",
    "EventSequence",
    "RawEchoFrame",
    "RawEchoProvider",
    "RawTargetRecord",
    "DatasetView",
    "DatasetManifest",
    "record_to_dict",
    "json_value",
]
