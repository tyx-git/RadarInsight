"""基于统一 manifest 的通用 parent 级数据集。"""

from __future__ import annotations

from dataclasses import replace
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import (
    DataFormatError,
    DataIntegrityError,
    DatasetManifest,
    EventData,
    EventSequence,
)
from .canonical_rd import load_canonical_rd_store
from .manifest import load_manifest, normalise_split_name, resolve_manifest_view
from .paths import resolve_rd_path, resolve_tr_path
from .raw_reader import read_npz, read_text
from .temporal import annotate_event_data


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        read_text(path, encodings=("utf-8-sig", "utf-8")).splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DataFormatError(
                f"TR JSONL 数据索引格式无效：{path}:{line_number}"
            ) from exc
        if not isinstance(value, dict):
            raise DataFormatError(
                f"TR JSONL 数据记录必须是对象：{path}:{line_number}"
            )
        rows.append(value)
    return rows


def _finite_numeric(value: Any, *, name: str) -> None:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fiu" and not np.isfinite(value).all():
            raise DataIntegrityError(f"{name} 包含非有限值")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_numeric(item, name=f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _finite_numeric(item, name=f"{name}[{index}]")
    elif isinstance(value, float) and not np.isfinite(value):
        raise DataIntegrityError(f"{name} 是非有限值")


def _same_identity_value(left: Any, right: Any, field: str) -> bool:
    if field == "label_id":
        return left == right
    return str(left) == str(right)


def _validate_tr_data_row(
    record: dict[str, Any],
    row: dict[str, Any],
    *,
    path: Path,
) -> None:
    """校验 TR 数据行与索引记录的身份字段一致。"""

    fields = (
        "sample_id",
        "parent_sample_id",
        "track_id",
        "causal_event_index",
        "label_id",
        "label_name",
        "split_name",
    )
    missing = [field for field in fields if field not in row]
    if missing:
        raise DataFormatError(
            f"TR 数据行缺少身份字段：{path}: {missing}"
        )
    for field in fields:
        if not _same_identity_value(record[field], row[field], field):
            raise DataIntegrityError(
                f"TR 数据行身份与索引不一致：{path}: "
                f"sample_id={record['sample_id']}, field={field}"
            )


class ParentSequenceDataset:
    """按 parent 返回一个因果有序的 EventSequence。

    本类暴露原始或数据产物观测以及 RD 数组，不执行缩放、角度编码、运动
    计算或模型特征拼接。
    """

    def __init__(
        self,
        project_root: str | Path,
        *,
        manifest_path: str | Path = "src/config/manifests.json",
        split_name: str | None = None,
        parent_ids: Iterable[str] | None = None,
        include_unknown: bool = False,
        training_view: bool = True,
        allowed_labels: Iterable[str] | None = None,
        task_name: str = "",
        allow_unknown_head_training: bool = False,
        view: Any | None = None,
        manifest: DatasetManifest | None = None,
        verify_files: bool = True,
        tensor_cache_size: int = 4,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.manifest = manifest or load_manifest(
            manifest_path, self.project_root, verify_files=verify_files
        )
        self.split_name = (
            None if split_name is None else normalise_split_name(split_name)
        )
        self.include_unknown = bool(include_unknown)
        self.training_view = bool(training_view)
        self._event_ids_by_parent: dict[str, list[str]] = {}
        self._tr_by_id = {
            str(row["sample_id"]): row for row in self.manifest.tr_records
        }
        self._rd_by_id = {
            str(row["sample_id"]): row for row in self.manifest.rd_records
        }
        self._tr_caches: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
        self._rd_caches: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._tensor_cache_size = max(1, int(tensor_cache_size))

        resolved_view = view or resolve_manifest_view(
            self.manifest,
            split_name=self.split_name,
            parent_ids=parent_ids,
            include_unknown=self.include_unknown,
            training_view=self.training_view,
            allowed_labels=allowed_labels,
            task_name=task_name,
            allow_unknown_head_training=allow_unknown_head_training,
        )
        self._canonical_rd = load_canonical_rd_store(self.project_root)
        self.view = (
            replace(
                resolved_view,
                data_binding=f"canonical-rd:{self._canonical_rd.binding}",
            )
            if self._canonical_rd is not None
            else resolved_view
        )
        self._event_ids_by_parent = {
            parent_id: list(sample_ids)
            for parent_id, sample_ids in resolved_view.event_ids_by_parent.items()
        }
        self.parent_ids = list(resolved_view.parent_ids)

    def __len__(self) -> int:
        return len(self.parent_ids)

    def _load_tr_rows(self, data_path: str) -> list[dict[str, Any]]:
        key = str(data_path)
        cached = self._tr_caches.get(key)
        if cached is not None:
            self._tr_caches.move_to_end(key)
            return cached
        path = resolve_tr_path(self.project_root, key, must_exist=True)
        rows = _read_jsonl(path)
        self._tr_caches[key] = rows
        while len(self._tr_caches) > self._tensor_cache_size:
            self._tr_caches.popitem(last=False)
        return rows

    def _load_rd_archive(self, data_path: str) -> dict[str, np.ndarray]:
        key = str(data_path)
        cached = self._rd_caches.get(key)
        if cached is not None:
            self._rd_caches.move_to_end(key)
            return cached
        path = resolve_rd_path(self.project_root, key, must_exist=True)
        payload = read_npz(
            path,
            required_keys=("amplitude", "phase", "time_seconds", "frame_event_indices"),
        )
        amplitude = np.asarray(payload["amplitude"])
        phase = np.asarray(payload["phase"])
        times = np.asarray(payload["time_seconds"])
        event_indices = np.asarray(payload["frame_event_indices"])
        if (
            amplitude.ndim < 3
            or phase.shape != amplitude.shape
            or times.ndim != 1
            or event_indices.ndim != 1
            or times.shape != (amplitude.shape[0],)
            or event_indices.shape != (amplitude.shape[0],)
        ):
            raise DataFormatError(f"RD 数据产物 shape 不一致：{path}")
        for name, array in (
            ("amplitude", amplitude),
            ("phase", phase),
            ("time_seconds", times),
            ("frame_event_indices", event_indices),
        ):
            if array.dtype.kind in "f" and not np.isfinite(array).all():
                raise DataIntegrityError(
                    f"RD 数据数组包含非有限值：{path}:{name}"
                )
        if event_indices.dtype.kind not in "iu":
            raise DataFormatError(f"RD frame_event_indices 必须是整数数组：{path}")
        if times.size > 1 and bool((np.diff(times.astype(np.float64)) < -1.0e-9).any()):
            raise DataIntegrityError(f"RD 局部 frame 时间倒退：{path}")
        result = {
            "amplitude": amplitude,
            "phase": phase,
            "time_seconds": times,
            "frame_event_indices": event_indices,
        }
        self._rd_caches[key] = result
        while len(self._rd_caches) > self._tensor_cache_size:
            self._rd_caches.popitem(last=False)
        return result

    @staticmethod
    def _select_rd_frames(
        record: dict[str, Any],
        archive: dict[str, np.ndarray],
    ) -> np.ndarray:
        """根据记录元数据或事件索引选择一个事件的 RD 帧。"""

        metadata = record.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        declared_indices = metadata.get("frame_indices")
        if declared_indices:
            try:
                selector = np.asarray(
                    [int(value) for value in declared_indices],
                    dtype=np.int64,
                )
            except (TypeError, ValueError) as exc:
                raise DataFormatError(
                    f"RD frame indices are invalid: {record['sample_id']}"
                ) from exc
        else:
            selector = np.flatnonzero(
                archive["frame_event_indices"]
                == int(record["causal_event_index"])
            ).astype(np.int64)
        if selector.size == 0:
            raise DataIntegrityError(
                f"RD record has no frames: {record['sample_id']}"
            )
        if len(set(selector.tolist())) != int(selector.size):
            raise DataIntegrityError(
                f"RD frame indices are duplicated: {record['sample_id']}"
            )
        if int(selector.min()) < 0 or int(selector.max()) >= archive["amplitude"].shape[0]:
            raise DataIntegrityError(
                f"RD frame index is out of bounds: {record['sample_id']}"
            )
        event_indices = archive["frame_event_indices"][selector]
        if not np.all(event_indices == int(record["causal_event_index"])):
            raise DataIntegrityError(
                f"RD 数据事件映射与索引不一致：{record['sample_id']}"
            )
        return selector

    def __getitem__(self, index: int) -> EventSequence:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("dataset index 必须是整数")
        if not 0 <= index < len(self.parent_ids):
            raise IndexError("dataset index 超出范围")
        parent_id = self.parent_ids[index]
        sample_ids = self._event_ids_by_parent[parent_id]
        first_id = sample_ids[0]
        first = self._tr_by_id.get(first_id) or self._rd_by_id[first_id]
        events: list[EventData] = []
        for sample_id in sample_ids:
            tr_record = self._tr_by_id.get(sample_id)
            rd_record = self._rd_by_id.get(sample_id)
            source_record = tr_record or rd_record
            if source_record is None:
                raise DataIntegrityError(
                    f"manifest event has no modality record: {sample_id}"
                )
            tr_values: dict[str, Any] | None = None
            if tr_record is not None and bool(tr_record["available"]):
                tr_path = resolve_tr_path(
                    self.project_root,
                    str(tr_record["data_path"]),
                    must_exist=True,
                )
                tr_rows = self._load_tr_rows(str(tr_record["data_path"]))
                tr_ids = [str(row.get("sample_id", "")) for row in tr_rows]
                if len(tr_ids) != len(set(tr_ids)):
                    raise DataIntegrityError(
                        f"TR 数据产物存在重复 sample_id：{tr_path}"
                    )
                expected_ids = {
                    value
                    for value in sample_ids
                    if self._tr_by_id[value]["data_path"]
                    == tr_record["data_path"]
                    and bool(self._tr_by_id[value]["available"])
                }
                if set(tr_ids) != expected_ids:
                    raise DataIntegrityError(
                        f"TR 数据产物事件覆盖范围与索引不一致：{tr_path}"
                    )
                tr_by_id = {str(row["sample_id"]): row for row in tr_rows}
                tr_row = tr_by_id.get(sample_id)
                if tr_row is None:
                    raise DataIntegrityError(
                        f"TR 数据产物缺少索引事件：{sample_id}"
                    )
                _validate_tr_data_row(tr_record, tr_row, path=tr_path)
                tr_values = tr_row.get("values")
                if not isinstance(tr_values, dict):
                    raise DataFormatError(
                        f"TR values 必须是对象：{sample_id}"
                    )
            rd_values: dict[str, Any] | None = None
            rd_mat_frame_numbers: list[int] = []
            canonical_event = (
                self._canonical_rd.event(
                    parent_id,
                    int(source_record["causal_event_index"]),
                )
                if self._canonical_rd is not None
                else None
            )
            if self._canonical_rd is not None:
                if canonical_event is not None:
                    rd_values = canonical_event.get("rd_values")
                if rd_values is not None:
                    rd_mat_frame_numbers = [
                        int(value)
                        for value in np.asarray(
                            rd_values.get("mat_frame_numbers", []),
                            dtype=np.int64,
                        ).tolist()
                    ]
            elif rd_record is not None and bool(rd_record["available"]):
                rd_path = str(rd_record["data_path"])
                archive = self._load_rd_archive(rd_path)
                selector = self._select_rd_frames(rd_record, archive)
                rd_values = {
                    "amplitude": archive["amplitude"][selector],
                    "phase": archive["phase"][selector],
                    "time_seconds": archive["time_seconds"][selector],
                    "frame_indices": selector,
                    "frame_metadata": list(
                        (rd_record.get("metadata") or {}).get("frame_metadata", [])
                    ),
                    "frame_source_paths": list(
                        (rd_record.get("metadata") or {}).get("frame_source_paths", [])
                    ),
                    "frame_count": int(selector.size),
                    "time_semantics": "parent_relative_cpi_elapsed_seconds",
                    "parent_anomalous": bool(
                        (rd_record.get("metadata") or {}).get("parent_anomalous", False)
                    ),
                    "training_excluded": bool(
                        (rd_record.get("metadata") or {}).get("training_excluded", False)
                    ),
                }
                rd_metadata = rd_record.get("metadata") or {}
                if not isinstance(rd_metadata, dict):
                    raise DataFormatError(f"RD metadata 必须是对象：{sample_id}")
                frame_metadata = rd_metadata.get("frame_metadata") or []
                if not isinstance(frame_metadata, list):
                    raise DataFormatError(f"RD frame_metadata 必须是列表：{sample_id}")
                if frame_metadata:
                    has_original_identity = any(
                        isinstance(item, dict) and item.get("frame_index") is not None
                        for item in frame_metadata
                    )
                    if has_original_identity:
                        for item in frame_metadata:
                            if not isinstance(item, dict) or item.get("frame_index") is None:
                                raise DataIntegrityError(
                                    f"RD frame_metadata 缺少原始 frame_index：{sample_id}"
                                )
                            try:
                                rd_mat_frame_numbers.append(int(item["frame_index"]))
                            except (TypeError, ValueError) as exc:
                                raise DataFormatError(
                                    f"RD 原始 frame_index 无效：{sample_id}"
                                ) from exc
                        if len(rd_mat_frame_numbers) != int(selector.size):
                            raise DataIntegrityError(
                                f"RD 原始 frame_index 数量与选择结果不一致：{sample_id}"
                            )
                expected_count = int(rd_metadata.get("rd_frame_count", selector.size))
                if expected_count != int(selector.size):
                    raise DataIntegrityError(f"RD frame_count 与选择结果不一致：{sample_id}")
            _finite_numeric(tr_values, name=f"{sample_id}.tr_values")
            if rd_values is not None:
                _finite_numeric(rd_values, name=f"{sample_id}.rd_values")
            events.append(
                EventData(
                    sample_id=sample_id,
                    parent_sample_id=parent_id,
                    track_id=str(first["track_id"]),
                    causal_event_index=int(
                        source_record["causal_event_index"]
                    ),
                    label_id=source_record["label_id"],
                    label_name=str(source_record["label_name"]),
                    split_name=str(source_record["split_name"]),
                    tr_values=tr_values,
                    rd_values=rd_values,
                    tr_available=bool(tr_record and tr_record["available"]),
                    rd_available=bool(rd_values is not None),
                    metadata={
                        "tr_record": tr_record,
                        "rd_record": rd_record,
                        "rd_frame_count": int(
                            rd_values.get("frame_count", 0)
                            if self._canonical_rd is not None and rd_values is not None
                            else (rd_record.get("metadata") or {}).get("rd_frame_count", 0)
                        ),
                        "rd_multiple_frames": bool(
                            int(
                                rd_values.get("frame_count", 0)
                                if self._canonical_rd is not None and rd_values is not None
                                else (rd_record.get("metadata") or {}).get("rd_frame_count", 0)
                            ) > 1
                        ),
                        "rd_time_semantics": (
                            "canonical_rd_tensor_time_seconds"
                            if self._canonical_rd is not None
                            else "parent_relative_cpi_elapsed_seconds"
                        ),
                        "rd_mat_frame_numbers": rd_mat_frame_numbers,
                        "canonical_rd": self._canonical_rd is not None,
                        "parent_anomalous": bool(
                            (tr_record.get("metadata") or {}).get(
                                "parent_anomalous", False
                            )
                        ),
                        "training_excluded": bool(
                            (tr_record.get("metadata") or {}).get(
                                "training_excluded", False
                            )
                        ),
                    },
                    tr_dt_is_negative=bool(
                        (tr_record.get("metadata") or {}).get(
                            "tr_dt_is_negative", False
                        )
                    ),
                    tr_time_order_status=str(
                        (tr_record.get("metadata") or {}).get(
                            "tr_time_order_status", "unknown"
                        )
                    ),
                )
            )
        events = annotate_event_data(events)
        return EventSequence(
            parent_sample_id=parent_id,
            track_id=str(first["track_id"]),
            label_id=first["label_id"],
            label_name=str(first["label_name"]),
            split_name=str(first["split_name"]),
            events=events,
            source_name=str(first["source_name"]),
            metadata={"dataset_name": str(first["dataset_name"])},
        )


__all__ = ["ParentSequenceDataset"]
