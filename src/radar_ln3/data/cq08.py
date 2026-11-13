"""唯一的 CQ08 专用数据源适配器。

只有本模块了解 CQ08 目录名称、文件名模式、中文原始列名、MAT 字段名和
标签编码。本模块输出通用契约，不计算模型特征。
"""

from __future__ import annotations

import io
import json
import math
import re
import struct
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .atomic import write_bytes
from .contracts import (
    DataFormatError,
    DataPairingError,
    DataReadError,
    DataSplitError,
    DatasetManifest,
    EventData,
    EventRecord,
    ParentData,
    ParentRecord,
    PathBoundaryError,
)
from .canonical_rd import load_canonical_rd_store
from .paths import (
    relative_project_path,
    resolve_project_path,
    resolve_source_path,
    rd_root,
    tr_root,
)
from .source import load_source_config, resolve_registered_source_path
from .manifest import load_manifest, normalise_split_name
from .raw_reader import (
    finite_array,
    read_delimited,
    read_bytes,
    read_json,
    read_mat,
    read_npz,
    read_text,
    sha256_file,
)
from .rd_sanitizer import (
    RD_AMPLITUDE_REPAIR_POLICY,
    RD_AMPLITUDE_REPAIR_SCHEMA,
    sanitize_rd_amplitude_db,
)
from .temporal import (
    RD_FRAME_TIME_SCHEMA_VERSION,
    annotate_event_records,
)


CQ08_DATASET_NAME = "CQ08"
CQ08_SOURCE_NAME = "CQ08"
CQ08_TR_SCHEMA_VERSION = "RADAR-LN3-CQ08-TR-DATA-V3"
CQ08_RD_SCHEMA_VERSION = "RADAR-LN3-CQ08-RD-DATA-V3"
RD_OUTPUT_SHAPE = (21, 384)

TRACK_PATTERN = re.compile(
    r"^Tracks_(?P<track_id>\d+)_(?P<label_id>\d+)_(?P<track_length>\d+)\.txt$",
    re.IGNORECASE,
)
POINT_PATTERN = re.compile(
    r"^PointTracks_(?P<track_id>\d+)_(?P<label_id>\d+)_(?P<track_length>\d+)\.txt$",
    re.IGNORECASE,
)
RAW_ECHO_PATTERN = re.compile(
    r"^(?P<track_id>\d+)_Label_(?P<label_id>\d+)\.dat$",
    re.IGNORECASE,
)
MAT_PATTERN = re.compile(
    r"^(?P<capture_id>\d+)_DAUR_RD_(?P<label_text>[^_]+)_(?P<frame_index>\d+)_(?P<track_id>\d+)\.mat$",
    re.IGNORECASE,
)

TRACK_COLUMNS = (
    "点时间",
    "批号",
    "滤波距离",
    "滤波方位",
    "滤波俯仰",
    "全速度",
    "X向速度",
    "Y向速度",
    "Z向速度",
    "航向",
)
POINT_COLUMNS = (
    "点时间",
    "批号",
    "距离",
    "方位",
    "俯仰",
    "多普勒速度",
    "和幅度",
    "信噪比",
    "原始点数量",
)

CQ08_LABEL_ID_TO_NAME = {
    1: "DroneTarget",
    2: "DroneTarget",
    3: "BirdTarget",
    4: "BalloonTarget",
    5: "ClutterTarget",
    6: "UnknownTarget",
}
CQ08_LABEL_NAME_TO_ID = {
    "DroneTarget": 0,
    "BirdTarget": 1,
    "BalloonTarget": 2,
    "ClutterTarget": 3,
    "UnknownTarget": 4,
}
CQ08_TEXT_LABELS = {
    "drone": "DroneTarget",
    "rotary drone": "DroneTarget",
    "rotary_drone": "DroneTarget",
    "fixed-wing uav": "DroneTarget",
    "fixed wing uav": "DroneTarget",
    "uav": "DroneTarget",
    "bird": "BirdTarget",
    "balloon": "BalloonTarget",
    "clutter": "ClutterTarget",
    "unknown": "UnknownTarget",
    "unknown target": "UnknownTarget",
    "other": "UnknownTarget",
    "other target": "UnknownTarget",
    "othertarget": "UnknownTarget",
}


def _as_float(value: Any) -> float | None:
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        result = float(text)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


def _snr_db(value: Any) -> float | None:
    """将 CQ08 点迹的线性信噪比转换为 V0/V16 使用的 dB。"""

    linear = _as_float(value)
    if linear is None or linear <= 0.0:
        return None
    result = 10.0 * math.log10(linear)
    return result if math.isfinite(result) else None


def _parse_time_seconds(value: Any) -> float:
    text = str(value).strip()
    parts = text.split(":")
    if len(parts) != 3:
        raise DataFormatError(f"unexpected CQ08 time format: {text!r}")
    hour, minute, second = parts
    try:
        result = int(hour) * 3600.0 + int(minute) * 60.0 + float(second)
    except ValueError as exc:
        raise DataFormatError(f"unexpected CQ08 time format: {text!r}") from exc
    if not np.isfinite(result):
        raise DataFormatError(f"non-finite CQ08 time: {text!r}")
    return result


def _rd_duration_seconds(debug: Any) -> tuple[float, str]:
    """读取 RD 局部时长，并在 CPI 计数无效时使用 PRT 回退。"""

    cpi_count = _as_float(getattr(debug, "cpi_count", None))
    prt_num = _as_float(getattr(debug, "prt_num", None))
    prt_seconds = _as_float(getattr(debug, "prt_s", None))
    if prt_num is None or prt_seconds is None or prt_num <= 0 or prt_seconds <= 0:
        return 0.0, "unavailable"
    if cpi_count is not None and cpi_count > 0:
        return float(cpi_count * prt_num * prt_seconds), "cpi"
    return float(prt_num * prt_seconds), "prt_fallback"


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("|", "__")
        .replace(":", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def _normalise_label(raw_label: str) -> tuple[int, str]:
    text = str(raw_label).strip()
    if not text:
        raise DataFormatError("CQ08 label is empty")
    if text in CQ08_LABEL_NAME_TO_ID:
        return CQ08_LABEL_NAME_TO_ID[text], text
    try:
        numeric = int(text)
    except ValueError:
        numeric = None
    if numeric is not None and numeric in CQ08_LABEL_ID_TO_NAME:
        name = CQ08_LABEL_ID_TO_NAME[numeric]
    else:
        name = CQ08_TEXT_LABELS.get(text.casefold())
        if name is None:
            raise DataFormatError(f"unsupported CQ08 label: {raw_label!r}")
    return CQ08_LABEL_NAME_TO_ID[name], name


def _strict_positive_integer(value: Any, *, name: str, path: Path) -> int:
    """读取必须为正整数的 CQ08 标识，拒绝隐式截断。"""

    number = _as_float(value)
    if number is None or number <= 0.0 or not float(number).is_integer():
        raise DataFormatError(f"CQ08 {name} is not a positive integer: {path}")
    return int(number)


def _validate_source_rows(
    rows: list[dict[str, str]],
    *,
    columns: tuple[str, ...],
    path: Path,
    length_hint: int,
    source_kind: str,
    expected_parent_id: int,
) -> None:
    """验证轨迹/点迹的列结构、文件长度和 parent 身份。"""

    if length_hint <= 0:
        raise DataFormatError(f"CQ08 {source_kind} length hint is not positive: {path}")
    if len(rows) != length_hint:
        raise DataPairingError(
            f"CQ08 {source_kind} row count differs from filename hint: "
            f"{path} ({len(rows)} != {length_hint})"
        )
    if not rows:
        raise DataFormatError(f"CQ08 {source_kind} file has no data rows: {path}")
    expected_columns = set(columns)
    parent_indices: set[int] = set()
    for row_number, row in enumerate(rows, start=1):
        if set(row) != expected_columns:
            raise DataFormatError(
                f"CQ08 {source_kind} columns differ at row {row_number}: {path}"
            )
        actual_index = _strict_positive_integer(
            row.get("批号"),
            name=f"{source_kind} parent ID",
            path=path,
        )
        parent_indices.add(actual_index)
        _parse_time_seconds(row.get("点时间"))
    if len(parent_indices) != 1:
        raise DataPairingError(
            f"CQ08 {source_kind} file contains multiple parent IDs: {path}"
        )
    actual_parent_id = next(iter(parent_indices))
    if actual_parent_id != int(expected_parent_id):
        raise DataPairingError(
            f"CQ08 {source_kind} parent ID differs from filename track ID: "
            f"{path} ({actual_parent_id} != {expected_parent_id})"
        )


def resolve_frame_source_path(
    project_root: Path,
    raw_path: str,
    *,
    source_root: Path,
    source_config: Mapping[str, Any],
) -> Path:
    """把 TR/RD 记录中的帧路径解析为已存在的 MAT 文件。"""

    if str(raw_path).replace("\\", "/").startswith("external/"):
        return resolve_source_path(
            project_root,
            raw_path,
            source_roots={"cq08": source_root},
            must_exist=True,
            allow_external=True,
        )
    return resolve_registered_source_path(
        project_root,
        raw_path,
        source_config,
        must_exist=True,
    )


def _deterministic_npz_bytes(values: Mapping[str, Any]) -> bytes:
    """以固定 ZIP 元数据写出 NPZ，避免重复运行产生伪差异。"""

    output = io.BytesIO()
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(values):
            array_buffer = io.BytesIO()
            np.lib.format.write_array(
                array_buffer,
                np.asarray(values[name]),
                allow_pickle=False,
            )
            info = zipfile.ZipInfo(
                filename=f"{name}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 0
            info.external_attr = 0o600 << 16
            archive.writestr(info, array_buffer.getvalue())
    return output.getvalue()


class CQ08Adapter:
    """适配已核实的 CQ08 航迹、点迹和 MAT 原始目录布局。"""

    dataset_name = CQ08_DATASET_NAME

    def __init__(
        self,
        project_root: str | Path,
        *,
        source_root: str | Path | None = None,
        split_path: str | Path | None = None,
        parent_limit: int | None = None,
        parent_split: str | None = None,
        registered_index: bool = False,
        manifest_path: str | Path = "src/config/manifests.json",
        verify_files: bool = True,
        manifest: DatasetManifest | None = None,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.source_config = load_source_config(self.project_root)
        self._canonical_rd = load_canonical_rd_store(self.project_root)
        self.registered_index = bool(registered_index)
        self.manifest_path = manifest_path
        self.verify_files = bool(verify_files)
        self._provided_manifest = manifest
        self.parent_split = (
            None
            if parent_split is None or str(parent_split).lower() == "all"
            else normalise_split_name(str(parent_split))
        )
        if source_root is None:
            configured_source_root = Path(
                self.source_config["source_roots"]["cq08"]["path"]
            )
            self.source_root = (
                configured_source_root.resolve()
                if configured_source_root.is_absolute()
                else (self.project_root / configured_source_root).resolve()
            )
        else:
            configured_source_root = Path(source_root)
            self.source_root = (
                configured_source_root.resolve()
                if configured_source_root.is_absolute()
                else (self.project_root / configured_source_root).resolve()
            )
        if not self.registered_index and not self.source_root.is_dir():
            raise DataReadError(f"CQ08 source root does not exist: {self.source_root}")
        try:
            self.source_root.relative_to(self.project_root)
        except ValueError:
            configured = self.source_config["source_roots"]["cq08"]
            if not bool(configured.get("allow_external", False)):
                raise PathBoundaryError(
                    "CQ08 source_root 位于项目外，但 source.json 未允许 external path"
                )
        else:
            relative_project_path(self.project_root, self.source_root)
        self.track_root = self.source_root / "航迹"
        self.point_root = self.source_root / "点迹"
        self.raw_root = self.source_root / "原始回波"
        self.mat_root = self.source_root / "MAT"
        if not self.registered_index:
            for path in (self.track_root, self.point_root, self.raw_root, self.mat_root):
                if not path.is_dir():
                    raise DataReadError(f"CQ08 source directory is missing: {path}")
        if parent_limit is None:
            self.parent_limit = None
        else:
            self.parent_limit = int(parent_limit)
            if self.parent_limit <= 0:
                raise ValueError("CQ08 parent_limit must be positive")
        self.split_path = self._select_split_path(split_path)
        self._track_files: dict[tuple[int, int], dict[str, Any]] = {}
        self._point_files: dict[tuple[int, int], dict[str, Any]] = {}
        self._raw_files: dict[tuple[int, int], dict[str, Any]] = {}
        self._mat_files: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self._track_rows: dict[tuple[int, int], list[dict[str, str]]] = {}
        self._point_rows: dict[tuple[int, int], list[dict[str, str]]] = {}
        self._rd_metadata: dict[int, list[dict[str, Any]]] = {}
        self._raw_echo_provider: CQ08RawEchoProvider | None = None
        self._split_excluded_parent_ids: set[str] = set()
        self._split_map = self._load_split_map(self.split_path)
        self._registered_manifest = None
        self._registered_parents: list[ParentRecord] = []
        self._registered_events: dict[str, list[EventRecord]] = {}
        self._registered_tr_rows: dict[str, dict[str, dict[str, Any]]] = {}
        if self.registered_index:
            self._load_registered_index()
        else:
            self._discover_source_files()

    def _load_registered_index(self) -> None:
        """只读取已生成的产品索引和产物，不扫描原始 MAT 目录。"""

        manifest = self._provided_manifest or load_manifest(
            self.manifest_path,
            self.project_root,
            verify_files=self.verify_files,
            allow_unmaterialized=False,
        )
        if manifest.dataset != CQ08_DATASET_NAME:
            raise DataFormatError(
                f"registered manifest dataset 不匹配：{manifest.dataset}"
            )
        self._registered_manifest = manifest
        tr_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rd_by_sample = {
            str(record["sample_id"]): record for record in manifest.rd_records
        }
        for record in manifest.tr_records:
            tr_by_parent[str(record["parent_sample_id"])].append(record)
        for parent_id in sorted(tr_by_parent):
            records = sorted(
                tr_by_parent[parent_id],
                key=lambda value: (
                    int(value["causal_event_index"]),
                    str(value["sample_id"]),
                ),
            )
            first = records[0]
            parent = ParentRecord(
                dataset_name=str(first["dataset_name"]),
                source_name=str(first["source_name"]),
                parent_sample_id=parent_id,
                track_id=str(first["track_id"]),
                label_id=first["label_id"],
                label_name=str(first["label_name"]),
                source_path=str(first["source_path"]),
                data_path=str(first["data_path"]),
                split_name=normalise_split_name(first["split_name"]),
                available=all(bool(value["available"]) for value in records),
                event_count=len(records),
                file_size=first.get("file_size"),
                sha256=first.get("sha256"),
                metadata={
                    "registered_index": True,
                    "raw_label_id": first["label_id"],
                },
            )
            self._registered_parents.append(parent)
            events: list[EventRecord] = []
            for tr_record in records:
                sample_id = str(tr_record["sample_id"])
                rd_record = rd_by_sample.get(sample_id)
                tr_row = self._registered_tr_row(tr_record, sample_id)
                tr_metadata = dict(tr_record.get("metadata") or {})
                rd_metadata = (
                    dict(rd_record.get("metadata") or {})
                    if rd_record is not None
                    else {}
                )
                frame_metadata = list(rd_metadata.get("frame_metadata") or [])
                frame_paths = list(rd_metadata.get("frame_source_paths") or [])
                if not frame_paths and rd_record is not None:
                    frame_paths = [
                        value
                        for value in str(rd_record.get("source_path") or "").split(";")
                        if value
                    ]
                event_metadata = {
                    "registered_index": True,
                    "tr_record": tr_record,
                    "track_point_index": tr_metadata.get("track_point_index"),
                    "track_time_raw": tr_metadata.get("track_time_raw"),
                    "point_available": tr_metadata.get("point_available"),
                    "tr_values": dict(tr_row.get("values") or {}),
                    "tr_timing": dict(tr_metadata.get("tr_timing") or {}),
                    "rd_frame_paths": frame_paths,
                    "rd_frame_metadata": frame_metadata,
                    "rd_mat_frame_numbers": [
                        int(item["frame_index"])
                        for item in frame_metadata
                        if isinstance(item, Mapping)
                        and item.get("frame_index") is not None
                    ],
                    "rd_available": bool(rd_record and rd_record.get("available")),
                    "rd_frame_count": int(rd_metadata.get("rd_frame_count", 0)),
                    "rd_multiple_frames": int(rd_metadata.get("rd_frame_count", 0)) > 1,
                    "rd_time_semantics": "parent_relative_cpi_elapsed_seconds",
                    "rd_time_schema_version": RD_FRAME_TIME_SCHEMA_VERSION,
                    "rd_record": rd_record,
                }
                events.append(
                    EventRecord(
                        dataset_name=str(tr_record["dataset_name"]),
                        source_name=str(tr_record["source_name"]),
                        sample_id=sample_id,
                        parent_sample_id=parent_id,
                        track_id=str(tr_record["track_id"]),
                        causal_event_index=int(tr_record["causal_event_index"]),
                        label_id=tr_record["label_id"],
                        label_name=str(tr_record["label_name"]),
                        split_name=normalise_split_name(tr_record["split_name"]),
                        available=bool(tr_record["available"]),
                        source_path=str(tr_record["source_path"]),
                        data_path=str(tr_record["data_path"]),
                        file_size=tr_record.get("file_size"),
                        sha256=tr_record.get("sha256"),
                        metadata=event_metadata,
                    )
                )
            self._registered_events[parent_id] = annotate_event_records(events)

    def _registered_tr_row(
        self,
        record: Mapping[str, Any],
        sample_id: str,
    ) -> dict[str, Any]:
        data_path = str(record.get("data_path") or "")
        rows = self._registered_tr_rows.get(data_path)
        if rows is None:
            path = resolve_project_path(
                self.project_root,
                data_path,
                must_exist=True,
            )
            rows = {}
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
                        f"registered TR JSONL 格式无效：{path}:{line_number}"
                    ) from exc
                if not isinstance(value, dict) or "sample_id" not in value:
                    raise DataFormatError(
                        f"registered TR 记录无效：{path}:{line_number}"
                    )
                rows[str(value["sample_id"])] = value
            self._registered_tr_rows[data_path] = rows
        value = rows.get(sample_id)
        if value is None:
            raise DataIntegrityError(f"registered TR 缺少事件：{sample_id}")
        return value

    def _select_split_path(self, split_path: str | Path | None) -> Path | None:
        if split_path is not None:
            return resolve_project_path(
                self.project_root,
                split_path,
                must_exist=True,
            )
        candidates = (
            self.project_root / "data" / "manifests" / "cq08_grouped_split.json",
            self.project_root
            / "data"
            / "processed"
            / "RADAR-LN3-V16-cq08-capture-content-grouped-index.csv",
        )
        return next((path for path in candidates if path.is_file()), None)

    def _load_split_map(self, path: Path | None) -> dict[str, str]:
        if path is None:
            return {}

        result: dict[str, str] = {}

        def add_assignment(parent: Any, split: Any) -> None:
            parent_id = str(parent).strip()
            if not parent_id:
                return
            split_name = normalise_split_name(split)
            old = result.setdefault(parent_id, split_name)
            if old != split_name:
                raise DataSplitError(
                    f"CQ08 split source crosses parent: {parent_id}"
                )

        if path.suffix.lower() == ".csv":
            rows = read_delimited(path, encodings=("utf-8-sig", "utf-8"))
            for row in rows:
                add_assignment(
                    row.get("parent_sample_id", ""),
                    row.get("split_name", ""),
                )
            return result
        payload = read_json(path)
        if not isinstance(payload, Mapping):
            raise DataSplitError("CQ08 split source must be an object")
        excluded = payload.get("excluded_product_only_parent_ids", [])
        if excluded is not None:
            if not isinstance(excluded, list):
                raise DataSplitError(
                    "CQ08 split excluded_product_only_parent_ids 必须是列表"
                )
            for parent_id in excluded:
                value = str(parent_id).strip()
                if value:
                    self._split_excluded_parent_ids.add(value)
        if isinstance(payload.get("parent_splits"), Mapping):
            for parent, split in payload["parent_splits"].items():
                add_assignment(parent, split)
            return result
        for key, split in (
            ("train_group_ids", "train"),
            ("val_group_ids", "validation"),
            ("validation_group_ids", "validation"),
            ("test_group_ids", "test"),
        ):
            for track_id in payload.get(key, []) or []:
                add_assignment(track_id, split)
        return result

    def _logical_path(self, path: Path) -> str:
        try:
            path.relative_to(self.project_root)
        except ValueError:
            try:
                relative = path.relative_to(self.source_root).as_posix()
            except ValueError as exc:
                raise DataReadError(f"CQ08 path is outside configured source root: {path}") from exc
            return f"external/cq08/{relative}"
        return relative_project_path(self.project_root, path)

    def _discover_source_files(self) -> None:
        track_candidates: list[tuple[tuple[int, int], Path, int]] = []
        for path in sorted(self.track_root.glob("Tracks_*.txt"), key=lambda value: value.name):
            match = TRACK_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            track_candidates.append(
                (
                    (int(match["track_id"]), int(match["label_id"])),
                    path,
                    int(match["track_length"]),
                )
            )
        track_candidates.sort(key=lambda value: value[0])
        selected_keys: set[tuple[int, int]] | None = None
        split_candidates = track_candidates
        if self.parent_split is not None:
            split_candidates = [
                value
                for value in track_candidates
                if self._split_for_track(int(value[0][0])) == self.parent_split
            ]
        if self.parent_limit is not None:
            selected_keys = {
                key for key, _, _ in split_candidates[: self.parent_limit]
            }
        elif self.parent_split is not None:
            selected_keys = {key for key, _, _ in split_candidates}
        for key, path, track_length_hint in track_candidates:
            if selected_keys is not None and key not in selected_keys:
                continue
            if key in self._track_files:
                raise DataPairingError(f"duplicate CQ08 track file identity: {key}")
            rows = read_delimited(path, encodings=("gbk", "utf-8-sig", "utf-8"))
            _validate_source_rows(
                rows,
                columns=TRACK_COLUMNS,
                path=path,
                length_hint=track_length_hint,
                source_kind="track",
                expected_parent_id=key[0],
            )
            self._track_files[key] = {
                "path": path,
                "track_length_hint": track_length_hint,
                "row_count": len(rows),
            }
            self._track_rows[key] = rows
        selected_track_ids = (
            None
            if selected_keys is None
            else {track_id for track_id, _ in selected_keys}
        )
        for path in sorted(self.point_root.glob("PointTracks_*.txt"), key=lambda value: value.name):
            match = POINT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            key = (int(match["track_id"]), int(match["label_id"]))
            if selected_keys is not None and key not in selected_keys:
                continue
            if key in self._point_files:
                raise DataPairingError(f"duplicate CQ08 point file identity: {key}")
            rows = read_delimited(path, encodings=("gbk", "utf-8-sig", "utf-8"))
            point_length_hint = int(match["track_length"])
            _validate_source_rows(
                rows,
                columns=POINT_COLUMNS,
                path=path,
                length_hint=point_length_hint,
                source_kind="point",
                expected_parent_id=key[0],
            )
            self._point_files[key] = {
                "path": path,
                "row_count": len(rows),
            }
            self._point_rows[key] = rows
        for key, track in self._track_files.items():
            point = self._point_files.get(key)
            if point is None:
                continue
            if int(track["row_count"]) != int(point["row_count"]):
                raise DataPairingError(
                    "CQ08 track/point row counts differ: "
                    f"track={key}, track_rows={track['row_count']}, "
                    f"point_rows={point['row_count']}"
                )
            track_ids = {
                _strict_positive_integer(
                    row.get("批号"),
                    name="track parent ID",
                    path=Path(track["path"]),
                )
                for row in self._track_rows[key]
            }
            point_ids = {
                _strict_positive_integer(
                    row.get("批号"),
                    name="point parent ID",
                    path=Path(point["path"]),
                )
                for row in self._point_rows[key]
            }
            if track_ids != point_ids:
                raise DataPairingError(
                    f"CQ08 track/point parent identity differs: {key}"
                )
        for path in sorted(self.raw_root.glob("*_Label_*.dat"), key=lambda value: value.name):
            match = RAW_ECHO_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            key = (int(match["track_id"]), int(match["label_id"]))
            if selected_keys is not None and key not in selected_keys:
                continue
            if key in self._raw_files:
                raise DataPairingError(f"duplicate CQ08 raw echo identity: {key}")
            self._raw_files[key] = {"path": path, "file_size": int(path.stat().st_size)}
        orphan_point_keys = sorted(set(self._point_files) - set(self._track_files))
        if orphan_point_keys:
            raise DataPairingError(
                "CQ08 point files reference tracks without a track source: "
                f"{orphan_point_keys[:5]}"
            )
        orphan_raw_keys = sorted(set(self._raw_files) - set(self._track_files))
        if orphan_raw_keys:
            raise DataPairingError(
                "CQ08 raw echo files reference tracks without a track source: "
                f"{orphan_raw_keys[:5]}"
            )
        if selected_track_ids is None:
            mat_paths = self.mat_root.glob("*.mat")
        else:
            mat_paths = (
                path
                for track_id in sorted(selected_track_ids)
                for path in self.mat_root.glob(f"*_DAUR_RD_*_{track_id}.mat")
            )
        for path in sorted(mat_paths, key=lambda value: value.name):
            match = MAT_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            track_id = int(match["track_id"])
            frame_number = int(match["frame_index"])
            if frame_number <= 0:
                raise DataPairingError(
                    f"CQ08 MAT frame number must be one-based: {path}"
                )
            self._mat_files[track_id].append(
                {
                    "path": path,
                    "capture_id": str(match["capture_id"]),
                    "label_text": str(match["label_text"]),
                    "frame_index": frame_number,
                }
            )
        for values in self._mat_files.values():
            values.sort(key=lambda value: (int(value["frame_index"]), str(value["path"])))
            frame_numbers = [int(value["frame_index"]) for value in values]
            if len(frame_numbers) != len(set(frame_numbers)):
                raise DataPairingError(
                    "CQ08 MAT frame numbers are duplicated within a track"
                )
        orphan_mat_tracks = sorted(
            set(self._mat_files) - {track_id for track_id, _ in self._track_files}
        )
        if orphan_mat_tracks:
            raise DataPairingError(
                "CQ08 MAT files reference tracks without a track source: "
                f"{orphan_mat_tracks[:5]}"
            )

    def _parent_keys(self) -> list[tuple[int, int]]:
        return [
            key
            for key in sorted(self._track_files)
            if not self._exclusion_reasons(key)
        ]

    def _exclusion_reasons(self, key: tuple[int, int]) -> tuple[str, ...]:
        """根据实际来源文件状态返回 parent 的排除原因。"""

        track_id = int(key[0])
        reasons: list[str] = []
        # 只有 split 文件显式登记的 product-only parent 才进入 audit。
        # 其他未覆盖 parent 必须在 assign_split 中失败，避免吞掉 split 缺口。
        parent_id = f"cq08|track|{track_id}"
        if parent_id in self._split_excluded_parent_ids or str(track_id) in self._split_excluded_parent_ids:
            reasons.append("not_in_split_source")
        if key not in self._point_files:
            reasons.append("missing_point_file")
        if key not in self._raw_files:
            reasons.append("missing_raw_echo")
        if not self._mat_files.get(track_id):
            reasons.append("missing_rd_mat")
        return tuple(reasons)

    def _split_for_track(self, track_id: int) -> str:
        split = self._split_map.get(
            f"cq08|track|{track_id}",
            self._split_map.get(str(track_id), ""),
        )
        if not split:
            return ""
        return normalise_split_name(split)

    def normalize_label(self, raw_label: str) -> tuple[int, str]:
        return _normalise_label(raw_label)

    def build_sample_id(self, parent: ParentRecord, event_index: int) -> str:
        return f"{parent.parent_sample_id}:event-{int(event_index)}"

    def _parent_record(self, key: tuple[int, int]) -> ParentRecord:
        track_id, raw_label_id = key
        _, label_name = self.normalize_label(str(raw_label_id))
        normalized_label_id, _ = self.normalize_label(label_name)
        track = self._track_files[key]
        point = self._point_files.get(key)
        raw = self._raw_files.get(key)
        return ParentRecord(
            dataset_name=CQ08_DATASET_NAME,
            source_name=CQ08_SOURCE_NAME,
            parent_sample_id=f"cq08|track|{track_id}",
            track_id=str(track_id),
            label_id=normalized_label_id,
            label_name=label_name,
            source_path=self._logical_path(Path(track["path"])),
            data_path=(
                f"data/cq08/tr/{_safe_name(f'cq08|track|{track_id}')}.jsonl"
            ),
            split_name=self._split_for_track(track_id),
            available=not bool(self._exclusion_reasons(key)),
            event_count=int(track["row_count"]),
            file_size=int(Path(track["path"]).stat().st_size),
            sha256=sha256_file(track["path"]),
            metadata={
                "raw_label_id": raw_label_id,
                "track_length_hint": int(track["track_length_hint"]),
                "point_source_path": (
                    self._logical_path(Path(point["path"])) if point else ""
                ),
                "raw_echo_source_path": (
                    self._logical_path(Path(raw["path"])) if raw else ""
                ),
                "raw_echo_file_size": int(raw["file_size"]) if raw else 0,
                "rd_mat_count": len(self._mat_files.get(track_id, [])),
                "rd_available": bool(self._mat_files.get(track_id)),
            },
        )

    def discover_excluded_parents(self) -> list[ParentRecord]:
        """返回因来源数据不完整而不进入 split 的 parent 审计记录。"""

        if self.registered_index and self._registered_manifest is not None:
            return [
                ParentRecord(**dict(record))
                for record in self._registered_manifest.audit.get(
                    "excluded_parents", []
                )
                if isinstance(record, Mapping)
            ]

        records: list[ParentRecord] = []
        for key in sorted(self._track_files):
            reasons = self._exclusion_reasons(key)
            if not reasons:
                continue
            record = self._parent_record(key)
            records.append(
                ParentRecord(
                    dataset_name=record.dataset_name,
                    source_name=record.source_name,
                    parent_sample_id=record.parent_sample_id,
                    track_id=record.track_id,
                    label_id=record.label_id,
                    label_name=record.label_name,
                    source_path=record.source_path,
                    data_path=record.data_path,
                    split_name="",
                    available=False,
                    event_count=record.event_count,
                    file_size=record.file_size,
                    sha256=record.sha256,
                    metadata={
                        **dict(record.metadata),
                        "excluded": True,
                        "exclusion_reasons": list(reasons),
                    },
                )
            )
        return records

    def discover_parents(self) -> list[ParentRecord]:
        if self.registered_index:
            parents = list(self._registered_parents)
            if self.parent_split is not None:
                parents = [parent for parent in parents if parent.split_name == self.parent_split]
            if self.parent_limit is not None:
                parents = parents[: self.parent_limit]
            return parents
        return [self._parent_record(key) for key in self._parent_keys()]

    def _load_rd_metadata(self, track_id: int) -> list[dict[str, Any]]:
        cached = self._rd_metadata.get(track_id)
        if cached is not None:
            return cached
        result: list[dict[str, Any]] = []
        for item in self._mat_files.get(track_id, []):
            path = Path(item["path"])
            try:
                payload = read_mat(
                    path,
                    variable_names=(
                        "index_trackPointNo",
                        "debug_info",
                    ),
                )
                point_index_value = np.asarray(payload.get("index_trackPointNo", -1)).reshape(-1)
                if point_index_value.size != 1:
                    raise DataFormatError(f"CQ08 MAT point index is not scalar: {path}")
                track_point_index = _strict_positive_integer(
                    point_index_value[0],
                    name="MAT track point index",
                    path=path,
                )
                debug = payload.get("debug_info")
                duration, duration_source = _rd_duration_seconds(debug)
                result.append(
                    {
                        "path": path,
                        "capture_id": str(item["capture_id"]),
                        "label_text": str(item["label_text"]),
                        "frame_index": int(item["frame_index"]),
                        "track_point_index": track_point_index,
                        "duration_seconds": max(0.0, duration),
                        "duration_source": duration_source,
                        "cpi_count": _as_int(getattr(debug, "cpi_count", None)),
                        "prt_num": _as_int(getattr(debug, "prt_num", None)),
                        "prt_seconds": _as_float(getattr(debug, "prt_s", None)),
                    }
                )
            except (DataFormatError, DataReadError):
                raise
            except Exception as exc:
                raise DataReadError(f"cannot inspect CQ08 MAT: {path}") from exc
        result.sort(key=lambda value: (int(value["frame_index"]), str(value["path"])))
        self._rd_metadata[track_id] = result
        return result

    def discover_events(self, parent: ParentRecord) -> list[EventRecord]:
        if self.registered_index:
            events = self._registered_events.get(parent.parent_sample_id)
            if events is None:
                raise DataPairingError(
                    f"registered manifest 缺少 parent 事件：{parent.parent_sample_id}"
                )
            return list(events)
        key = (int(parent.track_id), int(parent.metadata["raw_label_id"]))
        track_rows = self._track_rows[key]
        point_rows = self._point_rows.get(key, [])
        rd_by_point: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for frame in self._load_rd_metadata(int(parent.track_id)):
            _, frame_label = self.normalize_label(str(frame["label_text"]))
            if frame_label != parent.label_name:
                raise DataPairingError(
                    "CQ08 MAT label differs from track label: "
                    f"track={parent.parent_sample_id}, mat={frame_label}"
                )
            track_point_index = int(frame["track_point_index"])
            if not 1 <= track_point_index <= len(track_rows):
                raise DataPairingError(
                    "CQ08 MAT track point index is outside the track event range: "
                    f"track={parent.parent_sample_id}, point={track_point_index}, "
                    f"event_count={len(track_rows)}"
                )
            rd_by_point[track_point_index].append(frame)
        if len(point_rows) not in {0, len(track_rows)}:
            raise DataPairingError(
                "CQ08 track and point row counts differ: "
                f"track={parent.parent_sample_id}, "
                f"track_rows={len(track_rows)}, point_rows={len(point_rows)}"
            )
        events: list[EventRecord] = []
        for event_index, track_row in enumerate(track_rows):
            point_row = (
                point_rows[event_index] if event_index < len(point_rows) else None
            )
            expected_point_index = event_index + 1
            tr_values = {
                "time_seconds": _parse_time_seconds(track_row["点时间"]),
                "range_m": _as_float(track_row["滤波距离"]),
                "azimuth_deg": _as_float(track_row["滤波方位"]),
                "elevation_deg": _as_float(track_row["滤波俯仰"]),
                "speed_mps": _as_float(track_row["全速度"]),
                "radial_speed_mps": (
                    _as_float(point_row["多普勒速度"])
                    if point_row is not None
                    else None
                ),
                "velocity_x_mps": _as_float(track_row["X向速度"]),
                "velocity_y_mps": _as_float(track_row["Y向速度"]),
                "velocity_z_mps": _as_float(track_row["Z向速度"]),
                "course_deg": _as_float(track_row["航向"]),
                "amplitude_db": (
                    _as_float(point_row["和幅度"])
                    if point_row is not None
                    else None
                ),
                "snr_db": (
                    _snr_db(point_row["信噪比"])
                    if point_row is not None
                    else None
                ),
                "height_m": None,
                "latitude": None,
                "longitude": None,
            }
            if point_row is not None:
                tr_values.update(
                    {
                        "point_range_m": _as_float(point_row["距离"]),
                        "point_azimuth_deg": _as_float(point_row["方位"]),
                        "point_elevation_deg": _as_float(point_row["俯仰"]),
                        "point_radial_speed_mps": _as_float(point_row["多普勒速度"]),
                        "point_amplitude_raw": _as_float(point_row["和幅度"]),
                        "point_snr_raw": _as_float(point_row["信噪比"]),
                        "point_snr_db": _snr_db(point_row["信噪比"]),
                        "point_count": _as_int(point_row["原始点数量"]),
                    }
                )
            frames = rd_by_point.get(expected_point_index, [])
            source_paths = [self._logical_path(Path(frame["path"])) for frame in frames]
            sample_id = self.build_sample_id(parent, event_index)
            events.append(
                EventRecord(
                    dataset_name=CQ08_DATASET_NAME,
                    source_name=CQ08_SOURCE_NAME,
                    sample_id=sample_id,
                    parent_sample_id=parent.parent_sample_id,
                    track_id=parent.track_id,
                    causal_event_index=event_index,
                    label_id=parent.label_id,
                    label_name=parent.label_name,
                    split_name=parent.split_name,
                    available=True,
                    source_path=parent.source_path,
                    data_path=parent.data_path,
                    file_size=parent.file_size,
                    sha256=parent.sha256,
                    metadata={
                        "track_point_index": event_index + 1,
                        "tr_values": tr_values,
                        "track_time_raw": str(track_row["点时间"]),
                        "point_available": point_row is not None,
                        "rd_frame_paths": source_paths,
                        "raw_echo_source_path": str(
                            parent.metadata.get("raw_echo_source_path", "")
                        ),
                        "rd_mat_frame_numbers": [
                            int(frame["frame_index"]) for frame in frames
                        ],
                        "rd_frame_metadata": [
                            {
                                key: _json_safe(value)
                                for key, value in frame.items()
                                if key != "path"
                            }
                            for frame in frames
                        ],
                        "rd_available": bool(frames),
                        "rd_frame_count": len(frames),
                        "rd_multiple_frames": len(frames) > 1,
                        "rd_frame_order": "mat_frame_index_ascending",
                        "rd_time_semantics": "parent_relative_cpi_elapsed_seconds",
                        "rd_time_schema_version": RD_FRAME_TIME_SCHEMA_VERSION,
                        "tr_field_semantics": {
                            "radial_speed_mps": "CQ08 point column 多普勒速度",
                            "amplitude_db": (
                                "CQ08 point column 和幅度; linear magnitude retained under the "
                                "legacy V0 field name"
                            ),
                            "snr_db": "CQ08 point column 信噪比",
                            "speed_mps": "CQ08 track column 全速度",
                        },
                    },
                )
            )
        mapped_frames = sum(len(value) for value in rd_by_point.values())
        if mapped_frames != len(self._load_rd_metadata(int(parent.track_id))):
            raise DataPairingError(
                f"CQ08 RD frames were silently dropped: {parent.parent_sample_id}"
            )
        return annotate_event_records(events)

    def read_parent(self, parent: ParentRecord) -> ParentData:
        events = self.discover_events(parent)
        return ParentData(
            parent=parent,
            events=events,
            metadata={
                "event_count": len(events),
                "rd_event_count": sum(
                    bool(event.metadata.get("rd_available")) for event in events
                ),
            },
        )

    def read_event(self, event: EventRecord) -> EventData:
        if self.registered_index:
            return self._read_registered_event(event)
        if self._canonical_rd is not None:
            canonical = self._canonical_rd.event(
                event.parent_sample_id,
                event.causal_event_index,
            )
            rd_values = canonical.get("rd_values")
            timing = event.metadata.get("tr_timing")
            timing = dict(timing) if isinstance(timing, Mapping) else {}
            return EventData(
                sample_id=event.sample_id,
                parent_sample_id=event.parent_sample_id,
                track_id=event.track_id,
                causal_event_index=event.causal_event_index,
                label_id=event.label_id,
                label_name=event.label_name,
                split_name=event.split_name,
                tr_values=dict(event.metadata.get("tr_values") or {}),
                rd_values=rd_values,
                tr_available=True,
                rd_available=rd_values is not None,
                metadata={
                    **dict(event.metadata),
                    "canonical_rd": rd_values is not None,
                    "rd_time_semantics": "canonical_rd_tensor_time_seconds",
                },
                tr_time_seconds=float(timing["time_seconds"])
                if timing.get("time_seconds") is not None else None,
                tr_dt_seconds=float(timing["dt_seconds"])
                if timing.get("dt_seconds") is not None else None,
                tr_dt_valid=bool(timing["dt_valid"])
                if timing.get("dt_valid") is not None else None,
                tr_dt_reason=str(timing["dt_reason"])
                if timing.get("dt_reason") is not None else None,
                tr_dt_is_negative=bool(timing["dt_is_negative"])
                if timing.get("dt_is_negative") is not None else None,
                tr_time_order_status=str(timing["time_order_status"])
                if timing.get("time_order_status") is not None else None,
            )
        rd_frames: list[dict[str, Any]] = []
        frame_metadata = event.metadata.get("rd_frame_metadata") or []
        elapsed = 0.0
        for position, raw_path in enumerate(event.metadata.get("rd_frame_paths", [])):
            path = resolve_frame_source_path(
                self.project_root,
                raw_path,
                source_root=self.source_root,
                source_config=self.source_config,
            )
            payload = read_mat(
                path,
                variable_names=("data_proc_MTD_result_db", "data_proc_MTD_result"),
            )
            if "data_proc_MTD_result_db" not in payload:
                raise DataFormatError(f"CQ08 MAT lacks amplitude: {path}")
            complex_values = payload.get("data_proc_MTD_result")
            sanitized = sanitize_rd_amplitude_db(
                payload["data_proc_MTD_result_db"],
                complex_values,
                name=str(path),
            )
            amplitude = sanitized.values
            if amplitude.ndim != 2 or any(int(size) <= 0 for size in amplitude.shape):
                raise DataFormatError(
                    f"CQ08 RD amplitude must be a non-empty 2D array: {path}"
                )
            if complex_values is None:
                phase = np.zeros_like(amplitude, dtype=np.float32)
            else:
                complex_array = np.asarray(complex_values)
                if complex_array.shape != amplitude.shape:
                    raise DataFormatError(
                        f"CQ08 MAT complex/amplitude shapes differ: {path}: "
                        f"{complex_array.shape} != {amplitude.shape}"
                    )
                phase = np.angle(complex_array).astype(np.float32)
            phase = finite_array(phase, name=f"{path}:phase", dtype=np.float32)
            rd_frames.append(
                {
                    "amplitude": amplitude,
                    "phase": phase,
                    "source_path": raw_path,
                    "frame_index": (
                        int(frame_metadata[position]["frame_index"])
                        if position < len(frame_metadata)
                        and isinstance(frame_metadata[position], Mapping)
                        and frame_metadata[position].get("frame_index") is not None
                        else position
                    ),
                    "time_seconds": float(elapsed),
                    "amplitude_sanitization": sanitized.audit,
                }
            )
            duration = 0.0
            if position < len(frame_metadata) and isinstance(frame_metadata[position], Mapping):
                duration = max(0.0, float(frame_metadata[position].get("duration_seconds") or 0.0))
            elapsed += duration
        timing = event.metadata.get("tr_timing")
        timing = dict(timing) if isinstance(timing, Mapping) else {}
        return EventData(
            sample_id=event.sample_id,
            parent_sample_id=event.parent_sample_id,
            track_id=event.track_id,
            causal_event_index=event.causal_event_index,
            label_id=event.label_id,
            label_name=event.label_name,
            split_name=event.split_name,
            tr_values=dict(event.metadata.get("tr_values") or {}),
            rd_values={"frames": rd_frames} if rd_frames else None,
            tr_available=True,
            rd_available=bool(rd_frames),
            metadata={
                **dict(event.metadata),
                "rd_frame_count": len(rd_frames),
                "rd_multiple_frames": len(rd_frames) > 1,
                "rd_time_semantics": "parent_relative_cpi_elapsed_seconds",
                "rd_time_schema_version": RD_FRAME_TIME_SCHEMA_VERSION,
            },
            tr_time_seconds=(
                float(timing["time_seconds"])
                if timing.get("time_seconds") is not None
                else None
            ),
            tr_dt_seconds=(
                float(timing["dt_seconds"])
                if timing.get("dt_seconds") is not None
                else None
            ),
            tr_dt_valid=(
                bool(timing["dt_valid"])
                if timing.get("dt_valid") is not None
                else None
            ),
            tr_dt_reason=(
                str(timing["dt_reason"])
                if timing.get("dt_reason") is not None
                else None
            ),
            tr_dt_is_negative=(
                bool(timing["dt_is_negative"])
                if timing.get("dt_is_negative") is not None
                else None
            ),
            tr_time_order_status=(
                str(timing["time_order_status"])
                if timing.get("time_order_status") is not None
                else None
            ),
        )

    def _read_registered_event(self, event: EventRecord) -> EventData:
        """从已物化 TR/RD 产物读取事件，不触碰原始 MAT。"""

        tr_row = self._registered_tr_row(
            event.metadata.get("tr_record") or {
                "data_path": event.data_path,
            },
            event.sample_id,
        )
        tr_values = dict(tr_row.get("values") or {})
        rd_record = event.metadata.get("rd_record")
        rd_values: dict[str, Any] | None = None
        if isinstance(rd_record, Mapping) and bool(rd_record.get("available")):
            data_path = str(rd_record.get("data_path") or "")
            payload = read_npz(
                resolve_project_path(self.project_root, data_path, must_exist=True),
                required_keys=(
                    "amplitude",
                    "phase",
                    "time_seconds",
                    "frame_event_indices",
                ),
            )
            archive_indices = np.asarray(payload["frame_event_indices"])
            selector = np.flatnonzero(
                archive_indices == int(event.causal_event_index)
            )
            if selector.size:
                rd_values = {
                    "amplitude": np.asarray(payload["amplitude"])[selector],
                    "phase": np.asarray(payload["phase"])[selector],
                    "time_seconds": np.asarray(payload["time_seconds"])[selector],
                    "frame_indices": selector,
                    "frame_metadata": list(
                        (rd_record.get("metadata") or {}).get("frame_metadata", [])
                    ),
                    "frame_source_paths": list(
                        (rd_record.get("metadata") or {}).get("frame_source_paths", [])
                    ),
                    "frame_count": int(selector.size),
                    "time_semantics": "parent_relative_cpi_elapsed_seconds",
                }
        timing = dict(event.metadata.get("tr_timing") or {})
        return EventData(
            sample_id=event.sample_id,
            parent_sample_id=event.parent_sample_id,
            track_id=event.track_id,
            causal_event_index=event.causal_event_index,
            label_id=event.label_id,
            label_name=event.label_name,
            split_name=event.split_name,
            tr_values=tr_values,
            rd_values=rd_values,
            tr_available=True,
            rd_available=rd_values is not None,
            metadata=dict(event.metadata),
            tr_time_seconds=float(timing["time_seconds"])
            if timing.get("time_seconds") is not None
            else None,
            tr_dt_seconds=float(timing["dt_seconds"])
            if timing.get("dt_seconds") is not None
            else None,
            tr_dt_valid=bool(timing["dt_valid"])
            if timing.get("dt_valid") is not None
            else None,
            tr_dt_reason=str(timing["dt_reason"])
            if timing.get("dt_reason") is not None
            else None,
            tr_dt_is_negative=bool(timing["dt_is_negative"])
            if timing.get("dt_is_negative") is not None
            else None,
            tr_time_order_status=str(timing["time_order_status"])
            if timing.get("time_order_status") is not None
            else None,
        )

    def read_event_frames(self, event: EventRecord):
        """按事件读取 raw-DAT frame，供 features provider 使用。"""

        if self.registered_index and not self.source_root.is_dir():
            raise DataReadError(
                "registered-index 模式没有可用的原始回波目录；"
                "请注入 feature cache 或显式配置 raw source root"
            )
        if self._raw_echo_provider is None:
            self._raw_echo_provider = CQ08RawEchoProvider(
                self.project_root,
                source_root=self.source_root,
            )
        return self._raw_echo_provider.read_event_frames(event)

    def assign_split(self, parent_ids: list[str]) -> dict[str, str]:
        """为 parent 绑定固定 split；registered-index 直接复用 manifest。"""

        if self.registered_index:
            assignments = {
                str(parent.parent_sample_id): normalise_split_name(parent.split_name)
                for parent in self._registered_parents
                if parent.split_name
            }
            missing = sorted(set(parent_ids) - set(assignments))
            if missing:
                raise DataSplitError(
                    "registered manifest 缺少 parent split："
                    f"{missing[:5]}"
                )
            return {parent_id: assignments[parent_id] for parent_id in parent_ids}
        result: dict[str, str] = {}
        missing: list[str] = []
        for parent_id in parent_ids:
            track_id = parent_id.rsplit("|", 1)[-1]
            split = self._split_map.get(
                parent_id,
                self._split_map.get(track_id, ""),
            )
            if not split:
                missing.append(parent_id)
            else:
                result[parent_id] = normalise_split_name(split)
        if missing:
            raise DataSplitError(
                "CQ08 split input does not cover discovered parents; "
                f"count={len(missing)}, first={missing[:5]}"
            )
        if set(result) != set(parent_ids):
            raise DataSplitError("CQ08 split assignment is incomplete")
        return result

    def _data_paths(self, parent: ParentRecord) -> tuple[Path, Path]:
        tr = tr_root(self.project_root) / (
            f"{_safe_name(parent.parent_sample_id)}.jsonl"
        )
        rd = rd_root(self.project_root) / (
            f"{_safe_name(parent.parent_sample_id)}.npz"
        )
        return tr, rd

    @staticmethod
    def _resize_plane(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or any(int(size) <= 0 for size in values.shape):
            raise DataFormatError(
                f"CQ08 RD plane must be a non-empty 2D array: {values.shape}"
            )
        if values.shape == RD_OUTPUT_SHAPE:
            return values
        try:
            from scipy.ndimage import zoom
        except ImportError as exc:
            raise DataReadError("读取 CQ08 RD 时需要 scipy") from exc
        factors = (
            RD_OUTPUT_SHAPE[0] / values.shape[0],
            RD_OUTPUT_SHAPE[1] / values.shape[1],
        )
        result = zoom(
            values,
            factors,
            order=1,
            mode="nearest",
            prefilter=False,
        ).astype(np.float32)
        if result.shape != RD_OUTPUT_SHAPE:
            raise DataFormatError(
                f"CQ08 RD 数据 shape 在调整后仍不一致：{result.shape}"
            )
        return result

    def _load_resized_frame(
        self,
        frame: Mapping[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
        path = Path(frame["path"])
        payload = read_mat(
            path,
            variable_names=(
                "data_proc_MTD_result_db",
                "data_proc_MTD_result",
                "debug_info",
            ),
        )
        if "data_proc_MTD_result_db" not in payload:
            raise DataFormatError(f"CQ08 MAT lacks amplitude: {path}")
        complex_values = payload.get("data_proc_MTD_result")
        sanitized = sanitize_rd_amplitude_db(
            payload["data_proc_MTD_result_db"],
            complex_values,
            name=str(path),
        )
        amplitude_raw = sanitized.values
        amplitude = finite_array(
            self._resize_plane(amplitude_raw),
            name=f"{path}:amplitude",
            dtype=np.float32,
        )
        if complex_values is None:
            phase_raw = np.zeros_like(amplitude_raw, dtype=np.float32)
        else:
            complex_array = np.asarray(complex_values)
            if complex_array.shape != amplitude_raw.shape:
                raise DataFormatError(
                    f"CQ08 MAT complex/amplitude shapes differ: {path}: "
                    f"{complex_array.shape} != {amplitude_raw.shape}"
                )
            phase_raw = np.angle(complex_array).astype(np.float32)
        phase = finite_array(
            self._resize_plane(phase_raw),
            name=f"{path}:phase",
            dtype=np.float32,
        )
        debug = payload.get("debug_info")
        duration, _ = _rd_duration_seconds(debug)
        return amplitude, phase, max(0.0, duration), sanitized.audit

    def materialize_parent(
        self,
        parent: ParentRecord,
        events: list[EventRecord],
        *,
        split_name: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """按一个 parent 写出 TR/RD 规范化产物及其事件 artifact。"""

        if self.registered_index:
            raise DataReadError(
                "registered-index CQ08 adapter 只读，不能重新物化数据产物"
            )
        if normalise_split_name(split_name) != normalise_split_name(parent.split_name):
            raise DataSplitError("CQ08 materialization split differs from parent")
        tr_path, rd_path = self._data_paths(parent)
        tr_lines: list[str] = []
        for event in events:
            tr_lines.append(
                json.dumps(
                    {
                        "schema_version": CQ08_TR_SCHEMA_VERSION,
                        "sample_id": event.sample_id,
                        "parent_sample_id": event.parent_sample_id,
                        "track_id": event.track_id,
                        "causal_event_index": event.causal_event_index,
                        "label_id": event.label_id,
                        "label_name": event.label_name,
                        "split_name": event.split_name,
                        "values": _json_safe(event.metadata.get("tr_values") or {}),
                        "metadata": {
                            "track_point_index": event.metadata.get("track_point_index"),
                            "track_time_raw": event.metadata.get("track_time_raw"),
                            "point_available": event.metadata.get("point_available"),
                            "tr_timing": event.metadata.get("tr_timing"),
                        },
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
        tr_payload = "".join(tr_lines).encode("utf-8")
        tr_hash = write_bytes(
            tr_path, tr_payload, overwrite=overwrite, label="数据产物"
        )

        frames: list[np.ndarray] = []
        phases: list[np.ndarray] = []
        times: list[float] = []
        frame_event_indices: list[int] = []
        event_frame_indices: dict[str, list[int]] = {
            event.sample_id: [] for event in events
        }
        frame_sanitization: dict[str, dict[str, Any]] = {}
        frame_times: dict[str, float] = {}
        frame_cursor = 0
        elapsed = 0.0
        # TODO：后续将控制台进度输出替换为可注入的日志或回调接口。
        all_frames = self._load_rd_metadata(int(parent.track_id))
        frame_by_logical_path = {
            self._logical_path(Path(frame["path"])): frame for frame in all_frames
        }
        event_by_sample_id = {event.sample_id: event for event in events}
        event_by_path: dict[str, str] = {}
        for event in events:
            for raw_path in event.metadata.get("rd_frame_paths", []):
                logical_path = str(raw_path)
                if logical_path not in frame_by_logical_path:
                    raise DataPairingError(
                        f"CQ08 RD frame cannot be resolved for event: {event.sample_id}"
                    )
                if logical_path in event_by_path:
                    raise DataPairingError(
                        f"CQ08 RD frame is assigned to multiple events: {logical_path}"
                    )
                event_by_path[logical_path] = event.sample_id

        for source_frame in all_frames:
            logical_path = self._logical_path(Path(source_frame["path"]))
            sample_id = event_by_path.get(logical_path)
            if sample_id is None:
                raise DataPairingError(
                    f"CQ08 RD frame is not assigned to an event: {logical_path}"
                )
            amplitude, phase, duration, sanitization = self._load_resized_frame(source_frame)
            frame_sanitization[logical_path] = sanitization
            frame_times[logical_path] = float(elapsed)
            frames.append(amplitude)
            phases.append(phase)
            times.append(float(elapsed))
            elapsed += duration
            event = event_by_sample_id[sample_id]
            frame_event_indices.append(int(event.causal_event_index))
            event_frame_indices[sample_id].append(frame_cursor)
            frame_cursor += 1

        rd_available = bool(frames)
        rd_hash: str | None = None
        if rd_available:
            amplitude_array = np.stack(frames, axis=0).astype(np.float32)
            phase_array = np.stack(phases, axis=0).astype(np.float32)
            time_array = np.asarray(times, dtype=np.float32)
            event_array = np.asarray(frame_event_indices, dtype=np.int64)
            payload = _deterministic_npz_bytes(
                {
                    "amplitude": amplitude_array,
                    "frame_event_indices": event_array,
                    "phase": phase_array,
                    "schema_version": np.asarray(CQ08_RD_SCHEMA_VERSION),
                    "time_seconds": time_array,
                }
            )
            rd_hash = write_bytes(
                rd_path, payload, overwrite=overwrite, label="数据产物"
            )
        elif rd_path.exists():
            if not overwrite:
                raise FileExistsError(
                    f"refusing to overwrite a different empty RD artifact: {rd_path}"
                )
            rd_path.unlink()
            rd_hash = None

        tr_size = int(tr_path.stat().st_size)
        rd_size = int(rd_path.stat().st_size) if rd_available else None
        tr_artifacts: dict[str, Any] = {}
        rd_artifacts: dict[str, Any] = {}
        for event in events:
            tr_artifacts[event.sample_id] = {
                "available": True,
                "source_path": event.source_path,
                "data_path": relative_project_path(self.project_root, tr_path),
                "file_size": tr_size,
                "sha256": tr_hash,
                "metadata": {
                    "schema_version": CQ08_TR_SCHEMA_VERSION,
                    "dtype": "json",
                    "event_count": len(events),
                },
            }
            frame_indices = event_frame_indices.get(event.sample_id, [])
            frame_paths = list(event.metadata.get("rd_frame_paths", []))
            frame_metadata: list[dict[str, Any]] = []
            for position, item in enumerate(event.metadata.get("rd_frame_metadata", [])):
                metadata = dict(item) if isinstance(item, Mapping) else {}
                if position < len(frame_paths):
                    audit = frame_sanitization.get(str(frame_paths[position]))
                    if audit is not None:
                        metadata["amplitude_sanitization"] = audit
                    if str(frame_paths[position]) in frame_times:
                        metadata["rd_frame_time_seconds"] = frame_times[str(frame_paths[position])]
                frame_metadata.append(metadata)
            event_audits = [
                dict(item.get("amplitude_sanitization") or {})
                for item in frame_metadata
                if isinstance(item, Mapping) and item.get("amplitude_sanitization")
            ]
            rd_artifacts[event.sample_id] = {
                "available": bool(frame_indices),
                "source_path": ";".join(event.metadata.get("rd_frame_paths", [])),
                "data_path": (
                    relative_project_path(self.project_root, rd_path)
                    if frame_indices
                    else ""
                ),
                "file_size": rd_size if frame_indices else None,
                "sha256": rd_hash if frame_indices else None,
                "pairing_basis": "MAT.index_trackPointNo",
                "metadata": {
                    "schema_version": CQ08_RD_SCHEMA_VERSION,
                    "dtype": "float32",
                    "shape": list(RD_OUTPUT_SHAPE),
                    "frame_indices": frame_indices,
                    "frame_metadata": frame_metadata,
                    "rd_frame_count": len(frame_indices),
                    "event_count": len(events),
                    "time_semantics": "CPI-relative elapsed seconds",
                    "time_schema_version": RD_FRAME_TIME_SCHEMA_VERSION,
                    "frame_cardinality": "many_frames_per_event",
                    "frame_order": "frame_index_ascending",
                    "rd_amplitude_sanitization": {
                        "schema_version": RD_AMPLITUDE_REPAIR_SCHEMA,
                        "repair_policy": RD_AMPLITUDE_REPAIR_POLICY,
                        "frame_count": len(event_audits),
                        "repaired_frame_count": sum(
                            int(bool(audit.get("zero_power_replaced_count")))
                            for audit in event_audits
                        ),
                        "zero_power_replaced_count": sum(
                            int(audit.get("zero_power_replaced_count", 0))
                            for audit in event_audits
                        ),
                    },
                },
            }
        return {
            "tr": tr_artifacts,
            "rd": rd_artifacts,
            "metadata": {
                "materialized": True,
                "tr_path": relative_project_path(self.project_root, tr_path),
                "rd_path": (
                    relative_project_path(self.project_root, rd_path)
                    if rd_available
                    else ""
                ),
                # 创建或复用属于运行细节，不应进入 manifest 的可复现哈希。
                "rd_status": "materialized" if rd_available else "unavailable",
                "rd_frame_count": len(frames),
            },
        }


FRAME_HEAD = 0xFA55FA55
FRAME_END = 0x55FA55FA
RAW_RANGE_BIN_COUNT = 31


def _raw_u32(payload: bytes, offset: int, count: int) -> tuple[int, ...]:
    size = 4 * int(count)
    if offset < 0 or offset + size > len(payload):
        raise DataFormatError("CQ08 raw-DAT frame header越界")
    return struct.unpack_from(f"<{int(count)}I", payload, offset)


def _locate_raw_dat_target_bin(frame: Any, record: RawTargetRecord) -> tuple[int, int]:
    """在 raw-DAT 帧内执行局部 MTD 搜索并输出零基目标坐标。"""

    try:
        from scipy.signal.windows import taylor
    except ImportError as exc:
        raise DataReadError("解析 CQ08 raw-DAT 目标位置需要 scipy") from exc
    pulse_count = int(frame.prt_num)
    window = taylor(pulse_count, nbar=4, sll=30.0, norm=True, sym=True).astype(
        np.float64
    )
    rd = np.fft.fftshift(
        np.fft.fft(np.asarray(frame.iq, dtype=np.complex128) * window[None, :], axis=1),
        axes=1,
    )
    raw_bin = int(record.doppler_bin_raw)
    mapped_one_based = (
        raw_bin - pulse_count // 2
        if raw_bin > pulse_count / 2
        else raw_bin + pulse_count // 2
    )
    mapped_zero_based = int(np.clip(mapped_one_based - 1, 0, pulse_count - 1))
    range_start = max(0, 15 - 5)
    range_stop = min(RAW_RANGE_BIN_COUNT, 15 + 5 + 1)
    doppler_start = max(0, mapped_zero_based - 5)
    doppler_stop = min(pulse_count, mapped_zero_based + 5 + 1)
    local = np.abs(rd[range_start:range_stop, doppler_start:doppler_stop])
    if local.size == 0 or not np.isfinite(local).all():
        raise DataIntegrityError("CQ08 raw-DAT 局部 MTD 目标搜索无效")
    local_row, local_column = np.unravel_index(int(np.argmax(local)), local.shape)
    return range_start + int(local_row), doppler_start + int(local_column)


def _parse_raw_echo_file(path: Path) -> list[RawEchoFrame]:
    """严格解析一个 CQ08 raw-DAT 文件，不读取 MAT。"""

    from .contracts import RawEchoFrame, RawTargetRecord

    try:
        payload = read_bytes(path)
    except OSError as exc:
        raise DataReadError(f"无法读取 CQ08 raw-DAT：{path}") from exc
    frames: list[RawEchoFrame] = []
    offset = 0
    while offset < len(payload):
        head, length_words, _scan_angle, point_count = _raw_u32(payload, offset, 4)
        if head != FRAME_HEAD:
            raise DataFormatError(f"CQ08 raw-DAT frame头错误：{path}:{offset}")
        frame_length = int(length_words) * 4
        if frame_length < 36 or offset + frame_length > len(payload):
            raise DataFormatError(f"CQ08 raw-DAT frame长度错误：{path}:{offset}")
        footer_offset = offset + frame_length - 4
        if _raw_u32(payload, footer_offset, 1)[0] != FRAME_END:
            raise DataFormatError(f"CQ08 raw-DAT frame尾错误：{path}:{offset}")
        if point_count > 100_000:
            raise DataFormatError(f"CQ08 raw-DAT target数量异常：{path}:{offset}")
        target_offset = offset + 16
        target_words = _raw_u32(payload, target_offset, int(point_count) * 4)
        metadata_offset = target_offset + int(point_count) * 16
        frequency_mhz, cpi_count, prt_num, prt_raw, data_length = _raw_u32(
            payload, metadata_offset, 5
        )
        # src-v3 的原始解析链允许 CPIcount=0。该值只表示 CPI 字段
        # 无法提供正的观测时长，后续由 _rd_duration_seconds 使用
        # prt_num * prt_seconds 回退；它不是 raw-DAT 帧损坏的证据。
        if frequency_mhz <= 0 or prt_num <= 0 or prt_raw <= 0:
            raise DataFormatError(f"CQ08 raw-DAT radar参数异常：{path}:{offset}")
        sample_count = int(prt_num) * RAW_RANGE_BIN_COUNT * 2
        sample_offset = metadata_offset + 20
        sample_bytes = sample_count * 4
        if sample_offset + sample_bytes > footer_offset:
            raise DataFormatError(f"CQ08 raw-DAT I/Q长度越界：{path}:{offset}")
        samples = np.frombuffer(
            payload,
            dtype="<f4",
            count=sample_count,
            offset=sample_offset,
        )
        iq = (samples[0::2] + 1j * samples[1::2]).reshape(
            (RAW_RANGE_BIN_COUNT, int(prt_num)),
            order="F",
        )
        if not np.isfinite(iq.real).all() or not np.isfinite(iq.imag).all():
            raise DataIntegrityError(f"CQ08 raw-DAT I/Q包含非有限值：{path}:{offset}")
        raw_records = tuple(
            RawTargetRecord(
                target_identifier=int(target_words[index * 4]),
                event_index=int(target_words[index * 4 + 1]) - 1,
                range_bin=int(target_words[index * 4 + 2]),
                doppler_bin=int(target_words[index * 4 + 3]),
            )
            for index in range(int(point_count))
        )
        frame = RawEchoFrame(
            raw_frame_index=len(frames),
            frequency_hz=float(frequency_mhz) * 1.0e6,
            cpi_count=int(cpi_count),
            prt_num=int(prt_num),
            prt_seconds=float(prt_raw) * 0.0125e-6,
            target_records=raw_records,
            iq=np.asarray(iq, dtype=np.complex64),
            metadata={"source_path": str(path)},
        )
        canonical_records: list[RawTargetRecord] = []
        for record in raw_records:
            if record.event_index < 0:
                continue
            range_bin, doppler_bin = _locate_raw_dat_target_bin(frame, record)
            canonical_records.append(
                RawTargetRecord(
                    target_identifier=record.target_identifier,
                    event_index=record.event_index,
                    range_bin=range_bin,
                    doppler_bin=doppler_bin,
                )
            )
        frames.append(
            RawEchoFrame(
                raw_frame_index=frame.raw_frame_index,
                frequency_hz=frame.frequency_hz,
                cpi_count=frame.cpi_count,
                prt_num=frame.prt_num,
                prt_seconds=frame.prt_seconds,
                target_records=tuple(canonical_records),
                iq=frame.iq,
                metadata=frame.metadata,
            )
        )
        offset += frame_length
    if not frames or offset != len(payload):
        raise DataFormatError(f"CQ08 raw-DAT未完整解析：{path}")
    return frames


class CQ08RawEchoProvider:
    """按需读取 CQ08 一个 parent 的 raw-DAT frame。"""

    def __init__(
        self,
        project_root: str | Path,
        *,
        source_root: str | Path,
        cache_size: int = 2,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.source_root = Path(source_root).resolve()
        self.raw_root = self.source_root / "原始回波"
        if not self.raw_root.is_dir():
            raise DataReadError(f"CQ08 raw echo目录不存在：{self.raw_root}")
        if isinstance(cache_size, bool) or int(cache_size) <= 0:
            raise ValueError("CQ08 raw provider cache_size 必须是正整数")
        self._cache_size = int(cache_size)
        self._cache: OrderedDict[str, list[RawEchoFrame]] = OrderedDict()

    def _load_track_frames(self, track_id: str) -> list[RawEchoFrame]:
        key = str(track_id)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        candidates = sorted(
            self.raw_root.glob(f"{key}_Label_*.dat"),
            key=lambda path: path.name,
        )
        if len(candidates) != 1:
            raise DataPairingError(
                f"CQ08 raw-DAT无法唯一定位：track={key}, count={len(candidates)}"
            )
        frames = _parse_raw_echo_file(candidates[0])
        self._cache[key] = frames
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return frames

    def read_event_frames(self, event: Any):
        """按事件身份读取 raw-DAT frame。"""

        # A raw target record alone does not make an RD event available.  Keep
        # RD-missing events empty so micro-Doppler cannot leak through the raw
        # provider when the canonical RD view marks the event unavailable.
        if getattr(event, "rd_available", None) is False:
            return []
        track_id = str(getattr(event, "track_id", ""))
        event_index = int(getattr(event, "causal_event_index", -1))
        if not track_id or event_index < 0:
            raise DataFormatError("CQ08 raw provider收到的event身份无效")
        frames = self._load_track_frames(track_id)
        selected = [
            frame
            for frame in frames
            if any(record.event_index == event_index for record in frame.target_records)
        ]
        metadata = getattr(event, "metadata", {})
        if isinstance(metadata, Mapping):
            raw_numbers = metadata.get("rd_mat_frame_numbers") or []
            if not raw_numbers:
                rd_record = metadata.get("rd_record")
                rd_metadata = (
                    rd_record.get("metadata")
                    if isinstance(rd_record, Mapping)
                    else None
                )
                frame_metadata = (
                    rd_metadata.get("frame_metadata")
                    if isinstance(rd_metadata, Mapping)
                    else None
                )
                if isinstance(frame_metadata, list):
                    raw_numbers = [
                        item.get("frame_index")
                        for item in frame_metadata
                        if isinstance(item, Mapping)
                        and item.get("frame_index") is not None
                    ]
            if raw_numbers:
                try:
                    wanted = {int(value) - 1 for value in raw_numbers}
                except (TypeError, ValueError) as exc:
                    raise DataFormatError(
                        "CQ08 raw provider 的 MAT frame 编号无效"
                    ) from exc
                selected = [
                    frame for frame in frames if frame.raw_frame_index in wanted
                ]
        return selected

__all__ = [
    "CQ08Adapter",
    "CQ08_DATASET_NAME",
    "CQ08_SOURCE_NAME",
    "CQ08_LABEL_ID_TO_NAME",
    "CQ08_LABEL_NAME_TO_ID",
    "CQ08_TR_SCHEMA_VERSION",
    "CQ08_RD_SCHEMA_VERSION",
    "resolve_frame_source_path",
]
