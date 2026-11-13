"""总 manifest 和分层 JSONL 索引的加载、校验。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from pathlib import PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Iterator, Mapping

from .contracts import (
    DataFormatError,
    DataIntegrityError,
    DataPairingError,
    DataPathError,
    DataSplitError,
    DatasetView,
    DatasetManifest,
)
from .paths import (
    data_root_name,
    path_resolution_cache,
    resolve_index_path,
    resolve_path,
    resolve_project_path,
    resolve_rd_path,
    resolve_tr_path,
)
from .raw_reader import require_file, sha256_file
from .temporal import validate_temporal_records


MANIFEST_SCHEMA_VERSION = "RADAR-LN3-DATA-MANIFEST-V2"
TR_SCHEMA_VERSION = "RADAR-LN3-CQ08-TR-EVENT-INDEX-V1"
RD_SCHEMA_VERSION = "RADAR-LN3-CQ08-RD-FRAME-INDEX-V1"
PAIRING_SCHEMA_VERSION = "RADAR-LN3-CQ08-TR-RD-PAIRING-V1"
INDEX_FORMAT = "jsonl"
SPLIT_NAMES = frozenset({"train", "validation", "test"})
UNKNOWN_LABEL = "UnknownTarget"
DEFAULT_PROGRESS_INTERVAL = 5


ProgressCallback = Callable[[Mapping[str, Any]], None]


@dataclass
class _ManifestPathCache:
    """单次 manifest 校验内的路径校验结果。

    ``source_paths`` 按完整来源字符串去重（含 ``;`` 分隔项）；``data_paths``
    按 section 记忆已解析的数据路径，避免 1,549 个真实文件被 16 万条记录
    反复解析。
    """

    source_paths: set[str] = field(default_factory=set)
    data_paths: dict[tuple[str, str], Path] = field(default_factory=dict)


def normalise_progress_interval(value: int) -> int:
    """校验进度回调的间隔参数。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("progress_interval 必须是正整数")
    return value


def _relative_manifest_path(value: Any, *, field: str) -> str:
    """校验 manifest 中只能保存稳定的相对路径。"""

    raw = str(value).replace("\\", "/").strip()
    if not raw:
        return ""
    windows = PureWindowsPath(raw)
    posix = PurePosixPath(raw)
    if windows.is_absolute() or windows.drive or posix.is_absolute():
        raise DataPathError(f"manifest {field} 必须使用相对路径：{value!r}")
    if any(part == ".." for part in raw.split("/")):
        raise DataPathError(f"manifest {field} 不得包含 '..'：{value!r}")
    return raw


def _validate_source_path(
    project_root: Path,
    value: str,
    *,
    section: str,
    path_cache: _ManifestPathCache,
) -> None:
    """校验项目内来源或 external 逻辑来源标识。"""

    if not value:
        return
    if value in path_cache.source_paths:
        return
    parts = value.split(";")
    if any(not part for part in parts):
        raise DataPathError(f"{section}.source_path 包含空路径项")
    for item in parts:
        relative = _relative_manifest_path(item, field=f"{section}.source_path")
        if relative.startswith("external/"):
            external_parts = relative.split("/")
            if len(external_parts) < 3 or any(
                not part or part in {".", ".."} for part in external_parts[1:]
            ):
                raise DataPathError(f"{section}.source_path 的 external 逻辑路径无效：{item!r}")
            continue
        resolve_project_path(project_root, relative)
    path_cache.source_paths.add(value)


def normalise_split_name(value: Any) -> str:
    """将历史 val 别名规范化为产品使用的 validation。"""

    text = str(value).strip().lower()
    if text == "val":
        text = "validation"
    if text not in SPLIT_NAMES:
        raise DataSplitError(f"不支持的 split 名称：{value!r}")
    return text


def expected_baseline_training_view(record: Mapping[str, Any]) -> bool:
    """由统一身份字段计算 baseline training view。

    该函数是TR、RD和cache共享的唯一训练视图规则；RD不能通过自身
    sidecar字段重新推断训练资格。
    """

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise DataFormatError("记录 metadata 必须是对象")
    return (
        str(record.get("split_name", "")) == "train"
        and str(record.get("label_name", "")) != UNKNOWN_LABEL
        and not bool(metadata.get("training_excluded", False))
        and not bool(metadata.get("parent_anomalous", False))
    )


def validate_baseline_training_view(
    record: Mapping[str, Any],
    *,
    section: str,
) -> bool:
    """校验记录中的物化视图标记，禁止缺失值默认允许训练。"""

    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise DataFormatError(f"{section}.metadata 必须是对象")
    if "baseline_training_view" not in metadata:
        raise DataFormatError(
            f"{section} 缺少统一 baseline_training_view：{record.get('sample_id')}"
        )
    declared = metadata["baseline_training_view"]
    if not isinstance(declared, bool):
        raise DataFormatError(
            f"{section}.metadata.baseline_training_view 必须是布尔值："
            f"{record.get('sample_id')}"
        )
    expected = expected_baseline_training_view(record)
    if declared != expected:
        raise DataSplitError(
            f"{section} baseline_training_view 与统一视图不一致："
            f"{record.get('sample_id')}"
        )
    return declared


def resolve_manifest_view(
    manifest: DatasetManifest,
    *,
    split_name: str | None = None,
    parent_ids: Iterable[str] | None = None,
    include_unknown: bool = False,
    training_view: bool = True,
    allowed_labels: Iterable[str] | None = None,
    task_name: str = "",
    allow_unknown_head_training: bool = False,
) -> DatasetView:
    """基于TR主记录生成统一的split/task/训练视图。

    视图只从TR事件身份、split、label和异常标记生成；RD记录只负责
    提供同一sample_id的模态内容。这样test、Unknown和异常parent不会
    因为某一模态缺少字段而被误纳入训练。
    """

    normalised_split = (
        None if split_name is None else normalise_split_name(split_name)
    )
    if allow_unknown_head_training and (
        not training_view or normalised_split != "train" or task_name != "l3_head_fit"
    ):
        raise DataSplitError(
            "allow_unknown_head_training 只允许 l3_head_fit 的 train view"
        )
    allowed = (
        None
        if allowed_labels is None
        else tuple(sorted({str(value) for value in allowed_labels}))
    )
    if allowed is not None and not allowed:
        raise DataSplitError("allowed_labels 不能为空")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest.tr_records:
        validate_baseline_training_view(record, section="TR")
        if normalised_split is not None and str(record["split_name"]) != normalised_split:
            continue
        if allowed is not None and str(record["label_name"]) not in allowed:
            continue
        groups[str(record["parent_sample_id"])].append(record)

    requested = None if parent_ids is None else {str(value) for value in parent_ids}
    if requested is not None:
        missing = sorted(requested - set(groups))
        if missing:
            raise DataIntegrityError(
                f"requested parents are unavailable from the manifest: {missing[:3]}"
            )

    selected: dict[str, tuple[str, ...]] = {}
    for parent_id, records in groups.items():
        if requested is not None and parent_id not in requested:
            continue
        labels = {str(row["label_name"]) for row in records}
        splits = {str(row["split_name"]) for row in records}
        if len(labels) != 1 or len(splits) != 1:
            raise DataIntegrityError(f"统一视图 parent 的 label 或 split 冲突：{parent_id}")
        label_name = next(iter(labels))
        parent_split = next(iter(splits))
        if not include_unknown and label_name == UNKNOWN_LABEL:
            continue
        if training_view:
            if parent_split == "test":
                continue
            metadata = [dict(row.get("metadata") or {}) for row in records]
            if any(
                bool(item.get("training_excluded", False))
                or bool(item.get("parent_anomalous", False))
                for item in metadata
            ):
                continue
            if parent_split == "train" and not (
                allow_unknown_head_training
                and label_name == UNKNOWN_LABEL
            ) and not all(
                bool(item["baseline_training_view"])
                for item in metadata
            ):
                continue
        ordered = sorted(
            records,
            key=lambda row: (int(row["causal_event_index"]), str(row["sample_id"])),
        )
        indices = [int(row["causal_event_index"]) for row in ordered]
        if indices != list(range(len(ordered))):
            raise DataIntegrityError(f"统一视图 event index 不连续：{parent_id}")
        selected[parent_id] = tuple(str(row["sample_id"]) for row in ordered)

    selected = dict(sorted(selected.items()))
    if not selected:
        raise ValueError("manifest 数据集选择为空")
    sample_ids = tuple(
        sample_id
        for event_ids in selected.values()
        for sample_id in event_ids
    )
    view_name = "baseline_non_test" if training_view else "audit"
    if allowed is not None:
        view_name = f"{view_name}_task"
    if allow_unknown_head_training:
        view_name = f"{view_name}:openset_head_fit"
    if task_name:
        view_name = f"{view_name}:{task_name}"
    return DatasetView(
        view_name=view_name,
        split_name=normalised_split,
        include_unknown=bool(include_unknown),
        training_view=bool(training_view),
        allowed_labels=allowed,
        sample_ids=sample_ids,
        parent_ids=tuple(selected),
        event_ids_by_parent=selected,
        task_name=str(task_name),
    )


def canonical_json_hash(value: Any) -> str:
    """计算 JSON 结构的稳定 SHA-256。"""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def data_manifest_hash(manifest: DatasetManifest | Mapping[str, Any]) -> str:
    """计算只绑定数据索引的稳定 hash。

    产品 registry、总 manifest 自身 hash 以及事件引用缓存都属于派生登记，
    不应因为新增一个 cache descriptor 就使已有 feature cache 失效。
    """

    payload = (
        manifest.to_dict()
        if isinstance(manifest, DatasetManifest)
        else dict(manifest)
    )
    payload.pop("manifest_sha256", None)
    for key in tuple(payload):
        if key not in {
            "schema_version",
            "dataset",
            "data_root",
            "source_policy",
            "split",
            "TR",
            "RD",
            "pairing",
            "audit",
        }:
            payload.pop(key, None)
    audit = dict(payload.get("audit") or {})
    audit.pop("cache", None)
    # Product registrations are derived metadata.  L3 registration state now
    # lives only in ``product.scope``; keep dropping legacy ``audit.scope`` so
    # old manifests do not change the TR/RD data identity or invalidate caches.
    audit.pop("scope", None)
    payload["audit"] = audit
    return canonical_json_hash(payload)


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if not isinstance(value, Mapping):
        raise DataFormatError(f"manifest section {name!r} 必须是对象")
    return value


def _required(record: Mapping[str, Any], fields: Iterable[str], *, section: str) -> None:
    missing = sorted(field for field in fields if field not in record)
    if missing:
        raise DataFormatError(f"{section} 记录缺少字段：{missing}")


def _hash_value(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise DataFormatError(f"{field} 必须是 SHA-256 十六进制字符串或 null")
    if any(character not in "0123456789abcdefABCDEF" for character in value):
        raise DataFormatError(f"{field} 必须是 SHA-256 十六进制字符串或 null")
    return value


def _size_value(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise DataFormatError(f"{field} 必须是非负整数或 null")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DataFormatError(f"{field} 必须是非负整数或 null") from exc
    if result < 0:
        raise DataFormatError(f"{field} 必须是非负整数或 null")
    return result


def _validate_record_identity(
    record: Mapping[str, Any], *, section: str, frame_record: bool = False
) -> None:
    fields = (
        "dataset_name", "source_name", "sample_id", "parent_sample_id", "track_id",
        "causal_event_index", "label_id", "label_name", "split_name", "available",
        "source_path", "data_path", "file_size", "sha256", "metadata",
    )
    _required(record, fields, section=section)
    if frame_record:
        _required(record, ("record_type", "frame_index"), section=section)
        if record.get("record_type") != "frame":
            raise DataFormatError(f"{section} 只允许 record_type=frame")
        frame_index = record["frame_index"]
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise DataFormatError(f"{section}.frame_index 必须是非负整数")
    for field in (
        "dataset_name", "source_name", "sample_id", "parent_sample_id", "track_id",
        "label_name", "split_name", "source_path", "data_path",
    ):
        if not isinstance(record[field], str):
            raise DataFormatError(f"{section}.{field} 必须是字符串")
    if not str(record["sample_id"]).strip() or not str(record["parent_sample_id"]).strip():
        raise DataFormatError(f"{section} 记录的身份字段不能为空")
    if not str(record["dataset_name"]).strip() or not str(record["source_name"]).strip():
        raise DataFormatError(f"{section} 记录的来源身份不能为空")
    if not str(record["label_name"]).strip():
        raise DataFormatError(f"{section}.label_name 不能为空")
    if not isinstance(record["label_id"], (int, type(None))) or isinstance(record["label_id"], bool):
        raise DataFormatError(f"{section}.label_id 必须是整数或 null")
    event_index = record["causal_event_index"]
    if isinstance(event_index, bool) or not isinstance(event_index, int) or event_index < 0:
        raise DataFormatError(f"{section}.causal_event_index 必须是非负整数")
    if str(record["split_name"]) != normalise_split_name(record["split_name"]):
        raise DataSplitError("manifest split_name 必须使用 validation，而不是 val")
    if not isinstance(record["available"], bool):
        raise DataFormatError(f"{section}.available 必须是布尔值")
    if not isinstance(record["metadata"], Mapping):
        raise DataFormatError(f"{section}.metadata 必须是对象")
    _hash_value(record["sha256"], field=f"{section}.sha256")
    _size_value(record["file_size"], field=f"{section}.file_size")


def _validate_data_reference(
    project_root: Path,
    record: Mapping[str, Any],
    *,
    section: str,
    verify_files: bool,
    artifact_cache: dict[str, tuple[int, str]],
    path_cache: _ManifestPathCache,
) -> None:
    data_path = _relative_manifest_path(record["data_path"], field=f"{section}.data_path")
    if bool(record["available"]) and not data_path:
        raise DataIntegrityError(f"可用 {section} 记录缺少 data_path：{record['sample_id']}")
    if not data_path:
        return

    cache_key = (section, data_path)
    path = path_cache.data_paths.get(cache_key)
    if path is None:
        resolver = resolve_tr_path if section == "TR" else resolve_rd_path
        path = resolver(project_root, data_path, must_exist=verify_files)
        path_cache.data_paths[cache_key] = path
    if not verify_files:
        return
    observed_key = str(path)
    observed = artifact_cache.get(observed_key)
    if observed is None:
        observed = (int(path.stat().st_size), sha256_file(path))
        artifact_cache[observed_key] = observed
    declared_size = _size_value(record["file_size"], field=f"{section}.file_size")
    if declared_size is not None and declared_size != observed[0]:
        raise DataIntegrityError(f"数据文件大小不一致：{path}")
    declared_hash = _hash_value(record["sha256"], field=f"{section}.sha256")
    if declared_hash and declared_hash != observed[1]:
        raise DataIntegrityError(f"数据文件 hash 不一致：{path}")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    require_file(path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DataFormatError(f"JSONL 索引格式无效：{path}:{line_number}") from exc
                if not isinstance(value, Mapping):
                    raise DataFormatError(f"JSONL 索引记录必须是对象：{path}:{line_number}")
                yield dict(value)
    except UnicodeError as exc:
        raise DataFormatError(f"JSONL 索引编码无效：{path}") from exc
    except OSError as exc:
        raise DataFormatError(f"无法读取 JSONL 索引：{path}") from exc


def _descriptor_parts(section: Mapping[str, Any], *, name: str, schema_version: str) -> list[dict[str, Any]]:
    if section.get("schema_version") != schema_version:
        raise DataFormatError(f"{name} schema_version 不匹配")
    if section.get("format") != INDEX_FORMAT:
        raise DataFormatError(f"{name}.format 必须是 jsonl")
    record_count = section.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count < 0:
        raise DataFormatError(f"{name}.record_count 必须是非负整数")
    parts = section.get("parts")
    if not isinstance(parts, list) or any(not isinstance(item, Mapping) for item in parts):
        raise DataFormatError(f"{name}.parts 必须是对象列表")
    if record_count > 0 and not parts:
        raise DataFormatError(f"{name} 有记录但没有 sidecar part")
    result: list[dict[str, Any]] = []
    total = 0
    for position, raw_part in enumerate(parts):
        part = dict(raw_part)
        _required(part, ("path", "record_count", "file_size", "sha256"), section=f"{name}.parts[{position}]")
        part["path"] = _relative_manifest_path(part["path"], field=f"{name}.parts[{position}].path")
        if not part["path"]:
            raise DataPathError(f"{name}.parts[{position}].path 不能为空")
        if not part["path"].lower().endswith(".jsonl"):
            raise DataFormatError(f"{name}.parts[{position}].path 必须指向 JSONL 文件")
        count = part["record_count"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise DataFormatError(f"{name}.parts[{position}].record_count 必须是非负整数")
        _size_value(part["file_size"], field=f"{name}.parts[{position}].file_size")
        _hash_value(part["sha256"], field=f"{name}.parts[{position}].sha256")
        total += count
        result.append(part)
    if total != record_count:
        raise DataIntegrityError(f"{name}.record_count={record_count} 与 parts 总数 {total} 不一致")
    return result


def _load_index_parts(
    project_root: Path,
    section: Mapping[str, Any],
    *,
    name: str,
    schema_version: str,
    verify_files: bool,
) -> list[dict[str, Any]]:
    parts = _descriptor_parts(section, name=name, schema_version=schema_version)
    records: list[dict[str, Any]] = []
    for position, part in enumerate(parts):
        path = resolve_index_path(project_root, part["path"], section=name, must_exist=True)
        if not path.is_file():
            raise DataPathError(f"{name} sidecar 不存在：{path}")
        declared_size = _size_value(part["file_size"], field=f"{name}.parts[{position}].file_size")
        if verify_files and declared_size is not None and int(path.stat().st_size) != declared_size:
            raise DataIntegrityError(f"{name} sidecar 文件大小不一致：{path}")
        declared_hash = _hash_value(part["sha256"], field=f"{name}.parts[{position}].sha256")
        if verify_files and declared_hash and sha256_file(path) != declared_hash:
            raise DataIntegrityError(f"{name} sidecar hash 不一致：{path}")
        part_records = list(_iter_jsonl(path))
        if len(part_records) != int(part["record_count"]):
            raise DataIntegrityError(f"{name} sidecar record_count 不一致：{path}")
        records.extend(part_records)
    return records


def _validate_event_records(
    records: list[dict[str, Any]],
    *,
    section: str,
    project_root: Path,
    verify_files: bool,
    artifact_cache: dict[str, tuple[int, str]],
    path_cache: _ManifestPathCache,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> None:
    progress_interval = normalise_progress_interval(progress_interval)
    seen: set[str] = set()
    records_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        records_by_parent[str(record.get("parent_sample_id", ""))].append(record)
    parent_ids = sorted(records_by_parent)
    for processed_parent_count, parent_id in enumerate(parent_ids, start=1):
        for record in records_by_parent[parent_id]:
            _validate_record_identity(record, section=section)
            _validate_source_path(
                project_root,
                str(record["source_path"]),
                section=section,
                path_cache=path_cache,
            )
            _validate_data_reference(
                project_root,
                record,
                section=section,
                verify_files=verify_files,
                artifact_cache=artifact_cache,
                path_cache=path_cache,
            )
            sample_id = str(record["sample_id"])
            if sample_id in seen:
                raise DataIntegrityError(f"{section} 存在重复 sample_id：{sample_id}")
            seen.add(sample_id)
        if progress_callback is not None and (
            processed_parent_count % progress_interval == 0
            or processed_parent_count == len(parent_ids)
        ):
            progress_callback(
                {
                    "stage": "validation_progress",
                    "modality": section,
                    "processed": processed_parent_count,
                    "total": len(parent_ids),
                    "parent_number": processed_parent_count,
                    "parent_count": len(parent_ids),
                    "parent_sample_id": parent_id,
                    "event_count": len(records_by_parent[parent_id]),
                    "record_count": len(records),
                }
            )
    if section == "TR":
        validate_temporal_records(records)


def _aggregate_rd_frames(
    frame_records: list[dict[str, Any]],
    tr_records: list[dict[str, Any]],
    *,
    project_root: Path,
    verify_files: bool,
    artifact_cache: dict[str, tuple[int, str]],
    path_cache: _ManifestPathCache,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> list[dict[str, Any]]:
    progress_interval = normalise_progress_interval(progress_interval)
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_frames: set[tuple[str, int]] = set()
    tr_by_id = {str(record["sample_id"]): record for record in tr_records}
    tr_parent_ids = sorted(
        {str(record["parent_sample_id"]) for record in tr_records}
    )
    tr_event_count_by_parent: dict[str, int] = defaultdict(int)
    for record in tr_records:
        tr_event_count_by_parent[str(record["parent_sample_id"])] += 1
    frames_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in frame_records:
        frames_by_parent[str(record.get("parent_sample_id", ""))].append(record)
    for processed_parent_count, parent_id in enumerate(tr_parent_ids, start=1):
        parent_frames = frames_by_parent.get(parent_id, [])
        for record in parent_frames:
            _validate_record_identity(record, section="RD", frame_record=True)
            _validate_source_path(
                project_root,
                str(record["source_path"]),
                section="RD",
                path_cache=path_cache,
            )
            _validate_data_reference(
                project_root,
                record,
                section="RD",
                verify_files=verify_files,
                artifact_cache=artifact_cache,
                path_cache=path_cache,
            )
            key = (str(record["sample_id"]), int(record["frame_index"]))
            if key in seen_frames:
                raise DataIntegrityError(f"RD 存在重复 frame：{key}")
            seen_frames.add(key)
            by_sample[key[0]].append(record)
        if progress_callback is not None and (
            processed_parent_count % progress_interval == 0
            or processed_parent_count == len(tr_parent_ids)
        ):
            progress_callback(
                {
                    "stage": "validation_progress",
                    "modality": "RD",
                    "processed": processed_parent_count,
                    "total": len(tr_parent_ids),
                    "parent_number": processed_parent_count,
                    "parent_count": len(tr_parent_ids),
                    "parent_sample_id": parent_id,
                    "event_count": tr_event_count_by_parent[parent_id],
                    "rd_frame_count": len(parent_frames),
                    "record_count": len(frame_records),
                }
            )
    unknown = sorted(set(frames_by_parent) - set(tr_parent_ids))
    if unknown:
        raise DataPairingError(f"RD sidecar 引用了未知 parent_sample_id：{unknown[:3]}")
    unknown_samples = sorted(set(by_sample) - set(tr_by_id))
    if unknown_samples:
        raise DataPairingError(f"RD sidecar 引用了未知 sample_id：{unknown_samples[:3]}")
    result: list[dict[str, Any]] = []

    common_fields = (
        "dataset_name", "source_name", "sample_id", "parent_sample_id", "track_id",
        "causal_event_index", "label_id", "label_name", "split_name",
    )
    for tr_record in tr_records:
        sample_id = str(tr_record["sample_id"])
        frames = sorted(by_sample.get(sample_id, []), key=lambda item: int(item["frame_index"]))
        if not frames:
            result.append({
                **{field: tr_record[field] for field in common_fields},
                "available": False, "source_path": "", "data_path": "",
                "file_size": None, "sha256": None,
                "metadata": {
                    "frame_indices": [],
                    "frame_metadata": [],
                    "frame_source_paths": [],
                    "rd_frame_count": 0,
                    "frame_cardinality": "zero_frames",
                    "baseline_training_view": validate_baseline_training_view(
                        tr_record, section="TR"
                    ),
                    "parent_anomalous": bool(
                        tr_record.get("metadata", {}).get("parent_anomalous", False)
                    ),
                    "training_excluded": bool(
                        tr_record.get("metadata", {}).get("training_excluded", False)
                    ),
                    "tr_time_order_status": str(
                        tr_record.get("metadata", {}).get("tr_time_order_status", "unknown")
                    ),
                },
            })
            continue
        first = frames[0]
        for frame in frames[1:]:
            for field in common_fields:
                if frame[field] != first[field]:
                    raise DataPairingError(f"RD frame identity differs for {sample_id}：{field}")
            if frame["data_path"] != first["data_path"]:
                raise DataPairingError(f"RD frame data_path differs for {sample_id}")
        frame_indices = [int(frame["frame_index"]) for frame in frames]
        result.append({
            **{field: first[field] for field in common_fields},
            "available": True,
            "source_path": ";".join(str(frame["source_path"]) for frame in frames),
            "data_path": first["data_path"],
            "file_size": first["file_size"],
            "sha256": first["sha256"],
            "metadata": {
                "frame_indices": frame_indices,
                "rd_frame_count": len(frames),
                "frame_metadata": [dict(frame.get("metadata") or {}) for frame in frames],
                "frame_source_paths": [str(frame["source_path"]) for frame in frames],
                "frame_cardinality": "many_frames_per_event",
                "frame_order": "frame_index_ascending",
                "baseline_training_view": validate_baseline_training_view(
                    tr_record, section="TR"
                ),
                "parent_anomalous": bool(
                    tr_record.get("metadata", {}).get("parent_anomalous", False)
                ),
                "training_excluded": bool(
                    tr_record.get("metadata", {}).get("training_excluded", False)
                ),
                "tr_time_order_status": str(
                    tr_record.get("metadata", {}).get("tr_time_order_status", "unknown")
                ),
            },
        })
    return result


def _validate_relationships(
    tr_records: list[dict[str, Any]],
    rd_records: list[dict[str, Any]],
    *,
    source_policy: Mapping[str, Any],
) -> None:
    if len(tr_records) != len(rd_records):
        raise DataPairingError("TR 和 RD 事件数量不一致")
    tr_by_id = {str(row["sample_id"]): row for row in tr_records}
    rd_by_id = {str(row["sample_id"]): row for row in rd_records}
    if set(tr_by_id) != set(rd_by_id):
        raise DataPairingError("TR/RD 事件 sample_id 覆盖范围不一致")
    matching_fields = (
        "dataset_name", "source_name", "parent_sample_id", "track_id",
        "causal_event_index", "label_id", "label_name", "split_name",
    )
    label_to_id: dict[str, int | None] = {}
    id_to_label: dict[int, str] = {}
    for section, records in (("TR", tr_records), ("RD", rd_records)):
        for record in records:
            sample_id = str(record["sample_id"])
            other = rd_by_id[sample_id] if section == "TR" else tr_by_id[sample_id]
            validate_baseline_training_view(record, section=section)
            for field in matching_fields:
                if record[field] != other[field]:
                    raise DataPairingError(f"TR/RD 字段不一致：{sample_id}:{field}")
            label_name = str(record["label_name"])
            label_id = record["label_id"]
            old_id = label_to_id.setdefault(label_name, label_id)
            if old_id != label_id:
                raise DataIntegrityError(f"label_name 对应多个 label_id：{label_name}")
            if label_id is not None:
                old_name = id_to_label.setdefault(int(label_id), label_name)
                if old_name != label_name:
                    raise DataIntegrityError(f"label_id 对应多个 label_name：{label_id}")
    for sample_id, tr in tr_by_id.items():
        rd = rd_by_id[sample_id]
        tr_anomalous = bool(tr.get("metadata", {}).get("parent_anomalous", False))
        rd_anomalous = bool(rd.get("metadata", {}).get("parent_anomalous", False))
        if tr_anomalous != rd_anomalous:
            raise DataPairingError(f"TR/RD 异常 parent 标记不一致：{sample_id}")
        tr_excluded = bool(tr.get("metadata", {}).get("training_excluded", False))
        rd_excluded = bool(rd.get("metadata", {}).get("training_excluded", False))
        if tr_excluded != rd_excluded:
            raise DataPairingError(f"TR/RD training_excluded 标记不一致：{sample_id}")
    for section, records in (("TR", tr_records), ("RD", rd_records)):
        parent_splits: dict[str, set[str]] = defaultdict(set)
        parent_events: dict[str, list[int]] = defaultdict(list)
        parent_labels: dict[str, set[str]] = defaultdict(set)
        for record in records:
            parent = str(record["parent_sample_id"])
            parent_splits[parent].add(str(record["split_name"]))
            parent_events[parent].append(int(record["causal_event_index"]))
            parent_labels[parent].add(str(record["label_name"]))
        if any(len(values) != 1 for values in parent_splits.values()):
            raise DataSplitError(f"{section} parent 跨越 split 边界")
        for parent, indices in parent_events.items():
            if sorted(indices) != list(range(len(indices))):
                raise DataIntegrityError(f"{section} event index 不连续：{parent}")
            if len(parent_labels[parent]) != 1:
                raise DataIntegrityError(f"{section} parent 标签不一致：{parent}")
    tr_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rd_by_parent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in tr_records:
        tr_by_parent[str(record["parent_sample_id"])].append(record)
    for record in rd_records:
        rd_by_parent[str(record["parent_sample_id"])].append(record)
    for parent_id, records in tr_by_parent.items():
        anomalous = any(
            bool(row.get("metadata", {}).get("parent_anomalous", False))
            for row in records
        )
        excluded = any(
            bool(row.get("metadata", {}).get("training_excluded", False))
            for row in records
        )
        if anomalous and not excluded:
            raise DataIntegrityError(f"TR 异常 parent 未同步标记 training_excluded：{parent_id}")
        if excluded and any(
            bool(row["metadata"]["baseline_training_view"])
            for row in records
        ):
            raise DataIntegrityError(f"TR 异常 parent 仍出现在 baseline training view：{parent_id}")
        rd_records_for_parent = rd_by_parent.get(parent_id, [])
        if anomalous and any(
            not bool(row.get("metadata", {}).get("parent_anomalous", False))
            or not bool(row.get("metadata", {}).get("training_excluded", False))
            for row in rd_records_for_parent
        ):
            raise DataIntegrityError(f"RD 未同步继承异常 parent 标记：{parent_id}")
    if source_policy.get("unknown_training_exposure") is not False:
        raise DataSplitError("Unknown 不得进入 baseline training view")
    baseline_training = [
        row
        for row in tr_records
        if str(row["split_name"]) == "train"
        and bool(row["metadata"]["baseline_training_view"])
    ]
    if any(str(row["label_name"]) == UNKNOWN_LABEL for row in baseline_training):
        raise DataSplitError("Unknown 出现在 baseline training view")
    for section, records in (("TR", tr_records), ("RD", rd_records)):
        if any(
            str(row["split_name"]) == "test"
            and bool(row["metadata"]["baseline_training_view"])
            for row in records
        ):
            raise DataSplitError(f"test 数据出现在 {section} training view")


def _cached_path_resolution(function: Callable[..., Any]) -> Callable[..., Any]:
    """在一次 manifest 校验内复用路径解析结果；校验全程只读。"""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with path_resolution_cache():
            return function(*args, **kwargs)

    return wrapper


@_cached_path_resolution
def validate_manifest(
    manifest: Mapping[str, Any] | DatasetManifest,
    project_root: str | Path,
    *,
    verify_files: bool = True,
    allow_unmaterialized: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> DatasetManifest:
    """校验总 manifest、sidecar、split 边界和数据 hash。"""

    progress_interval = normalise_progress_interval(progress_interval)
    payload = manifest.to_dict() if isinstance(manifest, DatasetManifest) else dict(manifest)
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise DataFormatError("manifest schema_version 不匹配")
    declared_hash = payload.get("manifest_sha256")
    if declared_hash is not None:
        _hash_value(declared_hash, field="manifest_sha256")
        body = dict(payload)
        body.pop("manifest_sha256", None)
        if canonical_json_hash(body) != declared_hash:
            raise DataIntegrityError("manifest 自身 hash 不一致")
    dataset = str(payload.get("dataset") or "").strip()
    if not dataset:
        raise DataFormatError("manifest dataset 不能为空")
    data_root = str(payload.get("data_root") or "").replace("\\", "/")
    if data_root != data_root_name():
        raise DataPathError(f"不支持的 data_root：{data_root!r}")
    source_policy = payload.get("source_policy")
    split_policy = payload.get("split")
    if not isinstance(source_policy, Mapping) or not isinstance(split_policy, Mapping):
        raise DataFormatError("manifest source_policy 和 split 必须是对象")
    if source_policy.get("frame_level_split_forbidden") is not True:
        raise DataSplitError("必须禁止 frame-level split")
    if source_policy.get("parent_split_boundary") is not True:
        raise DataSplitError("必须启用 parent-level split 边界")
    if source_policy.get("unknown_training_exposure") is not False:
        raise DataSplitError("Unknown 不得进入 baseline training view")
    if split_policy.get("group_unit") != "parent_sample_id":
        raise DataSplitError("split group_unit 必须是 parent_sample_id")
    if split_policy.get("split_field") != "split_name":
        raise DataSplitError("split_field 必须是 split_name")
    tr_section = _section(payload, "TR")
    rd_section = _section(payload, "RD")
    pair_section = _section(payload, "pairing")
    if pair_section.get("schema_version") != PAIRING_SCHEMA_VERSION:
        raise DataFormatError("pairing schema_version 不匹配")
    if pair_section.get("strategy") != "sample_id_and_causal_event_index":
        raise DataPairingError("不支持的 pairing strategy")
    root = resolve_path(project_root)
    tr_records = _load_index_parts(root, tr_section, name="TR", schema_version=TR_SCHEMA_VERSION, verify_files=verify_files)
    rd_frames = _load_index_parts(root, rd_section, name="RD", schema_version=RD_SCHEMA_VERSION, verify_files=verify_files)
    tr_parent_count = len({str(row.get("parent_sample_id", "")) for row in tr_records})
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "validation_start",
                "modality": "TR",
                "processed": 0,
                "total": tr_parent_count,
                "parent_count": tr_parent_count,
                "record_count": len(tr_records),
                "verify_files": bool(verify_files),
            }
        )
    if tr_records:
        artifact_cache: dict[str, tuple[int, str]] = {}
        path_cache = _ManifestPathCache()
        _validate_event_records(
            tr_records,
            section="TR",
            project_root=root,
            verify_files=verify_files,
            artifact_cache=artifact_cache,
            path_cache=path_cache,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "validation_complete",
                    "modality": "TR",
                    "processed": tr_parent_count,
                    "total": tr_parent_count,
                    "parent_count": tr_parent_count,
                    "record_count": len(tr_records),
                }
            )
            progress_callback(
                {
                    "stage": "validation_start",
                    "modality": "RD",
                    "processed": 0,
                    "total": tr_parent_count,
                    "parent_count": tr_parent_count,
                    "record_count": len(rd_frames),
                    "verify_files": bool(verify_files),
                }
            )
        rd_records = _aggregate_rd_frames(
            rd_frames,
            tr_records,
            project_root=root,
            verify_files=verify_files,
            artifact_cache=artifact_cache,
            path_cache=path_cache,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
        # 聚合记录的 source_path 只由上面逐帧校验过的路径拼接而成，
        # 不再二次拆分校验（P0-3）；data_path 由 path_cache 命中。
        for record in rd_records:
            _validate_record_identity(record, section="RD")
            _validate_data_reference(
                root,
                record,
                section="RD",
                verify_files=verify_files,
                artifact_cache=artifact_cache,
                path_cache=path_cache,
            )
        _validate_relationships(tr_records, rd_records, source_policy=source_policy)
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "validation_complete",
                    "modality": "RD",
                    "processed": tr_parent_count,
                    "total": tr_parent_count,
                    "parent_count": tr_parent_count,
                    "record_count": len(rd_frames),
                }
            )
    else:
        rd_records = []
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "validation_complete",
                    "modality": "TR",
                    "processed": 0,
                    "total": 0,
                    "parent_count": 0,
                    "record_count": 0,
                }
            )
            progress_callback(
                {
                    "stage": "validation_start",
                    "modality": "RD",
                    "processed": 0,
                    "total": 0,
                    "parent_count": 0,
                    "record_count": len(rd_frames),
                    "verify_files": bool(verify_files),
                }
            )
            progress_callback(
                {
                    "stage": "validation_complete",
                    "modality": "RD",
                    "processed": 0,
                    "total": 0,
                    "parent_count": 0,
                    "record_count": 0,
                }
            )
    expected_events = int(pair_section.get("paired_event_count", len(tr_records)))
    if expected_events != len(tr_records) and (tr_records or verify_files):
        raise DataIntegrityError("pairing.paired_event_count 与实际事件数量不一致")
    if not allow_unmaterialized and not tr_records:
        raise DataIntegrityError("manifest 尚未生成 sidecar 记录")
    audit = dict(payload.get("audit") or {})
    audit.setdefault("materialized", bool(tr_records))
    registry_keys = {
        key: value
        for key, value in payload.items()
        if key
        not in {
            "schema_version",
            "dataset",
            "data_root",
            "source_policy",
            "split",
            "TR",
            "RD",
            "pairing",
            "audit",
            "manifest_sha256",
        }
    }
    return DatasetManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset=dataset,
        data_root=data_root,
        source_policy=dict(source_policy),
        split=dict(split_policy),
        tr_records=tr_records,
        rd_records=rd_records,
        tr_index=dict(tr_section),
        rd_index=dict(rd_section),
        pairing=dict(pair_section),
        audit=audit,
        registries=registry_keys,
        manifest_sha256=declared_hash,
    )


def load_manifest(
    path: str | Path,
    project_root: str | Path,
    *,
    verify_files: bool = True,
    allow_unmaterialized: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> DatasetManifest:
    """读取并校验唯一总 manifest。"""

    target = resolve_project_path(project_root, path, must_exist=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataFormatError(f"无法读取 manifest：{target}") from exc
    if not isinstance(payload, Mapping):
        raise DataFormatError("manifest 根节点必须是对象")
    return validate_manifest(
        payload,
        project_root,
        verify_files=verify_files,
        allow_unmaterialized=allow_unmaterialized,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )


__all__ = [
    "MANIFEST_SCHEMA_VERSION", "TR_SCHEMA_VERSION", "RD_SCHEMA_VERSION",
    "PAIRING_SCHEMA_VERSION", "INDEX_FORMAT", "SPLIT_NAMES", "DEFAULT_PROGRESS_INTERVAL",
    "ProgressCallback", "normalise_progress_interval", "canonical_json_hash", "data_manifest_hash",
    "normalise_split_name", "expected_baseline_training_view",
    "validate_baseline_training_view", "resolve_manifest_view",
    "load_manifest", "validate_manifest",
]
