"""通用事件缓存注册表的构建和加载。"""

from __future__ import annotations

import json
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.data.dataset import ParentSequenceDataset
from radar_ln3.data.manifest import (
    DEFAULT_PROGRESS_INTERVAL,
    data_manifest_hash,
    load_manifest,
    validate_baseline_training_view,
    validate_manifest,
)
from radar_ln3.data.paths import ensure_path_inside_root, relative_project_path

from .contracts import (
    CACHE_FORMAT,
    CACHE_KIND,
    CACHE_SCHEMA_VERSION,
    CacheFormatError,
    CacheIntegrityError,
    CachePathError,
)


DEFAULT_CACHE_RELATIVE_PATH = Path("data/cq08/cache/event-source-cache.jsonl")


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _normalise_progress_interval(value: int) -> int:
    """校验缓存构建进度的回调间隔。"""

    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("progress_interval 必须是正整数")
    return value


def _cache_basis(manifest: DatasetManifest) -> str:
    """计算不包含缓存自身注册项的 manifest 来源 hash。"""

    return data_manifest_hash(manifest)


def _cache_basis_candidates(manifest: DatasetManifest) -> tuple[str, ...]:
    """返回当前 manifest 及其显式父 manifest 的缓存来源 hash。

    L3 audit manifest 是在 calibration 后从正式 manifest 派生的视图：它
    必须拥有新的 data manifest hash 用于审计隔离，但仍复用同一份事件引用
    缓存。只有 manifest 内显式登记的 ``audit.l3_audit.source_manifest_sha256``
    才能作为事件缓存的父来源 hash，不能对任意 hash mismatch 放宽校验。
    """

    current = _cache_basis(manifest)
    candidates = [current]
    audit = manifest.audit if isinstance(manifest.audit, Mapping) else {}
    l3_audit = audit.get("l3_audit")
    if isinstance(l3_audit, Mapping):
        source = str(l3_audit.get("source_manifest_sha256") or "").lower()
        if len(source) == 64 and not set(source) - set("0123456789abcdef"):
            candidates.append(source)
    return tuple(dict.fromkeys(candidates))


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return
        if not overwrite:
            raise FileExistsError(f"拒绝覆盖内容不同的缓存：{path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.is_file() and path.read_bytes() == payload:
                    return
                raise FileExistsError(f"拒绝覆盖竞态产生的缓存：{path}")
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_cache_path(project_root: Path, cache_path: str | Path) -> Path:
    raw = Path(str(cache_path).replace("\\", "/"))
    target = raw.resolve() if raw.is_absolute() else (project_root / raw).resolve()
    try:
        ensure_path_inside_root(target, project_root)
        ensure_path_inside_root(target, (project_root / "data" / "cq08" / "cache").resolve())
    except ValueError as exc:
        raise CachePathError(f"缓存路径越过允许边界：{target}") from exc
    except Exception as exc:
        if isinstance(exc, CachePathError):
            raise
        raise CachePathError(f"缓存路径无效：{target}") from exc
    if target.suffix.lower() != ".jsonl":
        raise CachePathError(f"缓存路径必须是 JSONL：{target}")
    return target


def _event_cache_rows(
    manifest: DatasetManifest,
    *,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> list[dict[str, Any]]:
    progress_interval = _normalise_progress_interval(progress_interval)
    rd_by_id = {str(row["sample_id"]): row for row in manifest.rd_records}
    rows: list[dict[str, Any]] = []
    records_by_parent: dict[str, list[dict[str, Any]]] = {}
    for tr in manifest.tr_records:
        records_by_parent.setdefault(str(tr["parent_sample_id"]), []).append(tr)
    parent_ids = sorted(records_by_parent)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "cache_start",
                "processed": 0,
                "total": len(parent_ids),
                "record_count": len(manifest.tr_records),
            }
        )
    for processed_parent_count, parent_id in enumerate(parent_ids, start=1):
        for tr in records_by_parent[parent_id]:
            sample_id = str(tr["sample_id"])
            rd = rd_by_id.get(sample_id)
            rd_metadata = dict((rd or {}).get("metadata") or {})
            tr_metadata = dict(tr.get("metadata") or {})
            tr_timing = dict(tr_metadata.get("tr_timing") or {})
            frame_metadata = rd_metadata.get("frame_metadata") or []
            rd_frame_times = [
                float(item["rd_frame_time_seconds"])
                for item in frame_metadata
                if isinstance(item, Mapping) and item.get("rd_frame_time_seconds") is not None
            ]
            rows.append(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "sample_id": sample_id,
                    "parent_sample_id": str(tr["parent_sample_id"]),
                    "track_id": str(tr["track_id"]),
                    "causal_event_index": int(tr["causal_event_index"]),
                    "label_id": tr["label_id"],
                    "label_name": str(tr["label_name"]),
                    "split_name": str(tr["split_name"]),
                    "baseline_training_view": validate_baseline_training_view(
                        tr, section="TR"
                    ),
                    "tr_available": bool(tr["available"]),
                    "tr_data_path": str(tr.get("data_path") or ""),
                    "tr_file_size": tr.get("file_size"),
                    "tr_sha256": tr.get("sha256"),
                    "tr_time_seconds": tr_timing.get("time_seconds"),
                    "tr_dt_seconds": tr_timing.get("dt_seconds"),
                    "tr_dt_valid": tr_timing.get("dt_valid"),
                    "tr_dt_reason": tr_timing.get("dt_reason"),
                    "parent_anomalous": bool(tr_metadata.get("parent_anomalous", False)),
                    "training_excluded": bool(tr_metadata.get("training_excluded", False)),
                    "tr_dt_is_negative": bool(tr_metadata.get("tr_dt_is_negative", False)),
                    "tr_time_order_status": str(
                        tr_metadata.get("tr_time_order_status", "unknown")
                    ),
                    "rd_available": bool(rd and rd.get("available")),
                    "rd_data_path": str((rd or {}).get("data_path") or ""),
                    "rd_file_size": (rd or {}).get("file_size"),
                    "rd_sha256": (rd or {}).get("sha256"),
                    "rd_frame_indices": [int(value) for value in rd_metadata.get("frame_indices", [])],
                    "rd_frame_count": int(rd_metadata.get("rd_frame_count", 0)),
                    "rd_frame_times": rd_frame_times,
                    "rd_time_semantics": str(
                        rd_metadata.get("time_semantics", "parent_relative_cpi_elapsed_seconds")
                    ),
                }
            )
        if progress_callback is not None and (
            processed_parent_count % progress_interval == 0
            or processed_parent_count == len(parent_ids)
        ):
            progress_callback(
                {
                    "stage": "cache_progress",
                    "processed": processed_parent_count,
                    "total": len(parent_ids),
                    "parent_sample_id": parent_id,
                    "event_count": len(records_by_parent[parent_id]),
                    "record_count": len(manifest.tr_records),
                }
            )
    rows.sort(key=lambda value: (value["parent_sample_id"], value["causal_event_index"]))
    return rows


def _validate_rows(rows: list[dict[str, Any]], manifest: DatasetManifest) -> None:
    expected = {str(row["sample_id"]) for row in manifest.tr_records}
    tr_by_id = {str(row["sample_id"]): row for row in manifest.tr_records}
    actual = {str(row.get("sample_id", "")) for row in rows}
    if len(rows) != len(actual):
        raise CacheIntegrityError("事件缓存包含重复 sample_id")
    if actual != expected:
        raise CacheIntegrityError("事件缓存与 TR event 索引覆盖范围不一致")
    for row in rows:
        required = {
            "schema_version", "sample_id", "parent_sample_id", "track_id",
            "causal_event_index", "split_name", "tr_available", "rd_available",
            "baseline_training_view",
            "tr_data_path", "rd_data_path", "rd_frame_indices",
            "tr_time_seconds", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason",
            "rd_frame_count", "rd_frame_times", "rd_time_semantics",
            "parent_anomalous", "training_excluded", "tr_dt_is_negative",
            "tr_time_order_status",
        }
        missing = sorted(required - set(row))
        if missing:
            raise CacheFormatError(f"事件缓存记录缺少字段：{missing}")
        if row["schema_version"] != CACHE_SCHEMA_VERSION:
            raise CacheFormatError("事件缓存 schema_version 不匹配")
        if not isinstance(row["baseline_training_view"], bool):
            raise CacheFormatError("事件缓存 baseline_training_view 必须是布尔值")
        expected_view = validate_baseline_training_view(
            tr_by_id[str(row["sample_id"])], section="TR"
        )
        if row["baseline_training_view"] != expected_view:
            raise CacheIntegrityError(
                f"事件缓存 baseline_training_view 与TR统一视图不一致："
                f"{row['sample_id']}"
            )
        if not isinstance(row["rd_frame_indices"], list):
            raise CacheFormatError("事件缓存 rd_frame_indices 必须是列表")
        if row["tr_dt_valid"] is not None and not isinstance(row["tr_dt_valid"], bool):
            raise CacheFormatError("事件缓存 tr_dt_valid 必须是布尔值或 null")
        if not isinstance(row["rd_frame_count"], int) or row["rd_frame_count"] < 0:
            raise CacheFormatError("事件缓存 rd_frame_count 必须是非负整数")
        if not isinstance(row["rd_frame_times"], list):
            raise CacheFormatError("事件缓存 rd_frame_times 必须是列表")
        for key in ("parent_anomalous", "training_excluded", "tr_dt_is_negative"):
            if not isinstance(row[key], bool):
                raise CacheFormatError(f"事件缓存 {key} 必须是布尔值")


def build_event_cache(
    project_root: str | Path,
    manifest: DatasetManifest | Mapping[str, Any] | None = None,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    cache_path: str | Path = DEFAULT_CACHE_RELATIVE_PATH,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> dict[str, Any]:
    """从已校验 manifest 构建轻量事件引用缓存。

    缓存只保存数据产物引用和 frame 映射，不复制原始 MAT、DAT 或 RD payload。
    """

    root = Path(project_root).resolve()
    if manifest is None:
        checked = load_manifest(
            manifest_path,
            root,
            verify_files=True,
            allow_unmaterialized=False,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    elif isinstance(manifest, DatasetManifest):
        checked = manifest
    else:
        checked = validate_manifest(
            manifest,
            root,
            verify_files=True,
            allow_unmaterialized=False,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    target = _resolve_cache_path(root, cache_path)
    rows = _event_cache_rows(
        checked,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    _validate_rows(rows, checked)
    payload = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        for row in rows
    )
    _atomic_write(target, payload, overwrite=overwrite)
    descriptor = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "kind": CACHE_KIND,
        "format": CACHE_FORMAT,
        "path": relative_project_path(root, target),
        "record_count": len(rows),
        "file_size": int(target.stat().st_size),
        "sha256": _sha256(target),
        "source_manifest_basis_sha256": _cache_basis(checked),
        "payload_materialized": False,
        "runtime_lru_enabled": True,
        "source_sections": {
            "TR": dict(checked.tr_index),
            "RD": dict(checked.rd_index),
        },
    }
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "cache_complete",
                "processed": len({str(row["parent_sample_id"]) for row in rows}),
                "total": len({str(row["parent_sample_id"]) for row in rows}),
                "record_count": len(rows),
                "path": descriptor["path"],
            }
        )
    return descriptor


def load_event_cache(
    project_root: str | Path,
    manifest: DatasetManifest | Mapping[str, Any] | None = None,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """加载并校验 manifest 中注册的事件缓存。"""

    root = Path(project_root).resolve()
    if manifest is None:
        checked = load_manifest(
            manifest_path,
            root,
            verify_files=True,
            allow_unmaterialized=False,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    elif isinstance(manifest, DatasetManifest):
        checked = manifest
    else:
        checked = validate_manifest(
            manifest,
            root,
            verify_files=True,
            allow_unmaterialized=False,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    descriptor = dict(checked.audit.get("cache") or {})
    if descriptor.get("schema_version") != CACHE_SCHEMA_VERSION or descriptor.get("kind") != CACHE_KIND:
        raise CacheFormatError("manifest 没有注册兼容的事件缓存")
    target = _resolve_cache_path(root, str(descriptor.get("path") or ""))
    if not target.is_file():
        raise CachePathError(f"事件缓存不存在：{target}")
    if int(descriptor.get("file_size", -1)) != int(target.stat().st_size):
        raise CacheIntegrityError("事件缓存 file_size 不一致")
    if str(descriptor.get("sha256")) != _sha256(target):
        raise CacheIntegrityError("事件缓存 SHA-256 不一致")
    source_basis = str(descriptor.get("source_manifest_basis_sha256") or "").lower()
    if source_basis not in _cache_basis_candidates(checked):
        raise CacheIntegrityError("事件缓存来源 manifest hash 不一致")
    rows: list[dict[str, Any]] = []
    try:
        with target.open("r", encoding="utf-8-sig") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise CacheFormatError(f"事件缓存记录必须是对象：{target}:{line_number}")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise CacheFormatError(f"事件缓存 JSONL 格式无效：{target}") from exc
    if len(rows) != int(descriptor.get("record_count", -1)):
        raise CacheIntegrityError("事件缓存 record_count 不一致")
    _validate_rows(rows, checked)
    return descriptor, rows


class CachedParentSequenceDataset:
    """使用事件缓存校验并复用 parent EventSequence 的训练数据集。"""

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
        manifest: DatasetManifest | None = None,
        cache_size: int = 8,
        validate_event_cache: bool = True,
    ) -> None:
        root = Path(project_root).resolve()
        if validate_event_cache:
            self.cache_descriptor, self.cache_rows = load_event_cache(
                root, manifest=manifest, manifest_path=manifest_path
            )
        else:
            # 正式训练前可以复用已由独立 cache 阶段完成的事件缓存。
            # 训练仍会按 manifest 读取真实 TR/RD 产物；这里只跳过再次
            # 扫描 event-source-cache.jsonl 和重新计算其来源 hash。
            self.cache_descriptor = {}
            self.cache_rows = []
        self._base = ParentSequenceDataset(
            root,
            manifest_path=manifest_path,
            split_name=split_name,
            parent_ids=parent_ids,
            include_unknown=include_unknown,
            training_view=training_view,
            allowed_labels=allowed_labels,
            task_name=task_name,
            allow_unknown_head_training=allow_unknown_head_training,
            manifest=manifest,
        )
        self.project_root = root
        self.manifest = self._base.manifest
        self.view = self._base.view
        self.parent_ids = list(self._base.parent_ids)
        self._positions = {value: index for index, value in enumerate(self.parent_ids)}
        self._sequence_cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_size = max(1, int(cache_size))
        if validate_event_cache:
            cached_parents = {str(row["parent_sample_id"]) for row in self.cache_rows}
            missing = sorted(set(self.parent_ids) - cached_parents)
            if missing:
                raise CacheIntegrityError(f"事件缓存缺少 parent：{missing[:3]}")
            cached_sample_ids = {str(row["sample_id"]) for row in self.cache_rows}
            missing_events = sorted(set(self._base.view.sample_ids) - cached_sample_ids)
            if missing_events:
                raise CacheIntegrityError(
                    f"事件缓存缺少统一视图事件：{missing_events[:3]}"
                )

    def __len__(self) -> int:
        return len(self.parent_ids)

    def __getitem__(self, index: int):
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("缓存数据集 index 必须是整数")
        parent_id = self.parent_ids[index]
        cached = self._sequence_cache.get(parent_id)
        if cached is not None:
            self._sequence_cache.move_to_end(parent_id)
            return cached
        sequence = self._base[index]
        self._sequence_cache[parent_id] = sequence
        while len(self._sequence_cache) > self._cache_size:
            self._sequence_cache.popitem(last=False)
        return sequence


__all__ = [
    "ProgressCallback",
    "DEFAULT_CACHE_RELATIVE_PATH",
    "build_event_cache",
    "load_event_cache",
    "CachedParentSequenceDataset",
]
