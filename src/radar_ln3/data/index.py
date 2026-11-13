"""通用 parent/event 索引构建和分层 sidecar 写入。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .adapter import DataSourceAdapter
from .atomic import write_bytes, write_json
from .contracts import (
    DataFormatError,
    DataIntegrityError,
    DataSplitError,
    DatasetManifest,
    EventRecord,
    ParentRecord,
)
from .manifest import (
    DEFAULT_PROGRESS_INTERVAL,
    INDEX_FORMAT,
    MANIFEST_SCHEMA_VERSION,
    PAIRING_SCHEMA_VERSION,
    RD_SCHEMA_VERSION,
    TR_SCHEMA_VERSION,
    canonical_json_hash,
    expected_baseline_training_view,
    normalise_progress_interval,
    normalise_split_name,
    validate_manifest,
)
from .paths import (
    PROJECT_MANIFEST_PATH,
    RD_INDEX_PATH,
    TR_INDEX_PATH,
    data_root_name,
    relative_project_path,
    resolve_index_path,
)
from .temporal import annotate_event_records


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _jsonl_bytes(records: list[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for record in records
    )


def _write_sidecar(
    project_root: Path,
    *,
    section: str,
    path: str,
    records: list[Mapping[str, Any]],
    overwrite: bool,
) -> dict[str, Any]:
    target = resolve_index_path(project_root, path, section=section, must_exist=False)
    payload = _jsonl_bytes(records)
    payload_hash = write_bytes(target, payload, overwrite=overwrite, label="索引文件")
    record = {
        "path": relative_project_path(project_root, target),
        "record_count": len(records),
        "file_size": int(target.stat().st_size),
        "sha256": payload_hash,
    }
    return record


def _sorted_parents(adapter: DataSourceAdapter) -> list[ParentRecord]:
    parents = list(adapter.discover_parents())
    parents.sort(key=lambda value: (str(value.parent_sample_id), str(value.track_id)))
    ids = [str(value.parent_sample_id) for value in parents]
    if len(ids) != len(set(ids)):
        raise DataIntegrityError("adapter 返回了重复 parent_sample_id")
    return parents


def _sorted_events(adapter: DataSourceAdapter, parent: ParentRecord) -> list[EventRecord]:
    events = list(adapter.discover_events(parent))
    events.sort(key=lambda value: (int(value.causal_event_index), str(value.sample_id)))
    indices = [int(value.causal_event_index) for value in events]
    if indices != list(range(len(events))):
        raise DataIntegrityError(f"adapter event index 不连续：{parent.parent_sample_id}")
    if len({str(value.sample_id) for value in events}) != len(events):
        raise DataIntegrityError(f"adapter 返回了重复 sample_id：{parent.parent_sample_id}")
    if any(str(event.parent_sample_id) != str(parent.parent_sample_id) for event in events):
        raise DataIntegrityError("adapter event 的 parent_sample_id 不一致")
    if any(
        str(event.track_id) != str(parent.track_id)
        or event.label_id != parent.label_id
        or str(event.label_name) != str(parent.label_name)
        for event in events
    ):
        raise DataIntegrityError(f"adapter event 身份与 parent 不一致：{parent.parent_sample_id}")
    return annotate_event_records(events)


def _record_base(event: EventRecord, *, split_name: str, data: Mapping[str, Any], available: bool) -> dict[str, Any]:
    event_metadata = dict(event.metadata)
    view_record = {
        "split_name": split_name,
        "label_name": str(event.label_name),
        "metadata": event_metadata,
    }
    record_metadata = dict(data.get("metadata") or {})
    record_metadata.update(
        {
            "baseline_training_view": expected_baseline_training_view(view_record),
            "parent_available": bool(data.get("parent_available", True)),
            "track_point_index": event_metadata.get("track_point_index"),
            "track_time_raw": event_metadata.get("track_time_raw"),
            "point_available": event_metadata.get("point_available"),
            "parent_anomalous": bool(event_metadata.get("parent_anomalous", False)),
            "training_excluded": bool(event_metadata.get("training_excluded", False)),
            "tr_dt_is_negative": bool(event_metadata.get("tr_dt_is_negative", False)),
            "tr_time_order_status": str(
                event_metadata.get("tr_time_order_status", "unknown")
            ),
            "parent_anomaly_reasons": list(
                event_metadata.get("parent_anomaly_reasons", []) or []
            ),
        }
    )
    if event_metadata.get("tr_timing") is not None:
        record_metadata["tr_timing"] = dict(event_metadata["tr_timing"])
    record = {
        "dataset_name": str(event.dataset_name),
        "source_name": str(event.source_name),
        "sample_id": str(event.sample_id),
        "parent_sample_id": str(event.parent_sample_id),
        "track_id": str(event.track_id),
        "causal_event_index": int(event.causal_event_index),
        "label_id": event.label_id,
        "label_name": str(event.label_name),
        "split_name": split_name,
        "available": bool(available),
        "source_path": str(data.get("source_path", event.source_path)),
        "data_path": str(data.get("data_path", "")),
        "file_size": data.get("file_size"),
        "sha256": data.get("sha256"),
        "metadata": record_metadata,
    }
    return record


def _rd_frame_records(
    event: EventRecord,
    *,
    split_name: str,
    artifact: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if not bool(artifact.get("available", False)):
        return []
    metadata = dict(artifact.get("metadata") or {})
    frame_indices = metadata.get("frame_indices") or []
    source_paths = str(artifact.get("source_path", "")).split(";") if artifact.get("source_path") else []
    frame_metadata = metadata.get("frame_metadata") or []
    if frame_metadata and len(frame_metadata) != len(frame_indices):
        raise DataIntegrityError(f"RD frame metadata 与 frame_indices 数量不一致：{event.sample_id}")
    if len(source_paths) != len(frame_indices):
        raise DataIntegrityError(f"RD frame source 与 frame_indices 数量不一致：{event.sample_id}")
    baseline_training_view = expected_baseline_training_view(
        {
            "split_name": split_name,
            "label_name": str(event.label_name),
            "metadata": dict(event.metadata),
        }
    )
    result: list[dict[str, Any]] = []
    for position, frame_index in enumerate(frame_indices):
        item_metadata = dict(frame_metadata[position]) if position < len(frame_metadata) and isinstance(frame_metadata[position], Mapping) else {}
        item_metadata.update(
            {
                "baseline_training_view": baseline_training_view,
                "parent_anomalous": bool(event.metadata.get("parent_anomalous", False)),
                "training_excluded": bool(event.metadata.get("training_excluded", False)),
                "tr_time_order_status": str(
                    event.metadata.get("tr_time_order_status", "unknown")
                ),
            }
        )
        result.append({
            "record_type": "frame",
            "frame_index": int(frame_index),
            "dataset_name": str(event.dataset_name),
            "source_name": str(event.source_name),
            "sample_id": str(event.sample_id),
            "parent_sample_id": str(event.parent_sample_id),
            "track_id": str(event.track_id),
            "causal_event_index": int(event.causal_event_index),
            "label_id": event.label_id,
            "label_name": str(event.label_name),
            "split_name": split_name,
            "available": True,
            "source_path": source_paths[position],
            "data_path": str(artifact.get("data_path", "")),
            "file_size": artifact.get("file_size"),
            "sha256": artifact.get("sha256"),
            "metadata": {
                "pairing_basis": str(artifact.get("pairing_basis", "adapter_event_identity")),
                **item_metadata,
            },
        })
    return result


def build_manifest(
    adapter: DataSourceAdapter,
    project_root: str | Path,
    *,
    source_policy: Mapping[str, Any] | None = None,
    split_policy: Mapping[str, Any] | None = None,
    materialize: bool = True,
    overwrite: bool = False,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> DatasetManifest:
    """发现 parent/event，写出 sidecar，并构造小型总 manifest。"""

    progress_interval = normalise_progress_interval(progress_interval)
    root = Path(project_root).resolve()
    parents = _sorted_parents(adapter)
    excluded_parents: list[ParentRecord] = []
    excluded_discoverer = getattr(adapter, "discover_excluded_parents", None)
    if callable(excluded_discoverer):
        excluded_parents = list(excluded_discoverer())
        excluded_parents.sort(
            key=lambda value: (str(value.parent_sample_id), str(value.track_id))
        )
        excluded_ids = {str(value.parent_sample_id) for value in excluded_parents}
        parent_ids_seen = {str(value.parent_sample_id) for value in parents}
        if excluded_ids & parent_ids_seen:
            raise DataIntegrityError("可处理 parent 与 excluded parent 身份重叠")
        if len(excluded_ids) != len(excluded_parents):
            raise DataIntegrityError("adapter 返回了重复 excluded parent_sample_id")
    parent_ids = [str(parent.parent_sample_id) for parent in parents]
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "start",
                "dataset_name": str(adapter.dataset_name),
                "parent_count": len(parents),
                "excluded_parent_count": len(excluded_parents),
                "materialize": bool(materialize),
                "progress_interval": progress_interval,
            }
        )
    split_by_parent = {str(key): str(value) for key, value in adapter.assign_split(parent_ids).items()}
    if set(split_by_parent) != set(parent_ids):
        raise DataSplitError("adapter split assignment 未覆盖全部 parent")
    split_by_parent = {parent_id: normalise_split_name(split_name) for parent_id, split_name in split_by_parent.items()}

    tr_records: list[dict[str, Any]] = []
    rd_frames: list[dict[str, Any]] = []
    parent_metadata: dict[str, dict[str, Any]] = {}
    for parent_number, parent in enumerate(parents, start=1):
        parent_id = str(parent.parent_sample_id)
        split_name = split_by_parent[parent_id]
        events = _sorted_events(adapter, parent)
        if materialize:
            materializer = getattr(adapter, "materialize_parent", None)
            if not callable(materializer):
                raise DataFormatError("启用数据产物写出时，适配器必须提供 materialize_parent")
            materialized = materializer(parent, events, split_name=split_name, overwrite=overwrite)
        else:
            materialized = {"tr": {}, "rd": {}, "metadata": {"materialized": False}}
        parent_metadata[parent_id] = dict(materialized.get("metadata") or {})
        tr_artifacts = dict(materialized.get("tr") or {})
        rd_artifacts = dict(materialized.get("rd") or {})
        for event in events:
            sample_id = str(event.sample_id)
            tr_artifact = dict(tr_artifacts.get(sample_id) or {})
            rd_artifact = dict(rd_artifacts.get(sample_id) or {})
            tr_available = bool(materialize and tr_artifact.get("available", event.available))
            rd_available = bool(materialize and rd_artifact.get("available", False))
            tr_records.append(_record_base(event, split_name=split_name, data=tr_artifact, available=tr_available))
            rd_frames.extend(_rd_frame_records(event, split_name=split_name, artifact=rd_artifact) if rd_available else [])
        if progress_callback is not None and (
            parent_number % progress_interval == 0 or parent_number == len(parents)
        ):
            progress_callback(
                {
                    "stage": "parent_progress",
                    "dataset_name": str(adapter.dataset_name),
                    "processed": parent_number,
                    "total": len(parents),
                    "parent_number": parent_number,
                    "parent_count": len(parents),
                    "parent_sample_id": parent_id,
                    "event_count": len(events),
                    "split_name": split_name,
                    "materialize": bool(materialize),
                    "parent_anomalous": any(
                        bool(event.metadata.get("parent_anomalous", False))
                        for event in events
                    ),
                }
            )

    tr_records.sort(key=lambda value: (str(value["parent_sample_id"]), int(value["causal_event_index"])))
    rd_frames.sort(key=lambda value: (str(value["parent_sample_id"]), int(value["causal_event_index"]), int(value["frame_index"])))
    tr_index_path = str(TR_INDEX_PATH)
    rd_index_path = str(RD_INDEX_PATH)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "sidecar_start",
                "section": "TR",
                "path": tr_index_path,
                "record_count": len(tr_records),
            }
        )
    tr_part = _write_sidecar(root, section="TR", path=tr_index_path, records=tr_records, overwrite=overwrite)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "sidecar_complete",
                "section": "TR",
                **tr_part,
            }
        )
        progress_callback(
            {
                "stage": "sidecar_start",
                "section": "RD",
                "path": rd_index_path,
                "record_count": len(rd_frames),
            }
        )
    rd_part = _write_sidecar(root, section="RD", path=rd_index_path, records=rd_frames, overwrite=overwrite)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "sidecar_complete",
                "section": "RD",
                **rd_part,
            }
        )
    rd_sample_ids = {str(row["sample_id"]) for row in rd_frames}
    paired_event_count = len(tr_records)
    missing_rd_event_count = sum(
        1 for row in tr_records if str(row["sample_id"]) not in rd_sample_ids
    )
    payload = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": str(adapter.dataset_name),
        "data_root": data_root_name(),
        "source_policy": {"parent_split_boundary": True, "frame_level_split_forbidden": True, "unknown_training_exposure": False, **dict(source_policy or {})},
        "split": {"group_unit": "parent_sample_id", "split_field": "split_name", **dict(split_policy or {})},
        "TR": {"schema_version": TR_SCHEMA_VERSION, "format": INDEX_FORMAT, "record_count": len(tr_records), "parts": [tr_part]},
        "RD": {"schema_version": RD_SCHEMA_VERSION, "format": INDEX_FORMAT, "record_count": len(rd_frames), "parts": [rd_part]},
        "pairing": {"schema_version": PAIRING_SCHEMA_VERSION, "strategy": "sample_id_and_causal_event_index", "tr_index": "TR", "rd_index": "RD", "rd_frame_cardinality": "many_frames_per_event", "paired_event_count": paired_event_count, "missing_tr_event_count": 0, "missing_rd_event_count": missing_rd_event_count},
        "audit": {
            "parent_count": len(parents),
            "parent_ids": parent_ids,
            "parent_metadata": parent_metadata,
            "excluded_parent_count": len(excluded_parents),
            "excluded_parents": [parent.to_dict() for parent in excluded_parents],
            "materialized": bool(materialize),
            "data_semantics": {
                "event_unit": "one_TR_track_point",
                "rd_frame_cardinality": "many_frames_per_event",
                "rd_frame_order": "frame_index_ascending",
                "rd_time_semantics": "parent_relative_cpi_elapsed_seconds",
                "tr_time_semantics": "source_timestamp_seconds_in_causal_event_order",
                "tr_zero_dt_policy": "preserve_event_mark_invalid_dt",
                "negative_dt_policy": "preserve_event_mark_parent_excluded",
                "rd_time_substitutes_tr_time": False,
            },
        },
    }
    validated = validate_manifest(
        payload,
        root,
        verify_files=materialize,
        allow_unmaterialized=not materialize,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "complete",
                "dataset_name": str(adapter.dataset_name),
                "parent_count": len(parents),
                "excluded_parent_count": len(excluded_parents),
                "tr_record_count": len(tr_records),
                "rd_record_count": len(rd_frames),
                "pairing_count": paired_event_count,
                "materialized": bool(materialize),
            }
        )
    return validated


def write_manifest(
    manifest: DatasetManifest | Mapping[str, Any],
    project_root: str | Path,
    *,
    path: str | Path = PROJECT_MANIFEST_PATH,
    overwrite: bool = False,
    verify_files: bool = True,
    validate: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> Path:
    """校验并原子写入唯一总 manifest。

    ``validate=False`` 仅用于调用方刚刚完成 ``validate_manifest`` 的场景，
    跳过重复的整份 sidecar 重新校验；manifest 内容仍会重新封 hash。
    """

    root = Path(project_root).resolve()
    payload = manifest.to_dict() if isinstance(manifest, DatasetManifest) else dict(manifest)
    payload.pop("manifest_sha256", None)
    if validate:
        validate_manifest(
            payload,
            root,
            verify_files=bool(verify_files),
            allow_unmaterialized=True,
            progress_callback=progress_callback,
            progress_interval=progress_interval,
        )
    payload["manifest_sha256"] = canonical_json_hash(payload)
    target = (root / path if not Path(path).is_absolute() else Path(path)).resolve()
    relative_project_path(root, target)
    write_json(target, payload, overwrite=overwrite, label="总 manifest")
    return target


__all__ = ["ProgressCallback", "build_manifest", "write_manifest"]
