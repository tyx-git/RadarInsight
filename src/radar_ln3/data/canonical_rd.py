"""CQ08 canonical RD storage owned by the standalone radar_ln3 package.

The canonical source is an already prepared, provenance-bound RD index and
tensor collection.  This module copies that source into ``data/cq08`` and
loads it without importing any historical experiment package.  Event records
retain both tensor-frame and raw-DAT identities so raw-IQ features can use the
same causal frame selection as the canonical RD features.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import tempfile
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import DataFormatError, DataIntegrityError, DataReadError
from .manifest import load_manifest
from .paths import ensure_path_inside_root, relative_project_path, resolve_project_path


CANONICAL_RD_SCHEMA_VERSION = "RADAR-LN3-CQ08-CANONICAL-RD-V1"
CANONICAL_SOURCE_SCHEMA_VERSION = "cq08_rd_physical_canonical_v3_1"
CANONICAL_RD_RELATIVE_ROOT = Path("data/cq08/rd/canonical")
CANONICAL_RD_MANIFEST_NAME = "canonical-rd-manifest.json"
CANONICAL_RD_EVENT_INDEX_NAME = "canonical-rd-event-index.jsonl"
CANONICAL_RD_FRAME_INDEX_NAME = "canonical-rd-frame-index.csv"
DEFAULT_CANONICAL_FRAME_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)
CANONICAL_RD_VELOCITY_AXIS_HASH = (
    "5912f04e9119f420c668074b3f73dbfeabe1d466e4ad701f9824a1e0598b03fe"
)
_MAT_FRAME_RE = re.compile(r"_(?P<frame>\d+)_(?P<track>\d+)\.mat$", re.IGNORECASE)
_REQUIRED_FRAME_COLUMNS = {
    "schema_version",
    "parent_sample_id",
    "track_id",
    "source_name",
    "label_name",
    "raw_frame_index",
    "tensor_frame_index",
    "is_tensor_frame_kept",
    "track_point_index",
    "rd_mat_path",
    "rd_tensor_path",
    "canonical_rd_shape",
}
_REQUIRED_TENSOR_FIELDS = {
    "amplitude",
    "phase",
    "time_raw",
    "time_seconds",
    "velocity_mps",
    "range_offset_m",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_parent_name(parent_id: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(parent_id)).strip("._")
    if not value:
        raise DataFormatError(f"canonical RD parent ID 无法生成文件名：{parent_id!r}")
    return value


def canonical_rd_root(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    target = (root / CANONICAL_RD_RELATIVE_ROOT).resolve()
    ensure_path_inside_root(target, root / "data" / "cq08")
    return target


def canonical_rd_manifest_path(project_root: str | Path) -> Path:
    return canonical_rd_root(project_root) / CANONICAL_RD_MANIFEST_NAME


def canonical_rd_binding(project_root: str | Path) -> str | None:
    """Return the local canonical store identity, or ``None`` if absent."""

    path = canonical_rd_manifest_path(project_root)
    if not path.is_file():
        return None
    return _sha256_file(path)


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes"}:
        return True
    if text in {"0", "false", "no"}:
        return False
    raise DataFormatError(f"canonical RD 布尔字段无效：{value!r}")


def _parse_mat_frame_number(value: str, *, expected_track: str) -> int:
    match = _MAT_FRAME_RE.search(Path(str(value)).name)
    if match is None or str(match["track"]) != str(expected_track):
        raise DataIntegrityError(
            f"canonical RD MAT frame identity 无效：track={expected_track}, path={value}"
        )
    return int(match["frame"])


def _load_frame_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise DataReadError(f"canonical RD frame index 不存在：{path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(_REQUIRED_FRAME_COLUMNS - fields)
        if missing:
            raise DataFormatError(f"canonical RD frame index 缺少字段：{missing}")
        rows = [dict(row) for row in reader]
    if not rows:
        raise DataFormatError("canonical RD frame index 为空")
    if set(str(row.get("schema_version", "")) for row in rows) != {
        CANONICAL_SOURCE_SCHEMA_VERSION
    }:
        raise DataFormatError("canonical RD frame index schema_version 不一致")
    if any(not _parse_bool(row["is_tensor_frame_kept"]) for row in rows):
        raise DataIntegrityError("canonical RD frame index 含被丢弃的 tensor frame")
    seen: set[tuple[str, int]] = set()
    for row in rows:
        identity = (str(row["parent_sample_id"]), int(row["tensor_frame_index"]))
        if identity in seen:
            raise DataIntegrityError(f"canonical RD tensor frame 重复：{identity}")
        seen.add(identity)
        if str(row["canonical_rd_shape"]) != "21x384":
            raise DataFormatError("canonical RD shape 不是 21x384")
    return rows


def _load_tensor(path: Path) -> dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as payload:
            missing = sorted(_REQUIRED_TENSOR_FIELDS - set(payload.files))
            if missing:
                raise DataFormatError(f"canonical RD tensor 缺少字段：{path}: {missing}")
            values = {name: np.asarray(payload[name]) for name in _REQUIRED_TENSOR_FIELDS}
    except (OSError, ValueError) as exc:
        if isinstance(exc, DataFormatError):
            raise
        raise DataReadError(f"无法读取 canonical RD tensor：{path}") from exc
    amplitude = values["amplitude"]
    phase = values["phase"]
    times = values["time_seconds"]
    velocity = values["velocity_mps"]
    offsets = values["range_offset_m"]
    if (
        amplitude.ndim != 3
        or tuple(amplitude.shape[1:]) != (21, 384)
        or phase.shape != amplitude.shape
        or times.shape != (amplitude.shape[0],)
        or values["time_raw"].shape != times.shape
        or velocity.shape != (384,)
        or offsets.shape != (21,)
    ):
        raise DataFormatError(f"canonical RD tensor shape 无效：{path}")
    for name, value in values.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise DataIntegrityError(f"canonical RD tensor 含非有限值：{path}:{name}")
    if not np.all(np.diff(velocity.astype(np.float64)) > 0.0):
        raise DataFormatError(f"canonical RD velocity_mps 非严格递增：{path}")
    axis_hash = hashlib.sha256(
        np.ascontiguousarray(velocity.astype(np.float64)).tobytes(order="C")
    ).hexdigest()
    if axis_hash != CANONICAL_RD_VELOCITY_AXIS_HASH:
        raise DataIntegrityError(f"canonical RD velocity axis hash 不一致：{path}")
    return values


def _copy_tensor(source: Path, target: Path, *, overwrite: bool) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        if _sha256_file(target) != _sha256_file(source):
            raise FileExistsError(f"canonical RD 输出已存在且内容不同：{target}")
    elif target.exists() and overwrite:
        target.unlink()
    if not target.exists():
        temporary = target.with_name(f".{target.name}.tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    return {
        "path": target,
        "file_size": int(target.stat().st_size),
        "sha256": _sha256_file(target),
    }


def materialize_canonical_rd_store(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    source_frame_index_path: str | Path = DEFAULT_CANONICAL_FRAME_INDEX,
    output_root: str | Path = CANONICAL_RD_RELATIVE_ROOT,
    overwrite: bool = False,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Copy and bind canonical RD tensors into ``data/cq08``.

    The source is read only.  The generated store is self-contained under the
    project data root and is later consumed by :class:`CanonicalRDStore`.
    """

    root = Path(project_root).resolve()
    output = resolve_project_path(root, output_root)
    ensure_path_inside_root(output, root / "data" / "cq08")
    source_index = resolve_project_path(root, source_frame_index_path, must_exist=True)
    rows = _load_frame_rows(source_index)
    manifest = load_manifest(
        manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
    )
    tr_by_key = {
        (str(row["parent_sample_id"]), int(row["causal_event_index"])): row
        for row in manifest.tr_records
    }
    if len(tr_by_key) != len(manifest.tr_records):
        raise DataIntegrityError("TR manifest 存在重复 parent/event 身份")

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row["parent_sample_id"]), []).append(row)
    for values in grouped.values():
        values.sort(key=lambda item: int(item["tensor_frame_index"]))

    output.mkdir(parents=True, exist_ok=True)
    tensor_root = output / "tensors"
    tensor_root.mkdir(parents=True, exist_ok=True)
    tensor_descriptors: dict[str, dict[str, Any]] = {}
    for parent_id, parent_rows in sorted(grouped.items()):
        source_path = resolve_project_path(root, parent_rows[0]["rd_tensor_path"], must_exist=True)
        if any(str(row["rd_tensor_path"]) != str(parent_rows[0]["rd_tensor_path"]) for row in parent_rows):
            raise DataIntegrityError(f"parent 引用了多个 canonical RD tensor：{parent_id}")
        tensor = _load_tensor(source_path)
        tensor_indices = [int(row["tensor_frame_index"]) for row in parent_rows]
        if tensor_indices != list(range(len(tensor_indices))):
            raise DataIntegrityError(f"canonical RD tensor frame index 不连续：{parent_id}")
        if max(tensor_indices, default=-1) >= tensor["amplitude"].shape[0]:
            raise DataIntegrityError(f"canonical RD tensor frame index 越界：{parent_id}")
        target = tensor_root / f"{_safe_parent_name(parent_id)}.npz"
        descriptor = _copy_tensor(source_path, target, overwrite=overwrite)
        relative = relative_project_path(root, target)
        tensor_descriptors[parent_id] = {
            "path": relative,
            "source_path": relative_project_path(root, source_path),
            **{key: value for key, value in descriptor.items() if key != "path"},
            "frame_count": int(tensor["amplitude"].shape[0]),
        }

    event_rows: list[dict[str, Any]] = []
    frame_rows_by_event: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in rows:
        key = (str(row["parent_sample_id"]), int(row["track_point_index"]))
        frame_rows_by_event.setdefault(key, []).append(row)
    for values in frame_rows_by_event.values():
        values.sort(key=lambda item: int(item["tensor_frame_index"]))

    for key, tr in sorted(tr_by_key.items()):
        parent_id, event_index = key
        track_point = int(
            (tr.get("metadata") or {}).get("track_point_index", event_index + 1)
        )
        frame_rows = frame_rows_by_event.get((parent_id, track_point), [])
        available = bool(frame_rows)
        record: dict[str, Any] = {
            "schema_version": CANONICAL_RD_SCHEMA_VERSION,
            "sample_id": str(tr["sample_id"]),
            "parent_sample_id": parent_id,
            "track_id": str(tr["track_id"]),
            "causal_event_index": event_index,
            "track_point_index": track_point,
            "label_name": str(tr["label_name"]),
            "split_name": str(tr["split_name"]),
            "available": available,
            "data_path": tensor_descriptors[parent_id]["path"] if available else "",
            "tensor_frame_indices": [int(row["tensor_frame_index"]) for row in frame_rows],
            "legacy_mat_list_ordinals": [int(row["raw_frame_index"]) for row in frame_rows],
            "raw_dat_frame_indices": [
                _parse_mat_frame_number(row["rd_mat_path"], expected_track=str(tr["track_id"])) - 1
                for row in frame_rows
            ],
            "mat_frame_numbers": [
                _parse_mat_frame_number(row["rd_mat_path"], expected_track=str(tr["track_id"]))
                for row in frame_rows
            ],
            "rd_mat_paths": [str(row["rd_mat_path"]) for row in frame_rows],
            "rd_tensor_path": str(frame_rows[0]["rd_tensor_path"]) if frame_rows else "",
        }
        event_rows.append(record)

    event_index_path = output / CANONICAL_RD_EVENT_INDEX_NAME
    frame_index_path = output / CANONICAL_RD_FRAME_INDEX_NAME
    event_index_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in event_rows),
        encoding="utf-8",
    )
    if frame_index_path.exists() and not overwrite:
        if _sha256_file(frame_index_path) != _sha256_file(source_index):
            raise FileExistsError(f"canonical RD frame index 已存在且内容不同：{frame_index_path}")
    else:
        shutil.copyfile(source_index, frame_index_path)
    body = {
        "schema_version": CANONICAL_RD_SCHEMA_VERSION,
        "source_schema_version": CANONICAL_SOURCE_SCHEMA_VERSION,
        "source_frame_index": relative_project_path(root, source_index),
        "source_frame_index_sha256": _sha256_file(source_index),
        "manifest_path": relative_project_path(root, resolve_project_path(root, manifest_path, must_exist=True)),
        "source_manifest_sha256": manifest.manifest_sha256,
        "event_index_path": relative_project_path(root, event_index_path),
        "event_index_sha256": _sha256_file(event_index_path),
        "frame_index_path": relative_project_path(root, frame_index_path),
        "frame_index_sha256": _sha256_file(frame_index_path),
        "tensor_count": len(tensor_descriptors),
        "event_count": len(event_rows),
        "available_event_count": sum(int(row["available"]) for row in event_rows),
        "parent_count": len({str(row["parent_sample_id"]) for row in event_rows}),
        "tensor_descriptors": tensor_descriptors,
    }
    payload = {**body, "store_manifest_sha256": _canonical_json_hash(body)}
    _atomic_json(output / CANONICAL_RD_MANIFEST_NAME, payload)
    return {
        "manifest_path": relative_project_path(root, output / CANONICAL_RD_MANIFEST_NAME),
        "manifest_sha256": _sha256_file(output / CANONICAL_RD_MANIFEST_NAME),
        "event_count": len(event_rows),
        "available_event_count": int(sum(row["available"] for row in event_rows)),
        "tensor_count": len(tensor_descriptors),
    }


class CanonicalRDStore:
    """Validated local canonical RD event store with a small tensor cache."""

    def __init__(self, project_root: str | Path, *, cache_size: int = 4) -> None:
        self.project_root = Path(project_root).resolve()
        self.root = canonical_rd_root(self.project_root)
        self.manifest_path = self.root / CANONICAL_RD_MANIFEST_NAME
        if not self.manifest_path.is_file():
            raise DataReadError(f"canonical RD manifest 不存在：{self.manifest_path}")
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != CANONICAL_RD_SCHEMA_VERSION:
            raise DataFormatError("canonical RD manifest schema_version 不匹配")
        declared = str(payload.get("store_manifest_sha256") or "")
        body = dict(payload)
        body.pop("store_manifest_sha256", None)
        if declared != _canonical_json_hash(body):
            raise DataIntegrityError("canonical RD manifest 内容 hash 不一致")
        self.payload = payload
        event_path = resolve_project_path(self.project_root, str(payload["event_index_path"]), must_exist=True)
        if _sha256_file(event_path) != str(payload["event_index_sha256"]):
            raise DataIntegrityError("canonical RD event index hash 不一致")
        self.events: dict[tuple[str, int], dict[str, Any]] = {}
        with event_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (str(row["parent_sample_id"]), int(row["causal_event_index"]))
                if key in self.events:
                    raise DataIntegrityError(f"canonical RD event 重复：{key}")
                self.events[key] = dict(row)
        if len(self.events) != int(payload.get("event_count", -1)):
            raise DataIntegrityError("canonical RD event_count 不一致")
        self.binding = _sha256_file(self.manifest_path)
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self._cache_size = max(1, int(cache_size))

    def _tensor(self, path: str) -> dict[str, np.ndarray]:
        cached = self._cache.get(path)
        if cached is not None:
            self._cache.move_to_end(path)
            return cached
        value = _load_tensor(resolve_project_path(self.project_root, path, must_exist=True))
        self._cache[path] = value
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value

    def event(self, parent_sample_id: str, causal_event_index: int) -> dict[str, Any]:
        key = (str(parent_sample_id), int(causal_event_index))
        try:
            row = self.events[key]
        except KeyError as exc:
            raise DataIntegrityError(f"canonical RD 缺少事件：{key}") from exc
        if not bool(row["available"]):
            return {**row, "rd_values": None}
        tensor = self._tensor(str(row["data_path"]))
        indices = np.asarray(row["tensor_frame_indices"], dtype=np.int64)
        if indices.size == 0 or int(indices.max()) >= tensor["amplitude"].shape[0]:
            raise DataIntegrityError(f"canonical RD 事件帧索引无效：{key}")
        rd_values = {
            "amplitude": tensor["amplitude"][indices],
            "phase": tensor["phase"][indices],
            "time_seconds": tensor["time_seconds"][indices],
            "velocity_mps": tensor["velocity_mps"],
            "range_offset_m": tensor["range_offset_m"],
            "frame_indices": indices,
            "canonical_frame_indices": indices,
            "raw_dat_frame_indices": np.asarray(row["raw_dat_frame_indices"], dtype=np.int64),
            "mat_frame_numbers": np.asarray(row["mat_frame_numbers"], dtype=np.int64),
            "frame_count": int(indices.size),
            "time_semantics": "canonical_rd_tensor_time_seconds",
            "canonical": True,
        }
        return {**row, "rd_values": rd_values}


def load_canonical_rd_store(
    project_root: str | Path,
    *,
    required: bool = False,
) -> CanonicalRDStore | None:
    path = canonical_rd_manifest_path(project_root)
    if not path.is_file():
        if required:
            raise DataReadError(f"canonical RD store 不存在：{path}")
        return None
    return CanonicalRDStore(project_root)


__all__ = [
    "CANONICAL_RD_EVENT_INDEX_NAME",
    "CANONICAL_RD_MANIFEST_NAME",
    "CANONICAL_RD_RELATIVE_ROOT",
    "CANONICAL_RD_SCHEMA_VERSION",
    "CANONICAL_RD_VELOCITY_AXIS_HASH",
    "CanonicalRDStore",
    "canonical_rd_binding",
    "canonical_rd_manifest_path",
    "canonical_rd_root",
    "load_canonical_rd_store",
    "materialize_canonical_rd_store",
]
