"""由总 manifest 派生 CSV 和产品 artifact 注册信息。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
from collections.abc import Mapping
from typing import Any

from .atomic import write_json
from .contracts import DataFormatError, DatasetManifest
from .manifest import canonical_json_hash, data_manifest_hash, validate_manifest
from .paths import (
    ensure_path_inside_root,
    relative_project_path,
    resolve_project_path,
    resolve_tr_path,
)
from .raw_reader import sha256_file as _data_sha256_file


CSV_PATH = Path("data/cq08/csv/cq08-event-index.csv")


def _sha256(path: Path) -> str:
    return _data_sha256_file(path)


def _is_sha256(value: Any) -> bool:
    text = str(value or "")
    return len(text) == 64 and not set(text.lower()) - set("0123456789abcdef")


def _read_tr_values(root: Path, records: list[dict[str, Any]]) -> dict[str, str]:
    """按 data_path 缓存读取 TR JSONL，避免重复打开同一 parent 文件。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["data_path"]), []).append(record)
    result: dict[str, str] = {}
    for data_path, expected in grouped.items():
        path = resolve_tr_path(root, data_path, must_exist=True)
        local_result: dict[str, str] = {}
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict) or "sample_id" not in row:
                        raise DataFormatError(f"TR JSONL 记录无效：{path}:{line_number}")
                    values = row.get("values")
                    if not isinstance(values, dict):
                        raise DataFormatError(f"TR values 必须是对象：{path}:{line_number}")
                    sample_id = str(row["sample_id"])
                    if sample_id in local_result:
                        raise DataFormatError(
                            f"TR JSONL 存在重复 sample_id：{path}:{line_number}"
                        )
                    local_result[sample_id] = json.dumps(
                        values,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
        except (OSError, json.JSONDecodeError) as exc:
            raise DataFormatError(f"无法读取 TR JSONL：{path}") from exc
        expected_ids = {str(row["sample_id"]) for row in expected}
        actual_ids = set(local_result)
        if actual_ids != expected_ids:
            raise DataFormatError(f"TR JSONL 与总索引覆盖范围不一致：{path}")
        result.update(local_result)
    return result


def _iter_jsonl_records(
    root: Path,
    section: str,
    payload: dict[str, Any],
):
    """逐条读取一个已登记的 JSONL 索引，避免一次展开全部 RD 记录。"""

    section_payload = payload.get(section)
    if not isinstance(section_payload, dict):
        raise DataFormatError(f"manifest 缺少 {section} section")
    total = 0
    for part in section_payload.get("parts") or []:
        if not isinstance(part, dict):
            raise DataFormatError(f"manifest {section}.parts 记录无效")
        raw_path = str(part.get("path") or "")
        if section == "TR":
            path = resolve_tr_path(root, raw_path, must_exist=True)
        else:
            from .paths import resolve_rd_path

            path = resolve_rd_path(root, raw_path, must_exist=True)
        try:
            with path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise DataFormatError(f"{section} JSONL 记录必须是对象：{path}:{line_number}")
                    total += 1
                    yield value
        except (OSError, json.JSONDecodeError) as exc:
            raise DataFormatError(f"无法读取 {section} JSONL：{path}") from exc
    expected = int(section_payload.get("record_count", total))
    if expected != total:
        raise DataFormatError(f"{section} JSONL record_count 不一致")


def _load_manifest_payload(root: Path, path: str | Path) -> dict[str, Any]:
    target = resolve_project_path(root, path, must_exist=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DataFormatError(f"无法读取总 manifest：{target}") from exc
    if not isinstance(payload, dict):
        raise DataFormatError("总 manifest 根节点必须是对象")
    return payload


def build_event_csv_index(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    output_path: str | Path = CSV_PATH,
    overwrite: bool = False,
    manifest: DatasetManifest | None = None,
) -> dict[str, Any]:
    """从已验证 manifest 生成便于筛选和审计的事件 CSV。"""

    root = Path(project_root).resolve()
    payload = None if manifest is not None else _load_manifest_payload(root, manifest_path)
    if manifest is None:
        tr_records = list(_iter_jsonl_records(root, "TR", payload))
        rd_records = []
        rd_by_id: dict[str, dict[str, Any]] = {}
        for frame in _iter_jsonl_records(root, "RD", payload):
            sample_id = str(frame.get("sample_id", ""))
            entry = rd_by_id.setdefault(
                sample_id,
                {
                    "data_path": str(frame.get("data_path", "")),
                    "metadata": {"rd_frame_count": 0},
                },
            )
            entry["metadata"]["rd_frame_count"] = int(
                entry["metadata"]["rd_frame_count"]
            ) + 1
            if not entry["data_path"]:
                entry["data_path"] = str(frame.get("data_path", ""))
        rd_records = list(rd_by_id.values())
        # CSV 只绑定稳定的数据索引 hash，不能绑定会随 registry/cache
        # 登记变化的总 manifest 自身 hash。
        manifest_sha256 = data_manifest_hash(payload)
    else:
        tr_records = list(manifest.tr_records)
        rd_records = list(manifest.rd_records)
        manifest_sha256 = data_manifest_hash(manifest)
    target = resolve_project_path(root, output_path)
    tr_values = _read_tr_values(root, tr_records)
    if manifest is not None:
        rd_by_id = {
            str(record["sample_id"]): record for record in rd_records
        }
    fields = (
        "sample_id", "parent_sample_id", "track_id", "causal_event_index",
        "label_id", "label_name", "split_name", "baseline_training_view",
        "training_excluded", "parent_anomalous", "tr_data_path", "rd_data_path",
        "rd_frame_count", "tr_feature_values_json",
    )
    rows: list[dict[str, Any]] = []
    for record in tr_records:
        sample_id = str(record["sample_id"])
        rd = rd_by_id.get(sample_id, {})
        metadata = dict(record.get("metadata") or {})
        rd_metadata = dict(rd.get("metadata") or {})
        rows.append({
            "sample_id": sample_id,
            "parent_sample_id": str(record["parent_sample_id"]),
            "track_id": str(record["track_id"]),
            "causal_event_index": int(record["causal_event_index"]),
            "label_id": record["label_id"],
            "label_name": str(record["label_name"]),
            "split_name": str(record["split_name"]),
            "baseline_training_view": bool(metadata.get("baseline_training_view", False)),
            "training_excluded": bool(metadata.get("training_excluded", False)),
            "parent_anomalous": bool(metadata.get("parent_anomalous", False)),
            "tr_data_path": str(record.get("data_path", "")),
            "rd_data_path": str(rd.get("data_path", "")),
            "rd_frame_count": int(rd_metadata.get("rd_frame_count", 0)),
            "tr_feature_values_json": tr_values[sample_id],
        })
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".tmp", dir=target.parent, delete=False
    ) as raw_handle:
        class _DigestTextWriter:
            def __init__(self, handle):
                self.handle = handle
                self.digest = hashlib.sha256()

            def write(self, value: str) -> int:
                self.digest.update(value.encode("utf-8"))
                return self.handle.write(value)

            def flush(self) -> None:
                self.handle.flush()

            def fileno(self) -> int:
                return self.handle.fileno()

        temporary = Path(raw_handle.name)
        handle = _DigestTextWriter(raw_handle)
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
        candidate_hash = handle.digest.hexdigest()
    try:
        if target.exists() and not overwrite:
            existing = target.read_bytes()
            candidate = temporary.read_bytes()
            if existing != candidate:
                raise FileExistsError(f"拒绝覆盖内容不同的 CSV：{target}")
        else:
            if overwrite:
                os.replace(temporary, target)
                temporary = None
            else:
                try:
                    os.link(temporary, target)
                except FileExistsError:
                    if target.is_file() and target.read_bytes() == temporary.read_bytes():
                        return {
                            "path": relative_project_path(root, target),
                            "format": "csv",
                            "record_count": len(rows),
                            "file_size": int(target.stat().st_size),
                            "sha256": candidate_hash,
                            "source_manifest_sha256": manifest_sha256,
                        }
                    raise FileExistsError(f"拒绝覆盖竞态产生的不同 CSV：{target}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "path": relative_project_path(root, target),
        "format": "csv",
        "record_count": len(rows),
        "file_size": int(target.stat().st_size),
        "sha256": candidate_hash,
        "source_manifest_sha256": manifest_sha256,
    }


def _file_descriptor(root: Path, path: str, *, kind: str, task: str | None = None) -> dict[str, Any]:
    target = resolve_project_path(root, path)
    result: dict[str, Any] = {
        "path": relative_project_path(root, target),
        "kind": kind,
        "available": target.is_file(),
    }
    if target.is_file():
        result["file_size"] = int(target.stat().st_size)
        result["sha256"] = _sha256(target)
    if (
        target.is_file()
        and target.suffix.lower() == ".pt"
        and kind in {"base_weight", "checkpoint"}
    ):
        from radar_ln3.checkpoint import Checkpoint

        checkpoint = Checkpoint.load(root, target)
        result["model_state_sha256"] = checkpoint.verify_state_hash()
    if task is not None:
        result["task"] = task
    return result


def register_product_artifacts(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    csv_descriptor: dict[str, Any] | None = None,
    overwrite: bool = False,
    manifest: DatasetManifest | None = None,
    feature_cache_descriptor: Mapping[str, Any] | None = None,
    feature_cache_key: str | None = None,
    feature_cache_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
    validate_manifest_payload: bool = True,
    event_cache_descriptor: Mapping[str, Any] | None = None,
    l3_calibration_descriptor: Mapping[str, Any] | None = None,
    l3_head_descriptors: Mapping[str, Mapping[str, Any]] | None = None,
    artifact_root: str | Path = "artifacts/radar_ln3",
) -> Path:
    """把正式配置、缓存、基础权重和输出路径登记到总 manifest。"""

    root = Path(project_root).resolve()
    payload = _load_manifest_payload(root, manifest_path)
    if validate_manifest_payload and manifest is None:
        validate_manifest(
            payload,
            root,
            verify_files=False,
            allow_unmaterialized=False,
        )
    existing_csv = (
        (payload.get("product") or {}).get("data_derivatives", {}).get("event_csv")
        if isinstance(payload.get("product"), Mapping)
        else None
    )
    csv_info = csv_descriptor or (
        dict(existing_csv)
        if isinstance(existing_csv, Mapping)
        else build_event_csv_index(
            root,
            manifest_path=manifest_path,
            overwrite=overwrite,
            manifest=manifest,
        )
    )
    expected_data_hash = data_manifest_hash(manifest or payload)
    if str(csv_info.get("source_manifest_sha256") or "") != expected_data_hash:
        raise DataFormatError(
            "事件 CSV 未绑定当前稳定 data manifest hash"
        )
    if event_cache_descriptor is not None:
        cache_info = dict(event_cache_descriptor)
        if str(cache_info.get("source_manifest_basis_sha256") or "") != expected_data_hash:
            raise DataFormatError("事件缓存未绑定当前稳定 data manifest hash")
    else:
        cache_info = dict((payload.get("audit") or {}).get("cache") or {})
    audit_payload = dict(payload.get("audit") or {})
    if cache_info:
        audit_payload["cache"] = cache_info
    # L3 派生状态只保留在 product.scope 一份；旧 manifest 的 audit.scope
    # 在下次注册时一并剔除。
    audit_payload.pop("scope", None)
    payload["audit"] = audit_payload
    previous_product = payload.get("product")
    previous_derivatives = (
        previous_product.get("data_derivatives", {})
        if isinstance(previous_product, Mapping)
        else {}
    )
    previous_feature_caches = (
        dict(previous_derivatives.get("feature_caches", {}))
        if isinstance(previous_derivatives, Mapping)
        and isinstance(previous_derivatives.get("feature_caches"), Mapping)
        else {}
    )
    legacy_feature_cache = previous_derivatives.get("feature_cache")
    if isinstance(legacy_feature_cache, Mapping):
        previous_feature_caches.setdefault("default", dict(legacy_feature_cache))
    previous_formal_outputs = (
        previous_product.get("formal_outputs", {})
        if isinstance(previous_product, Mapping)
        else {}
    )
    registered_l3 = (
        dict(l3_calibration_descriptor)
        if l3_calibration_descriptor is not None
        else dict(previous_formal_outputs.get("l3_calibration", {}))
        if isinstance(previous_formal_outputs, Mapping)
        and isinstance(previous_formal_outputs.get("l3_calibration"), Mapping)
        else None
    )
    previous_l3_heads = (
        dict(previous_formal_outputs.get("l3_heads", {}))
        if isinstance(previous_formal_outputs, Mapping)
        and isinstance(previous_formal_outputs.get("l3_heads"), Mapping)
        else {}
    )
    registered_l3_heads = (
        {str(key): dict(value) for key, value in l3_head_descriptors.items()}
        if l3_head_descriptors is not None
        else previous_l3_heads
    )
    if registered_l3 is not None:
        l3_path = resolve_project_path(
            root,
            str(registered_l3.get("path") or ""),
            must_exist=True,
        )
        declared_l3_hash = str(registered_l3.get("sha256") or "").lower()
        descriptor_hash_matches = (
            _is_sha256(declared_l3_hash) and declared_l3_hash == _sha256(l3_path)
        )
        if (
            registered_l3.get("kind") != "l3_calibration"
            or int(registered_l3.get("file_size", -1)) != int(l3_path.stat().st_size)
            or not descriptor_hash_matches
        ):
            raise DataFormatError("L3 calibration artifact descriptor 不匹配")
    artifact_base = Path(artifact_root).as_posix().strip("/")
    registries = {
        "product": {
            "schema_version": "RADAR-LN3-PRODUCT-REGISTRY-V1",
            "configs": {
                name: _file_descriptor(root, path, kind="configuration")
                for name, path in {
                    "source": "src/config/source.json",
                    "model": "src/config/model.json",
                    "runtime": "src/config/runtime.json",
                    "product": "src/config/radar_ln3.json",
                    "l3": "src/config/radar_ln3_l3.json",
                }.items()
            },
            "data_derivatives": {
                "event_csv": dict(csv_info),
                "feature_cache_root": {
                    "path": "data/cq08/cache/features",
                    "kind": "feature_cache_directory",
                    "available": (root / "data/cq08/cache/features").is_dir(),
                },
                "feature_cache_manifest": _file_descriptor(
                    root,
                    "data/cq08/cache/features/feature-cache-manifest.json",
                    kind="feature_cache_manifest",
                ),
                "event_source_cache": cache_info,
                "l2_baseline_anchor": _file_descriptor(
                    root,
                    "data/cq08/cache/l2-baseline-anchor.pt",
                    kind="event_baseline_logits_anchor",
                    task="l2",
                ),
            },
            "formal_outputs": {
                "l1": _file_descriptor(root, f"{artifact_base}/l1/l1.pt", kind="checkpoint", task="l1"),
                "l2": _file_descriptor(root, f"{artifact_base}/l2/l2.pt", kind="checkpoint", task="l2"),
                **({"l3_calibration": registered_l3} if registered_l3 is not None else {}),
                **({"l3_heads": registered_l3_heads} if registered_l3_heads else {}),
            },
            "scope": {
                "unknown_training_exposure": 0,
                "test_payload_access_count": 0,
                "l3": (
                    str(registered_l3.get("status", "exploratory"))
                    if registered_l3 is not None
                    else "not_in_scope"
                ),
            },
        }
    }

    descriptors: dict[str, Mapping[str, Any]] = dict(feature_cache_descriptors or {})
    if feature_cache_descriptor is not None:
        descriptors[str(feature_cache_key or "default")] = feature_cache_descriptor

    for cache_key, raw_descriptor in descriptors.items():
        descriptor = dict(raw_descriptor)
        if descriptor.get("schema_version") != "RADAR-LN3-FEATURE-CACHE-MANIFEST-V1":
            raise DataFormatError("feature cache manifest schema_version 不匹配")
        if not isinstance(descriptor.get("manifest_sha256"), str) or len(
            str(descriptor["manifest_sha256"])
        ) != 64:
            raise DataFormatError("feature cache manifest 自身 hash 缺失")
        if isinstance(descriptor.get("record_count"), bool) or not isinstance(
            descriptor.get("record_count"), int
        ) or int(descriptor["record_count"]) <= 0:
            raise DataFormatError("feature cache manifest record_count 无效")
        cache_manifest_path = resolve_project_path(
            root,
            str(descriptor.get("path") or ""),
            must_exist=True,
        )
        ensure_path_inside_root(
            cache_manifest_path,
            (root / "data" / "cq08" / "cache" / "features").resolve(),
        )
        if int(descriptor.get("file_size", -1)) != int(cache_manifest_path.stat().st_size):
            raise DataFormatError("feature cache manifest file_size 不一致")
        if str(descriptor.get("sha256") or "") != _sha256(cache_manifest_path):
            raise DataFormatError("feature cache manifest hash 不一致")
        try:
            cache_payload = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise DataFormatError("feature cache manifest 无法读取") from exc
        if not isinstance(cache_payload, dict):
            raise DataFormatError("feature cache manifest 根节点必须是对象")
        declared_cache_hash = cache_payload.pop("manifest_sha256", None)
        if declared_cache_hash != descriptor["manifest_sha256"]:
            raise DataFormatError("feature cache manifest hash 登记不一致")
        if canonical_json_hash(cache_payload) != declared_cache_hash:
            raise DataFormatError("feature cache manifest 自身内容 hash 不一致")
        previous_feature_caches[str(cache_key)] = descriptor
    if previous_feature_caches:
        registries["product"]["data_derivatives"]["feature_caches"] = {
            key: previous_feature_caches[key]
            for key in sorted(previous_feature_caches)
        }
        if "default" in previous_feature_caches:
            registries["product"]["data_derivatives"]["feature_cache"] = previous_feature_caches["default"]
    payload.update(registries)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_json_hash(payload)
    target = resolve_project_path(root, manifest_path)
    if target.exists() and not overwrite:
        current = _load_manifest_payload(root, manifest_path)
        if current.get("manifest_sha256") != payload["manifest_sha256"]:
            raise FileExistsError(f"拒绝覆盖内容不同的总 manifest：{target}")
    write_json(target, payload, overwrite=overwrite, label="总 manifest")
    return target


__all__ = ["CSV_PATH", "build_event_csv_index", "register_product_artifacts"]
