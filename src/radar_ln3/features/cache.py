"""特征缓存的轻量、可审计存储格式。

该缓存只保存已经计算好的 feature values/validity 和事件身份，不复制原始
MAT、DAT 或 RD payload。缓存身份必须绑定统一 manifest hash、feature config
hash 以及可选的 train-only normalization hash。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import (
    FeatureCacheDescriptor,
    FeatureCacheError,
    FeatureCacheFormatError,
    FeatureCacheIntegrityError,
    FeatureCachePathError,
    FeatureInputError,
    FeatureShapeError,
)


FEATURE_CACHE_SCHEMA_VERSION = "RADAR-LN3-FEATURE-CACHE-V2"
_CACHE_FIELDS = frozenset(
    {
        "schema_version",
        "feature_name",
        "source_manifest_hash",
        "feature_config_hash",
        "view_name",
        "view_hash",
        "normalization_hash",
        "feature_names_json",
        "sample_id",
        "parent_sample_id",
        "causal_event_index",
        "split_name",
        "values",
        "valid",
    }
)
_CACHE_METADATA_FIELD = "metadata_json"


def _scalar_string(value: Any, *, name: str) -> str:
    array = np.asarray(value)
    if array.ndim != 0:
        raise FeatureCacheFormatError(f"缓存字段 {name} 必须是标量")
    return str(array.item())


def _load_metadata(payload: Mapping[str, Any]) -> dict[str, Any]:
    """读取可选的缓存 metadata，拒绝非对象 JSON。"""

    value = payload.get(_CACHE_METADATA_FIELD)
    if value is None:
        return {}
    try:
        decoded = json.loads(_scalar_string(value, name=_CACHE_METADATA_FIELD))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise FeatureCacheFormatError("缓存 metadata_json 无效") from exc
    if not isinstance(decoded, dict):
        raise FeatureCacheFormatError("缓存 metadata_json 必须编码对象")
    return decoded


@dataclass(frozen=True)
class FeatureCache:
    descriptor: FeatureCacheDescriptor
    feature_names: tuple[str, ...]
    sample_ids: tuple[str, ...]
    parent_sample_ids: tuple[str, ...]
    causal_event_indices: np.ndarray
    split_names: tuple[str, ...]
    values: np.ndarray
    valid: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sample_positions: Mapping[str, int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        feature_names = tuple(str(value) for value in self.feature_names)
        sample_ids = tuple(str(value) for value in self.sample_ids)
        parent_ids = tuple(str(value) for value in self.parent_sample_ids)
        split_names = tuple(str(value) for value in self.split_names)
        indices = np.asarray(self.causal_event_indices)
        values = np.asarray(self.values)
        valid = np.asarray(self.valid)
        count = len(sample_ids)
        if not feature_names or len(feature_names) != len(set(feature_names)):
            raise FeatureCacheFormatError("feature_names 必须非空且唯一")
        if count == 0 or count != len(set(sample_ids)):
            raise FeatureCacheIntegrityError("缓存 sample_id 必须非空且唯一")
        object.__setattr__(
            self,
            "sample_positions",
            {sample_id: index for index, sample_id in enumerate(sample_ids)},
        )
        if len(parent_ids) != count or len(split_names) != count:
            raise FeatureCacheIntegrityError("缓存事件身份字段长度不一致")
        if any(not value for value in parent_ids + split_names):
            raise FeatureCacheFormatError("缓存 parent/split 字段不能为空")
        if indices.dtype.kind not in "iu" or indices.shape != (count,) or np.any(indices < 0):
            raise FeatureCacheFormatError("causal_event_index shape/dtype 无效")
        if values.dtype != np.float32 or values.ndim < 2 or values.shape[0] != count:
            raise FeatureShapeError("缓存 values 必须是 [N,D] float32")
        if values.shape[-1] != len(feature_names) or not np.isfinite(values).all():
            raise FeatureCacheFormatError("缓存 values 宽度或有限值契约无效")
        valid_shapes = {(count,), values.shape, values.shape[:-1]}
        if valid.dtype != np.bool_ or valid.shape not in valid_shapes:
            raise FeatureShapeError(
                "缓存 valid 必须是事件、帧或与 values 同 shape 的 bool"
            )
        if valid.shape == values.shape and np.any(values[~valid] != 0.0):
            raise FeatureCacheIntegrityError("缓存无效 feature value 必须严格为零")
        if valid.shape == values.shape[:-1] and np.any(values[~valid] != 0.0):
            raise FeatureCacheIntegrityError("缓存无效帧 feature value 必须严格为零")
        if valid.shape == (count,) and np.any(values[~valid] != 0.0):
            raise FeatureCacheIntegrityError("缓存 invalid event value 必须严格为零")
        if not isinstance(self.metadata, Mapping):
            raise FeatureCacheFormatError("缓存 metadata 必须是对象")
        by_parent: dict[str, list[int]] = {}
        split_by_parent: dict[str, str] = {}
        seen_parent_ids: set[str] = set()
        previous_parent_id: str | None = None
        for parent_id, event_index in zip(parent_ids, indices.tolist(), strict=True):
            by_parent.setdefault(parent_id, []).append(int(event_index))
        for parent_id, split_name in zip(parent_ids, split_names, strict=True):
            previous_split = split_by_parent.setdefault(parent_id, split_name)
            if previous_split != split_name:
                raise FeatureCacheIntegrityError(
                    f"缓存 parent 跨越多个 split：{parent_id}"
                )
            if previous_parent_id is not None and parent_id != previous_parent_id:
                seen_parent_ids.add(previous_parent_id)
            if parent_id in seen_parent_ids:
                raise FeatureCacheIntegrityError(
                    f"缓存 parent 事件必须连续排列：{parent_id}"
                )
            previous_parent_id = parent_id
        for parent_id, event_indices in by_parent.items():
            if event_indices != list(range(len(event_indices))):
                raise FeatureCacheIntegrityError(
                    f"缓存 parent 内 event index 必须从 0 连续排列：{parent_id}"
                )
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "sample_ids", sample_ids)
        object.__setattr__(self, "parent_sample_ids", parent_ids)
        object.__setattr__(self, "split_names", split_names)


def _cache_payload(cache: FeatureCache) -> dict[str, Any]:
    descriptor = cache.descriptor
    payload = {
        "schema_version": np.asarray(descriptor.schema_version),
        "feature_name": np.asarray(descriptor.feature_name),
        "source_manifest_hash": np.asarray(descriptor.source_manifest_hash),
        "feature_config_hash": np.asarray(descriptor.feature_config_hash),
        "view_name": np.asarray(descriptor.view_name),
        "view_hash": np.asarray(descriptor.view_hash),
        "normalization_hash": np.asarray(descriptor.normalization_hash or ""),
        "feature_names_json": np.asarray(
            json.dumps(list(cache.feature_names), ensure_ascii=False, separators=(",", ":"))
        ),
        "sample_id": np.asarray(cache.sample_ids),
        "parent_sample_id": np.asarray(cache.parent_sample_ids),
        "causal_event_index": np.asarray(cache.causal_event_indices, dtype=np.int64),
        "split_name": np.asarray(cache.split_names),
        "values": np.asarray(cache.values, dtype=np.float32),
        "valid": np.asarray(cache.valid, dtype=np.bool_),
    }
    if cache.metadata:
        try:
            payload[_CACHE_METADATA_FIELD] = np.asarray(
                json.dumps(
                    dict(cache.metadata),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        except (TypeError, ValueError) as exc:
            raise FeatureCacheFormatError("缓存 metadata 无法规范化") from exc
    return payload


def _cache_equal(left: FeatureCache, right: FeatureCache) -> bool:
    """比较 cache 内容，支持重复构建时安全复用同一文件。"""

    return (
        left.descriptor.to_dict() == right.descriptor.to_dict()
        and left.feature_names == right.feature_names
        and left.sample_ids == right.sample_ids
        and left.parent_sample_ids == right.parent_sample_ids
        and left.split_names == right.split_names
        and np.array_equal(left.causal_event_indices, right.causal_event_indices)
        and np.array_equal(left.values, right.values)
        and np.array_equal(left.valid, right.valid)
        and dict(left.metadata) == dict(right.metadata)
    )


def save_feature_cache(
    path: str | Path,
    cache: FeatureCache,
    *,
    overwrite: bool = False,
) -> Path:
    """原子保存一个 feature cache；默认禁止覆盖已有文件。"""

    # TODO: 后续由上层统一日志系统记录 cache 的构建、复用和 hash 校验过程；
    # 当前保持纯函数式文件接口，不在特征层直接输出处理进度。
    target = Path(path).resolve()
    if target.exists() and not overwrite:
        try:
            existing = load_feature_cache(target)
        except FeatureCacheError as exc:
            raise FeatureCachePathError(
                f"已有 feature cache 无法校验，拒绝覆盖：{target}"
            ) from exc
        if _cache_equal(existing, cache):
            return target
        raise FeatureCachePathError(f"feature cache 内容不同，未允许覆盖：{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _cache_payload(cache)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", dir=target.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, target)
        else:
            try:
                os.link(temporary_path, target)
            except FileExistsError:
                existing = load_feature_cache(target)
                if _cache_equal(existing, cache):
                    return target
                raise FeatureCachePathError(
                    f"feature cache 在写入前出现了不同内容：{target}"
                )
        temporary_path = None
    except OSError as exc:
        raise FeatureCachePathError(f"feature cache 保存失败：{target}") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target


def load_feature_cache(
    path: str | Path,
    *,
    expected_schema_version: str = FEATURE_CACHE_SCHEMA_VERSION,
    expected_feature_name: str | None = None,
    expected_source_manifest_hash: str | None = None,
    expected_feature_config_hash: str | None = None,
    expected_view_name: str | None = None,
    expected_view_hash: str | None = None,
    expected_normalization_hash: str | None = None,
    expected_sample_ids: Sequence[str] | None = None,
    expected_split_name: str | None = None,
    allow_superset: bool = False,
) -> FeatureCache:
    target = Path(path).resolve()
    if not target.is_file():
        raise FeatureCachePathError(f"feature cache 不存在：{target}")
    try:
        with np.load(target, allow_pickle=False) as payload:
            actual_fields = frozenset(payload.files)
            allowed_fields = _CACHE_FIELDS | {_CACHE_METADATA_FIELD}
            if not _CACHE_FIELDS.issubset(actual_fields) or not actual_fields.issubset(allowed_fields):
                missing = sorted(_CACHE_FIELDS - actual_fields)
                extra = sorted(actual_fields - allowed_fields)
                raise FeatureCacheFormatError(
                    f"feature cache 字段不一致：missing={missing}, extra={extra}"
                )
            schema_version = _scalar_string(payload["schema_version"], name="schema_version")
            feature_name = _scalar_string(payload["feature_name"], name="feature_name")
            source_hash = _scalar_string(payload["source_manifest_hash"], name="source_manifest_hash")
            config_hash = _scalar_string(payload["feature_config_hash"], name="feature_config_hash")
            view_name = _scalar_string(payload["view_name"], name="view_name")
            view_hash = _scalar_string(payload["view_hash"], name="view_hash")
            normalization_value = _scalar_string(payload["normalization_hash"], name="normalization_hash")
            try:
                feature_names = tuple(
                    str(value)
                    for value in json.loads(
                        _scalar_string(payload["feature_names_json"], name="feature_names_json")
                    )
                )
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise FeatureCacheFormatError("feature_names_json 无效") from exc
            cache = FeatureCache(
                descriptor=FeatureCacheDescriptor(
                    schema_version=schema_version,
                    feature_name=feature_name,
                    source_manifest_hash=source_hash,
                    feature_config_hash=config_hash,
                    view_name=view_name,
                    view_hash=view_hash,
                    normalization_hash=normalization_value or None,
                ),
                feature_names=feature_names,
                sample_ids=tuple(str(value) for value in payload["sample_id"].tolist()),
                parent_sample_ids=tuple(
                    str(value) for value in payload["parent_sample_id"].tolist()
                ),
                causal_event_indices=np.asarray(payload["causal_event_index"], dtype=np.int64),
                split_names=tuple(str(value) for value in payload["split_name"].tolist()),
                values=np.asarray(payload["values"], dtype=np.float32),
                valid=np.asarray(payload["valid"], dtype=np.bool_),
                metadata=_load_metadata(payload),
            )
    except FeatureCacheError:
        raise
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise FeatureCacheFormatError(f"feature cache 无法读取：{target}") from exc
    _validate_cache_expectations(
        cache,
        expected_schema_version=expected_schema_version,
        expected_feature_name=expected_feature_name,
        expected_source_manifest_hash=expected_source_manifest_hash,
        expected_feature_config_hash=expected_feature_config_hash,
        expected_view_name=expected_view_name,
        expected_view_hash=expected_view_hash,
        expected_normalization_hash=expected_normalization_hash,
        expected_sample_ids=expected_sample_ids,
        expected_split_name=expected_split_name,
        allow_superset=allow_superset,
    )
    return cache


def _validate_cache_expectations(
    cache: FeatureCache,
    *,
    expected_schema_version: str | None,
    expected_feature_name: str | None,
    expected_source_manifest_hash: str | None,
    expected_feature_config_hash: str | None,
    expected_view_name: str | None,
    expected_view_hash: str | None,
    expected_normalization_hash: str | None,
    expected_sample_ids: Sequence[str] | None,
    expected_split_name: str | None,
    allow_superset: bool,
) -> None:
    descriptor = cache.descriptor
    if expected_schema_version is not None and descriptor.schema_version != expected_schema_version:
        raise FeatureCacheIntegrityError("feature cache schema_version 不匹配")
    if expected_feature_name is not None and descriptor.feature_name != expected_feature_name:
        raise FeatureCacheIntegrityError("feature cache feature_name 不匹配")
    if expected_source_manifest_hash is not None and descriptor.source_manifest_hash != expected_source_manifest_hash:
        raise FeatureCacheIntegrityError("feature cache source manifest hash 不匹配")
    if expected_feature_config_hash is not None and descriptor.feature_config_hash != expected_feature_config_hash:
        raise FeatureCacheIntegrityError("feature cache feature config hash 不匹配")
    if expected_view_name is not None and descriptor.view_name != expected_view_name:
        raise FeatureCacheIntegrityError("feature cache view_name 不匹配")
    if expected_view_hash is not None and descriptor.view_hash != expected_view_hash:
        raise FeatureCacheIntegrityError("feature cache view_hash 不匹配")
    actual_normalization_hash = descriptor.normalization_hash
    # normalization hash 是可选的期望约束。调用方未提供期望值时，
    # 允许缓存保留自身的归一化来源信息；只有显式绑定时才比较。
    if (
        expected_normalization_hash is not None
        and expected_normalization_hash != actual_normalization_hash
    ):
        raise FeatureCacheIntegrityError("feature cache normalization hash 不匹配")
    if expected_sample_ids is not None:
        expected = tuple(str(value) for value in expected_sample_ids)
        if allow_superset:
            if not set(expected).issubset(set(cache.sample_ids)):
                raise FeatureCacheIntegrityError("feature cache 缺少请求的 sample_id")
        elif cache.sample_ids != expected:
            raise FeatureCacheIntegrityError("feature cache sample_id 顺序或集合不一致")
    if expected_split_name is not None and any(
        value != str(expected_split_name) for value in cache.split_names
    ):
        raise FeatureCacheIntegrityError("feature cache split_name 不一致")


def feature_cache_sha256(path: str | Path) -> str:
    target = Path(path).resolve()
    if not target.is_file():
        raise FeatureCachePathError(f"feature cache 不存在：{target}")
    digest = hashlib.sha256()
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_feature_cache_from_records(
    records: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    valid: np.ndarray,
    *,
    descriptor: FeatureCacheDescriptor,
    feature_names: Sequence[str],
    metadata: Mapping[str, Any] | None = None,
) -> FeatureCache:
    """从统一 manifest 的事件记录构造严格对齐的 feature cache。"""

    if not records:
        raise FeatureInputError("feature cache records 不能为空")
    required = {"sample_id", "parent_sample_id", "causal_event_index", "split_name"}
    for record in records:
        if not required.issubset(record):
            raise FeatureCacheFormatError("feature cache record 缺少身份字段")
    cache = FeatureCache(
        descriptor=descriptor,
        feature_names=tuple(feature_names),
        sample_ids=tuple(str(record["sample_id"]) for record in records),
        parent_sample_ids=tuple(str(record["parent_sample_id"]) for record in records),
        causal_event_indices=np.asarray(
            [int(record["causal_event_index"]) for record in records], dtype=np.int64
        ),
        split_names=tuple(str(record["split_name"]) for record in records),
        values=np.asarray(values, dtype=np.float32),
        valid=np.asarray(valid, dtype=np.bool_),
        metadata=dict(metadata or {}),
    )
    return cache


__all__ = [
    "FEATURE_CACHE_SCHEMA_VERSION",
    "FeatureCache",
    "build_feature_cache_from_records",
    "feature_cache_sha256",
    "load_feature_cache",
    "save_feature_cache",
]
