"""按统一 DatasetView 构建和加载可审计的特征缓存。

本模块只编排通用 EventSequence、特征编码器和 FeatureCache。数据集适配、
raw-IQ frame 定位以及 source path 解析仍由 data 层和调用方注入的 provider
负责。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from radar_ln3.data.contracts import DatasetView, EventSequence, RawEchoProvider
from radar_ln3.data.paths import ensure_path_inside_root, relative_project_path, resolve_project_path

from .cache import (
    FEATURE_CACHE_SCHEMA_VERSION,
    FeatureCache,
    build_feature_cache_from_records,
    feature_cache_sha256,
    load_feature_cache,
    save_feature_cache,
)
from .contracts import (
    FeatureCacheDescriptor,
    FeatureCacheFormatError,
    FeatureInputError,
)
from .course import COURSE_FEATURE_NAME, encode_causal_course_sequence
from .micro_doppler import (
    BROADBAND_CANDIDATE_SAMPLING_MODES,
    MICRO_DOPPLER_FRAME_CAP,
    MICRO_DOPPLER_FRAME_NAMES,
    MICRO_DOPPLER_FRAME_VALUE_DIM,
    STABLE8_INDICES,
    STABLE8_NAMES,
    extract_current_event_broadband_micro_doppler,
    fit_parent_equal_median_iqr,
    normalize_stable8,
    stable8_normalization_hash,
)
from .motion import (
    TR_OBSERVATION_INCREMENT_NAMES,
    encode_causal_tr_observation_increment_sequence,
)
from .physical import (
    RD_PHYSICAL_NAMES,
    extract_current_event_rd_physical,
)
from .rd import (
    RD_QUALITY_NAMES,
    encode_rd_quality_from_event,
    prepare_rd_event,
    resolve_rd_velocity_axis,
)
from .tr import TR_VALUE_NAMES, encode_tr_sequence


FEATURE_CACHE_MANIFEST_SCHEMA = "RADAR-LN3-FEATURE-CACHE-MANIFEST-V1"
DEFAULT_FEATURE_CACHE_MANIFEST_NAME = "feature-cache-manifest.json"
FEATURE_CACHE_FILE_NAMES = {
    "tr_observation": "tr-observation.npz",
    "tr_observation_increment": "tr-observation-increment.npz",
    "course_circular_incoherence": "course-circular-incoherence.npz",
    "rd_quality": "rd-quality.npz",
    "rd_physical4": "rd-physical4.npz",
    "micro_doppler_evidence": "micro-doppler-evidence.npz",
    "micro_doppler_stable8": "micro-doppler-stable8.npz",
    "micro_doppler_stable8_frame": "micro-doppler-stable8-frame.npz",
}


ProgressCallback = Callable[[Mapping[str, Any]], None]


def _tr_value_validity(missing: np.ndarray) -> np.ndarray:
    """将 11 维原始字段缺失事实展开到 14 维编码值。"""

    mask = np.asarray(missing, dtype=np.bool_)
    if mask.ndim != 2 or mask.shape[1] != 11:
        raise FeatureInputError("TR missing mask shape 无效")
    result = np.ones((mask.shape[0], len(TR_VALUE_NAMES)), dtype=np.bool_)
    result[:, 0] = ~mask[:, 0]
    result[:, 1] = ~mask[:, 1]
    result[:, 2:4] = ~mask[:, 2, None]
    result[:, 4:6] = ~mask[:, 3, None]
    result[:, 6:8] = ~mask[:, 4, None]
    result[:, 8] = ~mask[:, 5]
    result[:, 9] = ~mask[:, 6]
    result[:, 10] = ~mask[:, 7]
    result[:, 11] = ~mask[:, 8]
    result[:, 12] = ~mask[:, 9]
    result[:, 13] = ~mask[:, 10]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_write(
    target: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    """以原子方式写入 bundle manifest，且默认拒绝竞态覆盖。"""

    encoded = (
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if not target.is_file():
            raise FeatureCacheFormatError(f"feature cache manifest 不是文件：{target}")
        if target.read_bytes() == encoded:
            return
        if not overwrite:
            raise FileExistsError(f"拒绝覆盖内容不同的 feature cache manifest：{target}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, target)
            return
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.is_file() and target.read_bytes() == encoded:
                return
            raise FileExistsError(
                f"拒绝覆盖竞态产生的 feature cache manifest：{target}"
            )
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _ordered_records(
    records: Sequence[Mapping[str, Any]],
    view: DatasetView,
) -> list[Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not sample_id or sample_id in by_id:
            raise FeatureInputError(f"feature cache records 存在重复或空 sample_id：{sample_id}")
        by_id[sample_id] = record
    result: list[Mapping[str, Any]] = []
    for sample_id in view.sample_ids:
        record = by_id.get(str(sample_id))
        if record is None:
            raise FeatureInputError(f"统一 view 缺少 feature cache record：{sample_id}")
        if str(record.get("parent_sample_id", "")) not in view.event_ids_by_parent:
            raise FeatureInputError(f"feature cache record parent 不属于 view：{sample_id}")
        result.append(record)
    return result


def _validate_event_order(
    sequence: EventSequence,
    *,
    expected_parent_id: str,
    expected_sample_ids: Sequence[str],
) -> None:
    if str(sequence.parent_sample_id) != str(expected_parent_id):
        raise FeatureInputError(
            "feature cache sequence parent 顺序与统一 view 不一致："
            f"expected={expected_parent_id}, actual={sequence.parent_sample_id}"
        )
    actual = tuple(str(event.sample_id) for event in sequence.events)
    expected = tuple(str(value) for value in expected_sample_ids)
    if actual != expected:
        raise FeatureInputError(
            f"feature cache event 顺序与统一 view 不一致：{expected_parent_id}"
        )
    indices = [int(event.causal_event_index) for event in sequence.events]
    if indices != list(range(len(indices))):
        raise FeatureInputError(f"feature cache event index 不连续：{expected_parent_id}")


def _record_shape_metadata(cache: FeatureCache) -> dict[str, Any]:
    return {
        "record_count": len(cache.sample_ids),
        "value_shape": list(cache.values.shape),
        "valid_shape": list(cache.valid.shape),
        "feature_names": list(cache.feature_names),
        "descriptor": cache.descriptor.to_dict(),
    }


def build_feature_cache_bundle(
    sequences: Iterable[EventSequence],
    records: Sequence[Mapping[str, Any]],
    *,
    view: DatasetView,
    source_manifest_hash: str,
    feature_config: Mapping[str, Any] | None = None,
    raw_echo_provider: RawEchoProvider | None = None,
    stable_median: np.ndarray | None = None,
    stable_scale: np.ndarray | None = None,
    require_stable_evidence: bool = True,
    max_rd_frames: int = MICRO_DOPPLER_FRAME_CAP,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = 5,
) -> dict[str, FeatureCache]:
    """按一个已解析 view 构建全部通用 feature cache。"""

    if not str(source_manifest_hash):
        raise FeatureInputError("feature cache source_manifest_hash 不能为空")
    if isinstance(progress_interval, bool) or int(progress_interval) <= 0:
        raise FeatureInputError("feature cache progress_interval 必须是正整数")
    if isinstance(max_rd_frames, bool) or int(max_rd_frames) != MICRO_DOPPLER_FRAME_CAP:
        raise FeatureInputError(
            f"当前 stable8 frame cap 固定为 {MICRO_DOPPLER_FRAME_CAP}"
        )
    if require_stable_evidence and raw_echo_provider is None:
        raise FeatureInputError("构建 stable8 cache 必须注入 raw_echo_provider")
    if (stable_median is None) != (stable_scale is None):
        raise FeatureInputError("stable8 median 和 scale 必须同时提供")

    ordered_records = _ordered_records(records, view)
    record_by_id = {
        str(record["sample_id"]): record for record in ordered_records
    }
    feature_config_value = dict(feature_config or {})
    feature_config_value.setdefault(
        "rd_processing_contract", "radar_ln3-canonical-rd-v1-prefer-local"
    )
    feature_config_value.setdefault("data_binding", view.data_binding)
    feature_config_value.setdefault(
        "micro_doppler_sampling_modes",
        [list(mode) for mode in BROADBAND_CANDIDATE_SAMPLING_MODES],
    )

    tr_values_rows: list[np.ndarray] = []
    tr_valid_rows: list[np.ndarray] = []
    increment_rows: list[np.ndarray] = []
    increment_valid_rows: list[np.ndarray] = []
    course_rows: list[np.ndarray] = []
    course_valid_rows: list[np.ndarray] = []
    course_support_rows: list[int] = []
    quality_rows: list[np.ndarray] = []
    physical_rows: list[np.ndarray] = []
    physical_valid_rows: list[bool] = []
    evidence_rows: list[np.ndarray] = []
    evidence_frame_mask_rows: list[np.ndarray] = []
    evidence_reliable_rows: list[np.ndarray] = []
    stable_summary_rows: list[np.ndarray] = []
    stable_summary_valid_rows: list[bool] = []
    stable_frame_rows: list[np.ndarray] = []
    stable_frame_valid_rows: list[np.ndarray] = []
    parent_ids: list[str] = []
    event_objects: list[Any] = []

    total_parents = len(view.parent_ids)
    total_events = len(view.sample_ids)
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "feature_cache_start",
                "view_name": view.view_name,
                "view_hash": view.view_hash,
                "parent_count": total_parents,
                "record_count": total_events,
            }
        )

    expected_parent_position = 0
    for sequence in sequences:
        if expected_parent_position >= len(view.parent_ids):
            raise FeatureInputError("feature cache sequence 数量超过统一 view")
        parent_id = str(view.parent_ids[expected_parent_position])
        expected_ids = view.event_ids_by_parent[parent_id]
        _validate_event_order(
            sequence,
            expected_parent_id=parent_id,
            expected_sample_ids=expected_ids,
        )
        event_values = [
            dict(event.tr_values)
            if isinstance(event.tr_values, Mapping) and event.tr_available
            else {}
            for event in sequence.events
        ]
        encoded_tr, missing = encode_tr_sequence(event_values)
        increments, increment_valid = encode_causal_tr_observation_increment_sequence(
            event_values
        )
        course, course_valid, course_support = encode_causal_course_sequence(
            event_values
        )
        tr_values_rows.extend(encoded_tr)
        tr_valid_rows.extend(_tr_value_validity(missing))
        increment_rows.extend(increments)
        increment_valid_rows.extend(increment_valid)
        course_rows.extend(course)
        course_valid_rows.extend(course_valid)
        course_support_rows.extend(int(value) for value in course_support.tolist())

        for event in sequence.events:
            prepared = prepare_rd_event(
                event.rd_values if event.rd_available else None,
                max_frames=None,
            )
            quality_rows.append(
                encode_rd_quality_from_event(
                    prepared,
                    max_rd_frames_per_event=int(max_rd_frames),
                )
            )
            if prepared is None:
                physical_rows.append(np.zeros((4,), dtype=np.float32))
                physical_valid_rows.append(False)
            else:
                physical = extract_current_event_rd_physical(
                    prepared.amplitude,
                    resolve_rd_velocity_axis(event.rd_values),
                    frame_mask=prepared.frame_mask,
                )
                physical_rows.append(physical.values)
                physical_valid_rows.append(bool(physical.valid))

            if raw_echo_provider is None:
                evidence = np.zeros(
                    (MICRO_DOPPLER_FRAME_CAP, MICRO_DOPPLER_FRAME_VALUE_DIM),
                    dtype=np.float32,
                )
                frame_mask = np.zeros((MICRO_DOPPLER_FRAME_CAP,), dtype=np.bool_)
                reliable = np.zeros((MICRO_DOPPLER_FRAME_CAP,), dtype=np.bool_)
                summary = np.zeros((len(STABLE8_NAMES),), dtype=np.float32)
                summary_valid = False
            elif not event.rd_available:
                evidence = np.zeros(
                    (MICRO_DOPPLER_FRAME_CAP, MICRO_DOPPLER_FRAME_VALUE_DIM),
                    dtype=np.float32,
                )
                frame_mask = np.zeros((MICRO_DOPPLER_FRAME_CAP,), dtype=np.bool_)
                reliable = np.zeros((MICRO_DOPPLER_FRAME_CAP,), dtype=np.bool_)
                summary = np.zeros((len(STABLE8_NAMES),), dtype=np.float32)
                summary_valid = False
            else:
                raw_frames = raw_echo_provider.read_event_frames(event)
                micro_event = extract_current_event_broadband_micro_doppler(
                    raw_frames,
                    event_index=int(event.causal_event_index),
                )
                evidence = np.asarray(micro_event.frame_values, dtype=np.float32)
                frame_mask = np.asarray(micro_event.frame_mask, dtype=np.bool_)
                reliable = np.asarray(
                    micro_event.frame_feature_reliable,
                    dtype=np.bool_,
                )
                summary = np.asarray(micro_event.values, dtype=np.float32)
                summary_valid = bool(micro_event.valid)
            evidence_rows.append(evidence)
            evidence_frame_mask_rows.append(frame_mask)
            evidence_reliable_rows.append(reliable)
            stable_summary_rows.append(summary)
            stable_summary_valid_rows.append(summary_valid)
            stable_frame_rows.append(evidence[:, STABLE8_INDICES])
            stable_frame_valid_rows.append(reliable)
            parent_ids.append(parent_id)
            event_objects.append(event)
        expected_parent_position += 1
        if progress_callback is not None and (
            expected_parent_position % int(progress_interval) == 0
            or expected_parent_position == total_parents
        ):
            progress_callback(
                {
                    "stage": "feature_cache_progress",
                    "view_name": view.view_name,
                    "processed": expected_parent_position,
                    "total": total_parents,
                    "parent_sample_id": parent_id,
                    "event_count": len(sequence.events),
                    "record_count": len(parent_ids),
                }
            )

    if expected_parent_position != len(view.parent_ids):
        raise FeatureInputError(
            "feature cache sequence 未覆盖统一 view 的全部 parent："
            f"expected={len(view.parent_ids)}, actual={expected_parent_position}"
        )
    if len(parent_ids) != len(view.sample_ids):
        raise FeatureInputError("feature cache event 数量与统一 view 不一致")
    if tuple(str(event.sample_id) for event in event_objects) != tuple(view.sample_ids):
        raise FeatureInputError("feature cache event 身份与统一 view 不一致")
    for event, parent_id in zip(event_objects, parent_ids, strict=True):
        record = record_by_id[event.sample_id]
        if (
            str(record["parent_sample_id"]) != parent_id
            or int(record["causal_event_index"]) != int(event.causal_event_index)
            or str(record["split_name"]) != str(event.split_name)
        ):
            raise FeatureInputError(
                f"feature cache record 与事件身份不一致：{event.sample_id}"
            )

    stable_summary_raw = np.asarray(stable_summary_rows, dtype=np.float32)
    stable_summary_valid = np.asarray(stable_summary_valid_rows, dtype=np.bool_)
    stable_frame_raw = np.asarray(stable_frame_rows, dtype=np.float32)
    stable_frame_valid = np.asarray(stable_frame_valid_rows, dtype=np.bool_)
    normalization = None
    if raw_echo_provider is not None:
        if stable_median is None:
            train_mask = np.asarray(
                [str(record["split_name"]) == "train" for record in ordered_records],
                dtype=np.bool_,
            )
            normalization = fit_parent_equal_median_iqr(
                stable_summary_raw[train_mask],
                stable_summary_valid[train_mask],
                np.asarray(parent_ids, dtype=str)[train_mask].tolist(),
                feature_names=STABLE8_NAMES,
            )
            stable_median = normalization.median
            stable_scale = normalization.scale
        else:
            stable_median = np.asarray(stable_median, dtype=np.float32)
            stable_scale = np.asarray(stable_scale, dtype=np.float32)
        stable_summary_values = normalize_stable8(
            stable_summary_raw,
            stable_summary_valid,
            median=stable_median,
            scale=stable_scale,
        )
        stable_frame_values = normalize_stable8(
            stable_frame_raw,
            stable_frame_valid,
            median=stable_median,
            scale=stable_scale,
        )
        normalization_hash = stable8_normalization_hash(
            stable_median,
            stable_scale,
        )
    else:
        stable_summary_values = np.zeros_like(stable_summary_raw, dtype=np.float32)
        stable_frame_values = np.zeros_like(stable_frame_raw, dtype=np.float32)
        normalization_hash = None
        if stable_median is None and any(
            str(record["split_name"]) != "train" for record in ordered_records
        ):
            raise FeatureInputError(
                "未提供 stable8 normalization 时，只能从 train view 拟合归一化参数"
            )

    source_records = ordered_records
    common_metadata = {
        "view_name": view.view_name,
        "view_hash": view.view_hash,
        "source_manifest_hash": str(source_manifest_hash),
        "data_binding": view.data_binding,
        "rd_processing_contract": "radar_ln3-canonical-rd-v1-prefer-local",
    }

    def make_cache(
        feature_name: str,
        values: np.ndarray,
        valid: np.ndarray,
        feature_names: Sequence[str],
        *,
        metadata: Mapping[str, Any] | None = None,
        use_normalization: bool = False,
    ) -> FeatureCache:
        config_hash = hashlib.sha256(
            json.dumps(
                {"feature_name": feature_name, "config": feature_config_value},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        descriptor = FeatureCacheDescriptor(
            schema_version=FEATURE_CACHE_SCHEMA_VERSION,
            feature_name=feature_name,
            source_manifest_hash=str(source_manifest_hash),
            feature_config_hash=config_hash,
            view_name=view.view_name,
            view_hash=view.view_hash,
            normalization_hash=normalization_hash if use_normalization else None,
        )
        return build_feature_cache_from_records(
            source_records,
            np.asarray(values, dtype=np.float32),
            np.asarray(valid, dtype=np.bool_),
            descriptor=descriptor,
            feature_names=feature_names,
            metadata={**common_metadata, **dict(metadata or {})},
        )

    caches = {
        "tr_observation": make_cache(
            "tr_observation",
            np.asarray(tr_values_rows, dtype=np.float32),
            np.asarray(tr_valid_rows, dtype=np.bool_),
            TR_VALUE_NAMES,
        ),
        "tr_observation_increment": make_cache(
            "tr_observation_increment",
            np.asarray(increment_rows, dtype=np.float32),
            np.asarray(increment_valid_rows, dtype=np.bool_),
            TR_OBSERVATION_INCREMENT_NAMES,
        ),
        "course_circular_incoherence": make_cache(
            "course_circular_incoherence",
            np.asarray(course_rows, dtype=np.float32),
            np.asarray(course_valid_rows, dtype=np.bool_),
            (COURSE_FEATURE_NAME,),
            metadata={"course_support": course_support_rows},
        ),
        "rd_quality": make_cache(
            "rd_quality",
            np.asarray(quality_rows, dtype=np.float32),
            np.ones((len(quality_rows), len(RD_QUALITY_NAMES)), dtype=np.bool_),
            RD_QUALITY_NAMES,
        ),
        "rd_physical4": make_cache(
            "rd_physical4",
            np.asarray(physical_rows, dtype=np.float32),
            np.asarray(physical_valid_rows, dtype=np.bool_),
            RD_PHYSICAL_NAMES,
        ),
        "micro_doppler_evidence": make_cache(
            "micro_doppler_evidence",
            np.asarray(evidence_rows, dtype=np.float32),
            np.asarray(evidence_frame_mask_rows, dtype=np.bool_),
            MICRO_DOPPLER_FRAME_NAMES,
            metadata={
                "frame_feature_reliable": np.asarray(
                    evidence_reliable_rows, dtype=np.bool_
                ).tolist(),
                "frame_cap": MICRO_DOPPLER_FRAME_CAP,
            },
        ),
        "micro_doppler_stable8": make_cache(
            "micro_doppler_stable8",
            stable_summary_values,
            stable_summary_valid,
            STABLE8_NAMES,
            use_normalization=True,
        ),
        "micro_doppler_stable8_frame": make_cache(
            "micro_doppler_stable8_frame",
            stable_frame_values,
            stable_frame_valid,
            STABLE8_NAMES,
            use_normalization=True,
            metadata={"frame_cap": MICRO_DOPPLER_FRAME_CAP},
        ),
    }
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "feature_cache_complete",
                "view_name": view.view_name,
                "view_hash": view.view_hash,
                "parent_count": total_parents,
                "record_count": total_events,
            }
        )
    return caches


def save_feature_cache_bundle(
    project_root: str | Path,
    caches: Mapping[str, FeatureCache],
    *,
    output_root: str | Path,
    manifest_name: str = DEFAULT_FEATURE_CACHE_MANIFEST_NAME,
    overwrite: bool = False,
) -> dict[str, Any]:
    """将一组 cache 写入 data 边界内，并生成 bundle manifest。"""

    root = Path(project_root).resolve()
    target_root = resolve_project_path(root, output_root)
    ensure_path_inside_root(target_root, (root / "data").resolve())
    if not target_root.is_dir() and target_root.exists():
        raise FeatureCacheFormatError(f"feature cache 输出根不是目录：{target_root}")
    if not str(manifest_name) or Path(manifest_name).name != str(manifest_name):
        raise FeatureCacheFormatError("feature cache manifest_name 必须是单层文件名")
    if not caches:
        raise FeatureInputError("feature cache bundle 不能为空")
    manifest_path = target_root / str(manifest_name)
    ensure_path_inside_root(manifest_path, target_root)
    files: dict[str, Any] = {}
    for feature_name in sorted(caches):
        cache = caches[feature_name]
        if feature_name not in FEATURE_CACHE_FILE_NAMES:
            raise FeatureCacheFormatError(f"未登记的 feature cache：{feature_name}")
        path = target_root / FEATURE_CACHE_FILE_NAMES[feature_name]
        saved = save_feature_cache(path, cache, overwrite=overwrite)
        files[feature_name] = {
            "path": relative_project_path(root, saved),
            "file_size": int(saved.stat().st_size),
            "sha256": feature_cache_sha256(saved),
            **_record_shape_metadata(cache),
        }
    first = next(iter(caches.values()))
    payload = {
        "schema_version": FEATURE_CACHE_MANIFEST_SCHEMA,
        "cache_schema_version": FEATURE_CACHE_SCHEMA_VERSION,
        "source_manifest_hash": first.descriptor.source_manifest_hash,
        "view_name": first.descriptor.view_name,
        "view_hash": first.descriptor.view_hash,
        "record_count": len(first.sample_ids),
        "files": files,
    }
    for cache in caches.values():
        if (
            cache.descriptor.source_manifest_hash != payload["source_manifest_hash"]
            or cache.descriptor.view_name != payload["view_name"]
            or cache.descriptor.view_hash != payload["view_hash"]
            or len(cache.sample_ids) != int(payload["record_count"])
        ):
            raise FeatureCacheFormatError("bundle 内 cache 的统一 view 或数量不一致")
    payload["manifest_sha256"] = _canonical_hash(payload)
    _atomic_json_write(manifest_path, payload, overwrite=overwrite)
    return {
        **payload,
        "path": relative_project_path(root, manifest_path),
        "file_size": int(manifest_path.stat().st_size),
        "sha256": _sha256(manifest_path),
    }


def subset_feature_cache_bundle(
    caches: Mapping[str, FeatureCache],
    records: Sequence[Mapping[str, Any]],
    *,
    view: DatasetView,
    source_manifest_hash: str,
) -> dict[str, FeatureCache]:
    """从已有 bundle 派生一个严格对齐的子视图 bundle。

    该操作只允许目标 view 是来源 bundle 的事件子集，适用于 L1/L2
    标签视图共享同一份 raw-DAT 计算结果的场景。派生后会重新绑定
    view_name/view_hash，避免训练侧把来源视图误当成目标视图。
    """

    if not caches:
        raise FeatureInputError("来源 feature cache bundle 不能为空")
    if not str(source_manifest_hash):
        raise FeatureInputError("feature cache source_manifest_hash 不能为空")
    target_records = _ordered_records(records, view)
    target_ids = tuple(str(record["sample_id"]) for record in target_records)
    result: dict[str, FeatureCache] = {}
    for feature_name, cache in caches.items():
        if cache.descriptor.source_manifest_hash != str(source_manifest_hash):
            raise FeatureInputError(
                f"来源 feature cache manifest hash 不一致：{feature_name}"
            )
        source_positions = {
            str(sample_id): index
            for index, sample_id in enumerate(cache.sample_ids)
        }
        missing = [sample_id for sample_id in target_ids if sample_id not in source_positions]
        if missing:
            raise FeatureInputError(
                f"来源 feature cache 不覆盖目标 view：{feature_name}:{missing[:3]}"
            )
        positions = np.asarray(
            [source_positions[sample_id] for sample_id in target_ids],
            dtype=np.int64,
        )
        metadata: dict[str, Any] = {}
        for key, value in dict(cache.metadata).items():
            if isinstance(value, list) and len(value) == len(cache.sample_ids):
                metadata[key] = [value[int(index)] for index in positions.tolist()]
            elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == len(cache.sample_ids):
                metadata[key] = value[positions].tolist()
            else:
                metadata[key] = value
        metadata.update(
            {
                "derived_from_view_name": cache.descriptor.view_name,
                "derived_from_view_hash": cache.descriptor.view_hash,
            }
        )
        descriptor = FeatureCacheDescriptor(
            schema_version=cache.descriptor.schema_version,
            feature_name=cache.descriptor.feature_name,
            source_manifest_hash=str(source_manifest_hash),
            feature_config_hash=cache.descriptor.feature_config_hash,
            view_name=view.view_name,
            view_hash=view.view_hash,
            normalization_hash=cache.descriptor.normalization_hash,
        )
        result[str(feature_name)] = build_feature_cache_from_records(
            target_records,
            np.asarray(cache.values[positions], dtype=np.float32),
            np.asarray(cache.valid[positions], dtype=np.bool_),
            descriptor=descriptor,
            feature_names=cache.feature_names,
            metadata=metadata,
        )
    return result


def load_feature_cache_bundle(
    project_root: str | Path,
    manifest_path: str | Path,
    *,
    expected_view: DatasetView | None = None,
    expected_source_manifest_hash: str | None = None,
    verify_integrity: bool = True,
) -> dict[str, FeatureCache]:
    """读取 bundle manifest，并按需执行完整 hash/view 校验。"""

    root = Path(project_root).resolve()
    target = resolve_project_path(root, manifest_path, must_exist=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FeatureCacheFormatError(f"无法读取 feature cache manifest：{target}") from exc
    if not isinstance(payload, Mapping):
        raise FeatureCacheFormatError("feature cache manifest 根节点必须是对象")
    if payload.get("schema_version") != FEATURE_CACHE_MANIFEST_SCHEMA:
        raise FeatureCacheFormatError("feature cache manifest schema_version 不匹配")
    if payload.get("cache_schema_version") != FEATURE_CACHE_SCHEMA_VERSION:
        raise FeatureCacheFormatError("feature cache schema_version 不匹配")
    declared_hash = payload.get("manifest_sha256")
    if verify_integrity and (
        not isinstance(declared_hash, str)
        or declared_hash != _canonical_hash(
            {key: value for key, value in payload.items() if key != "manifest_sha256"}
        )
    ):
        raise FeatureCacheFormatError("feature cache manifest 自身 hash 不一致")
    source_hash = str(payload.get("source_manifest_hash") or "")
    view_name = str(payload.get("view_name") or "")
    view_hash = str(payload.get("view_hash") or "")
    if verify_integrity and expected_source_manifest_hash is not None and source_hash != str(expected_source_manifest_hash):
        raise FeatureCacheFormatError("feature cache source manifest hash 不匹配")
    if expected_view is not None and (
        view_name != expected_view.view_name or view_hash != expected_view.view_hash
    ):
        raise FeatureCacheFormatError("feature cache bundle view 不匹配")
    files = payload.get("files")
    if not isinstance(files, Mapping) or not files:
        raise FeatureCacheFormatError("feature cache manifest 缺少 files")
    record_count = payload.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
        raise FeatureCacheFormatError("feature cache manifest record_count 无效")
    expected_sample_ids = None if expected_view is None else expected_view.sample_ids
    result: dict[str, FeatureCache] = {}
    for feature_name, raw_entry in files.items():
        if feature_name not in FEATURE_CACHE_FILE_NAMES or not isinstance(raw_entry, Mapping):
            raise FeatureCacheFormatError(f"feature cache manifest entry 无效：{feature_name}")
        entry = dict(raw_entry)
        path = resolve_project_path(root, str(entry.get("path") or ""), must_exist=True)
        ensure_path_inside_root(path, target.parent)
        if verify_integrity and int(entry.get("file_size", -1)) != int(path.stat().st_size):
            raise FeatureCacheFormatError(f"feature cache file_size 不一致：{path}")
        if verify_integrity and str(entry.get("sha256") or "") != feature_cache_sha256(path):
            raise FeatureCacheFormatError(f"feature cache hash 不一致：{path}")
        descriptor = entry.get("descriptor")
        if not isinstance(descriptor, Mapping):
            raise FeatureCacheFormatError(f"feature cache descriptor 缺失：{feature_name}")
        cache = load_feature_cache(
            path,
            expected_schema_version=(
                str(descriptor.get("schema_version") or FEATURE_CACHE_SCHEMA_VERSION)
            ),
            expected_feature_name=str(feature_name),
            expected_source_manifest_hash=source_hash if verify_integrity else None,
            expected_feature_config_hash=(
                str(descriptor.get("feature_config_hash") or "")
                if verify_integrity
                else None
            ),
            expected_view_name=view_name if verify_integrity else None,
            expected_view_hash=view_hash if verify_integrity else None,
            expected_normalization_hash=(
                str(descriptor.get("normalization_hash"))
                if verify_integrity and descriptor.get("normalization_hash") is not None
                else None
            ),
            expected_sample_ids=expected_sample_ids if verify_integrity else None,
        )
        if verify_integrity and cache.descriptor.to_dict() != dict(descriptor):
            raise FeatureCacheFormatError(f"feature cache descriptor 登记不一致：{feature_name}")
        if verify_integrity and (
            len(cache.sample_ids) != int(record_count)
            or int(entry.get("record_count", -1)) != int(record_count)
        ):
            raise FeatureCacheFormatError(f"feature cache record_count 登记不一致：{feature_name}")
        if verify_integrity and list(cache.values.shape) != list(entry.get("value_shape") or []):
            raise FeatureCacheFormatError(f"feature cache value shape 登记不一致：{feature_name}")
        if verify_integrity and list(cache.valid.shape) != list(entry.get("valid_shape") or []):
            raise FeatureCacheFormatError(f"feature cache valid shape 登记不一致：{feature_name}")
        if verify_integrity and list(cache.feature_names) != list(entry.get("feature_names") or []):
            raise FeatureCacheFormatError(f"feature cache feature_names 登记不一致：{feature_name}")
        result[str(feature_name)] = cache
    return result


__all__ = [
    "DEFAULT_FEATURE_CACHE_MANIFEST_NAME",
    "FEATURE_CACHE_FILE_NAMES",
    "FEATURE_CACHE_MANIFEST_SCHEMA",
    "ProgressCallback",
    "build_feature_cache_bundle",
    "load_feature_cache_bundle",
    "save_feature_cache_bundle",
    "subset_feature_cache_bundle",
]
