"""训练侧缓存编排。

本模块只负责调用通用 cache 模块，不定义 CQ08 字段解析或模型特征。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from radar_ln3.cache import (
    DEFAULT_CACHE_RELATIVE_PATH,
    CachedParentSequenceDataset,
    build_event_cache,
)
from radar_ln3.data.derived import register_product_artifacts
from radar_ln3.data.contracts import DatasetManifest
from radar_ln3.data.index import write_manifest
from radar_ln3.data.manifest import (
    DEFAULT_PROGRESS_INTERVAL,
    data_manifest_hash,
    load_manifest,
    resolve_manifest_view,
)
from radar_ln3.features import (
    FeatureCache,
    build_feature_cache_bundle,
    load_feature_cache_bundle,
    save_feature_cache_bundle,
    stable8_normalization_hash,
    subset_feature_cache_bundle,
)
from radar_ln3.model import load_model_config
from radar_ln3.training.common import (
    KNOWN_TRAINING_LABELS,
    TASK_LABELS,
    build_training_dataset,
)


DEFAULT_FEATURE_CACHE_ROOT = Path("data/cq08/cache/features")
DEFAULT_FEATURE_CACHE_MANIFEST = (
    DEFAULT_FEATURE_CACHE_ROOT / "feature-cache-manifest.json"
)
FEATURE_CACHE_MANIFEST_NAME = "feature-cache-manifest.json"


def _validate_l2_normalization(
    project_root: Path,
    train_cache: Mapping[str, FeatureCache],
    validation_cache: Mapping[str, FeatureCache],
) -> None:
    """拒绝使用与当前模型配置不一致的 stable8 cache。"""

    config = load_model_config(project_root)
    expected = stable8_normalization_hash(
        config.data.l2.stable8_median,
        config.data.l2.stable8_scale,
    )
    for split_name, caches in (("train", train_cache), ("validation", validation_cache)):
        for feature_name in ("micro_doppler_stable8", "micro_doppler_stable8_frame"):
            cache = caches.get(feature_name)
            if cache is None:
                raise ValueError(
                    f"L2 {split_name} feature cache 缺少 stable8：{feature_name}"
                )
            actual = cache.descriptor.normalization_hash
            if actual != expected:
                raise ValueError(
                    f"L2 {split_name} stable8 normalization hash 与当前模型配置不一致："
                    f"feature={feature_name}, cache={actual or '<missing>'}, "
                    f"current={expected}。请先使用 -Stage cache -RebuildCache 重建 cache。"
                )


ProgressCallback = Callable[[Mapping[str, Any]], None]


def build_training_cache(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    cache_path: str | Path = DEFAULT_CACHE_RELATIVE_PATH,
    overwrite: bool = False,
    verify_files: bool = True,
    manifest: DatasetManifest | None = None,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> dict[str, Any]:
    """构建事件引用缓存，并将其注册回总 manifest。"""

    root = Path(project_root).resolve()
    manifest = manifest or load_manifest(
        manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    descriptor = build_event_cache(
        root,
        manifest,
        cache_path=cache_path,
        overwrite=overwrite,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    manifest.audit["cache"] = descriptor
    target = write_manifest(
        manifest,
        root,
        path=manifest_path,
        overwrite=overwrite,
        verify_files=bool(verify_files),
        validate=False,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    return {"manifest_path": str(target), "cache": descriptor}


def load_training_dataset(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    split_name: str | None = None,
    parent_ids: Iterable[str] | None = None,
    include_unknown: bool = False,
    training_view: bool = True,
    allowed_labels: Iterable[str] | None = None,
    cache_size: int = 8,
    allow_unknown_head_training: bool = False,
) -> CachedParentSequenceDataset:
    """构建使用事件缓存和 parent LRU 的训练数据集。"""

    return CachedParentSequenceDataset(
        project_root,
        manifest_path=manifest_path,
        split_name=split_name,
        parent_ids=parent_ids,
        include_unknown=include_unknown,
        training_view=training_view,
        allowed_labels=allowed_labels,
        allow_unknown_head_training=allow_unknown_head_training,
        cache_size=cache_size,
    )


def build_feature_training_cache(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    output_root: str | Path = DEFAULT_FEATURE_CACHE_ROOT,
    manifest_name: str = "feature-cache-manifest.json",
    split_name: str | None = None,
    parent_ids: Iterable[str] | None = None,
    include_unknown: bool = False,
    training_view: bool = True,
    allowed_labels: Iterable[str] | None = None,
    task_name: str = "",
    allow_unknown_head_training: bool = False,
    raw_echo_provider: Any | None = None,
    stable_median: Any | None = None,
    stable_scale: Any | None = None,
    feature_config: Mapping[str, Any] | None = None,
    require_stable_evidence: bool = True,
    overwrite: bool = False,
    feature_cache_key: str | None = None,
    register: bool = True,
    manifest: DatasetManifest | None = None,
    verify_files: bool = True,
    validate_event_cache: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> dict[str, Any]:
    """按统一 view 构建 feature cache，并登记到总 manifest。"""

    root = Path(project_root).resolve()
    manifest = manifest or load_manifest(
        manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    dataset = CachedParentSequenceDataset(
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
        validate_event_cache=bool(validate_event_cache),
    )
    sequences = (dataset[index] for index in range(len(dataset)))
    caches = build_feature_cache_bundle(
        sequences,
        manifest.tr_records,
        view=dataset.view,
        source_manifest_hash=data_manifest_hash(manifest),
        feature_config=feature_config,
        raw_echo_provider=raw_echo_provider,
        stable_median=stable_median,
        stable_scale=stable_scale,
        require_stable_evidence=require_stable_evidence,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    bundle = save_feature_cache_bundle(
        root,
        caches,
        output_root=output_root,
        manifest_name=manifest_name,
        overwrite=overwrite,
    )
    registered_manifest = None
    if register:
        registered_manifest = register_product_artifacts(
            root,
            manifest_path=manifest_path,
            # feature cache 文件本身仍由 save_feature_cache 的 overwrite
            # 参数控制；这里需要原子更新总 manifest 的派生登记。
            overwrite=True,
            manifest=manifest,
            feature_cache_descriptor=bundle,
            feature_cache_key=feature_cache_key,
        )
    return {
        "manifest_path": None if registered_manifest is None else str(registered_manifest),
        "feature_cache_manifest": bundle,
        "view_name": dataset.view.view_name,
        "view_hash": dataset.view.view_hash,
        "record_count": len(dataset.view.sample_ids),
    }


def build_feature_training_caches(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    output_root: str | Path = DEFAULT_FEATURE_CACHE_ROOT,
    raw_echo_provider: Any,
    stable_median: Any,
    stable_scale: Any,
    overwrite: bool = False,
    manifest: DatasetManifest | None = None,
    verify_files: bool = True,
    validate_event_cache: bool = True,
    progress_callback: ProgressCallback | None = None,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> dict[str, Any]:
    """构建共享 raw-DAT cache，并派生正式 L1/L2 train/validation cache。

    每个 split 只执行一次 raw-DAT/micro-Doppler 提取；L1、L2 只从共享
    bundle 按各自统一 DatasetView 派生，避免重复解析原始回波。
    """

    if raw_echo_provider is None:
        raise ValueError("构建完整 feature cache 必须提供 raw_echo_provider")
    if stable_median is None or stable_scale is None:
        raise ValueError("构建完整 feature cache 必须提供 stable8 normalization")
    if isinstance(progress_interval, bool) or int(progress_interval) <= 0:
        raise ValueError("progress_interval 必须是正整数")

    root = Path(project_root).resolve()
    manifest = manifest or load_manifest(
        manifest_path,
        root,
        verify_files=bool(verify_files),
        allow_unmaterialized=False,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    source_manifest_hash = data_manifest_hash(manifest)
    result: dict[str, Any] = {
        "source_manifest_hash": source_manifest_hash,
        "bundles": {},
    }
    descriptors: dict[str, Mapping[str, Any]] = {}
    target_root = Path(output_root)
    # 统一视图的 bundle 只用于审计和派生，不直接作为训练输入。

    source_dataset = CachedParentSequenceDataset(
        root,
        manifest_path=manifest_path,
        split_name=None,
        include_unknown=False,
        training_view=True,
        allowed_labels=KNOWN_TRAINING_LABELS,
        task_name="feature_source",
        manifest=manifest,
        validate_event_cache=bool(validate_event_cache),
    )
    source_caches = build_feature_cache_bundle(
        (source_dataset[index] for index in range(len(source_dataset))),
        manifest.tr_records,
        view=source_dataset.view,
        source_manifest_hash=source_manifest_hash,
        raw_echo_provider=raw_echo_provider,
        stable_median=stable_median,
        stable_scale=stable_scale,
        require_stable_evidence=True,
        progress_callback=progress_callback,
        progress_interval=progress_interval,
    )
    shared_bundle = save_feature_cache_bundle(
        root,
        source_caches,
        output_root=target_root,
        manifest_name=FEATURE_CACHE_MANIFEST_NAME,
        overwrite=overwrite,
    )
    descriptors["shared_non_test"] = shared_bundle
    result["bundles"]["shared_non_test"] = shared_bundle

    for split_name in ("train", "validation"):
        for task_name in ("l1", "l2"):
            # 复用正式训练公共数据集的标签集合和 view 语义。
            task_view = resolve_manifest_view(
                manifest,
                split_name=split_name,
                include_unknown=False,
                training_view=True,
                allowed_labels=TASK_LABELS[task_name],
                task_name=task_name,
            )
            if source_dataset.view.data_binding != task_view.data_binding:
                from dataclasses import replace

                task_view = replace(
                    task_view,
                    data_binding=source_dataset.view.data_binding,
                )
            task_caches = subset_feature_cache_bundle(
                source_caches,
                manifest.tr_records,
                view=task_view,
                source_manifest_hash=source_manifest_hash,
            )
            task_bundle = save_feature_cache_bundle(
                root,
                task_caches,
                output_root=target_root / task_name / split_name,
                manifest_name=FEATURE_CACHE_MANIFEST_NAME,
                overwrite=overwrite,
            )
            key = f"{task_name}_{split_name}"
            descriptors[key] = task_bundle
            result["bundles"][key] = task_bundle

    if progress_callback is not None:
        progress_callback(
            {
                "stage": "feature_cache_split_complete",
                "split_name": "non_test",
                "bundle_count": 5,
                "source_record_count": len(source_dataset.view.sample_ids),
            }
        )

    # 总 manifest 是唯一总集合；子 bundle 只作为详细文件登记。
    registered_manifest = register_product_artifacts(
        root,
        manifest_path=manifest_path,
        # 缓存文件已在上面完成严格写入，登记阶段只更新总 manifest。
        overwrite=True,
        manifest=manifest,
        feature_cache_descriptors=descriptors,
    )
    result["manifest_path"] = str(registered_manifest)
    result["bundle_count"] = len(descriptors)
    return result


def load_task_feature_caches(
    project_root: str | Path,
    *,
    task: str,
    manifest_path: str | Path = "src/config/manifests.json",
    feature_cache_root: str | Path = DEFAULT_FEATURE_CACHE_ROOT,
    manifest: DatasetManifest | None = None,
    datasets: Mapping[str, CachedParentSequenceDataset] | None = None,
    validate_cache: bool = True,
    validate_event_cache: bool | None = None,
) -> tuple[dict[str, FeatureCache], dict[str, FeatureCache]]:
    """按正式 task 的 train/validation DatasetView 加载 feature cache。

    ``datasets`` 提供时直接复用调用方已构造的 train/validation dataset，
    不再重复解析事件缓存与 canonical RD。
    """

    if str(task) not in {"l1", "l2"}:
        raise ValueError(f"不支持的 feature cache task：{task}")
    root = Path(project_root).resolve()
    event_cache_validation = (
        bool(validate_cache)
        if validate_event_cache is None
        else bool(validate_event_cache)
    )
    manifest = manifest or load_manifest(
        manifest_path, root, verify_files=True, allow_unmaterialized=False
    )
    if datasets is None:
        train_dataset = build_training_dataset(
            root,
            task=str(task),
            split_name="train",
            manifest_path=manifest_path,
            manifest=manifest,
            validate_event_cache=event_cache_validation,
        )
        validation_dataset = build_training_dataset(
            root,
            task=str(task),
            split_name="validation",
            manifest_path=manifest_path,
            manifest=manifest,
            validate_event_cache=event_cache_validation,
        )
    else:
        try:
            train_dataset = datasets["train"]
            validation_dataset = datasets["validation"]
        except KeyError as exc:
            raise ValueError(f"datasets 必须包含 train/validation：{exc}") from exc
    source_hash = data_manifest_hash(manifest) if validate_cache else None
    root_path = Path(feature_cache_root)
    train_manifest = root_path / str(task) / "train" / FEATURE_CACHE_MANIFEST_NAME
    validation_manifest = root_path / str(task) / "validation" / FEATURE_CACHE_MANIFEST_NAME
    train_cache = load_feature_cache_bundle(
        root,
        train_manifest,
        expected_view=train_dataset.view,
        expected_source_manifest_hash=source_hash,
        verify_integrity=validate_cache,
    )
    validation_cache = load_feature_cache_bundle(
        root,
        validation_manifest,
        expected_view=validation_dataset.view,
        expected_source_manifest_hash=source_hash,
        verify_integrity=validate_cache,
    )
    if validate_cache:
        _validate_l2_normalization(root, train_cache, validation_cache)
    return train_cache, validation_cache


def load_feature_training_cache(
    project_root: str | Path,
    *,
    manifest_path: str | Path = "src/config/manifests.json",
    feature_manifest_path: str | Path = DEFAULT_FEATURE_CACHE_MANIFEST,
    split_name: str | None = None,
    parent_ids: Iterable[str] | None = None,
    include_unknown: bool = False,
    training_view: bool = True,
    allowed_labels: Iterable[str] | None = None,
    task_name: str = "",
    allow_unknown_head_training: bool = False,
) -> dict[str, FeatureCache]:
    """按同一 view 加载 feature cache，view/hash 不一致时失败。"""

    root = Path(project_root).resolve()
    manifest = load_manifest(
        manifest_path,
        root,
        verify_files=True,
        allow_unmaterialized=False,
    )
    dataset = CachedParentSequenceDataset(
        root,
        manifest_path=manifest_path,
        split_name=split_name,
        parent_ids=parent_ids,
        include_unknown=include_unknown,
        training_view=training_view,
        allowed_labels=allowed_labels,
        task_name=task_name,
        allow_unknown_head_training=allow_unknown_head_training,
    )
    return load_feature_cache_bundle(
        root,
        feature_manifest_path,
        expected_view=dataset.view,
        expected_source_manifest_hash=data_manifest_hash(manifest),
    )


__all__ = [
    "DEFAULT_FEATURE_CACHE_MANIFEST",
    "DEFAULT_FEATURE_CACHE_ROOT",
    "FEATURE_CACHE_MANIFEST_NAME",
    "ProgressCallback",
    "build_feature_training_cache",
    "build_feature_training_caches",
    "build_training_cache",
    "load_task_feature_caches",
    "load_feature_training_cache",
    "load_training_dataset",
]
