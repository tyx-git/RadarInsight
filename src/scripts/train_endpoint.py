from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


def _atomic_json_write(path: Path, payload: Any) -> None:
    """原子写入训练摘要，避免长训练结束时留下半个 JSON。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _print_summary(task: str, summary: dict[str, Any]) -> None:
    selected = next(
        item
        for item in summary["history"]
        if int(item["epoch"]) == int(summary["selected_epoch"])
    )
    validation = selected["validation"]
    _get_logger().info(
        json.dumps(
            {
                "task": task,
                "selected_epoch": summary["selected_epoch"],
                "accuracy": validation["accuracy"],
                "macro_f1": validation["macro_f1"],
                "per_class": validation["per_class"],
                "checkpoint": summary["checkpoint"],
                "manifest_path": summary.get("manifest_path"),
                "summary_path": summary.get("summary_path"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


_LOGGER = None


def _get_logger():
    """获取训练编排 logger，所有控制台状态同时写入日志文件。"""

    global _LOGGER
    if _LOGGER is None:
        from radar_ln3.logger import get_logger

        _LOGGER = get_logger("training.script")
    return _LOGGER


def _cache_paths(root: Path) -> tuple[Path, ...]:
    cache_root = root / "data" / "cq08" / "cache" / "features"
    name = "feature-cache-manifest.json"
    return (
        root / "data" / "cq08" / "rd" / "canonical" / "canonical-rd-manifest.json",
        cache_root / name,
        cache_root / "l1" / "train" / name,
        cache_root / "l1" / "validation" / name,
        cache_root / "l2" / "train" / name,
        cache_root / "l2" / "validation" / name,
    )


def _root_relative(project_root: Path, value: str | Path) -> str:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    try:
        return resolved.relative_to(project_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"--root 必须位于项目目录内：{value}") from exc


def _assert_output_absent(root: Path, config: Any, task: str) -> None:
    """正式训练不覆盖已有 checkpoint 或 epoch 产物。"""

    output_path = root / (
        config.artifacts.l1_output_path
        if task == "l1"
        else config.artifacts.l2_output_path
    )
    epoch_root = output_path.parent / "epochs"
    existing = [path for path in (output_path, *epoch_root.glob("*.pt")) if path.is_file()]
    if existing:
        names = ", ".join(str(path) for path in existing[:3])
        raise RuntimeError(
            f"{task.upper()} 正式训练产物已存在，拒绝静默覆盖：{names}；"
            "请人工备份或清理后重新执行。"
        )


def _archive_existing_outputs(
    root: Path,
    config: Any,
    *,
    task: str,
    mode: str = "residual",
) -> Path | None:
    """在显式替换训练产物前归档已存在的输出。"""

    if task == "l1":
        output_path = root / config.artifacts.l1_output_path
    elif task == "l2":
        output_path = root / (
            config.artifacts.l2_direct_grouped_output_path
            if mode == "direct_grouped"
            else config.artifacts.l2_direct_output_path
            if mode == "direct"
            else config.artifacts.l2_output_path
        )
    else:
        raise ValueError(f"不支持归档的训练 task：{task}")

    candidates = (
        output_path,
        output_path.parent / "epochs",
        output_path.parent / "training-summary.json",
    )
    existing = [path for path in candidates if path.is_file() or path.is_dir()]
    if not existing:
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_root = root / "artifacts" / "radar_ln3" / "replaced"
    archive_dir = archive_root / timestamp / task
    suffix = 1
    while archive_dir.exists():
        archive_dir = archive_root / f"{timestamp}-{suffix:02d}" / task
        suffix += 1
    archive_dir.mkdir(parents=True, exist_ok=False)
    moved: list[tuple[Path, Path]] = []
    try:
        for source in existing:
            destination = archive_dir / source.name
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
    except Exception:
        for source, destination in reversed(moved):
            shutil.move(str(destination), str(source))
        archive_dir.rmdir()
        raise
    return archive_dir


def _all_existing_checkpoints_are_stale(
    root: Path,
    output_path: Path,
    *,
    task: str,
    current_config_hash: str,
) -> bool:
    """只有所有 checkpoint 都带有效且不同的配置 hash 时才自动归档。"""

    from radar_ln3.checkpoint import Checkpoint

    checkpoint_paths = [
        path
        for path in (
            output_path,
            *sorted((output_path.parent / "epochs").glob("*.pt")),
        )
        if path.is_file()
    ]
    if not checkpoint_paths:
        return False

    for path in checkpoint_paths:
        try:
            checkpoint = Checkpoint.load(root, path, expected_task=task)
        except (OSError, RuntimeError, ValueError):
            return False
        saved_hash = checkpoint.payload.get("config_sha256")
        if (
            not isinstance(saved_hash, str)
            or len(saved_hash) != 64
            or any(character not in "0123456789abcdef" for character in saved_hash.lower())
            or saved_hash.lower() == current_config_hash.lower()
        ):
            return False
    return True


def _prepare_output_for_training(
    root: Path,
    config: Any,
    *,
    task: str,
    replace_existing: bool,
    mode: str = "residual",
) -> None:
    """归档显式替换项或可确认属于旧配置的产物，再执行覆盖保护。"""

    output_path = root / (
        config.artifacts.l1_output_path
        if task == "l1"
        else config.artifacts.l2_direct_grouped_output_path
        if mode == "direct_grouped"
        else config.artifacts.l2_direct_output_path
        if mode == "direct"
        else config.artifacts.l2_output_path
    )
    archive_reason = None
    if replace_existing:
        archive_reason = "显式请求替换"
    else:
        from radar_ln3.checkpoint import sha256_file

        if _all_existing_checkpoints_are_stale(
            root,
            output_path,
            task=task,
            current_config_hash=sha256_file(config.path),
        ):
            archive_reason = "checkpoint 配置 hash 与当前配置不一致"

    if archive_reason is not None:
        archive_dir = _archive_existing_outputs(
            root,
            config,
            task=task,
            mode=mode,
        )
        if archive_dir is not None:
            _get_logger().info(
                "[script] 已归档旧 %s 产物（%s）：%s",
                task.upper(),
                archive_reason,
                archive_dir,
            )


def _load_context(
    root: Path,
    manifest_path: Path,
    progress_interval: int,
    *,
    verify_files: bool,
    show_progress: bool,
):
    from radar_ln3.data.manifest import load_manifest

    return load_manifest(
        manifest_path,
        root,
        verify_files=verify_files,
        allow_unmaterialized=False,
        progress_callback=_print_progress if show_progress else None,
        progress_interval=progress_interval,
    )


def _print_progress(event: dict[str, Any]) -> None:
    # 复用产品命令入口的前端显式进度展示；统一日志系统接入后再替换这里。
    from core.cli import _print_progress as print_progress

    print_progress(event)


def _build_or_load_caches(
    root: Path,
    manifest_path: Path,
    *,
    progress_interval: int,
    rebuild_cache: bool,
    validate_caches: bool,
    skip_data_validation: bool = False,
):
    from core.training.cache import (
        build_feature_training_caches,
        load_task_feature_caches,
    )
    from radar_ln3.data.cq08 import CQ08Adapter, CQ08RawEchoProvider
    from radar_ln3.model import load_model_config

    validate_caches = bool(validate_caches and not skip_data_validation)
    cache_paths = _cache_paths(root)
    cache_ready = all(path.is_file() for path in cache_paths)
    manifest = _load_context(
        root,
        manifest_path,
        progress_interval,
        verify_files=validate_caches,
        show_progress=validate_caches,
    )

    if cache_ready and not rebuild_cache:
        if validate_caches:
            _get_logger().info("[script] 已发现完整 feature cache，开始严格校验并复用。")
        else:
            _get_logger().info("[script] 已发现完整 feature cache，训练阶段跳过训练前完整校验并直接复用。")
    else:
        if not rebuild_cache:
            missing = [str(path) for path in _cache_paths(root) if not path.is_file()]
            canonical_manifest = root / "data" / "cq08" / "rd" / "canonical" / "canonical-rd-manifest.json"
            if not canonical_manifest.is_file():
                raise RuntimeError(
                    "canonical RD store 不存在；请先运行："
                    "python src/scripts/materialize_canonical_rd.py "
                    "--project-root $root --manifest-path src/config/manifests.json "
                    "--output-root data/cq08/rd/canonical --overwrite"
                )
            raise RuntimeError(
                "已有 feature cache 不完整；当前训练脚本禁止自动生成或补建 cache。"
                f"缺失：{missing[:3]}。如需重建，请显式使用 -RebuildCache。"
            )
        adapter = CQ08Adapter(
            root,
            registered_index=True,
            manifest_path=manifest_path,
            verify_files=validate_caches,
        )
        provider = CQ08RawEchoProvider(
            root,
            source_root=adapter.source_root,
            cache_size=2,
        )
        config = load_model_config(root)
        result = build_feature_training_caches(
            root,
            manifest_path=manifest_path,
            output_root=root / "data" / "cq08" / "cache" / "features",
            raw_echo_provider=provider,
            stable_median=config.data.l2.stable8_median,
            stable_scale=config.data.l2.stable8_scale,
            overwrite=bool(rebuild_cache),
            manifest=manifest,
            verify_files=validate_caches,
            validate_event_cache=validate_caches,
            progress_callback=_print_progress,
            progress_interval=progress_interval,
        )
        _get_logger().info(
            json.dumps(
                {
                    "stage": "feature_cache",
                    "manifest_path": result["manifest_path"],
                    "bundle_count": result["bundle_count"],
                    "bundle_paths": {
                        key: value.get("path")
                        for key, value in result["bundles"].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        manifest = _load_context(
            root,
            manifest_path,
            progress_interval,
            verify_files=validate_caches,
            show_progress=validate_caches,
        )

    l1_train, l1_validation = load_task_feature_caches(
        root,
        task="l1",
        manifest_path=manifest_path,
        manifest=manifest,
        validate_cache=validate_caches,
    )
    l2_train, l2_validation = load_task_feature_caches(
        root,
        task="l2",
        manifest_path=manifest_path,
        manifest=manifest,
        validate_cache=validate_caches,
    )
    if validate_caches:
        _get_logger().info("[script] L1/L2 train/validation feature cache 校验通过。")
    else:
        _get_logger().info("[script] 训练阶段已加载 L1/L2 train/validation feature cache，跳过训练前完整校验。")
    return manifest, (l1_train, l1_validation), (l2_train, l2_validation)


def _train_l1(
    root: Path,
    manifest_path: Path,
    manifest: Any,
    caches: tuple[dict[str, Any], dict[str, Any]],
    *,
    device: str,
    replace_existing: bool,
    progress_interval: int,
    batch_size: int,
    config_path: Path,
    artifact_root: str,
):
    from radar_ln3.model import load_model_config
    from radar_ln3.training import train_l1

    config = load_model_config(root, config_path, artifact_root=artifact_root)
    _prepare_output_for_training(
        root,
        config,
        task="l1",
        replace_existing=replace_existing,
    )
    _assert_output_absent(root, config, "l1")
    summary = train_l1(
        root,
        manifest_path=manifest_path,
        manifest=manifest,
        device=device,
        arm="observation_increment",
        progress_callback=_print_progress,
        train_feature_caches=caches[0],
        validation_feature_caches=caches[1],
        validate_caches=False,
        progress_interval=progress_interval,
        batch_size=batch_size,
        config_path=config_path,
        artifact_root=artifact_root,
    )
    summary_path = root / artifact_root / "l1" / "training-summary.json"
    summary["summary_path"] = str(summary_path)
    _atomic_json_write(summary_path, summary)
    _print_summary("l1", summary)
    return summary


def _train_l2(
    root: Path,
    manifest_path: Path,
    manifest: Any,
    caches: tuple[dict[str, Any], dict[str, Any]],
    *,
    device: str,
    mode: str,
    replace_existing: bool,
    progress_interval: int,
    resume: bool,
    batch_size: int,
    config_path: Path,
    artifact_root: str,
):
    from radar_ln3.model import load_model_config
    from radar_ln3.training.l2_anchor import ensure_l2_baseline_anchor
    from radar_ln3.training import train_l2

    config = load_model_config(root, config_path, artifact_root=artifact_root)
    if mode not in {"residual", "direct", "direct_grouped"}:
        raise ValueError("L2 mode 无效")
    if mode == "residual":
        anchor_result = ensure_l2_baseline_anchor(
            root,
            manifest_path=manifest_path,
            manifest=manifest,
            feature_cache_manifest_path="data/cq08/cache/features/feature-cache-manifest.json",
            output_path=config.artifacts.l2_baseline_anchor_path,
            device=device,
            batch_size=batch_size,
            verify_files=False,
            validate_event_cache=False,
        )
        _get_logger().info(
            "[script] L2 baseline anchor 已准备：status=%s parent_count=%s record_count=%s computed_event_count=%s",
            anchor_result["status"],
            anchor_result["parent_count"],
            anchor_result["record_count"],
            anchor_result["computed_event_count"],
        )
    output_path = (
        root / config.artifacts.l2_direct_grouped_output_path
        if mode == "direct_grouped"
        else root / config.artifacts.l2_direct_output_path
        if mode == "direct"
        else root / config.artifacts.l2_output_path
    )
    epoch_paths = list((output_path.parent / "epochs").glob("epoch-*.pt"))
    if resume:
        if replace_existing:
            raise ValueError("-Resume 与 -ReplaceExisting 不能同时使用")
        if output_path.exists():
            raise RuntimeError(
                f"L2 final checkpoint 已存在，不能从 epoch checkpoints resume：{output_path}"
            )
        if not epoch_paths:
            raise RuntimeError("L2 resume 没有可恢复的 epoch checkpoint。")
    else:
        _prepare_output_for_training(
            root,
            config,
            task="l2",
            replace_existing=replace_existing,
            mode=mode,
        )
        if output_path.exists() or list((output_path.parent / "epochs").glob("epoch-*.pt")):
            raise RuntimeError(
                f"L2 {mode} 正式训练产物已存在，拒绝静默覆盖：{output_path}"
            )
    summary = train_l2(
        root,
        manifest_path=manifest_path,
        manifest=manifest,
        device=device,
        arm="robust_control",
        mode=mode,
        progress_callback=_print_progress,
        raw_echo_provider=None,
        train_feature_caches=caches[0],
        validation_feature_caches=caches[1],
        validate_caches=False,
        progress_interval=progress_interval,
        resume=resume,
        batch_size=batch_size,
        config_path=config_path,
        artifact_root=artifact_root,
    )
    summary_path = (
        root / artifact_root / "l2-direct-grouped" / "training-summary.json"
        if mode == "direct_grouped"
        else root / artifact_root / "l2-direct" / "training-summary.json"
        if mode == "direct"
        else root / artifact_root / "l2" / "training-summary.json"
    )
    summary["summary_path"] = str(summary_path)
    _atomic_json_write(summary_path, summary)
    _print_summary("l2", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="RadarInsight 正式 L1/L2 训练编排")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/radar_ln3"))
    parser.add_argument("--manifest-path", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "cache", "l1", "l2"), required=True)
    parser.add_argument("--l2-mode", choices=("residual", "direct", "direct_grouped"), required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), required=True)
    parser.add_argument("--progress-interval", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model-config-path", type=Path, required=True)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-data-validation", action="store_true")
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    artifact_root = _root_relative(root, args.root)
    from radar_ln3.artifacts import DEFAULT_ARTIFACT_ROOT

    if args.stage in {"all", "l1", "l2"} and not args.overwrite and Path(artifact_root) == DEFAULT_ARTIFACT_ROOT:
        raise FileExistsError(
            f"默认 Root {artifact_root} 已可能包含模型包；重新创建整包请显式传入 --overwrite"
        )
    if args.overwrite and args.stage in {"all", "l1", "l2"}:
        print(
            f"[train] 显式覆盖请求：{args.stage} 的 Root/阶段目录将在输出前打包归档。",
            flush=True,
        )
    manifest_path = args.manifest_path.resolve()
    from radar_ln3.logger import close_logging, configure_logging

    configure_logging(root, log_path=Path(artifact_root) / "logs" / "training.log")
    try:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"总 manifest 不存在：{manifest_path}")
        if args.progress_interval <= 0:
            raise ValueError("progress_interval 必须是正整数")
        if args.batch_size <= 0:
            raise ValueError("batch_size 必须是正整数")
        if args.resume and args.stage != "l2":
            raise ValueError("L2 resume 只能与 --stage l2 一起使用")

        if args.stage == "cache":
            _build_or_load_caches(
                root,
                manifest_path,
                progress_interval=args.progress_interval,
                rebuild_cache=args.rebuild_cache,
                validate_caches=True,
                skip_data_validation=args.skip_data_validation,
            )
            _get_logger().info("[script] feature cache 阶段完成。")
            return 0

        from radar_ln3.artifacts import prepare_artifact_directory, prepare_artifact_stage

        if args.stage == "all":
            prepare_artifact_directory(root / artifact_root, overwrite=args.overwrite)
        elif args.stage in {"l1", "l2"}:
            prepare_artifact_stage(root / artifact_root, args.stage, overwrite=args.overwrite)

        manifest, l1_caches, l2_caches = _build_or_load_caches(
            root,
            manifest_path,
            progress_interval=args.progress_interval,
            rebuild_cache=args.rebuild_cache,
            validate_caches=False,
            skip_data_validation=args.skip_data_validation,
        )
        if args.stage in {"all", "l1"}:
            _train_l1(
                root,
                manifest_path,
                manifest,
                l1_caches,
                device=args.device,
                replace_existing=args.replace_existing,
                progress_interval=args.progress_interval,
                batch_size=args.batch_size,
                config_path=args.model_config_path,
                artifact_root=artifact_root,
            )
        if args.stage in {"all", "l2"}:
            # L1 完成后总 manifest 会登记 L1 checkpoint，L2 使用最新总 manifest。
            if args.stage == "all":
                manifest = _load_context(
                    root,
                    manifest_path,
                    args.progress_interval,
                    verify_files=False,
                    show_progress=False,
                )
            _train_l2(
                root,
                manifest_path,
                manifest,
                l2_caches,
                device=args.device,
                mode=args.l2_mode,
                replace_existing=args.replace_existing,
                progress_interval=args.progress_interval,
                resume=args.resume,
                batch_size=args.batch_size,
                config_path=args.model_config_path,
                artifact_root=artifact_root,
            )
        _get_logger().info("[script] 训练流程完成。")
        return 0
    except Exception as exc:
        _get_logger().exception(
            "[script] 训练失败：%s: %s", type(exc).__name__, exc
        )
        raise
    finally:
        close_logging()


if __name__ == "__main__":
    raise SystemExit(main())
