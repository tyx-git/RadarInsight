"""命令入口。

本模块只负责解析参数、调用数据层并把处理进度展示给前端。数据发现、
配对、规范化和 manifest 校验仍由 ``radar_ln3.data`` 完成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from radar_ln3.data.cq08 import CQ08Adapter
from radar_ln3.data.index import build_manifest, write_manifest
from radar_ln3.data.contracts import DataError, DataIntegrityError
from radar_ln3.data.manifest import (
    DEFAULT_PROGRESS_INTERVAL,
    canonical_json_hash,
)
from radar_ln3.logger import LoggerError, close_logging, configure_logging, get_logger
from core.training.cache import build_training_cache


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("必须是正整数") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """创建产品命令行参数解析器。"""

    parser = argparse.ArgumentParser(description="RadarInsight 产品命令入口")
    parser.add_argument(
        "--handle-raw-data",
        action="store_true",
        help="发现 CQ08 原始数据、生成数据产物和分层索引，并写入统一 manifest",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="项目根目录，默认根据 cli.py 位置解析",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="CQ08 原始根目录，默认使用 <project-root>/OriData/23/CQ-08",
    )
    parser.add_argument(
        "--split-path",
        type=Path,
        default=None,
        help="parent 级 split 文件；未提供时使用产品内已登记的 CQ08 split",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("src/config/manifests.json"),
        help="统一 manifest 路径，必须位于项目根目录内",
    )
    parser.add_argument(
        "--parent-limit",
        type=_positive_integer,
        default=None,
        help="只处理按稳定顺序排列的前 N 个 parent，用于有界 smoke",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="明确允许覆盖已有的不同数据产物、sidecar 或 manifest",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/cq08/cache/event-source-cache.jsonl"),
        help="事件引用缓存路径，必须位于 data/cq08/cache 内",
    )
    return parser


def _print_progress(event: Mapping[str, Any]) -> None:
    """把数据层回调转换为前端当前阶段的显式展示。"""

    logger = get_logger("cli")
    stage = str(event.get("stage", ""))
    if stage == "start":
        logger.info(
            "[data] 开始处理 "
            f"dataset={event.get('dataset_name')} "
            f"parents={event.get('parent_count')} "
            f"excluded={event.get('excluded_parent_count', 0)} "
            f"materialize={event.get('materialize')}"
        )
    elif stage == "parent_progress":
        logger.info(
            "[data] 处理进度 "
            f"{event.get('processed')}/{event.get('total')} "
            f"{event.get('parent_sample_id')} "
            f"events={event.get('event_count')} "
            f"split={event.get('split_name')} "
            f"anomalous={event.get('parent_anomalous', False)}"
        )
    elif stage == "sidecar_start":
        logger.info(
            "[data] sidecar 写入开始 "
            f"section={event.get('section')} "
            f"records={event.get('record_count')} "
            f"path={event.get('path')}"
        )
    elif stage == "sidecar_complete":
        logger.info(
            "[data] sidecar 写入完成 "
            f"section={event.get('section')} "
            f"records={event.get('record_count')} "
            f"path={event.get('path')}"
        )
    elif stage == "validation_start":
        logger.info(
            "[data] 校验开始 "
            f"modality={event.get('modality')} "
            f"parents={event.get('total')} "
            f"records={event.get('record_count')}"
        )
    elif stage == "validation_progress":
        logger.info(
            "[data] 校验进度 "
            f"modality={event.get('modality')} "
            f"{event.get('processed')}/{event.get('total')} "
            f"parent={event.get('parent_sample_id')} "
            f"events={event.get('event_count')}"
        )
    elif stage == "validation_complete":
        logger.info(
            "[data] 校验完成 "
            f"modality={event.get('modality')} "
            f"parents={event.get('total')} "
            f"records={event.get('record_count')}"
        )
    elif stage == "cache_start":
        logger.info(
            "[data] cache 建立开始 "
            f"parents={event.get('total')} "
            f"records={event.get('record_count')}"
        )
    elif stage == "cache_progress":
        logger.info(
            "[data] cache 处理进度 "
            f"{event.get('processed')}/{event.get('total')} "
            f"parent={event.get('parent_sample_id')} "
            f"events={event.get('event_count')}"
        )
    elif stage == "cache_complete":
        logger.info(
            "[data] cache 建立完成 "
            f"parents={event.get('total')} "
            f"records={event.get('record_count')} "
            f"path={event.get('path')}"
        )
    elif stage == "feature_cache_start":
        logger.info(
            "[features] cache 构建开始 "
            f"view={event.get('view_name')} "
            f"parents={event.get('parent_count')} "
            f"records={event.get('record_count')}"
        )
    elif stage == "feature_cache_progress":
        logger.info(
            "[features] cache 处理进度 "
            f"view={event.get('view_name')} "
            f"{event.get('processed')}/{event.get('total')} "
            f"parent={event.get('parent_sample_id')} "
            f"events={event.get('event_count')} "
            f"records={event.get('record_count')}"
        )
    elif stage == "feature_cache_complete":
        logger.info(
            "[features] cache 构建完成 "
            f"view={event.get('view_name')} "
            f"parents={event.get('parent_count')} "
            f"records={event.get('record_count')}"
        )
    elif stage == "feature_cache_split_complete":
        logger.info(
            "[features] cache split 完成 "
            f"split={event.get('split_name')} "
            f"bundles={event.get('bundle_count')} "
            f"source_records={event.get('source_record_count')}"
        )
    elif stage in {"l1_start", "l2_start"}:
        task = stage.split("_", 1)[0].upper()
        stable8_count = event.get("stable8_valid_train_events")
        stable8_suffix = (
            f" stable8_valid_train_events={int(stable8_count)}"
            if stable8_count is not None
            else ""
        )
        start_epoch = event.get("start_epoch")
        epoch_suffix = (
            f" start_epoch={int(start_epoch)}" if start_epoch is not None else ""
        )
        resumed_from = event.get("resumed_from_epoch")
        resume_suffix = (
            f" resumed_from_epoch={int(resumed_from)}"
            if resumed_from is not None
            else ""
        )
        logger.info(
            f"[{task}] 训练开始 "
            f"train_parents={event.get('train_parents')} "
            f"validation_parents={event.get('validation_parents')}"
            f"{stable8_suffix}"
            f"{epoch_suffix}"
            f"{resume_suffix}"
        )
    elif stage in {"l1_step", "l2_step"}:
        step = int(event.get("step", 0) or 0)
        total = int(event.get("total", 0) or 0)
        task = stage.split("_", 1)[0].upper()
        logger.info(
            f"[{task}] 训练进度 "
            f"epoch={event.get('epoch')} "
            f"step={step}/{total} "
            f"loss={float(event.get('loss', 0.0)):.6f}"
        )
    elif stage in {"l1_epoch", "l2_epoch"}:
        task = stage.split("_", 1)[0].upper()
        validation = event.get("validation")
        if isinstance(validation, Mapping):
            logger.info(
                f"[{task}] epoch 完成 "
                f"epoch={event.get('epoch')} "
                f"accuracy={float(validation.get('accuracy', 0.0)):.6f} "
                f"macro_f1={float(validation.get('macro_f1', 0.0)):.6f} "
                f"elapsed={float(event.get('elapsed_seconds', 0.0)):.1f}s"
            )
    elif stage in {"l1_complete", "l2_complete"}:
        task = stage.split("_", 1)[0].upper()
        logger.info(
            f"[{task}] 训练完成 "
            f"selected_epoch={event.get('selected_epoch')} "
            f"checkpoint={event.get('path')}"
        )
        if event.get("manifest_path"):
            logger.info("[%s] 总 manifest 已更新 path=%s", task, event.get("manifest_path"))
    elif stage == "complete":
        logger.info(
            "[data] 数据产物和 sidecar 索引处理完成 "
            f"parents={event.get('parent_count')} "
            f"excluded={event.get('excluded_parent_count', 0)} "
            f"TR={event.get('tr_record_count')} "
            f"RD={event.get('rd_record_count')} "
            f"pairs={event.get('pairing_count')}"
        )
    # TODO：后续将前端显式输出替换为统一产品日志和进度事件协议。


def handle_raw_data(args: argparse.Namespace) -> Path:
    """执行 CQ08 数据产物写出、sidecar 索引和总 manifest 校验。"""

    project_root = Path(args.project_root).resolve()
    configure_logging(project_root)
    logger = get_logger("cli")
    adapter = CQ08Adapter(
        project_root,
        source_root=args.source_root,
        split_path=args.split_path,
        parent_limit=args.parent_limit,
    )
    manifest = build_manifest(
        adapter,
        project_root,
        materialize=True,
        overwrite=bool(args.overwrite),
        progress_callback=_print_progress,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
    )
    target = write_manifest(
        manifest,
        project_root,
        path=args.manifest_path,
        overwrite=bool(args.overwrite),
        # build_manifest 已完成唯一一次完整 TR/RD 文件校验与路径边界校验。
        verify_files=True,
        validate=False,
        progress_callback=_print_progress,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
    )
    cache_result = build_training_cache(
        project_root,
        manifest_path=args.manifest_path,
        cache_path=args.cache_path,
        overwrite=bool(args.overwrite),
        # 复用刚完成校验的 manifest；cache 登记阶段只更新 descriptor。
        manifest=manifest,
        verify_files=False,
        progress_callback=_print_progress,
        progress_interval=DEFAULT_PROGRESS_INTERVAL,
    )
    target = Path(cache_result["manifest_path"]).resolve()
    checked = manifest
    checked.audit["cache"] = dict(cache_result["cache"])
    written = json.loads(target.read_text(encoding="utf-8"))
    declared_hash = written.pop("manifest_sha256", None)
    if declared_hash != canonical_json_hash(written):
        raise DataIntegrityError(f"写入后的 manifest hash 不一致：{target}")
    tr_parts = checked.tr_index.get("parts") or []
    rd_parts = checked.rd_index.get("parts") or []
    logger.info(
        "[data] 索引已建立 "
        f"manifest={target} "
        f"TR_index={tr_parts[0].get('path') if tr_parts else ''} "
        f"RD_index={rd_parts[0].get('path') if rd_parts else ''} "
        f"TR={len(checked.tr_records)} "
        f"RD_frames={int(checked.rd_index.get('record_count', 0))} "
        f"pairs={len(checked.tr_records)} "
        f"excluded={int(checked.audit.get('excluded_parent_count', 0))} "
        f"cache={checked.audit.get('cache', {}).get('path', '')}"
    )
    # TODO：后续将汇总信息交给前端状态模型，不在命令入口拼接展示文本。
    return target


def main(argv: Sequence[str] | None = None) -> int:
    """解析并执行产品命令。"""

    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.handle_raw_data:
        parser.print_help()
        return 0
    try:
        handle_raw_data(args)
    except (DataError, FileExistsError, OSError, ValueError, LoggerError) as exc:
        try:
            configure_logging(Path(args.project_root).resolve())
            get_logger("cli").error("[data] 处理失败：%s", exc)
        except LoggerError:
            # 配置本身损坏时不能再依赖文件 handler，标准 logging 的
            # lastResort handler 仍会把错误送到控制台。
            get_logger("cli").error("[data] 处理失败：%s", exc)
        return 2
    finally:
        close_logging()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "handle_raw_data", "main"]
