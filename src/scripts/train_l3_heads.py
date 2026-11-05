"""Train the independent L3 knownness heads on frozen endpoint representations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train independent L3 knownness heads")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=Path("src/config/manifests.json"))
    parser.add_argument("--l1-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l2-checkpoint-path", type=Path, default=None)
    parser.add_argument("--l1-config-path", type=Path, default=None)
    parser.add_argument("--l2-config-path", type=Path, default=None)
    parser.add_argument("--model-config-path", type=Path, default=Path("src/config/model.json"))
    parser.add_argument(
        "--fit-feature-cache-manifest-path",
        type=Path,
        default=Path("data/cq08/cache/features/l3/head_fit/feature-cache-manifest.json"),
    )
    parser.add_argument(
        "--validation-feature-cache-manifest-path",
        type=Path,
        default=Path("data/cq08/cache/features/l3/validation/feature-cache-manifest.json"),
    )
    parser.add_argument(
        "--representation-root",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/representations"),
        help="冻结 endpoint representation 缓存目录",
    )
    parser.add_argument("--task", choices=("l1", "l2", "both"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--l3-config-path", type=Path, default=Path("src/config/radar_ln3_l3.json"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--head-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--l1-route-loss", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--l1-target-weight", type=float, default=None)
    parser.add_argument("--l1-clutter-weight", type=float, default=None)
    parser.add_argument("--l1-endpoint-weight", type=float, default=None)
    parser.add_argument("--l1-bce-unknown-weight", type=float, default=None)
    parser.add_argument("--l1-bird-loss-weight", type=float, default=None)
    parser.add_argument(
        "--l1-loss-mode",
        choices=("prefix_endpoint", "endpoint_tail"),
        default=None,
    )
    parser.add_argument("--l1-tail-weight", type=float, default=None)
    parser.add_argument("--l1-tail-events", type=int, default=None)
    parser.add_argument("--l1-target-prefix-weight", type=float, default=None)
    parser.add_argument("--l1-clutter-prefix-weight", type=float, default=None)
    parser.add_argument(
        "--unknown-loss-weight",
        type=float,
        default=None,
        help="L1 route loss 的 Unknown 权重；L2 请使用 --l2-unknown-loss-weight",
    )
    parser.add_argument("--l2-unknown-loss-weight", type=float, default=None)
    parser.add_argument("--l2-classwise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--bird-loss-weight", type=float, default=None)
    parser.add_argument("--unknown-far-limit", type=float, default=None)
    parser.add_argument(
        "--l1-ignore-unknown",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="L1 route loss 中忽略 Unknown parent，不把 Unknown 当作 route 负样本",
    )
    parser.add_argument(
        "--l1-selection-mode",
        choices=("fixed", "operating_point"),
        default=None,
    )
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--projection-dim", type=int, default=None)
    parser.add_argument("--min-events", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/radar_ln3/l3/heads"),
    )
    parser.add_argument("--l1-output-path", type=Path, default=None)
    parser.add_argument("--l2-output-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-register",
        action="store_true",
        help="实验 head 不登记到正式 product manifest",
    )
    parser.add_argument("--skip-event-cache-validation", action="store_true")
    parser.add_argument(
        "--skip-data-validation",
        action="store_true",
        help="复用前置 data_index 校验结果，不重复扫描完整 TR/RD 文件",
    )
    return parser


def _validate_loss_cli_contract(args: argparse.Namespace) -> None:
    if args.task in {"l2", "both"} and not args.l2_classwise and float(args.bird_loss_weight) != 1.0:
        raise ValueError(
            "非 classwise L2 knownness 不使用 Bird 类别权重；"
            "请将 --bird-loss-weight 设为 1.0 或启用 --l2-classwise"
        )


def _knownness_loss_arguments(args: argparse.Namespace, *, task: str) -> dict[str, Any]:
    """保持 L1 route Unknown 与 L2 Unknown 两个权重的训练契约分离。"""
    return {
        "unknown_loss_weight": float(args.unknown_loss_weight) if task == "l1" else 0.15,
        "l2_unknown_loss_weight": float(args.l2_unknown_loss_weight) if task == "l2" else 1.0,
        "l2_classwise": bool(args.l2_classwise) if task == "l2" else False,
        "bird_loss_weight": float(args.bird_loss_weight) if task == "l2" else 1.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.project_root.resolve()
    # 先检查显式 CLI 的任务边界，再读入两条分支的默认配置。
    if args.task == "l2" and args.unknown_loss_weight not in (None, 0.15):
        raise ValueError("--unknown-loss-weight 只适用于 L1 route loss；L2 请使用 --l2-unknown-loss-weight")
    if args.task == "l2" and (
        args.l1_target_prefix_weight not in (None, 0.0)
        or args.l1_clutter_prefix_weight not in (None, 0.0)
    ):
        raise ValueError("route 前缀辅助权重必须使用 --task l1，不能用于 L2 训练")
    target = root / args.l3_config_path
    try:
        training = json.loads(target.read_text(encoding="utf-8-sig"))["knownness_training"]
        l1 = training["l1"]
        l2 = training["l2"]
        defaults = {
            **training["common"],
            **{f"l1_{name}": value for name, value in l1.items() if name != "unknown_loss_weight"},
            "unknown_loss_weight": l1["unknown_loss_weight"],
            "l2_unknown_loss_weight": l2["unknown_loss_weight"],
            "l2_classwise": l2["classwise"],
            "bird_loss_weight": l2["bird_loss_weight"],
        }
        for name, value in defaults.items():
            if getattr(args, name) is None:
                setattr(args, name, value)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"knownness_training 配置字段无效：{target}") from exc
    _validate_loss_cli_contract(args)
    from radar_ln3.model import load_model_config
    from radar_ln3.data import load_manifest
    from radar_ln3.data.manifest import data_manifest_hash
    from radar_ln3.checkpoint import sha256_file
    from radar_ln3.data.derived import register_product_artifacts
    from radar_ln3.training.l3_heads import (
        materialize_frozen_endpoint_representation,
        endpoint_representation_path,
        load_frozen_endpoint_representation,
        save_frozen_endpoint_representation,
        train_l3_knownness_head,
    )

    config = load_model_config(root, args.model_config_path)
    manifest = load_manifest(
        root / args.manifest_path,
        root,
        verify_files=not bool(args.skip_data_validation),
        allow_unmaterialized=False,
    )
    tasks = ("l1", "l2") if args.task == "both" else (str(args.task),)
    result: dict[str, Any] = {"tasks": {}, "manifest_sha256": None}
    expected_endpoint_sha256 = {
        "l1": sha256_file(
            root / (args.l1_checkpoint_path or config.artifacts.l1_output_path)
        ),
        "l2": sha256_file(
            root / (args.l2_checkpoint_path or config.artifacts.l2_output_path)
        ),
    }
    for task in tasks:
        print(json.dumps({"stage": "knownness_start", "task": task, "epochs": args.epochs, "device": args.device}, ensure_ascii=False), flush=True)
        def load_or_materialize(split_name: str, cache_path: Path, allow_unknown: bool):
            target = endpoint_representation_path(
                root,
                task=task,
                split_name=split_name,
                representation_root=args.representation_root,
            )
            expected_cache_hash = sha256_file(root / cache_path)
            if target.exists():
                try:
                    representation = load_frozen_endpoint_representation(
                        root,
                        target,
                        expected_task=task,
                        expected_source_manifest_sha256=data_manifest_hash(manifest),
                        expected_feature_cache_manifest_sha256=expected_cache_hash,
                        expected_endpoint_sha256=expected_endpoint_sha256,
                        expected_config_sha256=sha256_file(config.path),
                    )
                    print(json.dumps({"stage": "representation_reused", "task": task, "split": split_name, "parents": len(representation.parent_ids), "path": str(target)}, ensure_ascii=False), flush=True)
                    return representation
                except (OSError, RuntimeError, ValueError) as exc:
                    print(json.dumps({"stage": "representation_rebuild", "task": task, "split": split_name, "reason": str(exc)}, ensure_ascii=False), flush=True)
            representation = materialize_frozen_endpoint_representation(
                root,
                task=task,
                manifest_path=args.manifest_path,
                feature_cache_manifest_path=cache_path,
                split_name=split_name,
                training_view=True,
                allow_unknown_head_training=allow_unknown,
                device=args.device,
                batch_size=args.batch_size,
                validate_event_cache=(
                    not args.skip_event_cache_validation and not args.skip_data_validation
                ),
                verify_files=not args.skip_data_validation,
                model_config_path=args.model_config_path,
                l1_checkpoint_path=args.l1_checkpoint_path,
                l2_checkpoint_path=args.l2_checkpoint_path,
                l1_config_path=args.l1_config_path,
                l2_config_path=args.l2_config_path,
            )
            save_frozen_endpoint_representation(
                root,
                representation,
                config=config,
                output_path=target,
                overwrite=True,
            )
            return representation

        fit = load_or_materialize(
            "train", args.fit_feature_cache_manifest_path, True
        )
        validation = load_or_materialize(
            "validation", args.validation_feature_cache_manifest_path, False
        )
        def progress(payload):
            print(json.dumps(payload, ensure_ascii=False), flush=True)

        is_l1 = task == "l1"
        summary = train_l3_knownness_head(
            root,
            task=task,
            fit=fit,
            validation=validation,
            config=config,
            output_path=(
                args.l1_output_path if task == "l1" else args.l2_output_path
            ) or (args.output_root / f"{task}-knownness.pt"),
            device=args.device,
            epochs=args.epochs,
            batch_size=args.head_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            projection_dim=args.projection_dim,
            min_events=args.min_events,
            seed=args.seed,
            overwrite=args.overwrite,
            l1_route_loss=(bool(args.l1_route_loss) if is_l1 else False),
            l1_target_weight=(float(args.l1_target_weight) if is_l1 else 1.0),
            l1_clutter_weight=(float(args.l1_clutter_weight) if is_l1 else 0.75),
            l1_endpoint_weight=(float(args.l1_endpoint_weight) if is_l1 else 0.0),
            l1_bce_unknown_weight=(float(args.l1_bce_unknown_weight) if is_l1 else 1.0),
            l1_bird_loss_weight=(float(args.l1_bird_loss_weight) if is_l1 else 1.0),
            l1_loss_mode=(str(args.l1_loss_mode) if is_l1 else "prefix_endpoint"),
            l1_tail_weight=(float(args.l1_tail_weight) if is_l1 else 0.0),
            l1_tail_events=int(args.l1_tail_events),
            l1_target_prefix_weight=(float(args.l1_target_prefix_weight) if is_l1 else 0.0),
            l1_clutter_prefix_weight=(float(args.l1_clutter_prefix_weight) if is_l1 else 0.0),
            **_knownness_loss_arguments(args, task=task),
            unknown_far_limit=float(args.unknown_far_limit),
            l1_ignore_unknown=(bool(args.l1_ignore_unknown) if is_l1 else False),
            l1_selection_mode=(str(args.l1_selection_mode) if is_l1 else "fixed"),
            progress_callback=progress,
        )
        result["tasks"][task] = summary
        result["manifest_sha256"] = fit.source_manifest_sha256
    if result["tasks"] and not args.skip_register:
        register_product_artifacts(
            root,
            manifest_path=args.manifest_path,
            overwrite=True,
            l3_head_descriptors={
                task: dict(summary["checkpoint"])
                for task, summary in result["tasks"].items()
            },
        )
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    from radar_ln3.logger import LoggerError, close_logging, configure_logging, get_logger

    try:
        configure_logging(args.project_root)
        get_logger("training.l3_heads").info(
            json.dumps(run(args), ensure_ascii=False, indent=2)
        )
    except (OSError, RuntimeError, ValueError, LoggerError) as exc:
        get_logger("training.l3_heads").error("L3 knownness head training failed: %s", exc)
        return 2
    finally:
        close_logging()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
