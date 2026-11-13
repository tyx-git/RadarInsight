"""Train the independent L3 knownness heads on frozen endpoint representations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train independent L3 knownness heads")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest-path", type=Path, default=Path("src/config/manifests.json"))
    parser.add_argument("--model-config-path", type=Path, default=Path("src/config/model.json"))
    parser.add_argument("--task", choices=("l1", "l2", "both"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default=None)
    parser.add_argument("--l3-config-path", type=Path, default=Path("src/config/radar_ln3_l3.json"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path("artifacts/radar_ln3"))
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
        help="跳过数据文件 size/hash 校验（verify_files=False）；manifest 结构与事件路径边界校验仍会执行",
    )
    return parser


def _knownness_settings(root: Path, config_path: Path) -> dict[str, Any]:
    """把 knownness_training 的三段配置读成训练参数字典。"""

    target = root / config_path
    try:
        training = json.loads(target.read_text(encoding="utf-8-sig"))["knownness_training"]
        l1 = training["l1"]
        l2 = training["l2"]
        settings = {
            **training["common"],
            **{f"l1_{name}": value for name, value in l1.items() if name != "unknown_loss_weight"},
            # 默认 L1 目标是 balanced BCE，route loss 权重不进入梯度；保留键位
            # 便于显式 route-loss 实验，配置缺省时沿用历史默认值。
            "unknown_loss_weight": l1.get("unknown_loss_weight", 0.15),
            "l2_unknown_loss_weight": l2["unknown_loss_weight"],
            "l2_classwise": l2["classwise"],
            "bird_loss_weight": l2.get("bird_loss_weight", 1.0),
        }
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"knownness_training 配置字段无效：{target}") from exc
    if not bool(settings["l2_classwise"]) and float(settings["bird_loss_weight"]) != 1.0:
        raise ValueError(
            "非 classwise L2 knownness 不使用 Bird 类别权重；"
            "请将 knownness_training.l2.bird_loss_weight 设为 1.0 或启用 classwise"
        )
    return settings


def _knownness_loss_arguments(settings: dict[str, Any], *, task: str) -> dict[str, Any]:
    """保持 L1 route Unknown 与 L2 Unknown 两个权重的训练契约分离。"""

    return {
        # L2 has its own explicitly named weight.  Do not pass the L1 route
        # weight through the L2 call as a dummy value.
        "unknown_loss_weight": float(settings["unknown_loss_weight"]) if task == "l1" else None,
        "l2_unknown_loss_weight": float(settings["l2_unknown_loss_weight"]) if task == "l2" else 1.0,
        "l2_classwise": bool(settings["l2_classwise"]) if task == "l2" else False,
        "bird_loss_weight": float(settings["bird_loss_weight"]) if task == "l2" else 1.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from radar_ln3.artifacts import is_default_artifact_root, l3_layout
    from radar_ln3.model import load_model_config
    from radar_ln3.data import load_manifest
    from radar_ln3.data.manifest import data_manifest_hash
    from radar_ln3.checkpoint import sha256_file
    from radar_ln3.data.derived import register_product_artifacts
    from radar_ln3.training.l3_heads import (
        build_head_fit_dataset,
        build_head_validation_dataset,
        materialize_frozen_endpoint_representation,
        endpoint_representation_path,
        load_frozen_endpoint_representation,
        save_frozen_endpoint_representation,
        train_l3_knownness_head,
    )
    from radar_ln3.training.l3 import _load_endpoint_models
    from radar_ln3.features import load_feature_cache_bundle

    root = args.project_root.resolve()
    layout = l3_layout(root, args.root)
    settings = _knownness_settings(root, args.l3_config_path)
    config = load_model_config(
        root, args.model_config_path, artifact_root=layout.root
    )
    manifest = load_manifest(
        root / args.manifest_path,
        root,
        verify_files=not bool(args.skip_data_validation),
        allow_unmaterialized=False,
    )
    tasks = ("l1", "l2") if args.task == "both" else (str(args.task),)
    result: dict[str, Any] = {"tasks": {}, "manifest_sha256": None}
    expected_endpoint_sha256 = {
        "l1": sha256_file(root / layout.l1_checkpoint),
        "l2": sha256_file(root / layout.l2_checkpoint),
    }
    manifest_hash = data_manifest_hash(manifest)
    config_hash = sha256_file(config.path)
    event_cache_validation_enabled = not args.skip_event_cache_validation
    effective_device = str(args.device or settings["device"])
    batch_size = int(args.batch_size or settings["batch_size"])
    endpoint_models = None
    feature_cache_bundles: dict[str, tuple[dict[str, Any], str]] = {}
    split_datasets = {
        "train": build_head_fit_dataset(
            root,
            manifest_path=args.manifest_path,
            manifest=manifest,
            validate_event_cache=event_cache_validation_enabled,
        ),
        "validation": build_head_validation_dataset(
            root,
            manifest_path=args.manifest_path,
            manifest=manifest,
            validate_event_cache=event_cache_validation_enabled,
        ),
    }
    cache_manifest_paths = dict.fromkeys(
        (layout.head_fit_feature_cache, layout.validation_feature_cache)
    )
    cache_manifest_hashes = {
        str(cache_path): sha256_file(root / cache_path)
        for cache_path in cache_manifest_paths
    }
    for task in tasks:
        print(json.dumps({"stage": "knownness_start", "task": task, "epochs": settings["epochs"], "device": effective_device}, ensure_ascii=False), flush=True)
        def load_or_materialize(split_name: str, cache_path: Path, allow_unknown: bool):
            nonlocal endpoint_models
            target = endpoint_representation_path(
                root,
                task=task,
                split_name=split_name,
                representation_root=layout.representations,
            )
            cache_key = str(cache_path)
            expected_cache_hash = cache_manifest_hashes[cache_key]
            if target.exists():
                try:
                    representation = load_frozen_endpoint_representation(
                        root,
                        target,
                        expected_task=task,
                        expected_source_manifest_sha256=manifest_hash,
                        expected_feature_cache_manifest_sha256=expected_cache_hash,
                        expected_view_hash=split_datasets[split_name].view.view_hash,
                        expected_endpoint_sha256=expected_endpoint_sha256,
                        expected_config_sha256=config_hash,
                    )
                    print(json.dumps({"stage": "representation_reused", "task": task, "split": split_name, "parents": len(representation.parent_ids), "path": str(target)}, ensure_ascii=False), flush=True)
                    return representation
                except (OSError, RuntimeError, ValueError) as exc:
                    print(json.dumps({"stage": "representation_rebuild", "task": task, "split": split_name, "reason": str(exc)}, ensure_ascii=False), flush=True)
            cached_bundle = feature_cache_bundles.get(cache_key)
            if cached_bundle is None:
                cached_bundle = (
                    load_feature_cache_bundle(
                        root,
                        root / cache_path,
                        expected_source_manifest_hash=manifest_hash,
                        verify_integrity=True,
                    ),
                    expected_cache_hash,
                )
                feature_cache_bundles[cache_key] = cached_bundle
            print(
                json.dumps(
                    {
                        "stage": "representation_materialize_start",
                        "task": task,
                        "split": split_name,
                        "device": effective_device,
                        "batch_size": batch_size,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if endpoint_models is None:
                endpoint_models = _load_endpoint_models(
                    root,
                    device=torch.device(effective_device),
                    config=config,
                    config_sha256=config_hash,
                    source_manifest_sha256=manifest_hash,
                    l1_checkpoint_path=layout.l1_checkpoint,
                    l2_checkpoint_path=layout.l2_checkpoint,
                )
            representation = materialize_frozen_endpoint_representation(
                root,
                task=task,
                manifest_path=args.manifest_path,
                feature_cache_manifest_path=cache_path,
                split_name=split_name,
                training_view=True,
                allow_unknown_head_training=allow_unknown,
                device=effective_device,
                batch_size=batch_size,
                validate_event_cache=event_cache_validation_enabled,
                verify_files=not args.skip_data_validation,
                model_config_path=args.model_config_path,
                l1_checkpoint_path=layout.l1_checkpoint,
                l2_checkpoint_path=layout.l2_checkpoint,
                manifest=manifest,
                config=config,
                endpoint_sha256=expected_endpoint_sha256,
                feature_caches=cached_bundle[0],
                feature_cache_sha256=cached_bundle[1],
                manifest_sha256=manifest_hash,
                dataset=split_datasets[split_name],
                endpoint_models=endpoint_models,
            )
            print(
                json.dumps(
                    {
                        "stage": "representation_materialize_complete",
                        "task": task,
                        "split": split_name,
                        "parents": len(representation.parent_ids),
                        "path": str(target),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            save_frozen_endpoint_representation(
                root,
                representation,
                config=config,
                config_sha256=config_hash,
                output_path=target,
                overwrite=True,
            )
            return representation

        fit = load_or_materialize(
            "train", layout.head_fit_feature_cache, True
        )
        validation = load_or_materialize(
            "validation", layout.validation_feature_cache, False
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
            output_path=layout.l1_head if is_l1 else layout.l2_head,
            device=effective_device,
            epochs=settings["epochs"],
            batch_size=settings["head_batch_size"],
            learning_rate=settings["learning_rate"],
            weight_decay=settings["weight_decay"],
            hidden_dim=settings["hidden_dim"],
            projection_dim=settings["projection_dim"],
            min_events=settings["min_events"],
            seed=settings["seed"],
            overwrite=args.overwrite,
            l1_route_loss=(bool(settings["l1_route_loss"]) if is_l1 else False),
            l1_target_weight=(float(settings["l1_target_weight"]) if is_l1 else 1.0),
            l1_clutter_weight=(float(settings["l1_clutter_weight"]) if is_l1 else 0.75),
            l1_endpoint_weight=(float(settings["l1_endpoint_weight"]) if is_l1 else 0.0),
            l1_bce_unknown_weight=(float(settings["l1_bce_unknown_weight"]) if is_l1 else 1.0),
            l1_bird_loss_weight=(float(settings["l1_bird_loss_weight"]) if is_l1 else 1.0),
            l1_loss_mode=(str(settings["l1_loss_mode"]) if is_l1 else "prefix_endpoint"),
            l1_tail_weight=(float(settings["l1_tail_weight"]) if is_l1 else 0.0),
            l1_tail_events=int(settings["l1_tail_events"]),
            l1_target_prefix_weight=(float(settings["l1_target_prefix_weight"]) if is_l1 else 0.0),
            l1_clutter_prefix_weight=(float(settings["l1_clutter_prefix_weight"]) if is_l1 else 0.0),
            **_knownness_loss_arguments(settings, task=task),
            unknown_far_limit=float(settings["unknown_far_limit"]),
            l1_ignore_unknown=(bool(settings["l1_ignore_unknown"]) if is_l1 else False),
            l1_selection_mode=(str(settings["l1_selection_mode"]) if is_l1 else "fixed"),
            progress_callback=progress,
        )
        result["tasks"][task] = summary
        result["manifest_sha256"] = fit.source_manifest_sha256
    register = not args.skip_register
    if register and not is_default_artifact_root(layout.root):
        print(
            json.dumps(
                {
                    "stage": "register_skipped",
                    "root": layout.root.as_posix(),
                    "reason": "非默认 Root 的 head 不登记到正式 product manifest",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        register = False
    if result["tasks"] and register:
        register_product_artifacts(
            root,
            manifest_path=args.manifest_path,
            overwrite=True,
            manifest=manifest,
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
