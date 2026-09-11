"""Single command-line entry point for the independently reproducible V27b run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file

from .cache import materialize_v27b_feature_caches
from .checkpoint import load_v27b_checkpoint
from .evaluation import (
    build_paired_report,
    evaluate_v27b_arm,
    write_arm_report,
    write_paired_report,
)
from .protocol import (
    V27B_ARMS,
    load_v27b_context,
    prepare_v27b_manifest,
    v27b_artifact_path,
)
from .runtime import configured_device
from .smoke import run_smoke
from .training import train_v27b
from .model import build_v27b_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated V27b experiment")
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Project root containing src-v3/",
    )
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Verify V27a inputs and write the V27b manifest")
    subparsers.add_parser("smoke", help="Run structural V27b contract tests")
    subparsers.add_parser("materialize", help="Materialize frozen V27a features")

    train = subparsers.add_parser("train", help="Train the conditional treatment head")
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate one V27b arm")
    evaluate.add_argument("--arm", choices=("control", *V27B_ARMS), required=True)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=0)

    paired = subparsers.add_parser("paired", help="Write a paired control/treatment report")
    paired.add_argument("--seed", type=int, default=42)
    paired.add_argument("--bootstrap-replicates", type=int, default=None)
    paired.add_argument("--checkpoint", type=Path, default=None)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _checkpoint_target(context: Any, seed: int, checkpoint: Path | None) -> Path:
    if checkpoint is not None:
        return resolve_project_path(context.project_root, checkpoint, must_exist=True)
    return v27b_artifact_path(
        context.project_root,
        context.config.artifacts.epoch_checkpoint_template,
        seed=seed,
        epoch=context.config.training.primary_epoch,
    )


def _evaluate_treatment(
    context: Any,
    *,
    seed: int,
    checkpoint: Path | None,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    model = build_v27b_model(context.config, seed=seed)
    target = _checkpoint_target(context, seed, checkpoint)
    payload = load_v27b_checkpoint(
        target,
        context=context,
        model=model,
        seed=seed,
        map_location="cpu",
    )
    return evaluate_v27b_arm(
        context,
        arm="conditional",
        split_name="val",
        model=model,
        checkpoint_epoch=int(payload["epoch"]),
        checkpoint_sha256=sha256_file(target),
        bootstrap_replicates=bootstrap_replicates,
        device=torch.device("cpu"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        config, binding, manifest_path, digest = prepare_v27b_manifest(root, args.config)
        _print(
            {
                "status": "prepared",
                "experiment_id": "V27b",
                "manifest_path": str(manifest_path),
                "manifest_sha256": digest,
                "input_binding_sha256": canonical_sha256(binding),
                "config_sha256": config.config_sha256,
            }
        )
        return 0

    context = load_v27b_context(root, args.config)
    if args.command == "smoke":
        _print(run_smoke(context))
        return 0
    if args.command == "materialize":
        _print(
            materialize_v27b_feature_caches(
                context,
                device=configured_device(context),
            )
        )
        return 0
    if args.command == "train":
        _print(train_v27b(context, seed=args.seed, epochs_override=args.epochs))
        return 0
    if args.command == "evaluate":
        if args.arm == "control":
            report = evaluate_v27b_arm(
                context,
                arm="control",
                split_name="val",
                bootstrap_replicates=args.bootstrap_replicates,
            )
        else:
            report = _evaluate_treatment(
                context,
                seed=args.seed,
                checkpoint=args.checkpoint,
                bootstrap_replicates=args.bootstrap_replicates,
            )
        report_path = write_arm_report(
            context, report, arm=args.arm, seed=args.seed
        )
        _print({"report_path": str(report_path), "report": report})
        return 0
    if args.command == "paired":
        seed = int(args.seed)
        replicates = (
            context.config.training.bootstrap_replicates
            if args.bootstrap_replicates is None
            else int(args.bootstrap_replicates)
        )
        control = evaluate_v27b_arm(
            context,
            arm="control",
            split_name="val",
            bootstrap_replicates=0,
        )
        treatment = _evaluate_treatment(
            context,
            seed=seed,
            checkpoint=args.checkpoint,
            bootstrap_replicates=0,
        )
        arm_paths = {
            "control": str(write_arm_report(context, control, arm="control", seed=seed)),
            "conditional": str(
                write_arm_report(context, treatment, arm="conditional", seed=seed)
            ),
        }
        paired_report = build_paired_report(
            context,
            control=control,
            treatment=treatment,
            seed=seed,
            bootstrap_replicates=replicates,
        )
        paired_path = write_paired_report(context, paired_report, seed=seed)
        _print(
            {
                "status": "paired_evaluation_complete",
                "arm_report_paths": arm_paths,
                "paired_report_path": str(paired_path),
                "paired_report": paired_report,
            }
        )
        return 0
    raise AssertionError(f"unhandled V27b command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
