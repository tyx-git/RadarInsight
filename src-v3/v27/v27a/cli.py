"""The single command-line entry point for the V27a experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.runtime import sha256_file

from .cache import materialize_frozen_logit_caches
from .checkpoint import load_v27a_checkpoint
from .evaluation import (
    build_paired_report,
    evaluate_v27a_arm,
    write_arm_report,
    write_paired_report,
)
from .protocol import (
    V27A_ARMS,
    load_v27a_context,
    prepare_v27a_manifest,
    v27a_artifact_path,
)
from .runtime import configured_device
from .smoke import run_smoke
from .training import train_v27a_arm
from .model import build_v27a_model


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated V27a experiment")
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Project root containing src-v3/",
    )
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare", help="Audit inputs and create the V27a manifest")
    subparsers.add_parser("smoke", help="Run structural contract smoke tests")
    subparsers.add_parser("materialize", help="Cache frozen V27 L2 logits and evidence")

    train = subparsers.add_parser("train", help="Train one V27a residual arm")
    train.add_argument("--arm", choices=V27A_ARMS, required=True)
    train.add_argument("--seed", type=int, default=None)
    train.add_argument("--epochs", type=int, default=None)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate control or a treatment arm")
    evaluate.add_argument("--arm", choices=("control", *V27A_ARMS), required=True)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--checkpoint", type=Path, default=None)
    evaluate.add_argument("--bootstrap-replicates", type=int, default=0)

    paired = subparsers.add_parser("paired", help="Evaluate all three arms and write paired report")
    paired.add_argument("--seed", type=int, default=42)
    paired.add_argument("--bootstrap-replicates", type=int, default=None)
    return parser


def _json_result(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _treatment_checkpoint(context: Any, arm: str, seed: int) -> Path:
    summary_path = v27a_artifact_path(
        context.project_root,
        context.config.artifacts.training_summary_template,
        arm=arm,
        seed=seed,
    )
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        selected = summary.get("global_best_checkpoint_path")
        if isinstance(selected, str) and selected.strip():
            candidate = Path(selected)
            return candidate if candidate.is_absolute() else context.project_root / candidate
    return v27a_artifact_path(
        context.project_root,
        context.config.artifacts.epoch_checkpoint_template,
        arm=arm,
        seed=seed,
        epoch=context.config.training.primary_epoch,
    )


def _evaluate_one(
    context: Any,
    *,
    arm: str,
    seed: int,
    checkpoint: Path | None,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if arm == "control":
        return evaluate_v27a_arm(
            context,
            arm="control",
            seed=seed,
            split_name="val",
            bootstrap_replicates=bootstrap_replicates,
        )
    model = build_v27a_model(context.config, arm=arm, seed=seed)
    target = checkpoint or _treatment_checkpoint(context, arm, seed)
    checkpoint_payload = load_v27a_checkpoint(
        target,
        context=context,
        model=model,
        arm=arm,
        seed=seed,
        map_location="cpu",
    )
    return evaluate_v27a_arm(
        context,
        arm=arm,
        seed=seed,
        split_name="val",
        model=model,
        checkpoint_epoch=int(checkpoint_payload["epoch"]),
        checkpoint_sha256=sha256_file(target),
        bootstrap_replicates=bootstrap_replicates,
        device=torch.device("cpu"),
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        config, binding, manifest_path, digest = prepare_v27a_manifest(root, args.config)
        _json_result(
            {
                "status": "prepared",
                "experiment_id": "V27a",
                "manifest_path": str(manifest_path),
                "manifest_sha256": digest,
                "binding_sha256": binding["binding_sha256"],
                "config_sha256": config.config_sha256,
            }
        )
        return 0

    context = load_v27a_context(root, args.config)
    if args.command == "smoke":
        _json_result(run_smoke(context))
        return 0
    if args.command == "materialize":
        _json_result(
            materialize_frozen_logit_caches(
                context,
                device=configured_device(context),
            )
        )
        return 0
    if args.command == "train":
        _json_result(
            train_v27a_arm(
                context,
                arm=args.arm,
                seed=args.seed,
                epochs_override=args.epochs,
            )
        )
        return 0
    if args.command == "evaluate":
        report = _evaluate_one(
            context,
            arm=args.arm,
            seed=args.seed,
            checkpoint=args.checkpoint,
            bootstrap_replicates=args.bootstrap_replicates,
        )
        path = write_arm_report(context, report, arm=args.arm, seed=args.seed)
        _json_result({"report_path": str(path), "report": report})
        return 0
    if args.command == "paired":
        seed = int(args.seed)
        replicates = (
            context.config.training.bootstrap_replicates
            if args.bootstrap_replicates is None
            else int(args.bootstrap_replicates)
        )
        control = _evaluate_one(
            context,
            arm="control",
            seed=seed,
            checkpoint=None,
            # Arm reports are immutable canonical per-arm diagnostics.  The
            # paired report owns the requested bootstrap and must not change
            # the Control report payload on a rerun.
            bootstrap_replicates=0,
        )
        mechanism = _evaluate_one(
            context,
            arm="mechanism",
            seed=seed,
            checkpoint=None,
            bootstrap_replicates=0,
        )
        main_report = _evaluate_one(
            context,
            arm="main",
            seed=seed,
            checkpoint=None,
            bootstrap_replicates=0,
        )
        arm_paths = {
            arm: str(
                write_arm_report(context, report, arm=arm, seed=seed)
            )
            for arm, report in (
                ("control", control),
                ("mechanism", mechanism),
                ("main", main_report),
            )
        }
        paired_report = build_paired_report(
            context,
            control=control,
            mechanism=mechanism,
            main=main_report,
            seed=seed,
            bootstrap_replicates=replicates,
        )
        paired_path = write_paired_report(context, paired_report)
        _json_result(
            {
                "status": "paired_evaluation_complete",
                "arm_report_paths": arm_paths,
                "paired_report_path": str(paired_path),
                "paired_report": paired_report,
            }
        )
        return 0
    raise AssertionError(f"unhandled V27a command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
