"""Single entry point for independently reproducible V27d runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .calibration import fit_dynamic_calibration
from .endpoint import materialize_endpoint_caches
from .evaluation import evaluate_v27d
from .features import materialize_feature_caches
from .protocol import (
    V27D_TASKS,
    V27D_VARIANTS,
    load_context,
    prepare_manifest,
)
from .smoke import run_smoke
from .training import train_heads


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the isolated V27d experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", help="Verify sources and write the V27d manifest")
    subparsers.add_parser("materialize-endpoints", help="Replay frozen L1/L2 classification endpoints")
    materialize = subparsers.add_parser("materialize-features", help="Build a V27d representation cache")
    materialize.add_argument("--variant", choices=V27D_VARIANTS, default="r1")
    subparsers.add_parser("smoke", help="Run V27d contract checks")
    train = subparsers.add_parser("train", help="Train independent knownness and quality heads")
    train.add_argument("--task", choices=V27D_TASKS, required=True)
    train.add_argument("--variant", choices=V27D_VARIANTS, default="r1")
    train.add_argument("--seed", type=int, required=True)
    calibrate = subparsers.add_parser("calibrate", help="Fit dynamic L3 calibration")
    calibrate.add_argument("--variant", choices=V27D_VARIANTS, default="r1")
    calibrate.add_argument("--seed", type=int, required=True)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate heads and dynamic L3")
    evaluate.add_argument("--variant", choices=V27D_VARIANTS, default="r1")
    evaluate.add_argument("--seed", type=int, required=True)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        config, manifest_path, digest = prepare_manifest(root, args.config)
        _print(
            {
                "status": "prepared",
                "experiment_id": "V27d",
                "manifest_path": str(manifest_path),
                "manifest_sha256": digest,
                "config_sha256": config.config_sha256,
            }
        )
        return 0

    context = load_context(root, args.config)
    if args.command == "materialize-endpoints":
        _print(materialize_endpoint_caches(context))
        return 0
    if args.command == "materialize-features":
        _print(materialize_feature_caches(context, variant=args.variant))
        return 0
    if args.command == "smoke":
        result = run_smoke(context)
        _print(result)
        return 0
    if args.command == "train":
        _print(train_heads(context, variant=args.variant, task=args.task, seed=args.seed))
        return 0
    if args.command == "calibrate":
        _print(fit_dynamic_calibration(context, variant=args.variant, seed=args.seed))
        return 0
    if args.command == "evaluate":
        _print(evaluate_v27d(context, variant=args.variant, seed=args.seed))
        return 0
    raise AssertionError(f"unhandled V27d command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
