"""Single reproducible command line entry point for V27b-r6."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import (
    compare_r6_arms,
    evaluate_r6_arm,
    write_r6_arm_report,
    write_r6_comparison_report,
)
from .features import materialize_feature_caches
from .protocol import R6_ARMS, R6_ROLES, load_r6_context, prepare_r6_manifest
from .training import train_r6_head


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the frozen-endpoint V27b-r6 ablation")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="verify frozen inputs and write the r6 manifest")
    commands.add_parser("materialize", help="materialize sealed mixed-pool feature sidecars")
    train = commands.add_parser("train", help="fit one fixed-epoch knownness arm")
    train.add_argument("--arm", choices=R6_ARMS, required=True)
    train.add_argument("--seed", type=int, required=True)
    evaluate = commands.add_parser("evaluate", help="score one trained knownness arm")
    evaluate.add_argument("--arm", choices=R6_ARMS, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--role", choices=R6_ROLES, required=True)
    compare = commands.add_parser("compare", help="produce one paired arm diagnostic")
    compare.add_argument("--seed", type=int, required=True)
    compare.add_argument("--role", choices=R6_ROLES, required=True)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        config, binding, path, digest = prepare_r6_manifest(root, args.config)
        _print({
            "status": "prepared",
            "experiment_id": config.experiment_id,
            "manifest_path": str(path),
            "manifest_sha256": digest,
            "input_binding": binding,
        })
        return 0
    context = load_r6_context(root, args.config)
    if args.command == "materialize":
        _print(materialize_feature_caches(context))
        return 0
    if args.command == "train":
        _print(train_r6_head(context, arm=args.arm, seed=args.seed))
        return 0
    if args.command == "evaluate":
        report = evaluate_r6_arm(context, arm=args.arm, seed=args.seed, role=args.role)
        path = write_r6_arm_report(
            context, report, arm=args.arm, seed=args.seed, role=args.role
        )
        _print({"report_path": str(path), "report": report})
        return 0
    if args.command == "compare":
        report = compare_r6_arms(context, seed=args.seed, role=args.role)
        path = write_r6_comparison_report(context, report, seed=args.seed, role=args.role)
        _print({"report_path": str(path), "report": report})
        return 0
    raise AssertionError(f"unhandled V27b-r6 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
