"""V27d-r16 command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import build_paired_report, evaluate_arm, summarize_seeds
from .protocol import R16_ARMS, load_context, prepare_manifest
from .smoke import run_smoke
from .training import train_r16_arm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V27d-r16 RD-frame continuous-time recurrence experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("smoke")
    train = commands.add_parser("train")
    train.add_argument("--arm", choices=R16_ARMS, required=True)
    train.add_argument("--seed", type=int, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, required=True)
    commands.add_parser("summarize")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        _, path, digest = prepare_manifest(root, args.config)
        _print({"status": "prepared", "manifest_path": str(path), "manifest_sha256": digest})
    elif args.command == "smoke":
        _print(run_smoke(load_context(root, args.config)))
    elif args.command == "train":
        _print(train_r16_arm(load_context(root, args.config), arm=args.arm, seed=args.seed))
    elif args.command == "evaluate":
        context = load_context(root, args.config)
        _print(
            {
                "control": evaluate_arm(context, arm="control", seed=args.seed),
                "treatment": evaluate_arm(context, arm="treatment", seed=args.seed),
                "paired": build_paired_report(context, seed=args.seed),
            }
        )
    elif args.command == "summarize":
        _print(summarize_seeds(load_context(root, args.config)))
    else:
        raise AssertionError(f"unhandled V27d-r16 command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
