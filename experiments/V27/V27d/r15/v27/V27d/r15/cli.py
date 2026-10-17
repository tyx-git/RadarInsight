"""V27d-r15 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_seed, evaluate_single_arm, summarize_seeds
from .features import materialize_observation_cache
from .protocol import R15_ARMS, load_context, prepare_manifest
from .training import train_r15_arm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V27d-r15 V10 observation-increment L1 migration")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("materialize")
    train = commands.add_parser("train")
    train.add_argument("--arm", choices=R15_ARMS, required=True)
    train.add_argument("--seed", type=int, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--arm", choices=R15_ARMS, default=None)
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
    elif args.command == "materialize":
        _print(materialize_observation_cache(load_context(root, args.config)))
    elif args.command == "train":
        _print(train_r15_arm(load_context(root, args.config), arm=args.arm, seed=args.seed))
    elif args.command == "evaluate":
        context = load_context(root, args.config)
        if args.arm is None:
            _print(evaluate_seed(context, seed=args.seed))
        else:
            _print(evaluate_single_arm(context, arm=args.arm, seed=args.seed))
    elif args.command == "summarize":
        _print(summarize_seeds(load_context(root, args.config)))
    else:
        raise AssertionError(f"unhandled V27d-r15 command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
