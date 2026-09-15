"""CLI for the isolated V27d-r8 paired attribution experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_all
from .features import materialize_population_caches
from .protocol import R8_ARMS, prepare_manifest
from .smoke import run_smoke
from .summary import summarize
from .training import train_arm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V27d-r8 joint V21e course and stable8")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("materialize")
    commands.add_parser("smoke")
    train = commands.add_parser("train")
    train.add_argument("--arm", choices=R8_ARMS, required=True)
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
    elif args.command == "materialize":
        _print(materialize_population_caches(root, args.config))
    elif args.command == "smoke":
        _print(run_smoke(root, args.config))
    elif args.command == "train":
        _print(train_arm(root, arm=args.arm, seed=args.seed, config_path=args.config))
    elif args.command == "evaluate":
        _print(evaluate_all(root, seed=args.seed, config_path=args.config))
    elif args.command == "summarize":
        _print(summarize(root, args.config))
    else:
        raise AssertionError(f"unhandled V27d-r8 command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
