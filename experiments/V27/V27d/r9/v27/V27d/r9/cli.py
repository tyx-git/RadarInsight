"""CLI for the isolated V27d-r9 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_seed, summarize_seeds
from .features import materialize_profile_caches
from .protocol import R9_SEEDS, prepare_manifest
from .training import train_seed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V27d-r9 causal Doppler profile-shape experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("materialize")
    train = commands.add_parser("train")
    train.add_argument("--seed", type=int, choices=R9_SEEDS, required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, choices=R9_SEEDS, required=True)
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
        _print(materialize_profile_caches(root, args.config))
    elif args.command == "train":
        _print(train_seed(root, seed=args.seed, config_path=args.config))
    elif args.command == "evaluate":
        _print(evaluate_seed(root, seed=args.seed, config_path=args.config))
    elif args.command == "summarize":
        _print(summarize_seeds(root, config_path=args.config))
    else:
        raise AssertionError(f"unhandled V27d-r9 command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
