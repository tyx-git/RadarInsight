"""CLI for the isolated V27d-r12 causal prefix aggregation experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_seed, summarize_seeds
from .protocol import R12_SEEDS, prepare_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V27d-r12 causal prefix lower-quartile aggregation"
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, choices=R12_SEEDS, required=True)
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
    elif args.command == "evaluate":
        _print(evaluate_seed(root, seed=args.seed, config_path=args.config))
    elif args.command == "summarize":
        _print(summarize_seeds(root, args.config))
    else:
        raise AssertionError(f"unhandled V27d-r12 command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
