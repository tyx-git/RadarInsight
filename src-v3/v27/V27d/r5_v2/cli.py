"""CLI for the V27d-r5 Bird-route v2 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .audit import audit_error_samples
from .evaluation import evaluate_experiment
from .features import materialize_population_caches
from .protocol import load_config, prepare_manifest
from .training import train_control


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V27d-r5 Bird-route v2")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("materialize")
    subparsers.add_parser("audit")
    train = subparsers.add_parser("train")
    train.add_argument("--seed", type=int, default=42)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, default=42)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        _, path, digest = prepare_manifest(root, args.config)
        _print({"status": "prepared", "manifest_path": str(path), "manifest_sha256": digest})
        return 0
    if args.command == "materialize":
        _print(materialize_population_caches(root, args.config))
        return 0
    if args.command == "audit":
        _print(audit_error_samples(root, args.config))
        return 0
    if args.command == "train":
        _print(train_control(root, seed=args.seed, config_path=args.config))
        return 0
    if args.command == "evaluate":
        _print(evaluate_experiment(root, seed=args.seed, config_path=args.config))
        return 0
    raise AssertionError(f"unhandled V27d-r5-v2 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
