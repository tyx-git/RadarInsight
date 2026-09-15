"""CLI for the isolated V27d-r5 Bird-route experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_experiment
from .features import materialize_population_caches
from .protocol import load_config, prepare_manifest
from .smoke import run_smoke
from .training import train_arm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V27d-r5 Bird-route knownness experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    subparsers.add_parser("materialize")
    subparsers.add_parser("smoke")
    train = subparsers.add_parser("train")
    train.add_argument("--arm", choices=("control", "stable8_treatment"), required=True)
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
        config, path, digest = prepare_manifest(root, args.config)
        _print({"status": "prepared", "manifest_path": str(path), "manifest_sha256": digest, "config_sha256": config["config_sha256"]})
        return 0
    if args.command == "materialize":
        _print(materialize_population_caches(root, args.config))
        return 0
    if args.command == "smoke":
        _print(run_smoke(root, args.config))
        return 0
    if args.command == "train":
        _print(train_arm(root, arm=args.arm, seed=args.seed, config_path=args.config))
        return 0
    if args.command == "evaluate":
        _print(evaluate_experiment(root, seed=args.seed, config_path=args.config))
        return 0
    raise AssertionError(f"unhandled V27d-r5 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
