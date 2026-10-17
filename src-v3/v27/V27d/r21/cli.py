"""r21 command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_comparison
from .protocol import load_config, prepare_manifest
from .training import train_arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V27d-r21 radial adapter-only experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    train = commands.add_parser("train")
    train.add_argument("--arm", choices=("zero_control", "radial_decomposed"), required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config = load_config(root, args.config)
    value: Any
    if args.command == "prepare":
        _, path, digest = prepare_manifest(root, args.config)
        value = {"status": "prepared", "manifest_path": str(path), "manifest_sha256": digest}
    elif args.command == "train":
        value = train_arm(root, config, arm=args.arm, seed=int(args.seed), device=args.device)
    else:
        value = evaluate_comparison(root, config, seed=int(args.seed), device=args.device)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
