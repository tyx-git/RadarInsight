"""r18 command-line entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .diagnosis import diagnose
from .evaluation import evaluate_comparison
from .training import train_arm
from .protocol import load_config, prepare_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V27d-r18 source-stratified radial-speed audit")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    diagnosis = commands.add_parser("diagnose")
    diagnosis.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--seed", type=int, default=42)
    evaluation.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    train = commands.add_parser("train")
    train.add_argument("--arm", choices=("full", "radial_speed_ablation"), required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config = load_config(root, args.config)
    if args.command == "prepare":
        _, path, digest = prepare_manifest(root, args.config)
        value: Any = {"status": "prepared", "manifest_path": str(path), "manifest_sha256": digest}
    elif args.command == "diagnose":
        value = diagnose(root, config, device=args.device)
    elif args.command == "evaluate":
        value = evaluate_comparison(
            root, config, seed=int(args.seed), device=args.device
        )
    else:
        value = train_arm(root, config, arm=args.arm, seed=int(args.seed), device=args.device)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
