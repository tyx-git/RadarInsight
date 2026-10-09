"""V27d-r02 命令行入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate
from .followup import run_followup
from .protocol import load_config, prepare_manifest
from .smoke import run_smoke
from .training import train


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run V27d-r02 Bird→Drone experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("smoke")
    train_parser = commands.add_parser("train")
    train_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    eval_parser = commands.add_parser("evaluate")
    eval_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    followup_parser = commands.add_parser("followup")
    followup_parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cuda")
    followup_parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    config = load_config(root, args.config)
    value: Any
    if args.command == "prepare":
        path, manifest = prepare_manifest(root, config)
        value = {
            "status": "prepared",
            "manifest_path": path.relative_to(root).as_posix(),
            "train_parent_count": manifest["split"]["train_parent_count"],
            "validation_parent_count": manifest["split"]["validation_parent_count"],
        }
    elif args.command == "smoke":
        value = run_smoke(root, config)
    elif args.command == "train":
        value = train(root, config, device=args.device)
    elif args.command == "followup":
        value = run_followup(root, config, device=args.device, folds=int(args.folds))
    else:
        value = evaluate(root, config, device=args.device)
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
