"""V27d-r17 唯一命令入口。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .data import build_r17_data
from .evaluation import evaluate_frozen, evaluate_l2_checkpoint
from .protocol import artifact_path, load_config, prepare_manifest
from .training import train_l2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V27d-r17 OriData Excel TR-only experiment")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare-data")
    commands.add_parser("prepare")
    evaluate = commands.add_parser("evaluate-frozen")
    evaluate.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    evaluate_l2 = commands.add_parser("evaluate-l2")
    evaluate_l2.add_argument("--checkpoint", type=Path, default=None)
    evaluate_l2.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    train = commands.add_parser("train-l2")
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--device", choices=("cpu", "cuda", "auto"), default=None)
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    config = load_config(root, args.config)
    if args.command == "prepare-data":
        _print(build_r17_data(root, config).__dict__)
    elif args.command == "prepare":
        _, path, digest = prepare_manifest(root, args.config)
        _print({"status": "prepared", "manifest_path": str(path), "manifest_sha256": digest})
    elif args.command == "evaluate-frozen":
        _print(evaluate_frozen(root, config, device=args.device))
    elif args.command == "evaluate-l2":
        checkpoint = args.checkpoint
        if checkpoint is None:
            checkpoint = artifact_path(
                root,
                config.artifacts["checkpoint_template"],
                task="l2",
                seed=int(config.training["seeds"][0]),
            )
        _print(
            evaluate_l2_checkpoint(
                root, config, checkpoint_path=checkpoint, device=args.device
            )
        )
    elif args.command == "train-l2":
        _print(train_l2(root, config, seed=int(args.seed), device=args.device))
    else:
        raise AssertionError(f"unhandled r17 command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
