"""Single reproducible command-line entry point for V27c-r22."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_r22_seed, summarize_r22_seeds
from .protocol import R22_ARMS, load_r22_context, prepare_r22_manifest
from .smoke import run_smoke
from .training import run_r22_training_step, train_r22_arm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V27c-r22 matched independent L2 motion retraining"
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="verify frozen inputs and write the immutable manifest")
    commands.add_parser("smoke", help="run necessary alignment and initial-replay checks")
    step = commands.add_parser("training-step", help="run one CPU update without writing a formal artifact")
    step.add_argument("--arm", required=True, choices=R22_ARMS)
    train = commands.add_parser("train", help="train one matched L2 arm through epoch 20")
    train.add_argument("--arm", required=True, choices=R22_ARMS)
    train.add_argument("--seed", required=True, type=int)
    evaluate = commands.add_parser("evaluate", help="evaluate the paired arms for one seed")
    evaluate.add_argument("--seed", required=True, type=int)
    commands.add_parser("summarize", help="summarize all three prespecified seed reports")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        config, binding, path, digest = prepare_r22_manifest(root, args.config)
        _print(
            {
                "status": "prepared",
                "experiment_id": config.experiment_id,
                "manifest_path": str(path),
                "manifest_sha256": digest,
                "input_binding": binding,
            }
        )
        return 0
    context = load_r22_context(root, args.config)
    if args.command == "smoke":
        _print(run_smoke(context))
        return 0
    if args.command == "training-step":
        _print(run_r22_training_step(context, arm=args.arm))
        return 0
    if args.command == "train":
        _print(train_r22_arm(context, arm=args.arm, seed=int(args.seed)))
        return 0
    if args.command == "evaluate":
        _print(evaluate_r22_seed(context, seed=int(args.seed)))
        return 0
    if args.command == "summarize":
        _print(summarize_r22_seeds(context))
        return 0
    raise AssertionError(f"unhandled V27c-r22 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
