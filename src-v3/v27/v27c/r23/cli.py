"""Single reproducible command-line entry point for V27c-r23."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

# PyTorch reads this before CUDA initialization; keep deterministic r23 runs
# reproducible without requiring a shell-specific environment setup.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .evaluation import evaluate_r23_seed, evaluate_r23_single_arm, summarize_r23_seeds
from .protocol import R23_ARMS, load_r23_context, prepare_r23_manifest
from .smoke import run_smoke
from .training import run_r23_training_step, train_r23_arm


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run V27c-r23 matched independent L2 motion retraining"
    )
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="verify frozen inputs and write the immutable manifest")
    commands.add_parser("smoke", help="run necessary alignment and initial-replay checks")
    step = commands.add_parser("training-step", help="run one CPU update without writing a formal artifact")
    step.add_argument("--arm", required=True, choices=R23_ARMS)
    train = commands.add_parser("train", help="train one factorial L2 arm with global validation selection")
    train.add_argument("--arm", required=True, choices=R23_ARMS)
    train.add_argument("--seed", required=True, type=int)
    evaluate = commands.add_parser("evaluate", help="evaluate the paired arms for one seed")
    evaluate.add_argument("--seed", required=True, type=int)
    evaluate.add_argument("--arm", choices=R23_ARMS, default=None, help="evaluate only one trained arm")
    commands.add_parser("summarize", help="summarize both prespecified seed reports")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        config, binding, path, digest = prepare_r23_manifest(root, args.config)
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
    context = load_r23_context(root, args.config)
    if args.command == "smoke":
        _print(run_smoke(context))
        return 0
    if args.command == "training-step":
        _print(run_r23_training_step(context, arm=args.arm))
        return 0
    if args.command == "train":
        _print(train_r23_arm(context, arm=args.arm, seed=int(args.seed)))
        return 0
    if args.command == "evaluate":
        if args.arm is None:
            _print(evaluate_r23_seed(context, seed=int(args.seed)))
        else:
            _print(evaluate_r23_single_arm(context, arm=args.arm, seed=int(args.seed)))
        return 0
    if args.command == "summarize":
        _print(summarize_r23_seeds(context))
        return 0
    raise AssertionError(f"unhandled V27c-r23 command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
