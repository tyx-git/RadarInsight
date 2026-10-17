"""Single entry point for V27d-r14."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluation import build_paired_report, evaluate_arm, summarize_seeds
from .protocol import R14_ARMS, R14_SEEDS, load_context, prepare_manifest
from .training import train_v27d_r14_arm


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the V27d-r14 density-weighted migration")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    train = sub.add_parser("train")
    train.add_argument("--arm", choices=R14_ARMS, required=True)
    train.add_argument("--seed", type=int, choices=R14_SEEDS, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--arm", choices=R14_ARMS, required=True)
    evaluate.add_argument("--seed", type=int, choices=R14_SEEDS, required=True)
    paired = sub.add_parser("paired")
    paired.add_argument("--seed", type=int, choices=R14_SEEDS, required=True)
    sub.add_parser("summary")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        _, path, digest = prepare_manifest(root, args.config)
        result = {"status": "prepared", "manifest_path": str(path), "manifest_sha256": digest}
    else:
        context = load_context(root, args.config)
        if args.command == "train":
            result = train_v27d_r14_arm(context, arm=args.arm, seed=args.seed)
        elif args.command == "evaluate":
            result = evaluate_arm(context, arm=args.arm, seed=args.seed)
        elif args.command == "paired":
            result = build_paired_report(context, seed=args.seed)
        elif args.command == "summary":
            result = summarize_seeds(context)
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
