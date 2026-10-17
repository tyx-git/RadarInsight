"""Single reproducible CLI for the isolated V27c motion experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_v27c_motion, summarize_v27c_seeds
from .features import materialize_feature_caches
from .protocol import load_v27c_context, prepare_v27c_manifest
from .training import train_v27c_motion


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the V27c causal TR-motion ablation")
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="verify frozen V27a inputs and write the V27c manifest")
    commands.add_parser("materialize", help="materialize cache-aligned numeric motion sidecars")
    train = commands.add_parser("train", help="train one fixed-epoch causal motion residual")
    train.add_argument("--seed", required=True, type=int)
    evaluate = commands.add_parser("evaluate", help="write one paired known-only validation report")
    evaluate.add_argument("--seed", required=True, type=int)
    commands.add_parser("summarize", help="summarize all three immutable seed reports")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        config, binding, path, digest = prepare_v27c_manifest(root, args.config)
        _print({"status": "prepared", "experiment_id": config.experiment_id, "manifest_path": str(path), "manifest_sha256": digest, "input_binding": binding})
        return 0
    context = load_v27c_context(root, args.config)
    if args.command == "materialize":
        _print(materialize_feature_caches(context))
        return 0
    if args.command == "train":
        _print(train_v27c_motion(context, seed=int(args.seed)))
        return 0
    if args.command == "evaluate":
        _print(evaluate_v27c_motion(context, seed=int(args.seed)))
        return 0
    if args.command == "summarize":
        _print(summarize_v27c_seeds(context))
        return 0
    raise AssertionError(f"unhandled V27c command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
