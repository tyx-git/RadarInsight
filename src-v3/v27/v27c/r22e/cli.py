"""Single entry point for the provenance-bound V27c-r22e evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .evaluation import evaluate_r22e_seed, summarize_r22e_seeds
from .protocol import load_r22e_context, prepare_r22e_manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen V27c-r22 checkpoints with the NumPy 2 compatibility correction"
    )
    parser.add_argument("--project-root", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="seal the derived evaluator provenance manifest")
    evaluate = commands.add_parser("evaluate", help="evaluate one completed r22 seed")
    evaluate.add_argument("--seed", required=True, type=int)
    commands.add_parser("summarize", help="aggregate the three pre-registered r22 seeds")
    return parser


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare":
        context, path = prepare_r22e_manifest(root)
        _print(
            {
                "status": "prepared",
                "experiment_id": context.manifest["experiment_id"],
                "manifest_path": str(path),
                "manifest_sha256": context.manifest_sha256,
                "upstream_r22_manifest_sha256": context.upstream.manifest_sha256,
            }
        )
        return 0
    context = load_r22e_context(root)
    if args.command == "evaluate":
        _print(evaluate_r22e_seed(context, seed=int(args.seed)))
        return 0
    if args.command == "summarize":
        _print(summarize_r22e_seeds(context))
        return 0
    raise AssertionError(f"unhandled V27c-r22e command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

