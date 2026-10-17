"""CLI for the independent V28 Bird-route r1 experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .diagnostics import fit_calibration, run_diagnostic
from .evidence import prepare_evidence
from .features import materialize_role
from .profile_head import fit_profile_head
from .protocol import load_context, load_unsealed_context, write_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V28 Bird-route r1 profile-shape experiment")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-evidence")
    sub.add_parser("materialize-features")
    sub.add_parser("diagnostic")
    train = sub.add_parser("train")
    train.add_argument("--seed", type=int, default=42)
    sub.add_parser("calibrate")
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    if args.command == "prepare-evidence":
        context = load_unsealed_context(root)
        result = prepare_evidence(context)
        write_manifest(root)
    elif args.command == "materialize-features":
        context = load_unsealed_context(root)
        result = [
            materialize_role(context, role="representation_fit", split="train"),
            materialize_role(context, role="representation_development", split="train"),
            materialize_role(context, role="calibration", split="val"),
        ]
        write_manifest(root)
    elif args.command == "train":
        result = fit_profile_head(load_context(root), seed=int(args.seed))
    elif args.command == "diagnostic":
        result = run_diagnostic(load_context(root))
    elif args.command == "calibrate":
        result = fit_calibration(load_context(root))
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
