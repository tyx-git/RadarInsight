"""Prepare an explicit model artifact root for a full or single-stage run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "l1", "l2", "l3"), required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    from radar_ln3.artifacts import prepare_artifact_directory, prepare_artifact_stage

    project_root = args.project_root.resolve()
    root = args.root if args.root.is_absolute() else project_root / args.root
    root = root.resolve()
    try:
        root.relative_to(project_root)
    except ValueError as exc:
        raise ValueError("--root must point inside the project directory") from exc
    if root == project_root:
        raise ValueError("--root cannot be the project directory itself")
    if args.scope == "all":
        prepared = prepare_artifact_directory(root, overwrite=args.overwrite)
    else:
        prepared = prepare_artifact_stage(root, args.scope, overwrite=args.overwrite)
    print(prepared)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"artifact root preparation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
