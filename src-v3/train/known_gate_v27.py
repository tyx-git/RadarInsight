"""Deprecated compatibility wrapper for the unified V27 CLI.

Historical name retained only so an accidental old command fails into the
correct task-aware path rather than training an L2-only model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from v27.v27_baseline.runtime import load_v27_context
from v27.v27_baseline.training import train_v27_task


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated: use python -m v27.v27_baseline.cli train --task ...")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--task", choices=("known_gate", "dbb_specialist"), default="dbb_specialist")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(train_v27_task(load_v27_context(args.project_root.resolve()), task=args.task, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
