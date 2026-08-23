"""Run V24c's sealed full 20-epoch strict-same-second-skip treatment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build.known_gate_v24c import CONFIG
from train.known_gate_v21 import train_known_gate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the sealed V24c strict same-second state-skip KnownGate diagnostic."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    resume = args.resume_checkpoint
    if resume is not None and not resume.is_absolute():
        resume = root / resume
    print(
        json.dumps(
            train_known_gate(root, root / CONFIG, resume_checkpoint=resume),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
