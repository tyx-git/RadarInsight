"""Train the V22a KnownGate RD temporal-residual local experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from radar_ln3.known_gate import V22A_RD_TEMPORAL_RESIDUAL_VARIANT
from train.known_gate_common import train_known_gate


CONFIG_RELATIVE = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V22a-cq08-liquid-known-gate-rd-temporal-residual-seed43-config.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the full 20-epoch V22a KnownGate RD temporal residual experiment."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config or CONFIG_RELATIVE
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    result = train_known_gate(
        root,
        config_path,
        resume_checkpoint=(
            None
            if args.resume_checkpoint is None
            else (args.resume_checkpoint if args.resume_checkpoint.is_absolute() else root / args.resume_checkpoint).resolve()
        ),
    )
    if result.get("variant") != V22A_RD_TEMPORAL_RESIDUAL_VARIANT:
        raise ValueError("V22a training returned a non-V22a variant")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
