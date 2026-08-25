"""Run V25d's full present-class-mean KnownGate training treatment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build.known_gate_v25d import CONFIG
from radar_ln3.known_gate import (
    KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
    V25D_SCHEMA_VERSION,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import V25D_PRESENT_CLASS_MEAN_VARIANT
from train.known_gate_v21 import train_known_gate


def train(project_root: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = (root / CONFIG).resolve()
    config = load_known_gate_config(config_path)
    if (
        config.schema_version != V25D_SCHEMA_VERSION
        or config.variant != V25D_PRESENT_CLASS_MEAN_VARIANT
        or config.train.class_weight_normalization
        != KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION
    ):
        raise ValueError("V25d trainer requires its sealed present-class-mean config")
    result = train_known_gate(root, config_path, resume_checkpoint=resume_checkpoint)
    telemetry = dict(result.get("present_class_mean_loss_mass_telemetry") or {})
    final = dict(dict(telemetry.get("by_scope") or {}).get("final") or {})
    if (
        result.get("status") != "completed"
        or telemetry.get("normalization") != KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION
        or final.get("contributing_batch_count") != 9860
        or final.get("mixed_batches_exactly_equal_class_mass") is not True
        or final.get("pure_batches_have_single_class_mass") is not True
    ):
        raise AssertionError("V25d completed training lacks its loss-mass mechanism evidence")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train V25d's present-class-mean KnownGate treatment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(
        json.dumps(
            train(args.project_root.resolve(), resume_checkpoint=args.resume_checkpoint),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
