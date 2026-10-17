"""Run V25e's full 20-epoch shared-gradient projection treatment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build.known_gate_v25e import CONFIG
from radar_ln3.known_gate import (
    V25E_SCHEMA_VERSION,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import V25E_SHARED_GRADIENT_PROJECTION_VARIANT
from train.known_gate_common import train_known_gate


EXPECTED_TOTAL_STEPS = 9860
EXPECTED_MIXED_BATCHES = 638
EXPECTED_PURE_BATCHES = 9222


def train(project_root: Path, *, resume_checkpoint: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = (root / CONFIG).resolve()
    config = load_known_gate_config(config_path)
    if (
        config.schema_version != V25E_SCHEMA_VERSION
        or config.variant != V25E_SHARED_GRADIENT_PROJECTION_VARIANT
        or config.train.class_weight_normalization != "batch_weight_sum"
    ):
        raise ValueError("V25e trainer requires its sealed shared-gradient projection config")
    result = train_known_gate(root, config_path, resume_checkpoint=resume_checkpoint)
    telemetry = dict(result.get("shared_gradient_projection_telemetry") or {})
    scope = dict(telemetry.get("by_scope") or {})
    if (
        result.get("status") != "completed"
        or int(scope.get("batch_count", -1)) != EXPECTED_TOTAL_STEPS
        or int(scope.get("mixed_known_clutter_batch_count", -1)) != EXPECTED_MIXED_BATCHES
        or int(scope.get("pure_batch_count", -1)) != EXPECTED_PURE_BATCHES
        or int(scope.get("projection_batch_count", -1)) != EXPECTED_MIXED_BATCHES
        or telemetry.get("projected_modules") != ["tr_encoder", "fusion", "liquid"]
        or telemetry.get("unchanged_modules") != ["rd_encoder", "classifier"]
        or telemetry.get("rd_and_classifier_gradients_exactly_unchanged") is not True
        or any(int(dict(scope.get("projection_count") or {}).get(module, 0)) <= 0 for module in ("tr_encoder", "fusion", "liquid"))
    ):
        raise AssertionError("V25e completed training lacks projection mechanism evidence")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V25e shared-gradient projection treatment.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()
    print(json.dumps(train(args.project_root.resolve(), resume_checkpoint=args.resume_checkpoint), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
