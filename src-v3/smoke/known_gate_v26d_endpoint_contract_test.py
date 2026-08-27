"""Static contract smoke test for the read-only V26d endpoint diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build.known_gate_v26d_endpoint import require_prepared
from radar_ln3.known_gate_v26d_endpoint import (
    CONTROL_EPOCH,
    FOLD_COUNT,
    TREATMENT_EPOCH,
)


def run(project_root: Path) -> dict[str, Any]:
    contract, readiness = require_prepared(project_root.resolve())
    checkpoint = dict(contract.get("checkpoint_protocol") or {})
    scope = dict(contract.get("scope") or {})
    folds = list(contract.get("folds") or ())
    if (
        checkpoint.get("outer_held_out_checkpoint_selection") is not False
        or checkpoint.get("per_fold_epoch_selection") is not False
        or checkpoint.get("checkpoint_scan") is not False
        or checkpoint.get("l2_checkpoint")
        != "sealed V26c epoch-15 L2 outputs are reused unchanged"
        or scope.get("training_executed") is not False
        or any(int(scope.get(name, -1)) != 0 for name in (
            "unknown_training_exposure",
            "unknown_normalization_exposure",
            "unknown_oof_prediction_count",
            "unknown_threshold_fit_exposure",
            "validation_access_count",
            "test_access_count",
        ))
        or len(folds) != FOLD_COUNT
        or any(int(row.get("control_epoch", -1)) != CONTROL_EPOCH for row in folds)
        or any(int(row.get("treatment_epoch", -1)) != TREATMENT_EPOCH for row in folds)
        or readiness.get("training_executed") is not False
        or readiness.get("deployment_eligible") is not False
    ):
        raise AssertionError("V26d endpoint diagnostic contract is invalid")
    return {
        "status": "passed",
        "fold_count": FOLD_COUNT,
        "control_epoch": CONTROL_EPOCH,
        "treatment_epoch": TREATMENT_EPOCH,
        "training_executed": False,
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V26d's endpoint diagnostic contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
