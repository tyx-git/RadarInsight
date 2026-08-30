"""Contract and semantic smoke test for V26f.

The smoke test is intentionally bounded and does not train, fit thresholds,
or run the mixed validation/test forward.  It verifies the prepared immutable
contract, the scalar/vector cache validator, and the mutually exclusive strict
open-set action boundaries.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from radar_ln3.known_gate_v26f import (
    ACTION_NAMES,
    CONTRACT,
    CONTRACT_SCHEMA,
    EXPECTED_PARENT_COUNTS,
    FOLD_COUNT,
    READINESS,
    READINESS_SCHEMA,
    validate_feature_cache,
    validate_prepared_contract,
)
from radar_ln3.open_set_architecture import ProductAction, resolve_semantic_action


def _cache_smoke() -> dict[str, object]:
    """Exercise both cache-validity ranks without touching project data."""

    with tempfile.TemporaryDirectory(prefix="radar-ln3-v26f-smoke-") as directory:
        path = Path(directory) / "cache.npz"
        ids = np.asarray(["s0", "s1"], dtype="U2")
        values = np.asarray([[0.25, 0.0], [0.0, 0.75]], dtype=np.float32)
        vector_valid = np.asarray([[True, False], [False, True]], dtype=np.bool_)
        np.savez(path, sample_ids=ids, values=values, valid=vector_valid)
        vector = validate_feature_cache(
            path,
            expected_sample_ids=["s0", "s1"],
            width=2,
            valid_rank=2,
            description="synthetic vector cache",
        )
        scalar_path = Path(directory) / "scalar-cache.npz"
        scalar_values = np.asarray([[0.1, 0.2], [0.0, 0.0]], dtype=np.float32)
        scalar_valid = np.asarray([True, False], dtype=np.bool_)
        np.savez(
            scalar_path,
            sample_ids=ids,
            values=scalar_values,
            valid=scalar_valid,
        )
        scalar = validate_feature_cache(
            scalar_path,
            expected_sample_ids=["s0", "s1"],
            width=2,
            valid_rank=1,
            description="synthetic scalar cache",
        )
    return {
        "vector_valid_rank": 2,
        "vector_valid_event_count": vector["valid_event_count"],
        "scalar_valid_rank": 1,
        "scalar_valid_event_count": scalar["valid_event_count"],
    }


def _semantic_smoke() -> dict[str, str]:
    cases = {
        "target_and_subclass": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=True,
            known_clutter_accepted=False,
            dbb_subclass=ProductAction.DRONE,
            dbb_subclass_accepted=True,
        ).action,
        "clutter": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=False,
            known_clutter_accepted=True,
        ).action,
        "target_without_subclass": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=True,
            known_clutter_accepted=False,
            dbb_subclass=ProductAction.DRONE,
            dbb_subclass_accepted=False,
        ).action,
        "neither": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=False,
            known_clutter_accepted=False,
        ).action,
        "conflict": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=True,
            known_clutter_accepted=True,
        ).action,
        "ineligible": resolve_semantic_action(
            observation_eligible=False,
            known_target_accepted=True,
            known_clutter_accepted=True,
        ).action,
    }
    expected = {
        "target_and_subclass": ProductAction.DRONE,
        "clutter": ProductAction.CLUTTER,
        "target_without_subclass": ProductAction.UNKNOWN_REJECTED,
        "neither": ProductAction.UNKNOWN_REJECTED,
        "conflict": ProductAction.PENDING_ABSTAIN,
        "ineligible": ProductAction.PENDING_ABSTAIN,
    }
    if cases != expected:
        raise AssertionError(
            "V26f strict-open-set action boundary differs: "
            f"actual={cases}, expected={expected}"
        )
    return {name: value.value for name, value in cases.items()}


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    contract, readiness = validate_prepared_contract(root)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or int(contract.get("validation_access_count", -1)) != 0
        or int(contract.get("test_access_count", -1)) != 0
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("unknown_threshold_fit_exposure", -1)) != 0
        or EXPECTED_PARENT_COUNTS["test"].get("ClutterTarget") != 8
    ):
        raise AssertionError("V26f contract scope or support is invalid")
    if set(ACTION_NAMES) != {
        "DroneTarget",
        "BirdTarget",
        "BalloonTarget",
        "ClutterTarget",
        "UNKNOWN_REJECTED",
        "PENDING_ABSTAIN",
    }:
        raise AssertionError("V26f action space changed")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V26f-frozen-mixed-open-set-audit-smoke-v1",
        "contract_path": str(root / CONTRACT),
        "readiness_path": str(root / READINESS),
        "contract_sha256": __import__(
            "radar_ln3.known_gate_v26f", fromlist=["sha256_file"]
        ).sha256_file(root / CONTRACT),
        "fold_count": FOLD_COUNT,
        "validation_parent_counts": EXPECTED_PARENT_COUNTS["val"],
        "test_parent_counts": EXPECTED_PARENT_COUNTS["test"],
        "cache_validator": _cache_smoke(),
        "semantic_actions": _semantic_smoke(),
        "training": "not_executed",
        "threshold_fitting": "not_executed",
        "mixed_forward": "not_executed",
        "deployment_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the V26f frozen mixed audit contract.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
