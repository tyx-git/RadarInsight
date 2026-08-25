"""Contract smoke test for the V25f frozen checkpoint selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build.known_gate_v25f import EXPECTED_EPOCHS, EXPERIMENT_CONTRACT, READINESS, _source
from radar_ln3.known_gate_v25f_selection import (
    V25F_SELECTOR_SCHEMA,
    assess_checkpoint,
    metrics_from_records,
    reference_from_records,
    select_best,
)


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    contract_path = root / EXPERIMENT_CONTRACT
    readiness_path = root / READINESS
    if not contract_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V25f contract/readiness is incomplete")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    if (
        contract.get("selector_schema") != V25F_SELECTOR_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("selector_schema") != V25F_SELECTOR_SCHEMA
        or readiness.get("training") != "not_executed"
        or readiness.get("test_payload_access_count") != 0
        or readiness.get("unknown_payload_access_count") != 0
    ):
        raise AssertionError("V25f readiness is stale or outside selector-only scope")
    source = _source(root)
    detail_records = {
        scope: [
            dict(row)
            for row in list(dict(source["detail"].get("records") or {}).get(scope) or ())
        ]
        for scope in ("parent", "eligible_prefix", "window")
    }
    reference = reference_from_records(
        detail_records,
        epoch=14,
        source={"variant": source["config"].variant},
    )
    candidate = assess_checkpoint(
        metrics_from_records(detail_records),
        reference=reference,
        epoch=14,
        prefix_action_contract={
            "minimum_evidence_events": 3,
            "events_before_minimum": "PENDING_ABSTAIN",
            "local_argmax_is_product_action": False,
        },
    )
    if (
        candidate.get("strict_product_safe_feasible") is not True
        or candidate.get("diagnostic_utility_feasible") is not True
        or select_best([candidate], mode="strict_product_safe") is None
        or select_best([candidate], mode="diagnostic_utility") is None
    ):
        raise AssertionError("V25f legacy checkpoint does not satisfy selector contract")
    if tuple(sorted(source["epoch_paths"])) != EXPECTED_EPOCHS:
        raise AssertionError("V25f source epoch checkpoint set is incomplete")
    if source["summary"].get("test_evaluation") != "not_executed":
        raise AssertionError("V25f smoke detected test access in V25a source")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25f-known-gate-causal-prefix-checkpoint-utility-selection-smoke-v1",
        "selector_schema": V25F_SELECTOR_SCHEMA,
        "source_v25a_epoch_count": len(source["epoch_paths"]),
        "legacy_epoch_strict_safe": True,
        "diagnostic_frontier_is_not_adoption_gate": True,
        "training": "not_executed",
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V25f selector contracts.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
