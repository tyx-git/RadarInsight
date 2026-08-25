"""Contract smoke test for V25g's frozen causal action policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build.known_gate_v25g import (
    CONTRACT_SCHEMA,
    EXPERIMENT_CONTRACT,
    READINESS,
    V25G_SCHEMA,
    _validate_source,
)
from radar_ln3.known_gate_v25g_policy import (
    V25G_MAX_THRESHOLD_CANDIDATES,
    V25G_PENDING_ACTION,
    apply_accept_pending_policy,
    class_metric,
    first_causal_decisions,
    select_train_only_threshold,
    summarize_policy_records,
)


def _synthetic_records() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for parent_id, true_id, confidences, predictions in (
        ("p-target", 0, (0.40, 0.60, 0.92), (0, 0, 0)),
        ("p-clutter", 1, (0.55, 0.62, 0.88), (0, 1, 1)),
    ):
        for event_index, (confidence, pred_id) in enumerate(
            zip(confidences, predictions)
        ):
            rows.append(
                {
                    "parent_sample_id": parent_id,
                    "track_id": parent_id,
                    "source_name": "synthetic",
                    "label_name": "ClutterTarget" if true_id else "DroneTarget",
                    "true_id": true_id,
                    "pred_id": pred_id,
                    "confidence": confidence,
                    "event_index": event_index,
                    "window_length": event_index + 1,
                    "product_action_eligible": event_index + 1 >= 3,
                }
            )
    return rows


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    contract = json.loads((root / EXPERIMENT_CONTRACT).read_text(encoding="utf-8"))
    readiness = json.loads((root / READINESS).read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != V25G_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("training") != "not_executed"
        or readiness.get("calibration") != "train_only_frozen_checkpoint_in_sample"
        or readiness.get("validation_used_for_threshold_fit") is not False
        or readiness.get("OOF") != "not_executed"
        or readiness.get("test_payload_access_count") != 0
        or readiness.get("unknown_payload_access_count") != 0
        or int((contract.get("resource_safety") or {}).get("max_threshold_candidates", -1))
        != V25G_MAX_THRESHOLD_CANDIDATES
        or (contract.get("resource_safety") or {}).get("sequential_split_loading") is not True
    ):
        raise AssertionError("V25g readiness is stale or outside the frozen scope")
    source = _validate_source(root)
    if source["summary"].get("test_evaluation") != "not_executed":
        raise AssertionError("V25g smoke detected test access in V25a source")

    rows = _synthetic_records()
    policy = apply_accept_pending_policy(rows, threshold=0.8)
    by_key = {
        (str(row["parent_sample_id"]), int(row["event_index"])): row
        for row in policy
    }
    if by_key[("p-target", 0)]["policy_action"] != V25G_PENDING_ACTION:
        raise AssertionError("V25g first event was not mapped to Pending")
    if by_key[("p-target", 1)]["policy_action"] != V25G_PENDING_ACTION:
        raise AssertionError("V25g second event was not mapped to Pending")
    if by_key[("p-target", 2)]["policy_action_id"] != 0:
        raise AssertionError("V25g target acceptance did not preserve local argmax")
    decisions = first_causal_decisions(policy)
    if len(decisions) != 2 or any(not row["policy_accepted"] for row in decisions):
        raise AssertionError("V25g first causal acceptance did not work")
    metrics = summarize_policy_records(decisions, scope="synthetic_parent_decisions")
    if class_metric(metrics, "KNOWN_TARGET_EVIDENCE", "recall") != 1.0:
        raise AssertionError("V25g synthetic target recall is incorrect")
    if class_metric(metrics, "KNOWN_CLUTTER_EVIDENCE", "recall") != 1.0:
        raise AssertionError("V25g synthetic clutter recall is incorrect")

    calibration = select_train_only_threshold(rows)
    if calibration.get("is_oof") is not False:
        raise AssertionError("V25g train-only calibration was mislabeled as OOF")
    if int(calibration.get("candidate_count", 0)) < 1:
        raise AssertionError("V25g threshold candidates are empty")
    if int(calibration.get("candidate_count", 0)) > V25G_MAX_THRESHOLD_CANDIDATES:
        raise AssertionError("V25g threshold candidate bound was exceeded")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25g-causal-decision-policy-smoke-v1",
        "policy_pending_is_not_unknown": True,
        "minimum_evidence_events": 3,
        "train_only_calibration_not_labeled_oof": True,
        "training": "not_executed",
        "OOF": "not_executed",
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V25g causal action policy.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
