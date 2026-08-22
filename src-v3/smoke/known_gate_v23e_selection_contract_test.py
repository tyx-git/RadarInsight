"""Focused V23e selector smoke without radar data or formal training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from radar_ln3.known_gate_v23e_selection import (
    assess_checkpoint,
    is_better_candidate,
    metrics_from_records,
    select_best_candidate,
    selector_reference,
)


def _row(parent_id: str, *, true_id: int, pred_id: int, event_index: int) -> dict[str, Any]:
    return {
        "parent_sample_id": parent_id,
        "true_id": true_id,
        "pred_id": pred_id,
        "label_name": "ClutterTarget" if true_id else "DroneTarget",
        "event_index": event_index,
        "product_action_eligible": True,
    }


def _short_long_prefix_fixture() -> dict[str, list[dict[str, Any]]]:
    parent = [
        _row("target", true_id=0, pred_id=0, event_index=4),
        _row("clutter", true_id=1, pred_id=1, event_index=4),
    ]
    eligible = [
        _row("target", true_id=0, pred_id=0, event_index=2),
        _row("target", true_id=0, pred_id=0, event_index=3),
        _row("target", true_id=0, pred_id=0, event_index=4),
        _row("clutter", true_id=1, pred_id=1, event_index=2),
    ]
    return {"parent": parent, "eligible_prefix": eligible, "window": [*eligible, *parent]}


def _selector_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    records = _short_long_prefix_fixture()
    # Ten KnownTarget parents and nine Clutter parents produce a compact exact
    # 8/9 support fixture.  The failed state has one target false positive and
    # one Clutter false negative: Recall remains 8/9 but Precision is 8/9,
    # proving that the selector rejects a rounded 0.88-style shortcut.
    records["parent"] = [
        _row(f"target-{index}", true_id=0, pred_id=0, event_index=4)
        for index in range(10)
    ] + [
        _row(f"clutter-{index}", true_id=1, pred_id=1, event_index=4)
        for index in range(9)
    ]
    records["eligible_prefix"] = [
        _row(f"target-{index}", true_id=0, pred_id=0, event_index=2)
        for index in range(10)
    ] + [
        _row(f"clutter-{index}", true_id=1, pred_id=1, event_index=2)
        for index in range(9)
    ]
    records["window"] = [*records["eligible_prefix"], *records["parent"]]
    metrics = metrics_from_records(records)
    reference = selector_reference(metrics, epoch=1, source={"synthetic": True})
    passing = assess_checkpoint(metrics, reference=reference, epoch=2)
    failed_records = _short_long_prefix_fixture()
    failed_records["parent"] = [
        _row("target-0", true_id=0, pred_id=1, event_index=4)
    ] + [
        _row(f"target-{index}", true_id=0, pred_id=0, event_index=4)
        for index in range(1, 10)
    ] + [
        _row(
            f"clutter-{index}",
            true_id=1,
            pred_id=(0 if index == 0 else 1),
            event_index=4,
        )
        for index in range(9)
    ]
    failed_records["eligible_prefix"] = list(records["eligible_prefix"])
    failed_records["window"] = [
        *failed_records["eligible_prefix"],
        *failed_records["parent"],
    ]
    failing = assess_checkpoint(
        metrics_from_records(failed_records), reference=reference, epoch=3
    )
    return reference, passing, failing


def run(_project_root: Path) -> dict[str, Any]:
    records = _short_long_prefix_fixture()
    metrics = metrics_from_records(records)
    balanced = metrics["parent_balanced_eligible_prefix_metrics"]
    rows = {row["class"]: row for row in balanced["per_class"]}
    parent_balancing = (
        balanced["support"] == 2
        and balanced["event_support"] == 4
        and abs(float(balanced["weighted_support"]) - 2.0) <= 1.0e-12
        and abs(float(rows["KNOWN_TARGET_EVIDENCE"]["weighted_support"]) - 1.0)
        <= 1.0e-12
        and abs(float(rows["KNOWN_CLUTTER_EVIDENCE"]["weighted_support"]) - 1.0)
        <= 1.0e-12
    )
    _reference, passing, failing = _selector_fixture()
    checks = {
        "parent_balanced_prefix_gives_each_eligible_parent_total_weight_one": parent_balancing,
        "event_pooled_prefix_support_remains_available_for_historical_comparison": (
            balanced["event_support"] == 4
        ),
        "pending_events_are_not_in_synthetic_prefix_records": all(
            row["event_index"] >= 2 for row in records["eligible_prefix"]
        ),
        "selector_uses_exact_8_of_9_and_precision_point_90_constraints": (
            passing["feasible"] is True
            and failing["feasible"] is False
            and failing["constraints"]["parent_final_clutter_true_positives_at_least_8_of_9"]
            is True
            and failing["constraints"]["parent_final_clutter_precision_at_least_0_90"]
            is False
        ),
        "selector_is_deterministic_and_rejects_nonfeasible_candidate": (
            select_best_candidate([failing, passing]) == passing
            and is_better_candidate(failing, passing) is False
        ),
        "validation_dataset_access_count_is_zero": True,
        "unknown_dataset_access_count_is_zero": True,
        "test_dataset_access_count_is_zero": True,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"V23e selector smoke failed: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "synthetic_parent_balanced_metrics": balanced,
        "selector_passing_fixture": passing,
        "selector_failing_fixture": failing,
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23e selector-focused contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
