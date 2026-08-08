from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from report.rd_physical_paired import evaluate_completed_runs


def _metrics(
    *,
    dbb_precision: float,
    dbb_recall: float,
    dbb_f1: float,
    cu_precision: float,
    cu_recall: float,
    cu_f1: float,
    dbb_to_cu: int,
    cu_to_dbb: int,
) -> dict[str, Any]:
    return {
        "DBB": {"precision": dbb_precision, "recall": dbb_recall, "f1": dbb_f1, "support": 209},
        "CU": {"precision": cu_precision, "recall": cu_recall, "f1": cu_f1, "support": 23},
        "errors": {"DBB_to_CU": dbb_to_cu, "CU_to_DBB": cu_to_dbb},
        "overall": {
            "accuracy": 0.95,
            "macro_precision": (dbb_precision + cu_precision) / 2.0,
            "macro_recall": (dbb_recall + cu_recall) / 2.0,
            "macro_f1": (dbb_f1 + cu_f1) / 2.0,
            "balanced_accuracy": (dbb_recall + cu_recall) / 2.0,
        },
    }


def _arm(metrics: dict[str, Any], *, seed: int, arm: str) -> dict[str, Any]:
    epochs = {
        epoch: {
            "epoch": epoch,
            "global_step": epoch * 541,
            "train_loss": 1.0 / epoch,
            "parent_order_sha256": f"seed-{seed}-epoch-{epoch}",
            "parent_metrics": deepcopy(metrics),
            "checkpoint_selection": {},
        }
        for epoch in range(1, 21)
    }
    return {
        "arm": arm,
        "seed": seed,
        "best_epoch": 20,
        "best_metrics": deepcopy(metrics),
        "epoch20_metrics": deepcopy(metrics),
        "epochs": epochs,
        "paired_initialization": {"initial_logits_sha256": f"initial-seed-{seed}"},
    }


def _runs() -> dict[int, dict[str, dict[str, Any]]]:
    control = _metrics(
        dbb_precision=0.97,
        dbb_recall=0.97,
        dbb_f1=0.97,
        cu_precision=0.74,
        cu_recall=0.74,
        cu_f1=0.74,
        dbb_to_cu=6,
        cu_to_dbb=6,
    )
    treatment = _metrics(
        dbb_precision=0.98,
        dbb_recall=0.96,
        dbb_f1=0.97,
        cu_precision=0.92,
        cu_recall=0.91,
        cu_f1=0.915,
        dbb_to_cu=5,
        cu_to_dbb=2,
    )
    return {
        seed: {
            "control": _arm(control, seed=seed, arm="control"),
            "treatment": _arm(treatment, seed=seed, arm="treatment"),
        }
        for seed in (42, 43, 44)
    }


def run(project_root: Path) -> dict[str, Any]:
    passing = evaluate_completed_runs(_runs())
    if passing["stage_one_passed"] is not True or passing["passing_seed_count"] != 3:
        raise AssertionError("V11 paired report rejected a fully passing synthetic protocol")
    failing_runs = _runs()
    for seed in (42, 43, 44):
        failing_runs[seed]["treatment"]["best_metrics"]["CU"]["recall"] = 0.85
        failing_runs[seed]["treatment"]["epochs"][20]["parent_metrics"]["CU"]["recall"] = 0.85
    failing = evaluate_completed_runs(failing_runs)
    if failing["stage_one_passed"] is not False or failing["stage_two_status"] != "blocked_by_stage_one":
        raise AssertionError("V11 paired report did not block a weak-CU synthetic protocol")
    mismatch_runs = _runs()
    mismatch_runs[42]["treatment"]["epochs"][7]["parent_order_sha256"] = "different-order"
    order_mismatch_rejected = False
    try:
        evaluate_completed_runs(mismatch_runs)
    except ValueError as exc:
        order_mismatch_rejected = "parent order differs" in str(exc)
    if not order_mismatch_rejected:
        raise AssertionError("V11 paired report accepted mismatched parent orders")
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V11-rd-physical-paired-report-test-v1",
        "passing_protocol_stage_one": passing["stage_one_passed"],
        "passing_seed_count": passing["passing_seed_count"],
        "weak_CU_protocol_blocked": not failing["stage_one_passed"],
        "order_mismatch_rejected": order_mismatch_rejected,
        "formal_training": "未执行",
        "test_evaluation": "未执行",
        "OOF": "未执行",
    }
    path = project_root / "src-v3/smoke/RADAR-LN3-V11-rd-physical-paired-report-test-result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(path)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the V11 paired post-training gate without formal artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
