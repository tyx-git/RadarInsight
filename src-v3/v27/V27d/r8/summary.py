"""Three-seed paired summary and conservative promotion decision for r8."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .protocol import R8_ARMS, R8_SEEDS, artifact_path, load_context, read_sealed_json, write_sealed_json


def _stats(values: list[float]) -> dict[str, Any]:
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0, "values": values}


def summarize(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    reports = {
        seed: read_sealed_json(artifact_path(root, config["artifacts"]["evaluation_template"].format(arm="all", seed=seed), must_exist=True))
        for seed in R8_SEEDS
    }
    rows: list[dict[str, Any]] = []
    for seed, report in reports.items():
        control = report["arms"]["zero_control"]
        row: dict[str, Any] = {"seed": int(seed), "control": {"development": control["development"]["actions"], "calibration": control["calibration"]["actions"]}, "arms": {}}
        for arm in R8_ARMS[1:]:
            treatment = report["arms"][arm]
            row["arms"][arm] = {
                "development": treatment["development"]["actions"],
                "calibration": treatment["calibration"]["actions"],
                "paired": report["paired"][arm],
            }
        rows.append(row)
    aggregate: dict[str, Any] = {}
    for arm in R8_ARMS[1:]:
        for pool in ("development", "calibration"):
            for metric in ("bird_action_precision", "bird_action_recall", "false_positive_rejection", "unknown_rejection"):
                key = f"{arm}_{pool}_{metric}"
                aggregate[key] = _stats([float(row["arms"][arm][pool][metric]) for row in rows])
            for metric in ("precision_delta_vs_control", "bird_recall_delta_vs_control", "false_positive_rejection_delta_vs_control"):
                key = f"{arm}_{pool}_{metric}"
                aggregate[key] = _stats([float(row["arms"][arm]["paired"][f"{pool}_{metric}"]) for row in rows])
    criteria: dict[str, Any] = {}
    for arm in R8_ARMS[1:]:
        dev_rejection = [float(row["arms"][arm]["paired"]["development_false_positive_rejection_delta_vs_control"]) for row in rows]
        cal_rejection = [float(row["arms"][arm]["paired"]["calibration_false_positive_rejection_delta_vs_control"]) for row in rows]
        cal_recall = [float(row["arms"][arm]["calibration"]["bird_action_recall"]) for row in rows]
        criteria[arm] = {
            "development_rejection_wins": sum(value > 0.0 for value in dev_rejection),
            "calibration_rejection_wins": sum(value > 0.0 for value in cal_rejection),
            "calibration_recall_safe": all(value >= 0.98 for value in cal_recall),
            "development_and_calibration_same_direction": all(d > 0.0 and c > 0.0 for d, c in zip(dev_rejection, cal_rejection, strict=True)),
            "formal_l3_support_sufficient": False,
        }
    eligible = [arm for arm in R8_ARMS[1:] if criteria[arm]["calibration_recall_safe"] and criteria[arm]["development_and_calibration_same_direction"]]
    decision = "candidate_for_independent_calibration" if eligible else "diagnostic_only_no_promotion"
    payload = {
        "schema_version": "RADAR-LN3-V27d-r8-three-seed-summary-v1",
        "experiment_id": "V27d-r8-joint-course-stable8",
        "manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(R8_SEEDS),
        "arms": list(R8_ARMS),
        "rows": rows,
        "aggregate": aggregate,
        "criteria": criteria,
        "candidate_arms": eligible,
        "formal_unknown_support": 12,
        "formal_unknown_support_required": int(config["calibration"]["formal_unknown_support_required"]),
        "decision": decision,
        "formal_l3_enabled": False,
        "deployment_eligible": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["summary_path"])
    write_sealed_json(path, payload)
    return {"status": "completed", "summary_path": str(path.relative_to(root)).replace("\\", "/"), "summary": payload}


__all__ = ["summarize"]
