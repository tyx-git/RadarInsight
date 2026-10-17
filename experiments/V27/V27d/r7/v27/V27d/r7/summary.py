"""Three-seed r7 summary and promotion decision."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from .protocol import R7_SEEDS, artifact_path, load_context, read_sealed_json, write_sealed_json


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0, "values": values}


def summarize(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    reports = {
        seed: read_sealed_json(artifact_path(root, config["artifacts"]["evaluation_template"].format(arm="paired", seed=seed), must_exist=True))
        for seed in R7_SEEDS
    }
    rows: list[dict[str, Any]] = []
    for seed, report in reports.items():
        control = report["arms"]["zero_control"]
        treatment = report["arms"]["stable8_treatment"]
        rows.append({
            "seed": seed,
            "zero_control": {"development": control["development"]["actions"], "calibration": control["calibration"]["actions"], "development_auroc": control["development"]["ranking"]["auroc_higher_is_known"], "calibration_auroc": control["calibration"]["ranking"]["auroc_higher_is_known"]},
            "stable8_treatment": {"development": treatment["development"]["actions"], "calibration": treatment["calibration"]["actions"], "development_auroc": treatment["development"]["ranking"]["auroc_higher_is_known"], "calibration_auroc": treatment["calibration"]["ranking"]["auroc_higher_is_known"]},
            "treatment_minus_zero_control": report["paired"]["treatment_minus_zero_control"],
            "treatment_minus_r6_liquid": report["paired"]["treatment_minus_r6_liquid"],
        })
    aggregate: dict[str, Any] = {}
    for name, getter in {
        "treatment_development_precision": lambda row: row["stable8_treatment"]["development"]["bird_action_precision"],
        "treatment_calibration_precision": lambda row: row["stable8_treatment"]["calibration"]["bird_action_precision"],
        "treatment_development_rejection": lambda row: row["stable8_treatment"]["development"]["false_positive_rejection"],
        "treatment_calibration_rejection": lambda row: row["stable8_treatment"]["calibration"]["false_positive_rejection"],
        "treatment_minus_control_development_rejection": lambda row: row["treatment_minus_zero_control"]["development_false_positive_rejection_delta"],
        "treatment_minus_control_calibration_rejection": lambda row: row["treatment_minus_zero_control"]["calibration_false_positive_rejection_delta"],
        "treatment_minus_r6_development_rejection": lambda row: row["treatment_minus_r6_liquid"]["development_false_positive_rejection_delta"],
        "treatment_minus_r6_calibration_rejection": lambda row: row["treatment_minus_r6_liquid"]["calibration_false_positive_rejection_delta"],
        "treatment_development_auroc": lambda row: row["stable8_treatment"]["development_auroc"],
        "treatment_calibration_auroc": lambda row: row["stable8_treatment"]["calibration_auroc"],
    }.items():
        aggregate[name] = _mean_std([float(getter(row)) for row in rows])
    max_additional_rejections = int(config["selection"]["max_additional_bird_rejections_per_pool"])
    recall_budget_rows = []
    for row in rows:
        report = reports[int(row["seed"])]
        treatment = report["arms"]["stable8_treatment"]
        r6 = treatment["r6_liquid_baseline"]
        dev_delta = int(treatment["development"]["actions"]["rejected_bird"]) - int(r6["development"]["rejected_bird"])
        cal_delta = int(treatment["calibration"]["actions"]["rejected_bird"]) - int(r6["calibration"]["rejected_bird"])
        recall_budget_rows.append({"seed": int(row["seed"]), "development_additional_bird_rejections": dev_delta, "calibration_additional_bird_rejections": cal_delta, "safe": dev_delta <= max_additional_rejections and cal_delta <= max_additional_rejections})
    recall_safe = all(bool(item["safe"]) for item in recall_budget_rows)
    calibration_wins = sum(
        row["treatment_minus_zero_control"]["calibration_false_positive_rejection_delta"] > 0.0
        for row in rows
    )
    development_wins = sum(
        row["treatment_minus_zero_control"]["development_false_positive_rejection_delta"] > 0.0
        for row in rows
    )
    treatment_has_stable_gain = bool(recall_safe and calibration_wins >= int(config["selection"]["minimum_calibration_seed_wins"]))
    decision = "stable8_candidate" if treatment_has_stable_gain else "diagnostic_only_no_promotion"
    payload = {
        "schema_version": "RADAR-LN3-V27d-r7-three-seed-summary-v1",
        "experiment_id": "V27d-r7-causal-stable8-dispersion",
        "manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(R7_SEEDS),
        "rows": rows,
        "aggregate": aggregate,
        "criteria": {
            "recall_safe": recall_safe,
            "recall_budget": recall_budget_rows,
            "max_additional_bird_rejections_per_pool": max_additional_rejections,
            "development_seed_wins": development_wins,
            "calibration_seed_wins": calibration_wins,
            "minimum_calibration_seed_wins": int(config["selection"]["minimum_calibration_seed_wins"]),
            "formal_l3_support_sufficient": False,
        },
        "decision": decision,
        "formal_l3_enabled": False,
        "deployment_eligible": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["summary_path"])
    write_sealed_json(path, payload)
    return {"status": "completed", "summary_path": str(path.relative_to(root)).replace("\\", "/"), "summary": payload}


__all__ = ["summarize"]
