"""Three-seed summary and promotion decision for V27d-r6."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

import torch

from v27.V27d.r5.training import _head_inputs as r5_head_inputs, load_checkpoint as load_r5_checkpoint

from .features import load_population_cache
from .protocol import R6_SEEDS, artifact_path, load_context, read_sealed_json, write_sealed_json
from .protocol import source_path


def _mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0, "values": values}


def _r5_baseline_action_counts(root: Path, config: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Replay the frozen r5 verifier on the r6 route-matched populations."""

    r5_config_path = source_path(root, config["source"]["r5_control_config_path"], must_exist=True)
    r5_model, _, _ = load_r5_checkpoint(root, arm="control", seed=42, config_path=r5_config_path)
    r5_model.eval()
    report = read_sealed_json(source_path(root, config["source"]["r5_control_report_path"], must_exist=True))
    threshold = float(report["arms"]["control"]["calibration_transfer"]["bird_knownness_probability"]["threshold_source"]["threshold"])
    result: dict[str, dict[str, int]] = {}
    for role in ("representation_development", "calibration"):
        cache = load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True))
        indices = torch.arange(len(cache["parent_ids"]))
        with torch.inference_mode():
            output = r5_model(**r5_head_inputs(cache, indices, torch.device("cpu")))
        scores = torch.sigmoid(output["knownness_logit_final"]).cpu()
        candidate = cache["tensors"]["candidate_population"].bool()
        positive = cache["tensors"]["bird_positive"].bool() & candidate
        negative = cache["tensors"]["route_false_positive_negative"].bool() & candidate
        accepted = candidate & scores.ge(threshold)
        result[role] = {
            "threshold_source": "r5_control_calibration_transfer",
            "threshold": threshold,
            "candidate_support": int(candidate.sum()),
            "bird_support": int(positive.sum()),
            "false_positive_support": int(negative.sum()),
            "accepted_bird": int((accepted & positive).sum()),
            "rejected_bird": int((positive & ~accepted).sum()),
            "accepted_false_positive": int((accepted & negative).sum()),
            "rejected_false_positive": int((negative & ~accepted).sum()),
        }
    return result


def summarize(project_root: str | Path, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    reports = {
        seed: read_sealed_json(artifact_path(root, config["artifacts"]["evaluation_template"].format(arm="paired", seed=seed), must_exist=True))
        for seed in R6_SEEDS
    }
    r5_baseline = _r5_baseline_action_counts(root, config)
    rows: list[dict[str, Any]] = []
    for seed, report in reports.items():
        gru = report["arms"]["gru_control"]
        liquid = report["arms"]["liquid_treatment"]
        rows.append({
            "seed": seed,
            "gru": {
                "development": gru["development"]["actions"],
                "calibration": gru["calibration"]["actions"],
                "development_auroc": gru["development"]["ranking"]["auroc_higher_is_known"],
                "calibration_auroc": gru["calibration"]["ranking"]["auroc_higher_is_known"],
            },
            "liquid": {
                "development": liquid["development"]["actions"],
                "calibration": liquid["calibration"]["actions"],
                "development_auroc": liquid["development"]["ranking"]["auroc_higher_is_known"],
                "calibration_auroc": liquid["calibration"]["ranking"]["auroc_higher_is_known"],
            },
            "liquid_minus_gru": report["paired"]["liquid_minus_gru"],
        })
    metric_paths = {
        "development_precision": lambda row: row["liquid"]["development"]["bird_action_precision"],
        "calibration_precision": lambda row: row["liquid"]["calibration"]["bird_action_precision"],
        "development_bird_recall": lambda row: row["liquid"]["development"]["bird_action_recall"],
        "calibration_bird_recall": lambda row: row["liquid"]["calibration"]["bird_action_recall"],
        "development_false_positive_rejection": lambda row: row["liquid"]["development"]["false_positive_rejection"],
        "calibration_false_positive_rejection": lambda row: row["liquid"]["calibration"]["false_positive_rejection"],
        "liquid_minus_gru_development_precision": lambda row: row["liquid_minus_gru"]["development_precision_delta"],
        "liquid_minus_gru_calibration_precision": lambda row: row["liquid_minus_gru"]["calibration_precision_delta"],
        "liquid_minus_gru_development_rejection": lambda row: row["liquid_minus_gru"]["development_false_positive_rejection_delta"],
        "liquid_minus_gru_calibration_rejection": lambda row: row["liquid_minus_gru"]["calibration_false_positive_rejection_delta"],
        "liquid_minus_gru_development_bird_recall": lambda row: row["liquid_minus_gru"]["development_bird_recall_delta"],
        "liquid_minus_gru_calibration_bird_recall": lambda row: row["liquid_minus_gru"]["calibration_bird_recall_delta"],
        "liquid_development_auroc": lambda row: row["liquid"]["development_auroc"],
        "liquid_calibration_auroc": lambda row: row["liquid"]["calibration_auroc"],
        "gru_development_auroc": lambda row: row["gru"]["development_auroc"],
        "gru_calibration_auroc": lambda row: row["gru"]["calibration_auroc"],
    }
    aggregate = {name: _mean_std([float(function(row)) for row in rows]) for name, function in metric_paths.items()}
    max_loss = int(config["selection"]["max_additional_bird_rejections_per_pool"])
    liquid_recall_safe = all(
        int(row["liquid"]["development"]["accepted_bird"]) >= int(r5_baseline["representation_development"]["accepted_bird"]) - max_loss
        and int(row["liquid"]["calibration"]["accepted_bird"]) >= int(r5_baseline["calibration"]["accepted_bird"]) - max_loss
        for row in rows
    )
    liquid_precision_gain = all(
        float(row["liquid"]["development"]["bird_action_precision"]) - float(row["gru"]["development"]["bird_action_precision"]) >= float(config["selection"]["min_precision_gain"])
        and float(row["liquid"]["calibration"]["bird_action_precision"]) - float(row["gru"]["calibration"]["bird_action_precision"]) >= float(config["selection"]["min_precision_gain"])
        for row in rows
    )
    liquid_direction = all(
        float(row["liquid_minus_gru"]["development_false_positive_rejection_delta"]) > 0.0
        and float(row["liquid_minus_gru"]["calibration_false_positive_rejection_delta"]) > 0.0
        for row in rows
    )
    absolute_milestone = all(
        float(row["liquid"]["development"]["false_positive_rejection"]) >= float(config["selection"]["min_development_unknown_rejection"])
        and float(row["liquid"]["calibration"]["false_positive_rejection"]) >= float(config["selection"]["min_calibration_unknown_rejection"])
        for row in rows
    )
    liquid_is_preferred = bool(liquid_recall_safe and liquid_precision_gain and liquid_direction and absolute_milestone)
    if liquid_is_preferred:
        decision = "liquid_treatment_candidate"
    elif all(float(row["liquid_minus_gru"]["calibration_false_positive_rejection_delta"]) == 0.0 for row in rows):
        decision = "no_liquid_calibration_gain"
    else:
        decision = "diagnostic_only_no_promotion"
    payload = {
        "schema_version": "RADAR-LN3-V27d-r6-three-seed-summary-v1",
        "experiment_id": "V27d-r6-bird-liquid-verifier",
        "manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(R6_SEEDS),
        "rows": rows,
        "aggregate": aggregate,
        "r5_baseline_route_matched": r5_baseline,
        "criteria": {
            "liquid_recall_safe": liquid_recall_safe,
            "liquid_precision_gain_each_seed_and_pool": liquid_precision_gain,
            "liquid_false_positive_rejection_same_direction_each_seed": liquid_direction,
            "absolute_milestone_met_each_seed": absolute_milestone,
            "formal_l3_support_sufficient": False,
        },
        "decision": decision,
        "formal_l3_enabled": False,
        "deployment_eligible": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["summary_report_path"])
    write_sealed_json(path, payload)
    return {"status": "completed", "summary_path": str(path.relative_to(root)).replace("\\", "/"), "summary": payload}


__all__ = ["summarize"]