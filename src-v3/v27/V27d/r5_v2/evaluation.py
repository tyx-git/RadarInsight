"""Evaluation and exploratory threshold transfer for V27d-r5-v2."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import sha256_file

from v27.V27d.r5.scoring import ranking_metrics

from .audit import audit_error_samples
from .features import load_population_cache
from .protocol import artifact_path, load_context, read_sealed_json, write_sealed_json
from .training import _cache_path, _output, _selection, load_checkpoint


def _transfer(scores: torch.Tensor, payload: Mapping[str, Any], threshold_record: Mapping[str, Any]) -> dict[str, Any]:
    tensors = payload["tensors"]
    positive = tensors["bird_route_positive"].bool()
    negative = tensors["bird_route_negative"].bool()
    threshold = threshold_record.get("threshold")
    if threshold is None:
        return {
            "feasible": False,
            "threshold": None,
            "bird_recall": 0.0,
            "unknown_rejection": 0.0,
            "correct_bird_new_rejections": int(positive.sum()),
            "positive_support": int(positive.sum()),
            "negative_support": int(negative.sum()),
        }
    accepted = scores.ge(float(threshold)) if bool(threshold_record.get("higher_is_known", True)) else scores.le(float(threshold))
    return {
        "feasible": True,
        "threshold": float(threshold),
        "higher_is_known": bool(threshold_record.get("higher_is_known", True)),
        "bird_recall": float(accepted[positive].float().mean()) if bool(positive.any()) else 0.0,
        "unknown_rejection": float((~accepted[negative]).float().mean()) if bool(negative.any()) else 0.0,
        "correct_bird_new_rejections": int((positive & ~accepted).sum()),
        "accepted_bird": int((positive & accepted).sum()),
        "rejected_bird": int((positive & ~accepted).sum()),
        "accepted_unknown": int((negative & accepted).sum()),
        "rejected_unknown": int((negative & ~accepted).sum()),
        "positive_support": int(positive.sum()),
        "negative_support": int(negative.sum()),
    }


def evaluate_experiment(project_root: str | Path, *, seed: int = 42, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    model, checkpoint, _ = load_checkpoint(root, seed=seed, config_path=config_path)
    model.eval()
    development = load_population_cache(_cache_path(root, config, "representation_development"))
    calibration = load_population_cache(_cache_path(root, config, "calibration"))
    development_output = _output(model, development, torch.device("cpu"))
    calibration_output = _output(model, calibration, torch.device("cpu"))
    development_scores = torch.sigmoid(development_output["knownness_logit_final"].cpu())
    calibration_scores = torch.sigmoid(calibration_output["knownness_logit_final"].cpu())
    development_selection = _selection(model, development, torch.device("cpu"), config)
    calibration_transfer = _transfer(calibration_scores, calibration, development_selection)
    dev_positive = development["tensors"]["bird_route_positive"].bool()
    dev_negative = development["tensors"]["bird_route_negative"].bool()
    cal_positive = calibration["tensors"]["bird_route_positive"].bool()
    cal_negative = calibration["tensors"]["bird_route_negative"].bool()
    r5_report = read_sealed_json(root / config["source"]["r5_report_path"])
    r5_control = r5_report["arms"]["control"]
    r5_dev = r5_control["development"]["bird_knownness_probability"]
    r5_cal = r5_control["calibration_transfer"]["bird_knownness_probability"]["transfer_diagnostics"]
    cal_unknown_delta = calibration_transfer["unknown_rejection"] - float(r5_cal["unknown_rejection"])
    dev_unknown_delta = development_selection["unknown_rejection"] - float(r5_dev["selection"]["unknown_rejection"])
    formal_support = int(cal_negative.sum())
    audit_path = artifact_path(root, config["artifacts"]["audit_path"], must_exist=True)
    report = {
        "schema_version": "RADAR-LN3-V27d-r5-bird-route-v2-evaluation-v1",
        "experiment_id": "V27d-r5-bird-route-v2",
        "direct_baseline": "V27d-r5-control",
        "seed": int(seed),
        "v2_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint": {
            "path": str(Path(config["artifacts"]["checkpoint_path"]).as_posix()),
            "sha256": sha256_file(artifact_path(root, config["artifacts"]["checkpoint_path"], must_exist=True)),
            "selected_epoch": checkpoint["selected_epoch"],
            "selection_device": "cpu_canonical_replay",
        },
        "population": {
            "representation_development": {
                "positive": int(dev_positive.sum()),
                "negative": int(dev_negative.sum()),
                "quality_eligible": int(development["tensors"]["quality_target_final"].sum()),
            },
            "calibration": {
                "positive": int(cal_positive.sum()),
                "negative": int(cal_negative.sum()),
                "quality_eligible": int(calibration["tensors"]["quality_target_final"].sum()),
            },
        },
        "development": {
            "ranking": ranking_metrics(development_scores, dev_positive, dev_negative),
            "selection": development_selection,
        },
        "calibration_transfer": {
            "ranking": ranking_metrics(calibration_scores, cal_positive, cal_negative),
            "threshold_source": {
                "pool": "representation_development",
                "threshold": development_selection.get("threshold"),
                "higher_is_known": development_selection.get("higher_is_known"),
            },
            "diagnostics": calibration_transfer,
        },
        "paired_against_r5_control": {
            "development_unknown_rejection_delta": dev_unknown_delta,
            "calibration_unknown_rejection_delta": cal_unknown_delta,
            "development_bird_recall_delta": development_selection["bird_recall"] - float(r5_dev["selection"]["bird_recall"]),
            "calibration_bird_recall_delta": calibration_transfer["bird_recall"] - float(r5_cal["bird_recall"]),
        },
        "objective": checkpoint.get("objective"),
        "success_criteria": {
            "calibration_bird_recall_ge_98_percent": calibration_transfer["bird_recall"] >= 0.98,
            "calibration_unknown_rejection_delta_ge_15pp": cal_unknown_delta >= 0.15,
            "development_and_calibration_unknown_direction_same": dev_unknown_delta > 0.0 and cal_unknown_delta > 0.0,
            "development_correct_bird_rejections_within_budget": development_selection["correct_bird_new_rejections"] <= int(config["selection"]["max_correct_bird_new_rejections"]),
            "calibration_no_additional_correct_bird_rejections": calibration_transfer["correct_bird_new_rejections"] <= int(config["selection"]["calibration_max_additional_correct_bird_rejections"]),
        },
        "endpoint_contract": {
            "classification_logits_changed": False,
            "classification_logits_input_to_head": False,
            "softmax_input_to_head": False,
            "label_identity_input_to_head": False,
            "formal_l3_enabled": False,
            "formal_conformal_support": formal_support >= int(config["calibration"]["formal_unknown_support_required"]),
            "calibration_scope": "exploratory_threshold_transfer_only",
        },
        "audit": {
            "path": str(audit_path.relative_to(root)).replace("\\", "/"),
            "calibration_unknown_support": formal_support,
            "formal_unknown_support_required": int(config["calibration"]["formal_unknown_support_required"]),
        },
        "status": "diagnostic_only",
        "deployment_eligible": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["evaluation_path"])
    write_sealed_json(path, report)
    return {"status": "completed", "evaluation_path": str(path.relative_to(root)).replace("\\", "/"), "report": report}


__all__ = ["evaluate_experiment"]
