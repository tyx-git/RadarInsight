"""Action-level evaluation for V27d-r6."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import relative_project_path, sha256_file
from v27.V27d.metrics import binary_auroc
from v27.V27d.r5.training import _head_inputs as r5_head_inputs, load_checkpoint as load_r5_checkpoint

from .features import load_population_cache
from .protocol import R6_ARMS, R6_ROLES, artifact_path, load_context, read_sealed_json, source_path, write_sealed_json
from .training import _inputs, load_checkpoint


def _candidate_masks(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    candidate = tensors["candidate_population"].bool()
    positive = tensors["bird_positive"].bool() & candidate
    negative = tensors["route_false_positive_negative"].bool() & candidate
    return candidate, positive, negative


def _action_metrics(payload: Mapping[str, Any], scores: torch.Tensor, threshold: float) -> dict[str, Any]:
    tensors = payload["tensors"]
    labels = [str(item) for item in payload["label_names"]]
    candidate, positive, negative = _candidate_masks(payload)
    accepted = candidate & scores.ge(float(threshold))
    accepted_bird = accepted & positive
    accepted_false_positive = accepted & negative
    support_bird = int(positive.sum())
    support_candidate = int(candidate.sum())
    by_label = Counter(label for label, flag in zip(labels, accepted.tolist(), strict=True) if flag)
    rejection_by_label: dict[str, dict[str, int]] = {}
    for label in sorted(set(labels)):
        mask = candidate & torch.tensor([item == label for item in labels], dtype=torch.bool)
        rejection_by_label[label] = {
            "support": int(mask.sum()),
            "accepted": int((accepted & mask).sum()),
            "rejected": int((mask & ~accepted).sum()),
        }
    return {
        "threshold": float(threshold),
        "candidate_support": support_candidate,
        "bird_support": support_bird,
        "false_positive_support": int(negative.sum()),
        "accepted_bird": int(accepted_bird.sum()),
        "rejected_bird": int((positive & ~accepted).sum()),
        "accepted_false_positive": int(accepted_false_positive.sum()),
        "rejected_false_positive": int((negative & ~accepted).sum()),
        "bird_action_precision": float(accepted_bird.sum().item() / max(1, accepted.sum().item())),
        "bird_action_recall": float(accepted_bird.sum().item() / max(1, support_bird)),
        "false_positive_rejection": float((negative & ~accepted).float().sum().item() / max(1, int(negative.sum()))),
        "unknown_rejection": float(((negative & torch.tensor([label == "UnknownTarget" for label in labels], dtype=torch.bool)) & ~accepted).sum().item() / max(1, sum(label == "UnknownTarget" and bool(flag) for label, flag in zip(labels, candidate.tolist(), strict=True)))),
        "accepted_label_counts": dict(by_label),
        "by_label": rejection_by_label,
        "pending_low_quality_excluded": int((~tensors["quality_target_final"].bool()).sum()),
    }


def _score_output(model: torch.nn.Module, payload: Mapping[str, Any]) -> torch.Tensor:
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        output = model(**_inputs(payload, indices, torch.device("cpu")))
    return output["verifier_score_final"].detach().cpu()


def _ranking(payload: Mapping[str, Any], scores: torch.Tensor) -> dict[str, Any]:
    _, positive, negative = _candidate_masks(payload)
    mask = positive | negative
    return {
        "support": int(mask.sum()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "auroc_higher_is_known": binary_auroc(scores[mask], positive[mask]),
        "score_mean_positive": float(scores[positive].mean()) if bool(positive.any()) else None,
        "score_mean_negative": float(scores[negative].mean()) if bool(negative.any()) else None,
    }


def _r5_route_matched_baseline(root: Path, config: Mapping[str, Any], payload: Mapping[str, Any], role: str) -> dict[str, Any]:
    r5_config_path = source_path(root, config["source"]["r5_control_config_path"], must_exist=True)
    model, _, _ = load_r5_checkpoint(root, arm="control", seed=42, config_path=r5_config_path)
    model.eval()
    report = read_sealed_json(source_path(root, config["source"]["r5_control_report_path"], must_exist=True))
    threshold = float(report["arms"]["control"]["calibration_transfer"]["bird_knownness_probability"]["threshold_source"]["threshold"])
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        output = model(**r5_head_inputs(payload, indices, torch.device("cpu")))
    scores = torch.sigmoid(output["knownness_logit_final"]).cpu()
    return _action_metrics(payload, scores, threshold) | {"threshold_source": "r5_control_calibration_transfer", "role": role}


def evaluate_arm(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    model, checkpoint, _ = load_checkpoint(root, arm=arm, seed=seed, config_path=config_path)
    model.eval()
    development = load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role="representation_development"), must_exist=True))
    calibration = load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role="calibration"), must_exist=True))
    dev_scores = _score_output(model, development)
    cal_scores = _score_output(model, calibration)
    threshold = float(checkpoint["selection"]["threshold"])
    return {
        "arm": arm,
        "seed": seed,
        "checkpoint": {"path": relative_project_path(root, artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)), "sha256": sha256_file(artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)), "selected_epoch": checkpoint["selected_epoch"]},
        "threshold_source": "representation_development",
        "threshold": threshold,
        "development": {"ranking": _ranking(development, dev_scores), "actions": _action_metrics(development, dev_scores, threshold)},
        "calibration": {"ranking": _ranking(calibration, cal_scores), "actions": _action_metrics(calibration, cal_scores, threshold)},
        "r5_baseline": {
            "development": _r5_route_matched_baseline(root, config, development, "representation_development"),
            "calibration": _r5_route_matched_baseline(root, config, calibration, "calibration"),
        },
        "endpoint_contract": {"classification_logits_changed": False, "classification_logits_input_to_head": False, "softmax_input_to_head": False, "label_identity_input_to_head": False, "formal_l3_enabled": False},
        "manifest_sha256": manifest["manifest_sha256"],
        "deployment_eligible": False,
    }


def evaluate_all(project_root: str | Path, *, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    arms = {arm: evaluate_arm(root, arm=arm, seed=seed, config_path=config_path) for arm in R6_ARMS}
    report = {
        "schema_version": "RADAR-LN3-V27d-r6-evaluation-v1",
        "experiment_id": "V27d-r6-bird-liquid-verifier",
        "seed": seed,
        "direct_baseline": "V27d-r5-control",
        "arms": arms,
        "paired": {
            "liquid_minus_gru": {
                "development_precision_delta": arms["liquid_treatment"]["development"]["actions"]["bird_action_precision"] - arms["gru_control"]["development"]["actions"]["bird_action_precision"],
                "calibration_precision_delta": arms["liquid_treatment"]["calibration"]["actions"]["bird_action_precision"] - arms["gru_control"]["calibration"]["actions"]["bird_action_precision"],
                "development_bird_recall_delta": arms["liquid_treatment"]["development"]["actions"]["bird_action_recall"] - arms["gru_control"]["development"]["actions"]["bird_action_recall"],
                "calibration_bird_recall_delta": arms["liquid_treatment"]["calibration"]["actions"]["bird_action_recall"] - arms["gru_control"]["calibration"]["actions"]["bird_action_recall"],
                "development_false_positive_rejection_delta": arms["liquid_treatment"]["development"]["actions"]["false_positive_rejection"] - arms["gru_control"]["development"]["actions"]["false_positive_rejection"],
                "calibration_false_positive_rejection_delta": arms["liquid_treatment"]["calibration"]["actions"]["false_positive_rejection"] - arms["gru_control"]["calibration"]["actions"]["false_positive_rejection"],
            }
        },
        "population": {role: {"positive": int(load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True))["tensors"]["bird_positive"].sum()), "negative": int(load_population_cache(artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True))["tensors"]["route_false_positive_negative"].sum())} for role in R6_ROLES},
        "formal_l3_enabled": False,
        "formal_unknown_support": 12,
        "formal_unknown_support_required": int(config["calibration"]["formal_unknown_support_required"]),
        "deployment_eligible": False,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["evaluation_template"].format(arm="paired", seed=seed))
    write_sealed_json(path, report)
    return {"status": "completed", "report_path": relative_project_path(root, path), "report": report}


__all__ = ["evaluate_all", "evaluate_arm"]
