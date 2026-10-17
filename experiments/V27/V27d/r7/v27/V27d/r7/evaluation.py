"""Paired r7 evaluation over development and calibration pools."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import relative_project_path, sha256_file
from v27.V27d.metrics import binary_auroc
from v27.V27d.r6.training import _inputs as r6_inputs, load_checkpoint as load_r6_checkpoint

from .features import load_population_cache
from .protocol import R7_ARMS, R7_ROLES, artifact_path, load_context, source_path, write_sealed_json
from .training import _inputs, load_checkpoint


def _masks(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    candidate = tensors["candidate_population"].bool()
    positive = tensors["bird_positive"].bool() & candidate
    negative = tensors["route_false_positive_negative"].bool() & candidate
    return candidate, positive, negative


def _device(config: Mapping[str, Any]) -> torch.device:
    requested = str(config["training"].get("device", "cuda")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r7 evaluation requests CUDA but CUDA is unavailable")
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("V27d-r7 evaluation device must be cpu or cuda")


def _actions(payload: Mapping[str, Any], scores: torch.Tensor, threshold: float) -> dict[str, Any]:
    tensors = payload["tensors"]
    labels = [str(item) for item in payload["label_names"]]
    candidate, positive, negative = _masks(payload)
    accepted = candidate & scores.ge(float(threshold))
    by_label: dict[str, dict[str, int]] = {}
    for label in sorted(set(labels)):
        mask = candidate & torch.tensor([item == label for item in labels], dtype=torch.bool)
        by_label[label] = {"support": int(mask.sum()), "accepted": int((accepted & mask).sum()), "rejected": int((mask & ~accepted).sum())}
    unknown = candidate & torch.tensor([label == "UnknownTarget" for label in labels], dtype=torch.bool)
    return {
        "threshold": float(threshold),
        "candidate_support": int(candidate.sum()),
        "bird_support": int(positive.sum()),
        "false_positive_support": int(negative.sum()),
        "accepted_bird": int((accepted & positive).sum()),
        "rejected_bird": int((positive & ~accepted).sum()),
        "accepted_false_positive": int((accepted & negative).sum()),
        "rejected_false_positive": int((negative & ~accepted).sum()),
        "bird_action_precision": float((accepted & positive).sum().item() / max(1, accepted.sum().item())),
        "bird_action_recall": float((accepted & positive).sum().item() / max(1, positive.sum().item())),
        "false_positive_rejection": float((negative & ~accepted).sum().item() / max(1, negative.sum().item())),
        "unknown_rejection": float((unknown & ~accepted).sum().item() / max(1, unknown.sum().item())),
        "accepted_label_counts": dict(Counter(label for label, flag in zip(labels, accepted.tolist(), strict=True) if flag)),
        "by_label": by_label,
        "pending_low_quality_excluded": int((~tensors["quality_target_final"].bool()).sum()),
    }


def _ranking(payload: Mapping[str, Any], scores: torch.Tensor) -> dict[str, Any]:
    _, positive, negative = _masks(payload)
    mask = positive | negative
    return {
        "support": int(mask.sum()),
        "positive": int(positive.sum()),
        "negative": int(negative.sum()),
        "auroc_higher_is_known": binary_auroc(scores[mask], positive[mask]),
        "score_mean_positive": float(scores[positive].mean()) if bool(positive.any()) else None,
        "score_mean_negative": float(scores[negative].mean()) if bool(negative.any()) else None,
    }


def _r6_baseline(root: Path, config: Mapping[str, Any], payload: Mapping[str, Any], seed: int, role: str) -> dict[str, Any]:
    r6_config = source_path(root, config["source"]["r6_config_path"], must_exist=True)
    model, checkpoint, _ = load_r6_checkpoint(root, arm="liquid_treatment", seed=seed, config_path=r6_config)
    device = _device(config)
    model = model.to(device).eval()
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        scores = model(**r6_inputs(payload, indices, device))["verifier_score_final"].cpu()
    return _actions(payload, scores, float(checkpoint["selection"]["threshold"])) | {
        "threshold_source": "r6_liquid_development_selection",
        "role": role,
        "checkpoint_path": relative_project_path(root, source_path(root, config["source"]["r6_liquid_checkpoint_template"].format(seed=seed), must_exist=True)),
    }


def evaluate_arm(project_root: str | Path, *, arm: str, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    model, checkpoint, _ = load_checkpoint(root, arm=arm, seed=seed, config_path=config_path)
    device = _device(config)
    model = model.to(device).eval()
    development = load_population_cache(_cache_path(root, config, "representation_development"))
    calibration = load_population_cache(_cache_path(root, config, "calibration"))
    use_branch = arm == "stable8_treatment"
    with torch.inference_mode():
        dev_scores = model(**_inputs(development, torch.arange(len(development["parent_ids"])), device, use_branch=use_branch))["verifier_score_final"].cpu()
        cal_scores = model(**_inputs(calibration, torch.arange(len(calibration["parent_ids"])), device, use_branch=use_branch))["verifier_score_final"].cpu()
    threshold = float(checkpoint["selection"]["threshold"])
    return {
        "arm": arm,
        "seed": int(seed),
        "use_stable8_branch": use_branch,
        "evaluation_device": str(device),
        "checkpoint": {"path": relative_project_path(root, artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)), "sha256": sha256_file(artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed), must_exist=True)), "selected_epoch": checkpoint["selected_epoch"]},
        "threshold_source": "representation_development",
        "threshold": threshold,
        "development": {"ranking": _ranking(development, dev_scores), "actions": _actions(development, dev_scores, threshold)},
        "calibration": {"ranking": _ranking(calibration, cal_scores), "actions": _actions(calibration, cal_scores, threshold)},
        "r6_liquid_baseline": {"development": _r6_baseline(root, config, development, seed, "representation_development"), "calibration": _r6_baseline(root, config, calibration, seed, "calibration")},
        "endpoint_contract": {"classification_logits_changed": False, "classification_logits_input_to_head": False, "softmax_input_to_head": False, "label_identity_input_to_head": False, "formal_l3_enabled": False},
        "manifest_sha256": manifest["manifest_sha256"],
        "deployment_eligible": False,
    }


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(role=role), must_exist=True)


def evaluate_all(project_root: str | Path, *, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    arms = {arm: evaluate_arm(root, arm=arm, seed=seed, config_path=config_path) for arm in R7_ARMS}
    treatment = arms["stable8_treatment"]
    control = arms["zero_control"]
    report = {
        "schema_version": "RADAR-LN3-V27d-r7-evaluation-v1",
        "experiment_id": "V27d-r7-causal-stable8-dispersion",
        "seed": int(seed),
        "direct_baseline": "V27d-r6-liquid_treatment",
        "arms": arms,
        "paired": {
            "treatment_minus_zero_control": {
                "development_precision_delta": treatment["development"]["actions"]["bird_action_precision"] - control["development"]["actions"]["bird_action_precision"],
                "calibration_precision_delta": treatment["calibration"]["actions"]["bird_action_precision"] - control["calibration"]["actions"]["bird_action_precision"],
                "development_bird_recall_delta": treatment["development"]["actions"]["bird_action_recall"] - control["development"]["actions"]["bird_action_recall"],
                "calibration_bird_recall_delta": treatment["calibration"]["actions"]["bird_action_recall"] - control["calibration"]["actions"]["bird_action_recall"],
                "development_false_positive_rejection_delta": treatment["development"]["actions"]["false_positive_rejection"] - control["development"]["actions"]["false_positive_rejection"],
                "calibration_false_positive_rejection_delta": treatment["calibration"]["actions"]["false_positive_rejection"] - control["calibration"]["actions"]["false_positive_rejection"],
            },
            "treatment_minus_r6_liquid": {
                "development_precision_delta": treatment["development"]["actions"]["bird_action_precision"] - treatment["r6_liquid_baseline"]["development"]["bird_action_precision"],
                "calibration_precision_delta": treatment["calibration"]["actions"]["bird_action_precision"] - treatment["r6_liquid_baseline"]["calibration"]["bird_action_precision"],
                "development_false_positive_rejection_delta": treatment["development"]["actions"]["false_positive_rejection"] - treatment["r6_liquid_baseline"]["development"]["false_positive_rejection"],
                "calibration_false_positive_rejection_delta": treatment["calibration"]["actions"]["false_positive_rejection"] - treatment["r6_liquid_baseline"]["calibration"]["false_positive_rejection"],
            },
        },
        "population": {role: {"positive": int(load_population_cache(_cache_path(root, config, role))["tensors"]["bird_positive"].sum()), "negative": int(load_population_cache(_cache_path(root, config, role))["tensors"]["route_false_positive_negative"].sum())} for role in R7_ROLES},
        "formal_l3_enabled": False,
        "formal_unknown_support": 12,
        "formal_unknown_support_required": int(config["calibration"]["formal_unknown_support_required"]),
        "deployment_eligible": False,
        "test_payload_access_count": 0,
        "evaluation_device": str(_device(config)),
    }
    path = artifact_path(root, config["artifacts"]["evaluation_template"].format(arm="paired", seed=seed))
    write_sealed_json(path, report)
    return {"status": "completed", "report_path": relative_project_path(root, path), "report": report}


__all__ = ["evaluate_all", "evaluate_arm"]