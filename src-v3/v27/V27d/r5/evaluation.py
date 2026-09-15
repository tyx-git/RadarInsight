"""Bird-route score evaluation and frozen threshold transfer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import relative_project_path, sha256_file, write_torch_if_absent_or_identical

from v27.V27d.features import load_feature_cache as load_r1_feature_cache
from v27.V27d.protocol import artifact_path as v27d_artifact_path

from .features import load_population_cache
from .protocol import R5_ARMS, artifact_path, load_context, write_sealed_json
from .scoring import fit_bird_prototypes, ranking_metrics, score_bird_projection, select_high_known_threshold, validate_scorer
from .training import load_checkpoint


def _cache_path(root: Path, config: Mapping[str, Any], *, arm: str, role: str) -> Path:
    return artifact_path(root, config["artifacts"]["cache_template"].format(arm=arm, role=role), must_exist=True)


def _inputs(payload: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    indices = torch.arange(len(payload["parent_ids"]))
    return {
        "summary_features": tensors["summary_features"][indices].to(device),
        "event_trajectory": tensors["event_trajectory"][indices].to(device),
        "task_facts_sequence": tensors["task_facts_sequence"][indices].to(device),
        "stable8_causal_features": tensors["stable8_causal_features"][indices].to(device),
        "padding_mask": tensors["padding_mask"][indices].to(device),
    }


def _output(model: torch.nn.Module, payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    with torch.inference_mode():
        output = model(**_inputs(payload, next(model.parameters()).device))
    return output["knownness_logit_final"].detach().cpu(), output["projection_final"].detach().cpu()


def _pool_masks(payload: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    positive = tensors["bird_route_positive"].bool()
    negative = tensors["bird_route_negative"].bool()
    population = tensors["bird_route_population"].bool()
    return positive & population, negative & population, population


def _threshold_transfer(scores: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, *, higher_is_known: bool, threshold_record: Mapping[str, Any]) -> dict[str, Any]:
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
    accepted = scores.ge(float(threshold)) if higher_is_known else scores.le(float(threshold))
    return {
        "feasible": True,
        "threshold": float(threshold),
        "higher_is_known": bool(higher_is_known),
        "bird_recall": float(accepted[positive].float().mean().item()) if bool(positive.any()) else 0.0,
        "unknown_rejection": float((~accepted[negative]).float().mean().item()) if bool(negative.any()) else 0.0,
        "correct_bird_new_rejections": int((positive & ~accepted).sum()),
        "positive_support": int(positive.sum()),
        "negative_support": int(negative.sum()),
    }


def _slice_diagnostics(payload: Mapping[str, Any], scores: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, transfer: Mapping[str, Any]) -> dict[str, Any]:
    tensors = payload["tensors"]
    labels = [str(value) for value in payload["label_names"]]
    accepted = torch.zeros_like(positive)
    if transfer.get("feasible"):
        threshold = float(transfer["threshold"])
        accepted = scores.ge(threshold) if bool(transfer["higher_is_known"]) else scores.le(threshold)
    return {
        "positive_support": int(positive.sum()),
        "negative_support": int(negative.sum()),
        "bird_route_population_support": int(tensors["bird_route_population"].sum()),
        "low_quality_count": int((~tensors["quality_target_final"].bool()).sum()),
        "clutter_count": sum(label == "ClutterTarget" for label in labels),
        "clutter_bird_route_count": sum(label == "ClutterTarget" and bool(tensors["l1_target_route"][row]) and bool(tensors["l2_predicted_bird"][row]) for row, label in enumerate(labels)),
        "accepted_positive": int((positive & accepted).sum()),
        "rejected_positive": int((positive & ~accepted).sum()),
        "accepted_negative": int((negative & accepted).sum()),
        "rejected_negative": int((negative & ~accepted).sum()),
        "bird_recall": float(accepted[positive].float().mean().item()) if bool(positive.any()) else 0.0,
        "unknown_rejection": float((~accepted[negative]).float().mean().item()) if bool(negative.any()) else 0.0,
    }


def _score_report(payload: Mapping[str, Any], logits: torch.Tensor, projection: torch.Tensor, scorer: Mapping[str, Any], *, threshold_source: Mapping[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    positive, negative, _ = _pool_masks(payload)
    probability = torch.sigmoid(logits)
    prototype = score_bird_projection(projection, dict(scorer))
    scores = {
        "bird_knownness_probability": probability,
        "prototype_cosine": prototype["prototype_cosine"],
        "prototype_distance": prototype["prototype_distance"],
        "mahalanobis_distance": prototype["mahalanobis_distance"],
    }
    orientation = {
        "bird_knownness_probability": True,
        "prototype_cosine": True,
        "prototype_distance": False,
        "mahalanobis_distance": False,
    }
    report: dict[str, Any] = {}
    for name, score in scores.items():
        ranking = ranking_metrics(score, positive, negative)
        selection = select_high_known_threshold(
            score,
            positive,
            negative,
            higher_is_known=orientation[name],
            recall_floor=0.98,
        ) if threshold_source is None else {
            "feasible": bool(threshold_source[name].get("feasible", False)),
            "threshold": threshold_source[name].get("threshold"),
            "higher_is_known": threshold_source[name].get("higher_is_known"),
            "source_pool": "representation_development",
        }
        report[name] = {"ranking": ranking, "selection": selection}
        if threshold_source is not None:
            source = threshold_source[name]
            report[name]["threshold_source"] = {
                "pool": "representation_development",
                "threshold": source.get("threshold"),
                "higher_is_known": source.get("higher_is_known"),
                "bird_recall": source.get("bird_recall"),
                "unknown_rejection": source.get("unknown_rejection"),
                "positive_support": source.get("positive_support"),
                "negative_support": source.get("negative_support"),
            }
            report[name]["transfer_diagnostics"] = _slice_diagnostics(
                payload,
                score,
                positive,
                negative,
                source,
            )
    return report, scores


def evaluate_experiment(project_root: str | Path, *, seed: int = 42, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, r1, manifest = load_context(root, config_path)
    arms: dict[str, Any] = {}
    scorer_payloads: dict[str, Any] = {}
    for arm in R5_ARMS:
        model, checkpoint, _ = load_checkpoint(root, arm=arm, seed=seed, config_path=config_path)
        fit = load_population_cache(_cache_path(root, config, arm=arm, role="representation_fit"))
        dev = load_population_cache(_cache_path(root, config, arm=arm, role="representation_development"))
        cal = load_population_cache(_cache_path(root, config, arm=arm, role="calibration"))
        if any(payload.get("r5_manifest_sha256") != manifest["manifest_sha256"] for payload in (fit, dev, cal)):
            raise ValueError("V27d-r5 population cache provenance differs from the active manifest")
        fit_logits, fit_projection = _output(model, fit)
        dev_logits, dev_projection = _output(model, dev)
        cal_logits, cal_projection = _output(model, cal)
        fit_positive, _, _ = _pool_masks(fit)
        scorer = fit_bird_prototypes(
            fit_projection[fit_positive],
            prototype_count=int(config["selection"]["prototype_count"]),
            ridge=float(config["selection"]["ridge"]),
        )
        scorer_path = artifact_path(root, config["artifacts"]["scorer_template"].format(arm=arm, seed=seed))
        write_torch_if_absent_or_identical(scorer_path, scorer)
        scorer_payloads[arm] = scorer
        dev_report, _ = _score_report(dev, dev_logits, dev_projection, scorer)
        thresholds = {name: record["selection"] for name, record in dev_report.items()}
        cal_report, cal_scores = _score_report(cal, cal_logits, cal_projection, scorer, threshold_source=thresholds)
        source_endpoint_equal = True
        for role, payload in (("representation_fit", fit), ("representation_development", dev), ("calibration", cal)):
            source_path = v27d_artifact_path(
                root,
                r1.config.artifacts.feature_cache_template,
                variant="r1",
                role=role,
                task="dbb_specialist",
            )
            source = load_r1_feature_cache(r1, source_path)
            source_endpoint_equal = source_endpoint_equal and torch.equal(
                source["tensors"]["endpoint_final_logits"],
                payload["tensors"]["endpoint_final_logits"],
            )
        arms[arm] = {
            "checkpoint": {
                "path": relative_project_path(root, artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed))),
                "sha256": sha256_file(artifact_path(root, config["artifacts"]["checkpoint_template"].format(arm=arm, seed=seed))),
                "selected_epoch": checkpoint["selected_epoch"],
            },
            "population": {
                role: {
                    "parent_count": len(payload["parent_ids"]),
                    "positive": int(payload["tensors"]["bird_route_positive"].sum()),
                    "negative": int(payload["tensors"]["bird_route_negative"].sum()),
                    "quality_eligible": int(payload["tensors"]["quality_target_final"].sum()),
                }
                for role, payload in (("representation_fit", fit), ("representation_development", dev), ("calibration", cal))
            },
            "development": dev_report,
            "calibration_transfer": cal_report,
            "scorer": {
                "path": relative_project_path(root, scorer_path),
                "sha256": sha256_file(scorer_path),
                "payload_sha256": scorer["scorer_sha256"],
            },
            "classification_logits_changed": False,
            "endpoint_logits_equal_to_frozen_r1_source": bool(source_endpoint_equal),
            "formal_l3_enabled": False,
            "deployment_eligible": False,
            "calibration_unknown_support": int(cal["tensors"]["bird_route_negative"].sum()),
            "formal_calibration_support_required": int(config["calibration"]["formal_unknown_support"]),
        }
    if not torch.equal(
        load_population_cache(_cache_path(root, config, arm="control", role="representation_development"))["tensors"]["endpoint_final_logits"],
        load_population_cache(_cache_path(root, config, arm="stable8_treatment", role="representation_development"))["tensors"]["endpoint_final_logits"],
    ):
        raise ValueError("V27d-r5 control/treatment endpoint logits changed")
    paired: dict[str, Any] = {}
    for role in ("representation_development", "calibration"):
        paired[role] = {}
        for score_name in arms["control"]["development"]:
            control = arms["control"]["development"][score_name] if role == "representation_development" else arms["control"]["calibration_transfer"][score_name]
            treatment = arms["stable8_treatment"]["development"][score_name] if role == "representation_development" else arms["stable8_treatment"]["calibration_transfer"][score_name]
            paired[role][score_name] = {
                "unknown_rejection_delta_treatment_minus_control": float(treatment.get("transfer_diagnostics", {}).get("unknown_rejection", 0.0) - control.get("transfer_diagnostics", {}).get("unknown_rejection", 0.0)) if role == "calibration" else float(treatment["selection"].get("unknown_rejection", 0.0) - control["selection"].get("unknown_rejection", 0.0)),
                "bird_recall_delta_treatment_minus_control": float(treatment.get("transfer_diagnostics", {}).get("bird_recall", 0.0) - control.get("transfer_diagnostics", {}).get("bird_recall", 0.0)) if role == "calibration" else float(treatment["selection"].get("bird_recall", 0.0) - control["selection"].get("bird_recall", 0.0)),
            }
    report = {
        "schema_version": "RADAR-LN3-V27d-r5-bird-route-evaluation-v1",
        "experiment_id": "V27d-r5-bird-route",
        "seed": int(seed),
        "r5_manifest_sha256": manifest["manifest_sha256"],
        "r1_manifest_sha256": r1.manifest_sha256,
        "arms": arms,
        "paired": paired,
        "calibration_scope": "exploratory_threshold_transfer_only",
        "formal_conformal_support": False,
        "deployment_eligible": False,
        "test_payload_access_count": 0,
    }
    report_path = artifact_path(root, config["artifacts"]["report_path"].format(seed=seed))
    write_sealed_json(report_path, report)
    return {"report_path": relative_project_path(root, report_path), "report": report}


__all__ = ["evaluate_experiment"]
