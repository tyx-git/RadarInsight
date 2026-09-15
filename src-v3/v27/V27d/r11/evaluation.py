"""Paired knownness evaluation for V27d-r11."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import relative_project_path, sha256_file

from .checkpoint import load_checkpoint
from .features import load_population_cache, validate_cache_binding
from .model import build_model
from .protocol import (
    R11_EXPERIMENT_ID,
    R11_ROLES,
    R11_SEEDS,
    R11_REPORT_SCHEMA_VERSION,
    artifact_path,
    load_context,
    read_sealed_json,
    write_sealed_json_if_identical,
)
from .training import (
    _action_metrics,
    _load_r6_reference,
    _ranking,
    _r6_inputs,
    _scores,
)


def _cache_hashes(caches: Mapping[str, Mapping[str, Any]]) -> dict[str, str]:
    return {role: str(caches[role]["cache_sha256"]) for role in R11_ROLES}


def _load_caches(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    caches = {
        role: load_population_cache(
            artifact_path(
                root,
                config["artifacts"]["cache_template"].format(role=role),
                must_exist=True,
            )
        )
        for role in R11_ROLES
    }
    for role, payload in caches.items():
        validate_cache_binding(payload, root=root, config=config, role=role)
        if payload.get("r11_manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError(f"V27d-r11 cache manifest differs: {role}")
    return caches


def _comparison(
    treatment: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    treatment_actions = treatment["actions"]
    baseline_actions = baseline["actions"]
    treatment_ranking = treatment["ranking"]
    baseline_ranking = baseline["ranking"]
    names = (
        "bird_action_precision",
        "bird_action_recall",
        "false_positive_rejection",
        "unknown_rejection",
    )
    action_delta = {
        name: float(treatment_actions[name]) - float(baseline_actions[name])
        for name in names
    }
    return {
        "threshold_delta": float(treatment["threshold"]) - float(baseline["threshold"]),
        "action_delta": action_delta,
        "auroc_delta": (
            float(treatment_ranking["auroc_higher_is_known"])
            - float(baseline_ranking["auroc_higher_is_known"])
        ),
        "accepted_bird_delta": int(treatment_actions["accepted_bird"]) - int(baseline_actions["accepted_bird"]),
        "accepted_false_positive_delta": int(treatment_actions["accepted_false_positive"]) - int(baseline_actions["accepted_false_positive"]),
    }


def _unknown_support(payload: Mapping[str, Any]) -> dict[str, int]:
    labels = [str(value) for value in payload["label_names"]]
    tensors = payload["tensors"]
    unknown = torch.tensor(
        [label == "UnknownTarget" for label in labels], dtype=torch.bool
    )
    candidate = tensors["candidate_population"].bool()
    quality = tensors["quality_target_final"].bool()
    return {
        "all_parents": int(unknown.sum()),
        "quality_eligible": int((unknown & quality).sum()),
        "route_candidate": int((unknown & candidate).sum()),
        "quality_eligible_route_candidate": int((unknown & quality & candidate).sum()),
    }


def evaluate_seed(
    project_root: str | Path,
    *,
    seed: int,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) not in R11_SEEDS:
        raise ValueError("V27d-r11 seed is not registered")
    caches = _load_caches(root, config, manifest)
    treatment_model = build_model(config, arm="shape_treatment", seed=seed)
    checkpoint_path = artifact_path(
        root,
        config["artifacts"]["checkpoint_template"].format(seed=seed),
        must_exist=True,
    )
    checkpoint = load_checkpoint(
        checkpoint_path,
        root=root,
        config=config,
        manifest=manifest,
        model=treatment_model,
        seed=seed,
        cache_hashes=_cache_hashes(caches),
    )
    treatment_model.eval()
    reference_model, reference_checkpoint, reference_path = _load_r6_reference(
        root,
        config,
        seed=seed,
    )
    reference_model.eval()
    threshold = float(checkpoint["metrics"]["threshold"])
    reference_threshold = float(reference_checkpoint["selection"]["threshold"])
    pools: dict[str, Any] = {}
    for role in ("representation_development", "calibration"):
        payload = caches[role]
        treatment_scores = _scores(
            treatment_model,
            payload,
            r6=False,
            device=torch.device("cpu"),
        )
        reference_scores = _scores(
            reference_model,
            payload,
            r6=True,
            device=torch.device("cpu"),
        )
        treatment = {
            "threshold": threshold,
            "ranking": _ranking(payload, treatment_scores),
            "actions": _action_metrics(payload, treatment_scores, threshold),
        }
        baseline = {
            "threshold": reference_threshold,
            "ranking": _ranking(payload, reference_scores),
            "actions": _action_metrics(payload, reference_scores, reference_threshold),
        }
        pools[role] = {
            "treatment": treatment,
            "direct_baseline": baseline,
            "comparison": _comparison(treatment, baseline),
            "unknown_support": _unknown_support(payload),
        }
    calibration_unknown_support = pools["calibration"]["unknown_support"]
    report = {
        "schema_version": R11_REPORT_SCHEMA_VERSION,
        "experiment_id": R11_EXPERIMENT_ID,
        "status": "completed",
        "seed": int(seed),
        "direct_baseline": "V27d-r6-liquid_treatment",
        "checkpoint": {
            "path": relative_project_path(root, checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "global_step": int(checkpoint["global_step"]),
        },
        "direct_baseline_checkpoint": {
            "path": relative_project_path(root, reference_path),
            "sha256": sha256_file(reference_path),
            "selected_epoch": int(reference_checkpoint["selected_epoch"]),
        },
        "pools": pools,
        "calibration": {
            "formal_l3_enabled": False,
            "unknown_support": calibration_unknown_support["all_parents"],
            "quality_eligible_unknown_support": calibration_unknown_support["quality_eligible"],
            "route_candidate_unknown_support": calibration_unknown_support["route_candidate"],
            "formal_unknown_support_required": int(config["calibration"]["formal_unknown_support_required"]),
            "formal_support_sufficient": False,
        },
        "endpoint_contract": {
            "classification_logits_input_to_branch": False,
            "softmax_input_to_branch": False,
            "label_identity_input_to_branch": False,
            "frozen_liquid_v2_core": True,
            "frozen_base_readout": True,
            "frozen_verifier_head": True,
        },
        "manifest_sha256": manifest["manifest_sha256"],
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "deployment_eligible": False,
    }
    report_path = artifact_path(
        root,
        config["artifacts"]["evaluation_template"].format(seed=seed),
    )
    write_sealed_json_if_identical(report_path, report)
    return {
        "status": "completed",
        "report_path": relative_project_path(root, report_path),
        "report": report,
    }


def _mean_std(values: list[float]) -> dict[str, Any]:
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "values": values,
    }


def summarize_seeds(
    project_root: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    reports = {
        seed: read_sealed_json(
            artifact_path(
                root,
                config["artifacts"]["evaluation_template"].format(seed=seed),
                must_exist=True,
            ),
            expected_schema=R11_REPORT_SCHEMA_VERSION,
        )
        for seed in R11_SEEDS
    }
    rows: list[dict[str, Any]] = []
    metric_names = (
        "bird_action_precision",
        "bird_action_recall",
        "false_positive_rejection",
        "unknown_rejection",
    )
    for seed, report in reports.items():
        row: dict[str, Any] = {"seed": seed, "pools": {}}
        for role in ("representation_development", "calibration"):
            treatment = report["pools"][role]["treatment"]
            baseline = report["pools"][role]["direct_baseline"]
            row["pools"][role] = {
                "treatment": treatment,
                "direct_baseline": baseline,
                "comparison": report["pools"][role]["comparison"],
            }
        rows.append(row)

    aggregate: dict[str, Any] = {}
    for role in ("representation_development", "calibration"):
        for side in ("treatment", "direct_baseline"):
            prefix = "treatment" if side == "treatment" else "direct_baseline"
            for metric in metric_names:
                aggregate[f"{role}_{prefix}_{metric}"] = _mean_std(
                    [float(row["pools"][role][side]["actions"][metric]) for row in rows]
                )
            aggregate[f"{role}_{prefix}_auroc"] = _mean_std(
                [float(row["pools"][role][side]["ranking"]["auroc_higher_is_known"]) for row in rows]
            )
        for metric in metric_names:
            aggregate[f"{role}_{metric}_delta"] = _mean_std(
                [
                    float(row["pools"][role]["comparison"]["action_delta"][metric])
                    for row in rows
                ]
            )
        aggregate[f"{role}_auroc_delta"] = _mean_std(
            [float(row["pools"][role]["comparison"]["auroc_delta"]) for row in rows]
        )

    maximum_loss = int(config["selection"]["max_additional_bird_rejections_per_pool"])
    recall_safe = all(
        int(row["pools"][role]["treatment"]["actions"]["accepted_bird"])
        >= int(row["pools"][role]["direct_baseline"]["actions"]["accepted_bird"])
        - maximum_loss
        for row in rows
        for role in ("representation_development", "calibration")
    )
    rejection_non_decreasing = all(
        float(row["pools"][role]["treatment"]["actions"]["false_positive_rejection"])
        >= float(row["pools"][role]["direct_baseline"]["actions"]["false_positive_rejection"])
        for row in rows
        for role in ("representation_development", "calibration")
    )
    rejection_gain_each_pool = all(
        float(row["pools"][role]["comparison"]["action_delta"]["false_positive_rejection"]) > 0.0
        for row in rows
        for role in ("representation_development", "calibration")
    )
    ranking_non_degrading = all(
        float(row["pools"][role]["comparison"]["auroc_delta"]) >= 0.0
        for row in rows
        for role in ("representation_development", "calibration")
    )
    same_direction = all(
        (
            float(row["pools"]["representation_development"]["comparison"]["action_delta"]["false_positive_rejection"]) > 0.0
        )
        == (
            float(row["pools"]["calibration"]["comparison"]["action_delta"]["false_positive_rejection"]) > 0.0
        )
        for row in rows
    )
    calibration_support = reports[R11_SEEDS[0]]["pools"]["calibration"]["unknown_support"]
    if any(
        report["pools"]["calibration"]["unknown_support"] != calibration_support
        for report in reports.values()
    ):
        raise ValueError("V27d-r11 calibration unknown support differs between seeds")
    formal_support_sufficient = int(calibration_support["all_parents"]) >= int(config["calibration"]["formal_unknown_support_required"])
    criteria = {
        "recall_safe_within_configured_budget": recall_safe,
        "false_positive_rejection_non_decreasing": rejection_non_decreasing,
        "false_positive_rejection_gain_each_seed_and_pool": rejection_gain_each_pool,
        "ranking_non_degrading_each_seed_and_pool": ranking_non_degrading,
        "development_calibration_same_direction": same_direction,
        "formal_calibration_support_sufficient": formal_support_sufficient,
    }
    candidate = bool(
        recall_safe
        and rejection_gain_each_pool
        and ranking_non_degrading
        and same_direction
    )
    payload = {
        "schema_version": "RADAR-LN3-V27d-r11-two-seed-summary-v1",
        "experiment_id": R11_EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(R11_SEEDS),
        "direct_baseline": "V27d-r6-liquid_treatment",
        "rows": rows,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": "shape_treatment_candidate_for_later_calibration" if candidate else "diagnostic_only_no_promotion",
        "formal_l3_enabled": False,
        "deployment_eligible": False,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
    }
    path = artifact_path(root, config["artifacts"]["summary_path"])
    write_sealed_json_if_identical(path, payload)
    return {
        "status": "completed",
        "summary_path": relative_project_path(root, path),
        "summary": payload,
    }


__all__ = ["evaluate_seed", "summarize_seeds"]
