"""Paired evaluation of causal-prefix lower-quartile aggregation."""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any, Mapping

import torch

from radar_ln3.runtime import relative_project_path, sha256_file, verify_file_hash
from v27.V27d.logging_utils import close_logger, configure_logger, log_event
from v27.V27d.metrics import binary_auroc
from v27.V27d.r6.features import load_population_cache
from v27.V27d.r6.protocol import load_context as load_r6_context
from v27.V27d.r6.training import load_checkpoint as load_r6_checkpoint

from .prefix_aggregate import aggregate_prefix_quantile
from .protocol import R12_EXPERIMENT_ID, R12_REPORT_SCHEMA_VERSION, R12_ROLES, R12_SEEDS, artifact_path, load_context, read_sealed_json, source_path, write_sealed_json_if_identical


def _cache_path(root: Path, config: Mapping[str, Any], role: str) -> Path:
    if role not in R12_ROLES:
        raise ValueError("V27d-r12 cache role is invalid")
    return source_path(
        root,
        config["source"]["r6_cache_template"].format(role=role),
        must_exist=True,
    )


def _load_caches(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    source = config["source"]
    r6_config_path = source_path(root, source["r6_config_path"], must_exist=True)
    r6_manifest_path = source_path(root, source["r6_manifest_path"], must_exist=True)
    verify_file_hash(r6_config_path, source["expected_r6_config_sha256"], field="V27d-r12 r6 config")
    verify_file_hash(r6_manifest_path, source["expected_r6_manifest_sha256"], field="V27d-r12 r6 manifest")
    _, r6_manifest = load_r6_context(root, config_path=r6_config_path)
    if r6_manifest["manifest_sha256"] != source["expected_r6_manifest_sha256"]:
        raise ValueError("V27d-r12 active r6 manifest hash differs")
    caches: dict[str, dict[str, Any]] = {}
    for role in R12_ROLES:
        path = _cache_path(root, config, role)
        verify_file_hash(
            path,
            source["expected_cache_file_sha256s"][role],
            field=f"V27d-r12 r6 {role} cache",
        )
        payload = load_population_cache(path)
        if payload.get("cache_sha256") != source["expected_cache_payload_sha256s"][role]:
            raise ValueError(f"V27d-r12 r6 {role} cache payload hash differs")
        if payload.get("r6_manifest_sha256") != source["expected_r6_manifest_sha256"]:
            raise ValueError(f"V27d-r12 r6 {role} manifest binding differs")
        caches[role] = payload
    if manifest.get("source", {}).get("r6_manifest_sha256") != source["expected_r6_manifest_sha256"]:
        raise ValueError("V27d-r12 manifest source binding differs")
    return caches


def _load_reference(
    root: Path,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[torch.nn.Module, dict[str, Any], Path]:
    if int(seed) not in R12_SEEDS:
        raise ValueError("V27d-r12 seed is not registered")
    source = config["source"]
    path = source_path(root, source["r6_checkpoint_template"].format(seed=seed), must_exist=True)
    verify_file_hash(
        path,
        source["expected_checkpoint_file_sha256s"][str(seed)],
        field=f"V27d-r12 r6 checkpoint seed {seed}",
    )
    r6_config_path = source_path(root, source["r6_config_path"], must_exist=True)
    model, payload, _ = load_r6_checkpoint(
        root,
        arm="liquid_treatment",
        seed=int(seed),
        config_path=r6_config_path,
    )
    if payload.get("head_state_sha256") != source["expected_checkpoint_head_state_sha256s"][str(seed)]:
        raise ValueError(f"V27d-r12 r6 head state hash differs for seed {seed}")
    return model.eval(), dict(payload), path


def _inputs(
    payload: Mapping[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors = payload["tensors"]
    indices = torch.arange(len(payload["parent_ids"]), dtype=torch.long)
    return {
        name: tensors[name][indices].to(device)
        for name in (
            "summary_features",
            "event_trajectory",
            "task_facts_sequence",
            "event_facts_sequence",
            "padding_mask",
        )
    }


def _model_scores(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        output = model(**_inputs(payload, torch.device("cpu")))
    return (
        output["verifier_score_sequence"].detach().cpu(),
        output["verifier_score_final"].detach().cpu(),
    )


def _masks(
    payload: Mapping[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensors = payload["tensors"]
    candidate = tensors["candidate_population"].bool()
    positive = candidate & tensors["bird_positive"].bool()
    negative = candidate & tensors["route_false_positive_negative"].bool()
    if not torch.equal(candidate, positive | negative) or bool((positive & negative).any()):
        raise ValueError("V27d-r12 population masks are invalid")
    return candidate, positive, negative


def _unknown_mask(payload: Mapping[str, Any]) -> torch.Tensor:
    labels = [str(value) for value in payload["label_names"]]
    candidate = payload["tensors"]["candidate_population"].bool()
    return candidate & torch.tensor(
        [label == "UnknownTarget" for label in labels], dtype=torch.bool
    )


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


def _actions(
    payload: Mapping[str, Any],
    scores: torch.Tensor,
    threshold: float,
) -> dict[str, Any]:
    candidate, positive, negative = _masks(payload)
    labels = [str(value) for value in payload["label_names"]]
    accepted = candidate & scores.ge(float(threshold))
    unknown = _unknown_mask(payload)
    by_label: dict[str, dict[str, int]] = {}
    for label in sorted(set(labels)):
        label_mask = candidate & torch.tensor(
            [item == label for item in labels], dtype=torch.bool
        )
        by_label[label] = {
            "support": int(label_mask.sum()),
            "accepted": int((accepted & label_mask).sum()),
            "rejected": int((label_mask & ~accepted).sum()),
        }
    return {
        "threshold": float(threshold),
        "candidate_support": int(candidate.sum()),
        "bird_support": int(positive.sum()),
        "false_positive_support": int(negative.sum()),
        "accepted_bird": int((accepted & positive).sum()),
        "rejected_bird": int((positive & ~accepted).sum()),
        "bird_action_precision": float((accepted & positive).sum().item() / max(1, accepted.sum().item())),
        "bird_action_recall": float((accepted & positive).sum().item() / max(1, positive.sum().item())),
        "accepted_false_positive": int((accepted & negative).sum()),
        "rejected_false_positive": int((negative & ~accepted).sum()),
        "false_positive_rejection": float((negative & ~accepted).sum().item() / max(1, negative.sum().item())),
        "unknown_support": int(unknown.sum()),
        "unknown_rejection": float((unknown & ~accepted).sum().item() / max(1, unknown.sum().item())),
        "pending_low_quality_excluded": int((~payload["tensors"]["quality_target_final"].bool()).sum()),
        "by_label": by_label,
    }


def _select_threshold(
    payload: Mapping[str, Any],
    scores: torch.Tensor,
    *,
    baseline_accepted_bird: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    _, positive, negative = _masks(payload)
    values = sorted(float(value) for value in scores[positive].tolist())
    if not values:
        raise ValueError("V27d-r12 development pool has no Bird positive")
    values.append(
        float(
            torch.nextafter(
                torch.tensor(values[-1], dtype=scores.dtype),
                torch.tensor(float("inf"), dtype=scores.dtype),
            ).item()
        )
    )
    budget = int(config["selection"]["max_additional_bird_rejections_per_pool"])
    minimum_bird = max(0, int(baseline_accepted_bird) - budget)
    minimum_recall = float(config["selection"]["minimum_bird_recall"])
    best: tuple[tuple[int, int, float], dict[str, Any]] | None = None
    for threshold in values:
        accepted = payload["tensors"]["candidate_population"].bool() & scores.ge(threshold)
        accepted_bird = int((accepted & positive).sum())
        if (
            accepted_bird < minimum_bird
            or accepted_bird / max(1, int(positive.sum())) < minimum_recall
        ):
            continue
        rejected_negative = int((negative & ~accepted).sum())
        accepted_candidate = accepted & (positive | negative)
        record = {
            "feasible": True,
            "threshold": threshold,
            "higher_is_known": True,
            "accepted_bird": accepted_bird,
            "rejected_bird": int(positive.sum()) - accepted_bird,
            "bird_support": int(positive.sum()),
            "accepted_false_positive": int((accepted & negative).sum()),
            "rejected_false_positive": rejected_negative,
            "false_positive_support": int(negative.sum()),
            "false_positive_rejection": rejected_negative / max(1, int(negative.sum())),
            "bird_precision": accepted_bird / max(1, int(accepted_candidate.sum())),
        }
        ranking = (rejected_negative, accepted_bird, -threshold)
        if best is None or ranking > best[0]:
            best = (ranking, record)
    if best is None:
        raise RuntimeError("V27d-r12 found no recall-safe threshold")
    return best[1]


def _comparison(
    treatment: Mapping[str, Any],
    baseline: Mapping[str, Any],
) -> dict[str, Any]:
    ta = treatment["actions"]
    ba = baseline["actions"]
    return {
        "threshold_delta": float(treatment["threshold"]) - float(baseline["threshold"]),
        "bird_action_precision_delta": float(ta["bird_action_precision"]) - float(ba["bird_action_precision"]),
        "bird_action_recall_delta": float(ta["bird_action_recall"]) - float(ba["bird_action_recall"]),
        "false_positive_rejection_delta": float(ta["false_positive_rejection"]) - float(ba["false_positive_rejection"]),
        "unknown_rejection_delta": float(ta["unknown_rejection"]) - float(ba["unknown_rejection"]),
        "auroc_delta": float(treatment["ranking"]["auroc_higher_is_known"]) - float(baseline["ranking"]["auroc_higher_is_known"]),
        "accepted_bird_delta": int(ta["accepted_bird"]) - int(ba["accepted_bird"]),
        "accepted_false_positive_delta": int(ta["accepted_false_positive"]) - int(ba["accepted_false_positive"]),
    }


def evaluate_seed(
    project_root: str | Path,
    *,
    seed: int,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) not in R12_SEEDS:
        raise ValueError("V27d-r12 seed is not registered")
    caches = _load_caches(root, config, manifest)
    reference, reference_checkpoint, reference_path = _load_reference(root, config, seed=seed)
    development = caches["representation_development"]
    calibration = caches["calibration"]
    development_sequence, development_final = _model_scores(reference, development)
    calibration_sequence, calibration_final = _model_scores(reference, calibration)
    quantile = float(config["aggregation"]["quantile"])
    development_treatment_scores = aggregate_prefix_quantile(
        development_sequence,
        development_final,
        development["tensors"]["prefix_route_mask"].bool(),
        development["tensors"]["candidate_population"].bool(),
        development["tensors"]["padding_mask"].bool(),
        quantile=quantile,
    )
    calibration_treatment_scores = aggregate_prefix_quantile(
        calibration_sequence,
        calibration_final,
        calibration["tensors"]["prefix_route_mask"].bool(),
        calibration["tensors"]["candidate_population"].bool(),
        calibration["tensors"]["padding_mask"].bool(),
        quantile=quantile,
    )
    reference_threshold = float(reference_checkpoint["selection"]["threshold"])
    baseline_development = {
        "threshold": reference_threshold,
        "ranking": _ranking(development, development_final),
        "actions": _actions(development, development_final, reference_threshold),
    }
    treatment_selection = _select_threshold(
        development,
        development_treatment_scores,
        baseline_accepted_bird=baseline_development["actions"]["accepted_bird"],
        config=config,
    )
    treatment_threshold = float(treatment_selection["threshold"])
    pools: dict[str, Any] = {}
    for role, scores, final in (
        ("representation_development", development_treatment_scores, development_final),
        ("calibration", calibration_treatment_scores, calibration_final),
    ):
        payload = caches[role]
        baseline = baseline_development if role == "representation_development" else {
            "threshold": reference_threshold,
            "ranking": _ranking(payload, final),
            "actions": _actions(payload, final, reference_threshold),
        }
        treatment = {
            "threshold": treatment_threshold,
            "selection": treatment_selection if role == "representation_development" else None,
            "ranking": _ranking(payload, scores),
            "actions": _actions(payload, scores, treatment_threshold),
        }
        pools[role] = {
            "treatment": treatment,
            "direct_baseline": baseline,
            "comparison": _comparison(treatment, baseline),
            "unknown_support": {
                "all_parents": int(sum(label == "UnknownTarget" for label in payload["label_names"])),
                "route_candidate": int(_unknown_mask(payload).sum()),
                "quality_eligible_route_candidate": int((_unknown_mask(payload) & payload["tensors"]["quality_target_final"].bool()).sum()),
            },
        }
    calibration_support = pools["calibration"]["unknown_support"]
    log_path = artifact_path(root, config["artifacts"]["log_template"].format(seed=seed))
    logger = configure_logger(log_path, name=f"v27d.r12.seed.{seed}")
    try:
        log_event(
            logger,
            "evaluation_complete",
            experiment_id=R12_EXPERIMENT_ID,
            seed=seed,
            aggregation="prefix_p25",
            development_unknown_rejection=pools["representation_development"]["treatment"]["actions"]["unknown_rejection"],
            calibration_unknown_rejection=pools["calibration"]["treatment"]["actions"]["unknown_rejection"],
            formal_l3_enabled=False,
            unknown_payload_access_count=0,
            test_payload_access_count=0,
        )
    finally:
        close_logger(logger)
    report = {
        "schema_version": R12_REPORT_SCHEMA_VERSION,
        "experiment_id": R12_EXPERIMENT_ID,
        "status": "completed",
        "seed": int(seed),
        "direct_baseline": "V27d-r6-liquid_treatment",
        "aggregation": {
            "method": config["aggregation"]["method"],
            "quantile": quantile,
            "threshold_source": "representation_development",
        },
        "direct_baseline_checkpoint": {
            "path": relative_project_path(root, reference_path),
            "sha256": sha256_file(reference_path),
            "selected_epoch": int(reference_checkpoint["selected_epoch"]),
        },
        "pools": pools,
        "calibration": {
            "formal_l3_enabled": False,
            "unknown_support": calibration_support["all_parents"],
            "route_candidate_unknown_support": calibration_support["route_candidate"],
            "quality_eligible_route_candidate_unknown_support": calibration_support["quality_eligible_route_candidate"],
            "formal_unknown_support_required": int(config["calibration"]["formal_unknown_support_required"]),
            "formal_support_sufficient": False,
        },
        "log_path": relative_project_path(root, log_path),
        "endpoint_contract": {
            "classification_logits_input_to_aggregator": False,
            "softmax_input_to_aggregator": False,
            "label_identity_input_to_aggregator": False,
            "liquid_v2_frozen": True,
            "knownness_head_frozen": True,
            "classification_endpoints_frozen": True,
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
            expected_schema=R12_REPORT_SCHEMA_VERSION,
        )
        for seed in R12_SEEDS
    }
    rows: list[dict[str, Any]] = []
    for seed, report in reports.items():
        rows.append(
            {
                "seed": seed,
                "development": report["pools"]["representation_development"],
                "calibration": report["pools"]["calibration"],
            }
        )
    metric_names = (
        "bird_action_precision",
        "bird_action_recall",
        "false_positive_rejection",
        "unknown_rejection",
    )
    aggregate: dict[str, Any] = {}
    for role_name, role_key in (
        ("development", "representation_development"),
        ("calibration", "calibration"),
    ):
        for side, label in (("treatment", "treatment"), ("direct_baseline", "direct_baseline")):
            for metric in metric_names:
                aggregate[f"{role_name}_{side}_{metric}"] = _mean_std(
                    [float(row[role_name][label]["actions"][metric]) for row in rows]
                )
            aggregate[f"{role_name}_{side}_auroc"] = _mean_std(
                [float(row[role_name][label]["ranking"]["auroc_higher_is_known"]) for row in rows]
            )
        for metric in metric_names:
            aggregate[f"{role_name}_{metric}_delta"] = _mean_std(
                [float(row[role_name]["comparison"][f"{metric}_delta"]) for row in rows]
            )
        aggregate[f"{role_name}_auroc_delta"] = _mean_std(
            [float(row[role_name]["comparison"]["auroc_delta"]) for row in rows]
        )

    minimum_recall = float(config["selection"]["minimum_bird_recall"])
    maximum_loss = int(config["selection"]["max_additional_bird_rejections_per_pool"])
    recall_floor = all(
        float(row[role]["treatment"]["actions"]["bird_action_recall"]) >= minimum_recall
        for row in rows
        for role in ("development", "calibration")
    )
    bird_recall_budget = all(
        int(row[role]["treatment"]["actions"]["accepted_bird"])
        >= int(row[role]["direct_baseline"]["actions"]["accepted_bird"]) - maximum_loss
        for row in rows
        for role in ("development", "calibration")
    )
    calibration_unknown_gain = all(
        float(row["calibration"]["comparison"]["unknown_rejection_delta"]) > 0.0
        for row in rows
    )
    development_route_gain = all(
        float(row["development"]["comparison"]["false_positive_rejection_delta"]) > 0.0
        for row in rows
    )
    same_direction = all(
        (
            float(row["development"]["comparison"]["false_positive_rejection_delta"]) > 0.0
        )
        == (
            float(row["calibration"]["comparison"]["false_positive_rejection_delta"]) > 0.0
        )
        for row in rows
    )
    calibration_support = reports[R12_SEEDS[0]]["calibration"]
    formal_support_sufficient = int(calibration_support["route_candidate_unknown_support"]) >= int(config["calibration"]["formal_unknown_support_required"])
    criteria = {
        "bird_recall_floor_met_each_seed_and_pool": recall_floor,
        "bird_recall_budget_met_each_seed_and_pool": bird_recall_budget,
        "calibration_unknown_rejection_gain_each_seed": calibration_unknown_gain,
        "development_route_rejection_gain_each_seed": development_route_gain,
        "development_calibration_same_direction": same_direction,
        "formal_calibration_support_sufficient": formal_support_sufficient,
    }
    candidate = bool(
        recall_floor
        and bird_recall_budget
        and calibration_unknown_gain
        and development_route_gain
        and same_direction
    )
    payload = {
        "schema_version": "RADAR-LN3-V27d-r12-two-seed-summary-v1",
        "experiment_id": R12_EXPERIMENT_ID,
        "manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(R12_SEEDS),
        "direct_baseline": "V27d-r6-liquid_treatment",
        "rows": rows,
        "aggregate": aggregate,
        "criteria": criteria,
        "decision": "prefix_p25_candidate_for_later_risk_calibration" if candidate else "diagnostic_only_no_promotion",
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