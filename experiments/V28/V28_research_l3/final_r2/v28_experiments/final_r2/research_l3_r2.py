"""V28 final r2 research-only action replay.

The r2 experiment changes the L3 coverage policy only.  Classifier endpoints,
knownness scores, profile residuals, scorer references and the source replay
are frozen.  The replay uses the current deterministic quality semantics and
the configured 0.99 L1 target candidate floor, then checks an aggregate
Pending budget of at most ten percent.  Pending is never converted into
Unknown rejection.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.runtime import (
    canonical_sha256,
    relative_project_path,
    sha256_file,
)

from v28.final_runtime import load_payload
from v28.final_scorer import score_class_conditional
from v28.final_resolver import resolve_research_action
from v28.heads import deterministic_quality_target
from v28.protocol import V28_L1_TASK, V28_L2_TASK, write_sealed_json
from v28.research_l3 import _action_metrics, _candidate, _probabilities


R2_ROOT = Path("src-v3/v28/final/r2")
RESULTS_ROOT = Path("src-v3/v28/final/results")
SOURCE_CALIBRATION = RESULTS_ROOT / "V28-final-research-calibration-v1.json"
SOURCE_REGISTRY = RESULTS_ROOT / "V28-final-research-registry-v1.json"
SOURCE_REPLAY = RESULTS_ROOT / "V28-final-research-replay-reproduction-v1.json"
L1_CACHE = Path("src-v3/v28/final/calibration-candidates/calibration-l1_known_gate-candidate-v1.pt")
L2_CACHE = Path("src-v3/v28/final/calibration-candidates/calibration-l2_dbb_specialist-candidate-v1.pt")
POLICY = R2_ROOT / "V28-final-research-r2-policy.json"
REPLAY = RESULTS_ROOT / "V28-final-research-r2-replay-v1.json"
REGISTRY = RESULTS_ROOT / "V28-final-research-r2-registry-v1.json"
NO_PENDING_POLICY = R2_ROOT / "V28-final-research-r2-no-pending-budget-policy.json"
NO_PENDING_REPLAY = RESULTS_ROOT / "V28-final-research-r2-no-pending-budget-replay-v1.json"
KNOWNNESS_ONLY_REPLAY = RESULTS_ROOT / "V28-final-research-r2-knownness-only-rerun-v1.json"

PENDING_BUDGET_RATE = 0.10
L1_TARGET_FLOOR = 0.99


def _read_sealed(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is not a JSON object")
    body = {key: item for key, item in value.items() if key != "payload_sha256"}
    if value.get("payload_sha256") != canonical_sha256(body):
        raise ValueError(f"{label} payload hash differs")
    return dict(value)


def _tensor_role_payload(root: Path, relative: Path) -> dict[str, Any]:
    path = (root / relative).resolve()
    path.relative_to((root / Path("src-v3/v28/final")).resolve())
    return load_payload(path)


def _current_quality_context(root: Path) -> SimpleNamespace:
    config_path = root / "src-v3/v28/config/V28-default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return SimpleNamespace(config=SimpleNamespace(heads=config["heads"]))


def _quality_flags(payload: Mapping[str, Any], *, task: str, context: Any) -> np.ndarray:
    tensors = payload["tensors"]
    lengths = tensors["lengths"].long()
    rows = torch.arange(lengths.shape[0])
    targets = deterministic_quality_target(
        context,
        task=task,
        task_facts=tensors["quality_features"][..., 38:].float(),
        event_facts=tensors["quality_features"][..., 19:38].float(),
        padding_mask=tensors["padding_mask"].bool(),
    )
    return targets[rows, lengths - 1].numpy().astype(bool)


def _cache_records(payload: Mapping[str, Any]) -> dict[str, int]:
    parent_ids = [str(value) for value in payload["parent_ids"]]
    if len(parent_ids) != len(set(parent_ids)):
        raise ValueError("V28 r2 cache contains duplicate parent ids")
    return {parent_id: index for index, parent_id in enumerate(parent_ids)}


def _source_bindings(root: Path) -> dict[str, Any]:
    calibration = _read_sealed(root / SOURCE_CALIBRATION, "V28 final v1 calibration")
    registry = _read_sealed(root / SOURCE_REGISTRY, "V28 final v1 registry")
    replay = _read_sealed(root / SOURCE_REPLAY, "V28 final v1 replay")
    if replay.get("registry_sha256") != registry.get("registry_sha256"):
        raise ValueError("V28 r2 source replay uses a different registry")
    if registry.get("calibration_report", {}).get("payload_sha256") != calibration.get("payload_sha256"):
        raise ValueError("V28 r2 source registry calibration binding differs")
    if registry.get("status") != "internal_exploratory" or registry.get("research_only") is not True:
        raise ValueError("V28 r2 source registry is not research-only")
    return {"calibration": calibration, "registry": registry, "replay": replay}


def _build_policy(
    root: Path,
    source: Mapping[str, Any],
    *,
    pending_limit: float | None = PENDING_BUDGET_RATE,
    variant: str = "frozen-final-endpoints-current-quality-semantics-pending-budget",
) -> dict[str, Any]:
    config_path = root / "src-v3/v28/config/V28-default.json"
    manifest_path = root / "src-v3/v28/manifests/V28-manifest-v2.json"
    body = {
        "schema_version": "RADAR-LN3-V28-final-research-r2-policy-v1",
        "experiment_id": "V28-final-research-r2",
        "variant": variant,
        "source_manifest_sha256": source["calibration"].get("system_manifest_sha256"),
        "current_manifest_file_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "current_config_sha256": sha256_file(config_path),
        "experiment_code": {
            "path": "src-v3/v28_experiments/final_r2/research_l3_r2.py",
            "sha256": sha256_file(root / "src-v3/v28_experiments/final_r2/research_l3_r2.py"),
        },
        "source_calibration_role": "calibration",
        "test_payload_access_count": 0,
        "pending_policy": {
            "enforced": pending_limit is not None,
            "max_rate": pending_limit,
            "max_count_rule": "ceil(parent_count * max_rate)" if pending_limit is not None else "none",
            "pending_is_not_unknown_rejection": True,
            "quality_override": False,
            "candidate_conflict_override": False,
        },
        "quality_semantics": {
            "implementation": "current_v28.heads.deterministic_quality_target",
            "zero_delta_dt_is_semantically_valid": True,
            "quality_facts_recomputed_from_frozen_calibration_cache": True,
        },
        "knownness_score_source": "frozen_v1_replay_records",
        "candidate_policy": {
            "l1_target_candidate_probability_floor": L1_TARGET_FLOOR,
            "l1_clutter_candidate_probability_floor": source["calibration"]["tasks"][V28_L1_TASK]["conformal"]["candidate_probability_floor_by_class"]["KNOWN_CLUTTER_EVIDENCE"],
            "l2_candidate_rule": "source_registry_class_conditional_scorer",
        },
        "frozen_components": [
            "L1 classifier endpoint",
            "L2 classifier endpoint",
            "L1/L2 knownness scores and thresholds",
            "L2 profile residual and scorer",
            "Unknown/Pending action labels",
        ],
    }
    return body


def _false_acceptance_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    dbb = {"DroneTarget", "BirdTarget", "BalloonTarget"}
    unknown = [record for record in records if record["label_name"] == "UnknownTarget"]
    clutter = [record for record in records if record["label_name"] == "ClutterTarget"]
    non_target = [*unknown, *clutter]
    dbb_predictions = [record for record in records if record["action"] in dbb]
    dbb_true_positive = sum(record["action"] == record["label_name"] for record in dbb_predictions)
    unknown_as_dbb = sum(record["action"] in dbb for record in unknown)
    unknown_as_clutter = sum(record["action"] == "Clutter" for record in unknown)
    unknown_error_acceptance = sum(record["action"] != "UNKNOWN_REJECTED" for record in unknown)
    clutter_as_dbb = sum(record["action"] in dbb for record in clutter)
    non_target_as_dbb = sum(record["action"] in dbb for record in non_target)

    def rate(successes: int, support: int) -> dict[str, Any]:
        return {
            "successes": int(successes),
            "support": int(support),
            "estimate": None if support == 0 else float(successes / support),
        }

    return {
        "unknown": {
            "support": len(unknown),
            "rejected_unknown_rejected": rate(sum(record["action"] == "UNKNOWN_REJECTED" for record in unknown), len(unknown)),
            "error_acceptance_any_non_rejection": rate(unknown_error_acceptance, len(unknown)),
            "false_acceptance_as_dbb": rate(unknown_as_dbb, len(unknown)),
            "false_acceptance_as_clutter": rate(unknown_as_clutter, len(unknown)),
        },
        "clutter": {
            "support": len(clutter),
            "false_alarm_as_dbb": rate(clutter_as_dbb, len(clutter)),
            "accepted_as_clutter": rate(sum(record["action"] == "Clutter" for record in clutter), len(clutter)),
        },
        "non_target_to_dbb_false_alarm": rate(non_target_as_dbb, len(non_target)),
        "dbb_output_false_discovery": rate(len(dbb_predictions) - dbb_true_positive, len(dbb_predictions)),
        "definition": {
            "unknown_error_acceptance": "UnknownTarget output is anything except UNKNOWN_REJECTED",
            "unknown_as_dbb": "UnknownTarget output is DroneTarget, BirdTarget or BalloonTarget",
            "non_target_to_dbb_false_alarm": "UnknownTarget or ClutterTarget output is a DBB target action",
            "dbb_output_false_discovery": "DBB action output is incorrect divided by all DBB action outputs",
        },
    }


def rerun_knownness_only(root: Path) -> dict[str, Any]:
    """Replay frozen final outputs with only knownness rejection enabled."""

    source = _source_bindings(root)
    calibration = source["calibration"]
    registry = source["registry"]
    source_replay = source["replay"]
    l1_threshold = float(calibration["tasks"][V28_L1_TASK]["knownness"]["selected"]["threshold"])
    l2_threshold = float(calibration["tasks"][V28_L2_TASK]["knownness"]["selected"]["threshold"])
    dbb = set(("DroneTarget", "BirdTarget", "BalloonTarget"))
    records: list[dict[str, Any]] = []

    for source_record in source_replay.get("records", []):
        record = dict(source_record)
        l1_prediction = str(record["l1_prediction"])
        l1_knownness = float(record["l1_knownness"])
        if l1_knownness < l1_threshold:
            action = "UNKNOWN_REJECTED"
            reason = "L1_KNOWNNESS_BELOW_THRESHOLD"
        elif l1_prediction == "KNOWN_CLUTTER_EVIDENCE":
            action = "Clutter"
            reason = "L1_CLUTTER_ACCEPTED_KNOWNNESS_ONLY"
        elif l1_prediction == "KNOWN_TARGET_EVIDENCE":
            l2_knownness = record.get("l2_knownness")
            l2_prediction = record.get("l2_prediction")
            if l2_knownness is None or l2_prediction not in dbb:
                raise ValueError(f"knownness-only replay lacks L2 evidence: {record.get('parent_sample_id')}")
            if float(l2_knownness) < l2_threshold:
                action = "UNKNOWN_REJECTED"
                reason = "L2_KNOWNNESS_BELOW_THRESHOLD"
            else:
                action = str(l2_prediction)
                reason = "DBB_ARGMAX_ACCEPTED_KNOWNNESS_ONLY"
        else:
            raise ValueError(f"knownness-only replay has invalid L1 prediction: {l1_prediction}")
        record["action"] = action
        record["reason_code"] = reason
        record["abstention_policy"] = {
            "deterministic_quality_gate_applied": False,
            "conformal_gate_applied": False,
            "knownness_gate_applied": True,
        }
        records.append(record)

    metrics = _action_metrics(records)
    false_acceptance = _false_acceptance_metrics(records)
    calibration_path = root / SOURCE_CALIBRATION
    report = {
        "schema_version": "RADAR-LN3-V28-knownness-only-rerun-v1",
        "experiment_id": "V28-final-research-r2-knownness-only-rerun",
        "status": "internal_exploratory_counterfactual",
        "research_only": True,
        "deployment_eligible": False,
        "source_manifest_sha256": calibration.get("system_manifest_sha256"),
        "source_registry_sha256": registry.get("registry_sha256"),
        "source_replay": {
            "path": relative_project_path(root, root / SOURCE_REPLAY),
            "file_sha256": sha256_file(root / SOURCE_REPLAY),
            "payload_sha256": source_replay.get("payload_sha256"),
        },
        "counterfactual": {
            "variable": "disable deterministic quality/maturity and conformal abstention; retain L1/L2 knownness rejection",
            "l1_knownness_threshold": l1_threshold,
            "l2_knownness_threshold": l2_threshold,
            "knownness_rejection_retained": True,
            "quality_gate_applied": False,
            "conformal_gate_applied": False,
            "fallback": "classifier argmax after knownness passes",
        },
        "evaluation_scope": {
            "role": "calibration",
            "calibration_is_fitting_role": True,
            "test_payload_access_count": 0,
            "post_selection_audit_access_count": 0,
            "independent_validation": False,
        },
        "parent_count": len(records),
        "action_metrics": metrics,
        "false_acceptance_metrics": false_acceptance,
        "records": records,
    }
    path = root / KNOWNNESS_ONLY_REPLAY
    write_sealed_json(path, report)
    return {
        "path": relative_project_path(root, path),
        "sha256": sha256_file(path),
        "parent_count": len(records),
        "action_metrics": metrics,
        "false_acceptance_metrics": false_acceptance,
        "evaluation_role": "calibration_counterfactual_only",
    }


def _replay(
    root: Path,
    *,
    source: Mapping[str, Any],
    policy: Mapping[str, Any],
    output_schema: str = "RADAR-LN3-V28-final-research-r2-replay-v1",
) -> dict[str, Any]:
    calibration = source["calibration"]
    registry = source["registry"]
    source_replay = source["replay"]
    l1_payload = _tensor_role_payload(root, L1_CACHE)
    l2_payload = _tensor_role_payload(root, L2_CACHE)
    l1_index = _cache_records(l1_payload)
    l2_index = _cache_records(l2_payload)
    source_records = list(source_replay.get("records", []))
    source_ids = [str(record["parent_sample_id"]) for record in source_records]
    if len(source_ids) != len(set(source_ids)) or set(source_ids) != set(l1_index):
        raise ValueError("V28 r2 source replay and L1 cache parent ids differ")
    if not set(l2_index).issubset(set(l1_index)):
        raise ValueError("V28 r2 L2 cache parents are absent from L1 cache")

    config_context = _current_quality_context(root)
    l1_quality = _quality_flags(l1_payload, task=V28_L1_TASK, context=config_context)
    l2_quality = _quality_flags(l2_payload, task=V28_L2_TASK, context=config_context)

    l1_tensors = l1_payload["tensors"]
    l1_lengths = l1_tensors["lengths"].long()
    l1_rows = torch.arange(l1_lengths.shape[0])
    l1_probabilities = _probabilities(
        l1_tensors["sequence_logits"][l1_rows, l1_lengths - 1].numpy(),
        float(calibration["tasks"][V28_L1_TASK]["temperature"]["temperature"]),
    )
    l2_tensors = l2_payload["tensors"]
    l2_lengths = l2_tensors["lengths"].long()
    l2_rows = torch.arange(l2_lengths.shape[0])
    l2_probabilities = _probabilities(
        l2_tensors["sequence_logits"][l2_rows, l2_lengths - 1].numpy(),
        float(calibration["tasks"][V28_L2_TASK]["temperature"]["temperature"]),
    )
    l2_scores = score_class_conditional(
        l2_probabilities,
        calibration["tasks"][V28_L2_TASK]["scorer"],
    )

    l1_names = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
    l1_floors = {
        "KNOWN_TARGET_EVIDENCE": float(policy["candidate_policy"]["l1_target_candidate_probability_floor"]),
        "KNOWN_CLUTTER_EVIDENCE": float(policy["candidate_policy"]["l1_clutter_candidate_probability_floor"]),
    }
    l1_threshold = float(calibration["tasks"][V28_L1_TASK]["knownness"]["selected"]["threshold"])
    l2_threshold = float(calibration["tasks"][V28_L2_TASK]["knownness"]["selected"]["threshold"])
    records: list[dict[str, Any]] = []
    for source_record in source_records:
        parent_id = str(source_record["parent_sample_id"])
        l1_row = l1_index[parent_id]
        l1_candidates = _candidate(l1_probabilities[l1_row], l1_names, l1_floors)
        l1_prediction = l1_names[int(l1_probabilities[l1_row].argmax())]
        l1_quality_eligible = bool(l1_quality[l1_row])
        l1_knownness = float(source_record["l1_knownness"]) >= l1_threshold
        l1_target = l1_prediction == "KNOWN_TARGET_EVIDENCE"
        l1_clutter = l1_prediction == "KNOWN_CLUTTER_EVIDENCE"

        l2_row = l2_index.get(parent_id)
        if l2_row is None:
            l2_candidates: list[str] = []
            l2_prediction = None
            l2_knownness = None
            l2_quality_eligible = None
        else:
            l2_candidates = list(l2_scores["candidate_sets"][l2_row])
            l2_prediction = str(l2_scores["predictions"][l2_row])
            l2_knownness = float(source_record["l2_knownness"]) >= l2_threshold
            l2_quality_eligible = bool(l2_quality[l2_row])

        decision = resolve_research_action(
            registry=registry,
            l1_candidate_set=l1_candidates,
            l1_prediction=l1_prediction,
            l1_knownness=l1_knownness,
            l1_quality_eligible=l1_quality_eligible,
            l1_target=l1_target,
            l1_clutter=l1_clutter,
            l2_candidate_set=l2_candidates,
            l2_prediction=l2_prediction,
            l2_knownness=l2_knownness,
            l2_quality_eligible=l2_quality_eligible,
            streaming=False,
        )
        records.append(
            {
                **source_record,
                "action": decision.action,
                "reason_code": decision.reason_code,
                "decision_details": decision.details,
                "l1_candidates": l1_candidates,
                "l2_candidates": l2_candidates,
                "l1_prediction": l1_prediction,
                "l2_prediction": l2_prediction,
                "l1_quality_eligible": l1_quality_eligible,
                "l2_quality_eligible": l2_quality_eligible,
                "r2_quality_policy": {
                    "quality_override_applied": False,
                    "source_quality_eligible": bool(source_record["l1_quality_eligible"]),
                    "current_semantic_quality_eligible": l1_quality_eligible,
                },
            }
        )

    metrics = _action_metrics(records)
    pending_count = sum(record["action"] == "PENDING_ABSTAIN" for record in records)
    max_rate = policy["pending_policy"].get("max_rate")
    maximum = None if max_rate is None else int(math.ceil(len(records) * float(max_rate)))
    within_budget = None if maximum is None else pending_count <= maximum
    body = {
        "schema_version": output_schema,
        "experiment_id": "V28-final-research-r2-no-pending-budget" if maximum is None else "V28-final-research-r2",
        "status": "completed" if within_budget is not False else "pending_budget_infeasible",
        "research_only": True,
        "deployment_eligible": False,
        "source_manifest_sha256": source["calibration"].get("system_manifest_sha256"),
        "current_v28_code": {
            "heads_sha256": sha256_file(root / "src-v3/v28/heads.py"),
            "config_sha256": sha256_file(root / "src-v3/v28/config/V28-default.json"),
        },
        "source_artifacts": {
            "calibration": {"path": relative_project_path(root, root / SOURCE_CALIBRATION), "sha256": sha256_file(root / SOURCE_CALIBRATION), "payload_sha256": calibration.get("payload_sha256")},
            "registry": {"path": relative_project_path(root, root / SOURCE_REGISTRY), "sha256": sha256_file(root / SOURCE_REGISTRY), "registry_sha256": registry.get("registry_sha256")},
            "replay": {"path": relative_project_path(root, root / SOURCE_REPLAY), "sha256": sha256_file(root / SOURCE_REPLAY), "payload_sha256": source_replay.get("payload_sha256")},
            "l1_cache": {"path": relative_project_path(root, root / L1_CACHE), "sha256": sha256_file(root / L1_CACHE), "cache_sha256": l1_payload.get("cache_sha256")},
            "l2_cache": {"path": relative_project_path(root, root / L2_CACHE), "sha256": sha256_file(root / L2_CACHE), "cache_sha256": l2_payload.get("cache_sha256")},
        },
        "policy": dict(policy),
        "quality_support": {"l1_parent_count": len(l1_quality), "l1_quality_false": int((~l1_quality).sum()), "l2_parent_count": len(l2_quality), "l2_quality_false": int((~l2_quality).sum())},
        "evaluation_scope": {"role": "calibration", "calibration_is_fitting_role": True, "test_payload_access_count": 0, "post_selection_audit_access_count": 0, "independent_validation": False},
        "pending_budget": {"enforced": bool(policy["pending_policy"].get("enforced")), "max_rate": max_rate, "max_count": maximum, "observed_count": pending_count, "observed_rate": float(metrics["pending_rate"]["estimate"]), "within_budget": within_budget},
        "action_metrics": metrics,
        "records": records,
    }
    return body


def run(root: Path) -> dict[str, Any]:
    source = _source_bindings(root)
    policy = _build_policy(root, source)
    policy_path = root / POLICY
    write_sealed_json(policy_path, policy)
    replay = _replay(root, source=source, policy=policy)
    replay_path = root / REPLAY
    write_sealed_json(replay_path, replay)
    registry_body = {
        "schema_version": "RADAR-LN3-V28-final-research-r2-registry-v1",
        "experiment_id": "V28-final-research-r2",
        "status": "internal_exploratory",
        "research_only": True,
        "deployment_eligible": False,
        "status_reasons": ["research_only", "calibration_role_replay_only", "new_unexposed_post_selection_audit_required"],
        "source_manifest_sha256": source["calibration"].get("system_manifest_sha256"),
        "policy": {"path": relative_project_path(root, policy_path), "sha256": sha256_file(policy_path), "payload_sha256": json.loads(policy_path.read_text(encoding="utf-8")).get("payload_sha256")},
        "source_registry_sha256": source["registry"].get("registry_sha256"),
        "frozen_endpoints": source["registry"].get("final_heads"),
        "head_compositions": source["registry"].get("head_compositions"),
        "scorers": source["registry"].get("scorers"),
        "knownness_score_source": policy["knownness_score_source"],
        "pending_policy": policy["pending_policy"],
        "replay": {"path": relative_project_path(root, replay_path), "sha256": sha256_file(replay_path), "payload_sha256": json.loads(replay_path.read_text(encoding="utf-8")).get("payload_sha256")},
        "post_selection_audit": {"status": "new_unexposed_pool_required", "threshold_fitting_allowed": False},
        "test_payload_access_count": 0,
    }
    registry_body["registry_sha256"] = canonical_sha256(registry_body)
    registry_path = root / REGISTRY
    write_sealed_json(registry_path, registry_body)
    return {"policy": relative_project_path(root, policy_path), "replay": relative_project_path(root, replay_path), "registry": relative_project_path(root, registry_path), "registry_sha256": registry_body["registry_sha256"], "pending_budget": replay["pending_budget"], "action_metrics": replay["action_metrics"]}


def run_without_pending_budget(root: Path) -> dict[str, Any]:
    """Keep refusal gates active while removing only the aggregate budget."""

    source = _source_bindings(root)
    policy = _build_policy(
        root,
        source,
        pending_limit=None,
        variant="frozen-final-endpoints-current-quality-semantics-no-pending-budget",
    )
    policy_path = root / NO_PENDING_POLICY
    write_sealed_json(policy_path, policy)
    replay = _replay(
        root,
        source=source,
        policy=policy,
        output_schema="RADAR-LN3-V28-final-research-r2-no-pending-budget-replay-v1",
    )
    replay_path = root / NO_PENDING_REPLAY
    write_sealed_json(replay_path, replay)
    return {
        "policy": relative_project_path(root, policy_path),
        "replay": relative_project_path(root, replay_path),
        "pending_budget": replay["pending_budget"],
        "action_metrics": replay["action_metrics"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V28 final r2 research-only Pending-budget replay")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("command", choices=("run", "no-pending-budget", "knownness-only-rerun"))
    args = parser.parse_args(argv)
    if args.command == "run":
        result = run(args.project_root.resolve())
    elif args.command == "no-pending-budget":
        result = run_without_pending_budget(args.project_root.resolve())
    else:
        result = rerun_knownness_only(args.project_root.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
