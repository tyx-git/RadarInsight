"""Final V28 frozen research L3 calibration and replay."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from radar_ln3.runtime import (
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)

from .final_resolver import resolve_research_action
from .final_runtime import CALIBRATION_ROOT, artifact, final_path, load_final_l1, load_final_l2
from .final_scorer import build_scorer_descriptor, score_class_conditional, serialize_score, validate_scorer_descriptor
from .head_training import _quality_targets, _select_knownness, _select_quality
from .protocol import (
    V28Context,
    V28_L1_CLASSES,
    V28_L1_TASK,
    V28_L2_CLASSES,
    V28_L2_TASK,
    _source_hashes,
    build_manifest,
    load_config,
    load_context,
    read_sealed_json,
)
from .replay import _action_metrics


RESULTS_ROOT = Path("src-v3/v28/final/results")
CALIBRATION = "V28-final-research-calibration-v1.json"
REPLAY = "V28-final-research-replay-v1.json"
REPLAY_REPRODUCTION = "V28-final-research-replay-reproduction-v1.json"
NO_ABSTAIN_REPLAY = "V28-final-research-replay-no-abstain-v1.json"
REGISTRY = "V28-final-research-registry-v1.json"


def _path(context: V28Context, name: str) -> Path:
    path = (context.project_root / RESULTS_ROOT / name).resolve()
    path.relative_to((context.project_root / RESULTS_ROOT).resolve())
    return path


def _research_context(
    project_root: Path,
    *,
    manifest_sha256: str | None,
) -> V28Context:
    if manifest_sha256 is None:
        return load_context(project_root, None)
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in manifest_sha256
    ):
        raise ValueError("manifest SHA-256 must contain exactly 64 hexadecimal characters")
    digest = manifest_sha256.lower()
    archive_root = project_root / "src-v3/v28/manifests/archive"
    candidate = archive_root / f"V28-manifest-v2.json.stale-before-{digest[:16]}"
    if not candidate.is_file() or sha256_file(candidate) != digest:
        raise ValueError("requested V28 historical manifest is not present or its hash differs")
    manifest = read_sealed_json(candidate)
    config = load_config(project_root, None)
    expected = build_manifest(project_root, config, _source_hashes(project_root, config))
    if manifest.get("config", {}).get("identity_sha256") != config.identity_sha256:
        raise ValueError("historical replay manifest uses a different V28 model configuration")
    for key in ("source_hashes", "public_dependency_hashes", "tasks", "liquid_v2", "protocol"):
        if manifest.get(key) != expected.get(key):
            raise ValueError(f"historical replay manifest differs from the current contract: {key}")
    # This replay loads frozen checkpoints, representations and heads; the
    # classifier training implementation is not part of its forward path.
    for relative_path in ("src-v3/v28/model.py", "src-v3/v28/data.py"):
        if manifest.get("code_hashes", {}).get(relative_path) != expected.get("code_hashes", {}).get(relative_path):
            raise ValueError(f"historical replay manifest differs from model code: {relative_path}")
    return V28Context(project_root, config, manifest, candidate, digest)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    value = dict(json.loads(path.read_text(encoding="utf-8")))
    declared = value.pop("payload_sha256", None)
    if declared != canonical_sha256(value):
        raise ValueError(f"{label} payload hash differs")
    return value


def _cache(context: V28Context, task: str) -> dict[str, Any]:
    path = final_path(context.project_root, CALIBRATION_ROOT / f"calibration-{task}-candidate-v1.pt")
    value = dict(torch.load(path, map_location="cpu", weights_only=False)); tensors = value["tensors"]
    tensor_hash = canonical_sha256([(name, str(tensors[name].dtype), tuple(tensors[name].shape), tensors[name].detach().cpu().contiguous().numpy().tobytes().hex()) for name in sorted(tensors)])
    expected = canonical_sha256({"metadata": {key: item for key, item in value.items() if key not in {"tensors", "cache_sha256"}}, "tensor_sha256": tensor_hash})
    if value.get("cache_sha256") != expected:
        raise ValueError(f"final calibration cache hash differs: {path}")
    return value


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    shifted = logits / float(temperature); shifted -= shifted.max(axis=1, keepdims=True); exp = np.exp(shifted); return exp / exp.sum(axis=1, keepdims=True)


def _temperature(logits: np.ndarray, labels: np.ndarray, context: V28Context) -> dict[str, Any]:
    lo = float(context.config.calibration["temperature_min"]); hi = float(context.config.calibration["temperature_max"]); count = int(context.config.calibration["temperature_grid_size"]); losses = []
    for temperature in np.exp(np.linspace(math.log(lo), math.log(hi), count)):
        probabilities = _probabilities(logits, float(temperature)); losses.append((float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean()), float(temperature)))
    loss, selected = min(losses); return {"status": "fitted", "temperature": selected, "nll": loss, "grid_size": count, "min": lo, "max": hi, "support": int(len(labels))}


def _conformal(probabilities: np.ndarray, labels: np.ndarray, names: tuple[str, ...], alpha: float) -> dict[str, Any]:
    quantiles: dict[str, float] = {}; floors: dict[str, float] = {}; supports: dict[str, int] = {}; coverage: dict[str, float] = {}
    for class_id, name in enumerate(names):
        selected = probabilities[labels == class_id, class_id]; supports[name] = int(len(selected))
        if not len(selected): quantiles[name] = 1.0; floors[name] = 0.0; coverage[name] = 0.0; continue
        nonconformity = 1.0 - selected; rank = min(len(nonconformity), max(1, int(math.ceil((len(nonconformity) + 1) * (1.0 - alpha))))); quantile = float(np.sort(nonconformity)[rank - 1]); quantiles[name] = quantile; floors[name] = 1.0 - quantile; coverage[name] = float((nonconformity <= quantile + 1e-12).mean())
    return {"method": "class_conditional_split_conformal_1_minus_calibrated_probability_higher", "error_rate": float(alpha), "quantile_by_class": quantiles, "candidate_probability_floor_by_class": floors, "support_by_class": supports, "coverage_by_class": coverage, "support": int(len(labels))}


def _candidate(probabilities: np.ndarray, names: tuple[str, ...], floors: Mapping[str, Any]) -> list[str]:
    return [name for index, name in enumerate(names) if float(probabilities[index]) + 1e-12 >= float(floors.get(name, 0.0))]


def _scores(context: V28Context, task: str) -> dict[str, Any]:
    payload = _cache(context, task); tensors = payload["tensors"]; lengths = tensors["lengths"].long(); rows = torch.arange(lengths.shape[0])
    if task == V28_L1_TASK:
        candidate = load_final_l1(context)
        with torch.inference_mode(): output = candidate["head"].forward_features(tensors["knownness_features"].float(), tensors["quality_features"].float(), tensors["padding_mask"].bool())
        path = candidate["path"]
    else:
        candidate = load_final_l2(context)
        with torch.inference_mode(): output = candidate["head"].forward_features(tensors["knownness_features"].float(), tensors["quality_features"].float(), tensors["padding_mask"].bool(), tensors["profile_features"].float())
        path = candidate["residual_path"]
    return {"payload": payload, "known_scores": torch.sigmoid(output["knownness_logits"])[rows, lengths - 1].numpy(), "quality_scores": torch.sigmoid(output["quality_logits"])[rows, lengths - 1].numpy(), "head": artifact(context.project_root, path), "composition": candidate["composition"]}


def calibrate(context: V28Context) -> dict[str, Any]:
    tasks: dict[str, Any] = {}
    for task in (V28_L1_TASK, V28_L2_TASK):
        scored = _scores(context, task); payload = scored["payload"]; tensors = payload["tensors"]; labels = [str(value) for value in payload["label_names"]]; lengths = tensors["lengths"].long(); rows = torch.arange(lengths.shape[0]); logits = tensors["sequence_logits"][rows, lengths - 1].numpy(); l1 = task == V28_L1_TASK; names = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE") if l1 else V28_L2_CLASSES
        known_indices = [index for index, label in enumerate(labels) if (label in {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget"} if l1 else label in V28_L2_CLASSES)]; known_ids = [0 if labels[index] != "ClutterTarget" else 1 for index in known_indices] if l1 else [names.index(labels[index]) for index in known_indices]; index_array = np.asarray(known_indices, dtype=np.int64); label_array = np.asarray(known_ids, dtype=np.int64); temperature = _temperature(logits[index_array], label_array, context); conformal = _conformal(_probabilities(logits[index_array], temperature["temperature"]), label_array, names, float(context.config.calibration["conformal_error_rate"]))
        candidate = load_final_l1(context) if l1 else load_final_l2(context)
        with torch.inference_mode():
            if l1:
                output = candidate["head"].forward_features(
                    tensors["knownness_features"].float(),
                    tensors["quality_features"].float(),
                    tensors["padding_mask"].bool(),
                )
            else:
                output = candidate["head"].forward_features(
                    tensors["knownness_features"].float(),
                    tensors["quality_features"].float(),
                    tensors["padding_mask"].bool(),
                    tensors["profile_features"].float(),
                )
        known_scores = torch.sigmoid(output["knownness_logits"])[rows, lengths - 1].numpy(); quality_scores = torch.sigmoid(output["quality_logits"])[rows, lengths - 1].numpy(); knownness = _select_knownness(task, {"labels": labels, "known_scores": known_scores}, context); facts = tensors["quality_features"].float(); qtargets = _quality_targets(task, facts[..., 38:], facts[..., 19:38], tensors["padding_mask"].bool(), context)[rows, lengths - 1].numpy().astype(bool); quality = _select_quality({"quality_scores": quality_scores, "quality_targets": qtargets}, context); scorer = build_scorer_descriptor(class_names=names, conformal=conformal, temperature=temperature) if not l1 else None
        facts = tensors["quality_features"].float(); qtargets = _quality_targets(task, facts[..., 38:], facts[..., 19:38], tensors["padding_mask"].bool(), context)[rows, lengths - 1].numpy().astype(bool); quality = _select_quality({"quality_scores": quality_scores, "quality_targets": qtargets}, context); scorer = build_scorer_descriptor(class_names=names, conformal=conformal, temperature=temperature) if not l1 else None
        tasks[task] = {"task": task, "head": scored["head"], "head_composition": scored["composition"], "representation": {"cache_sha256": payload["cache_sha256"], "parent_count": len(labels), "parent_ids_sha256": payload["parent_ids_sha256"], "label_counts": payload.get("label_counts")}, "temperature": temperature, "conformal": conformal, "knownness": knownness, "quality": {"status": "diagnostic_only", "learned_threshold": quality}, "scorer": scorer, "support": {"parent_count": len(labels), "known_parent_count": len(known_indices), "unknown_parent_count": sum(label == "UnknownTarget" for label in labels), "clutter_parent_count": sum(label == "ClutterTarget" for label in labels)}}
    unknown_support = min(int(tasks[V28_L1_TASK]["support"]["unknown_parent_count"]), int(tasks[V28_L2_TASK]["support"]["unknown_parent_count"])); minimum = int(context.config.calibration["minimum_unknown_support_for_formal"]); reasons = ["research_only", "previous_audit_pool_already_exposed", "new_unexposed_post_selection_audit_required"]; reasons += ["unknown_support_below_formal_minimum"] if unknown_support < minimum else []
    result = {"schema_version": "RADAR-LN3-V28-final-research-calibration-v1", "experiment_id": "V28-final-research-l3", "status": "internal_exploratory", "research_only": True, "deployment_eligible": False, "status_reasons": reasons, "system_manifest_sha256": context.manifest_sha256, "calibration_role": "validation_split_only", "unknown_support": unknown_support, "minimum_unknown_support_for_formal": minimum, "calibration_access_count": 2, "post_selection_audit_access_count": 0, "audit_used_for_fitting": False, "tasks": tasks}; path = _path(context, CALIBRATION); atomic_write_text(path, json_text({**result, "payload_sha256": canonical_sha256(result)})); return {"path": relative_project_path(context.project_root, path), "sha256": sha256_file(path), "payload": result}


def verify_scorer(context: V28Context) -> dict[str, Any]:
    calibration = _read_json(_path(context, CALIBRATION), "final calibration")
    return {"status": "verified", "research_only": True, "scorer": validate_scorer_descriptor(calibration["tasks"][V28_L2_TASK]["scorer"])}


def _verify_final_registry(context: V28Context, calibration: Mapping[str, Any]) -> dict[str, Any]:
    path = _path(context, REGISTRY)
    registry = _read_json(path, "final research registry")
    declared = registry.get("registry_sha256")
    body = {key: value for key, value in registry.items() if key != "registry_sha256"}
    if declared != canonical_sha256(body):
        raise ValueError("final research registry digest differs")
    if registry.get("status") != "internal_exploratory" or registry.get("research_only") is not True:
        raise ValueError("final research registry is not marked exploratory")
    if registry.get("deployment_eligible") is not False:
        raise ValueError("final research registry must remain unavailable for deployment")
    if registry.get("system_manifest_sha256") != calibration.get("system_manifest_sha256"):
        raise ValueError("final registry and calibration manifest bindings differ")
    calibration_binding = registry.get("calibration_report") or {}
    calibration_path = _path(context, CALIBRATION)
    calibration_payload_sha256 = json.loads(
        calibration_path.read_text(encoding="utf-8")
    ).get("payload_sha256")
    if (
        calibration_binding.get("file_sha256") != sha256_file(calibration_path)
        or calibration_binding.get("payload_sha256") != calibration_payload_sha256
    ):
        raise ValueError("final registry calibration report binding differs")
    if registry.get("final_heads") != {
        task: calibration["tasks"][task]["head"] for task in (V28_L1_TASK, V28_L2_TASK)
    }:
        raise ValueError("final registry head bindings differ from calibration")
    compositions = registry.get("head_compositions")
    if (
        not isinstance(compositions, Mapping)
        or registry.get("head_compositions_sha256") != canonical_sha256(compositions)
        or any(
            calibration["tasks"][task].get("head_composition") != compositions.get(task)
            for task in (V28_L1_TASK, V28_L2_TASK)
        )
    ):
        raise ValueError("final registry head composition binding differs")
    scorer = calibration["tasks"][V28_L2_TASK].get("scorer")
    registry_scorer = (registry.get("scorers") or {}).get(V28_L2_TASK)
    if (
        not isinstance(scorer, Mapping)
        or dict(registry_scorer or {}) != dict(scorer)
        or registry.get("scorer_bindings_sha256")
        != canonical_sha256(registry.get("scorers"))
    ):
        raise ValueError("final registry L2 scorer binding differs")
    validate_scorer_descriptor(scorer)
    replay_binding = registry.get("replay_report") or {}
    replay_path = _path(context, REPLAY)
    replay_payload = json.loads(replay_path.read_text(encoding="utf-8"))
    if (
        replay_binding.get("file_sha256") != sha256_file(replay_path)
        or replay_binding.get("payload_sha256") != replay_payload.get("payload_sha256")
    ):
        raise ValueError("final registry replay report binding differs")
    minimum_unknown = int(context.config.calibration["minimum_unknown_support_for_formal"])
    if int(registry.get("calibration_role", {}).get("minimum_unknown_support_for_formal", -1)) != minimum_unknown:
        raise ValueError("final registry minimum Unknown support differs from configuration")
    if int(calibration.get("unknown_support", -1)) < minimum_unknown:
        raise ValueError("final calibration Unknown support is below the configured minimum")
    audit = registry.get("post_selection_audit") or {}
    if audit.get("status") != "new_unexposed_pool_required" or audit.get("threshold_fitting_allowed") is not False:
        raise ValueError("final research registry audit boundary differs")
    for task in (V28_L1_TASK, V28_L2_TASK):
        representation = _cache(context, task)
        report_representation = calibration["tasks"][task].get("representation") or {}
        if (
            representation.get("task") != task
            or representation.get("role") != "calibration"
            or representation.get("cache_sha256") != report_representation.get("cache_sha256")
            or representation.get("parent_ids_sha256") != report_representation.get("parent_ids_sha256")
        ):
            raise ValueError(f"final calibration cache binding differs: {task}")
        support = calibration["tasks"][task].get("support") or {}
        if (
            len(representation.get("parent_ids", ())) != int(support.get("parent_count", -1))
            or len(representation.get("label_names", ())) != int(support.get("parent_count", -1))
            or sum(int(value) for value in (representation.get("label_counts") or {}).values())
            != int(support.get("parent_count", -1))
        ):
            raise ValueError(f"final calibration cache support differs: {task}")
    return {**registry, "registry_sha256": declared, "registry_file_sha256": sha256_file(path)}


def _final_quality(
    context: V28Context,
    *,
    task: str,
    tensors: Mapping[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    lengths = tensors["lengths"].long()
    rows = torch.arange(lengths.shape[0])
    facts = tensors["quality_features"].float()
    padding = tensors["padding_mask"].bool()
    eligible = _quality_targets(
        task,
        facts[..., 38:],
        facts[..., 19:38],
        padding,
        context,
    )[rows, lengths - 1].numpy().astype(bool)
    model = load_final_l1(context) if task == V28_L1_TASK else load_final_l2(context)
    with torch.inference_mode():
        if task == V28_L1_TASK:
            output = model["head"].forward_features(
                tensors["knownness_features"].float(), facts, padding
            )
        else:
            output = model["head"].forward_features(
                tensors["knownness_features"].float(),
                facts,
                padding,
                tensors["profile_features"].float(),
            )
    scores = torch.sigmoid(output["quality_logits"])[rows, lengths - 1].numpy()
    if not np.isfinite(scores).all():
        raise FloatingPointError(f"final quality scores contain non-finite values: {task}")
    return eligible, scores


def replay_calibration(context: V28Context) -> dict[str, Any]:
    """Reproduce the research L3 report on its calibration fitting role."""

    calibration_path = _path(context, CALIBRATION)
    calibration = _read_json(calibration_path, "final calibration")
    registry = _verify_final_registry(context, calibration)
    l1_task = calibration["tasks"][V28_L1_TASK]
    l2_task = calibration["tasks"][V28_L2_TASK]
    l1_payload = _cache(context, V28_L1_TASK)
    l2_payload = _cache(context, V28_L2_TASK)
    cache_bindings: dict[str, Any] = {}
    for task, payload in ((V28_L1_TASK, l1_payload), (V28_L2_TASK, l2_payload)):
        cache_path = final_path(
            context.project_root,
            CALIBRATION_ROOT / f"calibration-{task}-candidate-v1.pt",
        )
        cache_bindings[task] = {
            "path": relative_project_path(context.project_root, cache_path),
            "file_sha256": sha256_file(cache_path),
            "cache_sha256": payload["cache_sha256"],
            "role": payload["role"],
            "parent_count": len(payload["parent_ids"]),
            "parent_ids_sha256": payload["parent_ids_sha256"],
            "classifier_endpoint": payload.get("endpoint_binding"),
        }
    l1_tensors = l1_payload["tensors"]
    l2_tensors = l2_payload["tensors"]
    l1_ids = [str(value) for value in l1_payload["parent_ids"]]
    l2_ids = [str(value) for value in l2_payload["parent_ids"]]
    l1_labels = [str(value) for value in l1_payload["label_names"]]
    l2_labels = [str(value) for value in l2_payload["label_names"]]
    if len(set(l1_ids)) != len(l1_ids) or len(set(l2_ids)) != len(l2_ids):
        raise ValueError("final calibration cache contains duplicate parent ids")
    if set(l2_ids) - set(l1_ids):
        raise ValueError("final L2 calibration parents are absent from L1 cache")
    l1_by_id = dict(zip(l1_ids, l1_labels, strict=True))
    if any(l1_by_id[parent_id] != label for parent_id, label in zip(l2_ids, l2_labels, strict=True)):
        raise ValueError("final L1/L2 calibration parent labels differ")
    if any(label == "ClutterTarget" for label in l2_labels):
        raise ValueError("final L2 calibration cache contains Clutter")
    if tuple(l1_task["conformal"].get("support_by_class", {}).keys()) != V28_L1_CLASSES:
        raise ValueError("final L1 calibration class contract differs")
    if tuple(l2_task["conformal"].get("support_by_class", {}).keys()) != V28_L2_CLASSES:
        raise ValueError("final L2 calibration class contract differs")

    l1_model = load_final_l1(context)
    l2_model = load_final_l2(context)
    l1_lengths = l1_tensors["lengths"].long()
    l2_lengths = l2_tensors["lengths"].long()
    l1_rows = torch.arange(len(l1_ids))
    l2_rows = torch.arange(len(l2_ids))
    with torch.inference_mode():
        l1_output = l1_model["head"].forward_features(
            l1_tensors["knownness_features"].float(),
            l1_tensors["quality_features"].float(),
            l1_tensors["padding_mask"].bool(),
        )
        l2_output = l2_model["head"].forward_features(
            l2_tensors["knownness_features"].float(),
            l2_tensors["quality_features"].float(),
            l2_tensors["padding_mask"].bool(),
            l2_tensors["profile_features"].float(),
        )
    l1_logits = l1_tensors["sequence_logits"][l1_rows, l1_lengths - 1].numpy()
    l2_logits = l2_tensors["sequence_logits"][l2_rows, l2_lengths - 1].numpy()
    l1_temperature = float(l1_task["temperature"]["temperature"])
    l2_temperature = float(l2_task["temperature"]["temperature"])
    l1_probabilities = _probabilities(l1_logits, l1_temperature)
    l2_probabilities = _probabilities(l2_logits, l2_temperature)
    l1_knownness = torch.sigmoid(l1_output["knownness_logits"][l1_rows, l1_lengths - 1]).numpy()
    l2_knownness = torch.sigmoid(l2_output["knownness_logits"][l2_rows, l2_lengths - 1]).numpy()
    l1_quality, l1_quality_scores = _final_quality(
        context, task=V28_L1_TASK, tensors=l1_tensors
    )
    l2_quality, l2_quality_scores = _final_quality(
        context, task=V28_L2_TASK, tensors=l2_tensors
    )
    scorer = l2_task["scorer"]
    l2_score_result = score_class_conditional(l2_probabilities, scorer)
    l2_position = {parent_id: index for index, parent_id in enumerate(l2_ids)}
    l1_threshold = float(l1_task["knownness"]["selected"]["threshold"])
    l2_threshold = float(l2_task["knownness"]["selected"]["threshold"])
    l1_floors = l1_task["conformal"]["candidate_probability_floor_by_class"]
    records: list[dict[str, Any]] = []

    for index, (parent_id, label) in enumerate(zip(l1_ids, l1_labels, strict=True)):
        l1_candidates = _candidate(
            l1_probabilities[index], V28_L1_CLASSES, l1_floors
        )
        l1_prediction = V28_L1_CLASSES[int(l1_probabilities[index].argmax())]
        l2_index = l2_position.get(parent_id)
        if l2_index is None:
            l2_prediction = None
            l2_candidates: list[str] = []
            l2_knownness_value: float | None = None
            l2_quality_value: bool | None = None
            l2_quality_score: float | None = None
            l2_score: dict[str, Any] | None = None
        else:
            l2_prediction = str(l2_score_result["predictions"][l2_index])
            l2_candidates = list(l2_score_result["candidate_sets"][l2_index])
            l2_knownness_value = float(l2_knownness[l2_index])
            l2_quality_value = bool(l2_quality[l2_index])
            l2_quality_score = float(l2_quality_scores[l2_index])
            l2_score = serialize_score(l2_score_result, l2_index)
        l1_target = l1_prediction == "KNOWN_TARGET_EVIDENCE"
        l1_clutter = l1_prediction == "KNOWN_CLUTTER_EVIDENCE"
        decision = resolve_research_action(
            registry=registry,
            l1_candidate_set=l1_candidates,
            l1_prediction=l1_prediction,
            l1_knownness=bool(l1_knownness[index] >= l1_threshold),
            l1_quality_eligible=bool(l1_quality[index]),
            l1_target=l1_target,
            l1_clutter=l1_clutter,
            l2_candidate_set=l2_candidates,
            l2_prediction=l2_prediction,
            l2_knownness=(None if l2_knownness_value is None else bool(l2_knownness_value >= l2_threshold)),
            l2_quality_eligible=l2_quality_value,
            streaming=False,
        )
        records.append(
            {
                "parent_sample_id": parent_id,
                "label_name": label,
                "action": decision.action,
                "reason_code": decision.reason_code,
                "decision_details": decision.details,
                "l1_prediction": l1_prediction,
                "l2_prediction": l2_prediction,
                "l1_candidates": l1_candidates,
                "l2_candidates": l2_candidates,
                "l1_knownness": float(l1_knownness[index]),
                "l2_knownness": l2_knownness_value,
                "l1_quality_eligible": bool(l1_quality[index]),
                "l2_quality_eligible": l2_quality_value,
                "l1_quality_score": float(l1_quality_scores[index]),
                "l2_quality_score": l2_quality_score,
                "l2_scorer": l2_score,
            }
        )

    metrics = _action_metrics(records)
    previous_path = _path(context, REPLAY)
    calibration_file_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    previous = json.loads(previous_path.read_text(encoding="utf-8"))
    previous_metrics = previous.get("action_metrics") or {}
    previous_matrix = previous_metrics.get("confusion_matrix_rows_true_columns_action")
    reproduced = (
        metrics["confusion_matrix_rows_true_columns_action"] == previous_matrix
        and metrics["support"] == previous_matrix_support(previous_matrix)
        and metrics["unknown_rejection"]["rejected"] == previous_metrics.get("unknown_rejection", {}).get("rejected")
        and metrics["unknown_rejection"]["error_acceptance"] == previous_metrics.get("unknown_rejection", {}).get("error_acceptance")
        and metrics["unknown_rejection"]["pending"] == previous_metrics.get("unknown_rejection", {}).get("pending")
        and metrics["clutter_acceptance"]["accepted"] == previous_metrics.get("clutter_acceptance", {}).get("accepted")
        and metrics["clutter_acceptance"]["misrouted"] == previous_metrics.get("clutter_acceptance", {}).get("misrouted")
        and metrics["known_target_acceptance"]["accepted"] == previous_metrics.get("known_target_acceptance", {}).get("accepted")
        and metrics["known_target_acceptance"]["pending"] == previous_metrics.get("known_target_acceptance", {}).get("pending")
        and metrics["known_target_acceptance"]["wrong_action"] == previous_metrics.get("known_target_acceptance", {}).get("wrong_action")
    )
    report = {
        "schema_version": "RADAR-LN3-V28-final-research-replay-reproduction-v1",
        "experiment_id": "V28-final-research-l3",
        "status": "internal_exploratory_reproduction",
        "research_only": True,
        "deployment_eligible": False,
        "system_manifest_sha256": calibration.get("system_manifest_sha256"),
        "replay_code_sha256": sha256_file(Path(__file__).resolve()),
        "registry_sha256": registry["registry_sha256"],
        "final_head_compositions": registry["head_compositions"],
        "representation_cache_bindings": cache_bindings,
        "temperature_calibration": {
            task: calibration["tasks"][task]["temperature"]
            for task in (V28_L1_TASK, V28_L2_TASK)
        },
        "conformal_calibration": {
            task: calibration["tasks"][task]["conformal"]
            for task in (V28_L1_TASK, V28_L2_TASK)
        },
        "knownness_thresholds": {
            task: calibration["tasks"][task]["knownness"]
            for task in (V28_L1_TASK, V28_L2_TASK)
        },
        "quality_policy": {
            task: calibration["tasks"][task]["quality"]
            for task in (V28_L1_TASK, V28_L2_TASK)
        },
        "l2_scorer": scorer,
        "calibration_report": {
            "path": relative_project_path(context.project_root, calibration_path),
            "sha256": sha256_file(calibration_path),
            "payload_sha256": calibration_file_payload.get("payload_sha256"),
        },
        "evaluation_scope": {
            "role": "calibration",
            "calibration_is_fitting_role": True,
            "test_payload_access_count": 0,
            "post_selection_audit_access_count": 0,
            "audit_used": False,
            "independent_validation": False,
        },
        "parent_count": len(records),
        "previous_report": {
            "path": relative_project_path(context.project_root, previous_path),
            "sha256": sha256_file(previous_path),
            "payload_sha256": previous.get("payload_sha256"),
            "action_metrics_match": reproduced,
        },
        "action_metrics": metrics,
        "records": records,
    }
    output_path = _path(context, REPLAY_REPRODUCTION)
    atomic_write_text(output_path, json_text({**report, "payload_sha256": canonical_sha256(report)}))
    return {
        "path": relative_project_path(context.project_root, output_path),
        "sha256": sha256_file(output_path),
        "action_metrics_match_previous_report": reproduced,
        "parent_count": len(records),
        "registry_sha256": registry["registry_sha256"],
        "evaluation_role": "calibration_internal_reproduction",
    }


def replay_without_abstain(context: V28Context) -> dict[str, Any]:
    """Run a diagnostic action replay with Pending gates disabled.

    Deterministic quality/maturity and Conformal candidate-set abstention are
    bypassed. L1/L2 knownness thresholds remain active; samples below those
    thresholds still receive UNKNOWN_REJECTED.
    """

    calibration_path = _path(context, CALIBRATION)
    calibration = _read_json(calibration_path, "final calibration")
    # This counterfactual remaps already frozen parent outputs. Verify the
    # sealed report and endpoint chain without re-hashing multi-gigabyte tensor
    # caches that are not read by this action-only step.
    registry_path = _path(context, REGISTRY)
    registry = _read_json(registry_path, "final research registry")
    declared_registry_hash = registry.get("registry_sha256")
    registry_body = {
        key: value for key, value in registry.items() if key != "registry_sha256"
    }
    if declared_registry_hash != canonical_sha256(registry_body):
        raise ValueError("final research registry digest differs")
    if (
        registry.get("status") != "internal_exploratory"
        or registry.get("research_only") is not True
        or registry.get("deployment_eligible") is not False
        or registry.get("system_manifest_sha256")
        != calibration.get("system_manifest_sha256")
    ):
        raise ValueError("final research registry identity differs")
    calibration_file_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibration_binding = registry.get("calibration_report") or {}
    if (
        calibration_binding.get("file_sha256") != sha256_file(calibration_path)
        or calibration_binding.get("payload_sha256")
        != calibration_file_payload.get("payload_sha256")
    ):
        raise ValueError("final calibration report binding differs")
    compositions = registry.get("head_compositions")
    if (
        not isinstance(compositions, Mapping)
        or registry.get("head_compositions_sha256") != canonical_sha256(compositions)
    ):
        raise ValueError("final head composition digest differs")
    for task in (V28_L1_TASK, V28_L2_TASK):
        if calibration["tasks"][task].get("head_composition") != compositions.get(task):
            raise ValueError(f"final head composition binding differs: {task}")
        head_binding = registry["final_heads"][task]
        head_path = resolve_project_path(
            context.project_root, str(head_binding["path"]), must_exist=True
        )
        if sha256_file(head_path) != head_binding.get("sha256"):
            raise ValueError(f"final head hash differs: {task}")
    scorer = calibration["tasks"][V28_L2_TASK].get("scorer")
    registered_scorer = (registry.get("scorers") or {}).get(V28_L2_TASK)
    if (
        not isinstance(scorer, Mapping)
        or dict(registered_scorer or {}) != dict(scorer)
        or registry.get("scorer_bindings_sha256")
        != canonical_sha256(registry.get("scorers"))
    ):
        raise ValueError("final scorer binding differs")
    validate_scorer_descriptor(scorer)
    source_path = _path(context, REPLAY_REPRODUCTION)
    source = _read_json(source_path, "verified calibration replay")
    source_file = json.loads(source_path.read_text(encoding="utf-8"))
    source_file = json.loads(source_path.read_text(encoding="utf-8"))
    source_origin = source.get("previous_report") or {}
    origin_path = _path(context, REPLAY)
    origin_file = json.loads(origin_path.read_text(encoding="utf-8"))
    if (
        source_origin.get("path") != relative_project_path(context.project_root, origin_path)
        or source_origin.get("sha256") != sha256_file(origin_path)
        or source_origin.get("payload_sha256") != origin_file.get("payload_sha256")
        or source_origin.get("action_metrics_match") is not True
        or canonical_sha256(source) != source_file.get("payload_sha256")
    ):
        raise ValueError("verified replay source provenance differs")
    if source.get("registry_sha256") != registry.get("registry_sha256"):
        raise ValueError("no-abstain replay source uses a different frozen registry")
    scope = source.get("evaluation_scope") or {}
    if scope.get("role") != "calibration" or scope.get("independent_validation") is not False:
        raise ValueError("no-abstain replay source is outside the calibration diagnostic scope")

    l1_threshold = float(
        calibration["tasks"][V28_L1_TASK]["knownness"]["selected"]["threshold"]
    )
    l2_threshold = float(
        calibration["tasks"][V28_L2_TASK]["knownness"]["selected"]["threshold"]
    )
    records: list[dict[str, Any]] = []
    for source_record in source.get("records", []):
        record = dict(source_record)
        l1_prediction = str(record["l1_prediction"])
        l1_knownness = float(record["l1_knownness"])
        if l1_knownness < l1_threshold:
            action = "UNKNOWN_REJECTED"
            reason = "L1_KNOWNNESS_BELOW_THRESHOLD"
        elif l1_prediction == "KNOWN_CLUTTER_EVIDENCE":
            action = "Clutter"
            reason = "L1_CLUTTER_ACCEPTED_NO_ABSTAIN"
        elif l1_prediction == "KNOWN_TARGET_EVIDENCE":
            l2_knownness = record.get("l2_knownness")
            l2_prediction = record.get("l2_prediction")
            if l2_knownness is None or l2_prediction not in V28_L2_CLASSES:
                raise ValueError(
                    f"no-abstain target route lacks L2 evidence: {record.get('parent_sample_id')}"
                )
            if float(l2_knownness) < l2_threshold:
                action = "UNKNOWN_REJECTED"
                reason = "L2_KNOWNNESS_BELOW_THRESHOLD"
            else:
                action = str(l2_prediction)
                reason = "DBB_ARGMAX_ACCEPTED_NO_ABSTAIN"
        else:
            raise ValueError(
                f"no-abstain replay has an invalid L1 prediction: {record.get('l1_prediction')}"
            )
        record["action"] = action
        record["reason_code"] = reason
        record["abstention_policy"] = {
            "deterministic_quality_gate_applied": False,
            "conformal_candidate_gate_applied": False,
            "knownness_gate_applied": True,
        }
        records.append(record)

    metrics = _action_metrics(records)
    report = {
        "schema_version": "RADAR-LN3-V28-no-abstain-counterfactual-v1",
        "experiment_id": "V28-final-research-l3",
        "status": "internal_exploratory_counterfactual",
        "research_only": True,
        "deployment_eligible": False,
        "system_manifest_sha256": calibration.get("system_manifest_sha256"),
        "replay_code_sha256": sha256_file(Path(__file__).resolve()),
        "registry_sha256": registry["registry_sha256"],
        "source_replay": {
            "path": relative_project_path(context.project_root, source_path),
            "file_sha256": sha256_file(source_path),
            "payload_sha256": source_file.get("payload_sha256"),
        },
        "counterfactual": {
            "variable": "disable deterministic quality/maturity and conformal abstention",
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
        "records": records,
    }
    output_path = _path(context, NO_ABSTAIN_REPLAY)
    atomic_write_text(output_path, json_text({**report, "payload_sha256": canonical_sha256(report)}))
    return {
        "path": relative_project_path(context.project_root, output_path),
        "sha256": sha256_file(output_path),
        "parent_count": len(records),
        "action_metrics": metrics,
        "evaluation_role": "calibration_counterfactual_only",
    }


def previous_matrix_support(matrix: Any) -> int:
    if not isinstance(matrix, list) or any(not isinstance(row, list) for row in matrix):
        return -1
    return sum(sum(int(value) for value in row) for row in matrix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="V28 final frozen research L3")
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--manifest-sha256", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("verify-scorer")
    sub.add_parser("replay-calibration")
    sub.add_parser("replay-no-abstain")
    args = parser.parse_args(argv)
    if args.command == "calibrate":
        context = load_context(args.project_root.resolve(), None)
        result = calibrate(context)
    elif args.command == "verify-scorer":
        context = load_context(args.project_root.resolve(), None)
        result = verify_scorer(context)
    elif args.command == "replay-no-abstain":
        context = _research_context(
            args.project_root.resolve(),
            manifest_sha256=args.manifest_sha256,
        )
        result = replay_without_abstain(context)
    else:
        context = _research_context(
            args.project_root.resolve(),
            manifest_sha256=args.manifest_sha256,
        )
        result = replay_calibration(context)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__": raise SystemExit(main())
