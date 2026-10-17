"""Paired known-class evaluation for V27d-r9."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.runtime import relative_project_path, sha256_file
from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records

from .checkpoint import load_checkpoint
from .features import load_profile_cache, validate_profile_cache_binding
from .model import build_model
from .protocol import R9_CLASS_NAMES, R9_REPORT_SCHEMA_VERSION, R9_SEEDS, artifact_path, load_context, write_sealed_json
from .training import _cache_path, _inputs, _device


def _records(payload: Mapping[str, Any], logits: torch.Tensor, *, arm: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tensors = payload["tensors"]
    labels = tensors["labels"].numpy().astype(np.int64)
    lengths = tensors["lengths"].numpy().astype(np.int64)
    padding = tensors["padding_mask"]
    values = logits.detach().cpu().numpy()
    parent_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for row, (parent_id, label, length) in enumerate(zip(payload["parent_sample_ids"], labels.tolist(), lengths.tolist(), strict=True)):
        length = int(length)
        probabilities = _softmax(values[row, :length])
        final = probabilities[-1]
        parent_rows.append(
            {
                "parent_sample_id": str(parent_id),
                "true_id": int(label),
                "pred_id": int(np.argmax(final)),
                "probabilities": [float(value) for value in final.tolist()],
                "logits": [float(value) for value in values[row, length - 1].tolist()],
                "event_count": length,
                "arm": arm,
            }
        )
        for event in range(length):
            event_rows.append(
                {
                    "parent_sample_id": str(parent_id),
                    "true_id": int(label),
                    "pred_id": int(np.argmax(probabilities[event])),
                    "probabilities": [float(value) for value in probabilities[event].tolist()],
                    "logits": [float(value) for value in values[row, event].tolist()],
                    "event_index": event,
                    "window_length": event + 1,
                    "parent_length": length,
                    "profile_valid": bool(tensors["profile_valid"][row, event]),
                    "profile_frame_count": int(tensors["profile_frame_count"][row, event]),
                    "arm": arm,
                }
            )
    if bool(padding.any()) and np.any(values[padding.numpy()] != 0.0):
        raise ValueError("V27d-r9 evaluation received non-zero padded logits")
    return parent_rows, event_rows


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    denominator = values.sum(axis=-1, keepdims=True)
    if not np.isfinite(denominator).all() or np.any(denominator <= 0.0):
        raise FloatingPointError("V27d-r9 softmax denominator is invalid")
    return values / denominator


def _metrics(parents: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parent_metrics = summarize_records(parents, class_names=R9_CLASS_NAMES, scope="parent_final")
    eligible = [
        row
        for row in events
        if int(row["window_length"]) >= 3 and int(row["window_length"]) < int(row["parent_length"])
    ]
    prefixes = parent_balanced_prefix_records(eligible, minimum_events=3, class_names=R9_CLASS_NAMES)
    return {
        "parent_metrics": parent_metrics,
        "parent_balanced_eligible_prefix_metrics": summarize_records(prefixes, class_names=R9_CLASS_NAMES, scope="parent_balanced_eligible_prefix") if prefixes else None,
        "eligible_prefix_records": prefixes,
    }


def _paired(control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right):
        raise ValueError("V27d-r9 paired parent sets differ")
    transitions: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if int(a["true_id"]) != int(b["true_id"]):
            raise ValueError("V27d-r9 paired labels differ")
        a_correct = int(a["pred_id"]) == int(a["true_id"])
        b_correct = int(b["pred_id"]) == int(b["true_id"])
        transitions[f"{'correct' if a_correct else 'wrong'}_to_{'correct' if b_correct else 'wrong'}"] += 1
        if int(a["pred_id"]) != int(b["pred_id"]):
            changed.append({"parent_sample_id": parent_id, "true_id": int(a["true_id"]), "control_pred_id": int(a["pred_id"]), "treatment_pred_id": int(b["pred_id"]), "control_correct": a_correct, "treatment_correct": b_correct})
    return {
        "parent_count": len(left),
        "control_correct_count": sum(int(int(row["pred_id"]) == int(row["true_id"])) for row in left.values()),
        "treatment_correct_count": sum(int(int(row["pred_id"]) == int(row["true_id"])) for row in right.values()),
        "correctness_transitions": dict(sorted(transitions.items())),
        "changed_predictions": changed,
    }


def _residual_diagnostics(payload: Mapping[str, Any], output: Mapping[str, Any]) -> dict[str, Any]:
    padding = payload["tensors"]["padding_mask"]
    valid = output["profile_valid_sequence"].detach().cpu()
    correction = output["correction_logits"].detach().cpu()
    increments = output["pairwise_delta_increment"].detach().cpu()
    active = ~padding
    invalid = active & ~valid
    return {
        "active_event_count": int(active.sum()),
        "valid_profile_event_count": int(valid.sum()),
        "invalid_profile_event_count": int(invalid.sum()),
        "invalid_profile_correction_nonzero_count": int((correction[invalid].abs().amax(dim=-1) > 0.0).sum()) if bool(invalid.any()) else 0,
        "invalid_profile_increment_nonzero_count": int((increments[invalid].abs() > 0.0).sum()) if bool(invalid.any()) else 0,
        "correction_zero_sum_max": float(correction[active].sum(dim=-1).abs().max()) if bool(active.any()) else 0.0,
        "balloon_correction_nonzero_count": int((correction[..., 2][active].abs() > 0.0).sum()) if bool(active.any()) else 0,
        "correction_absolute_max": float(correction[active].abs().max()) if bool(active.any()) else 0.0,
    }


def _load_model(root: Path, config: Mapping[str, Any], manifest: Mapping[str, Any], seed: int) -> tuple[Any, dict[str, Any], dict[str, str]]:
    caches = {split: load_profile_cache(_cache_path(root, config, split)) for split in ("train", "val")}
    for split, payload in caches.items():
        validate_profile_cache_binding(payload, root=root, config=config, split=split)
    cache_hashes = {split: payload["cache_sha256"] for split, payload in caches.items()}
    model = build_model(config["model"], seed=seed)
    checkpoint_path = artifact_path(root, config["artifacts"]["checkpoint_template"].format(seed=seed), must_exist=True)
    checkpoint = load_checkpoint(checkpoint_path, root=root, config=config, manifest=manifest, model=model, seed=seed, cache_hashes=cache_hashes)
    return model, checkpoint, cache_hashes


def evaluate_seed(project_root: str | Path, *, seed: int, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) not in R9_SEEDS:
        raise ValueError("V27d-r9 evaluation seed is not registered")
    model, checkpoint, cache_hashes = _load_model(root, config, manifest, seed)
    device = _device(config)
    model = model.to(device).eval()
    validation = load_profile_cache(_cache_path(root, config, "val"))
    validate_profile_cache_binding(validation, root=root, config=config, split="val")
    indices = torch.arange(len(validation["parent_sample_ids"]), dtype=torch.long)
    with torch.inference_mode():
        treatment_output = model(**_inputs(validation, indices, device))
    treatment_parents, treatment_events = _records(validation, treatment_output["sequence_logits"], arm="causal_doppler_shape")
    control_parents, control_events = _records(validation, validation["tensors"]["frozen_l2_logits"], arm="frozen_v27a_main_control")
    treatment_metrics = _metrics(treatment_parents, treatment_events)
    control_metrics = _metrics(control_parents, control_events)
    treatment_matrix = np.asarray(treatment_metrics["parent_metrics"]["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
    control_matrix = np.asarray(control_metrics["parent_metrics"]["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
    treatment_correct = int(np.trace(treatment_matrix))
    control_correct = int(np.trace(control_matrix))
    new_errors = int((control_matrix.diagonal() - treatment_matrix.diagonal()).clip(min=0).sum())
    report = {
        "schema_version": R9_REPORT_SCHEMA_VERSION,
        "experiment_id": "V27d-r9",
        "status": "paired_known_class_validation",
        "seed": int(seed),
        "manifest_sha256": manifest["manifest_sha256"],
        "direct_baseline": "V27a-r3-main-epoch-020",
        "arms": {"frozen_v27a_main_control": control_metrics, "causal_doppler_shape": treatment_metrics},
        "paired": _paired(control_parents, treatment_parents),
        "deltas": {
            "total_correct_delta": treatment_correct - control_correct,
            "new_error_count": new_errors,
            "drone_to_bird_delta": int(treatment_matrix[0, 1] - control_matrix[0, 1]),
            "bird_to_drone_delta": int(treatment_matrix[1, 0] - control_matrix[1, 0]),
            "bird_recall_delta": float(treatment_metrics["parent_metrics"]["per_class"][1]["recall"] - control_metrics["parent_metrics"]["per_class"][1]["recall"]),
        },
        "residual_diagnostics": _residual_diagnostics(validation, treatment_output),
        "checkpoint": {
            "path": relative_project_path(root, artifact_path(root, config["artifacts"]["checkpoint_template"].format(seed=seed))),
            "sha256": sha256_file(artifact_path(root, config["artifacts"]["checkpoint_template"].format(seed=seed))),
            "selected_epoch": int(checkpoint["epoch"]),
        },
        "scope": {
            "unknown_training_exposure": 0,
            "unknown_evaluation_exposure": 0,
            "quality": "not_in_scope",
            "knownness": "not_in_scope",
            "formal_l3_enabled": False,
            "test_payload_access_count": 0,
            "liquid_v2_backbone": "frozen",
        },
        "deployment_eligible": False,
        "records": {"control_parent": control_parents, "treatment_parent": treatment_parents},
    }
    path = artifact_path(root, config["artifacts"]["evaluation_template"].format(seed=seed))
    write_sealed_json(path, report)
    return {**report, "report_path": relative_project_path(root, path)}


def summarize_seeds(project_root: str | Path, *, config_path: str | Path | None = None) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    reports = [
        evaluate_seed(root, seed=seed, config_path=config_path)
        for seed in R9_SEEDS
    ]
    per_seed: dict[str, Any] = {}
    promotion = config["promotion"]
    for report in reports:
        treatment = report["arms"]["causal_doppler_shape"]["parent_metrics"]
        bird = treatment["per_class"][1]
        deltas = report["deltas"]
        passed = (
            float(bird["recall"]) >= float(promotion["minimum_bird_recall"])
            and int(deltas["new_error_count"]) <= int(promotion["maximum_new_errors"])
            and int(deltas["drone_to_bird_delta"]) < 0
            and int(deltas["total_correct_delta"]) >= int(promotion["minimum_total_correct_gain"])
        )
        per_seed[str(report["seed"])] = {"bird_recall": bird["recall"], "total_correct_delta": deltas["total_correct_delta"], "new_error_count": deltas["new_error_count"], "drone_to_bird_delta": deltas["drone_to_bird_delta"], "passed": passed}
    summary = {
        "schema_version": "RADAR-LN3-V27d-r9-two-seed-summary-v1",
        "experiment_id": "V27d-r9",
        "status": "completed_two_seed_known_class_evaluation",
        "direct_baseline": "V27a-r3-main-epoch-020",
        "manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(R9_SEEDS),
        "per_seed": per_seed,
        "promotion_standard": dict(promotion),
        "promotion_passed": bool(all(item["passed"] for item in per_seed.values())),
        "formal_l3_enabled": False,
        "unknown_support": 0,
        "deployment_eligible": False,
    }
    path = artifact_path(root, config["artifacts"]["summary_path"])
    write_sealed_json(path, summary)
    return {**summary, "summary_path": relative_project_path(root, path)}


__all__ = ["evaluate_seed", "summarize_seeds"]
