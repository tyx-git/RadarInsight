"""Paired known-class evaluation for V27d-r10."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.runtime import relative_project_path, sha256_file
from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records

from .checkpoint import load_checkpoint
from .features import load_descriptor_cache, validate_cache_binding
from .model import CompactCausalDopplerResidual, build_model
from .protocol import (
    R10_CLASS_NAMES,
    R10_REPORT_SCHEMA_VERSION,
    R10_SEEDS,
    artifact_path,
    load_context,
    read_sealed_json,
    write_sealed_json_if_identical,
)
from .training import _cache_path, _device, _inputs


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    denominator = values.sum(axis=-1, keepdims=True)
    if not np.isfinite(denominator).all() or bool((denominator <= 0.0).any()):
        raise FloatingPointError("V27d-r10 softmax denominator is invalid")
    result = values / denominator
    if not np.isfinite(result).all():
        raise FloatingPointError("V27d-r10 probabilities are non-finite")
    return result


def _records(
    payload: Mapping[str, Any],
    logits: torch.Tensor,
    *,
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tensors = payload["tensors"]
    values = logits.detach().cpu().numpy()
    padding = tensors["padding_mask"].detach().cpu().numpy()
    labels = tensors["labels"].detach().cpu().numpy().astype(np.int64)
    lengths = tensors["lengths"].detach().cpu().numpy().astype(np.int64)
    if values.shape != (*padding.shape, len(R10_CLASS_NAMES)):
        raise ValueError("V27d-r10 evaluation logits shape differs")
    if np.any(values[padding] != 0.0):
        raise ValueError("V27d-r10 evaluation padded logits are non-zero")
    probabilities = _softmax(values.reshape(-1, len(R10_CLASS_NAMES))).reshape(values.shape)
    parent_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for row, (parent_id, label, length) in enumerate(
        zip(payload["parent_sample_ids"], labels.tolist(), lengths.tolist(), strict=True)
    ):
        length = int(length)
        if length <= 0 or length > values.shape[1]:
            raise ValueError("V27d-r10 evaluation parent length is out of bounds")
        final = probabilities[row, length - 1]
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
                    "pred_id": int(np.argmax(probabilities[row, event])),
                    "probabilities": [float(value) for value in probabilities[row, event].tolist()],
                    "logits": [float(value) for value in values[row, event].tolist()],
                    "event_index": event,
                    "window_length": event + 1,
                    "parent_length": length,
                    "descriptor_valid": bool(tensors["descriptor_valid"][row, event]),
                    "descriptor_frame_count": int(tensors["descriptor_frame_count"][row, event]),
                    "arm": arm,
                }
            )
    return parent_rows, event_rows


def _metrics(
    parents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    parent_metrics = summarize_records(
        parents,
        class_names=R10_CLASS_NAMES,
        scope="parent_final",
    )
    eligible_events = [
        row
        for row in events
        if int(row["window_length"]) >= 3
        and int(row["window_length"]) < int(row["parent_length"])
    ]
    prefixes = parent_balanced_prefix_records(
        eligible_events,
        minimum_events=3,
        class_names=R10_CLASS_NAMES,
    )
    return {
        "parent_metrics": parent_metrics,
        "parent_balanced_eligible_prefix_metrics": (
            summarize_records(
                prefixes,
                class_names=R10_CLASS_NAMES,
                scope="parent_balanced_eligible_prefix",
            )
            if prefixes
            else None
        ),
        "eligible_prefix_records": prefixes,
    }


def _paired(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right):
        raise ValueError("V27d-r10 paired parent sets differ")
    transitions: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    drone_bird_control = 0
    drone_bird_treatment = 0
    drone_bird_corrections = 0
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if int(a["true_id"]) != int(b["true_id"]):
            raise ValueError("V27d-r10 paired labels differ")
        control_correct = int(a["pred_id"]) == int(a["true_id"])
        treatment_correct = int(b["pred_id"]) == int(b["true_id"])
        transitions[
            f"{'correct' if control_correct else 'wrong'}_to_"
            f"{'correct' if treatment_correct else 'wrong'}"
        ] += 1
        true_id = int(a["true_id"])
        control_pred = int(a["pred_id"])
        treatment_pred = int(b["pred_id"])
        if true_id == 0 and control_pred == 1:
            drone_bird_control += 1
            if treatment_pred == 0:
                drone_bird_corrections += 1
        if true_id == 0 and treatment_pred == 1:
            drone_bird_treatment += 1
        if control_pred != treatment_pred:
            changed.append(
                {
                    "parent_sample_id": parent_id,
                    "true_id": true_id,
                    "control_pred_id": control_pred,
                    "treatment_pred_id": treatment_pred,
                    "control_correct": control_correct,
                    "treatment_correct": treatment_correct,
                }
            )
    return {
        "parent_count": len(left),
        "control_correct_count": sum(
            int(int(row["pred_id"]) == int(row["true_id"]))
            for row in left.values()
        ),
        "treatment_correct_count": sum(
            int(int(row["pred_id"]) == int(row["true_id"]))
            for row in right.values()
        ),
        "correctness_transitions": dict(sorted(transitions.items())),
        "correct_to_wrong_count": int(transitions["correct_to_wrong"]),
        "wrong_to_correct_count": int(transitions["wrong_to_correct"]),
        "drone_bird_control_error_count": drone_bird_control,
        "drone_bird_treatment_error_count": drone_bird_treatment,
        "drone_bird_correction_count": drone_bird_corrections,
        "changed_predictions": changed,
    }


def _residual_diagnostics(
    payload: Mapping[str, Any],
    output: Mapping[str, Any],
) -> dict[str, Any]:
    tensors = payload["tensors"]
    padding = tensors["padding_mask"]
    valid = output["descriptor_valid_sequence"].detach().cpu()
    correction = output["correction_logits"].detach().cpu()
    increments = output["pairwise_delta_increment"].detach().cpu()
    active = ~padding
    invalid = active & ~valid
    return {
        "active_event_count": int(active.sum()),
        "valid_descriptor_event_count": int(valid.sum()),
        "invalid_descriptor_event_count": int(invalid.sum()),
        "invalid_descriptor_correction_nonzero_count": int(
            (correction[invalid].abs().amax(dim=-1) > 0.0).sum()
        )
        if bool(invalid.any())
        else 0,
        "invalid_descriptor_increment_nonzero_count": int(
            (increments[invalid].abs() > 0.0).sum()
        )
        if bool(invalid.any())
        else 0,
        "correction_zero_sum_max": float(
            correction[active].sum(dim=-1).abs().max()
        )
        if bool(active.any())
        else 0.0,
        "balloon_correction_nonzero_count": int(
            (correction[..., 2][active].abs() > 0.0).sum()
        )
        if bool(active.any())
        else 0,
        "correction_absolute_max": float(correction[active].abs().max())
        if bool(active.any())
        else 0.0,
    }


def _load_model(
    root: Path,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    seed: int,
) -> tuple[CompactCausalDopplerResidual, dict[str, Any], dict[str, str]]:
    caches = {
        split: load_descriptor_cache(_cache_path(root, config, split))
        for split in ("train", "val")
    }
    for split, payload in caches.items():
        validate_cache_binding(payload, root=root, config=config, split=split)
    cache_hashes = {
        split: payload["cache_sha256"] for split, payload in caches.items()
    }
    model = build_model(config["model"], seed=seed)
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
        model=model,
        seed=seed,
        cache_hashes=cache_hashes,
    )
    return model, checkpoint, cache_hashes


def evaluate_seed(
    project_root: str | Path,
    *,
    seed: int,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    if int(seed) not in R10_SEEDS:
        raise ValueError("V27d-r10 evaluation seed is not registered")
    model, checkpoint, _ = _load_model(root, config, manifest, seed)
    device = _device(config)
    model = model.to(device).eval()
    validation = load_descriptor_cache(_cache_path(root, config, "val"))
    indices = torch.arange(len(validation["parent_sample_ids"]), dtype=torch.long)
    with torch.inference_mode():
        treatment_output = model(**_inputs(validation, indices, device))
    treatment_parents, treatment_events = _records(
        validation,
        treatment_output["sequence_logits"],
        arm="compact_causal_doppler_shape",
    )
    control_parents, control_events = _records(
        validation,
        validation["tensors"]["frozen_l2_logits"],
        arm="frozen_v27a_main_control",
    )
    treatment_metrics = _metrics(treatment_parents, treatment_events)
    control_metrics = _metrics(control_parents, control_events)
    paired = _paired(control_parents, treatment_parents)
    treatment_matrix = np.asarray(
        treatment_metrics["parent_metrics"]["confusion_matrix_rows_true_columns_pred"],
        dtype=np.int64,
    )
    control_matrix = np.asarray(
        control_metrics["parent_metrics"]["confusion_matrix_rows_true_columns_pred"],
        dtype=np.int64,
    )
    treatment_per_class = treatment_metrics["parent_metrics"]["per_class"]
    control_per_class = control_metrics["parent_metrics"]["per_class"]
    report = {
        "schema_version": R10_REPORT_SCHEMA_VERSION,
        "experiment_id": "V27d-r10",
        "status": "paired_known_class_validation",
        "seed": int(seed),
        "manifest_sha256": manifest["manifest_sha256"],
        "direct_baseline": "V27a-r3-main-epoch-020",
        "class_names": list(R10_CLASS_NAMES),
        "arms": {
            "frozen_v27a_main_control": control_metrics,
            "compact_causal_doppler_shape": treatment_metrics,
        },
        "paired": paired,
        "deltas": {
            "total_correct_delta": int(
                np.trace(treatment_matrix) - np.trace(control_matrix)
            ),
            "new_error_count": int(paired["correct_to_wrong_count"]),
            "drone_to_bird_error_delta": int(
                treatment_matrix[0, 1] - control_matrix[0, 1]
            ),
            "bird_to_drone_error_delta": int(
                treatment_matrix[1, 0] - control_matrix[1, 0]
            ),
            "drone_bird_correction_count": int(
                paired["drone_bird_correction_count"]
            ),
            "bird_recall_delta": float(
                treatment_per_class[1]["recall"] - control_per_class[1]["recall"]
            ),
        },
        "residual_diagnostics": _residual_diagnostics(validation, treatment_output),
        "checkpoint": {
            "path": relative_project_path(
                root,
                artifact_path(
                    root,
                    config["artifacts"]["checkpoint_template"].format(seed=seed),
                ),
            ),
            "sha256": sha256_file(
                artifact_path(
                    root,
                    config["artifacts"]["checkpoint_template"].format(seed=seed),
                )
            ),
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
        "records": {
            "control_parent": control_parents,
            "treatment_parent": treatment_parents,
        },
    }
    path = artifact_path(
        root,
        config["artifacts"]["evaluation_template"].format(seed=seed),
    )
    write_sealed_json_if_identical(path, report)
    return {**report, "report_path": relative_project_path(root, path)}


def summarize_seeds(
    project_root: str | Path,
    *,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    config, manifest = load_context(root, config_path)
    reports: list[dict[str, Any]] = []
    for seed in R10_SEEDS:
        path = artifact_path(
            root,
            config["artifacts"]["evaluation_template"].format(seed=seed),
            must_exist=True,
        )
        report = read_sealed_json(path, expected_schema=R10_REPORT_SCHEMA_VERSION)
        if int(report.get("seed", -1)) != seed or report.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("V27d-r10 evaluation report provenance differs")
        reports.append(report)
    promotion = config["promotion"]
    per_seed: dict[str, Any] = {}
    for report in reports:
        metrics = report["arms"]["compact_causal_doppler_shape"]["parent_metrics"]
        bird = metrics["per_class"][1]
        deltas = report["deltas"]
        passed = (
            float(bird["recall"]) >= float(promotion["minimum_bird_recall"])
            and int(deltas["new_error_count"]) <= int(promotion["maximum_new_errors"])
            and int(deltas["drone_bird_correction_count"]) >= int(promotion["minimum_drone_bird_corrections"])
            and int(deltas["total_correct_delta"]) >= int(promotion["minimum_total_correct_gain"])
        )
        per_seed[str(report["seed"])] = {
            "bird_precision": bird["precision"],
            "bird_recall": bird["recall"],
            "bird_f1": bird["f1"],
            "total_correct_delta": deltas["total_correct_delta"],
            "new_error_count": deltas["new_error_count"],
            "drone_bird_correction_count": deltas["drone_bird_correction_count"],
            "drone_to_bird_error_delta": deltas["drone_to_bird_error_delta"],
            "passed": passed,
        }
    summary = {
        "schema_version": "RADAR-LN3-V27d-r10-two-seed-summary-v1",
        "experiment_id": "V27d-r10",
        "status": "completed_two_seed_known_class_evaluation",
        "direct_baseline": "V27a-r3-main-epoch-020",
        "manifest_sha256": manifest["manifest_sha256"],
        "seeds": list(R10_SEEDS),
        "per_seed": per_seed,
        "promotion_standard": dict(promotion),
        "promotion_passed": bool(all(row["passed"] for row in per_seed.values())),
        "formal_l3_enabled": False,
        "unknown_support": 0,
        "deployment_eligible": False,
    }
    path = artifact_path(root, config["artifacts"]["summary_path"])
    write_sealed_json_if_identical(path, summary)
    return {**summary, "summary_path": relative_project_path(root, path)}


__all__ = ["evaluate_seed", "summarize_seeds"]
