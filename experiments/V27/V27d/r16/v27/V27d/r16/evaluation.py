"""Paired known-class evaluation for V27d-r16."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.runtime import relative_project_path
from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records

from .data import (
    R16L2Dataset,
    build_r16_dataloader,
    r16_frame_time_kwargs,
    r16_model_input_kwargs,
)
from .model import R16IntegratedL2Model
from .protocol import (
    R16_ARMS,
    R16_CLASS_NAMES,
    R16_REPORT_SCHEMA,
    R16_SUMMARY_SCHEMA,
    R16Context,
    artifact_path,
    read_sealed_json,
    write_sealed_json_if_identical,
)
from .training import load_checkpoint


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _records(
    batch: Mapping[str, Any],
    logits: np.ndarray,
    *,
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    padding = batch["tr_padding_mask"].cpu().numpy()
    lengths = batch["lengths"].cpu().numpy().astype(np.int64)
    labels = batch["labels"].cpu().numpy().astype(np.int64)
    probabilities = _softmax(logits)
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row, parent_id in enumerate(batch["parent_sample_ids"]):
        length = int(lengths[row])
        if length <= 0 or not np.array_equal(padding[row], np.arange(logits.shape[1]) >= length):
            raise ValueError("V27d-r16 evaluation padding differs from lengths")
        true_id = int(labels[row])
        final = length - 1
        parents.append(
            {
                "parent_sample_id": str(parent_id),
                "track_id": str(batch["track_ids"][row]),
                "source_name": str(batch["source_names"][row]),
                "true_id": true_id,
                "pred_id": int(np.argmax(probabilities[row, final])),
                "probabilities": [float(value) for value in probabilities[row, final]],
                "logits": [float(value) for value in logits[row, final]],
                "event_count": length,
                "arm": arm,
            }
        )
        for event in range(length):
            events.append(
                {
                    "parent_sample_id": str(parent_id),
                    "track_id": str(batch["track_ids"][row]),
                    "source_name": str(batch["source_names"][row]),
                    "true_id": true_id,
                    "pred_id": int(np.argmax(probabilities[row, event])),
                    "probabilities": [float(value) for value in probabilities[row, event]],
                    "logits": [float(value) for value in logits[row, event]],
                    "event_index": event,
                    "window_length": event + 1,
                    "parent_length": length,
                    "arm": arm,
                }
            )
    return parents, events


def _batch_time_usage(batch: Mapping[str, Any]) -> dict[str, int]:
    active = ~batch["tr_padding_mask"]
    mask = batch["rd_frame_mask"]
    present = mask.any(dim=-1) & active
    count = mask.to(dtype=torch.long).sum(dim=-1)
    valid = batch["rd_frame_time_valid"] & active
    fallback = batch["rd_frame_time_fallback"] & active
    adjacent = mask[..., 1:] & mask[..., :-1]
    deltas = batch["rd_frame_times"][..., 1:] - batch["rd_frame_times"][..., :-1]
    positive = (adjacent & deltas.gt(1.0e-8)).sum()
    zero = (adjacent & deltas.abs().le(1.0e-8)).sum()
    negative = (adjacent & deltas.lt(-1.0e-8)).sum()
    return {
        "rd_event_count": int(active.sum().item()),
        "rd_present_event_count": int(present.sum().item()),
        "rd_temporal_eligible_event_count": int(valid.sum().item()),
        "rd_fallback_event_count": int(fallback.sum().item()),
        "rd_single_or_missing_event_count": int((present & count.lt(2)).sum().item()),
        "rd_positive_delta_count": int(positive.item()),
        "rd_zero_delta_count": int(zero.item()),
        "rd_negative_delta_count": int(negative.item()),
    }


def _merge_counts(rows: Sequence[Mapping[str, int]]) -> dict[str, int]:
    result: Counter[str] = Counter()
    for row in rows:
        result.update({str(key): int(value) for key, value in row.items()})
    return dict(sorted(result.items()))


def evaluate_model(
    context: R16Context,
    *,
    model: R16IntegratedL2Model,
    dataset: R16L2Dataset,
    device: torch.device,
    checkpoint_epoch: int,
) -> dict[str, Any]:
    model.eval()
    parent_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    usage: list[dict[str, int]] = []
    with torch.inference_mode():
        for batch in build_r16_dataloader(
            dataset,
            batch_size=int(context.config.training["batch_size"]),
            shuffle=False,
        ):
            inputs = r16_model_input_kwargs(batch, device)
            sidecars = r16_frame_time_kwargs(batch, device)
            output = model(
                base_inputs=inputs,
                cached_v27_base_logits=batch["cached_v27_base_logits"].to(device=device, non_blocking=True),
                stable_summary_tokens=batch["stable_summary_tokens"].to(device=device, non_blocking=True),
                stable_summary_valid=batch["stable_summary_valid"].to(device=device, non_blocking=True),
                stable_frame_values=batch["stable_frame_values"].to(device=device, non_blocking=True),
                stable_frame_mask=batch["stable_frame_mask"].to(device=device, non_blocking=True),
                **sidecars,
            )
            logits = output["sequence_logits"].detach().cpu().numpy()
            if not np.isfinite(logits).all():
                raise FloatingPointError("V27d-r16 evaluation produced non-finite logits")
            parents, events = _records(batch, logits, arm=model.arm)
            parent_rows.extend(parents)
            event_rows.extend(events)
            usage.append(_batch_time_usage(batch))
    eligible = [
        row
        for row in event_rows
        if int(row["window_length"]) >= 3
        and int(row["window_length"]) < int(row["parent_length"])
    ]
    balanced = parent_balanced_prefix_records(
        eligible,
        minimum_events=3,
        class_names=R16_CLASS_NAMES,
    )
    return {
        "parent_metrics": summarize_records(
            parent_rows,
            class_names=R16_CLASS_NAMES,
            scope="parent_final",
        ),
        "parent_balanced_eligible_prefix_metrics": summarize_records(
            balanced,
            class_names=R16_CLASS_NAMES,
            scope="parent_balanced_eligible_prefix_diagnostic",
        ),
        "records": {"parent": parent_rows, "eligible_prefix": eligible, "window": event_rows},
        "rd_time_usage": _merge_counts(usage),
        "checkpoint_epoch": int(checkpoint_epoch),
    }


def evaluate_arm(context: R16Context, *, arm: str, seed: int) -> dict[str, Any]:
    if arm not in R16_ARMS or int(seed) not in (42, 44):
        raise ValueError("V27d-r16 evaluation arm or seed is invalid")
    device = _device(context)
    epoch = int(context.config.training["primary_epoch"])
    model, checkpoint = load_checkpoint(
        context,
        arm=arm,
        seed=seed,
        epoch=epoch,
        device=device,
    )
    dataset = R16L2Dataset(context, split="val")
    evaluation = evaluate_model(
        context,
        model=model,
        dataset=dataset,
        device=device,
        checkpoint_epoch=epoch,
    )
    checkpoint_path = artifact_path(
        context.project_root,
        context.config.artifacts["checkpoint_template"],
        arm=arm,
        seed=seed,
        epoch=epoch,
    )
    payload = {
        "schema_version": R16_REPORT_SCHEMA,
        "status": "completed_known_class_l2_evaluation_only",
        "experiment_id": "V27d-r16",
        "arm": arm,
        "seed": int(seed),
        "manifest_sha256": context.manifest_sha256,
        "checkpoint_path": relative_project_path(context.project_root, checkpoint_path),
        "checkpoint_model_state_sha256": checkpoint["model_state_sha256"],
        "parent_metrics": evaluation["parent_metrics"],
        "parent_balanced_eligible_prefix_metrics": evaluation["parent_balanced_eligible_prefix_metrics"],
        "rd_time_usage": evaluation["rd_time_usage"],
        "records": evaluation["records"],
        "scope": {
            "unknown_training_exposure": 0,
            "unknown_evaluation_exposure": 0,
            "test_payload_access_count": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
        },
        "deployment_eligible": False,
    }
    path = artifact_path(
        context.project_root,
        context.config.artifacts["evaluation_template"],
        arm=arm,
        seed=seed,
    )
    write_sealed_json_if_identical(path, payload)
    return {**payload, "report_path": str(path)}


def _class_rows(metrics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {str(row["class"]): dict(row) for row in metrics.get("per_class", ())}
    if set(rows) != set(R16_CLASS_NAMES):
        raise ValueError("V27d-r16 metric classes differ")
    return rows


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"):
        result[f"{key}_pp"] = 100.0 * (float(left[key]) - float(right[key]))
    left_rows, right_rows = _class_rows(left), _class_rows(right)
    result["per_class"] = {
        name: {
            metric: 100.0 * (float(left_rows[name][metric]) - float(right_rows[name][metric]))
            for metric in ("precision", "recall", "f1")
        }
        for name in R16_CLASS_NAMES
    }
    return result


def _paired_transitions(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right):
        raise ValueError("V27d-r16 paired parent sets differ")
    transitions: Counter[str] = Counter()
    changed: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if int(a["true_id"]) != int(b["true_id"]):
            raise ValueError("V27d-r16 paired labels differ")
        a_correct = int(a["pred_id"]) == int(a["true_id"])
        b_correct = int(b["pred_id"]) == int(b["true_id"])
        transitions[f"{'correct' if a_correct else 'wrong'}_to_{'correct' if b_correct else 'wrong'}"] += 1
        if int(a["pred_id"]) != int(b["pred_id"]):
            changed.append(
                {
                    "parent_sample_id": parent_id,
                    "true_id": int(a["true_id"]),
                    "control_pred_id": int(a["pred_id"]),
                    "treatment_pred_id": int(b["pred_id"]),
                    "control_correct": a_correct,
                    "treatment_correct": b_correct,
                }
            )
    return {
        "parent_count": len(left),
        "control_correct_count": sum(int(int(row["pred_id"]) == int(row["true_id"])) for row in left.values()),
        "treatment_correct_count": sum(int(int(row["pred_id"]) == int(row["true_id"])) for row in right.values()),
        "correctness_transitions": dict(sorted(transitions.items())),
        "changed_predictions": changed,
    }


def _r14_best_candidate(context: R16Context, seed: int) -> dict[str, Any]:
    from v27.V27d.r14.protocol import artifact_path as r14_artifact_path

    r14_path = r14_artifact_path(
        context.project_root,
        context.r14_context.config.artifacts["paired_template"],
        seed=seed,
        must_exist=True,
    )
    payload = read_sealed_json(r14_path)
    best = payload.get("arms", {}).get("best_l2_candidate")
    if not isinstance(best, Mapping):
        raise ValueError("V27d-r16 r14 best-L2 reference is missing")
    return dict(best)


def build_paired_report(context: R16Context, *, seed: int) -> dict[str, Any]:
    paths = {
        arm: artifact_path(
            context.project_root,
            context.config.artifacts["evaluation_template"],
            arm=arm,
            seed=seed,
            must_exist=True,
        )
        for arm in R16_ARMS
    }
    control = read_sealed_json(paths["control"], expected_schema=R16_REPORT_SCHEMA)
    treatment = read_sealed_json(paths["treatment"], expected_schema=R16_REPORT_SCHEMA)
    control_metrics = control["parent_metrics"]
    treatment_metrics = treatment["parent_metrics"]
    best = _r14_best_candidate(context, seed)
    payload = {
        "schema_version": R16_REPORT_SCHEMA,
        "status": "paired_known_class_l2_validation",
        "experiment_id": "V27d-r16",
        "seed": int(seed),
        "manifest_sha256": context.manifest_sha256,
        "arms": {
            "control": {"parent_metrics": control_metrics, "rd_time_usage": control["rd_time_usage"]},
            "treatment": {"parent_metrics": treatment_metrics, "rd_time_usage": treatment["rd_time_usage"]},
            "best_l2_candidate": best,
        },
        "paired_control_vs_treatment": _paired_transitions(
            control["records"]["parent"],
            treatment["records"]["parent"],
        ),
        "deltas": {
            "treatment_minus_control": _metric_delta(treatment_metrics, control_metrics),
            "control_minus_best_l2_candidate": _metric_delta(control_metrics, best["parent_metrics"]),
            "treatment_minus_best_l2_candidate": _metric_delta(treatment_metrics, best["parent_metrics"]),
        },
        "scope": {
            "comparison": "known_class_parent_final_and_eligible_prefix_only",
            "unknown_training_exposure": 0,
            "unknown_evaluation_exposure": 0,
            "test_payload_access_count": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
            "oof": "not_executed",
            "cross_source": "not_executed",
        },
        "deployment_eligible": False,
    }
    path = artifact_path(
        context.project_root,
        context.config.artifacts["paired_template"],
        seed=seed,
    )
    write_sealed_json_if_identical(path, payload)
    return {**payload, "report_path": str(path)}


def _aggregate(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "values": [float(value) for value in array],
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def summarize_seeds(context: R16Context) -> dict[str, Any]:
    reports = {
        str(seed): read_sealed_json(
            artifact_path(
                context.project_root,
                context.config.artifacts["paired_template"],
                seed=seed,
                must_exist=True,
            ),
            expected_schema=R16_REPORT_SCHEMA,
        )
        for seed in (42, 44)
    }
    payload = {
        "schema_version": R16_SUMMARY_SCHEMA,
        "status": "two_seed_known_class_summary",
        "experiment_id": "V27d-r16",
        "manifest_sha256": context.manifest_sha256,
        "seeds": [42, 44],
        "per_seed": {},
        "aggregate": {
            "control_accuracy": _aggregate([float(reports[str(seed)]["arms"]["control"]["parent_metrics"]["accuracy"]) for seed in (42, 44)]),
            "treatment_accuracy": _aggregate([float(reports[str(seed)]["arms"]["treatment"]["parent_metrics"]["accuracy"]) for seed in (42, 44)]),
            "treatment_minus_control_accuracy_pp": _aggregate([float(reports[str(seed)]["deltas"]["treatment_minus_control"]["accuracy_pp"]) for seed in (42, 44)]),
            "treatment_minus_control_macro_f1_pp": _aggregate([float(reports[str(seed)]["deltas"]["treatment_minus_control"]["macro_f1_pp"]) for seed in (42, 44)]),
            "treatment_temporal_eligible_event_count": _aggregate([float(reports[str(seed)]["arms"]["treatment"]["rd_time_usage"]["rd_temporal_eligible_event_count"]) for seed in (42, 44)]),
        },
        "scope": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
            "oof": "not_executed",
            "cross_source": "not_executed",
        },
        "deployment_eligible": False,
    }
    for seed in (42, 44):
        report = reports[str(seed)]
        payload["per_seed"][str(seed)] = {
            "control": report["arms"]["control"]["parent_metrics"],
            "treatment": report["arms"]["treatment"]["parent_metrics"],
            "treatment_minus_control": report["deltas"]["treatment_minus_control"],
            "treatment_minus_best_l2_candidate": report["deltas"]["treatment_minus_best_l2_candidate"],
            "rd_time_usage": report["arms"]["treatment"]["rd_time_usage"],
            "correctness_transitions": report["paired_control_vs_treatment"]["correctness_transitions"],
        }
    path = artifact_path(
        context.project_root,
        context.config.artifacts["summary_path"],
    )
    write_sealed_json_if_identical(path, payload)
    return {**payload, "report_path": str(path)}


def _device(context: R16Context) -> torch.device:
    if context.config.training["device"] == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("V27d-r16 requires CUDA, but CUDA is unavailable")
        return torch.device("cuda:0")
    return torch.device("cpu")


__all__ = ["build_paired_report", "evaluate_arm", "evaluate_model", "summarize_seeds"]