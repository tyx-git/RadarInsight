"""Known-only factorial evaluation for V27c-r23."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records

from .data import (
    R23L2Dataset,
    build_r23_dataloader,
    r23_model_input_kwargs,
    r23_reference_model_input_kwargs,
    r23_sidecar_kwargs,
)
from .protocol import (
    R23_ARM_SPECS,
    R23_ARMS,
    R23_L2_CLASSES,
    R23_REPORT_SCHEMA_VERSION,
    R23_SUMMARY_SCHEMA_VERSION,
    R23_V27A_MAIN_CONFUSION,
    R23Context,
    r23_artifact_path,
    read_sealed_json,
    write_sealed_json_if_identical,
)
from .training import load_r23_checkpoint


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits)
    if values.ndim != 3 or values.shape[-1] != len(R23_L2_CLASSES):
        raise ValueError("V27c-r23 logits have an invalid shape")
    if not np.isfinite(values).all():
        raise ValueError("V27c-r23 logits are non-finite")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    denominator = np.sum(exponent, axis=-1, keepdims=True)
    if not np.isfinite(denominator).all() or bool((denominator <= 0.0).any()):
        raise ValueError("V27c-r23 softmax denominator is invalid")
    return exponent / denominator


def _records(
    batch: Mapping[str, Any],
    logits: np.ndarray,
    *,
    arm: str,
    effective_motion: torch.Tensor | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    padding = batch["tr_padding_mask"].detach().cpu().numpy()
    lengths = batch["lengths"].detach().cpu().numpy().astype(np.int64)
    labels = batch["labels"].detach().cpu().numpy().astype(np.int64)
    source_motion = batch["motion_values"].detach().cpu().numpy()
    effective = (
        batch["motion_values"]
        if effective_motion is None
        else effective_motion
    ).detach().cpu().numpy()
    probabilities = _softmax(logits)
    if (
        padding.shape != probabilities.shape[:2]
        or source_motion.shape != (*probabilities.shape[:2], 9)
        or effective.shape != (*probabilities.shape[:2], 9)
    ):
        raise ValueError("V27c-r23 evaluation batch dimensions differ")
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row, parent_id in enumerate(batch["parent_sample_ids"]):
        length = int(lengths[row])
        if length <= 0 or not np.array_equal(
            padding[row], np.arange(probabilities.shape[1]) >= length
        ):
            raise ValueError("V27c-r23 evaluation parent padding differs")
        if np.any(source_motion[row, length:] != 0.0):
            raise ValueError("V27c-r23 padded source motion is non-zero")
        if np.any(effective[row, length:] != 0.0):
            raise ValueError("V27c-r23 padded effective motion is non-zero")
        true_id = int(labels[row])
        if not 0 <= true_id < len(R23_L2_CLASSES):
            raise ValueError("V27c-r23 parent label differs")
        source_active_motion = np.max(
            np.abs(source_motion[row, :length]), axis=-1
        ) > 0.0
        effective_active_motion = np.max(
            np.abs(effective[row, :length]), axis=-1
        ) > 0.0
        final = length - 1
        parents.append(
            {
                "parent_sample_id": str(parent_id),
                "track_id": str(batch["track_ids"][row]),
                "source_name": str(batch["source_names"][row]),
                "true_id": true_id,
                "pred_id": int(np.argmax(probabilities[row, final])),
                "probabilities": probabilities[row, final].astype(float).tolist(),
                "logits": logits[row, final].astype(float).tolist(),
                "event_count": length,
                "source_motion_nonzero_event_count": int(source_active_motion.sum()),
                "source_motion_nonzero_event_fraction": float(source_active_motion.mean()),
                "source_final_motion_nonzero": bool(source_active_motion[-1]),
                "effective_motion_nonzero_event_count": int(effective_active_motion.sum()),
                "effective_motion_nonzero_event_fraction": float(effective_active_motion.mean()),
                "effective_final_motion_nonzero": bool(effective_active_motion[-1]),
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
                    "probabilities": probabilities[row, event].astype(float).tolist(),
                    "event_index": event,
                    "window_length": event + 1,
                    "parent_length": length,
                    "source_motion_nonzero": bool(source_active_motion[event]),
                    "effective_motion_nonzero": bool(effective_active_motion[event]),
                    "arm": arm,
                }
            )
    return parents, events


def _metric_delta(
    reference: Mapping[str, Any], comparison: Mapping[str, Any]
) -> dict[str, float]:
    names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    )
    return {name: float(comparison[name]) - float(reference[name]) for name in names}


def _paired_transitions(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in left_records}
    right = {str(row["parent_sample_id"]): row for row in right_records}
    if set(left) != set(right):
        raise ValueError("V27c-r23 paired arms have different parent sets")
    transitions: dict[str, int] = defaultdict(int)
    confusion_delta = np.zeros(
        (len(R23_L2_CLASSES), len(R23_L2_CLASSES)), dtype=np.int64
    )
    for parent_id in sorted(left):
        before, after = left[parent_id], right[parent_id]
        if int(before["true_id"]) != int(after["true_id"]):
            raise ValueError("V27c-r23 paired arm labels differ")
        transitions[
            f"{'correct' if int(before['pred_id']) == int(before['true_id']) else 'wrong'}"
            f"_to_{'correct' if int(after['pred_id']) == int(after['true_id']) else 'wrong'}"
        ] += 1
        true_id = int(after["true_id"])
        confusion_delta[true_id, int(after["pred_id"])] += 1
        confusion_delta[true_id, int(before["pred_id"])] -= 1
    return {
        "parent_count": len(left),
        "correctness_transitions": dict(sorted(transitions.items())),
        "right_minus_left_confusion": confusion_delta.tolist(),
    }


def _paired_bootstrap(
    left_records: Sequence[Mapping[str, Any]],
    right_records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in left_records}
    right = {str(row["parent_sample_id"]): row for row in right_records}
    if set(left) != set(right):
        raise ValueError("V27c-r23 bootstrap parent sets differ")
    strata: dict[int, list[str]] = {index: [] for index in range(len(R23_L2_CLASSES))}
    for parent_id, row in left.items():
        if int(row["true_id"]) != int(right[parent_id]["true_id"]):
            raise ValueError("V27c-r23 bootstrap labels differ")
        strata[int(row["true_id"])].append(parent_id)
    if int(replicates) < 100 or any(not group for group in strata.values()):
        raise ValueError("V27c-r23 paired bootstrap lacks class support")
    rng = np.random.default_rng(int(seed))
    names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    )
    deltas = {name: [] for name in names}
    for _ in range(int(replicates)):
        selected = [
            parent_id
            for group in strata.values()
            for parent_id in rng.choice(group, size=len(group), replace=True).tolist()
        ]
        left_metrics = summarize_records(
            [left[parent_id] for parent_id in selected],
            class_names=R23_L2_CLASSES,
            scope="paired_bootstrap_left",
        )
        right_metrics = summarize_records(
            [right[parent_id] for parent_id in selected],
            class_names=R23_L2_CLASSES,
            scope="paired_bootstrap_right",
        )
        for name in names:
            deltas[name].append(float(right_metrics[name]) - float(left_metrics[name]))
    return {
        "method": "stratified_parent_paired_bootstrap",
        "seed": int(seed),
        "replicates": int(replicates),
        "metric_deltas_right_minus_left": {
            name: {
                "mean": float(np.mean(values)),
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
            for name, values in deltas.items()
        },
    }


def _infer(
    context: R23Context,
    *,
    arm: str,
    seed: int,
    zero_motion: bool = False,
    model: R23IntegratedL2Model | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if model is None:
        model, checkpoint = load_r23_checkpoint(
            context, arm=arm, seed=seed, device=torch.device("cpu")
        )
    else:
        checkpoint = {"checkpoint_provenance": "in_memory_validation"}
    inference_device = next(model.parameters()).device
    dataset = R23L2Dataset(context, split="val", arm=arm)
    loader = build_r23_dataloader(
        dataset, batch_size=context.config.training.batch_size, shuffle=False
    )
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            base_inputs = r23_model_input_kwargs(batch, inference_device)
            reference_inputs = r23_reference_model_input_kwargs(batch, inference_device)
            sidecars = r23_sidecar_kwargs(batch, inference_device)
            if zero_motion:
                sidecars["motion_values"] = torch.zeros_like(sidecars["motion_values"])
            effective_motion = (
                torch.zeros_like(batch["motion_values"])
                if zero_motion or R23_ARM_SPECS[arm]["motion_mode"] == "zero"
                else batch["motion_values"]
            )
            output = model(
                base_inputs=base_inputs,
                reference_base_inputs=reference_inputs,
                **sidecars,
            )
            rows, windows = _records(
                batch,
                output["sequence_logits"].detach().cpu().numpy(),
                arm=f"{arm}_zero_motion" if zero_motion else arm,
                effective_motion=effective_motion,
            )
            parents.extend(rows)
            events.extend(windows)
    if [row["parent_sample_id"] for row in parents] != dataset.parent_ids:
        raise ValueError("V27c-r23 evaluation changed canonical parent order")
    return parents, events, checkpoint


def assess_r23_checkpoint(
    metrics: Mapping[str, Any],
    *,
    epoch: int,
) -> dict[str, Any]:
    """Rank a checkpoint by all-class F1, prioritizing Bird and Drone."""

    if isinstance(epoch, bool) or int(epoch) <= 0:
        raise ValueError("V27c-r23 checkpoint assessment epoch is invalid")
    rows = {str(row["class"]): row for row in metrics.get("per_class") or ()}
    if set(rows) != set(R23_L2_CLASSES):
        raise ValueError("V27c-r23 checkpoint assessment classes differ")
    values = {
        name: float(rows[name]["f1"])
        for name in R23_L2_CLASSES
    }
    if not all(np.isfinite(value) for value in values.values()):
        raise ValueError("V27c-r23 checkpoint assessment is non-finite")
    macro_f1 = float(metrics["macro_f1"])
    accuracy = float(metrics["accuracy"])
    return {
        "schema_version": "RADAR-LN3-V27c-r23-checkpoint-selector-v1",
        "epoch": int(epoch),
        "feasible": True,
        "per_class_f1": values,
        "minimum_per_class_f1": min(values.values()),
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "selection_policy": "global_validation_per_class_f1",
        "ranking_key": [
            min(values.values()),
            values["BirdTarget"],
            values["DroneTarget"],
            values["BalloonTarget"],
            macro_f1,
            accuracy,
            -float(epoch),
        ],
    }


def _metrics(
    parents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    context: R23Context,
) -> dict[str, Any]:
    eligible = [
        row
        for row in events
        if int(row["window_length"]) >= context.config.training.eligible_prefix_min_events
        and int(row["window_length"]) < int(row["parent_length"])
    ]
    prefixes = parent_balanced_prefix_records(
        eligible,
        minimum_events=context.config.training.eligible_prefix_min_events,
        class_names=R23_L2_CLASSES,
    )
    return {
        "parent_metrics": summarize_records(
            parents, class_names=R23_L2_CLASSES, scope="parent_final"
        ),
        "parent_balanced_eligible_prefix_metrics": summarize_records(
            prefixes,
            class_names=R23_L2_CLASSES,
            scope="parent_balanced_eligible_prefix",
        ),
        "eligible_prefix_records": prefixes,
    }


def _cached_v27a_reference(
    context: R23Context,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = R23L2Dataset(context, split="val", arm="legacy_control")
    loader = build_r23_dataloader(
        dataset, batch_size=context.config.training.batch_size, shuffle=False
    )
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    reference_model = build_initialized_model(
        context, arm="legacy_control", seed=42, device=torch.device("cpu")
    )
    reference_model.eval()
    for batch in loader:
        base_inputs = r23_model_input_kwargs(batch, torch.device("cpu"))
        sidecars = r23_sidecar_kwargs(batch, torch.device("cpu"))
        with torch.inference_mode():
            reference_logits = reference_model.v27a_main_endpoint(
                cached_v27_base_logits=sidecars["cached_v27_base_logits"],
                stable_summary_tokens=sidecars["stable_summary_tokens"],
                stable_summary_valid=sidecars["stable_summary_valid"],
                stable_frame_values=sidecars["stable_frame_values"],
                stable_frame_mask=sidecars["stable_frame_mask"],
                padding_mask=base_inputs["tr_padding_mask"],
            )
        rows, windows = _records(
            batch,
            reference_logits.detach().cpu().numpy(),
            arm="frozen_v27a_main_reference",
            effective_motion=torch.zeros_like(batch["motion_values"]),
        )
        parents.extend(rows)
        events.extend(windows)
    matrix = summarize_records(
        parents, class_names=R23_L2_CLASSES, scope="parent_final"
    )["confusion_matrix_rows_true_columns_pred"]
    if matrix != [list(row) for row in R23_V27A_MAIN_CONFUSION]:
        raise AssertionError("V27c-r23 cached V27a Main reference confusion differs")
    return parents, events


def evaluate_r23_seed(context: R23Context, *, seed: int) -> dict[str, Any]:
    if seed not in context.config.training.seeds:
        raise ValueError("V27c-r23 evaluation seed is not registered")
    reference_parents, reference_events = _cached_v27a_reference(context)
    outputs = {arm: _infer(context, arm=arm, seed=seed) for arm in R23_ARMS}
    zero_outputs = {
        arm: _infer(context, arm=arm, seed=seed, zero_motion=True)
        for arm in R23_ARMS
    }
    reference_metrics = _metrics(reference_parents, reference_events, context)
    arm_metrics = {arm: _metrics(*outputs[arm][:2], context) for arm in R23_ARMS}
    zero_metrics = {
        arm: _metrics(*zero_outputs[arm][:2], context) for arm in R23_ARMS
    }

    def delta(left: str, right: str) -> dict[str, float]:
        return _metric_delta(
            arm_metrics[left]["parent_metrics"], arm_metrics[right]["parent_metrics"]
        )

    def zero_delta(arm: str) -> dict[str, float]:
        return _metric_delta(
            zero_metrics[arm]["parent_metrics"], arm_metrics[arm]["parent_metrics"]
        )

    interaction = {
        name: delta("robust_control", "robust_motion")[name]
        - delta("legacy_control", "legacy_motion")[name]
        for name in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
        )
    }
    report = {
        "schema_version": R23_REPORT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "known_only_factorial_input_validation",
        "seed": int(seed),
        "manifest_sha256": context.manifest_sha256,
        "class_names": list(R23_L2_CLASSES),
        "checkpoints": {
            arm: outputs[arm][2]["model_state_sha256"] for arm in R23_ARMS
        },
        "arms": {
            "frozen_v27a_main_reference": reference_metrics,
            **arm_metrics,
            "zero_motion_counterfactual": zero_metrics,
        },
        "deltas": {
            "arm_minus_frozen_v27a_main": {
                arm: _metric_delta(
                    reference_metrics["parent_metrics"], arm_metrics[arm]["parent_metrics"]
                )
                for arm in R23_ARMS
            },
            "motion_effect_within_legacy": delta("legacy_control", "legacy_motion"),
            "motion_effect_within_robust": delta("robust_control", "robust_motion"),
            "clipping_effect_without_motion": delta("legacy_control", "robust_control"),
            "clipping_effect_with_motion": delta("legacy_motion", "robust_motion"),
            "clipping_by_motion_interaction": interaction,
        },
        "paired": {
            "legacy_motion_minus_legacy_control": _paired_transitions(
                outputs["legacy_control"][0], outputs["legacy_motion"][0]
            ),
            "robust_motion_minus_robust_control": _paired_transitions(
                outputs["robust_control"][0], outputs["robust_motion"][0]
            ),
            "robust_control_minus_legacy_control": _paired_transitions(
                outputs["legacy_control"][0], outputs["robust_control"][0]
            ),
        },
        "paired_bootstrap": {
            "legacy_motion_minus_legacy_control": _paired_bootstrap(
                outputs["legacy_control"][0],
                outputs["legacy_motion"][0],
                seed=27_230 + int(seed),
                replicates=context.config.training.bootstrap_replicates,
            ),
            "robust_motion_minus_robust_control": _paired_bootstrap(
                outputs["robust_control"][0],
                outputs["robust_motion"][0],
                seed=27_240 + int(seed),
                replicates=context.config.training.bootstrap_replicates,
            ),
        },
        "counterfactual": {
            "type": "within_arm_zeroed_numeric_motion",
            "validity_mask_model_input": False,
            "zero_motion_is_not_expected_to_replay_independently_trained_control": True,
            "zero_motion_metrics": {
                arm: zero_metrics[arm]["parent_metrics"] for arm in R23_ARMS
            },
            "normal_minus_zero_motion": {
                arm: zero_delta(arm) for arm in R23_ARMS
            },
        },
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
        "records": {
            "frozen_v27a_main_reference_parent": reference_parents,
            **{f"{arm}_parent": outputs[arm][0] for arm in R23_ARMS},
            **{
                f"{arm}_zero_motion_parent": zero_outputs[arm][0]
                for arm in R23_ARMS
            },
        },
    }
    path = r23_artifact_path(
        context.project_root,
        context.config.artifacts.evaluation_report_template,
        seed=seed,
    )
    write_sealed_json_if_identical(path, report)
    return {**report, "report_path": str(path)}


def evaluate_r23_single_arm(
    context: R23Context,
    *,
    arm: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate one trained arm without requiring the other factorial arms."""

    if arm not in R23_ARMS:
        raise ValueError("V27c-r23 single-arm evaluation arm is invalid")
    if seed not in context.config.training.seeds:
        raise ValueError("V27c-r23 evaluation seed is not registered")
    reference_parents, reference_events = _cached_v27a_reference(context)
    parents, events, checkpoint = _infer(context, arm=arm, seed=seed)
    reference_metrics = _metrics(reference_parents, reference_events, context)
    arm_metrics = _metrics(parents, events, context)
    report = {
        "schema_version": "RADAR-LN3-V27c-r23-single-arm-evaluation-v1",
        "experiment_id": context.config.experiment_id,
        "arm": arm,
        "seed": int(seed),
        "checkpoint_state_sha256": checkpoint["model_state_sha256"],
        "reference": reference_metrics["parent_metrics"],
        "metrics": arm_metrics["parent_metrics"],
        "parent_balanced_eligible_prefix_metrics": arm_metrics[
            "parent_balanced_eligible_prefix_metrics"
        ],
        "delta_vs_frozen_v27a_main": _metric_delta(
            reference_metrics["parent_metrics"], arm_metrics["parent_metrics"]
        ),
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
    }
    return report


def summarize_r23_seeds(context: R23Context) -> dict[str, Any]:
    reports = []
    for seed in context.config.training.seeds:
        path = r23_artifact_path(
            context.project_root,
            context.config.artifacts.evaluation_report_template,
            seed=seed,
        )
        report = read_sealed_json(path, schema=R23_REPORT_SCHEMA_VERSION)
        if (
            int(report.get("seed", -1)) != seed
            or report.get("manifest_sha256") != context.manifest_sha256
        ):
            raise ValueError("V27c-r23 seed report does not belong to this manifest")
        reports.append(report)

    aggregate: dict[str, Any] = {}
    for arm in R23_ARMS:
        values = np.asarray(
            [
                report["arms"][arm]["parent_metrics"]["macro_f1"]
                for report in reports
            ],
            dtype=np.float64,
        )
        bird_recall = np.asarray(
            [
                report["arms"][arm]["parent_metrics"]["per_class"][1]["recall"]
                for report in reports
            ],
            dtype=np.float64,
        )
        aggregate[arm] = {
            "macro_f1": {
                "values": values.tolist(),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
            },
            "bird_recall": {
                "values": bird_recall.tolist(),
                "mean": float(bird_recall.mean()),
                "std": float(bird_recall.std(ddof=0)),
            },
        }
    interaction = np.asarray(
        [
            report["deltas"]["clipping_by_motion_interaction"]["macro_f1"]
            for report in reports
        ],
        dtype=np.float64,
    )
    safety = []
    reference_diagonal = np.diag(np.asarray(R23_V27A_MAIN_CONFUSION, dtype=np.int64))
    for report in reports:
        arm_matrices = {
            arm: np.asarray(
                report["arms"][arm]["parent_metrics"][
                    "confusion_matrix_rows_true_columns_pred"
                ],
                dtype=np.int64,
            )
            for arm in R23_ARMS
        }
        safety.append(
            {
                "seed": int(report["seed"]),
                "per_arm_known_correct_count": {
                    arm: int(np.trace(matrix))
                    for arm, matrix in arm_matrices.items()
                },
                "per_arm_correct_class_counts": {
                    arm: np.diag(matrix).astype(int).tolist()
                    for arm, matrix in arm_matrices.items()
                },
                "all_arms_at_or_above_v27a_main_class_correctness": all(
                    bool(np.all(np.diag(matrix) >= reference_diagonal))
                    for matrix in arm_matrices.values()
                ),
            }
        )
    summary = {
        "schema_version": R23_SUMMARY_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "two_seed_known_only_factorial_summary",
        "manifest_sha256": context.manifest_sha256,
        "seeds": list(context.config.training.seeds),
        "aggregate": aggregate,
        "clipping_by_motion_interaction_macro_f1": {
            "values": interaction.tolist(),
            "mean": float(interaction.mean()),
            "std": float(interaction.std(ddof=0)),
        },
        "per_seed_safety": safety,
        "interpretation_boundary": {
            "primary_comparison": "2x2 clipping_by_causal_motion on known-only L2",
            "unknown_rejection_claim": False,
            "l3_claim": False,
            "test_claim": False,
            "three_seed_claim": False,
        },
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "test_payload_access_count": 0,
        },
    }
    path = r23_artifact_path(
        context.project_root, context.config.artifacts.summary_report_path
    )
    write_sealed_json_if_identical(path, summary)
    return {**summary, "report_path": str(path)}


__all__ = ["evaluate_r23_seed", "summarize_r23_seeds"]
