"""Paired, known-only evaluation for independent r22 L2 retraining."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records

from .data import (
    R22L2Dataset,
    build_r22_dataloader,
    r22_model_input_kwargs,
    r22_sidecar_kwargs,
)
from .protocol import (
    R22_ARMS,
    R22_FROZEN_ERROR_PARENT_IDS,
    R22_L2_CLASSES,
    R22_REPORT_SCHEMA_VERSION,
    R22_SUMMARY_SCHEMA_VERSION,
    R22_V27A_MAIN_CONFUSION,
    R22Context,
    r22_artifact_path,
    read_sealed_json,
    write_sealed_json_if_identical,
)
from .training import load_r22_checkpoint


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=-1, keepdims=True)


def _records(
    batch: Mapping[str, Any], logits: np.ndarray, *, arm: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    padding = batch["tr_padding_mask"].numpy()
    lengths = batch["lengths"].numpy().astype(np.int64)
    labels = batch["labels"].numpy().astype(np.int64)
    motion = batch["motion_values"].numpy()
    probabilities = _softmax(logits)
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row, parent_id in enumerate(batch["parent_sample_ids"]):
        length = int(lengths[row])
        if length <= 0 or not np.array_equal(
            padding[row], np.arange(logits.shape[1]) >= length
        ):
            raise ValueError("V27c-r22 evaluation parent padding differs")
        true_id = int(labels[row])
        final = length - 1
        active_motion = np.abs(motion[row, :length]).amax(axis=-1) > 0.0
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
                "motion_nonzero_event_count": int(active_motion.sum()),
                "motion_nonzero_event_fraction": float(active_motion.mean()),
                "final_motion_nonzero": bool(active_motion[-1]),
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
                    "event_index": event,
                    "window_length": event + 1,
                    "parent_length": length,
                    "motion_nonzero": bool(active_motion[event]),
                    "arm": arm,
                }
            )
    return parents, events


def _metric_delta(reference: Mapping[str, Any], comparison: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: float(comparison[name]) - float(reference[name])
        for name in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
        )
    }


def _paired_transitions(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right):
        raise ValueError("V27c-r22 paired arms have different parent sets")
    transitions: dict[str, int] = defaultdict(int)
    confusion_delta = np.zeros((len(R22_L2_CLASSES), len(R22_L2_CLASSES)), dtype=np.int64)
    historic: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        control_row, treatment_row = left[parent_id], right[parent_id]
        if int(control_row["true_id"]) != int(treatment_row["true_id"]):
            raise ValueError("V27c-r22 paired arm labels differ")
        before = int(control_row["pred_id"]) == int(control_row["true_id"])
        after = int(treatment_row["pred_id"]) == int(treatment_row["true_id"])
        transitions[
            f"{'correct' if before else 'wrong'}_to_{'correct' if after else 'wrong'}"
        ] += 1
        confusion_delta[int(treatment_row["true_id"]), int(treatment_row["pred_id"])] += 1
        confusion_delta[int(control_row["true_id"]), int(control_row["pred_id"])] -= 1
        if parent_id in R22_FROZEN_ERROR_PARENT_IDS:
            historic.append(
                {
                    "parent_sample_id": parent_id,
                    "true_class": R22_L2_CLASSES[int(control_row["true_id"])],
                    "control_prediction": R22_L2_CLASSES[int(control_row["pred_id"])],
                    "motion_treatment_prediction": R22_L2_CLASSES[
                        int(treatment_row["pred_id"])
                    ],
                    "motion_final_nonzero": bool(treatment_row["final_motion_nonzero"]),
                    "motion_nonzero_event_fraction": float(
                        treatment_row["motion_nonzero_event_fraction"]
                    ),
                }
            )
    if {row["parent_sample_id"] for row in historic} != set(R22_FROZEN_ERROR_PARENT_IDS):
        raise AssertionError("V27c-r22 historical Drone-to-Bird error set differs")
    return {
        "parent_count": len(left),
        "correctness_transitions": dict(sorted(transitions.items())),
        "treatment_minus_control_confusion": confusion_delta.tolist(),
        "historic_drone_to_bird_error_audit": historic,
    }


def _paired_bootstrap(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in control}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    strata: dict[int, list[str]] = {index: [] for index in range(len(R22_L2_CLASSES))}
    for parent_id, row in left.items():
        if int(row["true_id"]) != int(right[parent_id]["true_id"]):
            raise ValueError("V27c-r22 bootstrap labels differ")
        strata[int(row["true_id"])].append(parent_id)
    if replicates < 100 or any(not values for values in strata.values()):
        raise ValueError("V27c-r22 paired bootstrap lacks class support")
    rng = np.random.default_rng(int(seed))
    metrics = {
        name: []
        for name in (
            "accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
            "balanced_accuracy",
        )
    }
    for _ in range(int(replicates)):
        selected = [
            parent
            for group in strata.values()
            for parent in rng.choice(group, size=len(group), replace=True).tolist()
        ]
        base = summarize_records(
            [left[parent] for parent in selected],
            class_names=R22_L2_CLASSES,
            scope="paired_bootstrap_control",
        )
        motion = summarize_records(
            [right[parent] for parent in selected],
            class_names=R22_L2_CLASSES,
            scope="paired_bootstrap_motion",
        )
        for name in metrics:
            metrics[name].append(float(motion[name]) - float(base[name]))
    return {
        "method": "stratified_parent_paired_bootstrap",
        "seed": int(seed),
        "replicates": int(replicates),
        "metric_deltas_treatment_minus_control": {
            name: {
                "mean": float(np.mean(values)),
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
            for name, values in metrics.items()
        },
    }


def _infer(
    context: R22Context,
    *,
    arm: str,
    seed: int,
    motion_mode: str = "normal",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if motion_mode not in {"normal", "zero"}:
        raise ValueError("V27c-r22 inference motion mode is invalid")
    model, checkpoint = load_r22_checkpoint(
        context, arm=arm, seed=seed, device=torch.device("cpu")
    )
    dataset = R22L2Dataset(context, split="val")
    loader = build_r22_dataloader(
        dataset,
        batch_size=context.config.training.batch_size,
        shuffle=False,
    )
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            base_inputs = r22_model_input_kwargs(batch, torch.device("cpu"))
            sidecars = r22_sidecar_kwargs(batch, torch.device("cpu"))
            if motion_mode == "zero":
                sidecars["motion_values"] = torch.zeros_like(sidecars["motion_values"])
            output = model(base_inputs=base_inputs, **sidecars)
            rows, windows = _records(
                batch,
                output["sequence_logits"].numpy(),
                arm=arm if motion_mode == "normal" else f"{arm}_zero_motion",
            )
            parents.extend(rows)
            events.extend(windows)
    expected = dataset.parent_ids
    if [row["parent_sample_id"] for row in parents] != expected:
        raise ValueError("V27c-r22 evaluation changed canonical parent order")
    return parents, events, checkpoint


def _metrics(
    parents: Sequence[Mapping[str, Any]], events: Sequence[Mapping[str, Any]], context: R22Context
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
        class_names=R22_L2_CLASSES,
    )
    return {
        "parent_metrics": summarize_records(
            parents, class_names=R22_L2_CLASSES, scope="parent_final"
        ),
        "parent_balanced_eligible_prefix_metrics": summarize_records(
            prefixes,
            class_names=R22_L2_CLASSES,
            scope="parent_balanced_eligible_prefix",
        ),
        "eligible_prefix_records": prefixes,
    }


def _cached_v27a_reference(context: R22Context) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build the immutable V27a Main reference from r21's audit cache only."""

    dataset = R22L2Dataset(context, split="val")
    loader = build_r22_dataloader(
        dataset,
        batch_size=context.config.training.batch_size,
        shuffle=False,
    )
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for batch in loader:
        rows, windows = _records(
            batch,
            batch["v27a_main_reference_logits"].numpy(),
            arm="frozen_v27a_main_reference",
        )
        parents.extend(rows)
        events.extend(windows)
    matrix = summarize_records(
        parents, class_names=R22_L2_CLASSES, scope="parent_final"
    )["confusion_matrix_rows_true_columns_pred"]
    if matrix != [list(row) for row in R22_V27A_MAIN_CONFUSION]:
        raise AssertionError("V27c-r22 cached V27a Main reference confusion differs")
    return parents, events


def evaluate_r22_seed(context: R22Context, *, seed: int) -> dict[str, Any]:
    """Write one complete, matched-arm validation report for a seed."""

    if seed not in context.config.training.seeds:
        raise ValueError("V27c-r22 evaluation seed is not registered")
    reference_parents, reference_events = _cached_v27a_reference(context)
    control_parents, control_events, control_checkpoint = _infer(
        context, arm="control", seed=seed
    )
    treatment_parents, treatment_events, treatment_checkpoint = _infer(
        context, arm="motion_treatment", seed=seed
    )
    treatment_zero_parents, treatment_zero_events, _ = _infer(
        context, arm="motion_treatment", seed=seed, motion_mode="zero"
    )
    reference_metrics = _metrics(reference_parents, reference_events, context)
    control_metrics = _metrics(control_parents, control_events, context)
    treatment_metrics = _metrics(treatment_parents, treatment_events, context)
    zero_metrics = _metrics(treatment_zero_parents, treatment_zero_events, context)
    paired = _paired_transitions(control_parents, treatment_parents)
    zero_paired = _paired_transitions(treatment_zero_parents, treatment_parents)
    report = {
        "schema_version": R22_REPORT_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "paired_known_only_integrated_l2_validation",
        "seed": int(seed),
        "manifest_sha256": context.manifest_sha256,
        "class_names": list(R22_L2_CLASSES),
        "checkpoints": {
            "control_state_sha256": control_checkpoint["model_state_sha256"],
            "motion_treatment_state_sha256": treatment_checkpoint["model_state_sha256"],
        },
        "arms": {
            "frozen_v27a_main_reference": reference_metrics,
            "control": control_metrics,
            "motion_treatment": treatment_metrics,
            "motion_treatment_zero_motion_counterfactual": zero_metrics,
        },
        "deltas": {
            "control_minus_frozen_v27a_main": _metric_delta(
                reference_metrics["parent_metrics"], control_metrics["parent_metrics"]
            ),
            "motion_treatment_minus_control": _metric_delta(
                control_metrics["parent_metrics"], treatment_metrics["parent_metrics"]
            ),
            "motion_treatment_minus_own_zero_motion": _metric_delta(
                zero_metrics["parent_metrics"], treatment_metrics["parent_metrics"]
            ),
        },
        "paired": paired,
        "paired_bootstrap": _paired_bootstrap(
            control_parents,
            treatment_parents,
            seed=27_220 + int(seed),
            replicates=context.config.training.bootstrap_replicates,
        ),
        "counterfactual": {
            "type": "within_motion_treatment_zeroed_numeric_motion",
            "validity_mask_model_input": False,
            "paired_vs_normal": zero_paired,
            "zero_motion_is_not_expected_to_replay_control_after_independent_training": True,
            "zero_motion_metrics": zero_metrics["parent_metrics"],
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
            "control_parent": control_parents,
            "motion_treatment_parent": treatment_parents,
            "motion_treatment_zero_motion_parent": treatment_zero_parents,
        },
    }
    path = r22_artifact_path(
        context.project_root,
        context.config.artifacts.evaluation_report_template,
        seed=seed,
    )
    write_sealed_json_if_identical(path, report)
    return {**report, "report_path": str(path)}


def summarize_r22_seeds(context: R22Context) -> dict[str, Any]:
    """Aggregate all prespecified seeds without selecting a winner."""

    reports: list[dict[str, Any]] = []
    for seed in context.config.training.seeds:
        path = r22_artifact_path(
            context.project_root,
            context.config.artifacts.evaluation_report_template,
            seed=seed,
        )
        report = read_sealed_json(path, schema=R22_REPORT_SCHEMA_VERSION)
        if (
            int(report.get("seed", -1)) != seed
            or report.get("manifest_sha256") != context.manifest_sha256
        ):
            raise ValueError("V27c-r22 seed report does not belong to this manifest")
        reports.append(report)
    treatment_f1 = np.asarray(
        [item["arms"]["motion_treatment"]["parent_metrics"]["macro_f1"] for item in reports],
        dtype=np.float64,
    )
    control_f1 = np.asarray(
        [item["arms"]["control"]["parent_metrics"]["macro_f1"] for item in reports],
        dtype=np.float64,
    )
    treatment_bird_precision = np.asarray(
        [
            item["arms"]["motion_treatment"]["parent_metrics"]["per_class"][1][
                "precision"
            ]
            for item in reports
        ],
        dtype=np.float64,
    )
    treatment_drone_to_bird = np.asarray(
        [
            item["arms"]["motion_treatment"]["parent_metrics"][
                "confusion_matrix_rows_true_columns_pred"
            ][0][1]
            for item in reports
        ],
        dtype=np.int64,
    )
    safety = []
    for report in reports:
        matrix = np.asarray(
            report["arms"]["motion_treatment"]["parent_metrics"][
                "confusion_matrix_rows_true_columns_pred"
            ],
            dtype=np.int64,
        )
        interval = report["paired_bootstrap"]["metric_deltas_treatment_minus_control"][
            "macro_f1"
        ]
        safety.append(
            {
                "seed": int(report["seed"]),
                "known_correct_count": int(np.trace(matrix)),
                "per_class_correct": np.diag(matrix).astype(int).tolist(),
                "treatment_minus_control_macro_f1_lower_95": float(
                    interval["lower_95"]
                ),
                "safe_against_v27a_main_correctness_floor": bool(
                    np.all(np.diag(matrix) >= np.asarray((109, 54, 52)))
                ),
            }
        )
    summary = {
        "schema_version": R22_SUMMARY_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "three_seed_matched_integrated_l2_summary",
        "manifest_sha256": context.manifest_sha256,
        "seeds": list(context.config.training.seeds),
        "aggregate": {
            "control_macro_f1": {
                "values": control_f1.tolist(),
                "mean": float(control_f1.mean()),
                "std": float(control_f1.std(ddof=0)),
            },
            "motion_treatment_macro_f1": {
                "values": treatment_f1.tolist(),
                "mean": float(treatment_f1.mean()),
                "std": float(treatment_f1.std(ddof=0)),
            },
            "motion_treatment_minus_control_macro_f1": {
                "values": (treatment_f1 - control_f1).tolist(),
                "mean": float((treatment_f1 - control_f1).mean()),
                "std": float((treatment_f1 - control_f1).std(ddof=0)),
            },
            "motion_treatment_bird_precision": {
                "values": treatment_bird_precision.tolist(),
                "mean": float(treatment_bird_precision.mean()),
                "std": float(treatment_bird_precision.std(ddof=0)),
            },
            "motion_treatment_drone_to_bird_errors": {
                "values": treatment_drone_to_bird.astype(int).tolist(),
                "mean": float(treatment_drone_to_bird.mean()),
            },
        },
        "per_seed_safety": safety,
        "interpretation_boundary": {
            "primary_comparison": "motion_treatment_minus_matched_control",
            "not_a_v27a_frozen_endpoint_ablation": True,
            "unknown_rejection_claim": False,
            "l3_claim": False,
            "test_claim": False,
        },
        "scope": {
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "test_payload_access_count": 0,
        },
    }
    path = r22_artifact_path(context.project_root, context.config.artifacts.summary_report_path)
    write_sealed_json_if_identical(path, summary)
    return {**summary, "report_path": str(path)}


__all__ = ["evaluate_r22_seed", "summarize_r22_seeds"]
