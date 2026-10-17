"""Paired validation for r22 checkpoints without mutating frozen r22 code."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from radar_ln3.runtime import relative_project_path, sha256_file
from v27.v27_baseline.metrics import parent_balanced_prefix_records, summarize_records
from v27.v27c.r22.data import (
    R22L2Dataset,
    build_r22_dataloader,
    r22_model_input_kwargs,
    r22_sidecar_kwargs,
)
from v27.v27c.r22.protocol import (
    R22_ARMS,
    R22_FROZEN_ERROR_PARENT_IDS,
    R22_L2_CLASSES,
    R22_V27A_MAIN_CONFUSION,
    r22_artifact_path,
    read_sealed_json as read_r22_sealed_json,
)
from v27.v27c.r22.training import load_r22_checkpoint

from .protocol import (
    R22E_REPORT_SCHEMA_VERSION,
    R22E_SUMMARY_SCHEMA_VERSION,
    R22EContext,
    r22e_report_path,
    r22e_summary_path,
    read_sealed_json,
    write_sealed_json_if_identical,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 3 or logits.shape[-1] != len(R22_L2_CLASSES):
        raise ValueError("V27c-r22e logits have an invalid shape")
    if not np.isfinite(logits).all():
        raise ValueError("V27c-r22e logits are non-finite")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    return values / np.sum(values, axis=-1, keepdims=True)


def _motion_activity(values: np.ndarray, *, length: int) -> np.ndarray:
    """Return an event-level active mask using NumPy 2-compatible reduction."""

    active = np.asarray(values[:length])
    if active.ndim != 2 or active.shape[1] != 9 or not np.isfinite(active).all():
        raise ValueError("V27c-r22e motion sidecar shape/value differs")
    return np.max(np.abs(active), axis=-1) > 0.0


def _motion_fields(
    source_active: np.ndarray, effective_active: np.ndarray
) -> dict[str, Any]:
    """Make source and actual model-input coverage unambiguous in reports."""

    if source_active.ndim != 1 or effective_active.shape != source_active.shape:
        raise ValueError("V27c-r22e motion activity dimensions differ")
    return {
        "source_motion_nonzero_event_count": int(source_active.sum()),
        "source_motion_nonzero_event_fraction": float(source_active.mean()),
        "source_final_motion_nonzero": bool(source_active[-1]),
        "effective_motion_nonzero_event_count": int(effective_active.sum()),
        "effective_motion_nonzero_event_fraction": float(effective_active.mean()),
        "effective_final_motion_nonzero": bool(effective_active[-1]),
        # The legacy-compatible aliases intentionally mean *effective* input,
        # never merely the available source sidecar.
        "motion_nonzero_event_count": int(effective_active.sum()),
        "motion_nonzero_event_fraction": float(effective_active.mean()),
        "final_motion_nonzero": bool(effective_active[-1]),
    }


def _records(
    batch: Mapping[str, Any],
    logits: np.ndarray,
    *,
    arm: str,
    source_motion: np.ndarray,
    effective_motion: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Convert complete parent trajectories to parent and prefix records."""

    padding = batch["tr_padding_mask"].detach().cpu().numpy()
    lengths = batch["lengths"].detach().cpu().numpy().astype(np.int64)
    labels = batch["labels"].detach().cpu().numpy().astype(np.int64)
    probabilities = _softmax(logits)
    expected_motion_shape = logits.shape[:2] + (9,)
    if (
        padding.dtype != np.bool_
        or padding.shape != logits.shape[:2]
        or source_motion.shape != expected_motion_shape
        or effective_motion.shape != expected_motion_shape
    ):
        raise ValueError("V27c-r22e batch/sidecar dimensions differ")
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for row, parent_id in enumerate(batch["parent_sample_ids"]):
        length = int(lengths[row])
        if length <= 0 or not np.array_equal(
            padding[row], np.arange(logits.shape[1]) >= length
        ):
            raise ValueError("V27c-r22e evaluation parent padding differs")
        if bool(np.any(source_motion[row, length:] != 0.0)) or bool(
            np.any(effective_motion[row, length:] != 0.0)
        ):
            raise ValueError("V27c-r22e padded motion is non-zero")
        true_id = int(labels[row])
        if not 0 <= true_id < len(R22_L2_CLASSES):
            raise ValueError("V27c-r22e parent label differs")
        source_active = _motion_activity(source_motion[row], length=length)
        effective_active = _motion_activity(effective_motion[row], length=length)
        motion = _motion_fields(source_active, effective_active)
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
                **motion,
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
                    "probabilities": [
                        float(value) for value in probabilities[row, event]
                    ],
                    "event_index": event,
                    "window_length": event + 1,
                    "parent_length": length,
                    "source_motion_nonzero": bool(source_active[event]),
                    "effective_motion_nonzero": bool(effective_active[event]),
                    "motion_nonzero": bool(effective_active[event]),
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
        raise ValueError("V27c-r22e paired arms have different parent sets")
    transitions: dict[str, int] = defaultdict(int)
    confusion_delta = np.zeros((len(R22_L2_CLASSES), len(R22_L2_CLASSES)), dtype=np.int64)
    historic: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        control_row, treatment_row = left[parent_id], right[parent_id]
        if int(control_row["true_id"]) != int(treatment_row["true_id"]):
            raise ValueError("V27c-r22e paired arm labels differ")
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
                    "motion_final_nonzero": bool(
                        treatment_row["effective_final_motion_nonzero"]
                    ),
                    "motion_nonzero_event_fraction": float(
                        treatment_row["effective_motion_nonzero_event_fraction"]
                    ),
                }
            )
    if {row["parent_sample_id"] for row in historic} != set(R22_FROZEN_ERROR_PARENT_IDS):
        raise AssertionError("V27c-r22e historical Drone-to-Bird error set differs")
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
            raise ValueError("V27c-r22e bootstrap labels differ")
        strata[int(row["true_id"])].append(parent_id)
    if replicates < 100 or any(not group for group in strata.values()):
        raise ValueError("V27c-r22e paired bootstrap lacks class support")
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


def _effective_motion(
    motion_values: torch.Tensor, *, arm: str, motion_mode: str
) -> torch.Tensor:
    if arm not in R22_ARMS or motion_mode not in {"normal", "zero"}:
        raise ValueError("V27c-r22e arm or motion mode is invalid")
    if arm == "control" or motion_mode == "zero":
        return torch.zeros_like(motion_values)
    return motion_values


def _infer(
    context: R22EContext,
    *,
    arm: str,
    seed: int,
    motion_mode: str = "normal",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    model, checkpoint = load_r22_checkpoint(
        context.upstream, arm=arm, seed=seed, device=torch.device("cpu")
    )
    dataset = R22L2Dataset(context.upstream, split="val")
    loader = build_r22_dataloader(
        dataset,
        batch_size=context.upstream.config.training.batch_size,
        shuffle=False,
    )
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            base_inputs = r22_model_input_kwargs(batch, torch.device("cpu"))
            sidecars = r22_sidecar_kwargs(batch, torch.device("cpu"))
            source_motion = sidecars["motion_values"].detach().cpu().numpy().copy()
            effective_motion = _effective_motion(
                sidecars["motion_values"], arm=arm, motion_mode=motion_mode
            )
            sidecars["motion_values"] = effective_motion
            output = model(base_inputs=base_inputs, **sidecars)
            rows, windows = _records(
                batch,
                output["sequence_logits"].detach().cpu().numpy(),
                arm=arm if motion_mode == "normal" else f"{arm}_zero_motion",
                source_motion=source_motion,
                effective_motion=effective_motion.detach().cpu().numpy(),
            )
            parents.extend(rows)
            events.extend(windows)
    if [row["parent_sample_id"] for row in parents] != dataset.parent_ids:
        raise ValueError("V27c-r22e evaluation changed canonical parent order")
    return parents, events, checkpoint


def _cached_v27a_reference(
    context: R22EContext,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    dataset = R22L2Dataset(context.upstream, split="val")
    loader = build_r22_dataloader(
        dataset,
        batch_size=context.upstream.config.training.batch_size,
        shuffle=False,
    )
    parents: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for batch in loader:
        source_motion = batch["motion_values"].detach().cpu().numpy().copy()
        rows, windows = _records(
            batch,
            batch["v27a_main_reference_logits"].detach().cpu().numpy(),
            arm="frozen_v27a_main_reference",
            source_motion=source_motion,
            effective_motion=np.zeros_like(source_motion),
        )
        parents.extend(rows)
        events.extend(windows)
    matrix = summarize_records(
        parents, class_names=R22_L2_CLASSES, scope="parent_final"
    )["confusion_matrix_rows_true_columns_pred"]
    if matrix != [list(row) for row in R22_V27A_MAIN_CONFUSION]:
        raise AssertionError("V27c-r22e cached V27a Main reference confusion differs")
    return parents, events


def _metrics(
    parents: Sequence[Mapping[str, Any]],
    events: Sequence[Mapping[str, Any]],
    context: R22EContext,
) -> dict[str, Any]:
    eligible = [
        row
        for row in events
        if int(row["window_length"]) >= context.upstream.config.training.eligible_prefix_min_events
        and int(row["window_length"]) < int(row["parent_length"])
    ]
    prefixes = parent_balanced_prefix_records(
        eligible,
        minimum_events=context.upstream.config.training.eligible_prefix_min_events,
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


def _training_provenance(
    context: R22EContext, *, arm: str, seed: int, checkpoint: Mapping[str, Any]
) -> dict[str, Any]:
    summary_path = r22_artifact_path(
        context.project_root,
        context.upstream.config.artifacts.training_summary_template,
        arm=arm,
        seed=seed,
    )
    summary = read_r22_sealed_json(
        summary_path,
        schema="RADAR-LN3-V27c-r22-integrated-l2-training-summary-v1",
    )
    checkpoint_path = r22_artifact_path(
        context.project_root,
        context.upstream.config.artifacts.checkpoint_template,
        arm=arm,
        seed=seed,
        epoch=context.upstream.config.training.primary_epoch,
    )
    checkpoint_hash = sha256_file(checkpoint_path)
    if (
        summary.get("status") != "completed_fixed_epoch_matched_l2_retraining"
        or summary.get("experiment_id") != context.upstream.config.experiment_id
        or summary.get("arm") != arm
        or int(summary.get("seed", -1)) != int(seed)
        or int(summary.get("selected_epoch", -1))
        != context.upstream.config.training.primary_epoch
        or summary.get("checkpoint_path")
        != relative_project_path(context.project_root, checkpoint_path)
        or summary.get("checkpoint_sha256") != checkpoint_hash
        or summary.get("checkpoint_state_sha256") != checkpoint.get("model_state_sha256")
        or checkpoint.get("manifest_sha256") != context.upstream.manifest_sha256
    ):
        raise ValueError("V27c-r22e upstream training provenance differs")
    return {
        "training_summary_path": relative_project_path(context.project_root, summary_path),
        "training_summary_file_sha256": sha256_file(summary_path),
        "training_summary_payload_sha256": summary["payload_sha256"],
        "checkpoint_path": relative_project_path(context.project_root, checkpoint_path),
        "checkpoint_file_sha256": checkpoint_hash,
        "checkpoint_state_sha256": checkpoint["model_state_sha256"],
        "selected_epoch": int(summary["selected_epoch"]),
    }


def _motion_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("V27c-r22e motion audit has no parents")
    source = np.asarray(
        [float(row["source_motion_nonzero_event_fraction"]) for row in records],
        dtype=np.float64,
    )
    effective = np.asarray(
        [float(row["effective_motion_nonzero_event_fraction"]) for row in records],
        dtype=np.float64,
    )
    return {
        "parent_count": len(records),
        "mean_source_motion_nonzero_event_fraction": float(source.mean()),
        "mean_effective_motion_nonzero_event_fraction": float(effective.mean()),
        "parents_with_any_source_motion": int(
            sum(int(row["source_motion_nonzero_event_count"]) > 0 for row in records)
        ),
        "parents_with_any_effective_motion": int(
            sum(int(row["effective_motion_nonzero_event_count"]) > 0 for row in records)
        ),
    }


def evaluate_r22e_seed(context: R22EContext, *, seed: int) -> dict[str, Any]:
    """Evaluate one completed r22 seed without changing its training artifacts."""

    if seed not in context.upstream.config.training.seeds:
        raise ValueError("V27c-r22e seed is not registered by frozen r22")
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
    report = {
        "schema_version": R22E_REPORT_SCHEMA_VERSION,
        "experiment_id": context.upstream.config.experiment_id,
        "derived_evaluator_experiment_id": context.manifest["experiment_id"],
        "status": "paired_known_only_integrated_l2_validation_evaluation_correction",
        "seed": int(seed),
        "upstream_r22_manifest_sha256": context.upstream.manifest_sha256,
        "evaluator_manifest_sha256": context.manifest_sha256,
        "class_names": list(R22_L2_CLASSES),
        "training_provenance": {
            "control": _training_provenance(
                context, arm="control", seed=seed, checkpoint=control_checkpoint
            ),
            "motion_treatment": _training_provenance(
                context,
                arm="motion_treatment",
                seed=seed,
                checkpoint=treatment_checkpoint,
            ),
        },
        "correction": context.manifest["correction"],
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
        "paired": _paired_transitions(control_parents, treatment_parents),
        "paired_bootstrap": _paired_bootstrap(
            control_parents,
            treatment_parents,
            seed=27_220 + int(seed),
            replicates=context.upstream.config.training.bootstrap_replicates,
        ),
        "counterfactual": {
            "type": "within_motion_treatment_zeroed_numeric_motion",
            "validity_mask_model_input": False,
            "paired_vs_normal": _paired_transitions(
                treatment_zero_parents, treatment_parents
            ),
            "zero_motion_is_not_expected_to_replay_control_after_independent_training": True,
            "zero_motion_metrics": zero_metrics["parent_metrics"],
        },
        "motion_audit": {
            "frozen_v27a_main_reference": _motion_audit(reference_parents),
            "control": _motion_audit(control_parents),
            "motion_treatment": _motion_audit(treatment_parents),
            "motion_treatment_zero_motion_counterfactual": _motion_audit(
                treatment_zero_parents
            ),
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
    path = r22e_report_path(context.project_root, seed=seed)
    write_sealed_json_if_identical(path, report)
    return {**report, "report_path": str(path)}


def summarize_r22e_seeds(context: R22EContext) -> dict[str, Any]:
    """Aggregate all pre-registered r22 seeds without checkpoint selection."""

    reports: list[dict[str, Any]] = []
    for seed in context.upstream.config.training.seeds:
        report = read_sealed_json(
            r22e_report_path(context.project_root, seed=seed),
            schema=R22E_REPORT_SCHEMA_VERSION,
        )
        if (
            int(report.get("seed", -1)) != seed
            or report.get("evaluator_manifest_sha256") != context.manifest_sha256
            or report.get("upstream_r22_manifest_sha256") != context.upstream.manifest_sha256
        ):
            raise ValueError("V27c-r22e report does not belong to this evaluator manifest")
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
    safety: list[dict[str, Any]] = []
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
        "schema_version": R22E_SUMMARY_SCHEMA_VERSION,
        "experiment_id": context.upstream.config.experiment_id,
        "derived_evaluator_experiment_id": context.manifest["experiment_id"],
        "status": "three_seed_matched_integrated_l2_summary_evaluation_correction",
        "upstream_r22_manifest_sha256": context.upstream.manifest_sha256,
        "evaluator_manifest_sha256": context.manifest_sha256,
        "seeds": list(context.upstream.config.training.seeds),
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
    path = r22e_summary_path(context.project_root)
    write_sealed_json_if_identical(path, summary)
    return {**summary, "report_path": str(path)}


__all__ = ["evaluate_r22e_seed", "summarize_r22e_seeds"]
