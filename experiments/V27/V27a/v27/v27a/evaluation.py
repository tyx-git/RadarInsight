"""Paired known-class evaluation for the V27a evidence-carrier arms."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from v27.v27_baseline.metrics import summarize_parent_groups, summarize_records

from .cache import V27AEvidenceCacheDataset, build_cache_dataloader
from .model import V27AEvidenceResidual
from .protocol import (
    V27A_ALL_ARMS,
    V27A_L2_CLASS_NAMES,
    V27A_REPORT_SCHEMA_VERSION,
    V27AContext,
    v27a_artifact_path,
    write_sealed_json_if_identical,
)
from .runtime import (
    cache_path,
    cache_payload_digest,
    configured_device,
    load_verified_cache,
)


def _softmax(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[-1] != len(V27A_L2_CLASS_NAMES):
        raise ValueError("V27a evaluation logits have an invalid shape")
    shifted = values - np.max(values, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    denominator = exponent.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise FloatingPointError("V27a evaluation softmax denominator is invalid")
    result = exponent / denominator
    if not np.isfinite(result).all():
        raise FloatingPointError("V27a evaluation probabilities are non-finite")
    return result


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    result = np.asarray(value)
    if result.dtype.kind in "fc" and not np.isfinite(result).all():
        raise ValueError(f"V27a evaluation field is non-finite: {name}")
    return result


def _canonical_split_name(split_name: str) -> str:
    value = str(split_name).strip().lower()
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _index_parent_records(
    records: Sequence[Mapping[str, Any]],
    *,
    context: str,
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in records:
        parent_id = str(row.get("parent_sample_id", "")).strip()
        if not parent_id or parent_id in indexed:
            raise ValueError(f"V27a {context} parent records are not uniquely indexed")
        label = row.get("true_id")
        prediction = row.get("pred_id")
        if (
            isinstance(label, bool)
            or isinstance(prediction, bool)
            or not isinstance(label, (int, np.integer))
            or not isinstance(prediction, (int, np.integer))
            or int(label) not in range(len(V27A_L2_CLASS_NAMES))
            or int(prediction) not in range(len(V27A_L2_CLASS_NAMES))
        ):
            raise ValueError(f"V27a {context} parent record class ids differ")
        indexed[parent_id] = row
    if not indexed:
        raise ValueError(f"V27a {context} parent records are empty")
    return indexed


def _parent_group_fields(record: Mapping[str, Any]) -> dict[str, str]:
    count = int(record["final_reliable_frame_count"])
    fraction = float(record["final_reliable_frame_fraction"])
    return {
        "reliable_frame_availability": "none" if count == 0 else "present",
        "reliable_frame_fraction_bucket": (
            "none"
            if count == 0
            else "partial" if fraction < 1.0 else "complete"
        ),
        "rd_presence": "present" if bool(record["final_rd_present"]) else "missing",
    }


def _residual_diagnostics(
    residual: np.ndarray,
    increments: np.ndarray,
    evidence: np.ndarray,
    *,
    padding: np.ndarray,
    reliable: np.ndarray,
    bound: float,
) -> dict[str, Any]:
    if residual.ndim != 3 or residual.shape[-1] != 3:
        raise ValueError("V27a residual diagnostics shape differs")
    active = ~padding
    if not np.isfinite(residual).all() or not np.isfinite(increments).all() or not np.isfinite(evidence).all():
        raise ValueError("V27a residual diagnostics contain non-finite values")
    active_residual = residual[active]
    zero_sum = np.abs(active_residual.sum(axis=-1)) if active_residual.size else np.zeros(0)
    no_reliable = active & ~reliable
    no_reliable_increment = increments[no_reliable]
    no_reliable_evidence = evidence[no_reliable]
    return {
        "active_event_count": int(active.sum()),
        "residual_absolute_max": float(np.abs(active_residual).max()) if active_residual.size else 0.0,
        "residual_bound": float(bound),
        "residual_bound_passed": bool(
            not active_residual.size or np.all(np.abs(active_residual) <= float(bound) + 1.0e-6)
        ),
        "residual_zero_sum_max": float(zero_sum.max()) if zero_sum.size else 0.0,
        "residual_zero_sum_passed": bool(not zero_sum.size or np.all(zero_sum <= 1.0e-6)),
        "no_reliable_event_count": int(no_reliable.sum()),
        "no_reliable_increment_nonzero_count": int(
            np.count_nonzero(np.abs(no_reliable_increment).max(axis=-1) > 0.0)
        ) if no_reliable_increment.size else 0,
        "no_reliable_evidence_nonzero_count": int(
            np.count_nonzero(np.abs(no_reliable_evidence).max(axis=-1) > 0.0)
        ) if no_reliable_evidence.size else 0,
        "no_reliable_exact_zero_passed": bool(
            (not no_reliable_increment.size or np.all(no_reliable_increment == 0.0))
            and (not no_reliable_evidence.size or np.all(no_reliable_evidence == 0.0))
        ),
    }


def _records_from_logits(
    payload: Mapping[str, Any],
    logits: np.ndarray,
    *,
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lengths = _as_numpy(payload["lengths"], name="lengths").astype(np.int64)
    padding = _as_numpy(payload["padding_mask"], name="padding_mask")
    labels = _as_numpy(payload["labels"], name="labels").astype(np.int64)
    baseline = _as_numpy(payload["event_baseline_logits"], name="baseline_logits")
    rd_present = _as_numpy(payload["event_rd_present"], name="event_rd_present")
    counts = _as_numpy(payload["event_reliable_frame_count"], name="frame_count")
    fractions = _as_numpy(
        payload["event_reliable_frame_fraction"], name="frame_fraction"
    )
    if logits.shape != baseline.shape:
        raise ValueError("V27a evaluation logits/cache shapes differ")
    if padding.dtype != np.bool_ or rd_present.dtype != np.bool_:
        raise ValueError("V27a evaluation boolean cache fields have invalid dtype")
    if counts.dtype.kind not in "iu" or fractions.dtype != np.float32:
        raise ValueError("V27a evaluation quality fields have invalid dtype")
    parent_ids = [str(value) for value in payload["parent_sample_ids"]]
    if len(parent_ids) != len(lengths) or len(labels) != len(parent_ids):
        raise ValueError("V27a evaluation parent metadata is misaligned")
    if np.any(labels < 0) or np.any(labels >= len(V27A_L2_CLASS_NAMES)):
        raise ValueError("V27a evaluation labels are out of range")
    probabilities = _softmax(logits.reshape(-1, logits.shape[-1])).reshape(logits.shape)
    parent_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    for row, parent_id in enumerate(parent_ids):
        length = int(lengths[row])
        if length <= 0 or length > logits.shape[1]:
            raise ValueError("V27a evaluation parent length is out of bounds")
        if not np.array_equal(padding[row], np.arange(logits.shape[1]) >= length):
            raise ValueError("V27a evaluation padding is not a suffix")
        label = int(labels[row])
        final = probabilities[row, length - 1]
        parent = {
            "parent_sample_id": parent_id,
            "true_id": label,
            "pred_id": int(np.argmax(final)),
            "probabilities": [float(value) for value in final.tolist()],
            "logits": [float(value) for value in logits[row, length - 1].tolist()],
            "event_count": length,
            "final_reliable_frame_count": int(counts[row, length - 1]),
            "final_reliable_frame_fraction": float(fractions[row, length - 1]),
            "final_rd_present": bool(rd_present[row, length - 1]),
            "arm": arm,
        }
        parent.update(_parent_group_fields(parent))
        parent_records.append(parent)
        for event_index in range(length):
            probability = probabilities[row, event_index]
            event_records.append(
                {
                    "parent_sample_id": parent_id,
                    "true_id": label,
                    "pred_id": int(np.argmax(probability)),
                    "probabilities": [float(value) for value in probability.tolist()],
                    "logits": [float(value) for value in logits[row, event_index].tolist()],
                    "event_index": event_index,
                    "window_length": event_index + 1,
                    "parent_length": length,
                    "reliable_frame_count": int(counts[row, event_index]),
                    "reliable_frame_fraction": float(fractions[row, event_index]),
                    "rd_present": bool(rd_present[row, event_index]),
                    "arm": arm,
                }
            )
    return parent_records, event_records


def _paired_correctness(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = _index_parent_records(control, context="control")
    right = _index_parent_records(treatment, context="treatment")
    if set(left) != set(right):
        raise ValueError("V27a paired arms do not cover the same parents")
    transitions: Counter[str] = Counter()
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if int(a["true_id"]) != int(b["true_id"]):
            raise ValueError("V27a paired arms disagree on a parent label")
        control_correct = int(a["pred_id"]) == int(a["true_id"])
        treatment_correct = int(b["pred_id"]) == int(b["true_id"])
        transitions[f"{'correct' if control_correct else 'wrong'}_to_{'correct' if treatment_correct else 'wrong'}"] += 1
    total = len(left)
    return {
        "parent_count": total,
        "control_correct_count": transitions["correct_to_correct"] + transitions["correct_to_wrong"],
        "treatment_correct_count": transitions["correct_to_correct"] + transitions["wrong_to_correct"],
        "control_to_treatment": dict(sorted(transitions.items())),
        "treatment_accuracy_delta": (
            (transitions["wrong_to_correct"] - transitions["correct_to_wrong"]) / total
            if total
            else None
        ),
    }


def _metric_delta(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
) -> dict[str, float | None]:
    control_metrics = summarize_records(
        control, class_names=V27A_L2_CLASS_NAMES, scope="control_bootstrap"
    )
    treatment_metrics = summarize_records(
        treatment, class_names=V27A_L2_CLASS_NAMES, scope="treatment_bootstrap"
    )
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    return {
        name: (
            float(treatment_metrics[name] - control_metrics[name])
            if control_metrics[name] is not None and treatment_metrics[name] is not None
            else None
        )
        for name in names
    }


def paired_bootstrap(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Stratified paired bootstrap over parent IDs."""

    left = _index_parent_records(control, context="bootstrap control")
    right = _index_parent_records(treatment, context="bootstrap treatment")
    if set(left) != set(right):
        raise ValueError("V27a paired bootstrap parent sets differ")
    if any(
        int(left[parent_id]["true_id"]) != int(right[parent_id]["true_id"])
        for parent_id in left
    ):
        raise ValueError("V27a paired bootstrap parent labels differ")
    groups: dict[int, list[str]] = defaultdict(list)
    for parent_id, row in left.items():
        groups[int(row["true_id"])].append(parent_id)
    if set(groups) != set(range(len(V27A_L2_CLASS_NAMES))) or any(not rows for rows in groups.values()):
        raise ValueError("V27a paired bootstrap requires support for every class")
    count = int(replicates)
    if count < 100:
        raise ValueError("V27a paired bootstrap requires at least 100 replicates")
    rng = np.random.default_rng(int(seed))
    samples: dict[str, list[float]] = {
        name: []
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    }
    for _ in range(count):
        selected_ids: list[str] = []
        for values in groups.values():
            selected_ids.extend(
                values[int(index)]
                for index in rng.integers(0, len(values), size=len(values)).tolist()
            )
        delta = _metric_delta(
            [left[parent_id] for parent_id in selected_ids],
            [right[parent_id] for parent_id in selected_ids],
        )
        for name in samples:
            value = delta[name]
            if value is None or not math.isfinite(value):
                raise FloatingPointError("V27a paired bootstrap produced a non-finite delta")
            samples[name].append(float(value))
    intervals: dict[str, Any] = {}
    for name, values in samples.items():
        array = np.asarray(values, dtype=np.float64)
        intervals[name] = {
            "lower_95": float(np.quantile(array, 0.025)),
            "upper_95": float(np.quantile(array, 0.975)),
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=1)),
        }
    return {
        "method": "stratified_parent_paired_bootstrap",
        "seed": int(seed),
        "replicates": count,
        "class_support": {
            V27A_L2_CLASS_NAMES[class_id]: len(values)
            for class_id, values in sorted(groups.items())
        },
        "metric_deltas_treatment_minus_control": intervals,
    }


def assess_v27a_checkpoint(
    evaluation: Mapping[str, Any],
    *,
    epoch: int,
) -> dict[str, Any]:
    metrics = dict(evaluation.get("parent_metrics") or {})
    rows = {str(row["class"]): dict(row) for row in metrics.get("per_class") or ()}
    if set(rows) != set(V27A_L2_CLASS_NAMES) or int(epoch) <= 0:
        raise ValueError("V27a checkpoint assessment class/epoch differs")
    values = [float(row[field]) for row in rows.values() for field in ("precision", "recall", "f1")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("V27a checkpoint assessment is non-finite")
    matrix = np.asarray(
        metrics["confusion_matrix_rows_true_columns_pred"], dtype=np.int64
    )
    if matrix.shape != (3, 3):
        raise ValueError("V27a checkpoint assessment confusion matrix differs")
    bidirectional = int(matrix[0, 1] + matrix[1, 0])
    minimum = min(float(row["f1"]) for row in rows.values())
    bird_f1 = float(rows["BirdTarget"]["f1"])
    drone_f1 = float(rows["DroneTarget"]["f1"])
    balloon_f1 = float(rows["BalloonTarget"]["f1"])
    macro_f1 = float(metrics["macro_f1"])
    accuracy = float(metrics["accuracy"])
    return {
        "schema_version": "RADAR-LN3-V27a-checkpoint-selector-v2",
        "arm": str(evaluation.get("arm")),
        "epoch": int(epoch),
        "feasible": True,
        "minimum_per_class_precision_recall_f1": float(minimum),
        "bidirectional_drone_bird_errors": bidirectional,
        "per_class_f1": {
            "DroneTarget": drone_f1,
            "BirdTarget": bird_f1,
            "BalloonTarget": balloon_f1,
        },
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "selection_policy": "global_validation_per_class_f1",
        "ranking_key": [
            float(minimum),
            bird_f1,
            drone_f1,
            balloon_f1,
            macro_f1,
            accuracy,
            -float(epoch),
        ],
    }


def evaluate_v27a_arm(
    context: V27AContext,
    *,
    arm: str,
    seed: int,
    split_name: str,
    model: V27AEvidenceResidual | None = None,
    checkpoint_epoch: int | None = None,
    checkpoint_sha256: str | None = None,
    bootstrap_replicates: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Evaluate control or one residual arm on a materialized cache."""

    if arm not in V27A_ALL_ARMS:
        raise ValueError(f"unsupported V27a evaluation arm: {arm}")
    split = _canonical_split_name(split_name)
    payload = load_verified_cache(context, split)
    cache_digest = cache_payload_digest(payload)
    baseline = _as_numpy(payload["event_baseline_logits"], name="baseline_logits").astype(np.float32)
    if arm == "control":
        if checkpoint_epoch is not None and int(checkpoint_epoch) != context.config.base.expected_l2_epoch:
            raise ValueError("V27a control checkpoint epoch differs from the frozen base")
        if checkpoint_sha256 is not None:
            if not _is_sha256(checkpoint_sha256):
                raise ValueError("V27a control checkpoint hash is malformed")
            if checkpoint_sha256.lower() != context.config.base.expected_l2_checkpoint_sha256:
                raise ValueError("V27a control checkpoint hash differs from the frozen base")
        logits = baseline.copy()
        residual = np.zeros_like(logits)
        increments = np.zeros_like(logits)
        evidence = np.zeros((*logits.shape[:2], context.config.model.hidden_dim), dtype=np.float32)
        evaluated_epoch = context.config.base.expected_l2_epoch
        evaluated_checkpoint_sha256 = context.config.base.expected_l2_checkpoint_sha256
        checkpoint_provenance = "frozen_v27_l2_base"
    else:
        if model is None:
            raise ValueError("V27a treatment evaluation requires a model")
        if model.arm != arm:
            raise ValueError("V27a evaluation model/arm differs")
        target_device = configured_device(context) if device is None else device
        model = model.to(target_device)
        model.eval()
        dataset = V27AEvidenceCacheDataset(
            cache_path(context, split)
        )
        loader = build_cache_dataloader(
            dataset,
            batch_size=context.config.training.batch_size,
            shuffle=False,
        )
        logits = np.zeros_like(baseline)
        residual = np.zeros_like(baseline)
        increments = np.zeros_like(baseline)
        evidence = np.zeros(
            (*baseline.shape[:2], context.config.model.hidden_dim), dtype=np.float32
        )
        cursor = 0
        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    name: value.to(device=target_device)
                    for name, value in (
                        ("baseline_logits", batch["baseline_logits"]),
                        ("summary_tokens", batch["summary_tokens"]),
                        ("summary_valid", batch["summary_valid"]),
                        ("frame_values", batch["frame_values"]),
                        ("frame_mask", batch["frame_mask"]),
                        ("padding_mask", batch["padding_mask"]),
                    )
                }
                output = model(**inputs)
                if not torch.equal(output["baseline_logits"], inputs["baseline_logits"]):
                    raise AssertionError("V27a evaluation model altered frozen baseline logits")
                size = int(inputs["baseline_logits"].shape[0])
                lengths = batch["length"].detach().cpu().numpy().astype(np.int64)
                if cursor + size > logits.shape[0]:
                    raise ValueError("V27a evaluation batch traversal exceeds cache")
                logits[cursor : cursor + size] = output["sequence_logits"].cpu().numpy()
                residual[cursor : cursor + size] = output["residual_logits"].cpu().numpy()
                increments[cursor : cursor + size] = output["residual_increment"].cpu().numpy()
                evidence[cursor : cursor + size] = output["evidence_update"].cpu().numpy()
                # The loader is deterministic and parent order is part of the
                # cache contract; lengths are checked to catch accidental
                # reordering even when the tensor shapes still fit.
                expected_lengths = payload["lengths"][cursor : cursor + size]
                if not np.array_equal(lengths, expected_lengths):
                    raise ValueError("V27a evaluation cache parent order differs")
                parent_ids = [str(value) for value in batch["parent_sample_id"]]
                expected_parent_ids = [
                    str(value)
                    for value in payload["parent_sample_ids"][cursor : cursor + size]
                ]
                if parent_ids != expected_parent_ids:
                    raise ValueError("V27a evaluation cache parent IDs differ")
                cursor += size
        if cursor != logits.shape[0]:
            raise ValueError("V27a evaluation did not traverse the complete cache")
        evaluated_epoch = int(checkpoint_epoch or context.config.training.primary_epoch)
        if checkpoint_sha256 is not None and not _is_sha256(checkpoint_sha256):
            raise ValueError("V27a treatment checkpoint hash is malformed")
        evaluated_checkpoint_sha256 = (
            checkpoint_sha256.lower() if checkpoint_sha256 is not None else None
        )
        checkpoint_provenance = (
            "saved_v27a_checkpoint"
            if evaluated_checkpoint_sha256 is not None
            else "in_memory_pre_save"
        )

    parent_records, event_records = _records_from_logits(payload, logits, arm=arm)
    active_padding = _as_numpy(payload["padding_mask"], name="padding_mask")
    summary_valid = _as_numpy(payload["event_summary_valid"], name="summary_valid")
    residual_report = _residual_diagnostics(
        residual,
        increments,
        evidence,
        padding=active_padding,
        reliable=summary_valid,
        bound=context.config.model.residual_bound,
    )
    eligible = [
        row
        for row in event_records
        if int(row["window_length"]) >= context.config.training.eligible_prefix_min_events
        and int(row["window_length"]) < int(row["parent_length"])
    ]
    from v27.v27_baseline.metrics import parent_balanced_prefix_records

    prefix_records = parent_balanced_prefix_records(
        eligible,
        minimum_events=context.config.training.eligible_prefix_min_events,
        class_names=V27A_L2_CLASS_NAMES,
    )
    parent_metrics = summarize_records(
        parent_records,
        class_names=V27A_L2_CLASS_NAMES,
        scope="parent_final",
    )
    prefix_metrics = summarize_records(
        prefix_records,
        class_names=V27A_L2_CLASS_NAMES,
        scope="parent_balanced_eligible_prefix_diagnostic",
    )
    bootstrap = (
        _stratified_bootstrap(parent_records, seed=2700, replicates=bootstrap_replicates)
        if bootstrap_replicates
        else {"method": "not_executed", "replicates": 0}
    )
    grouped = {
        field: summarize_parent_groups(
            parent_records,
            group_field=field,
            class_names=V27A_L2_CLASS_NAMES,
            scope_prefix=f"parent_final:{field}",
        )
        for field in (
            "reliable_frame_availability",
            "reliable_frame_fraction_bucket",
            "rd_presence",
        )
    }
    return {
        "schema_version": V27A_REPORT_SCHEMA_VERSION,
        "experiment_id": "V27a",
        "arm": arm,
        "seed": int(seed),
        "split": split,
        "checkpoint_epoch": int(evaluated_epoch),
        "evaluated_checkpoint_sha256": evaluated_checkpoint_sha256,
        "checkpoint_provenance": checkpoint_provenance,
        "base_checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "base_classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "cache_payload_sha256": cache_digest,
        "class_names": list(V27A_L2_CLASS_NAMES),
        "parent_count": len(parent_records),
        "window_count": len(event_records),
        "parent_metrics": parent_metrics,
        "parent_balanced_eligible_prefix_metrics": prefix_metrics,
        "parent_bootstrap": bootstrap,
        "grouped_parent_metrics": grouped,
        "residual_diagnostics": residual_report,
        "evaluation_scope": {
            "unknown": "sealed_and_not_used",
            "test": "sealed_and_not_used",
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
        },
        "records": {
            "parent": parent_records,
            "eligible_prefix": prefix_records,
            "window": event_records,
        },
    }


def _stratified_bootstrap(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[int(record["true_id"])].append(record)
    if set(groups) != set(range(len(V27A_L2_CLASS_NAMES))):
        raise ValueError("V27a bootstrap requires every class")
    if int(replicates) < 100:
        raise ValueError("V27a bootstrap requires at least 100 replicates")
    rng = np.random.default_rng(int(seed))
    values: dict[str, list[float]] = {
        name: []
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    }
    for _ in range(int(replicates)):
        selected: list[Mapping[str, Any]] = []
        for rows in groups.values():
            indices = rng.integers(0, len(rows), size=len(rows))
            selected.extend(rows[int(index)] for index in indices.tolist())
        metrics = summarize_records(selected, class_names=V27A_L2_CLASS_NAMES, scope="bootstrap")
        for name in values:
            values[name].append(float(metrics[name]))
    return {
        "method": "stratified_parent_bootstrap",
        "seed": int(seed),
        "replicates": int(replicates),
        "metrics": {
            name: {
                "lower_95": float(np.quantile(array, 0.025)),
                "upper_95": float(np.quantile(array, 0.975)),
                "mean": float(np.mean(array)),
                "std": float(np.std(array, ddof=1)),
            }
            for name, array in (
                (name, np.asarray(items, dtype=np.float64))
                for name, items in values.items()
            )
        },
    }


def _validate_formal_arm_report(
    context: V27AContext,
    report: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> str:
    """Verify that one report can enter the immutable primary paired result."""

    if report.get("experiment_id") != "V27a" or report.get("arm") != arm:
        raise ValueError("V27a paired report arm identity differs")
    if report.get("split") != "val" or int(report.get("seed", -1)) != int(seed):
        raise ValueError("V27a paired report split or seed differs")
    if report.get("class_names") != list(V27A_L2_CLASS_NAMES):
        raise ValueError("V27a paired report class contract differs")
    for key, expected in (
        ("base_checkpoint_sha256", context.config.base.expected_l2_checkpoint_sha256),
        ("base_classifier_state_sha256", context.config.base.expected_l2_classifier_state_sha256),
        ("manifest_sha256", context.manifest_sha256),
        ("config_identity_sha256", context.config.config_identity_sha256),
    ):
        if report.get(key) != expected:
            raise ValueError(f"V27a paired report provenance differs at {key}")
    cache_digest = report.get("cache_payload_sha256")
    checkpoint_digest = report.get("evaluated_checkpoint_sha256")
    if not _is_sha256(cache_digest) or not _is_sha256(checkpoint_digest):
        raise ValueError("V27a paired report cache or checkpoint digest is malformed")
    expected_epoch = (
        context.config.base.expected_l2_epoch
        if arm == "control"
        else context.config.training.primary_epoch
    )
    if int(report.get("checkpoint_epoch", -1)) != expected_epoch:
        raise ValueError("V27a paired report does not use the registered primary epoch")
    if arm == "control":
        if (
            checkpoint_digest != context.config.base.expected_l2_checkpoint_sha256
            or report.get("checkpoint_provenance") != "frozen_v27_l2_base"
        ):
            raise ValueError("V27a paired control provenance differs")
    elif report.get("checkpoint_provenance") != "saved_v27a_checkpoint":
        raise ValueError("V27a paired treatment does not reference a saved checkpoint")
    diagnostics = report.get("residual_diagnostics")
    if not isinstance(diagnostics, Mapping) or not all(
        diagnostics.get(key) is True
        for key in (
            "residual_bound_passed",
            "residual_zero_sum_passed",
            "no_reliable_exact_zero_passed",
        )
    ):
        raise ValueError("V27a paired report residual contract failed")
    records = report.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("V27a paired report records are missing")
    _index_parent_records(records.get("parent", ()), context=f"{arm} paired")
    return cache_digest.lower()


def build_paired_report(
    context: V27AContext,
    *,
    control: Mapping[str, Any],
    mechanism: Mapping[str, Any],
    main: Mapping[str, Any],
    seed: int,
    bootstrap_replicates: int | None = None,
) -> dict[str, Any]:
    cache_digests = {
        _validate_formal_arm_report(context, report, arm=arm, seed=seed)
        for arm, report in (
            ("control", control),
            ("mechanism", mechanism),
            ("main", main),
        )
    }
    if len(cache_digests) != 1:
        raise ValueError("V27a paired arms do not share the exact validation cache")
    control_records = control["records"]["parent"]
    mechanism_records = mechanism["records"]["parent"]
    main_records = main["records"]["parent"]
    replicates = (
        context.config.training.bootstrap_replicates
        if bootstrap_replicates is None
        else int(bootstrap_replicates)
    )
    if replicates < 100:
        raise ValueError("V27a paired report requires at least 100 bootstrap replicates")
    return {
        "schema_version": V27A_REPORT_SCHEMA_VERSION,
        "experiment_id": "V27a",
        "status": "paired_known_class_validation",
        "split": "val",
        "seed": int(seed),
        "base_checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "base_classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "validation_cache_payload_sha256": next(iter(cache_digests)),
        "arms": {
            "control": {
                "checkpoint_epoch": control["checkpoint_epoch"],
                "checkpoint_sha256": control["evaluated_checkpoint_sha256"],
                "parent_metrics": control["parent_metrics"],
                "parent_balanced_eligible_prefix_metrics": control[
                    "parent_balanced_eligible_prefix_metrics"
                ],
                "residual_diagnostics": control["residual_diagnostics"],
            },
            "mechanism": {
                "checkpoint_epoch": mechanism["checkpoint_epoch"],
                "checkpoint_sha256": mechanism["evaluated_checkpoint_sha256"],
                "parent_metrics": mechanism["parent_metrics"],
                "parent_balanced_eligible_prefix_metrics": mechanism[
                    "parent_balanced_eligible_prefix_metrics"
                ],
                "residual_diagnostics": mechanism["residual_diagnostics"],
            },
            "main": {
                "checkpoint_epoch": main["checkpoint_epoch"],
                "checkpoint_sha256": main["evaluated_checkpoint_sha256"],
                "parent_metrics": main["parent_metrics"],
                "parent_balanced_eligible_prefix_metrics": main[
                    "parent_balanced_eligible_prefix_metrics"
                ],
                "residual_diagnostics": main["residual_diagnostics"],
            },
        },
        "paired": {
            "mechanism_vs_control": _paired_correctness(control_records, mechanism_records),
            "main_vs_mechanism": _paired_correctness(mechanism_records, main_records),
            "mechanism_vs_control_bootstrap": paired_bootstrap(
                control_records,
                mechanism_records,
                seed=2701,
                replicates=replicates,
            ),
            "main_vs_mechanism_bootstrap": paired_bootstrap(
                mechanism_records,
                main_records,
                seed=2702,
                replicates=replicates,
            ),
        },
        "interpretation_scope": {
            "unknown_rejection": "not_reported",
            "knownness": "not_reported",
            "quality": "not_reported",
            "l3": "not_reported",
            "deployment": "not_eligible",
            "conclusion_scope": (
                "exploratory_single_seed"
                if len(context.config.training.seeds) == 1
                else "multi_seed_confirmation_pending"
            ),
        },
    }


def write_arm_report(
    context: V27AContext,
    report: Mapping[str, Any],
    *,
    arm: str,
    seed: int,
) -> Path:
    expected_epoch = (
        context.config.base.expected_l2_epoch
        if arm == "control"
        else context.config.training.primary_epoch
    )
    if (
        report.get("experiment_id") != "V27a"
        or report.get("arm") != arm
        or int(report.get("seed", -1)) != int(seed)
        or report.get("split") != "val"
        or int(report.get("checkpoint_epoch", -1)) != expected_epoch
    ):
        raise ValueError("V27a immutable arm report identity differs")
    _validate_formal_arm_report(context, report, arm=arm, seed=seed)
    path = v27a_artifact_path(
        context.project_root,
        context.config.artifacts.evaluation_report_template,
        arm=arm,
        seed=seed,
    )
    write_sealed_json_if_identical(path, dict(report))
    return path


def write_paired_report(context: V27AContext, report: Mapping[str, Any]) -> Path:
    path = v27a_artifact_path(
        context.project_root,
        context.config.artifacts.paired_evaluation_report_path,
    )
    write_sealed_json_if_identical(path, dict(report))
    return path


__all__ = ["assess_v27a_checkpoint", "build_paired_report", "evaluate_v27a_arm", "paired_bootstrap", "write_arm_report", "write_paired_report"]