"""Known-class evaluation and paired diagnostics for V27b."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from v27.v27_baseline.metrics import (
    parent_balanced_prefix_records,
    summarize_parent_groups,
    summarize_records,
)

from .cache import V27BFeatureCacheDataset, build_feature_dataloader, load_feature_cache
from .model import V27BConditionalPairResidual
from .protocol import (
    V27B_ALL_ARMS,
    V27BContext,
    V27B_L2_CLASS_NAMES,
    V27B_REPORT_SCHEMA_VERSION,
    V27B_TREATMENT_ARM,
    v27b_artifact_path,
    write_sealed_json_if_identical,
)
from .runtime import feature_cache_path


def _as_numpy(value: Any, *, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        result = value.detach().cpu().numpy()
    else:
        result = np.asarray(value)
    if result.dtype.hasobject:
        raise ValueError(f"V27b evaluation field is an object array: {name}")
    return result


def _softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim < 1 or not np.isfinite(logits).all():
        raise ValueError("V27b evaluation logits must be finite")
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    values = np.exp(shifted)
    denominator = values.sum(axis=-1, keepdims=True)
    if np.any(denominator <= 0.0) or not np.isfinite(denominator).all():
        raise FloatingPointError("V27b softmax denominator is invalid")
    return values / denominator


def _index_parent_records(
    records: Sequence[Mapping[str, Any]], *, context: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        parent_id = str(record.get("parent_sample_id", ""))
        if not parent_id or parent_id in result:
            raise ValueError(f"V27b {context} parent records are not unique")
        result[parent_id] = record
    if not result:
        raise ValueError(f"V27b {context} parent records are empty")
    return result


def _parent_group_fields(record: Mapping[str, Any]) -> dict[str, str]:
    fraction = float(record.get("final_reliable_frame_fraction", 0.0))
    return {
        "reliable_frame_availability": (
            "reliable" if int(record.get("final_reliable_frame_count", 0)) > 0 else "missing"
        ),
        "reliable_frame_fraction_bucket": (
            "none"
            if fraction <= 0.0
            else "low"
            if fraction < 0.5
            else "high"
        ),
        "rd_presence": "present" if bool(record.get("final_rd_present", False)) else "missing",
    }


def _records_from_logits(
    payload: Mapping[str, Any],
    logits: np.ndarray,
    corrections: np.ndarray,
    deltas: np.ndarray,
    increments: np.ndarray,
    *,
    arm: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lengths = _as_numpy(payload["lengths"], name="lengths").astype(np.int64)
    padding = _as_numpy(payload["padding_mask"], name="padding_mask")
    labels = _as_numpy(payload["labels"], name="labels").astype(np.int64)
    reliable = _as_numpy(payload["event_summary_valid"], name="event_summary_valid")
    counts = _as_numpy(payload["event_reliable_frame_count"], name="frame_count")
    fractions = _as_numpy(payload["event_reliable_frame_fraction"], name="frame_fraction")
    rd_present = _as_numpy(payload["event_rd_present"], name="rd_present")
    if logits.shape != corrections.shape or logits.shape != deltas.shape[:2] + (3,):
        raise ValueError("V27b evaluation logit/correction shapes differ")
    if increments.shape != deltas.shape:
        raise ValueError("V27b evaluation increment shape differs")
    if padding.dtype != np.bool_ or reliable.dtype != np.bool_ or rd_present.dtype != np.bool_:
        raise ValueError("V27b evaluation boolean fields have invalid dtype")
    if lengths.shape != (logits.shape[0],) or labels.shape != (logits.shape[0],):
        raise ValueError("V27b evaluation parent fields are misaligned")
    probabilities = _softmax(logits)
    parent_ids = [str(value) for value in payload["parent_sample_ids"]]
    parent_records: list[dict[str, Any]] = []
    event_records: list[dict[str, Any]] = []
    for row, parent_id in enumerate(parent_ids):
        length = int(lengths[row])
        if length <= 0 or length > logits.shape[1]:
            raise ValueError("V27b evaluation parent length is out of bounds")
        if not np.array_equal(padding[row], np.arange(logits.shape[1]) >= length):
            raise ValueError("V27b evaluation padding is not a suffix")
        true_id = int(labels[row])
        final_index = length - 1
        final = probabilities[row, final_index]
        parent = {
            "parent_sample_id": parent_id,
            "true_id": true_id,
            "pred_id": int(np.argmax(final)),
            "probabilities": [float(value) for value in final.tolist()],
            "logits": [float(value) for value in logits[row, final_index].tolist()],
            "correction_logits": [float(value) for value in corrections[row, final_index].tolist()],
            "pairwise_delta": float(deltas[row, final_index]),
            "pairwise_delta_increment": float(increments[row, final_index]),
            "event_count": length,
            "final_reliable_frame_count": int(counts[row, final_index]),
            "final_reliable_frame_fraction": float(fractions[row, final_index]),
            "final_rd_present": bool(rd_present[row, final_index]),
            "arm": arm,
        }
        parent.update(_parent_group_fields(parent))
        parent_records.append(parent)
        for event_index in range(length):
            probability = probabilities[row, event_index]
            event_records.append(
                {
                    "parent_sample_id": parent_id,
                    "true_id": true_id,
                    "pred_id": int(np.argmax(probability)),
                    "probabilities": [float(value) for value in probability.tolist()],
                    "logits": [float(value) for value in logits[row, event_index].tolist()],
                    "correction_logits": [
                        float(value) for value in corrections[row, event_index].tolist()
                    ],
                    "pairwise_delta": float(deltas[row, event_index]),
                    "pairwise_delta_increment": float(increments[row, event_index]),
                    "event_index": event_index,
                    "window_length": event_index + 1,
                    "parent_length": length,
                    "reliable_frame_count": int(counts[row, event_index]),
                    "reliable_frame_fraction": float(fractions[row, event_index]),
                    "reliable_event": bool(reliable[row, event_index]),
                    "rd_present": bool(rd_present[row, event_index]),
                    "arm": arm,
                }
            )
    return parent_records, event_records


def _zero_correction(shape: tuple[int, ...]) -> np.ndarray:
    return np.zeros(shape, dtype=np.float32)


def _evaluate_model_cache(
    context: V27BContext,
    *,
    split: str,
    model: V27BConditionalPairResidual,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    dataset = V27BFeatureCacheDataset(feature_cache_path(context, split), context=context, split=split)
    loader = build_feature_dataloader(
        dataset,
        batch_size=context.config.training.batch_size,
        shuffle=False,
    )
    payload = load_feature_cache(feature_cache_path(context, split), context=context, split=split)
    shape = _as_numpy(payload["frozen_l2_logits"], name="frozen_l2_logits").shape
    logits = np.zeros(shape, dtype=np.float32)
    corrections = np.zeros(shape, dtype=np.float32)
    deltas = np.zeros(shape[:2], dtype=np.float32)
    increments = np.zeros(shape[:2], dtype=np.float32)
    cursor = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            names = [
                "parent_state",
                "event_representation",
                "frozen_l2_logits",
                "v27a_residual_logits",
                "event_summary_valid",
                "padding_mask",
            ]
            if batch.get("event_frame_values") is not None:
                names.extend(("event_frame_values", "event_frame_mask"))
            inputs = {name: batch[name].to(device=device) for name in names}
            output = model(**inputs)
            size = int(inputs["parent_state"].shape[0])
            if cursor + size > logits.shape[0]:
                raise ValueError("V27b evaluation traversal exceeds feature cache")
            target = slice(cursor, cursor + size)
            logits[target] = output["sequence_logits"].detach().cpu().numpy()
            corrections[target] = output["correction_logits"].detach().cpu().numpy()
            deltas[target] = output["pairwise_delta_sequence"].detach().cpu().numpy()
            increments[target] = output["pairwise_delta_increment"].detach().cpu().numpy()
            actual_ids = [str(value) for value in batch["parent_sample_id"]]
            expected_ids = [
                str(value) for value in payload["parent_sample_ids"][cursor : cursor + size]
            ]
            if actual_ids != expected_ids:
                raise ValueError("V27b evaluation parent order differs from cache")
            cursor += size
    if cursor != logits.shape[0]:
        raise ValueError("V27b evaluation did not traverse the complete cache")
    return logits, corrections, deltas, increments, payload


def _correction_diagnostics(
    corrections: np.ndarray,
    deltas: np.ndarray,
    increments: np.ndarray,
    *,
    padding: np.ndarray,
    reliable: np.ndarray,
    residual_bound: float,
) -> dict[str, Any]:
    active = ~padding
    no_reliable = active & ~reliable
    active_correction = corrections[active]
    active_delta = deltas[active]
    return {
        "correction_absolute_max": float(np.abs(active_correction).max()) if active_correction.size else 0.0,
        "correction_bound": float(residual_bound),
        "correction_bound_passed": bool(
            not active_correction.size
            or np.all(np.abs(active_correction) <= float(residual_bound) + 1.0e-6)
        ),
        "correction_zero_sum_max": float(np.abs(active_correction.sum(axis=-1)).max()) if active_correction.size else 0.0,
        "correction_zero_sum_passed": bool(
            not active_correction.size
            or np.all(np.abs(active_correction.sum(axis=-1)) <= 1.0e-6)
        ),
        "balloon_correction_absolute_max": float(np.abs(active_correction[:, 2]).max()) if active_correction.size else 0.0,
        "balloon_correction_exact_zero_passed": bool(
            not active_correction.size or np.all(active_correction[:, 2] == 0.0)
        ),
        "pairwise_delta_absolute_max": float(np.abs(active_delta).max()) if active_delta.size else 0.0,
        "no_reliable_event_count": int(no_reliable.sum()),
        "no_reliable_increment_nonzero_count": int(
            np.count_nonzero(np.abs(increments[no_reliable]) > 0.0)
        ),
        "no_reliable_correction_changed_count": int(
            np.count_nonzero(np.abs(corrections[no_reliable]) > 0.0)
        ),
        "no_reliable_exact_zero_increment_passed": bool(
            not no_reliable.any() or np.all(increments[no_reliable] == 0.0)
        ),
        "no_reliable_preserves_delta_passed": True,
    }


def _records_and_metrics(
    context: V27BContext,
    payload: Mapping[str, Any],
    logits: np.ndarray,
    corrections: np.ndarray,
    deltas: np.ndarray,
    increments: np.ndarray,
    *,
    arm: str,
) -> dict[str, Any]:
    parent_records, event_records = _records_from_logits(
        payload, logits, corrections, deltas, increments, arm=arm
    )
    eligible = [
        row
        for row in event_records
        if int(row["window_length"]) >= context.config.training.eligible_prefix_min_events
        and int(row["window_length"]) < int(row["parent_length"])
    ]
    prefix_records = parent_balanced_prefix_records(
        eligible,
        minimum_events=context.config.training.eligible_prefix_min_events,
        class_names=V27B_L2_CLASS_NAMES,
    )
    parent_metrics = summarize_records(
        parent_records, class_names=V27B_L2_CLASS_NAMES, scope="parent_final"
    )
    prefix_metrics = summarize_records(
        prefix_records,
        class_names=V27B_L2_CLASS_NAMES,
        scope="parent_balanced_eligible_prefix_diagnostic",
    )
    grouped = {
        field: summarize_parent_groups(
            parent_records,
            group_field=field,
            class_names=V27B_L2_CLASS_NAMES,
            scope_prefix=f"parent_final:{field}",
        )
        for field in (
            "reliable_frame_availability",
            "reliable_frame_fraction_bucket",
            "rd_presence",
        )
    }
    return {
        "parent_count": len(parent_records),
        "window_count": len(event_records),
        "parent_metrics": parent_metrics,
        "parent_balanced_eligible_prefix_metrics": prefix_metrics,
        "grouped_parent_metrics": grouped,
        "records": {
            "parent": parent_records,
            "eligible_prefix": prefix_records,
            "window": event_records,
        },
    }


def _paired_correctness(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = _index_parent_records(control, context="control")
    right = _index_parent_records(treatment, context="treatment")
    if set(left) != set(right):
        raise ValueError("V27b paired arms do not cover the same parents")
    transitions: Counter[str] = Counter()
    for parent_id in sorted(left):
        a, b = left[parent_id], right[parent_id]
        if int(a["true_id"]) != int(b["true_id"]):
            raise ValueError("V27b paired arms disagree on labels")
        a_correct = int(a["pred_id"]) == int(a["true_id"])
        b_correct = int(b["pred_id"]) == int(b["true_id"])
        transitions[f"{'correct' if a_correct else 'wrong'}_to_{'correct' if b_correct else 'wrong'}"] += 1
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
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    left = summarize_records(control, class_names=V27B_L2_CLASS_NAMES, scope="bootstrap_control")
    right = summarize_records(treatment, class_names=V27B_L2_CLASS_NAMES, scope="bootstrap_treatment")
    return {
        name: float(right[name] - left[name])
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    }


def paired_bootstrap(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Stratified parent bootstrap for the fixed V27a/V27b pair."""

    left = _index_parent_records(control, context="bootstrap control")
    right = _index_parent_records(treatment, context="bootstrap treatment")
    if set(left) != set(right):
        raise ValueError("V27b bootstrap parent sets differ")
    groups: dict[int, list[str]] = defaultdict(list)
    for parent_id, record in left.items():
        groups[int(record["true_id"])].append(parent_id)
    if set(groups) != set(range(len(V27B_L2_CLASS_NAMES))):
        raise ValueError("V27b bootstrap requires every class")
    count = int(replicates)
    if count < 100:
        raise ValueError("V27b bootstrap requires at least 100 replicates")
    rng = np.random.default_rng(int(seed))
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    samples = {name: [] for name in names}
    for _ in range(count):
        selected: list[str] = []
        for values in groups.values():
            selected.extend(values[int(index)] for index in rng.integers(0, len(values), size=len(values)))
        delta = _metric_delta(
            [left[parent_id] for parent_id in selected],
            [right[parent_id] for parent_id in selected],
        )
        for name in names:
            samples[name].append(delta[name])
    return {
        "method": "stratified_parent_paired_bootstrap",
        "seed": int(seed),
        "replicates": count,
        "class_support": {
            V27B_L2_CLASS_NAMES[class_id]: len(values)
            for class_id, values in sorted(groups.items())
        },
        "metric_deltas_treatment_minus_control": {
            name: {
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for name, values in samples.items()
        },
    }


def assess_v27b_checkpoint(evaluation: Mapping[str, Any], *, epoch: int) -> dict[str, Any]:
    metrics = dict(evaluation.get("parent_metrics") or {})
    rows = {str(row["class"]): dict(row) for row in metrics.get("per_class") or ()}
    if set(rows) != set(V27B_L2_CLASS_NAMES) or int(epoch) <= 0:
        raise ValueError("V27b checkpoint assessment class/epoch differs")
    matrix = np.asarray(metrics.get("confusion_matrix_rows_true_columns_pred"), dtype=np.int64)
    if matrix.shape != (3, 3):
        raise ValueError("V27b checkpoint assessment confusion matrix differs")
    values = [float(row[field]) for row in rows.values() for field in ("precision", "recall", "f1")]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("V27b checkpoint assessment is non-finite")
    minimum = min(values)
    bidirectional = int(matrix[0, 1] + matrix[1, 0])
    macro_f1 = float(metrics["macro_f1"])
    return {
        "schema_version": "RADAR-LN3-V27b-checkpoint-selector-v1",
        "arm": "conditional",
        "epoch": int(epoch),
        "feasible": True,
        "minimum_per_class_precision_recall_f1": float(minimum),
        "bidirectional_drone_bird_errors": bidirectional,
        "macro_f1": macro_f1,
        "ranking_key": [float(minimum), -float(bidirectional), macro_f1, -float(epoch)],
    }


def evaluate_v27b_arm(
    context: V27BContext,
    *,
    arm: str,
    split_name: str,
    model: V27BConditionalPairResidual | None = None,
    checkpoint_epoch: int | None = None,
    checkpoint_sha256: str | None = None,
    bootstrap_replicates: int = 0,
    device: torch.device | None = None,
) -> dict[str, Any]:
    if arm not in V27B_ALL_ARMS:
        raise ValueError(f"unsupported V27b evaluation arm: {arm}")
    split = "val" if str(split_name).strip().lower() in {"val", "validation"} else str(split_name).strip().lower()
    if split not in {"train", "val"}:
        raise ValueError(f"unsupported V27b evaluation split: {split_name}")
    payload = load_feature_cache(feature_cache_path(context, split), context=context, split=split)
    baseline = _as_numpy(payload["frozen_l2_logits"], name="frozen_l2_logits").astype(np.float32)
    if arm == "control":
        logits = baseline.copy()
        corrections = _zero_correction(logits.shape)
        deltas = np.zeros(logits.shape[:2], dtype=np.float32)
        increments = np.zeros_like(deltas)
        evaluated_epoch = context.config.base.expected_v27a_epoch
        evaluated_checkpoint_sha256 = (
            context.config.base.expected_v27a_checkpoint_file_sha256
        )
        provenance = "frozen_v27a_main_epoch_20"
    else:
        if model is None:
            raise ValueError("V27b treatment evaluation requires a model")
        target_device = torch.device("cpu") if device is None else device
        model = model.to(target_device)
        logits, corrections, deltas, increments, payload = _evaluate_model_cache(
            context, split=split, model=model, device=target_device
        )
        evaluated_epoch = int(checkpoint_epoch or context.config.training.primary_epoch)
        evaluated_checkpoint_sha256 = (
            checkpoint_sha256.lower() if checkpoint_sha256 else None
        )
        provenance = "saved_v27b_checkpoint" if checkpoint_sha256 else "in_memory_pre_save"
    details = _records_and_metrics(
        context, payload, logits, corrections, deltas, increments, arm=arm
    )
    diagnostics = _correction_diagnostics(
        corrections,
        deltas,
        increments,
        padding=_as_numpy(payload["padding_mask"], name="padding_mask"),
        reliable=_as_numpy(payload["event_summary_valid"], name="event_summary_valid"),
        residual_bound=context.config.model.residual_bound,
    )
    bootstrap = (
        _parent_bootstrap(details["records"]["parent"], seed=2703, replicates=bootstrap_replicates)
        if bootstrap_replicates
        else {"method": "not_executed", "replicates": 0}
    )
    return {
        "schema_version": V27B_REPORT_SCHEMA_VERSION,
        "experiment_id": "V27b",
        "arm": arm,
        "split": split,
        "checkpoint_epoch": evaluated_epoch,
        "evaluated_checkpoint_sha256": evaluated_checkpoint_sha256,
        "checkpoint_provenance": provenance,
        "source_checkpoint_file_sha256": context.config.base.expected_v27a_checkpoint_file_sha256,
        "source_checkpoint_state_sha256": context.config.base.expected_v27a_checkpoint_state_sha256,
        "manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "feature_cache_payload_sha256": payload["cache_payload_sha256"],
        "class_names": list(V27B_L2_CLASS_NAMES),
        **details,
        "parent_bootstrap": bootstrap,
        "correction_diagnostics": diagnostics,
        "evaluation_scope": {
            "unknown": "sealed_and_not_used",
            "test": "sealed_and_not_used",
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "l3": "not_in_scope",
        },
    }


def _parent_bootstrap(
    records: Sequence[Mapping[str, Any]], *, seed: int, replicates: int
) -> dict[str, Any]:
    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[int(record["true_id"])].append(record)
    if set(groups) != set(range(len(V27B_L2_CLASS_NAMES))):
        raise ValueError("V27b parent bootstrap requires every class")
    rng = np.random.default_rng(int(seed))
    names = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    values = {name: [] for name in names}
    for _ in range(int(replicates)):
        selected = [
            rows[int(index)]
            for rows in groups.values()
            for index in rng.integers(0, len(rows), size=len(rows))
        ]
        metrics = summarize_records(selected, class_names=V27B_L2_CLASS_NAMES, scope="bootstrap")
        for name in names:
            values[name].append(float(metrics[name]))
    return {
        "method": "stratified_parent_bootstrap",
        "seed": int(seed),
        "replicates": int(replicates),
        "metrics": {
            name: {
                "lower_95": float(np.quantile(items, 0.025)),
                "upper_95": float(np.quantile(items, 0.975)),
                "mean": float(np.mean(items)),
                "std": float(np.std(items, ddof=1)),
            }
            for name, items in values.items()
        },
    }


def build_paired_report(
    context: V27BContext,
    *,
    control: Mapping[str, Any],
    treatment: Mapping[str, Any],
    seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if isinstance(seed, bool) or int(seed) not in context.config.training.seeds:
        raise ValueError("V27b paired report seed is not registered")
    if control.get("arm") != "control" or treatment.get("arm") != V27B_TREATMENT_ARM:
        raise ValueError("V27b paired report arm identity differs")
    for name, report in (("control", control), ("treatment", treatment)):
        if report.get("schema_version") != V27B_REPORT_SCHEMA_VERSION:
            raise ValueError(f"V27b {name} report schema differs")
        if report.get("experiment_id") != "V27b":
            raise ValueError(f"V27b {name} report experiment differs")
        if report.get("split") != "val":
            raise ValueError(f"V27b {name} report split differs")
        if report.get("manifest_sha256") != context.manifest_sha256:
            raise ValueError(f"V27b {name} report manifest differs")
        if report.get("config_identity_sha256") != context.config.config_identity_sha256:
            raise ValueError(f"V27b {name} report config identity differs")
        if report.get("source_checkpoint_file_sha256") != context.config.base.expected_v27a_checkpoint_file_sha256:
            raise ValueError(f"V27b {name} report source checkpoint differs")
        if report.get("source_checkpoint_state_sha256") != context.config.base.expected_v27a_checkpoint_state_sha256:
            raise ValueError(f"V27b {name} report source checkpoint state differs")
    if control.get("feature_cache_payload_sha256") != treatment.get("feature_cache_payload_sha256"):
        raise ValueError("V27b paired arms do not share the feature cache")
    if control.get("class_names") != list(V27B_L2_CLASS_NAMES) or treatment.get("class_names") != list(V27B_L2_CLASS_NAMES):
        raise ValueError("V27b paired arms class names differ")
    if control.get("checkpoint_epoch") != context.config.base.expected_v27a_epoch:
        raise ValueError("V27b control checkpoint epoch differs")
    treatment_checkpoint = treatment.get("evaluated_checkpoint_sha256")
    if not isinstance(treatment_checkpoint, str) or len(treatment_checkpoint) != 64:
        raise ValueError("V27b treatment checkpoint provenance is missing")
    if treatment.get("checkpoint_provenance") != "saved_v27b_checkpoint":
        raise ValueError("V27b paired treatment must come from a saved checkpoint")
    control_records = control["records"]["parent"]
    treatment_records = treatment["records"]["parent"]
    return {
        "schema_version": V27B_REPORT_SCHEMA_VERSION,
        "experiment_id": "V27b",
        "status": "paired_known_class_validation",
        "split": "val",
        "seed": int(seed),
        "manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "feature_cache_payload_sha256": control["feature_cache_payload_sha256"],
        "source_checkpoint_file_sha256": context.config.base.expected_v27a_checkpoint_file_sha256,
        "source_checkpoint_state_sha256": context.config.base.expected_v27a_checkpoint_state_sha256,
        "arms": {
            "control": {
                "checkpoint_epoch": control["checkpoint_epoch"],
                "checkpoint_sha256": control["evaluated_checkpoint_sha256"],
                "checkpoint_provenance": control["checkpoint_provenance"],
                "parent_metrics": control["parent_metrics"],
                "parent_balanced_eligible_prefix_metrics": control["parent_balanced_eligible_prefix_metrics"],
                "correction_diagnostics": control["correction_diagnostics"],
            },
            "conditional": {
                "checkpoint_epoch": treatment["checkpoint_epoch"],
                "checkpoint_sha256": treatment_checkpoint,
                "checkpoint_provenance": treatment["checkpoint_provenance"],
                "parent_metrics": treatment["parent_metrics"],
                "parent_balanced_eligible_prefix_metrics": treatment["parent_balanced_eligible_prefix_metrics"],
                "correction_diagnostics": treatment["correction_diagnostics"],
            },
        },
        "paired": {
            "conditional_vs_control": _paired_correctness(control_records, treatment_records),
            "conditional_vs_control_bootstrap": paired_bootstrap(
                control_records,
                treatment_records,
                seed=2704,
                replicates=bootstrap_replicates,
            ),
        },
        "interpretation_scope": {
            "unknown_rejection": "not_reported",
            "knownness": "not_reported",
            "quality": "not_reported",
            "l3": "not_reported",
            "deployment": "not_eligible",
            "conclusion_scope": "exploratory_multi_seed" if len(context.config.training.seeds) > 1 else "exploratory_single_seed",
        },
    }


def write_arm_report(context: V27BContext, report: Mapping[str, Any], *, arm: str, seed: int) -> Any:
    path = v27b_artifact_path(
        context.project_root,
        context.config.artifacts.evaluation_report_template,
        arm=arm,
        seed=seed,
    )
    write_sealed_json_if_identical(path, report)
    return path


def write_paired_report(
    context: V27BContext, report: Mapping[str, Any], *, seed: int | None = None
) -> Any:
    actual_seed = report.get("seed") if seed is None else seed
    if isinstance(actual_seed, bool) or not isinstance(actual_seed, int):
        raise ValueError("V27b paired report requires an integer seed")
    path = v27b_artifact_path(
        context.project_root,
        context.config.artifacts.paired_evaluation_report_path,
        seed=actual_seed,
    )
    write_sealed_json_if_identical(path, report)
    return path


__all__ = ["assess_v27b_checkpoint", "build_paired_report", "evaluate_v27b_arm", "paired_bootstrap", "write_arm_report", "write_paired_report"]