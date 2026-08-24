"""Read-only exact replay report for the frozen V25a gradient audit.

This module intentionally does not call any validation/test evaluator.  It
reconstructs V25a train updates from epoch-boundary checkpoints, compares them
to V25a's sealed telemetry, and decomposes only Clutter-containing batches.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v25b import (
    EXPECTED_EPOCHS,
    EXPECTED_STEPS_PER_EPOCH,
    EXPECTED_TOTAL_STEPS,
    EXPERIMENT_CONTRACT,
    READINESS,
    REPORT as OUTPUT,
    assert_prepared_v25b_binding,
    load_v25a_source,
)
from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    known_gate_device,
    seed_everything,
    sha256_file,
)
from radar_ln3.known_gate_gradient_telemetry import (
    V25A_BASE_LABELS,
    V25A_TASK_LABELS,
)
from radar_ln3.known_gate_v25b_gradient_decomposition import (
    V25B_MODULE_NAMES,
    V25B_REPORT_SCHEMA,
    V25B_VARIANT,
    V25B_DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE,
    V25B_DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE,
    V25B_DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE,
    V25B_DECOMPOSITION_GRADIENT_RELATIVE_L2_TOLERANCE,
    V25B_DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE,
    all_trainable_parameters,
    assert_no_raw_identifiers,
    finite_float,
    gradient_add,
    gradient_cosine,
    gradient_difference_metrics,
    gradient_dot,
    gradient_l2,
    gradients_exactly_equal,
    gradients_from_scalar,
    module_gradient_metrics,
    module_parameter_indices,
    parent_batch_sha256,
    parent_loss_terms_batch_weight_sum,
    snapshot_parameter_gradients,
    v25b_decomposition_tolerance_policy,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _load_checkpoint, _model_inputs, _state_dict_sha256
from train.known_gate_v25a import _restore_training_state


LOSS_ABSOLUTE_TOLERANCE = 2.0e-5
LOSS_RELATIVE_TOLERANCE = 2.0e-5
GRADIENT_NORM_ABSOLUTE_TOLERANCE = 5.0e-3
GRADIENT_NORM_RELATIVE_TOLERANCE = 5.0e-4
STATE_L2_RELATIVE_TOLERANCE = 2.0e-3
STATE_MAX_ABSOLUTE_TOLERANCE = 5.0e-3
DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE = V25B_DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE
# V25a's original weighted final-state CE takes PyTorch's fused weighted-mean
# path, while the diagnostic necessarily exposes per-parent unreduced terms.
# CUDA reduction order therefore prevents bitwise gradient equality even when
# the loss reconstruction is exact.  The source-to-explicit bridge therefore
# uses an absolute L2 floor, global relative L2, and a component guard that is
# relative to the observed gradient peak.  A fixed component-only threshold is
# invalid here: a 1e-4 coordinate discrepancy is material for a 1e-3 gradient,
# but ordinary fused-reduction rounding for a peak near 10.0.
DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE
)
DECOMPOSITION_GRADIENT_RELATIVE_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_RELATIVE_L2_TOLERANCE
)
DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE
)
# Repeated epoch-12/step-468 probes reached 1.08e-05 here; the full epoch-12
# replay's observed maximum was 7.17e-06.  The registered 2e-05 cap therefore
# leaves a bounded ~1.85x margin above the observed source/proxy reduction
# noise while remaining far below a meaningful per-coordinate discrepancy.
DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE
)

# A complete V25b replay performs 9,860 CUDA-backed updates and, on
# Clutter-containing batches, several read-only autograd traversals of the
# historical graph.  The analysis must not depend on a single very long-lived
# CUDA driver context: Windows event logs showed a native nvcuda64 fast-fail
# (0xC0000409), rather than a Python or reconstruction failure.  The parent
# process therefore remains CPU-only and launches one fresh process per frozen
# epoch.  A worker still executes all 493 ordered updates, verifies the epoch
# boundary, and returns only anonymous JSON to its parent.
V25B_ISOLATED_EPOCH_WORKER_SCHEMA = (
    "RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-"
    "isolated-epoch-worker-v1"
)
V25B_WORKER_FAILURE_OUTPUT_LIMIT = 8_000


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> None:
    content = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                "refusing to overwrite a different V25b exact replay report: "
                f"{path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _close(
    observed: float,
    expected: float,
    *,
    absolute: float,
    relative: float,
) -> bool:
    return abs(float(observed) - float(expected)) <= max(
        float(absolute), float(relative) * max(1.0, abs(float(expected)))
    )


def _decomposition_gradient_allowances(
    metrics: Mapping[str, float],
) -> dict[str, float]:
    """Return scale-aware, fail-closed bounds for a diagnostic gradient bridge.

    The final-state source loss uses a fused weighted-mean CUDA reduction,
    whereas V25b must expose parent terms through unreduced CE calls.  These
    are algebraically identical but not required to have bitwise-identical
    backward reductions.  The component guard is therefore measured against
    the largest source-gradient coordinate, while retaining a small absolute
    floor for low-magnitude batches.
    """

    actual_l2 = float(metrics["actual_l2"])
    actual_peak = float(metrics["actual_maximum_absolute_component"])
    if actual_l2 < 0.0 or actual_peak < 0.0:
        raise ValueError("V25b gradient reconstruction scale is invalid")
    return {
        "allowed_l2_residual": max(
            DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE,
            DECOMPOSITION_GRADIENT_RELATIVE_TOLERANCE * actual_l2,
        ),
        "allowed_maximum_absolute_component_residual": max(
            DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE,
            DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE
            * actual_peak,
        ),
        "component_peak_reference": actual_peak,
    }


def _assert_decomposition_gradient_reconstruction(
    metrics: Mapping[str, float],
    *,
    context: str,
) -> dict[str, float]:
    """Reject a source/proxy or parent/proxy gradient mismatch with context."""

    allowances = _decomposition_gradient_allowances(metrics)
    if (
        float(metrics["residual_l2"]) > allowances["allowed_l2_residual"]
        or float(metrics["maximum_absolute_component_residual"])
        > allowances["allowed_maximum_absolute_component_residual"]
    ):
        raise AssertionError(
            "V25b gradient reconstruction failed: "
            f"context={context}; metrics={dict(metrics)}; allowances={allowances}"
        )
    return allowances


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("V25b distribution is empty or non-finite")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p25": float(np.quantile(array, 0.25)),
        "p75": float(np.quantile(array, 0.75)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _optional_distribution(values: Iterable[float]) -> dict[str, Any]:
    values = [float(value) for value in values]
    return _distribution(values) if values else {"count": 0, "status": "empty"}


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _bool_count(value: bool) -> int:
    return int(bool(value))


def _stratum_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"batch_count": 0, "status": "empty"}
    clipped = [row for row in rows if bool(row["was_globally_clipped"])]
    mixed = [row for row in rows if row["pair_kind"] != "ClutterTarget+ClutterTarget"]
    cosines = [float(row["raw_pair_cosine"]) for row in mixed if row["raw_pair_cosine"] is not None]
    historical_cosines = [
        float(row["historical_pair_cosine"])
        for row in mixed
        if row["historical_pair_cosine"] is not None
    ]
    directional_conflicts = sum(
        _bool_count(row["raw_pair_cosine"] is not None and float(row["raw_pair_cosine"]) < 0.0)
        for row in mixed
    )
    return {
        "batch_count": len(rows),
        "clip_count": len(clipped),
        "clip_rate": _rate(len(clipped), len(rows)),
        "historical_total_gradient_l2": _distribution(
            float(row["historical_total_gradient_l2"]) for row in rows
        ),
        "actual_total_gradient_l2": _distribution(
            float(row["actual_total_gradient_l2"]) for row in rows
        ),
        "raw_clutter_gradient_l2": _distribution(
            float(row["raw_clutter_gradient_l2"]) for row in rows
        ),
        "historical_clutter_gradient_l2": _distribution(
            float(row["historical_clutter_gradient_l2"]) for row in rows
        ),
        "raw_known_gradient_l2": _optional_distribution(
            float(row["raw_known_gradient_l2"])
            for row in mixed
            if row["raw_known_gradient_l2"] is not None
        ),
        "historical_known_gradient_l2": _optional_distribution(
            float(row["historical_known_gradient_l2"])
            for row in mixed
            if row["historical_known_gradient_l2"] is not None
        ),
        "raw_pair_cosine": _optional_distribution(cosines),
        "historical_pair_cosine": _optional_distribution(historical_cosines),
        "raw_directional_conflict_count": directional_conflicts,
        "raw_directional_conflict_rate_among_mixed": _rate(
            directional_conflicts, len(mixed)
        ),
        "final_component_clutter_gradient_l2": _distribution(
            float(row["final_clutter_gradient_l2"]) for row in rows
        ),
        "prefix_component_clutter_gradient_l2": _distribution(
            float(row["prefix_clutter_gradient_l2"]) for row in rows
        ),
        "exact_reconstruction_relative_residual": _distribution(
            float(row["gradient_reconstruction"]["relative_residual_l2"])
            for row in rows
        ),
    }


def _module_summary(rows: Sequence[Mapping[str, Any]], *, prefix: str) -> dict[str, Any]:
    if not rows:
        return {"status": "empty"}
    result: dict[str, Any] = {}
    for module in V25B_MODULE_NAMES:
        result[module] = {
            "l2": _distribution(
                float(row[f"{prefix}_module_metrics"][module]["l2"])
                for row in rows
            ),
            "squared_fraction": _distribution(
                float(row[f"{prefix}_module_metrics"][module]["squared_fraction"])
                for row in rows
            ),
        }
    return result


def _observation_key(row: Mapping[str, Any]) -> dict[str, bool]:
    counts = dict(row["batch_observation_counts"])
    active = int(counts["active_event_count"])
    return {
        "has_same_second_event": int(counts["zero_delta_event_count"]) > 0,
        "all_active_events_have_rd": int(counts["rd_present_event_count"]) == active,
        "all_active_events_have_valid_course": int(
            counts["course_feature_valid_event_count"]
        )
        == active,
    }


def _telemetry_for_epoch(source: Mapping[str, Any], epoch: int) -> list[dict[str, Any]]:
    path = source["paths"][f"telemetry_epoch_{epoch:03d}"]
    payload = _read_json(path)
    records = [dict(row) for row in list(payload.get("records") or ())]
    if len(records) != EXPECTED_STEPS_PER_EPOCH:
        raise ValueError("V25b source telemetry epoch has unexpected step count")
    return records


def _tensors_on_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _objective(
    model: torch.nn.Module,
    tensors: Mapping[str, Any],
    *,
    class_weights: torch.Tensor,
    config: Any,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    # This call deliberately matches V25a byte-for-byte at the public model
    # boundary.  Its bounded-observation encoder derives no density ordinal,
    # so manufacturing one here would make an otherwise read-only replay a
    # different forward invocation.
    output = model(**_model_inputs(tensors))
    objective = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        tensors["labels"],
        tensors["tr_padding_mask"],
        class_weights=class_weights,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
        class_weight_normalization=config.train.class_weight_normalization,
    )
    return output, objective


def _parameter_state_difference(
    model: torch.nn.Module,
    expected_state: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    observed_state = model.state_dict()
    if set(observed_state) != set(expected_state):
        raise ValueError("V25b source epoch state has incompatible parameter keys")
    squares: list[torch.Tensor] = []
    maxima: list[torch.Tensor] = []
    expected_squares: list[torch.Tensor] = []
    for name in sorted(observed_state):
        observed = observed_state[name].detach().to(dtype=torch.float64)
        expected = expected_state[name].detach().to(
            device=observed.device, dtype=torch.float64
        )
        if observed.shape != expected.shape:
            raise ValueError("V25b source epoch state tensor shape differs")
        difference = observed - expected
        squares.append(torch.sum(difference.square()))
        expected_squares.append(torch.sum(expected.square()))
        maxima.append(difference.abs().max())
    residual_l2 = float(torch.sqrt(sum(squares)).detach().cpu().item())
    expected_l2 = float(torch.sqrt(sum(expected_squares)).detach().cpu().item())
    maximum = float(torch.stack(maxima).max().detach().cpu().item())
    return {
        "residual_l2": residual_l2,
        "relative_residual_l2": float(residual_l2 / max(expected_l2, 1.0e-12)),
        "maximum_absolute_component_residual": maximum,
    }


def _assert_epoch_state_match(metrics: Mapping[str, float], *, epoch: int) -> None:
    if (
        float(metrics["relative_residual_l2"]) > STATE_L2_RELATIVE_TOLERANCE
        or float(metrics["maximum_absolute_component_residual"])
        > STATE_MAX_ABSOLUTE_TOLERANCE
    ):
        raise AssertionError(
            "V25b exact replay diverged at epoch boundary "
            f"{epoch}: {dict(metrics)}"
        )


def _verify_step_replay(
    *,
    batch: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    objective: Mapping[str, torch.Tensor],
    actual_gradient_l2: float,
) -> dict[str, float]:
    batch_hash = parent_batch_sha256(list(batch["parent_sample_ids"]))
    if batch_hash != str(telemetry["parent_batch_sha256"]):
        raise AssertionError(
            "V25b replay ordered parent-batch hash differs from frozen telemetry"
        )
    task_labels = [str(value) for value in list(batch["task_label_names"])]
    base_labels = [str(value) for value in list(batch["base_label_names"])]
    expected_task_counts = {
        label: task_labels.count(label) for label in V25A_TASK_LABELS
    }
    expected_base_counts = {
        label: base_labels.count(label) for label in V25A_BASE_LABELS
    }
    if (
        expected_task_counts != {
            key: int(value) for key, value in dict(telemetry["task_label_counts"]).items()
        }
        or expected_base_counts != {
            key: int(value) for key, value in dict(telemetry["base_label_counts"]).items()
        }
        or bool(expected_base_counts["ClutterTarget"] > 0)
        != bool(telemetry["contains_clutter_parent"])
    ):
        raise AssertionError("V25b replay label composition differs from frozen telemetry")
    active = ~batch["tr_padding_mask"]
    # ``_verify_step_replay`` receives the CPU collated batch for identifier
    # binding, so derive only metadata available on that batch here.
    reasons = batch["tr_dt_reason"]
    rd_present = batch["rd_frame_mask"].any(dim=2) & active
    course_valid = batch["known_gate_feature_valid"][..., 0] & active
    observation = {
        "active_event_count": int(active.sum().item()),
        "zero_delta_event_count": int((active & (reasons == 2)).sum().item()),
        "positive_delta_event_count": int((active & (reasons == 3)).sum().item()),
        "rd_present_event_count": int(rd_present.sum().item()),
        "course_feature_valid_event_count": int(course_valid.sum().item()),
    }
    if observation != {
        key: int(value)
        for key, value in dict(telemetry["batch_observation_counts"]).items()
    }:
        raise AssertionError("V25b replay observation composition differs from frozen telemetry")
    loss_errors: dict[str, float] = {}
    for objective_key, telemetry_key in (
        ("loss", "total"),
        ("final_state_loss", "final_state"),
        ("prefix_auxiliary_loss", "prefix_auxiliary"),
    ):
        observed = float(objective[objective_key].detach().item())
        expected = float(dict(telemetry["loss"])[telemetry_key])
        error = abs(observed - expected)
        if not _close(
            observed,
            expected,
            absolute=LOSS_ABSOLUTE_TOLERANCE,
            relative=LOSS_RELATIVE_TOLERANCE,
        ):
            raise AssertionError(
                "V25b replay loss differs from frozen telemetry: "
                f"{objective_key} observed={observed} expected={expected}"
            )
        loss_errors[f"{telemetry_key}_absolute_error"] = error
    expected_norm = float(telemetry["pre_clip_global_l2"])
    norm_error = abs(float(actual_gradient_l2) - expected_norm)
    if not _close(
        actual_gradient_l2,
        expected_norm,
        absolute=GRADIENT_NORM_ABSOLUTE_TOLERANCE,
        relative=GRADIENT_NORM_RELATIVE_TOLERANCE,
    ):
        raise AssertionError(
            "V25b replay pre-clip norm differs from frozen telemetry: "
            f"observed={actual_gradient_l2} expected={expected_norm}"
        )
    expected_clipped = bool(telemetry["was_globally_clipped"])
    actual_clipped = bool(actual_gradient_l2 > float(telemetry["gradient_clip_threshold"]))
    if actual_clipped != expected_clipped:
        raise AssertionError("V25b replay clipping decision differs from frozen telemetry")
    return {
        "batch_hash_exact": 1.0,
        "pre_clip_global_l2_absolute_error": norm_error,
        "clip_decision_exact": 1.0,
        "label_composition_exact": 1.0,
        "observation_composition_exact": 1.0,
        **loss_errors,
    }


def _base_pair_kind(base_labels: Sequence[str]) -> str:
    values = sorted(str(value) for value in base_labels)
    allowed = set(V25A_BASE_LABELS)
    if len(values) != 2 or any(value not in allowed for value in values):
        raise ValueError("V25b batch base labels are invalid")
    return "+".join(values)


def _single_parent_gradients(
    terms: Mapping[str, Any], parameters: Sequence[torch.nn.Parameter], index: int) -> dict[str, Any]:
    raw_total = gradients_from_scalar(terms["raw_total_terms"][index], parameters)
    historical_total = gradients_from_scalar(
        terms["historical_total_terms"][index], parameters
    )
    historical_final = gradients_from_scalar(
        terms["historical_final_terms"][index], parameters
    )
    historical_prefix = gradients_from_scalar(
        terms["historical_prefix_terms"][index], parameters
    )
    return {
        "raw_total": raw_total,
        "historical_total": historical_total,
        "historical_final": historical_final,
        "historical_prefix": historical_prefix,
    }


def _aggregate_gradient(
    values: Sequence[Mapping[str, Any]],
    key: str,
) -> tuple[Any, ...] | None:
    gradients = [value[key] for value in values]
    return gradient_add(*gradients) if gradients else None


def _clutter_decomposition_record(
    *,
    batch: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    terms: Mapping[str, Any],
    original_loss: torch.Tensor,
    actual_gradient: tuple[torch.Tensor, ...],
    parameters: Sequence[torch.nn.Parameter],
    module_indices: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    parent_ids = [str(value) for value in batch["parent_sample_ids"]]
    base_labels = [str(value) for value in batch["base_label_names"]]
    task_labels = [str(value) for value in batch["task_label_names"]]
    if len(parent_ids) != 2 or len(base_labels) != 2 or len(task_labels) != 2:
        raise ValueError("V25b expects the frozen batch_size=2 source batch")
    if any(label not in V25A_TASK_LABELS for label in task_labels):
        raise ValueError("V25b batch task label is outside the KnownGate protocol")
    clutter_indices = [index for index, label in enumerate(base_labels) if label == "ClutterTarget"]
    known_indices = [index for index, label in enumerate(base_labels) if label != "ClutterTarget"]
    if not clutter_indices:
        raise ValueError("V25b decomposition requested for a batch without Clutter")
    parent_gradients = [
        _single_parent_gradients(terms, parameters, index)
        for index in range(len(parent_ids))
    ]
    if not gradients_exactly_equal(actual_gradient, snapshot_parameter_gradients(parameters)):
        raise AssertionError("V25b diagnostic autograd calls mutated parameter.grad")
    reconstructed = gradient_add(
        *(value["historical_total"] for value in parent_gradients)
    )
    explicit_proxy = gradients_from_scalar(terms["historical_total_loss"], parameters)
    source_to_parent_reconstruction = gradient_difference_metrics(
        actual_gradient, reconstructed
    )
    source_to_explicit_proxy = gradient_difference_metrics(
        actual_gradient, explicit_proxy
    )
    parent_terms_to_explicit_proxy = gradient_difference_metrics(
        explicit_proxy, reconstructed
    )
    loss_reconstructed = terms["historical_total_loss"]
    loss_error = abs(
        float(loss_reconstructed.detach().item())
        - float(original_loss.detach().item())
    )
    if loss_error > DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE:
        raise AssertionError("V25b decomposed parent loss does not reconstruct the batch loss")
    context = (
        f"epoch={int(telemetry['epoch'])}; step={int(telemetry['step'])}; "
        f"global_step={int(telemetry['global_step'])}; "
        f"pair={_base_pair_kind(base_labels)}"
    )
    source_proxy_allowances = _assert_decomposition_gradient_reconstruction(
        source_to_explicit_proxy,
        context="original_loss_backward_vs_explicit_unreduced_proxy; " + context,
    )
    parent_proxy_allowances = _assert_decomposition_gradient_reconstruction(
        parent_terms_to_explicit_proxy,
        context="sum_of_parent_terms_vs_explicit_unreduced_proxy; " + context,
    )
    source_parent_allowances = _assert_decomposition_gradient_reconstruction(
        source_to_parent_reconstruction,
        context="original_loss_backward_vs_sum_of_parent_terms; " + context,
    )
    clutter_values = [parent_gradients[index] for index in clutter_indices]
    known_values = [parent_gradients[index] for index in known_indices]
    clutter_raw = _aggregate_gradient(clutter_values, "raw_total")
    clutter_historical = _aggregate_gradient(clutter_values, "historical_total")
    clutter_final = _aggregate_gradient(clutter_values, "historical_final")
    clutter_prefix = _aggregate_gradient(clutter_values, "historical_prefix")
    known_raw = _aggregate_gradient(known_values, "raw_total")
    known_historical = _aggregate_gradient(known_values, "historical_total")
    if clutter_raw is None or clutter_historical is None or clutter_final is None or clutter_prefix is None:
        raise AssertionError("V25b Clutter decomposition unexpectedly has no Clutter gradient")
    raw_dot = gradient_dot(known_raw, clutter_raw) if known_raw is not None else None
    raw_cosine = gradient_cosine(known_raw, clutter_raw) if known_raw is not None else None
    historical_dot = (
        gradient_dot(known_historical, clutter_historical)
        if known_historical is not None
        else None
    )
    historical_cosine = (
        gradient_cosine(known_historical, clutter_historical)
        if known_historical is not None
        else None
    )
    row = {
        "epoch": int(telemetry["epoch"]),
        "step": int(telemetry["step"]),
        "global_step": int(telemetry["global_step"]),
        "parent_batch_sha256": parent_batch_sha256(parent_ids),
        "pair_kind": _base_pair_kind(base_labels),
        "task_label_signature": "+".join(sorted(task_labels)),
        "base_label_signature": _base_pair_kind(base_labels),
        "was_globally_clipped": bool(telemetry["was_globally_clipped"]),
        "global_clip_scale": finite_float(
            telemetry["global_clip_scale"], name="global clip scale"
        ),
        "batch_observation_counts": {
            key: int(value)
            for key, value in dict(telemetry["batch_observation_counts"]).items()
        },
        "parent_role_counts": {
            "clutter": len(clutter_indices),
            "known": len(known_indices),
        },
        "loss_coefficients": {
            "final": [
                float(value) for value in terms["final_coefficients"].detach().cpu().tolist()
            ],
            "prefix_normalization": [
                float(value)
                for value in terms["prefix_normalization_coefficients"].detach().cpu().tolist()
            ],
            "prefix_including_lambda": [
                float(value) for value in terms["prefix_coefficients"].detach().cpu().tolist()
            ],
            "prefix_parent_eligible": [
                bool(value)
                for value in terms["prefix_parent_eligible"].detach().cpu().tolist()
            ],
            "final_weight_sum": float(terms["final_weight_sum"].detach().item()),
            "prefix_weight_sum": float(terms["prefix_weight_sum"].detach().item()),
        },
        "actual_total_gradient_l2": gradient_l2(actual_gradient),
        "raw_clutter_gradient_l2": gradient_l2(clutter_raw),
        "historical_clutter_gradient_l2": gradient_l2(clutter_historical),
        "final_clutter_gradient_l2": gradient_l2(clutter_final),
        "prefix_clutter_gradient_l2": gradient_l2(clutter_prefix),
        "raw_known_gradient_l2": gradient_l2(known_raw) if known_raw is not None else None,
        "historical_known_gradient_l2": (
            gradient_l2(known_historical) if known_historical is not None else None
        ),
        "raw_pair_dot": raw_dot,
        "raw_pair_cosine": raw_cosine,
        "historical_pair_dot": historical_dot,
        "historical_pair_cosine": historical_cosine,
        "historical_total_gradient_l2": gradient_l2(reconstructed),
        "actual_module_metrics": module_gradient_metrics(
            actual_gradient, module_indices
        ),
        "historical_clutter_module_metrics": module_gradient_metrics(
            clutter_historical, module_indices
        ),
        "gradient_reconstruction": source_to_parent_reconstruction,
        "gradient_reconstruction_allowances": source_parent_allowances,
        "source_to_explicit_proxy_gradient_reconstruction": source_to_explicit_proxy,
        "source_to_explicit_proxy_gradient_reconstruction_allowances": source_proxy_allowances,
        "parent_terms_to_explicit_proxy_gradient_reconstruction": parent_terms_to_explicit_proxy,
        "parent_terms_to_explicit_proxy_gradient_reconstruction_allowances": parent_proxy_allowances,
        "parent_gradient_terms_do_not_mutate_parameter_grad": True,
        "loss_reconstruction_absolute_error": loss_error,
    }
    assert_no_raw_identifiers(row, raw_parent_ids=parent_ids)
    return row


def _initial_model_and_generator(root: Path, config: Any) -> tuple[Any, torch.optim.Optimizer, torch.Generator, torch.device]:
    seed_everything(config.train.seed)
    model, reference, pairing = build_known_gate_model_for_config(root, config)
    if reference is None or pairing.get("paired_baseline") is not True:
        raise AssertionError("V25b could not reconstruct V25a paired fresh initialization")
    del reference
    device = known_gate_device(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.train.dataloader_seed)
    return model, optimizer, generator, device


def _v25a_readiness_sha256(source: Mapping[str, Any]) -> str:
    """Read the exact readiness hash serialized into V25a checkpoints."""

    path = source["paths"]["readiness"]
    return sha256_file(path)


def _load_epoch_boundary(
    *,
    root: Path,
    source: Mapping[str, Any],
    config: Any,
    epoch_before: int,
) -> tuple[Any, torch.optim.Optimizer, torch.Generator, torch.device]:
    model, optimizer, generator, device = _initial_model_and_generator(root, config)
    checkpoint = source["paths"][f"checkpoint_epoch_{epoch_before:03d}"]
    # V25a's checkpoint binds its readiness file, not the config file.
    readiness_sha256 = _v25a_readiness_sha256(source)
    _load_checkpoint(
        checkpoint,
        model,
        optimizer,
        config=config,
        config_path=source["paths"]["config"],
        readiness_sha256=readiness_sha256,
        map_location=device,
    )
    run_state = _restore_training_state(
        torch.load(checkpoint, map_location=device, weights_only=False), generator
    )
    if (
        int(run_state.get("next_epoch", -1)) != epoch_before + 1
        or int(run_state.get("global_step", -1)) != epoch_before * EXPECTED_STEPS_PER_EPOCH
    ):
        raise AssertionError("V25b restored V25a checkpoint is not the expected epoch boundary")
    return model, optimizer, generator, device


def _replay_epoch(
    *,
    root: Path,
    source: Mapping[str, Any],
    epoch: int,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, float]]:
    config = source["config"]
    if epoch == 1:
        model, optimizer, generator, device = _initial_model_and_generator(root, config)
        expected_initial = str(source["summary"].get("initial_model_state_sha256", ""))
        observed_initial = _state_dict_sha256(model.state_dict())
        if not expected_initial or observed_initial != expected_initial:
            raise AssertionError(
                "V25b fresh replay model state differs from V25a frozen initialization"
            )
    else:
        model, optimizer, generator, device = _load_epoch_boundary(
            root=root, source=source, config=config, epoch_before=epoch - 1
        )
    train_dataset = build_known_gate_dataset(root, config, split_name="train")
    if len(train_dataset) != 986 or train_dataset.label_counts() != {
        "KNOWN_TARGET_EVIDENCE": 953,
        "KNOWN_CLUTTER_EVIDENCE": 33,
    }:
        raise AssertionError("V25b replay train Dataset support differs from V25a")
    train_loader = build_known_gate_dataloader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        generator=generator,
    )
    telemetry_rows = _telemetry_for_epoch(source, epoch)
    if len(train_loader) != EXPECTED_STEPS_PER_EPOCH:
        raise AssertionError("V25b replay DataLoader step count differs from V25a")
    model.train()
    class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
    parameters = all_trainable_parameters(model)
    module_indices = module_parameter_indices(model, parameters)
    decomposition_rows: list[dict[str, Any]] = []
    step_checks: list[dict[str, float]] = []
    for step, (batch, telemetry) in enumerate(zip(train_loader, telemetry_rows, strict=True), start=1):
        tensors = _tensors_on_device(batch, device)
        output, objective = _objective(
            model, tensors, class_weights=class_weights, config=config
        )
        optimizer.zero_grad(set_to_none=True)
        objective["loss"].backward(retain_graph=bool(telemetry["contains_clutter_parent"]))
        actual_gradient = snapshot_parameter_gradients(parameters)
        actual_norm = gradient_l2(actual_gradient)
        checks = _verify_step_replay(
            batch=batch,
            telemetry=telemetry,
            objective=objective,
            actual_gradient_l2=actual_norm,
        )
        checks.update({"epoch": float(epoch), "step": float(step)})
        step_checks.append(checks)
        if bool(telemetry["contains_clutter_parent"]):
            terms = parent_loss_terms_batch_weight_sum(
                output["sequence_logits"],
                tensors["labels"],
                tensors["tr_padding_mask"],
                class_weights=class_weights,
                prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
            )
            recomputed_loss = terms["historical_total_loss"]
            if not _close(
                float(recomputed_loss.detach().item()),
                float(objective["loss"].detach().item()),
                absolute=DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE,
                relative=DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE,
            ):
                raise AssertionError("V25b diagnostic loss differs from original batch loss")
            decomposition_rows.append(
                _clutter_decomposition_record(
                    batch=batch,
                    telemetry=telemetry,
                    terms=terms,
                    original_loss=objective["loss"],
                    actual_gradient=actual_gradient,
                    parameters=parameters,
                    module_indices=module_indices,
                )
            )
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.train.gradient_clip_norm
            ).item()
        )
        if not _close(
            grad_norm,
            actual_norm,
            absolute=GRADIENT_NORM_ABSOLUTE_TOLERANCE,
            relative=GRADIENT_NORM_RELATIVE_TOLERANCE,
        ):
            raise AssertionError("V25b replay clip_norm return differs from cloned pre-clip gradient")
        optimizer.step()
    expected_checkpoint = source["paths"][f"checkpoint_epoch_{epoch:03d}"]
    expected_payload = torch.load(expected_checkpoint, map_location=device, weights_only=False)
    boundary = _parameter_state_difference(model, expected_payload["model_state"])
    _assert_epoch_state_match(boundary, epoch=epoch)
    return decomposition_rows, step_checks, boundary


def _summary_from_rows(
    rows: Sequence[Mapping[str, Any]],
    step_checks: Sequence[Mapping[str, float]],
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    clipped = [row for row in rows if bool(row["was_globally_clipped"])]
    nonclipped = [row for row in rows if not bool(row["was_globally_clipped"])]
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_epoch: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_observation: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "has_same_second_event": {"true": [], "false": []},
        "all_active_events_have_rd": {"true": [], "false": []},
        "all_active_events_have_valid_course": {"true": [], "false": []},
    }
    for row in rows:
        by_pair[str(row["pair_kind"])].append(row)
        by_epoch[str(row["epoch"])].append(row)
        for name, value in _observation_key(row).items():
            by_observation[name]["true" if value else "false"].append(row)
    replay_errors = {
        key: _distribution(float(row[key]) for row in step_checks)
        for key in (
            "total_absolute_error",
            "final_state_absolute_error",
            "prefix_auxiliary_absolute_error",
            "pre_clip_global_l2_absolute_error",
        )
    }
    return {
        "integrity": {
            "replayed_epoch_count": len(boundaries),
            "replayed_step_count": len(step_checks),
            "expected_step_count": EXPECTED_TOTAL_STEPS,
            "every_parent_batch_hash_matched_exactly": all(
                value["batch_hash_exact"] == 1.0 for value in step_checks
            ),
            "every_clip_decision_matched_exactly": all(
                value["clip_decision_exact"] == 1.0 for value in step_checks
            ),
            "replay_absolute_error_distributions": replay_errors,
            "epoch_boundary_state_residuals": dict(boundaries),
            "state_tolerances": {
                "relative_l2": STATE_L2_RELATIVE_TOLERANCE,
                "maximum_absolute_component": STATE_MAX_ABSOLUTE_TOLERANCE,
            },
            "loss_norm_tolerances": {
                "loss_absolute": LOSS_ABSOLUTE_TOLERANCE,
                "loss_relative": LOSS_RELATIVE_TOLERANCE,
                "gradient_norm_absolute": GRADIENT_NORM_ABSOLUTE_TOLERANCE,
                "gradient_norm_relative": GRADIENT_NORM_RELATIVE_TOLERANCE,
            },
            "decomposition_tolerances": v25b_decomposition_tolerance_policy(),
        },
        "clutter_containing_batches": {
            "count": len(rows),
            "clipped_count": len(clipped),
            "not_clipped_count": len(nonclipped),
            "overall": _stratum_summary(rows),
            "clipped": _stratum_summary(clipped),
            "not_clipped": _stratum_summary(nonclipped),
            "by_base_label_pair": {
                name: _stratum_summary(group)
                for name, group in sorted(by_pair.items())
            },
            "by_epoch": {
                name: _stratum_summary(group)
                for name, group in sorted(by_epoch.items(), key=lambda item: int(item[0]))
            },
            "observation_strata": {
                name: {
                    value: _stratum_summary(group)
                    for value, group in values.items()
                }
                for name, values in by_observation.items()
            },
            "actual_gradient_module_breakdown": _module_summary(rows, prefix="actual"),
            "historical_clutter_gradient_module_breakdown": _module_summary(
                rows, prefix="historical_clutter"
            ),
        },
    }


def _isolated_epoch_worker_payload(
    *,
    root: Path,
    source: Mapping[str, Any],
    epoch: int,
) -> dict[str, Any]:
    """Replay one frozen epoch and return JSON-safe anonymous diagnostics."""

    rows, checks, boundary = _replay_epoch(root=root, source=source, epoch=epoch)
    return {
        "schema_version": V25B_ISOLATED_EPOCH_WORKER_SCHEMA,
        "epoch": int(epoch),
        "rows": rows,
        "checks": checks,
        "boundary": boundary,
    }


def _run_isolated_epoch_worker(
    *,
    root: Path,
    epoch: int,
) -> tuple[list[dict[str, Any]], list[dict[str, float]], dict[str, float]]:
    """Run one sealed epoch in a fresh CUDA process and validate its payload."""

    command = [
        sys.executable,
        "-X",
        "utf8",
        "-m",
        "report.known_gate_v25b",
        "--project-root",
        str(root),
        "--isolated-epoch-worker",
        str(epoch),
    ]
    completed = subprocess.run(
        command,
        cwd=root / "src-v3",
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout = str(completed.stdout).strip()
    stderr = str(completed.stderr).strip()
    if completed.returncode != 0:
        raise RuntimeError(
            "V25b isolated epoch worker failed; "
            f"epoch={epoch}; returncode={completed.returncode}; "
            f"stdout={stdout[-V25B_WORKER_FAILURE_OUTPUT_LIMIT:]}; "
            f"stderr={stderr[-V25B_WORKER_FAILURE_OUTPUT_LIMIT:]}"
        )
    try:
        payload = dict(json.loads(stdout))
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "V25b isolated epoch worker returned invalid JSON; "
            f"epoch={epoch}; stdout={stdout[-V25B_WORKER_FAILURE_OUTPUT_LIMIT:]}"
        ) from error
    if (
        payload.get("schema_version") != V25B_ISOLATED_EPOCH_WORKER_SCHEMA
        or int(payload.get("epoch", -1)) != int(epoch)
    ):
        raise RuntimeError(
            "V25b isolated epoch worker identity differs from requested epoch: "
            f"requested={epoch}; payload={payload.get('epoch')}"
        )
    rows = [dict(value) for value in list(payload.get("rows") or ())]
    checks = [dict(value) for value in list(payload.get("checks") or ())]
    boundary = dict(payload.get("boundary") or {})
    if len(checks) != EXPECTED_STEPS_PER_EPOCH:
        raise RuntimeError(
            "V25b isolated epoch worker has an incomplete replay: "
            f"epoch={epoch}; steps={len(checks)}"
        )
    if any(int(row.get("epoch", -1)) != epoch for row in rows):
        raise RuntimeError("V25b isolated epoch worker returned a cross-epoch row")
    expected_boundary = {
        "residual_l2",
        "relative_residual_l2",
        "maximum_absolute_component_residual",
    }
    if set(boundary) != expected_boundary:
        raise RuntimeError("V25b isolated epoch worker boundary schema is invalid")
    return rows, checks, {key: float(boundary[key]) for key in expected_boundary}


def run(project_root: Path, *, isolated_epoch_worker: int | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    source = load_v25a_source(root)
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    if not readiness_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("V25b prepare artifacts are missing; run prepare first")
    readiness = _read_json(readiness_path)
    contract = _read_json(contract_path)
    prepared_source_binding = assert_prepared_v25b_binding(
        root,
        source,
        contract=contract,
        readiness=readiness,
        contract_sha256=sha256_file(contract_path),
    )
    if (
        readiness.get("status") != "passed"
        or readiness.get("variant") != V25B_VARIANT
        or contract.get("schema_version") != "RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-experiment-contract-v1"
        or contract.get("internal_variant") != V25B_VARIANT
        or int(readiness.get("validation_payload_access_count", -1)) != 0
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or int(readiness.get("unknown_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V25b readiness/contract is stale or outside sealed diagnostic scope")
    if isolated_epoch_worker is not None:
        epoch = int(isolated_epoch_worker)
        if epoch not in EXPECTED_EPOCHS:
            raise ValueError("V25b isolated epoch worker is outside the frozen 20-epoch range")
        return _isolated_epoch_worker_payload(root=root, source=source, epoch=epoch)
    all_rows: list[dict[str, Any]] = []
    all_step_checks: list[dict[str, float]] = []
    boundaries: dict[str, dict[str, float]] = {}
    for epoch in EXPECTED_EPOCHS:
        rows, checks, boundary = _run_isolated_epoch_worker(root=root, epoch=epoch)
        all_rows.extend(rows)
        all_step_checks.extend(checks)
        boundaries[str(epoch)] = boundary
    if len(all_step_checks) != EXPECTED_TOTAL_STEPS:
        raise AssertionError("V25b did not replay every frozen V25a update")
    if not all_rows:
        raise AssertionError("V25b expected at least one Clutter-containing source batch")
    report_body = _summary_from_rows(all_rows, all_step_checks, boundaries)
    report = {
        "status": "completed_development_exact_replay_gradient_diagnosis_only",
        "schema_version": V25B_REPORT_SCHEMA,
        "variant": V25B_VARIANT,
        "scope": (
            "V25b is a read-only V25a train-update replay and parent-gradient "
            "decomposition. It makes no training intervention or product-performance claim."
        ),
        "experiment_contract_path": str(contract_path),
        "experiment_contract_sha256": sha256_file(contract_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "source_v25a": {
            "config_path": str(source["paths"]["config"]),
            "config_sha256": source["hashes"]["config"],
            "training_summary_path": str(source["paths"]["summary"]),
            "training_summary_sha256": source["hashes"]["summary"],
            "telemetry_metadata_path": str(source["paths"]["telemetry_metadata"]),
            "telemetry_metadata_sha256": source["hashes"]["telemetry_metadata"],
            "epoch_checkpoint_sha256": source["checkpoint_hashes"],
            "epoch_telemetry_sha256": source["telemetry_hashes"],
            "parent_order_sha256": source["parent_order_hashes"],
        },
        "prepared_source_binding_rechecked_exactly": prepared_source_binding,
        "replay_scope": {
            "train_dataset_constructed_for_frozen_replay": True,
            "validation_dataset_constructed": False,
            "validation_payload_access_count": 0,
            "test_payload_access_count": 0,
            "unknown_payload_access_count": 0,
            "l3_payload_access_count": 0,
            "full_cascade_payload_access_count": 0,
            "checkpoint_written": False,
            "optimizer_state_written": False,
            "checkpoint_selection_performed": False,
        },
        **report_body,
        "per_clutter_batch": all_rows,
        "interpretation_limits": [
            "Observed mixed-batch gradient direction or magnitude is associational, not causal proof of a class-weight, feature, source, or module mechanism.",
            "V25b preserves V25a historical batch_weight_sum semantics and does not test a loss or sampler remedy.",
            "Clutter-containing batches arise from only 33 train Clutter parents, so strata are sparse.",
            "No validation, OOF, test, Unknown, L3, full cascade, rejection, stable, P95, or deployment evidence is produced.",
        ],
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    # V25b retains only the ordered batch digest.  A second structural audit
    # prevents an accidental future report field from slipping raw source IDs
    # into the persisted object.
    forbidden_field_tokens = {
        "parent_sample_id",
        "parent_sample_ids",
        "track_id",
        "capture_id",
        "sample_id",
        "source_name",
    }
    report_json = json.dumps(report, ensure_ascii=False)
    if any(token in report_json for token in forbidden_field_tokens):
        raise AssertionError("V25b report serialization contains a forbidden raw identifier field")
    # The persistent object may contain only hashes, never the transient source
    # parent identifiers from the DataLoader.  This final train-only Dataset
    # construction proves the completed serialized object contains no source
    # parent value while staying outside validation/test/Unknown scope.
    parent_identity_audit_dataset = build_known_gate_dataset(
        root, source["config"], split_name="train"
    )
    source_parent_ids = [str(value) for value in parent_identity_audit_dataset.parent_ids]
    if len(source_parent_ids) != 986 or len(source_parent_ids) != len(set(source_parent_ids)):
        raise AssertionError("V25b source train parent identity audit population is invalid")
    if any(f'"{value}"' in report_json for value in source_parent_ids):
        raise AssertionError("V25b report serialization contains a raw parent identifier")
    assert_no_raw_identifiers(
        report,
        raw_parent_ids=source_parent_ids,
    )
    _write_new_or_identical(root / OUTPUT, report)
    return {**report, "report_path": str(root / OUTPUT)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay V25a updates and report anonymous per-parent gradients."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument(
        "--isolated-epoch-worker",
        type=int,
        default=None,
        help="Internal: replay one sealed epoch in a fresh CUDA process.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.project_root.resolve(),
                isolated_epoch_worker=args.isolated_epoch_worker,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
