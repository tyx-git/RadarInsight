"""Read-only V25c exact replay module-gradient conflict localization report.

The source is the completed V25a training run.  V25c neither trains nor
selects a checkpoint: it reconstructs each frozen train update in a fresh CUDA
worker and, for mixed Known/Clutter batches only, records anonymous scalar
conflict metrics for the full model and its sealed module partition.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
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
    load_v25a_source,
)
from build.known_gate_v25c import (
    EXPECTED_MIXED_KNOWN_CLUTTER_BATCHES,
    COMPLETION,
    EXPERIMENT_CONTRACT,
    READINESS,
    REPORT as OUTPUT,
    assert_prepared_v25c_binding,
)
from radar_ln3.data import DT_REASON_FIRST_EVENT
from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    known_gate_device,
    seed_everything,
    sha256_file,
)
from radar_ln3.known_gate_gradient_telemetry import V25A_BASE_LABELS, V25A_TASK_LABELS
from radar_ln3.known_gate_v25b_gradient_decomposition import (
    V25B_DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE,
    V25B_DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE,
    V25B_DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE,
    V25B_DECOMPOSITION_GRADIENT_RELATIVE_L2_TOLERANCE,
    V25B_DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE,
    all_trainable_parameters,
    assert_no_raw_identifiers,
    finite_float,
    gradient_add,
    gradient_difference_metrics,
    gradient_l2,
    gradients_exactly_equal,
    gradients_from_scalar,
    module_parameter_indices,
    parent_batch_sha256,
    parent_loss_terms_batch_weight_sum,
    snapshot_parameter_gradients,
)
from radar_ln3.known_gate_v25c_module_gradient_conflict import (
    V25C_COMPONENT_NAMES,
    V25C_COMPLETION_SCHEMA,
    V25C_WORKER_FAILURE_OUTPUT_LIMIT,
    V25C_WORKER_PAYLOAD_MAX_BYTES,
    V25C_WORKER_STDERR_READ_CHUNK_BYTES,
    V25C_MODULE_NAMES,
    V25C_PRIMARY_COMPONENT,
    V25C_REPORT_SCHEMA,
    V25C_SHARED_MODULE_NAMES,
    V25C_VARIANT,
    module_and_full_pair_conflict_metrics,
    extend_parent_components_with_raw_final_prefix,
    parent_component_additivity_metrics,
    parent_source_bridge_component_gradients,
    v25c_component_additivity_tolerance_policy,
    v25c_execution_resource_safety_policy,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_common import _load_checkpoint, _model_inputs, _state_dict_sha256
from train.known_gate_v25a import _restore_training_state


LOSS_ABSOLUTE_TOLERANCE = 2.0e-5
LOSS_RELATIVE_TOLERANCE = 2.0e-5
GRADIENT_NORM_ABSOLUTE_TOLERANCE = 5.0e-3
GRADIENT_NORM_RELATIVE_TOLERANCE = 5.0e-4
STATE_L2_RELATIVE_TOLERANCE = 2.0e-3
STATE_MAX_ABSOLUTE_TOLERANCE = 5.0e-3
DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE = V25B_DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE
DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE
)
DECOMPOSITION_GRADIENT_RELATIVE_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_RELATIVE_L2_TOLERANCE
)
DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE
)
DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE = (
    V25B_DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE
)
V25C_ISOLATED_EPOCH_WORKER_SCHEMA = (
    "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-"
    "localization-isolated-epoch-worker-v1"
)
V25C_REPLAY_ERROR_KEYS = (
    "total_absolute_error",
    "final_state_absolute_error",
    "prefix_auxiliary_absolute_error",
    "pre_clip_global_l2_absolute_error",
)


def _write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically persist a report without duplicating its JSON in RAM.

    V25c's per-mixed-batch section is intentionally rich.  Building one giant
    ``json.dumps`` string and then reading an existing report into a second
    giant string would create avoidable CPU-memory spikes.  Stream to a sibling
    temporary file, compare hashes in blocks, then atomically promote it.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(payload), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        if path.exists():
            if sha256_file(path) != sha256_file(temporary):
                raise FileExistsError(
                    "refusing to overwrite a different V25c exact replay report: "
                    f"{path}"
                )
            return
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_worker_payload(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write one bounded epoch sidecar for the coordinating process."""

    if path.exists():
        raise FileExistsError(
            "V25c isolated epoch worker refuses to overwrite an existing sidecar: "
            f"{path}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(payload), stream, ensure_ascii=False, separators=(",", ":"))
            stream.write("\n")
        size = int(temporary.stat().st_size)
        if not 0 < size <= V25C_WORKER_PAYLOAD_MAX_BYTES:
            raise ValueError(
                "V25c isolated epoch worker payload exceeds its frozen transport limit: "
                f"bytes={size}; limit={V25C_WORKER_PAYLOAD_MAX_BYTES}"
            )
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _close(observed: float, expected: float, *, absolute: float, relative: float) -> bool:
    return abs(float(observed) - float(expected)) <= max(
        float(absolute), float(relative) * max(1.0, abs(float(expected)))
    )


def _distribution(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("V25c distribution is empty or non-finite")
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


class _StreamingDistribution:
    """Bounded-memory distribution collector for frozen replay error scalars.

    Each epoch has a fixed 493 steps.  Keeping four short numeric arrays is
    sufficient for exact quantiles and avoids retaining one dict per step for
    all 9,860 updates in both a worker and its coordinating parent.
    """

    def __init__(self, *, keys: Sequence[str]) -> None:
        self._keys = tuple(str(key) for key in keys)
        if not self._keys or len(self._keys) != len(set(self._keys)):
            raise ValueError("V25c streaming distribution keys are invalid")
        self._values: dict[str, list[float]] = {key: [] for key in self._keys}
        self._all_exact: dict[str, bool] = {key: True for key in self._keys}

    def add(self, values: Mapping[str, float]) -> None:
        if set(values) != set(self._keys):
            raise ValueError("V25c streaming replay-error keys differ from protocol")
        for key in self._keys:
            value = float(values[key])
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("V25c streaming replay error is invalid")
            self._values[key].append(value)
            self._all_exact[key] = self._all_exact[key] and value == 0.0

    def merge(self, other: Mapping[str, Any]) -> None:
        if tuple(other.get("keys") or ()) != self._keys:
            raise ValueError("V25c worker replay-error key order differs from protocol")
        arrays = dict(other.get("values") or {})
        exact = dict(other.get("all_exact") or {})
        if set(arrays) != set(self._keys) or set(exact) != set(self._keys):
            raise ValueError("V25c worker replay-error payload is incomplete")
        for key in self._keys:
            values = [float(value) for value in list(arrays[key])]
            if any(not np.isfinite(value) or value < 0.0 for value in values):
                raise ValueError("V25c worker replay-error payload contains an invalid value")
            self._values[key].extend(values)
            self._all_exact[key] = self._all_exact[key] and bool(exact[key])

    def serialized(self) -> dict[str, Any]:
        return {
            "keys": list(self._keys),
            "values": {key: list(self._values[key]) for key in self._keys},
            "all_exact": dict(self._all_exact),
        }

    def count(self) -> int:
        counts = {len(values) for values in self._values.values()}
        if len(counts) != 1:
            raise AssertionError("V25c streaming replay-error arrays have inconsistent counts")
        return next(iter(counts))

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {key: _distribution(values) for key, values in self._values.items()}

    def every_exact(self, key: str) -> bool:
        if key not in self._all_exact:
            raise KeyError(f"V25c unknown replay-error exactness key: {key}")
        return bool(self._all_exact[key])


def _optional_distribution(values: Iterable[float]) -> dict[str, Any]:
    collected = [float(value) for value in values]
    return _distribution(collected) if collected else {"count": 0, "status": "empty"}


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _decomposition_gradient_allowances(metrics: Mapping[str, float]) -> dict[str, float]:
    actual_l2 = float(metrics["actual_l2"])
    actual_peak = float(metrics["actual_maximum_absolute_component"])
    if actual_l2 < 0.0 or actual_peak < 0.0:
        raise ValueError("V25c gradient reconstruction scale is invalid")
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
    metrics: Mapping[str, float], *, context: str
) -> dict[str, float]:
    allowances = _decomposition_gradient_allowances(metrics)
    if (
        float(metrics["residual_l2"]) > allowances["allowed_l2_residual"]
        or float(metrics["maximum_absolute_component_residual"])
        > allowances["allowed_maximum_absolute_component_residual"]
    ):
        raise AssertionError(
            "V25c gradient reconstruction failed: "
            f"context={context}; metrics={dict(metrics)}; allowances={allowances}"
        )
    return allowances


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


def _initial_model_and_generator(
    root: Path, config: Any
) -> tuple[Any, torch.optim.Optimizer, torch.Generator, torch.device]:
    seed_everything(config.train.seed)
    model, reference, pairing = build_known_gate_model_for_config(root, config)
    if reference is None or pairing.get("paired_baseline") is not True:
        raise AssertionError("V25c could not reconstruct V25a paired fresh initialization")
    del reference
    device = known_gate_device(config)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.train.dataloader_seed)
    return model, optimizer, generator, device


def _v25a_readiness_sha256(source: Mapping[str, Any]) -> str:
    return sha256_file(source["paths"]["readiness"])


def _load_epoch_boundary(
    *, root: Path, source: Mapping[str, Any], config: Any, epoch_before: int
) -> tuple[Any, torch.optim.Optimizer, torch.Generator, torch.device]:
    model, optimizer, generator, device = _initial_model_and_generator(root, config)
    checkpoint = source["paths"][f"checkpoint_epoch_{epoch_before:03d}"]
    _load_checkpoint(
        checkpoint,
        model,
        optimizer,
        config=config,
        config_path=source["paths"]["config"],
        readiness_sha256=_v25a_readiness_sha256(source),
        # Preserve V25b/V25a's CUDA checkpoint restoration path exactly.  The
        # optimizer state and model weights affect CUDA allocation and kernel
        # selection order; moving only this load to CPU changed the diagnostic
        # floating-point bridge on the same frozen source batch. Per-epoch
        # worker isolation and the explicit post-epoch release still bound the
        # lifetime of this transient allocation.
        map_location=device,
    )
    run_state = _restore_training_state(
        torch.load(checkpoint, map_location=device, weights_only=False), generator
    )
    if (
        int(run_state.get("next_epoch", -1)) != epoch_before + 1
        or int(run_state.get("global_step", -1)) != epoch_before * EXPECTED_STEPS_PER_EPOCH
    ):
        raise AssertionError("V25c restored V25a checkpoint is not the expected epoch boundary")
    return model, optimizer, generator, device


def _parameter_state_difference(
    model: torch.nn.Module, expected_state: Mapping[str, torch.Tensor]
) -> dict[str, float]:
    observed_state = model.state_dict()
    if set(observed_state) != set(expected_state):
        raise ValueError("V25c source epoch state has incompatible parameter keys")
    squares: list[torch.Tensor] = []
    expected_squares: list[torch.Tensor] = []
    maxima: list[torch.Tensor] = []
    for name in sorted(observed_state):
        observed = observed_state[name].detach().to(dtype=torch.float64)
        expected = expected_state[name].detach().to(device=observed.device, dtype=torch.float64)
        if observed.shape != expected.shape:
            raise ValueError("V25c source epoch state tensor shape differs")
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
            "V25c exact replay diverged at epoch boundary "
            f"{epoch}: {dict(metrics)}"
        )


def _telemetry_for_epoch(source: Mapping[str, Any], epoch: int) -> list[dict[str, Any]]:
    payload = _read_json(source["paths"][f"telemetry_epoch_{epoch:03d}"])
    records = [dict(row) for row in list(payload.get("records") or ())]
    if len(records) != EXPECTED_STEPS_PER_EPOCH:
        raise ValueError("V25c source telemetry epoch has unexpected step count")
    return records


def _verify_step_replay(
    *,
    batch: Mapping[str, Any],
    telemetry: Mapping[str, Any],
    objective: Mapping[str, torch.Tensor],
    actual_gradient_l2: float,
) -> dict[str, float]:
    batch_hash = parent_batch_sha256(list(batch["parent_sample_ids"]))
    if batch_hash != str(telemetry["parent_batch_sha256"]):
        raise AssertionError("V25c replay ordered parent-batch hash differs from frozen telemetry")
    task_labels = [str(value) for value in list(batch["task_label_names"])]
    base_labels = [str(value) for value in list(batch["base_label_names"])]
    expected_task_counts = {label: task_labels.count(label) for label in V25A_TASK_LABELS}
    expected_base_counts = {label: base_labels.count(label) for label in V25A_BASE_LABELS}
    if (
        expected_task_counts
        != {key: int(value) for key, value in dict(telemetry["task_label_counts"]).items()}
        or expected_base_counts
        != {key: int(value) for key, value in dict(telemetry["base_label_counts"]).items()}
        or bool(expected_base_counts["ClutterTarget"] > 0)
        != bool(telemetry["contains_clutter_parent"])
    ):
        raise AssertionError("V25c replay label composition differs from frozen telemetry")
    active = ~batch["tr_padding_mask"]
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
        key: int(value) for key, value in dict(telemetry["batch_observation_counts"]).items()
    }:
        raise AssertionError("V25c replay observation composition differs from frozen telemetry")
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
                "V25c replay loss differs from frozen telemetry: "
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
            "V25c replay pre-clip norm differs from frozen telemetry: "
            f"observed={actual_gradient_l2} expected={expected_norm}"
        )
    expected_clipped = bool(telemetry["was_globally_clipped"])
    actual_clipped = bool(actual_gradient_l2 > float(telemetry["gradient_clip_threshold"]))
    if actual_clipped != expected_clipped:
        raise AssertionError("V25c replay clipping decision differs from frozen telemetry")
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
    if len(values) != 2 or any(value not in V25A_BASE_LABELS for value in values):
        raise ValueError("V25c batch base labels are invalid")
    return "+".join(values)


def _role_index(base_labels: Sequence[str], role: str) -> int:
    if role == "clutter":
        matches = [index for index, label in enumerate(base_labels) if label == "ClutterTarget"]
    elif role == "known":
        matches = [index for index, label in enumerate(base_labels) if label != "ClutterTarget"]
    else:
        raise ValueError("V25c parent role is invalid")
    if len(matches) != 1:
        raise ValueError("V25c mixed batch does not contain exactly one requested role")
    return matches[0]


def _role_course_valid_summary(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Summarize course validity after the expected first-event invalidity."""

    active = ~batch["tr_padding_mask"][index]
    noninitial = active & (batch["tr_dt_reason"][index] != DT_REASON_FIRST_EVENT)
    denominator = int(noninitial.sum().item())
    valid = batch["known_gate_feature_valid"][index, :, 0] & noninitial
    numerator = int(valid.sum().item())
    if denominator == 0:
        return {
            "noninitial_active_event_count": 0,
            "course_valid_noninitial_event_count": 0,
            "course_valid_noninitial_event_fraction": None,
        }
    return {
        "noninitial_active_event_count": denominator,
        "course_valid_noninitial_event_count": numerator,
        "course_valid_noninitial_event_fraction": float(numerator / denominator),
    }


def _role_observation_summary(batch: Mapping[str, Any], index: int) -> dict[str, Any]:
    active = ~batch["tr_padding_mask"][index]
    reasons = batch["tr_dt_reason"][index]
    rd_present = batch["rd_frame_mask"][index].any(dim=1) & active
    active_count = int(active.sum().item())
    if active_count <= 0:
        raise ValueError("V25c mixed parent has no active event")
    course = _role_course_valid_summary(batch, index)
    return {
        "active_event_count": active_count,
        "zero_delta_event_count": int((active & (reasons == 2)).sum().item()),
        "positive_delta_event_count": int((active & (reasons == 3)).sum().item()),
        "rd_present_event_fraction": float(int(rd_present.sum().item()) / active_count),
        **course,
    }


def _component_additivity_allowances(metrics: Mapping[str, float]) -> dict[str, float]:
    """Return the separately registered tolerance for parent component sums."""

    policy = v25c_component_additivity_tolerance_policy()
    actual_l2 = float(metrics["actual_l2"])
    actual_peak = float(metrics["actual_maximum_absolute_component"])
    if actual_l2 < 0.0 or actual_peak < 0.0:
        raise ValueError("V25c component additivity scale is invalid")
    return {
        "allowed_l2_residual": max(
            float(policy["absolute_l2"]), float(policy["relative_l2"]) * actual_l2
        ),
        "allowed_maximum_absolute_component_residual": max(
            float(policy["maximum_absolute_component_floor"]),
            float(policy["maximum_component_relative_to_parent_total_peak"]) * actual_peak,
        ),
        "component_peak_reference": actual_peak,
    }


def _assert_component_additivity(
    metrics: Mapping[str, float], *, context: str
) -> dict[str, float]:
    allowances = _component_additivity_allowances(metrics)
    if (
        float(metrics["residual_l2"]) > allowances["allowed_l2_residual"]
        or float(metrics["maximum_absolute_component_residual"])
        > allowances["allowed_maximum_absolute_component_residual"]
    ):
        raise AssertionError(
            "V25c parent component additivity failed: "
            f"context={context}; metrics={dict(metrics)}; allowances={allowances}"
        )
    return allowances


def _additivity_allowances(
    metrics: Mapping[str, Mapping[str, float]], *, context: str
) -> dict[str, dict[str, float]]:
    return {
        name: _assert_component_additivity(value, context=f"{context}; {name}")
        for name, value in metrics.items()
    }


def _mixed_conflict_record(
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
        raise ValueError("V25c expects the frozen batch_size=2 source batch")
    if any(label not in V25A_TASK_LABELS for label in task_labels):
        raise ValueError("V25c batch task label is outside the KnownGate protocol")
    if base_labels.count("ClutterTarget") != 1:
        raise ValueError("V25c conflict record requires exactly one Clutter parent")
    known_index = _role_index(base_labels, "known")
    clutter_index = _role_index(base_labels, "clutter")
    known_label = str(base_labels[known_index])
    if known_label not in {"DroneTarget", "BirdTarget", "BalloonTarget"}:
        raise ValueError("V25c mixed batch Known parent has an invalid base label")
    # First reproduce V25b's source bridge in the original frozen batch order.
    # Do not insert V25c-only raw final/prefix traversals between these calls:
    # the bridge's contract is equivalence to V25b, not merely algebraic
    # closeness after an arbitrary diagnostic traversal schedule.
    source_components_by_index = {
        index: parent_source_bridge_component_gradients(terms, parameters, index)
        for index in range(len(parent_ids))
    }
    if not gradients_exactly_equal(actual_gradient, snapshot_parameter_gradients(parameters)):
        raise AssertionError("V25c diagnostic autograd calls mutated parameter.grad")
    reconstructed = gradient_add(
        *(source_components_by_index[index]["historical_total"] for index in range(len(parent_ids)))
    )
    explicit_proxy = gradients_from_scalar(terms["historical_total_loss"], parameters)
    source_to_parent = gradient_difference_metrics(actual_gradient, reconstructed)
    source_to_proxy = gradient_difference_metrics(actual_gradient, explicit_proxy)
    parent_to_proxy = gradient_difference_metrics(explicit_proxy, reconstructed)
    context = (
        f"epoch={int(telemetry['epoch'])}; step={int(telemetry['step'])}; "
        f"global_step={int(telemetry['global_step'])}; pair={_base_pair_kind(base_labels)}"
    )
    source_to_parent_allowances = _assert_decomposition_gradient_reconstruction(
        source_to_parent, context="original_loss_backward_vs_parent_sum; " + context
    )
    source_to_proxy_allowances = _assert_decomposition_gradient_reconstruction(
        source_to_proxy, context="original_loss_backward_vs_explicit_proxy; " + context
    )
    parent_to_proxy_allowances = _assert_decomposition_gradient_reconstruction(
        parent_to_proxy, context="parent_sum_vs_explicit_proxy; " + context
    )
    loss_error = abs(
        float(terms["historical_total_loss"].detach().item())
        - float(original_loss.detach().item())
    )
    if loss_error > DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE:
        raise AssertionError("V25c decomposed parent loss does not reconstruct the batch loss")

    # Only after the V25b-equivalent bridge has passed do we query the two
    # additional V25c raw components. Their values never feed backwards into
    # the already validated source bridge.
    components_by_index = {
        index: extend_parent_components_with_raw_final_prefix(
            source_components_by_index[index], terms, parameters, index
        )
        for index in range(len(parent_ids))
    }
    if not gradients_exactly_equal(actual_gradient, snapshot_parameter_gradients(parameters)):
        raise AssertionError("V25c extended diagnostic autograd calls mutated parameter.grad")
    known_components = components_by_index[known_index]
    clutter_components = components_by_index[clutter_index]
    known_additivity = parent_component_additivity_metrics(known_components)
    clutter_additivity = parent_component_additivity_metrics(clutter_components)
    known_additivity_allowances = _additivity_allowances(
        known_additivity, context="known parent component additivity; " + context
    )
    clutter_additivity_allowances = _additivity_allowances(
        clutter_additivity, context="clutter parent component additivity; " + context
    )
    component_conflicts = {
        component: module_and_full_pair_conflict_metrics(
            known_components[component], clutter_components[component], module_indices
        )
        for component in V25C_COMPONENT_NAMES
    }
    primary = component_conflicts[V25C_PRIMARY_COMPONENT]
    row = {
        "epoch": int(telemetry["epoch"]),
        "step": int(telemetry["step"]),
        "global_step": int(telemetry["global_step"]),
        "parent_batch_sha256": parent_batch_sha256(parent_ids),
        "pair_kind": _base_pair_kind(base_labels),
        "known_base_label": known_label,
        "task_label_signature": "+".join(sorted(task_labels)),
        "was_globally_clipped": bool(telemetry["was_globally_clipped"]),
        "global_clip_scale": finite_float(
            telemetry["global_clip_scale"], name="global clip scale"
        ),
        "known_observation": _role_observation_summary(batch, known_index),
        "clutter_observation": _role_observation_summary(batch, clutter_index),
        "loss_coefficients": {
            "final": [float(value) for value in terms["final_coefficients"].detach().cpu().tolist()],
            "prefix_normalization": [
                float(value)
                for value in terms["prefix_normalization_coefficients"].detach().cpu().tolist()
            ],
            "prefix_including_lambda": [
                float(value) for value in terms["prefix_coefficients"].detach().cpu().tolist()
            ],
            "prefix_parent_eligible": [
                bool(value) for value in terms["prefix_parent_eligible"].detach().cpu().tolist()
            ],
            "final_weight_sum": float(terms["final_weight_sum"].detach().item()),
            "prefix_weight_sum": float(terms["prefix_weight_sum"].detach().item()),
        },
        "actual_total_gradient_l2": gradient_l2(actual_gradient),
        "component_conflicts": component_conflicts,
        "primary_component": V25C_PRIMARY_COMPONENT,
        "primary_full_model_conflict": primary["full_model"],
        "primary_shared_module_conflicts": {
            module: primary["modules"][module] for module in V25C_SHARED_MODULE_NAMES
        },
        "primary_classifier_conflict": primary["modules"]["classifier"],
        "gradient_reconstruction": source_to_parent,
        "gradient_reconstruction_allowances": source_to_parent_allowances,
        "source_to_explicit_proxy_gradient_reconstruction": source_to_proxy,
        "source_to_explicit_proxy_gradient_reconstruction_allowances": source_to_proxy_allowances,
        "parent_terms_to_explicit_proxy_gradient_reconstruction": parent_to_proxy,
        "parent_terms_to_explicit_proxy_gradient_reconstruction_allowances": parent_to_proxy_allowances,
        "known_component_additivity": known_additivity,
        "known_component_additivity_allowances": known_additivity_allowances,
        "clutter_component_additivity": clutter_additivity,
        "clutter_component_additivity_allowances": clutter_additivity_allowances,
        "parent_gradient_terms_do_not_mutate_parameter_grad": True,
        "loss_reconstruction_absolute_error": loss_error,
    }
    assert_no_raw_identifiers(row, raw_parent_ids=parent_ids)
    return row


def _replay_epoch(
    *, root: Path, source: Mapping[str, Any], epoch: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    config = source["config"]
    if epoch == 1:
        model, optimizer, generator, device = _initial_model_and_generator(root, config)
        expected_initial = str(source["summary"].get("initial_model_state_sha256", ""))
        if not expected_initial or _state_dict_sha256(model.state_dict()) != expected_initial:
            raise AssertionError("V25c fresh replay model state differs from V25a frozen initialization")
    else:
        model, optimizer, generator, device = _load_epoch_boundary(
            root=root, source=source, config=config, epoch_before=epoch - 1
        )
    train_dataset = build_known_gate_dataset(root, config, split_name="train")
    if len(train_dataset) != 986 or train_dataset.label_counts() != {
        "KNOWN_TARGET_EVIDENCE": 953,
        "KNOWN_CLUTTER_EVIDENCE": 33,
    }:
        raise AssertionError("V25c replay train Dataset support differs from V25a")
    train_loader = build_known_gate_dataloader(
        train_dataset, batch_size=config.train.batch_size, shuffle=True, generator=generator
    )
    telemetry_rows = _telemetry_for_epoch(source, epoch)
    if len(train_loader) != EXPECTED_STEPS_PER_EPOCH:
        raise AssertionError("V25c replay DataLoader step count differs from V25a")
    model.train()
    class_weights = torch.tensor(config.train.class_weights, dtype=torch.float32, device=device)
    parameters = all_trainable_parameters(model)
    module_indices = module_parameter_indices(model, parameters)
    rows: list[dict[str, Any]] = []
    replay_errors = _StreamingDistribution(keys=V25C_REPLAY_ERROR_KEYS)
    every_parent_batch_hash_matched_exactly = True
    every_clip_decision_matched_exactly = True
    for step, (batch, telemetry) in enumerate(zip(train_loader, telemetry_rows, strict=True), start=1):
        tensors = _tensors_on_device(batch, device)
        output, objective = _objective(model, tensors, class_weights=class_weights, config=config)
        base_labels = [str(value) for value in batch["base_label_names"]]
        is_mixed = base_labels.count("ClutterTarget") == 1
        optimizer.zero_grad(set_to_none=True)
        objective["loss"].backward(retain_graph=is_mixed)
        actual_gradient = snapshot_parameter_gradients(parameters)
        actual_norm = gradient_l2(actual_gradient)
        check = _verify_step_replay(
            batch=batch,
            telemetry=telemetry,
            objective=objective,
            actual_gradient_l2=actual_norm,
        )
        replay_errors.add({key: float(check[key]) for key in V25C_REPLAY_ERROR_KEYS})
        every_parent_batch_hash_matched_exactly = (
            every_parent_batch_hash_matched_exactly
            and check["batch_hash_exact"] == 1.0
        )
        every_clip_decision_matched_exactly = (
            every_clip_decision_matched_exactly
            and check["clip_decision_exact"] == 1.0
        )
        if is_mixed:
            terms = parent_loss_terms_batch_weight_sum(
                output["sequence_logits"],
                tensors["labels"],
                tensors["tr_padding_mask"],
                class_weights=class_weights,
                prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
                prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
            )
            if not _close(
                float(terms["historical_total_loss"].detach().item()),
                float(objective["loss"].detach().item()),
                absolute=DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE,
                relative=DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE,
            ):
                raise AssertionError("V25c diagnostic loss differs from original batch loss")
            rows.append(
                _mixed_conflict_record(
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.gradient_clip_norm).item()
        )
        if not _close(
            grad_norm,
            actual_norm,
            absolute=GRADIENT_NORM_ABSOLUTE_TOLERANCE,
            relative=GRADIENT_NORM_RELATIVE_TOLERANCE,
        ):
            raise AssertionError("V25c replay clip_norm return differs from cloned pre-clip gradient")
        optimizer.step()
        # ``retain_graph`` is necessary only for the current mixed-batch
        # diagnostic.  Explicitly sever all local references after its scalar
        # metrics have been copied so a later non-mixed update cannot retain
        # an earlier CUDA graph through Python locals.
        if is_mixed:
            del terms
        del output, objective, actual_gradient, tensors, batch, telemetry, check
    expected_payload = torch.load(
        source["paths"][f"checkpoint_epoch_{epoch:03d}"],
        map_location="cpu",
        weights_only=False,
    )
    boundary = _parameter_state_difference(model, expected_payload["model_state"])
    _assert_epoch_state_match(boundary, epoch=epoch)
    del (
        expected_payload,
        class_weights,
        parameters,
        module_indices,
        train_loader,
        train_dataset,
        model,
        optimizer,
        telemetry_rows,
        generator,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return (
        rows,
        {
            "step_count": EXPECTED_STEPS_PER_EPOCH,
            "replay_errors": replay_errors.serialized(),
            "every_parent_batch_hash_matched_exactly": every_parent_batch_hash_matched_exactly,
            "every_clip_decision_matched_exactly": every_clip_decision_matched_exactly,
        },
        boundary,
    )


def _observation_stratum(row: Mapping[str, Any]) -> dict[str, bool]:
    known = dict(row["known_observation"])
    clutter = dict(row["clutter_observation"])
    known_noninitial = int(known["noninitial_active_event_count"])
    clutter_noninitial = int(clutter["noninitial_active_event_count"])
    return {
        "either_parent_has_same_second_event": (
            int(known["zero_delta_event_count"]) > 0
            or int(clutter["zero_delta_event_count"]) > 0
        ),
        "known_parent_has_same_second_event": int(known["zero_delta_event_count"]) > 0,
        "clutter_parent_has_same_second_event": int(clutter["zero_delta_event_count"]) > 0,
        "both_parents_all_rd": (
            float(known["rd_present_event_fraction"]) == 1.0
            and float(clutter["rd_present_event_fraction"]) == 1.0
        ),
        "known_parent_all_rd": float(known["rd_present_event_fraction"]) == 1.0,
        "clutter_parent_all_rd": float(clutter["rd_present_event_fraction"]) == 1.0,
        "known_parent_has_noninitial_event": known_noninitial > 0,
        "clutter_parent_has_noninitial_event": clutter_noninitial > 0,
        "known_parent_full_noninitial_course_coverage": (
            known_noninitial > 0
            and float(known["course_valid_noninitial_event_fraction"]) == 1.0
        ),
        "clutter_parent_full_noninitial_course_coverage": (
            clutter_noninitial > 0
            and float(clutter["course_valid_noninitial_event_fraction"]) == 1.0
        ),
    }


def _observation_value_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Expose continuous coverage rather than only binary report strata."""

    if not rows:
        return {"batch_count": 0, "status": "empty"}
    metrics = (
        "active_event_count",
        "zero_delta_event_count",
        "positive_delta_event_count",
        "rd_present_event_fraction",
        "noninitial_active_event_count",
        "course_valid_noninitial_event_count",
    )
    result: dict[str, Any] = {"batch_count": len(rows)}
    for role in ("known", "clutter"):
        values = [dict(row[f"{role}_observation"]) for row in rows]
        result[role] = {
            name: _distribution(float(value[name]) for value in values)
            for name in metrics
        }
        result[role]["course_valid_noninitial_event_fraction"] = _optional_distribution(
            float(value["course_valid_noninitial_event_fraction"])
            for value in values
            if value["course_valid_noninitial_event_fraction"] is not None
        )
    return result


def _component_summary(rows: Sequence[Mapping[str, Any]], component: str) -> dict[str, Any]:
    if not rows:
        return {"batch_count": 0, "status": "empty"}
    full = [dict(row["component_conflicts"])[component]["full_model"] for row in rows]
    modules = {
        module: [dict(row["component_conflicts"])[component]["modules"][module] for row in rows]
        for module in V25C_MODULE_NAMES
    }
    def pair_summary(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        negative = [value for value in values if bool(value["negative_cosine"])]
        return {
            "batch_count": len(values),
            "negative_cosine_count": len(negative),
            "negative_cosine_rate": _rate(len(negative), len(values)),
            "known_gradient_l2": _distribution(float(value["known_gradient_l2"]) for value in values),
            "clutter_gradient_l2": _distribution(float(value["clutter_gradient_l2"]) for value in values),
            "cosine": _optional_distribution(
                float(value["cosine"]) for value in values if value["cosine"] is not None
            ),
            "cancellation_fraction": _distribution(
                float(value["cancellation_fraction"]) for value in values
            ),
            "known_antialigned_projection_fraction": _distribution(
                float(value["known_antialigned_projection_fraction"]) for value in values
            ),
            "clutter_antialigned_projection_fraction": _distribution(
                float(value["clutter_antialigned_projection_fraction"]) for value in values
            ),
        }
    return {
        "full_model": pair_summary(full),
        "modules": {module: pair_summary(values) for module, values in modules.items()},
    }


def _summary_from_rows(
    rows: Sequence[Mapping[str, Any]],
    replay_errors: _StreamingDistribution,
    *,
    replayed_step_count: int,
    every_parent_batch_hash_matched_exactly: bool,
    every_clip_decision_matched_exactly: bool,
    boundaries: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    by_known_label: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_epoch: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_clip: dict[str, list[Mapping[str, Any]]] = {"clipped": [], "not_clipped": []}
    by_observation: dict[str, dict[str, list[Mapping[str, Any]]]] = {
        "either_parent_has_same_second_event": {"true": [], "false": []},
        "known_parent_has_same_second_event": {"true": [], "false": []},
        "clutter_parent_has_same_second_event": {"true": [], "false": []},
        "both_parents_all_rd": {"true": [], "false": []},
        "known_parent_all_rd": {"true": [], "false": []},
        "clutter_parent_all_rd": {"true": [], "false": []},
        "known_parent_has_noninitial_event": {"true": [], "false": []},
        "clutter_parent_has_noninitial_event": {"true": [], "false": []},
        "known_parent_full_noninitial_course_coverage": {"true": [], "false": []},
        "clutter_parent_full_noninitial_course_coverage": {"true": [], "false": []},
    }
    for row in rows:
        by_known_label[str(row["known_base_label"])].append(row)
        by_epoch[str(row["epoch"])].append(row)
        by_clip["clipped" if bool(row["was_globally_clipped"]) else "not_clipped"].append(row)
        for name, value in _observation_stratum(row).items():
            by_observation[name]["true" if value else "false"].append(row)
    if replay_errors.count() != replayed_step_count:
        raise AssertionError("V25c replay-error count differs from completed step count")
    primary_components = _component_summary(rows, V25C_PRIMARY_COMPONENT)
    component_additivity = {
        role: {
            term: {
                metric: _distribution(
                    float(row[f"{role}_component_additivity"][term][metric])
                    for row in rows
                )
                for metric in (
                    "residual_l2",
                    "relative_residual_l2",
                    "maximum_absolute_component_residual",
                    "maximum_component_residual_relative_to_actual_peak",
                )
            }
            for term in (
                "raw_total_equals_final_plus_prefix",
                "historical_total_equals_final_plus_prefix",
            )
        }
        for role in ("known", "clutter")
    }
    return {
        "integrity": {
            "replayed_epoch_count": len(boundaries),
            "replayed_step_count": replayed_step_count,
            "expected_step_count": EXPECTED_TOTAL_STEPS,
            "every_parent_batch_hash_matched_exactly": every_parent_batch_hash_matched_exactly,
            "every_clip_decision_matched_exactly": every_clip_decision_matched_exactly,
            "replay_absolute_error_distributions": replay_errors.summary(),
            "epoch_boundary_state_residuals": dict(boundaries),
            "state_tolerances": {
                "relative_l2": STATE_L2_RELATIVE_TOLERANCE,
                "maximum_absolute_component": STATE_MAX_ABSOLUTE_TOLERANCE,
            },
            "component_additivity_tolerances": v25c_component_additivity_tolerance_policy(),
            "component_additivity_residual_distributions": component_additivity,
        },
        "mixed_known_clutter_batches": {
            "count": len(rows),
            "expected_count": EXPECTED_MIXED_KNOWN_CLUTTER_BATCHES,
            "primary_component": V25C_PRIMARY_COMPONENT,
            "primary_historical_total": primary_components,
            "all_component_summaries": {
                component: _component_summary(rows, component)
                for component in V25C_COMPONENT_NAMES
            },
            "observation_value_distributions": _observation_value_summary(rows),
            "by_known_base_label": {
                label: _component_summary(group, V25C_PRIMARY_COMPONENT)
                for label, group in sorted(by_known_label.items())
            },
            "by_clip": {
                name: _component_summary(group, V25C_PRIMARY_COMPONENT)
                for name, group in by_clip.items()
            },
            "by_epoch": {
                name: _component_summary(group, V25C_PRIMARY_COMPONENT)
                for name, group in sorted(by_epoch.items(), key=lambda item: int(item[0]))
            },
            "observation_strata": {
                name: {
                    value: _component_summary(group, V25C_PRIMARY_COMPONENT)
                    for value, group in values.items()
                }
                for name, values in by_observation.items()
            },
        },
    }


def _isolated_epoch_worker_payload(
    *, root: Path, source: Mapping[str, Any], epoch: int
) -> dict[str, Any]:
    rows, integrity, boundary = _replay_epoch(root=root, source=source, epoch=epoch)
    return {
        "schema_version": V25C_ISOLATED_EPOCH_WORKER_SCHEMA,
        "epoch": int(epoch),
        "rows": rows,
        "integrity": integrity,
        "boundary": boundary,
    }


def _read_limited_text(stream: Any, *, limit: int) -> str:
    """Read only the diagnostic tail of a worker stderr stream."""

    if limit <= 0:
        raise ValueError("V25c worker stderr limit must be positive")
    # A deque of arbitrary chunks can discard the whole current chunk when a
    # diagnostic is larger than the limit.  Keep an exact bounded byte tail so
    # a large native/CUDA failure still leaves its most useful final text.
    retained = bytearray()
    while True:
        chunk = stream.read(V25C_WORKER_STDERR_READ_CHUNK_BYTES)
        if not chunk:
            break
        retained.extend(chunk)
        if len(retained) > limit:
            del retained[:-limit]
    return bytes(retained).decode("utf-8", errors="replace")


def _read_worker_payload(path: Path, *, epoch: int) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"V25c isolated epoch worker did not create its payload: epoch={epoch}")
    size = int(path.stat().st_size)
    if not 0 < size <= V25C_WORKER_PAYLOAD_MAX_BYTES:
        raise RuntimeError(
            "V25c isolated epoch worker payload has an invalid size: "
            f"epoch={epoch}; bytes={size}; limit={V25C_WORKER_PAYLOAD_MAX_BYTES}"
        )
    try:
        with path.open("rb") as stream:
            raw = stream.read(V25C_WORKER_PAYLOAD_MAX_BYTES + 1)
        if len(raw) != size or len(raw) > V25C_WORKER_PAYLOAD_MAX_BYTES:
            raise ValueError("sidecar byte count changed or exceeds its transport limit")
        return dict(json.loads(raw.decode("utf-8")))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(
            "V25c isolated epoch worker payload is invalid JSON: "
            f"epoch={epoch}; path={path}"
        ) from error


def _validated_worker_integrity(payload: Mapping[str, Any]) -> dict[str, Any]:
    integrity = dict(payload.get("integrity") or {})
    expected = {
        "step_count",
        "replay_errors",
        "every_parent_batch_hash_matched_exactly",
        "every_clip_decision_matched_exactly",
    }
    if set(integrity) != expected:
        raise RuntimeError("V25c isolated epoch worker integrity schema is invalid")
    if int(integrity["step_count"]) != EXPECTED_STEPS_PER_EPOCH:
        raise RuntimeError(
            "V25c isolated epoch worker has an incomplete replay: "
            f"steps={integrity['step_count']}"
        )
    replay_errors = dict(integrity["replay_errors"] or {})
    keys = tuple(replay_errors.get("keys") or ())
    arrays = dict(replay_errors.get("values") or {})
    exact = dict(replay_errors.get("all_exact") or {})
    if (
        keys != V25C_REPLAY_ERROR_KEYS
        or set(arrays) != set(V25C_REPLAY_ERROR_KEYS)
        or set(exact) != set(V25C_REPLAY_ERROR_KEYS)
        or any(len(list(arrays[key])) != EXPECTED_STEPS_PER_EPOCH for key in keys)
    ):
        raise RuntimeError("V25c isolated epoch worker replay-error schema is invalid")
    return integrity


def _run_isolated_epoch_worker(
    *, root: Path, epoch: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    """Run one epoch with a file sidecar instead of an unbounded stdout pipe."""

    payload_path: Path | None = None
    process: subprocess.Popen[bytes] | None = None
    stderr = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root / "tmp",
            prefix=f"v25c-worker-epoch-{epoch:03d}-",
            suffix=".json",
            delete=False,
        ) as stream:
            payload_path = Path(stream.name)
        # ``NamedTemporaryFile`` is used only to reserve a collision-resistant
        # name.  Remove its empty placeholder before launching the worker: the
        # worker itself must be the sole creator and atomically promotes its
        # sidecar, rather than overwriting a parent-created file.
        payload_path.unlink(missing_ok=False)
        command = [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "report.known_gate_v25c",
            "--project-root",
            str(root),
            "--isolated-epoch-worker",
            str(epoch),
            "--worker-payload-path",
            str(payload_path),
        ]
        process = subprocess.Popen(
            command,
            cwd=root / "src-v3",
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if process.stderr is None:
            raise RuntimeError("V25c isolated epoch worker has no stderr pipe")
        stderr = _read_limited_text(
            process.stderr, limit=V25C_WORKER_FAILURE_OUTPUT_LIMIT
        )
        returncode = int(process.wait())
        if returncode != 0:
            raise RuntimeError(
                "V25c isolated epoch worker failed; "
                f"epoch={epoch}; returncode={returncode}; stderr={stderr}"
            )
        payload = _read_worker_payload(payload_path, epoch=epoch)
        if (
            payload.get("schema_version") != V25C_ISOLATED_EPOCH_WORKER_SCHEMA
            or int(payload.get("epoch", -1)) != int(epoch)
        ):
            raise RuntimeError("V25c isolated epoch worker identity differs from requested epoch")
        rows = [dict(value) for value in list(payload.get("rows") or ())]
        integrity = _validated_worker_integrity(payload)
        boundary = dict(payload.get("boundary") or {})
        if any(int(row.get("epoch", -1)) != epoch for row in rows):
            raise RuntimeError("V25c isolated epoch worker returned a cross-epoch row")
        expected_boundary = {
            "residual_l2",
            "relative_residual_l2",
            "maximum_absolute_component_residual",
        }
        if set(boundary) != expected_boundary:
            raise RuntimeError("V25c isolated epoch worker boundary schema is invalid")
        return rows, integrity, {key: float(boundary[key]) for key in expected_boundary}
    finally:
        if process is not None and process.stderr is not None:
            process.stderr.close()
        if payload_path is not None:
            payload_path.unlink(missing_ok=True)


def _assert_report_contains_no_raw_identifier(root: Path, report: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    """Audit the in-memory object without cloning the whole report to a string."""

    dataset = build_known_gate_dataset(root, source["config"], split_name="train")
    parent_ids = [str(value) for value in dataset.parent_ids]
    if len(parent_ids) != 986 or len(parent_ids) != len(set(parent_ids)):
        raise AssertionError("V25c source train parent identity audit population is invalid")
    assert_no_raw_identifiers(report, raw_parent_ids=parent_ids)


def _completion_payload(
    *,
    report_path: Path,
    report: Mapping[str, Any],
    contract_path: Path,
    readiness_path: Path,
) -> dict[str, Any]:
    """Create the small result surface consumed by PowerShell and callers."""

    integrity = dict(report["integrity"])
    mixed = dict(report["mixed_known_clutter_batches"])
    return {
        "status": "completed_development_exact_replay_module_gradient_conflict_diagnosis_only",
        "schema_version": V25C_COMPLETION_SCHEMA,
        "variant": V25C_VARIANT,
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "report_bytes": int(report_path.stat().st_size),
        "experiment_contract_path": str(contract_path),
        "experiment_contract_sha256": sha256_file(contract_path),
        "readiness_path": str(readiness_path),
        "readiness_sha256": sha256_file(readiness_path),
        "replayed_epoch_count": int(integrity["replayed_epoch_count"]),
        "replayed_step_count": int(integrity["replayed_step_count"]),
        "expected_step_count": int(integrity["expected_step_count"]),
        "every_parent_batch_hash_matched_exactly": bool(
            integrity["every_parent_batch_hash_matched_exactly"]
        ),
        "every_clip_decision_matched_exactly": bool(
            integrity["every_clip_decision_matched_exactly"]
        ),
        "mixed_known_clutter_batch_count": int(mixed["count"]),
        "expected_mixed_known_clutter_batch_count": int(mixed["expected_count"]),
        "execution_resource_safety_policy": v25c_execution_resource_safety_policy(),
        "validation_payload_access_count": 0,
        "test_payload_access_count": 0,
        "unknown_payload_access_count": 0,
        "checkpoint_written": False,
        "optimizer_state_written": False,
        "checkpoint_selection_performed": False,
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }


def run(
    project_root: Path,
    *,
    isolated_epoch_worker: int | None = None,
    worker_payload_path: Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    source = load_v25a_source(root)
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    if not readiness_path.is_file() or not contract_path.is_file():
        raise FileNotFoundError("V25c prepare artifacts are missing; run prepare first")
    readiness = _read_json(readiness_path)
    contract = _read_json(contract_path)
    prepared_source_binding = assert_prepared_v25c_binding(
        root,
        source,
        contract=contract,
        readiness=readiness,
        contract_sha256=sha256_file(contract_path),
    )
    if (
        readiness.get("status") != "passed"
        or readiness.get("variant") != V25C_VARIANT
        or contract.get("schema_version")
        != "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-localization-experiment-contract-v1"
        or contract.get("internal_variant") != V25C_VARIANT
        or int(readiness.get("validation_payload_access_count", -1)) != 0
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or int(readiness.get("unknown_payload_access_count", -1)) != 0
        or readiness.get("deployment_eligible") is not False
    ):
        raise ValueError("V25c readiness/contract is stale or outside sealed diagnostic scope")
    if (isolated_epoch_worker is None) != (worker_payload_path is None):
        raise ValueError(
            "V25c worker payload path is required exactly for an isolated epoch worker"
        )
    if isolated_epoch_worker is not None:
        epoch = int(isolated_epoch_worker)
        if epoch not in EXPECTED_EPOCHS:
            raise ValueError("V25c isolated epoch worker is outside the frozen 20-epoch range")
        payload = _isolated_epoch_worker_payload(root=root, source=source, epoch=epoch)
        payload_path = Path(worker_payload_path).resolve()
        tmp_root = (root / "tmp").resolve()
        if payload_path.parent != tmp_root:
            raise ValueError("V25c isolated epoch worker payload must live directly under project tmp")
        _write_worker_payload(payload_path, payload)
        return {
            "status": "isolated_epoch_worker_completed",
            "schema_version": V25C_ISOLATED_EPOCH_WORKER_SCHEMA,
            "epoch": epoch,
            "payload_path": str(payload_path),
            "payload_bytes": int(payload_path.stat().st_size),
        }
    all_rows: list[dict[str, Any]] = []
    replay_errors = _StreamingDistribution(keys=V25C_REPLAY_ERROR_KEYS)
    replayed_step_count = 0
    every_parent_batch_hash_matched_exactly = True
    every_clip_decision_matched_exactly = True
    boundaries: dict[str, dict[str, float]] = {}
    for epoch in EXPECTED_EPOCHS:
        rows, integrity, boundary = _run_isolated_epoch_worker(root=root, epoch=epoch)
        all_rows.extend(rows)
        replay_errors.merge(dict(integrity["replay_errors"]))
        replayed_step_count += int(integrity["step_count"])
        every_parent_batch_hash_matched_exactly = (
            every_parent_batch_hash_matched_exactly
            and bool(integrity["every_parent_batch_hash_matched_exactly"])
        )
        every_clip_decision_matched_exactly = (
            every_clip_decision_matched_exactly
            and bool(integrity["every_clip_decision_matched_exactly"])
        )
        boundaries[str(epoch)] = boundary
    if replayed_step_count != EXPECTED_TOTAL_STEPS:
        raise AssertionError("V25c did not replay every frozen V25a update")
    if len(all_rows) != EXPECTED_MIXED_KNOWN_CLUTTER_BATCHES:
        raise AssertionError(
            "V25c mixed Known/Clutter batch support differs from frozen V25a: "
            f"observed={len(all_rows)} expected={EXPECTED_MIXED_KNOWN_CLUTTER_BATCHES}"
        )
    report_summary = _summary_from_rows(
        all_rows,
        replay_errors,
        replayed_step_count=replayed_step_count,
        every_parent_batch_hash_matched_exactly=every_parent_batch_hash_matched_exactly,
        every_clip_decision_matched_exactly=every_clip_decision_matched_exactly,
        boundaries=boundaries,
    )
    report = {
        "status": "completed_development_exact_replay_module_gradient_conflict_diagnosis_only",
        "schema_version": V25C_REPORT_SCHEMA,
        "variant": V25C_VARIANT,
        "scope": (
            "V25c is a read-only V25a train-update replay and module-level mixed "
            "Known/Clutter parent-gradient conflict localization. It makes no training "
            "intervention or product-performance claim."
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
        "execution_resource_safety_policy": v25c_execution_resource_safety_policy(),
        **report_summary,
        "per_mixed_known_clutter_batch": all_rows,
        "interpretation_limits": [
            "Negative gradients in the classifier head may be ordinary binary class competition rather than harmful shared-representation interference.",
            "Observed module-level conflict, clipping, subclass, RD, same-second, or course-coverage associations are not causal proof of a remedy.",
            "V25c preserves V25a historical batch_weight_sum semantics and does not test a loss, sampler, feature, or model intervention.",
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
    _assert_report_contains_no_raw_identifier(root, report, source)
    _write_new_or_identical(root / OUTPUT, report)
    completion = _completion_payload(
        report_path=root / OUTPUT,
        report=report,
        contract_path=contract_path,
        readiness_path=readiness_path,
    )
    _write_new_or_identical(root / COMPLETION, completion)
    # The parent process needs the report only until its atomically persisted
    # completion record is constructed. Drop the rich 638-record structure
    # before returning a compact object to the PowerShell caller.
    del report, report_summary, all_rows, replay_errors, boundaries
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay V25a updates and localize module-level mixed-parent gradients."
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
    parser.add_argument(
        "--worker-payload-path",
        type=Path,
        default=None,
        help="Internal: atomically receive one isolated epoch's bounded JSON sidecar.",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.project_root.resolve(),
                isolated_epoch_worker=args.isolated_epoch_worker,
                worker_payload_path=args.worker_payload_path,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
