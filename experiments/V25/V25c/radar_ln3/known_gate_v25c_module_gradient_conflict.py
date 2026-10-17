"""Read-only V25c module-level mixed-parent gradient conflict primitives.

V25c extends the completed V25b diagnostic without changing a single V25a
training update.  It operates only on a retained backward graph for a frozen
mixed Known/Clutter batch and converts parent-gradient vectors into anonymous
scalar module metrics.  No vector or source identifier is serializable from
this module.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .known_gate_v25b_gradient_decomposition import (
    GradientTuple,
    V25B_MODULE_NAMES,
    gradient_add,
    gradient_cosine,
    gradient_difference_metrics,
    gradient_dot,
    gradient_l2,
    gradients_from_scalar,
)


V25C_VARIANT = "v25c_v25a_exact_replay_module_gradient_conflict_localization"
V25C_EXPERIMENT_SCHEMA = (
    "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-"
    "localization-experiment-contract-v1"
)
V25C_READINESS_SCHEMA = (
    "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-"
    "localization-readiness-v1"
)
V25C_REPORT_SCHEMA = (
    "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-"
    "localization-report-v1"
)
V25C_COMPLETION_SCHEMA = (
    "RADAR-LN3-V25c-known-gate-exact-replay-module-gradient-conflict-"
    "localization-completion-v1"
)

# ``raw_prefix_including_lambda`` is deliberately named this way: the raw
# prefix CE itself omits lambda, while the raw-total parent contribution is
# raw_final + lambda * raw_prefix.  Every component family below therefore
# sums exactly to its corresponding total parent-gradient family.
V25C_COMPONENT_NAMES = (
    "raw_total",
    "historical_total",
    "raw_final",
    "historical_final",
    "raw_prefix_including_lambda",
    "historical_prefix",
)
V25C_SOURCE_BRIDGE_COMPONENT_NAMES = (
    "raw_total",
    "historical_total",
    "historical_final",
    "historical_prefix",
)
V25C_PRIMARY_COMPONENT = "historical_total"
V25C_MODULE_NAMES = V25B_MODULE_NAMES
V25C_SHARED_MODULE_NAMES = tuple(
    module for module in V25C_MODULE_NAMES if module != "classifier"
)

# A parent ``total`` is differentiated from one scalar, while its final and
# prefix components are independently differentiated and then added. CUDA
# reduction / accumulation order can therefore differ slightly even though the
# scalar identity is exact. This is a distinct bridge from V25b's source
# fused-loss bridge, with separately registered bounded tolerances.
#
# The relative component bridge was calibrated against fresh CUDA replays of
# the fixed source epoch-3/step-467 mixed Bird/Clutter batch. Its raw Clutter
# maximum-component residual was 6.18e-5 to 6.52e-5 and its historical Clutter
# residual reached 7.66e-5 of the parent-total peak, while each independent L2
# bridge passed. The old 5e-5 limit therefore rejected a reproducible
# floating-point accumulation difference. 1e-4 leaves a deliberately bounded
# 31% margin over the largest registered fixed-source anchor while still
# limiting a coordinate discrepancy to 0.01% of its parent-gradient peak; the
# absolute floor and all
# source-loss, source-gradient, batch-hash, clipping and boundary-state checks
# remain unchanged.
V25C_COMPONENT_ADDITIVITY_ABSOLUTE_L2_TOLERANCE = 3.0e-3
V25C_COMPONENT_ADDITIVITY_RELATIVE_L2_TOLERANCE = 5.0e-5
V25C_COMPONENT_ADDITIVITY_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE = 4.0e-4
V25C_COMPONENT_ADDITIVITY_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE = 1.0e-4
V25C_WORKER_PAYLOAD_MAX_BYTES = 16 * 1024 * 1024
V25C_WORKER_FAILURE_OUTPUT_LIMIT = 8_000
V25C_WORKER_STDERR_READ_CHUNK_BYTES = 64 * 1024


def v25c_component_additivity_tolerance_policy() -> dict[str, float | str]:
    """Return the explicit numerical bridge for total/final/prefix gradients."""

    return {
        "absolute_l2": V25C_COMPONENT_ADDITIVITY_ABSOLUTE_L2_TOLERANCE,
        "relative_l2": V25C_COMPONENT_ADDITIVITY_RELATIVE_L2_TOLERANCE,
        "maximum_absolute_component_floor": (
            V25C_COMPONENT_ADDITIVITY_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE
        ),
        "maximum_component_relative_to_parent_total_peak": (
            V25C_COMPONENT_ADDITIVITY_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE
        ),
        "maximum_component_calibration": (
            "fresh_cuda_epoch3_step467_clutter_raw_6.18e-5_to_6.52e-5_"
            "and_historical_7.66e-5; registered_1.0e-4"
        ),
        "bridge_semantics": (
            "independently_differentiated_parent_total_vs_final_plus_lambda_prefix"
        ),
    }


def v25c_execution_resource_safety_policy() -> dict[str, int | str | bool]:
    """Return the frozen bounded-memory transport policy for V25c replay."""

    return {
        "epoch_worker_transport": "atomic_json_sidecar_in_project_tmp_not_stdout",
        "worker_payload_max_bytes": V25C_WORKER_PAYLOAD_MAX_BYTES,
        "worker_stderr_tail_max_bytes": V25C_WORKER_FAILURE_OUTPUT_LIMIT,
        "worker_stderr_read_chunk_bytes": V25C_WORKER_STDERR_READ_CHUNK_BYTES,
        "per_step_checks_retained_as_dicts": False,
        "per_step_error_scalars_retained_for_exact_quantiles": True,
        "epoch_boundary_checkpoint_restore": "source_equivalent_cuda_map_location",
        "epoch_boundary_reference_comparison": "cpu_payload_after_replay",
        "final_report_write": "atomic_streamed_json_then_block_hash_compare",
        "completed_report_stdout": "compact_completion_only",
    }


def _validate_component_mapping(components: Mapping[str, GradientTuple]) -> None:
    if tuple(components) != V25C_COMPONENT_NAMES:
        raise ValueError("V25c parent component order differs from the frozen protocol")
    widths = {len(value) for value in components.values()}
    if len(widths) != 1 or not widths or next(iter(widths)) <= 0:
        raise ValueError("V25c parent component gradients have incompatible widths")


def parent_source_bridge_component_gradients(
    terms: Mapping[str, Any],
    parameters: Sequence[torch.nn.Parameter],
    index: int,
) -> dict[str, GradientTuple]:
    """Reproduce V25b's parent-gradient call order for the source bridge.

    V25c adds two raw final/prefix probes that V25b did not need.  Those probes
    must not be interleaved with the source-to-parent bridge: on CUDA, separate
    autograd traversals can choose different reduction/allocation paths even
    when their mathematical inputs are unchanged.  The caller therefore invokes
    this function for parent indices in their frozen batch order, validates the
    V25b bridge, and only then asks for V25c's extra raw components.
    """

    count = int(terms["raw_total_terms"].shape[0])
    if not 0 <= int(index) < count:
        raise IndexError("V25c parent component index is outside the batch")
    components = {
        "raw_total": gradients_from_scalar(terms["raw_total_terms"][index], parameters),
        "historical_total": gradients_from_scalar(
            terms["historical_total_terms"][index], parameters
        ),
        "historical_final": gradients_from_scalar(
            terms["historical_final_terms"][index], parameters
        ),
        "historical_prefix": gradients_from_scalar(
            terms["historical_prefix_terms"][index], parameters
        ),
    }
    if tuple(components) != V25C_SOURCE_BRIDGE_COMPONENT_NAMES:
        raise AssertionError("V25c source bridge component order differs from V25b")
    return components


def extend_parent_components_with_raw_final_prefix(
    source_components: Mapping[str, GradientTuple],
    terms: Mapping[str, Any],
    parameters: Sequence[torch.nn.Parameter],
    index: int,
) -> dict[str, GradientTuple]:
    """Append V25c-only raw final/prefix probes after source bridging."""

    if tuple(source_components) != V25C_SOURCE_BRIDGE_COMPONENT_NAMES:
        raise ValueError("V25c source component order differs from the frozen bridge")
    count = int(terms["raw_total_terms"].shape[0])
    if not 0 <= int(index) < count:
        raise IndexError("V25c parent component index is outside the batch")
    lambda_value = float(terms["prefix_auxiliary_weight"])
    if not math.isfinite(lambda_value) or lambda_value < 0.0:
        raise ValueError("V25c prefix auxiliary weight is invalid")
    components = {
        "raw_total": source_components["raw_total"],
        "historical_total": source_components["historical_total"],
        "raw_final": gradients_from_scalar(terms["raw_final_terms"][index], parameters),
        "historical_final": source_components["historical_final"],
        "raw_prefix_including_lambda": gradients_from_scalar(
            lambda_value * terms["raw_prefix_terms"][index], parameters
        ),
        "historical_prefix": source_components["historical_prefix"],
    }
    _validate_component_mapping(components)
    return components


def parent_component_gradients(
    terms: Mapping[str, Any],
    parameters: Sequence[torch.nn.Parameter],
    index: int,
) -> dict[str, GradientTuple]:
    """Read one parent's six frozen objective components without mutating ``.grad``.

    The caller retains V25a's original backward graph.  This function merely
    asks autograd for scalar derivatives; it never calls ``backward`` and never
    touches the optimiser or parameter gradient buffers.
    """

    return extend_parent_components_with_raw_final_prefix(
        parent_source_bridge_component_gradients(terms, parameters, index),
        terms,
        parameters,
        index,
    )


def parent_component_additivity_metrics(
    components: Mapping[str, GradientTuple],
) -> dict[str, dict[str, float]]:
    """Return scalar checks that total gradients equal final plus prefix.

    The reporter applies its registered exact-replay tolerance to these
    metrics.  Keeping the metrics in the per-batch record makes the component
    separation independently auditable without retaining raw vectors.
    """

    _validate_component_mapping(components)
    raw_reconstructed = gradient_add(
        components["raw_final"], components["raw_prefix_including_lambda"]
    )
    historical_reconstructed = gradient_add(
        components["historical_final"], components["historical_prefix"]
    )
    return {
        "raw_total_equals_final_plus_prefix": gradient_difference_metrics(
            components["raw_total"], raw_reconstructed
        ),
        "historical_total_equals_final_plus_prefix": gradient_difference_metrics(
            components["historical_total"], historical_reconstructed
        ),
    }


def _module_gradient(
    gradient: GradientTuple,
    indices: Sequence[int],
) -> GradientTuple:
    values = tuple(gradient[int(index)] for index in indices)
    if not values:
        raise ValueError("V25c module gradient slice is empty")
    return values


def gradient_pair_conflict_metrics(
    known_gradient: GradientTuple,
    clutter_gradient: GradientTuple,
) -> dict[str, float | bool | None]:
    """Describe alignment and cancellation for a pair of parent gradients.

    ``negative_*_projection_l2`` measures only the component that points
    against the other parent's unit-gradient direction.  ``cancellation`` is
    based on the norm lost when the two vectors are summed, so it remains
    interpretable even when the two parent gradient norms differ greatly.
    """

    known_l2 = gradient_l2(known_gradient)
    clutter_l2 = gradient_l2(clutter_gradient)
    dot = gradient_dot(known_gradient, clutter_gradient)
    cosine = gradient_cosine(known_gradient, clutter_gradient)
    summed_l2 = gradient_l2(gradient_add(known_gradient, clutter_gradient))
    separate_l2 = known_l2 + clutter_l2
    negative_dot = max(0.0, -dot)
    known_antialigned_projection = (
        negative_dot / clutter_l2 if clutter_l2 > 0.0 else 0.0
    )
    clutter_antialigned_projection = (
        negative_dot / known_l2 if known_l2 > 0.0 else 0.0
    )
    cancellation_l2 = max(0.0, separate_l2 - summed_l2)
    cancellation_fraction = cancellation_l2 / separate_l2 if separate_l2 > 0.0 else 0.0
    values = {
        "known_gradient_l2": known_l2,
        "clutter_gradient_l2": clutter_l2,
        "dot": dot,
        "cosine": cosine,
        "negative_cosine": bool(cosine is not None and cosine < 0.0),
        "negative_dot": negative_dot,
        "known_antialigned_projection_l2": known_antialigned_projection,
        "clutter_antialigned_projection_l2": clutter_antialigned_projection,
        "known_antialigned_projection_fraction": (
            known_antialigned_projection / known_l2 if known_l2 > 0.0 else 0.0
        ),
        "clutter_antialigned_projection_fraction": (
            clutter_antialigned_projection / clutter_l2 if clutter_l2 > 0.0 else 0.0
        ),
        "summed_gradient_l2": summed_l2,
        "separate_gradient_l2": separate_l2,
        "cancellation_l2": cancellation_l2,
        "cancellation_fraction": cancellation_fraction,
    }
    for name, value in values.items():
        if isinstance(value, bool) or value is None:
            continue
        if not math.isfinite(float(value)) or (
            float(value) < 0.0 and name not in {"dot", "cosine"}
        ):
            raise ValueError(f"V25c pair conflict metric is invalid: {name}")
    if not -1.0 <= float(cancellation_fraction) <= 1.0:
        raise ValueError("V25c cancellation fraction is outside its physical range")
    return values


def module_and_full_pair_conflict_metrics(
    known_gradient: GradientTuple,
    clutter_gradient: GradientTuple,
    module_indices: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Return scalar conflict metrics for the full model and every module."""

    if tuple(module_indices) != V25C_MODULE_NAMES:
        raise ValueError("V25c module partition order differs from the frozen protocol")
    if len(known_gradient) != len(clutter_gradient) or not known_gradient:
        raise ValueError("V25c pair gradients have incompatible parameter layouts")
    full_indices = [index for indices in module_indices.values() for index in indices]
    if sorted(full_indices) != list(range(len(known_gradient))):
        raise ValueError("V25c module partition does not cover every parameter exactly once")
    return {
        "full_model": gradient_pair_conflict_metrics(known_gradient, clutter_gradient),
        "modules": {
            module: gradient_pair_conflict_metrics(
                _module_gradient(known_gradient, module_indices[module]),
                _module_gradient(clutter_gradient, module_indices[module]),
            )
            for module in V25C_MODULE_NAMES
        },
    }
