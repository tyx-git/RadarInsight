"""Read-only V25b parent-gradient decomposition primitives.

V25b is deliberately not a new optimisation method.  It replays frozen V25a
updates and, only for batches that contain a Clutter parent, decomposes the
historical ``batch_weight_sum`` objective into anonymous per-parent gradient
contributions.  The helpers in this module never write ``parameter.grad`` and
never persist parent identifiers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .known_gate_gradient_telemetry import (
    V25A_GRADIENT_MODULE_NAMES,
    partition_gradient_modules,
)


V25B_VARIANT = "v25b_v25a_exact_replay_parent_gradient_decomposition"
V25B_EXPERIMENT_SCHEMA = (
    "RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-"
    "experiment-contract-v1"
)
V25B_READINESS_SCHEMA = (
    "RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-"
    "readiness-v1"
)
V25B_REPORT_SCHEMA = (
    "RADAR-LN3-V25b-known-gate-exact-replay-parent-gradient-decomposition-"
    "report-v1"
)
V25B_MODULE_NAMES = V25A_GRADIENT_MODULE_NAMES
V25B_FORBIDDEN_IDENTIFIER_KEYS = frozenset(
    {
        "parent_sample_id",
        "parent_sample_ids",
        "sample_id",
        "sample_ids",
        "track_id",
        "capture_id",
        "source_name",
    }
)

# The V25a final-state term is a fused CUDA weighted-mean CE, while V25b must
# expose unreduced parent terms.  They are algebraically identical but can
# differ at the floating-point reduction level.  These bounds are registered
# once here so the builder contract, reporter and smoke cannot drift apart.
V25B_DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE = 3.0e-6
V25B_DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE = 1.0e-4
# The failing epoch-12 source/proxy bridge was 3.08e-6 in relative L2 and
# the full targeted replay observed 2.70e-6.  Keep a deliberately narrow
# margin above that CUDA reduction noise instead of accepting a generic 1e-4
# relative discrepancy.
V25B_DECOMPOSITION_GRADIENT_RELATIVE_L2_TOLERANCE = 1.0e-5
V25B_DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE = 5.0e-5
V25B_DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE = 2.0e-5


GradientTuple = tuple[torch.Tensor, ...]


def v25b_decomposition_tolerance_policy() -> dict[str, float | str]:
    """Return the complete, contract-serializable V25b bridge policy."""

    return {
        "loss_absolute": V25B_DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE,
        "gradient_absolute_l2": V25B_DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE,
        "gradient_relative_l2": V25B_DECOMPOSITION_GRADIENT_RELATIVE_L2_TOLERANCE,
        "gradient_maximum_absolute_component_floor": (
            V25B_DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE
        ),
        "gradient_maximum_component_relative_to_actual_peak": (
            V25B_DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE
        ),
        "bridge_semantics": (
            "original_fused_loss_to_explicit_unreduced_proxy_and_explicit_proxy_to_parent_sum"
        ),
    }


def parent_batch_sha256(parent_ids: Sequence[str]) -> str:
    """Return the sealed V25a ordered-batch digest without retaining IDs."""

    values = [str(value) for value in parent_ids]
    if not values or len(values) != len(set(values)):
        raise ValueError("V25b parent batch must contain non-empty unique IDs")
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _zero_scalar(reference: torch.Tensor) -> torch.Tensor:
    """A differentiable scalar zero on the exact source graph/device."""

    return reference.sum() * 0.0


def _validate_loss_inputs(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    class_weights: torch.Tensor,
    *,
    prefix_auxiliary_weight: float,
    prefix_auxiliary_min_events: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if sequence_logits.ndim != 3:
        raise ValueError("V25b sequence_logits must have shape [batch, sequence, classes]")
    if labels.shape != (sequence_logits.shape[0],):
        raise ValueError("V25b labels must align with the batch dimension")
    if padding_mask.shape != sequence_logits.shape[:2] or padding_mask.dtype != torch.bool:
        raise ValueError("V25b padding mask must be boolean and align with logits")
    if not torch.isfinite(sequence_logits).all():
        raise ValueError("V25b logits must be finite")
    class_count = int(sequence_logits.shape[-1])
    if class_count < 2:
        raise ValueError("V25b requires at least two KnownGate classes")
    targets = labels.to(device=sequence_logits.device, dtype=torch.long)
    if bool(((targets < 0) | (targets >= class_count)).any()):
        raise ValueError("V25b task labels are outside the configured class range")
    weights = class_weights.to(device=sequence_logits.device, dtype=sequence_logits.dtype)
    if (
        weights.shape != (class_count,)
        or not torch.isfinite(weights).all()
        or bool((weights <= 0.0).any())
    ):
        raise ValueError("V25b class weights must be finite, positive, and class-aligned")
    if not 0.0 <= float(prefix_auxiliary_weight) <= 1.0:
        raise ValueError("V25b prefix auxiliary weight is invalid")
    if int(prefix_auxiliary_min_events) < 2:
        raise ValueError("V25b prefix supervision must exclude the first event")
    return targets, weights, ~padding_mask, class_count


def parent_loss_terms_batch_weight_sum(
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    prefix_auxiliary_weight: float,
    prefix_auxiliary_min_events: int,
) -> dict[str, Any]:
    """Decompose the frozen V25a objective into exact parent scalar terms.

    The returned historical terms sum to the loss semantics of
    ``parent_final_task_ce_with_prefix_auxiliary(...,
    class_weight_normalization='batch_weight_sum')``.  Raw terms intentionally
    omit class weighting so their direction and norm can be compared before
    the historical class-weight coefficients are applied.
    """

    targets, weights, active, class_count = _validate_loss_inputs(
        sequence_logits,
        labels,
        padding_mask,
        class_weights,
        prefix_auxiliary_weight=prefix_auxiliary_weight,
        prefix_auxiliary_min_events=prefix_auxiliary_min_events,
    )
    lengths = active.long().sum(dim=1)
    if bool((lengths <= 0).any()):
        raise ValueError("V25b loss decomposition received an empty parent")
    batch_size = int(sequence_logits.shape[0])
    batch_indices = torch.arange(batch_size, device=sequence_logits.device)
    final_indices = lengths - 1
    final_logits = sequence_logits[batch_indices, final_indices]
    raw_final_terms = nn.functional.cross_entropy(final_logits, targets, reduction="none")
    weighted_final_terms = nn.functional.cross_entropy(
        final_logits,
        targets,
        weight=weights,
        reduction="none",
    )
    final_weight_sum = weights[targets].sum()
    if not torch.isfinite(final_weight_sum) or float(final_weight_sum.detach().item()) <= 0.0:
        raise ValueError("V25b final parent weight denominator is invalid")
    historical_final_terms = weighted_final_terms / final_weight_sum

    zero = _zero_scalar(sequence_logits)
    raw_prefix_terms = torch.stack([zero for _ in range(batch_size)])
    historical_prefix_terms = torch.stack([zero for _ in range(batch_size)])
    prefix_parent_eligible = torch.zeros(
        (batch_size,), dtype=torch.bool, device=sequence_logits.device
    )
    prefix_weight_sum = zero.detach()
    if float(prefix_auxiliary_weight) > 0.0:
        event_numbers = torch.arange(
            1,
            sequence_logits.shape[1] + 1,
            device=sequence_logits.device,
        ).unsqueeze(0)
        eligible = (
            active
            & (event_numbers >= int(prefix_auxiliary_min_events))
            & (event_numbers < lengths.unsqueeze(1))
        )
        prefix_counts = eligible.sum(dim=1)
        prefix_parent_eligible = prefix_counts > 0
        expanded_targets = targets.unsqueeze(1).expand(-1, sequence_logits.shape[1])
        raw_per_event = nn.functional.cross_entropy(
            sequence_logits.reshape(-1, class_count),
            expanded_targets.reshape(-1),
            reduction="none",
        ).reshape(sequence_logits.shape[:2])
        weighted_per_event = nn.functional.cross_entropy(
            sequence_logits.reshape(-1, class_count),
            expanded_targets.reshape(-1),
            weight=weights,
            reduction="none",
        ).reshape(sequence_logits.shape[:2])
        raw_parent_means = (
            (raw_per_event * eligible.to(dtype=raw_per_event.dtype)).sum(dim=1)
            / prefix_counts.clamp_min(1)
        )
        weighted_parent_means = (
            (weighted_per_event * eligible.to(dtype=weighted_per_event.dtype)).sum(dim=1)
            / prefix_counts.clamp_min(1)
        )
        raw_prefix_terms = torch.where(
            prefix_parent_eligible,
            raw_parent_means,
            raw_parent_means * 0.0,
        )
        selected_weights = weights[targets[prefix_parent_eligible]]
        prefix_weight_sum = selected_weights.sum()
        if not torch.isfinite(prefix_weight_sum) or float(prefix_weight_sum.detach().item()) <= 0.0:
            raise ValueError("V25b prefix parent weight denominator is invalid")
        scaled = (
            float(prefix_auxiliary_weight)
            * weighted_parent_means
            / prefix_weight_sum
        )
        historical_prefix_terms = torch.where(
            prefix_parent_eligible,
            scaled,
            scaled * 0.0,
        )

    final_coefficients = weights[targets] / final_weight_sum
    prefix_normalization_coefficients = torch.zeros_like(final_coefficients)
    prefix_coefficients = torch.zeros_like(final_coefficients)
    if bool(prefix_parent_eligible.any()):
        prefix_normalization_coefficients[prefix_parent_eligible] = (
            weights[targets[prefix_parent_eligible]] / prefix_weight_sum
        )
        prefix_coefficients = (
            float(prefix_auxiliary_weight) * prefix_normalization_coefficients
        )
    historical_total_terms = historical_final_terms + historical_prefix_terms
    raw_total_terms = raw_final_terms + float(prefix_auxiliary_weight) * raw_prefix_terms
    return {
        "raw_final_terms": raw_final_terms,
        "raw_prefix_terms": raw_prefix_terms,
        "raw_total_terms": raw_total_terms,
        "historical_final_terms": historical_final_terms,
        "historical_prefix_terms": historical_prefix_terms,
        "historical_total_terms": historical_total_terms,
        "historical_total_loss": historical_total_terms.sum(),
        "final_weight_sum": final_weight_sum,
        "prefix_weight_sum": prefix_weight_sum,
        "final_coefficients": final_coefficients,
        "prefix_normalization_coefficients": prefix_normalization_coefficients,
        "prefix_coefficients": prefix_coefficients,
        "prefix_parent_eligible": prefix_parent_eligible,
        "lengths": lengths,
        "prefix_auxiliary_weight": float(prefix_auxiliary_weight),
    }


def all_trainable_parameters(model: nn.Module) -> tuple[nn.Parameter, ...]:
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    if not parameters:
        raise ValueError("V25b model has no trainable parameters")
    if any(not parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("V25b requires the frozen full-parameter V25a model")
    return parameters


def module_parameter_indices(
    model: nn.Module,
    parameters: Sequence[nn.Parameter],
) -> dict[str, tuple[int, ...]]:
    """Map the sealed V25a module partition into a parameter-vector layout."""

    groups = partition_gradient_modules(model)
    by_identity = {id(parameter): index for index, parameter in enumerate(parameters)}
    result: dict[str, tuple[int, ...]] = {}
    for module in V25B_MODULE_NAMES:
        indices = tuple(by_identity[id(parameter)] for _, parameter in groups[module])
        if not indices:
            raise AssertionError(f"V25b module partition is empty: {module}")
        result[module] = indices
    flat = [index for indices in result.values() for index in indices]
    if sorted(flat) != list(range(len(parameters))) or len(flat) != len(set(flat)):
        raise AssertionError("V25b module partition does not cover each trainable parameter once")
    return result


def snapshot_parameter_gradients(parameters: Sequence[nn.Parameter]) -> GradientTuple:
    """Clone existing gradients without converting absent gradients to None."""

    values: list[torch.Tensor] = []
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is None:
            values.append(torch.zeros_like(parameter))
        else:
            if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
                raise ValueError("V25b encountered a non-finite or malformed parameter gradient")
            values.append(gradient.detach().clone())
    return tuple(values)


def gradients_from_scalar(
    scalar: torch.Tensor,
    parameters: Sequence[nn.Parameter],
    *,
    retain_graph: bool = True,
) -> GradientTuple:
    """Read a scalar's gradient contribution without mutating ``.grad``.

    ``retain_graph`` defaults to the historical diagnostic behavior. Training
    interventions that read two scalar terms can retain for the first read and
    release the graph on the final read, preventing a per-step autograd graph
    from surviving until the next batch.
    """

    if scalar.ndim != 0:
        raise ValueError("V25b scalar-gradient source must be zero-dimensional")
    if not scalar.requires_grad:
        return tuple(torch.zeros_like(parameter) for parameter in parameters)
    values = torch.autograd.grad(
        scalar,
        tuple(parameters),
        retain_graph=bool(retain_graph),
        create_graph=False,
        allow_unused=True,
    )
    result: list[torch.Tensor] = []
    for parameter, gradient in zip(parameters, values, strict=True):
        value = torch.zeros_like(parameter) if gradient is None else gradient.detach()
        if value.shape != parameter.shape or not torch.isfinite(value).all():
            raise ValueError("V25b per-parent gradient is non-finite or malformed")
        result.append(value)
    return tuple(result)


def gradient_add(*gradients: GradientTuple) -> GradientTuple:
    if not gradients:
        raise ValueError("V25b cannot add an empty gradient list")
    width = len(gradients[0])
    if width == 0 or any(len(value) != width for value in gradients):
        raise ValueError("V25b gradient vectors have incompatible widths")
    return tuple(sum((value[index] for value in gradients), start=torch.zeros_like(gradients[0][index])) for index in range(width))


def gradient_l2(gradient: GradientTuple) -> float:
    if not gradient:
        raise ValueError("V25b gradient vector is empty")
    total = sum(
        (torch.sum(value.detach().to(dtype=torch.float64).square()) for value in gradient),
        start=torch.zeros((), dtype=torch.float64, device=gradient[0].device),
    )
    result = float(torch.sqrt(total).detach().cpu().item())
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("V25b gradient L2 is invalid")
    return result


def gradient_dot(left: GradientTuple, right: GradientTuple) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("V25b gradient dot operands are incompatible")
    total = sum(
        (
            torch.sum(
                lhs.detach().to(dtype=torch.float64)
                * rhs.detach().to(dtype=torch.float64)
            )
            for lhs, rhs in zip(left, right, strict=True)
        ),
        start=torch.zeros((), dtype=torch.float64, device=left[0].device),
    )
    result = float(total.detach().cpu().item())
    if not math.isfinite(result):
        raise ValueError("V25b gradient dot product is non-finite")
    return result


def gradient_cosine(left: GradientTuple, right: GradientTuple) -> float | None:
    denominator = gradient_l2(left) * gradient_l2(right)
    if denominator <= 0.0:
        return None
    value = gradient_dot(left, right) / denominator
    if not math.isfinite(value):
        raise ValueError("V25b gradient cosine is non-finite")
    return float(max(-1.0, min(1.0, value)))


def gradients_exactly_equal(left: GradientTuple, right: GradientTuple) -> bool:
    return len(left) == len(right) and all(
        torch.equal(lhs, rhs) for lhs, rhs in zip(left, right, strict=True)
    )


def module_gradient_metrics(
    gradient: GradientTuple,
    module_indices: Mapping[str, Sequence[int]],
) -> dict[str, dict[str, float]]:
    if tuple(module_indices) != V25B_MODULE_NAMES:
        raise ValueError("V25b module metric order differs from the sealed partition")
    module_squares: dict[str, float] = {}
    for module in V25B_MODULE_NAMES:
        values = [gradient[index] for index in module_indices[module]]
        square = sum(
            (torch.sum(value.detach().to(dtype=torch.float64).square()) for value in values),
            start=torch.zeros((), dtype=torch.float64, device=values[0].device),
        )
        module_squares[module] = float(square.detach().cpu().item())
    total = float(sum(module_squares.values()))
    if not math.isfinite(total) or total < 0.0:
        raise ValueError("V25b module gradient energy is invalid")
    return {
        module: {
            "l2": float(math.sqrt(max(0.0, square))),
            "squared_fraction": float(square / total) if total > 0.0 else 0.0,
        }
        for module, square in module_squares.items()
    }


def gradient_difference_metrics(
    observed: GradientTuple,
    reconstructed: GradientTuple,
) -> dict[str, float]:
    if len(observed) != len(reconstructed) or not observed:
        raise ValueError("V25b reconstruction vectors are incompatible")
    difference = tuple(
        actual - expected for actual, expected in zip(observed, reconstructed, strict=True)
    )
    observed_l2 = gradient_l2(observed)
    residual_l2 = gradient_l2(difference)
    observed_peak = max(
        float(value.detach().abs().max().cpu().item()) for value in observed
    )
    reconstructed_peak = max(
        float(value.detach().abs().max().cpu().item()) for value in reconstructed
    )
    maximum = max(
        float(value.detach().abs().max().cpu().item())
        for value in difference
    )
    return {
        "actual_l2": observed_l2,
        "actual_maximum_absolute_component": observed_peak,
        "reconstructed_maximum_absolute_component": reconstructed_peak,
        "residual_l2": residual_l2,
        "relative_residual_l2": float(residual_l2 / max(observed_l2, 1.0e-12)),
        "maximum_absolute_component_residual": maximum,
        "maximum_component_residual_relative_to_actual_peak": float(
            maximum / max(observed_peak, 1.0e-12)
        ),
    }


def assert_no_raw_identifiers(
    payload: Any,
    *,
    raw_parent_ids: Sequence[str],
) -> None:
    """Fail closed if a persisted diagnostic object retains source identities."""

    forbidden_values = {str(value) for value in raw_parent_ids}
    if not forbidden_values:
        raise ValueError("V25b raw parent identity audit received no parent IDs")

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if str(key) in V25B_FORBIDDEN_IDENTIFIER_KEYS:
                    raise ValueError("V25b report contains a forbidden raw identifier field")
                visit(nested)
        elif isinstance(value, (list, tuple)):
            for nested in value:
                visit(nested)
        elif isinstance(value, str) and value in forbidden_values:
            raise ValueError("V25b report contains a raw parent identifier value")

    visit(payload)


def finite_float(value: Any, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"V25b {name} is non-finite")
    return result


__all__ = [
    "GradientTuple",
    "V25B_EXPERIMENT_SCHEMA",
    "V25B_DECOMPOSITION_GRADIENT_ABSOLUTE_L2_TOLERANCE",
    "V25B_DECOMPOSITION_GRADIENT_MAX_ABSOLUTE_COMPONENT_FLOOR_TOLERANCE",
    "V25B_DECOMPOSITION_GRADIENT_MAX_COMPONENT_RELATIVE_TO_PEAK_TOLERANCE",
    "V25B_DECOMPOSITION_GRADIENT_RELATIVE_L2_TOLERANCE",
    "V25B_DECOMPOSITION_LOSS_ABSOLUTE_TOLERANCE",
    "V25B_FORBIDDEN_IDENTIFIER_KEYS",
    "V25B_MODULE_NAMES",
    "V25B_READINESS_SCHEMA",
    "V25B_REPORT_SCHEMA",
    "V25B_VARIANT",
    "all_trainable_parameters",
    "assert_no_raw_identifiers",
    "finite_float",
    "gradient_add",
    "gradient_cosine",
    "gradient_difference_metrics",
    "gradient_dot",
    "gradient_l2",
    "gradients_exactly_equal",
    "gradients_from_scalar",
    "module_gradient_metrics",
    "module_parameter_indices",
    "parent_batch_sha256",
    "parent_loss_terms_batch_weight_sum",
    "snapshot_parameter_gradients",
    "v25b_decomposition_tolerance_policy",
]
