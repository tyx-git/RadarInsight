"""V25e shared-module dominant-gradient conflict projection.

V25e is a training intervention, not a diagnostic replay.  It is deliberately
small: the historical V25a loss is evaluated unchanged, and only a mixed
KnownTarget/ClutterTarget batch can alter gradients in the three shared
representation modules identified by V25c.  RD encoder and classifier
gradients are left untouched.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .known_gate_gradient_decomposition import (
    GradientTuple,
    V25B_MODULE_NAMES,
    gradient_add,
    gradient_cosine,
    gradient_dot,
    gradient_l2,
    gradients_from_scalar,
    parent_loss_terms_batch_weight_sum,
    snapshot_parameter_gradients,
)


V25E_SHARED_MODULE_NAMES = ("tr_encoder", "fusion", "liquid")
V25E_PROJECTED_MODULE_NAMES = V25E_SHARED_MODULE_NAMES
V25E_UNCHANGED_MODULE_NAMES = ("rd_encoder", "classifier")
V25E_PROJECTION_EPSILON = 1.0e-12
V25E_LOSS_RECONSTRUCTION_ABSOLUTE_TOLERANCE = 3.0e-5


def _add(left: GradientTuple, right: GradientTuple) -> GradientTuple:
    return gradient_add(left, right)


def _scale(gradient: GradientTuple, value: float) -> GradientTuple:
    if not math.isfinite(float(value)):
        raise ValueError("V25e gradient scale is non-finite")
    return tuple(component * float(value) for component in gradient)


def _subtract(left: GradientTuple, right: GradientTuple) -> GradientTuple:
    return tuple(lhs - rhs for lhs, rhs in zip(left, right, strict=True))


def _module_gradient(
    gradient: GradientTuple,
    indices: Sequence[int],
) -> GradientTuple:
    values = tuple(gradient[int(index)] for index in indices)
    if not values:
        raise ValueError("V25e module gradient slice is empty")
    return values


def _validate_partition(
    module_indices: Mapping[str, Sequence[int]], width: int
) -> None:
    if tuple(module_indices) != V25B_MODULE_NAMES:
        raise ValueError("V25e module partition order differs from the frozen V25a partition")
    flat = [int(index) for indices in module_indices.values() for index in indices]
    if sorted(flat) != list(range(width)) or len(flat) != len(set(flat)):
        raise ValueError("V25e module partition does not cover parameters exactly once")


def project_dominant_gradient(
    known_gradient: GradientTuple,
    clutter_gradient: GradientTuple,
) -> tuple[GradientTuple, dict[str, Any]]:
    """Remove the dominant vector's component opposing the smaller vector.

    The smaller-norm parent gradient is retained exactly.  On a negative dot
    product, the larger-norm gradient is projected onto the orthogonal
    complement of the smaller gradient.  The returned update is their sum.
    Ties choose KnownTarget as the dominant side deterministically.
    """

    if len(known_gradient) != len(clutter_gradient) or not known_gradient:
        raise ValueError("V25e parent gradient vectors are incompatible")
    known_l2 = gradient_l2(known_gradient)
    clutter_l2 = gradient_l2(clutter_gradient)
    dot = gradient_dot(known_gradient, clutter_gradient)
    cosine = gradient_cosine(known_gradient, clutter_gradient)
    if known_l2 >= clutter_l2:
        dominant_name = "known"
        dominant = known_gradient
        smaller_name = "clutter"
        smaller = clutter_gradient
        dominant_l2 = known_l2
    else:
        dominant_name = "clutter"
        dominant = clutter_gradient
        smaller_name = "known"
        smaller = known_gradient
        dominant_l2 = clutter_l2
    smaller_l2 = min(known_l2, clutter_l2)
    projected = False
    coefficient = 0.0
    projected_dominant = dominant
    if dot < 0.0 and smaller_l2 > V25E_PROJECTION_EPSILON:
        smaller_sq = gradient_dot(smaller, smaller)
        if smaller_sq > V25E_PROJECTION_EPSILON:
            coefficient = dot / smaller_sq
            projected_dominant = _subtract(dominant, _scale(smaller, coefficient))
            projected = True
    updated = _add(smaller, projected_dominant)
    post_cosine = gradient_cosine(smaller, projected_dominant)
    delta = _subtract(updated, _add(known_gradient, clutter_gradient))
    record: dict[str, Any] = {
        "projected": projected,
        "dominant_parent": dominant_name,
        "retained_parent": smaller_name,
        "pre_dot": float(dot),
        "pre_cosine": None if cosine is None else float(cosine),
        "post_cosine": None if post_cosine is None else float(post_cosine),
        "known_gradient_l2": float(known_l2),
        "clutter_gradient_l2": float(clutter_l2),
        "dominant_gradient_l2": float(dominant_l2),
        "retained_gradient_l2": float(smaller_l2),
        "projection_coefficient": float(coefficient),
        "projection_delta_l2": float(gradient_l2(delta)),
        "updated_gradient_l2": float(gradient_l2(updated)),
    }
    return updated, record


def project_shared_module_gradients(
    known_gradient: GradientTuple,
    clutter_gradient: GradientTuple,
    module_indices: Mapping[str, Sequence[int]],
) -> tuple[dict[str, GradientTuple], dict[str, dict[str, Any]]]:
    """Project only V25e's sealed shared module set."""

    _validate_partition(module_indices, len(known_gradient))
    if len(clutter_gradient) != len(known_gradient):
        raise ValueError("V25e Known/Clutter gradient widths differ")
    updated: dict[str, GradientTuple] = {}
    audit: dict[str, dict[str, Any]] = {}
    for module in V25B_MODULE_NAMES:
        known_module = _module_gradient(known_gradient, module_indices[module])
        clutter_module = _module_gradient(clutter_gradient, module_indices[module])
        if module in V25E_PROJECTED_MODULE_NAMES:
            value, record = project_dominant_gradient(known_module, clutter_module)
            record["module"] = module
            updated[module] = value
            audit[module] = record
        else:
            original = _add(known_module, clutter_module)
            updated[module] = original
            audit[module] = {
                "module": module,
                "projected": False,
                "unchanged": True,
                "pre_cosine": gradient_cosine(known_module, clutter_module),
                "post_cosine": gradient_cosine(known_module, clutter_module),
                "projection_delta_l2": 0.0,
                "updated_gradient_l2": float(gradient_l2(original)),
            }
    return updated, audit




def apply_historical_mixed_batch_projection(
    *,
    parameters: Sequence[torch.nn.Parameter],
    module_indices: Mapping[str, Sequence[int]],
    sequence_logits: torch.Tensor,
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    class_weights: torch.Tensor,
    prefix_auxiliary_weight: float,
    prefix_auxiliary_min_events: int,
    original_loss: torch.Tensor,
) -> dict[str, Any]:
    """Apply V25e's sole training intervention after historical backward.

    The caller must first run the original V25a scalar loss through
    ``backward(retain_graph=True)``.  This function computes only an additive
    replacement delta for the sealed shared modules.  Therefore all parameters
    outside ``tr_encoder``, ``fusion`` and ``liquid`` retain the original fused
    V25a gradient bit-for-bit, and a pure-class batch never calls this helper.
    """

    if len(parameters) <= 0:
        raise ValueError("V25e requires at least one trainable parameter")
    _validate_partition(module_indices, len(parameters))
    if labels.shape != (2,):
        raise ValueError("V25e projection requires exactly two parents")
    target_values = [int(value) for value in labels.detach().cpu().tolist()]
    if sorted(target_values) != [0, 1]:
        raise ValueError("V25e projection requires exactly one Known and one Clutter parent")
    if original_loss.ndim != 0 or not original_loss.requires_grad:
        raise ValueError("V25e original historical loss must be a differentiable scalar")

    terms = parent_loss_terms_batch_weight_sum(
        sequence_logits,
        labels,
        padding_mask,
        class_weights=class_weights,
        prefix_auxiliary_weight=prefix_auxiliary_weight,
        prefix_auxiliary_min_events=prefix_auxiliary_min_events,
    )
    reconstruction_error = abs(
        float(terms["historical_total_loss"].detach().item())
        - float(original_loss.detach().item())
    )
    if reconstruction_error > V25E_LOSS_RECONSTRUCTION_ABSOLUTE_TOLERANCE:
        raise AssertionError(
            "V25e explicit historical parent terms do not reconstruct the source loss: "
            f"absolute_error={reconstruction_error}"
        )
    before = snapshot_parameter_gradients(parameters)
    known_index = target_values.index(0)
    clutter_index = target_values.index(1)
    known_gradient = gradients_from_scalar(
        terms["historical_total_terms"][known_index],
        parameters,
        retain_graph=True,
    )
    clutter_gradient = gradients_from_scalar(
        terms["historical_total_terms"][clutter_index],
        parameters,
        retain_graph=False,
    )
    module_updates, module_audit = project_shared_module_gradients(
        known_gradient, clutter_gradient, module_indices
    )

    with torch.no_grad():
        for module in V25E_PROJECTED_MODULE_NAMES:
            indices = tuple(int(index) for index in module_indices[module])
            explicit = tuple(
                known_gradient[index] + clutter_gradient[index] for index in indices
            )
            transformed = module_updates[module]
            if len(explicit) != len(transformed):
                raise AssertionError("V25e projected module widths differ")
            for parameter_index, source, destination in zip(
                indices, explicit, transformed, strict=True
            ):
                gradient = parameters[parameter_index].grad
                if gradient is None:
                    raise AssertionError("V25e historical backward left a projected gradient absent")
                gradient.add_(destination - source)

    after = snapshot_parameter_gradients(parameters)
    unchanged_modules = {
        module: all(
            torch.equal(before[int(index)], after[int(index)])
            for index in module_indices[module]
        )
        for module in V25E_UNCHANGED_MODULE_NAMES
    }
    if not all(unchanged_modules.values()):
        raise AssertionError("V25e modified an RD or classifier gradient")
    for module in V25E_PROJECTED_MODULE_NAMES:
        record = module_audit[module]
        post = record.get("post_cosine")
        if record["projected"] and post is not None and float(post) < -1.0e-6:
            raise AssertionError("V25e projected shared gradients remain negatively aligned")
    changed_modules = {
        module: bool(module_audit[module]["projected"])
        for module in V25E_PROJECTED_MODULE_NAMES
    }
    return {
        "mixed_batch": True,
        "historical_loss_reconstruction_absolute_error": reconstruction_error,
        "projected_module_count": int(sum(changed_modules.values())),
        "projected_modules": changed_modules,
        "shared_module_projection": {
            module: module_audit[module] for module in V25E_PROJECTED_MODULE_NAMES
        },
        "unchanged_modules_exactly_equal_to_historical_backward": unchanged_modules,
        "rd_and_classifier_gradients_exactly_unchanged": True,
        "projection_applies_only_to_historical_parent_gradient_proxy": True,
    }


__all__ = ["V25E_PROJECTED_MODULE_NAMES", "V25E_SHARED_MODULE_NAMES", "V25E_UNCHANGED_MODULE_NAMES", "apply_historical_mixed_batch_projection", "project_dominant_gradient", "project_shared_module_gradients"]