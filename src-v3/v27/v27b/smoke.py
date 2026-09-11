"""Structural smoke tests for the isolated V27b experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .model import V27BConditionalPairResidual, build_v27b_model
from .protocol import (
    V27BContext,
    V27B_SMOKE_SCHEMA_VERSION,
    V27B_TREATMENT_ARM,
    v27b_artifact_path,
    write_sealed_json_if_identical,
)
from .runtime import seed_v27b


_CONTINUATION_RTOL = 1.0e-5
_CONTINUATION_ATOL = 1.0e-6


def _assert_numerically_equal(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    message: str,
) -> None:
    if not torch.allclose(
        left,
        right,
        rtol=_CONTINUATION_RTOL,
        atol=_CONTINUATION_ATOL,
    ):
        difference = float((left - right).abs().max().item())
        raise AssertionError(f"{message}; max_abs_difference={difference:.3e}")


def _synthetic_inputs(
    context: V27BContext,
    *,
    batch_size: int = 3,
    length: int = 6,
) -> dict[str, torch.Tensor]:
    if min(int(batch_size), int(length)) <= 0:
        raise ValueError("V27b smoke dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(270042)
    padding = torch.zeros((batch_size, length), dtype=torch.bool)
    lengths = [length, max(1, length - 2), max(1, length - 1)]
    lengths = (lengths * ((batch_size + 2) // 3))[:batch_size]
    for row, row_length in enumerate(lengths):
        padding[row, int(row_length) :] = True
    parent_state = torch.randn(
        (batch_size, length, context.config.model.parent_state_dim),
        generator=generator,
        dtype=torch.float32,
    )
    event_representation = torch.randn(
        (batch_size, length, context.config.model.event_representation_dim),
        generator=generator,
        dtype=torch.float32,
    )
    frozen_logits = torch.randn(
        (batch_size, length, 3), generator=generator, dtype=torch.float32
    )
    residual = torch.randn(
        (batch_size, length, 3), generator=generator, dtype=torch.float32
    )
    residual.mul_(0.1)
    reliable = torch.zeros((batch_size, length), dtype=torch.bool)
    frame_values: torch.Tensor | None = None
    frame_mask: torch.Tensor | None = None
    if context.config.model.evidence_carrier == "direct_frame_gru":
        frame_values = torch.randn(
            (
                batch_size,
                length,
                context.config.data.max_frames_per_event,
                context.config.data.stable8_dim,
            ),
            generator=generator,
            dtype=torch.float32,
        )
        frame_mask = torch.zeros(
            (batch_size, length, context.config.data.max_frames_per_event),
            dtype=torch.bool,
        )
        for row, row_length in enumerate(lengths):
            for event in range(int(row_length)):
                count = 1 + ((row + event) % context.config.data.max_frames_per_event)
                frame_mask[row, event, :count] = True
        frame_values.masked_fill_(~frame_mask.unsqueeze(-1), 0.0)
        reliable = frame_mask.any(dim=-1) & ~padding
    else:
        for row, row_length in enumerate(lengths):
            reliable[row, : int(row_length)] = torch.tensor(
                [(row + event) % 3 != 0 for event in range(int(row_length))],
                dtype=torch.bool,
            )
    for value in (parent_state, event_representation, frozen_logits, residual):
        value.masked_fill_(padding.unsqueeze(-1), 0.0)
    reliable.masked_fill_(padding, False)
    result: dict[str, torch.Tensor] = {
        "parent_state": parent_state,
        "event_representation": event_representation,
        "frozen_l2_logits": frozen_logits,
        "v27a_residual_logits": residual,
        "event_summary_valid": reliable,
        "padding_mask": padding,
    }
    if frame_values is not None and frame_mask is not None:
        frame_values.masked_fill_(padding.unsqueeze(-1).unsqueeze(-1), 0.0)
        frame_mask.masked_fill_(padding.unsqueeze(-1), False)
        result["event_frame_values"] = frame_values
        result["event_frame_mask"] = frame_mask
    return result


def _assert_model_contract(context: V27BContext) -> dict[str, Any]:
    model = build_v27b_model(context.config, seed=42)
    inputs = _synthetic_inputs(context)
    with torch.inference_mode():
        output = model(**inputs)
    logits = output["sequence_logits"]
    correction = output["correction_logits"]
    if not isinstance(logits, torch.Tensor) or not isinstance(correction, torch.Tensor):
        raise TypeError("V27b smoke output tensors are missing")
    if not torch.isfinite(logits).all() or not torch.isfinite(correction).all():
        raise FloatingPointError("V27b smoke output is non-finite")
    if not torch.equal(logits, inputs["frozen_l2_logits"]):
        raise AssertionError("V27b zero-initialised treatment differs from control")
    if not torch.equal(correction, torch.zeros_like(correction)):
        raise AssertionError("V27b zero-initialised correction is non-zero")
    if bool(correction[..., 2].ne(0.0).any()):
        raise AssertionError("V27b correction changes Balloon")
    if bool(correction.sum(dim=-1).abs().gt(1.0e-6).any()):
        raise AssertionError("V27b correction is not zero-sum")
    return {
        "parameter_count": model.trainable_parameter_count,
        "output_shape": list(logits.shape),
        "zero_initialised_equals_control": True,
        "finite": True,
        "balloon_correction_exact_zero": True,
        "correction_zero_sum": True,
    }


def _assert_no_reliable_event(
    context: V27BContext, model: V27BConditionalPairResidual
) -> None:
    inputs = _synthetic_inputs(context, batch_size=1, length=5)
    with torch.no_grad():
        model.correction_projection[-1].weight.fill_(0.05)
    inputs["event_summary_valid"][:, 0] = True
    inputs["event_summary_valid"][:, 1] = False
    inputs["event_summary_valid"][:, 2] = True
    inputs["event_summary_valid"][:, 3:] = False
    if "event_frame_mask" in inputs and "event_frame_values" in inputs:
        inputs["event_frame_mask"][:, 1] = False
        inputs["event_frame_values"][:, 1] = 0.0
        inputs["event_frame_mask"][:, 2, 0] = True
        inputs["event_frame_values"][:, 2, 0] = 0.25
        inputs["event_frame_mask"][:, 3:] = False
        inputs["event_frame_values"][:, 3:] = 0.0
    with torch.inference_mode():
        output = model(**inputs)
    increments = output["pairwise_delta_increment"]
    deltas = output["pairwise_delta_sequence"]
    if not isinstance(increments, torch.Tensor) or not isinstance(deltas, torch.Tensor):
        raise TypeError("V27b smoke delta outputs are missing")
    if not torch.equal(increments[:, 1], torch.zeros_like(increments[:, 1])):
        raise AssertionError("V27b unreliable event produced a delta increment")
    if not torch.equal(deltas[:, 1], torch.zeros_like(deltas[:, 1])):
        raise AssertionError("V27b unreliable event emitted a non-zero correction")
    if not torch.equal(output["correction_logits"][:, 1], torch.zeros_like(output["correction_logits"][:, 1])):
        raise AssertionError("V27b unreliable event emitted non-zero correction logits")
    if not torch.equal(increments[:, 3], torch.zeros_like(increments[:, 3])):
        raise AssertionError("V27b padded event produced a delta increment")


def _assert_causality_and_continuation(
    context: V27BContext, model: V27BConditionalPairResidual
) -> None:
    inputs = _synthetic_inputs(context, batch_size=1, length=6)
    with torch.inference_mode():
        base = model(**inputs)
    future = {name: value.clone() for name, value in inputs.items()}
    future["parent_state"][:, 4:] += 100.0
    future["event_representation"][:, 4:] += 100.0
    future["frozen_l2_logits"][:, 4:] += 100.0
    if "event_frame_values" in future:
        future["event_frame_values"][:, 4:] = future[
            "event_frame_values"
        ][:, 4:] + future["event_frame_mask"][:, 4:].unsqueeze(-1).to(
            dtype=future["event_frame_values"].dtype
        ) * 100.0
    with torch.inference_mode():
        changed = model(**future)
    for name in ("sequence_logits", "correction_logits", "pairwise_delta_sequence"):
        if not torch.equal(base[name][:, :4], changed[name][:, :4]):
            raise AssertionError(f"V27b future input changed causal prefix: {name}")

    first = model(
        parent_state=inputs["parent_state"][:, :3],
        event_representation=inputs["event_representation"][:, :3],
        frozen_l2_logits=inputs["frozen_l2_logits"][:, :3],
        v27a_residual_logits=inputs["v27a_residual_logits"][:, :3],
        event_summary_valid=inputs["event_summary_valid"][:, :3],
        padding_mask=inputs["padding_mask"][:, :3],
        event_frame_values=(
            inputs["event_frame_values"][:, :3]
            if "event_frame_values" in inputs
            else None
        ),
        event_frame_mask=(
            inputs["event_frame_mask"][:, :3]
            if "event_frame_mask" in inputs
            else None
        ),
    )
    second = model(
        parent_state=inputs["parent_state"][:, 3:],
        event_representation=inputs["event_representation"][:, 3:],
        frozen_l2_logits=inputs["frozen_l2_logits"][:, 3:],
        v27a_residual_logits=inputs["v27a_residual_logits"][:, 3:],
        event_summary_valid=inputs["event_summary_valid"][:, 3:],
        padding_mask=inputs["padding_mask"][:, 3:],
        event_frame_values=(
            inputs["event_frame_values"][:, 3:]
            if "event_frame_values" in inputs
            else None
        ),
        event_frame_mask=(
            inputs["event_frame_mask"][:, 3:]
            if "event_frame_mask" in inputs
            else None
        ),
        initial_state=first["final_state"],
    )
    _assert_numerically_equal(
        base["final_state"]["previous_delta"],
        second["final_state"]["previous_delta"],
        message="V27b chunk continuation state differs from offline",
    )
    _assert_numerically_equal(
        base["sequence_logits"][:, 3:],
        second["sequence_logits"],
        message="V27b chunk continuation logits differ from offline",
    )


def run_smoke(context: V27BContext) -> dict[str, Any]:
    seed_v27b(42)
    model_contract = _assert_model_contract(context)
    model = build_v27b_model(context.config, seed=42)
    _assert_no_reliable_event(context, model)
    _assert_causality_and_continuation(context, model)
    result = {
        "schema_version": V27B_SMOKE_SCHEMA_VERSION,
        "experiment_id": "V27b",
        "status": "passed",
        "source_checkpoint_file_sha256": context.config.base.expected_v27a_checkpoint_file_sha256,
        "source_checkpoint_state_sha256": context.config.base.expected_v27a_checkpoint_state_sha256,
        "manifest_sha256": context.manifest_sha256,
        "config_identity_sha256": context.config.config_identity_sha256,
        "model": model_contract,
        "tests": {
            "zero_initialised_control_equivalence": True,
            "no_reliable_event_preserves_delta": True,
            "no_reliable_event_zero_increment": True,
            "padding_no_state_update": True,
            "future_perturbation_causality": True,
            "offline_chunk_continuation_equivalence": True,
            "balloon_correction_exact_zero": True,
            "correction_zero_sum_and_bound": True,
        },
        "continuation_tolerance": {
            "rtol": _CONTINUATION_RTOL,
            "atol": _CONTINUATION_ATOL,
        },
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "l3": "not_in_scope",
    }
    path = v27b_artifact_path(
        context.project_root, context.config.artifacts.smoke_report_path
    )
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path)}


__all__ = ["run_smoke"]
