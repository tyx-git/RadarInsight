"""Contract-level smoke tests for the V27a residual experiment."""

from __future__ import annotations

from typing import Any

import torch

from .model import V27AEvidenceResidual, build_v27a_model
from .protocol import (
    V27A_ARMS,
    V27AContext,
    v27a_artifact_path,
    write_sealed_json_if_identical,
)
from .runtime import seed_v27a


def _synthetic_inputs(
    *,
    batch_size: int = 3,
    length: int = 5,
    frame_width: int = 4,
) -> dict[str, torch.Tensor]:
    if min(int(batch_size), int(length), int(frame_width)) <= 0:
        raise ValueError("V27a smoke dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(270042)
    baseline = torch.randn(
        (batch_size, length, 3), generator=generator, dtype=torch.float32
    )
    padding = torch.zeros((batch_size, length), dtype=torch.bool)
    lengths = [length, max(1, length - 2), max(1, length - 1)]
    lengths = (lengths * ((batch_size + 2) // 3))[:batch_size]
    for row, row_length in enumerate(lengths):
        padding[row, int(row_length) :] = True
    frame_counts = torch.zeros((batch_size, length), dtype=torch.long)
    for row in range(batch_size):
        for event in range(lengths[row]):
            frame_counts[row, event] = (row + event) % (frame_width + 1)
    frame_mask = torch.zeros(
        (batch_size, length, frame_width), dtype=torch.bool
    )
    for row in range(batch_size):
        for event in range(length):
            count = int(frame_counts[row, event])
            if not padding[row, event]:
                frame_mask[row, event, :count] = True
    summary_valid = frame_mask.any(dim=-1) & ~padding
    summary = torch.randn(
        (batch_size, length, 8), generator=generator, dtype=torch.float32
    )
    frame_values = torch.randn(
        (batch_size, length, frame_width, 8),
        generator=generator,
        dtype=torch.float32,
    )
    summary.masked_fill_(~summary_valid.unsqueeze(-1), 0.0)
    frame_values.masked_fill_(~frame_mask.unsqueeze(-1), 0.0)
    baseline.masked_fill_(padding.unsqueeze(-1), 0.0)
    return {
        "baseline_logits": baseline,
        "summary_tokens": summary,
        "summary_valid": summary_valid,
        "frame_values": frame_values,
        "frame_mask": frame_mask,
        "padding_mask": padding,
    }


def _assert_finite(values: tuple[torch.Tensor, ...], message: str) -> None:
    if not all(torch.isfinite(value).all() for value in values):
        raise FloatingPointError(message)


def _assert_same_state(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    *,
    message: str,
) -> None:
    if set(left) != set(right) or any(
        not torch.allclose(left[name], right[name], atol=1.0e-6, rtol=1.0e-6)
        for name in left
    ):
        raise AssertionError(message)


def _assert_zero_init_and_capacity(context: V27AContext) -> dict[str, Any]:
    mechanism = build_v27a_model(context.config, arm="mechanism", seed=42)
    main = build_v27a_model(context.config, arm="main", seed=42)
    if mechanism.parameter_shapes != main.parameter_shapes:
        raise AssertionError("V27a treatment parameter shapes differ")
    if mechanism.trainable_parameter_count != main.trainable_parameter_count:
        raise AssertionError("V27a treatment parameter counts differ")
    for left, right in zip(
        mechanism.state_dict().items(), main.state_dict().items(), strict=True
    ):
        if left[0] != right[0] or not torch.equal(left[1], right[1]):
            raise AssertionError("V27a treatment initialization is not identical")
    if not torch.equal(
        mechanism.residual_projection.weight,
        torch.zeros_like(mechanism.residual_projection.weight),
    ):
        raise AssertionError("V27a residual projection is not zero initialized")
    return {
        "mechanism_parameter_count": mechanism.trainable_parameter_count,
        "main_parameter_count": main.trainable_parameter_count,
        "parameter_shapes_identical": True,
        "initialization_identical": True,
        "residual_zero_initialized": True,
    }


def _assert_model_contract(context: V27AContext, arm: str) -> dict[str, Any]:
    model = build_v27a_model(context.config, arm=arm, seed=42)
    inputs = _synthetic_inputs()
    with torch.inference_mode():
        output = model(**inputs)
    _assert_finite(
        (
            output["sequence_logits"],
            output["residual_logits"],
            output["residual_increment"],
            output["evidence_update"],
        ),
        "V27a smoke model output is non-finite",
    )
    if not torch.equal(output["baseline_logits"], inputs["baseline_logits"]):
        raise AssertionError("V27a model changed the frozen baseline logits")
    active = ~inputs["padding_mask"]
    residual = output["residual_logits"]
    if bool(residual[active].abs().max().item() > context.config.model.residual_bound + 1.0e-6):
        raise AssertionError("V27a residual exceeds the registered bound")
    if bool(residual[active].sum(dim=-1).abs().max().item() > 1.0e-6):
        raise AssertionError("V27a residual is not zero-sum")
    if not torch.equal(output["sequence_logits"], inputs["baseline_logits"]):
        raise AssertionError("V27a zero-init output differs from control")
    return {
        "arm": arm,
        "output_shape": list(output["sequence_logits"].shape),
        "zero_init_equals_control": True,
        "finite": True,
        "residual_bound_passed": True,
        "residual_zero_sum_passed": True,
    }


def _assert_no_reliable_event(model: V27AEvidenceResidual) -> None:
    inputs = _synthetic_inputs(batch_size=1, length=4, frame_width=3)
    with torch.no_grad():
        model.residual_projection.weight.zero_()
        model.residual_projection.weight[0].fill_(0.25)
        model.residual_projection.weight[2].fill_(-0.25)
    for event_index in (0, 2):
        inputs["frame_mask"][:, event_index] = False
        inputs["frame_values"][:, event_index] = 0.0
        inputs["summary_valid"][:, event_index] = False
        inputs["summary_tokens"][:, event_index] = 0.0
    inputs["frame_mask"][:, 1, :1] = True
    inputs["summary_valid"][:, 1] = True
    with torch.inference_mode():
        output = model(**inputs)
    for event_index in (0, 2):
        if not torch.equal(
            output["evidence_update"][:, event_index],
            torch.zeros_like(output["evidence_update"][:, event_index]),
        ):
            raise AssertionError("V27a no-reliable event produced an evidence update")
        if not torch.equal(
            output["residual_increment"][:, event_index],
            torch.zeros_like(output["residual_increment"][:, event_index]),
        ):
            raise AssertionError("V27a no-reliable event produced a residual increment")
    if not torch.equal(
        output["residual_logits"][:, 0],
        torch.zeros_like(output["residual_logits"][:, 0]),
    ):
        raise AssertionError("V27a first no-reliable event retained non-zero residual state")
    if not torch.allclose(
        output["residual_logits"][:, 2],
        output["residual_logits"][:, 1],
        atol=0.0,
        rtol=0.0,
    ):
        raise AssertionError("V27a no-reliable event did not preserve prior residual state")


def _assert_padding_and_future_causality(model: V27AEvidenceResidual) -> None:
    inputs = _synthetic_inputs(batch_size=1, length=5, frame_width=3)
    with torch.inference_mode():
        base = model(**inputs)
    padded = {name: value.clone() for name, value in inputs.items()}
    padded["padding_mask"][:, 3:] = True
    padded["baseline_logits"][:, 3:] = 0.0
    padded["summary_tokens"][:, 3:] = 0.0
    padded["summary_valid"][:, 3:] = False
    padded["frame_values"][:, 3:] = 0.0
    padded["frame_mask"][:, 3:] = False
    with torch.inference_mode():
        shortened = model(**padded)
    for name in ("sequence_logits", "residual_logits", "evidence_update"):
        if not torch.allclose(
            base[name][:, :3], shortened[name][:, :3], atol=1.0e-6, rtol=1.0e-6
        ):
            raise AssertionError(f"V27a padding changed active output: {name}")

    future = {name: value.clone() for name, value in inputs.items()}
    future["baseline_logits"][:, 3:] += 100.0
    future["summary_tokens"][:, 3:] = future["summary_tokens"][:, 3:].masked_fill(
        future["summary_valid"][:, 3:].unsqueeze(-1),
        future["summary_tokens"][:, 3:][future["summary_valid"][:, 3:]].new_tensor(100.0),
    )
    future["frame_values"][:, 3:] = future["frame_values"][:, 3:].masked_fill(
        future["frame_mask"][:, 3:].unsqueeze(-1),
        future["frame_values"][:, 3:][future["frame_mask"][:, 3:]].new_tensor(100.0),
    )
    with torch.inference_mode():
        perturbed = model(**future)
    for name in ("sequence_logits", "residual_logits", "evidence_update"):
        if not torch.allclose(
            base[name][:, :3], perturbed[name][:, :3], atol=1.0e-6, rtol=1.0e-6
        ):
            raise AssertionError(f"V27a future input changed prefix output: {name}")


def _assert_different_length_alignment(model: V27AEvidenceResidual) -> None:
    batch = _synthetic_inputs(batch_size=2, length=5, frame_width=3)
    batch["padding_mask"][1, 3:] = True
    for name in ("baseline_logits", "summary_tokens", "frame_values"):
        batch[name][1, 3:] = 0.0
    batch["summary_valid"][1, 3:] = False
    batch["frame_mask"][1, 3:] = False
    single = {name: value[1:2].clone() for name, value in batch.items()}
    with torch.inference_mode():
        batched = model(**batch)
        alone = model(**single)
    if not torch.allclose(
        batched["sequence_logits"][1, :3],
        alone["sequence_logits"][0, :3],
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError("V27a different-length batch output is misaligned")


def _assert_continuation(model: V27AEvidenceResidual) -> None:
    inputs = _synthetic_inputs(batch_size=1, length=5, frame_width=3)
    with torch.inference_mode():
        offline = model(**inputs)
        first = model(
            inputs["baseline_logits"][:, :2],
            inputs["summary_tokens"][:, :2],
            inputs["summary_valid"][:, :2],
            inputs["frame_values"][:, :2],
            inputs["frame_mask"][:, :2],
            inputs["padding_mask"][:, :2],
        )
        second = model(
            inputs["baseline_logits"][:, 2:],
            inputs["summary_tokens"][:, 2:],
            inputs["summary_valid"][:, 2:],
            inputs["frame_values"][:, 2:],
            inputs["frame_mask"][:, 2:],
            inputs["padding_mask"][:, 2:],
            initial_state=first["final_state"],
        )
    _assert_same_state(
        offline["final_state"],
        second["final_state"],
        message="V27a chunk continuation state differs from offline",
    )
    if not torch.allclose(
        offline["sequence_logits"][:, 2:],
        second["sequence_logits"],
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError("V27a chunk continuation logits differ from offline")

    state = model.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    online_rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for index in range(inputs["baseline_logits"].shape[1]):
            step = model.step_event(
                summary_tokens=inputs["summary_tokens"][:, index],
                summary_valid=inputs["summary_valid"][:, index],
                frame_values=inputs["frame_values"][:, index],
                frame_mask=inputs["frame_mask"][:, index],
                state=state,
            )
            state = {
                "parent_state": step["parent_state"],
                "previous_residual": step["previous_residual"],
            }
            online_rows.append(step["residual"])
    if not torch.allclose(
        offline["final_state"]["parent_state"], state["parent_state"], atol=1.0e-6, rtol=1.0e-6
    ) or not torch.allclose(
        offline["final_state"]["previous_residual"], state["previous_residual"], atol=1.0e-6, rtol=1.0e-6
    ):
        raise AssertionError("V27a online continuation state differs from offline")
    online_residual = torch.stack(online_rows, dim=1)
    if not torch.allclose(
        offline["residual_logits"], online_residual, atol=1.0e-6, rtol=1.0e-6
    ):
        raise AssertionError("V27a online continuation residual differs from offline")


def run_smoke(context: V27AContext) -> dict[str, Any]:
    """Run structural tests without opening Unknown/test or real caches."""

    seed_v27a(42)
    capacity = _assert_zero_init_and_capacity(context)
    arm_reports: dict[str, Any] = {}
    for arm in V27A_ARMS:
        arm_reports[arm] = _assert_model_contract(context, arm)
    for arm in V27A_ARMS:
        model = build_v27a_model(context.config, arm=arm, seed=42)
        _assert_no_reliable_event(model)
        _assert_padding_and_future_causality(model)
        _assert_different_length_alignment(model)
        _assert_continuation(model)
    result = {
        "schema_version": "RADAR-LN3-V27a-contract-smoke-v2",
        "experiment_id": "V27a",
        "status": "passed",
        "base_epoch": context.config.base.expected_l2_epoch,
        "base_checkpoint_sha256": context.config.base.expected_l2_checkpoint_sha256,
        "base_classifier_state_sha256": context.config.base.expected_l2_classifier_state_sha256,
        "manifest_sha256": context.manifest_sha256,
        "capacity": capacity,
        "arms": arm_reports,
        "tests": {
            "no_reliable_event_exact_zero": True,
            "padding_invariance": True,
            "future_perturbation_causality": True,
            "different_length_batch_alignment": True,
            "offline_chunk_online_equivalence": True,
        },
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "knownness": "not_in_scope",
        "quality": "not_in_scope",
        "l3": "not_in_scope",
    }
    report_path = v27a_artifact_path(
        context.project_root, context.config.artifacts.smoke_report_path
    )
    write_sealed_json_if_identical(report_path, result)
    return {**result, "report_path": str(report_path)}


__all__ = ["run_smoke"]
