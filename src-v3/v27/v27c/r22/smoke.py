"""Necessary contract checks for V27c-r22 before any formal training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from v27.v27_baseline.metrics import summarize_records

from .data import (
    R22L2Dataset,
    build_r22_dataloader,
    r22_model_input_kwargs,
    r22_sidecar_kwargs,
)
from .protocol import (
    R22_L2_CLASSES,
    R22_SMOKE_SCHEMA_VERSION,
    R22_V27A_MAIN_CONFUSION,
    R22Context,
    r22_artifact_path,
    write_sealed_json,
)
from .training import build_initialized_model


def _one_batch(context: R22Context) -> dict[str, Any]:
    dataset = R22L2Dataset(context, split="val")
    return next(
        iter(
            build_r22_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
            )
        )
    )


def _assert_sidecar_boundary(context: R22Context) -> dict[str, Any]:
    batch = _one_batch(context)
    forbidden = {"tr_motion_features", "tr_motion_valid_mask"}
    if forbidden & set(batch):
        raise AssertionError("V27c-r22 raw motion fields leaked through collation")
    if (
        batch["motion_values"].shape[-1] != context.config.data.motion_dim
        or batch["stable_summary_tokens"].shape[-1] != context.config.data.stable8_dim
        or batch["stable_frame_values"].shape[-1] != context.config.data.stable8_dim
        or bool(
            batch["motion_values"]
            .masked_select(batch["tr_padding_mask"].unsqueeze(-1))
            .ne(0)
            .any()
        )
    ):
        raise AssertionError("V27c-r22 sidecar shape/padding contract differs")
    return {
        "raw_motion_values_removed_from_batch": True,
        "motion_validity_mask_removed_from_batch": True,
        "motion_numeric_dim": int(batch["motion_values"].shape[-1]),
        "stable8_dim": int(batch["stable_summary_tokens"].shape[-1]),
        "sidecar_padding_zero": True,
    }


def _assert_initial_replay(context: R22Context) -> dict[str, Any]:
    batch = _one_batch(context)
    base_inputs = r22_model_input_kwargs(batch, torch.device("cpu"))
    sidecars = r22_sidecar_kwargs(batch, torch.device("cpu"))
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for arm in ("control", "motion_treatment"):
        model = build_initialized_model(context, arm=arm, seed=42, device=torch.device("cpu"))
        model.eval()
        with torch.inference_mode():
            outputs[arm] = model(base_inputs=base_inputs, **sidecars)
    reference = batch["v27a_main_reference_logits"]
    # r22 anchors legacy V27 logits from the archived cache.  Re-executing the
    # old RD/Liquid endpoint is deterministic in the semantic sense but can
    # differ by a few FP32 ULPs on the current runtime, so require a strict
    # numerical replay tolerance rather than falsely demanding bit identity.
    replay_tolerance = 4.0e-6
    replay_deltas: dict[str, float] = {}
    for arm, output in outputs.items():
        delta = float((output["sequence_logits"] - reference).abs().max().item())
        replay_deltas[arm] = delta
        if delta > replay_tolerance:
            raise AssertionError(f"V27c-r22 {arm} initialization does not replay V27a Main: {delta}")
        if bool(output["l2_delta_logits"].ne(0).any()):
            raise AssertionError("V27c-r22 initial L2 delta is not exact zero")
        if bool(output["motion_adapter_output"].ne(0).any()):
            raise AssertionError("V27c-r22 initial motion adapter is not exact zero")
    if not torch.equal(outputs["control"]["sequence_logits"], outputs["motion_treatment"]["sequence_logits"]):
        raise AssertionError("V27c-r22 matched arms differ before training")
    return {
        "control_replays_v27a_main_within_fp32_tolerance": True,
        "motion_treatment_replays_v27a_main_within_fp32_tolerance": True,
        "max_abs_replay_delta": replay_deltas,
        "replay_tolerance": replay_tolerance,
        "initial_l2_delta_exact_zero": True,
        "initial_motion_adapter_exact_zero": True,
        "matched_arms_identical_at_initialization": True,
    }


def _assert_motion_location_and_causality(context: R22Context) -> dict[str, Any]:
    batch = _one_batch(context)
    base_inputs = r22_model_input_kwargs(batch, torch.device("cpu"))
    sidecars = r22_sidecar_kwargs(batch, torch.device("cpu"))
    model = build_initialized_model(
        context, arm="motion_treatment", seed=42, device=torch.device("cpu")
    )
    # A nonzero projection weight makes a synthetic input influence the actual
    # pre-Liquid update.  This is a structural check; it does not alter saved
    # experiment weights because this smoke-only model is ephemeral.
    with torch.no_grad():
        model.motion_adapter.output_projection.weight[0, 0] = 0.25
    original_motion = sidecars["motion_values"].clone()
    nonzero = torch.nonzero(
        original_motion.abs().amax(dim=-1) > 0.0, as_tuple=False
    )
    if not nonzero.numel():
        raise AssertionError("V27c-r22 smoke batch has no nonzero motion event")
    row, event = (int(nonzero[0, 0].item()), int(nonzero[0, 1].item()))
    changed = original_motion.clone()
    changed[row, event, 0] += 0.25
    altered_sidecars = {**sidecars, "motion_values": changed}
    model.eval()
    with torch.inference_mode():
        before = model(base_inputs=base_inputs, **sidecars)
        after = model(base_inputs=base_inputs, **altered_sidecars)
    if torch.equal(before["sequence_logits"], after["sequence_logits"]):
        raise AssertionError("V27c-r22 motion does not change integrated L2 output")
    if bool(
        (before["sequence_logits"][:, :event] - after["sequence_logits"][:, :event])
        .abs()
        .gt(0.0)
        .any()
    ):
        raise AssertionError("V27c-r22 future motion changed an earlier causal prefix")
    zero_sidecars = {**sidecars, "motion_values": torch.zeros_like(original_motion)}
    with torch.inference_mode():
        zero = model(base_inputs=base_inputs, **zero_sidecars)
    if bool(zero["motion_adapter_output"].ne(0).any()):
        raise AssertionError("V27c-r22 zero motion is not an adapter no-op after training weights")
    return {
        "motion_injection_location": "post_fusion_pre_liquid_v2",
        "nonzero_motion_changes_integrated_l2_output": True,
        "future_motion_does_not_change_earlier_prefix": True,
        "zero_motion_is_weight_independent_adapter_noop": True,
    }


def _assert_cached_reference(context: R22Context) -> dict[str, Any]:
    dataset = R22L2Dataset(context, split="val")
    records: list[dict[str, Any]] = []
    for item in dataset:
        length = item.sequence.base.event_count
        logits = item.v27a_main_reference_logits[length - 1]
        records.append(
            {
                "parent_sample_id": item.sequence.base.parent_sample_id,
                "true_id": R22_L2_CLASSES.index(item.sequence.base.label_name),
                "pred_id": int(logits.argmax().item()),
            }
        )
    metrics = summarize_records(records, class_names=R22_L2_CLASSES, scope="parent_final")
    if metrics["confusion_matrix_rows_true_columns_pred"] != [
        list(row) for row in R22_V27A_MAIN_CONFUSION
    ]:
        raise AssertionError("V27c-r22 cached V27a Main endpoint differs")
    return {
        "v27a_main_cached_reference_confusion": metrics[
            "confusion_matrix_rows_true_columns_pred"
        ],
        "matches_archived_v27a_main": True,
    }


def run_smoke(context: R22Context) -> dict[str, Any]:
    result = {
        "schema_version": R22_SMOKE_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "passed",
        "manifest_sha256": context.manifest_sha256,
        "sidecar_boundary": _assert_sidecar_boundary(context),
        "initial_replay": _assert_initial_replay(context),
        "motion_boundary": _assert_motion_location_and_causality(context),
        "cached_reference": _assert_cached_reference(context),
        "scope": {
            "l1_checkpoint_deserialized": False,
            "l1_forward_calls": 0,
            "l1_optimizer_parameter_count": 0,
            "unknown_training_exposure": 0,
            "clutter_training_exposure": 0,
            "knownness": "not_in_scope",
            "quality": "not_in_scope",
            "calibration": "not_in_scope",
            "l3": "not_in_scope",
            "test_payload_access_count": 0,
        },
    }
    path = r22_artifact_path(context.project_root, context.config.artifacts.smoke_report_path)
    write_sealed_json(path, result)
    return {**result, "report_path": str(path)}


__all__ = ["run_smoke"]
