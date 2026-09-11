"""Necessary contract checks for V27c-r23 before any formal training."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from v27.v27_baseline.metrics import summarize_records

from .data import (
    R23L2Dataset,
    build_r23_dataloader,
    r23_model_input_kwargs,
    r23_reference_model_input_kwargs,
    r23_sidecar_kwargs,
)
from .protocol import (
    R23_L2_CLASSES,
    R23_SMOKE_SCHEMA_VERSION,
    R23_V27A_MAIN_CONFUSION,
    R23Context,
    r23_artifact_path,
    write_sealed_json,
)
from .training import build_initialized_model


def _one_batch(context: R23Context) -> dict[str, Any]:
    dataset = R23L2Dataset(context, split="val", arm="legacy_control")
    return next(
        iter(
            build_r23_dataloader(
                dataset,
                batch_size=2,
                shuffle=False,
            )
        )
    )


def _assert_sidecar_boundary(context: R23Context) -> dict[str, Any]:
    batch = _one_batch(context)
    forbidden = {"tr_motion_features", "tr_motion_valid_mask"}
    if forbidden & set(batch):
        raise AssertionError("V27c-r23 raw motion fields leaked through collation")
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
        or "motion_validity" in batch
    ):
        raise AssertionError("V27c-r23 sidecar shape/padding contract differs")
    return {
        "raw_motion_values_removed_from_batch": True,
        "motion_validity_mask_removed_from_batch": True,
        "motion_numeric_dim": int(batch["motion_values"].shape[-1]),
        "stable8_dim": int(batch["stable_summary_tokens"].shape[-1]),
        "sidecar_padding_zero": True,
    }


def _assert_initial_replay(context: R23Context) -> dict[str, Any]:
    batch = _one_batch(context)
    base_inputs = r23_model_input_kwargs(batch, torch.device("cpu"))
    sidecars = r23_sidecar_kwargs(batch, torch.device("cpu"))
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    reference_inputs = r23_reference_model_input_kwargs(batch, torch.device("cpu"))
    for arm in ("legacy_control", "legacy_motion"):
        model = build_initialized_model(context, arm=arm, seed=42, device=torch.device("cpu"))
        model.eval()
        with torch.inference_mode():
            outputs[arm] = model(
                base_inputs=base_inputs,
                reference_base_inputs=reference_inputs,
                **sidecars,
            )
    reference_model = build_initialized_model(
        context, arm="legacy_control", seed=42, device=torch.device("cpu")
    )
    reference_model.eval()
    with torch.inference_mode():
        reference = reference_model.v27a_main_endpoint(
            cached_v27_base_logits=sidecars["cached_v27_base_logits"],
            stable_summary_tokens=sidecars["stable_summary_tokens"],
            stable_summary_valid=sidecars["stable_summary_valid"],
            stable_frame_values=sidecars["stable_frame_values"],
            stable_frame_mask=sidecars["stable_frame_mask"],
            padding_mask=base_inputs["tr_padding_mask"],
        )
    # r23 anchors legacy V27 logits from the archived cache.  Re-executing the
    # old RD/Liquid endpoint is deterministic in the semantic sense but can
    # differ by a few FP32 ULPs on the current runtime, so require a strict
    # numerical replay tolerance rather than falsely demanding bit identity.
    replay_tolerance = 4.0e-6
    replay_deltas: dict[str, float] = {}
    for arm, output in outputs.items():
        delta = float((output["sequence_logits"] - reference).abs().max().item())
        replay_deltas[arm] = delta
        if delta > replay_tolerance:
            raise AssertionError(f"V27c-r23 {arm} initialization does not replay V27a Main: {delta}")
        if bool(output["l2_delta_logits"].ne(0).any()):
            raise AssertionError("V27c-r23 initial L2 delta is not exact zero")
        if bool(output["motion_adapter_output"].ne(0).any()):
            raise AssertionError("V27c-r23 initial motion adapter is not exact zero")
    if not torch.equal(outputs["legacy_control"]["sequence_logits"], outputs["legacy_motion"]["sequence_logits"]):
        raise AssertionError("V27c-r23 matched arms differ before training")
    return {
        "legacy_control_replays_v27a_main_within_fp32_tolerance": True,
        "legacy_motion_replays_v27a_main_within_fp32_tolerance": True,
        "max_abs_replay_delta": replay_deltas,
        "replay_tolerance": replay_tolerance,
        "initial_l2_delta_exact_zero": True,
        "initial_motion_adapter_exact_zero": True,
        "matched_arms_identical_at_initialization": True,
    }


def _assert_motion_location_and_causality(context: R23Context) -> dict[str, Any]:
    batch = _one_batch(context)
    base_inputs = r23_model_input_kwargs(batch, torch.device("cpu"))
    sidecars = r23_sidecar_kwargs(batch, torch.device("cpu"))
    model = build_initialized_model(
        context, arm="legacy_motion", seed=42, device=torch.device("cpu")
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
        raise AssertionError("V27c-r23 smoke batch has no nonzero motion event")
    row, event = (int(nonzero[0, 0].item()), int(nonzero[0, 1].item()))
    changed = original_motion.clone()
    changed[row, event, 0] += 0.25
    altered_sidecars = {**sidecars, "motion_values": changed}
    model.eval()
    reference_inputs = r23_reference_model_input_kwargs(batch, torch.device("cpu"))
    with torch.inference_mode():
        before = model(
            base_inputs=base_inputs,
            reference_base_inputs=reference_inputs,
            **sidecars,
        )
        after = model(
            base_inputs=base_inputs,
            reference_base_inputs=reference_inputs,
            **altered_sidecars,
        )
    if torch.equal(before["sequence_logits"], after["sequence_logits"]):
        raise AssertionError("V27c-r23 motion does not change integrated L2 output")
    if bool(
        (before["sequence_logits"][:, :event] - after["sequence_logits"][:, :event])
        .abs()
        .gt(0.0)
        .any()
    ):
        raise AssertionError("V27c-r23 future motion changed an earlier causal prefix")
    zero_sidecars = {**sidecars, "motion_values": torch.zeros_like(original_motion)}
    with torch.inference_mode():
        zero = model(
            base_inputs=base_inputs,
            reference_base_inputs=reference_inputs,
            **zero_sidecars,
        )
    if bool(zero["motion_adapter_output"].ne(0).any()):
        raise AssertionError("V27c-r23 zero motion is not an adapter no-op after training weights")
    return {
        "motion_injection_location": "post_fusion_pre_liquid_v2",
        "nonzero_motion_changes_integrated_l2_output": True,
        "future_motion_does_not_change_earlier_prefix": True,
        "zero_motion_is_weight_independent_adapter_noop": True,
    }


def _assert_factorial_inputs(context: R23Context) -> dict[str, Any]:
    legacy_dataset = R23L2Dataset(context, split="val", arm="legacy_control")
    robust_dataset = R23L2Dataset(context, split="val", arm="robust_control")
    changed_parent: str | None = None
    changed_event_count = 0
    robust_max_abs = 0.0
    for legacy, robust in zip(legacy_dataset, robust_dataset, strict=True):
        changed = (legacy.sequence.base.tr_features - robust.sequence.base.tr_features).abs()
        changed = changed.amax(dim=-1).gt(0.0)
        robust_max_abs = max(robust_max_abs, float(robust.sequence.base.tr_features.abs().max().item()))
        if changed_parent is None and bool(changed.any()):
            changed_parent = legacy.sequence.base.parent_sample_id
            changed_event_count = int(changed.sum().item())
    if changed_parent is None:
        raise AssertionError("V27c-r23 robust smoke input did not differ from legacy")
    if robust_max_abs > 3.0:
        raise AssertionError("V27c-r23 robust scaled TR input is unexpectedly unbounded")
    for legacy, robust in zip(legacy_dataset, robust_dataset, strict=True):
        if not torch.equal(
            legacy.sequence.base.tr_missing_mask,
            robust.sequence.base.tr_missing_mask,
        ):
            raise AssertionError("V27c-r23 clipping changed missing-mask semantics")
    return {
        "robust_input_differs_on_active_events": True,
        "first_changed_validation_parent": changed_parent,
        "first_changed_validation_event_count": changed_event_count,
        "missing_mask_semantics_preserved": True,
        "robust_scaled_tr_max_abs": robust_max_abs,
    }


def _assert_cached_reference(context: R23Context) -> dict[str, Any]:
    dataset = R23L2Dataset(context, split="val", arm="legacy_control")
    reference_model = build_initialized_model(
        context, arm="legacy_control", seed=42, device=torch.device("cpu")
    )
    reference_model.eval()
    records: list[dict[str, Any]] = []
    loader = build_r23_dataloader(dataset, batch_size=2, shuffle=False)
    with torch.inference_mode():
        for batch in loader:
            sidecars = r23_sidecar_kwargs(batch, torch.device("cpu"))
            base_inputs = r23_model_input_kwargs(batch, torch.device("cpu"))
            logits_batch = reference_model.v27a_main_endpoint(
                cached_v27_base_logits=sidecars["cached_v27_base_logits"],
                stable_summary_tokens=sidecars["stable_summary_tokens"],
                stable_summary_valid=sidecars["stable_summary_valid"],
                stable_frame_values=sidecars["stable_frame_values"],
                stable_frame_mask=sidecars["stable_frame_mask"],
                padding_mask=base_inputs["tr_padding_mask"],
            )
            lengths = batch["lengths"].cpu().tolist()
            labels = batch["labels"].cpu().tolist()
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                length = int(lengths[row])
                logits = logits_batch[row, length - 1]
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "true_id": int(labels[row]),
                        "pred_id": int(logits.argmax().item()),
                    }
                )
    metrics = summarize_records(records, class_names=R23_L2_CLASSES, scope="parent_final")
    if metrics["confusion_matrix_rows_true_columns_pred"] != [
        list(row) for row in R23_V27A_MAIN_CONFUSION
    ]:
        raise AssertionError("V27c-r23 cached V27a Main endpoint differs")
    return {
        "v27a_main_cached_reference_confusion": metrics[
            "confusion_matrix_rows_true_columns_pred"
        ],
        "matches_archived_v27a_main": True,
    }


def run_smoke(context: R23Context) -> dict[str, Any]:
    result = {
        "schema_version": R23_SMOKE_SCHEMA_VERSION,
        "experiment_id": context.config.experiment_id,
        "status": "passed",
        "manifest_sha256": context.manifest_sha256,
        "sidecar_boundary": _assert_sidecar_boundary(context),
        "initial_replay": _assert_initial_replay(context),
        "motion_boundary": _assert_motion_location_and_causality(context),
        "factorial_inputs": _assert_factorial_inputs(context),
        "clipping": {
            "fit_split": context.config.data.tr_clipping_fit_split,
            "quantile": context.config.data.tr_clipping_quantile,
            "audits": context.input_binding["tr_clipping_audits"],
            "robust_arm_differs_from_legacy": True,
        },
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
    path = r23_artifact_path(context.project_root, context.config.artifacts.smoke_report_path)
    write_sealed_json(path, result)
    return {**result, "report_path": str(path)}


__all__ = ["run_smoke"]
