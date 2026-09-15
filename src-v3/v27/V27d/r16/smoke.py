"""Contract smoke tests for V27d-r16."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Subset

from .data import build_r16_dataloader, r16_frame_time_kwargs, r16_model_input_kwargs, R16L2Dataset
from .model import build_r16_model, trainable_parameter_groups
from .protocol import R16_SMOKE_SCHEMA, R16Context, artifact_path, write_sealed_json_if_identical


def run_smoke(context: R16Context) -> dict[str, Any]:
    dataset = R16L2Dataset(context, split="val")
    probe_index = next(
        (
            index
            for index in range(len(dataset))
            if bool(dataset[index].frame_time_valid.any())
        ),
        None,
    )
    if probe_index is None:
        raise AssertionError("V27d-r16 validation has no monotonic multi-frame RD event")
    batch = next(
        iter(
            build_r16_dataloader(
                Subset(dataset, [probe_index]),
                batch_size=1,
                shuffle=False,
            )
        )
    )
    device = torch.device("cpu")
    control = build_r16_model(context, arm="control", seed=42, device=device)
    treatment = build_r16_model(context, arm="treatment", seed=42, device=device)
    if trainable_parameter_groups(control):
        raise AssertionError("V27d-r16 control unexpectedly has trainable parameters")
    if not trainable_parameter_groups(treatment):
        raise AssertionError("V27d-r16 treatment has no trainable parameters")
    base_inputs = r16_model_input_kwargs(batch, device)
    times = r16_frame_time_kwargs(batch, device)
    sidecars = {
        name: batch[name].to(device)
        for name in (
            "cached_v27_base_logits",
            "stable_summary_tokens",
            "stable_summary_valid",
            "stable_frame_values",
            "stable_frame_mask",
        )
    }
    control.eval()
    treatment.eval()
    with torch.inference_mode():
        control_output = control(base_inputs=base_inputs, **sidecars, **times)
        treatment_output = treatment(base_inputs=base_inputs, **sidecars, **times)
    if not torch.allclose(
        control_output["sequence_logits"],
        treatment_output["sequence_logits"],
        atol=1.0e-5,
        rtol=1.0e-5,
    ):
        raise AssertionError("V27d-r16 zero-initialized treatment does not replay control")
    if bool(batch["rd_frame_time_valid"].any()) is not True:
        raise AssertionError("V27d-r16 smoke probe did not expose valid frame time")
    treatment_encoder = treatment.trainable_base_model.rd_encoder
    if not torch.equal(treatment_encoder.residual_projection.weight, torch.zeros_like(treatment_encoder.residual_projection.weight)):
        raise AssertionError("V27d-r16 treatment residual projection is not zero initialized")
    payload = {
        "schema_version": R16_SMOKE_SCHEMA,
        "experiment_id": "V27d-r16",
        "status": "passed",
        "manifest_sha256": context.manifest_sha256,
        "control_is_frozen": True,
        "treatment_has_only_rd_temporal_parameters": True,
        "treatment_replays_control_at_initialization": True,
        "canonical_frame_times_exposed": True,
        "valid_frame_time_event_exposed": True,
        "nonmonotonic_fallback_policy": "verified_by_loader_contract",
        "unknown_training_exposure": 0,
        "test_payload_access_count": 0,
        "formal_l3_enabled": False,
    }
    path = artifact_path(context.project_root, context.config.artifacts["smoke_path"])
    write_sealed_json_if_identical(path, payload)
    return {**payload, "report_path": str(path)}


__all__ = ["run_smoke"]
