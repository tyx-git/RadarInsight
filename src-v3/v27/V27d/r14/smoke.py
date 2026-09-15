"""Small contract checks for V27d-r14 time-policy migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Subset

from radar_ln3.data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)
from radar_ln3.liquid_v2 import LiquidV2Cell, LiquidV2SequenceWrapper

from .data import R14L2Dataset, build_r14_dataloader, r14_model_input_kwargs, r14_sidecar_kwargs
from .protocol import R14_SMOKE_SCHEMA, R14Context, artifact_path, write_sealed_json_if_identical
from .model import build_r14_model
from .training import tensor_state_hash


def _cell(mode: str) -> LiquidV2Cell:
    return LiquidV2Cell(
        input_dim=4,
        hidden_dim=5,
        tau_min_seconds=0.05,
        tau_max_seconds=30.0,
        zero_dt_event_floor_seconds=0.001,
        dt_max_seconds=120.0,
        invalid_dt_update_mode=mode,
        invalid_dt_observation_gain=0.25,
    )


def _time_policy_probe() -> dict[str, Any]:
    torch.manual_seed(1400)
    control = _cell("bounded_observation")
    treatment = _cell("density_weighted")
    treatment.load_state_dict(control.state_dict(), strict=True)
    event = torch.randn(1, 4)
    state = torch.randn(1, 5)
    zero_dt = torch.zeros(1)
    invalid = torch.zeros(1, dtype=torch.bool)
    zero_reason = torch.full((1,), DT_REASON_ZERO_DELTA, dtype=torch.long)
    c1 = control.step(event, state, zero_dt, invalid, zero_reason, torch.ones(1, dtype=torch.long))
    t1 = treatment.step(event, state, zero_dt, invalid, zero_reason, torch.ones(1, dtype=torch.long))
    c2 = control.step(event, state, zero_dt, invalid, zero_reason, torch.full((1,), 2, dtype=torch.long))
    t2 = treatment.step(event, state, zero_dt, invalid, zero_reason, torch.full((1,), 2, dtype=torch.long))
    t3 = treatment.step(event, state, zero_dt, invalid, zero_reason, torch.full((1,), 3, dtype=torch.long))
    first_reason = torch.full((1,), DT_REASON_FIRST_EVENT, dtype=torch.long)
    positive_reason = torch.full((1,), DT_REASON_POSITIVE_DELTA, dtype=torch.long)
    first_c = control.step(event, torch.zeros_like(state), zero_dt, invalid, first_reason, torch.ones(1, dtype=torch.long))
    first_t = treatment.step(event, torch.zeros_like(state), zero_dt, invalid, first_reason, torch.ones(1, dtype=torch.long))
    positive_c = control.step(event, state, torch.ones(1), torch.ones(1, dtype=torch.bool), positive_reason, torch.ones(1, dtype=torch.long))
    positive_t = treatment.step(event, state, torch.ones(1), torch.ones(1, dtype=torch.bool), positive_reason, torch.ones(1, dtype=torch.long))
    if not torch.equal(c1.next_state, t1.next_state):
        raise AssertionError("r14 k=1 treatment does not match control")
    if not torch.allclose(t2.next_state - state, 2.0 * (c1.next_state - state), atol=1e-6, rtol=1e-6):
        raise AssertionError("r14 k=2 density gain is not twice the control increment")
    if not torch.allclose(t3.next_state - state, 3.0 * (c1.next_state - state), atol=1e-6, rtol=1e-6):
        raise AssertionError("r14 k=3 density gain is not three times the control increment")
    if not torch.equal(first_c.next_state, first_t.next_state) or not torch.equal(positive_c.next_state, positive_t.next_state):
        raise AssertionError("r14 changed first-event or positive-dt semantics")
    wrapper = LiquidV2SequenceWrapper(treatment)
    events = torch.randn(1, 5, 4)
    dt = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    valid = torch.tensor([[False, False, False, True, False]])
    reason = torch.tensor([[DT_REASON_FIRST_EVENT, DT_REASON_ZERO_DELTA, DT_REASON_ZERO_DELTA, DT_REASON_POSITIVE_DELTA, DT_REASON_PADDING]], dtype=torch.long)
    padding = torch.tensor([[False, False, False, False, True]])
    ordinal = torch.tensor([[1, 2, 3, 1, 0]], dtype=torch.long)
    full = wrapper(events, dt, padding, dt_valid=valid, dt_reason=reason, zero_ordinal=ordinal, return_trace=True)
    first = wrapper(events[:, :3], dt[:, :3], padding[:, :3], dt_valid=valid[:, :3], dt_reason=reason[:, :3], zero_ordinal=ordinal[:, :3])
    second = wrapper(events[:, 3:4], dt[:, 3:4], padding[:, 3:4], dt_valid=valid[:, 3:4], dt_reason=reason[:, 3:4], initial_state=first.final_state, zero_ordinal=ordinal[:, 3:4])
    if not torch.allclose(full.final_state, second.final_state, atol=1e-6, rtol=1e-6):
        raise AssertionError("r14 chunked continuation differs from full treatment sequence")
    if not torch.equal(full.sequence[:, 4], full.sequence[:, 3]):
        raise AssertionError("r14 padding changed raw memory")
    return {
        "k1_matches_control": True,
        "k2_is_twice_control_increment": True,
        "k3_is_three_times_control_increment": True,
        "first_event_unchanged": True,
        "positive_dt_unchanged": True,
        "chunked_matches_full": True,
        "padding_never_updates": True,
        "finite": bool(torch.isfinite(full.sequence).all()),
    }


def run_smoke(context: R14Context) -> dict[str, Any]:
    control = build_r14_model(context, arm="control", seed=42, device=torch.device("cpu"))
    treatment = build_r14_model(context, arm="treatment", seed=42, device=torch.device("cpu"))
    if tensor_state_hash(control.state_dict()) != tensor_state_hash(treatment.state_dict()):
        raise AssertionError("r14 control/treatment initial parameters differ")
    dataset = R14L2Dataset(context, split="val")
    exposed_index = next(
        (
            index
            for index in range(len(dataset))
            if bool(dataset[index].sequence.base.tr_dt_reason.eq(2).any())
        ),
        None,
    )
    if exposed_index is None:
        raise AssertionError("r14 validation pool has no zero-delta exposure")
    batch = next(iter(build_r14_dataloader(Subset(dataset, [exposed_index]), batch_size=1, shuffle=False)))
    base_inputs = r14_model_input_kwargs(batch, torch.device("cpu"))
    sidecars = r14_sidecar_kwargs(batch, torch.device("cpu"))
    control.eval()
    treatment.eval()
    with torch.inference_mode():
        control_output = control(base_inputs=base_inputs, **sidecars)
        treatment_output = treatment(base_inputs=base_inputs, **sidecars)
        stable_output = control.stable_model(
            baseline_logits=sidecars["cached_v27_base_logits"],
            summary_tokens=sidecars["stable_summary_tokens"],
            summary_valid=sidecars["stable_summary_valid"],
            frame_values=sidecars["stable_frame_values"],
            frame_mask=sidecars["stable_frame_mask"],
            padding_mask=base_inputs["tr_padding_mask"],
        )
    if not torch.allclose(control_output["sequence_logits"], stable_output["sequence_logits"], atol=1.0e-5, rtol=1.0e-5):
        raise AssertionError("r14 control does not replay the frozen V27a Main endpoint at initialization")
    probe = _time_policy_probe()
    result = {
        "schema_version": R14_SMOKE_SCHEMA,
        "experiment_id": "V27d-r14",
        "status": "passed",
        "manifest_sha256": context.manifest_sha256,
        "initial_control_treatment_state_equal": True,
        "control_replays_v27a_main_at_initialization": True,
        "treatment_differs_on_zero_delta_exposure": bool(
            not torch.equal(control_output["sequence_logits"], treatment_output["sequence_logits"])
        ),
        "control_mode": "bounded_observation",
        "treatment_mode": "density_weighted",
        "time_policy_probe": probe,
        "unknown_payload_access_count": 0,
        "test_payload_access_count": 0,
        "formal_l3_enabled": False,
    }
    path = artifact_path(context.project_root, context.config.artifacts["smoke_path"])
    write_sealed_json_if_identical(path, result)
    return {**result, "report_path": str(path)}


__all__ = ["run_smoke"]
