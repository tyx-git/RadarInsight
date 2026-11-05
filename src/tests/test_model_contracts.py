"""正式 L1/L2 模型的最小 contract smoke。"""

from __future__ import annotations

from pathlib import Path

import torch

from radar_ln3.checkpoint import Checkpoint, load_model_state
from radar_ln3.features import build_model_batch
from radar_ln3.model import build_l1_model, build_l2_model, load_model_config
from radar_ln3.data.contracts import EventData, EventSequence


def test_load_model_state_drops_only_retired_quality_head_keys() -> None:
    model = torch.nn.Linear(2, 1)
    state = dict(model.state_dict())
    state["trajectory_heads.quality_head.0.weight"] = torch.zeros(4, 2)
    state["trajectory_heads.quality_head.0.bias"] = torch.zeros(4)

    load_model_state(model, state)

    state["unrelated.retired.weight"] = torch.zeros(1)
    try:
        load_model_state(model, state)
    except RuntimeError as exc:
        assert "Unexpected key(s)" in str(exc)
    else:
        raise AssertionError("unrelated checkpoint keys must remain strict")


def _sequence(parent_id: str, label_id: int, label_name: str, length: int) -> EventSequence:
    events = []
    for index in range(length):
        events.append(
            EventData(
                sample_id=f"{parent_id}:event-{index}",
                parent_sample_id=parent_id,
                track_id=parent_id.rsplit("|", 1)[-1],
                causal_event_index=index,
                label_id=label_id,
                label_name=label_name,
                split_name="train",
                tr_values={
                    "range_m": 100.0 + index,
                    "radial_speed_mps": 2.0,
                    "elevation_deg": 1.0,
                    "azimuth_deg": 2.0,
                    "course_deg": 3.0 + index,
                    "speed_mps": 4.0 + index,
                    "amplitude_db": 10.0,
                    "snr_db": 8.0,
                    "height_m": 20.0,
                    "latitude": 30.0,
                    "longitude": 120.0,
                    "time_seconds": float(index),
                },
                rd_values=None,
                tr_available=True,
                rd_available=False,
                tr_time_seconds=float(index),
                tr_dt_seconds=0.0 if index == 0 else 1.0,
                tr_dt_valid=index > 0,
                tr_dt_reason=(
                    "first_event_no_predecessor"
                    if index == 0
                    else "positive_delta_at_second_precision"
                ),
            )
        )
    return EventSequence(
        parent_sample_id=parent_id,
        track_id=parent_id.rsplit("|", 1)[-1],
        label_id=label_id,
        label_name=label_name,
        split_name="train",
        events=events,
        source_name="smoke",
    )


def test_formal_models_forward_backward() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_model_config(root)
    l1_state = Checkpoint.load(root, config.artifacts.l1_base_weight_path).state
    l2_state = Checkpoint.load(root, config.artifacts.l2_base_weight_path).state
    stable_state = Checkpoint.load(root, config.artifacts.l2_stable_weight_path).state

    l1 = build_l1_model(config, base_state=l1_state)
    l1_batch = build_model_batch(
        [_sequence("smoke|track|1", 0, "DroneTarget", 2)],
        task="l1",
        max_rd_frames=config.data.max_rd_frames_per_point,
    )
    l1_output = l1(
        base_inputs={
            key: value
            for key, value in l1_batch.items()
            if key in {
                "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
                "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
                "rd_quality", "course_features", "course_valid", "course_support",
            }
        },
        observation_increment_values=l1_batch["observation_increment_values"],
    )
    l1_output["final_logits"].sum().backward()
    assert all(
        parameter.grad is None
        for parameter in l1.base.trajectory_heads.parameters()
    )

    l2 = build_l2_model(
        config,
        arm="robust_control",
        base_state=l2_state,
        stable_state=stable_state,
        seed=42,
    )
    l2_batch = build_model_batch(
        [_sequence("smoke|track|2", 0, "DroneTarget", 2)],
        task="l2",
        max_rd_frames=config.data.max_rd_frames_per_point,
    )
    inputs = {
        key: l2_batch[key]
        for key in {
            "tr_features", "tr_missing_mask", "tr_dt_seconds", "tr_dt_valid",
            "tr_dt_reason", "tr_padding_mask", "rd_frames", "rd_frame_mask",
            "rd_quality", "rd_physical_features", "rd_physical_valid",
            "micro_doppler_features", "micro_doppler_valid",
        }
    }
    anchor = l2.frozen_reference_logits(base_inputs=inputs)
    l2_output = l2(
        base_inputs=inputs,
        reference_base_inputs=inputs,
        baseline_logits=anchor.detach(),
        stable_summary_tokens=l2_batch["stable_summary_tokens"],
        stable_summary_valid=l2_batch["stable_summary_valid"],
        stable_frame_values=l2_batch["stable_frame_values"],
        stable_frame_mask=l2_batch["stable_frame_mask"],
        motion_values=l2_batch["motion_values"],
    )
    assert torch.equal(l2_output["baseline_logits"], anchor.detach())
    assert torch.equal(
        l2_output["frozen_reference_l2_logits"],
        l2_output["current_l2_logits"].detach(),
    )
    l2_output["sequence_logits"].sum().backward()
    assert all(
        parameter.grad is None
        for parameter in l2.frozen_reference_base_model.parameters()
    )
    assert all(
        parameter.grad is None
        for parameter in l2.trainable_base_model.trajectory_heads.parameters()
    )
