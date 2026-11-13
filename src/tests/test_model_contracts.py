"""正式 L1/L2 模型的最小 contract smoke。"""

from __future__ import annotations

from pathlib import Path
import hashlib

import torch
import pytest
import radar_ln3.checkpoint as checkpoint_module

from radar_ln3.checkpoint import Checkpoint, load_model_state
from radar_ln3.device import resolve_device
from radar_ln3.features import build_model_batch
from radar_ln3.model import build_l1_model, build_l2_model, load_model_config
from radar_ln3.model.io import (
    BASE_SEQUENCE_INPUTS,
    FULL_CASCADE_INPUTS,
    L1_MODEL_INPUTS,
    L1_SOURCE_ONLY_INPUTS,
    L1_TASK_INPUTS,
    L2_MODEL_INPUTS,
    L2_TASK_INPUTS,
    LEGACY_LV28_BATCH_ALIASES,
    LV28_INPUT_NAMES,
    RESERVED_EXPORT_INPUTS,
    select_inputs,
)
from radar_ln3.data.contracts import EventData, EventSequence
from radar_ln3.training.common import (
    ModelBatchCollator,
    build_training_loader,
    model_batch_kwargs,
)


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


def test_checkpoint_round_trip_in_experiment_directory(tmp_path: Path) -> None:
    checkpoint, status = Checkpoint.save(
        tmp_path,
        "experiments/tmp/L2/l2-knownness.pt",
        {"schema_version": "fixture", "model_state": {"weight": torch.ones(1)}},
    )

    assert status == "created"
    loaded = Checkpoint.load(tmp_path, "experiments/tmp/L2/l2-knownness.pt")
    assert torch.equal(loaded.state["weight"], torch.ones(1))
    provenance = checkpoint.provenance(kind="test")
    assert provenance["path"] == "experiments/tmp/L2/l2-knownness.pt"
    assert provenance["schema_version"] == "fixture"


def test_checkpoint_load_hashes_state_once_per_call(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "checkpoint.pt"
    state = {"weight": torch.arange(8, dtype=torch.float32)}
    expected_state_hash = checkpoint_module.state_hash(state)
    torch.save(
        {
            "task": "fixture",
            "model_state": state,
            "model_state_sha256": expected_state_hash,
        },
        path,
    )

    calls = 0
    original_state_hash = checkpoint_module.state_hash

    def count_state_hash(value):
        nonlocal calls
        calls += 1
        return original_state_hash(value)

    monkeypatch.setattr(checkpoint_module, "state_hash", count_state_hash)
    loaded = Checkpoint.load(tmp_path, path)
    assert torch.equal(loaded.state["weight"], state["weight"])
    assert loaded.verify_state_hash() == expected_state_hash
    assert calls == 1
    assert loaded.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_checkpoint_load_defers_file_hash_until_provenance(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "checkpoint.pt"
    state = {"weight": torch.arange(4, dtype=torch.float32)}
    torch.save({"task": "fixture", "model_state": state}, path)
    calls = 0
    original = checkpoint_module.sha256_file

    def count_file_hash(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(checkpoint_module, "sha256_file", count_file_hash)
    loaded = Checkpoint.load(tmp_path, path)
    assert calls == 0
    assert loaded.file_sha256 == original(path)
    assert calls == 1
    assert loaded.file_sha256 == loaded.file_sha256
    assert calls == 1


def test_checkpoint_file_hash_is_excluded_from_dataclass_equality(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.pt"
    torch.save({"task": "fixture", "model_state": {"weight": torch.ones(1)}}, path)
    first = Checkpoint.load(tmp_path, path)
    second = Checkpoint.load(tmp_path, path)
    assert first == second
    _ = first.file_sha256
    assert first == second


def test_checkpoint_save_does_not_reload_created_file(tmp_path: Path, monkeypatch) -> None:
    calls = 0
    original_load = checkpoint_module._load_file

    def count_load(path):
        nonlocal calls
        calls += 1
        return original_load(path)

    monkeypatch.setattr(checkpoint_module, "_load_file", count_load)
    checkpoint, status = Checkpoint.save(
        tmp_path,
        "checkpoint.pt",
        {"schema_version": "fixture", "model_state": {"weight": torch.ones(1)}},
    )
    assert status == "created"
    assert calls == 0
    assert checkpoint.verify_state_hash() == checkpoint.payload["model_state_sha256"]


def test_artifact_root_redirects_l1_and_l2_endpoint_outputs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    l1_config = load_model_config(root, artifact_root="experiments/tmp/L1")
    l2_config = load_model_config(root, artifact_root="experiments/tmp/L2")

    assert l1_config.artifacts.l1_output_path == "experiments/tmp/L1/l1/l1.pt"
    assert l2_config.artifacts.l2_output_path == "experiments/tmp/L2/l2/l2.pt"
    assert l2_config.artifacts.l2_direct_output_path == "experiments/tmp/L2/l2-direct/l2-direct.pt"
    assert l2_config.artifacts.l2_direct_grouped_output_path == (
        "experiments/tmp/L2/l2-direct-grouped/l2-direct-grouped.pt"
    )


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
    inputs = {key: l2_batch[key] for key in L2_MODEL_INPUTS}
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

    reference_output = l2(
        base_inputs=inputs,
        reference_base_inputs=inputs,
        baseline_logits=anchor.detach(),
        stable_summary_tokens=l2_batch["stable_summary_tokens"],
        stable_summary_valid=l2_batch["stable_summary_valid"],
        stable_frame_values=l2_batch["stable_frame_values"],
        stable_frame_mask=l2_batch["stable_frame_mask"],
        motion_values=l2_batch["motion_values"],
    )
    optimized_output = l2(
        base_inputs=inputs,
        reference_base_inputs=None,
        baseline_logits=anchor.detach(),
        stable_summary_tokens=l2_batch["stable_summary_tokens"],
        stable_summary_valid=l2_batch["stable_summary_valid"],
        stable_frame_values=l2_batch["stable_frame_values"],
        stable_frame_mask=l2_batch["stable_frame_mask"],
        motion_values=l2_batch["motion_values"],
        frozen_reference_logits_override=anchor.detach(),
    )
    assert torch.equal(
        optimized_output["sequence_logits"], reference_output["sequence_logits"]
    )


def test_model_batch_collator_matches_single_process_with_workers() -> None:
    sequences = [
        _sequence("workers|track|1", 0, "DroneTarget", 2),
        _sequence("workers|track|2", 1, "BirdTarget", 3),
    ]
    collator = ModelBatchCollator(
        task="l1",
        max_rd_frames=12,
        max_sequence_length=29,
        stable_dim=8,
        stable_indices=(1, 2, 3, 7, 11, 13, 14, 22),
    )
    expected = collator(sequences)
    loader = build_training_loader(
        sequences,
        collate_fn=collator,
        batch_size=2,
        shuffle=False,
        seed=42,
        num_workers=2,
        prefetch_factor=2,
        pin_memory=True,
    )
    actual = next(iter(loader))
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key


def test_model_config_uses_one_stable8_width() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_model_config(root)

    assert config.model.evidence_dim == 8
    assert config.data.stable8_dim == config.model.evidence_dim
    assert config.data.l2.micro_doppler_dim == config.data.stable8_dim
    assert len(config.data.l2.stable8_indices) == config.data.stable8_dim


def test_model_batch_kwargs_and_collator_share_config() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_model_config(root)

    l1_kwargs = model_batch_kwargs(config, "l1")
    assert l1_kwargs["stable_dim"] == config.data.stable8_dim
    assert l1_kwargs["stable_indices"] == config.data.l2.stable8_indices
    assert "clipping" not in l1_kwargs
    assert "stable_input_mode" not in l1_kwargs

    l2_kwargs = model_batch_kwargs(
        config,
        "l2",
        stable_input_mode="raw",
        allow_unmapped_labels=True,
    )
    range_bounds = config.data.l2.tr_clipping_bounds["range_m"]
    assert l2_kwargs["clipping"]["bounds"]["range_m"] == {
        "lower": range_bounds[0],
        "upper": range_bounds[1],
    }
    assert l2_kwargs["stable_median"].shape == (config.data.stable8_dim,)
    assert l2_kwargs["stable_scale"].shape == (config.data.stable8_dim,)
    assert l2_kwargs["stable_input_mode"] == "raw"
    assert l2_kwargs["allow_unmapped_labels"] is True

    l1_collator = ModelBatchCollator.from_config(config, "l1")
    assert l1_collator.stable_dim == config.data.stable8_dim
    assert l1_collator.stable_indices == config.data.l2.stable8_indices
    l2_collator = ModelBatchCollator.from_config(config, "l2")
    assert l2_collator.stable_median is not None
    assert l2_collator.stable_scale is not None
    assert l2_collator.clipping == l2_kwargs["clipping"]


def test_l2_collator_keeps_stable_sidecar_keys() -> None:
    from radar_ln3.training.common import STABLE_SIDECAR_KEYS

    root = Path(__file__).resolve().parents[2]
    config = load_model_config(root)
    collator = ModelBatchCollator.from_config(config, "l2")

    batch = collator([_sequence("sidecar|track|1", 0, "DroneTarget", 2)])

    assert set(STABLE_SIDECAR_KEYS) <= set(batch)
    assert batch["stable_summary_tokens"].shape[-1] == config.data.stable8_dim


def test_knownness_labels_are_derived_from_l2_vocabulary() -> None:
    from radar_ln3.labels import L2_LABELS, TASK_LABEL_TO_ID, TASK_LABELS
    from radar_ln3.training.l3_heads import HEAD_LABELS, L2_ROUTES

    assert TASK_LABELS["l2"] == L2_LABELS
    assert L2_ROUTES == L2_LABELS
    assert HEAD_LABELS == (*L2_LABELS, "ClutterTarget", "UnknownTarget")
    assert TASK_LABEL_TO_ID["l2"] == {
        label: index for index, label in enumerate(L2_LABELS)
    }


def test_export_input_contract_keeps_frozen_cascade_order() -> None:
    assert FULL_CASCADE_INPUTS[:9] == BASE_SEQUENCE_INPUTS
    assert FULL_CASCADE_INPUTS == (
        BASE_SEQUENCE_INPUTS
        + L2_TASK_INPUTS
        + L1_TASK_INPUTS
        + RESERVED_EXPORT_INPUTS
    )
    assert len(FULL_CASCADE_INPUTS) == 24
    assert len(set(FULL_CASCADE_INPUTS)) == 24
    assert len(RESERVED_EXPORT_INPUTS) == 8


def test_model_input_sets_are_derived_from_the_same_groups() -> None:
    assert len(L1_MODEL_INPUTS) == 12
    assert len(L2_MODEL_INPUTS) == 13
    assert set(L1_MODEL_INPUTS) == set(BASE_SEQUENCE_INPUTS) | set(L1_TASK_INPUTS)
    assert set(L2_MODEL_INPUTS) == set(BASE_SEQUENCE_INPUTS) | set(L2_TASK_INPUTS)
    assert L1_SOURCE_ONLY_INPUTS == L1_TASK_INPUTS + (
        "observation_increment_values",
    )


def test_lv28_contract_and_aliases_cover_one_legacy_graph() -> None:
    assert len(LV28_INPUT_NAMES) == 19
    assert len(set(LV28_INPUT_NAMES)) == 19
    assert set(LEGACY_LV28_BATCH_ALIASES) == set(LV28_INPUT_NAMES)
    renamed = {
        name: source
        for name, source in LEGACY_LV28_BATCH_ALIASES.items()
        if name != source
    }
    assert renamed == {"observation_increment": "observation_increment_values"}
    assert select_inputs({"a": 1, "b": 2, "c": 3}, ("b", "a")) == {"b": 2, "a": 1}


def test_select_inputs_reports_every_missing_key() -> None:
    with pytest.raises(ValueError) as error:
        select_inputs({"tr_features": None}, ("tr_features", "rd_frames", "rd_quality"))

    message = str(error.value)
    assert "rd_frames" in message
    assert "rd_quality" in message


def test_resolve_device_normalises_cpu_and_passthrough() -> None:
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(" CPU ") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_resolve_device_accepts_explicit_cuda_index(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("cuda") == torch.device("cuda:0")
    assert resolve_device("CUDA:1") == torch.device("cuda:1")
    assert resolve_device(torch.device("cuda:2")) == torch.device("cuda:2")


def test_resolve_device_rejects_unavailable_or_unknown_targets(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="配置要求 CUDA，但当前环境不可用"):
        resolve_device("cuda:0")
    with pytest.raises(ValueError, match="不支持的 device：gpu"):
        resolve_device("gpu")
    with pytest.raises(ValueError, match="不支持的 device：cuda:x"):
        resolve_device("cuda:x")
    with pytest.raises(ValueError, match="device 不能为空"):
        resolve_device("   ")
