"""产品 features 模块的 focused contract tests。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from radar_ln3.data.contracts import (
    DatasetView,
    EventData,
    EventSequence,
)
from radar_ln3.features import (
    BROADBAND_FRAME_VALUE_DIM,
    COURSE_FEATURE_NAME,
    FEATURE_CACHE_SCHEMA_VERSION,
    FeatureCache,
    FeatureCacheDescriptor,
    RawEchoFrame,
    RawTargetRecord,
    STABLE8_INDICES,
    TR_MISSING_DIM,
    TR_VALUE_DIM,
    aggregate_stable8_frame_values,
    build_feature_cache_from_records,
    build_feature_cache_bundle,
    build_model_batch,
    encode_causal_course_sequence,
    encode_causal_rd_quality,
    encode_causal_tr_motion_sequence,
    encode_tr_values,
    extract_current_event_broadband_micro_doppler,
    extract_current_event_rd_physical,
    feature_catalog,
    fit_parent_equal_median_iqr,
    load_feature_cache,
    load_feature_cache_bundle,
    normalize_stable8,
    prepare_rd_event,
    save_feature_cache,
    save_feature_cache_bundle,
)


def _tr_event(
    *,
    time_seconds: float,
    speed_mps: float = 10.0,
    radial_speed_mps: float = 5.0,
    course_deg: float | None = 0.0,
    range_m: float = 100.0,
) -> dict[str, float | None]:
    return {
        "time_seconds": time_seconds,
        "speed_mps": speed_mps,
        "radial_speed_mps": radial_speed_mps,
        "course_deg": course_deg,
        "range_m": range_m,
        "azimuth_deg": 0.0,
        "elevation_deg": 0.0,
        "amplitude_db": 10.0,
        "snr_db": 20.0,
    }


def test_tr_values_keep_missing_facts_separate_from_zero_values() -> None:
    values, missing = encode_tr_values(
        {
            **_tr_event(time_seconds=0.0, course_deg=90.0),
            "height_m": np.nan,
            "latitude": None,
            "longitude": -180.0,
        }
    )

    assert values.shape == (TR_VALUE_DIM,)
    assert missing.shape == (TR_MISSING_DIM,)
    assert values.dtype == np.float32
    assert missing.dtype == np.float32
    assert values[6] == pytest.approx(1.0)
    assert values[7] == pytest.approx(0.0, abs=1.0e-6)
    assert missing[8] == 1.0
    assert missing[9] == 1.0
    assert missing[10] == 0.0
    assert np.isfinite(values).all()


def test_model_batch_truncates_long_parent_to_causal_prefix() -> None:
    events = []
    for index in range(31):
        events.append(
            EventData(
                sample_id=f"long:event-{index}",
                parent_sample_id="long",
                track_id="long",
                causal_event_index=index,
                label_id=0,
                label_name="DroneTarget",
                split_name="train",
                tr_values=_tr_event(time_seconds=float(index), range_m=100.0 + index),
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
    sequence = EventSequence(
        parent_sample_id="long",
        track_id="long",
        label_id=0,
        label_name="DroneTarget",
        split_name="train",
        events=events,
        source_name="fixture",
    )

    batch = build_model_batch(
        [sequence],
        task="l1",
        max_sequence_length=29,
    )

    assert batch["original_lengths"].tolist() == [31]
    assert batch["sequence_truncated"].tolist() == [True]
    assert batch["lengths"].tolist() == [29]
    assert batch["tr_padding_mask"].shape == (1, 29)
    assert batch["tr_features"][0, -1, 0] == pytest.approx(128.0 / 5000.0)


def test_motion_does_not_fabricate_same_second_derivatives() -> None:
    events = [
        _tr_event(time_seconds=0.0, speed_mps=10.0),
        _tr_event(time_seconds=0.0, speed_mps=30.0, range_m=110.0),
        _tr_event(time_seconds=1.0, speed_mps=40.0, range_m=120.0),
    ]
    values, valid = encode_causal_tr_motion_sequence(
        events,
        dt_seconds=[0.0, 0.0, 1.0],
        dt_valid=[False, False, True],
    )

    assert values.shape == (3, 9)
    assert valid.shape == values.shape
    assert not valid[0].any()
    assert not valid[1].any()
    assert valid[2, 0]
    assert np.isfinite(values).all()


def test_course_circular_incoherence_is_causal_and_keeps_support() -> None:
    events = [
        _tr_event(time_seconds=0.0, course_deg=0.0),
        _tr_event(time_seconds=1.0, course_deg=0.0),
        _tr_event(time_seconds=2.0, course_deg=90.0),
        _tr_event(time_seconds=3.0, course_deg=180.0),
        _tr_event(time_seconds=4.0, course_deg=None),
    ]
    values, valid, support = encode_causal_course_sequence(events)

    assert values.shape == (5, 1)
    assert valid[:, 0].tolist() == [False, False, False, True, True]
    assert support.tolist() == [0, 1, 2, 3, 3]
    assert values[4, 0] == pytest.approx(values[3, 0])
    course_entry = feature_catalog()["features"]["course_circular_incoherence"]
    assert course_entry["schema"] == "causal-prefix-course-circular-incoherence-v1"
    assert course_entry["feature_name"] == COURSE_FEATURE_NAME


def test_rd_quality_marks_missing_and_bounds_large_frame_counts() -> None:
    missing = encode_causal_rd_quality(
        present=False,
        original_count=0,
        duration_seconds=0.0,
        local_time_non_monotonic=False,
        truncated=False,
        max_rd_frames_per_point=4,
    )
    large = encode_causal_rd_quality(
        present=True,
        original_count=100,
        duration_seconds=1.0,
        local_time_non_monotonic=True,
        truncated=True,
        max_rd_frames_per_event=4,
    )

    assert missing.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    assert np.all((large >= 0.0) & (large <= 1.0))


def test_physical4_uses_explicit_velocity_axis_and_invalid_mask() -> None:
    axis = np.linspace(-85.0, 85.0, 384, dtype=np.float64)
    amplitude = np.zeros((2, 3, 384), dtype=np.float32)
    amplitude[:, :, 190] = 0.0
    amplitude[:, :, 200] = 10.0
    invalid = extract_current_event_rd_physical(
        amplitude,
        axis,
        frame_mask=np.asarray([False, False]),
    )
    valid = extract_current_event_rd_physical(amplitude, axis)

    assert not invalid.valid
    assert np.array_equal(invalid.values, np.zeros((4,), dtype=np.float32))
    assert valid.valid
    assert valid.values.shape == (4,)
    assert np.isfinite(valid.values).all()
    assert np.all((valid.values >= 0.0) & (valid.values <= 1.0))


def test_stable8_uses_reliable_frame_median_only() -> None:
    values = np.zeros((3, BROADBAND_FRAME_VALUE_DIM), dtype=np.float32)
    values[0, list(STABLE8_INDICES)] = 0.2
    values[1, list(STABLE8_INDICES)] = 0.4
    values[2, list(STABLE8_INDICES)] = 0.9
    reliable = np.asarray([True, True, False], dtype=np.bool_)

    result = aggregate_stable8_frame_values(values, reliable)

    assert result.shape == (8,)
    assert np.all(result == 0.3)


def test_broadband_raw_iq_extractor_is_dataset_agnostic() -> None:
    iq = np.zeros((11, 64), dtype=np.complex64)
    iq[5] = np.exp(2j * np.pi * np.arange(64, dtype=np.float64) / 64.0)
    frame = RawEchoFrame(
        raw_frame_index=0,
        frequency_hz=15.6e9,
        cpi_count=64,
        prt_num=64,
        prt_seconds=20.0e-6,
        target_records=(
            RawTargetRecord(
                target_identifier=1,
                event_index=0,
                range_bin=5,
                doppler_bin=32,
            ),
        ),
        iq=iq,
    )

    event = extract_current_event_broadband_micro_doppler(
        [frame],
        [0],
        event_index=0,
    )

    assert event.values.shape == (8,)
    assert event.frame_mask.tolist() == [True] + [False] * 11
    assert event.frame_values.shape == (12, 23)
    assert event.valid
    assert np.isfinite(event.values).all()


def test_feature_cache_round_trip_binds_event_identity(tmp_path) -> None:
    records = [
        {
            "sample_id": "p:event-0",
            "parent_sample_id": "p",
            "causal_event_index": 0,
            "split_name": "train",
        },
        {
            "sample_id": "p:event-1",
            "parent_sample_id": "p",
            "causal_event_index": 1,
            "split_name": "train",
        },
    ]
    descriptor = FeatureCacheDescriptor(
        schema_version=FEATURE_CACHE_SCHEMA_VERSION,
        feature_name="test_feature",
        source_manifest_hash="manifest-sha",
        feature_config_hash="config-sha",
        view_name="train:l1-known",
        view_hash="view-sha",
        normalization_hash="normalization-sha",
    )
    cache = build_feature_cache_from_records(
        records,
        np.asarray([[1.0, 2.0], [0.0, 0.0]], dtype=np.float32),
        np.asarray([True, False], dtype=np.bool_),
        descriptor=descriptor,
        feature_names=("a", "b"),
    )
    path = save_feature_cache(tmp_path / "features.npz", cache)
    loaded = load_feature_cache(
        path,
        expected_feature_name="test_feature",
        expected_sample_ids=["p:event-0", "p:event-1"],
        expected_split_name="train",
        expected_view_name="train:l1-known",
        expected_view_hash="view-sha",
    )

    assert loaded.sample_ids == cache.sample_ids
    np.testing.assert_array_equal(loaded.values, cache.values)
    with pytest.raises(ValueError, match="sample_id"):
        load_feature_cache(path, expected_sample_ids=["p:event-0"])
    with pytest.raises(ValueError, match="view_hash"):
        load_feature_cache(path, expected_view_hash="different-view")


def test_feature_cache_rejects_non_contiguous_parent_events() -> None:
    descriptor = FeatureCacheDescriptor(
        schema_version=FEATURE_CACHE_SCHEMA_VERSION,
        feature_name="test_feature",
        source_manifest_hash="manifest-sha",
        feature_config_hash="config-sha",
        view_name="train:l1-known",
        view_hash="view-sha",
    )
    with pytest.raises(ValueError, match="连续"):
        build_feature_cache_from_records(
            [
                {
                    "sample_id": "p:event-1",
                    "parent_sample_id": "p",
                    "causal_event_index": 1,
                    "split_name": "train",
                }
            ],
            np.asarray([[1.0]], dtype=np.float32),
            np.asarray([True], dtype=np.bool_),
            descriptor=descriptor,
            feature_names=("a",),
        )


def test_stable8_normalization_is_parent_equal_and_invalid_rows_zero() -> None:
    values = np.asarray(
        [[1.0] * 8, [3.0] * 8, [100.0] * 8], dtype=np.float32
    )
    reliable = np.asarray([True, True, False], dtype=np.bool_)
    parents = ["p1", "p1", "p2"]
    normalization = fit_parent_equal_median_iqr(
        values,
        reliable,
        parents,
        feature_names=[f"f{index}" for index in range(8)],
    )
    normalized = normalize_stable8(
        values,
        reliable,
        median=normalization.median,
        scale=normalization.scale,
    )

    assert normalization.parent_count_used == 1
    assert np.array_equal(normalized[2], np.zeros((8,), dtype=np.float32))
    assert np.isfinite(normalized).all()


def test_prepare_rd_event_preserves_raw_amplitude_until_physical_stage() -> None:
    amplitude = np.ones((2, 3, 4), dtype=np.float32)
    event = prepare_rd_event(
        {
            "amplitude": amplitude,
            "phase": np.zeros_like(amplitude),
            "time_seconds": np.asarray([0.0, 0.1], dtype=np.float32),
            "frame_count": 2,
        }
    )

    assert event is not None
    np.testing.assert_array_equal(event.amplitude, amplitude)
    assert event.original_count == 2
    assert event.local_time_non_monotonic is False


def test_missing_rd_model_frames_require_explicit_spatial_shape() -> None:
    from radar_ln3.features import build_rd_model_frames

    with pytest.raises(ValueError, match="spatial_shape"):
        build_rd_model_frames(
            None,
            amplitude_center=0.0,
            amplitude_scale=1.0,
            max_frames=2,
        )
    frames, mask = build_rd_model_frames(
        None,
        amplitude_center=0.0,
        amplitude_scale=1.0,
        max_frames=2,
        spatial_shape=(21, 384),
    )
    assert frames.shape == (2, 2, 21, 384)
    assert not mask.any()


def test_feature_cache_bundle_binds_one_unified_view(tmp_path) -> None:
    class Provider:
        def read_event_frames(self, event):
            iq = np.zeros((31, 64), dtype=np.complex64)
            iq[15] = np.exp(
                2j * np.pi * np.arange(64, dtype=np.float64) / 64.0
            )
            return [
                RawEchoFrame(
                    raw_frame_index=0,
                    frequency_hz=15.6e9,
                    cpi_count=1,
                    prt_num=64,
                    prt_seconds=20.0e-6,
                    target_records=(
                        RawTargetRecord(1, event.causal_event_index, 15, 32),
                    ),
                    iq=iq,
                )
            ]

    events = [
        EventData(
            sample_id=f"p:event-{index}",
            parent_sample_id="p",
            track_id="1",
            causal_event_index=index,
            label_id=0,
            label_name="DroneTarget",
            split_name="train",
            tr_values=_tr_event(time_seconds=float(index)),
            rd_values={
                "amplitude": np.ones((1, 21, 384), dtype=np.float32) * 55.9,
                "phase": np.zeros((1, 21, 384), dtype=np.float32),
                "time_seconds": np.asarray([0.0], dtype=np.float32),
                "frame_count": 1,
            },
            tr_available=True,
            rd_available=True,
            tr_dt_seconds=0.0 if index == 0 else 1.0,
            tr_dt_valid=index > 0,
            tr_dt_reason=(
                "first_event_no_predecessor"
                if index == 0
                else "positive_delta_at_second_precision"
            ),
        )
        for index in range(2)
    ]
    sequence = EventSequence(
        parent_sample_id="p",
        track_id="1",
        label_id=0,
        label_name="DroneTarget",
        split_name="train",
        events=events,
        source_name="test",
    )
    view = DatasetView(
        view_name="audit",
        split_name=None,
        include_unknown=False,
        training_view=False,
        allowed_labels=None,
        sample_ids=tuple(event.sample_id for event in events),
        parent_ids=("p",),
        event_ids_by_parent={"p": tuple(event.sample_id for event in events)},
        task_name="l2",
    )
    records = [
        {
            "sample_id": event.sample_id,
            "parent_sample_id": "p",
            "causal_event_index": event.causal_event_index,
            "split_name": "train",
        }
        for event in events
    ]
    caches = build_feature_cache_bundle(
        [sequence],
        records,
        view=view,
        source_manifest_hash="a" * 64,
        raw_echo_provider=Provider(),
        stable_median=np.zeros((8,), dtype=np.float32),
        stable_scale=np.ones((8,), dtype=np.float32),
    )
    assert caches["micro_doppler_evidence"].values.shape == (2, 12, 23)
    assert caches["micro_doppler_stable8_frame"].valid.shape == (2, 12)
    save_feature_cache_bundle(tmp_path, caches, output_root="data/features")
    loaded = load_feature_cache_bundle(
        tmp_path,
        "data/features/feature-cache-manifest.json",
        expected_view=view,
        expected_source_manifest_hash="a" * 64,
    )
    assert loaded["tr_observation"].sample_ids == view.sample_ids
    batch = build_model_batch(
        [sequence],
        task="l2",
        feature_caches=loaded,
        expected_view_hash=view.view_hash,
    )
    assert batch["stable_summary_valid"].all()
    assert batch["stable_frame_mask"].any()
    assert batch["micro_doppler_valid"].all()
    with pytest.raises(ValueError, match="view_hash"):
        build_model_batch(
            [sequence],
            task="l2",
            feature_caches=loaded,
            expected_view_hash="different-view",
        )


def _two_event_sequence() -> EventSequence:
    events = [
        EventData(
            sample_id=f"p:event-{index}",
            parent_sample_id="p",
            track_id="p",
            causal_event_index=index,
            label_id=0,
            label_name="DroneTarget",
            split_name="train",
            tr_values=_tr_event(time_seconds=float(index)),
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
        for index in range(2)
    ]
    return EventSequence(
        parent_sample_id="p",
        track_id="p",
        label_id=0,
        label_name="DroneTarget",
        split_name="train",
        events=events,
        source_name="fixture",
    )


def _complete_l1_caches(
    sequence: EventSequence,
    *,
    view_hash: str,
) -> dict[str, FeatureCache]:
    sample_ids = tuple(str(event.sample_id) for event in sequence.events)

    def cache(
        name: str,
        dim: int,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> FeatureCache:
        return FeatureCache(
            descriptor=FeatureCacheDescriptor(
                schema_version=FEATURE_CACHE_SCHEMA_VERSION,
                feature_name=name,
                source_manifest_hash="a" * 64,
                feature_config_hash="b" * 64,
                view_name="audit",
                view_hash=view_hash,
            ),
            feature_names=tuple(f"{name}-{index}" for index in range(dim)),
            sample_ids=sample_ids,
            parent_sample_ids=tuple("p" for _ in sample_ids),
            causal_event_indices=np.arange(len(sample_ids), dtype=np.int64),
            split_names=tuple("train" for _ in sample_ids),
            values=np.zeros((len(sample_ids), dim), dtype=np.float32),
            valid=np.ones((len(sample_ids), dim), dtype=np.bool_),
            metadata=metadata or {},
        )

    return {
        "tr_observation": cache("tr_observation", TR_VALUE_DIM),
        "tr_observation_increment": cache("tr_observation_increment", 9),
        "course_circular_incoherence": cache(
            COURSE_FEATURE_NAME,
            1,
            metadata={"course_support": [0, 1]},
        ),
    }


def test_cache_positions_reuses_stored_sample_positions() -> None:
    from radar_ln3.features.tensor import _cache_positions

    sequence = _two_event_sequence()
    caches = _complete_l1_caches(sequence, view_hash="view-hash")
    positions = _cache_positions(
        caches,
        expected_view_hash="view-hash",
    )

    assert positions["tr_observation"] is caches["tr_observation"].sample_positions
    assert positions["tr_observation"]["p:event-1"] == 1


def test_l1_batch_does_not_reencode_complete_feature_cache(monkeypatch) -> None:
    import radar_ln3.features.tensor as tensor_module

    sequence = _two_event_sequence()
    caches = _complete_l1_caches(sequence, view_hash="view-hash")
    calls: list[str] = []
    for name in (
        "encode_tr_sequence",
        "encode_causal_tr_observation_increment_sequence",
        "encode_causal_course_sequence",
    ):
        original = getattr(tensor_module, name)

        def counted(original=original, name=name):
            def wrapper(*args, **kwargs):
                calls.append(name)
                return original(*args, **kwargs)

            return wrapper

        monkeypatch.setattr(tensor_module, name, counted())

    batch = build_model_batch(
        [sequence],
        task="l1",
        feature_caches=caches,
        expected_view_hash="view-hash",
    )

    assert calls == []
    assert batch["tr_features"].count_nonzero() == 0
    assert batch["reference_tr_features"].count_nonzero() == 0
    assert batch["observation_increment_values"].count_nonzero() == 0
    assert batch["course_support"].tolist() == [[0, 1]]


def test_l1_batch_encodes_reference_tr_once(monkeypatch) -> None:
    import radar_ln3.features.tensor as tensor_module

    calls = 0
    original = tensor_module.encode_tr_sequence

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tensor_module, "encode_tr_sequence", counted)
    batch = build_model_batch([_two_event_sequence()], task="l1")

    assert calls == 1
    assert batch["tr_features"].count_nonzero() > 0
