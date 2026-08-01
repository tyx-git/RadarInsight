from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pandas as pd
import pytest

import src.core.expert1_data as expert1_data

from src.core.data.split import (
    SplitResult,
    canonical_group_id,
    stratified_group_split,
    validate_grouped_split,
)
from src.process.standardize_tracks import (
    _derive_raw_radial_speed_mps,
    _normalize_excel_radial_speed_mps,
    _normalize_excel_time_seconds,
    _normalize_raw_data_time_seconds,
    _normalize_raw_range_m,
    _processed_output_dir,
    _versioned_output_name,
)


def test_track_time_normalizers_keep_source_units_explicit() -> None:
    raw_seconds_since_midnight = np.asarray([41162.798, 41163.618, 41164.437])
    excel_timer_milliseconds = np.asarray([1000.0, 3006.0, 5012.0])

    assert _normalize_raw_data_time_seconds(raw_seconds_since_midnight) == pytest.approx(
        [0.0, 0.82, 1.639]
    )
    assert _normalize_excel_time_seconds(excel_timer_milliseconds) == pytest.approx(
        [0.0, 2.006, 4.012]
    )


def test_versioned_tr_only_standardization_rejects_unsafe_output_names(
    tmp_path: Path,
) -> None:
    assert _versioned_output_name("raw_tracks", "v10") == "raw_tracks_tr_only_v10"
    assert _processed_output_dir(tmp_path, "raw_tracks_tr_only_v10") == (
        tmp_path / "data" / "processed" / "raw_tracks_tr_only_v10"
    ).resolve()
    for invalid in ("../raw_tracks", "nested/raw_tracks", "C:/raw_tracks", ""):
        with pytest.raises(ValueError):
            _processed_output_dir(tmp_path, invalid)
    for invalid_version in ("../v10", "V 10", ""):
        with pytest.raises(ValueError):
            _versioned_output_name("raw_tracks", invalid_version)


def test_raw_range_normalizer_converts_kilometres_to_metres() -> None:
    assert _normalize_raw_range_m(np.asarray([0.288, 4.160, 13.200])) == pytest.approx(
        [288.0, 4160.0, 13200.0]
    )


def test_excel_radial_speed_normalizer_uses_positive_receding_convention() -> None:
    assert _normalize_excel_radial_speed_mps(
        np.asarray([-12.5, 0.0, 8.0])
    ) == pytest.approx([12.5, 0.0, -8.0])


def test_raw_radial_speed_uses_only_current_and_prior_points() -> None:
    time_seconds = np.asarray([0.0, 0.82, 0.82, 1.639])
    range_m = np.asarray([2346.732, 2357.687, 2357.687, 2367.966])
    derived = _derive_raw_radial_speed_mps(
        time_seconds,
        range_m,
        np.zeros(4),
    )

    assert np.isnan(derived[0])
    assert derived[1:] == pytest.approx(
        [
            (2357.687 - 2346.732) / 0.82,
            (2357.687 - 2346.732) / 0.82,
            (2367.966 - 2357.687) / (1.639 - 0.82),
        ]
    )
    with pytest.raises(ValueError, match="non-zero source values"):
        _derive_raw_radial_speed_mps(time_seconds, range_m, np.asarray([0, 0, 1, 0]))
    with pytest.raises(ValueError, match="inconsistent range"):
        _derive_raw_radial_speed_mps(
            np.asarray([0.0, 0.0]),
            np.asarray([1000.0, 1001.0]),
            np.zeros(2),
        )


def test_raw_data_time_real_manifest_is_seconds_since_midnight() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "manifests" / "raw_tr_rd_manifest.json"
    if not manifest_path.exists():
        pytest.skip("raw TR manifest is not available")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    positive_deltas: list[float] = []
    start_offsets_from_filename: list[float] = []
    checked_records = 0
    for record in records:
        track_path = Path(record["tr_file"])
        if not track_path.is_absolute():
            track_path = project_root / track_path
        if not track_path.exists():
            pytest.skip(f"raw TR asset is not available: {track_path}")
        with h5py.File(track_path, "r") as handle:
            time_raw = np.asarray(handle["DATA_time"]).reshape(-1).astype(float)
        deltas = np.diff(time_raw)
        positive_deltas.extend(deltas[np.isfinite(deltas) & (deltas > 0)].tolist())
        timestamp = datetime.strptime(str(record["timestamp"]), "%Y%m%d%H%M%S")
        filename_seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
        start_offsets_from_filename.append(float(time_raw[0] - filename_seconds))
        normalized = _normalize_raw_data_time_seconds(time_raw)
        assert normalized[-1] == pytest.approx(time_raw[-1] - time_raw[0])
        checked_records += 1

    assert checked_records == len(records) == 77
    assert np.median(positive_deltas) == pytest.approx(0.82, abs=0.01)
    assert min(start_offsets_from_filename) >= 0.0
    assert max(start_offsets_from_filename) < 3600.0


def test_raw_range_real_manifest_has_metre_scale_and_physical_motion() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "manifests" / "raw_tr_rd_manifest.json"
    if not manifest_path.exists():
        pytest.skip("raw TR manifest is not available")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    range_native: list[float] = []
    geometric_speed_mps: list[float] = []
    for record in records:
        track_path = Path(record["tr_file"])
        if not track_path.is_absolute():
            track_path = project_root / track_path
        if not track_path.exists():
            pytest.skip(f"raw TR asset is not available: {track_path}")
        with h5py.File(track_path, "r") as handle:
            time_raw = np.asarray(handle["DATA_time"]).reshape(-1).astype(float)
            azimuth = np.deg2rad(
                np.asarray(handle["A"]).reshape(-1).astype(float)
            )
            elevation = np.deg2rad(
                np.asarray(handle["E"]).reshape(-1).astype(float)
            )
            native = np.asarray(handle["R"]).reshape(-1).astype(float)
        range_m = _normalize_raw_range_m(native)
        x = range_m * np.cos(elevation) * np.cos(azimuth)
        y = range_m * np.cos(elevation) * np.sin(azimuth)
        z = range_m * np.sin(elevation)
        delta_time = np.diff(time_raw)
        displacement = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2 + np.diff(z) ** 2)
        valid = np.isfinite(delta_time) & np.isfinite(displacement) & (delta_time > 0)
        range_native.extend(native[np.isfinite(native)].tolist())
        geometric_speed_mps.extend((displacement[valid] / delta_time[valid]).tolist())

    assert len(records) == 77
    assert np.median(range_native) == pytest.approx(4.1604, abs=0.01)
    assert np.median(_normalize_raw_range_m(np.asarray(range_native))) == pytest.approx(
        4160.4, abs=10.0
    )
    assert np.median(geometric_speed_mps) == pytest.approx(16.3535, abs=0.1)


def test_raw_radial_speed_real_manifest_replaces_zero_placeholder_causally() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "manifests" / "raw_tr_rd_manifest.json"
    if not manifest_path.exists():
        pytest.skip("raw TR manifest is not available")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    derived_values: list[float] = []
    missing_count = 0
    point_count = 0
    for record in records:
        track_path = Path(record["tr_file"])
        if not track_path.is_absolute():
            track_path = project_root / track_path
        if not track_path.exists():
            pytest.skip(f"raw TR asset is not available: {track_path}")
        with h5py.File(track_path, "r") as handle:
            time_raw = np.asarray(handle["DATA_time"]).reshape(-1).astype(float)
            range_native = np.asarray(handle["R"]).reshape(-1).astype(float)
            source_speed = np.asarray(handle["V"]).reshape(-1).astype(float)
        derived = _derive_raw_radial_speed_mps(
            _normalize_raw_data_time_seconds(time_raw),
            _normalize_raw_range_m(range_native),
            source_speed,
        )
        point_count += len(derived)
        missing_count += int((~np.isfinite(derived)).sum())
        derived_values.extend(derived[np.isfinite(derived)].tolist())

    assert point_count == 11366
    assert missing_count >= len(records)
    assert np.median(np.abs(derived_values)) == pytest.approx(11.5369, abs=0.1)


def test_excel_radial_speed_real_manifest_matches_range_rate_after_sign_fix() -> None:
    project_root = Path(__file__).resolve().parents[1]
    manifest_path = project_root / "data" / "manifests" / "excel_track_manifest.json"
    if not manifest_path.exists():
        pytest.skip("Excel track manifest is not available")
    records = json.loads(manifest_path.read_text(encoding="utf-8"))
    range_rates: list[float] = []
    normalized_speeds: list[float] = []
    for record in records:
        track_path = Path(record["track_file"])
        if not track_path.is_absolute():
            track_path = project_root / track_path
        if not track_path.exists():
            pytest.skip(f"Excel track asset is not available: {track_path}")
        frame = pd.read_excel(
            track_path,
            usecols=["Time", "Distance", "Speed_R", "Batch"],
        )
        for _, batch in frame.groupby("Batch", sort=False):
            time_seconds = batch["Time"].to_numpy(dtype=float) / 1000.0
            range_m = batch["Distance"].to_numpy(dtype=float)
            speed_mps = _normalize_excel_radial_speed_mps(
                batch["Speed_R"].to_numpy(dtype=float)
            )
            delta_time = np.diff(time_seconds)
            rate = np.divide(
                np.diff(range_m),
                delta_time,
                out=np.full(delta_time.shape, np.nan),
                where=delta_time > 1.0e-6,
            )
            valid = (
                np.isfinite(rate)
                & np.isfinite(speed_mps[1:])
                & (delta_time > 1.0e-6)
                & (np.abs(rate) < 200.0)
            )
            range_rates.extend(rate[valid].tolist())
            normalized_speeds.extend(speed_mps[1:][valid].tolist())

    rates = np.asarray(range_rates)
    speeds = np.asarray(normalized_speeds)
    assert len(rates) == 24160
    assert np.corrcoef(rates, speeds)[0, 1] > 0.95
    assert np.median(np.abs(rates - speeds)) < 1.0


def test_cq08_radial_speed_real_processed_data_is_positive_receding() -> None:
    project_root = Path(__file__).resolve().parents[1]
    index_path = project_root / "data" / "processed" / "cq08_track_index.csv"
    if not index_path.exists():
        pytest.skip("CQ08 track index is not available")
    index = pd.read_csv(index_path, low_memory=False)
    range_rates: list[float] = []
    radial_speeds: list[float] = []
    for relative_path in index["csv_path"].dropna().astype(str):
        track_path = project_root / relative_path
        if not track_path.exists():
            pytest.skip(f"CQ08 track asset is not available: {track_path}")
        frame = pd.read_csv(
            track_path,
            usecols=["time_seconds", "range_m", "radial_speed_mps"],
        )
        time_seconds = pd.to_numeric(frame["time_seconds"], errors="coerce").to_numpy(float)
        range_m = pd.to_numeric(frame["range_m"], errors="coerce").to_numpy(float)
        speed_mps = pd.to_numeric(
            frame["radial_speed_mps"], errors="coerce"
        ).to_numpy(float)
        delta_time = np.diff(time_seconds)
        rate = np.divide(
            np.diff(range_m),
            delta_time,
            out=np.full(delta_time.shape, np.nan),
            where=delta_time > 1.0e-6,
        )
        valid = np.isfinite(rate) & np.isfinite(speed_mps[1:]) & (delta_time > 1.0e-6)
        range_rates.extend(rate[valid].tolist())
        radial_speeds.extend(speed_mps[1:][valid].tolist())

    rates = np.asarray(range_rates)
    speeds = np.asarray(radial_speeds)
    active = (np.abs(rates) > 0.1) & (np.abs(speeds) > 0.1)
    assert len(rates) == 18356
    assert np.mean(np.sign(rates[active]) == np.sign(speeds[active])) > 0.95
    assert np.median(np.abs(rates - speeds)) < 1.1


def test_canonical_group_id_prefers_parent_then_window_prefix_then_sample_id() -> None:
    assert canonical_group_id({"parent_sample_id": " parent-1 ", "sample_id": "other|win0001"}) == "parent-1"
    assert canonical_group_id({"parent_sample_id": "NaN", "sample_id": "track-2|win0002"}) == "track-2"
    assert canonical_group_id({"parent_sample_id": None, "sample_id": "track-3"}) == "track-3"
    assert canonical_group_id({"source_name": "raw_tr_rd", "parent_sample_id": "parent-1", "sample_id": "other|win0001"}) == "raw_tr_rd::parent-1"
    assert canonical_group_id({"parent_sample_id": ["parent-1"], "sample_id": "other|win0001"}) == "['parent-1']"
    with pytest.raises(ValueError, match="sample_id"):
        canonical_group_id({"parent_sample_id": "none", "sample_id": " "})


def test_validate_grouped_split_separates_same_parent_across_sources() -> None:
    index_df = pd.DataFrame(
        [
            {"sample_id": "raw-parent|win0000", "parent_sample_id": "parent", "source_name": "raw_tr_rd", "label_name": "DroneTarget"},
            {"sample_id": "excel-parent|win0000", "parent_sample_id": "parent", "source_name": "excel_track", "label_name": "BirdTarget"},
        ]
    )
    split = SplitResult(train_ids=["raw-parent|win0000"], val_ids=["excel-parent|win0000"], test_ids=[])

    validate_grouped_split(index_df, split)


def test_sample_weight_uses_source_qualified_fallback_groups_for_missing_parents() -> None:
    raw_row = {"sample_id": "track|win0000", "parent_sample_id": float("nan"), "source_name": "raw_tr_rd", "label_name": "DroneTarget"}
    excel_row = {"sample_id": "track|win0000", "parent_sample_id": pd.NA, "source_name": "excel_track", "label_name": "DroneTarget"}
    parent_window_counts = dict(Counter(canonical_group_id(row) for row in (raw_row, excel_row)))

    assert parent_window_counts == {"raw_tr_rd::track": 1, "excel_track::track": 1}
    shared_kwargs = {
        "source_counts": {},
        "source_label_counts": {},
        "max_source_count": 1,
        "max_source_label_count": 1,
        "rd_sample_boost": 1.0,
        "raw_source_boost": 1.0,
        "excel_source_boost": 1.0,
        "cq08_source_boost": 1.0,
        "source_balance_power": 0.0,
        "source_label_balance_power": 0.0,
        "parent_window_balance_power": 1.0,
        "unknown_sample_boost": 1.0,
        "clutter_sample_boost": 1.0,
        "bird_sample_boost": 1.0,
        "max_sampler_weight": 10.0,
    }

    assert expert1_data._sample_weight(raw_row, parent_window_counts=parent_window_counts, **shared_kwargs) == pytest.approx(1.0)
    assert expert1_data._sample_weight(excel_row, parent_window_counts=parent_window_counts, **shared_kwargs) == pytest.approx(1.0)


def test_rd_fusion_eligibility_map_only_promotes_explicit_safe_windows(tmp_path) -> None:
    data_root = tmp_path / "data"
    processed_root = data_root / "processed"
    processed_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {"sample_id": "exact", "rd_fusion_eligible": True},
            {"sample_id": "fallback", "rd_fusion_eligible": False},
        ]
    ).to_csv(processed_root / "window_map.csv", index=False)

    eligibility = expert1_data._rd_fusion_eligibility_by_sample(
        data_root,
        "window_map.csv",
    )

    assert eligibility == {"exact": True, "fallback": False}


def test_sample_weight_does_not_apply_rd_boost_to_explicitly_ineligible_window() -> None:
    shared_kwargs = {
        "source_counts": {"cq08_track": 1},
        "source_label_counts": {("cq08_track", "DroneTarget"): 1},
        "parent_window_counts": {},
        "max_source_count": 1,
        "max_source_label_count": 1,
        "rd_sample_boost": 5.0,
        "raw_source_boost": 1.0,
        "excel_source_boost": 1.0,
        "cq08_source_boost": 1.0,
        "source_balance_power": 0.0,
        "source_label_balance_power": 0.0,
        "parent_window_balance_power": 0.0,
        "unknown_sample_boost": 1.0,
        "clutter_sample_boost": 1.0,
        "bird_sample_boost": 1.0,
        "max_sampler_weight": 100.0,
    }
    row = {
        "source_name": "cq08_track",
        "label_name": "DroneTarget",
        "rd_tensor_path": "data/processed/cq08_rd/example.npz",
        "rd_fusion_eligible": False,
    }

    assert expert1_data._sample_weight(row, **shared_kwargs) == pytest.approx(1.0)


def test_validate_grouped_split_rejects_duplicate_sample_ids() -> None:
    index_df = pd.DataFrame(
        [
            {"sample_id": "a|win0000", "parent_sample_id": "a", "label_name": "DroneTarget"},
            {"sample_id": "a|win0000", "parent_sample_id": "a", "label_name": "DroneTarget"},
        ]
    )
    split = SplitResult(train_ids=["a|win0000"], val_ids=[], test_ids=[])

    with pytest.raises(ValueError, match="duplicate sample_id"):
        validate_grouped_split(index_df, split)


def test_validate_grouped_split_rejects_parent_crossing_splits() -> None:
    index_df = pd.DataFrame(
        [
            {"sample_id": "a|win0000", "parent_sample_id": "a", "label_name": "DroneTarget"},
            {"sample_id": "a|win0001", "parent_sample_id": "a", "label_name": "DroneTarget"},
        ]
    )
    split = SplitResult(train_ids=["a|win0000"], val_ids=["a|win0001"], test_ids=[])

    with pytest.raises(ValueError, match="groups cross splits"):
        validate_grouped_split(index_df, split)


def test_stratified_group_split_is_deterministic_and_keeps_sibling_windows_together() -> None:
    index_df = pd.DataFrame(
        [
            {"sample_id": f"drone-{group}|win{window:04d}", "parent_sample_id": f"drone-{group}", "label_name": "DroneTarget"}
            for group in range(3)
            for window in range(2)
        ]
        + [
            {"sample_id": f"bird-{group}|win{window:04d}", "parent_sample_id": f"bird-{group}", "label_name": "BirdTarget"}
            for group in range(3)
            for window in range(2)
        ]
    )

    first = stratified_group_split(index_df, seed=17)
    second = stratified_group_split(index_df, seed=17)

    assert first == second
    assignments = {
        sample_id: split_name
        for split_name, sample_ids in (
            ("train", first.train_ids),
            ("val", first.val_ids),
            ("test", first.test_ids),
        )
        for sample_id in sample_ids
    }
    for group_id, group_df in index_df.groupby("parent_sample_id"):
        assert {assignments[sample_id] for sample_id in group_df["sample_id"]} == {assignments[f"{group_id}|win0000"]}


def test_stratified_group_split_rejects_conflicting_group_labels() -> None:
    index_df = pd.DataFrame(
        [
            {"sample_id": "a|win0000", "parent_sample_id": "a", "label_name": "DroneTarget"},
            {"sample_id": "a|win0001", "parent_sample_id": "a", "label_name": "BirdTarget"},
        ]
    )

    with pytest.raises(ValueError, match="groups have conflicting labels"):
        stratified_group_split(index_df)


@pytest.mark.parametrize("invalid_split_name", ["unexpected", "", None, float("nan")])
def test_loader_rejects_invalid_persisted_split_before_dataset_construction(
    invalid_split_name: object,
) -> None:
    frame = pd.DataFrame(
        [
            {"sample_id": "drone|win0000", "parent_sample_id": "drone", "label_name": "DroneTarget", "split_name": "train"},
            {"sample_id": "bird|win0000", "parent_sample_id": "bird", "label_name": "BirdTarget", "split_name": invalid_split_name},
            {"sample_id": "balloon|win0000", "parent_sample_id": "balloon", "label_name": "BalloonTarget", "split_name": "test"},
        ]
    )
    track_dataset = Mock()
    track_dataset.return_value.to_frame.return_value = frame

    with (
        patch.object(expert1_data, "TrackDataset", track_dataset),
        patch.object(expert1_data, "TorchTrackDataset") as torch_dataset,
        pytest.raises(ValueError, match="persisted split_name contains malformed labels"),
    ):
        expert1_data.build_expert1_dataloaders(
            data_root="data",
            index_name="persisted_split_index.csv",
            validate_csv_files=False,
        )

    torch_dataset.assert_not_called()


@pytest.mark.parametrize(
    ("invalid_row", "validate_csv_files"),
    [
        (
            {
                "sample_id": "aircraft|win0000",
                "parent_sample_id": "aircraft",
                "label_name": "AircraftTarget",
                "split_name": "unexpected",
                "is_trainable": True,
                "csv_path": "present.csv",
            },
            False,
        ),
        (
            {
                "sample_id": "drone|win0001",
                "parent_sample_id": "drone-extra",
                "label_name": "DroneTarget",
                "split_name": "unexpected",
                "is_trainable": False,
                "csv_path": "present.csv",
            },
            False,
        ),
        (
            {
                "sample_id": "bird|win0001",
                "parent_sample_id": "bird-extra",
                "label_name": "BirdTarget",
                "split_name": "unexpected",
                "is_trainable": True,
                "csv_path": "missing.csv",
            },
            True,
        ),
    ],
)
def test_loader_rejects_invalid_persisted_split_before_filtering_rows(
    tmp_path,
    invalid_row: dict[str, object],
    validate_csv_files: bool,
) -> None:
    (tmp_path / "present.csv").touch()
    frame = pd.DataFrame(
        [
            {
                "sample_id": "drone|win0000",
                "parent_sample_id": "drone",
                "label_name": "DroneTarget",
                "split_name": "train",
                "is_trainable": True,
                "csv_path": "present.csv",
            },
            {
                "sample_id": "bird|win0000",
                "parent_sample_id": "bird",
                "label_name": "BirdTarget",
                "split_name": "val",
                "is_trainable": True,
                "csv_path": "present.csv",
            },
            {
                "sample_id": "balloon|win0000",
                "parent_sample_id": "balloon",
                "label_name": "BalloonTarget",
                "split_name": "test",
                "is_trainable": True,
                "csv_path": "present.csv",
            },
            invalid_row,
        ]
    )
    track_dataset = Mock()
    track_dataset.return_value.to_frame.return_value = frame

    with (
        patch.object(expert1_data, "TrackDataset", track_dataset),
        patch.object(expert1_data, "TorchTrackDataset") as torch_dataset,
        pytest.raises(ValueError, match="persisted split_name contains malformed labels"),
    ):
        expert1_data.build_expert1_dataloaders(
            data_root=tmp_path / "data",
            index_name="persisted_split_index.csv",
            validate_csv_files=validate_csv_files,
        )

    torch_dataset.assert_not_called()
