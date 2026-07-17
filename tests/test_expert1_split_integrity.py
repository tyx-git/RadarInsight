from __future__ import annotations

from collections import Counter
from unittest.mock import Mock, patch

import pandas as pd
import pytest

import src.core.expert1_data as expert1_data

from src.core.data.split import (
    SplitResult,
    canonical_group_id,
    stratified_group_split,
    validate_grouped_split,
)


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
