from __future__ import annotations

import numpy as np
import pytest

from radar_ln3.data.contracts import EventData, EventRecord, ParentRecord
from radar_ln3.data.index import build_manifest, write_manifest
from radar_ln3.data.temporal import (
    TR_DT_REASON_FIRST,
    TR_DT_REASON_NEGATIVE,
    TR_DT_REASON_POSITIVE,
    TR_DT_REASON_ZERO,
    build_temporal_metadata,
)
from radar_ln3.cache.contracts import CACHE_SCHEMA_VERSION


def test_temporal_metadata_preserves_causal_order_and_marks_same_second() -> None:
    result = build_temporal_metadata([10.0, 10.0, 12.5])

    assert [item["dt_reason"] for item in result] == [
        TR_DT_REASON_FIRST,
        TR_DT_REASON_ZERO,
        TR_DT_REASON_POSITIVE,
    ]
    assert [item["dt_seconds"] for item in result] == [0.0, 0.0, 2.5]
    assert [item["dt_valid"] for item in result] == [False, False, True]


def test_negative_source_delta_is_preserved_and_marked_invalid() -> None:
    result = build_temporal_metadata([10.0, 9.0, 9.0])

    assert result[1]["dt_seconds"] == -1.0
    assert result[1]["dt_valid"] is False
    assert result[1]["dt_reason"] == TR_DT_REASON_NEGATIVE
    assert result[1]["dt_is_negative"] is True
    assert result[1]["time_order_status"] == "non_monotonic"


def test_negative_parent_events_are_training_excluded() -> None:
    events = [
        EventData(
            sample_id=f"bad:event-{index}",
            parent_sample_id="bad",
            track_id="bad",
            causal_event_index=index,
            label_id=0,
            label_name="DroneTarget",
            split_name="train",
            tr_values={"time_seconds": value},
            rd_values=None,
            tr_available=True,
            rd_available=False,
        )
        for index, value in enumerate((10.0, 9.0))
    ]
    from radar_ln3.data.temporal import annotate_event_records

    records = [
        EventRecord(
            dataset_name="fixture",
            source_name="fixture",
            sample_id=event.sample_id,
            parent_sample_id=event.parent_sample_id,
            track_id=event.track_id,
            causal_event_index=event.causal_event_index,
            label_id=event.label_id,
            label_name=event.label_name,
            split_name=event.split_name,
            available=True,
            source_path="external/fixture/track.txt",
            data_path="",
            metadata={"tr_values": event.tr_values},
        )
        for event in events
    ]
    annotated = annotate_event_records(records)

    assert all(item.metadata["parent_anomalous"] for item in annotated)
    assert all(item.metadata["training_excluded"] for item in annotated)
    assert annotated[1].metadata["tr_timing"]["dt_seconds"] == -1.0


def test_manifest_training_view_excludes_anomalous_parent(tmp_path) -> None:
    class FixtureAdapter:
        dataset_name = "fixture-dataset"

        def discover_parents(self) -> list[ParentRecord]:
            return [
                ParentRecord(
                    dataset_name=self.dataset_name,
                    source_name="fixture-source",
                    parent_sample_id="fixture|parent|bad",
                    track_id="bad",
                    label_id=0,
                    label_name="DroneTarget",
                    source_path="external/fixture/track.txt",
                    data_path="",
                    split_name="train",
                )
            ]

        def discover_events(self, parent: ParentRecord) -> list[EventRecord]:
            return [
                EventRecord(
                    dataset_name=self.dataset_name,
                    source_name="fixture-source",
                    sample_id=f"{parent.parent_sample_id}:event-{index}",
                    parent_sample_id=parent.parent_sample_id,
                    track_id=parent.track_id,
                    causal_event_index=index,
                    label_id=parent.label_id,
                    label_name=parent.label_name,
                    split_name=parent.split_name,
                    available=True,
                    source_path=parent.source_path,
                    data_path="",
                    metadata={"tr_values": {"time_seconds": value}},
                )
                for index, value in enumerate((10.0, 9.0))
            ]

        def assign_split(self, parent_ids: list[str]) -> dict[str, str]:
            return {parent_id: "train" for parent_id in parent_ids}

    manifest = build_manifest(FixtureAdapter(), tmp_path, materialize=False)
    assert all(row["metadata"]["parent_anomalous"] for row in manifest.tr_records)
    assert all(row["metadata"]["training_excluded"] for row in manifest.tr_records)
    assert manifest.tr_records[1]["metadata"]["tr_timing"]["dt_seconds"] == -1.0

    manifest_path = write_manifest(manifest, tmp_path, path="manifest.json")

    from radar_ln3.data.dataset import ParentSequenceDataset

    with pytest.raises(ValueError, match="选择为空"):
        ParentSequenceDataset(
            tmp_path,
            manifest_path=manifest_path,
            training_view=True,
            verify_files=True,
        )


def test_cache_schema_includes_temporal_provenance_version() -> None:
    assert CACHE_SCHEMA_VERSION == "RADAR-LN3-EVENT-SOURCE-CACHE-V4"
