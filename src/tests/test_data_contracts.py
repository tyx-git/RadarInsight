from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.cli import build_parser
from radar_ln3.data.collate import collate_parent_sequences
from radar_ln3.data.contracts import (
    DataPairingError,
    DataFormatError,
    DataPathError,
    DataSplitError,
    EventData,
    EventSequence,
    ParentRecord,
    PathBoundaryError,
    RawEchoFrame,
    RawTargetRecord,
)
from radar_ln3.data.cq08 import TRACK_COLUMNS, _normalise_label, _snr_db, _validate_source_rows
from radar_ln3.data.cq08 import CQ08Adapter
from radar_ln3.data.dataset import ParentSequenceDataset
from radar_ln3.data.index import build_manifest, write_manifest
from radar_ln3.data.manifest import (
    MANIFEST_SCHEMA_VERSION,
    PAIRING_SCHEMA_VERSION,
    RD_SCHEMA_VERSION,
    TR_SCHEMA_VERSION,
    canonical_json_hash,
    data_manifest_hash,
    load_manifest,
    resolve_manifest_view,
    validate_manifest,
)
from radar_ln3.data.paths import (
    resolve_data_path,
    resolve_project_path,
    resolve_rd_path,
    resolve_source_path,
    resolve_tr_path,
)
from radar_ln3.data.raw_reader import sha256_file
from radar_ln3.cache import CachedParentSequenceDataset, build_event_cache, load_event_cache


EMPTY_HASH = hashlib.sha256(b"").hexdigest()


def test_raw_echo_frame_allows_audited_zero_cpi_count() -> None:
    frame = RawEchoFrame(
        raw_frame_index=0,
        frequency_hz=15.6e9,
        cpi_count=0,
        prt_num=226,
        prt_seconds=20.0e-6,
        target_records=(
            RawTargetRecord(
                target_identifier=1,
                event_index=0,
                range_bin=15,
                doppler_bin=100,
            ),
        ),
        iq=np.zeros((31, 226), dtype=np.complex64),
    )

    assert frame.cpi_count == 0


def _descriptor(path: str, *, count: int = 0, size: int = 0, sha256: str = EMPTY_HASH) -> dict[str, object]:
    return {"path": path, "record_count": count, "file_size": size, "sha256": sha256}


def _empty_manifest() -> dict[str, object]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": "fixture",
        "data_root": "data/cq08",
        "source_policy": {
            "parent_split_boundary": True,
            "frame_level_split_forbidden": True,
            "unknown_training_exposure": False,
        },
        "split": {"group_unit": "parent_sample_id", "split_field": "split_name"},
        "TR": {
            "schema_version": TR_SCHEMA_VERSION,
            "format": "jsonl",
            "record_count": 0,
            "parts": [_descriptor("data/cq08/tr/cq08-tr-event-index.jsonl")],
        },
        "RD": {
            "schema_version": RD_SCHEMA_VERSION,
            "format": "jsonl",
            "record_count": 0,
            "parts": [_descriptor("data/cq08/rd/cq08-rd-frame-index.jsonl")],
        },
        "pairing": {
            "schema_version": PAIRING_SCHEMA_VERSION,
            "strategy": "sample_id_and_causal_event_index",
            "tr_index": "TR",
            "rd_index": "RD",
            "rd_frame_cardinality": "many_frames_per_event",
            "paired_event_count": 0,
            "missing_tr_event_count": 0,
            "missing_rd_event_count": 0,
        },
        "audit": {"materialized": False},
    }


def _record(
    sample_id: str,
    *,
    available: bool,
    data_path: str = "",
    file_size: int | None = None,
    sha256: str | None = None,
    event_index: int,
    label_name: str = "DroneTarget",
) -> dict[str, object]:
    return {
        "dataset_name": "fixture",
        "source_name": "fixture-source",
        "sample_id": sample_id,
        "parent_sample_id": "parent-1",
        "track_id": "track-1",
        "causal_event_index": event_index,
        "label_id": 0,
        "label_name": label_name,
        "split_name": "train",
        "available": available,
        "source_path": "external/fixture/source.dat",
        "data_path": data_path,
        "file_size": file_size,
        "sha256": sha256,
        "metadata": {"baseline_training_view": True},
    }


def _write_fixture_sidecars(root: Path, tr_records: list[dict[str, object]], rd_frames: list[dict[str, object]]) -> None:
    tr_path = root / "data/cq08/tr/cq08-tr-event-index.jsonl"
    rd_path = root / "data/cq08/rd/cq08-rd-frame-index.jsonl"
    tr_path.parent.mkdir(parents=True, exist_ok=True)
    rd_path.parent.mkdir(parents=True, exist_ok=True)
    tr_path.write_bytes(b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in tr_records))
    rd_path.write_bytes(b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode() for row in rd_frames))


def _with_sidecars(root: Path, tr_records: list[dict[str, object]], rd_frames: list[dict[str, object]]) -> dict[str, object]:
    _write_fixture_sidecars(root, tr_records, rd_frames)
    payload = _empty_manifest()
    tr_path = root / "data/cq08/tr/cq08-tr-event-index.jsonl"
    rd_path = root / "data/cq08/rd/cq08-rd-frame-index.jsonl"
    payload["TR"] = {"schema_version": TR_SCHEMA_VERSION, "format": "jsonl", "record_count": len(tr_records), "parts": [_descriptor("data/cq08/tr/cq08-tr-event-index.jsonl", count=len(tr_records), size=tr_path.stat().st_size, sha256=sha256_file(tr_path))]}
    payload["RD"] = {"schema_version": RD_SCHEMA_VERSION, "format": "jsonl", "record_count": len(rd_frames), "parts": [_descriptor("data/cq08/rd/cq08-rd-frame-index.jsonl", count=len(rd_frames), size=rd_path.stat().st_size, sha256=sha256_file(rd_path))]}
    payload["pairing"] = {**payload["pairing"], "paired_event_count": len(tr_records), "missing_rd_event_count": sum(1 for row in tr_records if str(row["sample_id"]) not in {str(frame["sample_id"]) for frame in rd_frames})}
    return payload


def test_empty_manifest_is_valid_but_unmaterialized(tmp_path: Path) -> None:
    root = tmp_path
    (root / "data/cq08/tr").mkdir(parents=True)
    (root / "data/cq08/rd").mkdir(parents=True)
    (root / "data/cq08/tr/cq08-tr-event-index.jsonl").write_bytes(b"")
    (root / "data/cq08/rd/cq08-rd-frame-index.jsonl").write_bytes(b"")
    result = validate_manifest(_empty_manifest(), root, verify_files=True)
    assert result.dataset == "fixture"
    assert result.tr_records == []


def test_path_boundaries_reject_escape_and_wrong_modality(tmp_path: Path) -> None:
    with pytest.raises(PathBoundaryError):
        resolve_project_path(tmp_path, "../outside.json")
    with pytest.raises(PathBoundaryError):
        resolve_tr_path(tmp_path, "data/cq08/rd/other.npz")
    with pytest.raises(PathBoundaryError):
        resolve_rd_path(tmp_path, "data/cq08/tr/other.npz")
    with pytest.raises(PathBoundaryError):
        resolve_project_path(tmp_path, "src-v3/data.json", allow_external=True)
    with pytest.raises(PathBoundaryError):
        resolve_source_path(tmp_path, "external/cq08/data.json", source_roots={"cq08": tmp_path / "src-v3"}, allow_external=True)


def test_manifest_source_paths_reject_absolute_and_escape_values(tmp_path: Path) -> None:
    payload = _empty_manifest()
    record = _record("parent-1:event-0", available=False, event_index=0)
    record["source_path"] = "F:/Project/RadarInsight/OriData/source.txt"
    tr_path = tmp_path / "data/cq08/tr/cq08-tr-event-index.jsonl"
    rd_path = tmp_path / "data/cq08/rd/cq08-rd-frame-index.jsonl"
    tr_path.parent.mkdir(parents=True)
    rd_path.parent.mkdir(parents=True)
    tr_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rd_path.write_bytes(b"")
    payload["TR"] = {"schema_version": TR_SCHEMA_VERSION, "format": "jsonl", "record_count": 1, "parts": [_descriptor("data/cq08/tr/cq08-tr-event-index.jsonl", count=1, size=tr_path.stat().st_size, sha256=sha256_file(tr_path))]}
    payload["pairing"] = {**payload["pairing"], "paired_event_count": 0}
    with pytest.raises(DataPathError):
        validate_manifest(payload, tmp_path, verify_files=False)


def test_cq08_source_semantics() -> None:
    row = {column: "0" for column in TRACK_COLUMNS}
    row["点时间"] = "12:00:00"
    row["批号"] = "2"
    with pytest.raises(DataPairingError):
        _validate_source_rows([row], columns=TRACK_COLUMNS, path=Path("Tracks_1_1_1.txt"), length_hint=1, source_kind="track", expected_parent_id=1)
    assert _normalise_label("Other") == (4, "UnknownTarget")
    assert _normalise_label("6") == (4, "UnknownTarget")
    assert _normalise_label("Clutter") == (3, "ClutterTarget")
    assert _snr_db("156170") == pytest.approx(51.9359761, rel=1e-7)


def test_collate_uses_suffix_padding_and_modality_masks() -> None:
    def event(parent: str, index: int, label_id: int, rd: bool) -> EventData:
        return EventData(sample_id=f"{parent}:e{index}", parent_sample_id=parent, track_id=parent, causal_event_index=index, label_id=label_id, label_name="DroneTarget", split_name="train", tr_values={"range_m": float(index)}, rd_values={"amplitude": np.zeros((1, 2, 21, 384))} if rd else None, tr_available=True, rd_available=rd)
    first = EventSequence(parent_sample_id="p1", track_id="p1", label_id=0, label_name="DroneTarget", split_name="train", source_name="fixture", events=[event("p1", 0, 0, False)])
    second = EventSequence(parent_sample_id="p2", track_id="p2", label_id=1, label_name="DroneTarget", split_name="train", source_name="fixture", events=[event("p2", 0, 1, True), event("p2", 1, 1, False)])
    batch = collate_parent_sequences([first, second])
    assert batch["event_mask"].tolist() == [[True, False], [True, True]]
    assert batch["padding_mask"].tolist() == [[False, True], [False, False]]
    assert batch["modality_available"]["RD"].tolist() == [[False, False], [True, False]]


def test_sidecar_manifest_and_dataset_reading(tmp_path: Path) -> None:
    tr_path = tmp_path / "data/cq08/tr/parent-1.jsonl"
    rd_path = tmp_path / "data/cq08/rd/parent-1.npz"
    tr_path.parent.mkdir(parents=True)
    rd_path.parent.mkdir(parents=True)
    tr_rows = [{"schema_version": "fixture-tr-v1", "sample_id": "parent-1:event-0", "parent_sample_id": "parent-1", "track_id": "track-1", "causal_event_index": 0, "label_id": 0, "label_name": "DroneTarget", "split_name": "train", "values": {"range_m": 1.0}}]
    tr_path.write_text(json.dumps(tr_rows[0]) + "\n", encoding="utf-8")
    np.savez(rd_path, amplitude=np.ones((1, 3, 4), dtype=np.float32), phase=np.zeros((1, 3, 4), dtype=np.float32), time_seconds=np.zeros((1,), dtype=np.float32), frame_event_indices=np.zeros((1,), dtype=np.int64))
    tr = _record("parent-1:event-0", available=True, data_path="data/cq08/tr/parent-1.jsonl", file_size=tr_path.stat().st_size, sha256=sha256_file(tr_path), event_index=0)
    rd_frame = {**_record("parent-1:event-0", available=True, data_path="data/cq08/rd/parent-1.npz", file_size=rd_path.stat().st_size, sha256=sha256_file(rd_path), event_index=0), "record_type": "frame", "frame_index": 0, "source_path": "external/fixture/frame.mat", "metadata": {}}
    payload = _with_sidecars(tmp_path, [tr], [rd_frame])
    target = write_manifest(payload, tmp_path, path="manifest.json")
    dataset = ParentSequenceDataset(tmp_path, manifest_path=target, training_view=True)
    sequence = dataset[0]
    assert sequence.events[0].rd_values["amplitude"].shape == (1, 3, 4)


def test_event_cache_registry_and_training_dataset(tmp_path: Path) -> None:
    tr_path = tmp_path / "data/cq08/tr/parent-1.jsonl"
    rd_path = tmp_path / "data/cq08/rd/parent-1.npz"
    tr_path.parent.mkdir(parents=True)
    rd_path.parent.mkdir(parents=True)
    tr_row = {
        "schema_version": "fixture-tr-v1",
        "sample_id": "parent-1:event-0",
        "parent_sample_id": "parent-1",
        "track_id": "track-1",
        "causal_event_index": 0,
        "label_id": 0,
        "label_name": "DroneTarget",
        "split_name": "train",
        "values": {"range_m": 1.0},
    }
    tr_path.write_text(json.dumps(tr_row) + "\n", encoding="utf-8")
    np.savez(
        rd_path,
        amplitude=np.ones((1, 3, 4), dtype=np.float32),
        phase=np.zeros((1, 3, 4), dtype=np.float32),
        time_seconds=np.zeros((1,), dtype=np.float32),
        frame_event_indices=np.zeros((1,), dtype=np.int64),
    )
    tr = _record(
        "parent-1:event-0",
        available=True,
        data_path="data/cq08/tr/parent-1.jsonl",
        file_size=tr_path.stat().st_size,
        sha256=sha256_file(tr_path),
        event_index=0,
    )
    rd_frame = {
        **_record(
            "parent-1:event-0",
            available=True,
            data_path="data/cq08/rd/parent-1.npz",
            file_size=rd_path.stat().st_size,
            sha256=sha256_file(rd_path),
            event_index=0,
        ),
        "record_type": "frame",
        "frame_index": 0,
        "source_path": "external/fixture/frame.mat",
        "metadata": {},
    }
    payload = _with_sidecars(tmp_path, [tr], [rd_frame])
    manifest_path = write_manifest(payload, tmp_path, path="manifest.json")
    manifest = load_manifest(manifest_path, tmp_path, verify_files=True)
    descriptor = build_event_cache(tmp_path, manifest, overwrite=False)
    manifest.audit["cache"] = descriptor
    write_manifest(manifest, tmp_path, path=manifest_path, overwrite=True)
    loaded_descriptor, rows = load_event_cache(tmp_path, manifest_path=manifest_path)
    assert loaded_descriptor["record_count"] == 1
    assert rows[0]["sample_id"] == "parent-1:event-0"
    assert loaded_descriptor["schema_version"] == "RADAR-LN3-EVENT-SOURCE-CACHE-V3"
    assert rows[0]["rd_frame_count"] == 1
    assert rows[0]["rd_frame_times"] == []
    dataset = CachedParentSequenceDataset(
        tmp_path,
        manifest_path=manifest_path,
        training_view=True,
        cache_size=1,
    )
    assert len(dataset) == 1
    assert dataset[0].events[0].sample_id == "parent-1:event-0"


def test_manifest_writer_reuses_identical_content_and_rejects_changes(tmp_path: Path) -> None:
    root = tmp_path
    (root / "data/cq08/tr").mkdir(parents=True)
    (root / "data/cq08/rd").mkdir(parents=True)
    (root / "data/cq08/tr/cq08-tr-event-index.jsonl").write_bytes(b"")
    (root / "data/cq08/rd/cq08-rd-frame-index.jsonl").write_bytes(b"")
    target = write_manifest(_empty_manifest(), root, path="manifest.json")
    original = target.read_bytes()
    assert write_manifest(_empty_manifest(), root, path="manifest.json") == target
    assert target.read_bytes() == original
    changed = _empty_manifest()
    changed["dataset"] = "other"
    with pytest.raises(FileExistsError):
        write_manifest(changed, root, path="manifest.json")


def test_data_manifest_hash_ignores_derived_registries() -> None:
    payload = _empty_manifest()
    original = data_manifest_hash(payload)
    payload["manifest_sha256"] = "f" * 64
    payload["product"] = {
        "data_derivatives": {
            "event_csv": {"sha256": "a" * 64},
            "feature_caches": {"l1_train": {"sha256": "b" * 64}},
        }
    }
    payload["audit"]["cache"] = {"sha256": "c" * 64}
    assert data_manifest_hash(payload) == original


def test_unregistered_split_parent_fails_closed(tmp_path: Path) -> None:
    adapter = object.__new__(CQ08Adapter)
    adapter.registered_index = False
    adapter._split_map = {"cq08|track|known": "train"}
    adapter._split_excluded_parent_ids = set()
    with pytest.raises(DataSplitError, match="does not cover discovered parents"):
        adapter.assign_split(["cq08|track|known", "cq08|track|unregistered"])


def test_unified_view_keeps_test_rd_out_of_training_and_cache(tmp_path: Path) -> None:
    train = _record(
        "train-parent:event-0",
        available=False,
        event_index=0,
    )
    train["parent_sample_id"] = "train-parent"
    train["track_id"] = "train-track"
    train["metadata"]["baseline_training_view"] = True

    test = _record(
        "test-parent:event-0",
        available=False,
        event_index=0,
    )
    test["parent_sample_id"] = "test-parent"
    test["track_id"] = "test-track"
    test["split_name"] = "test"
    test["metadata"]["baseline_training_view"] = False

    payload = _with_sidecars(tmp_path, [train, test], [])
    manifest_path = write_manifest(payload, tmp_path, path="manifest.json")
    manifest = load_manifest(manifest_path, tmp_path, verify_files=True)

    view = resolve_manifest_view(manifest, training_view=True)
    assert view.parent_ids == ("train-parent",)
    assert view.sample_ids == ("train-parent:event-0",)
    assert all(
        row["metadata"]["baseline_training_view"] is False
        for row in manifest.rd_records
        if row["split_name"] == "test"
    )

    dataset = ParentSequenceDataset(
        tmp_path,
        manifest_path=manifest_path,
        training_view=True,
        verify_files=True,
    )
    assert dataset.parent_ids == ["train-parent"]

    descriptor = build_event_cache(tmp_path, manifest, overwrite=False)
    assert descriptor["record_count"] == 2


def test_missing_unified_training_view_fails_closed(tmp_path: Path) -> None:
    record = _record(
        "train-parent:event-0",
        available=False,
        event_index=0,
    )
    record["metadata"].pop("baseline_training_view")
    payload = _with_sidecars(tmp_path, [record], [])
    with pytest.raises(DataFormatError, match="baseline_training_view"):
        validate_manifest(payload, tmp_path, verify_files=True)


def test_product_manifest_self_hash_loads() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = load_manifest(root / "src/config/manifests.json", root, verify_files=True)
    assert manifest.manifest_sha256
    assert isinstance(manifest.audit["materialized"], bool)


def test_split_alias_is_normalized_to_validation() -> None:
    from radar_ln3.data.manifest import normalise_split_name
    assert normalise_split_name("val") == "validation"


def test_generic_data_modules_do_not_import_cq08() -> None:
    root = Path(__file__).resolve().parents[1] / "radar_ln3" / "data"
    for module_name in ("index.py", "manifest.py", "dataset.py"):
        tree = ast.parse((root / module_name).read_text(encoding="utf-8"), filename=module_name)
        imported_modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported_modules.append(node.module or "")
        assert all("cq08" not in module.casefold() for module in imported_modules)


def test_cli_registers_raw_data_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["--handle-raw-data", "--project-root", "project", "--source-root", "external/cq08", "--split-path", "split.json", "--manifest-path", "manifest.json", "--parent-limit", "2", "--overwrite"])
    assert args.handle_raw_data is True
    assert args.parent_limit == 2


def test_generic_adapter_can_replace_cq08_adapter(tmp_path: Path) -> None:
    class FixtureAdapter:
        dataset_name = "fixture-dataset"

        def discover_parents(self) -> list[ParentRecord]:
            return [ParentRecord(dataset_name=self.dataset_name, source_name="fixture-source", parent_sample_id="fixture|parent|1", track_id="1", label_id=0, label_name="DroneTarget", source_path="external/fixture/track.txt", data_path="", split_name="val")]

        def discover_events(self, parent: ParentRecord):
            from radar_ln3.data.contracts import EventRecord
            return [EventRecord(dataset_name=self.dataset_name, source_name="fixture-source", sample_id=f"{parent.parent_sample_id}:event-0", parent_sample_id=parent.parent_sample_id, track_id=parent.track_id, causal_event_index=0, label_id=parent.label_id, label_name=parent.label_name, split_name=parent.split_name, available=True, source_path=parent.source_path, data_path="")]

        def assign_split(self, parent_ids: list[str]) -> dict[str, str]:
            return {parent_id: "val" for parent_id in parent_ids}

    manifest = build_manifest(FixtureAdapter(), tmp_path, materialize=False)
    assert manifest.dataset == "fixture-dataset"
    assert manifest.tr_records[0]["split_name"] == "validation"
    assert manifest.pair_records[0]["sample_id"] == "fixture|parent|1:event-0"


def test_excluded_parent_is_audit_only(tmp_path: Path) -> None:
    class FixtureAdapter:
        dataset_name = "fixture-dataset"

        def discover_parents(self) -> list[ParentRecord]:
            return [
                ParentRecord(
                    dataset_name=self.dataset_name,
                    source_name="fixture-source",
                    parent_sample_id="fixture|parent|1",
                    track_id="1",
                    label_id=0,
                    label_name="DroneTarget",
                    source_path="external/fixture/track.txt",
                    data_path="",
                    split_name="train",
                )
            ]

        def discover_excluded_parents(self) -> list[ParentRecord]:
            return [
                ParentRecord(
                    dataset_name=self.dataset_name,
                    source_name="fixture-source",
                    parent_sample_id="fixture|parent|excluded",
                    track_id="excluded",
                    label_id=0,
                    label_name="DroneTarget",
                    source_path="external/fixture/excluded.txt",
                    data_path="",
                    available=False,
                    metadata={"excluded": True, "exclusion_reasons": ["missing_rd_mat"]},
                )
            ]

        def discover_events(self, parent: ParentRecord):
            from radar_ln3.data.contracts import EventRecord

            return [
                EventRecord(
                    dataset_name=self.dataset_name,
                    source_name="fixture-source",
                    sample_id=f"{parent.parent_sample_id}:event-0",
                    parent_sample_id=parent.parent_sample_id,
                    track_id=parent.track_id,
                    causal_event_index=0,
                    label_id=parent.label_id,
                    label_name=parent.label_name,
                    split_name=parent.split_name,
                    available=True,
                    source_path=parent.source_path,
                    data_path="",
                )
            ]

        def assign_split(self, parent_ids: list[str]) -> dict[str, str]:
            return {parent_id: "train" for parent_id in parent_ids}

    manifest = build_manifest(FixtureAdapter(), tmp_path, materialize=False)
    assert manifest.audit["excluded_parent_count"] == 1
    assert manifest.audit["excluded_parents"][0]["parent_sample_id"] == "fixture|parent|excluded"
    assert all("excluded" not in str(row["sample_id"]) for row in manifest.tr_records)


def test_registered_cq08_adapter_does_not_require_raw_source_tree(tmp_path: Path) -> None:
    source_dir = tmp_path / "src/config"
    source_dir.mkdir(parents=True)
    (source_dir / "source.json").write_text(
        json.dumps(
            {
                "schema_version": "RADAR-LN3-CQ08-SOURCE-CONFIG-V1",
                "dataset": "CQ08",
                "source_roots": {
                    "cq08": {
                        "path": "missing-source",
                        "logical_prefixes": ["missing-source", "external/cq08"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    manifest = _empty_manifest()
    manifest["dataset"] = "CQ08"
    tr_path = tmp_path / "data/cq08/tr/cq08-tr-event-index.jsonl"
    rd_path = tmp_path / "data/cq08/rd/cq08-rd-frame-index.jsonl"
    tr_path.parent.mkdir(parents=True)
    rd_path.parent.mkdir(parents=True)
    tr_row = _record(
        "cq08|track|1:event-0",
        available=True,
        data_path="data/cq08/tr/cq08__track__1.jsonl",
        event_index=0,
    )
    tr_row["parent_sample_id"] = "cq08|track|1"
    tr_row["track_id"] = "1"
    tr_row["metadata"] = {
        "baseline_training_view": True,
        "tr_timing": {
                "time_seconds": 1.0,
                "dt_seconds": 0.0,
                "dt_valid": False,
                "dt_reason": "first_event_no_predecessor",
                "schema_version": "RADAR-LN3-TR-TIME-V1",
                "time_order_status": "monotonic",
            "dt_is_negative": False,
        },
    }
    tr_data_path = tmp_path / "data/cq08/tr/cq08__track__1.jsonl"
    tr_data_path.write_text(
        json.dumps(
            {
                "sample_id": tr_row["sample_id"],
                "parent_sample_id": tr_row["parent_sample_id"],
                "track_id": "1",
                "causal_event_index": 0,
                "label_id": 0,
                "label_name": "DroneTarget",
                "split_name": "train",
                "values": {"range_m": 1.0, "time_seconds": 1.0},
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tr_row["file_size"] = tr_data_path.stat().st_size
    tr_row["sha256"] = sha256_file(tr_data_path)
    _write_fixture_sidecars(tmp_path, [tr_row], [])
    manifest["TR"] = {
        **manifest["TR"],
        "record_count": 1,
        "parts": [_descriptor("data/cq08/tr/cq08-tr-event-index.jsonl", count=1, size=tr_path.stat().st_size, sha256=sha256_file(tr_path))],
    }
    manifest["pairing"] = {**manifest["pairing"], "paired_event_count": 1, "missing_rd_event_count": 1}
    manifest_path = tmp_path / "src/config/manifests.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    # 注册模式只依赖产品索引；原始 source root 和 MAT 目录均不存在。
    adapter = CQ08Adapter(tmp_path, registered_index=True)
    parents = adapter.discover_parents()
    assert [parent.parent_sample_id for parent in parents] == ["cq08|track|1"]
    events = adapter.discover_events(parents[0])
    assert events[0].sample_id == "cq08|track|1:event-0"
