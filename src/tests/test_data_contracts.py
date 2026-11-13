from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from core.cli import build_parser
from radar_ln3.data.contracts import (
    DataPairingError,
    DataFormatError,
    DataIntegrityError,
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
    normalise_progress_interval,
    resolve_manifest_view,
    validate_manifest,
)
from radar_ln3.data.paths import (
    resolve_data_path,
    resolve_project_artifact_path,
    resolve_project_path,
    resolve_rd_path,
    resolve_source_path,
    resolve_tr_path,
)
from radar_ln3.data.raw_reader import sha256_file
from radar_ln3.cache import CachedParentSequenceDataset, build_event_cache, load_event_cache
from radar_ln3.cache.contracts import CacheIntegrityError


EMPTY_HASH = hashlib.sha256(b"").hexdigest()


def test_sha256_file_does_not_reuse_process_cache_after_same_metadata_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"first")
    first = sha256_file(path)
    original_mtime = path.stat().st_mtime_ns
    path.write_bytes(b"other")
    os.utime(path, ns=(original_mtime, original_mtime))

    assert path.stat().st_size == 5
    assert sha256_file(path) != first
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


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


def test_experiment_artifact_paths_are_separate_from_data_paths(tmp_path: Path) -> None:
    with pytest.raises(PathBoundaryError):
        resolve_project_path(tmp_path, "experiments/tmp/L2/representations/l2-train.pt")

    target = resolve_project_artifact_path(
        tmp_path, "experiments/tmp/L2/representations/l2-train.pt"
    )
    assert target == tmp_path / "experiments/tmp/L2/representations/l2-train.pt"

    with pytest.raises(PathBoundaryError):
        resolve_project_artifact_path(tmp_path, "../outside.pt")
    with pytest.raises(PathBoundaryError):
        resolve_project_artifact_path(tmp_path, "src-v3/old-model.pt")


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
    assert loaded_descriptor["schema_version"] == "RADAR-LN3-EVENT-SOURCE-CACHE-V4"
    assert rows[0]["rd_frame_indices"] == [0]
    assert set(rows[0]) == {
        "sample_id", "parent_sample_id", "track_id", "causal_event_index",
        "label_id", "label_name", "split_name", "baseline_training_view",
        "tr_available", "tr_time_seconds", "tr_dt_seconds", "tr_dt_valid",
        "tr_dt_reason", "parent_anomalous", "training_excluded",
        "tr_dt_is_negative", "tr_time_order_status", "rd_available",
        "rd_frame_indices", "rd_time_semantics",
    }
    assert loaded_descriptor["files"] == [
        {
            "section": "RD",
            "path": "data/cq08/rd/parent-1.npz",
            "file_size": rd_path.stat().st_size,
            "sha256": sha256_file(rd_path),
        },
        {
            "section": "TR",
            "path": "data/cq08/tr/parent-1.jsonl",
            "file_size": tr_path.stat().st_size,
            "sha256": sha256_file(tr_path),
        },
    ]
    dataset = CachedParentSequenceDataset(
        tmp_path,
        manifest_path=manifest_path,
        training_view=True,
        cache_size=1,
    )
    assert len(dataset) == 1
    assert dataset[0].events[0].sample_id == "parent-1:event-0"
    assert not hasattr(dataset, "cache_rows")
    assert not hasattr(dataset, "cache_descriptor")
    assert not hasattr(dataset, "_positions")


def test_canonical_rd_event_index_is_compact_and_complete(
    tmp_path: Path, monkeypatch
) -> None:
    import radar_ln3.data.canonical_rd as canonical_rd

    tr = _record(
        "parent-1:event-0",
        available=True,
        data_path="data/cq08/tr/parent-1.jsonl",
        event_index=0,
    )
    tr["track_id"] = "1"
    rd_frame = {
        **_record(
            "parent-1:event-0",
            available=True,
            data_path="data/cq08/rd/parent-1.npz",
            event_index=0,
        ),
        "record_type": "frame",
        "frame_index": 0,
        "source_path": "external/fixture/frame.mat",
        "metadata": {},
    }
    rd_frame["track_id"] = "1"
    payload = _with_sidecars(tmp_path, [tr], [rd_frame])
    tensor_path = tmp_path / "data/cq08/rd/parent-1.npz"
    tensor_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tensor_path, amplitude=np.zeros((2, 21, 384), dtype=np.float32))
    tr_path = tmp_path / "data/cq08/tr/parent-1.jsonl"
    tr_path.parent.mkdir(parents=True, exist_ok=True)
    tr_path.write_text("", encoding="utf-8")
    manifest_path = write_manifest(payload, tmp_path, path="manifest.json")

    frame_index = tmp_path / "data/processed/canonical-frames.csv"
    frame_index.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "schema_version,parent_sample_id,track_id,source_name,label_name,"
        "raw_frame_index,tensor_frame_index,is_tensor_frame_kept,"
        "track_point_index,rd_mat_path,rd_tensor_path,canonical_rd_shape\n"
    )
    rows = "".join(
        "cq08_rd_physical_canonical_v3_1,parent-1,1,fixture-source,DroneTarget,"
        f"{ordinal},{ordinal},1,1,data/cq08/raw/frame_{ordinal + 5}_1.mat,"
        "data/cq08/rd/parent-1.npz,21x384\n"
        for ordinal in (0, 1)
    )
    frame_index.write_text(header + rows, encoding="utf-8")

    def fake_load_tensor(path: Path) -> dict[str, np.ndarray]:
        return {
            "amplitude": np.zeros((2, 21, 384), dtype=np.float32),
            "phase": np.zeros((2, 21, 384), dtype=np.float32),
            "time_raw": np.zeros((2,), dtype=np.float32),
            "time_seconds": np.arange(2, dtype=np.float32),
            "velocity_mps": np.arange(384, dtype=np.float32),
            "range_offset_m": np.zeros((21,), dtype=np.float32),
        }

    # 速度轴 hash 是固定产品常量，fixture 无法合成，因此只替换 tensor 读取；
    # 索引写入、manifest 封 hash 与 store 读取都走真实代码。
    monkeypatch.setattr(canonical_rd, "_load_tensor", fake_load_tensor)
    result = canonical_rd.materialize_canonical_rd_store(
        tmp_path,
        manifest_path=manifest_path,
        source_frame_index_path="data/processed/canonical-frames.csv",
        verify_files=False,
    )
    assert result["event_count"] == 1

    event_index = tmp_path / "data/cq08/rd/canonical/canonical-rd-event-index.jsonl"
    line = event_index.read_text(encoding="utf-8").splitlines()[0]
    row = json.loads(line)
    assert set(row) == {
        "sample_id", "parent_sample_id", "track_id", "causal_event_index",
        "track_point_index", "label_name", "split_name", "available",
        "data_path", "tensor_frame_indices", "mat_frame_numbers",
    }
    assert row["tensor_frame_indices"] == [0, 1]
    assert row["mat_frame_numbers"] == [5, 6]
    assert ", " not in line

    store = canonical_rd.CanonicalRDStore(tmp_path)
    event = store.event("parent-1", 0)
    assert event["available"] is True
    values = event["rd_values"]
    assert values["frame_count"] == 2
    assert "raw_dat_frame_indices" not in values
    np.testing.assert_array_equal(values["mat_frame_numbers"], [5, 6])


def test_validation_dataset_rejects_event_cache_missing_its_view(monkeypatch, tmp_path: Path) -> None:
    from types import SimpleNamespace
    import radar_ln3.cache.registry as registry

    class BaseDataset:
        def __init__(self, *args, **kwargs):
            self.manifest = kwargs["manifest"]
            self.parent_ids = ["validation-parent"]
            self.view = SimpleNamespace(
                sample_ids=("validation-parent:event-0",),
                split_name="validation",
            )

    monkeypatch.setattr(registry, "ParentSequenceDataset", BaseDataset)
    monkeypatch.setattr(
        registry,
        "load_event_cache",
        lambda *args, **kwargs: (
            {"record_count": 1},
            [{"parent_sample_id": "train-parent", "sample_id": "train-parent:event-0"}],
        ),
    )
    dataset = CachedParentSequenceDataset(
        tmp_path,
        manifest=SimpleNamespace(),
        split_name="validation",
        validate_event_cache=False,
    )

    with pytest.raises(CacheIntegrityError, match="事件缓存缺少 parent"):
        dataset.validate_event_cache()
    assert not dataset.event_cache_validated


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
    payload["audit"]["scope"] = {"l3": "formal_available"}
    payload["product"]["base_weights"] = {"l1": {"sha256": "d" * 64}}
    assert data_manifest_hash(payload) == original


@pytest.mark.parametrize(
    "descriptor_update",
    [
        {"sha256": "invalid"},
        {"sha256": "0" * 64},
        {"file_size": 999},
    ],
)
def test_register_product_rejects_invalid_l3_calibration_descriptor(
    tmp_path: Path, monkeypatch, descriptor_update: dict[str, object]
) -> None:
    import radar_ln3.data.derived as derived

    payload = _empty_manifest()
    calibration = tmp_path / "calibration.json"
    calibration.write_bytes(b"calibration")
    manifest_hash = data_manifest_hash(payload)
    descriptor = {
        "kind": "l3_calibration",
        "path": "calibration.json",
        "file_size": calibration.stat().st_size,
        "sha256": hashlib.sha256(calibration.read_bytes()).hexdigest(),
        "status": "exploratory",
    }
    descriptor.update(descriptor_update)
    monkeypatch.setattr(derived, "_load_manifest_payload", lambda *args: payload)
    monkeypatch.setattr(derived, "validate_manifest", lambda *args, **kwargs: payload)

    with pytest.raises(DataFormatError, match="L3 calibration artifact descriptor"):
        derived.register_product_artifacts(
            tmp_path,
            manifest_path="manifest.json",
            csv_descriptor={"source_manifest_sha256": manifest_hash},
            l3_calibration_descriptor=descriptor,
        )


def test_register_product_drops_audit_scope_and_base_weights(
    tmp_path: Path, monkeypatch
) -> None:
    import radar_ln3.data.derived as derived

    payload = _empty_manifest()
    payload["audit"]["scope"] = {"l3": "formal_available"}
    payload["product"] = {"base_weights": {"l1": {"path": "base/l1.pt"}}}
    expected_hash = data_manifest_hash(payload)
    monkeypatch.setattr(derived, "_load_manifest_payload", lambda *args: payload)
    monkeypatch.setattr(
        derived, "validate_manifest", lambda *args, **kwargs: payload
    )
    monkeypatch.setattr(
        derived, "_file_descriptor", lambda *args, **kwargs: {"path": "fixture"}
    )

    target = derived.register_product_artifacts(
        tmp_path,
        manifest_path="manifest.json",
        csv_descriptor={"source_manifest_sha256": expected_hash},
        overwrite=True,
    )

    written = json.loads(target.read_text(encoding="utf-8"))
    assert "scope" not in written["audit"]
    assert written["product"]["scope"]["l3"] == "not_in_scope"
    assert "base_weights" not in written["product"]
    assert data_manifest_hash(written) == expected_hash
    body = {
        key: value for key, value in written.items() if key != "manifest_sha256"
    }
    assert written["manifest_sha256"] == canonical_json_hash(body)


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
    assert manifest.tr_records[0]["sample_id"] == "fixture|parent|1:event-0"
    assert not hasattr(manifest, "pair_records")
    assert manifest.pairing["paired_event_count"] == 1
    assert manifest.pairing["missing_rd_event_count"] == 1


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


def test_path_resolution_cache_reuses_once_per_path(tmp_path: Path, monkeypatch) -> None:
    from radar_ln3.data import paths

    calls: list[str] = []
    original = Path.resolve

    def counting(self, strict: bool = False):
        calls.append(str(self))
        return original(self, strict)

    monkeypatch.setattr(Path, "resolve", counting)
    with paths.path_resolution_cache():
        first = paths.resolve_project_path(tmp_path, "data/cq08/tr/parent-1.jsonl")
        second = paths.resolve_project_path(tmp_path, "data/cq08/tr/parent-1.jsonl")

    assert first == second
    assert len(calls) == len(set(calls))
    assert len(calls) < 20

    outside = len(calls)
    paths.resolve_project_path(tmp_path, "data/cq08/tr/parent-1.jsonl")
    assert len(calls) > outside


def _paired_event_records(index: int) -> tuple[dict[str, object], dict[str, object]]:
    sample_id = f"parent-1:event-{index}"
    tr = _record(
        sample_id,
        available=True,
        data_path="data/cq08/tr/parent-1.jsonl",
        event_index=index,
    )
    rd = {
        **_record(
            sample_id,
            available=True,
            data_path="data/cq08/rd/parent-1.npz",
            event_index=index,
        ),
        "record_type": "frame",
        "frame_index": 0,
        "source_path": "external/fixture/frame.mat",
        "metadata": {},
    }
    return tr, rd


def test_validate_manifest_resolves_each_path_once(tmp_path: Path, monkeypatch) -> None:
    tr_records, rd_frames = zip(
        *(_paired_event_records(index) for index in range(3))
    )
    payload = _with_sidecars(tmp_path, list(tr_records), list(rd_frames))

    calls: list[str] = []
    original = Path.resolve

    def counting(self, strict: bool = False):
        calls.append(str(self))
        return original(self, strict)

    monkeypatch.setattr(Path, "resolve", counting)
    validate_manifest(payload, tmp_path, verify_files=False)

    assert len(calls) == len(set(calls))
    assert len(calls) < 60


def test_resolve_path_parent_cache_matches_full_resolve(tmp_path: Path) -> None:
    from radar_ln3.data import paths

    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    existing = nested / "c.txt"
    existing.write_text("payload", encoding="utf-8")

    with paths.path_resolution_cache():
        assert paths.resolve_path(existing) == existing.resolve()
        assert paths.resolve_path(nested) == nested.resolve()
        missing = nested / "missing" / "d.txt"
        assert paths.resolve_path(missing) == missing.resolve()


def test_resolve_path_falls_back_for_symlinked_leaf(tmp_path: Path) -> None:
    from radar_ln3.data import paths

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    target = outside / "payload.txt"
    target.write_text("payload", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("当前环境不允许创建符号链接")

    with paths.path_resolution_cache():
        assert paths.resolve_path(link) == target.resolve()
    with pytest.raises(PathBoundaryError):
        paths.resolve_project_path(tmp_path, "link.txt")


def test_validate_manifest_deduplicates_shared_paths(tmp_path: Path, monkeypatch) -> None:
    import radar_ln3.data.manifest as manifest_module

    tr = _record(
        "parent-1:event-0",
        available=True,
        data_path="data/cq08/tr/parent-1.jsonl",
        event_index=0,
    )
    tr["source_path"] = "raw/event.txt"
    frame_zero = {
        **_record(
            "parent-1:event-0",
            available=True,
            data_path="data/cq08/rd/parent-1.npz",
            event_index=0,
        ),
        "record_type": "frame",
        "frame_index": 0,
        "source_path": "raw/frame-0.mat",
        "metadata": {},
    }
    frame_one = {
        **frame_zero,
        "frame_index": 1,
        "source_path": "raw/frame-1.mat",
    }
    payload = _with_sidecars(tmp_path, [tr], [frame_zero, frame_one])

    project_calls: list[str] = []
    tr_path_calls: list[str] = []
    rd_path_calls: list[str] = []
    original_project = manifest_module.resolve_project_path
    original_tr = manifest_module.resolve_tr_path
    original_rd = manifest_module.resolve_rd_path

    def count_project(project_root, value, **kwargs):
        project_calls.append(str(value))
        return original_project(project_root, value, **kwargs)

    def count_tr(project_root, value, **kwargs):
        tr_path_calls.append(str(value))
        return original_tr(project_root, value, **kwargs)

    def count_rd(project_root, value, **kwargs):
        rd_path_calls.append(str(value))
        return original_rd(project_root, value, **kwargs)

    monkeypatch.setattr(manifest_module, "resolve_project_path", count_project)
    monkeypatch.setattr(manifest_module, "resolve_tr_path", count_tr)
    monkeypatch.setattr(manifest_module, "resolve_rd_path", count_rd)

    validate_manifest(payload, tmp_path, verify_files=False)

    # TR 来源 1 个 + RD 帧来源 2 个；聚合 RD source_path 不再二次校验。
    assert sorted(project_calls) == ["raw/event.txt", "raw/frame-0.mat", "raw/frame-1.mat"]
    # 共享 data_path 只解析一次。
    assert tr_path_calls == ["data/cq08/tr/parent-1.jsonl"]
    assert rd_path_calls == ["data/cq08/rd/parent-1.npz"]


def test_manifest_repeated_path_still_checks_declared_hash(tmp_path: Path) -> None:
    tr_path = tmp_path / "data/cq08/tr/parent-1.jsonl"
    tr_path.parent.mkdir(parents=True)
    tr_path.write_text("payload", encoding="utf-8")
    first = _record(
        "parent-1:event-0",
        available=True,
        data_path="data/cq08/tr/parent-1.jsonl",
        file_size=tr_path.stat().st_size,
        sha256=sha256_file(tr_path),
        event_index=0,
    )
    second = _record(
        "parent-1:event-1",
        available=True,
        data_path="data/cq08/tr/parent-1.jsonl",
        file_size=tr_path.stat().st_size,
        sha256="0" * 64,
        event_index=1,
    )
    payload = _with_sidecars(tmp_path, [first, second], [])

    with pytest.raises(DataIntegrityError, match="hash 不一致"):
        validate_manifest(payload, tmp_path, verify_files=True)


def test_write_manifest_can_skip_revalidation(tmp_path: Path, monkeypatch) -> None:
    import radar_ln3.data.index as index_module

    payload = _empty_manifest()
    validate_calls = []
    monkeypatch.setattr(
        index_module,
        "validate_manifest",
        lambda *args, **kwargs: validate_calls.append(1) or payload,
    )

    target = write_manifest(payload, tmp_path, path="manifest.json", validate=False)

    assert validate_calls == []
    written = json.loads(target.read_text(encoding="utf-8"))
    body = {key: value for key, value in written.items() if key != "manifest_sha256"}
    assert written["manifest_sha256"] == canonical_json_hash(body)


def test_build_training_cache_reuses_provided_manifest(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    import core.training.cache as training_cache

    manifest = SimpleNamespace(audit={})
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        training_cache,
        "load_manifest",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("必须复用调用方 manifest")
        ),
    )
    monkeypatch.setattr(
        training_cache,
        "build_event_cache",
        lambda root, provided, **kwargs: {"path": "cache"},
    )
    monkeypatch.setattr(
        training_cache,
        "write_manifest",
        lambda provided, root, **kwargs: captured.update(kwargs) or tmp_path / "manifest.json",
    )

    result = training_cache.build_training_cache(
        tmp_path,
        manifest=manifest,
        verify_files=False,
    )

    assert result == {"manifest_path": str(tmp_path / "manifest.json"), "cache": {"path": "cache"}}
    assert captured["validate"] is False
    assert manifest.audit["cache"] == {"path": "cache"}


def test_handle_raw_data_validates_once_and_reuses_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    import core.cli as cli_module

    manifest = SimpleNamespace(
        audit={},
        tr_index={},
        rd_index={},
        tr_records=[],
    )
    target = tmp_path / "manifest.json"
    target.write_text(
        json.dumps({"manifest_sha256": canonical_json_hash({})}),
        encoding="utf-8",
    )
    calls = {"write": [], "cache": []}
    monkeypatch.setattr(cli_module, "CQ08Adapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli_module, "build_manifest", lambda *args, **kwargs: manifest)
    monkeypatch.setattr(
        cli_module,
        "write_manifest",
        lambda provided, root, **kwargs: calls["write"].append(kwargs) or target,
    )
    monkeypatch.setattr(
        cli_module,
        "build_training_cache",
        lambda root, **kwargs: calls["cache"].append(kwargs)
        or {"manifest_path": str(target), "cache": {"path": "cache"}},
    )
    monkeypatch.setattr(cli_module, "configure_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli_module,
        "get_logger",
        lambda *args, **kwargs: SimpleNamespace(info=lambda *a, **k: None),
    )

    args = SimpleNamespace(
        project_root=tmp_path,
        source_root=None,
        split_path=None,
        parent_limit=None,
        overwrite=False,
        manifest_path=Path("manifest.json"),
        cache_path=Path("data/cq08/cache/event-source-cache.jsonl"),
    )
    result = cli_module.handle_raw_data(args)

    assert result == target
    assert calls["write"][0]["validate"] is False
    assert calls["cache"][-1]["manifest"] is manifest
    assert calls["cache"][-1]["verify_files"] is False
    assert manifest.audit["cache"] == {"path": "cache"}


def test_load_task_feature_caches_reuses_prebuilt_datasets(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    import core.training.cache as training_cache

    manifest = SimpleNamespace()
    datasets = {
        "train": SimpleNamespace(view=SimpleNamespace(view_name="train")),
        "validation": SimpleNamespace(view=SimpleNamespace(view_name="validation")),
    }
    monkeypatch.setattr(
        training_cache,
        "build_training_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("不应重建 dataset")
        ),
    )
    monkeypatch.setattr(
        training_cache,
        "load_feature_cache_bundle",
        lambda *args, **kwargs: {},
    )

    train_cache, validation_cache = training_cache.load_task_feature_caches(
        tmp_path,
        task="l1",
        manifest=manifest,
        datasets=datasets,
        validate_cache=False,
    )

    assert train_cache == {}
    assert validation_cache == {}


def test_ensure_l2_baseline_anchor_reuses_without_building_dataset(
    tmp_path: Path, monkeypatch
) -> None:
    from types import SimpleNamespace

    from radar_ln3.data.contracts import DatasetManifest
    from radar_ln3.training import l2_anchor

    tr_record = {
        "sample_id": "p:event-0",
        "parent_sample_id": "p",
        "track_id": "p",
        "causal_event_index": 0,
        "label_id": 0,
        "label_name": "DroneTarget",
        "split_name": "train",
        "metadata": {"baseline_training_view": True},
    }
    manifest = DatasetManifest(
        schema_version="x",
        dataset="CQ08",
        data_root="data/cq08",
        source_policy={},
        split={},
        tr_records=[tr_record],
        rd_records=[],
    )
    existing = SimpleNamespace(
        payload={"source_manifest_hash": data_manifest_hash(manifest)},
        sample_ids=("p:event-0",),
        parent_ids=("p",),
    )
    monkeypatch.setattr(l2_anchor, "load_model_config", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(l2_anchor, "_load_existing", lambda *args, **kwargs: existing)
    monkeypatch.setattr(
        l2_anchor,
        "CachedParentSequenceDataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("复用路径不应构造 dataset")
        ),
    )

    result = l2_anchor.ensure_l2_baseline_anchor(
        tmp_path,
        manifest=manifest,
        verify_files=False,
    )

    assert result["status"] == "reused"
    assert result["record_count"] == 1


def test_asd_evidence_metadata_omits_stable8_aliases() -> None:
    from radar_ln3.data.asd import ASD_EVIDENCE_SCHEMA, _evidence_metadata

    frame_mask = np.ones((12,), dtype=np.bool_)
    value = {
        "schema_version": ASD_EVIDENCE_SCHEMA,
        "frame_cap": 12,
        "event_index": 0,
        "stable8_values": np.full((8,), 0.5, dtype=np.float32),
        "stable8_valid": True,
        "stable8_frame_values": np.ones((12, 8), dtype=np.float32),
        "stable8_frame_mask": frame_mask,
        "micro_doppler_frame_values": np.ones((12, 23), dtype=np.float32),
        "micro_doppler_frame_mask": frame_mask,
        "micro_doppler_reliable": frame_mask,
        "raw_frame_indices": np.arange(12, dtype=np.int32),
    }

    metadata = _evidence_metadata(value, event_index=0, rd_frame_count=12)

    assert "micro_doppler_values" not in metadata
    assert "micro_doppler_valid" not in metadata
    assert metadata["stable8_values"].shape == (8,)
    assert metadata["stable8_frame_values"].shape == (12, 8)


def test_resolve_frame_source_path_routes_external_frames(monkeypatch, tmp_path) -> None:
    from radar_ln3.data import cq08

    calls: list[tuple[str, str, dict[str, object]]] = []
    source_root = tmp_path / "raw"
    resolved = tmp_path / "resolved.mat"

    def fake_source(project_root, raw_path, **options):
        calls.append(("source", raw_path, options))
        return resolved

    def fake_registered(project_root, raw_path, source_config, **options):
        calls.append(("registered", raw_path, options))
        return resolved

    monkeypatch.setattr(cq08, "resolve_source_path", fake_source)
    monkeypatch.setattr(cq08, "resolve_registered_source_path", fake_registered)

    external = cq08.resolve_frame_source_path(
        tmp_path,
        "external\\cq08\\frame.mat",
        source_root=source_root,
        source_config={"name": "cq08"},
    )
    internal = cq08.resolve_frame_source_path(
        tmp_path,
        "data\\cq08\\frame.mat",
        source_root=source_root,
        source_config={"name": "cq08"},
    )

    assert external == resolved
    assert internal == resolved
    assert calls[0][0] == "source"
    assert calls[0][2]["source_roots"] == {"cq08": source_root}
    assert calls[0][2]["must_exist"] is True
    assert calls[0][2]["allow_external"] is True
    assert calls[1][0] == "registered"
    assert calls[1][2] == {"must_exist": True}


def test_progress_interval_validator_has_one_implementation() -> None:
    from radar_ln3.cache import registry as cache_registry
    from radar_ln3.data import index as data_index

    assert data_index.normalise_progress_interval is normalise_progress_interval
    assert cache_registry.normalise_progress_interval is normalise_progress_interval

    assert normalise_progress_interval(7) == 7
    for invalid in (True, 0, -1, "5", 1.5, None):
        with pytest.raises(ValueError, match="progress_interval 必须是正整数"):
            normalise_progress_interval(invalid)
