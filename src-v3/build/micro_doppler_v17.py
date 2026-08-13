from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from radar_ln3.micro_doppler import (
    MICRO_DOPPLER_MASK_MODE,
    MICRO_DOPPLER_NAMES,
    MICRO_DOPPLER_SCHEMA,
    MICRO_DOPPLER_VALUE_DIM,
    STFT_FRAME_CAP,
    STFT_TIME_BIN_COUNT,
    STFT_VELOCITY_BIN_COUNT,
    extract_current_event_micro_doppler,
    micro_doppler_contract,
    parse_cq08_mat_frame_identity,
    parse_cq08_raw_echo,
)
from radar_ln3.rd_physical import (
    RD_PHYSICAL_MASK_MODE,
    RD_PHYSICAL_NAMES,
    RD_PHYSICAL_SCHEMA,
    RD_PHYSICAL_VALUE_DIM,
    extract_current_event_rd_physical,
    rd_physical_contract,
)


VERSION = "RADAR-LN3-V17"
SOURCE_VERSION = "RADAR-LN3-V16"
PAIRED_PROTOCOL = (
    "v17-dbb-rd-physical4-control-versus-plus-target-anchored-micro-doppler3-treatment-v3"
)
DBB_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
SPLITS = ("train", "val")
SEED = 42

SOURCE_INDEX = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
SOURCE_CONTRACT = Path("data/processed/RADAR-LN3-V16-cq08-product-local-data-contract.json")
SOURCE_PROVENANCE = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json")
SOURCE_STATS = Path("data/processed/RADAR-LN3-V16-cq08-tr-rd-train-normalization-stats.json")
SOURCE_CANONICAL_FRAME_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)
SOURCE_DBB_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V16-cq08-liquid-product-local-dbb-specialist-config.json"
)

DATA_CONTRACT = Path(
    "data/processed/RADAR-LN3-V17-cq08-dbb-target-anchored-v3-micro-doppler-data-contract.json"
)
CACHE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V17-cq08-dbb-target-anchored-v3-micro-doppler-cache-manifest.json"
)


def _summary_cache(split: str) -> Path:
    return Path(
        f"data/processed/RADAR-LN3-V17-cq08-dbb-{split}-"
        "target-anchored-v3-micro-doppler-summary-cache.npz"
    )


def _stft_cache(split: str) -> Path:
    return Path(
        f"data/processed/RADAR-LN3-V17-cq08-dbb-{split}-"
        "target-anchored-v3-micro-doppler-stft-cache.npz"
    )


def _physical_cache(split: str) -> Path:
    return Path(f"data/processed/RADAR-LN3-V17-cq08-dbb-{split}-rd-physical4-cache.npz")


def _config_path(arm: str) -> Path:
    return Path(
        f"config/RADAR-LN3/RADAR-LN3-V17-cq08-liquid-dbb-specialist-"
        f"rd-physical4-micro-doppler-target-anchored-v3-{arm}-seed42-config.json"
    )


def _artifact_dir(arm: str) -> str:
    return (
        "artifacts/RADAR-LN3-V17-cq08-liquid-dbb-specialist-"
        f"rd-physical4-micro-doppler-target-anchored-v3-{arm}-seed42-formal"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2) + "\n"


def _write_json_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    content = _json_text(payload)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite a different versioned file: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return "created"


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _parse_indices(value: Any) -> list[int]:
    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("frame-index field must decode to a list")
    values = [int(item) for item in payload]
    if values != sorted(values) or len(values) != len(set(values)):
        raise ValueError("frame indices must be sorted and unique")
    return values


def _normalize_windows_acl(path: Path) -> None:
    if os.name != "nt":
        return
    completed = subprocess.run(
        ["icacls", str(path), "/reset"],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise PermissionError(f"failed to adopt parent ACL for V17 cache {path}: {detail}")


def _same_npz(path: Path, arrays: Sequence[tuple[str, np.ndarray]]) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            if set(payload.files) != {name for name, _ in arrays}:
                return False
            return all(np.array_equal(np.asarray(payload[name]), array) for name, array in arrays)
    except (OSError, ValueError):
        return False


def _write_npz_new_or_identical(
    path: Path,
    arrays: Sequence[tuple[str, np.ndarray]],
) -> str:
    arrays = tuple((str(name), np.asarray(value)) for name, value in arrays)
    if path.exists():
        if not _same_npz(path, arrays):
            raise FileExistsError(f"refusing to overwrite a different versioned cache: {path}")
        _normalize_windows_acl(path)
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    try:
        with path.open("xb") as target:
            created = True
            with zipfile.ZipFile(
                target, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:
                for name, array in arrays:
                    import io

                    stream = io.BytesIO()
                    np.lib.format.write_array(stream, array, allow_pickle=False)
                    info = zipfile.ZipInfo(
                        f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o600 << 16
                    archive.writestr(
                        info,
                        stream.getvalue(),
                        compress_type=zipfile.ZIP_DEFLATED,
                        compresslevel=9,
                    )
            target.flush()
            os.fsync(target.fileno())
        _normalize_windows_acl(path)
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise
    if not _same_npz(path, arrays):
        path.unlink(missing_ok=True)
        raise ValueError(f"new V17 cache failed its immediate round-trip check: {path}")
    return "created"


class _CanonicalRD:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.path: Path | None = None
        self.amplitude: np.ndarray | None = None
        self.velocity: np.ndarray | None = None
        self.accessed_paths: set[str] = set()

    def load(self, value: str) -> tuple[np.ndarray, np.ndarray]:
        path = _resolve(self.root, value)
        if path == self.path and self.amplitude is not None and self.velocity is not None:
            return self.amplitude, self.velocity
        with np.load(path, allow_pickle=False) as payload:
            required = {"amplitude", "velocity_mps"}
            if not required.issubset(payload.files):
                raise ValueError(f"canonical RD tensor lacks {sorted(required)}: {path}")
            amplitude = np.asarray(payload["amplitude"])
            velocity = np.asarray(payload["velocity_mps"])
        if amplitude.ndim != 3 or amplitude.shape[1:] != (21, 384):
            raise ValueError(f"canonical RD tensor has invalid amplitude shape: {path}")
        if not np.isfinite(amplitude).all() or not np.isfinite(velocity).all():
            raise ValueError(f"canonical RD tensor contains a non-finite value: {path}")
        self.path = path
        self.amplitude = amplitude
        self.velocity = velocity
        self.accessed_paths.add(str(path))
        return amplitude, velocity


class _CanonicalFrameProvenance:
    """Resolve canonical tensor frames to their real zero-based raw-DAT frames."""

    REQUIRED_COLUMNS = (
        "schema_version",
        "parent_sample_id",
        "track_id",
        "source_name",
        "raw_frame_index",
        "tensor_frame_index",
        "is_tensor_frame_kept",
        "track_point_index",
        "rd_mat_path",
        "rd_tensor_path",
    )

    def __init__(self, root: Path, path: Path) -> None:
        self.root = root
        self.path = path
        frame = pd.read_csv(path, usecols=list(self.REQUIRED_COLUMNS), low_memory=False)
        if frame.empty or set(frame["schema_version"].astype(str)) != {
            "cq08_rd_physical_canonical_v3_1"
        }:
            raise ValueError("V17 canonical frame provenance has an invalid schema")
        if frame["source_name"].astype(str).ne("cq08_track").any():
            raise ValueError("V17 canonical frame provenance contains a non-CQ08 row")
        kept = frame["is_tensor_frame_kept"].map(_as_bool).to_numpy(dtype=np.bool_)
        if not bool(kept.all()):
            raise ValueError("V17 canonical v3_1 frame provenance unexpectedly contains dropped rows")
        for name in (
            "track_id",
            "raw_frame_index",
            "tensor_frame_index",
            "track_point_index",
        ):
            frame[name] = pd.to_numeric(frame[name], errors="raise").astype(np.int64)
        if frame.duplicated(["parent_sample_id", "tensor_frame_index"]).any():
            raise ValueError("V17 canonical frame provenance repeats a parent/tensor frame")
        records: dict[tuple[str, int], dict[str, Any]] = {}
        mismatch_count = 0
        mismatch_tracks: set[int] = set()
        shifts: list[int] = []
        for row in frame.itertuples(index=False):
            identity = parse_cq08_mat_frame_identity(str(row.rd_mat_path))
            track_id = int(row.track_id)
            if identity.track_id != track_id:
                raise ValueError(
                    "V17 canonical MAT filename track differs from frame provenance: "
                    f"track={track_id}, path={row.rd_mat_path}"
                )
            legacy_ordinal = int(row.raw_frame_index)
            raw_dat_index = int(identity.raw_dat_frame_index_zero_based)
            shift = raw_dat_index - legacy_ordinal
            shifts.append(shift)
            if shift != 0:
                mismatch_count += 1
                mismatch_tracks.add(track_id)
            records[(str(row.parent_sample_id), int(row.tensor_frame_index))] = {
                "track_id": track_id,
                "track_point_index": int(row.track_point_index),
                "legacy_mat_list_ordinal_zero_based": legacy_ordinal,
                "canonical_tensor_frame_index_zero_based": int(row.tensor_frame_index),
                "mat_acquisition_frame_number_one_based": int(
                    identity.frame_number_one_based
                ),
                "raw_dat_frame_index_zero_based": raw_dat_index,
                "rd_mat_path": str(row.rd_mat_path),
                "rd_tensor_path": str(row.rd_tensor_path),
            }
        if not records:
            raise ValueError("V17 canonical frame provenance is empty")
        self.records = records
        self.audit = {
            "schema_version": "cq08_rd_physical_canonical_v3_1",
            "row_count": len(records),
            "mat_filename_unparsed_count": 0,
            "mat_filename_track_id_mismatch_count": 0,
            "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count": mismatch_count,
            "affected_track_ids": sorted(mismatch_tracks),
            "raw_dat_minus_legacy_ordinal_minimum": min(shifts),
            "raw_dat_minus_legacy_ordinal_maximum": max(shifts),
        }

    def resolve_event(
        self,
        *,
        parent_sample_id: str,
        track_id: int,
        track_point_index: int,
        tensor_frame_indices: Sequence[int],
        legacy_mat_list_ordinals: Sequence[int],
        rd_tensor_path: str,
    ) -> list[dict[str, Any]]:
        if len(tensor_frame_indices) != len(legacy_mat_list_ordinals):
            raise ValueError("V17 canonical/legacy event frame counts differ")
        resolved: list[dict[str, Any]] = []
        for tensor_index, legacy_ordinal in zip(
            tensor_frame_indices, legacy_mat_list_ordinals, strict=True
        ):
            try:
                value = dict(self.records[(str(parent_sample_id), int(tensor_index))])
            except KeyError as exc:
                raise ValueError(
                    "V17 canonical frame provenance lacks an indexed tensor frame: "
                    f"parent={parent_sample_id}, tensor_frame={tensor_index}"
                ) from exc
            if (
                value["track_id"] != int(track_id)
                or value["track_point_index"] != int(track_point_index)
                or value["legacy_mat_list_ordinal_zero_based"] != int(legacy_ordinal)
                or _resolve(self.root, value["rd_tensor_path"])
                != _resolve(self.root, rd_tensor_path)
            ):
                raise ValueError(
                    "V17 index/canonical frame provenance mismatch: "
                    f"parent={parent_sample_id}, tensor_frame={tensor_index}, "
                    f"track_point={track_point_index}"
                )
            resolved.append(value)
        raw_indices = [value["raw_dat_frame_index_zero_based"] for value in resolved]
        if raw_indices != sorted(raw_indices) or len(raw_indices) != len(set(raw_indices)):
            raise ValueError("V17 resolved raw-DAT frame indices must be ordered and unique")
        return resolved


def _event_stats(values: np.ndarray, valid: np.ndarray, *, width: int) -> dict[str, Any]:
    selected = values[valid]
    empty = [None] * width
    return {
        "event_count": int(values.shape[0]),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "feature_minimum": selected.min(axis=0).astype(float).tolist() if len(selected) else empty,
        "feature_maximum": selected.max(axis=0).astype(float).tolist() if len(selected) else empty,
        "feature_mean": selected.mean(axis=0, dtype=np.float64).astype(float).tolist() if len(selected) else empty,
    }


def _build_split(
    root: Path,
    split: str,
    frame: pd.DataFrame,
    provenance_by_parent: dict[str, dict[str, Any]],
    canonical_frame_provenance: _CanonicalFrameProvenance,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = frame[
        frame["split_name"].astype(str).eq(split)
        & frame["label_name"].astype(str).isin(DBB_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V17 {split} DBB event population is empty or duplicated")
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    summary_values = np.zeros((len(selected), MICRO_DOPPLER_VALUE_DIM), dtype=np.float32)
    physical_values = np.zeros((len(selected), RD_PHYSICAL_VALUE_DIM), dtype=np.float32)
    valid = np.zeros((len(selected),), dtype=np.bool_)
    stft_tensors: list[np.ndarray] = []
    raw_dat_frame_indices: list[np.ndarray] = []
    legacy_mat_list_ordinals: list[np.ndarray] = []
    canonical_tensor_frame_indices: list[np.ndarray] = []
    mat_acquisition_frame_numbers: list[np.ndarray] = []
    frame_raw_summaries: list[np.ndarray] = []
    frame_period_estimable: list[np.ndarray] = []
    target_range_indices: list[np.ndarray] = []
    target_doppler_indices: list[np.ndarray] = []
    target_doppler_mps: list[np.ndarray] = []
    target_local_stft_peak_velocity_mps: list[np.ndarray] = []
    global_stft_peak_velocity_mps: list[np.ndarray] = []
    target_local_peak_power_ratio_to_global: list[np.ndarray] = []
    native_time_bin_counts: list[np.ndarray] = []
    frame_prt_num: list[np.ndarray] = []
    frame_prt_seconds: list[np.ndarray] = []
    frame_frequency_hz: list[np.ndarray] = []
    frame_cpi_count: list[np.ndarray] = []
    event_offsets = [0]
    raw_file_access_count = 0
    raw_frame_count = 0
    selected_raw_frame_count = 0
    modulation_estimable_frame_count = 0
    legacy_ordinal_raw_dat_mismatch_frame_count = 0
    legacy_ordinal_raw_dat_mismatch_track_ids: set[int] = set()
    rd_cache = _CanonicalRD(root)
    parent_label_counts: Counter[str] = Counter()
    parent_count = 0
    event_cursor = 0
    train_class_values: dict[str, list[np.ndarray]] = defaultdict(list)

    grouped = list(selected.groupby("parent_sample_id", sort=True))
    for parent_number, (parent_id_value, parent) in enumerate(grouped, start=1):
        parent_id = str(parent_id_value)
        labels = parent["label_name"].astype(str).unique().tolist()
        track_ids = pd.to_numeric(parent["track_id"], errors="raise").astype(int).unique().tolist()
        raw_paths = parent["cq08_raw_echo_file"].astype(str).unique().tolist()
        if len(labels) != 1 or len(track_ids) != 1 or len(raw_paths) != 1:
            raise ValueError(f"V17 parent label/raw path is not unique: {parent_id}")
        label = labels[0]
        track_id = int(track_ids[0])
        parent_label_counts[label] += 1
        parent_count += 1
        provenance = provenance_by_parent.get(parent_id)
        if provenance is None:
            raise ValueError(f"V17 parent lacks V16 provenance: {parent_id}")
        raw_path = _resolve(root, raw_paths[0])
        if str(raw_path) != str(Path(str(provenance["raw_echo_file"])).resolve()):
            raise ValueError(f"V17 raw path differs from V16 provenance: {parent_id}")
        actual_raw_hash = _sha256(raw_path)
        if actual_raw_hash != str(provenance["raw_sha256"]):
            raise ValueError(f"V17 raw file hash differs from V16 provenance: {parent_id}")
        raw_frames = parse_cq08_raw_echo(raw_path)
        raw_file_access_count += 1
        raw_frame_count += len(raw_frames)
        used_raw_dat_indices: set[int] = set()
        rd_paths = [
            str(value)
            for value in parent.loc[parent["rd_present"].map(_as_bool), "rd_tensor_path"]
            .dropna()
            .unique()
            if str(value).strip()
        ]
        if len(rd_paths) > 1:
            raise ValueError(f"V17 parent references multiple canonical RD tensors: {parent_id}")
        amplitude = velocity = None
        if rd_paths:
            amplitude, velocity = rd_cache.load(rd_paths[0])
        for row in parent.itertuples(index=False):
            present = _as_bool(row.rd_present)
            legacy_ordinals = _parse_indices(row.rd_raw_frame_indices_json)
            tensor_indices = _parse_indices(row.rd_tensor_frame_indices_json)
            expected_count = int(row.rd_frame_count)
            if len(legacy_ordinals) != len(tensor_indices) or len(tensor_indices) != expected_count:
                raise ValueError(f"V17 raw/tensor/count mapping mismatch: {row.sample_id}")
            if expected_count > STFT_FRAME_CAP:
                raise ValueError(f"V17 event exceeds frozen frame cap: {row.sample_id}")
            if present != bool(expected_count):
                raise ValueError(f"V17 RD presence/count mismatch: {row.sample_id}")
            resolved = canonical_frame_provenance.resolve_event(
                parent_sample_id=parent_id,
                track_id=track_id,
                track_point_index=int(row.track_point_index),
                tensor_frame_indices=tensor_indices,
                legacy_mat_list_ordinals=legacy_ordinals,
                rd_tensor_path=str(row.rd_tensor_path),
            )
            raw_dat_indices = [
                int(value["raw_dat_frame_index_zero_based"]) for value in resolved
            ]
            if set(raw_dat_indices) & used_raw_dat_indices:
                raise ValueError(
                    f"V17 raw-DAT frame is assigned to multiple events: {row.sample_id}"
                )
            used_raw_dat_indices.update(raw_dat_indices)
            mismatch_count = sum(
                int(value["legacy_mat_list_ordinal_zero_based"])
                != int(value["raw_dat_frame_index_zero_based"])
                for value in resolved
            )
            legacy_ordinal_raw_dat_mismatch_frame_count += mismatch_count
            if mismatch_count:
                legacy_ordinal_raw_dat_mismatch_track_ids.add(track_id)
            event = extract_current_event_micro_doppler(
                raw_frames,
                raw_dat_indices,
                track_point_index=int(row.track_point_index),
            )
            if present:
                if amplitude is None or velocity is None:
                    raise ValueError(f"V17 RD-present parent lacks canonical tensor: {parent_id}")
                if min(tensor_indices) < 0 or max(tensor_indices) >= amplitude.shape[0]:
                    raise ValueError(f"V17 canonical frame index is out of bounds: {row.sample_id}")
                physical = extract_current_event_rd_physical(
                    amplitude[np.asarray(tensor_indices, dtype=np.int64)], velocity
                )
                if not event.valid or not physical.valid:
                    raise ValueError(f"V17 present event produced invalid physical evidence: {row.sample_id}")
                summary_values[event_cursor] = event.values
                physical_values[event_cursor] = physical.values
                valid[event_cursor] = True
                active = event.frame_mask
                stft_tensors.append(event.stft_tensors[active])
                if event.raw_frame_indices[active].astype(int).tolist() != raw_dat_indices:
                    raise ValueError(
                        f"V17 extractor returned different raw-DAT frame indices: {row.sample_id}"
                    )
                raw_dat_frame_indices.append(event.raw_frame_indices[active])
                legacy_mat_list_ordinals.append(
                    np.asarray(
                        [
                            value["legacy_mat_list_ordinal_zero_based"]
                            for value in resolved
                        ],
                        dtype=np.int32,
                    )
                )
                canonical_tensor_frame_indices.append(
                    np.asarray(tensor_indices, dtype=np.int32)
                )
                mat_acquisition_frame_numbers.append(
                    np.asarray(
                        [
                            value["mat_acquisition_frame_number_one_based"]
                            for value in resolved
                        ],
                        dtype=np.int32,
                    )
                )
                frame_raw_summaries.append(event.frame_raw_summaries[active])
                frame_period_estimable.append(event.frame_period_estimable[active])
                target_range_indices.append(event.target_range_indices[active])
                target_doppler_indices.append(event.target_doppler_indices[active])
                target_doppler_mps.append(event.target_doppler_mps[active])
                target_local_stft_peak_velocity_mps.append(
                    event.target_local_stft_peak_velocity_mps[active]
                )
                global_stft_peak_velocity_mps.append(
                    event.global_stft_peak_velocity_mps[active]
                )
                target_local_peak_power_ratio_to_global.append(
                    event.target_local_peak_power_ratio_to_global[active]
                )
                native_time_bin_counts.append(event.native_time_bin_counts[active])
                selected_raw_frames = [raw_frames[index] for index in raw_dat_indices]
                frame_prt_num.append(
                    np.asarray([frame.prt_num for frame in selected_raw_frames], dtype=np.int32)
                )
                frame_prt_seconds.append(
                    np.asarray(
                        [frame.prt_seconds for frame in selected_raw_frames], dtype=np.float64
                    )
                )
                frame_frequency_hz.append(
                    np.asarray(
                        [frame.frequency_hz for frame in selected_raw_frames], dtype=np.float64
                    )
                )
                frame_cpi_count.append(
                    np.asarray([frame.cpi_count for frame in selected_raw_frames], dtype=np.int64)
                )
                selected_raw_frame_count += int(active.sum())
                modulation_estimable_frame_count += int(event.frame_period_estimable[active].sum())
                if split == "train":
                    train_class_values[label].append(event.values.copy())
            elif event.valid or np.any(event.values != 0.0):
                raise ValueError(f"V17 RD-missing event must have zero micro-Doppler evidence: {row.sample_id}")
            event_offsets.append(selected_raw_frame_count)
            event_cursor += 1
        if parent_number == 1 or parent_number % 50 == 0 or parent_number == len(grouped):
            print(
                f"[V17 build] split={split} parent={parent_number}/{len(grouped)} "
                f"events={event_cursor}/{len(selected)} raw_frames={selected_raw_frame_count}",
                flush=True,
            )

    if event_cursor != len(selected) or event_offsets[-1] != selected_raw_frame_count:
        raise ValueError(f"V17 {split} event/STFT offset accounting failed")
    if not np.array_equal(valid, selected["rd_present"].map(_as_bool).to_numpy(dtype=np.bool_)):
        raise ValueError(f"V17 {split} summary valid mask differs from current-event RD presence")
    if np.any(summary_values[~valid] != 0.0) or np.any(physical_values[~valid] != 0.0):
        raise ValueError(f"V17 {split} missing evidence rows are not exactly zero")
    tensors = (
        np.concatenate(stft_tensors, axis=0).astype(np.float16, copy=False)
        if stft_tensors
        else np.zeros((0, STFT_VELOCITY_BIN_COUNT, STFT_TIME_BIN_COUNT), dtype=np.float16)
    )
    raw_dat_indices_flat = (
        np.concatenate(raw_dat_frame_indices).astype(np.int32, copy=False)
        if raw_dat_frame_indices
        else np.zeros((0,), dtype=np.int32)
    )
    legacy_ordinals_flat = (
        np.concatenate(legacy_mat_list_ordinals).astype(np.int32, copy=False)
        if legacy_mat_list_ordinals
        else np.zeros((0,), dtype=np.int32)
    )
    tensor_indices_flat = (
        np.concatenate(canonical_tensor_frame_indices).astype(np.int32, copy=False)
        if canonical_tensor_frame_indices
        else np.zeros((0,), dtype=np.int32)
    )
    mat_frame_numbers_flat = (
        np.concatenate(mat_acquisition_frame_numbers).astype(np.int32, copy=False)
        if mat_acquisition_frame_numbers
        else np.zeros((0,), dtype=np.int32)
    )
    raw_summaries_flat = (
        np.concatenate(frame_raw_summaries, axis=0).astype(np.float32, copy=False)
        if frame_raw_summaries
        else np.zeros((0, MICRO_DOPPLER_VALUE_DIM), dtype=np.float32)
    )
    period_flat = (
        np.concatenate(frame_period_estimable).astype(np.bool_, copy=False)
        if frame_period_estimable
        else np.zeros((0,), dtype=np.bool_)
    )
    range_flat = (
        np.concatenate(target_range_indices).astype(np.int16, copy=False)
        if target_range_indices
        else np.zeros((0,), dtype=np.int16)
    )
    doppler_index_flat = (
        np.concatenate(target_doppler_indices).astype(np.int32, copy=False)
        if target_doppler_indices
        else np.zeros((0,), dtype=np.int32)
    )
    target_doppler_mps_flat = (
        np.concatenate(target_doppler_mps).astype(np.float32, copy=False)
        if target_doppler_mps
        else np.zeros((0,), dtype=np.float32)
    )
    target_local_peak_velocity_mps_flat = (
        np.concatenate(target_local_stft_peak_velocity_mps).astype(
            np.float32, copy=False
        )
        if target_local_stft_peak_velocity_mps
        else np.zeros((0,), dtype=np.float32)
    )
    global_peak_velocity_mps_flat = (
        np.concatenate(global_stft_peak_velocity_mps).astype(np.float32, copy=False)
        if global_stft_peak_velocity_mps
        else np.zeros((0,), dtype=np.float32)
    )
    target_local_peak_power_ratio_flat = (
        np.concatenate(target_local_peak_power_ratio_to_global).astype(
            np.float32, copy=False
        )
        if target_local_peak_power_ratio_to_global
        else np.zeros((0,), dtype=np.float32)
    )
    time_count_flat = (
        np.concatenate(native_time_bin_counts).astype(np.int16, copy=False)
        if native_time_bin_counts
        else np.zeros((0,), dtype=np.int16)
    )
    prt_num_flat = (
        np.concatenate(frame_prt_num).astype(np.int32, copy=False)
        if frame_prt_num
        else np.zeros((0,), dtype=np.int32)
    )
    prt_seconds_flat = (
        np.concatenate(frame_prt_seconds).astype(np.float64, copy=False)
        if frame_prt_seconds
        else np.zeros((0,), dtype=np.float64)
    )
    frequency_hz_flat = (
        np.concatenate(frame_frequency_hz).astype(np.float64, copy=False)
        if frame_frequency_hz
        else np.zeros((0,), dtype=np.float64)
    )
    cpi_count_flat = (
        np.concatenate(frame_cpi_count).astype(np.int64, copy=False)
        if frame_cpi_count
        else np.zeros((0,), dtype=np.int64)
    )
    flat_count = tensors.shape[0]
    if not all(
        len(value) == flat_count
        for value in (
            raw_dat_indices_flat,
            legacy_ordinals_flat,
            tensor_indices_flat,
            mat_frame_numbers_flat,
            raw_summaries_flat,
            period_flat,
            range_flat,
            doppler_index_flat,
            target_doppler_mps_flat,
            target_local_peak_velocity_mps_flat,
            global_peak_velocity_mps_flat,
            target_local_peak_power_ratio_flat,
            time_count_flat,
            prt_num_flat,
            prt_seconds_flat,
            frequency_hz_flat,
            cpi_count_flat,
        )
    ):
        raise ValueError(f"V17 {split} ragged STFT arrays are not aligned")
    train_feature_by_label = {
        label: {
            "event_count": len(values),
            "mean": np.stack(values).mean(axis=0, dtype=np.float64).astype(float).tolist(),
            "median": np.median(np.stack(values), axis=0).astype(float).tolist(),
        }
        for label, values in sorted(train_class_values.items())
        if values
    }
    arrays = {
        "sample_ids": sample_ids,
        "summary_values": summary_values,
        "physical_values": physical_values,
        "valid": valid,
        "event_offsets": np.asarray(event_offsets, dtype=np.int64),
        "stft_tensors": tensors,
        "raw_dat_frame_indices_zero_based": raw_dat_indices_flat,
        "legacy_mat_list_ordinals_zero_based": legacy_ordinals_flat,
        "canonical_tensor_frame_indices_zero_based": tensor_indices_flat,
        "mat_acquisition_frame_numbers_one_based": mat_frame_numbers_flat,
        "frame_raw_summaries": raw_summaries_flat,
        "frame_period_estimable": period_flat,
        "target_range_indices": range_flat,
        "target_doppler_indices": doppler_index_flat,
        "target_doppler_mps": target_doppler_mps_flat,
        "target_local_stft_peak_velocity_mps": target_local_peak_velocity_mps_flat,
        "global_stft_peak_velocity_mps": global_peak_velocity_mps_flat,
        "target_local_peak_power_ratio_to_global": (
            target_local_peak_power_ratio_flat
        ),
        "native_time_bin_counts": time_count_flat,
        "frame_prt_num": prt_num_flat,
        "frame_prt_seconds": prt_seconds_flat,
        "frame_frequency_hz": frequency_hz_flat,
        "frame_cpi_count": cpi_count_flat,
    }
    stats = {
        "split": split,
        "parent_count": parent_count,
        "parent_label_counts": dict(sorted(parent_label_counts.items())),
        "event_count": len(selected),
        "raw_file_access_count": raw_file_access_count,
        "raw_frame_count_in_accessed_files": raw_frame_count,
        "selected_current_event_raw_frame_count": selected_raw_frame_count,
        "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count": (
            legacy_ordinal_raw_dat_mismatch_frame_count
        ),
        "legacy_mat_list_ordinal_mismatch_track_ids": sorted(
            legacy_ordinal_raw_dat_mismatch_track_ids
        ),
        "canonical_rd_tensor_accessed_count": len(rd_cache.accessed_paths),
        "modulation_period_estimable_frame_count": modulation_estimable_frame_count,
        "modulation_period_estimable_frame_fraction": (
            modulation_estimable_frame_count / selected_raw_frame_count
            if selected_raw_frame_count
            else 0.0
        ),
        "micro_doppler": _event_stats(summary_values, valid, width=MICRO_DOPPLER_VALUE_DIM),
        "rd_physical": _event_stats(physical_values, valid, width=RD_PHYSICAL_VALUE_DIM),
        "train_only_feature_by_label": train_feature_by_label,
        "test_raw_payload_access_count": 0,
        "test_rd_payload_access_count": 0,
    }
    return arrays, stats


def _build_contract(
    source_contract: dict[str, Any],
    *,
    source_hashes: dict[str, str],
    cache_hashes: dict[str, dict[str, str]],
    split_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    contract = json.loads(json.dumps(source_contract))
    feature_contract = dict(contract["feature_contract"])
    feature_contract["rd_physical"] = rd_physical_contract()
    feature_contract["micro_doppler"] = micro_doppler_contract()
    contract.update(
        {
            "schema_version": VERSION,
            "contract_name": (
                "cq08_v16_grouped_dbb_rd_physical4_"
                "target_anchored_micro_doppler3_paired_v3"
            ),
            "contract_status": "prepared_for_manual_formal_paired_training",
            "source_contract": SOURCE_CONTRACT.as_posix(),
            "source_contract_sha256": source_hashes["contract"],
            "source_index": SOURCE_INDEX.as_posix(),
            "source_index_sha256": source_hashes["index"],
            "source_provenance": SOURCE_PROVENANCE.as_posix(),
            "source_provenance_sha256": source_hashes["provenance"],
            "source_normalization_stats": SOURCE_STATS.as_posix(),
            "source_normalization_stats_sha256": source_hashes["stats"],
            "source_canonical_frame_index": SOURCE_CANONICAL_FRAME_INDEX.as_posix(),
            "source_canonical_frame_index_sha256": source_hashes[
                "canonical_frame_index"
            ],
            "feature_contract": feature_contract,
            "v17_scope": {
                "task": "dbb_specialist",
                "labels": list(DBB_LABELS),
                "splits_processed": list(SPLITS),
                "test_rows_read_for_split_exclusion_only": True,
                "test_raw_payload_access_count": 0,
                "test_rd_payload_access_count": 0,
                "complete_stft_tensor_is_model_input": False,
                "treatment_model_input": list(MICRO_DOPPLER_NAMES),
            },
            "rd_physical_cache": {
                "schema": RD_PHYSICAL_SCHEMA,
                "cache_scope": "V16 capture/content-grouped DBB train and validation events only",
                "files": {
                    split: {
                        "path": _physical_cache(split).as_posix(),
                        "sha256": cache_hashes[split]["physical"],
                        **split_stats[split]["rd_physical"],
                    }
                    for split in SPLITS
                },
                "test_excluded": True,
                "test_rd_payload_access_count": 0,
            },
            "micro_doppler_cache": {
                "summary_schema": MICRO_DOPPLER_SCHEMA,
                "summary_files": {
                    split: {
                        "path": _summary_cache(split).as_posix(),
                        "sha256": cache_hashes[split]["summary"],
                        **split_stats[split]["micro_doppler"],
                    }
                    for split in SPLITS
                },
                "stft_files": {
                    split: {
                        "path": _stft_cache(split).as_posix(),
                        "sha256": cache_hashes[split]["stft"],
                        "event_count": split_stats[split]["event_count"],
                        "selected_frame_count": split_stats[split][
                            "selected_current_event_raw_frame_count"
                        ],
                    }
                    for split in SPLITS
                },
                "strict_alignment": (
                    "summary sample_ids exactly equal the configured split's DBB event set; "
                    "canonical tensor indices resolve through the SHA-256-bound canonical frame index "
                    "and MAT acquisition frame number to the true zero-based raw-DAT frame; "
                    "STFT event_offsets preserve that current-event mapping"
                ),
                "raw_frame_mapping": {
                    "legacy_index_field": "rd_raw_frame_indices_json",
                    "legacy_index_semantics": (
                        "zero-based ordinal in cq08_unified_manifest.rd_mat_files; never used as a raw-DAT frame index"
                    ),
                    "canonical_lookup_key": [
                        "parent_sample_id",
                        "rd_tensor_frame_indices_json",
                    ],
                    "canonical_provenance_field": "rd_mat_path",
                    "mat_filename_frame_semantics": "one-based acquisition frame number",
                    "raw_dat_frame_semantics": "MAT acquisition frame number minus one",
                    "per_frame_validation": (
                        "track ID, track-point index, tensor path and raw-DAT target record must all match"
                    ),
                    "audit": {
                        split: {
                            "mismatch_frame_count": split_stats[split][
                                "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count"
                            ],
                            "affected_track_ids": split_stats[split][
                                "legacy_mat_list_ordinal_mismatch_track_ids"
                            ],
                        }
                        for split in SPLITS
                    },
                },
                "evidence_only_target_peak_metadata": [
                    "target_doppler_indices",
                    "target_doppler_mps",
                    "target_local_stft_peak_velocity_mps",
                    "global_stft_peak_velocity_mps",
                    "target_local_peak_power_ratio_to_global",
                ],
                "evidence_only_sampling_metadata": [
                    "frame_prt_num",
                    "frame_prt_seconds",
                    "frame_frequency_hz",
                    "frame_cpi_count",
                ],
                "sampling_metadata_is_model_input": False,
                "test_excluded": True,
                "test_raw_payload_access_count": 0,
            },
            "training_contract_v17": {
                "paired_protocol": PAIRED_PROTOCOL,
                "control": "V16 DBB local architecture plus shared current-event rd_physical4",
                "treatment": "Control plus exactly three current-event micro-Doppler summary values",
                "complete_stft_tensor": "versioned provenance evidence only; not consumed by this classifier",
                "shared_initialization": (
                    "all shared tensors copied exactly; Treatment's three new Fusion columns exactly zero"
                ),
                "seed": SEED,
                "dataloader_seed": SEED,
                "epochs": 20,
                "early_stopping": False,
                "checkpoint_selection": (
                    "each arm's validation parent_min_precision_recall best; same-epoch trajectories are "
                    "attribution guards"
                ),
                "formal_training": "未执行",
                "full_cascade": "未执行",
                "oof": "未执行",
                "test": "未执行",
                "stable": "未执行",
                "p95": "未执行",
            },
            "v17_acceptance": {
                "primary_scope": "DBB validation parent-final local task only",
                "all_DBB_class_precision_recall_f1_minimum": 0.95,
                "paired_direction": (
                    "Treatment Drone/Bird minimum precision-recall-F1 and parent Macro-F1 must improve; "
                    "Drone<->Bird confusion count must decrease without increasing it in either direction"
                ),
                "balloon_guard": "Balloon recall may not lose more than one validation parent",
                "not_claimed": [
                    "five-class cascade performance",
                    "strict-open-set rejection performance",
                    "test generalization",
                    "stable performance",
                    "P95 latency",
                ],
            },
        }
    )
    contract["source_purity"]["raw_echo_use"] = (
        "CQ08-internal train/validation DBB current-event complex I/Q is used for V17 STFT and "
        "three summary values; the separately named external raw source remains excluded"
    )
    contract["source_purity"]["cq08_original_echo_use"] = (
        "allowed because it belongs to source_name=CQ08 and is bound to V16 capture/content "
        "provenance; it does not authorize the excluded external source named raw"
    )
    contract["data_scope"]["test_payload_access_for_hash_only"] = False
    contract["data_scope"]["test_used_for_normalization"] = False
    contract["data_scope"]["test_used_for_checkpoint_selection"] = False
    return contract


def _build_manifest(
    *,
    contract_hash: str,
    source_hashes: dict[str, str],
    cache_hashes: dict[str, dict[str, str]],
    split_stats: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": VERSION,
        "manifest_type": (
            "cq08_dbb_current_event_target_anchored_micro_doppler_paired_cache_v3"
        ),
        "paired_protocol": PAIRED_PROTOCOL,
        "source_name": "CQ08",
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": source_hashes["index"],
        "source_contract": SOURCE_CONTRACT.as_posix(),
        "source_contract_sha256": source_hashes["contract"],
        "source_provenance": SOURCE_PROVENANCE.as_posix(),
        "source_provenance_sha256": source_hashes["provenance"],
        "source_canonical_frame_index": SOURCE_CANONICAL_FRAME_INDEX.as_posix(),
        "source_canonical_frame_index_sha256": source_hashes[
            "canonical_frame_index"
        ],
        "data_contract": DATA_CONTRACT.as_posix(),
        "data_contract_sha256": contract_hash,
        "rd_physical_contract": rd_physical_contract(),
        "micro_doppler_contract": micro_doppler_contract(),
        "cache_files": {
            split: {
                "physical": {
                    "path": _physical_cache(split).as_posix(),
                    "sha256": cache_hashes[split]["physical"],
                },
                "summary": {
                    "path": _summary_cache(split).as_posix(),
                    "sha256": cache_hashes[split]["summary"],
                },
                "stft": {
                    "path": _stft_cache(split).as_posix(),
                    "sha256": cache_hashes[split]["stft"],
                },
                "statistics": split_stats[split],
            }
            for split in SPLITS
        },
        "cache_alignment_fields": {
            "summary": ["sample_ids", "values", "valid"],
            "physical": ["sample_ids", "values", "valid"],
            "stft": [
                "sample_ids",
                "valid",
                "event_offsets",
                "stft_tensors",
                "raw_dat_frame_indices_zero_based",
                "legacy_mat_list_ordinals_zero_based",
                "canonical_tensor_frame_indices_zero_based",
                "mat_acquisition_frame_numbers_one_based",
                "frame_raw_summaries",
                "frame_period_estimable",
                "target_range_indices",
                "target_doppler_indices",
                "target_doppler_mps",
                "target_local_stft_peak_velocity_mps",
                "global_stft_peak_velocity_mps",
                "target_local_peak_power_ratio_to_global",
                "native_time_bin_counts",
                "frame_prt_num",
                "frame_prt_seconds",
                "frame_frequency_hz",
                "frame_cpi_count",
            ],
        },
        "sample_id_model_input": False,
        "raw_frame_index_model_input": False,
        "legacy_mat_list_ordinal_is_raw_dat_frame_index": False,
        "test_excluded": True,
        "test_raw_payload_access_count": 0,
        "test_rd_payload_access_count": 0,
        "formal_training": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }


def _build_config(
    source: dict[str, Any],
    *,
    arm: str,
    source_hashes: dict[str, str],
    contract_hash: str,
    manifest_hash: str,
    cache_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    if arm not in {"control", "treatment"}:
        raise ValueError("V17 arm must be control or treatment")
    payload = json.loads(json.dumps(source))
    artifact = _artifact_dir(arm)
    payload["schema_version"] = VERSION
    payload["config_name"] = _config_path(arm).stem.replace("-config", "")
    payload["data"].update(
        {
            "contract_path": DATA_CONTRACT.as_posix(),
            "contract_schema_version": VERSION,
            "index_schema_version": SOURCE_VERSION,
            "expected_index_sha256": source_hashes["index"],
            "expected_contract_sha256": contract_hash,
            "expected_stats_sha256": source_hashes["stats"],
            "rd_physical_schema": RD_PHYSICAL_SCHEMA,
            "rd_physical_mask_mode": RD_PHYSICAL_MASK_MODE,
            "rd_physical_cache_manifest_path": CACHE_MANIFEST.as_posix(),
            "expected_rd_physical_cache_manifest_sha256": manifest_hash,
            "train_rd_physical_cache_path": _physical_cache("train").as_posix(),
            "validation_rd_physical_cache_path": _physical_cache("val").as_posix(),
            "expected_train_rd_physical_cache_sha256": cache_hashes["train"]["physical"],
            "expected_validation_rd_physical_cache_sha256": cache_hashes["val"]["physical"],
            "micro_doppler_schema": MICRO_DOPPLER_SCHEMA,
            "micro_doppler_mask_mode": MICRO_DOPPLER_MASK_MODE,
            "micro_doppler_cache_manifest_path": CACHE_MANIFEST.as_posix(),
            "expected_micro_doppler_cache_manifest_sha256": manifest_hash,
            "train_micro_doppler_cache_path": _summary_cache("train").as_posix(),
            "validation_micro_doppler_cache_path": _summary_cache("val").as_posix(),
            "expected_train_micro_doppler_cache_sha256": cache_hashes["train"]["summary"],
            "expected_validation_micro_doppler_cache_sha256": cache_hashes["val"]["summary"],
            "train_micro_doppler_stft_cache_path": _stft_cache("train").as_posix(),
            "validation_micro_doppler_stft_cache_path": _stft_cache("val").as_posix(),
            "expected_train_micro_doppler_stft_cache_sha256": cache_hashes["train"]["stft"],
            "expected_validation_micro_doppler_stft_cache_sha256": cache_hashes["val"]["stft"],
            "micro_doppler_frame_provenance_path": SOURCE_CANONICAL_FRAME_INDEX.as_posix(),
            "expected_micro_doppler_frame_provenance_sha256": source_hashes[
                "canonical_frame_index"
            ],
        }
    )
    payload["model"]["rd_physical_dim"] = RD_PHYSICAL_VALUE_DIM
    payload["model"]["micro_doppler_dim"] = (
        MICRO_DOPPLER_VALUE_DIM if arm == "treatment" else 0
    )
    payload["train"].update(
        {
            "seed": SEED,
            "dataloader_seed": SEED,
            "epochs": 20,
            "early_stopping_patience": 0,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                f"{VERSION}-dbb-target-anchored-micro-doppler-paired-checkpoint-v3"
            ),
            "paired_protocol": PAIRED_PROTOCOL,
            "paired_arm": arm,
            "paired_seed": SEED,
            "paired_counterpart_config_path": _config_path(
                "treatment" if arm == "control" else "control"
            ).as_posix(),
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": False,
        }
    )
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_paths = {
        "index": root / SOURCE_INDEX,
        "contract": root / SOURCE_CONTRACT,
        "provenance": root / SOURCE_PROVENANCE,
        "stats": root / SOURCE_STATS,
        "canonical_frame_index": root / SOURCE_CANONICAL_FRAME_INDEX,
        "config": root / SOURCE_DBB_CONFIG,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}
    index = pd.read_csv(source_paths["index"], low_memory=False)
    required = {
        "schema_version",
        "sample_id",
        "parent_sample_id",
        "track_id",
        "track_point_index",
        "causal_event_index",
        "split_name",
        "label_name",
        "rd_present",
        "rd_frame_count",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
        "rd_raw_frame_indices_json",
        "cq08_raw_echo_file",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"V16 index lacks V17 source fields: {missing}")
    if index["schema_version"].astype(str).ne(SOURCE_VERSION).any():
        raise ValueError("V17 source index contains a row outside RADAR-LN3-V16")
    if index["sample_id"].astype(str).duplicated().any():
        raise ValueError("V17 source index contains duplicate sample IDs")
    split_counts = index.groupby("split_name").size().to_dict()
    if set(split_counts) != {"train", "val", "test"}:
        raise ValueError(f"V17 source split set is invalid: {split_counts}")
    provenance = json.loads(source_paths["provenance"].read_text(encoding="utf-8"))
    provenance_by_parent = {
        str(value["parent_sample_id"]): dict(value)
        for value in provenance.get("records", [])
    }
    if len(provenance_by_parent) != int(provenance.get("parent_count", -1)):
        raise ValueError("V16 provenance parent records are incomplete")
    canonical_frame_provenance = _CanonicalFrameProvenance(
        root, source_paths["canonical_frame_index"]
    )

    split_stats: dict[str, dict[str, Any]] = {}
    write_status: dict[str, dict[str, str]] = {}
    cache_hashes: dict[str, dict[str, str]] = {}
    for split in SPLITS:
        arrays, statistics = _build_split(
            root,
            split,
            index,
            provenance_by_parent,
            canonical_frame_provenance,
        )
        split_stats[split] = statistics
        physical_fields = (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["physical_values"]),
            ("valid", arrays["valid"]),
        )
        summary_fields = (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["summary_values"]),
            ("valid", arrays["valid"]),
        )
        stft_fields = tuple(
            (name, arrays[name])
            for name in (
                "sample_ids",
                "valid",
                "event_offsets",
                "stft_tensors",
                "raw_dat_frame_indices_zero_based",
                "legacy_mat_list_ordinals_zero_based",
                "canonical_tensor_frame_indices_zero_based",
                "mat_acquisition_frame_numbers_one_based",
                "frame_raw_summaries",
                "frame_period_estimable",
                "target_range_indices",
                "target_doppler_indices",
                "target_doppler_mps",
                "target_local_stft_peak_velocity_mps",
                "global_stft_peak_velocity_mps",
                "target_local_peak_power_ratio_to_global",
                "native_time_bin_counts",
                "frame_prt_num",
                "frame_prt_seconds",
                "frame_frequency_hz",
                "frame_cpi_count",
            )
        )
        write_status[split] = {
            "physical": _write_npz_new_or_identical(root / _physical_cache(split), physical_fields),
            "summary": _write_npz_new_or_identical(root / _summary_cache(split), summary_fields),
            "stft": _write_npz_new_or_identical(root / _stft_cache(split), stft_fields),
        }
        cache_hashes[split] = {
            "physical": _sha256(root / _physical_cache(split)),
            "summary": _sha256(root / _summary_cache(split)),
            "stft": _sha256(root / _stft_cache(split)),
        }
        del arrays

    source_contract = json.loads(source_paths["contract"].read_text(encoding="utf-8"))
    contract_payload = _build_contract(
        source_contract,
        source_hashes=source_hashes,
        cache_hashes=cache_hashes,
        split_stats=split_stats,
    )
    contract_status = _write_json_new_or_identical(root / DATA_CONTRACT, contract_payload)
    contract_hash = _sha256(root / DATA_CONTRACT)
    manifest_payload = _build_manifest(
        contract_hash=contract_hash,
        source_hashes=source_hashes,
        cache_hashes=cache_hashes,
        split_stats=split_stats,
    )
    manifest_status = _write_json_new_or_identical(root / CACHE_MANIFEST, manifest_payload)
    manifest_hash = _sha256(root / CACHE_MANIFEST)

    source_config = json.loads(source_paths["config"].read_text(encoding="utf-8"))
    config_status: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    for arm in ("control", "treatment"):
        config_path = root / _config_path(arm)
        artifact = root / _artifact_dir(arm)
        payload = _build_config(
            source_config,
            arm=arm,
            source_hashes=source_hashes,
            contract_hash=contract_hash,
            manifest_hash=manifest_hash,
            cache_hashes=cache_hashes,
        )
        expected_config_text = _json_text(payload)
        if artifact.exists() and any(artifact.iterdir()):
            if (
                not config_path.is_file()
                or config_path.read_text(encoding="utf-8") != expected_config_text
            ):
                raise FileExistsError(
                    "refusing to prepare when a non-empty V17 artifact is not bound to the "
                    f"identical generated config: {artifact}"
                )
        config_status[arm] = _write_json_new_or_identical(config_path, payload)
        config_hashes[arm] = _sha256(config_path)

    return {
        "status": "completed",
        "schema_version": VERSION,
        "paired_protocol": PAIRED_PROTOCOL,
        "write_status": {
            "caches": write_status,
            "contract": contract_status,
            "manifest": manifest_status,
            "configs": config_status,
        },
        "cache_hashes": cache_hashes,
        "contract": {"path": str(root / DATA_CONTRACT), "sha256": contract_hash},
        "manifest": {"path": str(root / CACHE_MANIFEST), "sha256": manifest_hash},
        "configs": {
            arm: {"path": str(root / _config_path(arm)), "sha256": config_hashes[arm]}
            for arm in ("control", "treatment")
        },
        "split_statistics": split_stats,
        "canonical_frame_provenance_audit": canonical_frame_provenance.audit,
        "test_raw_payload_access_count": 0,
        "test_rd_payload_access_count": 0,
        "formal_training": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the V17 DBB current-event micro-Doppler paired data contract."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
