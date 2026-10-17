from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from radar_ln3.micro_doppler import parse_cq08_mat_frame_identity
















def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()






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














