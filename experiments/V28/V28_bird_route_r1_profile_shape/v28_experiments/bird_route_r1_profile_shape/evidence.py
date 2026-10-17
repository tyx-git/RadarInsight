"""Materialize and audit complete V18 micro-Doppler profile evidence for r1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from build.micro_doppler_v18 import _CanonicalFrameProvenance, _build_split

from .protocol import R1Context


EVIDENCE_FIELDS = (
    "sample_ids",
    "valid",
    "event_offsets",
    "raw_dat_frame_indices_zero_based",
    "legacy_mat_list_ordinals_zero_based",
    "canonical_tensor_frame_indices_zero_based",
    "mat_acquisition_frame_numbers_one_based",
    "frame_values",
    "frame_feature_reliable",
    "residual_doppler_profiles",
    "envelope_modulation_profiles",
    "target_range_indices",
    "target_doppler_indices",
    "target_doppler_mps",
    "refined_body_doppler_mps",
    "velocity_nyquist_mps",
    "rayleigh_velocity_mps",
    "residual_sideband_snr_db",
    "envelope_modulation_snr_db",
    "frame_prt_num",
    "frame_prt_seconds",
    "frame_frequency_hz",
    "frame_cpi_count",
)


def _source_path(context: R1Context, split: str) -> Path:
    template = str(context.config["data"]["source_evidence_template"])
    return (context.project_root / template.format(split=split)).resolve()


def _load_index(context: R1Context) -> pd.DataFrame:
    return pd.read_csv(context.project_root / context.config["data"]["index_path"], low_memory=False)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    np.savez_compressed(temporary, **{name: np.asarray(arrays[name]) for name in EVIDENCE_FIELDS})
    generated = Path(f"{temporary}.npz")
    if generated != path:
        generated.replace(path)
    if temporary.exists():
        temporary.unlink()


def _load_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != EVIDENCE_FIELDS:
            raise ValueError(f"Bird-route r1 evidence fields differ: {path}")
        return {name: np.asarray(archive[name]) for name in EVIDENCE_FIELDS}


def _build_one(context: R1Context, frame: pd.DataFrame, provenance: dict[str, dict[str, Any]], canonical: Any, split: str) -> dict[str, Any]:
    labels = ("DroneTarget", "BirdTarget", "BalloonTarget", "UnknownTarget")
    if split == context.config["data"]["test_split_name"]:
        labels = (*labels, "ClutterTarget")
    arrays, statistics = _build_split(
        context.project_root,
        split,
        frame,
        provenance,
        canonical,
        allowed_labels=labels,
        include_evidence=True,
    )
    payload = {
        "sample_ids": arrays["sample_ids"],
        "valid": arrays["valid"],
        "event_offsets": arrays["event_offsets"],
        **{name: arrays[name] for name in EVIDENCE_FIELDS if name not in {"sample_ids", "valid", "event_offsets"}},
    }
    path = _source_path(context, split)
    _write_npz(path, payload)
    return {
        "split": split,
        "path": str(path.relative_to(context.project_root).as_posix()),
        "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        "event_count": int(payload["sample_ids"].shape[0]),
        "valid_event_count": int(payload["valid"].sum()),
        "frame_count": int(payload["raw_dat_frame_indices_zero_based"].shape[0]),
        "labels": dict(Counter(frame.loc[frame["sample_id"].astype(str).isin(payload["sample_ids"].astype(str)), "label_name"].astype(str))),
        "statistics": statistics,
    }


def prepare_evidence(context: R1Context) -> dict[str, Any]:
    frame = _load_index(context)
    provenance_payload = json.loads((context.project_root / context.config["data"]["provenance_path"]).read_text(encoding="utf-8"))
    provenance = {str(item["parent_sample_id"]): dict(item) for item in provenance_payload.get("records", [])}
    canonical = _CanonicalFrameProvenance(context.project_root, context.project_root / context.config["data"]["canonical_frame_index_path"])
    # Test evidence is intentionally materialized only by the final audit
    # command, after the profile head and threshold are frozen.
    splits = (
        context.config["data"]["train_split_name"],
        context.config["data"]["validation_split_name"],
    )
    records = {split: _build_one(context, frame, provenance, canonical, split) for split in splits}
    audit = audit_evidence(context, frame=frame, records=records)
    return {"experiment_id": "V28-bird-route-r1-profile-shape", "status": "prepared", "records": records, "audit": audit}


def audit_evidence(context: R1Context, *, splits: tuple[str, ...] | None = None, frame: pd.DataFrame | None = None, records: Mapping[str, Any] | None = None) -> dict[str, Any]:
    frame = _load_index(context) if frame is None else frame
    report: dict[str, Any] = {"schema_version": "RADAR-LN3-V28-bird-route-r1-evidence-audit-v1", "experiment_id": "V28-bird-route-r1-profile-shape", "status": "passed", "test_access_count": 0, "splits": {}}
    requested_splits = splits or (
        context.config["data"]["train_split_name"],
        context.config["data"]["validation_split_name"],
    )
    for split in requested_splits:
        path = _source_path(context, split)
        cache = _load_cache(path)
        ids = cache["sample_ids"].astype(str)
        if len(ids) != len(set(ids.tolist())):
            raise ValueError(f"Bird-route r1 evidence has duplicate sample IDs: {split}")
        selected = frame[frame["sample_id"].astype(str).isin(ids)].copy()
        if len(selected) != len(ids):
            raise ValueError(f"Bird-route r1 evidence/index coverage differs: {split}")
        selected["sample_id"] = selected["sample_id"].astype(str)
        aligned = selected.set_index("sample_id").loc[ids.tolist()]
        parents = aligned["parent_sample_id"].astype(str).to_numpy()
        event_indices = pd.to_numeric(aligned["causal_event_index"], errors="raise").to_numpy(dtype=np.int64)
        previous = None
        seen: set[str] = set()
        for parent, event_index in zip(parents.tolist(), event_indices.tolist(), strict=True):
            if previous is not None and parent != previous:
                seen.add(previous)
            if parent in seen:
                raise ValueError(f"Bird-route r1 parent events are not contiguous: {split}/{parent}")
            previous = parent
            if event_index < 0:
                raise ValueError(f"Bird-route r1 event index is negative: {split}/{parent}")
        for parent, group in aligned.groupby("parent_sample_id", sort=False):
            values = pd.to_numeric(group["causal_event_index"], errors="raise").to_numpy(dtype=np.int64)
            if values.tolist() != list(range(len(values))):
                raise ValueError(f"Bird-route r1 event sequence is not causal: {split}/{parent}")
        offsets = cache["event_offsets"].astype(np.int64)
        if offsets.shape != (len(ids) + 1,) or offsets[0] != 0 or offsets[-1] != len(cache["raw_dat_frame_indices_zero_based"]):
            raise ValueError(f"Bird-route r1 event offsets differ: {split}")
        if np.any(np.diff(offsets) < 0):
            raise ValueError(f"Bird-route r1 event offsets are not monotonic: {split}")
        report["splits"][split] = {
            "path": str(path.relative_to(context.project_root).as_posix()),
            "sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "event_count": int(len(ids)),
            "parent_count": int(parents.size and len(set(parents.tolist()))),
            "frame_count": int(len(cache["raw_dat_frame_indices_zero_based"])),
            "valid_event_count": int(cache["valid"].sum()),
            "reliable_frame_count": int(cache["frame_feature_reliable"].sum()),
            "label_counts": dict(Counter(aligned["label_name"].astype(str).tolist())),
            "unknown_event_count": int((aligned["label_name"].astype(str) == "UnknownTarget").sum()),
            "sample_ids_sha256": __import__("hashlib").sha256("\n".join(ids.tolist()).encode()).hexdigest(),
        }
        if split == context.config["data"]["test_split_name"]:
            report["test_access_count"] = int(len(ids))
    return report


__all__ = ["EVIDENCE_FIELDS", "audit_evidence", "prepare_evidence"]
