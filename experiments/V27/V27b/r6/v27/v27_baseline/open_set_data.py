"""Versioned mixed-pool preparation for frozen V27 open-set stages.

Classification owns only known-class train/validation views.  This module is
the explicit boundary that opens ``UnknownTarget`` after L1/L2 checkpoints
have frozen, creates V27-local feature caches for those mixed role pools, and
records every split, component and cache hash needed by later head training
and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from radar_ln3.runtime import canonical_sha256, resolve_project_path, sha256_file

from .base_binding import load_frozen_base_binding
from .runtime import V27Context


OPEN_SET_POOL_SCHEMA = "RADAR-LN3-V27-open-set-pool-registry-v1"
OPEN_SET_CACHE_MANIFEST_SCHEMA = "RADAR-LN3-V27-open-set-feature-cache-manifest-v1"
OPEN_SET_POOL_REGISTRY_PATH = Path(
    "src-v3/v27/v27_baseline/calibration/V27-open-set-pools.json"
)
OPEN_SET_CACHE_MANIFEST_PATH = Path(
    "src-v3/v27/v27_baseline/calibration/V27-open-set-feature-caches.json"
)
MIXED_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
_DEVELOPMENT_ROLE = "representation_development"
_CALIBRATION_ROLE = "calibration"


@dataclass(frozen=True)
class OpenSetPoolPaths:
    registry: Path
    cache_manifest: Path


def open_set_pool_paths(project_root: str | Path) -> OpenSetPoolPaths:
    root = Path(project_root).resolve()
    return OpenSetPoolPaths(
        registry=resolve_project_path(root, OPEN_SET_POOL_REGISTRY_PATH),
        cache_manifest=resolve_project_path(root, OPEN_SET_CACHE_MANIFEST_PATH),
    )


def _parent_frame(context: V27Context) -> pd.DataFrame:
    path = resolve_project_path(context.project_root, context.config.data.index_path, must_exist=True)
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "sample_id",
        "parent_sample_id",
        "provenance_component_id",
        "capture_id",
        "split_name",
        "label_name",
        "causal_event_index",
        "tr_feature_values_json",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V27 open-set index lacks required columns: {missing}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("V27 open-set index repeats sample ids")
    parent = frame[
        [
            "parent_sample_id",
            "provenance_component_id",
            "capture_id",
            "split_name",
            "label_name",
        ]
    ].drop_duplicates("parent_sample_id")
    if len(parent) != int(frame["parent_sample_id"].nunique()):
        raise ValueError("V27 open-set parents have inconsistent metadata")
    if not set(parent["label_name"].astype(str)).issuperset(MIXED_LABELS):
        raise ValueError("V27 open-set index lacks a mixed-label population")
    return frame




def _label_counts(rows: pd.DataFrame) -> dict[str, int]:
    counts = rows.groupby("label_name", sort=False)["parent_sample_id"].nunique()
    return {label: int(counts.get(label, 0)) for label in MIXED_LABELS}


def _role_record(
    context: V27Context,
    frame: pd.DataFrame,
    *,
    role: str,
    split_name: str,
    parent_ids: Sequence[str],
) -> dict[str, Any]:
    selected_ids = tuple(str(value) for value in parent_ids)
    if not selected_ids or len(set(selected_ids)) != len(selected_ids):
        raise ValueError("V27 open-set role parent ids must be non-empty and unique")
    rows = frame[
        frame["parent_sample_id"].astype(str).isin(selected_ids)
    ].copy()
    parent = rows[
        ["parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name"]
    ].drop_duplicates("parent_sample_id")
    if len(parent) != len(selected_ids):
        raise ValueError("V27 open-set role parent coverage differs")
    if set(parent["split_name"].astype(str)) != {split_name}:
        raise ValueError("V27 open-set role crosses a split boundary")
    if not set(parent["label_name"].astype(str)).issubset(MIXED_LABELS):
        raise ValueError("V27 open-set role has an unsupported label")
    ordered = list(selected_ids)
    positions = {value: index for index, value in enumerate(ordered)}
    observed = parent["parent_sample_id"].astype(str).tolist()
    if set(observed) != set(ordered):
        raise ValueError("V27 open-set role identity differs")
    parent = parent.assign(_order=parent["parent_sample_id"].astype(str).map(positions)).sort_values("_order")
    sample_rows = rows.assign(_order=rows["parent_sample_id"].astype(str).map(positions)).sort_values(
        ["_order", "causal_event_index"], kind="stable"
    )
    if sample_rows["sample_id"].astype(str).duplicated().any():
        raise ValueError("V27 open-set role repeats an event")
    return {
        "role": role,
        "split_name": split_name,
        "parent_ids": ordered,
        "parent_ids_sha256": canonical_sha256(ordered),
        "parent_count": len(ordered),
        "parent_label_counts": _label_counts(parent),
        "event_count": int(len(sample_rows)),
        "sample_ids": sample_rows["sample_id"].astype(str).tolist(),
        "sample_ids_sha256": canonical_sha256(sample_rows["sample_id"].astype(str).tolist()),
        "provenance_component_ids": parent["provenance_component_id"].astype(str).tolist(),
        "provenance_component_ids_sha256": canonical_sha256(
            parent["provenance_component_id"].astype(str).tolist()
        ),
        "capture_ids": parent["capture_id"].astype(str).tolist(),
        "capture_ids_sha256": canonical_sha256(parent["capture_id"].astype(str).tolist()),
    }




def verify_open_set_pool_registry(
    registry: Mapping[str, Any],
    *,
    context: V27Context,
) -> dict[str, Any]:
    payload = dict(registry)
    # ``write_v27_sealed_json`` adds a second integrity envelope around the
    # custom registry digest.  Verify that envelope first while the custom
    # ``registry_sha256`` field is still present; otherwise a persisted sealed
    # registry can never be loaded even though the in-memory value passed.
    payload_digest = payload.pop("payload_sha256", None)
    if payload_digest is not None:
        if not isinstance(payload_digest, str) or payload_digest != canonical_sha256(payload):
            raise ValueError("V27 open-set pool payload hash cannot be recomputed")
    declared = payload.pop("registry_sha256", None)
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("V27 open-set pool registry hash cannot be recomputed")
    if payload.get("schema_version") != OPEN_SET_POOL_SCHEMA or payload.get("status") != "prepared":
        raise ValueError("V27 open-set pool registry schema/status differs")
    binding = load_frozen_base_binding(context.project_root)
    if payload.get("classification_base_binding_sha256") != binding["binding_sha256"]:
        raise ValueError("V27 open-set pool registry frozen-base binding differs")
    if payload.get("index_sha256") != context.data_binding["hashes"]["index"]:
        raise ValueError("V27 open-set pool registry index differs")
    roles = payload.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {
        "representation_fit",
        _DEVELOPMENT_ROLE,
        _CALIBRATION_ROLE,
    }:
        raise ValueError("V27 open-set pool roles differ")
    role_components: dict[str, set[str]] = {}
    role_parents: dict[str, set[str]] = {}
    frame = _parent_frame(context)
    for name, raw in roles.items():
        record = dict(raw)
        expected_split = (
            context.config.data.train_split_name
            if name in {"representation_fit", _DEVELOPMENT_ROLE}
            else context.config.data.validation_split_name
        )
        if record.get("role") != name or record.get("split_name") != expected_split:
            raise ValueError("V27 open-set role/split differs")
        ids = [str(value) for value in record.get("parent_ids") or ()]
        components = [str(value) for value in record.get("provenance_component_ids") or ()]
        if not ids or len(ids) != len(set(ids)) or len(ids) != len(components):
            raise ValueError("V27 open-set role parent/component binding differs")
        if record.get("parent_ids_sha256") != canonical_sha256(ids):
            raise ValueError("V27 open-set role parent hash differs")
        if record.get("provenance_component_ids_sha256") != canonical_sha256(components):
            raise ValueError("V27 open-set role component hash differs")
        if record.get("capture_ids_sha256") != canonical_sha256(
            [str(value) for value in record.get("capture_ids") or ()]
        ):
            raise ValueError("V27 open-set role capture hash differs")
        counts = dict(record.get("parent_label_counts") or {})
        if set(counts) != set(MIXED_LABELS) or any(int(counts[name]) <= 0 for name in MIXED_LABELS):
            raise ValueError("V27 open-set role label support differs")
        # Rebuild the complete role record from the current index.  Checking
        # only a self-consistent digest would allow an edited registry to
        # recompute its own hash and still pass validation.
        rebuilt = _role_record(
            context,
            frame,
            role=name,
            split_name=expected_split,
            parent_ids=ids,
        )
        for field in (
            "role",
            "split_name",
            "parent_ids",
            "parent_ids_sha256",
            "parent_count",
            "parent_label_counts",
            "event_count",
            "sample_ids",
            "sample_ids_sha256",
            "provenance_component_ids",
            "provenance_component_ids_sha256",
            "capture_ids",
            "capture_ids_sha256",
        ):
            if record.get(field) != rebuilt.get(field):
                raise ValueError(f"V27 open-set role differs from current index: {name}/{field}")
        role_components[name] = set(components)
        role_parents[name] = set(ids)
    names = tuple(role_components)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if role_components[left] & role_components[right]:
                raise ValueError(f"V27 open-set component overlap: {left}/{right}")
            if role_parents[left] & role_parents[right]:
                raise ValueError(f"V27 open-set parent overlap: {left}/{right}")
    result = {**payload, "registry_sha256": declared}
    if payload_digest is not None:
        result["payload_sha256"] = payload_digest
    return result






def _read_feature_cache(
    path: Path,
    *,
    width: int,
    feature_name: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a compact versioned cache into an exact sample-id mapping."""

    if not path.is_file():
        raise FileNotFoundError(f"V27 open-set {feature_name} source cache is missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"V27 open-set {feature_name} source cache fields differ")
        ids = [str(value) for value in archive["sample_ids"].tolist()]
        values = np.asarray(archive["values"])
        valid = np.asarray(archive["valid"])
    if (
        not ids
        or len(ids) != len(set(ids))
        or values.shape != (len(ids), width)
        or values.dtype != np.float32
        or valid.shape not in {(len(ids),), (len(ids), width)}
        or valid.dtype != np.bool_
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError(f"V27 open-set {feature_name} source cache schema/value differs")
    event_valid = valid if valid.ndim == 1 else valid.all(axis=1)
    if np.any(values[~event_valid] != 0.0):
        raise ValueError(f"V27 open-set {feature_name} source cache invalid rows differ")
    positions = {sample_id: index for index, sample_id in enumerate(ids)}
    return {
        sample_id: values[index].copy()
        for index, sample_id in enumerate(ids)
    }, {
        "path": path,
        "sha256": sha256_file(path),
        "sample_ids": ids,
        "positions": positions,
        "valid": valid.copy(),
        "event_valid": event_valid.copy(),
    }






















def load_open_set_pool_artifacts(context: V27Context) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and verify the prepared role/cache pair for later stages."""

    from .runtime import read_v27_sealed_json

    paths = open_set_pool_paths(context.project_root)
    registry = verify_open_set_pool_registry(read_v27_sealed_json(paths.registry), context=context)
    cache = read_v27_sealed_json(paths.cache_manifest)
    payload = dict(cache)
    declared = payload.pop("cache_manifest_sha256", None)
    sealed = payload.pop("payload_sha256", None)
    if sealed != canonical_sha256({**payload, "cache_manifest_sha256": declared}):
        raise ValueError("V27 open-set cache manifest payload hash differs")
    if not isinstance(declared, str) or declared != canonical_sha256(payload):
        raise ValueError("V27 open-set cache manifest hash differs")
    if (
        payload.get("schema_version") != OPEN_SET_CACHE_MANIFEST_SCHEMA
        or payload.get("status") != "prepared"
        or payload.get("pool_registry_sha256") != registry["registry_sha256"]
        or payload.get("frozen_base_binding_sha256")
        != registry["classification_base_binding_sha256"]
    ):
        raise ValueError("V27 open-set cache manifest binding differs")
    roles = payload.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != set(registry["roles"]):
        raise ValueError("V27 open-set cache roles differ")
    for role, record in dict(roles).items():
        if record.get("sample_ids_sha256") != registry["roles"][role]["sample_ids_sha256"]:
            raise ValueError("V27 open-set cache sample binding differs")
        expected_ids = [str(value) for value in registry["roles"][role]["sample_ids"]]
        widths = {"course": 1, "rd_physical": 4, "micro_doppler": 8}
        for key in ("course", "rd_physical", "micro_doppler"):
            entry = dict(record.get(key) or {})
            target = resolve_project_path(context.project_root, str(entry.get("path", "")), must_exist=True)
            if sha256_file(target) != entry.get("sha256"):
                raise ValueError(f"V27 open-set {role}/{key} cache hash differs")
            values, metadata = _read_feature_cache(
                target, width=widths[key], feature_name=f"{role}/{key}"
            )
            if metadata["sample_ids"] != expected_ids or set(values) != set(expected_ids):
                raise ValueError(f"V27 open-set {role}/{key} cache identity differs")
    result = {**payload, "cache_manifest_sha256": declared, "payload_sha256": sealed}
    return registry, result




__all__ = ["MIXED_LABELS", "OPEN_SET_CACHE_MANIFEST_PATH", "OPEN_SET_CACHE_MANIFEST_SCHEMA", "OPEN_SET_POOL_REGISTRY_PATH", "OPEN_SET_POOL_SCHEMA", "load_open_set_pool_artifacts", "open_set_pool_paths", "verify_open_set_pool_registry"]