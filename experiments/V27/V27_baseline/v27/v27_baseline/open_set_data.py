"""Versioned mixed-pool preparation for frozen V27 open-set stages.

Classification owns only known-class train/validation views.  This module is
the explicit boundary that opens ``UnknownTarget`` after L1/L2 checkpoints
have frozen, creates V27-local feature caches for those mixed role pools, and
records every split, component and cache hash needed by later head training
and calibration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _CanonicalFrameProvenance,
    _write_npz_new_or_identical,
)
from build.micro_doppler_v18 import _build_split
from radar_ln3.known_gate_features import extract_causal_course_circular_incoherence
from radar_ln3.runtime import (
    canonical_sha256,
    parse_bool,
    relative_project_path,
    resolve_project_path,
    sha256_file,
)

from .base_binding import load_frozen_base_binding
from .runtime import V27Context, write_v27_sealed_json


OPEN_SET_POOL_SCHEMA = "RADAR-LN3-V27-open-set-pool-registry-v1"
OPEN_SET_CACHE_MANIFEST_SCHEMA = "RADAR-LN3-V27-open-set-feature-cache-manifest-v1"
OPEN_SET_POOL_REGISTRY_PATH = Path(
    "src-v3/v27/v27_baseline/calibration/V27-open-set-pools.json"
)
OPEN_SET_CACHE_MANIFEST_PATH = Path(
    "src-v3/v27/v27_baseline/calibration/V27-open-set-feature-caches.json"
)
OPEN_SET_FEATURE_DIR = Path("src-v3/v27/v27_baseline/calibration/features")
MIXED_LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
_DEVELOPMENT_ROLE = "representation_development"
_CALIBRATION_ROLE = "calibration"
_TRAIN_KNOWN_COURSE_CACHE = Path(
    "data/processed/RADAR-LN3-V26c-cq08-known-gate-all-known-train-course-source-cache.npz"
)
_VAL_MIXED_COURSE_CACHE = Path(
    "data/processed/RADAR-LN3-V26-cq08-known-gate-val-mixed-open-set-course-cache.npz"
)
_TRAIN_KNOWN_PHYSICAL_CACHE = Path(
    "data/processed/RADAR-LN3-V26c-cq08-dbb-specialist-all-known-inference-rd-physical4-source-cache.npz"
)
_TRAIN_KNOWN_MICRO_CACHE = Path(
    "data/processed/RADAR-LN3-V26c-cq08-dbb-specialist-all-known-inference-micro-doppler8-source-cache.npz"
)
_VAL_MIXED_PHYSICAL_CACHE = Path(
    "data/processed/RADAR-LN3-V26b-cq08-l2-val-mixed-open-set-rd-physical4-cache.npz"
)
_VAL_MIXED_MICRO_CACHE = Path(
    "data/processed/RADAR-LN3-V26b-cq08-l2-val-mixed-open-set-micro-doppler8-cache.npz"
)
_UNKNOWN_TRAIN_PHYSICAL_CACHE = OPEN_SET_FEATURE_DIR / (
    "V27-train-unknown-all-rd-physical4-source-cache.npz"
)
_UNKNOWN_TRAIN_MICRO_CACHE = OPEN_SET_FEATURE_DIR / (
    "V27-train-unknown-all-micro-doppler8-source-cache.npz"
)
_V16_PROVENANCE_PATH = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
_CANONICAL_FRAME_INDEX_PATH = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)


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


def _hash_partition(component_id: str) -> int:
    """Return a deterministic component assignment independent of row order."""

    digest = hashlib.sha256(str(component_id).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


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


def build_open_set_pool_registry(context: V27Context) -> dict[str, Any]:
    """Create disjoint development/calibration pools without reading test.

    Train components are assigned by a stable SHA-256 parity.  Unknown is
    exposed only in the development role; validation is retained exclusively
    for calibration and may never be used to optimize trajectory-head weights.
    """

    # A frozen classifier binding is a prerequisite for exposing Unknown.
    # This check also prevents an operator from creating a pool against an
    # unfrozen or silently replaced classification endpoint.
    frozen_binding = load_frozen_base_binding(context.project_root)
    frame = _parent_frame(context)
    parent = frame[
        ["parent_sample_id", "provenance_component_id", "capture_id", "split_name", "label_name"]
    ].drop_duplicates("parent_sample_id")
    train = parent[
        parent["split_name"].astype(str).eq(context.config.data.train_split_name)
        & parent["label_name"].astype(str).isin(MIXED_LABELS)
    ].copy()
    calibration = parent[
        parent["split_name"].astype(str).eq(context.config.data.validation_split_name)
        & parent["label_name"].astype(str).isin(MIXED_LABELS)
    ].copy()
    if train.empty or calibration.empty:
        raise ValueError("V27 open-set pool source is empty")
    train["_bucket"] = train["provenance_component_id"].astype(str).map(_hash_partition)
    development = train[train["_bucket"].mod(2).eq(0)].copy()
    representation_fit = train[train["_bucket"].mod(2).eq(1)].copy()
    # The head requires both positive and negative knownness examples in its
    # held-out development partition.  If hash parity creates a degenerate
    # low-support label group, fail closed instead of moving an arbitrary
    # individual parent across the component boundary.
    for name, selected in (
        ("representation_fit", representation_fit),
        (_DEVELOPMENT_ROLE, development),
        (_CALIBRATION_ROLE, calibration),
    ):
        counts = _label_counts(selected)
        if any(counts[label] <= 0 for label in MIXED_LABELS):
            raise ValueError(f"V27 {name} pool lacks a mixed label: {counts}")
    records = {
        "representation_fit": _role_record(
            context,
            frame,
            role="representation_fit",
            split_name=context.config.data.train_split_name,
            parent_ids=representation_fit.sort_values("parent_sample_id")["parent_sample_id"].astype(str).tolist(),
        ),
        _DEVELOPMENT_ROLE: _role_record(
            context,
            frame,
            role=_DEVELOPMENT_ROLE,
            split_name=context.config.data.train_split_name,
            parent_ids=development.sort_values("parent_sample_id")["parent_sample_id"].astype(str).tolist(),
        ),
        _CALIBRATION_ROLE: _role_record(
            context,
            frame,
            role=_CALIBRATION_ROLE,
            split_name=context.config.data.validation_split_name,
            parent_ids=calibration.sort_values("parent_sample_id")["parent_sample_id"].astype(str).tolist(),
        ),
    }
    component_sets = {
        name: set(record["provenance_component_ids"])
        for name, record in records.items()
    }
    if component_sets["representation_fit"] & component_sets[_DEVELOPMENT_ROLE]:
        raise ValueError("V27 representation fit/development components overlap")
    if component_sets["representation_fit"] & component_sets[_CALIBRATION_ROLE]:
        raise ValueError("V27 representation fit/calibration components overlap")
    if component_sets[_DEVELOPMENT_ROLE] & component_sets[_CALIBRATION_ROLE]:
        raise ValueError("V27 development/calibration components overlap")
    payload: dict[str, Any] = {
        "schema_version": OPEN_SET_POOL_SCHEMA,
        "experiment_id": "V27",
        "status": "prepared",
        "classification_base_binding_sha256": frozen_binding["binding_sha256"],
        "index_path": relative_project_path(
            context.project_root,
            resolve_project_path(context.project_root, context.config.data.index_path),
        ),
        "index_sha256": context.data_binding["hashes"]["index"],
        "pool_partition": {
            "unit": "provenance_component_id",
            "method": "sha256_first_64_bits_mod_2",
            "representation_fit_bucket": 1,
            "representation_development_bucket": 0,
            "unknown_allowed_roles": ["representation_fit", _DEVELOPMENT_ROLE, _CALIBRATION_ROLE],
            "unknown_forbidden_roles": ["classification_training", "normalization", "checkpoint_selection"],
            "test_payload_access_count": 0,
        },
        "roles": records,
        "role_policy": {
            "representation_fit": "trajectory-head optimization only; classification frozen",
            _DEVELOPMENT_ROLE: "head selection and ranking diagnostic only; no calibration fitting",
            _CALIBRATION_ROLE: "dynamic calibration and L3 fitting only; no head optimization",
            "test": "sealed",
        },
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


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


def _role_rows(context: V27Context, role: Mapping[str, Any]) -> pd.DataFrame:
    frame = _parent_frame(context)
    positions = {str(value): index for index, value in enumerate(role["parent_ids"])}
    rows = frame[frame["parent_sample_id"].astype(str).isin(positions)].copy()
    rows["_order"] = rows["parent_sample_id"].astype(str).map(positions)
    rows = rows.sort_values(["_order", "causal_event_index"], kind="stable").reset_index(drop=True)
    expected = [str(value) for value in role["sample_ids"]]
    actual = rows["sample_id"].astype(str).tolist()
    if actual != expected:
        raise ValueError("V27 open-set role sample ordering differs")
    return rows


def _course_cache_arrays(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    values = np.zeros((len(rows), 1), dtype=np.float32)
    valid = np.zeros((len(rows), 1), dtype=np.bool_)
    cursor = 0
    for parent_id, parent in rows.groupby("parent_sample_id", sort=False):
        events = parent["causal_event_index"].astype(int).tolist()
        if events != list(range(len(parent))):
            raise ValueError(f"V27 open-set course parent event order differs: {parent_id}")
        courses: list[float | None] = []
        for raw in parent["tr_feature_values_json"].astype(str):
            parsed = json.loads(raw)
            if not isinstance(parsed, Mapping):
                raise ValueError("V27 open-set course feature payload is not an object")
            value = parsed.get("course_deg")
            courses.append(None if value is None else float(value))
        item_values, item_valid = extract_causal_course_circular_incoherence(courses)
        stop = cursor + len(parent)
        values[cursor:stop] = item_values
        valid[cursor:stop] = item_valid
        cursor = stop
    if cursor != len(rows) or np.any(values[~valid] != 0.0):
        raise ValueError("V27 open-set course cache construction differs")
    return {
        "sample_ids": rows["sample_id"].astype(str).to_numpy(dtype=np.str_),
        "values": values,
        "valid": valid,
    }


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


def _source_caches_for_role(
    context: V27Context,
    *,
    role: str,
) -> tuple[dict[str, Path], str]:
    """Select audited prior caches without changing their feature values."""

    if role in {"representation_fit", _DEVELOPMENT_ROLE}:
        return {
            "course": context.project_root / _TRAIN_KNOWN_COURSE_CACHE,
            "physical": context.project_root / _TRAIN_KNOWN_PHYSICAL_CACHE,
            "micro": context.project_root / _TRAIN_KNOWN_MICRO_CACHE,
        }, "train_known_values_reused_unknown_course_generated"
    if role == _CALIBRATION_ROLE:
        return {
            "course": context.project_root / _VAL_MIXED_COURSE_CACHE,
            "physical": context.project_root / _VAL_MIXED_PHYSICAL_CACHE,
            "micro": context.project_root / _VAL_MIXED_MICRO_CACHE,
        }, "val_mixed_values_reused"
    raise ValueError(f"V27 unsupported open-set role: {role}")


def _unknown_train_l2_source_caches(
    context: V27Context,
    *,
    rows: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Materialize label-free physical4/stable8 values for train Unknown only.

    This uses the exact V18 stable8 extractor, but the label is used solely to
    choose the isolated role population.  The resulting arrays are checked
    against RD presence and written under V27 provenance before any head sees
    them.  It prevents feature availability from becoming a proxy Unknown
    label in the frozen-backbone training pool.
    """

    requested_unknown_ids = set(
        rows.loc[
            rows["label_name"].astype(str).eq("UnknownTarget"), "sample_id"
        ].astype(str)
    )
    if not requested_unknown_ids:
        return {
            "sample_ids": np.asarray([], dtype=np.str_),
            "physical_values": np.zeros((0, 4), dtype=np.float32),
            "summary_values": np.zeros((0, 8), dtype=np.float32),
            "valid": np.zeros((0,), dtype=np.bool_),
        }, {"status": "not_required"}
    root = context.project_root
    all_index = _parent_frame(context)
    unknown = all_index[
        all_index["split_name"].astype(str).eq(context.config.data.train_split_name)
        & all_index["label_name"].astype(str).eq("UnknownTarget")
    ].copy()
    if unknown.empty:
        raise ValueError("V27 train Unknown source population is empty")
    unknown = unknown.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    expected_ids = unknown["sample_id"].astype(str).to_numpy(dtype=np.str_)
    expected_valid = unknown["rd_present"].map(
        lambda value: parse_bool(value, field="index.rd_present")
    ).to_numpy(dtype=np.bool_)
    unknown_physical_path = root / _UNKNOWN_TRAIN_PHYSICAL_CACHE
    unknown_micro_path = root / _UNKNOWN_TRAIN_MICRO_CACHE
    physical_exists = unknown_physical_path.is_file()
    micro_exists = unknown_micro_path.is_file()
    if physical_exists != micro_exists:
        raise FileNotFoundError(
            "V27 train Unknown physical/micro source cache pair differs"
        )

    statistics: Mapping[str, Any] | None = None
    provenance_path = root / _V16_PROVENANCE_PATH
    canonical_path = root / _CANONICAL_FRAME_INDEX_PATH
    if not physical_exists:
        if not provenance_path.is_file() or not canonical_path.is_file():
            raise FileNotFoundError(
                "V27 train Unknown extraction lacks V16 provenance/frame index"
            )
        source = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance = {
            str(item["parent_sample_id"]): dict(item)
            for item in list(source.get("records") or ())
        }
        if len(provenance) != int(source.get("parent_count", -1)):
            raise ValueError("V27 train Unknown V16 provenance is incomplete")
        canonical = _CanonicalFrameProvenance(root, canonical_path)
        arrays, statistics = _build_split(
            root,
            context.config.data.train_split_name,
            unknown,
            provenance,
            canonical,
            allowed_labels=("UnknownTarget",),
            include_evidence=False,
        )
        if (
            not np.array_equal(arrays["sample_ids"], expected_ids)
            or arrays["physical_values"].shape != (len(unknown), 4)
            or arrays["summary_values"].shape != (len(unknown), 8)
            or not np.array_equal(arrays["valid"], expected_valid)
            or np.any(arrays["physical_values"][~arrays["valid"]] != 0.0)
            or np.any(arrays["summary_values"][~arrays["valid"]] != 0.0)
        ):
            raise ValueError("V27 train Unknown stable8 extraction differs")
        physical_payload = (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["physical_values"]),
            ("valid", arrays["valid"]),
        )
        micro_payload = (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["summary_values"]),
            ("valid", arrays["valid"]),
        )
        _write_npz_new_or_identical(unknown_physical_path, physical_payload)
        _write_npz_new_or_identical(unknown_micro_path, micro_payload)
    # Both roles draw subsets from one globally ordered cache.  Existing files
    # are deliberately loaded and validated without rerunning raw extraction.
    with np.load(unknown_physical_path, allow_pickle=False) as physical, np.load(
        unknown_micro_path, allow_pickle=False
    ) as micro:
        if set(physical.files) != {"sample_ids", "values", "valid"} or set(
            micro.files
        ) != {"sample_ids", "values", "valid"}:
            raise ValueError("V27 train Unknown physical/micro source cache schema differs")
        physical_ids = np.asarray(physical["sample_ids"])
        micro_ids = np.asarray(micro["sample_ids"])
        physical_values = np.asarray(physical["values"])
        micro_values = np.asarray(micro["values"])
        physical_valid = np.asarray(physical["valid"])
        micro_valid = np.asarray(micro["valid"])
    if (
        physical_ids.ndim != 1
        or micro_ids.ndim != 1
        or not np.array_equal(physical_ids, micro_ids)
        or not np.array_equal(physical_valid, micro_valid)
        or physical_values.shape != (len(physical_ids), 4)
        or micro_values.shape != (len(micro_ids), 8)
        or physical_values.dtype != np.float32
        or micro_values.dtype != np.float32
        or physical_valid.shape != (len(physical_ids),)
        or physical_valid.dtype != np.bool_
        or not np.isfinite(physical_values).all()
        or not np.isfinite(micro_values).all()
        or np.any(physical_values < 0.0)
        or np.any(physical_values > 1.0)
        or np.any(micro_values < 0.0)
        or np.any(micro_values > 1.0)
    ):
        raise ValueError("V27 train Unknown physical/micro source cache values differ")
    cache_ids = np.asarray([str(value) for value in physical_ids.tolist()], dtype=np.str_)
    if not np.array_equal(cache_ids, expected_ids) or not np.array_equal(
        physical_valid, expected_valid
    ):
        raise ValueError("V27 train Unknown source cache identity/validity differs from index")
    if np.any(physical_values[~physical_valid] != 0.0) or np.any(
        micro_values[~physical_valid] != 0.0
    ):
        raise ValueError("V27 train Unknown invalid source values are not zero")
    source_arrays = {
        "sample_ids": cache_ids,
        "physical_values": physical_values,
        "summary_values": micro_values,
        "valid": physical_valid,
    }
    return source_arrays, {
        "status": "label_free_values_materialized",
        "physical_path": relative_project_path(context.project_root, unknown_physical_path),
        "physical_sha256": sha256_file(unknown_physical_path),
        "micro_path": relative_project_path(context.project_root, unknown_micro_path),
        "micro_sha256": sha256_file(unknown_micro_path),
        "v16_provenance_path": relative_project_path(context.project_root, provenance_path),
        "v16_provenance_sha256": sha256_file(provenance_path),
        "canonical_frame_index_path": relative_project_path(context.project_root, canonical_path),
        "canonical_frame_index_sha256": sha256_file(canonical_path),
        "statistics": dict(statistics or {"status": "reused_existing_cache"}),
        "unknown_label_used_for_feature_value": False,
    }


def _mixed_feature_arrays(
    context: V27Context,
    *,
    role: str,
    rows: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a mixed cache by exact reuse plus causal Unknown course values.

    The V27 baseline does not re-run raw-IQ stable8 extraction for known
    records: existing V26c/V26b values are verified by sample id and copied
    exactly.  Train Unknown values come from one explicit, label-free V18
    materialization cache and are reused by both representation roles.
    Unknown course remains fully causal and label-free.
    """

    sources, source_policy = _source_caches_for_role(context, role=role)
    course_values, course_meta = _read_feature_cache(
        sources["course"], width=1, feature_name="course"
    )
    physical_values, physical_meta = _read_feature_cache(
        sources["physical"], width=4, feature_name="physical4"
    )
    micro_values, micro_meta = _read_feature_cache(
        sources["micro"], width=8, feature_name="stable8"
    )
    identifiers = rows["sample_id"].astype(str).tolist()
    labels = rows["label_name"].astype(str).tolist()
    unknown_ids = {
        sample_id for sample_id, label in zip(identifiers, labels, strict=True)
        if label == "UnknownTarget"
    }
    unknown_l2, unknown_l2_provenance = (
        _unknown_train_l2_source_caches(context, rows=rows)
        if role in {"representation_fit", _DEVELOPMENT_ROLE}
        else (
            {
                "sample_ids": np.asarray([], dtype=np.str_),
                "physical_values": np.zeros((0, 4), dtype=np.float32),
                "summary_values": np.zeros((0, 8), dtype=np.float32),
                "valid": np.zeros((0,), dtype=np.bool_),
            },
            {"status": "not_required_val_mixed_cache_already_covers_unknown"},
        )
    )
    unknown_l2_positions = {
        str(sample_id): index
        for index, sample_id in enumerate(unknown_l2["sample_ids"].tolist())
    }
    course = np.zeros((len(rows), 1), dtype=np.float32)
    course_valid = np.zeros((len(rows), 1), dtype=np.bool_)
    physical = np.zeros((len(rows), 4), dtype=np.float32)
    micro = np.zeros((len(rows), 8), dtype=np.float32)
    physical_valid = np.zeros((len(rows),), dtype=np.bool_)
    micro_valid = np.zeros((len(rows),), dtype=np.bool_)
    generated_unknown_course = _course_cache_arrays(
        rows[rows["label_name"].astype(str).eq("UnknownTarget")].copy()
    ) if unknown_ids else None
    generated_course = (
        {
            str(sample_id): (generated_unknown_course["values"][index], generated_unknown_course["valid"][index])
            for index, sample_id in enumerate(generated_unknown_course["sample_ids"].tolist())
        }
        if generated_unknown_course is not None
        else {}
    )
    for index, sample_id in enumerate(identifiers):
        if sample_id in course_values:
            course[index] = course_values[sample_id]
            source_index = course_meta["positions"][sample_id]
            course_valid[index] = course_meta["valid"][source_index]
        elif sample_id in generated_course:
            course[index], course_valid[index] = generated_course[sample_id]
        else:
            raise ValueError(f"V27 open-set role lacks course value: {sample_id}")
        if sample_id in physical_values:
            source_index = physical_meta["positions"][sample_id]
            physical[index] = physical_values[sample_id]
            physical_valid[index] = bool(physical_meta["event_valid"][source_index])
        elif sample_id in unknown_l2_positions:
            source_index = unknown_l2_positions[sample_id]
            physical[index] = unknown_l2["physical_values"][source_index]
            physical_valid[index] = bool(unknown_l2["valid"][source_index])
        elif sample_id not in unknown_ids:
            raise ValueError(f"V27 open-set role lacks known physical4 value: {sample_id}")
        if sample_id in micro_values:
            source_index = micro_meta["positions"][sample_id]
            micro[index] = micro_values[sample_id]
            micro_valid[index] = bool(micro_meta["event_valid"][source_index])
        elif sample_id in unknown_l2_positions:
            source_index = unknown_l2_positions[sample_id]
            micro[index] = unknown_l2["summary_values"][source_index]
            micro_valid[index] = bool(unknown_l2["valid"][source_index])
        elif sample_id not in unknown_ids:
            raise ValueError(f"V27 open-set role lacks known stable8 value: {sample_id}")
    if (
        np.any(course[~course_valid] != 0.0)
        or np.any(physical[~physical_valid] != 0.0)
        or np.any(micro[~micro_valid] != 0.0)
    ):
        raise ValueError("V27 open-set mixed cache invalid values differ")
    # RD can be genuinely unavailable for a short suffix of an otherwise
    # valid known/Unknown track.  Require every parent to retain at least one
    # valid L2 observation; a parent with no physical evidence belongs to an
    # upstream quality/data issue, not an OOD target for this baseline.
    parent_valid = rows[["parent_sample_id"]].copy()
    parent_valid["l2_valid"] = physical_valid & micro_valid
    if not bool(parent_valid.groupby("parent_sample_id", sort=False)["l2_valid"].any().all()):
        raise ValueError("V27 open-set role has a parent with no L2 evidence")
    return {
        "sample_ids": np.asarray(identifiers, dtype=np.str_),
        "course_values": course,
        "course_valid": course_valid,
        "physical_values": physical,
        "micro_values": micro,
        "physical_valid": physical_valid,
        "micro_valid": micro_valid,
    }, {
        "policy": source_policy,
        "source_caches": {
            name: {
                "path": relative_project_path(context.project_root, metadata["path"]),
                "sha256": metadata["sha256"],
            }
            for name, metadata in (
                ("course", course_meta),
                ("physical", physical_meta),
                ("micro", micro_meta),
            )
        },
        "unknown_feature_handling": {
            "course": "causal value generated from processed TR only when absent from source cache",
            "physical4": "train Unknown values use one validated label-free V18 materialization cache",
            "stable8": "train Unknown values use one validated label-free V18 materialization cache",
            "unknown_label_used_for_feature_value": False,
        },
        "unknown_l2_extraction": unknown_l2_provenance,
        "unknown_event_count": int(sum(label == "UnknownTarget" for label in labels)),
        "unknown_physical_valid_count": int(
            physical_valid[np.asarray([label == "UnknownTarget" for label in labels])].sum()
        ),
        "unknown_micro_valid_count": int(
            micro_valid[np.asarray([label == "UnknownTarget" for label in labels])].sum()
        ),
    }


def _cache_paths_for_role(role: str) -> dict[str, Path]:
    stem = f"V27-{role}-mixed"
    return {
        "course": OPEN_SET_FEATURE_DIR / f"{stem}-course-cache.npz",
        "physical": OPEN_SET_FEATURE_DIR / f"{stem}-rd-physical4-cache.npz",
        "micro": OPEN_SET_FEATURE_DIR / f"{stem}-micro-doppler8-cache.npz",
    }


def _resolved_cache_paths_for_role(
    context: V27Context,
    *,
    role: str,
) -> dict[str, Path]:
    """Resolve every generated cache below the verified project root."""

    return {
        name: resolve_project_path(context.project_root, path)
        for name, path in _cache_paths_for_role(role).items()
    }


def _reuse_reference_l2_values(
    context: V27Context,
    *,
    role: str,
    arrays: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Prove V27 caches retain frozen known values exactly where applicable."""

    if role not in {"representation_fit", _CALIBRATION_ROLE}:
        return {"status": "not_applicable_development_holdout"}
    source, _ = _source_caches_for_role(context, role=role)
    physical_path = source["physical"]
    micro_path = source["micro"]
    with np.load(physical_path, allow_pickle=False) as physical, np.load(micro_path, allow_pickle=False) as micro:
        reference_ids = [str(value) for value in physical["sample_ids"].tolist()]
        if reference_ids != [str(value) for value in micro["sample_ids"].tolist()]:
            raise ValueError("V27 frozen L2 reference physical/micro ids differ")
        positions = {str(value): index for index, value in enumerate(arrays["sample_ids"].tolist())}
        selected_reference_ids = [value for value in reference_ids if value in positions]
        if not selected_reference_ids:
            raise ValueError("V27 mixed cache has no frozen L2 reference overlap")
        reference_positions = {
            value: index for index, value in enumerate(reference_ids)
        }
        index = np.asarray([positions[value] for value in selected_reference_ids], dtype=np.int64)
        source_index = np.asarray(
            [reference_positions[value] for value in selected_reference_ids], dtype=np.int64
        )
        if not (
            np.array_equal(arrays["physical_values"][index], physical["values"][source_index])
            and np.array_equal(arrays["micro_values"][index], micro["values"][source_index])
            and np.array_equal(arrays["physical_valid"][index], physical["valid"][source_index])
            and np.array_equal(arrays["micro_valid"][index], micro["valid"][source_index])
        ):
            raise ValueError("V27 mixed L2 cache differs from frozen known cache")
    return {
        "status": "exact_match",
        "reference_physical_path": relative_project_path(
            context.project_root, physical_path
        ),
        "reference_micro_path": relative_project_path(
            context.project_root, micro_path
        ),
        "known_event_count": len(selected_reference_ids),
        "source_known_event_count": len(reference_ids),
    }


def build_open_set_feature_caches(
    context: V27Context,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize role-specific mixed course/physical4/stable8 caches."""

    checked = verify_open_set_pool_registry(registry, context=context)
    results: dict[str, Any] = {}
    for role, raw_role in dict(checked["roles"]).items():
        record = dict(raw_role)
        rows = _role_rows(context, record)
        cache_paths = _resolved_cache_paths_for_role(context, role=role)
        arrays, provenance = _mixed_feature_arrays(context, role=role, rows=rows)
        _write_npz_new_or_identical(
            cache_paths["course"],
            (
                ("sample_ids", arrays["sample_ids"]),
                ("values", arrays["course_values"]),
                ("valid", arrays["course_valid"]),
            ),
        )
        _write_npz_new_or_identical(
            cache_paths["physical"],
            (
                ("sample_ids", arrays["sample_ids"]),
                ("values", arrays["physical_values"]),
                ("valid", arrays["physical_valid"]),
            ),
        )
        _write_npz_new_or_identical(
            cache_paths["micro"],
            (
                ("sample_ids", arrays["sample_ids"]),
                ("values", arrays["micro_values"]),
                ("valid", arrays["micro_valid"]),
            ),
        )
        results[role] = {
            "role_registry_sha256": checked["registry_sha256"],
            "sample_ids_sha256": record["sample_ids_sha256"],
            "course": {
                "path": relative_project_path(context.project_root, cache_paths["course"]),
                "sha256": sha256_file(cache_paths["course"]),
                "valid_value_count": int(arrays["course_valid"].sum()),
                "invalid_value_count": int((~arrays["course_valid"]).sum()),
            },
            "rd_physical": {
                "path": relative_project_path(context.project_root, cache_paths["physical"]),
                "sha256": sha256_file(cache_paths["physical"]),
            },
            "micro_doppler": {
                "path": relative_project_path(context.project_root, cache_paths["micro"]),
                "sha256": sha256_file(cache_paths["micro"]),
            },
            "feature_provenance": provenance,
            "frozen_l2_reference_match": _reuse_reference_l2_values(
                context, role=role, arrays=arrays
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": OPEN_SET_CACHE_MANIFEST_SCHEMA,
        "experiment_id": "V27",
        "status": "prepared",
        "pool_registry_sha256": checked["registry_sha256"],
        "frozen_base_binding_sha256": checked["classification_base_binding_sha256"],
        "feature_extractor": {
            "course": "V21e causal_prefix_course_circular_incoherence",
            "l2": "V18 current-event RD physical4 plus stable8 micro-Doppler",
            "unknown_label_used_for_feature_value": False,
            "test_payload_access_count": 0,
        },
        "roles": results,
    }
    payload["cache_manifest_sha256"] = canonical_sha256(payload)
    return payload


def _write_if_identical(path: Path, payload: Mapping[str, Any]) -> Path:
    from .runtime import read_v27_sealed_json

    if path.exists():
        existing = read_v27_sealed_json(path)
        if {
            key: value for key, value in existing.items() if key != "payload_sha256"
        } != dict(payload):
            raise FileExistsError(f"refusing to replace a different V27 open-set artifact: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    write_v27_sealed_json(path, dict(payload))
    return path


def _replace_prepared_cache_manifest(
    context: V27Context,
    path: Path,
    payload: Mapping[str, Any],
) -> Path:
    """Refresh a stale *prepared* cache manifest while preserving history.

    A cache manifest is derived metadata, not a fitted model or threshold.  A
    previous prepared manifest may describe an older extractor policy (as the
    first V27 draft did).  Archive that exact artifact before replacing it;
    fitted L3 artifacts are never handled by this helper.
    """

    from .runtime import read_v27_sealed_json

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_v27_sealed_json(path, dict(payload))
        return path
    existing = read_v27_sealed_json(path)
    if existing == dict(payload) or {
        key: value for key, value in existing.items() if key != "payload_sha256"
    } == dict(payload):
        return path
    if existing.get("status") != "prepared":
        raise FileExistsError("refusing to replace a non-prepared V27 cache manifest")
    archive_dir = (
        context.project_root
        / "artifacts"
        / "archive"
        / "v27"
        / "open-set-manifests"
    )
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = archive_dir / f"{path.name}.stale-before-{sha256_file(path)[:16]}"
    if archived.exists():
        if sha256_file(archived) != sha256_file(path):
            raise FileExistsError(f"V27 cache manifest archive collision: {archived}")
    else:
        path.replace(archived)
    write_v27_sealed_json(path, dict(payload))
    return path


def prepare_open_set_pools(context: V27Context) -> dict[str, Any]:
    """Prepare immutable mixed roles and feature caches after classifier freeze."""

    registry = build_open_set_pool_registry(context)
    checked = verify_open_set_pool_registry(registry, context=context)
    caches = build_open_set_feature_caches(context, checked)
    paths = open_set_pool_paths(context.project_root)
    _write_if_identical(paths.registry, registry)
    _replace_prepared_cache_manifest(context, paths.cache_manifest, caches)
    return {
        "status": "prepared",
        "pool_registry_path": str(paths.registry),
        "pool_registry_sha256": registry["registry_sha256"],
        "cache_manifest_path": str(paths.cache_manifest),
        "cache_manifest_sha256": caches["cache_manifest_sha256"],
        "roles": {
            name: {
                "parent_count": value["parent_count"],
                "parent_label_counts": value["parent_label_counts"],
            }
            for name, value in checked["roles"].items()
        },
        "test_payload_access_count": 0,
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


def open_set_role_cache_paths(
    context: V27Context,
    *,
    role: str,
    task: str | None = None,
) -> dict[str, str]:
    """Return verified cache paths for one registered role.

    The mixed feature manifest contains all modalities because it is shared by
    the two task views.  A task-specific caller must not pass irrelevant
    modalities into :class:`V27OpenSetDataset`: L1 owns the causal course
    cache, while L2 owns the physical4/stable8 pair.  ``task=None`` retains
    the all-modalities view for provenance tooling and older callers.
    """

    _, cache = load_open_set_pool_artifacts(context)
    record = dict(cache["roles"].get(role) or {})
    if not record:
        raise ValueError(f"V27 unknown open-set role: {role}")
    all_paths = {
        "course_cache_path": str(record["course"]["path"]),
        "rd_physical_cache_path": str(record["rd_physical"]["path"]),
        "micro_doppler_cache_path": str(record["micro_doppler"]["path"]),
    }
    if task is None:
        return all_paths
    if task == "known_gate":
        return {"course_cache_path": all_paths["course_cache_path"]}
    if task == "dbb_specialist":
        return {
            "rd_physical_cache_path": all_paths["rd_physical_cache_path"],
            "micro_doppler_cache_path": all_paths["micro_doppler_cache_path"],
        }
    raise ValueError(f"V27 unknown open-set task: {task}")


__all__ = ["MIXED_LABELS", "OPEN_SET_CACHE_MANIFEST_PATH", "OPEN_SET_CACHE_MANIFEST_SCHEMA", "OPEN_SET_POOL_REGISTRY_PATH", "OPEN_SET_POOL_SCHEMA", "build_open_set_feature_caches", "build_open_set_pool_registry", "load_open_set_pool_artifacts", "open_set_pool_paths", "open_set_role_cache_paths", "prepare_open_set_pools", "verify_open_set_pool_registry"]