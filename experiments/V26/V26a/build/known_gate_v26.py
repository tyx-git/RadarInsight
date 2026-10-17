"""Prepare the V26 Known-only baseline and its post-checkpoint open-set caches.

V26 deliberately keeps the L1 model binary.  UnknownTarget is never part of
the model class mapping, train normalization, sampler, optimizer or known-only
checkpoint ranking.  Mixed validation/test data are materialized into separate
audit caches and may only be consumed by the report after a completed
known-only checkpoint exists.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import _write_npz_new_or_identical
from radar_ln3.known_gate_features import extract_causal_course_circular_incoherence


SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
AUDIT_VALIDATION_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-val-mixed-open-set-course-cache.npz"
)
AUDIT_TEST_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-test-mixed-open-set-course-cache.npz"
)

KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
UNKNOWN_LABEL = "UnknownTarget"
AUDIT_LABELS = (*KNOWN_LABELS, UNKNOWN_LABEL)
EXPECTED_PARENT_COUNTS = {
    "train": {"DroneTarget": 476, "BirdTarget": 239, "BalloonTarget": 238, "ClutterTarget": 33, "UnknownTarget": 69},
    "val": {"DroneTarget": 113, "BirdTarget": 56, "BalloonTarget": 54, "ClutterTarget": 9, "UnknownTarget": 15},
    "test": {"DroneTarget": 109, "BirdTarget": 53, "BalloonTarget": 58, "ClutterTarget": 8, "UnknownTarget": 16},
}






def _course_value(payload: Any, *, sample_id: str) -> float | None:
    values = json.loads(str(payload))
    if not isinstance(values, dict):
        raise ValueError(f"V26 course payload is not an object: {sample_id}")
    raw = values.get("course_deg")
    if raw is None:
        return None
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"V26 course is non-finite: {sample_id}")
    return value


def _selected_index(path: Path, split: str, *, include_unknown: bool) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        usecols=[
            "sample_id",
            "parent_sample_id",
            "track_id",
            "split_name",
            "label_name",
            "causal_event_index",
            "tr_feature_values_json",
        ],
        low_memory=False,
    )
    labels = AUDIT_LABELS if include_unknown else KNOWN_LABELS
    selected = frame[
        frame["split_name"].astype(str).eq(split)
        & frame["label_name"].astype(str).isin(labels)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V26 {split} index selection is empty or repeats sample_id")
    support = (
        selected[["parent_sample_id", "label_name"]]
        .drop_duplicates("parent_sample_id")["label_name"]
        .value_counts()
        .to_dict()
    )
    actual = {label: int(support.get(label, 0)) for label in labels}
    expected = {
        label: EXPECTED_PARENT_COUNTS[split][label]
        for label in labels
    }
    if actual != expected:
        raise ValueError(f"V26 {split} parent support changed: expected={expected}, actual={actual}")
    return selected


def _build_course_cache(selected: pd.DataFrame, *, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    values = np.zeros((len(selected), 1), dtype=np.float32)
    valid = np.zeros((len(selected), 1), dtype=np.bool_)
    cursor = 0
    parent_count = 0
    unknown_parent_count = 0
    for parent_id, parent in selected.groupby("parent_sample_id", sort=True):
        labels = parent["label_name"].astype(str).unique().tolist()
        tracks = parent["track_id"].astype(int).unique().tolist()
        events = parent["causal_event_index"].astype(int).tolist()
        if len(labels) != 1 or len(tracks) != 1:
            raise ValueError(f"V26 parent identity is not unique: {parent_id}")
        if events != list(range(len(parent))):
            raise ValueError(f"V26 causal event order is not contiguous: {parent_id}")
        courses = [
            _course_value(row.tr_feature_values_json, sample_id=str(row.sample_id))
            for row in parent.itertuples(index=False)
        ]
        parent_values, parent_valid = extract_causal_course_circular_incoherence(courses)
        length = len(parent)
        values[cursor : cursor + length] = parent_values
        valid[cursor : cursor + length] = parent_valid
        parent_count += 1
        unknown_parent_count += int(labels[0] == UNKNOWN_LABEL)
        cursor += length
    if cursor != len(selected) or np.any(values[~valid] != 0.0):
        raise ValueError("V26 course cache did not preserve causal value/mask semantics")
    return {
        "sample_ids": sample_ids,
        "values": values,
        "valid": valid,
    }, {
        "split": split,
        "event_count": int(len(selected)),
        "parent_count": int(parent_count),
        "unknown_parent_count": int(unknown_parent_count),
        "feature_width": 1,
        "valid_value_count": int(valid.sum()),
        "invalid_value_count": int((~valid).sum()),
        "valid_value_fraction": float(valid.mean()),
        "raw_echo_payload_access_count": 0,
        "rd_tensor_payload_access_count": 0,
    }




def materialize_test_audit_cache(project_root: Path) -> tuple[Path, dict[str, Any]]:
    """Create the mixed test audit cache only after checkpoint freezing.

    The caller is the V26 report, which records and verifies the immutable
    best-checkpoint hash before invoking this function.  Keeping this helper
    separate from ``build`` makes accidental test access during prepare
    visible and fail-closed in the pipeline contract.
    """

    root = project_root.resolve()
    selected = _selected_index(root / SOURCE_INDEX, "test", include_unknown=True)
    arrays, statistics = _build_course_cache(selected, split="test")
    _write_npz_new_or_identical(root / AUDIT_TEST_CACHE, tuple(arrays.items()))
    return root / AUDIT_TEST_CACHE, statistics


__all__ = ["AUDIT_TEST_CACHE", "AUDIT_VALIDATION_CACHE", "EXPECTED_PARENT_COUNTS", "materialize_test_audit_cache"]






