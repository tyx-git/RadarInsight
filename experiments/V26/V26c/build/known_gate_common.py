from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pandas as pd

from radar_ln3.known_gate_features import extract_causal_course_circular_incoherence



MINIMUM_VALID_NONFIRST_COURSES = 3












def _course_value(payload: Any, *, sample_id: str) -> float | None:
    values = json.loads(str(payload))
    if not isinstance(values, dict):
        raise ValueError(f"V21e TR payload is not an object: {sample_id}")
    raw = values.get("course_deg")
    if raw is None:
        return None
    course = float(raw)
    if not math.isfinite(course):
        raise ValueError(f"V21e course is non-finite: {sample_id}")
    return course


def _build_split(
    selected: pd.DataFrame,
    *,
    split: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any], pd.DataFrame]:
    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    values = np.zeros((len(selected), 1), dtype=np.float32)
    valid = np.zeros((len(selected), 1), dtype=np.bool_)
    parent_rows: list[dict[str, Any]] = []
    first_event_excluded_count = 0
    first_event_zero_count = 0
    nonfirst_missing_count = 0
    nonfirst_zero_count = 0
    carried_valid_prefix_count = 0
    cursor = 0
    for parent_id, parent in selected.groupby("parent_sample_id", sort=True):
        labels = parent["label_name"].astype(str).unique().tolist()
        tracks = parent["track_id"].astype(int).unique().tolist()
        event_indices = parent["causal_event_index"].astype(int).tolist()
        if len(labels) != 1 or len(tracks) != 1:
            raise ValueError(f"V21e parent identity is not unique: {parent_id}")
        if event_indices != list(range(len(parent))):
            raise ValueError(f"V21e causal event order is not contiguous: {parent_id}")
        courses = [
            _course_value(row.tr_feature_values_json, sample_id=str(row.sample_id))
            for row in parent.itertuples(index=False)
        ]
        first_event_excluded_count += 1
        first_event_zero_count += int(
            courses[0] is not None and float(courses[0]) == 0.0
        )
        nonfirst_missing_count += sum(value is None for value in courses[1:])
        nonfirst_zero_count += sum(
            value is not None and float(value) == 0.0 for value in courses[1:]
        )
        parent_values, parent_valid = extract_causal_course_circular_incoherence(
            courses,
            minimum_valid_nonfirst_points=MINIMUM_VALID_NONFIRST_COURSES,
        )
        length = len(parent)
        values[cursor : cursor + length] = parent_values
        valid[cursor : cursor + length] = parent_valid
        for event_index in range(1, length):
            if (
                courses[event_index] is None
                and bool(parent_valid[event_index, 0])
            ):
                carried_valid_prefix_count += 1
        selected_values = parent_values[parent_valid]
        if selected_values.size == 0 or not bool(parent_valid[-1, 0]):
            raise ValueError(f"V21e parent lacks a valid final course prefix: {parent_id}")
        parent_rows.append(
            {
                "parent_sample_id": str(parent_id),
                "track_id": int(tracks[0]),
                "label_name": labels[0],
                "event_count": length,
                "valid_nonfirst_course_count": int(
                    sum(value is not None for value in courses[1:])
                ),
                "valid_prefix_count": int(parent_valid.sum()),
                "prefix_mean_incoherence": float(np.mean(selected_values)),
                "prefix_median_incoherence": float(np.median(selected_values)),
                "final_incoherence": float(parent_values[-1, 0]),
                "first_event_course_was_zero": bool(
                    courses[0] is not None and float(courses[0]) == 0.0
                ),
                "later_zero_course_count": int(
                    sum(
                        value is not None and float(value) == 0.0
                        for value in courses[1:]
                    )
                ),
            }
        )
        cursor += length
    if cursor != len(selected) or np.any(values[~valid] != 0.0):
        raise ValueError("V21e cache construction did not preserve event/mask semantics")
    parents = pd.DataFrame(parent_rows)
    statistics = {
        "split": split,
        "event_count": len(selected),
        "parent_count": len(parents),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_event_fraction": float(valid.mean()),
        "first_event_excluded_count": int(first_event_excluded_count),
        "first_event_zero_but_excluded_count": int(first_event_zero_count),
        "nonfirst_missing_course_count": int(nonfirst_missing_count),
        "nonfirst_true_zero_course_included_count": int(nonfirst_zero_count),
        "missing_event_carried_valid_prefix_count": int(carried_valid_prefix_count),
        "minimum_valid_nonfirst_courses": MINIMUM_VALID_NONFIRST_COURSES,
        "test_payload_access_count": 0,
    }
    return {"sample_ids": sample_ids, "values": values, "valid": valid}, statistics, parents









