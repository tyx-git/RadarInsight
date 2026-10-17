from __future__ import annotations


import numpy as np

from .micro_doppler import load_micro_doppler_cache
from .rd_physical import load_rd_physical_cache


RD_QUALITY_NAMES = (
    "rd_present",
    "rd_frame_count_scaled",
    "rd_local_duration_scaled",
    "rd_missing",
    "rd_local_time_non_monotonic",
    "rd_multiple_frames",
    "rd_frames_truncated",
)

DT_REASON_PADDING = 0
DT_REASON_FIRST_EVENT = 1
DT_REASON_ZERO_DELTA = 2
DT_REASON_POSITIVE_DELTA = 3






REQUIRED_INDEX_COLUMNS = {
    "schema_version",
    "sample_id",
    "parent_sample_id",
    "track_id",
    "source_name",
    "split_group_id",
    "split_name",
    "label_name",
    "label_id",
    "causal_event_index",
    "tr_time_seconds",
    "tr_dt_seconds",
    "tr_dt_is_negative",
    "tr_feature_values_json",
    "tr_missing_count",
    "rd_present",
    "rd_frame_count",
    "rd_tensor_path",
    "rd_tensor_frame_indices_json",
    "rd_local_duration_seconds",
    "rd_parent_coverage_status",
    "rd_parent_order_status",
    "rd_quality_missing",
    "rd_quality_partial_parent",
    "rd_quality_parent_order_anomaly",
}


# Compatibility name retained for historical callers.  New code should use
# ``parse_bool`` directly so malformed configuration values fail closed.




def encode_causal_rd_quality(
    *,
    present: bool,
    original_count: int,
    duration_seconds: float,
    local_time_non_monotonic: bool,
    truncated: bool,
    max_rd_frames_per_point: int,
) -> np.ndarray:
    """Encode only information observable at the current TR/RD event.

    Parent-level coverage and order summaries are intentionally excluded: they
    require future events and would leak target-completion information into an
    online causal prefix. The vector remains seven-dimensional for the current
    model interface, but its schema is explicitly versioned.
    """

    count = max(0, int(original_count))
    max_frames = max(1, int(max_rd_frames_per_point))
    duration = max(0.0, float(duration_seconds))
    values = np.asarray(
        [
            float(bool(present)),
            float(np.log1p(count) / np.log1p(max_frames)),
            float(np.clip(duration / 0.5, 0.0, 1.0)),
            float(not bool(present)),
            float(bool(local_time_non_monotonic)),
            float(count > 1),
            float(bool(truncated)),
        ],
        dtype=np.float32,
    )
    if values.shape != (len(RD_QUALITY_NAMES),) or not np.isfinite(values).all():
        raise ValueError("causal RD quality encoder produced an invalid vector")
    return values






# Canonical RD tensors are small enough for a single parent, but they are
# still untrusted compressed archives.  Bound both the archive and its
# declared uncompressed members before NumPy opens them; otherwise a malformed
# or zip-bomb-like .npz can allocate memory before the shape checks below run.










__all__ = ["DT_REASON_FIRST_EVENT", "DT_REASON_PADDING", "DT_REASON_POSITIVE_DELTA", "DT_REASON_ZERO_DELTA", "RD_QUALITY_NAMES", "REQUIRED_INDEX_COLUMNS", "encode_causal_rd_quality"]