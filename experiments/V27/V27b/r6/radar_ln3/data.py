from __future__ import annotations



from .micro_doppler import load_micro_doppler_cache
from .rd_physical import load_rd_physical_cache



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










# Canonical RD tensors are small enough for a single parent, but they are
# still untrusted compressed archives.  Bound both the archive and its
# declared uncompressed members before NumPy opens them; otherwise a malformed
# or zip-bomb-like .npz can allocate memory before the shape checks below run.










__all__ = ["DT_REASON_FIRST_EVENT", "DT_REASON_PADDING", "DT_REASON_POSITIVE_DELTA", "DT_REASON_ZERO_DELTA", "REQUIRED_INDEX_COLUMNS"]