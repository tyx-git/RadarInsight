from __future__ import annotations


TRACK_FEATURE_COLUMNS = [
    "time_seconds",
    "azimuth_deg",
    "range_m",
    "radial_speed_mps",
    "elevation_deg",
    "height_m",
    "amplitude_db",
    "snr_db",
    "course_deg",
    "speed_mps",
]


TRACK_META_COLUMNS = [
    "source_name",
    "record_id",
    "pair_id",
    "timestamp",
    "frame_index",
    "batch_id",
    "label_name_raw",
    "label_name",
    "label_id",
    "note_text",
]
