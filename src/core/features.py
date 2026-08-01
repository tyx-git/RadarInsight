from __future__ import annotations

"""
Raw trajectory columns and versioned model feature schemas.
"""

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

# Raw trajectory fields are kept in their source schema for provenance.  The
# amplitude column is a legacy name: CQ08 stores uncalibrated linear magnitude
# there, while Excel stores dB-like amplitude and raw TR has no amplitude.
TRACK_FEATURE_UNITS = {
    "time_seconds": "seconds",
    "azimuth_deg": "degrees",
    "range_m": "meters",
    "radial_speed_mps": "meters_per_second_positive_receding",
    "elevation_deg": "degrees",
    "height_m": "meters",
    "amplitude_db": "source_dependent_legacy_amplitude",
    "snr_db": "source_dependent_snr_db",
    "course_deg": "degrees",
    "speed_mps": "meters_per_second",
}

TRACK_SOURCE_FEATURE_AVAILABILITY = {
    "cq08_track": {
        "height_m": "missing",
        "amplitude_db": "linear_magnitude",
        "snr_db": "derived_db",
        "radial_speed_mps": "positive_receding_verified_against_range_rate",
    },
    "excel_track": {
        "height_m": "present",
        "amplitude_db": "db_like_amplitude",
        "snr_db": "missing",
        "radial_speed_mps": "source_speed_r_sign_inverted_to_positive_receding",
    },
    "raw_tr_rd": {
        "time_seconds": "source_seconds_since_midnight_relative_to_track_start",
        "range_m": "source_kilometres_converted_to_metres",
        "radial_speed_mps": "causal_backward_range_rate_positive_receding",
        "height_m": "missing",
        "amplitude_db": "missing",
        "snr_db": "source_value",
        "course_deg": "missing",
        "speed_mps": "missing",
    },
}

TRACK_SEQUENCE_FEATURE_COLUMNS_V1 = list(TRACK_FEATURE_COLUMNS)

# v2 keeps non-angular physical channels and replaces circular azimuth/course
# degrees with sin/cos pairs.  Relative time is emitted by encode_track_sequence.
TRACK_SEQUENCE_FEATURE_COLUMNS_V2 = [
    "relative_time_seconds",
    "azimuth_sin",
    "azimuth_cos",
    "range_m",
    "radial_speed_mps",
    "elevation_deg",
    "height_m",
    "amplitude_db",
    "snr_db",
    "course_sin",
    "course_cos",
    "speed_mps",
]

# v3 keeps the V2 semantic features and appends one binary missingness flag
# for every raw track field.  The flags are deliberately part of the feature
# schema so they travel through pretraining, Expert1, offline inference, and
# streaming without relying on zero-valued imputation semantics.
TRACK_SEQUENCE_MISSING_MASK_COLUMNS_V3 = [
    f"missing_{column}" for column in TRACK_FEATURE_COLUMNS
]
TRACK_SEQUENCE_FEATURE_COLUMNS_V3 = [
    *TRACK_SEQUENCE_FEATURE_COLUMNS_V2,
    *TRACK_SEQUENCE_MISSING_MASK_COLUMNS_V3,
]

# Relative time is intentionally defined against the first finite observation
# in the frame handed to the encoder.  A full parent, an offline causal CSV
# window, and a streaming window therefore share one reference rule.
TRACK_RELATIVE_TIME_REFERENCE = "first_finite_observation_in_input_frame"


def track_sequence_feature_columns(feature_schema: str = "v1") -> list[str]:
    normalized = str(feature_schema).strip().lower()
    if normalized == "v1":
        return list(TRACK_SEQUENCE_FEATURE_COLUMNS_V1)
    if normalized == "v2":
        return list(TRACK_SEQUENCE_FEATURE_COLUMNS_V2)
    if normalized == "v3":
        return list(TRACK_SEQUENCE_FEATURE_COLUMNS_V3)
    raise ValueError(f"unsupported trajectory feature schema: {feature_schema}")


def track_sequence_feature_dim(feature_schema: str = "v1") -> int:
    return len(track_sequence_feature_columns(feature_schema))


def track_sequence_binary_feature_indices(feature_schema: str = "v1") -> list[int]:
    """Return feature positions that must remain binary during normalization/augmentation."""
    normalized = str(feature_schema).strip().lower()
    if normalized != "v3":
        return []
    start = len(TRACK_SEQUENCE_FEATURE_COLUMNS_V2)
    return list(range(start, start + len(TRACK_SEQUENCE_MISSING_MASK_COLUMNS_V3)))


def track_sequence_continuous_feature_indices(feature_schema: str = "v1") -> list[int]:
    binary = set(track_sequence_binary_feature_indices(feature_schema))
    return [
        index
        for index in range(track_sequence_feature_dim(feature_schema))
        if index not in binary
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
