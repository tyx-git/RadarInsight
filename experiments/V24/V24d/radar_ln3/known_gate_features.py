"""Causal, label-free feature contracts for the V21 KnownGate experiments."""

from __future__ import annotations

from typing import Mapping




BASELINE_VARIANT = "baseline"
V21A_VARIANT = "v21a"
V21A_INTERFRAME_VARIANT = "v21a_interframe"
V21B_VARIANT = "v21b"
V21C_VARIANT = "v21c"
V21D_VARIANT = "v21d"
V21D_PREFIX_VARIANT = "v21d_prefix"
V21E_COURSE_PREFIX_VARIANT = "v21e_course_prefix"
V21F_COURSE_INTERFRAME_VARIANT = "v21f_course_interframe"
V21H_COURSE_ISO_VARIANT = "v21h_course_iso"
V21I_COURSE_D_RANGE_VARIANT = "v21i_course_d_range"
V22A_RD_TEMPORAL_RESIDUAL_VARIANT = "v22a_rd_temporal_residual"
# V24 deliberately keeps the only adopted V21e auxiliary input while creating
# a fresh, independently trained control for the next RD counter-evidence
# series.  It must remain a distinct variant so its checkpoints and reports
# cannot be mistaken for the historical V21e run.
V24_COURSE_REASON_AWARE_BASELINE_VARIANT = "v24_course_reason_aware_baseline"
# V24a appends one independent, current-event-only RD counter-evidence
# feature to the V24 course baseline.  It intentionally does not reuse V21a's
# optional preceding-event frame bridge: a one-frame current event is invalid.
V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT = "v24a_course_rd_peak_drift_mean"
# V24c deliberately reuses V24's only adopted causal-course input.  Its sole
# variable is the reason-aware liquid update at same-second observations.
V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT = "v24c_course_strict_same_second_skip"
# V25a is a fresh V24 replication whose only added behavior is read-only
# pre-clip gradient telemetry in the trainer.  It intentionally keeps the
# one adopted causal-course column and does not introduce a model feature.
V25A_GRADIENT_TELEMETRY_VARIANT = "v25a_course_preclip_gradient_telemetry"
# V25d is a loss-only follow-up to the completed V25a audit.  It reuses the
# exact selected V21e causal-course cache; a distinct variant keeps its cache
# identity explicit rather than relying on a fallback lookup.
V25D_PRESENT_CLASS_MEAN_VARIANT = "v25d_course_present_class_mean"
# V25e keeps the exact V25a causal-course cache.  Its sole variable is a
# mixed-batch shared-gradient projection, so no new feature column or cache
# schema is allowed.
V25E_SHARED_GRADIENT_PROJECTION_VARIANT = "v25e_shared_gradient_conflict_projection"
# V26 keeps the frozen V21e causal-course input for a fresh Known-only
# baseline.  The variant is intentionally distinct because its mixed
# validation/test open-set audit is a separate, post-checkpoint artifact.
V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT = "v26_known_only_open_set_baseline"
# V26j is the event-internal RD temporal aggregation ablation.  It consumes
# exactly the same selected causal-course cache as V26/V26e; its sole model
# change is represented separately so a causal-GRU checkpoint can never be
# mistaken for the masked-mean baseline.
V26J_RD_CAUSAL_GRU_VARIANT = "v26j_rd_causal_gru"

SPECTRAL_FEATURE_NAMES = (
    "native_stft_adjacent_spectral_affinity_median",
    "native_stft_adjacent_spectral_affinity_minimum",
)
INTERFRAME_SPECTRAL_FEATURE_NAMES = (
    "current_event_adjacent_rd_velocity_profile_cosine_mean",
    "current_event_adjacent_rd_velocity_profile_cosine_minimum",
    "current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
    "current_event_adjacent_rd_peak_velocity_drift_max_fraction",
    "current_event_zero_velocity_energy_fraction_std",
    "current_event_zero_velocity_energy_fraction_minimum",
)
MODAL_FEATURE_NAMES = (
    "target_row_rd_peak_vs_tr_radial_abs_deviation_median_over_85mps",
    "target_row_rd_peak_vs_tr_radial_abs_deviation_max_over_85mps",
)
ISO_FEATURE_NAMES = ("target_row_linear_energy_fraction_mean",)
D_RANGE_FEATURE_NAMES = ("mtd_minus_track_range_std_m_over_75m",)
D_RANGE_PREFIX_FEATURE_NAMES = (
    "causal_prefix_mtd_minus_tr_range_sample_std_m_over_75m",
)
COURSE_PREFIX_FEATURE_NAMES = (
    "causal_prefix_course_circular_incoherence",
)
COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    "current_event_adjacent_rd_peak_velocity_drift_mean_fraction",
)
COURSE_INTERFRAME_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    *INTERFRAME_SPECTRAL_FEATURE_NAMES,
)
COURSE_ISO_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    *ISO_FEATURE_NAMES,
)
COURSE_D_RANGE_FEATURE_NAMES = (
    *COURSE_PREFIX_FEATURE_NAMES,
    *D_RANGE_PREFIX_FEATURE_NAMES,
)
FEATURE_NAMES_BY_VARIANT: Mapping[str, tuple[str, ...]] = {
    BASELINE_VARIANT: (),
    V21A_VARIANT: SPECTRAL_FEATURE_NAMES,
    V21A_INTERFRAME_VARIANT: INTERFRAME_SPECTRAL_FEATURE_NAMES,
    V21B_VARIANT: MODAL_FEATURE_NAMES,
    V21C_VARIANT: ISO_FEATURE_NAMES,
    V21D_VARIANT: D_RANGE_FEATURE_NAMES,
    V21D_PREFIX_VARIANT: D_RANGE_PREFIX_FEATURE_NAMES,
    V21E_COURSE_PREFIX_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V21F_COURSE_INTERFRAME_VARIANT: COURSE_INTERFRAME_FEATURE_NAMES,
    V21H_COURSE_ISO_VARIANT: COURSE_ISO_FEATURE_NAMES,
    V21I_COURSE_D_RANGE_VARIANT: COURSE_D_RANGE_FEATURE_NAMES,
    # V22a reuses the frozen V21e course cache.  Its new RD evidence is an
    # internal model branch, deliberately not an external feature-cache field.
    V22A_RD_TEMPORAL_RESIDUAL_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT: (
        COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES
    ),
    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V25A_GRADIENT_TELEMETRY_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V25D_PRESENT_CLASS_MEAN_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V25E_SHARED_GRADIENT_PROJECTION_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
    V26J_RD_CAUSAL_GRU_VARIANT: COURSE_PREFIX_FEATURE_NAMES,
}
# These variants store an already-selected feature matrix rather than the
# original canonical KnownGate feature matrix.  They must therefore use
# cache-local column positions (0..width-1), not canonical feature indices.
# Keep this set shared by index construction and cache loading: a mismatch
# silently turns a reuse-only variant into a KeyError at Dataset construction.
























__all__ = ["BASELINE_VARIANT", "COURSE_PREFIX_FEATURE_NAMES", "COURSE_RD_PEAK_DRIFT_MEAN_FEATURE_NAMES", "COURSE_INTERFRAME_FEATURE_NAMES", "COURSE_ISO_FEATURE_NAMES", "COURSE_D_RANGE_FEATURE_NAMES", "D_RANGE_FEATURE_NAMES", "D_RANGE_PREFIX_FEATURE_NAMES", "FEATURE_NAMES_BY_VARIANT", "INTERFRAME_SPECTRAL_FEATURE_NAMES", "ISO_FEATURE_NAMES", "MODAL_FEATURE_NAMES", "SPECTRAL_FEATURE_NAMES", "V21A_VARIANT", "V21A_INTERFRAME_VARIANT", "V21B_VARIANT", "V21C_VARIANT", "V21D_VARIANT", "V21D_PREFIX_VARIANT", "V21E_COURSE_PREFIX_VARIANT", "V21F_COURSE_INTERFRAME_VARIANT", "V21H_COURSE_ISO_VARIANT", "V21I_COURSE_D_RANGE_VARIANT", "V22A_RD_TEMPORAL_RESIDUAL_VARIANT", "V24_COURSE_REASON_AWARE_BASELINE_VARIANT", "V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT", "V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT", "V25A_GRADIENT_TELEMETRY_VARIANT", "V25D_PRESENT_CLASS_MEAN_VARIANT", "V25E_SHARED_GRADIENT_PROJECTION_VARIANT", "V26_KNOWN_ONLY_OPEN_SET_BASELINE_VARIANT", "V26J_RD_CAUSAL_GRU_VARIANT"]