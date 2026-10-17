from __future__ import annotations






# PRT=142 has only four Drone, three Bird and zero Balloon train parents in the
# frozen V16 grouped split.  Its physical resolution is retained in evidence,
# but it is not allowed to create a class contribution.  Candidate modes all
# have at least eight train parents per DBB class; readiness independently
# verifies the frozen support before formal training is allowed.  Reliability
# also requires frame-level velocity/modulation Nyquist support.
# (pulse count, PRI in integer nanoseconds, carrier in integer MHz).  PRTnum
# alone is not a sampling contract: each pulse count occurs with two different
# PRI/carrier combinations in frozen V16.  Treating them as one mode could hide
# the same conditional direction reversal that invalidated V17.


BROADBAND_FRAME_NAMES = (
    "residual_negative_8_30mps_energy_fraction",
    "residual_positive_8_30mps_energy_fraction",
    "residual_negative_30_60mps_energy_fraction",
    "residual_positive_30_60mps_energy_fraction",
    "residual_negative_60_90mps_energy_fraction",
    "residual_positive_60_90mps_energy_fraction",
    "residual_total_8_90mps_energy_fraction",
    "residual_rms_offset_mps_over_90",
    "residual_q90_abs_offset_mps_over_90",
    "residual_positive_negative_symmetry",
    "residual_resolution_cell_entropy",
    "residual_top4_resolution_cell_concentration",
    "residual_resolution_cell_peak_count_over_12",
    "residual_sideband_snr_db_over_30",
    "pulse_envelope_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_300_600hz_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_600_1200hz_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_1200_2000hz_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_600_2000hz_fraction",
    "pulse_envelope_modulation_centroid_hz_over_2000",
    "pulse_envelope_modulation_top3_concentration",
    "pulse_envelope_modulation_entropy",
    "pulse_envelope_modulation_snr_db_over_30",
)

# Keep the complete 23-dimensional frame evidence above for physical audit, but
# expose only train-only transferable event summaries to the classifier.  Each
# retained descriptor clears the frozen global AUC/effect gate and keeps its
# Drone-minus-Bird effect direction in all eight calibrated pulse/PRI/carrier
# modes.  Potentially useful but mode-reversing descriptors remain evidence-only
# for a future explicitly mode-aware model; they cannot enter this global head.
BROADBAND_MEDIAN_FRAME_INDICES = (
    1,
    2,
    3,
    7,
    11,
    13,
    14,
    22,
)

# The original q90 descriptors either reversed direction by sampling mode or
# were near-duplicates of retained medians.  The complete frame evidence still
# permits future q90/flash research without exposing those unstable columns now.
BROADBAND_Q90_FRAME_INDICES: tuple[int, ...] = ()
BROADBAND_MICRO_DOPPLER_NAMES = tuple(
    f"median_{BROADBAND_FRAME_NAMES[index]}"
    for index in BROADBAND_MEDIAN_FRAME_INDICES
) + tuple(
    f"q90_{BROADBAND_FRAME_NAMES[index]}" for index in BROADBAND_Q90_FRAME_INDICES
)
BROADBAND_MICRO_DOPPLER_VALUE_DIM = len(BROADBAND_MICRO_DOPPLER_NAMES)
BROADBAND_MODE_STABLE_INFORMATIVE_NAMES = (
    "median_residual_positive_8_30mps_energy_fraction",
    "median_residual_negative_30_60mps_energy_fraction",
    "median_residual_positive_30_60mps_energy_fraction",
    "median_residual_rms_offset_mps_over_90",
    "median_residual_q90_abs_offset_mps_over_90",
    "median_residual_top4_resolution_cell_concentration",
    "median_residual_resolution_cell_peak_count_over_12",
    "median_residual_sideband_snr_db_over_30",
    "median_pulse_envelope_rms_clipped_0_2_over_2",
    "median_pulse_envelope_modulation_snr_db_over_30",
)
BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES = (
    "median_residual_q90_abs_offset_mps_over_90",
    "median_residual_resolution_cell_peak_count_over_12",
)

if len(BROADBAND_MEDIAN_FRAME_INDICES) != 8:
    raise RuntimeError("broadband micro-Doppler stable median contract must contain 8 columns")
if BROADBAND_Q90_FRAME_INDICES:
    raise RuntimeError("broadband micro-Doppler stable contract must not expose q90 columns")
if BROADBAND_MICRO_DOPPLER_VALUE_DIM != 8:
    raise RuntimeError("broadband micro-Doppler stable model contract must contain 8 columns")
if set(BROADBAND_MICRO_DOPPLER_NAMES) != (
    set(BROADBAND_MODE_STABLE_INFORMATIVE_NAMES)
    - set(BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES)
):
    raise RuntimeError("broadband stable model columns differ from the train-only selection")





































__all__ = ["BROADBAND_FRAME_NAMES", "BROADBAND_MEDIAN_FRAME_INDICES", "BROADBAND_MICRO_DOPPLER_NAMES", "BROADBAND_MICRO_DOPPLER_VALUE_DIM", "BROADBAND_MODE_STABLE_INFORMATIVE_NAMES", "BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES", "BROADBAND_Q90_FRAME_INDICES"]