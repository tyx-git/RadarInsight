from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import (
    _CanonicalFrameProvenance,
    _CanonicalRD,
    _as_bool,
    _json_text,
    _parse_indices,
    _resolve,
    _sha256,
    _write_json_new_or_identical,
    _write_npz_new_or_identical,
)
from radar_ln3.micro_doppler import parse_cq08_raw_echo
from radar_ln3.micro_doppler_broadband import (
    BROADBAND_CANDIDATE_SAMPLING_MODES,
    BROADBAND_EXPECTED_TRAIN_SAMPLING_MODES,
    BROADBAND_FRAME_CAP,
    BROADBAND_FRAME_NAMES,
    BROADBAND_FRAME_VALUE_DIM,
    BROADBAND_MODE_STABLE_INFORMATIVE_NAMES,
    BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES,
    BROADBAND_MICRO_DOPPLER_EVIDENCE_SCHEMA,
    BROADBAND_MICRO_DOPPLER_MASK_MODE,
    BROADBAND_MICRO_DOPPLER_NAMES,
    BROADBAND_MICRO_DOPPLER_SCHEMA,
    BROADBAND_MICRO_DOPPLER_VALUE_DIM,
    BROADBAND_MIN_TRAIN_PARENTS_PER_CLASS_PER_MODE,
    BROADBAND_MODULATION_PROFILE_BIN_COUNT,
    BROADBAND_DOPPLER_PROFILE_BIN_COUNT,
    aggregate_broadband_frame_values,
    broadband_micro_doppler_contract,
    broadband_sampling_mode,
    broadband_sampling_mode_name,
    extract_current_event_broadband_micro_doppler,
)
from radar_ln3.rd_physical import (
    RD_PHYSICAL_MASK_MODE,
    RD_PHYSICAL_SCHEMA,
    RD_PHYSICAL_VALUE_DIM,
    extract_current_event_rd_physical,
    rd_physical_contract,
)


VERSION = "RADAR-LN3-V18"
SOURCE_VERSION = "RADAR-LN3-V16"
PAIRED_PROTOCOL = (
    "v18-dbb-rd-physical4-control-versus-plus-broadband-multiscale-"
    "micro-doppler8-stable-treatment-v3"
)
CHECKPOINT_SELECTION = (
    "parent_min_precision_recall_f1_then_min_drone_bird_errors_then_macro_f1"
)
DBB_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
SPLITS = ("train", "val")
SEED = 42
REUSABLE_PAIRED_PROTOCOL = (
    "v18-dbb-rd-physical4-control-versus-plus-broadband-multiscale-"
    "micro-doppler21-pruned-treatment-v2"
)
REUSABLE_READINESS_SCHEMA = (
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler-"
    "training-readiness-v2-pruned21"
)
REUSABLE_SUMMARY_SCHEMA = (
    "cq08-current-event-raw-iq-broadband-multiscale-"
    "micro-doppler-summary-v5-pruned21"
)
REUSABLE_MANIFEST_TYPE = (
    "cq08_dbb_current_event_broadband_multiscale_"
    "micro_doppler21_pruned_paired_cache_v2"
)
REUSABLE_ONLY_FAILED_CHECK = (
    "no_globally_informative_drone_bird_feature_reverses_or_lacks_mode_support"
)
REUSABLE_MODEL_NAMES = (
    "median_residual_negative_8_30mps_energy_fraction",
    "median_residual_positive_8_30mps_energy_fraction",
    "median_residual_negative_30_60mps_energy_fraction",
    "median_residual_positive_30_60mps_energy_fraction",
    "median_residual_total_8_90mps_energy_fraction",
    "median_residual_rms_offset_mps_over_90",
    "median_residual_q90_abs_offset_mps_over_90",
    "median_residual_top4_resolution_cell_concentration",
    "median_residual_resolution_cell_peak_count_over_12",
    "median_residual_sideband_snr_db_over_30",
    "median_pulse_envelope_rms_clipped_0_2_over_2",
    "median_pulse_envelope_modulation_300_600hz_rms_clipped_0_2_over_2",
    "median_pulse_envelope_modulation_600_1200hz_rms_clipped_0_2_over_2",
    "median_pulse_envelope_modulation_1200_2000hz_rms_clipped_0_2_over_2",
    "median_pulse_envelope_modulation_centroid_hz_over_2000",
    "median_pulse_envelope_modulation_top3_concentration",
    "median_pulse_envelope_modulation_entropy",
    "median_pulse_envelope_modulation_snr_db_over_30",
    "q90_residual_negative_8_30mps_energy_fraction",
    "q90_residual_positive_8_30mps_energy_fraction",
    "q90_residual_total_8_90mps_energy_fraction",
)

SOURCE_INDEX = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
)
SOURCE_CONTRACT = Path(
    "data/processed/RADAR-LN3-V16-cq08-product-local-data-contract.json"
)
SOURCE_PROVENANCE = Path(
    "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
)
SOURCE_STATS = Path(
    "data/processed/RADAR-LN3-V16-cq08-tr-rd-train-normalization-stats.json"
)
SOURCE_CANONICAL_FRAME_INDEX = Path(
    "data/processed/cq08_rd_physical_canonical_frame_index_v3_1.csv"
)
SOURCE_DBB_CONFIG = Path(
    "config/RADAR-LN3/RADAR-LN3-V16-cq08-liquid-product-local-"
    "dbb-specialist-config.json"
)

DATA_CONTRACT = Path(
    "data/processed/RADAR-LN3-V18-cq08-dbb-broadband-multiscale-"
    "micro-doppler8-stable-data-contract.json"
)
CACHE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V18-cq08-dbb-broadband-multiscale-"
    "micro-doppler8-stable-cache-manifest.json"
)
REUSABLE_EVIDENCE_MANIFEST = Path(
    "data/processed/RADAR-LN3-V18-cq08-dbb-broadband-multiscale-"
    "micro-doppler21-pruned-cache-manifest.json"
)
REUSABLE_BLOCKED_READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V18-broadband-multiscale-"
    "micro-doppler21-pruned-train-readiness.json"
)


def _summary_cache(split: str) -> Path:
    return Path(
        f"data/processed/RADAR-LN3-V18-cq08-dbb-{split}-broadband-multiscale-"
        "micro-doppler8-stable-summary-cache.npz"
    )


def _evidence_cache(split: str) -> Path:
    return Path(
        f"data/processed/RADAR-LN3-V18-cq08-dbb-{split}-broadband-multiscale-"
        "micro-doppler-evidence-cache.npz"
    )


def _physical_cache(split: str) -> Path:
    return Path(
        f"data/processed/RADAR-LN3-V18-cq08-dbb-{split}-rd-physical4-cache.npz"
    )


def _config_path(arm: str) -> Path:
    return Path(
        "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-"
        f"rd-physical4-broadband-multiscale-micro-doppler8-stable-{arm}-seed42-config.json"
    )


def _artifact_dir(arm: str) -> str:
    return (
        "artifacts/RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-"
        f"broadband-multiscale-micro-doppler8-stable-{arm}-seed42-formal"
    )


def _event_stats(values: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    selected = values[valid]
    empty = [None] * int(values.shape[1])
    nonzero = valid & np.any(values != 0.0, axis=1)
    return {
        "event_count": int(values.shape[0]),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "valid_fraction": float(valid.mean()) if valid.size else 0.0,
        "nonzero_intervention_event_count": int(nonzero.sum()),
        "neutral_intervention_event_count": int((valid & ~nonzero).sum()),
        "feature_minimum": (
            selected.min(axis=0).astype(float).tolist() if len(selected) else empty
        ),
        "feature_maximum": (
            selected.max(axis=0).astype(float).tolist() if len(selected) else empty
        ),
        "feature_mean": (
            selected.mean(axis=0, dtype=np.float64).astype(float).tolist()
            if len(selected)
            else empty
        ),
    }


EVIDENCE_FIELD_NAMES = (
    "sample_ids",
    "valid",
    "event_offsets",
    "raw_dat_frame_indices_zero_based",
    "legacy_mat_list_ordinals_zero_based",
    "canonical_tensor_frame_indices_zero_based",
    "mat_acquisition_frame_numbers_one_based",
    "frame_values",
    "frame_feature_reliable",
    "residual_doppler_profiles",
    "envelope_modulation_profiles",
    "target_range_indices",
    "target_doppler_indices",
    "target_doppler_mps",
    "refined_body_doppler_mps",
    "velocity_nyquist_mps",
    "rayleigh_velocity_mps",
    "residual_sideband_snr_db",
    "envelope_modulation_snr_db",
    "frame_prt_num",
    "frame_prt_seconds",
    "frame_frequency_hz",
    "frame_cpi_count",
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _reaggregate_summary_from_evidence(
    evidence: dict[str, np.ndarray],
    *,
    event_count: int,
) -> np.ndarray:
    offsets = evidence["event_offsets"]
    valid = evidence["valid"]
    if (
        offsets.shape != (event_count + 1,)
        or offsets.dtype != np.int64
        or offsets[0] != 0
        or valid.shape != (event_count,)
        or valid.dtype != np.bool_
    ):
        raise ValueError("V18 reusable evidence event alignment is invalid")
    values = np.zeros(
        (event_count, BROADBAND_MICRO_DOPPLER_VALUE_DIM), dtype=np.float32
    )
    for event_index, event_valid in enumerate(valid):
        start, stop = int(offsets[event_index]), int(offsets[event_index + 1])
        if event_valid:
            values[event_index] = aggregate_broadband_frame_values(
                evidence["frame_values"][start:stop],
                evidence["frame_feature_reliable"][start:stop],
            )
    return values


def _concatenate(
    values: list[np.ndarray],
    *,
    dtype: np.dtype[Any] | type[Any],
    empty_shape: tuple[int, ...] = (),
) -> np.ndarray:
    if values:
        return np.concatenate(values, axis=0).astype(dtype, copy=False)
    return np.zeros((0, *empty_shape), dtype=dtype)


def _build_split(
    root: Path,
    split: str,
    frame: pd.DataFrame,
    provenance_by_parent: dict[str, dict[str, Any]],
    canonical_frame_provenance: _CanonicalFrameProvenance,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = frame[
        frame["split_name"].astype(str).eq(split)
        & frame["label_name"].astype(str).isin(DBB_LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    if selected.empty or selected["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V18 {split} DBB event population is empty or duplicated")

    sample_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    summary_values = np.zeros(
        (len(selected), BROADBAND_MICRO_DOPPLER_VALUE_DIM), dtype=np.float32
    )
    physical_values = np.zeros(
        (len(selected), RD_PHYSICAL_VALUE_DIM), dtype=np.float32
    )
    valid = np.zeros((len(selected),), dtype=np.bool_)
    event_offsets = [0]

    ragged: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "raw_dat_frame_indices_zero_based",
            "legacy_mat_list_ordinals_zero_based",
            "canonical_tensor_frame_indices_zero_based",
            "mat_acquisition_frame_numbers_one_based",
            "frame_values",
            "frame_feature_reliable",
            "residual_doppler_profiles",
            "envelope_modulation_profiles",
            "target_range_indices",
            "target_doppler_indices",
            "target_doppler_mps",
            "refined_body_doppler_mps",
            "velocity_nyquist_mps",
            "rayleigh_velocity_mps",
            "residual_sideband_snr_db",
            "envelope_modulation_snr_db",
            "frame_prt_num",
            "frame_prt_seconds",
            "frame_frequency_hz",
            "frame_cpi_count",
        )
    }
    raw_file_access_count = 0
    raw_frame_count = 0
    selected_raw_frame_count = 0
    reliable_frame_count = 0
    neutral_event_count = 0
    legacy_mismatch_frame_count = 0
    legacy_mismatch_track_ids: set[int] = set()
    parent_label_counts: Counter[str] = Counter()
    prt_parent_support: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {label: set() for label in DBB_LABELS}
    )
    sampling_mode_parent_support: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {label: set() for label in DBB_LABELS}
    )
    rd_cache = _CanonicalRD(root)

    grouped = list(selected.groupby("parent_sample_id", sort=True))
    event_cursor = 0
    for parent_number, (parent_id_value, parent) in enumerate(grouped, start=1):
        parent_id = str(parent_id_value)
        labels = parent["label_name"].astype(str).unique().tolist()
        track_ids = (
            pd.to_numeric(parent["track_id"], errors="raise").astype(int).unique().tolist()
        )
        raw_paths = parent["cq08_raw_echo_file"].astype(str).unique().tolist()
        if len(labels) != 1 or len(track_ids) != 1 or len(raw_paths) != 1:
            raise ValueError(f"V18 parent label/raw path is not unique: {parent_id}")
        label = labels[0]
        track_id = int(track_ids[0])
        parent_label_counts[label] += 1

        provenance = provenance_by_parent.get(parent_id)
        if provenance is None:
            raise ValueError(f"V18 parent lacks V16 provenance: {parent_id}")
        raw_path = _resolve(root, raw_paths[0])
        if str(raw_path) != str(Path(str(provenance["raw_echo_file"])).resolve()):
            raise ValueError(f"V18 raw path differs from V16 provenance: {parent_id}")
        if _sha256(raw_path) != str(provenance["raw_sha256"]):
            raise ValueError(f"V18 raw file hash differs from V16 provenance: {parent_id}")
        raw_frames = parse_cq08_raw_echo(raw_path)
        raw_file_access_count += 1
        raw_frame_count += len(raw_frames)
        used_raw_indices: set[int] = set()

        rd_paths = [
            str(value)
            for value in parent.loc[
                parent["rd_present"].map(_as_bool), "rd_tensor_path"
            ]
            .dropna()
            .unique()
            if str(value).strip()
        ]
        if len(rd_paths) > 1:
            raise ValueError(f"V18 parent references multiple RD tensors: {parent_id}")
        amplitude = velocity = None
        if rd_paths:
            amplitude, velocity = rd_cache.load(rd_paths[0])

        for row in parent.itertuples(index=False):
            present = _as_bool(row.rd_present)
            legacy_ordinals = _parse_indices(row.rd_raw_frame_indices_json)
            tensor_indices = _parse_indices(row.rd_tensor_frame_indices_json)
            expected_count = int(row.rd_frame_count)
            if (
                len(legacy_ordinals) != len(tensor_indices)
                or len(tensor_indices) != expected_count
                or expected_count > BROADBAND_FRAME_CAP
                or present != bool(expected_count)
            ):
                raise ValueError(f"V18 event frame mapping/count is invalid: {row.sample_id}")
            resolved = canonical_frame_provenance.resolve_event(
                parent_sample_id=parent_id,
                track_id=track_id,
                track_point_index=int(row.track_point_index),
                tensor_frame_indices=tensor_indices,
                legacy_mat_list_ordinals=legacy_ordinals,
                rd_tensor_path=str(row.rd_tensor_path),
            )
            raw_indices = [
                int(value["raw_dat_frame_index_zero_based"]) for value in resolved
            ]
            if set(raw_indices) & used_raw_indices:
                raise ValueError(
                    f"V18 raw-DAT frame is assigned to multiple events: {row.sample_id}"
                )
            used_raw_indices.update(raw_indices)
            mismatch_count = sum(
                int(value["legacy_mat_list_ordinal_zero_based"])
                != int(value["raw_dat_frame_index_zero_based"])
                for value in resolved
            )
            legacy_mismatch_frame_count += mismatch_count
            if mismatch_count:
                legacy_mismatch_track_ids.add(track_id)

            event = extract_current_event_broadband_micro_doppler(
                raw_frames,
                raw_indices,
                track_point_index=int(row.track_point_index),
            )
            if present:
                if amplitude is None or velocity is None:
                    raise ValueError(f"V18 RD-present parent lacks canonical tensor: {parent_id}")
                if min(tensor_indices) < 0 or max(tensor_indices) >= amplitude.shape[0]:
                    raise ValueError(
                        f"V18 canonical tensor frame index is out of bounds: {row.sample_id}"
                    )
                physical = extract_current_event_rd_physical(
                    amplitude[np.asarray(tensor_indices, dtype=np.int64)], velocity
                )
                if not event.valid or not physical.valid:
                    raise ValueError(
                        f"V18 present event produced invalid physical evidence: {row.sample_id}"
                    )
                summary_values[event_cursor] = event.values
                physical_values[event_cursor] = physical.values
                valid[event_cursor] = True
                active = event.frame_mask
                if event.raw_frame_indices[active].astype(int).tolist() != raw_indices:
                    raise ValueError(
                        f"V18 extractor returned different raw-DAT indices: {row.sample_id}"
                    )
                ragged["raw_dat_frame_indices_zero_based"].append(
                    event.raw_frame_indices[active]
                )
                ragged["legacy_mat_list_ordinals_zero_based"].append(
                    np.asarray(
                        [
                            value["legacy_mat_list_ordinal_zero_based"]
                            for value in resolved
                        ],
                        dtype=np.int32,
                    )
                )
                ragged["canonical_tensor_frame_indices_zero_based"].append(
                    np.asarray(tensor_indices, dtype=np.int32)
                )
                ragged["mat_acquisition_frame_numbers_one_based"].append(
                    np.asarray(
                        [
                            value["mat_acquisition_frame_number_one_based"]
                            for value in resolved
                        ],
                        dtype=np.int32,
                    )
                )
                for name, value in (
                    ("frame_values", event.frame_values[active]),
                    ("frame_feature_reliable", event.frame_feature_reliable[active]),
                    (
                        "residual_doppler_profiles",
                        event.residual_doppler_profiles[active].astype(np.float16),
                    ),
                    (
                        "envelope_modulation_profiles",
                        event.envelope_modulation_profiles[active].astype(np.float16),
                    ),
                    ("target_range_indices", event.target_range_indices[active]),
                    ("target_doppler_indices", event.target_doppler_indices[active]),
                    ("target_doppler_mps", event.target_doppler_mps[active]),
                    (
                        "refined_body_doppler_mps",
                        event.refined_body_doppler_mps[active],
                    ),
                    ("velocity_nyquist_mps", event.velocity_nyquist_mps[active]),
                    ("rayleigh_velocity_mps", event.rayleigh_velocity_mps[active]),
                    (
                        "residual_sideband_snr_db",
                        event.residual_sideband_snr_db[active],
                    ),
                    (
                        "envelope_modulation_snr_db",
                        event.envelope_modulation_snr_db[active],
                    ),
                ):
                    ragged[name].append(np.asarray(value))
                selected_frames = [raw_frames[index] for index in raw_indices]
                ragged["frame_prt_num"].append(
                    np.asarray([value.prt_num for value in selected_frames], dtype=np.int32)
                )
                ragged["frame_prt_seconds"].append(
                    np.asarray(
                        [value.prt_seconds for value in selected_frames], dtype=np.float64
                    )
                )
                ragged["frame_frequency_hz"].append(
                    np.asarray(
                        [value.frequency_hz for value in selected_frames], dtype=np.float64
                    )
                )
                ragged["frame_cpi_count"].append(
                    np.asarray([value.cpi_count for value in selected_frames], dtype=np.int64)
                )
                for prt_num in {int(value.prt_num) for value in selected_frames}:
                    prt_parent_support[prt_num][label].add(parent_id)
                for mode in {
                    broadband_sampling_mode(
                        value.prt_num, value.prt_seconds, value.frequency_hz
                    )
                    for value in selected_frames
                }:
                    sampling_mode_parent_support[
                        broadband_sampling_mode_name(mode)
                    ][label].add(parent_id)
                selected_raw_frame_count += int(active.sum())
                reliable_frame_count += int(event.frame_feature_reliable[active].sum())
                if not bool(event.frame_feature_reliable[active].any()):
                    neutral_event_count += 1
                    if np.any(event.values != 0.0):
                        raise ValueError(
                            f"V18 uncalibrated event is not intervention-neutral: {row.sample_id}"
                        )
            elif event.valid or np.any(event.values != 0.0):
                raise ValueError(
                    f"V18 RD-missing event must have zero micro-Doppler evidence: {row.sample_id}"
                )
            event_offsets.append(selected_raw_frame_count)
            event_cursor += 1

        if parent_number == 1 or parent_number % 50 == 0 or parent_number == len(grouped):
            print(
                f"[V18 build] split={split} parent={parent_number}/{len(grouped)} "
                f"events={event_cursor}/{len(selected)} raw_frames={selected_raw_frame_count}",
                flush=True,
            )

    if event_cursor != len(selected) or event_offsets[-1] != selected_raw_frame_count:
        raise ValueError(f"V18 {split} event/evidence offset accounting failed")
    expected_valid = selected["rd_present"].map(_as_bool).to_numpy(dtype=np.bool_)
    if not np.array_equal(valid, expected_valid):
        raise ValueError(f"V18 {split} valid mask differs from current-event RD presence")
    if np.any(summary_values[~valid] != 0.0) or np.any(physical_values[~valid] != 0.0):
        raise ValueError(f"V18 {split} missing evidence rows are not exactly zero")

    array_specs: dict[str, tuple[type[Any] | np.dtype[Any], tuple[int, ...]]] = {
        "raw_dat_frame_indices_zero_based": (np.int32, ()),
        "legacy_mat_list_ordinals_zero_based": (np.int32, ()),
        "canonical_tensor_frame_indices_zero_based": (np.int32, ()),
        "mat_acquisition_frame_numbers_one_based": (np.int32, ()),
        "frame_values": (np.float32, (BROADBAND_FRAME_VALUE_DIM,)),
        "frame_feature_reliable": (np.bool_, ()),
        "residual_doppler_profiles": (
            np.float16,
            (BROADBAND_DOPPLER_PROFILE_BIN_COUNT,),
        ),
        "envelope_modulation_profiles": (
            np.float16,
            (BROADBAND_MODULATION_PROFILE_BIN_COUNT,),
        ),
        "target_range_indices": (np.int16, ()),
        "target_doppler_indices": (np.int32, ()),
        "target_doppler_mps": (np.float32, ()),
        "refined_body_doppler_mps": (np.float32, ()),
        "velocity_nyquist_mps": (np.float32, ()),
        "rayleigh_velocity_mps": (np.float32, ()),
        "residual_sideband_snr_db": (np.float32, ()),
        "envelope_modulation_snr_db": (np.float32, ()),
        "frame_prt_num": (np.int32, ()),
        "frame_prt_seconds": (np.float64, ()),
        "frame_frequency_hz": (np.float64, ()),
        "frame_cpi_count": (np.int64, ()),
    }
    flat = {
        name: _concatenate(ragged[name], dtype=dtype, empty_shape=shape)
        for name, (dtype, shape) in array_specs.items()
    }
    if any(len(value) != selected_raw_frame_count for value in flat.values()):
        raise ValueError(f"V18 {split} ragged evidence arrays are not aligned")
    if not np.array_equal(
        flat["raw_dat_frame_indices_zero_based"] + 1,
        flat["mat_acquisition_frame_numbers_one_based"],
    ):
        raise ValueError(f"V18 {split} MAT acquisition/raw-DAT mapping is invalid")

    support = {
        str(prt_num): {
            label: len(labels[label]) for label in DBB_LABELS
        }
        for prt_num, labels in sorted(prt_parent_support.items())
    }
    mode_support = {
        mode_name: {label: len(labels[label]) for label in DBB_LABELS}
        for mode_name, labels in sorted(sampling_mode_parent_support.items())
    }
    arrays = {
        "sample_ids": sample_ids,
        "summary_values": summary_values,
        "physical_values": physical_values,
        "valid": valid,
        "event_offsets": np.asarray(event_offsets, dtype=np.int64),
        **flat,
    }
    statistics = {
        "split": split,
        "parent_count": len(grouped),
        "parent_label_counts": dict(sorted(parent_label_counts.items())),
        "event_count": len(selected),
        "source_pruned21_build_access_counts": None,
        "stable8_prepare_access_counts": {
            "raw_file_count": raw_file_access_count,
            "raw_frame_count": raw_frame_count,
            "rd_tensor_count": len(rd_cache.accessed_paths),
        },
        "payload_access_semantics": (
            "counts describe this stable8 prepare invocation; no reusable "
            "pruned21 evidence was used"
        ),
        "selected_current_event_raw_frame_count": selected_raw_frame_count,
        "feature_reliable_frame_count": reliable_frame_count,
        "feature_reliable_frame_fraction": (
            reliable_frame_count / selected_raw_frame_count
            if selected_raw_frame_count
            else 0.0
        ),
        "uncalibrated_neutral_intervention_event_count": neutral_event_count,
        "parent_support_by_prt_num_and_label": support,
        "parent_support_by_sampling_mode_and_label": mode_support,
        "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count": (
            legacy_mismatch_frame_count
        ),
        "legacy_mat_list_ordinal_mismatch_track_ids": sorted(
            legacy_mismatch_track_ids
        ),
        "micro_doppler": _event_stats(summary_values, valid),
        "rd_physical": _event_stats(physical_values, valid),
        "test_raw_payload_access_count": 0,
        "test_rd_payload_access_count": 0,
    }
    return arrays, statistics


def _reuse_prepared_evidence(
    root: Path,
    *,
    index: pd.DataFrame,
    source_hashes: dict[str, str],
) -> tuple[
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    manifest_path = root / REUSABLE_EVIDENCE_MANIFEST
    readiness_path = root / REUSABLE_BLOCKED_READINESS
    if not manifest_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError(
            "V18 stable8 evidence reuse requires both the pruned21 manifest and "
            f"blocked readiness report: manifest={manifest_path}, report={readiness_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    failed_checks = sorted(
        name
        for name, passed in dict(readiness.get("checks") or {}).items()
        if passed is not True
    )
    train_report = dict((readiness.get("split_reports") or {}).get("train") or {})
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("manifest_type") != REUSABLE_MANIFEST_TYPE
        or manifest.get("paired_protocol") != REUSABLE_PAIRED_PROTOCOL
        or manifest.get("test_excluded") is not True
        or int(manifest.get("test_raw_payload_access_count", -1)) != 0
        or int(manifest.get("test_rd_payload_access_count", -1)) != 0
        or readiness.get("status") != "blocked"
        or readiness.get("schema_version") != REUSABLE_READINESS_SCHEMA
        or readiness.get("paired_protocol") != REUSABLE_PAIRED_PROTOCOL
        or failed_checks != [REUSABLE_ONLY_FAILED_CHECK]
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or readiness.get("formal_training_at_report_generation") != "未执行"
        or set(train_report.get("direction_consistent_informative_feature_names") or [])
        != set(BROADBAND_MODE_STABLE_INFORMATIVE_NAMES)
        or set(BROADBAND_MICRO_DOPPLER_NAMES)
        != (
            set(BROADBAND_MODE_STABLE_INFORMATIVE_NAMES)
            - set(BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES)
        )
    ):
        raise ValueError(
            "V18 reusable pruned21 evidence/readiness does not prove the frozen stable8 selection"
        )
    contract_relative = Path(str(manifest.get("data_contract", "")))
    contract_path = _resolve(root, contract_relative)
    report_hashes = dict(readiness.get("contract_hashes") or {})
    if (
        not contract_path.is_file()
        or _sha256(contract_path) != str(manifest.get("data_contract_sha256"))
        or report_hashes
        != {
            "index_sha256": source_hashes["index"],
            "contract_sha256": str(manifest.get("data_contract_sha256")),
            "stats_sha256": source_hashes["stats"],
        }
        or str(manifest.get("source_index_sha256")) != source_hashes["index"]
        or str(manifest.get("source_contract_sha256")) != source_hashes["contract"]
        or str(manifest.get("source_provenance_sha256"))
        != source_hashes["provenance"]
        or str(manifest.get("source_canonical_frame_index_sha256"))
        != source_hashes["canonical_frame_index"]
        or dict(manifest.get("micro_doppler_contract") or {}).get("summary_schema")
        != REUSABLE_SUMMARY_SCHEMA
    ):
        raise ValueError("V18 reusable evidence provenance/hash chain is stale")

    built_arrays: dict[str, dict[str, np.ndarray]] = {}
    split_stats: dict[str, dict[str, Any]] = {}
    selected_indices = [REUSABLE_MODEL_NAMES.index(name) for name in BROADBAND_MICRO_DOPPLER_NAMES]
    for split in SPLITS:
        selected = index[
            index["split_name"].astype(str).eq(split)
            & index["label_name"].astype(str).isin(DBB_LABELS)
        ].copy()
        selected = selected.sort_values(
            ["parent_sample_id", "causal_event_index"], kind="stable"
        ).reset_index(drop=True)
        cache_entry = dict((manifest.get("cache_files") or {}).get(split) or {})
        paths: dict[str, Path] = {}
        for name in ("physical", "summary", "stft"):
            row = dict(cache_entry.get(name) or {})
            path = _resolve(root, str(row.get("path", "")))
            if not path.is_file() or _sha256(path) != str(row.get("sha256")):
                raise ValueError(
                    f"V18 reusable {split} {name} cache is missing or hash-mismatched"
                )
            paths[name] = path
        physical = _load_npz(paths["physical"])
        old_summary = _load_npz(paths["summary"])
        evidence = _load_npz(paths["stft"])
        expected_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
        if (
            set(physical) != {"sample_ids", "values", "valid"}
            or set(old_summary) != {"sample_ids", "values", "valid"}
            or set(evidence) != set(EVIDENCE_FIELD_NAMES)
            or not np.array_equal(physical["sample_ids"].astype(str), expected_ids)
            or not np.array_equal(old_summary["sample_ids"].astype(str), expected_ids)
            or not np.array_equal(evidence["sample_ids"].astype(str), expected_ids)
            or not np.array_equal(physical["valid"], old_summary["valid"])
            or not np.array_equal(evidence["valid"], old_summary["valid"])
            or physical["values"].shape != (len(selected), RD_PHYSICAL_VALUE_DIM)
            or physical["values"].dtype != np.float32
            or old_summary["values"].shape != (len(selected), len(REUSABLE_MODEL_NAMES))
            or old_summary["values"].dtype != np.float32
            or evidence["frame_values"].shape[1:] != (BROADBAND_FRAME_VALUE_DIM,)
            or evidence["frame_values"].dtype != np.float32
        ):
            raise ValueError(f"V18 reusable {split} cache alignment/schema is invalid")
        summary_values = _reaggregate_summary_from_evidence(
            evidence, event_count=len(selected)
        )
        expected_from_old = old_summary["values"][:, selected_indices]
        if not np.array_equal(summary_values, expected_from_old):
            raise ValueError(
                f"V18 stable8 {split} reaggregation differs from same-name pruned21 values"
            )
        statistics = json.loads(json.dumps(dict(cache_entry.get("statistics") or {})))
        source_access_counts = {
            "raw_file_count": int(statistics.pop("raw_file_access_count")),
            "raw_frame_count": int(
                statistics.pop("raw_frame_count_in_accessed_files")
            ),
            "rd_tensor_count": int(
                statistics.pop("canonical_rd_tensor_accessed_count")
            ),
        }
        statistics["micro_doppler"] = _event_stats(
            summary_values, old_summary["valid"]
        )
        statistics["source_pruned21_build_access_counts"] = source_access_counts
        statistics["stable8_prepare_access_counts"] = {
            "raw_file_count": 0,
            "raw_frame_count": 0,
            "rd_tensor_count": 0,
        }
        statistics["payload_access_semantics"] = (
            "source_pruned21_build_access_counts are historical counts from the "
            "SHA-256-bound evidence build; stable8_prepare_access_counts describe "
            "this reaggregation invocation"
        )
        statistics["summary_reaggregation_source"] = (
            "SHA-256-bound pruned21 frame evidence; no raw/RD payload re-read"
        )
        built_arrays[split] = {
            "sample_ids": expected_ids,
            "summary_values": summary_values,
            "physical_values": physical["values"],
            "valid": old_summary["valid"],
            **evidence,
        }
        split_stats[split] = statistics
    audit = {
        "mode": "reuse_sha256_bound_pruned21_frame_evidence",
        "source_manifest": REUSABLE_EVIDENCE_MANIFEST.as_posix(),
        "source_manifest_sha256": _sha256(manifest_path),
        "source_blocked_readiness": REUSABLE_BLOCKED_READINESS.as_posix(),
        "source_blocked_readiness_sha256": _sha256(readiness_path),
        "source_only_failed_check": REUSABLE_ONLY_FAILED_CHECK,
        "selection_population": "train_parent_only",
        "validation_used_for_feature_selection": False,
        "test_payload_access_count": 0,
        "retained_mode_stable_informative_names": list(
            BROADBAND_MODE_STABLE_INFORMATIVE_NAMES
        ),
        "collinear_evidence_only_names": list(
            BROADBAND_COLLINEAR_EVIDENCE_ONLY_NAMES
        ),
        "model_input_names": list(BROADBAND_MICRO_DOPPLER_NAMES),
    }
    return built_arrays, split_stats, audit


def _build_contract(
    source_contract: dict[str, Any],
    *,
    source_hashes: dict[str, str],
    cache_hashes: dict[str, dict[str, str]],
    split_stats: dict[str, dict[str, Any]],
    preparation_source: dict[str, Any],
) -> dict[str, Any]:
    contract = json.loads(json.dumps(source_contract))
    feature_contract = dict(contract["feature_contract"])
    feature_contract["rd_physical"] = rd_physical_contract()
    feature_contract["micro_doppler"] = broadband_micro_doppler_contract()
    contract.update(
        {
            "schema_version": VERSION,
            "contract_name": (
                "cq08_v16_grouped_dbb_rd_physical4_broadband_multiscale_"
                "micro_doppler8_stable_paired_v3"
            ),
            "contract_status": "prepared_for_manual_formal_paired_training",
            "source_contract": SOURCE_CONTRACT.as_posix(),
            "source_contract_sha256": source_hashes["contract"],
            "source_index": SOURCE_INDEX.as_posix(),
            "source_index_sha256": source_hashes["index"],
            "source_provenance": SOURCE_PROVENANCE.as_posix(),
            "source_provenance_sha256": source_hashes["provenance"],
            "source_normalization_stats": SOURCE_STATS.as_posix(),
            "source_normalization_stats_sha256": source_hashes["stats"],
            "source_canonical_frame_index": SOURCE_CANONICAL_FRAME_INDEX.as_posix(),
            "source_canonical_frame_index_sha256": source_hashes[
                "canonical_frame_index"
            ],
            "feature_contract": feature_contract,
            "v18_preparation_source": preparation_source,
            "v18_scope": {
                "task": "dbb_specialist",
                "labels": list(DBB_LABELS),
                "splits_processed": list(SPLITS),
                "test_rows_read_for_split_exclusion_only": True,
                "test_raw_payload_access_count": 0,
                "test_rd_payload_access_count": 0,
                "complete_evidence_profiles_are_model_input": False,
                "treatment_model_input": list(BROADBAND_MICRO_DOPPLER_NAMES),
                "prt_num_is_model_input": False,
                "uncalibrated_sampling_mode_intervention": "exact_zero",
            },
            "rd_physical_cache": {
                "schema": RD_PHYSICAL_SCHEMA,
                "cache_scope": (
                    "V16 capture/content-grouped DBB train and validation events only"
                ),
                "files": {
                    split: {
                        "path": _physical_cache(split).as_posix(),
                        "sha256": cache_hashes[split]["physical"],
                        **split_stats[split]["rd_physical"],
                    }
                    for split in SPLITS
                },
                "test_excluded": True,
                "test_rd_payload_access_count": 0,
            },
            "micro_doppler_cache": {
                "summary_schema": BROADBAND_MICRO_DOPPLER_SCHEMA,
                "evidence_schema": BROADBAND_MICRO_DOPPLER_EVIDENCE_SCHEMA,
                "summary_files": {
                    split: {
                        "path": _summary_cache(split).as_posix(),
                        "sha256": cache_hashes[split]["summary"],
                        **split_stats[split]["micro_doppler"],
                    }
                    for split in SPLITS
                },
                "evidence_files": {
                    split: {
                        "path": _evidence_cache(split).as_posix(),
                        "sha256": cache_hashes[split]["evidence"],
                        "event_count": split_stats[split]["event_count"],
                        "selected_frame_count": split_stats[split][
                            "selected_current_event_raw_frame_count"
                        ],
                    }
                    for split in SPLITS
                },
                "raw_frame_mapping": {
                    "legacy_index_field": "rd_raw_frame_indices_json",
                    "legacy_index_is_raw_dat_frame_index": False,
                    "canonical_lookup": (
                        "parent_sample_id plus canonical tensor frame index to MAT "
                        "acquisition identity to zero-based raw-DAT frame"
                    ),
                    "audit": {
                        split: {
                            "mismatch_frame_count": split_stats[split][
                                "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count"
                            ],
                            "affected_track_ids": split_stats[split][
                                "legacy_mat_list_ordinal_mismatch_track_ids"
                            ],
                        }
                        for split in SPLITS
                    },
                },
                "sampling_metadata_is_model_input": False,
                "evidence_profiles_are_model_input": False,
                "test_excluded": True,
                "test_raw_payload_access_count": 0,
            },
            "training_contract_v18": {
                "paired_protocol": PAIRED_PROTOCOL,
                "control": "V16 DBB local architecture plus current-event rd_physical4",
                "treatment": (
                    "Control plus exactly 8 train-only mode-stable current-event broadband multiscale "
                    "micro-Doppler summaries"
                ),
                "shared_initialization": (
                    "all shared tensors copied exactly; Treatment's 8 new Fusion "
                    "columns exactly zero"
                ),
                "seed": SEED,
                "dataloader_seed": SEED,
                "epochs": 20,
                "early_stopping": False,
                "checkpoint_selection": CHECKPOINT_SELECTION,
                "formal_training": "未执行",
                "full_cascade": "未执行",
                "oof": "未执行",
                "test": "未执行",
                "stable": "未执行",
                "p95": "未执行",
            },
            "v18_acceptance": {
                "primary_scope": "DBB validation parent-final local task only",
                "strictly_greater_than_parent_macro_f1_improvement": 0.02,
                "strictly_greater_than_drone_bird_weak_floor_improvement": 0.02,
                "all_DBB_class_precision_recall_f1_minimum": 0.95,
                "directional_guard": (
                    "Drone to Bird and Bird to Drone may not increase; their total "
                    "must decrease"
                ),
                "same_epoch_attribution_guards_required": True,
                "acceptance_is_not_epoch_selection": True,
                "not_claimed": [
                    "five-class cascade performance",
                    "strict-open-set rejection performance",
                    "test generalization",
                    "stable performance",
                    "P95 latency",
                ],
            },
        }
    )
    contract["source_purity"]["raw_echo_use"] = (
        "CQ08-internal train/validation DBB current-event complex I/Q only; the "
        "separately named external raw source remains excluded"
    )
    contract["source_purity"]["cq08_original_echo_use"] = (
        "allowed because it belongs to CQ08 and is SHA-256-bound to V16 "
        "capture/content provenance"
    )
    contract["data_scope"]["test_payload_access_for_hash_only"] = False
    contract["data_scope"]["test_used_for_normalization"] = False
    contract["data_scope"]["test_used_for_checkpoint_selection"] = False
    return contract


def _build_manifest(
    *,
    contract_hash: str,
    source_hashes: dict[str, str],
    cache_hashes: dict[str, dict[str, str]],
    split_stats: dict[str, dict[str, Any]],
    preparation_source: dict[str, Any],
) -> dict[str, Any]:
    evidence_fields = [
        "sample_ids",
        "valid",
        "event_offsets",
        "raw_dat_frame_indices_zero_based",
        "legacy_mat_list_ordinals_zero_based",
        "canonical_tensor_frame_indices_zero_based",
        "mat_acquisition_frame_numbers_one_based",
        "frame_values",
        "frame_feature_reliable",
        "residual_doppler_profiles",
        "envelope_modulation_profiles",
        "target_range_indices",
        "target_doppler_indices",
        "target_doppler_mps",
        "refined_body_doppler_mps",
        "velocity_nyquist_mps",
        "rayleigh_velocity_mps",
        "residual_sideband_snr_db",
        "envelope_modulation_snr_db",
        "frame_prt_num",
        "frame_prt_seconds",
        "frame_frequency_hz",
        "frame_cpi_count",
    ]
    return {
        "schema_version": VERSION,
        "manifest_type": (
            "cq08_dbb_current_event_broadband_multiscale_"
            "micro_doppler8_stable_paired_cache_v3"
        ),
        "paired_protocol": PAIRED_PROTOCOL,
        "source_name": "CQ08",
        "source_index": SOURCE_INDEX.as_posix(),
        "source_index_sha256": source_hashes["index"],
        "source_contract": SOURCE_CONTRACT.as_posix(),
        "source_contract_sha256": source_hashes["contract"],
        "source_provenance": SOURCE_PROVENANCE.as_posix(),
        "source_provenance_sha256": source_hashes["provenance"],
        "source_canonical_frame_index": SOURCE_CANONICAL_FRAME_INDEX.as_posix(),
        "source_canonical_frame_index_sha256": source_hashes[
            "canonical_frame_index"
        ],
        "data_contract": DATA_CONTRACT.as_posix(),
        "data_contract_sha256": contract_hash,
        "rd_physical_contract": rd_physical_contract(),
        "micro_doppler_contract": broadband_micro_doppler_contract(),
        "preparation_source": preparation_source,
        "cache_files": {
            split: {
                "physical": {
                    "path": _physical_cache(split).as_posix(),
                    "sha256": cache_hashes[split]["physical"],
                },
                "summary": {
                    "path": _summary_cache(split).as_posix(),
                    "sha256": cache_hashes[split]["summary"],
                },
                "stft": {
                    "path": _evidence_cache(split).as_posix(),
                    "sha256": cache_hashes[split]["evidence"],
                    "field_name_compatibility_note": (
                        "task config retains the historical stft-cache field name, "
                        "but this V18 file contains broadband evidence profiles and "
                        "no model-input STFT tensor"
                    ),
                },
                "statistics": split_stats[split],
            }
            for split in SPLITS
        },
        "cache_alignment_fields": {
            "summary": ["sample_ids", "values", "valid"],
            "physical": ["sample_ids", "values", "valid"],
            "stft": evidence_fields,
        },
        "sample_id_model_input": False,
        "raw_frame_index_model_input": False,
        "sampling_metadata_model_input": False,
        "legacy_mat_list_ordinal_is_raw_dat_frame_index": False,
        "complete_evidence_profiles_are_model_input": False,
        "test_excluded": True,
        "test_raw_payload_access_count": 0,
        "test_rd_payload_access_count": 0,
        "formal_training": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }


def _build_config(
    source: dict[str, Any],
    *,
    arm: str,
    source_hashes: dict[str, str],
    contract_hash: str,
    manifest_hash: str,
    cache_hashes: dict[str, dict[str, str]],
) -> dict[str, Any]:
    payload = json.loads(json.dumps(source))
    artifact = _artifact_dir(arm)
    payload["schema_version"] = VERSION
    payload["config_name"] = (
        "RADAR-LN3-V18-cq08-liquid-dbb-specialist-rd-physical4-"
        f"broadband-multiscale-micro-doppler8-stable-{arm}-seed42"
    )
    payload["data"].update(
        {
            "index_path": SOURCE_INDEX.as_posix(),
            "contract_path": DATA_CONTRACT.as_posix(),
            "normalization_stats_path": SOURCE_STATS.as_posix(),
            "index_schema_version": SOURCE_VERSION,
            "contract_schema_version": VERSION,
            "expected_index_sha256": source_hashes["index"],
            "expected_contract_sha256": contract_hash,
            "expected_stats_sha256": source_hashes["stats"],
            "rd_physical_schema": RD_PHYSICAL_SCHEMA,
            "rd_physical_mask_mode": RD_PHYSICAL_MASK_MODE,
            "rd_physical_cache_manifest_path": CACHE_MANIFEST.as_posix(),
            "expected_rd_physical_cache_manifest_sha256": manifest_hash,
            "train_rd_physical_cache_path": _physical_cache("train").as_posix(),
            "validation_rd_physical_cache_path": _physical_cache("val").as_posix(),
            "expected_train_rd_physical_cache_sha256": cache_hashes["train"][
                "physical"
            ],
            "expected_validation_rd_physical_cache_sha256": cache_hashes["val"][
                "physical"
            ],
            "micro_doppler_schema": BROADBAND_MICRO_DOPPLER_SCHEMA,
            "micro_doppler_mask_mode": BROADBAND_MICRO_DOPPLER_MASK_MODE,
            "micro_doppler_cache_manifest_path": CACHE_MANIFEST.as_posix(),
            "expected_micro_doppler_cache_manifest_sha256": manifest_hash,
            "train_micro_doppler_cache_path": _summary_cache("train").as_posix(),
            "validation_micro_doppler_cache_path": _summary_cache("val").as_posix(),
            "expected_train_micro_doppler_cache_sha256": cache_hashes["train"][
                "summary"
            ],
            "expected_validation_micro_doppler_cache_sha256": cache_hashes["val"][
                "summary"
            ],
            "train_micro_doppler_stft_cache_path": _evidence_cache("train").as_posix(),
            "validation_micro_doppler_stft_cache_path": _evidence_cache("val").as_posix(),
            "expected_train_micro_doppler_stft_cache_sha256": cache_hashes["train"][
                "evidence"
            ],
            "expected_validation_micro_doppler_stft_cache_sha256": cache_hashes["val"][
                "evidence"
            ],
            "micro_doppler_frame_provenance_path": (
                SOURCE_CANONICAL_FRAME_INDEX.as_posix()
            ),
            "expected_micro_doppler_frame_provenance_sha256": source_hashes[
                "canonical_frame_index"
            ],
        }
    )
    payload["model"]["rd_physical_dim"] = RD_PHYSICAL_VALUE_DIM
    payload["model"]["micro_doppler_dim"] = (
        BROADBAND_MICRO_DOPPLER_VALUE_DIM if arm == "treatment" else 0
    )
    payload["train"].update(
        {
            "seed": SEED,
            "dataloader_seed": SEED,
            "epochs": 20,
            "early_stopping_patience": 0,
            "checkpoint_selection_metric": CHECKPOINT_SELECTION,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "checkpoint_schema": (
                "RADAR-LN3-V18-dbb-broadband-multiscale-micro-doppler-"
                "paired-checkpoint-v3-stable8"
            ),
            "paired_protocol": PAIRED_PROTOCOL,
            "paired_arm": arm,
            "paired_seed": SEED,
            "paired_counterpart_config_path": _config_path(
                "treatment" if arm == "control" else "control"
            ).as_posix(),
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": False,
        }
    )
    return payload


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    source_paths = {
        "index": root / SOURCE_INDEX,
        "contract": root / SOURCE_CONTRACT,
        "provenance": root / SOURCE_PROVENANCE,
        "stats": root / SOURCE_STATS,
        "canonical_frame_index": root / SOURCE_CANONICAL_FRAME_INDEX,
        "config": root / SOURCE_DBB_CONFIG,
    }
    for path in source_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    nonempty_artifacts = [
        root / _artifact_dir(arm)
        for arm in ("control", "treatment")
        if (root / _artifact_dir(arm)).exists()
        and any((root / _artifact_dir(arm)).iterdir())
    ]
    if nonempty_artifacts:
        raise FileExistsError(
            "refusing to rebuild V18 data/config outputs after formal artifacts exist; "
            f"first={nonempty_artifacts[0]}, count={len(nonempty_artifacts)}"
        )
    # A prior prepare may have been interrupted after writing only part of the
    # output set.  The exclusive writers below reuse an existing file only
    # after a full array/text equality check and refuse every difference.  This
    # permits safe restart without silently overwriting a versioned cache.
    source_hashes = {name: _sha256(path) for name, path in source_paths.items()}

    index = pd.read_csv(source_paths["index"], low_memory=False)
    required = {
        "schema_version",
        "sample_id",
        "parent_sample_id",
        "track_id",
        "track_point_index",
        "causal_event_index",
        "split_name",
        "label_name",
        "rd_present",
        "rd_frame_count",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
        "rd_raw_frame_indices_json",
        "cq08_raw_echo_file",
    }
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"V16 index lacks V18 source fields: {missing}")
    if index["schema_version"].astype(str).ne(SOURCE_VERSION).any():
        raise ValueError("V18 source index contains a non-V16 row")
    if index["sample_id"].astype(str).duplicated().any():
        raise ValueError("V18 source index contains duplicate sample IDs")
    if set(index["split_name"].astype(str)) != {"train", "val", "test"}:
        raise ValueError("V18 source index does not preserve train/val/sealed-test splits")

    reusable_manifest_exists = (root / REUSABLE_EVIDENCE_MANIFEST).is_file()
    reusable_readiness_exists = (root / REUSABLE_BLOCKED_READINESS).is_file()
    if reusable_manifest_exists != reusable_readiness_exists:
        raise FileNotFoundError(
            "V18 stable8 found only one reusable pruned21 provenance artifact; "
            "both manifest and blocked readiness are required"
        )
    write_status: dict[str, dict[str, str]] = {}
    cache_hashes: dict[str, dict[str, str]] = {}
    if reusable_manifest_exists:
        built_arrays, split_stats, preparation_source = _reuse_prepared_evidence(
            root, index=index, source_hashes=source_hashes
        )
        canonical_audit = {
            "reused_from_sha256_bound_pruned21_evidence": True,
            "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count": 46,
            "affected_track_ids": [1010],
        }
    else:
        provenance = json.loads(
            source_paths["provenance"].read_text(encoding="utf-8")
        )
        provenance_by_parent = {
            str(value["parent_sample_id"]): dict(value)
            for value in provenance.get("records", [])
        }
        if len(provenance_by_parent) != int(provenance.get("parent_count", -1)):
            raise ValueError("V16 provenance parent records are incomplete")
        canonical = _CanonicalFrameProvenance(
            root, source_paths["canonical_frame_index"]
        )
        split_stats = {}
        built_arrays = {}
        for split in SPLITS:
            arrays, statistics = _build_split(
                root, split, index, provenance_by_parent, canonical
            )
            split_stats[split] = statistics
            built_arrays[split] = arrays
        preparation_source = {
            "mode": "full_raw_and_rd_rebuild",
            "selection_population": "frozen_train_parent_only",
            "validation_used_for_feature_selection": False,
            "test_payload_access_count": 0,
            "model_input_names": list(BROADBAND_MICRO_DOPPLER_NAMES),
        }
        canonical_audit = canonical.audit

    train_mode_support = split_stats["train"][
        "parent_support_by_sampling_mode_and_label"
    ]
    expected_train_mode_names = {
        broadband_sampling_mode_name(mode)
        for mode in BROADBAND_EXPECTED_TRAIN_SAMPLING_MODES
    }
    if set(train_mode_support) != expected_train_mode_names:
        raise ValueError(
            "V18 train sampling-mode population changed outside the frozen calibrated/neutral set: "
            f"actual={sorted(train_mode_support)}, expected={sorted(expected_train_mode_names)}"
        )
    for mode in BROADBAND_CANDIDATE_SAMPLING_MODES:
        mode_name = broadband_sampling_mode_name(mode)
        row = dict(train_mode_support.get(mode_name) or {})
        if any(
            int(row.get(label, 0))
            < BROADBAND_MIN_TRAIN_PARENTS_PER_CLASS_PER_MODE
            for label in DBB_LABELS
        ):
            raise ValueError(
                "V18 calibrated sampling mode lacks frozen train parent support: "
                f"mode={mode_name}, support={row}"
            )
    train_prt_support = split_stats["train"]["parent_support_by_prt_num_and_label"]
    prt_142 = dict(train_prt_support.get("142") or {})
    if prt_142 != {"DroneTarget": 4, "BirdTarget": 3, "BalloonTarget": 0}:
        raise ValueError(
            "V18 frozen PRT=142 unsupported-mode evidence changed: "
            f"actual={prt_142}"
        )

    # Only write versioned caches after all split-level extraction and frozen
    # support checks have passed.  A blocked build therefore cannot leave a
    # partially authoritative V18 cache set behind.
    for split in SPLITS:
        arrays = built_arrays[split]
        physical_fields = (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["physical_values"]),
            ("valid", arrays["valid"]),
        )
        summary_fields = (
            ("sample_ids", arrays["sample_ids"]),
            ("values", arrays["summary_values"]),
            ("valid", arrays["valid"]),
        )
        evidence_fields = tuple((name, arrays[name]) for name in EVIDENCE_FIELD_NAMES)
        write_status[split] = {
            "physical": _write_npz_new_or_identical(
                root / _physical_cache(split), physical_fields
            ),
            "summary": _write_npz_new_or_identical(
                root / _summary_cache(split), summary_fields
            ),
            "evidence": _write_npz_new_or_identical(
                root / _evidence_cache(split), evidence_fields
            ),
        }
        cache_hashes[split] = {
            "physical": _sha256(root / _physical_cache(split)),
            "summary": _sha256(root / _summary_cache(split)),
            "evidence": _sha256(root / _evidence_cache(split)),
        }
    del built_arrays

    source_contract = json.loads(source_paths["contract"].read_text(encoding="utf-8"))
    contract_payload = _build_contract(
        source_contract,
        source_hashes=source_hashes,
        cache_hashes=cache_hashes,
        split_stats=split_stats,
        preparation_source=preparation_source,
    )
    contract_status = _write_json_new_or_identical(root / DATA_CONTRACT, contract_payload)
    contract_hash = _sha256(root / DATA_CONTRACT)
    manifest_payload = _build_manifest(
        contract_hash=contract_hash,
        source_hashes=source_hashes,
        cache_hashes=cache_hashes,
        split_stats=split_stats,
        preparation_source=preparation_source,
    )
    manifest_status = _write_json_new_or_identical(root / CACHE_MANIFEST, manifest_payload)
    manifest_hash = _sha256(root / CACHE_MANIFEST)

    source_config = json.loads(source_paths["config"].read_text(encoding="utf-8"))
    config_status: dict[str, str] = {}
    config_hashes: dict[str, str] = {}
    for arm in ("control", "treatment"):
        config_path = root / _config_path(arm)
        artifact = root / _artifact_dir(arm)
        payload = _build_config(
            source_config,
            arm=arm,
            source_hashes=source_hashes,
            contract_hash=contract_hash,
            manifest_hash=manifest_hash,
            cache_hashes=cache_hashes,
        )
        expected_text = _json_text(payload)
        if artifact.exists() and any(artifact.iterdir()):
            if (
                not config_path.is_file()
                or config_path.read_text(encoding="utf-8") != expected_text
            ):
                raise FileExistsError(
                    "refusing to prepare when a non-empty V18 artifact is not "
                    f"bound to the identical generated config: {artifact}"
                )
        config_status[arm] = _write_json_new_or_identical(config_path, payload)
        config_hashes[arm] = _sha256(config_path)

    return {
        "status": "completed",
        "schema_version": VERSION,
        "paired_protocol": PAIRED_PROTOCOL,
        "write_status": {
            "caches": write_status,
            "contract": contract_status,
            "manifest": manifest_status,
            "configs": config_status,
        },
        "cache_hashes": cache_hashes,
        "contract": {"path": str(root / DATA_CONTRACT), "sha256": contract_hash},
        "manifest": {"path": str(root / CACHE_MANIFEST), "sha256": manifest_hash},
        "configs": {
            arm: {"path": str(root / _config_path(arm)), "sha256": config_hashes[arm]}
            for arm in ("control", "treatment")
        },
        "split_statistics": split_stats,
        "preparation_source": preparation_source,
        "canonical_frame_provenance_audit": canonical_audit,
        "test_raw_payload_access_count": 0,
        "test_rd_payload_access_count": 0,
        "formal_training": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the V18 DBB broadband multiscale micro-Doppler paired contract."
        )
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
