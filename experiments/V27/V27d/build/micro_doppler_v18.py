from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from build.micro_doppler_v17 import _CanonicalFrameProvenance, _CanonicalRD, _as_bool, _parse_indices, _resolve, _sha256
from radar_ln3.micro_doppler import parse_cq08_raw_echo
from radar_ln3.micro_doppler_broadband import BROADBAND_FRAME_CAP, BROADBAND_FRAME_VALUE_DIM, BROADBAND_MICRO_DOPPLER_VALUE_DIM, BROADBAND_MODULATION_PROFILE_BIN_COUNT, BROADBAND_DOPPLER_PROFILE_BIN_COUNT, broadband_sampling_mode, broadband_sampling_mode_name, extract_current_event_broadband_micro_doppler
from radar_ln3.rd_physical import RD_PHYSICAL_VALUE_DIM, extract_current_event_rd_physical


DBB_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")














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
    *,
    allowed_labels: Sequence[str] = DBB_LABELS,
    include_evidence: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build label-free current-event evidence for one explicitly scoped split.

    V18 itself keeps the historical three-DBB-label default.  A later
    audit-only caller may pass a wider explicit label scope to reproduce the
    *same* extraction for frozen-model routing diagnostics.  Labels select
    rows only; no label value enters either physical or micro-Doppler tensor.
    """

    scoped_labels = tuple(str(value) for value in allowed_labels)
    if not scoped_labels or len(set(scoped_labels)) != len(scoped_labels):
        raise ValueError("V18 split evidence scope must contain unique labels")
    if not isinstance(include_evidence, bool):
        raise ValueError("V18 split include_evidence must be boolean")
    selected = frame[
        frame["split_name"].astype(str).eq(split)
        & frame["label_name"].astype(str).isin(scoped_labels)
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

    ragged: dict[str, list[np.ndarray]] = (
        {
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
        if include_evidence
        else {}
    )
    raw_file_access_count = 0
    raw_frame_count = 0
    selected_raw_frame_count = 0
    reliable_frame_count = 0
    neutral_event_count = 0
    legacy_mismatch_frame_count = 0
    legacy_mismatch_track_ids: set[int] = set()
    parent_label_counts: Counter[str] = Counter()
    prt_parent_support: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {label: set() for label in scoped_labels}
    )
    sampling_mode_parent_support: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {label: set() for label in scoped_labels}
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
                if include_evidence:
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
                        np.asarray(
                            [value.prt_num for value in selected_frames], dtype=np.int32
                        )
                    )
                    ragged["frame_prt_seconds"].append(
                        np.asarray(
                            [value.prt_seconds for value in selected_frames],
                            dtype=np.float64,
                        )
                    )
                    ragged["frame_frequency_hz"].append(
                        np.asarray(
                            [value.frequency_hz for value in selected_frames],
                            dtype=np.float64,
                        )
                    )
                    ragged["frame_cpi_count"].append(
                        np.asarray(
                            [value.cpi_count for value in selected_frames], dtype=np.int64
                        )
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
    flat: dict[str, np.ndarray] = {}
    if include_evidence:
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
            label: len(labels[label]) for label in scoped_labels
        }
        for prt_num, labels in sorted(prt_parent_support.items())
    }
    mode_support = {
        mode_name: {label: len(labels[label]) for label in scoped_labels}
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
        "complete_evidence_profiles_materialized": include_evidence,
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













