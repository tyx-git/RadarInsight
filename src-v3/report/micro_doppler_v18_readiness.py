from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar_ln3.micro_doppler_broadband import (
    BROADBAND_CANDIDATE_SAMPLING_MODES,
    BROADBAND_EXPECTED_TRAIN_SAMPLING_MODES,
    BROADBAND_FRAME_CAP,
    BROADBAND_FRAME_NAMES,
    BROADBAND_FRAME_VALUE_DIM,
    BROADBAND_MICRO_DOPPLER_NAMES,
    BROADBAND_MICRO_DOPPLER_VALUE_DIM,
    BROADBAND_MIN_MODULATION_NYQUIST_HZ,
    BROADBAND_MIN_TRAIN_PARENTS_PER_CLASS_PER_MODE,
    BROADBAND_MODULATION_PROFILE_BIN_COUNT,
    BROADBAND_UNCALIBRATED_SAMPLING_MODES,
    BROADBAND_DOPPLER_PROFILE_BIN_COUNT,
    aggregate_broadband_frame_values,
    broadband_sampling_mode,
    broadband_sampling_mode_name,
)
from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)


CONFIGS = {
    arm: Path(
        "config/RADAR-LN3/RADAR-LN3-V18-cq08-liquid-dbb-specialist-"
        f"rd-physical4-broadband-multiscale-micro-doppler8-stable-{arm}-seed42-config.json"
    )
    for arm in ("control", "treatment")
}
OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V18-broadband-multiscale-micro-doppler8-stable-train-readiness.json"
)
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
SUMMARY_RECONSTRUCTION_ATOL = 1.0e-7
MIN_NONDEGENERATE_STD = 1.0e-4
MIN_NONDEGENERATE_UNIQUE = 8
MIN_DIRECTION_FREE_AUC = 0.55
MIN_ROBUST_EFFECT = 0.10


def _direction_free_auc(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if not left.size or not right.size:
        return 0.5
    values = np.concatenate([left, right])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(values):
        stop = cursor + 1
        while stop < len(values) and values[order[stop]] == values[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = 0.5 * (cursor + 1 + stop)
        cursor = stop
    rank_sum = float(ranks[: len(left)].sum(dtype=np.float64))
    auc = (
        rank_sum - len(left) * (len(left) + 1) / 2.0
    ) / float(len(left) * len(right))
    return float(max(auc, 1.0 - auc))


def _robust_effect(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    pooled = np.concatenate([left, right])
    scale = float(np.quantile(pooled, 0.75) - np.quantile(pooled, 0.25))
    if scale <= 1.0e-12:
        return 0.0
    return float((np.median(left) - np.median(right)) / scale)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _split_report(
    *,
    split: str,
    index: pd.DataFrame,
    summary_path: Path,
    evidence_path: Path,
) -> dict[str, Any]:
    selected = index[
        index["split_name"].astype(str).eq(split)
        & index["label_name"].astype(str).isin(LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    summary = _load_npz(summary_path)
    evidence = _load_npz(evidence_path)
    expected_summary_fields = {"sample_ids", "values", "valid"}
    expected_evidence_fields = {
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
    }
    if set(summary) != expected_summary_fields or set(evidence) != expected_evidence_fields:
        raise ValueError(f"V18 {split} cache field set differs from the frozen contract")

    sample_ids = summary["sample_ids"].astype(str)
    expected_ids = selected["sample_id"].astype(str).to_numpy()
    values = summary["values"]
    valid = summary["valid"]
    offsets = evidence["event_offsets"]
    if (
        not np.array_equal(sample_ids, expected_ids)
        or not np.array_equal(evidence["sample_ids"].astype(str), expected_ids)
        or not np.array_equal(evidence["valid"], valid)
        or values.shape != (len(selected), BROADBAND_MICRO_DOPPLER_VALUE_DIM)
        or values.dtype != np.float32
        or valid.shape != (len(selected),)
        or valid.dtype != np.bool_
        or offsets.shape != (len(selected) + 1,)
        or offsets.dtype != np.int64
        or offsets[0] != 0
    ):
        raise ValueError(f"V18 {split} cache alignment/shape contract failed")
    expected_valid = selected["rd_present"].map(
        lambda value: str(value).strip().lower() in {"1", "true", "yes", "y"}
    ).to_numpy(dtype=np.bool_)
    if not np.array_equal(valid, expected_valid):
        raise ValueError(f"V18 {split} valid mask differs from the index")
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
        or np.any(values[~valid] != 0.0)
    ):
        raise ValueError(f"V18 {split} summary values violate finite/bounds/missing rules")

    frame_count = int(offsets[-1])
    counts = np.diff(offsets)
    if np.any(counts < 0) or np.any(counts > BROADBAND_FRAME_CAP):
        raise ValueError(f"V18 {split} frame counts violate the frozen cap")
    if not np.array_equal(counts > 0, valid):
        raise ValueError(f"V18 {split} frame offsets differ from valid events")
    vector_fields = {
        name
        for name in expected_evidence_fields
        if name not in {"sample_ids", "valid", "event_offsets"}
    }
    if any(len(evidence[name]) != frame_count for name in vector_fields):
        raise ValueError(f"V18 {split} evidence arrays are not ragged-aligned")
    if (
        evidence["frame_values"].shape
        != (frame_count, BROADBAND_FRAME_VALUE_DIM)
        or evidence["frame_values"].dtype != np.float32
        or evidence["frame_feature_reliable"].dtype != np.bool_
        or evidence["residual_doppler_profiles"].shape
        != (frame_count, BROADBAND_DOPPLER_PROFILE_BIN_COUNT)
        or evidence["residual_doppler_profiles"].dtype != np.float16
        or evidence["envelope_modulation_profiles"].shape
        != (frame_count, BROADBAND_MODULATION_PROFILE_BIN_COUNT)
        or evidence["envelope_modulation_profiles"].dtype != np.float16
        or evidence["frame_prt_num"].dtype != np.int32
        or evidence["frame_prt_seconds"].dtype != np.float64
        or evidence["frame_frequency_hz"].dtype != np.float64
        or evidence["frame_cpi_count"].dtype != np.int64
    ):
        raise ValueError(f"V18 {split} evidence shape/dtype contract failed")
    finite_fields = (
        "frame_values",
        "residual_doppler_profiles",
        "envelope_modulation_profiles",
        "target_doppler_mps",
        "refined_body_doppler_mps",
        "velocity_nyquist_mps",
        "rayleigh_velocity_mps",
        "residual_sideband_snr_db",
        "envelope_modulation_snr_db",
        "frame_prt_seconds",
        "frame_frequency_hz",
    )
    if any(not np.isfinite(evidence[name]).all() for name in finite_fields):
        raise ValueError(f"V18 {split} evidence contains non-finite values")
    if (
        np.any(evidence["frame_values"] < 0.0)
        or np.any(evidence["frame_values"] > 1.0)
        or np.any(evidence["residual_doppler_profiles"] < 0.0)
        or np.any(evidence["residual_doppler_profiles"] > 1.0)
        or np.any(evidence["envelope_modulation_profiles"] < 0.0)
        or np.any(evidence["envelope_modulation_profiles"] > 1.0)
        or np.any(evidence["frame_prt_num"] <= 0)
        or np.any(evidence["frame_prt_seconds"] <= 0.0)
        or np.any(evidence["frame_frequency_hz"] <= 0.0)
        or np.any(evidence["velocity_nyquist_mps"] <= 0.0)
        or np.any(evidence["rayleigh_velocity_mps"] <= 0.0)
    ):
        raise ValueError(f"V18 {split} evidence violates physical bounds")
    if not np.array_equal(
        evidence["raw_dat_frame_indices_zero_based"] + 1,
        evidence["mat_acquisition_frame_numbers_one_based"],
    ):
        raise ValueError(f"V18 {split} MAT/raw-DAT frame identity differs")

    frame_sampling_modes = np.asarray(
        [
            broadband_sampling_mode(prt_num, pri, carrier)
            for prt_num, pri, carrier in zip(
                evidence["frame_prt_num"],
                evidence["frame_prt_seconds"],
                evidence["frame_frequency_hz"],
                strict=True,
            )
        ],
        dtype=np.int64,
    )
    candidate_mode_set = set(BROADBAND_CANDIDATE_SAMPLING_MODES)
    expected_reliable = (
        np.asarray(
            [tuple(row.tolist()) in candidate_mode_set for row in frame_sampling_modes],
            dtype=np.bool_,
        )
        & (evidence["velocity_nyquist_mps"] >= 90.0)
        & (0.5 / evidence["frame_prt_seconds"] >= BROADBAND_MIN_MODULATION_NYQUIST_HZ)
    )
    if not np.array_equal(evidence["frame_feature_reliable"], expected_reliable):
        raise ValueError(f"V18 {split} frame reliability differs from the frozen physics gate")

    reconstructed = np.zeros_like(values)
    reliable_event = np.zeros((len(selected),), dtype=np.bool_)
    prt_142_only_event = np.zeros((len(selected),), dtype=np.bool_)
    for event_index, event_valid in enumerate(valid):
        start, stop = int(offsets[event_index]), int(offsets[event_index + 1])
        if event_valid:
            reliable_event[event_index] = bool(
                evidence["frame_feature_reliable"][start:stop].any()
            )
            prt_142_only_event[event_index] = bool(
                stop > start
                and np.all(evidence["frame_prt_num"][start:stop] == 142)
            )
            reconstructed[event_index] = aggregate_broadband_frame_values(
                evidence["frame_values"][start:stop],
                evidence["frame_feature_reliable"][start:stop],
            )
    reconstruction_error = float(
        np.max(np.abs(reconstructed - values), initial=0.0)
    )
    if not np.allclose(
        reconstructed, values, atol=SUMMARY_RECONSTRUCTION_ATOL, rtol=0.0
    ):
        raise ValueError(
            f"V18 {split} summary is not reconstructible from frame evidence: "
            f"maximum_absolute_error={reconstruction_error}"
        )
    nonzero = np.any(values != 0.0, axis=1)
    if np.any(nonzero & ~reliable_event):
        raise ValueError(
            f"V18 {split} event without a reliable frame has a nonzero intervention"
        )
    prt_142_frame_mask = evidence["frame_prt_num"] == 142
    if (
        not prt_142_frame_mask.any()
        or np.any(evidence["frame_feature_reliable"][prt_142_frame_mask])
        or not prt_142_only_event.any()
        or np.any(values[prt_142_only_event] != 0.0)
    ):
        raise ValueError(f"V18 {split} PRT=142 is not strictly intervention-neutral")

    frame_event_indices = np.repeat(np.arange(len(selected), dtype=np.int64), counts)
    labels = selected["label_name"].astype(str).to_numpy()
    frame_labels = labels[frame_event_indices]
    frame_parents = selected["parent_sample_id"].astype(str).to_numpy()[
        frame_event_indices
    ]
    for event_index, row in enumerate(selected.itertuples(index=False)):
        start, stop = int(offsets[event_index]), int(offsets[event_index + 1])
        expected_legacy = np.asarray(
            json.loads(str(row.rd_raw_frame_indices_json)), dtype=np.int32
        )
        expected_tensor = np.asarray(
            json.loads(str(row.rd_tensor_frame_indices_json)), dtype=np.int32
        )
        if (
            not np.array_equal(
                evidence["legacy_mat_list_ordinals_zero_based"][start:stop],
                expected_legacy,
            )
            or not np.array_equal(
                evidence["canonical_tensor_frame_indices_zero_based"][start:stop],
                expected_tensor,
            )
        ):
            raise ValueError(
                f"V18 {split} evidence does not preserve index frame ordinals: "
                f"sample_id={row.sample_id}"
            )
    support_by_prt: dict[str, dict[str, int]] = {}
    for prt_num in sorted(set(evidence["frame_prt_num"].astype(int).tolist())):
        mask = evidence["frame_prt_num"] == prt_num
        support_by_prt[str(prt_num)] = {
            label: int(np.unique(frame_parents[mask & (frame_labels == label)]).size)
            for label in LABELS
        }
    support_by_sampling_mode: dict[str, dict[str, int]] = {}
    sampling_mode_diagnostics: dict[str, dict[str, Any]] = {}
    for mode in sorted(set(tuple(row.tolist()) for row in frame_sampling_modes)):
        mode_mask = np.all(
            frame_sampling_modes == np.asarray(mode, dtype=np.int64)[None, :],
            axis=1,
        )
        mode_name = broadband_sampling_mode_name(mode)
        support_by_sampling_mode[mode_name] = {
            label: int(
                np.unique(frame_parents[mode_mask & (frame_labels == label)]).size
            )
            for label in LABELS
        }
        sampling_mode_diagnostics[mode_name] = {
            "sampling_mode": {
                "prt_num": mode[0],
                "pri_nanoseconds": mode[1],
                "carrier_megahertz": mode[2],
            },
            "frame_count": int(mode_mask.sum()),
            "feature_reliable_frame_count": int(
                evidence["frame_feature_reliable"][mode_mask].sum()
            ),
            "parent_support_by_label": support_by_sampling_mode[mode_name],
        }

    usable = valid & reliable_event
    feature_rows: list[dict[str, Any]] = []
    for feature_id, feature_name in enumerate(BROADBAND_MICRO_DOPPLER_NAMES):
        feature = values[usable, feature_id]
        feature_rows.append(
            {
                "feature": feature_name,
                "minimum": float(feature.min()) if feature.size else None,
                "q05": float(np.quantile(feature, 0.05)) if feature.size else None,
                "median": float(np.median(feature)) if feature.size else None,
                "q95": float(np.quantile(feature, 0.95)) if feature.size else None,
                "maximum": float(feature.max()) if feature.size else None,
                "standard_deviation": float(np.std(feature)) if feature.size else 0.0,
                "unique_count_float32": int(np.unique(feature).size),
                "zero_fraction": float(np.mean(feature == 0.0)) if feature.size else 1.0,
                "one_fraction": float(np.mean(feature == 1.0)) if feature.size else 0.0,
            }
        )

    by_label: dict[str, Any] = {}
    parent_ids = selected["parent_sample_id"].astype(str).to_numpy()
    parent_rows: list[np.ndarray] = []
    parent_row_labels: list[str] = []
    parent_row_ids: list[str] = []
    for parent_id in sorted(set(parent_ids.tolist())):
        parent_mask = parent_ids == parent_id
        parent_labels = np.unique(labels[parent_mask])
        if parent_labels.size != 1:
            raise ValueError(f"V18 {split} parent-balanced feature row has invalid label")
        parent_rows.append(
            values[parent_mask].mean(axis=0, dtype=np.float64).astype(np.float32)
        )
        parent_row_labels.append(str(parent_labels[0]))
        parent_row_ids.append(parent_id)
    parent_values = np.stack(parent_rows).astype(np.float32, copy=False)
    parent_labels_array = np.asarray(parent_row_labels, dtype=np.str_)
    for label in LABELS:
        label_mask = (labels == label) & usable
        label_values = values[label_mask]
        if not label_values.size:
            raise ValueError(f"V18 {split} has no usable event for {label}")
        by_label[label] = {
            "parent_count": int(
                selected.loc[selected["label_name"].astype(str).eq(label), "parent_sample_id"].nunique()
            ),
            "usable_event_count": int(label_mask.sum()),
            "mean": label_values.mean(axis=0, dtype=np.float64).astype(float).tolist(),
            "median": np.median(label_values, axis=0).astype(float).tolist(),
            "q25": np.quantile(label_values, 0.25, axis=0).astype(float).tolist(),
            "q75": np.quantile(label_values, 0.75, axis=0).astype(float).tolist(),
        }
    drone = parent_values[parent_labels_array == "DroneTarget"]
    bird = parent_values[parent_labels_array == "BirdTarget"]
    discrimination = {
        name: {
            "direction_free_auc": _direction_free_auc(
                drone[:, feature_id], bird[:, feature_id]
            ),
            "drone_minus_bird_robust_effect_median_over_pooled_iqr": _robust_effect(
                drone[:, feature_id], bird[:, feature_id]
            ),
        }
        for feature_id, name in enumerate(BROADBAND_MICRO_DOPPLER_NAMES)
    }
    informative = [
        name
        for name, row in discrimination.items()
        if row["direction_free_auc"] >= MIN_DIRECTION_FREE_AUC
        and abs(row["drone_minus_bird_robust_effect_median_over_pooled_iqr"])
        >= MIN_ROBUST_EFFECT
    ]
    within_sampling_mode: dict[str, Any] = {}
    calibrated_mode_effects: dict[str, dict[str, float]] = {}
    for mode in BROADBAND_CANDIDATE_SAMPLING_MODES:
        mode_name = broadband_sampling_mode_name(mode)
        frame_matches_mode = np.all(
            frame_sampling_modes == np.asarray(mode, dtype=np.int64)[None, :],
            axis=1,
        )
        mode_event_values = np.zeros_like(values)
        mode_event_valid = np.zeros((len(selected),), dtype=np.bool_)
        for event_index, event_valid in enumerate(valid):
            if not event_valid:
                continue
            start, stop = int(offsets[event_index]), int(offsets[event_index + 1])
            mode_frame_mask = (
                evidence["frame_feature_reliable"][start:stop]
                & frame_matches_mode[start:stop]
            )
            if mode_frame_mask.any():
                mode_event_values[event_index] = aggregate_broadband_frame_values(
                    evidence["frame_values"][start:stop], mode_frame_mask
                )
                mode_event_valid[event_index] = True
        mode_parent_values: list[np.ndarray] = []
        mode_parent_labels: list[str] = []
        for parent_id in sorted(set(parent_ids[mode_event_valid].tolist())):
            parent_mode_mask = (parent_ids == parent_id) & mode_event_valid
            labels_for_parent = np.unique(labels[parent_mode_mask])
            if labels_for_parent.size != 1:
                raise ValueError(
                    f"V18 {split} mode={mode_name} parent-balanced row has invalid label"
                )
            mode_parent_values.append(
                mode_event_values[parent_mode_mask]
                .mean(axis=0, dtype=np.float64)
                .astype(np.float32)
            )
            mode_parent_labels.append(str(labels_for_parent[0]))
        stacked_mode_values = np.stack(mode_parent_values).astype(
            np.float32, copy=False
        )
        stacked_mode_labels = np.asarray(mode_parent_labels, dtype=np.str_)
        mode_drone = stacked_mode_values[stacked_mode_labels == "DroneTarget"]
        mode_bird = stacked_mode_values[stacked_mode_labels == "BirdTarget"]
        effects = {
            name: _robust_effect(
                mode_drone[:, feature_id], mode_bird[:, feature_id]
            )
            for feature_id, name in enumerate(BROADBAND_MICRO_DOPPLER_NAMES)
        }
        calibrated_mode_effects[mode_name] = effects
        within_sampling_mode[mode_name] = {
            "sampling_mode": {
                "prt_num": mode[0],
                "pri_nanoseconds": mode[1],
                "carrier_megahertz": mode[2],
            },
            "drone_parent_count": int(mode_drone.shape[0]),
            "bird_parent_count": int(mode_bird.shape[0]),
            "event_count": int(mode_event_valid.sum()),
            "aggregation": (
                "frame summaries restricted to this reliable pulse-count/PRI/carrier mode, then mean over "
                "eligible events within each parent"
            ),
            "drone_minus_bird_robust_effect_median_over_pooled_iqr": effects,
        }
    direction_consistent: list[str] = []
    for name in informative:
        global_effect = discrimination[name][
            "drone_minus_bird_robust_effect_median_over_pooled_iqr"
        ]
        sign = 1.0 if global_effect > 0.0 else -1.0
        if all(
            sign * calibrated_mode_effects[broadband_sampling_mode_name(mode)][name]
            >= 0.0
            for mode in BROADBAND_CANDIDATE_SAMPLING_MODES
        ) and sum(
            abs(calibrated_mode_effects[broadband_sampling_mode_name(mode)][name])
            >= MIN_ROBUST_EFFECT
            for mode in BROADBAND_CANDIDATE_SAMPLING_MODES
        ) >= 2:
            direction_consistent.append(name)
    mismatch = (
        evidence["legacy_mat_list_ordinals_zero_based"]
        != evidence["raw_dat_frame_indices_zero_based"]
    )
    mismatch_events = np.unique(frame_event_indices[mismatch])
    mismatch_tracks = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                selected.iloc[mismatch_events]["track_id"], errors="raise"
            ).tolist()
        }
    )
    return {
        "split": split,
        "parent_count": int(selected["parent_sample_id"].nunique()),
        "event_count": len(selected),
        "valid_event_count": int(valid.sum()),
        "usable_reliable_intervention_event_count": int(usable.sum()),
        "parent_discrimination_event_scope": (
            "all causal events per parent, including missing and unsupported-mode exact-zero interventions"
        ),
        "neutral_intervention_event_count": int((valid & ~nonzero).sum()),
        "prt_142_frame_count": int(prt_142_frame_mask.sum()),
        "prt_142_feature_reliable_frame_count": int(
            evidence["frame_feature_reliable"][prt_142_frame_mask].sum()
        ),
        "prt_142_only_event_count": int(prt_142_only_event.sum()),
        "prt_142_only_nonzero_intervention_event_count": int(
            (prt_142_only_event & nonzero).sum()
        ),
        "selected_frame_count": frame_count,
        "reliable_frame_count": int(expected_reliable.sum()),
        "reliable_frame_fraction": float(expected_reliable.mean()) if frame_count else 0.0,
        "parent_support_by_prt_num_and_label": support_by_prt,
        "parent_support_by_sampling_mode_and_label": support_by_sampling_mode,
        "sampling_mode_diagnostics": sampling_mode_diagnostics,
        "features": feature_rows,
        "by_label": by_label,
        "drone_bird_discrimination": discrimination,
        "drone_bird_discrimination_unit": (
            "one mean feature vector per parent over every causal event; missing and unsupported-mode events retain exact-zero intervention rows"
        ),
        "informative_feature_names": informative,
        "informative_feature_count": len(informative),
        "within_calibrated_sampling_mode_parent_balanced_effects": (
            within_sampling_mode
        ),
        "direction_consistent_informative_feature_names": direction_consistent,
        "direction_consistent_informative_feature_count": len(direction_consistent),
        "summary_reconstructed_from_frame_evidence": True,
        "summary_reconstruction_maximum_absolute_error": reconstruction_error,
        "raw_frame_mapping_audit": {
            "legacy_mat_list_ordinal_is_raw_dat_frame_index": False,
            "legacy_ordinal_raw_dat_mismatch_frame_count": int(mismatch.sum()),
            "legacy_ordinal_raw_dat_mismatch_track_ids": mismatch_tracks,
            "mat_acquisition_to_raw_dat_exact": True,
        },
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    configs = {arm: load_task_config(root / path) for arm, path in CONFIGS.items()}
    control = configs["control"]
    treatment = configs["treatment"]
    if (
        any(
            config.schema_version
            != BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
            or config.train.paired_protocol != BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL
            or config.train.paired_arm != arm
            for arm, config in configs.items()
        )
        or control.data != treatment.data
        or control.model.micro_doppler_dim != 0
        or treatment.model.micro_doppler_dim != BROADBAND_MICRO_DOPPLER_VALUE_DIM
    ):
        raise ValueError("V18 readiness loaded a non-V18 or non-paired config set")

    contract = verify_task_data_contract(root, control)
    index = pd.read_csv(root / control.data.index_path, low_memory=False)
    reports = {
        "train": _split_report(
            split=control.data.train_split_name,
            index=index,
            summary_path=root / str(control.data.train_micro_doppler_cache_path),
            evidence_path=root / str(control.data.train_micro_doppler_stft_cache_path),
        ),
        "val": _split_report(
            split=control.data.validation_split_name,
            index=index,
            summary_path=root / str(control.data.validation_micro_doppler_cache_path),
            evidence_path=root
            / str(control.data.validation_micro_doppler_stft_cache_path),
        ),
    }
    train = reports["train"]
    feature_checks = {
        row["feature"]: {
            "standard_deviation_at_least_0_0001": (
                row["standard_deviation"] >= MIN_NONDEGENERATE_STD
            ),
            "unique_count_at_least_8": (
                row["unique_count_float32"] >= MIN_NONDEGENERATE_UNIQUE
            ),
            "not_more_than_99_5_percent_at_zero": row["zero_fraction"] < 0.995,
            "not_more_than_99_5_percent_at_one": row["one_fraction"] < 0.995,
        }
        for row in train["features"]
    }
    support = train["parent_support_by_prt_num_and_label"]
    mode_support = train["parent_support_by_sampling_mode_and_label"]
    calibrated_support_passed = all(
        all(
            int(
                dict(
                    mode_support.get(broadband_sampling_mode_name(mode)) or {}
                ).get(label, 0)
            )
            >= BROADBAND_MIN_TRAIN_PARENTS_PER_CLASS_PER_MODE
            for label in LABELS
        )
        for mode in BROADBAND_CANDIDATE_SAMPLING_MODES
    )
    scope = contract["contract"]["v18_scope"]
    checks = {
        "all_8_features_non_degenerate_train_only": all(
            all(values.values()) for values in feature_checks.values()
        ),
        "all_8_train_only_drone_bird_features_have_auc_0_55_and_effect_0_10": (
            train["informative_feature_count"] == BROADBAND_MICRO_DOPPLER_VALUE_DIM
            and set(train["informative_feature_names"])
            == set(BROADBAND_MICRO_DOPPLER_NAMES)
        ),
        "all_8_parent_balanced_features_keep_direction_across_all_calibrated_sampling_modes": (
            train["direction_consistent_informative_feature_count"]
            == BROADBAND_MICRO_DOPPLER_VALUE_DIM
            and set(train["direction_consistent_informative_feature_names"])
            == set(BROADBAND_MICRO_DOPPLER_NAMES)
        ),
        "no_globally_informative_drone_bird_feature_reverses_or_lacks_mode_support": (
            set(train["informative_feature_names"])
            == set(train["direction_consistent_informative_feature_names"])
        ),
        "calibrated_sampling_modes_have_at_least_8_train_parents_per_dbb_class": (
            calibrated_support_passed
        ),
        "all_train_sampling_modes_are_frozen_as_calibrated_or_zero_intervention": (
            set(mode_support)
            == {
                broadband_sampling_mode_name(mode)
                for mode in BROADBAND_EXPECTED_TRAIN_SAMPLING_MODES
            }
        ),
        "all_uncalibrated_sampling_mode_frames_are_intervention_neutral": all(
            int(train["sampling_mode_diagnostics"][broadband_sampling_mode_name(mode)]["feature_reliable_frame_count"])
            == 0
            for mode in BROADBAND_UNCALIBRATED_SAMPLING_MODES
        ),
        "prt_142_support_is_frozen_4_drone_3_bird_0_balloon": (
            dict(support.get("142") or {})
            == {"DroneTarget": 4, "BirdTarget": 3, "BalloonTarget": 0}
        ),
        "all_prt_142_frames_and_prt_142_only_events_are_intervention_neutral": (
            train["prt_142_frame_count"] > 0
            and train["prt_142_feature_reliable_frame_count"] == 0
            and train["prt_142_only_event_count"] > 0
            and train["prt_142_only_nonzero_intervention_event_count"] == 0
        ),
        "train_summary_reconstructs_from_reliable_frame_evidence": (
            train["summary_reconstructed_from_frame_evidence"] is True
        ),
        "train_reliable_frame_fraction_is_positive": (
            train["reliable_frame_fraction"] > 0.0
        ),
        "train_raw_frame_mapping_uses_mat_acquisition_identity": (
            train["raw_frame_mapping_audit"]["mat_acquisition_to_raw_dat_exact"]
            is True
            and train["raw_frame_mapping_audit"]
            ["legacy_mat_list_ordinal_is_raw_dat_frame_index"]
            is False
        ),
        "frozen_track_1010_missing_mat_counterexample_is_preserved": (
            train["raw_frame_mapping_audit"]
            ["legacy_ordinal_raw_dat_mismatch_frame_count"]
            == 46
            and train["raw_frame_mapping_audit"]
            ["legacy_ordinal_raw_dat_mismatch_track_ids"]
            == [1010]
            and reports["val"]["raw_frame_mapping_audit"]
            ["legacy_ordinal_raw_dat_mismatch_frame_count"]
            == 0
        ),
        "sampling_metadata_and_profiles_are_not_model_inputs": (
            scope["prt_num_is_model_input"] is False
            and scope["complete_evidence_profiles_are_model_input"] is False
        ),
        "validation_is_report_only_not_readiness_gate": True,
        "test_payload_access_zero": (
            int(scope["test_raw_payload_access_count"]) == 0
            and int(scope["test_rd_payload_access_count"]) == 0
        ),
    }
    passed = all(checks.values())
    result = {
        "status": "passed" if passed else "blocked",
        "schema_version": (
            "RADAR-LN3-V18-broadband-multiscale-micro-doppler-"
            "training-readiness-v3-stable8"
        ),
        "paired_protocol": BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
        "paired_configs": {
            arm: {
                "path": str((root / path).resolve()),
                "sha256": sha256_file(root / path),
            }
            for arm, path in CONFIGS.items()
        },
        "contract_hashes": contract["hashes"],
        "checks": checks,
        "feature_checks": feature_checks,
        "train_only_thresholds": {
            "required_informative_feature_count": BROADBAND_MICRO_DOPPLER_VALUE_DIM,
            "minimum_direction_free_auc": MIN_DIRECTION_FREE_AUC,
            "minimum_absolute_robust_effect": MIN_ROBUST_EFFECT,
            "required_direction_consistent_informative_feature_count": (
                BROADBAND_MICRO_DOPPLER_VALUE_DIM
            ),
            "all_globally_informative_features_must_be_mode_consistent": True,
            "minimum_calibrated_sampling_modes_with_abs_effect_0_10": 2,
            "minimum_train_parents_per_class_per_calibrated_sampling_mode": (
                BROADBAND_MIN_TRAIN_PARENTS_PER_CLASS_PER_MODE
            ),
            "uncalibrated_sampling_mode_policy": "evidence retained; exact-zero model intervention",
        },
        "split_reports": reports,
        "interpretation": (
            "Train-only physical observability, reconstruction, support and discrimination screening passed; this does not prove a model gain."
            if passed
            else "Do not start formal paired training until failed train-only V18 checks are adjudicated."
        ),
        "validation_used_for_readiness_gate": False,
        "test": "未执行",
        "formal_training_at_report_generation": "未执行",
        "formal_training": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    result["report_path"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError(f"V18 train-only readiness failed: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit V18 train-only broadband micro-Doppler readiness."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
