from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar_ln3.micro_doppler import (
    FLICKER_CV_CLIP,
    MICRO_DOPPLER_NAMES,
    MICRO_DOPPLER_VALUE_DIM,
    MODULATION_PERIOD_CLIP_SECONDS,
    STFT_FRAME_CAP,
    STFT_TIME_BIN_COUNT,
    STFT_VELOCITY_BIN_COUNT,
)
from radar_ln3.task_models import (
    MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    MICRO_DOPPLER_PAIRED_PROTOCOL,
    load_task_config,
    sha256_file,
    verify_task_data_contract,
)


CONFIGS = {
    arm: Path(
        "config/RADAR-LN3/RADAR-LN3-V17-cq08-liquid-dbb-specialist-"
        f"rd-physical4-micro-doppler-target-anchored-v3-{arm}-seed42-config.json"
    )
    for arm in ("control", "treatment")
}
OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V17-target-anchored-v3-micro-doppler-train-readiness.json"
)
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
SAMPLING_FIELDS = (
    "frame_prt_num",
    "frame_prt_seconds",
    "frame_frequency_hz",
)
EVIDENCE_ONLY_STFT_FIELDS = (
    "sample_ids",
    "valid",
    "event_offsets",
    "stft_tensors",
    "raw_dat_frame_indices_zero_based",
    "legacy_mat_list_ordinals_zero_based",
    "canonical_tensor_frame_indices_zero_based",
    "mat_acquisition_frame_numbers_one_based",
    "frame_raw_summaries",
    "frame_period_estimable",
    "target_range_indices",
    "target_doppler_indices",
    "target_doppler_mps",
    "target_local_stft_peak_velocity_mps",
    "global_stft_peak_velocity_mps",
    "target_local_peak_power_ratio_to_global",
    "native_time_bin_counts",
    "frame_prt_num",
    "frame_prt_seconds",
    "frame_frequency_hz",
    "frame_cpi_count",
)
MAX_SAMPLING_LABEL_CRAMERS_V = 0.50
MIN_SHARED_MODE_EVENTS_PER_DRONE_BIRD_CLASS = 8
SUMMARY_RECONSTRUCTION_ATOL = 1.0e-7
MIN_TARGET_PEAK_ALIGNMENT_FRACTION = 0.90
MIN_TARGET_LOCAL_TO_GLOBAL_POWER_RATIO = 0.10


def _robust_effect(left: np.ndarray, right: np.ndarray) -> float:
    pooled = np.concatenate([left, right])
    q1, q3 = np.quantile(pooled, [0.25, 0.75])
    scale = max(float(q3 - q1), 1.0e-6)
    return float((np.median(left) - np.median(right)) / scale)


def _categorical_key(value: Any) -> str:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".12g")
    return str(int(value)) if isinstance(value, (int, np.integer)) else str(value)


def _categorical_distribution(values: np.ndarray, *, limit: int = 20) -> dict[str, Any]:
    keys, counts = np.unique(np.asarray(values), return_counts=True)
    order = np.argsort(-counts, kind="stable")
    rows = [
        {"value": _categorical_key(keys[index]), "count": int(counts[index])}
        for index in order[:limit]
    ]
    numeric = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(values)),
        "unique_count": int(len(keys)),
        "minimum": float(numeric.min()) if numeric.size else None,
        "maximum": float(numeric.max()) if numeric.size else None,
        "top_values": rows,
        "top_values_truncated": len(keys) > limit,
    }


def _cramers_v(labels: np.ndarray, categories: np.ndarray) -> float:
    label_values, label_ids = np.unique(np.asarray(labels, dtype=np.str_), return_inverse=True)
    category_values, category_ids = np.unique(
        np.asarray(categories, dtype=np.str_), return_inverse=True
    )
    if len(label_values) < 2 or len(category_values) < 2:
        return 0.0
    table = np.zeros((len(label_values), len(category_values)), dtype=np.float64)
    np.add.at(table, (label_ids, category_ids), 1.0)
    total = float(table.sum())
    row = table.sum(axis=1, keepdims=True)
    column = table.sum(axis=0, keepdims=True)
    expected = row @ column / total
    chi_square = float(np.sum(np.square(table - expected) / np.maximum(expected, 1.0e-12)))
    denominator = total * float(min(table.shape[0] - 1, table.shape[1] - 1))
    return float(np.sqrt(max(0.0, chi_square) / denominator))


def _sampling_mode(prt_num: int, prt_seconds: float, frequency_hz: float) -> str:
    return (
        f"prt_num={int(prt_num)}|prt_seconds={format(float(prt_seconds), '.12g')}|"
        f"frequency_hz={format(float(frequency_hz), '.12g')}"
    )


def _expected_native_time_bin_count(prt_num: int) -> int:
    segment_length = min(32 if int(prt_num) < 192 else 64, int(prt_num))
    hop = max(1, segment_length // 2)
    return max(1, (int(prt_num) - segment_length) // hop + 1)


def _recompute_event_summary(
    raw_summaries: np.ndarray,
    period_estimable: np.ndarray,
) -> np.ndarray:
    model = np.stack(
        [
            np.clip(raw_summaries[:, 0] / FLICKER_CV_CLIP, 0.0, 1.0),
            np.clip(raw_summaries[:, 1], 0.0, 1.0),
            np.clip(
                raw_summaries[:, 2] / MODULATION_PERIOD_CLIP_SECONDS,
                0.0,
                1.0,
            ),
        ],
        axis=1,
    )
    result = np.median(model, axis=0).astype(np.float32)
    result[2] = (
        np.float32(np.median(model[period_estimable, 2]))
        if bool(period_estimable.any())
        else np.float32(0.0)
    )
    return result


def _split_report(
    *,
    split: str,
    index: pd.DataFrame,
    summary_path: Path,
    stft_path: Path,
) -> dict[str, Any]:
    selected = index[
        index["split_name"].astype(str).eq(split)
        & index["label_name"].astype(str).isin(LABELS)
    ].copy()
    selected = selected.sort_values(
        ["parent_sample_id", "causal_event_index"], kind="stable"
    ).reset_index(drop=True)
    with np.load(summary_path, allow_pickle=False) as payload:
        sample_ids = np.asarray(payload["sample_ids"])
        values = np.asarray(payload["values"])
        valid = np.asarray(payload["valid"])
    expected_ids = selected["sample_id"].astype(str).to_numpy(dtype=np.str_)
    if (
        not np.array_equal(sample_ids, expected_ids)
        or values.shape != (len(selected), MICRO_DOPPLER_VALUE_DIM)
        or valid.shape != (len(selected),)
    ):
        raise ValueError(f"V17 {split} summary cache is not aligned to the index")
    expected_valid = selected["rd_present"].map(
        lambda value: value if isinstance(value, bool) else str(value).lower() == "true"
    ).to_numpy(dtype=np.bool_)
    if not np.array_equal(valid, expected_valid):
        raise ValueError(f"V17 {split} valid mask differs from current-event RD presence")
    with np.load(stft_path, allow_pickle=False) as payload:
        if set(payload.files) != set(EVIDENCE_ONLY_STFT_FIELDS):
            raise ValueError(
                f"V17 {split} STFT cache fields differ from the frozen evidence contract: "
                f"actual={sorted(payload.files)}"
            )
        stft = {name: np.asarray(payload[name]) for name in EVIDENCE_ONLY_STFT_FIELDS}
    offsets = stft["event_offsets"]
    stft_valid = stft["valid"]
    period_estimable = stft["frame_period_estimable"]
    stft_sample_ids = stft["sample_ids"]
    selected_frame_count = int(offsets[-1]) if offsets.size else -1
    if (
        not np.array_equal(stft_sample_ids, expected_ids)
        or not np.array_equal(stft_valid, expected_valid)
        or offsets.shape != (len(selected) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(period_estimable)
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"V17 {split} STFT cache event offsets are invalid")
    expected_flat_shapes = {
        "stft_tensors": (
            selected_frame_count,
            STFT_VELOCITY_BIN_COUNT,
            STFT_TIME_BIN_COUNT,
        ),
        "raw_dat_frame_indices_zero_based": (selected_frame_count,),
        "legacy_mat_list_ordinals_zero_based": (selected_frame_count,),
        "canonical_tensor_frame_indices_zero_based": (selected_frame_count,),
        "mat_acquisition_frame_numbers_one_based": (selected_frame_count,),
        "frame_raw_summaries": (selected_frame_count, MICRO_DOPPLER_VALUE_DIM),
        "frame_period_estimable": (selected_frame_count,),
        "target_range_indices": (selected_frame_count,),
        "target_doppler_indices": (selected_frame_count,),
        "target_doppler_mps": (selected_frame_count,),
        "target_local_stft_peak_velocity_mps": (selected_frame_count,),
        "global_stft_peak_velocity_mps": (selected_frame_count,),
        "target_local_peak_power_ratio_to_global": (selected_frame_count,),
        "native_time_bin_counts": (selected_frame_count,),
        "frame_prt_num": (selected_frame_count,),
        "frame_prt_seconds": (selected_frame_count,),
        "frame_frequency_hz": (selected_frame_count,),
        "frame_cpi_count": (selected_frame_count,),
    }
    if any(stft[name].shape != shape for name, shape in expected_flat_shapes.items()):
        actual_shapes = {name: list(stft[name].shape) for name in expected_flat_shapes}
        raise ValueError(
            f"V17 {split} STFT evidence arrays are not aligned: {actual_shapes}"
        )
    if (
        stft["stft_tensors"].dtype != np.float16
        or stft["raw_dat_frame_indices_zero_based"].dtype != np.int32
        or stft["legacy_mat_list_ordinals_zero_based"].dtype != np.int32
        or stft["canonical_tensor_frame_indices_zero_based"].dtype != np.int32
        or stft["mat_acquisition_frame_numbers_one_based"].dtype != np.int32
        or stft["frame_raw_summaries"].dtype != np.float32
        or stft["frame_period_estimable"].dtype != np.bool_
        or stft["target_range_indices"].dtype != np.int16
        or stft["target_doppler_indices"].dtype != np.int32
        or stft["target_doppler_mps"].dtype != np.float32
        or stft["target_local_stft_peak_velocity_mps"].dtype != np.float32
        or stft["global_stft_peak_velocity_mps"].dtype != np.float32
        or stft["target_local_peak_power_ratio_to_global"].dtype != np.float32
        or stft["native_time_bin_counts"].dtype != np.int16
        or stft["frame_prt_num"].dtype != np.int32
        or stft["frame_prt_seconds"].dtype != np.float64
        or stft["frame_frequency_hz"].dtype != np.float64
        or stft["frame_cpi_count"].dtype != np.int64
    ):
        raise ValueError(f"V17 {split} STFT evidence dtypes differ from the frozen contract")
    if (
        not np.isfinite(stft["stft_tensors"]).all()
        or np.any(stft["stft_tensors"] < 0.0)
        or np.any(stft["stft_tensors"] > 1.0)
        or not np.isfinite(stft["frame_raw_summaries"]).all()
        or not np.isfinite(stft["target_doppler_mps"]).all()
        or not np.isfinite(stft["target_local_stft_peak_velocity_mps"]).all()
        or not np.isfinite(stft["global_stft_peak_velocity_mps"]).all()
        or not np.isfinite(stft["target_local_peak_power_ratio_to_global"]).all()
        or np.any(stft["target_local_peak_power_ratio_to_global"] < 0.0)
        or np.any(stft["target_local_peak_power_ratio_to_global"] > 1.0)
        or np.any(stft["frame_raw_summaries"] < 0.0)
        or not np.isfinite(stft["frame_prt_seconds"]).all()
        or not np.isfinite(stft["frame_frequency_hz"]).all()
        or np.any(stft["frame_prt_num"] <= 0)
        or np.any(stft["frame_prt_seconds"] <= 0.0)
        or np.any(stft["frame_frequency_hz"] <= 0.0)
        or np.any(stft["frame_cpi_count"] < 0)
        or np.any(stft["target_range_indices"] < 0)
        or np.any(stft["target_range_indices"] >= 31)
        or np.any(stft["target_doppler_indices"] < 0)
        or np.any(stft["target_doppler_indices"] >= stft["frame_prt_num"])
        or np.any(stft["raw_dat_frame_indices_zero_based"] < 0)
        or np.any(stft["legacy_mat_list_ordinals_zero_based"] < 0)
        or np.any(stft["canonical_tensor_frame_indices_zero_based"] < 0)
        or np.any(stft["mat_acquisition_frame_numbers_one_based"] <= 0)
    ):
        raise ValueError(f"V17 {split} STFT evidence values violate physical bounds")
    if not np.array_equal(
        stft["raw_dat_frame_indices_zero_based"] + 1,
        stft["mat_acquisition_frame_numbers_one_based"],
    ):
        raise ValueError(
            f"V17 {split} MAT acquisition frame numbers do not map to raw-DAT frame indices"
        )
    mapping_mismatch = (
        stft["legacy_mat_list_ordinals_zero_based"]
        != stft["raw_dat_frame_indices_zero_based"]
    )
    counts = np.diff(offsets)
    if np.any(counts < 0) or np.any(counts > STFT_FRAME_CAP):
        raise ValueError(f"V17 {split} STFT event frame counts violate the frozen cap")
    if not np.array_equal(counts > 0, expected_valid):
        raise ValueError(f"V17 {split} STFT event frame counts differ from the valid mask")
    for event_index, row in enumerate(selected.itertuples(index=False)):
        start, stop = int(offsets[event_index]), int(offsets[event_index + 1])
        expected_legacy = np.asarray(
            [int(value) for value in json.loads(str(row.rd_raw_frame_indices_json))],
            dtype=np.int32,
        )
        expected_tensor = np.asarray(
            [
                int(value)
                for value in json.loads(str(row.rd_tensor_frame_indices_json))
            ],
            dtype=np.int32,
        )
        if (
            not np.array_equal(
                stft["legacy_mat_list_ordinals_zero_based"][start:stop],
                expected_legacy,
            )
            or not np.array_equal(
                stft["canonical_tensor_frame_indices_zero_based"][start:stop],
                expected_tensor,
            )
        ):
            raise ValueError(
                f"V17 {split} STFT evidence does not preserve index/canonical ordinals: "
                f"sample_id={row.sample_id}"
            )
    expected_native = np.asarray(
        [_expected_native_time_bin_count(value) for value in stft["frame_prt_num"]],
        dtype=np.int16,
    )
    if not np.array_equal(stft["native_time_bin_counts"], expected_native):
        raise ValueError(
            f"V17 {split} native STFT time counts differ from the frozen PRT rule"
        )
    target_peak_absolute_error_mps = np.abs(
        stft["target_local_stft_peak_velocity_mps"] - stft["target_doppler_mps"]
    )
    target_peak_aligned = target_peak_absolute_error_mps <= 8.0
    target_peak_alignment_fraction = (
        float(target_peak_aligned.mean()) if target_peak_aligned.size else 0.0
    )
    target_power_ratio = stft["target_local_peak_power_ratio_to_global"]
    target_power_sufficient = (
        target_power_ratio >= MIN_TARGET_LOCAL_TO_GLOBAL_POWER_RATIO
    )
    reconstructed = np.zeros_like(values)
    for event_index, event_is_valid in enumerate(valid):
        start, stop = int(offsets[event_index]), int(offsets[event_index + 1])
        if event_is_valid:
            reconstructed[event_index] = _recompute_event_summary(
                stft["frame_raw_summaries"][start:stop],
                stft["frame_period_estimable"][start:stop],
            )
    reconstruction_error = float(np.max(np.abs(reconstructed - values), initial=0.0))
    if not np.allclose(reconstructed, values, atol=SUMMARY_RECONSTRUCTION_ATOL, rtol=0.0):
        raise ValueError(
            f"V17 {split} summary cache is not reproducible from STFT evidence within "
            f"atol={SUMMARY_RECONSTRUCTION_ATOL}; "
            f"maximum_absolute_error={reconstruction_error}"
        )
    valid_values = values[valid]
    if valid_values.size == 0:
        raise ValueError(f"V17 {split} contains no valid micro-Doppler event")
    feature_rows = []
    for feature_id, name in enumerate(MICRO_DOPPLER_NAMES):
        feature = valid_values[:, feature_id]
        feature_rows.append(
            {
                "feature": name,
                "minimum": float(feature.min()),
                "q05": float(np.quantile(feature, 0.05)),
                "median": float(np.median(feature)),
                "q95": float(np.quantile(feature, 0.95)),
                "maximum": float(feature.max(initial=0.0)),
                "standard_deviation": float(np.std(feature)),
                "zero_fraction": float(np.mean(feature == 0.0)),
                "one_fraction": float(np.mean(feature == 1.0)),
                "unique_count_float32": int(np.unique(feature).size),
            }
        )
    by_label: dict[str, Any] = {}
    for label in LABELS:
        label_mask = selected["label_name"].astype(str).eq(label).to_numpy() & valid
        label_values = values[label_mask]
        if label_values.size == 0:
            raise ValueError(f"V17 {split} has no valid micro-Doppler event for {label}")
        label_event_indices = np.flatnonzero(label_mask)
        label_frame_chunks = [
            np.arange(int(offsets[index]), int(offsets[index + 1]), dtype=np.int64)
            for index in label_event_indices
            if int(offsets[index + 1]) > int(offsets[index])
        ]
        label_frame_indices = (
            np.concatenate(label_frame_chunks)
            if label_frame_chunks
            else np.zeros((0,), dtype=np.int64)
        )
        by_label[label] = {
            "parent_count": int(
                selected.loc[selected["label_name"].astype(str).eq(label), "parent_sample_id"].nunique()
            ),
            "valid_event_count": int(label_mask.sum()),
            "selected_stft_frame_count": int(label_frame_indices.size),
            "modulation_period_estimable_frame_count": int(
                period_estimable[label_frame_indices].sum()
            ),
            "modulation_period_estimable_frame_fraction": (
                float(period_estimable[label_frame_indices].mean())
                if label_frame_indices.size
                else 0.0
            ),
            "mean": label_values.mean(axis=0, dtype=np.float64).astype(float).tolist(),
            "median": np.median(label_values, axis=0).astype(float).tolist(),
            "q25": np.quantile(label_values, 0.25, axis=0).astype(float).tolist(),
            "q75": np.quantile(label_values, 0.75, axis=0).astype(float).tolist(),
            "sampling_metadata": {
                name: _categorical_distribution(stft[name][label_frame_indices])
                for name in (*SAMPLING_FIELDS, "frame_cpi_count")
            },
        }
    drone = values[selected["label_name"].astype(str).eq("DroneTarget").to_numpy() & valid]
    bird = values[selected["label_name"].astype(str).eq("BirdTarget").to_numpy() & valid]
    effects = {
        name: _robust_effect(drone[:, feature_id], bird[:, feature_id])
        for feature_id, name in enumerate(MICRO_DOPPLER_NAMES)
    }
    frame_event_indices = np.repeat(np.arange(len(selected), dtype=np.int64), counts)
    frame_labels = selected["label_name"].astype(str).to_numpy()[frame_event_indices]
    mismatch_event_indices = np.unique(frame_event_indices[mapping_mismatch])
    mismatch_track_ids = sorted(
        {
            int(value)
            for value in pd.to_numeric(
                selected.iloc[mismatch_event_indices]["track_id"], errors="raise"
            ).tolist()
        }
    )
    sampling_association = {
        name: {
            "cramers_v_with_three_dbb_labels": _cramers_v(
                frame_labels, stft[name].astype(str)
            ),
            "distribution": _categorical_distribution(stft[name]),
        }
        for name in SAMPLING_FIELDS
    }
    sampling_modes = np.asarray(
        [
            _sampling_mode(prt_num, prt_seconds, frequency_hz)
            for prt_num, prt_seconds, frequency_hz in zip(
                stft["frame_prt_num"],
                stft["frame_prt_seconds"],
                stft["frame_frequency_hz"],
                strict=True,
            )
        ],
        dtype=np.str_,
    )
    shared_mode_effect_rows: list[dict[str, Any]] = []
    for mode in np.unique(sampling_modes):
        mode_frame_mask = sampling_modes == mode
        mode_event_indices = np.unique(frame_event_indices[mode_frame_mask])
        mode_event_labels = selected["label_name"].astype(str).to_numpy()[mode_event_indices]
        drone_indices = mode_event_indices[mode_event_labels == "DroneTarget"]
        bird_indices = mode_event_indices[mode_event_labels == "BirdTarget"]
        if (
            drone_indices.size < MIN_SHARED_MODE_EVENTS_PER_DRONE_BIRD_CLASS
            or bird_indices.size < MIN_SHARED_MODE_EVENTS_PER_DRONE_BIRD_CLASS
        ):
            continue
        mode_effects = {
            name: _robust_effect(values[drone_indices, feature_id], values[bird_indices, feature_id])
            for feature_id, name in enumerate(MICRO_DOPPLER_NAMES)
        }
        shared_mode_effect_rows.append(
            {
                "sampling_mode": str(mode),
                "selected_frame_count": int(mode_frame_mask.sum()),
                "drone_valid_event_count": int(drone_indices.size),
                "bird_valid_event_count": int(bird_indices.size),
                "drone_minus_bird_robust_effect_median_over_pooled_iqr": mode_effects,
                "maximum_absolute_drone_bird_robust_effect": max(
                    abs(value) for value in mode_effects.values()
                ),
            }
        )
    maximum_within_mode_effect = max(
        (
            row["maximum_absolute_drone_bird_robust_effect"]
            for row in shared_mode_effect_rows
        ),
        default=0.0,
    )
    return {
        "split": split,
        "parent_count": int(selected["parent_sample_id"].nunique()),
        "event_count": int(len(selected)),
        "valid_event_count": int(valid.sum()),
        "invalid_event_count": int((~valid).sum()),
        "selected_stft_frame_count": int(offsets[-1]),
        "modulation_period_estimable_frame_count": int(period_estimable.sum()),
        "modulation_period_estimable_frame_fraction": (
            float(period_estimable.mean()) if period_estimable.size else 0.0
        ),
        "features": feature_rows,
        "by_label": by_label,
        "drone_minus_bird_robust_effect_median_over_pooled_iqr": effects,
        "maximum_absolute_drone_bird_robust_effect": max(abs(value) for value in effects.values()),
        "summary_reconstructed_from_frame_evidence_within_float32_tolerance": True,
        "summary_reconstruction_maximum_absolute_error": reconstruction_error,
        "summary_reconstruction_absolute_tolerance": SUMMARY_RECONSTRUCTION_ATOL,
        "sampling_mode_shortcut_audit": {
            "metadata_is_model_input": False,
            "cpi_count_is_association_gate_input": False,
            "association_by_field": sampling_association,
            "maximum_sampling_field_cramers_v": max(
                value["cramers_v_with_three_dbb_labels"]
                for value in sampling_association.values()
            ),
            "combined_mode_count": int(np.unique(sampling_modes).size),
            "shared_drone_bird_modes_with_minimum_support": shared_mode_effect_rows,
            "minimum_events_per_drone_bird_class_for_shared_mode": (
                MIN_SHARED_MODE_EVENTS_PER_DRONE_BIRD_CLASS
            ),
            "maximum_absolute_drone_bird_robust_effect_within_shared_mode": (
                maximum_within_mode_effect
            ),
            "frame_cpi_count_distribution": _categorical_distribution(
                stft["frame_cpi_count"]
            ),
        },
        "target_peak_alignment_audit": {
            "peak_semantics": "strongest integrated STFT peak inside the local-MTD target +/-8 m/s neighborhood",
            "peak_band_half_width_mps": 8.0,
            "aligned_frame_count": int(target_peak_aligned.sum()),
            "selected_frame_count": int(target_peak_aligned.size),
            "aligned_frame_fraction": target_peak_alignment_fraction,
            "median_absolute_error_mps": (
                float(np.median(target_peak_absolute_error_mps))
                if target_peak_absolute_error_mps.size
                else None
            ),
            "q95_absolute_error_mps": (
                float(np.quantile(target_peak_absolute_error_mps, 0.95))
                if target_peak_absolute_error_mps.size
                else None
            ),
            "maximum_absolute_error_mps": (
                float(target_peak_absolute_error_mps.max())
                if target_peak_absolute_error_mps.size
                else None
            ),
            "unconstrained_global_peak_aligned_frame_fraction": (
                float(
                    np.mean(
                        np.abs(
                            stft["global_stft_peak_velocity_mps"]
                            - stft["target_doppler_mps"]
                        )
                        <= 8.0
                    )
                )
                if target_peak_absolute_error_mps.size
                else 0.0
            ),
            "target_local_peak_power_ratio_to_global": {
                "minimum": (
                    float(target_power_ratio.min())
                    if target_power_ratio.size
                    else None
                ),
                "median": (
                    float(np.median(target_power_ratio))
                    if target_power_ratio.size
                    else None
                ),
                "q05": (
                    float(np.quantile(target_power_ratio, 0.05))
                    if target_power_ratio.size
                    else None
                ),
                "minimum_required_ratio": MIN_TARGET_LOCAL_TO_GLOBAL_POWER_RATIO,
                "minimum_required_relative_db": -10.0,
                "sufficient_frame_count": int(target_power_sufficient.sum()),
                "sufficient_frame_fraction": (
                    float(target_power_sufficient.mean())
                    if target_power_sufficient.size
                    else 0.0
                ),
            },
        },
        "raw_frame_mapping_audit": {
            "legacy_mat_list_ordinal_is_raw_dat_frame_index": False,
            "selected_frame_count": int(selected_frame_count),
            "legacy_ordinal_raw_dat_mismatch_frame_count": int(mapping_mismatch.sum()),
            "legacy_ordinal_raw_dat_mismatch_track_ids": mismatch_track_ids,
            "raw_dat_minus_legacy_ordinal_minimum": int(
                (
                    stft["raw_dat_frame_indices_zero_based"]
                    - stft["legacy_mat_list_ordinals_zero_based"]
                ).min(initial=0)
            ),
            "raw_dat_minus_legacy_ordinal_maximum": int(
                (
                    stft["raw_dat_frame_indices_zero_based"]
                    - stft["legacy_mat_list_ordinals_zero_based"]
                ).max(initial=0)
            ),
            "mat_acquisition_to_raw_dat_exact": True,
        },
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    configs = {arm: load_task_config(root / path) for arm, path in CONFIGS.items()}
    config = configs["control"]
    treatment = configs["treatment"]
    if (
        any(
            value.schema_version != MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION
            or value.train.paired_protocol != MICRO_DOPPLER_PAIRED_PROTOCOL
            or value.train.paired_arm != arm
            for arm, value in configs.items()
        )
        or config.data != treatment.data
    ):
        raise ValueError("V17 readiness loaded a non-V17 or non-paired config set")
    contract = verify_task_data_contract(root, config)
    index = pd.read_csv(root / config.data.index_path, low_memory=False)
    reports = {
        "train": _split_report(
            split=config.data.train_split_name,
            index=index,
            summary_path=root / str(config.data.train_micro_doppler_cache_path),
            stft_path=root / str(config.data.train_micro_doppler_stft_cache_path),
        ),
        "val": _split_report(
            split=config.data.validation_split_name,
            index=index,
            summary_path=root / str(config.data.validation_micro_doppler_cache_path),
            stft_path=root / str(config.data.validation_micro_doppler_stft_cache_path),
        ),
    }
    train = reports["train"]
    train_sampling = train["sampling_mode_shortcut_audit"]
    feature_checks = {
        row["feature"]: {
            "standard_deviation_at_least_0_005": row["standard_deviation"] >= 0.005,
            "unique_count_at_least_16": row["unique_count_float32"] >= 16,
            "not_more_than_99_percent_at_zero": row["zero_fraction"] < 0.99,
            "not_more_than_99_percent_at_one": row["one_fraction"] < 0.99,
        }
        for row in train["features"]
    }
    checks = {
        "all_three_features_non_degenerate_train_only": all(
            all(values.values()) for values in feature_checks.values()
        ),
        "modulation_period_estimable_for_at_least_5_percent_train_frames": (
            train["modulation_period_estimable_frame_fraction"] >= 0.05
        ),
        "at_least_one_train_only_drone_bird_robust_effect_abs_at_least_0_10": (
            train["maximum_absolute_drone_bird_robust_effect"] >= 0.10
        ),
        "train_summary_reconstructs_from_stft_frame_evidence_within_float32_tolerance": (
            train[
                "summary_reconstructed_from_frame_evidence_within_float32_tolerance"
            ]
            is True
        ),
        "train_sampling_fields_not_strongly_associated_with_dbb_label": (
            train_sampling["maximum_sampling_field_cramers_v"]
            <= MAX_SAMPLING_LABEL_CRAMERS_V
        ),
        "train_drone_bird_effect_survives_within_at_least_one_shared_sampling_mode": (
            train_sampling[
                "maximum_absolute_drone_bird_robust_effect_within_shared_mode"
            ]
            >= 0.10
        ),
        "train_stft_peak_band_contains_local_mtd_target_for_at_least_90_percent_frames": (
            train["target_peak_alignment_audit"]["aligned_frame_fraction"]
            >= MIN_TARGET_PEAK_ALIGNMENT_FRACTION
        ),
        "train_target_local_peak_is_within_10db_of_global_peak_for_at_least_90_percent_frames": (
            train["target_peak_alignment_audit"]
            ["target_local_peak_power_ratio_to_global"]
            ["sufficient_frame_fraction"]
            >= MIN_TARGET_PEAK_ALIGNMENT_FRACTION
        ),
        "train_raw_frame_mapping_uses_mat_acquisition_identity_not_legacy_ordinal": (
            train["raw_frame_mapping_audit"]["mat_acquisition_to_raw_dat_exact"]
            is True
            and train["raw_frame_mapping_audit"]
            ["legacy_mat_list_ordinal_is_raw_dat_frame_index"]
            is False
        ),
        "frozen_v16_track_1010_missing_mat_counterexample_is_preserved": (
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
        "test_payload_access_zero": (
            int(contract["contract"]["v17_scope"]["test_raw_payload_access_count"]) == 0
            and int(contract["contract"]["v17_scope"]["test_rd_payload_access_count"]) == 0
        ),
    }
    passed = all(checks.values())
    result = {
        "status": "passed" if passed else "blocked",
        "schema_version": "RADAR-LN3-V17-micro-doppler-training-readiness-v4",
        "config": str(root / CONFIGS["control"]),
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
        "sampling_shortcut_thresholds": {
            "maximum_sampling_field_cramers_v": MAX_SAMPLING_LABEL_CRAMERS_V,
            "minimum_shared_mode_events_per_drone_bird_class": (
                MIN_SHARED_MODE_EVENTS_PER_DRONE_BIRD_CLASS
            ),
            "minimum_absolute_within_mode_robust_effect": 0.10,
            "minimum_target_peak_alignment_fraction": MIN_TARGET_PEAK_ALIGNMENT_FRACTION,
            "minimum_target_local_to_global_power_ratio": (
                MIN_TARGET_LOCAL_TO_GLOBAL_POWER_RATIO
            ),
        },
        "split_reports": reports,
        "interpretation": (
            "Train-only distribution passes non-degeneracy, sampling-shortcut and within-mode "
            "effect screening; this does not prove model gain."
            if passed
            else "Do not start formal paired training until the failed train-only extractor or "
            "sampling-shortcut checks are adjudicated."
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
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["report_path"] = str(output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError(f"V17 micro-Doppler train-only readiness failed: {checks}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V17 train-only micro-Doppler feature readiness.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
