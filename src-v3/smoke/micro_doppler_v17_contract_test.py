from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat

from radar_ln3.micro_doppler import (
    MICRO_DOPPLER_SCHEMA,
    MICRO_DOPPLER_VALUE_DIM,
    RawEchoFrame,
    RawTargetRecord,
    STFT_FRAME_CAP,
    STFT_TIME_BIN_COUNT,
    STFT_VELOCITY_BIN_COUNT,
    extract_current_event_micro_doppler,
    micro_doppler_contract,
    parse_cq08_mat_frame_identity,
    parse_cq08_raw_echo,
)
from build.micro_doppler_v17 import (
    SOURCE_CANONICAL_FRAME_INDEX,
    _CanonicalFrameProvenance,
    _build_split,
)
from radar_ln3.task_models import (
    MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    MICRO_DOPPLER_PAIRED_PROTOCOL,
    LiquidTaskClassifier,
    build_v17_paired_models,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
)
from train.tasks import verify_v17_training_readiness
from report.micro_doppler_v17_readiness import _split_report


def _representative_event(
    root: Path,
    track_id: int,
    point_index: int,
) -> dict[str, Any]:
    raw_paths = list((root / "OriData/23/CQ-08/原始回波").glob(f"{track_id}_Label_*.dat"))
    if len(raw_paths) != 1:
        raise AssertionError(f"representative track {track_id} raw path is not unique")
    frames = parse_cq08_raw_echo(raw_paths[0])
    indices = [
        frame.raw_frame_index
        for frame in frames
        if any(record.track_point_index == point_index for record in frame.target_records)
    ][:3]
    if not indices:
        raise AssertionError(f"representative track {track_id} point {point_index} has no raw frame")
    event = extract_current_event_micro_doppler(
        frames, indices, track_point_index=point_index
    )
    if (
        not event.valid
        or event.values.shape != (MICRO_DOPPLER_VALUE_DIM,)
        or event.stft_tensors.shape
        != (STFT_FRAME_CAP, STFT_VELOCITY_BIN_COUNT, STFT_TIME_BIN_COUNT)
        or int(event.frame_mask.sum()) != len(indices)
        or not np.isfinite(event.values).all()
        or np.any(event.values < 0.0)
        or np.any(event.values > 1.0)
    ):
        raise AssertionError("representative micro-Doppler event violates shape/value contract")
    manifest = json.loads(
        (root / "data/manifests/cq08_unified_manifest.json").read_text(encoding="utf-8")
    )
    record = next(
        (value for value in manifest if int(value["track_id"]) == int(track_id)),
        None,
    )
    if record is None:
        raise AssertionError(f"representative track {track_id} is absent from the CQ08 manifest")
    mat_paths = [Path(value) for value in record.get("rd_mat_files", [])]
    if len(mat_paths) < len(indices):
        raise AssertionError(f"representative track {track_id} lacks aligned MATLAB frames")
    mat_by_raw_dat_index = {
        parse_cq08_mat_frame_identity(path).raw_dat_frame_index_zero_based: path
        for path in mat_paths
    }
    matlab_rows: list[int] = []
    matlab_points: list[int] = []
    matlab_doppler_mps: list[float] = []
    for raw_index in indices:
        try:
            mat_path = mat_by_raw_dat_index[raw_index]
        except KeyError as exc:
            raise AssertionError(
                f"representative track {track_id} lacks MAT acquisition frame {raw_index + 1}"
            ) from exc
        payload = loadmat(
            mat_path,
            variable_names=["debug_info", "index_trackPointNo"],
            squeeze_me=True,
            struct_as_record=False,
        )
        debug = payload["debug_info"]
        matlab_rows.append(
            int(np.asarray(debug.detected_range_bin).reshape(-1)[0]) - 1
        )
        matlab_points.append(
            int(np.asarray(payload["index_trackPointNo"]).reshape(-1)[0])
        )
        matlab_doppler_mps.append(
            float(np.asarray(debug.detected_doppler_mps).reshape(-1)[0])
        )
    python_rows = event.target_range_indices[event.frame_mask].astype(int).tolist()
    python_doppler_mps = event.target_doppler_mps[event.frame_mask].astype(float)
    doppler_error_mps = np.abs(
        python_doppler_mps - np.asarray(matlab_doppler_mps, dtype=np.float64)
    )
    if (
        python_rows != matlab_rows
        or any(value != point_index for value in matlab_points)
        or float(doppler_error_mps.max(initial=0.0)) > 0.05
    ):
        raise AssertionError(
            "V17 target location differs from the existing MATLAB audit path: "
            f"track={track_id}, python={python_rows}, matlab={matlab_rows}, "
            f"matlab_points={matlab_points}, doppler_error_mps={doppler_error_mps.tolist()}"
        )
    return {
        "track_id": track_id,
        "track_point_index": point_index,
        "raw_frame_indices": indices,
        "summary_values": event.values.astype(float).tolist(),
        "active_stft_frame_count": int(event.frame_mask.sum()),
        "period_estimable_frame_count": int(event.frame_period_estimable.sum()),
        "target_range_indices": event.target_range_indices[event.frame_mask].astype(int).tolist(),
        "matlab_target_range_indices": matlab_rows,
        "python_matlab_target_range_indices_exactly_equal": True,
        "target_doppler_mps": python_doppler_mps.tolist(),
        "matlab_detected_doppler_mps": matlab_doppler_mps,
        "python_matlab_target_doppler_maximum_absolute_error_mps": float(
            doppler_error_mps.max(initial=0.0)
        ),
        "target_local_stft_peak_velocity_mps": event.target_local_stft_peak_velocity_mps[
            event.frame_mask
        ]
        .astype(float)
        .tolist(),
        "global_stft_peak_velocity_mps": event.global_stft_peak_velocity_mps[
            event.frame_mask
        ]
        .astype(float)
        .tolist(),
        "target_local_stft_peak_target_maximum_absolute_error_mps": float(
            np.max(
                np.abs(
                    event.target_local_stft_peak_velocity_mps[event.frame_mask]
                    - event.target_doppler_mps[event.frame_mask]
                )
            )
        ),
        "target_inside_frozen_peak_band_all_frames": bool(
            np.all(
                np.abs(
                    event.target_local_stft_peak_velocity_mps[event.frame_mask]
                    - event.target_doppler_mps[event.frame_mask]
                )
                <= 8.0
            )
        ),
    }


def _target_anchored_peak_regression() -> dict[str, Any]:
    pulse_count = 256
    target_unshifted_bin = 25
    target_shifted_bin = target_unshifted_bin + pulse_count // 2
    raw_doppler_bin_one_based = target_unshifted_bin + 1
    pulse = np.arange(pulse_count, dtype=np.float64)
    target_tone = np.exp(
        2j * np.pi * float(target_unshifted_bin) * pulse / float(pulse_count)
    )
    iq = np.zeros((31, pulse_count), dtype=np.complex64)
    # The stronger zero-Doppler component is outside the target record's local
    # MTD neighborhood but dominates an unconstrained STFT peak search.
    iq[15] = (8.0 + 1.5 * target_tone).astype(np.complex64)
    frame = RawEchoFrame(
        raw_frame_index=0,
        frequency_hz=15.6e9,
        cpi_count=1,
        prt_num=pulse_count,
        prt_seconds=48.0e-6,
        target_records=(
            RawTargetRecord(
                raw_target_identifier=1,
                track_point_index=7,
                range_bin_raw=16,
                doppler_bin_raw=raw_doppler_bin_one_based,
            ),
        ),
        iq=iq,
    )
    event = extract_current_event_micro_doppler(
        [frame], [0], track_point_index=7
    )
    target_velocity = float(event.target_doppler_mps[0])
    local_peak_velocity = float(event.target_local_stft_peak_velocity_mps[0])
    global_peak_velocity = float(event.global_stft_peak_velocity_mps[0])
    local_to_global_power_ratio = float(
        event.target_local_peak_power_ratio_to_global[0]
    )
    if int(event.target_doppler_indices[0]) != target_shifted_bin:
        raise AssertionError("synthetic local MTD target bin differs from the injected tone")
    if (
        abs(local_peak_velocity - target_velocity) > 8.0
        or abs(global_peak_velocity - target_velocity) <= 8.0
        or abs(global_peak_velocity) > 4.0
    ):
        raise AssertionError(
            "target-anchored STFT peak did not separate the target tone from stronger zero-Doppler clutter"
        )
    return {
        "target_doppler_mps": target_velocity,
        "target_local_stft_peak_velocity_mps": local_peak_velocity,
        "global_stft_peak_velocity_mps": global_peak_velocity,
        "target_local_peak_power_ratio_to_global": local_to_global_power_ratio,
        "target_local_error_mps": abs(local_peak_velocity - target_velocity),
        "global_error_mps": abs(global_peak_velocity - target_velocity),
        "global_peak_captured_zero_doppler_clutter": True,
        "target_local_peak_retained_injected_target": True,
    }


def _track_1010_frame_mapping_regression(root: Path) -> dict[str, Any]:
    provenance = _CanonicalFrameProvenance(
        root, root / SOURCE_CANONICAL_FRAME_INDEX
    )
    index = pd.read_csv(
        root / "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv",
        low_memory=False,
    )
    rows = index[
        pd.to_numeric(index["track_id"], errors="raise").eq(1010)
        & pd.to_numeric(index["track_point_index"], errors="raise").eq(3)
    ]
    if len(rows) != 1:
        raise AssertionError("Track 1010 point 3 must have exactly one V16 event")
    row = rows.iloc[0]
    legacy = [int(value) for value in json.loads(str(row["rd_raw_frame_indices_json"]))]
    tensor = [
        int(value) for value in json.loads(str(row["rd_tensor_frame_indices_json"]))
    ]
    resolved = provenance.resolve_event(
        parent_sample_id=str(row["parent_sample_id"]),
        track_id=1010,
        track_point_index=3,
        tensor_frame_indices=tensor,
        legacy_mat_list_ordinals=legacy,
        rd_tensor_path=str(row["rd_tensor_path"]),
    )
    if len(resolved) != 1:
        raise AssertionError("Track 1010 point 3 regression fixture must resolve one frame")
    value = resolved[0]
    if (
        legacy != [0]
        or value["mat_acquisition_frame_number_one_based"] != 9
        or value["raw_dat_frame_index_zero_based"] != 8
    ):
        raise AssertionError(f"Track 1010 frame remap differs from the audited fixture: {value}")
    frames = parse_cq08_raw_echo(
        root / "OriData/23/CQ-08/原始回波/1010_Label_2.dat"
    )
    mapped_index = int(value["raw_dat_frame_index_zero_based"])
    legacy_index = legacy[0]
    mapped_points = [record.track_point_index for record in frames[mapped_index].target_records]
    legacy_points = [record.track_point_index for record in frames[legacy_index].target_records]
    if mapped_points != [3] or legacy_points != [1]:
        raise AssertionError(
            "Track 1010 raw-DAT target records no longer expose the audited legacy-index counterexample"
        )
    mat = loadmat(
        Path(value["rd_mat_path"]),
        variable_names=["data_out", "index_trackPointNo"],
        squeeze_me=True,
        struct_as_record=False,
    )
    matlab_iq = np.asarray(mat["data_out"])
    mapped_error = float(np.max(np.abs(matlab_iq - frames[mapped_index].iq)))
    legacy_error = float(np.max(np.abs(matlab_iq - frames[legacy_index].iq)))
    if (
        int(np.asarray(mat["index_trackPointNo"]).reshape(-1)[0]) != 3
        or mapped_error != 0.0
        or legacy_error <= 0.0
    ):
        raise AssertionError("Track 1010 MAT/raw-DAT numeric identity check failed")
    canonical = pd.read_csv(
        root / SOURCE_CANONICAL_FRAME_INDEX,
        usecols=[
            "parent_sample_id",
            "tensor_frame_index",
            "raw_frame_index",
            "track_point_index",
            "rd_mat_path",
            "rd_tensor_path",
        ],
        low_memory=False,
    )
    canonical = canonical[
        canonical["parent_sample_id"].astype(str).eq("cq08|track|1010")
    ].sort_values("tensor_frame_index", kind="stable")
    if len(canonical) != 46:
        raise AssertionError("Track 1010 must retain exactly 46 canonical/MAT frames")
    all_frame_errors: list[float] = []
    all_point_matches = True
    all_shifts: list[int] = []
    for canonical_row in canonical.itertuples(index=False):
        identity = parse_cq08_mat_frame_identity(str(canonical_row.rd_mat_path))
        raw_index = identity.raw_dat_frame_index_zero_based
        mat_payload = loadmat(
            Path(str(canonical_row.rd_mat_path)),
            variable_names=["data_out", "index_trackPointNo"],
            squeeze_me=True,
            struct_as_record=False,
        )
        point_index = int(
            np.asarray(mat_payload["index_trackPointNo"]).reshape(-1)[0]
        )
        all_point_matches = all_point_matches and (
            point_index == int(canonical_row.track_point_index)
            and any(
                record.track_point_index == point_index
                for record in frames[raw_index].target_records
            )
        )
        all_frame_errors.append(
            float(
                np.max(
                    np.abs(
                        np.asarray(mat_payload["data_out"]) - frames[raw_index].iq
                    )
                )
            )
        )
        all_shifts.append(raw_index - int(canonical_row.raw_frame_index))
    if (
        not all_point_matches
        or max(all_frame_errors) != 0.0
        or set(all_shifts) != {8}
    ):
        raise AssertionError("Track 1010 full 46-frame MAT/raw-DAT regression failed")
    return {
        "track_id": 1010,
        "track_point_index": 3,
        "legacy_mat_list_ordinal_zero_based": legacy_index,
        "legacy_ordinal_raw_dat_track_point_index": legacy_points[0],
        "canonical_tensor_frame_index_zero_based": tensor[0],
        "mat_acquisition_frame_number_one_based": value[
            "mat_acquisition_frame_number_one_based"
        ],
        "raw_dat_frame_index_zero_based": mapped_index,
        "mapped_raw_dat_track_point_index": mapped_points[0],
        "mapped_matlab_iq_maximum_absolute_error": mapped_error,
        "legacy_ordinal_matlab_iq_maximum_absolute_error": legacy_error,
        "all_retained_frame_count": len(canonical),
        "all_retained_frames_matlab_iq_maximum_absolute_error": max(
            all_frame_errors
        ),
        "all_retained_frames_track_point_match": all_point_matches,
        "all_retained_frames_raw_dat_minus_legacy_ordinal_values": sorted(
            set(all_shifts)
        ),
        "canonical_frame_provenance_audit": provenance.audit,
        "legacy_index_semantics_counterexample_confirmed": True,
    }


def _track_1010_builder_regression(root: Path) -> dict[str, Any]:
    index = pd.read_csv(
        root / "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv",
        low_memory=False,
    )
    selected = index[
        pd.to_numeric(index["track_id"], errors="raise").eq(1010)
    ].copy()
    provenance_payload = json.loads(
        (
            root
            / "data/processed/RADAR-LN3-V16-cq08-capture-content-provenance-manifest.json"
        ).read_text(encoding="utf-8")
    )
    provenance_by_parent = {
        str(value["parent_sample_id"]): dict(value)
        for value in provenance_payload.get("records", [])
        if int(value["track_id"]) == 1010
    }
    canonical = _CanonicalFrameProvenance(
        root, root / SOURCE_CANONICAL_FRAME_INDEX
    )
    arrays, statistics = _build_split(
        root,
        "train",
        selected,
        provenance_by_parent,
        canonical,
    )
    expected_raw = np.arange(8, 54, dtype=np.int32)
    expected_legacy = np.arange(46, dtype=np.int32)
    expected_mat = np.arange(9, 55, dtype=np.int32)
    expected_tensor = np.arange(46, dtype=np.int32)
    if (
        statistics["parent_count"] != 1
        or statistics["event_count"] != 18
        or statistics["micro_doppler"]["valid_event_count"] != 16
        or statistics["selected_current_event_raw_frame_count"] != 46
        or statistics[
            "legacy_mat_list_ordinal_differs_from_raw_dat_frame_count"
        ]
        != 46
        or statistics["legacy_mat_list_ordinal_mismatch_track_ids"] != [1010]
        or not np.array_equal(
            arrays["raw_dat_frame_indices_zero_based"], expected_raw
        )
        or not np.array_equal(
            arrays["legacy_mat_list_ordinals_zero_based"], expected_legacy
        )
        or not np.array_equal(
            arrays["mat_acquisition_frame_numbers_one_based"], expected_mat
        )
        or not np.array_equal(
            arrays["canonical_tensor_frame_indices_zero_based"], expected_tensor
        )
        or int(arrays["event_offsets"][-1]) != 46
    ):
        raise AssertionError(
            f"Track 1010 focused _build_split regression failed: {statistics}"
        )
    return {
        "parent_count": statistics["parent_count"],
        "event_count": statistics["event_count"],
        "valid_event_count": statistics["micro_doppler"]["valid_event_count"],
        "invalid_event_count": statistics["micro_doppler"]["invalid_event_count"],
        "selected_stft_frame_count": int(arrays["event_offsets"][-1]),
        "raw_dat_frame_index_range_zero_based": [
            int(expected_raw[0]),
            int(expected_raw[-1]),
        ],
        "legacy_mat_list_ordinal_range_zero_based": [0, 45],
        "mat_acquisition_frame_number_range_one_based": [9, 54],
        "canonical_tensor_frame_index_range_zero_based": [0, 45],
        "all_four_frame_index_sequences_exact": True,
        "cache_write": "未执行（focused in-memory builder regression）",
    }


def _synthetic_configs(root: Path) -> tuple[Any, Any]:
    source = load_task_config(
        root / "config/RADAR-LN3/RADAR-LN3-V16-cq08-liquid-product-local-dbb-specialist-config.json"
    )
    common_data = replace(
        source.data,
        contract_schema_version=MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        rd_physical_schema="current-event-rd-physical-spectrum-v1",
        rd_physical_mask_mode="gating_only",
        rd_physical_cache_manifest_path="manifest.json",
        expected_rd_physical_cache_manifest_sha256="a" * 64,
        train_rd_physical_cache_path="train-physical.npz",
        validation_rd_physical_cache_path="val-physical.npz",
        expected_train_rd_physical_cache_sha256="b" * 64,
        expected_validation_rd_physical_cache_sha256="c" * 64,
        micro_doppler_schema=MICRO_DOPPLER_SCHEMA,
        micro_doppler_mask_mode="gating_only",
        micro_doppler_cache_manifest_path="manifest.json",
        expected_micro_doppler_cache_manifest_sha256="a" * 64,
        train_micro_doppler_cache_path="train-summary.npz",
        validation_micro_doppler_cache_path="val-summary.npz",
        expected_train_micro_doppler_cache_sha256="d" * 64,
        expected_validation_micro_doppler_cache_sha256="e" * 64,
        train_micro_doppler_stft_cache_path="train-stft.npz",
        validation_micro_doppler_stft_cache_path="val-stft.npz",
        expected_train_micro_doppler_stft_cache_sha256="f" * 64,
        expected_validation_micro_doppler_stft_cache_sha256="1" * 64,
        micro_doppler_frame_provenance_path="canonical-frame-index.csv",
        expected_micro_doppler_frame_provenance_sha256="2" * 64,
    )
    common_train = replace(
        source.train,
        paired_protocol=MICRO_DOPPLER_PAIRED_PROTOCOL,
        paired_seed=42,
        dataloader_seed=42,
        checkpoint_schema=(
            "RADAR-LN3-V17-dbb-target-anchored-micro-doppler-"
            "paired-checkpoint-v3"
        ),
        cuda_deterministic_algorithms=False,
    )
    control = replace(
        source,
        schema_version=MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        config_name="RADAR-LN3-V17-synthetic-control",
        data=common_data,
        model=replace(source.model, rd_physical_dim=4, micro_doppler_dim=0),
        train=replace(
            common_train,
            paired_arm="control",
            paired_counterpart_config_path="synthetic-treatment.json",
            checkpoint_path="control.pt",
            last_checkpoint_path="control-last.pt",
            formal_artifact_dir="control",
            epoch_checkpoint_dir="control/epochs",
        ),
    )
    treatment = replace(
        source,
        schema_version=MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        config_name="RADAR-LN3-V17-synthetic-treatment",
        data=common_data,
        model=replace(source.model, rd_physical_dim=4, micro_doppler_dim=3),
        train=replace(
            common_train,
            paired_arm="treatment",
            paired_counterpart_config_path="synthetic-control.json",
            checkpoint_path="treatment.pt",
            last_checkpoint_path="treatment-last.pt",
            formal_artifact_dir="treatment",
            epoch_checkpoint_dir="treatment/epochs",
        ),
    )
    return control, treatment


def _paired_model_contract(root: Path) -> dict[str, Any]:
    control_config, treatment_config = _synthetic_configs(root)
    control, treatment, evidence = build_v17_paired_models(
        control_config, treatment_config
    )
    batch, length = 2, 4
    torch.manual_seed(17)
    common = {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor([[0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 1.0]]),
        "tr_dt_valid": torch.tensor([[False, False, True, True], [False, True, True, True]]),
        "tr_dt_reason": torch.tensor([[1, 2, 3, 3], [1, 3, 3, 3]], dtype=torch.long),
        "rd_frames": torch.randn(batch, length, 2, 2, 21, 384),
        "rd_frame_mask": torch.ones(batch, length, 2, dtype=torch.bool),
        "rd_quality": torch.rand(batch, length, 7),
        "tr_padding_mask": torch.zeros(batch, length, dtype=torch.bool),
        "rd_physical_features": torch.rand(batch, length, 4),
        "rd_physical_valid": torch.ones(batch, length, dtype=torch.bool),
    }
    micro = torch.rand(batch, length, 3)
    micro_valid = torch.ones(batch, length, dtype=torch.bool)
    control.eval()
    treatment.eval()
    with torch.inference_mode():
        control_logits = control(**common)["sequence_logits"]
        treatment_logits = treatment(
            **common,
            micro_doppler_features=micro,
            micro_doppler_valid=micro_valid,
        )["sequence_logits"]
    if not torch.equal(control_logits, treatment_logits):
        raise AssertionError("V17 paired initial logits are not exactly equal")
    treatment.train()
    loss = treatment(
        **common,
        micro_doppler_features=micro,
        micro_doppler_valid=micro_valid,
    )["sequence_logits"].square().mean()
    loss.backward()
    gradient = treatment.fusion[0].weight.grad
    if gradient is None:
        raise AssertionError("V17 treatment Fusion gradient is missing")
    new_gradient_max = float(
        gradient[:, control.fusion_input_dim :].abs().max().item()
    )
    if new_gradient_max <= 0.0:
        raise AssertionError("V17 new micro-Doppler columns receive zero gradient")

    treatment.eval()
    offline_single = {
        key: value[:1, :1]
        for key, value in common.items()
    }
    offline_micro = micro[:1, :1]
    offline_micro_valid = micro_valid[:1, :1]
    with torch.inference_mode():
        offline_output = treatment(
            **offline_single,
            micro_doppler_features=offline_micro,
            micro_doppler_valid=offline_micro_valid,
        )
        online_output = treatment.forward_online_point(
            tr_features=offline_single["tr_features"][:, 0],
            tr_missing_mask=offline_single["tr_missing_mask"][:, 0],
            tr_dt_seconds=offline_single["tr_dt_seconds"][:, 0],
            tr_dt_valid=offline_single["tr_dt_valid"][:, 0],
            tr_dt_reason=offline_single["tr_dt_reason"][:, 0],
            rd_frames=offline_single["rd_frames"][:, 0],
            rd_frame_mask=offline_single["rd_frame_mask"][:, 0],
            rd_quality=offline_single["rd_quality"][:, 0],
            rd_physical_features=offline_single["rd_physical_features"][:, 0],
            rd_physical_valid=offline_single["rd_physical_valid"][:, 0],
            micro_doppler_features=offline_micro[:, 0],
            micro_doppler_valid=offline_micro_valid[:, 0],
        )
    offline_online_logit_error = float(
        (
            offline_output["sequence_logits"][:, 0]
            - online_output["window_logits"]
        )
        .abs()
        .max()
        .item()
    )
    if not torch.allclose(
        offline_output["sequence_logits"][:, 0],
        online_output["window_logits"],
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError(
            "V17 Treatment offline/online logits differ for the same first event: "
            f"maximum_absolute_error={offline_online_logit_error}"
        )

    training_state = {
        "python_random_state": __import__("random").getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [],
        "dataloader_generator_state": torch.Generator().get_state(),
        "run_state": {"next_epoch": 2},
    }
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v17-smoke-") as temporary:
        required_hashes = {
            "evidence_manifest_sha256",
            "train_rd_physical_cache_sha256",
            "validation_rd_physical_cache_sha256",
            "train_micro_doppler_cache_sha256",
            "validation_micro_doppler_cache_sha256",
            "train_micro_doppler_stft_cache_sha256",
            "validation_micro_doppler_stft_cache_sha256",
            "micro_doppler_frame_provenance_sha256",
        }
        for arm, arm_model, arm_config in (
            ("control", control, control_config),
            ("treatment", treatment, treatment_config),
        ):
            optimizer = torch.optim.AdamW(arm_model.parameters(), lr=1.0e-4)
            path = Path(temporary) / f"{arm}-checkpoint.pt"
            save_task_checkpoint(
                path,
                arm_model,
                optimizer,
                config=arm_config,
                epoch=1,
                global_step=3,
                metrics={"scope": "synthetic_smoke", "arm": arm},
                training_state=training_state,
            )
            restored = LiquidTaskClassifier(arm_config)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-4)
            payload = load_task_checkpoint(
                path,
                restored,
                expected_schema=arm_config.train.checkpoint_schema,
                map_location="cpu",
                optimizer=restored_optimizer,
            )
            if not required_hashes.issubset(payload["data_contract"]):
                raise AssertionError(f"V17 {arm} checkpoint lacks evidence-cache hashes")
            if payload["model_contract"]["micro_doppler_dim"] != arm_model.micro_doppler_dim:
                raise AssertionError(f"V17 {arm} checkpoint micro-Doppler contract differs")
    return {
        **evidence,
        "initial_logits_exactly_equal": True,
        "new_micro_doppler_column_gradient_max": new_gradient_max,
        "offline_online_first_event_logits_match": True,
        "offline_online_first_event_maximum_absolute_error": offline_online_logit_error,
        "control_and_treatment_checkpoint_round_trip": True,
    }


def _readiness_binding_contract() -> dict[str, Any]:
    contract_hashes = {
        "index_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "stats_sha256": "3" * 64,
    }
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v17-readiness-") as temporary:
        root = Path(temporary)
        control = root / "control.json"
        treatment = root / "treatment.json"
        control.write_text('{"arm":"control"}\n', encoding="utf-8")
        treatment.write_text('{"arm":"treatment"}\n', encoding="utf-8")
        report_path = root / (
            "src-v3/processed/report/"
            "RADAR-LN3-V17-target-anchored-v3-micro-doppler-train-readiness.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "passed",
            "schema_version": "RADAR-LN3-V17-micro-doppler-training-readiness-v4",
            "paired_configs": {
                "control": {
                    "path": str(control.resolve()),
                    "sha256": hashlib.sha256(control.read_bytes()).hexdigest(),
                },
                "treatment": {
                    "path": str(treatment.resolve()),
                    "sha256": hashlib.sha256(treatment.read_bytes()).hexdigest(),
                },
            },
            "contract_hashes": contract_hashes,
            "checks": {"synthetic_train_only_gate": True},
            "validation_used_for_readiness_gate": False,
            "test": "未执行",
            "formal_training_at_report_generation": "未执行",
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        control_binding = verify_v17_training_readiness(
            root, control, contract_hashes
        )
        treatment_binding = verify_v17_training_readiness(
            root, treatment, contract_hashes
        )
        treatment.write_text('{"arm":"mutated"}\n', encoding="utf-8")
        stale_rejected = False
        try:
            verify_v17_training_readiness(root, treatment, contract_hashes)
        except PermissionError:
            stale_rejected = True
        if not stale_rejected:
            raise AssertionError("V17 readiness binding accepted a mutated paired config")
        return {
            "both_arms_bound": True,
            "shared_readiness_sha256": (
                control_binding["sha256"] == treatment_binding["sha256"]
            ),
            "mutated_config_rejected": True,
        }


def _synthetic_readiness_cache_contract() -> dict[str, Any]:
    labels = ("DroneTarget", "BirdTarget", "BalloonTarget")
    rows = []
    sample_ids: list[str] = []
    values: list[np.ndarray] = []
    frame_raw: list[np.ndarray] = []
    period_estimable: list[bool] = []
    for label_index, label in enumerate(labels):
        for event_index in range(8):
            sample_id = f"synthetic-{label_index}-{event_index}"
            sample_ids.append(sample_id)
            rows.append(
                {
                    "sample_id": sample_id,
                    "parent_sample_id": f"parent-{label_index}-{event_index}",
                    "track_id": 10_000 + label_index * 100 + event_index,
                    "causal_event_index": 1,
                    "split_name": "train",
                    "label_name": label,
                    "rd_present": True,
                    "rd_raw_frame_indices_json": json.dumps([len(sample_ids) - 1]),
                    "rd_tensor_frame_indices_json": json.dumps(
                        [len(sample_ids) - 1]
                    ),
                }
            )
            raw = np.asarray(
                [
                    0.20 + 0.05 * label_index + 0.001 * event_index,
                    0.10 + 0.03 * label_index,
                    0.032 + 0.004 * label_index,
                ],
                dtype=np.float32,
            )
            frame_raw.append(raw)
            period_estimable.append(True)
            values.append(
                np.asarray(
                    [raw[0] / 2.0, raw[1], raw[2] / 0.1], dtype=np.float32
                )
            )
    count = len(sample_ids)
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v17-cache-smoke-") as temporary:
        root = Path(temporary)
        summary_path = root / "summary.npz"
        stft_path = root / "stft.npz"
        np.savez_compressed(
            summary_path,
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            values=np.stack(values).astype(np.float32),
            valid=np.ones((count,), dtype=np.bool_),
        )
        np.savez_compressed(
            stft_path,
            sample_ids=np.asarray(sample_ids, dtype=np.str_),
            valid=np.ones((count,), dtype=np.bool_),
            event_offsets=np.arange(count + 1, dtype=np.int64),
            stft_tensors=np.full(
                (count, STFT_VELOCITY_BIN_COUNT, STFT_TIME_BIN_COUNT),
                0.5,
                dtype=np.float16,
            ),
            raw_dat_frame_indices_zero_based=np.arange(count, dtype=np.int32),
            legacy_mat_list_ordinals_zero_based=np.arange(count, dtype=np.int32),
            canonical_tensor_frame_indices_zero_based=np.arange(
                count, dtype=np.int32
            ),
            mat_acquisition_frame_numbers_one_based=np.arange(
                1, count + 1, dtype=np.int32
            ),
            frame_raw_summaries=np.stack(frame_raw).astype(np.float32),
            frame_period_estimable=np.asarray(period_estimable, dtype=np.bool_),
            target_range_indices=np.full((count,), 15, dtype=np.int16),
            target_doppler_indices=np.full((count,), 64, dtype=np.int32),
            target_doppler_mps=np.zeros((count,), dtype=np.float32),
            target_local_stft_peak_velocity_mps=np.zeros(
                (count,), dtype=np.float32
            ),
            global_stft_peak_velocity_mps=np.zeros((count,), dtype=np.float32),
            target_local_peak_power_ratio_to_global=np.ones(
                (count,), dtype=np.float32
            ),
            native_time_bin_counts=np.full((count,), 7, dtype=np.int16),
            frame_prt_num=np.full((count,), 128, dtype=np.int32),
            frame_prt_seconds=np.full((count,), 8.0e-5, dtype=np.float64),
            frame_frequency_hz=np.full((count,), 5.6e9, dtype=np.float64),
            frame_cpi_count=np.arange(count, dtype=np.int64),
        )
        report = _split_report(
            split="train",
            index=pd.DataFrame(rows),
            summary_path=summary_path,
            stft_path=stft_path,
        )
    return {
        "parent_count": report["parent_count"],
        "event_count": report["event_count"],
        "summary_reconstruction_error": report[
            "summary_reconstruction_maximum_absolute_error"
        ],
        "target_peak_alignment_fraction": report["target_peak_alignment_audit"][
            "aligned_frame_fraction"
        ],
        "sampling_mode_count": report["sampling_mode_shortcut_audit"][
            "combined_mode_count"
        ],
        "raw_frame_mapping_audit": report["raw_frame_mapping_audit"],
        "all_cache_fields_and_dtypes_validated": True,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    representatives = {
        "DroneTarget": _representative_event(root, 1, 1),
        "BirdTarget": _representative_event(root, 333, 16),
        "BalloonTarget": _representative_event(root, 3010, 1),
    }
    track_1010_mapping = _track_1010_frame_mapping_regression(root)
    track_1010_builder = _track_1010_builder_regression(root)
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V17-micro-doppler-focused-contract-v3",
        "extractor_contract": micro_doppler_contract(),
        "representative_events": representatives,
        "target_anchored_peak_regression": _target_anchored_peak_regression(),
        "track_1010_frame_mapping_regression": track_1010_mapping,
        "track_1010_builder_regression": track_1010_builder,
        "paired_model": _paired_model_contract(root),
        "readiness_binding": _readiness_binding_contract(),
        "synthetic_readiness_cache": _synthetic_readiness_cache_contract(),
        "historical_probe_warning": (
            "The old Drone CV=0.52 versus Bird CV=0.13 probe used a different Python I/Q reshape. "
            "These representative values use the MATLAB-audited Fortran-order parser; full train-only "
            "distribution evidence remains required before formal training interpretation."
        ),
        "formal_data_build": "未执行",
        "formal_training": "未执行",
        "full_cascade": "未执行",
        "OOF": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "P95": "未执行",
    }
    output = root / (
        "src-v3/smoke/"
        "RADAR-LN3-V17-target-anchored-v3-micro-doppler-focused-contract-result.json"
    )
    result["result_path"] = str(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused V17 micro-Doppler contract Smoke.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
