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

from build.micro_doppler_v17 import (
    SOURCE_CANONICAL_FRAME_INDEX,
    _CanonicalFrameProvenance,
)
from build.micro_doppler_v18 import (
    _build_config as build_v18_config,
    _config_path as v18_config_path,
)
from radar_ln3.data import CQ08PointSequenceDataset, collate_point_sequences
from radar_ln3.micro_doppler import RawEchoFrame, RawTargetRecord, parse_cq08_raw_echo
from radar_ln3.micro_doppler_broadband import (
    BROADBAND_CANDIDATE_SAMPLING_MODES,
    BROADBAND_EXPECTED_TRAIN_SAMPLING_MODES,
    BROADBAND_FRAME_VALUE_DIM,
    BROADBAND_MICRO_DOPPLER_NAMES,
    BROADBAND_MICRO_DOPPLER_SCHEMA,
    BROADBAND_MICRO_DOPPLER_VALUE_DIM,
    BROADBAND_MODULATION_PROFILE_BIN_COUNT,
    BROADBAND_UNCALIBRATED_SAMPLING_MODES,
    aggregate_broadband_frame_values,
    broadband_sampling_mode_name,
    BROADBAND_DOPPLER_PROFILE_BIN_COUNT,
    broadband_micro_doppler_contract,
    extract_broadband_micro_doppler_frame,
    extract_current_event_broadband_micro_doppler,
)
from report.micro_doppler_v18_readiness import _split_report
from report.micro_doppler_v18_paired import _strictly_exceeds_adoption_gain
from radar_ln3.task_models import (
    BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
    BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
    BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
    LiquidTaskClassifier,
    build_v17_paired_models,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
)
from train.tasks import build_checkpoint_selection, verify_v17_training_readiness


def _representative_event(
    root: Path, track_id: int, point_index: int
) -> dict[str, Any]:
    paths = list((root / "OriData/23/CQ-08/原始回波").glob(f"{track_id}_Label_*.dat"))
    if len(paths) != 1:
        raise AssertionError(f"V18 representative raw path is not unique: {track_id}")
    frames = parse_cq08_raw_echo(paths[0])
    indices = [
        frame.raw_frame_index
        for frame in frames
        if any(
            record.track_point_index == point_index for record in frame.target_records
        )
    ][:3]
    if not indices:
        raise AssertionError(f"V18 representative has no current-event frame: {track_id}")
    event = extract_current_event_broadband_micro_doppler(
        frames, indices, track_point_index=point_index
    )
    if (
        not event.valid
        or event.values.shape != (BROADBAND_MICRO_DOPPLER_VALUE_DIM,)
        or int(event.frame_mask.sum()) != len(indices)
        or event.frame_values.shape[-1] != BROADBAND_FRAME_VALUE_DIM
        or event.residual_doppler_profiles.shape[-1]
        != BROADBAND_DOPPLER_PROFILE_BIN_COUNT
        or event.envelope_modulation_profiles.shape[-1]
        != BROADBAND_MODULATION_PROFILE_BIN_COUNT
        or not np.isfinite(event.values).all()
        or np.any(event.values < 0.0)
        or np.any(event.values > 1.0)
    ):
        raise AssertionError("V18 representative event violates shape/value contract")
    return {
        "track_id": track_id,
        "track_point_index": point_index,
        "raw_frame_indices": indices,
        "prt_num": [int(frames[index].prt_num) for index in indices],
        "summary_values": event.values.astype(float).tolist(),
        "summary_nonzero": bool(np.any(event.values != 0.0)),
        "active_frame_count": int(event.frame_mask.sum()),
        "reliable_frame_count": int(event.frame_feature_reliable.sum()),
        "target_range_indices": event.target_range_indices[event.frame_mask]
        .astype(int)
        .tolist(),
        "velocity_nyquist_mps": event.velocity_nyquist_mps[event.frame_mask]
        .astype(float)
        .tolist(),
    }


def _synthetic_frame(
    *,
    prt_num: int,
    prt_seconds: float,
    carrier_hz: float,
    body_hz: float,
    sidebands_hz: tuple[float, ...],
    modulation_hz: float,
) -> RawEchoFrame:
    rng = np.random.default_rng(18042)
    time_seconds = np.arange(prt_num, dtype=np.float64) * prt_seconds
    signal = 8.0 * np.exp(2j * np.pi * body_hz * time_seconds)
    for number, offset_hz in enumerate(sidebands_hz, start=1):
        amplitude = 2.5 / number
        signal += amplitude * np.exp(
            2j * np.pi * (body_hz + offset_hz) * time_seconds
        )
        signal += amplitude * np.exp(
            2j * np.pi * (body_hz - offset_hz) * time_seconds
        )
    signal *= 1.0 + 0.35 * np.sin(2.0 * np.pi * modulation_hz * time_seconds)
    iq = (
        rng.normal(0.0, 0.05, size=(31, prt_num))
        + 1j * rng.normal(0.0, 0.05, size=(31, prt_num))
    )
    iq[15] += signal
    native_frequency = np.fft.fftshift(np.fft.fftfreq(prt_num, d=prt_seconds))
    target_index = int(np.argmin(np.abs(native_frequency - body_hz)))
    raw_bin = target_index + 1
    if raw_bin > prt_num // 2:
        raw_bin -= prt_num // 2
    else:
        raw_bin += prt_num // 2
    return RawEchoFrame(
        raw_frame_index=0,
        frequency_hz=carrier_hz,
        cpi_count=1,
        prt_num=prt_num,
        prt_seconds=prt_seconds,
        target_records=(
            RawTargetRecord(
                raw_target_identifier=1,
                track_point_index=1,
                range_bin_raw=16,
                doppler_bin_raw=int(raw_bin),
            ),
        ),
        iq=np.asarray(iq, dtype=np.complex64),
    )


def _synthetic_broadband_contract() -> dict[str, Any]:
    calibrated = _synthetic_frame(
        prt_num=360,
        prt_seconds=48.0e-6,
        carrier_hz=15.6e9,
        body_hz=1200.0,
        sidebands_hz=(3500.0, 7000.0, 9500.0),
        modulation_hz=900.0,
    )
    extracted = extract_broadband_micro_doppler_frame(
        calibrated, track_point_index=1
    )
    event = extract_current_event_broadband_micro_doppler(
        [calibrated], [0], track_point_index=1
    )
    if (
        not extracted.feature_reliable
        or not event.valid
        or not np.any(event.values != 0.0)
        or float(event.values[6]) <= 0.0
        or float(event.values[15:18].max(initial=0.0)) <= 0.0
    ):
        raise AssertionError("V18 synthetic broadband sideband/modulation was not retained")

    edge_crossing = _synthetic_frame(
        prt_num=360,
        prt_seconds=48.0e-6,
        carrier_hz=15.6e9,
        body_hz=9000.0,
        sidebands_hz=(3500.0,),
        modulation_hz=900.0,
    )
    edge = extract_broadband_micro_doppler_frame(
        edge_crossing, track_point_index=1
    )
    if (
        not edge.feature_reliable
        or float(edge.values[2]) <= 0.0
        or float(edge.values[3]) <= 0.0
    ):
        raise AssertionError(
            "V18 signed circular offset lost a sideband across the Doppler Nyquist edge"
        )

    zero_crossing = _synthetic_frame(
        prt_num=360,
        prt_seconds=48.0e-6,
        carrier_hz=15.6e9,
        body_hz=3500.0,
        sidebands_hz=(3500.0,),
        modulation_hz=900.0,
    )
    zero = extract_broadband_micro_doppler_frame(
        zero_crossing, track_point_index=1
    )
    if float(zero.values[2]) <= 0.0 or float(zero.values[3]) <= 0.0:
        raise AssertionError(
            "V18 relative sideband was incorrectly removed when it crossed absolute zero Doppler"
        )

    unsupported = _synthetic_frame(
        prt_num=142,
        prt_seconds=28.0e-6,
        carrier_hz=15.9e9,
        body_hz=1200.0,
        sidebands_hz=(3500.0, 7000.0),
        modulation_hz=900.0,
    )
    neutral = extract_current_event_broadband_micro_doppler(
        [unsupported], [0], track_point_index=1
    )
    if (
        not neutral.valid
        or neutral.frame_feature_reliable.any()
        or np.any(neutral.values != 0.0)
        or not np.any(neutral.frame_values[neutral.frame_mask] != 0.0)
    ):
        raise AssertionError(
            "V18 PRT=142 must retain evidence but contribute an exact-zero intervention"
        )
    return {
        "calibrated_summary_nonzero": True,
        "calibrated_total_sideband_fraction": float(event.values[6]),
        "calibrated_modulation_amplitude_maximum": float(
            event.values[15:18].max(initial=0.0)
        ),
        "nyquist_edge_negative_30_60mps_fraction": float(edge.values[2]),
        "nyquist_edge_positive_30_60mps_fraction": float(edge.values[3]),
        "nyquist_edge_crossing_sidebands_both_retained": True,
        "absolute_zero_crossing_sidebands_both_retained": True,
        "prt_142_evidence_retained": True,
        "prt_142_intervention_exactly_zero": True,
    }


def _track_1010_mapping(root: Path) -> dict[str, Any]:
    provenance = _CanonicalFrameProvenance(
        root, root / SOURCE_CANONICAL_FRAME_INDEX
    )
    records = [
        value for value in provenance.records.values() if int(value["track_id"]) == 1010
    ]
    if (
        len(records) != 46
        or any(
            int(value["raw_dat_frame_index_zero_based"])
            - int(value["legacy_mat_list_ordinal_zero_based"])
            != 8
            for value in records
        )
    ):
        raise AssertionError("V18 Track 1010 raw-DAT mapping regression failed")
    return {
        "canonical_frame_count": 46,
        "raw_dat_minus_legacy_ordinal": 8,
        "mapping_preserved": True,
    }


def _synthetic_configs(root: Path) -> tuple[Any, Any]:
    source = load_task_config(
        root
        / "config/RADAR-LN3/RADAR-LN3-V16-cq08-liquid-product-local-dbb-specialist-config.json"
    )
    common_data = replace(
        source.data,
        contract_schema_version=BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        rd_physical_schema="current-event-rd-physical-spectrum-v1",
        rd_physical_mask_mode="gating_only",
        rd_physical_cache_manifest_path="manifest.json",
        expected_rd_physical_cache_manifest_sha256="a" * 64,
        train_rd_physical_cache_path="train-physical.npz",
        validation_rd_physical_cache_path="val-physical.npz",
        expected_train_rd_physical_cache_sha256="b" * 64,
        expected_validation_rd_physical_cache_sha256="c" * 64,
        micro_doppler_schema=BROADBAND_MICRO_DOPPLER_SCHEMA,
        micro_doppler_mask_mode="gating_only",
        micro_doppler_cache_manifest_path="manifest.json",
        expected_micro_doppler_cache_manifest_sha256="a" * 64,
        train_micro_doppler_cache_path="train-summary.npz",
        validation_micro_doppler_cache_path="val-summary.npz",
        expected_train_micro_doppler_cache_sha256="d" * 64,
        expected_validation_micro_doppler_cache_sha256="e" * 64,
        train_micro_doppler_stft_cache_path="train-evidence.npz",
        validation_micro_doppler_stft_cache_path="val-evidence.npz",
        expected_train_micro_doppler_stft_cache_sha256="f" * 64,
        expected_validation_micro_doppler_stft_cache_sha256="1" * 64,
        micro_doppler_frame_provenance_path="canonical-frame-index.csv",
        expected_micro_doppler_frame_provenance_sha256="2" * 64,
    )
    common_train = replace(
        source.train,
        checkpoint_selection_metric=BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
        paired_protocol=BROADBAND_MICRO_DOPPLER_PAIRED_PROTOCOL,
        paired_seed=42,
        dataloader_seed=42,
        checkpoint_schema=(
            "RADAR-LN3-V18-dbb-broadband-multiscale-micro-doppler-"
            "paired-checkpoint-v1"
        ),
        cuda_deterministic_algorithms=False,
    )
    control = replace(
        source,
        schema_version=BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        config_name="RADAR-LN3-V18-synthetic-control",
        data=common_data,
        model=replace(source.model, rd_physical_dim=4, micro_doppler_dim=0),
        train=replace(
            common_train,
            paired_arm="control",
            paired_counterpart_config_path="treatment.json",
            checkpoint_path="control.pt",
            last_checkpoint_path="control-last.pt",
            formal_artifact_dir="control",
            epoch_checkpoint_dir="control/epochs",
        ),
    )
    treatment = replace(
        source,
        schema_version=BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        config_name="RADAR-LN3-V18-synthetic-treatment",
        data=common_data,
        model=replace(
            source.model,
            rd_physical_dim=4,
            micro_doppler_dim=BROADBAND_MICRO_DOPPLER_VALUE_DIM,
        ),
        train=replace(
            common_train,
            paired_arm="treatment",
            paired_counterpart_config_path="control.json",
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
    torch.manual_seed(18017)
    common = {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor(
            [[0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 1.0]]
        ),
        "tr_dt_valid": torch.tensor(
            [[False, False, True, True], [False, True, True, True]]
        ),
        "tr_dt_reason": torch.tensor(
            [[1, 2, 3, 3], [1, 3, 3, 3]], dtype=torch.long
        ),
        "rd_frames": torch.randn(batch, length, 2, 2, 21, 384),
        "rd_frame_mask": torch.ones(batch, length, 2, dtype=torch.bool),
        "rd_quality": torch.rand(batch, length, 7),
        "tr_padding_mask": torch.zeros(batch, length, dtype=torch.bool),
        "rd_physical_features": torch.rand(batch, length, 4),
        "rd_physical_valid": torch.ones(batch, length, dtype=torch.bool),
    }
    micro = torch.rand(batch, length, BROADBAND_MICRO_DOPPLER_VALUE_DIM)
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
        raise AssertionError("V18 paired initial logits are not exactly equal")
    treatment.train()
    loss = treatment(
        **common,
        micro_doppler_features=micro,
        micro_doppler_valid=micro_valid,
    )["sequence_logits"].square().mean()
    loss.backward()
    gradient = treatment.fusion[0].weight.grad
    if gradient is None:
        raise AssertionError("V18 Treatment Fusion gradient is missing")
    gradient_max = float(
        gradient[:, control.fusion_input_dim :].abs().max().item()
    )
    if gradient_max <= 0.0:
        raise AssertionError("V18 new 39 Fusion columns receive zero gradient")

    treatment.eval()
    offline_common = {key: value[:1, :1] for key, value in common.items()}
    with torch.inference_mode():
        offline = treatment(
            **offline_common,
            micro_doppler_features=micro[:1, :1],
            micro_doppler_valid=micro_valid[:1, :1],
        )["sequence_logits"][:, 0]
        online = treatment.forward_online_point(
            tr_features=offline_common["tr_features"][:, 0],
            tr_missing_mask=offline_common["tr_missing_mask"][:, 0],
            tr_dt_seconds=offline_common["tr_dt_seconds"][:, 0],
            tr_dt_valid=offline_common["tr_dt_valid"][:, 0],
            tr_dt_reason=offline_common["tr_dt_reason"][:, 0],
            rd_frames=offline_common["rd_frames"][:, 0],
            rd_frame_mask=offline_common["rd_frame_mask"][:, 0],
            rd_quality=offline_common["rd_quality"][:, 0],
            rd_physical_features=offline_common["rd_physical_features"][:, 0],
            rd_physical_valid=offline_common["rd_physical_valid"][:, 0],
            micro_doppler_features=micro[:1, 0],
            micro_doppler_valid=micro_valid[:1, 0],
        )["window_logits"]
    error = float((offline - online).abs().max().item())
    if not torch.allclose(offline, online, atol=1.0e-6, rtol=1.0e-6):
        raise AssertionError(f"V18 offline/online logits differ: {error}")

    training_state = {
        "python_random_state": __import__("random").getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_states": [],
        "dataloader_generator_state": torch.Generator().get_state(),
        "run_state": {"next_epoch": 2},
    }
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v18-smoke-") as temporary:
        for arm, model, config in (
            ("control", control, control_config),
            ("treatment", treatment, treatment_config),
        ):
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
            path = Path(temporary) / f"{arm}.pt"
            save_task_checkpoint(
                path,
                model,
                optimizer,
                config=config,
                epoch=1,
                global_step=3,
                metrics={"scope": "V18 synthetic smoke"},
                training_state=training_state,
            )
            restored = LiquidTaskClassifier(config)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1.0e-4)
            payload = load_task_checkpoint(
                path,
                restored,
                expected_schema=config.train.checkpoint_schema,
                map_location="cpu",
                optimizer=restored_optimizer,
            )
            expected_names = (
                list(BROADBAND_MICRO_DOPPLER_NAMES) if arm == "treatment" else []
            )
            if payload["model_contract"]["micro_doppler_names"] != expected_names:
                raise AssertionError(f"V18 {arm} checkpoint feature names differ")
    return {
        **evidence,
        "initial_logits_exactly_equal": True,
        "new_micro_doppler_column_gradient_max": gradient_max,
        "offline_online_first_event_logits_match": True,
        "offline_online_maximum_absolute_error": error,
        "checkpoint_round_trip": True,
    }


def _cache_dataset_contract(root: Path) -> dict[str, Any]:
    index = root / "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
    frame = __import__("pandas").read_csv(index, low_memory=False)
    selected = frame[
        frame["split_name"].astype(str).eq("train")
        & frame["label_name"].astype(str).eq("DroneTarget")
    ].sort_values(["parent_sample_id", "causal_event_index"], kind="stable")
    parent_id = str(selected.iloc[0]["parent_sample_id"])
    parent = selected[selected["parent_sample_id"].astype(str).eq(parent_id)]
    sample_ids = parent["sample_id"].astype(str).to_numpy(dtype=np.str_)
    values = np.zeros(
        (len(sample_ids), BROADBAND_MICRO_DOPPLER_VALUE_DIM), dtype=np.float32
    )
    values[:, 0] = np.linspace(0.1, 0.2, len(sample_ids), dtype=np.float32)
    valid = np.ones((len(sample_ids),), dtype=np.bool_)
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v18-dataset-") as temporary:
        cache = Path(temporary) / "summary.npz"
        np.savez_compressed(cache, sample_ids=sample_ids, values=values, valid=valid)
        dataset = CQ08PointSequenceDataset(
            root,
            index_path=index,
            split_name="train",
            max_sequence_length=29,
            max_rd_frames_per_point=12,
            rd_amplitude_center=55.90625,
            rd_amplitude_scale=6.911083992973033,
            expected_schema_version="RADAR-LN3-V16",
            micro_doppler_cache_path=cache,
            micro_doppler_value_dim=BROADBAND_MICRO_DOPPLER_VALUE_DIM,
            allowed_labels={"DroneTarget"},
            allowed_parent_ids={parent_id},
        )
        item = dataset[0]
        batch = collate_point_sequences([item])
        if (
            batch["micro_doppler_features"].shape[-1]
            != BROADBAND_MICRO_DOPPLER_VALUE_DIM
            or not torch.equal(
                batch["micro_doppler_features"][0, : len(sample_ids)],
                torch.from_numpy(values),
            )
        ):
            raise AssertionError("V18 cache to Dataset/Collate path changed values")
    return {
        "parent_sample_id": parent_id,
        "event_count": len(sample_ids),
        "feature_dim": BROADBAND_MICRO_DOPPLER_VALUE_DIM,
        "cache_dataset_collate_exact": True,
    }


def _checkpoint_selection_contract() -> dict[str, Any]:
    validation = {
        "parent_metrics": {
            "macro_f1": 0.95,
            "per_class": [
                {
                    "class": "DroneTarget",
                    "precision": 0.96,
                    "recall": 0.95,
                    "f1": 0.955,
                    "support": 10,
                },
                {
                    "class": "BirdTarget",
                    "precision": 0.95,
                    "recall": 0.94,
                    "f1": 0.945,
                    "support": 10,
                },
                {
                    "class": "BalloonTarget",
                    "precision": 0.97,
                    "recall": 0.96,
                    "f1": 0.965,
                    "support": 10,
                },
            ],
            "confusion_matrix_rows_true_columns_pred": [
                [9, 1, 0],
                [1, 9, 0],
                [0, 0, 10],
            ],
        }
    }
    selection = build_checkpoint_selection(
        validation, BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION
    )
    if selection["ranking_key"] != [0.94, -2.0, 0.95]:
        raise AssertionError("V18 checkpoint hierarchy differs from the frozen rule")
    if _strictly_exceeds_adoption_gain(0.02) or not _strictly_exceeds_adoption_gain(
        0.02000000001
    ):
        raise AssertionError("V18 adoption gain must be strictly greater than 0.02")
    return {
        "metric": selection["metric"],
        "ranking_key": selection["ranking_key"],
        "acceptance_threshold_not_part_of_epoch_ranking": True,
        "exactly_0_02_gain_is_rejected": True,
        "gain_above_0_02_is_accepted": True,
    }


def _generated_config_load_contract(root: Path) -> dict[str, Any]:
    source_path = (
        root
        / "config/RADAR-LN3/RADAR-LN3-V16-cq08-liquid-product-local-dbb-specialist-config.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source_hashes = {
        "index": "1" * 64,
        "stats": "2" * 64,
        "canonical_frame_index": "3" * 64,
    }
    cache_hashes = {
        split: {"physical": "4" * 64, "summary": "5" * 64, "evidence": "6" * 64}
        for split in ("train", "val")
    }
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v18-config-") as temporary:
        temporary_root = Path(temporary)
        loaded = {}
        for arm in ("control", "treatment"):
            payload = build_v18_config(
                source,
                arm=arm,
                source_hashes=source_hashes,
                contract_hash="7" * 64,
                manifest_hash="8" * 64,
                cache_hashes=cache_hashes,
            )
            path = temporary_root / v18_config_path(arm).name
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            loaded[arm] = load_task_config(path)
    if (
        loaded["control"].model.micro_doppler_dim != 0
        or loaded["treatment"].model.micro_doppler_dim
        != BROADBAND_MICRO_DOPPLER_VALUE_DIM
        or loaded["control"].train.checkpoint_selection_metric
        != BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION
        or loaded["treatment"].train.checkpoint_selection_metric
        != BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION
    ):
        raise AssertionError("V18 generated JSON configs do not load under the frozen contract")
    return {
        "control_micro_doppler_dim": 0,
        "treatment_micro_doppler_dim": BROADBAND_MICRO_DOPPLER_VALUE_DIM,
        "checkpoint_selection": BROADBAND_MICRO_DOPPLER_CHECKPOINT_SELECTION,
        "both_generated_json_configs_load": True,
    }


def _readiness_binding_contract() -> dict[str, Any]:
    hashes = {
        "index_sha256": "1" * 64,
        "contract_sha256": "2" * 64,
        "stats_sha256": "3" * 64,
    }
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v18-readiness-") as temporary:
        root = Path(temporary)
        control = root / "control.json"
        treatment = root / "treatment.json"
        control.write_text('{"arm":"control"}\n', encoding="utf-8")
        treatment.write_text('{"arm":"treatment"}\n', encoding="utf-8")
        report_path = root / (
            "src-v3/processed/report/"
            "RADAR-LN3-V18-broadband-multiscale-micro-doppler-train-readiness.json"
        )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "status": "passed",
            "schema_version": (
                "RADAR-LN3-V18-broadband-multiscale-micro-doppler-"
                "training-readiness-v1"
            ),
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
            "contract_hashes": hashes,
            "checks": {"synthetic_train_only_gate": True},
            "validation_used_for_readiness_gate": False,
            "test": "未执行",
            "formal_training_at_report_generation": "未执行",
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        first = verify_v17_training_readiness(
            root,
            control,
            hashes,
            schema_version=BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        )
        second = verify_v17_training_readiness(
            root,
            treatment,
            hashes,
            schema_version=BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
        )
        treatment.write_text('{"arm":"mutated"}\n', encoding="utf-8")
        rejected = False
        try:
            verify_v17_training_readiness(
                root,
                treatment,
                hashes,
                schema_version=BROADBAND_MICRO_DOPPLER_PAIRED_DBB_SCHEMA_VERSION,
            )
        except PermissionError:
            rejected = True
        if not rejected:
            raise AssertionError("V18 readiness accepted a mutated paired config")
    return {
        "both_arms_bound": True,
        "shared_readiness_sha256": first["sha256"] == second["sha256"],
        "mutated_config_rejected": True,
    }


def _synthetic_readiness_aggregation_contract() -> dict[str, Any]:
    labels = ("DroneTarget", "BirdTarget", "BalloonTarget")
    rows: list[dict[str, Any]] = []
    frame_values: list[np.ndarray] = []
    reliable: list[bool] = []
    prt_num: list[int] = []
    prt_seconds: list[float] = []
    frequency_hz: list[float] = []
    summary_values: list[np.ndarray] = []
    sample_ids: list[str] = []
    mode_sequence: list[tuple[int, int, int]] = []
    event_number = 0
    for mode in BROADBAND_EXPECTED_TRAIN_SAMPLING_MODES:
        mode_labels = labels if mode in BROADBAND_CANDIDATE_SAMPLING_MODES else ("DroneTarget",)
        for label in mode_labels:
            sample_id = f"synthetic-event-{event_number:03d}"
            parent_id = f"synthetic-parent-{event_number:03d}"
            base = {
                "DroneTarget": 0.30,
                "BirdTarget": 0.10,
                "BalloonTarget": 0.20,
            }[label]
            evidence_row = np.full(
                (BROADBAND_FRAME_VALUE_DIM,), base, dtype=np.float32
            )
            is_reliable = mode in BROADBAND_CANDIDATE_SAMPLING_MODES
            model_row = aggregate_broadband_frame_values(
                evidence_row[None, :], np.asarray([is_reliable], dtype=np.bool_)
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "parent_sample_id": parent_id,
                    "causal_event_index": 0,
                    "split_name": "train",
                    "label_name": label,
                    "rd_present": True,
                    "rd_raw_frame_indices_json": "[0]",
                    "rd_tensor_frame_indices_json": "[0]",
                    "track_id": event_number + 1,
                }
            )
            sample_ids.append(sample_id)
            frame_values.append(evidence_row)
            reliable.append(is_reliable)
            prt_num.append(mode[0])
            prt_seconds.append(mode[1] * 1.0e-9)
            frequency_hz.append(mode[2] * 1.0e6)
            summary_values.append(model_row)
            mode_sequence.append(mode)
            event_number += 1

    frame_count = len(rows)
    identifiers = np.asarray(sample_ids, dtype=np.str_)
    values = np.stack(summary_values).astype(np.float32, copy=False)
    valid = np.ones((frame_count,), dtype=np.bool_)
    offsets = np.arange(frame_count + 1, dtype=np.int64)
    frames = np.stack(frame_values).astype(np.float32, copy=False)
    reliability = np.asarray(reliable, dtype=np.bool_)
    raw_indices = np.zeros((frame_count,), dtype=np.int32)
    evidence = {
        "sample_ids": identifiers,
        "valid": valid,
        "event_offsets": offsets,
        "raw_dat_frame_indices_zero_based": raw_indices,
        "legacy_mat_list_ordinals_zero_based": raw_indices.copy(),
        "canonical_tensor_frame_indices_zero_based": raw_indices.copy(),
        "mat_acquisition_frame_numbers_one_based": np.ones(
            (frame_count,), dtype=np.int32
        ),
        "frame_values": frames,
        "frame_feature_reliable": reliability,
        "residual_doppler_profiles": np.zeros(
            (frame_count, BROADBAND_DOPPLER_PROFILE_BIN_COUNT), dtype=np.float16
        ),
        "envelope_modulation_profiles": np.zeros(
            (frame_count, BROADBAND_MODULATION_PROFILE_BIN_COUNT), dtype=np.float16
        ),
        "target_range_indices": np.full((frame_count,), 15, dtype=np.int16),
        "target_doppler_indices": np.zeros((frame_count,), dtype=np.int32),
        "target_doppler_mps": np.zeros((frame_count,), dtype=np.float32),
        "refined_body_doppler_mps": np.zeros((frame_count,), dtype=np.float32),
        "velocity_nyquist_mps": np.full((frame_count,), 100.0, dtype=np.float32),
        "rayleigh_velocity_mps": np.full((frame_count,), 2.0, dtype=np.float32),
        "residual_sideband_snr_db": np.full((frame_count,), 10.0, dtype=np.float32),
        "envelope_modulation_snr_db": np.full((frame_count,), 10.0, dtype=np.float32),
        "frame_prt_num": np.asarray(prt_num, dtype=np.int32),
        "frame_prt_seconds": np.asarray(prt_seconds, dtype=np.float64),
        "frame_frequency_hz": np.asarray(frequency_hz, dtype=np.float64),
        "frame_cpi_count": np.ones((frame_count,), dtype=np.int64),
    }
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v18-readiness-cache-") as temporary:
        temporary_root = Path(temporary)
        summary_path = temporary_root / "summary.npz"
        evidence_path = temporary_root / "evidence.npz"
        np.savez(
            summary_path,
            sample_ids=identifiers,
            values=values,
            valid=valid,
        )
        np.savez(evidence_path, **evidence)
        report = _split_report(
            split="train",
            index=pd.DataFrame(rows),
            summary_path=summary_path,
            evidence_path=evidence_path,
        )
    expected_mode_names = {
        broadband_sampling_mode_name(mode)
        for mode in BROADBAND_EXPECTED_TRAIN_SAMPLING_MODES
    }
    if (
        set(report["sampling_mode_diagnostics"]) != expected_mode_names
        or len(report["within_calibrated_sampling_mode_parent_balanced_effects"])
        != len(BROADBAND_CANDIDATE_SAMPLING_MODES)
        or report["prt_142_feature_reliable_frame_count"] != 0
        or report["prt_142_only_nonzero_intervention_event_count"] != 0
        or any(
            report["sampling_mode_diagnostics"][broadband_sampling_mode_name(mode)]
            ["feature_reliable_frame_count"]
            != 0
            for mode in BROADBAND_UNCALIBRATED_SAMPLING_MODES
        )
        or report["parent_discrimination_event_scope"]
        != "all causal events per parent, including missing and unsupported-mode exact-zero interventions"
    ):
        raise AssertionError("V18 synthetic readiness aggregation contract failed")
    return {
        "observed_sampling_mode_count": len(report["sampling_mode_diagnostics"]),
        "calibrated_sampling_mode_count": len(
            report["within_calibrated_sampling_mode_parent_balanced_effects"]
        ),
        "uncalibrated_sampling_mode_count": len(
            BROADBAND_UNCALIBRATED_SAMPLING_MODES
        ),
        "uncalibrated_modes_exact_zero": True,
        "parent_discrimination_includes_zero_interventions": True,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    result = {
        "status": "passed",
        "schema_version": (
            "RADAR-LN3-V18-broadband-multiscale-micro-doppler-"
            "focused-contract-v1"
        ),
        "extractor_contract": broadband_micro_doppler_contract(),
        "representative_events": {
            "DroneTarget": _representative_event(root, 1, 1),
            "BirdTarget": _representative_event(root, 333, 16),
            "BalloonTarget": _representative_event(root, 3010, 1),
        },
        "synthetic_broadband": _synthetic_broadband_contract(),
        "track_1010_frame_mapping": _track_1010_mapping(root),
        "cache_dataset_contract": _cache_dataset_contract(root),
        "paired_model": _paired_model_contract(root),
        "generated_config_load": _generated_config_load_contract(root),
        "checkpoint_selection": _checkpoint_selection_contract(),
        "readiness_binding": _readiness_binding_contract(),
        "synthetic_readiness_aggregation": (
            _synthetic_readiness_aggregation_contract()
        ),
        "v17_regression_scope": (
            "V17 modules/configs are loaded by the separate retained V17 smoke; "
            "V18 uses a distinct schema and does not overwrite V17 artifacts"
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
        "RADAR-LN3-V18-broadband-multiscale-micro-doppler-focused-contract-result.json"
    )
    result["result_path"] = str(output)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run focused V18 broadband micro-Doppler contract Smoke."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
