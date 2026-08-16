from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build.known_gate_v21 import _build_config
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_LABEL_TO_ID,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    FEATURE_NAMES_BY_VARIANT,
    V21A_VARIANT,
    extract_canonical_known_gate_features,
    extract_spectral_temporal_consistency,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from radar_ln3.micro_doppler import RawEchoFrame, RawTargetRecord
from radar_ln3.open_set_architecture import (
    ProductAction,
    resolve_semantic_action,
)


def _payload(variant: str, baseline_path: Path | None) -> dict[str, object]:
    evidence_dim = len(FEATURE_NAMES_BY_VARIANT[variant])
    artifact = f"artifacts/smoke-v21-{variant}-seed42"
    return {
        "schema_version": "RADAR-LN3-V21",
        "config_name": f"RADAR-LN3-V21-smoke-{variant}-seed42",
        "variant": variant,
        "task": {
            "name": "known_gate",
            "class_names": list(KNOWN_GATE_CLASS_NAMES),
            "label_to_id": dict(KNOWN_GATE_LABEL_TO_ID),
            "forbidden_training_labels": ["UnknownTarget"],
            "feature_names": list(FEATURE_NAMES_BY_VARIANT[variant]),
        },
        "data": {
            "index_path": "data/processed/index.csv",
            "contract_path": "data/processed/contract.json",
            "normalization_stats_path": "data/processed/stats.json",
            "train_split_name": "train",
            "validation_split_name": "val",
            "test_split_name": "test",
            "index_schema_version": "RADAR-LN3-V16",
            "contract_schema_version": "RADAR-LN3-V21-known-gate-data-contract-v1",
            "expected_index_sha256": "0" * 64,
            "expected_contract_sha256": "1" * 64,
            "expected_stats_sha256": "2" * 64,
            "max_sequence_length": 29,
            "max_rd_frames_per_point": 12,
            "rd_amplitude_center": 55.0,
            "rd_amplitude_scale": 7.0,
            "feature_cache_manifest_path": None if variant == "baseline" else "manifest.json",
            "expected_feature_cache_manifest_sha256": None if variant == "baseline" else "3" * 64,
            "train_feature_cache_path": None if variant == "baseline" else "train.npz",
            "validation_feature_cache_path": None if variant == "baseline" else "val.npz",
            "expected_train_feature_cache_sha256": None if variant == "baseline" else "4" * 64,
            "expected_validation_feature_cache_sha256": None if variant == "baseline" else "5" * 64,
        },
        "model": {
            "tr_value_dim": 14,
            "tr_missing_dim": 11,
            "rd_quality_dim": 7,
            "hidden_dim": 48,
            "rd_embedding_dim": 24,
            "liquid_hidden_dim": 48,
            "rd_height": 21,
            "rd_width": 384,
            "liquid_tau_min_seconds": 0.05,
            "liquid_tau_max_seconds": 30.0,
            "liquid_zero_dt_event_floor_seconds": 0.001,
            "liquid_dt_max_seconds": 120.0,
            "liquid_invalid_dt_update_mode": "bounded_observation",
            "liquid_invalid_dt_observation_gain": 0.25,
            "liquid_state_normalization": "recurrent",
            "rd_frame_aggregation": "masked_mean",
            "dt_reason_mode": "v3_reason_aware_dynamics_only",
            "evidence_dim": evidence_dim,
        },
        "train": {
            "batch_size": 2,
            "learning_rate": 0.0001,
            "seed": 42,
            "dataloader_seed": 42,
            "device": "cpu",
            "epochs": 20,
            "early_stopping_patience": 0,
            "gradient_clip_norm": 100.0,
            "checkpoint_selection_metric": "parent_min_precision_recall",
            "task_min_precision_recall_target": 0.90,
            "checkpoint_path": f"{artifact}/best_model.pt",
            "last_checkpoint_path": f"{artifact}/last_model.pt",
            "formal_artifact_dir": artifact,
            "epoch_checkpoint_dir": f"{artifact}/epochs",
            "class_weights": [0.5173137460650578, 14.93939393939394],
            "loss_mode": "parent_final_task_ce_with_prefix_auxiliary",
            "prefix_auxiliary_weight": 0.1,
            "prefix_auxiliary_min_events": 3,
            "checkpoint_schema": f"RADAR-LN3-V21-{variant}-known-gate-checkpoint-v1",
            "record_parent_order_hash": True,
            "save_every_epoch": True,
            "cuda_deterministic_algorithms": False,
            "full_parameter_training": True,
            "paired_baseline_config_path": None if baseline_path is None else str(baseline_path),
        },
    }


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    batch, length, frames = 2, 3, 2
    dt_reason = torch.tensor([[1, 2, 3], [1, 2, 3]], dtype=torch.long)
    values: dict[str, torch.Tensor] = {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]),
        "tr_dt_valid": torch.tensor([[False, False, True], [False, False, True]]),
        "tr_dt_reason": dt_reason,
        "rd_frames": torch.randn(batch, length, frames, 2, 21, 384),
        "rd_frame_mask": torch.ones(batch, length, frames, dtype=torch.bool),
        "rd_quality": torch.rand(batch, length, 7),
        "tr_padding_mask": torch.zeros(batch, length, dtype=torch.bool),
    }
    if evidence_dim:
        values["known_gate_features"] = torch.rand(batch, length, evidence_dim)
        values["known_gate_feature_valid"] = torch.ones(
            batch, length, evidence_dim, dtype=torch.bool
        )
    return values


def _feature_probe() -> dict[str, object]:
    pulse_count = 128
    sample = np.arange(pulse_count, dtype=np.float64)
    iq = np.zeros((31, pulse_count), dtype=np.complex64)
    iq[15] = np.exp(1j * 2.0 * np.pi * 0.12 * sample).astype(np.complex64)
    frame = RawEchoFrame(
        raw_frame_index=0,
        frequency_hz=15.6e9,
        cpi_count=1,
        prt_num=pulse_count,
        prt_seconds=24.0e-6,
        target_records=(RawTargetRecord(1, 7, 16, 20),),
        iq=iq,
    )
    spectral, spectral_valid = extract_spectral_temporal_consistency(
        [frame], [0], track_point_index=7
    )
    velocity = np.linspace(-85.0, 85.0, 384, dtype=np.float32).astype(np.float64)
    amplitude = np.full((2, 21, 384), -80.0, dtype=np.float32)
    target_row = 10
    first_peak = int(np.argmin(np.abs(velocity - 10.0)))
    second_peak = int(np.argmin(np.abs(velocity - 20.0)))
    amplitude[0, target_row, first_peak] = 0.0
    amplitude[1, target_row, second_peak] = 0.0
    range_offset = np.linspace(-75.0, 75.0, 21, dtype=np.float64)
    canonical, canonical_valid = extract_canonical_known_gate_features(
        amplitude,
        velocity,
        range_offset,
        tr_radial_speed_mps=10.0,
        tr_range_m=100.0,
        mtd_detected_range_m=[100.0, 120.0],
    )
    if not spectral_valid.all() or not canonical_valid.all():
        raise AssertionError("V21 synthetic feature probe unexpectedly produced invalid evidence")
    if not np.all((spectral >= 0.0) & (spectral <= 1.0)) or not np.all(
        (canonical >= 0.0) & (canonical <= 1.0)
    ):
        raise AssertionError("V21 synthetic feature probe violated [0,1]")
    missing_range, missing_range_valid = extract_canonical_known_gate_features(
        amplitude,
        velocity,
        range_offset,
        tr_radial_speed_mps=10.0,
        tr_range_m=None,
        mtd_detected_range_m=[100.0, 120.0],
    )
    if (
        not np.array_equal(missing_range[:3], canonical[:3])
        or not missing_range_valid[:3].all()
        or bool(missing_range_valid[3])
        or float(missing_range[3]) != 0.0
    ):
        raise AssertionError("V21 missing TR range must invalidate only d_range evidence")
    missing_radial, missing_radial_valid = extract_canonical_known_gate_features(
        amplitude,
        velocity,
        range_offset,
        tr_radial_speed_mps=None,
        tr_range_m=100.0,
        mtd_detected_range_m=[100.0, 120.0],
    )
    if (
        np.any(missing_radial[:2] != 0.0)
        or missing_radial_valid[:2].any()
        or not missing_radial_valid[2:].all()
        or not np.array_equal(missing_radial[2:], canonical[2:])
    ):
        raise AssertionError("V21 missing TR radial speed must invalidate only modal evidence")
    return {
        "spectral": spectral.astype(float).tolist(),
        "canonical": canonical.astype(float).tolist(),
        "missing_tr_range_valid": missing_range_valid.astype(bool).tolist(),
        "missing_tr_radial_valid": missing_radial_valid.astype(bool).tolist(),
    }


def run(project_root: Path) -> dict[str, object]:
    if UNKNOWN_LABEL_IN_MAPPING := ("UnknownTarget" in KNOWN_GATE_LABEL_TO_ID):
        raise AssertionError(f"UnknownTarget unexpectedly appears in KnownGate mapping: {UNKNOWN_LABEL_IN_MAPPING}")
    actions = {
        "dbb": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=True,
            known_clutter_accepted=False,
            dbb_subclass=ProductAction.BIRD,
            dbb_subclass_accepted=True,
        ).action,
        "clutter": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=False,
            known_clutter_accepted=True,
        ).action,
        "unknown_l1": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=False,
            known_clutter_accepted=False,
        ).action,
        "unknown_l2": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=True,
            known_clutter_accepted=False,
            dbb_subclass=ProductAction.DRONE,
            dbb_subclass_accepted=False,
        ).action,
        "pending": resolve_semantic_action(
            observation_eligible=False,
            known_target_accepted=False,
            known_clutter_accepted=False,
        ).action,
        "conflict": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=True,
            known_clutter_accepted=True,
        ).action,
        "subclass_without_route": resolve_semantic_action(
            observation_eligible=True,
            known_target_accepted=False,
            known_clutter_accepted=True,
            dbb_subclass=ProductAction.DRONE,
            dbb_subclass_accepted=True,
        ).action,
    }
    if actions != {
        "dbb": ProductAction.BIRD,
        "clutter": ProductAction.CLUTTER,
        "unknown_l1": ProductAction.UNKNOWN_REJECTED,
        "unknown_l2": ProductAction.UNKNOWN_REJECTED,
        "pending": ProductAction.PENDING_ABSTAIN,
        "conflict": ProductAction.PENDING_ABSTAIN,
        "subclass_without_route": ProductAction.PENDING_ABSTAIN,
    }:
        raise AssertionError("V21 semantic action table is not mutually exclusive")
    feature_probe = _feature_probe()
    with tempfile.TemporaryDirectory(prefix="radar-ln3-v21-smoke-") as temporary:
        temp = Path(temporary)
        baseline_path = temp / "baseline.json"
        treatment_path = temp / "v21a.json"
        baseline_path.write_text(
            json.dumps(_payload("baseline", None), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        treatment_path.write_text(
            json.dumps(_payload(V21A_VARIANT, baseline_path), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        generated_support = {
            "train": {
                "DroneTarget": 476,
                "BirdTarget": 239,
                "BalloonTarget": 238,
                "ClutterTarget": 33,
                "UnknownTarget": 69,
            },
            "val": {
                "DroneTarget": 113,
                "BirdTarget": 56,
                "BalloonTarget": 54,
                "ClutterTarget": 9,
                "UnknownTarget": 15,
            },
            "test": {
                "DroneTarget": 109,
                "BirdTarget": 53,
                "BalloonTarget": 58,
                "ClutterTarget": 8,
                "UnknownTarget": 16,
            },
        }
        generated_common = {
            "source_hashes": {"index": "0" * 64},
            "contract_hash": "1" * 64,
            "stats": {"amplitude": {"center": 55.0, "scale": 7.0}},
            "stats_hash": "2" * 64,
            "manifest_hash": "3" * 64,
            "cache_hashes": {"train": "4" * 64, "val": "5" * 64},
            "support": generated_support,
        }
        for generated_variant in ("baseline", "v21a", "v21b", "v21c", "v21d"):
            generated_path = temp / f"generated-{generated_variant}.json"
            generated_path.write_text(
                json.dumps(
                    _build_config(generated_variant, **generated_common),
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            generated = load_known_gate_config(generated_path)
            if generated.variant != generated_variant or generated.train.epochs != 20:
                raise AssertionError("V21 generated config failed its loader contract")
        baseline_config = load_known_gate_config(baseline_path)
        treatment_config = load_known_gate_config(treatment_path)
        treatment, baseline, evidence = build_known_gate_model_for_config(
            project_root, treatment_config
        )
        if baseline is None or evidence["new_fusion_columns_exactly_zero"] is not True:
            raise AssertionError("V21 paired model builder did not return a zero-column Baseline pair")
        batch = _batch(treatment_config.model.evidence_dim)
        treatment.eval()
        baseline.eval()
        with torch.inference_mode():
            treatment_logits = treatment(**batch)["sequence_logits"]
            baseline_batch = dict(batch)
            baseline_batch.pop("known_gate_features")
            baseline_batch.pop("known_gate_feature_valid")
            baseline_logits = baseline(**baseline_batch)["sequence_logits"]
        if not torch.equal(treatment_logits, baseline_logits):
            raise AssertionError("V21 zero-column Treatment initial logits differ from Baseline")
        online_state = None
        online_logits: list[torch.Tensor] = []
        with torch.inference_mode():
            for event_index in range(batch["tr_features"].shape[1]):
                step = treatment.forward_online_point(
                    tr_features=batch["tr_features"][:, event_index],
                    tr_missing_mask=batch["tr_missing_mask"][:, event_index],
                    tr_dt_seconds=batch["tr_dt_seconds"][:, event_index],
                    tr_dt_valid=batch["tr_dt_valid"][:, event_index],
                    tr_dt_reason=batch["tr_dt_reason"][:, event_index],
                    rd_frames=batch["rd_frames"][:, event_index],
                    rd_frame_mask=batch["rd_frame_mask"][:, event_index],
                    rd_quality=batch["rd_quality"][:, event_index],
                    liquid_state=online_state,
                    known_gate_features=batch["known_gate_features"][:, event_index],
                    known_gate_feature_valid=batch["known_gate_feature_valid"][:, event_index],
                )
                online_state = step["liquid_state"]
                online_logits.append(step["window_logits"])
        stacked = torch.stack(online_logits, dim=1)
        if not torch.allclose(stacked, treatment_logits, atol=1.0e-6, rtol=1.0e-6):
            raise AssertionError("V21 offline/online KnownGate logits differ")
        treatment.train()
        output = treatment(**batch)
        objective = parent_final_task_ce_with_prefix_auxiliary(
            output["sequence_logits"],
            torch.tensor([0, 1], dtype=torch.long),
            batch["tr_padding_mask"],
            class_weights=torch.tensor([1.0, 1.0]),
            prefix_auxiliary_weight=0.1,
            prefix_auxiliary_min_events=3,
        )
        objective["loss"].backward()
        new_gradient = treatment.fusion[0].weight.grad[:, baseline.fusion_input_dim :]
        if new_gradient is None or not torch.isfinite(new_gradient).all() or float(new_gradient.abs().max()) <= 0.0:
            raise AssertionError("V21 experiment evidence columns did not receive a finite non-zero gradient")
        if any(not parameter.requires_grad for parameter in treatment.parameters()):
            raise AssertionError("V21 Treatment does not expose full-parameter training")

    index_path = project_root / "data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv"
    support: dict[str, int] | None = None
    if index_path.is_file():
        index = pd.read_csv(index_path, usecols=["parent_sample_id", "split_name", "label_name"])
        parents = index.drop_duplicates("parent_sample_id")
        validation = parents[parents["split_name"].astype(str).eq("val")]
        support = {
            "known_target": int(validation["label_name"].astype(str).isin(KNOWN_GATE_LABEL_TO_ID).sum())
            - int((validation["label_name"].astype(str) == "ClutterTarget").sum()),
            "clutter": int((validation["label_name"].astype(str) == "ClutterTarget").sum()),
            "unknown_excluded": int((validation["label_name"].astype(str) == "UnknownTarget").sum()),
        }
        if support != {"known_target": 223, "clutter": 9, "unknown_excluded": 15}:
            raise AssertionError(f"V21 validation support changed: {support}")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21-known-gate-contract-smoke-v1",
        "unknown_absent_from_model_mapping": True,
        "semantic_actions": {key: value.value for key, value in actions.items()},
        "feature_probe": feature_probe,
        "paired_shared_initialization": True,
        "new_fusion_columns_exactly_zero": True,
        "new_fusion_columns_receive_gradient": True,
        "all_generated_configs_load": True,
        "full_parameter_training": True,
        "offline_online_logits_match": True,
        "validation_support": support,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused V21 KnownGate contract Smoke.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
