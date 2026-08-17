from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v21c import (
    BASELINE_CONFIG,
    CONFIG,
    OLD_SEED42_CONFIG,
)
from radar_ln3.known_gate import (
    build_known_gate_model_for_config,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    ISO_FEATURE_NAMES,
    KNOWN_GATE_FEATURE_DIM,
    V21C_VARIANT,
    extract_canonical_known_gate_features,
    load_known_gate_feature_cache,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _verify_contract, _verify_readiness


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(2103)
    batch, length, frames = 2, 4, 3
    return {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor(
            [[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 2.0, 1.0]],
            dtype=torch.float32,
        ),
        "tr_dt_valid": torch.tensor(
            [[False, False, True, True], [False, False, True, True]]
        ),
        "tr_dt_reason": torch.tensor(
            [[1, 2, 3, 3], [1, 2, 3, 3]], dtype=torch.long
        ),
        "rd_frames": torch.randn(batch, length, frames, 2, 21, 384),
        "rd_frame_mask": torch.ones(batch, length, frames, dtype=torch.bool),
        "rd_quality": torch.rand(batch, length, 7),
        "tr_padding_mask": torch.zeros(batch, length, dtype=torch.bool),
        "known_gate_features": torch.rand(batch, length, evidence_dim),
        "known_gate_feature_valid": torch.ones(
            batch, length, evidence_dim, dtype=torch.bool
        ),
    }


def _feature_probe() -> dict[str, Any]:
    velocity = np.linspace(-85.0, 85.0, 384, dtype=np.float32).astype(np.float64)
    offsets = np.linspace(-75.0, 75.0, 21, dtype=np.float64)
    target_row = int(np.argmin(np.abs(offsets)))
    amplitude = np.full((2, 21, 384), -80.0, dtype=np.float32)
    amplitude[:, target_row, :] = 0.0
    values, valid = extract_canonical_known_gate_features(
        amplitude,
        velocity,
        offsets,
        tr_radial_speed_mps=None,
        tr_range_m=None,
        mtd_detected_range_m=[100.0, 100.0],
    )
    target_power = 384.0
    background_power = float(20 * 384) * 1.0e-8
    expected = target_power / (target_power + background_power)
    if (
        not valid[2]
        or not np.allclose(values[2], expected, atol=1.0e-7)
        or valid[:2].any()
        or valid[3]
        or np.any(values[:2] != 0.0)
        or values[3] != 0.0
    ):
        raise AssertionError("V21c target-row linear power fraction formula changed")

    distributed = np.zeros((1, 21, 384), dtype=np.float32)
    distributed_values, distributed_valid = extract_canonical_known_gate_features(
        distributed,
        velocity,
        offsets,
        tr_radial_speed_mps=0.0,
        tr_range_m=100.0,
        mtd_detected_range_m=[100.0],
    )
    if (
        not distributed_valid[2]
        or not np.allclose(distributed_values[2], 1.0 / 21.0, atol=1.0e-7)
        or not 0.0 < distributed_values[2] < values[2] <= 1.0
    ):
        raise AssertionError("V21c iso does not preserve concentrated-versus-distributed energy")
    return {
        "concentrated_target_row_iso": float(values[2]),
        "uniform_range_iso": float(distributed_values[2]),
        "tr_missingness_does_not_invalidate_iso": True,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    feature_probe = _feature_probe()
    config_path = root / CONFIG
    baseline_path = root / BASELINE_CONFIG
    old_path = root / OLD_SEED42_CONFIG
    if not all(path.is_file() for path in (config_path, baseline_path, old_path)):
        raise FileNotFoundError(
            "V21c formal/selected-Baseline/old configs are incomplete"
        )
    config = load_known_gate_config(config_path)
    baseline_config = load_known_gate_config(baseline_path)
    if (
        config.variant != V21C_VARIANT
        or config.train.seed != 43
        or baseline_config.train.seed != 43
        or config.feature_names != ISO_FEATURE_NAMES
    ):
        raise AssertionError("V21c formal config is not the selected seed43 pair")
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if readiness.get("selected_baseline_seed") != 43:
        raise AssertionError("V21c readiness did not bind selected seed43")

    treatment, baseline, evidence = build_known_gate_model_for_config(root, config)
    if (
        baseline is None
        or evidence.get("paired_baseline") is not True
        or evidence.get("shared_parameters_and_buffers_exactly_equal") is not True
        or evidence.get("new_fusion_columns_exactly_zero") is not True
        or evidence.get("new_fusion_columns") != 1
    ):
        raise AssertionError("V21c paired initialization failed")
    batch = _batch(config.model.evidence_dim)
    treatment.eval()
    baseline.eval()
    with torch.inference_mode():
        treatment_logits = treatment(**batch)["sequence_logits"]
        baseline_batch = dict(batch)
        baseline_batch.pop("known_gate_features")
        baseline_batch.pop("known_gate_feature_valid")
        baseline_logits = baseline(**baseline_batch)["sequence_logits"]
    if not torch.equal(treatment_logits, baseline_logits):
        raise AssertionError("V21c zero-column initial logits differ from seed43 Baseline")

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
    if not torch.allclose(
        torch.stack(online_logits, dim=1),
        treatment_logits,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError("V21c offline/online logits differ")

    treatment.train()
    objective = parent_final_task_ce_with_prefix_auxiliary(
        treatment(**batch)["sequence_logits"],
        torch.tensor([0, 1], dtype=torch.long),
        batch["tr_padding_mask"],
        class_weights=torch.ones(2),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    objective["loss"].backward()
    new_gradient = treatment.fusion[0].weight.grad[:, baseline.fusion_input_dim :]
    if (
        new_gradient is None
        or not torch.isfinite(new_gradient).all()
        or float(new_gradient.abs().max()) <= 0.0
    ):
        raise AssertionError("V21c iso column did not receive finite non-zero gradient")
    if any(not parameter.requires_grad for parameter in treatment.parameters()):
        raise AssertionError("V21c does not expose full-parameter training")

    with tempfile.TemporaryDirectory(prefix="v21c-cache-smoke-") as temporary:
        path = Path(temporary) / "cache.npz"
        sample_ids = np.asarray(["a", "b"], dtype=np.str_)
        values = np.zeros((2, KNOWN_GATE_FEATURE_DIM), dtype=np.float32)
        valid = np.zeros_like(values, dtype=np.bool_)
        values[:, 4] = np.asarray([0.2, 0.8], dtype=np.float32)
        valid[:, 4] = True
        np.savez(path, sample_ids=sample_ids, values=values, valid=valid)
        loaded, loaded_valid, metadata = load_known_gate_feature_cache(
            path,
            expected_sample_ids=["a", "b"],
            variant=V21C_VARIANT,
        )
        if (
            loaded["b"].shape != (1,)
            or not np.array_equal(loaded["b"], values[1, 4:5])
            or not loaded_valid["a"].all()
            or metadata.get("feature_names") != list(ISO_FEATURE_NAMES)
        ):
            raise AssertionError("V21c one-column cache selection failed")

    old_config = load_known_gate_config(old_path)
    try:
        _verify_readiness(root, old_path, old_config, {"hashes": {}})
    except PermissionError as exc:
        if "seed42" not in str(exc) or "not comparable" not in str(exc):
            raise
    else:
        raise AssertionError("old seed42 V21c candidate was not fail-closed")

    baseline_checkpoint = root / BASELINE_ARTIFACT_LAST
    payload = torch.load(baseline_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("config_identity_sha256") != known_gate_config_identity_sha256(
        baseline_config
    ):
        raise AssertionError("V21c config support broke frozen Baseline checkpoint identity")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21c-target-row-iso-contract-smoke-v1",
        "feature_probe": feature_probe,
        "selected_baseline_seed": 43,
        "same_seed_initialization": True,
        "new_fusion_columns_exactly_zero": True,
        "initial_logits_exactly_equal": True,
        "new_fusion_columns_receive_gradient": True,
        "all_model_parameters_trainable": True,
        "offline_online_logits_match": True,
        "core_six_column_cache_selects_only_iso": True,
        "old_seed42_candidate_training_blocked": True,
        "primary_hypothesis_labels": ["DroneTarget", "ClutterTarget"],
        "safety_labels": ["BirdTarget", "BalloonTarget"],
        "frozen_baseline_checkpoint_identity_preserved": True,
        "test": "未执行",
    }


BASELINE_ARTIFACT_LAST = Path(
    "artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-formal/last_model.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the selected-seed43 V21c target-row iso experiment."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
