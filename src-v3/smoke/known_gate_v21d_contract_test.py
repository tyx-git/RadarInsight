from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v21d import (
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
    D_RANGE_PREFIX_FEATURE_NAMES,
    V21D_PREFIX_VARIANT,
    load_known_gate_feature_cache,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _verify_contract, _verify_readiness


BASELINE_ARTIFACT_LAST = Path(
    "artifacts/RADAR-LN3-V21-cq08-liquid-known-gate-baseline-seed43-formal/last_model.pt"
)


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(2104)
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


def _prefix_probe() -> dict[str, Any]:
    # Two event chunks. Every reported value uses only residuals observed up to that event.
    residuals: list[float] = []
    values: list[float] = []
    valid: list[bool] = []
    for current in ([1.0], [3.0, 5.0], []):
        residuals.extend(current)
        valid.append(len(residuals) >= 2)
        values.append(float(np.std(residuals, ddof=1)) if len(residuals) >= 2 else 0.0)
    expected = [0.0, 2.0, 2.0]
    if valid != [False, True, True] or not np.allclose(values, expected):
        raise AssertionError("V21d causal-prefix sample-standard-deviation semantics changed")
    return {
        "residual_sequence_m": [1.0, 3.0, 5.0],
        "event_prefix_std_m": values,
        "event_prefix_valid": valid,
        "missing_event_carries_already_valid_prefix_without_future_access": True,
        "ddof": 1,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    prefix_probe = _prefix_probe()
    config_path = root / CONFIG
    baseline_path = root / BASELINE_CONFIG
    old_path = root / OLD_SEED42_CONFIG
    if not all(path.is_file() for path in (config_path, baseline_path, old_path)):
        raise FileNotFoundError("V21d formal/Baseline/old configs are incomplete")
    config = load_known_gate_config(config_path)
    baseline_config = load_known_gate_config(baseline_path)
    if (
        config.variant != V21D_PREFIX_VARIANT
        or config.train.seed != 43
        or baseline_config.train.seed != 43
        or config.feature_names != D_RANGE_PREFIX_FEATURE_NAMES
    ):
        raise AssertionError("V21d formal config is not the selected seed43 pair")
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if readiness.get("selected_baseline_seed") != 43:
        raise AssertionError("V21d readiness did not bind seed43 Baseline")

    treatment, baseline, evidence = build_known_gate_model_for_config(root, config)
    if (
        baseline is None
        or evidence.get("paired_baseline") is not True
        or evidence.get("shared_parameters_and_buffers_exactly_equal") is not True
        or evidence.get("new_fusion_columns_exactly_zero") is not True
        or evidence.get("new_fusion_columns") != 1
    ):
        raise AssertionError("V21d paired initialization failed")
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
        raise AssertionError("V21d zero-column initial logits differ from Baseline")

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
        raise AssertionError("V21d offline/online logits differ")

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
    gradient = treatment.fusion[0].weight.grad[:, baseline.fusion_input_dim :]
    if (
        gradient is None
        or not torch.isfinite(gradient).all()
        or float(gradient.abs().max()) <= 0.0
    ):
        raise AssertionError("V21d feature column received no finite non-zero gradient")
    if any(not parameter.requires_grad for parameter in treatment.parameters()):
        raise AssertionError("V21d does not expose full-parameter training")

    with tempfile.TemporaryDirectory(prefix="v21d-cache-smoke-") as temporary:
        path = Path(temporary) / "cache.npz"
        sample_ids = np.asarray(["a", "b"], dtype=np.str_)
        values = np.asarray([[0.2], [0.8]], dtype=np.float32)
        valid = np.asarray([[True], [False]], dtype=np.bool_)
        values[~valid] = 0.0
        np.savez(path, sample_ids=sample_ids, values=values, valid=valid)
        loaded, loaded_valid, metadata = load_known_gate_feature_cache(
            path,
            expected_sample_ids=["a", "b"],
            variant=V21D_PREFIX_VARIANT,
        )
        if (
            loaded["a"].shape != (1,)
            or not np.array_equal(loaded["a"], values[0])
            or bool(loaded_valid["b"][0])
            or metadata.get("feature_names") != list(D_RANGE_PREFIX_FEATURE_NAMES)
        ):
            raise AssertionError("V21d independent one-column cache selection failed")

    old_config = load_known_gate_config(old_path)
    try:
        _verify_readiness(root, old_path, old_config, {"hashes": {}})
    except PermissionError as exc:
        if "current-event proxy" not in str(exc):
            raise
    else:
        raise AssertionError("old seed42 V21d proxy was not fail-closed")

    payload = torch.load(root / BASELINE_ARTIFACT_LAST, map_location="cpu", weights_only=False)
    if payload.get("config_identity_sha256") != known_gate_config_identity_sha256(
        baseline_config
    ):
        raise AssertionError("V21d support broke frozen Baseline checkpoint identity")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21d-causal-prefix-d-range-contract-smoke-v1",
        "prefix_probe": prefix_probe,
        "selected_baseline_seed": 43,
        "same_seed_initialization": True,
        "new_fusion_columns_exactly_zero": True,
        "initial_logits_exactly_equal": True,
        "new_fusion_columns_receive_gradient": True,
        "all_model_parameters_trainable": True,
        "offline_online_logits_match": True,
        "independent_cache_selects_only_causal_prefix_d_range": True,
        "old_current_event_proxy_training_blocked": True,
        "primary_hypothesis_labels": ["BirdTarget", "ClutterTarget"],
        "safety_labels": ["DroneTarget", "BalloonTarget"],
        "frozen_baseline_checkpoint_identity_preserved": True,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test selected-seed43 V21d causal-prefix d-range."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
