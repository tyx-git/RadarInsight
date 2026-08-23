"""Contract smoke test for V24a current-event RD peak-drift evidence."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v24a import CONFIG, READINESS
from radar_ln3.known_gate import build_known_gate_model_for_config, load_known_gate_config
from radar_ln3.known_gate_features import (
    V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
    extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction,
    load_known_gate_feature_cache,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _verify_contract, _verify_readiness


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(24043)
    batch, length, frames = 2, 5, 3
    features = torch.rand(batch, length, evidence_dim)
    valid = torch.ones(batch, length, evidence_dim, dtype=torch.bool)
    features[0, 1, 1] = 0.0
    valid[0, 1, 1] = False
    return {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor(
            [[0.0, 0.0, 1.0, 1.0, 2.0], [0.0, 0.0, 2.0, 1.0, 1.0]],
            dtype=torch.float32,
        ),
        "tr_dt_valid": torch.tensor(
            [[False, False, True, True, True], [False, False, True, True, True]]
        ),
        "tr_dt_reason": torch.tensor(
            [[1, 2, 3, 3, 3], [1, 2, 3, 3, 3]], dtype=torch.long
        ),
        "rd_frames": torch.randn(batch, length, frames, 2, 21, 384),
        "rd_frame_mask": torch.ones(batch, length, frames, dtype=torch.bool),
        "rd_quality": torch.rand(batch, length, 7),
        "tr_padding_mask": torch.zeros(batch, length, dtype=torch.bool),
        "known_gate_features": features,
        "known_gate_feature_valid": valid,
    }


def _synthetic_rd() -> tuple[np.ndarray, np.ndarray]:
    canonical = (
        Path(__file__).resolve().parents[2]
        / "data/processed/cq08_rd_physical_canonical_v3_1/cq08__track__1_rd_tensor.npz"
    )
    with np.load(canonical, allow_pickle=False) as payload:
        velocity = np.asarray(payload["velocity_mps"])
    amplitude = np.full((2, 21, 384), -80.0, dtype=np.float32)
    amplitude[0, :, 100] = 0.0
    amplitude[1, :, 140] = 0.0
    return amplitude, velocity


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    if not config_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V24a prepare artifacts are incomplete")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if (
        config.variant != V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT
        or config.model.evidence_dim != 2
        or config.feature_names[-1]
        != "current_event_adjacent_rd_peak_velocity_drift_mean_fraction"
        or readiness.get("status") != "passed"
    ):
        raise AssertionError("V24a config/readiness does not bind the new feature")
    amplitude, velocity = _synthetic_rd()
    invalid_value, invalid_valid = (
        extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction(
            amplitude[:1], velocity
        )
    )
    valid_zero_value, valid_zero_valid = (
        extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction(
            np.repeat(amplitude[:1], 2, axis=0), velocity
        )
    )
    drift_value, drift_valid = (
        extract_current_event_adjacent_rd_peak_velocity_drift_mean_fraction(
            amplitude, velocity
        )
    )
    if (
        invalid_value.tolist() != [0.0]
        or invalid_valid.tolist() != [False]
        or valid_zero_value.tolist() != [0.0]
        or valid_zero_valid.tolist() != [True]
        or drift_valid.tolist() != [True]
        or not 0.0 < float(drift_value[0]) <= 1.0
    ):
        raise AssertionError("V24a RD drift invalid/valid-zero semantics differ")
    treatment, control, pairing = build_known_gate_model_for_config(root, config)
    if (
        control is None
        or pairing.get("paired_baseline") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise AssertionError("V24a paired V24 initialization failed")
    batch = _batch(config.model.evidence_dim)
    treatment.eval()
    control.eval()
    with torch.inference_mode():
        treatment_logits = treatment(**batch)["sequence_logits"]
        control_batch = dict(batch)
        control_batch["known_gate_features"] = control_batch["known_gate_features"][..., :1]
        control_batch["known_gate_feature_valid"] = control_batch["known_gate_feature_valid"][..., :1]
        control_logits = control(**control_batch)["sequence_logits"]
    if not torch.equal(treatment_logits, control_logits):
        raise AssertionError("V24a zero appended drift column changes initial logits")
    audit_values = torch.zeros((1, 2, config.model.evidence_dim), dtype=torch.float32)
    audit_valid = torch.ones_like(audit_values, dtype=torch.bool)
    # A physical zero and an unavailable value remain different cache/audit
    # states, but validity is intentionally gating-only in V24a.  They must
    # not become an implicit RD-availability feature at the model input.
    audit_valid[0, 1, 1] = False
    gated = treatment._evidence_input(
        audit_values,
        audit_valid,
        leading_shape=(1, 2),
        dtype=torch.float32,
    )
    if gated is None or not torch.equal(gated[0, 0, 1], gated[0, 1, 1]):
        raise AssertionError("V24a invalid and physical-zero drift inputs must remain gating-only")
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
    gradient = treatment.fusion[0].weight.grad[:, -1]
    if (
        gradient is None
        or not torch.isfinite(gradient).all()
        or float(gradient.abs().max()) <= 0.0
    ):
        raise AssertionError("V24a drift fusion column receives no finite gradient")
    with tempfile.TemporaryDirectory(prefix="v24a-cache-smoke-") as temporary:
        path = Path(temporary) / "cache.npz"
        sample_ids = np.asarray(["a", "b"], dtype=np.str_)
        values = np.asarray([[0.2, 0.3], [0.0, 0.0]], dtype=np.float32)
        valid = np.asarray([[True, True], [False, False]], dtype=np.bool_)
        np.savez(path, sample_ids=sample_ids, values=values, valid=valid)
        loaded, loaded_valid, metadata = load_known_gate_feature_cache(
            path,
            expected_sample_ids=["a", "b"],
            variant=V24A_COURSE_RD_PEAK_DRIFT_MEAN_VARIANT,
        )
        if (
            not np.array_equal(loaded["a"], values[0])
            or bool(loaded_valid["b"][1])
            or metadata.get("feature_names", [])[-1]
            != "current_event_adjacent_rd_peak_velocity_drift_mean_fraction"
        ):
            raise AssertionError("V24a cache selection differs from contract")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V24a-known-gate-course-rd-peak-drift-mean-smoke-v1",
        "current_event_only": True,
        "one_frame_event_is_invalid": True,
        "valid_zero_drift_is_not_invalid": True,
        "rd_drift_validity_is_cache_audit_only_not_a_model_input": True,
        "no_preceding_event_bridge": True,
        "v24_shared_initialization_exact": True,
        "initial_logits_exactly_equal": True,
        "new_drift_column_receives_gradient": True,
        "unknown_training_exposure": 0,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the V24a RD-drift contract.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
