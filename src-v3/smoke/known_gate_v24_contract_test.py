"""Contract smoke test for V24's fresh course-reason-aware baseline."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v24 import CONFIG, READINESS
from radar_ln3.known_gate import build_known_gate_model_for_config, load_known_gate_config
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
    load_known_gate_feature_cache,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from report.known_gate_v24 import _clutter_wilson
from train.known_gate_v21 import _verify_contract, _verify_readiness


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(24043)
    batch, length, frames = 2, 5, 3
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
        "known_gate_features": torch.rand(batch, length, evidence_dim),
        "known_gate_feature_valid": torch.ones(
            batch, length, evidence_dim, dtype=torch.bool
        ),
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    if not config_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V24 prepare artifacts are incomplete")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if (
        config.variant != V24_COURSE_REASON_AWARE_BASELINE_VARIANT
        or config.feature_names != COURSE_PREFIX_FEATURE_NAMES
        or config.train.seed != 43
        or config.model.evidence_dim != 1
        or readiness.get("fresh_training_not_finetune") is not True
        or not str(readiness.get("initial_model_state_sha256", ""))
    ):
        raise AssertionError("V24 config/readiness does not bind the fresh baseline")
    treatment, baseline, pairing = build_known_gate_model_for_config(root, config)
    if (
        baseline is None
        or pairing.get("paired_baseline") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or pairing.get("new_fusion_columns_exactly_zero") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 1
    ):
        raise AssertionError("V24 paired fresh initialization failed")
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
        raise AssertionError("V24 zero-column initial logits differ from fresh baseline")
    online_state = None
    online_logits: list[torch.Tensor] = []
    with torch.inference_mode():
        for event_index in range(batch["tr_features"].shape[1]):
            output = treatment.forward_online_point(
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
            online_state = output["liquid_state"]
            online_logits.append(output["window_logits"])
    if not torch.allclose(
        torch.stack(online_logits, dim=1),
        treatment_logits,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError("V24 offline/online logits differ")
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
        raise AssertionError("V24 course feature column received no finite gradient")
    with tempfile.TemporaryDirectory(prefix="v24-cache-smoke-") as temporary:
        cache_path = Path(temporary) / "cache.npz"
        sample_ids = np.asarray(["a", "b"], dtype=np.str_)
        values = np.asarray([[0.2], [0.0]], dtype=np.float32)
        valid = np.asarray([[True], [False]], dtype=np.bool_)
        np.savez(cache_path, sample_ids=sample_ids, values=values, valid=valid)
        loaded, loaded_valid, metadata = load_known_gate_feature_cache(
            cache_path,
            expected_sample_ids=["a", "b"],
            variant=V24_COURSE_REASON_AWARE_BASELINE_VARIANT,
        )
        if (
            not np.array_equal(loaded["a"], values[0])
            or bool(loaded_valid["b"][0])
            or metadata.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        ):
            raise AssertionError("V24 course cache selection differs from contract")
    zero_predicted_clutter = _clutter_wilson(
        [{"true_id": 0, "pred_id": 0} for _ in range(223)]
        + [{"true_id": 1, "pred_id": 0} for _ in range(9)]
    )
    if (
        zero_predicted_clutter["clutter_precision"]["status"]
        != "undefined_zero_denominator"
        or zero_predicted_clutter["clutter_recall"]["estimate"] != 0.0
    ):
        raise AssertionError("V24 report cannot preserve a zero-predicted-Clutter result")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V24-known-gate-course-reason-aware-baseline-smoke-v1",
        "fresh_training_not_finetune": True,
        "same_seed_initialization": True,
        "new_course_fusion_column_exactly_zero": True,
        "initial_logits_exactly_equal": True,
        "new_fusion_column_receives_gradient": True,
        "offline_online_logits_match": True,
        "course_cache_width_is_one": True,
        "zero_predicted_clutter_report_is_defined": True,
        "unknown_training_exposure": 0,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the V24 baseline contract.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
