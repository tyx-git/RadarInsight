from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v21e import BASELINE_CONFIG, CONFIG
from radar_ln3.known_gate import (
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    V21E_COURSE_PREFIX_VARIANT,
    extract_causal_course_circular_incoherence,
    load_known_gate_feature_cache,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _verify_contract, _verify_readiness


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(2105)
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


def _course_probe() -> dict[str, Any]:
    # First event (123 degrees) must never seed the statistic.  The later true
    # zero-degree event is physical and must participate.  A missing event
    # carries an already-valid prefix without adding evidence.
    courses = [123.0, 350.0, 0.0, 10.0, None, 180.0]
    values, valid = extract_causal_course_circular_incoherence(courses)
    coherent_resultant = abs(
        np.mean(np.exp(1j * np.deg2rad(np.asarray([350.0, 0.0, 10.0]))))
    )
    expected = 1.0 - float(coherent_resultant)
    if valid[:, 0].tolist() != [False, False, False, True, True, True]:
        raise AssertionError("V21e minimum support/causal validity changed")
    if not np.isclose(float(values[3, 0]), expected, atol=1.0e-7):
        raise AssertionError("V21e circular formula or later zero-degree handling changed")
    if not np.array_equal(values[3], values[4]):
        raise AssertionError("V21e missing event did not carry the valid causal prefix")
    changed_first, changed_valid = extract_causal_course_circular_incoherence(
        [-77.0, 350.0, 0.0, 10.0, None, 180.0]
    )
    if not np.array_equal(values, changed_first) or not np.array_equal(valid, changed_valid):
        raise AssertionError("V21e first event leaked into the course statistic")
    wrapped, wrapped_valid = extract_causal_course_circular_incoherence(
        [0.0, 359.0, 0.0, 1.0]
    )
    if not bool(wrapped_valid[-1, 0]) or float(wrapped[-1, 0]) >= 0.001:
        raise AssertionError("V21e did not preserve circular wraparound")
    return {
        "courses_degrees": courses,
        "values": values[:, 0].astype(float).tolist(),
        "valid": valid[:, 0].tolist(),
        "first_event_unconditionally_excluded": True,
        "later_true_zero_degree_included": True,
        "minimum_valid_nonfirst_course_count": 3,
        "missing_event_carries_valid_prefix": True,
        "wraparound_is_circular": True,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    course_probe = _course_probe()
    config_path = root / CONFIG
    baseline_path = root / BASELINE_CONFIG
    if not all(path.is_file() for path in (config_path, baseline_path)):
        raise FileNotFoundError("V21e formal/Baseline configs are incomplete")
    config = load_known_gate_config(config_path)
    baseline_config = load_known_gate_config(baseline_path)
    if (
        config.variant != V21E_COURSE_PREFIX_VARIANT
        or config.train.seed != 43
        or baseline_config.train.seed != 43
        or config.feature_names != COURSE_PREFIX_FEATURE_NAMES
    ):
        raise AssertionError("V21e formal config is not the selected seed43 pair")
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if (
        readiness.get("selected_baseline_seed") != 43
        or readiness.get("post_hoc_feature_discovery") is not True
    ):
        raise AssertionError("V21e readiness did not bind the disclosed seed43 pair")

    treatment, baseline, evidence = build_known_gate_model_for_config(root, config)
    if (
        baseline is None
        or evidence.get("paired_baseline") is not True
        or evidence.get("shared_parameters_and_buffers_exactly_equal") is not True
        or evidence.get("new_fusion_columns_exactly_zero") is not True
        or evidence.get("new_fusion_columns") != 1
    ):
        raise AssertionError("V21e paired initialization failed")
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
        raise AssertionError("V21e zero-column initial logits differ from Baseline")

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
        raise AssertionError("V21e offline/online logits differ")

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
        raise AssertionError("V21e feature column received no finite non-zero gradient")
    if any(not parameter.requires_grad for parameter in treatment.parameters()):
        raise AssertionError("V21e does not expose full-parameter training")

    with tempfile.TemporaryDirectory(prefix="v21e-cache-smoke-") as temporary:
        path = Path(temporary) / "cache.npz"
        sample_ids = np.asarray(["a", "b"], dtype=np.str_)
        values = np.asarray([[0.2], [0.8]], dtype=np.float32)
        valid = np.asarray([[True], [False]], dtype=np.bool_)
        values[~valid] = 0.0
        np.savez(path, sample_ids=sample_ids, values=values, valid=valid)
        loaded, loaded_valid, metadata = load_known_gate_feature_cache(
            path,
            expected_sample_ids=["a", "b"],
            variant=V21E_COURSE_PREFIX_VARIANT,
        )
        if (
            loaded["a"].shape != (1,)
            or not np.array_equal(loaded["a"], values[0])
            or bool(loaded_valid["b"][0])
            or metadata.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        ):
            raise AssertionError("V21e independent one-column cache selection failed")

    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21e-causal-course-contract-smoke-v1",
        "course_probe": course_probe,
        "selected_baseline_seed": 43,
        "same_seed_initialization": True,
        "new_fusion_columns_exactly_zero": True,
        "initial_logits_exactly_equal": True,
        "new_fusion_columns_receive_gradient": True,
        "all_model_parameters_trainable": True,
        "offline_online_logits_match": True,
        "independent_cache_selects_only_causal_course_incoherence": True,
        "post_hoc_feature_discovery_disclosed": True,
        "unknown_training_exposure": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test selected-seed43 V21e causal course consistency."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
