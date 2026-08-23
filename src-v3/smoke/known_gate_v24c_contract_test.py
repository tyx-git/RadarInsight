"""Contract smoke test for V24c strict same-second recurrent-state skipping."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from build.known_gate_v24 import CONFIG as CONTROL_CONFIG
from build.known_gate_v24c import CONFIG, READINESS
from radar_ln3.data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)
from radar_ln3.known_gate import build_known_gate_model_for_config, load_known_gate_config
from radar_ln3.known_gate_features import (
    COURSE_PREFIX_FEATURE_NAMES,
    V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
    load_known_gate_feature_cache,
)
from radar_ln3.model import CfCLiquidCausalEncoder
from train.known_gate_v21 import _verify_contract, _verify_readiness


def _batch(evidence_dim: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(24043)
    batch, length, frames = 2, 4, 3
    return {
        "tr_features": torch.randn(batch, length, 14),
        "tr_missing_mask": torch.zeros(batch, length, 11),
        "tr_dt_seconds": torch.tensor(
            [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 2.0]],
            dtype=torch.float32,
        ),
        "tr_dt_valid": torch.tensor(
            [[False, False, False, True], [False, False, False, True]]
        ),
        "tr_dt_reason": torch.tensor(
            [
                [
                    DT_REASON_FIRST_EVENT,
                    DT_REASON_ZERO_DELTA,
                    DT_REASON_ZERO_DELTA,
                    DT_REASON_POSITIVE_DELTA,
                ],
                [
                    DT_REASON_FIRST_EVENT,
                    DT_REASON_ZERO_DELTA,
                    DT_REASON_ZERO_DELTA,
                    DT_REASON_POSITIVE_DELTA,
                ],
            ],
            dtype=torch.long,
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


def _assert_state_semantics() -> dict[str, bool]:
    torch.manual_seed(24044)
    strict = CfCLiquidCausalEncoder(
        4,
        4,
        invalid_dt_update_mode="strict_skip",
        state_normalization_mode="recurrent",
    )
    bounded = CfCLiquidCausalEncoder(
        4,
        4,
        invalid_dt_update_mode="bounded_observation",
        state_normalization_mode="recurrent",
    )
    bounded.load_state_dict(strict.state_dict())
    state = torch.tensor([[0.2, -0.4, 0.7, 0.1]], dtype=torch.float32)
    first_event = torch.tensor([[0.5, 0.2, -0.1, 0.3]], dtype=torch.float32)
    zero_a = torch.tensor([[1.2, -0.8, 0.4, 0.6]], dtype=torch.float32)
    zero_b = torch.tensor([[-0.9, 0.7, -0.2, 1.1]], dtype=torch.float32)
    positive = torch.tensor([[0.3, 0.9, -0.5, 0.2]], dtype=torch.float32)
    zero_dt = torch.zeros(1)
    positive_dt = torch.ones(1)
    invalid = torch.zeros(1, dtype=torch.bool)
    valid = torch.ones(1, dtype=torch.bool)
    first_reason = torch.full((1,), DT_REASON_FIRST_EVENT, dtype=torch.long)
    zero_reason = torch.full((1,), DT_REASON_ZERO_DELTA, dtype=torch.long)
    positive_reason = torch.full((1,), DT_REASON_POSITIVE_DELTA, dtype=torch.long)

    initialized = strict._step(first_event, zero_dt, state, invalid, first_reason)[0]
    strict_zero_a = strict._step(zero_a, zero_dt, initialized, invalid, zero_reason)[0]
    strict_zero_b = strict._step(zero_b, zero_dt, initialized, invalid, zero_reason)[0]
    bounded_zero = bounded._step(zero_a, zero_dt, initialized, invalid, zero_reason)[0]
    positive_state = strict._step(
        positive, positive_dt, strict_zero_a, valid, positive_reason
    )[0]
    if not torch.equal(strict_zero_a, initialized) or not torch.equal(
        strict_zero_b, initialized
    ):
        raise AssertionError("V24c strict skip did not restore exact incoming recurrent state")
    if torch.equal(bounded_zero, initialized):
        raise AssertionError("V24 bounded control unexpectedly did not update at zero delta")
    if torch.equal(initialized, state) or torch.equal(positive_state, strict_zero_a):
        raise AssertionError("V24c changed first-event or positive-dt semantics")
    return {
        "strict_zero_delta_state_equals_entering_state": True,
        "zero_delta_payload_cannot_change_state": True,
        "bounded_control_changes_state_at_zero_delta": True,
        "first_and_positive_updates_still_change_state": True,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    if not config_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V24c prepare artifacts are incomplete")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if (
        config.variant != V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT
        or config.feature_names != COURSE_PREFIX_FEATURE_NAMES
        or config.model.evidence_dim != 1
        or config.model.liquid_invalid_dt_update_mode != "strict_skip"
        or readiness.get("status") != "passed"
    ):
        raise AssertionError("V24c config/readiness does not bind strict same-second skip")
    strict_model, bounded_control, pairing = build_known_gate_model_for_config(root, config)
    if (
        bounded_control is None
        or pairing.get("dynamics_only_pair") is not True
        or pairing.get("shared_parameters_and_buffers_exactly_equal") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 0
        or strict_model.liquid.invalid_dt_update_mode != "strict_skip"
        or bounded_control.liquid.invalid_dt_update_mode != "bounded_observation"
        or any(
            not torch.equal(left, right)
            for left, right in zip(
                strict_model.state_dict().values(),
                bounded_control.state_dict().values(),
                strict=True,
            )
        )
    ):
        raise AssertionError("V24c and reconstructed V24 do not share exact initial state")

    semantics = _assert_state_semantics()
    batch = _batch(config.model.evidence_dim)
    strict_model.eval()
    bounded_control.eval()
    with torch.inference_mode():
        strict_output = strict_model(**batch)
        bounded_output = bounded_control(**batch)
    strict_sequence = strict_output["liquid_sequence"]
    if not torch.equal(strict_sequence[:, 1], strict_sequence[:, 0]) or not torch.equal(
        strict_sequence[:, 2], strict_sequence[:, 1]
    ):
        raise AssertionError("V24c strict model state/readout changes at zero-delta positions")
    if torch.equal(bounded_output["liquid_sequence"][:, 1], bounded_output["liquid_sequence"][:, 0]):
        raise AssertionError("V24 control is not distinguishable from strict skip at zero delta")

    no_zero_batch = {key: value.clone() for key, value in batch.items()}
    zero_mask = no_zero_batch["tr_dt_reason"] == DT_REASON_ZERO_DELTA
    no_zero_batch["tr_dt_seconds"][zero_mask] = 1.0
    no_zero_batch["tr_dt_valid"][zero_mask] = True
    no_zero_batch["tr_dt_reason"][zero_mask] = DT_REASON_POSITIVE_DELTA
    with torch.inference_mode():
        strict_no_zero = strict_model(**no_zero_batch)["sequence_logits"]
        bounded_no_zero = bounded_control(**no_zero_batch)["sequence_logits"]
    if not torch.equal(strict_no_zero, bounded_no_zero):
        raise AssertionError("V24c and V24 differ without same-second events")

    online_state = None
    online_logits: list[torch.Tensor] = []
    online_states: list[torch.Tensor] = []
    with torch.inference_mode():
        for event_index in range(batch["tr_features"].shape[1]):
            output = strict_model.forward_online_point(
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
            online_states.append(online_state)
            online_logits.append(output["window_logits"])
    if not torch.allclose(
        torch.stack(online_logits, dim=1),
        strict_output["sequence_logits"],
        atol=1.0e-6,
        rtol=1.0e-6,
    ) or not torch.allclose(
        torch.stack(online_states, dim=1),
        strict_sequence,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError("V24c offline and online states/logits differ")

    padded_dt = torch.cat([batch["tr_dt_seconds"], torch.zeros(2, 1)], dim=1)
    padded_valid = torch.cat([batch["tr_dt_valid"], torch.zeros(2, 1, dtype=torch.bool)], dim=1)
    padded_reason = torch.cat(
        [
            batch["tr_dt_reason"],
            torch.full((2, 1), DT_REASON_PADDING, dtype=torch.long),
        ],
        dim=1,
    )
    padded_mask = torch.cat(
        [batch["tr_padding_mask"], torch.ones(2, 1, dtype=torch.bool)], dim=1
    )
    padding_encoder = strict_model.liquid
    point_events = torch.randn(2, 5, padding_encoder.hidden_dim)
    with torch.inference_mode():
        original_state = padding_encoder(
            point_events[:, :4],
            batch["tr_dt_seconds"],
            batch["tr_padding_mask"],
            dt_valid=batch["tr_dt_valid"],
            dt_reason=batch["tr_dt_reason"],
        ).final_state
        padded_state = padding_encoder(
            point_events,
            padded_dt,
            padded_mask,
            dt_valid=padded_valid,
            dt_reason=padded_reason,
        ).final_state
    if not torch.equal(original_state, padded_state):
        raise AssertionError("V24c padding changed recurrent state")

    with tempfile.TemporaryDirectory(prefix="v24c-cache-smoke-") as temporary:
        cache_path = Path(temporary) / "cache.npz"
        sample_ids = np.asarray(["a", "b"], dtype=np.str_)
        values = np.asarray([[0.2], [0.0]], dtype=np.float32)
        valid = np.asarray([[True], [False]], dtype=np.bool_)
        np.savez(cache_path, sample_ids=sample_ids, values=values, valid=valid)
        loaded, loaded_valid, metadata = load_known_gate_feature_cache(
            cache_path,
            expected_sample_ids=["a", "b"],
            variant=V24C_COURSE_STRICT_SAME_SECOND_SKIP_VARIANT,
        )
        if (
            not np.array_equal(loaded["a"], values[0])
            or bool(loaded_valid["b"][0])
            or metadata.get("feature_names") != list(COURSE_PREFIX_FEATURE_NAMES)
        ):
            raise AssertionError("V24c reused course cache selection differs from contract")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V24c-known-gate-course-strict-same-second-skip-smoke-v1",
        "v24_control_and_v24c_initial_state_exact": True,
        "no_new_feature_or_fusion_column": True,
        **semantics,
        "no_same_second_logits_match_exactly": True,
        "same_second_changes_state_only_in_bounded_control": True,
        "offline_online_states_and_logits_match": True,
        "padding_has_no_effect": True,
        "course_cache_width_is_one": True,
        "unknown_training_exposure": 0,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test the V24c strict-skip contract.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
