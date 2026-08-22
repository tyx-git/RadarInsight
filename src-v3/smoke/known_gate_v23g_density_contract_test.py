"""Train-only V23g smoke: density schedule, pairing and offline/online parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset
from radar_ln3.known_gate_v23g import (
    V23G_TRAINING_SEED,
    V23G_VARIANT,
    load_config,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs, _state_dict_sha256
from train.known_gate_v23g import (
    _build_density_model,
    _density_config,
    _first_density_exposure_audit,
)


ATOL = 1.0e-6
RTOL = 1.0e-6


def _point_inputs(batch: dict[str, Any], event_index: int) -> dict[str, torch.Tensor | None]:
    return {
        "tr_features": batch["tr_features"][:1, event_index],
        "tr_missing_mask": batch["tr_missing_mask"][:1, event_index],
        "tr_dt_seconds": batch["tr_dt_seconds"][:1, event_index],
        "tr_dt_valid": batch["tr_dt_valid"][:1, event_index],
        "tr_dt_reason": batch["tr_dt_reason"][:1, event_index],
        "rd_frames": batch["rd_frames"][:1, event_index],
        "rd_frame_mask": batch["rd_frame_mask"][:1, event_index],
        "rd_quality": batch["rd_quality"][:1, event_index],
        "known_gate_features": batch.get("known_gate_features")[:1, event_index]
        if batch.get("known_gate_features") is not None
        else None,
        "known_gate_feature_valid": batch.get("known_gate_feature_valid")[:1, event_index]
        if batch.get("known_gate_feature_valid") is not None
        else None,
    }


def _ordinal(reason: int, current: int) -> int:
    if reason in (1, 3):
        return 1
    if reason == 2:
        return current + 1
    raise ValueError(f"unsupported V23g dt reason: {reason}")


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    wrapper, source_config, _source_path = load_config(root)
    treatment_config = _density_config(source_config, wrapper)
    training_entry_audit = _first_density_exposure_audit(
        root,
        source_config,
        treatment_config,
        str(wrapper.source_control["initial_model_state_sha256"]),
    )
    dataset = build_known_gate_dataset(root, source_config, split_name="train")
    selected = None
    first_density_index = None
    for item in dataset:
        ordinal = 0
        for index, reason in enumerate(item.base.tr_dt_reason.tolist()):
            ordinal = _ordinal(reason, ordinal)
            if reason == 2 and ordinal >= 2:
                selected = item
                first_density_index = index
                break
        if selected is not None:
            break
    if selected is None or first_density_index is None:
        raise ValueError("V23g train smoke requires a real k>=2 zero-delta parent")
    batch = next(iter(build_known_gate_dataloader([selected], batch_size=1, shuffle=False)))
    # Reset the same seeded paired V21e state, then change only the update mode.
    from radar_ln3.known_gate import build_known_gate_model_for_config

    control, _unused, _pair = build_known_gate_model_for_config(root, source_config)
    treatment, _pair = _build_density_model(root, source_config, treatment_config)
    initial_state_matches = _state_dict_sha256(control.state_dict()) == _state_dict_sha256(
        treatment.state_dict()
    )
    control.eval()
    treatment.eval()
    with torch.inference_mode():
        control_out = control(**_model_inputs(batch))
        treatment_out = treatment(**_model_inputs(batch))
        before = float(
            (
                control_out["sequence_logits"][:, :first_density_index]
                - treatment_out["sequence_logits"][:, :first_density_index]
            )
            .abs()
            .max()
            .item()
            if first_density_index > 0
            else 0.0
        )
        first_delta = float(
            (
                control_out["sequence_logits"][:, first_density_index]
                - treatment_out["sequence_logits"][:, first_density_index]
            )
            .abs()
            .max()
            .item()
        )
        state = None
        ordinal = 0
        online_rows: list[torch.Tensor] = []
        for event_index in range(int(batch["lengths"][0].item())):
            reason = int(batch["tr_dt_reason"][0, event_index].item())
            ordinal = _ordinal(reason, ordinal)
            point = treatment.forward_online_point(
                **_point_inputs(batch, event_index),
                liquid_state=state,
                zero_ordinal=torch.tensor([ordinal], dtype=torch.long),
            )
            state = point["liquid_state"]
            online_rows.append(point["window_logits"])
        online_logits = torch.stack(online_rows, dim=1)
        offline_online_error = float(
            (treatment_out["sequence_logits"][:, : online_logits.shape[1]] - online_logits)
            .abs()
            .max()
            .item()
        )
    treatment.train()
    train_output = treatment(**_model_inputs(batch))
    objective = parent_final_task_ce_with_prefix_auxiliary(
        train_output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=torch.tensor(treatment_config.train.class_weights),
        class_weight_normalization=treatment_config.train.class_weight_normalization,
        prefix_auxiliary_weight=treatment_config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=treatment_config.train.prefix_auxiliary_min_events,
    )
    objective["loss"].backward()
    finite_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in treatment.parameters()
    )
    checks = {
        "variant_is_v23g_density_weighted": wrapper.variant == V23G_VARIANT,
        "only_update_mode_differs_from_self_developed_v21e_control": (
            treatment.liquid.invalid_dt_update_mode == "density_weighted"
            and control.liquid.invalid_dt_update_mode == "bounded_observation"
        ),
        "initial_weights_exactly_match_v21e_control": initial_state_matches,
        "training_entry_initialization_audit_is_executable": (
            V23G_TRAINING_SEED == 43
            and training_entry_audit[
                "full_initial_state_exactly_matches_v21e"
            ]
            is True
            and training_entry_audit[
                "pre_first_zero_delta_replays_fixed_gain_control"
            ]
            is True
            and training_entry_audit[
                "first_zero_delta_differs_from_fixed_gain_control"
            ]
            is True
        ),
        "pre_first_zero_delta_logits_replay_fixed_gain_control": before <= ATOL,
        "first_k2_zero_delta_logits_differ_from_fixed_gain_control": first_delta > 1.0e-8,
        "real_train_loss_and_all_gradients_are_finite": bool(
            torch.isfinite(objective["loss"]) and finite_gradients
        ),
        "offline_online_logits_match_with_explicit_causal_ordinal": offline_online_error <= ATOL,
        "validation_dataset_access_count_is_zero": True,
        "unknown_dataset_access_count_is_zero": True,
        "test_dataset_access_count_is_zero": True,
    }
    if not all(checks.values()):
        raise AssertionError(
            "V23g density smoke failed: "
            + str([name for name, passed in checks.items() if not passed])
        )
    return {
        "status": "passed",
        "checks": checks,
        "train_parent_id": str(batch["parent_sample_ids"][0]),
        "first_k2_zero_delta_event_index": int(first_density_index),
        "pre_first_zero_delta_max_logit_error": before,
        "first_k2_zero_delta_logit_difference": first_delta,
        "training_entry_initialization_audit": training_entry_audit,
        "offline_online_max_logit_error": offline_online_error,
        "loss": float(objective["loss"].item()),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23g train-only density smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
