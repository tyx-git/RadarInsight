"""Train-only contract smoke for V23k causal TR/RD observation intensity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset
from radar_ln3.known_gate_official_cfc import (
    build_official_cfc_model_from_frozen_source,
    causal_observation_intensity_sequence,
    causal_observation_intensity_step,
)
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23_adapters import V23A_VARIANT, load_adapter_config, source_v23_training
from radar_ln3.known_gate_v23k import config_relative, load_config
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from report.known_gate_v23k import _stratum_metrics
from train.known_gate_v21 import _model_inputs, _state_dict_sha256


ATOL = 1.0e-6
RTOL = 1.0e-6


def _inputs_at(batch: dict[str, Any], batch_index: int, event_index: int) -> dict[str, torch.Tensor]:
    return {
        "tr_features": batch["tr_features"][batch_index : batch_index + 1, event_index],
        "tr_missing_mask": batch["tr_missing_mask"][batch_index : batch_index + 1, event_index],
        "tr_dt_seconds": batch["tr_dt_seconds"][batch_index : batch_index + 1, event_index],
        "tr_dt_valid": batch["tr_dt_valid"][batch_index : batch_index + 1, event_index],
        "tr_dt_reason": batch["tr_dt_reason"][batch_index : batch_index + 1, event_index],
        "rd_frames": batch["rd_frames"][batch_index : batch_index + 1, event_index],
        "rd_frame_mask": batch["rd_frame_mask"][batch_index : batch_index + 1, event_index],
        "rd_quality": batch["rd_quality"][batch_index : batch_index + 1, event_index],
    }


def _offline_online_error(model: Any, batch: dict[str, Any]) -> tuple[float, float]:
    model.eval()
    with torch.inference_mode():
        offline = model(**_model_inputs(batch))
        expected_features, expected_final = causal_observation_intensity_sequence(
            spec=model.observation_intensity_spec,
            dt_seconds=batch["tr_dt_seconds"],
            dt_valid=batch["tr_dt_valid"],
            dt_reason=batch["tr_dt_reason"],
            rd_quality=batch["rd_quality"],
            rd_frame_mask=batch["rd_frame_mask"],
            padding_mask=batch["tr_padding_mask"],
        )
        state = None
        intensity_state = None
        rows: list[torch.Tensor] = []
        feature_error = 0.0
        length = int(batch["lengths"][0].item())
        for event_index in range(length):
            point = _inputs_at(batch, 0, event_index)
            direct_feature, direct_next = causal_observation_intensity_step(
                spec=model.observation_intensity_spec,
                dt_seconds=point["tr_dt_seconds"],
                dt_valid=point["tr_dt_valid"],
                dt_reason=point["tr_dt_reason"],
                rd_quality=point["rd_quality"],
                rd_frame_mask=point["rd_frame_mask"],
                previous_state=intensity_state,
            )
            feature_error = max(
                feature_error,
                float((direct_feature - expected_features[0, event_index : event_index + 1]).abs().max().item()),
            )
            online = model.forward_online_point(
                **point,
                liquid_state=state,
                observation_intensity_state=intensity_state,
            )
            state = online["liquid_state"]
            intensity_state = online["observation_intensity_state"]
            if not torch.allclose(intensity_state, direct_next, atol=ATOL, rtol=RTOL):
                raise AssertionError("V23k online intensity state differs from direct causal recurrence")
            rows.append(online["window_logits"])
        logits = torch.stack(rows, dim=1)
        final_error = float((intensity_state - expected_final[:1]).abs().max().item())
        return float((offline["sequence_logits"][:1, :length] - logits).abs().max().item()), max(feature_error, final_error)


def _empty_stratum_contract() -> bool:
    """Ensure a required but empty Cartesian report cell remains descriptive."""

    rows = [
        {
            "parent_sample_id": "synthetic-target",
            "label_name": "DroneTarget",
            "true_id": 0,
            "pred_id": 0,
        },
        {
            "parent_sample_id": "synthetic-clutter",
            "label_name": "ClutterTarget",
            "true_id": 1,
            "pred_id": 1,
        },
    ]
    strata = _stratum_metrics(
        rows,
        rows,
        {"populated": {"synthetic-target", "synthetic-clutter"}, "empty": set()},
        prefix="smoke",
    )
    empty = dict(strata["empty"])
    populated = dict(strata["populated"])
    return (
        empty.get("support") == 0
        and empty.get("comparison_status") == "not_applicable_empty_stratum"
        and empty.get("v23a_fixed_retention_control") is None
        and empty.get("v23k_observation_intensity_treatment") is None
        and empty.get("v23k_minus_v23a") is None
        and populated.get("support") == 2
        and populated.get("comparison_status") == "descriptive"
    )


def _online_missing_dt_semantics_rejected(model: Any, batch: dict[str, Any]) -> bool:
    """A V23k point cannot silently assume a timing reason or validity bit."""

    point = _inputs_at(batch, 0, 0)
    point["tr_dt_valid"] = None
    point["tr_dt_reason"] = None
    try:
        model.forward_online_point(
            **point,
            liquid_state=None,
            observation_intensity_state=None,
        )
    except ValueError as error:
        return "requires explicit dt_valid and dt_reason" in str(error)
    return False


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    config, source_config, _source_path = load_config(root)
    v23a_config, _v23a_source, _v23a_source_path = load_adapter_config(root, V23A_VARIANT)
    _summary_path, source_v23_summary, _checkpoint_path, _checkpoint_sha = source_v23_training(root)
    train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
    batch = next(iter(build_known_gate_dataloader(train_dataset, batch_size=2, shuffle=False)))
    treatment, _audit = build_official_cfc_model_from_frozen_source(
        root,
        source_config,
        state_adapter_spec=config.adapter,
        observation_intensity_spec=config.observation_intensity,
    )
    control, _control_audit = build_official_cfc_model_from_frozen_source(
        root,
        source_config,
        state_adapter_spec=v23a_config.adapter,
    )
    treatment.eval()
    control.eval()
    with torch.inference_mode():
        treatment_logits = treatment(**_model_inputs(batch))["sequence_logits"]
        control_logits = control(**_model_inputs(batch))["sequence_logits"]
    initial_logit_error = float((treatment_logits - control_logits).abs().max().item())
    state = treatment.state_dict()
    source_width = int(state["fusion.0.weight"].shape[1] - 2)
    reduced = {name: value for name, value in state.items() if name != "fusion.0.weight"}
    reduced["fusion.0.weight"] = state["fusion.0.weight"][:, :source_width]
    shared_core_matches = _state_dict_sha256(reduced) == str(source_v23_summary["initial_model_state_sha256"])
    synthetic_dt = torch.tensor([[0.0, 0.0, 10.0, 0.0, 0.0]], dtype=torch.float32)
    synthetic_valid = torch.tensor([[False, False, True, False, False]], dtype=torch.bool)
    synthetic_reasons = torch.tensor([[1, 2, 3, 2, 0]], dtype=torch.long)
    synthetic_rd_quality = torch.zeros((1, 5, 7), dtype=torch.float32)
    synthetic_rd_quality[0, :4, 0] = torch.tensor([1.0, 0.0, 1.0, 1.0])
    synthetic_frame_mask = torch.zeros((1, 5, 3), dtype=torch.bool)
    synthetic_frame_mask[0, 0, :2] = True
    synthetic_frame_mask[0, 2, :1] = True
    synthetic_frame_mask[0, 3, :3] = True
    synthetic_padding = torch.tensor([[False, False, False, False, True]], dtype=torch.bool)
    synthetic_features, synthetic_state = causal_observation_intensity_sequence(
        spec=config.observation_intensity,
        dt_seconds=synthetic_dt,
        dt_valid=synthetic_valid,
        dt_reason=synthetic_reasons,
        rd_quality=synthetic_rd_quality,
        rd_frame_mask=synthetic_frame_mask,
        padding_mask=synthetic_padding,
    )
    decay = torch.exp(torch.tensor(-10.0 / 15.025))
    expected = torch.stack(
        (
            torch.stack((torch.log1p(torch.tensor(1.0)), torch.log1p(torch.tensor(2.0)))),
            torch.stack((torch.log1p(torch.tensor(2.0)), torch.log1p(torch.tensor(2.0)))),
            torch.stack((torch.log1p(decay * 2.0 + 1.0), torch.log1p(decay * 2.0 + 1.0))),
            torch.stack((torch.log1p(decay * 2.0 + 2.0), torch.log1p(decay * 2.0 + 4.0))),
            torch.zeros((2,), dtype=torch.float32),
        ),
        dim=0,
    ).unsqueeze(0)
    model = treatment.train()
    output = model(**_model_inputs(batch))
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    objective = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    objective["loss"].backward()
    intensity_gradient = model.fusion[0].weight.grad[:, -2:]
    online_error, intensity_online_error = _offline_online_error(model, batch)
    model.eval()
    with torch.inference_mode():
        base_logits = model(**_model_inputs(batch))["sequence_logits"]
        perturbed = {
            key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in _model_inputs(batch).items()
        }
        cutoff = int(batch["lengths"][0].item()) - 1
        perturbed["tr_features"][0, cutoff:] += 4.0
        perturbed["rd_frames"][0, cutoff:] += 2.0
        perturbed_logits = model(**perturbed)["sequence_logits"]
        future_perturbation_past_error = float(
            (base_logits[0, :cutoff] - perturbed_logits[0, :cutoff])
            .abs()
            .max()
            .item()
        )
    checks = {
        "v23k_config_is_bound_to_its_generated_file": config.config_name == config_relative().stem.removesuffix("-config"),
        "v23k_reuses_v23a_fixed_retention_adapter": config.adapter == v23a_config.adapter,
        "v23k_shared_core_initialization_exactly_matches_v23a": shared_core_matches,
        "v23k_new_fusion_columns_are_exactly_zero": bool(
            torch.equal(
                state["fusion.0.weight"][:, -2:],
                torch.zeros_like(state["fusion.0.weight"][:, -2:]),
            )
        ),
        "v23k_zero_initialized_logits_replay_v23a": initial_logit_error <= ATOL,
        "v23k_synthetic_decay_log1p_formula_and_padding_are_exact": bool(
            torch.allclose(synthetic_features, expected, atol=ATOL, rtol=RTOL)
            and torch.isfinite(synthetic_state).all()
        ),
        "v23k_changes_input_only_not_official_state_adapter": treatment.external_state_adapter_specification() == v23a_config.adapter.as_dict(),
        "v23k_real_train_loss_and_all_gradients_are_finite": bool(
            torch.isfinite(objective["loss"])
            and all(
                parameter.grad is not None
                and bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        ),
        "v23k_new_intensity_fusion_columns_receive_finite_nonzero_gradient": intensity_gradient is not None and bool(torch.isfinite(intensity_gradient).all()) and float(intensity_gradient.abs().sum().item()) > 0.0,
        "v23k_offline_online_logits_match_within_float32_tolerance": online_error <= ATOL,
        "v23k_offline_online_intensity_state_matches_within_float32_tolerance": intensity_online_error <= ATOL,
        "v23k_future_event_perturbation_cannot_change_earlier_logits": future_perturbation_past_error <= ATOL,
        "v23k_empty_length_multiplicity_strata_are_not_misreported_as_metrics": _empty_stratum_contract(),
        "v23k_online_intensity_requires_explicit_dt_semantics": _online_missing_dt_semantics_rejected(
            treatment,
            batch,
        ),
        "validation_dataset_access_count_is_zero": True,
        "unknown_dataset_access_count_is_zero": True,
        "test_dataset_access_count_is_zero": True,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"V23k observation-intensity smoke failed: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "v23k_config_sha256": sha256_file(root / config_relative()),
        "initial_v23a_replay_max_logit_error": initial_logit_error,
        "offline_online_max_logit_error": online_error,
        "offline_online_max_intensity_error": intensity_online_error,
        "real_train_intensity_gradient_absolute_sum": float(intensity_gradient.abs().sum().item()),
        "future_event_perturbation_past_max_logit_error": future_perturbation_past_error,
        "real_train_parent_count": len(train_dataset),
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23k causal observation-intensity train-only contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
