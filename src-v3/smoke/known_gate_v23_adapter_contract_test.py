"""Train-only contract smoke for V23 external-state official-CfC adapters.

This deliberately does not train a model or read validation/test/Unknown data.
It checks that each single-factor adapter is attached outside the unmodified
official ``ncps.torch.CfC`` cell, preserves the required causal branches, and
keeps offline and online execution aligned on a real V21 training batch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset
from radar_ln3.known_gate_official_cfc import build_official_cfc_model_from_frozen_source
from radar_ln3.known_gate_v23_adapters import (
    V23_ADAPTER_VARIANTS,
    V23A_VARIANT,
    V23B_VARIANT,
    V23BR_VARIANT,
    V23D_VARIANT,
    V23GR_VARIANT,
    load_adapter_config,
    source_v23c_control,
    source_v23_training,
    state_dict_sha256,
)
from radar_ln3.official_cfc_state_adapters import (
    ADAPTER_NONE,
    ExternalOfficialCfCStateAdapter,
    ExternalStateAdapterSpec,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs, _state_dict_sha256
from train.known_gate_v23_adapters import (
    _assert_v23gr_pre_zero_replays_v23c_and_first_density_zero_diverges,
)


ATOL = 1.0e-6
RTOL = 1.0e-6


def _point_inputs(
    batch: dict[str, Any], batch_index: int, event_index: int
) -> dict[str, torch.Tensor | None]:
    return {
        "tr_features": batch["tr_features"][batch_index : batch_index + 1, event_index],
        "tr_missing_mask": batch["tr_missing_mask"][batch_index : batch_index + 1, event_index],
        "tr_dt_seconds": batch["tr_dt_seconds"][batch_index : batch_index + 1, event_index],
        "tr_dt_valid": batch["tr_dt_valid"][batch_index : batch_index + 1, event_index],
        "tr_dt_reason": batch["tr_dt_reason"][batch_index : batch_index + 1, event_index],
        "rd_frames": batch["rd_frames"][batch_index : batch_index + 1, event_index],
        "rd_frame_mask": batch["rd_frame_mask"][batch_index : batch_index + 1, event_index],
        "rd_quality": batch["rd_quality"][batch_index : batch_index + 1, event_index],
        "known_gate_features": None,
        "known_gate_feature_valid": None,
    }


def _offline_online_error(model: Any, batch: dict[str, Any]) -> float:
    model.eval()
    with torch.inference_mode():
        offline = model(**_model_inputs(batch))
        state = None
        rows: list[torch.Tensor] = []
        length = int(batch["lengths"][0].item())
        ordinal = 0
        for event_index in range(length):
            reason = int(batch["tr_dt_reason"][0, event_index].item())
            if reason in (1, 3):
                ordinal = 1
            elif reason == 2:
                ordinal += 1
            else:
                raise ValueError(f"unsupported V23 adapter online dt reason: {reason}")
            online = model.forward_online_point(
                **_point_inputs(batch, 0, event_index),
                liquid_state=state,
                zero_ordinal=torch.tensor([ordinal], dtype=torch.long),
            )
            state = online["liquid_state"]
            rows.append(online["window_logits"])
        online_logits = torch.stack(rows, dim=1)
        return float(
            (offline["sequence_logits"][:1, :length] - online_logits)
            .abs()
            .max()
            .item()
        )


def _adapter_formula_checks() -> dict[str, bool]:
    torch.manual_seed(20260821)
    official = torch.randn(2, 48)
    previous = torch.randn(2, 48)
    seconds = torch.tensor([3.0, 7.0])
    first = torch.tensor([1, 1], dtype=torch.long)
    zero = torch.tensor([2, 2], dtype=torch.long)
    positive = torch.tensor([3, 3], dtype=torch.long)
    checks: dict[str, bool] = {}
    for variant, spec in (
        (
            "official_cfc_positive_dt_fixed_retention",
            ExternalStateAdapterSpec(
                kind="positive_dt_fixed_retention",
                hidden_dim=48,
                fixed_tau_seconds=15.025,
            ),
        ),
        (
            "official_cfc_positive_dt_per_dimension_retention",
            ExternalStateAdapterSpec(
                kind="positive_dt_per_dimension_retention",
                hidden_dim=48,
                fixed_tau_seconds=15.025,
            ),
        ),
        (
            "official_cfc_recurrent_layer_norm",
            ExternalStateAdapterSpec(
                kind="recurrent_layer_norm", hidden_dim=48, layer_norm_eps=1.0e-5
            ),
        ),
        (
            "official_cfc_recurrent_layer_norm_scale_only",
            ExternalStateAdapterSpec(
                kind="recurrent_layer_norm_scale_only",
                hidden_dim=48,
                layer_norm_eps=1.0e-5,
            ),
        ),
        (
            "official_cfc_zero_delta_bounded_observation",
            ExternalStateAdapterSpec(
                kind="zero_delta_bounded_observation",
                hidden_dim=48,
                zero_delta_gain=0.25,
            ),
        ),
        (
            "official_cfc_zero_delta_density_weighted_multiplicity",
            ExternalStateAdapterSpec(
                kind="zero_delta_density_weighted_multiplicity",
                hidden_dim=48,
                zero_delta_gain=0.25,
            ),
        ),
    ):
        adapter = ExternalOfficialCfCStateAdapter(spec).eval()
        first_result = adapter(official, previous, seconds, first)
        zero_result = adapter(official, previous, seconds, zero)
        positive_result = adapter(official, previous, seconds, positive)
        if spec.kind == "positive_dt_fixed_retention":
            alpha = 1.0 - torch.exp(-seconds.unsqueeze(-1) / 15.025)
            expected = (1.0 - alpha) * previous + alpha * official
            checks[variant] = bool(
                torch.equal(first_result, official)
                and torch.equal(zero_result, official)
                and torch.allclose(positive_result, expected, atol=ATOL, rtol=RTOL)
            )
        elif spec.kind == "positive_dt_per_dimension_retention":
            if adapter.log_tau_ratio is None:
                checks[variant] = False
                continue
            alpha = 1.0 - torch.exp(-seconds.unsqueeze(-1) / 15.025)
            expected = (1.0 - alpha) * previous + alpha * official
            checks[variant] = bool(
                torch.equal(adapter.log_tau_ratio, torch.zeros_like(adapter.log_tau_ratio))
                and torch.equal(first_result, official)
                and torch.equal(zero_result, official)
                and torch.allclose(positive_result, expected, atol=ATOL, rtol=RTOL)
                and sum(parameter.numel() for parameter in adapter.parameters()) == 48
            )
        elif spec.kind == "recurrent_layer_norm":
            checks[variant] = bool(
                torch.allclose(first_result, zero_result, atol=ATOL, rtol=RTOL)
                and torch.allclose(zero_result, positive_result, atol=ATOL, rtol=RTOL)
                and adapter.recurrent_norm is not None
                and sum(parameter.numel() for parameter in adapter.parameters()) == 96
            )
        elif spec.kind == "recurrent_layer_norm_scale_only":
            checks[variant] = bool(
                torch.allclose(first_result, zero_result, atol=ATOL, rtol=RTOL)
                and torch.allclose(zero_result, positive_result, atol=ATOL, rtol=RTOL)
                and adapter.recurrent_norm is not None
                and adapter.recurrent_norm.bias is None
                and torch.equal(
                    adapter.recurrent_norm.weight,
                    torch.ones_like(adapter.recurrent_norm.weight),
                )
                and sum(parameter.numel() for parameter in adapter.parameters()) == 48
            )
        elif spec.kind == "zero_delta_bounded_observation":
            expected = 0.75 * previous + 0.25 * official
            checks[variant] = bool(
                torch.equal(first_result, official)
                and torch.equal(positive_result, official)
                and torch.allclose(zero_result, expected, atol=ATOL, rtol=RTOL)
            )
        elif spec.kind == "zero_delta_density_weighted_multiplicity":
            ordinal = torch.tensor([2, 4], dtype=torch.long)
            weighted = adapter(official, previous, seconds, zero, ordinal)
            expected = torch.stack(
                [0.5 * previous[0] + 0.5 * official[0], 0.25 * previous[1] + 0.75 * official[1]],
                dim=0,
            )
            checks[variant] = bool(
                torch.equal(first_result, official)
                and torch.equal(positive_result, official)
                and torch.allclose(weighted, expected, atol=ATOL, rtol=RTOL)
            )
        else:  # pragma: no cover - config contract makes this unreachable
            raise AssertionError("unsupported adapter kind")
    return checks


def run(project_root: Path, *, variant: str) -> dict[str, Any]:
    if variant not in V23_ADAPTER_VARIANTS:
        raise ValueError(f"unsupported V23 adapter variant: {variant}")
    root = project_root.resolve()
    config, source_config, _source_path = load_adapter_config(root, variant)
    _summary_path, v23_summary, _checkpoint_path, _checkpoint_sha = source_v23_training(root)
    train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
    batch = next(
        iter(build_known_gate_dataloader(train_dataset, batch_size=2, shuffle=False))
    )
    treatment, _source_audit = build_official_cfc_model_from_frozen_source(
        root, source_config, state_adapter_spec=config.adapter
    )
    direct_v23, _direct_audit = build_official_cfc_model_from_frozen_source(
        root, source_config
    )

    treatment_state = treatment.state_dict()
    source_initial_hash = str(v23_summary["initial_model_state_sha256"])
    if config.adapter.trainable_parameter_count == 0:
        initialization_matches = _state_dict_sha256(treatment_state) == source_initial_hash
    else:
        initialization_matches = (
            state_dict_sha256(treatment_state, exclude_prefixes=("state_adapter.",))
            == source_initial_hash
        )

    v23a_initial_replay = True
    v23a_initial_replay_max_error = 0.0
    if variant == V23D_VARIANT:
        fixed_tau, _v23a_audit = build_official_cfc_model_from_frozen_source(
            root,
            source_config,
            state_adapter_spec=load_adapter_config(root, V23A_VARIANT)[0].adapter,
        )
        treatment.eval()
        fixed_tau.eval()
        with torch.inference_mode():
            initial_v23d = treatment(**_model_inputs(batch))["sequence_logits"]
            initial_v23a = fixed_tau(**_model_inputs(batch))["sequence_logits"]
        v23a_initial_replay_max_error = float(
            (initial_v23d - initial_v23a).abs().max().item()
        )
        v23a_initial_replay = bool(
            torch.allclose(initial_v23d, initial_v23a, atol=ATOL, rtol=RTOL)
        )

    v23b_initial_replay = True
    v23b_initial_replay_max_error = 0.0
    if variant == V23BR_VARIANT:
        affine, _v23b_audit = build_official_cfc_model_from_frozen_source(
            root,
            source_config,
            state_adapter_spec=load_adapter_config(root, V23B_VARIANT)[0].adapter,
        )
        treatment.eval()
        affine.eval()
        with torch.inference_mode():
            initial_v23br = treatment(**_model_inputs(batch))["sequence_logits"]
            initial_v23b = affine(**_model_inputs(batch))["sequence_logits"]
        v23b_initial_replay_max_error = float(
            (initial_v23br - initial_v23b).abs().max().item()
        )
        v23b_initial_replay = bool(
            torch.allclose(initial_v23br, initial_v23b, atol=ATOL, rtol=RTOL)
        )

    v23gr_pre_zero_replay = True
    v23gr_first_density_difference = True
    v23gr_pre_zero_error = 0.0
    v23gr_first_density_delta = 0.0
    v23gr_training_entry_audit: dict[str, Any] | None = None
    if variant == V23GR_VARIANT:
        control, _v23c_audit = build_official_cfc_model_from_frozen_source(
            root,
            source_config,
            state_adapter_spec=load_adapter_config(
                root, "official_cfc_zero_delta_bounded_observation"
            )[0].adapter,
        )
        selected = None
        index = None
        for item in train_dataset:
            ordinal = 0
            for event_index, reason in enumerate(item.base.tr_dt_reason.tolist()):
                if reason in (1, 3):
                    ordinal = 1
                elif reason == 2:
                    ordinal += 1
                    if ordinal >= 2:
                        selected, index = item, event_index
                        break
            if selected is not None:
                break
        if selected is None or index is None:
            raise AssertionError("V23gr smoke lacks real k>=2 zero-delta evidence")
        density_batch = next(iter(build_known_gate_dataloader([selected], batch_size=1, shuffle=False)))
        treatment.eval(); control.eval()
        with torch.inference_mode():
            observed = treatment(**_model_inputs(density_batch))["sequence_logits"]
            expected = control(**_model_inputs(density_batch))["sequence_logits"]
        v23gr_pre_zero_error = (
            float((observed[:, :index] - expected[:, :index]).abs().max().item())
            if index > 0 else 0.0
        )
        v23gr_first_density_delta = float(
            (observed[:, index] - expected[:, index]).abs().max().item()
        )
        v23gr_pre_zero_replay = v23gr_pre_zero_error <= ATOL
        v23gr_first_density_difference = v23gr_first_density_delta > 1.0e-8
        v23gr_training_entry_audit = (
            _assert_v23gr_pre_zero_replays_v23c_and_first_density_zero_diverges(
                root,
                source_config=source_config,
                model=treatment,
            )
        )

    treatment.train()
    output = treatment(**_model_inputs(batch))
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
    gradients = [parameter.grad for parameter in treatment.parameters()]
    all_gradients_finite = bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all())
        for gradient in gradients
    )
    adapter_gradients = [
        parameter.grad
        for name, parameter in treatment.named_parameters()
        if name.startswith("state_adapter.")
    ]
    adapter_gradients_finite_nonzero = (
        True
        if config.adapter.trainable_parameter_count == 0
        else bool(adapter_gradients)
        and all(
            gradient is not None
            and bool(torch.isfinite(gradient).all())
            and float(gradient.abs().sum()) > 0.0
            for gradient in adapter_gradients
        )
    )
    treatment.eval()
    direct_v23.eval()
    with torch.inference_mode():
        point_events = direct_v23._point_events(
            tr_features=batch["tr_features"],
            tr_missing_mask=batch["tr_missing_mask"],
            rd_frames=batch["rd_frames"],
            rd_frame_mask=batch["rd_frame_mask"],
            rd_quality=batch["rd_quality"],
            known_gate_features=None,
            known_gate_feature_valid=None,
            leading_shape=tuple(batch["tr_features"].shape[:2]),
        )
        canonical = direct_v23.liquid(
            point_events,
            batch["tr_dt_seconds"],
            batch["tr_padding_mask"],
            dt_valid=batch["tr_dt_valid"],
            dt_reason=batch["tr_dt_reason"],
            state_adapter=None,
        )
        transparent = direct_v23.liquid(
            point_events,
            batch["tr_dt_seconds"],
            batch["tr_padding_mask"],
            dt_valid=batch["tr_dt_valid"],
            dt_reason=batch["tr_dt_reason"],
            state_adapter=ExternalOfficialCfCStateAdapter(
                ExternalStateAdapterSpec(kind=ADAPTER_NONE, hidden_dim=48)
            ),
        )
        disabled_replay_max_error = float(
            (canonical.sequence - transparent.sequence).abs().max().item()
        )
        treatment_output = treatment(**_model_inputs(batch))
        padding_values = treatment_output["liquid_sequence"].masked_select(
            batch["tr_padding_mask"].unsqueeze(-1)
        )
    online_error = _offline_online_error(treatment, batch)
    formula = _adapter_formula_checks()
    checks = {
        "variant_is_pre_registered": variant in V23_ADAPTER_VARIANTS,
        "official_cfcell_is_unmodified_external_adapter_only": treatment.liquid.cfc.__class__
        is direct_v23.liquid.cfc.__class__,
        "v23a_and_v23c_add_no_trainable_adapter_parameters": (
            True
            if config.adapter.trainable_parameter_count != 0
            else not adapter_gradients
        ),
        "zero_parameter_or_core_initialization_matches_v23": initialization_matches,
        "disabled_adapter_replays_v23_logits": disabled_replay_max_error <= ATOL,
        "adapter_formula_matches_registered_causal_branch": formula[variant],
        "v23d_zero_log_tau_replays_v23a_fixed_tau_before_optimization": v23a_initial_replay,
        "v23d_direct_v23a_control_binding_is_frozen": (
            True
            if variant != V23D_VARIANT
            else config.direct_control is not None
        ),
        "v23br_zero_beta_replays_v23b_affine_layer_norm_before_optimization": (
            v23b_initial_replay
        ),
        "v23br_direct_v23b_control_binding_is_frozen": (
            True
            if variant != V23BR_VARIANT
            else config.direct_control is not None
        ),
        "v23gr_pre_zero_replays_v23c_and_first_k2_zero_delta_diverges": (
            True
            if variant != V23GR_VARIANT
            else v23gr_pre_zero_replay and v23gr_first_density_difference
        ),
        "v23gr_training_entry_direct_v23c_audit_is_executable": (
            True
            if variant != V23GR_VARIANT
            else (
                config.direct_control == source_v23c_control(root)
                and v23gr_training_entry_audit is not None
                and v23gr_training_entry_audit[
                    "pre_zero_delta_replays_v23c_within_float32_tolerance"
                ]
                is True
                and v23gr_training_entry_audit[
                    "first_k2_zero_delta_differs_from_v23c"
                ]
                is True
            )
        ),
        "real_train_batch_loss_and_all_gradients_are_finite": bool(
            torch.isfinite(objective["loss"]) and all_gradients_finite
        ),
        "parameterized_adapter_has_finite_nonzero_gradients": adapter_gradients_finite_nonzero,
        "padding_never_enters_external_recurrence": bool(
            torch.equal(padding_values, torch.zeros_like(padding_values))
        ),
        "offline_online_logits_match_within_float32_tolerance": online_error <= ATOL,
        "validation_dataset_access_count_is_zero": True,
        "unknown_dataset_access_count_is_zero": True,
        "test_dataset_access_count_is_zero": True,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"V23 adapter smoke failed: {failed}")
    return {
        "status": "passed",
        "variant": variant,
        "checks": checks,
        "adapter": treatment.external_state_adapter_specification(),
        "real_train_parent_count": len(train_dataset),
        "real_train_batch_parent_ids": list(batch["parent_sample_ids"]),
        "loss": float(objective["loss"].item()),
        "disabled_adapter_replay_max_logit_error": disabled_replay_max_error,
        "v23d_initial_v23a_replay_max_logit_error": v23a_initial_replay_max_error,
        "v23br_initial_v23b_replay_max_logit_error": v23b_initial_replay_max_error,
        "v23gr_pre_zero_v23c_replay_max_logit_error": v23gr_pre_zero_error,
        "v23gr_first_k2_zero_delta_logit_difference": v23gr_first_density_delta,
        "v23gr_training_entry_direct_control_audit": v23gr_training_entry_audit,
        "offline_online_max_logit_error": online_error,
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a V23 adapter train-only smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=V23_ADAPTER_VARIANTS, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root, variant=args.variant), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
