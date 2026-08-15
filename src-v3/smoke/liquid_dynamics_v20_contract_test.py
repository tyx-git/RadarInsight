"""Focused contract smoke for the V20 zero-delta dynamics pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from radar_ln3.data import (
    DT_REASON_FIRST_EVENT,
    DT_REASON_PADDING,
    DT_REASON_POSITIVE_DELTA,
    DT_REASON_ZERO_DELTA,
)
from radar_ln3.task_models import (
    TaskSequenceDataset,
    build_paired_model_for_config,
    build_task_dataloader,
    build_task_dataset,
    load_task_config,
    task_training_objective,
)
from report.liquid_dynamics_v20_readiness import VARIANTS, _config_paths


def _mode_probe() -> dict[str, bool | float]:
    torch.manual_seed(42)
    control = __import__("radar_ln3.model", fromlist=["CfCLiquidCausalEncoder"]).CfCLiquidCausalEncoder(
        4, 4, invalid_dt_update_mode="bounded_observation"
    )
    skip = __import__("radar_ln3.model", fromlist=["CfCLiquidCausalEncoder"]).CfCLiquidCausalEncoder(
        4, 4, invalid_dt_update_mode="skip"
    )
    density = __import__("radar_ln3.model", fromlist=["CfCLiquidCausalEncoder"]).CfCLiquidCausalEncoder(
        4, 4, invalid_dt_update_mode="density_weighted"
    )
    skip.load_state_dict(control.state_dict())
    density.load_state_dict(control.state_dict())
    event_a = torch.randn(1, 4)
    event_b = event_a + 7.0
    state = torch.randn(1, 4)
    zero_dt = torch.zeros(1)
    invalid = torch.zeros(1, dtype=torch.bool)
    zero_reason = torch.full((1,), DT_REASON_ZERO_DELTA, dtype=torch.long)
    skip_a = skip._step(event_a, zero_dt, state, invalid, zero_reason)[0]
    skip_b = skip._step(event_b, zero_dt, state, invalid, zero_reason)[0]
    reasons = torch.tensor(
        [[
            DT_REASON_FIRST_EVENT,
            DT_REASON_ZERO_DELTA,
            DT_REASON_ZERO_DELTA,
            DT_REASON_POSITIVE_DELTA,
            DT_REASON_ZERO_DELTA,
        ]],
        dtype=torch.long,
    )
    dt = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]])
    valid = torch.tensor([[False, False, False, True, False]])
    padding = torch.zeros((1, 5), dtype=torch.bool)
    events = torch.randn(1, 5, 4)
    offline = density(
        events,
        dt,
        padding,
        dt_valid=valid,
        dt_reason=reasons,
    )
    state_online = None
    online_outputs: list[torch.Tensor] = []
    ordinal = 0
    for index in range(5):
        reason = int(reasons[0, index].item())
        if reason in {DT_REASON_FIRST_EVENT, DT_REASON_POSITIVE_DELTA}:
            ordinal = 1
        elif reason == DT_REASON_ZERO_DELTA:
            ordinal += 1
        state_online, _, _ = density.step(
            events[:, index],
            dt[:, index],
            state_online,
            valid[:, index],
            reasons[:, index],
            torch.tensor([ordinal], dtype=torch.long),
        )
        online_outputs.append(density.readout(state_online))
    online = torch.stack(online_outputs, dim=1)
    padded_reasons = torch.tensor(
        [
            [
                DT_REASON_FIRST_EVENT,
                DT_REASON_ZERO_DELTA,
                DT_REASON_POSITIVE_DELTA,
                DT_REASON_PADDING,
                DT_REASON_PADDING,
            ],
            [
                DT_REASON_FIRST_EVENT,
                DT_REASON_POSITIVE_DELTA,
                DT_REASON_PADDING,
                DT_REASON_PADDING,
                DT_REASON_PADDING,
            ],
        ],
        dtype=torch.long,
    )
    padded_events = torch.randn(2, 5, 4)
    padded_dt = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
    )
    padded_valid = torch.tensor(
        [[False, False, True, False, False], [False, True, False, False, False]]
    )
    padded_mask = torch.tensor(
        [[False, False, False, True, True], [False, False, True, True, True]]
    )
    padded_ordinal = torch.tensor(
        [[1, 2, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.long
    )
    padded_derived = density(
        padded_events,
        padded_dt,
        padded_mask,
        dt_valid=padded_valid,
        dt_reason=padded_reasons,
    )
    padded_explicit = density(
        padded_events,
        padded_dt,
        padded_mask,
        dt_valid=padded_valid,
        dt_reason=padded_reasons,
        zero_ordinal=padded_ordinal,
    )
    gain_control = __import__(
        "radar_ln3.model", fromlist=["CfCLiquidCausalEncoder"]
    ).CfCLiquidCausalEncoder(
        4,
        4,
        invalid_dt_update_mode="bounded_observation",
        state_normalization_mode="readout_only",
    )
    gain_density = __import__(
        "radar_ln3.model", fromlist=["CfCLiquidCausalEncoder"]
    ).CfCLiquidCausalEncoder(
        4,
        4,
        invalid_dt_update_mode="density_weighted",
        state_normalization_mode="readout_only",
    )
    gain_density.load_state_dict(gain_control.state_dict())
    gain_event = torch.randn(1, 4)
    gain_state = torch.randn(1, 4)
    control_k1 = gain_control._step(
        gain_event,
        zero_dt,
        gain_state,
        invalid,
        zero_reason,
        torch.ones(1, dtype=torch.long),
    )[0]
    density_states = {
        ordinal: gain_density._step(
            gain_event,
            zero_dt,
            gain_state,
            invalid,
            zero_reason,
            torch.full((1,), ordinal, dtype=torch.long),
        )[0]
        for ordinal in (1, 2, 3, 4)
    }
    control_increment = control_k1 - gain_state
    return {
        "skip_ignores_zero_event_content": bool(torch.equal(skip_a, skip_b)),
        "skip_recurrent_norm_is_explicitly_non_identity_allowed": bool(
            not torch.equal(skip_a, state)
        ),
        "density_offline_online_exact": bool(torch.equal(offline.sequence, online)),
        "density_finite": bool(torch.isfinite(offline.sequence).all()),
        "density_online_max_error": float((offline.sequence - online).abs().max().item()),
        "density_padded_explicit_ordinal_exact": bool(
            torch.equal(padded_derived.sequence, padded_explicit.sequence)
        ),
        "density_k1_exactly_matches_control_gain": bool(
            torch.equal(control_k1, density_states[1])
        ),
        "density_k2_increment_is_twice_control": bool(
            torch.allclose(
                density_states[2] - gain_state,
                2.0 * control_increment,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        ),
        "density_k3_increment_is_three_times_control": bool(
            torch.allclose(
                density_states[3] - gain_state,
                3.0 * control_increment,
                atol=1.0e-6,
                rtol=1.0e-6,
            )
        ),
        "density_k3plus_gain_is_capped": bool(
            torch.equal(density_states[3], density_states[4])
        ),
        "density_gain_probe_is_non_degenerate": bool(
            control_increment.abs().max().item() > 1.0e-6
        ),
    }


def run(
    project_root: Path,
    *,
    variant: str = "skip",
    prepared: bool = False,
) -> dict[str, object]:
    root = project_root.resolve()
    if variant not in VARIANTS:
        raise ValueError(f"unsupported V20 smoke variant: {variant}")
    mode = _mode_probe()
    config_path = root / _config_paths(variant)["control"]
    config = load_task_config(config_path)
    control, treatment, pairing = build_paired_model_for_config(root, config)
    dataset = build_task_dataset(root, config, split_name=config.data.train_split_name)
    batch = next(iter(build_task_dataloader(dataset, batch_size=2, shuffle=False)))
    keys = (
        "tr_features",
        "tr_missing_mask",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_dt_reason",
        "tr_motion_features",
        "tr_motion_valid_mask",
        "rd_physical_features",
        "rd_physical_valid",
        "micro_doppler_features",
        "micro_doppler_valid",
        "rd_frames",
        "rd_frame_mask",
        "rd_quality",
        "tr_padding_mask",
    )
    tensors = {key: batch[key] for key in keys if key in batch}
    control.eval()
    treatment.eval()
    with torch.inference_mode():
        control_logits = control(**tensors)["sequence_logits"]
        treatment_logits = treatment(**tensors)["sequence_logits"]
    active = ~tensors["tr_padding_mask"]
    exposure_seen = torch.cumsum(
        (tensors["tr_dt_reason"] == DT_REASON_ZERO_DELTA).to(dtype=torch.int64),
        dim=1,
    ).gt(0)
    pre_exposure = active & (~exposure_seen)
    if not bool(pre_exposure.any()):
        raise RuntimeError("V20 smoke has no pre-exposure event")
    actual = {
        "initial_full_logits_exact_diagnostic": bool(
            torch.equal(control_logits, treatment_logits)
        ),
        "initial_logits_max_error": float((control_logits - treatment_logits).abs().max().item()),
        "initial_pre_exposure_logits_exact": bool(
            torch.equal(control_logits[pre_exposure], treatment_logits[pre_exposure])
        ),
        "initial_treatment_exposed_event_count": int((active & exposure_seen).sum().item()),
        "train_parent_count": len(dataset),
        "shared_state_entry_count": pairing["shared_state_entry_count"],
        "control_mode": control.liquid.invalid_dt_update_mode,
        "treatment_mode": treatment.liquid.invalid_dt_update_mode,
        "initial_control_logits_sha256": hashlib.sha256(
            control_logits.detach().cpu().contiguous().numpy().tobytes(order="C")
        ).hexdigest(),
    }
    exposed_parent_id = next(
        parent_id
        for parent_id in dataset.parent_ids
        if dataset.base._groups[parent_id]["tr_dt_reason"]
        .astype(str)
        .eq("zero_delta_at_second_precision")
        .any()
    )
    exposed_dataset = TaskSequenceDataset(
        dataset.base, dataset.spec, parent_ids=[exposed_parent_id]
    )
    exposed_batch = next(
        iter(build_task_dataloader(exposed_dataset, batch_size=1, shuffle=False))
    )
    exposed_tensors = {
        key: exposed_batch[key] for key in keys if key in exposed_batch
    }
    control.eval()
    treatment.train()
    control_exposed = control(**exposed_tensors)["sequence_logits"]
    treatment_output = treatment(**exposed_tensors)
    treatment_exposed = treatment_output["sequence_logits"]
    exposed_active = ~exposed_tensors["tr_padding_mask"]
    exposed_seen = torch.cumsum(
        (
            exposed_tensors["tr_dt_reason"] == DT_REASON_ZERO_DELTA
        ).to(dtype=torch.int64),
        dim=1,
    ).gt(0)
    exposed_pre = exposed_active & (~exposed_seen)
    exposed_after = exposed_active & exposed_seen
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    objective = task_training_objective(
        treatment_output,
        exposed_batch,
        config,
        class_weights=weights,
    )
    treatment.zero_grad(set_to_none=True)
    objective["loss"].backward()
    finite_gradients = [
        parameter.grad
        for parameter in treatment.parameters()
        if parameter.grad is not None
    ]
    actual.update(
        {
            "real_exposed_parent_id": exposed_parent_id,
            "real_exposed_parent_pre_treatment_logits_exact": bool(
                torch.equal(
                    control_exposed[exposed_pre], treatment_exposed[exposed_pre]
                )
            ),
            "real_exposed_parent_post_treatment_logits_differ": bool(
                exposed_after.any()
                and not torch.equal(
                    control_exposed[exposed_after], treatment_exposed[exposed_after]
                )
            ),
            "real_exposed_parent_loss_and_gradients_finite": bool(
                torch.isfinite(objective["loss"]).item()
                and finite_gradients
                and all(torch.isfinite(value).all().item() for value in finite_gradients)
            ),
            "real_exposed_parent_has_nonzero_gradient": bool(
                finite_gradients
                and max(float(value.abs().max().item()) for value in finite_gradients)
                > 0.0
            ),
        }
    )
    prepared_contract: dict[str, object] | str = "not_requested"
    if prepared:
        report_path = root / (
            "src-v3/processed/report/"
            f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-training-readiness.json"
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "passed"
            or report.get("schema_version")
            != f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-training-readiness-v1"
            or report.get("paired_protocol") != VARIANTS[variant]["protocol"]
            or report.get("test") != "未执行"
        ):
            raise ValueError("V20 readiness is not passed/test-sealed")
        prepared_contract = {
            "status": report["status"],
            "schema_version": report["schema_version"],
            "report_path": str(report_path),
            "test_payload_access_count": report["test_payload_access_count"],
        }
        if variant == "density":
            shared_summary_path = Path(
                str(dict(report.get("shared_control_reuse") or {}).get("summary", ""))
            )
            shared_summary = json.loads(shared_summary_path.read_text(encoding="utf-8"))
            actual["shared_control_initial_logits_sha256_matches"] = bool(
                actual["initial_control_logits_sha256"]
                == dict(shared_summary.get("paired_initialization") or {}).get(
                    "initial_logits_sha256"
                )
            )
    checks = {
        **mode,
        **actual,
        "paired_initialization_exact_before_treatment_exposure": actual[
            "initial_pre_exposure_logits_exact"
        ],
        "control_mode_is_bounded_observation": actual["control_mode"]
        == "bounded_observation",
        "treatment_mode_matches_variant": actual["treatment_mode"]
        == ("skip" if variant == "skip" else "density_weighted"),
        "prepared_readiness_passed": (
            prepared_contract != "not_requested"
            and prepared_contract["status"] == "passed"
            if isinstance(prepared_contract, dict)
            else True
        ),
    }
    passed = all(
        bool(value)
        for key, value in checks.items()
        if key
        not in {
            "density_online_max_error",
            "initial_logits_max_error",
            "initial_full_logits_exact_diagnostic",
            "initial_treatment_exposed_event_count",
        }
    ) and float(mode["density_online_max_error"]) <= 1.0e-6
    result = {
        "status": "passed" if passed else "failed",
        "schema_version": f"RADAR-LN3-V20-{VARIANTS[variant]['slug']}-smoke-v1",
        "experiment": VARIANTS[variant]["label"],
        "checks": checks,
        "prepared_contract": prepared_contract,
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_training": "未执行",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("V20 focused contract smoke failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused V20/V20r dynamics contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="skip")
    parser.add_argument("--prepared", action="store_true")
    args = parser.parse_args()
    run(args.project_root, variant=args.variant, prepared=args.prepared)


if __name__ == "__main__":
    main()
