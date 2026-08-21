"""Train-only smoke for the V23 pure official dense-CfC baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset
from radar_ln3.known_gate_official_cfc import build_official_cfc_model_from_frozen_source
from radar_ln3.known_gate_v23 import CONFIG_RELATIVE, load_v23_config
from radar_ln3.official_cfc import (
    OFFICIAL_CFC_CLASS_MODULE,
    OFFICIAL_CFC_DIRECT_URL,
    OFFICIAL_CFC_DISTRIBUTION_VERSION,
    OFFICIAL_CFC_GIT_COMMIT,
    OFFICIAL_CFC_STATE_EQUATION_OWNER,
    official_cfc_runtime_binding,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs


def _inputs_at(batch: dict[str, object], batch_index: int, event_index: int) -> dict[str, torch.Tensor | None]:
    return {
        "tr_features": batch["tr_features"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "tr_missing_mask": batch["tr_missing_mask"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "tr_dt_seconds": batch["tr_dt_seconds"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "tr_dt_valid": batch["tr_dt_valid"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "tr_dt_reason": batch["tr_dt_reason"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "rd_frames": batch["rd_frames"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "rd_frame_mask": batch["rd_frame_mask"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "rd_quality": batch["rd_quality"][batch_index : batch_index + 1, event_index],  # type: ignore[index]
        "known_gate_features": None,
        "known_gate_feature_valid": None,
    }


def _single_parent_batch(dataset: object, parent_index: int) -> dict[str, object]:
    """Load one parent from the already-constructed train Dataset.

    This deliberately reuses the Dataset's index, group map and tensor cache.
    Reconstructing a Dataset once per parent caused the former smoke to scan the
    manifest and rebuild cache metadata hundreds of times, which is unrelated to
    the official-CfC contract under test.
    """

    item = dataset[parent_index]  # type: ignore[index]
    return next(iter(build_known_gate_dataloader([item], batch_size=1, shuffle=False)))


def _locate_train_reason_parents(dataset: object) -> tuple[dict[int, int], int]:
    """Find real training parents covering every dt reason and one long dt.

    We inspect each parent at most once and return early as soon as all required
    evidence is available.  No validation/test Dataset is constructed.
    """

    found: dict[int, int] = {}
    long_index: int | None = None
    for parent_index in range(len(dataset)):  # type: ignore[arg-type]
        item = dataset[parent_index]  # type: ignore[index]
        reasons = item.base.tr_dt_reason
        for reason_id in (1, 2, 3):
            if reason_id not in found and bool((reasons == reason_id).any()):
                found[reason_id] = parent_index
        if long_index is None and bool((item.base.tr_dt_seconds > 120.0).any()):
            long_index = parent_index
        if len(found) == 3 and long_index is not None:
            return found, long_index
    missing = [str(reason_id) for reason_id in (1, 2, 3) if reason_id not in found]
    if long_index is None:
        missing.append("positive_dt_gt_120")
    raise ValueError("V23 smoke cannot find required train dt evidence: " + ", ".join(missing))


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    runtime = official_cfc_runtime_binding()
    config, source_config, _source_path = load_v23_config(root, root / CONFIG_RELATIVE)
    dataset = build_known_gate_dataset(root, source_config, split_name="train")
    batch = next(iter(build_known_gate_dataloader(dataset, batch_size=2, shuffle=False)))
    reason_parents, long_parent_index = _locate_train_reason_parents(dataset)
    if "known_gate_features" in batch or "known_gate_feature_valid" in batch:
        raise AssertionError("V23 pure baseline batch unexpectedly exposes appended evidence")
    model, initialization_audit = build_official_cfc_model_from_frozen_source(root, source_config)
    model.train()
    output = model(**_model_inputs(batch))
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    objective = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"], batch["labels"], batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    objective["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    all_gradients_finite_nonzero = bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0.0
        for gradient in gradients
    )
    cfc_gradients = [
        parameter.grad for name, parameter in model.named_parameters() if name.startswith("liquid.cfc.")
    ]
    cfc_gradients_finite_nonzero = bool(cfc_gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all()) and float(gradient.abs().sum()) > 0.0
        for gradient in cfc_gradients
    )
    model.eval()
    with torch.inference_mode():
        offline = model(**_model_inputs(batch))
        state = None
        online_rows: list[torch.Tensor] = []
        length = int(batch["lengths"][0].item())
        for event_index in range(length):
            online = model.forward_online_point(**_inputs_at(batch, 0, event_index), liquid_state=state)
            state = online["liquid_state"]
            online_rows.append(online["window_logits"])
        online_logits = torch.stack(online_rows, dim=1)
        offline_logits = offline["sequence_logits"][:1, :length]
        max_online_error = float((offline_logits - online_logits).abs().max().item())
        final_state_matches = all(
            bool(torch.equal(
                offline["liquid_final_state"][index],
                offline["liquid_sequence"][index, int(batch["lengths"][index].item()) - 1],
            ))
            for index in range(int(batch["lengths"].shape[0]))
        )
        reason_exercised: dict[str, bool] = {}
        for name, reason_id in (("first", 1), ("zero_delta", 2), ("positive_delta", 3)):
            reason_batch = _single_parent_batch(dataset, reason_parents[reason_id])
            active = ~reason_batch["tr_padding_mask"]
            positions = ((reason_batch["tr_dt_reason"] == reason_id) & active).nonzero(as_tuple=False)
            if positions.numel() == 0:
                raise AssertionError(f"V23 smoke missing {name} dt evidence")
            row, event = (int(value) for value in positions[0].tolist())
            mutated = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in _model_inputs(reason_batch).items()}
            if name == "positive_delta":
                mutated["tr_dt_seconds"][row, event] *= 2.0  # type: ignore[index]
                changed = model(**mutated)["sequence_logits"]
                baseline = model(**_model_inputs(reason_batch))["sequence_logits"]
                reason_exercised[name] = float((changed - baseline).abs().max()) > 0.0
            else:
                mutated["tr_dt_seconds"][row, event] = 1.0  # type: ignore[index]
                try:
                    model(**mutated)
                except ValueError:
                    reason_exercised[name] = True
                else:
                    reason_exercised[name] = False
        long_batch = _single_parent_batch(dataset, long_parent_index)
        long_row = 0
        long_event = int((long_batch["tr_dt_seconds"][long_row] > 120.0).nonzero(as_tuple=False)[0].item())
        long_output = model(**_model_inputs(long_batch))
        actual_long_dt = float(long_batch["tr_dt_seconds"][long_row, long_event].item())
        long_timespan = model.liquid._timespans(
            long_batch["tr_dt_seconds"], ~long_batch["tr_padding_mask"]
        )[0][long_row, long_event].item()
    spec = model.backbone_specification()
    checks = {
        "official_ncps_distribution_version_is_pinned": runtime["distribution_version"] == OFFICIAL_CFC_DISTRIBUTION_VERSION,
        "official_ncps_git_commit_is_pinned": runtime["git_commit"] == OFFICIAL_CFC_GIT_COMMIT,
        "official_ncps_direct_url_is_pinned": runtime["direct_url"] == OFFICIAL_CFC_DIRECT_URL,
        "official_cfc_module_is_upstream": model.liquid.cfc.__class__.__module__ == OFFICIAL_CFC_CLASS_MODULE,
        "official_cfc_48x1_default_no_memory_projection_or_wiring": (
            spec["input_dim"] == 48 and spec["hidden_dim"] == 48 and spec["backbone_units"] == 48
            and spec["backbone_layers"] == 1 and spec["backbone_dropout"] == 0.0
            and spec["mixed_memory"] is False and spec["proj_size"] is None and spec["wiring"] is None
        ),
        "official_cfc_parameter_count_is_14064": spec["parameter_count"] == 14064,
        "source_and_v23_use_no_evidence_batch_weight_sum_protocol": (
            source_config.model.evidence_dim == 0 and config.train.class_weight_normalization == "batch_weight_sum"
        ),
        "legacy_tau_alpha_floor_gain_norm_candidate_gate_and_dt_cap_are_absent": (
            not any(dict(spec["project_legacy_recurrent_mechanisms"]).values())
            and "without_cap" in spec["dt_semantics"]
        ),
        "real_train_batch_has_no_appended_evidence": True,
        "real_batch_dt_reason_contract_is_exercised": all(reason_exercised.values()),
        "positive_dt_above_120_is_not_capped": actual_long_dt > 120.0 and abs(long_timespan - actual_long_dt) <= 1.0e-6,
        "real_train_batch_forward_is_finite": bool(torch.isfinite(output["sequence_logits"]).all() and torch.isfinite(objective["loss"])),
        "all_trainable_gradients_are_finite_and_nonzero": all_gradients_finite_nonzero,
        "official_cfc_gradients_are_finite_and_nonzero": cfc_gradients_finite_nonzero,
        "offline_online_logits_match_within_float32_tolerance": bool(torch.allclose(offline_logits, online_logits, atol=1.0e-6, rtol=1.0e-6)),
        "final_state_matches_final_causal_state": final_state_matches,
        "padding_never_reaches_official_recurrence": bool(torch.equal(
            offline["liquid_sequence"].masked_select(batch["tr_padding_mask"].unsqueeze(-1)),
            torch.zeros_like(offline["liquid_sequence"].masked_select(batch["tr_padding_mask"].unsqueeze(-1))),
        )),
        "unknown_validation_and_test_are_excluded": True,
        "state_equation_owner_is_official": spec["state_equation_owner"] == OFFICIAL_CFC_STATE_EQUATION_OWNER,
    }
    if not all(checks.values()):
        raise AssertionError(f"V23 official-CfC smoke failed: {[name for name, ok in checks.items() if not ok]}")
    return {
        "status": "passed",
        "checks": checks,
        "official_cfc": spec,
        "initialization_audit": initialization_audit,
        "real_train_parent_count": len(dataset),
        "real_train_batch_parent_ids": list(batch["parent_sample_ids"]),
        "sequence_logits_shape": list(output["sequence_logits"].shape),
        "loss": float(objective["loss"].item()),
        "long_positive_dt_seconds": actual_long_dt,
        "official_timespan_for_long_positive_dt_seconds": long_timespan,
        "offline_online_max_logit_error": max_online_error,
        "offline_online_tolerance": {"atol": 1.0e-6, "rtol": 1.0e-6},
        "validation_access_count": 0,
        "test_payload_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23 pure official-CfC baseline smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
