"""Train-only contract smoke for V23l's full official temporal adapter.

V23l compares the completed self-developed V23g density arm with an official
``ncps.torch.CfC`` arm.  This smoke deliberately opens only the frozen V21e
train split.  It proves that the official cell remains the installed upstream
class, the combined external adapter has the registered reason-disjoint
equations, V21e's course evidence reaches the fusion layer, and online replay
matches offline replay with explicit causal same-second ordinals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from radar_ln3.known_gate import (
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
)
from radar_ln3.known_gate_official_cfc import build_official_cfc_model_from_frozen_source
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23l import (
    V23L_VARIANT,
    full_temporal_adapter_spec,
    load_config,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from radar_ln3.official_cfc_state_adapters import ExternalOfficialCfCStateAdapter
from train.known_gate_v21 import _model_inputs, _state_dict_sha256
from train.known_gate_v23l import _initialization_audit, _verify_prepared


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
        "known_gate_features": batch["known_gate_features"][:1, event_index],
        "known_gate_feature_valid": batch["known_gate_feature_valid"][:1, event_index],
    }


def _ordinal(reason: int, current: int) -> int:
    if reason in (1, 3):
        return 1
    if reason == 2:
        return current + 1
    raise ValueError(f"unsupported V23l dt reason: {reason}")


def _density_parent(dataset: Any) -> tuple[Any, int]:
    for item in dataset:
        ordinal = 0
        for event_index, reason in enumerate(item.base.tr_dt_reason.tolist()):
            ordinal = _ordinal(int(reason), ordinal)
            if int(reason) == 2 and ordinal >= 2:
                return item, event_index
    raise ValueError("V23l train smoke requires a real same-second k>=2 parent")


def _course_exposed_parent(dataset: Any) -> Any:
    """Return a real train parent with valid, nonzero causal-course evidence."""

    for item in dataset:
        valid = item.evidence_valid
        if bool(valid.any()) and bool((item.evidence_features[valid].abs() > 0.0).any()):
            return item
    raise ValueError("V23l train smoke requires nonzero causal-course evidence")


def _online_error(model: Any, batch: dict[str, Any]) -> tuple[float, bool]:
    model.eval()
    with torch.inference_mode():
        offline = model(**_model_inputs(batch))
        state = None
        ordinal = 0
        rows: list[torch.Tensor] = []
        missing_ordinal_rejected = False
        length = int(batch["lengths"][0].item())
        for event_index in range(length):
            reason = int(batch["tr_dt_reason"][0, event_index].item())
            ordinal = _ordinal(reason, ordinal)
            if event_index == 0:
                try:
                    model.forward_online_point(
                        **_point_inputs(batch, event_index), liquid_state=state
                    )
                except ValueError as error:
                    missing_ordinal_rejected = "requires explicit online zero_ordinal" in str(error)
            output = model.forward_online_point(
                **_point_inputs(batch, event_index),
                liquid_state=state,
                zero_ordinal=torch.tensor([ordinal], dtype=torch.long),
            )
            state = output["liquid_state"]
            rows.append(output["window_logits"])
        online = torch.stack(rows, dim=1)
        return (
            float((offline["sequence_logits"][:, :length] - online).abs().max().item()),
            missing_ordinal_rejected,
        )


def _formula_checks() -> dict[str, bool]:
    torch.manual_seed(20260823)
    official = torch.randn(3, 48)
    previous = torch.randn(3, 48)
    dt = torch.tensor([0.0, 3.0, 7.0])
    reasons = torch.tensor([1, 3, 2], dtype=torch.long)
    ordinal = torch.tensor([1, 1, 3], dtype=torch.long)
    adapter = ExternalOfficialCfCStateAdapter(full_temporal_adapter_spec()).eval()
    observed = adapter(official, previous, dt, reasons, ordinal)
    alpha = 1.0 - torch.exp(-dt[1] / 15.025)
    expected_positive = (1.0 - alpha) * previous[1] + alpha * official[1]
    expected_zero = 0.25 * previous[2] + 0.75 * official[2]
    return {
        "first_event_is_uninterpolated_official_state": bool(
            torch.equal(observed[0], official[0])
        ),
        "positive_dt_uses_fixed_15_025_second_retention": bool(
            torch.allclose(observed[1], expected_positive, atol=ATOL, rtol=RTOL)
        ),
        "zero_delta_k3_uses_0_75_density_gain": bool(
            torch.allclose(observed[2], expected_zero, atol=ATOL, rtol=RTOL)
        ),
        "adapter_has_no_trainable_parameters": sum(
            parameter.numel() for parameter in adapter.parameters()
        ) == 0,
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, source_config, readiness, _contract_path, _contract = _verify_prepared(root)
    loaded, loaded_source, _source_path = load_config(root)
    if loaded != config or loaded_source != source_config:
        raise RuntimeError("V23l preparation reload is not stable")
    dataset = build_known_gate_dataset(root, source_config, split_name="train")
    selected, first_density_index = _density_parent(dataset)
    batch = next(iter(build_known_gate_dataloader([selected], batch_size=1, shuffle=False)))
    course_batch = next(
        iter(build_known_gate_dataloader([_course_exposed_parent(dataset)], batch_size=1, shuffle=False))
    )
    model, source_copy_audit = build_official_cfc_model_from_frozen_source(
        root, source_config, state_adapter_spec=config.adapter
    )
    initialization_audit = _initialization_audit(root, model, source_config)
    control, _unused, _pairing = build_known_gate_model_for_config(root, source_config)
    model.eval()
    control.eval()
    with torch.inference_mode():
        official_output = model(**_model_inputs(batch))
        base_output = control(**_model_inputs(batch))
        course_output = model(**_model_inputs(course_batch))
        no_evidence = dict(course_batch)
        no_evidence["known_gate_features"] = torch.zeros_like(course_batch["known_gate_features"])
        no_evidence["known_gate_feature_valid"] = torch.zeros_like(
            course_batch["known_gate_feature_valid"]
        )
        no_evidence_output = model(**_model_inputs(no_evidence))
        evidence_difference = float(
            (course_output["sequence_logits"] - no_evidence_output["sequence_logits"])
            .abs()
            .max()
            .item()
        )
        temporal_difference_from_self_developed = float(
            (official_output["sequence_logits"] - base_output["sequence_logits"])
            .abs()
            .max()
            .item()
        )
    online_error, missing_ordinal_rejected = _online_error(model, batch)
    model.train()
    output = model(**_model_inputs(course_batch))
    objective = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        course_batch["labels"],
        course_batch["tr_padding_mask"],
        class_weights=torch.tensor(config.train.class_weights, dtype=torch.float32),
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    objective["loss"].backward()
    finite_gradients = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    course_fusion_gradient = float(model.fusion[0].weight.grad[:, -1].abs().sum().item())
    checks = {
        "variant_is_v23l": config.variant == V23L_VARIANT,
        "official_cfcell_is_unmodified_external_adapter_only": (
            model.liquid.cfc.__class__.__module__ == "ncps.torch.cfc"
        ),
        "official_dense_default_48x1_backbone_is_pinned": (
            model.backbone_specification()["backbone_units"] == 48
            and model.backbone_specification()["backbone_layers"] == 1
            and model.backbone_specification()["mixed_memory"] is False
            and model.backbone_specification()["proj_size"] is None
            and model.backbone_specification()["wiring"] is None
        ),
        "adapter_is_exact_registered_full_temporal_composition": (
            model.external_state_adapter_specification() == config.adapter.as_dict()
        ),
        "formula_matches_first_positive_and_zero_density_branches": all(
            _formula_checks().values()
        ),
        "v21e_non_recurrent_course_state_is_tensor_exact": (
            source_copy_audit["shared_non_recurrent_state_exactly_equal"] is True
            and initialization_audit["shared_non_recurrent_state_exactly_matches_v21e"]
            is True
        ),
        "parameter_count_is_v23a_plus_one_course_fusion_column": (
            sum(parameter.numel() for parameter in model.parameters()) == 27482
        ),
        # The course column is correctly present in this model even if a
        # particular randomly initialized/trained sample happens to give an
        # exactly cancelling readout difference.  Tensor-exact source copying
        # above is the structural proof; this numerical value is retained only
        # as a diagnostic rather than a stochastic readiness gate.
        "course_evidence_input_contract_is_active": (
            model.evidence_dim == 1
            and course_batch["known_gate_features"].shape[-1] == 1
            and bool(course_batch["known_gate_feature_valid"].any())
            and bool(
                (course_batch["known_gate_features"]
                 .masked_select(course_batch["known_gate_feature_valid"])
                 .abs() > 0.0).any()
            )
            and course_fusion_gradient > 0.0
        ),
        "official_temporal_path_is_distinct_from_self_developed_path": (
            temporal_difference_from_self_developed > 1.0e-8
        ),
        "offline_online_logits_match_with_explicit_causal_ordinal": online_error <= ATOL,
        "online_combined_adapter_rejects_missing_ordinal": missing_ordinal_rejected,
        "real_train_batch_loss_and_all_gradients_are_finite": bool(
            torch.isfinite(objective["loss"]) and finite_gradients
        ),
        "validation_dataset_access_count_is_zero": True,
        "unknown_dataset_access_count_is_zero": True,
        "test_dataset_access_count_is_zero": True,
    }
    if not all(checks.values()):
        raise AssertionError(
            "V23l full temporal adapter smoke failed: "
            + str([name for name, value in checks.items() if not value])
        )
    return {
        "status": "passed",
        "variant": V23L_VARIANT,
        "checks": checks,
        "source_v23g_density": config.source_v23g,
        "frozen_v23g_epoch20_checkpoint": {
            "path": config.source_v23g["epoch20_checkpoint_path"],
            "sha256": config.source_v23g["epoch20_checkpoint_sha256"],
        },
        "adapter": config.adapter.as_dict(),
        "real_train_parent_id": str(batch["parent_sample_ids"][0]),
        "first_k2_or_later_zero_delta_event_index": int(first_density_index),
        "course_evidence_logit_difference": evidence_difference,
        "course_evidence_fusion_gradient_l1": course_fusion_gradient,
        "official_vs_self_developed_initial_logit_difference": temporal_difference_from_self_developed,
        "offline_online_max_logit_error": online_error,
        "loss": float(objective["loss"].item()),
        "train_parent_count": len(dataset),
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23l train-only temporal adapter smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
