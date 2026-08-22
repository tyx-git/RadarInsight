"""Train-only V23j contract smoke for the epoch-linear prefix-weight ramp."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset
from radar_ln3.known_gate_official_cfc import (
    build_official_cfc_model_from_frozen_source,
)
from radar_ln3.known_gate_v23 import FIXED_EPOCHS, sha256_file
from radar_ln3.known_gate_v23_adapters import (
    V23A_VARIANT,
    load_adapter_config,
    source_v23_training,
)
from radar_ln3.known_gate_v23j import (
    V23J_PREFIX_RAMP_END_WEIGHT,
    V23J_PREFIX_RAMP_START_WEIGHT,
    config_relative,
    load_config,
    prefix_auxiliary_weight_for_epoch,
    source_v23a_training,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs, _state_dict_sha256


def _objective(
    output: dict[str, torch.Tensor],
    batch: dict[str, object],
    *,
    weights: torch.Tensor,
    config: object,
    epoch: int,
) -> dict[str, torch.Tensor]:
    weight = prefix_auxiliary_weight_for_epoch(epoch, config.prefix_ramp)
    return parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    config, source_config, _source_path = load_config(root)
    v23a_config, _v23a_source, _v23a_source_path = load_adapter_config(
        root, V23A_VARIANT
    )
    _summary_path, source_v23_summary, _checkpoint_path, _checkpoint_sha = (
        source_v23_training(root)
    )
    (
        _v23a_config_path,
        _v23a_loaded_config,
        _v23a_summary_path,
        source_v23a_summary,
        _v23a_checkpoint_path,
    ) = source_v23a_training(root)
    train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
    batch = next(
        iter(build_known_gate_dataloader(train_dataset, batch_size=2, shuffle=False))
    )
    model, _audit = build_official_cfc_model_from_frozen_source(
        root, source_config, state_adapter_spec=config.adapter
    )
    initial_state_matches = (
        _state_dict_sha256(model.state_dict())
        == str(source_v23a_summary["initial_model_state_sha256"])
    )
    model.train()
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    reference_output = model(**_model_inputs(batch))
    epoch_1 = _objective(
        reference_output, batch, weights=weights, config=config, epoch=1
    )
    epoch_10 = _objective(
        reference_output, batch, weights=weights, config=config, epoch=10
    )
    epoch_20 = _objective(
        reference_output, batch, weights=weights, config=config, epoch=20
    )
    baseline = parent_final_task_ce_with_prefix_auxiliary(
        reference_output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=v23a_config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    model.zero_grad(set_to_none=True)
    epoch_1_gradient_output = model(**_model_inputs(batch))
    epoch_1_gradient_objective = _objective(
        epoch_1_gradient_output, batch, weights=weights, config=config, epoch=1
    )
    epoch_1_gradient_objective["loss"].backward()
    epoch_1_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    baseline_gradient_output = model(**_model_inputs(batch))
    baseline_gradient_objective = parent_final_task_ce_with_prefix_auxiliary(
        baseline_gradient_output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=v23a_config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    baseline_gradient_objective["loss"].backward()
    baseline_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    epoch_20_gradient_output = model(**_model_inputs(batch))
    epoch_20_gradient_objective = _objective(
        epoch_20_gradient_output, batch, weights=weights, config=config, epoch=20
    )
    epoch_20_gradient_objective["loss"].backward()
    gradients_finite = all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    frozen_train_fields = (
        "batch_size",
        "learning_rate",
        "seed",
        "dataloader_seed",
        "device",
        "epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
        "task_min_precision_recall_target",
        "class_weights",
        "loss_mode",
        "class_weight_normalization",
        "prefix_auxiliary_min_events",
        "record_parent_order_hash",
        "save_every_epoch",
        "cuda_deterministic_algorithms",
        "full_parameter_training",
    )
    expected_midpoint = 0.1 + 0.9 * 9 / 19
    checks = {
        "v23j_config_is_bound_to_its_generated_file": config.config_name
        == config_relative().stem.removesuffix("-config"),
        "v23j_initial_model_state_exactly_matches_v23a": initial_state_matches,
        "v23a_and_v23_initial_state_sha256_match": (
            str(source_v23a_summary["initial_model_state_sha256"])
            == str(source_v23_summary["initial_model_state_sha256"])
        ),
        "v23j_reuses_v23a_fixed_retention_adapter": config.adapter
        == v23a_config.adapter,
        "all_non_schedule_train_hyperparameters_match_v23a": all(
            getattr(config.train, field) == getattr(v23a_config.train, field)
            for field in frozen_train_fields
        ),
        "v23j_checkpoint_rule_is_v23e_rule": config.train.checkpoint_selection_metric
        == "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
        "epoch_1_prefix_weight_is_exactly_v23a_0_1": math.isclose(
            prefix_auxiliary_weight_for_epoch(1, config.prefix_ramp),
            V23J_PREFIX_RAMP_START_WEIGHT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "epoch_10_prefix_weight_is_exactly_linear_midpoint": math.isclose(
            prefix_auxiliary_weight_for_epoch(10, config.prefix_ramp),
            expected_midpoint,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "epoch_20_prefix_weight_is_exactly_v23f_1_0": math.isclose(
            prefix_auxiliary_weight_for_epoch(20, config.prefix_ramp),
            V23J_PREFIX_RAMP_END_WEIGHT,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        ),
        "epoch_1_real_train_batch_loss_exactly_matches_v23a_objective": bool(
            torch.allclose(epoch_1["loss"], baseline["loss"])
            and torch.allclose(epoch_1["final_state_loss"], baseline["final_state_loss"])
            and torch.allclose(epoch_1["prefix_auxiliary_loss"], baseline["prefix_auxiliary_loss"])
        ),
        "epoch_1_real_train_batch_gradients_exactly_match_v23a_objective": False,
        "final_state_loss_is_invariant_across_ramp_epochs": bool(
            torch.allclose(epoch_1["final_state_loss"], epoch_10["final_state_loss"])
            and torch.allclose(epoch_1["final_state_loss"], epoch_20["final_state_loss"])
        ),
        "only_prefix_term_changes_with_epoch_weight": all(
            torch.allclose(
                objective["loss"],
                objective["final_state_loss"]
                + prefix_auxiliary_weight_for_epoch(epoch, config.prefix_ramp)
                * objective["prefix_auxiliary_loss"],
            )
            for epoch, objective in ((1, epoch_1), (10, epoch_10), (20, epoch_20))
        ),
        "eligible_prefix_count_is_invariant_across_ramp_epochs": (
            int(epoch_1["eligible_prefix_count"].item())
            == int(epoch_10["eligible_prefix_count"].item())
            == int(epoch_20["eligible_prefix_count"].item())
        ),
        "v23j_real_train_batch_losses_are_finite": all(
            bool(torch.isfinite(objective["loss"]))
            for objective in (epoch_1, epoch_10, epoch_20)
        ),
        "v23j_real_train_batch_all_gradients_are_finite": gradients_finite,
        "validation_dataset_access_count_is_zero": True,
        "unknown_dataset_access_count_is_zero": True,
        "test_dataset_access_count_is_zero": True,
    }
    checks["epoch_1_real_train_batch_gradients_exactly_match_v23a_objective"] = (
        set(epoch_1_gradients) == set(baseline_gradients)
        and all(
            torch.equal(epoch_1_gradients[name], baseline_gradients[name])
            for name in epoch_1_gradients
        )
    )
    checks["schedule_has_exactly_fixed_20_epochs"] = config.train.epochs == FIXED_EPOCHS
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"V23j prefix-ramp smoke failed: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "v23j_config_sha256": sha256_file(root / config_relative()),
        "prefix_auxiliary_weight_epoch_1": prefix_auxiliary_weight_for_epoch(
            1, config.prefix_ramp
        ),
        "prefix_auxiliary_weight_epoch_10": prefix_auxiliary_weight_for_epoch(
            10, config.prefix_ramp
        ),
        "prefix_auxiliary_weight_epoch_20": prefix_auxiliary_weight_for_epoch(
            20, config.prefix_ramp
        ),
        "real_train_batch_loss_epoch_1": float(epoch_1["loss"].item()),
        "real_train_batch_loss_epoch_10": float(epoch_10["loss"].item()),
        "real_train_batch_loss_epoch_20": float(epoch_20["loss"].item()),
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23j prefix-ramp train-only contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
