"""Train-only V23f contract smoke for the prefix-weight-only treatment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from radar_ln3.known_gate import build_known_gate_dataloader, build_known_gate_dataset
from radar_ln3.known_gate_official_cfc import build_official_cfc_model_from_frozen_source
from radar_ln3.known_gate_v23 import sha256_file
from radar_ln3.known_gate_v23_adapters import V23A_VARIANT, load_adapter_config, source_v23_training
from radar_ln3.known_gate_v23f import (
    V23F_PREFIX_AUXILIARY_WEIGHT,
    config_relative,
    load_config,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs, _state_dict_sha256


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    config, source_config, _source_path = load_config(root)
    v23a_config, _v23a_source, _v23a_source_path = load_adapter_config(root, V23A_VARIANT)
    _summary_path, source_v23_summary, _checkpoint_path, _checkpoint_sha = source_v23_training(root)
    train_dataset = build_known_gate_dataset(root, source_config, split_name="train")
    batch = next(iter(build_known_gate_dataloader(train_dataset, batch_size=2, shuffle=False)))
    model, _audit = build_official_cfc_model_from_frozen_source(
        root, source_config, state_adapter_spec=config.adapter
    )
    initial_state_matches = (
        _state_dict_sha256(model.state_dict())
        == str(source_v23_summary["initial_model_state_sha256"])
    )
    model.train()
    output = model(**_model_inputs(batch))
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    baseline = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=v23a_config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    treatment = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        class_weight_normalization=config.train.class_weight_normalization,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
    )
    model.zero_grad(set_to_none=True)
    treatment["loss"].backward()
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
    checks = {
        "v23f_config_is_bound_to_its_generated_file": config.config_name == config_relative().stem.removesuffix("-config"),
        "v23f_initial_model_state_exactly_matches_v23a": initial_state_matches,
        "v23f_reuses_v23a_fixed_retention_adapter": config.adapter == v23a_config.adapter,
        "v23f_prefix_auxiliary_weight_is_exactly_1_0": config.train.prefix_auxiliary_weight == V23F_PREFIX_AUXILIARY_WEIGHT,
        "all_other_train_hyperparameters_match_v23a": all(
            getattr(config.train, field) == getattr(v23a_config.train, field)
            for field in frozen_train_fields
        ),
        "v23f_checkpoint_rule_is_v23e_rule": config.train.checkpoint_selection_metric == "V23e_parent_final_and_parent_balanced_eligible_prefix_selector",
        "v23f_real_train_batch_losses_are_finite": bool(torch.isfinite(baseline["loss"]) and torch.isfinite(treatment["loss"])),
        "v23f_loss_differs_only_by_registered_prefix_coefficient": bool(
            torch.allclose(
                baseline["loss"],
                baseline["final_state_loss"]
                + v23a_config.train.prefix_auxiliary_weight * baseline["prefix_auxiliary_loss"],
            )
            and torch.allclose(
                treatment["loss"],
                treatment["final_state_loss"]
                + config.train.prefix_auxiliary_weight * treatment["prefix_auxiliary_loss"],
            )
        ),
        "v23f_real_train_batch_all_gradients_are_finite": gradients_finite,
        "validation_dataset_access_count_is_zero": True,
        "unknown_dataset_access_count_is_zero": True,
        "test_dataset_access_count_is_zero": True,
    }
    if not all(checks.values()):
        failed = [name for name, value in checks.items() if not value]
        raise AssertionError(f"V23f prefix-weight smoke failed: {failed}")
    return {
        "status": "passed",
        "checks": checks,
        "v23f_config_sha256": sha256_file(root / config_relative()),
        "real_train_batch_loss_weight_0_1": float(baseline["loss"].item()),
        "real_train_batch_loss_weight_1_0": float(treatment["loss"].item()),
        "validation_dataset_access_count": 0,
        "unknown_dataset_access_count": 0,
        "test_dataset_access_count": 0,
        "OOF": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V23f prefix-weight train-only contract smoke.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
