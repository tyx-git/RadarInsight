"""Fast contract checks for V25d's present-class-mean loss-only treatment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import RandomSampler

from build.known_gate_v25d import CONFIG, READINESS, _loss_semantics_audit
from radar_ln3.known_gate import (
    KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
    KNOWN_GATE_V25D_READINESS_SCHEMA,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate_features import V25D_PRESENT_CLASS_MEAN_VARIANT
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import (
    _model_inputs,
    _parent_order_sha256,
    _verify_contract,
    _verify_readiness,
)


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    if not config_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V25d prepare artifacts are incomplete; run prepare first")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if (
        config.variant != V25D_PRESENT_CLASS_MEAN_VARIANT
        or config.train.class_weight_normalization
        != KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION
        or readiness.get("schema_version") != KNOWN_GATE_V25D_READINESS_SCHEMA
        or readiness.get("status") != "passed"
    ):
        raise AssertionError("V25d config/readiness identity is stale")
    loss_audit = _loss_semantics_audit()
    model, reference, pairing = build_known_gate_model_for_config(root, config)
    if (
        reference is None
        or pairing.get("loss_only_pair") is not True
        or pairing.get("present_class_mean_loss_only_pair") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 0
    ):
        raise AssertionError("V25d loss-only paired initialization is incomplete")
    dataset = build_known_gate_dataset(root, config, split_name="train")
    if dataset.label_counts() != {
        "KNOWN_TARGET_EVIDENCE": 953,
        "KNOWN_CLUTTER_EVIDENCE": 33,
    }:
        raise AssertionError("V25d train population differs from V25a")
    loader = build_known_gate_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    model.eval()
    reference.eval()
    with torch.inference_mode():
        treatment_logits = model(**_model_inputs(batch))["sequence_logits"]
        reference_logits = reference(**_model_inputs(batch))["sequence_logits"]
    if not torch.equal(treatment_logits, reference_logits):
        raise AssertionError("V25d initial logits differ from V25a control")
    logits = torch.tensor(
        [
            [[1.0, -1.0], [0.4, -0.4], [0.1, -0.1], [0.7, -0.7]],
            [[-0.6, 0.6], [-0.2, 0.2], [-0.8, 0.8], [-0.1, 0.1]],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    padding = torch.zeros(2, 4, dtype=torch.bool)
    objective = parent_final_task_ce_with_prefix_auxiliary(
        logits,
        labels,
        padding,
        class_weights=torch.tensor(config.train.class_weights, dtype=torch.float32),
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
        class_weight_normalization=config.train.class_weight_normalization,
    )
    if (
        objective["final_effective_class_mass"].detach().cpu().tolist() != [0.5, 0.5]
        or objective["prefix_effective_class_mass"].detach().cpu().tolist() != [0.5, 0.5]
        or not torch.isfinite(objective["loss"])
    ):
        raise AssertionError("V25d runtime loss does not implement present-class mean")
    order_generator = torch.Generator(device="cpu")
    order_generator.manual_seed(config.train.dataloader_seed)
    expected_orders = dict(readiness.get("source_v25a_epoch_parent_order_hashes") or {})
    for epoch in range(1, 21):
        # PyTorch's map-style DataLoader consumes one generator value for its
        # iterator base seed before it iterates the RandomSampler.  Mirror that
        # documented implementation detail, then inspect sampler indices only:
        # fetching 20 full RD batches would turn a deterministic-order smoke
        # check into an unnecessary raw-tensor workload.
        torch.empty((), dtype=torch.int64).random_(generator=order_generator)
        sampler = RandomSampler(dataset, replacement=False, generator=order_generator)
        order = [str(dataset.parent_ids[index]) for index in sampler]
        if (
            len(order) != len(dataset)
            or len(set(order)) != len(order)
            or _parent_order_sha256(order) != str(expected_orders.get(str(epoch), ""))
        ):
            raise AssertionError(
                f"V25d epoch {epoch} loader order does not reproduce V25a"
            )
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25d-known-gate-course-present-class-mean-smoke-v1",
        "loss_only_pair": True,
        "initial_logits_exactly_equal_to_v25a": True,
        "mixed_final_and_prefix_mass_exactly_equal": True,
        "pure_class_historical_equivalence": loss_audit[
            "pure_class_loss_equivalent_to_batch_weight_sum"
        ],
        "all_twenty_parent_orders_reproduce_v25a": True,
        "train_parent_count": len(dataset),
        "unknown_training_exposure": 0,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V25d loss-only contracts.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
