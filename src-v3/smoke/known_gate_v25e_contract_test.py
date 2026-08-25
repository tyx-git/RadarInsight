"""Contract and helper smoke tests for V25e."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import RandomSampler

from build.known_gate_v25e import CONFIG, READINESS
from radar_ln3.known_gate import (
    KNOWN_GATE_V25E_READINESS_SCHEMA,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    load_known_gate_config,
)
from radar_ln3.known_gate import collate_known_gate_sequences
from radar_ln3.known_gate_v25b_gradient_decomposition import module_parameter_indices
from radar_ln3.known_gate_features import V25E_SHARED_GRADIENT_PROJECTION_VARIANT
from radar_ln3.known_gate_v25e_gradient_projection import (
    V25E_PROJECTED_MODULE_NAMES,
    V25E_UNCHANGED_MODULE_NAMES,
    project_dominant_gradient,
    project_shared_module_gradients,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs, _parent_order_sha256, _verify_contract, _verify_readiness


def _helper_smoke() -> dict[str, bool]:
    known = (torch.tensor([3.0]), torch.tensor([0.0]))
    clutter = (torch.tensor([-1.0]), torch.tensor([0.0]))
    updated, audit = project_dominant_gradient(known, clutter)
    positive, positive_audit = project_dominant_gradient(
        (torch.tensor([2.0]),), (torch.tensor([1.0]),)
    )
    modules = ("tr_encoder", "rd_encoder", "fusion", "liquid", "classifier")
    synthetic_known = tuple(torch.tensor([2.0]) for _ in modules)
    synthetic_clutter = tuple(torch.tensor([-1.0]) for _ in modules)
    partition = {name: (index,) for index, name in enumerate(modules)}
    projected, module_audit = project_shared_module_gradients(
        synthetic_known, synthetic_clutter, partition
    )
    return {
        "negative_conflict_projects_dominant_gradient": audit["projected"] is True and float(updated[0].item()) == -1.0,
        "smaller_gradient_is_retained": audit["retained_parent"] == "clutter",
        "positive_alignment_is_unchanged": positive_audit["projected"] is False and float(positive[0].item()) == 3.0,
        "only_sealed_shared_modules_are_projected": all(module_audit[name]["projected"] is True for name in V25E_PROJECTED_MODULE_NAMES) and all(module_audit[name]["projected"] is False for name in V25E_UNCHANGED_MODULE_NAMES),
        "rd_and_classifier_proxy_gradients_are_unchanged": all(module_audit[name]["unchanged"] is True for name in V25E_UNCHANGED_MODULE_NAMES) and all(torch.equal(projected[name][0], torch.tensor([1.0])) for name in V25E_UNCHANGED_MODULE_NAMES),
    }


def _real_mixed_batch_smoke(model: torch.nn.Module, dataset: object, config: object) -> bool:
    parent_ids = list(dataset.parent_ids)
    known_index = next(
        index
        for index, parent_id in enumerate(parent_ids)
        if str(dataset.base._groups[parent_id].iloc[0]["label_name"]) != "ClutterTarget"
    )
    clutter_index = next(
        index
        for index, parent_id in enumerate(parent_ids)
        if str(dataset.base._groups[parent_id].iloc[0]["label_name"]) == "ClutterTarget"
    )
    batch = collate_known_gate_sequences([dataset[known_index], dataset[clutter_index]])
    model.train()
    output = model(**_model_inputs(batch))
    weights = torch.tensor(config.train.class_weights, dtype=torch.float32)
    objective = parent_final_task_ce_with_prefix_auxiliary(
        output["sequence_logits"],
        batch["labels"],
        batch["tr_padding_mask"],
        class_weights=weights,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
        class_weight_normalization="batch_weight_sum",
    )
    parameters = tuple(model.parameters())
    indices = module_parameter_indices(model, parameters)
    model.zero_grad(set_to_none=True)
    objective["loss"].backward(retain_graph=True)
    from radar_ln3.known_gate_v25e_gradient_projection import apply_historical_mixed_batch_projection

    audit = apply_historical_mixed_batch_projection(
        parameters=parameters,
        module_indices=indices,
        sequence_logits=output["sequence_logits"],
        labels=batch["labels"],
        padding_mask=batch["tr_padding_mask"],
        class_weights=weights,
        prefix_auxiliary_weight=config.train.prefix_auxiliary_weight,
        prefix_auxiliary_min_events=config.train.prefix_auxiliary_min_events,
        original_loss=objective["loss"],
    )
    return bool(
        audit["mixed_batch"]
        and int(audit["projected_module_count"]) >= 1
        and audit["rd_and_classifier_gradients_exactly_unchanged"]
    )


def run(project_root: Path) -> dict[str, object]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    if not config_path.is_file() or not readiness_path.is_file():
        raise FileNotFoundError("V25e prepare artifacts are incomplete")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if config.variant != V25E_SHARED_GRADIENT_PROJECTION_VARIANT or readiness.get("schema_version") != KNOWN_GATE_V25E_READINESS_SCHEMA or readiness.get("status") != "passed":
        raise AssertionError("V25e config/readiness identity is stale")
    model, reference, pairing = build_known_gate_model_for_config(root, config)
    if reference is None or pairing.get("shared_gradient_projection_pair") is not True or int(pairing.get("new_fusion_columns", -1)) != 0:
        raise AssertionError("V25e paired initialization is incomplete")
    dataset = build_known_gate_dataset(root, config, split_name="train")
    if dataset.label_counts() != {"KNOWN_TARGET_EVIDENCE": 953, "KNOWN_CLUTTER_EVIDENCE": 33}:
        raise AssertionError("V25e train population differs from V25a")
    loader = build_known_gate_dataloader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    model.eval(); reference.eval()
    with torch.inference_mode():
        treatment_logits = model(**_model_inputs(batch))["sequence_logits"]
        reference_logits = reference(**_model_inputs(batch))["sequence_logits"]
    if not torch.equal(treatment_logits, reference_logits):
        raise AssertionError("V25e initial logits differ from V25a control")
    expected_orders = dict(readiness.get("source_v25a_epoch_parent_order_hashes") or {})
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.train.dataloader_seed)
    for epoch in range(1, 21):
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        order = [str(dataset.parent_ids[index]) for index in sampler]
        if _parent_order_sha256(order) != str(expected_orders.get(str(epoch), "")):
            raise AssertionError(f"V25e epoch {epoch} order differs from V25a")
    helper = _helper_smoke()
    if not all(helper.values()):
        raise AssertionError(f"V25e gradient helper smoke failed: {helper}")
    real_mixed_batch_projection = _real_mixed_batch_smoke(model, dataset, config)
    if not real_mixed_batch_projection:
        raise AssertionError("V25e real mixed-batch projection smoke failed")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V25e-known-gate-shared-gradient-conflict-projection-smoke-v1",
        "shared_gradient_projection_pair": True,
        "initial_logits_exactly_equal_to_v25a": True,
        "all_twenty_parent_orders_reproduce_v25a": True,
        "gradient_helper": helper,
        "real_mixed_batch_projection": True,
        "train_parent_count": len(dataset),
        "unknown_training_exposure": 0,
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V25e contracts and projection helper.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
