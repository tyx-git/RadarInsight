"""Focused contract checks for the V22 fixed-parent-denominator experiment."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from radar_ln3.known_gate import (
    KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
    KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION,
    KNOWN_GATE_LABEL_TO_ID,
    build_known_gate_dataloader,
    build_known_gate_dataset,
    build_known_gate_model_for_config,
    known_gate_config_identity_sha256,
    load_known_gate_config,
)
from radar_ln3.known_gate_v21e_oof import (
    config_relative as v21e_config_relative,
    load_fold_contract,
    prediction_relative as v21e_prediction_relative,
    sha256_file,
)
from radar_ln3.known_gate_v22_oof import (
    CONTRACT_RELATIVE,
    CONTRACT_SCHEMA,
    FOLD_COUNT,
    READINESS_RELATIVE,
    READINESS_SCHEMA,
    config_relative,
)
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from train.known_gate_v21 import _model_inputs, _parent_order_sha256, _state_dict_sha256


class _ParentLabelAuditDataset(Dataset[tuple[str, int]]):
    def __init__(self, rows: list[tuple[str, int]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, int]:
        return self.rows[index]


def _objective(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor,
    mode: str,
    *,
    prefix_weight: float = 0.0,
) -> dict[str, torch.Tensor]:
    mask = torch.zeros(logits.shape[:2], dtype=torch.bool)
    return parent_final_task_ce_with_prefix_auxiliary(
        logits,
        targets,
        mask,
        class_weights=weights,
        class_weight_normalization=mode,
        prefix_auxiliary_weight=prefix_weight,
        prefix_auxiliary_min_events=3,
    )


def _check_loss_math() -> dict[str, Any]:
    weights = torch.tensor([0.5, 15.0], dtype=torch.float64)
    logits = torch.tensor(
        [
            [[0.2, -0.1], [0.4, -0.2], [0.1, 0.8], [0.0, 1.0]],
            [[-0.2, 0.3], [0.1, 0.2], [0.6, -0.4], [-0.3, 0.9]],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    clutter_targets = torch.ones(2, dtype=torch.long)
    final_logits = logits[:, -1]
    unweighted = F.cross_entropy(final_logits, clutter_targets)
    historical = _objective(
        logits,
        clutter_targets,
        weights,
        KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION,
    )["final_state_loss"]
    fixed = _objective(
        logits,
        clutter_targets,
        weights,
        KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
    )["final_state_loss"]
    if not torch.allclose(historical, unweighted, rtol=0.0, atol=1.0e-12):
        raise AssertionError("historical same-class weighted mean did not cancel its weight")
    if not torch.allclose(fixed, unweighted * weights[1], rtol=0.0, atol=1.0e-12):
        raise AssertionError("fixed denominator did not preserve the Clutter class weight")

    mixed_targets = torch.tensor([0, 1], dtype=torch.long)
    per_parent = F.cross_entropy(final_logits, mixed_targets, reduction="none")
    fixed_mixed = _objective(
        logits,
        mixed_targets,
        weights,
        KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
        prefix_weight=0.1,
    )
    expected_final = (per_parent * weights[mixed_targets]).sum() / 2
    if not torch.allclose(
        fixed_mixed["final_state_loss"], expected_final, rtol=0.0, atol=1.0e-12
    ):
        raise AssertionError("V22 final loss is not fixed-parent-count normalized")
    prefix_logits = logits[:, 2]
    prefix_per_parent = F.cross_entropy(prefix_logits, mixed_targets, reduction="none")
    expected_prefix = (prefix_per_parent * weights[mixed_targets]).sum() / 2
    if not torch.allclose(
        fixed_mixed["prefix_auxiliary_loss"],
        expected_prefix,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise AssertionError("V22 prefix loss is not parent-then-class balanced")
    fixed_mixed["loss"].backward()
    if logits.grad is None or not torch.isfinite(logits.grad).all():
        raise AssertionError("V22 fixed-denominator loss produced no finite gradient")
    return {
        "historical_same_class_batch_cancels_class_weight": True,
        "fixed_parent_count_preserves_clutter_weight": True,
        "fixed_final_matches_declared_formula": True,
        "fixed_prefix_averages_parent_then_applies_class_weight": True,
        "finite_gradient": True,
    }


def _replay_historical_batch_exposure(
    root: Path,
    source_contract: dict[str, Any],
) -> list[dict[str, Any]]:
    full_config = load_known_gate_config(root / v21e_config_relative("treatment", 0))
    full_dataset = build_known_gate_dataset(
        root,
        full_config,
        split_name="train",
        allow_feature_cache_superset=True,
    )
    labels_by_parent = {
        str(parent_id): KNOWN_GATE_LABEL_TO_ID[
            str(full_dataset.base._groups[parent_id].iloc[0]["label_name"])
        ]
        for parent_id in full_dataset.parent_ids
    }
    results: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        fold = next(
            dict(row)
            for row in source_contract["folds"]
            if int(row["fold_id"]) == fold_id
        )
        train_ids = set(str(value) for value in fold["train_parent_ids"])
        rows = [
            (str(parent_id), int(labels_by_parent[str(parent_id)]))
            for parent_id in full_dataset.parent_ids
            if str(parent_id) in train_ids
        ]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(43)
        loader = DataLoader(
            _ParentLabelAuditDataset(rows),
            batch_size=2,
            shuffle=True,
            num_workers=0,
            generator=generator,
        )
        weights = torch.tensor(fold["class_weights"], dtype=torch.float64)
        prediction = json.loads(
            (root / v21e_prediction_relative("treatment", fold_id)).read_text(
                encoding="utf-8"
            )
        )
        expected_hashes = [
            str(row["parent_order_sha256"])
            for row in prediction["fold_training_history"]
        ]
        target_only_batches = total_batches = 0
        historical_clutter_mass = fixed_target_mass = fixed_clutter_mass = 0.0
        replayed_hashes: list[str] = []
        for _epoch in range(20):
            order: list[str] = []
            for parent_ids, labels in loader:
                order.extend(str(value) for value in parent_ids)
                labels = labels.to(dtype=torch.long)
                batch_weights = weights[labels]
                total_batches += 1
                target_only_batches += int(bool((labels == 0).all()))
                historical_clutter_mass += float(
                    batch_weights[labels == 1].sum().item()
                    / batch_weights.sum().item()
                )
                fixed_target_mass += float(
                    (batch_weights[labels == 0].sum() / labels.numel()).item()
                )
                fixed_clutter_mass += float(
                    (batch_weights[labels == 1].sum() / labels.numel()).item()
                )
            replayed_hashes.append(_parent_order_sha256(order))
        if replayed_hashes != expected_hashes:
            raise AssertionError(
                f"V22 audit could not reproduce V21e fold {fold_id} parent orders"
            )
        fixed_total_mass = fixed_target_mass + fixed_clutter_mass
        results.append(
            {
                "fold_id": fold_id,
                "epoch_count": 20,
                "total_batch_count": total_batches,
                "target_only_batch_fraction": target_only_batches / total_batches,
                "historical_clutter_effective_mass_fraction": (
                    historical_clutter_mass / total_batches
                ),
                "fixed_denominator_clutter_weighted_mass_fraction": (
                    fixed_clutter_mass / fixed_total_mass
                ),
                "all_epoch_parent_order_hashes_match_v21e": True,
            }
        )
    return results


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    contract_path = root / CONTRACT_RELATIVE
    readiness_path = root / READINESS_RELATIVE
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    source_contract, assignment = load_fold_contract(root)
    if (
        contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("status") != "passed"
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("oof_contract_sha256") != sha256_file(contract_path)
        or int(contract.get("known_parent_count", -1)) != 986
        or int(contract.get("provenance_component_count", -1)) != 975
        or int(contract.get("unknown_training_exposure", -1)) != 0
        or int(contract.get("validation_parent_count", -1)) != 0
        or int(contract.get("test_parent_count", -1)) != 0
    ):
        raise AssertionError("V22 contract/readiness scope is invalid")
    loss_checks = _check_loss_math()
    batch_exposure = _replay_historical_batch_exposure(root, source_contract)
    fold_checks: list[dict[str, Any]] = []
    real_batch_check: dict[str, Any] | None = None
    for fold_id in range(FOLD_COUNT):
        source_path = root / v21e_config_relative("treatment", fold_id)
        treatment_path = root / config_relative(fold_id)
        source = load_known_gate_config(source_path)
        treatment = load_known_gate_config(treatment_path)
        source_binding = next(
            row for row in source_contract["folds"] if int(row["fold_id"]) == fold_id
        )
        binding = next(
            row for row in contract["folds"] if int(row["fold_id"]) == fold_id
        )
        ignored_train = {
            "checkpoint_path",
            "last_checkpoint_path",
            "formal_artifact_dir",
            "epoch_checkpoint_dir",
            "checkpoint_schema",
            "paired_baseline_config_path",
            "class_weight_normalization",
        }
        source_train = {
            key: value
            for key, value in asdict(source.train).items()
            if key not in ignored_train
        }
        treatment_train = {
            key: value
            for key, value in asdict(treatment.train).items()
            if key not in ignored_train
        }
        if (
            treatment.schema_version != "RADAR-LN3-V22"
            or asdict(treatment.data) != asdict(source.data)
            or asdict(treatment.model) != asdict(source.model)
            or treatment_train != source_train
            or source.train.class_weight_normalization
            != KNOWN_GATE_HISTORICAL_CLASS_WEIGHT_NORMALIZATION
            or treatment.train.class_weight_normalization
            != KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION
            or binding["train_parent_ids_sha256"]
            != source_binding["train_parent_ids_sha256"]
            or binding["held_out_parent_ids_sha256"]
            != source_binding["held_out_parent_ids_sha256"]
            or abs(float(binding["loss_evidence"]["weighted_population_mean"]) - 1.0)
            > 1.0e-12
            or binding["treatment"]["config_identity_sha256"]
            != known_gate_config_identity_sha256(treatment)
        ):
            raise AssertionError(f"V22 fold {fold_id} is not a loss-only pair")
        model, reference, pairing = build_known_gate_model_for_config(root, treatment)
        if (
            reference is None
            or pairing.get("loss_only_pair") is not True
            or int(pairing.get("new_fusion_columns", -1)) != 0
            or _state_dict_sha256(model.state_dict())
            != _state_dict_sha256(reference.state_dict())
        ):
            raise AssertionError(f"V22 fold {fold_id} initial model differs from Control")
        if fold_id == 0:
            train_rows = [
                row
                for row in assignment["assignments"]
                if int(row["fold_id"]) != fold_id and int(row["event_count"]) >= 4
            ]
            target_id = next(
                str(row["parent_sample_id"])
                for row in train_rows
                if str(row["label_name"]) != "ClutterTarget"
            )
            clutter_id = next(
                str(row["parent_sample_id"])
                for row in train_rows
                if str(row["label_name"]) == "ClutterTarget"
            )
            dataset = build_known_gate_dataset(
                root,
                treatment,
                split_name="train",
                parent_ids=[target_id, clutter_id],
                allow_feature_cache_superset=True,
            )
            batch = next(
                iter(
                    build_known_gate_dataloader(
                        dataset,
                        batch_size=2,
                        shuffle=False,
                    )
                )
            )
            model.train()
            reference.eval()
            output = model(**_model_inputs(batch))
            with torch.inference_mode():
                reference_logits = reference(**_model_inputs(batch))["sequence_logits"]
            if not torch.equal(output["sequence_logits"].detach(), reference_logits):
                raise AssertionError("V22 real-batch initial logits differ from V21e Control")
            objective = parent_final_task_ce_with_prefix_auxiliary(
                output["sequence_logits"],
                batch["labels"],
                batch["tr_padding_mask"],
                class_weights=torch.tensor(treatment.train.class_weights),
                class_weight_normalization=KNOWN_GATE_FIXED_CLASS_WEIGHT_NORMALIZATION,
                prefix_auxiliary_weight=treatment.train.prefix_auxiliary_weight,
                prefix_auxiliary_min_events=treatment.train.prefix_auxiliary_min_events,
            )
            objective["loss"].backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if (
                not gradients
                or not torch.isfinite(objective["loss"])
                or not all(torch.isfinite(value).all() for value in gradients)
                or float(sum(value.abs().sum() for value in gradients).item()) <= 0.0
            ):
                raise AssertionError("V22 real CQ08 batch produced no finite gradient")
            real_batch_check = {
                "parent_ids": [target_id, clutter_id],
                "labels": [int(value) for value in batch["labels"].tolist()],
                "initial_logits_exactly_equal_to_v21e": True,
                "fixed_parent_count_loss_finite": True,
                "full_model_gradient_finite_and_nonzero": True,
            }
        fold_checks.append(
            {
                "fold_id": fold_id,
                "only_loss_normalization_changes": True,
                "initial_model_state_exactly_equal": True,
                "parent_order_contract_equal": True,
                "fold_train_weight_mean": float(
                    binding["loss_evidence"]["weighted_population_mean"]
                ),
            }
        )
    if real_batch_check is None:
        raise AssertionError("V22 real CQ08 batch smoke did not execute")
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V22-known-gate-fixed-parent-balanced-contract-smoke-v1",
        "loss_checks": loss_checks,
        "historical_batch_exposure_replay": batch_exposure,
        "fold_checks": fold_checks,
        "real_cq08_batch_check": real_batch_check,
        "known_parent_count": int(assignment["known_parent_count"]),
        "provenance_component_count": int(assignment["provenance_component_count"]),
        "unknown_training_exposure": 0,
        "validation_parent_assignment_count": 0,
        "test_parent_assignment_count": 0,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V22 KnownGate OOF.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
