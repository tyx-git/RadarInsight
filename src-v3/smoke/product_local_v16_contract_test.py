from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import torch

from radar_ln3.cascade import hierarchical_time_quality_fractions, validate_cascade_configs
from radar_ln3.losses import parent_final_task_ce_with_prefix_auxiliary
from radar_ln3.rejection import (
    CU_CLUTTER_PROBABILITY_SCORE,
    DBB_KNOWN_PROBABILITY_SCORE,
    GATE_DBB_PROBABILITY_SCORE,
    STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
    TRAIN_PARENT_GROUPED_OOF_SOURCE,
    StrictOpenSetRejectionCalibration,
    StrictOpenSetRejectionPolicy,
)
from radar_ln3.task_models import (
    PRODUCT_LOCAL_SCHEMA_VERSION,
    LiquidTaskClassifier,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    save_task_checkpoint,
    summarize_task_performance_guardrails,
    task_config_identity_sha256,
    verify_task_data_contract,
)


CONFIG_NAMES = {
    "dbb_cu_gate": "RADAR-LN3-V16-cq08-liquid-product-local-dbb-cu-gate-config.json",
    "dbb_specialist": "RADAR-LN3-V16-cq08-liquid-product-local-dbb-specialist-config.json",
    "cu_specialist": "RADAR-LN3-V16-cq08-liquid-product-local-cu-specialist-config.json",
}


def _expect_failure(call: Callable[[], Any], contains: str) -> str:
    try:
        call()
    except (ValueError, RuntimeError, PermissionError) as error:
        message = str(error)
        if contains.lower() not in message.lower():
            raise AssertionError(f"failure did not mention {contains!r}: {message}") from error
        return message
    raise AssertionError(f"expected failure containing {contains!r}")


def _loss_contract() -> dict[str, Any]:
    torch.manual_seed(20260812)
    logits = torch.randn(3, 5, 2, requires_grad=True)
    labels = torch.tensor([0, 1, 0], dtype=torch.long)
    padding = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, True, True],
            [False, False, True, True, True],
        ],
        dtype=torch.bool,
    )
    result = parent_final_task_ce_with_prefix_auxiliary(
        logits,
        labels,
        padding,
        class_weights=torch.tensor([1.0, 1.0]),
        prefix_auxiliary_weight=0.1,
        prefix_auxiliary_min_events=3,
    )
    result["loss"].backward()
    gradient = logits.grad
    if gradient is None:
        raise AssertionError("V16 local loss produced no gradient")
    if bool((gradient.masked_select(padding.unsqueeze(-1).expand_as(gradient)) != 0).any()):
        raise AssertionError("V16 local loss propagated gradient through padding")
    if float(gradient[0, :2].abs().max().item()) != 0.0:
        raise AssertionError("V16 local loss supervised prefixes before event 3")
    if float(gradient[1, :2].abs().max().item()) != 0.0:
        raise AssertionError("V16 local loss supervised short-parent prefixes before event 3")
    if float(gradient[2, :1].abs().max().item()) != 0.0:
        raise AssertionError("V16 local loss supervised a pre-final event for a two-event parent")
    if int(result["eligible_prefix_count"].item()) != 2:
        raise AssertionError("V16 local loss eligible-prefix count changed")
    if int(result["eligible_prefix_parent_count"].item()) != 1:
        raise AssertionError("V16 local loss eligible-parent count changed")
    lengths = (~padding).long().sum(dim=1)
    event_numbers = torch.arange(1, logits.shape[1] + 1).unsqueeze(0)
    unsupported_early_prefix = (
        (~padding)
        & (event_numbers < 3)
        & (event_numbers < lengths.unsqueeze(1))
    )
    unsupported_gradient_max = (
        float(
            gradient.masked_select(
                unsupported_early_prefix.unsqueeze(-1).expand_as(gradient)
            )
            .abs()
            .max()
            .item()
        )
        if bool(unsupported_early_prefix.any())
        else 0.0
    )
    if unsupported_gradient_max != 0.0:
        raise AssertionError("V16 local loss supervised an early non-final prefix")
    return {
        "total_loss": float(result["loss"].item()),
        "final_state_loss": float(result["final_state_loss"].item()),
        "prefix_auxiliary_loss": float(result["prefix_auxiliary_loss"].item()),
        "eligible_prefix_count": int(result["eligible_prefix_count"].item()),
        "eligible_parent_count": int(result["eligible_prefix_parent_count"].item()),
        "padding_gradient_max": float(
            gradient.masked_select(padding.unsqueeze(-1).expand_as(gradient)).abs().max().item()
        ),
        "events_before_minimum_nonfinal_gradient_max": unsupported_gradient_max,
    }


def _time_quality_contract() -> dict[str, Any]:
    values = hierarchical_time_quality_fractions(
        first_event_count=1,
        zero_delta_count=3,
        positive_delta_count=2,
        genuine_gap_count=0,
    )
    if values["gap_fraction"] != 0.0:
        raise AssertionError("same-second observations still entered gap_fraction")
    if abs(values["zero_dt_fraction"] - (4.0 / 6.0)) > 1.0e-12:
        raise AssertionError("V16 zero_dt_fraction changed")
    return {**values, "same_second_is_gap": False}


def _route_contract() -> dict[str, Any]:
    calibration = StrictOpenSetRejectionCalibration(
        schema_version=STRICT_OPEN_SET_REJECTION_CALIBRATION_SCHEMA,
        source=TRAIN_PARENT_GROUPED_OOF_SOURCE,
        gate_score_name=GATE_DBB_PROBABILITY_SCORE,
        dbb_known_score_name=DBB_KNOWN_PROBABILITY_SCORE,
        cu_score_name=CU_CLUTTER_PROBABILITY_SCORE,
        gate_accept_dbb_threshold=0.70,
        gate_accept_dbb_margin=0.20,
        dbb_accept_known_threshold=0.70,
        dbb_accept_known_margin=0.20,
        cu_accept_clutter_threshold=0.70,
        cu_accept_clutter_margin=0.20,
        calibration_quality_status="deployment_eligible",
        deployment_eligible=True,
        min_valid_points=2,
        min_track_quality=0.0,
        max_gap_fraction=1.0,
    )
    policy = StrictOpenSetRejectionPolicy(calibration, enabled=True)
    clutter = policy.decide(
        torch.tensor([0.90, 0.10]),
        torch.tensor([0.40, 0.35, 0.25]),
        torch.tensor([0.90, 0.10]),
        valid_point_count=3,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    unknown = policy.decide(
        torch.tensor([0.90, 0.10]),
        torch.tensor([0.40, 0.35, 0.25]),
        torch.tensor([0.55, 0.45]),
        valid_point_count=3,
        track_quality=1.0,
        gap_fraction=0.0,
    )
    if clutter.final_action != "CLUTTER_ACCEPTED" or clutter.accepted_label_id != 3:
        raise AssertionError("DBB knownness failure did not fall back to CU Clutter acceptance")
    if unknown.final_action != "UNKNOWN_REJECTED" or unknown.accepted_label_id is not None:
        raise AssertionError("DBB knownness failure did not fall back to CU Unknown rejection")
    if clutter.gate_action != "DBB_ROUTE_ACCEPTED_THEN_FALLBACK_TO_CU":
        raise AssertionError("DBB-to-CU fallback is not explicit in the action trace")
    return {
        "dbb_knownness_failure_then_clutter": clutter.final_action,
        "dbb_knownness_failure_then_unknown": unknown.final_action,
        "fallback_gate_action": clutter.gate_action,
    }


def _performance_contract() -> dict[str, Any]:
    evaluation = {
        "parent_metrics": {
            "per_class": [
                {"class": "A", "precision": 0.96, "recall": 0.97, "f1": 0.965, "support": 10},
                {"class": "B", "precision": 0.94, "recall": 0.96, "f1": 0.95, "support": 10},
            ]
        }
    }
    failed = summarize_task_performance_guardrails(evaluation, target=0.95)
    if failed["performance_status"] != "failed":
        raise AssertionError("V16 performance status ignored a failed class guardrail")
    completed_without_evaluation = summarize_task_performance_guardrails(None, target=0.95)
    if completed_without_evaluation["performance_status"] != "not_evaluated":
        raise AssertionError("V16 process/performance state separation changed")
    return {
        "failed_guardrail_status": failed["performance_status"],
        "not_evaluated_status": completed_without_evaluation["performance_status"],
        "observed_minimum": failed["observed_min_per_class_precision_recall_f1"],
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_root = root / "config" / "RADAR-LN3"
    configs = {
        name: load_task_config(config_root / filename)
        for name, filename in CONFIG_NAMES.items()
    }
    if any(config.schema_version != PRODUCT_LOCAL_SCHEMA_VERSION for config in configs.values()):
        raise AssertionError("V16 focused test loaded a non-V16 configuration")
    contracts = {
        name: verify_task_data_contract(root, config)
        for name, config in configs.items()
    }
    cascade_block_error = _expect_failure(
        lambda: validate_cascade_configs(configs),
        "restricted to independent task training",
    )
    train_datasets = {
        name: build_task_dataset(root, config, split_name=config.data.train_split_name)
        for name, config in configs.items()
    }
    validation_datasets = {
        name: build_task_dataset(root, config, split_name=config.data.validation_split_name)
        for name, config in configs.items()
    }
    cu_labels = set(train_datasets["cu_specialist"].base.frame["label_name"].astype(str))
    if cu_labels != {"ClutterTarget", "UnknownTarget"}:
        raise AssertionError(f"V16 CU training labels are invalid: {sorted(cu_labels)}")

    gate_config = configs["dbb_cu_gate"]
    gate_model = LiquidTaskClassifier(gate_config)
    checkpoint_path = root / "src-v3" / "smoke" / "RADAR-LN3-V16-product-local-contract-checkpoint.pt"
    try:
        save_task_checkpoint(
            checkpoint_path,
            gate_model,
            None,
            config=gate_config,
            epoch=0,
            global_step=0,
            metrics={"scope": "focused_contract"},
        )
        load_task_checkpoint(
            checkpoint_path,
            gate_model,
            expected_schema=gate_config.train.checkpoint_schema,
            map_location="cpu",
        )
        mutated = replace(
            gate_config,
            train=replace(gate_config.train, prefix_auxiliary_weight=0.2),
        )
        mismatch_error = _expect_failure(
            lambda: load_task_checkpoint(
                checkpoint_path,
                LiquidTaskClassifier(mutated),
                expected_schema=mutated.train.checkpoint_schema,
                map_location="cpu",
            ),
            "identity mismatch",
        )
    finally:
        checkpoint_path.unlink(missing_ok=True)

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V16-product-local-focused-contract-v1",
        "data_contract_hashes": contracts["dbb_cu_gate"]["hashes"],
        "task_parent_counts": {
            name: {
                "train": len(train_datasets[name]),
                "validation": len(validation_datasets[name]),
                "train_label_counts": train_datasets[name].label_counts(),
                "validation_label_counts": validation_datasets[name].label_counts(),
            }
            for name in CONFIG_NAMES
        },
        "cu_training_labels": sorted(cu_labels),
        "unknown_training_exposure_in_cu": train_datasets["cu_specialist"].label_counts()[
            "UnknownTarget"
        ],
        "loss": _loss_contract(),
        "time_quality": _time_quality_contract(),
        "route": _route_contract(),
        "performance": _performance_contract(),
        "checkpoint": {
            "round_trip": True,
            "config_identity_sha256": task_config_identity_sha256(gate_config),
            "mutated_config_rejected": True,
            "mismatch_error": mismatch_error,
        },
        "cascade_scope": {
            "assembly_blocked": True,
            "error": cascade_block_error,
        },
        "formal_training": "未执行",
        "full_cascade_evaluation": "未执行",
        "oof_calibration": "未执行",
        "test": "未执行",
        "stable": "未执行",
        "p95": "未执行",
    }
    output = root / "src-v3/smoke/RADAR-LN3-V16-product-local-focused-contract-result.json"
    result["result_path"] = str(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the V16 product-local task contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    run(args.project_root.resolve())


if __name__ == "__main__":
    main()
