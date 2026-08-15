"""Run the frozen V20r 2x2 checkpoint-weight versus inference-mode diagnosis.

The diagnostic never trains a model.  For each seed it combines the sealed
native best-checkpoint records with two cross evaluations:

* bounded Control weights evaluated with density-weighted dynamics;
* density Treatment weights evaluated with bounded-observation dynamics.

This separates the direct inference-time dynamics effect from the parameter
trajectory learned while training under either dynamics rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataset,
    evaluate_task,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    summarize_task_records,
    verify_task_data_contract,
)
from report.liquid_dynamics_v20_paired import (
    CLASS_NAMES,
    _comparison,
    _parent_metrics,
)


SEEDS = (42, 43, 44)
MODES = ("bounded_observation", "density_weighted")
SOURCE_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V20-density-weighted-multiseed-paired-validation.json"
)
OUTPUT_RELATIVE_PATH = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V20-density-weighted-multiseed-2x2-cross-mode-validation.json"
)
SCHEMA_VERSION = (
    "RADAR-LN3-V20-density-weighted-multiseed-2x2-cross-mode-validation-v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _canonical_parent_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = summarize_task_records(
        records,
        class_names=CLASS_NAMES,
        scope="parent_final",
    )
    return _parent_metrics({"parent_metrics": metrics})


def _records_by_parent(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in records:
        parent_id = str(row["parent_sample_id"])
        if parent_id in result:
            raise ValueError(f"duplicate parent-final record: {parent_id}")
        result[parent_id] = dict(row)
    return result


def _subset_metrics(
    records: Iterable[Mapping[str, Any]],
    parent_ids: set[str],
) -> dict[str, Any]:
    selected = [
        dict(row)
        for row in records
        if str(row["parent_sample_id"]) in parent_ids
    ]
    if not selected:
        return {"support": 0, "metrics": None}
    return {
        "support": len(selected),
        "metrics": _canonical_parent_metrics(selected),
    }


def _exposure_groups(dataset: Any) -> dict[str, set[str]]:
    groups = {"none": set(), "low_positive": set(), "high": set()}
    for parent_id in dataset.parent_ids:
        frame = dataset.base._groups[str(parent_id)]
        zero_count = int(
            (frame["tr_dt_reason"] == "zero_delta_at_second_precision").sum()
        )
        denominator = max(1, int(len(frame)) - 1)
        fraction = zero_count / denominator
        if zero_count == 0:
            groups["none"].add(str(parent_id))
        elif fraction < 0.5:
            groups["low_positive"].add(str(parent_id))
        else:
            groups["high"].add(str(parent_id))
    if set().union(*groups.values()) != set(dataset.parent_ids):
        raise ValueError("V20r exposure groups do not cover validation parents exactly")
    return groups


def _transition_summary(
    source_records: Iterable[Mapping[str, Any]],
    target_records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    source = _records_by_parent(source_records)
    target = _records_by_parent(target_records)
    if set(source) != set(target):
        raise ValueError("V20r cross-mode parent sets differ")
    corrected: list[str] = []
    regressed: list[str] = []
    changed: list[str] = []
    by_class = {
        name: {"support": 0, "corrected": 0, "regressed": 0, "changed": 0}
        for name in CLASS_NAMES
    }
    for parent_id in sorted(source):
        before = source[parent_id]
        after = target[parent_id]
        if int(before["true_id"]) != int(after["true_id"]):
            raise ValueError(f"V20r true label changed for {parent_id}")
        label = str(before["label_name"])
        row = by_class[label]
        row["support"] += 1
        before_correct = int(before["true_id"]) == int(before["pred_id"])
        after_correct = int(after["true_id"]) == int(after["pred_id"])
        if int(before["pred_id"]) != int(after["pred_id"]):
            changed.append(parent_id)
            row["changed"] += 1
        if not before_correct and after_correct:
            corrected.append(parent_id)
            row["corrected"] += 1
        elif before_correct and not after_correct:
            regressed.append(parent_id)
            row["regressed"] += 1
    return {
        "prediction_changed_count": len(changed),
        "corrected_count": len(corrected),
        "regressed_count": len(regressed),
        "net_correct_parent_change": len(corrected) - len(regressed),
        "changed_parent_ids": changed,
        "corrected_parent_ids": corrected,
        "regressed_parent_ids": regressed,
        "by_class": by_class,
    }


def _no_exposure_invariant(
    native_records: Iterable[Mapping[str, Any]],
    cross_records: Iterable[Mapping[str, Any]],
    no_exposure_ids: set[str],
) -> dict[str, Any]:
    native = _records_by_parent(
        row
        for row in native_records
        if str(row["parent_sample_id"]) in no_exposure_ids
    )
    cross = _records_by_parent(
        row
        for row in cross_records
        if str(row["parent_sample_id"]) in no_exposure_ids
    )
    if set(native) != no_exposure_ids or set(cross) != no_exposure_ids:
        raise ValueError("V20r no-exposure invariant parent set is incomplete")
    changed_predictions = [
        parent_id
        for parent_id in sorted(no_exposure_ids)
        if int(native[parent_id]["pred_id"]) != int(cross[parent_id]["pred_id"])
    ]
    max_confidence_difference = max(
        (
            abs(float(native[parent_id]["confidence"]) - float(cross[parent_id]["confidence"]))
            for parent_id in no_exposure_ids
        ),
        default=0.0,
    )
    return {
        "parent_count": len(no_exposure_ids),
        "prediction_changed_count": len(changed_predictions),
        "changed_parent_ids": changed_predictions,
        "maximum_absolute_predicted_class_confidence_difference": max_confidence_difference,
        "passed": len(changed_predictions) == 0 and max_confidence_difference <= 1.0e-5,
    }


def _load_native_records(
    root: Path,
    run: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    config_path = Path(str(run["config"])).resolve()
    summary_path = Path(str(run["summary"])).resolve()
    if not config_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"V20r source config/summary is missing: {run}")
    if sha256_file(config_path) != str(run["config_sha256"]):
        raise ValueError(f"V20r source config hash changed: {config_path}")
    if sha256_file(summary_path) != str(run["summary_sha256"]):
        raise ValueError(f"V20r source summary hash changed: {summary_path}")
    config = load_task_config(config_path)
    contract = verify_task_data_contract(root, config)
    if dict(contract["hashes"]) != dict(run["contract_hashes"]):
        raise ValueError(f"V20r source contract hashes changed: {config_path}")
    summary = _read_json(summary_path)
    if int(summary.get("best_epoch", -1)) != int(run["best_epoch"]):
        raise ValueError(f"V20r source best epoch changed: {summary_path}")
    detail_path = Path(str(summary.get("sample_detail_information", ""))).resolve()
    if not detail_path.is_file():
        raise FileNotFoundError(detail_path)
    details = _read_json(detail_path)
    if int(details.get("epoch", -1)) != int(run["best_epoch"]):
        raise ValueError(f"V20r sample detail epoch differs: {detail_path}")
    records = [dict(row) for row in details.get("records", {}).get("parent", [])]
    native_metrics = _canonical_parent_metrics(records)
    if native_metrics != dict(run["best_metrics"]):
        raise ValueError(f"V20r native sample detail metrics changed: {detail_path}")
    return config, {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "sample_detail": str(detail_path),
        "sample_detail_sha256": sha256_file(detail_path),
        "best_epoch": int(run["best_epoch"]),
        "native_metrics": native_metrics,
    }, records


def _cross_evaluate(
    root: Path,
    config: Any,
    dataset: Any,
    *,
    cross_mode: str,
    best_epoch: int,
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if cross_mode not in MODES:
        raise ValueError(f"unsupported V20r cross mode: {cross_mode}")
    checkpoint_path = root / str(config.train.checkpoint_path)
    model = LiquidTaskClassifier(config).to(device)
    checkpoint = load_task_checkpoint(
        checkpoint_path,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    if int(checkpoint.get("epoch", -1)) != best_epoch:
        raise ValueError(f"V20r cross-mode checkpoint epoch differs: {checkpoint_path}")
    native_mode = str(model.liquid.invalid_dt_update_mode)
    if native_mode == cross_mode:
        raise ValueError("V20r cross evaluation must use the non-native dynamics mode")
    state_before = _state_sha256(model)
    model.liquid.invalid_dt_update_mode = cross_mode
    evaluation = evaluate_task(
        model,
        dataset,
        config=config,
        device=device,
    )
    state_after = _state_sha256(model)
    model.liquid.invalid_dt_update_mode = native_mode
    if state_before != state_after:
        raise ValueError("V20r cross-mode evaluation mutated checkpoint tensors")
    records = [dict(row) for row in evaluation["records"]["parent"]]
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": best_epoch,
        "native_mode": native_mode,
        "evaluation_mode": cross_mode,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_unchanged": True,
        "metrics": _parent_metrics(evaluation),
    }, records


def _cell(
    metrics: Mapping[str, Any],
    records: list[dict[str, Any]],
    exposure_groups: Mapping[str, set[str]],
) -> dict[str, Any]:
    return {
        "metrics": dict(metrics),
        "exposure_strata": {
            name: _subset_metrics(records, parent_ids)
            for name, parent_ids in exposure_groups.items()
        },
    }


def _contrast(
    source_cell: Mapping[str, Any],
    target_cell: Mapping[str, Any],
    source_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> dict[str, Any]:
    strata: dict[str, Any] = {}
    for name in source_cell["exposure_strata"]:
        source = source_cell["exposure_strata"][name]
        target = target_cell["exposure_strata"][name]
        if int(source["support"]) != int(target["support"]):
            raise ValueError(f"V20r stratum support differs: {name}")
        strata[name] = {
            "support": int(source["support"]),
            "comparison": (
                None
                if source["metrics"] is None
                else _comparison(source["metrics"], target["metrics"])
            ),
        }
    return {
        "comparison": _comparison(source_cell["metrics"], target_cell["metrics"]),
        "transitions": _transition_summary(source_records, target_records),
        "exposure_strata": strata,
    }


def _summary(values: Iterable[float]) -> dict[str, Any]:
    rows = [float(value) for value in values]
    if not rows or not all(np.isfinite(value) for value in rows):
        raise ValueError("V20r aggregate requires finite values")
    return {
        "values": rows,
        "mean": float(mean(rows)),
        "median": float(median(rows)),
        "minimum": float(min(rows)),
        "maximum": float(max(rows)),
    }


def _interaction_effect(contrasts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    control_direct = contrasts["direct_mode_effect_control_weights"]
    treatment_direct = contrasts["direct_mode_effect_treatment_weights"]

    def difference(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, float]:
        source_delta = source["treatment_minus_control"]
        target_delta = target["treatment_minus_control"]
        return {
            name: float(target_delta[name]) - float(source_delta[name])
            for name in (
                "accuracy",
                "macro_f1",
                "drone_bird_weak_floor",
                "DroneTarget_floor",
                "BirdTarget_floor",
                "BalloonTarget_floor",
            )
        }

    strata: dict[str, Any] = {}
    for name in control_direct["exposure_strata"]:
        control_row = control_direct["exposure_strata"][name]
        treatment_row = treatment_direct["exposure_strata"][name]
        if control_row["comparison"] is None or treatment_row["comparison"] is None:
            strata[name] = {"support": int(control_row["support"]), "difference": None}
        else:
            strata[name] = {
                "support": int(control_row["support"]),
                "difference": difference(
                    control_row["comparison"], treatment_row["comparison"]
                ),
            }
    return {
        "definition": (
            "(density-minus-bounded under Treatment weights) minus "
            "(density-minus-bounded under Control weights)"
        ),
        "difference": difference(
            control_direct["comparison"], treatment_direct["comparison"]
        ),
        "exposure_strata": strata,
    }


def _aggregate(seed_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    contrast_names = (
        "direct_mode_effect_control_weights",
        "direct_mode_effect_treatment_weights",
        "training_weight_effect_bounded_mode",
        "training_weight_effect_density_mode",
        "native_total_effect",
    )
    result: dict[str, Any] = {}
    for contrast_name in contrast_names:
        seed_rows = [seed_results[str(seed)]["contrasts"][contrast_name] for seed in SEEDS]
        result[contrast_name] = {
            "macro_f1": _summary(
                row["comparison"]["treatment_minus_control"]["macro_f1"]
                for row in seed_rows
            ),
            "accuracy": _summary(
                row["comparison"]["treatment_minus_control"]["accuracy"]
                for row in seed_rows
            ),
            "none_stratum_macro_f1": _summary(
                row["exposure_strata"]["none"]["comparison"]
                ["treatment_minus_control"]["macro_f1"]
                for row in seed_rows
            ),
            "high_stratum_macro_f1": _summary(
                row["exposure_strata"]["high"]["comparison"]
                ["treatment_minus_control"]["macro_f1"]
                for row in seed_rows
            ),
        }
    interaction_rows = [seed_results[str(seed)]["interaction_effect"] for seed in SEEDS]
    result["weight_by_mode_interaction"] = {
        "macro_f1": _summary(
            row["difference"]["macro_f1"] for row in interaction_rows
        ),
        "accuracy": _summary(
            row["difference"]["accuracy"] for row in interaction_rows
        ),
        "none_stratum_macro_f1": _summary(
            row["exposure_strata"]["none"]["difference"]["macro_f1"]
            for row in interaction_rows
        ),
        "high_stratum_macro_f1": _summary(
            row["exposure_strata"]["high"]["difference"]["macro_f1"]
            for row in interaction_rows
        ),
    }
    return result


def run(project_root: Path, output_path: Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    source_path = root / SOURCE_RELATIVE_PATH
    source = _read_json(source_path)
    if (
        source.get("status") != "primary_signal_confirmed_but_full_gate_failed"
        or source.get("primary_signal_confirmed") is not True
        or source.get("full_adoption_confirmed") is not False
        or tuple(int(value) for value in source.get("seed_set", [])) != SEEDS
        or source.get("test") != "未执行"
        or source.get("deployment_eligible") is not False
    ):
        raise ValueError("V20r multiseed source report is stale or has an invalid scope")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results: dict[str, Any] = {}
    reference_dataset_contract: dict[str, Any] | None = None
    for seed in SEEDS:
        source_seed = dict(source["seed_results"][str(seed)])
        control_config, control_source, control_native_records = _load_native_records(
            root, source_seed["runs"]["control"]
        )
        treatment_config, treatment_source, treatment_native_records = _load_native_records(
            root, source_seed["runs"]["treatment"]
        )
        if (
            control_config.task != treatment_config.task
            or control_config.data != treatment_config.data
            or control_config.model.liquid_invalid_dt_update_mode
            != "bounded_observation"
            or treatment_config.model.liquid_invalid_dt_update_mode
            != "density_weighted"
        ):
            raise ValueError(f"V20r seed{seed} configs do not define the frozen 2x2 pair")
        dataset_contract = {
            "index_sha256": control_config.data.expected_index_sha256,
            "contract_sha256": control_config.data.expected_contract_sha256,
            "stats_sha256": control_config.data.expected_stats_sha256,
            "validation_split_name": control_config.data.validation_split_name,
            "parent_support": int(control_source["native_metrics"]["support"]),
        }
        if reference_dataset_contract is None:
            reference_dataset_contract = dataset_contract
        elif dataset_contract != reference_dataset_contract:
            raise ValueError("V20r seeds do not share one frozen validation contract")
        dataset = build_task_dataset(
            root,
            control_config,
            split_name=control_config.data.validation_split_name,
        )
        exposure_groups = _exposure_groups(dataset)
        control_cross, control_cross_records = _cross_evaluate(
            root,
            control_config,
            dataset,
            cross_mode="density_weighted",
            best_epoch=int(control_source["best_epoch"]),
            device=device,
        )
        treatment_cross, treatment_cross_records = _cross_evaluate(
            root,
            treatment_config,
            dataset,
            cross_mode="bounded_observation",
            best_epoch=int(treatment_source["best_epoch"]),
            device=device,
        )

        cells = {
            "control_weights_bounded_mode": _cell(
                control_source["native_metrics"],
                control_native_records,
                exposure_groups,
            ),
            "control_weights_density_mode": _cell(
                control_cross["metrics"],
                control_cross_records,
                exposure_groups,
            ),
            "treatment_weights_bounded_mode": _cell(
                treatment_cross["metrics"],
                treatment_cross_records,
                exposure_groups,
            ),
            "treatment_weights_density_mode": _cell(
                treatment_source["native_metrics"],
                treatment_native_records,
                exposure_groups,
            ),
        }
        contrasts = {
            "direct_mode_effect_control_weights": _contrast(
                cells["control_weights_bounded_mode"],
                cells["control_weights_density_mode"],
                control_native_records,
                control_cross_records,
            ),
            "direct_mode_effect_treatment_weights": _contrast(
                cells["treatment_weights_bounded_mode"],
                cells["treatment_weights_density_mode"],
                treatment_cross_records,
                treatment_native_records,
            ),
            "training_weight_effect_bounded_mode": _contrast(
                cells["control_weights_bounded_mode"],
                cells["treatment_weights_bounded_mode"],
                control_native_records,
                treatment_cross_records,
            ),
            "training_weight_effect_density_mode": _contrast(
                cells["control_weights_density_mode"],
                cells["treatment_weights_density_mode"],
                control_cross_records,
                treatment_native_records,
            ),
            "native_total_effect": _contrast(
                cells["control_weights_bounded_mode"],
                cells["treatment_weights_density_mode"],
                control_native_records,
                treatment_native_records,
            ),
        }
        no_exposure_checks = {
            "control_weights_mode_switch": _no_exposure_invariant(
                control_native_records,
                control_cross_records,
                exposure_groups["none"],
            ),
            "treatment_weights_mode_switch": _no_exposure_invariant(
                treatment_cross_records,
                treatment_native_records,
                exposure_groups["none"],
            ),
        }
        if not all(row["passed"] for row in no_exposure_checks.values()):
            raise ValueError(f"V20r seed{seed} violates the no-exposure mode invariant")
        interaction_effect = _interaction_effect(contrasts)
        seed_results[str(seed)] = {
            "seed": seed,
            "source_runs": {
                "control": control_source,
                "treatment": treatment_source,
            },
            "cross_evaluations": {
                "control_weights_density_mode": control_cross,
                "treatment_weights_bounded_mode": treatment_cross,
            },
            "exposure_group_parent_counts": {
                name: len(parent_ids) for name, parent_ids in exposure_groups.items()
            },
            "no_exposure_invariants": no_exposure_checks,
            "cells": cells,
            "contrasts": contrasts,
            "interaction_effect": interaction_effect,
        }

    aggregate = _aggregate(seed_results)
    output = root / (output_path or OUTPUT_RELATIVE_PATH)
    report = {
        "status": "completed",
        "schema_version": SCHEMA_VERSION,
        "experiment": "V20r multiseed 2x2 checkpoint-weight versus inference-mode diagnosis",
        "scope": "DBB specialist development validation parent-final only",
        "source_report": str(source_path),
        "source_report_sha256": sha256_file(source_path),
        "device": str(device),
        "seed_set": list(SEEDS),
        "dataset_contract": reference_dataset_contract,
        "cell_contract": {
            "control_weights_bounded_mode": "native Control",
            "control_weights_density_mode": "Control checkpoint with inference mode overridden only",
            "treatment_weights_bounded_mode": "Treatment checkpoint with inference mode overridden only",
            "treatment_weights_density_mode": "native Treatment",
        },
        "contrast_contract": {
            "direct_mode_effect_control_weights": "density minus bounded with Control tensors fixed",
            "direct_mode_effect_treatment_weights": "density minus bounded with Treatment tensors fixed",
            "training_weight_effect_bounded_mode": "Treatment minus Control tensors with bounded mode fixed",
            "training_weight_effect_density_mode": "Treatment minus Control tensors with density mode fixed",
            "native_total_effect": "native Treatment minus native Control",
            "weight_by_mode_interaction": (
                "difference-in-differences isolating whether density-trained "
                "weights use density inference differently from Control weights"
            ),
        },
        "checks": {
            "source_primary_signal_confirmed": True,
            "source_full_adoption_failed": True,
            "all_native_sample_details_reproduce_source_metrics": True,
            "all_cross_evaluations_preserve_checkpoint_tensors": True,
            "all_no_exposure_mode_switches_are_prediction_invariant": True,
            "all_seeds_share_validation_contract": True,
            "test_payload_access_count_is_zero": True,
        },
        "seed_results": seed_results,
        "aggregate": aggregate,
        "interpretation_boundary": {
            "direct_mode_contrasts_isolate_inference_dynamics_with_tensors_fixed": True,
            "training_weight_contrasts_include_the_full_20_epoch_optimization_path": True,
            "best_checkpoints_remain_validation_selected_within_each_arm": True,
            "same_validation_parents_are_not_independent_seed_samples": True,
        },
        "test": "未执行",
        "OOF": "未执行",
        "full_cascade": "未执行",
        "formal_rejection": "未执行",
        "stable": "未执行",
        "P95": "未执行",
        "deployment_eligible": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": report["status"],
        "report": str(output),
        "device": report["device"],
        "aggregate": report["aggregate"],
        "test": report["test"],
        "deployment_eligible": report["deployment_eligible"],
    }, ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the frozen V20r multiseed 2x2 cross-mode diagnosis."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    run(args.project_root, args.output)


if __name__ == "__main__":
    main()
