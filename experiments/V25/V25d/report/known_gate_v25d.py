"""Paired V25a-versus-V25d development report for present-class-mean loss."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


from build.known_gate_v25a import ARTIFACT as CONTROL_ARTIFACT
from build.known_gate_v25a import CONFIG as CONTROL_CONFIG
from build.known_gate_v25d import ARTIFACT, CONFIG, EXPERIMENT_CONTRACT, READINESS
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION,
    KNOWN_GATE_V25D_READINESS_SCHEMA,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    V25A_GRADIENT_TELEMETRY_VARIANT,
    V25D_PRESENT_CLASS_MEAN_VARIANT,
)
from report.known_gate_strict_same_second_skip import (
    _load_run,
    _metrics,
    _ordered_pair,
    _paired_bootstrap,
    _scope_safety,
    _write_json,
)
from train.known_gate_common import _verify_contract, _verify_readiness


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25d-known-gate-course-present-class-mean-paired-development.json"
)
EXPECTED_EPOCHS = tuple(range(1, 21))
EXPECTED_TOTAL_STEPS = 9860


def _per_class(metrics: Mapping[str, Any], class_name: str) -> dict[str, Any]:
    rows = {
        str(row["class"]): dict(row)
        for row in list(metrics.get("per_class") or ())
    }
    try:
        return rows[class_name]
    except KeyError as exc:
        raise ValueError(f"V25d metric lacks {class_name}") from exc


def _delta_metrics(
    control: Mapping[str, Any], treatment: Mapping[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in (
        "parent_final",
        "eligible_prefix",
        "parent_balanced_eligible_prefix",
        "window",
    ):
        before = dict(control[scope])
        after = dict(treatment[scope])
        before_clutter = _per_class(before, KNOWN_GATE_CLASS_NAMES[1])
        after_clutter = _per_class(after, KNOWN_GATE_CLASS_NAMES[1])
        before_known = _per_class(before, KNOWN_GATE_CLASS_NAMES[0])
        after_known = _per_class(after, KNOWN_GATE_CLASS_NAMES[0])
        result[scope] = {
            "accuracy_delta": float(after["accuracy"] - before["accuracy"]),
            "macro_precision_delta": float(
                after["macro_precision"] - before["macro_precision"]
            ),
            "macro_recall_delta": float(after["macro_recall"] - before["macro_recall"]),
            "macro_f1_delta": float(after["macro_f1"] - before["macro_f1"]),
            "known_target_precision_delta": float(
                after_known["precision"] - before_known["precision"]
            ),
            "known_target_recall_delta": float(
                after_known["recall"] - before_known["recall"]
            ),
            "clutter_precision_delta": float(
                after_clutter["precision"] - before_clutter["precision"]
            ),
            "clutter_recall_delta": float(
                after_clutter["recall"] - before_clutter["recall"]
            ),
            "clutter_f1_delta": float(after_clutter["f1"] - before_clutter["f1"]),
            "clutter_support": int(after_clutter["support"]),
        }
    return result


def _directional_changes(
    control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]], *, scope: str
) -> dict[str, int]:
    before, after = _ordered_pair(control, treatment, scope=scope)
    result = {
        "known_to_clutter": 0,
        "clutter_to_known": 0,
        "rescued": 0,
        "harmed": 0,
        "prediction_changed": 0,
    }
    for left, right in zip(before, after, strict=True):
        if int(left["pred_id"]) == int(right["pred_id"]):
            continue
        result["prediction_changed"] += 1
        if int(left["pred_id"]) == 0 and int(right["pred_id"]) == 1:
            result["known_to_clutter"] += 1
        if int(left["pred_id"]) == 1 and int(right["pred_id"]) == 0:
            result["clutter_to_known"] += 1
        left_correct = int(left["pred_id"]) == int(left["true_id"])
        right_correct = int(right["pred_id"]) == int(right["true_id"])
        if not left_correct and right_correct:
            result["rescued"] += 1
        if left_correct and not right_correct:
            result["harmed"] += 1
    return result


def _mass_telemetry(summary: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = dict(summary.get("present_class_mean_loss_mass_telemetry") or {})
    scopes = dict(telemetry.get("by_scope") or {})
    final = dict(scopes.get("final") or {})
    prefix = dict(scopes.get("prefix") or {})
    if (
        telemetry.get("normalization") != KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION
        or telemetry.get("class_names") != list(KNOWN_GATE_CLASS_NAMES)
        or telemetry.get("expected_mixed_batch_mass") != [0.5, 0.5]
        or int(final.get("contributing_batch_count", -1)) != EXPECTED_TOTAL_STEPS
        or int(final.get("mixed_batch_count", -1)) <= 0
        or final.get("mixed_batches_exactly_equal_class_mass") is not True
        or final.get("pure_batches_have_single_class_mass") is not True
        or int(prefix.get("contributing_batch_count", -1)) <= 0
        or prefix.get("mixed_batches_exactly_equal_class_mass") is not True
        or prefix.get("pure_batches_have_single_class_mass") is not True
    ):
        raise ValueError("V25d training summary lacks valid present-class-mean mechanism telemetry")
    return telemetry


def _orders(summary: Mapping[str, Any]) -> list[str]:
    rows = list(dict(summary.get("dataloader_contract") or {}).get("epoch_parent_order_hashes") or ())
    if len(rows) != len(EXPECTED_EPOCHS):
        raise ValueError("V25d summary lacks all twenty parent-order hashes")
    return [str(row.get("sha256", "")) for row in rows]


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path = root / CONFIG
    readiness_path = root / READINESS
    contract_path = root / EXPERIMENT_CONTRACT
    if not all(path.is_file() for path in (config_path, readiness_path, contract_path)):
        raise FileNotFoundError("V25d config/readiness/contract is incomplete; run prepare first")
    config = load_known_gate_config(config_path)
    contract_evidence = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract_evidence)
    if (
        config.variant != V25D_PRESENT_CLASS_MEAN_VARIANT
        or config.train.class_weight_normalization
        != KNOWN_GATE_PRESENT_CLASS_MEAN_NORMALIZATION
        or readiness.get("schema_version") != KNOWN_GATE_V25D_READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or sha256_file(contract_path) != config.data.expected_feature_experiment_contract_sha256
    ):
        raise ValueError("V25d config/readiness identity differs")
    control = _load_run(
        root,
        artifact=CONTROL_ARTIFACT,
        config_path=root / CONTROL_CONFIG,
        variant=V25A_GRADIENT_TELEMETRY_VARIANT,
        readiness_sha256=None,
    )
    treatment = _load_run(
        root,
        artifact=ARTIFACT,
        config_path=config_path,
        variant=V25D_PRESENT_CLASS_MEAN_VARIANT,
        readiness_sha256=str(readiness["sha256"]),
    )
    control_summary = dict(control["summary"])
    treatment_summary = dict(treatment["summary"])
    source_initial = str(readiness["source_v25a_initial_model_state_sha256"])
    pairing = dict(treatment_summary.get("paired_initialization") or {})
    if (
        str(control_summary.get("initial_model_state_sha256", "")) != source_initial
        or str(treatment_summary.get("initial_model_state_sha256", "")) != source_initial
        or pairing.get("loss_only_pair") is not True
        or pairing.get("present_class_mean_loss_only_pair") is not True
        or pairing.get("sealed_v25a_initial_model_state_reproduced") is not True
        or int(pairing.get("new_fusion_columns", -1)) != 0
        or _orders(control_summary) != _orders(treatment_summary)
        or _orders(treatment_summary)
        != [
            str(dict(readiness["source_v25a_epoch_parent_order_hashes"])[str(epoch)])
            for epoch in EXPECTED_EPOCHS
        ]
    ):
        raise ValueError("V25d does not prove frozen V25a loss-only pairing")
    telemetry = _mass_telemetry(treatment_summary)
    control_metrics = _metrics(control["rows"])
    treatment_metrics = _metrics(treatment["rows"])
    deltas = _delta_metrics(control_metrics, treatment_metrics)
    parent_bootstrap = _paired_bootstrap(
        control["rows"]["parent"], treatment["rows"]["parent"]
    )
    safety = _scope_safety(control_metrics, treatment_metrics)
    parent_treatment_clutter = _per_class(
        treatment_metrics["parent_final"], KNOWN_GATE_CLASS_NAMES[1]
    )
    parent_treatment_known = _per_class(
        treatment_metrics["parent_final"], KNOWN_GATE_CLASS_NAMES[0]
    )
    prefix_delta = deltas["parent_balanced_eligible_prefix"]
    acceptance = {
        "parent_final_clutter_recall_is_9_of_9": (
            int(parent_treatment_clutter["support"]) == 9
            and abs(float(parent_treatment_clutter["recall"]) - 1.0) <= 1.0e-12
        ),
        "parent_final_known_target_recall_is_223_of_223": (
            int(parent_treatment_known["support"]) == 223
            and abs(float(parent_treatment_known["recall"]) - 1.0) <= 1.0e-12
        ),
        "parent_final_clutter_precision_at_least_0_90": (
            float(parent_treatment_clutter["precision"]) >= 0.90
        ),
        "parent_balanced_prefix_clutter_precision_not_lower": (
            prefix_delta["clutter_precision_delta"] >= 0.0
        ),
        "parent_balanced_prefix_clutter_recall_not_lower": (
            prefix_delta["clutter_recall_delta"] >= 0.0
        ),
        "parent_balanced_prefix_clutter_f1_improves_at_least_2pp": (
            prefix_delta["clutter_f1_delta"] >= 0.02
        ),
        "parent_balanced_prefix_known_target_recall_not_lower": (
            prefix_delta["known_target_recall_delta"] >= 0.0
        ),
        "parent_balanced_prefix_macro_f1_not_lower": (
            prefix_delta["macro_f1_delta"] >= 0.0
        ),
        "mixed_loss_mass_mechanism_proved": True,
    }
    result = {
        "status": "completed_development_paired_loss_only_experiment",
        "schema_version": "RADAR-LN3-V25d-known-gate-course-present-class-mean-paired-development-v1",
        "scope": "V25d development KnownGate L1 loss-only experiment; no OOF, test, Unknown, L3, full cascade, rejection, stable or P95",
        "control": {
            "variant": V25A_GRADIENT_TELEMETRY_VARIANT,
            "summary_path": control["summary_path"],
            "summary_sha256": control["summary_sha256"],
            "checkpoint_path": control["checkpoint_path"],
            "checkpoint_sha256": control["checkpoint_sha256"],
            "best_epoch": int(control_summary["best_epoch"]),
        },
        "treatment": {
            "variant": V25D_PRESENT_CLASS_MEAN_VARIANT,
            "summary_path": treatment["summary_path"],
            "summary_sha256": treatment["summary_sha256"],
            "checkpoint_path": treatment["checkpoint_path"],
            "checkpoint_sha256": treatment["checkpoint_sha256"],
            "best_epoch": int(treatment_summary["best_epoch"]),
            "class_weight_normalization": config.train.class_weight_normalization,
        },
        "paired_integrity": {
            "initial_model_state_exactly_equal": True,
            "all_twenty_parent_orders_exactly_equal": True,
            "loss_only_pair": True,
            "new_fusion_columns": 0,
            "validation_support": {
                "known_target": 223,
                "clutter": 9,
            },
        },
        "loss_mass_mechanism": telemetry,
        "metrics": {
            "control": control_metrics,
            "treatment": treatment_metrics,
            "delta": deltas,
        },
        "directional_changes": {
            scope: _directional_changes(
                control["rows"]["parent" if scope == "parent_final" else scope],
                treatment["rows"]["parent" if scope == "parent_final" else scope],
                scope=scope,
            )
            for scope in ("parent_final", "eligible_prefix", "window")
        },
        "paired_parent_bootstrap": parent_bootstrap,
        "eligible_prefix_safety": safety,
        "pre_registered_acceptance": acceptance,
        "passes_all_pre_registered_development_gates": all(acceptance.values()),
        "validation_used_for_threshold_tuning": False,
        "OOF": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, result)
    return {
        "status": result["status"],
        "report": str(root / OUTPUT),
        "passes_all_pre_registered_development_gates": result[
            "passes_all_pre_registered_development_gates"
        ],
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V25d present-class-mean loss against V25a control."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()