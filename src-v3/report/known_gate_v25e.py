"""Paired V25a-versus-V25e development report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from build.known_gate_v25a import ARTIFACT as CONTROL_ARTIFACT, CONFIG as CONTROL_CONFIG
from build.known_gate_v25e import ARTIFACT, CONFIG, EXPERIMENT_CONTRACT, READINESS
from radar_ln3.known_gate import (
    KNOWN_GATE_CLASS_NAMES,
    KNOWN_GATE_V25E_READINESS_SCHEMA,
    load_known_gate_config,
    sha256_file,
)
from radar_ln3.known_gate_features import (
    V25A_GRADIENT_TELEMETRY_VARIANT,
    V25E_SHARED_GRADIENT_PROJECTION_VARIANT,
)
from report.known_gate_v24c import (
    _load_run,
    _metrics,
    _ordered_pair,
    _paired_bootstrap,
    _scope_safety,
    _write_json,
)
from train.known_gate_v21 import _verify_contract, _verify_readiness


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V25e-known-gate-course-shared-gradient-conflict-projection-"
    "paired-development.json"
)
EXPECTED_EPOCHS = tuple(range(1, 21))


def _per_class(metrics: Mapping[str, Any], name: str) -> dict[str, Any]:
    for row in list(metrics.get("per_class") or ()):
        if str(row["class"]) == name:
            return dict(row)
    raise ValueError(f"V25e metric lacks {name}")


def _deltas(control: Mapping[str, Any], treatment: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for scope in ("parent_final", "eligible_prefix", "parent_balanced_eligible_prefix", "window"):
        before, after = dict(control[scope]), dict(treatment[scope])
        bk, ak = _per_class(before, KNOWN_GATE_CLASS_NAMES[0]), _per_class(after, KNOWN_GATE_CLASS_NAMES[0])
        bc, ac = _per_class(before, KNOWN_GATE_CLASS_NAMES[1]), _per_class(after, KNOWN_GATE_CLASS_NAMES[1])
        result[scope] = {
            "accuracy_delta": float(after["accuracy"] - before["accuracy"]),
            "macro_precision_delta": float(after["macro_precision"] - before["macro_precision"]),
            "macro_recall_delta": float(after["macro_recall"] - before["macro_recall"]),
            "macro_f1_delta": float(after["macro_f1"] - before["macro_f1"]),
            "known_target_recall_delta": float(ak["recall"] - bk["recall"]),
            "known_target_precision_delta": float(ak["precision"] - bk["precision"]),
            "clutter_precision_delta": float(ac["precision"] - bc["precision"]),
            "clutter_recall_delta": float(ac["recall"] - bc["recall"]),
            "clutter_f1_delta": float(ac["f1"] - bc["f1"]),
            "clutter_support": int(ac["support"]),
        }
    return result


def _directional(control: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]], *, scope: str) -> dict[str, int]:
    before, after = _ordered_pair(control, treatment, scope=scope)
    result = {"known_to_clutter": 0, "clutter_to_known": 0, "rescued": 0, "harmed": 0, "prediction_changed": 0}
    for left, right in zip(before, after, strict=True):
        if int(left["pred_id"]) == int(right["pred_id"]):
            continue
        result["prediction_changed"] += 1
        if int(left["pred_id"]) == 0 and int(right["pred_id"]) == 1:
            result["known_to_clutter"] += 1
        if int(left["pred_id"]) == 1 and int(right["pred_id"]) == 0:
            result["clutter_to_known"] += 1
        left_ok = int(left["pred_id"]) == int(left["true_id"])
        right_ok = int(right["pred_id"]) == int(right["true_id"])
        result["rescued"] += int(not left_ok and right_ok)
        result["harmed"] += int(left_ok and not right_ok)
    return result


def _orders(summary: Mapping[str, Any]) -> list[str]:
    rows = list(dict(summary.get("dataloader_contract") or {}).get("epoch_parent_order_hashes") or ())
    if len(rows) != 20:
        raise ValueError("V25e summary lacks all parent-order hashes")
    return [str(row.get("sha256", "")) for row in rows]


def _telemetry(summary: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = dict(summary.get("shared_gradient_projection_telemetry") or {})
    scope = dict(telemetry.get("by_scope") or {})
    if (
        telemetry.get("projected_modules") != ["tr_encoder", "fusion", "liquid"]
        or telemetry.get("unchanged_modules") != ["rd_encoder", "classifier"]
        or int(scope.get("batch_count", -1)) != 9860
        or int(scope.get("mixed_known_clutter_batch_count", -1)) != 638
        or int(scope.get("pure_batch_count", -1)) != 9222
        or int(scope.get("projection_batch_count", -1)) != 638
        or telemetry.get("rd_and_classifier_gradients_exactly_unchanged") is not True
        or any(int(dict(scope.get("projection_count") or {}).get(name, 0)) <= 0 for name in ("tr_encoder", "fusion", "liquid"))
    ):
        raise ValueError("V25e training summary lacks valid projection telemetry")
    return telemetry


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config_path, readiness_path, contract_path = root / CONFIG, root / READINESS, root / EXPERIMENT_CONTRACT
    if not all(path.is_file() for path in (config_path, readiness_path, contract_path)):
        raise FileNotFoundError("V25e config/readiness/contract is incomplete")
    config = load_known_gate_config(config_path)
    contract = _verify_contract(root, config)
    readiness = _verify_readiness(root, config_path, config, contract)
    if config.variant != V25E_SHARED_GRADIENT_PROJECTION_VARIANT or readiness.get("schema_version") != KNOWN_GATE_V25E_READINESS_SCHEMA:
        raise ValueError("V25e config/readiness identity differs")
    control = _load_run(root, artifact=CONTROL_ARTIFACT, config_path=root / CONTROL_CONFIG, variant=V25A_GRADIENT_TELEMETRY_VARIANT, readiness_sha256=None)
    treatment = _load_run(root, artifact=ARTIFACT, config_path=config_path, variant=V25E_SHARED_GRADIENT_PROJECTION_VARIANT, readiness_sha256=str(readiness["sha256"]))
    cs, ts = dict(control["summary"]), dict(treatment["summary"])
    pair = dict(ts.get("paired_initialization") or {})
    source_initial = str(readiness["source_v25a_initial_model_state_sha256"])
    if (
        str(cs.get("initial_model_state_sha256", "")) != source_initial
        or str(ts.get("initial_model_state_sha256", "")) != source_initial
        or pair.get("shared_gradient_projection_pair") is not True
        or pair.get("sealed_v25a_initial_model_state_reproduced") is not True
        or int(pair.get("new_fusion_columns", -1)) != 0
        or _orders(cs) != _orders(ts)
        or _orders(ts) != [str(dict(readiness["source_v25a_epoch_parent_order_hashes"])[str(epoch)]) for epoch in EXPECTED_EPOCHS]
    ):
        raise ValueError("V25e does not prove frozen V25a pairing")
    telemetry = _telemetry(ts)
    cm, tm = _metrics(control["rows"]), _metrics(treatment["rows"])
    deltas = _deltas(cm, tm)
    clutter = _per_class(tm["parent_final"], KNOWN_GATE_CLASS_NAMES[1])
    known = _per_class(tm["parent_final"], KNOWN_GATE_CLASS_NAMES[0])
    prefix = deltas["parent_balanced_eligible_prefix"]
    acceptance = {
        "parent_final_clutter_recall_is_9_of_9": int(clutter["support"]) == 9 and float(clutter["recall"]) == 1.0,
        "parent_final_known_target_recall_is_223_of_223": int(known["support"]) == 223 and float(known["recall"]) == 1.0,
        "parent_final_clutter_precision_at_least_0_90": float(clutter["precision"]) >= 0.90,
        "parent_balanced_prefix_clutter_precision_not_lower": prefix["clutter_precision_delta"] >= 0.0,
        "parent_balanced_prefix_clutter_recall_not_lower": prefix["clutter_recall_delta"] >= 0.0,
        "parent_balanced_prefix_clutter_f1_improves_at_least_2pp": prefix["clutter_f1_delta"] >= 0.02,
        "parent_balanced_prefix_known_target_recall_not_lower": prefix["known_target_recall_delta"] >= 0.0,
        "parent_balanced_prefix_macro_f1_not_lower": prefix["macro_f1_delta"] >= 0.0,
        "projection_mechanism_proved": True,
    }
    result = {
        "status": "completed_development_paired_gradient_projection_experiment",
        "schema_version": "RADAR-LN3-V25e-known-gate-course-shared-gradient-conflict-projection-paired-development-v1",
        "scope": "V25e development KnownGate L1 gradient intervention; no OOF, Unknown, test, L3, cascade, rejection, stable or P95",
        "control": {"variant": V25A_GRADIENT_TELEMETRY_VARIANT, "summary_path": control["summary_path"], "summary_sha256": control["summary_sha256"], "best_epoch": int(cs["best_epoch"])},
        "treatment": {"variant": V25E_SHARED_GRADIENT_PROJECTION_VARIANT, "summary_path": treatment["summary_path"], "summary_sha256": treatment["summary_sha256"], "best_epoch": int(ts["best_epoch"])},
        "paired_integrity": {"initial_model_state_exactly_equal": True, "all_twenty_parent_orders_exactly_equal": True, "projected_modules": ["tr_encoder", "fusion", "liquid"], "unchanged_modules": ["rd_encoder", "classifier"], "new_fusion_columns": 0, "validation_support": {"known_target": 223, "clutter": 9}},
        "projection_telemetry": telemetry,
        "metrics": {"control": cm, "treatment": tm, "delta": deltas},
        "directional_changes": {scope: _directional(control["rows"]["parent" if scope == "parent_final" else scope], treatment["rows"]["parent" if scope == "parent_final" else scope], scope=scope) for scope in ("parent_final", "eligible_prefix", "window")},
        "paired_parent_bootstrap": _paired_bootstrap(control["rows"]["parent"], treatment["rows"]["parent"]),
        "eligible_prefix_safety": _scope_safety(cm, tm),
        "pre_registered_acceptance": acceptance,
        "passes_all_pre_registered_development_gates": all(acceptance.values()),
        "validation_used_for_threshold_tuning": False,
        "OOF": "not_executed", "test": "not_executed", "formal_rejection": "not_executed", "full_cascade": "not_executed", "stable": "not_executed", "P95": "not_executed", "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, result)
    return {"status": result["status"], "report": str(root / OUTPUT), "passes_all_pre_registered_development_gates": result["passes_all_pre_registered_development_gates"], "test": "not_executed"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate V25e against V25a control.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
