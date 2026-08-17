from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build.known_gate_v21e_multiseed import (
    NEW_TRAINING_SEEDS,
    READINESS,
    SEEDS,
    treatment_config_relative,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
)
from report.known_gate_v21e import LABELS
from report.known_gate_v21e_multiseed import aggregate_decision
from train.known_gate_v21 import _verify_contract, _verify_readiness


def _synthetic_seed_result(seed: int, *, prefix_safe: bool) -> dict[str, Any]:
    parent_delta = {
        "accuracy": 0.02,
        "macro_precision": 0.08,
        "macro_recall": 0.08,
        "macro_f1": 0.03,
        "known_target_precision": 0.01,
        "known_target_recall": 0.01,
        "known_target_f1": 0.01,
        "known_clutter_precision": 0.2,
        "known_clutter_recall": 0.1,
        "known_clutter_f1": 0.15,
        "weak_class_precision_recall_f1_floor": 0.1,
    }
    prefix_delta = dict(parent_delta)
    prefix_delta["macro_f1"] = 0.005
    prefix_delta["known_target_recall"] = 0.001 if prefix_safe else -0.001
    outcomes = {
        label: {
            "support": 1,
            "changed": 1,
            "rescued": 1 if label == "ClutterTarget" else 0,
            "harmed": 0,
            "net_rescue": 1 if label == "ClutterTarget" else 0,
        }
        for label in LABELS
    }
    return {
        "seed": seed,
        "scope_deltas": {
            "parent": parent_delta,
            "window": prefix_delta,
            "eligible_prefix": prefix_delta,
        },
        "parent_base_label_outcomes": outcomes,
        "combined_parent_net_rescue": 1,
        "gate": {
            "primary_macro_signal_passed": True,
            "paired_bootstrap_robust": True,
            "parent_direction_safety_passed": True,
            "eligible_prefix_guard_passed": prefix_safe,
        },
    }


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    readiness_path = root / READINESS
    if not readiness_path.is_file():
        raise FileNotFoundError(
            f"run build.known_gate_v21e_multiseed before Smoke: {readiness_path}"
        )
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    checks = dict(readiness.get("checks") or {})
    if (
        readiness.get("status") != "passed"
        or readiness.get("schema_version")
        != KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA
        or tuple(readiness.get("seed_set") or ()) != SEEDS
        or tuple(readiness.get("new_training_seeds") or ()) != NEW_TRAINING_SEEDS
        or not checks
        or not all(value is True for value in checks.values())
        or readiness.get("validation_used_for_readiness_gate") is not False
        or readiness.get("test") != "未执行"
        or int(readiness.get("test_payload_access_count", -1)) != 0
    ):
        raise AssertionError("V21e multiseed readiness contract failed")

    pairing: dict[str, Any] = {}
    for seed in NEW_TRAINING_SEEDS:
        config_path = root / treatment_config_relative(seed)
        binding = dict(dict(readiness.get("treatment_configs") or {}).get(str(seed)) or {})
        if (
            not config_path.is_file()
            or binding.get("sha256") != sha256_file(config_path)
            or Path(str(binding.get("path", ""))).resolve() != config_path.resolve()
        ):
            raise AssertionError(f"V21e seed{seed} config binding failed")
        config = load_known_gate_config(config_path)
        treatment, baseline, evidence = build_known_gate_model_for_config(root, config)
        if (
            config.train.seed != seed
            or baseline is None
            or evidence.get("shared_parameters_and_buffers_exactly_equal") is not True
            or evidence.get("new_fusion_columns_exactly_zero") is not True
            or int(evidence.get("new_fusion_columns", -1)) != 1
        ):
            raise AssertionError(f"V21e seed{seed} pairing failed")
        contract = _verify_contract(root, config)
        verified = _verify_readiness(root, config_path, config, contract)
        if (
            verified.get("schema_version")
            != KNOWN_GATE_V21E_MULTISEED_READINESS_SCHEMA
            or int(verified.get("paired_baseline_seed", -1)) != seed
        ):
            raise AssertionError(f"V21e seed{seed} training readiness route failed")
        pairing[str(seed)] = evidence
        del treatment, baseline

    synthetic = {
        "42": _synthetic_seed_result(42, prefix_safe=True),
        "43": _synthetic_seed_result(43, prefix_safe=False),
        "44": _synthetic_seed_result(44, prefix_safe=True),
    }
    decision = aggregate_decision(synthetic)
    if (
        decision.get("parent_final_confirmation_passed") is not True
        or decision.get("online_eligible_prefix_guard_passed") is not True
        or decision.get("advance_to_train_parent_grouped_oof") is not True
        or decision.get("online_known_gate_adoption_allowed") is not False
    ):
        raise AssertionError("V21e multiseed aggregate decision probe failed")

    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21e-multiseed-contract-smoke-v1",
        "seed_set": list(SEEDS),
        "new_training_seeds": list(NEW_TRAINING_SEEDS),
        "frozen_baseline_controls_reused": True,
        "completed_seed43_treatment_reused": True,
        "new_treatments_are_full_parameter_pairs": True,
        "pairing_evidence": pairing,
        "training_readiness_route_accepts_seed42_and_seed44": True,
        "parent_final_and_prefix_decisions_are_separate": True,
        "readiness_sha256": sha256_file(readiness_path),
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test V21e seed42/43/44 paired confirmation contracts."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
