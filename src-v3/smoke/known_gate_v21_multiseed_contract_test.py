from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import report.known_gate_v21_multiseed as multiseed_report

from build.known_gate_v21_multiseed import (
    BASE_CONFIG,
    NEW_SEEDS,
    frozen_training_core,
    multiseed_config_payload,
    selection_protocol,
)
from radar_ln3.known_gate import (
    KNOWN_GATE_BASELINE_MULTISEED_SEEDS,
    build_known_gate_model_for_config,
    load_known_gate_config,
    sha256_file,
    summarize_known_gate_records,
)


def _state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes(order="C"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _metrics(*, weak: float, macro_f1: float, clutter_f1: float, target_recall: float) -> dict[str, Any]:
    return {
        "accuracy": 0.95,
        "macro_precision": 0.90,
        "macro_recall": 0.90,
        "macro_f1": float(macro_f1),
        "per_class": [
            {
                "class": "KNOWN_TARGET_EVIDENCE",
                "precision": 0.99,
                "recall": float(target_recall),
                "f1": 0.98,
                "support": 223,
            },
            {
                "class": "KNOWN_CLUTTER_EVIDENCE",
                "precision": float(weak),
                "recall": float(weak),
                "f1": float(clutter_f1),
                "support": 9,
            },
        ],
    }


def _bootstrap_records(seed: int) -> list[dict[str, Any]]:
    label_support = {
        "DroneTarget": 113,
        "BirdTarget": 56,
        "BalloonTarget": 54,
        "ClutterTarget": 9,
    }
    target_error_count = {42: 1, 43: 2, 44: 3}[seed]
    clutter_correct_count = {42: 6, 43: 7, 44: 8}[seed]
    rows: list[dict[str, Any]] = []
    target_cursor = 0
    for label, support in label_support.items():
        true_id = 1 if label == "ClutterTarget" else 0
        for index in range(support):
            if true_id == 0:
                pred_id = 1 if target_cursor < target_error_count else 0
                target_cursor += 1
            else:
                pred_id = 1 if index < clutter_correct_count else 0
            rows.append(
                {
                    "parent_sample_id": f"{label}|{index:03d}",
                    "label_name": label,
                    "true_id": true_id,
                    "pred_id": pred_id,
                }
            )
    return rows


def _full_report_probe(root: Path) -> int:
    base_readiness_path = root / multiseed_report.BASE_READINESS
    multiseed_readiness_path = root / multiseed_report.MULTISEED_READINESS
    base_readiness_path.parent.mkdir(parents=True, exist_ok=True)
    base_readiness_path.write_text("{}\n", encoding="utf-8")
    base_readiness_sha256 = sha256_file(base_readiness_path)
    multiseed_readiness = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21-known-gate-baseline-multiseed-training-readiness-v1",
        "seeds": list(KNOWN_GATE_BASELINE_MULTISEED_SEEDS),
        "checks": {"synthetic_full_report_probe": True},
        "base_readiness_sha256": base_readiness_sha256,
        "test": "未执行",
        "deployment_eligible": False,
    }
    multiseed_readiness_path.parent.mkdir(parents=True, exist_ok=True)
    multiseed_readiness_path.write_text(
        json.dumps(multiseed_readiness, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    multiseed_readiness_sha256 = sha256_file(multiseed_readiness_path)
    for seed in KNOWN_GATE_BASELINE_MULTISEED_SEEDS:
        config_path = root / multiseed_report.config_relative(seed)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps({"seed": seed}) + "\n", encoding="utf-8")
        artifact = root / multiseed_report.artifact_relative(seed)
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "best_model.pt").write_bytes(f"seed{seed}".encode("ascii"))
        records = _bootstrap_records(seed)
        metrics = summarize_known_gate_records(records, scope="synthetic_full_report_probe")
        readiness_sha256 = (
            base_readiness_sha256 if seed == 42 else multiseed_readiness_sha256
        )
        summary = {
            "status": "completed",
            "performance_status": "failed",
            "task_name": "known_gate",
            "variant": "baseline",
            "seed": seed,
            "train_parent_count": 986,
            "validation_parent_count": 232,
            "validation_clutter_support": 9,
            "history": [{"epoch": epoch} for epoch in range(1, 21)],
            "full_parameter_training": True,
            "all_model_parameters_trainable": True,
            "model_state_changed_from_initialization": True,
            "test_evaluation": "not_executed",
            "oof_threshold_calibration": "not_executed",
            "deployment_eligible": False,
            "training_readiness": {"sha256": readiness_sha256},
            "best_epoch": 1,
            "best_validation_metrics": {"parent_metrics": metrics},
        }
        detail = {
            "variant": "baseline",
            "epoch": 1,
            "records": {"parent": records},
        }
        (artifact / "training-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (artifact / "sample_detail_information.json").write_text(
            json.dumps(detail, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    report, selection = multiseed_report.build_report(root)
    if (
        report["status"] != "completed"
        or selection["status"] != "selected_development_champion"
        or selection["selected_seed"] != 43
        or not (root / multiseed_report.REPORT_RELATIVE).is_file()
        or not (root / multiseed_report.SELECTION_RELATIVE).is_file()
    ):
        raise AssertionError("V21 multiseed full synthetic report/selection probe failed")
    return int(selection["selected_seed"])


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    base_path = root / BASE_CONFIG
    base_payload = json.loads(base_path.read_text(encoding="utf-8"))
    base_config = load_known_gate_config(base_path)
    if base_config.train.seed != 42 or base_config.variant != "baseline":
        raise AssertionError("V21 multiseed Smoke source is not seed42 Baseline")
    base_core = frozen_training_core(base_payload)
    state_hashes: dict[int, str] = {}
    parameter_counts: dict[int, int] = {}
    with tempfile.TemporaryDirectory(prefix="v21-known-gate-multiseed-") as temporary:
        temp_root = Path(temporary)
        for seed in NEW_SEEDS:
            payload = multiseed_config_payload(base_payload, seed)
            if frozen_training_core(payload) != base_core:
                raise AssertionError(f"V21 seed{seed} differs beyond seed/path/schema")
            config_path = temp_root / f"seed{seed}.json"
            config_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            config = load_known_gate_config(config_path)
            first, reference, evidence = build_known_gate_model_for_config(root, config)
            second, _, _ = build_known_gate_model_for_config(root, config)
            if reference is not None or evidence.get("paired_baseline") is not False:
                raise AssertionError("V21 additional Baseline seed unexpectedly built a Treatment pair")
            first_hash = _state_sha256(first.state_dict())
            second_hash = _state_sha256(second.state_dict())
            if first_hash != second_hash:
                raise AssertionError(f"V21 seed{seed} initialization is not repeatable")
            if any(not parameter.requires_grad for parameter in first.parameters()):
                raise AssertionError(f"V21 seed{seed} does not train every parameter")
            state_hashes[seed] = first_hash
            parameter_counts[seed] = sum(parameter.numel() for parameter in first.parameters())
    if len(set(state_hashes.values())) != len(state_hashes):
        raise AssertionError("V21 seed43/44 initial model states unexpectedly match")
    if len(set(parameter_counts.values())) != 1:
        raise AssertionError("V21 multiseed Baseline parameter counts differ")

    candidate_metrics = {
        42: _metrics(weak=0.70, macro_f1=0.85, clutter_f1=0.72, target_recall=0.99),
        43: _metrics(weak=0.80, macro_f1=0.83, clutter_f1=0.81, target_recall=0.98),
        44: _metrics(weak=0.80, macro_f1=0.86, clutter_f1=0.79, target_recall=0.99),
    }
    winner = max(
        KNOWN_GATE_BASELINE_MULTISEED_SEEDS,
        key=lambda seed: multiseed_report._selection_key(candidate_metrics[seed], seed),
    )
    if winner != 44:
        raise AssertionError("V21 multiseed selection did not prioritize weak floor then Macro-F1")
    exact_tie = _metrics(weak=0.80, macro_f1=0.86, clutter_f1=0.80, target_recall=0.99)
    tie_winner = max(
        (42, 43), key=lambda seed: multiseed_report._selection_key(exact_tie, seed)
    )
    if tie_winner != 42:
        raise AssertionError("V21 multiseed exact tie did not prefer the lower seed")
    protocol = selection_protocol()
    if (
        protocol.get("candidate_seeds") != list(KNOWN_GATE_BASELINE_MULTISEED_SEEDS)
        or protocol.get("protocol_frozen_before_seed43_or_seed44_training") is not True
        or protocol.get("selection_is_unbiased_test_estimate") is not False
    ):
        raise AssertionError("V21 multiseed selection protocol is incomplete")
    original_replicates = multiseed_report.BOOTSTRAP_REPLICATES
    try:
        multiseed_report.BOOTSTRAP_REPLICATES = 50
        bootstrap = multiseed_report._bootstrap(
            {
                seed: {"records": _bootstrap_records(seed)}
                for seed in KNOWN_GATE_BASELINE_MULTISEED_SEEDS
            }
        )
    finally:
        multiseed_report.BOOTSTRAP_REPLICATES = original_replicates
    bootstrap_win_count = sum(
        int(row["count"]) for row in bootstrap["selection_frequency"].values()
    )
    if (
        bootstrap_win_count != 50
        or bootstrap["stratum_support"]["ClutterTarget"] != 9
        or len(bootstrap["pairwise_deltas"]) != 3
    ):
        raise AssertionError("V21 multiseed paired parent bootstrap contract failed")
    with tempfile.TemporaryDirectory(prefix="v21-known-gate-multiseed-report-") as temporary:
        original_replicates = multiseed_report.BOOTSTRAP_REPLICATES
        try:
            multiseed_report.BOOTSTRAP_REPLICATES = 50
            full_report_selected_seed = _full_report_probe(Path(temporary))
        finally:
            multiseed_report.BOOTSTRAP_REPLICATES = original_replicates
    return {
        "status": "passed",
        "schema_version": "RADAR-LN3-V21-known-gate-baseline-multiseed-smoke-v1",
        "seeds": list(KNOWN_GATE_BASELINE_MULTISEED_SEEDS),
        "seed42_reused": True,
        "new_seed_initializations_repeatable": True,
        "new_seed_initializations_distinct": True,
        "frozen_training_core_equal": True,
        "all_parameters_trainable": True,
        "trainable_parameter_count": next(iter(parameter_counts.values())),
        "selection_priority_probe_winner": winner,
        "exact_tie_lower_seed_winner": tie_winner,
        "paired_parent_bootstrap_probe_replicates": bootstrap_win_count,
        "full_report_probe_selected_seed": full_report_selected_seed,
        "clutter_validation_support": 9,
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test V21 KnownGate Baseline multiseed contracts.")
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(run(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
