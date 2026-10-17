"""Prepare the frozen V26a post-checkpoint open-set rejection diagnostic.

V26a deliberately adds no trainable parameter and performs no training.  It
binds a small, pre-registered threshold scan to the already frozen V26
KnownGate checkpoint.  The binary logits remain a KnownTarget-vs-KnownClutter
route; the threshold is only a semantic-decision diagnostic and never turns
UnknownTarget into a third learned class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from build.micro_doppler_v17 import _write_json_new_or_identical
from radar_ln3.known_gate import load_known_gate_config, sha256_file


V26_CONFIG = Path(
    "config/RADAR-LN3/"
    "RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-baseline-"
    "seed43-config.json"
)
V26_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26-known-gate-known-only-open-set-baseline-training-"
    "readiness.json"
)
V26_TRAINING_SUMMARY = Path(
    "artifacts/RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-"
    "baseline-seed43-formal/training-summary.json"
)
V26_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V26-cq08-liquid-known-gate-known-only-open-set-"
    "baseline-seed43-formal/best_model.pt"
)
V26_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26-known-gate-known-only-open-set-development.json"
)
V26_EXPERIMENT_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-known-only-open-set-baseline-contract.json"
)
V26_FEATURE_MANIFEST = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-known-only-open-set-baseline-"
    "feature-cache-manifest.json"
)
V26_AUDIT_VALIDATION_CACHE = Path(
    "data/processed/"
    "RADAR-LN3-V26-cq08-known-gate-val-mixed-open-set-course-cache.npz"
)
V26A_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26a-cq08-known-gate-frozen-open-set-rejection-contract.json"
)
V26A_SELECTION = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-selection.json"
)
V26A_DOCUMENT = Path(
    "src-v3/docs/radar_ln3_v26a_frozen_open_set_rejection_experiment.md"
)

V26A_SCHEMA_VERSION = "RADAR-LN3-V26a-known-gate-frozen-open-set-rejection-v1"
V26A_VARIANT = "v26a_frozen_known_gate_threshold_rejection"
V26A_THRESHOLD_GRID = tuple(round(index / 100.0, 2) for index in range(50, 100))
V26A_BOOTSTRAP_REPLICATES = 5000
V26A_BOOTSTRAP_SEED = 26043

# These are screening constraints, not deployment claims.  They preserve the
# current known-route safety intent while making the small validation Unknown
# support explicit: at least 3 of 15 Unknown parents must be rejected before
# a threshold is called a useful diagnostic candidate.
V26A_KNOWN_ACCEPTANCE_FLOOR = 0.95
V26A_KNOWN_TARGET_RECALL_FLOOR = 0.98
V26A_KNOWN_CLUTTER_RECALL_FLOOR = 8.0 / 9.0
V26A_MIN_UNKNOWN_REJECTION_COUNT = 3


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _required_files(root: Path) -> dict[str, Path]:
    return {
        "config": root / V26_CONFIG,
        "readiness": root / V26_READINESS,
        "training_summary": root / V26_TRAINING_SUMMARY,
        "checkpoint": root / V26_CHECKPOINT,
        "v26_report": root / V26_REPORT,
        "v26_experiment_contract": root / V26_EXPERIMENT_CONTRACT,
        "v26_feature_manifest": root / V26_FEATURE_MANIFEST,
        "v26_validation_cache": root / V26_AUDIT_VALIDATION_CACHE,
        "document": root / V26A_DOCUMENT,
    }


def build(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    files = _required_files(root)
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "V26a requires the completed frozen V26 artifact; missing="
            + ",".join(missing)
        )

    config = load_known_gate_config(files["config"])
    readiness = _load_json(files["readiness"])
    summary = _load_json(files["training_summary"])
    v26_contract = _load_json(files["v26_experiment_contract"])

    if config.variant != "v26_known_only_open_set_baseline":
        raise ValueError("V26a source config is not the frozen V26 Known-only baseline")
    if readiness.get("status") != "passed" or readiness.get("test_payload_access_count") != 0:
        raise ValueError("V26a source readiness is not a sealed successful V26 readiness")
    if (
        summary.get("status") != "completed"
        or summary.get("variant") != config.variant
        or len(list(summary.get("history") or ())) != 20
        or summary.get("test_evaluation") != "not_executed"
        or summary.get("deployment_eligible") is not False
    ):
        raise ValueError("V26a source training summary is not a completed sealed 20-epoch run")
    checkpoint_epoch = int(summary.get("best_epoch", -1))
    if checkpoint_epoch < 1 or checkpoint_epoch > 20:
        raise ValueError("V26a source checkpoint epoch is invalid")
    open_set_contract = dict(v26_contract.get("open_set_audit_contract") or {})
    if (
        open_set_contract.get("unknown_is_model_class") is not False
        or open_set_contract.get("unknown_is_training_target") is not False
        or open_set_contract.get("rejection_head") != "not_available_in_binary_known_gate"
    ):
        raise ValueError("V26 source contract does not preserve strict open-set semantics")

    source_hashes = {name: sha256_file(path) for name, path in files.items()}
    contract = {
        "schema_version": V26A_SCHEMA_VERSION,
        "experiment_name": "V26a frozen KnownGate post-checkpoint open-set rejection diagnostic",
        "internal_variant": V26A_VARIANT,
        "scope": "semantic-decision rejection only; no model retraining and no L2/L3 deployment claim",
        "source_v26": {
            "config": _relative(root, files["config"]),
            "config_sha256": source_hashes["config"],
            "readiness": _relative(root, files["readiness"]),
            "readiness_sha256": source_hashes["readiness"],
            "training_summary": _relative(root, files["training_summary"]),
            "training_summary_sha256": source_hashes["training_summary"],
            "checkpoint": _relative(root, files["checkpoint"]),
            "checkpoint_sha256": source_hashes["checkpoint"],
            "checkpoint_epoch": checkpoint_epoch,
            "development_report": _relative(root, files["v26_report"]),
            "development_report_sha256": source_hashes["v26_report"],
            "experiment_contract": _relative(root, files["v26_experiment_contract"]),
            "experiment_contract_sha256": source_hashes["v26_experiment_contract"],
            "feature_manifest": _relative(root, files["v26_feature_manifest"]),
            "feature_manifest_sha256": source_hashes["v26_feature_manifest"],
            "validation_audit_cache": _relative(root, files["v26_validation_cache"]),
            "validation_audit_cache_sha256": source_hashes["v26_validation_cache"],
        },
        "variable": {
            "name": "frozen_max_known_route_confidence_threshold",
            "score": "max(softmax(KNOWN_TARGET_EVIDENCE), softmax(KNOWN_CLUTTER_EVIDENCE))",
            "score_is_true_knownness_probability": False,
            "control": "no_rejection; preserve frozen V26 argmax route",
            "threshold_grid": list(V26A_THRESHOLD_GRID),
            "comparison_unit": "parent_final_and_causal_prefix_action",
            "model_parameters_changed": False,
            "training_executed": False,
            "feature_cache_changed": False,
            "dt_semantics_changed": False,
            "normalization_changed": False,
        },
        "decision_semantics": {
            "ineligible_event_precedence": "PENDING_ABSTAIN",
            "eligible_low_score_action": "UNKNOWN_REJECTED",
            "eligible_high_score_action": "frozen_known_gate_argmax_route",
            "unknown_is_learned_class": False,
            "pending_is_unknown_rejection_success": False,
        },
        "threshold_selection": {
            "selection_split": "validation_mixed_open_set_audit",
            "selection_artifact": _relative(root, root / V26A_SELECTION),
            "test_used_for_selection": False,
            "objective": "maximize validation Unknown parent rejection under known safety floors",
            "known_acceptance_floor": V26A_KNOWN_ACCEPTANCE_FLOOR,
            "known_target_recall_floor": V26A_KNOWN_TARGET_RECALL_FLOOR,
            "known_clutter_recall_floor": V26A_KNOWN_CLUTTER_RECALL_FLOOR,
            "minimum_unknown_rejection_count": V26A_MIN_UNKNOWN_REJECTION_COUNT,
            "tie_break": [
                "higher_unknown_rejection_count",
                "higher_known_acceptance_rate",
                "lower_threshold",
            ],
            "fallback_if_no_useful_threshold": "select the highest threshold satisfying known safety floors for a stress audit; mark no_useful_threshold",
            "status_if_no_useful_threshold": "no_feasible_useful_threshold",
        },
        "bootstrap": {
            "method": "stratified parent bootstrap; known labels and Unknown parents resampled separately",
            "replicates": V26A_BOOTSTRAP_REPLICATES,
            "seed": V26A_BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "test_protocol": {
            "test_cache_materialized_during_prepare": False,
            "test_forward_allowed_only_after_threshold_selection": True,
            "test_labels_used_for_threshold_selection": False,
            "test_result_is_final_audit_only": True,
        },
        "source_document": {
            "path": _relative(root, files["document"]),
            "sha256": source_hashes["document"],
        },
        "deployment_eligible": False,
    }
    write_status = _write_json_new_or_identical(root / V26A_CONTRACT, contract)
    return {
        "status": "passed",
        "schema_version": V26A_SCHEMA_VERSION,
        "variant": V26A_VARIANT,
        "contract": str(root / V26A_CONTRACT),
        "contract_sha256": sha256_file(root / V26A_CONTRACT),
        "contract_write_status": write_status,
        "source_checkpoint": str(files["checkpoint"]),
        "source_checkpoint_sha256": source_hashes["checkpoint"],
        "source_checkpoint_epoch": checkpoint_epoch,
        "threshold_count": len(V26A_THRESHOLD_GRID),
        "test": "not_executed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the V26a frozen open-set rejection contract.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
