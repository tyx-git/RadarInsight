"""Contracts for the V22 fixed-parent-denominator KnownGate OOF experiment."""

from __future__ import annotations

from pathlib import Path

from .known_gate_oof import FOLD_COUNT, FIXED_EPOCHS, TRAINING_SEED


V22_LOSS_MODE = "parent_final_task_ce_with_prefix_auxiliary"
V22_CLASS_WEIGHT_NORMALIZATION = "fixed_parent_count"

V21E_FOLD_CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-contract.json"
)
V21E_ASSIGNMENT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-assignment.json"
)
V21E_TREATMENT_REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment.json"
)

CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof-contract.json"
)
READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-training-readiness.json"
)
PAIRED_REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22-known-gate-fixed-parent-balanced-paired-oof.json"
)
ARTIFACT_ROOT_RELATIVE = Path(
    "artifacts/RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof"
)

CONTRACT_SCHEMA = "RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-readiness-v1"
PREDICTION_SCHEMA = "RADAR-LN3-V22-known-gate-fixed-parent-balanced-oof-prediction-v1"
PAIRED_REPORT_SCHEMA = "RADAR-LN3-V22-known-gate-fixed-parent-balanced-paired-oof-report-v1"


def config_relative(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V22 fold")
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof-fold-"
        f"{int(fold_id):02d}-treatment-seed43-config.json"
    )


def artifact_relative(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V22 fold")
    return ARTIFACT_ROOT_RELATIVE / f"fold-{int(fold_id):02d}" / "treatment"


def prediction_relative(fold_id: int) -> Path:
    return artifact_relative(fold_id) / "held-out-predictions.json"


__all__ = ["ARTIFACT_ROOT_RELATIVE", "CONTRACT_RELATIVE", "CONTRACT_SCHEMA", "FOLD_COUNT", "FIXED_EPOCHS", "PAIRED_REPORT_RELATIVE", "PAIRED_REPORT_SCHEMA", "PREDICTION_SCHEMA", "READINESS_RELATIVE", "READINESS_SCHEMA", "TRAINING_SEED", "V21E_ASSIGNMENT_RELATIVE", "V21E_FOLD_CONTRACT_RELATIVE", "V21E_TREATMENT_REPORT_RELATIVE", "V22_CLASS_WEIGHT_NORMALIZATION", "V22_LOSS_MODE", "artifact_relative", "config_relative", "prediction_relative"]