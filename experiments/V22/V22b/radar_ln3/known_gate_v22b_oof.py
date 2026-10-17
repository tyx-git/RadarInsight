"""Contracts for the V22b RD temporal late-readout paired OOF experiment."""

from __future__ import annotations

from pathlib import Path

from .known_gate_oof import FOLD_COUNT, FIXED_EPOCHS, TRAINING_SEED


V22B_OOF_VARIANT = "v22b_rd_temporal_late_readout"
V22B_OOF_CONFIG_SCHEMA = "RADAR-LN3-V22b-oof-config-v1"
V22B_OOF_CONTRACT_SCHEMA = (
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-contract-v1"
)
V22B_OOF_READINESS_SCHEMA = (
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-readiness-v1"
)
V22B_OOF_PREDICTION_SCHEMA = (
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-prediction-v1"
)
V22B_OOF_REPORT_SCHEMA = (
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-paired-oof-report-v1"
)
V22B_OOF_CHECKPOINT_SCHEMA = (
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-checkpoint-v1"
)

V21E_FOLD_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-contract.json"
)
V21E_ASSIGNMENT = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-assignment.json"
)
V21E_CONTROL_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment.json"
)
V22B_OOF_CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V22b-cq08-known-gate-rd-temporal-late-readout-oof-contract.json"
)
V22B_OOF_READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-oof-training-readiness.json"
)
V22B_OOF_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V22b-known-gate-rd-temporal-late-readout-paired-oof.json"
)
V22B_OOF_ARTIFACT_ROOT = Path(
    "artifacts/"
    "RADAR-LN3-V22b-cq08-known-gate-rd-temporal-late-readout-paired-oof"
)


def config_relative(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V22b OOF fold")
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V22b-cq08-known-gate-rd-temporal-late-readout-oof-fold-"
        f"{int(fold_id):02d}-treatment-seed43-config.json"
    )


def artifact_relative(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V22b OOF fold")
    return V22B_OOF_ARTIFACT_ROOT / f"fold-{int(fold_id):02d}" / "treatment"


def prediction_relative(fold_id: int) -> Path:
    return artifact_relative(fold_id) / "held-out-predictions.json"


__all__ = ["FOLD_COUNT", "FIXED_EPOCHS", "TRAINING_SEED", "V21E_ASSIGNMENT", "V21E_CONTROL_REPORT", "V21E_FOLD_CONTRACT", "V22B_OOF_ARTIFACT_ROOT", "V22B_OOF_CHECKPOINT_SCHEMA", "V22B_OOF_CONFIG_SCHEMA", "V22B_OOF_CONTRACT", "V22B_OOF_CONTRACT_SCHEMA", "V22B_OOF_PREDICTION_SCHEMA", "V22B_OOF_READINESS", "V22B_OOF_READINESS_SCHEMA", "V22B_OOF_REPORT", "V22B_OOF_REPORT_SCHEMA", "V22B_OOF_VARIANT", "artifact_relative", "config_relative", "prediction_relative"]