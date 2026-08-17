"""Frozen paths and schemas for exploratory V21f paired KnownGate OOF."""

from __future__ import annotations

from pathlib import Path

from .known_gate_v21e_oof import (
    ASSIGNMENT_RELATIVE,
    ASSIGNMENT_SEED,
    FOLD_CONTRACT_RELATIVE as V21E_FOLD_CONTRACT_RELATIVE,
    FOLD_COUNT,
    FIXED_EPOCHS,
    INDEX_RELATIVE,
    TRAINING_SEED,
    artifact_relative as v21e_artifact_relative,
    config_relative as v21e_config_relative,
    fold_record,
    load_fold_contract,
    prediction_relative as v21e_prediction_relative,
)


ARM = "treatment"
V21E_ARM = "treatment"

V21E_COURSE_CACHE_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-train-causal-course-consistency-cache.npz"
)
V21E_FEATURE_CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-causal-course-consistency-contract.json"
)
V21E_FEATURE_MANIFEST_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-causal-course-consistency-"
    "feature-cache-manifest.json"
)
V21E_TREATMENT_REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment.json"
)
V21A_SPECTRAL_CACHE_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21a-cq08-known-gate-train-interframe-spectral-cache.npz"
)
V21A_FEATURE_CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21a-cq08-known-gate-interframe-spectral-contract.json"
)
V21A_FEATURE_MANIFEST_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21a-cq08-known-gate-interframe-spectral-cache-manifest.json"
)

COMBINED_CACHE_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21f-cq08-known-gate-train-course-interframe-cache.npz"
)
EXPERIMENT_CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21f-cq08-known-gate-course-interframe-oof-contract.json"
)
FEATURE_MANIFEST_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21f-cq08-known-gate-course-interframe-cache-manifest.json"
)
OOF_CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21f-cq08-known-gate-course-interframe-oof-fold-contract.json"
)
READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21f-known-gate-course-interframe-oof-training-readiness.json"
)
ATTRIBUTION_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21f-spectral-hard-bird-attribution.json"
)
PAIRED_REPORT_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21f-known-gate-course-interframe-paired-oof.json"
)
ARTIFACT_ROOT_RELATIVE = Path(
    "artifacts/"
    "RADAR-LN3-V21f-cq08-known-gate-course-interframe-paired-oof"
)

EXPERIMENT_CONTRACT_SCHEMA = (
    "RADAR-LN3-V21f-known-gate-course-interframe-exploratory-oof-contract-v1"
)
FEATURE_MANIFEST_SCHEMA = (
    "RADAR-LN3-V21f-known-gate-course-interframe-feature-cache-manifest-v1"
)
OOF_CONTRACT_SCHEMA = (
    "RADAR-LN3-V21f-known-gate-course-interframe-oof-fold-contract-v1"
)
READINESS_SCHEMA = (
    "RADAR-LN3-V21f-known-gate-course-interframe-oof-readiness-v1"
)
ATTRIBUTION_SCHEMA = (
    "RADAR-LN3-V21f-spectral-hard-bird-attribution-v1"
)
PREDICTION_SCHEMA = (
    "RADAR-LN3-V21f-known-gate-course-interframe-oof-prediction-v1"
)
PAIRED_REPORT_SCHEMA = (
    "RADAR-LN3-V21f-known-gate-course-interframe-paired-oof-report-v1"
)


def config_relative(arm: str, fold_id: int) -> Path:
    if arm != ARM or not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V21f OOF arm/fold")
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V21f-cq08-known-gate-oof-fold-{int(fold_id):02d}-"
        "treatment-seed43-config.json"
    )


def artifact_relative(arm: str, fold_id: int) -> Path:
    if arm != ARM or not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V21f OOF arm/fold")
    return ARTIFACT_ROOT_RELATIVE / f"fold-{int(fold_id):02d}" / ARM


def prediction_relative(arm: str, fold_id: int) -> Path:
    return artifact_relative(arm, fold_id) / "held-out-predictions.json"


__all__ = [
    "ARM",
    "ASSIGNMENT_RELATIVE",
    "ASSIGNMENT_SEED",
    "ATTRIBUTION_RELATIVE",
    "ATTRIBUTION_SCHEMA",
    "COMBINED_CACHE_RELATIVE",
    "EXPERIMENT_CONTRACT_RELATIVE",
    "EXPERIMENT_CONTRACT_SCHEMA",
    "FEATURE_MANIFEST_RELATIVE",
    "FEATURE_MANIFEST_SCHEMA",
    "FOLD_COUNT",
    "FIXED_EPOCHS",
    "INDEX_RELATIVE",
    "OOF_CONTRACT_RELATIVE",
    "OOF_CONTRACT_SCHEMA",
    "PAIRED_REPORT_RELATIVE",
    "PAIRED_REPORT_SCHEMA",
    "PREDICTION_SCHEMA",
    "READINESS_RELATIVE",
    "READINESS_SCHEMA",
    "TRAINING_SEED",
    "V21A_FEATURE_CONTRACT_RELATIVE",
    "V21A_FEATURE_MANIFEST_RELATIVE",
    "V21A_SPECTRAL_CACHE_RELATIVE",
    "V21E_ARM",
    "V21E_COURSE_CACHE_RELATIVE",
    "V21E_FEATURE_CONTRACT_RELATIVE",
    "V21E_FEATURE_MANIFEST_RELATIVE",
    "V21E_FOLD_CONTRACT_RELATIVE",
    "V21E_TREATMENT_REPORT_RELATIVE",
    "artifact_relative",
    "config_relative",
    "fold_record",
    "load_fold_contract",
    "prediction_relative",
    "v21e_artifact_relative",
    "v21e_config_relative",
    "v21e_prediction_relative",
]
