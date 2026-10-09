"""Shared contracts for V21h/V21i features appended to frozen V21e OOF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .known_gate_features import (
    COURSE_D_RANGE_FEATURE_NAMES,
    COURSE_ISO_FEATURE_NAMES,
    D_RANGE_PREFIX_FEATURE_NAMES,
    FEATURE_INDICES_BY_VARIANT,
    ISO_FEATURE_NAMES,
    V21C_VARIANT,
    V21H_COURSE_ISO_VARIANT,
    V21I_COURSE_D_RANGE_VARIANT,
)
from .known_gate_v21e_oof import (
    ASSIGNMENT_RELATIVE,
    ASSIGNMENT_SEED,
    FOLD_CONTRACT_RELATIVE,
    FOLD_COUNT,
    FIXED_EPOCHS,
    INDEX_RELATIVE,
    TRAINING_SEED,
    canonical_sha256,
    config_relative as v21e_config_relative,
    fold_record,
    load_fold_contract,
    relative_path,
    sequence_sha256,
    sha256_file,
    write_new_or_identical,
)


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
V21_CORE_CACHE_RELATIVE = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-train-feature-cache.npz"
)
V21_CORE_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-data-contract.json"
)
V21_CORE_MANIFEST_RELATIVE = Path(
    "data/processed/RADAR-LN3-V21-cq08-known-gate-feature-cache-manifest.json"
)
V21D_PREFIX_CACHE_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21d-cq08-known-gate-train-causal-prefix-d-range-cache.npz"
)
V21D_PREFIX_CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21d-cq08-known-gate-causal-prefix-d-range-clutter-bird-"
    "contract.json"
)


@dataclass(frozen=True)
class AppendSpec:
    code: str
    variant: str
    slug: str
    feature_names: tuple[str, ...]
    new_feature_names: tuple[str, ...]
    source_kind: str
    source_cache_relative: Path
    source_contract_relative: Path
    source_manifest_relative: Path
    source_indices: tuple[int, ...]
    primary_labels: tuple[str, str]
    safety_labels: tuple[str, str]
    combined_cache_relative: Path
    experiment_contract_relative: Path
    feature_manifest_relative: Path
    oof_contract_relative: Path
    readiness_relative: Path
    paired_report_relative: Path
    artifact_root_relative: Path
    prediction_schema: str
    paired_report_schema: str
    readiness_schema: str
    oof_contract_schema: str
    experiment_contract_schema: str
    feature_manifest_schema: str


def _make_spec(
    *,
    code: str,
    variant: str,
    slug: str,
    feature_names: tuple[str, ...],
    new_feature_names: tuple[str, ...],
    source_kind: str,
    source_cache_relative: Path,
    source_contract_relative: Path,
    source_manifest_relative: Path,
    source_indices: tuple[int, ...],
    primary_labels: tuple[str, str],
    safety_labels: tuple[str, str],
) -> AppendSpec:
    prefix = f"RADAR-LN3-{code}-cq08-known-gate-course-{slug}"
    return AppendSpec(
        code=code,
        variant=variant,
        slug=slug,
        feature_names=feature_names,
        new_feature_names=new_feature_names,
        source_kind=source_kind,
        source_cache_relative=source_cache_relative,
        source_contract_relative=source_contract_relative,
        source_manifest_relative=source_manifest_relative,
        source_indices=source_indices,
        primary_labels=primary_labels,
        safety_labels=safety_labels,
        combined_cache_relative=Path(f"data/processed/{prefix}-cache.npz"),
        experiment_contract_relative=Path(
            f"data/processed/{prefix}-experiment-contract.json"
        ),
        feature_manifest_relative=Path(
            f"data/processed/{prefix}-feature-manifest.json"
        ),
        oof_contract_relative=Path(f"data/processed/{prefix}-oof-contract.json"),
        readiness_relative=Path(
            f"src-v3/processed/report/{prefix}-oof-training-readiness.json"
        ),
        paired_report_relative=Path(
            f"src-v3/processed/report/{prefix}-paired-oof.json"
        ),
        artifact_root_relative=Path(f"artifacts/{prefix}-paired-oof"),
        prediction_schema=f"RADAR-LN3-{code}-known-gate-course-{slug}-oof-prediction-v1",
        paired_report_schema=f"RADAR-LN3-{code}-known-gate-course-{slug}-paired-oof-report-v1",
        readiness_schema=f"RADAR-LN3-{code}-known-gate-course-{slug}-oof-readiness-v1",
        oof_contract_schema=f"RADAR-LN3-{code}-known-gate-course-{slug}-oof-contract-v1",
        experiment_contract_schema=f"RADAR-LN3-{code}-known-gate-course-{slug}-experiment-contract-v1",
        feature_manifest_schema=f"RADAR-LN3-{code}-known-gate-course-{slug}-feature-manifest-v1",
    )


V21H_SPEC = _make_spec(
    code="V21h",
    variant=V21H_COURSE_ISO_VARIANT,
    slug="iso",
    feature_names=COURSE_ISO_FEATURE_NAMES,
    new_feature_names=ISO_FEATURE_NAMES,
    source_kind="core_cache_selected",
    source_cache_relative=V21_CORE_CACHE_RELATIVE,
    source_contract_relative=Path(
        "data/processed/"
        "RADAR-LN3-V21c-cq08-known-gate-target-row-iso-clutter-drone-"
        "contract.json"
    ),
    source_manifest_relative=V21_CORE_MANIFEST_RELATIVE,
    source_indices=FEATURE_INDICES_BY_VARIANT[V21C_VARIANT],
    primary_labels=("DroneTarget", "ClutterTarget"),
    safety_labels=("BirdTarget", "BalloonTarget"),
)
V21I_SPEC = _make_spec(
    code="V21i",
    variant=V21I_COURSE_D_RANGE_VARIANT,
    slug="d-range",
    feature_names=COURSE_D_RANGE_FEATURE_NAMES,
    new_feature_names=D_RANGE_PREFIX_FEATURE_NAMES,
    source_kind="independent_prefix_cache",
    source_cache_relative=V21D_PREFIX_CACHE_RELATIVE,
    source_contract_relative=V21D_PREFIX_CONTRACT_RELATIVE,
    source_manifest_relative=Path(
        "data/processed/"
        "RADAR-LN3-V21d-cq08-known-gate-causal-prefix-d-range-"
        "feature-cache-manifest.json"
    ),
    source_indices=(0,),
    primary_labels=("BirdTarget", "ClutterTarget"),
    safety_labels=("DroneTarget", "BalloonTarget"),
)

SPECS = {"h": V21H_SPEC, "i": V21I_SPEC}


def get_spec(code: str) -> AppendSpec:
    try:
        return SPECS[str(code).lower()]
    except KeyError as exc:
        raise ValueError("V21h/i OOF variant must be 'h' or 'i'") from exc


def config_relative(code: str, fold_id: int) -> Path:
    spec = get_spec(code)
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V21h/i OOF fold")
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-{spec.code}-cq08-known-gate-course-{spec.slug}-"
        f"oof-fold-{int(fold_id):02d}-treatment-seed43-config.json"
    )


def artifact_relative(code: str, fold_id: int) -> Path:
    spec = get_spec(code)
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V21h/i OOF fold")
    return spec.artifact_root_relative / f"fold-{int(fold_id):02d}" / "treatment"


def prediction_relative(code: str, fold_id: int) -> Path:
    return artifact_relative(code, fold_id) / "held-out-predictions.json"


__all__ = [
    "AppendSpec",
    "ASSIGNMENT_RELATIVE",
    "ASSIGNMENT_SEED",
    "FOLD_CONTRACT_RELATIVE",
    "FOLD_COUNT",
    "FIXED_EPOCHS",
    "INDEX_RELATIVE",
    "TRAINING_SEED",
    "V21E_COURSE_CACHE_RELATIVE",
    "V21E_FEATURE_CONTRACT_RELATIVE",
    "V21E_FEATURE_MANIFEST_RELATIVE",
    "V21E_TREATMENT_REPORT_RELATIVE",
    "V21_CORE_CACHE_RELATIVE",
    "V21_CORE_CONTRACT_RELATIVE",
    "V21_CORE_MANIFEST_RELATIVE",
    "V21D_PREFIX_CACHE_RELATIVE",
    "V21D_PREFIX_CONTRACT_RELATIVE",
    "V21H_SPEC",
    "V21I_SPEC",
    "SPECS",
    "artifact_relative",
    "canonical_sha256",
    "config_relative",
    "fold_record",
    "get_spec",
    "load_fold_contract",
    "prediction_relative",
    "relative_path",
    "sequence_sha256",
    "sha256_file",
    "v21e_config_relative",
    "write_new_or_identical",
]
