"""Frozen V23 official-CfC KnownGate OOF contracts.

V23a replaces the V21e recurrent encoder with official ``ncps.torch.CfC``.
V23b makes the identical architecture replacement over the frozen V22 loss
baseline.  They intentionally remain separate comparisons: V23 does not
rewrite either historical control checkpoint or prediction artifact.
"""

from __future__ import annotations

from pathlib import Path


FOLD_COUNT = 5
FIXED_EPOCHS = 20
TRAINING_SEED = 43
ARMS = ("v23a", "v23b")

V21E_CONTROL_REPORT_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-treatment.json"
)
V22_CONTROL_REPORT_RELATIVE = Path(
    "src-v3/processed/report/RADAR-LN3-V22-known-gate-fixed-parent-balanced-paired-oof.json"
)
V22_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof-contract.json"
)
V21E_FOLD_CONTRACT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-contract.json"
)
V21E_ASSIGNMENT_RELATIVE = Path(
    "data/processed/RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-assignment.json"
)


def variant_slug(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unsupported V23 arm: {arm}")
    return {
        "v23a": "v21e-loss",
        "v23b": "v22-fixed-parent-balanced-loss",
    }[arm]


def source_arm_name(arm: str) -> str:
    if arm not in ARMS:
        raise ValueError(f"unsupported V23 arm: {arm}")
    return "v21e" if arm == "v23a" else "v22"


def source_config_relative(arm: str, fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("V23 fold id is invalid")
    if arm == "v23a":
        return Path(
            "config/RADAR-LN3/"
            f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{int(fold_id):02d}-"
            "treatment-seed43-config.json"
        )
    if arm == "v23b":
        return Path(
            "config/RADAR-LN3/"
            "RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof-fold-"
            f"{int(fold_id):02d}-treatment-seed43-config.json"
        )
    raise ValueError(f"unsupported V23 arm: {arm}")


def source_prediction_relative(arm: str, fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("V23 fold id is invalid")
    if arm == "v23a":
        return Path(
            "artifacts/RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof/"
            f"fold-{int(fold_id):02d}/treatment/held-out-predictions.json"
        )
    if arm == "v23b":
        return Path(
            "artifacts/RADAR-LN3-V22-cq08-known-gate-fixed-parent-balanced-oof/"
            f"fold-{int(fold_id):02d}/treatment/held-out-predictions.json"
        )
    raise ValueError(f"unsupported V23 arm: {arm}")


def config_relative(arm: str, fold_id: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-{arm.upper()}-cq08-known-gate-official-cfc-oof-fold-"
        f"{int(fold_id):02d}-treatment-seed43-config.json"
    )


def artifact_relative(arm: str, fold_id: int) -> Path:
    return Path(
        "artifacts/"
        f"RADAR-LN3-{arm.upper()}-cq08-known-gate-official-cfc-"
        f"{variant_slug(arm)}-oof/fold-{int(fold_id):02d}/treatment"
    )


def prediction_relative(arm: str, fold_id: int) -> Path:
    return artifact_relative(arm, fold_id) / "held-out-predictions.json"


def contract_relative(arm: str) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-{arm.upper()}-cq08-known-gate-official-cfc-{variant_slug(arm)}-oof-contract.json"
    )


def readiness_relative(arm: str) -> Path:
    return Path(
        "src-v3/processed/report/"
        f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-{variant_slug(arm)}-oof-training-readiness.json"
    )


def report_relative(arm: str) -> Path:
    return Path(
        "src-v3/processed/report/"
        f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-{variant_slug(arm)}-paired-oof.json"
    )


def contract_schema(arm: str) -> str:
    return f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-oof-contract-v1"


def readiness_schema(arm: str) -> str:
    return f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-oof-readiness-v1"


def prediction_schema(arm: str) -> str:
    return f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-oof-prediction-v1"


def report_schema(arm: str) -> str:
    return f"RADAR-LN3-{arm.upper()}-known-gate-official-cfc-paired-oof-report-v1"


__all__ = [
    "ARMS",
    "FIXED_EPOCHS",
    "FOLD_COUNT",
    "TRAINING_SEED",
    "V21E_ASSIGNMENT_RELATIVE",
    "V21E_CONTROL_REPORT_RELATIVE",
    "V21E_FOLD_CONTRACT_RELATIVE",
    "V22_CONTRACT_RELATIVE",
    "V22_CONTROL_REPORT_RELATIVE",
    "artifact_relative",
    "config_relative",
    "contract_relative",
    "contract_schema",
    "prediction_relative",
    "prediction_schema",
    "readiness_relative",
    "readiness_schema",
    "report_relative",
    "report_schema",
    "source_arm_name",
    "source_config_relative",
    "source_prediction_relative",
    "variant_slug",
]
