"""Versioned paths and integrity helpers for the V26e long-horizon L1 OOF.

V26e changes only the L1 training horizon of V26c from 20 to 32 epochs.  The
fold assignment, feature caches, fold-train normalization, and V26c L2
epoch-15 outputs are reused read-only.  Unknown parents remain outside every
training and calibration population.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


from radar_ln3 import known_gate_v26c_oof as _v26c


FOLD_COUNT = 5
L1_TRAIN_EPOCHS = 32
L1_EPOCH = 32

BOUND_FILE_MAX_BYTES = 64 * 1024 * 1024

CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26e-cq08-l1-epoch32-known-only-oof-calibration-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26e-l1-epoch32-known-only-oof-calibration-readiness.json"
)
ARTIFACT_ROOT = Path(
    "artifacts/RADAR-LN3-V26e-cq08-l1-epoch32-known-only-oof-calibration"
)

L1_CHECKPOINT_SCHEMA = "RADAR-LN3-V26e-l1-epoch32-known-only-oof-checkpoint-v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()




def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def bound_path(root: Path, value: str | Path, description: str) -> Path:
    """Resolve a contract path while refusing project-root escape."""

    path = resolve_path(root, value)
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"V26e {description} escapes the project root: {value}") from exc
    return path


def require_bound_file(
    root: Path,
    value: str | Path,
    description: str,
    *,
    expected_sha256: str | None = None,
    max_bytes: int = BOUND_FILE_MAX_BYTES,
) -> Path:
    """Resolve, bound, size-check and optionally hash-check one artifact."""

    path = bound_path(root, value, description)
    if not path.is_file():
        raise FileNotFoundError(f"V26e {description} is missing: {path}")
    if path.stat().st_size > int(max_bytes):
        raise MemoryError(
            f"V26e {description} exceeds the bounded size: "
            f"{path.stat().st_size} > {int(max_bytes)} bytes"
        )
    if expected_sha256 is not None and sha256_file(path) != str(expected_sha256):
        raise ValueError(f"V26e {description} hash differs: {path}")
    return path












def fold_directory(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError(f"invalid V26e fold: {fold_id}")
    return ARTIFACT_ROOT / f"fold-{int(fold_id):02d}"


def l1_config_path(fold_id: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V26e-cq08-known-gate-oof-fold-{int(fold_id):02d}-seed43-config.json"
    )


def l1_data_contract_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26e-cq08-known-gate-oof-fold-{int(fold_id):02d}-data-contract.json"
    )


def l1_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "epoch-32-model.pt"




def l1_training_summary_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "training-summary.json"


def l1_prediction_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1-held-out-predictions.json"




# These are deliberately aliases to V26c's physical files.  V26e changes no
# feature values, normalization values, or L2 outputs.
l1_normalization_path = _v26c.l1_normalization_path
l1_train_feature_cache_path = _v26c.l1_train_feature_cache_path
l1_held_out_feature_cache_path = _v26c.l1_held_out_feature_cache_path
