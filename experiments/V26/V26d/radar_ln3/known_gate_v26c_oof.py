"""Versioned paths and small integrity helpers for V26c L1/L2 OOF calibration.

V26c intentionally has its own namespace.  It reuses V21e's *assignment*
only; it never reuses V21e's models, normalization, or predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


FOLD_COUNT = 5
L1_EPOCH = 14
L2_EPOCH = 15
MINIMUM_EVIDENCE_EVENTS = 3
L1_KNOWN_TARGET_ACTION_RECALL_FLOOR = 0.95
L2_CORRECT_ACTION_RECALL_FLOOR = 0.95
CLUTTER_PRECISION_FLOOR = 0.90
CLUTTER_RECALL_FLOOR = 8.0 / 9.0

V21E_ASSIGNMENT = Path(
    "data/processed/RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-assignment.json"
)
CONTRACT = Path(
    "data/processed/RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness.json"
)
ARTIFACT_ROOT = Path("artifacts/RADAR-LN3-V26c-cq08-l1-l2-known-only-oof-calibration")

CONTRACT_SCHEMA = "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26c-l1-l2-known-only-oof-calibration-readiness-v1"
L1_PREDICTION_SCHEMA = "RADAR-LN3-V26c-l1-known-only-oof-prediction-v1"
L2_PREDICTION_SCHEMA = "RADAR-LN3-V26c-l2-known-only-oof-prediction-v1"
L1_CHECKPOINT_SCHEMA = "RADAR-LN3-V26c-l1-known-only-oof-checkpoint-v1"
L2_CHECKPOINT_SCHEMA = "RADAR-LN3-V26c-l2-known-only-oof-checkpoint-v1"

L3_REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26c-l1-l2-known-only-cross-fitted-l3-calibration.json"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sequence_sha256(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def relative_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    """Persist an immutable V26c JSON artifact atomically.

    All V26c protocol artifacts are content addressed by their callers.  A
    partial write must therefore never be mistaken for a completed contract,
    checkpoint summary, or prediction.  Reusing an identical artifact is safe;
    replacing a different one is deliberately forbidden.
    """

    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different V26c artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        if temporary.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                "refusing to replace a different interrupted V26c artifact: "
                f"{temporary}"
            )
    else:
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"


def fold_directory(fold_id: int) -> Path:
    if not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError(f"invalid V26c fold: {fold_id}")
    return ARTIFACT_ROOT / f"fold-{int(fold_id):02d}"


def l1_config_path(fold_id: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V26c-cq08-known-gate-oof-fold-{int(fold_id):02d}-seed43-config.json"
    )


def l2_config_path(fold_id: int) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{int(fold_id):02d}-seed42-config.json"
    )










def l1_train_feature_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-known-gate-oof-fold-{int(fold_id):02d}-"
        "train-course-cache.npz"
    )




def l1_held_out_feature_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-known-gate-oof-fold-{int(fold_id):02d}-"
        "held-out-course-cache.npz"
    )


def l2_train_physical_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{int(fold_id):02d}-"
        "train-rd-physical4-cache.npz"
    )




def l2_held_out_physical_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{int(fold_id):02d}-"
        "held-out-rd-physical4-cache.npz"
    )


def l2_train_micro_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{int(fold_id):02d}-"
        "train-micro-doppler8-cache.npz"
    )








def l2_held_out_micro_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{int(fold_id):02d}-"
        "held-out-micro-doppler8-cache.npz"
    )


def l2_held_out_mixed_physical_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{int(fold_id):02d}-"
        "held-out-mixed-rd-physical4-cache.npz"
    )


def l2_held_out_mixed_micro_cache_path(fold_id: int) -> Path:
    return Path(
        "data/processed/"
        f"RADAR-LN3-V26c-cq08-dbb-specialist-oof-fold-{int(fold_id):02d}-"
        "held-out-mixed-micro-doppler8-cache.npz"
    )


def l1_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "epoch-14-model.pt"


def l1_last_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "last_model.pt"


def l2_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l2" / "epoch-15-model.pt"


def l2_last_checkpoint_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l2" / "last_model.pt"


def l1_training_summary_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1" / "training-summary.json"


def l2_training_summary_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l2" / "training-summary.json"


def l1_prediction_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l1-held-out-predictions.json"


def l2_prediction_path(fold_id: int) -> Path:
    return fold_directory(fold_id) / "l2-held-out-predictions.json"


def fold_record(contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    rows = [
        dict(row)
        for row in list(contract.get("folds") or ())
        if int(dict(row).get("fold_id", -1)) == int(fold_id)
    ]
    if len(rows) != 1:
        raise ValueError(f"V26c fold is unavailable or repeated: {fold_id}")
    row = rows[0]
    train_ids = sorted(str(value) for value in list(row.get("train_parent_ids") or ()))
    held_out_ids = sorted(str(value) for value in list(row.get("held_out_parent_ids") or ()))
    if (
        not train_ids
        or not held_out_ids
        or set(train_ids) & set(held_out_ids)
        or len(train_ids) + len(held_out_ids) != 986
        or sequence_sha256(train_ids) != row.get("train_parent_ids_sha256")
        or sequence_sha256(held_out_ids) != row.get("held_out_parent_ids_sha256")
    ):
        raise ValueError(f"V26c fold partition is invalid: {fold_id}")
    row["train_parent_ids"] = train_ids
    row["held_out_parent_ids"] = held_out_ids
    return row


__all__ = ["CONTRACT", "CONTRACT_SCHEMA", "FOLD_COUNT", "L1_EPOCH", "MINIMUM_EVIDENCE_EVENTS", "L1_KNOWN_TARGET_ACTION_RECALL_FLOOR", "L2_CORRECT_ACTION_RECALL_FLOOR", "CLUTTER_PRECISION_FLOOR", "CLUTTER_RECALL_FLOOR", "L1_CHECKPOINT_SCHEMA", "L1_PREDICTION_SCHEMA", "L2_EPOCH", "L2_CHECKPOINT_SCHEMA", "L2_PREDICTION_SCHEMA", "L3_REPORT", "READINESS", "READINESS_SCHEMA", "V21E_ASSIGNMENT", "fold_directory", "fold_record", "l1_checkpoint_path", "l1_config_path", "l1_held_out_feature_cache_path", "l1_last_checkpoint_path", "l1_prediction_path", "l1_train_feature_cache_path", "l1_training_summary_path", "l2_checkpoint_path", "l2_config_path", "l2_held_out_mixed_micro_cache_path", "l2_held_out_mixed_physical_cache_path", "l2_held_out_micro_cache_path", "l2_held_out_physical_cache_path", "l2_last_checkpoint_path", "l2_prediction_path", "l2_train_micro_cache_path", "l2_train_physical_cache_path", "l2_training_summary_path", "relative_path", "resolve_path", "sequence_sha256", "sha256_file", "write_new_or_identical"]