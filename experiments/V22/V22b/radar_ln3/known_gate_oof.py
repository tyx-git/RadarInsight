"""Frozen contracts for V21e KnownGate paired five-fold OOF evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


FOLD_COUNT = 5
FIXED_EPOCHS = 20
TRAINING_SEED = 43
ARMS = ("control", "treatment")

ASSIGNMENT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-assignment.json"
)
FOLD_CONTRACT_RELATIVE = Path(
    "data/processed/"
    "RADAR-LN3-V21e-cq08-known-gate-capture-content-grouped-oof-contract.json"
)
READINESS_RELATIVE = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-training-readiness.json"
)
ARTIFACT_ROOT_RELATIVE = Path(
    "artifacts/"
    "RADAR-LN3-V21e-cq08-known-gate-causal-course-paired-oof"
)

ASSIGNMENT_SCHEMA = (
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-assignment-v1"
)
FOLD_CONTRACT_SCHEMA = (
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-contract-v1"
)
READINESS_SCHEMA = (
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-readiness-v1"
)
PREDICTION_SCHEMA = (
    "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-prediction-v1"
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




def resolve_path(project_root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return (
        candidate.resolve()
        if candidate.is_absolute()
        else (project_root / candidate).resolve()
    )


def relative_path(project_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different OOF artifact: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def inverse_frequency_weights(counts: Sequence[int]) -> tuple[float, ...]:
    values = tuple(int(value) for value in counts)
    total = sum(values)
    if total <= 0 or any(value <= 0 for value in values):
        raise ValueError("OOF class-weight counts must all be positive")
    return tuple(total / (len(values) * value) for value in values)


def config_relative(arm: str, fold_id: int) -> Path:
    if arm not in ARMS or not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V21e OOF arm/fold")
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V21e-cq08-known-gate-oof-fold-{int(fold_id):02d}-"
        f"{arm}-seed43-config.json"
    )






def artifact_relative(arm: str, fold_id: int) -> Path:
    if arm not in ARMS or not 0 <= int(fold_id) < FOLD_COUNT:
        raise ValueError("invalid V21e OOF arm/fold")
    return ARTIFACT_ROOT_RELATIVE / f"fold-{int(fold_id):02d}" / arm


def prediction_relative(arm: str, fold_id: int) -> Path:
    return artifact_relative(arm, fold_id) / "held-out-predictions.json"


def load_fold_contract(project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = project_root.resolve()
    contract_path = root / FOLD_CONTRACT_RELATIVE
    assignment_path = root / ASSIGNMENT_RELATIVE
    if not contract_path.is_file() or not assignment_path.is_file():
        raise FileNotFoundError("V21e OOF prepare artifacts are missing")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    if (
        contract.get("schema_version") != FOLD_CONTRACT_SCHEMA
        or assignment.get("schema_version") != ASSIGNMENT_SCHEMA
        or contract.get("assignment_sha256") != sha256_file(assignment_path)
        or int(contract.get("fold_count", -1)) != FOLD_COUNT
        or int(assignment.get("fold_count", -1)) != FOLD_COUNT
        or int(contract.get("known_parent_count", -1)) != 986
        or int(assignment.get("known_parent_count", -1)) != 986
        or contract.get("grouping") != "capture_content_provenance_component"
        or assignment.get("grouping") != "capture_content_provenance_component"
        or contract.get("prediction_protocol")
        != "out_of_fold_fixed_final_epoch"
        or assignment.get("prediction_protocol") != "out_of_fold"
    ):
        raise ValueError("V21e OOF fold contract is invalid or stale")
    return contract, assignment


def fold_record(contract: Mapping[str, Any], fold_id: int) -> dict[str, Any]:
    matches = [
        dict(row)
        for row in list(contract.get("folds") or [])
        if int(row.get("fold_id", -1)) == int(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"V21e OOF fold is unavailable or repeated: {fold_id}")
    record = matches[0]
    train_ids = sorted(str(value) for value in record.get("train_parent_ids") or [])
    held_out_ids = sorted(
        str(value) for value in record.get("held_out_parent_ids") or []
    )
    if (
        not train_ids
        or not held_out_ids
        or set(train_ids) & set(held_out_ids)
        or len(train_ids) + len(held_out_ids) != 986
        or sequence_sha256(train_ids) != record.get("train_parent_ids_sha256")
        or sequence_sha256(held_out_ids)
        != record.get("held_out_parent_ids_sha256")
        or int(record.get("unknown_training_exposure", -1)) != 0
        or int(record.get("validation_parent_count", -1)) != 0
        or int(record.get("test_parent_count", -1)) != 0
    ):
        raise ValueError(f"V21e OOF fold {fold_id} partition is invalid")
    record["train_parent_ids"] = train_ids
    record["held_out_parent_ids"] = held_out_ids
    return record


__all__ = ["ARMS", "ARTIFACT_ROOT_RELATIVE", "ASSIGNMENT_RELATIVE", "ASSIGNMENT_SCHEMA", "FIXED_EPOCHS", "FOLD_CONTRACT_RELATIVE", "FOLD_CONTRACT_SCHEMA", "FOLD_COUNT", "PREDICTION_SCHEMA", "READINESS_RELATIVE", "READINESS_SCHEMA", "TRAINING_SEED", "artifact_relative", "config_relative", "fold_record", "inverse_frequency_weights", "load_fold_contract", "prediction_relative", "relative_path", "resolve_path", "sequence_sha256", "sha256_file", "write_new_or_identical"]