"""Immutable paths and protocol constants for the V26d endpoint audit.

V26d is deliberately a read-only follow-up to V26c.  It never retrains a
fold and it never chooses a checkpoint from an outer held-out fold.  The
single treatment is the already-materialized epoch-20 ``last_model.pt`` from
each V26c L1 trajectory, compared with V26c's frozen epoch-14 snapshot.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


CONTROL_EPOCH = 14
TREATMENT_EPOCH = 20
FOLD_COUNT = 5

CONTRACT = Path(
    "data/processed/"
    "RADAR-LN3-V26d-cq08-l1-epoch20-endpoint-diagnostic-contract.json"
)
READINESS = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic-readiness.json"
)
REPORT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic.json"
)
PARENT_DETAIL_CSV = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V26d-l1-epoch20-endpoint-parent-actions.csv"
)
ARTIFACT_ROOT = Path(
    "artifacts/RADAR-LN3-V26d-cq08-l1-epoch20-endpoint-diagnostic"
)

CONTRACT_SCHEMA = "RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic-contract-v1"
READINESS_SCHEMA = "RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic-readiness-v1"
TREATMENT_PREDICTION_SCHEMA = "RADAR-LN3-V26d-l1-epoch20-oof-prediction-v1"
REPORT_SCHEMA = "RADAR-LN3-V26d-l1-epoch20-endpoint-diagnostic-report-v1"


def treatment_prediction_path(fold_id: int) -> Path:
    fold = int(fold_id)
    if not 0 <= fold < FOLD_COUNT:
        raise ValueError(f"invalid V26d fold: {fold}")
    return ARTIFACT_ROOT / f"fold-{fold:02d}" / "l1-epoch20-held-out-predictions.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(root: Path, path: str | Path) -> str:
    return Path(path).resolve().relative_to(root.resolve()).as_posix()


def write_new_or_identical(path: Path, payload: Mapping[str, Any]) -> str:
    """Write a JSON artifact once, or verify that a replay is identical."""

    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to overwrite a different V26d artifact: {path}"
            )
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        if temporary.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                f"refusing to replace a different interrupted V26d artifact: {temporary}"
            )
    else:
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"


__all__ = ["ARTIFACT_ROOT", "CONTRACT", "CONTRACT_SCHEMA", "CONTROL_EPOCH", "FOLD_COUNT", "PARENT_DETAIL_CSV", "READINESS", "READINESS_SCHEMA", "REPORT", "REPORT_SCHEMA", "TREATMENT_EPOCH", "TREATMENT_PREDICTION_SCHEMA", "relative_path", "sha256_file", "treatment_prediction_path", "write_new_or_identical"]