from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


VERSION = "RADAR-LN3-V14"
KNOWN_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
UNKNOWN_LABEL = "UnknownTarget"
INDEX_RELATIVE = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
CONTRACT_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-strict-open-set-data-contract.json")
STATS_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-known-only-train-normalization-stats.json")
MANIFEST_RELATIVE = Path("data/processed/RADAR-LN3-V14-cq08-known-only-parent-grouped-oof-manifest.json")
DEFAULT_FOLDS = 5
DEFAULT_SEED = 42


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assignment_key(parent_id: str, *, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{parent_id}".encode("utf-8")).hexdigest()


def _sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different OOF manifest: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def _fold_assignments(parent_frame: pd.DataFrame, *, folds: int, seed: int) -> list[dict[str, Any]]:
    assignments: list[dict[str, Any]] = []
    for label in KNOWN_LABELS:
        parents = sorted(
            parent_frame.loc[parent_frame["label_name"].eq(label), "parent_sample_id"].astype(str),
            key=lambda value: (_assignment_key(value, seed=seed), value),
        )
        for index, parent_id in enumerate(parents):
            assignments.append({
                "parent_sample_id": parent_id,
                "label_name": label,
                "fold_id": int(index % folds),
            })
    return sorted(assignments, key=lambda row: row["parent_sample_id"])


def build(project_root: Path, *, folds: int = DEFAULT_FOLDS, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    root = project_root.resolve()
    fold_count = int(folds)
    if fold_count < 2:
        raise ValueError("OOF requires at least two parent-grouped folds")
    index_path = root / INDEX_RELATIVE
    contract_path = root / CONTRACT_RELATIVE
    stats_path = root / STATS_RELATIVE
    manifest_path = root / MANIFEST_RELATIVE
    for path in (index_path, contract_path, stats_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    frame = pd.read_csv(index_path, usecols=["parent_sample_id", "split_name", "label_name"])
    if frame.groupby("parent_sample_id")["split_name"].nunique().gt(1).any():
        raise ValueError("OOF source has a parent crossing splits")
    if frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("OOF source has a parent with multiple labels")
    parents = frame.drop_duplicates("parent_sample_id").copy()
    parents["parent_sample_id"] = parents["parent_sample_id"].astype(str)
    parents["split_name"] = parents["split_name"].astype(str)
    parents["label_name"] = parents["label_name"].astype(str)
    train = parents[parents["split_name"].eq("train")].copy()
    unknown_source = train[train["label_name"].eq(UNKNOWN_LABEL)]
    known = train[train["label_name"].isin(KNOWN_LABELS)].copy()
    invalid = sorted(set(train["label_name"]) - set((*KNOWN_LABELS, UNKNOWN_LABEL)))
    if invalid:
        raise ValueError(f"OOF source contains unsupported labels: {invalid}")
    if len(known) != 1011 or len(unknown_source) != 70:
        raise ValueError("OOF frozen known/Unknown train support changed")
    assignments = _fold_assignments(known, folds=fold_count, seed=int(seed))
    assigned_ids = [row["parent_sample_id"] for row in assignments]
    known_ids = sorted(known["parent_sample_id"].astype(str))
    if len(assignments) != len(known_ids) or len(set(assigned_ids)) != len(known_ids) or sorted(assigned_ids) != known_ids:
        raise ValueError("OOF assignment does not cover every known train parent exactly once")
    if any(row["label_name"] == UNKNOWN_LABEL for row in assignments):
        raise ValueError("UnknownTarget entered the OOF manifest")
    fold_summaries: list[dict[str, Any]] = []
    for fold_id in range(fold_count):
        held_out = [row for row in assignments if row["fold_id"] == fold_id]
        training = [row for row in assignments if row["fold_id"] != fold_id]
        held_ids = sorted(row["parent_sample_id"] for row in held_out)
        training_ids = sorted(row["parent_sample_id"] for row in training)
        if set(held_ids) & set(training_ids):
            raise ValueError(f"OOF fold {fold_id} leaks a held-out parent into training")
        if len(held_ids) + len(training_ids) != len(known_ids):
            raise ValueError(f"OOF fold {fold_id} does not partition known train parents")
        held_counts = {label: sum(row["label_name"] == label for row in held_out) for label in KNOWN_LABELS}
        train_counts = {label: sum(row["label_name"] == label for row in training) for label in KNOWN_LABELS}
        if any(value <= 0 for value in held_counts.values()) or any(value <= 0 for value in train_counts.values()):
            raise ValueError(f"OOF fold {fold_id} lacks one known label")
        fold_summaries.append({
            "fold_id": fold_id,
            "train_parent_count": len(training_ids),
            "held_out_parent_count": len(held_ids),
            "train_label_parent_counts": train_counts,
            "held_out_label_parent_counts": held_counts,
            "train_parent_ids_sha256": _sequence_sha256(training_ids),
            "held_out_parent_ids_sha256": _sequence_sha256(held_ids),
            "parent_overlap_count": 0,
            "unknown_train_parent_count": 0,
            "unknown_held_out_parent_count": 0,
            "validation_parent_count": 0,
            "test_parent_count": 0,
        })
    payload = {
        "schema_version": f"{VERSION}-known-only-parent-grouped-oof-manifest-v1",
        "source_index": str(INDEX_RELATIVE).replace("\\", "/"),
        "source_index_sha256": _sha256(index_path),
        "data_contract": str(CONTRACT_RELATIVE).replace("\\", "/"),
        "data_contract_sha256": _sha256(contract_path),
        "normalization_stats": str(STATS_RELATIVE).replace("\\", "/"),
        "normalization_stats_sha256": _sha256(stats_path),
        "fit_split": "train",
        "fit_population": "known_only",
        "grouping": "parent",
        "prediction_protocol": "out_of_fold",
        "fold_count": fold_count,
        "assignment_seed": int(seed),
        "assignment_rule": "within-label deterministic SHA-256(seed,parent_id) order round-robin folds",
        "known_labels": list(KNOWN_LABELS),
        "unknown_label": UNKNOWN_LABEL,
        "known_parent_count": len(known_ids),
        "known_parent_ids_sha256": _sequence_sha256(known_ids),
        "unknown_source_parent_count": len(unknown_source),
        "unknown_oof_parent_count": 0,
        "validation_parent_count": 0,
        "test_parent_count": 0,
        "every_known_parent_exactly_one_held_out_fold": True,
        "assignments": assignments,
        "folds": fold_summaries,
        "threshold_fit_contract": {
            "scores": [
                "gate_dbb_probability_and_margin",
                "dbb_top_probability_and_margin",
                "cu_clutter_probability_and_margin",
            ],
            "unknown_threshold_fit_count": 0,
            "validation_used": False,
            "test_used": False,
            "known_false_rejection_and_coverage_constraints_only": True,
        },
        "execution_status": {
            "manifest": "prepared",
            "fold_training": "未执行",
            "oof_prediction": "未执行",
            "threshold_fitting": "未执行",
            "formal_rejection": "未执行",
            "stable": "未执行",
            "test": "未执行",
            "p95": "未执行",
        },
    }
    write_status = _write_new_or_identical(manifest_path, payload)
    return {
        "status": "passed",
        "schema_version": payload["schema_version"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "write_status": write_status,
        "known_parent_count": len(known_ids),
        "unknown_source_parent_count": len(unknown_source),
        "unknown_oof_parent_count": 0,
        "fold_count": fold_count,
        "every_known_parent_exactly_one_held_out_fold": True,
        "fold_training": "未执行",
        "threshold_fitting": "未执行",
        "test": "未执行",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the V14 known-only parent-grouped OOF manifest.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    print(json.dumps(build(args.project_root.resolve(), folds=args.folds, seed=args.seed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
