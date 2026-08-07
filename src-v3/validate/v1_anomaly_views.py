from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from radar_ln3.config import load_config, resolve_project_path
from radar_ln3.data import CQ08PointSequenceDataset, build_point_dataloader


VERSION = "RADAR-LN3-V1"
SOURCE_VERSION = "RADAR-LN3-V0"
ANOMALY_PARENT_IDS = {
    "cq08|track|634",
    "cq08|track|785",
    "cq08|track|3338",
}
MAIN_INDEX_NAME = f"{VERSION}-cq08-tr-rd-main-index.csv"
AUDIT_INDEX_NAME = f"{VERSION}-cq08-tr-rd-anomaly-audit-index.csv"
MAIN_SPLIT_NAME = f"{VERSION}-cq08-tr-rd-main-split.json"
AUDIT_SPLIT_NAME = f"{VERSION}-cq08-tr-rd-anomaly-audit-split.json"
MANIFEST_NAME = f"{VERSION}-cq08-tr-rd-view-manifest.json"
CONTRACT_NAME = f"{VERSION}-cq08-tr-rd-data-contract.json"
SUMMARY_NAME = f"{VERSION}-cq08-tr-rd-view-summary.json"
SOURCE_INDEX_NAME = f"{SOURCE_VERSION}-cq08-tr-rd-point-index.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _load_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    required = {
        "schema_version", "sample_id", "parent_sample_id", "track_id", "source_name", "split_name",
        "original_split_name", "view_name", "label_name", "rd_frame_count", "rd_tensor_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"V1 view missing columns: {missing}: {path}")
    if frame["schema_version"].astype(str).ne(VERSION).any():
        raise ValueError(f"V1 view contains wrong schema version: {path}")
    if frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError(f"V1 view is not CQ08-pure: {path}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError(f"V1 view contains duplicate sample IDs: {path}")
    return frame


def _dataset_check(project_root: Path, config_path: Path, expected_parent_ids: set[str], expected_split: str, *, exclude_anomalous_parents: bool) -> dict[str, Any]:
    config = load_config(config_path)
    dataset = CQ08PointSequenceDataset(
        project_root,
        index_path=resolve_project_path(project_root, config.data.index_path),
        split_name=expected_split,
        max_sequence_length=config.data.max_sequence_length,
        max_rd_frames_per_point=config.data.max_rd_frames_per_point,
        rd_amplitude_center=config.data.rd_amplitude_center,
        rd_amplitude_scale=config.data.rd_amplitude_scale,
        exclude_anomalous_parents=exclude_anomalous_parents,
        sample_limit=None,
        expected_schema_version=config.schema_version,
    )
    actual = set(dataset.parent_ids)
    if actual != expected_parent_ids:
        raise ValueError(f"Dataset parent mismatch for {config_path.name}: missing={sorted(expected_parent_ids - actual)}, unexpected={sorted(actual - expected_parent_ids)}")
    loader = build_point_dataloader(dataset, batch_size=min(2, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    if batch["tr_features"].shape[-1] != 14 or batch["tr_missing_mask"].shape[-1] != 11 or tuple(batch["rd_frames"].shape[-3:]) != (2, 21, 384):
        raise ValueError(f"V1 Dataset shape mismatch for {config_path.name}")
    if batch["tr_dt_valid"].shape != batch["tr_dt_seconds"].shape:
        raise ValueError(f"V1 Dataset dt validity shape mismatch for {config_path.name}")
    if not batch["tr_features"].isfinite().all() or not batch["rd_frames"].isfinite().all():
        raise ValueError(f"V1 Dataset produced non-finite values for {config_path.name}")
    return {
        "config": str(config_path.relative_to(project_root)).replace("\\", "/"),
        "parent_count": len(dataset),
        "batch_shapes": {key: list(value.shape) for key, value in batch.items() if hasattr(value, "shape")},
    }


def validate(project_root: Path, run_dataset_checks: bool = False) -> dict[str, Any]:
    processed = project_root / "data" / "processed"
    source_path = processed / SOURCE_INDEX_NAME
    main_path = processed / MAIN_INDEX_NAME
    audit_path = processed / AUDIT_INDEX_NAME
    main_split_path = processed / MAIN_SPLIT_NAME
    audit_split_path = processed / AUDIT_SPLIT_NAME
    manifest_path = processed / MANIFEST_NAME
    contract_path = processed / CONTRACT_NAME
    summary_path = processed / SUMMARY_NAME
    paths = (source_path, main_path, audit_path, main_split_path, audit_split_path, manifest_path, contract_path, summary_path)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"V1 file missing: {path}")
    source = pd.read_csv(source_path, low_memory=False)
    main = _load_frame(main_path)
    audit = _load_frame(audit_path)
    source_parents = set(source["parent_sample_id"].astype(str))
    main_parents = set(main["parent_sample_id"].astype(str))
    audit_parents = set(audit["parent_sample_id"].astype(str))
    if audit_parents != ANOMALY_PARENT_IDS:
        raise ValueError(f"V1 audit parent set mismatch: {sorted(audit_parents)}")
    if main_parents & audit_parents:
        raise ValueError("V1 main and audit parent sets overlap")
    if main_parents | audit_parents != source_parents:
        raise ValueError("V1 main and audit parent sets do not cover V0")
    if main["view_name"].astype(str).ne("main").any() or audit["view_name"].astype(str).ne("anomaly_audit").any():
        raise ValueError("V1 view_name field is invalid")
    if main["split_name"].astype(str).isin({"audit"}).any():
        raise ValueError("V1 main contains audit split")
    if audit["split_name"].astype(str).ne("audit").any():
        raise ValueError("V1 audit split is not audit")
    if main["parent_sample_id"].astype(str).isin(ANOMALY_PARENT_IDS).any():
        raise ValueError("V1 main contains an anomaly parent")
    if len(main) + len(audit) != len(source):
        raise ValueError("V1 view point counts do not cover V0")
    source_by_id = source.set_index(source["sample_id"].astype(str))
    main_by_id = main.set_index(main["sample_id"].astype(str))
    audit_by_id = audit.set_index(audit["sample_id"].astype(str))
    main_sample_ids = set(main_by_id.index.astype(str))
    audit_sample_ids = set(audit_by_id.index.astype(str))
    source_sample_ids = set(source_by_id.index.astype(str))
    if main_sample_ids & audit_sample_ids or main_sample_ids | audit_sample_ids != source_sample_ids:
        raise ValueError("V1 sample IDs do not form a disjoint V0 partition")
    if main["original_split_name"].astype(str).tolist() != main["split_name"].astype(str).tolist():
        raise ValueError("V1 main original_split_name changed original split semantics")
    if audit["original_split_name"].astype(str).ne("train").any():
        raise ValueError("V1 audit parent unexpectedly came from a non-train split")
    if main.groupby("parent_sample_id")["split_name"].nunique().max() != 1 or audit.groupby("parent_sample_id")["split_name"].nunique().max() != 1:
        raise ValueError("V1 parent crosses split")
    main_split = json.loads(main_split_path.read_text(encoding="utf-8"))
    audit_split = json.loads(audit_split_path.read_text(encoding="utf-8"))
    main_group_ids = {str(item) for values in main_split["group_ids"].values() for item in values}
    audit_group_ids = {str(item) for values in audit_split["group_ids"].values() for item in values}
    if main_group_ids & audit_group_ids or main_group_ids | audit_group_ids != set(source["track_id"].astype(str)):
        raise ValueError("V1 split group IDs do not form a disjoint V0 partition")
    if set(audit_split["audit_parent_ids"]) != ANOMALY_PARENT_IDS:
        raise ValueError("V1 audit split does not list the configured anomaly parents")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    source_hash = _sha256(source_path)
    if source_hash != str(manifest["source_v0_index_sha256"]) or source_hash != str(summary["source_v0_index_sha256"]):
        raise ValueError("V1 source V0 hash provenance mismatch")
    if int(summary["main"]["parent_count"]) != main["parent_sample_id"].nunique() or int(summary["anomaly_audit"]["parent_count"]) != audit["parent_sample_id"].nunique():
        raise ValueError("V1 summary parent count mismatch")
    if str(contract["views"]["anomaly_audit"]["training_allowed"]).lower() != "false":
        raise ValueError("V1 anomaly audit is not marked training-forbidden")
    result: dict[str, Any] = {
        "status": "passed",
        "schema_version": VERSION,
        "source_v0_index_sha256": source_hash,
        "main": {
            "parent_count": int(main["parent_sample_id"].nunique()),
            "point_count": int(len(main)),
            "rd_frame_count": int(pd.to_numeric(main["rd_frame_count"], errors="raise").sum()),
            "split_parent_counts": {str(key): int(value) for key, value in main.groupby("split_name")["parent_sample_id"].nunique().items()},
            "label_parent_counts": {str(key): int(value) for key, value in main.groupby("label_name")["parent_sample_id"].nunique().items()},
            "index_sha256": _sha256(main_path),
        },
        "anomaly_audit": {
            "parent_count": int(audit["parent_sample_id"].nunique()),
            "point_count": int(len(audit)),
            "rd_frame_count": int(pd.to_numeric(audit["rd_frame_count"], errors="raise").sum()),
            "parent_ids": sorted(audit_parents),
            "index_sha256": _sha256(audit_path),
        },
        "v0_partition": True,
    }
    if run_dataset_checks:
        config_root = project_root / "config" / "RADAR-LN3"
        main_config = config_root / f"{VERSION}-cq08-tr-rd-main-smoke-config.json"
        audit_config = config_root / f"{VERSION}-cq08-tr-rd-anomaly-audit-config.json"
        train_main_parents = set(main.loc[main["split_name"].astype(str).eq("train"), "parent_sample_id"].astype(str))
        result["dataset_checks"] = [
            _dataset_check(project_root, main_config, train_main_parents, "train", exclude_anomalous_parents=False),
            _dataset_check(project_root, audit_config, audit_parents, "audit", exclude_anomalous_parents=False),
        ]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RADAR-LN3-V1 main and anomaly-audit views.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--run-dataset-checks", action="store_true")
    args = parser.parse_args()
    validate(args.project_root.resolve(), run_dataset_checks=args.run_dataset_checks)


if __name__ == "__main__":
    main()
