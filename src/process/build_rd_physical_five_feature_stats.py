from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.core.data.rd_physical_window_dataset import (
    PHYSICAL_RD_FEATURE_NAMES,
)
from src.core.data.rd_physical_five_dataset import RD_FIVE_LABELS, RD_FIVE_QUALITY_NAMES, PhysicalRDFiveWindowDataset, collate_physical_rd_five_windows, RD_FIVE_NORMALIZATION_SCHEMA_VERSION
from src.core.rd_physical_statistics import PHYSICAL_RD_STAT_DIM, PHYSICAL_RD_STATISTIC_NAMES, physical_rd_statistics


SCHEMA_VERSION = "rd_physical_five_feature_stats_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_rd_physical_five_feature_stats(
    project_root: Path,
    *,
    index_name: str = "expert1_rd_only_causal_window_index_v1_physical_v3_1.csv",
    normalization_stats_name: str = "expert1_rd_physical_train_db_normalization_five_v1.json",
    output_name: str = "expert1_rd_physical_feature_stats_five_v1.json",
    batch_size: int = 8,
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    data_root = project_root / "data"
    processed_root = data_root / "processed"
    normalization_path = processed_root / normalization_stats_name
    if not normalization_path.exists():
        raise FileNotFoundError(f"five-class normalization artifact is missing: {normalization_path}")
    normalization = json.loads(normalization_path.read_text(encoding="utf-8"))
    if normalization.get("schema_version") != RD_FIVE_NORMALIZATION_SCHEMA_VERSION:
        raise ValueError("five-class feature stats require the independent five-class normalization schema")
    amplitude = normalization.get("amplitude_db")
    if not isinstance(amplitude, dict):
        raise ValueError("five-class normalization is missing amplitude_db")
    dataset = PhysicalRDFiveWindowDataset(
        data_root,
        index_name=index_name,
        split_name="train",
        normalization_stats_name=normalization_stats_name,
        present_only=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_physical_rd_five_windows,
    )
    rows: list[np.ndarray] = []
    parent_ids: list[str] = []
    quality_rows: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            stats = physical_rd_statistics(
                batch.raw_amplitude_db,
                batch.features[:, :, 1],
                batch.features[:, :, 2],
                batch.padding_mask,
                batch.range_offset_m,
                batch.velocity_mps,
                amplitude_clip_low_db=float(amplitude["clip_low"]),
                amplitude_clip_high_db=float(amplitude["clip_high"]),
            )
            rows.append(stats.numpy())
            parent_ids.extend(batch.parent_sample_ids)
            quality_rows.append(batch.quality_metadata.numpy())
    values = np.concatenate(rows, axis=0)
    if values.shape[1] != PHYSICAL_RD_STAT_DIM or not np.isfinite(values).all():
        raise ValueError("five-class physical feature statistics contain invalid values")
    parent_group = (
        dataset.frame[["parent_sample_id", "parent_split_group_id"]]
        .drop_duplicates("parent_sample_id")
        .set_index("parent_sample_id")["parent_split_group_id"]
        .astype(str)
        .to_dict()
    )
    parent_values: dict[str, list[np.ndarray]] = {}
    for parent_id, row_values in zip(parent_ids, values):
        parent_values.setdefault(str(parent_id), []).append(row_values)
    parent_means = {key: np.mean(value, axis=0) for key, value in parent_values.items()}
    group_values: dict[str, list[np.ndarray]] = {}
    for parent_id, row_values in parent_means.items():
        group_values.setdefault(parent_group.get(parent_id, parent_id), []).append(row_values)
    balanced_values = np.stack([np.mean(group_rows, axis=0) for group_rows in group_values.values()])
    quality_values = np.concatenate(quality_rows, axis=0)
    quality_audit = {
        name: {
            "finite_ratio": float(np.isfinite(quality_values[:, index]).mean()),
            "unique_count": int(np.unique(quality_values[:, index]).size),
            "min": float(quality_values[:, index].min()),
            "median": float(np.median(quality_values[:, index])),
            "max": float(quality_values[:, index].max()),
            "std": float(quality_values[:, index].std()),
        }
        for index, name in enumerate(RD_FIVE_QUALITY_NAMES)
    }
    median = np.median(balanced_values, axis=0)
    q25 = np.quantile(balanced_values, 0.25, axis=0)
    q75 = np.quantile(balanced_values, 0.75, axis=0)
    scale = np.maximum((q75 - q25) / 1.349, 1.0e-4)
    index_path = processed_root / index_name
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": "train-only parent/content-group balanced five-class physical RD statistics; pairing status excluded from quality inputs",
        "index_name": index_name,
        "index_sha256": _sha256_file(index_path),
        "normalization_stats_name": normalization_stats_name,
        "normalization_stats_sha256": _sha256_file(normalization_path),
        "split_name": "train",
        "labels": list(RD_FIVE_LABELS),
        "feature_names": list(PHYSICAL_RD_FEATURE_NAMES),
        "generator_code_sha256": _sha256_file(Path(__file__).resolve()),
        "artifact_generator": "src/process/build_rd_physical_five_feature_stats.py",
        "quality_names": list(RD_FIVE_QUALITY_NAMES),
        "quality_metadata_names": list(RD_FIVE_QUALITY_NAMES),
        "quality_audit": quality_audit,
        "window_count": int(values.shape[0]),
        "parent_count": int(len(parent_means)),
        "content_group_count": int(len(group_values)),
        "aggregation": "window_mean -> parent_mean -> content_group_mean -> median/IQR",
        "statistic_names": list(PHYSICAL_RD_STATISTIC_NAMES),
        "center_median": [float(value) for value in median.tolist()],
        "scale_iqr_over_1_349": [float(value) for value in scale.tolist()],
        "distribution": {
            name: {
                "min": float(balanced_values[:, index].min()),
                "q01": float(np.quantile(balanced_values[:, index], 0.01)),
                "median": float(median[index]),
                "q99": float(np.quantile(balanced_values[:, index], 0.99)),
                "max": float(balanced_values[:, index].max()),
                "scale": float(scale[index]),
            }
            for index, name in enumerate(PHYSICAL_RD_STATISTIC_NAMES)
        },
        "test_evaluation_executed": False,
    }
    output_path = processed_root / output_name
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**payload, "output_path": output_path.relative_to(project_root).as_posix()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only five-class physical RD feature statistics.")
    parser.add_argument("--index-name", default="expert1_rd_only_causal_window_index_v1_physical_v3_1.csv")
    parser.add_argument("--normalization-stats-name", default="expert1_rd_physical_train_db_normalization_five_v1.json")
    parser.add_argument("--output-name", default="expert1_rd_physical_feature_stats_five_v1.json")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    result = build_rd_physical_five_feature_stats(
        Path.cwd(),
        index_name=args.index_name,
        normalization_stats_name=args.normalization_stats_name,
        output_name=args.output_name,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
