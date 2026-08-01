from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.core.data.labels import two_layer_label_fields


EXPERT1_LABELS = {
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
}


def build_expert1_index(
    project_root: Path,
    *,
    trainable_index_name: str = "trainable_track_index.csv",
    cq08_index_name: str = "cq08_track_index.csv",
    output_index_name: str = "expert1_track_index.csv",
    summary_name: str = "expert1_summary.json",
    strip_rd_metadata: bool = False,
) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    raw_trainable_path = processed_root / trainable_index_name
    cq08_index_path = processed_root / cq08_index_name

    raw_df = pd.read_csv(raw_trainable_path)
    if "coarse_label" not in raw_df.columns or "fine_label" not in raw_df.columns:
        fields = [
            two_layer_label_fields(
                row["label_name"],
                source_name=row.get("source_name", ""),
                batch_id=row.get("batch_id"),
                label_name_raw=row.get("label_name_raw", ""),
            )
            for row in raw_df.to_dict(orient="records")
        ]
        raw_df["coarse_label"] = [item["coarse_label"] for item in fields]
        raw_df["fine_label"] = [item["fine_label"] for item in fields]
    raw_df = raw_df[raw_df["label_name"].astype(str).isin(EXPERT1_LABELS)].copy()

    cq08_df = pd.read_csv(cq08_index_path)
    if "coarse_label" not in cq08_df.columns or "fine_label" not in cq08_df.columns:
        fields = [
            two_layer_label_fields(
                row["label_name"],
                source_name=row.get("source_name", ""),
                batch_id=row.get("batch_id"),
                label_name_raw=row.get("label_name_raw", ""),
            )
            for row in cq08_df.to_dict(orient="records")
        ]
        cq08_df["coarse_label"] = [item["coarse_label"] for item in fields]
        cq08_df["fine_label"] = [item["fine_label"] for item in fields]
    cq08_df = cq08_df[cq08_df["label_name"].astype(str).isin(EXPERT1_LABELS)].copy()
    cq08_df["is_trainable"] = True
    cq08_df["exclude_reason"] = ""

    combined = pd.concat([raw_df, cq08_df], ignore_index=True)
    if strip_rd_metadata:
        for column in ("rd_path", "rd_summary_csv_path", "rd_tensor_path"):
            combined[column] = ""
        combined["rd_phase_missing"] = False
    output_index_path = processed_root / output_index_name
    combined.to_csv(output_index_path, index=False, encoding="utf-8")

    summary = {
        "total_samples": int(len(combined)),
        "source_counts": combined["source_name"].value_counts(dropna=False).to_dict(),
        "label_counts": combined["label_name"].value_counts(dropna=False).to_dict(),
        "fine_label_counts": combined["fine_label"].value_counts(dropna=False).to_dict(),
        "with_rd_count": int(
            combined["rd_tensor_path"].fillna("").astype(str).str.strip().map(lambda x: x.lower() not in {"", "nan", "none"}).sum()
        ),
        "without_rd_count": int(
            (~combined["rd_tensor_path"].fillna("").astype(str).str.strip().map(lambda x: x.lower() not in {"", "nan", "none"})).sum()
        ),
        "source_trainable_index": raw_trainable_path.relative_to(project_root).as_posix(),
        "source_cq08_index": cq08_index_path.relative_to(project_root).as_posix(),
        "expert1_index": output_index_path.relative_to(project_root).as_posix(),
        "rd_metadata_contract": (
            "not_applicable_tr_only" if strip_rd_metadata else "source_index_passthrough"
        ),
    }
    (processed_root / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Expert1 parent trajectory index.")
    parser.add_argument("--trainable-index-name", default="trainable_track_index.csv")
    parser.add_argument("--cq08-index-name", default="cq08_track_index.csv")
    parser.add_argument("--output-index-name", default="expert1_track_index.csv")
    parser.add_argument("--summary-name", default="expert1_summary.json")
    parser.add_argument("--strip-rd-metadata", action="store_true")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    summary = build_expert1_index(
        project_root,
        trainable_index_name=args.trainable_index_name,
        cq08_index_name=args.cq08_index_name,
        output_index_name=args.output_index_name,
        summary_name=args.summary_name,
        strip_rd_metadata=args.strip_rd_metadata,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
