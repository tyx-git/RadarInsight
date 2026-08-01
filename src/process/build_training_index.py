from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.core.data.labels import two_layer_label_fields


EXCLUDED_LABELS = {"MixedLabel"}


def build_training_index(
    project_root: Path,
    *,
    track_index_name: str = "track_index.csv",
    flagged_index_name: str = "track_index_with_flags.csv",
    trainable_index_name: str = "trainable_track_index.csv",
    summary_name: str = "trainable_summary.json",
) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    track_index_path = processed_root / track_index_name
    index_df = pd.read_csv(track_index_path)
    if "coarse_label" not in index_df.columns or "fine_label" not in index_df.columns:
        fields = [
            two_layer_label_fields(
                row["label_name"],
                source_name=row.get("source_name", ""),
                batch_id=row.get("batch_id"),
                label_name_raw=row.get("label_name_raw", ""),
            )
            for row in index_df.to_dict(orient="records")
        ]
        index_df["coarse_label"] = [item["coarse_label"] for item in fields]
        index_df["fine_label"] = [item["fine_label"] for item in fields]

    eligibility = []
    for row in index_df.to_dict(orient="records"):
        reasons: list[str] = []
        label_name = str(row["label_name"])
        record_count = int(row["record_count"])
        time_span = float(row["time_seconds_max"]) - float(row["time_seconds_min"])

        if label_name in EXCLUDED_LABELS:
            reasons.append(f"excluded_label:{label_name}")
        if record_count < 2:
            reasons.append("too_few_records")
        if time_span <= 0.0:
            reasons.append("non_positive_time_span")

        eligibility.append(
            {
                **row,
                "is_trainable": len(reasons) == 0,
                "exclude_reason": ";".join(reasons),
            }
        )

    eligibility_df = pd.DataFrame(eligibility)
    trainable_df = eligibility_df[eligibility_df["is_trainable"]].copy()

    eligibility_df.to_csv(processed_root / flagged_index_name, index=False, encoding="utf-8")
    trainable_df.to_csv(processed_root / trainable_index_name, index=False, encoding="utf-8")

    summary = {
        "total_samples": int(len(eligibility_df)),
        "trainable_samples": int(len(trainable_df)),
        "excluded_samples": int(len(eligibility_df) - len(trainable_df)),
        "trainable_label_counts": trainable_df["label_name"].value_counts(dropna=False).to_dict(),
        "trainable_fine_label_counts": trainable_df["fine_label"].value_counts(dropna=False).to_dict(),
        "excluded_reason_counts": eligibility_df["exclude_reason"].value_counts(dropna=False).to_dict(),
    }
    summary.update(
        {
            "source_track_index": track_index_path.relative_to(project_root).as_posix(),
            "flagged_index": (processed_root / flagged_index_name).relative_to(project_root).as_posix(),
            "trainable_index": (processed_root / trainable_index_name).relative_to(project_root).as_posix(),
        }
    )
    (processed_root / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trainable trajectory index.")
    parser.add_argument("--track-index-name", default="track_index.csv")
    parser.add_argument("--flagged-index-name", default="track_index_with_flags.csv")
    parser.add_argument("--trainable-index-name", default="trainable_track_index.csv")
    parser.add_argument("--summary-name", default="trainable_summary.json")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]
    summary = build_training_index(
        project_root,
        track_index_name=args.track_index_name,
        flagged_index_name=args.flagged_index_name,
        trainable_index_name=args.trainable_index_name,
        summary_name=args.summary_name,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
