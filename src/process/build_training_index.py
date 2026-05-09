from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd

from src.core.data.labels import two_layer_label_fields


EXCLUDED_LABELS = {"MixedLabel"}


def build_training_index(project_root: Path) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    track_index_path = processed_root / "track_index.csv"
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

    eligibility_df.to_csv(processed_root / "track_index_with_flags.csv", index=False, encoding="utf-8")
    trainable_df.to_csv(processed_root / "trainable_track_index.csv", index=False, encoding="utf-8")

    summary = {
        "total_samples": int(len(eligibility_df)),
        "trainable_samples": int(len(trainable_df)),
        "excluded_samples": int(len(eligibility_df) - len(trainable_df)),
        "trainable_label_counts": trainable_df["label_name"].value_counts(dropna=False).to_dict(),
        "trainable_fine_label_counts": trainable_df["fine_label"].value_counts(dropna=False).to_dict(),
        "excluded_reason_counts": eligibility_df["exclude_reason"].value_counts(dropna=False).to_dict(),
    }
    (processed_root / "trainable_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary = build_training_index(project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
