from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


EXCLUDED_LABELS = {"MixedLabel"}


def build_system_index(project_root: Path) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    base_index_path = processed_root / "track_index.csv"
    cq08_index_path = processed_root / "cq08_track_index.csv"

    base_df = pd.read_csv(base_index_path)
    cq08_df = pd.read_csv(cq08_index_path)
    combined = pd.concat([base_df, cq08_df], ignore_index=True)

    eligibility = []
    for row in combined.to_dict(orient="records"):
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
    eligibility_df.to_csv(processed_root / "system_track_index_with_flags.csv", index=False, encoding="utf-8")
    trainable_df.to_csv(processed_root / "system_trainable_track_index.csv", index=False, encoding="utf-8")

    summary = {
        "total_samples": int(len(eligibility_df)),
        "trainable_samples": int(len(trainable_df)),
        "excluded_samples": int(len(eligibility_df) - len(trainable_df)),
        "source_counts": trainable_df["source_name"].value_counts(dropna=False).to_dict(),
        "label_counts": trainable_df["label_name"].value_counts(dropna=False).to_dict(),
    }
    (processed_root / "system_trainable_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary = build_system_index(project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
