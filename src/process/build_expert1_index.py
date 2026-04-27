from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd


EXPERT1_LABELS = {
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
}


def build_expert1_index(project_root: Path) -> Dict[str, object]:
    processed_root = project_root / "data" / "processed"
    raw_trainable_path = processed_root / "trainable_track_index.csv"
    cq08_index_path = processed_root / "cq08_track_index.csv"

    raw_df = pd.read_csv(raw_trainable_path)
    raw_df = raw_df[raw_df["label_name"].astype(str).isin(EXPERT1_LABELS)].copy()

    cq08_df = pd.read_csv(cq08_index_path)
    cq08_df = cq08_df[cq08_df["label_name"].astype(str).isin(EXPERT1_LABELS)].copy()
    cq08_df["is_trainable"] = True
    cq08_df["exclude_reason"] = ""

    combined = pd.concat([raw_df, cq08_df], ignore_index=True)
    combined.to_csv(processed_root / "expert1_track_index.csv", index=False, encoding="utf-8")

    summary = {
        "total_samples": int(len(combined)),
        "source_counts": combined["source_name"].value_counts(dropna=False).to_dict(),
        "label_counts": combined["label_name"].value_counts(dropna=False).to_dict(),
        "with_rd_count": int(
            combined["rd_tensor_path"].fillna("").astype(str).str.strip().map(lambda x: x.lower() not in {"", "nan", "none"}).sum()
        ),
        "without_rd_count": int(
            (~combined["rd_tensor_path"].fillna("").astype(str).str.strip().map(lambda x: x.lower() not in {"", "nan", "none"})).sum()
        ),
    }
    (processed_root / "expert1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    summary = build_expert1_index(project_root)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
