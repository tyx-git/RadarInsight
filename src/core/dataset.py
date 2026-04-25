from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .processed_dataset import ProcessedTrackDataset
from .samples import TrackSample


class TrackDataset:
    def __init__(self, data_root: Path | str, trainable_only: bool = True) -> None:
        self.data_root = Path(data_root)
        self.index_dataset = ProcessedTrackDataset(self.data_root, trainable_only=trainable_only)
        self.index_df = self.index_dataset.index_df.copy()

    def __len__(self) -> int:
        return len(self.index_df)

    def labels(self) -> list[str]:
        return sorted(self.index_df["label_name"].dropna().astype(str).unique().tolist())

    def sources(self) -> list[str]:
        return sorted(self.index_df["source_name"].dropna().astype(str).unique().tolist())

    def filter(
        self,
        source_name: Optional[str] = None,
        label_name: Optional[str] = None,
    ) -> "TrackDataset":
        clone = object.__new__(TrackDataset)
        clone.data_root = self.data_root
        clone.index_dataset = self.index_dataset
        clone.index_df = self.index_df
        if source_name is not None:
            clone.index_df = clone.index_df[clone.index_df["source_name"] == source_name].copy()
        if label_name is not None:
            clone.index_df = clone.index_df[clone.index_df["label_name"] == label_name].copy()
        return clone

    def iter_samples(self) -> Iterator[TrackSample]:
        project_root = self.data_root.parent
        for row in self.index_df.to_dict(orient="records"):
            yield TrackSample(
                sample_id=str(row["sample_id"]),
                source_name=str(row["source_name"]),
                label_name_raw=str(row.get("label_name_raw", row["label_name"])),
                label_name=str(row["label_name"]),
                csv_path=project_root / str(row["csv_path"]),
                rd_path=str(row.get("rd_path", "")),
                rd_summary_csv_path=str(row.get("rd_summary_csv_path", "")),
                record_count=int(row["record_count"]),
                time_seconds_min=float(row["time_seconds_min"]),
                time_seconds_max=float(row["time_seconds_max"]),
            )

    def to_frame(self) -> pd.DataFrame:
        return self.index_df.copy()
