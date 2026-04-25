from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass(frozen=True)
class ProcessedSample:
    dataset_name: str
    source_name: str
    sample_id: str
    sample_kind: str
    timestamp: str
    batch_id: str
    label_name_raw: str
    label_name: str
    record_count: int
    time_seconds_min: float
    time_seconds_max: float
    csv_path: str
    rd_path: str
    rd_summary_csv_path: str = ""


class ProcessedTrackDataset:
    def __init__(self, data_root: Path | str, trainable_only: bool = False) -> None:
        self.data_root = Path(data_root)
        self.processed_root = self.data_root / "processed"
        self.index_path = self.processed_root / ("trainable_track_index.csv" if trainable_only else "track_index.csv")
        if not self.index_path.exists():
            raise FileNotFoundError(f"processed track index not found: {self.index_path}")
        self.index_df = pd.read_csv(self.index_path)

    def __len__(self) -> int:
        return len(self.index_df)

    def labels(self) -> list[str]:
        return sorted(self.index_df["label_name"].dropna().astype(str).unique().tolist())

    def sources(self) -> list[str]:
        return sorted(self.index_df["source_name"].dropna().astype(str).unique().tolist())

    def iter_samples(
        self,
        source_name: Optional[str] = None,
        label_name: Optional[str] = None,
        sample_kind: Optional[str] = None,
    ) -> Iterable[ProcessedSample]:
        df = self.index_df
        if source_name is not None:
            df = df[df["source_name"] == source_name]
        if label_name is not None:
            df = df[df["label_name"] == label_name]
        if sample_kind is not None:
            df = df[df["sample_kind"] == sample_kind]
        for row in df.to_dict(orient="records"):
            yield ProcessedSample(
                dataset_name=str(row["dataset_name"]),
                source_name=str(row["source_name"]),
                sample_id=str(row["sample_id"]),
                sample_kind=str(row["sample_kind"]),
                timestamp=str(row["timestamp"]),
                batch_id=str(row["batch_id"]),
                label_name_raw=str(row.get("label_name_raw", row["label_name"])),
                label_name=str(row["label_name"]),
                record_count=int(row["record_count"]),
                time_seconds_min=float(row["time_seconds_min"]),
                time_seconds_max=float(row["time_seconds_max"]),
                csv_path=str(row["csv_path"]),
                rd_path=str(row["rd_path"]) if not pd.isna(row["rd_path"]) else "",
                rd_summary_csv_path=str(row["rd_summary_csv_path"]) if "rd_summary_csv_path" in row and not pd.isna(row["rd_summary_csv_path"]) else "",
            )

    def get_sample(self, sample_id: str) -> ProcessedSample:
        matched = self.index_df[self.index_df["sample_id"] == sample_id]
        if matched.empty:
            raise KeyError(f"sample not found: {sample_id}")
        row = matched.iloc[0].to_dict()
        return ProcessedSample(
            dataset_name=str(row["dataset_name"]),
            source_name=str(row["source_name"]),
            sample_id=str(row["sample_id"]),
            sample_kind=str(row["sample_kind"]),
            timestamp=str(row["timestamp"]),
            batch_id=str(row["batch_id"]),
            label_name_raw=str(row.get("label_name_raw", row["label_name"])),
            label_name=str(row["label_name"]),
            record_count=int(row["record_count"]),
            time_seconds_min=float(row["time_seconds_min"]),
            time_seconds_max=float(row["time_seconds_max"]),
            csv_path=str(row["csv_path"]),
            rd_path=str(row["rd_path"]) if not pd.isna(row["rd_path"]) else "",
            rd_summary_csv_path=str(row["rd_summary_csv_path"]) if "rd_summary_csv_path" in row and not pd.isna(row["rd_summary_csv_path"]) else "",
        )

    def load_track_frame(self, sample_id: str) -> pd.DataFrame:
        sample = self.get_sample(sample_id)
        csv_path = Path(sample.csv_path)
        if not csv_path.is_absolute():
            project_root = self.data_root.parent
            csv_path = project_root / csv_path
        return pd.read_csv(csv_path)
