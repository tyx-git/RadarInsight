from __future__ import annotations

from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .processed_dataset import ProcessedTrackDataset
from .samples import TrackSample
from .labels import two_layer_label_fields


def parse_required_bool(value: object, *, field_name: str) -> bool:
    """Parse persisted boolean metadata without Python string truthiness."""
    if value is None:
        raise ValueError(f"missing required boolean metadata: {field_name}")
    try:
        if pd.isna(value):
            raise ValueError(f"missing required boolean metadata: {field_name}")
    except TypeError:
        pass
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean metadata for {field_name}: {value!r}")


class TrackDataset:
    def __init__(self, data_root: Path | str, trainable_only: bool = True, index_name: str | None = None) -> None:
        self.data_root = Path(data_root)
        self.index_dataset = ProcessedTrackDataset(self.data_root, trainable_only=trainable_only, index_name=index_name)
        self.index_df = self.index_dataset.index_df.copy()

    def __len__(self) -> int:
        return len(self.index_df)

    def labels(self) -> list[str]:
        return sorted(self.index_df["label_name"].dropna().astype(str).unique().tolist())

    def fine_labels(self) -> list[str]:
        return sorted(self.index_df["fine_label"].dropna().astype(str).unique().tolist())

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
            clone.index_df = clone.index_df[
                (clone.index_df["label_name"] == label_name)
                | (clone.index_df["coarse_label"] == label_name)
                | (clone.index_df["fine_label"] == label_name)
            ].copy()
        return clone

    def iter_samples(self) -> Iterator[TrackSample]:
        project_root = self.data_root.parent
        for row in self.index_df.to_dict(orient="records"):
            label_fields = two_layer_label_fields(
                row["label_name"],
                source_name=row.get("source_name", ""),
                batch_id=row.get("batch_id"),
                label_name_raw=row.get("label_name_raw", ""),
            )
            yield TrackSample(
                sample_id=str(row["sample_id"]),
                source_name=str(row["source_name"]),
                label_name_raw=str(row.get("label_name_raw", row["label_name"])),
                label_name=str(row["label_name"]),
                coarse_label=str(row.get("coarse_label", label_fields["coarse_label"])),
                fine_label=str(row.get("fine_label", label_fields["fine_label"])),
                csv_path=project_root / str(row["csv_path"]),
                rd_path=str(row.get("rd_path", "")),
                rd_summary_csv_path=str(row.get("rd_summary_csv_path", "")),
                rd_tensor_path=str(row.get("rd_tensor_path", "")),
                rd_phase_missing=parse_required_bool(
                    row.get("rd_phase_missing"),
                    field_name="rd_phase_missing",
                ),
                height_missing=parse_required_bool(
                    row.get("height_missing"),
                    field_name="height_missing",
                ),
                record_count=int(row["record_count"]),
                time_seconds_min=float(row["time_seconds_min"]),
                time_seconds_max=float(row["time_seconds_max"]),
            )

    def to_frame(self) -> pd.DataFrame:
        return self.index_df.copy()


__all__ = [
    "TrackDataset",
]
