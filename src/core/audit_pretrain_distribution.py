from __future__ import annotations

"""Read-only diagnostics for full-track and causal-window pretraining domains.

The audit deliberately uses the same V2/V3 encoder, source-aware preparation,
and missingness mapping as the training path.  Index metadata supplies complete
population counts, while a deterministic bounded frame sample supplies feature
finite/missingness and relative-time diagnostics without creating a second data
preparation implementation.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .encoding import (
    encode_track_sequence,
    track_feature_missing_mask,
    track_sequence_feature_missing_mask,
    prepare_track_sequence_features,
)
from .features import TRACK_RELATIVE_TIME_REFERENCE, track_sequence_feature_columns


def _parse_bool(value: object, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        raise ValueError(f"missing boolean field: {field_name}")
    try:
        if pd.isna(value):
            raise ValueError(f"missing boolean field: {field_name}")
    except TypeError:
        pass
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean field {field_name}: {value!r}")


def _finite_values(values: list[float]) -> np.ndarray:
    if not values:
        return np.empty((0,), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _numeric_summary(values: list[float]) -> dict[str, float | int]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p01": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p01": float(np.quantile(finite, 0.01)),
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def summarize_track_frame(
    frame: pd.DataFrame,
    *,
    source_name: str,
    feature_schema: str = "v3",
    sample_id: str = "",
    parent_sample_id: str = "",
    window_mode: str = "full_track",
) -> dict[str, Any]:
    """Summarize one frame in the exact encoded domain used by pretraining."""
    encoded = encode_track_sequence(frame, feature_schema=feature_schema)
    prepared = prepare_track_sequence_features(
        encoded.features,
        source_name=source_name,
        feature_schema=feature_schema,
    )
    missing = track_sequence_feature_missing_mask(
        frame,
        feature_schema=feature_schema,
    )
    raw_missing = track_feature_missing_mask(frame)
    columns = track_sequence_feature_columns(feature_schema)
    try:
        relative_index = columns.index("relative_time_seconds")
    except ValueError:
        relative_index = 0
    relative_time = prepared[:, relative_index].astype(np.float64, copy=False)
    finite = np.isfinite(prepared)
    return {
        "sample_id": str(sample_id),
        "parent_sample_id": str(parent_sample_id),
        "source_name": str(source_name),
        "window_mode": str(window_mode),
        "sequence_length": int(prepared.shape[0]),
        "relative_time": _numeric_summary(relative_time.tolist()),
        "duration_seconds": float(np.max(relative_time)) if relative_time.size else 0.0,
        "encoded_finite_ratio": float(finite.mean()) if finite.size else 0.0,
        "encoded_missing_feature_ratio": float(missing.mean()) if missing.size else 0.0,
        "raw_finite_ratio": float(1.0 - raw_missing.mean()) if raw_missing.size else 0.0,
    }


def _resolve_csv_path(project_root: Path, raw_path: object) -> Path:
    path = Path(str(raw_path))
    return path if path.is_absolute() else project_root / path


def _population_summary(index_df: pd.DataFrame, *, causal: bool) -> dict[str, Any]:
    if causal:
        length = pd.to_numeric(index_df["window_point_count"], errors="coerce")
        duration = pd.to_numeric(index_df["window_end_seconds"], errors="coerce") - pd.to_numeric(
            index_df["window_start_seconds"], errors="coerce"
        )
        mode_counts = index_df["window_mode"].astype(str).value_counts().to_dict()
    else:
        length = pd.to_numeric(index_df["record_count"], errors="coerce")
        duration = pd.to_numeric(index_df["time_seconds_max"], errors="coerce") - pd.to_numeric(
            index_df["time_seconds_min"], errors="coerce"
        )
        mode_counts = {"full_track": int(len(index_df))}
    return {
        "sample_count": int(len(index_df)),
        "parent_count": int(
            index_df["parent_sample_id"].astype(str).nunique()
            if "parent_sample_id" in index_df.columns
            else index_df["sample_id"].astype(str).nunique()
        ),
        "source_counts": {
            str(key): int(value)
            for key, value in index_df["source_name"].astype(str).value_counts().to_dict().items()
        },
        "window_mode_counts": {str(key): int(value) for key, value in mode_counts.items()},
        "sequence_length": _numeric_summary(length.dropna().astype(float).tolist()),
        "duration_seconds": _numeric_summary(duration.dropna().astype(float).tolist()),
    }


def _frame_sample_summary(
    index_df: pd.DataFrame,
    *,
    project_root: Path,
    feature_schema: str,
    max_rows_per_source: int,
    causal: bool,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source_name, source_rows in index_df.groupby(index_df["source_name"].astype(str), sort=True):
        selected = source_rows.sort_values("sample_id", kind="stable").head(max_rows_per_source)
        for record in selected.to_dict(orient="records"):
            csv_path = _resolve_csv_path(project_root, record.get("csv_path", ""))
            if not csv_path.exists():
                continue
            frame = pd.read_csv(csv_path)
            if frame.empty:
                continue
            rows.append(
                summarize_track_frame(
                    frame,
                    source_name=str(record.get("source_name", source_name)),
                    feature_schema=feature_schema,
                    sample_id=str(record.get("sample_id", "")),
                    parent_sample_id=str(
                        record.get("parent_sample_id", record.get("sample_id", ""))
                    ),
                    window_mode=str(record.get("window_mode", "full_track")),
                )
            )
    source_summary: dict[str, Any] = {}
    for source_name in sorted({str(row["source_name"]) for row in rows}):
        source_rows = [row for row in rows if str(row["source_name"]) == source_name]
        source_summary[source_name] = _summarize_frame_records(source_rows)
    return {
        "sample_count": int(len(rows)),
        "max_rows_per_source": int(max_rows_per_source),
        "by_source": source_summary,
        "overall": _summarize_frame_records(rows),
    }


def _summarize_frame_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": int(len(rows)),
        "parent_count": int(len({str(row["parent_sample_id"]) for row in rows})),
        "sequence_length": _numeric_summary([float(row["sequence_length"]) for row in rows]),
        "duration_seconds": _numeric_summary([float(row["duration_seconds"]) for row in rows]),
        "relative_time_mean": _numeric_summary(
            [float(row["relative_time"]["mean"]) for row in rows]
        ),
        "relative_time_p90": _numeric_summary(
            [float(row["relative_time"]["p90"]) for row in rows]
        ),
        "encoded_finite_ratio": _numeric_summary(
            [float(row["encoded_finite_ratio"]) for row in rows]
        ),
        "encoded_missing_feature_ratio": _numeric_summary(
            [float(row["encoded_missing_feature_ratio"]) for row in rows]
        ),
        "raw_finite_ratio": _numeric_summary(
            [float(row["raw_finite_ratio"]) for row in rows]
        ),
    }


def audit_pretrain_distribution(
    data_root: Path | str,
    *,
    parent_index_name: str,
    causal_index_name: str,
    parent_manifest_path: Path | str,
    feature_schema: str = "v3",
    split_name: str = "train",
    max_rows_per_source: int = 32,
) -> dict[str, Any]:
    """Compare the same train parent set under full and causal sampling."""
    data_root = Path(data_root)
    project_root = data_root.parent
    manifest = json.loads(Path(parent_manifest_path).read_text(encoding="utf-8"))
    parent_ids = {str(value) for value in manifest.get("parent_sample_ids", [])}
    if not parent_ids:
        raise ValueError("pretraining parent manifest contains no parent_sample_ids")

    parent_index = pd.read_csv(data_root / "processed" / parent_index_name, low_memory=False)
    parent_rows = parent_index[parent_index["sample_id"].astype(str).isin(parent_ids)].copy()
    actual_parent_ids = set(parent_rows["sample_id"].astype(str))
    if actual_parent_ids != parent_ids:
        raise ValueError(
            "parent index and pretraining manifest differ: "
            f"missing={len(parent_ids - actual_parent_ids)}, "
            f"unexpected={len(actual_parent_ids - parent_ids)}"
        )

    causal_index = pd.read_csv(data_root / "processed" / causal_index_name, low_memory=False)
    causal_trainable = causal_index["is_trainable"].map(
        lambda value: _parse_bool(value, field_name="is_trainable")
    )
    causal_rows = causal_index[
        causal_trainable
        & causal_index["split_name"].astype(str).str.strip().str.lower().eq(str(split_name).lower())
        & causal_index["parent_sample_id"].astype(str).isin(parent_ids)
    ].copy()
    causal_parent_ids = set(causal_rows["parent_sample_id"].astype(str))
    if causal_parent_ids != parent_ids:
        raise ValueError(
            "causal index and pretraining manifest differ: "
            f"missing={len(parent_ids - causal_parent_ids)}, "
            f"unexpected={len(causal_parent_ids - parent_ids)}"
        )

    return {
        "schema_version": "pretrain-distribution-audit-v1",
        "feature_schema": str(feature_schema).strip().lower(),
        "relative_time_reference": TRACK_RELATIVE_TIME_REFERENCE,
        "parent_manifest": {
            "path": Path(parent_manifest_path).as_posix(),
            "parent_count": int(len(parent_ids)),
        },
        "full_track": {
            "index_name": parent_index_name,
            "population": _population_summary(parent_rows, causal=False),
            "frame_sample": _frame_sample_summary(
                parent_rows,
                project_root=project_root,
                feature_schema=feature_schema,
                max_rows_per_source=max_rows_per_source,
                causal=False,
            ),
        },
        "causal_window": {
            "index_name": causal_index_name,
            "split_name": str(split_name),
            "population": _population_summary(causal_rows, causal=True),
            "frame_sample": _frame_sample_summary(
                causal_rows,
                project_root=project_root,
                feature_schema=feature_schema,
                max_rows_per_source=max_rows_per_source,
                causal=True,
            ),
        },
        "decision": {
            "recommended_pretraining_sampling_mode": "causal_window",
            "reason": "downstream Expert1 and streaming consume bounded causal windows",
            "future_context_policy": "only rows persisted in the train split causal window are visible",
            "normalization_policy": "fit source statistics from the active pretraining sampling domain",
            "loss_and_model_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit TR-only pretraining full-track versus causal-window domains")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--parent-index-name", default="system_trainable_track_index.csv")
    parser.add_argument("--causal-index-name", default="expert1_realtime_causal_window_index_v9.csv")
    parser.add_argument("--parent-manifest-path", required=True)
    parser.add_argument("--feature-schema", default="v3")
    parser.add_argument("--split-name", default="train")
    parser.add_argument("--max-rows-per-source", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_rows_per_source < 1:
        raise ValueError("--max-rows-per-source must be positive")
    payload = audit_pretrain_distribution(
        args.data_root,
        parent_index_name=args.parent_index_name,
        causal_index_name=args.causal_index_name,
        parent_manifest_path=args.parent_manifest_path,
        feature_schema=args.feature_schema,
        split_name=args.split_name,
        max_rows_per_source=args.max_rows_per_source,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
