from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sequence_sha256(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()




def _tensor_path(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def _sample_values(values: np.ndarray, *, maximum: int = 4096) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size <= maximum:
        return flat.copy()
    stride = max(1, int(math.ceil(flat.size / maximum)))
    return flat[::stride][:maximum].copy()


def _write_new_or_identical(path: Path, payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different normalization statistics: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return "created"


def build(
    project_root: Path,
    index_path: Path,
    output_path: Path,
    split_name: str,
    *,
    allowed_labels: tuple[str, ...] | None = None,
    forbidden_labels: tuple[str, ...] = (),
    allowed_parent_ids: tuple[str, ...] | None = None,
    schema_version: str = "RADAR-LN3-V2",
    stats_name: str = "cq08_tr_rd_main_train_normalization_v1",
    scope: str = "V1 main train split; CQ08 parent/tensor set only; no validation or test statistics",
) -> dict[str, Any]:
    index_path = index_path.resolve()
    output_path = output_path.resolve()
    frame = pd.read_csv(index_path, low_memory=False)
    required = {
        "schema_version",
        "parent_sample_id",
        "split_name",
        "source_name",
        "label_name",
        "rd_present",
        "rd_tensor_path",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"normalization source index is missing columns: {missing}")
    split_frame = frame[frame["split_name"].astype(str).eq(str(split_name))].copy()
    if split_frame.empty:
        raise ValueError(f"normalization source index has no split={split_name}")
    if split_frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("normalization source contains a non-CQ08 source")
    allowed = None if allowed_labels is None else frozenset(str(value) for value in allowed_labels)
    forbidden = frozenset(str(value) for value in forbidden_labels)
    if allowed is not None and (not allowed or allowed & forbidden):
        raise ValueError("normalization allowed_labels must be non-empty and disjoint from forbidden_labels")
    selected = split_frame
    if allowed is not None:
        selected = selected[selected["label_name"].astype(str).isin(allowed)].copy()
    allowed_parents = (
        None
        if allowed_parent_ids is None
        else frozenset(str(value) for value in allowed_parent_ids)
    )
    if allowed_parents is not None:
        if not allowed_parents:
            raise ValueError("normalization allowed_parent_ids must be non-empty")
        selected = selected[
            selected["parent_sample_id"].astype(str).isin(allowed_parents)
        ].copy()
    observed_forbidden = sorted(set(selected["label_name"].astype(str)) & forbidden)
    if observed_forbidden:
        raise ValueError(f"normalization selected forbidden labels: {observed_forbidden}")
    if selected.empty:
        raise ValueError("normalization source has no rows after label filtering")
    parent_labels = (
        split_frame[["parent_sample_id", "label_name"]]
        .drop_duplicates("parent_sample_id")
        .assign(
            parent_sample_id=lambda value: value["parent_sample_id"].astype(str),
            label_name=lambda value: value["label_name"].astype(str),
        )
    )
    if split_frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("normalization source contains a parent with multiple labels")
    parents = selected["parent_sample_id"].astype(str).drop_duplicates().sort_values().tolist()
    all_parent_ids = parent_labels["parent_sample_id"].astype(str).sort_values().tolist()
    excluded_parent_ids = sorted(set(all_parent_ids) - set(parents))
    paths = (
        selected.loc[selected["rd_present"].astype(str).str.lower().isin({"true", "1", "yes"}), "rd_tensor_path"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not paths:
        raise ValueError("normalization source has no present RD tensors")
    excluded_paths = (
        split_frame.loc[
            split_frame["parent_sample_id"].astype(str).isin(excluded_parent_ids)
            & split_frame["rd_present"].astype(str).str.lower().isin({"true", "1", "yes"}),
            "rd_tensor_path",
        ]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    overlap = sorted(set(paths) & set(excluded_paths))
    if overlap:
        raise ValueError("normalization selected and excluded populations share RD tensor paths")
    total_count = 0
    total_sum = 0.0
    total_sum_sq = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    samples: list[np.ndarray] = []
    tensor_frame_count = 0
    for relative in paths:
        tensor_path = _tensor_path(project_root, relative)
        if not tensor_path.exists():
            raise FileNotFoundError(tensor_path)
        with np.load(tensor_path, allow_pickle=False) as payload:
            if "amplitude" not in payload.files:
                raise ValueError(f"RD tensor has no amplitude field: {tensor_path}")
            amplitude = np.asarray(payload["amplitude"], dtype=np.float32)
        if amplitude.ndim != 3 or tuple(amplitude.shape[1:]) != (21, 384):
            raise ValueError(f"invalid RD amplitude shape: {tensor_path}: {amplitude.shape}")
        if not np.isfinite(amplitude).all():
            raise ValueError(f"RD amplitude contains non-finite values: {tensor_path}")
        values = amplitude.astype(np.float64, copy=False)
        total_count += int(values.size)
        total_sum += float(values.sum(dtype=np.float64))
        total_sum_sq += float(np.square(values, dtype=np.float64).sum(dtype=np.float64))
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
        tensor_frame_count += int(amplitude.shape[0])
        samples.append(_sample_values(amplitude))
    sample_values = np.concatenate(samples, axis=0).astype(np.float64, copy=False)
    mean = total_sum / total_count
    variance = max(0.0, total_sum_sq / total_count - mean * mean)
    std = math.sqrt(variance)
    q01, q25, q50, q75, q99 = np.percentile(sample_values, [1, 25, 50, 75, 99]).tolist()
    robust_scale = float((q75 - q25) / 1.349) if q75 > q25 else 0.0
    scale = max(std, robust_scale, 1.0e-6)
    result: dict[str, Any] = {
        "schema_version": str(schema_version),
        "stats_name": str(stats_name),
        "scope": str(scope),
        "source_index": str(index_path.relative_to(project_root)).replace("\\", "/"),
        "source_index_sha256": _sha256(index_path),
        "source_split": str(split_name),
        "parent_count": len(parents),
        "selected_parent_ids_sha256": _sequence_sha256(parents),
        "selected_label_parent_counts": {
            str(key): int(value)
            for key, value in (
                parent_labels[parent_labels["parent_sample_id"].isin(parents)]
                .groupby("label_name")
                .size()
                .sort_index()
                .items()
            )
        },
        "allowed_labels": None if allowed is None else sorted(allowed),
        "forbidden_labels": sorted(forbidden),
        "excluded_parent_count": int(parent_labels["parent_sample_id"].nunique() - len(parents)),
        "excluded_parent_ids_sha256": _sequence_sha256(excluded_parent_ids),
        "excluded_label_parent_counts": {
            str(key): int(value)
            for key, value in (
                parent_labels[~parent_labels["parent_sample_id"].isin(parents)]
                .groupby("label_name")
                .size()
                .sort_index()
                .items()
            )
        },
        "rd_tensor_count": len(paths),
        "selected_rd_tensor_paths_sha256": _sequence_sha256(paths),
        "excluded_rd_tensor_count": len(excluded_paths),
        "excluded_rd_tensor_paths_sha256": _sequence_sha256(excluded_paths),
        "selected_excluded_rd_tensor_path_overlap_count": len(overlap),
        "rd_tensor_frame_count": tensor_frame_count,
        "amplitude_value_count": total_count,
        "amplitude": {
            "mean": mean,
            "std": std,
            "min": minimum,
            "max": maximum,
            "sample_q01": float(q01),
            "sample_q25": float(q25),
            "sample_median": float(q50),
            "sample_q75": float(q75),
            "sample_q99": float(q99),
            "center": float(q50),
            "scale": scale,
            "scale_rule": "max(population_std, sampled_IQR/1.349, 1e-6)",
            "sample_rule": "up to 4096 evenly strided amplitude values per unique train tensor",
        },
        "phase": {
            "encoding": "phase divided by pi and clipped to [-1,1]",
            "center": 0.0,
            "scale": 3.141592653589793,
        },
        "outputs": {
            "normalization_stats_path": str(output_path.relative_to(project_root)).replace("\\", "/"),
        },
    }
    if allowed_parents is not None:
        result["allowed_parent_ids_count"] = len(allowed_parents)
        result["allowed_parent_ids_sha256"] = _sequence_sha256(sorted(allowed_parents))
    result["write_status"] = _write_new_or_identical(output_path, result)
    return result



