from __future__ import annotations

import argparse
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


def _resolve(root: Path, value: str | Path) -> Path:
    candidate = Path(str(value).replace("\\", "/"))
    return candidate if candidate.is_absolute() else root / candidate


def _tensor_path(root: Path, value: str) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else root / path


def _sample_values(values: np.ndarray, *, maximum: int = 4096) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size <= maximum:
        return flat.copy()
    stride = max(1, int(math.ceil(flat.size / maximum)))
    return flat[::stride][:maximum].copy()


def build(project_root: Path, index_path: Path, output_path: Path, split_name: str) -> dict[str, Any]:
    index_path = index_path.resolve()
    output_path = output_path.resolve()
    frame = pd.read_csv(index_path, low_memory=False)
    required = {"schema_version", "parent_sample_id", "split_name", "rd_present", "rd_tensor_path"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"normalization source index is missing columns: {missing}")
    selected = frame[frame["split_name"].astype(str).eq(str(split_name))].copy()
    if selected.empty:
        raise ValueError(f"normalization source index has no split={split_name}")
    if selected["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("normalization source contains a non-CQ08 source")
    parents = selected["parent_sample_id"].astype(str).drop_duplicates().sort_values().tolist()
    paths = (
        selected.loc[selected["rd_present"].astype(str).str.lower().isin({"true", "1", "yes"}), "rd_tensor_path"]
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if not paths:
        raise ValueError("normalization source has no present RD tensors")
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
        "schema_version": "RADAR-LN3-V2",
        "stats_name": "cq08_tr_rd_main_train_normalization_v1",
        "scope": "V1 main train split; CQ08 parent/tensor set only; no validation or test statistics",
        "source_index": str(index_path.relative_to(project_root)).replace("\\", "/"),
        "source_index_sha256": _sha256(index_path),
        "source_split": str(split_name),
        "parent_count": len(parents),
        "rd_tensor_count": len(paths),
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train-only RADAR-LN3 V2 RD normalization statistics.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()
    result = build(args.project_root.resolve(), _resolve(args.project_root.resolve(), args.index), _resolve(args.project_root.resolve(), args.output), args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
