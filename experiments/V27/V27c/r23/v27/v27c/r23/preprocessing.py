"""Train-fitted TR clipping used only by the V27c-r23 factorial arms."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from radar_ln3.runtime import resolve_project_path


R23_CLIPPED_FIELDS = ("range_m", "speed_mps")
R23_KNOWN_LABELS = frozenset({"DroneTarget", "BirdTarget", "BalloonTarget"})


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def fit_tr_clipping(
    project_root: str | Path,
    *,
    index_path: str | Path,
    split_name: str,
    schema_version: str,
    quantile: float,
    range_lower_m: float,
    speed_lower_mps: float,
) -> dict[str, Any]:
    """Fit one-sided bounds from known train rows only.

    The fitted values are provenance, not learned parameters.  Validation rows
    are never consulted while fitting them.
    """

    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("r23 clipping quantile must be in (0.5, 1.0)")
    if range_lower_m < 0.0 or speed_lower_mps < 0.0:
        raise ValueError("r23 physical clipping lower bounds must be non-negative")
    root = Path(project_root).resolve()
    path = resolve_project_path(root, index_path, must_exist=True)
    frame = pd.read_csv(path, low_memory=False)
    required = {"schema_version", "split_name", "label_name", "parent_sample_id", "tr_feature_values_json"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"r23 clipping index is missing columns: {missing}")
    selected = frame[
        frame["schema_version"].astype(str).eq(str(schema_version))
        & frame["split_name"].astype(str).eq(str(split_name))
        & frame["label_name"].astype(str).isin(R23_KNOWN_LABELS)
    ]
    if selected.empty:
        raise ValueError("r23 clipping fit has no known training rows")
    values: dict[str, list[float]] = {name: [] for name in R23_CLIPPED_FIELDS}
    for raw in selected["tr_feature_values_json"].astype(str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("r23 clipping fit encountered invalid TR JSON") from exc
        if not isinstance(parsed, Mapping):
            raise ValueError("r23 clipping fit encountered a non-object TR JSON value")
        for name in R23_CLIPPED_FIELDS:
            value = _finite_number(parsed.get(name))
            if value is not None:
                values[name].append(value)
    if any(not values[name] for name in R23_CLIPPED_FIELDS):
        raise ValueError("r23 clipping fit lacks finite values for a clipped field")
    bounds = {
        "range_m": {
            "lower": float(range_lower_m),
            "upper": float(np.quantile(values["range_m"], quantile)),
            "fit_count": len(values["range_m"]),
        },
        "speed_mps": {
            "lower": float(speed_lower_mps),
            "upper": float(np.quantile(values["speed_mps"], quantile)),
            "fit_count": len(values["speed_mps"]),
        },
    }
    if any(
        not math.isfinite(float(item[key]))
        for item in bounds.values()
        for key in ("lower", "upper")
    ) or any(item["upper"] < item["lower"] for item in bounds.values()):
        raise ValueError("r23 clipping fit produced invalid bounds")
    return {
        "method": "known_train_only_one_sided_upper_quantile",
        "quantile": float(quantile),
        "lower_bound_policy": "physical_nonnegative",
        "index_path": str(path),
        "split_name": str(split_name),
        "schema_version": str(schema_version),
        "label_names": sorted(R23_KNOWN_LABELS),
        "parent_count": int(selected["parent_sample_id"].astype(str).nunique()),
        "event_count": int(len(selected)),
        "bounds": bounds,
    }


def clipping_audit(
    project_root: str | Path,
    *,
    index_path: str | Path,
    split_name: str,
    schema_version: str,
    clipping: Mapping[str, Any],
) -> dict[str, Any]:
    """Count changed finite values using fixed, already-fitted bounds."""

    root = Path(project_root).resolve()
    frame = pd.read_csv(resolve_project_path(root, index_path, must_exist=True), low_memory=False)
    selected = frame[
        frame["schema_version"].astype(str).eq(str(schema_version))
        & frame["split_name"].astype(str).eq(str(split_name))
        & frame["label_name"].astype(str).isin(R23_KNOWN_LABELS)
    ]
    counts = {name: 0 for name in R23_CLIPPED_FIELDS}
    finite_counts = {name: 0 for name in R23_CLIPPED_FIELDS}
    total = int(len(selected))
    for raw in selected["tr_feature_values_json"].astype(str):
        parsed = json.loads(raw)
        if not isinstance(parsed, Mapping):
            raise ValueError("r23 clipping audit encountered a non-object TR JSON value")
        for name in R23_CLIPPED_FIELDS:
            original = _finite_number(parsed.get(name))
            if original is None:
                continue
            finite_counts[name] += 1
            clipped = clip_tr_values(parsed, clipping)[name]
            if float(clipped) != original:
                counts[name] += 1
    return {
        "split_name": str(split_name),
        "parent_count": int(selected["parent_sample_id"].astype(str).nunique()),
        "event_count": total,
        "finite_counts": finite_counts,
        "changed_counts": counts,
        "changed_fraction": {
            name: float(counts[name] / finite_counts[name]) if finite_counts[name] else 0.0
            for name in R23_CLIPPED_FIELDS
        },
    }


def clip_tr_values(values: Mapping[str, Any], clipping: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copied TR record with only the registered fields clipped."""

    result = dict(values)
    bounds = clipping.get("bounds")
    if not isinstance(bounds, Mapping):
        raise ValueError("r23 clipping provenance lacks bounds")
    for name in R23_CLIPPED_FIELDS:
        item = bounds.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"r23 clipping provenance lacks {name} bounds")
        value = _finite_number(result.get(name))
        if value is None:
            continue
        lower = _finite_number(item.get("lower"))
        upper = _finite_number(item.get("upper"))
        if lower is None or upper is None or upper < lower:
            raise ValueError(f"r23 clipping provenance has invalid {name} bounds")
        result[name] = float(np.clip(value, lower, upper))
    return result


__all__ = ["R23_CLIPPED_FIELDS", "R23_KNOWN_LABELS", "clip_tr_values", "clipping_audit", "fit_tr_clipping"]