from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar_ln3.features import (
    CausalTRMotionState,
    TR_MOTION_CLIP,
    TR_MOTION_NAMES,
    parse_feature_json,
)


LABELS = (
    "DroneTarget",
    "BirdTarget",
    "BalloonTarget",
    "ClutterTarget",
    "UnknownTarget",
)
GATE_GROUP = {
    "DroneTarget": "DBB",
    "BirdTarget": "DBB",
    "BalloonTarget": "DBB",
    "ClutterTarget": "CU",
    "UnknownTarget": "CU",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _summary(values: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    selected = values[valid]
    count = int(selected.size)
    if not count:
        return {
            "valid_count": 0,
            "valid_fraction": 0.0,
            "zero_among_valid_fraction": None,
            "clip_fraction": None,
            "mean": None,
            "std": None,
            "p05": None,
            "median": None,
            "p95": None,
        }
    return {
        "valid_count": count,
        "valid_fraction": float(valid.mean()),
        "zero_among_valid_fraction": float(np.mean(selected == 0.0)),
        "clip_fraction": float(np.mean(np.abs(selected) >= TR_MOTION_CLIP - 1.0e-7)),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "p05": float(np.quantile(selected, 0.05)),
        "median": float(np.median(selected)),
        "p95": float(np.quantile(selected, 0.95)),
    }


def _feature_table(values: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, Any]]:
    return {
        name: _summary(values[:, feature_index], valid[:, feature_index])
        for feature_index, name in enumerate(TR_MOTION_NAMES)
    }


def run(project_root: Path, index_path: Path, output_path: Path) -> dict[str, Any]:
    required = [
        "schema_version",
        "parent_sample_id",
        "source_name",
        "split_name",
        "label_name",
        "causal_event_index",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_feature_values_json",
    ]
    frame = pd.read_csv(index_path, usecols=required, low_memory=False)
    frame = frame[frame["split_name"].astype(str).isin({"train", "val"})].copy()
    if frame.empty:
        raise ValueError("motion audit has no train/validation rows")
    if frame["schema_version"].astype(str).ne("RADAR-LN3-V3").any():
        raise ValueError("motion audit requires the V3 causal-dt index")
    if frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("motion audit received a non-CQ08 row")
    if set(frame["label_name"].astype(str)) != set(LABELS):
        raise ValueError("motion audit label set changed")
    if frame.groupby("parent_sample_id")["split_name"].nunique().gt(1).any():
        raise ValueError("motion audit found a parent crossing splits")
    if frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("motion audit found a parent with multiple labels")
    frame["causal_event_index"] = pd.to_numeric(frame["causal_event_index"], errors="raise").astype(int)
    frame = frame.sort_values(["parent_sample_id", "causal_event_index"], kind="stable")

    values_by_scope: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    valid_by_scope: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    parent_ids_by_scope: dict[tuple[str, str], set[str]] = defaultdict(set)
    first_event_valid = 0
    first_event_count = 0
    invalid_nonzero = 0
    nonfinite_count = 0
    radial_same_error: list[float] = []
    radial_opposite_error: list[float] = []
    radial_same_sign: list[bool] = []

    for parent_id, group in frame.groupby("parent_sample_id", sort=False):
        state = CausalTRMotionState()
        previous: dict[str, Any] | None = None
        split = str(group["split_name"].iloc[0])
        label = str(group["label_name"].iloc[0])
        gate_group = GATE_GROUP[label]
        for row in group.itertuples(index=False):
            raw_values = parse_feature_json(str(row.tr_feature_values_json))
            dt_seconds = float(row.tr_dt_seconds)
            dt_valid = _as_bool(row.tr_dt_valid)
            encoded, valid = state.update(raw_values, dt_seconds=dt_seconds, dt_valid=dt_valid)
            if not np.isfinite(encoded).all():
                nonfinite_count += int((~np.isfinite(encoded)).sum())
            invalid_nonzero += int(np.sum(encoded[~valid] != 0.0))
            if int(row.causal_event_index) == 0:
                first_event_count += 1
                first_event_valid += int(valid.any())
            for scope in ((split, label), (split, gate_group), (split, "ALL")):
                values_by_scope[scope].append(encoded)
                valid_by_scope[scope].append(valid)
                parent_ids_by_scope[scope].add(str(parent_id))
            if (
                split == "train"
                and previous is not None
                and dt_valid
                and dt_seconds > 0.0
                and raw_values.get("range_m") is not None
                and previous.get("range_m") is not None
                and raw_values.get("radial_speed_mps") is not None
                and previous.get("radial_speed_mps") is not None
            ):
                range_rate = (float(raw_values["range_m"]) - float(previous["range_m"])) / dt_seconds
                interval_radial = 0.5 * (
                    float(raw_values["radial_speed_mps"]) + float(previous["radial_speed_mps"])
                )
                radial_same_error.append(abs(interval_radial - range_rate))
                radial_opposite_error.append(abs(interval_radial + range_rate))
                radial_same_sign.append(bool(np.sign(interval_radial) * np.sign(range_rate) > 0.0))
            previous = raw_values

    scopes: dict[str, Any] = {}
    for (split, group_name), rows in sorted(values_by_scope.items()):
        values = np.stack(rows, axis=0)
        valid = np.stack(valid_by_scope[(split, group_name)], axis=0)
        scopes[f"{split}:{group_name}"] = {
            "parent_count": len(parent_ids_by_scope[(split, group_name)]),
            "event_count": int(values.shape[0]),
            "features": _feature_table(values, valid),
        }

    class_valid_spread: dict[str, dict[str, float]] = {}
    for split in ("train", "val"):
        for feature_name in TR_MOTION_NAMES:
            fractions = {
                label: scopes[f"{split}:{label}"]["features"][feature_name]["valid_fraction"]
                for label in LABELS
            }
            class_valid_spread[f"{split}:{feature_name}"] = {
                "minimum": float(min(fractions.values())),
                "maximum": float(max(fractions.values())),
                "range": float(max(fractions.values()) - min(fractions.values())),
            }

    all_clip_fractions = [
        feature["clip_fraction"]
        for scope in scopes.values()
        for feature in scope["features"].values()
        if feature["clip_fraction"] is not None
    ]
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V9-motion-feature-audit-v1",
        "scope": "CQ08 V3 train and validation only; test excluded; no threshold or classifier fitted",
        "index_path": str(index_path),
        "parent_count": int(frame["parent_sample_id"].nunique()),
        "event_count": int(len(frame)),
        "split_parent_counts": {
            str(key): int(value)
            for key, value in frame.groupby("split_name")["parent_sample_id"].nunique().items()
        },
        "label_parent_counts": {
            str(key): int(value)
            for key, value in frame.groupby("label_name")["parent_sample_id"].nunique().items()
        },
        "contract_checks": {
            "finite": nonfinite_count == 0,
            "invalid_values_exactly_zero": invalid_nonzero == 0,
            "first_events_have_no_valid_motion": first_event_valid == 0,
            "first_event_count": first_event_count,
            "maximum_clip_fraction_across_scopes": float(max(all_clip_fractions, default=0.0)),
        },
        "train_radial_sign_audit": {
            "valid_pair_count": len(radial_same_error),
            "same_sign_fraction": float(np.mean(radial_same_sign)),
            "median_abs_radial_minus_range_rate_mps": float(np.median(radial_same_error)),
            "median_abs_radial_plus_range_rate_mps": float(np.median(radial_opposite_error)),
            "selected_convention": "radial_speed_mps and delta_range/dt have the same sign",
        },
        "class_valid_fraction_spread": class_valid_spread,
        "scopes": scopes,
        "interpretation_limits": [
            "valid masks are not classifier inputs, but invalid values are zeros, so residual zero-pattern shortcuts remain possible and require paired performance checks",
            "label-stratified distributions are diagnostic only and do not define a threshold",
            "validation labels are used only for descriptive auditing; no normalization or parameter is fitted from validation",
            "UnknownTarget remains a closed-set label and this audit does not validate open-set rejection",
        ],
        "formal_training": "未执行",
        "validation_performance": "未评估",
        "test": "未执行",
    }
    if not all(result["contract_checks"][key] for key in (
        "finite",
        "invalid_values_exactly_zero",
        "first_events_have_no_valid_motion",
    )):
        raise AssertionError(f"V9 motion feature audit failed a contract check: {result['contract_checks']}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(output_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V9 causal TR motion feature distributions on CQ08 train/validation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src-v3/processed/report/RADAR-LN3-V9-causal-tr-motion-feature-audit.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    index_path = args.index if args.index.is_absolute() else root / args.index
    output_path = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(run(root, index_path.resolve(), output_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
