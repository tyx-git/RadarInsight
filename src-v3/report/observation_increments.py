from __future__ import annotations

import argparse
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from radar_ln3.features import (
    CausalTRObservationIncrementState,
    TR_OBSERVATION_INCREMENT_NAMES,
    TR_OBSERVATION_INCREMENT_SCHEMA,
    parse_feature_json,
    resolve_causal_tr_feature_contract,
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


def _summary(values: np.ndarray, valid: np.ndarray, clip: float) -> dict[str, Any]:
    selected = values[valid]
    if not selected.size:
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
        "valid_count": int(selected.size),
        "valid_fraction": float(valid.mean()),
        "zero_among_valid_fraction": float(np.mean(selected == 0.0)),
        "clip_fraction": float(np.mean(np.abs(selected) >= clip - 1.0e-7)),
        "mean": float(np.mean(selected)),
        "std": float(np.std(selected)),
        "p05": float(np.quantile(selected, 0.05)),
        "median": float(np.median(selected)),
        "p95": float(np.quantile(selected, 0.95)),
    }


def run(index_path: Path, output_path: Path) -> dict[str, Any]:
    required = [
        "schema_version",
        "parent_sample_id",
        "source_name",
        "split_name",
        "label_name",
        "causal_event_index",
        "tr_dt_seconds",
        "tr_dt_valid",
        "tr_dt_reason",
        "tr_feature_values_json",
    ]
    frame = pd.read_csv(index_path, usecols=required, low_memory=False)
    frame = frame[frame["split_name"].astype(str).isin({"train", "val"})].copy()
    if frame.empty:
        raise ValueError("V10 increment audit has no train/validation rows")
    if frame["schema_version"].astype(str).ne("RADAR-LN3-V3").any():
        raise ValueError("V10 increment audit requires the V3 causal-prefix index")
    if frame["source_name"].astype(str).ne("CQ08").any():
        raise ValueError("V10 increment audit received a non-CQ08 row")
    if set(frame["label_name"].astype(str)) != set(LABELS):
        raise ValueError("V10 increment audit label set changed")
    if frame.groupby("parent_sample_id")["split_name"].nunique().gt(1).any():
        raise ValueError("V10 increment audit found a parent crossing splits")
    if frame.groupby("parent_sample_id")["label_name"].nunique().gt(1).any():
        raise ValueError("V10 increment audit found a parent with multiple labels")
    frame["causal_event_index"] = pd.to_numeric(frame["causal_event_index"], errors="raise").astype(int)
    frame = frame.sort_values(["parent_sample_id", "causal_event_index"], kind="stable")

    contract = resolve_causal_tr_feature_contract(TR_OBSERVATION_INCREMENT_SCHEMA)
    values_by_scope: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    valid_by_scope: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    parent_ids_by_scope: dict[tuple[str, str], set[str]] = defaultdict(set)
    valid_by_dt: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    first_event_count = 0
    first_event_valid = 0
    invalid_nonzero = 0
    nonfinite_count = 0
    zero_dt_nonfirst_count = 0
    zero_dt_nonfirst_any_increment_valid = 0

    for parent_id, group in frame.groupby("parent_sample_id", sort=False):
        state = CausalTRObservationIncrementState()
        split = str(group["split_name"].iloc[0])
        label = str(group["label_name"].iloc[0])
        gate_group = GATE_GROUP[label]
        for row in group.itertuples(index=False):
            encoded, valid = state.update(parse_feature_json(str(row.tr_feature_values_json)))
            nonfinite_count += int((~np.isfinite(encoded)).sum())
            invalid_nonzero += int(np.sum(encoded[~valid] != 0.0))
            event_index = int(row.causal_event_index)
            if event_index == 0:
                first_event_count += 1
                first_event_valid += int(valid.any())
            dt_valid = _as_bool(row.tr_dt_valid)
            dt_group = "dt_valid" if dt_valid else "dt_invalid"
            valid_by_dt[(split, dt_group)].append(valid)
            if event_index > 0 and str(row.tr_dt_reason) == "zero_delta_at_second_precision":
                zero_dt_nonfirst_count += 1
                zero_dt_nonfirst_any_increment_valid += int(valid.any())
            for scope in ((split, label), (split, gate_group), (split, "ALL")):
                values_by_scope[scope].append(encoded)
                valid_by_scope[scope].append(valid)
                parent_ids_by_scope[scope].add(str(parent_id))

    scopes: dict[str, Any] = {}
    for (split, group_name), rows in sorted(values_by_scope.items()):
        values = np.stack(rows)
        valid = np.stack(valid_by_scope[(split, group_name)])
        scopes[f"{split}:{group_name}"] = {
            "parent_count": len(parent_ids_by_scope[(split, group_name)]),
            "event_count": int(values.shape[0]),
            "features": {
                name: _summary(values[:, index], valid[:, index], contract.clip_absolute)
                for index, name in enumerate(contract.value_names)
            },
        }

    class_valid_spread: dict[str, dict[str, float]] = {}
    for split in ("train", "val"):
        for feature_name in contract.value_names:
            fractions = {
                label: scopes[f"{split}:{label}"]["features"][feature_name]["valid_fraction"]
                for label in LABELS
            }
            class_valid_spread[f"{split}:{feature_name}"] = {
                "minimum": float(min(fractions.values())),
                "maximum": float(max(fractions.values())),
                "range": float(max(fractions.values()) - min(fractions.values())),
            }

    dt_validity: dict[str, Any] = {}
    for (split, dt_group), masks in sorted(valid_by_dt.items()):
        valid = np.stack(masks)
        dt_validity[f"{split}:{dt_group}"] = {
            "event_count": int(valid.shape[0]),
            "any_increment_valid_fraction": float(valid.any(axis=1).mean()),
            "per_feature_valid_fraction": {
                name: float(valid[:, index].mean())
                for index, name in enumerate(contract.value_names)
            },
        }

    clip_fractions = [
        item["clip_fraction"]
        for scope in scopes.values()
        for item in scope["features"].values()
        if item["clip_fraction"] is not None
    ]
    state_signature = inspect.signature(CausalTRObservationIncrementState.update)
    checks = {
        "finite": nonfinite_count == 0,
        "invalid_values_exactly_zero": invalid_nonzero == 0,
        "first_events_have_no_valid_increment": first_event_valid == 0,
        "state_signature_has_no_time_input": tuple(state_signature.parameters) == ("self", "values"),
        "zero_second_delta_nonfirst_events_can_still_form_observation_increment": (
            zero_dt_nonfirst_count > 0 and zero_dt_nonfirst_any_increment_valid == zero_dt_nonfirst_count
        ),
        "first_event_count": first_event_count,
        "zero_second_delta_nonfirst_event_count": zero_dt_nonfirst_count,
        "zero_second_delta_nonfirst_any_increment_valid_count": zero_dt_nonfirst_any_increment_valid,
        "maximum_clip_fraction_across_scopes": float(max(clip_fractions, default=0.0)),
    }
    required_checks = (
        "finite",
        "invalid_values_exactly_zero",
        "first_events_have_no_valid_increment",
        "state_signature_has_no_time_input",
        "zero_second_delta_nonfirst_events_can_still_form_observation_increment",
    )
    if not all(bool(checks[key]) for key in required_checks):
        raise AssertionError(f"V10 observation-increment audit failed: {checks}")

    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V10-observation-increment-audit-v1",
        "feature_schema": contract.as_dict(),
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
        "contract_checks": checks,
        "validity_by_original_dt_status_for_diagnostics_only": dt_validity,
        "class_valid_fraction_spread": class_valid_spread,
        "scopes": scopes,
        "interpretation_limits": [
            "observation increments are changes per report, not physical rates per second",
            "the encoder never consumes dt/dt_valid, but report cadence can still change increment magnitude across deployments",
            "valid masks are not classifier inputs; field missingness can still create zero patterns and is reported by class",
            "label-stratified distributions are diagnostic only and do not define a threshold",
            "validation labels are descriptive only; no parameter, normalization or classifier is fitted from validation",
            "UnknownTarget remains a closed-set label and this audit does not validate open-set rejection",
        ],
        "formal_training": "未执行",
        "validation_performance": "未评估",
        "test": "未执行",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["report_path"] = str(output_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V10 no-dt observation increments on CQ08 train/validation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--index",
        type=Path,
        default=Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src-v3/processed/report/RADAR-LN3-V10-causal-tr-observation-increment-audit.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    index_path = args.index if args.index.is_absolute() else root / args.index
    output_path = args.output if args.output.is_absolute() else root / args.output
    print(json.dumps(run(index_path.resolve(), output_path.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
