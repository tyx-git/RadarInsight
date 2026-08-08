from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from radar_ln3.features import (
    CausalTRObservationIncrementState,
    TR_OBSERVATION_INCREMENT_NAMES,
    parse_feature_json,
)


DEFAULT_INDEX = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
DEFAULT_DETAIL = Path(
    "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/sample_detail_information.json"
)
DEFAULT_OUTPUT = Path(
    "src-v3/processed/report/RADAR-LN3-V10-gate-residual-feature-audit.json"
)
SCHEMA_VERSION = "RADAR-LN3-V10-gate-residual-feature-audit-v1"
RESIDUAL_LABELS = ("BirdTarget", "UnknownTarget")
RD_FEATURE_NAMES = (
    "velocity_bandwidth_mps",
    "zero_velocity_energy_fraction",
    "velocity_spectral_entropy",
    "top10_velocity_energy_fraction",
    "abs_peak_velocity_mps",
    "phase_coherence",
    "velocity_profile_change",
)


def _resolve(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _parse_indices(value: Any, maximum: int) -> list[int]:
    payload = json.loads(str(value))
    if not isinstance(payload, list):
        raise ValueError("RD frame indices must decode to a list")
    return [int(item) for item in payload[:maximum]]


class _RDCache:
    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root
        self._path: Path | None = None
        self._arrays: dict[str, np.ndarray] | None = None

    def load(self, relative_path: str) -> dict[str, np.ndarray]:
        path = Path(str(relative_path).replace("\\", "/"))
        if not path.is_absolute():
            path = self.project_root / path
        path = path.resolve()
        if path == self._path and self._arrays is not None:
            return self._arrays
        with np.load(path, allow_pickle=False) as payload:
            required = {"amplitude", "phase", "velocity_mps"}
            if not required.issubset(payload.files):
                raise ValueError(f"RD tensor lacks physical arrays: {path}")
            arrays = {name: np.asarray(payload[name], dtype=np.float64) for name in required}
        amplitude = arrays["amplitude"]
        phase = arrays["phase"]
        velocity = arrays["velocity_mps"]
        if amplitude.ndim != 3 or phase.shape != amplitude.shape or velocity.shape != (amplitude.shape[-1],):
            raise ValueError(f"RD tensor has an invalid shape: {path}")
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError(f"RD tensor has a non-finite value: {path}")
        self._path = path
        self._arrays = arrays
        return arrays


def _rd_event_features(
    arrays: Mapping[str, np.ndarray],
    frame_indices: list[int],
    previous_profile: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    amplitude = arrays["amplitude"]
    phase = arrays["phase"]
    velocity = arrays["velocity_mps"]
    indices = np.asarray(frame_indices, dtype=np.int64)
    if indices.size == 0 or int(indices.min()) < 0 or int(indices.max()) >= amplitude.shape[0]:
        raise ValueError("RD residual audit received invalid frame indices")
    frame_amplitude = amplitude[indices]
    frame_phase = phase[indices]
    # Canonical amplitudes are logarithmic magnitudes. Convert back to a
    # non-negative relative power scale after subtracting each frame maximum;
    # this avoids overflow and preserves all normalized physical summaries.
    relative_power = np.power(10.0, (frame_amplitude - frame_amplitude.max(axis=(1, 2), keepdims=True)) / 10.0)
    profile = relative_power.sum(axis=(0, 1))
    total = float(profile.sum())
    if total <= 0.0 or not math.isfinite(total):
        raise ValueError("RD residual audit produced zero/non-finite profile energy")
    probability = profile / total
    mean_velocity = float(np.sum(probability * velocity))
    variance = float(np.sum(probability * np.square(velocity - mean_velocity)))
    bandwidth = math.sqrt(max(0.0, variance))
    zero_energy = float(probability[np.abs(velocity) <= 2.0].sum())
    nonzero = probability[probability > 0.0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / math.log(len(probability)))
    top_count = min(10, len(probability))
    top10 = float(np.partition(probability, -top_count)[-top_count:].sum())
    peak_velocity = float(abs(velocity[int(probability.argmax())]))
    complex_signal = relative_power * np.exp(1j * frame_phase)
    phase_coherence = float(abs(complex_signal.sum()) / relative_power.sum())
    profile_change = (
        0.0
        if previous_profile is None
        else float(0.5 * np.abs(probability - previous_profile).sum())
    )
    values = np.asarray(
        [
            bandwidth,
            zero_energy,
            entropy,
            top10,
            peak_velocity,
            phase_coherence,
            profile_change,
        ],
        dtype=np.float64,
    )
    valid = np.ones((len(RD_FEATURE_NAMES),), dtype=bool)
    if previous_profile is None:
        valid[-1] = False
        values[-1] = 0.0
    if not np.isfinite(values).all():
        raise ValueError("RD residual audit produced a non-finite feature")
    return values, valid, probability


def _summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p05": None,
            "median": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("residual summary received a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "p05": float(np.quantile(array, 0.05)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tr_names = list(TR_OBSERVATION_INCREMENT_NAMES)
    return {
        "parent_count": len(rows),
        "event_count": int(sum(int(row["event_count"]) for row in rows)),
        "rd_present_event_count": int(sum(int(row["rd_present_event_count"]) for row in rows)),
        "rd_present_fraction": _summary(float(row["rd_present_fraction"]) for row in rows),
        "tr_parent_features": {
            name: _summary(float(row["tr_features"][name]) for row in rows)
            for name in tr_names
        },
        "rd_parent_features": {
            name: _summary(float(row["rd_features"][name]) for row in rows if row["rd_features"].get(name) is not None)
            for name in RD_FEATURE_NAMES
        },
        "parent_ids": [str(row["parent_sample_id"]) for row in rows],
    }


def run(project_root: Path, index_path: Path, detail_path: Path, output_path: Path) -> dict[str, Any]:
    detail = json.loads(detail_path.read_text(encoding="utf-8"))
    if detail.get("split") != "val" or detail.get("task_name") != "dbb_cu_gate":
        raise ValueError("residual audit requires V10 DBB/CU validation details")
    parent_predictions = {
        str(row["parent_sample_id"]): dict(row) for row in detail["records"]["parent"]
    }
    required_columns = [
        "parent_sample_id",
        "split_name",
        "label_name",
        "causal_event_index",
        "tr_feature_values_json",
        "rd_present",
        "rd_tensor_path",
        "rd_tensor_frame_indices_json",
    ]
    frame = pd.read_csv(index_path, usecols=required_columns, low_memory=False)
    frame = frame[frame["split_name"].astype(str).eq("val")].copy()
    frame["causal_event_index"] = pd.to_numeric(frame["causal_event_index"], errors="raise").astype(int)
    frame = frame.sort_values(["parent_sample_id", "causal_event_index"], kind="stable")
    if set(frame["parent_sample_id"].astype(str)) != set(parent_predictions):
        raise ValueError("V10 residual index and validation detail parent sets differ")

    cache = _RDCache(project_root)
    parent_rows: list[dict[str, Any]] = []
    for parent_id, group in frame.groupby("parent_sample_id", sort=False):
        parent_id = str(parent_id)
        prediction = parent_predictions[parent_id]
        label = str(group["label_name"].iloc[0])
        if label not in RESIDUAL_LABELS:
            continue
        truth = 0 if label == "BirdTarget" else 1
        if int(prediction["true_id"]) != truth:
            raise ValueError(f"residual audit found a truth mismatch: {parent_id}")
        status = "wrong" if int(prediction["pred_id"]) != truth else "correct"

        tr_state = CausalTRObservationIncrementState()
        tr_values: list[np.ndarray] = []
        tr_valid: list[np.ndarray] = []
        rd_values: list[np.ndarray] = []
        rd_valid: list[np.ndarray] = []
        previous_profile: np.ndarray | None = None
        for row in group.itertuples(index=False):
            values, valid = tr_state.update(parse_feature_json(str(row.tr_feature_values_json)))
            tr_values.append(values)
            tr_valid.append(valid)
            if str(row.rd_present).strip().lower() not in {"1", "true", "yes", "y"}:
                previous_profile = None
                continue
            arrays = cache.load(str(row.rd_tensor_path))
            indices = _parse_indices(row.rd_tensor_frame_indices_json, maximum=12)
            rd_value, rd_is_valid, previous_profile = _rd_event_features(
                arrays,
                indices,
                previous_profile,
            )
            rd_values.append(rd_value)
            rd_valid.append(rd_is_valid)

        tr_array = np.stack(tr_values)
        tr_mask = np.stack(tr_valid)
        tr_parent = {
            name: float(np.median(tr_array[:, index][tr_mask[:, index]]))
            if tr_mask[:, index].any()
            else 0.0
            for index, name in enumerate(TR_OBSERVATION_INCREMENT_NAMES)
        }
        if rd_values:
            rd_array = np.stack(rd_values)
            rd_mask = np.stack(rd_valid)
            rd_parent = {
                name: float(np.median(rd_array[:, index][rd_mask[:, index]]))
                if rd_mask[:, index].any()
                else None
                for index, name in enumerate(RD_FEATURE_NAMES)
            }
        else:
            rd_parent = {name: None for name in RD_FEATURE_NAMES}
        parent_rows.append({
            "parent_sample_id": parent_id,
            "label_name": label,
            "prediction_status": status,
            "pred_id": int(prediction["pred_id"]),
            "confidence": float(prediction["confidence"]),
            "event_count": len(group),
            "rd_present_event_count": len(rd_values),
            "rd_present_fraction": float(len(rd_values) / len(group)),
            "tr_features": tr_parent,
            "rd_features": rd_parent,
        })

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        grouped[f"{row['label_name']}:{row['prediction_status']}"] .append(row)
    expected = {
        "BirdTarget:correct": 46,
        "BirdTarget:wrong": 6,
        "UnknownTarget:correct": 9,
        "UnknownTarget:wrong": 6,
    }
    actual = {name: len(grouped.get(name, [])) for name in expected}
    if actual != expected:
        raise AssertionError(f"V10 residual groups differ from paired metrics: {actual}")

    result = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "CQ08 V3 validation BirdTarget/UnknownTarget descriptive residual audit only; "
            "no threshold, classifier, normalization or parameter fitted; test excluded"
        ),
        "index_path": str(index_path),
        "detail_path": str(detail_path),
        "group_counts": actual,
        "tr_feature_names": list(TR_OBSERVATION_INCREMENT_NAMES),
        "rd_feature_names": list(RD_FEATURE_NAMES),
        "rd_contract": {
            "frames_per_event_maximum": 12,
            "uses_V3_mapped_frames_only": True,
            "uses_physical_velocity_axis": True,
            "uses_future_parent_statistics_as_model_input": False,
            "parent_aggregation_for_diagnostics_only": "median of causal event features",
        },
        "groups": {
            name: _group_summary(grouped[name]) for name in expected
        },
        "parents": parent_rows,
        "interpretation_limits": [
            "validation labels are used descriptively and must not define a deployment threshold",
            "six wrong Bird and six wrong Unknown parents are low-support residual groups",
            "physical RD differences can motivate one new input branch but are not model-performance evidence",
            "UnknownTarget remains a closed-set label; open-set reject/abstain is not evaluated",
            "test, multi-seed, stable and formal cascade are excluded",
        ],
        "test": "未执行",
        "formal_training": "V10正式训练已由用户完成；本描述性审计未启动训练",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["output_path"] = str(output_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Describe V10 Bird/Unknown Gate residual TR/RD features.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--detail", type=Path, default=DEFAULT_DETAIL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(
        root,
        _resolve(root, args.index),
        _resolve(root, args.detail),
        _resolve(root, args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
