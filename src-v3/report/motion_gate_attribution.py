from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from radar_ln3.features import TR_MOTION_NAMES
from radar_ln3.task_models import (
    BASE_LABELS,
    LiquidTaskClassifier,
    LiquidTaskConfig,
    TaskSequenceDataset,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    summarize_task_records,
    verify_task_data_contract,
)


SCHEMA_VERSION = "RADAR-LN3-V9-motion-gate-attribution-v1"
DEFAULT_V8_CONFIG = Path("config/RADAR-LN3/RADAR-LN3-V8-cq08-liquid-dbb-cu-gate-config.json")
DEFAULT_V9_CONFIG = Path("config/RADAR-LN3/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-config.json")
DEFAULT_V8_CHECKPOINT = Path("artifacts/RADAR-LN3-V8-cq08-liquid-dbb-cu-gate-formal/best_model.pt")
DEFAULT_V9_CHECKPOINT = Path("artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/best_model.pt")
DEFAULT_OUTPUT = Path("src-v3/processed/report/RADAR-LN3-V9-motion-gate-attribution.json")
DEFAULT_MOTION_AUDIT = Path("src-v3/processed/report/RADAR-LN3-V9-causal-tr-motion-feature-audit.json")
DEFAULT_ORIGINAL_ECHO_AUDIT = Path("src-v3/processed/report/RADAR-LN3-original-echo-audit-summary.json")


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"count": 0, "mean": None, "median": None, "minimum": None, "maximum": None}
    if not np.isfinite(array).all():
        raise ValueError("diagnostic summary received a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _metric_value(metrics: Mapping[str, Any], class_name: str, metric: str) -> float:
    for row in metrics["per_class"]:
        if row["class"] == class_name:
            return float(row[metric])
    raise KeyError(f"metric class not found: {class_name}")


def _gate_true_id(label_name: str) -> int:
    return 0 if label_name in {"DroneTarget", "BirdTarget", "BalloonTarget"} else 1


def _route_summary(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {label: [] for label in BASE_LABELS}
    for record in records:
        grouped[str(record["label_name"])].append(record)
    result: dict[str, Any] = {}
    for label_name in BASE_LABELS:
        rows = grouped[label_name]
        dbb = sum(int(row["pred_id"]) == 0 for row in rows)
        cu = len(rows) - dbb
        true_id = _gate_true_id(label_name)
        result[label_name] = {
            "support": len(rows),
            "predicted_DBB": int(dbb),
            "predicted_CU": int(cu),
            "predicted_CU_fraction": float(cu / len(rows)) if rows else None,
            "gate_correct": int(sum(int(row["pred_id"]) == true_id for row in rows)),
            "gate_error": int(sum(int(row["pred_id"]) != true_id for row in rows)),
            "mean_probability_CU": float(mean(float(row["probability_cu"]) for row in rows)) if rows else None,
            "mean_confidence": float(mean(float(row["confidence"]) for row in rows)) if rows else None,
        }
    return result


def _fixed_group(
    records: Iterable[Mapping[str, Any]],
    *,
    field: str,
    bounds: tuple[tuple[str, float, float, bool], ...],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    rows = list(records)
    result: dict[str, Any] = {}
    assigned: set[str] = set()
    for name, lower, upper, include_upper in bounds:
        selected = [
            row
            for row in rows
            if row.get(field) is not None
            and float(row[field]) >= lower
            and (float(row[field]) <= upper if include_upper else float(row[field]) < upper)
        ]
        assigned.update(str(row["record_key"]) for row in selected)
        result[name] = {
            "metrics": summarize_task_records(selected, class_names=class_names, scope=name),
            "base_label_routes": _route_summary(selected),
        }
    missing = [row for row in rows if row.get(field) is None]
    if missing:
        result["not_available"] = {
            "metrics": summarize_task_records(missing, class_names=class_names, scope="not_available"),
            "base_label_routes": _route_summary(missing),
        }
        assigned.update(str(row["record_key"]) for row in missing)
    if len(assigned) != len(rows):
        raise ValueError(f"grouping by {field} did not assign every record exactly once")
    return result


def _scenario_summary(
    window_records: list[dict[str, Any]],
    parent_records: list[dict[str, Any]],
    *,
    class_names: tuple[str, ...],
    dt_invalid_event_count: int,
) -> dict[str, Any]:
    length_bounds = (
        ("length_1_8", 1.0, 9.0, False),
        ("length_9_16", 9.0, 17.0, False),
        ("length_17_24", 17.0, 25.0, False),
        ("length_25_29", 25.0, 29.0, True),
    )
    fraction_bounds = (
        ("fraction_0_0.25", 0.0, 0.25, False),
        ("fraction_0.25_0.50", 0.25, 0.50, False),
        ("fraction_0.50_0.75", 0.50, 0.75, False),
        ("fraction_0.75_1.00", 0.75, 1.00, True),
    )
    confidence_bounds = (
        ("confidence_0_0.60", 0.0, 0.60, False),
        ("confidence_0.60_0.80", 0.60, 0.80, False),
        ("confidence_0.80_0.90", 0.80, 0.90, False),
        ("confidence_0.90_1.00", 0.90, 1.00, True),
    )
    return {
        "window_metrics": summarize_task_records(window_records, class_names=class_names, scope="causal_prefix_windows"),
        "parent_metrics": summarize_task_records(parent_records, class_names=class_names, scope="parent_final_prefix"),
        "window_base_label_routes": _route_summary(window_records),
        "parent_base_label_routes": _route_summary(parent_records),
        "parent_length_groups": _fixed_group(
            parent_records,
            field="window_length",
            bounds=length_bounds,
            class_names=class_names,
        ),
        "parent_dt_valid_fraction_groups": _fixed_group(
            parent_records,
            field="dt_valid_fraction_excluding_first",
            bounds=fraction_bounds,
            class_names=class_names,
        ),
        "parent_motion_valid_fraction_groups": _fixed_group(
            parent_records,
            field="motion_valid_fraction_excluding_first",
            bounds=fraction_bounds,
            class_names=class_names,
        ),
        "parent_confidence_groups": _fixed_group(
            parent_records,
            field="confidence",
            bounds=confidence_bounds,
            class_names=class_names,
        ),
        "dt_invalid_event_count": int(dt_invalid_event_count),
        "parent_records": parent_records,
    }


def _evaluate(
    model: LiquidTaskClassifier,
    dataset: TaskSequenceDataset,
    *,
    device: torch.device,
    batch_size: int,
    motion_replacements: Mapping[str, np.ndarray] | None,
    collect_motion: bool,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray]]]:
    model.eval()
    loader = build_task_dataloader(dataset, batch_size=batch_size, shuffle=False)
    window_records: list[dict[str, Any]] = []
    parent_records: list[dict[str, Any]] = []
    observed_motion: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    dt_invalid_event_count = 0
    with torch.inference_mode():
        for batch in loader:
            motion_values = batch["tr_motion_features"]
            motion_mask = batch["tr_motion_valid_mask"]
            if model.tr_motion_dim:
                if motion_replacements is not None:
                    replaced = torch.zeros_like(motion_values)
                    for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                        length = int(batch["lengths"][batch_index].item())
                        replacement = np.asarray(motion_replacements[str(parent_id)], dtype=np.float32)
                        if replacement.shape != (length, model.tr_motion_dim):
                            raise ValueError(f"replacement shape mismatch for parent={parent_id}")
                        replaced[batch_index, :length] = torch.from_numpy(replacement)
                    motion_values = replaced
                if collect_motion:
                    for batch_index, parent_id in enumerate(batch["parent_sample_ids"]):
                        length = int(batch["lengths"][batch_index].item())
                        observed_motion[str(parent_id)] = (
                            batch["tr_motion_features"][batch_index, :length].numpy().copy(),
                            motion_mask[batch_index, :length].numpy().copy(),
                        )
            tensor_batch = {
                key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            kwargs: dict[str, torch.Tensor] = {
                "tr_features": tensor_batch["tr_features"],
                "tr_missing_mask": tensor_batch["tr_missing_mask"],
                "tr_dt_seconds": tensor_batch["tr_dt_seconds"],
                "tr_dt_valid": tensor_batch["tr_dt_valid"],
                "rd_frames": tensor_batch["rd_frames"],
                "rd_frame_mask": tensor_batch["rd_frame_mask"],
                "rd_quality": tensor_batch["rd_quality"],
                "tr_padding_mask": tensor_batch["tr_padding_mask"],
            }
            if model.tr_motion_dim:
                kwargs["tr_motion_features"] = motion_values.to(device, non_blocking=device.type == "cuda")
                kwargs["tr_motion_valid_mask"] = tensor_batch["tr_motion_valid_mask"]
            output = model(**kwargs)
            probabilities = torch.softmax(output["sequence_logits"].float(), dim=-1).cpu()
            dt_invalid_event_count += int(output["dt_invalid_mask"].sum().item())
            for batch_index, parent_id_value in enumerate(batch["parent_sample_ids"]):
                parent_id = str(parent_id_value)
                length = int(batch["lengths"][batch_index].item())
                true_id = int(batch["labels"][batch_index].item())
                label_name = str(batch["label_names"][batch_index])
                dt_valid = batch["tr_dt_valid"][batch_index, :length].numpy().astype(bool)
                original_motion_mask = batch["tr_motion_valid_mask"][batch_index, :length].numpy().astype(bool)
                dt_fraction = float(dt_valid[1:].mean()) if length > 1 else None
                motion_fraction = float(original_motion_mask[1:].mean()) if length > 1 and original_motion_mask.shape[1] else None
                final: dict[str, Any] | None = None
                for event_index in range(length):
                    event_probability = probabilities[batch_index, event_index]
                    pred_id = int(event_probability.argmax().item())
                    prefix_dt_fraction = float(dt_valid[1 : event_index + 1].mean()) if event_index > 0 else None
                    prefix_motion_fraction = (
                        float(original_motion_mask[1 : event_index + 1].mean())
                        if event_index > 0 and original_motion_mask.shape[1]
                        else None
                    )
                    final = {
                        "record_key": f"{parent_id}|{event_index}",
                        "true_id": true_id,
                        "pred_id": pred_id,
                        "confidence": float(event_probability[pred_id].item()),
                        "probability_dbb": float(event_probability[0].item()),
                        "probability_cu": float(event_probability[1].item()),
                        "parent_sample_id": parent_id,
                        "track_id": str(batch["track_ids"][batch_index]),
                        "source_name": str(batch["source_names"][batch_index]),
                        "label_name": label_name,
                        "event_index": event_index,
                        "window_length": event_index + 1,
                        "dt_valid_fraction_excluding_first": prefix_dt_fraction,
                        "motion_valid_fraction_excluding_first": prefix_motion_fraction,
                    }
                    window_records.append(dict(final))
                if final is None:
                    raise ValueError(f"empty task sequence: {parent_id}")
                final["record_key"] = parent_id
                final["dt_valid_fraction_excluding_first"] = dt_fraction
                final["motion_valid_fraction_excluding_first"] = motion_fraction
                parent_records.append(final)
    if collect_motion and len(observed_motion) != len(parent_records):
        raise ValueError("motion collection did not cover every validation parent")
    return (
        _scenario_summary(
            window_records,
            parent_records,
            class_names=tuple(model.class_names),
            dt_invalid_event_count=dt_invalid_event_count,
        ),
        observed_motion,
    )


def _zero_motion(observed: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    return {parent_id: np.zeros_like(values, dtype=np.float32) for parent_id, (values, _) in observed.items()}


def _load_train_motion_template(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "passed" or payload.get("schema_version") != "RADAR-LN3-V9-motion-feature-audit-v1":
        raise ValueError("train motion audit has an incompatible schema or status")
    scope = payload.get("scopes", {}).get("train:ALL", {})
    features = scope.get("features", {})
    medians = np.asarray([features[name]["median"] for name in TR_MOTION_NAMES], dtype=np.float32)
    if medians.shape != (len(TR_MOTION_NAMES),) or not np.isfinite(medians).all():
        raise ValueError("train motion median template is invalid")
    if np.any(np.abs(medians) > 5.0):
        raise ValueError("train motion median template exceeds the frozen clip")
    return medians, {
        "path": str(path),
        "sha256": _sha256(path),
        "source_scope": "train:ALL",
        "statistic": "per-feature median among valid train values",
        "feature_names": list(TR_MOTION_NAMES),
        "values": medians.tolist(),
        "uses_validation_labels": False,
        "uses_test": False,
    }


def _time_reason_audit(index_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(
        index_path,
        usecols=["split_name", "parent_sample_id", "label_name", "causal_event_index", "tr_dt_seconds", "tr_dt_valid", "tr_dt_reason"],
        low_memory=False,
    )
    frame = frame[frame["split_name"].astype(str).isin({"train", "val"})].copy()
    if frame.empty or set(frame["split_name"].astype(str)) != {"train", "val"}:
        raise ValueError("time reason audit did not load both train and validation")
    valid_text = frame["tr_dt_valid"].astype(str).str.strip().str.lower()
    frame["_dt_valid"] = valid_text.isin({"1", "true", "yes", "y"})
    frame["causal_event_index"] = pd.to_numeric(frame["causal_event_index"], errors="raise").astype(int)
    frame["tr_dt_seconds"] = pd.to_numeric(frame["tr_dt_seconds"], errors="raise").astype(float)
    result: dict[str, Any] = {}
    for split_name in ("train", "val"):
        split = frame[frame["split_name"].astype(str).eq(split_name)]
        result[split_name] = {}
        for label_name in BASE_LABELS:
            rows = split[split["label_name"].astype(str).eq(label_name)]
            nonfirst = rows[rows["causal_event_index"].ne(0)]
            reasons = Counter(nonfirst["tr_dt_reason"].astype(str))
            result[split_name][label_name] = {
                "parent_count": int(rows["parent_sample_id"].astype(str).nunique()),
                "event_count": int(len(rows)),
                "first_event_count": int(rows["causal_event_index"].eq(0).sum()),
                "nonfirst_event_count": int(len(nonfirst)),
                "nonfirst_dt_valid_count": int(nonfirst["_dt_valid"].sum()),
                "nonfirst_dt_invalid_count": int((~nonfirst["_dt_valid"]).sum()),
                "nonfirst_zero_dt_count": int(nonfirst["tr_dt_seconds"].eq(0.0).sum()),
                "nonfirst_negative_dt_count": int(nonfirst["tr_dt_seconds"].lt(0.0).sum()),
                "nonfirst_reason_counts": dict(sorted(reasons.items())),
            }
    return {
        "scope": "CQ08 V3 train and validation only; test excluded",
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "by_split_and_base_label": result,
    }


def _original_echo_time_audit(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("report_schema") != "RADAR-LN3-original-echo-audit-v1":
        raise ValueError("original echo audit schema is incompatible")
    interpretation = payload.get("interpretation", {})
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "requested_sample_count": int(payload.get("requested_sample_count", 0)),
        "actual_sample_count": int(payload.get("actual_sample_count", 0)),
        "raw_frame_count": int(payload.get("frame_summary", {}).get("raw_frame_count", 0)),
        "raw_echo_has_temporal_order": bool(interpretation.get("raw_echo_has_temporal_order")),
        "raw_echo_has_cpi_prt_timing": bool(interpretation.get("raw_echo_has_cpi_prt_timing")),
        "raw_echo_has_explicit_wall_clock_time": bool(interpretation.get("raw_echo_has_explicit_wall_clock_time")),
        "mat_has_explicit_wall_clock_field": bool(interpretation.get("mat_has_explicit_wall_clock_field")),
        "raw_frame_time_semantics": str(payload.get("raw_frame_time_semantics", "")),
        "sample_limit_warning": str(interpretation.get("sample_limit_warning", "")),
    }


def _valid_position_template(
    observed: Mapping[str, tuple[np.ndarray, np.ndarray]],
    template: np.ndarray,
) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for parent_id, (values, valid_mask) in observed.items():
        if valid_mask.shape != values.shape or values.shape[-1] != template.shape[0]:
            raise ValueError(f"motion template shape mismatch for parent={parent_id}")
        replacement = np.zeros_like(values, dtype=np.float32)
        replacement[valid_mask] = np.broadcast_to(template, values.shape)[valid_mask]
        if np.any(replacement[~valid_mask] != 0.0) or not np.isfinite(replacement).all():
            raise AssertionError("train motion template violated invalid-zero or finite constraints")
        result[parent_id] = replacement
    return result


def _cross_parent_shuffle(
    observed: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    result = {parent_id: np.zeros_like(values, dtype=np.float32) for parent_id, (values, _) in observed.items()}
    feature_audit: dict[str, Any] = {}
    for feature_index, feature_name in enumerate(TR_MOTION_NAMES):
        by_parent: dict[str, list[tuple[str, int]]] = {}
        for parent_id, (_, valid_mask) in observed.items():
            event_indices = np.flatnonzero(valid_mask[:, feature_index])
            if event_indices.size:
                shuffled_indices = event_indices.copy()
                rng.shuffle(shuffled_indices)
                by_parent[parent_id] = [(parent_id, int(event_index)) for event_index in shuffled_indices]
        parent_order = list(by_parent)
        rng.shuffle(parent_order)
        receivers = [entry for parent_id in parent_order for entry in by_parent[parent_id]]
        if not receivers:
            feature_audit[feature_name] = {"valid_slot_count": 0, "status": "no_valid_values"}
            continue
        counts = Counter(parent_id for parent_id, _ in receivers)
        maximum_parent_slots = max(counts.values())
        if len(receivers) <= 1 or maximum_parent_slots * 2 > len(receivers):
            raise ValueError(f"cannot construct a cross-parent derangement for feature={feature_name}")
        lower = maximum_parent_slots
        upper = len(receivers) - maximum_parent_slots
        shift = int(rng.integers(lower, upper + 1))
        donors = receivers[-shift:] + receivers[:-shift]
        same_parent = sum(receiver[0] == donor[0] for receiver, donor in zip(receivers, donors, strict=True))
        if same_parent:
            raise AssertionError(f"cross-parent shuffle retained a same-parent donor for feature={feature_name}")
        source_values: list[float] = []
        assigned_values: list[float] = []
        for (receiver_parent, receiver_event), (donor_parent, donor_event) in zip(receivers, donors, strict=True):
            value = float(observed[donor_parent][0][donor_event, feature_index])
            result[receiver_parent][receiver_event, feature_index] = value
            source_values.append(float(observed[receiver_parent][0][receiver_event, feature_index]))
            assigned_values.append(value)
        if not np.array_equal(np.sort(np.asarray(source_values)), np.sort(np.asarray(assigned_values))):
            raise AssertionError(f"shuffle changed the marginal values for feature={feature_name}")
        feature_audit[feature_name] = {
            "valid_slot_count": len(receivers),
            "parent_count": len(counts),
            "maximum_slots_from_one_parent": int(maximum_parent_slots),
            "cyclic_shift": shift,
            "same_parent_assignment_count": int(same_parent),
            "marginal_values_preserved_exactly": True,
        }
    for parent_id, (original_values, valid_mask) in observed.items():
        shuffled = result[parent_id]
        if shuffled.shape != original_values.shape:
            raise AssertionError("shuffled motion shape changed")
        if np.any(shuffled[~valid_mask] != 0.0):
            raise AssertionError("shuffle populated an invalid motion position")
        if not np.isfinite(shuffled).all():
            raise AssertionError("shuffle produced a non-finite value")
    return result, {
        "seed": int(seed),
        "method": "per-feature global value permutation with exact receiver valid positions and cross-parent donors",
        "uses_labels": False,
        "uses_test": False,
        "feature_audit": feature_audit,
    }


def _record_map(scenario: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["parent_sample_id"]): row for row in scenario["parent_records"]}


def _paired_comparison(before: Mapping[str, Any], after: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    before_map = _record_map(before)
    after_map = _record_map(after)
    if set(before_map) != set(after_map):
        raise ValueError(f"paired comparison parent mismatch: {name}")
    rows: list[dict[str, Any]] = []
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    flip_matrix = np.zeros((2, 2), dtype=np.int64)
    probability_deltas: list[float] = []
    for parent_id in sorted(before_map):
        left = before_map[parent_id]
        right = after_map[parent_id]
        if left["label_name"] != right["label_name"] or int(left["true_id"]) != int(right["true_id"]):
            raise ValueError(f"paired comparison label mismatch: {parent_id}")
        true_id = int(left["true_id"])
        before_pred = int(left["pred_id"])
        after_pred = int(right["pred_id"])
        before_correct = before_pred == true_id
        after_correct = after_pred == true_id
        if before_correct and not after_correct:
            change = "regressed"
        elif not before_correct and after_correct:
            change = "corrected"
        elif before_pred != after_pred:
            change = "prediction_changed_same_correctness"
        else:
            change = "unchanged"
        label_name = str(left["label_name"])
        label_counts[label_name][change] += 1
        flip_matrix[before_pred, after_pred] += 1
        delta = float(right["probability_cu"]) - float(left["probability_cu"])
        probability_deltas.append(delta)
        if before_pred != after_pred or change in {"regressed", "corrected"}:
            rows.append({
                "parent_sample_id": parent_id,
                "label_name": label_name,
                "before_pred": before_pred,
                "after_pred": after_pred,
                "before_probability_cu": float(left["probability_cu"]),
                "after_probability_cu": float(right["probability_cu"]),
                "probability_cu_delta": delta,
                "change": change,
                "window_length": int(left["window_length"]),
                "dt_valid_fraction_excluding_first": left["dt_valid_fraction_excluding_first"],
                "motion_valid_fraction_excluding_first": left["motion_valid_fraction_excluding_first"],
            })
    return {
        "name": name,
        "parent_count": len(before_map),
        "flip_matrix_rows_before_columns_after_DBB_CU": flip_matrix.tolist(),
        "corrected_count": int(sum(row["change"] == "corrected" for row in rows)),
        "regressed_count": int(sum(row["change"] == "regressed" for row in rows)),
        "prediction_flip_count": int(sum(int(before_map[key]["pred_id"]) != int(after_map[key]["pred_id"]) for key in before_map)),
        "probability_cu_delta": _quantiles(probability_deltas),
        "changes_by_base_label": {
            label: dict(label_counts[label]) for label in BASE_LABELS
        },
        "changed_parent_records": rows,
    }


def _compact_metrics(scenario: Mapping[str, Any]) -> dict[str, float]:
    metrics = scenario["parent_metrics"]
    return {
        "accuracy": float(metrics["accuracy"]),
        "macro_precision": float(metrics["macro_precision"]),
        "macro_recall": float(metrics["macro_recall"]),
        "macro_f1": float(metrics["macro_f1"]),
        "dbb_precision": _metric_value(metrics, "DBB", "precision"),
        "dbb_recall": _metric_value(metrics, "DBB", "recall"),
        "dbb_f1": _metric_value(metrics, "DBB", "f1"),
        "cu_precision": _metric_value(metrics, "CU", "precision"),
        "cu_recall": _metric_value(metrics, "CU", "recall"),
        "cu_f1": _metric_value(metrics, "CU", "f1"),
    }


def _shuffle_summary(scenarios: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    compact = {name: _compact_metrics(scenario) for name, scenario in scenarios.items()}
    metric_names = next(iter(compact.values())).keys()
    aggregate = {
        metric: {
            "mean": float(mean(values[metric] for values in compact.values())),
            "minimum": float(min(values[metric] for values in compact.values())),
            "maximum": float(max(values[metric] for values in compact.values())),
        }
        for metric in metric_names
    }
    route_aggregate: dict[str, Any] = {}
    for label_name in BASE_LABELS:
        route_aggregate[label_name] = {}
        for field in ("predicted_DBB", "predicted_CU", "gate_error", "predicted_CU_fraction", "mean_probability_CU"):
            values = [scenario["parent_base_label_routes"][label_name][field] for scenario in scenarios.values()]
            route_aggregate[label_name][field] = {
                "mean": float(mean(float(value) for value in values)),
                "minimum": float(min(float(value) for value in values)),
                "maximum": float(max(float(value) for value in values)),
            }
    return {"replicate_count": len(scenarios), "parent_metrics": aggregate, "parent_base_label_routes": route_aggregate}


def _contract_comparison(v8: LiquidTaskConfig, v9: LiquidTaskConfig) -> dict[str, Any]:
    if v8.task.name != "dbb_cu_gate" or v9.task.name != "dbb_cu_gate":
        raise ValueError("motion attribution requires two DBB/CU Gate configurations")
    task_equal = asdict(v8.task) == asdict(v9.task)
    data_fields = (
        "index_path",
        "normalization_stats_path",
        "train_split_name",
        "validation_split_name",
        "test_split_name",
        "index_schema_version",
        "expected_index_sha256",
        "expected_stats_sha256",
        "max_sequence_length",
        "max_rd_frames_per_point",
        "rd_amplitude_center",
        "rd_amplitude_scale",
        "exclude_anomalous_parents",
        "rd_quality_schema",
    )
    model_fields = tuple(field for field in asdict(v8.model) if field != "tr_motion_dim")
    train_fields = (
        "batch_size",
        "learning_rate",
        "seed",
        "device",
        "epochs",
        "early_stopping_patience",
        "gradient_clip_norm",
        "checkpoint_selection_metric",
        "class_weights",
        "loss_mode",
        "allow_parameter_updates",
    )
    differences: list[dict[str, Any]] = []
    for scope, left, right, fields in (
        ("data", v8.data, v9.data, data_fields),
        ("model_except_motion", v8.model, v9.model, model_fields),
        ("train_except_paths_and_schema", v8.train, v9.train, train_fields),
    ):
        for field in fields:
            left_value = getattr(left, field)
            right_value = getattr(right, field)
            if left_value != right_value:
                differences.append({"scope": scope, "field": field, "v8": left_value, "v9": right_value})
    if not task_equal or differences:
        raise ValueError(f"V8/V9 paired contract changed outside motion: task_equal={task_equal}, differences={differences}")
    if v8.model.tr_motion_dim != 0 or v9.model.tr_motion_dim != len(TR_MOTION_NAMES):
        raise ValueError("V8/V9 motion dimensions do not represent the intended single-factor comparison")
    return {
        "paired_task_contract_equal": task_equal,
        "paired_data_fields_equal": True,
        "paired_model_fields_except_motion_equal": True,
        "paired_training_fields_equal": True,
        "single_changed_input_factor": {
            "v8_tr_motion_dim": int(v8.model.tr_motion_dim),
            "v9_tr_motion_dim": int(v9.model.tr_motion_dim),
            "v9_tr_motion_names": list(TR_MOTION_NAMES),
        },
        "intentional_non_model_differences": [
            "schema/config/checkpoint/artifact identifiers",
            "V9 data contract adds the causal motion schema while reusing the V3 index and V2 train-only RD statistics",
        ],
    }


def _reproduction_check(scenario: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    scenario_metrics = scenario["parent_metrics"]
    reference_metrics = reference["parent_metrics"]
    matrix_equal = scenario_metrics["confusion_matrix_rows_true_columns_pred"] == reference_metrics["confusion_matrix_rows_true_columns_pred"]
    scalar_fields = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    deltas = {field: abs(float(scenario_metrics[field]) - float(reference_metrics[field])) for field in scalar_fields}
    passed = matrix_equal and max(deltas.values(), default=0.0) <= 1.0e-9
    return {"passed": passed, "parent_confusion_matrix_equal": matrix_equal, "parent_metric_absolute_deltas": deltas}


def _render_markdown(report: Mapping[str, Any]) -> str:
    scenarios = report["scenarios"]
    names = [
        "v8_observed",
        "v9_observed",
        "v9_motion_zero",
        "v9_valid_positions_train_median",
        *sorted(name for name in scenarios if name.startswith("v9_motion_cross_parent_shuffle")),
    ]
    lines = [
        "# RADAR-LN3 V9 DBB/CU Gate motion归因报告",
        "",
        f"- Scope：{report['scope']}",
        f"- Device：`{report['device']}`",
        f"- V9 checkpoint SHA-256：`{report['checkpoints']['v9']['sha256']}`",
        f"- Validation：{report['validation_parent_count']} parent / {report['validation_window_count']} causal prefix windows",
        "",
        "## Parent级指标",
        "",
        "|场景|Accuracy|Macro-F1|DBB P|DBB R|DBB F1|CU P|CU R|CU F1|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in names:
        values = _compact_metrics(scenarios[name])
        lines.append(
            f"|{name}|{values['accuracy']:.6f}|{values['macro_f1']:.6f}|{values['dbb_precision']:.6f}|"
            f"{values['dbb_recall']:.6f}|{values['dbb_f1']:.6f}|{values['cu_precision']:.6f}|"
            f"{values['cu_recall']:.6f}|{values['cu_f1']:.6f}|"
        )
    lines.extend([
        "",
        "## Parent级基础五类路由错误",
        "",
        "|场景|Drone→CU|Bird→CU|Balloon→CU|Clutter→DBB|Unknown→DBB|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in names:
        routes = scenarios[name]["parent_base_label_routes"]
        lines.append(
            f"|{name}|{routes['DroneTarget']['predicted_CU']}|{routes['BirdTarget']['predicted_CU']}|"
            f"{routes['BalloonTarget']['predicted_CU']}|{routes['ClutterTarget']['predicted_DBB']}|"
            f"{routes['UnknownTarget']['predicted_DBB']}|"
        )
    paired = report["paired_comparisons"]["v8_to_v9_observed"]
    zero = report["paired_comparisons"]["v9_observed_to_motion_zero"]
    shuffle = report["shuffle_aggregate"]["parent_metrics"]
    lines.extend([
        "",
        "## 配对变化与解释边界",
        "",
        f"- V8→V9：corrected={paired['corrected_count']}，regressed={paired['regressed_count']}，prediction flips={paired['prediction_flip_count']}。",
        f"- V9原始→全零：corrected={zero['corrected_count']}，regressed={zero['regressed_count']}，prediction flips={zero['prediction_flip_count']}。",
        f"- 跨parent扰动Macro-F1均值/范围：{shuffle['macro_f1']['mean']:.6f} / [{shuffle['macro_f1']['minimum']:.6f}, {shuffle['macro_f1']['maximum']:.6f}]。",
        f"- 跨parent扰动CU F1均值/范围：{shuffle['cu_f1']['mean']:.6f} / [{shuffle['cu_f1']['minimum']:.6f}, {shuffle['cu_f1']['maximum']:.6f}]。",
        "- 全零和跨parent扰动均为同一已训练checkpoint上的post-hoc干预，不等价于重新训练消融；扰动保持每个接收parent的valid位置并使用不同parent的值，但会破坏时间连续性和跨特征物理一致性。",
        "- 本报告未读取test、未拟合阈值、未校准reject/abstain；UnknownTarget仍是闭集类别。",
    ])
    return "\n".join(lines) + "\n"


def run(
    project_root: Path,
    *,
    v8_config_path: Path,
    v9_config_path: Path,
    v8_checkpoint_path: Path,
    v9_checkpoint_path: Path,
    output_path: Path,
    motion_audit_path: Path,
    original_echo_audit_path: Path,
    device_name: str,
    batch_size: int,
    shuffle_seeds: tuple[int, ...],
) -> dict[str, Any]:
    if not shuffle_seeds:
        raise ValueError("at least one cross-parent shuffle seed is required")
    root = project_root.resolve()
    paths = (
        v8_config_path,
        v9_config_path,
        v8_checkpoint_path,
        v9_checkpoint_path,
        motion_audit_path,
        original_echo_audit_path,
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    v8_config = load_task_config(v8_config_path)
    v9_config = load_task_config(v9_config_path)
    paired_contract = _contract_comparison(v8_config, v9_config)
    v8_contract = verify_task_data_contract(root, v8_config)
    v9_contract = verify_task_data_contract(root, v9_config)
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device("cuda")
    elif device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(v9_config.train.seed))
    np.random.seed(int(v9_config.train.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(v9_config.train.seed))
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dataset = build_task_dataset(root, v9_config, split_name=v9_config.data.validation_split_name)
    if list(dataset.parent_ids) != sorted(dataset.parent_ids):
        raise ValueError("validation parent order is not deterministic")
    v8_model = LiquidTaskClassifier(v8_config).to(device)
    v9_model = LiquidTaskClassifier(v9_config).to(device)
    v8_checkpoint = load_task_checkpoint(
        v8_checkpoint_path,
        v8_model,
        expected_schema=v8_config.train.checkpoint_schema,
        map_location=device,
    )
    v9_checkpoint = load_task_checkpoint(
        v9_checkpoint_path,
        v9_model,
        expected_schema=v9_config.train.checkpoint_schema,
        map_location=device,
    )
    index_path = Path(v9_contract["paths"]["index"])
    time_reason_audit = _time_reason_audit(index_path)
    original_echo_time_audit = _original_echo_time_audit(original_echo_audit_path)
    scenarios: dict[str, dict[str, Any]] = {}
    scenarios["v8_observed"], _ = _evaluate(
        v8_model,
        dataset,
        device=device,
        batch_size=batch_size,
        motion_replacements=None,
        collect_motion=False,
    )
    scenarios["v9_observed"], observed_motion = _evaluate(
        v9_model,
        dataset,
        device=device,
        batch_size=batch_size,
        motion_replacements=None,
        collect_motion=True,
    )
    scenarios["v9_motion_zero"], _ = _evaluate(
        v9_model,
        dataset,
        device=device,
        batch_size=batch_size,
        motion_replacements=_zero_motion(observed_motion),
        collect_motion=False,
    )
    train_motion_template, train_motion_template_audit = _load_train_motion_template(motion_audit_path)
    scenarios["v9_valid_positions_train_median"], _ = _evaluate(
        v9_model,
        dataset,
        device=device,
        batch_size=batch_size,
        motion_replacements=_valid_position_template(observed_motion, train_motion_template),
        collect_motion=False,
    )
    shuffle_audits: dict[str, Any] = {}
    shuffle_scenarios: dict[str, dict[str, Any]] = {}
    for seed in shuffle_seeds:
        name = f"v9_motion_cross_parent_shuffle_seed_{seed}"
        replacement, shuffle_audit = _cross_parent_shuffle(observed_motion, seed=seed)
        scenarios[name], _ = _evaluate(
            v9_model,
            dataset,
            device=device,
            batch_size=batch_size,
            motion_replacements=replacement,
            collect_motion=False,
        )
        shuffle_audits[name] = shuffle_audit
        shuffle_scenarios[name] = scenarios[name]
    v8_reference = v8_checkpoint.get("metrics", {}).get("validation")
    v9_reference = v9_checkpoint.get("metrics", {}).get("validation")
    if not isinstance(v8_reference, Mapping) or not isinstance(v9_reference, Mapping):
        raise ValueError("best checkpoints do not contain validation reference metrics")
    paired_comparisons = {
        "v8_to_v9_observed": _paired_comparison(scenarios["v8_observed"], scenarios["v9_observed"], name="V8 observed to V9 observed"),
        "v9_observed_to_motion_zero": _paired_comparison(scenarios["v9_observed"], scenarios["v9_motion_zero"], name="V9 observed to V9 motion zero"),
        "v9_observed_to_valid_positions_train_median": _paired_comparison(
            scenarios["v9_observed"],
            scenarios["v9_valid_positions_train_median"],
            name="V9 observed to valid positions filled with train medians",
        ),
    }
    for name in sorted(shuffle_scenarios):
        paired_comparisons[f"v9_observed_to_{name}"] = _paired_comparison(
            scenarios["v9_observed"], shuffle_scenarios[name], name=f"V9 observed to {name}"
        )
    report: dict[str, Any] = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "generated_at_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "scope": "CQ08 V3 validation only; parent-final and causal-prefix Gate evaluation; test excluded; no threshold fitting",
        "device": str(device),
        "batch_size": int(batch_size),
        "validation_parent_count": len(dataset),
        "validation_window_count": int(sum(dataset.base._groups[parent_id].iloc[: dataset.base.max_sequence_length].shape[0] for parent_id in dataset.parent_ids)),
        "configs": {
            "v8": {"path": str(v8_config_path), "sha256": _sha256(v8_config_path), "name": v8_config.config_name},
            "v9": {"path": str(v9_config_path), "sha256": _sha256(v9_config_path), "name": v9_config.config_name},
        },
        "checkpoints": {
            "v8": {"path": str(v8_checkpoint_path), "sha256": _sha256(v8_checkpoint_path), "epoch": int(v8_checkpoint["epoch"]), "global_step": int(v8_checkpoint["global_step"])},
            "v9": {"path": str(v9_checkpoint_path), "sha256": _sha256(v9_checkpoint_path), "epoch": int(v9_checkpoint["epoch"]), "global_step": int(v9_checkpoint["global_step"])},
        },
        "data_contract": {
            "paired_contract": paired_contract,
            "v8_hashes": v8_contract["hashes"],
            "v9_hashes": v9_contract["hashes"],
            "same_index_sha256": v8_contract["hashes"]["index_sha256"] == v9_contract["hashes"]["index_sha256"],
            "same_train_stats_sha256": v8_contract["hashes"]["stats_sha256"] == v9_contract["hashes"]["stats_sha256"],
        },
        "reference_reproduction": {
            "v8": _reproduction_check(scenarios["v8_observed"], v8_reference),
            "v9": _reproduction_check(scenarios["v9_observed"], v9_reference),
        },
        "intervention_contract": {
            "motion_zero": "all nine motion values are zero at every valid and invalid event; the original valid mask remains supplied only for input validation/gating",
            "valid_positions_train_median": "retain every receiver valid position and replace its value with the frozen per-feature median computed by the existing train-only motion audit; invalid/padding positions stay exactly zero",
            "cross_parent_shuffle": "for each feature independently, retain every receiver valid position and the exact validation marginal values, assign every valid value from a different parent, and leave invalid/padding values exactly zero",
            "shuffle_seeds": list(shuffle_seeds),
            "limitations": [
                "post-hoc checkpoint interventions are not equivalent to retraining an ablation",
                "per-feature shuffling breaks temporal continuity and cross-feature physical consistency and can be out-of-distribution",
                "single-seed validation changes do not establish test or deployment generalization",
            ],
        },
        "shuffle_audits": shuffle_audits,
        "train_motion_template_audit": train_motion_template_audit,
        "time_reason_audit": time_reason_audit,
        "original_echo_time_audit": original_echo_time_audit,
        "scenarios": scenarios,
        "shuffle_aggregate": _shuffle_summary(shuffle_scenarios),
        "paired_comparisons": paired_comparisons,
        "decision_status": {
            "test_evaluation": "未执行",
            "multi_seed_training": "未执行",
            "train_parent_grouped_oof_rejection_calibration": "未执行",
            "formal_three_checkpoint_cascade": "未执行",
            "production_p95": "未执行",
        },
    }
    if not report["reference_reproduction"]["v8"]["passed"] or not report["reference_reproduction"]["v9"]["passed"]:
        raise AssertionError("direct validation did not reproduce the metrics embedded in a best checkpoint")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path = output_path.with_suffix(".md")
    markdown_path.write_text(_render_markdown(report), encoding="utf-8")
    compact = {
        "status": report["status"],
        "report_path": str(output_path),
        "markdown_path": str(markdown_path),
        "device": report["device"],
        "validation_parent_count": report["validation_parent_count"],
        "validation_window_count": report["validation_window_count"],
        "v8_parent_metrics": _compact_metrics(scenarios["v8_observed"]),
        "v9_parent_metrics": _compact_metrics(scenarios["v9_observed"]),
        "zero_parent_metrics": _compact_metrics(scenarios["v9_motion_zero"]),
        "train_median_template_parent_metrics": _compact_metrics(scenarios["v9_valid_positions_train_median"]),
        "shuffle_aggregate": report["shuffle_aggregate"],
        "reference_reproduction": report["reference_reproduction"],
    }
    return compact


def main() -> None:
    parser = argparse.ArgumentParser(description="Attribute V9 DBB/CU Gate gains to causal motion values versus valid-position shortcuts.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--v8-config", type=Path, default=DEFAULT_V8_CONFIG)
    parser.add_argument("--v9-config", type=Path, default=DEFAULT_V9_CONFIG)
    parser.add_argument("--v8-checkpoint", type=Path, default=DEFAULT_V8_CHECKPOINT)
    parser.add_argument("--v9-checkpoint", type=Path, default=DEFAULT_V9_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--motion-audit", type=Path, default=DEFAULT_MOTION_AUDIT)
    parser.add_argument("--original-echo-audit", type=Path, default=DEFAULT_ORIGINAL_ECHO_AUDIT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shuffle-seeds", type=int, nargs="+", default=(901, 902, 903))
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(
        root,
        v8_config_path=_resolve(root, args.v8_config),
        v9_config_path=_resolve(root, args.v9_config),
        v8_checkpoint_path=_resolve(root, args.v8_checkpoint),
        v9_checkpoint_path=_resolve(root, args.v9_checkpoint),
        output_path=_resolve(root, args.output),
        motion_audit_path=_resolve(root, args.motion_audit),
        original_echo_audit_path=_resolve(root, args.original_echo_audit),
        device_name=args.device,
        batch_size=max(1, int(args.batch_size)),
        shuffle_seeds=tuple(int(seed) for seed in args.shuffle_seeds),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
