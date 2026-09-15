"""Audit real Bird-route features before designing the next experiment.

The audit is intentionally read-only with respect to model artifacts.  It
joins the r6 route population to the CQ08 index and to the existing V21e
course, V27a stable8, and V27c motion caches.  It reports coverage and
cross-pool ranking evidence; it does not train a model or fit L3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import torch

from radar_ln3.runtime import (  # noqa: E402
    atomic_write_text,
    canonical_sha256,
    json_text,
    relative_project_path,
    sha256_file,
)


ROLES = ("representation_fit", "representation_development", "calibration")
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget")
R6_ROOT = Path("src-v3/v27/V27d/r6")
COURSE_ROOT = Path("src-v3/v27/v27_baseline/calibration/features")
MOTION_ROOT = Path("src-v3/v27/v27c/calibration/features")
INDEX_PATH = Path("data/processed/RADAR-LN3-V16-cq08-capture-content-grouped-index.csv")
OUTPUT_ROOT = Path("src-v3/v27/V27d/r7/feature-forensics")

R6_CACHE_NAMES = {
    "representation_fit": "V27d-r6-representation_fit-population.pt",
    "representation_development": "V27d-r6-representation_development-population.pt",
    "calibration": "V27d-r6-calibration-population.pt",
}
COURSE_CACHE_NAMES = {
    "representation_fit": "V27-representation_fit-mixed-course-cache.npz",
    "representation_development": "V27-representation_development-mixed-course-cache.npz",
    "calibration": "V27-calibration-mixed-course-cache.npz",
}
MOTION_CACHE_NAMES = {
    "train": "V27c-r10-train-frozen-v27a-main-motion-cache.pt",
    "val": "V27c-r10-val-frozen-v27a-main-motion-cache.pt",
}
STABLE8_NAMES = (
    "residual_positive_8_30mps_energy_fraction",
    "residual_negative_30_60mps_energy_fraction",
    "residual_positive_30_60mps_energy_fraction",
    "residual_rms_offset_mps_over_90",
    "residual_top4_resolution_cell_concentration",
    "residual_sideband_snr_db_over_30",
    "pulse_envelope_rms_clipped_0_2_over_2",
    "pulse_envelope_modulation_snr_db_over_30",
)


def _auroc(scores: Iterable[float], target: Iterable[bool]) -> float | None:
    values = np.asarray(list(scores), dtype=np.float64)
    labels = np.asarray(list(target), dtype=bool)
    if values.ndim != 1 or values.shape != labels.shape or not np.isfinite(values).all():
        raise ValueError("forensics AUROC inputs are invalid")
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    ordered = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    rank = 1.0
    while start < len(ordered):
        stop = start + 1
        while stop < len(ordered) and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = (rank + rank + stop - start - 1.0) / 2.0
        rank += stop - start
        start = stop
    rank_sum = float(ranks[labels].sum())
    return float(1.0 - (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def _auprc(scores: Iterable[float], target: Iterable[bool]) -> float | None:
    values = np.asarray(list(scores), dtype=np.float64)
    labels = np.asarray(list(target), dtype=bool)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-values, kind="mergesort")
    hits = labels[order].astype(np.float64)
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1, dtype=np.float64)
    return float((precision * hits).sum() / positives)


def _zfit(values: np.ndarray) -> tuple[float, float]:
    mean = float(np.mean(values))
    scale = float(np.std(values))
    return mean, scale if scale > 1.0e-12 else 1.0


def _zapply(values: np.ndarray, fit: tuple[float, float]) -> np.ndarray:
    return (values - fit[0]) / fit[1]


def _seal_json(path: Path, payload: Mapping[str, Any]) -> None:
    body = dict(payload)
    body["payload_sha256"] = canonical_sha256(payload)
    atomic_write_text(path, json_text(body, indent=2))


def _load_population(root: Path, role: str) -> tuple[Path, dict[str, Any]]:
    path = root / R6_ROOT / "cache" / R6_CACHE_NAMES[role]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"r6 population is not an object: {path}")
    # The r6 loader verifies schema, tensor shapes, padding, masks and cache hash.
    from v27.V27d.r6.features import load_population_cache

    return path, load_population_cache(path)


def _load_course(root: Path, role: str) -> tuple[Path, dict[str, np.ndarray]]:
    path = root / COURSE_ROOT / COURSE_CACHE_NAMES[role]
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"sample_ids", "values", "valid"}:
            raise ValueError(f"course fields differ: {path}")
        result = {name: np.asarray(archive[name]).copy() for name in archive.files}
    if result["sample_ids"].ndim != 1 or result["values"].shape != (len(result["sample_ids"]), 1):
        raise ValueError(f"course shapes differ: {path}")
    if result["valid"].shape != result["values"].shape or result["valid"].dtype != np.bool_:
        raise ValueError(f"course validity differs: {path}")
    if not np.isfinite(result["values"]).all():
        raise ValueError(f"course values are non-finite: {path}")
    return path, result


def _event_ids(index: pd.DataFrame, parent_id: str) -> list[str]:
    rows = index[index["parent_sample_id"].eq(parent_id)].sort_values("causal_event_index", kind="stable")
    event_indices = rows["causal_event_index"].astype(int).tolist()
    if event_indices != list(range(len(event_indices))):
        raise ValueError(f"causal event order is not contiguous: {parent_id}")
    return rows["sample_id"].astype(str).tolist()


def _verify_course_alignment(
    root: Path,
    role: str,
    payload: Mapping[str, Any],
    course: Mapping[str, np.ndarray],
    index: pd.DataFrame,
) -> dict[str, Any]:
    expected: list[str] = []
    for parent_id in payload["parent_ids"]:
        expected.extend(_event_ids(index, str(parent_id)))
    actual = course["sample_ids"].astype(str).tolist()
    if actual != expected:
        raise ValueError(f"course event order differs for {role}")
    path = root / COURSE_ROOT / COURSE_CACHE_NAMES[role]
    return {
        "status": "exact_event_order",
        "event_count": len(actual),
        "valid_event_count": int(course["valid"].sum()),
        "valid_fraction": float(course["valid"].mean()),
        "path": relative_project_path(root, path),
        "sha256": sha256_file(path),
    }


def _aggregate(payload: Mapping[str, Any], course: Mapping[str, np.ndarray], index: pd.DataFrame) -> pd.DataFrame:
    tensors = payload["tensors"]
    frame_values = tensors["event_frame_values"].numpy()
    frame_mask = tensors["event_frame_mask"].numpy().astype(bool)
    lengths = tensors["lengths"].numpy()
    course_positions = {str(value): row for row, value in enumerate(course["sample_ids"].tolist())}
    records: list[dict[str, Any]] = []
    for row, parent_value in enumerate(payload["parent_ids"]):
        parent_id = str(parent_value)
        length = int(lengths[row])
        event_medians: list[np.ndarray] = []
        event_valid: list[bool] = []
        for event in range(length):
            values = frame_values[row, event]
            valid = frame_mask[row, event]
            event_valid.append(bool(valid.any()))
            event_medians.append(np.median(values[valid], axis=0) if bool(valid.any()) else np.zeros(8))
        event_values = np.asarray(event_medians, dtype=np.float64)
        valid_values = event_values[np.asarray(event_valid, dtype=bool)]
        if len(valid_values):
            event_mean = valid_values.mean(axis=0)
            event_std = valid_values.std(axis=0)
            event_p90 = np.percentile(valid_values, 90, axis=0)
            event_last = valid_values[-1]
            adjacent = np.abs(np.diff(valid_values, axis=0))
            adjacent_mean = adjacent.mean(axis=0) if len(adjacent) else np.zeros(8)
            adjacent_p90 = np.percentile(adjacent, 90, axis=0) if len(adjacent) else np.zeros(8)
            prefix_mean = np.cumsum(valid_values, axis=0) / np.arange(1, len(valid_values) + 1)[:, None]
            prefix_distance = np.linalg.norm(valid_values - prefix_mean, axis=1)
        else:
            event_mean = event_std = event_p90 = event_last = np.zeros(8)
            adjacent_mean = adjacent_p90 = np.zeros(8)
            prefix_distance = np.zeros(1)
        all_frames = frame_values[row, :length][frame_mask[row, :length]]
        all_frames = np.asarray(all_frames, dtype=np.float64).reshape(-1, 8)
        if len(all_frames):
            frame_mean = all_frames.mean(axis=0)
            frame_std = all_frames.std(axis=0)
            frame_p90 = np.percentile(all_frames, 90, axis=0)
            frame_absmax = np.max(np.abs(all_frames), axis=0)
        else:
            frame_mean = frame_std = frame_p90 = frame_absmax = np.zeros(8)
        event_ids = _event_ids(index, parent_id)
        course_indices = [course_positions[event_id] for event_id in event_ids]
        course_values = course["values"][course_indices, 0].astype(np.float64)
        course_valid = course["valid"][course_indices, 0].astype(bool)
        record: dict[str, Any] = {
            "parent_id": parent_id,
            "label": str(payload["label_names"][row]),
            "candidate": bool(tensors["candidate_population"][row]),
            "positive": bool(tensors["bird_positive"][row]),
            "negative": bool(tensors["route_false_positive_negative"][row]),
            "quality": bool(tensors["quality_target_final"][row]),
            "course_valid_events": int(course_valid.sum()),
            "course_valid_fraction": float(course_valid.mean()),
            "course_mean": float(course_values[course_valid].mean()) if bool(course_valid.any()) else 0.0,
            "course_std": float(course_values[course_valid].std()) if bool(course_valid.any()) else 0.0,
            "course_p90": float(np.percentile(course_values[course_valid], 90)) if bool(course_valid.any()) else 0.0,
            "course_absmax": float(np.max(np.abs(course_values[course_valid]))) if bool(course_valid.any()) else 0.0,
            "stable_valid_events": int(sum(event_valid)),
            "stable_valid_fraction": float(np.mean(event_valid)) if event_valid else 0.0,
            "stable_frame_count": int(frame_mask[row, :length].sum()),
            "stable_frame_fraction": float(frame_mask[row, :length].sum() / max(1, length * frame_mask.shape[-1])),
            "stable_prefix_final_distance": float(prefix_distance[-1]),
            "stable_prefix_mean_distance": float(prefix_distance.mean()),
            "stable_prefix_p90_distance": float(np.percentile(prefix_distance, 90)),
            "stable_event_adj_mean_l2": float(np.linalg.norm(adjacent_mean)),
            "stable_event_adj_p90_l2": float(np.linalg.norm(adjacent_p90)),
        }
        for index_value, name in enumerate(STABLE8_NAMES):
            record.update(
                {
                    f"stable_frame_mean_{name}": float(frame_mean[index_value]),
                    f"stable_frame_std_{name}": float(frame_std[index_value]),
                    f"stable_frame_p90_{name}": float(frame_p90[index_value]),
                    f"stable_frame_absmax_{name}": float(frame_absmax[index_value]),
                    f"stable_event_mean_{name}": float(event_mean[index_value]),
                    f"stable_event_std_{name}": float(event_std[index_value]),
                    f"stable_event_p90_{name}": float(event_p90[index_value]),
                    f"stable_event_last_{name}": float(event_last[index_value]),
                    f"stable_event_adj_mean_{name}": float(adjacent_mean[index_value]),
                    f"stable_event_adj_p90_{name}": float(adjacent_p90[index_value]),
                }
            )
        records.append(record)
    return pd.DataFrame(records)


def _load_motion(root: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    provenance: dict[str, Any] = {}
    for split, name in MOTION_CACHE_NAMES.items():
        path = root / MOTION_ROOT / name
        payload = torch.load(path, map_location="cpu", weights_only=False)
        provenance[split] = {"path": relative_project_path(root, path), "sha256": sha256_file(path)}
        for row, parent_id in enumerate(payload["parent_ids"]):
            result[str(parent_id)] = {
                "split": split,
                "label": str(payload["label_names"][row]),
            }
    return result, provenance


def _motion_coverage(payload: Mapping[str, Any], motion: Mapping[str, Mapping[str, Any]], role: str) -> dict[str, Any]:
    expected_split = "val" if role == "calibration" else "train"
    parent_ids = [str(value) for value in payload["parent_ids"]]
    labels = [str(value) for value in payload["label_names"]]
    candidate = payload["tensors"]["candidate_population"].bool().tolist()
    covered = [parent_id for parent_id in parent_ids if parent_id in motion]
    unknown = [
        parent_id
        for parent_id, selected, label in zip(parent_ids, candidate, labels, strict=True)
        if selected and label == "UnknownTarget"
    ]
    return {
        "status": "complete_for_role" if len(covered) == len(parent_ids) else "incomplete_for_role",
        "expected_split": expected_split,
        "parent_support": len(parent_ids),
        "covered_parent_support": len(covered),
        "coverage_fraction": float(len(covered) / max(1, len(parent_ids))),
        "candidate_support": int(sum(candidate)),
        "candidate_covered_support": int(sum(parent_id in motion for parent_id, selected in zip(parent_ids, candidate, strict=True) if selected)),
        "candidate_unknown_support": len(unknown),
        "candidate_unknown_covered_support": int(sum(parent_id in motion for parent_id in unknown)),
        "split_mismatch_count": int(sum(motion[parent_id]["split"] != expected_split for parent_id in covered)),
        "label_mismatch_count": int(sum(motion[parent_id]["label"] != labels[parent_ids.index(parent_id)] for parent_id in covered)),
        "missing_parent_examples": [parent_id for parent_id in parent_ids if parent_id not in motion][:20],
    }


def _feature_names(frame: pd.DataFrame) -> list[str]:
    excluded = {"stable_valid_events", "stable_valid_fraction", "stable_frame_count", "stable_frame_fraction"}
    return [
        column
        for column in frame.columns
        if (column.startswith("stable_") or column.startswith("course_")) and column not in excluded
    ]


def _feature_summary(frame: pd.DataFrame, name: str, route: np.ndarray, target: np.ndarray) -> dict[str, Any] | None:
    values = frame[name].to_numpy(dtype=np.float64)
    usable = route & np.isfinite(values)
    if int(target[usable].sum()) < 2 or int((~target[usable]).sum()) < 2:
        return None
    higher = _auroc(values[usable], target[usable]) or 0.0
    lower = _auroc(-values[usable], target[usable]) or 0.0
    direction = 1.0 if higher >= lower else -1.0
    signed = direction * values[usable]
    return {
        "direction": "higher_known" if direction > 0 else "lower_known",
        "auroc": float(_auroc(signed, target[usable]) or 0.0),
        "auprc": _auprc(signed, target[usable]),
        "support": int(usable.sum()),
        "positive_support": int(target[usable].sum()),
        "negative_support": int((~target[usable]).sum()),
    }


def _transfer(fit: pd.DataFrame, other: pd.DataFrame, name: str) -> dict[str, Any] | None:
    fit_route = fit["candidate"].to_numpy(bool)
    other_route = other["candidate"].to_numpy(bool)
    fit_target = fit["positive"].to_numpy(bool)
    other_target = other["positive"].to_numpy(bool)
    fit_values = fit[name].to_numpy(dtype=np.float64)
    other_values = other[name].to_numpy(dtype=np.float64)
    fit_usable = fit_route & np.isfinite(fit_values)
    other_usable = other_route & np.isfinite(other_values)
    if int(fit_target[fit_usable].sum()) < 2 or int((~fit_target[fit_usable]).sum()) < 2:
        return None
    high = _auroc(fit_values[fit_usable], fit_target[fit_usable]) or 0.0
    low = _auroc(-fit_values[fit_usable], fit_target[fit_usable]) or 0.0
    direction = 1.0 if high >= low else -1.0
    feature_fit = direction * _zapply(fit_values[fit_usable], _zfit(fit_values[fit_usable]))
    fit_base_all = fit["r6_liquid_score"].to_numpy(dtype=np.float64)
    other_base_all = other["r6_liquid_score"].to_numpy(dtype=np.float64)
    base_fit = _zapply(fit_base_all[fit_usable], _zfit(fit_base_all[fit_route]))
    best_auc = -1.0
    best_weight = 0.0
    for weight in np.linspace(0.0, 2.0, 41):
        score = _auroc(base_fit + weight * feature_fit, fit_target[fit_usable])
        if score is not None and (score > best_auc or (score == best_auc and weight < best_weight)):
            best_auc = float(score)
            best_weight = float(weight)
    base_fit_mean_scale = _zfit(fit_base_all[fit_route])
    other_base = _zapply(other_base_all[other_usable], base_fit_mean_scale)
    other_feature = direction * _zapply(other_values[other_usable], _zfit(fit_values[fit_usable]))
    base_auc = _auroc(other_base, other_target[other_usable])
    combined_auc = _auroc(other_base + best_weight * other_feature, other_target[other_usable])
    return {
        "fit_direction": "higher_known" if direction > 0 else "lower_known",
        "fit_weight": best_weight,
        "fit_combined_auroc": best_auc,
        "other_base_auroc": base_auc,
        "other_combined_auroc": combined_auc,
        "other_delta_auroc": None if base_auc is None or combined_auc is None else float(combined_auc - base_auc),
        "other_usable_support": int(other_usable.sum()),
    }


def _liquid_scores(root: Path, payload: Mapping[str, Any]) -> np.ndarray:
    from v27.V27d.r6.training import _inputs, load_checkpoint

    model, _, _ = load_checkpoint(root, arm="liquid_treatment", seed=42)
    model.eval()
    indices = torch.arange(len(payload["parent_ids"]))
    with torch.inference_mode():
        output = model(**_inputs(payload, indices, torch.device("cpu")))
    return output["verifier_score_final"].cpu().numpy().astype(np.float64)


def run(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    index = pd.read_csv(root / INDEX_PATH, low_memory=False)
    index["parent_sample_id"] = index["parent_sample_id"].astype(str)
    index["sample_id"] = index["sample_id"].astype(str)
    index["causal_event_index"] = index["causal_event_index"].astype(int)
    motion, motion_provenance = _load_motion(root)
    frames: dict[str, pd.DataFrame] = {}
    payloads: dict[str, dict[str, Any]] = {}
    cache_provenance: dict[str, Any] = {}
    course_provenance: dict[str, Any] = {}
    for role in ROLES:
        cache_path, payload = _load_population(root, role)
        course_path, course = _load_course(root, role)
        _verify_course_alignment(root, role, payload, course, index)
        frame = _aggregate(payload, course, index)
        frame["r6_liquid_score"] = _liquid_scores(root, payload)
        payloads[role] = payload
        frames[role] = frame
        cache_provenance[role] = {
            "path": relative_project_path(root, cache_path),
            "sha256": sha256_file(cache_path),
            "payload_sha256": payload["cache_sha256"],
        }
        course_provenance[role] = {
            "path": relative_project_path(root, course_path),
            "sha256": sha256_file(course_path),
            "event_count": int(len(course["sample_ids"])),
            "valid_event_count": int(course["valid"].sum()),
        }

    fit = frames["representation_fit"]
    feature_results: dict[str, Any] = {}
    for name in _feature_names(fit):
        per_role: dict[str, Any] = {}
        for role, frame in frames.items():
            route = frame["candidate"].to_numpy(bool)
            target = frame["positive"].to_numpy(bool)
            unknown_route = route & frame["label"].eq("UnknownTarget").to_numpy(bool)
            per_role[role] = {
                "bird_vs_all_route_false_positive": _feature_summary(frame, name, route, target),
                "bird_vs_unknown_only": _feature_summary(frame, name, target | unknown_route, target),
            }
        feature_results[name] = {
            "per_role": per_role,
            "transfer_from_fit": {
                role: _transfer(fit, frame, name)
                for role, frame in frames.items()
                if role != "representation_fit"
            },
        }

    roles: dict[str, Any] = {}
    for role, frame in frames.items():
        route = frame["candidate"].to_numpy(bool)
        target = frame["positive"].to_numpy(bool)
        scores = frame["r6_liquid_score"].to_numpy(float)
        roles[role] = {
            "parent_count": int(len(frame)),
            "candidate_support": int(route.sum()),
            "bird_support": int((route & target).sum()),
            "false_positive_support": int((route & ~target).sum()),
            "candidate_label_counts": {
                label: int((route & frame["label"].eq(label).to_numpy(bool)).sum()) for label in LABELS
            },
            "r6_liquid": {
                "auroc": _auroc(scores[route], target[route]),
                "auprc": _auprc(scores[route], target[route]),
                "bird_score_mean": float(scores[route & target].mean()),
                "false_positive_score_mean": float(scores[route & ~target].mean()),
            },
        }

    motion_alignment = {role: _motion_coverage(payloads[role], motion, role) for role in ROLES}
    report: dict[str, Any] = {
        "schema_version": "RADAR-LN3-V27d-feature-forensics-v1",
        "experiment_id": "V27d-r7-feature-forensics",
        "direct_baseline": "V27d-r6-liquid_treatment-seed42",
        "status": "diagnostic_only_no_training",
        "analysis_scope": {
            "stable8_and_course_parent_aggregates": "final_parent_diagnostic_only",
            "model_input_ready": False,
            "reason": "The ranking probe summarizes the complete observed prefix at the final event; a training treatment must materialize the same statistics causally at every eligible prefix.",
        },
        "data_contract": {
            "index_path": relative_project_path(root, root / INDEX_PATH),
            "index_sha256": sha256_file(root / INDEX_PATH),
            "route_population": "r6 quality-eligible L1-target and L2-Bird candidates",
            "positive": "BirdTarget",
            "negative": "candidate non-BirdTarget; Unknown-only metrics are also reported",
            "unknown_is_l2_classifier_class": False,
            "l3_fitting": False,
            "test_payload_access_count": 0,
        },
        "provenance": {
            "r6_population": cache_provenance,
            "v21e_course": course_provenance,
            "v27c_motion": motion_provenance,
        },
        "roles": roles,
        "motion_alignment": motion_alignment,
        "feature_metrics": feature_results,
        "gating": {
            "course": {
                "eligible_for_follow_up": True,
                "causal_ready": True,
                "reason": "exact event-order coverage in fit/development/calibration and causal V21e source; use the existing causal extractor",
            },
            "stable8": {
                "eligible_for_follow_up": "after_causal_prefix_materialization",
                "causal_ready": False,
                "reason": "event_frame_values and event_frame_mask are present in every r6 population cache, but this report's aggregate ranking is final-parent diagnostic only; the treatment must derive prefix-safe statistics before each knownness readout",
            },
            "motion": {
                "eligible_for_follow_up": False,
                "reason": "V27c train/val sidecars do not cover the complete r6 route population or its Unknown candidates",
            },
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    output_dir = root / OUTPUT_ROOT
    output_dir.mkdir(parents=True, exist_ok=True)
    _seal_json(output_dir / "V27d-r7-feature-forensics.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run read-only V27d feature forensics")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run(args.project_root)
    print(json.dumps({"status": report["status"], "report_sha256": report["report_sha256"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
