from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.task_models import (
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    task_device,
)


SEEDS = (42, 43, 44)
DBB_LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget")
CU_LABELS = ("ClutterTarget", "UnknownTarget")
FRONTIER_SCORE_NAMES = (
    "p_cu",
    "energy",
    "nearest_dbb_cosine_distance",
    "nearest_dbb_standardized_distance",
)
STAGE_ONE_DBB_FLOOR = 0.95
STAGE_ONE_CU_FLOOR = 0.90
OUTPUT = Path("src-v3/processed/report/RADAR-LN3-V11-liquid-dbb-knownness-diagnostic.json")
PAIRED_REPORT = Path("src-v3/processed/report/RADAR-LN3-V11-rd-physical-paired-validation.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config_path(root: Path, seed: int) -> Path:
    return root / (
        "config/RADAR-LN3/"
        f"RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-treatment-seed{seed}-config.json"
    )


def _artifact_dir(root: Path, seed: int) -> Path:
    return root / (
        "artifacts/"
        f"RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-treatment-seed{seed}-formal"
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _auc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if positive.size == 0 or negative.size == 0 or not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("knownness AUC requires non-empty finite groups")
    ranks = _average_ranks(np.concatenate([positive, negative]))
    rank_sum = float(ranks[: positive.size].sum())
    return (rank_sum - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size)


def _logsumexp(values: np.ndarray) -> np.ndarray:
    maximum = values.max(axis=1, keepdims=True)
    return maximum[:, 0] + np.log(np.exp(values - maximum).sum(axis=1))


def _forward_split(
    root: Path,
    config: Any,
    model: LiquidTaskClassifier,
    *,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    dataset = build_task_dataset(root, config, split_name=split_name)
    loader = build_task_dataloader(dataset, batch_size=batch_size, shuffle=False)
    states: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            tensors = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            output = model(
                tr_features=tensors["tr_features"],
                tr_missing_mask=tensors["tr_missing_mask"],
                tr_dt_seconds=tensors["tr_dt_seconds"],
                tr_dt_valid=tensors["tr_dt_valid"],
                tr_motion_features=tensors.get("tr_motion_features"),
                tr_motion_valid_mask=tensors.get("tr_motion_valid_mask"),
                rd_physical_features=tensors.get("rd_physical_features"),
                rd_physical_valid=tensors.get("rd_physical_valid"),
                rd_frames=tensors["rd_frames"],
                rd_frame_mask=tensors["rd_frame_mask"],
                rd_quality=tensors["rd_quality"],
                tr_padding_mask=tensors["tr_padding_mask"],
            )
            batch_states = output["liquid_final_state"].detach().cpu().numpy().astype(np.float64)
            batch_logits = output["window_logits"].detach().cpu().numpy().astype(np.float64)
            if batch_states.ndim != 2 or batch_states.shape[1] != config.model.liquid_hidden_dim:
                raise ValueError("V11 knownness received an invalid liquid final state")
            states.append(batch_states)
            logits.append(batch_logits)
            for index, parent in enumerate(batch["parent_sample_ids"]):
                length = int(batch["lengths"][index].item())
                physical_valid = batch["rd_physical_valid"][index, :length]
                rows.append(
                    {
                        "parent_sample_id": str(parent),
                        "track_id": str(batch["track_ids"][index]),
                        "label_name": str(batch["label_names"][index]),
                        "gate_class": "DBB" if str(batch["label_names"][index]) in DBB_LABELS else "CU",
                        "sequence_length": length,
                        "rd_physical_valid_fraction": float(physical_valid.float().mean().item()),
                    }
                )
    state_matrix = np.concatenate(states, axis=0)
    logit_matrix = np.concatenate(logits, axis=0)
    if len(rows) != len(dataset) or state_matrix.shape[0] != len(rows) or logit_matrix.shape != (len(rows), 2):
        raise ValueError("V11 knownness split extraction is incomplete")
    return {"rows": rows, "states": state_matrix, "logits": logit_matrix}


def _fit_train_knownness(train: Mapping[str, Any]) -> dict[str, Any]:
    states = np.asarray(train["states"], dtype=np.float64)
    labels = np.asarray([row["label_name"] for row in train["rows"]], dtype=object)
    dbb_mask = np.isin(labels, DBB_LABELS)
    dbb_states = states[dbb_mask]
    scale = dbb_states.std(axis=0, ddof=1)
    scale_floor = max(float(np.median(scale)) * 1.0e-6, 1.0e-8)
    scale = np.maximum(scale, scale_floor)
    prototypes = np.stack([states[labels == label].mean(axis=0) for label in DBB_LABELS], axis=0)
    counts = {label: int((labels == label).sum()) for label in DBB_LABELS}
    return {
        "prototypes": prototypes,
        "scale": scale,
        "class_counts": counts,
        "scale_floor": scale_floor,
    }


def _knownness_scores(data: Mapping[str, Any], fitted: Mapping[str, Any]) -> dict[str, np.ndarray]:
    states = np.asarray(data["states"], dtype=np.float64)
    logits = np.asarray(data["logits"], dtype=np.float64)
    prototypes = np.asarray(fitted["prototypes"], dtype=np.float64)
    scale = np.asarray(fitted["scale"], dtype=np.float64)
    state_norm = np.linalg.norm(states, axis=1, keepdims=True)
    prototype_norm = np.linalg.norm(prototypes, axis=1, keepdims=True).T
    cosine = (states @ prototypes.T) / np.maximum(state_norm * prototype_norm, 1.0e-12)
    cosine_distance = 1.0 - cosine.max(axis=1)
    standardized = (states[:, None, :] - prototypes[None, :, :]) / scale[None, None, :]
    standardized_distance = np.sqrt(np.mean(standardized * standardized, axis=2)).min(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return {
        "p_cu": probabilities[:, 1],
        "max_softmax_uncertainty": 1.0 - probabilities.max(axis=1),
        "energy": -_logsumexp(logits),
        "nearest_dbb_cosine_distance": cosine_distance,
        "nearest_dbb_standardized_distance": standardized_distance,
    }


def _describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _binary_route_metrics(true_cu: np.ndarray, predicted_cu: np.ndarray) -> dict[str, Any]:
    true_cu = np.asarray(true_cu, dtype=bool)
    predicted_cu = np.asarray(predicted_cu, dtype=bool)
    if true_cu.ndim != 1 or predicted_cu.shape != true_cu.shape:
        raise ValueError("binary route metrics require aligned one-dimensional arrays")
    cu_support = int(true_cu.sum())
    dbb_support = int((~true_cu).sum())
    dbb_to_cu = int((~true_cu & predicted_cu).sum())
    cu_to_dbb = int((true_cu & ~predicted_cu).sum())
    dbb_true_positive = dbb_support - dbb_to_cu
    cu_true_positive = cu_support - cu_to_dbb
    dbb_precision = _safe_ratio(dbb_true_positive, dbb_true_positive + cu_to_dbb)
    dbb_recall = _safe_ratio(dbb_true_positive, dbb_support)
    cu_precision = _safe_ratio(cu_true_positive, cu_true_positive + dbb_to_cu)
    cu_recall = _safe_ratio(cu_true_positive, cu_support)

    def f1(precision: float, recall: float) -> float:
        return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    dbb_f1 = f1(dbb_precision, dbb_recall)
    cu_f1 = f1(cu_precision, cu_recall)
    return {
        "DBB": {
            "precision": dbb_precision,
            "recall": dbb_recall,
            "f1": dbb_f1,
            "support": dbb_support,
            "floor": min(dbb_precision, dbb_recall, dbb_f1),
        },
        "CU": {
            "precision": cu_precision,
            "recall": cu_recall,
            "f1": cu_f1,
            "support": cu_support,
            "floor": min(cu_precision, cu_recall, cu_f1),
        },
        "errors": {"DBB_to_CU": dbb_to_cu, "CU_to_DBB": cu_to_dbb},
        "accuracy": _safe_ratio(dbb_true_positive + cu_true_positive, true_cu.size),
        "macro_f1": (dbb_f1 + cu_f1) / 2.0,
    }


def _threshold_operating_points(scores: np.ndarray, true_cu: np.ndarray) -> list[dict[str, Any]]:
    scores = np.asarray(scores, dtype=np.float64)
    true_cu = np.asarray(true_cu, dtype=bool)
    if scores.ndim != 1 or true_cu.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("oracle frontier requires aligned finite one-dimensional scores")
    predicted_cu = np.zeros(scores.size, dtype=bool)
    points = [_binary_route_metrics(true_cu, predicted_cu)]
    for value in np.unique(scores)[::-1]:
        predicted_cu[scores == value] = True
        points.append(_binary_route_metrics(true_cu, predicted_cu))
    return points


def _compact_operating_point(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "DBB": dict(point["DBB"]),
        "CU": dict(point["CU"]),
        "errors": dict(point["errors"]),
        "accuracy": float(point["accuracy"]),
        "macro_f1": float(point["macro_f1"]),
    }


def _pareto_error_pairs(points: Sequence[Mapping[str, Any]]) -> list[dict[str, int]]:
    pairs = sorted(
        {
            (int(point["errors"]["DBB_to_CU"]), int(point["errors"]["CU_to_DBB"]))
            for point in points
        }
    )
    frontier = [
        pair
        for pair in pairs
        if not any(
            other[0] <= pair[0]
            and other[1] <= pair[1]
            and other != pair
            for other in pairs
        )
    ]
    return [
        {"DBB_to_CU": int(dbb_to_cu), "CU_to_DBB": int(cu_to_dbb)}
        for dbb_to_cu, cu_to_dbb in frontier
    ]


def _oracle_frontier(scores: np.ndarray, true_cu: np.ndarray) -> dict[str, Any]:
    points = _threshold_operating_points(scores, true_cu)
    recall_eligible = [point for point in points if point["CU"]["recall"] >= STAGE_ONE_CU_FLOOR]
    dbb_eligible = [point for point in points if point["DBB"]["floor"] >= STAGE_ONE_DBB_FLOOR]
    stage_one = [
        point
        for point in points
        if point["DBB"]["floor"] >= STAGE_ONE_DBB_FLOOR
        and point["CU"]["floor"] >= STAGE_ONE_CU_FLOOR
    ]
    minimum_false_reject = min(
        recall_eligible,
        key=lambda point: (
            point["errors"]["DBB_to_CU"],
            point["errors"]["CU_to_DBB"],
            -point["macro_f1"],
        ),
    )
    best_cu = (
        max(
            dbb_eligible,
            key=lambda point: (
                point["CU"]["floor"],
                point["macro_f1"],
                -point["errors"]["DBB_to_CU"],
                -point["errors"]["CU_to_DBB"],
            ),
        )
        if dbb_eligible
        else None
    )
    best_stage_one = (
        max(stage_one, key=lambda point: (point["macro_f1"], point["accuracy"]))
        if stage_one
        else None
    )
    return {
        "orientation": "higher_score_routes_to_CU",
        "threshold_values_emitted": False,
        "tie_policy": "all equal scores cross the diagnostic cut-point together",
        "operating_point_count": len(points),
        "pareto_error_pairs": _pareto_error_pairs(points),
        "minimum_DBB_to_CU_subject_to_CU_recall_at_least_0_90": _compact_operating_point(
            minimum_false_reject
        ),
        "DBB_floor_at_least_0_95_feasible": bool(dbb_eligible),
        "best_CU_floor_subject_to_DBB_floor_at_least_0_95": (
            _compact_operating_point(best_cu) if best_cu is not None else None
        ),
        "stage_one_feasible": bool(stage_one),
        "stage_one_feasible_operating_point_count": len(stage_one),
        "best_stage_one_operating_point": (
            _compact_operating_point(best_stage_one) if best_stage_one is not None else None
        ),
    }


def _validation_oracle_frontiers(
    validation: Mapping[str, Any], validation_scores: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    true_cu = np.asarray([row["gate_class"] == "CU" for row in validation["rows"]], dtype=bool)
    return {
        name: _oracle_frontier(np.asarray(validation_scores[name], dtype=np.float64), true_cu)
        for name in FRONTIER_SCORE_NAMES
    }


def _score_report(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    train_scores: Mapping[str, np.ndarray],
    validation_scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_labels = np.asarray([row["label_name"] for row in train["rows"]], dtype=object)
    validation_labels = np.asarray([row["label_name"] for row in validation["rows"]], dtype=object)
    validation_classes = np.asarray([row["gate_class"] for row in validation["rows"]], dtype=object)
    output: dict[str, Any] = {}
    for name in validation_scores:
        train_values = np.asarray(train_scores[name], dtype=np.float64)
        values = np.asarray(validation_scores[name], dtype=np.float64)
        comparisons = {
            "CU_vs_DBB": (validation_classes == "CU", validation_classes == "DBB"),
            "Unknown_vs_DBB": (validation_labels == "UnknownTarget", validation_classes == "DBB"),
            "Unknown_vs_Bird": (validation_labels == "UnknownTarget", validation_labels == "BirdTarget"),
            "Clutter_vs_DBB": (validation_labels == "ClutterTarget", validation_classes == "DBB"),
        }
        output[name] = {
            "orientation": "higher_score_means_more_CU_or_less_DBB_known",
            "train_DBB_reference": _describe(train_values[np.isin(train_labels, DBB_LABELS)]),
            "validation_by_label": {
                label: _describe(values[validation_labels == label]) for label in (*DBB_LABELS, *CU_LABELS)
            },
            "validation_auc": {
                comparison: {
                    "positive_class_auc": _auc(values[positive], values[negative]),
                    "directionless_auc": max(
                        _auc(values[positive], values[negative]),
                        1.0 - _auc(values[positive], values[negative]),
                    ),
                }
                for comparison, (positive, negative) in comparisons.items()
            },
        }
    return output


def _error_rows(
    validation: Mapping[str, Any],
    train: Mapping[str, Any],
    train_scores: Mapping[str, np.ndarray],
    validation_scores: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    logits = np.asarray(validation["logits"], dtype=np.float64)
    predictions = logits.argmax(axis=1)
    train_labels = np.asarray([row["label_name"] for row in train["rows"]], dtype=object)
    train_dbb = np.isin(train_labels, DBB_LABELS)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(validation["rows"]):
        true_id = 0 if row["label_name"] in DBB_LABELS else 1
        if int(predictions[index]) == true_id:
            continue
        scores: dict[str, Any] = {}
        for name, values in validation_scores.items():
            value = float(values[index])
            reference = np.asarray(train_scores[name], dtype=np.float64)[train_dbb]
            scores[name] = {
                "value": value,
                "percentile_within_train_DBB": float(
                    (np.count_nonzero(reference < value) + 0.5 * np.count_nonzero(reference == value))
                    / reference.size
                ),
            }
        output.append({**row, "true_id": true_id, "pred_id": int(predictions[index]), "scores": scores})
    return output


def _run_seed(root: Path, seed: int, *, batch_size: int) -> dict[str, Any]:
    config_path = _config_path(root, seed)
    config = load_task_config(config_path)
    device = task_device(config)
    checkpoint = _artifact_dir(root, seed) / "best_model.pt"
    model = LiquidTaskClassifier(config).to(device)
    payload = load_task_checkpoint(
        checkpoint,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    train = _forward_split(
        root,
        config,
        model,
        split_name=config.data.train_split_name,
        device=device,
        batch_size=batch_size,
    )
    validation = _forward_split(
        root,
        config,
        model,
        split_name=config.data.validation_split_name,
        device=device,
        batch_size=batch_size,
    )
    fitted = _fit_train_knownness(train)
    train_scores = _knownness_scores(train, fitted)
    validation_scores = _knownness_scores(validation, fitted)
    return {
        "seed": seed,
        "device": str(device),
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "best_epoch": int(payload["epoch"]),
        "train_parent_count": len(train["rows"]),
        "validation_parent_count": len(validation["rows"]),
        "prototype_contract": {
            "classes": list(DBB_LABELS),
            "class_counts": fitted["class_counts"],
            "state": "liquid_final_state",
            "state_dim": int(config.model.liquid_hidden_dim),
            "fit_split": "train_only",
            "scale": "train DBB per-dimension standard deviation with numerical floor only",
            "unknown_prototype": False,
            "clutter_in_DBB_prototype": False,
            "threshold_fitting": False,
        },
        "scores": _score_report(train, validation, train_scores, validation_scores),
        "validation_oracle_frontiers": _validation_oracle_frontiers(
            validation, validation_scores
        ),
        "validation_current_model_errors": _error_rows(
            validation, train, train_scores, validation_scores
        ),
    }


def run(project_root: Path, *, batch_size: int) -> dict[str, Any]:
    root = project_root.resolve()
    paired_path = root / PAIRED_REPORT
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired.get("stage_one_passed") is not False or paired.get("test_evaluation") != "未执行":
        raise ValueError("V11 knownness diagnostic requires the failed, test-sealed paired report")
    seeds = [_run_seed(root, seed, batch_size=batch_size) for seed in SEEDS]
    score_names = tuple(seeds[0]["scores"])
    median_auc: dict[str, Any] = {}
    for score in score_names:
        median_auc[score] = {}
        for comparison in seeds[0]["scores"][score]["validation_auc"]:
            values = [seed["scores"][score]["validation_auc"][comparison]["positive_class_auc"] for seed in seeds]
            median_auc[score][comparison] = {
                "seed_values": values,
                "median": float(statistics.median(values)),
                "minimum": float(min(values)),
                "maximum": float(max(values)),
            }
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V11-liquid-dbb-knownness-diagnostic-v2",
        "scope": "three Treatment best checkpoints; train-only DBB prototypes; validation diagnostics; test sealed",
        "paired_report": str(paired_path),
        "paired_report_sha256": _sha256(paired_path),
        "stage_one_passed": False,
        "hypothesis": (
            "Use class-conditional Drone/Bird/Balloon liquid-state proximity as an auxiliary DBB-knownness signal; "
            "Clutter and Unknown create no known prototype and remain potential CU/reject routes."
        ),
        "seeds": seeds,
        "median_validation_auc": median_auc,
        "interpretation_boundaries": [
            "This is a representation diagnostic, not a trained or calibrated rejection mechanism.",
            "No threshold is selected on validation and no validation sample enters prototype fitting.",
            "The Gate was trained with aggregate DBB/CU labels, not a prototype or metric-learning objective.",
            "AUC measures ranking only; it does not prove the frozen precision/recall operating point is attainable.",
            "Oracle frontiers reuse validation truth only to diagnose score attainability, emit no threshold values, and are invalid for deployment calibration or unbiased performance reporting.",
            "The same 232 validation parents are reused across seeds, so seeds are optimization replicates rather than independent data samples.",
            "Test, OOF, threshold fitting, new training, specialist/cascade evaluation and P95 remain unexecuted.",
        ],
        "new_training": "未执行",
        "threshold_fitting": "未执行",
        "test_evaluation": "未执行",
        "OOF": "未执行",
        "P95": "未执行",
    }
    output = root / OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(output)
    result["result_sha256"] = _sha256(output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose train-only DBB liquid-state knownness on V11 validation.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    print(json.dumps(run(args.project_root, batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
