"""Compare V12 Control/Treatment DBB-subclass knownness without fitting thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from radar_ln3.task_models import (
    DBB_AUXILIARY_CLASS_NAMES,
    DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION,
    DBB_AUXILIARY_PAIRED_PROTOCOL,
    DBB_AUXILIARY_TREATMENT_WEIGHT,
    LiquidTaskClassifier,
    build_task_dataloader,
    build_task_dataset,
    load_task_checkpoint,
    load_task_config,
    sha256_file,
    task_device,
    verify_task_data_contract,
)


SEEDS = (42, 43, 44)
ARMS = ("control", "treatment")
DBB_LABELS = tuple(DBB_AUXILIARY_CLASS_NAMES)
CU_LABELS = ("ClutterTarget", "UnknownTarget")
ALL_LABELS = (*DBB_LABELS, *CU_LABELS)
DISTANCE_SCORES = (
    "nearest_subclass_cosine_distance",
    "nearest_subclass_diagonal_mahalanobis",
)
SCORE_NAMES = ("p_cu", "gate_energy", *DISTANCE_SCORES)
COMPARISONS = (
    "CU_vs_DBB",
    "Unknown_vs_DBB",
    "Unknown_vs_Bird",
    "Clutter_vs_DBB",
    "Clutter_vs_Bird",
)
EXPECTED_COUNTS = {
    "train": {
        "DroneTarget": 488,
        "BirdTarget": 243,
        "BalloonTarget": 245,
        "ClutterTarget": 35,
        "UnknownTarget": 70,
    },
    "val": {
        "DroneTarget": 105,
        "BirdTarget": 52,
        "BalloonTarget": 52,
        "ClutterTarget": 8,
        "UnknownTarget": 15,
    },
}
PAIRED_REPORT = Path("src-v3/processed/report/RADAR-LN3-V12-dbb-auxiliary-paired-validation.json")
OUTPUT_REPORT = Path("src-v3/processed/report/RADAR-LN3-V12-dbb-knownness-paired-diagnostic.json")
UNKNOWN_BIRD_MINIMUM_DELTA = 0.02
UNKNOWN_DBB_MINIMUM_AUC = 0.90
CU_DBB_MAXIMUM_AUC_DECLINE = 0.01


def _config_path(seed: int, arm: str) -> Path:
    return Path(
        "config/RADAR-LN3/"
        f"RADAR-LN3-V12-cq08-liquid-dbb-cu-gate-dbb-auxiliary-{arm}-seed{seed}-config.json"
    )


def _parent_order_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["parent_sample_id"]).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


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


def auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """Return tie-aware AUC for a score whose larger value means positive."""

    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    if (
        positive.ndim != 1
        or negative.ndim != 1
        or positive.size == 0
        or negative.size == 0
        or not np.isfinite(positive).all()
        or not np.isfinite(negative).all()
    ):
        raise ValueError("AUC requires non-empty finite one-dimensional groups")
    ranks = _average_ranks(np.concatenate([positive, negative]))
    positive_rank_sum = float(ranks[: positive.size].sum())
    return (
        positive_rank_sum - positive.size * (positive.size + 1) / 2.0
    ) / (positive.size * negative.size)


def _describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("description requires a non-empty finite one-dimensional array")
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def fit_subclass_reference(states: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Fit train-only DBB subclass centers and pooled diagonal within-class variance."""

    states = np.asarray(states, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    if states.ndim != 2 or labels.shape != (states.shape[0],) or not np.isfinite(states).all():
        raise ValueError("subclass reference requires aligned finite state rows")
    centers: list[np.ndarray] = []
    residual_sum = np.zeros(states.shape[1], dtype=np.float64)
    dbb_count = 0
    counts: dict[str, int] = {}
    for label in DBB_LABELS:
        selected = states[labels == label]
        if selected.shape[0] < 2:
            raise ValueError(f"subclass reference lacks train support for {label}")
        center = selected.mean(axis=0)
        centers.append(center)
        residual_sum += np.square(selected - center).sum(axis=0)
        counts[label] = int(selected.shape[0])
        dbb_count += int(selected.shape[0])
    degrees_of_freedom = dbb_count - len(DBB_LABELS)
    if degrees_of_freedom <= 0:
        raise ValueError("subclass pooled variance has no degrees of freedom")
    pooled_variance = residual_sum / degrees_of_freedom
    positive = pooled_variance[pooled_variance > 0.0]
    variance_floor = max(
        (float(np.median(positive)) if positive.size else 1.0) * 1.0e-6,
        1.0e-10,
    )
    pooled_variance = np.maximum(pooled_variance, variance_floor)
    center_matrix = np.stack(centers, axis=0)
    if center_matrix.shape != (len(DBB_LABELS), states.shape[1]):
        raise AssertionError("subclass center matrix has an invalid shape")
    return {
        "centers": center_matrix,
        "pooled_variance": pooled_variance,
        "variance_floor": variance_floor,
        "class_counts": counts,
        "dbb_parent_count": dbb_count,
        "pooled_degrees_of_freedom": degrees_of_freedom,
        "cu_rows_excluded": int(np.isin(labels, CU_LABELS).sum()),
    }


def knownness_scores(
    states: np.ndarray,
    logits: np.ndarray,
    fitted: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    states = np.asarray(states, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    centers = np.asarray(fitted["centers"], dtype=np.float64)
    pooled_variance = np.asarray(fitted["pooled_variance"], dtype=np.float64)
    if (
        states.ndim != 2
        or logits.shape != (states.shape[0], 2)
        or centers.shape != (len(DBB_LABELS), states.shape[1])
        or pooled_variance.shape != (states.shape[1],)
        or not np.isfinite(states).all()
        or not np.isfinite(logits).all()
        or not np.isfinite(centers).all()
        or not np.isfinite(pooled_variance).all()
        or bool((pooled_variance <= 0.0).any())
    ):
        raise ValueError("knownness scoring received an invalid contract")
    state_norm = np.linalg.norm(states, axis=1, keepdims=True)
    center_norm = np.linalg.norm(centers, axis=1, keepdims=True).T
    cosine_similarity = (states @ centers.T) / np.maximum(state_norm * center_norm, 1.0e-12)
    cosine_distance = 1.0 - cosine_similarity.max(axis=1)
    squared_standardized = np.square(states[:, None, :] - centers[None, :, :]) / pooled_variance[None, None, :]
    diagonal_mahalanobis = np.sqrt(squared_standardized.sum(axis=2)).min(axis=1)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    probabilities = exponent / exponent.sum(axis=1, keepdims=True)
    maximum = logits.max(axis=1, keepdims=True)
    logsumexp = maximum[:, 0] + np.log(np.exp(logits - maximum).sum(axis=1))
    scores = {
        "p_cu": probabilities[:, 1],
        "gate_energy": -logsumexp,
        "nearest_subclass_cosine_distance": cosine_distance,
        "nearest_subclass_diagonal_mahalanobis": diagonal_mahalanobis,
    }
    if tuple(scores) != SCORE_NAMES or any(
        value.shape != (states.shape[0],) or not np.isfinite(value).all()
        for value in scores.values()
    ):
        raise AssertionError("knownness scores violate the frozen finite vector contract")
    return scores


def _forward_split(
    root: Path,
    config: Any,
    model: LiquidTaskClassifier,
    *,
    split_name: str,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    if split_name not in {config.data.train_split_name, config.data.validation_split_name}:
        raise PermissionError("V12 knownness diagnostic is restricted to train/validation")
    dataset = build_task_dataset(root, config, split_name=split_name)
    loader = build_task_dataloader(dataset, batch_size=batch_size, shuffle=False)
    rows: list[dict[str, Any]] = []
    state_blocks: list[np.ndarray] = []
    logit_blocks: list[np.ndarray] = []
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
            states = output["liquid_final_state"].detach().cpu().numpy().astype(np.float64)
            logits = output["window_logits"].detach().cpu().numpy().astype(np.float64)
            if states.shape != (len(batch["parent_sample_ids"]), config.model.liquid_hidden_dim):
                raise ValueError("V12 knownness extraction received an invalid final-state shape")
            if logits.shape != (states.shape[0], 2):
                raise ValueError("V12 knownness extraction received an invalid Gate-logit shape")
            state_blocks.append(states)
            logit_blocks.append(logits)
            for index, parent_id in enumerate(batch["parent_sample_ids"]):
                rows.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "track_id": str(batch["track_ids"][index]),
                        "label_name": str(batch["label_names"][index]),
                        "sequence_length": int(batch["lengths"][index].item()),
                    }
                )
    state_matrix = np.concatenate(state_blocks, axis=0)
    logit_matrix = np.concatenate(logit_blocks, axis=0)
    if (
        len(rows) != len(dataset)
        or state_matrix.shape != (len(rows), config.model.liquid_hidden_dim)
        or logit_matrix.shape != (len(rows), 2)
        or not np.isfinite(state_matrix).all()
        or not np.isfinite(logit_matrix).all()
    ):
        raise ValueError("V12 knownness split extraction is incomplete or non-finite")
    counts = {label: sum(row["label_name"] == label for row in rows) for label in ALL_LABELS}
    expected = EXPECTED_COUNTS[split_name]
    if counts != expected:
        raise ValueError(f"V12 knownness {split_name} label counts drifted: {counts}")
    return {
        "rows": rows,
        "states": state_matrix,
        "logits": logit_matrix,
        "parent_order_sha256": _parent_order_sha256(rows),
        "label_parent_counts": counts,
    }


def _comparison_masks(labels: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    dbb = np.isin(labels, DBB_LABELS)
    cu = np.isin(labels, CU_LABELS)
    bird = labels == "BirdTarget"
    unknown = labels == "UnknownTarget"
    clutter = labels == "ClutterTarget"
    return {
        "CU_vs_DBB": (cu, dbb),
        "Unknown_vs_DBB": (unknown, dbb),
        "Unknown_vs_Bird": (unknown, bird),
        "Clutter_vs_DBB": (clutter, dbb),
        "Clutter_vs_Bird": (clutter, bird),
    }


def _score_report(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    train_scores: Mapping[str, np.ndarray],
    validation_scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_labels = np.asarray([row["label_name"] for row in train["rows"]], dtype=object)
    validation_labels = np.asarray([row["label_name"] for row in validation["rows"]], dtype=object)
    train_dbb = np.isin(train_labels, DBB_LABELS)
    comparisons = _comparison_masks(validation_labels)
    report: dict[str, Any] = {}
    for score_name in SCORE_NAMES:
        train_values = np.asarray(train_scores[score_name], dtype=np.float64)
        values = np.asarray(validation_scores[score_name], dtype=np.float64)
        report[score_name] = {
            "orientation": "higher_score_means_more_CU_or_less_DBB_known",
            "train_DBB_reference": _describe(train_values[train_dbb]),
            "validation_by_label": {
                label: _describe(values[validation_labels == label]) for label in ALL_LABELS
            },
            "validation_auc": {
                name: {
                    "positive_class_auc": auc(values[positive], values[negative]),
                    "positive_class": name.split("_vs_")[0],
                    "negative_class": name.split("_vs_")[1],
                }
                for name, (positive, negative) in comparisons.items()
            },
        }
    return report


def _percentile(value: float, reference: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float64)
    if reference.ndim != 1 or reference.size == 0 or not np.isfinite(reference).all():
        raise ValueError("percentile reference must be non-empty and finite")
    return float(
        (np.count_nonzero(reference < value) + 0.5 * np.count_nonzero(reference == value))
        / reference.size
    )


def _semantic_gate_errors(
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    train_scores: Mapping[str, np.ndarray],
    validation_scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    train_labels = np.asarray([row["label_name"] for row in train["rows"]], dtype=object)
    train_dbb = np.isin(train_labels, DBB_LABELS)
    predictions = np.asarray(validation["logits"], dtype=np.float64).argmax(axis=1)
    directions = {
        "BirdTarget_to_CU": [],
        "UnknownTarget_to_DBB": [],
        "ClutterTarget_to_DBB": [],
        "DroneTarget_to_CU": [],
        "BalloonTarget_to_CU": [],
    }
    for index, row in enumerate(validation["rows"]):
        label = str(row["label_name"])
        prediction = int(predictions[index])
        key: str | None = None
        if label == "BirdTarget" and prediction == 1:
            key = "BirdTarget_to_CU"
        elif label == "UnknownTarget" and prediction == 0:
            key = "UnknownTarget_to_DBB"
        elif label == "ClutterTarget" and prediction == 0:
            key = "ClutterTarget_to_DBB"
        elif label == "DroneTarget" and prediction == 1:
            key = "DroneTarget_to_CU"
        elif label == "BalloonTarget" and prediction == 1:
            key = "BalloonTarget_to_CU"
        if key is None:
            continue
        score_rows: dict[str, Any] = {}
        for score_name in SCORE_NAMES:
            value = float(validation_scores[score_name][index])
            reference = np.asarray(train_scores[score_name], dtype=np.float64)[train_dbb]
            score_rows[score_name] = {
                "value": value,
                "percentile_within_train_DBB": _percentile(value, reference),
            }
        directions[key].append(
            {
                "parent_sample_id": str(row["parent_sample_id"]),
                "track_id": str(row["track_id"]),
                "sequence_length": int(row["sequence_length"]),
                "scores": score_rows,
            }
        )
    return {
        key: {
            "count": len(rows),
            "parent_sample_ids": sorted(row["parent_sample_id"] for row in rows),
            "rows": sorted(rows, key=lambda row: row["parent_sample_id"]),
        }
        for key, rows in directions.items()
    }


def _expected_semantic_errors(paired: Mapping[str, Any], seed: int, arm: str) -> dict[str, Any]:
    return dict(
        paired["seed_results"][str(seed)]["runs"][arm]["independent_best_checkpoint_validation"][
            "semantic_errors"
        ]
    )


def _run_arm(
    root: Path,
    paired: Mapping[str, Any],
    *,
    seed: int,
    arm: str,
    batch_size: int,
) -> dict[str, Any]:
    config_path = root / _config_path(seed, arm)
    config = load_task_config(config_path)
    expected_weight = 0.0 if arm == "control" else DBB_AUXILIARY_TREATMENT_WEIGHT
    if (
        config.schema_version != DBB_AUXILIARY_PAIRED_GATE_SCHEMA_VERSION
        or config.train.paired_protocol != DBB_AUXILIARY_PAIRED_PROTOCOL
        or config.train.paired_arm != arm
        or config.train.seed != seed
        or config.train.paired_seed != seed
        or config.train.dbb_auxiliary_loss_weight != expected_weight
    ):
        raise ValueError(f"V12 knownness config violates the frozen paired contract: {config_path}")
    contract = verify_task_data_contract(root, config)
    paired_run = paired["seed_results"][str(seed)]["runs"][arm]
    checkpoint = root / config.train.checkpoint_path
    summary_path = root / config.train.formal_artifact_dir / "training-summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if (
        summary.get("status") != "passed"
        or summary.get("stop_reason") != "epoch_limit"
        or summary.get("test_evaluation") != "not_executed"
        or Path(str(summary.get("best_checkpoint", ""))).resolve() != checkpoint.resolve()
        or sha256_file(checkpoint) != str(paired_run["best_checkpoint_sha256"])
        or sha256_file(summary_path) != str(paired_run["summary_sha256"])
    ):
        raise ValueError(f"V12 knownness artifact differs from the frozen paired report: {summary_path}")
    device = task_device(config)
    model = LiquidTaskClassifier(config).to(device)
    payload = load_task_checkpoint(
        checkpoint,
        model,
        expected_schema=config.train.checkpoint_schema,
        map_location=device,
    )
    if int(payload.get("epoch", -1)) != int(paired_run["best_epoch"]):
        raise ValueError("V12 knownness checkpoint epoch differs from the frozen paired report")
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
    train_labels = np.asarray([row["label_name"] for row in train["rows"]], dtype=object)
    fitted = fit_subclass_reference(np.asarray(train["states"]), train_labels)
    if fitted["class_counts"] != {label: EXPECTED_COUNTS["train"][label] for label in DBB_LABELS}:
        raise ValueError("V12 knownness train DBB subclass counts drifted")
    if int(fitted["cu_rows_excluded"]) != sum(EXPECTED_COUNTS["train"][label] for label in CU_LABELS):
        raise ValueError("V12 knownness did not exclude every CU train parent from fitting")
    train_scores = knownness_scores(np.asarray(train["states"]), np.asarray(train["logits"]), fitted)
    validation_scores = knownness_scores(
        np.asarray(validation["states"]),
        np.asarray(validation["logits"]),
        fitted,
    )
    semantic_errors = _semantic_gate_errors(train, validation, train_scores, validation_scores)
    expected_errors = _expected_semantic_errors(paired, seed, arm)
    for direction, expected in expected_errors.items():
        if (
            int(semantic_errors[direction]["count"]) != int(expected["count"])
            or semantic_errors[direction]["parent_sample_ids"] != list(expected["parent_sample_ids"])
        ):
            raise ValueError(f"V12 knownness Gate residuals differ from paired validation: {seed}/{arm}/{direction}")
    return {
        "seed": seed,
        "arm": arm,
        "device": str(device),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "best_epoch": int(payload["epoch"]),
        "data_contract_hashes": dict(contract["hashes"]),
        "train_parent_count": len(train["rows"]),
        "validation_parent_count": len(validation["rows"]),
        "train_parent_order_sha256": train["parent_order_sha256"],
        "validation_parent_order_sha256": validation["parent_order_sha256"],
        "train_label_parent_counts": train["label_parent_counts"],
        "validation_label_parent_counts": validation["label_parent_counts"],
        "reference_contract": {
            "fit_split": "train_only",
            "classes": list(DBB_LABELS),
            "class_counts": fitted["class_counts"],
            "state": "liquid_final_state_at_parent_final_causal_event",
            "state_dim": int(config.model.liquid_hidden_dim),
            "pooled_diagonal_variance": "within_DBB_subclass_train_residuals_only",
            "pooled_degrees_of_freedom": int(fitted["pooled_degrees_of_freedom"]),
            "variance_floor": float(fitted["variance_floor"]),
            "excluded_CU_train_parent_count": int(fitted["cu_rows_excluded"]),
            "unknown_prototype": False,
            "clutter_prototype": False,
            "validation_in_fit": False,
            "threshold_fitting": False,
        },
        "scores": _score_report(train, validation, train_scores, validation_scores),
        "validation_gate_errors": semantic_errors,
    }


def _auc_value(run: Mapping[str, Any], score: str, comparison: str) -> float:
    return float(run["scores"][score]["validation_auc"][comparison]["positive_class_auc"])


def build_paired_comparison(runs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    median_auc: dict[str, Any] = {arm: {} for arm in ARMS}
    paired_delta: dict[str, Any] = {}
    for arm in ARMS:
        for score in SCORE_NAMES:
            median_auc[arm][score] = {}
            for comparison in COMPARISONS:
                values = [_auc_value(runs[seed][arm], score, comparison) for seed in SEEDS]
                median_auc[arm][score][comparison] = {
                    "seed_values": values,
                    "median": float(statistics.median(values)),
                    "minimum": float(min(values)),
                    "maximum": float(max(values)),
                }
    for score in SCORE_NAMES:
        paired_delta[score] = {}
        for comparison in COMPARISONS:
            values = [
                _auc_value(runs[seed]["treatment"], score, comparison)
                - _auc_value(runs[seed]["control"], score, comparison)
                for seed in SEEDS
            ]
            paired_delta[score][comparison] = {
                "treatment_minus_control_seed_values": values,
                "median": float(statistics.median(values)),
            }
    return {"median_validation_auc": median_auc, "paired_auc_delta": paired_delta}


def evaluate_v13_design_evidence(
    runs: Mapping[int, Mapping[str, Any]],
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    candidates: dict[str, Any] = {}
    eligible: list[str] = []
    medians = comparison["median_validation_auc"]
    for score in DISTANCE_SCORES:
        seed_checks: dict[str, Any] = {}
        passing_seed_count = 0
        for seed in SEEDS:
            treatment_unknown_bird = _auc_value(runs[seed]["treatment"], score, "Unknown_vs_Bird")
            control_unknown_bird = _auc_value(runs[seed]["control"], score, "Unknown_vs_Bird")
            treatment_p_cu = _auc_value(runs[seed]["treatment"], "p_cu", "Unknown_vs_Bird")
            checks = {
                "gain_over_same_score_control_at_least_0_02": (
                    treatment_unknown_bird - control_unknown_bird >= UNKNOWN_BIRD_MINIMUM_DELTA
                ),
                "gain_over_treatment_p_cu_at_least_0_02": (
                    treatment_unknown_bird - treatment_p_cu >= UNKNOWN_BIRD_MINIMUM_DELTA
                ),
            }
            passed = all(checks.values())
            passing_seed_count += int(passed)
            seed_checks[str(seed)] = {
                "passed": passed,
                "checks": checks,
                "treatment_unknown_vs_bird_auc": treatment_unknown_bird,
                "control_unknown_vs_bird_auc": control_unknown_bird,
                "treatment_p_cu_unknown_vs_bird_auc": treatment_p_cu,
                "gain_over_control": treatment_unknown_bird - control_unknown_bird,
                "gain_over_treatment_p_cu": treatment_unknown_bird - treatment_p_cu,
            }
        treatment_unknown_bird_median = float(medians["treatment"][score]["Unknown_vs_Bird"]["median"])
        control_unknown_bird_median = float(medians["control"][score]["Unknown_vs_Bird"]["median"])
        treatment_p_cu_unknown_bird_median = float(
            medians["treatment"]["p_cu"]["Unknown_vs_Bird"]["median"]
        )
        treatment_unknown_dbb_median = float(medians["treatment"][score]["Unknown_vs_DBB"]["median"])
        treatment_cu_dbb_median = float(medians["treatment"][score]["CU_vs_DBB"]["median"])
        treatment_p_cu_cu_dbb_median = float(medians["treatment"]["p_cu"]["CU_vs_DBB"]["median"])
        median_checks = {
            "unknown_vs_bird_gain_over_control_at_least_0_02": (
                treatment_unknown_bird_median - control_unknown_bird_median
                >= UNKNOWN_BIRD_MINIMUM_DELTA
            ),
            "unknown_vs_bird_gain_over_treatment_p_cu_at_least_0_02": (
                treatment_unknown_bird_median - treatment_p_cu_unknown_bird_median
                >= UNKNOWN_BIRD_MINIMUM_DELTA
            ),
            "treatment_unknown_vs_DBB_auc_at_least_0_90": (
                treatment_unknown_dbb_median >= UNKNOWN_DBB_MINIMUM_AUC
            ),
            "treatment_CU_vs_DBB_auc_decline_vs_p_cu_at_most_0_01": (
                treatment_cu_dbb_median
                >= treatment_p_cu_cu_dbb_median - CU_DBB_MAXIMUM_AUC_DECLINE
            ),
        }
        passed = passing_seed_count >= 2 and all(median_checks.values())
        if passed:
            eligible.append(score)
        candidates[score] = {
            "passed": passed,
            "passing_seed_count": passing_seed_count,
            "required_passing_seed_count": 2,
            "seed_checks": seed_checks,
            "median_checks": median_checks,
            "median_values": {
                "treatment_unknown_vs_bird_auc": treatment_unknown_bird_median,
                "control_unknown_vs_bird_auc": control_unknown_bird_median,
                "treatment_p_cu_unknown_vs_bird_auc": treatment_p_cu_unknown_bird_median,
                "unknown_vs_bird_gain_over_control": (
                    treatment_unknown_bird_median - control_unknown_bird_median
                ),
                "unknown_vs_bird_gain_over_treatment_p_cu": (
                    treatment_unknown_bird_median - treatment_p_cu_unknown_bird_median
                ),
                "treatment_unknown_vs_DBB_auc": treatment_unknown_dbb_median,
                "treatment_CU_vs_DBB_auc": treatment_cu_dbb_median,
                "treatment_p_cu_CU_vs_DBB_auc": treatment_p_cu_cu_dbb_median,
                "CU_vs_DBB_auc_delta_vs_treatment_p_cu": (
                    treatment_cu_dbb_median - treatment_p_cu_cu_dbb_median
                ),
            },
        }
    return {
        "passed": bool(eligible),
        "eligible_candidates": eligible,
        "candidate_results": candidates,
        "meaning": "design-discussion evidence only; not a threshold, classifier, or deployment gate",
        "threshold_values_emitted": False,
    }


def run(project_root: Path, *, batch_size: int) -> dict[str, Any]:
    root = project_root.resolve()
    paired_path = root / PAIRED_REPORT
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if (
        paired.get("status") != "completed_but_gate_failed"
        or paired.get("v12_retention_passed") is not False
        or paired.get("test_evaluation") != "未执行"
    ):
        raise ValueError("V12 knownness requires the failed, test-sealed frozen paired report")
    runs: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        runs[seed] = {
            arm: _run_arm(root, paired, seed=seed, arm=arm, batch_size=batch_size)
            for arm in ARMS
        }
        if (
            runs[seed]["control"]["train_parent_order_sha256"]
            != runs[seed]["treatment"]["train_parent_order_sha256"]
            or runs[seed]["control"]["validation_parent_order_sha256"]
            != runs[seed]["treatment"]["validation_parent_order_sha256"]
        ):
            raise ValueError(f"V12 knownness paired parent order differs for seed {seed}")
    train_hashes = {runs[seed][arm]["train_parent_order_sha256"] for seed in SEEDS for arm in ARMS}
    validation_hashes = {
        runs[seed][arm]["validation_parent_order_sha256"] for seed in SEEDS for arm in ARMS
    }
    if len(train_hashes) != 1 or len(validation_hashes) != 1:
        raise ValueError("V12 knownness parent order differs across seeds")
    comparison = build_paired_comparison(runs)
    evidence = evaluate_v13_design_evidence(runs, comparison)
    report = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V12-dbb-knownness-paired-diagnostic-v1",
        "scope": "six V12 best checkpoints; train-only DBB subclass references; CQ08 V3 validation scoring; test sealed",
        "paired_report": str(paired_path),
        "paired_report_sha256": sha256_file(paired_path),
        "hypothesis": (
            "V12 subclass supervision may add train-only class-conditional DBB distance information beyond "
            "the aggregate Gate P(CU), while Clutter and Unknown create no known prototype."
        ),
        "score_contract": {
            "score_names": list(SCORE_NAMES),
            "orientation": "higher_score_means_more_CU_or_less_DBB_known",
            "p_cu": "softmax Gate probability for CU",
            "gate_energy": "negative logsumexp of the two Gate logits",
            "nearest_subclass_cosine_distance": "one minus maximum cosine similarity to a train-only DBB subclass center",
            "nearest_subclass_diagonal_mahalanobis": (
                "minimum square-root diagonal Mahalanobis distance to a train-only DBB subclass center; "
                "variance pooled from within-subclass train residuals"
            ),
        },
        "runs": {str(seed): runs[seed] for seed in SEEDS},
        "paired_comparison": comparison,
        "v13_design_evidence_gate": evidence,
        "interpretation_boundaries": [
            "This is a post-training representation diagnostic, not a trained or calibrated rejection mechanism.",
            "No threshold or oracle operating point is computed or emitted.",
            "Validation labels are used only to calculate diagnostic AUC and never enter centers or variance.",
            "AUC measures ranking and does not prove the frozen precision/recall operating point is attainable.",
            "The three seeds reuse the same 232 validation parents and are optimization replicates, not independent samples.",
            "The evidence gate was frozen before running this diagnostic and only authorizes or rejects later design discussion.",
            "Test, OOF, threshold fitting, new training, cascade/stable evaluation and P95 remain unexecuted.",
        ],
        "new_training": "未执行",
        "threshold_fitting": "未执行",
        "threshold_values_emitted": False,
        "test_evaluation": "未执行",
        "OOF": "未执行",
        "stable_evaluation": "未执行",
        "P95": "未执行",
    }
    output = root / OUTPUT_REPORT
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(output)
    report["report_sha256"] = sha256_file(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose paired V12 DBB-subclass knownness on train/validation.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    print(json.dumps(run(args.project_root, batch_size=args.batch_size), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
