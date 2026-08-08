from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch

from radar_ln3.rd_physical import RD_PHYSICAL_NAMES


SEEDS = (42, 43, 44)
ARMS = ("control", "treatment")
LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget")
DBB_LABELS = frozenset({"DroneTarget", "BirdTarget", "BalloonTarget"})
CU_LABELS = frozenset({"ClutterTarget", "UnknownTarget"})
SOURCE_INDEX = Path("data/processed/RADAR-LN3-V3-cq08-tr-rd-prefix-dt-index.csv")
TRAIN_CACHE = Path("data/processed/RADAR-LN3-V11-cq08-train-rd-physical-event-cache.npz")
VAL_CACHE = Path("data/processed/RADAR-LN3-V11-cq08-val-rd-physical-event-cache.npz")
PAIRED_REPORT = Path("src-v3/processed/report/RADAR-LN3-V11-rd-physical-paired-validation.json")
OUTPUT_JSON = Path("src-v3/processed/report/RADAR-LN3-V11-rd-physical-residual-analysis.json")
OUTPUT_CSV = Path("src-v3/processed/report/RADAR-LN3-V11-rd-physical-parent-transitions.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_dir(root: Path, arm: str, seed: int) -> Path:
    return root / (
        "artifacts/"
        f"RADAR-LN3-V11-cq08-liquid-dbb-cu-gate-rd-physical-{arm}-seed{seed}-formal"
    )


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=np.float64)
    start = 0
    while start < order.size:
        end = start + 1
        while end < order.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def _rank_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size == 0 or negative.size == 0:
        return None
    values = np.concatenate([positive, negative])
    ranks = _average_ranks(values)
    rank_sum = float(ranks[: positive.size].sum())
    return (rank_sum - positive.size * (positive.size + 1) / 2.0) / (positive.size * negative.size)


def _cohens_d(positive: np.ndarray, negative: np.ndarray) -> float | None:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    positive = positive[np.isfinite(positive)]
    negative = negative[np.isfinite(negative)]
    if positive.size < 2 or negative.size < 2:
        return None
    denominator = positive.size + negative.size - 2
    pooled_variance = (
        (positive.size - 1) * positive.var(ddof=1) + (negative.size - 1) * negative.var(ddof=1)
    ) / denominator
    if pooled_variance <= 0.0:
        return 0.0 if float(positive.mean()) == float(negative.mean()) else None
    return float((positive.mean() - negative.mean()) / math.sqrt(pooled_variance))


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "std": None, "p25": None, "median": None, "p75": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "p75": float(np.quantile(values, 0.75)),
    }


def _read_index(root: Path) -> dict[str, list[dict[str, str]]]:
    selected: dict[str, list[dict[str, str]]] = {"train": [], "val": []}
    required = {"sample_id", "parent_sample_id", "label_name", "split_name", "causal_event_index"}
    with (root / SOURCE_INDEX).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("V3 index lacks fields required for V11 residual analysis")
        for row in reader:
            split = str(row["split_name"])
            if split not in selected:
                continue
            label = str(row["label_name"])
            if label not in LABELS:
                raise ValueError(f"unexpected V11 label: {label}")
            selected[split].append({name: str(row[name]) for name in required})
    return selected


def _parent_physical_rows(root: Path, split: str, rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    cache_path = root / (TRAIN_CACHE if split == "train" else VAL_CACHE)
    with np.load(cache_path, allow_pickle=False) as payload:
        sample_ids = np.asarray(payload["sample_ids"]).astype(str)
        values = np.asarray(payload["values"], dtype=np.float64)
        valid = np.asarray(payload["valid"], dtype=np.bool_)
    expected_ids = [str(row["sample_id"]) for row in rows]
    if sample_ids.tolist() != expected_ids or values.shape != (len(rows), 4) or valid.shape != (len(rows),):
        raise ValueError(f"V11 {split} cache/index alignment failed during residual analysis")
    grouped: dict[str, list[int]] = defaultdict(list)
    metadata: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows):
        parent = str(row["parent_sample_id"])
        grouped[parent].append(index)
        candidate = (str(row["label_name"]), split)
        if parent in metadata and metadata[parent] != candidate:
            raise ValueError(f"V11 parent metadata is inconsistent: {parent}")
        metadata[parent] = candidate
    output: list[dict[str, Any]] = []
    for parent, indices in grouped.items():
        indices = sorted(indices, key=lambda index: int(rows[index]["causal_event_index"]))
        parent_values = values[np.asarray(indices, dtype=np.int64)]
        parent_valid = valid[np.asarray(indices, dtype=np.int64)]
        valid_values = parent_values[parent_valid]
        label, _ = metadata[parent]
        output.append(
            {
                "parent_sample_id": parent,
                "label_name": label,
                "gate_class": "DBB" if label in DBB_LABELS else "CU",
                "event_count": len(indices),
                "valid_event_count": int(parent_valid.sum()),
                "valid_fraction": float(parent_valid.mean()),
                "sequence_mean_including_missing_zero": parent_values.mean(axis=0),
                "valid_event_mean": (
                    valid_values.mean(axis=0) if valid_values.size else np.full(4, np.nan, dtype=np.float64)
                ),
                "final_event": parent_values[-1],
                "final_event_valid": bool(parent_valid[-1]),
            }
        )
    return sorted(output, key=lambda row: row["parent_sample_id"])


def _feature_analysis(parents: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "parent_count": len(parents),
        "label_parent_counts": {label: sum(row["label_name"] == label for row in parents) for label in LABELS},
        "rd_coverage": {},
        "aggregations": {},
    }
    for label in LABELS:
        selected = [row for row in parents if row["label_name"] == label]
        result["rd_coverage"][label] = {
            "parent_count": len(selected),
            "event_count": sum(int(row["event_count"]) for row in selected),
            "valid_event_count": sum(int(row["valid_event_count"]) for row in selected),
            "mean_parent_valid_fraction": (
                float(np.mean([row["valid_fraction"] for row in selected])) if selected else None
            ),
            "all_events_valid_parent_count": sum(float(row["valid_fraction"]) == 1.0 for row in selected),
        }
    for aggregation in ("sequence_mean_including_missing_zero", "valid_event_mean", "final_event"):
        matrix = np.asarray([row[aggregation] for row in parents], dtype=np.float64)
        labels = np.asarray([row["label_name"] for row in parents], dtype=object)
        classes = np.asarray([row["gate_class"] for row in parents], dtype=object)
        aggregation_result: dict[str, Any] = {"by_gate_class": {}, "by_semantic_label": {}, "separation": {}}
        for group_name, mask in (("DBB", classes == "DBB"), ("CU", classes == "CU")):
            aggregation_result["by_gate_class"][group_name] = {
                name: _summary(matrix[mask, index]) for index, name in enumerate(RD_PHYSICAL_NAMES)
            }
        for label in LABELS:
            mask = labels == label
            aggregation_result["by_semantic_label"][label] = {
                name: _summary(matrix[mask, index]) for index, name in enumerate(RD_PHYSICAL_NAMES)
            }
        comparisons = {
            "CU_vs_DBB": (classes == "CU", classes == "DBB"),
            "Unknown_vs_Bird": (labels == "UnknownTarget", labels == "BirdTarget"),
            "Clutter_vs_Bird": (labels == "ClutterTarget", labels == "BirdTarget"),
        }
        for comparison, (positive_mask, negative_mask) in comparisons.items():
            feature_rows: dict[str, Any] = {}
            for index, name in enumerate(RD_PHYSICAL_NAMES):
                auc = _rank_auc(matrix[positive_mask, index], matrix[negative_mask, index])
                feature_rows[name] = {
                    "positive_class_auc": auc,
                    "directionless_auc": None if auc is None else max(auc, 1.0 - auc),
                    "cohens_d_positive_minus_negative": _cohens_d(
                        matrix[positive_mask, index], matrix[negative_mask, index]
                    ),
                }
            aggregation_result["separation"][comparison] = feature_rows
        result["aggregations"][aggregation] = aggregation_result
    return result


def _load_parent_records(root: Path, arm: str, seed: int) -> list[dict[str, Any]]:
    path = _artifact_dir(root, arm, seed) / "sample_detail_information.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = [dict(row) for row in payload.get("records", {}).get("parent", [])]
    if len(records) != 232 or len({row["parent_sample_id"] for row in records}) != 232:
        raise ValueError(f"V11 parent sample details are incomplete: {path}")
    return records


def _binary_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in records:
        matrix[int(row["true_id"]), int(row["pred_id"])] += 1
    output: dict[str, Any] = {"confusion_matrix_rows_true_columns_pred": matrix.tolist()}
    for class_id, name in enumerate(("DBB", "CU")):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        output[name] = {
            "precision": precision,
            "recall": recall,
            "f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "support": int(matrix[class_id, :].sum()),
        }
    output["accuracy"] = float(np.trace(matrix) / matrix.sum())
    output["macro_f1"] = float(np.mean([output["DBB"]["f1"], output["CU"]["f1"]]))
    output["errors"] = {"DBB_to_CU": int(matrix[0, 1]), "CU_to_DBB": int(matrix[1, 0])}
    return output


def _prediction_analysis(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records: dict[tuple[int, str], list[dict[str, Any]]] = {
        (seed, arm): _load_parent_records(root, arm, seed) for seed in SEEDS for arm in ARMS
    }
    transitions: list[dict[str, Any]] = []
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        control = {row["parent_sample_id"]: row for row in records[(seed, "control")]}
        treatment = {row["parent_sample_id"]: row for row in records[(seed, "treatment")]}
        if set(control) != set(treatment):
            raise ValueError(f"V11 paired parent set differs for seed {seed}")
        transition_counts: dict[str, int] = defaultdict(int)
        semantic_route: dict[str, Any] = {}
        for label in LABELS:
            semantic_route[label] = {}
            for arm, source in (("control", control), ("treatment", treatment)):
                selected = [row for row in source.values() if row["label_name"] == label]
                correct = sum(int(row["true_id"]) == int(row["pred_id"]) for row in selected)
                semantic_route[label][arm] = {
                    "support": len(selected),
                    "correct_route_count": correct,
                    "route_recall": correct / len(selected) if selected else None,
                    "mean_confidence": float(np.mean([row["confidence"] for row in selected])) if selected else None,
                }
        for parent in sorted(control):
            left = control[parent]
            right = treatment[parent]
            control_correct = int(left["true_id"]) == int(left["pred_id"])
            treatment_correct = int(right["true_id"]) == int(right["pred_id"])
            if control_correct and treatment_correct:
                change = "both_correct"
            elif not control_correct and treatment_correct:
                change = "fixed_by_treatment"
            elif control_correct and not treatment_correct:
                change = "regressed_by_treatment"
            else:
                change = "both_wrong"
            transition_counts[change] += 1
            transitions.append(
                {
                    "seed": seed,
                    "parent_sample_id": parent,
                    "track_id": str(left["track_id"]),
                    "label_name": str(left["label_name"]),
                    "true_id": int(left["true_id"]),
                    "control_pred_id": int(left["pred_id"]),
                    "treatment_pred_id": int(right["pred_id"]),
                    "control_confidence": float(left["confidence"]),
                    "treatment_confidence": float(right["confidence"]),
                    "window_length": int(left["window_length"]),
                    "change": change,
                }
            )
        per_seed[str(seed)] = {
            "transition_counts": dict(sorted(transition_counts.items())),
            "semantic_route_metrics": semantic_route,
        }
    consensus_errors: dict[str, list[dict[str, Any]]] = {}
    majority_metrics: dict[str, Any] = {}
    for arm in ARMS:
        by_seed = {
            seed: {row["parent_sample_id"]: row for row in records[(seed, arm)]} for seed in SEEDS
        }
        parents = sorted(by_seed[SEEDS[0]])
        consensus: list[dict[str, Any]] = []
        majority_records: list[dict[str, Any]] = []
        for parent in parents:
            rows = [by_seed[seed][parent] for seed in SEEDS]
            wrong = [int(row["pred_id"]) != int(row["true_id"]) for row in rows]
            if all(wrong):
                consensus.append(
                    {
                        "parent_sample_id": parent,
                        "track_id": str(rows[0]["track_id"]),
                        "label_name": str(rows[0]["label_name"]),
                        "window_length": int(rows[0]["window_length"]),
                        "mean_wrong_confidence": float(np.mean([row["confidence"] for row in rows])),
                    }
                )
            votes = [int(row["pred_id"]) for row in rows]
            majority_records.append(
                {
                    "true_id": int(rows[0]["true_id"]),
                    "pred_id": 1 if sum(votes) >= 2 else 0,
                }
            )
        consensus_errors[arm] = consensus
        majority_metrics[arm] = _binary_metrics(majority_records)
    return {
        "per_seed": per_seed,
        "consensus_errors_all_three_seeds": consensus_errors,
        "three_seed_majority_vote_descriptive_only": majority_metrics,
    }, transitions


def _weight_analysis(root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    feature_norms: dict[str, list[float]] = {name: [] for name in RD_PHYSICAL_NAMES}
    for seed in SEEDS:
        path = _artifact_dir(root, "treatment", seed) / "best_model.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        weight = torch.as_tensor(payload["model_state"]["fusion.0.weight"], dtype=torch.float64)
        if tuple(weight.shape) != (48, 131):
            raise ValueError(f"unexpected V11 Treatment fusion shape: {path}: {tuple(weight.shape)}")
        physical = weight[:, -4:]
        base = weight[:, :-4]
        norms = torch.linalg.vector_norm(physical, dim=0)
        for name, value in zip(RD_PHYSICAL_NAMES, norms.tolist(), strict=True):
            feature_norms[name].append(float(value))
        rows[str(seed)] = {
            "best_epoch": int(payload["epoch"]),
            "best_checkpoint": str(path),
            "best_checkpoint_sha256": _sha256(path),
            "physical_feature_column_l2": {
                name: float(value) for name, value in zip(RD_PHYSICAL_NAMES, norms.tolist(), strict=True)
            },
            "physical_feature_column_mean_abs": {
                name: float(value)
                for name, value in zip(RD_PHYSICAL_NAMES, physical.abs().mean(dim=0).tolist(), strict=True)
            },
            "physical_block_frobenius": float(torch.linalg.vector_norm(physical)),
            "base_column_l2_median": float(torch.median(torch.linalg.vector_norm(base, dim=0))),
        }
    across_seed: dict[str, Any] = {}
    for name, values in feature_norms.items():
        mean = statistics.mean(values)
        across_seed[name] = {
            "values": values,
            "mean": mean,
            "population_std": statistics.pstdev(values),
            "coefficient_of_variation": statistics.pstdev(values) / mean if mean else None,
        }
    return {"per_seed": rows, "feature_norm_stability": across_seed}


def _epoch_trajectory_analysis(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for seed in SEEDS:
        output[str(seed)] = {}
        for arm in ARMS:
            path = _artifact_dir(root, arm, seed) / "training-summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            trajectory: list[dict[str, Any]] = []
            for record in summary.get("history", []):
                metrics = dict(record["validation"]["parent_metrics"])
                classes = {row["class"]: row for row in metrics["per_class"]}
                matrix = np.asarray(metrics["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
                dbb_floor = min(float(classes["DBB"][name]) for name in ("precision", "recall", "f1"))
                cu_floor = min(float(classes["CU"][name]) for name in ("precision", "recall", "f1"))
                trajectory.append(
                    {
                        "epoch": int(record["epoch"]),
                        "DBB_floor": dbb_floor,
                        "CU_floor": cu_floor,
                        "DBB_to_CU": int(matrix[0, 1]),
                        "CU_to_DBB": int(matrix[1, 0]),
                        "macro_f1": float(metrics["macro_f1"]),
                        "stage_one_class_floors_pass": dbb_floor >= 0.95 and cu_floor >= 0.90,
                    }
                )
            if len(trajectory) != 20:
                raise ValueError(f"V11 epoch trajectory is incomplete: {path}")
            eligible = [row for row in trajectory if row["DBB_floor"] >= 0.95]
            best_cu = max(eligible, key=lambda row: (row["CU_floor"], row["macro_f1"])) if eligible else None
            output[str(seed)][arm] = {
                "summary_path": str(path),
                "summary_sha256": _sha256(path),
                "epoch_count": len(trajectory),
                "epochs_passing_both_class_floors": [
                    row["epoch"] for row in trajectory if row["stage_one_class_floors_pass"]
                ],
                "best_CU_floor_subject_to_DBB_floor_at_least_0_95": best_cu,
                "trajectory": trajectory,
            }
    return output


def _best_window_metrics(root: Path) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for seed in SEEDS:
        output[str(seed)] = {}
        for arm in ARMS:
            path = _artifact_dir(root, arm, seed) / "training-summary.json"
            summary = json.loads(path.read_text(encoding="utf-8"))
            metrics = dict(summary["best_validation_metrics"]["window_metrics"])
            classes = {row["class"]: row for row in metrics["per_class"]}
            matrix = np.asarray(metrics["confusion_matrix_rows_true_columns_pred"], dtype=np.int64)
            output[str(seed)][arm] = {
                "best_epoch": int(summary["best_epoch"]),
                "support": int(metrics["support"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_precision": float(metrics["macro_precision"]),
                "macro_recall": float(metrics["macro_recall"]),
                "macro_f1": float(metrics["macro_f1"]),
                "DBB": {
                    name: float(classes["DBB"][name]) for name in ("precision", "recall", "f1", "support")
                },
                "CU": {
                    name: float(classes["CU"][name]) for name in ("precision", "recall", "f1", "support")
                },
                "errors": {"DBB_to_CU": int(matrix[0, 1]), "CU_to_DBB": int(matrix[1, 0])},
            }
    return output


def _write_transitions(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "parent_sample_id",
        "track_id",
        "label_name",
        "true_id",
        "control_pred_id",
        "treatment_pred_id",
        "control_confidence",
        "treatment_confidence",
        "window_length",
        "change",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    paired_path = root / PAIRED_REPORT
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if paired.get("status") != "completed_but_gate_failed" or paired.get("stage_one_passed") is not False:
        raise ValueError("V11 residual analysis requires a completed, failed paired report")
    index_rows = _read_index(root)
    train_parents = _parent_physical_rows(root, "train", index_rows["train"])
    val_parents = _parent_physical_rows(root, "val", index_rows["val"])
    prediction_analysis, transitions = _prediction_analysis(root)
    result = {
        "status": "passed",
        "schema_version": "RADAR-LN3-V11-rd-physical-residual-analysis-v1",
        "scope": "CQ08 V3 train feature description and validation paired residuals only; test sealed",
        "paired_report": str(paired_path),
        "paired_report_sha256": _sha256(paired_path),
        "paired_status": paired["status"],
        "stage_one_passed": False,
        "passing_seed_count": int(paired["passing_seed_count"]),
        "median_gate": paired["median_gate"],
        "predictions": prediction_analysis,
        "best_checkpoint_window_metrics": _best_window_metrics(root),
        "epoch_trajectory": _epoch_trajectory_analysis(root),
        "physical_feature_description": {
            "feature_names": list(RD_PHYSICAL_NAMES),
            "aggregation_limit": (
                "Parent means/final events are descriptive summaries; the liquid model consumes the full causal sequence, "
                "so these univariate AUC/effect sizes are not classifier performance."
            ),
            "train": _feature_analysis(train_parents),
            "validation": _feature_analysis(val_parents),
        },
        "learned_physical_fusion_weights": _weight_analysis(root),
        "interpretation_boundaries": [
            "Validation contains 232 parents but only 23 CU parents (8 Clutter, 15 Unknown).",
            "No p-value, confidence interval, deployment threshold or OOF estimate is produced.",
            "The four physical values are deterministic summaries of RD already consumed by the CNN; they add inductive bias, not new sensor data.",
            "Semantic-label route recall is not five-class precision/recall and cannot replace specialist or cascade evaluation.",
            "Test RD payloads and test metrics remain sealed and unexecuted.",
        ],
        "formal_training": "completed_by_user_before_this_read_only_analysis",
        "new_training": "未执行",
        "test_evaluation": "未执行",
        "OOF": "未执行",
        "threshold_fitting": "未执行",
        "P95": "未执行",
    }
    output_json = root / OUTPUT_JSON
    output_csv = root / OUTPUT_CSV
    _write_transitions(output_csv, transitions)
    result["parent_transition_csv"] = str(output_csv)
    result["parent_transition_csv_sha256"] = _sha256(output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result["result_path"] = str(output_json)
    result["result_sha256"] = _sha256(output_json)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze V11 paired residuals and four RD-physical feature separation.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.project_root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
