"""V27d-r15 的闭集、混合角色与 Bird/Unknown 错误评估。"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from .data import R15L1Dataset, build_r15_dataloader, r15_model_input_kwargs, r15_sidecar_kwargs
from .features import load_observation_cache
from .model import R15L1Model
from .protocol import (
    R15_ARMS,
    R15_BASE_LABELS,
    R15_CLASS_NAMES,
    R15_EXPERIMENT_ID,
    R15_REPORT_SCHEMA,
    R15_SUMMARY_REPORT_SCHEMA,
    R15Context,
    artifact_path,
    read_json,
    read_sealed_json,
    write_sealed_json_if_identical,
)
from .training import load_r15_checkpoint


def _label_id(label: str) -> int:
    if label in {"DroneTarget", "BirdTarget", "BalloonTarget"}:
        return 0
    if label == "ClutterTarget":
        return 1
    raise ValueError(f"V27d-r15 closed-set label is unsupported: {label}")


def _metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for row in records:
        matrix[int(row["true_id"]), int(row["pred_id"])] += 1
    per_class: list[dict[str, Any]] = []
    for index, name in enumerate(R15_CLASS_NAMES):
        tp = int(matrix[index, index])
        support = int(matrix[index].sum())
        predicted = int(matrix[:, index].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "class": name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    total = int(matrix.sum())
    return {
        "support": total,
        "accuracy": float(np.trace(matrix) / total) if total else None,
        "macro_precision": float(np.mean([item["precision"] for item in per_class])),
        "macro_recall": float(np.mean([item["recall"] for item in per_class])),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "balanced_accuracy": float(np.mean([item["recall"] for item in per_class])),
        "per_class": per_class,
        "confusion_matrix_rows_true_columns_pred": matrix.tolist(),
    }


def _baseline_views(context: R15Context) -> dict[str, dict[str, Any]]:
    manifest = read_json(
        context.project_root / context.config.source["v27d_endpoint_manifest_path"]
    )
    result: dict[str, dict[str, Any]] = {}
    for role in ("representation_development", "calibration"):
        entry = manifest["tasks"][f"known_gate:{role}"]
        payload = torch.load(
            context.project_root / entry["path"],
            map_location="cpu",
            weights_only=False,
        )
        tensors = payload["tensors"]
        result[role] = {
            "parent_ids": [str(value) for value in payload["parent_ids"]],
            "label_names": [str(value) for value in payload["label_names"]],
            "lengths": [int(value) for value in tensors["lengths"].tolist()],
            "logits": tensors["endpoint_sequence_logits"].detach().cpu(),
        }
    return result


def _baseline_records(view: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    logits = view["logits"]
    for row, (parent_id, label, length) in enumerate(
        zip(view["parent_ids"], view["label_names"], view["lengths"], strict=True)
    ):
        final = logits[row, int(length) - 1]
        records.append(
            {
                "parent_sample_id": str(parent_id),
                "label_name": str(label),
                "true_id": _label_id(str(label)) if str(label) != "UnknownTarget" else None,
                "pred_id": int(final.argmax()),
                "logits": [float(value) for value in final.tolist()],
                "arm": "v27_l1_epoch14",
            }
        )
    return records


def _error_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts = defaultdict(int)
    for row in records:
        label = str(row["label_name"])
        prediction = int(row["pred_id"])
        if label == "BirdTarget":
            counts["bird_support"] += 1
            counts["bird_predicted_clutter"] += int(prediction == 1)
        elif label == "UnknownTarget":
            counts["unknown_support"] += 1
            counts["unknown_predicted_target"] += int(prediction == 0)
        elif label in {"DroneTarget", "BalloonTarget"}:
            counts[f"{label}_support"] += 1
            counts[f"{label}_predicted_target"] += int(prediction == 0)
        elif label == "ClutterTarget":
            counts["clutter_support"] += 1
            counts["clutter_predicted_clutter"] += int(prediction == 1)
    return {
        **dict(counts),
        "bird_error_rate": counts["bird_predicted_clutter"] / max(1, counts["bird_support"]),
        "unknown_error_rate": counts["unknown_predicted_target"] / max(1, counts["unknown_support"]),
    }


def _delta(left: Mapping[str, Any], right: Mapping[str, Any], fields: Sequence[str]) -> dict[str, float]:
    return {field: float(right[field]) - float(left[field]) for field in fields}


def _infer_role(
    context: R15Context,
    *,
    arm: str,
    seed: int,
    view_name: str,
    device: torch.device,
    model: R15L1Model | None = None,
) -> list[dict[str, Any]]:
    if model is None:
        model, _ = load_r15_checkpoint(context, arm=arm, seed=seed, device=device)
    dataset = R15L1Dataset(context, view=view_name)
    loader = build_r15_dataloader(
        dataset,
        batch_size=int(context.config.training["batch_size"]),
        shuffle=False,
    )
    records: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = r15_model_input_kwargs(batch, device)
            sidecars = r15_sidecar_kwargs(batch, device)
            output = model(base_inputs=inputs, **sidecars)
            final = output["final_logits"].detach().cpu()
            for row, parent_id in enumerate(batch["parent_sample_ids"]):
                label = str(batch["base_label_names"][row])
                records.append(
                    {
                        "parent_sample_id": str(parent_id),
                        "label_name": label,
                        "true_id": _label_id(label) if label != "UnknownTarget" else None,
                        "pred_id": int(final[row].argmax()),
                        "logits": [float(value) for value in final[row].tolist()],
                        "arm": arm,
                        "seed": int(seed),
                    }
                )
    if [row["parent_sample_id"] for row in records] != dataset.parent_ids:
        raise ValueError(f"V27d-r15 inference changed parent order: {view_name}")
    return records


def _paired_transitions(
    baseline: Sequence[Mapping[str, Any]], treatment: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    left = {str(row["parent_sample_id"]): row for row in baseline}
    right = {str(row["parent_sample_id"]): row for row in treatment}
    if set(left) != set(right):
        raise ValueError("V27d-r15 paired records have different parent sets")
    transitions = defaultdict(int)
    details: list[dict[str, Any]] = []
    for parent_id in sorted(left):
        before = left[parent_id]
        after = right[parent_id]
        label = str(after["label_name"])
        if label != str(before["label_name"]):
            raise ValueError("V27d-r15 paired labels differ")
        before_correct = before["true_id"] is not None and int(before["pred_id"]) == int(before["true_id"])
        after_correct = after["true_id"] is not None and int(after["pred_id"]) == int(after["true_id"])
        transitions[f"{'correct' if before_correct else 'wrong'}_to_{'correct' if after_correct else 'wrong'}"] += 1
        if int(before["pred_id"]) != int(after["pred_id"]):
            details.append(
                {
                    "parent_sample_id": parent_id,
                    "label_name": label,
                    "baseline_prediction": int(before["pred_id"]),
                    "treatment_prediction": int(after["pred_id"]),
                }
            )
    return {"parent_count": len(left), "correctness_transitions": dict(sorted(transitions.items())), "changed_predictions": details}


def evaluate_seed(context: R15Context, *, seed: int) -> dict[str, Any]:
    device = torch.device("cpu")
    baseline_views = _baseline_views(context)
    validation_baseline = _baseline_records(
        {
            "parent_ids": [str(value) for value in context_manifest_validation_ids(context)],
            "label_names": context_manifest_validation_labels(context),
            "lengths": context_manifest_validation_lengths(context),
            "logits": context_manifest_validation_logits(context),
        }
    )
    # The calibration endpoint contains the same 232 known validation parents
    # plus Unknown.  Filter to the known-only V27 validation view by identity.
    known_validation_ids = set(R15L1Dataset(context, view="validation").parent_ids)
    validation_baseline = [row for row in _baseline_records(baseline_views["calibration"]) if row["parent_sample_id"] in known_validation_ids]
    validation_records: dict[str, list[dict[str, Any]]] = {}
    for arm in R15_ARMS:
        validation_records[arm] = _infer_role(context, arm=arm, seed=seed, view_name="validation", device=device)
    by_id = {row["parent_sample_id"]: row for row in validation_baseline}
    baseline_known = [by_id[parent_id] for parent_id in known_validation_ids]
    baseline_known.sort(key=lambda row: validation_records["control"].index(next(item for item in validation_records["control"] if item["parent_sample_id"] == row["parent_sample_id"])))

    roles: dict[str, Any] = {}
    for role in ("representation_development", "calibration"):
        role_baseline = _baseline_records(baseline_views[role])
        roles[role] = {"baseline": role_baseline}
        for arm in R15_ARMS:
            roles[role][arm] = _infer_role(context, arm=arm, seed=seed, view_name=role, device=device)

    closed = {
        "v27_l1_epoch14": {"records": baseline_known, "metrics": _metrics(baseline_known)},
        "control": {"records": validation_records["control"], "metrics": _metrics(validation_records["control"])},
        "v10_observation_increment": {"records": validation_records["v10_observation_increment"], "metrics": _metrics(validation_records["v10_observation_increment"])},
    }
    report: dict[str, Any] = {
        "schema_version": R15_REPORT_SCHEMA,
        "experiment_id": "V27d-r15",
        "status": "completed_v10_observation_increment_l1_evaluation",
        "seed": int(seed),
        "manifest_sha256": context.manifest_sha256,
        "class_names": list(R15_CLASS_NAMES),
        "closed_set": closed,
        "roles": {},
        "scope": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "formal_l3_enabled": False,
            "unknown_is_classifier_class": False,
        },
    }
    for role, value in roles.items():
        baseline = value["baseline"]
        role_report = {"baseline": {"metrics": _metrics([row for row in baseline if row["true_id"] is not None]), "errors": _error_summary(baseline)}}
        for arm in R15_ARMS:
            records = value[arm]
            role_report[arm] = {
                "metrics_known_only": _metrics([row for row in records if row["true_id"] is not None]),
                "errors": _error_summary(records),
                "paired_with_v27_l1": _paired_transitions(baseline, records),
            }
        role_report["treatment_minus_baseline_error_delta"] = {
            "bird_error_count_delta": role_report["v10_observation_increment"]["errors"].get("bird_predicted_clutter", 0) - role_report["baseline"]["errors"].get("bird_predicted_clutter", 0),
            "unknown_error_count_delta": role_report["v10_observation_increment"]["errors"].get("unknown_predicted_target", 0) - role_report["baseline"]["errors"].get("unknown_predicted_target", 0),
            "bird_error_rate_delta": role_report["v10_observation_increment"]["errors"]["bird_error_rate"] - role_report["baseline"]["errors"]["bird_error_rate"],
            "unknown_error_rate_delta": role_report["v10_observation_increment"]["errors"]["unknown_error_rate"] - role_report["baseline"]["errors"]["unknown_error_rate"],
        }
        report["roles"][role] = role_report
    report["closed_set_deltas"] = {
        "control_minus_v27_l1": _metric_delta(closed["v27_l1_epoch14"]["metrics"], closed["control"]["metrics"]),
        "v10_treatment_minus_control": _metric_delta(closed["control"]["metrics"], closed["v10_observation_increment"]["metrics"]),
    }
    report["closed_set_transitions"] = {
        "control_vs_v27_l1": _paired_transitions(validation_baseline, validation_records["control"]),
        "v10_treatment_vs_v27_l1": _paired_transitions(validation_baseline, validation_records["v10_observation_increment"]),
        "v10_treatment_vs_control": _paired_transitions(validation_records["control"], validation_records["v10_observation_increment"]),
    }
    path = artifact_path(context.project_root, context.config.artifacts["evaluation_template"], seed=seed)
    write_sealed_json_if_identical(path, report)
    return {**report, "report_path": str(path)}


def evaluate_single_arm(context: R15Context, *, arm: str, seed: int) -> dict[str, Any]:
    """Evaluate one retrained arm without requiring the comparison arm checkpoint."""

    if arm not in R15_ARMS:
        raise ValueError("V27d-r15 single-arm evaluation arm is invalid")
    baseline_views = _baseline_views(context)
    known_ids = set(R15L1Dataset(context, view="validation").parent_ids)
    baseline = [
        row
        for row in _baseline_records(baseline_views["calibration"])
        if row["parent_sample_id"] in known_ids
    ]
    records = _infer_role(
        context,
        arm=arm,
        seed=seed,
        view_name="validation",
        device=torch.device("cpu"),
    )
    baseline_metrics = _metrics(baseline)
    arm_metrics = _metrics(records)
    return {
        "schema_version": "RADAR-LN3-V27d-r15-single-arm-evaluation-v1",
        "experiment_id": R15_EXPERIMENT_ID,
        "arm": arm,
        "seed": int(seed),
        "baseline": baseline_metrics,
        "metrics": arm_metrics,
        "delta_vs_v27_l1": _delta(
            baseline_metrics,
            arm_metrics,
            ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"),
        ),
        "scope": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "formal_l3_enabled": False,
            "unknown_is_classifier_class": False,
        },
    }


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    fields = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    return {field: float(right[field]) - float(left[field]) for field in fields}


def _records_by_seed(context: R15Context) -> list[dict[str, Any]]:
    result = []
    for seed in context.config.training["seeds"]:
        path = artifact_path(context.project_root, context.config.artifacts["evaluation_template"], seed=int(seed))
        result.append(read_sealed_json(path, expected_schema=R15_REPORT_SCHEMA))
    return result


def summarize_seeds(context: R15Context) -> dict[str, Any]:
    reports = _records_by_seed(context)
    treatment_metrics = [report["closed_set"]["v10_observation_increment"]["metrics"] for report in reports]
    control_metrics = [report["closed_set"]["control"]["metrics"] for report in reports]
    summary = {
        "schema_version": R15_SUMMARY_REPORT_SCHEMA,
        "experiment_id": "V27d-r15",
        "status": "two_seed_summary",
        "manifest_sha256": context.manifest_sha256,
        "seeds": [int(value) for value in context.config.training["seeds"]],
        "closed_set": {
            "control": _aggregate_metrics(control_metrics),
            "v10_observation_increment": _aggregate_metrics(treatment_metrics),
            "v10_treatment_minus_control": _aggregate_deltas(
                [report["closed_set_deltas"]["v10_treatment_minus_control"] for report in reports]
            ),
        },
        "open_set_error_counts": {
            role: {
                "baseline": [report["roles"][role]["baseline"]["errors"] for report in reports],
                "control": [report["roles"][role]["control"]["errors"] for report in reports],
                "v10_observation_increment": [report["roles"][role]["v10_observation_increment"]["errors"] for report in reports],
                "treatment_minus_baseline": [report["roles"][role]["treatment_minus_baseline_error_delta"] for report in reports],
            }
            for role in ("representation_development", "calibration")
        },
        "scope": {
            "unknown_training_exposure": 0,
            "test_payload_access_count": 0,
            "formal_l3_enabled": False,
            "deployment_eligible": False,
        },
    }
    path = artifact_path(context.project_root, context.config.artifacts["summary_path"])
    write_sealed_json_if_identical(path, summary)
    return {**summary, "report_path": str(path)}


def _aggregate_metrics(metrics: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    fields = ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    result: dict[str, Any] = {}
    for field in fields:
        values = np.asarray([float(item[field]) for item in metrics], dtype=np.float64)
        result[field] = {"values": values.tolist(), "mean": float(values.mean()), "std": float(values.std(ddof=0))}
    result["per_class"] = [
        {
            "class": name,
            "precision_values": [float(item["per_class"][index]["precision"]) for item in metrics],
            "recall_values": [float(item["per_class"][index]["recall"]) for item in metrics],
            "f1_values": [float(item["per_class"][index]["f1"]) for item in metrics],
        }
        for index, name in enumerate(R15_CLASS_NAMES)
    ]
    return result


def _aggregate_deltas(deltas: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        field: {
            "values": [float(item[field]) for item in deltas],
            "mean": float(np.mean([float(item[field]) for item in deltas])),
        }
        for field in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy")
    }


# 以下三个小函数只负责读取 validation endpoint 的稳定顺序，避免把顺序
# 推断散落到评估流程中。它们在 evaluate_seed 前由 endpoint cache 提供。
def context_manifest_validation_ids(context: R15Context) -> list[str]:
    return _baseline_views(context)["calibration"]["parent_ids"]


def context_manifest_validation_labels(context: R15Context) -> list[str]:
    return _baseline_views(context)["calibration"]["label_names"]


def context_manifest_validation_lengths(context: R15Context) -> list[int]:
    return _baseline_views(context)["calibration"]["lengths"]


def context_manifest_validation_logits(context: R15Context) -> torch.Tensor:
    return _baseline_views(context)["calibration"]["logits"]


__all__ = ["evaluate_seed", "summarize_seeds"]
