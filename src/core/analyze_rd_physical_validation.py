from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .rd_physical_drone_bird_trainer import _metrics


RD_ONLY_LABELS = ("DroneTarget", "BirdTarget")


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _label_index(label_name: object) -> int:
    try:
        return RD_ONLY_LABELS.index(str(label_name))
    except ValueError as error:
        raise ValueError(f"physical validation report contains an invalid label: {label_name!r}") from error


def _present_predictions(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in predictions if _as_bool(row.get("rd_present", True))]


def _validate_prediction_contract(predictions: list[dict[str, object]]) -> None:
    required = {"rd_present", "aggregation_weight"}
    missing = sorted({field for row in predictions for field in required if field not in row}) if predictions else sorted(required)
    if missing:
        raise ValueError(f"physical validation predictions lack required fields: {missing}")
    weights = [float(row["aggregation_weight"]) for row in predictions]
    if not all(torch.isfinite(torch.tensor(weights)).tolist()) or any(weight < 0.0 for weight in weights):
        raise ValueError("physical validation aggregation weights must be finite and non-negative")


def _frame_bin(frame_count: int) -> str:
    if frame_count <= 16:
        return "<=16"
    if frame_count <= 32:
        return "17-32"
    if frame_count <= 64:
        return "33-64"
    return "65+"


def _calibration(predictions: list[dict[str, object]], bin_count: int) -> dict[str, object]:
    if bin_count < 1:
        raise ValueError("calibration bin_count must be positive")
    probabilities = torch.tensor([[row["probability_drone"], row["probability_bird"]] for row in predictions], dtype=torch.float32)
    labels = torch.tensor([_label_index(row["label_name_true"]) for row in predictions], dtype=torch.long)
    confidence, predicted = probabilities.max(dim=-1)
    correct = predicted.eq(labels).float()
    ece = 0.0
    bins: list[dict[str, object]] = []
    for index in range(bin_count):
        low = index / bin_count
        high = (index + 1) / bin_count
        selected = (confidence >= low) & ((confidence < high) if index + 1 < bin_count else (confidence <= high))
        count = int(selected.sum())
        if count:
            accuracy = float(correct[selected].mean())
            average_confidence = float(confidence[selected].mean())
            ece += count / len(predictions) * abs(accuracy - average_confidence)
        else:
            accuracy = None
            average_confidence = None
        bins.append({"lower": low, "upper": high, "count": count, "accuracy": accuracy, "confidence": average_confidence})
    brier = {}
    for index, name in enumerate(RD_ONLY_LABELS):
        targets = labels.eq(index).float()
        brier[name] = float((probabilities[:, index] - targets).pow(2).mean())
    return {
        "ece": ece,
        "brier": brier,
        "bins": bins,
        "posthoc_calibration_applied": False,
        "threshold_selection_performed": False,
        "scope": "validation diagnostic only; probabilities are the uncalibrated checkpoint softmax",
    }


def _stratified_metrics(predictions: list[dict[str, object]], key_name: str) -> dict[str, object]:
    groups: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        key = _frame_bin(int(row["rd_window_frame_count"])) if key_name == "frame_bin" else str(row[key_name])
        groups[key].append(row)
    payload: dict[str, object] = {}
    for key, rows in sorted(groups.items()):
        logits = torch.tensor([[row["logit_drone"], row["logit_bird"]] for row in rows], dtype=torch.float32)
        labels = torch.tensor([_label_index(row["label_name_true"]) for row in rows], dtype=torch.long)
        weights = torch.tensor([float(row["aggregation_weight"]) for row in rows], dtype=torch.float32)
        payload[key] = {
            "window_count": len(rows),
            "metrics": _metrics(logits, labels, [str(row["parent_sample_id"]) for row in rows], weights),
        }
    return payload


def _causal_parent_metrics(predictions: list[dict[str, object]]) -> dict[str, object]:
    by_parent: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in predictions:
        by_parent[str(row["parent_sample_id"])].append(row)
    details: list[dict[str, object]] = []
    final_logits: list[torch.Tensor] = []
    final_labels: list[int] = []
    final_parent_ids: list[str] = []
    for parent_id, rows in sorted(by_parent.items()):
        rows = sorted(rows, key=lambda row: (float(row["rd_time_end_seconds"]), str(row["sample_id"])))
        truths = {str(row["label_name_true"]) for row in rows}
        if len(truths) != 1:
            raise ValueError(f"physical validation parent has inconsistent labels: {parent_id}")
        logits = torch.tensor([[row["logit_drone"], row["logit_bird"]] for row in rows], dtype=torch.float32)
        weights = torch.tensor([float(row["aggregation_weight"]) for row in rows], dtype=torch.float32)
        cumulative = (logits * weights.unsqueeze(1)).cumsum(dim=0) / weights.cumsum(dim=0).clamp_min(1.0e-6).unsqueeze(1)
        decisions = cumulative.argmax(dim=-1)
        final_decision = int(decisions[-1])
        truth = _label_index(next(iter(truths)))
        last_nonfinal = [index for index, value in enumerate(decisions.tolist()) if int(value) != final_decision]
        retrospective_stable_index = 0 if not last_nonfinal else min(len(rows) - 1, last_nonfinal[-1] + 1)
        flips = int((decisions[1:] != decisions[:-1]).sum()) if len(rows) > 1 else 0
        final_logits.append(cumulative[-1])
        final_labels.append(truth)
        final_parent_ids.append(parent_id)
        details.append(
            {
                "parent_sample_id": parent_id,
                "label_name_true": rows[0]["label_name_true"],
                "final_label_pred": RD_ONLY_LABELS[final_decision],
                "final_correct": bool(final_decision == truth),
                "flip_count": flips,
                "retrospective_stable_index": retrospective_stable_index,
                "retrospective_stable_from_time_seconds": float(
                    rows[retrospective_stable_index]["rd_time_end_seconds"]
                ),
                "source_name": str(rows[0]["source_name"]) if "source_name" in rows[0] else None,
            }
        )
    final_metric_payload = _metrics(
        torch.stack(final_logits),
        torch.tensor(final_labels, dtype=torch.long),
        final_parent_ids,
    )
    retrospective_times = [float(item["retrospective_stable_from_time_seconds"]) for item in details]
    return {
        "metric_scope": "offline cumulative-prefix replay; each prefix prediction is causal",
        "online_stable_decision_available": False,
        "stability_timing_scope": (
            "retrospective only: the reported time is found after observing the complete parent sequence and must not be used as online latency"
        ),
        "parent_count": len(details),
        "final_accuracy": sum(item["final_correct"] for item in details) / max(1, len(details)),
        "final_metrics": final_metric_payload["window"],
        "parents_with_flip_count": sum(item["flip_count"] > 0 for item in details),
        "flip_count_total": sum(item["flip_count"] for item in details),
        "retrospective_stable_from_time_seconds": {
            "mean": sum(retrospective_times) / max(1, len(retrospective_times)),
            "p95": float(torch.tensor(retrospective_times).quantile(0.95)) if retrospective_times else 0.0,
        },
        "details": details,
    }


def _rescue_harm(predictions: list[dict[str, object]], baseline_report_path: Path | None) -> dict[str, object] | None:
    if baseline_report_path is None:
        return None
    baseline_payload = json.loads(baseline_report_path.read_text(encoding="utf-8"))
    baseline_predictions = baseline_payload.get("predictions", [])
    if not isinstance(baseline_predictions, list):
        raise ValueError("baseline validation report predictions must be a list")
    _validate_prediction_contract(baseline_predictions)
    baseline = {
        str(row["sample_id"]): bool(row["is_correct"])
        for row in _present_predictions(baseline_predictions)
    }
    current = {str(row["sample_id"]): bool(row["is_correct"]) for row in _present_predictions(predictions)}
    if set(current) != set(baseline):
        raise ValueError("baseline and current prediction sample IDs do not match")
    rescue = sum(current[sample_id] and not baseline[sample_id] for sample_id in current)
    harm = sum(not current[sample_id] and baseline[sample_id] for sample_id in current)
    return {"window_rescue": rescue, "window_harm": harm, "window_net": rescue - harm}


def analyze_physical_rd_validation(
    report_path: Path,
    output_path: Path,
    *,
    baseline_report_path: Path | None = None,
    calibration_bins: int = 10,
) -> dict[str, object]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("split_name") != "val" or report.get("test_evaluation_executed"):
        raise ValueError("physical validation analysis requires a validation-only report")
    predictions = report.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("physical validation report has no predictions")
    _validate_prediction_contract(predictions)
    present_predictions = _present_predictions(predictions)
    if not present_predictions:
        raise ValueError("physical validation report has no present-RD predictions")
    stratified: dict[str, object] = {
        "frame_bin": _stratified_metrics(present_predictions, "frame_bin"),
        "window_mode": _stratified_metrics(present_predictions, "window_mode"),
    }
    if all(str(row.get("source_name", "")).strip() for row in present_predictions):
        stratified["source_name"] = {
            "available": True,
            "groups": _stratified_metrics(present_predictions, "source_name"),
        }
    else:
        stratified["source_name"] = {
            "available": False,
            "reason": "source_name is absent from the input report",
            "groups": {},
        }
    payload = {
        "scope": "physical RD-only validation analysis; closed-set Drone/Bird, no test or fusion",
        "source_report": report_path.as_posix(),
        "calibration": _calibration(present_predictions, calibration_bins),
        "stratified": stratified,
        "causal_parent": _causal_parent_metrics(present_predictions),
        "online_stable_decision": {
            "available": False,
            "reason": "offline report has no production streaming state-machine output; retrospective stabilization is not online evidence",
        },
        "rejection_capability": {
            "supports_rejection": False,
            "unknown_target_output": False,
            "abstain_output": False,
            "status": "当前支路不具备已验证的拒识能力",
        },
        "evaluation_completion": report.get("evaluation_completion", {"full_split_evaluated": None}),
        "rescue_harm": _rescue_harm(present_predictions, baseline_report_path),
        "prediction_count": int(len(predictions)),
        "present_rd_prediction_count": int(len(present_predictions)),
        "missing_rd_prediction_count": int(len(predictions) - len(present_predictions)),
        "test_evaluation_executed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a physical RD-only validation prediction report.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--baseline-report")
    parser.add_argument("--calibration-bins", type=int, default=10)
    args = parser.parse_args()
    result = analyze_physical_rd_validation(
        Path(args.report),
        Path(args.output),
        baseline_report_path=Path(args.baseline_report) if args.baseline_report else None,
        calibration_bins=args.calibration_bins,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "causal_parent"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
