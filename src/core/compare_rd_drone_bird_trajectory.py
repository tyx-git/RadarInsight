from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    ordered = sorted(values)

    def value_at(fraction: float) -> float:
        position = (len(ordered) - 1) * fraction
        lower = int(position)
        upper = min(len(ordered) - 1, lower + 1)
        weight = position - lower
        return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)

    return {
        "count": len(ordered),
        "mean": float(mean(ordered)),
        "p10": value_at(0.10),
        "p50": value_at(0.50),
        "p90": value_at(0.90),
    }


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"prediction list missing: {path}")
    return predictions


def compare_rd_only_to_trajectory(
    rd_report_path: Path,
    trajectory_report_path: Path,
    output_path: Path,
) -> dict[str, object]:
    rd_rows = {str(item["sample_id"]): item for item in _load_predictions(rd_report_path)}
    trajectory_rows = {str(item["sample_id"]): item for item in _load_predictions(trajectory_report_path)}
    intersection = sorted(set(rd_rows) & set(trajectory_rows))
    if not intersection:
        raise ValueError("RD-only and trajectory reports have no shared samples")
    if len(intersection) != len(rd_rows):
        raise ValueError("RD-only report contains samples absent from trajectory report")

    baseline_confusion: Counter[tuple[str, str]] = Counter()
    rd_confusion: Counter[tuple[str, str]] = Counter()
    change_matrix: Counter[tuple[str, str, str]] = Counter()
    rescue_by_truth: Counter[str] = Counter()
    harm_by_truth: Counter[str] = Counter()
    rd_margin_by_outcome: defaultdict[str, list[float]] = defaultdict(list)
    rd_margin_by_change: defaultdict[str, list[float]] = defaultdict(list)
    parent_rows: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    for sample_id in intersection:
        rd_row = rd_rows[sample_id]
        trajectory_row = trajectory_rows[sample_id]
        for key in ("parent_sample_id", "label_name_true", "source_name"):
            if str(rd_row.get(key, "")) != str(trajectory_row.get(key, "")):
                raise ValueError(f"misaligned {key} for sample {sample_id}")
        truth = str(rd_row["label_name_true"])
        trajectory_pred = str(trajectory_row["label_name_traj_pred"])
        rd_pred = str(rd_row["label_name_rd_only_pred"])
        margin = float(rd_row["bird_logit_bird"]) - float(rd_row["bird_logit_drone"])
        baseline_confusion[(truth, trajectory_pred)] += 1
        rd_confusion[(truth, rd_pred)] += 1
        change_matrix[(truth, trajectory_pred, rd_pred)] += 1
        trajectory_correct = trajectory_pred == truth
        rd_correct = rd_pred == truth
        if not trajectory_correct and rd_correct:
            outcome = "rescue"
            rescue_by_truth[truth] += 1
        elif trajectory_correct and not rd_correct:
            outcome = "harm"
            harm_by_truth[truth] += 1
        elif trajectory_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        rd_margin_by_outcome[outcome].append(margin)
        rd_margin_by_change[f"{truth}|{trajectory_pred}|{rd_pred}"].append(margin)
        parent_rows[str(rd_row["parent_sample_id"])].append(
            {
                "truth": truth,
                "trajectory_pred": trajectory_pred,
                "rd_margin": margin,
            }
        )

    parent_outcomes: Counter[str] = Counter()
    parent_records: list[dict[str, object]] = []
    for parent_id, rows in sorted(parent_rows.items()):
        truth_values = {str(item["truth"]) for item in rows}
        if len(truth_values) != 1:
            raise ValueError(f"parent has inconsistent labels: {parent_id}")
        truth = next(iter(truth_values))
        votes = Counter(str(item["trajectory_pred"]) for item in rows)
        trajectory_parent_pred = votes.most_common(1)[0][0]
        rd_mean_margin = float(mean(float(item["rd_margin"]) for item in rows))
        rd_parent_pred = "BirdTarget" if rd_mean_margin > 0.0 else "DroneTarget"
        trajectory_correct = trajectory_parent_pred == truth
        rd_correct = rd_parent_pred == truth
        if not trajectory_correct and rd_correct:
            outcome = "rescue"
        elif trajectory_correct and not rd_correct:
            outcome = "harm"
        elif trajectory_correct:
            outcome = "both_correct"
        else:
            outcome = "both_wrong"
        parent_outcomes[outcome] += 1
        parent_records.append(
            {
                "parent_sample_id": parent_id,
                "label_name_true": truth,
                "trajectory_majority_pred": trajectory_parent_pred,
                "rd_mean_pred": rd_parent_pred,
                "window_count": len(rows),
                "rd_mean_bird_minus_drone_margin": rd_mean_margin,
                "outcome": outcome,
            }
        )

    payload = {
        "scope": "frozen validation-only CQ08 Drone/Bird sample intersection; no test or fusion",
        "inputs": {
            "rd_only_report": rd_report_path.as_posix(),
            "trajectory_report": trajectory_report_path.as_posix(),
        },
        "alignment": {
            "rd_only_sample_count": len(rd_rows),
            "trajectory_sample_count": len(trajectory_rows),
            "intersection_sample_count": len(intersection),
            "ids_parent_truth_source_aligned": True,
        },
        "window": {
            "truth_counts": dict(Counter(str(rd_rows[item]["label_name_true"]) for item in intersection)),
            "trajectory_confusion": {"|".join(key): value for key, value in sorted(baseline_confusion.items())},
            "rd_only_confusion": {"|".join(key): value for key, value in sorted(rd_confusion.items())},
            "change_matrix": {"|".join(key): value for key, value in sorted(change_matrix.items())},
            "rescue_by_truth": dict(rescue_by_truth),
            "harm_by_truth": dict(harm_by_truth),
            "rescue_total": int(sum(rescue_by_truth.values())),
            "harm_total": int(sum(harm_by_truth.values())),
            "net_gain": int(sum(rescue_by_truth.values()) - sum(harm_by_truth.values())),
            "rd_bird_minus_drone_margin_by_outcome": {
                key: _quantiles(values) for key, values in sorted(rd_margin_by_outcome.items())
            },
            "rd_bird_minus_drone_margin_by_change": {
                key: _quantiles(values) for key, values in sorted(rd_margin_by_change.items())
            },
        },
        "parent_majority_vs_rd_mean": {
            "parent_count": len(parent_records),
            "outcome_counts": dict(parent_outcomes),
            "rescue_count": int(parent_outcomes["rescue"]),
            "harm_count": int(parent_outcomes["harm"]),
            "net_gain": int(parent_outcomes["rescue"] - parent_outcomes["harm"]),
            "records": parent_records,
        },
        "limitations": [
            "Frozen trajectory prediction report contains labels but not trajectory logits; the report therefore compares RD confidence margins by outcome, not cross-model calibrated confidence deltas.",
            "track_point_anchor RD remains diagnostic-only; this comparison never enables fusion.",
        ],
        "test_evaluation_executed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validation-only RD-only and frozen trajectory predictions without fusion.")
    parser.add_argument("--rd-only-report", required=True)
    parser.add_argument("--trajectory-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = compare_rd_only_to_trajectory(
        Path(args.rd_only_report),
        Path(args.trajectory_report),
        Path(args.output),
    )
    print(json.dumps({
        "output": args.output,
        "alignment": payload["alignment"],
        "window": {key: value for key, value in payload["window"].items() if key not in {"change_matrix", "trajectory_confusion", "rd_only_confusion", "rd_bird_minus_drone_margin_by_change"}},
        "parent": {key: value for key, value in payload["parent_majority_vs_rd_mean"].items() if key != "records"},
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
