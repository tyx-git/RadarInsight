from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

import pandas as pd


def _json_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _bucket(value: float, boundaries: tuple[float, ...]) -> str:
    for boundary in boundaries:
        if value <= boundary:
            return f"<={boundary:g}"
    return f">{boundaries[-1]:g}"


def _outcome(truth: str, trajectory_pred: str, rd_pred: str) -> str:
    trajectory_correct = trajectory_pred == truth
    rd_correct = rd_pred == truth
    if not trajectory_correct and rd_correct:
        return "rescue"
    if trajectory_correct and not rd_correct:
        return "harm"
    return "both_correct" if trajectory_correct else "both_wrong"


def _frame_sequence_stats(indices: set[int]) -> tuple[int, int, float]:
    if not indices:
        return 0, 0, 0.0
    ordered = sorted(indices)
    span = ordered[-1] - ordered[0] + 1
    longest_run = current_run = 1
    for previous, current in zip(ordered, ordered[1:]):
        if current == previous + 1:
            current_run += 1
        else:
            current_run = 1
        longest_run = max(longest_run, current_run)
    return span, longest_run, len(ordered) / span


def _load_predictions(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    predictions = payload.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError(f"prediction list missing: {path}")
    return {str(row["sample_id"]): dict(row) for row in predictions}


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    outcomes = Counter(str(row["outcome"]) for row in rows)
    rd_accuracy = sum(bool(row["rd_correct"]) for row in rows) / len(rows)
    trajectory_accuracy = sum(bool(row["trajectory_correct"]) for row in rows) / len(rows)
    return {
        "count": len(rows),
        "outcomes": dict(outcomes),
        "rescue": int(outcomes["rescue"]),
        "harm": int(outcomes["harm"]),
        "net_gain": int(outcomes["rescue"] - outcomes["harm"]),
        "rd_accuracy": float(rd_accuracy),
        "trajectory_accuracy": float(trajectory_accuracy),
        "mean_selected_frame_count": float(mean(float(row["selected_frame_count"]) for row in rows)),
        "mean_parent_frame_count": float(mean(float(row["parent_frame_count"]) for row in rows)),
    }


def _summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, object]]:
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {group_name: _group_summary(group_rows) for group_name, group_rows in sorted(grouped.items())}


def diagnose_rd_drone_bird_residuals(
    *,
    rd_report_path: Path,
    trajectory_report_path: Path,
    window_map_path: Path,
    manifest_path: Path,
    frame_index_path: Path,
    window_index_path: Path,
    output_path: Path,
) -> dict[str, object]:
    """Diagnose validation-only RD rescue/harm using canonical RD provenance."""
    rd_predictions = _load_predictions(rd_report_path)
    trajectory_predictions = _load_predictions(trajectory_report_path)
    sample_ids = sorted(set(rd_predictions) & set(trajectory_predictions))
    if not sample_ids or len(sample_ids) != len(rd_predictions):
        raise ValueError("RD-only predictions must be fully contained in frozen trajectory predictions")

    window_map = pd.read_csv(window_map_path, low_memory=False).set_index("sample_id", drop=False)
    manifest = pd.read_csv(manifest_path, low_memory=False).set_index("segment_id", drop=False)
    window_index = pd.read_csv(window_index_path, low_memory=False).set_index("sample_id", drop=False)
    frame_index = pd.read_csv(frame_index_path, low_memory=False)
    frame_shapes = {
        (str(row.parent_sample_id), int(row.tensor_frame_index)): str(row.rd_shape)
        for row in frame_index.itertuples(index=False)
    }
    parent_frame_counts = frame_index.groupby("parent_sample_id").size().to_dict()

    rows: list[dict[str, Any]] = []
    missing_metadata: list[str] = []
    for sample_id in sample_ids:
        rd_row = rd_predictions[sample_id]
        trajectory_row = trajectory_predictions[sample_id]
        for key in ("parent_sample_id", "label_name_true", "source_name"):
            if str(rd_row.get(key, "")) != str(trajectory_row.get(key, "")):
                raise ValueError(f"misaligned {key} for {sample_id}")
        if sample_id not in window_map.index or sample_id not in window_index.index:
            missing_metadata.append(sample_id)
            continue
        map_row = window_map.loc[sample_id]
        parent_id = str(rd_row["parent_sample_id"])
        segment_ids = [str(value) for value in _json_list(map_row.get("segment_ids_json"))]
        tensor_indices: set[int] = set()
        for segment_id in segment_ids:
            if segment_id not in manifest.index:
                raise ValueError(f"missing manifest segment {segment_id}")
            segment = manifest.loc[segment_id]
            tensor_indices.update(int(value) for value in _json_list(segment.get("rd_tensor_frame_indices_json")))
        selected_shapes = Counter(
            frame_shapes.get((parent_id, tensor_index), "missing_shape")
            for tensor_index in tensor_indices
        )
        shape_names = sorted(name for name in selected_shapes if name != "missing_shape")
        shape_signature = "+".join(shape_names) if shape_names else "missing_shape"
        frame_span, max_contiguous_run, frame_density = _frame_sequence_stats(tensor_indices)
        truth = str(rd_row["label_name_true"])
        trajectory_pred = str(trajectory_row["label_name_traj_pred"])
        rd_pred = str(rd_row["label_name_rd_only_pred"])
        window_row = window_index.loc[sample_id]
        parent_frame_count = int(parent_frame_counts.get(parent_id, 0))
        rows.append(
            {
                "sample_id": sample_id,
                "parent_sample_id": parent_id,
                "truth": truth,
                "trajectory_pred": trajectory_pred,
                "rd_pred": rd_pred,
                "outcome": _outcome(truth, trajectory_pred, rd_pred),
                "trajectory_correct": trajectory_pred == truth,
                "rd_correct": rd_pred == truth,
                "rd_bird_minus_drone_margin": float(rd_row["bird_logit_bird"]) - float(rd_row["bird_logit_drone"]),
                "selected_frame_count": len(tensor_indices),
                "selected_frame_count_bucket": _bucket(float(len(tensor_indices)), (4.0, 8.0, 16.0)),
                "selected_frame_span": frame_span,
                "selected_frame_density": frame_density,
                "selected_frame_density_bucket": _bucket(frame_density, (0.5, 0.8, 0.95)),
                "max_contiguous_run": max_contiguous_run,
                "max_contiguous_run_bucket": _bucket(float(max_contiguous_run), (4.0, 8.0, 16.0)),
                "selected_shape_signature": shape_signature,
                "truth_shape_signature": f"{truth}|{shape_signature}",
                "selected_shape_diversity": str(len(shape_names)),
                "parent_frame_count": parent_frame_count,
                "parent_frame_count_bucket": _bucket(float(parent_frame_count), (64.0, 128.0, 256.0)),
                "window_duration_seconds": float(window_row["window_duration_seconds"]),
                "window_duration_bucket": _bucket(float(window_row["window_duration_seconds"]), (10.0, 20.0, 40.0)),
                "window_point_count": int(window_row["window_point_count"]),
                "window_point_count_bucket": _bucket(float(window_row["window_point_count"]), (4.0, 8.0, 16.0)),
                "segment_count": len(segment_ids),
            }
        )
    if missing_metadata:
        raise ValueError(f"missing v3 RD metadata for {len(missing_metadata)} predictions")

    parent_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parent_groups[str(row["parent_sample_id"])].append(row)
    parent_records = []
    for parent_id, parent_rows in parent_groups.items():
        summary = _group_summary(parent_rows)
        summary.update(
            {
                "parent_sample_id": parent_id,
                "truth": parent_rows[0]["truth"],
                "shape_signatures": sorted({str(row["selected_shape_signature"]) for row in parent_rows}),
            }
        )
        parent_records.append(summary)
    parent_records.sort(key=lambda row: (-int(row["harm"]), int(row["net_gain"]), -int(row["count"]), str(row["parent_sample_id"])))

    payload = {
        "scope": "validation-only RD-only versus frozen trajectory residual diagnosis; no fusion, threshold tuning, training, or test evaluation",
        "inputs": {
            "rd_report": str(rd_report_path),
            "trajectory_report": str(trajectory_report_path),
            "window_map": str(window_map_path),
            "manifest": str(manifest_path),
            "canonical_frame_index": str(frame_index_path),
            "window_index": str(window_index_path),
        },
        "alignment": {
            "rd_prediction_count": len(rd_predictions),
            "trajectory_prediction_count": len(trajectory_predictions),
            "intersection_count": len(sample_ids),
            "metadata_complete": True,
        },
        "overall": _group_summary(rows),
        "by_truth": _summarize_groups(rows, "truth"),
        "by_selected_shape_signature": _summarize_groups(rows, "selected_shape_signature"),
        "by_truth_shape_signature": _summarize_groups(rows, "truth_shape_signature"),
        "by_selected_shape_diversity": _summarize_groups(rows, "selected_shape_diversity"),
        "by_selected_frame_count": _summarize_groups(rows, "selected_frame_count_bucket"),
        "by_selected_frame_density": _summarize_groups(rows, "selected_frame_density_bucket"),
        "by_max_contiguous_run": _summarize_groups(rows, "max_contiguous_run_bucket"),
        "by_parent_frame_count": _summarize_groups(rows, "parent_frame_count_bucket"),
        "by_window_duration": _summarize_groups(rows, "window_duration_bucket"),
        "by_window_point_count": _summarize_groups(rows, "window_point_count_bucket"),
        "top_harm_parents": parent_records[:25],
        "test_evaluation_executed": False,
        "fusion_enabled": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Diagnose validation-only Drone/Bird RD rescue and harm by canonical RD provenance.")
    parser.add_argument("--rd-report", required=True)
    parser.add_argument("--trajectory-report", required=True)
    parser.add_argument("--window-map", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--frame-index", required=True)
    parser.add_argument("--window-index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    payload = diagnose_rd_drone_bird_residuals(
        rd_report_path=Path(args.rd_report),
        trajectory_report_path=Path(args.trajectory_report),
        window_map_path=Path(args.window_map),
        manifest_path=Path(args.manifest),
        frame_index_path=Path(args.frame_index),
        window_index_path=Path(args.window_index),
        output_path=Path(args.output),
    )
    print(json.dumps({"output": args.output, "overall": payload["overall"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
