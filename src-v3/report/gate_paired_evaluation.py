from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping

import numpy as np

from radar_ln3.task_models import BASE_LABELS, summarize_task_records


SCHEMA_VERSION = "RADAR-LN3-V9-V10-gate-paired-validation-v1"
CLASS_NAMES = ("DBB", "CU")
DEFAULT_BASELINE_DETAIL = Path(
    "artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/sample_detail_information.json"
)
DEFAULT_CANDIDATE_DETAIL = Path(
    "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/sample_detail_information.json"
)
DEFAULT_BASELINE_EVALUATION = Path(
    "artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/evaluation-val.json"
)
DEFAULT_CANDIDATE_EVALUATION = Path(
    "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/evaluation-val.json"
)
DEFAULT_BASELINE_TRAINING = Path(
    "artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/training-summary.json"
)
DEFAULT_CANDIDATE_TRAINING = Path(
    "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/training-summary.json"
)
DEFAULT_BASELINE_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V9-cq08-liquid-dbb-cu-gate-motion-formal/best_model.pt"
)
DEFAULT_CANDIDATE_CHECKPOINT = Path(
    "artifacts/RADAR-LN3-V10-cq08-liquid-dbb-cu-gate-observation-increment-formal/best_model.pt"
)
DEFAULT_OUTPUT = Path(
    "src-v3/processed/report/RADAR-LN3-V9-V10-gate-paired-validation.json"
)


def _resolve(root: Path, value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report input is not a JSON object: {path}")
    return payload


def _probability_cu(record: Mapping[str, Any]) -> float:
    prediction = int(record["pred_id"])
    confidence = float(record["confidence"])
    if prediction not in {0, 1} or not 0.0 <= confidence <= 1.0:
        raise ValueError("paired Gate record has invalid prediction/confidence")
    return confidence if prediction == 1 else 1.0 - confidence


def _record_key(record: Mapping[str, Any], scope: str) -> str:
    parent = str(record["parent_sample_id"])
    return parent if scope == "parent" else f"{parent}|event|{int(record['event_index'])}"


def _index_records(records: Iterable[Mapping[str, Any]], scope: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        key = _record_key(record, scope)
        if key in indexed:
            raise ValueError(f"duplicate paired record key: {key}")
        record["record_key"] = key
        record["probability_cu"] = _probability_cu(record)
        indexed[key] = record
    return indexed


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if not array.size:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p05": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
        }
    if not np.isfinite(array).all():
        raise ValueError("paired diagnostic received a non-finite value")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def _validate_pair(
    baseline: Mapping[str, dict[str, Any]],
    candidate: Mapping[str, dict[str, Any]],
    scope: str,
) -> None:
    if set(baseline) != set(candidate):
        raise ValueError(f"V9/V10 {scope} record key sets differ")
    for key in baseline:
        left = baseline[key]
        right = candidate[key]
        immutable = (
            "true_id",
            "parent_sample_id",
            "track_id",
            "source_name",
            "label_name",
            "event_index",
            "window_length",
        )
        changed = [name for name in immutable if left.get(name) != right.get(name)]
        if changed:
            raise ValueError(f"V9/V10 {scope} immutable fields differ for {key}: {changed}")
        expected_true = 0 if str(left["label_name"]) in BASE_LABELS[:3] else 1
        if int(left["true_id"]) != expected_true:
            raise ValueError(f"base label and Gate truth disagree for {key}")


def _metric_deltas(
    baseline_records: list[dict[str, Any]],
    candidate_records: list[dict[str, Any]],
    scope: str,
) -> dict[str, Any]:
    baseline = summarize_task_records(
        baseline_records,
        class_names=CLASS_NAMES,
        scope=f"V9:{scope}",
    )
    candidate = summarize_task_records(
        candidate_records,
        class_names=CLASS_NAMES,
        scope=f"V10:{scope}",
    )
    scalar_names = (
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "balanced_accuracy",
    )
    per_class: dict[str, Any] = {}
    for class_name in CLASS_NAMES:
        left = next(row for row in baseline["per_class"] if row["class"] == class_name)
        right = next(row for row in candidate["per_class"] if row["class"] == class_name)
        per_class[class_name] = {
            "V9": left,
            "V10": right,
            "delta_V10_minus_V9": {
                name: float(right[name]) - float(left[name])
                for name in ("precision", "recall", "f1")
            },
        }
    return {
        "V9": baseline,
        "V10": candidate,
        "delta_V10_minus_V9": {
            name: float(candidate[name]) - float(baseline[name]) for name in scalar_names
        },
        "per_class": per_class,
    }


def _base_label_routes(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {name: [] for name in BASE_LABELS}
    for record in records:
        grouped[str(record["label_name"])].append(record)
    result: dict[str, Any] = {}
    for label_name in BASE_LABELS:
        rows = grouped[label_name]
        true_id = 0 if label_name in BASE_LABELS[:3] else 1
        dbb = sum(int(row["pred_id"]) == 0 for row in rows)
        cu = len(rows) - dbb
        result[label_name] = {
            "support": len(rows),
            "predicted_DBB": int(dbb),
            "predicted_CU": int(cu),
            "gate_error": int(sum(int(row["pred_id"]) != true_id for row in rows)),
            "gate_error_fraction": (
                float(sum(int(row["pred_id"]) != true_id for row in rows) / len(rows))
                if rows
                else None
            ),
            "mean_probability_CU": (
                float(mean(float(row["probability_cu"]) for row in rows)) if rows else None
            ),
        }
    return result


def _transition_summary(
    baseline: Mapping[str, dict[str, Any]],
    candidate: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, int]] = {
        label: {
            "support": 0,
            "both_correct": 0,
            "corrected_by_V10": 0,
            "regressed_by_V10": 0,
            "both_wrong": 0,
            "prediction_changed": 0,
        }
        for label in BASE_LABELS
    }
    totals = {
        "support": 0,
        "both_correct": 0,
        "corrected_by_V10": 0,
        "regressed_by_V10": 0,
        "both_wrong": 0,
        "prediction_changed": 0,
        "DBB_to_CU": 0,
        "CU_to_DBB": 0,
    }
    for key in sorted(baseline):
        left = baseline[key]
        right = candidate[key]
        truth = int(left["true_id"])
        left_pred = int(left["pred_id"])
        right_pred = int(right["pred_id"])
        left_correct = left_pred == truth
        right_correct = right_pred == truth
        if left_correct and right_correct:
            state = "both_correct"
        elif not left_correct and right_correct:
            state = "corrected_by_V10"
        elif left_correct and not right_correct:
            state = "regressed_by_V10"
        else:
            state = "both_wrong"
        label = str(left["label_name"])
        totals["support"] += 1
        totals[state] += 1
        by_label[label]["support"] += 1
        by_label[label][state] += 1
        changed = left_pred != right_pred
        if changed:
            totals["prediction_changed"] += 1
            by_label[label]["prediction_changed"] += 1
            totals["DBB_to_CU" if left_pred == 0 else "CU_to_DBB"] += 1
            rows.append({
                "record_key": key,
                "parent_sample_id": str(left["parent_sample_id"]),
                "event_index": int(left["event_index"]),
                "window_length": int(left["window_length"]),
                "label_name": label,
                "true_id": truth,
                "V9_pred_id": left_pred,
                "V10_pred_id": right_pred,
                "transition": state,
                "V9_probability_CU": float(left["probability_cu"]),
                "V10_probability_CU": float(right["probability_cu"]),
                "probability_CU_delta": (
                    float(right["probability_cu"]) - float(left["probability_cu"])
                ),
            })
    return {
        "totals": totals,
        "by_base_label": by_label,
        "changed_records": rows,
        "probability_CU_delta_all": _quantiles(
            float(candidate[key]["probability_cu"]) - float(baseline[key]["probability_cu"])
            for key in baseline
        ),
        "probability_CU_delta_changed_only": _quantiles(
            row["probability_CU_delta"] for row in rows
        ),
    }


def _length_group(length: int) -> str:
    if length == 1:
        return "length_1"
    if length <= 4:
        return "length_2_4"
    if length <= 8:
        return "length_5_8"
    if length <= 16:
        return "length_9_16"
    if length <= 24:
        return "length_17_24"
    return "length_25_29"


def _length_groups(
    baseline: Mapping[str, dict[str, Any]],
    candidate: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, record in baseline.items():
        grouped[_length_group(int(record["window_length"]))].append(key)
    result: dict[str, Any] = {}
    for group_name in (
        "length_1",
        "length_2_4",
        "length_5_8",
        "length_9_16",
        "length_17_24",
        "length_25_29",
    ):
        keys = grouped.get(group_name, [])
        left = [baseline[key] for key in keys]
        right = [candidate[key] for key in keys]
        result[group_name] = {
            "metrics": _metric_deltas(left, right, group_name),
            "V9_base_label_routes": _base_label_routes(left),
            "V10_base_label_routes": _base_label_routes(right),
            "transitions": _transition_summary(
                {key: baseline[key] for key in keys},
                {key: candidate[key] for key in keys},
            )["totals"],
        }
    return result


def _assert_matches_evaluation(
    calculated: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    version: str,
) -> None:
    for scope_name, report_name in (("window", "window_metrics"), ("parent", "parent_metrics")):
        metrics = calculated[scope_name][version]
        expected = evaluation[report_name]
        if metrics["confusion_matrix_rows_true_columns_pred"] != expected["confusion_matrix_rows_true_columns_pred"]:
            raise AssertionError(f"{version} {scope_name} paired records do not reproduce evaluation matrix")
        for name in ("accuracy", "macro_precision", "macro_recall", "macro_f1", "balanced_accuracy"):
            if abs(float(metrics[name]) - float(expected[name])) > 1.0e-12:
                raise AssertionError(f"{version} {scope_name} paired records do not reproduce {name}")


def _training_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    history = list(payload.get("history", []))
    return {
        "status": payload.get("status"),
        "device": payload.get("device"),
        "train_parent_count": payload.get("train_parent_count"),
        "validation_parent_count": payload.get("validation_parent_count"),
        "best_epoch": payload.get("best_epoch"),
        "global_step": payload.get("global_step"),
        "stop_reason": payload.get("stop_reason"),
        "best_selection_metric_value": payload.get("best_selection_metric_value"),
        "best_parent_macro_f1": (
            payload.get("best_validation_metrics", {}).get("parent_metrics", {}).get("macro_f1")
        ),
        "best_window_macro_f1": (
            payload.get("best_validation_metrics", {}).get("window_metrics", {}).get("macro_f1")
        ),
        "epochs_executed": len(history),
        "final_train_loss": history[-1].get("train_loss") if history else None,
    }


def _markdown(result: Mapping[str, Any]) -> str:
    parent = result["metrics"]["parent"]
    window = result["metrics"]["window"]
    lines = [
        "# RADAR-LN3 V9/V10 Gate Paired Validation",
        "",
        "范围：CQ08 V3 validation，232 parent / 4474 causal-prefix window；test、stable、reject和正式级联均未执行。",
        "",
        "|层级|版本|Accuracy|Macro-F1|DBB P/R/F1|CU P/R/F1|矩阵|",
        "|---|---|---:|---:|---|---|---|",
    ]
    for scope_name, metrics in (("Parent", parent), ("Window", window)):
        for version in ("V9", "V10"):
            row = metrics[version]
            dbb = next(item for item in row["per_class"] if item["class"] == "DBB")
            cu = next(item for item in row["per_class"] if item["class"] == "CU")
            lines.append(
                f"|{scope_name}|{version}|{row['accuracy']:.6f}|{row['macro_f1']:.6f}|"
                f"{dbb['precision']:.6f}/{dbb['recall']:.6f}/{dbb['f1']:.6f}|"
                f"{cu['precision']:.6f}/{cu['recall']:.6f}/{cu['f1']:.6f}|"
                f"`{row['confusion_matrix_rows_true_columns_pred']}`|"
            )
    lines.extend([
        "",
        "## 基础五类路由误差",
        "",
        "|层级|类别|Support|V9错误|V10错误|变化|",
        "|---|---|---:|---:|---:|---:|",
    ])
    for scope_name in ("parent", "window"):
        left = result["base_label_routes"][scope_name]["V9"]
        right = result["base_label_routes"][scope_name]["V10"]
        for label in BASE_LABELS:
            lines.append(
                f"|{scope_name}|{label}|{left[label]['support']}|{left[label]['gate_error']}|"
                f"{right[label]['gate_error']}|{right[label]['gate_error'] - left[label]['gate_error']:+d}|"
            )
    parent_transitions = result["transitions"]["parent"]["totals"]
    window_transitions = result["transitions"]["window"]["totals"]
    lines.extend([
        "",
        "## Paired变化",
        "",
        f"- Parent：纠正{parent_transitions['corrected_by_V10']}，退化{parent_transitions['regressed_by_V10']}，预测变化{parent_transitions['prediction_changed']}。",
        f"- Window：纠正{window_transitions['corrected_by_V10']}，退化{window_transitions['regressed_by_V10']}，预测变化{window_transitions['prediction_changed']}。",
        "- V10当前只可作为Gate研究候选；CU P/R未达到0.90，因此不能进入正式软级联。",
        "- UnknownTarget仍是闭集标签，开放集reject/abstain未校准、未评估。",
        "",
    ])
    return "\n".join(lines)


def run(
    project_root: Path,
    *,
    baseline_detail_path: Path,
    candidate_detail_path: Path,
    baseline_evaluation_path: Path,
    candidate_evaluation_path: Path,
    baseline_training_path: Path,
    candidate_training_path: Path,
    baseline_checkpoint_path: Path,
    candidate_checkpoint_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    paths = {
        "V9_detail": baseline_detail_path,
        "V10_detail": candidate_detail_path,
        "V9_evaluation": baseline_evaluation_path,
        "V10_evaluation": candidate_evaluation_path,
        "V9_training": baseline_training_path,
        "V10_training": candidate_training_path,
        "V9_checkpoint": baseline_checkpoint_path,
        "V10_checkpoint": candidate_checkpoint_path,
    }
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"paired Gate input is missing ({name}): {path}")

    baseline_detail = _load_json(baseline_detail_path)
    candidate_detail = _load_json(candidate_detail_path)
    baseline_evaluation = _load_json(baseline_evaluation_path)
    candidate_evaluation = _load_json(candidate_evaluation_path)
    baseline_training = _load_json(baseline_training_path)
    candidate_training = _load_json(candidate_training_path)
    if baseline_detail.get("split") != "val" or candidate_detail.get("split") != "val":
        raise ValueError("paired Gate comparison is restricted to validation details")
    if baseline_detail.get("task_name") != "dbb_cu_gate" or candidate_detail.get("task_name") != "dbb_cu_gate":
        raise ValueError("paired Gate details belong to a different task")
    if Path(str(baseline_detail.get("checkpoint"))).resolve() != baseline_checkpoint_path.resolve():
        raise ValueError("V9 detail checkpoint path does not match the paired input")
    if Path(str(candidate_detail.get("checkpoint"))).resolve() != candidate_checkpoint_path.resolve():
        raise ValueError("V10 detail checkpoint path does not match the paired input")

    paired: dict[str, Any] = {}
    transitions: dict[str, Any] = {}
    routes: dict[str, Any] = {}
    indexed_by_scope: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    for scope in ("window", "parent"):
        left = _index_records(baseline_detail["records"][scope], scope)
        right = _index_records(candidate_detail["records"][scope], scope)
        _validate_pair(left, right, scope)
        indexed_by_scope[scope] = (left, right)
        paired[scope] = _metric_deltas(list(left.values()), list(right.values()), scope)
        transitions[scope] = _transition_summary(left, right)
        routes[scope] = {
            "V9": _base_label_routes(left.values()),
            "V10": _base_label_routes(right.values()),
        }

    _assert_matches_evaluation(paired, baseline_evaluation, "V9")
    _assert_matches_evaluation(paired, candidate_evaluation, "V10")
    window_left, window_right = indexed_by_scope["window"]
    result: dict[str, Any] = {
        "status": "passed",
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "CQ08 V3 validation only; paired 232 parents and 4474 causal-prefix windows; "
            "test, stable, rejection, multi-seed and formal cascade excluded"
        ),
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()
        },
        "training": {
            "V9": _training_summary(baseline_training),
            "V10": _training_summary(candidate_training),
        },
        "record_contract": {
            "window_count": len(window_left),
            "parent_count": len(indexed_by_scope["parent"][0]),
            "key_sets_equal": True,
            "truth_and_metadata_equal": True,
            "evaluation_reports_reproduced_exactly": True,
        },
        "metrics": paired,
        "base_label_routes": routes,
        "transitions": transitions,
        "prefix_length_groups": _length_groups(window_left, window_right),
        "acceptance": {
            "required_parent_DBB_precision": 0.95,
            "required_parent_DBB_recall": 0.95,
            "required_parent_CU_precision": 0.90,
            "required_parent_CU_recall": 0.90,
            "V10_DBB_passed": (
                paired["parent"]["per_class"]["DBB"]["V10"]["precision"] >= 0.95
                and paired["parent"]["per_class"]["DBB"]["V10"]["recall"] >= 0.95
            ),
            "V10_CU_passed": (
                paired["parent"]["per_class"]["CU"]["V10"]["precision"] >= 0.90
                and paired["parent"]["per_class"]["CU"]["V10"]["recall"] >= 0.90
            ),
            "formal_gate_module_release": False,
            "reason": (
                "V10 improves the paired single-seed validation Gate but CU precision/recall remain below 0.90; "
                "validation has only 23 CU parents and no multi-seed/test evidence"
            ),
        },
        "unknown_semantics": {
            "closed_set_UnknownTarget": True,
            "reject_or_abstain_calibrated": False,
            "open_set_evaluation": "未执行",
        },
        "test": "未执行",
        "stable": "未执行",
        "formal_cascade": "未执行",
        "multi_seed": "未执行",
        "p95": "未执行",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["output_path"] = str(output_path)
    markdown_path = output_path.with_suffix(".md")
    result["markdown_path"] = str(markdown_path)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare V9/V10 DBB/CU Gate validation records one-to-one.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--baseline-detail", type=Path, default=DEFAULT_BASELINE_DETAIL)
    parser.add_argument("--candidate-detail", type=Path, default=DEFAULT_CANDIDATE_DETAIL)
    parser.add_argument("--baseline-evaluation", type=Path, default=DEFAULT_BASELINE_EVALUATION)
    parser.add_argument("--candidate-evaluation", type=Path, default=DEFAULT_CANDIDATE_EVALUATION)
    parser.add_argument("--baseline-training", type=Path, default=DEFAULT_BASELINE_TRAINING)
    parser.add_argument("--candidate-training", type=Path, default=DEFAULT_CANDIDATE_TRAINING)
    parser.add_argument("--baseline-checkpoint", type=Path, default=DEFAULT_BASELINE_CHECKPOINT)
    parser.add_argument("--candidate-checkpoint", type=Path, default=DEFAULT_CANDIDATE_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    result = run(
        root,
        baseline_detail_path=_resolve(root, args.baseline_detail),
        candidate_detail_path=_resolve(root, args.candidate_detail),
        baseline_evaluation_path=_resolve(root, args.baseline_evaluation),
        candidate_evaluation_path=_resolve(root, args.candidate_evaluation),
        baseline_training_path=_resolve(root, args.baseline_training),
        candidate_training_path=_resolve(root, args.candidate_training),
        baseline_checkpoint_path=_resolve(root, args.baseline_checkpoint),
        candidate_checkpoint_path=_resolve(root, args.candidate_checkpoint),
        output_path=_resolve(root, args.output),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
