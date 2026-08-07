from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReportPaths:
    markdown_path: Path
    json_path: Path


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _format_optional_metric(value: object) -> str:
    """Format optional branch metrics without turning TR-only N/A into zero."""
    if value is None:
        return "not_applicable_tr_only"
    try:
        return _format_float(float(value))
    except (TypeError, ValueError):
        return str(value)


def _escape_md_cell(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _top_confusions(label_names: list[str], matrix: list[list[int]], top_k: int = 10) -> list[tuple[str, str, int]]:
    items: list[tuple[str, str, int]] = []
    for true_idx, row in enumerate(matrix):
        for pred_idx, count in enumerate(row):
            if true_idx == pred_idx:
                continue
            count_int = int(count)
            if count_int <= 0:
                continue
            items.append((label_names[true_idx], label_names[pred_idx], count_int))
    items.sort(key=lambda item: item[2], reverse=True)
    return items[:top_k]


def _label_at(label_names: list[str], index: int) -> str:
    if 0 <= index < len(label_names):
        return str(label_names[index])
    return f"class_{index}"


def _append_confusion_matrix(
    lines: list[str],
    label_names: list[str],
    title: str,
    matrix: object,
) -> None:
    if not isinstance(matrix, list) or not matrix:
        return
    width = max(len(matrix), max((len(row) for row in matrix if isinstance(row, list)), default=0))
    if width <= 0:
        return
    lines.append(f"#### {title}")
    header_labels = [_escape_md_cell(_label_at(label_names, index)) for index in range(width)]
    lines.append("| true \\ pred | " + " | ".join(header_labels) + " |")
    lines.append("|---|" + "|".join("---:" for _ in header_labels) + "|")
    for row_index in range(width):
        row = matrix[row_index] if row_index < len(matrix) and isinstance(matrix[row_index], list) else []
        values = [int(row[col_index]) if col_index < len(row) else 0 for col_index in range(width)]
        lines.append(f"| {_escape_md_cell(_label_at(label_names, row_index))} | " + " | ".join(str(value) for value in values) + " |")
    lines.append("")


def _append_transition_table(
    lines: list[str],
    title: str,
    transitions: dict[str, object],
    specs: list[tuple[str, str]],
) -> None:
    rows: list[tuple[str, dict[str, object]]] = []
    for key, label in specs:
        item = transitions.get(key, {}) if isinstance(transitions, dict) else {}
        if isinstance(item, dict) and item:
            rows.append((label, item))
    if not rows:
        return
    lines.append(title)
    lines.append(
        "| comparison | support | baseline_acc | candidate_acc | rescue | harm | net_gain | net_gain_rate | changed_rate |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for label, item in rows:
        lines.append(
            f"| {_escape_md_cell(label)} | {item.get('support', 0)} | "
            f"{_format_float(float(item.get('baseline_accuracy', 0.0)))} | "
            f"{_format_float(float(item.get('candidate_accuracy', 0.0)))} | "
            f"{item.get('rescue_count', 0)} | {item.get('harm_count', 0)} | "
            f"{item.get('net_gain_count', 0)} | "
            f"{_format_float(float(item.get('net_gain_rate', 0.0)))} | "
            f"{_format_float(float(item.get('changed_rate', 0.0)))} |"
        )
    lines.append("")


def _precision_view_value(item: dict[str, object], key: str) -> object:
    if key == "scale":
        return item.get(
            "conservative_precision_scale",
            item.get("scale", item.get("selected_joint_logit_delta_scale", 0.0)),
        )
    return item.get(f"conservative_precision_{key}", item.get(key, 0.0))


def build_markdown_report(evaluation_report: dict[str, object], summary: dict[str, object], title: str) -> str:
    report = evaluation_report
    summary_metrics = report["summary"]
    label_names = report["label_names"]
    confusion = report["classification_confusion_matrix"]
    per_class = report["per_class"]
    route_report = report["route"]
    by_source = report["by_source"]
    predictions = report["predictions"]
    critical_metrics = report.get("critical_metrics", {})
    test_metrics = summary.get("test_metrics", {}) if isinstance(summary, dict) else {}
    checkpoint_selection = report.get("checkpoint_selection", {})
    diagnostic_checkpoint_tests = report.get("diagnostic_checkpoint_tests", {})
    joint_logit_delta_probe = report.get("joint_logit_delta_probe", {})
    rd_time_sync_metrics = report.get("rd_time_sync_metrics", {})
    summary_precision_checkpoint = str(summary.get("precision_checkpoint", ""))
    summary_precision_checkpoint_name = str(summary.get("precision_checkpoint_name", ""))
    summary_precision_checkpoint_export = summary.get("precision_checkpoint_export", {})
    summary_precision_checkpoint_validation = summary.get("precision_checkpoint_validation", {})

    top_confusions = _top_confusions(label_names, confusion, top_k=10)
    wrong_predictions = [item for item in predictions if not item["is_correct"]]
    wrong_predictions = wrong_predictions[:20]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    evaluation_scope = str(report.get("evaluation_scope", "test"))
    if evaluation_scope == "validation_only":
        lines.append("- evaluation_scope: validation_only（本次未运行 test 推理）")
    else:
        if summary_metrics.get("rd_metrics_applicable") is False:
            lines.append("- rd_metrics: not_applicable_tr_only（纯 TR 路径未计算 RD 性能）")
        lines.append(f"- evaluation_scope: {evaluation_scope}")
    lines.append("")
    lines.append("## 总体指标")
    classification_metrics_status = str(
        report.get("classification_metrics_status", "evaluated")
    )
    if classification_metrics_status == "not_applicable_pretraining":
        lines.append("- classification_metrics_status: not_applicable_pretraining（未执行分类评估）")
        lines.append("- accuracy: 未执行")
        lines.append("- macro_precision: 未执行")
        lines.append("- macro_recall: 未执行")
        lines.append("- macro_f1: 未执行")
        lines.append("- route_accuracy: 未执行")
    else:
        lines.append(f"- accuracy: {_format_float(float(summary_metrics['accuracy']))}")
        lines.append(f"- macro_precision: {_format_float(float(summary_metrics['macro_precision']))}")
        lines.append(f"- macro_recall: {_format_float(float(summary_metrics['macro_recall']))}")
        lines.append(f"- macro_f1: {_format_float(float(summary_metrics['macro_f1']))}")
        lines.append(f"- route_accuracy: {_format_float(float(summary_metrics['route_accuracy']))}")
    if "selected_joint_logit_delta_scale" in summary_metrics:
        lines.append(f"- selected_joint_logit_delta_scale: {_format_float(float(summary_metrics['selected_joint_logit_delta_scale']))}")
    if "route_macro_recall" in summary_metrics:
        lines.append(f"- route_macro_recall: {_format_float(float(summary_metrics['route_macro_recall']))}")
    if "selected_fusion_mode" in report:
        lines.append(f"- selected_fusion_mode: {report['selected_fusion_mode']}")
    reroute_comparison = report.get("reroute_comparison", {})
    if isinstance(reroute_comparison, dict) and reroute_comparison:
        lines.append(
            f"- reroute_delta: accuracy={_format_float(float(reroute_comparison.get('accuracy_delta', 0.0)))}, "
            f"macro_f1={_format_float(float(reroute_comparison.get('macro_f1_delta', 0.0)))}"
        )
    reroute_config = report.get("reroute_config", {})
    lines.append(f"- loss: {_format_float(float(summary_metrics['loss']))}")
    if "skipped_non_finite_batches" in summary_metrics:
        lines.append(f"- skipped_non_finite_batches: {summary_metrics['skipped_non_finite_batches']}")
    lines.append("")
    if "device" in report:
        device_info = report["device"]
        lines.append("## 设备信息")
        lines.append(f"- requested_device: {device_info.get('requested_device')}")
        lines.append(f"- resolved_device: {device_info.get('resolved_device')}")
        lines.append(f"- cuda_available: {device_info.get('cuda_available')}")
        if "cuda_device_name" in device_info:
            lines.append(f"- cuda_device_name: {device_info.get('cuda_device_name')}")
    lines.append("")
    window_evaluation = report.get("window_evaluation", {})
    if isinstance(window_evaluation, dict) and window_evaluation.get("is_window_index"):
        parent_aggregation = window_evaluation.get("parent_aggregation", {})
        if isinstance(parent_aggregation, dict) and parent_aggregation:
            mean_logit = parent_aggregation.get("mean_logit", {})
            majority_vote = parent_aggregation.get("majority_vote", {})
            early_window = parent_aggregation.get("early_window", {})
            stable_decision = parent_aggregation.get("stable_decision", {})
            lines.append("## 窗口与父轨迹聚合")
            lines.append(f"- index_name: {window_evaluation.get('index_name')}")
            lines.append(f"- parent_count: {parent_aggregation.get('parent_count', 0)}")
            lines.append(f"- window_count: {parent_aggregation.get('window_count', 0)}")
            lines.append(f"- mean_windows_per_parent: {_format_float(float(parent_aggregation.get('mean_windows_per_parent', 0.0)))}")
            lines.append("| 口径 | accuracy | macro_precision | macro_recall | macro_f1 |")
            lines.append("|---|---:|---:|---:|---:|")
            for item, label in [
                (mean_logit, "parent mean-logit"),
                (majority_vote, "parent majority-vote"),
                (early_window, "first window"),
                (stable_decision, "stable decision"),
            ]:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"| {label} | {_format_float(float(item.get('accuracy', 0.0)))} | "
                    f"{_format_float(float(item.get('macro_precision', 0.0)))} | "
                    f"{_format_float(float(item.get('macro_recall', 0.0)))} | "
                    f"{_format_float(float(item.get('macro_f1', 0.0)))} |"
                )
            if isinstance(stable_decision, dict) and stable_decision:
                lines.append(
                    f"- stable_decision: accuracy={_format_float(float(stable_decision.get('accuracy', 0.0)))}, "
                    f"latency_mean={_format_float(float(stable_decision.get('latency_seconds_mean', 0.0)))}s, "
                    f"latency_p50={_format_float(float(stable_decision.get('latency_seconds_p50', 0.0)))}s, "
                    f"latency_p90={_format_float(float(stable_decision.get('latency_seconds_p90', 0.0)))}s"
                )
            flip_diagnostics = parent_aggregation.get("prediction_flip_diagnostics", {})
            if isinstance(flip_diagnostics, dict) and flip_diagnostics:
                lines.append("")
                lines.append("### Prediction Flip Diagnostics")
                lines.append(
                    f"- flipped_parent_count: {flip_diagnostics.get('flipped_parent_count', 0)} / "
                    f"{flip_diagnostics.get('parent_count', parent_aggregation.get('parent_count', 0))} "
                    f"({_format_float(float(flip_diagnostics.get('flipped_parent_rate', 0.0)))})"
                )
                lines.append(
                    f"- flip_count: mean={_format_float(float(flip_diagnostics.get('flip_count_mean', 0.0)))}, "
                    f"p50={_format_float(float(flip_diagnostics.get('flip_count_p50', 0.0)))}, "
                    f"p90={_format_float(float(flip_diagnostics.get('flip_count_p90', 0.0)))}, "
                    f"max={flip_diagnostics.get('flip_count_max', 0)}"
                )
                lines.append(
                    f"- flip_rate: mean={_format_float(float(flip_diagnostics.get('flip_rate_mean', 0.0)))}, "
                    f"p50={_format_float(float(flip_diagnostics.get('flip_rate_p50', 0.0)))}, "
                    f"p90={_format_float(float(flip_diagnostics.get('flip_rate_p90', 0.0)))}, "
                    f"max={_format_float(float(flip_diagnostics.get('flip_rate_max', 0.0)))}"
                )
            weak_class_recall = parent_aggregation.get("weak_class_recall", {})
            if isinstance(weak_class_recall, dict) and weak_class_recall:
                lines.append("")
                lines.append("### Parent Weak-Class Recall")
                lines.append("| 口径 | Unknown support | Unknown recall | Clutter support | Clutter recall | mean recall |")
                lines.append("|---|---:|---:|---:|---:|---:|")
                for key, label in [
                    ("mean_logit", "parent mean-logit"),
                    ("majority_vote", "parent majority-vote"),
                    ("early_window", "first window"),
                    ("stable_decision", "stable decision"),
                ]:
                    item = weak_class_recall.get(key, {})
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {label} | {item.get('unknown_support', 0)} | "
                        f"{_format_float(float(item.get('unknown_recall', 0.0)))} | "
                        f"{item.get('clutter_support', 0)} | "
                        f"{_format_float(float(item.get('clutter_recall', 0.0)))} | "
                        f"{_format_float(float(item.get('unknown_clutter_recall_mean', 0.0)))} |"
                    )
            hard_pair = parent_aggregation.get("unknown_clutter_hard_pair", {})
            if isinstance(hard_pair, dict) and hard_pair:
                lines.append("")
                _append_transition_table(
                    lines,
                    "### Unknown/Clutter hard-pair rescue/harm",
                    hard_pair,
                    [
                        ("window_to_parent_mean_logit", "window -> parent mean-logit"),
                        ("parent_first_to_mean_logit", "parent first -> mean-logit"),
                        ("parent_majority_vote_to_mean_logit", "parent vote -> mean-logit"),
                        ("parent_stable_to_mean_logit", "parent stable -> mean-logit"),
                    ],
                )
            parent_rescue_harm = parent_aggregation.get("parent_unknown_clutter_rescue_harm", {})
            if isinstance(parent_rescue_harm, dict) and parent_rescue_harm:
                _append_transition_table(
                    lines,
                    "### Parent Unknown/Clutter rescue/harm",
                    parent_rescue_harm,
                    [
                        ("first_window_to_mean_logit", "first window -> mean-logit"),
                        ("majority_vote_to_mean_logit", "majority vote -> mean-logit"),
                        ("stable_decision_to_mean_logit", "stable decision -> mean-logit"),
                    ],
                )
            top_flip_parent_records = parent_aggregation.get("top_flip_parent_records", [])
            if isinstance(top_flip_parent_records, list) and top_flip_parent_records:
                lines.append("### Top Flipping Parents")
                lines.append(
                    "| parent | source | true | windows | flips | flip_rate | first | stable | vote | mean-logit | mean_correct |"
                )
                lines.append("|---|---|---|---:|---:|---:|---|---|---|---|---:|")
                for item in top_flip_parent_records[:10]:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {_escape_md_cell(item.get('parent_sample_id', ''))} | {_escape_md_cell(item.get('source_name', ''))} | "
                        f"{_escape_md_cell(item.get('true_label', ''))} | {item.get('window_count', 0)} | "
                        f"{item.get('prediction_flip_count', 0)} | "
                        f"{_format_float(float(item.get('prediction_flip_rate', 0.0)))} | "
                        f"{_escape_md_cell(item.get('first_window_pred', ''))} | {_escape_md_cell(item.get('stable_decision_pred', ''))} | "
                        f"{_escape_md_cell(item.get('majority_vote_pred', ''))} | {_escape_md_cell(item.get('mean_logit_pred', ''))} | "
                        f"{int(bool(item.get('mean_logit_correct', False)))} |"
                    )
                lines.append("")
            hard_pair_parent_records = parent_aggregation.get("hard_pair_parent_records", [])
            if isinstance(hard_pair_parent_records, list) and hard_pair_parent_records:
                lines.append("### Unknown/Clutter Parent Samples")
                lines.append("| parent | source | true | windows | flips | first | stable | vote | mean-logit | mean_correct |")
                lines.append("|---|---|---|---:|---:|---|---|---|---|---:|")
                for item in hard_pair_parent_records[:10]:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {_escape_md_cell(item.get('parent_sample_id', ''))} | {_escape_md_cell(item.get('source_name', ''))} | "
                        f"{_escape_md_cell(item.get('true_label', ''))} | {item.get('window_count', 0)} | "
                        f"{item.get('prediction_flip_count', 0)} | {_escape_md_cell(item.get('first_window_pred', ''))} | "
                        f"{_escape_md_cell(item.get('stable_decision_pred', ''))} | {_escape_md_cell(item.get('majority_vote_pred', ''))} | "
                        f"{_escape_md_cell(item.get('mean_logit_pred', ''))} | {int(bool(item.get('mean_logit_correct', False)))} |"
                    )
                lines.append("")
            if any(
                isinstance(item, dict) and item.get("confusion_matrix")
                for item in [mean_logit, majority_vote, early_window, stable_decision]
            ):
                lines.append("### Parent Confusion Matrices")
                _append_confusion_matrix(lines, label_names, "parent mean-logit", mean_logit.get("confusion_matrix") if isinstance(mean_logit, dict) else [])
                _append_confusion_matrix(lines, label_names, "parent majority-vote", majority_vote.get("confusion_matrix") if isinstance(majority_vote, dict) else [])
                _append_confusion_matrix(lines, label_names, "first window", early_window.get("confusion_matrix") if isinstance(early_window, dict) else [])
                _append_confusion_matrix(lines, label_names, "stable decision", stable_decision.get("confusion_matrix") if isinstance(stable_decision, dict) else [])
            by_source_parent = parent_aggregation.get("by_source", {})
            if isinstance(by_source_parent, dict) and by_source_parent:
                lines.append("")
                lines.append("| parent source | count | mean-logit accuracy | mean-logit macro_f1 |")
                lines.append("|---|---:|---:|---:|")
                for source_name, item in by_source_parent.items():
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {_escape_md_cell(source_name)} | {item.get('count', 0)} | "
                        f"{_format_float(float(item.get('mean_logit_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('mean_logit_macro_f1', 0.0)))} |"
                    )
            lines.append("")
    if isinstance(rd_time_sync_metrics, dict) and rd_time_sync_metrics:
        lines.append("## RD Time Sync Diagnostics")
        lines.append("| status | support | macro_f1 | traj_macro_f1 | joint_macro_f1 | joint_gain | RD evidence macro_f1 | accuracy |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for status, item in sorted(rd_time_sync_metrics.items(), key=lambda pair: str(pair[0])):
            if not isinstance(item, dict):
                continue
            lines.append(
                f"| {_escape_md_cell(status)} | {item.get('support', 0)} | "
                f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
                f"{_format_float(float(item.get('traj_macro_f1', 0.0)))} | "
                f"{_format_float(float(item.get('joint_macro_f1', 0.0)))} | "
                f"{_format_float(float(item.get('joint_minus_traj_macro_f1', 0.0)))} | "
                f"{_format_float(float(item.get('rd_macro_f1', 0.0)))} | "
                f"{_format_float(float(item.get('accuracy', 0.0)))} |"
            )
        lines.append("")
    lines.append("## 训练产物")
    for label, key in [
        ("best_checkpoint", "best_checkpoint"),
        ("precision_checkpoint", "precision_checkpoint"),
        ("precision_checkpoint_name", "precision_checkpoint_name"),
        ("history_file", "history_file"),
        ("manifest_file", "manifest_file"),
        ("evaluation_report", "evaluation_report"),
    ]:
        if key in summary:
            lines.append(f"- {label}: {summary[key]}")
    lines.append("")
    if isinstance(checkpoint_selection, dict) and checkpoint_selection:
        lines.append("## Checkpoint 选择")
        lines.append(f"- candidate_status: {checkpoint_selection.get('candidate_status', '')}")
        lines.append(f"- selected_checkpoint: {checkpoint_selection.get('selected_checkpoint', '')}")
        if "selected_checkpoint_name" in checkpoint_selection:
            lines.append(f"- selected_checkpoint_name: {checkpoint_selection.get('selected_checkpoint_name', '')}")
        if "selected_checkpoint_role" in checkpoint_selection:
            lines.append(f"- selected_checkpoint_role: {checkpoint_selection.get('selected_checkpoint_role', '')}")
        if "selected_checkpoint_pool" in checkpoint_selection:
            lines.append(f"- selected_checkpoint_pool: {checkpoint_selection.get('selected_checkpoint_pool', '')}")
        precision_checkpoint = str(
            checkpoint_selection.get("precision_checkpoint")
            or summary_precision_checkpoint
            or ""
        )
        precision_checkpoint_name = str(
            checkpoint_selection.get("precision_checkpoint_name")
            or summary_precision_checkpoint_name
            or ""
        )
        if not precision_checkpoint and isinstance(checkpoint_selection, dict):
            precision_checkpoint = str(checkpoint_selection.get("precision_candidate_metrics", {}).get("path", ""))
        if not precision_checkpoint_name and isinstance(checkpoint_selection, dict):
            precision_checkpoint_name = str(checkpoint_selection.get("precision_candidate_metrics", {}).get("checkpoint_name", ""))
        if precision_checkpoint:
            lines.append(f"- precision_checkpoint: {precision_checkpoint}")
        if precision_checkpoint_name:
            lines.append(f"- precision_checkpoint_name: {precision_checkpoint_name}")
        precision_export = checkpoint_selection.get("precision_checkpoint_export", summary_precision_checkpoint_export)
        if isinstance(precision_export, dict) and precision_export:
            if precision_export.get("source_path"):
                lines.append(f"- precision_checkpoint_source: {precision_export.get('source_path')}")
            if precision_export.get("recommended_joint_logit_delta_scale") is not None:
                lines.append(
                    "- precision_checkpoint_joint_logit_delta_scale: "
                    f"{_format_float(float(precision_export.get('recommended_joint_logit_delta_scale', 0.0)))}"
                )
        precision_validation = checkpoint_selection.get(
            "precision_checkpoint_validation",
            summary_precision_checkpoint_validation,
        )
        if isinstance(precision_validation, dict) and precision_validation:
            lines.append(f"- precision_checkpoint_validation_status: {precision_validation.get('status', '')}")
            if precision_validation.get("selected_joint_logit_delta_scale") is not None:
                lines.append(
                    "- precision_checkpoint_validation_scale: "
                    f"{_format_float(float(precision_validation.get('selected_joint_logit_delta_scale', 0.0)))}"
                )
            if precision_validation.get("macro_f1") is not None:
                lines.append(
                    "- precision_checkpoint_validation_macro_f1: "
                    f"{_format_float(float(precision_validation.get('macro_f1', 0.0)))}"
                )
            if precision_validation.get("macro_f1_delta") is not None:
                lines.append(
                    "- precision_checkpoint_validation_macro_f1_delta: "
                    f"{_format_float(float(precision_validation.get('macro_f1_delta', 0.0)))}"
                )
        lines.append(f"- selected_reason: {checkpoint_selection.get('selected_reason', '')}")
        if "selected_checkpoint_eligible" in checkpoint_selection:
            lines.append(f"- selected_checkpoint_eligible: {checkpoint_selection.get('selected_checkpoint_eligible')}")
        if "release_eligible" in checkpoint_selection:
            lines.append(f"- release_eligible: {checkpoint_selection.get('release_eligible')}")
        release_failures = checkpoint_selection.get("release_gate_failures", [])
        if isinstance(release_failures, list) and release_failures:
            lines.append(f"- release_gate_failures: {', '.join(str(item) for item in release_failures)}")
        lines.append(f"- best_epoch: {checkpoint_selection.get('best_epoch', -1)}")
        release_metrics = checkpoint_selection.get("release_gate_metrics", {})
        if isinstance(release_metrics, dict) and release_metrics:
            lines.append("")
            lines.append("### Release Gate")
            lines.append("| metric | value |")
            lines.append("|---|---:|")
            for key in [
                "release_macro_f1",
                "release_final_minus_traj_macro_f1",
                "release_joint_minus_traj_macro_f1_on_rd",
                "release_rd_macro_f1_on_rd",
                "release_final_net_gain_rate_on_rd",
                "release_final_harm_rate_on_rd",
                "release_parent_macro_f1",
                "release_final_unknown_recall",
                "release_final_clutter_recall",
                "release_without_rd_accuracy",
                "release_hard_class_gate_passed",
            ]:
                if key in release_metrics:
                    value = release_metrics.get(key)
                    if isinstance(value, bool):
                        lines.append(f"| {key} | {value} |")
                    else:
                        lines.append(f"| {key} | {_format_float(float(value or 0.0))} |")
            hard_failures = release_metrics.get("release_hard_class_gate_failures", [])
            if isinstance(hard_failures, list) and hard_failures:
                lines.append(f"- release_hard_class_gate_failures: {', '.join(str(item) for item in hard_failures)}")
        diagnostic_results = []
        diagnostic_recommendation = {}
        precision_candidate = {}
        precision_readiness = {}
        precision_delta = {}
        if isinstance(diagnostic_checkpoint_tests, dict):
            recommendation = diagnostic_checkpoint_tests.get("diagnostic_test_recommendation", {})
            if isinstance(recommendation, dict):
                diagnostic_recommendation = recommendation
            precision_item = diagnostic_checkpoint_tests.get("precision_candidate", {})
            if isinstance(precision_item, dict):
                precision_candidate = precision_item
            readiness_item = diagnostic_checkpoint_tests.get("precision_candidate_readiness", {})
            if isinstance(readiness_item, dict):
                precision_readiness = readiness_item
            delta_item = diagnostic_checkpoint_tests.get("precision_candidate_delta_vs_selected", {})
            if isinstance(delta_item, dict):
                precision_delta = delta_item
            candidate = diagnostic_checkpoint_tests.get("results", [])
            if isinstance(candidate, list):
                diagnostic_results = candidate
        if not precision_candidate:
            precision_item = checkpoint_selection.get("precision_candidate_metrics", {})
            if isinstance(precision_item, dict):
                precision_candidate = precision_item
        if not precision_candidate and isinstance(checkpoint_selection, dict):
            precision_item = checkpoint_selection.get("precision_checkpoint_metrics", {})
            if isinstance(precision_item, dict):
                precision_candidate = precision_item
        if not precision_delta:
            delta_item = checkpoint_selection.get("precision_candidate_delta_vs_selected", {})
            if isinstance(delta_item, dict):
                precision_delta = delta_item
        if not precision_readiness:
            readiness_item = checkpoint_selection.get("precision_candidate_readiness", {})
            if isinstance(readiness_item, dict):
                precision_readiness = readiness_item
        if not precision_readiness:
            readiness_item = report.get("precision_candidate_readiness", {})
            if isinstance(readiness_item, dict):
                precision_readiness = readiness_item
        if not precision_readiness:
            readiness_item = summary.get("precision_candidate_readiness", {})
            if isinstance(readiness_item, dict):
                precision_readiness = readiness_item
        rd_joint_bottleneck = {}
        for candidate in [
            precision_validation.get("rd_joint_bottleneck_summary", {}) if isinstance(precision_validation, dict) else {},
            precision_candidate.get("rd_joint_bottleneck_summary", {}) if isinstance(precision_candidate, dict) else {},
            diagnostic_checkpoint_tests.get("rd_joint_bottleneck_summary", {}) if isinstance(diagnostic_checkpoint_tests, dict) else {},
            checkpoint_selection.get("rd_joint_bottleneck_summary", {}) if isinstance(checkpoint_selection, dict) else {},
            report.get("rd_joint_bottleneck_summary", {}),
            summary.get("rd_joint_bottleneck_summary", {}),
        ]:
            if isinstance(candidate, dict) and candidate:
                rd_joint_bottleneck = candidate
                break
        if precision_candidate:
            lines.append("")
            lines.append("### Precision Candidate")
            lines.append(f"- checkpoint_name: {precision_candidate.get('checkpoint_name', '')}")
            lines.append(f"- epoch: {precision_candidate.get('epoch', -1)}")
            lines.append(f"- basis: {precision_candidate.get('recommendation_basis', '')}")
            lines.append(f"- is_selected_checkpoint: {precision_candidate.get('is_selected_checkpoint', False)}")
            if "recommended_joint_logit_delta_scale" in precision_candidate:
                lines.append(
                    f"- recommended_joint_logit_delta_scale: "
                    f"{_format_float(float(precision_candidate.get('recommended_joint_logit_delta_scale', 0.0)))}"
                )
            failures = precision_candidate.get("precision_threshold_failures", [])
            if isinstance(failures, list) and failures:
                lines.append(f"- precision_threshold_failures: {', '.join(str(item) for item in failures)}")
            threshold_gaps = precision_candidate.get("precision_threshold_gaps", {})
            if not isinstance(threshold_gaps, dict):
                threshold_gaps = {
                    key: precision_candidate.get(key)
                    for key in [
                        "precision_candidate_unknown_recall_gap",
                        "precision_candidate_clutter_recall_gap",
                        "precision_candidate_joint_gain_gap",
                    ]
                    if key in precision_candidate
                }
            if threshold_gaps:
                lines.append("- precision_threshold_gaps:")
                lines.append("| gap | value |")
                lines.append("|---|---:|")
                for key in [
                    "precision_candidate_unknown_recall_gap",
                    "precision_candidate_clutter_recall_gap",
                    "precision_candidate_joint_gain_gap",
                    ]:
                    if key in threshold_gaps:
                        lines.append(f"| {key} | {_format_float(float(threshold_gaps.get(key, 0.0)))} |")
            calibration_probe = precision_candidate.get("precision_calibration_probe", {})
            if isinstance(calibration_probe, dict) and calibration_probe:
                recommended = calibration_probe.get("recommended", {})
                lines.append("")
                lines.append("### Precision Calibration Probe")
                lines.append(f"- status: {calibration_probe.get('status', '')}")
                lines.append(f"- ready: {calibration_probe.get('ready', False)}")
                if isinstance(recommended, dict) and recommended:
                    bias_by_label = recommended.get("precision_logit_bias_by_label", {})
                    if isinstance(bias_by_label, dict) and bias_by_label:
                        bias_text = ", ".join(
                            f"{label}={_format_float(float(value))}"
                            for label, value in sorted(bias_by_label.items(), key=lambda item: str(item[0]))
                        )
                        lines.append(f"- recommended_precision_logit_bias: {bias_text}")
                    failures = recommended.get("precision_threshold_failures", [])
                    if isinstance(failures, list) and failures:
                        lines.append(f"- recommended_failures: {', '.join(str(item) for item in failures)}")
                    lines.append(
                        "| macro_f1 | final_net_gain | final_harm | Unknown recall | Clutter recall | parent macro_f1 |"
                    )
                    lines.append("|---:|---:|---:|---:|---:|---:|")
                    lines.append(
                        f"| {_format_float(float(recommended.get('macro_f1', 0.0)))} | "
                        f"{_format_float(float(recommended.get('final_net_gain_rate_on_rd', 0.0)))} | "
                        f"{_format_float(float(recommended.get('final_harm_rate_on_rd', 0.0)))} | "
                        f"{_format_float(float(recommended.get('final_unknown_recall', 0.0)))} | "
                        f"{_format_float(float(recommended.get('final_clutter_recall', 0.0)))} | "
                        f"{_format_float(float(recommended.get('parent_mean_logit_macro_f1', 0.0)))} |"
                    )
                calibration_results = calibration_probe.get("results", [])
                if isinstance(calibration_results, list) and calibration_results:
                    lines.append("")
                    lines.append("| bias | macro_f1 | final_net_gain | final_harm | Unknown recall | Clutter recall | failures |")
                    lines.append("|---|---:|---:|---:|---:|---:|---|")
                    for item in calibration_results[:8]:
                        if not isinstance(item, dict):
                            continue
                        bias_by_label = item.get("precision_logit_bias_by_label", {})
                        if isinstance(bias_by_label, dict) and bias_by_label:
                            bias_text = ", ".join(
                                f"{label}={_format_float(float(value))}"
                                for label, value in sorted(bias_by_label.items(), key=lambda pair: str(pair[0]))
                            )
                        else:
                            bias_text = "none"
                        failures = item.get("precision_threshold_failures", [])
                        failure_text = ", ".join(str(value) for value in failures) if isinstance(failures, list) else ""
                        lines.append(
                            f"| {_escape_md_cell(bias_text)} | "
                            f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
                            f"{_format_float(float(item.get('final_net_gain_rate_on_rd', 0.0)))} | "
                            f"{_format_float(float(item.get('final_harm_rate_on_rd', 0.0)))} | "
                            f"{_format_float(float(item.get('final_unknown_recall', 0.0)))} | "
                            f"{_format_float(float(item.get('final_clutter_recall', 0.0)))} | "
                            f"{_escape_md_cell(failure_text)} |"
                        )
            if isinstance(precision_delta, dict) and precision_delta.get("selection_mismatch", False):
                lines.append(
                    f"- selection_mismatch: precision candidate `{precision_delta.get('precision_candidate_checkpoint_name', '')}` "
                    f"beats selected `{precision_delta.get('selected_checkpoint_name', '')}` by "
                    f"{_format_float(float(precision_delta.get('macro_f1_delta', 0.0)))} macro_f1; release gate is unchanged."
                )
            lines.append(
                "| macro_f1 | traj_macro_f1 | joint_gain_on_rd | RD evidence on-RD | final_net_gain | final_harm | Unknown recall | Clutter recall | parent macro_f1 | supported source F1 |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {_format_float(float(precision_candidate.get('macro_f1', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('traj_macro_f1', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('joint_minus_traj_macro_f1_on_rd', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('rd_macro_f1_on_rd', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('final_net_gain_rate_on_rd', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('final_harm_rate_on_rd', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('final_unknown_recall', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('final_clutter_recall', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('parent_mean_logit_macro_f1', 0.0)))} | "
                f"{_format_float(float(precision_candidate.get('source_balanced_supported_macro_f1', 0.0)))} |"
            )
            if isinstance(precision_delta, dict) and precision_delta:
                lines.append("")
                lines.append("| delta vs selected | value |")
                lines.append("|---|---:|")
                for key in [
                    "macro_f1_delta",
                    "traj_macro_f1_delta",
                    "joint_minus_traj_macro_f1_on_rd_delta",
                    "rd_macro_f1_on_rd_delta",
                    "final_net_gain_rate_on_rd_delta",
                    "final_harm_rate_on_rd_delta",
                    "final_unknown_recall_delta",
                    "final_clutter_recall_delta",
                    "parent_mean_logit_macro_f1_delta",
                ]:
                    if key in precision_delta:
                        lines.append(f"| {key} | {_format_float(float(precision_delta.get(key, 0.0)))} |")
        if isinstance(rd_joint_bottleneck, dict) and rd_joint_bottleneck:
            lines.append("")
            lines.append("### RD/Joint Bottleneck")
            lines.append(f"- status: {rd_joint_bottleneck.get('status', '')}")
            lines.append(f"- primary_bottleneck: {rd_joint_bottleneck.get('primary_bottleneck', '')}")
            lines.append(f"- next_action: {rd_joint_bottleneck.get('next_action', '')}")
            lines.append(
                "- full_training_rerun_recommended_now: "
                f"{rd_joint_bottleneck.get('full_training_rerun_recommended_now', False)}"
            )
            flags = rd_joint_bottleneck.get("diagnosis_flags", [])
            if isinstance(flags, list) and flags:
                lines.append(f"- diagnosis_flags: {', '.join(str(item) for item in flags)}")
            bottleneck_metrics = rd_joint_bottleneck.get("metrics", {})
            if isinstance(bottleneck_metrics, dict) and bottleneck_metrics:
                lines.append(
                    "| with_rd | scale | joint_gain_on_rd | RD evidence on-RD | final_net_gain | final_harm | Unknown recall | Clutter recall |"
                )
                lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
                lines.append(
                    f"| {int(bottleneck_metrics.get('with_rd_count', 0))} | "
                    f"{_format_float(float(bottleneck_metrics.get('selected_joint_logit_delta_scale', 0.0)))} | "
                    f"{_format_float(float(bottleneck_metrics.get('joint_minus_traj_macro_f1_on_rd', 0.0)))} | "
                    f"{_format_float(float(bottleneck_metrics.get('rd_macro_f1_on_rd', 0.0)))} | "
                    f"{_format_float(float(bottleneck_metrics.get('final_net_gain_rate_on_rd', 0.0)))} | "
                    f"{_format_float(float(bottleneck_metrics.get('final_harm_rate_on_rd', 0.0)))} | "
                    f"{_format_float(float(bottleneck_metrics.get('final_unknown_recall', 0.0)))} | "
                    f"{_format_float(float(bottleneck_metrics.get('final_clutter_recall', 0.0)))} |"
                )
            release_gaps = rd_joint_bottleneck.get("release_gaps", {})
            if isinstance(release_gaps, dict) and release_gaps:
                lines.append("")
                lines.append("| RD/joint gap | value |")
                lines.append("|---|---:|")
                for key in [
                    "precision_joint_minus_traj_macro_f1_on_rd_gap",
                    "joint_minus_traj_macro_f1_on_rd_gap",
                    "rd_macro_f1_on_rd_gap",
                    "final_net_gain_rate_on_rd_gap",
                    "final_harm_rate_on_rd_gap",
                ]:
                    if key in release_gaps:
                        lines.append(f"| {key} | {_format_float(float(release_gaps.get(key, 0.0)))} |")
            hard_pairs = rd_joint_bottleneck.get("hard_pair_bottlenecks", [])
            if isinstance(hard_pairs, list) and hard_pairs:
                lines.append("")
                lines.append("| hard pair | support | traj acc | rd acc | rd-traj | final acc | final-traj | rd net | final net |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
                for item in hard_pairs:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {_escape_md_cell(item.get('pair_name', ''))} | "
                        f"{int(item.get('support', 0))} | "
                        f"{_format_float(float(item.get('traj_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('rd_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('rd_minus_traj_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('final_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('final_minus_traj_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('rd_net_gain_rate', 0.0)))} | "
                        f"{_format_float(float(item.get('final_net_gain_rate', 0.0)))} |"
                    )
            source_rows = rd_joint_bottleneck.get("source_bottlenecks_on_rd", [])
            if isinstance(source_rows, list) and source_rows:
                lines.append("")
                lines.append("| source on-RD | support | traj F1 | joint F1 | joint-traj | RD F1 | final F1 | final-traj |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
                for item in source_rows:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {_escape_md_cell(item.get('source_name', ''))} | "
                        f"{int(item.get('support_on_rd', 0))} | "
                        f"{_format_float(float(item.get('traj_macro_f1_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('joint_macro_f1_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('joint_minus_traj_macro_f1_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('rd_macro_f1_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('final_macro_f1_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('final_minus_traj_macro_f1_on_rd', 0.0)))} |"
                    )
            time_sync_rows = rd_joint_bottleneck.get("time_sync_bottlenecks", [])
            if isinstance(time_sync_rows, list) and time_sync_rows:
                lines.append("")
                lines.append("| time sync | support | final F1 | traj F1 | joint F1 | joint-traj | RD F1 |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|")
                for item in time_sync_rows:
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {_escape_md_cell(item.get('rd_time_sync_status', ''))} | "
                        f"{int(item.get('support', 0))} | "
                        f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
                        f"{_format_float(float(item.get('traj_macro_f1', 0.0)))} | "
                        f"{_format_float(float(item.get('joint_macro_f1', 0.0)))} | "
                        f"{_format_float(float(item.get('joint_minus_traj_macro_f1', 0.0)))} | "
                        f"{_format_float(float(item.get('rd_macro_f1', 0.0)))} |"
                    )
        if isinstance(precision_readiness, dict) and precision_readiness:
            lines.append("")
            lines.append("### Precision Readiness")
            lines.append(f"- status: {precision_readiness.get('status', '')}")
            lines.append(f"- existing_scale_ready: {precision_readiness.get('existing_scale_ready', False)}")
            lines.append(
                "- full_training_rerun_recommended_now: "
                f"{precision_readiness.get('full_training_rerun_recommended_now', False)}"
            )
            if precision_readiness.get("next_action"):
                lines.append(f"- next_action: {precision_readiness.get('next_action', '')}")
            closest = precision_readiness.get("closest_candidate", {})
            if isinstance(closest, dict) and closest:
                lines.append(
                    "- closest_candidate: "
                    f"{closest.get('checkpoint_name', '')} scale={_format_float(float(closest.get('precision_scale', 0.0)))} "
                    f"macro_f1={_format_float(float(closest.get('macro_f1', 0.0)))} "
                    f"Unknown={_format_float(float(closest.get('final_unknown_recall', 0.0)))} "
                    f"Clutter={_format_float(float(closest.get('final_clutter_recall', 0.0)))}"
                )
                failures = closest.get("precision_threshold_failures", [])
                if isinstance(failures, list) and failures:
                    lines.append(f"- closest_failures: {', '.join(str(item) for item in failures)}")
                gaps = closest.get("precision_threshold_gaps", {})
                if isinstance(gaps, dict) and gaps:
                    lines.append("| closest gap | value |")
                    lines.append("|---|---:|")
                    for key in [
                        "precision_candidate_unknown_recall_gap",
                        "precision_candidate_clutter_recall_gap",
                        "precision_candidate_joint_gain_gap",
                    ]:
                        if key in gaps:
                            lines.append(f"| {key} | {_format_float(float(gaps.get(key, 0.0)))} |")
            readiness_results = precision_readiness.get("results", [])
            if isinstance(readiness_results, list) and readiness_results:
                lines.append("")
                lines.append("### Checkpoint Precision View")
                lines.append(
                    "| checkpoint | ready | precision_scale | epoch | macro_f1 | joint_gain_on_rd | final_net_gain | final_harm | Unknown recall | Clutter recall | failures |"
                )
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
                for item in readiness_results:
                    if not isinstance(item, dict):
                        continue
                    failures = item.get("precision_threshold_failures", [])
                    failure_text = ", ".join(str(value) for value in failures) if isinstance(failures, list) else ""
                    lines.append(
                        f"| {_escape_md_cell(item.get('checkpoint_name', ''))} | "
                        f"{int(bool(item.get('ready', False)))} | "
                        f"{_format_float(float(item.get('precision_scale', 0.0)))} | "
                        f"{item.get('epoch', -1)} | "
                        f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
                        f"{_format_float(float(item.get('joint_minus_traj_macro_f1_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('final_net_gain_rate_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('final_harm_rate_on_rd', 0.0)))} | "
                        f"{_format_float(float(item.get('final_unknown_recall', 0.0)))} | "
                        f"{_format_float(float(item.get('final_clutter_recall', 0.0)))} | "
                        f"{_escape_md_cell(failure_text)} |"
                    )
        if diagnostic_recommendation:
            lines.append("")
            lines.append("### Diagnostic Test Recommendation")
            lines.append(f"- checkpoint_name: {diagnostic_recommendation.get('checkpoint_name', '')}")
            lines.append(f"- epoch: {diagnostic_recommendation.get('epoch', -1)}")
            lines.append(f"- basis: {diagnostic_recommendation.get('recommendation_basis', '')}")
            lines.append(
                "| macro_f1 | traj_macro_f1 | joint_gain_on_rd | RD evidence on-RD | final_net_gain | final_harm | Unknown recall | Clutter recall | supported source F1 |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            lines.append(
                f"| {_format_float(float(diagnostic_recommendation.get('macro_f1', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('traj_macro_f1', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('joint_minus_traj_macro_f1_on_rd', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('rd_macro_f1_on_rd', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('final_net_gain_rate_on_rd', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('final_harm_rate_on_rd', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('final_unknown_recall', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('final_clutter_recall', 0.0)))} | "
                f"{_format_float(float(diagnostic_recommendation.get('source_balanced_supported_macro_f1', 0.0)))} |"
            )
        if diagnostic_results:
            lines.append("")
            lines.append("### Checkpoint 对比")
            lines.append(
                "| checkpoint | selected | selected_scale | best_probe_scale | precision_scale | epoch | macro_f1 | traj_macro_f1 | joint_gain_on_rd | RD evidence on-RD | final_net_gain | final_harm | Unknown recall | Clutter recall | parent macro_f1 | supported source F1 |"
            )
            lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for item in diagnostic_results:
                if not isinstance(item, dict):
                    continue
                if str(item.get("status", "ok")) != "ok":
                    lines.append(
                        f"| {_escape_md_cell(item.get('checkpoint_name', ''))} | {int(bool(item.get('is_selected_checkpoint', False)))} | "
                        f"{_format_float(float(item.get('selected_joint_logit_delta_scale', 0.0)))} | "
                        f"{_format_float(float(item.get('joint_logit_delta_probe_best_scale', 0.0)))} | "
                        f"{_format_float(float(item.get('conservative_precision_scale', 0.0)))} | "
                        f"{item.get('epoch', -1)} | missing | missing | missing | missing | missing | missing | missing | missing | missing | missing |"
                    )
                    continue
                lines.append(
                    f"| {_escape_md_cell(item.get('checkpoint_name', ''))} | {int(bool(item.get('is_selected_checkpoint', False)))} | "
                    f"{_format_float(float(item.get('selected_joint_logit_delta_scale', 0.0)))} | "
                    f"{_format_float(float(item.get('joint_logit_delta_probe_best_scale', item.get('selected_joint_logit_delta_scale', 0.0))))} | "
                    f"{_format_float(float(item.get('conservative_precision_scale', item.get('selected_joint_logit_delta_scale', 0.0))))} | "
                    f"{item.get('epoch', -1)} | {_format_float(float(item.get('macro_f1', 0.0)))} | "
                    f"{_format_float(float(item.get('traj_macro_f1', 0.0)))} | "
                    f"{_format_float(float(item.get('joint_minus_traj_macro_f1_on_rd', 0.0)))} | "
                    f"{_format_float(float(item.get('rd_macro_f1_on_rd', 0.0)))} | "
                    f"{_format_float(float(item.get('final_net_gain_rate_on_rd', 0.0)))} | "
                    f"{_format_float(float(item.get('final_harm_rate_on_rd', 0.0)))} | "
                    f"{_format_float(float(item.get('final_unknown_recall', 0.0)))} | "
                    f"{_format_float(float(item.get('final_clutter_recall', 0.0)))} | "
                    f"{_format_float(float(item.get('parent_mean_logit_macro_f1', 0.0)))} | "
                    f"{_format_float(float(item.get('source_balanced_supported_macro_f1', 0.0)))} |"
                )
            precision_view_rows = [item for item in diagnostic_results if isinstance(item, dict) and str(item.get("status", "ok")) == "ok"]
            if precision_view_rows and not (isinstance(precision_readiness, dict) and precision_readiness.get("results")):
                lines.append("")
                lines.append("### Checkpoint Precision View")
                lines.append(
                    "| checkpoint | precision_scale | epoch | macro_f1 | joint_gain_on_rd | final_net_gain | final_harm | Unknown recall | Clutter recall | parent macro_f1 | supported source F1 |"
                )
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for item in precision_view_rows:
                    lines.append(
                        f"| {_escape_md_cell(item.get('checkpoint_name', ''))} | "
                        f"{_format_float(float(_precision_view_value(item, 'scale')))} | "
                        f"{item.get('epoch', -1)} | "
                        f"{_format_float(float(_precision_view_value(item, 'macro_f1')))} | "
                        f"{_format_float(float(_precision_view_value(item, 'joint_minus_traj_macro_f1_on_rd')))} | "
                        f"{_format_float(float(_precision_view_value(item, 'final_net_gain_rate_on_rd')))} | "
                        f"{_format_float(float(_precision_view_value(item, 'final_harm_rate_on_rd')))} | "
                        f"{_format_float(float(_precision_view_value(item, 'final_unknown_recall')))} | "
                        f"{_format_float(float(_precision_view_value(item, 'final_clutter_recall')))} | "
                        f"{_format_float(float(item.get('parent_mean_logit_macro_f1', 0.0)))} | "
                        f"{_format_float(float(_precision_view_value(item, 'source_balanced_supported_macro_f1')))} |"
                    )
            lines.append("")
        if isinstance(joint_logit_delta_probe, dict) and joint_logit_delta_probe.get("results"):
            lines.append("### Joint Delta Probe")
            lines.append(f"- selection_metric: {joint_logit_delta_probe.get('selection_metric', '')}")
            lines.append(f"- validation_selected_scale: {_format_float(float(joint_logit_delta_probe.get('selected_scale', 0.0)))}")
            if "best_test_scale" in joint_logit_delta_probe:
                lines.append(f"- best_test_scale: {_format_float(float(joint_logit_delta_probe.get('best_test_scale', 0.0)))}")
            lines.append(
                "| scale | selected | best_test | macro_f1 | joint_gain_on_rd | final_net_gain | final_harm | Unknown recall | Clutter recall |"
            )
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            selected_scale = float(joint_logit_delta_probe.get("selected_scale", 0.0))
            best_test_scale = float(joint_logit_delta_probe.get("best_test_scale", selected_scale))
            for item in joint_logit_delta_probe.get("results", []):
                if not isinstance(item, dict):
                    continue
                scale = float(item.get("scale", 0.0))
                lines.append(
                    f"| {_format_float(scale)} | {int(abs(scale - selected_scale) <= 1.0e-6)} | "
                    f"{int(abs(scale - best_test_scale) <= 1.0e-6)} | "
                    f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
                    f"{_format_float(float(item.get('joint_minus_traj_macro_f1_on_rd', 0.0)))} | "
                    f"{_format_float(float(item.get('final_net_gain_rate_on_rd', 0.0)))} | "
                    f"{_format_float(float(item.get('final_harm_rate_on_rd', 0.0)))} | "
                    f"{_format_float(float(item.get('final_unknown_recall', 0.0)))} | "
                    f"{_format_float(float(item.get('final_clutter_recall', 0.0)))} |"
                )
            lines.append("")
            hard_pair_results = [
                item for item in joint_logit_delta_probe.get("results", [])
                if isinstance(item, dict) and isinstance(item.get("unknown_clutter_hard_pair"), dict)
            ]
            if hard_pair_results:
                lines.append("### Joint Delta Hard Pair")
                lines.append("| scale | selected | best_test | support | traj acc | rd acc | final acc | rd net | final net | Unknown recall | Clutter recall |")
                lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
                for item in hard_pair_results:
                    scale = float(item.get("scale", 0.0))
                    hard_pair = item.get("unknown_clutter_hard_pair", {})
                    if not isinstance(hard_pair, dict):
                        continue
                    lines.append(
                        f"| {_format_float(scale)} | {int(abs(scale - selected_scale) <= 1.0e-6)} | "
                        f"{int(abs(scale - best_test_scale) <= 1.0e-6)} | "
                        f"{int(hard_pair.get('support', 0))} | "
                        f"{_format_float(float(hard_pair.get('traj_accuracy', 0.0)))} | "
                        f"{_format_float(float(hard_pair.get('rd_accuracy', 0.0)))} | "
                        f"{_format_float(float(hard_pair.get('final_accuracy', 0.0)))} | "
                        f"{_format_float(float(hard_pair.get('rd_net_gain_rate', 0.0)))} | "
                        f"{_format_float(float(hard_pair.get('final_net_gain_rate', 0.0)))} | "
                        f"{_format_float(float(hard_pair.get('UnknownTarget_recall', 0.0)))} | "
                        f"{_format_float(float(hard_pair.get('ClutterTarget_recall', 0.0)))} |"
                    )
                lines.append("")
    lines.append("## 各类别指标")
    lines.append("| 类别 | support | precision | recall | f1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for item in per_class:
        lines.append(
            f"| {item['label_name']} | {item['support']} | "
            f"{_format_float(float(item['precision']))} | "
            f"{_format_float(float(item['recall']))} | "
            f"{_format_float(float(item['f1']))} |"
        )
    lines.append("")
    lines.append("## 主要混淆对")
    if top_confusions:
        lines.append("| 真实类别 | 预测类别 | 次数 |")
        lines.append("|---|---|---:|")
        for true_label, pred_label, count in top_confusions:
            lines.append(f"| {true_label} | {pred_label} | {count} |")
    else:
        lines.append("- 无明显混淆。")
    lines.append("")
    lines.append("## 路由表现")
    if route_report.get("collapsed_route"):
        lines.append(f"- collapsed_route: {route_report['collapsed_route']}")
    if "macro_recall" in route_report:
            lines.append(f"- macro_recall: {_format_float(float(route_report['macro_recall']))}")
    lines.append("| 路由 | support | correct | recall |")
    lines.append("|---|---:|---:|---:|")
    for item in route_report["per_route"]:
        lines.append(
            f"| {_escape_md_cell(item['route_name'])} | {item['support']} | {item['correct']} | {_format_float(float(item['recall']))} |"
        )
    lines.append("")
    reroute_metrics = {}
    if isinstance(critical_metrics, dict):
        candidate = critical_metrics.get("reroute", {})
        if isinstance(candidate, dict):
            reroute_metrics = candidate
    if reroute_metrics:
        lines.append("## 回流重路由")
        if isinstance(reroute_config, dict) and reroute_config:
            lines.append("| 配置项 | 值 |")
            lines.append("|---|---:|")
            for key in [
                "max_passes",
                "penalty",
                "max_total_penalty",
                "low_altitude_conf_threshold",
                "conventional_air_conf_threshold",
                "ground_conf_threshold",
            ]:
                if key in reroute_config:
                    value = reroute_config[key]
                    if isinstance(value, int):
                        lines.append(f"| {key} | {value} |")
                    else:
                        lines.append(f"| {key} | {_format_float(float(value))} |")
            lines.append("")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        metric_labels = [
            ("reroute_rate", "触发率"),
            ("reroute_avg_count", "平均回流次数"),
            ("reroute_recovery_rate", "回流后正确率"),
            ("reroute_fallback_rate", "Unknown 兜底率"),
            ("reroute_worsen_rate", "回流后错误率"),
            ("low_altitude_deep_used_rate", "Expert1 deep 使用率"),
            ("low_altitude_deep_accept_rate", "Expert1 deep 接受率"),
            ("low_altitude_deep_recovery_rate", "Expert1 deep recovery 率"),
            ("low_altitude_deep_harm_rate", "Expert1 deep harm 率"),
        ]
        for key, label in metric_labels:
            if key in reroute_metrics:
                lines.append(f"| {label} | {_format_float(float(reroute_metrics[key]))} |")
        distribution = reroute_metrics.get("reroute_count_distribution", {})
        if isinstance(distribution, dict) and distribution:
            lines.append("")
            lines.append("| 回流次数 | 样本数 |")
            lines.append("|---:|---:|")
            for count, value in sorted(distribution.items(), key=lambda item: int(item[0])):
                lines.append(f"| {count} | {value} |")
        reason_distribution = critical_metrics.get("reroute_reason_distribution", {}) if isinstance(critical_metrics, dict) else {}
        if isinstance(reason_distribution, dict) and reason_distribution:
            lines.append("")
            lines.append("| reject reason | 样本数 |")
            lines.append("|---|---:|")
            for reason, value in sorted(reason_distribution.items(), key=lambda item: str(item[0])):
                lines.append(f"| {reason} | {value} |")
        lines.append("")
    if isinstance(reroute_comparison, dict) and reroute_comparison:
        lines.append("## 回退前后对比")
        lines.append(f"- without_reroute_mode: {reroute_comparison.get('without_reroute_mode')}")
        lines.append(f"- with_reroute_mode: {reroute_comparison.get('with_reroute_mode')}")
        without_reroute = reroute_comparison.get("without_reroute", {})
        with_reroute = reroute_comparison.get("with_reroute", {})
        lines.append("| 口径 | accuracy | macro_f1 |")
        lines.append("|---|---:|---:|")
        if isinstance(without_reroute, dict):
            lines.append(
                f"| without | {_format_float(float(without_reroute.get('accuracy', 0.0)))} | "
                f"{_format_float(float(without_reroute.get('macro_f1', 0.0)))} |"
            )
        if isinstance(with_reroute, dict):
            lines.append(
                f"| with | {_format_float(float(with_reroute.get('accuracy', 0.0)))} | "
                f"{_format_float(float(with_reroute.get('macro_f1', 0.0)))} |"
            )
        lines.append(
            f"- changed/helped/harmed: {reroute_comparison.get('changed_count', 0)} / "
            f"{reroute_comparison.get('helped_count', 0)} / {reroute_comparison.get('harmed_count', 0)}"
        )
        per_class_delta = reroute_comparison.get("per_class_before_after_recall", {})
        if isinstance(per_class_delta, dict) and per_class_delta:
            lines.append("")
            lines.append("| 类别 | support | without recall | with recall | delta | helped | harmed |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            for label_name in label_names:
                item = per_class_delta.get(label_name, {})
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"| {label_name} | {item.get('support', 0)} | "
                    f"{_format_float(float(item.get('without_recall', 0.0)))} | "
                    f"{_format_float(float(item.get('with_recall', 0.0)))} | "
                    f"{_format_float(float(item.get('recall_delta', 0.0)))} | "
                    f"{item.get('helped_count', 0)} | {item.get('harmed_count', 0)} |"
                )
        sample_outcomes = reroute_comparison.get("sample_outcomes", [])
        if isinstance(sample_outcomes, list) and sample_outcomes:
            lines.append("")
            lines.append("| 回退样本 | true | without | with | path | reason | helped | harmed |")
            lines.append("|---|---|---|---|---|---|---:|---:|")
            for item in sample_outcomes[:20]:
                if not isinstance(item, dict):
                    continue
                path = " -> ".join(item.get("reroute_path", []))
                lines.append(
                    f"| {_escape_md_cell(item.get('sample_id', ''))} | {_escape_md_cell(item.get('true_label', ''))} | "
                    f"{_escape_md_cell(item.get('without_pred', ''))} | {_escape_md_cell(item.get('with_pred', ''))} | "
                    f"{_escape_md_cell(path)} | {_escape_md_cell(item.get('reject_reason', ''))} | "
                    f"{int(bool(item.get('reroute_helped', False)))} | "
                    f"{int(bool(item.get('reroute_harmed', False)))} |"
                )
        safety_pairs = reroute_comparison.get("safety_critical_pairs", {})
        if isinstance(safety_pairs, dict) and safety_pairs:
            lines.append("")
            lines.append("| 安全关键误判对 | before | after | delta | reduced | introduced |")
            lines.append("|---|---:|---:|---:|---:|---:|")
            for pair_name, item in safety_pairs.items():
                if not isinstance(item, dict):
                    continue
                lines.append(
                    f"| {_escape_md_cell(pair_name)} | {item.get('before_count', 0)} | "
                    f"{item.get('after_count', 0)} | {item.get('delta', 0)} | "
                    f"{item.get('reduced_count', 0)} | {item.get('introduced_count', 0)} |"
                )
        lines.append("")
    deep_records = [
        item for item in predictions
        if item.get("reroute_low_altitude_deep_used")
    ]
    if deep_records:
        lines.append("## Expert1 deep_reroute 样本")
        lines.append("| sample_id | true | pass1 | deep | final | accept reason | helped | harmed | pair_delta | residual_conf |")
        lines.append("|---|---|---|---|---|---|---:|---:|---:|---:|")
        for item in deep_records[:20]:
            lines.append(
                f"| {_escape_md_cell(item.get('sample_id', ''))} | {_escape_md_cell(item.get('label_name_true', ''))} | "
                f"{_escape_md_cell(item.get('reroute_low_altitude_pass1_pred_label', ''))} | "
                f"{_escape_md_cell(item.get('reroute_low_altitude_deep_pred_label', ''))} | "
                f"{_escape_md_cell(item.get('reroute_low_altitude_final_pred_label', ''))} | "
                f"{_escape_md_cell(item.get('reroute_low_altitude_deep_accept_reason', ''))} | "
                f"{int(bool(item.get('reroute_low_altitude_deep_helped', False)))} | "
                f"{int(bool(item.get('reroute_low_altitude_deep_harmed', False)))} | "
                f"{_format_float(float(item.get('low_deep_pair_correction_max', 0.0)))} | "
                f"{_format_float(float(item.get('low_deep_unknown_clutter_confidence', 0.0)))} |"
            )
        lines.append("")
    ground_metrics = {}
    if isinstance(critical_metrics, dict):
        candidate = critical_metrics.get("ground", {})
        if isinstance(candidate, dict):
            ground_metrics = candidate
    if ground_metrics:
        lines.append("## Ground 仲裁")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        metric_labels = [
            ("ground_group_precision", "Ground precision"),
            ("ground_group_recall", "Ground recall"),
            ("ground_group_f1", "Ground F1"),
            ("pred_ground_on_non_ground_count", "非 Ground 误判 Ground 数"),
            ("groundness_probability_mean_true_ground", "真 Ground groundness 均值"),
            ("groundness_probability_mean_non_ground", "非 Ground groundness 均值"),
            ("groundness_confidence_mean_true_ground", "真 Ground confidence 均值"),
            ("groundness_confidence_mean_non_ground", "非 Ground confidence 均值"),
            ("ground_high_confidence_count", "Ground 高置信样本数"),
            ("ground_candidate_high_confidence_count", "Ground candidate 高置信样本数"),
            ("ground_route_not_opposed_count", "Router 未强烈反对 Ground 数"),
            ("ground_top1_margin", "Ground top1 margin 均值"),
            ("ground_top1_margin_mean_true_ground", "真 Ground top1 margin 均值"),
            ("ground_top1_margin_mean_non_ground", "非 Ground top1 margin 均值"),
            ("ground_candidate_changed_prediction_count", "Ground candidate 改变预测数"),
            ("recovered_count", "Ground recovered_count"),
            ("false_positive_introduced_count", "Ground false_positive_introduced_count"),
        ]
        for key, label in metric_labels:
            if key not in ground_metrics:
                continue
            value = ground_metrics[key]
            if isinstance(value, int):
                lines.append(f"| {label} | {value} |")
            else:
                lines.append(f"| {label} | {_format_float(float(value))} |")
        lines.append("")
    aircraft_metrics = {}
    if isinstance(critical_metrics, dict):
        candidate = critical_metrics.get("aircraft", {})
        if isinstance(candidate, dict):
            aircraft_metrics = candidate
    if aircraft_metrics:
        lines.append("## Aircraft / Expert2 仲裁")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        for key, label in [
            ("aircraft_system_precision", "Aircraft precision"),
            ("aircraft_system_recall", "Aircraft recall"),
            ("expert2_positive_system_not_aircraft_count", "Expert2 positive but final not Aircraft"),
            ("final_aircraft_expert2_not_positive_count", "Final Aircraft but Expert2 not positive"),
            ("final_aircraft_expert2_not_positive_false_aircraft_count", "Final Aircraft no Expert2 FP"),
            ("non_aircraft_as_aircraft_count", "NonAircraft -> Aircraft count"),
            ("non_aircraft_as_aircraft_rate", "NonAircraft -> Aircraft rate"),
        ]:
            if key not in aircraft_metrics:
                continue
            value = aircraft_metrics[key]
            if isinstance(value, int):
                lines.append(f"| {label} | {value} |")
            else:
                lines.append(f"| {label} | {_format_float(float(value))} |")
        fp_by_source = aircraft_metrics.get("aircraft_false_positive_by_source", {})
        if isinstance(fp_by_source, dict) and fp_by_source:
            lines.append("")
            lines.append("| Aircraft FP source | count |")
            lines.append("|---|---:|")
            for source_name, count in sorted(fp_by_source.items(), key=lambda item: str(item[0])):
                lines.append(f"| {_escape_md_cell(source_name)} | {count} |")
        lines.append("")
    expert_internal = report.get("expert_internal", {})
    if isinstance(expert_internal, dict) and expert_internal:
        lines.append("## 专家内部性能")
        expert2 = expert_internal.get("expert2_conventional_air", {})
        if isinstance(expert2, dict) and expert2:
            lines.append("### Expert2 常规空中")
            lines.append(
                f"- accuracy={_format_float(float(expert2.get('accuracy', 0.0)))}, "
                f"macro_f1={_format_float(float(expert2.get('macro_f1', 0.0)))}, "
                f"aircraft_precision={_format_float(float(expert2.get('aircraft_precision', 0.0)))}, "
                f"aircraft_recall={_format_float(float(expert2.get('aircraft_recall', 0.0)))}"
            )
            matrix = expert2.get("confusion_matrix", [])
            labels = expert2.get("label_names", [])
            if isinstance(matrix, list) and isinstance(labels, list) and matrix:
                lines.append("| true/pred | " + " | ".join(str(label) for label in labels) + " |")
                lines.append("|---" + "|---:" * len(labels) + "|")
                for idx, row in enumerate(matrix):
                    label = labels[idx] if idx < len(labels) else str(idx)
                    lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
        expert3 = expert_internal.get("expert3_high_altitude_rule", {})
        if isinstance(expert3, dict) and expert3:
            lines.append("### Expert3 高空规则")
            lines.append(
                f"- selected={expert3.get('selected_count', 0)}, "
                f"expected={expert3.get('expected_count', 0)}, "
                f"precision={_format_float(float(expert3.get('precision', 0.0)))}, "
                f"recall={_format_float(float(expert3.get('recall', 0.0)))}"
            )
        expert4 = expert_internal.get("expert4_ground", {})
        if isinstance(expert4, dict) and expert4:
            lines.append("### Expert4 地面")
            lines.append(
                f"- accuracy={_format_float(float(expert4.get('accuracy', 0.0)))}, "
                f"macro_f1={_format_float(float(expert4.get('macro_f1', 0.0)))}, "
                f"people_as_car_rate={_format_float(float(expert4.get('people_as_car_rate', 0.0)))}"
            )
            matrix = expert4.get("confusion_matrix", [])
            labels = expert4.get("scope_label_names", [])
            if isinstance(matrix, list) and isinstance(labels, list) and matrix:
                lines.append("| true/pred | " + " | ".join(str(label) for label in labels) + " |")
                lines.append("|---" + "|---:" * len(labels) + "|")
                for idx, row in enumerate(matrix):
                    label = labels[idx] if idx < len(labels) else str(idx)
                    lines.append(f"| {label} | " + " | ".join(str(value) for value in row) + " |")
        expert1 = expert_internal.get("expert1_low_altitude", {})
        if isinstance(expert1, dict) and expert1:
            trajectory = expert1.get("trajectory", {})
            rd_on_rd = expert1.get("rd_on_rd", {})
            lines.append("### Expert1 轨迹/RD")
            if isinstance(trajectory, dict):
                lines.append(
                    f"- trajectory: accuracy={_format_float(float(trajectory.get('accuracy', 0.0)))}, "
                    f"macro_f1={_format_float(float(trajectory.get('macro_f1', 0.0)))}"
                )
            if isinstance(rd_on_rd, dict):
                lines.append(
                    f"- rd_on_rd: accuracy={_format_float(float(rd_on_rd.get('accuracy', 0.0)))}, "
                    f"macro_f1={_format_float(float(rd_on_rd.get('macro_f1', 0.0)))}"
                )
            lines.append(
                f"- traj_confidence_mean={_format_float(float(expert1.get('traj_confidence_mean', 0.0)))}, "
                f"traj_entropy_mean={_format_float(float(expert1.get('traj_entropy_mean', 0.0)))}, "
                f"traj_conf>0.8 correct/wrong={expert1.get('traj_confidence_gt_0_8_correct_count', 0)} / "
                f"{expert1.get('traj_confidence_gt_0_8_wrong_count', 0)}"
            )
        rd_proto = expert_internal.get("rd_prototype_similarity", {})
        if isinstance(rd_proto, dict) and rd_proto:
            labels = rd_proto.get("label_names", [])
            matrix = rd_proto.get("cosine_matrix", [])
            if isinstance(labels, list) and isinstance(matrix, list) and matrix:
                lines.append("### RD CosFace 原型相似度")
                lines.append("| class | " + " | ".join(str(label) for label in labels) + " |")
                lines.append("|---" + "|---:" * len(labels) + "|")
                for idx, row in enumerate(matrix):
                    label = labels[idx] if idx < len(labels) else str(idx)
                    lines.append(
                        f"| {label} | "
                        + " | ".join(_format_float(float(value)) for value in row)
                        + " |"
                    )
        lines.append("")
    expert1_scope = report.get("expert1_low_altitude", {})
    if isinstance(expert1_scope, dict) and "routed" in expert1_scope:
        lines.append("## Expert1 低空双口径")
        lines.append("| 口径 | count | accuracy | macro_f1 | Unknown recall | Clutter recall |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for key, label in [("routed", "router_assigned"), ("true_low_altitude", "true_low_altitude")]:
            item = expert1_scope.get(key, {})
            if not isinstance(item, dict):
                continue
            per_class_recall = item.get("per_class_recall", {})
            count_key = "routed_count" if key == "routed" else "true_low_altitude_count"
            lines.append(
                f"| {label} | {expert1_scope.get(count_key, 0)} | "
                f"{_format_float(float(item.get('accuracy', 0.0)))} | "
                f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
                f"{_format_float(float(per_class_recall.get('UnknownTarget', 0.0) if isinstance(per_class_recall, dict) else 0.0))} | "
                f"{_format_float(float(per_class_recall.get('ClutterTarget', 0.0) if isinstance(per_class_recall, dict) else 0.0))} |"
            )
        lines.append("")
    branch_metrics = report.get("branch_metrics", {})
    if isinstance(branch_metrics, dict) and "rd" in branch_metrics:
        final_branch = branch_metrics.get("final", {})
        traj_branch = branch_metrics.get("trajectory", {})
        joint_branch = branch_metrics.get("joint", {})
        if (not isinstance(joint_branch, dict) or not joint_branch) and isinstance(summary_metrics, dict):
            joint_branch = {
                "accuracy": summary_metrics.get("joint_accuracy", test_metrics.get("joint_accuracy", 0.0)),
                "macro_f1": summary_metrics.get("joint_macro_f1", test_metrics.get("joint_macro_f1", 0.0)),
                "accuracy_on_rd": summary_metrics.get("joint_accuracy_on_rd", test_metrics.get("joint_accuracy_on_rd", 0.0)),
                "macro_f1_on_rd": summary_metrics.get("joint_macro_f1_on_rd", test_metrics.get("joint_macro_f1_on_rd", 0.0)),
                "minus_traj_macro_f1_on_rd": summary_metrics.get(
                    "joint_minus_traj_macro_f1_on_rd",
                    test_metrics.get("joint_minus_traj_macro_f1_on_rd", 0.0),
                ),
            }
        rd_branch = branch_metrics.get("rd", {})
        if isinstance(rd_branch, dict) and rd_branch.get("status") == "not_applicable_tr_only":
            lines.append("## Expert1 V2 分支")
            lines.append("- joint: not_applicable_tr_only（纯 TR-only 无 TR+RD joint 分支）")
            lines.append("- RD evidence: not_applicable_tr_only（纯 TR-only 无 RD 分支）")
            rd_branch = None
        if isinstance(final_branch, dict) and isinstance(traj_branch, dict) and isinstance(rd_branch, dict):
            lines.append("## Expert1 V2 分支")
            lines.append("| 分支 | accuracy | macro_f1 |")
            lines.append("|---|---:|---:|")
            lines.append(
                f"| final | {_format_float(float(final_branch.get('accuracy', 0.0)))} | "
                f"{_format_float(float(final_branch.get('macro_f1', 0.0)))} |"
            )
            lines.append(
                f"| trajectory | {_format_float(float(traj_branch.get('accuracy', 0.0)))} | "
                f"{_format_float(float(traj_branch.get('macro_f1', 0.0)))} |"
            )
            if isinstance(joint_branch, dict) and joint_branch:
                lines.append(
                    f"| TR+RD joint | {_format_float(float(joint_branch.get('accuracy', 0.0)))} | "
                    f"{_format_float(float(joint_branch.get('macro_f1', 0.0)))} |"
                )
            lines.append(
                f"| RD evidence | {_format_float(float(rd_branch.get('accuracy', 0.0)))} | "
                f"{_format_float(float(rd_branch.get('macro_f1', 0.0)))} |"
            )
            if isinstance(joint_branch, dict) and joint_branch:
                lines.append(
                    f"- joint_on_rd: accuracy={_format_float(float(joint_branch.get('accuracy_on_rd', 0.0)))}, "
                    f"macro_f1={_format_float(float(joint_branch.get('macro_f1_on_rd', 0.0)))}, "
                    f"minus_traj_macro_f1={_format_float(float(joint_branch.get('minus_traj_macro_f1_on_rd', 0.0)))}"
                )
            if "accuracy_on_rd" in rd_branch or "macro_f1_on_rd" in rd_branch:
                lines.append(
                    f"- rd_evidence_on_rd: accuracy={_format_float(float(rd_branch.get('accuracy_on_rd', 0.0)))}, "
                    f"macro_f1={_format_float(float(rd_branch.get('macro_f1_on_rd', 0.0)))}"
                )
            lines.append("| RD evidence 贡献指标 | 值 |")
            lines.append("|---|---:|")
            for key, label in [
                ("rescue_count", "rd_rescue_count"),
                ("harm_count", "rd_harm_count"),
                ("rescue_rate_on_rd", "rd_rescue_rate_on_rd"),
                ("harm_rate_on_rd", "rd_harm_rate_on_rd"),
                ("net_gain_count", "rd_net_gain_count"),
                ("net_gain_rate_on_rd", "rd_net_gain_rate_on_rd"),
                ("rescue_unadopted_count", "rd_rescue_unadopted_count"),
                ("harm_adopted_count", "rd_harm_adopted_count"),
                ("known_confidence_mean", "known_confidence_mean"),
                ("residual_score_mean", "residual_score_mean"),
                ("residual_score_on_residual_mean", "residual_score_on_residual_mean"),
                ("traj_residual_accuracy", "traj_residual_accuracy"),
                ("residual_accuracy_on_rd", "rd_residual_accuracy_on_rd"),
                ("unknown_clutter_detail_accuracy_on_residual_rd", "rd_unknown_clutter_detail_accuracy_on_residual_rd"),
                ("clutter_morphology_accuracy_on_residual_rd", "rd_clutter_morphology_accuracy_on_residual_rd"),
                ("unknown_clutter_detail_clutter_recall", "rd_unknown_clutter_detail_clutter_recall"),
                ("clutter_morphology_clutter_recall", "rd_clutter_morphology_clutter_recall"),
            ]:
                if key not in rd_branch:
                    continue
                value = rd_branch[key]
                if isinstance(value, int):
                    lines.append(f"| {label} | {value} |")
                else:
                    lines.append(f"| {label} | {_format_float(float(value))} |")
            pair_weights = rd_branch.get("pair_fusion_weight_on_rd_by_pair", {})
            if isinstance(pair_weights, dict) and pair_weights:
                lines.append("")
                lines.append("| bounded RD residual 权重 | on-RD 平均权重 |")
                lines.append("|---|---:|")
                for pair_name, value in pair_weights.items():
                    lines.append(f"| {pair_name} | {_format_float(float(value))} |")
            hard_pair_report = rd_branch.get("hard_pair_report", {})
            if isinstance(hard_pair_report, dict) and hard_pair_report:
                lines.append("")
                lines.append("| RD hard pair | support | traj acc | rd acc | final acc | rd rescue | rd harm | rd net | final net |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
                for pair_name, item in hard_pair_report.items():
                    if not isinstance(item, dict):
                        continue
                    lines.append(
                        f"| {pair_name} | {item.get('support', 0)} | "
                        f"{_format_float(float(item.get('traj_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('rd_accuracy', 0.0)))} | "
                        f"{_format_float(float(item.get('final_accuracy', 0.0)))} | "
                        f"{item.get('rd_rescue_count', 0)} | {item.get('rd_harm_count', 0)} | "
                        f"{item.get('rd_net_gain_count', 0)} | {item.get('final_net_gain_count', 0)} |"
                    )
            lines.append("")
    lines.append("## 按来源统计")
    if isinstance(critical_metrics, dict):
        if "source_balanced_accuracy" in critical_metrics:
            lines.append(f"- source_balanced_accuracy: {_format_float(float(critical_metrics['source_balanced_accuracy']))}")
        if "min_source_accuracy" in critical_metrics:
            lines.append(f"- min_source_accuracy: {_format_float(float(critical_metrics['min_source_accuracy']))}")
        if "source_balanced_macro_f1" in critical_metrics:
            lines.append(f"- source_balanced_macro_f1: {_format_float(float(critical_metrics['source_balanced_macro_f1']))}")
        if "min_source_macro_f1" in critical_metrics:
            lines.append(f"- min_source_macro_f1: {_format_float(float(critical_metrics['min_source_macro_f1']))}")
    lines.append("| 来源 | count | final acc | final F1 | traj F1 | joint F1 | joint-on-RD F1 | RD-evidence-on-RD F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for source_name, item in by_source.items():
        lines.append(
            f"| {_escape_md_cell(source_name)} | {item['count']} | {_format_float(float(item['accuracy']))} | "
            f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
            f"{_format_float(float(item.get('traj_macro_f1', 0.0)))} | "
            f"{_format_float(float(item.get('joint_macro_f1', 0.0)))} | "
            f"{_format_optional_metric(item.get('joint_macro_f1_on_rd'))} | "
            f"{_format_optional_metric(item.get('rd_macro_f1_on_rd'))} |"
        )
    lines.append("")
    lines.append("## 来源 x 类别指标")
    lines.append("| 来源 | 类别 | support | predicted | precision | recall | f1 |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for source_name, item in by_source.items():
        per_label = item.get("per_label", {}) if isinstance(item, dict) else {}
        if not isinstance(per_label, dict):
            continue
        for label_name in label_names:
            label_item = per_label.get(label_name)
            if not isinstance(label_item, dict):
                continue
            lines.append(
                f"| {_escape_md_cell(source_name)} | {_escape_md_cell(label_name)} | {label_item.get('support', 0)} | "
                f"{label_item.get('predicted', 0)} | "
                f"{_format_float(float(label_item.get('precision', 0.0)))} | "
                f"{_format_float(float(label_item.get('recall', 0.0)))} | "
                f"{_format_float(float(label_item.get('f1', 0.0)))} |"
            )
    lines.append("")
    lines.append("## 典型错分样本（前20条）")
    if wrong_predictions:
        lines.append("| sample_id | source | 真实类别 | 预测类别 | 真实路由 | 初始路由 | 最终路由 | 回退路径 | reject reason | low reject score |")
        lines.append("|---|---|---|---|---|---|---|---|---|---:|")
        for item in wrong_predictions:
            router_route = item.get("route_name_router_pred", item["route_name_pred"])
            final_route = item.get("route_name_final", item["route_name_pred"])
            reroute_path = " -> ".join(item.get("reroute_path", []))
            reject_reason = item.get("reroute_reject_reason", "")
            low_reject_score = item.get("low_reject_score", 0.0)
            lines.append(
                f"| {_escape_md_cell(item['sample_id'])} | {_escape_md_cell(item['source_name'])} | {_escape_md_cell(item['label_name_true'])} | "
                f"{_escape_md_cell(item['label_name_pred'])} | {_escape_md_cell(item['route_name_true'])} | {_escape_md_cell(router_route)} | {_escape_md_cell(final_route)} | "
                f"{_escape_md_cell(reroute_path)} | {_escape_md_cell(reject_reason)} | {_format_float(float(low_reject_score))} |"
            )
    elif classification_metrics_status == "not_applicable_pretraining":
        lines.append("- 未执行分类评估，因此无错分样本清单。")
    else:
        scope_label = "验证集" if evaluation_scope == "validation_only" else "测试集"
        lines.append(f"- 当前{scope_label}中无错分样本。")
    lines.append("")
    return "\n".join(lines)


def write_real_experiment_report(
    output_dir: Path,
    evaluation_report: dict[str, object],
    summary: dict[str, object],
    timestamp_label: str,
) -> ReportPaths:
    reports_root = Path("reports") / "real" / timestamp_label
    reports_root.mkdir(parents=True, exist_ok=True)
    report_name = output_dir.name
    json_path = reports_root / f"{report_name}.json"
    markdown_path = reports_root / f"{report_name}.md"
    json_path.write_text(json.dumps(evaluation_report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown_text = build_markdown_report(
        evaluation_report=evaluation_report,
        summary=summary,
        title=f"实验报告：{report_name}",
    )
    markdown_path.write_text(markdown_text, encoding="utf-8")
    return ReportPaths(markdown_path=markdown_path, json_path=json_path)
