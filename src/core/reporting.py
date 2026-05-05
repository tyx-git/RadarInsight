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

    top_confusions = _top_confusions(label_names, confusion, top_k=10)
    wrong_predictions = [item for item in predictions if not item["is_correct"]]
    wrong_predictions = wrong_predictions[:20]

    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## 总体指标")
    lines.append(f"- accuracy: {_format_float(float(summary_metrics['accuracy']))}")
    lines.append(f"- macro_precision: {_format_float(float(summary_metrics['macro_precision']))}")
    lines.append(f"- macro_recall: {_format_float(float(summary_metrics['macro_recall']))}")
    lines.append(f"- macro_f1: {_format_float(float(summary_metrics['macro_f1']))}")
    lines.append(f"- route_accuracy: {_format_float(float(summary_metrics['route_accuracy']))}")
    if "route_macro_recall" in summary_metrics:
        lines.append(f"- route_macro_recall: {_format_float(float(summary_metrics['route_macro_recall']))}")
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
    lines.append("## 训练产物")
    lines.append(f"- best_checkpoint: {summary['best_checkpoint']}")
    lines.append(f"- history_file: {summary['history_file']}")
    lines.append(f"- manifest_file: {summary['manifest_file']}")
    lines.append(f"- evaluation_report: {summary['evaluation_report']}")
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
            f"| {item['route_name']} | {item['support']} | {item['correct']} | {_format_float(float(item['recall']))} |"
        )
    lines.append("")
    reroute_metrics = {}
    if isinstance(critical_metrics, dict):
        candidate = critical_metrics.get("reroute", {})
        if isinstance(candidate, dict):
            reroute_metrics = candidate
    if reroute_metrics:
        lines.append("## 回流重路由")
        lines.append("| 指标 | 值 |")
        lines.append("|---|---:|")
        metric_labels = [
            ("reroute_rate", "触发率"),
            ("reroute_avg_count", "平均回流次数"),
            ("reroute_recovery_rate", "回流后正确率"),
            ("reroute_fallback_rate", "Unknown 兜底率"),
            ("reroute_worsen_rate", "回流后错误率"),
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
        lines.append("")
    lines.append("## 按来源统计")
    if isinstance(critical_metrics, dict):
        if "source_balanced_accuracy" in critical_metrics:
            lines.append(f"- source_balanced_accuracy: {_format_float(float(critical_metrics['source_balanced_accuracy']))}")
        if "min_source_accuracy" in critical_metrics:
            lines.append(f"- min_source_accuracy: {_format_float(float(critical_metrics['min_source_accuracy']))}")
    lines.append("| 来源 | count | accuracy |")
    lines.append("|---|---:|---:|")
    for source_name, item in by_source.items():
        lines.append(f"| {source_name} | {item['count']} | {_format_float(float(item['accuracy']))} |")
    lines.append("")
    lines.append("## 典型错分样本（前20条）")
    if wrong_predictions:
        lines.append("| sample_id | source | 真实类别 | 预测类别 | 真实路由 | 初始路由 | 最终路由 |")
        lines.append("|---|---|---|---|---|---|---|")
        for item in wrong_predictions:
            router_route = item.get("route_name_router_pred", item["route_name_pred"])
            final_route = item.get("route_name_final", item["route_name_pred"])
            lines.append(
                f"| {item['sample_id']} | {item['source_name']} | {item['label_name_true']} | "
                f"{item['label_name_pred']} | {item['route_name_true']} | {router_route} | {final_route} |"
            )
    else:
        lines.append("- 当前测试集中无错分样本。")
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
