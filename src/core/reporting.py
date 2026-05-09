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
                    f"| {item.get('sample_id', '')} | {item.get('true_label', '')} | "
                    f"{item.get('without_pred', '')} | {item.get('with_pred', '')} | "
                    f"{path} | {item.get('reject_reason', '')} | "
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
                    f"| {pair_name} | {item.get('before_count', 0)} | "
                    f"{item.get('after_count', 0)} | {item.get('delta', 0)} | "
                    f"{item.get('reduced_count', 0)} | {item.get('introduced_count', 0)} |"
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
                lines.append(f"| {source_name} | {count} |")
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
        rd_branch = branch_metrics.get("rd", {})
        if isinstance(final_branch, dict) and isinstance(traj_branch, dict) and isinstance(rd_branch, dict):
            lines.append("## Expert1 RD 分支")
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
            lines.append(
                f"| rd | {_format_float(float(rd_branch.get('accuracy', 0.0)))} | "
                f"{_format_float(float(rd_branch.get('macro_f1', 0.0)))} |"
            )
            if "accuracy_on_rd" in rd_branch or "macro_f1_on_rd" in rd_branch:
                lines.append(
                    f"- rd_on_rd: accuracy={_format_float(float(rd_branch.get('accuracy_on_rd', 0.0)))}, "
                    f"macro_f1={_format_float(float(rd_branch.get('macro_f1_on_rd', 0.0)))}"
                )
            lines.append("| RD 贡献指标 | 值 |")
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
                lines.append("| RD 局部接管 gate | on-RD 平均权重 |")
                lines.append("|---|---:|")
                for pair_name, value in pair_weights.items():
                    lines.append(f"| {pair_name} | {_format_float(float(value))} |")
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
    lines.append("| 来源 | count | accuracy | macro_f1 | macro_recall |")
    lines.append("|---|---:|---:|---:|---:|")
    for source_name, item in by_source.items():
        lines.append(
            f"| {source_name} | {item['count']} | {_format_float(float(item['accuracy']))} | "
            f"{_format_float(float(item.get('macro_f1', 0.0)))} | "
            f"{_format_float(float(item.get('macro_recall', 0.0)))} |"
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
                f"| {source_name} | {label_name} | {label_item.get('support', 0)} | "
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
                f"| {item['sample_id']} | {item['source_name']} | {item['label_name_true']} | "
                f"{item['label_name_pred']} | {item['route_name_true']} | {router_route} | {final_route} | "
                f"{reroute_path} | {reject_reason} | {_format_float(float(low_reject_score))} |"
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
