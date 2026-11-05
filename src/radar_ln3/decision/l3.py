"""L3 semantic decisions and calibration for frozen L1/L2 endpoints."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from radar_ln3.checkpoint import canonical_hash
from radar_ln3.data.paths import resolve_project_path

L1_CLASSES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
L2_CLASSES = ("DroneTarget", "BirdTarget", "BalloonTarget")
KNOWN_LABELS = (*L2_CLASSES, "ClutterTarget")
UNKNOWN_LABEL = "UnknownTarget"
L3_ACTIONS = (*L2_CLASSES, "Clutter", "UNKNOWN_REJECTED", "PENDING_ABSTAIN")
L3_CALIBRATION_SCHEMA = "RADAR-LN3-L3-CALIBRATION-V3"
L3_CLASS_TO_ACTION = {
    "DroneTarget": "DroneTarget",
    "BirdTarget": "BirdTarget",
    "BalloonTarget": "BalloonTarget",
    "ClutterTarget": "Clutter",
}


@dataclass(frozen=True)
class L3CalibrationPolicy:
    """阈值搜索约束；拟合时会完整复制到 calibration artifact。"""

    known_acceptance_floor: Mapping[str, float] = field(
        default_factory=lambda: {
            "DroneTarget": 0.95,
            "BirdTarget": 0.95,
            "BalloonTarget": 0.95,
            "ClutterTarget": 0.8888888888888888,
        }
    )
    conformal_error_rate: float = 0.1
    minimum_class_support: int = 8
    minimum_unknown_support: int = 15
    minimum_audit_unknown_support: int = 16
    maximum_unknown_false_acceptance_rate: float = 0.1
    maximum_known_selective_risk: float = 0.15
    maximum_audit_unknown_false_acceptance_rate: float = 0.05
    maximum_audit_known_selective_risk: float = 0.1
    threshold_grid_size: int = 101
    temperature_grid_size: int = 81
    temperature_min: float = 0.05
    temperature_max: float = 20.0

    def __post_init__(self) -> None:
        floors = dict(self.known_acceptance_floor)
        if set(floors) != set(KNOWN_LABELS):
            raise ValueError("L3 known_acceptance_floor 必须覆盖四个已知标签")
        if any(
            not math.isfinite(float(value)) or not 0.0 < float(value) <= 1.0
            for value in floors.values()
        ):
            raise ValueError("L3 known acceptance floor 必须位于 (0,1]")
        integer_fields = (
            ("minimum_class_support", self.minimum_class_support),
            ("minimum_unknown_support", self.minimum_unknown_support),
            ("minimum_audit_unknown_support", self.minimum_audit_unknown_support),
            ("threshold_grid_size", self.threshold_grid_size),
            ("temperature_grid_size", self.temperature_grid_size),
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            for _, value in integer_fields
        ):
            raise ValueError("L3 event count、support 和 grid 参数必须是整数")
        for name in (
            "conformal_error_rate",
            "maximum_unknown_false_acceptance_rate",
            "maximum_known_selective_risk",
            "maximum_audit_unknown_false_acceptance_rate",
            "maximum_audit_known_selective_risk",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"L3 {name} 必须位于 (0,1)")
        if (
            int(self.minimum_class_support) < 1
            or int(self.minimum_unknown_support) < 1
            or int(self.minimum_audit_unknown_support) < int(self.minimum_unknown_support)
        ):
            raise ValueError("L3 最小支持数必须为正整数")
        if int(self.threshold_grid_size) < 2 or int(self.temperature_grid_size) < 2:
            raise ValueError("L3 校准网格至少需要两个点")
        if (
            not math.isfinite(float(self.temperature_min))
            or not math.isfinite(float(self.temperature_max))
            or self.temperature_min <= 0.0
            or self.temperature_max <= self.temperature_min
        ):
            raise ValueError("L3 temperature 搜索范围无效")

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "known_acceptance_floor": {
                name: float(self.known_acceptance_floor[name])
                for name in KNOWN_LABELS
            },
        }


def _softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    shifted = values / temperature
    shifted -= shifted.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _fit_temperature(
    logits: np.ndarray,
    label_ids: np.ndarray,
    policy: L3CalibrationPolicy,
) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(label_ids, dtype=np.int64)
    if not len(labels):
        raise ValueError("L3 temperature 拟合缺少已知类样本")
    grid = np.exp(
        np.linspace(
            math.log(policy.temperature_min),
            math.log(policy.temperature_max),
            int(policy.temperature_grid_size),
        )
    )
    row_indices = np.arange(len(labels))
    choices: list[tuple[float, float]] = []
    for temperature in grid:
        probabilities = _softmax(values, float(temperature))
        loss = float(
            -np.log(
                np.clip(
                    probabilities[row_indices, labels],
                    np.finfo(np.float64).tiny,
                    1.0,
                )
            ).mean()
        )
        choices.append((loss, float(temperature)))
    loss, selected = min(choices)
    return {
        "status": "fitted",
        "temperature": selected,
        "nll": loss,
        "support": int(len(labels)),
        "grid_size": int(policy.temperature_grid_size),
        "search_range": [float(policy.temperature_min), float(policy.temperature_max)],
    }


def _fit_conformal_floors(
    probabilities: np.ndarray,
    label_ids: np.ndarray,
    class_names: Sequence[str],
    policy: L3CalibrationPolicy,
) -> dict[str, Any]:
    values = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(label_ids, dtype=np.int64)
    floors: dict[str, float | None] = {}
    quantiles: dict[str, float | None] = {}
    supports: dict[str, int] = {}
    for class_id, class_name in enumerate(class_names):
        selected = values[labels == class_id, class_id]
        support = int(selected.size)
        supports[str(class_name)] = support
        if support < int(policy.minimum_class_support):
            floors[str(class_name)] = None
            quantiles[str(class_name)] = None
            continue
        nonconformity = 1.0 - selected
        rank = min(
            support,
            max(
                1,
                int(
                    math.ceil(
                        (support + 1)
                        * (1.0 - float(policy.conformal_error_rate))
                    )
                ),
            ),
        )
        quantile = float(np.sort(nonconformity)[rank - 1])
        floors[str(class_name)] = float(1.0 - quantile)
        quantiles[str(class_name)] = quantile
    return {
        "method": "class_conditional_split_conformal_1_minus_probability",
        "error_rate": float(policy.conformal_error_rate),
        "candidate_probability_floor_by_class": floors,
        "nonconformity_quantile_by_class": quantiles,
        "support_by_class": supports,
        "minimum_class_support": int(policy.minimum_class_support),
    }


def _threshold_candidates(scores: np.ndarray, count: int) -> list[float]:
    values = np.asarray(scores, dtype=np.float64)
    unique = np.unique(values)
    if unique.size > int(count):
        quantiles = np.linspace(0.0, 1.0, int(count))
        unique = np.unique(np.quantile(unique, quantiles, method="nearest"))
    candidates = {0.0, 1.0, *(float(value) for value in unique)}
    candidates.update(
        float(np.nextafter(value, 1.0))
        for value in unique
        if float(value) < 1.0
    )
    return sorted(value for value in candidates if 0.0 <= value <= 1.0)


def _wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "support": 0,
        }
    estimate = float(successes) / float(total)
    z = 1.959963984540054
    denominator = 1.0 + z * z / float(total)
    center = (estimate + z * z / (2.0 * float(total))) / denominator
    radius = z * math.sqrt(
        estimate * (1.0 - estimate) / float(total)
        + z * z / (4.0 * float(total) * float(total))
    ) / denominator
    return {
        "estimate": estimate,
        "lower_95": max(0.0, center - radius),
        "upper_95": min(1.0, center + radius),
        "support": int(total),
    }


def _fit_knownness_threshold(
    probabilities: np.ndarray,
    *,
    class_id: int,
    score_values: np.ndarray | None = None,
    positive: np.ndarray,
    unknown: np.ndarray,
    recall_floor: float,
    policy: L3CalibrationPolicy,
) -> dict[str, Any]:
    scores = (
        np.asarray(probabilities, dtype=np.float64)[:, int(class_id)]
        if score_values is None
        else np.asarray(score_values, dtype=np.float64)
    )
    positive_support = int(positive.sum())
    unknown_support = int(unknown.sum())
    candidates: list[dict[str, Any]] = []
    for threshold in _threshold_candidates(scores, policy.threshold_grid_size):
        accepted = scores >= threshold
        recall = float(accepted[positive].mean()) if positive_support else 0.0
        false_acceptance = (
            float(accepted[unknown].mean()) if unknown_support else None
        )
        recall_interval = _wilson(int(accepted[positive].sum()), positive_support)
        false_acceptance_interval = _wilson(
            int(accepted[unknown].sum()), unknown_support
        )
        safe = bool(
            positive_support >= int(policy.minimum_class_support)
            and unknown_support > 0
            and recall_interval["lower_95"] >= float(recall_floor)
            and false_acceptance_interval["upper_95"]
            <= float(policy.maximum_unknown_false_acceptance_rate)
        )
        candidates.append(
            {
                "threshold": float(threshold),
                "known_recall": recall,
                "known_recall_interval": recall_interval,
                "unknown_false_acceptance_rate": false_acceptance,
                "unknown_false_acceptance_interval": false_acceptance_interval,
                "positive_support": positive_support,
                "unknown_support": unknown_support,
                "safe": safe,
            }
        )
    safe = [row for row in candidates if row["safe"]]
    if not safe:
        # 约束不可行时，先按点估计的 Unknown 误接受上限筛选，再最大化
        # 已知覆盖。该 operating point 仍明确标记为 fallback_infeasible，
        # 只用于诊断，不得绕过正式 audit 部署门槛。
        pointwise_allowed = [
            row
            for row in candidates
            if row["unknown_false_acceptance_rate"] is None
            or float(row["unknown_false_acceptance_rate"])
            <= float(policy.maximum_unknown_false_acceptance_rate)
        ]
        pool = pointwise_allowed or candidates
        diagnostic = max(
            pool,
            key=lambda row: (
                float(row["known_recall"]),
                -float(
                    1.0
                    if row["unknown_false_acceptance_rate"] is None
                    else row["unknown_false_acceptance_rate"]
                ),
                -float(row["threshold"]),
            ),
        )
        return {
            "status": "fallback_infeasible",
            "selected": diagnostic,
            "candidate_count": len(candidates),
            "positive_support": positive_support,
            "unknown_support": unknown_support,
            "minimum_unknown_support": int(policy.minimum_unknown_support),
            "support_sufficient": unknown_support >= policy.minimum_unknown_support,
        }
    selected = max(
        safe,
        key=lambda row: (
            float(row["known_recall"]),
            -float(
                1.0
                if row["unknown_false_acceptance_rate"] is None
                else row["unknown_false_acceptance_rate"]
            ),
            -float(row["threshold"]),
        ),
    )
    return {
        "status": "fitted",
        "selected": selected,
        "candidate_count": len(candidates),
        "positive_support": positive_support,
        "unknown_support": unknown_support,
        "minimum_unknown_support": int(policy.minimum_unknown_support),
        "support_sufficient": unknown_support >= policy.minimum_unknown_support,
        "recall_floor": float(recall_floor),
        "maximum_unknown_false_acceptance_rate": float(
            policy.maximum_unknown_false_acceptance_rate
        ),
    }


def _candidate_set(
    probability: np.ndarray,
    class_names: Sequence[str],
    floors: Mapping[str, Any],
) -> list[str]:
    return [
        str(name)
        for index, name in enumerate(class_names)
        if floors.get(str(name)) is not None
        and float(probability[index]) + 1.0e-12
        >= float(floors[str(name)])
    ]


def _fit_l2_margin_thresholds(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """为每个 L2 route 保存 validation true-class 的保守 top-2 margin floor。"""

    values = np.asarray(probabilities, dtype=np.float64)
    result: dict[str, float] = {}
    known_mask = np.isin(labels, L2_CLASSES)
    for class_id, class_name in enumerate(L2_CLASSES):
        rows = values[known_mask & (labels == class_name)]
        if len(rows) == 0:
            result[class_name] = 1.0
            continue
        ordered = np.sort(rows, axis=1)[:, ::-1]
        margins = ordered[:, 0] - ordered[:, 1]
        result[class_name] = float(np.clip(np.quantile(margins, 0.1), 0.0, 1.0))
    return result


def _base_action(
    index: int,
    *,
    l1_probabilities: np.ndarray,
    l2_probabilities: np.ndarray,
    l1_floors: Mapping[str, Any],
    l2_floors: Mapping[str, Any],
    l1_knownness: Mapping[str, Any],
    l2_knownness: Mapping[str, Any],
    l1_knownness_scores: np.ndarray,
    l2_knownness_scores: np.ndarray,
    l2_margin_thresholds: Mapping[str, float] | None = None,
) -> str:
    l1_prediction = L1_CLASSES[int(np.argmax(l1_probabilities[index]))]
    l1_route = "target" if l1_prediction == L1_CLASSES[0] else "clutter"
    l1_threshold = l1_knownness[l1_route].get("selected")
    l1_score = float(l1_knownness_scores[index, 0 if l1_route == "target" else 1])
    if l1_threshold is None or l1_score < float(l1_threshold["threshold"]):
        return "UNKNOWN_REJECTED"
    l1_candidates = _candidate_set(l1_probabilities[index], L1_CLASSES, l1_floors)
    if not l1_candidates:
        return "UNKNOWN_REJECTED"
    if len(l1_candidates) != 1 or l1_candidates[0] != l1_prediction:
        return "PENDING_ABSTAIN"
    if l1_route == "clutter":
        return "Clutter"
    l2_prediction = L2_CLASSES[int(np.argmax(l2_probabilities[index]))]
    l2_threshold = l2_knownness[l2_prediction].get("selected")
    l2_score = float(l2_knownness_scores[index, L2_CLASSES.index(l2_prediction)])
    if l2_threshold is None or l2_score < float(l2_threshold["threshold"]):
        return "UNKNOWN_REJECTED"
    l2_candidates = _candidate_set(l2_probabilities[index], L2_CLASSES, l2_floors)
    if not l2_candidates:
        return "UNKNOWN_REJECTED"
    if len(l2_candidates) != 1:
        ordered = np.sort(l2_probabilities[index])[::-1]
        margin = float(ordered[0] - ordered[1])
        threshold = float((l2_margin_thresholds or {}).get(l2_prediction, 1.0))
        if l2_prediction in l2_candidates and margin >= threshold:
            return l2_prediction
        return "UNKNOWN_REJECTED"
    if l2_candidates[0] != l2_prediction:
        return "PENDING_ABSTAIN"
    return l2_prediction


def _audit_metrics(
    actions: np.ndarray,
    labels: np.ndarray,
    *,
    policy: L3CalibrationPolicy,
) -> dict[str, Any]:
    known_mask = np.isin(labels, KNOWN_LABELS)
    unknown_mask = labels == UNKNOWN_LABEL
    committed_actions = np.isin(actions, (*L2_CLASSES, "Clutter"))
    known_accepted = known_mask & committed_actions
    expected_actions = np.asarray(
        [L3_CLASS_TO_ACTION.get(str(label), "UNKNOWN_REJECTED") for label in labels],
        dtype=object,
    )
    known_correct = known_accepted & (actions == expected_actions)
    known_errors = known_accepted & ~known_correct
    accepted_count = int(known_accepted.sum())
    known_support = int(known_mask.sum())
    rejected_count = int(known_support - accepted_count)
    unknown_support = int(unknown_mask.sum())
    unknown_false_accept = int(
        (
            unknown_mask
            & np.isin(actions, (*L2_CLASSES, "Clutter"))
        ).sum()
    )
    recall_by_label: dict[str, float | None] = {}
    recall_interval_by_label: dict[str, dict[str, Any]] = {}
    precision_by_label: dict[str, float | None] = {}
    precision_interval_by_label: dict[str, dict[str, Any]] = {}
    coverage_by_label: dict[str, float | None] = {}
    coverage_interval_by_label: dict[str, dict[str, Any]] = {}
    support_by_label: dict[str, int] = {}
    for label in KNOWN_LABELS:
        mask = labels == label
        support = int(mask.sum())
        accepted = int(known_accepted[mask].sum())
        correct = int(known_correct[mask].sum())
        support_by_label[label] = support
        recall_by_label[label] = (
            float(correct / support) if support else None
        )
        recall_interval_by_label[label] = _wilson(
            correct, support
        )
        precision_by_label[label] = (
            float(correct / accepted) if accepted else None
        )
        precision_interval_by_label[label] = _wilson(correct, accepted)
        coverage_by_label[label] = (
            float(accepted / support) if support else None
        )
        coverage_interval_by_label[label] = _wilson(accepted, support)
    selective_risk = (
        float(known_errors.sum() / accepted_count) if accepted_count else None
    )
    accepted_precision = (
        float(known_correct.sum() / accepted_count) if accepted_count else None
    )
    known_coverage = (
        float(accepted_count / known_support) if known_support else None
    )
    unknown_far = (
        float(unknown_false_accept / unknown_support) if unknown_support else None
    )
    unknown_far_interval = _wilson(unknown_false_accept, unknown_support)
    selective_risk_interval = _wilson(int(known_errors.sum()), accepted_count)
    support_sufficient = (
        unknown_support >= int(policy.minimum_unknown_support)
        and all(
            support_by_label[label] >= int(policy.minimum_class_support)
            for label in KNOWN_LABELS
        )
    )
    floors_met = all(
        recall_interval_by_label[label]["lower_95"] is not None
        and float(recall_interval_by_label[label]["lower_95"])
        >= float(policy.known_acceptance_floor[label])
        for label in KNOWN_LABELS
    )
    safe = bool(
        support_sufficient
        and floors_met
        and unknown_far_interval["upper_95"] is not None
        and unknown_far_interval["upper_95"]
        <= float(policy.maximum_unknown_false_acceptance_rate)
        and selective_risk_interval["upper_95"] is not None
        and selective_risk_interval["upper_95"]
        <= float(policy.maximum_known_selective_risk)
    )
    return {
        "safe": safe,
        "support_sufficient": support_sufficient,
        "support_by_label": support_by_label,
        "known_correct_acceptance_by_label": recall_by_label,
        "known_correct_acceptance_interval_by_label": recall_interval_by_label,
        "known_accepted_precision_by_label": precision_by_label,
        "known_accepted_precision_interval_by_label": precision_interval_by_label,
        "known_coverage_by_label": coverage_by_label,
        "known_coverage_interval_by_label": coverage_interval_by_label,
        "known_accepted_support": accepted_count,
        "known_rejected_support": rejected_count,
        "known_support": known_support,
        "known_accepted_precision": accepted_precision,
        "known_coverage": known_coverage,
        "unknown_support": unknown_support,
        "unknown_false_acceptance_count": unknown_false_accept,
        "unknown_false_acceptance_rate": unknown_far,
        "unknown_false_acceptance_interval": unknown_far_interval,
        "known_selective_risk": selective_risk,
        "known_selective_risk_interval": selective_risk_interval,
        "pending_count": int((actions == "PENDING_ABSTAIN").sum()),
        "coverage": float((actions != "PENDING_ABSTAIN").mean()),
    }


def _probability_for_row(
    logits: Sequence[float],
    temperature: float,
    class_count: int,
) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (class_count,) or not np.isfinite(values).all():
        raise ValueError("L3 logits shape 或数值无效")
    if not math.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("L3 temperature 必须为正有限值")
    return _softmax(values[None, :], temperature)[0]


def _validate_calibration(calibration: Mapping[str, Any]) -> dict[str, Any]:
    schema = calibration.get("schema_version")
    if schema != L3_CALIBRATION_SCHEMA:
        raise ValueError("L3 calibration schema_version 不匹配")
    body = {key: value for key, value in calibration.items() if key != "artifact_sha256"}
    if calibration.get("artifact_sha256") != canonical_hash(body):
        raise ValueError("L3 calibration artifact hash 不匹配")
    if calibration.get("status") not in {"exploratory", "unavailable", "fitted", "audit_failed", "formal_available"}:
        raise ValueError("L3 calibration status 无效")
    value = dict(calibration)
    knownness_source = str(value.get("knownness_source") or "")
    if knownness_source != "independent_l3_head":
        raise ValueError("L3 calibration 必须绑定独立 knownness heads")
    heads = value.get("head_checkpoint_sha256")
    if (
        not isinstance(heads, Mapping)
        or set(heads) != {"l1", "l2"}
        or any(
            len(str(item)) != 64
            or set(str(item).lower()) - set("0123456789abcdef")
            for item in heads.values()
        )
    ):
        raise ValueError("L3 calibration knownness head hash binding 无效")
    l1_router_source = str(value.get("l1_router_source", "classifier_endpoint"))
    if l1_router_source not in {"classifier_endpoint", "independent_l3_head"}:
        raise ValueError("L3 calibration l1_router_source 无效")
    l2_threshold_mode = str(value.get("l2_knownness_threshold_mode", "shared"))
    if l2_threshold_mode not in {"shared", "per_class"}:
        raise ValueError("L3 calibration l2_knownness_threshold_mode 无效")
    source_hash = str(value.get("source_manifest_sha256", "")).lower()
    source_role = str(value.get("source_role", ""))
    if (
        len(source_hash) != 64
        or set(source_hash) - set("0123456789abcdef")
        or source_role not in {"train_group_oof", "calibration", "exploratory_validation"}
    ):
        raise ValueError("L3 calibration source provenance 无效")
    policy_payload = value.get("policy")
    if not isinstance(policy_payload, Mapping):
        raise ValueError("L3 calibration policy 缺失")
    policy = L3CalibrationPolicy(**dict(policy_payload))
    if policy.to_dict() != dict(policy_payload):
        raise ValueError("L3 calibration policy contract 不一致")
    parent_ids = value.get("parent_ids")
    if (
        not isinstance(parent_ids, list)
        or not parent_ids
        or len(parent_ids) != len(set(str(item) for item in parent_ids))
        or value.get("parent_ids_sha256")
        != canonical_hash(sorted(str(item) for item in parent_ids))
    ):
        raise ValueError("L3 calibration parent binding 无效")
    endpoints = value.get("endpoint_sha256")
    if (
        not isinstance(endpoints, Mapping)
        or set(endpoints) != {"l1", "l2"}
        or any(
            len(str(digest)) != 64
            or set(str(digest).lower()) - set("0123456789abcdef")
            for digest in endpoints.values()
        )
    ):
        raise ValueError("L3 calibration endpoint hash binding 无效")
    for section in ("temperature",):
        values = value.get(section)
        if not isinstance(values, Mapping) or set(values) != {"l1", "l2"}:
            raise ValueError(f"L3 calibration {section} contract 无效")
        for task in ("l1", "l2"):
            item = values[task]
            if not isinstance(item, Mapping):
                raise ValueError(f"L3 calibration temperature 无效：{task}")
            temperature = float(item.get("temperature", float("nan")))
            if (
                not math.isfinite(temperature)
                or temperature <= 0.0
                or int(item.get("support", 0)) <= 0
            ):
                raise ValueError(f"L3 calibration temperature 无效：{task}")
    conformal = value.get("conformal")
    if not isinstance(conformal, Mapping) or set(conformal) != {"l1", "l2"}:
        raise ValueError("L3 calibration conformal contract 无效")
    for task, class_names in (("l1", L1_CLASSES), ("l2", L2_CLASSES)):
        section = conformal[task]
        floors = section.get("candidate_probability_floor_by_class")
        if not isinstance(floors, Mapping) or set(floors) != set(class_names):
            raise ValueError(f"L3 calibration conformal floors 无效：{task}")
        for class_name, floor in floors.items():
            if floor is not None and (
                not math.isfinite(float(floor)) or not 0.0 <= float(floor) <= 1.0
            ):
                raise ValueError(f"L3 calibration conformal floor 越界：{task}/{class_name}")
    scorer = value.get("l2_scorer")
    if scorer is not None:
        from radar_ln3.decision.scorer import validate_scorer_descriptor

        validate_scorer_descriptor(scorer)
    knownness = value.get("knownness_thresholds")
    if not isinstance(knownness, Mapping) or set(knownness) != {"l1", "l2"}:
        raise ValueError("L3 calibration knownness contract 无效")
    route_contract = (("l1", ("target", "clutter")), ("l2", L2_CLASSES))
    for task, routes in route_contract:
        section = knownness[task]
        if not isinstance(section, Mapping) or set(section) != set(routes):
            raise ValueError(f"L3 calibration knownness routes 无效：{task}")
        for route in routes:
            selected = section[route].get("selected")
            if selected is not None:
                threshold = float(selected.get("threshold", float("nan")))
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"L3 knownness threshold 越界：{task}/{route}")
    margins = value.get("l2_top2_margin_thresholds")
    if margins is not None:
        if not isinstance(margins, Mapping) or set(margins) != set(L2_CLASSES):
            raise ValueError("L3 L2 top2 margin thresholds contract 无效")
        if any(
            not math.isfinite(float(item)) or not 0.0 <= float(item) <= 1.0
            for item in margins.values()
        ):
            raise ValueError("L3 L2 top2 margin threshold 越界")
    bird_gate = value.get("bird_recovery_gate")
    if bird_gate is not None:
        if not isinstance(bird_gate, Mapping):
            raise ValueError("L3 bird_recovery_gate 必须是对象")
        if not isinstance(bird_gate.get("enabled"), bool):
            raise ValueError("L3 bird_recovery_gate.enabled 必须是布尔值")
        if bird_gate.get("enabled"):
            for key in ("probability_floor", "margin_floor"):
                threshold = float(bird_gate.get(key, float("nan")))
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"L3 bird_recovery_gate {key} 越界")
    clutter_bird_gate = value.get("l1_clutter_bird_recovery_gate")
    if clutter_bird_gate is not None:
        if not isinstance(clutter_bird_gate, Mapping):
            raise ValueError("L3 l1_clutter_bird_recovery_gate 必须是对象")
        if not isinstance(clutter_bird_gate.get("enabled"), bool):
            raise ValueError("L3 l1_clutter_bird_recovery_gate.enabled 必须是布尔值")
        if clutter_bird_gate.get("enabled"):
            for key in (
                "l1_target_score_min",
                "l1_target_score_max",
                "l2_bird_probability_floor",
                "l2_bird_knownness_threshold",
            ):
                threshold = float(clutter_bird_gate.get(key, float("nan")))
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"L3 l1_clutter_bird_recovery_gate {key} 越界")
            if float(clutter_bird_gate["l1_target_score_min"]) > float(
                clutter_bird_gate["l1_target_score_max"]
            ):
                raise ValueError("L3 l1_clutter_bird_recovery_gate target score 区间无效")
            if str(bird_gate.get("knownness_route", "BirdTarget")) != "BirdTarget":
                raise ValueError("L3 bird_recovery_gate 只允许 BirdTarget route")
            for key in ("knownness_threshold",):
                threshold = float(bird_gate.get(key, float("nan")))
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"L3 bird_recovery_gate {key} 越界")
    target_bird_gate = value.get("l1_target_bird_recovery_gate")
    if target_bird_gate is not None:
        if not isinstance(target_bird_gate, Mapping):
            raise ValueError("L3 l1_target_bird_recovery_gate 必须是对象")
        if not isinstance(target_bird_gate.get("enabled"), bool):
            raise ValueError("L3 l1_target_bird_recovery_gate.enabled 必须是布尔值")
        if target_bird_gate.get("enabled"):
            for key in (
                "l1_target_score_min",
                "l1_target_score_max",
                "l2_bird_probability_floor",
                "l2_bird_knownness_threshold",
                "l2_bird_margin_floor",
            ):
                threshold = float(target_bird_gate.get(key, float("nan")))
                if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                    raise ValueError(f"L3 l1_target_bird_recovery_gate {key} 越界")
            if float(target_bird_gate["l1_target_score_min"]) > float(
                target_bird_gate["l1_target_score_max"]
            ):
                raise ValueError("L3 l1_target_bird_recovery_gate target score 区间无效")
    strategy = value.get("l3_2_strategy")
    if strategy is not None:
        if not isinstance(strategy, Mapping) or strategy.get("name") != "bird_recovery_v1":
            raise ValueError("L3 l3_2_strategy contract 无效")
        threshold = float(strategy.get("l1_target_threshold", float("nan")))
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("L3_2 L1 target threshold 越界")
    target_override = value.get("l1_target_knownness_override")
    if target_override is not None:
        if not isinstance(target_override, Mapping) or strategy is not None:
            raise ValueError("L3 L1 target knownness override contract 无效")
        threshold = float(target_override.get("threshold", float("nan")))
        digest = str(target_override.get("fit_artifact_sha256", "")).lower()
        if (
            not math.isfinite(threshold)
            or not 0.0 <= threshold <= 1.0
            or target_override.get("source") != "validation_l1_knownness_fit"
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError("L3 L1 target knownness override 无效")
    conformal_override = value.get("l1_target_conformal_floor_override")
    if conformal_override is not None:
        if not isinstance(conformal_override, Mapping):
            raise ValueError("L3 L1 target conformal floor override contract 无效")
        floor = float(conformal_override.get("floor", float("nan")))
        digest = str(conformal_override.get("fit_artifact_sha256", "")).lower()
        if (
            not math.isfinite(floor)
            or not 0.0 <= floor <= 1.0
            or conformal_override.get("source") != "validation_l1_knownness_fit"
            or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError("L3 L1 target conformal floor override 无效")
        base_floor = value["conformal"]["l1"][
            "candidate_probability_floor_by_class"
        ]["KNOWN_TARGET_EVIDENCE"]
        if base_floor is None or floor > float(base_floor) + 1.0e-12:
            raise ValueError("L3 L1 target conformal floor override 只能降低原始 floor")
    if value.get("status") == "formal_available":
        if value.get("deployment_eligible") is not True:
            raise ValueError("formal_available calibration must be deployment eligible")
    elif bool(value.get("deployment_eligible")) != (value.get("status") == "fitted"):
        raise ValueError("L3 deployment eligibility/status 不一致")
    if int(value.get("test_payload_access_count", -1)) != 0:
        raise ValueError("L3 calibration artifact 已读取 test payload")
    if value.get("status") == "fitted":
        audit = value.get("audit")
        if (
            not isinstance(audit, Mapping)
            or audit.get("passed") is not True
            or audit.get("role") != "post_selection_audit"
            or audit.get("thresholds_frozen_before_access") is not True
            or audit.get("source_manifest_sha256") == source_hash
        ):
            raise ValueError("L3 fitted artifact 缺少独立 audit binding")
    return value


def _seal_calibration(body: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(body)
    return {**value, "artifact_sha256": canonical_hash(value)}


def fit_l3_calibration(
    *,
    parent_ids: Sequence[str],
    labels: Sequence[str],
    l1_logits: np.ndarray,
    l2_logits: np.ndarray,
    source_manifest_sha256: str,
    endpoint_sha256: Mapping[str, str],
    source_role: str,
    policy: L3CalibrationPolicy,
    l1_knownness_scores: np.ndarray | None = None,
    l2_knownness_scores_by_route: np.ndarray | None = None,
    head_checkpoint_sha256: Mapping[str, str] | None = None,
    l1_router_from_head: bool = False,
    l2_knownness_threshold_mode: str = "shared",
) -> dict[str, Any]:
    """拟合探索性温度、共形候选集和 knownness 阈值。"""

    ids = tuple(str(value) for value in parent_ids)
    names = tuple(str(value) for value in labels)
    l1_values = np.asarray(l1_logits, dtype=np.float64)
    l2_values = np.asarray(l2_logits, dtype=np.float64)
    count = len(ids)
    if count == 0 or len(set(ids)) != count:
        raise ValueError("L3 parent ids 必须非空且唯一")
    if len(names) != count:
        raise ValueError("L3 parent 与标签长度不一致")
    if set(names) - {*KNOWN_LABELS, UNKNOWN_LABEL}:
        raise ValueError("L3 校准标签超出语义决策合同")
    if l1_values.shape != (count, len(L1_CLASSES)) or l2_values.shape != (count, len(L2_CLASSES)):
        raise ValueError("L3 L1/L2 logits shape 不一致")
    if not np.isfinite(l1_values).all() or not np.isfinite(l2_values).all():
        raise ValueError("L3 logits 必须有限")
    l1_head_scores = None if l1_knownness_scores is None else np.asarray(l1_knownness_scores, dtype=np.float64)
    l2_head_scores = None if l2_knownness_scores_by_route is None else np.asarray(l2_knownness_scores_by_route, dtype=np.float64)
    if l1_head_scores is None or l2_head_scores is None:
        raise ValueError("L3 calibration 必须使用独立 L1/L2 knownness heads")
    if l1_head_scores.shape != (count, 2):
        raise ValueError("L3 L1 knownness head score shape 不一致")
    if l2_head_scores.shape != (count, len(L2_CLASSES)):
        raise ValueError("L3 L2 knownness head score shape 不一致")
    if any(
        not np.isfinite(value).all() or np.any((value < 0.0) | (value > 1.0))
        for value in (l1_head_scores, l2_head_scores)
    ):
        raise ValueError("L3 knownness head score 必须是 [0,1] 内有限值")
    if l2_knownness_threshold_mode not in {"shared", "per_class"}:
        raise ValueError("l2_knownness_threshold_mode 必须为 shared 或 per_class")
    if not isinstance(head_checkpoint_sha256, Mapping) or set(head_checkpoint_sha256) != {"l1", "l2"}:
        raise ValueError("L3 knownness head checkpoint hash binding 缺失")
    if any(
        len(str(value)) != 64 or set(str(value).lower()) - set("0123456789abcdef")
        for value in head_checkpoint_sha256.values()
    ):
        raise ValueError("L3 knownness head checkpoint hash binding 无效")
    digest_pattern = set("0123456789abcdef")
    if (
        len(source_manifest_sha256) != 64
        or set(source_manifest_sha256.lower()) - digest_pattern
        or set(endpoint_sha256) != {"l1", "l2"}
        or any(
            len(str(value)) != 64 or set(str(value).lower()) - digest_pattern
            for value in endpoint_sha256.values()
        )
    ):
        raise ValueError("L3 source 或 endpoint SHA-256 binding 无效")
    if source_role not in {"train_group_oof", "calibration", "exploratory_validation"}:
        raise ValueError("L3 threshold fitting role 不允许使用 test/audit")

    label_array = np.asarray(names, dtype=object)
    l1_known_mask = np.isin(label_array, KNOWN_LABELS)
    l2_known_mask = np.isin(label_array, L2_CLASSES)
    l1_label_ids = np.asarray(
        [0 if label in L2_CLASSES else 1 for label in label_array[l1_known_mask]],
        dtype=np.int64,
    )
    l2_label_ids = np.asarray(
        [L2_CLASSES.index(label) for label in label_array[l2_known_mask]],
        dtype=np.int64,
    )
    if not l1_known_mask.any() or not l2_known_mask.any():
        raise ValueError("L3 calibration 缺少已知类样本")
    l1_temperature = _fit_temperature(l1_values[l1_known_mask], l1_label_ids, policy)
    l2_temperature = _fit_temperature(l2_values[l2_known_mask], l2_label_ids, policy)
    l1_probabilities = _softmax(l1_values, float(l1_temperature["temperature"]))
    l2_probabilities = _softmax(l2_values, float(l2_temperature["temperature"]))
    if l1_router_from_head:
        route_totals = np.asarray(l1_head_scores, dtype=np.float64).sum(axis=1, keepdims=True)
        if np.any(route_totals <= 1.0e-12):
            raise ValueError("L1 route head score 总和必须为正")
        l1_decision_probabilities = np.asarray(l1_head_scores, dtype=np.float64) / route_totals
    else:
        l1_decision_probabilities = l1_probabilities
    l2_margin_thresholds = _fit_l2_margin_thresholds(l2_probabilities, label_array)
    l1_conformal = _fit_conformal_floors(
        l1_decision_probabilities[l1_known_mask], l1_label_ids, L1_CLASSES, policy
    )
    l2_conformal = _fit_conformal_floors(
        l2_probabilities[l2_known_mask], l2_label_ids, L2_CLASSES, policy
    )
    unknown = label_array == UNKNOWN_LABEL
    l1_knownness = {
        "target": _fit_knownness_threshold(
            l1_probabilities,
            class_id=0,
            positive=np.isin(label_array, L2_CLASSES),
            unknown=unknown,
            recall_floor=min(
                float(policy.known_acceptance_floor[label]) for label in L2_CLASSES
            ),
            policy=policy,
            score_values=l1_head_scores[:, 0],
        ),
        "clutter": _fit_knownness_threshold(
            l1_probabilities,
            class_id=1,
            positive=label_array == "ClutterTarget",
            unknown=unknown,
            recall_floor=float(policy.known_acceptance_floor["ClutterTarget"]),
            policy=policy,
            score_values=l1_head_scores[:, 1],
        ),
    }
    l2_knownness = {
        class_name: _fit_knownness_threshold(
            l2_probabilities,
            class_id=class_id,
            positive=label_array == class_name,
            unknown=unknown,
            recall_floor=float(policy.known_acceptance_floor[class_name]),
            policy=policy,
            score_values=l2_head_scores[:, class_id],
        )
        for class_id, class_name in enumerate(L2_CLASSES)
    }
    # L2 使用一个覆盖三个已知目标类别的 shared knownness gate。
    if l2_knownness_threshold_mode == "shared":
        if np.allclose(l2_head_scores, l2_head_scores[:, :1], atol=1.0e-7, rtol=0.0):
            shared_l2 = _fit_knownness_threshold(
                l2_probabilities,
                class_id=0,
                positive=l2_known_mask,
                unknown=unknown,
                recall_floor=min(
                    float(policy.known_acceptance_floor[label]) for label in L2_CLASSES
                ),
                policy=policy,
                score_values=l2_head_scores[:, 0],
            )
            l2_knownness = {class_name: shared_l2 for class_name in L2_CLASSES}
    base_actions = [
        _base_action(
            index,
            l1_probabilities=l1_decision_probabilities,
            l2_probabilities=l2_probabilities,
            l1_floors=l1_conformal["candidate_probability_floor_by_class"],
            l2_floors=l2_conformal["candidate_probability_floor_by_class"],
            l1_knownness=l1_knownness,
            l2_knownness=l2_knownness,
            l1_knownness_scores=l1_head_scores,
            l2_knownness_scores=l2_head_scores,
            l2_margin_thresholds=l2_margin_thresholds,
        )
        for index in range(count)
    ]
    reasons = ["independent_post_selection_audit_required"]
    if int(unknown.sum()) < int(policy.minimum_unknown_support):
        reasons.append("unknown_support_below_l3_minimum")
    if any(
        item.get("selected") is None
        for item in (*l1_knownness.values(), *l2_knownness.values())
    ):
        reasons.append("one_or_more_knownness_thresholds_unavailable")
    elif any(
        item.get("status") != "fitted"
        for item in (*l1_knownness.values(), *l2_knownness.values())
    ):
        reasons.append("knownness_constraints_infeasible")
    if any(
        value is None
        for section in (l1_conformal, l2_conformal)
        for value in section["candidate_probability_floor_by_class"].values()
    ):
        reasons.append("one_or_more_conformal_classes_below_support")
    exploratory_ready = not any(
        reason in {
            "one_or_more_knownness_thresholds_unavailable",
            "one_or_more_conformal_classes_below_support",
        }
        for reason in reasons
    )
    from radar_ln3.decision.scorer import build_scorer_descriptor

    l2_scorer = build_scorer_descriptor(
        class_names=L2_CLASSES,
        conformal={
            "quantile_by_class": l2_conformal["nonconformity_quantile_by_class"],
            "support_by_class": l2_conformal["support_by_class"],
        },
        temperature=l2_temperature,
    )
    body = {
        "schema_version": L3_CALIBRATION_SCHEMA,
        "status": "exploratory" if exploratory_ready else "unavailable",
        "deployment_eligible": False,
        "status_reasons": reasons,
        "source_manifest_sha256": source_manifest_sha256.lower(),
        "endpoint_sha256": {key: str(value).lower() for key, value in endpoint_sha256.items()},
        "knownness_source": "independent_l3_head",
        "l1_router_source": "independent_l3_head" if l1_router_from_head else "classifier_endpoint",
        "l2_knownness_threshold_mode": str(l2_knownness_threshold_mode),
        "head_checkpoint_sha256": {key: str(value).lower() for key, value in head_checkpoint_sha256.items()},
        "source_role": source_role,
        "parent_ids": list(ids),
        "parent_ids_sha256": canonical_hash(sorted(ids)),
        "label_support": {
            label: int(np.sum(label_array == label))
            for label in (*KNOWN_LABELS, UNKNOWN_LABEL)
        },
        "policy": policy.to_dict(),
        "temperature": {"l1": l1_temperature, "l2": l2_temperature},
        "conformal": {"l1": l1_conformal, "l2": l2_conformal},
        "l1_target_conformal_floor_override": None,
        "l2_scorer": l2_scorer,
        "knownness_thresholds": {"l1": l1_knownness, "l2": l2_knownness},
        "l2_top2_margin_thresholds": l2_margin_thresholds,
        "bird_recovery_gate": None,
        "unknown_threshold_fit_support": int(unknown.sum()),
        "unknown_used_for_threshold_fit": True,
        "test_payload_access_count": 0,
        "audit": None,
    }
    return _seal_calibration(body)


def _effective_l1_conformal_floors(
    calibration: Mapping[str, Any],
) -> Mapping[str, Any]:
    l1_floors = calibration["conformal"]["l1"]["candidate_probability_floor_by_class"]
    conformal_override = calibration.get("l1_target_conformal_floor_override")
    if conformal_override is not None:
        l1_floors = dict(l1_floors)
        l1_floors["KNOWN_TARGET_EVIDENCE"] = min(
            float(l1_floors["KNOWN_TARGET_EVIDENCE"]),
            float(conformal_override["floor"]),
        )
    return l1_floors


def _decision_from_probabilities(
    calibration: Mapping[str, Any],
    *,
    l1_probabilities: np.ndarray,
    l2_probabilities: np.ndarray | None,
    streaming: bool,
    l1_knownness_scores: np.ndarray,
    l2_knownness_scores: np.ndarray,
) -> tuple[str, str, dict[str, Any]]:
    l1_predicted_id = int(np.argmax(l1_probabilities))
    l1_prediction = L1_CLASSES[l1_predicted_id]
    route = "target" if l1_predicted_id == 0 else "clutter"
    l1_knownness = calibration["knownness_thresholds"]["l1"][route]["selected"]
    l1_knownness_threshold = float(l1_knownness["threshold"])
    strategy = calibration.get("l3_2_strategy")
    if route == "target" and strategy is not None:
        l1_knownness_threshold = float(strategy["l1_target_threshold"])
    target_override = calibration.get("l1_target_knownness_override")
    if route == "target" and target_override is not None:
        l1_knownness_threshold = float(target_override["threshold"])
    l1_score = float(l1_knownness_scores[0 if route == "target" else 1])
    if l1_score < l1_knownness_threshold:
        target_bird_gate = calibration.get("l1_target_bird_recovery_gate")
        if (
            route == "target"
            and target_bird_gate is not None
            and bool(target_bird_gate.get("enabled"))
            and l2_probabilities is not None
            and float(target_bird_gate["l1_target_score_min"])
            <= l1_score
            <= float(target_bird_gate["l1_target_score_max"])
        ):
            l2_predicted_id = int(np.argmax(l2_probabilities))
            if l2_predicted_id == L2_CLASSES.index("BirdTarget"):
                l2_bird_probability = float(l2_probabilities[l2_predicted_id])
                l2_bird_score = float(l2_knownness_scores[l2_predicted_id])
                ordered = np.sort(l2_probabilities)[::-1]
                l2_bird_margin = float(ordered[0] - ordered[1])
                l2_candidates = _candidate_set(
                    l2_probabilities,
                    L2_CLASSES,
                    calibration["conformal"]["l2"]["candidate_probability_floor_by_class"],
                )
                if (
                    l2_candidates == ["BirdTarget"]
                    and l2_bird_probability
                    >= float(target_bird_gate["l2_bird_probability_floor"])
                    and l2_bird_score
                    >= float(target_bird_gate["l2_bird_knownness_threshold"])
                    and l2_bird_margin
                    >= float(target_bird_gate["l2_bird_margin_floor"])
                ):
                    return "BirdTarget", "L1_TARGET_BIRD_RECOVERY", {
                        "l1_prediction": l1_prediction,
                        "l1_target_knownness_score": l1_score,
                        "l1_knownness_threshold": l1_knownness_threshold,
                        "l2_prediction": "BirdTarget",
                        "l2_bird_probability": l2_bird_probability,
                        "l2_bird_knownness_score": l2_bird_score,
                        "l2_bird_margin": l2_bird_margin,
                        "l2_candidates": l2_candidates,
                        "recovery_gate": dict(target_bird_gate),
                    }
        return "UNKNOWN_REJECTED", "L1_KNOWNNESS_BELOW_THRESHOLD", {
            "l1_prediction": l1_prediction,
            "l1_knownness_score": l1_score,
            "l1_knownness_threshold": l1_knownness_threshold,
        }
    l1_floors = _effective_l1_conformal_floors(calibration)
    l1_candidates = _candidate_set(l1_probabilities, L1_CLASSES, l1_floors)
    if not l1_candidates:
        return "UNKNOWN_REJECTED", "L1_CONFORMAL_SET_EMPTY", {
            "l1_prediction": l1_prediction,
            "l1_knownness_score": l1_score,
            "l1_candidates": [],
        }
    if len(l1_candidates) != 1 or l1_candidates[0] != l1_prediction:
        return "PENDING_ABSTAIN", "L1_ROUTE_AMBIGUOUS", {
            "l1_prediction": l1_prediction,
            "l1_candidates": l1_candidates,
        }
    if route == "clutter":
        clutter_bird_gate = calibration.get("l1_clutter_bird_recovery_gate")
        if (
            clutter_bird_gate is not None
            and bool(clutter_bird_gate.get("enabled"))
            and l2_probabilities is not None
        ):
            l2_predicted_id = int(np.argmax(l2_probabilities))
            if l2_predicted_id == L2_CLASSES.index("BirdTarget"):
                l1_target_score = float(l1_knownness_scores[0])
                l2_bird_probability = float(l2_probabilities[l2_predicted_id])
                l2_bird_score = float(l2_knownness_scores[l2_predicted_id])
                if (
                    float(clutter_bird_gate["l1_target_score_min"])
                    <= l1_target_score
                    <= float(clutter_bird_gate["l1_target_score_max"])
                    and l2_bird_probability
                    >= float(clutter_bird_gate["l2_bird_probability_floor"])
                    and l2_bird_score
                    >= float(clutter_bird_gate["l2_bird_knownness_threshold"])
                ):
                    return "BirdTarget", "L1_CLUTTER_BIRD_RECOVERY", {
                        "l1_prediction": l1_prediction,
                        "l1_target_knownness_score": l1_target_score,
                        "l2_prediction": "BirdTarget",
                        "l2_bird_probability": l2_bird_probability,
                        "l2_bird_knownness_score": l2_bird_score,
                        "recovery_gate": dict(clutter_bird_gate),
                    }
        return "Clutter", "L1_CLUTTER_ACCEPTED", {
            "l1_prediction": l1_prediction,
            "l1_knownness_score": l1_score,
            "l1_candidates": l1_candidates,
        }
    if l2_probabilities is None:
        return "PENDING_ABSTAIN", "L2_EVIDENCE_MISSING", {
            "l1_prediction": l1_prediction,
        }
    l2_predicted_id = int(np.argmax(l2_probabilities))
    l2_prediction = L2_CLASSES[l2_predicted_id]
    l2_knownness = calibration["knownness_thresholds"]["l2"][l2_prediction]["selected"]
    bird_gate = calibration.get("bird_recovery_gate")
    l2_knownness_threshold = float(l2_knownness["threshold"])
    if (
        l2_prediction == "BirdTarget"
        and bird_gate is not None
        and bool(bird_gate.get("enabled"))
    ):
        l2_knownness_threshold = float(bird_gate["knownness_threshold"])
    l2_score = float(l2_knownness_scores[l2_predicted_id])
    if l2_score < l2_knownness_threshold:
        return "UNKNOWN_REJECTED", "L2_KNOWNNESS_BELOW_THRESHOLD", {
            "l1_prediction": l1_prediction,
            "l2_prediction": l2_prediction,
            "l2_knownness_score": l2_score,
            "l2_knownness_threshold": l2_knownness_threshold,
        }
    l2_candidates = _candidate_set(
        l2_probabilities,
        L2_CLASSES,
        calibration["conformal"]["l2"]["candidate_probability_floor_by_class"],
    )
    if not l2_candidates:
        return "UNKNOWN_REJECTED", "L2_CONFORMAL_SET_EMPTY", {
            "l1_prediction": l1_prediction,
            "l2_prediction": l2_prediction,
            "l2_candidates": [],
        }
    if (
        l2_prediction == "BirdTarget"
        and bird_gate is not None
        and bool(bird_gate.get("enabled"))
    ):
        bird_probability = float(l2_probabilities[L2_CLASSES.index("BirdTarget")])
        ordered = np.sort(l2_probabilities)[::-1]
        bird_margin = float(ordered[0] - ordered[1])
        probability_floor = float(bird_gate["probability_floor"])
        margin_floor = float(bird_gate["margin_floor"])
        if bird_probability < probability_floor:
            return "UNKNOWN_REJECTED", "L2_BIRD_PROBABILITY_BELOW_THRESHOLD", {
                "l1_prediction": l1_prediction,
                "l2_prediction": l2_prediction,
                "l2_bird_probability": bird_probability,
                "l2_bird_probability_floor": probability_floor,
            }
        if bird_margin < margin_floor:
            return (
                "PENDING_ABSTAIN" if streaming else "UNKNOWN_REJECTED",
                "L2_BIRD_MARGIN_BELOW_THRESHOLD",
                {
                    "l1_prediction": l1_prediction,
                    "l2_prediction": l2_prediction,
                    "l2_bird_margin": bird_margin,
                    "l2_bird_margin_floor": margin_floor,
                },
            )
    if len(l2_candidates) > 1:
        ordered = np.sort(l2_probabilities)[::-1]
        l2_margin = float(ordered[0] - ordered[1])
        margin_threshold = float(
            calibration.get("l2_top2_margin_thresholds", {}).get(l2_prediction, 1.0)
        )
        if (
            l2_prediction in l2_candidates
            and l2_margin + 1.0e-12 >= margin_threshold
        ):
            return l2_prediction, "KNOWN_DBB_CLASS_ACCEPTED", {
                "l1_prediction": l1_prediction,
                "l2_prediction": l2_prediction,
                "l2_candidates": l2_candidates,
                "l2_top2_margin": l2_margin,
                "l2_top2_margin_threshold": margin_threshold,
                "l1_knownness_score": l1_score,
                "l2_knownness_score": l2_score,
            }
        return (
            "PENDING_ABSTAIN" if streaming else "UNKNOWN_REJECTED",
            "L2_CLASS_AMBIGUOUS",
            {
                "l1_prediction": l1_prediction,
                "l2_candidates": l2_candidates,
                "l2_top2_margin": l2_margin,
                "l2_top2_margin_threshold": margin_threshold,
            },
        )
    if l2_candidates[0] != l2_prediction:
        return "PENDING_ABSTAIN", "L2_ROUTE_CONFLICT", {
            "l1_prediction": l1_prediction,
            "l2_prediction": l2_prediction,
            "l2_candidates": l2_candidates,
        }
    return l2_prediction, "KNOWN_DBB_CLASS_ACCEPTED", {
        "l1_prediction": l1_prediction,
        "l2_prediction": l2_prediction,
        "l1_candidates": l1_candidates,
        "l2_candidates": l2_candidates,
        "l1_knownness_score": l1_score,
        "l2_knownness_score": l2_score,
        "knownness_source": "independent_l3_head",
    }


def resolve_l3(
    calibration: Mapping[str, Any],
    *,
    policy: L3CalibrationPolicy | None = None,
    parent_sample_id: str,
    l1_logits: Sequence[float],
    l2_logits: Sequence[float] | None,
    l1_knownness_scores: Sequence[float] | None = None,
    l2_knownness_scores: Sequence[float] | None = None,
    streaming: bool = False,
    allow_exploratory: bool = False,
) -> dict[str, Any]:
    """解析一个 parent；传入 policy 时 calibration 必须已在调用边界校验。"""

    value = calibration if policy is not None else _validate_calibration(calibration)
    if policy is None:
        policy = L3CalibrationPolicy(**value["policy"])
    details: dict[str, Any] = {
        "parent_sample_id": str(parent_sample_id),
        "calibration_status": value["status"],
        "calibration_sha256": value["artifact_sha256"],
        "research_only": not bool(value.get("deployment_eligible")),
        "knownness_source": "independent_l3_head",
    }
    if value["status"] == "unavailable":
        return {"action": "PENDING_ABSTAIN", "reason_code": "CALIBRATION_INSUFFICIENT_SUPPORT", "details": details}
    if not bool(value.get("deployment_eligible")) and not allow_exploratory:
        return {"action": "PENDING_ABSTAIN", "reason_code": "CALIBRATION_NOT_DEPLOYABLE", "details": details}
    if l1_knownness_scores is None or l2_knownness_scores is None:
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "L3_KNOWNNESS_HEAD_MISSING",
            "details": {**details, "knownness_source": "independent_l3_head"},
        }
    l1_scores = np.asarray(l1_knownness_scores, dtype=np.float64)
    l2_scores = np.asarray(l2_knownness_scores, dtype=np.float64)
    if (
        l1_scores.shape != (len(L1_CLASSES),)
        or l2_scores.shape != (len(L2_CLASSES),)
        or not np.isfinite(l1_scores).all()
        or not np.isfinite(l2_scores).all()
        or np.any((l1_scores < 0.0) | (l1_scores > 1.0))
        or np.any((l2_scores < 0.0) | (l2_scores > 1.0))
    ):
        return {
            "action": "PENDING_ABSTAIN",
            "reason_code": "L3_KNOWNNESS_HEAD_INVALID",
            "details": details,
        }
    l1 = _probability_for_row(
        l1_logits,
        float(value["temperature"]["l1"]["temperature"]),
        len(L1_CLASSES),
    )
    l2 = (
        None
        if l2_logits is None
        else _probability_for_row(
            l2_logits,
            float(value["temperature"]["l2"]["temperature"]),
            len(L2_CLASSES),
        )
    )
    if value.get("l1_router_source", "classifier_endpoint") == "independent_l3_head":
        route_scores = l1_scores
        route_total = float(route_scores.sum())
        if route_total <= 1.0e-12:
            return {
                "action": "PENDING_ABSTAIN",
                "reason_code": "L3_KNOWNNESS_HEAD_INVALID",
                "details": details,
            }
        l1 = l1_scores / route_total
    action, reason, decision_details = _decision_from_probabilities(
        value,
        l1_probabilities=l1,
        l2_probabilities=l2,
        streaming=bool(streaming),
        l1_knownness_scores=l1_scores,
        l2_knownness_scores=l2_scores,
    )
    return {
        "action": action,
        "reason_code": reason,
        "details": {
            **details,
            "knownness_source": "independent_l3_head",
            **decision_details,
        },
    }
def audit_l3_calibration(
    calibration: Mapping[str, Any],
    *,
    parent_ids: Sequence[str],
    labels: Sequence[str],
    evidences: Sequence[Mapping[str, Any]],
    audit_role: str,
    audit_source_manifest_sha256: str,
    thresholds_frozen_before_access: bool,
) -> dict[str, Any]:
    """评估冻结阈值；calibration 需先由 load_l3_calibration 校验。"""

    value = calibration
    ids = tuple(str(item) for item in parent_ids)
    actual = tuple(str(item) for item in labels)
    if audit_role != "post_selection_audit":
        raise ValueError("L3 audit role 必须是 post_selection_audit")
    audit_source_hash = str(audit_source_manifest_sha256).lower()
    if (
        len(audit_source_hash) != 64
        or set(audit_source_hash) - set("0123456789abcdef")
        or audit_source_hash == str(value["source_manifest_sha256"]).lower()
        or thresholds_frozen_before_access is not True
    ):
        raise ValueError("L3 audit 必须来自新数据 manifest，且阈值需提前冻结")
    if not ids or len(ids) != len(set(ids)) or len(ids) != len(actual) or len(ids) != len(evidences):
        raise ValueError("L3 audit parent、label、evidence 未对齐")
    if set(actual) - {*KNOWN_LABELS, UNKNOWN_LABEL}:
        raise ValueError("L3 audit 标签超出语义决策合同")
    overlap = set(ids).intersection(str(item) for item in value["parent_ids"])
    if overlap:
        raise ValueError(f"L3 audit 与阈值拟合 parent 重叠：{sorted(overlap)[:3]}")
    policy = L3CalibrationPolicy(**value["policy"])
    records: list[dict[str, Any]] = []
    for parent_id, label, evidence in zip(ids, actual, evidences, strict=True):
        result = resolve_l3(
            value,
            policy=policy,
            parent_sample_id=parent_id,
            l1_logits=evidence["l1_logits"],
            l2_logits=evidence.get("l2_logits"),
            l1_knownness_scores=evidence.get("l1_knownness_scores"),
            l2_knownness_scores=evidence.get("l2_knownness_scores"),
            allow_exploratory=True,
        )
        records.append({"label": label, **result})
    labels_array = np.asarray(actual, dtype=object)
    actions = np.asarray([str(row["action"]) for row in records], dtype=object)
    metrics = _audit_metrics(actions, labels_array, policy=policy)
    audit_support_sufficient = (
        metrics["unknown_support"]
        >= int(value["policy"]["minimum_audit_unknown_support"])
    )
    audit_error_constraints_pass = bool(
        metrics["unknown_false_acceptance_interval"]["upper_95"] is not None
        and metrics["unknown_false_acceptance_interval"]["upper_95"]
        <= float(value["policy"]["maximum_audit_unknown_false_acceptance_rate"])
        and metrics["known_selective_risk_interval"]["upper_95"] is not None
        and metrics["known_selective_risk_interval"]["upper_95"]
        <= float(value["policy"]["maximum_audit_known_selective_risk"])
    )
    passed = bool(
        metrics["safe"]
        and audit_support_sufficient
        and audit_error_constraints_pass
        and all(
            item.get("status") == "fitted"
            and isinstance(item.get("selected"), Mapping)
            and item["selected"].get("safe") is True
            for section in value["knownness_thresholds"].values()
            for item in section.values()
        )
        and value["status"] == "exploratory"
    )
    body = {key: item for key, item in value.items() if key != "artifact_sha256"}
    body["audit"] = {
        "role": audit_role,
        "source_manifest_sha256": audit_source_hash,
        "parent_ids": list(ids),
        "parent_ids_sha256": canonical_hash(sorted(ids)),
        "support": int(len(ids)),
        "minimum_unknown_support": int(value["policy"]["minimum_audit_unknown_support"]),
        "support_sufficient": audit_support_sufficient,
        "metrics": metrics,
        "passed": passed,
        "threshold_fitting_allowed": False,
        "thresholds_frozen_before_access": True,
        "audit_payload_access_count": int(len(ids)),
    }
    body["status"] = "fitted" if passed else "audit_failed"
    body["deployment_eligible"] = passed
    body["status_reasons"] = [] if passed else ["post_selection_audit_constraints_failed"]
    return _seal_calibration(body)


def load_l3_calibration(
    project_root: str | Path,
    path: str | Path,
) -> dict[str, Any]:
    target = resolve_project_path(project_root, path, must_exist=True)
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"L3 calibration 文件无法读取：{target}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("L3 calibration 文件根节点必须是对象")
    return _validate_calibration(payload)


def save_l3_calibration(
    project_root: str | Path,
    path: str | Path,
    calibration: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """在项目根目录内原子保存 sealed L3 calibration JSON。"""

    value = _validate_calibration(calibration)
    root = Path(project_root).resolve()
    target = resolve_project_path(root, path, must_exist=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        existing = load_l3_calibration(root, target)
        if existing == value:
            return target
        raise FileExistsError(f"拒绝覆盖不同的 L3 calibration：{target}")
    text = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


__all__ = [
    "L1_CLASSES",
    "L2_CLASSES",
    "L3_ACTIONS",
    "L3_CALIBRATION_SCHEMA",
    "L3CalibrationPolicy",
    "audit_l3_calibration",
    "fit_l3_calibration",
    "load_l3_calibration",
    "resolve_l3",
    "save_l3_calibration",
]

