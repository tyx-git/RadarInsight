"""Evaluate V24d's cross-fitted L3 score-band Pending policy.

The source L1 OOF scores are immutable. Every outer fold receives thresholds
fitted only on the other four folds' parent-final records; those thresholds are
then applied unchanged to parent, eligible-prefix, and window records.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from build.known_gate_v24d import (
    CONTRACT_SCHEMA,
    DIRECTION_REQUIREMENTS,
    EXPERIMENT_CONTRACT,
    FOLD_COUNT,
    MINIMUM_EVIDENCE_EVENTS,
    PREFIX_SAFETY_REQUIREMENTS,
    PROTOCOL,
    PROTOCOL_SCHEMA,
    READINESS,
    READINESS_SCHEMA,
    REPORT_SCHEMA,
    SELECTION_REQUIREMENTS,
    SOURCE_LABEL_SUPPORT,
    SOURCE_PARENT_COUNT,
    _implementation_audit,
    _source_prediction_audit,
)
from radar_ln3.known_gate_oof import load_fold_contract, sha256_file
from report.known_gate_oof import _load_arm


OUTPUT = Path(
    "src-v3/processed/report/"
    "RADAR-LN3-V24d-known-gate-l3-oof-score-band-pending-cross-fitted.json"
)
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 24044
ACTION_NAMES = (
    "KNOWN_TARGET_EVIDENCE",
    "KNOWN_CLUTTER_EVIDENCE",
    "PENDING_ABSTAIN",
)
TARGET_ACTION = 0
CLUTTER_ACTION = 1
PENDING_ACTION = 2
SCOPES = ("parent", "eligible_prefix", "window")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _require_preparation(root: Path) -> dict[str, Any]:
    paths = {
        "protocol": root / PROTOCOL,
        "contract": root / EXPERIMENT_CONTRACT,
        "readiness": root / READINESS,
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V24d preparation is incomplete: {missing}")
    protocol = _read_json(paths["protocol"])
    contract = _read_json(paths["contract"])
    readiness = _read_json(paths["readiness"])
    source = _source_prediction_audit(root)
    implementation = _implementation_audit(root)
    checks = dict(readiness.get("checks") or {})
    if (
        protocol.get("schema_version") != PROTOCOL_SCHEMA
        or contract.get("schema_version") != CONTRACT_SCHEMA
        or readiness.get("schema_version") != READINESS_SCHEMA
        or readiness.get("status") != "passed"
        or contract.get("protocol_sha256") != sha256_file(paths["protocol"])
        or readiness.get("protocol_sha256") != sha256_file(paths["protocol"])
        or readiness.get("contract_sha256") != sha256_file(paths["contract"])
        or contract.get("source_oof") != source
        or readiness.get("source_oof") != source
        or contract.get("implementation_files_sha256") != implementation
        or readiness.get("implementation_files_sha256") != implementation
        or int(readiness.get("known_train_parent_count", -1)) != SOURCE_PARENT_COUNT
        or int(readiness.get("clutter_parent_support", -1))
        != SOURCE_LABEL_SUPPORT["ClutterTarget"]
        or int(readiness.get("fold_count", -1)) != FOLD_COUNT
        or readiness.get("threshold_fit_unit") != "parent_final_OOF_only"
        or readiness.get("threshold_fit_held_out_labels_accessed") is not False
        or int(readiness.get("validation_payload_access_count", -1)) != 0
        or int(readiness.get("test_payload_access_count", -1)) != 0
        or int(readiness.get("unknown_training_exposure", -1)) != 0
        or readiness.get("deployment_eligible") is not False
        or not checks
        or not all(value is True for value in checks.values())
    ):
        raise ValueError("V24d protocol/readiness/OOF source binding is stale")
    return {
        "protocol": protocol,
        "contract": contract,
        "readiness": readiness,
        "source": source,
        "implementation": implementation,
        "paths": paths,
    }


def _probability(row: Mapping[str, Any]) -> float:
    value = float(row.get("known_clutter_probability", float("nan")))
    target = float(row.get("known_target_probability", float("nan")))
    if not np.isfinite(value) or not np.isfinite(target) or not (0.0 <= value <= 1.0):
        raise ValueError("V24d received a non-finite or out-of-range OOF score")
    if abs(value + target - 1.0) > 1.0e-5:
        raise ValueError("V24d OOF class probabilities do not sum to one")
    return value


def _score_band_actions(
    scores: np.ndarray,
    *,
    target_threshold: float,
    clutter_threshold: float,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    target_threshold = float(target_threshold)
    clutter_threshold = float(clutter_threshold)
    if (
        not np.isfinite(scores).all()
        or np.any(scores < 0.0)
        or np.any(scores > 1.0)
        or not (0.0 <= target_threshold <= 1.0)
        or not (0.0 <= clutter_threshold <= 1.0)
        or target_threshold > clutter_threshold
    ):
        raise ValueError("V24d score-band threshold or score is invalid")
    if eligible is None:
        eligible = np.ones(scores.shape, dtype=np.bool_)
    if eligible.shape != scores.shape:
        raise ValueError("V24d eligibility mask shape differs from scores")
    action = np.full(scores.shape, PENDING_ACTION, dtype=np.int64)
    active = np.asarray(eligible, dtype=np.bool_)
    # A zero-width band is deliberately reserved for the immutable L1 raw
    # argmax baseline.  Letting an arbitrary equal pair through would make
    # the declared score-band rule ambiguous at the shared boundary; the
    # implementation must therefore not silently reinterpret it as a 0.5
    # split.  The sole legal degenerate candidate exactly reproduces the
    # binary softmax argmax (ties go to class 0 / KnownTarget, as PyTorch
    # argmax does).
    if target_threshold == clutter_threshold:
        if target_threshold != 0.5:
            raise ValueError(
                "V24d zero-width score bands are reserved for raw argmax at 0.5"
            )
        action[active] = np.where(scores[active] <= 0.5, TARGET_ACTION, CLUTTER_ACTION)
        return action
    action[active & (scores <= target_threshold)] = TARGET_ACTION
    action[active & (scores >= clutter_threshold)] = CLUTTER_ACTION
    return action


def _arrays(
    rows: Sequence[Mapping[str, Any]],
    *,
    action_key: str,
    parent_balanced: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("V24d cannot summarize an empty scope")
    true = np.asarray([int(row["true_id"]) for row in rows], dtype=np.int64)
    action = np.asarray([int(row[action_key]) for row in rows], dtype=np.int64)
    if np.any((true < TARGET_ACTION) | (true > CLUTTER_ACTION)) or np.any(
        (action < TARGET_ACTION) | (action > PENDING_ACTION)
    ):
        raise ValueError("V24d action or label id is outside its declared space")
    if not parent_balanced:
        return true, action, np.ones(true.shape, dtype=np.float64)
    counts: Counter[str] = Counter(str(row["parent_sample_id"]) for row in rows)
    if not counts:
        raise ValueError("V24d parent-balanced scope lacks parents")
    weights = np.asarray(
        [1.0 / counts[str(row["parent_sample_id"])] for row in rows],
        dtype=np.float64,
    )
    if not np.isclose(float(weights.sum()), float(len(counts)), atol=1.0e-9):
        raise ValueError("V24d parent-balanced prefix weights do not sum to parent support")
    return true, action, weights


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _action_metrics_from_arrays(
    true: np.ndarray,
    action: np.ndarray,
    weights: np.ndarray,
    *,
    scope: str,
    event_support: int,
    parent_support: int,
    aggregation: str,
) -> dict[str, Any]:
    if not (true.shape == action.shape == weights.shape) or true.size == 0:
        raise ValueError("V24d metrics arrays are malformed")
    matrix = np.zeros((2, 3), dtype=np.float64)
    np.add.at(matrix, (true, action), weights)
    return _action_metrics_from_matrix(
        matrix,
        scope=scope,
        event_support=event_support,
        parent_support=parent_support,
        aggregation=aggregation,
    )


def _action_metrics_from_matrix(
    matrix: np.ndarray,
    *,
    scope: str,
    event_support: int,
    parent_support: int,
    aggregation: str,
) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all() or bool((matrix < 0.0).any()):
        raise ValueError("V24d action confusion matrix is malformed")
    support = float(matrix.sum())
    if support <= 0.0:
        raise ValueError("V24d action confusion matrix has zero support")
    accepted = float(matrix[:, :2].sum())
    correct = float(matrix[0, 0] + matrix[1, 1])
    wrong_target_to_clutter = float(matrix[0, 1])
    wrong_clutter_to_target = float(matrix[1, 0])
    per_class: list[dict[str, Any]] = []
    flat: dict[str, float | None] = {}
    for class_id, name in enumerate(ACTION_NAMES[:2]):
        true_support = float(matrix[class_id].sum())
        accepted_as_class = float(matrix[:, class_id].sum())
        true_positive = float(matrix[class_id, class_id])
        precision = _safe_divide(true_positive, accepted_as_class)
        recall = _safe_divide(true_positive, true_support)
        f1 = (
            _safe_divide(2.0 * precision * recall, precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        prefix = "known_target" if class_id == TARGET_ACTION else "known_clutter"
        flat[f"{prefix}_action_precision"] = precision
        flat[f"{prefix}_acceptance_recall"] = recall
        flat[f"{prefix}_action_f1"] = f1
        flat[f"{prefix}_pending_rate"] = _safe_divide(float(matrix[class_id, 2]), true_support)
        per_class.append(
            {
                "class": name,
                "true_support": true_support,
                "accepted_as_class_support": accepted_as_class,
                "action_precision": precision,
                "acceptance_recall": recall,
                "action_f1": f1,
                "pending_abstain_rate": flat[f"{prefix}_pending_rate"],
            }
        )
    pending = float(matrix[:, PENDING_ACTION].sum())
    result: dict[str, Any] = {
        "scope": scope,
        "aggregation": aggregation,
        "event_support": int(event_support),
        "parent_support": int(parent_support),
        "weighted_support": support,
        "action_names": list(ACTION_NAMES),
        "action_confusion_rows_true_columns_action": matrix.tolist(),
        "overall_action_coverage": _safe_divide(accepted, support),
        "pending_abstain_rate": _safe_divide(pending, support),
        "accepted_action_accuracy": _safe_divide(correct, support),
        "selective_accuracy": _safe_divide(correct, accepted),
        "selective_risk": _safe_divide(accepted - correct, accepted),
        "incorrect_accepted_action_rate": _safe_divide(accepted - correct, support),
        "known_target_to_clutter_weight": wrong_target_to_clutter,
        "clutter_to_known_target_weight": wrong_clutter_to_target,
        "known_target_to_clutter_rate": _safe_divide(wrong_target_to_clutter, float(matrix[0].sum())),
        "clutter_to_known_target_rate": _safe_divide(wrong_clutter_to_target, float(matrix[1].sum())),
        "per_class": per_class,
        **flat,
    }
    return result


def _action_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    action_key: str,
    scope: str,
    parent_balanced: bool = False,
) -> dict[str, Any]:
    true, action, weights = _arrays(rows, action_key=action_key, parent_balanced=parent_balanced)
    return _action_metrics_from_arrays(
        true,
        action,
        weights,
        scope=scope,
        event_support=len(rows),
        parent_support=len({str(row["parent_sample_id"]) for row in rows}),
        aggregation=(
            "each eligible parent has total weight one" if parent_balanced else "each record weight one"
        ),
    )


def _selection_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    values = {
        key: metrics.get(key) for key in SELECTION_REQUIREMENTS
    }
    if any(value is None or not np.isfinite(float(value)) for value in values.values()):
        return {key: 0.0 for key in values}
    return {key: float(value) for key, value in values.items()}


def _candidate_key_from_counts(
    *,
    target_action_counts: np.ndarray,
    clutter_action_counts: np.ndarray,
    true_support: np.ndarray,
    raw_metrics: Mapping[str, Any],
    target_threshold: float,
    clutter_threshold: float,
) -> tuple[bool, float, dict[str, bool], tuple[float, ...]]:
    target_action_counts = np.asarray(target_action_counts, dtype=np.float64)
    clutter_action_counts = np.asarray(clutter_action_counts, dtype=np.float64)
    true_support = np.asarray(true_support, dtype=np.float64)
    if (
        target_action_counts.shape != (2,)
        or clutter_action_counts.shape != (2,)
        or true_support.shape != (2,)
        or bool((target_action_counts < 0.0).any())
        or bool((clutter_action_counts < 0.0).any())
        or bool((target_action_counts + clutter_action_counts > true_support + 1.0e-12).any())
    ):
        raise ValueError("V24d threshold candidate counts are invalid")
    target_true_positive = float(target_action_counts[TARGET_ACTION])
    clutter_true_positive = float(clutter_action_counts[CLUTTER_ACTION])
    total_support = float(true_support.sum())
    violations = {
        "known_clutter_action_precision": max(
            0.0,
            float(SELECTION_REQUIREMENTS["known_clutter_action_precision"])
            - (_safe_divide(clutter_true_positive, float(clutter_action_counts.sum())) or 0.0),
        ),
        "known_clutter_acceptance_recall": max(
            0.0,
            float(SELECTION_REQUIREMENTS["known_clutter_acceptance_recall"])
            - (_safe_divide(clutter_true_positive, float(true_support[CLUTTER_ACTION])) or 0.0),
        ),
        "known_target_acceptance_recall": max(
            0.0,
            float(SELECTION_REQUIREMENTS["known_target_acceptance_recall"])
            - (_safe_divide(target_true_positive, float(true_support[TARGET_ACTION])) or 0.0),
        ),
        "overall_action_coverage": max(
            0.0,
            float(SELECTION_REQUIREMENTS["overall_action_coverage"])
            - _safe_divide(
                float(target_action_counts.sum() + clutter_action_counts.sum()), total_support
            ),
        ),
    }
    values = {
        "known_clutter_action_precision": (
            _safe_divide(clutter_true_positive, float(clutter_action_counts.sum())) or 0.0
        ),
        "known_clutter_acceptance_recall": (
            _safe_divide(clutter_true_positive, float(true_support[CLUTTER_ACTION])) or 0.0
        ),
        "known_target_acceptance_recall": (
            _safe_divide(target_true_positive, float(true_support[TARGET_ACTION])) or 0.0
        ),
        "overall_action_coverage": _safe_divide(
            float(target_action_counts.sum() + clutter_action_counts.sum()), total_support
        ),
    }
    if values["overall_action_coverage"] is None:
        raise AssertionError("V24d threshold candidate has zero support")
    direction_checks = {
        "known_target_to_clutter_rate_not_higher_than_raw": float(
            _safe_divide(
                float(clutter_action_counts[TARGET_ACTION]),
                float(true_support[TARGET_ACTION]),
            )
            or 0.0
        )
        <= float(raw_metrics["known_target_to_clutter_rate"]),
        "clutter_to_known_target_rate_not_higher_than_raw": float(
            _safe_divide(
                float(target_action_counts[CLUTTER_ACTION]),
                float(true_support[CLUTTER_ACTION]),
            )
            or 0.0
        )
        <= float(raw_metrics["clutter_to_known_target_rate"]),
    }
    if set(direction_checks) != set(DIRECTION_REQUIREMENTS):
        raise AssertionError("V24d direction requirements differ from the frozen protocol")
    direction_violations = {
        key: max(
            0.0,
            float(candidate_value) - float(raw_metrics[metric_key]),
        )
        for key, metric_key, candidate_value in (
            (
                "known_target_to_clutter_rate_not_higher_than_raw",
                "known_target_to_clutter_rate",
                _safe_divide(
                    float(clutter_action_counts[TARGET_ACTION]),
                    float(true_support[TARGET_ACTION]),
                )
                or 0.0,
            ),
            (
                "clutter_to_known_target_rate_not_higher_than_raw",
                "clutter_to_known_target_rate",
                _safe_divide(
                    float(target_action_counts[CLUTTER_ACTION]),
                    float(true_support[CLUTTER_ACTION]),
                )
                or 0.0,
            ),
        )
    }
    feasible = not any(violations.values()) and all(direction_checks.values())
    coverage = values["overall_action_coverage"]
    minimum_recall = min(
        values["known_target_acceptance_recall"],
        values["known_clutter_acceptance_recall"],
    )
    ranking = (
        0.0 if feasible else 1.0,
        0.0 if feasible else float(sum(violations.values()) + sum(direction_violations.values())),
        -coverage,
        -minimum_recall,
        float(clutter_threshold - target_threshold),
        float(target_threshold),
        float(clutter_threshold),
    )
    return (
        feasible,
        float(sum(violations.values()) + sum(direction_violations.values())),
        direction_checks,
        ranking,
    )


def _select_thresholds(
    parent_rows: Sequence[Mapping[str, Any]],
    *,
    fit_fold_ids: Sequence[int],
) -> dict[str, Any]:
    if not parent_rows:
        raise ValueError("V24d cannot fit thresholds without parent rows")
    scores = np.asarray([_probability(row) for row in parent_rows], dtype=np.float64)
    true = np.asarray([int(row["true_id"]) for row in parent_rows], dtype=np.int64)
    raw_action = np.asarray([int(row["pred_id"]) for row in parent_rows], dtype=np.int64)
    weights = np.ones(true.shape, dtype=np.float64)
    raw_metrics = _action_metrics_from_arrays(
        true,
        raw_action,
        weights,
        scope="threshold_fit_parent_final_raw_argmax",
        event_support=len(parent_rows),
        parent_support=len(parent_rows),
        aggregation="each parent-final OOF record weight one",
    )
    raw_action_from_score = _score_band_actions(
        scores,
        target_threshold=0.5,
        clutter_threshold=0.5,
    )
    if not np.array_equal(raw_action_from_score, raw_action):
        raise ValueError(
            "V24d parent-final source pred_id does not reproduce its binary score argmax"
        )
    threshold_values = np.unique(np.concatenate((scores, np.asarray([0.0, 1.0]))))
    if threshold_values.ndim != 1 or threshold_values.size < 2:
        raise ValueError("V24d lacks sufficient score breakpoints for exact calibration")
    target_counts_by_threshold = np.zeros((threshold_values.size, 2), dtype=np.float64)
    clutter_counts_by_threshold = np.zeros((threshold_values.size, 2), dtype=np.float64)
    for index, threshold in enumerate(threshold_values.tolist()):
        for class_id in (TARGET_ACTION, CLUTTER_ACTION):
            class_mask = true == class_id
            target_counts_by_threshold[index, class_id] = float(
                np.count_nonzero(class_mask & (scores <= threshold))
            )
            clutter_counts_by_threshold[index, class_id] = float(
                np.count_nonzero(class_mask & (scores >= threshold))
            )
    true_support = np.asarray(
        [float(np.count_nonzero(true == class_id)) for class_id in (TARGET_ACTION, CLUTTER_ACTION)],
        dtype=np.float64,
    )
    best: dict[str, Any] | None = None
    feasible_count = 0
    candidate_count = 0
    raw_matrix = np.asarray(
        raw_metrics["action_confusion_rows_true_columns_action"], dtype=np.float64
    )
    raw_target_counts = raw_matrix[:, TARGET_ACTION]
    raw_clutter_counts = raw_matrix[:, CLUTTER_ACTION]
    raw_feasible, raw_violation, raw_direction_checks, raw_ranking = _candidate_key_from_counts(
        target_action_counts=raw_target_counts,
        clutter_action_counts=raw_clutter_counts,
        true_support=true_support,
        raw_metrics=raw_metrics,
        target_threshold=0.5,
        clutter_threshold=0.5,
    )
    best = {
        "target_threshold": 0.5,
        "clutter_threshold": 0.5,
        "score_band_width": 0.0,
        "decision_mode": "raw_argmax_degenerate_baseline",
        "fit_feasible": raw_feasible,
        "total_requirement_violation": raw_violation,
        "direction_requirement_checks": raw_direction_checks,
        "target_action_counts": raw_target_counts.copy(),
        "clutter_action_counts": raw_clutter_counts.copy(),
        "ranking": raw_ranking,
    }
    candidate_count = 1
    feasible_count = int(raw_feasible)
    for target_index, target_threshold in enumerate(threshold_values.tolist()):
        target_counts = target_counts_by_threshold[target_index]
        # Strictly ordered score-band endpoints make target, Pending and
        # Clutter mutually exclusive.  The only zero-width possibility is
        # the raw-argmax baseline inserted above.
        for clutter_index in range(target_index + 1, threshold_values.size):
            clutter_threshold = float(threshold_values[clutter_index])
            clutter_counts = clutter_counts_by_threshold[clutter_index]
            feasible, violation, direction_checks, ranking = _candidate_key_from_counts(
                target_action_counts=target_counts,
                clutter_action_counts=clutter_counts,
                true_support=true_support,
                raw_metrics=raw_metrics,
                target_threshold=float(target_threshold),
                clutter_threshold=float(clutter_threshold),
            )
            candidate_count += 1
            feasible_count += int(feasible)
            candidate = {
                "target_threshold": float(target_threshold),
                "clutter_threshold": float(clutter_threshold),
                "score_band_width": float(clutter_threshold - target_threshold),
                "decision_mode": "strict_score_band",
                "fit_feasible": feasible,
                "total_requirement_violation": violation,
                "direction_requirement_checks": direction_checks,
                "target_action_counts": target_counts.copy(),
                "clutter_action_counts": clutter_counts.copy(),
                "ranking": ranking,
            }
            if best is None or tuple(candidate["ranking"]) < tuple(best["ranking"]):
                best = candidate
    if best is None or candidate_count == 0:
        raise AssertionError("V24d exact threshold candidate set has no legal candidates")
    best_matrix = np.column_stack(
        (
            np.asarray(best.pop("target_action_counts"), dtype=np.float64),
            np.asarray(best.pop("clutter_action_counts"), dtype=np.float64),
        )
    )
    best_matrix = np.column_stack((best_matrix, true_support - best_matrix.sum(axis=1)))
    best["metrics"] = _action_metrics_from_matrix(
        best_matrix,
        scope="threshold_fit_parent_final",
        event_support=len(parent_rows),
        parent_support=len(parent_rows),
        aggregation="each parent-final OOF record weight one",
    )
    return {
        "fit_fold_ids": [int(value) for value in sorted(fit_fold_ids)],
        "fit_parent_support": len(parent_rows),
        "candidate_count": candidate_count,
        "candidate_space": (
            "all strict score decision-equivalence breakpoint pairs plus the "
            "single raw-argmax (0.5, 0.5) baseline"
        ),
        "feasible_candidate_count": feasible_count,
        "selection_mode": "feasible_max_coverage" if best["fit_feasible"] else "minimum_requirement_violation",
        "fit_raw_argmax_metrics": raw_metrics,
        **{key: value for key, value in best.items() if key != "ranking"},
    }


def _validate_source_records(
    records_by_scope: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    if set(records_by_scope) != set(SCOPES):
        raise ValueError("V24d source OOF scopes differ from the frozen contract")
    expected_counts = {"parent": SOURCE_PARENT_COUNT, "eligible_prefix": 15965, "window": 18923}
    if {scope: len(rows) for scope, rows in records_by_scope.items()} != expected_counts:
        raise ValueError("V24d source OOF scope counts differ from the frozen V21e report")
    parent_ids = [str(row["parent_sample_id"]) for row in records_by_scope["parent"]]
    support = Counter(str(row["label_name"]) for row in records_by_scope["parent"])
    if len(parent_ids) != len(set(parent_ids)) or dict(sorted(support.items())) != SOURCE_LABEL_SUPPORT:
        raise ValueError("V24d source OOF parent identity or label support changed")
    for scope, rows in records_by_scope.items():
        keys: set[tuple[str, int]] = set()
        for row in rows:
            fold_id = int(row.get("fold_id", -1))
            key = (str(row.get("parent_sample_id", "")), int(row.get("event_index", -1)))
            if (
                fold_id not in range(FOLD_COUNT)
                or not key[0]
                or key[1] < 0
                or key in keys
                or int(row.get("true_id", -1)) not in {0, 1}
                or int(row.get("pred_id", -1)) not in {0, 1}
            ):
                raise ValueError(f"V24d source {scope} record is malformed or duplicated")
            keys.add(key)
            _probability(row)
            if scope != "window" and not bool(row.get("product_action_eligible")):
                raise ValueError(f"V24d {scope} contains a product-ineligible source record")


def _cross_fit_thresholds(
    parent_rows: Sequence[Mapping[str, Any]],
) -> dict[int, dict[str, Any]]:
    rows_by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in parent_rows:
        rows_by_fold[int(row["fold_id"])].append(dict(row))
    if set(rows_by_fold) != set(range(FOLD_COUNT)):
        raise ValueError("V24d parent records do not cover every outer fold")
    results: dict[int, dict[str, Any]] = {}
    for held_out_fold in range(FOLD_COUNT):
        fit_folds = [fold for fold in range(FOLD_COUNT) if fold != held_out_fold]
        fit_rows = [row for fold in fit_folds for row in rows_by_fold[fold]]
        if any(int(row["fold_id"]) == held_out_fold for row in fit_rows):
            raise AssertionError("V24d held-out fold leaked into threshold fitting")
        selection = _select_thresholds(fit_rows, fit_fold_ids=fit_folds)
        selection["held_out_fold_id"] = held_out_fold
        selection["held_out_parent_support"] = len(rows_by_fold[held_out_fold])
        results[held_out_fold] = selection
    return results


def _annotate_actions(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    thresholds: Mapping[int, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        copied = dict(row)
        fold_id = int(copied["fold_id"])
        if fold_id not in thresholds:
            raise ValueError("V24d record has no cross-fitted threshold binding")
        eligible = bool(copied["product_action_eligible"])
        if scope != "window" and not eligible:
            raise ValueError(f"V24d {scope} record is unexpectedly ineligible")
        raw_action = int(copied["pred_id"]) if eligible else PENDING_ACTION
        threshold = thresholds[fold_id]
        calibrated_action = int(
            _score_band_actions(
                np.asarray([_probability(copied)], dtype=np.float64),
                target_threshold=float(threshold["target_threshold"]),
                clutter_threshold=float(threshold["clutter_threshold"]),
                eligible=np.asarray([eligible], dtype=np.bool_),
            )[0]
        )
        copied.update(
            {
                "raw_action_id": raw_action,
                "calibrated_action_id": calibrated_action,
                "raw_action": ACTION_NAMES[raw_action],
                "calibrated_action": ACTION_NAMES[calibrated_action],
                "threshold_target": float(threshold["target_threshold"]),
                "threshold_clutter": float(threshold["clutter_threshold"]),
            }
        )
        result.append(copied)
    return result


def _scope_metrics(rows_by_scope: Mapping[str, Sequence[Mapping[str, Any]]], *, action_key: str) -> dict[str, Any]:
    return {
        "parent_final": _action_metrics(
            rows_by_scope["parent"], action_key=action_key, scope="parent_final"
        ),
        "eligible_prefix": _action_metrics(
            rows_by_scope["eligible_prefix"], action_key=action_key, scope="eligible_prefix"
        ),
        "parent_balanced_eligible_prefix": _action_metrics(
            rows_by_scope["eligible_prefix"],
            action_key=action_key,
            scope="parent_balanced_eligible_prefix",
            parent_balanced=True,
        ),
        "window": _action_metrics(
            rows_by_scope["window"], action_key=action_key, scope="window"
        ),
    }


DELTA_METRICS = (
    "accepted_action_accuracy",
    "selective_accuracy",
    "selective_risk",
    "overall_action_coverage",
    "pending_abstain_rate",
    "incorrect_accepted_action_rate",
    "known_target_action_precision",
    "known_target_acceptance_recall",
    "known_target_action_f1",
    "known_clutter_action_precision",
    "known_clutter_acceptance_recall",
    "known_clutter_action_f1",
    "known_target_to_clutter_rate",
    "clutter_to_known_target_rate",
)


def _deltas(raw: Mapping[str, Any], calibrated: Mapping[str, Any]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in DELTA_METRICS:
        left = raw.get(key)
        right = calibrated.get(key)
        result[key] = None if left is None or right is None else float(right - left)
    return result


def _wilson(successes: int, support: int, *, z: float = 1.959963984540054) -> dict[str, Any]:
    if support < 0 or successes < 0 or successes > support:
        raise ValueError("V24d Wilson count is invalid")
    if support == 0:
        return {
            "support": 0,
            "successes": 0,
            "estimate": None,
            "lower_95": None,
            "upper_95": None,
            "status": "undefined_zero_denominator",
        }
    proportion = float(successes / support)
    denominator = 1.0 + z * z / support
    center = (proportion + z * z / (2.0 * support)) / denominator
    half = z * np.sqrt(
        proportion * (1.0 - proportion) / support + z * z / (4.0 * support * support)
    ) / denominator
    return {
        "support": int(support),
        "successes": int(successes),
        "estimate": proportion,
        "lower_95": float(max(0.0, center - half)),
        "upper_95": float(min(1.0, center + half)),
        "status": "defined",
    }


def _parent_wilson(metrics: Mapping[str, Any]) -> dict[str, Any]:
    matrix = np.asarray(metrics["action_confusion_rows_true_columns_action"], dtype=np.float64)
    if matrix.shape != (2, 3) or not np.allclose(matrix, np.round(matrix), atol=1.0e-12):
        raise ValueError("V24d parent Wilson requires integral parent action counts")
    counts = np.round(matrix).astype(np.int64)
    clutter_true_positive = int(counts[CLUTTER_ACTION, CLUTTER_ACTION])
    target_true_positive = int(counts[TARGET_ACTION, TARGET_ACTION])
    return {
        "method": "Wilson score interval, two-sided 95 percent",
        "action_confusion_rows_true_columns_action": counts.tolist(),
        "known_clutter_action_precision": _wilson(
            clutter_true_positive, int(counts[:, CLUTTER_ACTION].sum())
        ),
        "known_clutter_acceptance_recall": _wilson(
            clutter_true_positive, int(counts[CLUTTER_ACTION].sum())
        ),
        "known_target_action_precision": _wilson(
            target_true_positive, int(counts[:, TARGET_ACTION].sum())
        ),
        "known_target_acceptance_recall": _wilson(
            target_true_positive, int(counts[TARGET_ACTION].sum())
        ),
        "overall_action_coverage": _wilson(
            int(counts[:, :2].sum()), int(counts.sum())
        ),
        "clutter_parent_support": int(counts[CLUTTER_ACTION].sum()),
    }


def _metric_vector(metrics: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in DELTA_METRICS:
        value = metrics.get(key)
        values[key] = 0.0 if value is None else float(value)
    return values


def _ci(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("V24d bootstrap contains no finite values")
    return {
        "median": float(np.median(array)),
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def _component_bootstrap(
    assignment: Mapping[str, Any], parent_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    parent_by_id = {str(row["parent_sample_id"]): dict(row) for row in parent_rows}
    if len(parent_by_id) != len(parent_rows) or len(parent_by_id) != SOURCE_PARENT_COUNT:
        raise ValueError("V24d bootstrap parent records are incomplete or repeated")
    assignments = {
        str(row["parent_sample_id"]): dict(row)
        for row in list(assignment.get("assignments") or ())
    }
    if set(assignments) != set(parent_by_id):
        raise ValueError("V24d assignment/parent records differ")
    component_indices: dict[str, list[int]] = defaultdict(list)
    component_labels: dict[str, Counter[str]] = defaultdict(Counter)
    ordered_rows = [parent_by_id[parent_id] for parent_id in sorted(parent_by_id)]
    for index, row in enumerate(ordered_rows):
        source = assignments[str(row["parent_sample_id"])]
        component = str(source["provenance_component_id"])
        if (
            component != str(row["provenance_component_id"])
            or str(source["label_name"]) != str(row["label_name"])
        ):
            raise ValueError("V24d bootstrap provenance identity differs from source predictions")
        component_indices[component].append(index)
        component_labels[component][str(row["label_name"])] += 1
    strata: dict[str, list[str]] = defaultdict(list)
    for component, labels in component_labels.items():
        signature = "+".join(f"{label}:{labels[label]}" for label in sorted(labels))
        strata[signature].append(component)
    true = np.asarray([int(row["true_id"]) for row in ordered_rows], dtype=np.int64)
    raw_action = np.asarray([int(row["raw_action_id"]) for row in ordered_rows], dtype=np.int64)
    calibrated_action = np.asarray(
        [int(row["calibrated_action_id"]) for row in ordered_rows], dtype=np.int64
    )
    weights = np.ones(true.shape, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    deltas: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        sampled_components: list[str] = []
        for components in strata.values():
            sampled_components.extend(rng.choice(components, size=len(components), replace=True).tolist())
        sampled_indices = np.asarray(
            [index for component in sampled_components for index in component_indices[component]],
            dtype=np.int64,
        )
        raw_metrics = _action_metrics_from_arrays(
            true[sampled_indices],
            raw_action[sampled_indices],
            weights[sampled_indices],
            scope="component_bootstrap_raw_parent_final",
            event_support=int(sampled_indices.size),
            parent_support=int(sampled_indices.size),
            aggregation="sampled capture/content component parents weight one",
        )
        calibrated_metrics = _action_metrics_from_arrays(
            true[sampled_indices],
            calibrated_action[sampled_indices],
            weights[sampled_indices],
            scope="component_bootstrap_calibrated_parent_final",
            event_support=int(sampled_indices.size),
            parent_support=int(sampled_indices.size),
            aggregation="sampled capture/content component parents weight one",
        )
        raw_vector = _metric_vector(raw_metrics)
        calibrated_vector = _metric_vector(calibrated_metrics)
        for key in DELTA_METRICS:
            deltas[key].append(calibrated_vector[key] - raw_vector[key])
    return {
        "method": "paired capture/content provenance-component bootstrap stratified by component label signature",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "component_count": len(component_indices),
        "stratum_component_counts": {
            key: len(values) for key, values in sorted(strata.items())
        },
        "score_band_minus_raw_argmax": {key: _ci(values) for key, values in deltas.items()},
    }


def _parent_changes(parent_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    changed: list[dict[str, Any]] = []
    for row in sorted(parent_rows, key=lambda value: str(value["parent_sample_id"])):
        raw_action = int(row["raw_action_id"])
        calibrated_action = int(row["calibrated_action_id"])
        if raw_action == calibrated_action:
            continue
        true_id = int(row["true_id"])
        raw_correct = raw_action == true_id
        calibrated_correct = calibrated_action == true_id
        changed.append(
            {
                "parent_sample_id": str(row["parent_sample_id"]),
                "track_id": str(row["track_id"]),
                "fold_id": int(row["fold_id"]),
                "provenance_component_id": str(row["provenance_component_id"]),
                "label_name": str(row["label_name"]),
                "raw_action": ACTION_NAMES[raw_action],
                "calibrated_action": ACTION_NAMES[calibrated_action],
                "known_clutter_probability": _probability(row),
                "threshold_target": float(row["threshold_target"]),
                "threshold_clutter": float(row["threshold_clutter"]),
                "raw_correct_action": raw_correct,
                "calibrated_correct_action": calibrated_correct,
                "change_type": (
                    "rescued"
                    if not raw_correct and calibrated_correct
                    else "harmed"
                    if raw_correct and not calibrated_correct
                    else "abstained_from_raw_error"
                    if not raw_correct and calibrated_action == PENDING_ACTION
                    else "abstained_from_raw_correct"
                    if raw_correct and calibrated_action == PENDING_ACTION
                    else "changed_noncorrect_action"
                ),
            }
        )
    return {
        "changed_parent_count": len(changed),
        "rescued_parent_count": sum(row["change_type"] == "rescued" for row in changed),
        "harmed_parent_count": sum(row["change_type"] == "harmed" for row in changed),
        "abstained_from_raw_error_count": sum(
            row["change_type"] == "abstained_from_raw_error" for row in changed
        ),
        "abstained_from_raw_correct_count": sum(
            row["change_type"] == "abstained_from_raw_correct" for row in changed
        ),
        "changes": changed,
    }


def _provenance_audit(
    parent_rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, Any]
) -> dict[str, Any]:
    assignment_by_parent = {
        str(row["parent_sample_id"]): dict(row)
        for row in list(assignment.get("assignments") or ())
    }
    changed_by_capture: dict[str, Counter[str]] = defaultdict(Counter)
    changed_by_component: dict[str, Counter[str]] = defaultdict(Counter)
    parent_by_fold: Counter[int] = Counter()
    clutter_by_fold: Counter[int] = Counter()
    for row in parent_rows:
        parent_id = str(row["parent_sample_id"])
        source = assignment_by_parent[parent_id]
        fold_id = int(row["fold_id"])
        parent_by_fold[fold_id] += 1
        if int(row["true_id"]) == CLUTTER_ACTION:
            clutter_by_fold[fold_id] += 1
        if int(row["raw_action_id"]) != int(row["calibrated_action_id"]):
            changed_by_capture[str(source["capture_id"])][str(row["label_name"])] += 1
            changed_by_component[str(source["provenance_component_id"])][str(row["label_name"])] += 1
    return {
        "grouping": "capture_content_provenance_component",
        "capture_count": len({str(row["capture_id"]) for row in assignment_by_parent.values()}),
        "provenance_component_count": len(
            {str(row["provenance_component_id"]) for row in assignment_by_parent.values()}
        ),
        "parent_support_by_fold": {str(fold): int(parent_by_fold[fold]) for fold in range(FOLD_COUNT)},
        "clutter_parent_support_by_fold": {
            str(fold): int(clutter_by_fold[fold]) for fold in range(FOLD_COUNT)
        },
        "captures_with_parent_action_change": len(changed_by_capture),
        "components_with_parent_action_change": len(changed_by_component),
        "changed_parent_labels_by_capture": {
            capture: dict(sorted(labels.items()))
            for capture, labels in sorted(changed_by_capture.items())
        },
        "changed_parent_labels_by_component": {
            component: dict(sorted(labels.items()))
            for component, labels in sorted(changed_by_component.items())
        },
    }


def _prefix_safety(raw: Mapping[str, Any], calibrated: Mapping[str, Any]) -> dict[str, Any]:
    coverage_requirement = float(
        PREFIX_SAFETY_REQUIREMENTS["parent_balanced_eligible_prefix_action_coverage"]
    )
    values = {
        "coverage_at_least_requirement": float(calibrated["overall_action_coverage"])
        >= coverage_requirement,
        "selective_risk_not_higher_than_raw": float(calibrated["selective_risk"])
        <= float(raw["selective_risk"]),
        "known_target_to_clutter_rate_not_higher_than_raw": float(
            calibrated["known_target_to_clutter_rate"]
        )
        <= float(raw["known_target_to_clutter_rate"]),
        "clutter_to_known_target_rate_not_higher_than_raw": float(
            calibrated["clutter_to_known_target_rate"]
        )
        <= float(raw["clutter_to_known_target_rate"]),
    }
    return {
        "requirements": PREFIX_SAFETY_REQUIREMENTS,
        "checks": values,
        "passed": all(values.values()),
    }


def _fold_evaluation(
    parent_rows: Sequence[Mapping[str, Any]],
    prefix_rows: Sequence[Mapping[str, Any]],
    *,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "threshold_fit": dict(selection),
        "held_out_parent_final": {
            "raw_argmax": _action_metrics(
                parent_rows, action_key="raw_action_id", scope="held_out_parent_final_raw"
            ),
            "score_band": _action_metrics(
                parent_rows,
                action_key="calibrated_action_id",
                scope="held_out_parent_final_score_band",
            ),
        },
        "held_out_parent_balanced_eligible_prefix": {
            "raw_argmax": _action_metrics(
                prefix_rows,
                action_key="raw_action_id",
                scope="held_out_parent_balanced_eligible_prefix_raw",
                parent_balanced=True,
            ),
            "score_band": _action_metrics(
                prefix_rows,
                action_key="calibrated_action_id",
                scope="held_out_parent_balanced_eligible_prefix_score_band",
                parent_balanced=True,
            ),
        },
    }


def build_report(project_root: Path) -> dict[str, Any]:
    root = project_root.resolve()
    preparation = _require_preparation(root)
    _, assignment = load_fold_contract(root)
    source = _load_arm(root, "treatment")
    records_by_scope = {
        scope: [dict(row) for row in source["records_by_scope"][scope]]
        for scope in SCOPES
    }
    _validate_source_records(records_by_scope)
    thresholds = _cross_fit_thresholds(records_by_scope["parent"])
    pooled_oracle = _select_thresholds(
        records_by_scope["parent"], fit_fold_ids=list(range(FOLD_COUNT))
    )
    annotated = {
        scope: _annotate_actions(rows, scope=scope, thresholds=thresholds)
        for scope, rows in records_by_scope.items()
    }
    raw_metrics = _scope_metrics(annotated, action_key="raw_action_id")
    calibrated_metrics = _scope_metrics(annotated, action_key="calibrated_action_id")
    metrics_delta = {
        scope: _deltas(raw_metrics[scope], calibrated_metrics[scope])
        for scope in raw_metrics
    }
    per_fold: list[dict[str, Any]] = []
    for fold_id in range(FOLD_COUNT):
        parent_rows = [row for row in annotated["parent"] if int(row["fold_id"]) == fold_id]
        prefix_rows = [
            row for row in annotated["eligible_prefix"] if int(row["fold_id"]) == fold_id
        ]
        per_fold.append(
            {
                "fold_id": fold_id,
                **_fold_evaluation(parent_rows, prefix_rows, selection=thresholds[fold_id]),
            }
        )
    parent_raw = raw_metrics["parent_final"]
    parent_calibrated = calibrated_metrics["parent_final"]
    parent_values = _selection_values(parent_calibrated)
    parent_requirement_checks = {
        key: parent_values[key] >= float(requirement)
        for key, requirement in SELECTION_REQUIREMENTS.items()
    }
    parent_direction_checks = {
        "known_target_to_clutter_not_higher_than_raw": float(
            parent_calibrated["known_target_to_clutter_rate"]
        )
        <= float(parent_raw["known_target_to_clutter_rate"]),
        "clutter_to_known_target_not_higher_than_raw": float(
            parent_calibrated["clutter_to_known_target_rate"]
        )
        <= float(parent_raw["clutter_to_known_target_rate"]),
    }
    prefix = _prefix_safety(
        raw_metrics["parent_balanced_eligible_prefix"],
        calibrated_metrics["parent_balanced_eligible_prefix"],
    )
    parent_passed = all(parent_requirement_checks.values()) and all(
        parent_direction_checks.values()
    )
    candidate = parent_passed and bool(prefix["passed"])
    report = {
        "status": (
            "completed_cross_fitted_oof_calibration_candidate"
            if candidate
            else "completed_cross_fitted_oof_gate_failed"
        ),
        "schema_version": REPORT_SCHEMA,
        "experiment": "V24d L3 cross-fitted OOF score-band Pending calibration",
        "scope": (
            "known train provenance-grouped V21e treatment OOF only; no L1 retraining, "
            "validation, test, Unknown, full cascade, stable, rejection, or deployment"
        ),
        "source_l1": {
            "variant": "V21e causal-course treatment OOF; V24d reuses scores only",
            "source_oof": preparation["source"],
            "source_binary_metrics_by_scope": source["metrics_by_scope"],
        },
        "implementation_files_sha256": preparation["implementation"],
        "l3_action_semantics": {
            "raw_argmax_before_minimum_evidence": "PENDING_ABSTAIN",
            "score_band_after_minimum_evidence": True,
            "minimum_evidence_events": MINIMUM_EVIDENCE_EVENTS,
            "actions": list(ACTION_NAMES),
            "unknown_action_present": False,
            "pending_counts_as_unknown_rejection": False,
        },
        "cross_fit_protocol": {
            "fit_unit": "parent_final_OOF_only",
            "outer_protocol": "leave_one_outer_fold_out",
            "held_out_fold_labels_used_for_threshold_selection": False,
            "prefix_or_window_labels_used_for_threshold_selection": False,
            "candidate_space": {
                "method": "all score decision-equivalence breakpoints",
                "threshold_values": "unique parent-final OOF scores union {0.0, 1.0}",
                "pair_rule": (
                    "strict t_target < t_clutter plus the single raw-argmax "
                    "baseline (0.5, 0.5)"
                ),
                "uses_approximation_grid": False,
            },
            "selection_requirements": SELECTION_REQUIREMENTS,
            "selection_direction_requirements": DIRECTION_REQUIREMENTS,
            "fold_thresholds": [thresholds[fold_id] for fold_id in range(FOLD_COUNT)],
        },
        "pooled_oof_oracle_diagnostic": {
            "uses_all_oof_parent_labels": True,
            "not_cross_fitted": True,
            "not_an_adoption_or_threshold_selection_result": True,
            "purpose": (
                "distinguish complete score-band infeasibility from outer-fold threshold "
                "instability only"
            ),
            "selection": pooled_oracle,
        },
        "raw_argmax": {"metrics": raw_metrics},
        "score_band": {"metrics": calibrated_metrics},
        "score_band_minus_raw_argmax": metrics_delta,
        "fold_evaluations": per_fold,
        "parent_final_wilson_intervals": {
            "raw_argmax": _parent_wilson(parent_raw),
            "score_band": _parent_wilson(parent_calibrated),
            "clutter_parent_support": SOURCE_LABEL_SUPPORT["ClutterTarget"],
        },
        "paired_component_bootstrap": _component_bootstrap(assignment, annotated["parent"]),
        "parent_final_action_changes": _parent_changes(annotated["parent"]),
        "provenance_audit": _provenance_audit(annotated["parent"], assignment),
        "frozen_decision": {
            "all_parent_final_selection_requirements_passed": all(
                parent_requirement_checks.values()
            ),
            "parent_final_requirement_checks": parent_requirement_checks,
            "parent_final_wrong_direction_checks": parent_direction_checks,
            "parent_balanced_eligible_prefix_safety": prefix,
            "cross_fitted_l3_calibration_candidate": candidate,
            "pooled_oof_oracle_has_feasible_score_band": bool(
                pooled_oracle["fit_feasible"]
            ),
            "no_l1_model_change_or_retraining": True,
            "pending_is_not_unknown_rejection": True,
            "development_or_oof_only_no_release_claim": True,
        },
        "protocol_path": str(root / PROTOCOL),
        "protocol_sha256": sha256_file(root / PROTOCOL),
        "experiment_contract_path": str(root / EXPERIMENT_CONTRACT),
        "experiment_contract_sha256": sha256_file(root / EXPERIMENT_CONTRACT),
        "readiness_path": str(root / READINESS),
        "readiness_sha256": sha256_file(root / READINESS),
        "formal_training": "not_executed_not_applicable",
        "OOF": "reused_completed_V21e_treatment_predictions_only",
        "validation": "not_executed",
        "test": "not_executed",
        "formal_rejection": "not_executed",
        "full_cascade": "not_executed",
        "stable": "not_executed",
        "P95": "not_executed",
        "deployment_eligible": False,
    }
    _write_json(root / OUTPUT, report)
    return {**report, "report_path": str(root / OUTPUT)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V24d cross-fitted L3 score-band Pending calibration."
    )
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[2]
    )
    args = parser.parse_args()
    print(json.dumps(build_report(args.project_root.resolve()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()