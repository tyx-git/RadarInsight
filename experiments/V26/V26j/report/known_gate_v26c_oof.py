"""Cross-fitted L3 calibration from strict V26c Known-only L1/L2 OOF scores.

V26c deliberately separates the two score-fitting populations:

* L1 target/clutter thresholds are fitted from four-fold parent-final OOF
  records across all four Known labels.
* L2 DBB-subclass acceptance is fitted only from four-fold Drone/Bird/Balloon
  records.  L2's inference-only Clutter rows are retained for route auditing,
  but never become an L2 threshold-selection signal.

The fifth fold is then scored with those frozen thresholds.  Unknown, validation
and test data are inaccessible here.  The resulting report is calibration
evidence only; five fold checkpoints cannot be presented as one deployable L3
endpoint and therefore no final registry is written.
"""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from radar_ln3.known_gate_v26c_oof import FOLD_COUNT, L1_KNOWN_TARGET_ACTION_RECALL_FLOOR, L2_CORRECT_ACTION_RECALL_FLOOR, CLUTTER_PRECISION_FLOOR, CLUTTER_RECALL_FLOOR, MINIMUM_EVIDENCE_EVENTS, V21E_ASSIGNMENT
from radar_ln3.open_set_architecture import ProductAction, resolve_semantic_action


LABELS = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
DBB_LABELS = LABELS[:3]
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 26063
GRID = tuple(round(value / 100.0, 2) for value in range(50, 100))
SELECTION_GRID_CHUNK = 8
MAX_DETAIL_BYTES = 2 * 1024 * 1024


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"V26c expected JSON object: {path}")
    return value




def _write_new_or_identical_text(path: Path, text: str, *, description: str) -> str:
    """Atomically write a bounded immutable report/detail artifact."""

    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite different V26c {description}: {path}")
        return "reused_identical"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        if temporary.read_text(encoding="utf-8") != text:
            raise FileExistsError(
                "refusing to replace a different interrupted V26c "
                f"{description}: {temporary}"
            )
    else:
        temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return "created"










def _validated_probability(value: Any, *, field: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"V26c score is non-finite or out of range: {field}")
    return result


def _joint_rows(l1: Mapping[str, Any], l2: Mapping[str, Any], fold_id: int) -> dict[str, list[dict[str, Any]]]:
    joint = dict(l2.get("joint_l1_l2_scores") or {})
    raw_records = dict(joint.get("records") or {})
    records = {name: [dict(row) for row in list(raw_records.get(name) or ())] for name in ("parent", "eligible_prefix", "window")}
    if (
        set(raw_records) != set(records)
        or not all(records.values())
        or joint.get("l1_scores_reused_from_sealed_prediction") is not True
        or int(joint.get("l1_forward_count", -1)) != 0
        or int(joint.get("l2_held_out_forward_count", -1)) != 1
        or joint.get("mixed_l2_inference_only_for_held_out_clutter") is not True
    ):
        raise ValueError(f"V26c fold {fold_id} has incomplete joint L1/L2 records")
    l1_records = dict(l1.get("records") or {})
    if set(l1_records) != {"window", "eligible_prefix", "parent"}:
        raise ValueError(f"V26c fold {fold_id} sealed L1 record scopes differ")
    for scope in ("window", "eligible_prefix", "parent"):
        l1_keys = {
            (str(row.get("parent_sample_id", "")), int(row.get("event_index", -1)))
            for row in list(l1_records.get(scope) or ())
        }
        joint_keys = {
            (str(row.get("parent_sample_id", "")), int(row.get("event_index", -1)))
            for row in records[scope]
        }
        if not l1_keys or l1_keys != joint_keys:
            raise ValueError(
                f"V26c fold {fold_id} sealed L1 and joint {scope} record keys differ"
            )
    l1_parent = {str(row["parent_sample_id"]): dict(row) for row in list(l1_records.get("parent") or ())}
    joint_parent = {str(row["parent_sample_id"]): dict(row) for row in records["parent"]}
    if set(l1_parent) != set(joint_parent) or len(l1_parent) != len(joint_parent):
        raise ValueError(f"V26c fold {fold_id} L1 and joint parent coverage differs")
    for scope, rows in records.items():
        if not rows:
            raise ValueError(f"V26c fold {fold_id} lacks {scope} records")
        keys = [
            (str(row.get("parent_sample_id", "")), int(row.get("event_index", -1)))
            for row in rows
        ]
        if len(keys) != len(set(keys)):
            raise ValueError(f"V26c fold {fold_id} repeats a {scope} parent/event record")
        for row in rows:
            label = str(row.get("label_name", ""))
            target = _validated_probability(row.get("l1_known_target_probability"), field="l1_known_target_probability")
            clutter = _validated_probability(row.get("l1_known_clutter_probability"), field="l1_known_clutter_probability")
            subclass = _validated_probability(row.get("l2_subclass_probability"), field="l2_subclass_probability")
            if (
                int(row.get("fold_id", -1)) != fold_id
                or label not in LABELS
                or abs(target + clutter - 1.0) > 1.0e-5
                or str(row.get("l2_top_subclass", "")) not in DBB_LABELS
                or not isinstance(row.get("l2_inference_only_for_clutter"), bool)
                or bool(row.get("l2_inference_only_for_clutter")) != (label == "ClutterTarget")
                or int(row.get("l2_training_exposure_for_this_parent", -1))
                != (0 if label == "ClutterTarget" else 1)
                or not isinstance(row.get("product_action_eligible"), bool)
                or int(row.get("window_length", 0)) < 1
                or int(row.get("event_index", -1)) != int(row.get("window_length", 0)) - 1
                or bool(row.get("product_action_eligible"))
                != (int(row.get("window_length", 0)) >= MINIMUM_EVIDENCE_EVENTS)
            ):
                raise ValueError(f"V26c fold {fold_id} joint score record is malformed")
            row["l1_known_target_probability"] = target
            row["l1_known_clutter_probability"] = clutter
            row["l2_subclass_probability"] = subclass
    for parent_id, row in joint_parent.items():
        left = l1_parent[parent_id]
        if (
            str(row["label_name"]) != str(left["label_name"])
            or abs(float(row["l1_known_target_probability"]) - float(left["known_target_probability"])) > 1.0e-6
            or abs(float(row["l1_known_clutter_probability"]) - float(left["known_clutter_probability"])) > 1.0e-6
        ):
            raise ValueError(f"V26c fold {fold_id} joint L1 score differs at parent={parent_id}")
    return records


def _assignment_by_parent(root: Path, all_parent_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    path = root / V21E_ASSIGNMENT
    assignment = _read(path)
    if (
        assignment.get("schema_version") != "RADAR-LN3-V21e-known-gate-capture-content-grouped-oof-assignment-v1"
        or assignment.get("grouping") != "capture_content_provenance_component"
        or int(assignment.get("known_parent_count", -1)) != len(all_parent_ids)
        or int(assignment.get("unknown_training_exposure", -1)) != 0
        or int(assignment.get("unknown_oof_prediction_count", -1)) != 0
        or int(assignment.get("validation_parent_assignment_count", -1)) != 0
        or int(assignment.get("test_parent_assignment_count", -1)) != 0
    ):
        raise ValueError("V26c provenance assignment is invalid")
    rows = {str(row["parent_sample_id"]): dict(row) for row in list(assignment.get("assignments") or ())}
    if set(rows) != set(all_parent_ids) or len(rows) != len(all_parent_ids):
        raise ValueError("V26c OOF parents do not match the frozen provenance assignment")
    components: dict[str, int] = Counter()
    for parent_id, row in rows.items():
        if (
            str(row.get("label_name", "")) not in LABELS
            or not str(row.get("provenance_component_id", ""))
            or int(row.get("fold_id", -1)) not in range(FOLD_COUNT)
        ):
            raise ValueError(f"V26c provenance assignment row is malformed: {parent_id}")
        components[str(row["provenance_component_id"])] += 1
    if len(components) != int(assignment.get("provenance_component_count", -1)) or max(components.values()) > 2:
        raise ValueError("V26c provenance component assignment cardinality differs")
    return rows


def _annotate_provenance(rows: Sequence[Mapping[str, Any]], assignment: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        parent_id = str(row["parent_sample_id"])
        provenance = dict(assignment[parent_id])
        if (
            str(row["label_name"]) != str(provenance["label_name"])
            or int(row["fold_id"]) != int(provenance["fold_id"])
        ):
            raise ValueError(f"V26c score/provenance identity differs for {parent_id}")
        row["provenance_component_id"] = str(provenance["provenance_component_id"])
        row["capture_id"] = str(provenance["capture_id"])
        result.append(row)
    return result


def _actions(rows: Sequence[Mapping[str, Any]], thresholds: Mapping[str, float]) -> list[dict[str, Any]]:
    required = {
        "l1_known_target_acceptance",
        "l1_known_clutter_acceptance",
        "l2_subclass_acceptance",
    }
    if set(thresholds) != required:
        raise ValueError("V26c L3 threshold fields differ from the protocol")
    target_threshold = _validated_probability(thresholds["l1_known_target_acceptance"], field="l1 target threshold")
    clutter_threshold = _validated_probability(thresholds["l1_known_clutter_acceptance"], field="l1 clutter threshold")
    subclass_threshold = _validated_probability(thresholds["l2_subclass_acceptance"], field="l2 subclass threshold")
    if target_threshold + clutter_threshold <= 1.0:
        raise ValueError("V26c L1 thresholds would permit dual evidence acceptance")
    result: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        label = str(row["label_name"])
        l2_is_inference_only_for_clutter = bool(row["l2_inference_only_for_clutter"])
        if l2_is_inference_only_for_clutter != (label == "ClutterTarget"):
            raise ValueError(
                "V26c L2 inference-only marker does not match the Known label"
            )
        known_target_accepted = (
            float(row["l1_known_target_probability"]) >= target_threshold
        )
        known_clutter_accepted = (
            float(row["l1_known_clutter_probability"]) >= clutter_threshold
        )
        # L2 is a DBB *subclass* specialist, not a second route selector.  It
        # can supply evidence only after L1 has accepted the KnownTarget route.
        # A held-out Clutter row is also materialized through L2 solely to
        # reproduce the physical call path.  Suppressing both cases here keeps
        # "background accepted" mutually exclusive with DBB evidence and makes
        # the vectorized L1 threshold grid exactly match product actions.
        use_l2_subclass_evidence = (
            not l2_is_inference_only_for_clutter and known_target_accepted
        )
        dbb_action = (
            None if not use_l2_subclass_evidence
            else ProductAction(str(row["l2_top_subclass"]))
        )
        dbb_subclass_accepted = (
            False
            if not use_l2_subclass_evidence
            else float(row["l2_subclass_probability"]) >= subclass_threshold
        )
        decision = resolve_semantic_action(
            observation_eligible=bool(row["product_action_eligible"]),
            known_target_accepted=known_target_accepted,
            known_clutter_accepted=known_clutter_accepted,
            dbb_subclass=dbb_action,
            dbb_subclass_accepted=dbb_subclass_accepted,
        )
        row.update(
            {
                "semantic_action": decision.action.value,
                "semantic_reason": decision.reason,
                "known_target_accepted": decision.known_target_accepted,
                "known_clutter_accepted": decision.known_clutter_accepted,
                "dbb_subclass_accepted": decision.dbb_subclass_accepted,
                "l2_subclass_evidence_used_for_action": use_l2_subclass_evidence,
                "l1_known_target_acceptance_threshold": target_threshold,
                "l1_known_clutter_acceptance_threshold": clutter_threshold,
                "l2_subclass_acceptance_threshold": subclass_threshold,
            }
        )
        result.append(row)
    return result


def _safe_divide(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    parent_balanced: bool = False,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("V26c cannot summarize empty action records")
    action_names = (*DBB_LABELS, "ClutterTarget", "UNKNOWN_REJECTED", "PENDING_ABSTAIN")
    matrix = np.zeros((len(LABELS), len(action_names)), dtype=np.float64)
    label_index = {name: index for index, name in enumerate(LABELS)}
    action_index = {name: index for index, name in enumerate(action_names)}
    row_weights = np.ones(len(rows), dtype=np.float64)
    if parent_balanced:
        parent_counts = Counter(str(row["parent_sample_id"]) for row in rows)
        row_weights = np.asarray(
            [1.0 / float(parent_counts[str(row["parent_sample_id"])]) for row in rows],
            dtype=np.float64,
        )
    for row in rows:
        label, action = str(row["label_name"]), str(row["semantic_action"])
        if label not in label_index or action not in action_index:
            raise ValueError("V26c action record is outside its Known action space")
    for row_index, row in enumerate(rows):
        label, action = str(row["label_name"]), str(row["semantic_action"])
        matrix[label_index[label], action_index[action]] += row_weights[row_index]
    correct = np.asarray([matrix[index, index] for index in range(len(LABELS))], dtype=np.float64)
    per_class: list[dict[str, Any]] = []
    for index, label in enumerate(LABELS):
        true_support = float(matrix[index].sum())
        predicted = float(matrix[:, index].sum())
        precision = _safe_divide(float(correct[index]), predicted)
        recall = _safe_divide(float(correct[index]), true_support)
        f1 = (
            _safe_divide(2.0 * precision * recall, precision + recall)
            if precision is not None and recall is not None and precision + recall > 0.0
            else None
        )
        per_class.append(
            {
                "class": label,
                "support": int(true_support) if not parent_balanced else true_support,
                "accepted_as_class": int(predicted) if not parent_balanced else predicted,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "unknown_rejected": int(matrix[index, 4]) if not parent_balanced else float(matrix[index, 4]),
                "pending_abstain": int(matrix[index, 5]) if not parent_balanced else float(matrix[index, 5]),
            }
        )
    known_target_support = float(matrix[:3].sum())
    known_target_correct = float(sum(matrix[index, index] for index in range(3)))
    clutter_support = float(matrix[3].sum())
    clutter_correct = float(matrix[3, 3])
    false_clutter = float(matrix[:3, 3].sum())
    accepted_total = float(matrix[:, :4].sum())
    return {
        "scope": scope,
        "event_support": len(rows),
        "parent_support": len({str(row["parent_sample_id"]) for row in rows}),
        "aggregation": "parent_balanced" if parent_balanced else "event",
        "action_names": list(action_names),
        "confusion_rows_true_known_label_columns_semantic_action": matrix.tolist(),
        "per_class": per_class,
        "known_target_recall": _safe_divide(float(known_target_correct), float(known_target_support)),
        "clutter_precision": _safe_divide(float(clutter_correct), float(matrix[:, 3].sum())),
        "clutter_recall": _safe_divide(float(clutter_correct), float(clutter_support)),
        "known_to_clutter_error_rate": _safe_divide(float(false_clutter), float(known_target_support)),
        "clutter_to_known_error_rate": _safe_divide(float(matrix[3, :3].sum()), float(clutter_support)),
        "unknown_rejected_known_count": int(matrix[:, 4].sum()) if not parent_balanced else float(matrix[:, 4].sum()),
        "pending_known_count": int(matrix[:, 5].sum()) if not parent_balanced else float(matrix[:, 5].sum()),
        "accepted_action_accuracy": _safe_divide(float(np.trace(matrix[:, :4])), float(matrix.sum())),
        "selective_accuracy": _safe_divide(float(np.trace(matrix[:, :4])), accepted_total),
        "action_coverage": _safe_divide(accepted_total, float(matrix.sum())),
    }


def _l1_score_arrays(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([str(row["label_name"]) for row in rows], dtype=object)
    target_scores = np.asarray([float(row["l1_known_target_probability"]) for row in rows], dtype=np.float64)
    clutter_scores = np.asarray([float(row["l1_known_clutter_probability"]) for row in rows], dtype=np.float64)
    if not np.isfinite(target_scores).all() or not np.isfinite(clutter_scores).all() or np.any(np.abs(target_scores + clutter_scores - 1.0) > 1.0e-5):
        raise ValueError("V26c L1 fit scores are invalid")
    return labels, np.column_stack((target_scores, clutter_scores))


def _l1_candidate_metrics(
    labels: np.ndarray,
    target_scores: np.ndarray,
    clutter_scores: np.ndarray,
    *,
    target_thresholds: np.ndarray,
    clutter_thresholds: np.ndarray,
    l2_subclass_accepted: np.ndarray,
    l2_correct_subclass_accepted: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate one threshold-grid block without per-record Python dispatch.

    For parent-final records every parent is eligible.  The resolver's action
    can consequently be represented with three broadcast boolean arrays:
    clutter acceptance, target-plus-subclass-independent route eligibility,
    and neither.  L2 acceptance is optimized separately, so this function only
    produces L1 statistics used to rank the L1 grid.  Both L2 boolean arrays
    are deliberately false for Clutter rows: inference-only Clutter L2 scores
    cannot influence threshold selection.
    """

    shape = (target_thresholds.size, clutter_thresholds.size)
    target_accept = target_scores[None, None, :] >= target_thresholds[:, None, None]
    clutter_accept = clutter_scores[None, None, :] >= clutter_thresholds[None, :, None]
    if (
        l2_subclass_accepted.shape != labels.shape
        or l2_correct_subclass_accepted.shape != labels.shape
        or bool((l2_correct_subclass_accepted & ~l2_subclass_accepted).any())
        or bool(l2_subclass_accepted[labels == "ClutterTarget"].any())
        or bool(l2_correct_subclass_accepted[labels == "ClutterTarget"].any())
    ):
        raise ValueError("V26c L1 fit received invalid or Clutter-derived L2 evidence")
    # Materialize the pair grid before reducing over parents.  Reducing the
    # unbroadcast singleton dimensions would silently return [1,C] or [T,1]
    # and mis-bind a target threshold to the wrong clutter threshold.
    target_accept = np.broadcast_to(target_accept, (target_thresholds.size, clutter_thresholds.size, labels.size))
    clutter_accept = np.broadcast_to(clutter_accept, (target_thresholds.size, clutter_thresholds.size, labels.size))
    label_masks = {label: labels == label for label in LABELS}
    target_mask = labels != "ClutterTarget"
    clutter_predictions = clutter_accept
    clutter_tp = clutter_predictions[:, :, label_masks["ClutterTarget"]].sum(axis=2, dtype=np.int64)
    clutter_predicted = clutter_predictions.sum(axis=2, dtype=np.int64)
    known_to_clutter = clutter_predictions[:, :, labels != "ClutterTarget"].sum(axis=2, dtype=np.int64)
    clutter_to_target_route = target_accept[:, :, label_masks["ClutterTarget"]].sum(axis=2, dtype=np.int64)
    target_final_action = (
        target_accept[:, :, target_mask]
        & l2_subclass_accepted[None, None, target_mask]
    ).sum(axis=2, dtype=np.int64)
    target_correct_action = (
        target_accept[:, :, target_mask]
        & l2_correct_subclass_accepted[None, None, target_mask]
    ).sum(axis=2, dtype=np.int64)
    return {
        "shape": np.asarray(shape, dtype=np.int64),
        "clutter_tp": clutter_tp,
        "clutter_predicted": clutter_predicted,
        "known_to_clutter": known_to_clutter,
        "clutter_to_target_route": clutter_to_target_route,
        "target_final_action": target_final_action,
        "target_correct_action": target_correct_action,
    }


def _l1_selection_key(
    *,
    clutter_precision: float,
    clutter_recall: float,
    known_target_final_action_recall: float,
    known_to_clutter_error_rate: float,
    clutter_to_target_route_rate: float,
    target_threshold: float,
    clutter_threshold: float,
) -> tuple[float, ...]:
    feasible = float(
        clutter_precision > CLUTTER_PRECISION_FLOOR
        and clutter_recall >= CLUTTER_RECALL_FLOOR
        and known_target_final_action_recall >= L1_KNOWN_TARGET_ACTION_RECALL_FLOOR
    )
    violation = (
        max(0.0, CLUTTER_PRECISION_FLOOR - clutter_precision + np.finfo(np.float64).eps)
        + max(0.0, CLUTTER_RECALL_FLOOR - clutter_recall)
        + max(0.0, L1_KNOWN_TARGET_ACTION_RECALL_FLOOR - known_target_final_action_recall)
    )
    return (
        feasible,
        -violation,
        min(clutter_precision, clutter_recall, known_target_final_action_recall),
        known_target_final_action_recall,
        clutter_precision,
        clutter_recall,
        -known_to_clutter_error_rate,
        -clutter_to_target_route_rate,
        -(target_threshold + clutter_threshold),
    )


def _fit_l1_thresholds(
    rows: Sequence[Mapping[str, Any]], *, l2_subclass_threshold: float
) -> tuple[dict[str, float], dict[str, Any]]:
    if not rows:
        raise ValueError("V26c cannot fit L1 thresholds without parent-final rows")
    labels, scores = _l1_score_arrays(rows)
    target_scores, clutter_scores = scores[:, 0], scores[:, 1]
    supports = Counter(labels.tolist())
    if set(supports) != set(LABELS) or any(supports[label] == 0 for label in LABELS):
        raise ValueError("V26c L1 threshold fitting lacks Known label support")
    l2_subclass_threshold = _validated_probability(
        l2_subclass_threshold, field="fixed L2 subclass threshold"
    )
    l2_subclass_accepted = np.zeros(labels.shape, dtype=np.bool_)
    l2_correct_subclass_accepted = np.zeros(labels.shape, dtype=np.bool_)
    dbb_mask = labels != "ClutterTarget"
    dbb_rows = [row for row, keep in zip(rows, dbb_mask, strict=True) if bool(keep)]
    l2_scores = np.asarray(
        [float(row["l2_subclass_probability"]) for row in dbb_rows], dtype=np.float64
    )
    l2_top = np.asarray([str(row["l2_top_subclass"]) for row in dbb_rows], dtype=object)
    if (
        not np.isfinite(l2_scores).all()
        or np.any(l2_scores < 0.0)
        or np.any(l2_scores > 1.0)
        or any(value not in DBB_LABELS for value in l2_top.tolist())
    ):
        raise ValueError("V26c L1 threshold fitting received invalid DBB-only L2 scores")
    l2_subclass_accepted[dbb_mask] = l2_scores >= l2_subclass_threshold
    l2_correct_subclass_accepted[dbb_mask] = (
        (l2_scores >= l2_subclass_threshold) & (l2_top == labels[dbb_mask])
    )
    best_key: tuple[float, ...] | None = None
    best_thresholds: dict[str, float] | None = None
    best_summary: dict[str, Any] | None = None
    candidate_count = 0
    for target_start in range(0, len(GRID), SELECTION_GRID_CHUNK):
        target_values = np.asarray(GRID[target_start : target_start + SELECTION_GRID_CHUNK], dtype=np.float64)
        clutter_values = np.asarray(GRID, dtype=np.float64)
        legal = target_values[:, None] + clutter_values[None, :] > 1.0
        if not bool(legal.any()):
            continue
        values = _l1_candidate_metrics(
            labels,
            target_scores,
            clutter_scores,
            target_thresholds=target_values,
            clutter_thresholds=clutter_values,
            l2_subclass_accepted=l2_subclass_accepted,
            l2_correct_subclass_accepted=l2_correct_subclass_accepted,
        )
        for target_index, target_value in enumerate(target_values.tolist()):
            for clutter_index, clutter_value in enumerate(clutter_values.tolist()):
                if not bool(legal[target_index, clutter_index]):
                    continue
                candidate_count += 1
                clutter_tp = int(values["clutter_tp"][target_index, clutter_index])
                clutter_predicted = int(values["clutter_predicted"][target_index, clutter_index])
                known_to_clutter = int(values["known_to_clutter"][target_index, clutter_index])
                clutter_to_target_route = int(
                    values["clutter_to_target_route"][target_index, clutter_index]
                )
                target_correct_action = int(
                    values["target_correct_action"][target_index, clutter_index]
                )
                clutter_precision = _safe_divide(float(clutter_tp), float(clutter_predicted)) or 0.0
                clutter_recall = _safe_divide(float(clutter_tp), float(supports["ClutterTarget"])) or 0.0
                known_to_clutter_rate = _safe_divide(float(known_to_clutter), float(len(rows) - supports["ClutterTarget"])) or 0.0
                clutter_to_target_route_rate = _safe_divide(
                    float(clutter_to_target_route), float(supports["ClutterTarget"])
                ) or 0.0
                known_target_final_action_recall = _safe_divide(
                    float(target_correct_action), float(len(rows) - supports["ClutterTarget"])
                ) or 0.0
                key = _l1_selection_key(
                    clutter_precision=clutter_precision,
                    clutter_recall=clutter_recall,
                    known_target_final_action_recall=known_target_final_action_recall,
                    known_to_clutter_error_rate=known_to_clutter_rate,
                    clutter_to_target_route_rate=clutter_to_target_route_rate,
                    target_threshold=float(target_value),
                    clutter_threshold=float(clutter_value),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_thresholds = {
                        "l1_known_target_acceptance": float(target_value),
                        "l1_known_clutter_acceptance": float(clutter_value),
                    }
                    best_summary = {
                        "clutter_precision": clutter_precision,
                        "clutter_recall": clutter_recall,
                        "known_target_final_action_recall": known_target_final_action_recall,
                        "known_to_clutter_error_rate": known_to_clutter_rate,
                        "clutter_to_target_l1_route_rate": clutter_to_target_route_rate,
                        "known_target_final_action_coverage": _safe_divide(
                            float(values["target_final_action"][target_index, clutter_index]),
                            float(len(rows) - supports["ClutterTarget"]),
                        ),
                    }
    if best_key is None or best_thresholds is None or best_summary is None:
        raise RuntimeError("V26c L1 grid has no legal threshold candidate")
    return best_thresholds, {
        "fit_parent_count": len(rows),
        "fit_label_parent_support": dict(sorted(supports.items())),
        "candidate_space": "pre-registered 0.50..0.99 L1 score grid with target+clutter>1",
        "candidate_count": candidate_count,
        "selection_key": list(best_key),
        "fit_l1_route_metrics": best_summary,
        "feasible_under_l1_clutter_and_final_known_target_safety": bool(best_key[0] > 0.0),
        "fixed_dbb_subclass_threshold": l2_subclass_threshold,
        "l2_inference_only_clutter_rows_used": False,
    }


def _l2_selection_key(
    *,
    correct_action_recall: float,
    selective_accuracy: float,
    action_coverage: float,
    threshold: float,
) -> tuple[float, ...]:
    # The L2 specialist has no Clutter class.  Preserve a high conditional
    # correct-action recall while preferring the least restrictive evidence
    # threshold; any withheld target below it becomes strict Unknown rather
    # than falling back to Clutter.
    feasible = float(correct_action_recall >= L2_CORRECT_ACTION_RECALL_FLOOR)
    violation = max(0.0, L2_CORRECT_ACTION_RECALL_FLOOR - correct_action_recall)
    return (
        feasible,
        -violation,
        selective_accuracy,
        action_coverage,
        correct_action_recall,
        -threshold,
    )


def _fit_l2_threshold(rows: Sequence[Mapping[str, Any]]) -> tuple[float, dict[str, Any]]:
    fit_rows = [dict(row) for row in rows if str(row["label_name"]) in DBB_LABELS]
    if len(fit_rows) != len(rows):
        raise ValueError("V26c L2 threshold fitting received inference-only Clutter rows")
    labels = np.asarray([str(row["label_name"]) for row in fit_rows], dtype=object)
    top = np.asarray([str(row["l2_top_subclass"]) for row in fit_rows], dtype=object)
    scores = np.asarray([float(row["l2_subclass_probability"]) for row in fit_rows], dtype=np.float64)
    if not np.isfinite(scores).all() or np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("V26c L2 threshold scores are invalid")
    support = Counter(labels.tolist())
    if any(support[label] == 0 for label in DBB_LABELS):
        raise ValueError("V26c L2 threshold fitting lacks a DBB label")
    correct = top == labels
    best_key: tuple[float, ...] | None = None
    best_threshold: float | None = None
    best_summary: dict[str, Any] | None = None
    for threshold in GRID:
        accepted = scores >= threshold
        correct_accepted = int(np.count_nonzero(accepted & correct))
        correct_action_recall = _safe_divide(float(correct_accepted), float(len(fit_rows))) or 0.0
        action_coverage = _safe_divide(float(np.count_nonzero(accepted)), float(len(fit_rows))) or 0.0
        selective_accuracy = _safe_divide(
            float(correct_accepted), float(np.count_nonzero(accepted))
        ) or 0.0
        key = _l2_selection_key(
            correct_action_recall=correct_action_recall,
            selective_accuracy=selective_accuracy,
            action_coverage=action_coverage,
            threshold=float(threshold),
        )
        if best_key is None or key > best_key:
            best_key, best_threshold = key, float(threshold)
            best_summary = {
                "dbb_correct_action_recall": correct_action_recall,
                "dbb_selective_accuracy": selective_accuracy,
                "dbb_subclass_action_coverage": action_coverage,
                "dbb_wrong_subclass_accepted_count": int(np.count_nonzero(accepted & ~correct)),
                "dbb_unknown_due_to_low_subclass_evidence_count": int(np.count_nonzero(~accepted)),
            }
    if best_key is None or best_threshold is None or best_summary is None:
        raise RuntimeError("V26c L2 grid has no threshold candidate")
    return best_threshold, {
        "fit_parent_count": len(fit_rows),
        "fit_label_parent_support": dict(sorted(support.items())),
        "candidate_space": "pre-registered 0.50..0.99 L2 top-subclass probability grid",
        "candidate_count": len(GRID),
        "selection_key": list(best_key),
        "fit_l2_subclass_metrics": best_summary,
        "feasible_under_dbb_correct_action_recall": bool(best_key[0] > 0.0),
        "inference_only_clutter_rows_excluded": True,
    }


def _fit_thresholds(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, float], dict[str, Any]]:
    l2, l2_fit = _fit_l2_threshold([row for row in rows if str(row["label_name"]) in DBB_LABELS])
    l1, l1_fit = _fit_l1_thresholds(rows, l2_subclass_threshold=l2)
    thresholds = {**l1, "l2_subclass_acceptance": l2}
    return thresholds, {
        "l1": l1_fit,
        "l2": l2_fit,
        "feasible": bool(
            l1_fit["feasible_under_l1_clutter_and_final_known_target_safety"]
            and l2_fit["feasible_under_dbb_correct_action_recall"]
        ),
    }


def _ci(values: Iterable[float]) -> dict[str, float]:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("V26c bootstrap contains no finite values")
    return {
        "median": float(np.median(data)),
        "lower_95": float(np.quantile(data, 0.025)),
        "upper_95": float(np.quantile(data, 0.975)),
    }


def _bootstrap(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Bootstrap already-calibrated parent actions without refitting thresholds.

    The outer-fold thresholds have already been applied before this function
    is called.  Resampling those action records preserves the cross-fitted
    operating point; refitting or silently selecting a pooled threshold here
    would turn the confidence interval into a different (and optimistic)
    estimand.
    """
    parents = {str(row["parent_sample_id"]): dict(row) for row in rows}
    if len(parents) != len(rows) or len(parents) != 986:
        raise ValueError("V26c component bootstrap parent coverage is invalid")
    components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    component_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for parent_id in sorted(parents):
        row = parents[parent_id]
        component = str(row.get("provenance_component_id", ""))
        if not component:
            raise ValueError("V26c bootstrap record lacks provenance component")
        components[component].append(row)
        component_labels[component][str(row["label_name"])] += 1
    strata: dict[str, list[str]] = defaultdict(list)
    for component, labels in component_labels.items():
        signature = "+".join(f"{label}:{labels[label]}" for label in sorted(labels))
        strata[signature].append(component)
    if len(components) != 975:
        raise ValueError("V26c component bootstrap cardinality differs from V21e assignment")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values: dict[str, list[float]] = defaultdict(list)
    for _ in range(BOOTSTRAP_REPLICATES):
        selected_components: list[str] = []
        for component_ids in strata.values():
            selected_components.extend(rng.choice(component_ids, size=len(component_ids), replace=True).tolist())
        sampled: list[dict[str, Any]] = []
        for component in selected_components:
            sampled.extend(components[component])
        metrics = _metrics(sampled, scope="parent_final_component_bootstrap")
        for name in (
            "clutter_precision",
            "clutter_recall",
            "known_target_recall",
            "accepted_action_accuracy",
            "known_to_clutter_error_rate",
            "clutter_to_known_error_rate",
            "action_coverage",
        ):
            value = metrics[name]
            if value is not None:
                values[name].append(float(value))
    return {
        "method": "capture/content provenance-component bootstrap stratified by component label signature; cross-fitted action fixed per sampled parent fold",
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
        "confidence_level": 0.95,
        "component_count": len(components),
        "stratum_component_counts": {name: len(items) for name, items in sorted(strata.items())},
        "metrics": {name: _ci(items) for name, items in sorted(values.items())},
    }


def _write_detail_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    fields = (
        "fold_id",
        "provenance_component_id",
        "capture_id",
        "parent_sample_id",
        "label_name",
        "l1_known_target_probability",
        "l1_known_clutter_probability",
        "l1_known_target_acceptance_threshold",
        "l1_known_clutter_acceptance_threshold",
        "l2_top_subclass",
        "l2_subclass_probability",
        "l2_subclass_acceptance_threshold",
        "l2_subclass_evidence_used_for_action",
        "semantic_action",
        "semantic_reason",
        "l2_inference_only_for_clutter",
    )
    lines: list[str] = []
    # Use csv.writer against an in-memory stream only for the tiny parent table.
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in sorted(rows, key=lambda value: (int(value["fold_id"]), str(value["parent_sample_id"]))):
        writer.writerow({field: row.get(field) for field in fields})
    text = stream.getvalue()
    if len(text.encode("utf-8")) > MAX_DETAIL_BYTES:
        raise MemoryError("V26c parent detail CSV exceeds its bounded size")
    return _write_new_or_identical_text(path, text, description="parent detail CSV")







