"""Independent frozen-endpoint L1 to L2 to L3 test replay."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .protocol import V28_L2_CLASSES






def _wilson(successes: int, total: int) -> dict[str, Any]:
    if total <= 0:
        return {"estimate": None, "lower_95": None, "upper_95": None, "support": 0}
    estimate = successes / total
    z = 1.959963984540054
    denominator = 1.0 + z * z / total
    center = (estimate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(estimate * (1.0 - estimate) / total + z * z / (4.0 * total * total)) / denominator
    return {"estimate": float(estimate), "lower_95": float(max(0.0, center - radius)), "upper_95": float(min(1.0, center + radius)), "support": int(total)}


def _action_metrics(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    actions = ("DroneTarget", "BirdTarget", "BalloonTarget", "Clutter", "UNKNOWN_REJECTED", "PENDING_ABSTAIN")
    true_names = {"DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget"}
    rows: list[dict[str, Any]] = []
    matrix = np.zeros((len(true_names), len(actions)), dtype=np.int64)
    true_order = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget", "UnknownTarget")
    for record in records:
        true_name = str(record["label_name"])
        action = str(record["action"])
        if true_name not in true_names or action not in actions:
            raise ValueError("V28 replay action label escaped contract")
        matrix[true_order.index(true_name), actions.index(action)] += 1
    action_true_index = {
        "DroneTarget": 0,
        "BirdTarget": 1,
        "BalloonTarget": 2,
        "Clutter": 3,
        "UNKNOWN_REJECTED": 4,
    }
    for index, action in enumerate(actions):
        predicted = int(matrix[:, index].sum())
        true_index = action_true_index.get(action)
        true_support = int(matrix[true_index].sum()) if true_index is not None else None
        tp = int(matrix[true_index, index]) if true_index is not None else None
        precision = (tp / predicted) if tp is not None and predicted else None
        recall = (tp / true_support) if tp is not None and true_support else None
        f1 = (2.0 * precision * recall / (precision + recall)) if precision is not None and recall is not None and precision + recall else None
        rows.append({"action": action, "precision": None if precision is None else float(precision), "recall": None if recall is None else float(recall), "f1": None if f1 is None else float(f1), "predicted_support": predicted, "true_support": true_support, "true_positive": tp, "metric_defined": true_index is not None})
    total = int(matrix.sum())
    accepted = [record for record in records if record["action"] != "PENDING_ABSTAIN"]
    accepted_correct = sum((record["action"] == record["label_name"] or (record["label_name"] == "ClutterTarget" and record["action"] == "Clutter") or (record["label_name"] == "UnknownTarget" and record["action"] == "UNKNOWN_REJECTED")) for record in accepted)
    unknown = [record for record in records if record["label_name"] == "UnknownTarget"]
    clutter = [record for record in records if record["label_name"] == "ClutterTarget"]
    known = [record for record in records if record["label_name"] in V28_L2_CLASSES]
    known_by_class = {
        label: {
            "accepted": sum(record["action"] == label for record in known if record["label_name"] == label),
            "pending": sum(record["action"] == "PENDING_ABSTAIN" for record in known if record["label_name"] == label),
            "wrong_action": sum(record["action"] not in {label, "PENDING_ABSTAIN"} for record in known if record["label_name"] == label),
            "support": sum(record["label_name"] == label for record in known),
        }
        for label in V28_L2_CLASSES
    }
    for values in known_by_class.values():
        values["acceptance_rate"] = _wilson(values["accepted"], values["support"])
        values["false_rejection_rate"] = _wilson(values["support"] - values["accepted"], values["support"])
    return {"actions": rows, "true_labels": list(true_order), "action_labels": list(actions), "confusion_matrix_rows_true_columns_action": matrix.tolist(), "support": total, "pending_rate": _wilson(sum(record["action"] == "PENDING_ABSTAIN" for record in records), total), "coverage": _wilson(len(accepted), total), "accepted_accuracy": _wilson(accepted_correct, len(accepted)), "selective_risk": _wilson(len(accepted) - accepted_correct, len(accepted)), "unknown_rejection": {"rejected": sum(record["action"] == "UNKNOWN_REJECTED" for record in unknown), "error_acceptance": sum(record["action"] in V28_L2_CLASSES or record["action"] == "Clutter" for record in unknown), "pending": sum(record["action"] == "PENDING_ABSTAIN" for record in unknown), "rate": _wilson(sum(record["action"] == "UNKNOWN_REJECTED" for record in unknown), len(unknown))}, "clutter_acceptance": {"accepted": sum(record["action"] == "Clutter" for record in clutter), "misrouted": sum(record["action"] != "Clutter" for record in clutter), "rate": _wilson(sum(record["action"] == "Clutter" for record in clutter), len(clutter))}, "known_target_acceptance": {"accepted": sum(record["action"] == record["label_name"] for record in known), "pending": sum(record["action"] == "PENDING_ABSTAIN" for record in known), "wrong_action": sum(record["action"] not in {*V28_L2_CLASSES, "PENDING_ABSTAIN"} for record in known), "rate": _wilson(sum(record["action"] == record["label_name"] for record in known), len(known)), "by_class": known_by_class}}



