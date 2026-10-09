"""Excel TR 逐点/逐事件适配与 r20 只读诊断。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .denoising import DenoisingCalibration, SequenceDenoisingResult, score_sequence


@dataclass(frozen=True)
class PointwiseEvent:
    parent_sample_id: str
    event_index: int
    label_name: str
    values: Mapping[str, Any]


def event_rows(sequence: Any) -> tuple[PointwiseEvent, ...]:
    parent_id = str(getattr(sequence, "parent_sample_id", ""))
    label_name = str(getattr(sequence, "label_name", ""))
    rows = tuple(getattr(sequence, "event_rows", ()) or ())
    return tuple(
        PointwiseEvent(parent_id, index, label_name, dict(row))
        for index, row in enumerate(rows)
    )


def score_eventwise(
    calibration: DenoisingCalibration,
    sequence: Any,
) -> tuple[tuple[PointwiseEvent, ...], SequenceDenoisingResult]:
    """逐点返回事件行与原 r20 同口径分数。"""

    points = event_rows(sequence)
    result = score_sequence(calibration, sequence)
    if len(points) != len(result.keep_weight):
        raise ValueError("逐点事件数与降噪输出长度不一致")
    return points, result


def apply_keep_weight_to_event_rows(
    sequence: Any,
    result: SequenceDenoisingResult,
) -> list[dict[str, Any]]:
    """只对输入事件值做软缩放，保留事件顺序和 row 字段。"""

    rows = [dict(row) for row in (getattr(sequence, "event_rows", ()) or ())]
    if len(rows) != len(result.keep_weight):
        raise ValueError("事件行与 keep_weight 长度不一致")
    output: list[dict[str, Any]] = []
    for row, weight in zip(rows, result.keep_weight, strict=True):
        updated = dict(row)
        updated["r01_keep_weight"] = float(weight)
        updated["r01_remove_score"] = float(result.remove_score[len(output)])
        output.append(updated)
    return output


def summarize_identity_effect(
    before_logits: np.ndarray,
    after_logits: np.ndarray,
    before_pred: np.ndarray,
    after_pred: np.ndarray,
) -> dict[str, Any]:
    """判定逐点门控是否真正影响最终 parent 分类。"""

    before_logits = np.asarray(before_logits, dtype=np.float64)
    after_logits = np.asarray(after_logits, dtype=np.float64)
    before_pred = np.asarray(before_pred, dtype=np.int64)
    after_pred = np.asarray(after_pred, dtype=np.int64)
    if before_logits.shape != after_logits.shape:
        raise ValueError("before/after logits shape differs")
    if before_pred.shape != after_pred.shape:
        raise ValueError("before/after prediction shape differs")
    delta = after_logits - before_logits
    return {
        "sample_count": int(len(before_pred)),
        "logit_max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "logit_mean_abs_delta": float(np.mean(np.abs(delta))) if delta.size else 0.0,
        "prediction_changed_count": int(np.sum(before_pred != after_pred)),
        "prediction_changed_fraction": float(np.mean(before_pred != after_pred)) if before_pred.size else 0.0,
        "identity_gate": bool(np.allclose(before_logits, after_logits) and np.array_equal(before_pred, after_pred)),
    }


__all__ = [
    "PointwiseEvent",
    "apply_keep_weight_to_event_rows",
    "event_rows",
    "score_eventwise",
    "summarize_identity_effect",
]
