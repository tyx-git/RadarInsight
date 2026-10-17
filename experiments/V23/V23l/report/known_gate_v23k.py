"""Development-only report for V23k causal TR/RD observation intensity."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from radar_ln3.known_gate import summarize_known_gate_records
from report.known_gate_common import _metric_vector, _ordered_pair


MIN_STABLE_STRATUM_SUPPORT = 8








def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left, right = _metric_vector(before), _metric_vector(after)
    return {key: float(right[key] - left[key]) for key in left}




def _rows_for_ids(rows: Sequence[Mapping[str, Any]], ids: set[str]) -> list[dict[str, Any]]:
    result = [dict(row) for row in rows if str(row["parent_sample_id"]) in ids]
    if len(result) != len(ids):
        raise ValueError("V23k stratum parent records are incomplete")
    return result


def _stratum_metrics(
    control: Sequence[Mapping[str, Any]],
    treatment: Sequence[Mapping[str, Any]],
    groups: Mapping[str, set[str]],
    *,
    prefix: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    labels = ("DroneTarget", "BirdTarget", "BalloonTarget", "ClutterTarget")
    for name, ids in groups.items():
        if not ids:
            # The frozen validation partition has two empty cross-cells.  They
            # are part of the requested Cartesian table, but do not constitute
            # an estimate: avoid inventing zero-valued performance or trying to
            # bootstrap an absent population.
            result[name] = {
                "support": 0,
                "minimum_stable_support": MIN_STABLE_STRATUM_SUPPORT,
                "stable_estimate": False,
                "comparison_status": "not_applicable_empty_stratum",
                "base_label_support": {label: 0 for label in labels},
                "v23a_fixed_retention_control": None,
                "v23k_observation_intensity_treatment": None,
                "v23k_minus_v23a": None,
                "warning": "empty descriptive stratum; no estimate or comparison is available",
            }
            continue
        before, after = _ordered_pair(_rows_for_ids(control, ids), _rows_for_ids(treatment, ids), scope="parent")
        before_metrics = summarize_known_gate_records(before, scope=f"v23a_{prefix}_{name}")
        after_metrics = summarize_known_gate_records(after, scope=f"v23k_{prefix}_{name}")
        support_by_label = {label: sum(str(row["label_name"]) == label for row in before) for label in labels}
        result[name] = {
            "support": len(before),
            "minimum_stable_support": MIN_STABLE_STRATUM_SUPPORT,
            "stable_estimate": len(before) >= MIN_STABLE_STRATUM_SUPPORT,
            "comparison_status": "descriptive",
            "base_label_support": support_by_label,
            "v23a_fixed_retention_control": before_metrics,
            "v23k_observation_intensity_treatment": after_metrics,
            "v23k_minus_v23a": _delta(before_metrics, after_metrics),
            "warning": None if len(before) >= MIN_STABLE_STRATUM_SUPPORT else "low-support descriptive stratum; do not infer a stable effect",
        }
    return result









