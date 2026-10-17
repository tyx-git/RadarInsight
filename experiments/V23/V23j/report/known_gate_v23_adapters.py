"""Paired development report for one completed V23 external-adapter experiment."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from radar_ln3.known_gate import summarize_known_gate_records
from report.known_gate_common import _bootstrap, _metric_vector, _ordered_pair












def _scope_metrics(records: Sequence[Mapping[str, Any]], scope: str) -> dict[str, Any]:
    return summarize_known_gate_records(records, scope=f"{scope}_recomputed")


def _delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, float]:
    left, right = _metric_vector(before), _metric_vector(after)
    return {name: float(right[name] - left[name]) for name in left}

























