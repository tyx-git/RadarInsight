"""Clutter/Unknown specialist task boundary.

The implementation lives in ``radar_ln3.task_models`` so training, reporting
and Smoke use one verified contract.  This module keeps the CU experiment's
label boundary explicit inside the requested test directory.
"""

from __future__ import annotations

from radar_ln3.task_models import TaskSpec, task_spec


CU_TASK: TaskSpec = task_spec("cu_specialist")


def encode_label(label_name: str) -> int:
    """Map only the original Clutter/Unknown label to the CU task id."""

    return CU_TASK.encode(label_name)


__all__ = ["CU_TASK", "encode_label"]
