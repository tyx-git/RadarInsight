"""DBB/CU gate and specialist task boundaries.

The directory name intentionally follows the project naming request.  Since
the hyphen is not a Python identifier, normal runtime imports use the shared
``radar_ln3.task_models`` package and this file is a readable contract anchor.
"""

from __future__ import annotations

from radar_ln3.task_models import TaskSpec, compose_soft_cascade, task_spec


GATE_TASK: TaskSpec = task_spec("dbb_cu_gate")
DBB_TASK: TaskSpec = task_spec("dbb_specialist")
CU_TASK: TaskSpec = task_spec("cu_specialist")


def compose(gate_logits, dbb_logits, cu_logits):
    """Return five-class probabilities through the soft DBB/CU cascade."""

    return compose_soft_cascade(gate_logits, dbb_logits, cu_logits)


__all__ = ["CU_TASK", "DBB_TASK", "GATE_TASK", "compose"]
