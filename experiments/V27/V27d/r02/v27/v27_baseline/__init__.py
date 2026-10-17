"""Complete V27 L1-L2-L3 Liquid V2 baseline package."""

from .protocol import (
    V27_EXPERIMENT_ID,
    V27_L1_CLASS_NAMES,
    V27_L1_TASK_NAME,
    V27_L2_CLASS_NAMES,
    V27_L2_TASK_NAME,
)

V27_CLASS_NAMES = V27_L2_CLASS_NAMES
V27_TASK_NAME = "complete_l1_l2_l3_baseline"

__all__ = [
    "V27_CLASS_NAMES",
    "V27_EXPERIMENT_ID",
    "V27_L1_CLASS_NAMES",
    "V27_L1_TASK_NAME",
    "V27_L2_CLASS_NAMES",
    "V27_L2_TASK_NAME",
    "V27_TASK_NAME",
]
