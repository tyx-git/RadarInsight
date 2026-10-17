"""Public namespace for the current strict-open-set Radar-LN3 product."""

from .open_set_architecture import (
    ARCHITECTURE_SCHEMA,
    DBB_SPECIALIST_TASK_NAME,
    KNOWN_GATE_TASK_NAME,
    ProductAction,
)


SCHEMA_VERSION = ARCHITECTURE_SCHEMA
CURRENT_TASKS = (KNOWN_GATE_TASK_NAME, DBB_SPECIALIST_TASK_NAME)

__all__ = [
    "ARCHITECTURE_SCHEMA",
    "CURRENT_TASKS",
    "DBB_SPECIALIST_TASK_NAME",
    "KNOWN_GATE_TASK_NAME",
    "ProductAction",
    "SCHEMA_VERSION",
]
