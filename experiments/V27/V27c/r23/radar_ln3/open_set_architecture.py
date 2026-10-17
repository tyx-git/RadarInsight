"""Canonical strict-open-set action semantics for the current product.

Historical experiment modules are no longer product entry points.  The
current topology is L1 KnownGate, L2 DBB specialist, and L3 semantic decision;
Unknown is an action rather than a learned L1/L2 class.
"""

from __future__ import annotations

from enum import StrEnum


ARCHITECTURE_SCHEMA = "RADAR-LN3-V21-strict-open-set-architecture-v1"
KNOWN_GATE_TASK_NAME = "known_gate"
DBB_SPECIALIST_TASK_NAME = "dbb_specialist"




class ProductAction(StrEnum):
    DRONE = "DroneTarget"
    BIRD = "BirdTarget"
    BALLOON = "BalloonTarget"
    CLUTTER = "ClutterTarget"
    UNKNOWN_REJECTED = "UNKNOWN_REJECTED"
    PENDING_ABSTAIN = "PENDING_ABSTAIN"








__all__ = ["ARCHITECTURE_SCHEMA", "DBB_SPECIALIST_TASK_NAME", "KNOWN_GATE_TASK_NAME", "ProductAction"]