"""Immutable configuration and provenance protocol for the complete V27 baseline.

The V27 system has two independently parameterised learned layers (L1 and
L2), plus an L3 action resolver.  This module owns the one configuration
parser and the one data/implementation registry.  It deliberately does not
open test or Unknown payloads while preparing classifier training.
"""

from __future__ import annotations





V27_EXPERIMENT_ID = "V27"
V27_L1_TASK_NAME = "known_gate"
V27_L2_TASK_NAME = "dbb_specialist"
V27_L1_CLASS_NAMES = ("KNOWN_TARGET_EVIDENCE", "KNOWN_CLUTTER_EVIDENCE")
V27_L2_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")






































































__all__ = ["V27_EXPERIMENT_ID", "V27_L1_CLASS_NAMES", "V27_L1_TASK_NAME", "V27_L2_CLASS_NAMES", "V27_L2_TASK_NAME"]