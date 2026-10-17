"""Configuration, artifact, and provenance contracts for V27a.

V27a is intentionally a separate package.  It consumes the immutable V27 L2
classification base and never writes to the V27 manifest or L3 registry.
"""

from __future__ import annotations





V27A_EXPERIMENT_ID = "V27a"
V27A_SCHEMA_VERSION = "RADAR-LN3-V27a-micro-doppler-evidence-decoupling-v1"

V27A_L2_CLASS_NAMES = ("DroneTarget", "BirdTarget", "BalloonTarget")




























































__all__ = ["V27A_L2_CLASS_NAMES", "V27A_EXPERIMENT_ID", "V27A_SCHEMA_VERSION"]