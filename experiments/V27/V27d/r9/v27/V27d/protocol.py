"""V27d configuration, path and provenance contracts.

V27d is an independent experiment package.  It consumes immutable artifacts
from the earlier V27 stages, but writes every derived artifact below this
directory.  The package deliberately does not load the active V27 baseline
manifest because that manifest is a historical classifier snapshot and may
not match the current working tree.
"""

from __future__ import annotations




V27D_EXPERIMENT_ID = "V27d"
V27D_SCHEMA_VERSION = "RADAR-LN3-V27d-frozen-endpoint-knownness-v1"






















































__all__ = ["V27D_EXPERIMENT_ID", "V27D_SCHEMA_VERSION"]