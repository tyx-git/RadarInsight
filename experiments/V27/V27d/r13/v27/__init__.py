"""Container package for independently reproducible V27 experiments.

The experiment implementations live in sibling packages such as
``v27.v27_baseline`` and ``v27.v27a``.  They intentionally own separate
configuration, artifact, and provenance contracts.
"""

__all__ = ["v27_baseline", "v27a", "v27b"]
