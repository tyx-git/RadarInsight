"""V27b-r6: frozen-endpoint L2 knownness representation experiment.

The r6 package is intentionally independent from the historical r1--r5
classification-residual path.  It leaves the frozen V27a Main DBB logits
unchanged and studies only a separate L2 knownness representation.
"""

from .protocol import R6_EXPERIMENT_ID, R6_SCHEMA_VERSION

__all__ = ["R6_EXPERIMENT_ID", "R6_SCHEMA_VERSION"]
