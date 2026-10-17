"""Configuration and immutable provenance contracts for V27c.

V27c is intentionally a narrow, known-only experiment.  It starts from the
frozen V27a Main classifier endpoint and tests one additional signal only:
the V9 ``causal-tr-motion-v1`` numeric values, after their validity mask has
been used for numerical gating.  The mask itself is never model input.
"""

from __future__ import annotations





V27C_EXPERIMENT_ID = "V27c"
V27C_SCHEMA_VERSION = "RADAR-LN3-V27c-motion-late-residual-v1"



































































__all__ = ["V27C_EXPERIMENT_ID", "V27C_SCHEMA_VERSION"]