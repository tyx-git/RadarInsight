"""Materialized frozen V27a features used by V27b.

The cache is deliberately a V27b-owned artifact.  It contains only causal
outputs of the frozen V27a model and the small set of quality facts needed for
evaluation; it never contains Unknown or test rows.  Keeping this boundary
explicit makes it possible to train the new correction head without invoking
the historical V27a loader's old manifest paths.
"""

from __future__ import annotations

from typing import Any, Mapping









def cache_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a V27b payload using the stable V27a semantic cache hash scheme."""

    from v27.v27a.cache import cache_payload_sha256 as _semantic_hash

    return _semantic_hash(payload)




















__all__ = ["cache_payload_sha256"]