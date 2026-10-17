"""Fail-closed V27 L3 action contract and registry validation.

L3 is deliberately a semantic boundary.  It consumes calibrated verdicts and
prediction sets produced by :mod:`v27.open_set`; it never turns a raw softmax,
distance, or uncalibrated score into a product action.
"""

from __future__ import annotations





L3_SCHEMA_VERSION = "RADAR-LN3-V27-l3-calibrated-action-v3"






























__all__ = ["L3_SCHEMA_VERSION"]