"""V25e shared-module dominant-gradient conflict projection.

V25e is a training intervention, not a diagnostic replay.  It is deliberately
small: the historical V25a loss is evaluated unchanged, and only a mixed
KnownTarget/ClutterTarget batch can alter gradients in the three shared
representation modules identified by V25c.  RD encoder and classifier
gradients are left untouched.
"""

from __future__ import annotations





V25E_SHARED_MODULE_NAMES = ("tr_encoder", "fusion", "liquid")
V25E_PROJECTED_MODULE_NAMES = V25E_SHARED_MODULE_NAMES
V25E_UNCHANGED_MODULE_NAMES = ("rd_encoder", "classifier")




















__all__ = ["V25E_PROJECTED_MODULE_NAMES", "V25E_SHARED_MODULE_NAMES", "V25E_UNCHANGED_MODULE_NAMES"]