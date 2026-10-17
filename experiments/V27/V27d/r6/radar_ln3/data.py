from __future__ import annotations






DT_REASON_PADDING = 0
DT_REASON_FIRST_EVENT = 1
DT_REASON_ZERO_DELTA = 2
DT_REASON_POSITIVE_DELTA = 3








# Compatibility name retained for historical callers.  New code should use
# ``parse_bool`` directly so malformed configuration values fail closed.










# Canonical RD tensors are small enough for a single parent, but they are
# still untrusted compressed archives.  Bound both the archive and its
# declared uncompressed members before NumPy opens them; otherwise a malformed
# or zip-bomb-like .npz can allocate memory before the shape checks below run.










__all__ = ["DT_REASON_FIRST_EVENT", "DT_REASON_PADDING", "DT_REASON_POSITIVE_DELTA", "DT_REASON_ZERO_DELTA"]