"""Causal, label-free feature contracts for the V21 KnownGate experiments."""

from __future__ import annotations





V21E_COURSE_PREFIX_VARIANT = "v21e_course_prefix"
# V24 deliberately keeps the only adopted V21e auxiliary input while creating
# a fresh, independently trained control for the next RD counter-evidence
# series.  It must remain a distinct variant so its checkpoints and reports
# cannot be mistaken for the historical V21e run.
# V24a appends one independent, current-event-only RD counter-evidence
# feature to the V24 course baseline.  It intentionally does not reuse V21a's
# optional preceding-event frame bridge: a one-frame current event is invalid.
# V24c deliberately reuses V24's only adopted causal-course input.  Its sole
# variable is the reason-aware liquid update at same-second observations.
# V25a is a fresh V24 replication whose only added behavior is read-only
# pre-clip gradient telemetry in the trainer.  It intentionally keeps the
# one adopted causal-course column and does not introduce a model feature.
# V25d is a loss-only follow-up to the completed V25a audit.  It reuses the
# exact selected V21e causal-course cache; a distinct variant keeps its cache
# identity explicit rather than relying on a fallback lookup.
# V25e keeps the exact V25a causal-course cache.  Its sole variable is a
# mixed-batch shared-gradient projection, so no new feature column or cache
# schema is allowed.
# V26 keeps the frozen V21e causal-course input for a fresh Known-only
# baseline.  The variant is intentionally distinct because its mixed
# validation/test open-set audit is a separate, post-checkpoint artifact.
# V26j is the event-internal RD temporal aggregation ablation.  It consumes
# exactly the same selected causal-course cache as V26/V26e; its sole model
# change is represented separately so a causal-GRU checkpoint can never be
# mistaken for the masked-mean baseline.

# These variants store an already-selected feature matrix rather than the
# original canonical KnownGate feature matrix.  They must therefore use
# cache-local column positions (0..width-1), not canonical feature indices.
# Keep this set shared by index construction and cache loading: a mismatch
# silently turns a reuse-only variant into a KeyError at Dataset construction.
























__all__ = ["V21E_COURSE_PREFIX_VARIANT"]