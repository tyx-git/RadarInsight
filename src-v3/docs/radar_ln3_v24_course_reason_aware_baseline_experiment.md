# RADAR-LN3 V24 course-reason-aware baseline

V24 establishes a new, independently trained development control for the
V24 RD counter-evidence series.  It is not a continuation of any historical
checkpoint.

The model is the self-developed `CfCLiquidCausalEncoder` KnownGate used by
V21e: 48 hidden units, recurrent LayerNorm, learnable per-dimension time
constants, `masked_mean` RD aggregation, and reason-aware
`bounded_observation` dynamics with a 0.25 same-second observation gain.  The
only auxiliary evidence is the previously adopted causal-prefix course circular
incoherence feature.

The V16 capture/content grouped index, V21 train-only normalization, and the
already frozen V21e course feature caches are reused byte-for-byte.  The model
is initialized afresh from the V21 seed43 baseline configuration, then shares
every base parameter with that fresh baseline and initializes the one course
fusion column to exactly zero.  No V21e trained checkpoint and no V21 baseline
trained checkpoint is loaded into V24.

V24 runs exactly 20 full epochs with seed 43, batch size 2, the existing
`parent_final_task_ce_with_prefix_auxiliary` loss (prefix weight 0.1), the
historical batch-weight-sum class weighting, and the existing
parent-min-precision/recall checkpoint selector.  It is a development-only
KnownGate baseline: OOF, test, Unknown training or threshold fitting, L3,
full cascade, formal rejection, stable evaluation, and P95 remain
`not_executed`.

The report must show parent-final, eligible-prefix, parent-balanced
eligible-prefix, and window metrics; Clutter support and Wilson intervals;
parent-stratified bootstrap intervals; and every Known-to-Clutter and
Clutter-to-Known parent.  The development split has only nine Clutter parents,
so no single point estimate is treated as a release claim.
