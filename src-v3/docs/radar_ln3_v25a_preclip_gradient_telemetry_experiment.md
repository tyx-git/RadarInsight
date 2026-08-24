# V25a: V24 pre-clip gradient telemetry audit

## Purpose

V25a is an observation-only replication of the sealed V24 KnownGate baseline.
It tests whether the existing global gradient clipping is concentrated in
specific training phases, parent-label batch compositions, observation-quality
strata, or model modules.  It is not a model-improvement experiment and does
not introduce a new input feature, loss term, optimiser setting, decision
threshold, or deployment claim.

## Frozen control

The source control is V24, `v24_course_reason_aware_baseline`, seed 43.  V25a
reconstructs the same 48-wide self-developed `CfCLiquidCausalEncoder`, the
same causal-course evidence column, the same V21e train/validation caches, the
same `batch_size=2`, 20 epochs, AdamW defaults (`lr=1e-4`, `weight_decay=0.01`),
historical `batch_weight_sum` loss semantics, parent-final plus 0.1 causal-prefix
auxiliary loss, bounded-observation reason-aware dynamics, global clip norm 100,
and parent-final checkpoint selection.  It binds every V24 epoch parent-order
SHA-256 and reconstructs the V24 initial model-state SHA-256.

The V24 paired reference is the separate V21 seed-43 no-evidence model.  Its
state SHA-256 is audited independently; it must not be compared as identical
to the V24/V25a one-evidence-column model.

## Sole variable

After `loss.backward()` and before the existing
`torch.nn.utils.clip_grad_norm_(..., 100)`, V25a reads gradients and records:

- a SHA-256 of the ordered parent batch, never raw parent identifiers;
- KnownGate and base-label batch counts;
- active, same-second, positive-dt, RD-present, and valid-course observation
  counts;
- total, final-state, and prefix losses;
- pre-clip L2 norm and squared-gradient fraction for `tr_encoder`,
  `rd_encoder`, `fusion`, `liquid`, and `classifier`;
- the existing clipping call's returned pre-clip global norm, clipping flag,
  and resulting scale.

The recorder neither writes to gradients nor changes model inputs, optimizer
state, random state, data order, checkpoint ranking, or validation evaluation.

## Artifact and recovery semantics

Each completed epoch is stored as an immutable JSON file in the V25a artifact
under `gradient-telemetry/`.  A file is staged before the epoch checkpoint and
promoted after the checkpoint.  On epoch-boundary resume, the only recoverable
interruption is a checkpoint-bound final pending file; it is verified and
promoted.  Pending data from an uncheckpointed next epoch is archived as
orphaned and regenerated from the saved epoch boundary.  If interruption
occurs after `last_model.pt` is durable but before the latest epoch checkpoint
or newly selected best checkpoint is durable, resume verifies the boundary and
recreates only those latest files using the loaded epoch state; a missing older
epoch checkpoint fails closed because it cannot be reconstructed faithfully.

## Required interpretation

The report must include all 9,860 updates; whole-run and per-epoch pre-clip
norm distributions; clip rate and scale; early (1--3), selected, and late
(18--20) comparisons; label-signature and contains-Clutter strata; module norm
and squared-fraction comparisons for clipped and non-clipped steps; and
RD/course/same-second observation strata.  These are descriptive associations,
not evidence that a label, source, or module causes a gradient spike.

No OOF, test, Unknown, L3, full cascade, rejection, stable, latency, or
deployment evaluation is permitted in V25a.
