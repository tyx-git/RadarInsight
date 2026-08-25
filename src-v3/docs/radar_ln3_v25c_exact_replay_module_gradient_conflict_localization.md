# V25c: V25a exact replay module-gradient conflict localization

## Purpose

V25c is a development-only mechanism diagnosis of the completed V25a
KnownGate training run.  It does not train a new model, change a loss weight,
alter a checkpoint, fit a threshold, select an epoch, or make a
product-performance claim.

V25b established that 611 of 638 mixed Known/Clutter batches have a negative
raw full-model cosine.  That fact is insufficient to justify an optimization
intervention: binary-classifier gradients can be expected to compete in the
classifier head.  V25c narrows the question to the shared representation:

> Is the negative Known/Clutter alignment concentrated in `classifier`, or is
> it also reproducibly present in shared `tr_encoder`, `rd_encoder`, `fusion`,
> and `liquid` parameters, with meaningful cancellation and association with
> clipping, Known subclass, RD coverage, same-second observations, or causal
> course coverage?

The answer locates a mechanism.  It does not prove that a particular loss,
sampler, feature, capture condition, or module caused a validation error.

## Frozen source and scope

The only source is completed V25a seed 43, a V24-equivalent self-developed
`CfCLiquidCausalEncoder` run with causal-course evidence:

- 20 epochs and 9,860 updates, batch size 2;
- historical `batch_weight_sum` Known/Clutter class weighting;
- parent-final loss plus the existing 0.1 eligible causal-prefix auxiliary
  loss from event 3;
- AdamW defaults (`lr=1e-4`, `weight_decay=0.01`) and global clip 100;
- all 20 frozen epoch checkpoints, telemetry files, and ordered-parent hashes.

Preparation freezes the source hashes and the hashes of all V25c and
transitive replay files.  Before any replay constructs a train Dataset, it
rechecks the whole source binding.  V25c builds only that train Dataset.  It
does not construct validation/test/Unknown data, L3 scoring, cascade,
rejection, stable, or latency evaluation.  Persisted data contains only an
ordered parent-batch SHA-256 and aggregate label/observation statistics;
parent, track, capture, sample, and source identifiers are forbidden.

## Exact replay and component protocol

Every epoch runs in a fresh CUDA worker.  The worker reconstructs the exact
V25a initialization or epoch-boundary checkpoint, restores the loader
generator state, then reproduces every forward, loss, backward, clip, and
AdamW update.  Each step must reproduce the ordered batch hash, total/final/
prefix losses, pre-clip norm, and clipping decision.  Each epoch must end at
the frozen checkpoint state within registered CUDA tolerance.

## Resource and transport safety

V25c has a deliberately bounded worker protocol because the diagnostic retains
638 rich anonymous mixed-batch records. A worker writes its one-epoch result
to an atomic JSON sidecar directly under project `tmp`; it never returns that
payload through stdout. The coordinator accepts at most 16 MiB per sidecar,
keeps only the 638 required mixed records plus four short numeric replay-error
arrays, and deletes every sidecar immediately after validation. Worker stderr
is retained only as an 8 KiB tail if a worker fails.

The final report is streamed atomically and hash-compared in blocks. The Python
entry point writes a compact completion record and emits only that record to
stdout; the PowerShell pipeline validates the report hash and size from the
completion record rather than loading the multi-megabyte report with
`ConvertFrom-Json`. Checkpoint restoration deliberately preserves V25b's CUDA
map-location path: replacing it with a CPU-only restoration changed the
floating-point source bridge on the same frozen batch. Each fresh worker exits
after one epoch and releases its CUDA context, while the post-replay reference
checkpoint is compared from CPU memory. These implementation choices preserve
the V25a/V25b replay math, source data, and metric.

For each of the 638 frozen mixed batches containing exactly one Known parent
and one Clutter parent, V25c retains the original backward graph.  It reads,
without modifying `parameter.grad`, these six per-parent components:

\[
g^{raw}_{total},\quad g^{hist}_{total},\quad
g^{raw}_{final},\quad g^{hist}_{final},\quad
g^{raw}_{prefix},\quad g^{hist}_{prefix}.
\]

Here `raw_prefix` includes the frozen auxiliary coefficient \(\lambda=0.1\).
For both parents V25c verifies

\[
g_{total}=g_{final}+g_{prefix}.
\]

It also verifies that the two historical parent totals reconstruct the original
`loss.backward()` gradient through the same bounded CUDA fused-loss/proxy
bridge used by V25b.  Any mismatch fails closed; no completed report is
written.

To make that source bridge genuinely comparable with V25b, V25c first queries
`raw_total`, `historical_total`, `historical_final`, and `historical_prefix`
for parent indices in their frozen batch order. Only after all source bridge
checks pass does it query V25c's extra raw final/prefix components. Thus the
extra module-conflict diagnosis cannot alter the CUDA autograd traversal order
of the inherited source bridge.

The parent total versus independently differentiated final-plus-prefix bridge
has its own registered tolerance.  It is not bitwise: CUDA may accumulate a
single parent total and two separately queried component gradients in a
different order.  Its bound is reported independently from the original-loss
reconstruction bound, so ordinary reduction noise cannot be mislabeled as
representation conflict.

The maximum-component relative bridge is (1\times10^{-4}). It was calibrated
on fresh replays of the fixed epoch-3, step-467 mixed Bird/Clutter source batch:
raw Clutter measured (6.18\times10^{-5}) to (6.52\times10^{-5}) and historical
Clutter reached (7.66\times10^{-5}), while their independent L2 bridges passed.
The previous (5\times10^{-5}) bound was therefore below repeatable CUDA
accumulation noise. The registered bound leaves approximately 31% headroom over
the largest fixed-source anchor and still limits a coordinate discrepancy to
0.01% of its parent-gradient peak. The L2 tolerance, absolute component floor,
source-loss and source-gradient reconstruction, hash, clipping, and
epoch-boundary checks are unchanged. The inherited V25b source bridge is also
queried first in its original CUDA call order and remains separately enforced;
this wider component-only bridge cannot mask a source replay mismatch.

## Module conflict metrics

Each component is reported for the full model and exactly these disjoint
parameter partitions:

```text
tr_encoder | rd_encoder | fusion | liquid | classifier
```

For Known gradient \(g_K\) and Clutter gradient \(g_C\), V25c reports the
two norms, dot product, cosine, anti-aligned projection, and norm loss after
summing.  The cancellation fraction is

\[
\frac{\|g_K\|+\|g_C\|-\|g_K+g_C\|}
     {\|g_K\|+\|g_C\|}.
\]

The primary view is `historical_total`, because it is the actual V25a update
weighting.  Raw, final, and prefix views distinguish class-weight amplification
from the underlying directional relation.

The report stratifies the primary view by Known subclass, clipped/not-clipped,
and separately by the Known-side and Clutter-side RD completeness, same-second
presence, and non-initial course coverage.  It therefore does not collapse a
target-side observation condition and a background-side condition into one
ambiguous batch label.

Course coverage is intentionally defined over non-initial active events.  The
first event has no predecessor by contract and its invalid course feature is
expected; treating any first-event invalidity as a missing-course stratum would
make the old all-events-valid indicator permanently false.

## Decision boundary for a later training intervention

- If negative alignment is primarily classifier-local, or shared-module
  cancellation is small and unassociated with clipped batches / at least two
  Known subclasses, treat it as ordinary binary discrimination.  Do not
  introduce loss surgery or a hard-coded mixed-batch weight.
- Only if shared TR/RD/fusion/liquid conflict is material, repeats across at
  least two Known subclasses, and is enriched among clipped or coherent
  observation strata should a subsequent V25d training intervention be
  designed.
- Even then, V25c does not choose a remedy.  A 0.5/0.5 mixed-batch loss would
  simultaneously alter conflict proportions and total Clutter exposure, so it
  is not a clean next experiment.

No V25c result is OOF, test, open-set, L3, full-cascade, rejection, stable,
P95, or deployment evidence.
