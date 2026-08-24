# V25b: V25a exact replay parent-gradient decomposition

## Purpose

V25b is a development-only diagnosis of the completed V25a training run.  It
does not train a new KnownGate model, alter a V25a/V24 checkpoint, tune a
threshold, select a new epoch, or make a product-performance claim.  Its only
operation is to replay the frozen V25a training updates and decompose the
historical batch loss into anonymous Known and Clutter parent-gradient terms.

The target question is narrower than V22: when a batch contains a rare
Clutter parent under V25a's historical `batch_weight_sum` weighting, are
gradient spikes associated with (a) a larger weighted Clutter contribution,
(b) an oppositely directed raw Known/Clutter contribution, or (c) an observed
batch condition such as RD availability or same-second multiplicity?  The
report must describe associations only; it cannot establish a causal source or
a remedy.

## Frozen source and scope

The source is the completed V25a seed-43 V24-equivalent run:

- 20 epochs, 9,860 updates, batch size 2;
- the V25a self-developed `CfCLiquidCausalEncoder`, causal-course input,
  reason-aware bounded-observation dynamics, and all trainable parameters;
- AdamW defaults (`lr=1e-4`, `weight_decay=0.01`) and global gradient clip 100;
- `parent_final_task_ce_with_prefix_auxiliary`, parent-final main loss plus
  0.1 causal-prefix auxiliary loss beginning at event 3;
- historical `batch_weight_sum` class weighting, with Known weight
  0.5173137460650578 and Clutter weight 14.93939393939394;
- V25a's frozen epoch checkpoints, ordered-parent hashes, and pre-clip
  telemetry files.

Preparation freezes SHA-256 bindings for the V25a config, V25a contract and
readiness records, training summary, telemetry metadata, every one of the 20
epoch checkpoints, every one of the 20 telemetry files, and every epoch parent
order.  Before either smoke or full analysis constructs the train Dataset, V25b
recomputes and compares that complete binding.  A later self-consistent V25a
replacement is therefore rejected rather than silently analyzed as if it were
the originally prepared source.

The same preparation record also freezes the SHA-256 values of the V25b
decomposition primitive, builder, reporter, smoke, pipeline script, protocol,
and the V25a/V21 training, loss, KnownGate, data, feature, raw-feature helper,
RD physical validation, liquid-model, architecture, logging, and telemetry
modules that define its replay behavior.  `analyze` fails closed if any of
those implementation files changed after preparation; rerun `prepare` (which
preserves the prior record in an audit archive when it is not identical) before
asking it to emit a report.

V25b builds only the V25a train Dataset required to replay a batch.  It does
not construct or read a validation Dataset, test Dataset, Unknown population,
L3 scorer, full cascade, rejection evaluator, stable evaluator, or latency
evaluator.  Raw parent identifiers exist transiently only inside the source
DataLoader; persisted output stores the ordered batch SHA-256 and label/count
metadata, never parent, track, capture, sample, or source identifiers.

## Exact replay protocol

For epoch 1, V25b reconstructs the frozen fresh seed-43 V25a model and its
independent DataLoader generator.  For epochs 2--20, it loads the preceding
V25a epoch checkpoint and restores its Python, NumPy, CPU/CUDA, and
DataLoader-generator states.  Each frozen epoch is replayed in a fresh worker
Python/CUDA process; the parent coordinator remains CPU-only and accepts only
anonymous JSON after the worker verifies all 493 steps and its epoch boundary.
This bounds CUDA-driver lifetime without changing the replayed model, data
order, loss, optimizer, or diagnostics.  Every replayed update runs the
original forward, original loss, original backward call, original global clip,
and original AdamW step.

Every step must reproduce all of the following before its optimiser update:

- ordered parent-batch SHA-256 exactly;
- total, final, and prefix loss within registered CUDA replay tolerance;
- pre-clip global norm and clipping decision within registered CUDA replay
  tolerance;
- at every epoch boundary, the replayed model state relative to the frozen
  epoch checkpoint within registered CUDA replay tolerance.

For a Clutter-containing batch only, V25b keeps the original backward graph,
clones the actual `parameter.grad`, and uses `torch.autograd.grad` solely to
read diagnostic parent terms.  It then proves that the diagnostic calls did
not change `.grad` and that the sum of historical parent terms reconstructs
the original `loss.backward()` gradient.  A replay or reconstruction mismatch
fails closed and no completed diagnosis is written.

The V25a final-state loss is a fused CUDA weighted-mean cross-entropy while
the diagnostic exposes its per-parent unreduced terms.  The check is therefore
numeric rather than bitwise.  It separately verifies the original fused-loss
gradient against an explicit unreduced proxy, then the proxy against the sum
of parent terms.  Each bridge uses an absolute L2 floor, a relative L2 limit,
and a component limit proportional to the largest source-gradient coordinate
(with a small absolute floor).  This prevents a fixed coordinate threshold
from falsely failing ordinary rounding on a large-gradient batch; a larger
scale-normalized mismatch still fails closed.

## Loss decomposition

For parent \(i\), let \(y_i\) be the KnownGate label, \(w_i\) its historical
class weight, \(f_i\) its final-state cross entropy, and \(p_i\) its mean
eligible-prefix cross entropy.  Let \(E\) be the parents with at least one
eligible prefix and let \(\lambda=0.1\).  V25a's batch loss is

\[
L = \sum_i \frac{w_i}{\sum_j w_j}f_i
    + \lambda \sum_{i\in E}\frac{w_i}{\sum_{j\in E}w_j}p_i.
\]

V25b reports the unweighted raw parent gradient

\[
g_i^{\mathrm{raw}} = \nabla_\theta(f_i + \lambda p_i),
\]

and the exact historical contribution

\[
g_i^{\mathrm{hist}} =
 \frac{w_i}{\sum_jw_j}\nabla_\theta f_i
 + \lambda\mathbf{1}(i\in E)
 \frac{w_i}{\sum_{j\in E}w_j}\nabla_\theta p_i.
\]

For a mixed Known/Clutter batch, the report aggregates these into
\(g_K\) and \(g_C\), supplies their L2 norms, dot products and cosines, and
also retains separate final and prefix contributions.  For a
Clutter+Clutter batch, the Known aggregate is absent rather than represented
by a fabricated example.

## Required report and interpretation boundary

The completed report must include every V25a update's replay integrity summary;
every Clutter-containing batch's anonymous decomposition; clipped versus
non-clipped strata; Clutter+Drone/Bird/Balloon/Clutter strata; same-second,
RD-present, and course-valid observation strata; and TR/RD/fusion/liquid/
classifier energy breakdowns.  The report must state that the observed
associations do not prove that label weighting, a feature family, an RD
condition, a source, or a module causes a spike.

No V25b checkpoint, OOF calibration, test, Unknown, L3, full-cascade,
rejection, stable, P95, or deployment conclusion is permitted.
