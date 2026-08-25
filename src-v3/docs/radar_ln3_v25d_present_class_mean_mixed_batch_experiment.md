# V25d: mixed-batch present-class-mean KnownGate loss

## Question

V25a/V25b/V25c established that the historical two-parent mixed
KnownTarget/Clutter batch has highly asymmetric effective loss mass under
`batch_weight_sum`.  V25d tests one narrow causal question:

> Does equalizing the total contribution of the classes actually present in a
> mixed batch improve the causal-prefix Clutter boundary without damaging the
> parent-final KnownTarget boundary?

It is a development-only L1 KnownGate experiment.  It does not train on
UnknownTarget, fit an open-set threshold, run OOF, test, L3, cascade,
rejection, stable, or latency evaluation.

## Frozen control and sole variable

The completed V25a seed-43 run is the direct control.  V25d reuses exactly its
V21e causal-course feature cache, V16/V21 index and normalization, model,
reason-aware `bounded_observation` dynamics, batch size two, AdamW defaults,
learning rate, global clip 100, prefix weight 0.1, seed, data-loader generator,
20 epochs, parent order, fresh initialization, and parent-final checkpoint
selection rule.

The sole changed field is:

```text
train.class_weight_normalization
V25a: batch_weight_sum
V25d: present_class_mean
```

No feature, sampler, model parameter, optimizer, scheduler, clipping threshold,
checkpoint criterion, data order, or training duration is changed.

## Reduction semantics

Let \(B\) be the contributing parents in a local-task final-state batch,
\(C_B\) its currently present classes, and \(\ell_i\) unweighted parent CE.
The historical V25a reduction is

\[
L_{\mathrm{V25a}}=
\frac{\sum_{i\in B}w_{y_i}\ell_i}
     {\sum_{i\in B}w_{y_i}}.
\]

For the observed one-Known/one-Clutter mixed batch, this gives the two classes
roughly 3.35% and 96.65% loss mass, respectively.  V25d instead uses

\[
L_{\mathrm{V25d}}=
\frac{1}{|C_B|}\sum_{c\in C_B}
\frac{1}{n_c}\sum_{i:y_i=c}\ell_i.
\]

Thus a two-class mixed batch is exactly 0.5/0.5.  This is not a tuned 0.5
hyperparameter: it follows from the mean over the two classes that are present.
For a pure-class batch, the result equals V25a's weighted-mean reduction because
the same class weight cancels from numerator and denominator.

The prefix branch first averages legal non-final prefixes inside a parent, then
applies the same present-class mean over the parents that have legal prefixes.
If only one class has legal prefixes in a batch, it remains historical-equivalent.

## Required mechanism evidence

The trainer records final and prefix effective class mass per epoch.  It fails
closed if a contributing mixed final or prefix batch is not exactly 0.5/0.5, if
a pure batch does not carry one class mass, if the 9,860 final updates are not
accounted for, or if any V25d parent order differs from its frozen V25a order.

Preparation also verifies deterministic pure-class equivalence on synthetic
logits and exact V25a/V25d fresh initial model states and initial logits.

## Pre-registered development decision

All of the following are required before considering multi-seed confirmation:

1. Parent-final Clutter recall remains 9/9, KnownTarget recall remains 223/223,
   and Clutter precision is at least 0.90.
2. Parent-balanced eligible-prefix Clutter precision and recall do not decrease,
   Clutter F1 improves by at least 2 percentage points, and KnownTarget prefix
   recall and macro-F1 do not decrease.
3. The report proves the V25d mass mechanism and exact paired initialization /
   data order.

The parent-final Clutter support is only nine, so the report includes a paired,
base-label-stratified parent bootstrap and retains all support counts.  Passing
this development gate is not a claim of cross-source generalization or product
readiness; multi-seed confirmation precedes any OOF decision.
