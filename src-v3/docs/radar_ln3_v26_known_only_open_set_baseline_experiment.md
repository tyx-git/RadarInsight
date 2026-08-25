# RADAR-LN3 V26 Known-only baseline and strict open-set audit

## Experiment version

V26 baseline.

## Variable

The model remains a binary KnownGate:

- `KNOWN_TARGET_EVIDENCE`
- `KNOWN_CLUTTER_EVIDENCE`

`UnknownTarget` is not a model class. It is excluded from the training loss,
train normalization, sampler, optimizer population, and checkpoint ranking.
The V21e causal course feature is reused byte-for-byte as the single evidence
column. Training is a fresh seed-43, full 20-epoch run and does not load a
trained V21e/V25 checkpoint.

## Validation

Checkpoint selection uses only the 232 known validation parents. After the
selected checkpoint is frozen and hash-bound, the report reads separate mixed
validation and test audit caches:

- validation: 247 parents, including 15 `UnknownTarget` parents;
- test: 244 parents, including 16 `UnknownTarget` parents.

The audit reports known parent-final, eligible-prefix and window metrics, plus
Unknown routing to the two known logits and maximum-known-confidence
diagnostics. Because this baseline has no knownness/rejection head, an Unknown
route is not a rejection decision; `Unknown_rejected` and `Pending/Abstain`
remain unavailable and are never counted as successful Unknown rejection.

Clutter metrics always report their parent support (9 in validation and 8 in
test) and stratified parent bootstrap intervals. No five-class accuracy,
full-cascade, L2/L3, OOF threshold, stable, P95 or deployment claim is made.
