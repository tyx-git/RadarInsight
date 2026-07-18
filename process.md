# Process

## Stage P4 - Expert1 Calibration and Evaluation Control Plane

Status: 100%

- 2026-07-17: Read `bug.md`, `CLAUDE.md`, the deep-learning operating protocol, MOE calibration guidance, and evaluation guidance.
- 2026-07-17: Confirmed the existing unrelated workspace change is the prepared test import; `bug.md` is untracked and `.history/` remains excluded.
- 2026-07-17: Confirmed full training is prohibited for this task; validation is limited to focused pytest, compile/import, and static checks.
- 2026-07-17: Added P4 red tests; old implementation accepted unauthorized finite bias and raised on malformed checkpoint values.
- 2026-07-17: Added strict authorization and sanitized lock reconstruction across trainer, reselector, export, and system loader.
- 2026-07-17: Moved release gate, candidate status, readiness, bottleneck, deployment action, and export eligibility to a validation control snapshot before the sole test pass.
- 2026-07-17: Passed 16 focused P4 regression tests and `py_compile`; the full 35-test module retains two unrelated P3 expectation failures.

## Stage P0 - Contracts, RD Availability, and Evaluation Boundaries

Status: 100%

- 2026-07-18: Audited label/tensor, RD availability/window alignment, and evaluation control-plane contracts without full training or `.history/` access.
- 2026-07-18: Fixed `RDShardLocalBatchSampler` so undersized RD shards are replenished only from the active shard, preserving shard-local cache behavior.
- 2026-07-18: Passed 19 RD/throughput tests, 14 split-integrity tests, 6 evaluation-control tests, targeted `py_compile`, and trainer/reselector imports.
- 2026-07-18: User confirmed the deleted tracking files and `.anthor/` directory are intentional unrelated workspace changes; they remain unstaged.

## Stage P1 - Parameter and Configuration Call Chains

Status: 100%

- 2026-07-18: Compared `config/config.json` sections with their dataclass fields and focused on Expert1 configuration loading and persistence boundaries.
- 2026-07-18: Added a red regression showing project defaults overwrote explicit Expert1 section values; changed the loader to apply project values only when the section omits them.
- 2026-07-18: Passed the two new configuration tests and compiled every `src` module while excluding `.history/`.
- 2026-07-18: The broader parameter/calibration selection stopped after 29 passing tests at the known P3 precision-candidate metric assertion; no P1 regression was observed.

## Stage P2 - Branch Semantics and Design Intent

Status: 100%

- 2026-07-18: Audited Expert1 trajectory-only, explicit-availability TR+RD, fallback, calibration, routing, and system branch boundaries.
- 2026-07-18: Added a red regression showing a zero four-dimensional placeholder entered fusion when `rd_is_multimodal` was omitted.
- 2026-07-18: Made availability-mask omission a conservative trajectory-only path; explicit availability remains mandatory for fusion.
- 2026-07-18: Passed 32 RD/routing/system focused tests, 3 control-plane tests, focused compile, and imports without full training.

## Stage P3 - Unused, Redundant, and Unreachable Implementations

Status: 100%

- 2026-07-18: Used a standard-library AST audit over `src/**/*.py`, explicitly excluding `.history/`; all 91 import candidates were public re-exports or required runtime evidence, so none were deleted.
- 2026-07-18: Restored the precision-candidate gap expectation after the historical `-0.01` to `-0.05` tolerance change.
- 2026-07-18: Separated negative proportional-fallback diagnostics from the relaxed candidate tolerance so any negative gain triggers the Stage 7 retraining recommendation.
- 2026-07-18: Passed all 35 precision-control tests with 20 subtests, compiled `src` and `tests` outside `.history/`, and imported trainer/reselector modules.

## Stage P4 - Cross-Module Architecture, Serialization, and Performance

Status: 100%

- 2026-07-18: Revalidated the P4 precision-control implementation from `4c2e996`, including validation-only calibration locks, checkpoint/registry/export reconstruction, malformed payload handling, and test-report isolation.
- 2026-07-18: Passed all 35 precision-control tests with 20 subtests and all 9 Stage 2 dataloader tests; targeted compile and imports also passed.
- 2026-07-18: Performed static sampler/cache/worker/H2D/quick-validation review; no new evidence justified a benchmark, so no smoke latency was presented as deployment performance.

## Stage P5 - Model Asset Audit and Strict Retention

Status: 100%

- 2026-07-18: Generated `tmp/model_asset_audit.json` and `tmp/model_retention_plan.json` for 1,725 assets in `artifacts/` and `reports/`, explicitly excluding `.history/`, runtime caches, and user-owned `.anthor/`.
- 2026-07-18: Calculated SHA256 for every audited asset; 906 assets have conservative text-reference protection and 819 have incomplete protocol metadata, so all 1,725 are retained.
- 2026-07-18: Formed 322 conservative comparable groups; no asset satisfied strict Pareto dominance plus reference-clearance criteria, so no deletion plan or deletion log was generated.
- 2026-07-18: Passed 14 checkpoint/registry/export tests with 4 subtests, reselector compilation, and audit schema/count/zero-deletion validation.

## Final Acceptance

Status: 100%

- 2026-07-18: Passed the complete focused suite for this effort: 83 tests and 20 subtests across configuration, split integrity, RD assembly/sampling, routing, and precision controls.
- 2026-07-18: Compiled all `src` and `tests` Python modules while excluding `.history/`, imported all changed control-plane modules, and passed `git diff --check`.
- 2026-07-18: Verified `test_used_for_selection=False` remains explicit in the test reporting path and retained all unclassified or referenced model assets.
- 2026-07-18: Full training, independent session-level evaluation, and real-hardware benchmarking remain required before making model-performance or deployment claims.
