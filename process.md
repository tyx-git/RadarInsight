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
