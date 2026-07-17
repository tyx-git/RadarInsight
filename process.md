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
