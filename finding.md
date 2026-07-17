# Findings

## F001 - Untrusted finite calibration is not policy authorization

Severity: High
Evidence: Old `_locked_precision_parameters` accepted any finite five-value bias; focused red tests reproduced `[9.0]*5`, nonzero Drone/Bird/Balloon, invalid finite residual values, non-finite values, bad lengths, mappings, and non-iterables.
Impact: Checkpoints, registry records, callers, or export payloads can persist calibration that validation never authorized.
Action: Resolved with `_authorized_precision_logit_bias` and `_sanitized_precision_lock`; all untrusted precision bias boundaries now fall back to a legal zero vector.

## F002 - Test metrics still own release decisions

Severity: High
Evidence: Old final flow derived release eligibility, bottleneck summary, and candidate status directly from `test_metrics` after the test loader pass.
Impact: Test observations can alter selection/release/readiness actions, contaminating the held-out diagnostic set.
Action: Resolved by building `validation_control_metrics` before the one test pass and publishing held-out observations under `test_reporting` with `report_scope="test"` and `test_used_for_selection=false`.

## F003 - Export metadata can retain stale calibration projections

Severity: High
Evidence: Calibration appears at top level, nested config, label maps, and `locked_precision_parameters`; rebuilding only some fields can leave contradictory payload state.
Impact: The same checkpoint can behave differently across load/export entry points.
Action: Resolved by accepting only validation locks from registry records and rebuilding top-level, nested config, label-map, lock, and provenance fields during export.

## F004 - Full smoke has unrelated P3 expectation drift

Severity: Low
Evidence: `python -m unittest tests.test_expert1_precision_controls_smoke` runs 35 tests; two existing P3 metric assertions fail: `test_precision_candidate_prefers_clutter_stability_when_macro_f1_is_close` and `test_rd_joint_bottleneck_summary_recommends_training_after_stage7_shift`.
Impact: The full mixed smoke module is not green, but the 16 P4 control-plane regressions and compile checks pass.
Action: Preserve these unrelated P3 assertions for the next P3 calibration/metric review; do not alter them in this P4 security fix.
