# RADAR-LN3 V24a current-event RD peak-drift counter-evidence

V24a is a development-only, single-factor KnownGate experiment.  Its sole
change from the completed V24 control is one appended evidence input:
`current_event_adjacent_rd_peak_velocity_drift_mean_fraction`.

For a current event with ordered canonical RD amplitudes `A[f,r,v]`, each
frame is converted to relative linear power after subtracting that frame's
maximum dB value, then reduced over range to a velocity profile.  Let `p_f` be
the velocity bin with maximum profile power and `v(p_f)` its physical velocity.
The feature is

`mean_f(abs(v(p_{f+1}) - v(p_f)) / (v_max - v_min))`.

Only adjacent RD frames belonging to the *same current event* are allowed.
V24a does not prepend a preceding event's final frame, does not read future
events, and does not use raw echo or MAT payloads.  An event with fewer than
two current RD frames stores `0.0` with `valid=false`.  This is an invalid
sentinel, not a fabricated valid zero; a physically zero drift with at least
two frames is stored with `valid=true`.

The original V24 causal-prefix course circular incoherence column is copied
byte-for-byte.  V24a reconstitutes the V24 fresh initialization through its
configuration, copies every shared parameter/buffer, and initializes only the
one newly appended drift fusion column to zero.  It must not load the trained
V24 checkpoint.

Frozen items are the data split and train-only normalization, model topology,
CfC dynamics, loss, batch size two, seed/order 43, learning rate, checkpoint
selector, and all 20 epochs.  Unknown and test remain excluded.  The result is
not a full cascade, open-set, L3, stable, OOF, or deployment result.

The paired report must include parent-final, eligible-prefix,
parent-balanced eligible-prefix, and window metrics; parent-stratified paired
bootstrap; Clutter Wilson intervals with parent support nine; high/low/invalid
course strata; RD-drift validity strata; capture strata; and every changed
parent.  The pre-registered directional audit is high-course-incoherence
Bird-to-Clutter and total Known-to-Clutter reduction, subject to no worsening
of Clutter-to-Known, KnownTarget recall, or prefix safety.
