# Retrain Checklist

## Purpose

This checklist defines the recommended workflow after every new training run.

It is designed to keep the project aligned with the current best validated checkpoint and avoid spending time on runs that only improve a secondary metric while hurting the main objective.

## Current Reference

- Preferred reference checkpoint: `artifacts/202604101750-12-4-cuda`
- Reference clip-level metrics:
  - `Accuracy = 0.8515`
  - `Macro F1 = 0.8363`
- Reference session-level metrics:
  - `Accuracy = 0.8750`
  - `Macro F1 = 0.8810`

Any new run should be compared against this checkpoint first.

## Standard Workflow

### 1. Train

```powershell
.\.venv\Scripts\python.exe manage.py train --train-control train.json --device cuda
```

Record the produced artifact directory.

### 2. Clip-Level Evaluation

```powershell
.\.venv\Scripts\python.exe manage.py evaluate artifacts\<run_name>\best_fusion_model.pt --train-control train.json --device cuda
```

Check:

- `Accuracy`
- `Macro F1`
- confusion matrix in `evaluation_report.json`
- confirm that evaluation logs show the expected eval-time fusion overrides

### 3. Session-Level Evaluation

```powershell
.\.venv\Scripts\python.exe manage.py session-infer artifacts\<run_name>\best_fusion_model.pt --train-control train.json --device cuda
```

Check:

- `session_level.accuracy`
- `session_level.macro_f1`
- `session_level_peak` only as supporting evidence, not the main promotion criterion

### 4. Hard-Session Mining

```powershell
.\.venv\Scripts\python.exe manage.py mine-hard-sessions artifacts\<run_name>\best_fusion_model.pt --train-control train.json --device cuda --top-k 3
```

Inspect:

- `hard_session_mining_report.json`
- `all_sessions/session_summary.csv`
- `focused_sessions/session_clip_predictions.csv`

### 5. Focused Session Diagnosis

If the mined sessions look important, export diagnosis for those specific sessions:

```powershell
.\.venv\Scripts\python.exe manage.py session-diagnose artifacts\<run_name>\best_fusion_model.pt <session_id_1> <session_id_2> <session_id_3> --train-control train.json --device cuda
```

Use the CSV outputs to inspect:

- clip-level top-1 and top-2 probabilities
- track router outputs
- ship specialist outputs
- Bird-vs-Ship specialist outputs
- per-session vote counts and mean probabilities

## Promotion Rule

Promote a new run as the preferred baseline only if it improves the overall operating point, not just a single metric.

### Promote

Promote when all of the following are true:

- clip-level `Accuracy` is not worse in a meaningful way
- clip-level `Macro F1` is not worse in a meaningful way
- session-level metrics are not worse
- the main error boundaries do not regress sharply

### Do Not Promote

Do not promote when one of the following happens:

- `Macro F1` goes up but `Accuracy` drops noticeably
- `session_level` stays flat while clip-level degrades
- `ShipTarget -> BirdTarget` or `ShipTarget -> DroneTarget` grows sharply
- `BirdTarget -> ShipTarget` grows sharply

## Current Main Error Boundaries

The current hard boundaries remain:

- `ShipTarget -> DroneTarget`
- `BirdTarget -> ShipTarget`

These should be treated as the first comparison targets for every new run.

## Practical Decision Tree

### Case A: Better clip and better session metrics

- keep the new run
- update `mermory.md`
- use it as the new reference baseline

### Case B: Better clip metrics, same session metrics

- inspect hard-session mining output
- keep only if the error structure also improves

### Case C: Worse clip metrics, same session metrics

- do not promote
- keep only as an experiment artifact

### Case D: Worse clip and worse session metrics

- reject immediately

## Notes

- The project currently shows meaningful run-to-run variance even under seeded training.
- A single rerun is not enough evidence to replace the current best checkpoint.
- The hard-session mining outputs should be treated as the main debugging entry point before changing model structure or training policy.
- The current preferred route is stable training plus evaluation-time anti-collapse fusion overrides, not aggressive train-time anti-collapse.
