# Hard Session Mining

## Purpose

This command automatically ranks the hardest evaluated sessions and exports focused diagnosis files.

## Command

```powershell
.\.venv\Scripts\python.exe manage.py mine-hard-sessions artifacts\<run_name>\best_fusion_model.pt --train-control train.json --device cuda --top-k 3
```

## Outputs

- `hard_session_mining_report.json`
- `all_sessions/session_summary.csv`
- `all_sessions/session_clip_predictions.csv`
- `focused_sessions/session_summary.csv`
- `focused_sessions/session_clip_predictions.csv`

## Selection Rule

Sessions are ranked primarily by `num_error_clips` descending and then by `clip_accuracy` ascending.

## Current Use

This tool is intended to isolate the few dominant failure sessions that limit overall clip accuracy and block progress toward higher accuracy targets.
