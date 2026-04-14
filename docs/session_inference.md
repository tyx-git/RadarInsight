# Session-Level Inference

## Purpose

This command evaluates a checkpoint at both clip level and session level.

Session-level prediction is computed by averaging clip logits that belong to the same session and then taking the argmax.

## Command

```powershell
.\.venv\Scripts\python.exe manage.py session-infer artifacts\<run_name>\best_fusion_model.pt --train-control train.json --device cuda
```

## Outputs

- `evaluation_report.json`: clip-level metrics plus embedded `session_level` metrics
- `session_inference_report.json`: combined clip/session inference summary intended for session review workflows

## Current Note

This is an evaluation-time aggregation feature only.

It does not yet train a dedicated session-level bag model, but it establishes the measurement path needed for future session-level training.
