# Session-Grouped Training

## Purpose

This training scaffold pushes optimization one step closer to session-level recognition without replacing the existing clip-level model.

## Mechanism

- optional session-grouped batches
- optional session-logit auxiliary loss on the mean logits within a batch

## Recommended Starting Values

Add the following fields to a training control JSON when you want to test this route:

```json
{
  "session_grouped_batches": true,
  "session_logit_loss_weight": 0.15
}
```

## Note

This is not yet a full bag-level session model. It is a low-risk scaffold that can be tested before introducing a dedicated session encoder or multi-instance attention head.
