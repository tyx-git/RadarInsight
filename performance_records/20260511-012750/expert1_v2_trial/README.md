# Expert1 V2 trial checkpoint evaluation

Date: 2026-05-11

This record preserves the measurable result from the first Expert1 V2 training attempt.

## Source artifacts

- Training attempt output: `artifacts/20260511-012750/expert1_v2_trial`
- Evaluated checkpoint: `artifacts/20260511-012750/expert1_v2_trial/fallback_candidate_model.pt`
- Trial config: `tmp/expert1_v2_trial_config.json`
- Evaluation script: `tmp/evaluate_expert1_checkpoint.py`

The full 2-epoch training command timed out after 1 hour before writing `training_summary.json`.
The checkpoint had already been saved as a fallback candidate, so it was evaluated separately on val/test.

## Checkpoint metadata

- checkpoint selection reason: `stage2_rd_composite`
- checkpoint selection eligible: `false`
- configured epochs: 2
- stage1_traj_only_epochs: 0
- stage2_ramp_epochs: 1

## Test metrics

| metric | value |
|---|---:|
| final accuracy | 0.7581 |
| final macro F1 | 0.6392 |
| trajectory macro F1 | 0.6428 |
| trajectory macro F1 on RD | 0.6659 |
| joint macro F1 on RD | 0.4255 |
| joint - trajectory macro F1 on RD | -0.2403 |
| RD macro F1 on RD | 0.0765 |
| RD harm rate on RD | 0.5865 |
| final net gain rate on RD | 0.0042 |
| joint RD weight mean on RD | 0.4500 |
| with RD accuracy | 0.7975 |
| without RD accuracy | 0.6301 |

## Val metrics

| metric | value |
|---|---:|
| final accuracy | 0.7614 |
| final macro F1 | 0.5975 |
| trajectory macro F1 | 0.6138 |
| trajectory macro F1 on RD | 0.6254 |
| joint macro F1 on RD | 0.4978 |
| joint - trajectory macro F1 on RD | -0.1277 |
| RD macro F1 on RD | 0.0740 |
| RD harm rate on RD | 0.5798 |
| final net gain rate on RD | -0.0042 |
| joint RD weight mean on RD | 0.4500 |
| with RD accuracy | 0.7815 |
| without RD accuracy | 0.6912 |

## Conclusion

This V2 attempt is not usable as a production Expert1 checkpoint.

The joint branch is undertrained and substantially weaker than the trajectory anchor on RD samples.
The reliability head saturates at the configured RD cap of 0.45, so it is not yet learning when to trust RD.
The RD auxiliary branch is also severely degraded in this attempt.

Next training should not start with Stage2 fully active from epoch 1.
The joint branch needs a safer curriculum: preserve trajectory anchor first, train RD evidence without final takeover, then ramp joint RD weight from 0 with a lower initial cap.
