# Expert1 V2 smoke training, joint-isolated final

Date: 2026-05-11

This record preserves the complete result after isolating `joint_logits` from final logits.

## Source artifacts

- Run output: `artifacts/20260511-133618/expert1_v2_smoke`
- Config: `tmp/expert1_v2_smoke_config.json`
- Best checkpoint selected by trainer: `fallback_candidate_model.pt`

## Speed / fusion controls

- epochs: 5
- max_train_batches_per_epoch: 32
- quick_val_max_batches: 12
- full_val_interval_epochs: 3
- low_altitude_deep_reroute_aux_weight: 0.0
- joint_logit_delta_scale: 0.0
- rd_specialist_residual_scale: 0.5

## Test metrics

| metric | value |
|---|---:|
| final accuracy | 0.6742 |
| final macro F1 | 0.5972 |
| trajectory macro F1 | 0.5972 |
| trajectory macro F1 on RD | 0.6409 |
| joint macro F1 on RD | 0.0159 |
| joint - trajectory macro F1 on RD | -0.6250 |
| RD macro F1 on RD | 0.1449 |
| RD harm rate on RD | 0.5781 |
| joint RD weight mean on RD | 0.2035 |
| with RD accuracy | 0.7342 |
| without RD accuracy | 0.4795 |
| source balanced macro F1 | 0.3781 |
| min source macro F1 | 0.0800 |

## Conclusion

Isolating `joint_logits` prevents joint from directly harming final logits. Final and trajectory metrics become identical, and RD weight no longer saturates at 0.45.

However, the joint head itself collapses in this short smoke run, and RD evidence remains weak. This run is useful as a safety check, not as a final model.

The next meaningful run should use the same safe final path but remove `max_train_batches_per_epoch` so trajectory/RD can train on the full dataset. If that remains too slow, optimize data loading or reduce RD auxiliary loss groups rather than letting immature joint logits back into final.
