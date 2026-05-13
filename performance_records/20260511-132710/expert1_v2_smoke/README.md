# Expert1 V2 smoke training

Date: 2026-05-11

This record preserves the complete result of the 5-epoch Expert1 V2 smoke run.

## Source artifacts

- Run output: `artifacts/20260511-132710/expert1_v2_smoke`
- Config: `tmp/expert1_v2_smoke_config.json`
- Best checkpoint selected by trainer: `fallback_candidate_model.pt`

## Speed controls

- epochs: 5
- max_train_batches_per_epoch: 32
- quick_val_max_batches: 12
- full_val_interval_epochs: 3
- full_val_on_last_epoch: true
- compute_eval_aux_loss: false
- enable_batch_parameter_rollback: false
- low_altitude_deep_reroute_aux_weight: 0.0

## Test metrics

| metric | value |
|---|---:|
| final accuracy | 0.6774 |
| final macro F1 | 0.6004 |
| trajectory macro F1 | 0.5914 |
| trajectory macro F1 on RD | 0.6172 |
| joint macro F1 on RD | 0.3932 |
| joint - trajectory macro F1 on RD | -0.2241 |
| RD macro F1 on RD | 0.1478 |
| RD harm rate on RD | 0.5907 |
| joint RD weight mean on RD | 0.4500 |
| with RD accuracy | 0.7089 |
| without RD accuracy | 0.5753 |
| source balanced macro F1 | 0.4006 |
| min source macro F1 | 0.2000 |

## Conclusion

The run completed quickly, but the model is not usable.

The core failure persists: the joint branch remains much weaker than trajectory on RD samples, while the reliability weight saturates at the 0.45 cap. RD evidence is also still too weak, especially for Drone/Bird where RD collapses heavily toward Bird.

The next optimization should prevent immature `joint_logits` from entering final logits directly. Joint should first be trained/reported as auxiliary evidence, while final stays trajectory-anchored with at most a small bounded residual.
