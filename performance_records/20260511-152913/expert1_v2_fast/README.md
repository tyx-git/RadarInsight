# Expert1 V2 fast full training

Date: 2026-05-11

This record preserves the complete result of the user-run fast Expert1 V2 training.

## Source artifacts

- Run output: `artifacts/20260511-152913/expert1_v2_fast`
- Markdown report: `reports/real/20260511-152913/expert1_v2_fast.md`
- JSON report: `reports/real/20260511-152913/expert1_v2_fast.json`
- Best checkpoint selected by trainer: `fallback_candidate_model.pt`

## Speed / fusion controls

- epochs: 5
- max_train_batches_per_epoch: 0
- quick_val_max_batches: 12
- full_val_interval_epochs: 3
- full_val_on_last_epoch: true
- compute_eval_aux_loss: false
- enable_batch_parameter_rollback: false
- low_altitude_deep_reroute_aux_weight: 0.0
- joint_logit_delta_scale: 0.0
- rd_specialist_residual_scale: 0.5

## Test metrics

| metric | 20260510-105622 baseline | 20260511-152913 fast | delta |
|---|---:|---:|---:|
| final accuracy | 0.8548 | 0.7903 | -0.0645 |
| final macro F1 | 0.7327 | 0.6922 | -0.0405 |
| trajectory accuracy | 0.8581 | 0.7903 | -0.0677 |
| trajectory macro F1 | 0.7486 | 0.6922 | -0.0564 |
| RD accuracy on RD | 0.7173 | 0.5527 | -0.1646 |
| RD macro F1 on RD | 0.6174 | 0.4481 | -0.1693 |
| with RD accuracy | 0.8819 | 0.8397 | -0.0422 |
| without RD accuracy | 0.7671 | 0.6301 | -0.1370 |
| source balanced macro F1 | 0.4788 | 0.4281 | -0.0507 |
| raw_tr_rd macro F1 | 0.0800 | 0.0800 | 0.0000 |
| excel_track macro F1 | 0.5948 | 0.4594 | -0.1354 |

## Per-class deltas

| class | baseline F1 | fast F1 | delta | recall delta |
|---|---:|---:|---:|---:|
| DroneTarget | 0.9252 | 0.8754 | -0.0497 | -0.0397 |
| BirdTarget | 0.7895 | 0.6667 | -0.1228 | -0.1125 |
| BalloonTarget | 0.9533 | 0.9143 | -0.0390 | -0.0556 |
| ClutterTarget | 0.4000 | 0.5263 | +0.1263 | +0.1429 |
| UnknownTarget | 0.5957 | 0.4783 | -0.1175 | -0.1667 |

## Source deltas

| source | baseline accuracy | fast accuracy | delta | baseline macro F1 | fast macro F1 | delta |
|---|---:|---:|---:|---:|---:|---:|
| cq08_track | 0.8927 | 0.8498 | -0.0429 | 0.7615 | 0.7449 | -0.0166 |
| excel_track | 0.7671 | 0.6301 | -0.1370 | 0.5948 | 0.4594 | -0.1354 |
| raw_tr_rd | 0.2500 | 0.2500 | 0.0000 | 0.0800 | 0.0800 | 0.0000 |

## Diagnosis

This checkpoint is not usable.

The main failure is not RD/joint overriding trajectory: `joint_logit_delta_scale=0.0`, and final metrics are exactly equal to trajectory metrics. The model is effectively a weakened trajectory model with RD/joint only reported as auxiliary diagnostics.

The largest regression is trajectory anchor quality, especially on `excel_track` without RD. RD evidence also regresses heavily versus the 20260510 baseline. The current fast config reduced or disabled several training signals that helped the old model:

- training shortened from 12 epochs to 5;
- deep reroute auxiliary loss disabled;
- forced fusion and gate losses reduced;
- joint logits isolated from final;
- RD residual contribution reduced;
- validation was accelerated, though full val still ran at epoch 3 and 5.

The next run should restore the stronger trajectory/RD evidence training recipe while keeping the safety rule that joint logits do not enter final until joint-on-RD beats trajectory-on-RD.
