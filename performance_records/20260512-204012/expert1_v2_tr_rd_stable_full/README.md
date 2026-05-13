# Expert1 V2 TR+RD stable full 20260512-204012

Source artifact: `artifacts/20260512-204012/expert1_v2_tr_rd_stable_full`

Reports:

- `reports/real/20260512-204012/expert1_v2_tr_rd_stable_full.md`
- `reports/real/20260512-204012/expert1_v2_tr_rd_stable_full.json`

Key test metrics:

| metric | value |
|---|---:|
| accuracy | 0.8645 |
| macro F1 | 0.7526 |
| trajectory macro F1 | 0.7358 |
| joint-on-RD macro F1 | 0.7618 |
| joint-on-RD - trajectory-on-RD | +0.0158 |
| RD evidence on-RD macro F1 | 0.5677 |
| with-RD / without-RD accuracy | 0.8734 / 0.8356 |
| source-balanced macro F1 | 0.5217 |
| cq08 / excel / raw macro F1 | 0.7659 / 0.5991 / 0.2000 |

Conclusion:

Stable training recovered trajectory and without-RD/Excel, but reduced RD/joint contribution. This is not a System-ready Expert1 candidate.
