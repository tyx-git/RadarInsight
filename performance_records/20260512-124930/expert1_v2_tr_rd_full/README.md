# Expert1 V2 TR+RD Full 20260512-124930

Source command:

```powershell
.\.venv\Scripts\python.exe -m src.core.train_expert1 --config tmp/expert1_v2_tr_rd_full_config.json
```

Key result:

- Best validation epoch: 11
- Test final accuracy / macro F1: 0.8613 / 0.7706
- Test trajectory accuracy / macro F1: 0.7774 / 0.6993
- Test joint-on-RD accuracy / macro F1: 0.8987 / 0.7964
- joint-on-RD minus trajectory-on-RD macro F1: +0.0863
- final rescue / harm / net on RD: 31 / 5 / +26
- RD-only rescue / harm / net on RD: 21 / 41 / -20
- source-balanced macro F1: 0.5228
- cq08 / excel / raw macro F1: 0.8007 / 0.5677 / 0.2000

Conclusion:

TR+RD joint is useful on RD samples, but this run is not ready for System integration. The main bottlenecks are trajectory degradation during stage2, weak no-RD/excel performance, and unstable raw_tr_rd behavior. This record is kept as the baseline before the stable-stage training fix.
