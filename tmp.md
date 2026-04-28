同样 3 epoch、无预训练/无专家 checkpoint，accuracy 78.3%、macro F1 55.6%。

  - Expert1 CUDA：accuracy 0.7710，macro_f1 0.6630
  - Expert2 shared-head CUDA：accuracy 0.9848
  - Expert4 shared-head CUDA：accuracy 0.9412
  - System CUDA，router 未均衡：accuracy 0.7380，route_macro_recall 0.3333
  - System CUDA，balanced router 回灌后：accuracy 0.7530，route_macro_recall 0.4510


191524 取得了最高测试 accuracy/macro_f1（0.8677 / 0.7794）且无 RD accuracy 0.808，但它没有通过新的 RD Bird 单边门槛，best selection 回落为 fallback。这
  个结果说明：高精度和 RD Bird 质量仍冲突。