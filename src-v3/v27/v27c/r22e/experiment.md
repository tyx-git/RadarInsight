# V27c-r22e：冻结 r22 的派生评估修订

## 边界

`r22e` 不是新的训练实验，也不改变 V27c-r22 的模型、数据、seed、checkpoint 或
训练 manifest。它只在独立目录中读取并验证 r22 的原始 manifest、训练摘要与固定
epoch-20 checkpoint，然后重做 known-only validation。

## 原因

原始 r22 evaluator 在写入任何正式报告前失败：安装的 NumPy 2 不再提供
`numpy.ndarray.amax` 实例方法。修订将该数值归约替换为等价的
`np.max(np.abs(values), axis=-1)`。

此外，r22e 将原始 sidecar 中的 motion 覆盖率和各 arm 实际送入模型的 motion 覆盖率
分开记录：control 与 zero-motion 反事实的后者必须为零。这是审计澄清，不改变 logits
或模型行为。

## 产物

- evaluator manifest：`manifests/V27c-r22e-evaluation-manifest.json`
- 单 seed 配对报告：`reports/V27c-r22e-seed{seed}-paired-validation.json`
- 三 seed 汇总：`reports/V27c-r22e-three-seed-summary.json`

所有 r22e 产物均包含 r22 manifest file hash、r22e manifest hash，以及已评估的训练
checkpoint 与 training-summary hash。
