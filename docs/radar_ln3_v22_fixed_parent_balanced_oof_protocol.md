# RADAR-LN3 V22 KnownGate固定parent分母OOF协议

状态：代码修复与实验协议已实现；正式五折训练进行中，fold 0--2已完成，fold 3曾在首个checkpoint前因`nvcuda64.dll`原生崩溃中断。

## 1. 已确认机制

V21e使用逆频率class weight，但历史loss调用PyTorch加权`mean`。该归约的分母是当前batch中目标权重之和，而不是固定parent数。`batch_size=2`且同类目标batch占约93.1%–93.6%时，同类batch中的class weight会完全抵消。因此历史配置声明的Clutter大权重没有按全局parent目标稳定作用。

这证明的是loss机制缺陷，不预先证明V22性能一定改善。Clutter P/R同步改善和跨fold方差下降仍是假设，必须由完整五折成对OOF验证。

## 2. 唯一实验变量

- Control：冻结V21e seed43 OOF Treatment，`class_weight_normalization=batch_weight_sum`。
- Treatment：V22，`class_weight_normalization=fixed_parent_count`。
- 公式：每个batch的final loss为`sum(w[y_i] * CE_i) / parent_count`。
- prefix：先在每个parent内部平均合法prefix，再乘该parent class weight，最后除以有prefix的parent数。
- 每折逆频率权重保持V21e原值，且在完整fold-train人口上的平均值精确为1。

V22不改采样器、不增大batch、不加入新特征、不改网络、dt、液态、optimizer、学习率、seed、20 epoch、parent顺序、五折assignment、normalization、checkpoint时点或标签。

## 3. 配对与数据边界

- 复用V21e的986个known train parent、975个capture/content provenance component和五折assignment。
- 每个V22折的模型初态与对应V21e Control逐tensor相等。
- 每个epoch parent顺序必须与V21e对应折完全一致。
- held-out折只在第20 epoch结束后前向一次，不用于checkpoint选择。
- UnknownTarget、development validation与test均不进入训练、归一化、选优或阈值拟合。

## 4. 预注册验证

性能猜测只有同时满足以下条件才记为支持：

1. pooled parent-final Clutter Precision、Recall、F1均严格改善；
2. KnownTarget Recall不下降，Drone/Bird/Balloon各自net rescue均非负；
3. eligible-prefix Macro-F1、KnownTarget Recall、Clutter Recall均不下降；
4. 五折Clutter Precision、Recall、F1的描述性population standard deviation均不增，且至少一项严格下降；
5. capture/content provenance-component paired bootstrap的Clutter Precision、Clutter Recall与Macro-F1差值95%下界均大于0。

满足上述全部条件只说明固定分母机制获得支持；还需parent Macro-F1严格提升超过2个百分点，才记为可采纳候选信号。由于Clutter总support只有33、每折仅6–7个，第4项只能作为描述性证据，不能单独证明稳定性。若任一安全门槛失败，则猜测不成立，保留V21e研究基线并停止把V22推进L3。

## 5. 未覆盖范围

V22仍只是KnownGate局部训练实验。PENDING_ABSTAIN没有直接监督；L2 DBB knownness fallback、L3阈值、Unknown open-set、正式拒识、完整级联、stable、test、P95和部署均为“未执行”。

## 6. 中断恢复与CUDA运行前检查

- 每个待训练fold启动前由PowerShell流水线直接查询GPU 0显存；可用显存低于4096 MiB时停止，不创建或移动训练产物。该检查只约束运行环境，不改变模型、数据、随机数或实验变量。
- 已有完整`last_model.pt`的fold只允许从epoch边界恢复。
- 首个checkpoint前发生进程级中断时默认fail-closed，原artifact保持不动。人工排除外部原因后，必须显式传`-RestartPreCheckpointFolds <fold>`；脚本仅在目录中不存在任何`.pt`且文件只包含`training.log`/`failure.json`时，将整个旧目录改名归档，再从冻结初态完整重训。
- 2026-08-19 fold 3故障的Windows Application事件为`python.exe`在`nvcuda64.dll`中触发`0xC0000409`（BEX64）；同一时刻GPU由游戏等进程高占用。fold 3首三个确定性DataLoader batch已在CPU完成前向、V22 fixed-parent loss及反向，loss与梯度均有限，因此没有发现fold 3数据或V22 loss特例。
