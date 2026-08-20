# RADAR-LN3 V21e：KnownGate RD模态只读消融

## [KnownGate]

### [Direct:核验冻结V21e五折模型对RD内容和RD质量提示的条件依赖]

[]1. 原始臂逐折加载冻结V21e Treatment的epoch 20 checkpoint，在原held-out parent上重放；parent、window、eligible-prefix硬预测必须逐项等于正式OOF prediction，概率最大绝对误差不得超过`1e-6`。复现失败即停止，不生成消融结论。

[]2. `rd_content_zero_quality_preserved`臂把RD embedding强制为精确零，同时保留原始7维因果RD质量向量。该输入有意制造“质量称RD存在、内容却为零”的非物理反事实，只能诊断冻结模型对RD张量内容的依赖，不能作为可部署缺失输入。

[]3. `rd_contract_missing`臂把RD embedding强制为精确零，并把每个非padding事件的质量向量设为合法缺失语义`[0,0,0,1,0,0,0]`。全航迹所有事件均缺RD仍属于分布偏移，只能诊断冻结模型条件依赖，不能代表重新训练无RD模型的性能。

[]4. 三臂共享同一checkpoint、held-out parent、顺序、TR、dt、course evidence和推理代码；不创建optimizer、不反向传播、不选checkpoint、不拟合阈值，不读取Unknown、validation或test。

[]5. 分别报告parent-final、eligible-prefix和全部window指标，按Drone/Bird/Balloon/Clutter、fold、最终RD帧数、RD事件覆盖率、parent长度和最终RD缺失分层；输出逐parent logit margin变化、硬预测转移、exact McNemar以及capture/content provenance-component配对bootstrap。

[]5a. 所有实际读取的五折checkpoint、配置、index、normalization、course cache、held-out prediction和986份RD tensor都必须在推理前后通过SHA-256一致性检查；另报告fusion中TR embedding、RD embedding、RD quality和course evidence四个输入块的权重范数，仅用于确认连接，不从权重绝对值推断类别方向。

[]6. 原始减消融的差值只表示冻结模型在当前数据上的条件依赖。由于液态状态和fusion非线性，两个消融臂的差不能解释成可加的“质量贡献”；消融近中性也不能证明RD物理上无用，消融负向也不能证明新增RD特征必然提升。

[]7. 本实验是KnownGate known-only局部诊断。Unknown仍是严格open-set且不作为训练类；完整L1/L2/L3、阈值、正式拒识、test、stable和P95均不执行。
