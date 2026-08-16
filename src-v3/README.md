# RADAR-LN3 src-v3入口

当前产品架构请先阅读[`docs/radar_ln3_v21_strict_open_set_architecture.md`](../docs/radar_ln3_v21_strict_open_set_architecture.md)。

- V21及后续：`known_gate -> dbb_specialist -> semantic_decision_layer`，Unknown只作为严格open-set动作/真值语义。
- V5-V20：历史实验与artifact复现代码。`dbb_cu_gate`和`cu_specialist`不得用于推导当前架构。
- `task_plan_v3.md`、`process_v3.md`、`findings_v3.md`是累积审计日志；其中早期架构描述按当时版本成立，发生冲突时以V21架构文档为准。
- test、正式拒识、stable、P95和部署状态必须以对应版本报告为准，不得从局部训练完成状态推断。
