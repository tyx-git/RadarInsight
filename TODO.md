实验版本：V24 course-reason-aware-baseline。变量：无；从头重训 V21e 自研 KnownGate，冻结因果航向圆形不一致度、48 宽 CfCLiquidCausalEncoder、recurrent LayerNorm、bounded_observation（zero-dt gain=0.25）、masked_mean RD、loss、数据、初始化协议与 20 epoch。验证：严格复现 V21e 开发集与训练合同，并固定 parent-final、eligible-prefix、window、Clutter support、parent bootstrap/Wilson 区间及 Known→Clutter/Clutter→Known parent 审计，作为 V24a-c 的唯一配对 Control。

实验版本：V24a rd-peak-drift-counterevidence。变量：仅在 V24 baseline 上追加当前事件相邻 RD 帧峰速度漂移均值特征 current_event_adjacent_rd_peak_velocity_drift_mean_fraction；RD 帧不足两帧时仅标记 invalid，不伪造零值，其他模型、loss、dt、数据顺序、初始化和 20 epoch 全冻结。验证：该 RD 反证据能否减少高航向不一致 Bird→Clutter 与总体 Known→Clutter，同时 Clutter→Known、KnownTarget Recall、eligible-prefix 安全指标均不恶化；必报航向不一致、RD 有效性、capture 分层和逐 parent 改变。

实验版本：V24b course-rd-contradiction-interaction。变量：仅在通过 V24a 方向性检查后，增加固定交互特征 q_t=I_t*d_t，其中 I_t 为因果航向圆形不一致度，d_t 为 V24a RD 峰速度漂移；任一特征 invalid 时交互项 invalid，其他条件全冻结。验证：高航向不一致但 RD 仍具目标动态的轨迹能否被作为 Bird/目标反证据保留，从而进一步降低高置信 Bird→Clutter，且不得牺牲 Clutter Recall 或前缀安全性。

实验版本：V24c same-second-skip-diagnostic。变量：仅将自研主干 zero-delta 更新从 bounded_observation 改为 skip，即同秒事件 alpha=0；首事件初始化、正 dt 连续更新、特征、loss、数据顺序、初始化和 20 epoch 全冻结。验证：同秒观测频率是否构成 KnownGate 的类别/采样捷径；按 zero-delta 比例、航迹长度及类别分层比较 parent-final、eligible-prefix 与 window，确认移除同秒状态累积后是否系统性改变 Clutter/Bird 边界。
