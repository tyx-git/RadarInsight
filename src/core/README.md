# core
核心代码放在这里。
## 当前分层结构
- `data/`
  - 数据索引、特征定义、dataset、collate、dataloader
- `modeling/`
  - 轨迹编码器、路由器、专家、系统模型
- `training/`
  - baseline / system / expert1 / 其他专家 / router / pretrain 训练入口与配置
- `evaluation/`
  - 指标与中文报告
- `utils/`
  - 运行路径等工具
- 当前已完成第二阶段解耦：
  - `labels / processed_dataset / samples / track_dataset / split / rd / torch_dataset / collate / dataloaders`
  - 这些模块的**真实实现**已迁入 `src/core/data/`
  - 顶层同名文件当前仅保留为兼容壳，继续兼容旧导入路径
## 当前已建立
- `processed_dataset.py`
  - 读取 `data/processed/track_index.csv`
  - 提供处理后轨迹样本索引访问
  - 提供按 `sample_id` 加载单个标准化轨迹表的接口
  - 同时支持：
    - `label_name_raw`
    - `label_name`
  - 支持 `trainable_only=True`
    - 直接读取 `data/processed/trainable_track_index.csv`
- `labels.py`
  - 定义当前标准标签表
- `features.py`
  - 定义当前轨迹特征列与元信息列
- `samples.py`
  - 定义单个轨迹样本对象
- `dataset.py`
  - 提供面向训练输入的 `TrackDataset`
- `split.py`
  - 提供基础分层切分器 `stratified_split`
- `encoding.py`
  - 负责轨迹序列数值清洗与特征矩阵编码
- `torch_dataset.py`
  - 提供面向 PyTorch 的样本数据集
- `collate.py`
  - 提供 batch padding / mask 组织逻辑
- `dataloaders.py`
  - 提供从 `data/processed/trainable_track_index.csv` 到 dataloader 的一键构建入口
- `models.py`
  - 当前提供第一版 GRU 序列分类 baseline
- `metrics.py`
  - 当前提供基础分类指标
- `train_config.py`
  - baseline 训练配置
- `trainer.py`
  - baseline 训练循环、验证、测试和 checkpoint 保存
- `train_baseline.py`
  - baseline 训练命令行入口
## 对齐 `main.md` 后补齐的架构骨架
- `physical.py`
  - 从标准化轨迹表中提取显式物理特征
  - 对应 `main.md` 中“双通道轨迹表征”的物理特征支路
- `trajectory.py`
  - 提供 Transformer 轨迹编码器和表征构建器
  - 对应 `main.md` 中“深度时序特征 + 显式物理特征融合”
- `router.py`
  - 提供深度门控路由网络骨架
  - 对应 `main.md` 中“深度门控路由网络”
- `experts.py`
  - 提供低空可疑目标、常规空中目标、地面目标、高空规则过滤专家骨架
  - 对应 `main.md` 中“四专家协同”
- `system.py`
  - 提供轨迹建模、规则兜底路由、低空双流专家、常规空中专家、地面专家的统一系统编排入口
  - 对应 `main.md` 中“三级级联总体架构”
- `architecture.py`
  - 提供当前系统级架构规格定义
  - 用于把代码层和 `main.md` 的总架构约束显式对齐
- `system_config.py`
  - 第一版系统级训练配置
- `system_metrics.py`
  - 第一版系统级分类+路由指标
- `system_trainer.py`
  - 第一版 route-aware 多专家训练循环
- `train_system.py`
  - 第一版系统训练入口
- `expert1_config.py`
  - 专家一单独训练配置
- `expert1_data.py`
  - 专家一低空五类子集 dataloader
- `expert1_trainer.py`
  - 专家一单独训练、验证、测试与报告生成
- `train_expert1.py`
  - 专家一单独训练入口
- `expert_other_config.py`
  - 专家二/专家四单独训练配置
- `expert_other_data.py`
  - 按标签子集构建专家二/专家四 dataloader
- `expert_other_trainer.py`
  - 专家二/专家四单独训练、验证、测试与报告生成
- `train_expert2.py`
  - 专家二单独训练入口
- `train_expert4.py`
  - 专家四单独训练入口
- `router_train_config.py`
  - 路由器单独训练配置
- `router_trainer.py`
  - 路由器单独训练、验证、测试与报告生成
- `train_router.py`
  - 路由器单独训练入口
- `reporting.py`
  - 将系统训练的 `evaluation_report.json` 转成中文 markdown 报告
  - 自动写入 `reports/real/`
- 当前已进一步对齐 `main.md` 的关键结构：
  - 路由器加入了基于物理特征的规则兜底修正
  - 低空专家升级为“大参数量轨迹常开分支 + 幅度/相位 RD 多模态 Transformer 分支 + 统计特征分支”
  - 常规空中专家升级为时序卷积 + 注意力池化 + 物理分支融合
  - 地面专家升级为深特征分支 + 物理分支门控融合
  - 低空专家已把 `UnknownTarget` 正式接回全局标签体系，用于承接 CosFace 低置信 reject
  - 专家一已可脱离其他专家单独训练，便于先把低空多模态幅度/相位 RD 建模稳定下来
  - 专家二和专家四也已具备单独训练入口，后续可在 expert1 稳定后继续分别收敛
  - 路由器也已具备单独训练闭环，可在不干扰 expert1 的前提下继续推进 coarse route 建模
- `run_paths.py`
  - 统一生成按时间戳归档的训练产物目录
  - 当前训练产物会写到：
    - `artifacts/<timestamp>/<experiment_name>/`
    - `reports/real/<timestamp>/<experiment_name>.md|json`


数据层
  features.py → physical.py → encoding.py → labels.py
  理解：雷达数据有哪些维度，如何从原始 DataFrame 变成模型输入

表征层
  trajectory.py（重点！TransformerTrackEncoder → TrackRepresentationBuilder）
  models.py（GRUSequenceClassifier 作为对比基线）

路由 + 专家层
  router.py（DeepGatingRouter：神经网络 + 物理规则覆写）
  experts.py（四个专家模型，重点看 LowAltitudeSuspectExpert）
  
系统集成层
  architecture.py → system.py（所有组件如何拼成 RadarClassificationSystem）
  
训练管线
  pipeline_config.py → pipeline_trainer.py（总览训练流程）
  然后分别看各 trainer：pretrain → router → expert1 → expert_other → system

数据处理管线
  process/schema.py → build_all_processed.py → standardize_tracks.py