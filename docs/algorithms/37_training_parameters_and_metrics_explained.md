# 训练参数与指标讲解
## 为什么要读这篇文档
训练日志里经常会出现 `train_loss`、`val_loss`、`accuracy`、`macro_f1`、`epoch`、`batch_size`、`lr` 这些词。很多初学者看到它们时，只知道“数字在变化”，但不知道这些数字到底代表什么、该怎么判断训练是否正常。这篇文档的目标，就是把当前项目训练中最常见、最重要的参数和指标讲清楚。
## 最基础的训练参数
### `epoch`
`epoch` 表示模型完整看过一遍训练集。比如训练 `5` 个 `epoch`，就是把整个训练集从头到尾学了 `5` 遍。`epoch` 太少，模型可能还没学会；`epoch` 太多，模型可能开始过拟合。
### `batch_size`
`batch_size` 表示每一步训练同时送进模型多少个样本。例如 `batch_size=4`，表示一次更新时同时看 `4` 个样本。`batch_size` 小，训练更稳但更慢；`batch_size` 大，吞吐更高，但更吃显存，也可能改变训练行为。
### `lr`
`lr` 是 `learning_rate` 的简写，表示学习率。它控制模型每次参数更新的步子有多大。`lr` 太大，模型容易震荡甚至发散；`lr` 太小，模型学习很慢，可能很多轮下来都没有明显提升。
### `weight_decay`
`weight_decay` 是权重衰减，也是一种正则化。它的作用是避免模型参数变得过大，帮助减少过拟合。你可以把它理解为“让模型不要太激进”。
### `num_workers`
`num_workers` 控制数据加载时使用多少个子进程。在调试、Windows、本地 CPU 环境中，`0` 往往最稳；在更强的训练环境里，可以适当调大来提升数据加载速度。
## 最重要的损失指标
### `train_loss`
`train_loss` 表示模型在训练集上的错误程度。在本项目里，它不是单一损失，而是多个损失的组合，可能包括：主分类损失、轨迹辅助损失、RD 辅助损失、粗分类损失、细分类损失、细边界损失。因此 `train_loss` 看起来偏高，不一定代表训练坏了，也可能只是因为损失项本来就比较多。
### `val_loss`
`val_loss` 表示模型在验证集上的错误程度。验证集不参与参数更新，所以它更能反映模型对“没见过的数据”的表现。通常来说，`train_loss` 下降说明模型会拟合训练数据，`val_loss` 下降说明模型泛化能力可能在改善。
### `loss`
在 `evaluation_report.json` 里通常会看到一个单独的 `loss`。它表示独立评估阶段的损失值。这个值和训练阶段的 `train_loss` 不一定完全同义，因为训练时可能叠加了更多辅助损失，而独立评估时往往更接近最终决策头的损失。也就是说，训练 `loss` 和评估 `loss` 不能机械地一一对比。
### 如何一起看 `train_loss` 和 `val_loss`
如果 `train_loss` 和 `val_loss` 一起下降，通常是好现象；如果 `train_loss` 持续下降但 `val_loss` 上升，通常说明开始过拟合；如果两者都很高且长期不动，通常说明模型结构、输入特征或数据链路存在问题。
## 准确率与宏平均指标
### `train_accuracy`
`train_accuracy` 表示训练集上的分类正确率。比如 `0.90` 就表示训练样本中有 `90%` 被判对。它能反映模型是否把训练数据学会了。
### `val_accuracy`
`val_accuracy` 表示验证集上的分类正确率。它比 `train_accuracy` 更重要，因为它更接近模型的真实泛化能力。
### `accuracy`
`accuracy` 一般出现在最终评估文件里，表示独立测试集或独立评估集上的整体正确率。这个值直观，但不能单独作为唯一判断标准，因为它可能被某个容易类别“拉高”。
### `macro_precision`
`macro_precision` 是宏平均精确率。它会先分别计算每个类别的 `precision`，再把各类平均。它回答的是：如果把每个类别都看得一样重要，那么模型整体的“预测为某类时有多准”表现如何。这个指标适合观察模型是否对某一类特别偏置。
### `macro_recall`
`macro_recall` 是宏平均召回率。它会先分别计算每个类别的 `recall`，再做平均。它回答的是：如果每个类别都同等重要，模型整体能把各类真实样本找回来多少。这个指标非常适合发现“某一类持续漏检”的问题。
### `macro_f1`
`macro_f1` 是宏平均 F1 分数，也是本项目最重要的指标之一。它先计算每个类别自己的 `f1`，然后做平均，因此不会让大类、易分类把结果“冲高”。在当前项目里，`macro_f1` 往往比单纯的 `accuracy` 更能反映模型是否真的全面变好。
### 为什么本项目特别重视 `macro_f1`
因为雷达识别任务里，经常会出现某个类别特别容易、某个类别特别难的情况。如果只看 `accuracy`，模型可能看起来很强，但实际上某个关键类别几乎全错。`macro_f1` 更适合发现这种“整体看着不错、细节其实很差”的问题。
## 单类别指标
在 `evaluation_report.json` 里，每个类别通常都会有 `precision`、`recall`、`f1` 和 `support`。
### `precision`
`precision` 表示：模型一旦预测成这个类，它有多大概率是真的。精确率高，说明这个类的误报少。
### `recall`
`recall` 表示：真实属于这个类的样本里，模型找回来了多少。召回率高，说明漏检少。
### `f1`
`f1` 是 `precision` 和 `recall` 的综合指标。它适合评价“既要少误报，也要少漏报”的情况。
### `support`
`support` 表示评估集中这个类别一共有多少真实样本。它能帮助你判断某个类别指标是否可靠。如果某类只有极少样本，那么该类指标波动可能会很大。
## 训练后最关键的文件
### `training_history.json`
这个文件按 `epoch` 记录训练过程，通常包括：`train_loss`、`train_accuracy`、`train_macro_f1`、`val_loss`、`val_accuracy`、`val_macro_f1`、`learning_rate`。它适合看训练趋势、过拟合、收敛是否稳定。
### `training_metadata.json`
这个文件保存训练配置、标签映射、参数量、数据切分、归一化统计等。它是复现实验的重要依据。
### `evaluation_report.json`
这个文件保存独立评估结果，通常包括：`loss`、`accuracy`、`macro_precision`、`macro_recall`、`macro_f1`、`confusion_matrix`、`per_class`。这是判断模型是否真的提升的核心文件。
### `best_fusion_model.pt`
这是训练过程中保存下来的最佳 checkpoint。后续评估、图表生成、校准，通常都基于它展开。
## 输入尺寸相关参数
### `sequence_length`
表示轨迹分支一次使用多少个时间步。如果它太大，模型可能过度依赖插值后的伪时序点；如果太小，又可能丢掉轨迹动态信息。
### `rd_frames`
表示每个样本使用多少帧 RD 数据。它决定 RD 分支能看到多长的时间演化过程。
### `rd_height`
表示 RD 张量的高度，通常与距离方向有关。
### `rd_width`
表示 RD 张量的宽度，通常与多普勒方向有关。
### `rd_input_channels`
表示 RD 分支输入了多少个通道。当前项目已经不是单通道幅度图，而是多通道组合，目的是从不同角度描述微动结构。
## 结构相关参数
### `track_projection_dim`
轨迹分支早期投影层的隐藏维度。
### `tcn_channels`
`Residual TCN` 每一层的通道规模。
### `transformer_layers`
轨迹分支中 Transformer 编码层的数量。
### `transformer_heads`
Transformer 的多头注意力头数。
### `fusion_hidden_size`
融合层与主表示空间使用的隐藏维度。
### `x3d_stem_channels`
RD 分支 `X3D-M` 风格主干的 stem 通道数。
### `x3d_stage_channels`
RD 主干各 stage 的通道规模。
## 训练稳定性相关参数
### `label_smoothing`
标签平滑会把硬标签稍微变软一点，作用是减少模型过度自信，有时能带来更稳的泛化能力。
### `use_class_balanced_loss`
如果启用，损失函数会更关注样本相对少、相对难的类别。
### `class_balance_beta`
控制类别平衡损失有多强。
### `warmup_epochs`
学习率预热轮数。预热的作用是训练开始时不要一下子用太大的学习率，先慢慢升上去，通常更稳。
### `max_grad_norm`
梯度裁剪阈值，用来限制梯度过大，降低训练不稳定或梯度爆炸的风险。
## 多头训练相关参数
### `track_aux_loss_weight`
轨迹辅助头损失的权重。它越大，模型越会被要求“即使只看轨迹，也得学出区分能力”。
### `rd_aux_loss_weight`
RD 辅助头损失的权重。它越大，模型越会被要求“即使只看 RD，也得学出区分能力”。
### `hierarchical_aux_loss_weight`
粗分类头的损失权重。它帮助模型先学会大类分离，再做细分类。
### `detail_aux_loss_weight`
细分类头的损失权重。它直接强化细粒度类别区分能力。
### `detail_boundary_loss_weight`
细边界损失权重。它用来拉开同一粗类内部各细类之间的边界。
### `detail_boundary_margin`
细边界损失里使用的目标边距。你可以把它理解为“不同细类之间至少要分开多少”。
### `detail_logit_residual_weight`
细分类 logits 残差注入最终输出时的权重。这个值越大，最终结果越容易受到细分类头影响。
### `detail_cosine_scale`
余弦细分类头里的缩放参数。因为余弦分类器输出的是角度相似度，不是普通线性 logits，所以需要合适的缩放，训练才会稳定。
## 一次健康训练通常长什么样
一个比较健康的训练过程，通常表现为：`train_loss` 下降，`train_accuracy` 上升，`train_macro_f1` 上升；`val_loss` 不会持续爆炸，`val_macro_f1` 能跟着提升，哪怕速度比训练集慢。如果训练集指标越来越漂亮，但验证集一直差或者大幅波动，那通常意味着过拟合、数据切分不合理，或者输入特征链路仍存在问题。
## 在本项目里，最应该优先看什么
在这个项目里，推荐的观察优先级通常是：先看 `macro_f1`，再看混淆矩阵，再看每类 `f1`，再看 `accuracy`，最后看 `loss`。原因很简单：`accuracy` 容易被大类或容易类“抬高”，而 `macro_f1` 更能暴露真正的类别不平衡问题。
## 给初学者的最短记忆版
如果你只想先记住最核心的几项，请记这几个：`epoch` 是训练轮数，`batch_size` 是每步样本数，`lr` 是学习率，`train_loss` 和 `val_loss` 看错误程度，`accuracy` 看整体正确率，`macro_precision` 看整体预测准度，`macro_recall` 看整体找回能力，`macro_f1` 看整体平衡表现。真正判断模型是否进步时，不要只盯一个数字，而要把这些指标和混淆矩阵一起看。
