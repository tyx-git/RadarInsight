# V27 实验

保存 V27 系列实验归档。V27a、V27b、V27c 和 V27d 为系列归档；V27d 的重要转折子实验 `r17`–`r21` 另行保存。使用 Excel 数据源的 V27d-r17 至 r21 在实验标题中加 `[E]`。

`00_基线` 和 `01_V27a` 保留为早期历史协议与架构材料；V27a、V27b、V27c 的实验提交已保留在 Git 历史中，V27d 的新增实验按系列和重要子实验分批提交。大型 checkpoint、缓存和日志不复制到文档目录。

## 代码归档映射

V27 代码已按子实验拆解为独立包，保存在 `experiments/V27/`（共 31 个包，带 `r` 的子实验嵌套为目录，如 `V27d/r15`、`V27b/r6`；索引见 `experiments/V27/README.md`）。

| 文档 | 代码归档包 |
|---|---|
| `基线/V27完整L1_L2_L3系统基线与架构合同.md` | `V27_baseline`（唯一携带 ps1 的包） |
| `V27a/V27a_微多普勒逐帧证据解耦实验.md` | `V27a`（台账记为 V27a-r3） |
| `V27b/实验归档.md` | `V27b` + `V27b/r6` |
| `V27c/实验归档.md` | `V27c` + `V27c/r22` + `V27c/r22e` + `V27c/r23` |
| `V27d/实验归档.md`、`V27d/V27d-r1_*.md` | `V27d`（台账 V27d-r1 = 本包 r1 control 臂）+ 21 个 `V27d/r*` 子包 |
| 台账未收录的 `V27d/r01`、`V27d/r02`、`V27d/r22` | 同名包，标题取自代码协议 docstring，属文档缺口 |

与 V21–V26 不同：`src-v3/v27/**` 迁移后**不删除**。它同时是配置、cache、manifest、checkpoint 的产物根，也是各 `protocol.py` 实现文件哈希清单绑定的路径（如 `v27_baseline/protocol.py:848` 起列出 `src-v3/v27/...` 与 `src-v3/radar_ln3/...`），删除即破坏契约校验与历史复现。

## smoke 剥离（2026-10-17 清理轮）

按用户口径，临时自检脚本与临时结果一并清除：删除包内 `smoke.py`、`src-v3/v27/**/smoke.py`（29 个）、`src-v3/smoke/**`（248 个 V4–V12 自检脚本）、`src-v3/validate/**`（12 个）以及各链路的 smoke 结果 json、`data/processed` 与 `config` 下的 smoke 件；同时剥离 `cli.py` 的 `smoke` 子命令与 `protocol.py` 哈希清单里的 `smoke.py` 项，57 个 ps1 去掉 `Invoke-Smoke`、`'smoke'` 阶段与 `-Stage smoke` 入口。

后果：归档包只保留代码与 CLI，**契约自检停用**。封存清单仍记录被删 `smoke.py` 的哈希（`V27-manifest.json` 的 25 项 `code_hashes` 现恰缺该文件），复算必然判为差异，属预期状态。需要原件时用 `git show pre-smoke-cleanup-20261017:src-v3/v27/v27_baseline/smoke.py` 取回。

剥离后各包 README 的「运行」示例由 `-Stage smoke` 改为 `-Stage prepare`，入口列不再含 `smoke`。

## 重复结果与 archive 旧快照清理（2026-10-17 第二轮）

`src-v3/v27/**/archive/**` 中**未被任何活文件引用**的旧快照删除 258 个（V27d 各 `pre-*-fix` 的 cache/checkpoint 1.68 GB、V27a 轮转 manifest 7 个）；被 `reports/*.json`、`runs/*.json`、`config/*.json`、`experiment.md` 按文件名引用的快照全部保留，provenance 链不断。V27a/V27d 的 `-vN` manifest 版本族只保留序号最大的一版（如 `V27a-r3-global-best-manifest-v9.json`）。逐字节相同的重复结果各保留一份。删除台账见 `tmp/cleanup2/deleted_ledger.txt`。
