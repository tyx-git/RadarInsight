# 第一阶段实验清单

## 目标
第一阶段只做三组单源实验，目的是先判断各来源单独是否可学，不急于追统一10类最高分。

## 数据来源
- `LegacySession`：`77` 个 session
- `CQ-08-MAT`：`911` 个 session
- `Carrada`：`248` 个 session

对应的 session 列表文件位于：
- `artifacts/phase1_lists/legacysession_sessions.json`
- `artifacts/phase1_lists/cq_08_mat_sessions.json`
- `artifacts/phase1_lists/carrada_sessions.json`

## 实验1：Legacy-only
目的：建立旧主线四类基线，作为后续对照参考。

运行方式：
- 使用 `artifacts/phase1_lists/legacysession_sessions.json` 中的 `session_id`
- 调用 `manage.py train` 时把这些 `session_id` 作为位置参数传入

示例命令模板：
```powershell
.\.venv\Scripts\python.exe manage.py train --train-control train_10cls.json --device cuda <这里接 Legacy-only 的 session_id 列表>
```

重点观察：
- `AircraftTarget`
- `BirdTarget`
- `DroneTarget`
- `ShipTarget`

## 实验2：CQ-08-only
目的：判断 `CQ-08` 单独训练时是否稳定，尤其关注 `Bird/Drone/Balloon/Clutter/Other` 五类边界。

运行方式：
- 使用 `artifacts/phase1_lists/cq_08_mat_sessions.json` 中的 `session_id`

示例命令模板：
```powershell
.\.venv\Scripts\python.exe manage.py train --train-control train_10cls.json --device cuda <这里接 CQ-08-only 的 session_id 列表>
```

重点观察：
- `BirdTarget`
- `DroneTarget`
- `BalloonTarget`
- `ClutterTarget`
- `OtherTarget`

## 实验3：Carrada-only
目的：判断 `Carrada` 单独训练时是否稳定，重点观察 `People/Cyclist` 边界。

运行方式：
- 使用 `artifacts/phase1_lists/carrada_sessions.json` 中的 `session_id`

示例命令模板：
```powershell
.\.venv\Scripts\python.exe manage.py train --train-control train_10cls.json --device cuda <这里接 Carrada-only 的 session_id 列表>
```

重点观察：
- `CarTarget`
- `PeopleTarget`
- `CyclistTarget`
- `PeopleTarget -> CyclistTarget`
- `CyclistTarget -> PeopleTarget`

## 统一记录要求
三组实验都必须记录：
- `accuracy`
- `macro_f1`
- confusion matrix
- 各类准确率

## 判断标准
- 如果某来源单独都学不稳，说明问题主要在该来源内部
- 如果某来源单独能学，但一混合就退化，说明主要问题在 cross-source conflict
- 只有三组单源基线明确后，第二阶段的两两组合实验才有意义
