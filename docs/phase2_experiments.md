# 第二阶段实验清单

## 目标
第二阶段只做两两组合实验，目的是定位：

- 哪一对来源一混就开始退化
- 是哪一对来源的冲突最强
- 10类退化到底更像来自 `CQ-08` 还是 `Carrada`

## 实验组合

### 实验1：Legacy + CQ-08
- 列表文件：`artifacts/phase2_lists/legacy_cq08_sessions.json`
- 总 session 数：`988`

建议命令：
```powershell
.\.venv\Scripts\python.exe manage.py train --train-control train_10cls.json --data-root data/sessions --output-dir artifacts/phase2_legacy_cq08 --device cuda --session-id-file artifacts/phase2_lists/legacy_cq08_sessions.json
```

重点观察：
- `Legacy` 四类是否被明显拖坏
- `Bird/Drone` 是否因为 `CQ-08` 加入而更混
- `Balloon/Clutter/Other` 是否稳定

### 实验2：Legacy + Carrada
- 列表文件：`artifacts/phase2_lists/legacy_carrada_sessions.json`
- 总 session 数：`325`

建议命令：
```powershell
.\.venv\Scripts\python.exe manage.py train --train-control train_10cls.json --data-root data/sessions --output-dir artifacts/phase2_legacy_carrada --device cuda --session-id-file artifacts/phase2_lists/legacy_carrada_sessions.json
```

重点观察：
- `Legacy` 四类是否被明显拖坏
- `People/Cyclist` 是否稳定
- `Carrada` 是否因为混入旧数据而退化

### 实验3：CQ-08 + Carrada
- 列表文件：`artifacts/phase2_lists/cq08_carrada_sessions.json`
- 总 session 数：`1159`

建议命令：
```powershell
.\.venv\Scripts\python.exe manage.py train --train-control train_10cls.json --data-root data/sessions --output-dir artifacts/phase2_cq08_carrada --device cuda --session-id-file artifacts/phase2_lists/cq08_carrada_sessions.json
```

重点观察：
- `People/Cyclist`
- `Bird/Clutter/Other`
- 这组实验最容易暴露 `pseudo-track` 与 `native track` 的冲突

## 统一评估要求
三组实验都必须统一记录：

- overall `accuracy`
- overall `macro_f1`
- source 内部分项表现
- 关键混淆链

关键混淆链至少记录：

- `PeopleTarget -> CyclistTarget`
- `CyclistTarget -> PeopleTarget`
- `BirdTarget -> OtherTarget`
- `ClutterTarget -> BirdTarget`
- `ClutterTarget -> OtherTarget`

## 判断标准
- 如果 `Legacy + CQ-08` 掉得最厉害，说明 `CQ-08` 是主要扰动源
- 如果 `Legacy + Carrada` 掉得最厉害，说明 `Carrada` 是主要扰动源
- 如果单独都还行，但 `CQ-08 + Carrada` 掉得最厉害，说明主要问题在跨来源冲突，而不是单源质量
