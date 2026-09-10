# OD-main-0909 改造计划：step4.5 动态区域 / 方向级相位 / ID 继承

> 目标仓库：`/home/moga/桌面/OD-main-0909`
> 基线：`OD--main` 在 2026-09-08 12:55 的快照（已建本地 git，baseline commit）
> 原则：**纯静态完全冻结；只改纯动态与动静结合；不做摄像头红绿灯识别；运动遮挡导致的 ID 断暂不处理。**
> 状态：**已实现并通过测试；5 个验收 clip 最终 labels / IDs 与基线一致（静态零漂移）。**
> 实现说明见本文件第 18 节。

---

## 0. 一句话目标

全链路先跑一遍，得到稳定的 car-only 轨迹；再用**有高速行驶证据的轨迹**反推动态区域；
只在动态区域内重做一次 only-car 跟踪、ID 继承和相位感知拼接；区域外静态 ID / box / yaw 一律不动。

---

## 1. 当前基线链路（OD--main）

```text
step1  lidar 推理 + 相机可见度预过滤
step2  class 预关联 → static-first 身份跟踪 → hard filter → 同中心去重
       → 短轨迹过滤 → 静态 yaw 稳定 → 整合 yaw → 最终类别一致性
step3  Car box XY 拟合 + 地面/车顶 Z 拟合
step4  轨迹稳健长度 >= 6m 的 Car 改标为 Truck
step5  点数/短链终检 → Car-only → box 转 base_link → 导出
```

关键事实：

- 区域 / 相位当前**不在主链路**；历史区域版在 step2 pass2，历史相位模块是只读诊断。
- Car-only 当前在 step5 最末尾；step4 只做 Car→Truck。
- step3 是 per-track 几何拟合，依赖 `track_id` 做尺寸中位数、车顶/地面时序回退、track 高度先验。
- step2 跟踪器 = `StaticFirstTracker`（静态 slot → 动态跟踪 → 拼接 → 拓扑 → gap recovery → slot motion coordination）+ `ConservativeTracker`（运动关联 + 物理门限）。
- 当前 tracker 在关联里使用 yaw：`_slot_cost` 硬门限、`_topology_cost` 硬门限、`_cost` yaw 项 / BEV IoU、stop/arrival bridge 的 yaw_delta。
- 当前 yaw 锁定来源：
  1. `geometry/yaw_vehicle_dynamic.py::_motion_targets`：heading 变化 >75° 时沿用旧 heading；
  2. `geometry/static_yaw.py::stabilize_static_yaw`：slot-bound 轨迹在 departure cutoff 之前整段写成停车 yaw，departure 漏检时运动/转弯段也被锁；
  3. `apply_motion_yaw` 当前默认让运动 heading 覆盖 detector yaw。

---

## 2. 已对齐的决策

| # | 决策 | 内容 |
|---|---|---|
| D1 | 静态冻结 | 纯静态停车、静态 slot、静态 yaw、静态 box 全部冻结；区域外 ID/box/轨迹不动 |
| D2 | 动态区域定义 | 只有**有高速行驶证据**的 car 轨迹扫过的区域才算动态区域；其余默认静态 |
| D3 | 动态区域形状 | 只用 swept box 面积，**`buffer_radius = 0`**；不做 1m buffer |
| D4 | 区域延伸 | 沿**稳定方向**直线延伸 **30m**，把停止线 / 排队 / 起步段包进来 |
| D5 | 重跟踪范围 | 只在动态区域内部重跟踪；静态 ID 冻结 |
| D6 | 区域 / 相位顺序 | **做法 2**：旧 car-only 轨迹建区域 → 区域内重跟踪 → 新轨迹建相位 → 相位感知拼接 |
| D7 | 相位模型 | 方向级四相位 `region/direction_phase.py`；右转常绿、掉头按左转、同轴直行/左转互斥、不同轴独立、允许许可左转 |
| D8 | ID 继承优先级 | 静态锚 ID > 高速主轨迹旧 ID > 新 ID |
| D9 | 多合一 | 按旧 ID 观测数投票，哪个多继承哪个 |
| D10 | 一拆多 | 一般不应发生；真发生原 ID 给主轨迹，其余发新 ID |
| D11 | 合并校验 | 同帧不允许重复 ID；同一 ID 不允许不符合物理的跳变 |
| D12 | 槽位释放 | 必须确认车 1 离开后，才释放停车位 B 与车 1 ID 的绑定；不允许跨车复用 slot ID |
| D13 | yaw 解耦 | **只针对动态**；动态关联只用运动轨迹，不用 yaw 门限；静态保持原样 |
| D14 | step2 yaw | 动态段用 detector 原始 yaw（`apply_motion_yaw=False`）；静态停车 yaw 逻辑保持原样；`stabilize_static_yaw` 增加逐观测静止判定 |
| D15 | 物理保护 | 保留 `reverse_step_gate`、速度/加速度/方向连续性、距离/Mahalanobis、`stop_bind` / `arrival_bind` / `departure` / `ingress`、gap recovery；step4.5 的 tracker 必须 `enforce_motion_constraints=True` |
| D16 | Car-only | step4 先 `Car → Truck`，再删除 Truck / 非 Car；只有 Car 进入 step4.5 |
| D17 | 第二遍 box fit | step4.5 最终 ID 定下来后，只对动态 / 重跟踪段跑第二遍 box fit；静态冻结段保留 step3 结果 |
| D18 | 验收 clips | 使用 `~/桌面/new/` 下的 5 个 clip |
| D19 | 诊断 | `run_end_to_end.py` 增加 `--keep-intermediate`；step4.5 诊断写入 `work/` |
| D20 | 步骤命名 | 新增 `pipeline/step4_5_region_phase_retrack.py`；step5 仍为最终过滤 / 导出 |

---

## 3. 目标链路

```text
step1  lidar 推理 + 可见度（不变）
step2  class 预关联 → static-first 身份跟踪 → hard filter → 去重 → 短轨迹
       → 静态 yaw 稳定（纯停车段保持原样）
       → 整合 yaw（动态段 apply_motion_yaw=False，保留 detector yaw）
       → 物理回推保护保留
step3  Car box 拟合（不变；为 step4.5 提供初始 box）
step4  Car→Truck，然后只保留 Car（Truck / 非 Car 全删）
step4.5
  4.5a  用 step4 的 car-only 轨迹建动态区域（高速证据 + swept box + 稳定方向 30m 延伸）
  4.5b  在动态区域内重跑 only-car 跟踪（运动-only 关联、物理保护 ON、静态 slot 逻辑禁用）
  4.5c  用重跟踪后的轨迹建方向级四相位
  4.5d  ID 继承 + 槽位释放检查 + 相位感知拼接
  4.5e  只对动态 / 重跟踪段跑第二遍 box fit
step5  最终点数/短链过滤 + Car-only 兜底 + 转 base_link + 导出
```

---

## 4. step4：Car-only

### 4.1 行为

输入：step3 JSON。
输出：只含 `class_name == "Car"` 的检测；大尺寸 Car 先被改成 Truck，再被删除。

顺序固定：

1. 对 `class_name == "Car"` 的轨迹计算 `median(max(dx, dy))`；
2. `>= 6.0m` 的整条轨迹改标为 `Truck`；
3. 删除所有规范类别不是 `Car` 的检测（Truck / Bus / Pedestrian / Nonmotorized_vehicle / Vehicle 等）；
4. `num_detections` 同步更新。

### 4.2 诊断

- `large_car_tracks_relabelled`
- `large_car_detections_relabelled`
- `car_only_removed`
- `classes_removed`
- `before_detections` / `after_detections`

### 4.3 注意

- 静态停放的 Car 仍然保留在 step4 输出中，只是不参与“动态区域证据”。
- step5 的 Car-only 保留为兜底，不再承担主要过滤职责。

---

## 5. step4.5a：动态区域

### 5.1 输入

- step4 的 car-only 轨迹（世界坐标）；
- step3 / step2 的静态 slot 信息（用于假高速保护）；
- clip（`CoordinateProvider`）。

### 5.2 高速证据

对每条 car-only 轨迹：

```text
path_length          >= min_track_length (默认 15m)
p90_speed            >= high_speed_threshold (默认 5 m/s)
high_speed_steps     >= min_high_speed_observations (默认 3)
相邻观测 gap         <= max_segment_gap_sec (默认 2.0s)
```

满足条件的轨迹为 **dynamic candidate track**。

### 5.3 swept box 区域

- 对每条 dynamic candidate track 的**全部观测**（包括该轨迹中的低速段）做 swept box 栅格化；
- 使用 box 的真实 `(center, size, yaw)`；
- **不使用 buffer**：`buffer_radius = 0`，等价于只用历史 `dynamic_region.py` 的 `core_mask`；
- 静态 slot footprint 不做额外膨胀；若高速轨迹本身压过某停车位，按真实扫过处理；
- 假高速保护：一条高速轨迹若有 ≥50% 观测落在静态 slot 中心 1.5m 内 → 判为跳 ID，拒绝；可选保留历史 `static_slot_hard_exclusion_radius=1.0` 检查。

### 5.4 稳定方向 30m 延伸

对每条 dynamic candidate track：

1. 找**稳定段**：连续 ≥5 帧、heading 波动 ≤12°、路径 ≥10m；
2. 取稳定段端点的 heading 作为延伸方向；
3. 在轨迹的**起点和终点**分别沿稳定方向反向 / 正向延伸 **30m**；
4. 延伸走廊宽度取该 track 的 box 宽度（不做横向 buffer）；
5. 把延伸走廊栅格化后并入动态区域；
6. 延伸后仍然只输出多边形，不做 buffer。

配置：

```text
extension_length_m = 30.0
extension_heading_tolerance_deg = 12.0
extension_min_stable_observations = 5
extension_min_stable_path_m = 10.0
```

### 5.5 输出

- `dynamic_polygons`：动态区域多边形（含延伸）；
- `core_polygons`：仅 swept box / 延伸走廊（不含 buffer）；
- `dynamic_region_mask`：栅格 mask；
- 诊断：high-speed tracks、rejected static-overlap tracks、延伸段、区域面积、被冻结轨迹数。

### 5.6 冻结判定

- **dynamic candidate track**：有高速证据；
- **frozen track**：没有高速证据，或轨迹内观测全部落在动态区域外；
- 区域内的静止车辆（parked，速度长期低于阈值）必须冻结，不能因为被其他车的 swept box 扫到就参与重跟踪；
- 重跟踪候选 = `dynamic candidate track` 的观测 + 与它有明确连续 / 继承关系的边界观测。

---

## 6. step4.5b：区域内重跟踪

### 6.1 输入 / 输出

- 输入：step4 car-only 轨迹、动态区域 mask、冻结轨迹集合；
- 输出：重跟踪后的动态轨迹（新 ID 或继承 ID）；冻结轨迹保持不变。

### 6.2 关联规则

- 使用 `ConservativeTracker` 或等价的运动关联；
- **只用运动轨迹**：center / size / timestamp / velocity / 物理连续性；
- 关联 cost 与门限中**不使用 yaw**：删除 yaw 硬门限、yaw cost 项；BEV IoU 如保留，也只作为诊断或极低权重，不使用 yaw 作为硬 gate；
- 静态 slot 逻辑在动态区域内禁用（不做 slot discovery / static assignment / topology / coordination）；
- `enforce_motion_constraints = True`；
- 保留：`reverse_step_gate`、方向连续性、速度连续性、加速度连续性、距离 / Mahalanobis 门限。

### 6.3 重跟踪范围

- 只处理重跟踪候选观测；
- 冻结轨迹的观测不进入关联，不生成新 ID；
- 边界上的继承通过下一节的 ID 继承完成。

### 6.4 与旧 ID 的关系

- 重跟踪**不直接沿用旧 step2 ID**；
- 旧 ID 只用于：
  1. 建区域 / 相位；
  2. ID 继承投票；
  3. 诊断对比。
- 动态轨迹可以完全重新发号；最终 ID 由 ID 继承决定。

---

## 7. step4.5c：方向级四相位

### 7.1 模型

使用 `region/direction_phase.py`：

1. 按稳定段 heading 聚类方向；
2. 反向方向配对成轴；
3. 每个 `方向 × movement` 独立推信号（green / red / waiting_area / unknown）；
4. 推标准四相位：`A轴直行 → A轴左转 → B轴直行 → B轴左转`；
5. 输出：
   - `direction_signal_timeline`
   - `axis_phase_timeline`
   - `track_traffic_states`（`moving / waiting_red / waiting_left_area / yielding / waiting_queue / uncertain`）

### 7.2 输入

- **重跟踪后的新动态轨迹**；
- 冻结的静态轨迹（作为方向和路口的上下文，不参与拼接主体）；
- step4 的 car-only 轨迹（用于诊断对比）。

### 7.3 规则

- 只看轨迹，不看摄像头；
- 单 clip 单路口；
- 右转始终放行；
- 掉头按左转；
- 同轴内直行 / 左转互斥；
- 不同轴独立；
- 同轴直行相位内允许许可左转 / 待转区。

### 7.4 阈值

- 先实现相位，用默认阈值跑通 5 个 clip；
- 阈值后面根据诊断再调；
- 实现者可以先定一版保守默认值，并在诊断中记录。

---

## 8. step4.5d：ID 继承 + 槽位释放 + 相位拼接

### 8.1 ID 继承优先级

对每条重跟踪后的动态轨迹：

```text
1. 静态锚 ID（从静态停车位驶出 / 驶入，且槽位释放检查通过）
2. 高速主轨迹旧 ID（按观测数投票）
3. 新 ID
```

### 8.2 多合一

- 统计最终轨迹包含的旧 step2 ID 观测数；
- 哪个旧 ID 出现最多，就继承哪个；
- 平票时依次比较：观测数、时间跨度、平均 score、旧 track 长度；
- 若最高票是静态 ID，先做槽位释放检查；不通过则顺延到下一候选或新 ID。

### 8.3 一拆多

- 默认不允许发生；
- 若发生：原 ID 给主轨迹（观测最多 / 最连续 / 物理最一致），其余发新 ID；
- 记录诊断，供人工复核。

### 8.4 同帧唯一

- 合并完成后逐帧检查 `track_id` 是否重复；
- 若重复：保留证据更强的一条（观测数 / 连续性 / score），另一条发新 ID；
- 发新 ID 后重新跑物理连续性检查。

### 8.5 物理跳变检查

- 对每个最终 ID 的连续观测：
  - 反向跳变：`reverse_step_gate`；
  - 速度 / 加速度连续性：沿用现有 m/s 门限；
  - 距离 / Mahalanobis：沿用现有门限；
- 不通过时：
  1. 先尝试切断该 ID（拆成两段）；
  2. 拆出来的后段发新 ID；
  3. 记录诊断。

### 8.6 槽位释放（重点补的逻辑）

问题：当前 `_assign_static` 会把同一个 slot 在不同时间段的静止观测都赋给同一个 slot ID；
`_coordinate_slot_motion` 只在检测到明确 departure 时才重写后续占用。
如果车 1 低速驶离 / 遮挡 / departure 漏检，车 2 之后停进 B 的静止帧可能直接沿用 B 的 slot ID。

补的逻辑：

1. 对每个静态 slot 建立 **occupancy epoch**：
   - 按时间排序静止观测；
   - 在以下位置切 epoch：长时间 gap、明确 departure、明显运动段；
   - 每个 epoch 记录 `start_ts / end_ts / owner_track_id / evidence`；
2. 第一个 epoch 继承原 slot ID（历史车 1）；
3. 后续 epoch 默认发新 ID；
4. 只有当“新车 epoch”与“旧车 departure 轨迹”满足以下条件时，才允许继承旧 slot ID：
   - 旧车最后一次静止观测 < 新车第一次观测；
   - 两者之间存在明确 departure 证据（`_radial_departure` + `_track_is_clear_motion` + bridge）；
   - 新车的运动方向 / 位置 / 尺寸与旧车离开路径一致；
   - 无同帧并发占用冲突；
5. 不通过则新车发新 ID，绝不跨车复用 slot ID。

### 8.7 相位感知拼接

在物理连续性通过的基础上，用 `track_traffic_states` 做保守合并：

- **waiting_red**：同 direction / 同进口、同一停止线附近，A 段在红灯停车、B 段在绿灯后起步 → 合并；允许 gap 由阈值控制；
- **green start**：信号由红转绿、B 段从停止状态开始运动 → 与 A 段合并；
- **yielding**：右转短停，不按红灯处理；允许跨短停合并；
- **U-turn**：按左转处理；
- **禁止**：
  - 不同 direction / 不同轴；
  - 同帧已有两条轨迹；
  - 速度方向不一致；
  - 物理跳变检查不通过；
  - 尺寸 / 类别不兼容。

### 8.8 诊断

- `id_inheritance`：每条最终轨迹的候选旧 ID、投票数、最终选择、槽位释放检查结果；
- `slot_epochs`：每个 slot 的 epoch / owner；
- `merges` / `splits` / `new_ids`；
- `duplicate_resolutions`；
- `physical_violations`；
- `phase_stitches`：每条相位合并的证据。

---

## 9. step4.5e：第二遍 box fit

### 9.1 范围

- 只对 **动态 / 重跟踪** 的轨迹或检测跑第二遍 box fit；
- 冻结静态轨迹保留 step3 的 box；
- 混合轨迹（静态段 + 动态段）只重跑动态段，静态段保留 step3 box。

### 9.2 实现方式

- 给 `geometry/car_box_fit.py` / `step3_car_box_fit.py` 增加：
  - `--only-track-ids` 或
  - `--locked-track-ids` / `--locked-detections` 或
  - 一个 `retracked` / `region` 标记；
- 第二遍只处理标记为动态的检测；
- 输出的 `box_lidar` 回填到 step4.5 JSON。

### 9.3 注意

- 混合轨迹边界前后的尺寸 / 高度平滑是分段处理的；这是静态冻结带来的已知取舍；
- 在诊断中记录每个混合轨迹的静态段 / 动态段长度。

---

## 10. step2 yaw 改造（动态-only）

### 10.1 动态 yaw

- `YawVehicleDynamicConfig.apply_motion_yaw = False`（动态段保留 detector yaw）；
- `confirmed_motion_heading` 只用于诊断 / 排除静止分支，不写 `box_lidar[6]`；
- 这样急转弯时不会被 `_motion_targets` 的 75° hold 锁住。

### 10.2 静态 yaw

- `stabilize_static_yaw` 的静态停车目标、投票、row 校正保持原样；
- 增加逐观测静止判定：
  - 只对“确认静止 run”内的观测写停车 yaw；
  - 驶离 / 转弯 / 运动段不写；
- 保留 departure cutoff 作为额外保护；
- 纯静态停车结果必须与基线逐字段一致。

### 10.3 跟踪关联

- step2 的静态 slot 匹配、静态 ID、静态 box 保持原样；
- **动态 yaw 解耦在 step4.5 的重跟踪里做**；
- step2 动态 ID 只作为区域 / 相位 / 继承候选，不作为最终 ID。

---

## 11. step5 与导出

- step5 保持：点数 / 短链终检 → Car-only 兜底 → 转 base_link → 导出；
- step4.5 输出的最终 ID / box 进入 step5；
- `run_end_to_end.py` 串入 step4.5，并增加 `--keep-intermediate`。

---

## 12. 诊断与不变量

### 12.1 静态冻结断言

- 对每条冻结轨迹：
  - 检测集合、`track_id`、`box_lidar`、`class_name`、`score`、yaw 必须与 step4 输入一致；
  - 只允许新增诊断字段；
- 若不一致 → 报错并输出差异明细。

### 12.2 同帧唯一

- 每帧 `track_id` 不得重复。

### 12.3 物理连续性

- 每个最终 ID 的连续观测必须通过：
  - 反向跳变门限；
  - 速度 / 加速度连续性；
  - 距离 / Mahalanobis 门限；
- 不通过的数量写入诊断。

### 12.4 槽位不串车

- 每个 slot epoch 只能有一个 owner；
- 不同 epoch 不得共用同一个 ID（除非有明确 departure + 继承证据）。

### 12.5 中间产物

- `--keep-intermediate` 时，把以下内容写到 `work/<clip>/`：
  - step1 raw JSON
  - step2 JSON + diagnostics
  - step3 JSON + diagnostics
  - step4 JSON + diagnostics
  - step4.5 JSON + diagnostics
  - 动态区域 PNG / 相位 PNG（可选）

---

## 13. 验收

### 13.1 数据

使用 `~/桌面/new/`：

```text
scene_crossroad_my_record_20260803_090400_clip1
scene_crossroad_my_record_20260803_090400_clip2
scene_crossroad_my_record_20260827_163412_clip1
scene_crossroad_my_record_20260827_164838_clip5_pre
scene_crossroad_my_record_20260827_164838_clip6
```

### 13.2 指标

- 静态冻结：090400_clip1/2 的冻结轨迹 labels / IDs 漂移必须为 **0**；
- 动态连续性：纯动态 / 时快时慢轨迹的 ID 段数下降，ID switch 下降；
- 急转弯 yaw：运动段 yaw 与运动 heading 的误差下降，无 >75° 的锁死；
- 动静结合：静止→运动、运动→静止的 ID 连续性；slot 不跨车复用；
- 同帧无重复 ID；
- 无物理跳变；
- 不出现“两车变一车”：合并必须有人工可复核的证据；
- 相位：红灯等待 / 绿灯起步 / 右转 yielding 的拼接正确，普通路段不误启用。

### 13.3 对比

- 每个 clip 保留 baseline 输出与新版输出；
- 逐 track 对比：ID 映射、段数、起止时间、yaw 轨迹、box 变化；
- 先看诊断，再改阈值。

---

## 14. 实施顺序

1. 本计划文档（本文件）。
2. 移植 `region/` 模块；实现 `buffer=0` + 稳定方向 30m 延伸；补单测。
3. step4：Car→Truck，然后只保留 Car；补单测。
4. step4.5a/b：区域 + 相位只读诊断；在 5 个 clip 上跑通、看诊断。
5. step4.5c：区域内重跟踪 + ID 继承 + 槽位释放；补单测。
6. step4.5d：相位感知拼接；补单测。
7. step4.5e：动态-only 第二遍 box fit。
8. step2 yaw 改造 + 静态冻结断言。
9. `run_end_to_end.py` 串入 step4.5 + `--keep-intermediate`。
10. 5 个 clip 全链路回归 + 报告。

每一步单独 commit；先保证单测，再跑全链路。

---

## 15. 暂不做 / 明确排除

- 运动遮挡导致的 ID 断（用户明确暂时不管）；
- 摄像头红绿灯颜色识别；
- 纯静态场景的主动改动（只做不退化验证）；
- 跨 clip 的全局 ID；
- 多路口场景；
- 对 `step1` 推理模型的改动。

---

## 16. 待定项（实现中再定）

- 延伸走廊宽度：默认取 track box 宽度；是否需要按 lane 方向稍微加宽（暂定 0 buffer）。
- 相位拼接的具体 gap 阈值：
  - waiting_red 最大等待 gap；
  - yielding 最大短停 gap；
  - green start 窗口；
- 区域重跟踪候选的精确集合（dynamic candidate + 边界 band 的宽度）。
- 混合轨迹第二遍 box fit 的静态段 / 动态段边界平滑策略。
- 一拆多时主轨迹的判定权重。
- `clip5_pre` 已带 label 的处理方式（用 raw / 用 pre 需要确认）。
- 方向聚类 / 停止线估计在 5 个 clip 上的阈值微调。

---

## 17. 风险

1. **静态冻结被 step2 改动破坏**：step2 在区域生成之前运行，无法天然知道哪些是静态；用静态冻结断言兜底。
2. **区域延伸把路边停车扫进来**：只用 swept box 宽度、不做横向 buffer；延伸仍可能覆盖紧贴车道的停车位，需要用静态 slot 保护 + 诊断。
3. **相位拼接把两辆车合并**：保守阈值 + 同帧唯一 + 物理连续性 + 人工可复核诊断。
4. **第二遍 box fit 导致混合轨迹尺寸跳变**：分段处理，记录边界差异。
5. **旧 ID 继承导致同帧重复**：继承后必须逐帧去重，重复时降级发新 ID。
6. **槽位释放漏检**：宁可不继承（发新 ID），也不跨车复用旧 ID。

---

## 18. 实现状态（2026-09-09）

### 已实现

- `region/` 移植并落地：`dynamic_region.py`、`region_mask.py`、`traffic_light.py`、
  `direction_phase.py`、`parking_region.py`。
- `DynamicRegionConfig`：默认 `buffer_radius=0`；静态 slot footprint 默认不挖洞；
  稳定方向两端各 30m 延伸；硬静态 slot 剔除关闭，保留 50% overlap 假高速保护。
- `step4_car_size_filter.py`：先 `Car→Truck`，再删除 Truck / 非 Car。
- `region/retrack.py`：
  - 动态候选筛选 / 区域 mask / 区域外冻结；
  - 区域内运动-only 重跟踪（`use_yaw=False`，物理门限保留）；
  - ID 继承（静态锚 > 高速旧 ID > 新 ID；多合一投票；同帧唯一；合并连续性检查）；
  - 槽位释放（显式 departure / arrival / stop_bind，或物理连续驶离/驶入）；
  - 方向级四相位（含冻结轨迹作为相位上下文）；
  - 相位感知拼接（waiting_red / green start / yielding，动态片段可并入冻结目标）；
  - 动态段第二遍 box fit，静态段保留 Step3 box；
  - `static_freeze` 断言：区域外检测逐字段不变，违反直接报错。
- `step4_5_region_phase_retrack.py` + batch。
- `run_end_to_end.py`：串入 step4.5，新增 `--keep-intermediate` / `--work-root`。
- step2 yaw：
  - `apply_motion_yaw=False`，动态段保留 detector yaw；
  - `stabilize_static_yaw` 增加逐观测静止判定，纯静态停车行为保持不变；
  - `ConservativeTracker` 新增 `use_yaw` 开关（默认 True，step4.5 用 False）。
- README / batch 脚本同步更新。

### 验证

- 单元测试：`98 tests, OK`。
- 5 个验收 clip（`~/桌面/new/`）全链路：
  - step1 推理 → step2 → step3 → step4 → step4.5 → step5；
  - `090400_clip1/2`、`163412_clip1`、`164838_clip5_pre`、`164838_clip6`
    最终 labels / IDs 与基线一致；
  - `static_freeze.passed=True`（全部 clip）；
  - `run_end_to_end.py --keep-intermediate` 单 clip 端到端跑通。

### 已知取舍

- 5 个验收 clip 上最终 labels / IDs 与基线一致，说明这些 clip 的动静切换
  在 step2 后已基本稳定；step4.5 的能力由单测覆盖（槽位释放、相位拼接、
  静态冻结）。
- 动态 detector yaw 有时比基线 motion heading 噪声大；当前按“动态 yaw
  保留 detector 原始值”的决策执行，若后续发现某类急转弯 detector yaw
  不可信，可在 step2 增加“确认转弯”平滑，而不是恢复整段 motion heading。
- 运动遮挡 gap 不主动 stitching，避免两车变一车；需要时再单独评估。

---

## 19. 队列 / movement / moving seed 对齐（2026-09-09 晚）

> 本节是第 5 / 8 节的补充和部分替代：high-speed candidate 只作为“强 seed”，
> 不再作为唯一的动态区域 / 重跟踪入口；纯静态冻结和路口等待语义改为
> moving seed + 车道队列 + 双重队列判断。

### 19.1 moving seed（低速动态目标）

moving seed 不再要求 `p90_speed >= 5 m/s`。判据：

```text
net_displacement >= 8~10m
net / path      >= 0.5
duration        >= 3s
p90_speed       >= low_speed_floor        # 暂定 1.0~1.5 m/s，待定
至少若干步持续位移 > 0.5m
```

说明：

- `path` 是累计移动距离，抖动会累加；**静止/移动判断必须用 net 位移和 center span**。
- high-speed candidate（`p90>=5m/s`、`high_speed_steps>=3`）继续保留为强 seed，
  但不再是唯一 seed。
- 示例：
  - car29：net=19.35m, span=10.62m, p90=3.25m/s → moving seed；
  - car142：net=17.89m, span=13.25m, p90=2.26m/s → moving seed；
  - id22（163412 停车区）：path=12.24m 但 net=0.028m, span=0.75m → 不是 seed。

### 19.2 纯静态冻结 / 解冻

冻结条件：

```text
net        < 1m
center span < 1m
每步位移无 >1m 的突跳
且在密集 slot / 非运动区域里没有长连续历史就“突然出现”
→ 默认冻结
```

解冻条件（hard，冻结容易解冻难）：

1. 自身有长连续轨迹：

   ```text
   net >= 8~10m
   net/path >= 0.5
   duration >= 3s
   p90 >= low_speed_floor
   ```

2. 由已确认的 moving seed / 同 lane 的 queue 带出来：

   ```text
   同 lane / queue
   + 时间不重叠
   + 纵向顺序
   + 前后位置连续
   ```

纯停车绝不参与 stitching。090400 与 163412 远处停车区的数据：

```text
090400_clip1/2 纯静止轨迹 max_span 0.11~0.51m
163412_clip1 远处停车区 ID 8~23 max_span <= 0.75m, net <= 0.58m
```

1m 的 net / span 门限是安全的。

### 19.3 动态区域与左转小尾巴

- 动态区域 = 强 seed + moving seed 的 swept box（`buffer=0`）。
- **直行 seed：稳定方向两端各延伸 30m。**
- **转弯 seed（左转 / 右转）：不向前延伸 30m，向前只用实际 swept area；
  向后按队列长度（20m）延伸。**
- 左转 seed：在末端加一段 **弧长约 5m** 的小尾巴（替代原来的 8m），
  用来覆盖待转区 / 等待左转的位置和转弯路径。
- 对向车道（如 car68）由方向 / lane gate 排除，不进入本方向 queue。

### 19.4 方向 / lane / queue

- 方向：沿用 `direction_phase` 的方向 / 轴输出。
- lane：同一方向内的 movement lane（left / straight / right）。
- queue（车队）：
  - 同一 lane 上，纵向相邻成员距离 <=20m 归入同一 queue；
  - member = 一辆车的一段时间连续片段；
  - queue state = `moving / stopped / waiting`。
- 双重队列判断：
  - 本 queue 前进 + 隔壁 queue 停止 → 隔壁等待；
  - 本 queue 停止 + 隔壁 queue 前进 → 本 queue 等待；
  - 两个都停 → 红灯 / 未知；
  - 两个都前进 → 正常通行（右转允许）。
  - 主要相邻对：`left <-> straight`、`straight <-> right`。

### 19.5 同一辆车的拼接（queue 内）

不使用单对 track 的 gap 阈值，改为：

```text
同 lane / queue
+ 时间不重叠（hard）
+ 纵向顺序
+ 前后位置连续（误差 <= 1~1.5m）
+ 中间没有其他 member 占用同一纵向位置
→ 判为同一辆车，可拼接
```

- car29 → car14：时间不重叠、位置差 ~0.5m → 拼；
- car142 → car356 → car456：互不重叠、位置连续 → 拼；
- car5 与 car29：22 帧时间重叠、位置差 ~8m → 两辆车，绝不拼。

### 19.6 movement 兼容矩阵 / 变道 / 右转

```text
straight lane ↔ straight 检测：允许
rightmost lane ↔ right 检测：允许（右转常绿，硬 gate 只允许最右侧一条）
left lane ↔ left 检测：
    仅左转相位 / 待转区许可 / queue 已确认 waiting
straight lane ↔ left 检测：禁止（硬 gate，不能直行流突然左转）
任意方向 ↔ 对向车道检测：禁止
```

变道：

- 允许同方向内变道 / 超车（例如左转前从 straight 变到 left）；
- **一次只允许跨 1 条 lane**；
- **横向位移 <= 5m**（正常车道 3.5m；超过 5m 一律禁止，作为
  “不可到对向车道”的简化硬约束）；
- 变道必须连续、平缓，不能单帧跳；
- movement 兼容项用 hard gate；变道用有条件放行 / 高成本，不直接禁死。

### 19.7 Pass1 / Pass2（都放在 step4.5）

```text
Pass 1（整体）:
  step4 car-only
  → moving seed
  → 动态区域 + 30m 直线 + 左转 5m 小尾巴
  → 方向 → movement lane → queue
  → queue 状态时间轴

Pass 2（局部）:
  → 同 lane / queue 内按 19.5 做同一辆车拼接
  → 双重 queue 判断等待 / 放行
  → movement 兼容矩阵 + 变道规则（hard gate）
  → 对向车道 / 跨方向 gate
  → 输出最终 ID；纯静态、区域外、时间重叠 track 不参与
```

### 19.8 待定 / 后续微调

- `low_speed_floor` 具体值：1.0 还是 1.5 m/s；
- moving seed 的 net 阈值：8m 还是 10m；
- queue state 的速度 / 位移门限；
- 变道连续性的具体横向速度 / heading 阈值；
- 左转 5m 弧长对应的半径：可后续查标准转弯区尺寸（用户建议可上网查标准）；
- queue 跨停止线 / 进入路口后的建模细节。

### 19.9 行驶方向过滤 / 遮挡状态 / 动态 yaw 对齐（已实现）

- **pass1 后行驶方向过滤**：
  - 对动态候选 track 的每个 detection，用它自己前后 ±2 帧的中心位置（**排除当前检测本身**）
    计算局部 driving heading；
  - 局部窗口不足时回退到整条轨迹 robust heading；
  - box yaw 与 driving heading 的轴向偏差 >60° → 标记 `direction_noise`，
    只删该帧 detection，不删整条 ID；
  - 只作用在动态候选上，纯静态停车不参与。
- **occlusion / lost 状态**（只在 step4.5 动态区域）：
  - `max_occlusion_gap = 2.0s`；
  - 丢失期间仍参与匈牙利，用 Kalman 预测 + 宽松距离/运动方向 gate，
    跳过已经膨胀的 Mahalanobis / IoU 硬门限；
  - weak seed 也启用；
  - 恢复成功保留原 ID；超过 2s 才结束；
  - ambiguous 时按 cost 分给最优。
- **pass2 后动态 yaw 对齐**：
  - 对动态/重跟踪 detection，用局部 driving heading 把 box yaw 写为运动方向；
  - 消除 180° 调转的方向歧义；
  - 静态冻结 track 的 yaw 不变。

### 19.10 最新默认参数（2026-09-10）

- `occlusion_max_gap_sec = 2.6s`：clip6 `28 -> 62` 的实际 gap 是 2.5s；
  2.6s 是刚好能接上的最小余量。
- `direction_filter_enabled = False`：行驶方向噪点过滤暂时关闭，
  因为它会给部分车辆造成 yaw 锁死；代码保留，后续设计新的过滤约束后再启用。
- `yaw_align_enabled = False`：pass2 后的动态 yaw 覆盖也暂时关闭，
  避免运动方向直接锁死 box yaw；后续可改成只做 π 等价翻转。
- 当前生效的只有：moving seed / 区域扩展 / queue 拼接 /
  槽位释放 / occlusion 2.6s / 静态冻结 / 方向车道 gate。

### 19.11 单帧噪点 / yaw 反转最终对齐（2026-09-10）

- **queue 语义保持中位数纵向位置**：区间相邻分组的尝试已回滚，
  因为它会把一条长轨迹的纵向区间拉宽，导致大量 track 链成一个大队列，
  影响已有合并。130→55 如果这样拼不上，就不强行拼。
- **单帧 overlap 噪点过滤（启用）**：
  - 每个 frame 内，非纯停车 Car 两框 `BEV IoU > 0.02`；
  - 读取原始 `lidar/lidar_top/*.bin`，分别统计两个框内点数；
  - 删除点数少的那一帧 detection，只删单帧；
  - 点数相同则不动；
  - 纯停车 track 不参与；
  - 示例：clip5 car130 vs car34，IoU=0.0382，点 14 vs 202，
    删除 car130 的 t=655.8 帧。
- **yaw 反转（启用，放在最终 ID 定下来之后）**：
  - 用整条最终运动轨迹的 robust heading；
  - 每帧计算 directed `yaw - heading`（wrap 到 [-pi, pi]）；
  - 若 track 的中位 `|directed| > 150°`，整条 track 的 yaw 加 pi；
  - 位置 / 尺寸 / 中心不动，不重跑 box fit；
  - 纯停车不参与。
- **moving fragment seed**：暂不加入；如果 130→55 在 pass2 拼不上，
  就保持分开，等后续需要再评估。

### 19.12 pass2 入口改为“区域内 Car 全进”（2026-09-10 最终确认）

- 动态区域 = 道路/行驶区，**区域内不存在纯停车**；区域内静止就是等灯/排队/让行。
- pass2 范围从原来的：
  ```text
  (旧 track ∈ seeds) 且 (detection ∈ dynamic region)
  ```
  改为：
  ```text
  Car detection 中心 ∈ dynamic region
  ```
  不再要求旧 track 是 seed。
- seed 只用于：建动态区域、方向/lane/queue，不再作为 pass2 第二道门槛。
- 区域外仍然冻结；`buffer=0` 不变。
- 效果：clip5 car130 进入 pass2，与 car55 关联，最终按观测数投票保留 55。

### 19.13 seed 职责最终收口（2026-09-10）

- seed 现在**只负责生成动态区域**（以及区域上的方向/lane/queue 建模）；
- pass2 入口已经完全是：`Car detection 中心 ∈ dynamic region`；
- `queue_stitch` 的 component 合并条件从“必须含 seed”改为：
  ```text
  component 至少有一个 detection:
      _step45_retracked == True
      或 region == "dynamic"
  ```
- 区域外 frozen track 没有 dynamic 标记，不会被 queue 合并。
- 全量 5 clip 回归：labels / IDs 与上一版 region-only 结果一致，
  说明这次只是语义收口，没有引入新的合并。
