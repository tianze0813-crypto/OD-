# 四类别 LiDAR 预标注流水线（Car / Truck / Pedestrian / Nonmotorized_vehicle）

输入：SUSTechPOINTS 原始 clip（含 `lidar/lidar_top/*.bin` 与 `transforms/`）。
输出：每帧写在 `<clip>_pre/label/<frame_id>.json`，可直接用 SUSTechPOINTS 打开。

## 常用命令（就这三条）

```bash
# 1) 单条 clip：原地跑（原目录改名成 <clip>_pre，标签写在里面）
bash hybrid_run.sh /media/moga/police/scene_001_crossroad_my_record_20260914_141401_clip4 --in-place --overwrite

# 2) 一批 clip：父目录下直接是各个 clip（自动收集，跳过 *_pre 与 lost+found）
bash hybrid_run.sh /media/moga/police --in-place --overwrite

# 3) 分场景 / step2 的嵌套结构（<scene>/step2/<clip>）：逐层遍历
DATA_ROOT=/media/moga/GEN2/0915
for scene in "$DATA_ROOT"/*/; do
  for clip in "${scene%/}"/step2/scene_*clip*/; do
    clip="${clip%/}"; name="$(basename "$clip")"
    [ -d "$clip/lidar/lidar_top" ] || continue     # 不是 clip 就跳过
    [[ "$name" == *_pre ]] && continue             # 已标注的跳过
    bash hybrid_run.sh "$clip" --in-place --overwrite
  done
done
```

跑的是什么：**BEVFusion 推理一次**（Car 与 Truck 共用这一份检测，动静态区域只算一次）
+ **VoxelNeXt 推理一次**（行人 / 非机动车）。每次运行入口会打印一行确认：

```text
[hybrid] chains=('car', 'truck', 'vru') | 车链 Car+Truck: detector=BEVFusion mode=lidar
         weights=models/bevfusion_mmdet3d_lidaronly.pth ... | VRU: detector=voxelnext
         weights=voxelnext_vru_1head2cls_epoch20.pth
```

输出：`<clip>_pre/label/*.json`（base_link，SUST 直接能开），
另有 `<clip>_pre/vehicle_pass_diagnostics.json`（车链过程数据：槽位、动态区域、挂车折叠、几何还原）。

两点注意：① 权重是 git-lfs 指针，首次先 `git lfs pull`；② `--in-place` 会把原目录改名成
`<clip>_pre`（原路径不再存在），已标注的 `*_pre` 不会被重复处理（要重跑就改回不带 `_pre` 的名字）。

> 想先看效果不动输入、要导出到别的目录、单链调试、回退跑法 —— 都在文末《附录》。

# 详细说明

## 接入链路：两次推理、两条后处理（Car+Truck 合并）

```text
推理 #1  BEVFusion（纯雷达，10 类，一次前向）
             ↓
        车链 pipeline/vehicle_pass.py（一次后处理）
             ↓ 类别合并（truck/bus/trailer/工程车 -> Truck，挂车默认并成 Truck）
             ↓ main_chain/pipeline/step_vehicle_chain.py
                  step2 共享身份跟踪（静态 slot / 动态区域证据按类别优先级仲裁，Car 优先）
                  → Car 视图 step3/step4（轿车精修）
                  → step4.5 共享动态区域 + 重跟踪 + ID 继承 + 相位拼接（只算一次）
                  → Car 视图 step5 终检 + base_link
             ↓ Truck 几何还原（只借共享那遍的 id）+ Truck 分支
                  step2_5 → step3_refinement（Truck 几何 + yaw v2）→ truck_postprocess
                  → 轨迹级 Truck/Trailer 统一 → base_link
             ↓ 按 frame_id 合成                                      -> Car               obj_id 从 0
                                                                     -> Truck             obj_id +1000
推理 #2  VoxelNeXt VRU（单头两类）pipeline/hybrid_expD_vru.py（原样不动）
                                                                     -> Pedestrian / NMV  obj_id +2000
```

要点（用户 2026-09-21 决定）：

1. **Car 与 Truck 共用同一份 BEVFusion 原始检测**（只推理一次），
   **动静态区域合起来只算一次**：静态 slot（停车位）与动态区域（高速证据 + swept box）
   的输入是 Car/Truck 的并集 —— 之前两条链各算一遍，而且各自只看得到自己那条链的类别。
2. **Car 与 Truck 冲突一律按 Car 算**（同一槽位 / 同一物理目标被 car 头与 truck 头重复表示）：
   做成类别优先级表 `tracking.CLASS_PRIORITY`，落在混合轨迹定类、同中心去重排序、
   槽位类别投票、槽位绑定顺序四处。标注侧类别人工好修，所以不做更细的仲裁。
3. 旧的「Car 被 Truck 覆盖 >= 阈值就删整条 Car 轨迹」规则**已停用**（车链内部按 Car 优先
   裁决冲突），代码保留在 `_drop_cars_covered_by_trucks`，需要时恢复。
4. **行人 / 非机动车完全不动**：VoxelNeXt 单头两类推理 + 原来的 VRU 后处理。
5. 想要旧的「Car / Truck 两条独立链」做 A/B：加 `--no-car-truck-merged`。

> 探测范围（2026-09-21 起）对齐标注 ROI：`point_cloud_range = [-54.0, -80.4, -5.0, 54.0, 20.4, 3.0]`、
> `BEV_GRID = [1440, 1344, 41]`。数据 `lidar_top` 系里 **前进 = -y、侧向 = ±x**，原来是 ±54 对称，
> 所以前方只能看到 54 m；现在前 80.4 / 后 20.4 / 侧 ±54。单一真源在两个
> `bevfusion/configs/police_bevfusion_mmdet3d*.py`，`tests/test_bevfusion_range.py` 守一致性。

> 单链调试入口仍保留：`pipeline/hybrid_expD_car.py`（Car 只做通用后处理）、
> `pipeline/hybrid_expD_truck.py`（Truck 单链），以及 main_chain 自己的
> `run_end_to_end.py`（Waymo 检测器 + Car 专属精修）。
> 旧的两链模式（main Car + VOD 五类非车）也还在：`--chains car,noncar`。

入口是 `hybrid_run.sh`，会自动探测本机 OpenPCDet 环境（默认
`~/miniconda3/envs/openpcdet`），不需要手动激活 conda。


## 2026-09-21 行为变更清单（对比改动前）

| # | 变更 | 影响 |
| --- | --- | --- |
| 1 | Car 与 Truck **合并成一条后处理**：一次身份跟踪 + 一次动态区域 | Car 的 id/框数现在由「Car+Truck 并集」决定，不再等于旧 Car 单链的数；动静态区域不再各算一份 |
| 2 | **Car 优先**：car 与 truck 冲突（同一槽位 / 同一目标被两类重复表示）一律算 Car | 混类轨迹统一、同中心去重、槽位类别投票、槽位绑定顺序四处按优先级取 Car |
| 3 | 旧的「Car 被 Truck 覆盖 ≥ 阈值就删整条 Car 轨迹」**停用** | 合并阶段不再互相删除；函数与参数保留，恢复调用即可回退 |
| 4 | 挂车（trailer）**并进 Truck**（去重/并集沿用原规则，剩余纯挂车改名 Truck） | 标注侧没有 Trailer 类；`--trailer-policy` 默认 `to-truck` |
| 5 | 工程车（construction_vehicle）**并进 Truck** | 之前是直接丢弃（映射成 `Engineering_vehicle`，不在目标类别里） |
| 6 | 检测范围扩到 **前 80.4 / 后 20.4 / 侧 ±54 m** | 原来是 ±54 对称，前方只能看 54 m；BEV 网格 `[1440, 1344, 41]` |
| 7 | 按类别的过滤门槛 | 稀疏度 Car 5 / Truck 10；短轨迹 Car 3 / Truck 4（原来是一条链一个值） |
| 8 | VRU 链**完全不动** | VoxelNeXt 单头两类推理 + 原后处理、原参数 |


## 车链（Car + Truck 合并后处理）

入口：`pipeline/vehicle_pass.py`（编排）+ `main_chain/pipeline/step_vehicle_chain.py`（共享阶段驱动）。
默认开启；`--no-car-truck-merged` 回退到原来的 Car / Truck 两条独立链。

| # | 阶段 | 在哪 | 共享 / 分叉 |
| --- | --- | --- | --- |
| 0 | 类别合并（跟踪前，时机不变） | `geometry/truck_trailer_rules.merge_classes_pre(keep_other_classes=True)` | 共享；`truck/bus/trailer/construction_vehicle -> Truck`（挂车默认并成 Truck，因为标注侧没有 Trailer） |
| 1 | 类别白名单 + 早期范围/分数过滤 | `hybrid_expD_noncar._noncar_filter/_early_range_filter` + `apply_category_score_filter` | 共享；per-class：Car 0.2 / Truck 0.2 / Trailer 0.25 |
| 2 | **身份跟踪**（类过滤 + 槽位发现 + StaticFirstTracker + 硬过滤 + 同中心去重 + 短轨迹 + 静态/整合 yaw + 轨迹级类别统一） | `main_chain` step2 | **共享（并集）**；per-class 稀疏度 Car 5 / Truck 10、短轨迹 Car 3 / Truck 4；冲突按 Car 优先 |
| 3 | 轿车框拟合 + 尺寸闸门 | `main_chain` step3/step4 | Car 只在 Car 视图上跑（Truck 几何不动）；step4 的「大 Car 改写成 Truck」默认关 |
| 4 | **动态区域 + 重跟踪 + ID 继承 + 队列/相位拼接** | `main_chain` step4.5 | **共享（并集，只算一次）** |
| 5 | Truck 几何还原 | `vehicle_pass.restore_geometry` | 只把 Truck/Trailer 的 `box_lidar` 还原成 BEVFusion 原值，保留共享那遍的 `track_id` |
| 6 | Truck 定稿：`step2_5` → `step3_refinement`（Truck 几何 + yaw v2）→ `truck_postprocess` → 轨迹级 Truck 统一 | `pipeline/hybrid_expD_truck.py` 的原阶段 | 分叉（Truck 专用参数：`static_yaw_enabled=False`、`static_rotation_enabled=False`、`yaw_impl="v2"`、`truck_merge_enabled=False`） |
| 7 | Car 终检 + base_link | `main_chain` step5 | 分叉（点 / 短轨 + Car-only + base_link） |

参数默认值在 `pipeline/vehicle_pass.py::DEFAULTS`（车链共用 + Truck 分支）与
`main_chain/pipeline/step_vehicle_chain.py::DEFAULTS`（共享阶段的 Car/Truck per-class 门槛）。
`--trailer-policy` 默认 `to-truck`（纯挂车轨迹也并成 Truck）。

实测（RTX A4000，80 帧 clip，BEVFusion 纯雷达已在缓存）：

| 场景 | 车链耗时 | 说明 |
| --- | --- | --- |
| 停车场 clip2（Car 2758 / Truck 8） | 114.6 s | 旧的 Car 链 102.6 s + Truck 链 30~38 s |
| 十字路口 clip4（Car 831 / Truck 359） | **45.8 s** | 含共享 step4.5；整链（含 BEV 预处理+推理 25 s）71 s |

调参顺序建议：先看 `vehicle_pass_diagnostics.json` 里的
`truck_trailer_class_merge` / `class_filter` / `early_class_score_filter` /
`car_step5` / `truck_geometry_restore` / `truck_branch`，再去看
`vehicle_chain/<clip>_step2_diagnostics.json`（槽位、按类别短轨迹、混类轨迹统一）
与 `<clip>_step45_diagnostics.json`（动态区域、候选轨迹、static_freeze）。


## Truck 分支（合并车链里的 Truck 定稿）

> 车链的 Truck 分支直接复用原来 Truck 单链的这几个阶段（`step2_5` → `step3_refinement` →
> `truck_postprocess` → 轨迹级类别统一）。下面的阶段说明、依赖与已知问题对车链同样成立；
> 单独跑 Truck 单链的命令见文末《附录 A》。

2026-09-20 起检测器换成 **BEVFusion**（mmdet3d 1.x 官方 20 epoch 权重），并引入 **Trailer（挂车）** 与「货车/挂车类别合并」。
只动这条链：Car（main_chain）与 VRU 链的代码路径、阈值、行为都没变
（`pipeline/hybrid_expD_noncar.py` 只多了两个默认保持旧行为的透传开关）。

### 顺序（用户 2026-09-20 明确）

```text
检测 → 范围/分数过滤（早期）→ 类别合并 → ID 跟踪 → 短轨迹/硬过滤 → 精修 → 轨迹级类别统一 → 导出 SUST
```

| # | 阶段 | 判据 / 阈值 | 产物 |
| --- | --- | --- | --- |
| 0 | 检测：`pipeline/step1_bevfusion_truck.py` → `bevfusion/scripts/infer_mmdet3d.py` | 10 类，raw 阈值 `0.1`，**z 在出口统一成框中心** | `<work>/step1_*/<clip>_raw.json` |
| 1 | 早期过滤（链内原有，位置不变） | 类别范围（Truck/Trailer）+ ROI 前 80 / 后 20 / 左右 40 + 分数 **Truck 0.2** / Trailer 0.25 | `truck_filtered_raw.json` |
| 2 | **类别合并** `geometry/truck_trailer_rules.py::merge_classes_pre`（跟踪前、lidar 系） | ①`truck`/`bus`→Truck、`trailer`→Trailer；②**重复**：挂车被货车罩住（IoM ≥ 0.70 或 IoU ≥ 0.50）→ 丢挂车、以 Truck 为准；③**有交集**：IoU ≥ 0.05 → **并集合并成一个大长 Truck**（分数取两者较大者） | `<clip>_raw_classmerge.json` |
| 3 | ID 跟踪 `pipeline/step2_identity.py` | 静态 slot 优先（Truck 链 `disable_slot_binding=True`，不用）+ 匈牙利关联；只加 `track_id`，不改几何 | `truck_step2.json` |
| 4 | 短轨迹 / 硬过滤 `step2_5`（跟踪之后） | 类别纠正 → 硬过滤第二遍（稀疏度 ≥10 点、可见性 ≥0.05、ROI/分数复核）→ **轨迹观测 < 4 帧整条删**；「静止 + 相邻帧 IoU≥0.35 + yaw 抖动 → 整条删」在 Truck 链**已关** | `truck_step2_5.json` |
| 5 | 精修 `step3` | 通用几何 + Truck 专用 `geometry/truck_postprocess.py`（见下表） | `truck_step3.json` → `truck.json` |
| 6 | **轨迹级类别统一** `truck_trailer_rules.unify_track_classes` | 同一 `obj_id` 里出现过 Truck → 该 id 的所有框都算 Truck（`--trailer-policy keep` 时纯挂车轨迹保留 Trailer，`to-truck` 则一律并成 Truck） | `truck.json` |
| 7 | 导出 SUST | 已转 base_link（只转一次），`obj_id` 从 1000 起 | `<clip>/label_truck/*.json`；整批合成后 `<clip>_pre/label/*.json` |

### 检测器两种模式（默认纯雷达）

| 模式 | 权重 | 单帧 | 单 clip 全链 | 预处理 |
| --- | --- | --- | --- | --- |
| `lidar`（默认） | `models/bevfusion_mmdet3d_lidaronly.pth`（46 MB） | 108 ms | 30~38 s | 只写 5 列 bin + infos（**不去畸变**） |
| `fusion` | `models/bevfusion_mmdet3d_lidarcam.pth`（160 MB） | 前向 191 ms + 读图 495 ms/帧 | 80~100 s | 4 路鱼眼去畸变 + bin + infos |

实测（5 条交警 clip / 400 帧，过完整条链）：纯雷达 1330 个 Truck 标签、长度 P50 8.84 / max 17.00 / >12 m 357；
C+L 1340 / 9.14 / 13.89 / 351 —— 基本持平，纯雷达快约 2.5 倍，故设为默认。

### Truck 专用后处理（`geometry/truck_postprocess.py`）

| 步 | 内容 | 默认 |
| --- | --- | --- |
| ① | yaw 旋转帧修正：与局部运动方向偏差 ≥ `15°` 的帧，yaw 改成局部运动方向（按 π 等价取与前一帧连续的分支） | 开 |
| ①b | 静止 Truck 的 yaw 平滑：yaw 取 mod π 后 5° 分箱众数投票 + 箱内圆中位 | 开 |
| ② | IoU 并集合并（把"长车被拆成两段"并成并集长框，含合并 id 的统一尺寸/朝向） | **关**（`merge_enabled=False`） |
| ③ | xy 贴合（复用 main 链 Car 的单面收缩逻辑） | **关**（`xy_fit_mode="off"`） |

②③ 关掉的原因（实测）：并集会把同一目标的重复框并成超宽框（实测 13.07×3.00 + 13.39×3.02 → 13.96×7.96），
且并集框 yaw 用点云 PCA 重算时可能被掰歪 90°；xy 贴合即使只贴合长轴，273 个 Truck 框里也有 20.5%
被沿长轴挪中心（29/273 挪 >2 m，最大 4.9 m）、25/273 长度被砍半（卡车常只看到一个端面，
face-visibility 拟合把可见点簇边缘当成了"面"）。需要时改 `TruckPostConfig` 即可回退。

> 注意区分：② 是**链内**的 Truck-Truck 并集（默认关）；第 2 步的挂车→货车并集是**类别合并**（默认开）。

### yaw 与 box 精修现状

- yaw 五层：①detector 原始 yaw 为主（`apply_motion_yaw=False`）；②`geometry_yaw_v2` —— 静止轨迹用点云 PCA 主轴
  （≥5 帧、每帧 ≥15 点、长宽比 ≥2、内点偏差 ≤15°、内点占比 ≥60%），直线运动段用运动方向（≥5 步、路径 ≥1.5 m、
  净速 ≥1.0 m/s、集中度 ≥0.75），转弯/遮挡保留 detector yaw，且不做静态方向投票；③④ 上表 ①/①b；⑤链级 `static_yaw_enabled=False`。
- box 尺寸：只有 step3 通用几何的**轨迹级分位数统一**（Truck 长轴取 0.75 分位），点云 span 只作下界不允许收缩；
  此外 Truck 的长度就是检测器输出 + 类别合并带来的并集增长。
- z：出口统一为框中心 + 通用几何的环形地面估计（clearance 4 cm，单次最多移 0.32 m）。

### 缓存与 in-place 注意事项

- Truck 链第一次处理某条 clip 时，会在 `bevfusion/data/police/<clip>/` 生成 **5 列 bin 副本 + 标定副本**
  （`fusion` 模式还会在 `bevfusion/work/undist/<clip>/` 生成去畸变图），并生成 infos；
  这些都在 `.gitignore` 里，属于**可随时删除的缓存**，删掉后重跑会自动重建。
- 缓存**自足**：标定是复制（不是软链），所以即使源 clip 被改过名，也能从缓存重建 infos；
  但 `--clip` 指向的源路径本身仍要存在才能重新预处理。
- **`--in-place` 会把源 clip 改名成 `<clip>_pre`**（标签写在里面），这是该模式的预期行为：
  下次批跑收集时会跳过 `*_pre`，所以**已标注的 clip 不会被重复处理**；
  若要重跑某条，把它改回不带 `_pre` 的名字即可（或直接对 `<clip>_pre` 用 `--chains truck` 重跑，
  因为缓存还在，只需重新推理+后处理）。

### 依赖（BEVFusion 相关）

| 依赖 | 说明 |
| --- | --- |
| conda 环境 `mmdet3d` | torch 2.1.2+cu121 / mmcv 2.1.0 / mmdet 3.3.0 / mmdet3d 1.4.0（editable） |
| mmdetection3d 源码 | 环境变量 `MMDET3D_ROOT`（默认 `~/MMDetection/mmdetection3d` 等常见位置） |
| CUDA 算子 | 首次需 `~/miniconda3/envs/mmdet3d/bin/python bevfusion/scripts/build_bev_pool.py`（编 `bev_pool_ext` + `voxel_layer`，缓存到 `~/.cache/torch_extensions`） |
| 环境变量 | `BEVFUSION_ROOT`（默认 `<project>/bevfusion`）、`BEVFUSION_PYTHON`、`BEVFUSION_TRUCK_CFG/CKPT`、`BEVFUSION_TRUCK_LIDAR_CFG/CKPT`、`BEVFUSION_DATA_ROOT` |

细节见 `bevfusion/README.md`（含 z 约定、9 列框、鱼眼去畸变等 5 个坑）。

### 已知问题

1. **并集长框会被 4 帧生命周期门槛删掉**：挂车与货车只在 3 帧贴合时（"路过贴一下"），并出的 17~18 m 长框
   自成短轨迹被删；真·铰接车（每帧都在）不受影响。要保留可用 `--short-track-max-frames 2`，或给
   `merged_from≥2` 的框豁免。实测 clip14 的并集框被跟踪器接成了同一条 id（17.4/17.7/18.0 m），只是轨迹只有 3 帧。
2. 纯挂车轨迹（与任何货车都不相交）在 5 条测试 clip 里都没能活过 4 帧门槛，所以目前输出几乎只有 Truck。
3. H800 / 云端尚未同步（云端没有 mmdet3d 环境，需要在云端建环境 + 编算子）。


## 两条后处理的权重

| 链 | 权重 | 推理配置 | raw 分数门槛 |
| --- | --- | --- | --- |
| 车链（Car + Truck）<br>**一次推理** | `models/bevfusion_mmdet3d_lidaronly.pth`（默认，纯雷达，46 MB）<br>可选 `models/bevfusion_mmdet3d_lidarcam.pth`（C+L，慢约 2.5 倍） | `bevfusion/configs/police_bevfusion_mmdet3d_lidaronly.py`<br>可选 `..._mmdet3d.py` | `--bev-raw-threshold`（默认 `0.1`）；链内 per-class：Car 0.2 / Truck 0.2 / Trailer 0.25 |
| 车链（旧单链） | Car：`main_chain/models/vn_waymo_v2_4gpu_full_epoch10.pth`（Waymo）<br>Truck：可选 `models/voxelnext_truckB_epoch15.pth` | `main_chain/models/voxelnext_v2_waymo_infer.yaml`<br>`models/voxelnext_truck_infer.yaml` | `--score-thresh` / `--truck-raw-threshold` |
| VRU | `models/voxelnext_vru_1head2cls_epoch20.pth` | `models/voxelnext_vru_infer.yaml` | `--vru-raw-threshold`（默认 `0.3`） |

`models/*.pth` 只有 133 字节 = Git LFS 指针，先 `git lfs pull` 再跑。


## 合并规则

1. 车链一次输出 Car（obj_id 0+）与 Truck（+1000），VRU 链输出 Ped/NMV（+2000），
   按 `frame_id` 拼到一起；同帧 `obj_id` 冲突时自动分配一个空闲 id。
2. **旧的「Car 被 Truck 覆盖 ≥ 阈值就删整条 Car 轨迹」规则已停用**（2026-09-21）。
   现在 Car/Truck 归属冲突在车链内部就按 **Car 优先**裁决（同一槽位、同一目标被两类
   重复表示时都算 Car），所以合并阶段不再需要互相删除。
   `_drop_cars_covered_by_trucks` 函数体与 `--car-truck-cover-threshold` 参数都保留着，
   恢复调用即可回退到旧行为。


## 批量收集规则（对应顶部第 2、3 条命令）

- **一个大目录下直接是 clip**：`hybrid_run.sh <父目录>` 会自动收集逐个处理；
  跳过 `*_pre`（已标注）、`lost+found`、没有 `lidar/lidar_top/*.bin` 的目录。
- **分场景 / step2 的嵌套结构**（`<scene>/step2/<clip>/`）：需要 shell 逐层遍历，
  命令见顶部第 3 条。
- `--in-place` 把每个 clip 就地改名成 `<clip>_pre`；想跳过 SUST 里已有输出的，可以在循环里加
  `[ -d "$SUST/${name}_pre" ] && continue`。
- 不想动输入 clip（导出到别的目录 / 只看效果）：见《附录 A》。

## 参数与默认值

默认路径（车链）的阈值分两处：

| 位置 | 管什么 |
| --- | --- |
| `pipeline/vehicle_pass.py::DEFAULTS` | 类别合并阈值、类别白名单、早期范围/分数、Truck 分支（step2_5 / step3_refinement / truck_postprocess / 挂车策略） |
| `main_chain/pipeline/step_vehicle_chain.py::DEFAULTS` | 共享阶段的门槛：per-class 分数、per-class 稀疏度（Car 5 / Truck 10）、per-class 短轨迹（Car 3 / Truck 4）、范围、可见度、Car 的 step4 是否按尺寸改类、Car 的 step5 点数/短轨 |

常用默认值：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| 检测范围 | 前 80.4 / 后 20.4 / 侧 ±54 m | `bevfusion/configs/police_bevfusion_mmdet3d*.py`（`tests/test_bevfusion_range.py` 守一致性） |
| BEV raw 阈值 | 0.1 | `--bev-raw-threshold` |
| 链内分数 | Car 0.2 / Truck 0.2 / Trailer 0.25 | `vehicle_pass.DEFAULTS["class_score_thresholds"]` |
| 稀疏度 | Car 5 / Truck 10 点 | `step_vehicle_chain.DEFAULTS["class_sparsity"]` |
| 短轨迹 | Car 3 / Truck 4 帧 | `step_vehicle_chain.DEFAULTS["class_min_lifecycle"]` |
| 挂车策略 | `to-truck` | `--trailer-policy keep` 可保留 Trailer 类 |
| Truck yaw | v2（保留 detector yaw，关静态方向投票/静态 yaw 锁） | `vehicle_pass.DEFAULTS["truck_yaw_*"]` |
| VRU | Ped/NMV 0.2、范围 60/20/40、短轨 4、行人 15 m + 20 帧、NMV 静止 15 m、yaw legacy | `pipeline/hybrid_expD_vru.py::DEFAULTS`（未改动） |

回退路径（`--no-car-truck-merged`、`--chains car,noncar`、单链调试入口）的参数表见文末《附录 A.6》。

### 入口参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `<input_root>` | 必填 | 单个 clip 目录，或包含多个 clip 的父目录 |
| `<output_root>` | `~/SUSTechPOINTS/data` | 导出模式的输出根目录；`--in-place` 时忽略 |
| `--chains` | `car,truck,vru` | 要跑的后处理，逗号分隔；默认 `car`+`truck` 合成**一条车链**跑完再跑 `vru`。可选 `car` / `truck` / `vru` / `noncar`（旧五类单链） |
| `--no-car-truck-merged` | 关（即默认合并） | 回退到 Car / Truck 两条独立链，A/B 用 |
| `--car-truck-cover-threshold` | `0.5` | **已停用**：Car/Truck 冲突改由车链内部的 Car 优先裁决（函数体保留） |
| `--keep-chain-labels` | 关 | 额外把 `label_car/` `label_truck/` `label_vru/` 写进输出 clip |
| `--in-place` | 关 | 原地端到端：输入 clip 改名 `<clip>_pre` |
| `--export-sust` | **开** | 导出到 `<output_root>/<clip>_pre`；与 `--in-place` 互斥 |
| `--no-export-sust` | 关 | 只跑链路，不落盘 |
| `--output-tag` | 空 | 在 `<clip>` 和 `_pre` 之间加标签 |
| `--output-suffix` | 空 | 直接指定输出后缀（如 `_vehicle`），输出名不套 `_pre` |
| `--link-only` | 关 | 输出目录只软链 image/lidar/transforms，不复制数据 |
| `--overwrite` | 关 | 输出已存在时先删除再生成 |
| `--python` | 自动探测 | 指定运行/后处理 Python |
| `--skip-install` | 关 | 只允许使用已有 CUDA/OpenPCDet 环境 |
| `--drop-vis-below` | `0.05` | 非车链可见度硬过滤阈值（Truck/VRU 链内部固定 0.05） |

### Truck 相关参数（车链也用；`voxelnext` 那两行只在回退时有效）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--truck-detector` | `bevfusion` | Truck 检测器：`bevfusion`（BEVFusion）或 `voxelnext`（旧 VoxelNeXt truckB） |
| `--truck-detector-mode` | `lidar` | BEVFusion 模式：`lidar`（纯雷达，46 MB，快）或 `fusion`（C+L，160 MB，读 4 路相机图） |
| `--truck-ckpt` / `--truck-cfg` | `models/voxelnext_truckB_epoch15.pth` / `models/voxelnext_truck_infer.yaml` | 仅 `--truck-detector voxelnext` 时使用 |
| `--truck-raw-threshold` | `0.1` | 检测器 raw 分数门槛（链内类别阈值另外把关） |
| `--trailer-score-threshold` | `0.25` | Trailer 类别分数阈值（Truck 固定 0.2） |
| `--trailer-dup-iom` / `--trailer-dup-iou` | `0.70` / `0.50` | 判"挂车是货车的重复框"的 IoM / IoU 门限 |
| `--trailer-merge-iou` | `0.05` | 挂车与货车有交集判据（命中就并集成大长 Truck） |
| `--trailer-policy` | **`to-truck`** | `to-truck`（默认，标注侧没有 Trailer）：一律并成 Truck；`keep`：纯挂车轨迹保留 Trailer |
| `--no-trailer-rules` | 关 | 关掉类别合并（去重/并集/轨迹统一），回退到只有 Truck 的旧行为 |

其余（范围 80/20/40、Truck 0.2 / Trailer 0.25、短轨迹 4、yaw v2、Truck 后处理各开关）
是 `pipeline/hybrid_expD_truck.py::DEFAULTS` 的模块默认值，改那里即可。

### VRU 链参数（透出部分）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--vru-ckpt` | `models/voxelnext_vru_1head2cls_epoch20.pth` | VRU 权重 |
| `--vru-cfg` | `models/voxelnext_vru_infer.yaml` | 推理配置 |
| `--vru-raw-threshold` | `0.3` | VRU raw 推理分数门槛 |

其余（范围 60/20/40、Ped/NMV 阈值 0.2/0.2、短轨迹 4、行人 15m、NMV 静止 15m、
行人生命周期 20 帧）是 `pipeline/hybrid_expD_vru.py::DEFAULTS` 的模块默认值。

`pedestrian_min_frames` 目前只在 VRU 链自己的 CLI 上透出
（`hybrid_expD_vru.py --pedestrian-min-frames`，`0` = 关闭）；hybrid 入口暂未加对应参数，
要改就动 `DEFAULTS`。

### 实测耗时

本机 RTX A4000 16GB、不与其他大任务并行时，单个 80 帧 clip（含 BEVFusion 预处理 + 推理）：

| 场景 | 车链（Car+Truck 一次） | VRU 链 | 合计 |
| --- | --- | --- | --- |
| clip4 十字路口（Car 830 / Truck 380） | 46.2 s | 23.6 s | **93.0 s** |
| clip5 十字路口（Car 561 / Truck 242） | 32.2 s | 24.3 s | **78.1 s** |
| clip14 十字路口（Car 365 / Truck 401） | 26.4 s | 21.7 s | **70.6 s** |
| clip17 十字路口（Car 529 / Truck 210） | 25.3 s | 40.6 s | **90.1 s** |

对照：改动前是 Car 链 102.6 s + Truck 链 30~38 s（各自一遍推理与后处理）≈ 166 s/clip，
所以现在每 clip 省 **44%~58%**。每次运行入口都会打印一行实测值：

```text
[hybrid] <clip>: 计时 vehicle=32.2s, car=32.2s, truck=0.0s, vru=24.3s, merge=0.0s,
                export=0.1s, total_without_export=56.5s, total=78.1s
```

（`car` 与 `vehicle` 是同一个数：车链一次跑完同时产出 Car 与 Truck，`truck=0`。
车链内部 `step_vehicle_chain` 的诊断会写在输出目录的 `vehicle_pass_diagnostics.json`。）


## VRU 链（`pipeline/hybrid_expD_vru.py`）

- 保留 `Pedestrian` + `Nonmotorized_vehicle`
- 范围：前 60 / 后 20 / 左右 40 m；行人额外 15 m 半径
- 分数阈值：Pedestrian `0.2`、Nonmotorized_vehicle `0.2`；短轨迹过滤 4 帧（链级，`<=` 语义）
- yaw：沿用旧版 `legacy`
- NMV 静止过滤：world 系首尾净位移 ≤ 15 m 的轨迹丢弃（行人不过滤）
- 行人生命周期过滤：**只对行人**，观测帧数 < 20 的轨迹整条删除
  （`pedestrian_min_frames=20`，严格 `<`；非机动车仍走链级 4 帧规则，输出不受影响）
- `obj_id` 从 2000 起

> 原「世界系一排行人共线（同一帧内 ≥ 6 个、垂距 ≤ 1.0 m）整排删除」规则已废弃：
> 它区分不了真行人与噪声——车旁行人的世界轨迹与车辆路径共线，而真正的绿篱排 id 太少触发不了。
> 行人噪点（以短命碎片为主）改由上面的 `pedestrian_min_frames` 处理。


## 目录

```text
hybrid_run.sh           入口（两条后处理：车链 Car+Truck -> VRU）
main_chain/             最新 OD-main-0909 快照（Car 链路 + region 动静态区域）
                        pipeline/step_vehicle_chain.py  车链共享阶段驱动（step2 + step3/4/4.5/5）
                        pipeline/step1_bevfusion.py      BEVFusion 检测入口
pipeline/               各链主体：
                        vehicle_pass.py   车链编排（Car+Truck 一次后处理，默认路径）
                        hybrid_expD_vru.py            VRU 链（VoxelNeXt 单头两类，原样不动）
                        hybrid_expD_noncar.py         非车后处理主体（step2_5 / step3 等阶段）
                        hybrid_expD_truck.py          Truck 单链（调试/回退）+ Truck 分支阶段
                        hybrid_expD_car.py            Car 通用后处理单链（调试/回退）
                        hybrid_main_car.py            main_chain Car 链调用
                        hybrid_merge.py               标签合并
bevfusion/              Truck 链的 BEVFusion 工具箱（配置 + prep/infer/评测脚本，见其 README）
geometry/truck_trailer_rules.py  货车/挂车类别合并（去重 / 并集 / 轨迹级类别统一）
geometry/truck_postprocess.py   Truck 专用后处理（yaw 修正 / 静止平滑 / 可选并集与贴合）
geometry_yaw_v2/        Truck 链用的新版 yaw（直线运动方向 / 静态方向 / 动态 yaw）
classification/         类别归一化与 track 投票
filtering/              可见度、硬过滤、五类输出、低置信类别过滤
tracking/               跟踪、坐标变换、SUST label 映射
geometry/               yaw、Car 几何、Truck/NMV 精修
inference/              OpenPCDet LiDAR 推理
models/                 各链的配置与 checkpoint
scripts/                入口脚本、merge_two_chains（旧两链合成）、remerge_truck_car（只重跑 Truck）、
                        run_bevfusion_test_chains.py（BEV 原始检测 → 测试三支标签合成）
tests/                  单元测试
```

测试：

```bash
# 车链（混合链路侧）
~/miniconda3/envs/openpcdet/bin/python -m unittest discover -s tests -t .
# main_chain（Car 链 + 动静态区域 + 车链共享驱动）
cd main_chain && ~/miniconda3/envs/openpcdet/bin/python -m unittest discover -s tests -t .
```

车链相关测试：`tests/test_bevfusion_range.py`（检测范围配置一致性）、
`tests/test_class_policy.py`（挂车/工程车并 Truck、覆盖规则停用）、
`tests/test_vehicle_pass.py`（类别视图切分、Truck 几何还原）、
`main_chain/tests/test_class_priority.py`（Car 优先的四处落点）。

## 附录 A：导出到别的目录 / 其他运行模式

### A.1 导出到指定目录（原始 clip 保留不动）

结果写到 `<output_root>/<clip>_pre`；`<output_root>` 省略时默认 `~/SUSTechPOINTS/data`。

```bash
bash hybrid_run.sh <clip> <output_root> --overwrite
```

例如：

```bash
bash hybrid_run.sh \
  /media/moga/police/0903/nonmotor_lane_my_record_20260903_100541/step2/scene_nonmotor_lane_my_record_20260903_100541_clip43 \
  /home/moga/桌面/SUSTechPOINTS/data --overwrite
```

输出：

```text
<output_root>/
└── <clip>_pre/
    ├── 原始 clip 数据（lidar / image / transforms）
    └── label/<frame_id>.json
```

- `--output-tag truckb_e15`：输出名变成 `<clip>_truckb_e15_pre`，方便保留多组对比。
- `--output-suffix _vehicle`：输出名变成 `<clip>_vehicle`（不套 `_pre`），适合测试输出。
- `--link-only`：输出目录里 image/lidar/transforms 只放软链，不复制数据（快）。
- `--keep-chain-labels`：额外写 `label_car/`、`label_truck/`、`label_vru/` 便于按链排查。

### A.2 仅调试：只跑链路、不落盘

```bash
bash hybrid_run.sh <clip> /tmp/unused --no-export-sust --overwrite
```

只打印统计信息，临时 JSON 跑完自动删除，不产生 `<clip>_pre`，也不改输入。

### A.3 选链

```bash
--chains car,truck,vru   # 默认：车链（Car+Truck 合并后处理）+ VRU
--chains car,truck       # 只车链
--chains vru             # 只行人/非机动车
```

### A.4 复用已经跑好的 BEVFusion 原始检测（不重新推理）

```bash
bash hybrid_run.sh <input_root> <output_root> --bev-raw-dir <存放 <clip名>_raw.json 的目录> ...
```


## 附录 B：调试

### B.1 单条链单独跑（排查用）

```bash
# 车链（Car+Truck 一次后处理）—— --export 只把两支写进 clip 的 label_car/ label_truck/，
# 合成后的 label/ 由整批入口（hybrid_run.sh）负责；诊断在 <work-root>/vehicle_pass_diagnostics.json
python pipeline/vehicle_pass.py --raw-json <bev_raw.json> --clip <clip> \
    --work-root /tmp/vehicle --diagnostics /tmp/vehicle/diag.json --export

# 只跑 BEVFusion 推理，拿 raw json（纯雷达）
python pipeline/step1_bevfusion_truck.py --clip <clip> --work-root work/step1 --mode lidar --score-thresh 0.1

# VRU 单链（先用 models/voxelnext_vru_infer.yaml + voxelnext_vru_1head2cls_epoch20.pth 出 raw）
python pipeline/hybrid_expD_vru.py --raw-json <vru_raw.json> --clip <clip> --out-json work/vru.json --label-subdir label_vru
```

### B.2 main_chain 与本项目自己的检测入口

```bash
# main_chain 自己的 Car 链（Waymo 权重 + Car 专属精修 Step3/Step4/Step4.5/Step5）
cd main_chain && python run_end_to_end.py --clip <clip> --export-sust

# 只跑 BEVFusion 推理拿 raw json（车链/单链都用得上）
python pipeline/step1_bevfusion_truck.py --clip <clip> --work-root work/step1 --mode lidar --score-thresh 0.1
```

### B.3 跑测试

```bash
~/miniconda3/envs/openpcdet/bin/python -m unittest discover -s tests -t .            # 混合链路侧
cd main_chain && ~/miniconda3/envs/openpcdet/bin/python -m unittest discover -s tests -t .
```

### B.4 跑完先看哪里

```text
<clip>_pre/
├── label/                        ← 合并后的标签（Car 1+ / Truck +1000 / PedNMV +2000）
├── label_car / label_truck / label_vru   ← 分链标签（--keep-chain-labels）
├── vehicle_pass_diagnostics.json ← 车链过程数据：shared_region（槽位数/类别、动态区域面积、
│                                   候选轨迹、可重跟踪检测数、static_freeze）、挂车折叠数、
│                                   Truck 几何还原数、Car step5 统计
├── lidar / image / transforms
└── （每次运行入口也会打印一行计时：vehicle / vru / merge / export / total）
```

常见问题：权重只有 133 字节 → `git lfs pull`；`<clip>` 不见了 → `--in-place` 已改名成
`<clip>_pre`；已标注的 `*_pre` 不会被重复处理（要重跑就改回不带 `_pre` 的名字）。



## 附录 C：回退路径与单链调试（**本分支不推荐**）

本分支的正式路径就是上面那条车链；下面这些旧跑法代码里都还在，只用于对比 / 排查。
要正式跑旧链路，建议切到对应分支（例如 `hybrid-main-car-expd-noncar`），别混着当生产用。

### C.1 回退跑法

```bash
bash hybrid_run.sh <input_root> <output_root> --no-car-truck-merged --overwrite
#   车链拆回 Car / Truck 两条独立链：Car 走 main_chain（Waymo Car 头 + Step4.5），
#   Truck 走 BEVFusion 单链；两条链各自独立跟踪，动静态区域各算一份

bash hybrid_run.sh <input_root> <output_root> --chains car,noncar --overwrite
#   更旧的「main Car + VOD 五类非车」两链（见附录 B）

--car-detector waymo             # 显式指定 Waymo car 头（默认 auto：车链可用就用 BEVFusion，否则 Waymo）
--car-pipeline hybrid            # Car 走通用后处理单链（pipeline/hybrid_expD_car.py）
--car-truck-cover-threshold 0.5  # 已停用：Car 被 Truck 覆盖就删整条 Car 轨迹（函数体保留）
```

### C.2 Truck 单链的运行命令

```bash
# 单条 clip（默认纯雷达；结果写 <clip>/label_truck/）
python pipeline/hybrid_expD_truck.py --clip <clip> --out-json work/truck.json \
    --detector bevfusion --detector-mode lidar --label-subdir label_truck

# 整批（车链 Car+Truck -> VRU，合成 <clip>_pre/label）
bash hybrid_run.sh <input_root> --in-place --overwrite
python scripts/run_hybrid_prelabel.py <input_root> <output_root> --chains truck

# 回退
... --truck-detector voxelnext      # 旧 VoxelNeXt truckB 权重
... --no-trailer-rules              # 只保留 Truck，不做挂车去重/并集/统一
... --truck-detector-mode fusion    # 改用 C+L 权重
```

### C.3 只改了 Truck 时：复用 Car/VRU 标签重跑

> 仅回退路径适用：默认路径下 Truck 的 id 来自与 Car 共享的那一遍跟踪与动态区域，
> 单独重跑 Truck 链对不上号（要么整条车链重跑，要么 `--no-car-truck-merged`）。

> 【2026-09-21】默认路径下 Truck 的 id 来自与 Car 共享的那一遍跟踪与动态区域，
> **单独重跑 Truck 链已经对不上号**：要么整条车链重跑（默认），要么加
> `--no-car-truck-merged` 回到两条独立链再用本脚本。

```bash
python scripts/remerge_truck_car.py --output-root <含 <clip>_pre 的目录> [--write]
```

读 `<clip>_pre/label/` 里已有的 Car 与 `label_vru/`，**只重跑 Truck 链**（原始推理 + 链），
再按上面的合并规则重写 `label/` 和 `label_truck/`。改 Truck 参数或合并规则时用它，
可以省掉最慢的 Car 链。

### C.4 旧的 Car 单链说明（`pipeline/hybrid_expD_car.py`）

2026-09-21 起 Car 有两个可选实现、检测器也有两档，**后处理语义都产出 SUST 可读的 base_link 标签**：

| `--car-pipeline` | 实现 | 检测器 `--car-detector` | 说明 |
| --- | --- | --- | --- |
| `main_chain`（默认） | `main_chain/`（= OD-main-0909 快照） | `waymo`（默认）/ `bevfusion` | Waymo Car + Step4.5；`bevfusion` 时注入 BEVFusion 的 car 头结果（也可在 main_chain 里直接 `run_end_to_end.py --detector bevfusion` 自己跑 BEVFusion step1） |
| `hybrid` | `pipeline/hybrid_expD_car.py` | `bevfusion` / `waymo` | **只做通用后处理**：类别范围过滤 → 硬过滤 → 静态优先+匈牙利跟踪 → 短轨迹/硬过滤第二遍 → 通用几何 → base_link。不跑 main_chain 的 Car 专属精修（step3_car_box_fit / step4 size filter / step4.5 region retrack / step5） |

`hybrid_expD_car.py` 默认参数（可用命令行覆盖）：Car 分数 **0.2** / 范围 前 80-后 20-侧 40 /
稀疏度 ≤**5** 点 / 可见度 0.05 / 短轨迹 **3** 帧 / yaw v2（允许静态 slot 与静态 yaw 稳定）；
`obj_id` 从 1 起（合成三链时 orchestrator 会分段 0+/1000+/2000+）。

```bash
# 单条 clip（只做通用后处理，检测器用 BEVFusion 的 car 头）
python pipeline/hybrid_expD_car.py --detector bevfusion --clip <clip> \
    --out-json work/car.json --label-subdir label_car --link-dir <输出目录>
# 整批里切换：--car-pipeline hybrid --car-detector bevfusion（三链共用同一份 BEVFusion 原始检测）
python scripts/run_hybrid_prelabel.py <input_root> <output_root> --chains car,truck,vru \
    --car-pipeline hybrid --car-detector bevfusion --vru-detector bevfusion
```

**为什么还需要 `hybrid` 这条 / main_chain 做了哪些对齐**（停车场场景 clip5，BEVFusion 纯雷达 raw @0.2）：

| 配置 | Car 框数 | 说明 |
| --- | --- | --- |
| BEV 原始检测 | 1910 | 阈值 0.2 |
| main_chain（原 Waymo 门限） | 203 | step2 分数 0.3 / 稀疏度 10 点 / 生命周期 4 帧 + 白名单按**原始字符串**比（`car` 不匹配 `Car`，整批删） |
| main_chain + 对齐门限（现在） | **1552** | BEV 模式自动用 0.2 / 5 点 / 3 帧，白名单别名感知（已在 OD-main-0909 改并同步） |
| hybrid Car 链 | **1681** | 只做通用后处理；差额（~130）来自 main 额外的 step4 尺寸过滤 / step4.5 重跟踪 / step5 运动过滤 |

对齐清单（BEV 模式下 main_chain 与测试链的参数现在一致）：

| 参数 | Waymo 模式 | BEV 模式（= 测试链） |
| --- | --- | --- |
| step2 分数阈值 | 0.3 | 0.2 |
| step2 稀疏度 | ≤10 点 | ≤5 点 |
| step2 生命周期 | 4 帧 | 3 帧 |
| 类名归一 | 原样比 | 原样比 + `CLASS_MAP` casefold 归一 |
| 范围 / 可见度 | 80-20-40 / 0.05 | 同 |

main_chain 的硬过滤用**原始类别字符串**比对白名单、且分数/点数门限是给 Waymo 头调的，
换成 BEVFusion 的 car 头会削掉 90%+；`hybrid` 这条链走 `canonical_class_name` 归一，
分数阈值 0.2，保留率 85~90%。

> 测试用的一键脚本：`scripts/run_bevfusion_test_chains.py`（原始检测 → Car(hybrid)+Truck+VRU 合成一份标签）。

### C.5 回退路径的参数表

#### Car 链参数（`--car-pipeline hybrid`）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--car-pipeline` | `main_chain` | `main_chain`（Waymo Car + Step4.5）或 `hybrid`（`pipeline/hybrid_expD_car.py`，只做通用后处理） |
| `--car-detector` | `waymo` | `waymo` 或 `bevfusion`（BEVFusion 的 car 头；hybrid 链默认建议 bevfusion） |
| `--car-score-threshold` | `0.2` | `--car-pipeline hybrid` 时的 Car 分数阈值 |

#### main_chain Car 参数（OD-main-0909 默认）

`main_chain/run_end_to_end.py` 使用以下固定默认值，目前没有全部透出成 hybrid 入口参数：

| 环节 | 参数 | 默认值 |
| --- | --- | --- |
| Step1 推理 | `--score-thresh` | `0.3` |
| Step1 可见度 | `--drop-vis-below` | `0.05` |
| Step5 终检 | `--sparsity-max-points` | `5` |
| Step5 终检 | `--short-track-max-frames` | `3` |
| Step4.5 动态区域 | `--extension-length-m` | `30.0` |
| Step4.5 相位拼接 | `--phase-merge-max-gap-sec` | `30.0` |
| Step4.5 右转 yielding | `--yielding-max-gap-sec` | `6.0` |

## 附录 D：旧五类非车链（`--chains car,noncar`，回退）

`--chains car,noncar` 时回到旧的"main Car + VOD 五类非车"两链模式，合并走
`pipeline/hybrid_merge.py::merge_label_frames`（含 Car/非车 互斥吸收）。

```text
main_chain/（Waymo Car + Step4.5）
  -> VOD e12 非车四类（Ped / NMV / Truck / Bus）
  -> 非车链第一步把 Bus 折叠成 Truck
  -> 按 frame_id 合并为四类
```

> Bus 的折叠发生在非车链最早的 `early_non_car_class_filter`
> （`pipeline/hybrid_expD_noncar.py::_noncar_filter`）里，紧跟 raw 推理输出，
> 早于打分过滤 / 跟踪 / 类别投票 / 几何精修，最终标签中不会出现 `Bus`。

### 旧非车链参数（VOD e12）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--noncar-ckpt` | `models/vod_2cls_ft_e12.pth` | 非车权重；可切 `vod_2cls_ft_e25.pth` |
| `--noncar-cfg` | `models/voxelnext_fiveclass_nuscenes_infer.yaml` | 推理配置 |
| `--noncar-raw-threshold` | `0.2`（= 类别阈值最小值） | 非车 raw 推理分数门槛 |
| `--truck-score-threshold` | `0.4` | Truck 最终分数阈值 |
| `--bus-score-threshold` | `0.4` | 非车链已把 Bus 折叠为 Truck，此参数不再生效（保留仅为兼容） |
| `--pedestrian-score-threshold` | `0.2` | Pedestrian 最终分数阈值 |
| `--nonmotorized-score-threshold` | `0.2` | NMV 最终分数阈值 |
| `--score-threshold` | 未设置 | 一次性覆盖以上四个类别阈值；raw 门槛仍取最小值 |
| `--pedestrian-max-distance` | `15.0` | Pedestrian 最大距离（米） |
| `--nonmotorized-max-distance` | `60.0` | NMV 最大距离（米） |
| `--sparsity-max-points` | `10` | 非车链点数过滤阈值 |
| `--short-track-max-frames` | `4` | 非车链短轨迹过滤帧数 |
| `--nonmotorized-min-net-displacement` | `15.0` | 只保留 world 系首尾净位移大于该值的 NMV 轨迹 |

默认非车权重 `models/vod_2cls_ft_e12.pth`：VOD 2 类微调，冻结骨干与
Car/Truck/Bus 头，只训练 Pedestrian / Nonmotorized_vehicle 头。
旧模式合并阶段另有一条规则：Truck 与 Car 单帧 BEV 重叠面积 / Car 面积 > 0.5 时，
只删该帧的 Car（三链模式下换成"删整条 Car 轨迹"，见《合并规则》）。

---
