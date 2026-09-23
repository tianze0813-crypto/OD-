# 四类别 LiDAR 预标注流水线（Car / Truck / Pedestrian / Nonmotorized_vehicle）

输入：SUSTechPOINTS 原始 clip（含 `lidar/lidar_top/*.bin` 与 `transforms/`）。
输出：每帧写在 `<clip>_pre/label/<frame_id>.json`（**base_link** 系，obj_id 已分段），
可直接用 SUSTechPOINTS 打开。

一条链路：**BEVFusion 推理一次**（Car 与 Truck 共用这份检测，动静态区域只算一次）
+ **VoxelNeXt 推理一次**（行人 / 非机动车），最后按 frame_id 合成一份 `label/`。

---

## 常用命令（就这三条）

> `--include-pre` 是**命令行开关，默认关**（不加它 = 批量遍历父目录时跳过所有 `*_pre`，只跑没标注过的）。
> 下面第 2、3 条按**带 `--include-pre`** 的写法给 —— 平时最常见的场景是「连已标注过的一起重跑」；
> 带上它时 `<clip>_pre` 会被**就地覆盖重跑**（目录名不变、只重写 `label/`，不会生成 `<clip>_pre_pre`）。
> 第 1 条是**直接点名**单个 clip：传进来的目录本身带 `_pre` 时不用加 `--include-pre`，一样就地覆盖。
> 新 clip 第一次跑就把 `--include-pre` 删掉。

```bash
# 1) 单条 clip：原地跑（原目录改名成 <clip>_pre，标签写在里面）
#    传进来的目录本身带 _pre 时 = 就地覆盖重跑（要加 --overwrite）
bash hybrid_run.sh /media/moga/police/scene_001_crossroad_my_record_20260914_141401_clip4 --in-place --overwrite
bash hybrid_run.sh /media/moga/police/scene_001_crossroad_my_record_20260914_141401_clip4_pre --in-place --overwrite

# 2) 一批 clip：父目录下直接是各个 clip（带 --include-pre：已标注的也一起重跑）
bash hybrid_run.sh /media/moga/police/1111 --in-place --overwrite --include-pre
#    只跑新 clip（跳过已标注的）= 把 --include-pre 去掉：
#    bash hybrid_run.sh /media/moga/police/1111 --in-place --overwrite

# 3) 分场景 / step2 的嵌套结构（<scene>/step2/<clip>）：逐层遍历
DATA_ROOT=/media/moga/police/0922
for scene in "$DATA_ROOT"/*/; do
  for clip in "${scene%/}"/step2/scene_*clip*/; do
    clip="${clip%/}"; name="$(basename "$clip")"
    [ -d "$clip/lidar/lidar_top" ] || continue     # 不是 clip 就跳过
    bash hybrid_run.sh "$clip" --in-place --overwrite --include-pre
    # 只想跑新 clip：把上面一行的 --include-pre 去掉，并把下面这行注释解开
    # [[ "$name" == *_pre ]] && continue
  done
done
```

**注意 `_collect_clips` 只扫 `input_root` 的直接子目录**，不会递归下钻：`input_root` 必须是
「里面直接放着 clip」的那一层（如上面的 `1111/` 或 `<scene>/step2/`）。

入口每次会打印一行确认实际跑的是哪个检测器：

```text
[hybrid] chains=('car', 'truck', 'vru') | 车链 Car+Truck: detector=BEVFusion mode=lidar
         weights=models/bevfusion_mmdet3d_lidaronly.pth ... | VRU: detector=voxelnext
         weights=voxelnext_vru_1head2cls_epoch20.pth
```

## 输出

| 产物 | 何时写 | 说明 |
| --- | --- | --- |
| `<clip>_pre/label/<frame_id>.json` | 总是 | SUST 标签，base_link 系 |
| `<clip>_pre/vehicle_pass_diagnostics.json` | 加 `--keep-vehicle-diagnostics` | 车链全过程诊断（槽位 / 动态区域 / 挂车折叠 / 几何还原 / yaw），调试用 |
| `<clip>_pre/label_car`、`label_truck`、`label_vru` | 加 `--keep-chain-labels` | 分链标签，便于对比 |

`obj_id` 分段：**Car `0+`、Truck `1000+`、Pedestrian / Nonmotorized_vehicle `2000+`**。
同一帧内 id 冲突会自动换空闲 id；Car 与 Truck 重叠时**按 Car 算**（2026-09-21 用户决定，
旧的「Car 被 Truck 覆盖 ≥ 阈值就删整条 Car」规则已停用）。

---

## 一次运行到底执行了什么

```text
hybrid_run.sh                                   挑 openpcdet 环境的 python，exec 入口脚本
└─ scripts/run_hybrid_prelabel.py               编排：推理一次 -> 车链 -> VRU -> 合并 -> 写 label/
   │
   ├─ pipeline/step1_bevfusion_truck.py                    【子进程，mmdet3d 环境】
   │   ├─ bevfusion/scripts/prep_data.py           5 列 bin + 去畸变（mode=lidar 时跳过图像）
   │   ├─ bevfusion/scripts/mmdet3d_prep.py        middle-format infos
   │   └─ bevfusion/scripts/infer_mmdet3d.py       BEVFusion 推理（10 类）
   │        + bevfusion/scripts/build_bev_pool.py  （被裸名 import）
   │        配置/权重：configs/police_bevfusion_mmdet3d[_lidaronly].py
   │                  models/bevfusion_mmdet3d_lidaronly.pth（默认）/ _lidarcam.pth
   │
   ├─ pipeline/vehicle_pass.py                            合并车链（Car + Truck）
   │   ├─ 类别合并（跟踪前） geometry/truck_trailer_rules.merge_classes_pre
   │   ├─ 类别白名单 + 早期范围/分数过滤
   │   ├─ 【子进程】main_chain/pipeline/step_vehicle_chain.py     ← Car 的共享阶段
   │   │     step2（共享）  类过滤 + StaticFirst 身份跟踪 + 硬过滤 + 去重 + 短轨迹
   │   │                  + 静态 yaw 稳定 + 整合 yaw + 轨迹级类别统一
   │   │     step3（Car）   轿车框拟合（XY + 车顶/地面 Z）
   │   │     step4（Car）   轿车尺寸闸门
   │   │     step4.5（共享）动态区域 + 区域 mask + 运动-only 重跟踪 + ID 继承
   │   │                  + 队列/相位拼接 + 动态段第二遍 box fit + π 等价翻转
   │   │     step5（Car）   终检（点数/短链）+ 转 base_link
   │   ├─ Truck 几何还原      vehicle_pass.restore_geometry（只还原 Truck/Trailer 的 box）
   │   └─ Truck 分支         step2_5 -> step3_refinement（yaw v2 + Truck 几何）
   │                        -> geometry/truck_postprocess -> 轨迹级类别统一 -> base_link
   │
   ├─ pipeline/hybrid_expD_vru.py                          VRU 链（VoxelNeXt）
   │   └─ pipeline/step1_lidar_inference.py 【子进程】models/voxelnext_vru_1head2cls_epoch20.pth
   │
   └─ scripts/run_hybrid_prelabel.py::_merge_chain_labels  三条链按 frame_id 合成一份 label/
```

> ⚠️ `main_chain/` 不是历史残留：Car 的 step2~step5 就是那份代码（子进程、独立 `sys.path`）。
> 它和仓库根目录下的 `geometry / filtering / tracking / classification` **已经分叉**，两棵都要留。
> 详见 `main_chain/README.md`。

---

## 目录结构

```text
hybrid_run.sh                    入口 shell（挑 python）
scripts/run_hybrid_prelabel.py   编排入口
pipeline/                        编排 + Truck/VRU 链
  vehicle_pass.py                合并车链（Car+Truck）编排        ← 最常改
  step1_bevfusion_truck.py       BEVFusion 推理（三链共用一份 raw）
  step1_lidar_inference.py       VoxelNeXt 推理（VRU / Truck 回退）
  hybrid_expD_vru.py             VRU 链
  hybrid_expD_truck.py           Truck 单链（回退；也给车链提供类别词表）
  hybrid_expD_noncar.py          旧五类非车链（--chains noncar）；车链复用它的过滤器
  step2_5_class_correction.py    类别归一 + 二次硬过滤
  step3_refinement.py            Truck 几何 + yaw v2
  step2_identity.py / truck_car_tracking.py   旧五类链的跟踪/合并（被 noncar 链用）
  hybrid_merge.py                Car + 非车 两链合并（仅 noncar 模式）
main_chain/                      Car 共享阶段（OD-main-0909 vendor 快照，勿就地改）
geometry/ filtering/ tracking/ classification/   根目录这棵：Truck / VRU / 入口用
geometry_yaw_v2/                 Truck 专用 yaw（静态锁已关）
models/                          在用权重/配置（见下）
bevfusion/                       BEVFusion 推理工程（configs + scripts + 缓存 data/ work/）
tests/                           测试：root/ + main_chain/ 两棵树各一套
bak/                             已归档（**gitignore**，见 bak/README.md）
```

`models/` 里在用的：`bevfusion_mmdet3d_lidaronly.pth`（默认）、`bevfusion_mmdet3d_lidarcam.pth`
（fusion 模式）、`voxelnext_vru_1head2cls_epoch20.pth` + `voxelnext_vru_infer.yaml`（VRU）、
`voxelnext_truckB_epoch15.pth` + `voxelnext_truck_infer.yaml`（Truck 回退，入口会校验存在）、
`vod_2cls_ft_e12.pth` + `voxelnext_fiveclass_nuscenes_infer.yaml`（`--chains noncar`）。

---

## 参数与默认值

在 `pipeline/vehicle_pass.py::DEFAULTS`（车链）、`main_chain/pipeline/step_vehicle_chain.py::DEFAULTS`
（Car 共享阶段的门槛）、`pipeline/hybrid_expD_vru.py::DEFAULTS`（VRU）。

| 项 | Car / Truck（车链） | VRU |
| --- | --- | --- |
| 分数门槛 | Car 0.2 / Truck 0.2 / Trailer 0.25 | Ped 0.2 / NMV 0.2 |
| 范围 前/后/侧 (m) | 80 / 20 / 40 | 60 / 20 / 40（行人另限 15 m，NMV 60 m）|
| 稀疏度（框内点数）| Car ≤5 / Truck ≤10 | ≤10 |
| 短轨迹 | Car ≤3 / Truck ≤4 帧 | ≤4 帧（行人另加 <20 帧整条删）|
| 挂车 | `--trailer-policy to-truck`（默认并成 Truck，标注侧无 Trailer 类）| — |
| 静态刚性框 | `--static-rigid`（默认关）| — |
| yaw | Car：main_chain 整合 yaw（动态段保留 detector yaw）；Truck：`geometry_yaw_v2` | `legacy` |

常用开关：`--chains car,truck,vru`（默认）｜`--truck-detector-mode lidar|fusion`（默认 lidar）
｜`--vru-detector voxelnext|bevfusion`（默认 voxelnext）｜`--bev-raw-dir`（复用已生成的 raw，不重跑推理）
｜`--keep-vehicle-diagnostics`｜`--keep-chain-labels`｜`--link-only`｜`--output-tag`

---

## 当前行为现状（改动前先读这里）

### yaw

| 链 | 动态段 | 静止/停车段 | 开关 |
| --- | --- | --- | --- |
| **Car**（main_chain step2） | 保留 **detector 原始 yaw**（`apply_motion_yaw=False` 写死） | `static_yaw` 把确认近零速的帧**锁到 slot / 停车排的圆中位 yaw**；另有静止多帧点云 PCA 主轴覆盖 | **没有开关**（见「已知问题」） |
| Car（step4.5） | `yaw_reversal`：每帧与「首→末轨迹方向」差 >90° 就 `+π`（`Step45Config.yaw_reversal_enabled=True`） | 无 | 有开关，但入口没透出 |
| **Truck**（geometry_yaw_v2） | 直线段用运动方向（`apply_straight_motion_yaw=True`）；转弯/遮挡保留 detector yaw | **不做静态锁**（`apply_static_direction_vote=False` + `truck_static_yaw_enabled=False`）；点云主轴规则已删 | 有（`truck_*`） |
| Truck 后处理 | ①与局部运动方向差 ≥15° 的帧改成运动方向；①b 静止 Truck 的 yaw 5° 分箱众数平滑；④π 翻转 | 同左 | `TruckPostConfig` |
| VRU | legacy yaw（`geometry/` 那棵） | 静态 slot / 静态 yaw 稳定开着 | 部分 |

### Car 高度（最终版，2026-09-22）

纯逐帧实测车顶 → 在「车顶往下 1.45~1.70 m」里找地面 → ID 高度取该 ID 各帧量到的最大深度并全 ID 统一
→ 逐帧底 = 本帧车顶 − ID 高度（只把底往下加）→ 某帧找不到地面时用本 ID 高度、再回退 1.70
→ 静态停车排：排高度取排内 ID 高度的低分位；⑤.5 排内邻居一致性（相邻 5 辆、沿排线性趋势，坡度 >2°
且高度激增 >0.50 m 时只往下拉回趋势）。

### 其它

- 车链里 `car_size_relabel=False`：不再按尺寸把大 Car 改写成 Truck（Truck 由 BEVFusion truck 头负责）。
- Truck 后处理的 IoU 并集合并默认**关**（`merge_enabled=False`）；xy 贴合默认 **off**。
- 区域外 / 无高速证据的检测在 step4.5 里**整体冻结**（ID + box + yaw 都不动）。

---

## 测试

只有一个测试目录 `tests/`，里面两套（对应两棵已分叉的模块树）：

```bash
# 全部（一条命令）：tests/root 原地跑，tests/main_chain 由子进程跑
pytest

# 只看 Car 共享阶段（main_chain 那棵）
pytest tests/main_chain

# 只看 Truck/VRU/编排（根目录那棵）
pytest tests/root
```

`tests/root/test_main_chain_suite.py` 是个「转发」用例：它用独立子进程调起
`tests/main_chain`。两棵树同名包内容不同，**不能塞进同一个进程**（详见 `tests/README.md`）。

依赖：`/home/moga/miniconda3/envs/openpcdet/bin/python -m pytest`（环境里装了 ROS 的 pytest
插件，`pytest.ini` 里已经把它们关掉了，否则收集阶段就会报错）。

---

## 已知问题 / 待办

1. **Car 的静态 yaw 锁没有开关**（已定位 + 量化）。链路：`StaticFirstTracker` 把车绑到静态 slot
   → `main_chain/geometry/static_yaw.py::stabilize_static_yaw` 把「近零速 run + departure 之前」
   的帧 yaw 改成该 slot 的圆中位（世界系）→ `yaw_static_direction._static_direction_targets`
   再把这些帧统一成「轴 ± 0/π」（决定朝向正反）→ 少量静止轨迹还会被
   `_stationary_pointcloud_targets` 用多帧点云 PCA 主轴覆盖。目标是**世界系固定值**，
   所以同一辆车每帧在世界系里 yaw 完全相同（= 整条轨迹朝向被冻住）。

   实测（`/media/moga/police/1111` 两条 clip，2026-09-23，关掉 static_yaw + 方向投票前后对比）：

   | clip | Car 框 | static_yaw 改写框数（最大角度）| 方向投票覆盖 | 关掉后轴向变化（平均/最大）| 朝向被翻 180° |
   | --- | --- | --- | --- | --- | --- |
   | `..._095943_clip3_pre` | 455 | 253（13.7°）| 311 | 1.05° / 13.7° | 2 |
   | `..._110928_clip2_pre` | 1398 | 929（31.5°）| 1027 | 1.25° / 34.0° | **69**（对照 detector 原始 yaw：68/1176 框 >90°）|

   Truck 那侧有 `truck_static_yaw_enabled=False` / `apply_static_direction_vote=False` 可关，**Car 侧没有**。
   需求：给 `main_chain` step2 加 `static_yaw_enabled` + yaw 配置入参，从 `step_vehicle_chain`
   透出 CLI（`--no-car-static-yaw`），再用上面两条 clip 做 A/B。
2. `README`/`main_chain/README.md` 里仍有少量指向 `bak/` 的旧描述（回退路径、已删的旧单链）。
3. `bevfusion/data/`（约 16 GB）是预处理缓存，里面混着旧实验的 `*_pre` / `*_pre_pre` 条目，可清。
4. `bevfusion/work/infos/` 与 `bevfusion/data/` 是**共享缓存：同一时间只跑一个批跑**，否则互相抢。

---

## 附录 A：回退路径

| 想要 | 怎么做 |
| --- | --- |
| Truck 用旧 VoxelNeXt 权重 | `--truck-detector voxelnext --truck-ckpt models/voxelnext_truckB_epoch15.pth`（Car 会因此不可用：Car 只走合并车链）|
| 旧五类单链 | `--chains noncar`（`models/vod_2cls_ft_e12.pth`）|
| 打开静态刚性框 | `--static-rigid` |
| 留分链标签 / 诊断 | `--keep-chain-labels` / `--keep-vehicle-diagnostics` |
| 不落盘只跑链路 | `--no-export-sust` |

## 附录 B：`bak/`（2026-09-23 归档，已 gitignore）

`deprecated/`（旧五类链快照）、`legacy-root-modules/`（根目录 4 个零引用 filtering 模块 +
5 个 batch/base_link 脚本 + 旧 `inference/`）、`legacy-tests/`、`dev-scripts/`
（`check_step1_env` / `merge_two_chains` / `remerge_truck_car`）、`bevfusion-tools/`
（bench/diag/eval/export/report + sweeps 配置）、`models-unused/`（11 个零引用权重，约 0.6 GB）、
`work-probes/`（89 个一次性探针脚本）、`main_chain-unused/`（main_chain 里运行时不加载的部分）。
逐条原因和恢复方法见 `bak/README.md`。

> ⚠️ `bevfusion/scripts/build_bev_pool.py` 与 `prep_data.py` 被同目录脚本以**裸名 import**
> （`from build_bev_pool import register`），属于 live 文件，不要归档。

## 附录 C：坐标系

`label.psr.rotation.z` 是绕 `base_link` 局部 z 轴的弧度；Step5 / 五类出口会把 lidar 系的 box
旋转到 base_link 再写标签（尺寸不变）。中间阶段（step2~step4.5）都在 `lidar_top` 系，
点云裁剪与点数统计同理。详见 `main_chain/README.md` 的坐标表。
