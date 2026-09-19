# 四类别 LiDAR 预标注流水线（Car / Truck / Pedestrian / Nonmotorized_vehicle）

输入：SUSTechPOINTS 原始 clip（含 `lidar/lidar_top/*.bin` 与 `transforms/`）。
输出：每帧写在 `<clip>_pre/label/<frame_id>.json`，可直接用 SUSTechPOINTS 打开。

## 接入链路：三条独立链，按 Car -> Truck -> VRU 顺序执行

```text
1. Car    main_chain/（Waymo Car + Step4.5）                   -> Car
2. Truck  pipeline/hybrid_expD_truck.py（Truck 专用后处理）    -> Truck                    obj_id +1000
3. VRU    pipeline/hybrid_expD_vru.py                          -> Pedestrian / NMV          obj_id +2000
                                    ↓
                     按 frame_id 合并成一份 label/
```

三条链各自独立推理、独立后处理、独立调参（范围 / 阈值 / yaw 策略互不干扰），
最后才合并。顺序固定为 **Car -> Truck -> 最后 Pedestrian/Nonmotorized_vehicle**。

> 旧的两链模式（main Car + VOD 五类非车，Bus 在非车链开头折叠成 Truck）仍保留：
> 加 `--chains car,noncar`，参数见《附录：旧五类非车链》。

入口是 `hybrid_run.sh`，会自动探测本机 OpenPCDet 环境（默认
`~/miniconda3/envs/openpcdet`），不需要手动激活 conda。

## 三条链的权重

| 链 | 权重 | 推理配置 | raw 分数门槛 |
| --- | --- | --- | --- |
| Car | `main_chain/models/vn_waymo_v2_4gpu_full_epoch10.pth` | `main_chain/models/voxelnext_v2_waymo_infer.yaml` | main_chain 内部固定 |
| Truck | `models/voxelnext_truckB_epoch15.pth` | `models/voxelnext_truck_infer.yaml` | `--truck-raw-threshold`（默认 `0.4`） |
| VRU | `models/voxelnext_vru_1head2cls_epoch20.pth` | `models/voxelnext_vru_infer.yaml` | `--vru-raw-threshold`（默认 `0.3`） |

`models/*.pth` 只有 133 字节 = Git LFS 指针，先 `git lfs pull` 再跑。

## 合并规则

1. 三条链的框按 `frame_id` 拼到一起；同帧 `obj_id` 冲突时自动分配一个空闲 id。
2. **Car 被 Truck 覆盖的面积 / Car 面积 ≥ 阈值 -> 删掉该 Car 轨迹的全部帧**
   （`--car-truck-cover-threshold`，默认 `0.5`，给 `0` 关闭）。
   - 判据是"交面积 / Car 面积"，**不是 IoU** —— Truck 比 Car 大得多，IoU 天然偏小；
   - 同一 Car id 只要有任意一帧命中，**整条轨迹（所有帧）都删**，不是只删重叠那一帧。

## Truck 链（`pipeline/hybrid_expD_truck.py`）

- 只保留 `Truck`（该链的类别归一化已把 Bus 并入 Truck）
- 范围：前 80 / 后 20 / 左右 40 m
- 分数阈值：Truck `0.4`；短轨迹过滤 4 帧
- yaw：`v2` —— 直线行驶的轨迹用运动方向作 yaw，拐弯保留 detector yaw；
  不做静态方向投票、不把动态轨迹钉到停车位 id、跳过静态 yaw 稳定
- `obj_id` 从 1000 起

Truck 专用后处理 `geometry/truck_postprocess.py`，执行顺序：

| 步 | 内容 | 默认 |
| --- | --- | --- |
| ① | yaw 旋转帧修正：与局部运动方向偏差 ≥ 阈值的帧，yaw 改成局部运动方向 | 开，阈值 `15°` |
| ①b | 静止 Truck 的 yaw 平滑：yaw 取 mod π 后分箱众数投票 + 圆中位（取抖动中间方向） | 开 |
| ② | IoU 并集合并（把"长车被拆成两段"并成并集长框，含合并 id 的统一尺寸/朝向） | **关**（`merge_enabled=False`） |
| ③ | xy 贴合（复用 main 链 Car 的单面收缩逻辑） | **关**（`xy_fit_mode="off"`） |

②③ 关掉的原因（实测）：

- 并集会把"同一目标重复检出"的两个平行框并成超宽框（实测 13.07×3.00 + 13.39×3.02
  被并成 13.96×7.96），并且并集框的 yaw 用点云 PCA 重算时可能被掰歪 90°；
- xy 贴合即使只贴合长轴，273 个 Truck 框里也有 20.5% 被沿长轴挪中心
  （56/273 挪 >0.5m，29/273 挪 >2m，最大 4.9m）、25/273 的框长度被砍掉一半
  —— 框会"突出"（卡车常只看到一个端面，face-visibility 拟合把可见点簇边缘当成了"面"）。

需要时改 `TruckPostConfig` 的 `merge_enabled=True` / `xy_fit_mode="long_axis"` 即可回退。

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

## 只改了 Truck 时：复用 Car/VRU 标签重跑

```bash
python scripts/remerge_truck_car.py --output-root <含 <clip>_pre 的目录> [--write]
```

读 `<clip>_pre/label/` 里已有的 Car 与 `label_vru/`，**只重跑 Truck 链**（原始推理 + 链），
再按上面的合并规则重写 `label/` 和 `label_truck/`。改 Truck 参数或合并规则时用它，
可以省掉最慢的 Car 链。

## 两种运行模式

### 模式一：原地端到端（不导出 SUST）

输入 clip 在**原位置**改名成 `<clip>_pre`，并把合并后的 `label/` 写进去；
不会额外保留一份 raw，也不会往 SUST 拷贝。

```bash
bash hybrid_run.sh <clip> --in-place --overwrite
```

输出：

```text
<clip>_pre/
├── lidar/
├── image/
├── transforms/
└── label/<frame_id>.json
```

注意：运行成功后原路径 `<clip>` 会消失，变成 `<clip>_pre`；已有同名
`<clip>_pre` 时会被 `--overwrite` 先删再生成。

### 模式二：导出到 SUST

原始 clip 保留不动，结果写到 `<output_root>/<clip>_pre`。

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

`<output_root>` 省略时默认是 `~/SUSTechPOINTS/data`。
`--output-tag truckb_e15` 可以把输出名改成 `<clip>_truckb_e15_pre`，方便保留多组对比结果。
加 `--keep-chain-labels` 会额外写出 `label_truck/` 与 `label_vru/`，便于按链排查。

### 仅调试：只跑链路、不落盘

```bash
bash hybrid_run.sh <clip> /tmp/unused --no-export-sust --overwrite
```

只打印统计信息，临时 JSON 跑完自动删除，不产生 `<clip>_pre`，也不改输入。

## 批量运行（两种目录结构）

批量按目录结构分两种，两种输出模式（原地 / 导出 SUST）都适用。

### 批量 A：一个大目录下直接就是一批 clip

`<clip_parent>/` 下面直接是各个 clip 目录，每个 clip 含 `lidar/lidar_top/*.bin`。
`hybrid_run.sh` 会自动扫描并逐个处理。

```bash
bash hybrid_run.sh /path/to/clips --in-place --overwrite
bash hybrid_run.sh /home/moga/桌面/预标测效/ /home/moga/桌面/SUSTechPOINTS/data --overwrite
```

### 批量 B：分场景 / step2 的嵌套结构（police/0903）

`/media/moga/police/0903` 是 `<scene>/step2/<clip>/` 结构，需要用 shell 逐层遍历：

```bash
DATA_ROOT=/media/moga/GEN2/0915/
SUST=/home/moga/桌面/SUSTechPOINTS/data

for scene in "$DATA_ROOT"/*/; do
  for clip in "${scene%/}"/step2/scene_*clip*/; do
    clip="${clip%/}"
    name="$(basename "$clip")"

    [ -d "$clip/lidar/lidar_top" ] || continue      # 不是有效 clip
    [[ "$name" == *_pre ]] && continue              # 已经是输出，跳过
    [ -d "$SUST/${name}_pre" ] && continue          # SUST 已有输出，跳过

    echo "== 处理 $clip =="
    bash hybrid_run.sh "$clip" --in-place --overwrite
    # 导出 SUST 模式：把上面一行换成
    # bash hybrid_run.sh "$clip" "$SUST" --overwrite
  done
done
```

## 参数与默认值

### 入口参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `<input_root>` | 必填 | 单个 clip 目录，或包含多个 clip 的父目录 |
| `<output_root>` | `~/SUSTechPOINTS/data` | 导出 SUST 时的输出根目录；`--in-place` 时忽略 |
| `--chains` | `car,truck,vru` | 要跑的链，逗号分隔，按 car -> truck -> vru 顺序执行；可选 `car` / `truck` / `vru` / `noncar`（旧五类单链） |
| `--car-truck-cover-threshold` | `0.5` | Car 被 Truck 覆盖的面积 / Car 面积达到该值就删掉整条 Car 轨迹；`0` 关闭 |
| `--keep-chain-labels` | 关 | 额外把 `label_truck/` `label_vru/` 写进输出 clip |
| `--in-place` | 关 | 原地端到端：输入 clip 改名 `<clip>_pre` |
| `--export-sust` | **开** | 导出到 `<output_root>/<clip>_pre`；与 `--in-place` 互斥 |
| `--no-export-sust` | 关 | 只跑链路，不落盘 |
| `--output-tag` | 空 | 在 `<clip>` 和 `_pre` 之间加标签 |
| `--overwrite` | 关 | 输出已存在时先删除再生成 |
| `--python` | 自动探测 | 指定运行/后处理 Python |
| `--skip-install` | 关 | 只允许使用已有 CUDA/OpenPCDet 环境 |
| `--drop-vis-below` | `0.05` | 非车链可见度硬过滤阈值（Truck/VRU 链内部固定 0.05） |

### Truck 链参数（透出部分）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--truck-ckpt` | `models/voxelnext_truckB_epoch15.pth` | Truck 权重 |
| `--truck-cfg` | `models/voxelnext_truck_infer.yaml` | 推理配置 |
| `--truck-raw-threshold` | `0.4` | Truck raw 推理分数门槛 |

其余（范围 80/20/40、Truck 分数 0.4、短轨迹 4、yaw v2、Truck 后处理各开关）
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

### main_chain Car 参数（OD-main-0909 默认）

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

### 实测耗时

本机 RTX A4000 16GB、不与其他大任务并行时，单个 80 帧 clip（`155112_clip6`）：

| 阶段 | 耗时 |
| --- | --- |
| Car 链（main_chain：raw 推理 + Step2~Step5 + Step4.5） | 102.6 s |
| Truck 链（raw 推理 + 后处理） | 23.5 s |
| VRU 链（raw 推理 + 后处理） | 39.1 s |
| 三链合并 + Car/Truck 覆盖规则 | 0.5 s |
| 导出 `<clip>_pre`（约 600 MB copytree） | 0.6 s |
| **合计** | **166.5 s（≈2 分 47 秒）/clip** |

每次运行入口都会打印这一行，可直接看自己机器/场景的实测值：

```text
[hybrid] <clip>: 计时 car=102.6s, truck=23.5s, vru=39.1s, merge=0.5s, export=0.6s,
                total_without_export=165.7s, total=166.5s
```

- Car 链占约 6 成；CPU 后处理（跟踪关联、可见度、几何精修）是主要瓶颈，阈值越低越慢。
- 框越多越慢：同批三条 clip 实测 150–167 s/clip。
- 只改 Truck 时用 `scripts/remerge_truck_car.py` 约 **25 s/clip**（复用 Car/VRU 标签，只重跑 Truck 链）。

## 目录

```text
hybrid_run.sh           入口（三链：Car -> Truck -> VRU）
main_chain/             最新 OD-main-0909 快照（Waymo Car 链路 + Step4.5）
pipeline/               各链主体：hybrid_expD_noncar / hybrid_expD_truck / hybrid_expD_vru
                        hybrid_main_car（Car 链调用）、hybrid_merge（标签合并）
geometry/truck_postprocess.py   Truck 专用后处理（yaw 修正 / 静止平滑 / 可选并集与贴合）
geometry_yaw_v2/        Truck 链用的新版 yaw（直线运动方向 / 静态方向 / 动态 yaw）
classification/         类别归一化与 track 投票
filtering/              可见度、硬过滤、五类输出、低置信类别过滤
tracking/               跟踪、坐标变换、SUST label 映射
geometry/               yaw、Car 几何、Truck/NMV 精修
inference/              OpenPCDet LiDAR 推理
models/                 三条链的配置与 checkpoint
scripts/                入口脚本、merge_two_chains（旧两链合成）、remerge_truck_car（只重跑 Truck）
tests/                  单元测试
```

测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

## 附录：旧五类非车链（`--chains car,noncar`）

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
