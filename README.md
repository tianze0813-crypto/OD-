# 五类别 LiDAR 预标注流水线（混合链路）

输入：SUSTechPOINTS 原始 clip（含 `lidar/lidar_top/*.bin` 与 `transforms/`）。
输出：五类预标注 **Car / Truck / Bus / Pedestrian / Nonmotorized_vehicle**，
每帧写在 `<clip>_pre/label/<frame_id>.json`，可直接用 SUSTechPOINTS 打开。

混合链路：

```text
main_chain/（最新 OD-main-0909 快照，Waymo Car + Step4.5）
  -> VOD e12 非车四类（Ped / NMV / Truck / Bus）
  -> 按 frame_id 合并为五类
```

入口是 `hybrid_run.sh`，会自动探测本机 OpenPCDet 环境（默认
`~/miniconda3/envs/openpcdet`），不需要手动激活 conda。

## 两种运行模式

### 模式一：原地端到端（不导出 SUST）

输入 clip 在**原位置**改名成 `<clip>_pre`，并把合并后的 `label/` 写进去；
不会额外保留一份 raw，也不会往 SUST 拷贝。等价于 `OD-main-0909` 里
`run_end_to_end.py` 的默认行为。

单 clip：

```bash
bash hybrid_run.sh <clip> --in-place --overwrite
```

批量：见下面《批量运行（两种目录结构）》。

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

原始 clip 保留不动，结果写到 `<sust_root>/<clip>_pre`。

单 clip：

```bash
bash hybrid_run.sh <clip> <sust_root> --overwrite
```

例如：

```bash
bash hybrid_run.sh \
  /media/moga/police/0903/nonmotor_lane_my_record_20260903_100541/step2/scene_nonmotor_lane_my_record_20260903_100541_clip43 \
  /home/moga/桌面/SUSTechPOINTS/data --overwrite
```

批量：见下面《批量运行（两种目录结构）》。

输出：

```text
<sust_root>/
└── <clip>_pre/
    ├── 原始 clip 数据（lidar / image / transforms）
    └── label/<frame_id>.json
```

`<sust_root>` 省略时默认是 `~/SUSTechPOINTS/data`。
`--output-tag vod_e12` 可以把输出名改成 `<clip>_vod_e12_pre`，方便保留多组
对比结果。

### 仅调试：只跑链路、不落盘

```bash
bash hybrid_run.sh <clip> /tmp/unused --no-export-sust --overwrite
```

只打印统计信息，临时 JSON 跑完自动删除，不产生 `<clip>_pre`，也不改输入。
这个模式只用于调参/回归，不是生产输出模式。

## 批量运行（两种目录结构）

批量按目录结构分两种，两种输出模式（原地 / 导出 SUST）都适用。

### 批量 A：一个大目录下直接就是一批 clip

`<clip_parent>/` 下面直接是各个 clip 目录，每个 clip 含 `lidar/lidar_top/*.bin`。
`hybrid_run.sh` 会自动扫描并逐个处理。

原地端到端：

```bash
bash hybrid_run.sh /path/to/clips --in-place --overwrite
```

导出 SUST：

```bash
bash hybrid_run.sh /path/to/clips /home/moga/桌面/SUSTechPOINTS/data --overwrite
```

### 批量 B：分场景 / step2 的嵌套结构（police/0903）

`/media/moga/police/0903` 是 `<scene>/step2/<clip>/` 结构，需要用 shell 逐层遍历：

```bash
DATA_ROOT=/media/moga/police/0903
SUST=/home/moga/桌面/SUSTechPOINTS/data

for scene in "$DATA_ROOT"/*/; do
  for clip in "${scene%/}"/step2/scene_*clip*/; do
    clip="${clip%/}"
    name="$(basename "$clip")"

    [ -d "$clip/lidar/lidar_top" ] || continue      # 不是有效 clip
    [[ "$name" == *_pre ]] && continue              # 已经是输出，跳过
    [ -d "$SUST/${name}_pre" ] && continue          # SUST 已有输出，跳过

    echo "== 处理 $clip =="
    # 原地模式：
    bash hybrid_run.sh "$clip" --in-place --overwrite
    # 导出 SUST 模式：把上面一行换成
    # bash hybrid_run.sh "$clip" "$SUST" --overwrite
  done
done
```

说明：

- A 适合所有 clip 都平铺在同一个父目录的场景；
- B 适合 `/media/moga/police/0903/<scene>/step2/...` 这种分场景嵌套结构；
- 两种模式都支持 A / B，区别只是输出落在原位置还是 SUST；
- 原地模式可以删掉 `[ -d "$SUST/${name}_pre" ] && continue` 这一行，
  这样即使之前导出过 SUST，也会在每个 clip 原位置再生成一份 `_pre`。

## 参数与默认值

### 入口参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `<input_root>` | 必填 | 单个 clip 目录，或包含多个 clip 的父目录 |
| `<output_root>` | `~/SUSTechPOINTS/data` | 导出 SUST 时的输出根目录；`--in-place` 时忽略 |
| `--in-place` | 关 | 原地端到端：输入 clip 改名 `<clip>_pre` |
| `--export-sust` | **开** | 导出到 `<output_root>/<clip>_pre`；与 `--in-place` 互斥 |
| `--no-export-sust` | 关 | 只跑链路，不落盘 |
| `--output-tag` | 空 | 在 `<clip>` 和 `_pre` 之间加标签 |
| `--overwrite` | 关 | 输出已存在时先删除再生成 |
| `--python` | 自动探测 | 指定运行/后处理 Python；也可用 `OPENPCDET_PYTHON` |
| `--skip-install` | 关 | 只允许使用已有 CUDA/OpenPCDet 环境 |
| `--drop-vis-below` | `0.05` | 非车链可见度硬过滤阈值 |

### 非车链参数（VOD e12）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--noncar-ckpt` | `models/vod_2cls_ft_e12.pth` | 非车权重；可切 `vod_2cls_ft_e25.pth` |
| `--noncar-cfg` | `models/voxelnext_fiveclass_nuscenes_infer.yaml` | 推理配置 |
| `--noncar-raw-threshold` | `0.15`（= 类别阈值最小值） | 非车 raw 推理分数门槛 |
| `--truck-score-threshold` | `0.4` | Truck 最终分数阈值 |
| `--bus-score-threshold` | `0.4` | Bus 最终分数阈值 |
| `--pedestrian-score-threshold` | `0.15` | Pedestrian 最终分数阈值 |
| `--nonmotorized-score-threshold` | `0.20` | NMV 最终分数阈值 |
| `--score-threshold` | 未设置 | 一次性覆盖以上四个类别阈值；raw 门槛仍取最小值 |
| `--pedestrian-max-distance` | `20.0` | Pedestrian 最大距离（米） |
| `--nonmotorized-max-distance` | `60.0` | NMV 最大距离（米） |
| `--sparsity-max-points` | `10` | 非车链点数过滤阈值 |
| `--short-track-max-frames` | `4` | 非车链短轨迹过滤帧数 |
| `--nonmotorized-min-net-displacement` | `15.0` | 只保留 world 系首尾净位移大于该值的 NMV 轨迹 |

说明：`--noncar-raw-threshold` 最终会被限制为
`min(显式 raw 值, min(类别阈值))`；默认 Ped/NMV 为 0.15/0.20 时 raw=0.15。

### main_chain Car 参数（OD-main-0909 默认）

`hybrid_run.sh` 调用 `main_chain/run_end_to_end.py` 时使用以下固定默认值，
目前没有全部透出成 hybrid 入口参数；main 链路本身按这套参数运行：

| 环节 | 参数 | 默认值 |
| --- | --- | --- |
| Step1 推理 | `--score-thresh` | `0.3` |
| Step1 可见度 | `--drop-vis-below` | `0.05` |
| Step5 终检 | `--sparsity-max-points` | `5` |
| Step5 终检 | `--short-track-max-frames` | `3` |
| Step4.5 动态区域 | `--extension-length-m` | `30.0` |
| Step4.5 相位拼接 | `--phase-merge-max-gap-sec` | `30.0` |
| Step4.5 右转 yielding | `--yielding-max-gap-sec` | `6.0` |

注意：hybrid 入口的 `--sparsity-max-points` 和 `--short-track-max-frames`
是给**非车链**的，不会改 main_chain 的 Step5 参数。

## 权重与阈值

默认非车权重：`models/vod_2cls_ft_e12.pth`（VOD 2 类微调，冻结骨干与
Car/Truck/Bus 头，只训练 Pedestrian / Nonmotorized_vehicle 头）。
Car 由 `main_chain/` 的 Waymo 权重生成。

默认非车类别分数阈值：

- Truck / Bus：`0.4`
- Pedestrian：`0.15`
- Nonmotorized_vehicle：`0.20`
- raw 门槛：`min(...) = 0.15`

可选 `--noncar-ckpt models/vod_2cls_ft_e25.pth`：Pedestrian 召回更高，耗时更长。
若 `models/*.pth` 只有 133 字节，说明是 Git LFS 指针，先 `git lfs pull`。

合并阶段还有两条后处理规则：

1. NMV 只保留 world 系首尾净位移 `> 15m` 的轨迹；
2. Truck/Bus 与 Car 在单帧内 BEV 重叠面积 / Car 面积 `> 0.5` 时，只删该帧的 Car。

## 实测耗时

RTX A4000 16GB、不与其他大任务并行时，单个 80 帧 clip：

- 默认 e12（Ped 0.15 / NMV 0.20，主链含 Step4.5）：约 2.6–3.0 分钟/clip；
- 4-clip 实测平均约 2.8 分钟（主链约 109–118s，非车链约 47–61s）；
- Ped/NMV 调回 0.1：非车 raw 约 2.0–2.2 万框/clip，约 4.5–5 分钟/clip；
- e25：候选更多，耗时更长。

CPU 后处理（跟踪关联、可见度、几何精修）是主要瓶颈；阈值越低，低分候选
越多，耗时越久。

## 目录

```text
hybrid_run.sh       混合链路入口（main-Car + VOD-非Car）
main_chain/         最新 OD-main-0909 快照（Waymo Car 链路 + Step4.5）
pipeline/           非车 Step1/Step2/Step2.5/Step3 主链路
classification/     类别归一化与 track 投票
filtering/          可见度、硬过滤、五类输出
tracking/           跟踪、坐标变换、SUST label 映射
geometry/           yaw、Car 几何、Truck/NMV 精修
inference/          OpenPCDet LiDAR 推理
models/             配置与 checkpoint
scripts/            环境检查与一键入口
tests/              单元测试
```

测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
