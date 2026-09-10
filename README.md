# 五类别 LiDAR 预标注流水线

把 SUSTechPOINTS 的原始 clip（含 `lidar/lidar_top/*.bin`）推理成五类预标注：
**Car / Truck / Bus / Pedestrian / Nonmotorized_vehicle**，输出可直接在
SUSTechPOINTS 打开的 `<clip>_pre/label/*.json`。

本分支（`hybrid-main-car-expd-noncar`）默认走**混合链路**：先用 `main_chain/` 的
Waymo 生成并固定 Car（`main_chain/` 已同步到最新 OD-main-0909 快照，含 Step4.5
动态区域重跟踪），再用 `models/vod_2cls_ft_e12.pth` 保留其余四类，最后按
`frame_id` 合并为一个 `<clip>_pre`。

## 本机环境（已就绪）

入口脚本会自动探测可用的 **OpenPCDet** 环境（本机为 conda 的 `openpcdet`，
Python 3.10.20，CUDA 12.4，RTX A4000），无需手动安装依赖。

- 手动指定解释器：`export OPENPCDET_PYTHON=<python>` 或用 `--python <python>`。
- 环境自检：`python scripts/check_step1_env.py --cfg models/voxelnext_fiveclass_nuscenes_infer.yaml --ckpt models/vod_2cls_ft_e12.pth`。

## 单个运行

第一个参数填**单个 clip 目录**，第二个参数是输出目录（SUST 数据根，可选）。

```bash
bash hybrid_run.sh <clip> <sust_root> --overwrite
```

本机示例：

```bash
bash hybrid_run.sh \
  /home/moga/桌面/test/scene_crossroad_my_record_20260827_164838_clip5/ \
  /home/moga/桌面/SUSTechPOINTS/data/ --overwrite
```

输出：

```text
<sust_root>/<clip>_pre/
└── label/<frame_id>.json
```

## 批量运行

第一个参数填**父目录**，脚本会扫描其下所有含 `lidar/lidar_top/*.bin` 的 clip：

```bash
bash hybrid_run.sh <clip_parent> <sust_root> --overwrite
```

本机示例：

```bash
bash hybrid_run.sh /home/moga/桌面/test /home/moga/桌面/SUSTechPOINTS/data/ --overwrite
```

## 批量遍历 police/0903 的 step2 clip

`/media/moga/police/0903/<scene>/step2/scene_*_clipN/` 是分阶段的 SUST clip
（每个场景下有 `step1/step2/step3`，预测数据在 `step2` 下的 `scene_*_clipN/`）。
下面的 for 循环会逐个遍历这些 clip 并跑混合链路：

```bash
DATA_ROOT=/media/moga/police/0903
SUST=/home/moga/桌面/SUSTechPOINTS/data

for scene in "$DATA_ROOT"/*/; do
  for clip in "${scene%/}"/step2/scene_*_clip*/; do
    [ -d "$clip" ] || continue                       # 只处理目录，跳过 .zip
    echo "== 处理 ${clip%/} =="
    bash hybrid_run.sh "$clip" "$SUST" --overwrite
  done
done
```

不想落盘到 SUST（只跑链路、不导出）时，给 `hybrid_run.sh` 加 `--no-export-sust`：

```bash
DATA_ROOT=/media/moga/police/0903
for scene in "$DATA_ROOT"/*/; do
  for clip in "${scene%/}"/step2/scene_*_clip*/; do
    [ -d "$clip" ] || continue
    echo "== 处理（不导出）${clip%/} =="
    bash hybrid_run.sh "$clip" "$SUST" --overwrite --no-export-sust
  done
done
```

只想跑某个场景时，上面的外层 for 可改成 `for scene in "$DATA_ROOT"/*scene名*/; do`；
想跳过已生成的 `*_pre`，把 `--overwrite` 去掉即可（输出已存在会报错，不想中断就先删除旧 `_pre`）。

## 导出到 SUST（可选）

第二个参数就是 SUSTechPOINTS 数据根目录，**默认导出**为
`<sust_root>/<clip>_pre/`。**不导出**时给 `hybrid_run.sh` 加 `--no-export-sust`，
只跑链路、不落盘（`output_root` 不会被创建）：

```bash
bash hybrid_run.sh <clip> /tmp/foo --no-export-sust --overwrite
```

- 输出已存在时需加 `--overwrite`（先删再生成）。

### 原地端到端（不导出到 SUST）

加 `--in-place`：不往 SUST 拷，也不额外留一份 raw，直接把每个输入 clip
在**原位置**改名为 `<clip>_pre` 并写入 `label/`，等价于 main-chain 的
端到端原地产出（原 clip 路径会消失）：

```bash
bash hybrid_run.sh <clip> --in-place --overwrite
```

批量循环：

```bash
DATA_ROOT=/media/moga/police/0903
for scene in "$DATA_ROOT"/*/; do
  for clip in "${scene%/}"/step2/scene_*_clip[0-9]*/; do
    [ -d "$clip" ] || continue
    [[ "${clip%/}" == *_pre ]] && continue
    echo "== 原地处理 ${clip%/} =="
    bash hybrid_run.sh "$clip" --in-place --overwrite
  done
done
```

## 权重（默认 e12）

最终生产权重为 `models/vod_2cls_ft_e12.pth`（VOD 2 类微调，冻结骨干与
Car/Truck/Bus 头，只训练 Pedestrian / Nonmotorized_vehicle 头）；Car 仍由
`main_chain/` 的 Waymo 权重 `main_chain/models/vn_waymo_v2_4gpu_full_epoch10.pth`
生成。默认类别分数阈值为：

- `Truck` / `Bus`: `0.4`
- `Pedestrian`: `0.15`
- `Nonmotorized_vehicle`: `0.20`

非车 raw 推理阈值默认取上述类别阈值的最小值（当前 `0.15`）。

Step3 几何精修后、导出 base_link 之前，`Nonmotorized_vehicle` 只保留
**world 系首尾净位移 `> 15m`** 的轨迹：首尾中心取该轨迹最早和最晚的可用观测，
pose 缺失的帧不参与计算；净位移 `<= 15m` 的轨迹整条删除，不足两帧可用的轨迹
保留（无法测量）。首尾净位移不受相邻帧抖动影响。阈值可通过
`--nonmotorized-min-net-displacement` 调整。

Car 与非车合并时还有一条逐帧规则：若同一帧内有 **Truck 或 Bus 与某个 Car 的
BEV 交集面积 / Car 面积 `> 0.5`**，只删除这一帧的这个 Car，Truck/Bus 保留，
该 Car 在其他帧不受影响。原有 `BEV IoU >= 0.8` 的整条轨迹吸收规则保持不变。

可用 `--noncar-ckpt models/vod_2cls_ft_e25.pth` 切换到 e25 版本（行人召回更高，
耗时也更长）。若 `models/*.pth` 只有 133 字节，说明是 Git LFS 指针，先
`git lfs pull`，或用真实权重覆盖后再跑。

## 实测耗时

RTX A4000 16GB、不与其他大任务并行时，混合链路单个 80 帧 clip：

- `vod_2cls_ft_e12.pth`（默认，Ped `0.15` / NMV `0.20`，主链含
  Step4.5）：约 2.6–3.0 分钟/clip；本机 4-clip 实测平均约 2.8 分钟
  （主链约 109–118s，非车链约 47–61s）。
- 若把 Ped/NMV 阈值调回 `0.1`：非车 raw 检测数约 2.0–2.2 万/clip，
  单 clip 约 4.5–5 分钟。
- `vod_2cls_ft_e25.pth`：Ped/NMV 候选比 e12 更多，耗时更长。

CPU 后处理（跟踪关联、可见度、几何精修）是主要瓶颈；阈值越低、低分候选
越多，耗时越久。

## 目录

```text
hybrid_run.sh       本分支混合入口（main-Car + VOD-非Car）
main_chain/         最新 OD-main-0909 快照（Waymo Car 链路 + Step4.5）
pipeline/           当前 Step1/Step2/Step2.5/Step3 主链路
classification/     Step2.5 类别归一化和 track 投票
filtering/          可见度、硬过滤、final 五类输出
tracking/           类别无关跟踪、坐标变换、SUST label 映射
geometry/           yaw、Car 几何、Truck/NMV 精修
inference/          OpenPCDet LiDAR 推理
models/             配置和 checkpoint
scripts/            环境检查与一键入口
tests/              单元测试
```

运行后处理单元测试：

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
