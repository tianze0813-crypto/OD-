# 五类别 LiDAR 预标注流水线

把 SUSTechPOINTS 的原始 clip（含 `lidar/lidar_top/*.bin`）推理成五类预标注：
**Car / Truck / Bus / Pedestrian / Nonmotorized_vehicle**，输出可直接在
SUSTechPOINTS 打开的 `<clip>_pre/label/*.json`。

本分支（`hybrid-main-car-expd-noncar`）默认走**混合链路**：先用 `main_chain/` 的
Waymo 生成并固定 Car，再用 `models/expD_e8.pth` 保留其余四类，最后按 `frame_id`
合并为一个 `<clip>_pre`。

## 本机环境（已就绪）

入口脚本会自动探测可用的 **OpenPCDet** 环境（本机为 conda 的 `openpcdet`，
Python 3.10.20，CUDA 12.4，RTX A4000），无需手动安装依赖。

- 手动指定解释器：`export OPENPCDET_PYTHON=<python>` 或用 `--python <python>`。
- 环境自检：`python scripts/check_step1_env.py --cfg models/voxelnext_fiveclass_nuscenes_infer.yaml --ckpt models/expD_e8.pth`。

## 单个运行

第一个参数填**单个 clip 目录**，第二个参数是输出目录（SUST 数据根，可选）。

```bash
bash hybrid_run.sh <clip> <sust_root> --overwrite
bash run.sh <clip> <sust_root> --overwrite
```

本机示例：

```bash
bash hybrid_run.sh \
  /home/moga/桌面/test/scene_crossroad_my_record_20260827_163412_clip0 \
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
bash run.sh <clip_parent> <sust_root> --overwrite
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

只想跑某个场景时，上面的外层 for 可改成 `for scene in "$DATA_ROOT"/*scene名*/; do`；
想跳过已生成的 `*_pre`，把 `--overwrite` 去掉即可（输出已存在会报错，不想中断就先删除旧 `_pre`）。

## 导出到 SUST（可选）

第二个参数就是 SUSTechPOINTS 数据根目录，**默认导出**为
`<sust_root>/<clip>_pre/`。

- **不导出**：`run.sh` 加 `--no-export-sust`，只跑链路、不落盘：

  ```bash
  bash run.sh <clip> /tmp/foo --no-export-sust --overwrite
  ```

- `hybrid_run.sh` 始终导出；不需要落盘时把 `<sust_root>` 指向任意临时目录即可。
- 输出已存在时需加 `--overwrite`（先删再生成）。

## 权重

默认权重为 `models/expD_e8.pth`，`run.sh` 可用 `--weight` 切换别名，`waymo` 只能配
`--mode inference`。查看可用别名：

```bash
bash run.sh --list-weights
```

若 `models/*.pth` 只有 133 字节，说明是 Git LFS 指针，先 `git lfs pull`，或用真实
权重覆盖后再跑。

## 目录

```text
run.sh              五类别一键入口（单模型，可开关 SUST 导出）
hybrid_run.sh       本分支混合入口（main-Car + expD-非Car）
main_chain/         main 分支快照（Waymo Car 链路）
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
