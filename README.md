# BEV 预标注端到端后处理流水线

输入：未标注的 SUST 原始 clip（含 `lidar/lidar_top/*.bin` 与 `transforms/`）。

输出：输入 clip 同级目录下的 `<clip>_pre`，即原始 clip 改名后增加 `label/`。

## 一键端到端

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python run_end_to_end.py \
  --clip /path/to/scene_clip \
  --export-sust
```

批量目录：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python run_end_to_end.py \
  --clip-dir /path/to/clips \
  --export-sust
```

`--export-sust` 可选：不传时只保留 `<clip>_pre`；传入时会把 `<clip>_pre`
复制到 `SUSTechPOINTS/data/<clip>_pre`。

如果 `<clip>_pre` 已存在，需要加 `--overwrite`。

中间 JSON 全部写入系统临时目录，跑完自动删除；不再生成 `work/`，也不再复制
中间 `_step2/_step3/_step4` clip。

端到端脚本内部依次调用五个 step：

```text
原始 clip
  -> step1  lidar 推理 + 相机可见度预过滤
  -> step2  identity / class / hard-filter / yaw
  -> step3  Car box 拟合
  -> step4  Truck 尺寸 / 插值 / 重合过滤
  -> step5 低置信类别过滤（默认删 Truck，删纯静态非机动车）
  -> SUST clip
```

## 分步运行

### Step 1：lidar 推理 + 可见度预过滤

使用 OpenPCDet 环境：

```bash
/home/moga/miniconda3/envs/openpcdet/bin/python pipeline/step1_lidar_inference.py \
  --clip /path/to/scene_clip
```

输出：`work/step1_inference/<clip>_raw.json`。

模型权重 `models/*.pth` 已 gitignore，不提交仓库；配置文件
`models/voxelnext_v2_waymo_infer.yaml` 会被 git 跟踪。

### Step 2：identity / class / filter / yaw

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step2_identity_class_filter_yaw.py \
  --in-json work/step1_inference/<clip>_raw.json \
  --clip /path/to/scene_clip \
  --out-json work/step2_identity/<clip>_step2.json \
  --out-clip work/step2_identity/data/<clip>_step2
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step2_identity_class_filter_yaw_batch.py --overwrite
```

### Step 3：Car box 拟合

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step3_car_box_fit.py \
  --step2-json work/step2_identity/<clip>_step2.json \
  --step2-diagnostics work/step2_identity/<clip>_step2_diagnostics.json \
  --clip /path/to/scene_clip \
  --out-json work/step3_car_box_fit/<clip>_step3.json \
  --out-clip work/step3_car_box_fit/data/<clip>_step3
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step3_car_box_fit_batch.py --overwrite
```

### Step 4：Truck 尺寸 / 插值 / 重合过滤

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_truck_size_interp_overlap.py \
  --step3-json work/step3_car_box_fit/<clip>_step3.json \
  --clip work/step3_car_box_fit/data/<clip>_step3 \
  --out-json work/step4_truck_size_interp_overlap/<clip>_step4.json \
  --out-clip work/step4_truck_size_interp_overlap/data/<clip>_step4
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_truck_size_interp_overlap_batch.py --overwrite
```

### Step 5：低置信类别过滤

默认开启，会删除：

```text
1. 所有 Truck 检测
2. 纯静态的非机动车（Cyclist / Nonmotorized_vehicle）
```

其中“纯静态非机动车”指整条轨迹没有任何可靠运动证据；等红灯后继续运动的非机动车会保留。

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step5_class_motion_filter.py \
  --step4-json work/step4_truck_size_interp_overlap/<clip>_step4.json \
  --clip work/step4_truck_size_interp_overlap/data/<clip>_step4 \
  --out-json work/step5_class_motion_filter/<clip>_step5.json \
  --out-clip work/step5_class_motion_filter/data/<clip>_step5
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step5_class_motion_filter_batch.py --overwrite
```

关闭过滤的可选项：

```bash
--keep-truck
--keep-static-nonmotorized
```

## 目录说明

```text
classification/  Step2 类别精修
filtering/       Step1/Step2 可见度与硬过滤
tracking/        保守跟踪器 + 静态优先跟踪器
geometry/        Step2 yaw，Step3 Car box，Step4 Truck box
inference/       Step1 OpenPCDet 推理脚本
pipeline/        step1..step5 单条与批量入口
archive/         不再参与当前链路的旧版本/旧预览文件
tests/           当前链路的单元测试
models/          推理配置与本地权重（权重不入 git）
```

## Truck 重合过滤规则

同一帧中任意两个 Truck 的 BEV IoU 大于 0 时，删除 yaw stability 更低的一条：

```text
1. yaw stability 高者保留
2. stability 完全相等时，生命周期更长者保留
3. 仍然相同，保留较小 track_id
```

yaw 旋转的重复轨迹不再做 yaw 修复，直接由重合过滤删除。

## 测试

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python -m unittest discover -s tests -v
```
