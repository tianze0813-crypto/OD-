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

如果只需要最终的 Car 标签，可额外开启可选 Step6：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python run_end_to_end.py \
  --clip /path/to/scene_clip \
  --car-only
```

如果 `<clip>_pre` 已存在，需要加 `--overwrite`。

中间 JSON 全部写入系统临时目录，跑完自动删除；不再生成 `work/`，也不再复制
中间 `_step2/_step3/_step4` clip。

端到端脚本内部依次调用五个 step；传入 `--car-only` 时再调用可选 Step6：

```text
原始 clip
  -> step1  lidar 推理 + 相机可见度预过滤
  -> step2  identity / class / hard-filter / yaw
  -> step3  Car box 拟合
  -> step4  Truck 尺寸 / 插值 / 重合过滤
  -> step5 低置信类别过滤（默认删 Truck，删纯静态非机动车）
  -> step6（可选）只保留最终标签类别 Car
  -> SUST clip
```

## 坐标系与 base_link 约定

- 推理输入是 `lidar/lidar_top/*.bin`；模型输出的 `box_lidar` 和 Step2--Step5
  的所有点云、框几何都在 `lidar_top` 局部坐标系，格式为
  `[x, y, z, dx, dy, dz, yaw]`，长度单位为米，`yaw` 是绕局部 z 轴的弧度。
- 最终 `label/<frame>.json` 的 `psr.position`、`psr.scale`、`psr.rotation.z`
  仍然使用这个 `lidar_top` 坐标系；`box_to_label` 只做字段和类别映射，不做
  坐标变换。SUST 的 `Vehicle` 会映射为最终 `obj_type: Car`。
- 当前 `CoordinateProvider` 按以下方向读取标定：

  ```text
  base_from_pose      = tf2base_link.pose
  base_from_lidar_top = tf2base_link.lidar_top
  world_from_lidar_top(t)
      = world_from_pose(t) @ inv(base_from_pose) @ base_from_lidar_top
  ```

  `pose_data.txt` 每行按当前代码解释为
  `timestamp_ns, x, y, z, qx, qy, qz, qw`，提供 `world_from_pose`。矩阵必须是
  有限的 4x4 刚体变换，时间戳单位必须是纳秒，四元数顺序必须是 `x,y,z,w`。
- 相机可见度使用静态变换
  `cam_from_lidar_top = inv(base_from_cam) @ base_from_lidar_top`；没有
  `lidar_top` 外参的旧 clip 才回退到 `pose`。

把 `base_link` 的原点从 front 雷达改到 top 雷达本身不会要求重写跟踪或标签坐标，
前提是同步重算 `tf2base_link` 下所有传感器外参，并保持 `pose_data.txt` 的 pose
语义不变。若 top 就是新的 base 原点，通常应满足
`tf2base_link.lidar_top` 为单位阵（具体仍以标定工具输出为准）。不要只改
`pose` 或只改 `lidar_top`，也不要把检测框先转换到 `base_link` 后仍称为
`box_lidar`；否则 world 跟踪、yaw、点云拟合、相机投影和最终 label 会同时产生
平移/旋转偏差。若 `pose_data.txt` 也改成了 `world_from_lidar_top`，则需要同步
改写 `CoordinateProvider` 公式，不能直接套用当前实现。

## 分步运行

### Step 1：lidar 推理 + 可见度预过滤

使用 OpenPCDet 环境：

```bash
/home/moga/miniconda3/envs/openpcdet/bin/python pipeline/step1_lidar_inference.py \
  --clip /path/to/scene_clip
```

输出：`work/step1_inference/<clip>_raw.json`。

模型权重和配置文件均位于 `models/`，会随仓库一起跟踪。

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

### Step 6（可选）：只保留 Car 标签

Step6 只删除检测，不改变保留下来的框、yaw、track_id 或坐标。判断依据是最终
`box_to_label` 的规范类别，因此内部 `Vehicle` 也会保留并导出为 `Car`。

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step6_car_only_filter.py \
  --step5-json work/step5_class_motion_filter/<clip>_step5.json \
  --clip work/step5_class_motion_filter/data/<clip>_step5 \
  --out-json work/step6_car_only_filter/<clip>_step6.json \
  --out-clip work/step6_car_only_filter/data/<clip>_step6
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step6_car_only_filter_batch.py --overwrite
```

## 目录说明

```text
classification/  Step2 类别精修
filtering/       Step1/Step2 可见度与硬过滤
tracking/        保守跟踪器 + 静态优先跟踪器
geometry/        Step2 yaw，Step3 Car box，Step4 Truck box
inference/       Step1 OpenPCDet 推理脚本
pipeline/        step1..step6 单条与批量入口
archive/         不再参与当前链路的旧版本/旧预览文件
tests/           当前链路的单元测试
models/          推理配置与模型权重
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
