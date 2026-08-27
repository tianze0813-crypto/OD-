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
  -> step5 最终点数/短链过滤 + box 转换到 base_link
  -> step6（可选）只保留最终标签类别 Car
  -> SUST clip
```

## 坐标系与 base_link 约定

### 当前使用的坐标系

代码使用列向量和齐次变换，约定 `p_dst = T_dst_from_src @ p_src`。当前各坐标系的
职责如下：

| 坐标系 | 代码中的来源 | 用途 | 是否写入最终 label |
| --- | --- | --- | --- |
| `lidar_top` | `lidar/lidar_top/*.bin`，OpenPCDet 原始输出 | Step1--Step4 的 `box_lidar`、点云裁剪，以及 Step5 点数统计 | 否，Step5 后 box 已转换 |
| `pose` | `transforms/pose_data.txt` 对应的局部帧 | 作为 `world_from_pose` 的输入帧；`CoordinateProvider` 的中间帧 | 否 |
| `base_link` | `transforms/calib.json` 的 `tf2base_link` | Step5 转换后的 box 和最终 label 坐标 | 是，Step5 后写入 label |
| `world` | `pose_data.txt` 的位姿输出帧 | 跟踪中心、静态车位、运动判断、yaw 稳定、Truck 中心平滑/插值 | 否 |

**当前最终结论：`label/<frame>.json` 中的 box 是 `base_link` 局部坐标，
不是 `lidar_top` 坐标，也不是 `world` 坐标。** Step5 之前内部字段仍叫
`box_lidar`，但转换后会附加 `box_frame: "base_link"`；`tracking.box_to_label()`
只把转换后的 `box_lidar` 映射到 SUST 的 `psr` 字段，不再做坐标变换：

```text
box_lidar = [x, y, z, dx, dy, dz, yaw]
label.psr.position = [x, y, z]
label.psr.scale    = [dx, dy, dz]
label.psr.rotation.z = yaw
```

`x,y,z` 是 box 中心，`dx,dy,dz` 是沿 box 局部 x/y/z 轴的长度、宽度、高度，
`yaw` 是绕 `base_link` 局部 z 轴的弧度。Step5 使用标定旋转矩阵旋转 heading
并重新计算 yaw；尺寸在刚体变换下保持不变。若外参含 roll/pitch，七参数 box
只能保存 heading 的 XY 投影，不能表达完整倾斜的 3D 姿态。
`Vehicle` 在导出时按 `CLASS_MAP` 变为 `obj_type: Car`。

### world 变换和约束

`calib.json` 中 `tf2base_link.<sensor>` 表示 `base_from_sensor`。当前
`CoordinateProvider` 使用：

```text
base_from_pose      = tf2base_link.pose
base_from_lidar_top = tf2base_link.lidar_top

world_from_lidar_top(t)
    = world_from_pose(t)
    @ inv(base_from_pose)
    @ base_from_lidar_top
```

也就是对一个 top 雷达点依次执行：

```text
lidar_top -> base_link -> pose -> world
```

`pose_data.txt` 每行只读取前 8 列，格式必须是
`timestamp_ns, x, y, z, qx, qy, qz, qw`；时间戳按纳秒解释，四元数顺序是
`x,y,z,w`。代码会在相邻位姿间对平移线性插值、对四元数 SLERP；相邻位姿间隔
超过 0.6 秒时使用较近的一帧，超出时间范围时使用端点帧。`tf2base_link` 中参与
计算的矩阵必须是有限的 4x4 矩阵。

各步骤对坐标的实际使用是：

1. Step1 从 `lidar_top` 点云推理，输出的 `box_lidar` 原样进入后续流程。
2. Step2 用 `world_from_lidar_top` 做跨帧 identity、静态/动态判断和 yaw 世界角计算，
   结果再写回 `lidar_top` 的 `box_lidar`。
3. Step3 的 Car 点云拟合、Step4 的 Truck 点云拟合都直接在 `lidar_top` 中进行；
   Truck 的中心平滑和缺失帧插值暂时在 `world` 中计算，写回前再逆变换回
   `lidar_top`。
4. Step5 在 `lidar_top` 点云中统计 box 内点数，按点数和轨迹长度过滤，然后把保留
   box 转换到 `base_link`；点云文件本身不转换。
5. Step6（可选）只删除非 Car 检测，不改变 Step5 已转换的 box。
6. `label/` 导出直接使用 Step5/Step6 的 `box_lidar`，因此最终坐标是
   `base_link` 局部帧。

### 当前可见度模块的特别说明

可见度实现按已恢复的远端版本保持不变。`filtering/camera_visibility.py` 当前构造
的是：

```text
cam_from_pose = inv(base_from_cam) @ base_from_pose
```

因此它假定传入的 box 已经在 `pose` 局部帧；它不读取 `pose_data.txt`。但本仓库的
Step1 会把 OpenPCDet 直接对 `lidar/lidar_top/*.bin` 的输出传给该模块，并没有在
Step1 中做 `lidar_top -> pose` 转换。也就是说：

- 如果当前数据的 `pose` 与 `lidar_top` 实际是同一坐标帧，这段投影可以直接使用；
- 如果两者存在平移或旋转差异，Step1 的 visibility 比例和 5% 可见度过滤可能不准，
  Step4 为插入 Truck 计算的 visibility 也有同样前提；
- 这不改变 Step5 的坐标转换规则；visibility 仍在 box 转成 `base_link` 之前计算。
  要修复输入帧契约，必须统一上游 box 的输入帧或修改可见度实现，不能只改 README。

### 更换 base_link 的影响

代码没有把 `base_link` 原点硬编码为 front 或 top 雷达；实际原点由每个 clip 的
`tf2base_link` 标定决定。Step5 使用 `tf2base_link.lidar_top` 将 box 从 top
转换到当前 base。把原点从 front 改到 top 后，应同步重算
`tf2base_link` 下所有传感器外参，并保持 `pose_data.txt` 仍表示同一个
`world_from_pose`。若 top 就是新 base 原点，通常 `tf2base_link.lidar_top`
应为单位阵（以实际标定工具输出为准），此时转换结果数值上基本不变。

不要只改 `pose` 或只改 `lidar_top`，也不要把 Step5 前的 box 提前转换后继续参与
Step2--Step4 的 `lidar_top` 点云计算；否则点云拟合、跟踪、yaw 和可见度投影会出现
偏差。若 `pose_data.txt` 的语义也改成了
`world_from_lidar_top`，必须同步改写 `CoordinateProvider` 的公式，不能继续直接
套用当前实现。

当前仓库不会转换 `lidar/lidar_top/*.bin`。因此 SUST 接入端必须确认 label 使用
`base_link`、点云使用 `lidar_top` 是否被支持；仓库本身没有在 label 中额外写入
坐标系声明。

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

Step2 的原有 hard filter 和短轨迹过滤保持不变；Step5 的点数/短链过滤是对
Step4 结果执行的独立最终闸门，两个阶段不会互相替代。

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

### Step 5：最终过滤 + box 转换到 base_link

Step5 不再按类别删除 Truck 或非机动车。它只执行两项最终过滤，然后转换保留
box 的坐标：

```text
1. box 内点数 <= 5：删除该检测
2. 轨迹长度 <= 3 帧：删除该轨迹的全部检测
3. 对剩余 box 应用 lidar_top -> base_link 的静态外参
```

点数使用原始 `lidar/lidar_top/<frame>.bin` 和转换前的 `box_lidar` 统计；点云文件
不改写。转换后的检测仍使用兼容字段名 `box_lidar`，同时写入
`box_frame: "base_link"`。

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step5_class_motion_filter.py \
  --step4-json work/step4_truck_size_interp_overlap/<clip>_step4.json \
  --clip work/step4_truck_size_interp_overlap/data/<clip>_step4 \
  --out-json work/step5_class_motion_filter/<clip>_step5.json \
  --out-clip work/step5_class_motion_filter/data/<clip>_step5
```

可调阈值（默认值为 `5` 和 `3`）：

```bash
--sparsity-max-points 5
--short-track-max-frames 3
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step5_class_motion_filter_batch.py --overwrite
```

### Step 6（可选）：只保留 Car 标签

Step6 只删除检测，不改变 Step5 已转换到 `base_link` 的框、yaw、track_id 或坐标。判断依据是最终
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
filtering/       Step1/Step2 可见度与硬过滤，Step5 最终过滤
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
