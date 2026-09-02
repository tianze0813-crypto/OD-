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
中间 `_step2/_step3` clip。

端到端脚本的有效链路如下。当前分支针对五类模型，Step4/Step5 不再参与端到端链路：

```text
原始 clip
  -> step1  lidar 推理 + 相机可见度预过滤
  -> step2  identity / class / hard-filter / yaw
  -> step3  多类别轨迹几何稳定 + Car 点云 XY/Z 精拟合
  -> final 五类规范化 + box 转换到 base_link
  -> SUST clip

模型输出类别固定为：`Car`、`Truck`、`Bus`、`Pedestrian`、
`Nonmotorized_vehicle`。类别在 Step2 完成 ID 后按轨迹多数票稳定，不再按
box 长度把 Car/Truck/Bus 互相改标。
```

## 坐标系与 base_link 约定

### 当前使用的坐标系

代码使用列向量和齐次变换，约定 `p_dst = T_dst_from_src @ p_src`。当前各坐标系的
职责如下：

| 坐标系 | 代码中的来源 | 用途 | 是否写入最终 label |
| --- | --- | --- | --- |
| `lidar_top` | `lidar/lidar_top/*.bin`，OpenPCDet 原始输出 | Step1--Step3 的 `box_lidar`、点云裁剪与几何拟合 | 否，final 后 box 已转换 |
| `pose` | `transforms/pose_data.txt` 对应的局部帧 | 作为 `world_from_pose` 的输入帧；`CoordinateProvider` 的中间帧 | 否 |
| `base_link` | `transforms/calib.json` 的 `tf2base_link` | final 转换后的 box 和最终 label 坐标 | 是，final 后写入 label |
| `world` | `pose_data.txt` 的位姿输出帧 | 跟踪中心、静态车位、运动判断和 yaw 稳定 | 否 |

**当前最终结论：`label/<frame>.json` 中的 box 是 `base_link` 局部坐标，
不是 `lidar_top` 坐标，也不是 `world` 坐标。** final 之前内部字段仍叫
`box_lidar`，但转换后会附加 `box_frame: "base_link"`；`tracking.box_to_label()`
只把转换后的 `box_lidar` 映射到 SUST 的 `psr` 字段，不再做坐标变换：

```text
box_lidar = [x, y, z, dx, dy, dz, yaw]
label.psr.position = [x, y, z]
label.psr.scale    = [dx, dy, dz]
label.psr.rotation.z = yaw
```

`x,y,z` 是 box 中心，`dx,dy,dz` 是沿 box 局部 x/y/z 轴的长度、宽度、高度，
`yaw` 是绕 `base_link` 局部 z 轴的弧度。final 阶段使用标定旋转矩阵旋转 heading
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
3. Step3 先对五类轨迹做保守几何稳定，再对 Car 在 `lidar_top` 中进行
   shrink-only XY、地面/车顶 Z 精拟合。
4. final 阶段保留五类检测，不执行旧 Step4 的尺寸改标、旧 Step5 的
   Car-only、点数和短链闸门；仅校验类别/box 并转换到 `base_link`。
5. `label/` 导出直接使用 final 阶段的 `box_lidar`，因此最终坐标是
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
- 如果两者存在平移或旋转差异，Step1 的 visibility 比例和 5% 可见度过滤可能不准；
- visibility 仍在 box 转成 `base_link` 之前计算；final 阶段只做坐标转换。
  要修复输入帧契约，必须统一上游 box 的输入帧或修改可见度实现，不能只改 README。

### 更换 base_link 的影响

代码没有把 `base_link` 原点硬编码为 front 或 top 雷达；实际原点由每个 clip 的
`tf2base_link` 标定决定。final 阶段使用 `tf2base_link.lidar_top` 将 box 从 top
转换到当前 base。把原点从 front 改到 top 后，应同步重算
`tf2base_link` 下所有传感器外参，并保持 `pose_data.txt` 仍表示同一个
`world_from_pose`。若 top 就是新 base 原点，通常 `tf2base_link.lidar_top`
应为单位阵（以实际标定工具输出为准），此时转换结果数值上基本不变。

不要只改 `pose` 或只改 `lidar_top`，也不要把 final 前的 box 提前转换后继续参与
Step2--Step3 的 `lidar_top` 点云计算；否则点云拟合、跟踪、yaw 和可见度投影会出现
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
默认使用 `models/vn5_nuscenes_checkpoint_epoch_12.pth` 与
`models/voxelnext_fiveclass_nuscenes_infer.yaml`；如部署环境不同，可通过
`--cfg` 和 `--ckpt` 覆盖。

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

Step2 保留分数、范围、可见度和短轨迹等基础标注过滤；类别使用五类模型结果，
按 track 多数票稳定，不执行旧的 Vehicle/Car 尺寸重分类。

### Step 3：多类别几何与 Car 精拟合

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

如需对已有 Step3 JSON 单独执行五类 final 适配：

```bash
python pipeline/five_class_postprocess.py \
  --input-json work/step3_car_box_fit/<clip>_step3.json \
  --clip /path/to/scene_clip \
  --output-json work/final/<clip>_final.json
```

Step3 会先对五类轨迹完成保守几何稳定和静态轨迹尺寸平滑，再对 Car 使用
经过验证的 shrink-only XY/地面/车顶拟合，并用最终 XY footprint
重新裁点。车顶搜索从当前地面边界（无地面证据时从原 box 下边界）开始，以 `5cm`
步长检查重叠的 `10cm` 水平截面。截面必须同时满足点数、长短轴跨度、中心覆盖、稳健
中心对齐和 `6 x 4` 网格二维连通约束；至少连续两个窗口成立，并且上方 `5cm` 不再
连续时，才把该截面认定为车顶。中心覆盖和稳健中心对齐只参与车顶的 Z 证据判定，
不改变现有 XY 拟合。更高但狭窄、不连通或整体偏到 box 一侧的树枝噪点不会直接决定
上边界。

最终 Z 仍严格保留原来的双边界与 track 高度回退规则：

```text
地面和车顶都有：高度合理时使用两个边界；不合理时保留地面并套用 track 高度
只有地面：保留地面，向上套用 track 高度
只有车顶：保留车顶，向下套用 track 高度
两个都没有：保留原 z 中心，只替换为 track 高度

另外，两个边界都存在但车顶连续窗口较短、且拟合高度偏离该 track 的稳健高度先验
时，会按“保留地面 + track 高度”回退，避免把车身/挡风玻璃层当作车顶。地面估计
本身不改变；只有单条 Car track 出现明显双峰并反复突跳时才修正：中间异常段用前后
可信地面帧做时序插值，轨迹开头或结尾连续达到最小聚类长度的异常段可使用唯一低地面
锚点。短边缘段和普通单次跳变保持原值。
```

### 旧 Step 4/Step 5（兼容脚本）

`pipeline/step4_*` 和 `pipeline/step5_*` 保留给旧 Car 流程回放，但不再由
`run_end_to_end.py` 调用。五类端到端 final 阶段不做尺寸改标、Car-only、
点数或短链过滤，仅执行类别/box 校验和坐标转换。

旧 Step4 的行为仍是：

```text
1. 只检查 class_name == Car 的检测
2. 计算每条 track 的 max(dx, dy) 中位数
3. 中位数 >= 6.0m：该 track 的 Car 全部改为 Truck
4. 其他检测和 box 字段保持不变
```

该旧行为不适用于五类模型。

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_car_size_filter.py \
  --step3-json work/step3_car_box_fit/<clip>_step3.json \
  --out-json work/step4_car_size_filter/<clip>_step4.json
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_car_size_filter_batch.py --overwrite
```

旧 Step5：最终过滤 + Car-only + box 转换到 base_link

仅旧流程固定只保留最终规范类别为 `Car` 的检测。五类流程请使用
`pipeline/five_class_postprocess.py` 或端到端入口内置的 final 阶段。

```text
1. box 内点数 <= 5：删除该检测
2. 轨迹长度 <= 3 帧：删除该轨迹的全部检测
3. 删除规范类别不是 `Car` 的检测（内部 `Vehicle` 映射为 `Car`，会保留）
4. 对剩余 box 应用 lidar_top -> base_link 的静态外参
```

点数使用原始 `lidar/lidar_top/<frame>.bin` 和转换前的 `box_lidar` 统计；点云文件
不改写。转换后的检测仍使用兼容字段名 `box_lidar`，同时写入
`box_frame: "base_link"`。

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step5_class_motion_filter.py \
  --step4-json work/step4_car_size_filter/<clip>_step4.json \
  --clip work/step3_car_box_fit/data/<clip>_step3 \
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

### 已有 base_link 检测直接运行 Step3

如果输入 clip 的 `label/*.json` 已经是 `base_link` 坐标，且希望跳过模型推理、只执行
Step2 + Step3，可以使用专用适配入口。它会临时把 box 逆变换到 `lidar_top`，先重新
建立跨帧 track，再做点云拟合，最后转回 `base_link` 写入输出 SUST clip，不执行 Step5：

```bash
python pipeline/step3_base_link_sust.py \
  --clip-dir /media/zhu/GEN2/test1 \
  --output-root /home/zhu/桌面/sust/data \
  --overwrite
```

该入口只处理存在非空 `lidar/lidar_top/*.bin`、标定和 label 的 clip；空 clip 会被
报告为无效输入。

### Step 6（兼容脚本）：只保留 Car 标签

端到端流程不再调用 Step6。下面的脚本仍保留给旧的 Car-only JSON 使用；它只删除检测，不改变已经转换到 `base_link` 的框、
yaw、track_id 或坐标。

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
filtering/       Step1/Step2 可见度与硬过滤，final 五类输出与坐标转换
tracking/        保守跟踪器 + 静态优先跟踪器
geometry/        Step2 yaw，Step3 多类别几何与 Car 地面/车顶精拟合
inference/       Step1 OpenPCDet 推理脚本
pipeline/        step1、step2、step3、step4、step5 主链路；step6 为兼容入口
archive/         不再参与当前链路的旧版本/旧预览文件
tests/           当前链路的单元测试
models/          推理配置与模型权重
```

## 测试

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python -m unittest discover -s tests -v
```
