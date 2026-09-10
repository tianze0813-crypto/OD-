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

中间 JSON 默认写入系统临时目录，跑完自动删除；加 `--keep-intermediate`
可把它们保留在 `--work-root`（默认 `work/end_to_end`）下，便于逐步归因。

端到端脚本的有效链路如下，Step4 已完成 Car-only，Step4.5 只在动态区域内
重做 only-car 跟踪 / ID 继承 / 方向级相位拼接，Step5 做最终过滤与导出：

```text
原始 clip
  -> step1  lidar 推理 + 相机可见度预过滤
  -> step2  identity / class / hard-filter / yaw（动态段保留 detector yaw）
  -> step3  Car box XY 拟合 + 地面/车顶 Z 拟合
  -> step4  Car→Truck 尺寸闸门，然后只保留 Car
  -> step4.5 动态区域（swept box + 稳定方向 30m 延伸，无 buffer）
              -> 区域外静态 ID / box 冻结
              -> 区域内运动-only 重跟踪
              -> ID 继承 + 槽位释放检查
              -> 方向级四相位感知拼接
              -> 动态段第二遍 box fit
  -> step5 最终点数/短链过滤 + Car-only 兜底 + box 转换到 base_link
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
| `world` | `pose_data.txt` 的位姿输出帧 | 跟踪中心、静态车位、运动判断和 yaw 稳定 | 否 |

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
3. Step3 的 Car box 拟合直接在 `lidar_top` 中进行；先确定最终 XY footprint，
   再在该 footprint 内自下而上寻找连续车顶截面。
4. Step4 只读取 Step3 的 box 尺寸；轨迹稳健长度达到 `6m` 的 Car 统一改标为
   `Truck`，不改变 box、点云、yaw 或坐标。
5. Step5 在 `lidar_top` 点云中统计 box 内点数，按点数和轨迹长度过滤，然后把保留
   box 转换到 `base_link`；点云文件本身不转换。
6. Step5 在转换前固定只保留规范类别 `Car`，再将 box 转到 `base_link`；
   因此 Step6 不再是端到端链路中的必要步骤。
7. `label/` 导出直接使用 Step5 的 `box_lidar`，因此最终坐标是
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
Step3 结果执行的独立最终闸门，两个阶段不会互相替代。

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

Step3 会先完成 shrink-only XY 拟合和静态轨迹尺寸平滑，再用最终 XY footprint
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

### Step 4：Car → Truck 尺寸闸门 + Car-only

Step4 不做点云拟合或坐标变换，顺序固定：

```text
1. 只检查 class_name == Car 的检测
2. 计算每条 track 的 max(dx, dy) 中位数
3. 中位数 >= 6.0m：该 track 的 Car 全部改为 Truck
4. 删除所有规范类别不是 Car 的检测（Truck / Bus / Pedestrian /
   Nonmotorized_vehicle 等）
```

只有 Car 进入 Step4.5。

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_car_size_filter.py \
  --step3-json work/step3/<clip>_step3.json \
  --out-json work/step4/<clip>_step4.json
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_car_size_filter_batch.py --overwrite
```

### Step 4.5：动态区域 / 重跟踪 / ID 继承 / 相位拼接

Step4.5 是全链路唯一会修改 Car ID 的阶段，原则是**区域外静态完全冻结**：

```text
1. 用 Step4 的 car-only 轨迹（高速证据：>=5 m/s、>=15 m、>=3 帧）
   建动态区域；只用 swept box，无 buffer；
   沿稳定方向在轨迹两端各延伸 30m（覆盖停止线 / 排队 / 起步段）
2. 区域外 / 无高速证据的检测保持 Step4 的 ID 和 box 不变
3. 只在动态区域内对候选检测做运动-only 关联（关联不用 yaw）
   并保留 reverse / 速度 / 加速度 / 距离物理门限
4. ID 继承：静态锚 ID > 高速主轨迹旧 ID > 新 ID；
   多合一按观测数投票；同帧不重复；一个旧 ID 不主动拆成两条
5. 槽位释放：旧车明确离开 / 物理连续驶离后才允许新车继承 slot ID；
   没有证据时宁可不继承，也不跨车复用
6. 用重跟踪后的轨迹建方向级四相位
   （右转常绿、掉头按左转、同轴直行/左转互斥）
7. 相位感知拼接：waiting_red / 绿灯起步 / 右转 yielding 的碎片按方向、
   停止线、空间桥接和物理连续性保守合并
8. 只对动态 / 重跟踪段再跑一遍 box fit；静态冻结段保留 Step3 结果
9. 最终 ID 定下来后，用最终运动轨迹（首->末观测方向）直接比较每个
   detection 的 yaw；偏离超过 90° 的单帧自己加 pi，即选择离最终轨迹更近的
   π 等价表示，不做中位数投票
10. step4.5 专用横向跳变保护：关联候选偏离最近运动轴 >2.5m 直接拒绝；
    同一门限同时用于 ID 继承 / queue / phase 拼接，防止拆开的横向跳变
    又被合回；step2 默认关闭
```

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_5_region_phase_retrack.py \
  --step4-json work/step4/<clip>_step4.json \
  --clip /path/to/<clip> \
  --step2-diagnostics work/step2/<clip>_step2_diagnostics.json \
  --out-json work/step4_5/<clip>_step45.json \
  --diagnostics work/step4_5/<clip>_step45_diagnostics.json
```

批量：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step4_5_region_phase_retrack_batch.py --overwrite
```

诊断字段包括 `dynamic_region` / `selection` / `retracking` /
`id_inheritance` / `phase_stitching` / `box_fit` / `yaw_reversal` /
`static_freeze`。
`static_freeze.passed=False` 会直接报错，避免误改静态。

### Step 5：最终过滤 + box 转换到 base_link

Step5 先执行两项终检，再做 Car-only 兜底（Step4 后通常为空），最后转换保留
box 的坐标：

```text
1. box 内点数 <= 5：删除该检测
2. 轨迹长度 <= 3 帧：删除该轨迹的全部检测
3. 删除规范类别不是 Car 的检测
4. 对剩余 box 应用 lidar_top -> base_link 的静态外参
```

点数使用原始 `lidar/lidar_top/<frame>.bin` 和转换前的 `box_lidar` 统计；点云文件
不改写。转换后的检测仍使用兼容字段名 `box_lidar`，同时写入
`box_frame: "base_link"`。

单条：

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python pipeline/step5_class_motion_filter.py \
  --step4-json work/step4_5/<clip>_step45.json \
  --clip /path/to/<clip> \
  --out-json work/step5/<clip>_step5.json \
  --out-clip work/step5/data/<clip>_step5
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

Step5 已固定执行同样的 Car-only 逻辑，因此端到端流程不再调用 Step6。下面的脚本
仍保留给旧的 Step5 JSON 使用；它只删除检测，不改变已经转换到 `base_link` 的框、
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
filtering/       Step1/Step2 可见度与硬过滤，Step4/Step5 最终类别与点数过滤
tracking/        保守跟踪器 + 静态优先跟踪器
geometry/        Step2 yaw，Step3 Car box 与地面/车顶拟合
region/          动态区域、区域 mask、方向级四相位、step4.5 重跟踪/ID 继承
inference/       Step1 OpenPCDet 推理脚本
pipeline/        step1、step2、step3、step4、step4.5、step5 主链路；step6 为兼容入口
archive/         不再参与当前链路的旧版本/旧预览文件
tests/           当前链路的单元测试
models/          推理配置与模型权重
```

## 测试

```bash
/home/moga/miniconda3/envs/sustechpoints/bin/python -m unittest discover -s tests -v
```
