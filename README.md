# 五类别 BEV 预标注流水线

这是一套面向 SUSTechPOINTS 的 LiDAR 预标注后处理链路。它读取未标注的 SUST
clip，使用五类别 VoxelNeXt 权重推理、跨帧跟踪、类别稳定和几何精修，最后生成
可直接在 SUSTechPOINTS 中打开的 `label/*.json`。

当前主链路只使用 Step1、Step2、Step2.5、Step3 和 final；旧的 Step4/Step5/Step6
已经移到 `deprecated/`，不参与五类别端到端运行。

## 一键运行

在另一台机器上把仓库和模型文件准备好后，默认只需要执行：

```bash
cd 五类别预标链路
bash scripts/run_five_class.sh
```

默认路径是：

```text
输入：~/sust/data
输出：~/SUSTechPOINTS/data
```

也可以显式传入两个路径，分别表示输入目录和输出目录：

```bash
bash scripts/run_five_class.sh \
  ~/sust/data \
  ~/SUSTechPOINTS/data
```

只跑原始 Step1 推理（不跟踪、不做后处理）时使用第二个入口：

```bash
bash scripts/run_raw_inference.sh
```

它默认把每个 clip 的原始结果保存为 `~/sust/raw_inference/<clip>_raw.json`，默认
推理阈值为 `0.1`；可用 `--raw-output`、`--score-thresh` 和 `--overwrite` 调整。

两种入口都只保留最终产物。完整链路的 Step1/Step2/Step2.5/Step3 JSON、诊断文件和
临时 clip 都写入系统临时目录，进程结束后自动删除；原始推理模式也只把 raw JSON
复制到 `--raw-output`，不会在输入 clip 下生成旁路文件。

输入目录下面的每个子目录都应是一个原始 clip：

```text
~/sust/data/<clip>/
├── lidar/lidar_top/*.bin
├── transforms/calib.json
├── transforms/pose_data.txt
└── camera/                  # 可选，按当前 SUST clip 结构提供
```

输出目录会生成：

```text
~/SUSTechPOINTS/data/<clip>_pre/
└── label/<frame_id>.json
```

一键脚本默认使用 `models/vn5_nuscenes_checkpoint_epoch_12.pth`。输入目录中的原始
clip 不会被改名或写入，结果以 `<clip>_pre` 的副本形式输出，便于反复调试。

常用选项：

```bash
# 覆盖已经存在的输出
bash scripts/run_five_class.sh ~/sust/data ~/SUSTechPOINTS/data --overwrite

# 只检查/安装环境，不运行推理
bash scripts/run_five_class.sh --check-only

# 已经准备好环境时，禁止自动安装
bash scripts/run_five_class.sh --skip-install

# 使用其他五类别权重
bash scripts/run_five_class.sh \
  ~/sust/data ~/SUSTechPOINTS/data \
  --ckpt models/nusc_frozen20_epoch20.pth

# 临时让全部类别使用同一个阈值
bash scripts/run_five_class.sh \
  ~/sust/data ~/SUSTechPOINTS/data \
  --score-thresh 0.3

# 完整链路只跑验证，不导出最终 clip 到 SUST
bash scripts/run_five_class.sh \
  ~/sust/data ~/SUSTechPOINTS/data \
  --no-export-sust
```

`--score-thresh` 是兼容参数，会覆盖五个类别的阈值。未传该参数时使用当前默认值：

| 类别 | 默认阈值 |
| --- | ---: |
| Car | 0.25 |
| Truck | 0.40 |
| Bus | 0.40 |
| Pedestrian | 0.30 |
| Nonmotorized_vehicle | 0.30 |

## 自动环境处理

入口是 `scripts/run_five_class.py`，Shell 文件只是一个方便调用的包装器。脚本按下面
顺序处理运行环境：

1. 优先检查 `--python`、`OPENPCDET_PYTHON`、已有 `openpcdet`/`sustechpoints` conda
   环境，以及当前 Python。
2. 如果找到已经具备 CUDA PyTorch、`spconv`、OpenPCDet 和后处理依赖的环境，直接使用，
   不重复安装。
3. 如果没有可用环境且系统安装了 conda，自动创建 `fiveclass-prelabel`（Python 3.10）。
4. 自动安装 `requirements-step1.txt`、CUDA 版 PyTorch 2.5.1、`spconv-cu124==2.3.8`。
5. 如果找不到 OpenPCDet 源码，默认克隆官方仓库；也可以提前设置
   `OPENPCDET_ROOT=/path/to/OpenPCDet`，避免使用自动克隆。
6. 调用 `scripts/check_step1_env.py` 检查 CUDA、五类别配置、checkpoint head 数量和关键
   Python 模块，检查通过后才开始推理。

自动安装需要网络、pip/conda 权限和 NVIDIA 驱动。没有 NVIDIA GPU 时，后处理代码仍可
单独测试，但 `inference/run_prelabel.py` 会因模型调用 `.cuda()` 而无法完成推理。模型
权重不会自动下载，必须存在于仓库 `models/` 或通过 `--ckpt` 指定。

可选环境变量：

```bash
export OPENPCDET_PYTHON=/path/to/python
export OPENPCDET_ROOT=/path/to/OpenPCDet
export OPENPCDET_REPO=https://github.com/open-mmlab/OpenPCDet.git
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
```

如果另一台机器已经有可用的 OpenPCDet 环境，推荐直接运行：

```bash
OPENPCDET_PYTHON=/path/to/openpcdet/bin/python \
bash scripts/run_five_class.sh ~/sust/data ~/SUSTechPOINTS/data
```

## 当前全链路

```text
原始 SUST clip
  -> Step1: LiDAR 推理 + 相机可见度 metadata
  -> Step1 后置类别分数过滤
  -> Step2: 类别无关 identity 跟踪 + 第一遍硬过滤 + 同中心去重
  -> Step2.5: 按 track_id 类别投票 + 第二遍硬过滤 + 短轨迹过滤
  -> Step3: 公共 yaw + Car 专用贴合 + 其他四类通用几何 + Truck/NMV 分治精修
  -> final: 五类规范化 + lidar_top -> base_link
  -> <clip>_pre/label/*.json
```

### Step1：LiDAR 推理

`pipeline/step1_lidar_inference.py` 调用 `inference/run_prelabel.py`，读取
`lidar/lidar_top/*.bin` 的 XYZI 点云，输出：

```text
box_lidar = [x, y, z, dx, dy, dz, yaw]
class_name
score
```

同时按所有相机计算 box 的可见度、遮挡比例和截断比例，并将结果写入
`det['visibility']`。Step1 不因为可见度删除检测，保证 identity 阶段能看到完整的模型
输出。

推理使用所有类别中最低的阈值获取候选。随后的一次轻量分数过滤在 Step2 之前执行，按
类别删除低于各自阈值的候选。它只做分数过滤，不做范围、点数或可见度过滤。

### Step2：类别无关 ID 跟踪和第一遍过滤

`pipeline/step2_identity.py` 使用 `tracking/tracker_conservative.py`，跟踪时不把类别
作为硬约束，因为同一目标在不同帧可能被模型识别成不同类别。

跟踪依据包括世界坐标中心、运动预测、速度、协方差、尺寸和 BEV IoU。静止车辆会建立
停车位 anchor，动态车辆会按运动轨迹关联，轨迹间也会做保守拼接。Step2 只添加
`track_id`，不会在跟踪阶段修改 box 的几何字段。

第一遍硬过滤包括：

- 无效 box、未知类别和类别白名单；
- 类别分数阈值；
- 标注范围：横向 `|x| <= 40m`，前后范围 `-80m <= y <= 20m`；
- 点云框内点数 `<= 5`；
- 相机可见度 `<= 0.05`；
- 距离自车超过 `20m` 的 Pedestrian；
- 同帧中心距离 `<= 0.35m` 的重复框。

### Step2.5：按 ID 修正类别

`pipeline/step2_5_class_correction.py` 在 ID 已经稳定后处理类别。它先将别名归一化到：

```text
Car, Truck, Bus, Pedestrian, Nonmotorized_vehicle
```

然后以同一个 `track_id` 的全部检测为一组，用出现次数最多的类别作为该轨迹的最终类别，
并把该 ID 的所有检测统一为这个类别。票数相同时，使用类别分数总和做平票处理。

这一步只修改 `class_name`，保护 `track_id`、`box_lidar`、框数量和帧结构。它不使用 box
尺寸把 Bus 改成 Truck，也不把非机动车改成 Car。类别稳定后会再跑一遍硬过滤，最后删除
生命周期不足的轨迹（默认观测帧数 `<= 3` 删除）。

### Step3：公共 yaw 和类别精修

Step3 先对五类执行公共 yaw：动态目标使用运动方向，静止目标使用多帧稳定方向和点云
轴，行人使用独立的两帧方向策略。公共 yaw 只修改 `box_lidar[6]`。

随后按类别分治：

**Car** 保留原有几何精修：

- 公共 yaw 后直接进入 Car 专用贴合，不经过通用轨迹几何平滑；
- 使用点云可见车面做 shrink-only XY 贴合，不因噪点盲目扩大 box；
- 静止车只在证据不足的轴上使用轨迹稳健尺寸；
- 使用地面点拟合底部，使用连续车顶点云拟合顶部；
- 证据不足时回退到轨迹高度和输入中心。

Car 几何只修改 `box_lidar[0:6]`，不修改 yaw、ID、类别或框是否存在。

**其他四类通用几何** 对 Truck、Bus、Pedestrian 和
Nonmotorized_vehicle 执行轨迹级尺寸稳定、静态/动态中心平滑和地面 Z 修正。Car
明确排除在这个通用阶段之外。

**Truck** 处理重复和重叠：

- 同帧高 BEV IoU 重叠；
- 连续多帧中度重叠；
- 两个 Truck 过近；
- Truck 与 Car 高 IoU 重复。

合并后以 Truck 代表 ID，Car 重复观测可以转成 Truck，同一帧的重复框删除。

**Nonmotorized_vehicle** 处理大小不一致：

- 按轨迹计算稳健的中位物理尺寸；
- 小框优先保留检测中心，再补齐统一尺寸；
- 大框同时修正中心和尺寸；
- 用修正后的中心轨迹重新更新 yaw。

Bus 和 Pedestrian 在 Step3 经过公共 yaw 和通用几何，不执行 Truck/NMV 的专项规则。

### final：规范化和坐标转换

`filtering/five_class_output.py` 不再执行旧 Step5 的点数过滤、短轨迹过滤或 Car-only
过滤，只做：

- 五类别名称规范化；
- 无效类别和无效 box 校验；
- `lidar_top -> base_link` 的中心和 heading 转换；
- 写入 `box_frame: "base_link"`。

最终 SUST label 的字段为：

```text
obj_id = track_id
obj_type = 五类之一
psr.position = [x, y, z]
psr.scale = [dx, dy, dz]
psr.rotation.z = yaw
```

## 坐标系和标定

Step1 到 Step3 的 `box_lidar` 和点云都在 `lidar_top` 局部坐标中。跟踪和 yaw 使用：

```text
world_from_lidar_top
  = world_from_pose
  @ inverse(tf2base_link.pose)
  @ tf2base_link.lidar_top
```

final 才通过 `tf2base_link.lidar_top` 转成 `base_link`，所以最终 `label/*.json` 使用
的是 `base_link` 坐标，而不是 `lidar_top` 或 `world`。

每个输入 clip 必须带有自己的：

```text
transforms/calib.json
transforms/pose_data.txt
```

标定文件不会被一键脚本隐式覆盖。若要切换标定，应在每个输入 clip 的
`transforms/calib.json` 中准备正确的 `tf2base_link`；不要把已经转成 `base_link` 的 box
提前送回 Step2/Step3 做点云拟合。

## 手动调试

环境检查：

```bash
python3 scripts/check_step1_env.py \
  --cfg models/voxelnext_fiveclass_nuscenes_infer.yaml \
  --ckpt models/vn5_nuscenes_checkpoint_epoch_12.pth
```

分步执行时，建议使用同一个已通过检查的 Python：

```bash
$OPENPCDET_PYTHON pipeline/step1_lidar_inference.py \
  --clip /path/to/scene_clip \
  --work-root work/step1

$OPENPCDET_PYTHON pipeline/step2_identity.py \
  --in-json work/step1/<clip>_raw.json \
  --clip /path/to/scene_clip \
  --out-json work/step2/<clip>_step2.json \
  --diagnostics work/step2/<clip>_step2_diagnostics.json

$OPENPCDET_PYTHON pipeline/step2_5_class_correction.py \
  --step2-json work/step2/<clip>_step2.json \
  --step2-diagnostics work/step2/<clip>_step2_diagnostics.json \
  --clip /path/to/scene_clip \
  --out-json work/step2_5/<clip>_step2_5.json \
  --diagnostics work/step2_5/<clip>_step2_5_diagnostics.json

$OPENPCDET_PYTHON pipeline/step3_refinement.py \
  --step2-5-json work/step2_5/<clip>_step2_5.json \
  --step2-5-diagnostics work/step2_5/<clip>_step2_5_diagnostics.json \
  --clip /path/to/scene_clip \
  --out-json work/step3/<clip>_step3.json \
  --diagnostics work/step3/<clip>_step3_diagnostics.json
```

端到端入口也可以直接调用，`--preserve-input` 适合调试机批处理：

```bash
$OPENPCDET_PYTHON run_end_to_end.py \
  --clip-dir ~/sust/data \
  --sust-root ~/SUSTechPOINTS/data \
  --inference-python "$OPENPCDET_PYTHON" \
  --post-python "$OPENPCDET_PYTHON" \
  --export-sust --preserve-input --overwrite
```

## 目录和测试

```text
classification/  Step2.5 类别归一化和 track 投票
filtering/       可见度、硬过滤、final 五类输出
tracking/        类别无关跟踪、坐标变换、SUST label 映射
geometry/        yaw、Car 几何、Truck/NMV 精修
inference/       OpenPCDet LiDAR 推理
pipeline/        当前 Step1/Step2/Step2.5/Step3 主链路
deprecated/      旧 Car-only Step2/Step3/Step4/Step5/Step6
models/          配置和 checkpoint
scripts/         环境检查与一键入口
tests/           单元测试
```

运行后处理单元测试：

```bash
python3 -m unittest discover -s tests -p 'test_*.py'
```

当前仓库测试覆盖跟踪、类别投票、硬过滤、Car 几何、Truck/NMV 精修、坐标转换和 SUST
输出契约。
