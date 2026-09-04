# 五类别 BEV 预标注流水线

这是一套面向 SUSTechPOINTS 的 LiDAR 预标注后处理链路。它读取未标注的 SUST
clip，使用五类别 VoxelNeXt 权重推理、跨帧跟踪、类别稳定和几何精修，最后生成
可直接在 SUSTechPOINTS 中打开的 `label/*.json`。

当前主链路只使用 Step1、Step2、Step2.5、Step3 和 final；旧的 Step4/Step5/Step6
已经移到 `deprecated/`，不参与五类别端到端运行。

## 批量运行（最常用）

输入 clip 的父目录即可，脚本会扫描父目录下所有 `lidar/lidar_top/*.bin` 的 clip；
下面是最常用的批量命令，指定权重并导出到 SUST：

```bash
bash run.sh /path/to/clip_parent ~/SUSTechPOINTS/data --weight epoch20
```

第二参数可以省略，默认使用 `~/SUSTechPOINTS/data`；不写 `--weight` 时使用默认权重。
导出为：

```text
~/SUSTechPOINTS/data/<clip>_pre/
└── label/<frame_id>.json
```

原始 clip 不保留过程文件，中间 JSON、诊断文件和临时 clip 全部自动清理。

## 本机使用（当前环境已就绪）

### 当前用户所用环境

当前用户 `moga` 在本机使用的 conda 环境是 **`openpcdet`**（Python 3.10.20），
解释器路径为：

```text
/home/moga/miniconda3/envs/openpcdet/bin/python
```

`run.sh` / `scripts/run_five_class.py` 会按顺序探测候选 Python（`--python`、
`OPENPCDET_PYTHON`、当前 Python，以及现成的 `openpcdet` / `sustechpoints` conda
环境），`openpcdet` 已具备 CUDA PyTorch、`spconv`、OpenPCDet 和后处理依赖，因此会被
自动选中，`run.sh` 默认 `--skip-install`，不会联网重复安装。

实测 CUDA 可用（CUDA 12.4），GPU 为 **NVIDIA RTX A4000**；运行
`scripts/check_step1_env.py` 返回 `Step1 environment check: OK`，pcdet 源码位于
`/home/moga/桌面/OpenPcdet/OD预标注/OpenPCDet`。关键版本如下：

| 组件 | 本机版本 |
| --- | --- |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu124 |
| torchvision | 0.20.1+cu124 |
| spconv-cu124 | 2.3.8 |
| pcdet | 0.6.0+0（editable source） |
| NumPy | 2.2.6 |
| SciPy | 1.15.3 |
| OpenCV | 4.13.0.92 |
| Pillow | 12.2.0 |
| Pandas | 2.3.3 |
| av2 | 0.3.6 |
| Kornia | 0.8.2 |
| Numba | 0.66.0 |
| llvmlite | 0.48.0 |
| PyYAML | 6.0.3 |
| easydict | 1.13 |
| tensorboardX | 2.6.5 |
| scikit-image | 0.25.2 |
| tqdm | 4.68.3 |
| SharedArray | 3.2.4 |
| pyquaternion | 0.9.9 |

### 一键运行（完整链路，导出到 SUST）

输入 clip 的父目录即可，脚本会扫描父目录下所有 `lidar/lidar_top/*.bin` 的 clip：

```bash
bash run.sh /media/moga/police/test <sust_data_dir> --weight epoch20
```

`--weight` 可换为 `epoch12` / `epoch15` / `epoch17` / `epoch20` / `argo2`，
`waymo` 只能配 `--mode inference`。导出为：

```text
<sust_data_dir>/<clip>_pre/
└── label/<frame_id>.json
```

要同时跑多个权重，给每个权重一个独立的输出根，例如：

```bash
bash run.sh /media/moga/police/test <sust_data_dir>/epoch12 --weight epoch12
bash run.sh /media/moga/police/test <sust_data_dir>/epoch15 --weight epoch15
```

> **导出到 SUST 是可选的，不是必须的。** `run.sh` 的第二个参数就是 SUSTechPOINTS
> 的数据根目录，路径由使用者按自己的安装来填写（本机为
> `/home/moga/桌面/SUSTechPOINTS/data`），脚本不会假设一个固定路径。若不想导出到
> SUST，可改用 `--no-export-sust` 只跑链路、不落盘，或把第二个参数指向任意临时
> 目录，事后自行拷贝。

> **调试期命名（多权重一次导出对比）。** 这一套链路会把每个 clip 的后处理结果导出为
> `<clip>_pre/`、原始推理导出为 `<clip>_raw.json`。调试期想同时平铺多个权重到同一个
> SUST 数据目录且互不覆盖时，把名字加上权重后缀，例如
> `<clip>_epoch12_pre/`、`<clip>_epoch12_raw.json`、`<clip>_epoch20_pre/`。
> **正式使用时不加权重、只保留 `_pre` 后缀**（原始推理不落盘为场景）。

### 只跑原始推理（保留 Step1 raw JSON）

```bash
bash run.sh /media/moga/police/test <sust_data_dir> \
  --mode inference --raw-output <raw_dir> \
  --weight epoch20
```

每个 clip 输出一个 `<clip>_raw.json`，字段包含 `box_lidar`、`class_name`、`score`
以及相机可见度元数据；调试期按上述约定重命名为 `<clip>_<weight>_raw.json`。原始推理
使用最低类别阈值，不做范围 / 点数 / 可见度硬过滤。

### 阈值与自检

`--score-thresh` 会覆盖五类阈值，未传时使用默认值：

| 类别 | 默认阈值 |
| --- | ---: |
| Car | 0.25 |
| Truck | 0.40 |
| Bus | 0.40 |
| Pedestrian | 0.30 |
| Nonmotorized_vehicle | 0.30 |

环境自检：

```bash
$OPENPCDET_PYTHON scripts/check_step1_env.py \
  --cfg models/voxelnext_fiveclass_nuscenes_infer.yaml \
  --ckpt models/vn5_nuscenes_checkpoint_epoch_12.pth
```

## 权重选择

默认权重是 `models/vn5_nuscenes_checkpoint_epoch_12.pth`，可用
`bash run.sh --list-weights` 查看：

| 别名 | checkpoint | 可用模式 |
| --- | --- | --- |
| `default` / `epoch12` | `vn5_nuscenes_checkpoint_epoch_12.pth` | 完整链路 |
| `epoch15` | `nusc_frozen20_epoch15.pth` | 完整链路 |
| `epoch17` | `nusc_frozen20_epoch17.pth` | 完整链路 |
| `epoch20` | `nusc_frozen20_epoch20.pth` | 完整链路 |
| `argo2` | `argo2_protected_epoch6.pth` | 完整链路 |
| `waymo` | `vn_waymo_v2_4gpu_full_epoch10.pth` | 仅 `--mode inference` |

`waymo` 是三类头 checkpoint，只能跑原始推理；其他别名都支持完整链路。也可以直接
用 `--ckpt /path/to/model.pth` 指定任意权重。

## 从零配置（新机器）

全新机器可按下面步骤手动搭建，也可以直接用一键入口自动安装。运行链路分两部分：

- **Step1 推理**：需要 CUDA 版 PyTorch、匹配 CUDA 的 `spconv`、OpenPCDet 源码
  (`pcdet`)、相机可见度依赖。
- **Step2/Step2.5/Step3/final**：使用标准 Python 数值和图像库，见
  [requirements-step1.txt](requirements-step1.txt)。

### 手动搭建

1. 安装 Miniconda/Anaconda，并准备好 NVIDIA 驱动 + CUDA 12.4（与 torch 匹配）。
2. 创建并激活 Python 3.10 环境：

```bash
conda create -n openpcdet python=3.10 pip -y
conda activate openpcdet
```

3. 安装 CUDA 版 PyTorch 2.5.1 与匹配的 spconv：

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install spconv-cu124==2.3.8
```

4. 安装后处理依赖：

```bash
pip install -r requirements-step1.txt
```

5. 安装 OpenPCDet（源码目录须含 `pcdet/` 与 `setup.py`）：

```bash
git clone --depth 1 https://github.com/open-mmlab/OpenPCDet.git
pip install -e ./OpenPCDet
```

6. 自检：

```bash
python scripts/check_step1_env.py \
  --cfg models/voxelnext_fiveclass_nuscenes_infer.yaml \
  --ckpt models/vn5_nuscenes_checkpoint_epoch_12.pth
```

### 自动入口

`scripts/run_five_class.py` / `scripts/run_five_class.sh` 会：

1. 探测 `--python`、`OPENPCDET_PYTHON`、现成 `openpcdet`/`sustechpoints` 环境及当前 Python。
2. 找到已具备依赖的环境直接使用；否则创建 `fiveclass-prelabel`（Python 3.10）。
3. 自动安装 `requirements-step1.txt`、torch 2.5.1、`spconv-cu124==2.3.8`。
4. 找不到 OpenPCDet 源码时克隆官方仓库。
5. 运行 `check_step1_env.py`，检查通过后才开始推理。

```bash
bash scripts/run_five_class.sh ~/sust/data ~/SUSTechPOINTS/data
```

可选环境变量：

```bash
export OPENPCDET_PYTHON=/path/to/python
export OPENPCDET_ROOT=/path/to/OpenPCDet
export OPENPCDET_REPO=https://github.com/open-mmlab/OpenPCDet.git
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124
```

已有 OpenPCDet 环境时推荐显式指定：

```bash
OPENPCDET_PYTHON=/path/to/openpcdet/bin/python \
bash scripts/run_five_class.sh ~/sust/data ~/SUSTechPOINTS/data
```

新环境按 `requirements-step1.txt` 安装的版本与本机 `openpcdet` 环境不完全一致（例如
NumPy 1.26.4、Kornia 0.6.12），两者都能跑通当前链路。Step1 要求
`torch.cuda.is_available()` 为真；没有 NVIDIA GPU 时只能测后处理，不能跑模型推理。
模型权重不会自动下载，须放进 `models/` 或用 `--ckpt` 指定。

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
run.sh            本机依赖就绪时的一键端到端入口
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
