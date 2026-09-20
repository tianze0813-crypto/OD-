# bevfusion/ —— Truck 链用到的 BEVFusion（mmdet3d 1.x）工具箱

被 `pipeline/step1_bevfusion_truck.py` 调用；本目录只放**代码/配置**，数据与缓存都在 `data/`、`work/`（已 gitignore）。

```
configs/
  police_bevfusion_mmdet3d.py            C+L 推理配置（继承官方 lidar-cam 配置，4 路相机 + 384x512）
  police_bevfusion_mmdet3d_lidaronly.py  纯雷达推理配置（默认使用）
  upstream/                              上游基础配置（从 mmdetection3d projects/BEVFusion/configs 复制，Apache-2.0）
  mmdet_base/default_runtime.py          上游 default_runtime.py（同上）
scripts/
  build_bev_pool.py    编 mmdet3d BEVFusion 需要的两个 CUDA 算子（bev_pool_ext / voxel_layer，JIT）
  prep_data.py         鱼眼去畸变 + lidar_top 5 列 bin + infos（--no-images 跳过去畸变）
  mmdet3d_prep.py      生成 mmdet3d middle-format infos
  infer_mmdet3d.py     mmdet3d BEVFusion 推理 -> raw json（lidar 系，10 类，z 统一成框中心）
  report_truck.py      卡车类数量/长度分布汇总
  eval_vs_ref.py       与参考标注做召回/精度/长度偏差
  diag_z.py            诊断 z 系统性偏差（底面 vs 中心）
  bench_speed.py       每帧数据/前向耗时拆解
  to_sust.sh           把 label 目录写成 SUST 可打开的数据集
```

## 与本项目的衔接

- 配置里 `data_root` 只是占位；`infer_mmdet3d.py` 会用 `BEVFUSION_ROOT`（默认 `<project>/bevfusion`）
  或 `BEVFUSION_DATA_ROOT` 覆盖成绝对路径。
- 权重放在本项目 `models/` 下（git-lfs）：`bevfusion_mmdet3d_lidaronly.pth`（默认）、`bevfusion_mmdet3d_lidarcam.pth`。
- 上游 `configs/upstream/`、`configs/mmdet_base/` 是 mmdetection3d 的原文件（Apache-2.0），
  为了让本目录不依赖 mmdetection3d 仓库里的配置文件路径；要跟着上游升级时替换它们即可。

## 一次性准备

```bash
# 1) mmdet3d 环境（torch 2.1.2+cu121 / mmcv 2.1.0 / mmdet 3.3.0 / mmdet3d 1.4.0 editable）
export MMDET3D_ROOT=~/MMDetection/mmdetection3d        # mmdetection3d 源码位置
# 2) 编 CUDA 算子（缓存到 ~/.cache/torch_extensions，约 1~2 分钟）
~/miniconda3/envs/mmdet3d/bin/python scripts/build_bev_pool.py
```

## 关键坑（已在新代码里绕开）

1. **z 约定**：mmdet3d 的 LiDAR 框 z 是**框底面**，本项目链路/SUST 用**框中心** → `infer_mmdet3d.py` 默认 `z += dz/2`（`--z-convention bottom` 可关）。
2. **框是 9 列**（x,y,z,dx,dy,dz,yaw,vx,vy）→ 只保留前 7 列，速度另存 `velocity`。
3. **纯雷达模式不需要图**：不去畸变、pipeline 不含图像变换（`modality.use_camera=False`）。
4. C+L 模式必须去畸变：4 路环视是 KANNALA_BRANDT 鱼眼，而 LSS 视角变换是针孔模型。
5. 官方 C+L 配置的 `img_backbone.init_cfg` 会去 GitHub 下 swin 预训练（离线会失败）→ 已置 `init_cfg=None`（整权重里含 SwinT）。
