#!/usr/bin/env python3
"""Truck 链 Step1（BEVFusion 版）——替换原来的 VoxelNeXt truck 检测器。

三步：
  ① 预处理：4 路环视鱼眼去畸变成针孔图 + lidar_top 写成 5 列 bin + 生成 nuScenes 风格 infos
     （缓存复用，已存在就跳过）
  ② 推理：mmdet3d 1.x 版 BEVFusion（官方 20 epoch 权重，nv val NDS 71.4 / mAP 68.6）出 10 类框，
     输出前已把 z 从「框底面」统一成「框中心」（见 bevfusion_truck/scripts/infer_mmdet3d.py）
  ③ 挂相机可见性元数据（与旧 step1_lidar_inference.py 完全一致，供后面 hard filter 用）

输出：``<work-root>/<clip.name>_raw.json``，结构与 ``inference/run_prelabel.py`` 的一致
（frames: [{frame_id, detections:[{class_name, score, box_lidar}]}]，lidar 系），
可被 step2 及之后的所有阶段直接消费。

本脚本自身跑在 **openpcdet** 环境（需要 filtering.camera_visibility）；
①② 通过外部 BEVFusion 工程用 **mmdet3d** 环境的 python 以子进程执行，
路径可用环境变量覆盖（云端部署时必改）：
    BEVFUSION_ROOT    默认 <project>/bevfusion（配置/脚本/缓存都随项目走）
    BEVFUSION_PYTHON  默认 ~/miniconda3/envs/mmdet3d/bin/python（mmdet3d 环境）
    BEVFUSION_TRUCK_CFG / BEVFUSION_TRUCK_CKPT         C+L 配置/权重（默认 models/bevfusion_mmdet3d_lidarcam.pth）
    BEVFUSION_TRUCK_LIDAR_CFG / ..._LIDAR_CKPT         纯雷达配置/权重（mode=lidar，默认）
    MMDET3D_ROOT      mmdetection3d 源码位置（默认 ~/MMDetection/mmdetection3d 等常见位置）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering import camera_visibility  # noqa: E402

BEVFUSION_ROOT = Path(os.environ.get(
    "BEVFUSION_ROOT", str(PROJECT_ROOT / "bevfusion")))      # 配置/脚本/缓存都随项目走
BEVFUSION_PYTHON = Path(os.environ.get(
    "BEVFUSION_PYTHON", str(Path.home() / "miniconda3" / "envs" / "mmdet3d" / "bin" / "python")))
BEVFUSION_CFG = os.environ.get("BEVFUSION_TRUCK_CFG",
                               "configs/police_bevfusion_mmdet3d.py")
BEVFUSION_CKPT = os.environ.get(
    "BEVFUSION_TRUCK_CKPT", str(PROJECT_ROOT / "models" / "bevfusion_mmdet3d_lidarcam.pth"))
BEVFUSION_Z = os.environ.get("BEVFUSION_TRUCK_Z", "center")
# 纯雷达（lidar-only）权重：46MB，9.1 FPS，相机分支不参与 -> 不去畸变、不读图
BEVFUSION_LIDAR_CFG = os.environ.get(
    "BEVFUSION_TRUCK_LIDAR_CFG", "configs/police_bevfusion_mmdet3d_lidaronly.py")
BEVFUSION_LIDAR_CKPT = os.environ.get(
    "BEVFUSION_TRUCK_LIDAR_CKPT", str(PROJECT_ROOT / "models" / "bevfusion_mmdet3d_lidaronly.pth"))


def validate_clip(clip: Path) -> None:
    lidar_top = clip / "lidar" / "lidar_top"
    if not lidar_top.is_dir() or not list(lidar_top.glob("*.bin")):
        raise ValueError(f"clip 缺少 lidar/lidar_top/*.bin：{clip}")
    for name in ("transforms", "image"):
        if not (clip / name).is_dir():
            raise ValueError(f"clip 缺少 {name}/：{clip}（BEVFusion 相机分支需要）")
    if not (clip / "transforms" / "calib.json").is_file():
        raise ValueError(f"clip 缺少 transforms/calib.json：{clip}")


def _run(cmd) -> None:
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def prepare(clip: Path, jobs: int = 6, images: bool = True) -> None:
    """去畸变（可选）+ 5 列 bin + infos（bevfusion_truck/scripts/prep_data.py，幂等）。"""
    cmd = [BEVFUSION_PYTHON, BEVFUSION_ROOT / "scripts" / "prep_data.py",
           "--data-root", clip.parent, "--out-root", BEVFUSION_ROOT,
           "--clips", clip.name, "--jobs", str(jobs)]
    if not images:          # 【改动】纯雷达模式：跳过鱼眼去畸变（省掉 ~90% 预处理时间）
        cmd.append("--no-images")
    _run(cmd)
    # 【改动】再生成 mmdet3d 需要的 middle-format infos（纯雷达模式下图片时间戳回退到原始 image/）
    _run([BEVFUSION_PYTHON, BEVFUSION_ROOT / "scripts" / "mmdet3d_prep.py",
          "--data-root", clip.parent, "--out-root", BEVFUSION_ROOT,
          "--clips", clip.name])


def infer(clip: Path, raw_json: Path, score_thresh: float, cfg: str, ckpt: str,
          z_convention: str) -> None:
    _run([BEVFUSION_PYTHON, BEVFUSION_ROOT / "scripts" / "infer_mmdet3d.py",
          "--cfg_file", str(BEVFUSION_ROOT / cfg if not Path(cfg).is_absolute() else cfg),
          "--ckpt", str(BEVFUSION_ROOT / ckpt if not Path(ckpt).is_absolute() else ckpt),
          "--out_json", str(raw_json), "--clips", clip.name,
          "--score_thresh", str(score_thresh),
          "--z-convention", z_convention])


def run_inference(clip: Path, args) -> Path:
    raw_json = Path(args.work_root) / f"{clip.name}_raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    mode = str(getattr(args, "mode", "fusion"))
    if mode == "lidar":
        # 纯雷达：不需要 image/ 目录，也不需要去畸变
        lidar_top = clip / "lidar" / "lidar_top"
        if not lidar_top.is_dir() or not list(lidar_top.glob("*.bin")):
            raise ValueError(f"clip 缺少 lidar/lidar_top/*.bin：{clip}")
        if not (clip / "transforms" / "calib.json").is_file():
            raise ValueError(f"clip 缺少 transforms/calib.json：{clip}")
    else:
        validate_clip(clip)
    if not args.skip_prepare:
        prepare(clip, jobs=args.jobs, images=(mode != "lidar"))
    cfg = args.cfg if mode != "lidar" else os.environ.get(
        "BEVFUSION_TRUCK_LIDAR_CFG", BEVFUSION_LIDAR_CFG)
    ckpt = args.ckpt if mode != "lidar" else os.environ.get(
        "BEVFUSION_TRUCK_LIDAR_CKPT", BEVFUSION_LIDAR_CKPT)
    if mode == "lidar" and (args.cfg != BEVFUSION_CFG or args.ckpt != BEVFUSION_CKPT):  # 显式传了就用显式值
        cfg, ckpt = args.cfg, args.ckpt        # 显式传了就用显式值
    infer(clip, raw_json, args.score_thresh, cfg, ckpt, args.z_convention)
    if not raw_json.is_file():
        raise RuntimeError(f"BEVFusion 推理没有产出 {raw_json}")

    if not args.no_visibility_check:
        frames = json.loads(raw_json.read_text(encoding="utf-8"))
        stats = camera_visibility.filter_raw_frames(frames, clip, 0.0, args.vis_occl_tol)
        raw_json.write_text(json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"visibility metadata: checked={stats['checked']} "
              f"dropped={stats['dropped']}", flush=True)
    return raw_json


def collect_clips(args):
    clips = [Path(c).resolve() for c in args.clip]
    for root in args.clip_dir:
        root = Path(root).resolve()
        for cand in (sorted(root.iterdir()) if root.is_dir() else []):
            try:      # 【改动】跳过 lost+found 等权限不足/无关条目
                if cand.name in {"lost+found", ".Trash-1000"} or cand.name.startswith("."):
                    continue
                if (cand / "lidar" / "lidar_top").is_dir():
                    clips.append(cand)
            except OSError:
                continue
    if not clips:
        raise SystemExit("没有输入 clip（--clip 或 --clip-dir）")
    return clips


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", default=[])
    parser.add_argument("--clip-dir", action="append", default=[])
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step1_inference")
    parser.add_argument("--mode", choices=["fusion", "lidar"], default="fusion",
                        help="fusion=C+L（默认，读 4 路图）; lidar=纯雷达权重（不读图/不去畸变，快 6 倍）")
    parser.add_argument("--cfg", default=BEVFUSION_CFG)
    parser.add_argument("--ckpt", default=BEVFUSION_CKPT)
    parser.add_argument("--score-thresh", type=float, default=0.1,
                        help="检测器原始阈值（链内类别阈值在 step2 把关）")
    parser.add_argument("--z-convention", default=BEVFUSION_Z,
                        choices=["center", "bottom"])
    parser.add_argument("--skip-prepare", action="store_true",
                        help="复用已有去畸变图/infos（调试用）")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--vis-occl-tol", type=float, default=0.3)
    parser.add_argument("--no-visibility-check", action="store_true")
    args = parser.parse_args()

    if not BEVFUSION_PYTHON.exists():
        raise SystemExit(f"BEVFUSION_PYTHON 不存在：{BEVFUSION_PYTHON}")
    if not (BEVFUSION_ROOT / "scripts" / "infer_mmdet3d.py").is_file():
        raise SystemExit(f"BEVFUSION_ROOT 下找不到 scripts/infer_mmdet3d.py：{BEVFUSION_ROOT}")

    summaries = []
    for index, clip in enumerate(collect_clips(args), 1):
        print(f"[{index}] {clip.name}", flush=True)
        raw = run_inference(clip, args)
        summaries.append({"clip": str(clip), "raw_json": str(raw)})
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
