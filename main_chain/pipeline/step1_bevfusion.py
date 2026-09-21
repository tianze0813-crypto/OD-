#!/usr/bin/env python3
"""Step1（BEVFusion 检测器）：用 BEVFusion 出 raw json，替代自带的 Waymo/VoxelNeXt step1。

三步（与 pipeline/step1_lidar_inference.py 输出同构，后续 step2~step5 不用改）：
  ① 预处理：lidar_top 写 5 列 bin（纯雷达模式到这里就够了）；C+L 模式额外做 4 路鱼眼去畸变
  ② 推理：BEVFusion（mmdet3d 1.x 官方权重）出 10 类框，出口已把 z 统一成「框中心」
  ③ 挂相机可见性元数据（本仓库自己的 filtering.camera_visibility）

本脚本跑在 openpcdet 环境，①② 通过子进程调用外部 BEVFusion 工具箱（mmdet3d 环境）。
路径解析（都可用环境变量覆盖）：
  BEVFUSION_ROOT / BEVFUSION_PYTHON        工具箱与解释器
  BEVFUSION_CFG / BEVFUSION_CKPT           C+L 配置/权重
  BEVFUSION_LIDAR_CFG / BEVFUSION_LIDAR_CKPT   纯雷达配置/权重（--mode lidar，默认）
  权重找不到时会在 <BEVFUSION_ROOT>/../models/ 和本仓库 models/ 下找
  bevfusion_mmdet3d_*.pth
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from filtering import camera_visibility  # noqa: E402

def _bevfusion_root() -> Path:
    """工具箱位置：环境变量优先，其次按常见布局猜（本仓库旁边 / 上级目录 / 本级）。"""
    candidates = [os.environ.get("BEVFUSION_ROOT"),
                  ROOT.parent / "OD--hybrid-main-car-expd-noncar" / "bevfusion",
                  ROOT.parent / "bevfusion",
                  ROOT / "bevfusion"]
    for cand in candidates:
        if cand and (Path(cand) / "scripts" / "infer_mmdet3d.py").is_file():
            return Path(cand)
    raise SystemExit("找不到 BEVFusion 工具箱：设 BEVFUSION_ROOT 指向含 scripts/infer_mmdet3d.py 的目录")


def _python() -> Path:
    cand = Path(os.environ.get("BEVFUSION_PYTHON",
                               Path.home() / "miniconda3/envs/mmdet3d/bin/python"))
    if not cand.exists():
        raise SystemExit(f"BEVFUSION_PYTHON 不存在：{cand}（需要 mmdet3d 环境）")
    return cand


def _weight(name: str, env_keys) -> Path:
    for key in env_keys:
        value = os.environ.get(key)
        if value:
            return Path(value)
    for base in (_bevfusion_root().parent / "models", ROOT / "models"):
        cand = base / name
        if cand.is_file():
            return cand
    return _bevfusion_root().parent / "models" / name        # 交回默认值，报错信息更直观


BEVFUSION_LIDAR_CFG = os.environ.get("BEVFUSION_LIDAR_CFG",
                                     os.environ.get("BEVFUSION_TRUCK_LIDAR_CFG",
                                                    "configs/police_bevfusion_mmdet3d_lidaronly.py"))
BEVFUSION_LIDAR_CKPT_NAME = "bevfusion_mmdet3d_lidaronly.pth"
BEVFUSION_CFG = os.environ.get("BEVFUSION_CFG",
                               os.environ.get("BEVFUSION_TRUCK_CFG",
                                              "configs/police_bevfusion_mmdet3d.py"))
BEVFUSION_CKPT_NAME = "bevfusion_mmdet3d_lidarcam.pth"


def _run(cmd) -> None:
    print(">> " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def run_inference(clip: Path, args) -> Path:
    root = _bevfusion_root()
    python = _python()
    mode = str(getattr(args, "mode", "lidar"))
    raw_json = Path(args.work_root) / f"{clip.name}_raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    lidar_top = clip / "lidar" / "lidar_top"
    if not lidar_top.is_dir() or not list(lidar_top.glob("*.bin")):
        raise ValueError(f"clip 缺少 lidar/lidar_top/*.bin：{clip}")
    if not (clip / "transforms" / "calib.json").is_file():
        raise ValueError(f"clip 缺少 transforms/calib.json：{clip}")
    if mode != "lidar" and not (clip / "image").is_dir():
        raise ValueError(f"C+L 模式需要 image/：{clip}")

    prep_cmd = [python, root / "scripts" / "prep_data.py",
                "--data-root", clip.parent, "--out-root", root,
                "--clips", clip.name, "--jobs", str(getattr(args, "jobs", 6))]
    if mode == "lidar":
        prep_cmd.append("--no-images")       # 纯雷达：不去畸变
    _run(prep_cmd)
    _run([python, root / "scripts" / "mmdet3d_prep.py",
          "--data-root", clip.parent, "--out-root", root, "--clips", clip.name])

    cfg = (BEVFUSION_LIDAR_CFG if mode == "lidar" else BEVFUSION_CFG)
    ckpt = (_weight(BEVFUSION_LIDAR_CKPT_NAME, ("BEVFUSION_LIDAR_CKPT", "BEVFUSION_TRUCK_LIDAR_CKPT"))
            if mode == "lidar"
            else _weight(BEVFUSION_CKPT_NAME, ("BEVFUSION_CKPT", "BEVFUSION_TRUCK_CKPT")))
    _run([python, root / "scripts" / "infer_mmdet3d.py",
          "--cfg_file", str(root / cfg if not Path(cfg).is_absolute() else cfg),
          "--ckpt", str(ckpt), "--out_json", str(raw_json), "--clips", clip.name,
          "--score_thresh", str(getattr(args, "score_thresh", 0.2)),
          "--z-convention", "center"])
    if not raw_json.is_file():
        raise RuntimeError(f"BEVFusion 推理没有产出 {raw_json}")

    if not getattr(args, "no_visibility_check", False):
        frames = json.loads(raw_json.read_text(encoding="utf-8"))
        stats = camera_visibility.filter_raw_frames(
            frames, clip, 0.0, getattr(args, "vis_occl_tol", 0.3))
        raw_json.write_text(json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"visibility metadata: checked={stats['checked']} dropped={stats['dropped']}",
              flush=True)
    return raw_json


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, type=Path)
    ap.add_argument("--work-root", type=Path, required=True)
    ap.add_argument("--mode", choices=["lidar", "fusion"], default="lidar")
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--vis-occl-tol", type=float, default=0.3)
    ap.add_argument("--no-visibility-check", action="store_true")
    args = ap.parse_args()
    raw = run_inference(args.clip.resolve(), args)
    print(json.dumps({"clip": str(args.clip), "raw_json": str(raw)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
