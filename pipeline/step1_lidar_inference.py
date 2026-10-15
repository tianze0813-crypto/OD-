#!/usr/bin/env python3
"""Step 1: run lidar-top inference.

Input is an unlabelled SUST clip (``lidar/lidar_top/*.bin`` plus
``transforms/``).  Output is the ``*_raw.json`` consumed by step 2.

【改动 2026-10-10】``--fuse-lidars``：推理前先把 lidar_front / lidar_rear 按标定
融合进 lidar_top（补 top 雷达近场盲区），融合结果落在 ``<work-root>/lidar_fused/<clip>/``。
融合点云是过程数据（~130MB/clip），**默认推理一结束就删**，只留 ``<clip>_raw.json``；
要留着复用（调试 / 复跑）加 ``--keep-lidar-fusion``。

【改动 2026-10-14】orchestration（scripts/run_hybrid_prelabel.py）会带
``--keep-lidar-fusion`` 跑：融合点云要活到**公共可见度阶段**（pipeline/final_visibility.py
用它的近场点做点云高度比），阶段跑完由编排那边删掉。

Run this step with the OpenPCDet environment, for example:
    python pipeline/step1_lidar_inference.py \
        --clip /path/to/scene_clip
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import lidar_fusion


DEFAULT_CFG = PROJECT_ROOT / "models" / "voxelnext_fiveclass_nuscenes_infer.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "models" / "vn5_nuscenes_checkpoint_epoch_12.pth"


def is_clip_dir(path: Path) -> bool:
    try:
        if path.name in {"lost+found", ".Trash-1000"} or path.name.startswith("."):
            return False
        lidar_top = path / "lidar" / "lidar_top"
        return path.is_dir() and lidar_top.is_dir() and any(lidar_top.glob("*.bin"))
    except OSError:
        return False


def validate_clip(clip: Path) -> None:
    if not clip.is_dir():
        raise ValueError(f"输入不是目录：{clip}")
    lidar_top = clip / "lidar" / "lidar_top"
    if not lidar_top.is_dir() or not list(lidar_top.glob("*.bin")):
        raise ValueError(f"clip 缺少 lidar/lidar_top/*.bin：{clip}")
    if not (clip / "transforms").is_dir():
        raise ValueError(f"clip 缺少 transforms/：{clip}")


def collect_clips(args) -> list[Path]:
    clips = [Path(c).resolve() for c in args.clip]
    for root in args.clip_dir:
        root = Path(root).resolve()
        if not root.is_dir():
            raise ValueError(f"--clip-dir 不是目录：{root}")
        for candidate in sorted(root.iterdir()):      # 【改动】权限不足的条目跳过
            try:
                if is_clip_dir(candidate):
                    clips.append(candidate)
            except OSError:
                continue
    if not clips:
        raise ValueError("请至少给一个 --clip 或 --clip-dir")
    seen = set()
    for clip in clips:
        validate_clip(clip)
        if clip.name in seen:
            raise ValueError(f"同一批存在同名 clip：{clip.name}")
        seen.add(clip.name)
    return clips


def _cleanup_fused(fused_dir: Path) -> None:
    """删掉本次融合出的点云（默认跑完即清，避免 work/ 越积越大）。"""
    try:
        shutil.rmtree(fused_dir, ignore_errors=True)
        parent = fused_dir.parent
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass


def run_inference(clip: Path, args) -> Path:
    raw_json = args.work_root / f"{clip.name}_raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    lidar_dir = clip / "lidar" / "lidar_top"
    fused_dir = None
    if getattr(args, "fuse_lidars", False):
        # 公共预处理：front/rear -> top 系拼接，补 top 近场盲区。
        fused_dir = args.work_root / "lidar_fused" / clip.name
        if not any(fused_dir.glob("*.bin")):
            stats = lidar_fusion.fuse_clip(clip, fused_dir)
            print(f"lidar fusion: {stats['frames']} 帧, 附加 "
                  f"{stats['attached_points']} 点 -> {fused_dir}", flush=True)
        lidar_dir = fused_dir
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "inference" / "run_prelabel.py"),
        "--cfg_file", str(args.cfg),
        "--ckpt", str(args.ckpt),
        "--lidar_dir", str(lidar_dir),
        "--out_json", str(raw_json),
        "--score_thresh", str(args.score_thresh),
    ]
    try:
        print(">> " + " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

        # 【改动 2026-10-14】遮挡属性/可见度过滤已移到三条链合并后的公共阶段
        # （pipeline/final_visibility.py），raw 阶段不再挂 visibility 元数据。
        return raw_json
    finally:
        # 【改动 2026-10-10】融合点云是过程数据，默认跑完即删（失败也删）。
        if fused_dir is not None and not getattr(args, "keep_lidar_fusion", False):
            _cleanup_fused(fused_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clip", action="append", default=[],
                        help="输入无 label clip，可多次传入")
    parser.add_argument("--clip-dir", action="append", default=[],
                        help="扫描目录下所有含 lidar/lidar_top/*.bin 的 clip")
    parser.add_argument("--work-root", type=Path,
                        default=PROJECT_ROOT / "work" / "step1_inference")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--score-thresh", type=float, default=0.3)
    parser.add_argument("--fuse-lidars", action="store_true",
                        help="推理前融合 lidar_front/rear 到 lidar_top（补近场盲区）")
    parser.add_argument("--keep-lidar-fusion", action="store_true",
                        help="【默认关】保留 <work-root>/lidar_fused/<clip> 融合点云"
                             "（默认推理一结束就删，失败也删；orchestration 会打开它，"
                             "留给公共可见度阶段用，阶段跑完再删）")
    args = parser.parse_args()

    for path, name in ((args.cfg, "cfg"), (args.ckpt, "ckpt")):
        if not Path(path).exists():
            raise SystemExit(f"{name} 不存在：{path}")

    clips = collect_clips(args)
    args.work_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index, clip in enumerate(clips, 1):
        print(f"[{index}/{len(clips)}] {clip.name}", flush=True)
        raw_json = run_inference(clip, args)
        summaries.append({
            "clip": str(clip),
            "raw_json": str(raw_json),
        })
    print(json.dumps({"clips": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
