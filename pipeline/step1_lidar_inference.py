#!/usr/bin/env python3
"""Step 1: run lidar-top inference and camera-visibility pre-filtering.

Input is an unlabelled SUST clip (``lidar/lidar_top/*.bin`` plus
``transforms/``).  Output is the ``*_raw.json`` consumed by step 2.

Run this step with the OpenPCDet environment, for example:
    /home/moga/miniconda3/envs/openpcdet/bin/python pipeline/step1_lidar_inference.py \
        --clip /path/to/scene_clip
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering import camera_visibility


DEFAULT_CFG = PROJECT_ROOT / "models" / "voxelnext_v2_waymo_infer.yaml"
DEFAULT_CKPT = PROJECT_ROOT / "models" / "vn_waymo_v2_4gpu_full_epoch10.pth"


def is_clip_dir(path: Path) -> bool:
    lidar_top = path / "lidar" / "lidar_top"
    return path.is_dir() and lidar_top.is_dir() and any(lidar_top.glob("*.bin"))


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
        clips.extend(sorted(p for p in root.iterdir() if is_clip_dir(p)))
    if not clips:
        raise ValueError("请至少给一个 --clip 或 --clip-dir")
    seen = set()
    for clip in clips:
        validate_clip(clip)
        if clip.name in seen:
            raise ValueError(f"同一批存在同名 clip：{clip.name}")
        seen.add(clip.name)
    return clips


def run_inference(clip: Path, args) -> Path:
    raw_json = args.work_root / f"{clip.name}_raw.json"
    raw_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "inference" / "run_prelabel.py"),
        "--cfg_file", str(args.cfg),
        "--ckpt", str(args.ckpt),
        "--lidar_dir", str(clip / "lidar" / "lidar_top"),
        "--out_json", str(raw_json),
        "--score_thresh", str(args.score_thresh),
    ]
    print(">> " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    if not args.no_visibility_check:
        frames = json.loads(raw_json.read_text(encoding="utf-8"))
        stats = camera_visibility.filter_raw_frames(
            frames, clip, args.drop_vis_below, args.vis_occl_tol)
        raw_json.write_text(
            json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"visibility filter: checked={stats['checked']} "
              f"dropped={stats['dropped']}", flush=True)
    return raw_json


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
    parser.add_argument("--drop-vis-below", type=float, default=0.05)
    parser.add_argument("--vis-occl-tol", type=float, default=0.3)
    parser.add_argument("--no-visibility-check", action="store_true")
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
