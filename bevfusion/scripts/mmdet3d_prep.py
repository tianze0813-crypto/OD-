#!/usr/bin/env python3
"""复用 prep_data.py 的产物，生成 mmdet3d 1.x 版 BEVFusion 需要的 middle-format infos。

不重新处理数据：直接用
  data/police/<clip>/lidar/lidar_top/*.bin     (5 列，prep_data.py 写的)
  work/undist/<clip>/<cam>/*.jpg               (去畸变针孔图)
输出 work/infos/police_mmdet3d_infos.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prep_data import CAMS_DEFAULT, cam2lidar, scale_K  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                    help="存放 clip 的目录（每个 clip 一个子目录，含 image/lidar/transforms）")
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parent.parent),
                    help="缓存/产物根（默认 = <project>/bevfusion）")
    ap.add_argument("--cams", default=",".join(CAMS_DEFAULT))
    ap.add_argument("--out-size", default="1920x1536")
    ap.add_argument("--clips", default="",
                    help="只处理这些 clip（逗号分隔），默认全部（缓存目录里遗留的也会被尝试）")
    args = ap.parse_args()

    root = Path(args.out_root)
    data_root = Path(args.data_root)
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    ow, oh = (int(x) for x in args.out_size.split("x"))

    # 【修】同 prep_data：给了 --clips 就不限定 scene_ 前缀
    cached = [d.name for d in (root / "data" / "police").iterdir()
              if d.is_dir() and not d.name.startswith(".")]
    if args.clips:
        want = {c.strip() for c in args.clips.split(",") if c.strip()}
        clips = sorted(c for c in cached if c in want)
    else:
        clips = sorted(c for c in cached if c.startswith("scene_"))
    infos = []
    idx = 0
    for clip in clips:
        cdir = data_root / clip
        bins = sorted((root / "data" / "police" / clip / "lidar" / "lidar_top").glob("*.bin"))
        # 【修】标定优先用 prep_data 复制到缓存里的那份（缓存自足：in-place 批跑会把源 clip 改名成
        #      <clip>_pre，源路径随时可能不在）；缓存里没有才回源目录。都没有就跳过而不是崩。
        calib_path = root / "data" / "police" / clip / "transforms" / "calib.json"
        if not calib_path.is_file():
            calib_path = cdir / "transforms" / "calib.json"
        if not calib_path.is_file() or not bins:
            print(f"[skip] {clip}: 标定或 bin 不完整（calib={calib_path.is_file()}, bin={len(bins)}）")
            continue
        calib = json.loads(calib_path.read_text())
        for bp in bins:
            ts = bp.stem
            images = {}
            for cam in cams:
                # 与 prep_data 相同的去畸变 K
                w, h = int(calib[cam]["imgw"]), int(calib[cam]["imgh"])
                s = min(ow / w, oh / h)
                Kp = scale_K(np.asarray(calib[cam]["K"], np.float64), s)
                Kp[0, 2] += (ow - int(round(w * s))) // 2
                Kp[1, 2] += (oh - int(round(h * s))) // 2
                M = cam2lidar(calib, cam)
                cam2ego = np.asarray(calib["tf2base_link"][cam], np.float64)
                img_dir = root / "work" / "undist" / clip / cam
                if not img_dir.is_dir():        # 纯雷达模式：没有去畸变图，用原图配对时间戳
                    img_dir = cdir / "image" / cam
                stems = [p.stem for p in img_dir.glob("*.jpg")]
                if stems:                       # 图也不在（如源被改名）-> 纯雷达模式下无妨，直接用雷达时间戳
                    it = min(stems, key=lambda t: abs(int(t) - int(ts)))
                else:
                    it = ts
                images[cam] = {
                    "img_path": f"{clip}/image/{cam}/{it}.jpg",
                    "cam2img": Kp.astype(np.float32),
                    "cam2ego": cam2ego.astype(np.float32),
                    "lidar2cam": np.linalg.inv(M).astype(np.float32),
                    "cam2lidar": M.astype(np.float32),
                    "timestamp": int(it),
                    "sample_data_token": f"{clip}_{cam}_{it}",
                }
            infos.append({
                "sample_idx": idx,
                "token": f"{clip}_{ts}",
                "scene_token": clip,
                "timestamp": int(ts),
                "ego2global": np.eye(4, dtype=np.float32),
                "lidar_points": {
                    "lidar_path": f"{clip}/lidar/lidar_top/{ts}.bin",
                    "num_pts_feats": 5,
                    "lidar2ego": np.eye(4, dtype=np.float32),
                    "timestamp": int(ts),
                },
                "images": images,
            })
            idx += 1

    out = root / "work" / "infos" / "police_mmdet3d_infos.pkl"
    # mmengine 要求 ann 文件是 dict（取 data_list）
    with open(out, "wb") as f:
        pickle.dump({"metainfo": {"dataset": "police_bevfusion",
                                  "version": "police-0914"},
                     "data_list": infos}, f)
    print(f"[mmdet3d infos] {len(infos)} 帧 / {len(clips)} clip -> {out}")
    if infos:
        print("  cameras:", list(infos[0]["images"].keys()))
        print("  lidar  :", infos[0]["lidar_points"]["lidar_path"])
        print("  img[0] :", infos[0]["images"][cams[0]]["img_path"])


if __name__ == "__main__":
    main()
