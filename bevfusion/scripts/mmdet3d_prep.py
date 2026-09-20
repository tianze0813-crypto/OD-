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
    args = ap.parse_args()

    root = Path(args.out_root)
    data_root = Path(args.data_root)
    cams = [c.strip() for c in args.cams.split(",") if c.strip()]
    ow, oh = (int(x) for x in args.out_size.split("x"))

    clips = sorted(d.name for d in (root / "data" / "police").glob("scene_*") if d.is_dir())
    infos = []
    idx = 0
    for clip in clips:
        cdir = data_root / clip
        calib = json.loads((cdir / "transforms" / "calib.json").read_text())
        bins = sorted((root / "data" / "police" / clip / "lidar" / "lidar_top").glob("*.bin"))
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
                it = min((p.stem for p in img_dir.glob("*.jpg")), key=lambda t: abs(int(t) - int(ts)))
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
