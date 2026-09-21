#!/usr/bin/env python3
"""把 BEVFusion 的【原始检测】(raw json，lidar_top 系) 直接导成 SUST 数据集。

不跑任何链（不做跟踪/过滤/精修），只做三件事：
  ① 类别映射到项目口径：car→Car、pedestrian→Pedestrian、bicycle/motorcycle→Nonmotorized_vehicle、
     truck/bus→Truck、trailer→Trailer（--classes 可只留需要的类别，默认留 Car/Pedestrian/NMV）
  ② 坐标换算 lidar_top → base_link：**复用项目自己的 tracking.box_lidar_to_base_link**（只转一次，
     与其它链路口径一致，不存在重复换算）
  ③ obj_id：轻量贪心最近邻（同类、BEV 中心距 ≤ 4m），只为在 SUST 里能看出同一条轨迹，**不是正式跟踪**

输出：<sust-root>/<name>/{image,lidar,transforms,readme.json 软链, label/*.json}

用法：
  python bevfusion/scripts/export_sust_raw.py \
      --raw-json work/x_raw.json --clip /media/moga/police/<clip> --name <clip>_bevfusion_heads
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]      # <project>
sys.path.insert(0, str(PROJECT))
from tracking import tracker_conservative as tracking  # noqa: E402

MAP_CLASS = {
    "car": "Car", "truck": "Truck", "bus": "Truck", "trailer": "Trailer",
    "pedestrian": "Pedestrian", "bicycle": "Nonmotorized_vehicle",
    "motorcycle": "Nonmotorized_vehicle",
}
DEFAULT_CLASSES = "Car,Pedestrian,Nonmotorized_vehicle"


def light_track(frames, max_dist=4.0):
    """逐帧贪心最近邻给 obj_id（只为可视化；不是正式跟踪）。"""
    live, next_id = [], 1
    for frame in frames:
        used = set()
        for det in sorted(frame["detections"], key=lambda d: -d["score"]):
            cls, cx, cy = det["obj_type"], float(det["box"][0]), float(det["box"][1])
            best, best_d = None, max_dist
            for i, (lc, lx, ly, lid) in enumerate(live):
                if i in used or lc != cls:
                    continue
                d = math.hypot(cx - lx, cy - ly)
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                det["obj_id"] = next_id
                live.append((cls, cx, cy, next_id))
                used.add(len(live) - 1)
                next_id += 1
            else:
                det["obj_id"] = live[best][3]
                live[best] = (cls, cx, cy, live[best][3])
                used.add(best)
        live = [live[i] for i in range(len(live)) if i in used]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-json", required=True, type=Path)
    ap.add_argument("--clip", required=True, type=Path, help="源 clip（提供 calib 与 image/lidar/transforms）")
    ap.add_argument("--name", required=True, help="SUST 数据集名")
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--keep-unmapped", action="store_true",
                    help="映射表里没有的类别原样保留（做'真·原始检测'直出时用）")
    ap.add_argument("--classes", default=DEFAULT_CLASSES,
                    help="只保留这些（项目口径）类别，逗号分隔；留空=全部")
    ap.add_argument("--sust-root", type=Path,
                    default=Path.home() / "桌面" / "SUSTechPOINTS" / "data")
    ap.add_argument("--no-link", action="store_true", help="只写 label，不建 SUST 目录/软链")
    args = ap.parse_args()

    clip = args.clip.resolve()
    calib = json.loads((clip / "transforms" / "calib.json").read_text())
    base_from_lidar = np.asarray(calib["tf2base_link"]["lidar_top"], np.float64)
    keep = {c.strip() for c in args.classes.split(",") if c.strip()}

    raw = json.loads(args.raw_json.read_text(encoding="utf-8"))
    frames, counts, lengths = [], Counter(), defaultdict(list)
    for fr in raw:
        dets = []
        for d in fr["detections"]:
            if d["score"] < args.score_thresh:
                continue
            raw_name = str(d["class_name"])
            obj_type = MAP_CLASS.get(raw_name.lower())
            if obj_type is None:
                if not args.keep_unmapped:      # 默认丢未映射类别
                    continue
                obj_type = raw_name             # 原样保留（traffic_cone / barrier / ...）
            if keep and obj_type not in keep:
                continue
            box = tracking.box_lidar_to_base_link(d["box_lidar"][:7], base_from_lidar)
            dets.append({"obj_type": obj_type, "source_class": d["class_name"],
                         "score": float(d["score"]), "box": [float(v) for v in box]})
            counts[obj_type] += 1
            lengths[obj_type].append(max(box[3], box[4]))
        frames.append({"frame_id": str(fr["frame_id"]), "detections": dets})
    frames = light_track(frames)

    out_dir = args.sust_root / args.name
    if not args.no_link:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for sub in ("image", "lidar", "transforms", "readme.json"):
            src = clip / sub
            if src.exists():                     # 跨设备只能软链（硬链会报 cross-device）
                (out_dir / sub).symlink_to(src)
    label_dir = out_dir / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    for fr in frames:
        labels = []
        for d in fr["detections"]:
            b = d["box"]
            labels.append({
                "obj_id": str(d["obj_id"]), "obj_type": d["obj_type"],
                "score": round(d["score"], 4),
                "psr": {"position": {"x": round(b[0], 4), "y": round(b[1], 4), "z": round(b[2], 4)},
                        "rotation": {"x": 0.0, "y": 0.0, "z": round(b[6], 4)},
                        "scale": {"x": round(b[3], 4), "y": round(b[4], 4), "z": round(b[5], 4)}},
                "source_class": d["source_class"],
            })
        (label_dir / f"{fr['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"== {args.name}: {len(frames)} 帧, 阈值 {args.score_thresh}, 类别 {dict(counts)}")
    for k in sorted(lengths):
        a = np.asarray(lengths[k])
        print(f"   {k:22s} 长度 P50 {np.median(a):5.2f} max {a.max():5.2f}")
    print(f"   -> {label_dir}")


if __name__ == "__main__":
    main()
