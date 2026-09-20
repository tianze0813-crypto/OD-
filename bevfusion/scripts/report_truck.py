#!/usr/bin/env python3
"""汇总 BEVFusion 原始推理里的「卡车类」结果：数量 / 分数 / 长度分布。

用法: python scripts/report_truck.py [--glob 'out/*.raw.json'] [--thresh 0.2,0.3,0.4]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

GROUP = {"truck": "Truck", "bus": "Truck", "trailer": "Trailer",
         "construction_vehicle": "construction_vehicle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="out/*.raw.json")
    ap.add_argument("--thresh", default="0.2,0.3,0.4")
    args = ap.parse_args()
    ths = [float(x) for x in args.thresh.split(",")]
    files = sorted(glob.glob(args.glob))
    if not files:
        print("没有原始推理 json")
        return

    for f in files:
        name = os.path.basename(f).replace(".raw.json", "")
        frames = json.load(open(f))
        print(f"\n=== {name}  ({len(frames)} 帧)")
        for th in ths:
            per_frame = defaultdict(list)
            lens = defaultdict(list)
            for fr in frames:
                cs = defaultdict(int)
                for d in fr["detections"]:
                    if d["score"] < th:
                        continue
                    g = GROUP.get(d["class_name"])
                    if g is None:
                        continue
                    b = d["box_lidar"]
                    cs[g] += 1
                    lens[g].append(max(b[3], b[4]))
                for g in ("Truck", "Trailer", "construction_vehicle"):
                    per_frame[g].append(cs.get(g, 0))
            parts = []
            for g in ("Truck", "Trailer"):
                a = np.asarray(lens[g]) if lens[g] else np.zeros(0)
                if not len(a):
                    parts.append(f"{g}: 0")
                    continue
                pf = np.asarray(per_frame[g])
                parts.append(
                    f"{g}: {len(a):4d} 框 ({len(a)/len(frames):4.1f}/帧, 有框帧 {100*(pf>0).mean():4.0f}%) "
                    f"长度 P50 {np.median(a):5.2f} P90 {np.percentile(a,90):5.2f} max {a.max():5.2f} "
                    f"| >12m {100*(a>12).mean():4.1f}% >15m {100*(a>15).mean():4.1f}%")
            print(f"  阈值 {th}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
