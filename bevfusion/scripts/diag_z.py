#!/usr/bin/env python3
"""诊断框 z 的系统性偏差：把预测框与参考标注配对后看 Δz。

关键判据：如果 Δz ≈ -H/2（即 Δz 与框高 dz 的回归斜率 ≈ -0.5），
说明预测的 z 是「框底面」（mmdet3d LiDAR 框的 origin=(0.5,0.5,0)），
而参考标注的 z 是「框中心」——修法就是 z += dz/2。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_vs_ref import GROUP_TRUCK, greedy, load_pred, load_ref  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="存 clip 的目录")
    ap.add_argument("--raw-glob", default="out/mmdet3d/*.raw.json")
    ap.add_argument("--ref-root", default="work/ref_labels")
    ap.add_argument("--ref-score", type=float, default=0.4)
    ap.add_argument("--score-thresh", type=float, default=0.2)
    ap.add_argument("--dist", type=float, default=2.0)
    args = ap.parse_args()

    data_root, ref_root = Path(args.data_root), Path(args.ref_root)
    dz_all, h_all, dxy = [], [], []
    for raw in sorted(Path().glob(args.raw_glob)):
        clip = raw.name.replace(".raw.json", "")
        cdir = data_root / clip
        rdir = ref_root / clip
        if not (cdir.exists() and rdir.exists()):
            continue
        base = np.asarray(json.loads((cdir / "transforms" / "calib.json").read_text())
                          ["tf2base_link"]["lidar_top"], np.float64)
        ref = dict(load_ref(rdir, base, args.ref_score))
        pred = dict(load_pred(raw, cdir, args.score_thresh, GROUP_TRUCK))
        for ts, gtb in ref.items():
            pb = pred.get(ts, np.zeros((0, 7)))
            if not len(gtb) or not len(pb):
                continue
            ps = np.hstack([pb, np.ones((len(pb), 1))])
            pairs, _ = greedy(ps, gtb, args.dist)
            for pi, gi in pairs:
                dz_all.append(pb[pi, 2] - gtb[gi, 2])       # 预测z - 参考z
                h_all.append(pb[pi, 5])                     # 预测框高
                dxy.append(np.hypot(pb[pi, 0] - gtb[gi, 0], pb[pi, 1] - gtb[gi, 1]))
    dz = np.asarray(dz_all); h = np.asarray(h_all); dxy = np.asarray(dxy)
    if not len(dz):
        print("没有配对"); return
    print(f"配对 {len(dz)} 对（{args.raw_glob}，score>={args.score_thresh}，BEV 距离<={args.dist}m）")
    print(f"  Δz = 预测z − 参考z : P10 {np.percentile(dz,10):+.2f} P50 {np.median(dz):+.2f} P90 {np.percentile(dz,90):+.2f} m")
    print(f"  框高 dz          : P50 {np.median(h):.2f} m  → 一半 {np.median(h)/2:+.2f} m")
    slope = np.polyfit(h, dz, 1)[0]
    print(f"  Δz 对框高的回归斜率 = {slope:+.3f}   (=-0.5 ⇒ z 是底面，需要 +dz/2)")
    print(f"  修正后 Δz' = Δz + dz/2 : P50 {np.median(dz + h/2):+.2f} m")
    print(f"  修正后残差标准差 {np.std(dz + h/2):.2f} m（修正前 {np.std(dz):.2f} m）")
    print(f"  （参考数据）BEV 中心距 P50 {np.median(dxy):.2f} m")


if __name__ == "__main__":
    main()
