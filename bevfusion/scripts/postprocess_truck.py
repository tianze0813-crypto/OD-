#!/usr/bin/env python3
"""Truck / Trailer 后处理（按用户口径，逐帧、在 lidar 系做）：

  ①【重复】挂车框基本被货车框罩住（IoM = 交面积/挂车面积 >= --dup-iom，或 IoU >= --dup-iou）
     → 同一辆车被 truck 头和 trailer 头各出了一次 → **以 Truck 为准**，丢掉 Trailer。
  ②【有交集】挂车与货车部分相交（IoU >= --merge-iou）→ **合并成一个大长 Truck（2D OBB 并集）**，
     丢掉该 Trailer。并集算法直接复用 hybrid_code/geometry/truck_postprocess._union_box
     （平行时沿长框轴取并集，z 取两者上下界的并集），与现有链路口径完全一致。
  ③ 与任何货车都不相交 → 保留为独立 Trailer。

合并后又变大的货车会继续和剩余挂车比较（多轮迭代），支持"车头+挂车+再加一节"的情况。

用法:
  python scripts/postprocess_truck.py --raw in.json --out out.json
  for f in out/mmdet3d/*.raw.json; do ... done
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HYB = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(HYB))
from geometry.truck_postprocess import _union_box  # noqa: E402
from tracking import tracker_conservative as tracking  # noqa: E402

TRUCK_CLASSES = {'truck', 'bus'}
TRAILER_CLASSES = {'trailer'}


def _iou_and_iom(truck_box, trailer_box):
    ck, sk, yk = truck_box[:2], truck_box[3:6], truck_box[6]
    ct, st, yt = trailer_box[:2], trailer_box[3:6], trailer_box[6]
    iou = tracking.bev_iou(ck, sk, yk, ct, st, yt)
    pk = tracking.rectangle_corners(ck, sk, yk)
    pt = tracking.rectangle_corners(ct, st, yt)
    inter = tracking.polygon_area(tracking.convex_intersection(pk, pt))
    area_t = tracking.polygon_area(pt)
    iom = 0.0 if area_t <= 1e-9 else inter / area_t
    return iou, iom


def process_frame(dets, dup_iom, dup_iou, merge_iou, max_iter=8):
    trucks = [d for d in dets if d['class_name'] in TRUCK_CLASSES]
    trailers = [d for d in dets if d['class_name'] in TRAILER_CLASSES]
    others = [d for d in dets if d['class_name'] not in TRUCK_CLASSES | TRAILER_CLASSES]
    stat = Counter()
    for _ in range(max_iter):
        changed = False
        for t in list(trailers):
            best, best_iou = None, 0.0
            for k in trucks:
                iou, iom = _iou_and_iom(k['box_lidar'], t['box_lidar'])
                if iom >= dup_iom or iou >= dup_iou:          # ① 重复 -> 保 truck
                    trailers.remove(t)
                    stat['dup_dropped'] += 1
                    changed = True
                    best = None
                    break
                if iou >= merge_iou and iou > best_iou:        # ② 有交集 -> 并集
                    best, best_iou = k, iou
            if best is not None:
                merged, _ = _union_box(best['box_lidar'], t['box_lidar'])
                best['box_lidar'] = [float(v) for v in merged]
                best['score'] = float(max(best['score'], t['score']))
                best['merged_from'] = best.get('merged_from', 1) + 1
                trailers.remove(t)
                stat['merged'] += 1
                changed = True
        if not changed:
            break
    stat['truck_left'] = len(trucks)
    stat['trailer_left'] = len(trailers)
    return others + trucks + trailers, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dup-iom', type=float, default=0.70, help='挂车被货车罩住的比例 -> 判为重复')
    ap.add_argument('--dup-iou', type=float, default=0.50, help='或 BEV IoU 超过此值 -> 判为重复')
    ap.add_argument('--merge-iou', type=float, default=0.05, help='有交集判据（BEV IoU）')
    args = ap.parse_args()

    frames = json.loads(Path(args.raw).read_text())
    total = Counter()
    len_before, len_after = [], []
    for fr in frames:
        for d in fr['detections']:
            if d['score'] >= 0.2 and d['class_name'] in TRUCK_CLASSES:
                len_before.append(max(d['box_lidar'][3], d['box_lidar'][4]))
        fr['detections'], st = process_frame(fr['detections'], args.dup_iom, args.dup_iou, args.merge_iou)
        total.update(st)
        for d in fr['detections']:
            if d['score'] >= 0.2 and d['class_name'] in TRUCK_CLASSES:
                len_after.append(max(d['box_lidar'][3], d['box_lidar'][4]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(frames, ensure_ascii=False) + '\n', encoding='utf-8')
    nf = len(frames)
    print(f"[pp] {Path(args.raw).name} -> {out.name}")
    print(f"     重复丢掉 Trailer {total['dup_dropped']} 个 | 并集合并 {total['merged']} 次 "
          f"| 剩余 Trailer {total['trailer_left']} 个 | Truck {total['truck_left']} 个 "
          f"(共 {nf} 帧)")
    if len_before:
        a, b = np.asarray(len_before), np.asarray(len_after)
        print(f"     Truck 长度(≥0.2) 前 P50 {np.median(a):5.2f} max {a.max():5.2f} "
              f"| 后 P50 {np.median(b):5.2f} max {b.max():5.2f} (>12m {100*(b>12).mean():.1f}%)")
        print(f"     合并来自多框的 Truck 数: {total['merged']}")


if __name__ == '__main__':
    main()
