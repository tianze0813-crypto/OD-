#!/usr/bin/env python3
"""BEVFusion 的 Truck 结果 vs 参考标注（现有链的机器预标）做匹配统计。

⚠️ 参考标注是「机器预标」不是人工 GT —— 所以这是 A/B 对比（BEVFusion vs 现有链），
   不是绝对精度。口径与现有链一致：Bus→Truck；匹配用 BEV 中心距 <= --dist。

用法: python scripts/eval_vs_ref.py [--ref-root work/ref_labels] [--dist 2.0]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HYB = Path(__file__).resolve().parent.parent.parent   # 本项目根（tracking/geometry 就在下面）
sys.path.insert(0, str(HYB))
from tracking import tracker_conservative as tracking  # noqa: E402

GROUP_TRUCK = {"truck", "bus"}
GROUP_TRAILER = {"trailer"}


def load_ref(clip_dir: Path, base_from_lidar, score_thresh=0.0):
    """参考标注：base_link 系，已带 score（机器预标）。"""
    ref = []
    for f in sorted(clip_dir.glob("*.json")):
        boxes = []
        for o in json.load(open(f)):
            if o["obj_type"] != "Truck":
                continue
            if float(o.get("score", 1.0)) < score_thresh:
                continue
            s, p, r = o["psr"]["scale"], o["psr"]["position"], o["psr"]["rotation"]
            boxes.append([p["x"], p["y"], p["z"], s["x"], s["y"], s["z"], r["z"]])
        ref.append((f.stem, np.asarray(boxes, np.float64).reshape(-1, 7)))
    return ref


def load_pred(raw_json: Path, clip_dir: Path, th: float, group):
    base_from_lidar = np.asarray(
        json.loads((clip_dir / "transforms" / "calib.json").read_text())["tf2base_link"]["lidar_top"],
        np.float64)
    out = []
    for fr in json.loads(raw_json.read_text()):
        boxes = []
        for d in fr["detections"]:
            if d["score"] < th or d["class_name"] not in group:
                continue
            boxes.append(tracking.box_lidar_to_base_link(d["box_lidar"][:7], base_from_lidar))
        out.append((fr["frame_id"], np.asarray(boxes, np.float64).reshape(-1, 7)))
    return out


def greedy(pred, gt, dist):
    used = np.zeros(len(gt), bool)
    pairs = []
    for pi in np.argsort(-pred[:, 6] if pred.shape[1] > 6 else np.zeros(len(pred))):
        d = np.hypot(gt[:, 0] - pred[pi, 0], gt[:, 1] - pred[pi, 1])
        d[used] = 1e9
        if len(d):
            gi = int(np.argmin(d))
            if d[gi] <= dist:
                used[gi] = True
                pairs.append((pi, gi))
    return pairs, used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="存 clip 的目录")
    ap.add_argument("--raw-glob", default="out/*.raw.json")
    ap.add_argument("--ref-root", default="work/ref_labels")
    ap.add_argument("--ref-score", type=float, default=0.4,
                    help="参考标注自身的分数门限（现有链 Truck 用 0.4）")
    ap.add_argument("--dist", type=float, default=2.0)
    ap.add_argument("--score-list", default="0.2,0.3,0.4")
    args = ap.parse_args()

    data_root, ref_root = Path(args.data_root), Path(args.ref_root)
    ths = [float(x) for x in args.score_list.split(",")]
    total = {t: dict(npred=0, ngt=0, tp=0, dl=[], tp_len=[], gt_len=[], tp_len_long=0, n_long=0)
             for t in ths}
    print(f"参考标注: {args.ref_root}（机器预标, 自身 score>={args.ref_score}）  匹配: BEV 中心距<={args.dist}m")
    for raw in sorted(Path().glob(args.raw_glob)):
        clip = raw.name.replace(".raw.json", "")
        cdir = data_root / clip
        rdir = ref_root / clip
        if not (cdir.exists() and rdir.exists()):
            continue
        base_from_lidar = np.asarray(
            json.loads((cdir / "transforms" / "calib.json").read_text())["tf2base_link"]["lidar_top"], np.float64)
        ref = load_ref(rdir, base_from_lidar, args.ref_score)
        ref_by_ts = dict(ref)
        line = []
        for th in ths:
            pred = dict(load_pred(raw, cdir, th, GROUP_TRUCK))
            st = total[th]
            for ts, gtb in ref:
                st["ngt"] += len(gtb)
                st["n_long"] += int((gtb[:, 3] > 12).sum()) if len(gtb) else 0
                pb = pred.get(ts, np.zeros((0, 7)))
                st["npred"] += len(pb)
                if not len(gtb) or not len(pb):
                    continue
                ps = np.hstack([pb, np.ones((len(pb), 1))])   # 占位第7列
                pairs, used = greedy(ps, gtb, args.dist)
                st["tp"] += len(pairs)
                long_hit = 0
                for pi, gi in pairs:
                    st["dl"].append(pb[pi, 3] - gtb[gi, 3])
                    st["tp_len"].append(pb[pi, 3])
                    st["gt_len"].append(gtb[gi, 3])
                    if gtb[gi, 3] > 12:
                        long_hit += 1
                st["tp_len_long"] += long_hit
            r = st["tp"] / max(st["ngt"], 1)
            p = st["tp"] / max(st["npred"], 1)
            line.append(f"th{th}: 召回 {r*100:4.1f}% 精度 {p*100:4.1f}%")
        print(f"  {clip[-24:]:24s} ref Truck {sum(len(v) for v in ref_by_ts.values()):4d} 框 | " + " | ".join(line))
    print("\n== 汇总（5 条 clip）")
    for th in ths:
        st = total[th]
        dl = np.asarray(st["dl"]) if st["dl"] else np.zeros(0)
        print(f"  阈值 {th}: 参考 {st['ngt']} 框 / 预测 {st['npred']} 框 / 命中 {st['tp']} "
              f"| 召回 {st['tp']/max(st['ngt'],1)*100:5.1f}%  精度 {st['tp']/max(st['npred'],1)*100:5.1f}%")
        if len(dl):
            print(f"           长度偏差 (预测-参考) P50 {np.median(dl):+5.2f} P90 {np.percentile(dl,90):+5.2f} m"
                  f" | 命中框长 P50 {np.median(st['tp_len']):5.2f} m")
        print(f"           长车(参考 >12m) {st['n_long']} 个，命中 {st['tp_len_long']} "
              f"(召回 {st['tp_len_long']/max(st['n_long'],1)*100:.1f}%)")


if __name__ == "__main__":
    main()
