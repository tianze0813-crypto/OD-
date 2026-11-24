#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 nuScenes 官方评测口径评测「预标 vs 人工 GT」。

评测对象：
  - 预标: <pre_root>/<clip>/*.json      (SUSTechPOINTS label, zip 内 label/ 原样解出)
  - GT  : <gt_root>/<clip>/label/*.json (02_anno_result 回流的人工 GT)

口径（与 nuScenes devkit 1.1.11 完全一致）：
  - dist_fcn = center_distance (xy)，dist_ths = [0.5, 1.0, 2.0, 4.0]
  - mAP = 各距离阈值 AP 的均值；AP 用 101 点插值、min_recall=0.1、min_precision=0.1
  - TP 指标在 dist_th_tp=2.0m 的匹配上计算：mATE / mASE / mAOE / mAVE
  - NDS(9 项) = (5*mAP + mATE + mASE + mAOE + mAVE) / 9   （缺 attribute，mAAE 不计）
  - NDS(8 项) = (5*mAP + mATE + mASE + mAOE) / 8          （再缺 velocity）
  - 类别：Car / Truck / Pedestrian / Nonmotorized_vehicle
      GT 里的 Stone_block / Cone_block 等非评测类忽略（不当 GT，也不因漏检扣分）
      大车统一映射到 Truck（Bus/Van/Trailer/... -> Truck）
  - 两种距离口径：
      near：Car/Truck/Nonmotorized_vehicle <= 60m，Pedestrian <= 20m
      full：不过滤（全量）
    两侧都按「各自框中心」到 ego 原点(0,0) 的 XY 距离过滤
  - 速度：用 obj_id 轨迹跨帧中心差分（GT / 预标各自算），逐帧 10Hz，全部帧参与评测

用法::

    python3 scripts/eval_nuscenes_prelabel.py --root <DATA_ROOT> --clips clips.txt \
        --ranges 50,50,50,20 --mode-name near --out <OUT_DIR>

目录约定::

    <DATA_ROOT>/gt/<clip>/label/*.json    人工 GT（SUSTechPOINTS label）
    <DATA_ROOT>/pre/<clip>/*.json         预标（SUSTechPOINTS label）
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import types
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# 以最小依赖方式接入 nuScenes devkit 的评测函数
# （nuscenes.utils.data_classes 只是 eval/common/utils.py 的类型标注依赖，
#   这里用空壳替代，避免拖入 cv2 / sklearn / lidarseg 等无关依赖）
# ---------------------------------------------------------------------------
_nusc_pkg_dir = None
for _p in list(sys.path):
    if os.path.isdir(os.path.join(_p, 'nuscenes', 'eval')):
        _nusc_pkg_dir = os.path.join(_p, 'nuscenes')
        break
if _nusc_pkg_dir is None:
    raise ImportError('nuscenes devkit not found in sys.path')
_pkg_nusc = types.ModuleType('nuscenes')
_pkg_nusc.__path__ = [_nusc_pkg_dir]          # 不执行真正的 nuscenes/__init__.py（避免拖入 cv2/sklearn）
sys.modules['nuscenes'] = _pkg_nusc

_pkg_utils = types.ModuleType('nuscenes.utils')
_pkg_utils.__path__ = []
sys.modules['nuscenes.utils'] = _pkg_utils
_mod_dc = types.ModuleType('nuscenes.utils.data_classes')
_mod_dc.Box = object
sys.modules['nuscenes.utils.data_classes'] = _mod_dc

from nuscenes.eval.common.data_classes import EvalBoxes                       # noqa: E402
from nuscenes.eval.common.utils import (center_distance, scale_iou, yaw_diff,  # noqa: E402
                                        velocity_l2, attr_acc, cummean)
from nuscenes.eval.detection.data_classes import DetectionBox, DetectionMetricData  # noqa: E402
from nuscenes.eval.detection import data_classes as _det_dc                   # noqa: E402
from nuscenes.eval.detection.algo import calc_ap, calc_tp, accumulate as _official_accumulate  # noqa: E402

EVAL_CLASSES = ['Car', 'Truck', 'Pedestrian', 'Nonmotorized_vehicle']
# 允许非 nuScenes 原生类名通过 DetectionBox 的断言
for _c in EVAL_CLASSES:
    if _c not in _det_dc.DETECTION_NAMES:
        _det_dc.DETECTION_NAMES.append(_c)

DIST_THS = [0.5, 1.0, 2.0, 4.0]     # nuScenes 官方 dist_ths
DIST_TH_TP = 2.0                     # nuScenes 官方 dist_th_tp
MIN_RECALL = 0.1
MIN_PRECISION = 0.1
MEAN_AP_WEIGHT = 5

# 大车 -> Truck；GT 里出现的新类名进 UNKNOWN 统计
CLASS_MAP = {
    'Car': 'Car', 'car': 'Car',
    'Truck': 'Truck', 'truck': 'Truck', 'Bus': 'Truck', 'bus': 'Truck',
    'Van': 'Truck', 'van': 'Truck', 'Trailer': 'Truck', 'trailer': 'Truck',
    'Truck_head': 'Truck', 'tractor': 'Truck', 'Lorry': 'Truck',
    'Construction_vehicle': 'Truck', 'construction_vehicle': 'Truck',
    'Pedestrian': 'Pedestrian', 'pedestrian': 'Pedestrian', 'Ped': 'Pedestrian',
    'Nonmotorized_vehicle': 'Nonmotorized_vehicle',
    'nonmotorized_vehicle': 'Nonmotorized_vehicle',
    'Nonmotorized': 'Nonmotorized_vehicle', 'Cyclist': 'Nonmotorized_vehicle',
    'Bicycle': 'Nonmotorized_vehicle', 'bicycle': 'Nonmotorized_vehicle',
    'Motorcycle': 'Nonmotorized_vehicle', 'motorcycle': 'Nonmotorized_vehicle',
    'Bicyclist': 'Nonmotorized_vehicle', 'Tricycle': 'Nonmotorized_vehicle',
}
IGNORE_TYPES = {'Stone_block', 'Cone_block', 'Barrier', 'Traffic_cone', 'cone', 'other'}

RANGE_NEAR = {'Car': 60.0, 'Truck': 60.0, 'Nonmotorized_vehicle': 60.0, 'Pedestrian': 20.0}
NEAR_MODE = 'near'   # 近距口径在输出里的 mode 名称，由 --mode-name 覆盖


# ---------------------------------------------------------------------------
# 读 label
# ---------------------------------------------------------------------------
def load_frames(label_dir):
    """{ts_ns(int): [box, ...]}，box 统一成 dict。"""
    frames = {}
    for path in glob.glob(os.path.join(label_dir, '*.json')):
        ts = int(os.path.basename(path)[:-5])
        objs = json.load(open(path))
        out = []
        for o in objs:
            try:
                psr = o['psr']
                p, s, r = psr['position'], psr['scale'], psr['rotation']
                xy = (float(p['x']), float(p['y']))
                if not all(map(math.isfinite, (p['x'], p['y'], p['z'], s['x'], s['y'], s['z'], r['z']))):
                    continue
                out.append({
                    'raw_cls': o['obj_type'],
                    'oid': str(o.get('obj_id')),
                    'x': xy[0], 'y': xy[1], 'z': float(p['z']),
                    'l': float(s['x']), 'w': float(s['y']), 'h': float(s['z']),
                    'yaw': float(r['z']),
                    'score': float(o['score']) if o.get('score') is not None else None,
                    'vx': 0.0, 'vy': 0.0,
                })
            except Exception:
                continue
        frames[ts] = out
    return frames


def add_eval_class(frames, stats):
    for ts, boxes in frames.items():
        for b in boxes:
            c = CLASS_MAP.get(b['raw_cls'])
            if c is None:
                stats['unknown'][b['raw_cls']] += 1
                c = None
            else:
                stats['cls_count'][b['raw_cls']] += 1
            b['cls'] = c
    return frames


def compute_velocity(frames, dt_max=0.2):
    """用 obj_id 轨迹做中心差分；ego 系下 GT/预标同样处理，AVE 与全局系等价。"""
    tracks = defaultdict(list)
    for ts in sorted(frames):
        for b in frames[ts]:
            tracks[(b['cls'], b['oid'])].append((ts, b))
    for lst in tracks.values():
        lst.sort(key=lambda t: t[0])
        n = len(lst)
        for i, (ts, b) in enumerate(lst):
            vx = vy = 0.0
            done = False
            if 0 < i < n - 1:
                t0, b0 = lst[i - 1]
                t2, b2 = lst[i + 1]
                dt = (t2 - t0) / 1e9
                if 0 < dt <= 2 * dt_max:
                    vx, vy = (b2['x'] - b0['x']) / dt, (b2['y'] - b0['y']) / dt
                    done = True
            if not done and i + 1 < n:
                t2, b2 = lst[i + 1]
                dt = (t2 - ts) / 1e9
                if 0 < dt <= dt_max:
                    vx, vy = (b2['x'] - b['x']) / dt, (b2['y'] - b['y']) / dt
                    done = True
            if not done and i > 0:
                t0, b0 = lst[i - 1]
                dt = (ts - t0) / 1e9
                if 0 < dt <= dt_max:
                    vx, vy = (b['x'] - b0['x']) / dt, (b['y'] - b0['y']) / dt
                    done = True
            if done:
                b['vx'], b['vy'] = vx, vy
    return frames


# ---------------------------------------------------------------------------
# 组 EvalBoxes
# ---------------------------------------------------------------------------
def make_box(token, b):
    from pyquaternion import Quaternion
    return DetectionBox(
        sample_token=token,
        translation=(b['x'], b['y'], b['z']),
        size=(max(b['w'], 1e-3), max(b['l'], 1e-3), max(b['h'], 1e-3)),   # nuScenes: (w, l, h)
        rotation=tuple(Quaternion(axis=[0, 0, 1], angle=b['yaw']).elements),
        velocity=(b['vx'], b['vy']),
        ego_translation=(b['x'], b['y'], b['z']),
        num_pts=1,
        detection_name=b['cls'],
        detection_score=float(b['score']) if b['score'] is not None else -1.0,
        attribute_name='',
    )


def build_evalboxes(frames, clip, mode):
    eb = EvalBoxes()
    for ts in sorted(frames):
        token = f'{clip}::{ts}'
        boxes = []
        for b in frames[ts]:
            if b['cls'] is None:
                continue
            if mode == NEAR_MODE and math.hypot(b['x'], b['y']) > RANGE_NEAR[b['cls']]:
                continue
            boxes.append(make_box(token, b))
        eb.add_boxes(token, boxes)
    return eb


def merge_evalboxes(items):
    out = EvalBoxes()
    for eb in items:
        for tok in eb.sample_tokens:
            out.add_boxes(tok, eb[tok])
    return out


def filter_evalboxes(eb, lo, hi):
    out = EvalBoxes()
    for tok in eb.sample_tokens:
        keep = [b for b in eb[tok] if lo <= math.hypot(b.translation[0], b.translation[1]) < hi]
        out.add_boxes(tok, keep)
    return out


# ---------------------------------------------------------------------------
# 快速版 accumulate：与官方 accumulate 语义一致，只把「找最近未匹配 GT」向量化
# ---------------------------------------------------------------------------
def prep_cache(gt_boxes, pred_boxes, class_name):
    """按官方 accumulate 的遍历顺序预置：predictions 全局按 (score, index) 降序。"""
    gt_by = defaultdict(list)
    for b in gt_boxes.all:
        if b.detection_name == class_name:
            gt_by[b.sample_token].append(b)

    preds = [b for b in pred_boxes.all if b.detection_name == class_name]   # 官方 pred_boxes.all 顺序
    order = sorted(range(len(preds)), key=lambda i: (preds[i].detection_score, i), reverse=True)
    pred_by_frame = defaultdict(list)
    for k in order:                       # 已按官方顺序排好
        pred_by_frame[preds[k].sample_token].append(k)

    frames = []
    npos = 0
    for tok in gt_boxes.sample_tokens:
        G = gt_by.get(tok, [])
        ks = pred_by_frame.get(tok, [])
        npos += len(G)
        P = [preds[k] for k in ks]
        gc = np.array([g.translation[:2] for g in G], dtype=float).reshape(-1, 2)
        pc = np.array([p.translation[:2] for p in P], dtype=float).reshape(-1, 2)
        D = np.linalg.norm(pc[:, None, :] - gc[None, :, :], axis=2) if (len(P) and len(G)) \
            else np.zeros((len(P), len(G)), dtype=float)
        frames.append((tok, ks, P, G, D))
    return {'frames': frames, 'preds': preds, 'order': order, 'npos': npos}


def match_frames(cache, dist_th, conf_th=None):
    """逐帧贪心匹配（置信度降序），返回 {pred_index: (is_tp, gt_box)}。"""
    outcome = {}
    for tok, ks, P, G, D in cache['frames']:
        if not P:
            continue
        idx = [j for j, k in enumerate(ks)
               if conf_th is None or cache['preds'][k].detection_score >= conf_th]
        if not idx:
            continue
        taken = np.zeros(len(G), dtype=bool)
        for j in idx:
            row = np.where(taken, np.inf, D[j])
            if len(G):
                jj = int(np.argmin(row))
                if row[jj] < dist_th:
                    taken[jj] = True
                    outcome[ks[j]] = (True, G[jj])
                    continue
            outcome[ks[j]] = (False, None)
    return outcome


def accumulate_fast(cache, dist_th):
    """复刻 nuScenes devkit accumulate()，返回 DetectionMetricData。"""
    npos = cache['npos']
    if npos == 0:
        return DetectionMetricData.no_predictions()

    outcome = match_frames(cache, dist_th)
    preds = cache['preds']

    tp, fp, conf = [], [], []
    md = {'trans_err': [], 'vel_err': [], 'scale_err': [], 'orient_err': [], 'attr_err': [], 'conf': []}
    for k in cache['order']:
        p = preds[k]
        is_tp, g = outcome.get(k, (False, None))
        if is_tp:
            tp.append(1); fp.append(0); conf.append(p.detection_score)
            md['trans_err'].append(center_distance(g, p))
            md['vel_err'].append(velocity_l2(g, p))
            md['scale_err'].append(1 - scale_iou(g, p))
            md['orient_err'].append(yaw_diff(g, p, period=2 * np.pi))
            md['attr_err'].append(1 - attr_acc(g, p))
            md['conf'].append(p.detection_score)
        else:
            tp.append(0); fp.append(1); conf.append(p.detection_score)

    if len(md['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp_c = np.cumsum(tp).astype(float)
    fp_c = np.cumsum(fp).astype(float)
    conf_arr = np.array(conf)
    prec = tp_c / (fp_c + tp_c)
    rec = tp_c / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf_i = np.interp(rec_interp, rec, conf_arr, right=0)
    rec = rec_interp

    for key in md:
        if key == 'conf':
            continue
        tmp = cummean(np.array(md[key]))
        md[key] = np.interp(conf_i[::-1], np.array(md['conf'])[::-1], tmp[::-1])[::-1]

    return DetectionMetricData(recall=rec, precision=prec, confidence=conf_i,
                               trans_err=md['trans_err'], vel_err=md['vel_err'],
                               scale_err=md['scale_err'], orient_err=md['orient_err'],
                               attr_err=md['attr_err'])


def confusion(gt_boxes, pred_boxes, classes):
    """每个 GT 框找最近的预测框（任意类，<2m）：统计「类对且几何对 / 类错 / 完全漏检」。"""
    import math as _m
    rows = defaultdict(int)
    for tok in gt_boxes.sample_tokens:
        G = [b for b in gt_boxes[tok] if b.detection_name in classes]
        P = list(pred_boxes[tok])
        if not G:
            continue
        if not P:
            for g in G:
                rows[(g.detection_name, 'MISS')] += 1
            continue
        gc = np.array([g.translation[:2] for g in G], dtype=float)
        pc = np.array([p.translation[:2] for p in P], dtype=float)
        D = np.linalg.norm(gc[:, None, :] - pc[None, :, :], axis=2)
        j = np.argmin(D, axis=1)
        for i, g in enumerate(G):
            d = D[i, j[i]]
            if d < DIST_TH_TP:
                rows[(g.detection_name, P[j[i]].detection_name)] += 1
            else:
                rows[(g.detection_name, 'MISS')] += 1
    return rows


def eval_boxes(gt_boxes, pred_boxes, classes, fast=True):
    out = {}
    for cls in classes:
        cache = prep_cache(gt_boxes, pred_boxes, cls)
        aps = []
        mds = {}
        for dth in DIST_THS:
            md = accumulate_fast(cache, dth)
            mds[dth] = md
            aps.append(calc_ap(md, MIN_RECALL, MIN_PRECISION))
        md_tp = mds[DIST_TH_TP]
        out[cls] = {
            'ap': aps,
            'mAP': float(np.mean(aps)),
            'mATE': float(calc_tp(md_tp, MIN_RECALL, 'trans_err')),
            'mASE': float(calc_tp(md_tp, MIN_RECALL, 'scale_err')),
            'mAOE': float(calc_tp(md_tp, MIN_RECALL, 'orient_err')),
            'mAVE': float(calc_tp(md_tp, MIN_RECALL, 'vel_err')),
            'n_gt': cache['npos'],
            'n_pred': len(cache['preds']),
        }
    return out


def nds_scores(res, classes):
    valid = [c for c in classes if res[c]['n_gt'] > 0]
    if not valid:
        return {'mAP': float('nan'), 'mATE': float('nan'), 'mASE': float('nan'),
                'mAOE': float('nan'), 'mAVE': float('nan'), 'NDS9': float('nan'), 'NDS8': float('nan')}
    mAP = float(np.mean([res[c]['mAP'] for c in valid]))
    ate = float(np.mean([res[c]['mATE'] for c in valid]))
    ase = float(np.mean([res[c]['mASE'] for c in valid]))
    aoe = float(np.mean([res[c]['mAOE'] for c in valid]))
    ave = float(np.mean([res[c]['mAVE'] for c in valid]))
    # 官方 DetectionMetrics.tp_scores = max(0, 1 - error)，NDS = (5*mAP + Σ(1-error)) / (5 + #TP)
    nds9 = (MEAN_AP_WEIGHT * mAP + sum(max(0.0, 1.0 - e) for e in (ate, ase, aoe, ave))) / (MEAN_AP_WEIGHT + 4)
    nds8 = (MEAN_AP_WEIGHT * mAP + sum(max(0.0, 1.0 - e) for e in (ate, ase, aoe))) / (MEAN_AP_WEIGHT + 3)
    return {'mAP': mAP, 'mATE': ate, 'mASE': ase, 'mAOE': aoe, 'mAVE': ave,
            'NDS9': nds9, 'NDS8': nds8}


# ---------------------------------------------------------------------------
# 预标可用性：2m 匹配下的 P/R（模拟“不人工干预能直接用多少”）
# ---------------------------------------------------------------------------
def usability(gt_boxes, pred_boxes, classes, conf_ths=(0.0, 0.1, 0.3, 0.5)):
    """2m 匹配下的 P/R + 三误差同时达标的“可直接采纳率”。"""
    rows = []
    for cls in classes:
        cache = prep_cache(gt_boxes, pred_boxes, cls)
        for cth in conf_ths:
            outcome = match_frames(cache, DIST_TH_TP, conf_th=cth)
            n_tp = sum(1 for v in outcome.values() if v[0])
            n_fp = sum(1 for v in outcome.values() if not v[0])
            n_pred = n_tp + n_fp
            n_clean = 0
            n_clean_geo = 0
            for k, (is_tp, g) in outcome.items():
                if not is_tp:
                    continue
                p = cache['preds'][k]
                geo_ok = center_distance(g, p) < 1.0 and (1 - scale_iou(g, p)) < 0.3
                if geo_ok:
                    n_clean_geo += 1
                if geo_ok and yaw_diff(g, p, period=2 * np.pi) < 0.3:
                    n_clean += 1
            npos = cache['npos']
            fn = npos - n_tp
            prec = n_tp / n_pred if n_pred else 0.0
            rec = n_tp / npos if npos else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            rows.append({'class': cls, 'conf_th': cth, 'n_gt': npos, 'n_pred': n_pred,
                         'tp': n_tp, 'fp': n_fp, 'fn': fn, 'precision': prec, 'recall': rec,
                         'f1': f1, 'clean_tp': n_clean,
                         'clean_rate': (n_clean / npos) if npos else 0.0,
                         'clean_prec': (n_clean / n_pred) if n_pred else 0.0,
                         'clean_geo_tp': n_clean_geo,
                         'clean_geo_rate': (n_clean_geo / npos) if npos else 0.0,
                         'clean_geo_prec': (n_clean_geo / n_pred) if n_pred else 0.0})
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True,
                    help='数据根目录：<root>/gt/<clip>/label/*.json 与 <root>/pre/<clip>/*.json')
    ap.add_argument('--clips', required=True, help='每行一个 clip 名的文本文件')
    ap.add_argument('--out', default=None)
    ap.add_argument('--clips-limit', type=int, default=0)
    ap.add_argument('--ranges', default='60,60,60,20',
                    help='近距口径上限，顺序 Car,Truck,Nonmotorized_vehicle,Pedestrian（米）')
    ap.add_argument('--mode-name', default='near', help='近距口径在输出中的 mode 名称')
    args = ap.parse_args()

    global RANGE_NEAR, NEAR_MODE
    NEAR_MODE = args.mode_name
    vals = [float(x) for x in args.ranges.split(',')]
    assert len(vals) == 4, '--ranges 需要 4 个数：Car,Truck,Nonmotor,Pedestrian'
    RANGE_NEAR = dict(zip(['Car', 'Truck', 'Nonmotorized_vehicle', 'Pedestrian'], vals))

    root = args.root
    out_dir = args.out or os.path.join(root, 'out')
    os.makedirs(out_dir, exist_ok=True)
    clips = [l.strip() for l in open(args.clips) if l.strip()]
    if args.clips_limit:
        clips = clips[:args.clips_limit]

    stats = {'unknown': defaultdict(int), 'cls_count': defaultdict(int)}
    gt_all, pre_all = {}, {}
    for c in clips:
        gt_dir = os.path.join(root, 'gt', c, 'label')
        pre_dir = os.path.join(root, 'pre', c)
        g = add_eval_class(load_frames(gt_dir), stats)
        p = add_eval_class(load_frames(pre_dir), stats)
        compute_velocity(g)
        compute_velocity(p)
        gt_all[c], pre_all[c] = g, p
        common = set(g) & set(p)
        print(f'[load] {c}: frames gt={len(g)} pre={len(p)} common={len(common)}')

    print('[types] GT   :', dict(stats['cls_count']))
    print('[types] 未映射:', dict(stats['unknown']))

    results = {}
    for mode in (NEAR_MODE, 'full'):
        gt_eb, pre_eb = {}, {}
        for c in clips:
            # 只评测两边都有的帧
            common = set(gt_all[c]) & set(pre_all[c])
            g = {ts: gt_all[c][ts] for ts in common}
            p = {ts: pre_all[c][ts] for ts in common}
            gt_eb[c] = build_evalboxes(g, c, mode)
            pre_eb[c] = build_evalboxes(p, c, mode)
        results[mode] = {'gt': gt_eb, 'pre': pre_eb}

    report = {}
    rows_class, rows_overall, rows_clip, rows_scene, rows_bin = [], [], [], [], []
    rows_conf = []
    usab_rows = []

    for mode in (NEAR_MODE, 'full'):
        gt_eb, pre_eb = results[mode]['gt'], results[mode]['pre']
        g_all = merge_evalboxes([gt_eb[c] for c in clips])
        p_all = merge_evalboxes([pre_eb[c] for c in clips])

        res = eval_boxes(g_all, p_all, EVAL_CLASSES)
        overall = nds_scores(res, EVAL_CLASSES)
        report[(mode, 'all')] = (res, overall)
        rows_overall.append({'mode': mode, 'scope': 'ALL', **overall})
        for cls in EVAL_CLASSES:
            r = res[cls]
            rows_class.append({'mode': mode, 'class': cls,
                               'AP@0.5': r['ap'][0], 'AP@1.0': r['ap'][1],
                               'AP@2.0': r['ap'][2], 'AP@4.0': r['ap'][3],
                               'mAP': r['mAP'], 'ATE': r['mATE'], 'ASE': r['mASE'],
                               'AOE': r['mAOE'], 'AVE': r['mAVE'],
                               'n_gt': r['n_gt'], 'n_pred': r['n_pred']})

        # 分场景
        scenes = sorted({c.split('_nonmotor')[0] for c in clips})
        for sc in scenes:
            cs = [c for c in clips if c.startswith(sc)]
            res_s = eval_boxes(merge_evalboxes([gt_eb[c] for c in cs]),
                               merge_evalboxes([pre_eb[c] for c in cs]), EVAL_CLASSES)
            ov = nds_scores(res_s, EVAL_CLASSES)
            rows_scene.append({'mode': mode, 'scene': sc, 'n_clip': len(cs), **ov})

        # 每 clip
        for c in clips:
            res_c = eval_boxes(gt_eb[c], pre_eb[c], EVAL_CLASSES)
            ov = nds_scores(res_c, EVAL_CLASSES)
            row = {'mode': mode, 'clip': c, **{k: ov[k] for k in ('mAP', 'NDS9', 'NDS8')}}
            for cls in EVAL_CLASSES:
                row['mAP_' + cls] = res_c[cls]['mAP']
                row['nGT_' + cls] = res_c[cls]['n_gt']
                row['nPred_' + cls] = res_c[cls]['n_pred']
            rows_clip.append(row)

        # 距离分桶
        bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 1e9)]
        for lo, hi in bins:
            g_bin = filter_evalboxes(g_all, lo, hi)
            n_gt_bin = len(g_bin.all)
            name = f'{lo}-{hi}' if hi < 1e8 else f'{lo}+'
            if n_gt_bin == 0:
                rows_bin.append({'mode': mode, 'bin': name, 'mAP': None, 'n_gt': 0,
                                 **{f'nGT_{c}': 0 for c in EVAL_CLASSES}})
                continue
            res_b = eval_boxes(g_bin, filter_evalboxes(p_all, lo, hi), EVAL_CLASSES)
            ov = nds_scores(res_b, EVAL_CLASSES)
            rows_bin.append({'mode': mode, 'bin': name, **ov, 'n_gt': n_gt_bin,
                             **{f'mAP_{c}': res_b[c]['mAP'] for c in EVAL_CLASSES},
                             **{f'nGT_{c}': res_b[c]['n_gt'] for c in EVAL_CLASSES},
                             **{f'nPred_{c}': res_b[c]['n_pred'] for c in EVAL_CLASSES}})

        # 类别混淆
        cm = confusion(g_all, p_all, EVAL_CLASSES)
        for (gc, pc), n in sorted(cm.items(), key=lambda kv: (-kv[1], kv[0])):
            rows_conf.append({'mode': mode, 'gt_class': gc, 'pred_class': pc, 'count': n})

        # 可用性
        us = usability(g_all, p_all, EVAL_CLASSES)
        for r in us:
            r['mode'] = mode
        usab_rows.extend(us)

    # ---------------- 落盘 ----------------
    def dump(name, rows, fields):
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: ('' if v is None else (round(v, 6) if isinstance(v, float) else v))
                            for k, v in r.items()})
        print('[out]', path)

    dump('metrics_by_class.csv', rows_class,
         ['mode', 'class', 'AP@0.5', 'AP@1.0', 'AP@2.0', 'AP@4.0', 'mAP', 'ATE', 'ASE', 'AOE', 'AVE', 'n_gt', 'n_pred'])
    dump('metrics_overall.csv', rows_overall,
         ['mode', 'scope', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'NDS9', 'NDS8'])
    dump('metrics_per_clip.csv', rows_clip,
         ['mode', 'clip', 'mAP', 'NDS9', 'NDS8'] + [f'{p}_{c}' for c in EVAL_CLASSES for p in ('mAP', 'nGT', 'nPred')])
    dump('metrics_per_scene.csv', rows_scene, ['mode', 'scene', 'n_clip', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'NDS9', 'NDS8'])
    dump('metrics_by_range_bin.csv', rows_bin,
         ['mode', 'bin', 'mAP', 'mATE', 'mASE', 'mAOE', 'mAVE', 'NDS9', 'NDS8', 'n_gt'] +
         [f'{p}_{c}' for c in EVAL_CLASSES for p in ('mAP', 'nGT', 'nPred')])
    dump('confusion.csv', rows_conf, ['mode', 'gt_class', 'pred_class', 'count'])
    dump('usability.csv', usab_rows,
         ['mode', 'class', 'conf_th', 'n_gt', 'n_pred', 'tp', 'fp', 'fn', 'precision', 'recall', 'f1',
          'clean_tp', 'clean_rate', 'clean_prec', 'clean_geo_tp', 'clean_geo_rate', 'clean_geo_prec'])

    import datetime
    n_frames = sum(len(set(gt_all[c]) & set(pre_all[c])) for c in clips)
    meta = {
        'generated_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'n_clip': len(clips), 'n_frame': n_frames, 'clips': clips,
        'mode_name': NEAR_MODE, 'ranges': RANGE_NEAR,
        'eval_classes': EVAL_CLASSES,
        'dist_ths': DIST_THS, 'dist_th_tp': DIST_TH_TP,
        'min_recall': MIN_RECALL, 'min_precision': MIN_PRECISION,
        'gt_counts': {k: int(v) for k, v in stats['cls_count'].items()},
        'ignored_types': {k: int(v) for k, v in stats['unknown'].items()},
    }
    json.dump(meta, open(os.path.join(out_dir, 'meta.json'), 'w'), ensure_ascii=False, indent=1)
    json.dump({f'{m}|{s}': {'metrics': r[0], 'overall': r[1]} for (m, s), r in report.items()},
              open(os.path.join(out_dir, 'metrics_raw.json'), 'w'), ensure_ascii=False, indent=1)
    print('[done]', out_dir)


if __name__ == '__main__':
    main()
