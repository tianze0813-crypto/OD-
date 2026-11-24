#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 eval_nuscenes_prelabel.py 产出的 CSV 汇总成 Markdown 报告（数值全部从 CSV 现算）。"""
import argparse
import csv
import json
import os

CLS_ORDER = ['Car', 'Truck', 'Pedestrian', 'Nonmotorized_vehicle']
CLS_CN = {'Car': '轿车', 'Truck': '大车', 'Pedestrian': '行人', 'Nonmotorized_vehicle': '非机动车'}


def read(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def f(x, nd=3):
    try:
        return f'{float(x):.{nd}f}'
    except (TypeError, ValueError):
        return '—'


def pct(x, nd=1):
    try:
        return f'{float(x)*100:.{nd}f}%'
    except (TypeError, ValueError):
        return '—'


def short_clip(name):
    n = name.replace('scene_1_', '')
    parts = n.split('_')
    batch = parts[1] if len(parts) > 1 else ''
    clip = [x for x in parts if x.startswith('clip')]
    return f'{batch}/{clip[0]}' if clip else n


def table(headers, rows):
    out = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True, help='eval_nuscenes_prelabel.py 的输出目录')
    ap.add_argument('--mode', default='near')
    ap.add_argument('--mode-label', default=None)
    ap.add_argument('--report', default=None)
    ap.add_argument('--title', default='非机动车道预标评测报告（nuScenes 官方口径）')
    ap.add_argument('--notes', default='', help='附加说明（追加在开头）')
    args = ap.parse_args()

    d = args.out_dir
    mode = args.mode
    meta = {}
    mp = os.path.join(d, 'meta.json')
    if os.path.exists(mp):
        meta = json.load(open(mp))
    ranges = meta.get('ranges', {})
    if args.mode_label:
        mode_label = args.mode_label
    elif ranges:
        mode_label = (f"近距（车 {ranges.get('Car', '?')}m、大车 {ranges.get('Truck', '?')}m、"
                      f"非机动车 {ranges.get('Nonmotorized_vehicle', '?')}m、行人 {ranges.get('Pedestrian', '?')}m）")
    else:
        mode_label = '近距口径'

    report_path = args.report or os.path.join(d, 'EVAL_REPORT.md')
    overall = {(r['mode'], r['scope']): r for r in read(os.path.join(d, 'metrics_overall.csv'))}
    byc = read(os.path.join(d, 'metrics_by_class.csv'))
    bins = read(os.path.join(d, 'metrics_by_range_bin.csv'))
    scenes = read(os.path.join(d, 'metrics_per_scene.csv'))
    clips = read(os.path.join(d, 'metrics_per_clip.csv'))
    usab = read(os.path.join(d, 'usability.csv'))
    conf = read(os.path.join(d, 'confusion.csv'))

    L = []
    A = L.append
    A(f'# {args.title}\n')
    scope = (f"{meta.get('n_clip', '?')} 个 clip / {meta.get('n_frame', '?')} 帧" if meta else '')
    A(f"> 生成时间：{meta.get('generated_at', '')} ｜ 数据：{scope}\n")
    if args.notes:
        A(args.notes + '\n')
    A('- 指标实现：nuScenes devkit **1.1.11** 的 `accumulate / calc_ap / calc_tp`'
      '（dist_ths=0.5/1/2/4m、dist_th_tp=2.0m、min_recall=0.1、min_precision=0.1），与官方函数逐类逐阈值对拍一致')
    A(f'- 距离口径：**{mode_label}**；另附全量（不限距）对照')
    A('')

    near, full = overall[(mode, 'ALL')], overall[('full', 'ALL')]
    A('## 一、结论速览\n')
    A('| 口径 | mAP | NDS(9项,含速度) | NDS(8项,不含速度) | mATE | mASE | mAOE | mAVE |')
    A('|---|---|---|---|---|---|---|---|')
    for tag, r in ((mode_label, near), ('全量（不限距）', full)):
        A(f"| {tag} | **{f(r['mAP'])}** | **{f(r['NDS9'])}** | {f(r['NDS8'])} | {f(r['mATE'])} | "
          f"{f(r['mASE'])} | {f(r['mAOE'])} | {f(r['mAVE'])} |")
    A('')
    A('> NDS 说明：官方 10 项 =（5·mAP + mATE + mASE + mAOE + mAVE + mAAE）/10；本批 label 无 attribute 字段，'
      'mAAE 无法计算，故 NDS(9项)=(5·mAP+4项TP)/9 为主口径，NDS(8项) 为不含速度的参考值。\n')

    A('## 二、分类别结果\n')
    for mo, tag in ((mode, mode_label), ('full', '全量')):
        A(f'### {tag}\n')
        rows = []
        for r in byc:
            if r['mode'] != mo:
                continue
            rows.append([CLS_CN.get(r['class'], r['class']), f(r['n_gt'], 0), f(r['n_pred'], 0),
                         f(r['AP@0.5']), f(r['AP@1.0']), f(r['AP@2.0']), f(r['AP@4.0']),
                         f"**{f(r['mAP'])}**", f(r['ATE']), f(r['ASE']), f(r['AOE']), f(r['AVE'])])
        A(table(['类别', 'GT 数', '预标框数', 'AP@0.5m', 'AP@1m', 'AP@2m', 'AP@4m', 'mAP',
                 'ATE(m)', 'ASE(1-IoU)', 'AOE(rad)', 'AVE(m/s)'], rows))
        A('')

    A('## 三、按目标距离分桶（GT 与预标两侧都按各自中心过滤）\n')
    rows = []
    for r in bins:
        if r['mode'] != 'full':
            continue
        def cell(cls):
            return f(r['mAP_' + cls]) if int(r['nGT_' + cls]) > 0 else '—'
        rows.append([r['bin'], f(r['n_gt'], 0), f(r['mAP']), f(r['mATE']), f(r['mASE']), f(r['mAOE']),
                     f(r['mAVE']), cell('Car'), cell('Truck'), cell('Pedestrian'),
                     cell('Nonmotorized_vehicle')])
    A(table(['距离(m)', 'GT 数', 'mAP', 'ATE', 'ASE', 'AOE', 'AVE', 'mAP 轿车', 'mAP 大车',
             'mAP 行人', 'mAP 非机动车'], rows))
    A('')
    rows = []
    for r in bins:
        if r['mode'] != 'full':
            continue
        rows.append([r['bin'], r['nGT_Car'], r['nPred_Car'], r['nGT_Truck'], r['nPred_Truck'],
                     r['nGT_Nonmotorized_vehicle'], r['nPred_Nonmotorized_vehicle']])
    A('**GT / 预标框数量对照**：\n')
    A(table(['距离(m)', 'GT 轿车', '预标轿车', 'GT 大车', '预标大车', 'GT 非机动车', '预标非机动车'], rows))
    A('')

    if scenes:
        A('## 四、分场景\n')
        rows = []
        for r in scenes:
            if r['mode'] != mode:
                continue
            rows.append([r['scene'].replace('scene_1_', ''), r['n_clip'], f(r['mAP']), f(r['NDS9']),
                         f(r['mATE']), f(r['mASE']), f(r['mAOE']), f(r['mAVE'])])
        A(table(['场景批次', 'clip 数', 'mAP', 'NDS9', 'ATE', 'ASE', 'AOE', 'AVE'], rows))
        A('')

    A('## 五、分 clip（按 mAP 排序）\n')
    rows = []
    data = [r for r in clips if r['mode'] == mode]
    data.sort(key=lambda r: -float(r['mAP']))
    for r in data:
        def cell2(cls):
            return f(r['mAP_' + cls]) if int(r['nGT_' + cls]) > 0 else '—'
        rows.append([short_clip(r['clip']), f(r['mAP']), f(r['NDS9']),
                     cell2('Car'), cell2('Truck'), cell2('Pedestrian'), cell2('Nonmotorized_vehicle'),
                     r['nGT_Car'], r['nPred_Car']])
    A(table(['clip', 'mAP', 'NDS9', '轿车', '大车', '行人', '非机动车', 'GT 轿车', '预标轿车'], rows))
    A('')

    A('## 六、类别混淆（每个 GT 找 <2m 的最近预测框）\n')
    for mo, tag in ((mode, mode_label), ('full', '全量')):
        rows = []
        per_gt = {}
        for r in conf:
            if r['mode'] != mo:
                continue
            per_gt[r['gt_class']] = per_gt.get(r['gt_class'], 0) + int(r['count'])
        for r in conf:
            if r['mode'] != mo:
                continue
            tot = per_gt.get(r['gt_class'], 1)
            rows.append([CLS_CN.get(r['gt_class'], r['gt_class']), r['pred_class'], r['count'],
                         pct(int(r['count']) / tot)])
        A(f'### {tag}\n')
        A(table(['GT 类别', '最近的预测类', '数量', '占比'], rows))
        A('')

    A('## 七、预标可用性：有多少可以不用人工改\n')
    A('匹配口径：与 GT 中心距离 <2m 视为命中（=nuScenes TP 阈值）；'
      '**几何达标** = 命中且中心误差<1m、尺度误差(1-IoU)<0.3（不看朝向）。\n')
    A('**覆盖率** = 几何达标数/GT；**净可用率** = 几何达标数/(GT+误检框数)。\n')
    for mo, tag in ((mode, mode_label), ('full', '全量')):
        A(f'### {tag}\n')
        rows = []
        tot_gt = tot_pred = tot_tp = tot_fp = tot_geo = tot_strict = 0
        for cls in CLS_ORDER:
            r = next((x for x in usab if x['mode'] == mo and x['class'] == cls and x['conf_th'] == '0.0'), None)
            if not r:
                continue
            n_gt, n_pred, tp, fp = int(r['n_gt']), int(r['n_pred']), int(r['tp']), int(r['fp'])
            geo, strict = int(r['clean_geo_tp']), int(r['clean_tp'])
            tot_gt += n_gt; tot_pred += n_pred; tot_tp += tp; tot_fp += fp; tot_geo += geo; tot_strict += strict
            rows.append([CLS_CN[cls], n_gt, n_pred, pct(r['precision']), pct(r['recall']), geo,
                         pct(geo / max(n_gt, 1)), pct(geo / max(n_gt + fp, 1)), pct(strict / max(n_gt, 1))])
        rows.append(['**合计**', tot_gt, tot_pred, pct(tot_tp / max(tot_pred, 1)), pct(tot_tp / max(tot_gt, 1)),
                     tot_geo, pct(tot_geo / max(tot_gt, 1)), pct(tot_geo / max(tot_gt + tot_fp, 1)),
                     pct(tot_strict / max(tot_gt, 1))])
        A(table(['类别', 'GT 数', '预标框数', '精确率', '召回率', '几何达标数', '覆盖率', '净可用率',
                 '严格达标率(含朝向)'], rows))
        A('')

    # ---------- 自动结论 ----------
    A('## 八、要点（按数据自动生成）\n')
    for r in byc:
        if r['mode'] != mode:
            continue
        cls = r['class']
        u = next((x for x in usab if x['mode'] == mode and x['class'] == cls and x['conf_th'] == '0.0'), None)
        miss = sum(int(c['count']) for c in conf if c['mode'] == mode and c['gt_class'] == cls
                   and c['pred_class'] == 'MISS')
        mis = {c['pred_class']: int(c['count']) for c in conf
               if c['mode'] == mode and c['gt_class'] == cls and c['pred_class'] not in ('MISS', cls)}
        mis.pop(cls, None)
        mis_txt = '、'.join(f'{k} {v}' for k, v in sorted(mis.items(), key=lambda kv: -kv[1])[:2])
        A(f"- **{CLS_CN[cls]}**：mAP {f(r['mAP'])}（AP@2m {f(r['AP@2.0'])}），ATE {f(r['ATE'])}m、AOE {f(r['AOE'])}rad；"
          f"GT {r['n_gt']}、预标框 {r['n_pred']}"
          + (f"，精确率 {pct(u['precision'])}、召回率 {pct(u['recall'])}、几何达标率 {pct(u['clean_geo_rate'])}" if u else '')
          + f"；完全漏检 {miss} 个" + (f"（被误判成 {mis_txt}）" if mis_txt else '') + '。')
    A('')
    A('**距离衰减（全量分桶）**：' + '；'.join(
        f"{r['bin']} mAP {f(r['mAP'])}（GT {r['n_gt']}）" for r in bins if r['mode'] == 'full') + '。')
    A('')
    worst = sorted([r for r in clips if r['mode'] == mode], key=lambda r: float(r['mAP']))[:3]
    best = sorted([r for r in clips if r['mode'] == mode], key=lambda r: -float(r['mAP']))[:3]
    A('**最好/最差 clip**：' + '，'.join(f"{short_clip(r['clip'])} {f(r['mAP'])}" for r in best) +
      ' ／ ' + '，'.join(f"{short_clip(r['clip'])} {f(r['mAP'])}" for r in worst) + '。')
    A('')
    A('## 九、复现方式\n')
    A('```bash')
    A('python3 scripts/eval_nuscenes_prelabel.py --root <DATA_ROOT> --clips <clips.txt> \\')
    A('    --ranges 50,50,50,20 --mode-name near --out <OUT_DIR>')
    A('python3 scripts/make_eval_report.py --out-dir <OUT_DIR>')
    A('```')

    open(report_path, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
    print('[report]', report_path)


if __name__ == '__main__':
    main()
