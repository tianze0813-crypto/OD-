#!/usr/bin/env python3
"""mmdet3d 1.x 版 BEVFusion 推理：交警 clip -> raw json（lidar_top 系，10 类）。

输出的 json 结构与 OpenPCDet 那条链路完全一致，所以 report_truck.py / eval_vs_ref.py /
nusc_official/export_sust.py 都能直接复用。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # <project>/bevfusion
PROJECT = ROOT.parent                                  # <project>
def _resolve_mmdet3d_root() -> Path:
    """mmdetection3d 源码位置：优先 MMDET3D_ROOT，其次猜常见路径。"""
    candidates = [os.environ.get("MMDET3D_ROOT"),
                  str(Path.home() / "MMDetection" / "mmdetection3d"),
                  str(Path.home() / "桌面" / "MMDetection" / "mmdetection3d")]
    for cand in candidates:
        if cand and (Path(cand) / "projects" / "BEVFusion").is_dir():
            return Path(cand)
    raise SystemExit("找不到 mmdetection3d 源码，请设环境变量 MMDET3D_ROOT 指向它")


MMDET = _resolve_mmdet3d_root()
sys.path.insert(0, str(MMDET))          # custom_imports: projects.BEVFusion.bevfusion
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from build_bev_pool import register as register_bev_pool  # noqa: E402

register_bev_pool()                     # 注入 bev_pool_ext（必须在 import 项目模块前）

from mmengine.config import Config  # noqa: E402
from mmengine.dataset import pseudo_collate  # noqa: E402
from mmengine.registry import init_default_scope  # noqa: E402
from mmdet3d.apis import init_model  # noqa: E402
from mmdet3d.registry import DATASETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg_file', default='configs/police_bevfusion_mmdet3d.py')
    ap.add_argument('--ckpt', default=str(PROJECT / 'models' / 'bevfusion_mmdet3d_lidarcam.pth'))
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--score_thresh', type=float, default=0.1)
    ap.add_argument('--clips', default='')
    ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--z-convention', default='center', choices=['center', 'bottom'],
                    help="mmdet3d 的 LiDAR 框 z 存的是【框底面】(origin=(0.5,0.5,0))，"
                         "而本目录其它链路/SUST/参考标注都是【框中心】。"
                         "默认 center: 输出前做 z += dz/2 的统一换算；bottom: 保持 mmdet3d 原始语义")
    args = ap.parse_args()

    cfg = Config.fromfile(args.cfg_file)
    # 数据根：BEVFUSION_DATA_ROOT 优先，否则用本工具箱的 data/police（配置里只是占位）
    data_root = os.environ.get('BEVFUSION_DATA_ROOT') or str(ROOT / 'data' / 'police')
    if not data_root.endswith('/'):
        data_root += '/'
    cfg.data_root = data_root
    cfg.val_dataloader.dataset.data_root = data_root
    print(f"[data] data_root = {data_root}")
    init_default_scope(cfg.get('default_scope', 'mmdet3d'))
    model = init_model(cfg, str(Path(args.ckpt).resolve()), device='cuda:0')
    model.eval()
    print(f"[model] {sum(p.numel() for p in model.parameters())/1e6:.1f}M 参数, ckpt={args.ckpt}")

    ds_cfg = cfg.val_dataloader.dataset
    dataset = DATASETS.build(ds_cfg)
    # mmengine 的 serialize_data 会把 dataset.data_list 清空，原始 info 直接读 ann 文件
    ann = Path(cfg.data_root) / ds_cfg['ann_file']
    import pickle
    raw_infos = pickle.load(open(ann, 'rb'))['data_list']
    print(f"[dataset] {len(dataset)} 帧 (ann={ann.name})")

    idx = list(range(len(dataset)))
    if args.clips:
        want = {c.strip() for c in args.clips.split(',') if c.strip()}
        idx = [i for i in idx if raw_infos[i]['scene_token'] in want]
    if args.max_frames > 0:
        idx = idx[:args.max_frames]
    print(f"[infer] {len(idx)} 帧, clips={sorted({raw_infos[i]['scene_token'] for i in idx})}")

    class_names = list(cfg.metainfo['classes'])
    results, times = [], []
    for i in idx:
        info = raw_infos[i]
        data = pseudo_collate([dataset[i]])
        t0 = time.time()
        with torch.no_grad():
            preds = model.test_step(data)
        times.append(time.time() - t0)
        inst = preds[0].pred_instances_3d
        boxes = inst.bboxes_3d.tensor.detach().cpu().numpy()
        scores = inst.scores_3d.detach().cpu().numpy()
        labels = inst.labels_3d.detach().cpu().numpy()
        keep = scores >= args.score_thresh
        dets = []
        for b, sc, l in zip(boxes[keep], scores[keep], labels[keep]):
            # mmdet3d 的 LiDAR 框是 9 列 (x,y,z,dx,dy,dz,yaw,vx,vy)；本目录统一只保留前 7 列
            bb = b[:7].astype(np.float64).copy()
            if args.z_convention == 'center':
                # mmdet3d 底面 -> 中心（在本征 lidar 系沿框自身 z 轴抬 dz/2）
                bb[2] += bb[5] / 2.0
            det = {'class_name': class_names[int(l)], 'score': float(sc),
                   'box_lidar': bb.tolist()}
            if len(b) > 7:
                det['velocity'] = [float(b[7]), float(b[8])]   # 保留速度，供后续筛选参考
            dets.append(det)
        ts = Path(info['lidar_points']['lidar_path']).stem
        results.append({'frame_id': ts, 'scene_token': info['scene_token'],
                        'num_points': -1, 'detections': dets})
        if (len(results)) % 20 == 0:
            print(f"  {len(results)}/{len(idx)}  {times[-1]*1000:.0f} ms/帧  框 {len(dets)}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False) + '\n', encoding='utf-8')
    import collections
    cnt = collections.Counter(d['class_name'] for fr in results for d in fr['detections'])
    print(f"[z] 约定: {args.z_convention}")
    print(f"[done] {out}  {len(results)} 帧 {sum(cnt.values())} 框 (阈值 {args.score_thresh})")
    print(f"       类别 {dict(cnt.most_common())}")
    if times:
        print(f"       单帧 P50 {np.median(times)*1000:.0f} ms")


if __name__ == '__main__':
    main()
