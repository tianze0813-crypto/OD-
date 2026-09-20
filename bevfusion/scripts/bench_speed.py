#!/usr/bin/env python3
"""测「每帧」耗时拆解：数据加载/预处理 vs 模型前向。用法（mmdet3d 环境）:
  python scripts/bench_speed.py --pipeline mmdet3d           --n 20   # C+L
  python scripts/bench_speed.py --pipeline mmdet3d_lidaronly --n 20   # 纯雷达
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))


def bench_mmdet3d(n, cfg_file, ckpt):
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # 复用 infer_mmdet3d 的解析
    from infer_mmdet3d import MMDET
    sys.path.insert(0, str(MMDET))
    from build_bev_pool import register
    register()
    from mmengine.config import Config
    from mmengine.dataset import pseudo_collate
    from mmengine.registry import init_default_scope
    from mmdet3d.apis import init_model
    from mmdet3d.registry import DATASETS
    cfg = Config.fromfile(cfg_file)
    init_default_scope('mmdet3d')
    model = init_model(cfg, str(Path(ckpt).resolve()), device='cuda:0')
    model.eval()
    dataset = DATASETS.build(cfg.val_dataloader.dataset)
    data_t, fwd_t = [], []
    for i in range(n):
        t0 = time.time()
        data = pseudo_collate([dataset[i]])
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            model.test_step(data)
        torch.cuda.synchronize()
        t2 = time.time()
        data_t.append(t1 - t0)
        fwd_t.append(t2 - t1)
    return data_t, fwd_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pipeline', required=True, choices=['mmdet3d', 'mmdet3d_lidaronly'])
    ap.add_argument('--n', type=int, default=20)
    args = ap.parse_args()
    if args.pipeline == 'mmdet3d':
        d, f = bench_mmdet3d(args.n, 'configs/police_bevfusion_mmdet3d.py',
                             'checkpoints/bevfusion_mmdet3d_lidarcam.pth')
    else:
        d, f = bench_mmdet3d(args.n, 'configs/police_bevfusion_mmdet3d_lidaronly.py',
                             'checkpoints/bevfusion_mmdet3d_lidaronly.pth')
    d, f = np.asarray(d), np.asarray(f)
    print(f"\n[{args.pipeline}] {args.n} 帧  数据(读图+读点+体素化) P50 {np.median(d)*1000:6.0f} ms | "
          f"前向 P50 {np.median(f)*1000:6.0f} ms | 合计 {np.median(d+f)*1000:6.0f} ms/帧 "
          f"({1/np.median(d+f):.1f} FPS)")


if __name__ == '__main__':
    main()
