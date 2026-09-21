#!/usr/bin/env python3
"""测试用：把【BEVFusion 原始检测】喂进本项目为测试适配的后处理（三条链），合成一份标签。

与整批 orchestrator 的区别：**Car 走新的 pipeline/hybrid_expD_car.py**
（只做通用后处理：追踪 + 硬过滤 + base_link，不跑 main_chain/Waymo 的 Car 专属精修）；
Truck 走 pipeline/hybrid_expD_truck.py（BEVFusion 适配 + 货车/挂车类别合并）；
VRU 走 pipeline/hybrid_expD_vru.py。三条链**共用同一份 raw json**（只推理一次）。

输出：<out-root>/<clip名><suffix>/{label/, image→, lidar→, transforms→, readme.json→}
      （软链，SUST 可直接打开；obj_id 分段：Car 0+ / Truck 1000+ / VRU 2000+，与整批一致）

用法：
  python scripts/run_bevfusion_test_chains.py <raw-json 目录> <clip 目录> [...] \
      --out-root <输出根> [--suffix _bev_test] [--car-score 0.2] [--overwrite]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from pipeline.hybrid_expD_car import run as run_car            # noqa: E402
from pipeline.hybrid_expD_truck import run as run_truck        # noqa: E402
from pipeline.hybrid_expD_vru import run as run_vru            # noqa: E402
from run_hybrid_prelabel import _frames_to_labels, _merge_chain_labels  # noqa: E402


def process(clip: Path, raw: Path, work: Path, out_root: Path, suffix: str,
            car_score: float, overwrite: bool, car_truck_cover: float) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    chain_labels, chain_frames = {}, {}
    for name, fn in (("car", run_car), ("truck", run_truck), ("vru", run_vru)):
        out_json = work / f"{name}.json"
        diag = work / f"{name}_diag.json"
        if name == "car":
            fn(raw, clip, out_json, diag, class_score_thresholds={"Car": float(car_score)})
        else:
            fn(raw, clip, out_json, diag)
        chain_frames[name] = json.loads(out_json.read_text(encoding="utf-8"))
    # 与整批一致：Car 0+ / Truck 1000+ / VRU 2000+，再按 frame_id 合成 + Car/Truck 互斥
    for name, offset in (("car", 0), ("truck", 1000), ("vru", 2000)):
        chain_labels[name] = _frames_to_labels(chain_frames[name], offset)
    merged, merge_diag = _merge_chain_labels(
        chain_labels, ("car", "truck", "vru"),
        car_truck_cover_threshold=car_truck_cover)

    target = Path(out_root) / f"{clip.name}{suffix}"
    if target.exists():
        if not overwrite:
            raise RuntimeError(f"输出已存在: {target}（加 --overwrite）")
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for sub in ("image", "lidar", "transforms", "readme.json"):
        src = clip / sub
        if src.exists():
            (target / sub).symlink_to(src.resolve())
    label_dir = target / "label"
    label_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for frame in merged:
        labels = frame.get("labels", [])
        total += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"clip": clip.name, "labels": total, "merge": merge_diag, "out": str(target)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw_dir", type=Path, help="存放 <clip名>_raw.json 的目录（留档的 BEVFusion 原始检测）")
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--suffix", default="_bev_test")
    ap.add_argument("--work-root", type=Path, default=Path("/tmp/bevtest_chains"))
    ap.add_argument("--car-score", type=float, default=0.2)
    ap.add_argument("--car-truck-cover", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    for clip in args.clips:
        clip = clip.resolve()
        raw = args.raw_dir / f"{clip.name}_raw.json"
        if not raw.is_file():
            raise SystemExit(f"缺少原始检测: {raw}")
        result = process(clip, raw, args.work_root / clip.name, args.out_root,
                         args.suffix, args.car_score, args.overwrite,
                         args.car_truck_cover)
        print(f"[ok] {result['clip']}: {result['labels']} 个标签 -> {result['out']}")


if __name__ == "__main__":
    main()
