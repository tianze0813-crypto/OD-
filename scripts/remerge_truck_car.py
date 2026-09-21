#!/usr/bin/env python3
"""【改动】复用已算好的 Car / VRU 标签，只重跑 Truck 链，然后按新规则重新合并。

适用场景：只改了 Truck 链参数或合并规则，不想重跑耗时的 Car 链。

【2026-09-21 注意】Car 与 Truck 已经合并成一条后处理（pipeline/vehicle_pass.py）：
Truck 的 id 来自与 Car 共享的那一遍身份跟踪 / 动态区域，所以**单独重跑 Truck 链已经
拿不到同一套 id**。默认路径下要重跑就整条车链重跑（加 --no-car-truck-merged 才回到
老的两条独立链，此时本脚本仍然适用）。

规则（2026-09-18 用户指定）：
  1. Truck 链不再做 IoU 并集合并（geometry/truck_postprocess.py: merge_enabled=False）
  2. Car 被 Truck 覆盖的面积 / Car 面积 >= 阈值 → 删掉【该 Car id 的全部帧】
     （2026-09-21 起该规则在车链内部被 Car 优先取代，本脚本传 0 关闭）

用法：
  python scripts/remerge_truck_car.py --output-root <含 <clip>_pre 的目录> [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_hybrid_prelabel as rh                        # noqa: E402
from pipeline.hybrid_expD_truck import run as run_truck  # noqa: E402


def _read_labels(label_dir: Path, keep_types) -> dict:
    out = {}
    if not label_dir.is_dir():
        return out
    for path in sorted(label_dir.glob("*.json")):
        items = json.loads(path.read_text(encoding="utf-8"))
        if keep_types is not None:
            items = [x for x in items if x.get("obj_type") in keep_types]
        out[path.stem] = items
    return out


def _write_labels(frames, label_dir: Path) -> int:
    import shutil
    if label_dir.is_dir():
        shutil.rmtree(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for frame in frames:
        labels = frame.get("labels", [])
        total += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path,
                        default=Path.home() / "miniconda3/envs/openpcdet/bin/python")
    parser.add_argument("--truck-cfg", type=Path, default=rh.TRUCK_CFG)
    parser.add_argument("--truck-ckpt", type=Path, default=rh.TRUCK_CKPT)
    parser.add_argument("--truck-raw-threshold", type=float, default=0.1)
    # 【改动】Truck 检测器与模式（默认 BEVFusion 纯雷达，与整批链路一致）
    parser.add_argument("--truck-detector", choices=["bevfusion", "voxelnext"],
                        default="bevfusion")
    parser.add_argument("--truck-detector-mode", choices=["lidar", "fusion"],
                        default="lidar")
    parser.add_argument("--trailer-score-threshold", type=float, default=0.25)
    parser.add_argument("--no-trailer-rules", action="store_true")
    parser.add_argument("--car-truck-cover-threshold", type=float, default=0.5)
    parser.add_argument("--write", action="store_true",
                        help="真正写盘（默认只 dry-run 打印统计）")
    args = parser.parse_args()

    clips = sorted(p for p in args.output_root.expanduser().resolve().glob("*_pre")
                   if p.is_dir() and (p / "lidar" / "lidar_top").is_dir())
    if not clips:
        raise RuntimeError(f"没有找到 *_pre clip: {args.output_root}")
    rows = []
    for clip in clips:
        rh._print(f"{clip.name}: 复用 Car/VRU 标签 + 重跑 Truck 链")
        car = _read_labels(clip / "label", {"Car"})
        vru = _read_labels(clip / "label_vru", None)
        if not car:
            raise RuntimeError(f"{clip.name}: 没有可复用的 Car 标签（label/ 里没有 Car）")
        with tempfile.TemporaryDirectory(prefix=f"remerge_{clip.name}_") as temp:
            work = Path(temp)
            if args.truck_detector == "bevfusion":   # 【改动】与整批链路同一套检测器
                raw = rh._run_raw_bevfusion(args.python, clip, work / "truck_raw",
                                            args.truck_raw_threshold,
                                            args.truck_detector_mode)
            else:
                raw = rh._run_raw(args.python, clip, args.truck_cfg.resolve(),
                                  args.truck_ckpt.resolve(), work / "truck_raw",
                                  "truck", args.truck_raw_threshold)
            out = work / "truck.json"
            diag_path = work / "truck_diagnostics.json"
            result = run_truck(raw, clip, out, diag_path,
                               class_score_thresholds={"Truck": 0.2,
                                                       "Trailer": args.trailer_score_threshold},
                               trailer_rules=not args.no_trailer_rules)
            truck_frames = json.loads(out.read_text(encoding="utf-8"))
            truck = rh._frames_to_labels(truck_frames, rh.TRUCK_ID_OFFSET)
            truck_stats = {
                "detector": str(args.truck_detector),
                "detector_mode": str(args.truck_detector_mode),
                "checkpoint": str(args.truck_ckpt),
                "final_detections": (result or {}).get("final_detections"),
                "merge": (json.loads(diag_path.read_text(encoding="utf-8"))
                          .get("truck_postprocess", {}).get("iou_merge")
                          if diag_path.is_file() else None),
                "frames": len(truck)}
            merged, merge_diag = rh._merge_chain_labels(
                {"car": car, "truck": truck, "vru": vru},
                ["car", "truck", "vru"],
                car_truck_cover_threshold=args.car_truck_cover_threshold)
        if args.write:
            total = _write_labels(merged, clip / "label")
            _write_labels([{"frame_id": k, "labels": v} for k, v in truck.items()],
                          clip / "label_truck")
            row = {"clip": clip.name, "labels": total}
        else:
            row = {"clip": clip.name,
                   "labels": sum(len(f["labels"]) for f in merged)}
        row.update({"truck": truck_stats,
                    "merge": {k: v for k, v in merge_diag.items()
                              if k != "frames_missing_per_chain"}})
        rows.append(row)
        rh._print(f"{clip.name}: {json.dumps(row, ensure_ascii=False)}")
    print(json.dumps({"clips": rows, "write": args.write},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
