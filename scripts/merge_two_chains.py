#!/usr/bin/env python3
"""把 Truck 链与 VRU 链的输出合并成一份 SUST label。

* Truck 链写 label_truck/（obj_id 已 +1000）
* VRU  链写 label_vru/  （obj_id 已 +2000）
* 本脚本合成 label/，类别名保持不变。

用法：
  python scripts/merge_two_chains.py --bag <目录> [--write]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def merge_clip(clip: Path, truck_subdir: str, vru_subdir: str,
               out_subdir: str, write: bool) -> dict:
    truck_dir = clip / truck_subdir
    vru_dir = clip / vru_subdir
    frames = sorted({p.name for p in truck_dir.glob("*.json")} |
                    {p.name for p in vru_dir.glob("*.json")}) if (
        truck_dir.is_dir() or vru_dir.is_dir()) else []
    labels = Counter()
    out_dir = clip / out_subdir
    collisions = 0
    for name in frames:
        merged = []
        for src in (truck_dir, vru_dir):
            path = src / name
            if path.is_file():
                merged.extend(json.loads(path.read_text(encoding="utf-8")))
        for item in merged:
            labels[item["obj_type"]] += 1
        # 真正的冲突 = 【同一帧内】同一 obj_id 出现多次（跨帧重复是轨迹的正常行为）
        per_frame = Counter(str(item["obj_id"]) for item in merged)
        collisions += sum(1 for v in per_frame.values() if v > 1)
        if write:
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / name).write_text(
                json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8")
    return {"clip": clip.name, "frames": len(frames),
            "labels": dict(labels), "total": sum(labels.values()),
            "id_collisions": collisions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--truck-subdir", default="label_truck")
    parser.add_argument("--vru-subdir", default="label_vru")
    parser.add_argument("--out-subdir", default="label")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    clips = sorted(p for p in args.bag.glob("scene_*") if p.is_dir())
    bad = 0
    for clip in clips:
        info = merge_clip(clip, args.truck_subdir, args.vru_subdir,
                          args.out_subdir, args.write)
        bad += info["id_collisions"]
        print("%-34s 帧%3d  标签%5d %-52s id冲突 %d" % (
            info["clip"][-24:], info["frames"], info["total"],
            info["labels"], info["id_collisions"]))
    print("\n写盘: %s   合计 id 冲突: %d" % (
        "YES" if args.write else "NO（dry-run）", bad))


if __name__ == "__main__":
    main()
