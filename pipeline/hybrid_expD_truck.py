"""Truck 链（大型车）：独立的推理后处理。

与 VRU 链完全分离的参数（【改动】按用户 2026-09-18 需求）：
  * 范围      前 80 / 后 20 / 左右 40
  * 分数阈值  Truck 0.4
  * 短轨迹    4（保留短轨迹过滤，阈值暂定 4）
  * 静止过滤  不做（卡车停车等灯是常态；链内静止过滤只作用于 Nonmotorized_vehicle，
              本链已把 NMV 过滤掉，因此天然不生效）
  * yaw       新版 v2（apply_motion_yaw=False：动态段/拐弯保留 detector yaw）
  * obj_id    从 1000 开始（与 VRU 链的 2000 起互不冲突）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.hybrid_expD_noncar import run as _noncar_run  # noqa: E402
from tracking import tracker_conservative as tracking        # noqa: E402

KEEP_CLASSES = ("Truck",)
ID_OFFSET = 1000
LABEL_SUBDIR = "label_truck"

# 本链的默认参数（可被 run(...) 的 overrides 覆盖）
DEFAULTS: Dict[str, Any] = dict(
    keep_classes=KEEP_CLASSES,
    class_score_thresholds={"Truck": 0.4},
    range_front=80.0,          # 【改动】前 80
    range_rear=20.0,
    range_side=40.0,
    sparsity_max_points=10,
    visibility_min_ratio=0.05,
    short_track_max_frames=4,  # 【改动】短轨迹过滤保留，阈值 4
    nonmotorized_min_net_displacement=0.0,   # 本链无 NMV，不生效
    yaw_impl="v2",             # 【改动】新版 yaw：动态段保留 detector yaw
    static_rotation_classes=("Truck", "Bus"),
    # 【改动】不把动态轨迹钉到停车位 id（保留跟踪器其余逻辑）
    disable_slot_binding=True,
    # 【改动】跳过静态 yaw 稳定（不把静止段 yaw 锁到停车方向）
    static_yaw_enabled=False,
    # 【改动】yaw v2 开关：关静态方向投票；直线行驶的轨迹用运动方向作 yaw
    yaw_vehicle_flags={"apply_static_direction_vote": False,
                       "apply_straight_motion_yaw": True,
                       "apply_motion_yaw": False},
    # 【改动】Truck 专用后处理：①yaw旋转帧 ②IoU并集合并(0.1) ③xy贴合 ④yaw翻转(>90°)
    truck_postprocess=True,
    truck_merge_enabled=False,   # 【改动】关掉 step3 自带的 Truck 合并，改用并集长框合并
)


def run(raw_json: Path, clip: Path, out_json: Path,
        diagnostics_path: Optional[Path] = None,
        **overrides: Any) -> Dict[str, Any]:
    params = dict(DEFAULTS)
    params.update(overrides)
    return _noncar_run(raw_json, clip, out_json, diagnostics_path, **params)


def export_labels(frames, clip: Path, subdir: str = LABEL_SUBDIR,
                  id_offset: int = ID_OFFSET) -> int:
    """把链路输出写成 SUST label（base_link 系），obj_id 加 id_offset。"""
    import shutil
    label_dir = Path(clip) / subdir
    if label_dir.is_dir():
        shutil.rmtree(label_dir)
    label_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for frame in frames:
        labels = []
        for det in frame.get("detections", []):
            if det.get("track_id") is None:
                continue
            item = tracking.box_to_label(det)
            try:
                item["obj_id"] = str(int(item["obj_id"]) + int(id_offset))
            except (TypeError, ValueError):
                item["obj_id"] = "t" + str(item["obj_id"])
            labels.append(item)
        total += len(labels)
        (label_dir / f"{frame['frame_id']}.json").write_text(
            json.dumps(labels, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", required=True, type=Path)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out-json", required=True, type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--label-subdir", default=LABEL_SUBDIR)
    parser.add_argument("--id-offset", type=int, default=ID_OFFSET)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--range-front", type=float, default=DEFAULTS["range_front"])
    parser.add_argument("--range-rear", type=float, default=DEFAULTS["range_rear"])
    parser.add_argument("--range-side", type=float, default=DEFAULTS["range_side"])
    parser.add_argument("--truck-score-threshold", type=float, default=0.4)
    parser.add_argument("--short-track-max-frames", type=int, default=4)
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--yaw-impl", default="v2")
    args = parser.parse_args()
    diag = run(args.raw_json, args.clip, args.out_json, args.diagnostics,
               class_score_thresholds={"Truck": float(args.truck_score_threshold)},
               range_front=args.range_front, range_rear=args.range_rear,
               range_side=args.range_side,
               short_track_max_frames=int(args.short_track_max_frames),
               sparsity_max_points=int(args.sparsity_max_points),
               yaw_impl=str(args.yaw_impl))
    if not args.no_export:
        frames = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
        n = export_labels(frames, Path(args.clip), args.label_subdir, args.id_offset)
        diag = dict(diag or {})
        diag["exported_labels"] = n
        print("exported %d labels -> %s" % (n, Path(args.clip) / args.label_subdir))
    print("truck_chain final_detections=%s yaw_impl=%s" % (
        (diag or {}).get("final_detections"), (diag or {}).get("yaw_impl")))


if __name__ == "__main__":
    main()
