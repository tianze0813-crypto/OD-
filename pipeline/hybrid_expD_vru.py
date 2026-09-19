"""VRU 链（行人 + 非机动车）：独立的推理后处理。

与 Truck 链完全分离的参数（【改动】按用户 2026-09-18 需求）：
  * 范围      方框 前 60 / 后 20 / 左右 40；行人另加 15m 半径
  * 分数阈值  Pedestrian 0.2 / Nonmotorized_vehicle 0.2
  * 短轨迹    4（链级，非机动车等照旧）
  * 行人门槛  20  （【改动】2026-09-19 用户要求：**只对行人**，生命周期 <20 帧的轨迹整条删；
                     非机动车不受影响。实现在 tracking.apply_post_filters 的 class_min_frames）
  * 静止过滤  非机动车净位移 < 15m 丢弃（行人不过滤）
  * yaw       沿用旧版 legacy（用户要求新版 yaw 只给 Truck）
  * obj_id    从 2000 开始（与 Truck 链的 1000 起互不冲突）
  * 【改动】2026-09-19：原"世界系一排行人"整排过滤规则已按用户要求删除，
             行人噪点改由 pedestrian_min_frames（生命周期 <20 帧整条删）处理。
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

KEEP_CLASSES = ("Pedestrian", "Nonmotorized_vehicle")
ID_OFFSET = 2000
LABEL_SUBDIR = "label_vru"

# 本链的默认参数（可被 run(...) 的 overrides 覆盖）
DEFAULTS: Dict[str, Any] = dict(
    keep_classes=KEEP_CLASSES,
    class_score_thresholds={"Pedestrian": 0.2, "Nonmotorized_vehicle": 0.2},
    range_front=60.0,          # 【改动】前 60
    range_rear=20.0,
    range_side=40.0,
    sparsity_max_points=10,
    visibility_min_ratio=0.05,
    short_track_max_frames=4,    # 链级短轨迹阈值（<= 语义），非机动车等照旧
    pedestrian_min_frames=20,    # 【改动】只对行人：帧数 <20 的轨迹整条删（2026-09-19 用户要求）
    pedestrian_max_distance=15.0,            # 【改动】行人 15m 半径
    nonmotorized_max_distance=60.0,
    nonmotorized_min_net_displacement=15.0,  # 【改动】非机动车静止过滤
    yaw_impl="legacy",         # 【改动】用户要求新版 yaw 只给 Truck
    static_rotation_classes=("Nonmotorized_vehicle",),
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
                item["obj_id"] = "v" + str(item["obj_id"])
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
    parser.add_argument("--pedestrian-score-threshold", type=float, default=0.2)
    parser.add_argument("--nonmotorized-score-threshold", type=float, default=0.2)
    parser.add_argument("--pedestrian-max-distance", type=float, default=15.0)
    parser.add_argument("--nonmotorized-max-distance", type=float, default=60.0)
    parser.add_argument("--nonmotorized-min-net-displacement", type=float, default=15.0)
    parser.add_argument("--short-track-max-frames", type=int, default=4)
    parser.add_argument("--pedestrian-min-frames", type=int, default=20,
                        help="【改动】仅行人：生命周期 < 该帧数的轨迹整条过滤（默认 20，0=关闭）")
    parser.add_argument("--sparsity-max-points", type=int, default=10)
    parser.add_argument("--yaw-impl", default="legacy")
    args = parser.parse_args()
    diag = run(args.raw_json, args.clip, args.out_json, args.diagnostics,
               class_score_thresholds={"Pedestrian": float(args.pedestrian_score_threshold),
                                       "Nonmotorized_vehicle": float(args.nonmotorized_score_threshold)},
               pedestrian_max_distance=float(args.pedestrian_max_distance),
               nonmotorized_max_distance=float(args.nonmotorized_max_distance),
               nonmotorized_min_net_displacement=float(args.nonmotorized_min_net_displacement),
               range_front=args.range_front, range_rear=args.range_rear,
               range_side=args.range_side,
               short_track_max_frames=int(args.short_track_max_frames),
               pedestrian_min_frames=int(args.pedestrian_min_frames),   # 【改动】
               sparsity_max_points=int(args.sparsity_max_points),
               yaw_impl=str(args.yaw_impl))
    if not args.no_export:
        frames = json.loads(Path(args.out_json).read_text(encoding="utf-8"))
        n = export_labels(frames, Path(args.clip), args.label_subdir, args.id_offset)
        diag = dict(diag or {})
        diag["exported_labels"] = n
        print("exported %d labels -> %s" % (n, Path(args.clip) / args.label_subdir))
    print("vru_chain final_detections=%s yaw_impl=%s" % (
        (diag or {}).get("final_detections"), (diag or {}).get("yaw_impl")))


if __name__ == "__main__":
    main()
