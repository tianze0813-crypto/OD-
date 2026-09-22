#!/usr/bin/env python3
"""车链（Car + Truck）共享阶段驱动 —— 一次身份跟踪 + 一次动态区域重跟踪。

背景（用户 2026-09-21）：
Car 与 Truck 都改用同一份 BEVFusion 原始检测，**动静态区域（静态 slot / 动态区域 /
重跟踪 / ID 继承）本质是同一套计算**，之前两条链各算一遍，而且各自只看到自己那条链
的类别（轿车的车位进不了卡车那遍的 slot，反之亦然）。这里把它们合成一次：

    step2（共享）      类过滤/归一 + 候选槽位发现 + 身份跟踪 + 硬过滤 + 同中心去重
                       + 短轨迹 + 静态 yaw + 整合 yaw + 轨迹级类别统一
                       —— 输入是 Car/Truck 的并集，类别用【类别优先级】仲裁（Car 优先）
    step3（Car）       轿车框拟合（只在 Car 视图上跑，Truck 几何不动）
    step4（Car）       轿车尺寸闸门（车链里由 truck 头出 Truck，这里关掉 Car→Truck 改写）
    step4.5（共享）    动态区域（高速证据）+ 区域 mask + 运动-only 重跟踪 + ID 继承 +
                       队列/相位拼接 —— 并集上跑一次
    step5（Car）       终检（点数/短链）+ Car-only + box 转 base_link

Truck 的几何由 BEVFusion 原值交给自己那条链（step2_5 / step3_refinement / truck_postprocess）
定稿，所以本驱动只改 Car 的几何：step3/step4 在 Car 视图上跑，step4.5 之后由调用方
（pipeline/vehicle_pass.py）把 Truck 的 ``box_lidar`` 从原始检测还原回来。

输出（work_root 下）：
    <clip>_step2.json / _step2_diagnostics.json
    <clip>_step45.json / _step45_diagnostics.json     ← 共用 id 的并集（Truck 几何待还原）
    <clip>_car.json / _car_diagnostics.json           ← Car-only，base_link，可直接写 label
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import step2_identity_class_filter_yaw as step2      # noqa: E402
from pipeline import step3_car_box_fit as step3                    # noqa: E402
from pipeline import step4_car_size_filter as step4                # noqa: E402
from pipeline import step4_5_region_phase_retrack as step45        # noqa: E402
from pipeline import step5_class_motion_filter as step5            # noqa: E402

from classification.class_refinement import ClassRefinementConfig  # noqa: E402
from filtering.car_size_filter import LargeCarFilterConfig         # noqa: E402
from filtering.final_filter import FinalFilterConfig               # noqa: E402
from filtering.hard_filters import HardFilterConfig                # noqa: E402
from geometry.car_box_fit import CarBoxFitConfig                    # noqa: E402
from tracking import tracker_conservative as tracking              # noqa: E402

# 车链默认参数（Car / Truck 各自与单链行为对齐）
DEFAULTS: Dict[str, Any] = dict(
    class_score_threshold=0.2,                 # Car 0.2 / Truck 0.2（raw 门槛另在入口把关）
    class_sparsity=(("Car", 5), ("Truck", 10)),
    class_min_lifecycle=(("Car", 3), ("Truck", 4)),
    range_front=80.0,
    range_rear=20.0,
    range_side=40.0,
    visibility_min_ratio=0.05,
    # 车链里 Truck 由 BEVFusion 的 truck 头负责，step4 不再按尺寸把 Car 改写成 Truck
    car_size_relabel=False,
    car_size_truck_length_min=6.0,
    car_sparsity_max_points=5,
    car_short_track_max_frames=3,
)


def _count(frames: Sequence[Dict[str, Any]]) -> int:
    return sum(len(frame.get("detections", [])) for frame in frames)


def _class_of(det: Dict[str, Any]) -> str:
    return tracking.canonical_class_name(det.get("class_name")) or ""


def _select_class(frames: Sequence[Dict[str, Any]], wanted: str
                  ) -> list:
    """切出单类别视图（帧结构、track_id、其它字段都原样保留）。"""
    view = []
    for frame in frames:
        kept = [det for det in frame.get("detections", [])
                if _class_of(det) == wanted]
        item = {key: value for key, value in frame.items()
                if key != "detections"}
        item["detections"] = kept
        item["num_detections"] = len(kept)
        view.append(item)
    return view


def _merge_car_back(union_frames: Sequence[Dict[str, Any]],
                    car_frames: Sequence[Dict[str, Any]]) -> list:
    """把 Car 视图（step3/step4 的结果）里的 Car 框替换回并集，Truck 原样保留。"""
    by_key = {}
    for frame in car_frames:
        for det in frame.get("detections", []):
            by_key[(str(frame["frame_id"]), int(det.get("track_id", -1)))] = det
    merged = []
    replaced = 0
    for frame in union_frames:
        kept = []
        for det in frame.get("detections", []):
            if _class_of(det) != "Car":
                kept.append(det)
                continue
            key = (str(frame["frame_id"]), int(det.get("track_id", -1)))
            updated = by_key.get(key)
            if updated is None:       # Car 在 step3/step4 被删（理论上不会发生）
                continue
            kept.append(updated)
            replaced += 1
        item = {key: value for key, value in frame.items()
                if key != "detections"}
        item["detections"] = kept
        item["num_detections"] = len(kept)
        merged.append(item)
    return merged, replaced


def run(raw_json: Path, clip: Path, work_root: Path, *,
        keep_classes: Tuple[str, ...] = ("Car", "Truck"),
        class_config: Optional[ClassRefinementConfig] = None,
        **overrides: Any) -> Dict[str, Any]:
    params = dict(DEFAULTS)
    params.update(overrides)

    raw_json = Path(raw_json)
    clip = Path(clip)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    base = clip.name

    # 【改动】空输入硬失败（上游 BEVFusion 空输出的话，别产半成品）
    _probe = json.loads(raw_json.read_text(encoding="utf-8"))
    if not isinstance(_probe, list) or not _probe:
        raise SystemExit(f"[vehicle-chain] 输入 raw json 是空的（0 帧）：{raw_json}")
    if _count(_probe) == 0:
        raise SystemExit(
            f"[vehicle-chain] 输入 raw json 里没有任何框（{len(_probe)} 帧）：{raw_json}")

    # ---- step2：共享身份跟踪（Car/Truck 并集，类别优先级仲裁，Car 优先）----
    step2_json = work_root / f"{base}_step2.json"
    step2_diag = work_root / f"{base}_step2_diagnostics.json"
    hard_config = HardFilterConfig(
        score_threshold=float(params["class_score_threshold"]),
        range_front=float(params["range_front"]),
        range_rear=float(params["range_rear"]),
        range_side=float(params["range_side"]),
        sparsity_max_points=max(int(v) for _c, v in params["class_sparsity"]),
        class_sparsity_max_points=tuple(
            (str(c), int(v)) for c, v in params["class_sparsity"]),
        visibility_min_ratio=float(params["visibility_min_ratio"]),
        keep_classes=tuple(str(c) for c in keep_classes),
    )
    step2_diagnostics = step2.run(
        raw_json, clip, step2_json, None, step2_diag,
        hard_filter_config=hard_config,
        class_config=class_config or ClassRefinementConfig(),
        min_lifecycle=min(int(v) for _c, v in params["class_min_lifecycle"]),
        class_min_lifecycle={str(c): int(v)
                             for c, v in params["class_min_lifecycle"]})

    frames = json.loads(step2_json.read_text(encoding="utf-8"))
    union_count = _count(frames)

    # ---- step3/step4：只对 Car 视图做轿车精修（Truck 几何不动）----
    car_view_json = work_root / f"{base}_car_view.json"
    car_view_json.write_text(
        json.dumps(_select_class(frames, "Car"), ensure_ascii=False) + "\n",
        encoding="utf-8")
    step3_json = work_root / f"{base}_car_step3.json"
    step3_diag = work_root / f"{base}_car_step3_diagnostics.json"
    car_size_config = LargeCarFilterConfig(
        truck_length_min=(1e9 if not params["car_size_relabel"]
                          else float(params["car_size_truck_length_min"])))
    step3_result = step3.run(car_view_json, step2_diag, clip, step3_json, None,
                             step3_diag,
                             # 【改动】静态刚性框（OD-main-0909 同步）：默认关
                             config=CarBoxFitConfig(static_rigid_enabled=bool(
                                 params.get("static_rigid", False))))
    step4_json = work_root / f"{base}_car_step4.json"
    step4_diag = work_root / f"{base}_car_step4_diagnostics.json"
    step4_result = step4.run(step3_json, step4_json, step4_diag,
                             config=car_size_config)
    car_frames = json.loads(step4_json.read_text(encoding="utf-8"))
    union_step4, replaced = _merge_car_back(frames, car_frames)
    union_step4_json = work_root / f"{base}_union_step4.json"
    union_step4_json.write_text(
        json.dumps(union_step4, ensure_ascii=False) + "\n", encoding="utf-8")

    # ---- step4.5：共享动态区域 / 重跟踪 / ID 继承（并集上跑一次）----
    step45_json = work_root / f"{base}_step45.json"
    step45_diag = work_root / f"{base}_step45_diagnostics.json"
    step45_diagnostics = step45.run(union_step4_json, clip, step2_diag,
                                    step45_json, step45_diag)

    # ---- step5：Car 终检 + base_link ----
    tracked = json.loads(step45_json.read_text(encoding="utf-8"))
    car_tracked_json = work_root / f"{base}_car_tracked.json"
    car_tracked_json.write_text(
        json.dumps(_select_class(tracked, "Car"), ensure_ascii=False) + "\n",
        encoding="utf-8")
    car_json = work_root / f"{base}_car.json"
    car_diag = work_root / f"{base}_car_diagnostics.json"
    step5_result = step5.run(
        car_tracked_json, clip, car_json, None, car_diag,
        FinalFilterConfig(
            max_points_in_box=int(params["car_sparsity_max_points"]),
            max_track_length=int(params["car_short_track_max_frames"])))

    return {
        "clip": str(clip),
        "raw_json": str(raw_json),
        "step2_json": str(step2_json),
        "step2_diagnostics": str(step2_diag),
        "union_step4_json": str(union_step4_json),
        "step45_json": str(step45_json),
        "step45_diagnostics": str(step45_diag),
        "car_json": str(car_json),
        "car_diagnostics": str(car_diag),
        "union_detections": union_count,
        "car_boxes_merged": replaced,
        "car_step5": {
            "before": step5_result.get("before_detections"),
            "after": step5_result.get("after_detections"),
            "point_removed": step5_result.get("point_filter_removed"),
            "short_track_removed": step5_result.get("short_track_removed"),
            "car_only_removed": step5_result.get("car_only_removed"),
        },
        "params": {key: (list(value) if isinstance(value, tuple) else value)
                   for key, value in params.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--keep-classes", default="Car,Truck")
    parser.add_argument("--class-score-threshold", type=float,
                        default=DEFAULTS["class_score_threshold"])
    parser.add_argument("--sparsity-car", type=int, default=5)
    parser.add_argument("--sparsity-truck", type=int, default=10)
    parser.add_argument("--short-car", type=int, default=3)
    parser.add_argument("--short-truck", type=int, default=4)
    parser.add_argument("--range-front", type=float, default=DEFAULTS["range_front"])
    parser.add_argument("--range-rear", type=float, default=DEFAULTS["range_rear"])
    parser.add_argument("--range-side", type=float, default=DEFAULTS["range_side"])
    parser.add_argument("--static-rigid", action="store_true",
                        help="【改动】静态 Car 轨迹叠帧拟一个刚性 box，固定在世界系（默认关）")
    parser.add_argument("--car-size-relabel", action="store_true",
                        help="按尺寸把大 Car 改写成 Truck（车链默认关，Truck 由 truck 头负责）")
    args = parser.parse_args()
    result = run(
        args.raw_json, args.clip, args.work_root,
        keep_classes=tuple(c.strip() for c in args.keep_classes.split(",") if c.strip()),
        class_score_threshold=args.class_score_threshold,
        class_sparsity=(("Car", args.sparsity_car), ("Truck", args.sparsity_truck)),
        class_min_lifecycle=(("Car", args.short_car), ("Truck", args.short_truck)),
        range_front=args.range_front, range_rear=args.range_rear,
        range_side=args.range_side,
        car_size_relabel=bool(args.car_size_relabel),
        static_rigid=bool(args.static_rigid))
    print(json.dumps({k: v for k, v in result.items() if k != "params"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
