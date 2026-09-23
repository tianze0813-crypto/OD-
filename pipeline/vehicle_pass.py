#!/usr/bin/env python3
"""车链（Car + Truck）在混合链路侧的编排 —— 推理一次、后处理一次。

用户 2026-09-21 决定：Car 与 Truck 共用同一份 BEVFusion 原始检测，
动静态区域（静态 slot / 动态区域 / 重跟踪 / ID 继承）合成一次算；
Car 与 Truck 冲突（同一槽位 / 同一物理目标被两类重复表示）**一律按 Car 算**。
行人 / 非机动车仍走原来的 VoxelNeXt VRU 链，一行不改。

链路（本模块负责把两段接起来）：

    BEVFusion raw json（10 类）
      ↓ 类别合并（跟踪前，保持原时机）    geometry/truck_trailer_rules.merge_classes_pre
      ↓ 类别白名单 + 早期范围/分数过滤
      ↓ main_chain/pipeline/step_vehicle_chain.py（子进程，独立 sys.path）
          共享 step2（并集跟踪：槽位/动态证据按类别优先级仲裁，Car 优先）
          → Car 视图 step3/step4（轿车精修）
          → 共享 step4.5（动态区域 + 运动-only 重跟踪 + ID 继承 + 相位拼接）
          → Car 视图 step5（终检 + base_link）
      ↓ Truck 几何还原（Truck 的 box 由自己那条链定稿，main 那套只借 id）
      ↓ Truck 分支：step2_5（类别归一 + 二次硬过滤）→ step3_refinement（Truck 几何 + yaw v2）
                    → truck_postprocess → 轨迹级 Truck/Trailer 统一 → base_link
      ↓ 返回 car_frames / truck_frames（obj_id 偏移由调用方加：0+ / 1000+）

为什么 Truck 要还原几何：main_chain 的 step3/step4.5 会跑轿车式 box fit / 第二遍 box fit，
对卡车不适用（会把长车几何掰歪），所以 step4.5 之后把 Truck 的 ``box_lidar`` 按
「同帧 + 同类 + 最近中心」还原成 BEVFusion 原值——只保留共享那遍给的 track_id。
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classification.class_refinement import ClassRefinementConfig   # noqa: E402
from filtering.five_class_output import apply_five_class_output    # noqa: E402
from filtering.hard_filters import (HardFilterConfig,              # noqa: E402
                                    apply_category_score_filter)
from geometry import truck_trailer_rules as trailer_rules          # noqa: E402
from pipeline.hybrid_expD_noncar import (                          # noqa: E402
    _early_range_filter, _noncar_filter, _yaw_v2_config,
    drop_spinning_vehicle)
from pipeline.hybrid_expD_truck import _enable_trailer_vocabulary  # noqa: E402
from pipeline import step2_5_class_correction as step2_5           # noqa: E402
from pipeline import step3_refinement                              # noqa: E402
from tracking import tracker_conservative as tracking              # noqa: E402

_enable_trailer_vocabulary()      # Trailer 只在本链词表里（不改共享文件的行为）

MAIN_CHAIN_ROOT = ROOT / "main_chain"
STEP_VEHICLE_CHAIN = MAIN_CHAIN_ROOT / "pipeline" / "step_vehicle_chain.py"
LABEL_SUBDIR_CAR = "label_car"
LABEL_SUBDIR_TRUCK = "label_truck"
CAR_ID_OFFSET = 0
TRUCK_ID_OFFSET = 1000

DEFAULTS: Dict[str, Any] = dict(
    # ---- 车链共用（Car / Truck）----
    # 标注侧没有 Trailer：类别合并（跟踪前）之后，剩余纯挂车框在下一阶段直接并成 Truck
    keep_classes=("Car", "Truck"),
    class_score_thresholds={"Car": 0.2, "Truck": 0.2, "Trailer": 0.25},
    range_front=80.0,
    range_rear=20.0,
    range_side=40.0,
    # ---- 类别合并（跟踪前，保持原时机与阈值）----
    trailer_rules=True,
    trailer_dup_iom=0.70,
    trailer_dup_iou=0.50,
    trailer_merge_iou=0.05,
    trailer_policy="to-truck",       # 标注侧没有 Trailer -> 一律并成 Truck
    # ---- Car 几何：静态刚性框（OD-main-0909 同步；默认关）----
    static_rigid=False,
    # ---- Car 静态 yaw 落定位置（2026-09-23 定为 step45 = B1）----
    # step45      : static_yaw 只算/导出，几何跑 detector 原生局部系；轴+方向由 step4.5
    #               settle 在几何之后写入（几何与 yaw 修正解耦）
    # step45-axis : 几何按修正后的轴拟合，settle 只做 π 等价翻转（备选口径）
    # step2       : 旧行为（static_yaw 写轴 + 方向投票都在 step2）
    car_yaw_settle="step45",
    # ---- Truck 分支（复用原 Truck 单链参数）----
    truck_sparsity_max_points=10,
    truck_visibility_min_ratio=0.05,
    truck_short_track_max_frames=4,
    truck_yaw_impl="v2",
    truck_static_yaw_enabled=False,
    truck_static_rotation_enabled=False,
    truck_static_rotation_classes=("Truck", "Bus"),
    truck_yaw_vehicle_flags={"apply_static_direction_vote": False,
                             "apply_straight_motion_yaw": True,
                             "apply_motion_yaw": False},
    truck_merge_enabled=False,
    truck_postprocess=True,
)


def _count(frames: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(frame.get("detections", [])) for frame in frames)


def _class_of(det: Mapping[str, Any]) -> str:
    return tracking.canonical_class_name(det.get("class_name")) or ""


_TRUCK_RAW = ("truck", "trailer", "bus", "construction_vehicle",
              "engineering_vehicle")


def _is_truck_family(det: Mapping[str, Any]) -> bool:
    """卡车族：共享那遍与原始检测的类别归一可能不一致（Trailer/Truck），匹配时放宽。"""
    if _class_of(det) in ("Truck", "Trailer"):
        return True
    return str(det.get("class_name", "")).strip().casefold() in _TRUCK_RAW


def _fold_trailer_to_truck(frames: Sequence[Mapping[str, Any]]) -> int:
    """把剩下的纯挂车框并成 Truck（标注侧没有 Trailer 类别）。

    时机：类别合并（跟踪前）之后、类别白名单之前。挂车与货车的去重 / 并集合并仍由
    ``merge_classes_pre`` 完成，这里只处理"没被任何货车罩住 / 相交"的剩余挂车框 ——
    不并成 Truck 的话它们会被随后的类别白名单直接丢掉。
    """
    folded = 0
    for frame in frames:
        for det in frame.get("detections", []):
            if (str(det.get("class_name", "")).strip().casefold() == "trailer"
                    or _class_of(det) == "Trailer"):
                det["class_name"] = "Truck"
                folded += 1
    return folded


def _filter_classes(frames: Sequence[Mapping[str, Any]],
                    keep_classes: Sequence[str]) -> Dict[str, int]:
    """按类别白名单过滤（接受模型原始名），保留原始类名字符串。"""
    kept_total = 0
    for frame in frames:
        kept = [det for det in frame.get("detections", [])
                if _class_of(det) in keep_classes]
        kept_total += len(kept)
        frame["detections"] = kept
        frame["num_detections"] = len(kept)
    return {"keep_classes": list(keep_classes), "detections_after": kept_total}


def _select_class(frames: Sequence[Mapping[str, Any]],
                  wanted: "str | Sequence[str]") -> List[Dict[str, Any]]:
    """切出单类别（或若干类别）视图（帧结构 / track_id / 其它字段原样）。

    【注意】根目录 tracking 里 ``Trailer`` 归一后仍是 ``Trailer``，所以 Truck 视图要
    同时收 ``Trailer``（Truck 分支后面会按 trailer_policy 再统一成 Truck）。
    """
    wanted_set = {wanted} if isinstance(wanted, str) else set(wanted)
    view = []
    for frame in frames:
        kept = [det for det in frame.get("detections", [])
                if _class_of(det) in wanted_set]
        item = {key: value for key, value in frame.items() if key != "detections"}
        item["detections"] = kept
        item["num_detections"] = len(kept)
        view.append(item)
    return view


def restore_geometry(out_json: Path, raw_json: Path,
                     max_distance: float = 1.0) -> Dict[str, Any]:
    """把 Truck 的 ``box_lidar`` 还原成原始检测的值（只保留共享那遍给的 id）。

    与旧的 pipeline/truck_car_tracking.py 里的同名步骤一致，只是搬进车链编排里，
    并限定在 Truck/Trailer 上（Car 保留 main_chain 精修后的几何）。
    """
    frames = json.loads(Path(out_json).read_text(encoding="utf-8"))
    source = json.loads(Path(raw_json).read_text(encoding="utf-8"))
    raw_by_frame: Dict[str, list] = {
        str(frame["frame_id"]): list(frame.get("detections", []))
        for frame in source}
    restored = 0
    missing = 0
    for frame in frames:
        ts = str(frame["frame_id"])
        candidates = raw_by_frame.get(ts, [])
        used: set = set()
        for det in frame.get("detections", []):
            if _class_of(det) == "Car":
                continue
            box = det.get("box_lidar")
            if not isinstance(box, list) or len(box) < 7 or not candidates:
                continue
            best = None
            for index, raw_det in enumerate(candidates):
                if index in used:
                    continue
                same = (_class_of(raw_det) == _class_of(det)
                        or _is_truck_family(raw_det))
                if not same:
                    continue
                raw_box = raw_det.get("box_lidar")
                if not isinstance(raw_box, list) or len(raw_box) < 7:
                    continue
                distance = math.hypot(float(raw_box[0]) - float(box[0]),
                                      float(raw_box[1]) - float(box[1]))
                if best is None or distance < best[0]:
                    best = (distance, index, raw_box)
            if best is None or best[0] > float(max_distance):
                missing += 1
                continue
            used.add(best[1])
            det["box_lidar"] = list(best[2])
            det["_vehicle_pass_geometry_restored"] = True
            restored += 1
    Path(out_json).write_text(
        json.dumps(frames, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return {"restored": restored, "unmatched": missing}


def _summarize_shared_stage(step2_diag: Path, step45_diag: Path) -> Dict[str, Any]:
    """把共享 step2 / step4.5 的诊断摘成小块，方便直接看「动静态区域只算了一次」。"""
    summary: Dict[str, Any] = {}
    if Path(step2_diag).is_file():
        d2 = json.loads(Path(step2_diag).read_text(encoding="utf-8"))
        tracking_diag = d2.get("tracking", {})
        slots = tracking_diag.get("slot_details", [])
        summary.update({
            "step2_detections": d2.get("final_detections"),
            "tracks_total": tracking_diag.get("tracks_total"),
            "slots": len(slots),
            "slot_classes": dict(Counter(str(s.get("class_name")) for s in slots)),
            "slot_bound_tracks": sum(1 for s in slots if s.get("track_id") is not None),
            "hard_filter_removed": (d2.get("hard_filters", {})
                                    .get("detections_removed")),
            "short_tracks_dropped": (d2.get("short_track_filter", {})
                                     .get("tracks_dropped")),
            "min_lifecycle_by_class": (d2.get("short_track_filter", {})
                                       .get("min_lifecycle_by_class")),
            "mixed_tracks_unified": (d2.get("class_finalization", {})
                                     .get("mixed_tracks_unified")),
            "same_center_removed": (d2.get("same_center_deduplication", {})
                                    .get("boxes_removed")),
        })
    if Path(step45_diag).is_file():
        d45 = json.loads(Path(step45_diag).read_text(encoding="utf-8"))
        summary.update({
            "dynamic_region_area_m2": (d45.get("dynamic_region_mask", {})
                                       .get("dynamic_area_m2")),
            "dynamic_candidate_tracks": d45.get("candidate_tracks"),
            "retrackable_detections": (d45.get("selection", {})
                                       .get("retrackable_detections")),
            "static_freeze_passed": d45.get("static_freeze", {}).get("passed"),
            "retracking": {key: d45.get("retracking", {}).get(key)
                           for key in ("matches", "births", "static_locks",
                                       "occlusion_recoveries",
                                       "lateral_jump_triggered")},
            "queue_merges": (d45.get("queue_stitching", {}).get("merges")),
            "phase_merges": len(d45.get("phase_stitching", {}).get("applied", [])),
            "slot_static_yaw_vote": {
                key: d45.get("slot_static_yaw_vote", {}).get(key)
                for key in ("enabled", "candidate_tracks",
                            "candidate_detections", "accepted_tracks",
                            "voted_tracks", "ambiguous_tracks",
                            "flipped_detections")
            },
            "height_prior_bottom_fit": {
                key: d45.get("height_prior_bottom_fit", {}).get(key)
                for key in ("enabled", "reference_height_m",
                            "height_tolerance_m", "min_height_m",
                            "candidate_boxes", "fitted_boxes",
                            "skipped_no_points", "skipped_not_fittable",
                            "skipped_height_floor")
            },
        })
    return summary


def _run_truck_branch(tracked_json: Path, clip: Path, work_root: Path,
                      params: Mapping[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Truck 几何/类别定稿：step2_5 -> step3_refinement -> truck_postprocess -> 类别统一。"""
    work_root = Path(work_root)
    base = Path(clip).name
    coords = tracking.CoordinateProvider(Path(clip))

    hard_config = HardFilterConfig(
        score_threshold=float(params["class_score_thresholds"]["Truck"]),
        range_front=float(params["range_front"]),
        range_rear=float(params["range_rear"]),
        range_side=float(params["range_side"]),
        sparsity_max_points=int(params["truck_sparsity_max_points"]),
        visibility_min_ratio=float(params["truck_visibility_min_ratio"]),
        keep_classes=("Truck", "Trailer"),
        class_score_thresholds=(("Truck", float(params["class_score_thresholds"]["Truck"])),
                                ("Trailer", float(params["class_score_thresholds"]["Trailer"]))),
    )
    step2_5_json = work_root / f"{base}_truck_step2_5.json"
    step2_5_diag = work_root / f"{base}_truck_step2_5_diagnostics.json"
    step2_5.run(
        tracked_json, params["tracked_diagnostics"], Path(clip), step2_5_json,
        diagnostics_path=step2_5_diag,
        hard_filter_config=hard_config,
        class_config=ClassRefinementConfig(),
        min_lifecycle=int(params["truck_short_track_max_frames"]),
        static_rotation_enabled=bool(params["truck_static_rotation_enabled"]),
        static_rotation_classes=tuple(params["truck_static_rotation_classes"]),
    )

    from geometry.box_geometry import GeometryConfig
    from geometry.multiclass_refinement import (NonmotorizedSizeConfig,
                                                TruckOverlapConfig)
    step3_json = work_root / f"{base}_truck_step3.json"
    step3_diag = work_root / f"{base}_truck_step3_diagnostics.json"
    step3_refinement.run(
        step2_5_json, step2_5_diag, Path(clip), step3_json,
        diagnostics_path=step3_diag,
        geometry_config=GeometryConfig(),
        truck_config=TruckOverlapConfig(),
        nonmotorized_config=NonmotorizedSizeConfig(),
        car_refinement_enabled=False,
        yaw_impl=str(params["truck_yaw_impl"]),
        static_yaw_enabled=bool(params["truck_static_yaw_enabled"]),
        yaw_vehicle_config=_yaw_v2_config(params["truck_yaw_vehicle_flags"]),
        truck_merge_enabled=bool(params["truck_merge_enabled"]),
    )

    processed = json.loads(step3_json.read_text(encoding="utf-8"))
    diagnostics: Dict[str, Any] = {}
    if params["truck_postprocess"]:
        from geometry.truck_postprocess import (TruckPostConfig,
                                                apply_truck_postprocess)
        diagnostics["truck_postprocess"] = apply_truck_postprocess(
            processed, coords, Path(clip), TruckPostConfig())
    _spinning, spin_stats = drop_spinning_vehicle(processed)
    diagnostics["spinning_truck_bus"] = spin_stats

    if params["trailer_rules"]:      # 轨迹级：同一 id 里出现过 Truck -> 整条 Truck
        processed, unify = trailer_rules.unify_track_classes(processed)
        if str(params["trailer_policy"]) == "to-truck":
            processed, policy = trailer_rules.trailer_to_truck_all(processed)
            unify["trailer_policy"] = policy
        diagnostics["track_class_unify"] = unify

    output, final_diag = apply_five_class_output(processed, coords)
    diagnostics["final_output"] = final_diag
    diagnostics["truck_final_detections"] = _count(output)
    return output, diagnostics


def run(raw_json: Path, clip: Path, work_root: Path,
        python: Optional[Path] = None,
        diagnostics_path: Optional[Path] = None,
        **overrides: Any) -> Dict[str, Any]:
    """一次推理（外部已跑好 raw json）-> 一次车链后处理。

    返回 ``{"car_frames", "truck_frames", "diagnostics"}``：
    两个 frames 都已经是 **base_link**、带 ``track_id``，调用方按 0+ / 1000+ 加 obj_id
    后合成一份 ``label/``。
    """
    params = dict(DEFAULTS)
    params.update(overrides)
    python = Path(python or sys.executable)
    raw_json = Path(raw_json)
    clip = Path(clip)
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    base = clip.name

    source = json.loads(raw_json.read_text(encoding="utf-8"))
    if not isinstance(source, list):
        raise ValueError(f"raw json 必须是帧列表：{raw_json}")
    # 【改动】空输入硬失败：BEVFusion 那边一旦返回 0 帧（例如 infos 被并发覆盖，
    # 见 bevfusion/scripts/infer_mmdet3d.py 里的检查），车链在这里立刻停，
    # 绝不会再往下产出一份「只有 VRU 标签」的半成品。
    if not source:
        raise RuntimeError(
            f"BEVFusion 原始检测是空的（0 帧）：{raw_json} —— 检查该 clip 的 BEV 推理日志，"
            f"串行重跑这个 clip（infos 可能被另一个并发进程覆盖过）")
    if _count(source) == 0:
        raise RuntimeError(
            f"BEVFusion 原始检测里没有任何框（{len(source)} 帧 / 0 框）：{raw_json} —— "
            f"raw 阈值 0.1 下这通常意味着推理没真正跑到该 clip，串行重跑确认")
    diagnostics: Dict[str, Any] = {
        "pipeline": "vehicle_pass",
        "clip": str(clip.resolve()),
        "source_raw_json": str(raw_json.resolve()),
        "input_frames": len(source),
        "input_detections": _count(source),
        "input_classes": sorted({str(det.get("class_name"))
                                 for frame in source
                                 for det in frame.get("detections", [])}),
    }

    # ---- 1) 类别合并（跟踪前，保持原时机）----
    frames: List[Dict[str, Any]] = [dict(frame) for frame in source]
    if params["trailer_rules"]:
        frames, merge_report = trailer_rules.merge_classes_pre(
            frames,
            dup_iom=float(params["trailer_dup_iom"]),
            dup_iou=float(params["trailer_dup_iou"]),
            merge_iou=float(params["trailer_merge_iou"]),
            keep_other_classes=True)      # 车链：Car 保留，工程车归一成 Truck
        diagnostics["truck_trailer_class_merge"] = merge_report
    else:
        diagnostics["truck_trailer_class_merge"] = {"enabled": False}

    # ---- 1b) 剩余纯挂车框 -> Truck（标注侧没有 Trailer；时机仍在跟踪之前）----
    folded_trailer = _fold_trailer_to_truck(frames)
    diagnostics["trailer_fold_to_truck"] = {
        "boxes": folded_trailer,
        "policy": "类别合并之后、跟踪之前：剩余纯挂车框并成 Truck（标注侧无 Trailer）"}

    # ---- 2) 类别白名单（Bus 折进 Truck）+ 早期范围 / 分数过滤 ----
    diagnostics["class_filter"] = _noncar_filter(
        frames, keep_classes=tuple(params["keep_classes"]))
    range_stats = _early_range_filter(
        frames, range_front=float(params["range_front"]),
        range_rear=float(params["range_rear"]),
        range_side=float(params["range_side"]),
        pedestrian_max_distance=float(params["range_front"]),
        nonmotorized_max_distance=float(params["range_front"]))
    diagnostics["early_range_filter"] = range_stats
    score_config = HardFilterConfig(
        score_threshold=min(float(v) for v in
                            params["class_score_thresholds"].values()),
        class_score_thresholds=tuple(
            (str(k), float(v)) for k, v in params["class_score_thresholds"].items()),
        range_front=float(params["range_front"]),
        range_rear=float(params["range_rear"]),
        range_side=float(params["range_side"]),
        keep_classes=tuple(params["keep_classes"]))
    diagnostics["early_class_score_filter"] = apply_category_score_filter(
        frames, score_config)
    vehicle_raw = work_root / f"{base}_vehicle_raw.json"
    vehicle_raw.write_text(json.dumps(frames, ensure_ascii=False) + "\n",
                           encoding="utf-8")

    # ---- 3) main_chain 车链共享阶段（子进程：独立 sys.path，避免 module 名冲突）----
    if not STEP_VEHICLE_CHAIN.is_file():
        raise RuntimeError(f"main_chain 缺少车链驱动：{STEP_VEHICLE_CHAIN}")
    steps_root = work_root / "vehicle_chain"
    steps_root.mkdir(parents=True, exist_ok=True)
    command = [python, STEP_VEHICLE_CHAIN,
               "--raw-json", vehicle_raw, "--clip", clip,
               "--work-root", steps_root,
               "--car-yaw-settle", str(params["car_yaw_settle"])]
    if params.get("static_rigid"):
        command.append("--static-rigid")
    print("[vehicle-pass] $ " + " ".join(str(value) for value in command),
          flush=True)
    subprocess.run([str(value) for value in command], check=True)
    step45_json = steps_root / f"{base}_step45.json"
    car_json = steps_root / f"{base}_car.json"
    car_diag = steps_root / f"{base}_car_diagnostics.json"
    tracked_diagnostics = steps_root / f"{base}_step2_diagnostics.json"
    for required in (step45_json, car_json, tracked_diagnostics):
        if not required.is_file():
            raise RuntimeError(f"车链驱动没有产出 {required.name}")
    diagnostics["car_step5"] = {
        key: value for key, value in
        json.loads(car_diag.read_text(encoding="utf-8")).items()
        if key in ("before_detections", "after_detections",
                   "point_filter_removed", "short_track_removed",
                   "car_only_removed")}

    # ---- 3b) 共享阶段的关键指标（槽位 / 动态区域只算一次，落进诊断便于核对）----
    diagnostics["shared_region"] = _summarize_shared_stage(
        tracked_diagnostics, steps_root / f"{base}_step45_diagnostics.json")

    # ---- 4) Truck 几何还原（共享那遍只借 id）----
    restore = restore_geometry(step45_json, vehicle_raw)
    diagnostics["truck_geometry_restore"] = restore

    # ---- 5) Truck 分支（几何 / 类别定稿）----
    tracked = json.loads(step45_json.read_text(encoding="utf-8"))
    truck_view = _select_class(tracked, ("Truck", "Trailer"))
    truck_view_json = work_root / f"{base}_truck_tracked.json"
    truck_view_json.write_text(json.dumps(truck_view, ensure_ascii=False) + "\n",
                               encoding="utf-8")
    truck_params = dict(params)
    truck_params["tracked_diagnostics"] = tracked_diagnostics
    truck_frames, truck_diag = _run_truck_branch(
        truck_view_json, clip, work_root, truck_params)
    diagnostics["truck_branch"] = truck_diag
    diagnostics["truck_frames"] = len(truck_frames)
    diagnostics["truck_detections"] = _count(truck_frames)

    # ---- 6) Car 分支结果（main_chain step5 已转 base_link）----
    car_frames = json.loads(car_json.read_text(encoding="utf-8"))
    diagnostics["car_frames"] = len(car_frames)
    diagnostics["car_detections"] = _count(car_frames)
    diagnostics["output_classes"] = sorted({
        _class_of(det) for frame in list(car_frames) + list(truck_frames)
        for det in frame.get("detections", []) if det.get("track_id") is not None})
    diagnostics["final_detections"] = (_count(car_frames)
                                       + _count(truck_frames))
    diagnostics["params"] = {key: (list(value) if isinstance(value, tuple) else value)
                             for key, value in params.items()}

    # 【改动】过程数据默认不落盘（最终产出目录里不要 vehicle_pass_diagnostics.json）；
    # 调试时显式传 --diagnostics/ diagnostics_path 才写。
    if diagnostics_path is not None:
        target = Path(diagnostics_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    return {"car_frames": car_frames, "truck_frames": truck_frames,
            "diagnostics": diagnostics,
            "car_json": str(car_json),
            "step45_json": str(step45_json)}


def export_labels(frames: Sequence[Mapping[str, Any]], clip: Path,
                  subdir: str, id_offset: int) -> int:
    """把某一支的车链输出写成 SUST label（base_link 系，obj_id 加偏移）。"""
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
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--diagnostics", type=Path, default=None,
                        help="【改动】过程诊断写到哪（默认不写；批跑里由 "
                             "--keep-vehicle-diagnostics 控制）")
    parser.add_argument("--export", action="store_true",
                        help="把 Car/Truck 分别写到 <clip>/label_car、label_truck")
    parser.add_argument("--trailer-policy", choices=["keep", "to-truck"],
                        default=DEFAULTS["trailer_policy"])
    parser.add_argument("--car-yaw-settle",
                        choices=["step2", "step45", "step45-axis"],
                        default=DEFAULTS["car_yaw_settle"],
                        help="Car 静态 yaw 落定位置（默认 step45 = B1）")
    args = parser.parse_args()
    result = run(args.raw_json, args.clip, args.work_root, args.python,
                 args.diagnostics, trailer_policy=args.trailer_policy,
                 car_yaw_settle=args.car_yaw_settle)
    if args.export:
        car = export_labels(result["car_frames"], args.clip, LABEL_SUBDIR_CAR,
                            CAR_ID_OFFSET)
        truck = export_labels(result["truck_frames"], args.clip, LABEL_SUBDIR_TRUCK,
                              TRUCK_ID_OFFSET)
        print(f"exported car={car} truck={truck}")
    print(json.dumps({k: v for k, v in result["diagnostics"].items()
                      if k in ("car_detections", "truck_detections",
                               "final_detections", "output_classes")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
