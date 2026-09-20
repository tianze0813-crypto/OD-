"""Truck 链的「货车 / 挂车」类别合并规则（2026-09-20 用户口径）。

链路顺序（用户 2026-09-20 明确）：
    检测 → 范围/分数过滤（早期） → **类别合并（本模块）** → ID 跟踪
         → 短轨迹/硬过滤等其他过滤 → 精修 → 导出 SUST label

本模块只负责第 3 步「类别合并」，全部在 **跟踪之前**、在 lidar 系对检测框做：

  ① 类名归一：BEVFusion 的 truck/bus -> Truck，trailer -> Trailer（其它类不在本链范围内，丢掉）。
  ②【重复】挂车被货车罩住（交面积/挂车面积 IoM >= dup_iom，或 BEV IoU >= dup_iou）
     → 同一辆车被两个 head 各出一次 → **以 Truck 为准**，丢掉该挂车。
  ③【有交集】挂车与货车部分相交（IoU >= merge_iou）→ **并集合并成一个大长 Truck**
     （分数取两者较大者，保证并出来的长框能通过后续阈值），丢掉该挂车。
     并集口径复用 geometry.truck_postprocess._union_box（平行(<=30°)沿长框轴、z 取上下界并集）。
     合并后长框在整条轨迹里是稳定的（真铰接车每一帧都在），所以能正常被跟踪上；
     只是偶尔贴一下的挂车会形成短轨迹，交给跟踪之后的短轨迹过滤处理。

跟踪之后只剩一条类别一致性规则（需要 track_id）：
  ④ 同一个 obj_id 里出现过 Truck → 该 id 的所有框都算 Truck（`unify_track_classes`）。

本模块不做坐标换算：输入/输出都是 lidar 系；base_link 换算仍由 filtering.five_class_output
统一完成一次。
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from tracking import tracker_conservative as tracking

TRUCK_FROM = {"truck": "Truck", "bus": "Truck", "Truck": "Truck", "Bus": "Truck"}
TRAILER_FROM = {"trailer": "Trailer", "Trailer": "Trailer"}

DEFAULTS = dict(
    trailer_dup_iom=0.70,     # 挂车被货车罩住的比例 -> 判为重复
    trailer_dup_iou=0.50,     # 或者 BEV IoU 超过它就判为重复
    trailer_merge_iou=0.05,   # 有交集判据
    trailer_merge_max_iter=8,
)


def normalize_class(name: Any) -> Optional[str]:
    """BEVFusion 原始类名 -> 本链内部两类：Truck / Trailer。"""
    text = str(name).strip()
    if text in TRUCK_FROM:
        return "Truck"
    if text in TRAILER_FROM:
        return "Trailer"
    return None


def _iou_and_iom(truck_box: Sequence[float], trailer_box: Sequence[float]) -> Tuple[float, float]:
    ck, sk, yk = truck_box[:2], truck_box[3:6], truck_box[6]
    ct, st, yt = trailer_box[:2], trailer_box[3:6], trailer_box[6]
    iou = tracking.bev_iou(ck, sk, yk, ct, st, yt)
    pk = tracking.rectangle_corners(ck, sk, yk)
    pt = tracking.rectangle_corners(ct, st, yt)
    inter = tracking.polygon_area(tracking.convex_intersection(pk, pt))
    area_t = tracking.polygon_area(pt)
    iom = 0.0 if area_t <= 1e-9 else inter / area_t
    return iou, iom


def merge_classes_pre(frames: Sequence[Mapping[str, Any]],
                      dup_iom: float = DEFAULTS["trailer_dup_iom"],
                      dup_iou: float = DEFAULTS["trailer_dup_iou"],
                      merge_iou: float = DEFAULTS["trailer_merge_iou"],
                      max_iter: int = DEFAULTS["trailer_merge_max_iter"],
                      ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """跟踪之前的类别合并：类名归一 + 挂车去重 + 有交集并集成大长 Truck。"""
    from geometry.truck_postprocess import _union_box

    output: List[Dict[str, Any]] = []
    stats: Counter = Counter()
    merges: List[Dict[str, Any]] = []
    for frame in frames:
        dets: List[Dict[str, Any]] = []
        for det in frame.get("detections", []):
            cls = normalize_class(det.get("class_name", ""))
            if cls is None:
                stats["dropped_unrelated_class"] += 1
                continue
            item = dict(det)
            item["class_name"] = cls
            dets.append(item)

        trucks = [d for d in dets if d["class_name"] == "Truck"]
        trailers = [d for d in dets if d["class_name"] == "Trailer"]
        others = [d for d in dets if d["class_name"] not in ("Truck", "Trailer")]

        for _ in range(max_iter):                       # ③ 有交集 -> 并集
            changed = False
            for trailer in list(trailers):
                best, best_iou = None, 0.0
                for truck in trucks:
                    iou, iom = _iou_and_iom(truck["box_lidar"], trailer["box_lidar"])
                    if iom >= dup_iom or iou >= dup_iou:     # ② 重复 -> 以 Truck 为准
                        trailers.remove(trailer)
                        stats["dropped_duplicate_trailer"] += 1
                        changed = True
                        best = None
                        break
                    if iou >= merge_iou and iou > best_iou:
                        best, best_iou = truck, iou
                if best is None:
                    continue
                before = max(float(best["box_lidar"][3]), float(best["box_lidar"][4]))
                merged, _ = _union_box(best["box_lidar"], trailer["box_lidar"])
                best["box_lidar"] = [float(v) for v in merged]
                best["score"] = float(max(best.get("score", 0.0), trailer.get("score", 0.0)))
                best["merged_from"] = int(best.get("merged_from", 1)) + 1
                trailers.remove(trailer)
                stats["merged_trailer_into_truck"] += 1
                changed = True
                merges.append({
                    "frame_id": str(frame.get("frame_id")),
                    "iou": round(float(best_iou), 4),
                    "length_before": round(float(before), 3),
                    "length_after": round(float(max(merged[3], merged[4])), 3),
                })
            if not changed:
                break

        out_frame = dict(frame)
        out_frame["detections"] = others + trucks + trailers
        out_frame["num_detections"] = len(out_frame["detections"])
        output.append(out_frame)

    stats["frames_with_trailer_left"] = sum(
        1 for f in output if any(d["class_name"] == "Trailer" for d in f["detections"]))
    return output, {
        "stage": "class_merge (pre-tracking)",
        "policy": ("① truck/bus -> Truck, trailer -> Trailer; "
                   "② trailer covered by truck (IoM>=%.2f or IoU>=%.2f) -> drop trailer; "
                   "③ trailer intersecting truck (IoU>=%.2f) -> union into one long Truck "
                   "(score = max)" % (dup_iom, dup_iou, merge_iou)),
        "thresholds": {"dup_iom": float(dup_iom), "dup_iou": float(dup_iou),
                       "merge_iou": float(merge_iou)},
        "counts": dict(sorted(stats.items())),
        "merged_total": len(merges),
        "merged_examples": merges[:20],
    }


apply_frame_rules = merge_classes_pre      # 兼容旧名字（独立脚本用过）


def find_mixed_tracks(frames: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, int]]:
    """找出「同一个 track_id 里既有 Truck 又有 Trailer」的 id。"""
    per_track: Dict[str, Counter] = {}
    for frame in frames:
        for det in frame.get("detections", []):
            tid = det.get("track_id")
            if tid is None:
                continue
            per_track.setdefault(str(tid), Counter())[str(det.get("class_name"))] += 1
    return {tid: dict(cnt) for tid, cnt in per_track.items()
            if cnt.get("Truck", 0) > 0 and cnt.get("Trailer", 0) > 0}


def unify_track_classes(frames: Sequence[Mapping[str, Any]],
                        ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """④ 同一个 id 里出现过 Truck → 该 id 的框全部算 Truck（跟踪之后、写标签之前）。"""
    mixed = find_mixed_tracks(frames)
    to_truck = set(mixed)
    output = copy.deepcopy(list(frames))
    flipped = 0
    for frame in output:
        for det in frame.get("detections", []):
            if str(det.get("track_id")) in to_truck and det.get("class_name") != "Truck":
                det["class_name"] = "Truck"
                flipped += 1
    counts: Counter = Counter()
    for frame in output:
        for det in frame.get("detections", []):
            counts[str(det.get("class_name"))] += 1
    return output, {
        "stage": "track_class_unify (post-tracking)",
        "policy": "per-track: any Truck in the id -> whole id is Truck",
        "mixed_tracks": len(mixed),
        "mixed_track_examples": dict(list(mixed.items())[:20]),
        "boxes_flipped_to_truck": flipped,
        "class_counts_after": dict(sorted(counts.items())),
    }


def trailer_to_truck_all(frames: Sequence[Mapping[str, Any]],
                         ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """可选策略：纯挂车轨迹也一律标成 Truck（--trailer-policy to-truck）。"""
    output = copy.deepcopy(list(frames))
    flipped = 0
    for frame in output:
        for det in frame.get("detections", []):
            if det.get("class_name") == "Trailer":
                det["class_name"] = "Truck"
                flipped += 1
    return output, {"policy": "all Trailer -> Truck", "boxes_flipped": flipped}
