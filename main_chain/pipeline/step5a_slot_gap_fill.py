#!/usr/bin/env python3
"""Step 5A：静态 slot Car 的"内部空洞补帧"（世界系叠框，不做点叠拟合）。

背景（用户 2026-09-23）：
停在 slot 上的车偶尔会被检测器漏掉一两帧，轨迹中间出现空洞 → 标注残缺。
本步在 step4.5（几何 / ID / 动静态区域都已定稿）之后、step5（终检 + base_link）
之前，只把「非动态区域、绑定了静态 slot 的 Car」的内部空洞补上。

口径（用户逐条拍板）：

* 只补 Car：候选 = step2 ``tracking.slot_details`` 中 ``class_name == "Car"`` 的
  track，且该轨迹全部检测 ``region == "static"``、未被 step4.5 重跟踪；
* 只补轨迹首末观测**之间**的内部洞（不做端点外延）；``全补`` —— 单洞长度与每轨迹
  补帧总量默认都不限（config 里预留 ``max_hole_frames`` / ``max_fills_per_track``）；
* 出一个 box 的方式 = **纯框叠**（不做点云叠帧拟合：实测点叠会把极端稀疏的远车压到
  尺寸下限，框叠和相邻帧更一致）::

      世界中心   = 观测帧世界中心的中位数（z 一起带，避免俯仰/侧倾把 z 差漏进 xy）
      尺寸/高度 = 逐帧拟合框 dx/dy/dz 的中位数（各自取中位）
      yaw       = step2 static_yaw_stabilization.slots[].target_world_yaw
                  （direction_flip 则 +pi），缺失时回退该轨迹观测 world yaw 的环中位

* 返还门槛：该帧必须有 ``lidar/lidar_top/<frame_id>.bin``，且把世界框投回该帧 lidar 系
  后**框内点数 >= 6**（与 step5 的 ``count <= 5`` 删除口径对齐，补了不会被删）；
* 守卫①：洞两侧邻居的世界中心一致（< ``neighbor_center_tol_m``）且轴一致；
* 守卫②：该帧已有别的 Car/Truck 框与补框 BEV IoU > ``overlap_iou_threshold``
  → 跳过（遮挡帧不补，避免补成两辆车）；
* 只**新增**检测，绝不修改 step4.5 已定稿的检测 / 几何；新增检测带
  ``_step5a_filled`` 标记、``region = "static"``、``score = 0.0``，
  ``visibility`` 取相邻帧该目标的深拷贝（补帧没有真实相机可见度）。

输出：``<clip>_step5a.json``（并集帧表，Truck 原样）+
``<clip>_step5a_diagnostics.json``。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from filtering.hard_filters import count_points_in_boxes       # noqa: E402
from geometry.yaw_static_direction import _world_yaw_to_local  # noqa: E402
from tracking import tracker_conservative as tracking          # noqa: E402

# 视作"车"的类别（守卫②：同帧已有车压住补框位置就不补）
VEHICLE_CLASSES = ("Car", "Truck")


@dataclass(frozen=True)
class Step5aConfig:
    """Step 5A 阈值。默认值 = 用户 2026-09-23 拍板口径。"""

    enabled: bool = True
    # 返还门槛：框内点数 >= 该值才补（step5 是 `<= 5` 删，所以这里取 6）。
    min_points_in_box: int = 6
    # 叠框至少要有这么多观测帧（step2 已按 min_lifecycle 删过短轨迹，这里是兜底）。
    min_observations: int = 3
    # 单洞长度上限 / 每条轨迹补帧总量上限。None = 不限（"全补"）。
    max_hole_frames: Optional[int] = None
    max_fills_per_track: Optional[int] = None
    # 守卫①：洞两侧邻居的世界中心差 / 轴差（相对补框与彼此）。
    neighbor_center_tol_m: float = 1.0
    neighbor_yaw_tol_deg: float = 30.0
    # 守卫②：与同帧已有车框的 BEV IoU 上限（同 step4.5 overlap_filter 口径）。
    overlap_iou_threshold: float = 0.02

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enabled),
            "min_points_in_box": int(self.min_points_in_box),
            "min_observations": int(self.min_observations),
            "max_hole_frames": (None if self.max_hole_frames is None
                                else int(self.max_hole_frames)),
            "max_fills_per_track": (None if self.max_fills_per_track is None
                                    else int(self.max_fills_per_track)),
            "neighbor_center_tol_m": float(self.neighbor_center_tol_m),
            "neighbor_yaw_tol_deg": float(self.neighbor_yaw_tol_deg),
            "overlap_iou_threshold": float(self.overlap_iou_threshold),
        }


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _load_lidar_xyz(clip: Path, frame_id: str) -> Optional[np.ndarray]:
    """与 step5 / step4.5 同口径：lidar_top 原始 bin 的 xyz（缺帧返回 None）。"""
    path = Path(clip) / "lidar" / "lidar_top" / f"{frame_id}.bin"
    if not path.is_file():
        return None
    values = np.fromfile(path, dtype=np.float32)
    if values.size == 0 or values.size % 4 != 0:
        return None
    return values.reshape(-1, 4)[:, :3]


def _class_of(det: Mapping[str, Any]) -> str:
    return tracking.canonical_class_name(det.get("class_name")) or ""


def _circular_median_pi(values: Sequence[float]) -> float:
    """箱体朝向的稳健代表（周期 pi），与 step2 static_yaw 同口径。"""
    if not values:
        return 0.0
    doubled = 2.0 * np.asarray(list(values), dtype=np.float64)
    center = 0.5 * math.atan2(float(np.median(np.sin(doubled))),
                              float(np.median(np.cos(doubled))))
    candidates = [center, center + math.pi / 2.0]
    return min(candidates, key=lambda x: sum(
        tracking.angle_distance(v, x, modulo_pi=True) for v in values))


def _collect_observations(
        frames: Sequence[Mapping[str, Any]],
) -> Dict[int, List[Tuple[int, Mapping[str, Any]]]]:
    """track_id -> [(frame_index, detection)]，按帧序。"""
    observations: Dict[int, List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        for det in frame.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None:
                continue
            observations[int(track_id)].append((frame_index, det))
    return {key: sorted(value, key=lambda item: item[0])
            for key, value in observations.items()}


def _slot_axis(
        step2_diagnostics: Mapping[str, Any], track_id: int,
) -> Tuple[Optional[float], bool]:
    """step2 static_yaw 给的停车轴（世界系）+ 是否需要 +pi。"""
    for entry in step2_diagnostics.get(
            "static_yaw_stabilization", {}).get("slots", []):
        if entry.get("track_id") is None:
            continue
        if int(entry["track_id"]) != int(track_id):
            continue
        value = entry.get("target_world_yaw")
        if value is None:
            return None, False
        return float(value), bool(entry.get("direction_flip"))
    return None, False


def _stack_world_box(
        items: Sequence[Tuple[int, Mapping[str, Any]]],
        timestamps: Sequence[int],
        coords: tracking.CoordinateProvider,
        step2_diagnostics: Mapping[str, Any],
        track_id: int,
) -> Optional[Dict[str, Any]]:
    """纯框叠：世界系中心中位 + 逐帧拟合尺寸中位 + 停车轴。"""
    centers: List[np.ndarray] = []
    sizes: List[np.ndarray] = []
    world_yaws: List[float] = []
    skipped_no_pose = 0
    for frame_index, det in items:
        if not tracking.finite_box(dict(det)):
            continue
        world_from_lidar = coords.world_from_lidar(timestamps[frame_index])
        if world_from_lidar is None:
            skipped_no_pose += 1
            continue
        box = det["box_lidar"]
        centers.append(tracking.center_world(box, world_from_lidar))
        sizes.append(np.asarray(box[3:6], dtype=np.float64))
        world_yaws.append(tracking.yaw_world(float(box[6]), world_from_lidar))
    if len(centers) < 2:
        return None
    axis, flip = _slot_axis(step2_diagnostics, track_id)
    if axis is None:
        world_yaw = _circular_median_pi(world_yaws)
        yaw_source = "track_circular_median"
    else:
        world_yaw = tracking.wrap_angle(axis + math.pi if flip else axis)
        yaw_source = "slot_target_world_yaw"
    return {
        "world_center": np.median(np.asarray(centers, dtype=np.float64), axis=0),
        "size": np.median(np.asarray(sizes, dtype=np.float64), axis=0),
        "world_yaw": float(world_yaw),
        "yaw_flipped": bool(flip),
        "yaw_source": yaw_source,
        "stack_frames": len(centers),
        "skipped_no_pose": skipped_no_pose,
    }


def _hole_runs(missing: Sequence[int]) -> List[Tuple[int, int]]:
    """把缺失帧序号压成连续段 [(start, end)]（闭区间）。"""
    runs: List[Tuple[int, int]] = []
    for index in sorted(missing):
        if runs and index == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], index)
        else:
            runs.append((index, index))
    return runs


def _neighbor_of(observed: Sequence[int], frame_index: int,
                 side: str) -> Optional[int]:
    """洞两侧最近的"有观测且在邻域表里"的帧序号。"""
    if side == "left":
        candidates = [value for value in observed if value < frame_index]
        return max(candidates) if candidates else None
    candidates = [value for value in observed if value > frame_index]
    return min(candidates) if candidates else None


def _neighbour_visibility(frame: Mapping[str, Any],
                          track_id: int) -> Optional[Any]:
    for det in frame.get("detections", []):
        if det.get("track_id") is not None and int(det["track_id"]) == int(track_id):
            return det.get("visibility")
    return None


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run(step45_json: Path, clip: Path, step2_diagnostics: Path,
        out_json: Path, diagnostics_path: Path,
        config: Step5aConfig = Step5aConfig()) -> Dict[str, Any]:
    frames = json.loads(Path(step45_json).read_text(encoding="utf-8"))
    if not isinstance(frames, list):
        raise ValueError(f"step5a input must be a list of frames: {step45_json}")
    step2 = json.loads(Path(step2_diagnostics).read_text(encoding="utf-8"))
    clip = Path(clip)
    coords = tracking.CoordinateProvider(clip)

    before_frames = copy.deepcopy(frames)
    before_detections = sum(len(frame.get("detections", [])) for frame in frames)

    diagnostics: Dict[str, Any] = {
        "pipeline": "step5a_slot_gap_fill",
        "source_step45_json": str(Path(step45_json).resolve()),
        "source_clip": str(clip.resolve()),
        "source_step2_diagnostics": str(Path(step2_diagnostics).resolve()),
        "config": config.to_dict(),
        "policy": {
            "scope": "static-slot Car only, interior gaps of the observation span",
            "box_source": ("world-frame box stacking: median centre + median "
                           "per-frame size + slot axis"),
            "return_gate": "lidar bin exists and points_in_box >= min_points_in_box",
            "guards": [
                "bracketing neighbours agree in world centre and axis",
                "no same-frame Car/Truck box with BEV IoU above the threshold",
            ],
            "mutated_fields": "appends only (_step5a_filled detections)",
        },
        "candidate_slots": 0,
        "candidate_tracks": [],
        "skipped_tracks": [],
        "tracks": [],
        "fills": [],
        "skipped_frames": [],
        "inserted_detections": 0,
        "before_detections": before_detections,
    }
    if not config.enabled:
        diagnostics.update({
            "enabled": False,
            "after_detections": before_detections,
            "skip_reason_counts": {},
            "disabled_reason": "Step5aConfig.enabled is False",
        })
        _write(out_json, frames)
        _write(diagnostics_path, diagnostics)
        return diagnostics

    slot_details = step2.get("tracking", {}).get("slot_details", [])
    slot_car_ids = {
        int(item["track_id"])
        for item in slot_details
        if str(item.get("class_name", "")) == "Car"
        and item.get("track_id") is not None
    }
    diagnostics["candidate_slots"] = len(slot_car_ids)
    diagnostics["candidate_tracks"] = sorted(slot_car_ids)

    observations = _collect_observations(frames)
    timestamps = [int(frame["frame_id"]) for frame in frames]
    points_cache: Dict[str, Optional[np.ndarray]] = {}

    for track_id in sorted(slot_car_ids):
        items = observations.get(track_id)
        if not items:
            diagnostics["skipped_tracks"].append(
                {"track_id": track_id, "reason": "slot_track_absent"})
            continue
        classes = {_class_of(det) for _index, det in items}
        if classes != {"Car"}:
            diagnostics["skipped_tracks"].append(
                {"track_id": track_id, "reason": "not_pure_car",
                 "classes": sorted(classes)})
            continue
        outside = [
            index for index, det in items
            if det.get("region") != "static" or det.get("_step45_retracked")]
        if outside:
            diagnostics["skipped_tracks"].append(
                {"track_id": track_id, "reason": "outside_static_region",
                 "frame_count": len(outside), "frames": outside[:20]})
            continue
        if len(items) < int(config.min_observations):
            diagnostics["skipped_tracks"].append(
                {"track_id": track_id, "reason": "too_few_observations",
                 "observations": len(items)})
            continue

        stacked = _stack_world_box(items, timestamps, coords, step2, track_id)
        if stacked is None:
            diagnostics["skipped_tracks"].append(
                {"track_id": track_id, "reason": "stack_failed",
                 "observations": len(items)})
            continue

        observed = [frame_index for frame_index, _det in items]
        observed_set = set(observed)
        missing = [index for index in range(observed[0], observed[-1] + 1)
                   if index not in observed_set]
        track_entry: Dict[str, Any] = {
            "track_id": track_id,
            "observations": len(observed),
            "span": [observed[0], observed[-1]],
            "hole_frames": missing,
            "hole_runs": [list(item) for item in _hole_runs(missing)],
            "world_center": [round(float(value), 4)
                             for value in stacked["world_center"]],
            "size": [round(float(value), 4) for value in stacked["size"]],
            "world_yaw": round(float(stacked["world_yaw"]), 6),
            "yaw_source": stacked["yaw_source"],
            "yaw_flipped": stacked["yaw_flipped"],
            "stack_frames": stacked["stack_frames"],
            "stack_skipped_no_pose": stacked["skipped_no_pose"],
            "fills": [],
            "skipped_frames": [],
        }

        candidate_frames = list(missing)
        if config.max_hole_frames is not None:
            allowed: set = set()
            for start, end in _hole_runs(missing):
                if end - start + 1 <= int(config.max_hole_frames):
                    allowed.update(range(start, end + 1))
            candidate_frames = [index for index in candidate_frames
                                if index in allowed]

        neighbor_positions: Dict[int, np.ndarray] = {}
        neighbor_yaws: Dict[int, float] = {}
        for frame_index, det in items:
            world_from_lidar = coords.world_from_lidar(timestamps[frame_index])
            if world_from_lidar is None or not tracking.finite_box(dict(det)):
                continue
            neighbor_positions[frame_index] = tracking.center_world(
                det["box_lidar"], world_from_lidar)
            neighbor_yaws[frame_index] = tracking.yaw_world(
                float(det["box_lidar"][6]), world_from_lidar)

        fills_this_track = 0
        for frame_index in candidate_frames:
            if (config.max_fills_per_track is not None
                    and fills_this_track >= int(config.max_fills_per_track)):
                record = {"track_id": track_id, "frame_index": frame_index,
                          "frame_id": str(frames[frame_index]["frame_id"]),
                          "filled": False,
                          "reason": "max_fills_per_track_reached"}
                diagnostics["skipped_frames"].append(record)
                track_entry["skipped_frames"].append(record)
                continue
            record = _try_fill(
                frames, frame_index, track_id, stacked, coords, clip,
                config, neighbor_positions, neighbor_yaws, points_cache)
            if record.get("filled"):
                frames[frame_index].setdefault("detections", []).append(
                    record.pop("detection"))
                frames[frame_index]["num_detections"] = len(
                    frames[frame_index]["detections"])
                fills_this_track += 1
                diagnostics["fills"].append(record)
                track_entry["fills"].append(record["frame_index"])
                diagnostics["inserted_detections"] += 1
            else:
                diagnostics["skipped_frames"].append(record)
                track_entry["skipped_frames"].append(record)
        diagnostics["tracks"].append(track_entry)

    verification = _verify_appended_only(before_frames, frames)
    diagnostics["append_only_check"] = verification
    diagnostics["after_detections"] = sum(
        len(frame.get("detections", [])) for frame in frames)
    diagnostics["skip_reason_counts"] = _count_reasons(
        diagnostics["skipped_frames"])
    if not verification["passed"]:
        raise AssertionError(
            f"step5a append-only check failed: {verification['mismatches'][:3]}")

    _write(out_json, frames)
    _write(diagnostics_path, diagnostics)
    return diagnostics


def _count_reasons(records: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record.get("reason", "unknown"))] += 1
    return dict(sorted(counts.items()))


def _try_fill(
        frames: Sequence[Dict[str, Any]],
        frame_index: int,
        track_id: int,
        stacked: Mapping[str, Any],
        coords: tracking.CoordinateProvider,
        clip: Path,
        config: Step5aConfig,
        neighbor_positions: Mapping[int, np.ndarray],
        neighbor_yaws: Mapping[int, float],
        points_cache: Dict[str, Optional[np.ndarray]],
) -> Dict[str, Any]:
    """对单个洞帧做全部守卫 + 点数门槛；filled=True 时带上要追加的 detection。"""
    frame = frames[frame_index]
    record: Dict[str, Any] = {
        "track_id": track_id,
        "frame_index": frame_index,
        "frame_id": str(frame["frame_id"]),
        "filled": False,
    }
    timestamp = int(frame["frame_id"])
    world_from_lidar = coords.world_from_lidar(timestamp)
    lidar_from_world = coords.lidar_from_world(timestamp)
    if world_from_lidar is None or lidar_from_world is None:
        record["reason"] = "no_pose"
        return record

    # 守卫①：洞两侧邻居在世界系里必须"没动"，且与叠出来的框一致。
    observed = sorted(neighbor_positions)
    left = _neighbor_of(observed, frame_index, "left")
    right = _neighbor_of(observed, frame_index, "right")
    if left is None or right is None:
        record["reason"] = "no_bracketing_neighbour"
        return record
    center_tol = float(config.neighbor_center_tol_m)
    axis_tol = math.radians(float(config.neighbor_yaw_tol_deg))
    gap = float(np.linalg.norm(
        neighbor_positions[left][:2] - neighbor_positions[right][:2]))
    record["neighbour_gap_m"] = round(gap, 4)
    if gap > center_tol:
        record["reason"] = "neighbour_moved"
        return record
    axis_delta = tracking.angle_distance(
        neighbor_yaws[left], neighbor_yaws[right], modulo_pi=True)
    record["neighbour_axis_delta_deg"] = round(math.degrees(axis_delta), 3)
    if axis_delta > axis_tol:
        record["reason"] = "neighbour_axis_mismatch"
        return record
    stacked_center = np.asarray(stacked["world_center"], dtype=np.float64)
    for side, index in (("left", left), ("right", right)):
        distance = float(np.linalg.norm(
            neighbor_positions[index][:2] - stacked_center[:2]))
        if distance > center_tol:
            record["reason"] = f"stack_vs_{side}_neighbour_moved"
            record["stack_offset_m"] = round(distance, 4)
            return record
        delta = tracking.angle_distance(
            neighbor_yaws[index], float(stacked["world_yaw"]), modulo_pi=True)
        if delta > axis_tol:
            record["reason"] = f"stack_vs_{side}_neighbour_axis"
            record["axis_delta_deg"] = round(math.degrees(delta), 3)
            return record

    # 世界系刚性框 -> 该帧 lidar 系。
    point = lidar_from_world @ np.array(
        [stacked_center[0], stacked_center[1], stacked_center[2], 1.0])
    size = np.asarray(stacked["size"], dtype=np.float64)
    box = [float(point[0]), float(point[1]), float(point[2]),
           float(size[0]), float(size[1]), float(size[2]),
           float(_world_yaw_to_local(float(stacked["world_yaw"]),
                                     world_from_lidar))]
    record["box_lidar"] = [round(value, 4) for value in box]

    # "该帧得有点云"：bin 不存在直接跳过。
    frame_id = str(frame["frame_id"])
    if frame_id not in points_cache:
        points_cache[frame_id] = _load_lidar_xyz(clip, frame_id)
    points = points_cache[frame_id]
    if points is None:
        record["reason"] = "no_lidar_frame"
        return record

    # 守卫②：同帧已有别的 Car/Truck 压住补框位置 -> 不补。
    for det in frame.get("detections", []):
        if _class_of(det) not in VEHICLE_CLASSES:
            continue
        if det.get("track_id") is not None and int(det["track_id"]) == int(track_id):
            record["reason"] = "track_already_present"
            return record
        other = det.get("box_lidar")
        if not (isinstance(other, list) and len(other) >= 7
                and tracking.finite_box(dict(det))):
            continue
        iou = tracking.bev_iou(
            box[:2], np.asarray(box[3:6], dtype=np.float64), float(box[6]),
            other[:2], np.asarray(other[3:6], dtype=np.float64), float(other[6]))
        if iou > float(config.overlap_iou_threshold):
            record["reason"] = "overlap_with_existing_vehicle"
            record["overlap_track_id"] = det.get("track_id")
            record["overlap_iou"] = round(float(iou), 4)
            return record

    count = int(count_points_in_boxes(points, [box])[0])
    record["points_in_box"] = count
    if count < int(config.min_points_in_box):
        record["reason"] = "too_few_points"
        return record

    detection: Dict[str, Any] = {
        "class_name": "Car",
        "score": 0.0,
        "box_lidar": box,
        "track_id": int(track_id),
        "region": "static",
        "_step5a_filled": True,
    }
    # visibility 取相邻帧（最近一侧邻居，并列取左）该目标的深拷贝。
    source = left
    visibility = _neighbour_visibility(frames[left], track_id)
    if visibility is None:
        source = right
        visibility = _neighbour_visibility(frames[right], track_id)
    if visibility is not None:
        detection["visibility"] = copy.deepcopy(visibility)
        record["visibility_source_frame"] = str(frames[source]["frame_id"])
    else:
        record["visibility_source_frame"] = None
    record["filled"] = True
    record["detection"] = detection
    return record


def _verify_appended_only(source: Sequence[Mapping[str, Any]],
                          output: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """证明 step5a 只追加（不修改）检测，且没有制造同帧重复 id。"""
    mismatches: List[Dict[str, Any]] = []
    appended = 0
    if len(source) != len(output):
        raise AssertionError("step5a changed the number of frames")
    for frame_index, (raw_frame, out_frame) in enumerate(zip(source, output)):
        if raw_frame.get("frame_id") != out_frame.get("frame_id"):
            raise AssertionError("step5a changed frame order")
        raw_dets = list(raw_frame.get("detections", []))
        out_dets = list(out_frame.get("detections", []))
        if out_dets[:len(raw_dets)] != raw_dets:
            mismatches.append({"frame_index": frame_index,
                               "reason": "existing detection mutated"})
            continue
        extras = out_dets[len(raw_dets):]
        appended += len(extras)
        for det in extras:
            if not det.get("_step5a_filled"):
                mismatches.append({"frame_index": frame_index,
                                   "reason": "appended detection lacks marker"})
        ids = [det.get("track_id") for det in out_dets
               if det.get("track_id") is not None]
        if len(ids) != len(set(ids)):
            mismatches.append({"frame_index": frame_index,
                               "reason": "duplicate track_id in frame"})
    return {"frames": len(source), "appended_detections": appended,
            "mismatches": mismatches, "passed": not mismatches}


def _write(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step45-json", type=Path, required=True)
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--step2-diagnostics", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--min-points-in-box", type=int, default=6)
    parser.add_argument("--max-hole-frames", type=int, default=None,
                        help="单洞长度上限（默认不限 = 全补）")
    parser.add_argument("--max-fills-per-track", type=int, default=None,
                        help="每条轨迹补帧总量上限（默认不限 = 全补）")
    parser.add_argument("--neighbor-center-tol-m", type=float, default=1.0)
    parser.add_argument("--neighbor-yaw-tol-deg", type=float, default=30.0)
    parser.add_argument("--overlap-iou", type=float, default=0.02)
    args = parser.parse_args()
    diagnostics_path = args.diagnostics or args.out_json.with_name(
        args.out_json.stem + "_diagnostics.json")
    config = Step5aConfig(
        min_points_in_box=args.min_points_in_box,
        max_hole_frames=args.max_hole_frames,
        max_fills_per_track=args.max_fills_per_track,
        neighbor_center_tol_m=args.neighbor_center_tol_m,
        neighbor_yaw_tol_deg=args.neighbor_yaw_tol_deg,
        overlap_iou_threshold=args.overlap_iou)
    diagnostics = run(args.step45_json, args.clip, args.step2_diagnostics,
                      args.out_json, diagnostics_path, config)
    print(json.dumps({
        "candidate_slots": diagnostics["candidate_slots"],
        "tracks_considered": len(diagnostics["tracks"]),
        "skipped_tracks": len(diagnostics["skipped_tracks"]),
        "inserted_detections": diagnostics["inserted_detections"],
        "skipped_frames": len(diagnostics["skipped_frames"]),
        "before_detections": diagnostics["before_detections"],
        "after_detections": diagnostics["after_detections"],
        "skip_reason_counts": diagnostics["skip_reason_counts"],
        "append_only_passed": diagnostics["append_only_check"]["passed"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
