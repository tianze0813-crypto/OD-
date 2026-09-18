"""Truck 专用后处理（【改动】按用户 2026-09-18 需求）。

顺序（用户指定，④ 在 ③ 之后）：
  ① fix_rotated_yaw        yaw 相对【局部运动方向】偏差 >= 30° 的帧：保留框，只把 yaw 修正过来
  ② merge_overlapping_trucks  同一帧内 BEV IoU >= 0.1 的两个 Truck 合并成【并集长框】，yaw 重新算
  ③ fit_truck_xy           只调 xy：先移动中心、再按可见性收缩（复用 main 链的 _fit_xy_shrink_only）
                           不做尺寸下限限制，不动 z
  ④ flip_yaw_reversals     轨迹主方向偏差 > 90° 的帧 yaw += pi（main 链 revert_dynamic_yaw 逻辑）
"""
from __future__ import annotations

import copy
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from geometry import box_geometry
from geometry.car_box_fit import CarBoxFitConfig, _fit_xy_shrink_only
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class TruckPostConfig:
    # ① yaw 旋转帧
    yaw_fix_threshold_deg: float = 15.0     # 【改动】30 -> 15：转弯帧偏差 15~30° 之前修不到
    yaw_fix_speed_min: float = 0.35
    yaw_fix_half_window: int = 2
    # 【改动·默认关】"统一朝向"只允许作用在发生合并的 box 上（用户 2026-09-18 明确）；
    #   这条会把【所有】转弯轨迹整条换成运动方向，范围超出要求，故默认关。
    #   打开后：转弯轨迹(集中度<0.95 且净位移>=1m) 每帧 yaw = 局部运动方向(速度>=1m/s)
    yaw_turn_unify: bool = False
    yaw_turn_concentration_max: float = 0.95   # 集中度低于此值视为转弯
    yaw_turn_min_net: float = 1.0              # 净位移下限(米)
    yaw_turn_speed_min: float = 1.0            # 【改动】转弯统一用运动方向时的速度下限(米/秒)
    # ② IoU 并集合并（【改动】2026-09-18 用户要求去掉，默认不再执行）
    merge_enabled: bool = False
    merge_iou_threshold: float = 0.10
    merge_max_iterations: int = 6
    # 【改动】合并后同一 id 统一 box：尺寸/朝向取"大簇"的中位，位置用上一帧观测到的世界位置
    merge_unify_size: bool = True
    # 【改动】统一朝向：只对【发生合并的 id】做（非合并的车完全不碰）
    merge_unify_yaw: bool = True
    merge_position_mode: str = "previous_world"   # previous_world | keep
    # 【改动】静止 Truck 的 yaw 平滑
    static_smooth_enabled: bool = True
    static_max_net: float = 1.5          # 净位移小于此值视为静止(米)
    static_max_step: float = 0.60        # 单帧最大位移小于此值(米)
    static_mode_bin_deg: float = 5.0     # 众数投票的分箱宽度(度)
    static_min_observations: int = 3
    merge_pca_min_points: int = 25
    # 【改动】并集框 yaw 用点云 PCA 重算时，允许的最大偏离；超过就保留并集参考轴
    #   实测 clip4 两个源框只差 9.1°(平行)，PCA 却给出差 76° 的轴 → 框横在车上
    merge_pca_max_deviation_deg: float = 30.0
    # ③ xy 贴合
    xy_fit_enabled: bool = True
    # 【改动】默认 off：实测（273 个 Truck 框）即使"只贴合长轴"，也有 20% 的框
    #   被沿长轴挪中心（10.6% 挪 >2m，最大 4.89m）、10% 的框长度被砍一半 → 框"突出"。
    #   原因：卡车常只看到一个端面，face-visibility 拟把可见点簇边缘当成"面"。
    #   可选值 off | long_axis | full（需要时再开）
    xy_fit_mode: str = "off"
    # ④ yaw 翻转
    reversal_threshold_deg: float = 90.0
    reversal_min_net: float = 1.0
    truck_class: str = "Truck"
    vehicle_classes: Tuple[str, ...] = ("Truck", "Bus")


def _wrap(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _is_truck(det: Dict[str, Any], config: TruckPostConfig) -> bool:
    name = tracking.canonical_class_name(det.get("class_name", ""))
    return name in config.vehicle_classes


def _collect_tracks(frames: Sequence[Dict[str, Any]],
                    coords: tracking.CoordinateProvider,
                    config: TruckPostConfig) -> Dict[int, List[Dict[str, Any]]]:
    """按 track_id 收集 (frame_index, world中心, world yaw, det)。"""
    tracks: Dict[int, List[Dict[str, Any]]] = {}
    for frame_index, frame in enumerate(frames):
        world_from_lidar = coords.world_from_lidar(int(frame["frame_id"]))
        if world_from_lidar is None:
            continue
        for det in frame.get("detections", []):
            if not _is_truck(det, config) or det.get("track_id") is None:
                continue
            if not tracking.finite_box(det):
                continue
            box = det["box_lidar"]
            tracks.setdefault(int(det["track_id"]), []).append({
                "frame_index": frame_index,
                "timestamp": int(frame["frame_id"]),
                "det": det,
                "world": tracking.center_world(box, world_from_lidar),
                "yaw": tracking.yaw_world(float(box[6]), world_from_lidar),
                "world_from_lidar": world_from_lidar,
                "size": np.asarray(box[3:6], dtype=np.float64),
            })
    for items in tracks.values():
        items.sort(key=lambda item: item["timestamp"])
    return tracks


def _local_heading(items: Sequence[Dict[str, Any]], index: int,
                   half: int, speed_min: float) -> Optional[float]:
    lo = max(0, index - half)
    hi = min(len(items), index + half + 1)
    window = items[lo:hi]
    if len(window) < 3:
        return None
    times = np.asarray([x["timestamp"] for x in window], dtype=np.float64) / 1e9
    times = times - float(np.mean(times))
    points = np.asarray([x["world"][:2] for x in window], dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    denominator = float(times @ times)
    if denominator <= 1e-9:
        return None
    velocity = times @ centered / denominator
    if float(np.linalg.norm(velocity)) < float(speed_min):
        return None
    return float(math.atan2(float(velocity[1]), float(velocity[0])))


def _track_is_turning(items: Sequence[Dict[str, Any]],
                      config: TruckPostConfig) -> bool:
    """【改动】集中度(净位移/路径长) 低于阈值 → 转弯轨迹。"""
    if len(items) < 3:
        return False
    points = np.asarray([x["world"][:2] for x in items], dtype=np.float64)
    net = float(np.linalg.norm(points[-1] - points[0]))
    path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    if path <= 1e-9 or net < float(config.yaw_turn_min_net):
        return False
    return (net / path) < float(config.yaw_turn_concentration_max)


def _merged_box_target(pool: Sequence[Sequence[Any]]) -> Optional[
        Tuple[np.ndarray, float, int]]:
    """【改动】合并 id 的统一 box = "大 box"（并集框）的中位尺寸 + 中位朝向。

    * 长度：取并集框长边的【中位】（不用单帧最大，避免离群框）
    * 宽/高：取"长边 >= 0.9×目标长"那一档框的中位（并集框的宽被并集算法撑大过，
      不能直接用；同长度档的中位宽才是这辆车的宽）
    * 朝向：同长度档框 world yaw 的圆中位（mod pi）
    * pool 元素 = [dx, dy, dz, world_yaw, is_union]
    """
    if not pool:
        return None
    norm = []
    for item in pool:
        dx, dy, dz, yaw = (float(item[0]), float(item[1]), float(item[2]),
                           float(item[3]))
        flag = bool(item[4]) if len(item) > 4 else False
        norm.append((max(dx, dy), min(dx, dy), dz, yaw, flag))
    union = [x for x in norm if x[4]]
    if not union:
        return None
    target_long = float(np.median([x[0] for x in union]))
    near = [x for x in norm if x[0] >= target_long * 0.9]
    if not near:
        near = union
    size = np.asarray([float(np.median([x[0] for x in near])),
                       float(np.median([x[1] for x in near])),
                       float(np.median([x[2] for x in near]))], dtype=np.float64)
    # 【改动】朝向取"同长度档框"的圆中位（不取并集框：并集框的轴向本身可能被掰歪）
    yaws = np.asarray([x[3] for x in near], dtype=np.float64)
    vector = np.mean(np.exp(1j * 2.0 * yaws))
    if abs(vector) < 1e-9:
        yaw_world = float(np.median(np.mod(yaws, math.pi)))
    else:
        yaw_world = float((math.atan2(float(vector.imag), float(vector.real)) / 2.0)
                          % math.pi)
    return size, yaw_world, len(union)


def fix_rotated_yaw(frames: List[Dict[str, Any]],
                    coords: tracking.CoordinateProvider,
                    config: TruckPostConfig = TruckPostConfig()) -> Dict[str, Any]:
    """① 与局部运动方向偏差 >= 阈值 的帧：保留框，把 yaw 修正到运动方向。"""
    tracks = _collect_tracks(frames, coords, config)
    from geometry.yaw_static_direction import _world_yaw_to_local
    threshold = math.radians(float(config.yaw_fix_threshold_deg))
    fixed = 0
    turn_fixed = 0
    turn_tracks = 0
    details: List[Dict[str, Any]] = []
    for track_id, items in sorted(tracks.items()):
        if len(items) < 3:
            continue
        # 【改动】转弯轨迹：整条都改用局部运动方向（不再只看 >=阈值 的帧）
        turning = (bool(config.yaw_turn_unify)
                   and _track_is_turning(items, config))
        if turning:
            turn_tracks += 1
        changed_in_track = 0
        previous = None
        for index, item in enumerate(items):
            speed_min = (float(config.yaw_turn_speed_min) if turning
                         else float(config.yaw_fix_speed_min))
            heading = _local_heading(items, index, config.yaw_fix_half_window,
                                     speed_min)
            if heading is None:
                continue
            deviation = abs(_wrap(float(item["yaw"]) - heading))
            deviation = min(deviation, math.pi - deviation)
            if not turning and deviation < threshold:
                continue
            # 选与前一帧连续（或与当前 yaw 同类）的 pi 等价表示
            reference = previous if previous is not None else float(item["yaw"])
            candidate = heading
            if abs(_wrap(candidate - reference)) > math.pi / 2.0:
                candidate = _wrap(candidate + math.pi)
            if abs(_wrap(float(item["yaw"]) - candidate)) < 1e-9:
                previous = candidate
                continue
            local = _world_yaw_to_local(candidate, item["world_from_lidar"])
            item["det"]["box_lidar"][6] = float(local)
            item["yaw"] = float(candidate)
            previous = float(candidate)
            fixed += 1
            changed_in_track += 1
            if turning:
                turn_fixed += 1
                item["det"]["_truck_yaw_turn"] = True
        if changed_in_track:
            details.append({"track_id": int(track_id),
                            "observations": len(items),
                            "turning": bool(turning),
                            "fixed_frames": changed_in_track})
    return {"enabled": True, "fixed_detections": fixed,
            "turn_unified_frames": turn_fixed, "turn_tracks": turn_tracks,
            "threshold_deg": float(config.yaw_fix_threshold_deg),
            "tracks": details}


def _corners(box: Sequence[float]) -> np.ndarray:
    x, y, dx, dy, yaw = (float(box[0]), float(box[1]), float(box[3]),
                         float(box[4]), float(box[6]))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    lx = np.array([dx / 2, dx / 2, -dx / 2, -dx / 2])
    ly = np.array([dy / 2, -dy / 2, -dy / 2, dy / 2])
    return np.stack([x + lx * cosine - ly * sine,
                     y + lx * sine + ly * cosine], axis=1)


def _world_xy_to_local(xy_world: Sequence[float],
                       world_from_lidar: np.ndarray) -> Tuple[float, float]:
    """【改动】世界系 xy → 该帧 box 坐标系(base_link) 的 xy。"""
    planar = np.asarray(world_from_lidar, dtype=np.float64)[:2, :2]
    translation = np.asarray(world_from_lidar, dtype=np.float64)[:2, 3]
    residual = np.asarray(xy_world, dtype=np.float64) - translation
    try:
        local = np.linalg.solve(planar, residual)
    except np.linalg.LinAlgError:
        local = np.linalg.lstsq(planar, residual, rcond=None)[0]
    return float(local[0]), float(local[1])


def _yaw_delta(a: float, b: float) -> float:
    """框的 yaw 是 mod pi 的，返回最小夹角（弧度）。"""
    delta = abs(_wrap(float(a) - float(b)))
    return min(delta, math.pi - delta)


def _union_box(left: Sequence[float], right: Sequence[float]) -> Tuple[List[float], float]:
    """2D OBB 并集，返回 (box7, 参考角)。

    【改动】参考轴不能取"两中心连线方向"——两框中心很接近（长车/重复检出）时
    该方向几乎是噪声，会把宽度虚增（实测把两个平行的 13x3 框并成了 13.96x7.96）。
    改为按 yaw 差判断：平行(<=30°) 用长框自身的 yaw 作轴。
    """
    delta = _yaw_delta(left[6], right[6])
    length_left = max(float(left[3]), float(left[4]))
    length_right = max(float(right[3]), float(right[4]))
    longer = left if length_left >= length_right else right
    if delta <= math.radians(30.0):
        reference = float(longer[6])                    # 平行 -> 沿长框轴
    else:
        reference = math.atan2(float(right[1]) - float(left[1]),
                               float(right[0]) - float(left[0]))
    cosine, sine = math.cos(reference), math.sin(reference)
    points = np.vstack([_corners(left), _corners(right)])
    local = (points - np.array([float(longer[0]), float(longer[1])])) @ np.array(
        [[cosine, sine], [-sine, cosine]]).T
    lo, hi = local.min(0), local.max(0)
    center = (lo + hi) / 2.0
    world = np.array([longer[0], longer[1]]) + np.array(
        [cosine * center[0] - sine * center[1],
         sine * center[0] + cosine * center[1]])
    z_lo = min(float(left[2]) - float(left[5]) / 2,
               float(right[2]) - float(right[5]) / 2)
    z_hi = max(float(left[2]) + float(left[5]) / 2,
               float(right[2]) + float(right[5]) / 2)
    return [float(world[0]), float(world[1]), (z_lo + z_hi) / 2.0,
            float(hi[0] - lo[0]), float(hi[1] - lo[1]), float(z_hi - z_lo),
            float(reference)], float(reference)


def merge_overlapping_trucks(
        frames: List[Dict[str, Any]], coords: tracking.CoordinateProvider,
        lidar: Optional[box_geometry._LidarCache] = None,
        config: TruckPostConfig = TruckPostConfig()) -> Dict[str, Any]:
    """② IoU >= 阈值 的 Truck 合并成【轨迹级】的长车（【改动】按用户要求"补全到所有帧"）。

    第 1 遍：扫所有帧，任意一帧里 track A / track B 的 BEV IoU >= 阈值
             → 认为 A、B 是同一辆车，并查集归到同一组
    第 2 遍：逐帧把同组的框并成一个并集框（组内只剩 1 个框的帧原样保留）
             → 整条轨迹统一成同一个规范 id、统一是并集长框
    yaw 由点云主轴重算。
    """
    # ---------- 第 1 遍：找组 ----------
    parent: Dict[int, int] = {}

    def find(x: int) -> int:
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    members: Dict[int, List[Dict[str, Any]]] = {}
    observations: Counter = Counter()
    lengths: Dict[int, List[float]] = {}
    for frame in frames:
        trucks = [d for d in frame.get("detections", [])
                  if _is_truck(d, config) and d.get("track_id") is not None]
        for det in trucks:
            tid = int(det["track_id"])
            find(tid)
            observations[tid] += 1
            lengths.setdefault(tid, []).append(max(float(det["box_lidar"][3]),
                                                   float(det["box_lidar"][4])))
        for a in range(len(trucks)):
            for b in range(a + 1, len(trucks)):
                ba = trucks[a]["box_lidar"]; bb = trucks[b]["box_lidar"]
                iou = tracking.bev_iou(ba[:2], ba[3:6], ba[6], bb[:2], bb[3:6], bb[6])
                if iou >= float(config.merge_iou_threshold):
                    union(int(trucks[a]["track_id"]), int(trucks[b]["track_id"]))

    groups: Dict[int, List[int]] = {}
    for tid in list(parent):
        groups.setdefault(find(tid), []).append(tid)
    canonical: Dict[int, int] = {}
    merged_groups = 0
    for root, tids in groups.items():
        # 规范 id 取组内观测最多（并列取更长）的那个
        target = max(tids, key=lambda t: (observations.get(t, 0),
                                          float(np.median(lengths.get(t, [0.0])))))
        for tid in tids:
            canonical[tid] = target
        if len(tids) > 1:
            merged_groups += 1

    # ---------- 第 2 遍：逐帧并组内框 ----------
    merged_pairs = 0
    merged_frames = 0
    details: List[Dict[str, Any]] = []
    for frame in frames:
        detections = frame.get("detections", [])
        by_group: Dict[int, List[int]] = {}
        for index, det in enumerate(detections):
            if not _is_truck(det, config) or det.get("track_id") is None:
                continue
            by_group.setdefault(canonical.get(int(det["track_id"]),
                                              int(det["track_id"])), []).append(index)
        if not by_group:
            continue
        remove: set = set()
        frame_changed = False
        for group_id, indices in by_group.items():
            alive = list(indices)
            while len(alive) >= 2:
                base = max(alive, key=lambda i: max(float(detections[i]["box_lidar"][3]),
                                                    float(detections[i]["box_lidar"][4])))
                best = None
                for i in alive:
                    if i == base:
                        continue
                    ba = detections[base]["box_lidar"]; bb = detections[i]["box_lidar"]
                    iou = tracking.bev_iou(ba[:2], ba[3:6], ba[6], bb[:2], bb[3:6], bb[6])
                    if best is None or iou > best[0]:
                        best = (iou, i)
                iou, other = best
                source_a = [float(v) for v in detections[base]["box_lidar"][:7]]
                source_b = [float(v) for v in detections[other]["box_lidar"][:7]]
                box, _reference = _union_box(source_a, source_b)
                if lidar is not None:
                    points = lidar.get(frame["frame_id"])
                    if points is not None:
                        axis = _pca_yaw(points, box, config)
                        if axis is not None:
                            box[6] = axis
                detections[base]["box_lidar"] = box
                detections[base]["_truck_union"] = True          # 【改动】并集框标记
                detections[base]["score"] = max(float(detections[base].get("score", 0.0)),
                                                float(detections[other].get("score", 0.0)))
                detections[base]["track_id"] = int(group_id)
                details.append({
                    "frame_id": str(frame["frame_id"]), "group_id": int(group_id),
                    "iou": round(float(iou), 4),
                    "a": [round(v, 3) for v in source_a],
                    "b": [round(v, 3) for v in source_b],
                    "union": [round(float(v), 3) for v in box[:7]],
                    "center_separation": round(float(np.hypot(
                        source_b[0] - source_a[0], source_b[1] - source_a[1])), 3),
                    "yaw_delta_deg": round(math.degrees(
                        _yaw_delta(source_a[6], source_b[6])), 2),
                })
                remove.add(other)
                alive = [i for i in alive if i != other]
                merged_pairs += 1
                frame_changed = True
            # 组内只剩一个框的帧：只统一 id
            for index in indices:
                if index in remove:
                    continue
                detections[index]["track_id"] = int(group_id)
        if remove:
            frame["detections"] = [d for i, d in enumerate(detections) if i not in remove]
            frame["num_detections"] = len(frame["detections"])
        if frame_changed:
            merged_frames += 1
    # ---------- 第 3 遍【改动】同一 id 统一 box ----------
    #   * 尺寸/朝向 = 并集框("大簇")的中位尺寸 + 中位朝向
    #   * 位置：原本小的框(小变大) → xy 用上一帧【观测到】的世界位置
    unify = {"enabled": bool(config.merge_unify_size), "unified_tracks": 0,
             "unified_boxes": 0, "enlarged_boxes": 0, "moved_boxes": 0,
             "position_mode": str(config.merge_position_mode), "tracks": []}
    if bool(config.merge_unify_size) and merged_groups:
        from geometry.yaw_static_direction import _world_yaw_to_local
        items_by_track = _collect_tracks(frames, coords, config)
        for root, tids in groups.items():
            if len(tids) < 2:
                continue
            target_id = int(canonical.get(tids[0], root))
            items = items_by_track.get(target_id)
            if not items:
                items = items_by_track.get(int(root))
            if not items:
                continue
                # 【改动】池子 = 该 id 全部框（含并集框标记）；目标 = 并集框("大 box")的中位
            pool = [[float(x["size"][0]), float(x["size"][1]), float(x["size"][2]),
                     float(x["yaw"]), bool(x["det"].get("_truck_union"))]
                    for x in items]
            target = _merged_box_target(pool)
            if target is None:
                continue
            size, yaw_world, big_count = target
            before_sizes = [round(max(float(x["size"][0]), float(x["size"][1])), 3)
                            for x in items]
            enlarged = moved = 0
            for index, item in enumerate(items):
                box = item["det"]["box_lidar"]
                long_axis = 0 if float(box[3]) >= float(box[4]) else 1
                long_now = max(float(box[3]), float(box[4]))
                was_small = long_now < float(size[0]) - 1e-6
                if bool(config.merge_unify_yaw):
                    box[6] = float(_world_yaw_to_local(yaw_world,
                                                       item["world_from_lidar"]))
                    item["yaw"] = float(yaw_world)
                    box[3], box[4], box[5] = (float(size[0]), float(size[1]),
                                              float(size[2]))
                else:                       # 不统一朝向：长边写回它原本的长轴
                    box[3 + long_axis] = float(size[0])
                    box[3 + (1 - long_axis)] = float(size[1])
                    box[5] = float(size[2])
                if was_small:
                    enlarged += 1
                    if (str(config.merge_position_mode) == "previous_world"
                            and index > 0):
                        previous_world = items[index - 1]["world"][:2]
                        local_x, local_y = _world_xy_to_local(
                            previous_world, item["world_from_lidar"])
                        box[0], box[1] = local_x, local_y
                        moved += 1
                item["det"]["_truck_merged"] = True
            unify["unified_tracks"] += 1
            unify["unified_boxes"] += len(items)
            unify["enlarged_boxes"] += enlarged
            unify["moved_boxes"] += moved
            unify["tracks"].append({
                "track_id": target_id,
                "root": int(root),
                "members": [int(t) for t in tids],
                "observations": len(items),
                "big_pool": int(big_count),
                "union_boxes": int(sum(1 for x in items
                                       if x["det"].get("_truck_union"))),
                "target_size": [round(float(v), 3) for v in size],
                "target_yaw_world_deg": round(math.degrees(yaw_world), 2),
                "length_before": [min(before_sizes), max(before_sizes)],
                "enlarged_boxes": enlarged,
                "moved_boxes": moved,
            })
            # 兜底：组内没被标到的框（理论上不会）
            for item in items:
                item["det"]["_truck_merged"] = True
    return {"enabled": True, "iou_threshold": float(config.merge_iou_threshold),
            "merged_groups": merged_groups, "merged_pairs": merged_pairs,
            "merged_frames": merged_frames, "size_unification": unify,
            "details": details[:40]}


def _pca_yaw(points: np.ndarray, box: Sequence[float],
             config: TruckPostConfig) -> Optional[float]:
    """并集框内点云的主轴方向（弧度）。"""
    xy = points[:, :2]
    yaw = float(box[6])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    relative = xy - np.array([float(box[0]), float(box[1])])
    local = relative @ np.array([[cosine, sine], [-sine, cosine]]).T
    inside = ((np.abs(local[:, 0]) <= float(box[3]) / 2.0)
              & (np.abs(local[:, 1]) <= float(box[4]) / 2.0))
    if int(inside.sum()) < int(config.merge_pca_min_points):
        return None
    patch = local[inside]
    covariance = np.cov(patch.T)
    values, vectors = np.linalg.eigh(covariance)
    if float(values[-1]) <= 1e-9:
        return None
    axis = vectors[:, -1]
    angle = float(math.atan2(float(axis[1]), float(axis[0])))
    while angle - yaw > math.pi / 2:
        angle -= math.pi
    while yaw - angle > math.pi / 2:
        angle += math.pi
    # 【改动】PCA 只能"微调"，不能把框掰横：偏离超过阈值就放弃，保留并集参考轴
    deviation = abs(angle - yaw)
    deviation = min(deviation, math.pi - deviation)
    if deviation > math.radians(float(config.merge_pca_max_deviation_deg)):
        return None
    return float(angle)


def fit_truck_xy(frames: List[Dict[str, Any]],
                 coords: tracking.CoordinateProvider, clip,
                 geometry_config=None,
                 config: TruckPostConfig = TruckPostConfig()) -> Dict[str, Any]:
    """③ 只调 xy：先移动中心、再按可见性收缩（复用 main 链 _fit_xy_shrink_only）。

    【改动】mode：
      * "off"        完全不贴合
      * "long_axis"  只贴合长轴（默认）：只有长轴方向动中心与长度，宽度与横向中心保持 detector 值
      * "full"       原行为（长轴+短轴都动）
    合并过的 id（_truck_merged）已统一尺寸/位置，跳过。
    """
    mode = str(config.xy_fit_mode)
    if not config.xy_fit_enabled or mode == "off":
        return {"enabled": False, "mode": mode}
    from geometry.car_box_fit import CarBoxFitConfig
    fit_config = CarBoxFitConfig()
    tracks = box_geometry._build_tracks(frames, coords)
    lidar = box_geometry._LidarCache(clip)
    fitted = 0
    tracks_touched = 0
    for track_id, items in sorted(tracks.items()):
        if not items:
            continue
        name = tracking.canonical_class_name(
            items[0]["det"].get("class_name", ""))
        if name not in config.vehicle_classes:
            continue
        touched = 0
        for item in items:
            det = item["det"]
            box = det.get("box_lidar")
            if not isinstance(box, list) or len(box) < 7:
                continue
            points = lidar.get(item["frame_id"])
            if points is None:
                continue
            ground_z, _count = box_geometry._estimate_ground(
                points, box, fit_config)
            result = _fit_xy_shrink_only(points, box, ground_z, fit_config)
            if result is None:
                continue
            center_local = np.asarray(result["center_local"], dtype=np.float64)
            size_local = np.asarray(result["size_local"], dtype=np.float64)
            if (not np.all(np.isfinite(center_local))
                    or not np.all(np.isfinite(size_local))):
                continue
            if det.get("_truck_merged"):
                continue                      # 【改动】已统一尺寸/位置的 id 不再贴合
            yaw = float(box[6])
            cosine, sine = math.cos(yaw), math.sin(yaw)
            delta = np.asarray(center_local, dtype=np.float64)
            target = np.asarray([float(size_local[0]), float(size_local[1])],
                                dtype=np.float64)
            if mode == "long_axis":
                # 【改动】只动长轴：短轴长度与横向中心保持原值
                axis = 0 if float(box[3]) >= float(box[4]) else 1
                keep = np.asarray([float(box[3]), float(box[4])], dtype=np.float64)
                delta = np.zeros(2, dtype=np.float64)
                delta[axis] = float(center_local[axis])
                target = keep.copy()
                target[axis] = float(max(size_local[axis], 1e-3))
            box[0] = float(box[0]) + (cosine * delta[0] - sine * delta[1])
            box[1] = float(box[1]) + (sine * delta[0] + cosine * delta[1])
            box[3] = float(max(target[0], 1e-3))
            box[4] = float(max(target[1], 1e-3))
            det["_truck_xy_fit"] = True
            fitted += 1
            touched += 1
        if touched:
            tracks_touched += 1
    return {"enabled": True, "mode": mode, "fitted_detections": fitted,
            "tracks_touched": tracks_touched}


def smooth_static_truck_yaw(
        frames: List[Dict[str, Any]], coords: tracking.CoordinateProvider,
        config: TruckPostConfig = TruckPostConfig()) -> Dict[str, Any]:
    """【改动】静止 Truck 的 yaw 平滑。

    * 静止判据：净位移 < static_max_net 且 单帧最大位移 < static_max_step
    * 众数投票：把各帧 yaw(取 mod pi) 按 static_mode_bin_deg 分箱，票数最多的箱为主方向
      （这样"180° 反转"只会落在同一个 mod-pi 箱里，不会分裂）
    * 抖动中间位置：取主方向箱内样本的【圆形中位】作为整条轨迹的 yaw
      （yaw 是 mod pi 的，用 2*theta 的圆均值再除以 2 求）
    """
    if not config.static_smooth_enabled:
        return {"enabled": False}
    from geometry.yaw_static_direction import _world_yaw_to_local
    tracks = _collect_tracks(frames, coords, config)
    bin_width = math.radians(max(float(config.static_mode_bin_deg), 0.5))
    smoothed_tracks = 0
    smoothed_boxes = 0
    details: List[Dict[str, Any]] = []
    for track_id, items in sorted(tracks.items()):
        if len(items) < int(config.static_min_observations):
            continue
        points = np.asarray([x["world"][:2] for x in items], dtype=np.float64)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        net = float(np.linalg.norm(points[-1] - points[0]))
        max_step = float(np.max(steps)) if len(steps) else 0.0
        if net >= float(config.static_max_net) or max_step >= float(config.static_max_step):
            continue                                  # 不是静止车
        yaws = np.asarray([float(x["yaw"]) for x in items], dtype=np.float64)
        mods = np.mod(yaws, math.pi)
        bins = np.round(mods / bin_width).astype(np.int64)
        counts = Counter(int(b) for b in bins)
        dominant, votes = counts.most_common(1)[0]
        selected = [float(y) for y, b in zip(yaws, bins) if int(b) == dominant]
        # 圆形中位（mod pi）：2*theta 圆均值
        doubled = 2.0 * np.asarray(selected, dtype=np.float64)
        vector = np.mean(np.exp(1j * doubled))
        if abs(vector) < 1e-9:
            continue
        target = float((math.atan2(float(vector.imag), float(vector.real)) / 2.0)
                       % math.pi)
        before = [float(x["det"]["box_lidar"][6]) for x in items]
        for item in items:
            item["det"]["box_lidar"][6] = float(
                _world_yaw_to_local(target, item["world_from_lidar"]))
        smoothed_tracks += 1
        smoothed_boxes += len(items)
        details.append({
            "track_id": int(track_id),
            "observations": len(items),
            "net_displacement": round(net, 3),
            "max_step": round(max_step, 3),
            "dominant_votes": int(votes),
            "distinct_bins": len(counts),
            "target_yaw_deg": round(math.degrees(target), 2),
        })
    return {"enabled": True, "smoothed_tracks": smoothed_tracks,
            "smoothed_boxes": smoothed_boxes,
            "mode_bin_deg": float(config.static_mode_bin_deg),
            "details": details[:40]}


def flip_yaw_reversals(frames: List[Dict[str, Any]],
                       coords: tracking.CoordinateProvider,
                       config: TruckPostConfig = TruckPostConfig()) -> Dict[str, Any]:
    """④ 轨迹主方向偏差 > 阈值 的帧：yaw += pi（main 链 revert_dynamic_yaw 逻辑）。"""
    tracks = _collect_tracks(frames, coords, config)
    threshold = math.radians(float(config.reversal_threshold_deg))
    flipped = 0
    details: List[Dict[str, Any]] = []
    for track_id, items in sorted(tracks.items()):
        if len(items) < 2:
            continue
        points = np.asarray([x["world"][:2] for x in items], dtype=np.float64)
        net_vector = points[-1] - points[0]
        net = float(np.linalg.norm(net_vector))
        if net < float(config.reversal_min_net):
            continue
        path = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if path <= 1e-9:
            continue
        heading = float(math.atan2(float(net_vector[1]), float(net_vector[0])))
        count = 0
        for item in items:
            deviation = abs(_wrap(float(item["yaw"]) - heading))
            deviation = min(deviation, math.pi - deviation)
            if deviation <= threshold:
                continue
            box = item["det"]["box_lidar"]
            box[6] = float(_wrap(float(box[6]) + math.pi))
            item["yaw"] = float(_wrap(float(item["yaw"]) + math.pi))
            item["det"]["_truck_yaw_reversed"] = True
            flipped += 1
            count += 1
        if count:
            details.append({"track_id": int(track_id),
                            "observations": len(items),
                            "flipped_frames": count,
                            "net_displacement": round(net, 3)})
    return {"enabled": True, "threshold_deg": float(config.reversal_threshold_deg),
            "reversed_detections": flipped, "tracks": details}


def apply_truck_postprocess(frames: List[Dict[str, Any]],
                            coords: tracking.CoordinateProvider, clip,
                            config: TruckPostConfig = TruckPostConfig()
                            ) -> Dict[str, Any]:
    """按用户指定顺序执行 ① -> ② -> ③ -> ④。"""
    diagnostics: Dict[str, Any] = {}
    diagnostics["yaw_fix"] = fix_rotated_yaw(frames, coords, config)
    # 【改动】静止 Truck 的 yaw 平滑放在合并之前，这样并集用的是统一后的 yaw
    diagnostics["static_yaw_smooth"] = smooth_static_truck_yaw(frames, coords, config)
    if bool(config.merge_enabled):          # 【改动】默认关：不再做 IoU 并集合并
        lidar = box_geometry._LidarCache(clip)
        diagnostics["iou_merge"] = merge_overlapping_trucks(
            frames, coords, lidar, config)
    else:
        diagnostics["iou_merge"] = {"enabled": False,
                                    "reason": "用户要求去掉 IoU 并集合并"}
    diagnostics["xy_fit"] = fit_truck_xy(frames, coords, clip, None, config)
    return diagnostics
