"""V2 yaw preview with motion confirmation.

This stage runs after identity tracking and all filters. Motion/static evidence
selects only the source of ``box_lidar[6]`` and never feeds back into tracking.

【改动】2026-09-20：删除「静止多帧点云主轴」规则
（原 yaw_mode ``stationary_multiframe_pointcloud_axis``）。该规则用一段被路面/邻车
污染的多帧点云 PCA 轴去覆盖静止车的 yaw，在 0914 clip17 的 obj 1011 上把 yaw 拉歪
16.4°（用户要求整条去掉）。现在静止且没有运动证据的轨迹一律保留 detector 原始 yaw。
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from geometry_yaw_v2.yaw_static_direction import (
    _departure_cutoffs,
    _static_direction_targets,
    _world_yaw_to_local,
    _wrap,
)
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class YawVehicleDynamicConfig:
    static_min_votes: int = 4
    static_min_margin: float = 0.15
    motion_confirm_steps: int = 5
    motion_confirm_min_path: float = 1.5
    motion_confirm_min_net_speed: float = 1.0
    motion_confirm_min_concentration: float = 0.75
    motion_confirm_min_forward_steps: int = 4
    motion_confirm_min_step_progress: float = 0.10
    motion_max_observation_gap: float = 1.2
    motion_initial_distance: float = 3.0
    motion_fit_half_window: int = 3
    motion_fit_min_speed: float = 0.65
    motion_fit_min_displacement: float = 0.45
    stationary_min_observations: int = 5
    stationary_center_spread90: float = 0.45
    # A single distant ID-switch frame must not make a parked fragment look
    # stationary and overwrite its raw yaw.
    stationary_max_center_radius: float = 1.0
    # 【改动】2026-09-20 删除「静止多帧点云主轴」规则时，把它的 8 个 pointcloud_*
    # 阈值一并删掉（点云采样帧数/每帧点数/轴比/内点偏离/内点率/方向裕度/原始 yaw 稳定性/
    # 原始轴冲突）。
    # Dynamic detector yaw is accurate enough; the motion model must not
    # overwrite it during sharp turns / occlusion.  Motion evidence is still
    # computed for diagnostics.
    apply_motion_yaw: bool = False
    # 【改动】True = 保留"静态方向投票"（把静止段 yaw 锁到停车方向）。
    # Truck 链设为 False：不再静态锁死。
    apply_static_direction_vote: bool = True
    # 【改动】直线行驶的轨迹用运动方向作 yaw（修正 detector yaw 的系统偏差）。
    # 判据：净位移/路径长度 >= straight_motion_min_concentration
    #       且 局部拟合航向的圆散布 <= straight_motion_max_heading_spread_deg
    apply_straight_motion_yaw: bool = True
    straight_motion_min_concentration: float = 0.95
    straight_motion_max_heading_spread_deg: float = 15.0


def _track_items(
        frames: Sequence[Dict[str, Any]], coords: tracking.CoordinateProvider,
        tracking_diagnostics: Mapping[str, Any],
        static_yaw_diagnostics: Mapping[str, Any],
) -> Dict[int, List[Dict[str, Any]]]:
    static_ids = {
        int(item["track_id"])
        for item in tracking_diagnostics.get("slot_details", [])
    }
    cutoffs = _departure_cutoffs(static_yaw_diagnostics)
    tracks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            tid = det.get("track_id")
            if tid is None or not tracking.finite_box(det):
                continue
            tid = int(tid)
            if tid in static_ids and timestamp < cutoffs.get(tid, math.inf):
                continue
            if str(det.get("class_name", "")) == "Pedestrian":
                continue
            tracks[tid].append({
                "frame_index": frame_index,
                "detection_index": detection_index,
                "frame_id": str(frame["frame_id"]),
                "timestamp": timestamp,
                "det": det,
                "world": tracking.center_world(det["box_lidar"], world_from_lidar),
                "world_from_lidar": world_from_lidar,
            })
    for items in tracks.values():
        items.sort(key=lambda item: item["timestamp"])
    return tracks


def _window_metrics(items: Sequence[Dict[str, Any]], start: int,
                    steps: int) -> Dict[str, Any] | None:
    window = items[start:start + steps + 1]
    if len(window) != steps + 1:
        return None
    times = np.asarray([item["timestamp"] for item in window],
                       dtype=np.float64) / 1e9
    intervals = np.diff(times)
    if np.any(intervals <= 1e-3):
        return None
    points = np.asarray([item["world"][:2] for item in window],
                        dtype=np.float64)
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    path = float(np.sum(lengths))
    net_vector = points[-1] - points[0]
    net = float(np.linalg.norm(net_vector))
    duration = float(times[-1] - times[0])
    if net <= 1e-6 or duration <= 1e-6:
        return None
    axis = net_vector / net
    progress = vectors @ axis
    return {
        "intervals": intervals,
        "path": path,
        "net": net,
        "duration": duration,
        "concentration": net / max(path, 1e-9),
        "net_speed": net / duration,
        "forward_steps": int(np.count_nonzero(progress > 0.10)),
        "progress": progress,
        "heading": math.atan2(float(net_vector[1]), float(net_vector[0])),
    }


def _confirm_motion_onset(
        items: Sequence[Dict[str, Any]],
        config: YawVehicleDynamicConfig) -> Tuple[int, int, Dict[str, Any]] | None:
    steps = config.motion_confirm_steps
    for start in range(max(0, len(items) - steps)):
        metrics = _window_metrics(items, start, steps)
        if metrics is None:
            continue
        if float(np.max(metrics["intervals"])) > config.motion_max_observation_gap:
            continue
        forward_steps = int(np.count_nonzero(
            metrics["progress"] > config.motion_confirm_min_step_progress))
        if (metrics["path"] < config.motion_confirm_min_path
                or metrics["net_speed"] < config.motion_confirm_min_net_speed
                or metrics["concentration"]
                < config.motion_confirm_min_concentration
                or forward_steps < config.motion_confirm_min_forward_steps):
            continue
        end = start + steps
        detail = {
            "confirmation_window_start": items[start]["frame_index"],
            "confirmation_window_end": items[end]["frame_index"],
            "path": round(float(metrics["path"]), 4),
            "net": round(float(metrics["net"]), 4),
            "net_speed": round(float(metrics["net_speed"]), 4),
            "concentration": round(float(metrics["concentration"]), 4),
            "forward_steps": forward_steps,
        }
        return start, end, detail
    return None


def _initial_motion_heading(items: Sequence[Dict[str, Any]], start: int,
                            minimum_end: int,
                            config: YawVehicleDynamicConfig) -> Tuple[float, int]:
    start_xy = items[start]["world"][:2]
    selected_end = minimum_end
    for end in range(minimum_end, min(len(items), start + 13)):
        gaps = np.diff([x["timestamp"] for x in items[start:end + 1]]) / 1e9
        if len(gaps) and float(np.max(gaps)) > config.motion_max_observation_gap:
            break
        points = np.asarray([x["world"][:2] for x in items[start:end + 1]])
        vectors = np.diff(points, axis=0)
        path = float(np.sum(np.linalg.norm(vectors, axis=1)))
        net_vector = points[-1] - points[0]
        net = float(np.linalg.norm(net_vector))
        if (net >= config.motion_initial_distance
                and net / max(path, 1e-9)
                >= config.motion_confirm_min_concentration):
            selected_end = end
            break
    vector = items[selected_end]["world"][:2] - start_xy
    return math.atan2(float(vector[1]), float(vector[0])), selected_end


def _fit_heading(items: Sequence[Dict[str, Any]], index: int, onset: int,
                 config: YawVehicleDynamicConfig) -> float | None:
    lo = index
    lower_bound = max(onset, index - config.motion_fit_half_window)
    while lo > lower_bound:
        gap = (items[lo]["timestamp"] - items[lo - 1]["timestamp"]) / 1e9
        if gap > config.motion_max_observation_gap:
            break
        lo -= 1
    hi = index + 1
    upper_bound = min(len(items), index + config.motion_fit_half_window + 1)
    while hi < upper_bound:
        gap = (items[hi]["timestamp"] - items[hi - 1]["timestamp"]) / 1e9
        if gap > config.motion_max_observation_gap:
            break
        hi += 1
    window = items[lo:hi]
    if len(window) < 3:
        return None
    times = np.asarray([x["timestamp"] for x in window], dtype=np.float64) / 1e9
    times -= float(np.mean(times))
    denominator = float(np.dot(times, times))
    if denominator <= 1e-9:
        return None
    points = np.asarray([x["world"][:2] for x in window], dtype=np.float64)
    centered = points - np.mean(points, axis=0)
    velocity = times @ centered / denominator
    speed = float(np.linalg.norm(velocity))
    displacement = float(np.linalg.norm(points[-1] - points[0]))
    if (speed < config.motion_fit_min_speed
            or displacement < config.motion_fit_min_displacement):
        return None
    return math.atan2(float(velocity[1]), float(velocity[0]))


def _motion_targets(
        tracks: Mapping[int, Sequence[Dict[str, Any]]],
        config: YawVehicleDynamicConfig,
) -> Tuple[Dict[Tuple[int, int], float], set[int], List[Dict[str, Any]],
           set[int]]:
    targets: Dict[Tuple[int, int], float] = {}
    moving_ids: set[int] = set()
    straight_ids: set[int] = set()          # 【改动】
    details = []
    for tid, items in tracks.items():
        confirmation = _confirm_motion_onset(items, config)
        if confirmation is None:
            continue
        onset, confirmation_end, confirmation_detail = confirmation
        initial_heading, initial_end = _initial_motion_heading(
            items, onset, confirmation_end, config)
        headings: Dict[int, float] = {}
        for index in range(onset, len(items)):
            fitted = _fit_heading(items, index, onset, config)
            if fitted is not None:
                headings[index] = fitted

        # Symmetric regression already smooths the trajectory. A local circular
        # mean removes isolated fit noise without the causal drift seen in V1.
        smoothed: Dict[int, float] = {}
        valid_indices = sorted(headings)
        for index in valid_indices:
            local = [headings[j] for j in valid_indices if abs(j - index) <= 1]
            vector = np.mean(np.exp(1j * np.asarray(local)))
            if abs(vector) >= 0.55:
                smoothed[index] = float(np.angle(vector))

        held = initial_heading
        output_headings = [initial_heading for _ in items]
        for index in range(onset, len(items)):
            if index in smoothed:
                candidate = smoothed[index]
                # An isolated reversal is physically implausible; keep the last
                # reliable direction. Genuine turns remain gradual in the fit.
                if abs(_wrap(candidate - held)) <= math.radians(75.0):
                    held = candidate
            output_headings[index] = held

        for item, heading in zip(items, output_headings):
            box = item["det"]["box_lidar"]
            target = heading
            if float(box[3]) < float(box[4]):
                target -= math.pi / 2.0
            targets[(item["frame_index"], item["detection_index"])] = target
        moving_ids.add(tid)
        # 【改动】直线行驶判据：净位移/路径长度 + 局部拟合航向的圆散布
        straight = False
        spread_deg = None
        concentration = None
        if len(headings) >= 3:
            pts = np.asarray([x["world"][:2] for x in items[onset:]], dtype=np.float64)
            if len(pts) >= 3:
                net = float(np.linalg.norm(pts[-1] - pts[0]))
                path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
                concentration = net / max(path, 1e-9)
            vec = np.mean(np.exp(1j * np.asarray(list(headings.values()))))
            magnitude = float(abs(vec))
            spread_deg = math.degrees(math.sqrt(max(
                0.0, -2.0 * math.log(max(magnitude, 1e-9)))))
            if (concentration is not None
                    and concentration >= float(config.straight_motion_min_concentration)
                    and spread_deg <= float(config.straight_motion_max_heading_spread_deg)):
                straight = True
        if straight:
            straight_ids.add(tid)
        details.append({
            "track_id": tid,
            "observations": len(items),
            "yaw_mode": "confirmed_motion_heading",
            "onset_observation_index": onset,
            "onset_frame": items[onset]["frame_index"],
            "initial_heading_end_frame": items[initial_end]["frame_index"],
            "initial_world_heading": round(float(initial_heading), 6),
            "fitted_heading_samples": len(headings),
            "straight_motion": bool(straight),                    # 【改动】
            "straight_concentration": (None if concentration is None
                                       else round(float(concentration), 4)),
            "heading_spread_deg": (None if spread_deg is None
                                   else round(float(spread_deg), 3)),
            "prefix_frames_backfilled": onset,
            **confirmation_detail,
        })
    return targets, moving_ids, details, straight_ids   # 【改动】


def _raw_detection_map(
        frames: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    result = {}
    for frame in frames:
        timestamp = int(frame["frame_id"])
        for det in frame.get("detections", []):
            if det.get("track_id") is not None:
                result[(timestamp, int(det["track_id"]))] = det
    return result




def apply_yaw_vehicle_dynamic(
        final_frames: Sequence[Dict[str, Any]],
        pre_yaw_frames: Sequence[Dict[str, Any]],
        coords: tracking.CoordinateProvider,
        clip: Path,
        tracking_diagnostics: Mapping[str, Any],
        static_yaw_diagnostics: Mapping[str, Any],
        config: YawVehicleDynamicConfig = YawVehicleDynamicConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = copy.deepcopy(list(final_frames))
    # Reuse only V1's reviewed static direction vote. Its config fields are
    # structurally compatible with the static helper.
    static_targets, static_details = _static_direction_targets(
        output, pre_yaw_frames, coords, static_yaw_diagnostics, config)
    tracks = _track_items(
        output, coords, tracking_diagnostics, static_yaw_diagnostics)
    motion_targets, moving_ids, motion_details, straight_ids = _motion_targets(
        tracks, config)   # 【改动】
    cutoffs = _departure_cutoffs(static_yaw_diagnostics)

    counts = Counter()
    for frame_index, frame in enumerate(output):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            tid = det.get("track_id")
            if tid is None or not tracking.finite_box(det):
                continue
            tid = int(tid)
            target = static_targets.get(tid)
            mode = None
            # 【修·2026-09-21】驶离停车位（超过 cutoff）之后不能再拿静止方向锁 yaw：
            # 旧代码在这种情况下 target 仍非空、但 mode 保持 None -> 既把停车方向写进 yaw，
            # 又让诊断 boxes_by_mode 出现 None 键（与字符串键混排时 sorted() 直接 TypeError）。
            if (target is not None and timestamp >= cutoffs.get(tid, math.inf)):
                target = None
            # 【改动】apply_static_direction_vote=False 时不把静止段 yaw 锁到停车方向
            if (target is not None
                    and getattr(config, "apply_static_direction_vote", True)):
                mode = "static_direction_vote"
            elif target is None or not getattr(
                    config, "apply_static_direction_vote", True):
                motion_target = motion_targets.get(
                    (frame_index, detection_index))
                use_straight = (getattr(config, "apply_straight_motion_yaw", False)
                                and tid in straight_ids)          # 【改动】
                if motion_target is not None and (config.apply_motion_yaw
                                                  or use_straight):
                    target = motion_target
                    mode = ("confirmed_motion_heading" if config.apply_motion_yaw
                            else "straight_motion_heading")       # 【改动】
                # 【改动】2026-09-20 这里原来还有一条「静止多帧点云主轴」兜底
                # （point_targets -> stationary_multiframe_pointcloud_axis），已按用户
                # 要求删除：运动证据用不上时不再改写 yaw，保留 detector 原始朝向。
            if target is None:
                continue
            det["box_lidar"][6] = _world_yaw_to_local(target, world_from_lidar)
            counts[mode] += 1

    _verify_yaw_only(final_frames, output)
    return output, {
        "policy": {
            "pipeline_position": "after_identity_class_filters_and_short_tracks",
            "tracking_feedback": False,
            "mutated_field": "box_lidar[6]",
            "apply_motion_yaw": bool(config.apply_motion_yaw),
            "priority": (
                ["static_direction_vote"]
                + (["confirmed_motion_heading"] if config.apply_motion_yaw
                   else [])
                + ["keep_original"]                                  # 【改动】删掉静止点云主轴
            ),
        },
        "boxes_by_mode": dict(sorted(counts.items(),
                                    key=lambda kv: str(kv[0]))),
        "static": {"tracks": len(static_details), "details": static_details},
        "motion": {"tracks": len(motion_details), "details": motion_details},
        # 【改动】2026-09-20 规则已删除，这里保留一条显式记录便于排查（旧日志里
        # 该键带 tracks/details，现在恒为 enabled=False）。
        "stationary_pointcloud": {
            "enabled": False,
            "reason": "rule removed (2026-09-20): 静止多帧点云 PCA 主轴覆盖已删除，"
                      "静止且无运动证据的轨迹保留 detector yaw",
        },
    }


def _verify_yaw_only(before: Sequence[Dict[str, Any]],
                     after: Sequence[Dict[str, Any]]) -> None:
    if len(before) != len(after):
        raise AssertionError("V2 yaw preview changed frame count")
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("V2 yaw preview changed frame order")
        left = left_frame.get("detections", [])
        right = right_frame.get("detections", [])
        if len(left) != len(right):
            raise AssertionError("V2 yaw preview changed detection count")
        for left_det, right_det in zip(left, right):
            comparable = copy.deepcopy(right_det)
            comparable["box_lidar"][6] = left_det["box_lidar"][6]
            if comparable != left_det:
                raise AssertionError("V2 yaw preview changed a non-yaw field")
