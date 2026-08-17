"""V2 yaw preview with motion confirmation and stationary point-cloud axes.

This stage runs after identity tracking and all filters. Motion/static evidence
selects only the source of ``box_lidar[6]`` and never feeds back into tracking.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from geometry.yaw_static_direction import (
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
    pointcloud_min_valid_frames: int = 5
    pointcloud_min_points_per_frame: int = 15
    pointcloud_axis_ratio: float = 2.0
    pointcloud_axis_inlier_deviation: float = math.radians(15.0)
    pointcloud_axis_min_inlier_fraction: float = 0.60
    pointcloud_direction_min_margin: float = 0.15
    pointcloud_raw_yaw_stability: float = 0.98
    pointcloud_raw_axis_conflict: float = math.radians(20.0)


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
) -> Tuple[Dict[Tuple[int, int], float], set[int], List[Dict[str, Any]]]:
    targets: Dict[Tuple[int, int], float] = {}
    moving_ids: set[int] = set()
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
        details.append({
            "track_id": tid,
            "observations": len(items),
            "yaw_mode": "confirmed_motion_heading",
            "onset_observation_index": onset,
            "onset_frame": items[onset]["frame_index"],
            "initial_heading_end_frame": items[initial_end]["frame_index"],
            "initial_world_heading": round(float(initial_heading), 6),
            "fitted_heading_samples": len(headings),
            "prefix_frames_backfilled": onset,
            **confirmation_detail,
        })
    return targets, moving_ids, details


def _raw_detection_map(
        frames: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, int], Dict[str, Any]]:
    result = {}
    for frame in frames:
        timestamp = int(frame["frame_id"])
        for det in frame.get("detections", []):
            if det.get("track_id") is not None:
                result[(timestamp, int(det["track_id"]))] = det
    return result


def _pca_axis(points: np.ndarray) -> Tuple[float, float] | None:
    if len(points) < 3:
        return None
    radius = np.linalg.norm(points, axis=1)
    trimmed = points[radius <= np.percentile(radius, 90.0)]
    if len(trimmed) < 3:
        return None
    covariance = np.cov(trimmed.T)
    values, vectors = np.linalg.eigh(covariance)
    if float(values[-2]) <= 1e-9:
        return None
    axis = vectors[:, int(np.argmax(values))]
    return (math.atan2(float(axis[1]), float(axis[0])),
            float(values[-1] / values[-2]))


def _count_points_in_raw_box(points: np.ndarray,
                             box: Sequence[float]) -> int:
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    cosine, sine = math.cos(-yaw), math.sin(-yaw)
    relative_x = points[:, 0] - x
    relative_y = points[:, 1] - y
    local_x = relative_x * cosine - relative_y * sine
    local_y = relative_x * sine + relative_y * cosine
    return int(np.count_nonzero(
        (np.abs(local_x) <= dx / 2.0)
        & (np.abs(local_y) <= dy / 2.0)
        & (np.abs(points[:, 2] - z) <= dz / 2.0)
    ))


def _raw_yaw_conflicts_with_pointcloud(
        frame_evidence: Sequence[Tuple[float, float, int]],
        pointcloud_axis: float,
        config: YawVehicleDynamicConfig,
) -> Tuple[bool, Dict[str, float]]:
    """Protect a stable detector yaw from a contradictory PCA axis.

    Circular point extraction can include a nearby wall, kerb, or another
    object. A PCA axis is therefore not allowed to replace an already stable
    directed yaw when the two axes strongly disagree.
    """
    raw_yaws = np.asarray([value for value, _weight, _count in frame_evidence],
                          dtype=np.float64)
    directed_vector = np.mean(np.exp(1j * raw_yaws))
    axial_vector = np.mean(np.exp(2j * raw_yaws))
    directed_stability = float(abs(directed_vector))
    raw_axis = 0.5 * float(np.angle(axial_vector))
    axis_conflict = float(tracking.angle_distance(
        raw_axis, pointcloud_axis, modulo_pi=True))
    rejected = (
        directed_stability >= config.pointcloud_raw_yaw_stability
        and axis_conflict >= config.pointcloud_raw_axis_conflict
    )
    return rejected, {
        "raw_directed_yaw_stability": round(directed_stability, 4),
        "raw_axis_world_yaw": round(raw_axis, 6),
        "raw_pointcloud_axis_conflict": round(axis_conflict, 6),
    }


def _stationary_pointcloud_targets(
        tracks: Mapping[int, Sequence[Dict[str, Any]]], moving_ids: set[int],
        pre_yaw_frames: Sequence[Dict[str, Any]], clip: Path,
        config: YawVehicleDynamicConfig,
) -> Tuple[
        Dict[Tuple[int, int], float],
        List[Dict[str, Any]],
        List[Dict[str, Any]],
]:
    raw = _raw_detection_map(pre_yaw_frames)
    lidar_cache: Dict[str, np.ndarray] = {}
    targets: Dict[Tuple[int, int], float] = {}
    details = []
    rejections = []
    for tid, items in tracks.items():
        if tid in moving_ids or len(items) < config.stationary_min_observations:
            continue
        centers = np.asarray([x["world"][:2] for x in items], dtype=np.float64)
        center = np.median(centers, axis=0)
        spread90 = float(np.percentile(np.linalg.norm(centers - center, axis=1), 90))
        if spread90 > config.stationary_center_spread90:
            continue

        aggregate = []
        per_frame_axes = []
        frame_evidence = []
        for item in items:
            frame_id = item["frame_id"]
            if frame_id not in lidar_cache:
                values = np.fromfile(
                    clip / "lidar" / "lidar_top" / f"{frame_id}.bin",
                    dtype=np.float32)
                if values.size % 4 != 0:
                    continue
                lidar_cache[frame_id] = values.reshape(-1, 4)[:, :3]
            points = lidar_cache[frame_id]
            box = np.asarray(item["det"]["box_lidar"][:7], dtype=np.float64)
            delta_xy = points[:, :2] - box[:2]
            radius = 0.58 * math.hypot(float(box[3]), float(box[4]))
            delta_z = points[:, 2] - float(box[2])
            mask = (
                np.linalg.norm(delta_xy, axis=1) <= radius
            ) & (
                delta_z >= -float(box[5]) / 2.0 + 0.12
            ) & (
                delta_z <= float(box[5]) / 2.0 + 0.15
            )
            local_points = points[mask]
            if len(local_points) < config.pointcloud_min_points_per_frame:
                continue
            homogeneous = np.c_[local_points, np.ones(len(local_points))]
            world_points = (item["world_from_lidar"] @ homogeneous.T).T[:, :2]
            relative = world_points - item["world"][:2]
            per_axis = _pca_axis(relative)
            if per_axis is None:
                continue
            aggregate.append(relative)
            per_frame_axes.append(per_axis[0])
            raw_det = raw.get((item["timestamp"], tid), item["det"])
            raw_world_yaw = tracking.yaw_world(
                float(raw_det["box_lidar"][6]), item["world_from_lidar"])
            raw_point_count = _count_points_in_raw_box(
                points, raw_det["box_lidar"])
            weight = raw_point_count * max(
                float(raw_det.get("score", 0.0)), 0.05)
            frame_evidence.append((raw_world_yaw, weight, raw_point_count))
        if len(aggregate) < config.pointcloud_min_valid_frames:
            continue
        combined = _pca_axis(np.concatenate(aggregate, axis=0))
        if combined is None:
            continue
        axis, ratio = combined
        if ratio < config.pointcloud_axis_ratio:
            continue
        deviations = np.asarray([
            tracking.angle_distance(value, axis, modulo_pi=True)
            for value in per_frame_axes
        ])
        inliers = deviations <= config.pointcloud_axis_inlier_deviation
        inlier_fraction = float(np.mean(inliers))
        if inlier_fraction < config.pointcloud_axis_min_inlier_fraction:
            continue

        raw_conflict, raw_conflict_detail = _raw_yaw_conflicts_with_pointcloud(
            frame_evidence, axis, config)
        if raw_conflict:
            rejections.append({
                "track_id": tid,
                "observations": len(items),
                "reason": "stable_raw_yaw_conflicts_with_pointcloud_axis",
                "axis_world_yaw": round(float(axis), 6),
                **raw_conflict_detail,
            })
            continue

        votes = [0.0, 0.0]
        raw_vote_counts = Counter()
        for raw_yaw, weight, _point_count in frame_evidence:
            side = 0 if abs(_wrap(raw_yaw - axis)) <= math.pi / 2.0 else 1
            votes[side] += float(weight)
            raw_vote_counts[side] += 1
        winner = 0 if votes[0] >= votes[1] else 1
        direction_margin = abs(votes[0] - votes[1]) / max(sum(votes), 1e-9)
        if direction_margin < config.pointcloud_direction_min_margin:
            # Direction is ambiguous but the geometric axis remains valid.
            first_raw = frame_evidence[0][0]
            winner = 0 if abs(_wrap(first_raw - axis)) <= math.pi / 2.0 else 1
        target = _wrap(axis + (math.pi if winner else 0.0))
        for item in items:
            box = item["det"]["box_lidar"]
            box_axis_target = target
            if float(box[3]) < float(box[4]):
                box_axis_target -= math.pi / 2.0
            targets[(item["frame_index"], item["detection_index"])] = box_axis_target
        details.append({
            "track_id": tid,
            "observations": len(items),
            "yaw_mode": "stationary_multiframe_pointcloud_axis",
            "center_spread90": round(spread90, 4),
            "valid_pointcloud_frames": len(aggregate),
            "aggregate_points": int(sum(len(x) for x in aggregate)),
            "axis_ratio": round(float(ratio), 4),
            "axis_inlier_fraction": round(inlier_fraction, 4),
            "axis_world_yaw": round(float(axis), 6),
            "direction_votes_weighted": [round(x, 3) for x in votes],
            "direction_votes_frames": [raw_vote_counts[0], raw_vote_counts[1]],
            "direction_margin": round(float(direction_margin), 4),
            "target_world_yaw": round(float(target), 6),
            **raw_conflict_detail,
        })
    return targets, details, rejections


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
    motion_targets, moving_ids, motion_details = _motion_targets(tracks, config)
    point_targets, point_details, point_rejections = (
        _stationary_pointcloud_targets(
        tracks, moving_ids, pre_yaw_frames, Path(clip), config)
    )
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
            if target is not None and timestamp < cutoffs.get(tid, math.inf):
                mode = "static_direction_vote"
            else:
                target = motion_targets.get((frame_index, detection_index))
                if target is not None:
                    mode = "confirmed_motion_heading"
                else:
                    target = point_targets.get((frame_index, detection_index))
                    if target is not None:
                        mode = "stationary_multiframe_pointcloud_axis"
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
            "priority": [
                "static_direction_vote",
                "confirmed_motion_heading",
                "stationary_multiframe_pointcloud_axis",
                "keep_original",
            ],
        },
        "boxes_by_mode": dict(sorted(counts.items())),
        "static": {"tracks": len(static_details), "details": static_details},
        "motion": {"tracks": len(motion_details), "details": motion_details},
        "stationary_pointcloud": {
            "tracks": len(point_details), "details": point_details,
            "stable_raw_yaw_rejections": len(point_rejections),
            "rejection_details": point_rejections,
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
