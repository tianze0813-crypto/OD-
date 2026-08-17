#!/usr/bin/env python3
"""Step 4 Truck post-processing.

This stage is intentionally narrow: it mutates only ``Truck`` boxes and may add
Truck boxes to existing frames inside a track's observed span. Non-Truck
detections are copied through unchanged.

For each Truck track we:

* estimate one physical size for the whole track from lidar-top points;
* pull observed centers toward a bounded world-frame trajectory, with stronger
  correction where the current box clearly drifts away from the point cloud;
* linearly interpolate missing Truck boxes between two observed frames of the
  same ``track_id``;
* convert every interpolated/smoothed world result back to the lidar-top local
  frame before writing the SUST label.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from filtering import camera_visibility
from geometry import box_geometry as box_geometry
from tracking import tracker_conservative as tracking


_GROUND_CONFIG = box_geometry.GeometryConfig()
_SIZE_BOUNDS = box_geometry._SIZE_BOUNDS


@dataclass(frozen=True)
class TruckBoxFitConfig:
    min_track_observations: int = 5
    min_body_points: int = 15
    body_crop_margin: float = 0.70
    body_z_margin: float = 0.80
    ground_remove_margin: float = 0.12
    xy_low_percentile: float = 2.0
    xy_high_percentile: float = 98.0
    z_low_percentile: float = 1.0
    z_high_percentile: float = 99.0
    coverage_keep_axis: float = 0.58
    size_padding: Tuple[float, float, float] = (0.25, 0.20, 0.18)
    size_evidence_min_frames: int = 5
    size_growth_cap_ratio: float = 1.18
    min_roof_points: int = 18
    max_gap_sec: float = 1.20
    stable_min_points: int = 22
    stable_min_coverage: float = 0.62
    stable_max_point_deviation: float = 0.65
    stable_max_yaw_deviation: float = 0.22
    stable_max_shift: float = 0.42
    severe_max_shift: float = 1.20
    stable_path_blend: float = 0.62
    severe_path_blend: float = 0.90
    path_half_window: int = 2
    path_gap: float = 1.00
    z_max_shift: float = 0.55
    ground_clearance: float = 0.04
    max_push_frames: int = 5
    # "Any overlap is noise" policy: duplicate Truck fragments are removed even
    # when only a small corner of the two boxes intersects.
    overlap_iou_threshold: float = 0.0
    # Yaw stability is the primary key.  Fall back to lifecycle only when
    # both tracks have exactly equal stability; no numerical tolerance is used.
    overlap_stability_tie_eps: float = 0.0


def _oriented_size(box: Sequence[float]) -> Tuple[np.ndarray, bool]:
    dx = float(box[3])
    dy = float(box[4])
    dz = float(box[5])
    long_x = dx >= dy
    size = np.asarray([dx, dy, dz] if long_x else [dy, dx, dz],
                      dtype=np.float64)
    return size, long_x


def _body_points(points: np.ndarray, box: Sequence[float],
                 ground_z: float | None, size_xy: np.ndarray,
                 config: TruckBoxFitConfig) -> Tuple[np.ndarray, np.ndarray]:
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    local = box_geometry._local_xy(points[:, :2], (x, y), yaw)
    half = size_xy[:2] / 2.0 + config.body_crop_margin
    bottom = float(ground_z) if ground_z is not None else z - dz / 2.0 - 0.25
    top = z + dz / 2.0 + config.body_z_margin
    mask = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (points[:, 2] >= bottom)
        & (points[:, 2] <= top)
    )
    return local[mask], points[mask, 2]


def _point_evidence(points: np.ndarray, box: Sequence[float],
                    ground_z: float | None, size_xy: np.ndarray,
                    config: TruckBoxFitConfig) -> Dict[str, Any] | None:
    local, z_values = _body_points(points, box, ground_z, size_xy, config)
    if len(local) < config.min_body_points:
        return None

    bottom = float(ground_z) if ground_z is not None else float(
        box[2] - box[5] / 2.0 - 0.25)
    above = z_values >= bottom + config.ground_remove_margin
    valid_local = local[above]
    valid_z = z_values[above]
    if len(valid_local) < max(8, config.min_body_points // 2):
        valid_local = local
        valid_z = z_values
    if len(valid_local) < config.min_body_points:
        return None

    lo_xy = np.percentile(valid_local[:, :2], config.xy_low_percentile, axis=0)
    hi_xy = np.percentile(valid_local[:, :2], config.xy_high_percentile, axis=0)
    span_xy = hi_xy - lo_xy
    z_lo, z_hi = np.percentile(valid_z, [
        config.z_low_percentile, config.z_high_percentile])
    z_base = float(ground_z) if ground_z is not None else float(z_lo)
    height = max(0.20, float(z_hi) - z_base)
    coverage = np.clip(span_xy / np.maximum(size_xy[:2], 1e-6), 0.0, 1.2)
    roof_points = int(np.sum(valid_z >= float(z_hi - 0.70)))
    return {
        "count": int(len(valid_local)),
        "lo_xy": lo_xy,
        "hi_xy": hi_xy,
        "span_xy": span_xy,
        "height": height,
        "coverage": coverage,
        "median_local_xy": np.median(valid_local[:, :2], axis=0),
        "roof_points": roof_points,
        "height_evidence": (
            ground_z is not None and roof_points >= config.min_roof_points),
    }


def _segments(items: Sequence[MutableMapping[str, Any]],
              gap_sec: float) -> List[List[MutableMapping[str, Any]]]:
    result: List[List[MutableMapping[str, Any]]] = []
    current: List[MutableMapping[str, Any]] = []
    for item in items:
        if current:
            dt = (int(item["timestamp"]) - int(current[-1]["timestamp"])) / 1e9
            if dt > gap_sec:
                result.append(current)
                current = []
        current.append(item)
    if current:
        result.append(current)
    return result


def _face_shift_from_evidence(item: MutableMapping[str, Any],
                              evidence: Mapping[str, Any]) -> np.ndarray:
    """Return a bounded local xy shift that aligns observed box faces.

    A median anchor is wrong for long, partially scanned vehicles: it pulls the
    center toward the middle of whatever part of the side is visible. This
    function instead aligns the box face to the observed low/high percentile on
    axes where only one side has evidence.
    """
    box = item["det"]["box_lidar"]
    half = np.asarray([float(box[3]) / 2.0, float(box[4]) / 2.0],
                      dtype=np.float64)
    lo = np.asarray(evidence["lo_xy"], dtype=np.float64)
    hi = np.asarray(evidence["hi_xy"], dtype=np.float64)
    coverage = np.asarray(evidence["coverage"], dtype=np.float64)
    delta_local = np.zeros(2, dtype=np.float64)

    for axis in range(2):
        if coverage[axis] >= 0.68:
            delta_local[axis] = float((lo[axis] + hi[axis]) / 2.0)
            continue
        gap_lo = float(lo[axis] + half[axis])
        gap_hi = float(half[axis] - hi[axis])
        if gap_lo <= gap_hi:
            delta_local[axis] = float(lo[axis] + half[axis])
        else:
            delta_local[axis] = float(hi[axis] - half[axis])

    yaw = float(item["det"]["box_lidar"][6])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    return np.asarray([
        delta_local[0] * cosine - delta_local[1] * sine,
        delta_local[0] * sine + delta_local[1] * cosine,
    ], dtype=np.float64)


def _estimate_track_size(items: Sequence[MutableMapping[str, Any]],
                         lidar: box_geometry._LidarCache,
                         config: TruckBoxFitConfig) -> Tuple[np.ndarray, List[int]]:
    physical_sizes: List[np.ndarray] = []
    weights: List[float] = []
    candidates: List[np.ndarray] = []
    evidence_indices: List[int] = []

    for item in items:
        box = item["det"]["box_lidar"]
        physical_size, _long_x = _oriented_size(box)
        physical_sizes.append(physical_size)
        points = lidar.get(item["frame_id"])
        if points is None:
            weights.append(max(float(item["det"].get("score", 0.0)), 0.05))
            candidates.append(physical_size.copy())
            continue
        ground_z, _ = box_geometry._estimate_ground(points, box, _GROUND_CONFIG)
        size_xy = np.asarray([float(box[3]), float(box[4])], dtype=np.float64)
        evidence = _point_evidence(points, box, ground_z, size_xy, config)
        item["evidence"] = evidence
        if evidence is None:
            weights.append(max(float(item["det"].get("score", 0.0)), 0.05))
            candidates.append(physical_size.copy())
            continue

        weight = (max(float(item["det"].get("score", 0.0)), 0.05)
                  * math.sqrt(max(int(evidence["count"]), 1)))
        weights.append(weight)
        candidate = physical_size.copy()
        for axis in range(2):
            if float(evidence["coverage"][axis]) >= config.coverage_keep_axis:
                candidate[axis] = (float(evidence["span_xy"][axis])
                                   + config.size_padding[axis])
        if evidence["height_evidence"]:
            candidate[2] = float(evidence["height"]) + config.size_padding[2]
        candidates.append(candidate)
        evidence_indices.append(int(item["frame_index"]))

    # Robust detector-size prior.  Low-coverage frames must not contribute
    # their own (possibly wild) physical size to the final fit.
    physical_sizes_array = np.asarray(physical_sizes, dtype=np.float64)
    if physical_sizes_array.size:
        physical_ref = np.asarray([
            box_geometry._weighted_quantile(
                physical_sizes_array[:, axis], weights, 0.75)
            for axis in range(3)
        ], dtype=np.float64)
    else:
        physical_ref = np.asarray([6.0, 2.5, 3.0], dtype=np.float64)

    # Tight point-cloud fit.  Each axis that is observed with enough coverage
    # contributes ``span + padding``; the track size is a robust upper
    # quantile of those observations rather than a maximum over all frames.
    # This lets consistent lidar evidence shrink an over-sized detector box.
    axis_values: List[List[float]] = [[], [], []]
    axis_weights: List[List[float]] = [[], [], []]
    for item, candidate in zip(items, candidates):
        evidence = item.get("evidence")
        if evidence is None:
            continue
        weight = (max(float(item["det"].get("score", 0.0)), 0.05)
                  * math.sqrt(max(int(evidence["count"]), 1)))
        for axis in range(2):
            if float(evidence["coverage"][axis]) >= config.coverage_keep_axis:
                axis_values[axis].append(float(candidate[axis]))
                axis_weights[axis].append(weight)
        if evidence["height_evidence"]:
            axis_values[2].append(float(candidate[2]))
            axis_weights[2].append(weight)

    fixed = physical_ref.copy()
    for axis in range(3):
        if len(axis_values[axis]) >= config.size_evidence_min_frames:
            fitted = box_geometry._weighted_quantile(
                axis_values[axis], axis_weights[axis], 0.90)
            fitted = min(
                float(fitted),
                float(physical_ref[axis]) * config.size_growth_cap_ratio)
            fixed[axis] = float(fitted)

    bounds = _SIZE_BOUNDS.get("Truck")
    if bounds is not None:
        fixed = np.asarray([
            float(np.clip(fixed[axis], bounds[axis][0], bounds[axis][1]))
            for axis in range(3)
        ], dtype=np.float64)
    fixed = np.maximum(fixed, np.asarray([1.0, 0.8, 0.8], dtype=np.float64))
    return fixed, evidence_indices


def _wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _filter_truck_overlaps(frames: Sequence[Dict[str, Any]],
                           config: TruckBoxFitConfig) -> Dict[str, Any]:
    """Drop the shorter-lived side of every overlapping Truck pair.

    Two Truck boxes that intersect in any amount are treated as duplicate
    fragments of one physical object.  This also removes yaw-rotated fragment
    tracks that cross the stable track instead of trying to repair their yaw.
    """
    by_id: Dict[int, Dict[str, Any]] = {}
    for frame in frames:
        for det in frame.get("detections", []):
            if (det.get("class_name") != "Truck"
                    or det.get("track_id") is None
                    or not tracking.finite_box(det)):
                continue
            tid = int(det["track_id"])
            entry = by_id.setdefault(tid, {"frames": set(), "yaws": []})
            entry["frames"].add(int(frame["frame_id"]))
            entry["yaws"].append(float(det["box_lidar"][6]))

    lifecycle = {tid: len(v["frames"]) for tid, v in by_id.items()}
    stability = {}
    for tid, v in by_id.items():
        yaws = np.asarray(v["yaws"], dtype=np.float64)
        stability[tid] = float(np.abs(np.mean(np.exp(2.0j * yaws)))) if yaws.size else 0.0

    drop: set[int] = set()
    overlap_events = []
    for frame in frames:
        trucks = [det for det in frame.get("detections", [])
                  if det.get("class_name") == "Truck"
                  and det.get("track_id") is not None
                  and tracking.finite_box(det)]
        for i in range(len(trucks)):
            for j in range(i + 1, len(trucks)):
                a, b = trucks[i], trucks[j]
                iou = tracking.bev_iou(
                    a["box_lidar"][:2], a["box_lidar"][3:5], a["box_lidar"][6],
                    b["box_lidar"][:2], b["box_lidar"][3:5], b["box_lidar"][6])
                if iou <= config.overlap_iou_threshold:
                    continue
                id_a, id_b = int(a["track_id"]), int(b["track_id"])
                if id_a in drop or id_b in drop:
                    continue
                if abs(stability[id_a] - stability[id_b]) >= config.overlap_stability_tie_eps:
                    loser = (id_a if stability[id_a] < stability[id_b]
                             else id_b)
                elif lifecycle[id_a] != lifecycle[id_b]:
                    loser = id_a if lifecycle[id_a] < lifecycle[id_b] else id_b
                else:
                    loser = max(id_a, id_b)
                drop.add(loser)
                overlap_events.append({
                    "frame_id": str(frame["frame_id"]),
                    "truck_a": id_a,
                    "truck_b": id_b,
                    "iou": round(float(iou), 4),
                    "lifecycle_a": lifecycle[id_a],
                    "lifecycle_b": lifecycle[id_b],
                    "stability_a": round(float(stability[id_a]), 4),
                    "stability_b": round(float(stability[id_b]), 4),
                    "dropped": loser,
                })

    removed = 0
    for frame in frames:
        old = frame.get("detections", [])
        frame["detections"] = [d for d in old if d.get("track_id") not in drop]
        removed += len(old) - len(frame["detections"])
        frame["num_detections"] = len(frame["detections"])
    return {
        "trucks_checked": len(by_id),
        "trucks_removed": len(drop),
        "dropped_track_ids": sorted(drop),
        "boxes_removed": removed,
        "overlap_events": overlap_events,
    }


def _smooth_world_centers(items: Sequence[MutableMapping[str, Any]],
                          config: TruckBoxFitConfig) -> Dict[str, Any]:
    stats = {"segments": 0, "stable_boxes": 0, "severe_boxes": 0}
    for segment in _segments(items, config.max_gap_sec):
        stats["segments"] += 1
        for item in segment:
            raw = item["raw_world"][:2]
            item["point_world_xy"] = raw.copy()
            evidence = item.get("evidence")
            if evidence is not None:
                shift_lidar = _face_shift_from_evidence(item, evidence)
                item["point_world_xy"] = raw + (
                    item["world_from_lidar"][:3, :3] @ np.asarray(
                        [shift_lidar[0], shift_lidar[1], 0.0]))[:2]

        for index, item in enumerate(segment):
            lo = max(0, index - config.path_half_window)
            hi = min(len(segment), index + config.path_half_window + 1)
            window = [other for other in segment[lo:hi]
                      if abs(int(other["timestamp"]) - int(item["timestamp"])) / 1e9
                      <= config.path_gap]
            if not window:
                item["smoothed_world_xy"] = item["point_world_xy"].copy()
                continue
            times = np.asarray([
                (int(other["timestamp"]) - int(item["timestamp"])) / 1e9
                for other in window
            ], dtype=np.float64)
            values = np.asarray([other["point_world_xy"] for other in window])
            weights = np.exp(-0.5 * (times / 0.45) ** 2)
            weights *= np.asarray([
                max(float(other["det"].get("score", 0.0)), 0.10)
                * math.sqrt(max(int(other.get("evidence", {}).get("count", 1)), 1))
                if isinstance(other.get("evidence"), dict) else 1.0
                for other in window
            ])
            degree = 2 if len(window) >= 5 else 1
            design = np.column_stack([times ** power for power in range(degree + 1)])
            coefficients, *_ = np.linalg.lstsq(
                design * np.sqrt(weights)[:, None],
                values * np.sqrt(weights)[:, None], rcond=None)
            fitted = coefficients[0]
            stable = bool(item.get("stable", False))
            blend = config.stable_path_blend if stable else config.severe_path_blend
            max_shift = config.stable_max_shift if stable else config.severe_max_shift
            candidate = item["raw_world"][:2] + blend * (fitted - item["raw_world"][:2])
            item["smoothed_world_xy"] = box_geometry._clip_vector(
                candidate, item["raw_world"][:2], max_shift)
            item["trajectory_residual"] = float(np.linalg.norm(
                fitted - item["raw_world"][:2]))
            stats["stable_boxes" if stable else "severe_boxes"] += 1
    return stats


def _classify_stability(items: Sequence[MutableMapping[str, Any]],
                        config: TruckBoxFitConfig) -> None:
    world_yaws = np.asarray([float(item["yaw_world"]) for item in items])
    mean_vector = np.asarray([
        float(np.mean(np.cos(world_yaws))),
        float(np.mean(np.sin(world_yaws))),
    ])
    median_yaw = math.atan2(mean_vector[1], mean_vector[0])
    for item in items:
        evidence = item.get("evidence")
        count = int(evidence["count"]) if evidence else 0
        coverage = (float(np.mean(evidence["coverage"]))
                    if evidence else 0.0)
        point_dev = float(np.linalg.norm(
            item["point_world_xy"] - item["raw_world"][:2]))
        yaw_dev = abs(_wrap_angle(item["yaw_world"] - median_yaw))
        item["stable"] = bool(
            count >= config.stable_min_points
            and coverage >= config.stable_min_coverage
            and point_dev <= config.stable_max_point_deviation
            and yaw_dev <= config.stable_max_yaw_deviation)


def _interpolated_local_center(world_center: Sequence[float],
                               world_from_lidar: np.ndarray) -> np.ndarray:
    world = np.asarray([float(world_center[0]), float(world_center[1]),
                        float(world_center[2]), 1.0], dtype=np.float64)
    return (np.linalg.inv(world_from_lidar) @ world)[:3]


def _interpolate_world_yaw(prev_yaw: float, next_yaw: float,
                           ratio: float) -> float:
    delta = _wrap_angle(next_yaw - prev_yaw)
    return _wrap_angle(prev_yaw + ratio * delta)


def _world_yaw_to_local(world_yaw: float,
                        world_from_lidar: np.ndarray) -> float:
    vec_world = np.asarray([math.cos(world_yaw), math.sin(world_yaw), 0.0])
    vec_local = world_from_lidar[:3, :3].T @ vec_world
    return math.atan2(float(vec_local[1]), float(vec_local[0]))


def _insert_missing_boxes(frames: Sequence[Dict[str, Any]],
                          frame_index: Mapping[int, int],
                          all_timestamps: Sequence[int],
                          track_items: Sequence[MutableMapping[str, Any]],
                          fixed_size: np.ndarray,
                          coords: tracking.CoordinateProvider,
                          config: TruckBoxFitConfig) -> Dict[str, Any]:
    inserted = 0
    events: List[Dict[str, Any]] = []
    det_ids: List[int] = []
    for prev_item, next_item in zip(track_items, track_items[1:]):
        prev_ts = int(prev_item["timestamp"])
        next_ts = int(next_item["timestamp"])
        if next_ts <= prev_ts:
            continue
        gap_sec = (next_ts - prev_ts) / 1e9
        if gap_sec <= 0.0 or gap_sec > config.max_gap_sec:
            continue
        lo = int(np.searchsorted(all_timestamps, prev_ts, side="right"))
        hi = int(np.searchsorted(all_timestamps, next_ts, side="left"))
        if hi <= lo:
            continue
        if hi - lo > config.max_push_frames:
            continue
        prev_world = prev_item["final_world"]
        next_world = next_item["final_world"]
        prev_yaw = float(prev_item["yaw_world"])
        next_yaw = float(next_item["yaw_world"])
        span = (next_ts - prev_ts) / 1e9
        for ts in all_timestamps[lo:hi]:
            frame = frames[frame_index[int(ts)]]
            world_from_lidar = coords.world_from_lidar(int(ts))
            if world_from_lidar is None:
                continue
            ratio = (int(ts) - prev_ts) / 1e9 / max(span, 1e-6)
            world_center = np.asarray([
                prev_world[0] + ratio * (next_world[0] - prev_world[0]),
                prev_world[1] + ratio * (next_world[1] - prev_world[1]),
                prev_world[2] + ratio * (next_world[2] - prev_world[2]),
            ], dtype=np.float64)
            local = _interpolated_local_center(world_center, world_from_lidar)
            world_yaw = _interpolate_world_yaw(prev_yaw, next_yaw, ratio)
            local_yaw = _world_yaw_to_local(world_yaw, world_from_lidar)
            long_x = float(prev_item["det"]["box_lidar"][3]) >= float(
                prev_item["det"]["box_lidar"][4])
            if long_x:
                scale = [float(fixed_size[0]), float(fixed_size[1]), float(fixed_size[2])]
            else:
                scale = [float(fixed_size[1]), float(fixed_size[0]), float(fixed_size[2])]
            det = {
                "class_name": "Truck",
                "score": 0.0,
                "box_lidar": [
                    float(local[0]), float(local[1]), float(local[2]),
                    scale[0], scale[1], scale[2], float(local_yaw),
                ],
                "track_id": int(prev_item["track_id"]),
            }
            frame["detections"].append(det)
            det_ids.append(id(det))
            inserted += 1
            events.append({
                "frame_id": str(ts),
                "track_id": int(prev_item["track_id"]),
                "between": [prev_ts, next_ts],
                "ratio": round(float(ratio), 4),
            })
    return {"inserted": inserted, "events": events, "det_ids": det_ids}


def _finalize_existing_boxes(items: Sequence[MutableMapping[str, Any]],
                             fixed_size: np.ndarray,
                             config: TruckBoxFitConfig) -> None:
    for item in items:
        box = item["det"]["box_lidar"]
        long_x = float(box[3]) >= float(box[4])
        if long_x:
            scale = [float(fixed_size[0]), float(fixed_size[1]), float(fixed_size[2])]
        else:
            scale = [float(fixed_size[1]), float(fixed_size[0]), float(fixed_size[2])]
        box[3:6] = scale
        world_xy = item["smoothed_world_xy"]
        world_z = float(item["raw_world"][2])
        ground_z = item.get("ground_z")
        if ground_z is not None:
            target_world_z = float(box_geometry._transform_point(
                item["world_from_lidar"],
                [float(box[0]), float(box[1]),
                 float(ground_z) + config.ground_clearance + float(fixed_size[2]) / 2.0])[2])
            world_z = target_world_z
        local = box_geometry._transform_point(
            item["lidar_from_world"], [world_xy[0], world_xy[1], world_z])
        old_z = float(box[2])
        z_delta = float(np.clip(local[2] - old_z, -config.z_max_shift, config.z_max_shift))
        box[:3] = [float(local[0]), float(local[1]), float(old_z + z_delta)]
        item["final_world"] = tracking.center_world(
            box, item["world_from_lidar"])


def apply_truck_box_fit(
        frames: Sequence[Dict[str, Any]],
        coords: tracking.CoordinateProvider,
        clip: Path,
        config: TruckBoxFitConfig = TruckBoxFitConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = copy.deepcopy(list(frames))
    overlap_filter = _filter_truck_overlaps(output, config)
    all_tracks = box_geometry._build_tracks(output, coords)
    truck_tracks = {tid: items for tid, items in all_tracks.items()
                    if any(d.get("class_name") == "Truck" for d in
                           (item["det"] for item in items))}
    lidar = box_geometry._LidarCache(Path(clip))
    all_timestamps = sorted(int(f["frame_id"]) for f in output)
    frame_index = {int(f["frame_id"]): i for i, f in enumerate(output)}

    track_details = []
    inserted_events: List[Dict[str, Any]] = []
    inserted_det_ids: set[int] = set()
    for track_id, items in sorted(truck_tracks.items()):
        items = [item for item in items if item["det"].get("class_name") == "Truck"]
        items.sort(key=lambda item: item["timestamp"])
        for item in items:
            item["track_id"] = track_id
            box = item["det"]["box_lidar"]
            item["yaw_world"] = tracking.yaw_world(
                float(box[6]), item["world_from_lidar"])
            points = lidar.get(item["frame_id"])
            ground_z = None
            if points is not None:
                ground_z, _ = box_geometry._estimate_ground(points, box, _GROUND_CONFIG)
            item["ground_z"] = ground_z
            item["point_count"] = 0
            item["evidence"] = None
            if points is not None:
                size_xy = np.asarray([float(box[3]), float(box[4])],
                                     dtype=np.float64)
                item["evidence"] = _point_evidence(
                    points, box, ground_z, size_xy, config)
                if item["evidence"] is not None:
                    item["point_count"] = int(item["evidence"]["count"])
            item["raw_world_xy"] = item["raw_world"][:2].copy()
            item["point_world_xy"] = item["raw_world"][:2].copy()
            if item["evidence"] is not None:
                shift_lidar = _face_shift_from_evidence(item, item["evidence"])
                item["point_world_xy"] = item["raw_world"][:2] + (
                    item["world_from_lidar"][:3, :3] @ np.asarray(
                        [shift_lidar[0], shift_lidar[1], 0.0]))[:2]
            item["smoothed_world_xy"] = item["raw_world"][:2].copy()

        fixed_size, evidence_frames = _estimate_track_size(items, lidar, config)
        _classify_stability(items, config)
        smoothing = _smooth_world_centers(items, config)
        _finalize_existing_boxes(items, fixed_size, config)

        inserted = _insert_missing_boxes(
            output, frame_index, all_timestamps, items, fixed_size, coords, config)
        inserted_events.extend(inserted["events"])
        inserted_det_ids.update(inserted["det_ids"])
        track_details.append({
            "track_id": track_id,
            "observations": len(items),
            "inserted_boxes": inserted["inserted"],
            "fixed_physical_size": [round(float(x), 4) for x in fixed_size],
            "evidence_frames": evidence_frames,
            "point_evidence_frames": sum(
                item.get("evidence") is not None for item in items),
            "ground_evidence_frames": sum(
                item.get("ground_z") is not None for item in items),
            "stable_boxes": sum(bool(item.get("stable")) for item in items),
            "severe_boxes": sum(not bool(item.get("stable")) for item in items),
            "max_point_deviation": round(max(
                (float(np.linalg.norm(item["point_world_xy"] - item["raw_world"][:2]))
                 for item in items), default=0.0), 4),
            "max_center_shift": round(max(
                (float(np.linalg.norm(item["smoothed_world_xy"] - item["raw_world"][:2]))
                 for item in items), default=0.0), 4),
            "smoothing": smoothing,
        })

    # Compute occlusion-aware visibility only for newly inserted Truck boxes.
    # Other detections, including existing Trucks, stay byte-for-byte frozen.
    try:
        cams = camera_visibility.load_clip_cameras(Path(clip))
        visibility_stats = {"frames": 0, "checked": 0, "tag1": 0, "tag2": 0}
        for frame in output:
            if not any(id(d) in inserted_det_ids
                       for d in frame.get("detections", [])):
                continue
            temp_dets = copy.deepcopy(frame["detections"])
            camera_visibility.compute_frame_visibility(
                temp_dets, cams, occl_tol=0.3)
            for actual, temp in zip(frame["detections"], temp_dets):
                if id(actual) in inserted_det_ids:
                    actual["visibility"] = copy.deepcopy(temp.get("visibility"))
                    visibility_stats["checked"] += 1
                    visibility_stats[
                        "tag1" if temp.get("visibility", {}).get("tag") == 1
                        else "tag2"] += 1
            visibility_stats["frames"] += 1
    except Exception as exc:  # visibility is non-fatal in this first pass
        visibility_stats = {"error": str(exc)}

    for frame in output:
        frame["num_detections"] = len(frame.get("detections", []))

    before_detections = sum(len(f.get("detections", [])) for f in frames)
    after_detections = sum(len(f.get("detections", [])) for f in output)
    return output, {
        "policy": {
            "pipeline_position": "after_car_box_fit",
            "stage": "truck_box_fit",
            "scope": "Truck only",
            "coordinate_frame": "lidar_top local label, world for smoothing",
            "mutated_fields": [
                "Truck box_lidar[0:6]", "Truck visibility",
                "frame detections for interpolated Trucks"],
            "frozen_fields": [
                "non-Truck detections", "track_id", "class_name"],
            "interpolation": "world linear interpolation inside observed span",
            "max_push_frames": config.max_push_frames,
        },
        "truck_tracks": len(truck_tracks),
        "truck_overlap_filter": overlap_filter,
        "before_detections": before_detections,
        "after_detections": after_detections,
        "interpolated_truck_boxes": sum(
            detail["inserted_boxes"] for detail in track_details),
        "net_detection_change": after_detections - before_detections,
        "visibility": visibility_stats,
        "inserted_events": inserted_events,
        "details": track_details,
    }
