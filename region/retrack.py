"""Step 4.5 dynamic-region re-tracking and ID inheritance.

This module implements the reviewed plan:

1. build a buffer-free dynamic region from high-speed car-only tracks;
2. freeze every detection outside the region / without high-speed evidence;
3. re-track only the dynamic-region candidate detections with motion-only
   association (no yaw gates) and the existing physical continuity gates;
4. inherit IDs conservatively: static anchor > high-speed main old ID > new ID;
5. build a direction-level four-phase model from the re-tracked trajectories;
6. phase-aware stitch red-light / yielding / green-start fragments;
7. re-run box fitting only for the dynamic / re-tracked detections.

The module never mutates the frozen detections' geometry.  The only fields it
adds to frozen detections are diagnostics.
"""

from __future__ import annotations

import copy
import math
from collections import Counter, defaultdict
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from filtering.hard_filters import count_points_in_boxes
from geometry.car_box_fit import CarBoxFitConfig, apply_car_box_fit
from geometry.yaw_static_direction import _world_yaw_to_local
from region.dynamic_region import (
    DynamicRegionConfig,
    DynamicRegionResult,
    _is_high_speed,
    _track_speed_stats,
    build_dynamic_regions,
)
from region.region_mask import DynamicRegionMask
from region.traffic_light import TrafficLightConfig, build_traffic_light_model
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class Step45Config:
    """Thresholds for step 4.5.  Conservative defaults, tuned on the 5 clips."""

    region: DynamicRegionConfig = field(default_factory=DynamicRegionConfig)
    traffic_light: TrafficLightConfig = field(
        default_factory=TrafficLightConfig)
    # Re-track association.
    dynamic_max_gap_sec: float = 1.8
    # Dynamic-region occlusion state (PLAN section 19, final edge case).
    # 2.6s is the minimum value that connects clip6 28 -> 62.
    occlusion_max_gap_sec: float = 2.6
    # Experimental pass-1 driving-direction noise filter.  Disabled by
    # default because it can lock yaw on some vehicles; kept behind a flag.
    direction_filter_enabled: bool = False
    # Experimental post-pass2 dynamic yaw overwrite.  Disabled by default;
    # enable only after the yaw-lock issue is resolved.
    yaw_align_enabled: bool = False
    # Reviewed single-frame overlap noise filter (pass 1):
    # same-frame Car boxes with IoU > threshold -> remove the one with fewer
    # lidar points inside its box; equal points -> do nothing.
    overlap_filter_enabled: bool = True
    overlap_iou_threshold: float = 0.02
    # Reviewed whole-track yaw reversal (after final IDs):
    # if the motion trajectory is opposite to box yaw, add pi to yaw only.
    yaw_reversal_enabled: bool = True
    yaw_reversal_threshold_deg: float = 150.0
    # Moving seed / pure static classification (PLAN section 19).
    moving_seed_net_min_m: float = 8.0
    moving_seed_concentration_min: float = 0.5
    moving_seed_duration_min_sec: float = 3.0
    moving_seed_low_speed_floor_mps: float = 1.0
    # Weak moving seed: short start / low-speed pull-away (e.g. clip6 7->387).
    # It is used as a queue stitch anchor and as a small region seed.
    weak_seed_net_min_m: float = 2.0
    weak_seed_concentration_min: float = 0.8
    weak_seed_duration_min_sec: float = 1.0
    weak_seed_low_speed_floor_mps: float = 1.0
    static_net_max_m: float = 1.0
    static_span_max_m: float = 1.0
    static_step_max_m: float = 1.0
    # Queue / same-vehicle stitching.
    queue_longitudinal_gap_m: float = 20.0
    queue_stitch_position_tolerance_m: float = 1.5
    queue_stitch_lateral_tolerance_m: float = 5.0
    lane_change_max_lateral_m: float = 5.0
    left_turn_tail_arc_length_m: float = 5.0
    # ID inheritance / slot release.
    boundary_max_gap_sec: float = 2.0
    boundary_max_distance_m: float = 3.5
    boundary_max_heading_deg: float = 45.0
    boundary_max_size_delta: float = 0.35
    boundary_max_speed_mps: float = 1.5
    slot_release_tolerance_sec: float = 2.0
    # Merging a fragment into an existing final id must pass a reachable
    # bridge.  This prevents "merge then split" ID inflation.
    merge_gate_base_m: float = 2.5
    merge_gate_per_sec_m: float = 1.5
    merge_gate_max_m: float = 15.0
    # Phase-aware stitching.
    phase_merge_max_gap_sec: float = 30.0
    yielding_max_gap_sec: float = 6.0
    bridge_base_m: float = 2.5
    bridge_per_sec_m: float = 1.5
    bridge_max_m: float = 12.0
    phase_bridge_max_m: float = 3.0
    heading_tolerance_deg: float = 45.0
    size_delta_max: float = 0.35
    # Physical continuity after ID inheritance.
    reverse_step_gate: float = 0.30
    reverse_cosine: float = -0.25
    acceleration_floor: float = 4.0
    acceleration_per_sec: float = 12.0
    acceleration_speed_factor: float = 0.35
    # New IDs start above all existing IDs.
    new_id_base: Optional[int] = None


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _finite_box(det: Mapping[str, Any]) -> bool:
    return tracking.finite_box(dict(det))


def collect_world_tracks(
        frames: Sequence[Mapping[str, Any]],
        coords: tracking.CoordinateProvider,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[Tuple[int, int],
                                                    Dict[str, Any]]]:
    """Collect world-frame observations grouped by the existing track id."""
    tracks: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for frame_index, frame in enumerate(frames):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            track_id = det.get("track_id")
            if track_id is None or not _finite_box(det):
                continue
            box = det["box_lidar"]
            item = {
                "timestamp": float(timestamp) / 1e9,
                "timestamp_ns": timestamp,
                "frame_index": frame_index,
                "detection_index": detection_index,
                "det": det,
                "track_id": int(track_id),
                "world": tracking.center_world(box, world_from_lidar)[:2],
                "yaw": tracking.yaw_world(float(box[6]), world_from_lidar),
                "size": np.asarray(box[3:6], dtype=np.float64),
                "class_name": str(det.get("class_name", "")),
            }
            tracks[int(track_id)].append(item)
            by_key[(frame_index, detection_index)] = item
    for items in tracks.values():
        items.sort(key=lambda value: value["timestamp"])
    return dict(tracks), by_key


def track_motion_stats(
        items: Sequence[Mapping[str, Any]],
) -> Optional[Dict[str, float]]:
    """Net displacement / span / speed statistics used by the 19.x rules."""
    ordered = sorted(items, key=lambda item: item["timestamp"])
    if len(ordered) < 2:
        return None
    centers = np.asarray([item["world"] for item in ordered], dtype=np.float64)
    median = np.median(centers, axis=0)
    radii = np.linalg.norm(centers - median, axis=1)
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    times = np.asarray([item["timestamp"] for item in ordered],
                       dtype=np.float64)
    intervals = np.diff(times)
    valid = intervals > 1e-3
    speeds = (steps[valid] / intervals[valid]
              if np.any(valid) else np.zeros(0, dtype=np.float64))
    path = float(np.sum(steps))
    net = float(np.linalg.norm(centers[-1] - centers[0]))
    duration = float(times[-1] - times[0])
    return {
        "observations": len(ordered),
        "duration": round(duration, 4),
        "path": round(path, 4),
        "net": round(net, 4),
        "concentration": round(net / max(path, 1e-9), 4),
        "median_span": round(float(np.median(radii)), 4),
        "max_span": round(float(np.max(radii)), 4),
        "p90_span": round(float(np.percentile(radii, 90.0)), 4),
        "max_step": round(float(np.max(steps)) if len(steps) else 0.0, 4),
        "p90_speed": round(float(np.percentile(speeds, 90.0))
                           if len(speeds) else 0.0, 4),
        "median_speed": round(float(np.median(speeds))
                              if len(speeds) else 0.0, 4),
    }


def is_moving_seed(stats: Mapping[str, Any], config: Step45Config) -> bool:
    """Low-speed-but-clearly-moving seed (PLAN 19.1)."""
    return (
        float(stats["net"]) >= float(config.moving_seed_net_min_m)
        and float(stats["concentration"])
        >= float(config.moving_seed_concentration_min)
        and float(stats["duration"]) >= float(config.moving_seed_duration_min_sec)
        and float(stats["p90_speed"])
        >= float(config.moving_seed_low_speed_floor_mps)
    )


def is_weak_moving_seed(
        stats: Mapping[str, Any],
        config: Step45Config,
) -> bool:
    """Short but clear start / pull-away fragment (PLAN 19 edge case)."""
    return (
        float(stats["net"]) >= float(config.weak_seed_net_min_m)
        and float(stats["concentration"])
        >= float(config.weak_seed_concentration_min)
        and float(stats["duration"]) >= float(config.weak_seed_duration_min_sec)
        and float(stats["p90_speed"])
        >= float(config.weak_seed_low_speed_floor_mps)
    )


def is_pure_static(stats: Mapping[str, Any], config: Step45Config) -> bool:
    """Pure parking / stationary jitter (PLAN 19.2)."""
    return (
        float(stats["net"]) < float(config.static_net_max_m)
        and float(stats["max_span"]) < float(config.static_span_max_m)
        and float(stats["max_step"]) < float(config.static_step_max_m)
    )


def seed_track_ids(
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        region_config: DynamicRegionConfig,
        config: Step45Config,
) -> Dict[int, Dict[str, Any]]:
    """Strong (high-speed) + moving seeds used for region / re-tracking."""
    seeds: Dict[int, Dict[str, Any]] = {}
    for track_id, items in tracks.items():
        stats = track_motion_stats(items)
        if stats is None:
            continue
        high_speed_stats = _track_speed_stats(items, region_config)
        strong = (high_speed_stats is not None
                  and _is_high_speed(high_speed_stats, region_config))
        if (strong or is_moving_seed(stats, config)
                or is_weak_moving_seed(stats, config)):
            seeds[int(track_id)] = stats
    return seeds


def _step_headings(
        items: Sequence[Mapping[str, Any]],
) -> Tuple[List[float], List[float]]:
    """Return valid step headings (rad) and step lengths for a track."""
    ordered = sorted(items, key=lambda item: item["timestamp"])
    headings: List[float] = []
    lengths: List[float] = []
    for left, right in zip(ordered, ordered[1:]):
        vector = np.asarray(right["world"], dtype=np.float64) \
            - np.asarray(left["world"], dtype=np.float64)
        length = float(np.linalg.norm(vector))
        if length <= 1e-6:
            continue
        headings.append(math.atan2(float(vector[1]), float(vector[0])))
        lengths.append(length)
    return headings, lengths


def _circular_median_heading(headings: Sequence[float]) -> Optional[float]:
    if not headings:
        return None
    values = np.asarray(headings, dtype=np.float64)
    mean = math.atan2(float(np.mean(np.sin(values))),
                      float(np.mean(np.cos(values))))
    inliers = [float(value) for value in values
               if abs(_wrap_angle(float(value) - mean)) <= math.radians(60.0)]
    if len(inliers) < 2:
        return None
    values = np.asarray(inliers, dtype=np.float64)
    return math.atan2(float(np.mean(np.sin(values))),
                      float(np.mean(np.cos(values))))


def _local_headings(
        items: Sequence[Mapping[str, Any]],
        half_window: int = 2,
) -> List[Optional[float]]:
    """Local driving heading from each detection's own temporal neighbours."""
    ordered = sorted(items, key=lambda item: item["timestamp"])
    if len(ordered) < 2:
        return [None] * len(ordered)
    headings = [None] * len(ordered)
    for index in range(len(ordered)):
        lo = max(0, index - half_window)
        hi = min(len(ordered), index + half_window + 1)
        values: List[float] = []
        lengths: List[float] = []
        # Deliberately exclude the current detection itself: the detection
        # being tested may be the noisy one, so its own position must not
        # define the reference heading.
        if index - lo >= 2:
            side_values, side_lengths = _step_headings(ordered[lo:index])
            values.extend(side_values)
            lengths.extend(side_lengths)
        if hi - (index + 1) >= 2:
            side_values, side_lengths = _step_headings(ordered[index + 1:hi])
            values.extend(side_values)
            lengths.extend(side_lengths)
        if len(values) < 2:
            continue
        median_length = float(np.median(lengths)) if lengths else 0.0
        kept = [value for value, length in zip(values, lengths)
                if length <= max(3.0 * median_length, 1.5)]
        headings[index] = _circular_median_heading(kept)
    return headings


def _robust_track_heading(
        items: Sequence[Mapping[str, Any]],
) -> Optional[float]:
    headings, lengths = _step_headings(items)
    if len(headings) < 2:
        return None
    kept = [value for value, length in zip(headings, lengths)
            if length <= max(3.0 * float(np.median(lengths)), 1.5)]
    return _circular_median_heading(kept)


def direction_filter(
        frames: Sequence[Mapping[str, Any]],
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        dynamic_ids: set[int],
        config: Step45Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Pass-1 driving-direction filter (PLAN 19 edge case).

    A detection whose box yaw deviates more than 60 degrees (modulo pi) from
    its own local driving heading is direction noise.  Only dynamic candidate
    tracks are filtered; pure static tracks are left untouched.
    """
    threshold = math.radians(60.0)
    noisy: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for track_id, items in tracks.items():
        if int(track_id) not in dynamic_ids:
            continue
        ordered = sorted(items, key=lambda item: item["timestamp"])
        local = _local_headings(ordered)
        fallback = _robust_track_heading(ordered)
        for item, heading in zip(ordered, local):
            chosen = heading if heading is not None else fallback
            if chosen is None:
                continue
            deviation = tracking.angle_distance(
                float(item["yaw"]), float(chosen), modulo_pi=True)
            if deviation <= threshold:
                continue
            noisy[(int(item["frame_index"]), int(item["detection_index"]))] = {
                "track_id": int(track_id),
                "frame_index": int(item["frame_index"]),
                "detection_index": int(item["detection_index"]),
                "box_yaw_deg": round(math.degrees(float(item["yaw"])), 3),
                "driving_heading_deg": round(math.degrees(float(chosen)), 3),
                "deviation_deg": round(math.degrees(float(deviation)), 3),
                "local_heading_used": heading is not None,
            }
    output: List[Dict[str, Any]] = []
    removed = 0
    for frame_index, frame in enumerate(frames):
        kept = []
        for detection_index, det in enumerate(frame.get("detections", [])):
            if (frame_index, detection_index) in noisy:
                removed += 1
                continue
            kept.append(copy.deepcopy(det))
        new_frame = copy.deepcopy(frame)
        new_frame["detections"] = kept
        new_frame["num_detections"] = len(kept)
        output.append(new_frame)
    details = list(noisy.values())
    return output, details, {
        "direction_threshold_deg": 60.0,
        "dynamic_tracks_checked": sum(
            1 for track_id in tracks if int(track_id) in dynamic_ids),
        "noise_detections_removed": removed,
        "noise_details": details[:200],
    }


def _load_lidar_xyz(clip: Path, frame_id: str) -> Optional[np.ndarray]:
    path = Path(clip) / "lidar" / "lidar_top" / f"{frame_id}.bin"
    if not path.is_file():
        return None
    values = np.fromfile(path, dtype=np.float32)
    if values.size % 4 != 0:
        return None
    return values.reshape(-1, 4)[:, :3]


def single_frame_overlap_filter(
        frames: Sequence[Mapping[str, Any]],
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        clip: Path,
        config: Step45Config,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Pass-1 single-frame overlap noise filter (reviewed).

    Same-frame Car boxes with BEV IoU > threshold are compared by the number
    of lidar points inside each box.  The detection with fewer points is
    removed for that frame only; equal point counts leave both untouched.
    Pure parked tracks are excluded.
    """
    stats_by_id: Dict[int, Optional[Dict[str, float]]] = {}
    eligible: set[int] = set()
    for track_id, items in tracks.items():
        stats = track_motion_stats(items)
        stats_by_id[int(track_id)] = stats
        if stats is not None and not is_pure_static(stats, config):
            eligible.add(int(track_id))

    removed: set[Tuple[int, int]] = set()
    details: List[Dict[str, Any]] = []
    for frame_index, frame in enumerate(frames):
        detections = frame.get("detections", [])
        indices = [
            index for index, det in enumerate(detections)
            if det.get("track_id") is not None
            and int(det["track_id"]) in eligible
        ]
        if len(indices) < 2:
            continue
        points = None
        for left in range(len(indices)):
            for right in range(left + 1, len(indices)):
                i, j = indices[left], indices[right]
                if (frame_index, i) in removed or (frame_index, j) in removed:
                    continue
                box_i = detections[i].get("box_lidar")
                box_j = detections[j].get("box_lidar")
                if not (isinstance(box_i, list) and len(box_i) >= 7
                        and isinstance(box_j, list) and len(box_j) >= 7):
                    continue
                iou = tracking.bev_iou(
                    box_i[:2], np.asarray(box_i[3:6], dtype=np.float64),
                    float(box_i[6]),
                    box_j[:2], np.asarray(box_j[3:6], dtype=np.float64),
                    float(box_j[6]))
                if iou <= float(config.overlap_iou_threshold):
                    continue
                if points is None:
                    points = _load_lidar_xyz(
                        Path(clip), str(frame["frame_id"]))
                if points is None:
                    continue
                counts = count_points_in_boxes(points, [box_i, box_j])
                if counts[0] < counts[1]:
                    removed.add((frame_index, i))
                elif counts[1] < counts[0]:
                    removed.add((frame_index, j))
                else:
                    continue
                details.append({
                    "frame_index": frame_index,
                    "frame_id": str(frame["frame_id"]),
                    "kept_track_id": int(
                        detections[j if counts[0] < counts[1] else i][
                            "track_id"]),
                    "removed_track_id": int(
                        detections[i if counts[0] < counts[1] else j][
                            "track_id"]),
                    "iou": round(float(iou), 4),
                    "points_kept": int(max(counts)),
                    "points_removed": int(min(counts)),
                })

    output: List[Dict[str, Any]] = []
    removed_count = 0
    for frame_index, frame in enumerate(frames):
        kept = []
        for detection_index, det in enumerate(frame.get("detections", [])):
            if (frame_index, detection_index) in removed:
                removed_count += 1
                continue
            kept.append(copy.deepcopy(det))
        new_frame = copy.deepcopy(frame)
        new_frame["detections"] = kept
        new_frame["num_detections"] = len(kept)
        output.append(new_frame)
    return output, details, {
        "enabled": True,
        "iou_threshold": float(config.overlap_iou_threshold),
        "eligible_tracks": len(eligible),
        "noise_detections_removed": removed_count,
        "removed_details": details[:200],
    }


def candidate_track_ids(
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        config: DynamicRegionConfig,
) -> Dict[int, Dict[str, Any]]:
    """Return high-speed-evidence tracks (dynamic region candidates)."""
    candidates: Dict[int, Dict[str, Any]] = {}
    for track_id, items in tracks.items():
        stats = _track_speed_stats(items, config)
        if stats is None:
            continue
        if _is_high_speed(stats, config):
            candidates[int(track_id)] = stats
    return candidates


def build_region(
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        static_slots: Sequence[Mapping[str, Any]],
        config: DynamicRegionConfig,
        accepted_track_ids: Optional[set[int]] = None,
) -> DynamicRegionResult:
    """Build the reviewed buffer-free dynamic region."""
    reference_points = np.asarray([
        [float(slot["center"][0]), float(slot["center"][1])]
        for slot in static_slots if slot.get("center") is not None
    ], dtype=np.float64).reshape(-1, 2)
    return build_dynamic_regions(
        tracks, static_slots=static_slots,
        reference_points=reference_points,
        accepted_track_ids=accepted_track_ids,
        config=config)


def region_mask(result: DynamicRegionResult,
                config: DynamicRegionConfig) -> DynamicRegionMask:
    return DynamicRegionMask.from_polygons(
        [item["polygon"] for item in result.dynamic_polygons],
        resolution=float(config.resolution))


def select_retrackable(
        frames: Sequence[Mapping[str, Any]],
        coords: tracking.CoordinateProvider,
        mask: DynamicRegionMask,
        candidates: Mapping[int, Mapping[str, Any]],
) -> Tuple[set[Tuple[int, int]], Dict[str, Any]]:
    """Detections eligible for step-4.5 re-tracking.

    A detection is retrackable only when its old track has high-speed
    evidence and its world centre lies inside the dynamic region.  Parked
    cars inside the region are therefore frozen, and a slow fragment of a
    high-speed car is linked later by ID inheritance.
    """
    keys: set[Tuple[int, int]] = set()
    frozen_tracks: set[int] = set()
    inside_region = 0
    for frame_index, frame in enumerate(frames):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            track_id = det.get("track_id")
            if track_id is None or not _finite_box(det):
                continue
            track_id = int(track_id)
            if track_id not in candidates:
                frozen_tracks.add(track_id)
                continue
            center = tracking.center_world(det["box_lidar"], world_from_lidar)
            if mask.contains_point(float(center[0]), float(center[1])):
                keys.add((frame_index, detection_index))
                inside_region += 1
            else:
                frozen_tracks.add(track_id)
    return keys, {
        "retrackable_detections": len(keys),
        "candidate_tracks": len(candidates),
        "frozen_tracks": len(frozen_tracks),
        "inside_region_detections": inside_region,
    }


def retrack_dynamic(
        frames: Sequence[Mapping[str, Any]],
        coords: tracking.CoordinateProvider,
        retrackable: set[Tuple[int, int]],
        config: Step45Config,
) -> Dict[str, Any]:
    """Run motion-only association on the retrackable detections.

    ``_step45_new_id`` and ``_step45_retracked`` are written onto the source
    detections (the step-4 JSON dicts), never onto frozen detections.
    """
    track_frames: List[Dict[str, Any]] = []
    source_detections: List[List[Dict[str, Any]]] = []
    for frame_index, frame in enumerate(frames):
        detections: List[Dict[str, Any]] = []
        sources: List[Dict[str, Any]] = []
        for detection_index, det in enumerate(frame.get("detections", [])):
            if (frame_index, detection_index) not in retrackable:
                continue
            copied = copy.deepcopy(det)
            copied.pop("track_id", None)
            detections.append(copied)
            sources.append(det)
        track_frames.append({
            "frame_id": frame["frame_id"],
            "num_points": frame.get("num_points", 0),
            "num_detections": len(detections),
            "detections": detections,
        })
        source_detections.append(sources)

    tracker = tracking.ConservativeTracker(
        coords,
        min_static_hits=10 ** 9,
        dynamic_max_gap=float(config.dynamic_max_gap_sec),
        use_yaw=False,
        occlusion_enabled=True,
        occlusion_max_gap=float(config.occlusion_max_gap_sec),
    )
    # Occlusion gaps during motion are explicitly out of scope and two cars
    # must not be merged.  The tracker's 3 s tracklet stitching is therefore
    # disabled; ID inheritance below still merges fragments that step 2 gave
    # the same old id, and phase stitching handles red-light / yielding gaps.
    tracked, diagnostics = tracker.process(
        track_frames, enable_stitching=False)

    assigned = 0
    for out_frame, sources in zip(tracked, source_detections):
        if len(out_frame.get("detections", [])) != len(sources):
            raise AssertionError("dynamic re-tracking changed detection count")
        for out_det, source in zip(out_frame.get("detections", []), sources):
            new_id = out_det.get("track_id")
            if new_id is None:
                continue
            source["_step45_new_id"] = int(new_id)
            source["_step45_retracked"] = True
            assigned += 1
    diagnostics.update({
        "mode": "step45_motion_only_retrack",
        "use_yaw": False,
        "assigned_detections": assigned,
    })
    return diagnostics


def _slot_track_ids(step2_diagnostics: Mapping[str, Any]) -> set[int]:
    return {
        int(item.get("track_id", item.get("slot_id", 0)))
        for item in step2_diagnostics.get("tracking", {}).get(
            "slot_details", [])
    }


def _slot_events(
        step2_diagnostics: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    coordination = step2_diagnostics.get("tracking", {}).get(
        "slot_motion_coordination", {})
    return (
        list(coordination.get("departures", [])),
        list(coordination.get("arrivals", coordination.get("ingresses", []))),
        list(coordination.get("stop_binds", [])),
    )


def _endpoint_speed(
        items: Sequence[Mapping[str, Any]],
        *,
        at_end: bool,
        count: int = 3,
) -> float:
    selected = list(items[-count:] if at_end else items[:count])
    if len(selected) < 2:
        return 0.0
    selected.sort(key=lambda item: item["timestamp"])
    duration = float(selected[-1]["timestamp"] - selected[0]["timestamp"])
    if duration <= 1e-3:
        return 0.0
    path = sum(
        float(np.linalg.norm(
            np.asarray(selected[index + 1]["world"], dtype=np.float64)
            - np.asarray(selected[index]["world"], dtype=np.float64)))
        for index in range(len(selected) - 1))
    return path / duration


def _boundary_anchor(
        group: Sequence[Mapping[str, Any]],
        frozen_tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        slot_ids: set[int],
        departures: Sequence[Mapping[str, Any]],
        arrivals: Sequence[Mapping[str, Any]],
        stop_binds: Sequence[Mapping[str, Any]],
        config: Step45Config,
) -> Optional[Tuple[int, str, Dict[str, Any]]]:
    """Find a frozen track that should own this dynamic fragment.

    Static slot anchors require an explicit step-2 departure / arrival /
    stop-bind event, which is the reviewed slot-release guard.  A non-slot
    frozen track is only used when the fragment's old IDs already contain it.
    """
    if not group:
        return None
    ordered = sorted(group, key=lambda item: item["timestamp"])
    first, last = ordered[0], ordered[-1]
    old_ids = {int(item["track_id"]) for item in ordered}
    best: Optional[Tuple[float, int, str, Dict[str, Any]]] = None

    def physical_evidence(
            track_items: Sequence[Mapping[str, Any]],
            *,
            at_end: bool,
            boundary_item: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        endpoint = track_items[-1] if at_end else track_items[0]
        heading = tracking.angle_distance(
            float(endpoint["yaw"]), float(boundary_item["yaw"]),
            modulo_pi=True)
        size_delta = float(np.linalg.norm(
            np.asarray(endpoint["size"], dtype=np.float64)
            - np.asarray(boundary_item["size"], dtype=np.float64))) / max(
                float(np.linalg.norm(boundary_item["size"])), 1.0)
        speed = _endpoint_speed(track_items, at_end=at_end)
        if (heading <= math.radians(config.boundary_max_heading_deg)
                and size_delta <= config.boundary_max_size_delta
                and speed <= config.boundary_max_speed_mps):
            return {
                "reason": "physical_continuity",
                "endpoint_speed_mps": round(speed, 4),
                "heading_deg": round(math.degrees(heading), 3),
                "size_delta": round(size_delta, 4),
            }
        return None

    for track_id, items in frozen_tracks.items():
        track_id = int(track_id)
        if not items:
            continue
        track_items = sorted(items, key=lambda item: item["timestamp"])
        track_first, track_last = track_items[0], track_items[-1]
        # Departure: frozen track ends, dynamic fragment starts.
        gap = first["timestamp"] - track_last["timestamp"]
        if 0.0 <= gap <= config.boundary_max_gap_sec:
            distance = float(np.linalg.norm(
                np.asarray(first["world"], dtype=np.float64)
                - np.asarray(track_last["world"], dtype=np.float64)))
            if distance <= config.boundary_max_distance_m:
                evidence = None
                if track_id in slot_ids:
                    evidence = next((
                        item for item in departures
                        if int(item.get("inherited_track_id", -1)) == track_id
                        and float(item.get("start_timestamp", 0)) / 1e9
                        <= first["timestamp"] + config.slot_release_tolerance_sec
                        and (old_ids == {track_id}
                             or int(item.get("dynamic_track_id", -1)) in old_ids)
                    ), None)
                    if evidence is None and old_ids == {track_id}:
                        evidence = {"reason": "dynamic_fragment_keeps_slot_id"}
                    if evidence is None:
                        # The slot track ended immediately before this
                        # fragment and the bridge is physically continuous:
                        # the old car has left even if step 2 did not emit an
                        # explicit departure event.  This is the reviewed
                        # static->dynamic handoff.
                        physical = physical_evidence(
                            track_items, at_end=True, boundary_item=first)
                        if physical is not None:
                            evidence = {
                                "reason": (
                                    "physical_departure_without_explicit_event"),
                                **physical,
                            }
                else:
                    evidence = physical_evidence(
                        track_items, at_end=True, boundary_item=first)
                    if evidence is None and track_id in old_ids:
                        evidence = {"reason": "frozen_track_same_old_id"}
                if evidence is not None:
                    score = gap + distance
                    if best is None or score < best[0]:
                        best = (score, track_id, "departure", evidence)
        # Arrival: dynamic fragment ends, frozen track starts.
        gap = track_first["timestamp"] - last["timestamp"]
        if 0.0 <= gap <= config.boundary_max_gap_sec:
            distance = float(np.linalg.norm(
                np.asarray(last["world"], dtype=np.float64)
                - np.asarray(track_first["world"], dtype=np.float64)))
            if distance <= config.boundary_max_distance_m:
                evidence = None
                if track_id in slot_ids:
                    evidence = next((
                        item for item in arrivals
                        if int(item.get("bound_track_id", -1)) == track_id
                        and int(item.get("dynamic_track_id", -1)) in old_ids
                    ), None)
                    if evidence is None:
                        evidence = next((
                            item for item in stop_binds
                            if int(item.get("bound_track_id", -1)) == track_id
                            and int(item.get("dynamic_track_id", -1)) in old_ids
                        ), None)
                    if evidence is None and old_ids == {track_id}:
                        evidence = {"reason": "dynamic_fragment_keeps_slot_id"}
                    if evidence is None:
                        physical = physical_evidence(
                            track_items, at_end=False, boundary_item=last)
                        if physical is not None:
                            evidence = {
                                "reason": (
                                    "physical_arrival_without_explicit_event"),
                                **physical,
                            }
                else:
                    evidence = physical_evidence(
                        track_items, at_end=False, boundary_item=last)
                    if evidence is None and track_id in old_ids:
                        evidence = {"reason": "frozen_track_same_old_id"}
                if evidence is not None:
                    score = gap + distance
                    if best is None or score < best[0]:
                        best = (score, track_id, "arrival", evidence)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _physical_split(
        observations: Sequence[Mapping[str, Any]],
        config: Step45Config,
) -> Optional[int]:
    """Return the first index that violates physical continuity, if any."""
    ordered = sorted(observations, key=lambda item: item["timestamp"])
    if len(ordered) < 3:
        return None
    for index in range(1, len(ordered) - 1):
        prior = ordered[index - 1]
        current = ordered[index]
        candidate = ordered[index + 1]
        dt = float(candidate["timestamp"] - current["timestamp"])
        if dt <= 1e-3:
            continue
        direction = np.asarray(current["world"], dtype=np.float64) \
            - np.asarray(prior["world"], dtype=np.float64)
        step = np.asarray(candidate["world"], dtype=np.float64) \
            - np.asarray(current["world"], dtype=np.float64)
        direction_norm = float(np.linalg.norm(direction))
        step_norm = float(np.linalg.norm(step))
        if direction_norm >= config.reverse_step_gate and step_norm >= config.reverse_step_gate:
            cosine = float(np.dot(direction, step)) / max(
                direction_norm * step_norm, 1e-9)
            if cosine < config.reverse_cosine:
                return index + 1
        prior_dt = float(current["timestamp"] - prior["timestamp"])
        if prior_dt > 1e-3:
            prior_speed = float(np.linalg.norm(direction)) / prior_dt
            candidate_speed = step_norm / dt
            allowed = max(
                config.acceleration_floor,
                config.acceleration_per_sec * dt
                + config.acceleration_speed_factor * prior_speed)
            if candidate_speed - prior_speed > allowed:
                return index + 1
    return None


def _continuity_ok(
        existing: Sequence[Mapping[str, Any]],
        group: Sequence[Mapping[str, Any]],
        config: Step45Config,
) -> bool:
    """Whether a fragment can be appended to an existing final id."""
    if not existing or not group:
        return True
    # Only the closest boundary pair matters: a departing car is far away
    # from the parking spot later in the fragment, but its first observation
    # must connect to the frozen track's last observation.
    nearest_pair = min(
        ((value, item) for value in existing for item in group),
        key=lambda pair: abs(
            float(pair[0]["timestamp"]) - float(pair[1]["timestamp"])))
    nearest, item = nearest_pair
    gap = abs(float(nearest["timestamp"]) - float(item["timestamp"]))
    distance = float(np.linalg.norm(
        np.asarray(nearest["world"], dtype=np.float64)
        - np.asarray(item["world"], dtype=np.float64)))
    gate = min(
        float(config.merge_gate_max_m),
        float(config.merge_gate_base_m)
        + float(config.merge_gate_per_sec_m) * gap)
    if distance > gate:
        return False
    size_delta = float(np.linalg.norm(
        np.asarray(nearest["size"], dtype=np.float64)
        - np.asarray(item["size"], dtype=np.float64))) / max(
            float(np.linalg.norm(nearest["size"])), 1.0)
    return size_delta <= 0.50


def inherit_ids(
        frames: List[Dict[str, Any]],
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        retrackable: set[Tuple[int, int]],
        candidates: Mapping[int, Mapping[str, Any]],
        step2_diagnostics: Mapping[str, Any],
        config: Step45Config,
) -> Dict[str, Any]:
    """Assign conservative final IDs to the motion-only fragments.

    Frozen detections are never changed.  A dynamic fragment tries, in order:
    an explicit static boundary anchor, its majority old high-speed ID, and
    finally a brand-new ID.  A candidate is rejected if it would collide with
    another track in the same frame.
    """
    slot_ids = _slot_track_ids(step2_diagnostics)
    departures, arrivals, stop_binds = _slot_events(step2_diagnostics)

    # Build frozen tracks and current per-frame id occupancy.
    frozen_tracks: Dict[int, List[Dict[str, Any]]] = {}
    used_by_frame: Dict[int, set[int]] = defaultdict(set)
    for frame_index, frame in enumerate(frames):
        for det in frame.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None or det.get("_step45_retracked"):
                continue
            used_by_frame[frame_index].add(int(track_id))
    for track_id, items in tracks.items():
        kept = [dict(item) for item in items
                if not item["det"].get("_step45_retracked")]
        if kept:
            kept.sort(key=lambda item: item["timestamp"])
            frozen_tracks[int(track_id)] = kept

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for track_id, items in tracks.items():
        for item in items:
            if (item["frame_index"], item["detection_index"]) in retrackable:
                new_id = item["det"].get("_step45_new_id")
                if new_id is not None:
                    groups[int(new_id)].append(item)

    next_id = int(config.new_id_base or 0)
    if next_id <= 0:
        next_id = 1 + max(
            [int(track_id) for track_id in tracks]
            + [int(det.get("track_id", 0))
               for frame in frames for det in frame.get("detections", [])
               if det.get("track_id") is not None] + [0])
    assigned_items: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for frozen_id, items in frozen_tracks.items():
        assigned_items[int(frozen_id)].extend(items)
    assignments: List[Dict[str, Any]] = []
    for new_id, items in sorted(groups.items(), key=lambda pair: -len(pair[1])):
        old_counts = Counter(int(item["track_id"]) for item in items)
        ordered_old = sorted(
            old_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        candidates_list: List[Tuple[Optional[int], str, Dict[str, Any]]] = []
        anchor = _boundary_anchor(
            items, frozen_tracks, slot_ids, departures, arrivals, stop_binds,
            config)
        if anchor is not None:
            candidates_list.append((anchor[0], f"static_{anchor[1]}",
                                    dict(anchor[2])))
        for old_id, count in ordered_old:
            if old_id not in slot_ids or old_id in {item["track_id"] for item in items}:
                candidates_list.append((old_id, "majority_old_id",
                                        {"votes": count}))

        chosen: Optional[int] = None
        chosen_reason = "new_id"
        chosen_evidence: Dict[str, Any] = {}
        for candidate_id, reason, evidence in candidates_list:
            if candidate_id is None:
                chosen = next_id
                next_id += 1
                chosen_reason = reason
                chosen_evidence = evidence
                break
            if any(candidate_id in used_by_frame[item["frame_index"]]
                   for item in items):
                continue
            # A slot id must not be reused across cars unless the explicit
            # boundary evidence above linked this fragment to it.  When every
            # detection in the fragment already carries that slot id, step 2
            # itself bound them to the same car, so keeping it is safe.
            if candidate_id in slot_ids and not reason.startswith("static_"):
                if set(old_counts) != {int(candidate_id)}:
                    continue
            if (candidate_id in assigned_items
                    and not _continuity_ok(
                        assigned_items[candidate_id], items, config)):
                continue
            chosen = int(candidate_id)
            chosen_reason = reason
            chosen_evidence = evidence
            break
        if chosen is None:
            # Never split an old id just because the continuity gate failed.
            # Fall back to the strongest old id that is safe to reuse.  A
            # slot id is only safe when every detection already carries it
            # (step 2 itself bound them to one car).
            fallback = [
                old_id for old_id, _count in ordered_old
                if old_id not in slot_ids or set(old_counts) == {old_id}
            ]
            if fallback:
                chosen = int(fallback[0])
                chosen_reason = "fallback_old_id"
                chosen_evidence = {
                    "reason": "continuity_failed_but_old_id_kept",
                    "votes": old_counts[chosen],
                }
            else:
                chosen = next_id
                next_id += 1
                chosen_reason = "new_id"
                chosen_evidence = {}
        for item in items:
            item["det"]["track_id"] = int(chosen)
            used_by_frame[item["frame_index"]].add(int(chosen))
        assigned_items[int(chosen)].extend(items)
        assignments.append({
            "step45_new_id": int(new_id),
            "final_id": int(chosen),
            "reason": chosen_reason,
            "old_id_votes": {str(key): value for key, value in ordered_old},
            "evidence": chosen_evidence,
            "detections": len(items),
        })

    final_groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for track_id, items in tracks.items():
        for item in items:
            final_id = item["det"].get("track_id")
            if final_id is not None:
                final_groups[int(final_id)].append(item)
    # Physical violations are reported, never repaired by splitting a track.
    # The reviewed decision is that one old id should not be split into two;
    # merge-side continuity checks above prevent creating new impossible jumps.
    physical_violations: List[Dict[str, Any]] = []
    for final_id, items in final_groups.items():
        if not any(item["det"].get("_step45_retracked") for item in items):
            continue
        violation = _physical_split(items, config)
        if violation is None:
            continue
        ordered = sorted(items, key=lambda item: item["timestamp"])
        physical_violations.append({
            "track_id": int(final_id),
            "at_timestamp": ordered[violation]["timestamp"],
            "detections_after": len(ordered) - violation,
        })

    all_ids = {
        int(det["track_id"])
        for frame in frames for det in frame.get("detections", [])
        if det.get("track_id") is not None
    }
    return {
        "assignments": assignments,
        "splits": [],
        "physical_violations": physical_violations,
        "next_id": int(next_id),
        "tracks_total": len(all_ids),
    }


def _direction_assignments(
        by_final_id: Mapping[int, Sequence[Mapping[str, Any]]],
        config: Step45Config,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    """Direction/movement context plus geometric fallback for static tracks."""
    model_tracks = {
        final_id: [
            {
                "timestamp": float(item["timestamp"]),
                "world": np.asarray(item["world"], dtype=np.float64),
                "yaw": float(item["yaw"]),
                "size": np.asarray(item["size"], dtype=np.float64),
                "class_name": str(item.get("class_name", "Car")),
            }
            for item in sorted(items, key=lambda value: value["timestamp"])
        ]
        for final_id, items in by_final_id.items() if len(items) >= 2
    }
    result = build_traffic_light_model(
        model_tracks, config=config.traffic_light)
    directions = list(
        result.diagnostics.get("direction_phase", {}).get("directions", []))
    classification = {
        int(item["track_id"]): item for item in result.track_classification
        if item.get("track_id") is not None
    }
    direction_points: Dict[int, List[np.ndarray]] = defaultdict(list)
    for track_id, item in classification.items():
        items = by_final_id.get(int(track_id), [])
        if not items:
            continue
        center = np.median(
            np.asarray([value["world"] for value in items], dtype=np.float64),
            axis=0)
        direction_points[int(item.get("direction_id", -1))].append(center)

    assignments: Dict[int, Dict[str, Any]] = {}
    for final_id, items in by_final_id.items():
        final_id = int(final_id)
        info = classification.get(final_id)
        if (info is not None and info.get("movement")
                and info.get("direction_id") is not None):
            assignments[final_id] = {
                "direction_id": int(info["direction_id"]),
                "movement": str(info["movement"]),
                "heading_deg": info.get("stable_heading_deg"),
            }
            continue
        center = np.median(
            np.asarray([value["world"] for value in items], dtype=np.float64),
            axis=0)
        best: Optional[Tuple[float, int]] = None
        for direction in directions:
            points = direction_points.get(int(direction["direction_id"]))
            if not points:
                continue
            distance = min(float(np.linalg.norm(center - point))
                           for point in points)
            if best is None or distance < best[0]:
                best = (distance, int(direction["direction_id"]))
        assignments[final_id] = {
            "direction_id": None if best is None else best[1],
            "movement": None,
        }
    return directions, assignments


def _movement_compatible(
        movement_a: Optional[str],
        movement_b: Optional[str],
        v_a: float,
        v_b: float,
        config: Step45Config,
) -> bool:
    """PLAN 19.6 movement matrix / lane-change / right-turn gate."""
    if not movement_a or not movement_b or movement_a == movement_b:
        if movement_a == "right" or movement_b == "right":
            # Right turn is only allowed on the right side of the other lane.
            return abs(v_a - v_b) <= float(config.lane_change_max_lateral_m)
        return True
    movement_pair = {movement_a, movement_b}
    lateral = abs(v_a - v_b)
    if movement_pair == {"left", "straight"}:
        if lateral > float(config.lane_change_max_lateral_m):
            return False
        left_v = v_a if movement_a == "left" else v_b
        straight_v = v_b if movement_a == "left" else v_a
        return left_v <= straight_v
    if movement_pair == {"right", "straight"}:
        if lateral > float(config.lane_change_max_lateral_m):
            return False
        right_v = v_a if movement_a == "right" else v_b
        straight_v = v_b if movement_a == "right" else v_a
        return right_v >= straight_v
    if movement_pair == {"left", "right"}:
        return False
    return True


def queue_stitch(
        frames: List[Dict[str, Any]],
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        step2_diagnostics: Mapping[str, Any],
        mask: DynamicRegionMask,
        seed_ids: set[int],
        config: Step45Config,
) -> Dict[str, Any]:
    """PLAN 19.4-19.6 queue-based same-vehicle stitching.

    The three reviewed conditions are hard:

    * same direction / queue;
    * the two fragments must not overlap in time;
    * the later fragment must start at the earlier fragment's end position.
    """
    by_final: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for items in tracks.values():
        for item in items:
            final_id = item["det"].get("track_id")
            if final_id is not None:
                by_final[int(final_id)].append(item)
    for items in by_final.values():
        items.sort(key=lambda value: value["timestamp"])
    if len(by_final) < 2:
        return {"queues": 0, "edges": 0, "merges": [], "modified_keys": []}
    stats = {final_id: track_motion_stats(items)
             for final_id, items in by_final.items()}

    directions, assignments = _direction_assignments(by_final, config)
    direction_by_id = {int(item["direction_id"]): item for item in directions}

    # Build queues: cluster lateral lanes first (lane width ~3.5m), then
    # group longitudinal neighbours <=20m inside each lane.  This is the
    # reviewed median-position semantics; interval-based grouping was tested
    # and reverted because it can chain many tracks into one huge queue.
    queues: List[List[Tuple[float, float, int]]] = []
    for direction in directions:
        direction_id = int(direction["direction_id"])
        origin = np.asarray(direction["origin"], dtype=np.float64)
        forward = np.asarray(direction["forward"], dtype=np.float64)
        right = np.asarray(direction["right"], dtype=np.float64)
        rows: List[Tuple[float, float, int]] = []
        for final_id, assignment in assignments.items():
            if assignment.get("direction_id") != direction_id:
                continue
            items = by_final.get(final_id)
            if not items:
                continue
            center = np.median(
                np.asarray([value["world"] for value in items],
                           dtype=np.float64), axis=0)
            u = float(np.dot(center - origin, forward))
            v = float(np.dot(center - origin, right))
            rows.append((u, v, final_id))
        rows.sort(key=lambda row: row[1])
        lane_clusters: List[List[Tuple[float, float, int]]] = []
        for row in rows:
            if (not lane_clusters
                    or abs(row[1] - lane_clusters[-1][-1][1]) > 3.5):
                lane_clusters.append([row])
            else:
                lane_clusters[-1].append(row)
        for cluster in lane_clusters:
            cluster.sort(key=lambda row: row[0])
            current: List[Tuple[float, float, int]] = []
            for row in cluster:
                if not current:
                    current = [row]
                    continue
                previous = current[-1]
                if abs(row[0] - previous[0]) \
                        <= float(config.queue_longitudinal_gap_m):
                    current.append(row)
                else:
                    queues.append(current)
                    current = [row]
            if current:
                queues.append(current)

    parent = {final_id: final_id for final_id in by_final}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    edges: List[Dict[str, Any]] = []
    for queue in queues:
        for i in range(len(queue)):
            for j in range(i + 1, len(queue)):
                _u1, v_1, id_1 = queue[i]
                _u2, v_2, id_2 = queue[j]
                items_1 = by_final[id_1]
                items_2 = by_final[id_2]
                # Determine the earlier-end -> later-start order explicitly;
                # queue rows are sorted by longitudinal position, not time.
                if items_1[-1]["timestamp"] < items_2[0]["timestamp"]:
                    id_a, id_b = id_1, id_2
                    items_a, items_b = items_1, items_2
                    v_a, v_b = v_1, v_2
                elif items_2[-1]["timestamp"] < items_1[0]["timestamp"]:
                    id_a, id_b = id_2, id_1
                    items_a, items_b = items_2, items_1
                    v_a, v_b = v_2, v_1
                else:
                    continue
                a_end = items_a[-1]
                b_start = items_b[0]
                direction_id = assignments[id_a].get("direction_id")
                direction = (direction_by_id.get(int(direction_id))
                             if direction_id is not None else None)
                if direction is None:
                    continue
                origin = np.asarray(direction["origin"], dtype=np.float64)
                forward = np.asarray(direction["forward"], dtype=np.float64)
                u_end = float(np.dot(
                    np.asarray(a_end["world"], dtype=np.float64) - origin,
                    forward))
                u_start = float(np.dot(
                    np.asarray(b_start["world"], dtype=np.float64) - origin,
                    forward))
                if u_start + float(
                        config.queue_stitch_position_tolerance_m) < u_end:
                    continue
                bridge = float(np.linalg.norm(
                    np.asarray(b_start["world"], dtype=np.float64)
                    - np.asarray(a_end["world"], dtype=np.float64)))
                if bridge > float(config.queue_stitch_position_tolerance_m):
                    continue
                if abs(float(v_a) - float(v_b)) \
                        > float(config.queue_stitch_lateral_tolerance_m):
                    continue
                # Only stitch inside/near the dynamic region.
                end_inside = mask.contains_point(
                    float(a_end["world"][0]), float(a_end["world"][1]))
                start_inside = mask.contains_point(
                    float(b_start["world"][0]), float(b_start["world"][1]))
                if not end_inside and not start_inside:
                    continue
                movement_a = assignments[id_a].get("movement")
                movement_b = assignments[id_b].get("movement")
                if not _movement_compatible(
                        movement_a, movement_b, float(v_a), float(v_b),
                        config):
                    continue
                blocked = False
                for _u_c, _v_c, id_c in queue:
                    if id_c in (id_a, id_b):
                        continue
                    items_c = by_final[id_c]
                    c_start, c_end = items_c[0], items_c[-1]
                    if not (a_end["timestamp"] < c_start["timestamp"]
                            and c_end["timestamp"] < b_start["timestamp"]):
                        continue
                    center_c = np.median(
                        np.asarray([value["world"] for value in items_c],
                                   dtype=np.float64), axis=0)
                    u_c = float(np.dot(center_c - origin, forward))
                    if min(u_end, u_start) - 2.0 <= u_c \
                            <= max(u_end, u_start) + 2.0:
                        blocked = True
                        break
                if blocked:
                    continue
                edges.append({
                    "from": int(id_a),
                    "to": int(id_b),
                    "bridge_m": round(bridge, 4),
                    "movement_a": movement_a,
                    "movement_b": movement_b,
                    "direction_id": int(direction_id),
                })
                root_a, root_b = find(int(id_a)), find(int(id_b))
                if root_a != root_b:
                    parent[root_b] = root_a

    components: Dict[int, List[int]] = defaultdict(list)
    for final_id in by_final:
        components[find(final_id)].append(final_id)

    merges: List[Dict[str, Any]] = []
    modified_keys: List[Tuple[int, int]] = []
    occupancy: Dict[int, set[int]] = defaultdict(set)
    for final_id, items in by_final.items():
        for item in items:
            occupancy[item["frame_index"]].add(int(final_id))

    for members in components.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        # A queue component must contain at least one clear moving seed and
        # every pair of its members must be time-disjoint.
        has_seed = False
        for member in members:
            member_stats = stats.get(member)
            if member_stats is None:
                continue
            if (member in seed_ids
                    or is_moving_seed(member_stats, config)
                    or is_weak_moving_seed(member_stats, config)):
                has_seed = True
                break
        if not has_seed:
            continue
        time_conflict = False
        for i in range(len(members)):
            a_items = by_final[members[i]]
            for j in range(i + 1, len(members)):
                b_items = by_final[members[j]]
                if (a_items[0]["timestamp"] < b_items[-1]["timestamp"]
                        and b_items[0]["timestamp"] < a_items[-1]["timestamp"]):
                    time_conflict = True
                    break
            if time_conflict:
                break
        if time_conflict:
            continue
        member_set = set(members)
        counts = {member: len(by_final[member]) for member in members}
        chosen = max(members, key=lambda member: (counts[member], -member))
        collision = any(
            chosen in (occupancy[item["frame_index"]] - member_set)
            for member in members for item in by_final[member])
        if collision:
            continue
        for member in members:
            for item in by_final[member]:
                item["det"]["track_id"] = int(chosen)
                item["det"]["_step45_queue_stitched"] = True
                modified_keys.append((item["frame_index"],
                                      item["detection_index"]))
                occupancy[item["frame_index"]].discard(int(member))
                occupancy[item["frame_index"]].add(int(chosen))
        merges.append({
            "members": [int(member) for member in members],
            "final_id": int(chosen),
            "detections": sum(counts.values()),
        })
    return {
        "queues": len(queues),
        "edges": len(edges),
        "merges": merges,
        "modified_keys": modified_keys,
        "direction_assignments": assignments,
    }


def _state_intervals(
        traffic_states: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    return {
        int(item["track_id"]): item for item in traffic_states
        if item.get("track_id") is not None
    }


def _state_at(
        state: Mapping[str, Any], timestamp: float,
        keys: Sequence[str]) -> Optional[str]:
    for interval in state.get("intervals", []):
        start = float(interval.get("start_timestamp", 0.0))
        end = float(interval.get("end_timestamp", start))
        if start <= timestamp <= end + 1e-6:
            value = str(interval.get("state", "uncertain"))
            return value if value in keys else None
    return None


def phase_stitch(
        frames: List[Dict[str, Any]],
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        coords: tracking.CoordinateProvider,
        config: Step45Config,
) -> Dict[str, Any]:
    """Conservative phase-aware merge of re-tracked dynamic fragments.

    Only tracks that contain at least one step-4.5 re-tracked detection are
    candidates.  Frozen static tracks are never merged or moved.
    """
    by_final_id: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for items in tracks.values():
        for item in items:
            final_id = item["det"].get("track_id")
            if final_id is not None:
                by_final_id[int(final_id)].append(item)
    dynamic_ids = {
        final_id for final_id, items in by_final_id.items()
        if any(item["det"].get("_step45_retracked") for item in items)
    }
    if not dynamic_ids:
        return {
            "candidates": 0,
            "merges": [],
            "applied": [],
            "dynamic_tracks": 0,
        }
    # Frozen static tracks are included as phase context (red-light queues,
    # stop-line evidence).  Only dynamic ids are merge candidates.
    model_tracks = {
        final_id: [
            {
                "timestamp": float(item["timestamp"]),
                "world": np.asarray(item["world"], dtype=np.float64),
                "yaw": float(item["yaw"]),
                "size": np.asarray(item["size"], dtype=np.float64),
                "class_name": str(item.get("class_name", "Car")),
            }
            for item in sorted(items, key=lambda value: value["timestamp"])
        ]
        for final_id, items in by_final_id.items()
        if len(items) >= 2
    }
    result = build_traffic_light_model(
        model_tracks, config=config.traffic_light)
    states = _state_intervals(result.track_traffic_states)
    by_id = {final_id: sorted(items, key=lambda item: item["timestamp"])
             for final_id, items in by_final_id.items()}

    candidates: List[Tuple[float, int, int, int, int, Dict[str, Any]]] = []
    for end_id, end_items in by_id.items():
        end = end_items[-1]
        end_state = states.get(end_id, {})
        end_signal = _state_at(
            end_state, float(end["timestamp"]),
            ("waiting_red", "waiting_queue", "yielding", "uncertain"))
        end_speed = _endpoint_speed(end_items, at_end=True)
        for start_id, start_items in by_id.items():
            if start_id == end_id:
                continue
            if end_id not in dynamic_ids and start_id not in dynamic_ids:
                continue
            start = start_items[0]
            gap = float(start["timestamp"] - end["timestamp"])
            if gap <= 0.0 or gap > float(config.phase_merge_max_gap_sec):
                continue
            start_state = states.get(start_id, {})
            start_signal = _state_at(
                start_state, float(start["timestamp"]),
                ("moving", "uncertain"))
            start_speed = _endpoint_speed(start_items, at_end=False)
            stopped_end = (end_signal is not None
                           or end_speed <= 2.0)
            moving_start = (start_signal == "moving"
                            or start_speed >= 1.0)
            moving_end = (end_signal == "moving"
                          or end_speed >= 1.0)
            stopped_start = (start_signal in (
                "waiting_red", "waiting_queue", "yielding", "uncertain")
                or start_speed <= 2.0)
            departure_case = stopped_end and moving_start
            arrival_case = moving_end and stopped_start
            if not (departure_case or arrival_case):
                continue
            end_movement = str(end_state.get("movement", ""))
            start_movement = str(start_state.get("movement", ""))
            if end_movement and start_movement and end_movement != start_movement:
                continue
            movement = end_movement or start_movement
            end_direction = end_state.get("direction_id")
            start_direction = start_state.get("direction_id")
            if (end_direction is not None and start_direction is not None
                    and end_direction != start_direction):
                continue
            direction_known = (end_direction is not None
                               and start_direction is not None)
            if movement == "right" or arrival_case:
                max_gap = float(config.yielding_max_gap_sec)
            else:
                max_gap = float(config.phase_merge_max_gap_sec)
            if gap > max_gap:
                continue
            distance = float(np.linalg.norm(
                np.asarray(start["world"], dtype=np.float64)
                - np.asarray(end["world"], dtype=np.float64)))
            if gap <= 2.0:
                bridge = min(
                    float(config.bridge_max_m),
                    float(config.bridge_base_m)
                    + float(config.bridge_per_sec_m) * gap)
            else:
                # A car waiting at a red light stays within a few metres of
                # the stop line; a loose bridge would merge different queued
                # cars.
                bridge = min(float(config.phase_bridge_max_m),
                             1.0 + 0.5 * gap)
            if not direction_known:
                bridge = min(
                    bridge,
                    0.5 * float(config.bridge_base_m) + 0.5 * gap)
            if distance > bridge:
                continue
            heading_delta = abs(_wrap_angle(
                float(start["yaw"]) - float(end["yaw"])))
            heading_delta = min(heading_delta, math.pi - heading_delta)
            if heading_delta > math.radians(config.heading_tolerance_deg):
                continue
            size_delta = float(np.linalg.norm(
                np.asarray(start["size"], dtype=np.float64)
                - np.asarray(end["size"], dtype=np.float64))) / max(
                    float(np.linalg.norm(end["size"])), 1.0)
            if size_delta > config.size_delta_max:
                continue
            blocked = False
            for other_id, other_items in by_id.items():
                if other_id in (end_id, start_id):
                    continue
                for item in other_items:
                    timestamp = float(item["timestamp"])
                    if not (end["timestamp"] < timestamp < start["timestamp"]):
                        continue
                    if min(
                            float(np.linalg.norm(
                                np.asarray(item["world"], dtype=np.float64)
                                - np.asarray(end["world"], dtype=np.float64))),
                            float(np.linalg.norm(
                                np.asarray(item["world"], dtype=np.float64)
                                - np.asarray(start["world"], dtype=np.float64)))
                    ) <= bridge:
                        blocked = True
                        break
                if blocked:
                    break
            if blocked:
                continue
            if end_id in dynamic_ids and start_id in dynamic_ids:
                if len(by_id[end_id]) >= len(by_id[start_id]):
                    source_id, target_id = start_id, end_id
                else:
                    source_id, target_id = end_id, start_id
            elif end_id in dynamic_ids:
                source_id, target_id = end_id, start_id
            else:
                source_id, target_id = start_id, end_id
            # A mixed id (frozen parking frames + dynamic departure frames)
            # must never be used as a phase-merge source: remapping it would
            # mutate frozen detections.
            if any(not item["det"].get("_step45_retracked")
                   for item in by_id[source_id]):
                continue
            score = gap + 0.25 * distance + 0.5 * heading_delta
            candidates.append((score, end_id, start_id, source_id, target_id, {
                "gap_sec": round(gap, 3),
                "distance_m": round(distance, 3),
                "end_state": end_signal or "unknown",
                "start_state": start_signal or "unknown",
                "end_speed_mps": round(end_speed, 3),
                "start_speed_mps": round(start_speed, 3),
                "movement": movement,
                "case": "departure" if departure_case else "arrival",
            }))

    occupancy: Dict[int, set[int]] = defaultdict(set)
    for track_id, items in by_final_id.items():
        for item in items:
            occupancy[item["frame_index"]].add(int(track_id))
    remapped: Dict[int, int] = {}
    merges: List[Dict[str, Any]] = []
    applied: List[Dict[str, Any]] = []
    for candidate in sorted(candidates):
        score, end_id, start_id, source_id, target_id, evidence = candidate
        if source_id in remapped or target_id in remapped:
            continue
        collision = any(
            target_id in (occupancy[item["frame_index"]] - {source_id})
            for item in by_id[source_id])
        if collision:
            continue
        for item in by_id[source_id]:
            occupancy[item["frame_index"]].discard(int(source_id))
            occupancy[item["frame_index"]].add(int(target_id))
            item["det"]["track_id"] = int(target_id)
        remapped[int(source_id)] = int(target_id)
        record = {
            "from_track_id": int(source_id),
            "to_track_id": int(target_id),
            "end_track_id": int(end_id),
            "start_track_id": int(start_id),
            "score": round(float(score), 4),
            "detections": len(by_id[source_id]),
            **evidence,
        }
        merges.append(record)
        applied.append(record)
    return {
        "candidates": len(candidates),
        "merges": merges,
        "applied": applied,
        "dynamic_tracks": len(dynamic_ids),
    }


def align_dynamic_yaw(
        frames: List[Dict[str, Any]],
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        coords: tracking.CoordinateProvider,
        dynamic_ids: set[int],
        config: Step45Config,
) -> Dict[str, Any]:
    """Pass-2 dynamic yaw alignment (PLAN 19 final rule).

    For moving dynamic tracks, choose the pi-equivalent yaw that points along
    the local driving direction.  This removes the 180-degree flip ambiguity
    without touching detector yaw in the perpendicular direction.  Static
    tracks are never modified.
    """
    by_final: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for items in tracks.values():
        for item in items:
            final_id = item["det"].get("track_id")
            if final_id is not None:
                by_final[int(final_id)].append(item)
    changed = 0
    details: List[Dict[str, Any]] = []
    for final_id, items in by_final.items():
        if final_id not in dynamic_ids:
            continue
        if not any(item["det"].get("_step45_retracked")
                   or item["det"].get("region") == "dynamic"
                   for item in items):
            continue
        ordered = sorted(items, key=lambda item: item["timestamp"])
        if len(ordered) < 3:
            continue
        local = _local_headings(ordered)
        fallback = _robust_track_heading(ordered)
        for item, heading in zip(ordered, local):
            if not (item["det"].get("_step45_retracked")
                    or item["det"].get("region") == "dynamic"):
                continue
            chosen = heading if heading is not None else fallback
            if chosen is None:
                continue
            timestamp_ns = int(item.get("timestamp_ns", 0))
            world_from_lidar = coords.world_from_lidar(timestamp_ns)
            if world_from_lidar is None:
                continue
            target_local = _world_yaw_to_local(float(chosen), world_from_lidar)
            box = item["det"].get("box_lidar")
            if not isinstance(box, list) or len(box) < 7:
                continue
            box[6] = float(_wrap_angle(target_local))
            changed += 1
        if changed and len(details) < 200:
            details.append({
                "track_id": int(final_id),
                "observations": len(ordered),
            })
    return {
        "dynamic_yaw_aligned": changed,
        "tracks": len(details),
        "details": details,
    }


def revert_dynamic_yaw(
        frames: List[Dict[str, Any]],
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        config: Step45Config,
) -> Dict[str, Any]:
    """Whole-track yaw reversal after final IDs (reviewed).

    If the final motion trajectory is consistently opposite to box yaw
    (median directed difference > threshold), add pi to every detection yaw
    of that track.  Position and size are never changed.  Pure parked tracks
    are excluded.
    """
    by_final: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for items in tracks.values():
        for item in items:
            final_id = item["det"].get("track_id")
            if final_id is not None:
                by_final[int(final_id)].append(item)
    threshold = math.radians(float(config.yaw_reversal_threshold_deg))
    flipped = 0
    details: List[Dict[str, Any]] = []
    for final_id, items in by_final.items():
        stats = track_motion_stats(items)
        if stats is None or is_pure_static(stats, config):
            continue
        if float(stats["net"]) < 1.0:
            continue
        heading = _robust_track_heading(items)
        if heading is None:
            continue
        directed = np.asarray([
            _wrap_angle(float(item["yaw"]) - float(heading))
            for item in items
        ], dtype=np.float64)
        median_abs = float(np.median(np.abs(directed)))
        if median_abs <= threshold:
            continue
        for item in items:
            box = item["det"].get("box_lidar")
            if not isinstance(box, list) or len(box) < 7:
                continue
            box[6] = float(_wrap_angle(float(box[6]) + math.pi))
            item["yaw"] = float(_wrap_angle(float(item["yaw"]) + math.pi))
            flipped += 1
        details.append({
            "track_id": int(final_id),
            "observations": len(items),
            "trajectory_heading_deg": round(math.degrees(float(heading)), 3),
            "median_directed_diff_deg": round(
                math.degrees(median_abs), 3),
        })
    return {
        "enabled": bool(config.yaw_reversal_enabled),
        "threshold_deg": float(config.yaw_reversal_threshold_deg),
        "reversed_detections": flipped,
        "reversed_tracks": len(details),
        "details": details[:200],
    }


def dynamic_box_fit(
        frames: Sequence[Mapping[str, Any]],
        clip: Any,
        coords: tracking.CoordinateProvider,
        tracking_diagnostics: Mapping[str, Any],
        static_yaw_diagnostics: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Re-fit boxes only for step-4.5 re-tracked detections."""
    source = copy.deepcopy(list(frames))
    fitted, diagnostics = apply_car_box_fit(
        source, coords, clip, tracking_diagnostics,
        static_yaw_diagnostics, CarBoxFitConfig())
    replaced = 0
    for original, updated in zip(frames, fitted):
        for det, fitted_det in zip(
                original.get("detections", []), updated.get("detections", [])):
            if not det.get("_step45_retracked"):
                continue
            if (isinstance(fitted_det.get("box_lidar"), list)
                    and len(fitted_det["box_lidar"]) >= 7):
                det["box_lidar"] = list(fitted_det["box_lidar"])
                replaced += 1
    diagnostics.update({
        "mode": "step45_dynamic_only_box_fit",
        "dynamic_boxes_replaced": replaced,
        "frozen_boxes_kept": sum(
            len(frame.get("detections", [])) for frame in frames) - replaced,
    })
    return list(frames), diagnostics


def verify_static_freeze(
        before: Sequence[Mapping[str, Any]],
        after: Sequence[Mapping[str, Any]],
        retrackable: set[Tuple[int, int]],
) -> Dict[str, Any]:
    """Assert every non-retrackable detection is byte-for-byte unchanged."""
    if len(before) != len(after):
        raise AssertionError("step4.5 changed frame count")
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    for frame_index, (left_frame, right_frame) in enumerate(zip(before, after)):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("step4.5 changed frame order")
        left_dets = left_frame.get("detections", [])
        right_dets = right_frame.get("detections", [])
        if len(left_dets) != len(right_dets):
            raise AssertionError("step4.5 changed detection count")
        for detection_index, (left, right) in enumerate(
                zip(left_dets, right_dets)):
            if (frame_index, detection_index) in retrackable:
                continue
            checked += 1
            left_copy = copy.deepcopy(left)
            right_copy = copy.deepcopy(right)
            for value in (left_copy, right_copy):
                value.pop("_step45_new_id", None)
                value.pop("_step45_retracked", None)
                value.pop("_step45_world", None)
            if left_copy != right_copy:
                mismatches.append({
                    "frame_index": frame_index,
                    "detection_index": detection_index,
                    "track_id": left.get("track_id"),
                })
                if len(mismatches) >= 20:
                    break
    return {
        "checked_frozen_detections": checked,
        "mismatches": mismatches,
        "passed": not mismatches,
    }
