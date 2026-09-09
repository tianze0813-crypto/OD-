"""Lock confirmed parking observations to one robust world-frame yaw."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from tracking import tracker_conservative as tracking
from tracking import tracker_static_first as static_first


@dataclass(frozen=True)
class StaticYawConfig:
    inlier_deviation: float = math.radians(20.0)
    row_validation_deviation: float = math.radians(15.0)
    row_min_slots: int = 2
    # Only observations inside a confirmed near-zero-motion run may be locked
    # to the parking yaw.  This keeps pure parked cars unchanged while a
    # departure / turn segment keeps its detector yaw.
    stationary_speed_gate: float = 0.30
    stationary_step_gate: float = 0.30
    stationary_max_gap_sec: float = 1.5
    stationary_min_run: int = 2
    # A whole track whose steps are all small is treated as parked even if
    # detector jitter gives one step a slightly higher instantaneous speed.
    # This keeps the reviewed pure-static behaviour byte-for-byte stable.
    stationary_whole_track_speed_gate: float = 0.80
    stationary_whole_track_step_gate: float = 0.35


def _world_yaw_to_local_yaw(
        world_yaw: float, world_from_lidar: np.ndarray) -> float:
    """Invert :func:`tracking.yaw_world` without assuming a yaw-only pose."""
    world_xy = np.array(
        [math.cos(world_yaw), math.sin(world_yaw)], dtype=np.float64)
    planar = world_from_lidar[:2, :2]
    try:
        local_xy = np.linalg.solve(planar, world_xy)
    except np.linalg.LinAlgError:
        local_xy = np.linalg.lstsq(planar, world_xy, rcond=None)[0]
    return math.atan2(float(local_xy[1]), float(local_xy[0]))


def _row_yaw_priors(slots: Sequence[static_first.StaticSlot]) -> Dict[int, float]:
    grouped: Dict[int, List[float]] = {}
    for slot in slots:
        if slot.row_id is not None:
            grouped.setdefault(int(slot.row_id), []).append(float(slot.yaw))
    return {
        row_id: static_first.circular_median_pi(values)
        for row_id, values in grouped.items()
        if len(values) >= 2
    }


def _departure_start_by_track(
        motion_coordination: Mapping[str, Any]) -> Dict[int, int]:
    starts = {}
    for item in motion_coordination.get("departures", []):
        if item.get("start_timestamp") is None:
            continue
        starts[int(item["inherited_track_id"])] = int(item["start_timestamp"])
    return starts


def _stationary_detection_ids(
        items: Sequence[tuple[int, Dict[str, Any], float, np.ndarray]],
        config: StaticYawConfig,
) -> tuple[set[int], Dict[str, Any]]:
    """Return ``id(det)`` keys inside confirmed near-zero-motion runs.

    A whole-track check preserves the reviewed pure-static behaviour.  For a
    track that stops and then departs, only the actual dwell run is locked;
    the moving / turning frames keep their detector yaw.
    """
    ordered = sorted(items, key=lambda item: item[0])
    if len(ordered) < 2:
        return ({id(item[1]) for item in ordered},
                {"reason": "short_track", "observations": len(ordered)})
    times = np.asarray([item[0] for item in ordered], dtype=np.float64) / 1e9
    points = np.asarray([item[3][:2] for item in ordered], dtype=np.float64)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    duration = max(float(times[-1] - times[0]), 1e-3)
    overall_speed = float(np.sum(steps)) / duration
    max_step = float(np.max(steps)) if len(steps) else 0.0
    if (overall_speed <= float(config.stationary_whole_track_speed_gate)
            and max_step <= float(config.stationary_whole_track_step_gate)):
        return ({id(item[1]) for item in ordered}, {
            "reason": "whole_track_stationary",
            "observations": len(ordered),
            "overall_speed": round(overall_speed, 4),
            "max_step": round(max_step, 4),
        })

    flags: List[bool] = []
    for index, (timestamp, _det, _yaw, world) in enumerate(ordered):
        local_speeds: List[float] = []
        local_steps: List[float] = []
        for other_index in (index - 1, index + 1):
            if not (0 <= other_index < len(ordered)):
                continue
            other_timestamp = ordered[other_index][0]
            dt = abs(other_timestamp - timestamp) / 1e9
            if dt <= 1e-3 or dt > float(config.stationary_max_gap_sec):
                continue
            other_world = ordered[other_index][3]
            step = float(np.linalg.norm(
                np.asarray(world[:2], dtype=np.float64)
                - np.asarray(other_world[:2], dtype=np.float64)))
            local_steps.append(step)
            local_speeds.append(step / dt)
        if not local_speeds:
            flags.append(False)
            continue
        flags.append(
            min(local_speeds) <= float(config.stationary_speed_gate)
            and min(local_steps) <= float(config.stationary_step_gate))

    accepted = [False] * len(flags)
    run_start = None
    for index, flag in enumerate(flags + [False]):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            if index - run_start >= int(config.stationary_min_run):
                for position in range(run_start, index):
                    accepted[position] = True
            run_start = None
    keys = {id(ordered[index][1])
            for index, flag in enumerate(accepted) if flag}
    return keys, {
        "reason": "run_gated",
        "observations": len(ordered),
        "stationary_observations": int(sum(accepted)),
        "overall_speed": round(overall_speed, 4),
    }


def stabilize_static_yaw(
        frames: List[Dict[str, Any]], coords: tracking.CoordinateProvider,
        slots: Sequence[static_first.StaticSlot],
        motion_coordination: Mapping[str, Any],
        config: StaticYawConfig = StaticYawConfig()) -> Dict[str, Any]:
    """Change only yaw for confirmed parking portions of static slots.

    A confirmed departure inherits the parked ID, so its motion segment is
    excluded beginning at the dynamic segment's first timestamp. Missing
    detections are never generated and no non-yaw field is changed.
    """
    before = copy.deepcopy(frames)
    slot_by_track = {
        int(slot.track_id): slot for slot in slots if slot.track_id is not None
    }
    departure_start = _departure_start_by_track(motion_coordination)
    row_priors = _row_yaw_priors(slots)
    observations: Dict[
        int, List[tuple[int, Dict[str, Any], float, np.ndarray]]] = {
        track_id: [] for track_id in slot_by_track
    }

    for frame in frames:
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for det in frame.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None or int(track_id) not in slot_by_track:
                continue
            track_id = int(track_id)
            if timestamp >= departure_start.get(track_id, math.inf):
                continue
            if not tracking.finite_box(det):
                continue
            world_yaw = tracking.yaw_world(
                float(det["box_lidar"][6]), world_from_lidar)
            world_center = tracking.center_world(
                det["box_lidar"], world_from_lidar)
            observations[track_id].append(
                (timestamp, det, world_yaw, world_center))

    targets = {}
    slot_details = []
    for track_id, slot in slot_by_track.items():
        items = observations[track_id]
        if not items:
            continue
        slot_yaw = float(slot.yaw)
        inliers = [
            world_yaw
            for _timestamp, _det, world_yaw, _world in items
            if tracking.angle_distance(
                world_yaw, slot_yaw, modulo_pi=True)
            <= config.inlier_deviation
        ]
        target = (static_first.circular_median_pi(inliers) if inliers else slot_yaw)

        row_prior = (None if slot.row_id is None
                     else row_priors.get(int(slot.row_id)))
        row_corrected = False
        if (row_prior is not None
                and tracking.angle_distance(target, row_prior, modulo_pi=True)
                > config.row_validation_deviation
                and len(inliers) < max(config.row_min_slots, len(items) // 2)):
            target = row_prior
            row_corrected = True
        targets[track_id] = target
        slot_details.append({
            "track_id": track_id,
            "slot_id": int(slot.slot_id),
            "parking_observations": len(items),
            "yaw_inliers": len(inliers),
            "target_world_yaw": round(float(target), 6),
            "row_id": slot.row_id,
            "row_corrected": row_corrected,
            "departure_start_timestamp": departure_start.get(track_id),
        })

    stationary_by_track: Dict[int, set[int]] = {}
    stationary_details = []
    for track_id in targets:
        keys, detail = _stationary_detection_ids(
            observations[track_id], config)
        stationary_by_track[track_id] = keys
        stationary_details.append({"track_id": track_id, **detail})

    changed = 0
    moving_excluded = 0
    absolute_delta = []
    for frame in frames:
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for det in frame.get("detections", []):
            track_id = det.get("track_id")
            if track_id is None:
                continue
            track_id = int(track_id)
            target = targets.get(track_id)
            if target is None or timestamp >= departure_start.get(track_id, math.inf):
                continue
            if id(det) not in stationary_by_track.get(track_id, set()):
                moving_excluded += 1
                continue
            old_yaw = float(det["box_lidar"][6])
            new_yaw = _world_yaw_to_local_yaw(target, world_from_lidar)
            delta = tracking.angle_distance(old_yaw, new_yaw, modulo_pi=True)
            det["box_lidar"][6] = float(new_yaw)
            absolute_delta.append(delta)
            if delta > 1e-9:
                changed += 1

    _verify_yaw_only(before, frames)
    return {
        "policy": {
            "scope": "confirmed_static_parking_portion_only",
            "departure_cutoff": "dynamic_segment_start_timestamp",
            "stationary_gate": (
                "only confirmed near-zero-motion runs are locked; moving / "
                "departure frames keep their detector yaw"),
            "stationary_speed_gate": float(config.stationary_speed_gate),
            "stationary_step_gate": float(config.stationary_step_gate),
            "inlier_deviation_deg": round(math.degrees(config.inlier_deviation), 3),
            "row_validation_deviation_deg": round(
                math.degrees(config.row_validation_deviation), 3),
            "mutated_field": "box_lidar[6]",
        },
        "static_slots_available": len(slot_by_track),
        "static_slots_stabilized": len(targets),
        "parking_boxes_stabilized": len(absolute_delta),
        "parking_boxes_moving_excluded": moving_excluded,
        "parking_boxes_changed": changed,
        "mean_axis_delta_deg": (0.0 if not absolute_delta else round(
            math.degrees(float(np.mean(absolute_delta))), 4)),
        "max_axis_delta_deg": (0.0 if not absolute_delta else round(
            math.degrees(float(np.max(absolute_delta))), 4)),
        "stationary_gate_details": stationary_details,
        "slots": slot_details,
    }


def _verify_yaw_only(before: Sequence[Dict[str, Any]],
                     after: Sequence[Dict[str, Any]]) -> None:
    if len(before) != len(after):
        raise AssertionError("static yaw stabilization changed frame count")
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("static yaw stabilization changed frame order")
        left_detections = left_frame.get("detections", [])
        right_detections = right_frame.get("detections", [])
        if len(left_detections) != len(right_detections):
            raise AssertionError("static yaw stabilization changed detection count")
        for left, right in zip(left_detections, right_detections):
            comparable = copy.deepcopy(right)
            comparable["box_lidar"][6] = left["box_lidar"][6]
            if comparable != left:
                raise AssertionError(
                    "static yaw stabilization changed a non-yaw field")
