"""Static-direction voting for confirmed parking slots.

Only the reviewed static-direction helpers live here.  The old dynamic-heading
preview implementation has been archived under ``archive/geometry/``.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class StaticYawDirectionConfig:
    static_min_votes: int = 4
    static_min_margin: float = 0.15


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _world_yaw_to_local(world_yaw: float,
                        world_from_lidar: np.ndarray) -> float:
    world_xy = np.array([math.cos(world_yaw), math.sin(world_yaw)],
                        dtype=np.float64)
    planar = world_from_lidar[:2, :2]
    try:
        local_xy = np.linalg.solve(planar, world_xy)
    except np.linalg.LinAlgError:
        local_xy = np.linalg.lstsq(planar, world_xy, rcond=None)[0]
    return math.atan2(float(local_xy[1]), float(local_xy[0]))


def _direction_side(raw_yaw: float, axis_yaw: float) -> int:
    """Return 0 for axis direction and 1 for the opposite direction."""
    return 0 if abs(_wrap(raw_yaw - axis_yaw)) <= math.pi / 2.0 else 1


def _departure_cutoffs(
        static_yaw_diagnostics: Mapping[str, Any]) -> Dict[int, int]:
    cutoffs = {}
    for item in static_yaw_diagnostics.get("slots", []):
        value = item.get("departure_start_timestamp")
        if value is not None:
            cutoffs[int(item["track_id"])] = int(value)
    return cutoffs


def _static_direction_targets(
        final_frames: Sequence[Dict[str, Any]],
        pre_yaw_frames: Sequence[Dict[str, Any]],
        coords: tracking.CoordinateProvider,
        static_yaw_diagnostics: Mapping[str, Any],
        config: StaticYawDirectionConfig,
) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    """Vote on the directed sign of each already-stabilized static axis."""
    cutoffs = _departure_cutoffs(static_yaw_diagnostics)
    raw_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for frame in pre_yaw_frames:
        for det in frame.get("detections", []):
            tid = det.get("track_id")
            if tid is not None:
                raw_by_key[(int(frame["frame_id"]), int(tid))] = det

    slot_ids = {
        int(item["track_id"]): item
        for item in static_yaw_diagnostics.get("slots", [])
    }
    raw_angles: Dict[int, List[float]] = defaultdict(list)
    axis_angles: Dict[int, List[float]] = defaultdict(list)
    for frame in final_frames:
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for det in frame.get("detections", []):
            tid = det.get("track_id")
            if tid is None or int(tid) not in slot_ids:
                continue
            tid = int(tid)
            if timestamp >= cutoffs.get(tid, math.inf):
                continue
            raw_det = raw_by_key.get((timestamp, tid))
            if raw_det is None or not tracking.finite_box(det) or not tracking.finite_box(raw_det):
                continue
            raw_angles[tid].append(tracking.yaw_world(
                float(raw_det["box_lidar"][6]), world_from_lidar))
            axis_angles[tid].append(tracking.yaw_world(
                float(det["box_lidar"][6]), world_from_lidar))

    targets: Dict[int, float] = {}
    details = []
    for tid, item in slot_ids.items():
        raw = raw_angles.get(tid, [])
        axes = axis_angles.get(tid, [])
        if not raw or not axes:
            continue
        axis = float(np.angle(np.mean(np.exp(2j * np.asarray(axes)))) / 2.0)
        votes = Counter(_direction_side(angle, axis) for angle in raw)
        total = votes[0] + votes[1]
        winner = 0 if votes[0] >= votes[1] else 1
        margin = abs(votes[0] - votes[1]) / max(total, 1)
        accepted = (total >= config.static_min_votes
                    and margin >= config.static_min_margin)
        target = _wrap(axis + (math.pi if winner else 0.0))
        if not accepted:
            target = axis
        targets[tid] = target
        details.append({
            "track_id": tid,
            "votes_axis": votes[0],
            "votes_opposite": votes[1],
            "winner": "axis" if winner == 0 else "opposite",
            "margin": round(float(margin), 4),
            "accepted": accepted,
            "target_world_yaw": round(float(target), 6),
            "departure_start_timestamp": item.get("departure_start_timestamp"),
        })
    return targets, details
