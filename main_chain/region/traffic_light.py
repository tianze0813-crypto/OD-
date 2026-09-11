#!/usr/bin/env python3
"""Traffic-light / movement-group inference from pass1 trajectories.

Reviewed assumptions (global defaults; intentionally hard-coded here):

  * only trajectories are used -- no camera traffic-light colour detection;
  * one clip contains at most one intersection (100 m radar range);
  * straight and left-turn movements are mutually exclusive;
  * right turns are always allowed;
  * U-turns are treated as left turns;
  * movement groups are described by the swept area of their tracks.

The module is a pass1 diagnostic/analysis stage.  It classifies robust
dynamic tracks into straight / left / right movement groups, detects stop and
start events, and infers a simple phase timeline that can later be used to
merge red-light waiting / stop-and-go fragments in step2.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from region.direction_phase import build_direction_phase_diagnostics
from region.dynamic_region import DynamicTrack, _normalize_track
from region.parking_region import (
    Bounds,
    Polygon,
    _compute_bounds,
    _grid_shape,
    _polygon_area,
    _polygon_centroid,
    _to_pixel,
)


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _heading(points: np.ndarray) -> float:
    vector = points[-1] - points[0]
    return math.atan2(float(vector[1]), float(vector[0]))


def _robust_heading_sequence(
        points: np.ndarray, timestamps: np.ndarray,
        min_displacement: float,
) -> List[Tuple[float, np.ndarray, float]]:
    """Heading at each observation using a displacement window (metres).

    A fixed-frame heading window is very sensitive to detector centre jitter.
    Requiring a minimum displacement (default 3 m) gives a much more stable
    heading sequence for turn / lane-change detection.
    """

    count = int(len(points))
    if count < 2:
        return []
    minimum = max(float(min_displacement), 1e-3)
    sequence: List[Tuple[float, np.ndarray, float]] = []
    for index in range(count):
        forward = index + 1
        while (forward < count
               and float(np.linalg.norm(points[forward] - points[index]))
               < minimum):
            forward += 1
        if forward < count:
            vector = points[forward] - points[index]
        else:
            backward = index - 1
            while (backward >= 0
                   and float(np.linalg.norm(points[index] - points[backward]))
                   < minimum):
                backward -= 1
            if backward < 0:
                continue
            vector = points[index] - points[backward]
        sequence.append((
            float(timestamps[index]),
            np.asarray(points[index], dtype=np.float64).copy(),
            math.atan2(float(vector[1]), float(vector[0])),
        ))
    return sequence


def _stable_segment(
        points: np.ndarray, timestamps: np.ndarray,
        config: "TrafficLightConfig",
) -> Tuple[int, int, float, np.ndarray]:
    """First long segment of a trajectory with a stable heading.

    The first observations of a track are often mid-manoeuvre or noisy, so
    lane positions must come from the stable part of the trajectory, not from
    the entry.  The *first* qualifying segment is used (not the globally
    longest one) so a turning track is still described by its approach
    heading instead of its exit heading.

    Returns ``(start_index, end_index, heading, median_point)``.
    """

    count = int(len(points))
    if count < 2:
        return 0, max(count - 1, 0), 0.0, np.asarray(points[0]).copy()

    steps: List[Optional[float]] = []
    for index in range(count - 1):
        vector = points[index + 1] - points[index]
        if float(np.linalg.norm(vector)) < 1e-6:
            steps.append(None)
        else:
            steps.append(math.atan2(float(vector[1]), float(vector[0])))

    window = max(2, min(int(config.stable_window_observations),
                        max(1, count // 3)))
    tolerance = math.radians(float(config.stable_heading_tolerance_deg))
    minimum_path = float(config.stable_min_path_m)

    def _segment_result(start: int, end: int) -> Tuple[int, int, float,
                                                        np.ndarray]:
        segment = points[start:end + 1]
        vector = segment[-1] - segment[0]
        heading = math.atan2(float(vector[1]), float(vector[0]))
        return start, end, heading, np.median(segment, axis=0)

    best: Optional[Tuple[float, int, int]] = None
    for start in range(0, max(1, count - window)):
        end = start + window
        while end < count:
            values = [value for value in steps[start:end]
                      if value is not None]
            if len(values) < window:
                break
            reference = float(np.median(
                np.asarray(values, dtype=np.float64)))
            deltas = np.asarray(values, dtype=np.float64) - reference
            deltas = (deltas + math.pi) % (2.0 * math.pi) - math.pi
            spread = float(np.max(np.abs(deltas)))
            if spread > tolerance:
                break
            end += 1
        end -= 1
        if end - start + 1 < window + 1:
            continue
        path = float(np.sum(np.linalg.norm(
            np.diff(points[start:end + 1], axis=0), axis=1)))
        if best is None or path > best[0]:
            best = (path, start, end)
        if path >= minimum_path:
            return _segment_result(start, end)

    # Fallback: longest run of usable consecutive steps (e.g. a pure turn
    # with no straight part at all).
    run_start: Optional[int] = None
    longest: Optional[Tuple[int, int]] = None
    for index in range(count):
        valid = index < count - 1 and steps[index] is not None
        if valid and run_start is None:
            run_start = index
        if not valid and run_start is not None:
            candidate = (run_start, index)
            if longest is None or (candidate[1] - candidate[0]
                                   > longest[1] - longest[0]):
                longest = candidate
            run_start = None
    if run_start is not None:
        candidate = (run_start, count - 1)
        if longest is None or (candidate[1] - candidate[0]
                               > longest[1] - longest[0]):
            longest = candidate
    if longest is not None:
        return _segment_result(longest[0], longest[1])
    if best is not None:
        return _segment_result(best[1], best[2])
    return _segment_result(0, count - 1)


def _group_label(raw_movement: str) -> str:
    """Map a diagnostic raw movement to its phase group label."""

    if raw_movement in ("left", "right", "straight"):
        return raw_movement
    if raw_movement in ("uturn", "waiting_left"):
        return "left"
    return "straight"


# The reviewed traffic-phase rule is motor-vehicle-only.  Pedestrians and
# cyclists may still be tracked by step2, but they must not create lanes,
# stop events, movement groups or phase transitions.
NON_MOTOR_VEHICLE_CLASSES = {
    "pedestrian", "person", "cyclist", "bicycle", "motorcycle", "rider",
    "nonmotorized_vehicle", "non-motorized_vehicle", "nonmotorizedvehicle",
}


def _is_motor_vehicle_class(class_name: Any) -> bool:
    """Return True for motor vehicles (and legacy class-less inputs)."""

    text = str(class_name or "").strip()
    if not text:
        # Legacy/test track inputs carry no class; keep them to avoid
        # silently dropping the whole diagnostic.
        return True
    return text.casefold() not in NON_MOTOR_VEHICLE_CLASSES


def _turn_features(
        points: np.ndarray, timestamps: np.ndarray,
        config: "TrafficLightConfig",
) -> Dict[str, Any]:
    """Robust turn / lane-change / U-turn features for one trajectory.

    Reviewed 2026-09-09: trajectory judgement uses **centre positions only**.
    Detector yaw / box size are not read at all, so yaw noise or an ID switch
    that flips the box heading cannot change the classification.
    """

    features: Dict[str, Any] = {
        "entry_point": points[0].copy() if len(points) else np.zeros(2),
        "exit_point": points[-1].copy() if len(points) else np.zeros(2),
        "entry_heading": 0.0,
        "exit_heading": 0.0,
        "net_turn_deg": 0.0,
        "cum_turn_deg": 0.0,
        "cum_abs_turn_deg": 0.0,
        "turn_consistency": 0.0,
        "lateral_offset_m": 0.0,
        "max_lateral_offset_m": 0.0,
        "sign_changes": 0,
        "heading_flip_count": 0,
        "heading_rate_max_deg_per_m": 0.0,
        "max_step_flip_deg": 0.0,
        "is_uturn": False,
        "heading_flip": False,
        "heading_inconsistent": False,
        "turn_in_place": False,
        "lane_change": False,
        "lane_change_then_turn": False,
        "consecutive_lane_change": False,
        "lane_change_while_turning": False,
    }
    if len(points) < 2:
        return features
    sequence = _robust_heading_sequence(
        points, timestamps, config.heading_displacement_m)
    if len(sequence) >= 2:
        headings = [float(item[2]) for item in sequence]
        entry_heading = headings[0]
        exit_heading = headings[-1]
    else:
        window = max(2, min(int(config.heading_window), len(points)))
        entry_heading = _heading(points[:window])
        exit_heading = _heading(points[-window:])
        headings = [entry_heading, exit_heading]

    deltas = [_wrap_angle(headings[index + 1] - headings[index])
              for index in range(len(headings) - 1)]
    flip_limit = math.radians(float(config.heading_flip_deg))
    flips = [delta for delta in deltas if abs(delta) > flip_limit]
    cleaned = [0.0 if abs(delta) > flip_limit else delta for delta in deltas]
    cum_turn = float(sum(cleaned))
    cum_abs = float(sum(abs(delta) for delta in cleaned))
    consistency = abs(cum_turn) / cum_abs if cum_abs > 1e-9 else 0.0
    sign_changes = sum(1 for index in range(1, len(cleaned))
                       if cleaned[index] * cleaned[index - 1] < 0.0)
    net_turn = _wrap_angle(exit_heading - entry_heading)
    heading_rate_max = 0.0
    for index in range(len(sequence) - 1):
        delta = abs(_wrap_angle(float(sequence[index + 1][2])
                                - float(sequence[index][2])))
        distance = float(np.linalg.norm(
            np.asarray(sequence[index + 1][1], dtype=np.float64)
            - np.asarray(sequence[index][1], dtype=np.float64)))
        if distance > 1e-6:
            heading_rate_max = max(
                heading_rate_max, math.degrees(delta) / distance)

    entry_forward = np.asarray(
        [math.cos(entry_heading), math.sin(entry_heading)], dtype=np.float64)
    entry_right = np.asarray(
        [math.sin(entry_heading), -math.cos(entry_heading)], dtype=np.float64)
    relative = points - points[0]
    lateral = relative @ entry_right
    features.update(
        entry_point=points[0].copy(),
        exit_point=points[-1].copy(),
        entry_heading=entry_heading,
        exit_heading=exit_heading,
        net_turn_deg=math.degrees(net_turn),
        cum_turn_deg=math.degrees(cum_turn),
        cum_abs_turn_deg=math.degrees(cum_abs),
        turn_consistency=float(consistency),
        lateral_offset_m=float(lateral[-1]),
        max_lateral_offset_m=float(np.max(np.abs(lateral))),
        sign_changes=int(sign_changes),
        heading_flip_count=len(flips),
        heading_rate_max_deg_per_m=float(heading_rate_max),
        max_step_flip_deg=math.degrees(
            max((abs(delta) for delta in deltas), default=0.0)),
    )

    # Centre-only artefact detection.  A trajectory whose net heading change
    # is ~180 deg while the sustained cumulative turn is small and the lateral
    # displacement is negligible is a centre jump / association artefact
    # ("turning in place"), not a real U-turn.
    turn_in_place = bool(
        abs(features["net_turn_deg"]) >= float(config.uturn_min_deg)
        and abs(features["lateral_offset_m"])
        < float(config.uturn_min_lateral_m)
    )
    is_uturn = bool(
        abs(features["cum_turn_deg"]) >= float(config.uturn_min_deg)
        and abs(features["net_turn_deg"]) >= float(config.uturn_net_min_deg)
        and consistency >= float(config.uturn_consistency_min)
        and heading_rate_max <= float(
            config.uturn_max_heading_rate_deg_per_m)
        and not turn_in_place
    )
    heading_flip = bool(
        (features["heading_flip_count"] > 0
         and abs(features["net_turn_deg"]) < float(config.turn_min_deg))
        or turn_in_place
    )
    # A trajectory whose heading oscillates (many sign changes, low net
    # consistency) is a detector / association artefact, not a real turn.
    heading_inconsistent = bool(
        consistency < 0.5 and sign_changes >= 2)
    exit_heading_diff = abs(math.degrees(
        _wrap_angle(exit_heading - entry_heading)))
    lane_change = bool(
        float(config.lane_change_min_lateral_m)
        <= abs(features["lateral_offset_m"])
        <= float(config.lane_change_max_lateral_m)
        and abs(features["net_turn_deg"]) < float(config.turn_min_deg)
        and exit_heading_diff <= float(config.lane_change_exit_heading_deg)
    )
    if (not lane_change and sign_changes >= 1
            and abs(features["net_turn_deg"]) < float(config.turn_min_deg)
            and abs(features["lateral_offset_m"])
            >= float(config.lane_change_min_lateral_m)):
        lane_change = True
    lane_change_then_turn = _lane_change_then_turn(points, features, config)
    consecutive_lane_change = bool(
        abs(features["lateral_offset_m"])
        > float(config.max_lane_change_lanes) * float(config.lane_width_m))
    lane_change_while_turning = bool(
        lane_change
        and abs(features["cum_turn_deg"]) >= float(config.turn_min_deg))
    features.update(
        is_uturn=is_uturn,
        heading_flip=heading_flip,
        heading_inconsistent=heading_inconsistent,
        turn_in_place=turn_in_place,
        lane_change=lane_change,
        lane_change_then_turn=lane_change_then_turn,
        consecutive_lane_change=consecutive_lane_change,
        lane_change_while_turning=lane_change_while_turning,
    )
    return features


def _lane_change_then_turn(
        points: np.ndarray, features: Mapping[str, Any],
        config: "TrafficLightConfig",
) -> bool:
    """Detect a lane change followed by a turn (a traffic-law violation).

    The first ~60% of the trajectory must look like a lane change (lateral
    shift of about one lane while returning to the entry heading), and the
    remaining part must contain a sustained turn of at least ``turn_min_deg``.
    """

    if len(points) < 5:
        return False
    entry = np.asarray(features["entry_point"], dtype=np.float64)
    heading = float(features["entry_heading"])
    forward = np.asarray([math.cos(heading), math.sin(heading)],
                         dtype=np.float64)
    right = np.asarray([math.sin(heading), -math.cos(heading)],
                       dtype=np.float64)
    lateral = (points - entry) @ right
    split = max(2, int(len(points) * 0.6))
    if split >= len(points):
        return False
    early_lateral = float(lateral[split - 1])
    if not (float(config.lane_change_min_lateral_m)
            <= abs(early_lateral)
            <= float(config.lane_change_max_lateral_m)):
        return False
    window = max(2, min(int(config.heading_window), split))
    split_heading = _heading(points[split - window:split])
    if abs(math.degrees(_wrap_angle(split_heading - heading))) > float(
            config.lane_change_exit_heading_deg):
        return False
    late_window = max(2, min(int(config.heading_window), len(points) - split))
    late_heading = _heading(points[-late_window:])
    late_change = abs(math.degrees(_wrap_angle(late_heading - split_heading)))
    return bool(late_change >= float(config.turn_min_deg))


def _movement_scores(
        features: Mapping[str, Any], waiting_left: bool,
        config: "TrafficLightConfig",
) -> Dict[str, float]:
    """Score movement hypotheses; traffic-law style penalties are negative."""

    scores: Dict[str, float] = {
        "straight": 1.0,
        "left": 0.0,
        "right": 0.0,
        "uturn": 0.0,
        "lane_change": 0.0,
        "waiting_left": 0.0,
    }
    cum_turn = float(features["cum_turn_deg"])
    net_turn = float(features["net_turn_deg"])
    consistency = float(features["turn_consistency"])
    turn_min = float(config.turn_min_deg)
    if abs(cum_turn) >= turn_min:
        key = "left" if cum_turn > 0 else "right"
        scores[key] += 1.0 + consistency
    elif abs(cum_turn) >= turn_min * 0.5:
        key = "left" if cum_turn > 0 else "right"
        scores[key] += 0.5 + 0.5 * consistency
    exit_heading_diff = abs(math.degrees(_wrap_angle(
        float(features["exit_heading"]) - float(features["entry_heading"]))))
    if abs(net_turn) >= turn_min and exit_heading_diff >= turn_min:
        key = "left" if net_turn > 0 else "right"
        scores[key] += 0.5
    if features["lane_change"]:
        scores["lane_change"] += 1.0
        scores["straight"] += 0.5
    if features["lane_change_then_turn"]:
        scores["lane_change"] -= float(config.lane_change_then_turn_penalty)
    if features["consecutive_lane_change"]:
        scores["lane_change"] -= float(config.consecutive_lane_change_penalty)
    if features["lane_change_while_turning"]:
        scores["lane_change"] -= float(config.lane_change_while_turning_penalty)
    if features["is_uturn"]:
        scores["uturn"] += 3.0
    if waiting_left:
        scores["waiting_left"] += 1.0
        scores["left"] += 0.5
    if features["heading_flip"]:
        scores["straight"] += 1.0
        scores["left"] -= 1.0
        scores["right"] -= 1.0
    if features.get("heading_inconsistent", False):
        scores["straight"] += 1.0
        scores["left"] -= 1.0
        scores["right"] -= 1.0
    if features.get("turn_in_place", False):
        # "Turning in place" is not physically possible for a car; treat it
        # as a yaw artefact and merge into straight.
        scores["straight"] += 2.0
        scores["left"] -= 2.0
        scores["right"] -= 2.0
        scores["uturn"] -= 3.0
    return scores


def _pick_raw_movement(scores: Mapping[str, float]) -> str:
    """Pick the winning movement.

    Only the four real movements can win: ``lane_change`` / ``waiting_left``
    are diagnostic scores that can never reach the straight prior (1.0), so
    including them only made the argmax look like it could return them.
    """

    candidates = ("straight", "left", "right", "uturn")
    return max(candidates, key=lambda key: (float(scores.get(key, 0.0)),
                                            key == "straight"))


def _cluster_directions(
        entries: Sequence[Tuple[int, Mapping[str, Any]]],
        config: "TrafficLightConfig",
) -> List[List[Tuple[int, Mapping[str, Any]]]]:
    """Cluster tracks into travel directions by their stable heading.

    Reviewed 2026-09-09: the entry point / entry heading is too early and too
    noisy to define an approach (it produced 9-15 "approaches" for a 4-leg
    intersection).  The heading of the first stable trajectory segment is
    used instead.  There is deliberately no lateral gate: two parallel roads
    with the same heading are the same travel direction.  Opposite directions
    differ by ~180 degrees and stay separate.
    """

    tolerance = math.radians(float(config.direction_heading_tolerance_deg))
    gate = float(config.direction_lateral_gate_m)
    remaining = list(entries)
    groups: List[List[Tuple[int, Mapping[str, Any]]]] = []
    while remaining:
        seed = remaining.pop(0)
        seed_heading = float(seed[1]["stable_heading"])
        seed_point = np.asarray(seed[1]["stable_point"], dtype=np.float64)
        seed_right = np.asarray([math.sin(seed_heading),
                                 -math.cos(seed_heading)],
                                dtype=np.float64)
        group = [seed]
        rest: List[Tuple[int, Mapping[str, Any]]] = []
        for item in remaining:
            heading = float(item[1]["stable_heading"])
            point = np.asarray(item[1]["stable_point"], dtype=np.float64)
            lateral = abs(float((point - seed_point) @ seed_right))
            if (abs(_wrap_angle(heading - seed_heading)) <= tolerance
                    and lateral <= gate):
                group.append(item)
            else:
                rest.append(item)
        remaining = rest
        groups.append(group)
    return groups


def _lane_lateral_values(
        entries: Sequence[Tuple[int, Mapping[str, Any]]],
        origin: np.ndarray, forward: np.ndarray, right: np.ndarray,
) -> List[Tuple[float, float, int, Mapping[str, Any]]]:
    """Lateral / forward position of each track on its stable segment.

    The stable segment (see :func:`_stable_segment`) is used instead of the
    first observations: the entry is often mid-manoeuvre or noisy, which is
    the root cause of the old lane over-segmentation.
    """

    values: List[Tuple[float, float, int, Mapping[str, Any]]] = []
    for track_id, stats in entries:
        points = np.asarray(stats.get("stable_points", []),
                            dtype=np.float64)
        if len(points) == 0:
            continue
        lateral = float(np.median((points - origin) @ right))
        forward_coord = float(np.median((points - origin) @ forward))
        values.append((lateral, forward_coord, int(track_id), stats))
    values.sort(key=lambda item: item[0])
    return values


def _build_direction_lanes(
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        config: "TrafficLightConfig",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One lane per movement (left / straight / right) per travel direction.

    Reviewed rules (2026-09-09):

    * directions come from the stable-segment heading (:func:`_cluster_directions`),
      not from the entry point;
    * a lane record is a traffic-rule lane group:
      ``left`` = left-turn movements, ``straight`` = all through movements (no
      further subdivision by lateral position), ``right`` = right-turn
      movements;
    * a direction needs at least ``min_direction_tracks`` tracks to emit
      lanes; isolated tracks still contribute to movement groups, stop events
      and phases but do not create a lane record;
    * lane lateral position and width come from each track's stable segment.
    """

    entries: List[Tuple[int, Mapping[str, Any]]] = [
        (int(track_id), stats) for track_id, stats in classified]
    if not entries:
        return [], {"direction_count": 0, "unsupported_directions": 0,
                    "unsupported_direction_tracks": 0}
    directions = _cluster_directions(entries, config)
    minimum_tracks = max(1, int(config.min_direction_tracks))
    lane_width = float(config.lane_width_m)
    lanes: List[Dict[str, Any]] = []
    lane_counter = 0
    unsupported_directions = 0
    unsupported_tracks = 0

    for direction_index, group in enumerate(directions):
        if len(group) < minimum_tracks:
            unsupported_directions += 1
            unsupported_tracks += len(group)
            continue

        sin_sum = sum(math.sin(float(item[1]["stable_heading"]))
                      for item in group)
        cos_sum = sum(math.cos(float(item[1]["stable_heading"]))
                      for item in group)
        mean_heading = math.atan2(sin_sum, cos_sum)
        forward = np.asarray([math.cos(mean_heading), math.sin(mean_heading)],
                             dtype=np.float64)
        right = np.asarray([math.sin(mean_heading), -math.cos(mean_heading)],
                           dtype=np.float64)
        origin = np.median(np.asarray(
            [item[1]["stable_point"] for item in group],
            dtype=np.float64), axis=0)

        values = _lane_lateral_values(group, origin, forward, right)
        buckets: Dict[str, List[Tuple[float, float, int,
                                      Mapping[str, Any]]]] = {
            "left": [], "straight": [], "right": []}
        for lateral, forward_coord, track_id, stats in values:
            movement = _group_label(str(stats["raw_movement"]))
            if movement not in buckets:
                movement = "straight"
            buckets[movement].append(
                (lateral, forward_coord, track_id, stats))
            stats["_stable_lateral"] = float(lateral)
            stats["_direction_id"] = int(direction_index)

        # Stop-line estimate: median forward coordinate of stop events.
        stop_forwards: List[float] = []
        exit_forwards: List[float] = []
        for _track_id, stats in group:
            for _start, _end, position in _stop_intervals(stats, config):
                stop_forwards.append(float(
                    (np.asarray(position, dtype=np.float64) - origin)
                    @ forward))
            points = np.asarray([item["world"]
                                 for item in stats["normalized"]],
                                dtype=np.float64)
            if len(points):
                exit_forwards.append(float((points[-1] - origin) @ forward))
        stop_line_forward = (
            float(np.median(stop_forwards))
            if len(stop_forwards) >= int(config.stop_line_min_events)
            else None)
        direction_exit_forward = (
            float(np.median(exit_forwards)) if exit_forwards else None)

        present = [movement for movement in ("left", "straight", "right")
                   if buckets[movement]]
        centres: Dict[str, float] = {
            movement: float(np.median(
                [member[0] for member in buckets[movement]]))
            for movement in present}
        # Traffic-rule lane order: left lane left of the through lane, right
        # lane right of it.  A few misclassified movements must not flip the
        # drawn order.
        straight_centre = centres.get("straight")
        if straight_centre is None and len(centres) > 1:
            straight_centre = float(np.median(list(centres.values())))
        if straight_centre is not None:
            if "left" in centres:
                centres["left"] = min(centres["left"],
                                      straight_centre - lane_width)
            if "right" in centres:
                centres["right"] = max(centres["right"],
                                       straight_centre + lane_width)

        for index, movement in enumerate(present):
            members = buckets[movement]
            centre_lateral = centres[movement]
            # Centre-only geometry: the lane band is the spread of the
            # members' stable-segment lateral centres plus one vehicle width.
            member_laterals = np.asarray([member[0] for member in members],
                                         dtype=np.float64)
            if len(member_laterals):
                extent_low = float(np.percentile(
                    member_laterals, float(config.lane_extent_percentile)))
                extent_high = float(np.percentile(
                    member_laterals,
                    100.0 - float(config.lane_extent_percentile)))
            else:
                extent_low = extent_high = centre_lateral
            counts = {"straight": 0, "left": 0, "right": 0}
            for member in members:
                counts[_group_label(str(member[3]["raw_movement"]))] += 1
            lanes.append({
                "lane_id": f"lane_{lane_counter}",
                "direction_id": int(direction_index),
                "corridor_id": int(direction_index),
                "lane_index": index,
                "lane_count": len(present),
                "heading": mean_heading,
                "center": origin + right * centre_lateral,
                "forward": forward,
                "right": right,
                "direction_origin": origin.copy(),
                "lateral_center": centre_lateral,
                "width": max(lane_width,
                             float(extent_high - extent_low) + 0.5 * lane_width),
                "movement": movement,
                "dominant_movement": movement,
                "driving_extent": [round(extent_low, 3),
                                   round(extent_high, 3)],
                "track_count": len(members),
                "track_ids": [member[2] for member in members],
                "movement_counts": counts,
                "straight_fraction": (
                    float(counts["straight"])
                    / max(sum(counts.values()), 1)),
                "stop_line_forward": stop_line_forward,
                "direction_exit_forward": direction_exit_forward,
            })
            lane_counter += 1
    return lanes, {
        "direction_count": len(directions),
        "unsupported_directions": unsupported_directions,
        "unsupported_direction_tracks": unsupported_tracks,
    }


def _adjacent_right_tracks(
        stats: Mapping[str, Any],
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        config: "TrafficLightConfig",
) -> List[Mapping[str, Any]]:
    """Tracks about one to 2.5 lane widths to the right of ``stats``.

    The waiting-left (待转区) rule needs the adjacent through lane, but the
    movement-level lane model does not model individual lanes.  Any track of
    the same travel direction whose stable lateral position sits one lane
    width to the right provides the same evidence.
    """

    lateral = stats.get("_stable_lateral")
    direction_id = stats.get("_direction_id")
    if lateral is None or direction_id is None:
        return []
    low = float(config.lane_half_width_m)
    high = float(config.lane_width_m) * 2.5
    result: List[Mapping[str, Any]] = []
    for _track_id, other in classified:
        if other is stats:
            continue
        if other.get("_direction_id") != direction_id:
            continue
        other_lateral = other.get("_stable_lateral")
        if other_lateral is None:
            continue
        delta = float(other_lateral) - float(lateral)
        if low <= delta <= high:
            result.append(other)
    result.sort(key=lambda other: abs(
        (float(other["_stable_lateral"]) - float(lateral))
        - float(config.lane_width_m)))
    return result


def _stop_intervals(
        stats: Mapping[str, Any], config: "TrafficLightConfig",
) -> List[Tuple[float, float, np.ndarray]]:
    normalized = stats["normalized"]
    times = np.asarray([item["timestamp"] for item in normalized],
                       dtype=np.float64)
    points = np.asarray([item["world"] for item in normalized],
                        dtype=np.float64)
    if len(times) < 2:
        return []
    intervals = np.diff(times)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid = intervals > 1e-3
    speeds = np.full(len(steps), np.inf, dtype=np.float64)
    speeds[valid] = steps[valid] / intervals[valid]
    low = speeds <= float(config.stop_speed_threshold)
    result: List[Tuple[float, float, np.ndarray]] = []
    index = 0
    while index < len(low):
        if not low[index]:
            index += 1
            continue
        start = index
        while index + 1 < len(low) and low[index + 1]:
            index += 1
        end = index
        duration = float(times[end + 1] - times[start])
        if duration >= float(config.stop_min_duration):
            result.append((
                float(times[start]), float(times[end + 1]),
                points[(start + end + 1) // 2].copy()))
        index += 1
    return result


def _speeds_in_window(
        stats: Mapping[str, Any], window_start: float, window_end: float,
) -> List[float]:
    normalized = stats["normalized"]
    times = np.asarray([item["timestamp"] for item in normalized],
                       dtype=np.float64)
    points = np.asarray([item["world"] for item in normalized],
                        dtype=np.float64)
    speeds: List[float] = []
    for index in range(len(times) - 1):
        if times[index] < window_start or times[index + 1] > window_end:
            continue
        delta_t = float(times[index + 1] - times[index])
        if delta_t <= 1e-3:
            continue
        speeds.append(float(np.linalg.norm(
            points[index + 1] - points[index])) / delta_t)
    return speeds


def _displacement_in_window(
        stats: Mapping[str, Any], window_start: float, window_end: float,
) -> float:
    normalized = stats["normalized"]
    points = [item["world"] for item in normalized
              if window_start <= float(item["timestamp"]) <= window_end]
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(
        np.asarray(points[-1], dtype=np.float64)
        - np.asarray(points[0], dtype=np.float64)))


def _mean_speed_in_window(
        stats: Mapping[str, Any], window_start: float, window_end: float,
) -> float:
    speeds = _speeds_in_window(stats, window_start, window_end)
    return float(np.mean(speeds)) if speeds else 0.0


def _apply_lane_context(
        classified: List[Tuple[int, Dict[str, Any]]],
        config: "TrafficLightConfig",
) -> Dict[str, Any]:
    """Movement-lane majority voting, lane-change penalties and waiting-left.

    Implements the reviewed rules:

    * a lane change must end in a straight movement, otherwise it is penalised;
    * consecutive lane changes / lane changes while turning are penalised;
    * an isolated weak turn inside a straight-dominant approach (>=70% straight)
      is merged into straight, unless it has strong curvature evidence, a
      coherent turn partner, or is waiting in a left-turn area;
    * a track that waits in a left-turn area while the adjacent lane to its
      right is moving (>30 km/h) is kept as left even when its net heading is
      small.

    Lanes are movement-level (left / straight / right) per travel direction;
    see :func:`_build_direction_lanes`.
    """

    lanes, direction_diagnostics = _build_direction_lanes(classified, config)
    lane_by_id = {lane["lane_id"]: lane for lane in lanes}
    lane_by_track: Dict[int, Mapping[str, Any]] = {}
    for lane in lanes:
        for track_id in lane.get("track_ids", []):
            lane_by_track[int(track_id)] = lane
    for track_id, stats in classified:
        lane = lane_by_track.get(int(track_id))
        stats["entry_lane_id"] = lane["lane_id"] if lane is not None else None
        stats["_entry_lane"] = lane

    for track_id, stats in classified:
        stats["waiting_left"] = False
        stats["adjacent_lane_id"] = None
        stats["waiting_evidence"] = []
        if stats.get("_entry_lane") is None:
            continue
        adjacent_tracks = _adjacent_right_tracks(stats, classified, config)
        if not adjacent_tracks:
            continue
        adjacent = adjacent_tracks[0]
        stats["adjacent_lane_id"] = adjacent.get("entry_lane_id")
        for start, end, position in _stop_intervals(stats, config):
            margin = float(config.waiting_window_margin_sec)
            window_start = start - margin
            window_end = end + margin
            adjacent_moving = False
            for other in adjacent_tracks:
                speeds = _speeds_in_window(other, window_start, window_end)
                if any(speed >= float(config.adjacent_moving_speed_mps)
                       for speed in speeds):
                    adjacent_moving = True
                    break
            if not adjacent_moving:
                continue
            displacement = _displacement_in_window(stats, start, end)
            mean_speed = _mean_speed_in_window(stats, start, end)
            if (displacement <= float(config.waiting_self_displacement_m)
                    and mean_speed <= float(config.waiting_self_speed_mps)):
                stats["waiting_left"] = True
                stats["waiting_evidence"].append({
                    "start_timestamp": round(float(start), 4),
                    "end_timestamp": round(float(end), 4),
                    "position": [round(float(value), 4)
                                 for value in position],
                    "adjacent_lane_id": adjacent.get("entry_lane_id"),
                    "self_displacement_m": round(float(displacement), 4),
                    "self_mean_speed_mps": round(float(mean_speed), 4),
                })

    for track_id, stats in classified:
        scores = _movement_scores(
            stats["features"], bool(stats["waiting_left"]), config)
        stats["scores"] = scores
        stats["raw_movement"] = _pick_raw_movement(scores)

    lane_members: Dict[str, List[Dict[str, Any]]] = {
        lane["lane_id"]: [] for lane in lanes}
    for _track_id, stats in classified:
        lane_id = stats.get("entry_lane_id")
        if lane_id in lane_members:
            lane_members[lane_id].append(stats)

    # Direction-level majority: a side of the road that is overwhelmingly
    # straight must not create a turn lane from one noisy outlier.
    stats_by_track = {int(track_id): stats for track_id, stats in classified}
    direction_members: Dict[int, List[Dict[str, Any]]] = {}
    for lane in lanes:
        members = [stats_by_track[int(track_id)]
                   for track_id in lane.get("track_ids", [])
                   if int(track_id) in stats_by_track]
        direction_members.setdefault(
            int(lane.get("direction_id", -1)), []).extend(members)
    direction_straight_fraction: Dict[int, float] = {}
    for direction_id, members in direction_members.items():
        total = len(members)
        straight = sum(
            1 for stats in members
            if _group_label(str(stats["raw_movement"])) == "straight")
        direction_straight_fraction[direction_id] = (
            float(straight) / total if total else 0.0)

    for lane in lanes:
        members = lane_members[lane["lane_id"]]
        counts = {"straight": 0, "left": 0, "right": 0}
        for stats in members:
            counts[_group_label(str(stats["raw_movement"]))] += 1
        total = sum(counts.values())
        lane["movement_counts"] = counts
        lane["straight_fraction"] = (
            float(counts["straight"]) / total if total else 0.0)
        # Keep the traffic-rule label (leftmost=left, rightmost=right,
        # middle=straight) for the lane map; fall back to the majority.
        lane["dominant_movement"] = str(
            lane.get("movement", max(counts, key=counts.get)))
        lane["track_count"] = len(members)

    reclassified = 0
    for _track_id, stats in classified:
        features = stats["features"]
        raw_movement = str(stats["raw_movement"])
        final = _group_label(raw_movement)
        reason = ""
        lane = stats.get("_entry_lane")
        if lane is not None:
            direction_id = int(lane.get("direction_id", -1))
            straight_fraction = max(
                float(lane["straight_fraction"]),
                float(direction_straight_fraction.get(direction_id, 0.0)))
            members = direction_members.get(direction_id, [])
            if (final in ("left", "right")
                    and straight_fraction >= float(
                        config.straight_dominant_fraction)):
                strong_turn = bool(
                    (abs(float(features["net_turn_deg"]))
                     >= float(config.strong_turn_deg)
                     and float(features["turn_consistency"])
                     >= float(config.turn_consistency_min))
                    or (abs(float(features["net_turn_deg"]))
                        >= float(config.turn_min_deg)
                        and float(features["turn_consistency"]) >= 0.8))
                partner_count = sum(
                    1 for other in members
                    if other is not stats
                    and _group_label(str(other["raw_movement"])) == final
                    and abs(float(other["features"]["net_turn_deg"]))
                    >= float(config.turn_min_deg))
                if not (strong_turn or stats["waiting_left"]
                        or partner_count >= int(config.min_turn_partner_tracks)):
                    final = "straight"
                    reason = "straight_lane_minority"
            if final == "straight" and stats["waiting_left"]:
                final = "left"
                reason = "waiting_left_promoted"
            if (final == "straight" and not features["lane_change"]
                    and not features["heading_flip"]
                    and not features["turn_in_place"]
                    and abs(float(features["net_turn_deg"]))
                    >= float(config.strong_turn_deg)
                    and float(features["turn_consistency"])
                    >= float(config.turn_consistency_min)):
                final = ("left" if float(features["net_turn_deg"]) > 0
                         else "right")
                reason = "curvature_promoted"
        stats["final_movement"] = final
        stats["movement"] = final
        stats["reclassify_reason"] = reason
        stats["reclassified"] = bool(raw_movement != final)
        if stats["reclassified"]:
            reclassified += 1

    lane_records = []
    for lane in lanes:
        lane_records.append({
            "lane_id": lane["lane_id"],
            "direction_id": lane.get("direction_id"),
            "corridor_id": lane.get("corridor_id"),
            "lane_index": lane.get("lane_index"),
            "lane_count": lane.get("lane_count"),
            "heading_deg": round(math.degrees(float(lane["heading"])), 3),
            "center": [round(float(value), 4)
                       for value in np.asarray(lane["center"])],
            "direction_origin": [round(float(value), 4)
                                 for value in np.asarray(
                                     lane.get("direction_origin",
                                              lane["center"]))],
            "forward": [round(float(value), 4)
                        for value in np.asarray(
                            lane.get("forward", (1.0, 0.0)))],
            "width_m": lane.get("width"),
            "width": lane.get("width"),
            "driving_extent": lane.get("driving_extent"),
            "stop_line_forward": lane.get("stop_line_forward"),
            "direction_exit_forward": lane.get("direction_exit_forward"),
            "track_count": lane["track_count"],
            "track_ids": list(lane["track_ids"]),
            "movement_counts": dict(lane["movement_counts"]),
            "straight_fraction": round(
                float(lane["straight_fraction"]), 4),
            "movement": lane.get("movement"),
            "dominant_movement": lane["dominant_movement"],
        })
    diagnostics = {
        "lane_count": len(lanes),
        "direction_count": direction_diagnostics["direction_count"],
        "unsupported_directions": direction_diagnostics[
            "unsupported_directions"],
        "unsupported_direction_tracks": direction_diagnostics[
            "unsupported_direction_tracks"],
        "reclassified_tracks": reclassified,
        "waiting_left_tracks": sum(
            1 for _tid, stats in classified if stats["waiting_left"]),
        "lane_change_tracks": sum(
            1 for _tid, stats in classified
            if stats["features"]["lane_change"]),
        "heading_flip_tracks": sum(
            1 for _tid, stats in classified
            if stats["features"]["heading_flip"]),
        "uturn_tracks": sum(
            1 for _tid, stats in classified
            if stats["features"]["is_uturn"]),
        "lane_groups": lane_records,
    }
    return diagnostics


@dataclass(frozen=True)
class TrafficLightConfig:
    resolution: float = 1.0
    bounds_margin: float = 10.0

    # Robust trajectory filtering.
    min_observations: int = 5
    min_path_length: float = 15.0
    min_net_ratio: float = 0.55
    min_p90_speed: float = 1.5

    # Movement classification (world frame, positive angle = left turn).
    heading_window: int = 3
    heading_displacement_m: float = 3.0
    straight_max_deg: float = 30.0
    turn_min_deg: float = 45.0
    strong_turn_deg: float = 50.0
    turn_consistency_min: float = 0.6
    uturn_min_deg: float = 150.0
    uturn_net_min_deg: float = 120.0
    uturn_consistency_min: float = 0.7
    # A physically plausible U-turn needs lateral displacement (>=2x the
    # minimum turning radius).  A 180-degree heading change with almost no
    # lateral offset is a detector-yaw / association artefact ("turning in
    # place"), which must be merged into straight.
    uturn_min_lateral_m: float = 6.0
    heading_flip_deg: float = 120.0

    # Movement-level lane model (Chinese standard lane width 3.5-3.75 m).
    # Reviewed 2026-09-09: one lane record per movement (left / straight /
    # right) per travel direction; no lateral clustering inside a direction.
    lane_width_m: float = 3.75
    lane_half_width_m: float = 1.75
    lane_extent_percentile: float = 2.0
    # Stable-segment estimation: lane lateral positions come from the first
    # long segment with a nearly constant heading, not from the entry.
    stable_window_observations: int = 5
    stable_heading_tolerance_deg: float = 12.0
    stable_min_path_m: float = 10.0
    # Direction clustering: travel directions are clustered by the stable
    # heading of the first stable segment.  Opposite directions (~180 deg)
    # stay separate; the lateral gate keeps parallel but spatially distinct
    # roads apart.
    direction_heading_tolerance_deg: float = 35.0
    direction_lateral_gate_m: float = 30.0
    # A direction needs at least this many tracks to emit lane records;
    # isolated tracks still contribute to movement groups, stop events and
    # phases, but do not create a lane.
    min_direction_tracks: int = 2
    # Stop-line estimation: median forward coordinate of stop events.
    stop_line_min_events: int = 2
    # A real U-turn has finite curvature; a 180-degree yaw / heading flip has
    # a very high heading rate.
    uturn_max_heading_rate_deg_per_m: float = 30.0
    straight_dominant_fraction: float = 0.70
    # A weak turn inside a straight-dominant direction is kept when at least
    # this many *other* tracks in the direction support the same turn (self
    # excluded).  Two real turn tracks therefore support each other.
    min_turn_partner_tracks: int = 1

    # Lane-change / overtake detection and traffic-law style penalties.
    lane_change_min_lateral_m: float = 1.5
    lane_change_max_lateral_m: float = 4.0
    lane_change_exit_heading_deg: float = 30.0
    max_lane_change_lanes: float = 1.5
    lane_change_then_turn_penalty: float = 0.8
    consecutive_lane_change_penalty: float = 0.8
    lane_change_while_turning_penalty: float = 0.8

    # Left-turn waiting area (待转区): compare with the adjacent right lane.
    adjacent_moving_speed_mps: float = 30.0 / 3.6
    waiting_self_speed_mps: float = 1.0
    waiting_self_displacement_m: float = 2.0
    waiting_window_margin_sec: float = 0.5

    # Stop / start detection.
    stop_speed_threshold: float = 0.8
    stop_min_duration: float = 0.6
    moving_speed_threshold: float = 1.0

    # Movement-group clustering by entry/exit position.
    group_cluster_radius: float = 8.0
    min_group_tracks: int = 2

    # Phase timeline.
    phase_bin_sec: float = 0.2
    right_turn_always_green: bool = True
    uturn_as_left: bool = True

    # Direction-level four-phase diagnostics (read-only; see
    # ``region.direction_phase``).  The reviewed plan is the standard Chinese
    # four-phase plan: axis A straight -> axis A left -> axis B straight ->
    # axis B left.  Right turns are always allowed and permissive left turns
    # are allowed during the same axis' straight phase.
    axis_pair_tolerance_deg: float = 45.0
    left_permissive_enabled: bool = True
    direction_state_smooth_sec: float = 0.8
    direction_state_gap_sec: float = 1.2
    # A signal colour persists until new evidence arrives.  Unknown bins are
    # therefore held at the previous state for up to this long; longer gaps
    # stay unknown instead of inventing a phase.
    direction_state_hold_sec: float = 3.0
    direction_state_min_duration_sec: float = 1.0
    stop_line_tolerance_m: float = 4.0
    # Stop-line estimate: use the per-track *first* stop (arrival at the
    # queue / line) and guard against a waiting-area entry being much further
    # downstream than the median first stop.
    stop_line_first_outlier_m: float = 10.0
    stop_line_first_percentile: float = 75.0
    crossing_step_m: float = 2.0
    # Green evidence is concentrated around the stop-line crossing instead of
    # the whole downstream trajectory: a vehicle that crossed several seconds
    # ago must not keep its group green forever.
    crossing_green_before_sec: float = 0.5
    crossing_green_after_sec: float = 2.0
    phase_min_duration_sec: float = 2.0
    phase_transition_penalty: float = 0.4
    phase_reverse_penalty: float = 1.5
    phase_skip_penalty: float = 2.5
    phase_unknown_penalty: float = 0.3

    # Gating: only enable traffic-light logic when the clip has enough
    # sustained dynamic activity and a real straight/left mix.  A normal road
    # segment with a few mostly-straight vehicles is not an intersection.
    min_robust_tracks: int = 25
    min_straight_tracks: int = 5
    min_left_tracks: int = 3
    min_stop_events: int = 5
    max_straight_fraction: float = 0.85
    # The reviewed decision (2026-09-09) is to run the traffic-light phase
    # logic on every clip; keep the old gate behind this flag for later use.
    enable_gating: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "bounds_margin": self.bounds_margin,
            "min_observations": self.min_observations,
            "min_path_length": self.min_path_length,
            "min_net_ratio": self.min_net_ratio,
            "min_p90_speed": self.min_p90_speed,
            "heading_window": self.heading_window,
            "heading_displacement_m": self.heading_displacement_m,
            "straight_max_deg": self.straight_max_deg,
            "turn_min_deg": self.turn_min_deg,
            "strong_turn_deg": self.strong_turn_deg,
            "turn_consistency_min": self.turn_consistency_min,
            "uturn_min_deg": self.uturn_min_deg,
            "uturn_net_min_deg": self.uturn_net_min_deg,
            "uturn_consistency_min": self.uturn_consistency_min,
            "uturn_min_lateral_m": self.uturn_min_lateral_m,
            "heading_flip_deg": self.heading_flip_deg,
            "lane_width_m": self.lane_width_m,
            "lane_half_width_m": self.lane_half_width_m,
            "lane_extent_percentile": self.lane_extent_percentile,
            "stable_window_observations": self.stable_window_observations,
            "stable_heading_tolerance_deg": self.stable_heading_tolerance_deg,
            "stable_min_path_m": self.stable_min_path_m,
            "direction_heading_tolerance_deg": (
                self.direction_heading_tolerance_deg),
            "direction_lateral_gate_m": self.direction_lateral_gate_m,
            "min_direction_tracks": self.min_direction_tracks,
            "stop_line_min_events": self.stop_line_min_events,
            "uturn_max_heading_rate_deg_per_m": (
                self.uturn_max_heading_rate_deg_per_m),
            "straight_dominant_fraction": self.straight_dominant_fraction,
            "min_turn_partner_tracks": self.min_turn_partner_tracks,
            "lane_change_min_lateral_m": self.lane_change_min_lateral_m,
            "lane_change_max_lateral_m": self.lane_change_max_lateral_m,
            "lane_change_exit_heading_deg": self.lane_change_exit_heading_deg,
            "max_lane_change_lanes": self.max_lane_change_lanes,
            "lane_change_then_turn_penalty": self.lane_change_then_turn_penalty,
            "consecutive_lane_change_penalty": self.consecutive_lane_change_penalty,
            "lane_change_while_turning_penalty": self.lane_change_while_turning_penalty,
            "adjacent_moving_speed_mps": self.adjacent_moving_speed_mps,
            "waiting_self_speed_mps": self.waiting_self_speed_mps,
            "waiting_self_displacement_m": self.waiting_self_displacement_m,
            "waiting_window_margin_sec": self.waiting_window_margin_sec,
            "stop_speed_threshold": self.stop_speed_threshold,
            "stop_min_duration": self.stop_min_duration,
            "moving_speed_threshold": self.moving_speed_threshold,
            "group_cluster_radius": self.group_cluster_radius,
            "min_group_tracks": self.min_group_tracks,
            "phase_bin_sec": self.phase_bin_sec,
            "right_turn_always_green": self.right_turn_always_green,
            "uturn_as_left": self.uturn_as_left,
            "axis_pair_tolerance_deg": self.axis_pair_tolerance_deg,
            "left_permissive_enabled": self.left_permissive_enabled,
            "direction_state_smooth_sec": self.direction_state_smooth_sec,
            "direction_state_gap_sec": self.direction_state_gap_sec,
            "direction_state_hold_sec": self.direction_state_hold_sec,
            "direction_state_min_duration_sec": (
                self.direction_state_min_duration_sec),
            "stop_line_tolerance_m": self.stop_line_tolerance_m,
            "stop_line_first_outlier_m": self.stop_line_first_outlier_m,
            "stop_line_first_percentile": self.stop_line_first_percentile,
            "crossing_step_m": self.crossing_step_m,
            "crossing_green_before_sec": self.crossing_green_before_sec,
            "crossing_green_after_sec": self.crossing_green_after_sec,
            "phase_min_duration_sec": self.phase_min_duration_sec,
            "phase_transition_penalty": self.phase_transition_penalty,
            "phase_reverse_penalty": self.phase_reverse_penalty,
            "phase_skip_penalty": self.phase_skip_penalty,
            "phase_unknown_penalty": self.phase_unknown_penalty,
            "min_robust_tracks": self.min_robust_tracks,
            "min_straight_tracks": self.min_straight_tracks,
            "min_left_tracks": self.min_left_tracks,
            "min_stop_events": self.min_stop_events,
            "max_straight_fraction": self.max_straight_fraction,
            "enable_gating": self.enable_gating,
            "assumptions": [
                "trajectory-only inference; no camera light colour",
                "single intersection per clip (100 m radar)",
                "standard four-phase plan: axis straight -> axis left -> "
                "other axis straight -> other axis left",
                "straight/left are mutually exclusive inside one axis; "
                "different axes have independent states",
                "permissive left is allowed during the same axis' straight "
                "phase",
                "right turn is always allowed",
                "U-turn is treated as left turn",
                "lane change must end in a straight movement (traffic-law "
                "style penalty otherwise)",
            ],
        }


@dataclass
class TrafficLightResult:
    bounds: Bounds
    resolution: float
    traffic_light_enabled: bool = False
    groups: List[Dict[str, Any]] = field(default_factory=list)
    stop_events: List[Dict[str, Any]] = field(default_factory=list)
    phase_timeline: List[Dict[str, Any]] = field(default_factory=list)
    track_classification: List[Dict[str, Any]] = field(default_factory=list)
    # Direction-level four-phase diagnostics (read-only; see
    # ``region.direction_phase``).  The legacy ``phase_timeline`` above is
    # kept unchanged for before/after comparison.
    direction_signal_timeline: List[Dict[str, Any]] = field(
        default_factory=list)
    axis_phase_timeline: List[Dict[str, Any]] = field(default_factory=list)
    track_traffic_states: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": "world",
            "traffic_light_enabled": bool(self.traffic_light_enabled),
            "bounds": {
                "xmin": round(float(self.bounds[0]), 4),
                "ymin": round(float(self.bounds[1]), 4),
                "xmax": round(float(self.bounds[2]), 4),
                "ymax": round(float(self.bounds[3]), 4),
            },
            "resolution": float(self.resolution),
            "groups": self.groups,
            "stop_events": self.stop_events,
            "phase_timeline": self.phase_timeline,
            "direction_signal_timeline": self.direction_signal_timeline,
            "axis_phase_timeline": self.axis_phase_timeline,
            "track_traffic_states": self.track_traffic_states,
            "track_classification": self.track_classification,
            "diagnostics": self.diagnostics,
            "config": self.config,
        }


def _robust_track_stats(items: DynamicTrack,
                        config: TrafficLightConfig) -> Optional[Dict[str, Any]]:
    normalized = _normalize_track(items)
    if len(normalized) < int(config.min_observations):
        return None
    times = np.asarray([item["timestamp"] for item in normalized],
                       dtype=np.float64)
    points = np.asarray([item["world"] for item in normalized],
                        dtype=np.float64)
    intervals = np.diff(times)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    valid = intervals > 1e-3
    if not np.any(valid):
        return None
    speeds = steps[valid] / intervals[valid]
    path_length = float(np.sum(steps))
    net = float(np.linalg.norm(points[-1] - points[0]))
    if path_length < float(config.min_path_length):
        return None
    features = _turn_features(points, times, config)
    # U-turns have a low net/path ratio by construction; keep them because the
    # reviewed rule treats a U-turn as a left turn.
    if (net / max(path_length, 1e-9) < float(config.min_net_ratio)
            and not features["is_uturn"]):
        return None
    if float(np.percentile(speeds, 90)) < float(config.min_p90_speed):
        return None
    scores = _movement_scores(features, False, config)
    raw_movement = _pick_raw_movement(scores)
    movement = _group_label(raw_movement)
    stable_start, stable_end, stable_heading, stable_point = _stable_segment(
        points, times, config)
    return {
        "movement": movement,
        "raw_movement": raw_movement,
        "final_movement": movement,
        "class_name": str(normalized[0].get("class_name", "")),
        "features": features,
        "scores": scores,
        "start_point": points[0],
        "end_point": points[-1],
        "start_heading": float(features["entry_heading"]),
        "end_heading": float(features["exit_heading"]),
        "heading_change_deg": round(float(features["net_turn_deg"]), 3),
        "path_length": round(path_length, 3),
        "net": round(net, 3),
        "p90_speed": round(float(np.percentile(speeds, 90)), 3),
        "duration": round(float(times[-1] - times[0]), 3),
        "normalized": normalized,
        "stable_indices": (int(stable_start), int(stable_end)),
        "stable_points": points[int(stable_start):int(stable_end) + 1],
        "stable_heading": float(stable_heading),
        "stable_point": np.asarray(stable_point, dtype=np.float64),
        "waiting_left": False,
        "reclassified": False,
        "reclassify_reason": "",
        "entry_lane_id": None,
        "adjacent_lane_id": None,
    }


def _detect_stop_events(track_id: int, stats: Mapping[str, Any],
                         config: TrafficLightConfig) -> List[Dict[str, Any]]:
    events = []
    for start, end, position in _stop_intervals(stats, config):
        events.append({
            "track_id": int(track_id),
            "movement": str(stats["movement"]),
            "raw_movement": str(stats.get("raw_movement",
                                          stats["movement"])),
            "waiting_left": bool(stats.get("waiting_left", False)),
            "start_timestamp": float(start),
            "end_timestamp": float(end),
            "duration": round(float(end - start), 3),
            "position": [round(float(x), 4) for x in position],
        })
    return events


def _estimate_stop_line_forward(
        first_forwards: Sequence[float],
        all_forwards: Sequence[float],
        config: TrafficLightConfig,
) -> Optional[float]:
    """Robust stop-line estimate from per-track first-stop coordinates.

    The first stop of a track is the closest observation to its arrival at
    the queue / stop line.  Later stops may be queue creep or waiting-area
    creep and are deliberately ignored.  ``max(first_forwards)`` is used when
    the first stops cluster together; if one track's first stop is much
    further downstream than the median (a left-turn vehicle entering the
    waiting area without stopping at the line), a percentile is used instead.
    """

    first = sorted(float(value) for value in first_forwards)
    all_values = sorted(float(value) for value in all_forwards)
    if not first:
        return (float(np.median(all_values)) if all_values else None)
    if len(first) == 1:
        return float(first[0])
    median = float(np.median(first))
    maximum = float(first[-1])
    if maximum - median <= float(config.stop_line_first_outlier_m):
        return maximum
    return float(np.percentile(first, float(config.stop_line_first_percentile)))


def _build_direction_context(
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        config: TrafficLightConfig,
) -> List[Dict[str, Any]]:
    """Build direction records for the direction-level phase diagnostic.

    ``_build_direction_lanes`` deliberately drops directions with fewer than
    ``min_direction_tracks`` tracks.  The phase model must not: isolated
    tracks still carry signal evidence.  This function re-runs the same
    deterministic ``_cluster_directions`` call and attaches the direction
    geometry / stop-line context to every track.
    """

    entries = [(int(track_id), stats) for track_id, stats in classified]
    if not entries:
        return []
    groups = _cluster_directions(entries, config)
    directions: List[Dict[str, Any]] = []
    for direction_index, group in enumerate(groups):
        sin_sum = sum(math.sin(float(item[1]["stable_heading"]))
                      for item in group)
        cos_sum = sum(math.cos(float(item[1]["stable_heading"]))
                      for item in group)
        mean_heading = math.atan2(sin_sum, cos_sum)
        forward = np.asarray([math.cos(mean_heading), math.sin(mean_heading)],
                             dtype=np.float64)
        right = np.asarray([math.sin(mean_heading), -math.cos(mean_heading)],
                           dtype=np.float64)
        origin = np.median(np.asarray(
            [item[1]["stable_point"] for item in group],
            dtype=np.float64), axis=0)
        track_ids = [int(item[0]) for item in group]
        movement_track_ids: Dict[str, List[int]] = {
            "straight": [], "left": [], "right": []}
        first_forwards: List[float] = []
        all_forwards: List[float] = []
        exit_forwards: List[float] = []
        for track_id, stats in group:
            movement = _group_label(str(stats.get("movement", "straight")))
            if movement not in movement_track_ids:
                movement = "straight"
            movement_track_ids[movement].append(int(track_id))
            first: Optional[float] = None
            for _start, _end, position in _stop_intervals(stats, config):
                value = float((np.asarray(position, dtype=np.float64)
                               - origin) @ forward)
                all_forwards.append(value)
                if first is None:
                    first = value
            if first is not None:
                first_forwards.append(float(first))
            points = np.asarray([item["world"]
                                 for item in stats["normalized"]],
                                dtype=np.float64)
            if len(points):
                exit_forwards.append(float((points[-1] - origin) @ forward))
        stop_line = _estimate_stop_line_forward(
            first_forwards, all_forwards, config)
        directions.append({
            "direction_id": int(direction_index),
            "heading": float(mean_heading),
            "forward": forward,
            "right": right,
            "origin": origin,
            "track_ids": track_ids,
            "movement_track_ids": movement_track_ids,
            "stop_line_forward": stop_line,
            "stop_line_forward_median": (
                float(np.median(all_forwards)) if all_forwards else None),
            "exit_forward": (float(np.median(exit_forwards))
                             if exit_forwards else None),
            "stop_event_count": len(all_forwards),
            "first_stop_forwards": [round(float(value), 4)
                                    for value in first_forwards],
        })
    return directions


def _cluster_groups(
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        config: TrafficLightConfig,
) -> List[Dict[str, Any]]:
    """Group tracks by movement type.

    The reviewed global rule is intersection-level: all straight movements
    share one phase, all left movements share the mutually exclusive phase,
    and right turns are always allowed.  Entry/exit positions are kept only as
    diagnostics; they are not used to split the phase groups.
    """

    by_movement: Dict[str, Dict[str, Any]] = {}
    for track_id, stats in classified:
        movement = str(stats["movement"])
        group = by_movement.get(movement)
        if group is None:
            group = {
                "movement": movement,
                "track_ids": [],
                "entry_points": [],
                "exit_points": [],
                "points": [],
            }
            by_movement[movement] = group
        group["track_ids"].append(int(track_id))
        group["entry_points"].append(np.asarray(stats["start_point"]))
        group["exit_points"].append(np.asarray(stats["end_point"]))
        group["points"].extend(
            [item["world"] for item in stats["normalized"]])
    kept = []
    for movement, group in sorted(by_movement.items()):
        if len(group["track_ids"]) < int(config.min_group_tracks):
            continue
        entry_center = np.mean(np.asarray(group["entry_points"]), axis=0)
        exit_center = np.mean(np.asarray(group["exit_points"]), axis=0)
        points = np.asarray(group["points"], dtype=np.float64)
        if len(points) >= 3:
            hull = cv2.convexHull(points.astype(np.float32))
            polygon = [(round(float(x), 4), round(float(y), 4))
                       for x, y in hull.reshape(-1, 2)]
        else:
            polygon = []
        group_record = {
            "group_id": f"{movement}_0",
            "movement": movement,
            "track_ids": group["track_ids"],
            "track_count": len(group["track_ids"]),
            "entry_center": [round(float(x), 4) for x in entry_center],
            "exit_center": [round(float(x), 4) for x in exit_center],
            "polygon": polygon,
            "area_m2": _polygon_area(polygon) if polygon else 0.0,
        }
        centroid = _polygon_centroid(polygon) if polygon else (
            float(entry_center[0]), float(entry_center[1]))
        group_record["centroid"] = [centroid[0], centroid[1]]
        kept.append(group_record)
    return kept


def _infer_phase_timeline(
        groups: Sequence[Mapping[str, Any]],
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        stop_events: Sequence[Mapping[str, Any]],
        config: TrafficLightConfig,
) -> List[Dict[str, Any]]:
    all_times = [
        item["timestamp"]
        for _track_id, stats in classified
        for item in stats["normalized"]
    ]
    if not all_times:
        return []
    start_time = min(all_times)
    end_time = max(all_times)
    bin_seconds = float(config.phase_bin_sec)
    bin_count = max(1, int(math.ceil((end_time - start_time) / bin_seconds)))
    movement_exists = {str(group["movement"]) for group in groups}
    stop_intervals: Dict[str, List[Tuple[float, float]]] = {
        "straight": [], "left": []}
    for event in stop_events:
        movement = str(event.get("movement", ""))
        if movement in stop_intervals:
            stop_intervals[movement].append((
                float(event["start_timestamp"]),
                float(event["end_timestamp"]),
            ))

    # Reviewed global rule: straight and left are mutually exclusive; right is
    # always allowed.  A straight stop event means straight red / left green
    # (and vice versa).  Right-turn stops are treated as yielding, not red.
    timeline: List[Dict[str, Any]] = []
    current_state = {"straight": True, "left": False,
                     "right": bool(config.right_turn_always_green)}
    previous_key = None
    for bin_index in range(bin_count):
        timestamp = start_time + bin_index * bin_seconds
        straight_stopped = any(
            start <= timestamp <= end
            for start, end in stop_intervals["straight"])
        left_stopped = any(
            start <= timestamp <= end
            for start, end in stop_intervals["left"])
        state = dict(current_state)
        observed = straight_stopped or left_stopped
        if straight_stopped and not left_stopped and "left" in movement_exists:
            state["straight"] = False
            state["left"] = True
        elif left_stopped and not straight_stopped and "straight" in movement_exists:
            state["straight"] = True
            state["left"] = False
        # If both stop or neither stops, keep the previous mutually exclusive
        # state; the fixed rule forbids simultaneous straight/left green.
        state["right"] = bool(config.right_turn_always_green)
        current_state = dict(state)
        key = (state["straight"], state["left"], state["right"])
        if previous_key is None or key != previous_key:
            if timeline:
                timeline[-1]["end_timestamp"] = round(
                    start_time + bin_index * bin_seconds, 4)
            timeline.append({
                "start_timestamp": round(
                    start_time + bin_index * bin_seconds, 4),
                "end_timestamp": None,
                "green": [name for name in ("straight", "left", "right")
                          if state[name]],
                "source": "observed" if observed else "default_hold",
            })
            previous_key = key
    if timeline:
        timeline[-1]["end_timestamp"] = round(end_time, 4)
    return timeline


def build_traffic_light_model(
        dynamic_tracks: Mapping[int, DynamicTrack],
        *,
        reference_points: Any = None,
        config: TrafficLightConfig = TrafficLightConfig(),
) -> TrafficLightResult:
    """Classify tracks, cluster movement groups, and infer a phase timeline."""

    reference = np.asarray(reference_points, dtype=np.float64).reshape(-1, 2) \
        if reference_points is not None else np.zeros((0, 2), dtype=np.float64)
    classified: List[Tuple[int, Dict[str, Any]]] = []
    all_points = [reference]
    non_motor_vehicle_tracks = 0
    non_motor_vehicle_class_counts: Dict[str, int] = {}
    for track_id, items in dynamic_tracks.items():
        stats = _robust_track_stats(items, config)
        if stats is None:
            continue
        all_points.append(np.asarray([item["world"]
                                      for item in stats["normalized"]],
                                     dtype=np.float64))
        if not _is_motor_vehicle_class(stats.get("class_name", "")):
            non_motor_vehicle_tracks += 1
            key = str(stats.get("class_name", "") or "unknown")
            non_motor_vehicle_class_counts[key] = (
                non_motor_vehicle_class_counts.get(key, 0) + 1)
            continue
        classified.append((int(track_id), stats))

    if not classified:
        # No track passed the robust motion gate (short / slow / jittery
        # tracks only).  Skip the direction / phase / traffic-light model
        # entirely: queue and phase stitching then see empty directions and
        # states and perform no merges.
        return TrafficLightResult(
            bounds=_compute_bounds(all_points, config.bounds_margin),
            resolution=float(config.resolution),
            traffic_light_enabled=False,
            diagnostics={
                "traffic_light_enabled": False,
                "gate_passed": False,
                "gating_enabled": bool(getattr(config, "enable_gating", False)),
                "tracks_total": len(dynamic_tracks),
                "robust_tracks": 0,
                "motor_vehicle_tracks": 0,
                "non_motor_vehicle_tracks": int(non_motor_vehicle_tracks),
                "non_motor_vehicle_class_counts": (
                    non_motor_vehicle_class_counts),
                "movement_counts": {"straight": 0, "left": 0, "right": 0},
                "raw_movement_counts": {
                    "straight": 0, "left": 0, "right": 0, "uturn": 0,
                    "lane_change": 0, "waiting_left": 0,
                },
                "groups": 0,
                "stop_events": 0,
                "phase_intervals": 0,
                "legacy_phase_flip_count": 0,
                "axis_count": 0,
                "direction_phase": {
                    "axes": [],
                    "directions": [],
                    "direction_signal_timeline": [],
                    "axis_phase_timeline": [],
                    "axis_phase_state_counts": {},
                    "axis_phase_flip_count": 0,
                    "axis_conflict_bins": {},
                    "axis_conflict_count": 0,
                    "track_traffic_state_counts": {},
                    "movement_lane_mismatch_count": 0,
                    "grid": None,
                },
            },
            config=config.to_dict(),
        )

    lane_diagnostics = _apply_lane_context(classified, config)
    groups = _cluster_groups(classified, config)
    stop_events = [
        event
        for track_id, stats in classified
        for event in _detect_stop_events(track_id, stats, config)
    ]
    # Direction-level four-phase diagnostics.  Read-only: it adds fields to
    # the result/diagnostics and never changes movement labels or tracking.
    direction_context = _build_direction_context(classified, config)
    stop_intervals_by_track = {
        int(track_id): _stop_intervals(stats, config)
        for track_id, stats in classified
    }
    direction_phase = build_direction_phase_diagnostics(
        classified, direction_context, stop_intervals_by_track, config)
    traffic_states_by_track = direction_phase["track_traffic_states"]
    movement_counts = {
        movement: sum(1 for _tid, stats in classified
                      if stats["movement"] == movement)
        for movement in ("straight", "left", "right")
    }
    raw_movement_counts = {
        movement: sum(1 for _tid, stats in classified
                      if stats.get("raw_movement") == movement)
        for movement in ("straight", "left", "right", "uturn",
                         "lane_change", "waiting_left")
    }
    robust_total = len(classified)
    straight_fraction = (movement_counts["straight"]
                         / max(robust_total, 1))
    enable_checks = {
        "robust_tracks": robust_total,
        "straight_tracks": movement_counts["straight"],
        "left_tracks": movement_counts["left"],
        "raw_left_tracks": raw_movement_counts["left"],
        "raw_uturn_tracks": raw_movement_counts["uturn"],
        "waiting_left_tracks": raw_movement_counts["waiting_left"],
        "lane_change_tracks": raw_movement_counts["lane_change"],
        "stop_events": len(stop_events),
        "straight_fraction": round(float(straight_fraction), 4),
        "min_robust_tracks": int(config.min_robust_tracks),
        "min_straight_tracks": int(config.min_straight_tracks),
        "min_left_tracks": int(config.min_left_tracks),
        "min_stop_events": int(config.min_stop_events),
        "max_straight_fraction": float(config.max_straight_fraction),
    }
    gate_passed = bool(
        robust_total >= int(config.min_robust_tracks)
        and movement_counts["straight"] >= int(config.min_straight_tracks)
        and movement_counts["left"] >= int(config.min_left_tracks)
        and len(stop_events) >= int(config.min_stop_events)
        and straight_fraction <= float(config.max_straight_fraction)
    )
    # Reviewed decision 2026-09-09: do not pre-filter clips; run the phase
    # logic everywhere and rely on the per-track cross-validation instead.
    # The old gate remains available through ``enable_gating=True``.
    traffic_light_enabled = bool(
        gate_passed or not bool(getattr(config, "enable_gating", False)))
    phase_timeline = (
        _infer_phase_timeline(groups, classified, stop_events, config)
        if traffic_light_enabled else [])
    bounds = _compute_bounds(all_points, config.bounds_margin)
    track_classification = []
    movement_lane_mismatch = 0
    for track_id, stats in classified:
        features = stats["features"]
        lane = stats.get("_entry_lane")
        if (lane is not None
                and str(lane.get("movement")) != str(stats["movement"])):
            movement_lane_mismatch += 1
        track_classification.append({
            "track_id": track_id,
            "class_name": stats.get("class_name", ""),
            "movement": stats["movement"],
            "raw_movement": stats.get("raw_movement", stats["movement"]),
            "reclassified": bool(stats.get("reclassified", False)),
            "reclassify_reason": stats.get("reclassify_reason", ""),
            "heading_change_deg": stats["heading_change_deg"],
            "net_turn_deg": round(float(features["net_turn_deg"]), 3),
            "cum_turn_deg": round(float(features["cum_turn_deg"]), 3),
            "turn_consistency": round(
                float(features["turn_consistency"]), 3),
            "lateral_offset_m": round(
                float(features["lateral_offset_m"]), 3),
            "sign_changes": int(features["sign_changes"]),
            "lane_change": bool(features["lane_change"]),
            "heading_flip": bool(features["heading_flip"]),
            "uturn": bool(features["is_uturn"]),
            "turn_in_place": bool(features.get("turn_in_place", False)),
            "heading_inconsistent": bool(
                features.get("heading_inconsistent", False)),
            "heading_rate_max_deg_per_m": round(
                float(features.get("heading_rate_max_deg_per_m", 0.0)), 3),
            "waiting_left": bool(stats.get("waiting_left", False)),
            "entry_lane_id": stats.get("entry_lane_id"),
            "adjacent_lane_id": stats.get("adjacent_lane_id"),
            "direction_id": stats.get("phase_direction_id"),
            "axis_id": stats.get("phase_axis_id"),
            "stop_line_forward": (
                None if stats.get("phase_stop_line_forward") is None
                else round(float(stats["phase_stop_line_forward"]), 4)),
            "movement_lane_consistent": (
                stats.get("_entry_lane") is None
                or str(stats["_entry_lane"].get("movement"))
                == str(stats["movement"])),
            "traffic_state_counts": dict(
                traffic_states_by_track.get(int(track_id), {}).get(
                    "counts", {})),
            "traffic_state_intervals": list(
                traffic_states_by_track.get(int(track_id), {}).get(
                    "intervals", [])),
            "path_length": stats["path_length"],
            "net": stats["net"],
            "p90_speed": stats["p90_speed"],
            "duration": stats["duration"],
        })
    direction_records: List[Dict[str, Any]] = []
    for record in direction_phase["directions"]:
        movement_ids = record.get("movement_track_ids", {})
        direction_records.append({
            "direction_id": int(record["direction_id"]),
            "axis_id": int(record["axis_id"]),
            "axis_sign": int(record["axis_sign"]),
            "heading_deg": round(math.degrees(float(record["heading"])), 3),
            "forward": [round(float(value), 4)
                        for value in record["forward"]],
            "right": [round(float(value), 4)
                      for value in record["right"]],
            "origin": [round(float(value), 4)
                       for value in record["origin"]],
            "track_count": len(record["track_ids"]),
            "movement_counts": {
                key: len(value) for key, value in movement_ids.items()},
            "stop_line_forward": (
                None if record.get("stop_line_forward") is None
                else round(float(record["stop_line_forward"]), 4)),
            "stop_line_forward_median": (
                None if record.get("stop_line_forward_median") is None
                else round(float(record["stop_line_forward_median"]), 4)),
            "exit_forward": (
                None if record.get("exit_forward") is None
                else round(float(record["exit_forward"]), 4)),
            "stop_event_count": int(record.get("stop_event_count", 0)),
            "first_stop_forwards": list(
                record.get("first_stop_forwards", [])),
        })
    axis_records: List[Dict[str, Any]] = []
    for axis in direction_phase["axes"]:
        axis_records.append({
            "axis_id": int(axis["axis_id"]),
            "direction_ids": list(axis["direction_ids"]),
            "reference_direction_id": int(axis["reference_direction_id"]),
            "heading_deg": round(math.degrees(float(axis["heading"])), 3),
            "forward": [round(float(value), 4)
                        for value in axis["forward"]],
            "track_count": int(axis["track_count"]),
        })
    traffic_state_counts: Dict[str, int] = {}
    for item in traffic_states_by_track.values():
        for state, count in item.get("counts", {}).items():
            traffic_state_counts[str(state)] = (
                traffic_state_counts.get(str(state), 0) + int(count))
    diagnostics = {
        "traffic_light_enabled": traffic_light_enabled,
        "enable_checks": enable_checks,
        "gate_passed": gate_passed,
        "gating_enabled": bool(getattr(config, "enable_gating", False)),
        "tracks_total": len(dynamic_tracks),
        "robust_tracks": robust_total,
        "motor_vehicle_tracks": robust_total,
        "non_motor_vehicle_tracks": int(non_motor_vehicle_tracks),
        "non_motor_vehicle_class_counts": non_motor_vehicle_class_counts,
        "movement_counts": movement_counts,
        "raw_movement_counts": raw_movement_counts,
        "groups": len(groups),
        "stop_events": len(stop_events),
        "phase_intervals": len(phase_timeline),
        "legacy_phase_flip_count": max(0, len(phase_timeline) - 1),
        "axis_count": len(axis_records),
        "direction_phase": {
            "axes": axis_records,
            "directions": direction_records,
            "direction_signal_timeline": direction_phase[
                "direction_signal_timeline"],
            "axis_phase_timeline": direction_phase["axis_phase"]["timeline"],
            "axis_phase_state_counts": direction_phase["axis_phase"][
                "state_counts"],
            "axis_phase_flip_count": int(direction_phase["axis_phase"].get(
                "phase_flip_count", 0)),
            "axis_conflict_bins": direction_phase["axis_phase"].get(
                "axis_conflict_bins", {}),
            "axis_conflict_count": int(direction_phase["axis_phase"].get(
                "axis_conflict_count", 0)),
            "track_traffic_state_counts": traffic_state_counts,
            "movement_lane_mismatch_count": movement_lane_mismatch,
            "grid": direction_phase["grid"],
        },
        **lane_diagnostics,
    }
    return TrafficLightResult(
        bounds=bounds,
        resolution=config.resolution,
        traffic_light_enabled=traffic_light_enabled,
        groups=groups,
        stop_events=stop_events,
        phase_timeline=phase_timeline,
        direction_signal_timeline=direction_phase[
            "direction_signal_timeline"],
        axis_phase_timeline=direction_phase["axis_phase"]["timeline"],
        track_traffic_states=[
            traffic_states_by_track[key]
            for key in sorted(traffic_states_by_track)],
        track_classification=track_classification,
        diagnostics=diagnostics,
        config=config.to_dict(),
    )


def save_traffic_light_json(result: TrafficLightResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def render_traffic_light_png(
        result: TrafficLightResult,
        path: Path,
        *,
        dynamic_tracks: Optional[Mapping[int, DynamicTrack]] = None,
        title: Optional[str] = None,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, (axis, timeline_axis) = plt.subplots(
        2, 1, figsize=(14, 10),
        gridspec_kw={"height_ratios": [3, 1]})

    movement_color = {"straight": "#1f77b4", "left": "#2ca02c",
                      "right": "#ff7f0e", "uturn": "#d62728"}
    movement_label = {"straight": "straight", "left": "left",
                      "right": "right", "uturn": "U-turn(left)"}
    labelled = set()
    if dynamic_tracks:
        classification = {int(item["track_id"]): item
                          for item in result.track_classification}
        for track_id, items in dynamic_tracks.items():
            stats = classification.get(int(track_id))
            if stats is None:
                continue
            movement = stats["movement"]
            points = np.asarray([item["world"] for item in _normalize_track(items)],
                                dtype=np.float64)
            if stats.get("reclassified"):
                axis.plot(points[:, 0], points[:, 1],
                          color="magenta", linewidth=1.5,
                          linestyle="--", alpha=0.95,
                          label=("reclassified track"
                                 if "reclassified" not in labelled else None))
                labelled.add("reclassified")
            else:
                axis.plot(points[:, 0], points[:, 1],
                          color=movement_color.get(movement, "0.7"),
                          linewidth=1.0, alpha=0.85,
                          label=(movement_label.get(movement, movement)
                                 if movement not in labelled else None))
                labelled.add(movement)
            if stats.get("reclassified"):
                axis.text(points[-1, 0], points[-1, 1],
                          f"id{track_id}", fontsize=6, color="magenta",
                          zorder=4)
            if stats.get("waiting_left"):
                axis.scatter(points[-1, 0], points[-1, 1], s=70,
                             facecolors="none", edgecolors="cyan",
                             linewidths=1.6, zorder=4,
                             label=("waiting-left track"
                                    if "waiting-left" not in labelled else None))
                axis.text(points[-1, 0], points[-1, 1] + 1.0,
                          f"id{track_id}", fontsize=6, color="cyan",
                          zorder=4)
                labelled.add("waiting-left")
            if stats.get("uturn"):
                axis.text(points[-1, 0], points[-1, 1] - 1.0,
                          f"U{track_id}", fontsize=6, color="#d62728",
                          zorder=4)

    # Lane anchors: helps verify lane selection against ID fragments.
    for lane in result.diagnostics.get("lane_groups", []):
        center = np.asarray(lane["center"], dtype=np.float64)
        heading = math.radians(float(lane["heading_deg"]))
        direction = np.asarray([math.cos(heading), math.sin(heading)],
                               dtype=np.float64)
        start = center - 5.0 * direction
        end = center + 5.0 * direction
        axis.plot([start[0], end[0]], [start[1], end[1]],
                  color="0.45", linewidth=0.8, alpha=0.6, zorder=1)
        axis.scatter(center[0], center[1], s=12, c="black", marker="s",
                     alpha=0.7, zorder=2)
        axis.text(center[0], center[1], str(lane["lane_id"]),
                  fontsize=6, color="0.25", zorder=3)

    if result.stop_events:
        stops = np.asarray([[event["position"][0], event["position"][1]]
                            for event in result.stop_events], dtype=np.float64)
        axis.scatter(stops[:, 0], stops[:, 1], s=30, c="red", marker="x",
                     linewidths=1.0, label="stop event")

    xmin, ymin, xmax, ymax = result.bounds
    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    counts = result.diagnostics.get("movement_counts", {})
    axis.set_title(
        (title or "traffic-light movement inference")
        + f"  |  enabled={result.traffic_light_enabled}"
        + f"  straight={counts.get('straight', 0)}"
        + f" left={counts.get('left', 0)}"
        + f" right={counts.get('right', 0)}"
        + f" stops={len(result.stop_events)}"
        + f" waiting_left={result.diagnostics.get('waiting_left_tracks', 0)}"
        + f" reclassified={result.diagnostics.get('reclassified_tracks', 0)}")
    axis.legend(loc="best", fontsize=8)

    if result.traffic_light_enabled and result.phase_timeline:
        row = {"straight": 2, "left": 1, "right": 0}
        for interval in result.phase_timeline:
            start = float(interval["start_timestamp"])
            end = float(interval.get("end_timestamp") or start)
            for name in interval["green"]:
                timeline_axis.barh(row.get(name, 0), max(end - start, 1e-3),
                                   left=start, height=0.8,
                                   color=movement_color[name], alpha=0.7)
        timeline_axis.set_yticks([2, 1, 0])
        timeline_axis.set_yticklabels(["straight", "left", "right"])
        timeline_axis.set_xlim(
            float(result.phase_timeline[0]["start_timestamp"]),
            float(result.phase_timeline[-1]["end_timestamp"]))
        timeline_axis.set_xlabel("timestamp [s]")
        timeline_axis.set_title("inferred phase timeline")
    else:
        timeline_axis.axis("off")
        timeline_axis.text(0.5, 0.5,
                           "traffic-light logic disabled: not enough "
                           "sustained dynamic / straight-left evidence",
                           ha="center", va="center", fontsize=11)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
