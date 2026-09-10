#!/usr/bin/env python3
"""Direction-level four-phase traffic-signal diagnostics.

This is a read-only analysis layer on top of the existing movement / lane
model in :mod:`region.traffic_light`.  It does **not** change tracking
behaviour.  The reviewed signal plan is the standard Chinese four-phase plan:

    axis A straight -> axis A left -> axis B straight -> axis B left

with right turns always allowed and permissive left turns allowed during the
same axis' straight phase.  A left-turn vehicle may enter the waiting area
while its axis straight phase is green; it then yields to the opposing
straight movement and is released by the protected left phase.

The old global straight/left state machine in ``traffic_light.py`` cannot
express this plan: one stopped straight vehicle in one approach flips the
whole intersection to "left green", and a stopped left vehicle flips it back.
This module builds the missing structure:

* pair opposite travel directions into axes;
* collect per-direction / per-movement motion evidence on a time grid;
* infer a per-direction signal state (green / red / permissive / unknown);
* infer one globally consistent four-phase timeline with hysteresis;
* assign a per-observation ``traffic_state`` to every robust track:
  ``moving / waiting_red / waiting_left_area / yielding / waiting_queue /
  uncertain``.

Everything is emitted as plain JSON-compatible dicts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _cfg(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def _json_point(value: np.ndarray) -> List[float]:
    return [round(float(value[0]), 4), round(float(value[1]), 4)]


def _round(value: Any, digits: int = 4) -> float:
    return round(float(value), digits)


# ---------------------------------------------------------------------------
# axis pairing
# ---------------------------------------------------------------------------

def pair_directions_into_axes(
        directions: Sequence[Mapping[str, Any]],
        config: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Pair opposite travel directions into axes.

    ``_cluster_directions`` already keeps opposite headings separate.  For a
    standard four-leg intersection the four directions pair into two axes
    (NS / EW).  A direction without an opposite partner becomes a
    single-direction axis (T-junction, one-way road, partial clip).

    Returns ``(directions, axes)`` where the direction records are copies with
    ``axis_id`` / ``axis_sign`` filled in.
    """

    tolerance = math.radians(float(
        _cfg(config, "axis_pair_tolerance_deg", 45.0)))
    records = [dict(item) for item in directions]
    pair_costs: List[Tuple[float, int, int]] = []
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            difference = abs(_wrap_angle(
                float(records[left]["heading"])
                - float(records[right]["heading"])))
            cost = abs(math.pi - difference)
            if cost <= tolerance:
                pair_costs.append((float(cost), left, right))
    used: set[int] = set()
    pairs: List[Tuple[int, int]] = []
    for _cost, left, right in sorted(pair_costs):
        if left in used or right in used:
            continue
        used.update((left, right))
        pairs.append((left, right))

    axes: List[Dict[str, Any]] = []

    def _axis_forward(reference: Mapping[str, Any]) -> np.ndarray:
        heading = float(reference["heading"])
        return np.asarray([math.cos(heading), math.sin(heading)],
                          dtype=np.float64)

    for left, right in pairs:
        # Choose the direction whose heading is in the upper half plane as
        # the deterministic axis reference.  The two headings differ by pi,
        # so exactly one of them satisfies sin(heading) >= 0.
        if math.sin(float(records[left]["heading"])) >= 0.0:
            reference, opposite = left, right
        else:
            reference, opposite = right, left
        forward = _axis_forward(records[reference])
        axis_id = len(axes)
        records[reference]["axis_id"] = axis_id
        records[reference]["axis_sign"] = 1
        records[opposite]["axis_id"] = axis_id
        records[opposite]["axis_sign"] = -1
        axes.append({
            "axis_id": axis_id,
            "direction_ids": [int(records[reference]["direction_id"]),
                              int(records[opposite]["direction_id"])],
            "reference_direction_id": int(
                records[reference]["direction_id"]),
            "heading": _round(records[reference]["heading"], 6),
            "forward": _json_point(forward),
            "track_count": sum(
                len(records[index]["track_ids"]) for index in (left, right)),
        })

    for index, record in enumerate(records):
        if index in used:
            continue
        axis_id = len(axes)
        record["axis_id"] = axis_id
        record["axis_sign"] = 1
        forward = _axis_forward(record)
        axes.append({
            "axis_id": axis_id,
            "direction_ids": [int(record["direction_id"])],
            "reference_direction_id": int(record["direction_id"]),
            "heading": _round(record["heading"], 6),
            "forward": _json_point(forward),
            "track_count": len(record["track_ids"]),
        })
    axes.sort(key=lambda item: (
        math.atan2(float(item["forward"][1]), float(item["forward"][0]))
        % (2.0 * math.pi), item["axis_id"]))
    axis_id_map = {item["axis_id"]: index for index, item in enumerate(axes)}
    for record in records:
        record["axis_id"] = axis_id_map[int(record["axis_id"])]
    for axis in axes:
        axis["axis_id"] = axis_id_map[int(axis["axis_id"])]
    return records, axes


# ---------------------------------------------------------------------------
# time grid and per-track samples
# ---------------------------------------------------------------------------

def build_time_grid(
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        bin_sec: float,
) -> Optional[Dict[str, Any]]:
    times = [
        float(item["timestamp"])
        for _track_id, stats in classified
        for item in stats.get("normalized", [])
    ]
    if not times:
        return None
    start = min(times)
    end = max(times)
    bin_sec = max(float(bin_sec), 0.05)
    count = max(1, int(math.ceil((end - start) / bin_sec)))
    centers = start + (np.arange(count, dtype=np.float64) + 0.5) * bin_sec
    return {
        "start": float(start),
        "end": float(end),
        "bin_sec": float(bin_sec),
        "count": int(count),
        "centers": centers,
    }


def _bin_index(grid: Mapping[str, Any], timestamp: float) -> int:
    index = int(math.floor(
        (float(timestamp) - float(grid["start"])) / float(grid["bin_sec"])))
    return max(0, min(int(grid["count"]) - 1, index))


def collect_track_samples(
        stats: Mapping[str, Any],
        direction: Mapping[str, Any],
        stop_intervals: Sequence[Tuple[float, float, np.ndarray]],
        config: Any,
) -> Optional[Dict[str, Any]]:
    """Project one track onto its direction axis and estimate motion state."""

    normalized = list(stats.get("normalized", []))
    if len(normalized) < 2:
        return None
    origin = np.asarray(direction["origin"], dtype=np.float64)
    forward = np.asarray(direction["forward"], dtype=np.float64)
    right = np.asarray(direction["right"], dtype=np.float64)
    times = np.asarray([float(item["timestamp"]) for item in normalized],
                       dtype=np.float64)
    points = np.asarray([item["world"] for item in normalized],
                        dtype=np.float64)
    along = (points - origin) @ forward
    lateral = (points - origin) @ right
    intervals = np.diff(times)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speeds = np.zeros(len(times), dtype=np.float64)
    valid = intervals > 1e-3
    speeds[1:][valid] = steps[valid] / intervals[valid]
    if len(times) > 1:
        speeds[0] = speeds[1]
    stopped = np.zeros(len(times), dtype=bool)
    for start, end, _position in stop_intervals:
        stopped |= (times >= float(start) - 1e-3) & (
            times <= float(end) + 1e-3)
    return {
        "track_id": int(stats.get("track_id", -1)),
        "movement": str(stats.get("movement", "straight")),
        "raw_movement": str(stats.get("raw_movement",
                                      stats.get("movement", "straight"))),
        "waiting_left": bool(stats.get("waiting_left", False)),
        "direction_id": int(direction["direction_id"]),
        "axis_id": int(direction["axis_id"]),
        "times": times,
        "along": along,
        "lateral": lateral,
        "speeds": speeds,
        "stopped": stopped,
    }


# ---------------------------------------------------------------------------
# per-group evidence and signal state
# ---------------------------------------------------------------------------

def _fill_short_unknowns(states: List[str], gap: int) -> List[str]:
    if gap <= 0:
        return list(states)
    result = list(states)
    index = 0
    while index < len(result):
        if result[index] != "unknown":
            index += 1
            continue
        start = index
        while index < len(result) and result[index] == "unknown":
            index += 1
        length = index - start
        if (length <= gap and start > 0 and index < len(result)
                and result[start - 1] == result[index]):
            for offset in range(start, index):
                result[offset] = result[start - 1]
    return result


def _hold_unknowns(
        states: List[str],
        hold: int,
) -> Tuple[List[str], List[bool]]:
    """Hold the previous signal state through short evidence gaps.

    A traffic light does not turn "unknown"; when no track of a group is
    visible for a short interval the previous colour is the best estimate.
    Long gaps (more than ``hold`` bins) are left unknown instead of inventing
    a phase.
    """

    result = list(states)
    held = [False] * len(states)
    index = 0
    while index < len(result):
        if result[index] != "unknown":
            index += 1
            continue
        start = index
        while index < len(result) and result[index] == "unknown":
            index += 1
        length = index - start
        if start == 0 or length > hold:
            continue
        previous = result[start - 1]
        if previous == "unknown":
            continue
        for offset in range(start, index):
            result[offset] = previous
            held[offset] = True
    return result, held


def _majority_filter(states: List[str], window: int) -> List[str]:
    if window <= 1:
        return list(states)
    half = window // 2
    result: List[str] = []
    for index in range(len(states)):
        low = max(0, index - half)
        high = min(len(states), index + half + 1)
        counts: Dict[str, int] = {}
        for value in states[low:high]:
            counts[value] = counts.get(value, 0) + 1
        known = {key: value for key, value in counts.items()
                 if key != "unknown"}
        if known:
            best = max(known, key=lambda key: (known[key],
                                               key != "unknown"))
            result.append(best)
        else:
            result.append("unknown")
    return result


def _hysteresis(states: List[str], min_bins: int) -> List[str]:
    if min_bins <= 1 or not states:
        return list(states)
    result: List[str] = [states[0]]
    current = states[0]
    candidate: Optional[str] = None
    candidate_count = 0
    for state in states[1:]:
        if state == current:
            candidate = None
            candidate_count = 0
            result.append(current)
            continue
        if candidate == state:
            candidate_count += 1
        else:
            candidate = state
            candidate_count = 1
        if candidate_count >= min_bins:
            current = state
            candidate = None
            candidate_count = 0
        result.append(current)
    return result


def _raw_group_state(
        movement: str,
        counts: Mapping[str, int],
) -> str:
    active = int(counts.get("active", 0))
    moving = int(counts.get("moving", 0))
    moving_past = int(counts.get("moving_past", 0))
    moving_before = int(counts.get("moving_before", 0))
    stopped = int(counts.get("stopped", 0))
    crossing = int(counts.get("crossing", 0))
    downstream = int(counts.get("downstream", 0))
    upstream = int(counts.get("upstream", 0))
    at_line = int(counts.get("at_line", 0))
    if active <= 0:
        return "unknown"
    # A vehicle crossing the stop line (or already moving downstream of it)
    # is the strongest green evidence.  Merely moving on the approach may be
    # queue creep, so it is not accepted as green.
    if crossing > 0 or moving_past > 0:
        return "green"
    if stopped > 0:
        if movement == "left" and downstream > 0 and upstream == 0 \
                and at_line == 0:
            return "waiting_area"
        return "red"
    if moving > 0:
        # Approach motion without a crossing is not enough to call green.
        return "uncertain"
    return "unknown"


def collect_group_evidence(
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        directions: Sequence[Mapping[str, Any]],
        stop_intervals_by_track: Mapping[int, Sequence[Tuple[float, float,
                                                              np.ndarray]]],
        grid: Mapping[str, Any],
        config: Any,
) -> Dict[str, Any]:
    """Build per-(direction, movement) motion evidence on the time grid."""

    count = int(grid["count"])
    bin_sec = float(grid["bin_sec"])
    start = float(grid["start"])
    direction_by_id = {int(item["direction_id"]): item for item in directions}
    stop_tolerance = float(_cfg(config, "stop_line_tolerance_m", 4.0))
    crossing_step = float(_cfg(config, "crossing_step_m", 2.0))
    moving_speed = float(_cfg(config, "moving_speed_threshold", 1.0))

    groups: Dict[str, Dict[str, Any]] = {}
    samples_by_track: Dict[int, Dict[str, Any]] = {}
    for track_id, stats in classified:
        direction_id = stats.get("phase_direction_id")
        if direction_id is None:
            continue
        direction = direction_by_id.get(int(direction_id))
        if direction is None:
            continue
        samples = collect_track_samples(
            stats, direction,
            stop_intervals_by_track.get(int(track_id), ()), config)
        if samples is None:
            continue
        samples["track_id"] = int(track_id)
        samples_by_track[int(track_id)] = samples
        movement = str(stats.get("movement", "straight"))
        if movement not in ("straight", "left", "right"):
            movement = "straight"
        key = f"{int(direction_id)}:{movement}"
        group = groups.get(key)
        if group is None:
            group = {
                "direction_id": int(direction_id),
                "movement": movement,
                "track_ids": [],
                "active": np.zeros(count, dtype=np.int32),
                "moving": np.zeros(count, dtype=np.int32),
                "moving_past": np.zeros(count, dtype=np.int32),
                "moving_before": np.zeros(count, dtype=np.int32),
                "stopped": np.zeros(count, dtype=np.int32),
                "at_line": np.zeros(count, dtype=np.int32),
                "upstream": np.zeros(count, dtype=np.int32),
                "downstream": np.zeros(count, dtype=np.int32),
                "crossing": np.zeros(count, dtype=np.int32),
            }
            groups[key] = group
        group["track_ids"].append(int(track_id))

        line = direction.get("stop_line_forward")
        times = samples["times"]
        along_values = samples["along"]
        speeds = samples["speeds"]
        sample_count = len(times)
        max_gap_sec = float(_cfg(config, "max_segment_gap_sec", 2.0))
        crossing_flags = _crossing_flags(
            times, along_values, line, stop_tolerance, crossing_step,
            max_gap_sec)
        green_before = float(_cfg(config, "crossing_green_before_sec", 0.5))
        green_after = float(_cfg(config, "crossing_green_after_sec", 2.0))
        green_flags = np.zeros(sample_count, dtype=bool)
        for index in np.flatnonzero(crossing_flags):
            crossing_time = float(times[index])
            green_flags |= ((times >= crossing_time - green_before)
                            & (times <= crossing_time + green_after))
        last_sample_bin = -1
        for sample_index, timestamp in enumerate(times):
            bin_index = _bin_index(grid, float(timestamp))
            speed = float(speeds[sample_index])
            along = float(along_values[sample_index])
            stopped = bool(samples["stopped"][sample_index]) or (
                speed <= float(_cfg(config, "stop_speed_threshold", 0.8)))
            moving = (not stopped) and speed > moving_speed
            # One sample per track per bin; prefer a moving sample.
            if bin_index != last_sample_bin:
                group["active"][bin_index] += 1
                last_sample_bin = bin_index
            if moving:
                group["moving"][bin_index] += 1
                if bool(green_flags[sample_index]):
                    group["moving_past"][bin_index] += 1
                else:
                    group["moving_before"][bin_index] += 1
            if stopped:
                group["stopped"][bin_index] += 1
                if line is None:
                    group["upstream"][bin_index] += 1
                elif along < float(line) - stop_tolerance:
                    group["upstream"][bin_index] += 1
                elif along > float(line) + stop_tolerance:
                    group["downstream"][bin_index] += 1
                else:
                    group["at_line"][bin_index] += 1
            if bool(crossing_flags[sample_index]):
                group["crossing"][bin_index] += 1
    return {"groups": groups, "samples_by_track": samples_by_track}


def _crossing_flags(
        times: np.ndarray,
        along_values: np.ndarray,
        line: Optional[float],
        tolerance: float,
        crossing_step: float,
        max_gap_sec: float,
) -> np.ndarray:
    """Mark samples where a track makes real progress across the stop line.

    A slow waiting-area creep may never move ``crossing_step`` in a single
    frame pair.  The check therefore accumulates progress since the last
    sample that was still clearly upstream of the line, which handles both
    fast crossings and slow queue/waiting-area creep.
    """

    count = len(times)
    flags = np.zeros(count, dtype=bool)
    if count < 2:
        return flags
    anchor = 0
    for index in range(1, count):
        delta_t = float(times[index] - times[index - 1])
        if delta_t <= 0.0 or delta_t > max_gap_sec:
            anchor = index
            continue
        if line is None:
            if float(along_values[index] - along_values[anchor]) >= crossing_step:
                flags[index] = True
                anchor = index
            continue
        if float(along_values[index]) < float(line) - tolerance:
            anchor = index
            continue
        if (float(along_values[index] - along_values[anchor])
                >= crossing_step):
            flags[index] = True
            anchor = index
    return flags


def _smooth_counts(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.astype(np.float64)
    kernel = np.ones(int(window), dtype=np.float64) / float(window)
    return np.convolve(values.astype(np.float64), kernel, mode="same")


def infer_group_signal_timeline(
        evidence: Mapping[str, Any],
        grid: Mapping[str, Any],
        config: Any,
) -> Dict[str, Dict[str, Any]]:
    """Infer per-(direction, movement) signal state intervals."""

    count = int(grid["count"])
    start = float(grid["start"])
    bin_sec = float(grid["bin_sec"])
    smooth_sec = float(_cfg(config, "direction_state_smooth_sec", 0.8))
    gap_sec = float(_cfg(config, "direction_state_gap_sec", 1.2))
    min_duration_sec = float(
        _cfg(config, "direction_state_min_duration_sec", 1.0))
    smooth_bins = max(1, int(round(smooth_sec / bin_sec)))
    gap_bins = max(1, int(round(gap_sec / bin_sec)))
    min_bins = max(1, int(round(min_duration_sec / bin_sec)))

    result: Dict[str, Dict[str, Any]] = {}
    for key, group in evidence["groups"].items():
        moving = _smooth_counts(group["moving"], smooth_bins)
        moving_past = _smooth_counts(group["moving_past"], smooth_bins)
        moving_before = _smooth_counts(group["moving_before"], smooth_bins)
        stopped = _smooth_counts(group["stopped"], smooth_bins)
        active = _smooth_counts(group["active"], smooth_bins)
        crossing = _smooth_counts(group["crossing"], smooth_bins)
        downstream = _smooth_counts(group["downstream"], smooth_bins)
        upstream = _smooth_counts(group["upstream"], smooth_bins)
        at_line = _smooth_counts(group["at_line"], smooth_bins)
        raw: List[str] = []
        for index in range(count):
            raw.append(_raw_group_state(
                str(group["movement"]),
                {
                    "active": int(round(active[index])),
                    "moving": int(round(moving[index])),
                    "moving_past": int(round(moving_past[index])),
                    "moving_before": int(round(moving_before[index])),
                    "stopped": int(round(stopped[index])),
                    "crossing": int(round(crossing[index])),
                    "downstream": int(round(downstream[index])),
                    "upstream": int(round(upstream[index])),
                    "at_line": int(round(at_line[index])),
                },
            ))
        hold_sec = float(_cfg(config, "direction_state_hold_sec", 3.0))
        hold_bins = max(1, int(round(hold_sec / bin_sec)))
        held_states, _held_flags = _hold_unknowns(raw, hold_bins)
        filled = _fill_short_unknowns(held_states, gap_bins)
        filtered = _majority_filter(filled, smooth_bins)
        states = _hysteresis(filtered, min_bins)

        intervals: List[Dict[str, Any]] = []
        for index, state in enumerate(states):
            if not intervals or intervals[-1]["state"] != state:
                intervals.append({
                    "state": state,
                    "start_index": index,
                    "end_index": index,
                    "start_timestamp": _round(start + index * bin_sec),
                    "end_timestamp": None,
                })
            else:
                intervals[-1]["end_index"] = index
        for interval in intervals:
            interval["end_timestamp"] = _round(
                start + (int(interval["end_index"]) + 1) * bin_sec)
            interval["duration"] = _round(
                float(interval["end_timestamp"])
                - float(interval["start_timestamp"]))
            low = int(interval["start_index"])
            high = int(interval["end_index"]) + 1
            interval["source"] = (
                "observed" if any(raw[index] != "unknown"
                                  for index in range(low, high))
                else "held")
        counts: Dict[str, int] = {}
        for state in states:
            counts[state] = counts.get(state, 0) + 1
        result[key] = {
            "direction_id": int(group["direction_id"]),
            "movement": str(group["movement"]),
            "track_ids": list(group["track_ids"]),
            "states": states,
            "intervals": intervals,
            "counts": counts,
            "evidence_totals": {
                "moving": int(np.sum(group["moving"])),
                "moving_past": int(np.sum(group["moving_past"])),
                "moving_before": int(np.sum(group["moving_before"])),
                "stopped": int(np.sum(group["stopped"])),
                "at_line": int(np.sum(group["at_line"])),
                "upstream": int(np.sum(group["upstream"])),
                "downstream": int(np.sum(group["downstream"])),
                "crossing": int(np.sum(group["crossing"])),
            },
        }
    return result


# ---------------------------------------------------------------------------
# four-phase timeline
# ---------------------------------------------------------------------------

def _phase_definitions(
        axes: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Two-phase-per-axis definitions for the two strongest axes."""

    ranked = sorted(axes, key=lambda item: (
        -int(item.get("track_count", 0)), int(item.get("axis_id", 0))))
    selected = ranked[:2]
    phases: List[Dict[str, Any]] = []
    for axis in selected:
        phases.append({
            "phase": f"axis_{int(axis['axis_id'])}_straight",
            "axis_id": int(axis["axis_id"]),
            "movement": "straight",
        })
        phases.append({
            "phase": f"axis_{int(axis['axis_id'])}_left",
            "axis_id": int(axis["axis_id"]),
            "movement": "left",
        })
    return phases


def _phase_allows(
        phase: Mapping[str, Any],
        direction: Mapping[str, Any],
        movement: str,
        config: Any,
) -> Tuple[bool, str]:
    # Right turns are always allowed, on every axis, in every phase.
    if movement == "right":
        return True, "always"
    axis_id = int(direction.get("axis_id", -1))
    if axis_id != int(phase["axis_id"]):
        return False, "red"
    if movement == str(phase["movement"]):
        return True, "protected" if movement == "left" else "green"
    if (movement == "left" and str(phase["movement"]) == "straight"
            and bool(_cfg(config, "left_permissive_enabled", True))):
        return True, "permissive"
    return False, "red"


def _phase_emission(
        phase: Mapping[str, Any],
        directions: Sequence[Mapping[str, Any]],
        group_signals: Mapping[str, Mapping[str, Any]],
        bin_index: int,
        config: Any,
) -> float:
    score = 0.0
    for direction in directions:
        direction_id = int(direction["direction_id"])
        for movement in ("straight", "left"):
            key = f"{direction_id}:{movement}"
            signal = group_signals.get(key)
            if signal is None:
                continue
            state = signal["states"][bin_index]
            if state == "unknown":
                continue
            allowed, mode = _phase_allows(phase, direction, movement, config)
            if allowed:
                if state == "green":
                    score += 1.0
                elif state == "red":
                    score -= 1.0
                elif state == "waiting_area":
                    score += 0.5 if mode == "permissive" else -0.5
                elif state == "uncertain":
                    score += 0.1
            else:
                if state == "green":
                    score -= 1.0
                elif state == "red":
                    score += 1.0
                elif state == "waiting_area":
                    score += 0.3 if movement == "left" else -0.3
                elif state == "uncertain":
                    score -= 0.1
    return score


def infer_axis_phase_timeline(
        directions: Sequence[Mapping[str, Any]],
        axes: Sequence[Mapping[str, Any]],
        group_signals: Mapping[str, Mapping[str, Any]],
        grid: Mapping[str, Any],
        config: Any,
) -> Dict[str, Any]:
    """Viterbi over the standard four-phase cycle plus an ``unknown`` state."""

    phases = _phase_definitions(axes)
    count = int(grid["count"])
    start = float(grid["start"])
    bin_sec = float(grid["bin_sec"])
    if not phases:
        return {"phases": [], "timeline": [], "state_counts": {}}
    selected_axes = sorted({int(phase["axis_id"]) for phase in phases})
    direction_by_axis: Dict[int, List[int]] = {
        axis: [int(direction["direction_id"]) for direction in directions
               if int(direction.get("axis_id", -1)) == axis]
        for axis in selected_axes
    }

    state_count = len(phases) + 1
    unknown_index = len(phases)
    emission = np.zeros((count, state_count), dtype=np.float64)
    for bin_index in range(count):
        for phase_index, phase in enumerate(phases):
            emission[bin_index, phase_index] = _phase_emission(
                phase, directions, group_signals, bin_index, config)
        emission[bin_index, unknown_index] = -0.1

    next_penalty = float(_cfg(config, "phase_transition_penalty", 0.4))
    previous_penalty = float(_cfg(config, "phase_reverse_penalty", 1.5))
    skip_penalty = float(_cfg(config, "phase_skip_penalty", 2.5))
    unknown_penalty = float(_cfg(config, "phase_unknown_penalty", 0.3))
    transition = np.full((state_count, state_count), -skip_penalty,
                         dtype=np.float64)
    for left in range(len(phases)):
        for right in range(len(phases)):
            if left == right:
                transition[left, right] = 0.0
            elif (left + 1) % len(phases) == right:
                transition[left, right] = -next_penalty
            elif (right + 1) % len(phases) == left:
                transition[left, right] = -previous_penalty
            else:
                transition[left, right] = -skip_penalty
        transition[left, unknown_index] = -unknown_penalty
        transition[unknown_index, left] = -unknown_penalty
    transition[unknown_index, unknown_index] = 0.0

    dp = np.full((count, state_count), -1e9, dtype=np.float64)
    back = np.zeros((count, state_count), dtype=np.int32)
    dp[0] = emission[0]
    for bin_index in range(1, count):
        for state in range(state_count):
            best_score = -1e9
            best_previous = 0
            for previous in range(state_count):
                score = dp[bin_index - 1, previous] + transition[previous, state]
                if score > best_score:
                    best_score = score
                    best_previous = previous
            dp[bin_index, state] = best_score + emission[bin_index, state]
            back[bin_index, state] = best_previous
    path = [int(np.argmax(dp[count - 1]))]
    for bin_index in range(count - 1, 0, -1):
        path.append(int(back[bin_index, path[-1]]))
    path.reverse()

    # Merge very short intervals into the neighbour with the stronger mean
    # emission.  This enforces the reviewed minimum phase duration.
    min_duration_sec = float(_cfg(config, "phase_min_duration_sec", 2.0))
    min_bins = max(1, int(round(min_duration_sec / bin_sec)))
    changed = True
    while changed and len(path) > 1:
        changed = False
        runs: List[Tuple[int, int, int]] = []
        run_start = 0
        for index in range(1, len(path) + 1):
            if index == len(path) or path[index] != path[run_start]:
                runs.append((run_start, index, int(path[run_start])))
                run_start = index
        for run_index, (low, high, state) in enumerate(runs):
            if high - low >= min_bins:
                continue
            neighbours = []
            if run_index > 0:
                neighbours.append(runs[run_index - 1])
            if run_index + 1 < len(runs):
                neighbours.append(runs[run_index + 1])
            if not neighbours:
                continue
            best = max(
                neighbours,
                key=lambda item: float(np.mean(
                    emission[item[0]:item[1], item[2]])))
            for index in range(low, high):
                path[index] = best[2]
            changed = True
            break

    timeline: List[Dict[str, Any]] = []
    for index, state in enumerate(path):
        phase = phases[state] if state < len(phases) else None
        if not timeline or timeline[-1]["phase"] != (
                None if phase is None else phase["phase"]):
            if timeline:
                timeline[-1]["end_timestamp"] = _round(
                    start + index * bin_sec)
            green: List[Dict[str, Any]] = []
            if phase is not None:
                for direction in directions:
                    for movement in ("straight", "left"):
                        allowed, mode = _phase_allows(
                            phase, direction, movement, config)
                        if allowed:
                            green.append({
                                "direction_id": int(direction["direction_id"]),
                                "movement": movement,
                                "mode": mode,
                            })
                for direction in directions:
                    green.append({
                        "direction_id": int(direction["direction_id"]),
                        "movement": "right",
                        "mode": "always",
                    })
            timeline.append({
                "phase": None if phase is None else phase["phase"],
                "axis_id": None if phase is None else int(phase["axis_id"]),
                "movement": None if phase is None else str(phase["movement"]),
                "start_timestamp": _round(start + index * bin_sec),
                "end_timestamp": None,
                "green": green,
                "source": "unknown" if phase is None else "observed",
            })
    if timeline:
        timeline[-1]["end_timestamp"] = _round(
            start + count * bin_sec)
    for interval in timeline:
        low = _bin_index(grid, float(interval["start_timestamp"]))
        high = max(low + 1, _bin_index(grid, float(interval["end_timestamp"])))
        interval["mean_emission"] = _round(float(np.mean(
            emission[low:high, path[low:high]])) if high > low else 0.0)
        if interval["phase"] is not None and float(interval["mean_emission"]) <= 0.0:
            interval["source"] = "default_hold"
    state_counts: Dict[str, int] = {}
    for state in path:
        name = "unknown" if state >= len(phases) else phases[state]["phase"]
        state_counts[name] = state_counts.get(name, 0) + 1
    # Conflict diagnostics: perpendicular axes must not both report a green
    # straight (or both a green protected left) in the same bin.  These bins
    # are the first thing to inspect when the per-track evidence disagrees
    # with a real signal plan.
    conflict_bins = {"straight": 0, "left": 0}
    for bin_index in range(count):
        for movement in ("straight", "left"):
            green_axes = []
            for axis in selected_axes:
                green = False
                for direction_id in direction_by_axis.get(axis, []):
                    signal = group_signals.get(
                        f"{direction_id}:{movement}")
                    if (signal is not None
                            and signal["states"][bin_index] == "green"):
                        green = True
                        break
                if green:
                    green_axes.append(axis)
            if len(green_axes) >= 2:
                conflict_bins[movement] += 1
    return {
        "phases": phases,
        "timeline": timeline,
        "state_counts": state_counts,
        "phase_flip_count": max(0, len(timeline) - 1),
        "axis_conflict_bins": conflict_bins,
        "axis_conflict_count": int(sum(conflict_bins.values())),
    }


# ---------------------------------------------------------------------------
# per-track traffic state
# ---------------------------------------------------------------------------

def _compress_track_states(
        times: Sequence[float],
        states: Sequence[str],
        reasons: Sequence[str],
) -> List[Dict[str, Any]]:
    intervals: List[Dict[str, Any]] = []
    for index, state in enumerate(states):
        reason = reasons[index] if index < len(reasons) else ""
        if not intervals or (intervals[-1]["state"] != state
                             or intervals[-1]["reason"] != reason):
            intervals.append({
                "state": state,
                "reason": reason,
                "start_timestamp": _round(float(times[index])),
                "end_timestamp": _round(float(times[index])),
            })
        else:
            intervals[-1]["end_timestamp"] = _round(float(times[index]))
    for interval in intervals:
        interval["duration"] = _round(
            float(interval["end_timestamp"])
            - float(interval["start_timestamp"]))
    return intervals


def assign_track_traffic_states(
        evidence: Mapping[str, Any],
        directions: Sequence[Mapping[str, Any]],
        group_signals: Mapping[str, Mapping[str, Any]],
        grid: Mapping[str, Any],
        config: Any,
) -> Dict[int, Dict[str, Any]]:
    """Assign moving / waiting_red / waiting_left_area / yielding states."""

    direction_by_id = {int(item["direction_id"]): item for item in directions}
    stop_tolerance = float(_cfg(config, "stop_line_tolerance_m", 4.0))
    moving_speed = float(_cfg(config, "moving_speed_threshold", 1.0))
    result: Dict[int, Dict[str, Any]] = {}
    for track_id, samples in evidence["samples_by_track"].items():
        direction = direction_by_id.get(int(samples["direction_id"]))
        if direction is None:
            continue
        movement = str(samples["movement"])
        line = direction.get("stop_line_forward")
        straight_key = f"{int(samples['direction_id'])}:straight"
        straight_signal = group_signals.get(straight_key)
        left_key = f"{int(samples['direction_id'])}:left"
        left_signal = group_signals.get(left_key)
        states: List[str] = []
        reasons: List[str] = []
        for index, timestamp in enumerate(samples["times"]):
            bin_index = _bin_index(grid, float(timestamp))
            signal = group_signals.get(
                f"{int(samples['direction_id'])}:{movement}")
            state = (signal["states"][bin_index]
                     if signal is not None else "unknown")
            speed = float(samples["speeds"][index])
            stopped = bool(samples["stopped"][index]) or (
                speed <= float(_cfg(config, "stop_speed_threshold", 0.8)))
            along = float(samples["along"][index])
            downstream = (line is not None
                          and along > float(line) + stop_tolerance)
            straight_green = (
                straight_signal is not None
                and straight_signal["states"][bin_index] == "green")
            left_allowed = state in ("green", "waiting_area") or straight_green
            if not stopped and speed > moving_speed:
                if movement == "right":
                    states.append("moving")
                    reasons.append("right_always_allowed")
                elif state == "red":
                    states.append("moving")
                    reasons.append("moving_against_red")
                else:
                    states.append("moving")
                    reasons.append("signal_green")
                continue
            if movement == "right":
                states.append("yielding")
                reasons.append("right_turn_stop")
            elif movement == "left":
                if downstream and left_allowed:
                    states.append("waiting_left_area")
                    reasons.append("inside_waiting_area")
                elif state == "red" and not straight_green:
                    states.append("waiting_red")
                    reasons.append("left_red")
                elif state == "waiting_area":
                    states.append("waiting_left_area")
                    reasons.append("waiting_area_stop")
                elif state == "green":
                    states.append("waiting_queue")
                    reasons.append("left_green_but_stopped")
                else:
                    states.append("uncertain")
                    reasons.append("unknown_signal")
            else:
                if state == "red":
                    states.append("waiting_red")
                    reasons.append("straight_red")
                elif state == "green":
                    states.append("waiting_queue")
                    reasons.append("green_but_stopped")
                else:
                    states.append("uncertain")
                    reasons.append("unknown_signal")
        intervals = _compress_track_states(
            samples["times"], states, reasons)
        counts: Dict[str, int] = {}
        for state in states:
            counts[state] = counts.get(state, 0) + 1
        result[int(track_id)] = {
            "track_id": int(track_id),
            "movement": movement,
            "direction_id": int(samples["direction_id"]),
            "axis_id": int(samples["axis_id"]),
            "intervals": intervals,
            "counts": counts,
        }
    return result


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def build_direction_phase_diagnostics(
        classified: Sequence[Tuple[int, Mapping[str, Any]]],
        directions: Sequence[Mapping[str, Any]],
        stop_intervals_by_track: Mapping[int, Sequence[Tuple[float, float,
                                                              np.ndarray]]],
        config: Any,
) -> Dict[str, Any]:
    """Run the full direction-level four-phase diagnostic.

    ``directions`` must already carry ``direction_id`` / ``heading`` /
    ``forward`` / ``right`` / ``origin`` / ``track_ids`` /
    ``stop_line_forward`` (see ``traffic_light._build_direction_context``).
    ``stop_intervals_by_track`` maps track id to the same
    ``(start, end, position)`` tuples used by ``_stop_intervals``.
    """

    records, axes = pair_directions_into_axes(directions, config)
    for record in records:
        for track_id in record.get("track_ids", []):
            for _tid, stats in classified:
                if int(_tid) == int(track_id):
                    stats["phase_direction_id"] = int(record["direction_id"])
                    stats["phase_axis_id"] = int(record["axis_id"])
                    stats["phase_axis_sign"] = int(record["axis_sign"])
                    stats["phase_forward"] = np.asarray(
                        record["forward"], dtype=np.float64)
                    stats["phase_right"] = np.asarray(
                        record["right"], dtype=np.float64)
                    stats["phase_origin"] = np.asarray(
                        record["origin"], dtype=np.float64)
                    stats["phase_stop_line_forward"] = record.get(
                        "stop_line_forward")
                    break
    grid = build_time_grid(
        classified, float(_cfg(config, "phase_bin_sec", 0.2)))
    if grid is None:
        return {
            "directions": records,
            "axes": axes,
            "grid": None,
            "group_signals": {},
            "axis_phase": {"phases": [], "timeline": [], "state_counts": {}},
            "track_traffic_states": {},
        }
    evidence = collect_group_evidence(
        classified, records, stop_intervals_by_track, grid, config)
    group_signals = infer_group_signal_timeline(evidence, grid, config)
    axis_phase = infer_axis_phase_timeline(
        records, axes, group_signals, grid, config)
    track_states = assign_track_traffic_states(
        evidence, records, group_signals, grid, config)

    direction_signal_timeline: List[Dict[str, Any]] = []
    for key, signal in sorted(
            group_signals.items(),
            key=lambda item: (item[1]["direction_id"], item[1]["movement"])):
        direction_signal_timeline.append({
            "direction_id": int(signal["direction_id"]),
            "movement": str(signal["movement"]),
            "track_ids": list(signal["track_ids"]),
            "intervals": signal["intervals"],
            "counts": signal["counts"],
            "evidence_totals": signal["evidence_totals"],
        })
    return {
        "directions": records,
        "axes": axes,
        "grid": {
            "start": _round(grid["start"]),
            "end": _round(grid["end"]),
            "bin_sec": _round(grid["bin_sec"]),
            "count": int(grid["count"]),
        },
        "group_signals": group_signals,
        "direction_signal_timeline": direction_signal_timeline,
        "axis_phase": axis_phase,
        "track_traffic_states": track_states,
    }
