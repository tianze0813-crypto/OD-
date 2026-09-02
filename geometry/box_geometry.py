"""Conservative geometry stabilization after the identity pipeline.

This stage freezes identity, class, yaw, and box presence. It estimates one
physical size per track, locks confirmed parking portions in world space, and
smooths the observed centers of all remaining portions. Lidar points provide
bounded position and ground evidence; they are never used as an unconstrained
min/max box fitter.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class GeometryConfig:
    min_track_observations: int = 5
    ground_ring_inner_margin: float = 0.18
    ground_ring_outer_margin: float = 1.20
    ground_z_below: float = 0.75
    ground_z_above: float = 0.45
    ground_min_points: int = 18
    ground_clearance_vehicle: float = 0.04
    ground_clearance_small: float = 0.025
    max_ground_z_move: float = 0.32
    body_crop_margin: float = 0.22
    static_point_blend: float = 0.30
    # Weight of the robust visible-side anchor after the parking slot has
    # supplied the prior. The previous implementation applied this factor
    # twice, reducing the effective correction to roughly nine percent.
    static_anchor_weight: float = 0.72
    dynamic_point_blend: float = 0.30
    dynamic_fit_blend: float = 0.65
    max_observation_gap: float = 1.20
    fit_half_window: int = 3
    # A later preview enabled this without changing the reviewed default.
    # The fixed parking anchor remains the long-term prior, while each frame
    # may follow a reliable visible vehicle face by a bounded amount.
    static_framewise_snap: bool = False
    static_snap_blend: float = 0.82
    static_snap_half_window: int = 2
    static_snap_max_gap: float = 0.80


_SIZE_QUANTILES = {
    "Car": (0.50, 0.50, 0.55),
    "Truck": (0.75, 0.55, 0.60),
    "Cyclist": (0.50, 0.50, 0.55),
    "Pedestrian": (0.50, 0.50, 0.55),
    "Bus": (0.75, 0.55, 0.60),
    "Nonmotorized_vehicle": (0.50, 0.50, 0.55),
}

_SIZE_BOUNDS = {
    # Broad guards only. The track itself, not these values, determines size.
    "Car": ((3.20, 6.20), (1.45, 2.65), (1.20, 2.45)),
    "Truck": ((5.50, 17.00), (2.00, 3.80), (2.10, 4.80)),
    "Cyclist": ((1.10, 3.80), (0.50, 1.60), (0.90, 2.30)),
    "Pedestrian": ((0.45, 1.40), (0.40, 1.30), (1.10, 2.20)),
    "Bus": ((7.00, 20.00), (2.20, 4.20), (2.40, 5.00)),
    "Nonmotorized_vehicle": ((1.10, 4.00), (0.50, 1.80), (0.90, 2.50)),
}

_MAX_POINT_SHIFT = {
    "Car": 0.28,
    "Truck": 0.42,
    "Cyclist": 0.20,
    "Pedestrian": 0.14,
    "Bus": 0.48,
    "Nonmotorized_vehicle": 0.20,
}

_MAX_PATH_SHIFT = {
    "Car": 0.55,
    "Truck": 0.75,
    "Cyclist": 0.42,
    "Pedestrian": 0.30,
    "Bus": 0.85,
    "Nonmotorized_vehicle": 0.42,
}

_MAX_STATIC_SNAP_SHIFT = {
    "Car": 0.48,
    "Truck": 0.62,
    "Cyclist": 0.32,
    "Pedestrian": 0.22,
    "Bus": 0.70,
    "Nonmotorized_vehicle": 0.32,
}


def _class_name(items: Sequence[Mapping[str, Any]]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        counts[str(item["det"].get("class_name", ""))] += 1
    return max(counts, key=counts.get) if counts else "Car"


def _weighted_quantile(values: Sequence[float], weights: Sequence[float],
                       quantile: float) -> float:
    values_array = np.asarray(values, dtype=np.float64)
    weights_array = np.asarray(weights, dtype=np.float64)
    finite = np.isfinite(values_array) & np.isfinite(weights_array) & (weights_array > 0)
    if not np.any(finite):
        return float(np.median(values_array[np.isfinite(values_array)]))
    values_array = values_array[finite]
    weights_array = weights_array[finite]
    order = np.argsort(values_array)
    values_array = values_array[order]
    weights_array = weights_array[order]
    cumulative = np.cumsum(weights_array) - 0.5 * weights_array
    cumulative /= max(float(np.sum(weights_array)), 1e-9)
    return float(np.interp(float(quantile), cumulative, values_array))


def _robust_xy(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.asarray([
        _weighted_quantile(values[:, axis], weights, 0.50)
        for axis in range(values.shape[1])
    ], dtype=np.float64)


def _physical_size(box: Sequence[float]) -> np.ndarray:
    dx, dy, dz = (float(box[index]) for index in (3, 4, 5))
    return np.asarray([max(dx, dy), min(dx, dy), dz], dtype=np.float64)


def _estimate_track_size(items: Sequence[MutableMapping[str, Any]],
                         class_name: str) -> np.ndarray:
    sizes = np.asarray([_physical_size(item["det"]["box_lidar"])
                        for item in items], dtype=np.float64)
    weights = np.asarray([
        max(float(item["det"].get("score", 0.0)), 0.05)
        * math.sqrt(max(float(item.get("point_count", 1)), 1.0))
        for item in items
    ], dtype=np.float64)
    quantiles = _SIZE_QUANTILES.get(class_name, (0.50, 0.50, 0.55))
    result = np.asarray([
        _weighted_quantile(sizes[:, axis], weights, quantiles[axis])
        for axis in range(3)
    ], dtype=np.float64)

    # Point extents are lower-bound evidence only. They can prevent a partial
    # detector box from clipping observed points, but can never shrink a box.
    spans = [item["point_span"] for item in items
             if item.get("point_span") is not None]
    # Sparse small-object returns frequently include a nearby rider, road edge,
    # or another target. Their dimensions therefore stay detector-prior driven;
    # lidar still refines center and ground below.
    if class_name not in {"Car", "Truck"}:
        spans = []
    if spans:
        span_array = np.asarray(spans, dtype=np.float64)
        lower_bound = np.percentile(span_array, 75.0, axis=0)
        padding = np.asarray([0.12, 0.10, 0.08], dtype=np.float64)
        raw_high = np.percentile(sizes, 90.0, axis=0)
        expansion_cap = raw_high + np.asarray(
            [0.45 if class_name == "Truck" else 0.25, 0.18, 0.18])
        result = np.maximum(result, np.minimum(lower_bound + padding, expansion_cap))

    bounds = _SIZE_BOUNDS.get(class_name)
    if bounds is not None:
        result = np.asarray([
            np.clip(result[axis], bounds[axis][0], bounds[axis][1])
            for axis in range(3)
        ], dtype=np.float64)
    return result


class _LidarCache:
    def __init__(self, clip: Path):
        self.clip = Path(clip)
        self.cache: Dict[str, np.ndarray | None] = {}

    def get(self, frame_id: str) -> np.ndarray | None:
        key = str(frame_id)
        if key in self.cache:
            return self.cache[key]
        path = self.clip / "lidar" / "lidar_top" / f"{key}.bin"
        if not path.is_file():
            self.cache[key] = None
            return None
        values = np.fromfile(path, dtype=np.float32)
        if values.size == 0 or values.size % 4 != 0:
            self.cache[key] = None
            return None
        points = values.reshape(-1, 4)[:, :3].astype(np.float64, copy=False)
        self.cache[key] = points
        return points


def _local_xy(points_xy: np.ndarray, center_xy: Sequence[float],
              yaw: float) -> np.ndarray:
    relative = points_xy - np.asarray(center_xy, dtype=np.float64)
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    return np.column_stack((
        relative[:, 0] * cosine + relative[:, 1] * sine,
        -relative[:, 0] * sine + relative[:, 1] * cosine,
    ))


def _estimate_ground(points: np.ndarray, box: Sequence[float],
                     config: GeometryConfig) -> Tuple[float | None, int]:
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    local = _local_xy(points[:, :2], (x, y), yaw)
    inner = np.asarray([dx / 2.0, dy / 2.0]) + config.ground_ring_inner_margin
    outer = np.asarray([dx / 2.0, dy / 2.0]) + config.ground_ring_outer_margin
    in_outer = (np.abs(local[:, 0]) <= outer[0]) & (np.abs(local[:, 1]) <= outer[1])
    in_inner = (np.abs(local[:, 0]) <= inner[0]) & (np.abs(local[:, 1]) <= inner[1])
    expected = z - dz / 2.0
    z_gate = ((points[:, 2] >= expected - config.ground_z_below)
              & (points[:, 2] <= expected + config.ground_z_above))
    candidates = points[in_outer & ~in_inner & z_gate]
    if len(candidates) < config.ground_min_points:
        return None, int(len(candidates))

    # Select the dominant low surface, then fit a local plane. Restricting the
    # histogram to its lower 70% prevents vehicle bodies from winning the mode.
    z_values = candidates[:, 2]
    high = float(np.percentile(z_values, 70.0))
    low = float(np.min(z_values))
    if high - low < 0.04:
        selected = candidates
    else:
        edges = np.arange(low, high + 0.061, 0.06)
        if len(edges) < 2:
            return float(np.median(z_values)), int(len(candidates))
        counts, edges = np.histogram(z_values[z_values <= high], bins=edges)
        center = float((edges[int(np.argmax(counts))]
                        + edges[int(np.argmax(counts)) + 1]) / 2.0)
        selected = candidates[np.abs(z_values - center) <= 0.12]
    if len(selected) < max(10, config.ground_min_points // 2):
        return None, int(len(selected))

    design = np.column_stack((selected[:, 0], selected[:, 1],
                              np.ones(len(selected))))
    keep = np.ones(len(selected), dtype=bool)
    coefficients = None
    for _ in range(3):
        if int(np.count_nonzero(keep)) < 8:
            break
        coefficients, *_ = np.linalg.lstsq(
            design[keep], selected[keep, 2], rcond=None)
        residual = selected[:, 2] - design @ coefficients
        median = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - median)))
        keep = np.abs(residual - median) <= max(0.07, 3.0 * 1.4826 * mad)
    if coefficients is None or int(np.count_nonzero(keep)) < 8:
        return None, int(np.count_nonzero(keep))
    if math.hypot(float(coefficients[0]), float(coefficients[1])) > 0.35:
        return None, int(np.count_nonzero(keep))
    estimate = float(coefficients[0] * x + coefficients[1] * y + coefficients[2])
    if abs(estimate - expected) > config.ground_z_below:
        return None, int(np.count_nonzero(keep))
    return estimate, int(np.count_nonzero(keep))


def _body_evidence(points: np.ndarray, box: Sequence[float], ground_z: float | None,
                   physical_size: np.ndarray, class_name: str,
                   config: GeometryConfig) -> Tuple[int, np.ndarray | None, np.ndarray]:
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    long_x = dx >= dy
    size_xy = (physical_size[:2] if long_x else physical_size[[1, 0]])
    local = _local_xy(points[:, :2], (x, y), yaw)
    half = size_xy / 2.0 + config.body_crop_margin
    bottom = ground_z if ground_z is not None else z - dz / 2.0
    mask = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (points[:, 2] >= bottom + 0.10)
        & (points[:, 2] <= bottom + physical_size[2] + 0.25)
    )
    body_local_xy = local[mask]
    body_z = points[mask, 2] - bottom
    count = int(len(body_local_xy))
    if count < 8:
        return count, None, np.zeros(2, dtype=np.float64)

    q05 = np.percentile(body_local_xy, 5.0, axis=0)
    q95 = np.percentile(body_local_xy, 95.0, axis=0)
    span_xy_field = q95 - q05
    span_physical = np.asarray([
        max(span_xy_field), min(span_xy_field),
        float(np.percentile(body_z, 95.0) - np.percentile(body_z, 5.0)),
    ], dtype=np.float64)

    origin_local = _local_xy(np.zeros((1, 2)), (x, y), yaw)[0]
    delta = np.zeros(2, dtype=np.float64)
    max_shift = _MAX_POINT_SHIFT.get(class_name, 0.25)
    for axis in range(2):
        dimension = float(size_xy[axis])
        coverage = float(span_xy_field[axis] / max(dimension, 1e-6))
        if coverage < (0.18 if class_name in {
                "Cyclist", "Pedestrian", "Nonmotorized_vehicle"} else 0.25):
            continue
        if abs(float(origin_local[axis])) < dimension / 2.0 + 0.50:
            continue
        observed = (float(np.percentile(body_local_xy[:, axis], 90.0))
                    if origin_local[axis] > 0.0
                    else float(np.percentile(body_local_xy[:, axis], 10.0)))
        expected = math.copysign(dimension / 2.0, float(origin_local[axis]))
        delta[axis] = float(np.clip(observed - expected, -max_shift, max_shift))
    cosine, sine = math.cos(yaw), math.sin(yaw)
    delta_lidar = np.asarray([
        delta[0] * cosine - delta[1] * sine,
        delta[0] * sine + delta[1] * cosine,
    ], dtype=np.float64)
    return count, span_physical, delta_lidar


def _static_visible_face_delta(
        points: np.ndarray, box: Sequence[float], ground_z: float | None,
        physical_size: np.ndarray, class_name: str,
        config: GeometryConfig,
) -> Tuple[int, np.ndarray | None, np.ndarray, int]:
    """Align a parked box to lidar-visible faces without fitting its size.

    A visible side is usually thin along its surface normal, so reliability is
    judged from coverage along the *other* box axis. This keeps a narrow but
    long vehicle side usable when distance or occlusion leaves only that side.
    """
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    long_x = dx >= dy
    size_xy = (physical_size[:2] if long_x else physical_size[[1, 0]])
    local = _local_xy(points[:, :2], (x, y), yaw)
    half = size_xy / 2.0 + config.body_crop_margin
    bottom = ground_z if ground_z is not None else z - dz / 2.0
    mask = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (points[:, 2] >= bottom + 0.12)
        & (points[:, 2] <= bottom + physical_size[2] + 0.18)
    )
    body_local_xy = local[mask]
    body_z = points[mask, 2] - bottom
    count = int(len(body_local_xy))
    if count < 12:
        return count, None, np.zeros(2, dtype=np.float64), 0

    q10 = np.percentile(body_local_xy, 10.0, axis=0)
    q90 = np.percentile(body_local_xy, 90.0, axis=0)
    span_xy_field = q90 - q10
    z_span = float(np.percentile(body_z, 90.0)
                   - np.percentile(body_z, 10.0))
    span_physical = np.asarray([
        max(span_xy_field), min(span_xy_field), z_span,
    ], dtype=np.float64)
    if z_span < 0.28:
        return count, span_physical, np.zeros(2, dtype=np.float64), 0

    origin_local = _local_xy(np.zeros((1, 2)), (x, y), yaw)[0]
    local_delta = np.zeros(2, dtype=np.float64)
    valid_axes = 0
    max_shift = _MAX_POINT_SHIFT.get(class_name, 0.25)
    for axis in range(2):
        dimension = float(size_xy[axis])
        if abs(float(origin_local[axis])) < dimension / 2.0 + 0.50:
            continue
        other_axis = 1 - axis
        other_coverage = float(
            span_xy_field[other_axis] / max(float(size_xy[other_axis]), 1e-6))
        # A long visible flank strongly locates vehicle width. Locating the
        # front/rear face needs less width coverage because occlusion often
        # leaves only a corner of a parked car.
        min_other_coverage = 0.30 if axis == 1 else 0.16
        if other_coverage < min_other_coverage:
            continue
        observed = (float(q90[axis]) if origin_local[axis] > 0.0
                    else float(q10[axis]))
        expected = math.copysign(dimension / 2.0, float(origin_local[axis]))
        local_delta[axis] = float(np.clip(
            observed - expected, -max_shift, max_shift))
        valid_axes += 1

    cosine, sine = math.cos(yaw), math.sin(yaw)
    delta_lidar = np.asarray([
        local_delta[0] * cosine - local_delta[1] * sine,
        local_delta[0] * sine + local_delta[1] * cosine,
    ], dtype=np.float64)
    return count, span_physical, delta_lidar, valid_axes


def _smooth_static_snap_targets(
        items: Sequence[MutableMapping[str, Any]], class_name: str,
        config: GeometryConfig,
) -> Dict[int, np.ndarray]:
    """Robustly smooth per-frame visible-face targets in world XY."""
    result: Dict[int, np.ndarray] = {}
    radius = _MAX_STATIC_SNAP_SHIFT.get(class_name, 0.40)
    for index, item in enumerate(items):
        lo = max(0, index - config.static_snap_half_window)
        hi = min(len(items), index + config.static_snap_half_window + 1)
        window = [other for other in items[lo:hi]
                  if abs(int(other["timestamp"]) - int(item["timestamp"])) / 1e9
                  <= config.static_snap_max_gap]
        if not window:
            result[id(item)] = item["point_world"][:2].copy()
            continue
        values = np.asarray([other["point_world"][:2] for other in window])
        times = np.asarray([
            (int(other["timestamp"]) - int(item["timestamp"])) / 1e9
            for other in window
        ], dtype=np.float64)
        base_weights = np.asarray([
            max(float(other["det"].get("score", 0.0)), 0.10)
            * math.sqrt(max(float(other.get("point_count", 1)), 1.0))
            for other in window
        ], dtype=np.float64)
        temporal = np.exp(-0.5 * (times / 0.45) ** 2)
        weights = np.sqrt(np.maximum(base_weights * temporal, 1e-9))
        design = np.column_stack((np.ones(len(window)), times))
        coefficients, *_ = np.linalg.lstsq(
            design * weights[:, None], values * weights[:, None], rcond=None)
        fitted = coefficients[0]
        residuals = np.linalg.norm(values - design @ coefficients, axis=1)
        median = float(np.median(residuals))
        mad = float(np.median(np.abs(residuals - median)))
        scale = max(0.05, 1.4826 * mad)
        robust = np.minimum(1.0, (2.5 * scale) / np.maximum(residuals, 1e-9))
        robust_weights = weights * np.sqrt(robust)
        coefficients, *_ = np.linalg.lstsq(
            design * robust_weights[:, None],
            values * robust_weights[:, None], rcond=None)
        fitted = coefficients[0]
        current = item["point_world"][:2]
        candidate = ((1.0 - config.static_snap_blend) * current
                     + config.static_snap_blend * fitted)
        result[id(item)] = _clip_vector(
            candidate, item["static_anchor_world_xy"], radius)
    return result


def _transform_point(matrix: np.ndarray, point: Sequence[float]) -> np.ndarray:
    value = np.asarray([float(point[0]), float(point[1]), float(point[2]), 1.0])
    return (matrix @ value)[:3]


def _clip_vector(candidate: np.ndarray, origin: np.ndarray, limit: float) -> np.ndarray:
    delta = candidate - origin
    distance = float(np.linalg.norm(delta))
    if distance <= limit or distance <= 1e-9:
        return candidate
    return origin + delta * (float(limit) / distance)


def _segments(items: Sequence[MutableMapping[str, Any]],
              config: GeometryConfig) -> List[List[MutableMapping[str, Any]]]:
    result: List[List[MutableMapping[str, Any]]] = []
    current: List[MutableMapping[str, Any]] = []
    for item in items:
        if current:
            gap = (int(item["timestamp"]) - int(current[-1]["timestamp"])) / 1e9
            if gap > config.max_observation_gap:
                result.append(current)
                current = []
        current.append(item)
    if current:
        result.append(current)
    return result


def _smooth_segment(segment: Sequence[MutableMapping[str, Any]], class_name: str,
                    config: GeometryConfig) -> Dict[int, np.ndarray]:
    if len(segment) < 3:
        return {id(item): item["point_world"][:2].copy() for item in segment}
    result: Dict[int, np.ndarray] = {}
    limit = _MAX_PATH_SHIFT.get(class_name, 0.50)
    for index, item in enumerate(segment):
        lo = max(0, index - config.fit_half_window)
        hi = min(len(segment), index + config.fit_half_window + 1)
        window = segment[lo:hi]
        times = np.asarray([
            (int(other["timestamp"]) - int(item["timestamp"])) / 1e9
            for other in window
        ], dtype=np.float64)
        values = np.asarray([other["point_world"][:2] for other in window])
        degree = 2 if len(window) >= 5 else 1
        design = np.column_stack([times ** power for power in range(degree + 1)])
        temporal = np.exp(-0.5 * (times / 0.9) ** 2)
        scores = np.asarray([
            max(float(other["det"].get("score", 0.0)), 0.10)
            for other in window
        ])
        weights = np.sqrt(temporal * scores)
        coefficients, *_ = np.linalg.lstsq(
            design * weights[:, None], values * weights[:, None], rcond=None)
        fitted = coefficients[0]
        candidate = ((1.0 - config.dynamic_fit_blend) * item["point_world"][:2]
                     + config.dynamic_fit_blend * fitted)
        result[id(item)] = _clip_vector(candidate, item["raw_world"][:2], limit)
    return result


def _departure_cutoffs(static_diagnostics: Mapping[str, Any]) -> Dict[int, int]:
    return {
        int(item["track_id"]): int(item["departure_start_timestamp"])
        for item in static_diagnostics.get("slots", [])
        if item.get("departure_start_timestamp") is not None
    }


def _slot_centers(tracking_diagnostics: Mapping[str, Any]) -> Dict[int, np.ndarray]:
    result = {}
    for item in tracking_diagnostics.get("slot_details", []):
        if item.get("track_id") is not None:
            result[int(item["track_id"])] = np.asarray(item["center"], dtype=np.float64)
    return result


def _build_tracks(frames: Sequence[Dict[str, Any]], coords: tracking.CoordinateProvider
                  ) -> Dict[int, List[MutableMapping[str, Any]]]:
    tracks: Dict[int, List[MutableMapping[str, Any]]] = defaultdict(list)
    for frame_index, frame in enumerate(frames):
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection_index, det in enumerate(frame.get("detections", [])):
            if det.get("track_id") is None or not tracking.finite_box(det):
                continue
            box = det["box_lidar"]
            tracks[int(det["track_id"])].append({
                "frame_index": frame_index,
                "detection_index": detection_index,
                "frame_id": str(frame["frame_id"]),
                "timestamp": timestamp,
                "det": det,
                "world_from_lidar": world_from_lidar,
                "lidar_from_world": np.linalg.inv(world_from_lidar),
                "raw_world": tracking.center_world(box, world_from_lidar),
            })
    for items in tracks.values():
        items.sort(key=lambda item: item["timestamp"])
    return tracks


def verify_geometry_only(before: Sequence[Dict[str, Any]],
                         after: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if len(before) != len(after):
        raise AssertionError("geometry stage changed frame count")
    checked = 0
    changed = 0
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("geometry stage changed frame order")
        left_detections = left_frame.get("detections", [])
        right_detections = right_frame.get("detections", [])
        if len(left_detections) != len(right_detections):
            raise AssertionError("geometry stage changed detection count")
        for left, right in zip(left_detections, right_detections):
            if left.get("track_id") != right.get("track_id"):
                raise AssertionError("geometry stage changed track_id")
            if left.get("class_name") != right.get("class_name"):
                raise AssertionError("geometry stage changed class_name")
            if float(left["box_lidar"][6]) != float(right["box_lidar"][6]):
                raise AssertionError("geometry stage changed yaw")
            comparable = copy.deepcopy(right)
            comparable["box_lidar"][:6] = left["box_lidar"][:6]
            if comparable != left:
                raise AssertionError("geometry stage changed a protected field")
            if not np.allclose(left["box_lidar"][:6], right["box_lidar"][:6],
                               atol=0.0, rtol=0.0):
                changed += 1
            checked += 1
    return {
        "passed": True,
        "detections_checked": checked,
        "geometry_changed": changed,
        "protected": ["track_id", "class_name", "box_lidar[6]", "box_presence"],
    }


def apply_geometry_legacy(
        frames: Sequence[Dict[str, Any]], coords: tracking.CoordinateProvider,
        clip: Path, tracking_diagnostics: Mapping[str, Any],
        static_yaw_diagnostics: Mapping[str, Any],
        config: GeometryConfig = GeometryConfig(),
        classes: Sequence[str] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = copy.deepcopy(list(frames))
    tracks = _build_tracks(output, coords)
    lidar = _LidarCache(Path(clip))
    static_ids = {
        int(item["track_id"])
        for item in static_yaw_diagnostics.get("slots", [])
    }
    cutoffs = _departure_cutoffs(static_yaw_diagnostics)
    slot_centers = _slot_centers(tracking_diagnostics)

    track_details = []
    static_boxes = 0
    dynamic_boxes = 0
    ground_boxes = 0
    point_boxes = 0
    static_snap_boxes = 0
    static_snap_face_axes = 0
    fallback_tracks = 0
    processed_boxes = 0

    allowed_classes = None if classes is None else {str(value) for value in classes}
    skipped_tracks = 0
    for track_id, items in sorted(tracks.items()):
        class_name = _class_name(items)
        if allowed_classes is not None and class_name not in allowed_classes:
            skipped_tracks += 1
            continue
        processed_boxes += len(items)
        cutoff = cutoffs.get(track_id, math.inf)
        initial_sizes = np.asarray([_physical_size(item["det"]["box_lidar"])
                                    for item in items])
        initial_size = np.median(initial_sizes, axis=0)

        # First pass: point counts, ground, and conservative lower-bound spans.
        for item in items:
            points = lidar.get(item["frame_id"])
            item["point_count"] = 0
            item["point_span"] = None
            item["ground_z"] = None
            item["ground_points"] = 0
            if points is None:
                continue
            ground_z, ground_points = _estimate_ground(
                points, item["det"]["box_lidar"], config)
            item["ground_z"] = ground_z
            item["ground_points"] = ground_points
            count, span, _delta = _body_evidence(
                points, item["det"]["box_lidar"], ground_z,
                initial_size, class_name, config)
            item["point_count"] = count
            item["point_span"] = span

        if len(items) >= config.min_track_observations:
            fixed_size = _estimate_track_size(items, class_name)
            size_mode = "track_robust_with_point_lower_bound"
        else:
            fixed_size = initial_size
            size_mode = "short_track_median_fallback"
            fallback_tracks += 1

        # Second pass: bounded point anchor correction with the final size.
        for item in items:
            box = item["det"]["box_lidar"]
            points = lidar.get(item["frame_id"])
            delta = np.zeros(2, dtype=np.float64)
            face_axes = 0
            if points is not None:
                is_parked_observation = (
                    track_id in static_ids and item["timestamp"] < cutoff)
                if config.static_framewise_snap and is_parked_observation:
                    count, _span, delta, face_axes = _static_visible_face_delta(
                        points, box, item.get("ground_z"), fixed_size,
                        class_name, config)
                else:
                    count, _span, delta = _body_evidence(
                        points, box, item.get("ground_z"), fixed_size,
                        class_name, config)
                item["point_count"] = count
            blend = (config.static_point_blend if track_id in static_ids
                     else config.dynamic_point_blend)
            # For static observations retain the full local visible-side
            # correction here; the robust track-level anchor applies the one
            # deliberate blend later. Dynamic observations stay conservative.
            applied_blend = 1.0 if track_id in static_ids else blend
            point_xy = np.asarray(box[:2], dtype=np.float64) + applied_blend * delta
            point_local = np.asarray([point_xy[0], point_xy[1], float(box[2])])
            item["point_world"] = _transform_point(
                item["world_from_lidar"], point_local)
            item["point_delta"] = blend * delta
            item["static_face_axes"] = face_axes
            if float(np.linalg.norm(delta)) > 1e-6:
                point_boxes += 1
            if face_axes > 0:
                static_snap_boxes += 1
                static_snap_face_axes += face_axes
            if item.get("ground_z") is not None:
                ground_local = [float(box[0]), float(box[1]), float(item["ground_z"])]
                item["ground_world_z"] = float(_transform_point(
                    item["world_from_lidar"], ground_local)[2])
            else:
                item["ground_world_z"] = None

        static_items = [item for item in items
                        if track_id in static_ids and item["timestamp"] < cutoff]
        dynamic_items = [
            item for item in items
            if not (track_id in static_ids and item["timestamp"] < cutoff)
        ]

        max_xy_shift = 0.0
        max_z_shift = 0.0
        if static_items:
            weights = np.asarray([
                max(float(item["det"].get("score", 0.0)), 0.10)
                * math.sqrt(max(float(item.get("point_count", 1)), 1.0))
                for item in static_items
            ])
            point_world_xy = np.asarray([item["point_world"][:2]
                                         for item in static_items])
            fitted_xy = _robust_xy(point_world_xy, weights)
            slot_xy = slot_centers.get(track_id)
            if slot_xy is not None:
                fitted_xy = ((1.0 - config.static_anchor_weight) * slot_xy
                             + config.static_anchor_weight * fitted_xy)
                fitted_xy = _clip_vector(fitted_xy, slot_xy,
                                         _MAX_POINT_SHIFT.get(class_name, 0.25))
            ground_values = [float(item["ground_world_z"])
                             for item in static_items
                             if item.get("ground_world_z") is not None]
            min_static_ground = max(3, int(math.ceil(0.10 * len(static_items))))
            if len(ground_values) >= min_static_ground:
                clearance = (config.ground_clearance_small
                             if class_name in {"Cyclist", "Pedestrian",
                                                "Nonmotorized_vehicle"}
                             else config.ground_clearance_vehicle)
                fixed_world_z = float(np.median(ground_values)
                                      + clearance + fixed_size[2] / 2.0)
            else:
                fixed_world_z = float(np.median(
                    [item["raw_world"][2] for item in static_items]))
            fixed_world = np.asarray([fitted_xy[0], fitted_xy[1], fixed_world_z])
            for item in static_items:
                item["static_anchor_world_xy"] = fitted_xy.copy()
            snap_targets = (_smooth_static_snap_targets(
                static_items, class_name, config)
                if config.static_framewise_snap else {})
            for item in static_items:
                box = item["det"]["box_lidar"]
                target_xy = snap_targets.get(id(item), fitted_xy)
                local = _transform_point(
                    item["lidar_from_world"],
                    [float(target_xy[0]), float(target_xy[1]), fixed_world_z])
                # A failed ground estimate must never turn a bad detector z
                # into a metre-scale jump. Keep the world-fixed target where
                # reliable ground exists, but cap the per-observation vertical
                # correction in the same way as the dynamic path.
                z_delta = float(np.clip(
                    local[2] - float(box[2]), -config.max_ground_z_move,
                    config.max_ground_z_move))
                new_center = np.asarray([
                    local[0], local[1], float(box[2]) + z_delta])
                max_xy_shift = max(max_xy_shift, float(np.linalg.norm(
                    new_center[:2] - np.asarray(box[:2], dtype=np.float64))))
                max_z_shift = max(max_z_shift, abs(z_delta))
                box[:3] = [float(value) for value in new_center]
                static_boxes += 1
                if len(ground_values) >= min_static_ground:
                    ground_boxes += 1

        for segment in _segments(dynamic_items, config):
            smoothed_xy = _smooth_segment(segment, class_name, config)
            ground_sequence = np.asarray([
                (float(item["ground_world_z"])
                 if item.get("ground_world_z") is not None else np.nan)
                for item in segment
            ])
            valid_ground = np.isfinite(ground_sequence)
            min_dynamic_ground = max(3, int(math.ceil(0.15 * len(segment))))
            use_ground = int(np.count_nonzero(valid_ground)) >= min_dynamic_ground
            if use_ground:
                indices = np.arange(len(segment), dtype=np.float64)
                ground_sequence[~valid_ground] = np.interp(
                    indices[~valid_ground], indices[valid_ground],
                    ground_sequence[valid_ground])
                padded = np.pad(ground_sequence, (1, 1), mode="edge")
                ground_sequence = np.asarray([
                    np.median(padded[index:index + 3])
                    for index in range(len(segment))
                ])
            clearance = (config.ground_clearance_small
                         if class_name in {"Cyclist", "Pedestrian",
                                            "Nonmotorized_vehicle"}
                         else config.ground_clearance_vehicle)
            for index, item in enumerate(segment):
                box = item["det"]["box_lidar"]
                xy = smoothed_xy[id(item)]
                world_z = (float(ground_sequence[index] + clearance
                                 + fixed_size[2] / 2.0)
                           if use_ground and np.isfinite(ground_sequence[index])
                           else float(item["point_world"][2]))
                local = _transform_point(
                    item["lidar_from_world"], [xy[0], xy[1], world_z])
                z_delta = float(np.clip(
                    local[2] - float(box[2]), -config.max_ground_z_move,
                    config.max_ground_z_move))
                new_center = np.asarray([local[0], local[1], float(box[2]) + z_delta])
                max_xy_shift = max(max_xy_shift, float(np.linalg.norm(
                    new_center[:2] - np.asarray(box[:2], dtype=np.float64))))
                max_z_shift = max(max_z_shift, abs(z_delta))
                box[:3] = [float(value) for value in new_center]
                dynamic_boxes += 1
                if use_ground and np.isfinite(ground_sequence[index]):
                    ground_boxes += 1

        # Preserve each observation's detector long-axis convention. This matters for
        # the few pedestrian frames whose long field is dy and whose frozen yaw
        # already includes the corresponding 90-degree adjustment.
        axis_swaps = 0
        for item in items:
            box = item["det"]["box_lidar"]
            if float(box[3]) >= float(box[4]):
                box[3:6] = [float(fixed_size[0]), float(fixed_size[1]),
                            float(fixed_size[2])]
            else:
                box[3:6] = [float(fixed_size[1]), float(fixed_size[0]),
                            float(fixed_size[2])]
                axis_swaps += 1

        point_frames = sum(item.get("point_count", 0) >= 8 for item in items)
        ground_frames = sum(item.get("ground_z") is not None for item in items)
        track_details.append({
            "track_id": track_id,
            "class_name": class_name,
            "observations": len(items),
            "size_mode": size_mode,
            "original_physical_size_median": [round(float(x), 4)
                                               for x in initial_size],
            "fixed_physical_size": [round(float(x), 4) for x in fixed_size],
            "point_evidence_frames": point_frames,
            "ground_evidence_frames": ground_frames,
            "static_visible_face_frames": sum(
                item.get("static_face_axes", 0) > 0 for item in items),
            "static_visible_face_axes": sum(
                int(item.get("static_face_axes", 0)) for item in items),
            "static_boxes": len(static_items),
            "dynamic_boxes": len(dynamic_items),
            "dy_long_axis_boxes": axis_swaps,
            "max_xy_center_shift": round(max_xy_shift, 4),
            "max_z_center_shift": round(max_z_shift, 4),
        })

    invariant_check = verify_geometry_only(frames, output)
    return output, {
        "policy": {
            "pipeline_position": "after_identity_class_and_yaw",
            "coordinate_frame": "box_lidar and lidar_top local frame",
            "static_center": (
                "bounded_framewise_visible_face_snap_around_parking_anchor"
                if config.static_framewise_snap
                else "fixed_world_center_for_confirmed_parking_portion"),
            "dynamic_center": "bounded_point_anchor_then_local_polynomial_world_path",
            "size": "one_physical_size_per_track_with_truck_upper_quantile",
            "pointcloud": "bounded_position_and_size_lower_bound_only",
            "ground": "outer_footprint_ring_robust_plane",
            "mutated_fields": "box_lidar[0:6]",
            "frozen_fields": ["track_id", "class_name", "box_lidar[6]", "box_presence"],
            "no_interpolation": True,
        },
        "tracks": len(tracks),
        "tracks_refined": len(tracks) - skipped_tracks,
        "tracks_skipped": skipped_tracks,
        "class_scope": (None if allowed_classes is None else sorted(allowed_classes)),
        "boxes": processed_boxes,
        "static_boxes": static_boxes,
        "dynamic_boxes": dynamic_boxes,
        "point_adjusted_boxes": point_boxes,
        "static_snap_boxes": static_snap_boxes,
        "static_snap_face_axes": static_snap_face_axes,
        "ground_adjusted_boxes": ground_boxes,
        "fallback_tracks": fallback_tracks,
        "invariant_check": invariant_check,
        "details": track_details,
    }
