#!/usr/bin/env python3
"""High-speed dynamic region construction.

The reviewed logic for dense parking scenes is deliberately inverted:

  * an area is a **dynamic region** only when there is sustained high-speed
    driving evidence;
  * everything else defaults to static / dense parking.

This avoids brittle parking-aisle width thresholds: a 3 m aisle, a 6 m aisle,
road-side parking and dense lots all stay in the default static class unless a
vehicle actually drives fast through them.

Dynamic regions are built from track-level speed evidence:
  * robust path length >= ``min_track_length``;
  * p90 speed >= ``high_speed_threshold``;
  * at least ``min_high_speed_observations`` steps above the threshold;
  * consecutive observations are connected unless the time gap is too large;
  * the resulting polylines are buffered by ``buffer_radius`` and connected
    into polygons.

Only dynamic polygons are written; the complement is the default static
region.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from region.parking_region import (
    Bounds,
    Polygon,
    _as_points,
    _compute_bounds,
    _ellipse_kernel,
    _grid_shape,
    _mask_to_polygons,
    _morphology,
    _polygon_area,
    _polygon_centroid,
    _to_pixel,
)
from tracking.tracker_conservative import center_world, yaw_world

DynamicTrack = Sequence[Tuple[float, float, float]]


@dataclass(frozen=True)
class DynamicRegionConfig:
    resolution: float = 1.0
    bounds_margin: float = 10.0

    # High-speed evidence (agreed defaults: 5 m/s, 15 m, 3 frames).
    high_speed_threshold: float = 5.0
    min_track_length: float = 15.0
    min_high_speed_observations: int = 3
    max_segment_gap_sec: float = 2.0

    # Dynamic region is the swept vehicle box only.  The reviewed decision is
    # buffer_radius = 0: roadside parked cars must not be swallowed by a
    # dilated corridor.
    buffer_radius: float = 0.0
    close_radius: float = 1.0
    min_region_area_m2: float = 50.0
    min_core_area_m2: float = 5.0
    polygon_approx_eps_m: float = 1.0

    # Reject high-speed tracks that are actually hopping across static parking
    # slots (a common ID-switch failure mode).  The hard any-observation check
    # is disabled by default: a real car departing a parking spot must still
    # create a dynamic region.  The 50% overlap-fraction rule is kept.
    static_slot_exclusion_radius: float = 1.5
    static_slot_hard_exclusion_radius: float = 0.0
    static_track_overlap_fraction: float = 0.5
    # Do not carve static-slot footprints out of a buffer-free dynamic region;
    # if a high-speed track really swept through a parking position, that is
    # real evidence.
    static_slot_mask_margin: float = 0.5
    mask_static_slot_footprints: bool = False

    # Extend a high-speed track's swept corridor by this many metres along its
    # stable heading at both ends.  This covers stop lines, red-light queues
    # and acceleration segments that do not themselves have high-speed
    # evidence.  Extension is only along the stable heading; no lateral buffer.
    extension_length_m: float = 30.0
    extension_heading_tolerance_deg: float = 12.0
    extension_min_stable_observations: int = 5
    extension_min_stable_path_m: float = 10.0
    extension_step_m: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "bounds_margin": self.bounds_margin,
            "high_speed_threshold": self.high_speed_threshold,
            "min_track_length": self.min_track_length,
            "min_high_speed_observations": self.min_high_speed_observations,
            "max_segment_gap_sec": self.max_segment_gap_sec,
            "buffer_radius": self.buffer_radius,
            "close_radius": self.close_radius,
            "min_region_area_m2": self.min_region_area_m2,
            "min_core_area_m2": self.min_core_area_m2,
            "polygon_approx_eps_m": self.polygon_approx_eps_m,
            "static_slot_exclusion_radius": (
                self.static_slot_exclusion_radius),
            "static_slot_hard_exclusion_radius": (
                self.static_slot_hard_exclusion_radius),
            "static_track_overlap_fraction": (
                self.static_track_overlap_fraction),
            "static_slot_mask_margin": self.static_slot_mask_margin,
            "mask_static_slot_footprints": (
                self.mask_static_slot_footprints),
            "extension_length_m": self.extension_length_m,
            "extension_heading_tolerance_deg": (
                self.extension_heading_tolerance_deg),
            "extension_min_stable_observations": (
                self.extension_min_stable_observations),
            "extension_min_stable_path_m": (
                self.extension_min_stable_path_m),
            "extension_step_m": self.extension_step_m,
        }


@dataclass
class DynamicRegionResult:
    bounds: Bounds
    resolution: float
    dynamic_polygons: List[Dict[str, Any]] = field(default_factory=list)
    core_polygons: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": "world",
            "bounds": {
                "xmin": round(float(self.bounds[0]), 4),
                "ymin": round(float(self.bounds[1]), 4),
                "xmax": round(float(self.bounds[2]), 4),
                "ymax": round(float(self.bounds[3]), 4),
            },
            "resolution": float(self.resolution),
            "dynamic_polygons": self.dynamic_polygons,
            "core_polygons": self.core_polygons,
            "legend": {
                "dynamic_polygons": (
                    "orange filled polygon = high-speed dynamic region "
                    "(swept box + 30m stable-heading extension, no buffer)"),
                "core_polygons": (
                    "core swept area used for disabling static slots"),
                "complement": "outside dynamic polygons = default static",
                "high_speed_tracks": "blue polylines = high-speed evidence",
                "other_dynamic_tracks": "gray polylines = other dynamic tracks",
                "static_slots": "red x = static slot centers",
            },
            "diagnostics": self.diagnostics,
            "config": self.config,
        }


def _normalize_track(items: DynamicTrack) -> List[Dict[str, Any]]:
    """Normalise tuple/dict observations to a common world-frame form."""

    normalized: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, Mapping):
            timestamp = float(item.get("timestamp", item.get("ts", 0.0)))
            world = np.asarray(item.get("world", item.get("center", (0.0, 0.0))),
                               dtype=np.float64)[:2]
            yaw = float(item.get("yaw", 0.0))
            size = np.asarray(item.get("size", (4.5, 2.0, 1.6)),
                              dtype=np.float64)[:3]
            class_name = str(item.get("class_name", ""))
        else:
            timestamp = float(item[0])
            world = np.asarray(item[1:3], dtype=np.float64)
            yaw = float(item[3]) if len(item) > 3 else 0.0
            size = (np.asarray(item[4:7], dtype=np.float64)
                    if len(item) >= 7 else np.array([4.5, 2.0, 1.6]))
            class_name = ""
        normalized.append({
            "timestamp": timestamp,
            "world": world,
            "yaw": yaw,
            "size": size,
            "class_name": class_name,
        })
    normalized.sort(key=lambda value: value["timestamp"])
    return normalized


def _box_corners(world: np.ndarray, yaw: float,
                 size: np.ndarray) -> np.ndarray:
    heading = np.array([math.cos(float(yaw)), math.sin(float(yaw))],
                       dtype=np.float64)
    cross = np.array([-heading[1], heading[0]], dtype=np.float64)
    half_length = 0.5 * float(size[0])
    half_width = 0.5 * float(size[1])
    center = np.asarray(world, dtype=np.float64)[:2]
    return np.asarray([
        center - half_length * heading - half_width * cross,
        center + half_length * heading - half_width * cross,
        center + half_length * heading + half_width * cross,
        center - half_length * heading + half_width * cross,
    ], dtype=np.float64)


def _track_speed_stats(
        items: DynamicTrack,
        config: DynamicRegionConfig,
) -> Optional[Dict[str, Any]]:
    normalized = _normalize_track(items)
    if len(normalized) < 2:
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
    p90_speed = float(np.percentile(speeds, 90.0))
    max_speed = float(np.max(speeds))
    high_speed_steps = int(np.count_nonzero(
        speeds >= float(config.high_speed_threshold)))
    return {
        "observations": len(normalized),
        "duration": round(float(times[-1] - times[0]), 4),
        "path_length": round(path_length, 4),
        "p90_speed": round(p90_speed, 4),
        "max_speed": round(max_speed, 4),
        "high_speed_steps": high_speed_steps,
    }


def _is_high_speed(stats: Mapping[str, Any],
                   config: DynamicRegionConfig) -> bool:
    return (
        float(stats["path_length"]) >= float(config.min_track_length)
        and float(stats["p90_speed"]) >= float(config.high_speed_threshold)
        and int(stats["high_speed_steps"])
        >= int(config.min_high_speed_observations)
    )


def _static_overlap_fraction(
        items: DynamicTrack,
        static_slots: Sequence[Mapping[str, Any]],
        radius: float,
) -> float:
    """Fraction of track observations close to a static slot center.

    A high-speed track that keeps landing on parked-slot centres is an
    ID-switch artefact, not a road user.
    """

    normalized = _normalize_track(items)
    if not normalized or not static_slots:
        return 0.0
    centers = np.asarray([
        np.asarray(slot.get("center", (0.0, 0.0)), dtype=np.float64)[:2]
        for slot in static_slots
    ], dtype=np.float64)
    close = 0
    for item in normalized:
        distances = np.linalg.norm(centers - item["world"], axis=1)
        if float(np.min(distances)) <= float(radius):
            close += 1
    return float(close) / float(len(normalized))


def _track_hits_static_slot(
        items: DynamicTrack,
        static_slots: Sequence[Mapping[str, Any]],
        radius: float,
) -> bool:
    """True when any observation lies inside a static slot exclusion circle."""
    normalized = _normalize_track(items)
    if radius <= 0.0 or not normalized or not static_slots:
        return False
    centers = np.asarray([
        np.asarray(slot.get("center", (0.0, 0.0)), dtype=np.float64)[:2]
        for slot in static_slots
    ], dtype=np.float64)
    for item in normalized:
        if float(np.min(np.linalg.norm(
                centers - item["world"], axis=1))) <= float(radius):
            return True
    return False


def _static_slot_mask(
        static_slots: Sequence[Mapping[str, Any]],
        bounds: Bounds,
        resolution: float,
        shape: Tuple[int, int],
        margin: float,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for slot in static_slots:
        center = np.asarray(slot.get("center", (0.0, 0.0)),
                            dtype=np.float64)[:2]
        yaw = float(slot.get("yaw", 0.0))
        size = np.asarray(slot.get("size", (4.5, 2.0, 1.6)),
                          dtype=np.float64)[:3]
        expanded = size.copy()
        expanded[0] += 2.0 * float(margin)
        expanded[1] += 2.0 * float(margin)
        corners = _box_corners(center, yaw, expanded)
        columns, rows = _to_pixel(corners, bounds, resolution, shape)
        polygon = np.column_stack([columns, rows]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask


def _rasterize_tracks(
        tracks: Mapping[int, DynamicTrack],
        bounds: Bounds,
        resolution: float,
        shape: Tuple[int, int],
        config: DynamicRegionConfig,
) -> np.ndarray:
    """Rasterize the swept area of each high-speed vehicle box."""

    seed = np.zeros(shape, dtype=np.uint8)
    for items in tracks.values():
        normalized = _normalize_track(items)
        previous = None
        for item in normalized:
            corners = _box_corners(item["world"], item["yaw"], item["size"])
            columns, rows = _to_pixel(corners, bounds, resolution, shape)
            polygon = np.column_stack([columns, rows]).astype(np.int32)
            cv2.fillPoly(seed, [polygon], 1)
            if previous is not None:
                gap = item["timestamp"] - previous["timestamp"]
                if gap <= float(config.max_segment_gap_sec):
                    previous_corners = _box_corners(
                        previous["world"], previous["yaw"], previous["size"])
                    previous_columns, previous_rows = _to_pixel(
                        previous_corners, bounds, resolution, shape)
                    current_columns, current_rows = _to_pixel(
                        corners, bounds, resolution, shape)
                    hull_points = np.vstack([
                        np.column_stack([previous_columns, previous_rows]),
                        np.column_stack([current_columns, current_rows]),
                    ]).astype(np.float32)
                    hull = cv2.convexHull(hull_points).astype(np.int32)
                    cv2.fillPoly(seed, [hull], 1)
            previous = item
    return seed


def _annotate_dynamic_polygons(
        polygons: List[Polygon],
        tracks: Mapping[int, DynamicTrack],
        config: DynamicRegionConfig,
) -> List[Dict[str, Any]]:
    annotated = []
    for index, polygon in enumerate(polygons):
        polygon_array = np.asarray(polygon, dtype=np.float32)
        track_ids = []
        for track_id, items in tracks.items():
            for item in _normalize_track(items):
                world = item["world"]
                if cv2.pointPolygonTest(
                        polygon_array,
                        (float(world[0]), float(world[1])), False) >= 0:
                    track_ids.append(int(track_id))
                    break
        centroid = _polygon_centroid(polygon)
        annotated.append({
            "id": f"dynamic_{index}",
            "polygon": [[round(float(x), 4), round(float(y), 4)]
                        for x, y in polygon],
            "area_m2": _polygon_area(polygon),
            "centroid": [centroid[0], centroid[1]],
            "high_speed_track_count": len(track_ids),
            "high_speed_track_ids": sorted(track_ids),
            "buffer_radius": float(config.buffer_radius),
        })
    return annotated


def _wrap_angle(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _mean_heading(points: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return (circular mean heading, max deviation) for a point window."""
    if len(points) < 2:
        return None
    vectors = np.diff(points, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    if not np.all(lengths > 1e-6):
        return None
    headings = np.arctan2(vectors[:, 1], vectors[:, 0])
    mean = math.atan2(float(np.mean(np.sin(headings))),
                      float(np.mean(np.cos(headings))))
    deviation = max(abs(_wrap_angle(float(value) - mean))
                    for value in headings)
    return mean, float(deviation)


def _stable_endpoint_heading(
        normalized: Sequence[Mapping[str, Any]],
        *,
        at_start: bool,
        config: DynamicRegionConfig,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """Find a stable heading at one end of a track.

    Returns ``(anchor_world_xy, heading, median_size)`` or ``None``.  A stable
    segment requires at least ``extension_min_stable_observations`` consecutive
    observations, heading deviation <= ``extension_heading_tolerance_deg`` and
    path length >= ``extension_min_stable_path_m``.
    """
    count = len(normalized)
    minimum = int(config.extension_min_stable_observations)
    if count < minimum or minimum < 2:
        return None
    max_gap = float(config.max_segment_gap_sec)
    tolerance = math.radians(float(config.extension_heading_tolerance_deg))
    minimum_path = float(config.extension_min_stable_path_m)

    windows: List[Sequence[Mapping[str, Any]]] = []
    if at_start:
        # Shortest stable prefix first; this preserves the entry heading.
        for end in range(minimum, count + 1):
            windows.append(normalized[:end])
    else:
        # Shortest stable suffix first; this preserves the exit heading.
        for start in range(count - minimum, -1, -1):
            windows.append(normalized[start:])

    for window in windows:
        times = np.asarray([float(item["timestamp"]) for item in window],
                           dtype=np.float64)
        if len(times) >= 2 and float(np.max(np.diff(times))) > max_gap:
            continue
        points = np.asarray([item["world"] for item in window],
                            dtype=np.float64)
        if float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) \
                < minimum_path:
            continue
        heading_info = _mean_heading(points)
        if heading_info is None:
            continue
        heading, deviation = heading_info
        if deviation > tolerance:
            continue
        sizes = np.asarray([item["size"] for item in window],
                           dtype=np.float64)
        median_size = np.median(sizes, axis=0)
        anchor = np.asarray(
            window[0]["world"] if at_start else window[-1]["world"],
            dtype=np.float64)[:2]
        return anchor, float(heading), median_size
    return None


def _extend_track(
        normalized: Sequence[Mapping[str, Any]],
        config: DynamicRegionConfig,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extend a high-speed track's swept corridor along its stable headings."""
    extended = [dict(item) for item in normalized]
    details: List[Dict[str, Any]] = []
    length = float(config.extension_length_m)
    if length <= 0.0 or len(normalized) < 2:
        return extended, details
    step = max(float(config.extension_step_m), 0.5)
    count = int(round(length / step))
    if count <= 0:
        return extended, details

    for at_start in (True, False):
        info = _stable_endpoint_heading(
            normalized, at_start=at_start, config=config)
        if info is None:
            continue
        anchor, heading, size = info
        direction = np.array([math.cos(heading), math.sin(heading)],
                             dtype=np.float64)
        if at_start:
            direction = -direction
        base_timestamp = float(
            normalized[0]["timestamp"] if at_start
            else normalized[-1]["timestamp"])
        sign = -1.0 if at_start else 1.0
        for index in range(1, count + 1):
            world = anchor + direction * (step * index)
            extended.append({
                "timestamp": base_timestamp + sign * 0.1 * index,
                "world": world.astype(np.float64),
                "yaw": float(heading),
                "size": np.asarray(size, dtype=np.float64),
                "class_name": "",
                "synthetic_extension": True,
            })
        details.append({
            "end": "start" if at_start else "end",
            "heading_deg": round(math.degrees(heading), 3),
            "length_m": round(length, 3),
            "step_m": round(step, 3),
            "anchor": [round(float(anchor[0]), 4),
                       round(float(anchor[1]), 4)],
        })
    extended.sort(key=lambda item: float(item["timestamp"]))
    return extended, details


def _extend_high_speed_tracks(
        tracks: Mapping[int, Sequence[Mapping[str, Any]]],
        config: DynamicRegionConfig,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[Dict[str, Any]]]:
    extended: Dict[int, List[Dict[str, Any]]] = {}
    details: List[Dict[str, Any]] = []
    for track_id, items in tracks.items():
        track_extended, track_details = _extend_track(
            _normalize_track(items), config)
        extended[int(track_id)] = track_extended
        if track_details:
            details.append({
                "track_id": int(track_id),
                "extensions": track_details,
            })
    return extended, details


def build_dynamic_regions(
        dynamic_tracks: Mapping[int, DynamicTrack],
        *,
        static_slots: Sequence[Mapping[str, Any]] = (),
        reference_points: Any = None,
        config: DynamicRegionConfig = DynamicRegionConfig(),
) -> DynamicRegionResult:
    """Build high-speed dynamic polygons from world-frame tracks.

    ``dynamic_tracks`` maps track id -> observations.  Observations may be
    tuples ``(timestamp_sec, x, y)`` or dicts with ``timestamp``, ``world``,
    ``yaw`` and ``size``.  The swept vehicle box is rasterized; when yaw/size
    are omitted a default car box is used.

    ``static_slots`` are static slot records (dicts with ``center``, ``yaw``,
    ``size``).  High-speed tracks that keep landing on static slot centres are
    rejected as ID-switch artefacts.  Static slot footprints are not carved
    out by default: with ``buffer_radius=0`` a parked car is only inside the
    region if a real vehicle box swept through its position.
    """

    reference = _as_points(reference_points)
    all_points: List[np.ndarray] = [reference]
    track_stats: Dict[int, Dict[str, Any]] = {}
    kept_tracks: Dict[int, DynamicTrack] = {}
    rejected_static_tracks: List[Dict[str, Any]] = []
    for track_id, items in dynamic_tracks.items():
        normalized = _normalize_track(items)
        stats = _track_speed_stats(normalized, config)
        if stats is None:
            continue
        track_stats[int(track_id)] = stats
        points = np.asarray([item["world"] for item in normalized],
                            dtype=np.float64)
        if points.size:
            all_points.append(points)
        if not _is_high_speed(stats, config):
            continue
        if _track_hits_static_slot(
                normalized, static_slots,
                config.static_slot_hard_exclusion_radius):
            rejected_static_tracks.append({
                "track_id": int(track_id),
                "reason": "static_slot_hard_overlap",
                "static_overlap_fraction": round(
                    _static_overlap_fraction(
                        normalized, static_slots,
                        config.static_slot_exclusion_radius), 4),
                **stats,
            })
            continue
        overlap = _static_overlap_fraction(
            normalized, static_slots, config.static_slot_exclusion_radius)
        if overlap >= float(config.static_track_overlap_fraction):
            rejected_static_tracks.append({
                "track_id": int(track_id),
                "reason": "static_slot_overlap_fraction",
                "static_overlap_fraction": round(float(overlap), 4),
                **stats,
            })
            continue
        kept_tracks[int(track_id)] = normalized

    extended_tracks, extension_details = _extend_high_speed_tracks(
        kept_tracks, config)
    for items in extended_tracks.values():
        points = np.asarray([item["world"] for item in items],
                            dtype=np.float64)
        if points.size:
            all_points.append(points)

    bounds = _compute_bounds(all_points, config.bounds_margin)
    shape = _grid_shape(bounds, config.resolution)
    seed = _rasterize_tracks(
        extended_tracks, bounds, config.resolution, shape, config)
    if seed.any():
        seed = _morphology(seed, config.close_radius,
                           config.resolution, close=True)
        core_mask = seed.copy()
        mask = _morphology(seed, config.buffer_radius,
                           config.resolution, close=False)
    else:
        core_mask = np.zeros(shape, dtype=np.uint8)
        mask = np.zeros(shape, dtype=np.uint8)

    if config.mask_static_slot_footprints and static_slots:
        static_mask = _static_slot_mask(
            static_slots, bounds, config.resolution, shape,
            config.static_slot_mask_margin)
    else:
        static_mask = np.zeros(shape, dtype=np.uint8)
    if static_mask.any():
        mask[static_mask.astype(bool)] = 0
        core_mask[static_mask.astype(bool)] = 0

    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    kept_mask = np.zeros(shape, dtype=np.uint8)
    components = []
    for label in range(1, int(count)):
        area_m2 = float(stats[label, cv2.CC_STAT_AREA]) * (
            float(config.resolution) ** 2)
        if area_m2 < float(config.min_region_area_m2):
            continue
        kept_mask[labels == label] = 1
        track_count = 0
        for track_id, items in extended_tracks.items():
            for item in _normalize_track(items):
                columns, rows = _to_pixel(
                    item["world"].reshape(1, 2),
                    bounds, config.resolution, shape)
                if int(labels[int(rows[0]), int(columns[0])]) == label:
                    track_count += 1
                    break
        components.append({
            "label": int(label),
            "area_m2": round(area_m2, 3),
            "high_speed_track_count": track_count,
        })

    polygons = _mask_to_polygons(
        kept_mask, bounds, config.resolution,
        config.min_region_area_m2, config.polygon_approx_eps_m)
    dynamic_polygons = _annotate_dynamic_polygons(
        polygons, extended_tracks, config)
    core_polygons = _annotate_dynamic_polygons(
        _mask_to_polygons(
            core_mask, bounds, config.resolution,
            config.min_core_area_m2, config.polygon_approx_eps_m),
        extended_tracks, config)

    diagnostics = {
        "dynamic_tracks_total": len(dynamic_tracks),
        "tracks_with_stats": len(track_stats),
        "high_speed_tracks": len(kept_tracks),
        "high_speed_track_ids": sorted(kept_tracks),
        "rejected_static_overlap_tracks": rejected_static_tracks,
        "dynamic_polygons": len(dynamic_polygons),
        "core_polygons": len(core_polygons),
        "dynamic_area_m2": round(
            float(np.count_nonzero(kept_mask))
            * float(config.resolution) ** 2, 3),
        "core_area_m2": round(
            float(np.count_nonzero(core_mask))
            * float(config.resolution) ** 2, 3),
        "static_slot_masked_area_m2": round(
            float(np.count_nonzero(static_mask))
            * float(config.resolution) ** 2, 3),
        "dynamic_components": components,
        "high_speed_threshold": float(config.high_speed_threshold),
        "min_track_length": float(config.min_track_length),
        "min_high_speed_observations": int(
            config.min_high_speed_observations),
        "buffer_radius": float(config.buffer_radius),
        "static_slot_exclusion_radius": float(
            config.static_slot_exclusion_radius),
        "static_slot_hard_exclusion_radius": float(
            config.static_slot_hard_exclusion_radius),
        "static_track_overlap_fraction": float(
            config.static_track_overlap_fraction),
        "mask_static_slot_footprints": bool(
            config.mask_static_slot_footprints),
        "extension_length_m": float(config.extension_length_m),
        "extended_tracks": len(extension_details),
        "extension_details": extension_details,
        "track_stats": track_stats,
    }
    return DynamicRegionResult(
        bounds=bounds,
        resolution=config.resolution,
        dynamic_polygons=dynamic_polygons,
        core_polygons=core_polygons,
        diagnostics=diagnostics,
        config=config.to_dict(),
    )


def build_dynamic_regions_from_frames(
        tracked_frames: Sequence[Mapping[str, Any]],
        tracking_diagnostics: Mapping[str, Any],
        coords: Any,
        config: DynamicRegionConfig = DynamicRegionConfig(),
) -> DynamicRegionResult:
    """Convenience wrapper used by step2's first pass.

    ``tracked_frames`` are the first-pass frames (lidar_top boxes plus
    ``track_id``); ``tracking_diagnostics`` must contain ``slot_details``.
    """

    slot_details = list(tracking_diagnostics.get("slot_details", []))
    static_ids = {
        int(item.get("track_id", item.get("slot_id", 0)))
        for item in slot_details
    }
    dynamic_tracks: Dict[int, List[Dict[str, Any]]] = {}
    for frame in tracked_frames:
        timestamp = int(frame["frame_id"])
        world_from_lidar = coords.world_from_lidar(timestamp)
        if world_from_lidar is None:
            continue
        for detection in frame.get("detections", []):
            track_id = detection.get("track_id")
            if track_id is None or int(track_id) in static_ids:
                continue
            box = detection.get("box_lidar")
            if not isinstance(box, list) or len(box) < 7:
                continue
            dynamic_tracks.setdefault(int(track_id), []).append({
                "timestamp": float(timestamp) / 1e9,
                "world": center_world(box, world_from_lidar)[:2],
                "yaw": yaw_world(float(box[6]), world_from_lidar),
                "size": np.asarray(box[3:6], dtype=np.float64),
            })
    for items in dynamic_tracks.values():
        items.sort(key=lambda item: item["timestamp"])
    reference_points = np.asarray([
        [float(item["center"][0]), float(item["center"][1])]
        for item in slot_details if item.get("center") is not None
    ], dtype=np.float64).reshape(-1, 2)
    return build_dynamic_regions(
        dynamic_tracks,
        static_slots=slot_details,
        reference_points=reference_points,
        config=config)


def save_dynamic_regions_json(result: DynamicRegionResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def render_dynamic_regions_png(
        result: DynamicRegionResult,
        path: Path,
        *,
        dynamic_tracks: Optional[Mapping[int, DynamicTrack]] = None,
        static_points: Any = None,
        title: Optional[str] = None,
        max_track_points: int = 20000,
) -> None:
    """Render dynamic polygons, high-speed tracks and static slot centers."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    static = _as_points(static_points)
    kept_ids = set(int(track_id) for track_id in
                   result.diagnostics.get("high_speed_track_ids", []))

    xmin, ymin, xmax, ymax = result.bounds
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    figure, axis = plt.subplots(
        figsize=(max(8.0, min(20.0, width / 8.0)),
                 max(6.0, min(20.0, height / 8.0))))

    if dynamic_tracks:
        total = 0
        for track_id, items in dynamic_tracks.items():
            points = np.asarray(
                [item["world"] for item in _normalize_track(items)],
                dtype=np.float64)
            if points.size == 0:
                continue
            if total > max_track_points:
                break
            if int(track_id) in kept_ids:
                axis.plot(points[:, 0], points[:, 1], color="#1565c0",
                          linewidth=1.1, alpha=0.85,
                          label="high-speed track" if total == 0 else None)
            else:
                axis.plot(points[:, 0], points[:, 1], color="0.78",
                          linewidth=0.7, alpha=0.7,
                          label="other dynamic track" if total == 0 else None)
            total += len(points)

    if static.size:
        axis.scatter(static[:, 0], static[:, 1], s=14, c="#d62728",
                     marker="x", linewidths=0.8, label="static slot")

    first_patch = True
    for item in result.dynamic_polygons:
        points = np.asarray(item["polygon"], dtype=np.float64)
        if len(points) < 3:
            continue
        axis.add_patch(PolygonPatch(
            points, closed=True, facecolor="#ff9800", edgecolor="#e65100",
            alpha=0.28, linewidth=1.2,
            label="dynamic region (high-speed)" if first_patch else None))
        first_patch = False

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_title(title or "high-speed dynamic regions")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
