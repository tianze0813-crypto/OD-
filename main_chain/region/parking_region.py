#!/usr/bin/env python3
"""Automatic parking / road region construction for step2 diagnostics.

The builder consumes:
  * static slot records (world-frame centers / sizes) discovered by step2;
  * static detection points (world-frame) that were assigned to those slots;
  * dynamic detection points (world-frame) with their track ids.

It produces tight parking polygons and road polygons in the world frame.  The
main design goals are deliberately conservative:

  * a parking region must contain enough slot evidence and enough area;
  * a road corridor carved out by dynamic traffic is removed from parking
    regions, so a car driving through the scene is never forced to become a
    parking track;
  * polygons are built on a metric grid, which makes thresholds and diagnostics
    easy to reason about.

This module is intentionally independent from the production step2 pipeline.
It is used first as an offline validation tool on the 090400 clips.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Bounds = Tuple[float, float, float, float]
Polygon = List[Tuple[float, float]]


@dataclass(frozen=True)
class ParkingRegionConfig:
    """Thresholds for grid-based parking / road region construction."""

    # Grid
    resolution: float = 1.0
    bounds_margin: float = 10.0

    # Parking clusters: slots are grouped by the gap between their oriented
    # vehicle rectangles plus heading compatibility.  Each kept cluster becomes
    # an approximate oriented rectangle; multiple rectangles are preferred
    # over one large connected polygon.
    slot_footprint_extra: float = 0.3
    cluster_connect_gap: float = 1.0
    cluster_max_yaw_delta: float = 0.8
    rectangle_margin: float = 0.5
    # Recursively split oversized clusters so one rectangle never covers too
    # many slots or too large an area.
    rectangle_max_slots: int = 30
    rectangle_max_extent: float = 30.0
    rectangle_min_split_gap: float = 4.5

    # A region is kept only when it has enough slot evidence and area.
    # 4 is the agreed minimum: smaller clusters are usually detector noise.
    min_slots_per_region: int = 4
    min_region_area_m2: float = 50.0
    min_polygon_area_m2: float = 20.0

    # Road seed: rasterize each robust dynamic track as a polyline.  Only
    # tracks with real through-motion evidence are allowed to carve a road;
    # short parking-lot shuffles are ignored.  The road seed is also clipped
    # against the parking seed so a maneuver inside a dense parking region
    # cannot delete the region.
    road_max_segment_gap_sec: float = 2.0
    road_min_track_net_displacement: float = 5.0
    road_min_track_mean_speed: float = 1.0
    road_dilate_radius: float = 2.5
    road_close_radius: float = 0.5
    min_road_area_m2: float = 20.0

    # Polygon simplification.
    polygon_approx_eps_m: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "resolution": self.resolution,
            "bounds_margin": self.bounds_margin,
            "slot_footprint_extra": self.slot_footprint_extra,
            "cluster_connect_gap": self.cluster_connect_gap,
            "cluster_max_yaw_delta": self.cluster_max_yaw_delta,
            "rectangle_margin": self.rectangle_margin,
            "rectangle_max_slots": self.rectangle_max_slots,
            "rectangle_max_extent": self.rectangle_max_extent,
            "rectangle_min_split_gap": self.rectangle_min_split_gap,
            "min_slots_per_region": self.min_slots_per_region,
            "min_region_area_m2": self.min_region_area_m2,
            "min_polygon_area_m2": self.min_polygon_area_m2,
            "road_max_segment_gap_sec": self.road_max_segment_gap_sec,
            "road_min_track_net_displacement": (
                self.road_min_track_net_displacement),
            "road_min_track_mean_speed": self.road_min_track_mean_speed,
            "road_dilate_radius": self.road_dilate_radius,
            "road_close_radius": self.road_close_radius,
            "min_road_area_m2": self.min_road_area_m2,
            "polygon_approx_eps_m": self.polygon_approx_eps_m,
        }


@dataclass(frozen=True)
class SlotRecord:
    """Minimal static-slot description consumed by the region builder."""

    track_id: int
    center: np.ndarray
    size: np.ndarray
    yaw: float = 0.0
    row_id: Optional[int] = None
    evidence_hits: int = 0
    matched_hits: int = 0

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "SlotRecord":
        center = np.asarray(value.get("center", (0.0, 0.0)),
                            dtype=np.float64)[:2]
        size = np.asarray(value.get("size", (4.5, 2.0, 1.6)),
                          dtype=np.float64)[:3]
        if size.shape != (3,):
            size = np.array([4.5, 2.0, 1.6], dtype=np.float64)
        return cls(
            track_id=int(value.get("track_id", value.get("slot_id", 0))),
            center=center,
            size=size,
            yaw=float(value.get("yaw", 0.0)),
            row_id=(None if value.get("row_id") is None
                    else int(value["row_id"])),
            evidence_hits=int(value.get("evidence_hits", 0)),
            matched_hits=int(value.get("matched_hits", 0)),
        )

    @property
    def footprint_radius(self) -> float:
        return 0.5 * math.hypot(float(self.size[0]), float(self.size[1]))


@dataclass
class ParkingRegionResult:
    """Polygons and diagnostics produced by :func:`build_parking_regions`."""

    bounds: Bounds
    resolution: float
    parking_polygons: List[Dict[str, Any]] = field(default_factory=list)
    road_polygons: List[Dict[str, Any]] = field(default_factory=list)
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
            "parking_polygons": self.parking_polygons,
            "road_polygons": self.road_polygons,
            "legend": {
                "parking_polygons": "green filled polygon = parking region",
                "road_polygons": "orange filled polygon = road corridor",
                "static_points": "light green dots = static detection points",
                "dynamic_points": "gray dots = dynamic detection points",
                "uncolored": "neither parking nor road with enough evidence",
            },
            "diagnostics": self.diagnostics,
            "config": self.config,
        }


def _as_points(values: Any) -> np.ndarray:
    if values is None:
        return np.zeros((0, 2), dtype=np.float64)
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    array = array.reshape(-1, array.shape[-1]) if array.ndim > 1 else array.reshape(-1, 1)
    if array.shape[1] < 2:
        return np.zeros((0, 2), dtype=np.float64)
    return array[:, :2]


def _compute_bounds(points: Sequence[np.ndarray], margin: float) -> Bounds:
    arrays = [array for array in points if array.size]
    if not arrays:
        return (-margin, -margin, margin, margin)
    merged = np.vstack(arrays)
    xmin, ymin = merged.min(axis=0)
    xmax, ymax = merged.max(axis=0)
    return (
        float(xmin - margin),
        float(ymin - margin),
        float(xmax + margin),
        float(ymax + margin),
    )


def _grid_shape(bounds: Bounds, resolution: float) -> Tuple[int, int]:
    width = int(math.ceil((bounds[2] - bounds[0]) / resolution)) + 1
    height = int(math.ceil((bounds[3] - bounds[1]) / resolution)) + 1
    return max(height, 1), max(width, 1)


def _to_pixel(points: np.ndarray, bounds: Bounds,
              resolution: float, shape: Tuple[int, int]
              ) -> Tuple[np.ndarray, np.ndarray]:
    if points.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    xmin, ymin = bounds[0], bounds[1]
    height, width = shape
    columns = np.floor((points[:, 0] - xmin) / resolution).astype(np.int64)
    rows = np.floor((points[:, 1] - ymin) / resolution).astype(np.int64)
    columns = np.clip(columns, 0, width - 1)
    rows = np.clip(rows, 0, height - 1)
    return columns, rows


def _ellipse_kernel(radius_m: float, resolution: float) -> np.ndarray:
    radius_px = max(1, int(round(float(radius_m) / float(resolution))))
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius_px + 1, 2 * radius_px + 1))


def _morphology(mask: np.ndarray, radius_m: float, resolution: float,
                *, close: bool = False) -> np.ndarray:
    if radius_m <= 0:
        return mask
    kernel = _ellipse_kernel(radius_m, resolution)
    operation = cv2.MORPH_CLOSE if close else cv2.MORPH_DILATE
    return cv2.morphologyEx(mask, operation, kernel)


def _slot_seed_mask(
        slots: Sequence[SlotRecord],
        static_points: np.ndarray,
        bounds: Bounds,
        resolution: float,
        shape: Tuple[int, int],
        config: ParkingRegionConfig,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if slots:
        for slot in slots:
            center = np.asarray(slot.center, dtype=np.float64)[:2]
            size = np.asarray(slot.size, dtype=np.float64)[:3]
            if size.shape != (3,) or not np.all(np.isfinite(size)) \
                    or size[0] <= 0.0 or size[1] <= 0.0:
                radius = float(config.slot_footprint_extra) + 2.5
                columns, rows = _to_pixel(
                    center.reshape(1, 2), bounds, resolution, shape)
                cv2.circle(mask, (int(columns[0]), int(rows[0])),
                           int(round(radius / resolution)), 1, -1)
                continue
            heading = np.array([math.cos(float(slot.yaw)),
                                math.sin(float(slot.yaw))],
                               dtype=np.float64)
            normal = np.array([-heading[1], heading[0]], dtype=np.float64)
            half_length = 0.5 * float(size[0]) + float(
                config.slot_footprint_extra)
            half_width = 0.5 * float(size[1]) + float(
                config.slot_footprint_extra)
            corners = np.asarray([
                center - half_length * heading - half_width * normal,
                center + half_length * heading - half_width * normal,
                center + half_length * heading + half_width * normal,
                center - half_length * heading + half_width * normal,
            ], dtype=np.float64)
            columns, rows = _to_pixel(corners, bounds, resolution, shape)
            polygon = np.column_stack([columns, rows]).astype(np.int32)
            cv2.fillPoly(mask, [polygon], 1)
    elif static_points.size:
        columns, rows = _to_pixel(static_points, bounds, resolution, shape)
        mask[rows, columns] = 1
    if mask.any():
        mask = _morphology(
            mask, config.parking_connect_radius, resolution, close=False)
        mask = _morphology(
            mask, config.parking_close_radius, resolution, close=True)
    return mask


def _road_seed_mask(
        dynamic_points: np.ndarray,
        dynamic_track_ids: np.ndarray,
        dynamic_timestamps: Optional[np.ndarray],
        bounds: Bounds,
        resolution: float,
        shape: Tuple[int, int],
        config: ParkingRegionConfig,
) -> np.ndarray:
    if dynamic_points.size == 0 or dynamic_track_ids.size == 0:
        return np.zeros(shape, dtype=np.uint8)
    columns, rows = _to_pixel(dynamic_points, bounds, resolution, shape)
    timestamps = None
    if dynamic_timestamps is not None:
        timestamps = np.asarray(dynamic_timestamps, dtype=np.float64).reshape(-1)
        if timestamps.size != len(columns):
            raise ValueError(
                "dynamic_timestamps must have one element per dynamic point: "
                f"{timestamps.size} != {len(columns)}")
    grouped: Dict[int, List[Tuple[int, int, float]]] = {}
    for index, (column, row, track_id) in enumerate(zip(
            columns.tolist(), rows.tolist(), dynamic_track_ids.tolist())):
        timestamp = 0.0 if timestamps is None else float(timestamps[index])
        grouped.setdefault(int(track_id), []).append(
            (int(column), int(row), timestamp))
    seed = np.zeros(shape, dtype=np.uint8)
    for points in grouped.values():
        if timestamps is not None:
            points.sort(key=lambda item: item[2])
        if len(points) >= 2 and timestamps is not None:
            pixel_xy = np.asarray([(item[0], item[1]) for item in points],
                                  dtype=np.float64)
            net = float(np.linalg.norm(pixel_xy[-1] - pixel_xy[0])
                        * float(resolution))
            path = float(np.sum(np.linalg.norm(
                np.diff(pixel_xy, axis=0), axis=1)) * float(resolution))
            duration = max(float(points[-1][2] - points[0][2]), 1e-3)
            mean_speed = path / duration
            if (net < float(config.road_min_track_net_displacement)
                    and mean_speed < float(config.road_min_track_mean_speed)):
                continue
        for index, (column, row, timestamp) in enumerate(points):
            seed[row, column] = 1
            if index == 0:
                continue
            previous_column, previous_row, previous_timestamp = points[index - 1]
            if (timestamps is not None
                    and float(timestamp - previous_timestamp)
                    > float(config.road_max_segment_gap_sec)):
                continue
            cv2.line(seed, (previous_column, previous_row),
                     (column, row), 1, 1)
    return seed


def _component_records(
        mask: np.ndarray,
        slots: Sequence[SlotRecord],
        bounds: Bounds,
        resolution: float,
        min_slots: int,
        min_area_m2: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return a filtered component mask plus kept / all component records."""
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    kept = np.zeros(mask.shape, dtype=np.uint8)
    kept_records: List[Dict[str, Any]] = []
    all_records: List[Dict[str, Any]] = []
    for label in range(1, int(count)):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        area_m2 = float(area_px) * float(resolution) ** 2
        component = labels == label
        slot_ids = []
        for slot in slots:
            columns, rows = _to_pixel(
                slot.center.reshape(1, 2), bounds, resolution, mask.shape)
            if component[int(rows[0]), int(columns[0])]:
                slot_ids.append(int(slot.track_id))
        record = {
            "label": int(label),
            "area_m2": round(area_m2, 3),
            "slot_count": len(slot_ids),
            "slot_ids": sorted(slot_ids),
            "kept": bool(area_m2 >= float(min_area_m2)
                         and len(slot_ids) >= int(min_slots)),
        }
        all_records.append(record)
        if not record["kept"]:
            continue
        kept[component] = 1
        kept_records.append({
            key: record[key] for key in
            ("label", "area_m2", "slot_count", "slot_ids")
        })
    return kept, kept_records, all_records


def _mask_contains_points(
        mask: np.ndarray,
        points: np.ndarray,
        bounds: Bounds,
        resolution: float,
) -> int:
    if points.size == 0 or not mask.any():
        return 0
    columns, rows = _to_pixel(points, bounds, resolution, mask.shape)
    return int(np.count_nonzero(mask[rows, columns]))


def _angle_distance_mod_pi(a: float, b: float) -> float:
    delta = abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)
    return min(delta, math.pi - delta)


def _circular_median_mod_pi(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    doubled = 2.0 * np.asarray(values, dtype=np.float64)
    center = 0.5 * math.atan2(float(np.median(np.sin(doubled))),
                              float(np.median(np.cos(doubled))))
    candidates = [center, center + math.pi / 2.0]
    return min(candidates, key=lambda candidate: sum(
        _angle_distance_mod_pi(value, candidate) for value in values))


def _slot_support_radius(slot: SlotRecord, direction: np.ndarray,
                         extra: float) -> float:
    """Support radius of an oriented vehicle rectangle in a direction."""
    heading = np.array([math.cos(float(slot.yaw)),
                        math.sin(float(slot.yaw))], dtype=np.float64)
    cross = np.array([-heading[1], heading[0]], dtype=np.float64)
    half_length = 0.5 * float(slot.size[0]) + float(extra)
    half_width = 0.5 * float(slot.size[1]) + float(extra)
    return (abs(float(np.dot(direction, heading))) * half_length
            + abs(float(np.dot(direction, cross))) * half_width)


def _cluster_axes(slots: Sequence[SlotRecord]) -> Tuple[np.ndarray, np.ndarray]:
    heading_yaw = _circular_median_mod_pi([slot.yaw for slot in slots])
    heading = np.array([math.cos(heading_yaw), math.sin(heading_yaw)],
                       dtype=np.float64)
    row_axis = np.array([-heading[1], heading[0]], dtype=np.float64)
    return heading, row_axis


def _split_cluster_recursive(
        slots: Sequence[SlotRecord],
        config: ParkingRegionConfig,
) -> List[List[SlotRecord]]:
    """Split an oversized cluster into compact rectangles.

    The split prefers the largest gap along the row / heading axes.  Adjacent
    parking slots are roughly 2.5 m apart, so only gaps larger than
    ``rectangle_min_split_gap`` are used as natural separators.  A split is
    only accepted when both sides keep at least ``min_slots_per_region`` slots;
    this prevents the recursive split from creating 1-3 slot edge fragments
    that would then be discarded as noise.
    """

    slots = list(slots)
    minimum = max(1, int(config.min_slots_per_region))
    if len(slots) <= 1 or len(slots) < 2 * minimum:
        return [slots]
    heading, row_axis = _cluster_axes(slots)
    centers = np.asarray([slot.center for slot in slots], dtype=np.float64)
    along_row = centers @ row_axis
    along_heading = centers @ heading
    row_extent = float(np.ptp(along_row))
    heading_extent = float(np.ptp(along_heading))
    if (len(slots) <= int(config.rectangle_max_slots)
            and max(row_extent, heading_extent)
            <= float(config.rectangle_max_extent)):
        return [slots]

    # Try the largest natural gaps first, but only accept a gap that leaves
    # enough slots on both sides.
    candidates: List[Tuple[float, np.ndarray, float]] = []
    for axis, coordinates in ((row_axis, along_row), (heading, along_heading)):
        order = np.argsort(coordinates)
        sorted_coords = coordinates[order]
        if len(sorted_coords) < 2:
            continue
        gaps = np.diff(sorted_coords)
        for gap_index, gap in enumerate(gaps):
            if float(gap) < float(config.rectangle_min_split_gap):
                continue
            split_value = 0.5 * (float(sorted_coords[gap_index])
                                 + float(sorted_coords[gap_index + 1]))
            left_count = int(np.count_nonzero(coordinates <= split_value))
            right_count = int(len(coordinates) - left_count)
            if left_count >= minimum and right_count >= minimum:
                candidates.append((float(gap), axis, split_value))
    if candidates:
        _gap, best_axis, best_split = max(candidates,
                                          key=lambda item: item[0])
    else:
        # No usable aisle: split the longer dimension near its median, while
        # keeping both sides at or above the minimum cluster size.
        if row_extent >= heading_extent:
            best_axis = row_axis
            coordinates = along_row
        else:
            best_axis = heading
            coordinates = along_heading
        order = np.argsort(coordinates)
        sorted_coords = coordinates[order]
        split_index = int(np.clip(len(sorted_coords) // 2,
                                  minimum, len(sorted_coords) - minimum))
        if split_index <= 0 or split_index >= len(sorted_coords):
            return [slots]
        best_split = 0.5 * (float(sorted_coords[split_index - 1])
                            + float(sorted_coords[split_index]))
    left = [slot for slot in slots
            if float(np.dot(slot.center, best_axis)) <= float(best_split)]
    right = [slot for slot in slots
             if float(np.dot(slot.center, best_axis)) > float(best_split)]
    if len(left) < minimum or len(right) < minimum:
        return [slots]
    return (_split_cluster_recursive(left, config)
            + _split_cluster_recursive(right, config))


def _slot_cluster_records(
        slots: Sequence[SlotRecord],
        config: ParkingRegionConfig,
) -> List[Dict[str, Any]]:
    """Cluster slots into nearby, heading-compatible groups.

    Two slots are connected when the gap between their oriented vehicle
    rectangles is small enough.  Each cluster is additionally split along the
    row axis when a large gap is found, so a long parking row becomes one or
    more compact rectangles instead of one oversized connected polygon.
    """

    count = len(slots)
    parent = list(range(count))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = root(left), root(right)
        if a != b:
            parent[b] = a

    for left in range(count):
        a = slots[left]
        for right in range(left):
            b = slots[right]
            delta = b.center - a.center
            distance = float(np.linalg.norm(delta))
            if distance <= 1e-6:
                union(left, right)
                continue
            direction = delta / distance
            gap = (distance
                   - _slot_support_radius(a, direction,
                                          config.slot_footprint_extra)
                   - _slot_support_radius(b, direction,
                                          config.slot_footprint_extra))
            if gap > float(config.cluster_connect_gap):
                continue
            if _angle_distance_mod_pi(a.yaw, b.yaw) > float(
                    config.cluster_max_yaw_delta):
                continue
            norm_a = float(np.linalg.norm(a.size))
            norm_b = float(np.linalg.norm(b.size))
            if (float(np.linalg.norm(a.size - b.size))
                    / max(norm_a, norm_b, 1.0) > 0.6):
                continue
            union(left, right)

    groups: Dict[int, List[int]] = {}
    for index in range(count):
        groups.setdefault(root(index), []).append(index)

    records: List[Dict[str, Any]] = []
    for member_indices in groups.values():
        member_slots = [slots[index] for index in member_indices]
        for subcluster in _split_cluster_recursive(member_slots, config):
            records.append({
                "slots": subcluster,
                "slot_ids": sorted(int(slot.track_id)
                                   for slot in subcluster),
                "slot_count": len(subcluster),
                "kept": len(subcluster) >= int(
                    config.min_slots_per_region),
            })
    records.sort(key=lambda item: (-item["slot_count"],
                                   item["slot_ids"][0] if item["slot_ids"]
                                   else 0))
    return records


def _cluster_rectangle(
        cluster_slots: Sequence[SlotRecord],
        config: ParkingRegionConfig,
) -> Polygon:
    """Return an oriented bounding rectangle around a slot cluster."""

    heading_yaw = _circular_median_mod_pi([slot.yaw for slot in cluster_slots])
    heading = np.array([math.cos(heading_yaw), math.sin(heading_yaw)],
                       dtype=np.float64)
    cross = np.array([-heading[1], heading[0]], dtype=np.float64)
    corners: List[np.ndarray] = []
    for slot in cluster_slots:
        slot_heading = np.array([math.cos(float(slot.yaw)),
                                 math.sin(float(slot.yaw))],
                                dtype=np.float64)
        slot_cross = np.array([-slot_heading[1], slot_heading[0]],
                              dtype=np.float64)
        half_length = 0.5 * float(slot.size[0]) + float(
            config.slot_footprint_extra)
        half_width = 0.5 * float(slot.size[1]) + float(
            config.slot_footprint_extra)
        for along_sign in (-1.0, 1.0):
            for cross_sign in (-1.0, 1.0):
                corners.append(
                    np.asarray(slot.center, dtype=np.float64)
                    + along_sign * half_length * slot_heading
                    + cross_sign * half_width * slot_cross)
    points = np.asarray(corners, dtype=np.float64)
    along = points @ heading
    across = points @ cross
    center = (0.5 * (float(along.min()) + float(along.max())) * heading
              + 0.5 * (float(across.min()) + float(across.max())) * cross)
    half_along = 0.5 * (float(along.max()) - float(along.min())) + float(
        config.rectangle_margin)
    half_across = 0.5 * (float(across.max()) - float(across.min())) + float(
        config.rectangle_margin)
    rectangle = [
        center - half_along * heading - half_across * cross,
        center + half_along * heading - half_across * cross,
        center + half_along * heading + half_across * cross,
        center - half_along * heading + half_across * cross,
    ]
    return [(round(float(point[0]), 4), round(float(point[1]), 4))
            for point in rectangle]


def _rectangles_mask(
        rectangles: Sequence[Polygon],
        bounds: Bounds,
        resolution: float,
        shape: Tuple[int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for rectangle in rectangles:
        points = np.asarray(rectangle, dtype=np.float64)
        if len(points) < 3:
            continue
        columns, rows = _to_pixel(points, bounds, resolution, shape)
        polygon = np.column_stack([columns, rows]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 1)
    return mask



def _mask_to_polygons(
        mask: np.ndarray,
        bounds: Bounds,
        resolution: float,
        min_area_m2: float,
        approx_eps_m: float,
) -> List[Polygon]:
    if not mask.any():
        return []
    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[Polygon] = []
    for contour in contours:
        area_m2 = float(cv2.contourArea(contour)) * float(resolution) ** 2
        if area_m2 < float(min_area_m2):
            continue
        epsilon_px = max(float(approx_eps_m) / float(resolution), 1.0)
        approx = cv2.approxPolyDP(contour, epsilon_px, True)
        if len(approx) < 3:
            continue
        points = []
        for column, row in approx.reshape(-1, 2):
            points.append((
                round(float(bounds[0] + (float(column) + 0.5) * resolution), 4),
                round(float(bounds[1] + (float(row) + 0.5) * resolution), 4),
            ))
        polygons.append(points)
    return polygons


def _polygon_contains(polygon: Polygon, point: np.ndarray) -> bool:
    if len(polygon) < 3:
        return False
    return cv2.pointPolygonTest(
        np.asarray(polygon, dtype=np.float32),
        (float(point[0]), float(point[1])), False) >= 0.0


def _polygon_centroid(polygon: Polygon) -> Tuple[float, float]:
    points = np.asarray(polygon, dtype=np.float64)
    if len(points) < 3:
        return (float(points[:, 0].mean()), float(points[:, 1].mean()))
    # Polygon centroid by the shoelace formula.
    x = points[:, 0]
    y = points[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    signed_area = float(np.sum(cross)) / 2.0
    if abs(signed_area) < 1e-9:
        return (float(x.mean()), float(y.mean()))
    cx = float(np.sum((x + np.roll(x, -1)) * cross)) / (6.0 * signed_area)
    cy = float(np.sum((y + np.roll(y, -1)) * cross)) / (6.0 * signed_area)
    return (round(cx, 4), round(cy, 4))


def _polygon_area(polygon: Polygon) -> float:
    points = np.asarray(polygon, dtype=np.float64)
    if len(points) < 3:
        return 0.0
    x = points[:, 0]
    y = points[:, 1]
    return round(abs(float(np.sum(x * np.roll(y, -1)
                                  - np.roll(x, -1) * y))) / 2.0, 3)


def _annotate_polygons(
        polygons: List[Polygon],
        slots: Sequence[SlotRecord],
        prefix: str,
) -> List[Dict[str, Any]]:
    annotated = []
    for index, polygon in enumerate(polygons):
        slot_ids = sorted(
            int(slot.track_id) for slot in slots
            if _polygon_contains(polygon, slot.center))
        centroid = _polygon_centroid(polygon)
        annotated.append({
            "id": f"{prefix}_{index}",
            "polygon": [[round(float(x), 4), round(float(y), 4)]
                        for x, y in polygon],
            "area_m2": _polygon_area(polygon),
            "centroid": [centroid[0], centroid[1]],
            "slot_count": len(slot_ids),
            "slot_ids": slot_ids,
        })
    return annotated


def build_parking_regions(
        slots: Sequence[SlotRecord],
        static_points: Any,
        dynamic_points: Any,
        dynamic_track_ids: Any,
        dynamic_timestamps: Any = None,
        config: ParkingRegionConfig = ParkingRegionConfig(),
) -> ParkingRegionResult:
    """Build parking and road polygons from world-frame observations.

    ``static_points`` are detection centers assigned to static slots.
    ``dynamic_points`` / ``dynamic_track_ids`` / ``dynamic_timestamps``
    describe moving detections.  The caller is responsible for filtering
    dynamic points to tracks with enough motion evidence; this builder only
    sees the resulting points.  ``dynamic_timestamps`` are in seconds and are
    used to avoid connecting observations across long occlusion gaps.
    """

    slots = list(slots)
    static_points = _as_points(static_points)
    dynamic_points = _as_points(dynamic_points)
    track_ids = np.asarray(dynamic_track_ids, dtype=np.int64).reshape(-1)
    if track_ids.size != len(dynamic_points):
        raise ValueError(
            "dynamic_track_ids must have one element per dynamic point: "
            f"{track_ids.size} != {len(dynamic_points)}")
    timestamps = None
    if dynamic_timestamps is not None:
        timestamps = np.asarray(dynamic_timestamps, dtype=np.float64).reshape(-1)
        if timestamps.size != len(dynamic_points):
            raise ValueError(
                "dynamic_timestamps must have one element per dynamic point: "
                f"{timestamps.size} != {len(dynamic_points)}")

    bounds = _compute_bounds(
        [static_points, dynamic_points,
         np.asarray([slot.center for slot in slots], dtype=np.float64)
         if slots else np.zeros((0, 2), dtype=np.float64)],
        config.bounds_margin)
    shape = _grid_shape(bounds, config.resolution)

    cluster_records = _slot_cluster_records(slots, config)
    parking_polygons: List[Dict[str, Any]] = []
    kept_clusters: List[Dict[str, Any]] = []
    for record in cluster_records:
        rectangle = _cluster_rectangle(record["slots"], config)
        area_m2 = _polygon_area(rectangle)
        record["area_m2"] = area_m2
        record["rectangle"] = rectangle
        if (not record["kept"]
                or area_m2 < float(config.min_region_area_m2)):
            record["kept"] = False
            continue
        kept_clusters.append(record)
        centroid = _polygon_centroid(rectangle)
        parking_polygons.append({
            "id": f"parking_{len(parking_polygons)}",
            "polygon": [[float(x), float(y)] for x, y in rectangle],
            "area_m2": area_m2,
            "centroid": [centroid[0], centroid[1]],
            "slot_count": int(record["slot_count"]),
            "slot_ids": list(record["slot_ids"]),
        })

    parking_mask = _rectangles_mask(
        [item["polygon"] for item in parking_polygons],
        bounds, config.resolution, shape)

    road_seed = _road_seed_mask(
        dynamic_points, track_ids, timestamps, bounds, config.resolution,
        shape, config)
    # A dynamic maneuver inside a parking rectangle must not delete that
    # rectangle.  Only dynamic paths outside the rectangles can become roads.
    if parking_mask.any() and road_seed.any():
        road_seed = road_seed & (~parking_mask.astype(bool)).astype(np.uint8)
    if road_seed.any():
        road_mask = _morphology(
            road_seed, config.road_close_radius, config.resolution,
            close=True)
        road_mask = _morphology(
            road_mask, config.road_dilate_radius, config.resolution,
            close=False)
    else:
        road_mask = np.zeros(shape, dtype=np.uint8)

    road_polygons = _annotate_polygons(
        _mask_to_polygons(
            road_mask, bounds, config.resolution,
            config.min_road_area_m2, config.polygon_approx_eps_m),
        [], "road")

    components = [{
        "label": index,
        "area_m2": float(item["area_m2"]),
        "slot_count": int(item["slot_count"]),
        "slot_ids": list(item["slot_ids"]),
    } for index, item in enumerate(kept_clusters)]
    all_components = [{
        "label": index,
        "area_m2": float(item.get("area_m2", 0.0)),
        "slot_count": int(item["slot_count"]),
        "slot_ids": list(item["slot_ids"]),
        "kept": bool(item["kept"]),
    } for index, item in enumerate(cluster_records)]

    diagnostics = {
        "slots_total": len(slots),
        "static_points": int(len(static_points)),
        "dynamic_points": int(len(dynamic_points)),
        "dynamic_tracks": int(len(np.unique(track_ids)))
        if track_ids.size else 0,
        "grid_shape": {"width": int(shape[1]), "height": int(shape[0])},
        "parking_seed_cells": int(np.count_nonzero(parking_mask)),
        "road_cells": int(np.count_nonzero(road_mask)),
        "parking_candidate_cells": int(np.count_nonzero(parking_mask)),
        "parking_final_cells": int(np.count_nonzero(parking_mask)),
        "parking_components": components,
        "parking_components_all": all_components,
        "parking_polygons": len(parking_polygons),
        "road_polygons": len(road_polygons),
        "parking_slot_count": sum(
            item["slot_count"] for item in parking_polygons),
        "static_points_inside_parking": _mask_contains_points(
            parking_mask, static_points, bounds, config.resolution),
        "dynamic_points_inside_parking": _mask_contains_points(
            parking_mask, dynamic_points, bounds, config.resolution),
        "dynamic_points_inside_road": _mask_contains_points(
            road_mask, dynamic_points, bounds, config.resolution),
        "slot_centers_inside_parking": _mask_contains_points(
            parking_mask,
            np.asarray([slot.center for slot in slots], dtype=np.float64)
            if slots else np.zeros((0, 2), dtype=np.float64),
            bounds, config.resolution),
        "road_area_m2": round(
            float(np.count_nonzero(road_mask)) * config.resolution ** 2, 3),
        "parking_area_m2": round(
            float(np.count_nonzero(parking_mask)) * config.resolution ** 2, 3),
    }
    return ParkingRegionResult(
        bounds=bounds,
        resolution=config.resolution,
        parking_polygons=parking_polygons,
        road_polygons=road_polygons,
        diagnostics=diagnostics,
        config=config.to_dict(),
    )


def save_regions_json(result: ParkingRegionResult, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")


def load_regions_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_regions_png(
        result: ParkingRegionResult,
        path: Path,
        *,
        dynamic_points: Any = None,
        static_points: Any = None,
        title: Optional[str] = None,
        max_dynamic_points: int = 20000,
) -> None:
    """Render a diagnostic PNG overlay (world frame, equal aspect)."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dynamic = _as_points(dynamic_points)
    static = _as_points(static_points)

    xmin, ymin, xmax, ymax = result.bounds
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    fig_width = max(8.0, min(20.0, width / 8.0))
    fig_height = max(6.0, min(20.0, height / 8.0))
    figure, axis = plt.subplots(figsize=(fig_width, fig_height))

    if dynamic.size:
        if len(dynamic) > max_dynamic_points:
            step = int(math.ceil(len(dynamic) / max_dynamic_points))
            dynamic = dynamic[::step]
        axis.scatter(dynamic[:, 0], dynamic[:, 1], s=1.2, c="0.75",
                     linewidths=0, label="dynamic points")
    if static.size:
        axis.scatter(static[:, 0], static[:, 1], s=0.6, c="#b7e1b0",
                     linewidths=0, label="static points")

    first_parking = True
    for item in result.parking_polygons:
        points = np.asarray(item["polygon"], dtype=np.float64)
        if len(points) >= 3:
            axis.add_patch(PolygonPatch(
                points, closed=True, facecolor="#2ca02c", edgecolor="#1b5e20",
                alpha=0.28, linewidth=1.2,
                label="parking region (green)" if first_parking else None))
            first_parking = False
    first_road = True
    for item in result.road_polygons:
        points = np.asarray(item["polygon"], dtype=np.float64)
        if len(points) >= 3:
            axis.add_patch(PolygonPatch(
                points, closed=True, facecolor="#ff9800", edgecolor="#e65100",
                alpha=0.18, linewidth=1.0,
                label="road corridor (orange)" if first_road else None))
            first_road = False

    axis.set_xlim(xmin, xmax)
    axis.set_ylim(ymin, ymax)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linestyle=":", linewidth=0.4, alpha=0.5)
    axis.set_xlabel("world x [m]")
    axis.set_ylabel("world y [m]")
    axis.set_title(title or "parking / road regions")
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
