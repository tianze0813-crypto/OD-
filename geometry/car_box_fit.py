"""Car box fitting: shrink-only XY and ground/roof Z fitting.

This stage is intentionally narrower than the earlier four-face rebuild:

* only ``Car`` boxes are changed;
* Z uses a two-boundary fit: ground is the lower boundary and roof is the
  upper boundary.  Both sides are fitted when both are clear; otherwise the
  clear side moves and the missing side uses the track-level height prior;
* XY never expands a step-2 detector box.  Each axis is fitted independently in
  the lidar-top frame:

  - both faces of an axis are visible and their point clouds are clear ->
    shrink both faces toward the point cloud with a small class-specific
    padding;
  - one face is visible and the other is not -> fit the visible face, fix it,
    and leave the opposite face at the original detector boundary;
  - no reliable body points -> keep the original step-2 XY box.

All coordinates stay in ``lidar_top``.  Point clouds and ``box_lidar`` are
already both expressed in that frame, so no world/base_link transform is
used for the actual XY fitting.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from geometry import box_geometry as box_geometry
from tracking import tracker_conservative as tracking


@dataclass(frozen=True)
class CarBoxFitConfig:
    # Ground / z policy.  These values are kept identical to the reviewed two-boundary
    # face-fit stage, so "z-axis algorithm unchanged" is explicit.
    ground_ring_inner_margin: float = 0.18
    ground_ring_outer_margin: float = 1.20
    ground_z_below: float = 0.75
    ground_z_above: float = 0.45
    ground_min_points: int = 18
    ground_clearance_vehicle: float = 0.04
    ground_clearance_small: float = 0.025
    body_z_margin: float = 0.55
    roof_min_points: int = 18
    roof_min_span: float = 0.45
    roof_percentile: float = 99.0
    # Far one-sided roof path.  When ground is unavailable and only a sparse
    # roof cluster is observed, the standard 18-point roof gate is replaced by
    # a compact upper-cluster fit so the box top follows the visible surface.
    roof_only_min_points: int = 8
    roof_only_min_span: float = 0.30
    roof_only_z_bin: float = 0.06
    roof_only_z_band: float = 0.10
    roof_only_cluster_min_abs: int = 4
    roof_only_cluster_min_frac: float = 0.25
    roof_only_cluster_spread: float = 0.25
    roof_only_top_percentile: float = 90.0
    roof_only_clearance: float = 0.04
    # Ground/roof are treated as the two boundaries of the Z axis.  Both
    # boundaries are fitted when both are clear; otherwise only the clear
    # boundary moves and the other side uses the track-level height prior.
    z_roof_clearance: float = 0.04
    z_min_center_change: float = 0.03

    # XY shrink-only policy.
    body_crop_margin: float = 0.22
    body_min_points: int = 10
    xy_low_percentile: float = 2.0
    xy_high_percentile: float = 98.0
    xy_padding_long: float = 0.15
    xy_padding_short: float = 0.10
    # Face visibility is judged by a dense band near the observed low/high
    # percentile, not by the overall point span.  This avoids shrinking an
    # axis into a flat slice when only one real face is visible.
    face_band_ratio: float = 0.12
    face_band_min: float = 0.18
    face_band_max: float = 0.60
    face_min_points: int = 8
    face_min_ratio: float = 0.10
    face_density_ratio: float = 1.60
    # Physical lower guards for the Car class.  The long and short physical
    # axes are mapped back to box-local x/y according to the detector orientation.
    min_extent_long: float = 3.20
    min_extent_short: float = 1.45
    # Static-only second-pass box-size smoothing for distant sparse boxes.
    # It applies only to single-side or none axes, aligns to the invisible
    # side, and uses a track-level robust physical size.
    size_smooth_min_observations: int = 5
    size_smooth_min_change: float = 0.03


def _class_name(items: Sequence[Mapping[str, Any]]) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        counts[str(item["det"].get("class_name", ""))] += 1
    return max(counts, key=counts.get) if counts else "Car"


def _roof_evidence(points: np.ndarray, box: Sequence[float],
                   ground_z: float | None,
                   config: CarBoxFitConfig) -> Tuple[float | None, int]:
    """Robust roof height from the upper body quantile.

    This is copied from the reviewed two-boundary implementation and is intentionally
    left unchanged for the preview.
    """
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    local = box_geometry._local_xy(points[:, :2], (x, y), yaw)
    half = np.asarray([dx, dy], dtype=np.float64) / 2.0 + 0.35
    bottom = float(ground_z) if ground_z is not None else z - dz / 2.0 - 0.20
    top = max(z + dz / 2.0, bottom + dz) + config.body_z_margin
    mask = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (points[:, 2] >= bottom + 0.12)
        & (points[:, 2] <= top)
    )
    z_values = points[mask, 2]
    if len(z_values) < 8:
        return None, 0
    z_floor = float(ground_z) if ground_z is not None else float(
        np.percentile(z_values, 1.0))
    roof_floor = max(float(z_floor + 0.45),
                     float(np.percentile(z_values, 72.0)))
    roof_mask = z_values >= roof_floor
    roof_local = local[mask][roof_mask]
    if (len(roof_local) >= config.roof_min_points
            and float(np.ptp(roof_local[:, 0])) >= config.roof_min_span):
        return float(np.percentile(z_values[roof_mask],
                                   config.roof_percentile)), int(len(roof_local))
    return None, int(len(roof_local))


def _roof_only_boundary(
        points: np.ndarray, box: Sequence[float],
        config: CarBoxFitConfig) -> Tuple[float | None, Dict[str, Any]]:
    """Fit the upper boundary for a one-sided roof observation.

    The Car box crop policy is reused so only the boundary estimator changes.  The
    highest supported z cluster is selected, and the boundary is represented
    by ``roof_only_top_percentile`` inside that cluster.  A single high
    outlier therefore cannot set the box top.
    """
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    local = box_geometry._local_xy(points[:, :2], (x, y), yaw)
    half = np.asarray([dx, dy], dtype=np.float64) / 2.0 + 0.35
    bottom = z - dz / 2.0 - 0.20
    top = max(z + dz / 2.0, bottom + dz) + config.body_z_margin
    mask = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (points[:, 2] >= bottom + 0.12)
        & (points[:, 2] <= top)
    )
    z_values = points[mask, 2]
    detail = {
        "crop_points": int(len(z_values)),
        "box_top": round(float(z + dz / 2.0), 4),
    }
    if len(z_values) < config.roof_only_min_points:
        detail["rejected_reason"] = "too_few_crop_points"
        return None, detail

    z_floor = float(np.percentile(z_values, 1.0))
    roof_floor = max(z_floor + 0.45,
                     float(np.percentile(z_values, 72.0)))
    roof_mask = z_values >= roof_floor
    roof_values = z_values[roof_mask]
    roof_x = local[mask][roof_mask, 0]
    roof_y = local[mask][roof_mask, 1]
    detail["roof_points"] = int(len(roof_values))
    detail["roof_floor"] = round(float(roof_floor), 4)

    if len(roof_values) < config.roof_only_min_points:
        detail["rejected_reason"] = "too_few_roof_points"
        return None, detail
    xy_span = max(float(np.ptp(roof_x)), float(np.ptp(roof_y)))
    detail["roof_xy_span"] = round(float(xy_span), 4)
    if xy_span < config.roof_only_min_span:
        detail["rejected_reason"] = "roof_span_too_small"
        return None, detail

    low = float(np.min(roof_values))
    high = float(np.max(roof_values))
    if high - low < config.roof_only_z_bin:
        roof_z = float(np.percentile(
            roof_values, config.roof_only_top_percentile))
        support = int(len(roof_values))
    else:
        bins = np.arange(low, high + config.roof_only_z_bin,
                         config.roof_only_z_bin)
        counts, edges = np.histogram(roof_values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        band = config.roof_only_z_band
        scores = np.asarray([
            np.sum(counts[
                (centers >= float(center) - band)
                & (centers <= float(center) + band)
            ])
            for center in centers
        ], dtype=np.float64)
        min_support = max(
            config.roof_only_cluster_min_abs,
            int(round(config.roof_only_cluster_min_frac
                      * len(roof_values))))
        supported = np.flatnonzero(scores >= min_support)
        if len(supported) == 0:
            detail["rejected_reason"] = "no_supported_roof_cluster"
            return None, detail
        best_index = int(supported[-1])  # highest supported cluster
        peak_center = float(centers[best_index])
        cluster = roof_values[
            np.abs(roof_values - peak_center) <= band]
        if float(np.ptp(cluster)) > config.roof_only_cluster_spread:
            median = float(np.median(cluster))
            cluster = roof_values[np.abs(roof_values - median) <= 0.12]
            if len(cluster) < config.roof_only_cluster_min_abs:
                detail["rejected_reason"] = "roof_cluster_not_compact"
                return None, detail
        roof_z = float(np.percentile(
            cluster, config.roof_only_top_percentile))
        support = int(len(cluster))

    detail.update({
        "roof_z": round(float(roof_z), 4),
        "roof_support": int(support),
        "roof_z_spread": round(float(np.ptp(roof_values)), 4),
    })
    return roof_z, detail


def _fit_xy_shrink_only(
        points: np.ndarray, box: Sequence[float], ground_z: float | None,
        config: CarBoxFitConfig) -> Dict[str, Any] | None:
    """Fit a Car's XY cross-section while never expanding the step-2 box.

    The result is expressed in the step-2 box-local frame.  Each axis is treated
    as three independent cases:

    * neither face visible -> keep the original extent and center;
    * exactly one face visible -> fit that visible face and keep the opposite
      detector face unchanged;
    * both faces visible and clear -> fit both faces inward.

    ``center_local`` is a delta relative to the original step-2 box center, and
    ``size_local`` is the final extent along the original box-local x/y axes.
    """
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    local = box_geometry._local_xy(points[:, :2], (x, y), yaw)
    half = np.asarray([dx, dy], dtype=np.float64) / 2.0 + config.body_crop_margin
    bottom = float(ground_z) if ground_z is not None else z - dz / 2.0 - 0.20
    top = max(z + dz / 2.0, bottom + dz) + config.body_z_margin
    mask = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (points[:, 2] >= bottom + 0.12)
        & (points[:, 2] <= top)
    )
    body = local[mask]
    if len(body) < config.body_min_points:
        return None

    dims = np.asarray([dx, dy], dtype=np.float64)
    original_bounds = {
        "x": (-dims[0] / 2.0, dims[0] / 2.0),
        "y": (-dims[1] / 2.0, dims[1] / 2.0),
    }
    long_x = dx >= dy
    bounds: Dict[str, Tuple[float, float]] = {}
    modes: Dict[str, str] = {}
    evidence: Dict[str, Dict[str, bool]] = {"x": {}, "y": {}}

    for axis in (0, 1):
        name = "x" if axis == 0 else "y"
        dim = float(dims[axis])
        orig_lo, orig_hi = original_bounds[name]
        values = body[:, axis]
        q_lo = float(np.percentile(values, config.xy_low_percentile))
        q_hi = float(np.percentile(values, config.xy_high_percentile))
        span = q_hi - q_lo
        band = float(np.clip(
            dim * config.face_band_ratio,
            config.face_band_min, config.face_band_max))
        lo_count = int(np.count_nonzero(values <= q_lo + band))
        hi_count = int(np.count_nonzero(values >= q_hi - band))
        lo_density = lo_count / max(band, 1e-6)
        hi_density = hi_count / max(band, 1e-6)
        overall_density = len(values) / max(span, 1e-6)

        def _visible(count: int, density: float) -> bool:
            return (
                count >= config.face_min_points
                and count >= len(values) * config.face_min_ratio
                and density >= config.face_density_ratio * overall_density
            )

        lo_visible = _visible(lo_count, lo_density)
        hi_visible = _visible(hi_count, hi_density)

        is_long = (axis == 0 and long_x) or (axis == 1 and not long_x)
        padding = config.xy_padding_long if is_long else config.xy_padding_short
        min_extent = (config.min_extent_long if is_long
                      else config.min_extent_short)
        # A shrink-only stage must never enlarge a step-2 box, even when the
        # detector box is below the class lower guard.
        min_extent = min(min_extent, dim)

        if lo_visible and hi_visible:
            lo = q_lo - padding
            hi = q_hi + padding
            mode = "both"
            evidence[name] = {"lo": True, "hi": True}
        elif lo_visible:
            lo = q_lo - padding
            hi = orig_hi
            mode = "single_lo"
            evidence[name] = {"lo": True, "hi": False}
        elif hi_visible:
            lo = orig_lo
            hi = q_hi + padding
            mode = "single_hi"
            evidence[name] = {"lo": False, "hi": True}
        else:
            lo, hi = orig_lo, orig_hi
            mode = "none"
            evidence[name] = {"lo": False, "hi": False}

        # Shrink-only hard constraint: the final face can never move outside
        # the original detector boundary.
        lo = max(orig_lo, lo)
        hi = min(orig_hi, hi)

        if hi - lo < min_extent:
            if mode == "both":
                center = (lo + hi) / 2.0
                lo = max(orig_lo, center - min_extent / 2.0)
                hi = min(orig_hi, center + min_extent / 2.0)
                if hi - lo < min_extent:
                    lo, hi = orig_lo, orig_hi
                    mode = "none"
                    evidence[name] = {"lo": False, "hi": False}
            elif mode == "single_lo":
                lo = hi - min_extent
            else:
                hi = lo + min_extent

        bounds[name] = (lo, hi)
        modes[name] = mode

    center_local = np.asarray([
        (bounds["x"][0] + bounds["x"][1]) / 2.0,
        (bounds["y"][0] + bounds["y"][1]) / 2.0,
    ], dtype=np.float64)
    size_local = np.asarray([
        max(bounds["x"][1] - bounds["x"][0], 1e-6),
        max(bounds["y"][1] - bounds["y"][0], 1e-6),
    ], dtype=np.float64)

    coverage_x = float(np.clip(
        (float(np.percentile(body[:, 0], config.xy_high_percentile))
         - float(np.percentile(body[:, 0], config.xy_low_percentile)))
        / max(dims[0], 1e-6), 0.0, 1.0))
    coverage_y = float(np.clip(
        (float(np.percentile(body[:, 1], config.xy_high_percentile))
         - float(np.percentile(body[:, 1], config.xy_low_percentile)))
        / max(dims[1], 1e-6), 0.0, 1.0))

    return {
        "center_local": center_local,
        "size_local": size_local,
        "modes": modes,
        "bounds_xy": bounds,
        "evidence": evidence,
        "point_count": int(len(body)),
        "coverage": {"x": coverage_x, "y": coverage_y},
        "face_visibility": evidence,
    }


def _smooth_static_box_result(
        result: Dict[str, Any] | None, box: Sequence[float],
        ref_long: float, ref_short: float,
        config: CarBoxFitConfig) -> Tuple[Dict[str, Any], bool]:
    """Apply static-only size smoothing for single-side/none XY axes.

    The returned result keeps the same face modes.  The fitted visible side
    stays fixed; only the invisible side is adjusted to reconstruct the
    smoothed size.  ``none`` keeps the current center.  Axes with ``both`` are
    left tight.
    """
    dims = np.asarray([box[3], box[4]], dtype=np.float64)
    long_x = dims[0] >= dims[1]
    if result is None:
        modes = {"x": "none", "y": "none"}
        center_local = np.zeros(2, dtype=np.float64)
        size_local = dims.copy()
    else:
        modes = dict(result.get("modes", {"x": "none", "y": "none"}))
        center_local = np.asarray(result["center_local"], dtype=np.float64)
        size_local = np.asarray(result["size_local"], dtype=np.float64)

    original_bounds = {
        "x": (-dims[0] / 2.0, dims[0] / 2.0),
        "y": (-dims[1] / 2.0, dims[1] / 2.0),
    }
    new_center = np.zeros(2, dtype=np.float64)
    new_size = np.zeros(2, dtype=np.float64)
    changed = False

    for axis in (0, 1):
        name = "x" if axis == 0 else "y"
        is_long = (axis == 0 and long_x) or (axis == 1 and not long_x)
        ref_size = ref_long if is_long else ref_short
        orig_lo, orig_hi = original_bounds[name]
        cur_lo = float(center_local[axis] - size_local[axis] / 2.0)
        cur_hi = float(center_local[axis] + size_local[axis] / 2.0)
        mode = modes.get(name, "none")

        if mode == "both":
            lo, hi = cur_lo, cur_hi
        elif mode == "single_lo":
            fixed_lo = cur_lo
            lo = fixed_lo
            hi = min(orig_hi, fixed_lo + ref_size)
        elif mode == "single_hi":
            fixed_hi = cur_hi
            lo = max(orig_lo, fixed_hi - ref_size)
            hi = fixed_hi
        else:
            center = (cur_lo + cur_hi) / 2.0
            lo = max(orig_lo, center - ref_size / 2.0)
            hi = min(orig_hi, center + ref_size / 2.0)

        lo = max(orig_lo, lo)
        hi = min(orig_hi, hi)
        new_center[axis] = (lo + hi) / 2.0
        new_size[axis] = hi - lo
        if abs(new_size[axis] - size_local[axis]) >= config.size_smooth_min_change:
            changed = True

    return {
        "center_local": new_center,
        "size_local": new_size,
        "modes": modes,
    }, changed


def _track_height(items: Sequence[MutableMapping[str, Any]],
                  config: CarBoxFitConfig) -> float:
    heights: List[float] = []
    for item in items:
        ground_z = item.get("ground_z")
        roof_z = item.get("roof_z")
        roof_points = int(item.get("roof_points", 0))
        if (ground_z is not None and roof_z is not None
                and roof_points >= config.roof_min_points):
            heights.append(float(roof_z) - float(ground_z))
    if heights:
        return float(np.median(heights))
    original = [float(item["det"]["box_lidar"][5]) for item in items]
    return float(np.median(original)) if original else 1.5


def _fit_z_boundaries(item: MutableMapping[str, Any], height: float,
                      config: CarBoxFitConfig) -> Tuple[float, float, str]:
    """Fit the ground and roof boundaries of one Car observation.

    Ground is the lower Z boundary and roof is the upper boundary.  Both sides
    are fitted only when both have clear point-cloud evidence; otherwise the
    clear boundary moves and the missing side is reconstructed from the
    track-level height prior.
    """
    box = item["det"]["box_lidar"]
    original_center = float(box[2])
    ground_z = item.get("ground_z")
    ground_points = int(item.get("ground_points", 0))
    roof_z = item.get("roof_z")
    roof_points = int(item.get("roof_points", 0))

    ground_visible = (ground_z is not None
                      and ground_points >= config.ground_min_points)
    roof_visible = (roof_z is not None
                    and roof_points >= config.roof_min_points)
    bottom = (float(ground_z) + config.ground_clearance_vehicle
              if ground_visible else None)
    top = (float(roof_z) + config.z_roof_clearance
           if roof_visible else None)

    z_bounds = box_geometry._SIZE_BOUNDS["Car"][2]
    if bottom is not None and top is not None:
        raw_height = top - bottom
        if z_bounds[0] <= raw_height <= z_bounds[1]:
            fit_height = raw_height
            fit_z = (bottom + top) / 2.0
            mode = "both"
        else:
            # Both surfaces exist but disagree with the Car height prior.
            # Trust the lower ground boundary and use the track height.
            fit_height = height
            fit_z = bottom + height / 2.0
            mode = "ground_prior"
    elif bottom is not None:
        fit_height = height
        fit_z = bottom + height / 2.0
        mode = "ground"
    elif top is not None:
        fit_height = height
        fit_z = top - height / 2.0
        mode = "roof_downward"
    else:
        fit_height = height
        fit_z = original_center
        mode = "raw_fallback"

    # A tiny boundary movement only creates jitter without improving the box.
    if abs(fit_z - original_center) <= config.z_min_center_change:
        fit_z = original_center
    return float(fit_z), float(fit_height), mode


def apply_car_box_fit(
        frames: Sequence[Dict[str, Any]],
        coords: tracking_tracking.CoordinateProvider,
        clip: Path,
        tracking_diagnostics: Mapping[str, Any],
        static_yaw_diagnostics: Mapping[str, Any],
        config: CarBoxFitConfig = CarBoxFitConfig(),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    output = copy.deepcopy(list(frames))
    tracks = box_geometry._build_tracks(output, coords)
    lidar = box_geometry._LidarCache(Path(clip))
    static_ids = {int(x["track_id"])
                  for x in static_yaw_diagnostics.get("slots", [])}
    cutoffs = {int(x["track_id"]): int(x["departure_start_timestamp"])
               for x in static_yaw_diagnostics.get("slots", [])
               if x.get("departure_start_timestamp") is not None}

    static_boxes = dynamic_boxes = 0
    ground_boxes = roof_boxes = 0
    z_both_boxes = z_ground_boxes = z_roof_boxes = z_fallback_boxes = 0
    roof_only_fit_boxes = 0
    both_boxes = single_boxes = none_boxes = 0
    size_smoothed_boxes = 0
    car_boxes = 0
    track_details: List[Dict[str, Any]] = []

    for track_id, items in sorted(tracks.items()):
        class_name = _class_name(items)
        if class_name != "Car":
            continue

        for item in items:
            box = item["det"]["box_lidar"]
            points = lidar.get(item["frame_id"])
            item["ground_z"] = None
            item["ground_points"] = 0
            item["roof_z"] = None
            item["roof_points"] = 0
            item["xy_result"] = None
            if points is None:
                continue
            ground_z, ground_points = box_geometry._estimate_ground(points, box, config)
            item["ground_z"] = ground_z
            item["ground_points"] = ground_points
            roof_z, roof_points = _roof_evidence(points, box, ground_z, config)
            item["roof_z"] = roof_z
            item["roof_points"] = roof_points
            item["points"] = points
            item["xy_result"] = _fit_xy_shrink_only(
                points, box, ground_z, config)

        height = _track_height(items, config)
        bounds = box_geometry._SIZE_BOUNDS.get("Car")
        if bounds is not None:
            height = float(np.clip(height, bounds[2][0], bounds[2][1]))

        if track_id in static_ids:
            static_items = [
                item for item in items
                if item["timestamp"] < cutoffs.get(track_id, math.inf)
            ]
            if len(static_items) >= config.size_smooth_min_observations:
                long_sizes: List[float] = []
                short_sizes: List[float] = []
                for item in static_items:
                    box = item["det"]["box_lidar"]
                    result = item["xy_result"]
                    if result is not None:
                        size = np.asarray(result["size_local"],
                                          dtype=np.float64)
                    else:
                        size = np.asarray([box[3], box[4]], dtype=np.float64)
                    if float(box[3]) >= float(box[4]):
                        long_sizes.append(float(size[0]))
                        short_sizes.append(float(size[1]))
                    else:
                        long_sizes.append(float(size[1]))
                        short_sizes.append(float(size[0]))
                ref_long = float(np.median(long_sizes))
                ref_short = float(np.median(short_sizes))
                for item in static_items:
                    smoothed, changed = _smooth_static_box_result(
                        item["xy_result"], item["det"]["box_lidar"],
                        ref_long, ref_short, config)
                    item["xy_result"] = smoothed
                    if changed:
                        size_smoothed_boxes += 1

        for item in items:
            box = item["det"]["box_lidar"]
            result = item["xy_result"]
            points = item.get("points")
            fit_z, fit_height, z_mode = _fit_z_boundaries(
                item, height, config)

            if result is not None:
                center_local = np.asarray(result["center_local"],
                                          dtype=np.float64)
                cosine, sine = math.cos(float(box[6])), math.sin(float(box[6]))
                delta_lidar = np.asarray([
                    center_local[0] * cosine - center_local[1] * sine,
                    center_local[0] * sine + center_local[1] * cosine,
                ], dtype=np.float64)
                box[0] = float(box[0]) + float(delta_lidar[0])
                box[1] = float(box[1]) + float(delta_lidar[1])
                box[3] = float(result["size_local"][0])
                box[4] = float(result["size_local"][1])
                modes = result["modes"]
                if "both" in modes.values():
                    both_boxes += 1
                elif any(str(mode).startswith("single")
                         for mode in modes.values()):
                    single_boxes += 1
                else:
                    none_boxes += 1
            else:
                none_boxes += 1

            box[2] = fit_z
            box[5] = fit_height
            item["z_mode"] = z_mode

            # Far static car where the ground ring is empty: the two-boundary fit may have
            # fallen back to the original detector top.  Refit the upper
            # boundary from the single-frame roof cluster so the visible roof
            # surface actually touches the box top.
            if (track_id in static_ids
                    and item.get("ground_z") is None
                    and points is not None):
                roof_only_z, roof_only_detail = _roof_only_boundary(
                    points, box, config)
                if roof_only_z is not None:
                    roof_top = roof_only_z + config.roof_only_clearance
                    refit_z = roof_top - float(box[5]) / 2.0
                    if abs(refit_z - float(box[2])) > config.z_min_center_change:
                        box[2] = float(refit_z)
                        item["z_mode"] = "roof_only_fit"
                        roof_only_fit_boxes += 1
                    item["roof_only_detail"] = roof_only_detail

            if z_mode == "both":
                z_both_boxes += 1
                ground_boxes += 1
                roof_boxes += 1
            elif z_mode == "ground":
                z_ground_boxes += 1
                ground_boxes += 1
            elif z_mode == "ground_prior":
                z_ground_boxes += 1
                ground_boxes += 1
            elif z_mode == "roof_downward":
                z_roof_boxes += 1
                roof_boxes += 1
            else:
                z_fallback_boxes += 1
            car_boxes += 1
            if track_id in static_ids:
                static_boxes += 1
            else:
                dynamic_boxes += 1

        mode_counts = {"x": defaultdict(int), "y": defaultdict(int)}
        for item in items:
            result = item["xy_result"]
            for name in ("x", "y"):
                if result is None:
                    mode_counts[name]["none"] += 1
                else:
                    mode_counts[name][result["modes"][name]] += 1

        track_details.append({
            "track_id": track_id,
            "class_name": "Car",
            "observations": len(items),
            "height": round(height, 4),
            "median_fitted_size": [
                round(float(np.median([
                    item["det"]["box_lidar"][3]
                    for item in items])), 4),
                round(float(np.median([
                    item["det"]["box_lidar"][4]
                    for item in items])), 4),
            ],
            "xy_modes": {name: dict(mode_counts[name])
                         for name in ("x", "y")},
            "z_modes": {mode: sum(item.get("z_mode") == mode
                                  for item in items)
                        for mode in ("both", "ground", "ground_prior",
                                     "roof_downward", "roof_only_fit",
                                     "raw_fallback")},
            "roof_only_fit_boxes": sum(
                item.get("z_mode") == "roof_only_fit"
                for item in items),
        })

    invariant = box_geometry.verify_geometry_only(frames, output)
    final_detections = sum(len(f.get("detections", [])) for f in output)
    return output, {
        "policy": {
            "pipeline_position": "after_identity_class_and_yaw",
            "geometry_version": "car_box_fit",
            "coordinate_frame": "lidar_top local frame",
            "xy": "car_only_shrink_to_point_cloud_static_size_smoothing",
            "xy_single_side": "fit_visible_face_and_leave_opposite_detector_face",
            "z": "ground_roof_two_boundary_fit_with_far_roof_only_cluster",
            "mutated_fields": "Car box_lidar[0:6] only",
            "frozen_fields": [
                "track_id",
                "class_name",
                "box_lidar[6]",
                "box_presence",
                "all_non_Car_box_lidar",
            ],
            "no_interpolation": True,
        },
        "tracks": len(tracks),
        "car_tracks": len(track_details),
        "car_boxes": car_boxes,
        "static_boxes": static_boxes,
        "dynamic_boxes": dynamic_boxes,
        "both_side_boxes": both_boxes,
        "single_side_boxes": single_boxes,
        "unchanged_xy_boxes": none_boxes,
        "size_smoothed_boxes": size_smoothed_boxes,
        "z_both_boxes": z_both_boxes,
        "z_ground_boxes": z_ground_boxes,
        "z_roof_boxes": z_roof_boxes,
        "z_fallback_boxes": z_fallback_boxes,
        "roof_only_fit_boxes": roof_only_fit_boxes,
        "ground_adjusted_boxes": ground_boxes,
        "roof_adjusted_boxes": roof_boxes,
        "final_detections": final_detections,
        "invariant_check": invariant,
        "details": track_details,
    }
