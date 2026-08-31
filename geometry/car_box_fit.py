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
from collections import Counter, defaultdict
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
    # Roof evidence is searched bottom-up inside the final fitted XY
    # footprint.  Overlapping 10 cm windows advance in 5 cm steps; this keeps a
    # thin lidar roof return visible in at least two adjacent windows while a
    # one-layer outlier cannot become the roof.
    roof_min_points: int = 6
    roof_footprint_inset: float = 0.05
    roof_scan_step: float = 0.05
    roof_window_height: float = 0.10
    roof_search_padding: float = 0.35
    roof_min_contiguous_windows: int = 2
    roof_gap_above: float = 0.05
    roof_min_long_span: float = 0.60
    roof_min_short_span: float = 0.25
    roof_min_long_ratio: float = 0.14
    roof_min_short_ratio: float = 0.12
    roof_grid_long_bins: int = 6
    roof_grid_short_bins: int = 4
    roof_min_occupied_cells: int = 3
    roof_min_connected_cells: int = 3
    # A candidate roof must have an actual return in the central 40% x 40%
    # footprint.  The former q05/q95 envelope test let two diagonal/noisy
    # patches surround an empty center and incorrectly pass as a roof.
    roof_center_region_ratio: float = 0.40
    # The robust center of a roof section must also remain near the box center.
    # Median coordinates keep a few slanted branch returns from moving this
    # check while rejecting a complete patch that only clips the center.
    roof_center_centroid_ratio: float = 0.35
    roof_min_center_points: int = 2
    roof_percentile: float = 95.0
    roof_diagnostic_examples: int = 40
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


def _largest_grid_component(occupied: np.ndarray) -> int:
    """Return the largest 8-connected component in a boolean footprint grid."""
    visited = np.zeros_like(occupied, dtype=bool)
    largest = 0
    rows, columns = occupied.shape
    for row in range(rows):
        for column in range(columns):
            if not occupied[row, column] or visited[row, column]:
                continue
            stack = [(row, column)]
            visited[row, column] = True
            size = 0
            while stack:
                current_row, current_column = stack.pop()
                size += 1
                for row_delta in (-1, 0, 1):
                    for column_delta in (-1, 0, 1):
                        if row_delta == 0 and column_delta == 0:
                            continue
                        next_row = current_row + row_delta
                        next_column = current_column + column_delta
                        if not (0 <= next_row < rows
                                and 0 <= next_column < columns):
                            continue
                        if (occupied[next_row, next_column]
                                and not visited[next_row, next_column]):
                            visited[next_row, next_column] = True
                            stack.append((next_row, next_column))
            largest = max(largest, size)
    return largest


def _roof_shape_check(
        local_xy: np.ndarray, half: np.ndarray, long_axis: int,
        config: CarBoxFitConfig) -> Dict[str, Any]:
    """Check that one horizontal section owns a connected central 2-D patch."""
    short_axis = 1 - long_axis
    long_size = 2.0 * float(half[long_axis])
    short_size = 2.0 * float(half[short_axis])
    q05, q95 = np.percentile(local_xy, [5.0, 95.0], axis=0)
    spans = q95 - q05
    min_long_span = max(config.roof_min_long_span,
                        config.roof_min_long_ratio * long_size)
    min_short_span = max(config.roof_min_short_span,
                         config.roof_min_short_ratio * short_size)

    center_half = np.maximum(
        half * float(config.roof_center_region_ratio), 1e-6)
    center_mask = np.all(np.abs(local_xy) <= center_half, axis=1)
    center_points = int(np.count_nonzero(center_mask))
    robust_center = np.median(local_xy, axis=0)
    centroid_half = np.maximum(
        half * float(config.roof_center_centroid_ratio), 1e-6)
    center_aligned = bool(np.all(np.abs(robust_center) <= centroid_half))
    center_covered = bool(
        center_points >= int(config.roof_min_center_points)
        and center_aligned
    )

    axes = [long_axis, short_axis]
    bins = np.asarray([config.roof_grid_long_bins,
                       config.roof_grid_short_bins], dtype=np.int32)
    normalized = ((local_xy[:, axes] + half[axes])
                  / np.maximum(2.0 * half[axes], 1e-6))
    indices = np.floor(normalized * bins).astype(np.int32)
    indices = np.clip(indices, 0, bins - 1)
    counts = np.zeros(tuple(int(value) for value in bins), dtype=np.int32)
    np.add.at(counts, (indices[:, 0], indices[:, 1]), 1)
    occupied = counts > 0
    occupied_cells = int(np.count_nonzero(occupied))
    largest_component = _largest_grid_component(occupied)
    accepted = bool(
        float(spans[long_axis]) >= min_long_span
        and float(spans[short_axis]) >= min_short_span
        and center_covered
        and occupied_cells >= config.roof_min_occupied_cells
        and largest_component >= config.roof_min_connected_cells
    )
    return {
        "long_span": round(float(spans[long_axis]), 4),
        "short_span": round(float(spans[short_axis]), 4),
        "required_long_span": round(float(min_long_span), 4),
        "required_short_span": round(float(min_short_span), 4),
        "center_covered": center_covered,
        "center_points": center_points,
        "center_aligned": center_aligned,
        "robust_center": [
            round(float(robust_center[0]), 4),
            round(float(robust_center[1]), 4),
        ],
        "center_region": [
            round(float(2.0 * center_half[0]), 4),
            round(float(2.0 * center_half[1]), 4),
        ],
        "occupied_cells": occupied_cells,
        "largest_connected_cells": largest_component,
        "accepted": accepted,
    }


def _roof_evidence(
        points: np.ndarray, box: Sequence[float], ground_z: float | None,
        config: CarBoxFitConfig,
) -> Tuple[float | None, int, Dict[str, Any]]:
    """Find the Car roof from bottom-up, overlapping horizontal sections.

    ``box`` already contains the final Step3 XY fit.  A roof candidate must be
    a connected two-dimensional patch that covers the footprint center, occur
    in at least two adjacent 10 cm windows, and be followed by an unsupported
    interval.  Higher narrow branches therefore do not replace the roof.
    """
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    bottom = (float(ground_z) if ground_z is not None
              else z - dz / 2.0)
    original_top = z + dz / 2.0
    scan_step = max(float(config.roof_scan_step), 0.01)
    window_height = max(float(config.roof_window_height), scan_step)
    search_top = max(original_top, bottom + dz) + config.roof_search_padding
    half = np.maximum(
        np.asarray([dx, dy], dtype=np.float64) / 2.0
        - config.roof_footprint_inset,
        0.05,
    )
    local = box_geometry._local_xy(points[:, :2], (x, y), yaw)
    crop_mask = (
        (np.abs(local[:, 0]) <= half[0])
        & (np.abs(local[:, 1]) <= half[1])
        & (points[:, 2] >= bottom)
        & (points[:, 2] <= search_top)
    )
    crop_z = points[crop_mask, 2]
    crop_local = local[crop_mask]
    detail: Dict[str, Any] = {
        "scan_direction": "bottom_up",
        "scan_start": round(float(bottom), 4),
        "search_top": round(float(search_top), 4),
        "scan_step": round(scan_step, 4),
        "window_height": round(window_height, 4),
        "crop_points": int(len(crop_z)),
        "footprint": [round(float(dx), 4), round(float(dy), 4)],
    }
    if len(crop_z) < config.roof_min_points:
        detail["rejected_reason"] = "too_few_crop_points"
        return None, 0, detail

    window_count = max(
        1, int(math.floor((search_top - bottom - window_height) / scan_step)) + 1)
    starts = bottom + np.arange(window_count, dtype=np.float64) * scan_step
    long_axis = 0 if dx >= dy else 1
    accepted: List[int] = []
    records: Dict[int, Dict[str, Any]] = {}
    for index, start in enumerate(starts):
        stop = float(start + window_height)
        window_mask = (crop_z >= start) & (crop_z < stop)
        support = int(np.count_nonzero(window_mask))
        if support < config.roof_min_points:
            continue
        shape = _roof_shape_check(
            crop_local[window_mask], half, long_axis, config)
        if not shape["accepted"]:
            continue
        accepted.append(index)
        records[index] = {
            "band": [round(float(start), 4), round(stop, 4)],
            "support": support,
            "shape": shape,
        }

    detail["supported_windows"] = accepted
    if not accepted:
        detail["rejected_reason"] = "no_continuous_roof_section"
        return None, 0, detail

    runs: List[List[int]] = []
    for index in accepted:
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    minimum_run = max(1, int(config.roof_min_contiguous_windows))
    qualified = [run for run in runs if len(run) >= minimum_run]
    detail["supported_runs"] = runs
    detail["qualified_runs"] = qualified
    if not qualified:
        detail["rejected_reason"] = "roof_section_not_vertically_continuous"
        return None, 0, detail

    accepted_set = set(accepted)
    gap_windows = max(1, int(math.ceil(
        config.roof_gap_above / scan_step - 1e-9)))
    selected_run: List[int] | None = None
    for run in reversed(qualified):
        following = range(run[-1] + 1, run[-1] + 1 + gap_windows)
        enough_search_space = run[-1] + gap_windows < window_count
        if enough_search_space and not any(index in accepted_set
                                           for index in following):
            selected_run = run
            break
    if selected_run is None:
        detail["rejected_reason"] = "no_upward_discontinuity"
        return None, 0, detail

    selected_index = selected_run[-1]
    selected_start = starts[selected_index]
    selected_stop = float(selected_start + window_height)
    selected_mask = ((crop_z >= selected_start)
                     & (crop_z < selected_stop))
    selected_z = crop_z[selected_mask]
    roof_z = float(np.percentile(selected_z, config.roof_percentile))
    record = records[selected_index]
    detail.update({
        "selected_run": selected_run,
        "selected_window": selected_index,
        "selected_band": record["band"],
        "roof_gap_reached": True,
        "roof_z": round(roof_z, 4),
        "roof_points": int(len(selected_z)),
        "shape": record["shape"],
    })
    return roof_z, int(len(selected_z)), detail


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


def _box_with_xy_result(
        box: Sequence[float], result: Mapping[str, Any] | None,
) -> List[float]:
    """Return a box whose XY fields contain the final shrink-only result."""
    fitted = [float(value) for value in box[:7]]
    if result is None:
        return fitted
    center_local = np.asarray(result["center_local"], dtype=np.float64)
    cosine, sine = math.cos(fitted[6]), math.sin(fitted[6])
    fitted[0] += float(center_local[0] * cosine - center_local[1] * sine)
    fitted[1] += float(center_local[0] * sine + center_local[1] * cosine)
    fitted[3] = float(result["size_local"][0])
    fitted[4] = float(result["size_local"][1])
    return fitted


def _track_height(items: Sequence[MutableMapping[str, Any]],
                  config: CarBoxFitConfig) -> float:
    heights: List[float] = []
    minimum_height, maximum_height = box_geometry._SIZE_BOUNDS["Car"][2]
    for item in items:
        ground_z = item.get("ground_z")
        roof_z = item.get("roof_z")
        roof_points = int(item.get("roof_points", 0))
        if (ground_z is not None and roof_z is not None
                and roof_points >= config.roof_min_points):
            candidate = float(roof_z) - float(ground_z)
            if minimum_height <= candidate <= maximum_height:
                heights.append(candidate)
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
        coords: tracking.CoordinateProvider,
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
    both_boxes = single_boxes = none_boxes = 0
    size_smoothed_boxes = 0
    car_boxes = 0
    roof_evidence_boxes = 0
    roof_rejections: Counter[str] = Counter()
    roof_examples: List[Dict[str, Any]] = []
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
            item["roof_detail"] = {"rejected_reason": "missing_lidar"}
            item["xy_result"] = None
            if points is None:
                continue
            ground_z, ground_points = box_geometry._estimate_ground(points, box, config)
            item["ground_z"] = ground_z
            item["ground_points"] = ground_points
            item["points"] = points
            item["xy_result"] = _fit_xy_shrink_only(
                points, box, ground_z, config)

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

        # Roof ownership is evaluated only after the final XY fit.  This keeps
        # points that were removed by shrink-only fitting from influencing Z.
        track_roof_rejections: Counter[str] = Counter()
        for item in items:
            points = item.get("points")
            if points is None:
                reason = str(item["roof_detail"]["rejected_reason"])
                roof_rejections[reason] += 1
                track_roof_rejections[reason] += 1
                continue
            roof_box = _box_with_xy_result(
                item["det"]["box_lidar"], item["xy_result"])
            roof_z, roof_points, roof_detail = _roof_evidence(
                points, roof_box, item.get("ground_z"), config)
            item["roof_z"] = roof_z
            item["roof_points"] = roof_points
            item["roof_detail"] = roof_detail
            if roof_z is not None:
                roof_evidence_boxes += 1
            else:
                reason = str(roof_detail.get(
                    "rejected_reason", "unknown_roof_rejection"))
                roof_rejections[reason] += 1
                track_roof_rejections[reason] += 1
            if len(roof_examples) < config.roof_diagnostic_examples:
                roof_examples.append({
                    "frame_id": item["frame_id"],
                    "track_id": track_id,
                    "roof_found": roof_z is not None,
                    "detail": roof_detail,
                })

        height = _track_height(items, config)
        bounds = box_geometry._SIZE_BOUNDS.get("Car")
        if bounds is not None:
            height = float(np.clip(height, bounds[2][0], bounds[2][1]))

        for item in items:
            box = item["det"]["box_lidar"]
            result = item["xy_result"]
            fit_z, fit_height, z_mode = _fit_z_boundaries(
                item, height, config)

            if result is not None:
                fitted_xy = _box_with_xy_result(box, result)
                box[0], box[1] = fitted_xy[0], fitted_xy[1]
                box[3], box[4] = fitted_xy[3], fitted_xy[4]
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
                                     "roof_downward", "raw_fallback")},
            "roof_evidence_boxes": sum(
                item.get("roof_z") is not None for item in items),
            "roof_rejections": dict(sorted(track_roof_rejections.items())),
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
            "z": "final_xy_bottom_up_roof_sections_with_track_height_fallback",
            "z_fallback": [
                "both_valid_use_both_boundaries",
                "both_invalid_keep_ground_and_use_track_height",
                "ground_only_keep_ground_and_use_track_height",
                "roof_only_keep_roof_and_use_track_height",
                "neither_keep_original_center_and_use_track_height",
            ],
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
        "roof_evidence_boxes": roof_evidence_boxes,
        "roof_rejections": dict(sorted(roof_rejections.items())),
        "roof_examples": roof_examples,
        "ground_adjusted_boxes": ground_boxes,
        "roof_adjusted_boxes": roof_boxes,
        "final_detections": final_detections,
        "invariant_check": invariant,
        "details": track_details,
    }
