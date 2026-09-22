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
    # A short roof run can be a lower body/windshield layer rather than the
    # roof.  Reject it only when it also disagrees with the robust track
    # height; strong roof evidence remains untouched.
    z_height_abs_tolerance: float = 0.18
    z_height_relative_tolerance: float = 0.08
    z_weak_roof_max_windows: int = 4
    # Ground is repaired only for a track-local bimodal sequence with repeated
    # abrupt jumps.  Single-frame or monotonic ground changes are preserved.
    ground_temporal_jump: float = 0.25
    ground_temporal_max_gap: float = 0.35
    ground_bimodal_gap: float = 0.25
    ground_min_jump_count: int = 2
    ground_min_cluster_samples: int = 3
    # ---- Car 顶/底分档（2026-09-21 最终版：车顶为主锚 + 地面/先验定框底）----
    #   每帧：框内点云剔离群点后 z 的 90 分位 = z_top_frame；buffer = z_top_base - z_top_frame
    #         框顶 = z_top_base；框底 = max(地面, z_top_base - h_prior)，h_prior ∈ {1.45, 1.7}
    #   每 ID：静止车 z_top_base 定死成世界系常数；运动车每帧跟实测车顶
    #         无地面 + 静态区域 + 能凑出直线排 → 用这一排的共底地面（反推真实地面）
    #   高度夹在 [1.45, 1.70]；只写 box_lidar[2]/[5]
    car_height_policy_enabled: bool = True
    car_height_low_m: float = 1.45
    car_height_high_m: float = 1.70
    car_height_inset_m: float = 0.10
    car_height_clearance_m: float = 0.04
    car_height_roof_percentile: float = 90.0
    car_height_roof_min_points: int = 5
    car_height_roof_min_spread_m: float = 0.15
    car_height_ring_inner_m: float = 0.15
    car_height_ring_outer_m: float = 1.30
    car_height_ground_min_points: int = 15
    car_height_ground_percentile: float = 10.0
    car_height_ground_sanity_m: float = 2.50
    # 停车区"同排共底" + 地面高度先验（2026-09-22 定）
    car_height_ground_prior_m: float = 0.30      # 地面相对参考路面的允许偏差（m）
    car_height_ground_prior_mode: str = "ego_local"   # ego_local（相对该帧自车附近路面）| absolute_zero
    car_height_ego_ring_inner_m: float = 3.0
    car_height_ego_ring_outer_m: float = 8.0
    car_height_parking_min_observations: int = 5  # 确认停车：静止观测帧数下限
    car_height_row_ground_enabled: bool = True
    car_height_row_ground_percentile: float = 10.0  # 排地面 = 排内各车环地面的低分位（"取底"）
    car_height_row_max_gap_long_m: float = 9.2      # 沿排间距上限 = 两个车长
    car_height_row_max_gap_lat_m: float = 3.8       # 垂直排间距上限 = 两个车宽
    car_height_row_radius_m: float = 25.0
    car_height_row_line_tolerance_m: float = 0.80
    car_height_row_min_members: int = 3
    # 单帧取顶的统计量："max" = 波段内最高回波（无偏，噪点交给时序门控拉回）；
    #                    "p90" = 框内点的 90 分位（抗噪但远处会落在腰线上，偏低）
    car_height_roof_statistic: str = "max"
    car_height_roof_gate_m: float = 0.15          # 本帧实测与时序参考差超过它 → 用参考（拉回来）
    car_height_roof_window_frames: int = 2       # 动态车：帧顶参考 = 前后各 N 帧的中位（相邻帧约束）
    car_height_roof_outlier_sigma: float = 3.0   # 静止车：基准车顶先按 MAD 剔异常帧
    car_height_buffer_split: bool = False
    car_height_buffer_threshold_m: float = 0.10
    ground_min_cluster_samples: int = 3
    # ---- Car 顶/底分档（2026-09-21 定稿）：地面优先，找不到地面才用"框内最高点=车顶" ----
    #   逐帧：先用"框周围一圈"的点估地面（远近都试：远距离也有地面点）
    #         地面可得 → 底 = 地面 + 间隙（在地上），顶 = 底 + 该轨迹统一高度
    #         地面不可得 → 顶 = 框内最高回波（= 车顶），底 = 顶 - 统一高度
    #   轨迹级：高度统一 = 优先取 r < near_m 的测量（取 percentile，遮挡帧把顶拉低的能兜回来）
    #           再夹到 [low, high]；只写 box_lidar[2]/[5]，XY 与 yaw 不动
    car_height_policy_enabled: bool = True
    car_height_low_m: float = 1.45
    car_height_high_m: float = 1.70
    car_height_near_m: float = 50.0
    car_height_inset_m: float = 0.10
    car_height_clearance_m: float = 0.04
    car_height_ceiling_slack_m: float = 0.50
    car_height_ring_inner_m: float = 0.15
    car_height_ring_outer_m: float = 1.30
    car_height_ground_min_points: int = 15
    car_height_ground_percentile: float = 10.0
    car_height_ground_sanity_m: float = 2.50
    car_height_track_percentile: float = 75.0

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
    """轨迹类别名（先归一到工程类别，兼容 BEVFusion raw 的小写别名）。"""
    counts: Dict[str, int] = defaultdict(int)
    for item in items:
        raw = str(item["det"].get("class_name", ""))
        canonical = tracking.canonical_class_name(raw) or raw
        counts[canonical] += 1
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


def _repair_track_ground(
        items: Sequence[MutableMapping[str, Any]],
        config: CarBoxFitConfig) -> int:
    """Repair only a clearly bimodal, abruptly switching ground sequence.

    ``_estimate_ground`` intentionally remains unchanged for ordinary frames.
    When one track alternates between two surfaces, the higher cluster is
    treated as the competing surface only if the lower cluster is the larger
    one and the sequence contains repeated large jumps.  Values in that
    cluster are linearly interpolated from the nearest lower-cluster samples.
    A long prefix or suffix can use its single lower-cluster anchor; short edge
    runs remain untouched.
    """
    valid = [
        (index, item, float(item["ground_z"]))
        for index, item in enumerate(items)
        if item.get("ground_z") is not None
        and int(item.get("ground_points", 0)) >= config.ground_min_points
    ]
    if (len(valid) < 2 * config.ground_min_cluster_samples):
        return 0

    values = np.asarray([entry[2] for entry in valid], dtype=np.float64)
    ordered = np.sort(values)
    gaps = np.diff(ordered)
    if len(gaps) == 0:
        return 0
    split = int(np.argmax(gaps))
    largest_gap = float(gaps[split])
    low_count = split + 1
    high_count = len(ordered) - low_count
    if (largest_gap < config.ground_bimodal_gap
            or low_count < config.ground_min_cluster_samples
            or high_count < config.ground_min_cluster_samples
            or low_count < high_count):
        return 0

    jumps = 0
    for (_, left, left_z), (_, right, right_z) in zip(valid, valid[1:]):
        gap_sec = (int(right["timestamp"]) - int(left["timestamp"])) / 1e9
        if (gap_sec <= config.ground_temporal_max_gap
                and abs(right_z - left_z) >= config.ground_temporal_jump):
            jumps += 1
    if jumps < config.ground_min_jump_count:
        return 0

    low_limit = float(ordered[split])
    low_entries = [entry for entry in valid if entry[2] <= low_limit]
    high_entries = [entry for entry in valid if entry[2] > low_limit]
    high_indices = {entry[0] for entry in high_entries}

    def _high_run_length(index: int) -> int:
        start = index
        while start - 1 in high_indices:
            start -= 1
        stop = index
        while stop + 1 in high_indices:
            stop += 1
        return stop - start + 1

    repaired = 0
    for index, item, original in high_entries:
        previous = max((entry for entry in low_entries if entry[0] < index),
                       default=None, key=lambda entry: entry[0])
        following = min((entry for entry in low_entries if entry[0] > index),
                        default=None, key=lambda entry: entry[0])
        if previous is None and following is None:
            continue
        if previous is None or following is None:
            # A long high-cluster prefix/suffix can have only one lower
            # anchor.  It is still repairable when the track has already met
            # the repeated-jump and bimodality checks above; short edge runs
            # remain untouched.
            if _high_run_length(index) < config.ground_min_cluster_samples:
                continue
            anchor = following if previous is None else previous
            target = float(anchor[2])
        else:
            _, previous_item, previous_z = previous
            _, following_item, following_z = following
            previous_ts = int(previous_item["timestamp"])
            following_ts = int(following_item["timestamp"])
            current_ts = int(item["timestamp"])
            if following_ts <= previous_ts:
                continue
            alpha = (current_ts - previous_ts) / (following_ts - previous_ts)
            target = previous_z + float(np.clip(alpha, 0.0, 1.0)) * (following_z - previous_z)
        if abs(target - original) < config.ground_temporal_jump / 2.0:
            continue
        item["ground_detail"] = {
            "repaired": True,
            "original_ground_z": round(original, 4),
            "repaired_ground_z": round(float(target), 4),
            "reason": "track_bimodal_ground_temporal_jump",
        }
        item["ground_z"] = float(target)
        repaired += 1
    return repaired


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
        height_tolerance = max(
            float(config.z_height_abs_tolerance),
            float(config.z_height_relative_tolerance) * float(height),
        )
        roof_detail = item.get("roof_detail")
        selected_run = (roof_detail.get("selected_run")
                        if isinstance(roof_detail, Mapping) else None)
        weak_roof = (isinstance(selected_run, (list, tuple))
                     and len(selected_run) <= config.z_weak_roof_max_windows)
        inconsistent_track_height = abs(raw_height - float(height)) > height_tolerance
        if (z_bounds[0] <= raw_height <= z_bounds[1]
                and not (weak_roof and inconsistent_track_height)):
            fit_height = raw_height
            fit_z = (bottom + top) / 2.0
            mode = "both"
        else:
            # Both surfaces exist but a weak roof candidate disagrees with the
            # track height prior (or violates the physical Car bounds). Trust
            # the lower ground boundary and use the track height.
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


def _car_height_ego_ground(points: np.ndarray,
                           config: CarBoxFitConfig) -> float | None:
    """该帧"自车附近路面"的 z（3~8 m 环上的低分位）——地面先验的参考。"""
    radius = np.hypot(points[:, 0], points[:, 1])
    annulus = ((radius >= float(config.car_height_ego_ring_inner_m))
               & (radius <= float(config.car_height_ego_ring_outer_m)))
    values = points[annulus, 2]
    if values.size < 50:
        return None
    return float(np.percentile(values, 10.0))


def _car_height_ground_ok(ground: float | None,
                          ego_ground: float | None,
                          box: Sequence[float],
                          base_from_lidar: np.ndarray | None,
                          config: CarBoxFitConfig) -> bool:
    """地面高度先验：相对参考路面（或 base_link 的 0）偏差不超过阈值才算可信地面。"""
    if ground is None:
        return False
    if str(config.car_height_ground_prior_mode).lower() == "absolute_zero":
        if base_from_lidar is None:
            return True
        base_z = float((base_from_lidar @ np.array(
            [float(box[0]), float(box[1]), float(ground), 1.0]))[2])
        return abs(base_z) <= float(config.car_height_ground_prior_m)
    if ego_ground is None:
        return True
    return abs(float(ground) - float(ego_ground)) <= float(config.car_height_ground_prior_m)



def _car_height_ring_ground(points: np.ndarray, box: Sequence[float],
                            config: CarBoxFitConfig) -> float | None:
    """框 footprint 外扩 [inner, outer] 的一圈里估地面（低分位）。"""
    x, y, z, dx, dy, _dz, yaw = (float(value) for value in box[:7])
    inner = np.asarray([dx / 2.0, dy / 2.0]) + float(config.car_height_ring_inner_m)
    outer = np.asarray([dx / 2.0, dy / 2.0]) + float(config.car_height_ring_outer_m)
    local = box_geometry._local_xy(points[:, :2], (x, y), yaw)
    in_outer = (np.abs(local[:, 0]) <= outer[0]) & (np.abs(local[:, 1]) <= outer[1])
    in_inner = (np.abs(local[:, 0]) <= inner[0]) & (np.abs(local[:, 1]) <= inner[1])
    ring_z = points[in_outer & ~in_inner, 2]
    if ring_z.size < config.car_height_ground_min_points:
        return None
    estimate = float(np.percentile(ring_z, float(config.car_height_ground_percentile)))
    if abs(estimate - (z - float(box[5]) / 2.0)) > float(config.car_height_ground_sanity_m):
        return None
    return estimate


def _car_height_roof_frame(points: np.ndarray, box: Sequence[float],
                           config: CarBoxFitConfig,
                           low_z: float | None = None,
                           high_z: float | None = None) -> float | None:
    """框内点云（只取 low_z~high_z 之间的车体点）剔离群点后 z 的 90 分位 = 该帧车顶。

    必须排除框 footprint 里的地面点（否则 p90 会被地面拉低），也要排除可信上限之上的
    东西（树冠/墙），所以给一个 z 区间。
    """
    x, y, z, dx, dy, dz, yaw = (float(value) for value in box[:7])
    half = np.maximum(np.asarray([dx, dy], dtype=np.float64) / 2.0
                      - float(config.car_height_inset_m), 0.05)
    local = box_geometry._local_xy(points[:, :2], (x, y), yaw)
    mask = (np.abs(local[:, 0]) <= half[0]) & (np.abs(local[:, 1]) <= half[1])
    if low_z is None:
        low_z = z - dz / 2.0 + 0.10            # 兜底：盒底往上一点
    if high_z is None:
        high_z = z + dz / 2.0 + float(config.car_height_ceiling_slack_m)
    mask = mask & (points[:, 2] >= low_z) & (points[:, 2] <= high_z)
    values = points[mask, 2]
    if values.size < int(config.car_height_roof_min_points):
        return None
    if str(config.car_height_roof_statistic).lower() != "p90":
        return float(values.max())          # 波段内最高回波（时序门控负责挡噪点）
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 1e-6:
        keep = np.abs(values - median) <= max(
            3.0 * 1.4826 * mad, float(config.car_height_roof_min_spread_m))
        values = values[keep]
    if values.size < int(config.car_height_roof_min_points):
        return None
    return float(np.percentile(values, float(config.car_height_roof_percentile)))


def _car_height_longest_row(points_xy: np.ndarray, line_tolerance: float):
    """在一组中心里找"最长直线排"（共线子集）的索引列表。"""
    count = len(points_xy)
    if count < 2:
        return []
    best: List[int] = []
    for i in range(count):
        for j in range(i + 1, count):
            direction = points_xy[j] - points_xy[i]
            length = float(np.linalg.norm(direction))
            if length < 1e-6:
                continue
            normal = np.asarray([-direction[1], direction[0]]) / length
            offset = points_xy - points_xy[i]
            distance = np.abs(offset @ normal)
            inliers = [int(k) for k in np.flatnonzero(distance <= line_tolerance)]
            if len(inliers) > len(best):
                best = inliers
    return best


def _apply_car_height_policy(
        tracks: Mapping[int, List[MutableMapping[str, Any]]], lidar: Any,
        config: CarBoxFitConfig, static_ids: Sequence[int] = (),
        row_of_track: Mapping[int, int] | None = None,
        parking_observations: Mapping[int, int] | None = None,
        cutoffs: Mapping[int, int] | None = None,
        base_from_lidar: np.ndarray | None = None) -> Dict[str, Any]:
    """车顶为主锚：框顶钉在 z_top_base，框底 = max(地面, z_top_base - 先验高度)。

    只处理 Car、只写 box_lidar[2]/[5]。静止轨迹的 z_top_base 定死成世界系常数；
    运动轨迹每帧跟自己的实测车顶。无地面 + 静态区域时尝试用"直线排"共底反推地面。
    """
    stats: Dict[str, Any] = {
        "enabled": bool(config.car_height_policy_enabled),
        "policy": {
            "anchor": "roof (z_top_base) primary; bottom = max(ground, roof - prior)",
            "roof": "p90 of the in-box z after outlier rejection",
            "height_range_m": [config.car_height_low_m, config.car_height_high_m],
            "static_tracks": "z_top_base fixed in the world frame",
            "row_ground": "static region + no ground: longest straight row shares one ground",
            "scope": "Car only, writes box_lidar[2]/[5]",
        },
        "tracks": 0, "boxes": 0,
        "static_tracks": 0, "moving_tracks": 0,
        "ground_frames": 0, "row_ground_frames": 0, "no_ground_frames": 0,
        "tracks_without_roof": 0,
    }
    if not config.car_height_policy_enabled:
        return stats
    low, high = float(config.car_height_low_m), float(config.car_height_high_m)
    clearance = float(config.car_height_clearance_m)
    static_set = {int(value) for value in static_ids}
    row_map = {int(k): int(v) for k, v in (row_of_track or {}).items() if v is not None}
    parking_map = {int(k): int(v) for k, v in (parking_observations or {}).items()}
    cutoff_map = {int(k): int(v) for k, v in (cutoffs or {}).items()}
    min_parking = int(config.car_height_parking_min_observations)
    priors = (low, high)

    def _is_confirmed_parking(track_id: int) -> bool:
        if track_id not in static_set:
            return False
        if parking_map and parking_map.get(track_id, 0) < min_parking:
            return False
        return True

    def _is_parked_frame(track_id: int, frame_id: str) -> bool:
        cutoff = cutoff_map.get(track_id)
        return cutoff is None or int(frame_id) < int(cutoff)

    ego_grounds: Dict[str, float | None] = {}

    prepared: Dict[int, Dict[str, Any]] = {}
    for track_id, items in sorted(tracks.items()):
        if _class_name(items) != "Car":
            continue
        stats["tracks"] += 1
        per_frame: List[Dict[str, Any]] = []
        for item in items:
            box = item["det"]["box_lidar"]
            points = lidar.get(item["frame_id"])
            roof = ground = None
            if points is not None:
                if item["frame_id"] not in ego_grounds:
                    ego_grounds[item["frame_id"]] = _car_height_ego_ground(points, config)
                ground = _car_height_ring_ground(points, box, config)
                if not _car_height_ground_ok(ground, ego_grounds[item["frame_id"]], box,
                                             base_from_lidar, config):
                    ground = None
                low_z = (ground + 0.30 if ground is not None
                         else float(box[2]) - float(box[5]) / 2.0 + 0.10)
                high_z = ((ground + float(config.car_height_high_m)
                           + float(config.car_height_ceiling_slack_m))
                          if ground is not None else None)
                roof = _car_height_roof_frame(points, box, config, low_z, high_z)
            per_frame.append({"item": item, "box": box, "roof": roof, "ground": ground,
                              "parked": _is_parked_frame(track_id, item["frame_id"])})
        roofs = [f["roof"] for f in per_frame if f["roof"] is not None]
        if not roofs:
            stats["tracks_without_roof"] += 1
            continue
        is_static = track_id in static_set
        stats["static_tracks" if is_static else "moving_tracks"] += 1
        # 各帧车顶的世界系 z（噪点帧会在这一步被时序约束拉回来）
        roof_world: List[float | None] = []
        for frame in per_frame:
            if frame["roof"] is None:
                roof_world.append(None)
                continue
            box = frame["box"]
            matrix = frame["item"]["world_from_lidar"]
            roof_world.append(float((matrix @ np.array(
                [float(box[0]), float(box[1]), float(frame["roof"]), 1.0]))[2]))
        measured_world = [value for value in roof_world if value is not None]
        # 参考序列：静止车 = 定死的世界系常数（先剔异常帧）；动态车 = 前后各 N 帧滚动中位
        references: List[float | None] = [None] * len(per_frame)
        if is_static:
            base_world = None
            if measured_world:
                values = np.asarray(measured_world, dtype=np.float64)
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
                if mad > 1e-6:
                    keep = np.abs(values - median) <= float(config.car_height_roof_outlier_sigma) * 1.4826 * mad
                    if int(np.count_nonzero(keep)) >= 1:
                        values = values[keep]
                base_world = float(np.median(values))
            references = [base_world] * len(per_frame)
        else:
            window = max(1, int(config.car_height_roof_window_frames))
            fallback = float(np.median(measured_world)) if measured_world else None
            for index in range(len(per_frame)):
                local = [roof_world[other] for other in range(max(0, index - window),
                                                              min(len(per_frame), index + window + 1))
                         if roof_world[other] is not None]
                references[index] = float(np.median(local)) if local else fallback
            base_world = None
        # 轨迹高度：实测(车顶-地面)的中位，连续夹到 [low, high]（先验只做上下限）
        sampled = [f["roof"] - f["ground"] - clearance for f in per_frame
                   if f["roof"] is not None and f["ground"] is not None]
        if sampled:
            track_height = float(np.clip(float(np.median(sampled)), low, high))
        else:
            # 全轨迹没有地面（只在静止直线排里可能反推出地面）→ 先按住框现有高度夹取
            reference = [float(f["box"][5]) for f in per_frame if f["roof"] is not None]
            track_height = float(np.clip(float(np.median(reference)) if reference else low, low, high))
        prepared[track_id] = {"per_frame": per_frame, "is_static": is_static,
                              "base_world": base_world, "references": references,
                              "height": track_height}
        for frame in per_frame:
            if frame["ground"] is not None:
                stats["ground_frames"] += 1
            elif frame["roof"] is None:
                stats["no_ground_frames"] += 1

    # 停车区"同排共底"：确认停车的 Car 轨迹按排（优先用 slot 的 row_id）共用同一个底
    row_shared: Dict[int, float] = {}
    if bool(config.car_height_row_ground_enabled):
        members: Dict[Any, List[int]] = {}
        parked_tracks = [track_id for track_id, value in prepared.items()
                         if _is_confirmed_parking(track_id)
                         and any(f["parked"] for f in value["per_frame"])]
        # 没有 slot 排号的，用"最长共线排"兜底（中心连成线、覆盖最多 box）
        fallback_centres = {}
        for track_id in parked_tracks:
            if row_map.get(track_id) is not None:
                continue
            item = prepared[track_id]["per_frame"][0]["item"]
            if item.get("raw_world") is not None:
                fallback_centres[track_id] = np.asarray(item["raw_world"][:2], dtype=np.float64)
        fallback_row: List[int] = []
        if len(fallback_centres) >= 2:
            order = list(fallback_centres)
            picked = _car_height_longest_row(
                np.asarray([fallback_centres[t] for t in order]),
                float(config.car_height_row_line_tolerance_m))
            fallback_row = [order[k] for k in picked]
        for track_id in parked_tracks:
            key = row_map.get(track_id)
            if key is None:
                key = "collinear_row" if track_id in fallback_row else ("lonely", track_id)
            members.setdefault(key, []).append(track_id)
        for key, group in members.items():
            if len(group) < 2:
                continue
            # 间距上限：沿排 ≤ 两个车长，垂直排 ≤ 两个车宽
            centres = {}
            for track_id in group:
                item = prepared[track_id]["per_frame"][0]["item"]
                if item.get("raw_world") is None:
                    continue
                centres[track_id] = np.asarray(item["raw_world"][:2], dtype=np.float64)
            # 间距上限：沿排 ≤ 两个车长、垂直排 ≤ 两个车宽（排轴用首尾中心连线近似）
            ordered = [t for t in group if t in centres]
            pooled: List[float] = []
            for track_id in ordered:
                near = []
                for other in ordered:
                    if other == track_id:
                        near.append(other)
                        continue
                    axis = centres[ordered[-1]] - centres[ordered[0]]
                    length = float(np.linalg.norm(axis))
                    if length < 1e-6:
                        near.append(other)
                        continue
                    axis = axis / length
                    delta = centres[other] - centres[track_id]
                    along = abs(float(delta @ axis))
                    lateral = abs(float(delta[0] * axis[1] - delta[1] * axis[0]))
                    if (along <= float(config.car_height_row_max_gap_long_m)
                            and lateral <= float(config.car_height_row_max_gap_lat_m)):
                        near.append(other)
                if len(near) < 2:
                    continue
                pooled.extend(f["ground"] for t in near for f in prepared[t]["per_frame"]
                              if f["ground"] is not None and f["parked"])
            if len(pooled) >= 2:
                shared = float(np.percentile(pooled, float(config.car_height_row_ground_percentile)))
                for track_id in group:
                    row_shared[track_id] = shared
                stats["row_shared_bottoms"] = stats.get("row_shared_bottoms", 0) + 1

    # 直线排共底（只在静态区域、且该车在本帧没有地面时用）
    missing = [(tid, value) for tid, value in prepared.items()
               if value["is_static"] and any(f["ground"] is None for f in value["per_frame"])]
    row_ground: Dict[int, float] = {}
    if missing:
        centres = {}
        for tid, value in prepared.items():
            if not value["is_static"]:
                continue
            item = value["per_frame"][0]["item"]
            if item.get("raw_world") is None:
                continue
            centres[tid] = np.asarray(item["raw_world"][:2], dtype=np.float64)
        for tid, value in missing:
            if tid not in centres:
                continue
            near = [other for other in centres
                    if float(np.linalg.norm(centres[other] - centres[tid]))
                    <= float(config.car_height_row_radius_m)]
            row = _car_height_longest_row(np.asarray([centres[o] for o in near]),
                                          float(config.car_height_row_line_tolerance_m))
            members = [near[k] for k in row]
            if len(members) < int(config.car_height_row_min_members) or tid not in members:
                continue
            grounds = [f["ground"] for o in members for f in prepared[o]["per_frame"]
                       if f["ground"] is not None]
            if grounds:
                row_ground[tid] = float(np.median(grounds))
            else:
                # 整排都没有地面 → 用"车顶 − 先验高度(按下限)"反推这一排的地面
                reverse = [f["roof"] - high
                           for o in members for f in prepared[o]["per_frame"]
                           if f["roof"] is not None]
                if reverse:
                    row_ground[tid] = float(np.median(reverse))

    for track_id, value in prepared.items():
        per_frame, prior = value["per_frame"], value["height"]
        tops: List[float] = []
        for index, frame in enumerate(per_frame):
            box, item = frame["box"], frame["item"]
            matrix = item["world_from_lidar"]
            centre = matrix @ np.array([float(box[0]), float(box[1]), float(box[2]), 1.0])
            reference = value["references"][index]
            top_reference = None
            if reference is not None:
                top_reference = float((item["lidar_from_world"] @ np.array(
                    [centre[0], centre[1], float(reference), 1.0]))[2])
            # 两遍一起作用：本帧实测为主；只有偏离时序参考超过阈值（噪点/异常帧）才用参考拉回来
            if frame["roof"] is not None:
                top = float(frame["roof"])
                if (top_reference is not None
                        and abs(top - top_reference) > float(config.car_height_roof_gate_m)):
                    top = top_reference
            elif top_reference is not None:
                top = top_reference
            else:
                tops.append(0.0)
                continue
            tops.append(top)
        if not any(tops):
            continue
        # 高度：底 = max(地面, 顶 - 先验)；再夹到 [low, high]
        heights: List[float] = []
        grounds: List[float | None] = []
        for frame, top in zip(per_frame, tops):
            ground = frame["ground"]
            if track_id in row_shared and _is_confirmed_parking(track_id) and frame["parked"]:
                ground = row_shared[track_id]          # 停车排：全员服从同一个底
                stats["row_shared_frames"] = stats.get("row_shared_frames", 0) + 1
            elif ground is None and track_id in row_ground:
                ground = row_ground[track_id]
                stats["row_ground_frames"] += 1
            grounds.append(ground)
            bottom = top - prior
            if ground is not None:
                bottom = max(float(ground), bottom)
            height = top - bottom
            height = min(max(height, low), high)
            if not config.car_height_buffer_split:
                heights.append(height)
                continue
            buffer = 0.0
            if frame["roof"] is not None:
                buffer = max(0.0, top - float(frame["roof"]))
            if buffer > float(config.car_height_buffer_threshold_m):
                height = min(max(height + buffer / 2.0, low), high)
            heights.append(height)
        # ID 内 3 帧中位平滑
        smoothed = []
        for index in range(len(heights)):
            window = heights[max(0, index - 1): index + 2]
            smoothed.append(float(np.median(window)))
        for frame, top, height in zip(per_frame, tops, smoothed):
            box = frame["box"]
            box[2] = float(top - height / 2.0)
            box[5] = float(height)
            stats["boxes"] += 1
    return stats


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
    ground_temporal_repaired_boxes = 0
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
            item["ground_detail"] = {
                "repaired": False,
                "original_ground_z": (None if ground_z is None
                                       else round(float(ground_z), 4)),
            }
            item["points"] = points
            item["xy_result"] = _fit_xy_shrink_only(
                points, box, ground_z, config)

        ground_temporal_repaired_boxes += _repair_track_ground(items, config)

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
            "ground_temporal_repaired": sum(
                bool(item.get("ground_detail", {}).get("repaired"))
                for item in items),
        })

    # 最后一次拟合的最后一道：Car 顶/底分档（地面优先 + 同 id 高度统一）
    _slots = list(static_yaw_diagnostics.get("slots", []))
    car_height = _apply_car_height_policy(
        tracks, lidar, config, static_ids,
        row_of_track={int(x["track_id"]): x.get("row_id") for x in _slots
                      if x.get("track_id") is not None},
        parking_observations={int(x["track_id"]): int(x.get("parking_observations") or 0)
                              for x in _slots if x.get("track_id") is not None},
        cutoffs=cutoffs)

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
        "ground_temporal_repaired_boxes": ground_temporal_repaired_boxes,
        "final_detections": final_detections,
        "car_height_policy": car_height,
        "invariant_check": invariant,
        "details": track_details,
    }
