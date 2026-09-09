"""Legacy class filters retained for compatibility.

The active pipeline Step 5 no longer calls this module.  It now performs the
final point-count/lifecycle gate and converts boxes to ``base_link``.  These
helpers remain available for callers that explicitly relied on the previous
Truck/non-motorized class policy.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np

from tracking import tracker_conservative as tracking


NONMOTORIZED_CLASSES = frozenset({"Cyclist", "Nonmotorized_vehicle"})


@dataclass(frozen=True)
class LowConfidenceClassFilterConfig:
    drop_truck: bool = True
    drop_static_nonmotorized: bool = True
    # A non-motorized track is considered moving when any robust motion
    # evidence exceeds one of these gates.
    min_observations: int = 3
    min_net_displacement: float = 1.5
    min_center_spread: float = 1.0
    min_step_displacement: float = 1.0
    max_step_gap_sec: float = 1.2


def _is_moving_nonmotorized(
        items: Sequence[Mapping[str, Any]],
        coords: tracking.CoordinateProvider,
        config: LowConfidenceClassFilterConfig,
) -> tuple[bool, Dict[str, Any]]:
    """Return True when a Cyclist track has credible movement evidence."""
    ordered = sorted(items, key=lambda item: int(item["timestamp"]))
    positions = []
    timestamps = []
    for item in ordered:
        world_from_lidar = coords.world_from_lidar(int(item["timestamp"]))
        if world_from_lidar is None or not tracking.finite_box(item["det"]):
            continue
        box = item["det"]["box_lidar"]
        positions.append(tracking.center_world(box, world_from_lidar)[:2])
        timestamps.append(int(item["timestamp"]))

    detail = {
        "observations": len(ordered),
        "valid_observations": len(positions),
    }
    if len(positions) < config.min_observations:
        return False, detail

    xy = np.asarray(positions, dtype=np.float64)
    center = np.median(xy, axis=0)
    spread = float(np.percentile(
        np.linalg.norm(xy - center, axis=1), 90.0))
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    max_step = 0.0
    for left, right in zip(positions, positions[1:]):
        max_step = max(max_step, float(np.linalg.norm(right - left)))
    moving = bool(
        net >= config.min_net_displacement
        or spread >= config.min_center_spread
        or max_step >= config.min_step_displacement
    )
    detail.update({
        "net_displacement": round(net, 4),
        "center_spread90": round(spread, 4),
        "max_step": round(max_step, 4),
        "moving": moving,
    })
    return moving, detail


def apply_low_confidence_class_filter(
        frames: Sequence[Dict[str, Any]],
        coords: tracking.CoordinateProvider,
        config: LowConfidenceClassFilterConfig = LowConfidenceClassFilterConfig(),
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    before = copy.deepcopy(list(frames))
    output = copy.deepcopy(list(frames))

    truck_boxes = 0
    truck_tracks = set()
    for frame in output:
        for det in frame.get("detections", []):
            if det.get("class_name") == "Truck":
                truck_boxes += 1
                if det.get("track_id") is not None:
                    truck_tracks.add(int(det["track_id"]))

    nonmotorized_tracks: Dict[int, List[Mapping[str, Any]]] = {}
    for frame_index, frame in enumerate(output):
        timestamp = int(frame["frame_id"])
        for det in frame.get("detections", []):
            if det.get("class_name") not in NONMOTORIZED_CLASSES:
                continue
            if det.get("track_id") is None:
                continue
            nonmotorized_tracks.setdefault(int(det["track_id"]), []).append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "det": det,
            })

    static_nonmotorized_ids = set()
    nonmotorized_details = []
    for track_id, items in sorted(nonmotorized_tracks.items()):
        moving, detail = _is_moving_nonmotorized(items, coords, config)
        detail["track_id"] = track_id
        nonmotorized_details.append(detail)
        if config.drop_static_nonmotorized and not moving:
            static_nonmotorized_ids.add(track_id)

    drop_nonmotorized_boxes = 0
    if config.drop_static_nonmotorized:
        for frame in output:
            kept = []
            for det in frame.get("detections", []):
                if (det.get("class_name") in NONMOTORIZED_CLASSES
                        and det.get("track_id") in static_nonmotorized_ids):
                    drop_nonmotorized_boxes += 1
                    continue
                kept.append(det)
            frame["detections"] = kept
            frame["num_detections"] = len(kept)

    if config.drop_truck:
        removed_truck_boxes = 0
        for frame in output:
            kept = [det for det in frame.get("detections", [])
                    if det.get("class_name") != "Truck"]
            removed_truck_boxes += (
                len(frame.get("detections", [])) - len(kept))
            frame["detections"] = kept
            frame["num_detections"] = len(kept)
    else:
        removed_truck_boxes = 0

    _verify_removal_only(before, output)
    before_detections = sum(len(f.get("detections", [])) for f in before)
    after_detections = sum(len(f.get("detections", [])) for f in output)
    return output, {
        "policy": {
            "pipeline_position": "after_truck_size_interpolation_overlap",
            "drop_truck": config.drop_truck,
            "drop_static_nonmotorized": config.drop_static_nonmotorized,
            "nonmotorized_motion_gates": {
                "min_observations": config.min_observations,
                "min_net_displacement": config.min_net_displacement,
                "min_center_spread": config.min_center_spread,
                "min_step_displacement": config.min_step_displacement,
            },
            "mutated_fields": "remove selected detections only",
        },
        "truck_tracks_before": len(truck_tracks),
        "truck_boxes_before": truck_boxes,
        "truck_boxes_removed": removed_truck_boxes,
        "nonmotorized_tracks_checked": len(nonmotorized_details),
        "static_nonmotorized_tracks_removed": len(static_nonmotorized_ids),
        "static_nonmotorized_boxes_removed": drop_nonmotorized_boxes,
        "nonmotorized_details": nonmotorized_details,
        "before_detections": before_detections,
        "after_detections": after_detections,
        "detections_removed": before_detections - after_detections,
    }


def _verify_removal_only(
        before: Sequence[Dict[str, Any]],
        after: Sequence[Dict[str, Any]]) -> None:
    if len(before) != len(after):
        raise AssertionError("step5 filter changed frame count")
    for left_frame, right_frame in zip(before, after):
        if left_frame.get("frame_id") != right_frame.get("frame_id"):
            raise AssertionError("step5 filter changed frame order")
        left_dets = left_frame.get("detections", [])
        right_dets = right_frame.get("detections", [])
        cursor = 0
        for det in right_dets:
            while cursor < len(left_dets) and left_dets[cursor] != det:
                cursor += 1
            if cursor >= len(left_dets):
                raise AssertionError("step5 filter changed a kept detection")
            cursor += 1
